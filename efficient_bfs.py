import copy
import heapq
import time

import acclerated_upper_bounds
from MaxHeap import MaxHeap, SimpleMaxHeap, EfficientBFSHeapObj
from OptimalAlg import OptimalAlg
from base_task import BaseTask
from filter_search import RefinedBFSValue


class TimerProxy:
    """计时器代理，用于控制循环的生命周期"""

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    @property
    def is_active(self):
        """代理拦截口：未超时返回 True，超时返回 False"""
        return (time.time() - self.start_time) <= self.timeout_seconds


class EfficientBFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0
        self.local_search = False
        self.branching_strategy = "density_gap"

        self.heap_class = 'tradition'

        self.s_max = None

        # === Dive-and-Bound 混合策略组件 ===
        self.use_dive = False  # 开关
        self.dfs_stack = []  # 下潜用的后进先出栈
        self.is_diving = False  # 当前状态标志
        self.dive_interval = 200  # 触发频率：每处理 200 个 BFS 节点下潜一次
        self.dive_max_nodes = 50  # 下潜深度/广度限制：每次下潜最多探索 50 个节点，防止卡死
        self.current_dive_count = 0  # 当前下潜计数器

        self.probing_trigger_count = 0  # the number of nodes where probing succeed to discard the first element in hs
        self.probing_trigger_depth_list = []  # record the depth of each probing-effective point
        self.probing_trigger_depth_list = []  # record the depth of each probing-effective point
        self.max_depth = 0

        # === 级联 EMA 自适应控制器 ===
        self.m_current = 3
        # 初始认为 0, 1, 2 三个位置都有可能赢 (总和为1)
        self.ema_probs = [0.34, 0.33, 0.33]
        self.ema_alpha = 0.15
        self.ema_threshold = 0.05

        self.timer_proxy = None

        # === 继承机制控制器 ===
        self.inherit_bounds = True  # 默认开启继承

        # === 新增：上界评估器类型 ===
        self.ub_type = 'ub2'  # 默认使用 Slicing
        self.use_cascade = False  # 是否开启级联过滤

    def _get_optimizer(self):
        """根据 ub_type 动态实例化对应的 Lazy Optimizer"""
        if self.ub_type == 'ub0':
            return acclerated_upper_bounds.LazyPlainOptimizer(self.model)
        elif self.ub_type == 'ub2':
            return acclerated_upper_bounds.LazySlicingOptimizer(self.model)
        else:
            # 兼容其他情况，默认回退
            return acclerated_upper_bounds.LazySlicingOptimizer(self.model)

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.probing_trigger_count = 0
        self.probing_trigger_depth_list = []
        self.max_depth = 0

        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def push_heap(self, s, lbd_v, visited=False, first_child=False, heuristic_sequence=None, candidate=None, w=None,
                  s_max_v=0, depth=0, forbidden_sets=None, f_local=float('inf')):

        # 1. 统一构建完整的节点
        max_idx = max(s) if len(s) > 0 else 0
        node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited,
                                   first_child=first_child, heuristic_sequence=heuristic_sequence,
                                   max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)
        node.depth = depth
        node.forbidden_sets = forbidden_sets if forbidden_sets is not None else []

        new_g = self.g(node)


        # --- 核心逻辑：级联过滤 (Cascading Bounds) ---
        if self.use_cascade:
            # 第一段：使用极快的 PlainOptimizer (ub0) 进行初筛
            opt_fast = acclerated_upper_bounds.LazyPlainOptimizer(self.model)
            opt_fast.build(base=s, remaining=candidate)
            h_fast = opt_fast.solve(candidate, node.budget)

            # 初筛剪枝判定 (注意也要考虑 alpha)
            f_fast = new_g + h_fast
            if self.alpha * f_fast <= self.g(self.s_max):
                return None  # 被 ub0 秒杀，节省了计算 ub2 的巨量时间

        # 第二段：初筛没杀掉，或者没开启级联，执行正式评估
        new_h = self.h(node)
        final_v = new_g + new_h

        # ====== 全透视追踪：界限评估 ======
        print(f"  ├── [EVAL] 评估子节点 S: {s}")
        print(f"  │   ├── g(S): {new_g:.2f} | h(S): {new_h:.2f} | 原始 UB: {final_v:.2f}")
        print(f"  │   └── 对比条件: min(UB, f_local:{f_local:.2f}) * alpha:{self.alpha} <= s_max_v:{s_max_v:.2f}")

        if min(final_v, f_local) * self.alpha <= self.g(self.s_max):
            print("  │   └── ❌ [KILLED] 剪枝生效，节点已被抹杀。")
            return None

        print("  │   └── ✅ [SURVIVED] 界限达标，准备入堆！")
        # ===============================

        # 2. 统一进行界限判定和 Alpha 剪枝
        if final_v >= s_max_v:
            if self.use_alpha:
                new_ub = new_g + self.alpha * new_h
                # 根据开关决定是否继承父节点的界限
                if self.inherit_bounds:
                    lbd_v = min(new_ub, lbd_v)
                else:
                    lbd_v = new_ub
                v = RefinedBFSValue(new_ub, lbd_v, self.d(s))
            else:
                new_ub = new_g + new_h
                if self.inherit_bounds:
                    lbd_v = min(new_ub, lbd_v)
                else:
                    lbd_v = new_ub
                v = RefinedBFSValue(new_ub, lbd_v, self.d(s))

            node.v = v
            # ====== 就是这里！丢失的入堆/栈逻辑找回 ======
            if getattr(self, 'is_diving', False) and self.current_dive_count < self.dive_max_nodes:
                self.dfs_stack.append(node)
            else:
                self.max_heap.push(node)
            return node
            # =========================================

        # 被剪枝，直接丢弃
        return None

    def greedy_add(self, node):
        def density(ele, base_set):
            return self.model.marginal_gain(ele, list(base_set)) / self.model.cost_of_singleton(ele)

        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        forbidden_sets = getattr(node, 'forbidden_sets', [])

        remaining_elements = set(candidate)
        cur_cost = 0  # Tracks the cost of elements added ON TOP of the base

        # Initialize the Lazy Optimizer
        opt = self._get_optimizer()
        opt.build(base=base, remaining=remaining_elements)

        # Initial upper bound
        # Assuming node.budget represents the remaining capacity for the candidate set
        f_local = self.g(sol) + opt.solve(remaining_elements, budget)
        heuristic_sequence = []

        # 1. Initialize max-heap for the outer greedy loop
        h = []
        tie_breaker = 0
        for e in remaining_elements:
            heapq.heappush(h, (-density(e, sol), tie_breaker, e))
            tie_breaker += 1

        while self.timer_proxy.is_active and h:
            # 2. Pop the element with the highest upper-bound density
            neg_ds, _, u = heapq.heappop(h)

            # === 约束注入拦截逻辑 ===
            is_forbidden = False
            for f_set in forbidden_sets:
                # 如果禁用的集合 f_set 只差元素 u 就凑齐了，那么 u 不能加
                if u in f_set:
                    # 检查当前解是否已经包含了 f_set 中除了 u 以外的所有元素
                    if f_set - {u} <= sol:
                        is_forbidden = True
                        break
            if is_forbidden:
                continue
                # ========================

            # 3. Lazy Budget Check: Discard instantly if it no longer fits the knapsack
            cost_u = self.model.cost_of_singleton(u)
            if cur_cost + cost_u > budget:
                continue

            # 4. Evaluate actual density against the dynamically updating solution set
            actual_ds = density(u, sol)

            # 5. Clean up stale/violating elements at the top of the heap
            while h:
                top_e = h[0][2]
                if cur_cost + self.model.cost_of_singleton(top_e) > budget:
                    heapq.heappop(h)
                else:
                    break

            # 6. Check the lazy condition
            if not h or actual_ds >= -h[0][0]:
                # u is the true maximum element. Process it.
                sol.add(u)
                heuristic_sequence.append(u)
                cur_cost += cost_u

                # --- LAZY OPTIMIZER UPPER BOUND ---
                # Update base seamlessly without rebuilding the inner heap
                opt.update_base(sol)

                # Ensure the optimizer strictly uses the remaining budget (budget)
                remaining_for_opt = set(node.candidate) - sol
                f_temp = self.g(sol) + opt.solve(remaining_for_opt, budget)
                if f_local is None or f_temp < f_local:
                    f_local = f_temp

            else:
                # Push the element back into the heap with its newly calculated density
                heapq.heappush(h, (-actual_ds, tie_breaker, u))
                tie_breaker += 1

        return list(sol), f_local, heuristic_sequence

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0
        root.depth = 0
        root.forbidden_sets = []

        f_upper = self.f(root)
        s_max, f_local, heuristic_sequence = self.greedy_add(root)

        v = RefinedBFSValue(self.f(root), f_upper, self.d(root.s))
        root.v = v
        root.heuristic_sequence = heuristic_sequence

        self.max_heap.push(root)

        return root, f_upper, heuristic_sequence, s_max

    def branching(self, node, heuristic_sequence, f_local=float('inf')):
        # push first child
        first_ele = heuristic_sequence[0]
        new_candidate = list(set(node.candidate) - {first_ele})
        new_lbd = node.v.lbd_v
        open_list_change = 0

        # push second child
        self.push_heap(s=node.s, lbd_v=new_lbd, first_child=False,
                       candidate=new_candidate, w=node.budget, s_max_v=self.g(self.s_max), depth=node.depth + 1,
                       f_local=f_local)
        open_list_change += 1

        if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
            open_list_change += 1

            new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
            new_heuristic_sequence.pop(0)

            self.push_heap(s=list(set(node.s) | {first_ele}), lbd_v=new_lbd, first_child=True,
                           heuristic_sequence=new_heuristic_sequence,
                           candidate=new_candidate,
                           w=node.budget - self.model.cost_of_singleton(first_ele), s_max_v=self.g(self.s_max))

        return open_list_change

    def branching_with_injection(self, node, heuristic_sequence, tau=0.85, max_k=3):
        """
        基于约束注入的块分支规则
        """
        if not heuristic_sequence:
            return 0
        # 1. 识别高密度簇 (Cluster)
        # 利用你之前的密度跳变思想，找出前几个表现极其接近的“巨头”
        first_d = self.model.marginal_gain(heuristic_sequence[0], node.s) / \
                  self.model.cost_of_singleton(heuristic_sequence[0])

        cluster = [heuristic_sequence[0]]
        for i in range(1, min(len(heuristic_sequence), max_k)):
            current_d = self.model.marginal_gain(heuristic_sequence[i], node.s) / \
                        self.model.cost_of_singleton(heuristic_sequence[i])
            if current_d >= first_d * tau:
                cluster.append(heuristic_sequence[i])
            else:
                break

        open_list_change = 0
        new_lbd = node.v.lbd_v

        # --- 分支 1：左分支 (全包含路径) ---
        # 尝试将整个 Cluster 塞进去
        total_cost = sum(self.model.cost_of_singleton(e) for e in cluster)
        if node.cost + total_cost <= self.model.budget:
            new_s = list(set(node.s) | set(cluster))
            new_candidate = list(set(node.candidate) - set(cluster))
            # 继承父节点的序列（剔除掉已经加入的 cluster）
            new_hs = [e for e in heuristic_sequence if e not in cluster]

            self.push_heap(s=new_s, lbd_v=new_lbd, first_child=True,
                           heuristic_sequence=new_hs,
                           candidate=new_candidate,
                           w=node.budget - total_cost,
                           s_max_v=self.g(self.s_max))
            # 注意：左分支通常继承 forbidden_sets
            open_list_change += 1

        # --- 分支 2：右分支 (约束注入路径) ---
        # 关键创新：候选集 candidate 不减少 (或仅减少首元素)，但注入“互斥约束”
        # 强制要求在该子树下，cluster 中的元素不能同时被选满
        current_forbidden = getattr(node, 'forbidden_sets', [])
        new_forbidden = current_forbidden + [set(cluster)]

        # 对于右分支，为了保证完备性，我们剔除 cluster 中的第一个元素（防止死循环）
        # 但通过 forbidden_sets 限制了剩下的元素组合
        new_candidate_right = list(set(node.candidate) - {cluster[0]})
        new_hs_right = heuristic_sequence[1:]

        # 这里我们利用了之前实现的“惰性定界”，将 new_hs_right 传下去
        self.push_heap(s=node.s, lbd_v=new_lbd, first_child=False,
                       heuristic_sequence=new_hs_right,
                       candidate=new_candidate_right,
                       w=node.budget,
                       s_max_v=self.g(self.s_max))

        # ⚠️ 重要：由于 push_heap 内部可能还没适配 forbidden_sets 传参，
        # 如果你的 EfficientBFSHeapObj 构造函数没改，记得在这里手动补上
        # last_node = self.max_heap.peek() 或修改 push_heap 接口

        open_list_change += 1
        return open_list_change

    def recursive_branching(self, node, heuristic_sequence, tau=0.8, max_k=4):
        """
        修复版：真·密度跳变多叉分支
        不再依赖贪心序列，而是实时海选当前状态下的“头号竞争者”
        """
        base_set = set(node.s)

        # 1. 横向海选：评估所有候选元素在当前 base_set 下的绝对初始密度
        candidates_density = []
        for e in node.candidate:
            d = self.model.marginal_gain(e, list(base_set)) / self.model.cost_of_singleton(e)
            candidates_density.append((d, e))

        # 按密度降序排列，寻找真正的“头牌”和它的“平替”们
        candidates_density.sort(key=lambda x: x[0], reverse=True)

        # 2. 寻找密度跳变点 k
        cluster = [candidates_density[0][1]]
        max_d = candidates_density[0][0]
        k = 1

        # tau 建议保持 0.8 或 0.85，因为现在找的是真正的平替，密度会非常接近
        while k < len(candidates_density) and k < max_k:
            if candidates_density[k][0] < max_d * tau:
                break
            cluster.append(candidates_density[k][1])
            k += 1

        # 如果 k=1，说明老大毫无争议（没有平替），退化为标准二元分支
        if k == 1:
            # 注意：退化时依然可以使用 heuristic_sequence，保留加速特性
            return self.branching(node, heuristic_sequence)

        open_list_change = 0
        new_lbd = node.v.lbd_v

        # 3. 提取出高密度冗余簇
        # 【修正】根据前缀包含法则，T0 应该是“仅排除老大”，而不是“排除所有人”
        candidate_T0 = list(set(node.candidate) - {cluster[0]})

        # === T0 预评估 ===
        ub_T0 = self.g(node.s) + self.fast_evaluate_ub(node.s, candidate_T0, node.budget)

        if ub_T0 * self.alpha > self.g(self.s_max):
            # 熔断保护：T0 (即不选老大) 杀不死，退化为普通二叉分支
            return self.branching(node, heuristic_sequence)
        else:
            # T0 必死！说明最优解必然包含老大 cluster[0]。
            # 接下来用前缀包含法 (10x, 110, 111...) 完美覆盖剩余空间
            current_s = list(base_set)
            current_cost = 0
            branches_to_push = []
            # 正序构建前缀
            for i in range(k):
                target_ele = cluster[i]
                cost_u = self.model.cost_of_singleton(target_ele)

                if node.budget >= current_cost + cost_u:
                    # 💡 修正 1：base_set | cluster[:i+1] 的完美实现
                    current_s.append(target_ele)
                    current_cost += cost_u

                    # 💡 修正 2：排除逻辑
                    if i < k - 1:
                        # 状态 10x, 110: 包含了前 i 个，必须明确排除第 i+1 个
                        new_candidate_Ti = list(set(node.candidate) - set(cluster[:i + 2]))
                        is_first = False
                    else:
                        # 状态 111: 到达前缀末端，全包含，无需额外排除 cluster 内元素
                        new_candidate_Ti = list(set(node.candidate) - set(cluster))
                        is_first = True  # 只有全包含才是顺着贪心走的左分支

                    branches_to_push.append({
                        's': list(current_s),  # 注意拷贝
                        'candidate': new_candidate_Ti,
                        'cost': current_cost,
                        'first_child': is_first
                    })
                else:
                    # 短前缀装不下，长前缀更装不下
                    break

            # ⚠️ 倒序 push，保证最强的全包含分支 (111) 浮在栈顶供 DFS 优先探索
            for branch in reversed(branches_to_push):
                self.push_heap(s=branch['s'],
                               lbd_v=new_lbd,
                               first_child=branch['first_child'],
                               heuristic_sequence=None,
                               candidate=branch['candidate'],
                               w=node.budget - branch['cost'],
                               s_max_v=self.g(self.s_max),
                               depth=node.depth + 1)
                open_list_change += 1

        return open_list_change

    # def branching_with_binary_collapse(self, node, heuristic_sequence, tau=0.85, f_local=float('inf')):
    #     def density(e, s):
    #         return self.model.marginal_gain(e, s) / self.model.cost_of_singleton(e)
    #
    #     def get_top_2_elements(candidate_list, base_set, remaining_budget):
    #         """
    #         以 O(N) 复杂度扫描候选集，找出当前状态下密度最高的前2个合法元素
    #         返回: (e1, d1), (e2, d2)
    #         """
    #         best_e1, best_e2 = None, None
    #         best_d1, best_d2 = -float('inf'), -float('inf')
    #
    #         # 转换为 list 以匹配你的 model 接口
    #         base_list = list(base_set)
    #
    #         for e in candidate_list:
    #             cost = self.model.cost_of_singleton(e)
    #
    #             # 必须校验容量：装不下的元素直接无视
    #             if cost > remaining_budget:
    #                 continue
    #
    #             gain = self.model.marginal_gain(e, base_list)
    #             d = gain / cost
    #
    #             # 擂台法：维护前两大元素
    #             if d > best_d1:
    #                 # 原来的老大退居老二
    #                 best_d2 = best_d1
    #                 best_e2 = best_e1
    #                 # 新元素上位老大
    #                 best_d1 = d
    #                 best_e1 = e
    #             elif d > best_d2:
    #                 # 没打过老大，但打赢了老二，替换老二
    #                 best_d2 = d
    #                 best_e2 = e
    #
    #         return best_e1, best_d1, best_e2, best_d2
    #
    #     # 1. 以 O(N) 极速获取头两号交椅
    #     # e1, d1, e2, d2 = get_top_2_elements(node.candidate, node.s, node.budget)
    #     e1, e2 = heuristic_sequence[0], heuristic_sequence[1],
    #     d1, d2 = density(e1, node.s), density(e2, node.s)
    #
    #     # print(f"examining node:{node.s}, {set(self.model.ground_set) - set(node.candidate)}")
    #     # print(f"heuristic:{heuristic_sequence[:2]}, e1:{e1}, d1:{d1}, e2:{e2}, d2:{d2}")
    #
    #     # 极端情况防御：如果没有合法元素，或者只有一个元素，直接正常分支或结束
    #     if e1 is None:
    #         return
    #     if e2 is None or d2 < tau * d1:
    #         # 没有老二，或者老二差距太大构不成威胁，退化为普通二叉分支
    #         # （你可以调用原来的 self.branching(node, node.heuristic_sequence)）
    #         self.branching(node, heuristic_sequence, f_local)
    #
    #     # ================= 触发双向坍缩 (Double Collapse) =================
    #     open_list_change = 0
    #     current_lb = self.g(self.s_max)
    #     print(f"current_lb:{current_lb}")
    #
    #     # 默认状态
    #     left_base = list(set(node.s) | {e1})
    #     left_cand = list(set(node.candidate) - {e1})
    #     left_budget = node.budget - self.model.cost_of_singleton(e1)
    #
    #     right_base = list(node.s)
    #     right_cand = list(set(node.candidate) - {e1})
    #     right_budget = node.budget
    #
    #     # --- A. 左分支 Look-ahead 测试 ---
    #     # 试探：选 e1，但不选 e2
    #     cand_test_left = list(set(node.candidate) - {e1, e2})
    #     UB_left_test = self.fast_evaluate_ub(left_base, cand_test_left, left_budget)
    #     print(
    #         f"base_left:{left_base}, cand_left:{set(self.model.ground_set) - set(cand_test_left)}, b:{left_budget}, UB_left:{UB_left_test}")
    #
    #     if UB_left_test * self.alpha <= current_lb:
    #         # 💥 左坍缩：选了 e1 必须选 e2！
    #         if left_budget >= self.model.cost_of_singleton(e2):  # 确保 e2 能装下
    #             left_base.append(e2)
    #             left_cand.remove(e2)
    #             left_budget -= self.model.cost_of_singleton(e2)
    #             # print(f"💥 [Left Collapse] {e1} implies {e2}")
    #
    #     # --- B. 右分支 Look-ahead 测试 ---
    #     # 试探：不选 e1，也不选 e2
    #     cand_test_right = list(set(node.candidate) - {e1, e2})
    #     UB_right_test = self.fast_evaluate_ub(right_base, cand_test_right, right_budget)
    #
    #     if UB_right_test * self.alpha <= current_lb:
    #         # 💥 右坍缩：不选 e1 必须选 e2！
    #         if right_budget >= self.model.cost_of_singleton(e2):
    #             right_base.append(e2)
    #             right_cand.remove(e2)
    #             right_budget -= self.model.cost_of_singleton(e2)
    #             # print(f"💥 [Right Collapse] NOT {e1} implies {e2}")
    #
    #     # ================= 入堆 =================
    #     # 右分支 (子节点 2)
    #     self.push_heap(s=right_base,
    #                    lbd_v=node.v.lbd_v,
    #                    first_child=False,
    #                    candidate=right_cand,
    #                    w=right_budget,
    #                    s_max_v=current_lb,
    #                    depth=node.depth + 1)
    #     open_list_change += 1
    #
    #     # 左分支 (子节点 1)
    #     if node.budget >= self.model.cost_of_singleton(e1):
    #         self.push_heap(s=left_base,
    #                        lbd_v=node.v.lbd_v,
    #                        first_child=False,  # 因为发生改变，强制子节点重算序列
    #                        candidate=left_cand,
    #                        w=left_budget,
    #                        s_max_v=current_lb,
    #                        depth=node.depth + 1)
    #         open_list_change += 1
    #
    #     return open_list_change

    def branching_cluster_collapse(self, node, heuristic_sequence, tau=0.85):
        """
        升级版：簇坍缩 (Cluster Collapse)
        专门针对 Youtube 等含有大量高重叠度平替元素的数据集
        """

        def density(e, s):
            return self.model.marginal_gain(e, list(s)) / self.model.cost_of_singleton(e)

        if not heuristic_sequence:
            return 0

        e1 = heuristic_sequence[0]
        cost_e1 = self.model.cost_of_singleton(e1)
        d1 = density(e1, node.s)

        open_list_change = 0
        current_lb = self.g(self.s_max)

        # ==========================================================
        # 1. 默认状态初始化
        # ==========================================================
        left_base = list(set(node.s) | {e1})
        left_cand = list(set(node.candidate) - {e1})
        left_budget = node.budget - cost_e1

        right_base = list(node.s)
        right_cand = list(set(node.candidate) - {e1})
        right_budget = node.budget

        # ==========================================================
        # 2. 核心大杀器：O(K) 簇互斥横扫清洗 (Cluster Clean)
        # 扫描前 15 个高密度元素，揪出所有 e1 的“寄生克隆体”
        # ==========================================================
        removed_clones_count = 0
        scan_limit = min(15, len(heuristic_sequence))  # 视野扩大到前 15 名

        for i in range(1, scan_limit):
            e_test = heuristic_sequence[i]
            if e_test not in left_cand:
                continue

            d_orig = density(e_test, node.s)

            # 只有初始密度足够高（接近 e1）的才有可能是同级别的克隆体
            if d_orig < tau * d1:
                break  # 已经掉出第一梯队，不用再往下看了

            # 计算在 e1 阴影下的真实剩余价值
            gain_after_e1 = self.model.marginal_gain(e_test, left_base)
            d_new = gain_after_e1 / self.model.cost_of_singleton(e_test)

            # # 阈值判定：如果密度暴跌 80% (即不足原来的 0.2)，判定为互斥克隆体
            # if d_new < 0.2 * d_orig:
            #     left_cand.remove(e_test)
            #     removed_clones_count += 1

            # ====== 放宽判定并强制输出分析日志 ======
            if d_new < 0.8 * d_orig:  # 只要衰减了 20% 就视为互相牵制
                left_cand.remove(e_test)
                removed_clones_count += 1
                if node.depth < 3:  # 看看前几层到底发生了什么
                    print(
                        f"[Analysis] {e1} 压制了 {e_test}: 初始密度 {d_orig:.2f} -> 跌至 {d_new:.2f} (保留率 {d_new / d_orig * 100:.1f}%)")

        # ====== 插入点 1：监测坍缩命中情况 ======
        if removed_clones_count > 0 and node.depth < 5:
            # 只打印浅层（前 5 层）的坍缩，防止深层日志刷屏
            print(
                f"💥 [Cluster Collapse] Depth: {node.depth} | 选定主节点: {e1} | 成功物理超度了 {removed_clones_count} 个克隆体!")
            print(f"   -> 剔除的元素可能导致了虚高 UB，当前左分支剩余候选集大小: {len(left_cand)}")
        # ====================================

        # ==========================================================
        # 3. 极速入堆 (跳过昂贵的 look-ahead，交由 push_heap 处理)
        # ==========================================================

        # --- 右分支 (不选 e1) ---
        self.push_heap(s=right_base,
                       lbd_v=node.v.lbd_v,
                       first_child=False,
                       heuristic_sequence=None,
                       candidate=right_cand,
                       w=right_budget,
                       s_max_v=current_lb,
                       depth=node.depth + 1)
        open_list_change += 1

        # --- 左分支 (选 e1，且享受了物理级大清洗) ---
        if node.budget >= cost_e1:
            self.push_heap(s=left_base,
                           lbd_v=node.v.lbd_v,
                           # 如果发生了清洗，状态巨变，必须要求子节点重算贪心序列
                           first_child=False if removed_clones_count > 0 else True,
                           # 如果没清洗，可以继承剔除了 e1 的序列以节省时间
                           heuristic_sequence=None if removed_clones_count > 0 else heuristic_sequence[1:],
                           candidate=left_cand,
                           w=left_budget,
                           s_max_v=current_lb,
                           depth=node.depth + 1)
            open_list_change += 1

        return open_list_change
    def branching_volume_biased(self, node, heuristic_sequence, top_n=5):
        """
        策略 B：大体积优先。
        在密度最高的前 top_n 个元素中，选择 cost（体积）最大的元素进行二元分支。
        """
        # 1. 截取高密度候选池（防止越界）
        pool = heuristic_sequence[:top_n]
        if not pool:
            return 0

        # 2. 选出体积最大的目标元素
        target_ele = max(pool, key=lambda e: self.model.cost_of_singleton(e))

        new_candidate = list(set(node.candidate) - {target_ele})
        new_lbd = node.v.lbd_v
        open_list_change = 0

        # ================= 重要细节 =================
        # 因为 target_ele 可能不是 heuristic_sequence 的第一个元素 (index 0)
        # 根据子模性，跳过中间元素会导致后续元素的边际收益发生变化
        # 所以这里的 first_child 统一设为 False，且 heuristic_sequence 传 None
        # ==========================================

        # Child 2: Exclude target_ele
        self.push_heap(s=node.s,
                       lbd_v=new_lbd,
                       first_child=False,
                       heuristic_sequence=None,
                       candidate=new_candidate,
                       w=node.budget,
                       s_max_v=self.g(self.s_max),
                       depth=node.depth + 1)
        open_list_change += 1

        # Child 1: Include target_ele
        if node.cost + self.model.cost_of_singleton(target_ele) <= self.model.budget:
            open_list_change += 1
            self.push_heap(s=list(set(node.s) | {target_ele}),
                           lbd_v=new_lbd,
                           first_child=False,
                           heuristic_sequence=None,
                           candidate=new_candidate,
                           w=node.budget - self.model.cost_of_singleton(target_ele),
                           s_max_v=self.g(self.s_max),
                           depth=node.depth + 1)
        return open_list_change

    def branching_probing(self, node, heuristic_sequence, top_m=3):
        # 只要当前的 UB 距离我们手里握着的全局 LB 还有一段距离
        # 就说明这棵树还有很大的剪枝潜力，不要心疼算力，老老实实火力全开 (m=3) 去排雷
        if node.v.lbd_v > 1.1 * self.g(self.s_max):
            m = 3
        else:
            # 如果 UB 已经跟 LB 差不多了，哪怕找到“承重墙”也剪不了多少树了
            # 这时候再降级到 m=1 或 m=2
            m = 1

        pool = heuristic_sequence[:top_m]
        if not pool:
            return 0

        best_ele = pool[0]
        lowest_ub = float('inf')

        # 假设当前节点的初始评估上限是 node.v.lbd_v
        if node.v.lbd_v > 1.15 * self.g(self.s_max):
            return self.branching(node, heuristic_sequence)

        # 1. 模拟探测：寻找杀伤力最大的元素
        for e in pool:
            temp_candidates = set(node.candidate) - {e}
            # 模拟计算剔除 e 之后的 UB
            simulated_ub = self.fast_evaluate_ub(node.s, temp_candidates, node.budget)

            if simulated_ub < lowest_ub:
                lowest_ub = simulated_ub
                best_ele = e

        # 2. 选定最佳目标进行分支
        target_ele = best_ele

        if target_ele != pool[0]:
            self.probing_trigger_count += 1
            self.probing_trigger_depth_list.append(node.depth)

        new_candidate = list(set(node.candidate) - {target_ele})
        new_lbd = node.v.lbd_v
        open_list_change = 0

        # Child 2: Exclude
        self.push_heap(s=node.s,
                       lbd_v=new_lbd,
                       first_child=False,
                       heuristic_sequence=None,
                       candidate=new_candidate,
                       w=node.budget,
                       s_max_v=self.g(self.s_max),
                       depth=node.depth + 1)
        open_list_change += 1

        # Child 1: Include
        if node.cost + self.model.cost_of_singleton(target_ele) <= self.model.budget:
            open_list_change += 1
            self.push_heap(s=list(set(node.s) | {target_ele}),
                           lbd_v=new_lbd,
                           first_child=False,  # 理由同上：非顺序截断需强制重算
                           heuristic_sequence=None,
                           candidate=new_candidate,
                           w=node.budget - self.model.cost_of_singleton(target_ele),
                           s_max_v=self.g(self.s_max),
                           depth=node.depth + 1)

        return open_list_change

    def branching_probing_adapt(self, node, heuristic_sequence):
        if self.m_current == 1:
            return self.branching(node, heuristic_sequence)

        # 1. 动态确定本次探测宽度
        m = self.m_current
        pool = heuristic_sequence[:m]
        if not pool: return 0

        best_ele = pool[0]
        lowest_ub = float('inf')

        # 2. 执行探测
        for e in pool:
            temp_candidates = set(node.candidate) - {e}
            simulated_ub = self.fast_evaluate_ub(node.s, temp_candidates, node.budget)
            if simulated_ub < lowest_ub:
                lowest_ub = simulated_ub
                best_ele = e

        # 3. 统计命中情况
        hit_index = pool.index(best_ele)
        # ================= 级联 EMA 核心逻辑 =================
        # 批量更新当前活跃的几个 Index 的存活概率
        for i in range(self.m_current):
            is_hit = 1.0 if i == hit_index else 0.0
            self.ema_probs[i] = self.ema_alpha * is_hit + (1 - self.ema_alpha) * self.ema_probs[i]

        # 从后往前判定降级
        if self.m_current == 3 and self.ema_probs[2] < self.ema_threshold:
            self.m_current = 2
            # print(f"📉 [EMA] 第3名长期无效 (概率跌至 {self.ema_probs[2]:.3f})，降级为 m=2")
        elif self.m_current == 2 and self.ema_probs[1] < self.ema_threshold:
            self.m_current = 1
            # print(f"📉 [EMA] 第2名长期无效 (概率跌至 {self.ema_probs[1]:.3f})，退化为无探测贪心 (m=1)")
        # ===================================================
        target_ele = best_ele
        # ... 后续 push_heap 逻辑 ...
        if target_ele != pool[0]:
            self.probing_trigger_count += 1
            self.probing_trigger_depth_list.append(node.depth)

        new_candidate = list(set(node.candidate) - {target_ele})
        new_lbd = node.v.lbd_v
        open_list_change = 0

        # Child 2: Exclude
        self.push_heap(s=node.s,
                       lbd_v=new_lbd,
                       first_child=False,
                       heuristic_sequence=None,
                       candidate=new_candidate,
                       w=node.budget,
                       s_max_v=self.g(self.s_max),
                       depth=node.depth + 1)
        open_list_change += 1

        # Child 1: Include
        if node.cost + self.model.cost_of_singleton(target_ele) <= self.model.budget:
            open_list_change += 1
            self.push_heap(s=list(set(node.s) | {target_ele}),
                           lbd_v=new_lbd,
                           first_child=False,  # 理由同上：非顺序截断需强制重算
                           heuristic_sequence=None,
                           candidate=new_candidate,
                           w=node.budget - self.model.cost_of_singleton(target_ele),
                           s_max_v=self.g(self.s_max),
                           depth=node.depth + 1)

        return open_list_change

    def branching_naive(self, node, heuristic_sequence=None):
        """
        朴素 N 叉分支 (Naive N-ary Branching)
        将所有可行的元素各自分支，利用前缀排除法避免组合重复。
        用于作为 Baseline 展示二元/高级分支策略的性能优势。
        """
        open_list_change = 0
        base_set = list(node.s)
        candidates = list(node.candidate)

        if not candidates:
            return 0

        # 按边际收益密度对候选集进行降序排序
        candidates.sort(
            key=lambda x: self.model.marginal_gain(x, base_set) / self.model.cost_of_singleton(x),
            reverse=True
        )

        new_lbd = node.v.lbd_v

        # 💡 倒序遍历入堆：保证最高密度的分支最后入栈 (栈顶)，适配 DFS 下潜
        for i in reversed(range(len(candidates))):
            target_ele = candidates[i]
            cost_u = self.model.cost_of_singleton(target_ele)

            # 校验背包容量
            if node.budget >= cost_u:
                new_s = list(set(base_set) | {target_ele})
                # 排除当前元素及排在它前面的所有元素，避免重复遍历
                new_candidate = list(set(node.candidate) - set(candidates[:i + 1]))

                self.push_heap(s=new_s,
                               lbd_v=new_lbd,
                               first_child=False,  # 朴素分支强制子节点重算启发式序列
                               heuristic_sequence=None,
                               candidate=new_candidate,
                               w=node.budget - cost_u,
                               s_max_v=self.g(self.s_max),
                               depth=node.depth + 1)
                open_list_change += 1

        return open_list_change

    def optimize(self):
        start_time = time.time()
        self.timer_proxy = TimerProxy(timeout_seconds=5000)

        root, f_upper, heuristic_sequence, self.s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(self.s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = self.s_max
            stop_time = time.time()
            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   'push_back_count': 0, 'probing_trigger_count': 0, 'probing_depth_list': [],
                   'max_depth': self.max_depth}
            return ret

        sol = self.s_max
        node_count = 0
        open_list_count = 1

        while self.timer_proxy.is_active and self.max_heap.size() > 0 or len(self.dfs_stack) > 0:
            # ================= 节点弹出逻辑 =================
            if len(self.dfs_stack) > 0:
                # 引擎 A：DFS 下潜模式
                node = self.dfs_stack.pop()
                self.current_dive_count += 1
                # 如果下潜探索的节点数达到上限，强制结束本次下潜
                if self.current_dive_count >= self.dive_max_nodes:
                    # 把栈里剩下的节点全部“倒回”全局大根堆，防止解空间丢失
                    for remaining_node in self.dfs_stack:
                        self.max_heap.push(remaining_node)  # 用你原有的入堆方法
                    self.dfs_stack.clear()
                    self.is_diving = False
            else:
                # 引擎 B：正常的 BFS 模式
                node = self.max_heap.pop()
                node_count += 1

                # ====== 全透视追踪：节点弹出 ======
                print(
                    f"\n🟢 [POP] Node #{node_count} | Depth: {node.depth} | Cost: {node.cost}/{self.model.budget} | UB: {node.v.lbd_v:.2f}")
                print(f"   => 当前集合 S: {node.s}")
                print(f"   => 剩余候选集大小: {len(node.candidate)}")
                # ===============================

                # 触发判定：是否到了该下潜的时候？
                if getattr(self, 'use_dive', False) and node_count > 0 and node_count % self.dive_interval == 0:
                    self.is_diving = True
                    self.current_dive_count = 0
                    print(f"🌊 [DIVE] Initiating Dive at BFS node {node_count}...")
            # =========================================================

            if self.max_depth < node.depth:
                self.max_depth = node.depth

            f_local, heuristic_sequence = node.v.lbd_v, None

            # 只有未访问过的节点需要评估
            if not node.visited:
                has_legacy = (node.heuristic_sequence is not None and len(node.heuristic_sequence) > 0)
                # 左分支沿用老路
                if node.first_child and has_legacy:
                    f_local = node.v.lbd_v
                    heuristic_sequence = node.heuristic_sequence
                # 右分支重新计算
                else:
                    s_final, f_local, heuristic_sequence = self.greedy_add(node)
                    if not self.timer_proxy.is_active:
                        break
                    f_upper = min(f_upper, node.v.lbd_v)

                    # --- 更新全局最优 LB ---
                    if self.g(s_final) > self.g(self.s_max) + 1e-6:
                        # ====== 插入点 2：监测下界突破 ======
                        print(
                            f"🌟 [LB 突破!] 树深度: {node.depth} | 新的全局最优 LB: {self.g(s_final):.4f} (原为 {self.g(self.s_max):.4f})")
                        # ==================================
                        self.s_max = s_final

                    # --- Local Search 兜底提界 ---
                    if self.local_search:
                        if self.g(s_final) > 0.98 * self.g(self.s_max):
                            enhanced_s, enhanced_val = self.fast_local_swap_hs(self.s_max, heuristic_sequence)
                            if enhanced_val > self.g(self.s_max):
                                self.s_max = enhanced_s
                                print(f"🚀 [LS Hit] Improved global LB to {enhanced_val:.2f}")

                if self.branching_strategy != 'fullbab' and min(node.v.lbd_v, f_local) * self.alpha <= self.g(
                        self.s_max):
                    continue

            if node.visited or node.first_child:
                heuristic_sequence = node.heuristic_sequence

            if self.is_on_the_edge(node):
                continue

            # ================= 对比实验开关 =================
            if self.branching_strategy == "traditional":
                # 策略 A: 传统二元分支
                self.branching(node, heuristic_sequence, f_local=f_local)
            elif self.branching_strategy == "fullbab":
                # 策略 A: 传统二元分支
                self.branching(node, heuristic_sequence)
            elif self.branching_strategy == "density_gap":
                # 策略 D: 密度跳变多叉分支
                self.recursive_branching(node, heuristic_sequence, tau=0.8, max_k=4)
            elif self.branching_strategy == 'volume_biased':
                self.branching_volume_biased(node, heuristic_sequence, top_n=5)
            elif self.branching_strategy == 'probing':
                self.branching_probing(node, heuristic_sequence)
            elif self.branching_strategy == 'cluster_collapse':
                self.branching_cluster_collapse(node, heuristic_sequence, f_local)
            if self.branching_strategy == "naive":
                self.branching_naive(node, heuristic_sequence)
            # ================================================

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               'probing_trigger_count': self.probing_trigger_count,
               'probing_trigger_depth_list': self.probing_trigger_depth_list, 'max_depth': self.max_depth,
               "TLE": not self.timer_proxy.is_active}

        return ret

    def fast_evaluate_ub(self, base, candidate_T0, budget):
        opt = acclerated_upper_bounds.LazyPlainOptimizer(self.model)
        opt.build(base=base, remaining=candidate_T0)
        return self.g(base) + opt.solve(candidate_T0, budget)

    def fast_local_swap(self, current_sol, candidate_set):
        budget = self.model.budget
        best_sol = list(current_sol)
        best_val = self.g(best_sol)
        current_cost = self.model.cost_of_set(best_sol)

        out_candidates = list(best_sol)
        in_candidates = list(set(candidate_set) - set(best_sol))
        improved = True

        while improved:
            improved = False
            for e_out in out_candidates:
                for e_in in in_candidates:
                    cost_diff = self.model.cost_of_singleton(e_in) - self.model.cost_of_singleton(e_out)
                    if current_cost + cost_diff <= budget:
                        temp_sol = list((set(best_sol) - {e_out}) | {e_in})
                        temp_val = self.g(temp_sol)
                        if temp_val > best_val:
                            best_val = temp_val
                            best_sol = temp_sol
                            current_cost += cost_diff

                            out_candidates.remove(e_out)
                            out_candidates.append(e_in)
                            in_candidates.remove(e_in)
                            in_candidates.append(e_out)
                            improved = True
                            break

                if improved:
                    break

        return best_sol, best_val

    def fast_local_swap_hs(self, current_sol, candidate_set, max_candidates=30):
        """
        首优退出版 Local Search
        :param current_sol: 当前解
        :param candidate_set: 候选集 (建议传入节点的 heuristic_sequence 而不是整个 ground_set)
        :param max_candidates: 截断参数，限制最大扫描数量，防止退化为 O(kn)
        """
        budget = self.model.budget
        best_sol = list(current_sol)
        best_val = self.g(best_sol)
        current_cost = self.model.cost_of_set(best_sol)

        out_candidates = list(best_sol)

        in_candidates = list(set(candidate_set) - set(best_sol))
        if max_candidates and len(in_candidates) > max_candidates:
            in_candidates = in_candidates[:max_candidates]

        for e_out in out_candidates:
            for e_in in in_candidates:
                cost_diff = self.model.cost_of_singleton(e_in) - self.model.cost_of_singleton(e_out)

                # 检查背包容量
                if current_cost + cost_diff <= budget:
                    temp_sol = list((set(best_sol) - {e_out}) | {e_in})
                    temp_val = self.g(temp_sol)

                    # 优化点 2：首优退出 (First Improvement)
                    # 只要发现界限有提升，立刻返回给 BFS 外层，不做任何多余的留恋和列表维护
                    if temp_val > best_val:
                        return temp_sol, temp_val

        # 如果遍历完截断的候选集都没有发现任何提升，原样返回
        return best_sol, best_val

