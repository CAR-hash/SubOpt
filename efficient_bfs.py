import copy
import heapq
import time

import acclerated_upper_bounds
from MaxHeap import MaxHeap, SimpleMaxHeap, EfficientBFSHeapObj
from OptimalAlg import OptimalAlg
from base_task import BaseTask
from filter_search import RefinedBFSValue


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
                  s_max_v=0, depth=0, forbidden_sets=None):

        # 1. 统一构建完整的节点
        max_idx = max(s) if len(s) > 0 else 0
        node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited,
                                   first_child=first_child, heuristic_sequence=heuristic_sequence,
                                   max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)
        node.depth = depth
        node.forbidden_sets = forbidden_sets if forbidden_sets is not None else []

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        # 2. 统一进行界限判定和 Alpha 剪枝
        if final_v >= s_max_v:
            if self.use_alpha:
                lbd_v = min(new_g + self.alpha * new_h, lbd_v)
                v = RefinedBFSValue(new_g + self.alpha * new_h, lbd_v, self.d(s))
            else:
                lbd_v = min(new_g + new_h, lbd_v)
                v = RefinedBFSValue(new_g + new_h, lbd_v, self.d(s))

            node.v = v

            # 3. 终极分流：进栈还是进堆？
            if getattr(self, 'is_diving', False) and self.current_dive_count < self.dive_max_nodes:
                # 【下潜模式】：压入 DFS 栈
                self.dfs_stack.append(node)
            else:
                # 【正常模式】：压入大根堆
                self.max_heap.push(node)

            return node

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
        opt = acclerated_upper_bounds.LazySlicingOptimizer(self.model)
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

        while h:

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

    def branching(self, node, heuristic_sequence):
        # push first child
        first_ele = heuristic_sequence[0]
        new_candidate = list(set(node.candidate) - {first_ele})
        # new_lbd = min(node.v.lbd_v, f_local)
        new_lbd = node.v.lbd_v
        open_list_change = 0

        # push second child
        self.push_heap(s=node.s, lbd_v=new_lbd, first_child=False,
                       candidate=new_candidate, w=node.budget, s_max_v=self.g(self.s_max), depth=node.depth + 1)
        open_list_change += 1

        # print(f"current lb_star:{self.g(s_max)}, first ele:{first_ele}, density:{self.model.density(first_ele,s)}")
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

        k = len(cluster)
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
        candidate_T0 = list(set(node.candidate) - set(cluster))

        # === T0 预评估 (快速预判切除这 k 个巨头后的后果) ===
        ub_T0 = self.g(node.s) + self.fast_evaluate_ub(node.s, candidate_T0, node.budget)

        if ub_T0 * self.alpha > self.g(self.s_max):
            # 熔断保护：T0 杀不死，退化为普通二叉分支
            return self.branching(node, heuristic_sequence)
        else:
            # T0 必死！使用 MECE 阶梯分割法完美覆盖剩余空间

            # ⚠️ 倒序循环 (从 k-1 递减到 0)
            # 先 push 最弱的分支 (沉在栈底)，最后 push 最强的 T_0 分支 (浮在栈顶供 DFS 优先探索)
            for i in range(k - 1, -1, -1):
                target_ele = cluster[i]

                if node.budget >= self.model.cost_of_singleton(target_ele):
                    new_s = list(base_set | {target_ele})

                    # 💡 完备性核心逻辑：
                    # 当前分支必须排除 cluster 中排在 target_ele 前面的所有元素
                    # 也就是排除了 cluster[:i]，同时自己 target_ele 被选中了也不在候选集里
                    # 所以新的候选集 = 原候选集 - cluster[:i+1]
                    new_candidate_Ti = list(set(node.candidate) - set(cluster[:i + 1]))
                    self.push_heap(s=new_s, lbd_v=new_lbd, first_child=False,
                                   heuristic_sequence=None,
                                   candidate=new_candidate_Ti,
                                   w=node.budget - self.model.cost_of_singleton(target_ele),
                                   s_max_v=self.g(self.s_max),
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
        """
        策略 C：探测分支 (Strong Branching / Probing)。
        对前 top_m 个高密度元素进行剔除模拟，选择能让 UB 降得最低的元素进行分支。
        """
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

    def optimize(self):
        start_time = time.time()

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

        while self.max_heap.size() > 0 or len(self.dfs_stack) > 0:
            # ================= 节点弹出逻辑 (双引擎切换) =================
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
                    # print("🛬 [DIVE] Max depth reached. Surfacing to BFS...")
            else:
                # 引擎 B：正常的 BFS 模式
                node = self.max_heap.pop()
                node_count += 1

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
                safety_margin = 1.05
                is_safe = (node.v.lbd_v * self.alpha > self.g(self.s_max) * safety_margin)
                has_legacy = (node.heuristic_sequence is not None and len(node.heuristic_sequence) > 0)

                # 左分支直接沿用老路，绝对不跑 greedy_add
                if node.first_child and has_legacy:
                    f_local = node.v.lbd_v
                    heuristic_sequence = node.heuristic_sequence

                # 右分支虽然偏离轨道，但如果上界看起来很高，懒得去精确计算，直接放行
                elif not node.first_child and is_safe and has_legacy:
                    f_local = node.v.lbd_v
                    heuristic_sequence = node.heuristic_sequence

                # 🛑 只有一种情况跑 greedy_add：
                # 右分支（或根节点）且上界逼近危险区！必须重算精确上界进行剪杀，或寻找新的 s_max 路线。
                else:
                    s_final, f_local, heuristic_sequence = self.greedy_add(node)
                    f_upper = min(f_upper, node.v.lbd_v)

                    # --- 更新全局最优 LB ---
                    if self.g(s_final) > self.g(self.s_max):
                        self.s_max = s_final

                    # --- Local Search 兜底提界 ---
                    if self.local_search:
                        if self.g(s_final) > 0.98 * self.g(self.s_max):
                            enhanced_s, enhanced_val = self.fast_local_swap_hs(self.s_max, heuristic_sequence)
                            if enhanced_val > self.g(self.s_max):
                                self.s_max = enhanced_s
                                print(f"🚀 [LS Hit] Improved global LB to {enhanced_val:.2f}")

                if min(node.v.lbd_v, f_local) * self.alpha <= self.g(self.s_max):
                    continue

            # f_local, heuristic_sequence = node.v.lbd_v, None
            # if not node.visited and not node.first_child:
            #     s_final, f_local, heuristic_sequence = self.greedy_add(node)
            #     # node.v.lbd_v = min(node.v.lbd_v, f_local)
            #     f_upper = min(f_upper, node.v.lbd_v)
            #
            #     if self.g(s_final) > self.g(self.s_max):
            #         self.s_max = s_final
            #
            #     if self.local_search:
            #         if self.g(s_final) > 0.98 * self.g(self.s_max):
            #             # enhanced_s, enhanced_val = self.fast_local_swap(self.s_max, node.candidate)
            #             # ✅ 修改为 self.model.ground_set
            #             # enhanced_s, enhanced_val = self.fast_local_swap(self.s_max, self.model.ground_set)
            #             enhanced_s, enhanced_val = self.fast_local_swap_hs(self.s_max, heuristic_sequence)
            #             if enhanced_val > self.g(self.s_max):
            #                 self.s_max = enhanced_s
            #                 print(f"LS improved LB to {enhanced_val:.2f}")
            #         # print(f"s_max updated:{self.s_max}, early pruning:{min(node.v.lbd_v, f_local) * self.alpha}")
            #
            #     if min(node.v.lbd_v, f_local) * self.alpha <= self.g(self.s_max):
            #         continue
            #
            #     if self.use_alpha:
            #         if self.g(self.s_max) >= f_upper:
            #             sol = self.s_max
            #             break
            #     else:
            #         if self.g(self.s_max) >= self.alpha * f_upper:
            #             sol = self.s_max
            #             break

            if node.visited or node.first_child:
                heuristic_sequence = node.heuristic_sequence

            if self.is_on_the_edge(node):
                continue

            # ================= 对比实验开关 =================
            if self.branching_strategy == "traditional":
                # 策略 A: 传统二元分支
                self.branching(node, heuristic_sequence)

            elif self.branching_strategy == "density_gap":
                # 策略 D: 密度跳变多叉分支
                self.recursive_branching(node, heuristic_sequence, tau=0.8, max_k=4)

            elif self.branching_strategy == 'volume_biased':
                self.branching_volume_biased(node, heuristic_sequence, top_n=5)

            elif self.branching_strategy == 'probing':
                self.branching_probing(node, heuristic_sequence)

            elif self.branching_strategy == 'injection':
                self.branching_with_injection(node, heuristic_sequence)

            # ================================================

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count, 'probing_trigger_count': self.probing_trigger_count,
               'probing_trigger_depth_list': self.probing_trigger_depth_list, 'max_depth': self.max_depth}

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