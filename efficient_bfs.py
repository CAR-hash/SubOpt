import copy
import csv
import heapq
import os
import time
from datetime import datetime, timezone

import acclerated_upper_bounds
from branching_strategies import get_branching_strategy
from MaxHeap import MaxHeap, SimpleMaxHeap, EfficientBFSHeapObj
from OptimalAlg import OptimalAlg
from base_task import BaseTask
from filter_search import RefinedBFSValue
from push_child_builder import PushChildBuilder
from search_observer import PrintingSearchObserver, SearchObserver

# Upper-bound kinds supported by ``EfficientBFS.configure_upper_bound`` (experiments: ub0 / ub2).
SUPPORTED_UB_KINDS = frozenset({"ub0", "ub2"})

_OPT_SOLVE_LOG_FIELDNAMES = (
    "wall_iso_utc",
    "duration_sec",
    "optimizer_class",
    "configured_ub_type",
    "task",
    "seed",
    "budget",
    "branching",
    "heuristic_cli",
    "call_index",
)


def canonical_ub_kind(kind: str) -> str:
    """
    Normalize CLI / config strings to ``'ub0'`` or ``'ub2'``.

    Accepts optional ``+`` suffixes (e.g. ``ub2+`` → ``ub2``). Other values raise ``ValueError``.
    """
    if not isinstance(kind, str):
        raise TypeError(f"upper bound kind must be str, got {type(kind).__name__}")
    k = kind.strip().lower()
    if k in ("ub0", "ub0+"):
        return "ub0"
    if k in ("ub2", "ub2+"):
        return "ub2"
    raise ValueError(
        f"Unsupported upper bound kind {kind!r}. "
        f"Use one of: {', '.join(sorted(SUPPORTED_UB_KINDS))} (optional '+' suffix on ub0/ub2)."
    )


class TimerProxy:
    """计时器代理，用于控制循环的生命周期"""

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    @property
    def is_active(self):
        """代理拦截口：未超时返回 True，超时返回 False"""
        return (time.time() - self.start_time) <= self.timeout_seconds


class BfsSearchContext:
    """
    One ``optimize()`` run: mutable counters / incumbent bound plus a handle to the solver.

    Template and future branching code can take a ``BfsSearchContext`` instead of threading
    many ``solver`` fields; ``solver`` remains the source of truth for ``s_max``, heap, timer.
    """

    __slots__ = ("solver", "start_time", "f_upper", "node_count", "open_list_count")

    def __init__(self, solver: "EfficientBFS", start_time: float, f_upper: float):
        self.solver = solver
        self.start_time = start_time
        self.f_upper = f_upper
        self.node_count = 0
        self.open_list_count = 1

    @property
    def alpha(self) -> float:
        return self.solver.alpha

    @property
    def branching_strategy(self) -> str:
        return self.solver.branching_strategy

    @property
    def local_search(self) -> bool:
        return self.solver.local_search

    @property
    def model(self):
        return self.solver.model

    @property
    def timer_proxy(self):
        return self.solver.timer_proxy

    @property
    def max_heap(self):
        return self.solver.max_heap

    def g(self, s):
        return self.solver.g(s)

    @property
    def s_max(self):
        return self.solver.s_max

    @s_max.setter
    def s_max(self, value):
        self.solver.s_max = value

    def current_lb(self) -> float:
        return self.solver.g(self.solver.s_max)

    def should_continue(self) -> bool:
        return self.timer_proxy.is_active and self.max_heap.size() > 0


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

        # Per-run wall-clock cap (seconds). Used by ``optimize`` to build ``TimerProxy``.
        # Override via :class:`compute_efficient_config.EfficientRunConfig.time_limit_seconds`.
        self.time_limit_seconds = 5000.0

        # === 继承机制控制器 ===
        self.inherit_bounds = True  # 默认开启继承

        # === 新增：上界评估器类型 ===
        self.ub_type = 'ub2'  # 默认使用 Slicing
        # Auxiliary UB for fast look-ahead evaluations (defaults to historical behavior: ub0/plain).
        self.aux_ub_type = 'ub0'
        self.use_cascade = False  # 是否开启级联过滤

        self.search_observer: SearchObserver = PrintingSearchObserver()

        # Per ``opt.solve`` timing (LazyPlain / LazySlicing); see :meth:`get_optimizer` / ``compute_efficient`` JSON ``opt_solve_log``.
        self.opt_solve_log_path = None
        self.opt_solve_log_meta = None  # optional dict: task, seed, budget, branching, heuristic_cli
        self._opt_solve_call_seq = 0

    def _obs_log(self, event: str, message: str = "", **fields):
        self.search_observer.log(event, message, **fields)

    def _obs_metric(self, name: str, value: float, **labels):
        self.search_observer.metric(name, value, **labels)

    def _obs_timer(self, label: str, seconds: float):
        self.search_observer.log("timer", f"[Timer] {label}: {seconds:.6f}s")
        self.search_observer.metric("timer.seconds", seconds, label=label)

    def push_child(self) -> PushChildBuilder:
        """Fluent builder for ``push_heap`` / ``push_heap_with_ub``."""
        return PushChildBuilder(self)

    def get_optimizer(self, kind: str):
        """
        Build ``LazyPlainOptimizer`` or ``LazySlicingOptimizer`` for :attr:`model`.

        When ``opt_solve_log_path`` is set, wraps with :class:`optimizer_solve_profiler.TimedOptimizerProxy`
        so each ``solve()`` is timed and appended via :meth:`_record_opt_solve`. To change or drop
        monitoring later, edit this method (and optionally :meth:`_record_opt_solve`).
        """
        if kind == "plain":
            inner = acclerated_upper_bounds.LazyPlainOptimizer(self.model)
        elif kind == "slicing":
            inner = acclerated_upper_bounds.LazySlicingOptimizer(self.model)
        else:
            raise ValueError("get_optimizer kind must be 'plain' or 'slicing', got %r" % (kind,))
        if not self.opt_solve_log_path:
            return inner
        from optimizer_solve_profiler import TimedOptimizerProxy

        return TimedOptimizerProxy(inner, self._record_opt_solve)

    def _record_opt_solve(self, optimizer_class_name: str, duration_sec: float) -> None:
        path = self.opt_solve_log_path
        if not path:
            return
        meta = self.opt_solve_log_meta or {}
        self._opt_solve_call_seq += 1
        call_index = self._opt_solve_call_seq
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        row = {
            "wall_iso_utc": datetime.now(timezone.utc).isoformat(),
            "duration_sec": f"{duration_sec:.9f}",
            "optimizer_class": optimizer_class_name,
            "configured_ub_type": getattr(self, "ub_type", ""),
            "task": meta.get("task", ""),
            "seed": meta.get("seed", ""),
            "budget": meta.get("budget", ""),
            "branching": meta.get("branching", ""),
            "heuristic_cli": meta.get("heuristic_cli", ""),
            "call_index": call_index,
        }
        with open(path, "a", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=_OPT_SOLVE_LOG_FIELDNAMES)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def configure_upper_bound(self, kind: str) -> None:
        """
        Single entry point for which UB the solver uses: sets ``ub_type``, ``inner_h`` (via
        :meth:`set_h`), and :meth:`setOpt` so heap ``h`` and lazy optimizers stay aligned.

        Accepts ``ub0``, ``ub0+``, ``ub2``, ``ub2+``; see :func:`canonical_ub_kind`.
        """
        canonical = canonical_ub_kind(kind)
        self.ub_type = canonical
        self.set_h(heuristic=canonical)
        self.setOpt(canonical)

    def configure_aux_upper_bound(self, kind: str) -> None:
        """
        Configure UB kind used by auxiliary/fast-evaluation paths (e.g. ``fast_evaluate_ub``,
        lazy-binary branch probing). Accepts ``ub0``, ``ub0+``, ``ub2``, ``ub2+``.
        """
        self.aux_ub_type = canonical_ub_kind(kind)

    def _get_optimizer(self):
        """根据 ub_type 动态实例化对应的 Lazy Optimizer"""
        if self.ub_type == 'ub0':
            return self.get_optimizer("plain")
        elif self.ub_type == 'ub2':
            return self.get_optimizer("slicing")
        # 兼容其他情况，默认回退
        return self.get_optimizer("slicing")

    def _get_aux_optimizer(self):
        """Auxiliary UB evaluator used by look-ahead/probing paths."""
        if self.aux_ub_type == 'ub2':
            return self.get_optimizer("slicing")
        return self.get_optimizer("plain")

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
            opt_fast = self.get_optimizer("plain")
            opt_fast.build(base=s, remaining=candidate)
            h_fast = opt_fast.solve(candidate, node.budget)

            # 初筛剪枝判定 (注意也要考虑 alpha)
            f_fast = new_g + h_fast
            if self.alpha * f_fast <= self.g(self.s_max):
                return None  # 被 ub0 秒杀，节省了计算 ub2 的巨量时间

        # 第二段：初筛没杀掉，或者没开启级联，执行正式评估
        t_start_h = time.perf_counter()  # 添加开始
        new_h = self.h(node)
        t_end_h = time.perf_counter()  # 添加结束
        self._obs_timer("push_heap -> self.h(node) 耗时", t_end_h - t_start_h)
        final_v = new_g + new_h

        # ====== 全透视追踪：界限评估 ======
        self._obs_log("push_heap_eval", f"  ├── [EVAL] 评估子节点 S: {s}")
        self._obs_log("push_heap_eval", f"  │   ├── g(S): {new_g:.2f} | h(S): {new_h:.2f} | 原始 UB: {final_v:.2f}")
        self._obs_log("push_heap_eval",
                      f"  │   └── 对比条件: min(UB, f_local:{f_local:.2f}) * alpha:{self.alpha} <= s_max_v:{s_max_v:.2f}")

        if min(final_v, f_local) * self.alpha <= self.g(self.s_max):
            self._obs_log("push_heap_eval", "  │   └── ❌ [KILLED] 剪枝生效，节点已被抹杀。")
            return None

        self._obs_log("push_heap_eval", "  │   └── ✅ [SURVIVED] 界限达标，准备入堆！")
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

        t_start_root = time.perf_counter()
        s_max, f_local, heuristic_sequence = self.greedy_add(root)
        t_end_root = time.perf_counter()
        self._obs_timer("首次 greedy_add 耗时", t_end_root - t_start_root)

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
        (self.push_child()
         .s(node.s).lbd_v(new_lbd).first_child(False).candidate(new_candidate).w(node.budget)
         .incumbent_lb_as_s_max_v().depth(node.depth + 1).f_local(f_local).execute())
        open_list_change += 1

        if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
            open_list_change += 1
            new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
            new_heuristic_sequence.pop(0)

            (self.push_child()
             .s(list(set(node.s) | {first_ele})).lbd_v(new_lbd).first_child(True)
             .heuristic_sequence(new_heuristic_sequence).candidate(new_candidate)
             .w(node.budget - self.model.cost_of_singleton(first_ele)).incumbent_lb_as_s_max_v()
             .depth(node.depth + 1).execute())

        return open_list_change

    def branching_lazy_binary(self, node, heuristic_sequence):
        """
        基于惰性评估 (Lazy Evaluation) 的极速双叉分支 (带全透视日志)
        """
        if not heuristic_sequence:
            return 0

        e1 = heuristic_sequence[0]
        cost_e1 = self.model.cost_of_singleton(e1)
        current_lb = self.g(self.s_max)
        open_list_change = 0

        # 1. 初始化惰性评估器 (继承父节点状态)
        t_start_lazy_build = time.perf_counter()  # 添加开始
        opt = self._get_aux_optimizer()
        opt.build(base=set(node.s), remaining=set(node.candidate))
        t_end_lazy_build = time.perf_counter()  # 添加结束
        self._obs_timer("lazy_binary -> opt.build 耗时", t_end_lazy_build - t_start_lazy_build)

        # ==========================================================
        # A. 右分支 (不选 e1)
        # ==========================================================
        right_cand = list(set(node.candidate) - {e1})
        t_start_lazy_solve_r = time.perf_counter()  # 添加开始
        ub_delta_right = opt.solve(remaining_set=set(right_cand), budget=node.budget)
        ub_right = self.model.objective(node.s) + ub_delta_right
        t_end_lazy_solve_r = time.perf_counter()  # 添加结束
        self._obs_timer("lazy_binary -> opt.solve (右分支) 耗时", t_end_lazy_solve_r - t_start_lazy_solve_r)

        if self.alpha * ub_right > current_lb:
            self._obs_log("lazy_binary",
                          f"      ├── 🌿 [BRANCH] 生成右分支(不选) S: {list(node.s)} | 排除: {e1} | UB: {ub_right:.4f}")
            (self.push_child().precomputed_ub(ub_right).s(list(node.s)).candidate(right_cand)
             .w(node.budget).depth(node.depth + 1).execute())
            open_list_change += 1
        else:
            self._obs_log("lazy_binary",
                          f"      ├── ✂️ [PRUNED] 丢弃右分支(不选) S: {list(node.s)} | 排除: {e1} | UB: {ub_right:.4f} <= LB")

        # ==========================================================
        # B. 左分支 (选 e1)
        # ==========================================================
        if node.budget >= cost_e1:
            left_s = list(set(node.s) | {e1})
            left_cand = list(set(node.candidate) - {e1})
            left_budget = node.budget - cost_e1

            # 【极速 O(1) 增量】
            opt.update_base(set(left_s))
            ub_delta_left = opt.solve(remaining_set=set(left_cand), budget=left_budget)
            ub_left = self.model.objective(left_s) + ub_delta_left

            if self.alpha * ub_left > current_lb:
                self._obs_log("lazy_binary",
                              f"      ├── 🌿 [BRANCH] 生成左分支(选入) S: {left_s} | 新增: {e1} | UB: {ub_left:.4f}")
                (self.push_child().precomputed_ub(ub_left).s(left_s).candidate(left_cand).w(left_budget)
                 .depth(node.depth + 1).heuristic_sequence(heuristic_sequence[1:]).execute())
                open_list_change += 1
            else:
                self._obs_log("lazy_binary",
                              f"      ├── ✂️ [PRUNED] 丢弃左分支(选入) S: {left_s} | 新增: {e1} | UB: {ub_left:.4f} <= LB")

        return open_list_change

    def push_heap_with_ub(self, s, candidate, w, depth, pre_computed_ub, heuristic_sequence=None):
        """
        专为惰性评估设计的入堆方法。
        直接接收算好的上界，绝对不再调用昂贵的 self.h() 重算。
        """
        node = EfficientBFSHeapObj(
            s=s,
            candidate=candidate,
            w=w,
            visited=False,
            max_idx=0
        )
        v = RefinedBFSValue(pre_computed_ub, pre_computed_ub, self.d(s))
        # 强行注入预先算好的极速上界
        node.v = v
        node.cost = self.model.budget - w
        node.depth = depth
        node.heuristic_sequence = heuristic_sequence

        # 直接推入你重写过容差比较逻辑的 SimpleMaxHeap
        self.max_heap.push(node)

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

            (self.push_child().s(new_s).lbd_v(new_lbd).first_child(True).heuristic_sequence(new_hs)
             .candidate(new_candidate).w(node.budget - total_cost).incumbent_lb_as_s_max_v().execute())
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
        (self.push_child().s(node.s).lbd_v(new_lbd).first_child(False).heuristic_sequence(new_hs_right)
         .candidate(new_candidate_right).w(node.budget).incumbent_lb_as_s_max_v().execute())

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
                (self.push_child().s(branch['s']).lbd_v(new_lbd).first_child(branch['first_child'])
                 .heuristic_sequence(None).candidate(branch['candidate']).w(node.budget - branch['cost'])
                 .incumbent_lb_as_s_max_v().depth(node.depth + 1).execute())
                open_list_change += 1

        return open_list_change

    def branching_binary_collapse(self, node, heuristic_sequence, tau=0.85, f_local=float('inf')):
        # --- 增加防御性检查 ---
        if not heuristic_sequence:
            return 0

        # 如果只有一个元素，无法进行双向坍缩测试，直接退化到普通分支
        if len(heuristic_sequence) < 2:
            return self.branching(node, heuristic_sequence, f_local)

        def density(e, s):
            return self.model.marginal_gain(e, s) / self.model.cost_of_singleton(e)

        e1, e2 = heuristic_sequence[0], heuristic_sequence[1],
        d1, d2 = density(e1, node.s), density(e2, node.s)

        # 极端情况防御：如果没有合法元素，或者只有一个元素，直接正常分支或结束
        if e1 is None:
            return
        if e2 is None:
            # 没有老二，退化为普通二叉分支
            self.branching(node, heuristic_sequence, f_local)

        # ================= 触发双向坍缩 (Double Collapse) =================
        open_list_change = 0
        current_lb = self.g(self.s_max)

        # 默认状态
        left_base = list(set(node.s) | {e1})
        left_cand = list(set(node.candidate) - {e1})
        left_budget = node.budget - self.model.cost_of_singleton(e1)

        right_base = list(node.s)
        right_cand = list(set(node.candidate) - {e1})
        right_budget = node.budget

        # --- A. 左分支 Look-ahead 测试 ---
        # 试探：选 e1，但不选 e2
        cand_test_left = list(set(node.candidate) - {e1, e2})
        UB_left_test = self.fast_evaluate_ub(left_base, cand_test_left, left_budget)

        if UB_left_test * self.alpha <= current_lb:
            # 💥 左坍缩：选了 e1 必须选 e2！
            if left_budget >= self.model.cost_of_singleton(e2):  # 确保 e2 能装下
                left_base.append(e2)
                left_cand.remove(e2)
                left_budget -= self.model.cost_of_singleton(e2)
                # print(f"💥 [Left Collapse] {e1} implies {e2}")

        # --- B. 右分支 Look-ahead 测试 ---
        # 试探：不选 e1，也不选 e2
        cand_test_right = list(set(node.candidate) - {e1, e2})
        UB_right_test = self.fast_evaluate_ub(right_base, cand_test_right, right_budget)

        if UB_right_test * self.alpha <= current_lb:
            # 💥 右坍缩：不选 e1 必须选 e2！
            if right_budget >= self.model.cost_of_singleton(e2):
                right_base.append(e2)
                right_cand.remove(e2)
                right_budget -= self.model.cost_of_singleton(e2)
                # print(f"💥 [Right Collapse] NOT {e1} implies {e2}")

        # ================= 入堆 =================
        # 右分支 (子节点 2)
        (self.push_child().s(right_base).lbd_v(node.v.lbd_v).first_child(False).candidate(right_cand)
         .w(right_budget).s_max_v(current_lb).depth(node.depth + 1).execute())
        open_list_change += 1

        # 左分支 (子节点 1)
        if node.budget >= self.model.cost_of_singleton(e1):
            (self.push_child().s(left_base).lbd_v(node.v.lbd_v).first_child(False).candidate(left_cand)
             .w(left_budget).s_max_v(current_lb).depth(node.depth + 1).execute())
            open_list_change += 1

        return open_list_change

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
        removed_clones = []
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
                # left_cand.remove(e_test)
                removed_clones.append(e_test)
                removed_clones_count += 1

            # ==========================================================
            # 💥 核心杀招：左侧互斥，右侧破缺 (Symmetry Breaking)
            # ==========================================================
            for clone in removed_clones:
                left_cand.remove(clone)  # 左分支：选了 e1，就不选克隆体 (互斥)
                right_cand.remove(clone)  # 右分支：连 e1 都不选，更不可能选它的克隆体 (对称性破缺)

        # ====== 插入点 1：监测坍缩命中情况 ======
        if removed_clones_count > 0 and node.depth < 5:
            # 只打印浅层（前 5 层）的坍缩，防止深层日志刷屏
            self._obs_log(
                "cluster_collapse",
                f"💥 [Cluster Collapse] Depth: {node.depth} | 选定主节点: {e1} | 成功物理超度了 {removed_clones_count} 个克隆体!")
            self._obs_log(
                "cluster_collapse",
                f"   -> 剔除的元素可能导致了虚高 UB，当前左分支剩余候选集大小: {len(left_cand)}")
            self._obs_metric("cluster_collapse.removed_clones", float(removed_clones_count), depth=node.depth)
        # ====================================

        # ==========================================================
        # 3. 极速入堆 (跳过昂贵的 look-ahead，交由 push_heap 处理)
        # ==========================================================

        # --- 右分支 (不选 e1) ---
        (self.push_child().s(right_base).lbd_v(node.v.lbd_v).first_child(False).heuristic_sequence(None)
         .candidate(right_cand).w(right_budget).s_max_v(current_lb).depth(node.depth + 1).execute())
        open_list_change += 1

        # --- 左分支 (选 e1，且享受了物理级大清洗) ---
        if node.budget >= cost_e1:
            (self.push_child().s(left_base).lbd_v(node.v.lbd_v)
             .first_child(False if removed_clones_count > 0 else True)
             .heuristic_sequence(None if removed_clones_count > 0 else heuristic_sequence[1:])
             .candidate(left_cand).w(left_budget).s_max_v(current_lb).depth(node.depth + 1).execute())
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
        (self.push_child().s(node.s).lbd_v(new_lbd).first_child(False).heuristic_sequence(None)
         .candidate(new_candidate).w(node.budget).incumbent_lb_as_s_max_v().depth(node.depth + 1).execute())
        open_list_change += 1

        # Child 1: Include target_ele
        if node.cost + self.model.cost_of_singleton(target_ele) <= self.model.budget:
            open_list_change += 1
            (self.push_child().s(list(set(node.s) | {target_ele})).lbd_v(new_lbd).first_child(False)
             .heuristic_sequence(None).candidate(new_candidate)
             .w(node.budget - self.model.cost_of_singleton(target_ele)).incumbent_lb_as_s_max_v()
             .depth(node.depth + 1).execute())
        return open_list_change

    def branching_probing(self, node, heuristic_sequence, top_m=3):
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
        (self.push_child().s(node.s).lbd_v(new_lbd).first_child(False).heuristic_sequence(None)
         .candidate(new_candidate).w(node.budget).incumbent_lb_as_s_max_v().depth(node.depth + 1).execute())
        open_list_change += 1

        # Child 1: Include
        if node.cost + self.model.cost_of_singleton(target_ele) <= self.model.budget:
            open_list_change += 1
            (self.push_child().s(list(set(node.s) | {target_ele})).lbd_v(new_lbd).first_child(False)
             .heuristic_sequence(None).candidate(new_candidate)
             .w(node.budget - self.model.cost_of_singleton(target_ele)).incumbent_lb_as_s_max_v()
             .depth(node.depth + 1).execute())

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
        (self.push_child().s(node.s).lbd_v(new_lbd).first_child(False).heuristic_sequence(None)
         .candidate(new_candidate).w(node.budget).incumbent_lb_as_s_max_v().depth(node.depth + 1).execute())
        open_list_change += 1

        # Child 1: Include
        if node.cost + self.model.cost_of_singleton(target_ele) <= self.model.budget:
            open_list_change += 1
            (self.push_child().s(list(set(node.s) | {target_ele})).lbd_v(new_lbd).first_child(False)
             .heuristic_sequence(None).candidate(new_candidate)
             .w(node.budget - self.model.cost_of_singleton(target_ele)).incumbent_lb_as_s_max_v()
             .depth(node.depth + 1).execute())

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

                (self.push_child().s(new_s).lbd_v(new_lbd).first_child(False).heuristic_sequence(None)
                 .candidate(new_candidate).w(node.budget - cost_u).incumbent_lb_as_s_max_v()
                 .depth(node.depth + 1).execute())
                open_list_change += 1

        return open_list_change

    def optimize(self):
        """Best-first search on ``max_heap``; orchestration uses :meth:`_bfs_search_template`."""
        return self._bfs_search_template()

    def _bfs_search_template(self):
        """
        Template method: timer + root, optional trivial exit, main loop, result dict.

        Hook sequence: ``ctx.should_continue`` → ``_bfs_pop_open`` → ``_bfs_update_max_depth``
        → ``_bfs_process_node`` (greedy / prune / ``_bfs_apply_branching``) on :class:`BfsSearchContext`.
        """
        start_time = time.time()
        self.timer_proxy = TimerProxy(timeout_seconds=self.time_limit_seconds)

        t_start_root = time.perf_counter()
        root, f_upper, heuristic_sequence, self.s_max = self.push_root()
        t_end_root = time.perf_counter()
        self._obs_timer("push_root (含首次 greedy_add) 耗时", t_end_root - t_start_root)

        if self.g(self.s_max) >= self.alpha * f_upper:
            stop_time = time.time()
            sol = self.s_max
            return {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                    'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                    'push_back_count': 0, 'probing_trigger_count': 0, 'probing_depth_list': [],
                    'max_depth': self.max_depth}

        ctx = BfsSearchContext(self, start_time, f_upper)
        self._bfs_main_loop(ctx)
        return self._bfs_pack_result(ctx)

    def _bfs_main_loop(self, ctx: BfsSearchContext):
        while ctx.should_continue():
            node = self._bfs_pop_open(ctx)
            self._bfs_update_max_depth(node)
            if self._bfs_process_node(node, ctx) == "break":
                break

    def _bfs_pop_open(self, ctx: BfsSearchContext):
        node = ctx.max_heap.pop()
        ctx.node_count += 1
        self._bfs_on_node_popped(node, ctx)
        return node

    def _bfs_on_node_popped(self, node, ctx: BfsSearchContext):
        self._obs_log(
            "pop",
            f"\n🟢 [POP] Node #{ctx.node_count} | Depth: {node.depth} | Cost: {node.cost}/{ctx.model.budget} | UB: {node.v.lbd_v:.2f}")
        self._obs_log("pop", f"   => 当前集合 S: {node.s}")
        self._obs_log("pop", f"   => 剩余候选集大小: {len(node.candidate)}")
        self._obs_metric("search.node_pop", float(ctx.node_count), depth=node.depth, ub=float(node.v.lbd_v))

    def _bfs_update_max_depth(self, node):
        if self.max_depth < node.depth:
            self.max_depth = node.depth

    def _bfs_process_node(self, node, ctx: BfsSearchContext):
        """
        Greedy refresh, pruning, edge skip, branching. Return ``"break"`` to exit the main loop.
        """
        f_local, heuristic_sequence = node.v.lbd_v, None

        if not node.visited:
            has_legacy = (node.heuristic_sequence is not None and len(node.heuristic_sequence) > 0)
            if node.first_child and has_legacy:
                f_local = node.v.lbd_v
                heuristic_sequence = node.heuristic_sequence
            else:
                t_start_greedy = time.perf_counter()
                s_final, f_local, heuristic_sequence = self.greedy_add(node)
                t_end_greedy = time.perf_counter()
                self._obs_timer("greedy_add 耗时", t_end_greedy - t_start_greedy)

                if not ctx.timer_proxy.is_active:
                    return "break"
                ctx.f_upper = min(ctx.f_upper, node.v.lbd_v)

                if ctx.g(s_final) > ctx.current_lb() + 1e-6:
                    old_lb = ctx.current_lb()
                    self._obs_log(
                        "lb",
                        f"🌟 [LB 突破!] 树深度: {node.depth} | 新的全局最优 LB: {ctx.g(s_final):.4f} (原为 {old_lb:.4f})")
                    self._obs_metric("search.lb", float(ctx.g(s_final)), depth=node.depth, old_lb=float(old_lb))
                    ctx.s_max = s_final

                if ctx.local_search:
                    if ctx.g(s_final) > 0.98 * ctx.current_lb():
                        enhanced_s, enhanced_val = self.fast_local_swap_hs(ctx.s_max, heuristic_sequence)
                        if enhanced_val > ctx.current_lb():
                            ctx.s_max = enhanced_s
                            self._obs_log("local_search", f"🚀 [LS Hit] Improved global LB to {enhanced_val:.2f}")
                            self._obs_metric("search.local_search_lb", float(enhanced_val))

            if ctx.branching_strategy != 'fullbab' and min(node.v.lbd_v, f_local) * ctx.alpha <= ctx.current_lb():
                return None

        if node.visited or node.first_child:
            heuristic_sequence = node.heuristic_sequence

        if self.is_on_the_edge(node):
            return None

        self._bfs_apply_branching(ctx, node, heuristic_sequence, f_local)
        return None

    def _bfs_apply_branching(self, ctx: BfsSearchContext, node, heuristic_sequence, f_local):
        t_start_branch = time.perf_counter()
        strategy = get_branching_strategy(ctx.branching_strategy)
        strategy.branch(self, node, heuristic_sequence, f_local)
        t_end_branch = time.perf_counter()
        self._obs_timer(f"策略 {ctx.branching_strategy} 整体执行耗时", t_end_branch - t_start_branch)
        self._obs_metric("search.branch_seconds", t_end_branch - t_start_branch, strategy=ctx.branching_strategy)

    def _bfs_pack_result(self, ctx: BfsSearchContext):
        stop_time = time.time()
        sol = ctx.s_max
        assert sol is not None, "No solution found."
        return {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                'time': stop_time - ctx.start_time, 'node_count': ctx.node_count,
                "open_list_count": ctx.open_list_count,
                'probing_trigger_count': self.probing_trigger_count,
                'probing_trigger_depth_list': self.probing_trigger_depth_list, 'max_depth': self.max_depth,
                "TLE": not ctx.timer_proxy.is_active}

    def fast_evaluate_ub(self, base, candidate_T0, budget):
        opt = self._get_aux_optimizer()
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


class AdaptiveEfficientBFS(EfficientBFS):
    """
    自适应极速分支定界求解器。
    核心思想：根据节点剩余预算的压力，在 ub2 (Slicing，紧致但昂贵) 和 ub0 (Plain，较松但极速) 之间动态切换引擎，
    在保证算法数学完备性（Upper Bound 绝对合法）的前提下，实现算力收益最大化。
    """

    def __init__(self, model):
        super().__init__(model)
        # 动态切换阈值：当节点剩余预算大于总预算的这个比例时，启用 ub2 压制树规模
        self.adaptive_ratio = 0.4
        # 单元素收益缓存，用于抹除右分支空集的重复计算开销
        self.singleton_gain_cache = {}

    def greedy_add(self, node):
        def density(ele, base_set):
            # 极速逻辑：若是空集 S=[] 且有缓存，直接 O(1) 返回
            if not base_set and ele in self.singleton_gain_cache:
                return self.singleton_gain_cache[ele]

            gain = self.model.marginal_gain(ele, list(base_set))
            cost = self.model.cost_of_singleton(ele)

            # 若是空集计算，顺手存入缓存
            if not base_set:
                self.singleton_gain_cache[ele] = gain / cost
            return gain / cost

        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        forbidden_sets = getattr(node, 'forbidden_sets', [])
        remaining_elements = set(candidate)
        cur_cost = 0
        heuristic_sequence = []

        # =================================================================
        # 1. 动态引擎切换 (数学完备性保证)
        # 绝不放弃计算上界，而是根据预算压力选择对应精度的 Optimizer
        # =================================================================
        use_tight_bound = budget > (self.model.budget * self.adaptive_ratio)

        if use_tight_bound:
            # 预算充足，组合空间大：上 Slicing 优化器 (ub2)
            opt = self.get_optimizer("slicing")
        else:
            # 预算较少，规模可控：上 Plain 优化器 (ub0)
            opt = self.get_optimizer("plain")

        # 统一构建底层数据结构
        opt.build(base=base, remaining=remaining_elements)

        # 获取初始合法上界（无论是 ub0 还是 ub2，在这里算出的都是严谨的理论上限）
        f_local = self.g(sol) + opt.solve(remaining_elements, budget)

        # =================================================================
        # 2. 贪心序列初始化
        # =================================================================
        h = []
        tie_breaker = 0
        for e in remaining_elements:
            heapq.heappush(h, (-density(e, sol), tie_breaker, e))
            tie_breaker += 1

        # =================================================================
        # 3. 核心贪心选择循环
        # =================================================================
        while self.timer_proxy.is_active and h:
            neg_ds, _, u = heapq.heappop(h)

            # --- 约束注入拦截逻辑 ---
            is_forbidden = False
            for f_set in forbidden_sets:
                if u in f_set:
                    if f_set - {u} <= sol:
                        is_forbidden = True
                        break
            if is_forbidden:
                continue

            cost_u = self.model.cost_of_singleton(u)
            if cur_cost + cost_u > budget:
                continue

            actual_ds = density(u, sol)

            # 淘汰由于基础集合更新导致密度暴跌的元素
            while h:
                top_e = h[0][2]
                if cur_cost + self.model.cost_of_singleton(top_e) > budget:
                    heapq.heappop(h)
                else:
                    break

            if not h or actual_ds >= -h[0][0]:
                # 确认选中元素 u
                sol.add(u)
                heuristic_sequence.append(u)
                cur_cost += cost_u

                # ==== 动态上界追踪 ====
                if use_tight_bound:
                    # 只有在 ub2 模式下，才在贪心循环内步步收紧界限
                    opt.update_base(sol)
                    remaining_for_opt = set(node.candidate) - sol
                    f_temp = self.g(sol) + opt.solve(remaining_for_opt, budget)
                    if f_local is None or f_temp < f_local:
                        f_local = f_temp
                # ⚠️ ub0 模式下：直接忽略循环内部的上界更新，沿用初始合法的 f_local 换取极速

            else:
                heapq.heappush(h, (-actual_ds, tie_breaker, u))
                tie_breaker += 1

        # 返回局部下界(sol)，合法上界(f_local)，以及启发式序列(heuristic_sequence)
        return list(sol), f_local, heuristic_sequence