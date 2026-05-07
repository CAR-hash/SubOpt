"""
Unit tests for ``efficient_bfs`` and closely coupled helpers
(``push_child_builder``, ``BfsSearchContext``, observers, branching registry).
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from base_task import BaseTask

from branching_strategies import BRANCHING_REGISTRY, get_branching_strategy
from efficient_bfs import BfsSearchContext, EfficientBFS, TimerProxy, canonical_ub_kind
from push_child_builder import PushChildBuilder
from search_observer import CompositeSearchObserver, NullSearchObserver


class TinyKnapsackTask(BaseTask):
    """Minimal modular coverage-style task (no external datasets)."""

    def __init__(self, n: int = 6, budget: float = 4.0):
        super().__init__()
        self._ground = list(range(n))
        self.budget = budget
        self.costs_obj = [1.0 for _ in range(n)]
        self._val = [1.0 + 0.15 * i for i in range(n)]

    @property
    def ground_set(self):
        return self._ground

    def internal_objective(self, S):
        if not S:
            return 0.0
        return float(sum(self._val[i] for i in S))


def _make_alg(n: int = 6, budget: float = 4.0, branching: str = "traditional") -> EfficientBFS:
    model = TinyKnapsackTask(n=n, budget=budget)
    alg = EfficientBFS(model)
    alg.use_alpha = True
    alg.alpha = 0.85
    alg.local_search = False
    alg.use_cascade = False
    alg.branching_strategy = branching
    alg.search_observer = NullSearchObserver()
    alg.set_d("g")
    alg.configure_upper_bound("ub0")
    alg.build()
    return alg


class TestCanonicalUbKind(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(canonical_ub_kind("ub2+"), "ub2")
        self.assertEqual(canonical_ub_kind("UB0"), "ub0")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            canonical_ub_kind("dom")


class TestConfigureUpperBound(unittest.TestCase):
    def test_aligns_ub_type_and_inner_h(self):
        model = TinyKnapsackTask(n=3)
        alg = EfficientBFS(model)
        alg.configure_upper_bound("ub0+")
        self.assertEqual(alg.ub_type, "ub0")
        self.assertEqual(alg.inner_h.__name__, "h_ub0")


class TestTimerProxy(unittest.TestCase):
    def test_active_within_timeout(self):
        p = TimerProxy(3600.0)
        self.assertTrue(p.is_active)


class TestBranchingStrategies(unittest.TestCase):
    def test_registry_contains_expected_keys(self):
        for name in (
            "traditional",
            "fullbab",
            "density_gap",
            "volume_biased",
            "probing",
            "cluster_collapse",
            "naive",
            "binary_collapse",
            "lazy_binary",
        ):
            with self.subTest(name=name):
                self.assertIn(name, BRANCHING_REGISTRY)
                self.assertIs(get_branching_strategy(name), BRANCHING_REGISTRY[name])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            get_branching_strategy("not_a_real_strategy")


class TestSearchObserver(unittest.TestCase):
    def test_null_is_silent(self):
        o = NullSearchObserver()
        o.log("x", "hello")
        o.metric("m", 1.0, a=1)

    def test_composite_forwards(self):
        inner = MagicMock()
        o = CompositeSearchObserver([inner])
        o.log("e", "msg", k=1)
        o.metric("name", 2.0, x=3)
        inner.log.assert_called_once_with("e", "msg", k=1)
        inner.metric.assert_called_once_with("name", 2.0, x=3)


class TestPushChildBuilder(unittest.TestCase):
    def test_evaluated_forwards_to_push_heap(self):
        solver = MagicMock()
        solver.g.return_value = 7.0
        solver.push_heap.return_value = "ok"
        ret = (
            PushChildBuilder(solver)
            .s([0])
            .lbd_v(10.0)
            .candidate([1, 2])
            .w(3.0)
            .depth(1)
            .first_child(False)
            .s_max_v(7.0)
            .execute()
        )
        self.assertEqual(ret, "ok")
        solver.push_heap.assert_called_once()
        args, kwargs = solver.push_heap.call_args
        self.assertEqual(args[0], [0])
        self.assertEqual(args[1], 10.0)
        self.assertEqual(kwargs.get("candidate"), [1, 2])
        self.assertEqual(kwargs.get("w"), 3.0)
        self.assertEqual(kwargs.get("depth"), 1)

    def test_incumbent_lb_fills_s_max_v_when_omitted(self):
        solver = MagicMock()
        solver.g.return_value = 42.0
        solver.push_heap.return_value = None
        PushChildBuilder(solver).s([]).lbd_v(1.0).candidate([]).w(1.0).incumbent_lb_as_s_max_v().execute()
        kwargs = solver.push_heap.call_args[1]
        self.assertEqual(kwargs["s_max_v"], 42.0)

    def test_precomputed_ub_requires_fields(self):
        solver = MagicMock()
        with self.assertRaises(ValueError):
            PushChildBuilder(solver).precomputed_ub(1.0).execute()

    def test_precomputed_ub_calls_push_heap_with_ub(self):
        solver = MagicMock()
        (
            PushChildBuilder(solver)
            .precomputed_ub(9.5)
            .s([0])
            .candidate([1])
            .w(2.0)
            .depth(3)
            .heuristic_sequence([1, 2])
            .execute()
        )
        solver.push_heap_with_ub.assert_called_once_with([0], [1], 2.0, 3, 9.5, [1, 2])


class TestBfsSearchContext(unittest.TestCase):
    def test_delegates_to_solver(self):
        model = TinyKnapsackTask(n=3, budget=2.0)
        alg = EfficientBFS(model)
        alg.s_max = [0]
        alg.alpha = 0.9
        alg.branching_strategy = "naive"
        alg.local_search = False
        alg.timer_proxy = TimerProxy(100.0)
        alg.max_heap = MagicMock()
        alg.max_heap.size.return_value = 0

        ctx = BfsSearchContext(alg, start_time=0.0, f_upper=100.0)
        self.assertIs(ctx.solver, alg)
        self.assertEqual(ctx.alpha, 0.9)
        self.assertEqual(ctx.branching_strategy, "naive")
        self.assertFalse(ctx.local_search)
        self.assertIs(ctx.model, model)
        self.assertFalse(ctx.should_continue())

        ctx.s_max = [1]
        self.assertEqual(alg.s_max, [1])


class TestEfficientBFSIntegration(unittest.TestCase):
    def test_build_initializes_heap(self):
        alg = _make_alg()
        self.assertIsNotNone(alg.max_heap)
        self.assertEqual(alg.max_heap.size(), 0)

    def test_optimize_returns_expected_keys(self):
        alg = _make_alg(n=5, budget=3.0, branching="traditional")
        out = alg.optimize()
        for key in ("S", "c(S)", "f(S)", "time", "node_count", "open_list_count", "max_depth"):
            with self.subTest(key=key):
                self.assertIn(key, out)
        self.assertIsInstance(out["S"], list)
        # Early optimum exit omits ``TLE``; main loop result includes it.
        if "TLE" in out:
            self.assertFalse(out["TLE"])

    def test_push_child_on_real_solver_runs(self):
        alg = _make_alg(n=4, budget=10.0)
        alg.s_max = [0]
        root_lbd = 50.0
        node = (
            alg.push_child()
            .s([1])
            .lbd_v(root_lbd)
            .candidate([2, 3])
            .w(alg.model.budget - 1.0)
            .depth(1)
            .first_child(False)
            .incumbent_lb_as_s_max_v()
            .execute()
        )
        # May be pruned depending on bounds; either None or a heap node
        if node is not None:
            self.assertTrue(alg.max_heap.size() >= 0)


if __name__ == "__main__":
    unittest.main()
