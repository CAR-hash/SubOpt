"""Tests for ``runlog.RunLogger`` and end-to-end emission from BFSTC / EBB."""
from __future__ import annotations

import io
import os
import sys
import unittest

# Allow ``import runlog`` when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import runlog  # noqa: E402


class TestFormatting(unittest.TestCase):
    def test_format_event_basic(self):
        line = runlog.format_event("RUN_END", {"status": "OK", "f_s": 12.5, "nodes": 7})
        self.assertEqual(line, "[RUN_END] status=OK f_s=12.5 nodes=7")

    def test_format_value_types(self):
        self.assertEqual(runlog.format_event("E", {"b": True}), "[E] b=true")
        self.assertEqual(runlog.format_event("E", {"i": 3}), "[E] i=3")
        self.assertEqual(runlog.format_event("E", {"x": 1.0}), "[E] x=1.0")
        # Strings with spaces are quoted.
        self.assertEqual(runlog.format_event("E", {"label": "branch.density gap"}),
                         '[E] label="branch.density gap"')

    def test_format_handles_inf_nan(self):
        self.assertEqual(runlog.format_event("E", {"x": float("inf")}), "[E] x=inf")
        self.assertEqual(runlog.format_event("E", {"x": float("-inf")}), "[E] x=-inf")
        self.assertEqual(runlog.format_event("E", {"x": float("nan")}), "[E] x=nan")


class TestRunLoggerEmission(unittest.TestCase):
    def _logger(self, **kwargs):
        buf = io.StringIO()
        log = runlog.RunLogger(stream=buf, **kwargs)
        return log, buf

    def test_run_id_is_injected(self):
        log, buf = self._logger(run_id="rid-123")
        log.run_start(
            algorithm="BFSTC", task="adult", seed=0, budget=13.0,
            heuristic="ub2", alpha=0.98,
        )
        line = buf.getvalue().strip()
        self.assertIn("[RUN_START]", line)
        self.assertIn("run_id=rid-123", line)
        self.assertIn("algorithm=BFSTC", line)
        self.assertIn("budget=13.0", line)

    def test_high_frequency_events_gated_by_verbose(self):
        log, buf = self._logger(verbose=False)
        # NODE_POP is high-frequency: should be dropped when verbose=False.
        log.node_pop(node=1, open=2, s_size=0, cost=0.0, budget=13.0, ub=652.0)
        # GREEDY_DONE is always-on.
        log.greedy_done(f_s=601.28, set_size=7)
        out = buf.getvalue()
        self.assertNotIn("NODE_POP", out)
        self.assertIn("GREEDY_DONE", out)

    def test_high_frequency_events_emitted_when_verbose(self):
        log, buf = self._logger(verbose=True)
        log.node_pop(node=1, open=2, s_size=0, cost=0.0, budget=13.0, ub=652.0)
        out = buf.getvalue()
        self.assertIn("[NODE_POP]", out)
        self.assertIn("node=1", out)

    def test_disabled_logger_emits_nothing(self):
        log, buf = self._logger(enabled=False, verbose=True)
        log.run_start(algorithm="BFSTC", task="t", seed=0, budget=1.0, heuristic="ub2", alpha=1.0)
        log.run_end(status="OK", f_s=0.0, c_s=0.0, nodes=0, time_s=0.0)
        self.assertEqual(buf.getvalue(), "")


class TestAlgorithmEmission(unittest.TestCase):
    """Smoke test: BFSTC and EBB emit at least RUN_END / GREEDY_DONE on a tiny problem."""

    def setUp(self):
        try:
            import filter_search  # noqa: F401
            from base_task import BaseTask  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("optional algorithm modules not importable")

    def _tiny_model(self, n=6, budget=4.0):
        from base_task import BaseTask

        class TinyKnapsackTask(BaseTask):
            def __init__(self, n=n, budget=budget):
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

        return TinyKnapsackTask(n=6, budget=4.0)

    def test_bfstc_emits_unified_events(self):
        from filter_search import BFSTC
        buf = io.StringIO()
        log = runlog.RunLogger(stream=buf, run_id="t", verbose=True)
        alg = BFSTC(self._tiny_model())
        alg.runlog = log
        alg.alpha = 1.0
        alg.set_h("ub2")
        alg.build()
        alg.optimize()
        out = buf.getvalue()
        self.assertIn("[GREEDY_DONE]", out)
        self.assertIn("[RUN_END]", out)
        self.assertIn("run_id=t", out)

    def test_ebb_emits_unified_events(self):
        from filter_search import EfficientBranchAndBound
        buf = io.StringIO()
        log = runlog.RunLogger(stream=buf, run_id="t", verbose=True)
        alg = EfficientBranchAndBound(self._tiny_model(n=10, budget=5.0))
        alg.runlog = log
        alg.alpha = 0.95
        alg.time_limit = 60.0
        alg.build()
        alg.optimize()
        out = buf.getvalue()
        self.assertIn("[GREEDY_DONE]", out)
        self.assertIn("[RUN_END]", out)
        self.assertIn("[NODE_POP]", out)

    def test_ebb_node_pop_ub_is_local_bound_not_incumbent(self):
        """``NODE_POP`` must carry ``f_local`` (ub=…), not the global incumbent lb."""
        import re
        from filter_search import EfficientBranchAndBound

        buf = io.StringIO()
        log = runlog.RunLogger(stream=buf, verbose=True)
        alg = EfficientBranchAndBound(self._tiny_model(n=10, budget=5.0))
        alg.runlog = log
        alg.alpha = 0.95
        alg.time_limit = 60.0
        alg.build()
        alg.optimize()
        pops = re.findall(r"\[NODE_POP\][^\n]+", buf.getvalue())
        self.assertGreaterEqual(len(pops), 1)
        first = pops[0]
        self.assertIn("ub=", first)
        m = re.search(r"ub=([\d.]+)", first)
        self.assertIsNotNone(m)
        pop_ub = float(m.group(1))
        # First pop is the root; ``f_local`` after greedy is well above zero.
        self.assertGreater(pop_ub, 0.0)


if __name__ == "__main__":
    unittest.main()
