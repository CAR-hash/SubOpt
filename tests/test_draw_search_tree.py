"""Tests for ``tools/draw_search_tree.py`` and ``runlog.RunLogger.node_push``."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest

# Allow importing top-level modules and the ``tools`` package when running directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import runlog  # noqa: E402
from tools import draw_search_tree as dst  # noqa: E402


class TestParseEventLine(unittest.TestCase):
    def test_basic(self):
        ev, kv = dst.parse_event_line("[NODE_PUSH] parent_id=0 id=1 s=[1,2] ub=3.5 status=pushed")
        self.assertEqual(ev, "NODE_PUSH")
        self.assertEqual(kv["parent_id"], "0")
        self.assertEqual(kv["s"], "[1,2]")
        self.assertEqual(kv["status"], "pushed")

    def test_quoted_value(self):
        ev, kv = dst.parse_event_line('[RUN_END] status=OK final_s="[1, 2]"')
        self.assertEqual(ev, "RUN_END")
        self.assertEqual(kv["final_s"], "[1, 2]")

    def test_non_event_line_returns_none(self):
        self.assertIsNone(dst.parse_event_line("trigger count:0, depth_list:[]"))
        self.assertIsNone(dst.parse_event_line(""))


class TestBuildTree(unittest.TestCase):
    def _run(self, lines):
        events = [parsed for parsed in (dst.parse_event_line(l) for l in lines)
                  if parsed is not None]
        return dst.build_tree(events)

    def test_minimal_tree(self):
        # Root pops, pushes two children, pops the survivor, finds the solution.
        view = self._run([
            "[RUN_START] algorithm=BFSTC task=t seed=0 budget=3.0 heuristic=ub2 alpha=1.0 strategy=none",
            "[NODE_POP] node=1 id=0 s=[] ub=10",
            "[NODE_PUSH] parent_id=0 id=1 s=[1] ub=8 status=pushed",
            "[NODE_PUSH] parent_id=0 id=2 s=[2] ub=4 status=pruned_pre",
            "[NODE_PRUNE] reason=alpha_lb ub=4 lb=5 id=2",
            "[NODE_POP] node=2 id=1 s=[1] ub=8",
            "[INCUMBENT] f_s=8 prev=0 set_size=1 s=[1]",
            "[RUN_END] status=OK f_s=8 c_s=1 nodes=2 time_s=0.5 final_s=[1]",
        ])
        # Root + 2 children = 3 nodes.
        self.assertEqual(len(view.nodes), 3)
        self.assertEqual(view.nodes[0].pop_seq, 1)
        self.assertEqual(view.nodes[1].pop_seq, 2)
        self.assertEqual(view.nodes[1].s, "[1]")
        self.assertEqual(view.nodes[2].status, "pruned_pre")
        self.assertEqual(view.nodes[2].prune_reason, "alpha_lb")
        self.assertEqual(view.final_s, "[1]")
        self.assertEqual(view.incumbent_f, 8.0)


class TestRenderTree(unittest.TestCase):
    def _build(self, lines):
        events = [parsed for parsed in (dst.parse_event_line(l) for l in lines)
                  if parsed is not None]
        return dst.build_tree(events)

    def test_render_includes_root_and_pop_indices(self):
        view = self._build([
            "[RUN_START] algorithm=BFSTC task=t seed=0 budget=3.0 heuristic=ub2 alpha=1.0",
            "[NODE_POP] node=1 id=0 s=[] ub=10",
            "[NODE_PUSH] parent_id=0 id=1 s=[1] ub=8 status=pushed",
            "[NODE_PUSH] parent_id=0 id=2 s=[2] ub=4 status=pruned_pre",
            "[NODE_PRUNE] reason=alpha_lb ub=4 lb=5 id=2",
            "[NODE_POP] node=2 id=1 s=[1] ub=8",
            "[RUN_END] status=OK f_s=8 c_s=1 nodes=2 time_s=0.5 final_s=[1]",
        ])
        text = dst.render_tree(view)
        # Root marker.
        self.assertIn("(Root)", text)
        # Both children appear; popped one carries (2), pruned one carries [pruned: ...].
        self.assertIn("(2) [1]", text)
        self.assertIn("[pruned: alpha_lb]", text)
        # The popped final leaf is annotated as Done.
        self.assertIn("[Done]", text)

    def test_hide_pruned(self):
        view = self._build([
            "[NODE_POP] node=1 id=0 s=[] ub=10",
            "[NODE_PUSH] parent_id=0 id=1 s=[1] ub=8 status=pushed",
            "[NODE_PUSH] parent_id=0 id=2 s=[2] ub=4 status=pruned_pre",
            "[NODE_PRUNE] reason=alpha_lb ub=4 lb=5 id=2",
        ])
        text = dst.render_tree(view, hide_pruned=True)
        self.assertNotIn("[pruned", text)
        self.assertIn("[1]", text)


class TestNodePushLogger(unittest.TestCase):
    def test_node_push_emits_when_verbose(self):
        buf = io.StringIO()
        log = runlog.RunLogger(stream=buf, verbose=True)
        log.node_push(parent_id=0, id=1, s="[1,2]", ub=3.5, status="pushed")
        out = buf.getvalue()
        self.assertIn("[NODE_PUSH]", out)
        self.assertIn("parent_id=0", out)
        self.assertIn("status=pushed", out)
        self.assertIn("s=[1,2]", out)

    def test_node_push_silent_when_not_verbose(self):
        buf = io.StringIO()
        log = runlog.RunLogger(stream=buf, verbose=False)
        log.node_push(parent_id=0, id=1, s="[1,2]", ub=3.5, status="pushed")
        self.assertEqual(buf.getvalue(), "")

    def test_next_node_id_is_monotonic(self):
        log = runlog.RunLogger(stream=io.StringIO())
        ids = [log.next_node_id() for _ in range(5)]
        self.assertEqual(ids, [1, 2, 3, 4, 5])

    def test_fmt_set(self):
        self.assertEqual(runlog.fmt_set([3, 1, 2]), "[1,2,3]")
        self.assertEqual(runlog.fmt_set([]), "[]")
        self.assertEqual(runlog.fmt_set(None), "[]")


class TestEndToEndBFSTC(unittest.TestCase):
    """Capture a real BFSTC run and round-trip it through the parser."""

    def setUp(self):
        try:
            from filter_search import BFSTC  # noqa: F401
            from base_task import BaseTask  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("filter_search not importable")

    def _model(self):
        from base_task import BaseTask

        class TinyKnapsackTask(BaseTask):
            def __init__(self, n=6, budget=3.0):
                super().__init__()
                self._ground = list(range(n))
                self.budget = budget
                self.costs_obj = [1.0 for _ in range(n)]
                self._val = [1.0 + 0.15 * i for i in range(n)]

            @property
            def ground_set(self):
                return self._ground

            def internal_objective(self, S):
                return float(sum(self._val[i] for i in S)) if S else 0.0

        return TinyKnapsackTask(n=6, budget=3.0)

    def test_bfstc_log_roundtrips(self):
        from filter_search import BFSTC

        with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False, encoding="utf-8") as fp:
            log_path = fp.name
            log = runlog.RunLogger(stream=fp, verbose=True, run_id="rt")
            log.run_start(
                algorithm="BFSTC", task="tiny", seed=0, budget=3.0,
                heuristic="ub2", alpha=1.0,
            )
            alg = BFSTC(self._model())
            alg.alpha = 1.0
            alg.set_h("ub2")
            alg.runlog = log
            alg.build()
            alg.optimize()

        try:
            view = dst.build_tree(dst.iter_events(log_path))
            text = dst.render_tree(view)
            # The render should at least name the algorithm and have a root.
            self.assertIn("(Root)", text)
            self.assertIn("BFSTC", text)
            # A non-trivial run should pop at least one node beyond the root.
            popped = [n for n in view.nodes.values() if n.pop_seq is not None]
            self.assertGreaterEqual(len(popped), 1)
        finally:
            os.unlink(log_path)


if __name__ == "__main__":
    unittest.main()
