import json
import os
import tempfile
import unittest
from pathlib import Path

import compute_efficient_config


class TestLoadEfficientRunConfig(unittest.TestCase):
    def test_defaults_merge_partial(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "num": 50}, f)
            path = f.name
        try:
            cfg = compute_efficient_config.load_efficient_run_config(path)
            self.assertEqual(cfg.task, "youtube")
            self.assertEqual(cfg.num, 50)
            self.assertEqual(cfg.heuristic, "ub2")
            self.assertEqual(cfg.heuristics, ["ub2"])
            self.assertEqual(cfg.branching, ["traditional", "density_gap", "look_ahead"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_key_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "sensor", "typo_field": 1}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                compute_efficient_config.load_efficient_run_config(path)
            self.assertIn("typo_field", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_heuristic(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"heuristic": "ub99"}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                compute_efficient_config.load_efficient_run_config(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_heuristics_list_supported(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "heuristics": ["ub0", "ub2"]}, f)
            path = f.name
        try:
            cfg = compute_efficient_config.load_efficient_run_config(path)
            self.assertEqual(cfg.heuristics, ["ub0", "ub2"])
            self.assertEqual(cfg.heuristic, "ub0")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_heuristics_list_invalid_value_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "heuristics": ["ub0", "bad"]}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                compute_efficient_config.load_efficient_run_config(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_flat_raises_when_datasets_key_present(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"datasets": [{"task": "youtube"}], "archive": "1"}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                compute_efficient_config.load_efficient_run_config(path)
            self.assertIn("load_efficient_run_configs", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_example_file_loads_multi(self):
        root = Path(__file__).resolve().parent.parent
        ex = root / "configs" / "compute_efficient.example.json"
        cfgs = compute_efficient_config.load_efficient_run_configs(ex)
        self.assertEqual(len(cfgs), 2)
        self.assertEqual(cfgs[0].task, "youtube")
        self.assertEqual(cfgs[0].num, 100)
        self.assertEqual(cfgs[1].task, "adult")
        self.assertEqual(cfgs[1].num, 200)
        self.assertIsNone(cfgs[0].opt_solve_log)

    def test_load_configs_flat_returns_singleton(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "facebook", "num": 10}, f)
            path = f.name
        try:
            cfgs = compute_efficient_config.load_efficient_run_configs(path)
            self.assertGreaterEqual(len(cfgs), 1)
            self.assertEqual(cfgs[0].task, "facebook")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_multi_datasets_unknown_entry_key(self):
        raw = {
            "archive": "1",
            "datasets": [{"task": "youtube", "bad_key": 1}],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(raw, f)
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                compute_efficient_config.load_efficient_run_configs(path)
            self.assertIn("datasets[0]", str(ctx.exception))
            self.assertIn("bad_key", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_resolve_archive_104(self):
        root = Path(__file__).resolve().parent.parent
        prev = os.getcwd()
        try:
            os.chdir(root)
            p = compute_efficient_config.resolve_config_path_for_archive("104")
            self.assertTrue(p.is_file())
            cfgs = compute_efficient_config.load_efficient_run_configs(p)
            self.assertGreaterEqual(len(cfgs), 1)
            self.assertTrue(all(c.archive == "104" for c in cfgs))
            self.assertIn("youtube", [c.task for c in cfgs])
        finally:
            os.chdir(prev)

    def test_resolve_unknown_archive_raises(self):
        root = Path(__file__).resolve().parent.parent
        prev = os.getcwd()
        try:
            os.chdir(root)
            with self.assertRaises(FileNotFoundError):
                compute_efficient_config.resolve_config_path_for_archive(
                    "no_such_archive_zzzzz_999"
                )
        finally:
            os.chdir(prev)

    def test_default_algorithm_is_efficient_bfs(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube"}, f)
            path = f.name
        try:
            cfg = compute_efficient_config.load_efficient_run_config(path)
            self.assertEqual(cfg.algorithm, "EfficientBFS")
            self.assertFalse(cfg.adaptive)
            self.assertTrue(cfg.supports_branching)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_explicit_algorithm_bfstc_disables_branching(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "algorithm": "BFSTC"}, f)
            path = f.name
        try:
            cfg = compute_efficient_config.load_efficient_run_config(path)
            self.assertEqual(cfg.algorithm, "BFSTC")
            self.assertFalse(cfg.supports_branching)
            self.assertFalse(cfg.adaptive)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_algorithm_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "algorithm": "NotARealAlg"}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                compute_efficient_config.load_efficient_run_config(path)
            self.assertIn("algorithm", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_legacy_adaptive_true_is_promoted_with_warning(self):
        import warnings
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "adaptive": True}, f)
            path = f.name
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", DeprecationWarning)
                cfg = compute_efficient_config.load_efficient_run_config(path)
            self.assertEqual(cfg.algorithm, "AdaptiveEfficientBFS")
            self.assertTrue(cfg.adaptive)
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_explicit_algorithm_overrides_legacy_adaptive_flag(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            # adaptive=true but explicit algorithm wins; cfg.adaptive stays in sync.
            json.dump({"task": "youtube", "adaptive": True, "algorithm": "EfficientBFS"}, f)
            path = f.name
        try:
            cfg = compute_efficient_config.load_efficient_run_config(path)
            self.assertEqual(cfg.algorithm, "EfficientBFS")
            self.assertFalse(cfg.adaptive)
        finally:
            Path(path).unlink(missing_ok=True)


class TestAlgorithmFactory(unittest.TestCase):
    """
    Tests that build_algorithm() picks the right class and gracefully degrades when
    optional dependencies (filter_search) are missing.
    """

    def _make_cfg(self, **overrides):
        # Minimal config dict to drive the dataclass through the validator.
        base = {"task": "youtube"}
        base.update(overrides)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(base, f)
            path = f.name
        try:
            return compute_efficient_config.load_efficient_run_config(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_supported_set_matches_config_choices(self):
        import algorithm_factory
        self.assertEqual(
            algorithm_factory.SUPPORTED_ALGORITHMS,
            compute_efficient_config.ALGORITHM_CHOICES,
        )

    def test_branching_subset(self):
        import algorithm_factory
        self.assertTrue(
            algorithm_factory.ALGORITHMS_WITH_BRANCHING.issubset(
                algorithm_factory.SUPPORTED_ALGORITHMS
            )
        )
        self.assertIn("EfficientBFS", algorithm_factory.ALGORITHMS_WITH_BRANCHING)
        self.assertIn("AdaptiveEfficientBFS", algorithm_factory.ALGORITHMS_WITH_BRANCHING)
        self.assertNotIn("BFSTC", algorithm_factory.ALGORITHMS_WITH_BRANCHING)
        self.assertNotIn("EfficientBranchAndBound", algorithm_factory.ALGORITHMS_WITH_BRANCHING)

    def test_build_unknown_algorithm_raises(self):
        import algorithm_factory
        # Bypass config validation by faking a cfg-like object with a bogus algorithm.
        class _FakeCfg:
            algorithm = "MadeUp"
        with self.assertRaises(ValueError):
            algorithm_factory.build_algorithm(_FakeCfg(), model=None, heuristic="ub2")

    def test_strip_plus_suffix(self):
        import algorithm_factory
        self.assertEqual(algorithm_factory._strip_plus("ub2+"), "ub2")
        self.assertEqual(algorithm_factory._strip_plus("ub0+"), "ub0")
        self.assertEqual(algorithm_factory._strip_plus("ub2"), "ub2")
        self.assertEqual(algorithm_factory._strip_plus("ub0"), "ub0")


class TestTimeLimitSeconds(unittest.TestCase):
    def test_default_time_limit_is_5000(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube"}, f)
            path = f.name
        try:
            cfg = compute_efficient_config.load_efficient_run_config(path)
            self.assertEqual(cfg.time_limit_seconds, 5000.0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_explicit_time_limit_loads(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "time_limit_seconds": 120.5}, f)
            path = f.name
        try:
            cfg = compute_efficient_config.load_efficient_run_config(path)
            self.assertEqual(cfg.time_limit_seconds, 120.5)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_non_positive_time_limit_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "time_limit_seconds": 0}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                compute_efficient_config.load_efficient_run_config(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_non_numeric_time_limit_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"task": "youtube", "time_limit_seconds": "abc"}, f)
            path = f.name
        try:
            with self.assertRaises(TypeError):
                compute_efficient_config.load_efficient_run_config(path)
        finally:
            Path(path).unlink(missing_ok=True)


class TestFilterSearchAlgorithms(unittest.TestCase):
    """Smoke tests for the BFSTC / EBB cleanups."""

    def test_bfstc_does_not_shadow_h(self):
        import filter_search
        # Stub model: only attributes accessed in __init__/build are needed here.
        class _M:
            ground_set = []
            budget = 1.0
        alg = filter_search.BFSTC(_M())
        # ``__init__`` no longer assigns ``self.h = None``; ``self.h`` should resolve to
        # the base-class bound method which guards on ``is_on_the_edge``.
        self.assertTrue(callable(alg.h))
        self.assertNotIn("h", alg.__dict__)

    def test_ebb_default_verbose_is_false(self):
        import filter_search
        class _M:
            ground_set = []
            budget = 1.0
        alg = filter_search.EfficientBranchAndBound(_M())
        self.assertFalse(alg.verbose)

    def test_ebb_set_h_is_noop(self):
        import filter_search
        class _M:
            ground_set = []
            budget = 1.0
        alg = filter_search.EfficientBranchAndBound(_M())
        # All three labels accepted without raising; nothing is mutated.
        for kind in ("ub0", "ub2", "dom"):
            alg.set_h(kind)

    def test_ebb_no_recursive_bab_method(self):
        import filter_search
        # The dead recursive ``bab`` was deleted; only ``bab_stack`` should remain.
        self.assertFalse(hasattr(filter_search.EfficientBranchAndBound, "bab"))
        self.assertTrue(hasattr(filter_search.EfficientBranchAndBound, "bab_stack"))

