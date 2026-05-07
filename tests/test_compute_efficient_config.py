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

