"""
JSON configuration for :mod:`compute_efficient`.

Two shapes are supported:

* **Flat (legacy):** a single JSON object with all keys (including ``task``).
* **Multi-dataset:** the same keys as shared defaults at the root, plus a non-empty
  ``datasets`` array of objects. Each object is merged on top of those defaults; every
  merged row must have a non-empty ``task`` (set per entry or via shared defaults).

Use :func:`load_efficient_run_configs` from the driver (always returns a non-empty list).
:func:`load_efficient_run_config` only accepts the flat shape (for tests and simple files).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Mapping, Optional

HEURISTIC_CHOICES: FrozenSet[str] = frozenset({"ub0", "ub0+", "ub2", "ub2+"})

ALGORITHM_CHOICES: FrozenSet[str] = frozenset({
    "EfficientBFS",
    "AdaptiveEfficientBFS",
    "BFSTC",
    "EfficientBranchAndBound",
})

# Algorithms that support branching strategies; others ignore ``branching``.
_ALGORITHMS_WITH_BRANCHING: FrozenSet[str] = frozenset({
    "EfficientBFS",
    "AdaptiveEfficientBFS",
})

_DEFAULT_FLAT: dict[str, Any] = {
    "task": "sensor",
    "num": 100,
    "archive": "27",
    "algorithm": "EfficientBFS",
    "heuristic": "ub2",
    "heuristics": ["ub2"],
    "aux_heuristic": "ub0",
    "alpha": 0.8,
    "sorting": "d",
    "branching": ["traditional", "density_gap", "look_ahead"],
    "start_seed": 0,
    "stop_seed": 1,
    "local_search": False,
    "no_inherit": False,
    "cascade": False,
    "opt_solve_log": None,
    # ``adaptive`` is deprecated: prefer ``algorithm`` = ``"AdaptiveEfficientBFS"``.
    # Kept for backward compatibility; auto-promotes to the new field when the JSON file
    # only contains ``adaptive`` (and no explicit ``algorithm``).
    "adaptive": False,
    "adaptive_ratio": 0.4,
    "budget_start": 10.0,
    "budget_num_points": 1,
    "budget_interval": 1.0,
    # Per-run wall-clock cap (seconds). Applied to ``EfficientBFS`` (TimerProxy) and to
    # ``BFSTC`` / ``EfficientBranchAndBound`` (their internal ``time_limit`` field).
    "time_limit_seconds": 5000.0,
}

ALLOWED_KEYS: FrozenSet[str] = frozenset(_DEFAULT_FLAT.keys())


def resolve_config_path_for_archive(archive: str) -> Path:
    """
    Resolve the JSON path for an experiment archive id.

    Tries, in order:

    1. ``result/archive-{id}/compute_efficient.json``
    2. ``configs/compute_efficient_archive{id}.json``
    """
    a = str(archive).strip()
    if not a:
        raise ValueError("archive id must be non-empty")
    candidates = [
        Path("result") / f"archive-{a}" / "compute_efficient.json",
        Path("configs") / f"compute_efficient_archive{a}.json",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError("No JSON config found for archive %r. Tried: %s" % (a, tried))


@dataclass(frozen=True)
class EfficientRunConfig:
    task: str
    num: int
    archive: str
    algorithm: str
    heuristic: str
    heuristics: list
    aux_heuristic: str
    alpha: float
    sorting: str
    branching: list
    start_seed: int
    stop_seed: int
    local_search: bool
    no_inherit: bool
    cascade: bool
    opt_solve_log: Optional[str]
    adaptive: bool
    adaptive_ratio: float
    budget_start: float
    budget_num_points: int
    budget_interval: float
    time_limit_seconds: float

    @property
    def supports_branching(self) -> bool:
        """Whether ``cfg.branching`` is meaningful for the configured algorithm."""
        return self.algorithm in _ALGORITHMS_WITH_BRANCHING


def default_efficient_config_json() -> str:
    """Pretty-printed example JSON (multi-dataset shape with two tasks)."""
    sample = {k: v for k, v in _DEFAULT_FLAT.items() if k != "task"}
    sample["datasets"] = [
        {"task": "youtube"},
        {"task": "adult", "num": 200},
    ]
    return json.dumps(sample, indent=2, ensure_ascii=False) + "\n"


def _coerce_bool(v: Any, key: str) -> bool:
    if isinstance(v, bool):
        return v
    raise TypeError(f"{key!r} must be a boolean, got {type(v).__name__}")


def _merge_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(raw) - ALLOWED_KEYS
    if unknown:
        raise ValueError(
            "Unknown config key(s): %s. Allowed: %s"
            % (", ".join(sorted(unknown)), ", ".join(sorted(ALLOWED_KEYS)))
        )
    merged = dict(_DEFAULT_FLAT)
    merged.update(raw)
    # Track which keys were supplied by the user (vs. injected defaults). Needed for
    # legacy-vs-explicit disambiguation when promoting the deprecated ``adaptive`` flag.
    merged["_user_keys"] = frozenset(raw)
    return merged


def _resolve_algorithm(m: Mapping[str, Any]) -> str:
    """
    Resolve the algorithm name with backward-compat handling for the deprecated
    ``adaptive`` flag.

    Rules:
    * If ``algorithm`` is set explicitly by the user, use it (and validate).
    * Otherwise, if the user set the legacy ``adaptive=true`` flag, auto-map to
      ``AdaptiveEfficientBFS`` and emit a ``DeprecationWarning``.
    * Otherwise fall back to the merged value (default ``EfficientBFS``).
    """
    user_keys = m.get("_user_keys", frozenset())
    legacy_adaptive = bool(m.get("adaptive", False))
    user_set_algorithm = "algorithm" in user_keys
    user_set_adaptive = "adaptive" in user_keys and legacy_adaptive

    if user_set_algorithm:
        algorithm = str(m["algorithm"]).strip()
    elif user_set_adaptive:
        import warnings as _warnings
        _warnings.warn(
            "Config field 'adaptive' is deprecated; use "
            "'algorithm': 'AdaptiveEfficientBFS' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        algorithm = "AdaptiveEfficientBFS"
    else:
        algorithm = str(m.get("algorithm", _DEFAULT_FLAT["algorithm"]))

    if algorithm not in ALGORITHM_CHOICES:
        raise ValueError(
            "algorithm must be one of %s, got %r"
            % (", ".join(sorted(ALGORITHM_CHOICES)), algorithm)
        )
    return algorithm


def _merged_dict_to_config(m: Mapping[str, Any]) -> EfficientRunConfig:
    algorithm = _resolve_algorithm(m)

    heuristic = m["heuristic"]
    if heuristic not in HEURISTIC_CHOICES:
        raise ValueError(
            "heuristic must be one of %s, got %r"
            % (", ".join(sorted(HEURISTIC_CHOICES)), heuristic)
        )
    raw_heuristics = m.get("heuristics")
    if raw_heuristics is None:
        heuristics = [heuristic]
    else:
        if not isinstance(raw_heuristics, list) or not raw_heuristics:
            raise ValueError("heuristics must be a non-empty JSON array of strings")
        if not all(isinstance(x, str) and x for x in raw_heuristics):
            raise ValueError("heuristics must be a non-empty list of non-empty strings")
        bad_h = [h for h in raw_heuristics if h not in HEURISTIC_CHOICES]
        if bad_h:
            raise ValueError(
                "heuristics contains unsupported value(s): %s. Allowed: %s"
                % (", ".join(sorted(set(bad_h))), ", ".join(sorted(HEURISTIC_CHOICES)))
            )
        heuristics = list(raw_heuristics)
    heuristic = heuristics[0]
    aux_heuristic = m["aux_heuristic"]
    if aux_heuristic not in HEURISTIC_CHOICES:
        raise ValueError(
            "aux_heuristic must be one of %s, got %r"
            % (", ".join(sorted(HEURISTIC_CHOICES)), aux_heuristic)
        )

    branching = m["branching"]
    if not isinstance(branching, list) or not branching:
        raise ValueError("branching must be a non-empty JSON array of strings")
    if not all(isinstance(x, str) and x for x in branching):
        raise ValueError("branching must be a non-empty list of non-empty strings")

    num = m["num"]
    if not isinstance(num, int) or num < 1:
        raise ValueError("num must be an integer >= 1")

    for key in ("start_seed", "stop_seed", "budget_num_points"):
        v = m[key]
        if not isinstance(v, int):
            raise TypeError(f"{key!r} must be an integer, got {type(v).__name__}")

    if m["budget_num_points"] < 1:
        raise ValueError("budget_num_points must be >= 1")

    for key in ("alpha", "adaptive_ratio", "budget_start", "budget_interval", "time_limit_seconds"):
        v = m[key]
        if not isinstance(v, (int, float)):
            raise TypeError(f"{key!r} must be a number, got {type(v).__name__}")
    if float(m["time_limit_seconds"]) <= 0.0:
        raise ValueError("time_limit_seconds must be positive")

    osl = m["opt_solve_log"]
    if osl is not None and (not isinstance(osl, str) or not osl.strip()):
        raise ValueError("opt_solve_log must be null or a non-empty string path")

    ar = float(m["adaptive_ratio"])
    if ar <= 0.0:
        raise ValueError("adaptive_ratio must be positive")

    task = m.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string (set in shared defaults or in each datasets[] entry)")

    # Keep ``adaptive`` in sync with ``algorithm`` so downstream code that still reads
    # the legacy field does not silently disagree with the resolved algorithm.
    adaptive_flag = (algorithm == "AdaptiveEfficientBFS")

    return EfficientRunConfig(
        task=str(task).strip(),
        num=num,
        archive=str(m["archive"]),
        algorithm=algorithm,
        heuristic=str(heuristic),
        heuristics=list(heuristics),
        aux_heuristic=str(aux_heuristic),
        alpha=float(m["alpha"]),
        sorting=str(m["sorting"]),
        branching=list(branching),
        start_seed=m["start_seed"],
        stop_seed=m["stop_seed"],
        local_search=_coerce_bool(m["local_search"], "local_search"),
        no_inherit=_coerce_bool(m["no_inherit"], "no_inherit"),
        cascade=_coerce_bool(m["cascade"], "cascade"),
        opt_solve_log=osl.strip() if isinstance(osl, str) else None,
        adaptive=adaptive_flag,
        adaptive_ratio=ar,
        budget_start=float(m["budget_start"]),
        budget_num_points=m["budget_num_points"],
        budget_interval=float(m["budget_interval"]),
        time_limit_seconds=float(m["time_limit_seconds"]),
    )


def load_efficient_run_config(path: str | Path) -> EfficientRunConfig:
    """Load a **flat** single-run JSON file (no ``datasets`` key)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p.resolve()}")

    with p.open(encoding="utf-8-sig") as fp:
        raw = json.load(fp)

    if not isinstance(raw, dict):
        raise TypeError("JSON root must be an object (dict), got %s" % type(raw).__name__)

    if "datasets" in raw:
        raise ValueError(
            'This file uses a "datasets" list. Use load_efficient_run_configs() '
            "or run compute_efficient.py (it loads all datasets in one command)."
        )

    return _merged_dict_to_config(_merge_raw(raw))


def load_efficient_run_configs(path: str | Path) -> list[EfficientRunConfig]:
    """
    Load one or more :class:`EfficientRunConfig` instances from a JSON file.

    * If the root object contains ``datasets`` (non-empty array), each element is merged
      onto shared defaults built from ``_DEFAULT_FLAT`` plus all root keys except ``datasets``.
    * Otherwise, the file is treated as a single flat config (same as :func:`load_efficient_run_config`).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p.resolve()}")

    with p.open(encoding="utf-8-sig") as fp:
        raw = json.load(fp)

    if not isinstance(raw, dict):
        raise TypeError("JSON root must be an object (dict), got %s" % type(raw).__name__)

    if "datasets" not in raw:
        return [_merged_dict_to_config(_merge_raw(raw))]

    dss = raw["datasets"]
    if not isinstance(dss, list) or len(dss) < 1:
        raise ValueError("datasets must be a non-empty JSON array of objects")

    shared = {k: v for k, v in raw.items() if k != "datasets"}
    unknown_root = set(shared) - ALLOWED_KEYS
    if unknown_root:
        raise ValueError(
            "Unknown key(s) in config root: %s. Allowed: %s"
            % (
                ", ".join(sorted(unknown_root)),
                ", ".join(sorted(ALLOWED_KEYS | frozenset({"datasets"}))),
            )
        )

    base = dict(_DEFAULT_FLAT)
    base.update(shared)

    out: list[EfficientRunConfig] = []
    for i, ds in enumerate(dss):
        if not isinstance(ds, dict):
            raise TypeError("datasets[%d] must be an object, got %s" % (i, type(ds).__name__))
        bad = set(ds) - ALLOWED_KEYS
        if bad:
            raise ValueError(
                "Unknown key(s) in datasets[%d]: %s" % (i, ", ".join(sorted(bad)))
            )
        m = dict(base)
        m.update(ds)
        # User-provided keys = root-level shared keys ∪ per-entry overrides; needed for
        # legacy-vs-explicit ``algorithm``/``adaptive`` disambiguation.
        m["_user_keys"] = frozenset(set(shared) | set(ds))
        task = m.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                "datasets[%d]: merged config must have a non-empty task "
                "(set \"task\" on this entry or in shared defaults)" % i
            )
        out.append(_merged_dict_to_config(m))
    return out
