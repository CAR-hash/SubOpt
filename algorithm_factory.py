"""
Algorithm factory for ``compute_efficient.py``.

Centralizes how a configured solver (one of :data:`SUPPORTED_ALGORITHMS`) is instantiated
and configured from an :class:`compute_efficient_config.EfficientRunConfig`. Handles the
differences between BFS-style algorithms (which support branching strategies, cascade
filtering, local search, etc.) and the branch-and-bound / BFSTC variants imported from
:mod:`filter_search`, which have a smaller configuration surface.

Public surface:
    * :data:`SUPPORTED_ALGORITHMS` - the canonical algorithm-name set.
    * :data:`ALGORITHMS_WITH_BRANCHING` - subset that consumes ``cfg.branching``.
    * :func:`build_algorithm` - construct + configure an algorithm instance for one run.
"""
from __future__ import annotations

from typing import FrozenSet

import efficient_bfs

# Lazy/optional import: ``filter_search`` is large; only require it when the user asks
# for one of the algorithms it owns.
_filter_search = None


def _get_filter_search():
    global _filter_search
    if _filter_search is None:
        import filter_search as _fs  # noqa: WPS433 - intentional lazy import
        _filter_search = _fs
    return _filter_search


SUPPORTED_ALGORITHMS: FrozenSet[str] = frozenset({
    "EfficientBFS",
    "AdaptiveEfficientBFS",
    "BFSTC",
    "EfficientBranchAndBound",
})

ALGORITHMS_WITH_BRANCHING: FrozenSet[str] = frozenset({
    "EfficientBFS",
    "AdaptiveEfficientBFS",
})


def _strip_plus(kind: str) -> str:
    """``'ub2+'`` → ``'ub2'``; pass-through otherwise."""
    return kind[:-1] if isinstance(kind, str) and kind.endswith("+") else kind


def _build_efficient_bfs(cfg, *, model, heuristic: str, strategy: str):
    """Build an ``EfficientBFS`` / ``AdaptiveEfficientBFS`` for one (heuristic, strategy)."""
    if cfg.algorithm == "AdaptiveEfficientBFS":
        alg = efficient_bfs.AdaptiveEfficientBFS(model)
        alg.adaptive_ratio = cfg.adaptive_ratio
    else:
        alg = efficient_bfs.EfficientBFS(model)

    alg.use_alpha = True
    alg.local_search = cfg.local_search
    alg.alpha = cfg.alpha
    alg.set_d(cfg.sorting)
    alg.configure_upper_bound(heuristic)
    alg.configure_aux_upper_bound(cfg.aux_heuristic)
    alg.inherit_bounds = not cfg.no_inherit
    if cfg.cascade:
        alg.use_cascade = True
    alg.branching_strategy = strategy
    alg.time_limit_seconds = float(cfg.time_limit_seconds)
    return alg


def _build_filter_search_algorithm(cfg, *, model, heuristic: str):
    """Build a ``BFSTC`` or ``EfficientBranchAndBound`` from :mod:`filter_search`."""
    fs = _get_filter_search()
    if cfg.algorithm == "BFSTC":
        alg = fs.BFSTC(model)
    elif cfg.algorithm == "EfficientBranchAndBound":
        alg = fs.EfficientBranchAndBound(model)
    else:  # pragma: no cover - guarded by build_algorithm()
        raise ValueError(f"unsupported filter-search algorithm: {cfg.algorithm!r}")

    # filter_search variants use the base ``set_h`` API and only know ``ub0`` / ``ub2``
    # (with the ``+`` variants collapsed). EBB additionally ignores the heuristic
    # entirely (its ``set_h`` is a no-op).
    alg.alpha = cfg.alpha
    alg.set_h(_strip_plus(heuristic))
    alg.time_limit = float(cfg.time_limit_seconds)
    return alg


def build_algorithm(cfg, *, model, heuristic: str, strategy: str = "n/a"):
    """
    Instantiate and configure the solver implied by ``cfg.algorithm``.

    Parameters
    ----------
    cfg :
        :class:`compute_efficient_config.EfficientRunConfig` for the current dataset.
    model :
        The task / model object (already seeded for the current run).
    heuristic :
        Upper-bound kind for this run (one of ``cfg.heuristics``).
    strategy :
        Branching strategy. Ignored when ``cfg.algorithm`` is not in
        :data:`ALGORITHMS_WITH_BRANCHING`; callers should pass ``"none"`` in that case.
    """
    if cfg.algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            "unsupported algorithm %r; expected one of %s"
            % (cfg.algorithm, ", ".join(sorted(SUPPORTED_ALGORITHMS)))
        )

    if cfg.algorithm in ALGORITHMS_WITH_BRANCHING:
        return _build_efficient_bfs(cfg, model=model, heuristic=heuristic, strategy=strategy)
    return _build_filter_search_algorithm(cfg, model=model, heuristic=heuristic)
