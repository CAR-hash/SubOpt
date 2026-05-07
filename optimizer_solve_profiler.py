"""
Wrap ``LazyPlainOptimizer`` / ``LazySlicingOptimizer`` instances to time each ``solve()`` call.

Used by :meth:`efficient_bfs.EfficientBFS.get_optimizer` when ``opt_solve_log_path`` is set.
"""
from __future__ import annotations

import time
from typing import Any, Callable


class TimedOptimizerProxy:
    """Delegates ``build`` / ``update_base`` / ``solve``; records duration of ``solve``."""

    __slots__ = ("_inner", "_record")

    def __init__(self, inner: Any, record: Callable[[str, float], None]):
        self._inner = inner
        self._record = record

    def build(self, *args, **kwargs):
        return self._inner.build(*args, **kwargs)

    def update_base(self, *args, **kwargs):
        return self._inner.update_base(*args, **kwargs)

    def solve(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._inner.solve(*args, **kwargs)
        finally:
            self._record(type(self._inner).__name__, time.perf_counter() - t0)
