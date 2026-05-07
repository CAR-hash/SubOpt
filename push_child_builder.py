"""
Command builder for enqueueing BFS children on ``EfficientBFS.max_heap``.

Supports:

- **Evaluated push** — forwards to ``push_heap`` (runs ``h``, cascade, alpha checks).
- **Precomputed UB push** — forwards to ``push_heap_with_ub`` (no ``h`` call).

Use ``solver.push_child()`` for a fluent chain ending in ``execute()``.
"""
from __future__ import annotations

from typing import Any, List, Optional


class PushChildBuilder:
    __slots__ = (
        "_solver",
        "_mode",
        "_s",
        "_lbd_v",
        "_candidate",
        "_w",
        "_visited",
        "_first_child",
        "_heuristic_sequence",
        "_s_max_v",
        "_depth",
        "_forbidden_sets",
        "_f_local",
        "_pre_ub",
    )

    def __init__(self, solver: Any):
        self._solver = solver
        self._mode = "evaluated"
        self._s: Optional[List] = None
        self._lbd_v: Optional[float] = None
        self._candidate: Optional[List] = None
        self._w: Optional[float] = None
        self._visited = False
        self._first_child = False
        self._heuristic_sequence = None
        self._s_max_v: Optional[float] = None
        self._depth = 0
        self._forbidden_sets = None
        self._f_local = float("inf")
        self._pre_ub: Optional[float] = None

    def s(self, s):
        self._s = s
        return self

    def lbd_v(self, v: float):
        self._lbd_v = v
        return self

    def candidate(self, c):
        self._candidate = c
        return self

    def w(self, w):
        self._w = w
        return self

    def depth(self, d: int):
        self._depth = d
        return self

    def visited(self, value: bool = True):
        self._visited = value
        return self

    def first_child(self, value: bool = True):
        self._first_child = value
        return self

    def heuristic_sequence(self, hs):
        self._heuristic_sequence = hs
        return self

    def s_max_v(self, v: float):
        self._s_max_v = v
        return self

    def incumbent_lb_as_s_max_v(self):
        """Set ``s_max_v`` to ``g(s_max)`` on the solver (common pruning reference)."""
        self._s_max_v = self._solver.g(self._solver.s_max)
        return self

    def forbidden_sets(self, fs):
        self._forbidden_sets = fs
        return self

    def f_local(self, v: float):
        self._f_local = v
        return self

    def precomputed_ub(self, ub: float):
        """Switch to ``push_heap_with_ub``; call ``execute()`` after ``s``, ``candidate``, ``w``, ``depth``."""
        self._mode = "with_ub"
        self._pre_ub = ub
        return self

    def execute(self):
        if self._mode == "with_ub":
            if self._s is None or self._candidate is None or self._w is None or self._pre_ub is None:
                raise ValueError("precomputed_ub push requires s, candidate, w, depth, and precomputed_ub(...)")
            self._solver.push_heap_with_ub(
                self._s,
                self._candidate,
                self._w,
                self._depth,
                self._pre_ub,
                self._heuristic_sequence,
            )
            return None

        if self._s is None or self._lbd_v is None or self._candidate is None or self._w is None:
            raise ValueError("evaluated push requires s, lbd_v, candidate, w")
        s_max_v = self._s_max_v if self._s_max_v is not None else self._solver.g(self._solver.s_max)
        fs = self._forbidden_sets
        return self._solver.push_heap(
            self._s,
            self._lbd_v,
            visited=self._visited,
            first_child=self._first_child,
            heuristic_sequence=self._heuristic_sequence,
            candidate=self._candidate,
            w=self._w,
            s_max_v=s_max_v,
            depth=self._depth,
            forbidden_sets=fs,
            f_local=self._f_local,
        )
