"""
Unified run-log format for ``compute_efficient.py``.

All three algorithm families (``EfficientBFS`` / ``AdaptiveEfficientBFS``,
``BFSTC``, ``EfficientBranchAndBound``) emit the **same set of events** so a single
parser can monitor any of them. Each event is one line, formatted as logfmt::

    [EVENT] key1=value1 key2=value2 ...

Conventions
~~~~~~~~~~~

- ``EVENT`` is an UPPER_SNAKE_CASE identifier (e.g. ``RUN_START``).
- Keys are ``lower_snake_case``. Values are formatted via :func:`_fmt_value`
  (numbers without trailing whitespace, strings without spaces or with quotes).
- Numerical values that may be very small / very large are formatted with
  :func:`format_number` (six significant digits when possible).
- Lines are flushed to stdout as soon as they are emitted; the ``TeeLogger``
  decorator in ``compute_efficient.py`` mirrors stdout to a file on disk.

Standard events
~~~~~~~~~~~~~~~

* ``RUN_START`` — emitted once per ``(seed, budget, strategy_or_algo, heuristic)``
  combination, just before ``alg.optimize()``. Carries the run identity.
* ``RUN_END`` — emitted once at the end of ``alg.optimize()``. Carries final
  ``status`` (``OK`` | ``TLE``), ``f_s``, ``c_s``, ``nodes``, ``time_s``.
* ``GREEDY_DONE`` — emitted at the end of the bootstrap greedy fill (root
  greedy_add). Carries ``f_s`` and ``set_size``.
* ``INCUMBENT`` — emitted whenever the global lower bound improves. Carries
  the new ``f_s``, the previous ``prev`` value, and ``set_size``.
* ``NODE_POP`` — emitted each time a node is popped from the open list / stack.
  Carries ``node`` (1-based index), ``open`` (heap / stack size after pop),
  ``s_size``, ``cost``, ``budget``, ``ub`` (best upper bound known for this node).
* ``NODE_BRANCH`` — emitted when a popped node is expanded into children.
  Carries ``children`` and ``strategy`` (use ``"none"`` for algorithms that
  don't have a branching dimension).
* ``NODE_PRUNE`` — emitted when a node is killed by an upper-bound test.
  Carries ``reason`` (e.g. ``alpha_lb``, ``edge``, ``empty``), ``ub``, and ``lb``.
* ``TIMER`` — wall-clock span for a hot region. Carries ``label`` and ``seconds``.

The minimal contract is: every algorithm emits ``RUN_START``, ``RUN_END``, and at
least one of the node events while running. Internal high-frequency events
(every ``[POP]`` line) only fire when ``runlog.verbose`` is true.

Verbosity
~~~~~~~~~

A logger has two knobs:

* ``enabled`` — when ``False``, all events are dropped (used by tests).
* ``verbose`` — when ``True``, ``NODE_POP`` / ``NODE_BRANCH`` / ``NODE_PRUNE`` /
  ``TIMER`` events at every single node are printed. When ``False`` (the default
  for production runs), only ``RUN_START`` / ``RUN_END`` / ``GREEDY_DONE`` /
  ``INCUMBENT`` are printed; the high-frequency node events are skipped.

Set ``runlog.verbose = True`` only when actively debugging an algorithm.
"""
from __future__ import annotations

import sys
from typing import Any, Mapping, Optional, TextIO

# Event types tagged as "high-frequency": only emitted when runlog.verbose is True.
# Every other event is always emitted as long as the logger is enabled.
_HIGH_FREQUENCY_EVENTS = frozenset({
    "NODE_POP",
    "NODE_BRANCH",
    "NODE_PRUNE",
    "NODE_PUSH",
    "TIMER",
})


def fmt_set(s) -> str:
    """
    Canonical, parser-friendly serialization of a node's ``s`` set.

    Accepts any iterable of ints, returns ``"[1,2,3]"`` (sorted, no spaces) so the
    token survives logfmt without quoting and is round-trippable via :func:`parse_set`.
    Used by ``s=`` / ``parent_s=`` fields on tree-mode events.
    """
    if s is None:
        return "[]"
    return "[" + ",".join(str(x) for x in sorted(s)) + "]"


def format_number(v: Any) -> str:
    """Format a number for logfmt output (no spaces, ~6 sig digits)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        # Compact form: if integral and small, use .1f; else %g.
        if abs(v) < 1e15 and v == int(v):
            return f"{v:.1f}"
        return f"{v:.6g}"
    return str(v)


def _fmt_value(v: Any) -> str:
    """logfmt serialization of one value (numbers, strings, bool, None)."""
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return format_number(v)
    s = str(v)
    if not s:
        return '""'
    # Quote if the value would clash with the key=value separator.
    if any(ch.isspace() for ch in s) or "=" in s or '"' in s:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def format_event(event: str, fields: Mapping[str, Any]) -> str:
    """Render ``[EVENT] k1=v1 k2=v2 ...``. Keys are emitted in insertion order."""
    parts = [f"[{event}]"]
    for k, v in fields.items():
        parts.append(f"{k}={_fmt_value(v)}")
    return " ".join(parts)


class RunLogger:
    """
    Emits unified run-log events to ``stream`` (default ``sys.stdout``).

    Construct with run-level context (``algorithm``, ``task``, ``seed``, ``budget``,
    ``heuristic``, ``alpha``) and the logger will silently include them as ``run_id``
    on every emitted line so downstream parsers can correlate events across files.
    """

    __slots__ = ("stream", "enabled", "verbose", "_run_id", "_node_id_counter")

    def __init__(
        self,
        *,
        stream: Optional[TextIO] = None,
        enabled: bool = True,
        verbose: bool = False,
        run_id: Optional[str] = None,
    ):
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = enabled
        self.verbose = verbose
        self._run_id = run_id
        self._node_id_counter = 0

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    def set_run_id(self, run_id: Optional[str]) -> None:
        self._run_id = run_id

    def event(self, event: str, **fields: Any) -> None:
        """
        Emit one event. ``fields`` are appended in iteration order.

        ``run_id`` is auto-injected (when set) right after the event tag, so the
        resulting line is ``[EVENT] run_id=... <user fields>``.
        """
        if not self.enabled:
            return
        if (event in _HIGH_FREQUENCY_EVENTS) and (not self.verbose):
            return
        # Compose run_id + user fields, preserving insertion order of user fields.
        merged: dict[str, Any] = {}
        if self._run_id is not None:
            merged["run_id"] = self._run_id
        merged.update(fields)
        line = format_event(event, merged)
        self.stream.write(line + "\n")
        try:
            self.stream.flush()
        except Exception:
            # Best effort: some streams (e.g. test buffers) may not support flush.
            pass

    # --- Convenience wrappers (keep callsites short and uniform) ---

    def run_start(
        self,
        *,
        algorithm: str,
        task: str,
        seed: int,
        budget: float,
        heuristic: str,
        alpha: float,
        strategy: str = "none",
        **extra: Any,
    ) -> None:
        self.event(
            "RUN_START",
            algorithm=algorithm,
            task=task,
            seed=seed,
            budget=budget,
            heuristic=heuristic,
            alpha=alpha,
            strategy=strategy,
            **extra,
        )

    def run_end(
        self,
        *,
        status: str,
        f_s: float,
        c_s: float,
        nodes: int,
        time_s: float,
        **extra: Any,
    ) -> None:
        self.event(
            "RUN_END",
            status=status,
            f_s=f_s,
            c_s=c_s,
            nodes=nodes,
            time_s=time_s,
            **extra,
        )

    def greedy_done(self, *, f_s: float, set_size: int, **extra: Any) -> None:
        self.event("GREEDY_DONE", f_s=f_s, set_size=set_size, **extra)

    def incumbent(self, *, f_s: float, prev: float, set_size: int, **extra: Any) -> None:
        self.event("INCUMBENT", f_s=f_s, prev=prev, set_size=set_size, **extra)

    def node_pop(
        self,
        *,
        node: int,
        open: int,
        s_size: int,
        cost: float,
        budget: float,
        ub: float,
        **extra: Any,
    ) -> None:
        self.event(
            "NODE_POP",
            node=node,
            open=open,
            s_size=s_size,
            cost=cost,
            budget=budget,
            ub=ub,
            **extra,
        )

    def node_branch(self, *, children: int, strategy: str = "none", **extra: Any) -> None:
        self.event("NODE_BRANCH", children=children, strategy=strategy, **extra)

    def node_prune(self, *, reason: str, ub: float, lb: float, **extra: Any) -> None:
        self.event("NODE_PRUNE", reason=reason, ub=ub, lb=lb, **extra)

    def node_push(
        self,
        *,
        parent_id: int,
        id: int,
        s: str,
        ub: float,
        status: str = "pushed",
        **extra: Any,
    ) -> None:
        """
        Emit one tree-structure event per child created by the search.

        ``status`` is ``pushed`` (made it to the open list) or ``pruned_pre`` (rejected
        by the pre-push UB / alpha-LB test before ever entering the open list).
        ``parent_id`` is ``0`` for the root's direct children. ``s`` is the canonical
        :func:`fmt_set` rendering of the new node's set so a parser can reconstruct
        the tree (note: a child can share its set with its parent — see
        ``EfficientBFS.branching_lazy_binary`` 's right branch — so ``id`` /
        ``parent_id`` are the authoritative identifiers, not ``s`` itself).
        """
        self.event(
            "NODE_PUSH",
            parent_id=parent_id,
            id=id,
            s=s,
            ub=ub,
            status=status,
            **extra,
        )

    def timer(self, *, label: str, seconds: float, **extra: Any) -> None:
        self.event("TIMER", label=label, seconds=seconds, **extra)

    def next_node_id(self) -> int:
        """
        Allocate a fresh, monotonically increasing node id for tree-mode events.

        The root is conventionally ``0``; the first child handed out is ``1``. Each
        ``RunLogger`` instance keeps its own counter so two concurrent runs cannot
        collide on ids.
        """
        self._node_id_counter += 1
        return self._node_id_counter


class NullRunLogger(RunLogger):
    """A logger that drops every event; safe default for unit tests / library use."""

    def __init__(self) -> None:
        super().__init__(enabled=False)


def make_run_id(
    *,
    algorithm: str,
    task: str,
    seed: int,
    budget: float,
    heuristic: str,
) -> str:
    """Stable run identifier used by ``RunLogger`` so events can be correlated."""
    return f"{algorithm}|{task}|seed={seed}|budget={budget}|h={heuristic}"
