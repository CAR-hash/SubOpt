"""
Observers for EfficientBFS logging and metrics (Observer pattern).

Attach one implementation to ``EfficientBFS.search_observer`` (or a
:class:`CompositeSearchObserver` list). The solver calls ``log`` / ``metric``
via ``_obs_log`` / ``_obs_metric`` / ``_obs_timer``.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SearchObserver(Protocol):
    def log(self, event: str, message: str = "", **fields: Any) -> None:
        """Structured or free-form log line (``message`` is printed verbatim when set)."""

    def metric(self, name: str, value: float, **labels: Any) -> None:
        """Numeric measurement (durations, counts, bounds); use ``labels`` for dimensions."""


class NullSearchObserver:
    __slots__ = ()

    def log(self, event: str, message: str = "", **fields: Any) -> None:
        pass

    def metric(self, name: str, value: float, **labels: Any) -> None:
        pass


class PrintingSearchObserver:
    """Default: print ``log`` messages as-is; print compact ``metric`` lines."""

    __slots__ = ()

    def log(self, event: str, message: str = "", **fields: Any) -> None:
        if message:
            print(message)
        elif fields:
            print(f"[{event}]", fields)

    def metric(self, name: str, value: float, **labels: Any) -> None:
        rest = " ".join(f"{k}={v}" for k, v in labels.items())
        line = f"[metric] {name}={value:.9f}"
        if rest:
            line = f"{line} {rest}"
        print(line)


class CompositeSearchObserver:
    __slots__ = ("_observers",)

    def __init__(self, observers: list[SearchObserver]):
        self._observers = list(observers)

    def log(self, event: str, message: str = "", **fields: Any) -> None:
        for o in self._observers:
            o.log(event, message, **fields)

    def metric(self, name: str, value: float, **labels: Any) -> None:
        for o in self._observers:
            o.metric(name, value, **labels)
