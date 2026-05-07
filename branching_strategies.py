"""
Pluggable branching strategies for EfficientBFS (Strategy pattern).

Each strategy wraps the corresponding method on the solver; the main loop
selects an implementation via ``BRANCHING_REGISTRY`` using ``branching_strategy`` name.
"""
from __future__ import annotations

from typing import Any, Protocol


class BranchingStrategy(Protocol):
    def branch(self, solver: Any, node: Any, heuristic_sequence: Any, f_local: float) -> int:
        """Expand ``node``; return the number of children pushed (open-list delta)."""


class TraditionalBranching:
    __slots__ = ()

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.branching(node, heuristic_sequence, f_local=f_local)


class FullBabBranching:
    __slots__ = ()

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.branching(node, heuristic_sequence)


class DensityGapBranching:
    __slots__ = ("tau", "max_k")

    def __init__(self, tau=0.8, max_k=4):
        self.tau = tau
        self.max_k = max_k

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.recursive_branching(node, heuristic_sequence, tau=self.tau, max_k=self.max_k)


class VolumeBiasedBranching:
    __slots__ = ("top_n",)

    def __init__(self, top_n=5):
        self.top_n = top_n

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.branching_volume_biased(node, heuristic_sequence, top_n=self.top_n)


class ProbingBranching:
    __slots__ = ()

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.branching_probing(node, heuristic_sequence)


class ClusterCollapseBranching:
    __slots__ = ("tau",)

    def __init__(self, tau=0.85):
        self.tau = tau

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.branching_cluster_collapse(node, heuristic_sequence, tau=self.tau)


class NaiveBranching:
    __slots__ = ()

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.branching_naive(node, heuristic_sequence)


class BinaryCollapseBranching:
    __slots__ = ("tau",)

    def __init__(self, tau=0.85):
        self.tau = tau

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.branching_binary_collapse(
            node, heuristic_sequence, tau=self.tau, f_local=f_local
        )


class LazyBinaryBranching:
    __slots__ = ()

    def branch(self, solver, node, heuristic_sequence, f_local):
        return solver.branching_lazy_binary(node, heuristic_sequence)


BRANCHING_REGISTRY: dict[str, BranchingStrategy] = {
    "traditional": TraditionalBranching(),
    "fullbab": FullBabBranching(),
    "density_gap": DensityGapBranching(tau=0.8, max_k=4),
    "volume_biased": VolumeBiasedBranching(top_n=5),
    "probing": ProbingBranching(),
    "cluster_collapse": ClusterCollapseBranching(tau=0.85),
    "naive": NaiveBranching(),
    "binary_collapse": BinaryCollapseBranching(tau=0.85),
    "lazy_binary": LazyBinaryBranching(),
}


def get_branching_strategy(name: str) -> BranchingStrategy:
    try:
        return BRANCHING_REGISTRY[name]
    except KeyError as e:
        known = ", ".join(sorted(BRANCHING_REGISTRY))
        raise ValueError(f"Unknown branching_strategy {name!r}. Known: {known}") from e


def register_branching_strategy(name: str, strategy: BranchingStrategy) -> None:
    """Register or replace a strategy (e.g. experiments or monkey-patching)."""
    BRANCHING_REGISTRY[name] = strategy
