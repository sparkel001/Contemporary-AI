from dataclasses import dataclass


@dataclass
class SearchStats:
    """Counters accumulate if reused; pass a fresh instance for each run."""

    visited: int = 0
    expanded: int = 0
    generated: int = 0
    stale_pops: int = 0
    frontier_peak: int = 0
    leaf_evaluations: int = 0
    cutoffs: int = 0
    pruned_branches: int = 0
