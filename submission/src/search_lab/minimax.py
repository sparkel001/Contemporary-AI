"""Full left-to-right Minimax and Alpha-Beta; repeated references count per visit."""

from math import inf
from .stats import SearchStats


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []

    @classmethod
    def from_dict(cls, data, maximizing=True):
        """Accept the PDF's tree_structure JSON without changing its schema."""
        if "tree_structure" in data:
            data = data["tree_structure"]
        if "is_maximizing_player" in data and data["is_maximizing_player"] != maximizing:
            raise ValueError("players must alternate from MAX root")
        return cls(data.get("value"), [cls.from_dict(c, not maximizing)
                                      for c in data.get("children", [])])


def _search(node, maximizing, alpha, beta, pruning, stats):
    stats.visited += 1
    if not node.children:
        if node.value is None:
            raise ValueError("leaf must have a value")
        stats.leaf_evaluations += 1
        return node.value
    stats.expanded += 1
    best = -inf if maximizing else inf
    for i, child in enumerate(node.children):
        stats.generated += 1
        value = _search(child, not maximizing, alpha, beta, pruning, stats)
        best = max(best, value) if maximizing else min(best, value)
        if maximizing:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
        remaining = len(node.children) - i - 1
        if pruning and alpha >= beta and remaining:
            stats.cutoffs += 1
            stats.pruned_branches += remaining
            break
    return best


def minimax(root, stats=None):
    return _search(root, True, -inf, inf, False, stats if stats is not None else SearchStats())


def alpha_beta(root, stats=None):
    return _search(root, True, -inf, inf, True, stats if stats is not None else SearchStats())
