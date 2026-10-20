"""Measured extensions; BLOCK 01 public interfaces and rules stay unchanged."""
from collections import deque
from dataclasses import dataclass
from math import inf
from time import perf_counter_ns

from .astar import DIRECTIONS, manhattan, terrain_heuristic
from .stats import SearchStats
from .treasure import GameState


@dataclass
class BenchStats(SearchStats):
    skipped_nodes: int = 0
    cache_probes: int = 0
    cache_hits: int = 0
    cache_stores: int = 0
    ordering_nodes: int = 0
    ordering_leaves: int = 0
    ordering_ns: int = 0
    tie_researches: int = 0


def dijkstra_reference(costs, start, end):
    """Independent O(V^2) Dijkstra, no heap or production A* code."""
    pending = {(r, c) for r, row in enumerate(costs) for c, v in enumerate(row) if v is not None}
    distances = {start: 0}
    if start not in pending or end not in pending:
        return -1
    while pending:
        p = min(pending, key=lambda q: (distances.get(q, inf), q))
        distance = distances.get(p, inf)
        if distance == inf:
            break
        if p == end:
            return distance
        pending.remove(p)
        for q in ((p[0]-1, p[1]), (p[0]+1, p[1]), (p[0], p[1]-1), (p[0], p[1]+1)):
            if q in pending:
                distances[q] = min(distances.get(q, inf), distance + costs[q[0]][q[1]])
    return -1


def build_heuristic(costs, start, end, mode):
    """Topology relaxation: BFS step distance with all non-goal weights set to m.

    For reachable p, h(p)=m*(steps(p,end)-1)+c(end). This is a lower bound
    because each real non-goal entry costs at least m. Walls are preserved.
    Unreachable components use zero, never infinity, to preserve A* behavior.
    Returns callable and explicit preprocessing node count.
    """
    if mode == 'zero':
        return lambda p: 0, 0
    if mode == 'manhattan':
        return terrain_heuristic(costs, start, end), 0
    if mode != 'topology':
        raise ValueError('unknown heuristic')
    minimum = min((v for r, row in enumerate(costs) for c, v in enumerate(row)
                   if v is not None and (r, c) != end), default=0)
    steps = {}
    if costs[end[0]][end[1]] is not None:
        steps[end] = 0
    queue = deque(steps)
    while queue:
        p = queue.popleft()
        for dr, dc in DIRECTIONS:
            q = p[0]+dr, p[1]+dc
            if (0 <= q[0] < len(costs) and 0 <= q[1] < len(costs[0])
                    and costs[q[0]][q[1]] is not None and q not in steps):
                steps[q] = steps[p] + 1
                queue.append(q)

    def h(p):
        if p == end or p not in steps:
            return 0
        return (steps[p]-1) * minimum + costs[end[0]][end[1]]

    return h, len(steps)


def tree_sizes(root):
    """Structural instrumentation only; sizes count repeated references per path."""
    sizes = {}

    def walk(node):
        size = 1 + sum(walk(child) for child in node.children)
        sizes[id(node)] = size
        return size

    walk(root)
    return sizes


def ordered_tree_search(root, mode='left', stats=None, sizes=None):
    """Order by first-leaf sample, never by an oracle Minimax value.

    Sample traversals and leaf reads are separately counted, included in time.
    """
    if mode not in ('minimax', 'left', 'right', 'ordered'):
        raise ValueError('unknown order')
    s = stats if stats is not None else BenchStats()
    sizes = tree_sizes(root) if sizes is None else sizes

    def sample(node):
        while True:
            s.ordering_nodes += 1
            if not node.children:
                s.ordering_leaves += 1
                return node.value
            node = node.children[0]

    def search(node, maximizing, alpha, beta):
        s.visited += 1
        if not node.children:
            if node.value is None:
                raise ValueError('leaf must have a value')
            s.leaf_evaluations += 1
            return node.value
        children = node.children
        if mode == 'right':
            children = children[::-1]
        elif mode == 'ordered':
            before = perf_counter_ns()
            children = sorted(children, key=sample, reverse=maximizing)
            s.ordering_ns += perf_counter_ns() - before
        s.expanded += 1
        best = -inf if maximizing else inf
        for i, child in enumerate(children):
            s.generated += 1
            value = search(child, not maximizing, alpha, beta)
            best = max(best, value) if maximizing else min(best, value)
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            remaining = children[i+1:]
            if mode != 'minimax' and alpha >= beta and remaining:
                s.cutoffs += 1
                s.pruned_branches += len(remaining)
                s.skipped_nodes += sum(sizes[id(c)] for c in remaining)
                break
        return best

    return search(root, True, -inf, inf)


@dataclass(frozen=True)
class TTEntry:
    value: int
    flag: str


def subtree_nodes(remaining, depth):
    total = term = 1
    for i in range(min(remaining, depth)):
        term *= remaining - i
        total += term
    return total


class TreasureSearch:
    """Fail-soft Alpha-Beta with window-aware transposition bounds.

    Key includes full immutable state (including both scores), remaining depth.
    Tables belong to one solver/game and are cold for each benchmark repeat.
    """
    def __init__(self, game, order='left', cache=False, pruning=True):
        if order not in ('left', 'right', 'ordered'):
            raise ValueError('unknown order')
        self.game, self.order, self.cache, self.pruning = game, order, cache, pruning
        self.table = {}

    def choices(self, state, stats):
        choices = list(state.remaining)
        if self.order == 'right':
            choices.reverse()
        elif self.order == 'ordered':
            before = perf_counter_ns()
            position = state.position_a if state.turn == 'A' else state.position_b
            # Both players prefer high own immediate gain; no extra move() calls.
            choices.sort(key=lambda t: self.game.values[t]-self.game.distances[position, t], reverse=True)
            stats.ordering_nodes += len(choices)
            stats.ordering_ns += perf_counter_ns() - before
        return choices

    def value(self, state, depth, alpha=-inf, beta=inf, stats=None):
        """Narrow-window result may be a bound, not necessarily an exact value."""
        if type(depth) is not int or depth < 0:
            raise ValueError('depth must be a nonnegative integer')
        if state.turn not in ('A', 'B'):
            raise ValueError('turn must be A or B')
        if not alpha < beta:
            raise ValueError('alpha must be smaller than beta')
        return self._value(state, depth, alpha, beta, stats)

    def _value(self, state, depth, alpha=-inf, beta=inf, stats=None):
        """Recursive worker; the public entry points validate caller inputs."""
        s = stats if stats is not None else BenchStats()
        s.visited += 1
        alpha_original, beta_original = alpha, beta
        key = (state, depth)
        if self.cache:
            s.cache_probes += 1
            entry = self.table.get(key)
            if entry is not None:
                s.cache_hits += 1
                if entry.flag == 'EXACT':
                    return entry.value
                if entry.flag == 'LOWER_BOUND':
                    alpha = max(alpha, entry.value)
                else:
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value
        if depth == 0 or not state.remaining:
            s.leaf_evaluations += 1
            best = state.utility
            flag = 'EXACT'
        else:
            s.expanded += 1
            maximizing = state.turn == 'A'
            best = -inf if maximizing else inf
            choices = self.choices(state, s)
            for i, target in enumerate(choices):
                s.generated += 1
                child = self.game.move(state, target)
                value = self._value(child, depth-1, alpha, beta, s)
                best = max(best, value) if maximizing else min(best, value)
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                remaining = len(choices)-i-1
                if self.pruning and alpha >= beta and remaining:
                    s.cutoffs += 1
                    s.pruned_branches += remaining
                    s.skipped_nodes += remaining * subtree_nodes(len(state.remaining)-1, depth-1)
                    break
            flag = ('UPPER_BOUND' if best <= alpha_original else
                    'LOWER_BOUND' if best >= beta_original else 'EXACT')
        if self.cache:
            self.table[key] = TTEntry(best, flag)
            s.cache_stores += 1
        return best

    def solve(self, depth, state=None, stats=None):
        if type(depth) is not int or depth < 0:
            raise ValueError('depth must be a nonnegative integer')
        state = state if state is not None else GameState()
        if state.turn not in ('A', 'B'):
            raise ValueError('turn must be A or B')
        s = stats if stats is not None else BenchStats()
        if not depth or not state.remaining:
            return {'best_target': None, 'best_value': self._value(state, depth, stats=s)}
        s.visited += 1
        s.expanded += 1
        maximizing = state.turn == 'A'
        best, target = (-inf if maximizing else inf), None
        alpha, beta = -inf, inf
        rank = {t: i for i, t in enumerate(state.remaining)}
        for choice in self.choices(state, s):
            child = self.game.move(state, choice)
            s.generated += 1
            value = self._value(child, depth-1, alpha, beta, s)
            # Equality from a narrow window may only be a bound. Re-search an
            # earlier canonical candidate before allowing it to replace a tie.
            if target is not None and value == best and rank[choice] < rank[target]:
                s.tie_researches += 1
                s.generated += 1
                value = self._value(child, depth-1, -inf, inf, s)
            if (target is None or (value > best if maximizing else value < best)
                    or (value == best and rank[choice] < rank[target])):
                best, target = value, choice
            if self.pruning:
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
        return {'best_target': target, 'best_value': best}
