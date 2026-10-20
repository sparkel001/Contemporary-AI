"""Problem 4: shared traversable map, explicit immutable game state."""

from dataclasses import dataclass, replace
from math import inf
from .astar import astar, manhattan
from .stats import SearchStats


def distance_matrix(grid, positions, stats=None):
    """Return (labels, 6x6 matrix). Compute every ordered pair with A*."""
    labels = list(positions)
    costs = [[None if v == 'X' else 1 if v in positions else v for v in row]
             for row in grid]
    minimum = min(v for row in costs for v in row if v is not None)
    matrix = []
    for source in labels:
        row = []
        for target in labels:
            goal = positions[target]
            _, cost = astar(positions[source], goal, costs,
                            lambda p: minimum * manhattan(p, goal), stats)
            row.append(cost)
        matrix.append(row)
    return labels, matrix


@dataclass(frozen=True)
class GameState:
    position_a: str = 'A'
    position_b: str = 'B'
    remaining: tuple[str, ...] = ('T1', 'T2', 'T3', 'T4')
    score_a: int = 0
    score_b: int = 0
    turn: str = 'A'

    @property
    def utility(self):
        return self.score_a - self.score_b


class TreasureGame:
    def __init__(self, labels, matrix, values):
        if len(set(labels)) != len(labels) or len(matrix) != len(labels):
            raise ValueError('invalid distance matrix')
        if any(len(row) != len(labels) for row in matrix):
            raise ValueError('invalid distance matrix')
        self.distances = {(a, b): matrix[i][j] for i, a in enumerate(labels)
                          for j, b in enumerate(labels)}
        self.values = dict(values)

    def move(self, state, target):
        if state.turn not in ('A', 'B') or target not in state.remaining:
            raise ValueError('illegal move')
        position = state.position_a if state.turn == 'A' else state.position_b
        cost = self.distances[position, target]
        if cost < 0:
            raise ValueError('unreachable target')
        gain = self.values[target] - cost
        remaining = tuple(t for t in state.remaining if t != target)
        if state.turn == 'A':
            return replace(state, position_a=target, score_a=state.score_a + gain,
                           remaining=remaining, turn='B')
        return replace(state, position_b=target, score_b=state.score_b + gain,
                       remaining=remaining, turn='A')

    def solve(self, depth, algorithm='alpha_beta', state=None, stats=None):
        if type(depth) is not int or depth < 0:
            raise ValueError('depth must be a nonnegative integer')
        if algorithm not in ('minimax', 'alpha_beta'):
            raise ValueError('unknown algorithm')
        state = state if state is not None else GameState()
        if state.turn not in ('A', 'B'):
            raise ValueError('turn must be A or B')
        s = stats if stats is not None else SearchStats()

        def search(current, h, alpha, beta):
            s.visited += 1
            if h == 0 or not current.remaining:
                s.leaf_evaluations += 1
                return None, current.utility
            maximizing = current.turn == 'A'
            best, target = (-inf if maximizing else inf), None
            s.expanded += 1
            for i, choice in enumerate(current.remaining):
                s.generated += 1
                _, value = search(self.move(current, choice), h - 1, alpha, beta)
                if target is None or (value > best if maximizing else value < best):
                    best, target = value, choice
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                remaining = len(current.remaining) - i - 1
                if algorithm == 'alpha_beta' and alpha >= beta and remaining:
                    s.cutoffs += 1
                    s.pruned_branches += remaining
                    break
            return target, best

        target, value = search(state, depth, -inf, inf)
        return {'best_target': target, 'best_value': value}
