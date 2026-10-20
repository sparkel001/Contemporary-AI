"""Four-neighbour A*, with deterministic FIFO tie breaking and reopening."""

from heapq import heappop, heappush
from itertools import count
from math import inf

from .stats import SearchStats

DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def validate_grid(grid, start, end):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be a nonempty rectangle")
    for p in (start, end):
        if len(p) != 2 or any(type(v) is not int for v in p):
            raise ValueError("coordinates must contain two integers")
        if not (0 <= p[0] < len(grid) and 0 <= p[1] < len(grid[0])):
            raise ValueError("coordinate outside grid")


def astar(start, end, costs, heuristic, stats=None):
    """Return (path, cost); None cells are walls. Costs must be nonnegative."""
    validate_grid(costs, start, end)
    s = stats if stats is not None else SearchStats()
    if any(v is not None and (type(v) is not int or v < 0) for row in costs for v in row):
        raise ValueError("costs must be nonnegative integers or None")
    if costs[start[0]][start[1]] is None or costs[end[0]][end[1]] is None:
        return [], -1
    order = count()
    queue = [(heuristic(start), next(order), 0, start)]
    distance, parent = {start: 0}, {}
    s.generated += 1
    s.frontier_peak = max(s.frontier_peak, 1)
    while queue:
        _, _, g, p = heappop(queue)
        if g != distance[p]:
            s.stale_pops += 1
            continue
        s.visited += 1
        if p == end:
            path = [p]
            while p in parent:
                p = parent[p]
                path.append(p)
            return path[::-1], g
        s.expanded += 1
        for dr, dc in DIRECTIONS:
            q = p[0] + dr, p[1] + dc
            if not (0 <= q[0] < len(costs) and 0 <= q[1] < len(costs[0])):
                continue
            weight = costs[q[0]][q[1]]
            if weight is None:
                continue
            candidate = g + weight
            if candidate < distance.get(q, inf):
                distance[q], parent[q] = candidate, p
                heappush(queue, (candidate + heuristic(q), next(order), candidate, q))
                s.generated += 1
        s.frontier_peak = max(s.frontier_peak, len(queue))
    return [], -1


def maze_path(start, end, grid, heuristic="manhattan", stats=None):
    """Problem 1: official return type is list[tuple[int, int]]."""
    validate_grid(grid, start, end)
    if heuristic not in ("zero", "manhattan"):
        raise ValueError("heuristic must be zero or manhattan")
    if any(v not in ("S", "E", ".", "X") for row in grid for v in row):
        raise ValueError("unknown maze cell")
    costs = [[None if v == "X" else 1 for v in row] for row in grid]
    h = (lambda p: 0) if heuristic == "zero" else (lambda p: manhattan(p, end))
    return astar(start, end, costs, h, stats)[0]


def terrain_costs(grid):
    """S uses entry cost 1; start is not charged and an optimal path need not revisit it."""
    costs = []
    for row in grid:
        converted = []
        for v in row:
            if v == "X":
                converted.append(None)
            elif v in ("S", "E"):
                converted.append(1 if v == "S" else 0)
            elif type(v) is int and v >= 0:
                converted.append(v)
            elif isinstance(v, str) and v.isascii() and v.isdigit():
                converted.append(int(v))
            else:
                raise ValueError("terrain must be a nonnegative integer, S, E, or X")
        costs.append(converted)
    return costs


def terrain_heuristic(costs, start, end):
    """Lower bound: at least d-1 non-goal entries cost >= m, then pay cE."""
    weights = [v for r, row in enumerate(costs) for c, v in enumerate(row)
               if v is not None and (r, c) != end]
    minimum = min(weights, default=0)
    final_cost = costs[end[0]][end[1]] or 0

    def h(p):
        if p == end:
            return 0
        return minimum * max(manhattan(p, end) - 1, 0) + final_cost

    return h


def terrain_cost(start, end, grid, heuristic="scaled", stats=None):
    """Problem 2: return minimum entry cost, or -1 when unreachable."""
    validate_grid(grid, start, end)
    if heuristic not in ("zero", "scaled"):
        raise ValueError("heuristic must be zero or scaled")
    costs = terrain_costs(grid)
    h = (lambda p: 0) if heuristic == "zero" else terrain_heuristic(costs, start, end)
    return astar(start, end, costs, h, stats)[1]
