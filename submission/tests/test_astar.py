from collections import deque
from random import Random
import pytest

from search_lab.astar import (maze_path, terrain_cost, terrain_costs,
                              terrain_heuristic, manhattan)
from search_lab.examples import MAZE, TERRAIN
from search_lab.stats import SearchStats


def reference_cost(costs, start, end):
    """Independent Bellman-Ford oracle (no A* heap or heuristic)."""
    nodes = [(r, c) for r, row in enumerate(costs) for c, v in enumerate(row) if v is not None]
    distance = {p: float('inf') for p in nodes}
    distance[start] = 0
    for _ in range(len(nodes) - 1):
        changed = False
        for p in nodes:
            for q in nodes:
                if manhattan(p, q) == 1 and distance[p] + costs[q[0]][q[1]] < distance[q]:
                    distance[q] = distance[p] + costs[q[0]][q[1]]
                    changed = True
        if not changed:
            break
    value = distance.get(end, float('inf'))
    return -1 if value == float('inf') else value


@pytest.mark.parametrize('h', ['zero', 'manhattan'])
def test_official_maze(h):
    stats = SearchStats()
    path = maze_path((0, 0), (3, 4), MAZE, h, stats)
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    assert stats.visited == stats.expanded + 1
    assert stats.generated >= stats.visited


@pytest.mark.parametrize('h', ['zero', 'scaled'])
def test_official_terrain(h):
    assert terrain_cost((0, 0), (4, 4), TERRAIN, h) == 7


@pytest.mark.parametrize('solver, grid, expected', [
    (maze_path, [['S', 'X', 'E']], []),
    (maze_path, [['S', 'X'], ['X', 'E']], []),
    (terrain_cost, [['S', 'X', 'E']], -1),
    (maze_path, [['X', 'E']], []),
    (terrain_cost, [['S', 'X']], -1),
])
def test_unreachable(solver, grid, expected):
    assert solver((0, 0), (len(grid)-1, len(grid[0])-1), grid) == expected


def test_single_cell_and_zero_final_step():
    assert maze_path((0, 0), (0, 0), [['S']]) == [(0, 0)]
    assert terrain_cost((0, 0), (0, 0), [['S']]) == 0
    assert terrain_cost((0, 0), (0, 1), [['S', 'E']]) == 0
    assert manhattan((0, 0), (0, 1)) > 0  # raw Manhattan overestimates
    assert terrain_cost((0, 0), (0, 3), [['S', '0', 0, 'E']]) == 0


def test_weighted_detour_and_entry_direction():
    assert terrain_cost((0, 0), (0, 2), [['S', 99, 'E'], [1, 1, 1]]) == 3
    assert terrain_cost((0, 0), (0, 1), [[2, 9]]) == 9
    assert terrain_cost((0, 1), (0, 0), [[2, 9]]) == 2


@pytest.mark.parametrize('grid', [[], [[]], [['S'], ['.', 'E']]])
def test_invalid_shape(grid):
    with pytest.raises(ValueError):
        maze_path((0, 0), (0, 0), grid)


@pytest.mark.parametrize('grid', [[['S', -1, 'E']], [['S', 1.5, 'E']], [['S', 'bad', 'E']]])
def test_invalid_weight(grid):
    with pytest.raises(ValueError):
        terrain_cost((0, 0), (0, 2), grid)


def test_invalid_coordinates_and_heuristic():
    with pytest.raises(ValueError):
        maze_path((-1, 0), (0, 0), [['S']])
    with pytest.raises(ValueError):
        terrain_cost((0, 0), (0, 0), [['S']], 'manhattan')


def test_seeded_grids_and_admissibility():
    rng = Random(20261019)
    for _ in range(35):
        grid = [[rng.choice([0, 1, 2, 8, 'X']) for _ in range(4)] for _ in range(4)]
        grid[0][0], grid[3][3] = 'S', 'E'
        costs = terrain_costs(grid)
        expected = reference_cost(costs, (0, 0), (3, 3))
        for mode in ('zero', 'scaled'):
            assert terrain_cost((0, 0), (3, 3), grid, mode) == expected
        h = terrain_heuristic(costs, (0, 0), (3, 3))
        for r in range(4):
            for c in range(4):
                p = r, c
                if costs[r][c] is None:
                    continue
                exact = reference_cost(costs, p, (3, 3))
                if exact >= 0:
                    assert 0 <= h(p) <= exact
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    q = r + dr, c + dc
                    if 0 <= q[0] < 4 and 0 <= q[1] < 4 and costs[q[0]][q[1]] is not None:
                        assert h(p) <= costs[q[0]][q[1]] + h(q)


def test_seeded_maze_against_bfs():
    rng = Random(7)
    for _ in range(30):
        grid = [[rng.choice(['.', '.', 'X']) for _ in range(5)] for _ in range(5)]
        grid[0][0], grid[4][4] = 'S', 'E'
        queue, distances = deque([(0, 0)]), {(0, 0): 0}
        while queue:
            p = queue.popleft()
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                q = p[0]+dr, p[1]+dc
                if 0 <= q[0] < 5 and 0 <= q[1] < 5 and grid[q[0]][q[1]] != 'X' and q not in distances:
                    distances[q] = distances[p]+1
                    queue.append(q)
        for mode in ('zero', 'manhattan'):
            path = maze_path((0, 0), (4, 4), grid, mode)
            assert len(path)-1 == distances.get((4, 4), -1)
            if path:
                assert path[0] == (0, 0) and path[-1] == (4, 4)
                assert all(manhattan(a, b) == 1 for a, b in zip(path, path[1:]))
                assert all(grid[r][c] != 'X' for r, c in path)


def test_positive_lower_bound_on_every_official_cell():
    costs = terrain_costs(TERRAIN)
    h = terrain_heuristic(costs, (0, 0), (4, 4))
    assert h((0, 0)) == 7
    assert h((3, 4)) == h((4, 4)) == 0
    for r in range(5):
        for c in range(5):
            assert h((r, c)) <= reference_cost(costs, (r, c), (4, 4))
