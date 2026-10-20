"""Correctness tests for experimental extensions, not timing assertions."""
from dataclasses import replace
from itertools import permutations
from math import inf
from pathlib import Path
from random import Random
import sys

import pytest

from search_lab.astar import astar, manhattan, terrain_costs
from search_lab.examples import TERRAIN, TREASURE_MAP, POSITIONS, VALUES, official_tree
from search_lab.minimax import TreeNode, minimax, alpha_beta
from search_lab.optimizations import (BenchStats, TreasureSearch, build_heuristic,
                                     dijkstra_reference, ordered_tree_search, tree_sizes)
from search_lab.stats import SearchStats
from search_lab.treasure import TreasureGame, GameState, distance_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_benchmarks import GAME_MODES, game_solver, make_tree, map_cases, decision_evidence, summarize


@pytest.fixture
def game():
    return TreasureGame(*distance_matrix(TREASURE_MAP, POSITIONS), VALUES)


@pytest.mark.parametrize('mode', ['zero', 'manhattan', 'topology'])
def test_admissible_consistent_and_optimal(mode):
    rng = Random(111)
    grids = [terrain_costs(TERRAIN), [[1, 0]], [[1, None, 0]]]
    for _ in range(15):
        grid = [[rng.choice([0, 1, 3, 9, None]) for _ in range(4)] for _ in range(4)]
        grid[0][0], grid[-1][-1] = 1, 0
        grids.append(grid)
    for grid in grids:
        start, end = (0, 0), (len(grid)-1, len(grid[0])-1)
        h, _ = build_heuristic(grid, start, end, mode)
        assert astar(start, end, grid, h)[1] == dijkstra_reference(grid, start, end)
        assert h(end) == 0
        nodes = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value is not None]
        for p in nodes:
            exact = dijkstra_reference(grid, p, end)
            if exact >= 0:
                assert 0 <= h(p) <= exact
            for q in nodes:
                if manhattan(p, q) == 1:
                    assert h(p) <= grid[q[0]][q[1]] + h(q)


def test_topology_dominates_on_reachable_cells():
    grid = [[1, None, 1, 1], [1, None, 1, 1], [1, 1, 1, 0]]
    end = (0, 3)
    base, _ = build_heuristic(grid, (0, 0), end, 'manhattan')
    better, _ = build_heuristic(grid, (0, 0), end, 'topology')
    # Remove zero-weight cell so topology yields a strict obstacle-aware bound.
    grid[2][3] = 1
    base, _ = build_heuristic(grid, (0, 0), end, 'manhattan')
    better, _ = build_heuristic(grid, (0, 0), end, 'topology')
    assert better((0, 0)) > base((0, 0))
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value is not None:
                assert better((r, c)) >= base((r, c))


@pytest.mark.parametrize('mode', ['minimax', 'left', 'right', 'ordered'])
def test_tree_order_and_skipped_node_accounting(mode):
    roots = [official_tree(), TreeNode(-3)]
    roots += [make_tree(seed, 5, unbalanced=bool(seed % 2)) for seed in range(8)]
    for root in roots:
        stats = BenchStats()
        sizes = tree_sizes(root)
        assert ordered_tree_search(root, mode, stats, sizes) == minimax(root)
        assert stats.visited + stats.skipped_nodes == sizes[id(root)]
        if mode == 'left':
            baseline = SearchStats()
            alpha_beta(root, baseline)
            assert stats.visited == baseline.visited
            assert stats.leaf_evaluations == baseline.leaf_evaluations
        if mode != 'ordered':
            assert stats.ordering_leaves == 0


@pytest.mark.parametrize('mode', GAME_MODES)
def test_all_official_reachable_states_and_depths(game, mode):
    for length in range(5):
        for sequence in permutations(VALUES, length):
            state = GameState()
            for target in sequence:
                state = game.move(state, target)
            for depth in range(5-length):
                result = game_solver(game, mode).solve(depth, state)
                assert result == game.solve(depth, 'minimax', state)


@pytest.mark.parametrize('order', ['left', 'right', 'ordered'])
def test_cache_bounds_and_windows(game, order):
    solver = TreasureSearch(game, order, cache=True)
    state = GameState()
    exact = game.solve(4, 'minimax')['best_value']
    for alpha, beta in ((-10, -5), (8, 10), (0, 1), (2, 4), (-inf, inf)):
        value = solver.value(state, 4, alpha, beta)
        if exact <= alpha:
            assert exact <= value <= alpha
        elif exact >= beta:
            assert beta <= value <= exact
        else:
            assert value == exact
        # Check each stored bound against a fresh complete search of that state.
        for (cached_state, h), entry in solver.table.items():
            truth = game.solve(h, 'minimax', cached_state)['best_value']
            if entry.flag == 'EXACT':
                assert entry.value == truth
            elif entry.flag == 'LOWER_BOUND':
                assert entry.value <= truth
            else:
                assert entry.value >= truth
    stats = BenchStats()
    assert solver.value(state, 4, stats=stats) == exact
    assert stats.cache_hits == 1 and stats.leaf_evaluations == 0
    # Scores and depth must be part of key; never reuse a shifted U incorrectly.
    assert solver.value(replace(state, score_a=17), 4) == exact+17
    assert solver.value(state, 2) == -1


def test_random_games_ties_and_cache():
    rng = Random(43)
    labels = ['A', 'B', 'T1', 'T2', 'T3', 'T4']
    for _ in range(30):
        matrix = [[0 if i == j else rng.randint(0, 4) for j in range(6)] for i in range(6)]
        game = TreasureGame(labels, matrix, {t: rng.randint(0, 4) for t in labels[2:]})
        for depth in (1, 2, 3, 4):
            expected = game.solve(depth, 'minimax')
            for mode in GAME_MODES:
                assert game_solver(game, mode).solve(depth) == expected
    tied = TreasureGame(labels, [[0]*6 for _ in labels], {t: 1 for t in labels[2:]})
    for mode in GAME_MODES:
        assert game_solver(tied, mode).solve(4) == {'best_target': 'T1', 'best_value': 0}


def test_dataset_reproducibility_and_evidence(game):
    assert map_cases() == map_cases()
    evidence = decision_evidence(game)
    for row in evidence:
        score_a = sum(step['gain'] for step in row['line'] if step['player'] == 'A')
        score_b = sum(step['gain'] for step in row['line'] if step['player'] == 'B')
        assert score_a-score_b == row['value']
        child = game.move(GameState(), row['first'])
        assert row['value'] == game.solve(row['depth']-1, 'minimax', child)['best_value']


def test_summary_recomputes_median_iqr():
    rows = [dict(case='x', mode='zero', trial=i, prep_ns=i, core_ns=2*i,
                 total_ns=3*i, ordering_ns=0, visited=1, expanded=0,
                 leaf_evaluations=0, cutoffs=0, cache_hits=0, optimal=True) for i in range(1, 6)]
    summary = summarize(rows)[0]
    assert summary['core_ns'] == 6
    assert summary['core_ns_iqr'] == 4
    assert summary['total_ns'] == 9


@pytest.mark.parametrize('depth', [-1, True, 1.5])
def test_experimental_value_rejects_invalid_depth(game, depth):
    with pytest.raises(ValueError, match='depth'):
        TreasureSearch(game).value(GameState(), depth)


@pytest.mark.parametrize('depth', [0, 2])
def test_experimental_search_rejects_invalid_turn(game, depth):
    state = replace(GameState(), turn='C')
    with pytest.raises(ValueError, match='turn'):
        TreasureSearch(game).solve(depth, state)
    with pytest.raises(ValueError, match='turn'):
        TreasureSearch(game).value(state, depth)


def test_experimental_value_rejects_closed_window(game):
    with pytest.raises(ValueError, match='alpha'):
        TreasureSearch(game).value(GameState(), 4, alpha=2, beta=2)


def test_summary_rejects_hidden_counter_drift():
    rows = [dict(case='x', mode='zero', trial=i, prep_ns=1, core_ns=2,
                 total_ns=3, ordering_ns=0, generated=i+1, optimal=True) for i in range(2)]
    with pytest.raises(AssertionError):
        summarize(rows)
