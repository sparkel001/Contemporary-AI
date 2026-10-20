from dataclasses import replace
from itertools import permutations
import pytest
from search_lab.examples import TREASURE_MAP, POSITIONS, VALUES
from search_lab.stats import SearchStats
from search_lab.treasure import distance_matrix, TreasureGame, GameState
from test_astar import reference_cost


@pytest.fixture
def game():
    return TreasureGame(*distance_matrix(TREASURE_MAP, POSITIONS), VALUES)


def test_official_map_literal_and_matrix():
    assert [' '.join(map(str, row)) for row in TREASURE_MAP] == [
        'A 1 1 X 2 1 T1', '1 X 6 X 1 X 1', '1 1 T2 1 1 1 1',
        'X 1 X 8 X 1 X', '1 1 1 1 T3 1 1', '1 X 9 X 1 X 1', 'T4 1 1 X 1 1 B']
    assert POSITIONS == {'A': (0, 0), 'B': (6, 6), 'T1': (0, 6), 'T2': (2, 2), 'T3': (4, 4), 'T4': (6, 0)}
    assert VALUES == {'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}
    labels, matrix = distance_matrix(TREASURE_MAP, POSITIONS)
    costs = [[None if v == 'X' else v if type(v) is int else 1 for v in row] for row in TREASURE_MAP]
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            assert matrix[i][j] == reference_cost(costs, POSITIONS[a], POSITIONS[b])
            assert matrix[i][j] == matrix[j][i]
        assert matrix[i][i] == 0


@pytest.mark.parametrize('depth,target,value', [(2, 'T2', -1), (4, 'T3', 3)])
def test_official_solutions(game, depth, target, value):
    full, pruned = SearchStats(), SearchStats()
    expected = {'best_target': target, 'best_value': value}
    assert game.solve(depth, 'minimax', stats=full) == expected
    assert game.solve(depth, 'alpha_beta', stats=pruned) == expected
    assert pruned.visited <= full.visited
    assert pruned.leaf_evaluations <= full.leaf_evaluations
    assert full.cutoffs == 0
    assert full.visited == (17 if depth == 2 else 65)
    assert full.leaf_evaluations == (12 if depth == 2 else 24)


def test_moves_positions_scores_and_negative_gain(game):
    state = GameState()
    after_a = game.move(state, 'T2')
    assert after_a.position_a == 'T2' and after_a.position_b == 'B'
    assert after_a.score_a == 2 and after_a.score_b == 0
    assert after_a.remaining == ('T1', 'T3', 'T4') and after_a.turn == 'B'
    after_b = game.move(after_a, 'T3')
    assert after_b.position_a == 'T2' and after_b.position_b == 'T3'
    assert after_b.score_a == 2 and after_b.score_b == 3 and after_b.utility == -1
    assert state == GameState()  # transitions do not mutate parent
    assert game.move(state, 'T3').score_a == -1
    with pytest.raises(ValueError):
        game.move(after_a, 'T2')


def test_passing_treasures_and_player_does_not_collect_or_block():
    grid = [['A', 'T1', 'B', 'T2', 'T3', 'T4']]
    positions = {v: (0, i) for i, v in enumerate(grid[0])}
    game = TreasureGame(*distance_matrix(grid, positions), VALUES)
    state = game.move(GameState(), 'T4')
    assert state.remaining == ('T1', 'T2', 'T3')
    assert state.position_b == 'B' and state.score_a == 2
    assert game.distances['T4', 'A'] == 5  # collected cells stay traversable


def test_terminal_depth_and_score_offset(game):
    state = replace(GameState(), score_a=11, score_b=4)
    assert game.solve(0, state=state) == {'best_target': None, 'best_value': 7}
    assert game.solve(4, state=replace(state, remaining=())) == {'best_target': None, 'best_value': 7}
    assert game.solve(4, state=state) == {'best_target': 'T3', 'best_value': 10}
    assert game.solve(9) == game.solve(4)
    with pytest.raises(ValueError):
        game.solve(-1)


def test_every_reachable_state_against_independent_permutation_oracle(game):
    def oracle(state, h):
        if h == 0 or not state.remaining:
            return state.utility
        scores = [oracle(game.move(state, t), h-1) for t in state.remaining]
        return (max if state.turn == 'A' else min)(scores)
    for length in range(5):
        for sequence in permutations(VALUES, length):
            state = GameState()
            for target in sequence:
                state = game.move(state, target)
            for h in range(5-length):
                expected = oracle(state, h)
                for algorithm in ('minimax', 'alpha_beta'):
                    assert game.solve(h, algorithm, state)['best_value'] == expected


def test_ties_keep_first_target_and_unreachable_is_explicit():
    labels = ['A', 'B', 'T1', 'T2']
    game = TreasureGame(labels, [[0] * 4 for _ in labels], {'T1': 1, 'T2': 1})
    state = GameState(remaining=('T1', 'T2'))
    for algorithm in ('minimax', 'alpha_beta'):
        assert game.solve(1, algorithm, state) == {'best_target': 'T1', 'best_value': 1}
    game.distances['A', 'T1'] = -1
    with pytest.raises(ValueError, match='unreachable'):
        game.move(state, 'T1')
