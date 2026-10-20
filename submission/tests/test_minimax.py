from random import Random
import pytest
from search_lab.examples import official_tree
from search_lab.minimax import TreeNode, minimax, alpha_beta
from search_lab.stats import SearchStats


def test_official_tree_and_left_to_right_counts():
    full, pruned = SearchStats(), SearchStats()
    assert minimax(official_tree(), full) == alpha_beta(official_tree(), pruned) == 5
    assert (full.visited, full.leaf_evaluations, full.cutoffs) == (15, 8, 0)
    assert (pruned.visited, pruned.leaf_evaluations, pruned.cutoffs, pruned.pruned_branches) == (14, 7, 1, 1)


def test_official_json_input():
    def pair(a, b):
        return {'is_maximizing_player': True, 'children': [{'value': a}, {'value': b}]}
    data = {'tree_structure': {'is_maximizing_player': True, 'children': [
        {'is_maximizing_player': False, 'children': [pair(3, 5), pair(6, 9)]},
        {'is_maximizing_player': False, 'children': [pair(6, 9), pair(1, 2)]}]}}
    root = TreeNode.from_dict(data)
    assert minimax(root) == alpha_beta(root) == 5


@pytest.mark.parametrize('solver', [minimax, alpha_beta])
def test_leaf_unbalanced_negative_and_missing(solver):
    assert solver(TreeNode(-9)) == -9
    root = TreeNode(children=[TreeNode(-7), TreeNode(children=[TreeNode(-8), TreeNode(-2)])])
    assert solver(root) == -7
    with pytest.raises(ValueError):
        solver(TreeNode())


def test_equality_prunes_remaining_siblings():
    root = TreeNode(children=[TreeNode(3), TreeNode(children=[TreeNode(3), TreeNode(-99)])])
    stats = SearchStats()
    assert alpha_beta(root, stats) == minimax(root) == 3
    assert stats.pruned_branches == stats.cutoffs == 1
    assert stats.leaf_evaluations == 2


def test_seeded_trees():
    rng = Random(15)
    def tree(depth):
        if not depth or rng.random() < .25:
            return TreeNode(rng.randint(-20, 20))
        return TreeNode(children=[tree(depth-1) for _ in range(rng.randint(1, 4))])
    for _ in range(50):
        root = tree(4)
        full, pruned = SearchStats(), SearchStats()
        assert minimax(root, full) == alpha_beta(root, pruned)
        assert pruned.visited <= full.visited
        assert pruned.leaf_evaluations <= full.leaf_evaluations
