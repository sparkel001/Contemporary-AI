"""Literal official fixtures, transcribed from PDF pp. 4, 7, 12 and 14."""

from .minimax import TreeNode

MAZE = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.'],
]
TERRAIN = [
    ['S', 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 'E'],
]
TREASURE_MAP = [
    ['A', 1, 1, 'X', 2, 1, 'T1'],
    [1, 'X', 6, 'X', 1, 'X', 1],
    [1, 1, 'T2', 1, 1, 1, 1],
    ['X', 1, 'X', 8, 'X', 1, 'X'],
    [1, 1, 1, 1, 'T3', 1, 1],
    [1, 'X', 9, 'X', 1, 'X', 1],
    ['T4', 1, 1, 'X', 1, 1, 'B'],
]
POSITIONS = {'A': (0, 0), 'B': (6, 6), 'T1': (0, 6),
             'T2': (2, 2), 'T3': (4, 4), 'T4': (6, 0)}
VALUES = {'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}


def official_tree():
    leaves = [TreeNode(value=v) for v in (3, 5, 6, 9, 1, 2)]
    max1, max2, max3 = [TreeNode(children=leaves[i:i+2]) for i in (0, 2, 4)]
    return TreeNode(children=[TreeNode(children=[max1, max2]),
                              TreeNode(children=[max2, max3])])
