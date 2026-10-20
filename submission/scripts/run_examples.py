"""Run official examples; write reproducible outputs and raw counters."""
import json
from dataclasses import asdict
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from search_lab.astar import maze_path, terrain_cost
from search_lab.examples import MAZE, TERRAIN, TREASURE_MAP, POSITIONS, VALUES, official_tree
from search_lab.minimax import minimax, alpha_beta
from search_lab.stats import SearchStats
from search_lab.treasure import distance_matrix, TreasureGame


def main():
    results = {}

    def run(name, function):
        stats = SearchStats()
        before = perf_counter_ns()
        output = function(stats)
        results[name] = {'output': output, 'stats': asdict(stats),
                         'elapsed_ns': perf_counter_ns() - before}
        return output

    for h in ('zero', 'manhattan'):
        run('q1_' + h, lambda s: maze_path((0, 0), (3, 4), MAZE, h, s))
    for h in ('zero', 'scaled'):
        run('q2_' + h, lambda s: terrain_cost((0, 0), (4, 4), TERRAIN, h, s))
    for algorithm in (minimax, alpha_beta):
        run('q3_' + algorithm.__name__, lambda s: algorithm(official_tree(), s))
    labels, matrix = run('q4_distances', lambda s: distance_matrix(TREASURE_MAP, POSITIONS, s))
    game = TreasureGame(labels, matrix, VALUES)
    for depth in (2, 4):
        for algorithm in ('minimax', 'alpha_beta'):
            run(f'q4_H{depth}_{algorithm}', lambda s: game.solve(depth, algorithm, stats=s))
    folder = ROOT / 'results'
    folder.mkdir(exist_ok=True)
    text = json.dumps(results, ensure_ascii=False, indent=2)
    (folder / 'examples.json').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
