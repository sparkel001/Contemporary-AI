"""Independently audit raw trials, summary files and numerical analysis tables."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def source_hash_matches(root, sources):
    """Resolve recorded relative paths with either separator on any host."""
    return {path: hashlib.sha256(
        (root / Path(path.replace('\\', '/'))).read_bytes()
    ).hexdigest() == digest for path, digest in sources.items()}


def independent_summary(folder):
    groups = {}
    counts = {}
    for family in ('astar', 'trees', 'treasure'):
        rows = read_csv(folder/f'{family}.csv')
        counts[family] = len(rows)
        for row in rows:
            assert row['optimal'] == 'True'
            assert int(row['total_ns']) == int(row['core_ns'])+int(row['prep_ns'])
            if family == 'astar':
                assert int(row['heap_pop_calls']) == int(row['visited'])+int(row['stale_pops'])
                assert int(row['heap_push_calls']) == max(int(row['generated'])-1, 0)
                assert int(row['heap_operations']) == int(row['heap_push_calls'])+int(row['heap_pop_calls'])
                assert len(json.loads(row['path_json']))-1 == int(row['path_length'])
            groups.setdefault((row['case'], row['mode']), []).append(row)
    result = {}
    for key, trials in groups.items():
        assert sorted(int(r['trial']) for r in trials) == list(range(len(trials)))
        row = {k: v for k, v in trials[0].items() if k not in ('trial', 'path_json') and v != ''}
        for field in row:
            if field not in ('prep_ns', 'core_ns', 'total_ns', 'ordering_ns'):
                assert len({t[field] for t in trials}) == 1, (key, field)
        for field in ('prep_ns', 'core_ns', 'total_ns', 'ordering_ns'):
            values = sorted(int(t[field]) for t in trials)
            # Independent linear interpolation for inclusive quartiles.
            def quantile(q):
                index = (len(values)-1)*q
                low = int(index)
                return values[low] + (values[min(low+1, len(values)-1)]-values[low])*(index-low)
            row[field] = quantile(.5)
            row[field+'_q1'], row[field+'_q3'] = quantile(.25), quantile(.75)
            row[field+'_iqr'] = quantile(.75)-quantile(.25)
            row[field+'_min'], row[field+'_max'] = min(values), max(values)
        for field, value in list(row.items()):
            if isinstance(value, str):
                if value in ('True', 'False'):
                    row[field] = value == 'True'
                else:
                    try:
                        row[field] = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass
        row['repeats'] = len(trials)
        result[key] = row
    return result, counts


def audit(root=ROOT):
    folder = root/'results/benchmarks'
    recomputed, counts = independent_summary(folder)
    stored = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert len(stored) == len(recomputed)
    for row in stored:
        assert row == recomputed[row['case'], row['mode']], row['case']
    for row in read_csv(folder/'summary.csv'):
        expected = recomputed[row['case'], row['mode']]
        for field, value in row.items():
            if value == '' and field not in expected:
                continue
            if isinstance(expected[field], (int, float)) and not isinstance(expected[field], bool):
                assert float(value) == expected[field], (row['case'], field)
            else:
                assert value == str(expected[field]), (row['case'], field)
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    source_matches = source_hash_matches(root, config['source_sha256'])
    assert all(source_matches.values()), source_matches
    # Parse the prose tables, not the summary generator. Check every data row.
    analysis = (root/'results/analysis.md').read_text(encoding='utf-8')
    checked = 0
    def number(cell):
        return float(cell.replace('%', ''))
    for line in analysis.splitlines():
        if not line.startswith('| '):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        key = tuple(cells[:2])
        if key not in recomputed:
            continue
        row = recomputed[key]
        if len(cells) == 10:  # official table
            for i, field in enumerate(('visited','expanded','leaf_evaluations','cutoffs','skipped_nodes'), 3):
                assert number(cells[i]) == row[field]
            assert cells[8] == f"{row['core_ns']/1000:.2f} / {row['core_ns_iqr']/1000:.2f}"
            assert number(cells[9]) == round(row['total_ns']/1000, 2)
        elif len(cells) == 7:  # tree table
            for i, field in enumerate(('visited','leaf_evaluations','ordering_leaves','skipped_nodes'), 2):
                assert number(cells[i]) == row[field]
            assert number(cells[6]) == round(row['total_ns']/1000, 2)
        elif len(cells) == 9:  # treasure table
            for i, field in enumerate(('visited','leaf_evaluations','cache_hits','cache_probes'), 2):
                assert number(cells[i]) == row[field]
            assert number(cells[6]) == round(row['cache_hit_rate']*100, 1)
            assert number(cells[7]) == row['tie_researches']
            assert number(cells[8]) == round(row['core_ns']/1000, 2)
        else:
            raise AssertionError(cells)
        checked += 1
    sensitivity_checks = 0
    for line in analysis.splitlines():
        if not line.startswith(('| maze |', '| weighted |')):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        kind, density, weight = cells[0], float(cells[1]), int(cells[2])
        group = [r for r in recomputed.values() if r['family'] == kind and r.get('density') == density and r.get('max_weight') == weight]
        assert int(cells[3]) == sum(r['cost'] >= 0 for r in group if r['mode']=='zero')
        for index, mode in enumerate(('zero', 'manhattan', 'topology'), 4):
            assert float(cells[index]) == round(statistics.mean(r['expanded'] for r in group if r['mode']==mode), 2)
        sensitivity_checks += 1
    # Recompute branch trajectories from the literal map/matrix, not prose.
    import sys
    sys.path.insert(0, str(root/'src'))
    from search_lab.treasure import TreasureGame, GameState
    game = TreasureGame(config['treasure']['labels'], config['treasure']['matrix'], config['treasure']['values'])
    branches = json.loads((folder/'decision_branches.json').read_text(encoding='utf-8'))
    for branch in branches:
        state = GameState()
        for step in branch['line']:
            source = state.position_a if state.turn=='A' else state.position_b
            assert source == step['source'] and state.turn == step['player']
            assert step['cost'] == game.distances[source, step['target']]
            assert step['gain'] == game.values[step['target']]-step['cost']
            state = game.move(state, step['target'])
            assert (state.score_a,state.score_b,state.utility) == (step['score_a'],step['score_b'],step['utility'])
        assert state.utility == branch['value']
        first = game.move(GameState(), branch['first'])
        assert game.solve(branch['depth']-1, 'minimax', first)['best_value']==branch['value']
    return dict(raw_counts=counts, configurations=len(recomputed), analysis_rows_checked=checked,
                sensitivity_rows_checked=sensitivity_checks, branches_checked=len(branches),
                source_hashes_match=True, all_checks_passed=True)


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
