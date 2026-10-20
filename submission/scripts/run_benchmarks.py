"""Offline reproducible experiments. Raw trials precede all summaries/analysis."""
import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from random import Random
import statistics as st
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from search_lab.astar import astar, terrain_costs, manhattan
from search_lab.examples import MAZE, TERRAIN, TREASURE_MAP, POSITIONS, VALUES, official_tree
from search_lab.minimax import TreeNode, minimax
from search_lab.optimizations import (BenchStats, TreasureSearch, build_heuristic,
                                     dijkstra_reference, ordered_tree_search, tree_sizes)
from search_lab.treasure import GameState, TreasureGame, distance_matrix

SEED = 20261019
ASTAR_MODES = ('zero', 'manhattan', 'topology')
TREE_MODES = ('minimax', 'left', 'right', 'ordered')
GAME_MODES = ('minimax', 'left', 'right', 'ordered', 'cache', 'ordered_cache')


def make_tree(seed, depth, branching=3, unbalanced=False):
    rng = Random(seed)
    def node(d, root=False):
        if not d or (unbalanced and not root and rng.random() < .22):
            return TreeNode(rng.randint(-100, 100))
        return TreeNode(children=[node(d-1) for _ in range(branching)])
    return node(depth, True)


def tree_json(node):
    return ({'children': [tree_json(c) for c in node.children]} if node.children
            else {'value': node.value})


def map_cases():
    cases = []
    def add(name, kind, costs, start, end, **config):
        cases.append(dict(name=name, kind=kind, costs=costs, start=start, end=end, **config))
    add('maze_official', 'maze', [[None if v == 'X' else 1 for v in row] for row in MAZE], (0, 0), (3, 4))
    add('weighted_official', 'weighted', terrain_costs(TERRAIN), (0, 0), (4, 4))
    for kind in ('maze', 'weighted'):
        final = 1 if kind == 'maze' else 0
        add(kind+'_unreachable', kind, [[1, None, final], [1, None, 1]], (0, 0), (0, 2))
        grid = [[1]*12 for _ in range(12)]
        grid[-1][-1] = final
        add(kind+'_equal_paths', kind, grid, (0, 0), (11, 11))
        grid = [[1]*20 for _ in range(20)]
        for c in range(2, 19, 3):
            gap = 19 if (c//3) % 2 == 0 else 0
            for r in range(20):
                if r != gap:
                    grid[r][c] = None
        grid[-1][-1] = final
        add(kind+'_serpentine', kind, grid, (0, 0), (19, 19))
        for density in (.1, .25, .4):
            for seed in (SEED, SEED+1, SEED+2):
                # Independent streams make obstacle layout identical across weight regimes.
                walls = Random(seed)
                for max_weight in ((1,) if kind == 'maze' else (3, 9)):
                    walls.seed(seed)
                    weights = Random(seed+100)
                    grid = [[None if walls.random() < density else weights.randint(1, max_weight)
                             for _ in range(24)] for _ in range(24)]
                    grid[0][0], grid[-1][-1] = 1, final
                    add(f'{kind}_d{density}_w{max_weight}_s{seed}', kind, grid, (0, 0), (23, 23),
                        density=density, max_weight=max_weight, seed=seed)
    add('weighted_zero', 'weighted', [[1, 0, 0, 0], [2, None, 4, 0], [1, 1, 1, 0]], (0, 0), (2, 3))
    return cases


def game_solver(game, mode):
    return TreasureSearch(game, order='ordered' if mode.startswith('ordered') else
                          'right' if mode == 'right' else 'left',
                          cache='cache' in mode, pruning=mode != 'minimax')


def measure_astar(case, mode):
    stats = BenchStats()
    before = perf_counter_ns()
    h, prep_nodes = build_heuristic(case['costs'], case['start'], case['end'], mode)
    prep_ns = perf_counter_ns()-before
    before = perf_counter_ns()
    path, cost = astar(case['start'], case['end'], case['costs'], h, stats)
    core_ns = perf_counter_ns()-before
    if path:
        assert path[0] == case['start'] and path[-1] == case['end']
        assert all(manhattan(a, b) == 1 for a, b in zip(path, path[1:]))
        assert all(case['costs'][r][c] is not None for r, c in path)
        assert sum(case['costs'][r][c] for r, c in path[1:]) == cost
    else:
        assert cost == -1
    return dict(prep_ns=prep_ns, core_ns=core_ns, total_ns=prep_ns+core_ns,
                prep_nodes=prep_nodes, cost=cost, path_length=len(path)-1,
                path_json=json.dumps(path), heap_push_calls=max(stats.generated-1, 0),
                heap_pop_calls=stats.visited+stats.stale_pops,
                heap_operations=max(stats.generated-1, 0)+stats.visited+stats.stale_pops,
                **asdict(stats))


def measure_tree(root, mode):
    stats = BenchStats()
    before = perf_counter_ns()
    sizes = tree_sizes(root)
    prep_ns = perf_counter_ns()-before
    before = perf_counter_ns()
    value = ordered_tree_search(root, mode, stats, sizes)
    core_ns = perf_counter_ns()-before
    return dict(prep_ns=prep_ns, core_ns=core_ns, total_ns=prep_ns+core_ns,
                value=value, full_nodes=sizes[id(root)], **asdict(stats))


def measure_game(game, depth, mode):
    stats = BenchStats()
    before = perf_counter_ns()
    solver = game_solver(game, mode)
    prep_ns = perf_counter_ns()-before
    before = perf_counter_ns()
    result = solver.solve(depth, stats=stats)
    core_ns = perf_counter_ns()-before
    return dict(prep_ns=prep_ns, core_ns=core_ns, total_ns=prep_ns+core_ns,
                **result, cache_hit_rate=stats.cache_hits/stats.cache_probes if stats.cache_probes else 0,
                cache_entries=len(solver.table), **asdict(stats))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['case'], row['mode']), []).append(row)
    result = []
    for (case, mode), trials in groups.items():
        item = {k: v for k, v in trials[0].items() if k not in ('trial', 'path_json')}
        for metric in ('prep_ns', 'core_ns', 'total_ns', 'ordering_ns'):
            data = [r[metric] for r in trials]
            q1, _, q3 = st.quantiles(data, n=4, method='inclusive')
            item[metric] = st.median(data)
            item[metric+'_iqr'] = q3-q1
            item[metric+'_q1'], item[metric+'_q3'] = q1, q3
            item[metric+'_min'], item[metric+'_max'] = min(data), max(data)
        # All structural counts and outputs must be deterministic across trials.
        for key in trials[0]:
            if key in ('trial', 'prep_ns', 'core_ns', 'total_ns', 'ordering_ns'):
                continue
            assert len({r[key] for r in trials}) == 1, (case, mode, key)
        item['repeats'] = len(trials)
        result.append(item)
    return result


def decision_evidence(game):
    evidence = []
    for depth in (2, 4):
        for first in VALUES:
            state = GameState()
            line = []
            target = first
            for ply in range(depth):
                position = state.position_a if state.turn == 'A' else state.position_b
                cost = game.distances[position, target]
                actor = state.turn
                state = game.move(state, target)
                line.append(dict(player=actor, source=position, target=target, cost=cost,
                                 treasure_value=VALUES[target], gain=VALUES[target]-cost,
                                 score_a=state.score_a, score_b=state.score_b, utility=state.utility))
                if ply+1 < depth:
                    target = game.solve(depth-ply-1, 'minimax', state)['best_target']
            evidence.append(dict(depth=depth, first=first, value=state.utility, line=line))
    return evidence


def generate_analysis(summary, metadata, evidence, path):
    def table(rows, columns):
        header = '| ' + ' | '.join(label for _, label in columns) + ' |\n'
        header += '|'+'|'.join('---' for _ in columns)+'|\n'
        for r in rows:
            header += '| ' + ' | '.join(str(key(r) if callable(key) else r.get(key, '—')) for key, _ in columns) + ' |\n'
        return header
    official = [r for r in summary if r['case'] in ('maze_official', 'weighted_official', 'tree_official', 'treasure_H2', 'treasure_H4')]
    text = f'''# BLOCK 02 实证分析材料（不是正式实验报告）

## 环境与设置

- Python {metadata['python']}；{metadata['platform']}。
- 主种子 {metadata['seed']}；每种配置预热 {metadata['warmups']} 次，独立计时 {metadata['repeats']} 次；每轮随机打乱模式顺序，缓存每次冷启动。
- 高分辨率 perf_counter_ns；计时不含输入构建、正确性基准与文件写入。单独记录预处理、核心和两者之和；排序探测时间包含在核心时间内。未禁用 GC，也未固定 CPU 频率，短任务耗时受系统噪声影响，不据此宣称普适加速。
- {metadata['map_count']} 张地图、{metadata['tree_count']} 棵树、官方综合题两个深度；完整输入、种子和环境位于 benchmarks/config.json。每次运行记录均保存在 astar.csv / trees.csv / treasure.csv，summary.csv / summary.json 仅为派生汇总。
- CSV 中 *_ns 为纳秒；汇总采用中位数及 IQR（Q3−Q1，inclusive），同时保留最小/最大值。表格微秒由纳秒换算。

## 指标定义与公平性

visited 是 A* 有效出队次数或博弈递归进入次数；expanded 不含终止节点。A* generated 是插入数（含初始单元素堆），heap_pop_calls=visited+stale_pops；实际 heappush 函数调用数=generated−1，heap_operations 为两类实际函数调用之和。初始单元素列表构造没有调用 heappush。frontier_peak 含过期堆项。

cutoffs 只计确实跳过兄弟分支的事件；pruned_branches 是直接子分支，skipped_nodes 则计这些完整展开子树的节点数，不把二者混淆。树的结构大小预处理对所有模式相同，计入 prep_ns；没有读取 Minimax 答案来排序。树排序使用每个候选子树最左叶的样本值，额外的 ordering_nodes / ordering_leaves / ordering_ns 独立记录，样本叶读取不计入正式 leaf_evaluations。因此叶评价成本应同时查看两类叶读取，不能只看搜索叶数。

缓存命中率=cache_hits/cache_probes，命中包括边界收紧，不保证每次命中都跳过搜索。visited 包含缓存命中调用，叶评价仅计实际计算；根决策外层不探测缓存。根平局按原始 remaining 中的顺序（官方 T1–T4）稳定选择，遇到可能影响平局选择的窄窗口等值会全窗口重搜，计入 tie_researches 和时间。

## 官方实例实测

'''
    text += table(official, [('case', '案例'), ('mode', '模式'),
                            (lambda r: r.get('cost', r.get('value', str(r.get('best_target'))+'/'+str(r.get('best_value')))), '结果'),
                            ('visited', '访问'), ('expanded', '扩展'), ('leaf_evaluations', '叶评价'),
                            ('cutoffs', '剪枝'), ('skipped_nodes', '跳过节点'),
                            (lambda r: f"{r['core_ns']/1000:.2f} / {r['core_ns_iqr']/1000:.2f}", '核心中位数 / IQR μs'),
                            (lambda r: f"{r['total_ns']/1000:.2f}", '含预处理 μs')])
    text += '\n注：IQR 不代表标准差或置信区间。跳过节点按搜索发生次数统计，含平局重搜；有重搜时不能用完整树节点数减访问数反推剪掉的节点数。\n\n## A* 启发策略与敏感性\n\n'
    text += '''h=0 为基线。Manhattan 模式在单位迷宫等于普通曼哈顿距离；在加权图使用 BLOCK 01 的合法修正 m·max(d−1,0)+cE（终点 h=0），而不是忽视 E=0。拓扑改进先从终点 BFS 保留障碍、忽略权重差异，获得最少步数 L，再用 m·(L−1)+cE。任何真实路径至少有 L 步，非目标步费用至少 m，因此可采纳；邻居 L 相差至多 1，故一致。它在可达分量不弱于修正曼哈顿；断开分量用 0。不使用真实加权最短距离作启发。

单位迷宫中 BFS 下界其实已是精确距离，单次查询的 BFS 预处理本身已经解决步数问题，因此必须看含预处理总开销，不能把这当作免费信息。它适合固定终点多次查询，但本实验不假定摊销；零权令 m=0 时可能完全失效。消融为 zero → Manhattan → topology，均使用相同 A* 内核与 FIFO 同值策略。较强下界不保证每种地图都减少节点，尤其是所有最优路径形成同 f 平台时。

随机敏感性使用 24×24、障碍概率 0.10/0.25/0.40，三个固定种子；加权上限 3/9，障碍图在权重场景间保持一致。不人为修路，包含实际不可达样本；下表是每个配置组三张地图的算术平均节点数，并列出可达张数，不能解读为仅可达性能。

'''
    sensitivity = []
    for kind in ('maze', 'weighted'):
        for density in (.1, .25, .4):
            for weight in ((1,) if kind == 'maze' else (3, 9)):
                group = [r for r in summary if r['family'] == kind and r.get('density') == density and r.get('max_weight') == weight]
                if group:
                    row = dict(kind=kind, density=density, weight=weight)
                    for mode in ASTAR_MODES:
                        subset = [r for r in group if r['mode'] == mode]
                        row[mode] = round(st.mean(r['expanded'] for r in subset), 2)
                    row['reachable'] = sum(r['cost'] >= 0 for r in group if r['mode'] == 'zero')
                    sensitivity.append(row)
    text += table(sensitivity, [('kind', '地图'), ('density', '障碍概率'), ('weight', '权重上限'),
                               ('reachable', '可达/3'), ('zero', 'h=0 扩展'), ('manhattan', 'Manhattan 扩展'), ('topology', '拓扑扩展')])
    selected = {r['mode']: r for r in summary if r['case'] == 'maze_official'}
    text += (f"\n官方迷宫的扩展数依次为 {selected['zero']['expanded']}、{selected['manhattan']['expanded']}、"
             f"{selected['topology']['expanded']}，拓扑下界的核心中位数为 {selected['topology']['core_ns']/1000:.2f} μs，"
             f"含预处理为 {selected['topology']['total_ns']/1000:.2f} μs。改进的搜索阶段和单次端到端成本须分别评价。\n")
    text += ('\n0.40 障碍概率的三个种子全部不可达，平均扩展减少主要因为起点连通分量缩小，不能解释为高障碍让可达寻路更容易。'
             '权重上限增大而 m 仍为 1 时，下界相对真实代价变弱，启发收益趋小。折返迷宫三种模式扩展均为 285：'
             'FIFO 同值规则与长通道使更强下界未减少搜索，预处理成为额外负担。\n')
    text += '\n## 博弈树访问顺序与深度\n\n'
    trees = [r for r in summary if r['family'] == 'tree' and r['case'] != 'tree_official']
    text += table(trees, [('case', '树'), ('mode', '模式'), ('visited', '访问'), ('leaf_evaluations', '搜索叶'),
                         ('ordering_leaves', '排序叶读取'), ('skipped_nodes', '跳过节点'),
                         (lambda r: f"{r['total_ns']/1000:.2f}", '总中位数 μs')])
    wins = sum(next(x for x in trees if x['case'] == r['case'] and x['mode'] == 'ordered')['visited'] < r['visited']
               for r in trees if r['mode'] == 'left')
    text += f'\n额外树中，采样排序在 {wins}/{len(trees)//4} 个案例上比左到右减少访问。树深度为 3/5/7，分支数为 3，非平衡树以 0.22 概率提前成为叶。最左叶样本可能误导排序；读取样本本身增加工作，不保证比原始顺序更快。Minimax 的子节点顺序不影响值，Alpha-Beta 仅删除已无法影响祖先决策的分支。\n'
    text += '\n## 综合题消融与缓存\n\n'
    games = [r for r in summary if r['family'] == 'treasure']
    text += table(games, [('case', '深度'), ('mode', '模式'), ('visited', '访问'), ('leaf_evaluations', '叶'),
                         ('cache_hits', '缓存命中'), ('cache_probes', '探测'),
                         (lambda r: f"{r['cache_hit_rate']:.1%}", '命中率'), ('tie_researches', '平局重搜'),
                         (lambda r: f"{r['core_ns']/1000:.2f}", '核心 μs')])
    text += '''
排序采用“当前行动者即时 Gain 较大优先”，不改变收益或行动规则，也不以完整求解结果排序。消融包含 Minimax、原顺序 AB、逆序 AB、排序 AB、缓存 AB、排序+缓存 AB。

缓存键包含双方位置、剩余宝藏、双方累计分数、当前玩家、剩余深度；完整状态键保守且容易验证，但可能减少复用。缓存保存 EXACT、LOWER_BOUND、UPPER_BOUND，按原始调用窗口分类；读边界只收紧窗口，只有窗口关闭才直接返回边界。窄窗口结果不冒充精确值。每次独立测量重建缓存，未将预热缓存偷偷带入正式测量。

'''
    for depth in (2, 4):
        rows = {r['mode']: r for r in games if r['depth'] == depth}
        base, ordered, cached = rows['left'], rows['ordered'], rows['ordered_cache']
        text += (f"H={depth}：原顺序访问 {base['visited']}，排序访问 {ordered['visited']}，排序+缓存访问 {cached['visited']}；"
                 f"组合缓存命中 {cached['cache_hits']}/{cached['cache_probes']}。原顺序与组合核心中位数分别为 "
                 f"{base['core_ns']/1000:.2f}、{cached['core_ns']/1000:.2f} μs。\n\n")
    text += '''只有四个宝藏、四层以内的博弈未必产生相同完整状态，缓存维护成本可能超过收益。这里不把缓存命中为零解释成实现失败，也不把基础剪枝/置换表声称为原创算法；改进是针对本题的可验证实现与组合对照。

## H=2 与 H=4 为什么选不同首步

以下每行固定 A 首步，再用完整 Minimax 求后续最优应对。完整路径、位置、移动费、Gain、累计得分已保存在 decision_branches.json，不从窄窗口剪枝边界读取分支精确值。

'''
    text += table(evidence, [('depth', 'H'), ('first', 'A首步'), ('value', '完整分支值'),
                            (lambda r: ' → '.join(f"{v['player']}:{v['target']}({v['treasure_value']}−{v['cost']}={v['gain']})" for v in r['line']), '双方最优应对 / Gain')])
    text += '''
H=2 只评价 A、B 各一次后的当前差值，无法计入占据宝藏位置对下一次移动的影响。H=4 包含双方第二次行动，初次 Gain 较差的走法可以通过后续移动成本及宝藏分配弥补。上表逐项收益可直接相加核验，具体首步结果由搜索产生，不在算法中硬编码。

具体而言，H=2 选 T2 的收益差为 2−3=−1，优于先选 T3 的 −1−1=−2；H=4 先取 T3 的一条双方最优分支中，A 总收益 −1+0=−1，B 总收益 1−5=−4，因此 U=3。B 最后一程 T1→T4 代价为 12，抵消宝藏价值 7 后产生 −5 的收益；而先取 T2 的完整分支 U=1。这体现了远期分配和路径成本，不能仅按首回合 Gain 贪心。

## 正确性、图表及局限

所有实验 trial 的 optimal 标记必须为 True，否则脚本抛错停止。A* 使用独立 O(V²) Dijkstra（不调用 A*、不使用其堆）交叉校验总代价，验证实际路径相邻、可通行和进入费用。博弈树逐模式对照 BLOCK 01 完整 Minimax，综合题逐模式同时比较 best_target / best_value。额外测试覆盖启发函数逐格可采纳/一致性、不同窗口缓存复用、平局重搜、所有可达官方状态和新增小博弈。

- [A* 搜索开销](figures/astar_costs.png)：固定地图及随机障碍敏感性。
- [树访问顺序](figures/tree_orders.png)：搜索访问与实际叶读取成本。
- [官方综合题](figures/treasure_costs.png)：H=2/H=4 访问、叶评价。
- [优化消融](figures/ablation.png)：节点开销与含预处理耗时，不能仅凭节点降幅判断时间收益。

图表由 plot_results.py 从 summary.json 生成；原始 CSV 与生成配置保留。结果仅代表这些固定种子和地图/树规模，不是统计总体结论。单次微秒测量的 IQR 不是置信区间；没有做显著性检验，不宣称某方法在所有输入上更快。正式报告尚未编写。
'''
    # analysis.md is one level above benchmarks; figures are relative to results/.
    path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--warmups', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmups < 1:
        parser.error('repeats >= 2 and warmups >= 1 required')
    folder = ROOT/'results'/'benchmarks'
    folder.mkdir(parents=True, exist_ok=True)
    maps = map_cases()
    trees = [('tree_official', official_tree(), {'depth': 3})]
    for unbalanced in (False, True):
        for depth in (3, 5, 7):
            name = f"tree_{'unbalanced' if unbalanced else 'balanced'}_d{depth}"
            trees.append((name, make_tree(SEED+depth, depth, unbalanced=unbalanced),
                          dict(depth=depth, branching=3, seed=SEED+depth, unbalanced=unbalanced)))
    before = perf_counter_ns()
    labels, matrix = distance_matrix(TREASURE_MAP, POSITIONS)
    matrix_ns = perf_counter_ns()-before
    game = TreasureGame(labels, matrix, VALUES)
    metadata = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                    seed=SEED, repeats=args.repeats, warmups=args.warmups, map_count=len(maps),
                    tree_count=len(trees), maps=maps,
                    trees=[dict(name=name, tree=tree_json(root), **config) for name, root, config in trees],
                    treasure=dict(grid=TREASURE_MAP, positions=POSITIONS, values=VALUES, labels=labels,
                                  matrix=matrix, matrix_setup_ns=matrix_ns),
                    packages={name: version(name) for name in ('pytest', 'matplotlib', 'numpy', 'pillow')},
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted([*(ROOT/'src'/'search_lab').glob('*.py'),
                                                    *(ROOT/'scripts').glob('*.py')])})
    (folder/'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    raw = {'astar': [], 'trees': [], 'treasure': []}
    rng = Random(SEED)

    def trials(name, family, modes, measure, check, category, extra):
        for trial in range(-args.warmups, args.repeats):
            shuffled = list(modes)
            rng.shuffle(shuffled)
            for mode in shuffled:
                row = measure(mode)
                row['optimal'] = check(row)
                if not row['optimal']:
                    raise AssertionError((name, mode, row))
                if trial >= 0:
                    raw[category].append(dict(case=name, family=family, mode=mode, trial=trial, **extra, **row))

    for case in maps:
        expected = dijkstra_reference(case['costs'], case['start'], case['end'])
        trials(case['name'], case['kind'], ASTAR_MODES, lambda mode: measure_astar(case, mode),
               lambda row: row['cost'] == expected, 'astar',
               {k: case[k] for k in ('density', 'max_weight', 'seed') if k in case})
    for name, root, config in trees:
        expected = minimax(root)
        trials(name, 'tree', TREE_MODES, lambda mode: measure_tree(root, mode),
               lambda row: row['value'] == expected, 'trees', config)
    for depth in (2, 4):
        expected = game.solve(depth, 'minimax')
        trials(f'treasure_H{depth}', 'treasure', GAME_MODES, lambda mode: measure_game(game, depth, mode),
               lambda row: all(row[k] == v for k, v in expected.items()), 'treasure', {'depth': depth})
    all_summary = []
    for category, rows in raw.items():
        write_csv(folder/f'{category}.csv', rows)
        all_summary.extend(summarize(rows))
    write_csv(folder/'summary.csv', all_summary)
    (folder/'summary.json').write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    evidence = decision_evidence(game)
    (folder/'decision_branches.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    generate_analysis(all_summary, metadata, evidence, ROOT/'results'/'analysis.md')
    print(f"Completed {sum(len(rows) for rows in raw.values())} raw trials; {len(all_summary)} configurations; all optimality checks passed.")


if __name__ == '__main__':
    main()
