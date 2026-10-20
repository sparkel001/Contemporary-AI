"""Render report-ready Chinese figures from recorded benchmark summaries."""
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'zero': 'h=0', 'manhattan': '合法曼哈顿', 'topology': '拓扑下界',
          'minimax': 'Minimax', 'left': 'AB 原顺序', 'right': 'AB 逆序',
          'ordered': 'AB 排序', 'cache': 'AB 缓存', 'ordered_cache': 'AB 排序+缓存'}
COLORS = ['#64748b', '#2563eb', '#0d9488', '#f59e0b', '#8b5cf6', '#e11d48']


def configure_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimSun'):
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        raise RuntimeError('需要中文字体：Microsoft YaHei / Noto Sans CJK SC 等；请离线安装后重试')
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def bars(ax, rows, metric, title, ylabel, error=None):
    values = [r[metric] for r in rows]
    scale = 1000 if metric.endswith('_ns') else 1
    values = [v/scale for v in values]
    yerr = ([[max(0, r[metric]-r[metric+'_q1'])/scale for r in rows],
             [max(0, r[metric+'_q3']-r[metric])/scale for r in rows]] if error else None)
    rects = ax.bar(range(len(rows)), values, color=COLORS[:len(rows)], yerr=yerr, capsize=3)
    ax.set_xticks(range(len(rows)), [LABELS[r['mode']] for r in rows], rotation=18, ha='right')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.margins(y=.23)
    for i, (rect, value) in enumerate(zip(rects, values)):
        height = value + (yerr[1][i] if yerr else 0)
        ax.text(rect.get_x()+rect.get_width()/2, height, f'{value:.1f}' if scale > 1 else f'{value:g}',
                ha='center', va='bottom', fontsize=9)


def main():
    configure_font()
    rows = json.loads((ROOT/'results/benchmarks/summary.json').read_text(encoding='utf-8'))
    if not rows or not all(r['optimal'] for r in rows):
        raise ValueError('missing or invalid correctness data')
    folder = ROOT/'results/figures'
    folder.mkdir(exist_ok=True)

    def select(case, modes):
        return [next(r for r in rows if r['case'] == case and r['mode'] == mode) for mode in modes]

    def save(fig, name, caption):
        fig.text(.5, .012, caption, ha='center', fontsize=9, color='#475569')
        fig.tight_layout(rect=(0, .055, 1, .95))
        fig.savefig(folder/name, dpi=180)
        plt.close(fig)

    astar_modes = ('zero', 'manhattan', 'topology')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('A*：启发函数与搜索开销', fontsize=16, fontweight='bold')
    for ax, case, title in zip(axes[0], ('maze_official', 'weighted_official'),
                               ('官方迷宫：扩展节点', '官方加权地图：扩展节点')):
        bars(ax, select(case, astar_modes), 'expanded', title, '扩展节点数')
    for ax, kind in zip(axes[1], ('maze', 'weighted')):
        for mode, color in zip(astar_modes, COLORS):
            means = []
            for density in (.1, .25, .4):
                subset = [r for r in rows if r['family'] == kind and r['mode'] == mode and r.get('density') == density]
                means.append(st.mean(r['expanded'] for r in subset))
            ax.plot([.1, .25, .4], means, 'o-', color=color, label=LABELS[mode])
        ax.set_title(('随机迷宫' if kind == 'maze' else '随机加权图')+'：障碍密度敏感性', loc='left', fontweight='bold')
        ax.set_xlabel('障碍生成概率')
        ax.set_ylabel('平均扩展数（含不可达样本）')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    save(fig, 'astar_costs.png', '固定地图为单配置计数；随机图每密度迷宫 3 张 / 加权 6 张；全部使用独立 Dijkstra 校验')

    modes = ('minimax', 'left', 'right', 'ordered')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Alpha-Beta：访问顺序与排序成本', fontsize=16, fontweight='bold')
    bars(axes[0, 0], select('tree_official', modes), 'visited', '官方树：访问节点', '访问数')
    for ax, name in ((axes[0, 1], 'balanced'), (axes[1, 0], 'unbalanced')):
        for mode, color in zip(modes, COLORS):
            values = [select(f'tree_{name}_d{d}', (mode,))[0]['visited'] for d in (3, 5, 7)]
            ax.plot([3, 5, 7], values, 'o-', label=LABELS[mode], color=color)
        ax.set_title(('平衡树' if name == 'balanced' else '非平衡树')+'：深度敏感性', loc='left', fontweight='bold')
        ax.set_xlabel('最大深度')
        ax.set_ylabel('访问节点数（对数轴）')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    subset = select('tree_balanced_d7', modes)
    ax = axes[1, 1]
    ax.bar(range(4), [r['leaf_evaluations'] for r in subset], label='搜索叶评价', color='#2563eb')
    ax.bar(range(4), [r['ordering_leaves'] for r in subset], bottom=[r['leaf_evaluations'] for r in subset],
           label='额外排序样本叶读取', color='#f59e0b')
    ax.set_xticks(range(4), [LABELS[m] for m in modes], rotation=18)
    ax.set_title('深度 7 平衡树：不能忽略排序探测', loc='left', fontweight='bold')
    ax.set_ylabel('叶读取次数')
    ax.legend()
    save(fig, 'tree_orders.png', '所有模式返回同一完整 Minimax 值；样本排序不保证改善，排序额外读取单独计数')

    modes = ('minimax', 'left', 'right', 'ordered', 'cache', 'ordered_cache')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('官方四宝藏：搜索开销与消融对照', fontsize=16, fontweight='bold')
    for row, depth in enumerate((2, 4)):
        subset = select(f'treasure_H{depth}', modes)
        bars(axes[row, 0], subset, 'visited', f'H={depth}：访问节点', '递归进入次数')
        bars(axes[row, 1], subset, 'leaf_evaluations', f'H={depth}：叶评价', '实际叶评价次数')
    save(fig, 'treasure_costs.png', '规则、地图及四宝藏不变；稳定平局规则；每次冷缓存；H=2 为 T2/−1，H=4 为 T3/3')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('优化消融：节点减少不等于总耗时减少', fontsize=16, fontweight='bold')
    subset = select('maze_serpentine', astar_modes)
    bars(axes[0, 0], subset, 'expanded', '折返迷宫：搜索扩展', '节点数')
    bars(axes[0, 1], subset, 'total_ns', '折返迷宫：包含启发预处理', '中位数耗时（μs）', 'total_ns_iqr')
    modes = ('left', 'ordered', 'cache', 'ordered_cache')
    subset = select('treasure_H4', modes)
    bars(axes[1, 0], subset, 'visited', 'H=4：排序 / 缓存 / 组合', '访问节点数')
    bars(axes[1, 1], subset, 'total_ns', 'H=4：排序及缓存维护计入时间', '中位数耗时（μs）', 'total_ns_iqr')
    save(fig, 'ablation.png', '柱高为中位数，误差线为 Q1–Q3（不是置信区间）；原始重复记录及预处理耗时见 benchmarks/*.csv')
    print('Generated 4 PNG figures in results/figures')


if __name__ == '__main__':
    main()
