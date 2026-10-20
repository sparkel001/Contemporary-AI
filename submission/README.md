# 当代人工智能实验二：搜索算法

## 运行

算法支持 Python 3.10+；包含绘图的完整环境需要 Python 3.11+，本次使用 Windows / Python 3.14.4。当前项目的 `.venv` 已安装依赖，可直接运行（无需联网）：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_examples.py
```

新环境先安装依赖，再运行；安装步骤与测试分开：

```powershell
python -m venv .venv
python -m pip --python .venv/Scripts/python.exe install -r requirements.txt
```

若本机 `venv` 的 ensurepip 失败，但虚拟环境 Python 已创建，可使用上述宿主 pip 的 `--python` 命令安装。离线机器需要预先安装 pytest，或从已准备的 wheel 目录安装：

```powershell
python -m pip --python .venv/Scripts/python.exe install --no-index --find-links <wheel目录> -r requirements.txt
```

测试不读取外部 PDF、不下载数据、不访问网络；自动 fixture 禁止 socket 连接。随机测试固定种子。示例脚本从任意工作目录均可运行，结果写入 `results/examples.json`。算法结果和计数确定，`elapsed_ns` 随机器和负载变化，不作为测试断言。

## 接口与规则

导入前将 `src` 加入 `PYTHONPATH`（pytest.ini 与示例脚本已处理）。

```python
from search_lab.astar import maze_path, terrain_cost
from search_lab.minimax import TreeNode, minimax, alpha_beta
from search_lab.stats import SearchStats

stats = SearchStats()
path = maze_path((0, 0), (0, 1), [['S', 'E']], stats=stats)
assert path == [(0, 0), (0, 1)]
assert terrain_cost((0, 0), (0, 1), [['S', 'E']]) == 0
assert alpha_beta(TreeNode(children=[TreeNode(3), TreeNode(5)])) == 5
```

| 题目 / PDF 页码 | 输入与返回 |
|---|---|
| 一 / 3–5 | `maze_path(start, end, grid, heuristic='manhattan', stats=None)` 返回完整坐标元组列表，无路为 `[]`；`heuristic='zero'` 对应 h=0 |
| 二 / 5–7 | `terrain_cost(start, end, grid, heuristic='scaled', stats=None)` 返回整数代价，无路为 `-1`；可选 `'zero'` |
| 三 / 8–13 | `minimax(root, stats=None)`、`alpha_beta(root, stats=None)` 返回 MAX 最佳分数；`TreeNode.from_dict` 支持原文 `tree_structure` JSON |
| 四 / 14–15 | `distance_matrix(grid, positions, stats=None)` 返回标签顺序和矩阵；`TreasureGame(labels, matrix, values).solve(H, algorithm='alpha_beta', state=None, stats=None)` 返回 `best_target` / `best_value` 字典 |

四方向遍历固定为右、下、左、上；A* 同 f 值按入队顺序取出，允许更优 g 更新及重新入队，忽略过期堆项。起点不收取进入费用。负权、未知格子、非矩形和越界输入抛出 `ValueError`，障碍起终点按不可达处理。

题二支持整数及数字字符串，E 的进入代价严格为 0。题面未规定再次进入字符 S 的费用，实现内部将 S 规范化为 1；这是实现约定，不是题面给定的规则。起点本身不收取进入费用，且非负代价最优路可取为无环路径，无需再次进入起点，因此该约定不改变最优结果。支持零权地形作为新增边界情况。

题三严格从左到右、根 MAX、逐层交替，无排序或记忆化。第 12 页的 `max2` 被两个 MIN 节点引用；按展开树逐次访问，不按 Python 对象身份去重。第 10–11 页 JSON 的重复子树同样测试。

题四原图逐格保存在 `examples.py`，A=(0,0)、B=(6,6)，T1=(0,6)/9、T2=(2,2)/6、T3=(4,4)/7、T4=(6,0)/7。A/B/T 格子进入费用均为 1，不套用题二 E=0 规则。全部 36 个有序点对通过 A* 计算（包括对角线）。

`GameState` 保存双方位置标签、剩余宝藏、双方累计收益、当前玩家；标签对应原图固定坐标。每次行动只移除选中的宝藏、更新行动者的位置与分数，Gain=价值−移动代价，允许负收益。路径经过宝藏不拾取、玩家不阻挡、已获取宝藏格不移除。每层为一名玩家的一次行动；达到 H 或宝藏耗尽时仅计算当前 Score_A−Score_B。默认候选顺序 T1–T4，同值保留先遇到者。H=0 或已终局时直接以当前 `Score_A−Score_B` 作为 `best_value`，并返回 `best_target=None`；默认初始状态下该值为0。官方地图全部可达；自定义地图出现不可达目标时明确抛错，不擅自补充跳过回合规则。

## 题二启发函数说明

原始曼哈顿距离不可直接使用：地图 `[S,E]` 的真实费用为 0，但曼哈顿距离为 1。设 d 为到目标的曼哈顿距离，m 为所有可通行非目标格的最小进入费用，cE 为目标进入费用。使用：

```text
h(E) = 0
h(n) = max(d(n,E) - 1, 0) * m + cE   (n != E)
```

任何到目标的路径至少有 d 步，最后一步费用 cE，其余至少 d−1 次非目标进入各不小于 m，因此不会高估。相邻非目标格间 d 最多变化 1，h 的下降至多为 m，不大于实际边费用；进入目标时 h 恰为 cE。因此启发函数可采纳且一致。若存在零权非目标格则 m=0，官方 E=0 时退化为 h=0。官方样例 m=1，起点 h=7。测试逐格对照独立 Bellman-Ford 验证下界，随机地图同时检查一致性；没有硬编码搜索答案。

## 统计定义

可选 `SearchStats` 不改变官方返回格式。复用对象时累加计数；单次比较应新建对象。

- `visited`：A* 非过期出队次数（含目标）；博弈递归进入次数（含根和叶）。
- `expanded`：实际展开后继的节点次数，不含目标/终止节点。
- `generated`：A* 入堆次数（含起点）；博弈实际递归进入的子分支数（不含根）。
- `stale_pops`、`frontier_peak`：过期堆项数与最大堆长度（含过期项）。
- `leaf_evaluations`：叶节点或深度终止评价次数。
- `cutoffs`：确实跳过至少一个兄弟分支的剪枝事件数。
- `pruned_branches`：上述跳过的直接子分支数，不冒充全部未访问后代数。

题三官方树中的 `max2` 被左右两个 MIN 节点引用；统计按展开树的访问次数计算，同一 Python 对象被不同父节点访问时分别计数。`pruned_branches` 只计被跳过的直接子分支，`skipped_nodes` 才统计这些分支中未访问的后代节点。

示例脚本另记录 `perf_counter_ns` 耗时，为后续性能对照预留接口。

## 实测输出

题一两种启发均返回：

```text
[(0,0), (0,1), (0,2), (1,2), (1,3), (1,4), (2,4), (3,4)]
```

| 题目 / 模式 | 结果 | 访问数 | 叶求值数 | 剪枝事件 / 跳过分支 |
|---|---|---:|---:|---:|
| 一 h=0 | 7 步 | 16 | — | — |
| 一 Manhattan | 7 步 | 14 | — | — |
| 二 h=0 | 7 | 15 | — | — |
| 二 修正下界 | 7 | 11 | — | — |
| 三 Minimax | 5 | 15 | 8 | 0 / 0 |
| 三 Alpha-Beta | 5 | 14 | 7 | 1 / 1 |
| 四 H=2 Minimax | T2 / -1 | 17 | 12 | 0 / 0 |
| 四 H=2 Alpha-Beta | T2 / -1 | 13 | 8 | 2 / 4 |
| 四 H=4 Minimax | T3 / 3 | 65 | 24 | 0 / 0 |
| 四 H=4 Alpha-Beta | T3 / 3 | 47 | 16 | 5 / 6 |

距离矩阵行列均按 A、B、T1、T2、T3、T4 排列。本题矩阵对称是当前地图的性质：六个端点进入费用均为1，移动费用只取决于进入的格子，反向路径经过相同格子集合，因此费用相同；一般按进入格收费的地图不保证距离对称。

```text
 0 12 10  4  8  8
12  0  8  8  4 10
10  8  0  6  6 12
 4  8  6  0  6  6
 8  4  6  6  0  6
 8 10 12  6  6  0
```

测试覆盖官方四题、独立 BFS/Bellman-Ford 对照、固定种子地图和博弈树、所有可达宝藏局面的不同深度、零权/不可达/同起终点/非法输入、收益与位置更新、经过宝藏和玩家、终局、同值剪枝及左到右访问计数。测试记录见 `results/pytest.txt`，原始输出及计数见 `results/examples.json`。

历史执行结果：`35 passed in 0.40s`，pytest 8.4.2。测试后的最新结果见 `results/pytest.txt`。

## 初始文件树（当前新增文件见后文）

```text
project/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── src/search_lab/
│   ├── __init__.py
│   ├── astar.py
│   ├── examples.py
│   ├── minimax.py
│   ├── stats.py
│   └── treasure.py
├── tests/
│   ├── conftest.py
│   ├── test_astar.py
│   ├── test_minimax.py
│   └── test_treasure.py
├── scripts/run_examples.py
└── results/
    ├── examples.json
    └── pytest.txt
```

## 性能实验复现

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_benchmarks.py --warmups 2 --repeats 15
.\.venv\Scripts\python.exe scripts/plot_results.py
```

所有输入固定种子并保存完整实例，不需 PDF、联网或外部数据。首次安装依赖后可以断网运行。找不到时脚本明确报错，避免输出乱码图。脚本从任意工作目录可运行，会重建同名结果文件；需要保留多次独立实验时请先复制 `results/benchmarks`。

本轮涵盖 36 张地图、7 棵树、148 个配置，每配置预热 2 次、冷启动独立重复 15 次，共 2,220 条原始记录。随机地图包含三种障碍密度、三个种子、两种加权范围，另有官方、等价最短路、不可达、折返和零权案例。树覆盖官方、深度 3/5/7 的平衡与非平衡树。官方四宝藏仅比较 H=2/H=4，不修改题面。

扩展代码集中在 `src/search_lab/optimizations.py`，没有修改 BLOCK 01 的接口。A* 对比 h=0、合法修正曼哈顿、BFS 拓扑下界；树对比完整 Minimax、左右访问及最左叶样本排序；综合题加入即时 Gain 排序与带 EXACT/LOWER_BOUND/UPPER_BOUND 的状态缓存。平局保持原始候选次序，需要时进行全窗口重搜。缓存每次测量清空；排序探测、BFS 和结构统计预处理均如实记录成本。

关键实测现象：官方迷宫扩展数从 Manhattan 的 13 降为拓扑下界的 7，但 BFS 增加单次总成本；H=4 原顺序 AB 访问 47，排序后 37；官方缓存命中率为 0%，没有搜索节点收益。完整结果与局限见 [分析材料](results/analysis.md)，此文件不是最终实验报告。

新增文件与产物：

```text
src/search_lab/optimizations.py
scripts/run_benchmarks.py
scripts/plot_results.py
tests/test_benchmarks.py
results/analysis.md
results/benchmarks/
  astar.csv                 # 逐次路径、代价、堆操作、计数、时间、正确性
  trees.csv                # 逐次根值、剪枝、节点和排序探测成本
  treasure.csv             # 逐次首步、收益、缓存和剪枝统计
  summary.csv / summary.json
  config.json              # 完整输入、环境、包版本、种子、源文件 SHA256
  decision_branches.json   # 两个深度全部首步的精确值及最优应对展开
results/figures/
  astar_costs.png
  tree_orders.png
  treasure_costs.png
  ablation.png
```

时间统计采用中位数、Q1/Q3、IQR、最小与最大值；节点数为主要指标。图中误差线是 Q1–Q3，不是置信区间。算法正确性对照独立 Dijkstra 和原有完整 Minimax，测试额外验证不同 alpha/beta 窗口下的缓存上下界及平局稳定性。微秒级计时不保证跨机器一致；原始计数、最优值和固定地图可复现。

