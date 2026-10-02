# Experiment 1 - Text Classification

## 1. 简介

本实验使用 TF-IDF 进行文本特征表示，并比较：
- Naive Bayes
- SVM
- Logistic Regression

同时使用简单 MLP 绘制训练损失曲线。

另外通过一个文本清洗消融实验比较：
Raw Text 与 Cleaned Text。

## 2. 环境

Python 3.10

安装：

```bash
conda create -n project1 python=3.10
conda activate project1
conda install numpy pandas scikit-learn matplotlib
```

也可以直接：

```bash
pip install -r requirements.txt
```

## 3. 数据准备

数据文件需要放在本目录下的 `data/` 中：

```
data/
├── train_data.csv               # 训练集：text + target 两列，共 7368 条
└── test_data_unlabeled.csv      # 测试集：仅 text 一列，共 2457 条
```

如果数据放在其他位置，可以通过环境变量 `PROJECT1_DATA_DIR` 指定数据目录：

```bash
# Windows (PowerShell)
$env:PROJECT1_DATA_DIR = "D:/your/data/dir"
# Linux / macOS
export PROJECT1_DATA_DIR=/your/data/dir
```

## 4. 文件说明


```text
submission/
├── README.md
├── requirements.txt
├── data_utils.py
├── text_clean.py
├── features.py
├── models.py
├── tune.py
├── visualize.py
├── loss_curve.py
├── ablation.py
├── predict.py
└── outputs/
    ├── tuning_results.csv
    ├── model_comparison.png
    ├── tfidf_effect.png
    ├── loss_curve.png
    ├── ablation_results.csv
    ├── ablation_comparison.png
    └── predictions.csv
```
## 5. 运行方式

```bash
python models.py
python tune.py
python visualize.py
python loss_curve.py
python ablation.py
python predict.py
```

## 6. 输出结果

`outputs/` 目录下的输出文件：

| 文件 | 说明 |
|------|------|
| `tuning_results.csv` | 不同模型/参数在验证集上的结果 |
| `model_comparison.png` | 模型最佳 Macro-F1 对比图 |
| `tfidf_effect.png` | TF-IDF 参数影响图 |
| `loss_curve.png` | MLP 训练损失曲线 |
| `ablation_results.csv` | 文本清洗消融实验结果 |
| `ablation_comparison.png` | 消融实验对比图 |
| `predictions.csv` | 最终测试集预测（无表头，无索引） |
