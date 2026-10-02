# -*- coding: utf-8 -*-
"""
visualize.py
为实验报告绘制两张图（只读取调参结果，不重新训练模型）：
    图一：不同模型在验证集上的最佳 Macro-F1 对比（柱状图）
    图二：TF-IDF 参数对验证集 Macro-F1 的影响（折线图，只看 LogisticRegression）
"""

import os

import matplotlib
matplotlib.use("Agg")  # 无界面后端，直接保存图片，不弹窗
import matplotlib.pyplot as plt
import pandas as pd

# 中文字体设置：避免标题/坐标轴中文显示为方块乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

OUTPUT_DIR = "outputs"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "tuning_results.csv")

# 分类色（蓝 / 橙 / 绿）
COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_GREEN = "#2ca02c"

# 每个模型一个固定颜色，不随柱子排序变化
MODEL_COLORS = {
    "NaiveBayes": COLOR_BLUE,
    "SVM": COLOR_ORANGE,
    "LogisticRegression": COLOR_GREEN,
}


def plot_model_comparison(df):
    """图一：每个模型取 macro_f1 最高的一组参数，按值降序画对比柱状图。"""
    best = df.loc[df.groupby("model_name")["macro_f1"].idxmax()].reset_index(drop=True)
    best = best.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    names = best["model_name"].tolist()
    values = best["macro_f1"].tolist()
    colors = [MODEL_COLORS.get(n, "#7f7f7f") for n in names]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, values, color=colors, width=0.5)

    # 柱子上方标注具体数值
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11)

    ax.set_xlabel("模型")
    ax.set_ylabel("验证集 Macro-F1")
    ax.set_title("不同模型在验证集上的最佳 Macro-F1 对比")
    ax.set_ylim(0, 0.95)  # 保留 0 基线，给数值标注留出空间
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_tfidf_effect(df):
    """图二：只看 LogisticRegression，两条线分别对应 ngram_range=(1,1) 与 (1,2)。"""
    lr = df[df["model_name"] == "LogisticRegression"].copy()

    # 对每个 (ngram_range, max_features) 取最好 C 的 macro_f1，以隔离 TF-IDF 参数的影响
    lr_best = lr.groupby(["ngram_range", "max_features"], as_index=False)["macro_f1"].max()

    fig, ax = plt.subplots(figsize=(6, 5))

    # 两条线：固定蓝/橙两色，marker 区分
    for ngram, color, marker in [("(1, 1)", COLOR_BLUE, "o"),
                                 ("(1, 2)", COLOR_ORANGE, "s")]:
        sub = lr_best[lr_best["ngram_range"] == ngram].sort_values("max_features")
        ax.plot(sub["max_features"], sub["macro_f1"],
                marker=marker, color=color, linewidth=2, markersize=7,
                label=f"ngram_range={ngram}")

    ax.set_xlabel("max_features")
    ax.set_ylabel("验证集 Macro-F1")
    ax.set_title("TF-IDF 参数对验证集 Macro-F1 的影响")
    ax.set_xticks([10000, 30000, 50000])
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "tfidf_effect.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)

    plot_model_comparison(df)
    plot_tfidf_effect(df)

    print("图片已保存:")
    print("  -", os.path.join(OUTPUT_DIR, "model_comparison.png"))
    print("  -", os.path.join(OUTPUT_DIR, "tfidf_effect.png"))


if __name__ == "__main__":
    main()
