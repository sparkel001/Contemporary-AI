# -*- coding: utf-8 -*-
"""
ablation.py
文本清洗消融实验：对比「原始文本」与「清洗文本」两组的分类性能。

控制变量：两组只在是否清洗文本上不同，其余条件（数据划分、TF-IDF、模型参数）一致。
"""

import os

import matplotlib
matplotlib.use("Agg")  # 无界面后端，直接保存图片，不弹窗
import matplotlib.pyplot as plt
import pandas as pd

from data_utils import load_train_and_split
from text_clean import clean_series
from features import build_tfidf
from models import train_nb

# 中文字体设置：避免标题/坐标轴中文显示为方块乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 最佳参数（来自调参结果，不重新选择）
MAX_FEATURES = 30000
NGRAM_RANGE = (1, 1)
ALPHA = 0.1

OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ablation_results.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "ablation_comparison.png")


def run_experiment(X_train, X_val, y_train, y_val):
    """对给定文本跑 TF-IDF + MultinomialNB，返回 (accuracy, macro_f1)。"""
    X_train_tfidf, X_val_tfidf, _, _ = build_tfidf(
        X_train, X_val,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )
    _, _, acc, f1 = train_nb(
        X_train_tfidf, y_train, X_val_tfidf, y_val,
        alpha=ALPHA, verbose=False,
    )
    return acc, f1


def plot_comparison(acc_raw, f1_raw, acc_cleaned, f1_cleaned):
    """画 Raw vs Cleaned 的 Accuracy / Macro-F1 分组柱状图。"""
    categories = ["Raw Text", "Cleaned Text"]
    x = [0, 1]
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))

    bars_acc = ax.bar([xi - width / 2 for xi in x],
                      [acc_raw, acc_cleaned], width,
                      label="Accuracy", color="#1f77b4")
    bars_f1 = ax.bar([xi + width / 2 for xi in x],
                     [f1_raw, f1_cleaned], width,
                     label="Macro-F1", color="#ff7f0e")

    # 柱子上方标注具体数值
    for bars in (bars_acc, bars_f1):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("文本设置")
    ax.set_ylabel("分数")
    ax.set_title("文本清洗消融实验对比")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_val, y_train, y_val = load_train_and_split()

    # 实验 A：原始文本
    acc_raw, f1_raw = run_experiment(X_train, X_val, y_train, y_val)

    # 实验 B：清洗文本
    X_train_clean = clean_series(X_train)
    X_val_clean = clean_series(X_val)
    acc_cleaned, f1_cleaned = run_experiment(X_train_clean, X_val_clean, y_train, y_val)

    # 差值（Cleaned - Baseline）
    diff_acc = acc_cleaned - acc_raw
    diff_f1 = f1_cleaned - f1_raw

    print("=== Text Cleaning Ablation Study ===")
    print()
    print("Baseline (Raw Text)")
    print(f"Accuracy: {acc_raw:.4f}")
    print(f"Macro-F1: {f1_raw:.4f}")
    print()
    print("Cleaned Text")
    print(f"Accuracy: {acc_cleaned:.4f}")
    print(f"Macro-F1: {f1_cleaned:.4f}")
    print()
    print("Difference")
    print(f"Accuracy: {diff_acc:+.4f}")
    print(f"Macro-F1: {diff_f1:+.4f}")

    results_df = pd.DataFrame({
        "setting": ["Raw Text", "Cleaned Text"],
        "accuracy": [round(acc_raw, 4), round(acc_cleaned, 4)],
        "macro_f1": [round(f1_raw, 4), round(f1_cleaned, 4)],
    })
    results_df.to_csv(OUTPUT_CSV, index=False)

    plot_comparison(acc_raw, f1_raw, acc_cleaned, f1_cleaned)

    print(f"\n结果已保存到: {OUTPUT_CSV}")
    print(f"图片已保存到: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
