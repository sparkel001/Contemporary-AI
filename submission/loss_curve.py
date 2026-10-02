# -*- coding: utf-8 -*-
"""
loss_curve.py
为实验报告绘制「训练损失随迭代轮数变化」的折线图。

sklearn 的 LogisticRegression / SVC 不记录逐轮 loss，只有 MLPClassifier 提供
loss_curve_ 属性，故用单隐层 MLP 跑出 loss 曲线。只使用训练集和验证集。
"""

import os

import matplotlib
matplotlib.use("Agg")  # 无界面后端，直接保存图片，不弹窗
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from data_utils import load_train_and_split
from text_clean import clean_series
from features import build_tfidf

# 中文字体设置：避免标题/坐标轴中文显示为方块乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

RANDOM_STATE = 42

# TF-IDF 最佳参数（来自验证集调参结果）
MAX_FEATURES = 50000
NGRAM_RANGE = (1, 1)

# MLP 超参
HIDDEN_LAYER_SIZES = (100,)
MAX_ITER = 300

OUTPUT_DIR = "outputs"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "loss_curve.png")


def plot_loss_curve(loss_curve):
    """把 MLP 每轮 loss 画成折线图并保存到 outputs/loss_curve.png。"""
    epochs = list(range(1, len(loss_curve) + 1))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(epochs, loss_curve, color="#1f77b4", linewidth=2,
            marker="o", markersize=3)

    ax.set_xlabel("迭代轮数 (epoch)")
    ax.set_ylabel("训练损失")
    ax.set_title("MLP 训练损失随迭代轮数的变化")
    ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_val, y_train, y_val = load_train_and_split()

    X_train_clean = clean_series(X_train)
    X_val_clean = clean_series(X_val)

    X_train_tfidf, X_val_tfidf, _, _ = build_tfidf(
        X_train_clean, X_val_clean,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=False,
        verbose=False,
    )
    model.fit(X_train_tfidf, y_train)

    loss_curve = model.loss_curve_
    print("=" * 50)
    print(f"实际迭代轮数: {len(loss_curve)}")
    print(f"初始 loss: {loss_curve[0]:.4f}")
    print(f"最终 loss: {loss_curve[-1]:.4f}")

    plot_loss_curve(loss_curve)
    print(f"\n图片已保存到: {OUTPUT_PNG}")

    # 在验证集上评估 MLP
    y_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    print("=" * 50)
    print("MLP 在验证集上的表现:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")


if __name__ == "__main__":
    main()
