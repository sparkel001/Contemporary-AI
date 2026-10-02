# -*- coding: utf-8 -*-
"""
data_utils.py
数据加载与划分工具。

数据目录默认取本文件所在目录（项目根目录）下的 data/，可用环境变量
PROJECT1_DATA_DIR 覆盖；找不到 CSV 时抛出带使用提示的 FileNotFoundError。
数据集为 20 Newsgroups 的 10 类文本分类（标签 0~9）。
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 项目根目录：本文件所在目录（不依赖 cwd）
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

TRAIN_FILENAME = "train_data.csv"
TEST_FILENAME = "test_data_unlabeled.csv"


def get_data_dir():
    """返回数据目录：优先用环境变量 PROJECT1_DATA_DIR，否则用项目下的 data/。"""
    env = os.environ.get("PROJECT1_DATA_DIR")
    return Path(env) if env else DEFAULT_DATA_DIR


def get_data_path(filename):
    """返回某数据文件的绝对路径；找不到时抛出带使用提示的 FileNotFoundError。"""
    data_dir = get_data_dir()
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到数据文件: {path}\n"
            f"默认查找位置: {data_dir}\n"
            f"请将 {filename} 放入 {data_dir}/，"
            f"或设置环境变量 PROJECT1_DATA_DIR 指向包含该文件的目录。"
        )
    return path


def load_train_and_split(val_size=0.2, random_state=42):
    """读取 train_data.csv 并分层切出训练/验证集，返回 (X_train, X_val, y_train, y_val)。"""
    train_df = pd.read_csv(get_data_path(TRAIN_FILENAME))

    X = train_df["text"]
    y = train_df["target"]

    # 分层切分：stratify=y 保证各类别比例在切分前后一致
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )

    return X_train, X_val, y_train, y_val


def load_test_data():
    """读取 test_data_unlabeled.csv，返回 (测试集文本, 测试集 id)。"""
    test_df = pd.read_csv(get_data_path(TEST_FILENAME))

    X_test = test_df["text"]
    if "id" in test_df.columns:
        test_ids = test_df["id"]
    else:
        test_ids = test_df.index

    return X_test, test_ids


if __name__ == "__main__":
    # 简单自测：加载并划分数据，打印样本数量与标签分布
    X_train, X_val, y_train, y_val = load_train_and_split()

    print("=" * 50)
    print("训练集样本数量:", len(X_train))
    print("验证集样本数量:", len(X_val))

    print("=" * 50)
    print("训练集标签分布:")
    print(pd.DataFrame({
        "数量": y_train.value_counts().sort_index(),
        "比例": y_train.value_counts(normalize=True).sort_index(),
    }))
    print("-" * 50)
    print("验证集标签分布:")
    print(pd.DataFrame({
        "数量": y_val.value_counts().sort_index(),
        "比例": y_val.value_counts(normalize=True).sort_index(),
    }))
