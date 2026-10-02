# -*- coding: utf-8 -*-
"""
tune.py
验证集调参：对 TF-IDF 参数与模型超参（SVM/LR 的 C、NaiveBayes 的 alpha）做网格
搜索，选出验证集表现最好的组合。整个脚本只用训练集和验证集。
"""

import os

import pandas as pd

from data_utils import load_train_and_split
from text_clean import clean_series
from features import build_tfidf
from models import train_svm, train_lr, train_nb

RANDOM_STATE = 42

# 搜索范围
MAX_FEATURES_LIST = [10000, 30000, 50000]
NGRAM_RANGE_LIST = [(1, 1), (1, 2)]
SVM_C_LIST = [0.1, 1.0, 10.0]
LR_C_LIST = [0.1, 1.0, 10.0]
NB_ALPHA_LIST = [0.1, 0.5, 1.0, 2.0]

OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "tuning_results.csv")


def main():
    X_train, X_val, y_train, y_val = load_train_and_split()

    X_train_clean = clean_series(X_train)
    X_val_clean = clean_series(X_val)

    # 网格搜索
    records = []
    for max_features in MAX_FEATURES_LIST:
        for ngram_range in NGRAM_RANGE_LIST:
            print(f"处理 TF-IDF: max_features={max_features}, ngram_range={ngram_range} ...")

            X_train_tfidf, X_val_tfidf, _, _ = build_tfidf(
                X_train_clean, X_val_clean,
                max_features=max_features,
                ngram_range=ngram_range,
            )

            # SVM 的 C
            for C in SVM_C_LIST:
                _, _, acc, f1 = train_svm(
                    X_train_tfidf, y_train, X_val_tfidf, y_val,
                    C=C, verbose=False,
                )
                records.append({
                    "model_name": "SVM",
                    "max_features": max_features,
                    "ngram_range": str(ngram_range),
                    "param_name": "C",
                    "param_value": C,
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1, 4),
                })

            # 逻辑回归的 C
            for C in LR_C_LIST:
                _, _, acc, f1 = train_lr(
                    X_train_tfidf, y_train, X_val_tfidf, y_val,
                    C=C, verbose=False,
                )
                records.append({
                    "model_name": "LogisticRegression",
                    "max_features": max_features,
                    "ngram_range": str(ngram_range),
                    "param_name": "C",
                    "param_value": C,
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1, 4),
                })

            # 朴素贝叶斯的 alpha
            for alpha in NB_ALPHA_LIST:
                _, _, acc, f1 = train_nb(
                    X_train_tfidf, y_train, X_val_tfidf, y_val,
                    alpha=alpha, verbose=False,
                )
                records.append({
                    "model_name": "NaiveBayes",
                    "max_features": max_features,
                    "ngram_range": str(ngram_range),
                    "param_name": "alpha",
                    "param_value": alpha,
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1, 4),
                })

    results_df = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    results_df = results_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    print("=" * 70)
    print("调参结果（按 macro-F1 降序）:")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print("=" * 70)
    print("最佳组合:")
    for key, value in best.items():
        print(f"  {key}: {value}")

    print(f"\n结果已保存到: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
