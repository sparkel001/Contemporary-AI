# -*- coding: utf-8 -*-
"""
predict.py
用最终方案在测试集上生成提交文件。

最终方案：Full Cleaning + TF-IDF + MultinomialNB(alpha=0.1)；
预处理方式、TF-IDF 和模型参数均与验证集调参阶段保持一致。
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from data_utils import load_train_and_split, load_test_data
from text_clean import clean_series

# 固定参数（来自已有调参结果，不重新选择）
MAX_FEATURES = 30000
NGRAM_RANGE = (1, 1)
MIN_DF = 2
ALPHA = 0.1

OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")


def main():
    X_train, X_val, y_train, y_val = load_train_and_split()
    X_test, _ = load_test_data()

    X_train_p = clean_series(X_train)
    X_val_p = clean_series(X_val)
    X_test_p = clean_series(X_test)

    # 用「训练集 + 验证集」全部数据作为最终训练数据
    X_full = pd.concat([X_train_p, X_val_p], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    # TF-IDF 只在 X_full 上 fit，测试集只 transform（防数据泄露）
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
    )
    X_full_tfidf = vectorizer.fit_transform(X_full)
    X_test_tfidf = vectorizer.transform(X_test_p)

    model = MultinomialNB(alpha=ALPHA)
    model.fit(X_full_tfidf, y_full)

    y_test_pred = model.predict(X_test_tfidf)

    # 单列、无表头、无索引，与课程要求一致
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(y_test_pred).to_csv(OUTPUT_CSV, index=False, header=False)

    print("=" * 50)
    print("预测标签分布:")
    print(pd.Series(y_test_pred).value_counts().sort_index())
    print("预测总条数:", len(y_test_pred))

    print(f"\n提交文件已保存到: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
