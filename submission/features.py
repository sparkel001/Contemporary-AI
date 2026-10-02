# -*- coding: utf-8 -*-
"""
features.py
特征构建：把清洗后的文本转成 TF-IDF 稀疏矩阵。

TfidfVectorizer 只在训练集上 fit，验证集/测试集只 transform，避免数据泄露。
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(X_train, X_val, X_test=None, max_features=50000, ngram_range=(1, 2)):
    """构建 TF-IDF 特征，返回 (训练集、验证集、测试集矩阵, vectorizer)；X_test 不传则跳过测试集。"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
    )

    # 只在训练集上 fit，验证集/测试集只 transform（防数据泄露）
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    X_test_tfidf = vectorizer.transform(X_test) if X_test is not None else None

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


if __name__ == "__main__":
    from data_utils import load_train_and_split
    from text_clean import clean_series

    X_train, X_val, y_train, y_val = load_train_and_split()

    X_train_clean = clean_series(X_train)
    X_val_clean = clean_series(X_val)

    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = build_tfidf(
        X_train_clean, X_val_clean
    )

    print("=" * 50)
    print("X_train_tfidf 形状:", X_train_tfidf.shape)
    print("X_val_tfidf   形状:", X_val_tfidf.shape)

    print("=" * 50)
    print("词汇表大小:", len(vectorizer.vocabulary_))

    nnz = X_train_tfidf.nnz
    total = X_train_tfidf.shape[0] * X_train_tfidf.shape[1]
    print("=" * 50)
    print("训练集矩阵非零元素数量:", nnz)
    print("训练集矩阵元素总数:", total)
    print("训练集矩阵稀疏度(非零比例):", nnz / total)
