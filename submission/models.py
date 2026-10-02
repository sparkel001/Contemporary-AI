# -*- coding: utf-8 -*-
"""
models.py
基线模型训练与评估：线性 SVM、逻辑回归与朴素贝叶斯。只用训练集和验证集。
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42


def evaluate(y_true, y_pred, name, verbose=True):
    """计算并（可选）打印 accuracy、macro-F1 与分类报告，返回 (accuracy, macro_f1)。"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    if verbose:
        print("=" * 50)
        print(f"模型: {name}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Macro-F1 : {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, digits=4))

    return acc, f1


def train_svm(X_train, y_train, X_val, y_val, C=1.0, verbose=True):
    """训练线性 SVM 并在验证集上评估，返回 (model, y_pred, accuracy, f1)。"""
    model = SVC(kernel="linear", C=C, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc, f1 = evaluate(y_val, y_pred, f"SVM (linear, C={C})", verbose=verbose)
    return model, y_pred, acc, f1


def train_lr(X_train, y_train, X_val, y_val, C=1.0, verbose=True):
    """训练逻辑回归并在验证集上评估，返回 (model, y_pred, accuracy, f1)。"""
    model = LogisticRegression(max_iter=1000, C=C, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc, f1 = evaluate(y_val, y_pred, f"LogisticRegression (C={C})", verbose=verbose)
    return model, y_pred, acc, f1


def train_nb(X_train, y_train, X_val, y_val, alpha=1.0, verbose=True):
    """训练朴素贝叶斯（MultinomialNB）并在验证集上评估，返回 (model, y_pred, accuracy, f1)。"""
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc, f1 = evaluate(y_val, y_pred, f"MultinomialNB (alpha={alpha})", verbose=verbose)
    return model, y_pred, acc, f1


if __name__ == "__main__":
    from data_utils import load_train_and_split
    from text_clean import clean_series
    from features import build_tfidf

    X_train, X_val, y_train, y_val = load_train_and_split()

    X_train_clean = clean_series(X_train)
    X_val_clean = clean_series(X_val)

    X_train_tfidf, X_val_tfidf, _, _ = build_tfidf(X_train_clean, X_val_clean)

    # 训练三个基线模型
    svm_model, svm_pred, svm_acc, svm_f1 = train_svm(
        X_train_tfidf, y_train, X_val_tfidf, y_val, C=1.0
    )
    lr_model, lr_pred, lr_acc, lr_f1 = train_lr(
        X_train_tfidf, y_train, X_val_tfidf, y_val, C=1.0
    )
    nb_model, nb_pred, nb_acc, nb_f1 = train_nb(
        X_train_tfidf, y_train, X_val_tfidf, y_val, alpha=1.0
    )

    print("=" * 50)
    print("结果对比:")
    print(f"{'模型':<22}{'Accuracy':<12}{'Macro-F1':<12}")
    print(f"{'SVM (linear)':<22}{svm_acc:<12.4f}{svm_f1:<12.4f}")
    print(f"{'LogisticRegression':<22}{lr_acc:<12.4f}{lr_f1:<12.4f}")
    print(f"{'MultinomialNB':<22}{nb_acc:<12.4f}{nb_f1:<12.4f}")
