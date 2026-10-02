# -*- coding: utf-8 -*-
"""
text_clean.py
文本清洗工具：对 20 Newsgroups 原始邮件文本做确定性清洗，去掉邮件头、引用、
签名、标点等噪声，不依赖任何从数据拟合的统计量。

remove_other_headers_only 只删除 Subject 之外的邮件头，其余内容原样保留，
是组件消融实验「Remove Other Headers Only」及最终预测采用的预处理（见 predict.py）。
"""

import re

import pandas as pd

# 需要去除的邮件头前缀（统一小写，匹配时大小写不敏感）
HEADER_PREFIXES = (
    "from:", "subject:", "organization:", "lines:", "reply-to:",
    "nntp-posting-host:", "distribution:", "x-newsreader:",
    "newsgroups:", "followup-to:", "sender:", "message-id:",
    "date:", "path:", "xref:", "in-reply-to:", "references:",
)

# 需要去除的引用归属行前缀（统一小写，匹配时大小写不敏感）
ATTRIBUTION_PREFIXES = (
    "in article",
    "in reply to",
)

# 除 Subject 之外的邮件头前缀（与 HEADER_PREFIXES 一致，只是跳过 subject:）
OTHER_HEADER_PREFIXES = tuple(p for p in HEADER_PREFIXES if p != "subject:")


def clean_text(text):
    """清洗单条原始文本，返回清洗后的字符串（仅含小写字母、数字和空格）。"""
    # 非字符串（如 NaN）返回空串
    if not isinstance(text, str):
        return ""

    lines = text.splitlines()

    # 逐行过滤：去邮件头、去引用行、遇签名分隔线则截断
    kept = []
    for line in lines:
        stripped = line.lstrip()
        lowered = stripped.lower()

        if lowered.startswith(HEADER_PREFIXES):
            continue

        if lowered.startswith(ATTRIBUTION_PREFIXES):
            continue

        # 兼容 "=>" 引用前缀：先剥掉行首的 "=" 和空格，再判断是否以 ">" 开头
        quote_check = stripped.lstrip("=").lstrip()
        if quote_check.startswith(">"):
            continue

        if stripped.startswith("--"):
            break  # 去掉签名分隔线及其后所有内容

        kept.append(line)

    text = " ".join(kept)
    text = text.lower()

    # 只保留小写字母、数字和空格，其余替换成空格（避免单词粘连）
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    text = re.sub(r" +", " ", text)
    text = text.strip()

    return text


def clean_series(texts):
    """对 pandas Series 中的每一条文本调用 clean_text，返回清洗后的 Series。"""
    return texts.apply(clean_text)


def remove_other_headers_only(text):
    """只删除 Subject 之外的邮件头，Subject、正文、引用、签名等全部原样保留。

    删除字段与 clean_text 的 HEADER_PREFIXES 一致，只是跳过 subject:，且不转
    小写、不去标点、不去引用/签名。这是组件消融实验「Remove Other Headers
    Only」及最终预测采用的预处理。
    """
    if not isinstance(text, str):
        return text
    kept = []
    for line in text.splitlines():
        if line.lstrip().lower().startswith(OTHER_HEADER_PREFIXES):
            continue
        kept.append(line)
    return "\n".join(kept)


if __name__ == "__main__":
    # 简单自测：取第一条样本做清洗前后对比
    from data_utils import load_train_and_split

    X_train, X_val, y_train, y_val = load_train_and_split()

    raw = X_train.iloc[0]
    cleaned = clean_text(raw)

    print("=" * 50)
    print("清洗前文本:")
    print(raw)
    print("-" * 50)
    print("清洗前长度:", len(raw))

    print("=" * 50)
    print("清洗后文本:")
    print(cleaned)
    print("-" * 50)
    print("清洗后长度:", len(cleaned))
