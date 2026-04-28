#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
from typing import List, Sequence, Optional

def remove_rows_by_reason(
    file1_path: str,
    file2_path: str,
    output1_path: str,
    output2_path: str,
    # 同时支持两列里匹配（存在就检查，不存在就跳过）
    columns_to_check: Sequence[str] = ("reason_single", "reason_full"),
    # 同时支持多种错误模式（子串或正则）
    error_patterns: Optional[List[str]] = None,
    use_regex: bool = False,
    case_sensitive: bool = True,
    # pattern 之间关系：False=任一命中(OR)，True=全部命中(AND)
    match_all_patterns: bool = False,
    # 记录被删除的行
    removed_log_csv: Optional[str] = None,
):
    """
    统一在 file1 的指定列中查找错误模式；命中的 group_index 同步从 file2 删除。

    - 若 df1 缺失 group_index 列，则退化为用行索引对齐（不推荐，但保持与旧逻辑一致）
    - 支持多正则或多子串；支持 OR/AND 关系；支持大小写开关
    - 输出每个 pattern 的命中统计 & 总体统计；可选保存被删样本日志
    """
    if error_patterns is None:
        # 你给过的两类典型错误，默认都查
        error_patterns = [
            r"exec_error:NameError: name 'result' is not defined",
            r"single_exec_error:NameError: name 'shape' is not defined; pred_step_missing",
            r"empty_code"
        ]

    # 读取
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    print(f"原始数据 - 文件1: {len(df1)} 行, 文件2: {len(df2)} 行")

    # 列存在性检查（存在就用，不存在就忽略）
    cols_exist = [c for c in columns_to_check if c in df1.columns]
    if not cols_exist:
        raise ValueError(f"在文件1中未找到可用的错误信息列（期望之一：{list(columns_to_check)}）")

    # 编译正则（若使用正则）
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_patterns = []
    if use_regex:
        for p in error_patterns:
            compiled_patterns.append(re.compile(p, flags=flags))
    else:
        # 子串匹配，统一大小写开关时，将文本与模式都降/保持大小写
        compiled_patterns = error_patterns[:]  # 占位

    def match_series(s: pd.Series) -> pd.Series:
        """对单列做匹配，返回 bool Series"""
        # 转为字符串，避免 NaN
        st = s.astype(str)
        if not case_sensitive:
            st = st.str.lower()

        # 针对每个模式得到一个 bool Series
        masks = []
        if use_regex:
            for creg in compiled_patterns:
                masks.append(st.str.contains(creg))
        else:
            # 子串匹配
            patterns = compiled_patterns
            if not case_sensitive:
                patterns = [p.lower() for p in patterns]
            for p in patterns:
                masks.append(st.str.contains(re.escape(p), regex=True))

        if not masks:
            return pd.Series(False, index=s.index)

        # 将各模式的匹配按 OR/AND 组合
        if match_all_patterns:
            m = masks[0].copy()
            for mm in masks[1:]:
                m &= mm
        else:
            m = masks[0].copy()
            for mm in masks[1:]:
                m |= mm
        return m

    # 跨多列做匹配（列间 OR）
    col_masks = []
    for c in cols_exist:
        col_masks.append(match_series(df1[c]))
    hit_mask = col_masks[0].copy()
    for m in col_masks[1:]:
        hit_mask |= m

    error_rows = df1[hit_mask]
    print(f"命中错误模式的行数: {len(error_rows)}")

    if len(error_rows) == 0:
        print("未命中任何错误模式，直接原样输出。")
        df1.to_csv(output1_path, index=False)
        df2.to_csv(output2_path, index=False)
        return

    # 获得需移除的 group_index
    if 'group_index' in df1.columns:
        group_indices_to_remove = error_rows['group_index'].tolist()
    else:
        print("警告：文件1缺少 group_index 列，回退为按行索引删除（与旧逻辑一致）。")
        group_indices_to_remove = error_rows.index.tolist()

    # 输出被删样本日志
    if removed_log_csv:
        cols_for_log = ['group_index'] if 'group_index' in df1.columns else []
        cols_for_log += [c for c in cols_exist if c in df1.columns]
        error_rows[cols_for_log].to_csv(removed_log_csv, index=False)
        print(f"已保存被删样本日志: {removed_log_csv}")

    # 清洗两个文件
    df1_clean = df1[~hit_mask].copy()
    if 'group_index' in df2.columns and ('group_index' in df1.columns):
        df2_clean = df2[~df2['group_index'].isin(group_indices_to_remove)].copy()
    else:
        # 文件2没有 group_index 的极端情况：不改动 file2，但提醒
        if 'group_index' not in df2.columns:
            print("警告：文件2缺少 group_index 列，无法同步删除，保持原样。")
        df2_clean = df2.copy()

    # 逐 pattern 统计（以列优先 OR 的匹配方式统计）
    print("\n按错误模式统计（列间为 OR）:")
    for p in error_patterns:
        if use_regex:
            creg = re.compile(p, flags=flags)
            count = False
            acc_mask = pd.Series(False, index=df1.index)
            for c in cols_exist:
                acc_mask |= df1[c].astype(str).str.contains(creg)
            count = int(acc_mask.sum())
        else:
            pp = p if case_sensitive else p.lower()
            acc_mask = pd.Series(False, index=df1.index)
            for c in cols_exist:
                col = df1[c].astype(str)
                if not case_sensitive:
                    col = col.str.lower()
                acc_mask |= col.str.contains(re.escape(pp), regex=True)
            count = int(acc_mask.sum())
        print(f"  {p}: {count} 行")

    # 保存
    df1_clean.to_csv(output1_path, index=False)
    df2_clean.to_csv(output2_path, index=False)

    print(f"\n处理完成：")
    print(f"  文件1: {len(df1)} -> {len(df1_clean)} 行")
    print(f"  文件2: {len(df2)} -> {len(df2_clean)} 行")
    print(f"  共删除 group_index 数量: {len(set(group_indices_to_remove))}")

if __name__ == "__main__":
    base_dir = "/data/baiyixue/CAD/inference_result/main/Qwen3-8B-coop-sft-full-260326/std"

    f1 = os.path.join(base_dir, "cands.csv")
    f2 = os.path.join(base_dir, "summary.csv")
    o1 = os.path.join(base_dir, "cands.csv")
    o2 = os.path.join(base_dir, "summary.csv")
    removed_log = os.path.join(base_dir, "removed_by_reason.log.csv")

    remove_rows_by_reason(
        file1_path=f1,
        file2_path=f2,
        output1_path=o1,
        output2_path=o2,
        columns_to_check=("reason_single", "reason_full", "gen_error"),
        error_patterns=[
            # 你现有两种：result 未定义（full里常见），shape 未定义（single里常见）
            r"exec_error:NameError: name 'result' is not defined",
            r"single_exec_error:NameError: name 'shape' is not defined; pred_step_missing",
            r"empty_code"
        ],
        use_regex=False,         # 默认子串匹配即可
        case_sensitive=True,     # 区分大小写（你的日志通常固定大小写）
        match_all_patterns=False,# 任一命中即可删除（OR）
        removed_log_csv=removed_log,
    )
