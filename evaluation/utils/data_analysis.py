#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, re

# ========= 文件路径配置 =========
TEST_DIR = "/home/baiyixue/project/op-cad/inference_results"
MODE = "std"
MODEL = "gpt-4"
MAIN_CSV = os.path.join(TEST_DIR,MODEL,MODE,"cands.csv")
OP_CSV   = "/home/baiyixue/project/op-cad/data/prompt.csv"  # 含 group_index, op
OUT_TXT  = os.path.join(TEST_DIR,MODEL, MODE, "stats_summary.txt")
OUT_DIR  = os.path.join(TEST_DIR,MODEL,MODE, "op_split_stats")
# =================================

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 工具函数 ----------

def safe_stat(values):
    """兼容 Series / list / ndarray / 标量；返回 (mean, median)，无有效数返回 (nan, nan)"""
    if values is None:
        return (np.nan, np.nan)
    if isinstance(values, pd.Series):
        s = pd.to_numeric(values, errors="coerce").dropna()
        return (s.mean() if len(s) else np.nan, s.median() if len(s) else np.nan)
    if isinstance(values, (list, tuple, np.ndarray)):
        s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        return (s.mean() if len(s) else np.nan, s.median() if len(s) else np.nan)
    try:
        v = float(values)
        s = pd.Series([v]).dropna()
        return (float(s.mean()), float(s.median())) if len(s) else (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)

def first_success(df_sub, col):
    ok = df_sub[df_sub[col] == 1]
    return ok.iloc[0] if len(ok) > 0 else None

def _coerce_k(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["k_index"] = pd.to_numeric(out.get("k_index"), errors="coerce").fillna(0).astype(int)
    return out

def _pass_at_k(df: pd.DataFrame, ok_col: str, k_max: int) -> float:
    """
    计算 pass@k：对每个 group_index，在 k_index ∈ [0, k_max] 内是否出现成功（max==1）
    """
    if df.empty:
        return np.nan
    d = _coerce_k(df)
    sub = d[d["k_index"] <= k_max]
    if sub.empty:
        return 0.0
    by_group = sub.groupby("group_index")[ok_col].max()
    return float(by_group.mean()) if len(by_group) else np.nan

def compute_geometry_metrics(df):
    """
    计算：
      - pass@k（任意 k，等价于原来的 group 内 max）
      - pass@1（只看 k=0）
      - pass@2（看 k∈{0,1}）
      - Mean/Median CD/HD（取每组首次成功样本）
    """
    if df.empty:
        return {
            "pass@k_full": np.nan, "pass@k_single": np.nan,
            "pass1_full": np.nan, "pass1_single": np.nan,
            "pass2_full": np.nan, "pass2_single": np.nan,
            "mean_cd_full": np.nan, "med_cd_full": np.nan,
            "mean_hd_full": np.nan, "med_hd_full": np.nan,
            "mean_cd_single": np.nan, "med_cd_single": np.nan,
            "mean_hd_single": np.nan, "med_hd_single": np.nan,
        }

    # 任意 k 成功率（等同你原先的 pass@k）
    pass_at_k_full   = df.groupby("group_index")["exec_ok_full"].max().mean()
    pass_at_k_single = df.groupby("group_index")["exec_ok_single"].max().mean()

    # pass@1 / pass@2
    pass1_full   = _pass_at_k(df, "exec_ok_full",   0)
    pass1_single = _pass_at_k(df, "exec_ok_single", 0)
    pass2_full   = _pass_at_k(df, "exec_ok_full",   1)
    pass2_single = _pass_at_k(df, "exec_ok_single", 1)

    # 几何统计：每组取首次成功样本
    records_full, records_single = [], []
    for _, gdf in df.groupby("group_index"):
        gdf = gdf.sort_values("k_index")
        r_f = first_success(gdf, "exec_ok_full")
        r_s = first_success(gdf, "exec_ok_single")
        if r_f is not None: records_full.append(r_f)
        if r_s is not None: records_single.append(r_s)

    df_full   = pd.DataFrame(records_full)
    df_single = pd.DataFrame(records_single)

    mean_cd_full,  med_cd_full  = safe_stat(df_full.get("cd_full"))
    mean_hd_full,  med_hd_full  = safe_stat(df_full.get("hd_full"))
    mean_cd_single,med_cd_single= safe_stat(df_single.get("cd_single"))
    mean_hd_single,med_hd_single= safe_stat(df_single.get("hd_single"))

    return {
        "pass@k_full": pass_at_k_full,
        "pass@k_single": pass_at_k_single,
        "pass1_full": pass1_full, "pass1_single": pass1_single,
        "pass2_full": pass2_full, "pass2_single": pass2_single,
        "mean_cd_full": mean_cd_full, "med_cd_full": med_cd_full,
        "mean_hd_full": mean_hd_full, "med_hd_full": med_hd_full,
        "mean_cd_single": mean_cd_single, "med_cd_single": med_cd_single,
        "mean_hd_single": mean_hd_single, "med_hd_single": med_hd_single,
    }

def compute_cf_metrics(df):
    """Chamfer/Fillet 区块指标（加入 pass@1 / pass@2，不再输出 IR）"""
    if df.empty:
        return {
            "pass@k_cf": np.nan, "pass1_cf": np.nan, "pass2_cf": np.nan,
            "mean_cf_iou": np.nan, "med_cf_iou": np.nan
        }

    pass_at_k_cf = df.groupby("group_index")["exec_ok_full"].max().mean()
    pass1_cf = _pass_at_k(df, "exec_ok_full", 0)
    pass2_cf = _pass_at_k(df, "exec_ok_full", 1)

    cf_eval = df[df["exec_ok_full"] == 1]
    mean_cf_iou = float(cf_eval["cf_iou"].mean(skipna=True)) if "cf_iou" in cf_eval else np.nan
    med_cf_iou  = float(cf_eval["cf_iou"].median(skipna=True)) if "cf_iou" in cf_eval else np.nan

    return {"pass@k_cf": pass_at_k_cf, "pass1_cf": pass1_cf, "pass2_cf": pass2_cf,
            "mean_cf_iou": mean_cf_iou, "med_cf_iou": med_cf_iou}

def summarize_geometry(name, metrics):
    """
    输出几何指标时，将所有 CD/HD 指标 ×1000 后显示（单位转化或增强可读性）。
    """
    scale = 1000.0
    lines = [f"=== {name.upper()} ==="]
    lines.append(
        f"PASS@1: full={metrics['pass1_full']:.3f}, single={metrics['pass1_single']:.3f}"
    )
    lines.append(
        f"PASS@2: full={metrics['pass2_full']:.3f}, single={metrics['pass2_single']:.3f}"
    )
    lines.append(
        f"Geometry (×1000):\n"
        f"  Full  : Mean CD={metrics['mean_cd_full']*scale:.3f}, Median CD={metrics['med_cd_full']*scale:.3f}, "
        f"Mean HD={metrics['mean_hd_full']*scale:.3f}, Median HD={metrics['med_hd_full']*scale:.3f}\n"
        f"  Single: Mean CD={metrics['mean_cd_single']*scale:.3f}, Median CD={metrics['med_cd_single']*scale:.3f}, "
        f"Mean HD={metrics['mean_hd_single']*scale:.3f}, Median HD={metrics['med_hd_single']*scale:.3f}"
    )
    return "\n".join(lines) + "\n\n"

def summarize_cf_block(cf_metrics):
    return (
        f"=== CF 特征指标 ===\n"
        f"pass@1(cf): {cf_metrics['pass1_cf']:.3f}, pass@2(cf): {cf_metrics['pass2_cf']:.3f}\n"
        f"Mean cf_iou={cf_metrics['mean_cf_iou']:.3f}, Median cf_iou={cf_metrics['med_cf_iou']:.3f}\n\n"
    )

# ====== 失败分类（保持你确认过的规则） ======

SYNTAX_KEYS = [
    # Python/解析相关（语法优先）
    "pyparsing.exceptions.parseexception", "parseexception", "syntaxerror",
    "typeerror", "attributeerror", "nameerror", "indexerror", "keyerror",
    # 常见语义/参数类
    "unsupported operand", "missing 1 required positional argument",
    "could not convert string", "invalid literal", "division by zero",
]

GEOM_KEYS = [
    # 几何/occ/cq相关
    "ocp.standard.", "workplane object must have at least one solid", "no solid on the stack",
    "empty mesh", "result_is_none", "non_positive_volume", "boolean operation failed",
    "topods_shape is null", "shape is null", "bopalgo", "sewing failed",
    "cannot compute mass", "gp_axis", "invalid shape",
    # 管线中的几何类 reason
    "pred_step_missing", "metric_exception:runtimeerror:empty mesh",
]

def _norm_reason(x):
    return (str(x) if x is not None else "").strip().lower()

def _split_reason_tokens(r: str):
    # 以 ; 或 | 或换行切片，兼容复合 reason
    return [t.strip() for t in re.split(r"[;\|\n]+", r) if t.strip()]

def classify_error_type(reason: str) -> str:
    """
    语法优先：先命中 SYNTAX_KEYS → 'syntax'；否则若命中 GEOM_KEYS → 'geometry'；否则默认 'geometry'
    """
    r = _norm_reason(reason)
    if not r:
        return "geometry"
    tokens = _split_reason_tokens(r)

    # 整体匹配
    if any(k in r for k in SYNTAX_KEYS):
        return "syntax"
    if any(k in r for k in GEOM_KEYS):
        return "geometry"

    # 分 token 匹配
    for t in tokens:
        if any(k in t for k in SYNTAX_KEYS):
            return "syntax"
    for t in tokens:
        if any(k in t for k in GEOM_KEYS):
            return "geometry"

    return "geometry"  # 兜底

def classify_fail_reasons(df):
    """
    针对执行失败样本做错误类型分类：
    - 若 exec_ok_single==0 → 用 reason_single 判断；若空则用 reason_full
    - 若 exec_ok_single 为空 → 用 reason_full 代替
    - 若 exec_ok_full==0 且 single 也失败 → 继承 single 类型；否则用 reason_full
    返回仅包含“至少一侧执行失败”的 dataframe
    """

    def _strip_exec_prefix(x):
        r = (str(x) if x is not None else "").strip()
        return r[len("exec_error:"):] if r.lower().startswith("exec_error:") else r

    def _is_failed(col, frame):
        s = pd.to_numeric(frame.get(col), errors="coerce")
        return s.eq(0)

    out = df.copy()
    out["single_failed"] = _is_failed("exec_ok_single", out)
    out["full_failed"]   = _is_failed("exec_ok_full",   out)
    out = out[out["single_failed"] | out["full_failed"]].copy()

    def _class_single(row):
        if row["single_failed"]:
            rs = row.get("reason_single", "")
            if pd.isna(rs) or not str(rs).strip():
                rs = row.get("reason_full", "")
            return classify_error_type(_strip_exec_prefix(rs))
        elif pd.isna(row.get("exec_ok_single")):
            return classify_error_type(_strip_exec_prefix(row.get("reason_full", "")))
        else:
            return np.nan

    def _class_full(row):
        if not row["full_failed"]:
            return np.nan
        if row["single_failed"]:
            return row["single_fail_kind"]
        return classify_error_type(_strip_exec_prefix(row.get("reason_full", "")))

    out["single_fail_kind"] = out.apply(_class_single, axis=1)
    out["full_fail_kind"]   = out.apply(_class_full,   axis=1)
    return out

def compute_fail_ratios(df_fail, col_kind):
    """计算失败样本中（仅失败行）几何/语法占比；返回 (geometry_ratio, syntax_ratio)"""
    df_valid = df_fail[df_fail[col_kind].isin(["geometry", "syntax"])].copy()
    if df_valid.empty:
        return np.nan, np.nan
    total = len(df_valid)
    geom_n = (df_valid[col_kind] == "geometry").sum()
    syntax_n = (df_valid[col_kind] == "syntax").sum()
    return geom_n / total, syntax_n / total

def summarize_fail_ratios_txt(tag, g_s, s_s, g_f, s_f):
    return (
        f"{tag}:\n"
        f"  Single → Geometry={g_s:.3f}, Syntax={s_s:.3f}\n"
        f"  Full   → Geometry={g_f:.3f}, Syntax={s_f:.3f}\n\n"
    )

# ---------- 主流程 ----------

def main():
    # 读取并合并 op 映射
    df_raw = pd.read_csv(MAIN_CSV)
    ops = pd.read_csv(OP_CSV)
    df = df_raw.merge(ops, on="group_index", how="left")  # 兼容 pandas 老版本: on="group_index"

    # === 1. OVERALL（含 CF，但几何不含 CF）===
    overall_geo_all = compute_geometry_metrics(df)
    df_no_cf = df[df["op"] != "chamfer_fillet"].copy()
    overall_geo_nocf = compute_geometry_metrics(df_no_cf)
    # 用非 CF 的几何统计覆盖 overall
    for key in [
        "mean_cd_full", "med_cd_full", "mean_hd_full", "med_hd_full",
        "mean_cd_single", "med_cd_single", "mean_hd_single", "med_hd_single",
    ]:
        overall_geo_all[key] = overall_geo_nocf[key]
    overall_geo = overall_geo_all

    # CF 区块
    df_cf = df[df["op"] == "chamfer_fillet"].copy()
    overall_cf = compute_cf_metrics(df_cf)

    overall_txt = summarize_geometry("Overall", overall_geo) + summarize_cf_block(overall_cf)

    # === 2. Extrude ===
    df_extrude = df[df["op"] == "extrude"].copy()
    extrude_metrics = compute_geometry_metrics(df_extrude)
    extrude_txt = summarize_geometry("Extrude", extrude_metrics)
    df_extrude.to_csv(os.path.join(OUT_DIR, "extrude_subset.csv"), index=False)

    # === 3. Revolve ===
    df_revolve = df[df["op"] == "revolve"].copy()
    revolve_metrics = compute_geometry_metrics(df_revolve)
    revolve_txt = summarize_geometry("Revolve", revolve_metrics)
    df_revolve.to_csv(os.path.join(OUT_DIR, "revolve_subset.csv"), index=False)

    # === 4. Chamfer/Fillet ===
    cf_txt = summarize_geometry("Chamfer_Fillet", compute_geometry_metrics(df_cf))
    cf_txt += summarize_cf_block(overall_cf)
    df_cf.to_csv(os.path.join(OUT_DIR, "chamfer_fillet_subset.csv"), index=False)

    # === 5. 失败样本（仅失败）语法/几何占比 ===
    df_fail = classify_fail_reasons(df)

    # overall
    g_ratio_single, s_ratio_single = compute_fail_ratios(df_fail, "single_fail_kind")
    g_ratio_full,   s_ratio_full   = compute_fail_ratios(df_fail, "full_fail_kind")

    # 分 op
    ops_list = ["extrude", "revolve", "chamfer_fillet"]
    rows_csv = []
    fail_txt = "=== 执行失败样本错误类型占比（仅失败）===\n"
    fail_txt += summarize_fail_ratios_txt("OVERALL", g_ratio_single, s_ratio_single, g_ratio_full, s_ratio_full)
    rows_csv.append({"scope":"OVERALL","part":"single","geometry_ratio":g_ratio_single,"syntax_ratio":s_ratio_single})
    rows_csv.append({"scope":"OVERALL","part":"full",  "geometry_ratio":g_ratio_full,  "syntax_ratio":s_ratio_full})

    for op in ops_list:
        sub = df_fail[df_fail["op"] == op]
        g_s, s_s = compute_fail_ratios(sub, "single_fail_kind")
        g_f, s_f = compute_fail_ratios(sub, "full_fail_kind")
        fail_txt += summarize_fail_ratios_txt(op.upper(), g_s, s_s, g_f, s_f)
        rows_csv.append({"scope":op,"part":"single","geometry_ratio":g_s,"syntax_ratio":s_s})
        rows_csv.append({"scope":op,"part":"full",  "geometry_ratio":g_f,"syntax_ratio":s_f})

    # 导出失败占比 CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    fail_csv = os.path.join(OUT_DIR, "fail_reason_ratios.csv")
    pd.DataFrame(rows_csv).to_csv(fail_csv, index=False)

    # === 6. 写入汇总 TXT（把失败占比一并追加进去）===
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(overall_txt)
        f.write(extrude_txt)
        f.write(revolve_txt)
        # 如需包含 cf_txt 可取消下一行注释
        # f.write(cf_txt)
        f.write(fail_txt)

    print("✅ 指标统计完成（含 pass@1/2 与失败样本语法/几何占比）")
    print(f"TXT 汇总: {OUT_TXT}")
    print(f"子集 CSV 输出: {OUT_DIR}")
    print(f"失败占比 CSV: {fail_csv}")

if __name__ == "__main__":
    main()
