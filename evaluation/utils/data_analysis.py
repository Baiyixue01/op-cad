#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, re, argparse
from pathlib import Path

# ========= 默认根路径配置（可被命令行覆盖） =========
TEST_DIR_DEFAULT = "/data/baiyixue/CAD/inference_result/main/llama_3.1_8b_coop_sft_full"
OP_CSV_DEFAULT   = "/home/baiyixue/project/op-cad/data/prompt.csv"  # 含 group_index, op
TXT_ROOT_DEFAULT = "/home/baiyixue/project/op-cad/results_v2_large"
# ======================================

# ---------- 工具函数（保持你原版逻辑） ----------
def safe_stat(values):
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
    if df.empty:
        return np.nan
    d = _coerce_k(df)
    sub = d[d["k_index"] <= k_max]
    if sub.empty:
        return 0.0
    by_group = sub.groupby("group_index")[ok_col].max()
    return float(by_group.mean()) if len(by_group) else np.nan

def compute_geometry_metrics(df):
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

    pass_at_k_full   = df.groupby("group_index")["exec_ok_full"].max().mean()
    pass_at_k_single = df.groupby("group_index")["exec_ok_single"].max().mean()

    pass1_full   = _pass_at_k(df, "exec_ok_full",   0)
    pass1_single = _pass_at_k(df, "exec_ok_single", 0)
    pass2_full   = _pass_at_k(df, "exec_ok_full",   1)
    pass2_single = _pass_at_k(df, "exec_ok_single", 1)

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
    scale = 1000.0
    lines = [f"=== {name.upper()} ==="]
    lines.append(f"PASS@1: full={metrics['pass1_full']:.3f}, single={metrics['pass1_single']:.3f}")
    lines.append(f"PASS@2: full={metrics['pass2_full']:.3f}, single={metrics['pass2_single']:.3f}")
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

SYNTAX_KEYS = [
    "pyparsing.exceptions.parseexception", "parseexception", "syntaxerror",
    "typeerror", "attributeerror", "nameerror", "indexerror", "keyerror",
    "unsupported operand", "missing 1 required positional argument",
    "could not convert string", "invalid literal", "division by zero",
]

GEOM_KEYS = [
    "ocp.standard.", "workplane object must have at least one solid", "no solid on the stack",
    "empty mesh", "result_is_none", "non_positive_volume", "boolean operation failed",
    "topods_shape is null", "shape is null", "bopalgo", "sewing failed",
    "cannot compute mass", "gp_axis", "invalid shape",
    "pred_step_missing", "metric_exception:runtimeerror:empty mesh",
]

def _norm_reason(x):
    return (str(x) if x is not None else "").strip().lower()

def _split_reason_tokens(r: str):
    return [t.strip() for t in re.split(r"[;\|\n]+", r) if t.strip()]

def classify_error_type(reason: str) -> str:
    r = _norm_reason(reason)
    if not r:
        return "geometry"
    tokens = _split_reason_tokens(r)
    if any(k in r for k in SYNTAX_KEYS): return "syntax"
    if any(k in r for k in GEOM_KEYS):   return "geometry"
    for t in tokens:
        if any(k in t for k in SYNTAX_KEYS): return "syntax"
    for t in tokens:
        if any(k in t for k in GEOM_KEYS):   return "geometry"
    return "geometry"

def classify_fail_reasons(df):
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

# ---------- 评测流程 ----------
def evaluate_one(main_csv: Path, model: str, mode: str, ops_map: pd.DataFrame, txt_root: Path):
    out_dir = main_csv.parent / "op_split_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = txt_root / f"{model}_{mode}.txt"

    # 读数据；确保 group_index 按字符串合并
    df_raw = pd.read_csv(main_csv, dtype={"group_index": str})
    ops = ops_map.copy()
    if "group_index" not in ops.columns:
        raise KeyError("OP_CSV 缺少 group_index 列")
    ops["group_index"] = ops["group_index"].astype(str)

    df = df_raw.merge(ops, on="group_index", how="left")

    # overall（几何不含 CF）
    overall_geo_all = compute_geometry_metrics(df)
    df_no_cf = df[df["op"] != "chamfer_fillet"].copy()
    overall_geo_nocf = compute_geometry_metrics(df_no_cf)
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

    # Extrude
    df_extrude = df[df["op"] == "extrude"].copy()
    extrude_metrics = compute_geometry_metrics(df_extrude)
    extrude_txt = summarize_geometry("Extrude", extrude_metrics)
    df_extrude.to_csv(out_dir / "extrude_subset.csv", index=False)

    # Revolve
    df_revolve = df[df["op"] == "revolve"].copy()
    revolve_metrics = compute_geometry_metrics(df_revolve)
    revolve_txt = summarize_geometry("Revolve", revolve_metrics)
    df_revolve.to_csv(out_dir / "revolve_subset.csv", index=False)

    # CF 子集
    df_cf.to_csv(out_dir / "chamfer_fillet_subset.csv", index=False)

    # 失败样本分类与占比
    df_fail = classify_fail_reasons(df)
    g_ratio_single, s_ratio_single = compute_fail_ratios(df_fail, "single_fail_kind")
    g_ratio_full,   s_ratio_full   = compute_fail_ratios(df_fail, "full_fail_kind")

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
        rows_csv.append({"scope":op,"part":"full",  "geometry_ratio":g_f,  "syntax_ratio":s_f})

    # 导出失败占比 CSV
    (out_dir / "fail_reason_ratios.csv").write_text(
        pd.DataFrame(rows_csv).to_csv(index=False), encoding="utf-8"
    )

    # 汇总 TXT
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(overall_txt)
        f.write(extrude_txt)
        f.write(revolve_txt)
        f.write(fail_txt)

    print(f"✅ 完成：{model}/{mode}  ->  {out_txt}")

def infer_model_mode(cands_path: Path, root_dir: Path):
    """
    支持两种目录形态：
      1) ROOT/MODEL/MODE/cands.csv          -> model=MODEL, mode=MODE
      2) (ROOT即模型目录) ROOT/MODE/cands.csv -> model=ROOT.name, mode=MODE
    如层级更深，退回从末尾取：.../<model>/<mode>/cands.csv
    """
    try:
        rel = cands_path.relative_to(root_dir)
        parts = rel.parts
    except ValueError:
        # 不可相对化时，直接从末尾两级推断
        parts = cands_path.parts

    # 优先识别末尾两级
    if len(parts) >= 3:
        # .../<model>/<mode>/cands.csv
        model = parts[-3]
        mode  = parts[-2]
    elif len(parts) == 2:
        # <mode>/cands.csv，model 用 root 目录名
        model = root_dir.name
        mode  = parts[-2]
    else:
        # 兜底：用父级名与祖父级名
        mode  = cands_path.parent.name
        grand = cands_path.parent.parent
        model = grand.name if grand else root_dir.name
    return model, mode

# ---------- 主入口：遍历全部模型/模式 ----------
def main():
    parser = argparse.ArgumentParser(description="OP-CAD 批量评测（支持总目录或单模型目录）")
    parser.add_argument("--input_dir", default=TEST_DIR_DEFAULT, help="包含结果的目录（总目录或单模型目录）")
    parser.add_argument("--op_csv",   default=OP_CSV_DEFAULT,    help="含 group_index 与 op 的CSV")
    parser.add_argument("--txt_root", default=TXT_ROOT_DEFAULT,  help="汇总 TXT 输出根目录")
    args = parser.parse_args()

    TEST_DIR = args.input_dir
    OP_CSV   = args.op_csv
    TXT_ROOT = args.txt_root

    os.makedirs(TXT_ROOT, exist_ok=True)

    ops_map = pd.read_csv(OP_CSV, dtype={"group_index": str})
    test_dir = Path(TEST_DIR)

    # 匹配 pattern: input_dir/**/cands.csv
    cands_list = list(test_dir.rglob("cands.csv"))
    if not cands_list:
        print(f"未在 {TEST_DIR} 下找到任何 cands.csv")
        return

    for cands in sorted(cands_list):
        try:
            model, mode = infer_model_mode(cands, test_dir)
            evaluate_one(cands, model, mode, ops_map, Path(TXT_ROOT))
        except Exception as e:
            print(f"[失败] {cands}: {e}")

if __name__ == "__main__":
    main()



# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pandas as pd
# import numpy as np
# import os, re
# from pathlib import Path

# # ========= 根路径配置（只改这里） =========
# TEST_DIR = "/home/baiyixue/project/op-cad/inference_results"
# OP_CSV   = "/home/baiyixue/project/op-cad/data/prompt.csv"  # 含 group_index, op
# TXT_ROOT = "/home/baiyixue/project/op-cad/results"
# # ======================================

# os.makedirs(TXT_ROOT, exist_ok=True)

# # ---------- 工具函数（保持你原版逻辑） ----------
# def safe_stat(values):
#     if values is None:
#         return (np.nan, np.nan)
#     if isinstance(values, pd.Series):
#         s = pd.to_numeric(values, errors="coerce").dropna()
#         return (s.mean() if len(s) else np.nan, s.median() if len(s) else np.nan)
#     if isinstance(values, (list, tuple, np.ndarray)):
#         s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
#         return (s.mean() if len(s) else np.nan, s.median() if len(s) else np.nan)
#     try:
#         v = float(values)
#         s = pd.Series([v]).dropna()
#         return (float(s.mean()), float(s.median())) if len(s) else (np.nan, np.nan)
#     except Exception:
#         return (np.nan, np.nan)

# def first_success(df_sub, col):
#     ok = df_sub[df_sub[col] == 1]
#     return ok.iloc[0] if len(ok) > 0 else None

# def _coerce_k(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     out["k_index"] = pd.to_numeric(out.get("k_index"), errors="coerce").fillna(0).astype(int)
#     return out

# def _pass_at_k(df: pd.DataFrame, ok_col: str, k_max: int) -> float:
#     if df.empty:
#         return np.nan
#     d = _coerce_k(df)
#     sub = d[d["k_index"] <= k_max]
#     if sub.empty:
#         return 0.0
#     by_group = sub.groupby("group_index")[ok_col].max()
#     return float(by_group.mean()) if len(by_group) else np.nan

# def compute_geometry_metrics(df):
#     if df.empty:
#         return {
#             "pass@k_full": np.nan, "pass@k_single": np.nan,
#             "pass1_full": np.nan, "pass1_single": np.nan,
#             "pass2_full": np.nan, "pass2_single": np.nan,
#             "mean_cd_full": np.nan, "med_cd_full": np.nan,
#             "mean_hd_full": np.nan, "med_hd_full": np.nan,
#             "mean_cd_single": np.nan, "med_cd_single": np.nan,
#             "mean_hd_single": np.nan, "med_hd_single": np.nan,
#         }

#     pass_at_k_full   = df.groupby("group_index")["exec_ok_full"].max().mean()
#     pass_at_k_single = df.groupby("group_index")["exec_ok_single"].max().mean()

#     pass1_full   = _pass_at_k(df, "exec_ok_full",   0)
#     pass1_single = _pass_at_k(df, "exec_ok_single", 0)
#     pass2_full   = _pass_at_k(df, "exec_ok_full",   1)
#     pass2_single = _pass_at_k(df, "exec_ok_single", 1)

#     records_full, records_single = [], []
#     for _, gdf in df.groupby("group_index"):
#         gdf = gdf.sort_values("k_index")
#         r_f = first_success(gdf, "exec_ok_full")
#         r_s = first_success(gdf, "exec_ok_single")
#         if r_f is not None: records_full.append(r_f)
#         if r_s is not None: records_single.append(r_s)

#     df_full   = pd.DataFrame(records_full)
#     df_single = pd.DataFrame(records_single)

#     mean_cd_full,  med_cd_full  = safe_stat(df_full.get("cd_full"))
#     mean_hd_full,  med_hd_full  = safe_stat(df_full.get("hd_full"))
#     mean_cd_single,med_cd_single= safe_stat(df_single.get("cd_single"))
#     mean_hd_single,med_hd_single= safe_stat(df_single.get("hd_single"))

#     return {
#         "pass@k_full": pass_at_k_full,
#         "pass@k_single": pass_at_k_single,
#         "pass1_full": pass1_full, "pass1_single": pass1_single,
#         "pass2_full": pass2_full, "pass2_single": pass2_single,
#         "mean_cd_full": mean_cd_full, "med_cd_full": med_cd_full,
#         "mean_hd_full": mean_hd_full, "med_hd_full": med_hd_full,
#         "mean_cd_single": mean_cd_single, "med_cd_single": med_cd_single,
#         "mean_hd_single": mean_hd_single, "med_hd_single": med_hd_single,
#     }

# def compute_cf_metrics(df):
#     if df.empty:
#         return {
#             "pass@k_cf": np.nan, "pass1_cf": np.nan, "pass2_cf": np.nan,
#             "mean_cf_iou": np.nan, "med_cf_iou": np.nan
#         }

#     pass_at_k_cf = df.groupby("group_index")["exec_ok_full"].max().mean()
#     pass1_cf = _pass_at_k(df, "exec_ok_full", 0)
#     pass2_cf = _pass_at_k(df, "exec_ok_full", 1)

#     cf_eval = df[df["exec_ok_full"] == 1]
#     mean_cf_iou = float(cf_eval["cf_iou"].mean(skipna=True)) if "cf_iou" in cf_eval else np.nan
#     med_cf_iou  = float(cf_eval["cf_iou"].median(skipna=True)) if "cf_iou" in cf_eval else np.nan

#     return {
#         "pass@k_cf": pass_at_k_cf,
#         "pass1_cf": pass1_cf,
#         "pass2_cf": pass2_cf,
#         "mean_cf_iou": mean_cf_iou,
#         "med_cf_iou": med_cf_iou
#     }

# # ========= 新增：std/coop 共同通过样本上的 CF 统计 =========
# def compute_joint_cf_metrics(df_std_cf: pd.DataFrame, df_coop_cf: pd.DataFrame):
#     """
#     在 CF 子集上，计算“std 和 coop 都通过”的样本上的 CF-IoU 平均值和中位数。

#     df_std_cf / df_coop_cf: 只包含 op == 'chamfer_fillet' 的行。
#     """
#     if df_std_cf.empty or df_coop_cf.empty:
#         return {
#             "num_joint_success": 0,
#             "std_joint_mean_cf": np.nan,
#             "std_joint_med_cf": np.nan,
#             "coop_joint_mean_cf": np.nan,
#             "coop_joint_med_cf": np.nan,
#         }

#     # 各模式下按 group_index 聚合是否成功
#     std_pass = df_std_cf.groupby("group_index")["exec_ok_full"].max()
#     coop_pass = df_coop_cf.groupby("group_index")["exec_ok_full"].max()

#     std_ok_ids = set(std_pass[std_pass == 1].index.astype(str))
#     coop_ok_ids = set(coop_pass[coop_pass == 1].index.astype(str))

#     # 共同通过的样本
#     joint_ids = std_ok_ids & coop_ok_ids
#     if len(joint_ids) == 0:
#         return {
#             "num_joint_success": 0,
#             "std_joint_mean_cf": np.nan,
#             "std_joint_med_cf": np.nan,
#             "coop_joint_mean_cf": np.nan,
#             "coop_joint_med_cf": np.nan,
#         }

#     def _agg(df_cf: pd.DataFrame):
#         sub = df_cf[
#             (df_cf["group_index"].astype(str).isin(joint_ids)) &
#             (df_cf["exec_ok_full"] == 1)
#         ].copy()
#         if sub.empty or "cf_iou" not in sub.columns:
#             return np.nan, np.nan
#         s = pd.to_numeric(sub["cf_iou"], errors="coerce").dropna()
#         if s.empty:
#             return np.nan, np.nan
#         return float(s.mean()), float(s.median())

#     std_mean, std_med = _agg(df_std_cf)
#     coop_mean, coop_med = _agg(df_coop_cf)

#     return {
#         "num_joint_success": len(joint_ids),
#         "std_joint_mean_cf": std_mean,
#         "std_joint_med_cf": std_med,
#         "coop_joint_mean_cf": coop_mean,
#         "coop_joint_med_cf": coop_med,
#     }

# def summarize_geometry(name, metrics):
#     scale = 1000.0
#     lines = [f"=== {name.upper()} ==="]
#     lines.append(f"PASS@1: full={metrics['pass1_full']:.3f}, single={metrics['pass1_single']:.3f}")
#     lines.append(f"PASS@2: full={metrics['pass2_full']:.3f}, single={metrics['pass2_single']:.3f}")
#     lines.append(
#         f"Geometry (×1000):\n"
#         f"  Full  : Mean CD={metrics['mean_cd_full']*scale:.3f}, Median CD={metrics['med_cd_full']*scale:.3f}, "
#         f"Mean HD={metrics['mean_hd_full']*scale:.3f}, Median HD={metrics['med_hd_full']*scale:.3f}\n"
#         f"  Single: Mean CD={metrics['mean_cd_single']*scale:.3f}, Median CD={metrics['med_cd_single']*scale:.3f}, "
#         f"Mean HD={metrics['mean_hd_single']*scale:.3f}, Median HD={metrics['med_hd_single']*scale:.3f}"
#     )
#     return "\n".join(lines) + "\n\n"

# def summarize_cf_block(cf_metrics):
#     return (
#         f"=== CF 特征指标 ===\n"
#         f"pass@1(cf): {cf_metrics['pass1_cf']:.3f}, pass@2(cf): {cf_metrics['pass2_cf']:.3f}\n"
#         f"Mean cf_iou={cf_metrics['mean_cf_iou']:.3f}, Median cf_iou={cf_metrics['med_cf_iou']:.3f}\n\n"
#     )

# SYNTAX_KEYS = [
#     "pyparsing.exceptions.parseexception", "parseexception", "syntaxerror",
#     "typeerror", "attributeerror", "nameerror", "indexerror", "keyerror",
#     "unsupported operand", "missing 1 required positional argument",
#     "could not convert string", "invalid literal", "division by zero",
# ]

# GEOM_KEYS = [
#     "ocp.standard.", "workplane object must have at least one solid", "no solid on the stack",
#     "empty mesh", "result_is_none", "non_positive_volume", "boolean operation failed",
#     "topods_shape is null", "shape is null", "bopalgo", "sewing failed",
#     "cannot compute mass", "gp_axis", "invalid shape",
#     "pred_step_missing", "metric_exception:runtimeerror:empty mesh",
# ]

# def _norm_reason(x):
#     return (str(x) if x is not None else "").strip().lower()

# def _split_reason_tokens(r: str):
#     return [t.strip() for t in re.split(r"[;\|\n]+", r) if t.strip()]

# def classify_error_type(reason: str) -> str:
#     r = _norm_reason(reason)
#     if not r:
#         return "geometry"
#     tokens = _split_reason_tokens(r)
#     if any(k in r for k in SYNTAX_KEYS): return "syntax"
#     if any(k in r for k in GEOM_KEYS):   return "geometry"
#     for t in tokens:
#         if any(k in t for k in SYNTAX_KEYS): return "syntax"
#     for t in tokens:
#         if any(k in t for k in GEOM_KEYS):   return "geometry"
#     return "geometry"

# def classify_fail_reasons(df):
#     def _strip_exec_prefix(x):
#         r = (str(x) if x is not None else "").strip()
#         return r[len("exec_error:"):] if r.lower().startswith("exec_error:") else r

#     def _is_failed(col, frame):
#         s = pd.to_numeric(frame.get(col), errors="coerce")
#         return s.eq(0)

#     out = df.copy()
#     out["single_failed"] = _is_failed("exec_ok_single", out)
#     out["full_failed"]   = _is_failed("exec_ok_full",   out)
#     out = out[out["single_failed"] | out["full_failed"]].copy()

#     def _class_single(row):
#         if row["single_failed"]:
#             rs = row.get("reason_single", "")
#             if pd.isna(rs) or not str(rs).strip():
#                 rs = row.get("reason_full", "")
#             return classify_error_type(_strip_exec_prefix(rs))
#         elif pd.isna(row.get("exec_ok_single")):
#             return classify_error_type(_strip_exec_prefix(row.get("reason_full", "")))
#         else:
#             return np.nan

#     def _class_full(row):
#         if not row["full_failed"]:
#             return np.nan
#         if row["single_failed"]:
#             return row["single_fail_kind"]
#         return classify_error_type(_strip_exec_prefix(row.get("reason_full", "")))

#     out["single_fail_kind"] = out.apply(_class_single, axis=1)
#     out["full_fail_kind"]   = out.apply(_class_full,   axis=1)
#     return out

# def compute_fail_ratios(df_fail, col_kind):
#     df_valid = df_fail[df_fail[col_kind].isin(["geometry", "syntax"])].copy()
#     if df_valid.empty:
#         return np.nan, np.nan
#     total = len(df_valid)
#     geom_n = (df_valid[col_kind] == "geometry").sum()
#     syntax_n = (df_valid[col_kind] == "syntax").sum()
#     return geom_n / total, syntax_n / total

# def summarize_fail_ratios_txt(tag, g_s, s_s, g_f, s_f):
#     return (
#         f"{tag}:\n"
#         f"  Single → Geometry={g_s:.3f}, Syntax={s_s:.3f}\n"
#         f"  Full   → Geometry={g_f:.3f}, Syntax={s_f:.3f}\n\n"
#     )

# # ---------- 单个模型/模式的评测流程 ----------
# def evaluate_one(main_csv: Path, model: str, mode: str, ops_map: pd.DataFrame):
#     out_dir = main_csv.parent / "op_split_stats"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_txt = Path(TXT_ROOT) / f"{model}_{mode}.txt"

#     # 读数据；确保 group_index 按字符串合并，避免类型不一致
#     df_raw = pd.read_csv(main_csv, dtype={"group_index": str})
#     ops = ops_map.copy()
#     if "group_index" not in ops.columns:
#         raise KeyError("OP_CSV 缺少 group_index 列")
#     ops["group_index"] = ops["group_index"].astype(str)

#     df = df_raw.merge(ops, on="group_index", how="left")

#     # overall（几何不含 CF）
#     overall_geo_all = compute_geometry_metrics(df)
#     df_no_cf = df[df["op"] != "chamfer_fillet"].copy()
#     overall_geo_nocf = compute_geometry_metrics(df_no_cf)
#     for key in [
#         "mean_cd_full", "med_cd_full", "mean_hd_full", "med_hd_full",
#         "mean_cd_single", "med_cd_single", "mean_hd_single", "med_hd_single",
#     ]:
#         overall_geo_all[key] = overall_geo_nocf[key]
#     overall_geo = overall_geo_all

#     # CF 区块
#     df_cf = df[df["op"] == "chamfer_fillet"].copy()
#     overall_cf = compute_cf_metrics(df_cf)

#     overall_txt = summarize_geometry("Overall", overall_geo) + summarize_cf_block(overall_cf)

#     # Extrude
#     df_extrude = df[df["op"] == "extrude"].copy()
#     extrude_metrics = compute_geometry_metrics(df_extrude)
#     extrude_txt = summarize_geometry("Extrude", extrude_metrics)
#     df_extrude.to_csv(out_dir / "extrude_subset.csv", index=False)

#     # Revolve
#     df_revolve = df[df["op"] == "revolve"].copy()
#     revolve_metrics = compute_geometry_metrics(df_revolve)
#     revolve_txt = summarize_geometry("Revolve", revolve_metrics)
#     df_revolve.to_csv(out_dir / "revolve_subset.csv", index=False)

#     # CF 子集导出（如果后面还要用）
#     df_cf.to_csv(out_dir / "chamfer_fillet_subset.csv", index=False)

#     # 失败样本分类与占比
#     df_fail = classify_fail_reasons(df)
#     g_ratio_single, s_ratio_single = compute_fail_ratios(df_fail, "single_fail_kind")
#     g_ratio_full,   s_ratio_full   = compute_fail_ratios(df_fail, "full_fail_kind")

#     ops_list = ["extrude", "revolve", "chamfer_fillet"]
#     rows_csv = []
#     fail_txt = "=== 执行失败样本错误类型占比（仅失败）===\n"
#     fail_txt += summarize_fail_ratios_txt("OVERALL", g_ratio_single, s_ratio_single, g_ratio_full, s_ratio_full)
#     rows_csv.append({"scope":"OVERALL","part":"single","geometry_ratio":g_ratio_single,"syntax_ratio":s_ratio_single})
#     rows_csv.append({"scope":"OVERALL","part":"full",  "geometry_ratio":g_ratio_full,  "syntax_ratio":s_ratio_full})

#     for op in ops_list:
#         sub = df_fail[df_fail["op"] == op]
#         g_s, s_s = compute_fail_ratios(sub, "single_fail_kind")
#         g_f, s_f = compute_fail_ratios(sub, "full_fail_kind")
#         fail_txt += summarize_fail_ratios_txt(op.upper(), g_s, s_s, g_f, s_f)
#         rows_csv.append({"scope":op,"part":"single","geometry_ratio":g_s,"syntax_ratio":s_s})
#         rows_csv.append({"scope":op,"part":"full",  "geometry_ratio":g_f,"syntax_ratio":s_f})

#     # 导出失败占比 CSV
#     (out_dir / "fail_reason_ratios.csv").write_text(
#         pd.DataFrame(rows_csv).to_csv(index=False), encoding="utf-8"
#     )

#     # 汇总 TXT
#     with open(out_txt, "w", encoding="utf-8") as f:
#         f.write(overall_txt)
#         f.write(extrude_txt)
#         f.write(revolve_txt)
#         # 如需 CF 几何段落，可在这里追加
#         # f.write(cf_txt)
#         f.write(fail_txt)

#     print(f"✅ 完成：{model}/{mode}  ->  {out_txt}")

# # ---------- 主入口：遍历全部模型/模式 ----------
# def main():
#     ops_map = pd.read_csv(OP_CSV, dtype={"group_index": str})
#     test_dir = Path(TEST_DIR)

#     # 1) 收集所有 cands.csv 路径，并按 model → mode 归类
#     cands_list = list(test_dir.rglob("cands.csv"))
#     if not cands_list:
#         print(f"未在 {TEST_DIR} 下找到任何 cands.csv")
#         return

#     model_mode_paths = {}  # {model: {mode: Path}}
#     for cands in sorted(cands_list):
#         try:
#             rel = cands.relative_to(test_dir)        # e.g. MODEL/MODE/cands.csv
#             parts = rel.parts
#             if len(parts) < 3:
#                 print(f"[跳过] 非期望层级：{rel}")
#                 continue
#             model, mode = parts[0], parts[1]
#             model_mode_paths.setdefault(model, {})[mode] = cands
#         except Exception as e:
#             print(f"[解析路径失败] {cands}: {e}")

#     # 2) 先照旧对每个 (model, mode) 做 evaluate_one
#     for model, mode_dict in model_mode_paths.items():
#         for mode, cands in mode_dict.items():
#             try:
#                 evaluate_one(cands, model, mode, ops_map)
#             except Exception as e:
#                 print(f"[评测失败] {model}/{mode} @ {cands}: {e}")

#     # 3) 对有 std 和 coop 两种模式的模型，计算“共同通过”的 CF 平均值
#     joint_rows = []
#     for model, mode_dict in model_mode_paths.items():
#         if "std" not in mode_dict or "coop" not in mode_dict:
#             continue  # 没有成对的 std/coop，就跳过

#         path_std  = mode_dict["std"]
#         path_coop = mode_dict["coop"]

#         try:
#             df_std = pd.read_csv(path_std, dtype={"group_index": str})
#             df_coop = pd.read_csv(path_coop, dtype={"group_index": str})

#             ops = ops_map.copy()
#             ops["group_index"] = ops["group_index"].astype(str)

#             df_std = df_std.merge(ops, on="group_index", how="left")
#             df_coop = df_coop.merge(ops, on="group_index", how="left")

#             df_std_cf = df_std[df_std["op"] == "chamfer_fillet"].copy()
#             df_coop_cf = df_coop[df_coop["op"] == "chamfer_fillet"].copy()

#             metrics = compute_joint_cf_metrics(df_std_cf, df_coop_cf)
#             metrics["model"] = model

#             joint_rows.append(metrics)
#             print(
#                 f"✅ Joint CF: {model} — "
#                 f"joint={metrics['num_joint_success']} "
#                 f"std_mean={metrics['std_joint_mean_cf']:.3f}, "
#                 f"coop_mean={metrics['coop_joint_mean_cf']:.3f}"
#             )
#         except Exception as e:
#             print(f"[Joint CF 计算失败] {model}: {e}")

#     # 4) 导出 joint CF 结果
#     if joint_rows:
#         joint_df = pd.DataFrame(joint_rows)[
#             [
#                 "model",
#                 "num_joint_success",
#                 "std_joint_mean_cf", "std_joint_med_cf",
#                 "coop_joint_mean_cf", "coop_joint_med_cf",
#             ]
#         ]
#         out_joint = Path(TEST_DIR) / "joint_cf_metrics.csv"
#         joint_df.to_csv(out_joint, index=False)
#         print(f"✅ Joint CF 结果已保存到: {out_joint}")
#     else:
#         print("⚠️ 没有任何模型同时存在 std 和 coop，未生成 joint CF 结果。")

# if __name__ == "__main__":
#     main()
