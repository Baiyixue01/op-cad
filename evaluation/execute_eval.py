# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# rerun_full.py
# 批量对已修正的 full.py 复跑，并更新 cands.csv（可选重算 summary.csv）

# 功能要点：
# - 读取 --rerun-full-csv（含 group_index，可选 k_index）
# - 在 <OUT_DIR>/middle_step/<pid>/full_path/ 下找到 k{k}_full.py，执行并产出 k{k}_full.step
# - 采用 GT “整体形状”与之比对，计算 cd_full/hd_full 等
# - 仅更新 cands.csv 的 FULL 相关列，不改 SINGLE 相关列
# - 若启用 --write-summary，会对涉及的 pid 重算 summary.csv

# 依赖（复用已有函数/模块）：
# - utils.compute_3D.get_cd_hd, utils.compute_3D.MetricsResult
# - 系统已安装 cadquery、numpy、pandas、scipy（用于你现有流程）
# """

# import os, re, ast, json, traceback
# import argparse
# import multiprocessing as mp
# from functools import partial
# from typing import Optional, Tuple, List

# import numpy as np
# import pandas as pd
# import cadquery as cq
# from tqdm import tqdm

# # 尽量减少 OpenBLAS 抢核
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

# # ========== 计算 3D 指标 ==========
# def _safe_get_cd_hd(pred_step_path, gt_step_path, *, num_points=None, angles=None):
#     """包装 get_cd_hd，任何异常都转成 MetricsResult(ok=False, reason=...)；支持固定角度。"""
#     from utils.compute_3D import get_cd_hd, MetricsResult
#     try:
#         if angles is None:
#             return get_cd_hd(pred_step_path=pred_step_path, gt_step_path=gt_step_path)
#         else:
#             return get_cd_hd(pred_step_path=pred_step_path, gt_step_path=gt_step_path, angles=angles)
#     except Exception as e:
#         reason = f"metric_exception:{type(e).__name__}:{e}"
#         return MetricsResult(None, None, None, ok=False, reason=reason)

# # ========== 执行 full.py ==========
# def safe_exec_from_path(py_path: str, globals_dict=None):
#     """执行保存到磁盘的 Python/CadQuery 脚本；返回 (ok, locals, err)。"""
#     glb = {"cq": cq, "np": np}
#     if globals_dict:
#         glb.update(globals_dict)
#     loc = {}
#     try:
#         with open(py_path, "r", encoding="utf-8") as f:
#             src = f.read()
#         exec(compile(src, py_path, "exec"), glb, glb)
#         return True, loc, ""
#     except Exception as e:
#         return False, {}, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

# # ========== GT 路径解析（独立实现，避免依赖大脚本） ==========
# def _combo_names_from_indices(indices: List[int]) -> List[str]:
#     s = [str(i) for i in indices]
#     combos = []
#     if indices:
#         combos.append("_".join(s))
#         combos.append("-".join(s))
#         combos.append(",".join(s))
#         combos.append(s[-1])  # 兜底：只用最后一个编号
#     return combos

# def _numbers_in_folder_suffix(folder_name: str, step_prefix: str) -> List[int]:
#     suf = folder_name[len(step_prefix):]
#     nums = re.findall(r"\d+", suf)
#     return [int(x) for x in nums]

# def _parse_group_info_txt(path: str) -> dict:
#     """
#     解析 GT_SINGLE_STEP_DIR/<group>/group_info.txt
#     支持形如: step0: [{0:'...'}]
#     返回: {"step0":[0], "step1":[1,2], ...}
#     """
#     out = {}
#     if not os.path.exists(path):
#         return out
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             s = line.strip()
#             if not s or s.startswith("#"):
#                 continue
#             m = re.match(r"^step(\d+)\s*:\s*(\[.*\])\s*$", s)
#             if not m:
#                 continue
#             step = f"step{m.group(1)}"
#             try:
#                 arr = ast.literal_eval(m.group(2))
#                 idxs = []
#                 for d in arr:
#                     if isinstance(d, dict) and d:
#                         key = next(iter(d.keys()))
#                         try:
#                             idxs.append(int(key))
#                         except:
#                             pass
#                 out[step] = idxs
#             except Exception:
#                 continue
#     return out

# def _extract_group_and_step(pid: str) -> Tuple[str, str]:
#     parts = str(pid).split("/")
#     group = "/".join(parts[:-1]) if len(parts) > 1 else ""
#     step  = parts[-1]
#     return group, step

# def _pick_full_step_path(op_orient_group_dir: str, expected_indices: List[int]) -> Optional[str]:
#     if not os.path.isdir(op_orient_group_dir):
#         return None
#     combos = _combo_names_from_indices(expected_indices)
#     candidates = []
#     for c in combos:
#         last = c.split("_")[-1].split("-")[-1].split(",")[-1] if c else None
#         candidates += [
#             os.path.join(op_orient_group_dir, c, "next_model.step"),
#             os.path.join(op_orient_group_dir, c, "3D.step"),
#         ]
#         if last:
#             candidates += [
#                 os.path.join(op_orient_group_dir, c, last, "next_model.step"),
#                 os.path.join(op_orient_group_dir, last, "next_model.step"),
#                 os.path.join(op_orient_group_dir, last, "3D.step"),
#             ]
#     for p in candidates:
#         if p and os.path.exists(p):
#             return p
#     return None
# # ====== 在 GT 路径解析区，补上 single 的定位 ======
# def _pick_single_step_path(group_dir: str, indices: List[int]) -> Optional[str]:
#     """
#     在 <group_dir> 下，为“单步（isolated）形状”挑选 3D.step 路径。
#     - 单个 i -> step{i}/3D.step
#     - 多个 [i,j,k] -> stepi_j_k/3D.step（兼容 -, , 变体）
#     """
#     if not os.path.isdir(group_dir) or not indices:
#         return None

#     def _combo_candidates(idxs: List[int]) -> List[str]:
#         s = [str(x) for x in idxs]
#         if len(idxs) == 1:
#             return [f"step{s[0]}"]
#         return [f"step{'_'.join(s)}", f"step{'-'.join(s)}", f"step{','.join(s)}"]

#     for d in _combo_candidates(indices):
#         p = os.path.join(group_dir, d, "3D.step")
#         if os.path.exists(p):
#             return p
#     return None

# def resolve_gt_single_path(pid: str, gt_single_step_dir: str, dedup_csv: str) -> Optional[str]:
#     dedup = load_dedup_map(dedup_csv)
#     group, step = _extract_group_and_step(pid)
#     group_base = re.sub(r"/step\d+$", "", group.strip())
#     base_used = dedup.get(group_base, group_base)

#     group_dir = os.path.join(gt_single_step_dir, base_used.replace("/", os.sep))
#     gi_path   = os.path.join(group_dir, "group_info.txt")
#     m = _parse_group_info_txt(gi_path)
#     expected = m.get(step, [])
#     return _pick_single_step_path(group_dir, expected)


# def load_dedup_map(dedup_csv: str) -> dict:
#     """读取去重映射表：返回 {group_index: canonical_group_index}"""
#     if not os.path.exists(dedup_csv):
#         return {}
#     df = pd.read_csv(dedup_csv)
#     df.columns = [c.lower() for c in df.columns]
#     mapping = {}
#     if "group_index" in df.columns and "duplicate_of_group_index" in df.columns:
#         for _, r in df.iterrows():
#             g = str(r["group_index"]).strip()
#             d = str(r["duplicate_of_group_index"]).strip()
#             if d and d.lower() != "nan":
#                 mapping[g] = d
#     return mapping

# def resolve_gt_full_path(pid: str, gt_single_step_dir: str, op_orient_dir: str, dedup_csv: str) -> Optional[str]:
#     """
#     返回“累计到该步的整体形状”的 GT STEP 路径（用于 full 指标）。
#     """
#     dedup = load_dedup_map(dedup_csv)
#     group, step = _extract_group_and_step(pid)
#     group_base = re.sub(r"/step\d+$", "", group.strip())
#     base_used = dedup.get(group_base, group_base)

#     group_dir = os.path.join(gt_single_step_dir, base_used.replace("/", os.sep))
#     gi_path   = os.path.join(group_dir, "group_info.txt")
#     m = _parse_group_info_txt(gi_path)
#     expected = m.get(step, [])

#     op_orient_group_dir = os.path.join(op_orient_dir, base_used.replace("/", os.sep))
#     gt_full = _pick_full_step_path(op_orient_group_dir, expected)
#     return gt_full

# # ========== cands / summary ==========
# FULL_COLS = [
#     "exec_ok_full","pred_full_path","cd_full","hd_full",
#     "angle_full","reason_full","pred_full_exists","metric_ok_full"
# ]

# SINGLE_COLS = [
#     "exec_ok_single","pred_single_path","cd_single","hd_single",
#     "angle_single","reason_single","pred_single_exists","metric_ok_single"
# ]

# def update_cands_rows(cands_df: pd.DataFrame, updates: List[dict], cols: List[str] = None) -> pd.DataFrame:
#     if cols is None:
#         cols = FULL_COLS
#     if cands_df is None or cands_df.empty:
#         # 只带给定列，不破坏结构
#         return pd.DataFrame(updates)
#     cands_df = cands_df.copy()
#     for upd in updates:
#         pid = str(upd["group_index"])
#         k   = int(upd["k_index"])
#         mask = (cands_df["group_index"].astype(str) == pid) & \
#                (pd.to_numeric(cands_df["k_index"], errors="coerce") == k)
#         if mask.any():
#             for col in cols:
#                 if col in upd:
#                     cands_df.loc[mask, col] = upd[col]
#         else:
#             cands_df = pd.concat([cands_df, pd.DataFrame([upd])], ignore_index=True)
#     return cands_df

# def compute_summary_for_pid(rows: List[dict], pid: str, op_kind: str) -> dict:
#     """简化版 summary 计算（与你主流程一致的逻辑）。"""
#     import numpy as _np
#     if op_kind == "chamfer_fillet":
#         cf_rows = [r for r in rows if "cf_hit_rate" in r]
#         cf_hit_mean = _np.nanmean([r.get("cf_hit_rate", _np.nan) for r in cf_rows]) if cf_rows else _np.nan
#         cf_err_mean = _np.nanmean([r.get("cf_err_rate", _np.nan) for r in cf_rows]) if cf_rows else _np.nan
#         cf_hits_total = int(_np.nansum([r.get("cf_hits_fillet", 0) + r.get("cf_hits_chamfer", 0) for r in cf_rows])) if cf_rows else 0
#         cf_num_pred_fillet = int(_np.nansum([r.get("cf_num_pred_fillet", 0) for r in cf_rows])) if cf_rows else 0
#         cf_num_pred_chamfer = int(_np.nansum([r.get("cf_num_pred_chamfer", 0) for r in cf_rows])) if cf_rows else 0
#         cf_num_gt_fillet = int(_np.nansum([r.get("cf_num_gt_fillet", 0) for r in cf_rows])) if cf_rows else 0
#         cf_num_gt_chamfer = int(_np.nansum([r.get("cf_num_gt_chamfer", 0) for r in cf_rows])) if cf_rows else 0
#         return {
#             "group_index": pid, "k_index": "summary", "op_type": "Chamfer/Fillet",
#             "cf_hit_rate_mean": cf_hit_mean, "cf_err_rate_mean": cf_err_mean,
#             "cf_hits_total": cf_hits_total,
#             "cf_num_pred_fillet": cf_num_pred_fillet, "cf_num_pred_chamfer": cf_num_pred_chamfer,
#             "cf_num_gt_fillet": cf_num_gt_fillet, "cf_num_gt_chamfer": cf_num_gt_chamfer,
#             "n_total": len(rows),
#         }

#     # 几何类：选有效行的均值与最优
#     def _best_and_mean(rows, cd_key, hd_key, ok_key):
#         valid = [r for r in rows if r.get(ok_key) == 1 and
#                  (r.get(cd_key) is not None) and (r.get(hd_key) is not None)]
#         if valid:
#             best_row = min(valid, key=lambda r: (r.get(cd_key, 1e9) or 1e9) + (r.get(hd_key, 1e9) or 1e9))
#             cd_mean = float(_np.nanmean([r[cd_key] for r in valid]))
#             hd_mean = float(_np.nanmean([r[hd_key] for r in valid]))
#         else:
#             best_row, cd_mean, hd_mean = None, _np.nan, _np.nan
#         return best_row, cd_mean, hd_mean, len(valid)

#     best_f, cd_mean_f, hd_mean_f, n_valid_f = _best_and_mean(rows, "cd_full", "hd_full", "metric_ok_full")
#     best_s, cd_mean_s, hd_mean_s, n_valid_s = _best_and_mean(rows, "cd_single", "hd_single", "metric_ok_single")

#     return {
#         "group_index": pid,
#         "k_index": "summary",
#         "op_type": "Geometry",
#         "best_k_single": (best_s or {}).get("k_index", None),
#         "cd_single_best": (best_s or {}).get("cd_single", _np.nan),
#         "hd_single_best": (best_s or {}).get("hd_single", _np.nan),
#         "cd_single_mean": cd_mean_s,
#         "hd_single_mean": hd_mean_s,
#         "best_k_full": (best_f or {}).get("k_index", None),
#         "cd_full_best": (best_f or {}).get("cd_full", _np.nan),
#         "hd_full_best": (best_f or {}).get("hd_full", _np.nan),
#         "cd_full_mean": cd_mean_f,
#         "hd_full_mean": hd_mean_f,
#         "n_total": len(rows),
#         "n_exec_ok_single": sum(int(r.get("exec_ok_single", 0) == 1) for r in rows),
#         "n_exec_ok_full":   sum(int(r.get("exec_ok_full",   0) == 1) for r in rows),
#         "n_pred_exist_single": sum(int(r.get("pred_single_exists", 0) == 1) for r in rows),
#         "n_pred_exist_full":   sum(int(r.get("pred_full_exists",   0) == 1) for r in rows),
#         "n_metric_ok_single": n_valid_s,
#         "n_metric_ok_full": n_valid_f,
#     }

# # ========== 复跑单元 ==========
# def full_paths_for(out_dir: str, pid: str, k: int):
#     base_tmp_dir = os.path.join(out_dir, "middle_step", pid)
#     full_dir = os.path.join(base_tmp_dir, "full_path")
#     return os.path.join(full_dir, f"k{k}_full.py"), os.path.join(full_dir, f"k{k}_full.step")

# def single_paths_for(out_dir: str, pid: str, k: int):
#     base_tmp_dir = os.path.join(out_dir, "middle_step", pid)
#     single_dir = os.path.join(base_tmp_dir, "single_step")
#     return os.path.join(single_dir, f"k{k}_single.py"), os.path.join(single_dir, f"k{k}_single.step")

# def recompute_single_once(task_tuple: tuple, out_dir: str, gt_single_step_dir: str, dedup_csv: str):
#     """
#     对 (pid,k)：执行 single.py → 评测 SINGLE → 返回更新用字典（仅 single 列）
#     """
#     pid, k = task_tuple

#     gt_single_step = resolve_gt_single_path(pid, gt_single_step_dir, dedup_csv)
#     single_py, single_step_path = single_paths_for(out_dir, pid, k)

#     if os.path.exists(single_py):
#         ok_single, _, err_single = safe_exec_from_path(single_py)
#     else:
#         ok_single, err_single = False, f"file_missing:{single_py}"
#     pred_single_step_exists = ok_single and os.path.exists(single_step_path)

#     from utils.compute_3D import MetricsResult
#     if pred_single_step_exists and gt_single_step:
#         res_s = _safe_get_cd_hd(pred_step_path=single_step_path, gt_step_path=gt_single_step)
#     else:
#         miss = "gt_step_missing" if not gt_single_step else "pred_step_missing"
#         res_s = MetricsResult(None, None, None, ok=False, reason=miss)

#     reason_s = ""
#     if not ok_single:
#         reason_s = f"single_exec_error:{(err_single.splitlines()[-1] if err_single else 'unknown')}"
#     if not res_s.ok and getattr(res_s, "reason", ""):
#         reason_s = (reason_s + "; " if reason_s else "") + res_s.reason

#     return {
#         "group_index": pid,
#         "k_index": k,
#         "exec_ok_single": int(ok_single),
#         "pred_single_path": single_step_path if pred_single_step_exists else "",
#         "cd_single": res_s.cd,
#         "hd_single": res_s.hd,
#         "angle_single": _fmt_angle_triple(getattr(res_s, "best_euler_angle", None)),
#         "reason_single": reason_s if not res_s.ok else "",
#         "pred_single_exists": int(pred_single_step_exists),
#         "metric_ok_single": int(res_s.ok and (res_s.cd is not None) and (res_s.hd is not None)),
#     }

# def recompute_full_once(task_tuple: tuple, out_dir: str, gt_single_step_dir: str, op_orient_dir: str, dedup_csv: str):
#     """
#     (已修正)
#     对 (pid,k)：执行 full.py → 评测 FULL → 返回更新用字典
#     - 仅在 step0 时，才尝试继承 single 指标 (与 eval.py 一致)
#     - 在 step>=1 时，使用 angles=[0] 固定角度 (与 eval.py 一致)
#     """
#     from utils.compute_3D import MetricsResult

#     pid, k = task_tuple

#     # --- 1. (新增) 判断是否为 step0 ---
#     m = re.search(r"step(\d+)", pid)
#     step_num = int(m.group(1)) if m else -1
#     first_step = (step_num == 0)

#     # --- GT full 路径 ---
#     gt_full_step = resolve_gt_full_path(pid, gt_single_step_dir, op_orient_dir, dedup_csv)

#     # --- full.py 路径 ---
#     full_py, full_step_path = full_paths_for(out_dir, pid, k)
    
#     # --- 执行 full.py ---
#     if os.path.exists(full_py):
#         ok_full, _, err_full = safe_exec_from_path(full_py)
#     else:
#         ok_full, err_full = False, f"file_missing:{full_py}"
#     pred_full_step_exists = ok_full and os.path.exists(full_step_path)

#     # --- 判断是否直接继承 single (逻辑修正) ---
#     inherit_from_single = False
#     res_f = None
#     reason_f = ""

#     # (修正点 1： 之前是 k == 0，现在是 first_step)
#     if first_step: 
#         # 仅 step0 尝试继承 single
#         single_py, single_step_path = single_paths_for(out_dir, pid, k)
#         if os.path.exists(single_step_path):
#             gt_single_step = resolve_gt_single_path(pid, gt_single_step_dir, dedup_csv)
#             if gt_single_step:
#                 # single 指标总是用全角度 (angles=None 默认)
#                 res_s = _safe_get_cd_hd(pred_step_path=single_step_path, gt_step_path=gt_single_step)
#                 if res_s.ok:
#                     inherit_from_single = True
#                     res_f = MetricsResult(cd=res_s.cd,
#                                           hd=res_s.hd,
#                                           best_euler_angle=res_s.best_euler_angle,
#                                           ok=True,
#                                           reason="inherit_from_single_step0")
    
#     # --- 若不继承，则正常评测 FULL (逻辑修正) ---
#     if not inherit_from_single:
#         if pred_full_step_exists and gt_full_step:
#             # (修正点 2：新增 angles=[0])
#             # step≥1 (或 step0 继承失败)
#             # 强制使用 0 度角，与 eval.py 保持一致
#             res_f = _safe_get_cd_hd(pred_step_path=full_step_path,
#                                     gt_step_path=gt_full_step,
#                                     angles=[0]) 
#         else:
#             miss = "gt_step_missing" if not gt_full_step else "pred_step_missing"
#             res_f = MetricsResult(None, None, None, ok=False, reason=miss)

#     # --- reason 汇总 ---
#     if not ok_full:
#         reason_f = f"full_exec_error:{(err_full.splitlines()[-1] if err_full else 'unknown')}"
#     if res_f and (not res_f.ok) and getattr(res_f, "reason", ""):
#         reason_f = (reason_f + "; " if reason_f else "") + res_f.reason
#     elif res_f is None:
#         reason_f = "metric_result_is_none"

#     # --- 返回 ---
#     return {
#         "group_index": pid,
#         "k_index": k,
#         "exec_ok_full": int(ok_full),
#         "pred_full_path": full_step_path if pred_full_step_exists else "",
#         "cd_full": res_f.cd if res_f else np.nan,
#         "hd_full": res_f.hd if res_f else np.nan,
#         "angle_full": _fmt_angle_triple(getattr(res_f, "best_euler_angle", None)),
#         "reason_full": reason_f,
#         "pred_full_exists": int(pred_full_step_exists),
#         "metric_ok_full": int(res_f.ok and (res_f.cd is not None) and (res_f.hd is not None)) if res_f else 0,
#     }


# # ========== 主流程 ==========
# def parse_args():
#     p = argparse.ArgumentParser(description="复跑修正的 full.py 并更新 cands/summary")
#     # 定位 OUT_DIR
#     p.add_argument("--out-root", required=True, help="输出根目录（包含 <test_name>/<mode> 子目录）")
#     p.add_argument("--test-name", required=True, help="测试名（与现有目录一致）")
#     p.add_argument("--mode", choices=["std","cop"], default="std", help="模式子目录名")
#     # 复跑清单
#     p.add_argument("--rerun-full-csv", default=None, help="CSV：group_index（必需），k_index（可选）")
#     p.add_argument("--default-ks", default="0,1", help="缺省重跑槽位列表，如 0,1 或 0")
#     # GT 定位
#     p.add_argument("--gt-single-step-dir", required=True, help="单步GT STEP根目录（用于读取 group_info.txt）")
#     p.add_argument("--op-orient-dir", required=True, help="整体形状（累计到 stepN）的 STEP 根目录")
#     p.add_argument("--dedup-csv", required=True, help="去重映射 CSV（group_index, duplicate_of_group_index）")
#     # 并行与写盘
#     p.add_argument("--nproc", type=int, default=64, help="进程数")
#     p.add_argument("--write-summary", action="store_true", default=True, help="更新 summary.csv（默认开）")
    
#     # ====== CLI：在 parse_args() 里添加两个参数 ======
#     p.add_argument("--rerun-single-csv", default=None, help="CSV：group_index（必需），k_index（可选），用于复跑 single.py")
#     p.add_argument("--default-ks-single", default="0,1", help="single 缺省槽位列表，如 0,1 或 0")
#     return p.parse_args()

# def _fmt_angle_triple(v):
#     """
#     把 best_euler_angle 转成形如 "(90, 270, 180)" 的字符串。
#     允许输入 list/tuple/np.ndarray/str/None。
#     非法或缺失 -> 返回空字符串 ""（与 CSV 空值一致）。
#     """
#     import numpy as _np
#     if v is None:
#         return ""
#     # 已经是字符串就原样返回（如 "(0, 90, 90)"）
#     if isinstance(v, str):
#         s = v.strip()
#         # 简单校验一下，避免像 "[0, 90, 90]" 混进来
#         if s.startswith("(") and s.endswith(")"):
#             return s
#         # 尝试把 "0, 90, 90" 这种也包上括号
#         if "," in s and not (s.startswith("[") or s.endswith("]")):
#             return f"({s})"
#         # 其他字符串（比如 "nan"）都视为无效
#         return ""

#     # list/tuple/np.ndarray -> 取前三个
#     if isinstance(v, (list, tuple, set, _np.ndarray)):
#         arr = list(v) if not isinstance(v, _np.ndarray) else v.tolist()
#         if len(arr) < 3:
#             return ""
#         # 你可以按需 round 或取 int；这里和你示例保持整型输出
#         try:
#             a, b, c = int(round(float(arr[0]))), int(round(float(arr[1]))), int(round(float(arr[2])))
#             return f"({a}, {b}, {c})"
#         except Exception:
#             return ""
#     # 其他类型一律当作无
#     return ""


# def main():
#     args = parse_args()
#     OUT_DIR = os.path.join(args.out_root, args.test_name, args.mode)
#     cand_out_path = os.path.join(OUT_DIR, "cands.csv")
#     summary_out_path = os.path.join(OUT_DIR, "summary.csv")

#     if not os.path.exists(cand_out_path):
#         raise FileNotFoundError(f"cands.csv 不存在：{cand_out_path}")

#     # 加载现有 cands / summary
#     cands = pd.read_csv(cand_out_path)
#     if not cands.empty:
#         cands["group_index"] = cands["group_index"].astype(str)
#         cands["k_index"] = pd.to_numeric(cands["k_index"], errors="coerce").astype("Int64")
#     sums = pd.read_csv(summary_out_path) if (args.write_summary and os.path.exists(summary_out_path)) else pd.DataFrame()
#     if not sums.empty:
#         sums["group_index"] = sums["group_index"].astype(str)

#     # === 复跑 FULL ===
#     if args.rerun_full_csv:
#         dfr = pd.read_csv(args.rerun_full_csv)
#         dfr.columns = [c.lower() for c in dfr.columns]
#         dfr = dfr.dropna(subset=["group_index"])
#         dfr["group_index"] = dfr["group_index"].astype(str).str.strip()

#         default_ks = [int(x) for x in str(args.default_ks).split(",") if x.strip() != ""]
#         targets = []
#         for _, r in dfr.iterrows():
#             pid = str(r["group_index"])
#             ks = [int(r["k_index"])] if "k_index" in dfr.columns and pd.notna(r.get("k_index")) else default_ks
#             for k in ks:
#                 targets.append((pid, k))

#         worker = partial(
#             recompute_full_once,
#             out_dir=OUT_DIR,
#             gt_single_step_dir=args.gt_single_step_dir,
#             op_orient_dir=args.op_orient_dir,
#             dedup_csv=args.dedup_csv
#         )

#         # 实时写入
#         header_written = os.path.exists(cand_out_path)
#         with mp.Pool(processes=args.nproc) as pool:
#             for i, ret in enumerate(
#                 tqdm(pool.imap_unordered(worker, targets, chunksize=1),
#                      total=len(targets), desc="[RerunFull]", ncols=100)
#             ):
#                 # 内存更新（只改匹配的 group_index/k_index）
#                 cands = update_cands_rows(cands, [ret], cols=FULL_COLS)
#                 # 实时保存
#                 cands.to_csv(cand_out_path, index=False)
#                 header_written = True

#                 # 每 50 次增量更新 summary
#                 if args.write_summary and (i + 1) % 50 == 0:
#                     pid = ret["group_index"]
#                     rows_pid = pd.read_csv(cand_out_path)
#                     rows_pid = rows_pid[(rows_pid["group_index"] == str(pid)) & (rows_pid["k_index"].isin([0, 1]))]
#                     op_kind = "geometry"
#                     row_dicts = rows_pid.to_dict(orient="records")
#                     new_sum = compute_summary_for_pid(row_dicts, str(pid), op_kind)
#                     if not sums.empty and {"group_index","k_index"}.issubset(sums.columns):
#                         sums = sums[~(
#                             (sums["group_index"].astype(str) == str(pid)) &
#                             (sums["k_index"].astype(str).isin(["summary","summary_best"]))
#                         )]
#                     # === 修正结束 ===
                    
#                     sums = pd.concat([sums, pd.DataFrame([new_sum])], ignore_index=True)
#                     sums.to_csv(summary_out_path, index=False)

#     # === 复跑 SINGLE ===
#     if args.rerun_single_csv:
#         dfr_s = pd.read_csv(args.rerun_single_csv)
#         dfr_s.columns = [c.lower() for c in dfr_s.columns]
#         dfr_s = dfr_s.dropna(subset=["group_index"])
#         dfr_s["group_index"] = dfr_s["group_index"].astype(str).str.strip()
#         default_ks_single = [int(x) for x in str(args.default_ks_single).split(",") if x.strip() != ""]
#         targets_s = []
#         for _, r in dfr_s.iterrows():
#             pid = str(r["group_index"])
#             ks = [int(r["k_index"])] if "k_index" in dfr_s.columns and pd.notna(r.get("k_index")) else default_ks_single
#             for k in ks:
#                 targets_s.append((pid, k))

#         worker_s = partial(
#             recompute_single_once,
#             out_dir=OUT_DIR,
#             gt_single_step_dir=args.gt_single_step_dir,
#             dedup_csv=args.dedup_csv
#         )

#         header_written = os.path.exists(cand_out_path)
#         with mp.Pool(processes=args.nproc) as pool:
#             for i, ret in enumerate(
#                 tqdm(pool.imap_unordered(worker_s, targets_s, chunksize=1),
#                      total=len(targets_s), desc="[RerunSingle]", ncols=100)
#             ):
#                 # 内存更新（只改匹配的 group_index/k_index）
#                 cands = update_cands_rows(cands, [ret], cols=SINGLE_COLS)
#                 # 实时保存
#                 cands.to_csv(cand_out_path, index=False)
#                 header_written = True

#                 # 同样每 50 次更新 summary
#                 if args.write_summary and (i + 1) % 50 == 0:
#                     pid = ret["group_index"]
#                     rows_pid = pd.read_csv(cand_out_path)
#                     rows_pid = rows_pid[(rows_pid["group_index"] == str(pid)) & (rows_pid["k_index"].isin([0, 1]))]
#                     op_kind = "geometry"
#                     row_dicts = rows_pid.to_dict(orient="records")
#                     new_sum = compute_summary_for_pid(row_dicts, str(pid), op_kind)
#                     sums = pd.concat([sums, pd.DataFrame([new_sum])], ignore_index=True)
#                     sums.to_csv(summary_out_path, index=False)

#     print("✅ [RERUN实时模式] 所有结果已实时写入 cands.csv（summary 每 50 次同步更新）")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image

# ====== 配置区：按需修改 ======
RENDER_ROOT = Path("/data/baiyixue/gm_cop/render")
OP_ROOT     = Path("/data/baiyixue/CAD/op_orientated_render_data")

TARGET_NAME = "location.png"   # 在 op_orientated_render_data 中的目标图片名
FULL_SUFFIX = "_full"          # ds_cop 这边的后缀，用来替换成 _compare
COMPARE_SUFFIX = "_compare"    # 输出图片后缀
# =============================


def concat_horiz(im1: Image.Image, im2: Image.Image) -> Image.Image:
    """
    把两张图横向拼接，高度对齐，第二张图按比例缩放到与第一张同高
    """
    # 统一为 RGB
    if im1.mode != "RGB":
        im1 = im1.convert("RGB")
    if im2.mode != "RGB":
        im2 = im2.convert("RGB")

    w1, h1 = im1.size
    w2, h2 = im2.size

    # 将第二张图按比例缩放到与第一张同高
    if h2 != h1:
        new_w2 = int(w2 * (h1 / h2))
        im2 = im2.resize((new_w2, h1), Image.BILINEAR)
        w2, h2 = im2.size

    new_img = Image.new("RGB", (w1 + w2, h1))
    new_img.paste(im1, (0, 0))
    new_img.paste(im2, (w1, 0))

    return new_img


def process_all():
    # 查找所有 k*_full.png
    png_paths = list(RENDER_ROOT.rglob("k*_full.png"))

    print(f"Found {len(png_paths)} full images under {RENDER_ROOT}")

    for p in png_paths:
        try:
            rel = p.relative_to(RENDER_ROOT)
        except ValueError:
            # 理论上不会发生
            print(f"[Skip] cannot make relative path for {p}")
            continue

        # 期望结构: <group_index>/stepX/full_path/k0_full.png
        parts = rel.parts
        if len(parts) < 4:
            print(f"[Skip] unexpected path structure: {rel}")
            continue

        group_index = parts[0]    # 如 00224_index_6
        step_name   = parts[1]    # 如 step1

        # 构造 op_orientated_render_data 中 location.png 的路径：
        op_img_path = OP_ROOT / group_index / step_name / TARGET_NAME

        if not op_img_path.is_file():
            print(f"[Missing] {op_img_path} for {p}")
            continue

        # 读图并拼接
        try:
            im1 = Image.open(p)
            im2 = Image.open(op_img_path)
        except Exception as e:
            print(f"[Error] open image failed: {p} or {op_img_path}, err={e}")
            continue

        merged = concat_horiz(im1, im2)

        # 构造输出路径：k0_full.png -> k0_compare.png
        stem = p.stem  # k0_full
        if stem.endswith(FULL_SUFFIX):
            new_stem = stem[:-len(FULL_SUFFIX)] + COMPARE_SUFFIX
        else:
            new_stem = stem + COMPARE_SUFFIX

        out_path = p.with_name(new_stem + ".png")

        try:
            merged.save(out_path)
            print(f"[OK] {out_path}")
        except Exception as e:
            print(f"[Error] save failed: {out_path}, err={e}")


if __name__ == "__main__":
    process_all()
