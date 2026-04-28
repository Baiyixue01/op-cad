import os
import re
import ast
import json
import shutil
import traceback
import subprocess
import signal
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


# =========================
# 配置区：按你的实际路径修改
# =========================
CANDS_CSV = "/data/baiyixue/CAD/inference_result/noco/llama_3.1_8b/std/cands.csv"

GT_SINGLE_STEP_DIR = "/data/baiyixue/CAD/step_files"
OP_ORIENT_DIR = "/data/baiyixue/CAD/op_oriented_step"
DEDUP_CSV = "/home/baiyixue/project/flowcad/data/dedup.csv"

# 关键：项目根目录。用于 cwd / PYTHONPATH
PROJECT_ROOT = "/home/baiyixue/project/op-cad"

# 如果有点云 GT，就填；没有就保持 None
GT_SINGLE_PC_DIR = None
GT_FULL_PC_DIR = None

NUM_POINTS = 2048
SAFE_METRIC_TIMEOUT_SEC = 180

# 并发数：建议先 8 或 16
MAX_WORKERS = 16

# 是否只修复 metric 缺失/失败、但 pred step 文件存在的行
ONLY_FIX_WHEN_PRED_EXISTS = True

# 是否只修复 reason 中包含 subprocess_exit_code:1 的记录
ONLY_FIX_SUBPROCESS_EXIT_1 = False

# 输出
BACKUP_SUFFIX = ".bak_before_metric_refresh"
TMP_OUT_SUFFIX = ".tmp_repaired"


# =========================
# 与原逻辑一致的一些工具函数
# =========================
@dataclass
class MetricsResult:
    cd: Optional[float]
    hd: Optional[float]
    best_euler_angle: Optional[tuple]
    ok: bool
    reason: str = ""


_dedup_map = None


def load_dedup_map() -> dict:
    global _dedup_map
    if _dedup_map is not None:
        return _dedup_map

    if not os.path.exists(DEDUP_CSV):
        print(f"[WARN] dedup csv not found: {DEDUP_CSV}")
        _dedup_map = {}
        return _dedup_map

    df = pd.read_csv(DEDUP_CSV)
    df.columns = [c.lower() for c in df.columns]

    if "group_index" not in df.columns or "duplicate_of_group_index" not in df.columns:
        print("[WARN] dedup csv missing columns: group_index, duplicate_of_group_index")
        _dedup_map = {}
        return _dedup_map

    mapping = {}
    for _, r in df.iterrows():
        g = str(r["group_index"]).strip()
        d = str(r["duplicate_of_group_index"]).strip()
        if d and d.lower() != "nan":
            mapping[g] = d

    _dedup_map = mapping
    return _dedup_map


def _extract_group_and_step(pid: str) -> Tuple[str, str]:
    parts = str(pid).split("/")
    group = "/".join(parts[:-1]) if len(parts) > 1 else ""
    step = parts[-1]
    return group, step


def _parse_group_info_txt(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"^step(\d+)\s*:\s*(\[.*\])\s*$", s)
            if not m:
                continue

            step = f"step{m.group(1)}"
            try:
                arr = ast.literal_eval(m.group(2))
                idxs = []
                for d in arr:
                    if isinstance(d, dict) and d:
                        key = next(iter(d.keys()))
                        try:
                            idxs.append(int(key))
                        except Exception:
                            pass
                out[step] = idxs
            except Exception:
                pass
    return out


def _combo_names_from_indices(indices: List[int]) -> List[str]:
    s = [str(i) for i in indices]
    combos = []
    if indices:
        combos.append("_".join(s))
        combos.append("-".join(s))
        combos.append(",".join(s))
        combos.append(s[-1])
    return combos


def _pick_single_step_path(group_dir: str, indices: List[int]) -> Optional[str]:
    if not os.path.isdir(group_dir) or not indices:
        return None

    def _combo_candidates(idxs: List[int]) -> List[str]:
        s = [str(x) for x in idxs]
        if len(idxs) == 1:
            return [f"step{s[0]}"]
        return [f"step{'_'.join(s)}", f"step{'-'.join(s)}", f"step{','.join(s)}"]

    ideal_dirs = _combo_candidates(indices)
    for d in ideal_dirs:
        p = os.path.join(group_dir, d, "3D.step")
        if os.path.exists(p):
            return p
    return None


def _pick_single_pc_path(group_dir: str, indices: List[int]) -> Optional[str]:
    if not os.path.isdir(group_dir) or not indices:
        return None

    candidates = []
    combos = [f"step{c}" for c in _combo_names_from_indices(indices)]
    for name in combos:
        candidates.extend([
            os.path.join(group_dir, name, "3D.npy"),
            os.path.join(group_dir, name, "pointcloud.npy"),
            os.path.join(group_dir, name, "pc.npy"),
            os.path.join(group_dir, f"{name}.npy"),
        ])

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _pick_full_step_path(op_orient_group_dir: str, expected_indices: List[int]) -> Optional[str]:
    if not os.path.isdir(op_orient_group_dir):
        return None

    combos = _combo_names_from_indices(expected_indices)
    candidates = []

    for c in combos:
        last = c.split("_")[-1].split("-")[-1].split(",")[-1] if c else None
        candidates += [
            os.path.join(op_orient_group_dir, c, "next_model.step"),
            os.path.join(op_orient_group_dir, c, "3D.step"),
        ]
        if last:
            candidates += [
                os.path.join(op_orient_group_dir, c, last, "next_model.step"),
                os.path.join(op_orient_group_dir, last, "next_model.step"),
                os.path.join(op_orient_group_dir, last, "3D.step"),
            ]

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _pick_full_pc_path(op_orient_group_dir: str, expected_indices: List[int]) -> Optional[str]:
    if not os.path.isdir(op_orient_group_dir):
        return None

    combos = _combo_names_from_indices(expected_indices)
    candidates = []

    for c in combos:
        last = c.split("_")[-1].split("-")[-1].split(",")[-1] if c else None
        candidates += [
            os.path.join(op_orient_group_dir, c, "next_model.npy"),
            os.path.join(op_orient_group_dir, c, "3D.npy"),
            os.path.join(op_orient_group_dir, c, "pointcloud.npy"),
            os.path.join(op_orient_group_dir, c, "pc.npy"),
            os.path.join(op_orient_group_dir, f"{c}.npy"),
        ]
        if last:
            candidates += [
                os.path.join(op_orient_group_dir, c, last, "next_model.npy"),
                os.path.join(op_orient_group_dir, c, last, "3D.npy"),
                os.path.join(op_orient_group_dir, last, "next_model.npy"),
                os.path.join(op_orient_group_dir, last, "3D.npy"),
                os.path.join(op_orient_group_dir, f"{last}.npy"),
            ]

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def resolve_gt_paths(pid: str) -> Tuple[Optional[str], Optional[str]]:
    dedup = load_dedup_map()

    group, step = _extract_group_and_step(pid)
    group_base = re.sub(r"/step\d+$", "", group.strip())

    if group_base in dedup:
        base_used = dedup[group_base]
    else:
        base_used = group_base

    group_dir = os.path.join(GT_SINGLE_STEP_DIR, base_used.replace("/", os.sep))
    gi_path = os.path.join(group_dir, "group_info.txt")
    mapping = _parse_group_info_txt(gi_path)
    expected = mapping.get(step, [])

    gt_single = None
    if GT_SINGLE_PC_DIR:
        pc_group_dir = os.path.join(GT_SINGLE_PC_DIR, base_used.replace("/", os.sep))
        gt_single = _pick_single_pc_path(pc_group_dir, expected)
    if gt_single is None:
        gt_single = _pick_single_step_path(group_dir, expected)

    gt_full = None
    if GT_FULL_PC_DIR:
        full_pc_group_dir = os.path.join(GT_FULL_PC_DIR, base_used.replace("/", os.sep))
        gt_full = _pick_full_pc_path(full_pc_group_dir, expected)
    if gt_full is None:
        op_orient_group_dir = os.path.join(OP_ORIENT_DIR, base_used.replace("/", os.sep))
        gt_full = _pick_full_step_path(op_orient_group_dir, expected)

    return gt_single, gt_full


# =========================
# 隔离子进程执行 3D metric
# =========================
_SUBPROCESS_JSON_MARKER = "__FLOWCAD_JSON__="


def _format_subprocess_failure(returncode: int, stderr: str = "") -> str:
    if returncode < 0:
        sig_num = -returncode
        try:
            sig_name = signal.Signals(sig_num).name
        except Exception:
            sig_name = f"SIG{sig_num}"
        return f"subprocess_terminated_by_signal:{sig_name}"

    detail = f"subprocess_exit_code:{returncode}"
    stderr = (stderr or "").strip()
    if stderr:
        detail += f"; stderr={stderr}"
    return detail


def _extract_subprocess_json(stdout: str):
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_SUBPROCESS_JSON_MARKER):
            payload = line[len(_SUBPROCESS_JSON_MARKER):]
            return json.loads(payload)
    raise ValueError("missing subprocess json payload")


def _run_isolated_python(payload: dict, timeout: int):
    runner = r'''
import json
import sys

MARKER = "__FLOWCAD_JSON__="

def emit(obj):
    print(MARKER + json.dumps(obj, ensure_ascii=False))

def main():
    try:
        payload = json.loads(sys.argv[1])
        from reward.utils.compute_3D import get_cd_hd

        kwargs = {
            "pred_step_path": payload["pred_step_path"],
            "gt_step_path": payload["gt_step_path"],
        }
        if payload.get("num_points") is not None:
            kwargs["num_points"] = payload["num_points"]
        if payload.get("angles") is not None:
            kwargs["angles"] = payload["angles"]

        res = get_cd_hd(**kwargs)
        emit({
            "ok": bool(getattr(res, "ok", False)),
            "reason": getattr(res, "reason", ""),
            "cd": getattr(res, "cd", None),
            "hd": getattr(res, "hd", None),
            "best_euler_angle": getattr(res, "best_euler_angle", None),
        })
    except Exception as exc:
        emit({
            "ok": False,
            "err": f"{type(exc).__name__}: {exc}",
        })
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = PROJECT_ROOT if not old_pythonpath else PROJECT_ROOT + os.pathsep + old_pythonpath

    return subprocess.run(
        [sys.executable, "-c", runner, json.dumps(payload, ensure_ascii=False)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT,
        env=env,
    )


def safe_get_cd_hd(pred_step_path, gt_step_path, *, num_points=None, angles=None):
    try:
        payload = {
            "pred_step_path": pred_step_path,
            "gt_step_path": gt_step_path,
            "num_points": num_points,
            "angles": angles,
        }
        proc = _run_isolated_python(payload, timeout=SAFE_METRIC_TIMEOUT_SEC)

        if proc.returncode != 0:
            try:
                data = _extract_subprocess_json(proc.stdout)
                reason = data.get("err", "") or _format_subprocess_failure(proc.returncode, proc.stderr)
                return MetricsResult(None, None, None, ok=False, reason=reason)
            except Exception:
                return MetricsResult(None, None, None, ok=False,
                                     reason=_format_subprocess_failure(proc.returncode, proc.stderr))

        data = _extract_subprocess_json(proc.stdout)
        if not data.get("ok", False):
            return MetricsResult(
                None, None, None,
                ok=False,
                reason=data.get("reason") or data.get("err", "metric_failed")
            )

        best_angles = data.get("best_euler_angle")
        if isinstance(best_angles, list):
            best_angles = tuple(best_angles)

        return MetricsResult(
            data.get("cd"),
            data.get("hd"),
            best_angles,
            ok=True,
            reason=data.get("reason", "")
        )

    except subprocess.TimeoutExpired:
        return MetricsResult(None, None, None, ok=False,
                             reason=f"metric_timeout:{SAFE_METRIC_TIMEOUT_SEC}s")
    except Exception as e:
        return MetricsResult(None, None, None, ok=False,
                             reason=f"metric_exception:{type(e).__name__}:{e}")


# =========================
# 修复逻辑
# =========================
def is_nan_like(x):
    return pd.isna(x) or x == "" or x is None


def should_retry_metric(row: pd.Series) -> bool:
    pred_single_exists = int(pd.to_numeric(row.get("pred_single_exists", 0), errors="coerce")
                             if not is_nan_like(row.get("pred_single_exists", 0)) else 0)
    pred_full_exists = int(pd.to_numeric(row.get("pred_full_exists", 0), errors="coerce")
                           if not is_nan_like(row.get("pred_full_exists", 0)) else 0)

    metric_ok_single = int(pd.to_numeric(row.get("metric_ok_single", 0), errors="coerce")
                           if not is_nan_like(row.get("metric_ok_single", 0)) else 0)
    metric_ok_full = int(pd.to_numeric(row.get("metric_ok_full", 0), errors="coerce")
                         if not is_nan_like(row.get("metric_ok_full", 0)) else 0)

    reason_single = str(row.get("reason_single", "") or "")
    reason_full = str(row.get("reason_full", "") or "")

    cd_single_missing = is_nan_like(row.get("cd_single"))
    hd_single_missing = is_nan_like(row.get("hd_single"))
    cd_full_missing = is_nan_like(row.get("cd_full"))
    hd_full_missing = is_nan_like(row.get("hd_full"))

    single_need = pred_single_exists == 1 and (metric_ok_single == 0 or cd_single_missing or hd_single_missing)
    full_need = pred_full_exists == 1 and (metric_ok_full == 0 or cd_full_missing or hd_full_missing)

    if ONLY_FIX_WHEN_PRED_EXISTS and not (single_need or full_need):
        return False

    if ONLY_FIX_SUBPROCESS_EXIT_1:
        has_subprocess_1 = ("subprocess_exit_code:1" in reason_single) or ("subprocess_exit_code:1" in reason_full)
        if not has_subprocess_1:
            return False

    pred_single_path = str(row.get("pred_single_path", "") or "")
    pred_full_path = str(row.get("pred_full_path", "") or "")
    path_exists = (pred_single_exists == 1 and pred_single_path and os.path.exists(pred_single_path)) or \
                  (pred_full_exists == 1 and pred_full_path and os.path.exists(pred_full_path))

    return path_exists


def repair_one_row(row: pd.Series) -> Dict:
    row = row.to_dict()
    pid = str(row["group_index"])

    gt_single_path, gt_full_path = resolve_gt_paths(pid)

    pred_single_exists = int(pd.to_numeric(row.get("pred_single_exists", 0), errors="coerce")
                             if not is_nan_like(row.get("pred_single_exists", 0)) else 0)
    pred_full_exists = int(pd.to_numeric(row.get("pred_full_exists", 0), errors="coerce")
                           if not is_nan_like(row.get("pred_full_exists", 0)) else 0)

    pred_single_path = str(row.get("pred_single_path", "") or "")
    pred_full_path = str(row.get("pred_full_path", "") or "")

    if pred_single_exists == 1 and pred_single_path and os.path.exists(pred_single_path):
        if gt_single_path and os.path.exists(gt_single_path):
            res_s = safe_get_cd_hd(
                pred_step_path=pred_single_path,
                gt_step_path=gt_single_path,
                num_points=NUM_POINTS,
                angles=None
            )
            row["cd_single"] = res_s.cd
            row["hd_single"] = res_s.hd
            row["angle_single"] = res_s.best_euler_angle
            row["metric_ok_single"] = int(res_s.ok and (res_s.cd is not None) and (res_s.hd is not None))
            row["exec_ok_single"] = int(row["metric_ok_single"])
            row["reason_single"] = "" if row["metric_ok_single"] == 1 else (res_s.reason or "metric_failed")
        else:
            row["cd_single"] = np.nan
            row["hd_single"] = np.nan
            row["angle_single"] = None
            row["metric_ok_single"] = 0
            row["exec_ok_single"] = 0
            row["reason_single"] = "gt_step_missing"
    else:
        row["metric_ok_single"] = 0
        row["exec_ok_single"] = 0
        if not row.get("reason_single"):
            row["reason_single"] = "pred_step_missing"

    if pred_full_exists == 1 and pred_full_path and os.path.exists(pred_full_path):
        if gt_full_path and os.path.exists(gt_full_path):
            res_f = safe_get_cd_hd(
                pred_step_path=pred_full_path,
                gt_step_path=gt_full_path,
                num_points=NUM_POINTS,
                angles=[0]
            )
            row["cd_full"] = res_f.cd
            row["hd_full"] = res_f.hd
            row["angle_full"] = res_f.best_euler_angle
            row["metric_ok_full"] = int(res_f.ok and (res_f.cd is not None) and (res_f.hd is not None))
            row["exec_ok_full"] = int(row["metric_ok_full"])
            row["reason_full"] = "" if row["metric_ok_full"] == 1 else (res_f.reason or "metric_failed")
        else:
            row["cd_full"] = np.nan
            row["hd_full"] = np.nan
            row["angle_full"] = None
            row["metric_ok_full"] = 0
            row["exec_ok_full"] = 0
            row["reason_full"] = "gt_step_missing"
    else:
        row["metric_ok_full"] = 0
        row["exec_ok_full"] = 0
        if not row.get("reason_full"):
            row["reason_full"] = "pred_step_missing"

    return row


def _normalize_cell_value(val):
    if isinstance(val, (list, tuple, dict)):
        return json.dumps(val, ensure_ascii=False)
    return val


def _repair_worker(args):
    idx, old_row = args
    pid = old_row["group_index"]
    k_idx = old_row["k_index"]
    try:
        new_row = repair_one_row(old_row)
        return {
            "idx": idx,
            "pid": pid,
            "k_idx": k_idx,
            "ok": True,
            "new_row": new_row,
        }
    except Exception as e:
        return {
            "idx": idx,
            "pid": pid,
            "k_idx": k_idx,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def main():
    if not os.path.exists(CANDS_CSV):
        raise FileNotFoundError(CANDS_CSV)

    df = pd.read_csv(CANDS_CSV, low_memory=False)
    print(f"[INFO] loaded rows: {len(df)}")

    repair_indices = []
    for idx, row in df.iterrows():
        if should_retry_metric(row):
            repair_indices.append(idx)

    print(f"[INFO] rows to repair: {len(repair_indices)}")

    if not repair_indices:
        print("[INFO] nothing to repair")
        return

    backup_path = CANDS_CSV + BACKUP_SUFFIX
    if not os.path.exists(backup_path):
        shutil.copy2(CANDS_CSV, backup_path)
        print(f"[INFO] backup saved to: {backup_path}")
    else:
        print(f"[INFO] backup already exists: {backup_path}")

    tasks = [(idx, df.loc[idx].copy()) for idx in repair_indices]

    success_single = 0
    success_full = 0
    fail_single = 0
    fail_full = 0
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_repair_worker, t) for t in tasks]

        total = len(futures)
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res["ok"]:
                print(f"[{n}/{total}] done {res['pid']} k={res['k_idx']}")
            else:
                print(f"[{n}/{total}] failed {res['pid']} k={res['k_idx']}: {res['error']}")
            results.append(res)

    for res in results:
        if not res["ok"]:
            continue

        idx = res["idx"]
        new_row = res["new_row"]

        for col, val in new_row.items():
            df.at[idx, col] = _normalize_cell_value(val)

        if int(new_row.get("metric_ok_single", 0)) == 1:
            success_single += 1
        else:
            fail_single += 1

        if int(new_row.get("metric_ok_full", 0)) == 1:
            success_full += 1
        else:
            fail_full += 1

    for res in results:
        if not res["ok"]:
            fail_single += 1
            fail_full += 1
            print(f"[ERROR] failed on {res['pid']} k={res['k_idx']}: {res['error']}")
            print(res["traceback"])

    tmp_out = CANDS_CSV + TMP_OUT_SUFFIX
    df.to_csv(tmp_out, index=False)
    shutil.move(tmp_out, CANDS_CSV)

    print("[DONE] cands.csv refreshed")
    print(f"[STAT] single success={success_single}, single fail={fail_single}")
    print(f"[STAT] full   success={success_full}, full fail={fail_full}")


if __name__ == "__main__":
    main()