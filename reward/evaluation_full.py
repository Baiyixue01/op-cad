#!/usr/bin/env python3
"""Full-sequence CAD evaluation.

This script consumes JSONL built by reward/build_full_sequence_test.py. Each row
contains all step instructions for one CAD id. The model is asked to generate a
complete CadQuery script for the whole sequence, then the final exported STEP is
compared against the cumulative GT STEP of the largest step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import evaluation as ev
import gt_lookup
from model_call import call_model as cm
from model_call.highlight_paths import resolve_highlight_embedding_path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].removeprefix("python").strip()
    return text


def last_result_var(code: str) -> str:
    result = None
    numbered = None
    any_lhs = None
    for line in code.splitlines():
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
        if not m:
            continue
        lhs = m.group(1)
        any_lhs = lhs
        if lhs == "result":
            result = lhs
        elif re.fullmatch(r"result_\d+", lhs):
            numbered = lhs
    return result or numbered or any_lhs or "result"


def build_full_prompt(task: Dict[str, object], *, use_highlight_embedding: bool = False) -> str:
    lines = [
        "### Role",
        "You are an expert CAD modeling assistant specialized in CadQuery.",
        "Generate one complete executable CadQuery Python script for the full modeling sequence.",
        "",
    ]
    if use_highlight_embedding:
        from model_call.prompt import EMBEDDING_NOTICE

        lines += ["### Embedding Guidance", EMBEDDING_NOTICE, ""]

    lines += [
        "### Task",
        "Build the final CAD model by applying every step in order.",
        "Use the step instructions below as the only modeling requirements.",
        "",
        "### Step Instructions",
    ]
    for item in task.get("instructions", []) or []:
        step = int(item.get("step", 0))
        instruction = str(item.get("instruction", "")).strip()
        lines.append(f"## step{step}: {instruction}")

    lines += [
        "",
        "### Output Requirements",
        "1. Output only Python code, with no Markdown or explanations.",
        "2. The code must be directly executable from an empty Python process.",
        "3. Import CadQuery as `cq` and import `Plane` and `Vector`.",
        "4. Preserve clear `## stepN: ...` comments before each step.",
        "5. Build steps cumulatively. The final solid must be assigned to `result` or the last `result_N` variable.",
        "6. Do not export files; the evaluator will append the export statement.",
    ]
    return "\n".join(lines)


def build_export_script(code: str, out_step: str) -> str:
    code = strip_code_fences(code)
    lhs = last_result_var(code)
    head = (
        "import cadquery as cq\n"
        "from cadquery import Plane, Vector, Workplane\n"
        "from functools import reduce\n"
    )
    if "import cadquery as cq" in code:
        head = ""
    tail = f"""

# === FULL_SEQUENCE export ===
def _export_any(obj, path):
    try:
        cq.exporters.export(obj.val(), path)
    except Exception:
        cq.exporters.export(obj, path)

_export_any({lhs}, r"{out_step}")
"""
    return (head + "\n" + code.rstrip() + tail).strip() + "\n"


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_or_missing(pred_path: str, gt_path: Optional[str]):
    from utils.compute_3D import MetricsResult

    if not gt_path:
        return MetricsResult(None, None, None, ok=False, reason="gt_full_missing")
    if not os.path.exists(pred_path):
        return MetricsResult(None, None, None, ok=False, reason="pred_full_missing")
    return ev._safe_get_cd_hd(pred_step_path=pred_path, gt_step_path=gt_path, angles=[0])


def eval_one(task: Dict[str, object], args) -> Dict[str, object]:
    group_id = str(task["group_id"])
    group_index = str(task["group_index"])
    task_dir = os.path.join(args.out_root, "full_traces", group_id)
    ensure_dir(task_dir)

    prompt = build_full_prompt(task, use_highlight_embedding=bool(args.highlight_embedding))
    prompt_path = os.path.join(task_dir, "prompt.txt")
    Path(prompt_path).write_text(prompt, encoding="utf-8")

    highlight_path = None
    if args.highlight_embedding:
        row_like = {"group_index": group_index}
        highlight_path = resolve_highlight_embedding_path(row_like, embed_dir=args.embed_dir)

    cands = cm.get_model_candidates(
        prompt,
        args.k,
        thinking=args.thinking,
        highlight_embedding_path=highlight_path,
    )

    best_row = None
    cand_rows: List[Dict[str, object]] = []
    gt_info = gt_lookup.resolve_gt_for_group_index(
        group_index,
        dedup_csv=args.dedup_csv,
        gt_single_step_dir=args.gt_single_step_dir,
        op_orient_dir=args.op_orient_dir,
        gt_single_pc_dir=args.gt_single_pc_dir,
        gt_full_pc_dir=args.gt_full_pc_dir,
    )
    gt_full = str(gt_info.get("gt_full_path") or "")
    resolved_group_index = str(gt_info.get("resolved_group_index") or group_index)

    for k_idx, item in enumerate(cands):
        code = item.get("code", "") if isinstance(item, dict) else str(item or "")
        gen_meta = item if isinstance(item, dict) else {}
        code_path = os.path.join(task_dir, f"k{k_idx}_full.py")
        step_path = os.path.join(task_dir, f"k{k_idx}_full.step")
        script = build_export_script(code, step_path)
        Path(code_path).write_text(script, encoding="utf-8")

        ok_exec, _, err_exec = ev.safe_exec_from_path(code_path)
        pred_exists = ok_exec and os.path.exists(step_path)
        res = metric_or_missing(step_path, gt_full) if pred_exists else metric_or_missing("", gt_full)

        reason = ""
        if not ok_exec:
            reason = f"exec_error:{(err_exec.splitlines()[-1] if err_exec else 'unknown')}"
        if not res.ok and getattr(res, "reason", ""):
            reason = (reason + "; " if reason else "") + res.reason

        row = {
            "group_id": group_id,
            "group_index": group_index,
            "gt_query_group_index": group_index,
            "gt_resolved_group_index": resolved_group_index,
            "gt_is_duplicate": int(bool(gt_info.get("is_duplicate"))),
            "gt_dedup_cycle": int(bool(gt_info.get("dedup_cycle"))),
            "gt_dedup_trace": " -> ".join(str(x) for x in gt_info.get("dedup_trace", []) or []),
            "max_step": task.get("max_step"),
            "num_steps": task.get("num_steps"),
            "k_index": k_idx,
            "exec_ok_full": int(ok_exec),
            "pred_full_exists": int(pred_exists),
            "pred_full_path": step_path if pred_exists else "",
            "gt_full_path": gt_full or "",
            "cd_full": res.cd,
            "hd_full": res.hd,
            "angle_full": getattr(res, "best_euler_angle", None),
            "metric_ok_full": int(res.ok and res.cd is not None and res.hd is not None),
            "reason_full": reason,
            "gen_backend": gen_meta.get("backend", ""),
            "gen_input_tokens": gen_meta.get("input_tokens"),
            "gen_output_tokens": gen_meta.get("output_tokens"),
            "gen_total_tokens": gen_meta.get("total_tokens"),
            "gen_error": gen_meta.get("err", ""),
            "gen_prompt_len": len(prompt),
            "prompt_path": prompt_path,
            "code_path": code_path,
            "highlight_embedding_path": highlight_path or "",
        }
        cand_rows.append(row)
        if best_row is None or (
            row["metric_ok_full"] > best_row["metric_ok_full"]
            or (row["metric_ok_full"] == best_row["metric_ok_full"] and _nan_to_inf(row["cd_full"]) < _nan_to_inf(best_row["cd_full"]))
        ):
            best_row = row

    summary = dict(best_row or {})
    summary["k_index"] = "summary_best"
    summary["n_candidates"] = len(cand_rows)
    return {"cands": cand_rows, "summary": [summary] if summary else []}


def _nan_to_inf(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("inf")
    except Exception:
        return float("inf")


def configure_runtime(args) -> None:
    cm.set_runtime_config(
        gen_mode=args.gen_mode,
        provider=args.provider,
        vllm_endpoint_key=args.vllm_endpoint_key,
        openai_model=args.openai_model,
        http_model=args.http_model,
        temperature=args.gen_temperature,
        timeout_s=args.gen_timeout,
    )
    ev.OP_ORIENT_DIR = args.op_orient_dir
    ev.GT_FULL_PC_DIR = args.gt_full_pc_dir
    ev.GT_SINGLE_PC_DIR = args.gt_single_pc_dir
    ev.DEDUP_CSV = args.dedup_csv
    ev._dedup_map = None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate full-sequence CAD generation.")
    ap.add_argument("--full-jsonl", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--gt-single-step-dir", required=True)
    ap.add_argument("--op-orient-dir", required=True)
    ap.add_argument("--dedup-csv", required=True)
    ap.add_argument("--gt-single-pc-dir", default=None)
    ap.add_argument("--gt-full-pc-dir", default=None)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--gen-mode", choices=["local", "local-highlight", "api", "auto"], default=None)
    ap.add_argument("--provider", choices=["openai", "http", "local", "siliconflow", "vllm"], default=None)
    ap.add_argument("--vllm-endpoint-key", default="port1")
    ap.add_argument("--openai-model", default=None)
    ap.add_argument("--http-model", default=None)
    ap.add_argument("--gen-temperature", type=float, default=None)
    ap.add_argument("--gen-timeout", type=int, default=None)
    ap.add_argument("--thinking", action="store_true", default=False)
    ap.add_argument("--highlight-embedding", action="store_true", default=False)
    ap.add_argument("--embed-dir", default=None)
    args = ap.parse_args()

    configure_runtime(args)
    ensure_dir(args.out_root)
    cands_path = os.path.join(args.out_root, "cands.csv")
    summary_path = os.path.join(args.out_root, "summary.csv")

    tasks = read_jsonl(args.full_jsonl)
    if args.limit:
        tasks = tasks[: args.limit]

    done = set()
    if args.resume and os.path.exists(summary_path):
        try:
            old = pd.read_csv(summary_path)
            done = set(old["group_id"].astype(str))
        except Exception:
            done = set()

    for idx, task in enumerate(tasks, 1):
        if str(task["group_id"]) in done:
            continue
        print(f"[FULL] {idx}/{len(tasks)} {task['group_id']} max_step={task.get('max_step')}")
        out = eval_one(task, args)
        ev._append_csv(cands_path, out["cands"])
        ev._append_csv(summary_path, out["summary"])

    print(f"[done] full eval saved: {args.out_root}")


if __name__ == "__main__":
    main()
