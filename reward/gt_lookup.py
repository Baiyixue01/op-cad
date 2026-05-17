#!/usr/bin/env python3
"""Resolve CAD GT paths with exact group_index deduplication.

The dedup table uses full ids like ``00002_index_1/step0``. For full-sequence
evaluation, resolve that exact id first, then query the existing GT layout with
the canonical id.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional


def _clean_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def load_exact_dedup_map(dedup_csv: str) -> Dict[str, str]:
    """Return {full_group_index: duplicate_of_full_group_index}."""
    if not dedup_csv or not os.path.exists(dedup_csv):
        return {}

    with open(dedup_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        field_map = {name.lower(): name for name in reader.fieldnames}
        group_col = field_map.get("group_index")
        duplicate_col = field_map.get("duplicate_of_group_index")
        if not group_col or not duplicate_col:
            return {}

        mapping: Dict[str, str] = {}
        for row in reader:
            group_index = _clean_cell(row.get(group_col))
            duplicate_of = _clean_cell(row.get(duplicate_col))
            if group_index and duplicate_of:
                mapping[group_index] = duplicate_of
        return mapping


def canonical_group_index(group_index: str, dedup_csv: str, *, max_hops: int = 8) -> Dict[str, object]:
    """Resolve exact full-id duplicates, preserving a short trace for debugging."""
    current = _clean_cell(group_index)
    mapping = load_exact_dedup_map(dedup_csv)
    trace = [current] if current else []
    seen = {current} if current else set()

    for _ in range(max_hops):
        nxt = mapping.get(current, "")
        if not nxt:
            break
        trace.append(nxt)
        if nxt in seen:
            return {
                "query_group_index": group_index,
                "resolved_group_index": nxt,
                "is_duplicate": True,
                "dedup_trace": trace,
                "dedup_cycle": True,
            }
        seen.add(nxt)
        current = nxt

    return {
        "query_group_index": group_index,
        "resolved_group_index": current,
        "is_duplicate": len(trace) > 1,
        "dedup_trace": trace,
        "dedup_cycle": False,
    }


def resolve_gt_for_group_index(
    group_index: str,
    *,
    dedup_csv: str,
    gt_single_step_dir: str,
    op_orient_dir: str,
    gt_single_pc_dir: Optional[str] = None,
    gt_full_pc_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Resolve GT single/full paths after exact dedup replacement."""
    import evaluation as ev

    info = canonical_group_index(group_index, dedup_csv)
    resolved = str(info["resolved_group_index"])

    ev.OP_ORIENT_DIR = op_orient_dir
    ev.GT_SINGLE_PC_DIR = gt_single_pc_dir
    ev.GT_FULL_PC_DIR = gt_full_pc_dir
    ev.DEDUP_CSV = dedup_csv
    ev._dedup_map = None

    gt_single, gt_full = ev.resolve_gt_paths(resolved, gt_single_step_dir)
    info.update(
        {
            "gt_single_path": gt_single or "",
            "gt_full_path": gt_full or "",
        }
    )
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Query CAD GT paths with exact group_index dedup.")
    ap.add_argument("--group-index", required=True)
    ap.add_argument("--dedup-csv", required=True)
    ap.add_argument("--gt-single-step-dir", required=True)
    ap.add_argument("--op-orient-dir", required=True)
    ap.add_argument("--gt-single-pc-dir", default=None)
    ap.add_argument("--gt-full-pc-dir", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    info = resolve_gt_for_group_index(
        args.group_index,
        dedup_csv=args.dedup_csv,
        gt_single_step_dir=args.gt_single_step_dir,
        op_orient_dir=args.op_orient_dir,
        gt_single_pc_dir=args.gt_single_pc_dir,
        gt_full_pc_dir=args.gt_full_pc_dir,
    )
    text = json.dumps(info, ensure_ascii=False, indent=2)
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
