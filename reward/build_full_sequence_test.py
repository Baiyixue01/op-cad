#!/usr/bin/env python3
"""Build full-sequence CAD eval data from step-level split rows.

For each base id in the requested split, this script keeps the largest step and
parses the corresponding COP pre-code file for headers like:

    ## step3: Add ...

The output is JSONL, one full modeling task per base id.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

STEP_RE = re.compile(r"^(?P<base>.+)/step(?P<step>\d+)$")
HEADER_RE = re.compile(r"^\s*##\s*step(?P<step>\d+)\s*:\s*(?P<instruction>.*?)\s*$")


def parse_group_index(group_index: str) -> Tuple[str, int]:
    m = STEP_RE.match(str(group_index).strip())
    if not m:
        raise ValueError(f"bad group_index: {group_index!r}")
    return m.group("base"), int(m.group("step"))


def load_split_ids(path: Path, key: str) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if key not in obj:
        raise KeyError(f"split key {key!r} not found in {path}: {list(obj.keys())}")
    value = obj[key]
    if isinstance(value, dict):
        out: List[str] = []
        for v in value.values():
            if isinstance(v, list):
                out.extend(str(x) for x in v)
        return out
    if isinstance(value, list):
        return [str(x) for x in value]
    raise TypeError(f"split {key!r} must be list or dict of lists")


def pre_code_path(base: str, step: int, root: Path) -> Path:
    return root / f"{base}_step{step}.py"


def parse_step_headers(src: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line in src.splitlines():
        m = HEADER_RE.match(line)
        if m:
            out[int(m.group("step"))] = m.group("instruction").strip()
    return out


def prompt_map(prompt_csv: Path) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(prompt_csv)
    df.columns = [c.lower() for c in df.columns]
    required = {"group_index", "prompt_text", "op"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{prompt_csv} missing columns: {sorted(missing)}")
    rows: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        gid = str(r["group_index"]).strip()
        rows[gid] = {
            "prompt_text": "" if pd.isna(r["prompt_text"]) else str(r["prompt_text"]),
            "op": "" if pd.isna(r["op"]) else str(r["op"]).strip(),
        }
    return rows


def iter_tasks(
    split_ids: Iterable[str],
    prompts: Dict[str, Dict[str, str]],
    pre_code_dir: Path,
) -> Iterable[Dict[str, object]]:
    max_by_base: Dict[str, int] = {}
    for gid in split_ids:
        try:
            base, step = parse_group_index(gid)
        except ValueError:
            continue
        max_by_base[base] = max(step, max_by_base.get(base, -1))

    for base in sorted(max_by_base):
        max_step = max_by_base[base]
        gid = f"{base}/step{max_step}"
        path = pre_code_path(base, max_step, pre_code_dir)
        src = path.read_text(encoding="utf-8") if path.exists() else ""
        header_instructions = parse_step_headers(src)

        steps: List[Dict[str, object]] = []
        for step in range(max_step + 1):
            step_gid = f"{base}/step{step}"
            meta = prompts.get(step_gid, {})
            header_text = header_instructions.get(step, "")
            prompt_text = meta.get("prompt_text") or header_text
            steps.append(
                {
                    "step": step,
                    "group_index": step_gid,
                    "instruction": prompt_text,
                    "header_instruction": header_text,
                    "op": meta.get("op", ""),
                }
            )

        yield {
            "group_id": base,
            "group_index": gid,
            "max_step": max_step,
            "pre_code_path": str(path),
            "pre_code_exists": path.exists(),
            "num_steps": len(steps),
            "instructions": steps,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build full-sequence eval JSONL from COP pre-code headers.")
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--split-key", default="test")
    ap.add_argument("--prompts-csv", required=True)
    ap.add_argument("--pre-code-cop-dir", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    split_ids = load_split_ids(Path(args.split_json), args.split_key)
    prompts = prompt_map(Path(args.prompts_csv))
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for task in iter_tasks(split_ids, prompts, Path(args.pre_code_cop_dir)):
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"[OK] wrote {n} full-sequence tasks -> {out_path}")


if __name__ == "__main__":
    main()
