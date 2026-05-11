"""Resolve highlight embedding .npy paths (aligned with stage2 manifest / directory layout)."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Optional


def normalize_sample_id(
    sample_id: Optional[str],
    group_index: str,
    step_id: int | str,
) -> str:
    if sample_id:
        sid = str(sample_id).strip().replace("/", "_")
        sid = re.sub(r"_?step_?(\d+)$", r"_step\1", sid)
        sid = re.sub(r"_+", "_", sid)
        if "_step" in sid:
            return sid
    return f"{group_index}_step{int(step_id)}"


def infer_step_from_group_index(group_index: str) -> int:
    m = re.search(r"step(\d+)", str(group_index))
    return int(m.group(1)) if m else 0


def resolve_highlight_embedding_path(
    row: Mapping[str, Any],
    *,
    source: str,
    gt_embed_dir: Optional[str],
    pred_embed_dir: Optional[str],
) -> Optional[str]:
    """
    Resolution order (same spirit as stage2 manifests):
    1) Column `{source}_embedding_path` with an existing file path
    2) `{gt|pred}_embed_dir / {sample_id}.npy` where sample_id is normalized from group_index (+ optional sample_id column)
    """
    source = (source or "pred").strip().lower()
    if source not in ("gt", "pred"):
        raise ValueError(f"embedding source must be gt or pred, got: {source}")

    key = f"{source}_embedding_path"
    raw = None
    if key in row:
        raw = row.get(key)
    if raw is None:
        for k in row:
            if str(k).lower().replace(" ", "_") == key:
                raw = row[k]
                break

    if raw is not None and str(raw).strip() and str(raw).lower() != "nan":
        p = os.path.abspath(str(raw).strip())
        if os.path.isfile(p):
            return p

    gid = str(row.get("group_index", "")).strip()
    step_id = infer_step_from_group_index(gid)
    sid = normalize_sample_id(row.get("sample_id"), gid, step_id)

    root = gt_embed_dir if source == "gt" else pred_embed_dir
    if root:
        cand = os.path.join(os.path.abspath(root), f"{sid}.npy")
        if os.path.isfile(cand):
            return cand

    return None
