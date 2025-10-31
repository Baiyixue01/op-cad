#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, traceback
import pandas as pd

# 解析操作行：= base.edges(cq.NearestToPointSelector((x,y,z))).fillet/chamfer(args)
OP_LINE_RE = re.compile(
    r"""=\s*(?P<base>\w+)\.edges\(\s*cq\.NearestToPointSelector\(\s*(?P<pt>\([^)]+\))\s*\)\s*\)\.
        (?P<op>fillet|chamfer)\(\s*(?P<args>[^)]*)\)""",
    re.IGNORECASE | re.VERBOSE
)

# 标签（如：# Fillet_3 / # Chamfer_4）
TAG_RE = re.compile(r"#\s*(?P<kind>Fillet|Chamfer)_(?P<idx>\d+)\s*$", re.IGNORECASE)

# 作为“块边界”的标签：遇到这些就停止当前块扫描，防止误扫到下一个操作
BOUNDARY_RE = re.compile(
    r"""
    # 下一个同类或异类操作标签
    \#\s*(?:Fillet|Chamfer)_(\d+)\s*$ |
    # 典型建模对（根据你的代码风格，可按需扩展）
    \#\s*Sketch-Extrude_pair_\d+\s*$ |
    \#\s*Sketch-Revolve_pair_\d+\s*$ |
    \#\s*Sketch|^result_\d+\s*=    # 宽松兜底：新的大段开始/新的 result 赋值
    """,
    re.IGNORECASE | re.VERBOSE
)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_next_ops(next_op: str):
    if not isinstance(next_op, str) or not next_op.strip():
        return []
    parts = [x.strip() for x in next_op.split("|") if x.strip()]
    out = []
    for p in parts:
        m = re.match(r"(Fillet|Chamfer)\s+(\d+)", p, re.IGNORECASE)
        if m:
            out.append((m.group(1).capitalize(), m.group(2)))
    return out

def find_tag_line_numbers(src_lines, kind: str, idx: str):
    tag = f"# {kind}_{idx}"
    hits = []
    for i, line in enumerate(src_lines):
        if line.strip().lower() == tag.lower():
            hits.append(i)
    return hits

def find_block_end(lines, start_ln: int) -> int:
    """
    返回当前标签块的结束行号的“下一行”（半开区间右端）。
    从 start_ln+1 往后找，遇到块边界（下一个标签/建模块）立即停止。
    若未找到，则返回 len(lines)。
    """
    for i in range(start_ln + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("#") and BOUNDARY_RE.search(line):
            return i
    return len(lines)

def collect_ops_in_block(lines, tag_ln: int):
    """
    在当前标签块内，收集所有操作行（支持 result_4_0 / result_4_1 这种多行）。
    返回：[(op_ln, match_obj), ...]
    """
    end_ln = find_block_end(lines, tag_ln)
    hits = []
    for i in range(tag_ln + 1, end_ln):
        L = lines[i]
        Llow = L.lower()
        if ".edges(" in Llow and (".fillet(" in Llow or ".chamfer(" in Llow):
            m = OP_LINE_RE.search(L)
            if m:
                hits.append((i, m))
    return hits

def safe_exec(code: str, env: dict):
    exec(code, env, env)
    return env

def to_float(s):
    try: return float(s)
    except Exception: return None

def extract_args(op: str, args_str: str):
    vals = [a.strip() for a in args_str.split(",") if a.strip()]
    if op.lower() == "fillet":
        return {"radius": to_float(vals[0]) if vals else None}
    else:
        d1 = to_float(vals[0]) if len(vals) >= 1 else None
        d2 = to_float(vals[1]) if len(vals) >= 2 else None
        return {"d1": d1, "d2": d2}

def edge_record(edge, idx: int):
    try: length = float(edge.Length())
    except Exception: length = None
    try:
        c = edge.Center()
        center = (float(c.x), float(c.y), float(c.z))
    except Exception:
        center = None
    try: g = edge.geomType()
    except Exception: g = None
    try: verts = [(float(v.X), float(v.Y), float(v.Z)) for v in edge.Vertices()]
    except Exception: verts = None
    return {"edge_index": idx, "length": length, "center": center, "geomType": g, "vertices": verts}

def parse_group_for_code_path(group_full: str):
    """
    group_full: '04912_index_6/step3' -> base_group='04912_index_6', step='3'
    仅用于找到 .py 源码路径；不会改变 JSON 里保存的 group_index（原样保留）。
    """
    m = re.match(r"(.+?)/step(\d+)$", group_full, re.IGNORECASE)
    if not m:
        # 兼容无 step 的奇形输入；尽力而为
        return group_full, None
    return m.group(1), m.group(2)

def process_one_op(code_dir: str, group_full: str, kind: str, idx: str, out_dir: str):
    base_group, step = parse_group_for_code_path(group_full)
    code_path = os.path.join(code_dir, f"{base_group}.py")
    if not os.path.exists(code_path):
        raise RuntimeError(f"代码不存在: {code_path}")

    src = read_text(code_path)
    lines = src.splitlines()

    tag_lines = find_tag_line_numbers(lines, kind, idx)
    if not tag_lines:
        raise RuntimeError(f"未在 {code_path} 找到注释标签: # {kind}_{idx}")

    # 仅处理第一个匹配（通常唯一）
    tag_ln = tag_lines[0]
    # —— 从“只取第一行” 改为 “块内所有操作行”
    op_hits = collect_ops_in_block(lines, tag_ln)
    if not op_hits:
        raise RuntimeError(f"{code_path} 标签 # {kind}_{idx} 后未找到操作行")

    # 统一执行：到块内首条操作之前（一次即可）
    first_op_ln = op_hits[0][0]
    prelude = (
        "import cadquery as cq\n"
        "from cadquery import Plane, Vector\n"
        "from functools import reduce\n"
    )
    head_code = "\n".join(lines[:first_op_ln]) + "\n"
    env = {}
    try:
        safe_exec(prelude + head_code, env)
    except Exception as e:
        raise RuntimeError(f"前置代码执行失败：{e}")

    # 聚合本块所有选择到的边
    all_vals = []
    selector_points = []
    calls = []  # 逐次调用记录：base/pt/op/args（便于溯源）

    for _, m in op_hits:
        base_var = m.group("base")
        pt_str   = m.group("pt")
        op_name  = m.group("op")  # fillet | chamfer
        params   = extract_args(op_name, m.group("args"))

        selector_points.append(pt_str)
        call_rec = {
            "base_var": base_var,
            "selector_point": pt_str,
            "op": op_name.lower(),
            "params": params
        }
        calls.append(call_rec)

        pick_expr = f"{base_var}.edges(cq.NearestToPointSelector({pt_str}))"
        try:
            edges = eval(pick_expr, env, env)
            all_vals.extend(list(edges.vals()))
        except Exception as e:
            # 某一条失败，不中断整块；记录到 calls 里
            call_rec["select_error"] = str(e)

    # JSON 里保留完整 group_index（原样）
    rec = {
        "group_index": group_full,              # 原样
        "op_tag": f"{kind}_{idx}",
        "op_kind": kind,
        "selector_points": selector_points,     # 本块内多次选择的点
        "num_edges": len(all_vals),
        "edges": [edge_record(e, i) for i, e in enumerate(all_vals)],
        "calls": calls                          # 每次调用的溯源信息
    }

    # 如果整块内参数一致，额外给个汇总（可选，不作为强依赖）
    if kind.lower() == "fillet":
        uniq = {json.dumps(c["params"], sort_keys=True) for c in calls if "params" in c and c["params"] is not None}
        if len(uniq) == 1:
            rec["radius"] = list({c["params"].get("radius") for c in calls if "params" in c})[0]
    else:
        d1s = {c["params"].get("d1") for c in calls if "params" in c and c["params"] is not None}
        d2s = {c["params"].get("d2") for c in calls if "params" in c and c["params"] is not None}
        if len(d1s) == 1: rec["d1"] = list(d1s)[0]
        if len(d2s) == 1: rec["d2"] = list(d2s)[0]

    # 物理路径：按 base_group/step 分目录（不影响 JSON 字段）
    save_dir = os.path.join(out_dir, base_group, f"step{step}" if step else "stepNA")
    ensure_dir(save_dir)
    out_path = os.path.join(save_dir, f"{kind}_{idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return out_path, rec["num_edges"]

def build_index(out_dir: str):
    """
    按文件结构构建两类索引：
    - by_pair: { "<group_full>::<op_tag>": path }
    - by_kind_idx: { "Fillet_3": [paths...] }
      其中 <group_full> 是 "04912_index_6/step3" 这种完整形式（以目录结构推回）
    """
    index = {"by_pair": {}, "by_kind_idx": {}}
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if not fn.endswith(".json") or fn == "index.json":
                continue
            path = os.path.join(root, fn)         # .../04912_index_6/step3/Fillet_3.json
            rel = os.path.relpath(path, out_dir)  # 04912_index_6/step3/Fillet_3.json
            parts = rel.split(os.sep)
            if len(parts) != 3:
                continue
            base_group, step_dir, op_file = parts
            m = re.match(r"step(\w+)", step_dir, re.IGNORECASE)
            if not m:
                continue
            group_full = f"{base_group}/{step_dir}"      # 恢复 "04912_index_6/step3"
            op_tag = os.path.splitext(op_file)[0]       # Fillet_3
            key = f"{group_full}::{op_tag}"
            index["by_pair"][key] = path
            index["by_kind_idx"].setdefault(op_tag, []).append(path)

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return index

def extract_mode(args):
    ops_df = pd.read_csv(args.ops_csv)
    map_df = pd.read_csv(args.map_csv)
    ops_df.columns = [c.strip().lower() for c in ops_df.columns]
    map_df.columns = [c.strip().lower() for c in map_df.columns]

    # ops: 只要求 group_index, op；map: 只要求 group_index, next_op
    need_ops = {"op", "group_index"}
    need_map = {"group_index", "next_op"}
    miss_ops = need_ops - set(ops_df.columns)
    miss_map = need_map - set(map_df.columns)
    if miss_ops: raise KeyError(f"ops_csv 缺少列: {miss_ops}")
    if miss_map: raise KeyError(f"map_csv 缺少列: {miss_map}")

    # 目标 group：op == chamfer_fillet
    targets = set(str(x) for x in ops_df.loc[ops_df["op"] == "chamfer_fillet", "group_index"].tolist())

    summary = []
    for _, row in map_df.iterrows():
        group_full = str(row["group_index"]).strip()  # 形如 "04912_index_6/step3"
        if group_full not in targets:
            continue

        next_op = str(row.get("next_op", "")).strip()
        pairs = parse_next_ops(next_op)
        if not pairs:
            continue

        for kind, idx in pairs:
            try:
                out_path, n_edges = process_one_op(
                    code_dir=args.code_dir,
                    group_full=group_full,
                    kind=kind,
                    idx=idx,
                    out_dir=args.out_dir,
                )
                summary.append({
                    "group_index": group_full, "op_tag": f"{kind}_{idx}",
                    "ok": True, "num_edges": n_edges, "out_path": out_path
                })
                print(f"[OK] {group_full}/{kind}_{idx} -> {out_path} (edges={n_edges})")
            except Exception as e:
                summary.append({
                    "group_index": group_full, "op_tag": f"{kind}_{idx}",
                    "ok": False, "reason": str(e)
                })
                print(f"[ERR] {group_full}/{kind}_{idx}: {e}")
                if args.verbose: traceback.print_exc()

    ensure_dir(args.out_dir)
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False, encoding="utf-8")
    print(f"\n[SUM] 写入汇总: {os.path.join(args.out_dir, 'summary.csv')}")

    build_index(args.out_dir)
    print(f"[INDEX] 已生成: {os.path.join(args.out_dir, 'index.json')}")

def load_index(out_dir: str):
    idx_path = os.path.join(out_dir, "index.json")
    if not os.path.exists(idx_path):
        print("[INFO] 未发现 index.json，正在重建 …")
        return build_index(out_dir)
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)

def query_mode(args):
    index = load_index(args.out_dir)
    results = []

    # 精确：group_full + op_tag
    if args.group and args.op_tag:
        key = f"{args.group}::{args.op_tag}"
        path = index["by_pair"].get(key)
        if path:
            results.append(os.path.join(args.out_dir, path))

    # 模糊：按 kind+idx 或 op_tag
    if not results:
        if args.op_tag:
            paths = index["by_kind_idx"].get(args.op_tag, [])
            results.extend(os.path.join(args.out_dir, p) for p in paths)
        elif args.kind and args.idx is not None:
            op_tag = f"{args.kind.capitalize()}_{args.idx}"
            paths = index["by_kind_idx"].get(op_tag, [])
            results.extend(os.path.join(args.out_dir, p) for p in paths)

    if not results:
        print("[NONE] 未匹配到结果")
        return

    for p in results:
        print(f"\n===== {p} =====")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[ERR] 打开失败: {e}")

def main():
    ap = argparse.ArgumentParser(description="Extract & Query GT edges for Fillet/Chamfer (group_index 原样保存)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ext = sub.add_parser("extract", help="抽取 GT JSON")
    ap_ext.add_argument("--ops-csv", required=True)
    ap_ext.add_argument("--map-csv", required=True)
    ap_ext.add_argument("--code-dir", required=True)
    ap_ext.add_argument("--out-dir", required=True)
    ap_ext.add_argument("--verbose", action="store_true")

    ap_q = sub.add_parser("query", help="查询 GT JSON")
    ap_q.add_argument("--out-dir", required=True)
    ap_q.add_argument("--group", help='完整 group_index，如 "04912_index_6/step3"')
    ap_q.add_argument("--op-tag", help="例如 Chamfer_4 / Fillet_3")
    ap_q.add_argument("--kind", choices=["Chamfer", "Fillet"], help="类别查询")
    ap_q.add_argument("--idx", type=int, help="编号，如 5 表示 Chamfer_5")

    args = ap.parse_args()
    if args.cmd == "extract":
        extract_mode(args)
    else:
        query_mode(args)

if __name__ == "__main__":
    main()
