# -*- coding: utf-8 -*-
# 预处理加速版（I/O 缓存 + 索引 + itertuples + 逐组注释）
# 建议在运行前设置：
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ===================== 手动改这里 =====================
MASTER_CSV    = "/home/baiyixue/project/op-cad/data/prompt.csv"                 # 含 group_index 的总表
OPS_CSV       = "/home/baiyixue/project/op-cad/data/pre_next_op_info.csv"       # 含 group_index, previous_op 的表
JSON_DIR      = "/data/baiyixue/CAD/cleaned_json"                                # 清洗后的 JSON 根目录
OUT_DIR       = "/home/baiyixue/project/op-cad/data/pre_code"                    # 预处理代码输出目录
OUT_INDEX_CSV = "/home/baiyixue/project/op-cad/data/prelude_index.csv"
OVERWRITE     = True                                                             # 已存在是否覆盖

# —— 新增：注释插入相关配置 ——
STEP_FILES_DIR = "/data/baiyixue/CAD/step_files"                                 # 含 group_info.txt 的根目录
DEDUP_CSV      = "/home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv"
LANG_DESC_CSV  = "/home/baiyixue/project/op-cad/data/prompt.csv"                # 含 group_index 或 id_index 与 prompt_text
INSERT_NL_COMMENTS = True                                                        # 是否在每组 step 前插入 “## stepN: prompt_text”
if INSERT_NL_COMMENTS:
    OUT_DIR = "/home/baiyixue/project/op-cad/data/pre_code_cop"

# ===================== 依赖 =====================
import os, re, json, glob, hashlib, ast
import pandas as pd
from typing import Optional, List, Dict, Tuple, Set
from tqdm import tqdm
from functools import lru_cache
from pathlib import Path

os.makedirs(OUT_DIR, exist_ok=True)

# ===================== 正则预编译 & 小工具 =====================
STEP_TAIL_RE    = re.compile(r"/step(\d+)$")
MULTISPACE_RE   = re.compile(r"\s+")
DICT_LINE_RE    = re.compile(r"^step(\d+)\s*:\s*(\[.*\])\s*$")

def norm_name(s: str) -> str:
    """名称清洗：去两端空白、把内部多空格压一格"""
    return MULTISPACE_RE.sub(" ", str(s).strip())

def sha10(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def _gid_to_filename(gid: str, out_dir: str) -> str:
    """
    现有规则：输出文件名是 gid.replace('/', '_') + '.py'
    例如 gid='00004/index/2/step1' -> '00004_index_2_step1.py'
    """
    return os.path.join(out_dir, f"{gid.replace('/','_')}.py")

def build_existing_gid_set(
    candidate_gids,
    out_dir: str,
    out_index_csv: Optional[str] = None
) -> Set[str]:
    """
    返回：已经“有产出文件”的 group_index 集合。
    - 扫描 out_dir 下的 *.py（如 00004_index_2_step1.py）
    - 若有 OUT_INDEX_CSV，则用其中 prev_code_path 存在性再确认一遍
    """
    existing: Set[str] = set()

    # 1) 扫描输出目录中的 *.py 文件名集合
    existing_files = {os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "*.py"))}

    # 2) 按同一映射规则检查候选 gid 是否已存在对应文件
    for gid in candidate_gids:
        expect_name = f"{str(gid).replace('/','_')}.py"
        if expect_name in existing_files:
            existing.add(str(gid))

    # 3)（可选）再用索引表加一道保险
    if out_index_csv and os.path.exists(out_index_csv):
        try:
            df_done = pd.read_csv(out_index_csv)
            cols = [c.lower() for c in df_done.columns]
            df_done.columns = cols
            if "group_index" in cols and "prev_code_path" in cols:
                for gi, pth in zip(df_done["group_index"].astype(str), df_done["prev_code_path"].astype(str)):
                    pth = (pth or "").strip()
                    if pth and os.path.exists(pth):
                        existing.add(gi)
        except Exception as e:
            print(f"[WARN] read OUT_INDEX_CSV failed: {e}")

    return existing

# ===================== 一次性预加载（去重表 & 语言描述） =====================
_DEDUP_MAP: Dict[str, str] = {}
if os.path.exists(DEDUP_CSV):
    _dedup_df = pd.read_csv(DEDUP_CSV)
    _dedup_df.columns = [c.lower() for c in _dedup_df.columns]
    if "group_index" in _dedup_df.columns and "duplicate_of_group_index" in _dedup_df.columns:
        _DEDUP_MAP = dict(zip(_dedup_df["group_index"].astype(str), _dedup_df["duplicate_of_group_index"].astype(str)))

def resolve_canonical_group_index_fast(g: str) -> str:
    v = _DEDUP_MAP.get(str(g))
    if v and str(v).lower() != "nan":
        return str(v).strip()
    return g

_LANG_MAP_GROUP: Dict[str, str] = {}
_LANG_MAP_ID: Dict[str, str] = {}
if os.path.exists(LANG_DESC_CSV):
    _lang = pd.read_csv(LANG_DESC_CSV)
    _lang.columns = [c.lower() for c in _lang.columns]
    if "prompt_text" in _lang.columns:
        if "group_index" in _lang.columns:
            for gi, txt in zip(_lang["group_index"].astype(str), _lang["prompt_text"].astype(str)):
                _LANG_MAP_GROUP[gi] = txt
        if "id_index" in _lang.columns:
            for ii, txt in zip(_lang["id_index"].astype(str), _lang["prompt_text"].astype(str)):
                _LANG_MAP_ID[ii] = txt

def get_step_prompt(canonical_gid_list: List[str]) -> Dict[str, str]:
    """
    输入若干 '.../stepN'，输出 { 'N': prompt_text }
    兼容 group_index / id_index 两种键
    """
    out: Dict[str, str] = {}
    for gi in canonical_gid_list:
        m = STEP_TAIL_RE.search(gi)
        if not m:
            continue
        st = m.group(1)  # 'N'
        txt = _LANG_MAP_GROUP.get(gi)
        if not txt:
            ii = gi.replace("/", "_")
            txt = _LANG_MAP_ID.get(ii)
        if txt:
            out[st] = str(txt).strip()
    return out

# ===================== JSON 文件索引 & 缓存解析 =====================
_JSON_INDEX_BUILT = False
_JSON_NAME2PATH: Dict[str, str] = {}

def _build_json_index():
    global _JSON_INDEX_BUILT, _JSON_NAME2PATH
    if _JSON_INDEX_BUILT:
        return
    base = Path(JSON_DIR)
    if base.exists():
        for p in base.rglob("*.json"):
            _JSON_NAME2PATH[p.name] = str(p)
    _JSON_INDEX_BUILT = True

@lru_cache(maxsize=4096)
def find_json_by_group_index_fast(group_index: str) -> Optional[str]:
    _build_json_index()
    base = STEP_TAIL_RE.sub("", group_index.strip())   # 去掉 /stepN
    filename1 = f"{base}.json"
    filename2 = f"{base.replace('/','_')}.json"
    if filename1 in _JSON_NAME2PATH:
        return _JSON_NAME2PATH[filename1]
    if filename2 in _JSON_NAME2PATH:
        return _JSON_NAME2PATH[filename2]
    # 兜底轻量匹配（只做一次，之后走缓存）
    for k, v in _JSON_NAME2PATH.items():
        if base in k or base.replace('/', '_') in k:
            return v
    return None

@lru_cache(maxsize=2048)
def parse_json_fast(json_file: str) -> Tuple[Dict, ...]:
    with open(json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    parsed_data: List[Dict] = []
    ap = parsed_data.append
    for item in data:
        for key, value in item.items():
            kn = str(key)
            if "Sketch-Extrude" in kn:
                ap({
                    "name": kn,
                    "origin": value.get("origin"),
                    "normal": value.get("normal"),
                    "x_axis": value.get("x_axis"),
                    "y_axis": value.get("y_axis"),
                    "operation": value.get("operation"),
                    "type": value.get("type"),
                    "extent_one": value.get("extent_one"),
                    "extent_two": value.get("extent_two"),
                    "profile": value.get("Profile", {}),
                })
            elif "Sketch-Revolve" in kn:
                ap({
                    "name": kn,
                    "origin": value.get("origin"),
                    "normal": value.get("normal"),
                    "x_axis": value.get("x_axis"),
                    "y_axis": value.get("y_axis"),
                    "operation": value.get("operation"),
                    "merge": value.get("merge"),
                    "axis_start": value.get("axis_start"),
                    "axis_end": value.get("axis_end"),
                    "angle": value.get("angle"),
                    "profile": value.get("Profile", {}),
                })
            elif "Fillet" in kn:
                ap({"name": kn, "edge": value.get("edge"), "radius": value.get("radius")})
            elif "Chamfer" in kn:
                ap({
                    "name": kn, "edge": value.get("edge"),
                    "distance_one": value.get("distance_one"),
                    "distance_two": value.get("distance_two"),
                    "chamfer_type": value.get("chamfer_type"),
                })
    return tuple(parsed_data)

# ===================== group_info.txt 缓存解析 =====================
_GROUPINFO_CACHE: Dict[str, Tuple[Dict[int, List[int]], Dict[str, int]]] = {}

def parse_group_info_txt_fast(path: str) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
    hit = _GROUPINFO_CACHE.get(path)
    if hit is not None:
        return hit
    step2idxs: Dict[int, List[int]] = {}
    op2step: Dict[str, int] = {}
    if not os.path.exists(path):
        _GROUPINFO_CACHE[path] = (step2idxs, op2step)
        return step2idxs, op2step

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = DICT_LINE_RE.match(s)
            if not m:
                continue
            stepn = int(m.group(1))
            try:
                arr = ast.literal_eval(m.group(2))
                idxs: List[int] = []
                contains_line_or_arc = False
                for d in arr:
                    if isinstance(d, dict) and d:
                        (k, v) = next(iter(d.items()))
                        try:
                            idxs.append(int(k))
                        except:
                            pass
                        op2step[norm_name(str(v))] = stepn
                step2idxs[stepn] = idxs
            except:
                pass

    _GROUPINFO_CACHE[path] = (step2idxs, op2step)
    return step2idxs, op2step

# ===================== 代码生成 =====================
def _build_sketch_lines(profile: Dict) -> List[str]:
    """
    把 Profile 里的 loops 转为 CadQuery 的 2D 草图命令序列。
    注意：每个 loop 结束后若包含 Line/Arc，补一次 close()。
    """
    lines: List[str] = []
    for _, loop in profile.items():
        has_open = False
        for shape in loop:
            tp = shape.get("type")
            if tp == "Circle":
                center = tuple(shape["center"]); radius = shape["radius"]
                lines.append(f"wp = wp.moveTo{center}.circle({radius})")
            elif tp == "Line":
                start = tuple(shape["start"]); end = tuple(shape["end"])
                lines.append(f"wp = wp.moveTo{start}.lineTo{end}")
                has_open = True
            elif tp == "Arc":
                start = tuple(shape["start"]); mid = tuple(shape["mid"]); end = tuple(shape["end"])
                lines.append(f"wp = wp.moveTo{start}.threePointArc({mid}, {end})")
                has_open = True
        if has_open:
            lines.append("wp = wp.close()")
    return lines

def generate_code(parsed_data, include_export=False, step_comment_inserter=None) -> str:
    """
    step_comment_inserter: 可选函数 f(op_name_norm) -> Optional[str]
      若返回字符串，则在该操作前插入一行注释，如 "## step1: Remove three pockets"
    """
    code_lines: List[str] = [
        "import cadquery as cq",
        "from cadquery import Plane, Vector",
        "from functools import reduce"
    ]
    results = []

    for index, data in enumerate(parsed_data):
        name = data["name"].replace(" ", "_")
        name_norm = norm_name(data["name"])

        # —— 每组 step 的第一条 op 前插注释 —— #
        if callable(step_comment_inserter):
            cm = step_comment_inserter(name_norm)
            if isinstance(cm, str) and cm:
                code_lines.append(f"\n{cm}")

        if "Sketch-Revolve" in name:
            origin = tuple(data["origin"])
            normal = tuple(data["normal"])
            x_axis = tuple(data["x_axis"])
            operation = data["operation"]
            profile = data["profile"]
            merge = data.get("merge", True)
            axis_start = tuple(data["axis_start"])
            axis_end = tuple(data["axis_end"])
            angle = data["angle"]

            code_lines.append(f"\n# {name}")
            code_lines.append(f"normal_vector = Vector{normal}")
            code_lines.append(f"x_dir = Vector{x_axis}")
            code_lines.append(f"origin = {origin}")
            code_lines.append("custom_plane = Plane(origin=origin, normal=normal_vector, xDir=x_dir)")
            code_lines.append("wp = cq.Workplane(inPlane=custom_plane)")
            code_lines.extend(_build_sketch_lines(profile))

            revolve_code = f"wp.revolve({angle}, {axis_start}, {axis_end})"
            if operation == 1:  # new body
                if index == 0:
                    code_lines.append(f"result_{index} = {revolve_code}")
                else:
                    if not merge:
                        revolve_code = f"wp.revolve({angle}, {axis_start}, {axis_end}, combine=False)"
                    code_lines.append(f"result_{index} = result_{index - 1}.union({revolve_code})")
            else:  # cut
                if index > 0:
                    code_lines.append(f"result_{index} = result_{index - 1}.cut({revolve_code})")
                else:
                    code_lines.append(f"result_{index} = {revolve_code}  # Initial object for subtraction")
            results.append(f"result_{index}")

        elif "Sketch-Extrude" in name:
            origin = tuple(data["origin"])
            normal = tuple(data["normal"])
            x_axis = tuple(data["x_axis"])
            operation = data["operation"]
            profile = data["profile"]
            extent_one = data["extent_one"]
            extent_two = data["extent_two"]

            code_lines.append(f"\n# {name}")
            code_lines.append(f"normal_vector = Vector{normal}")
            code_lines.append(f"x_dir = Vector{x_axis}")
            code_lines.append(f"origin = {origin}")
            code_lines.append("custom_plane = Plane(origin=origin, normal=normal_vector, xDir=x_dir)")
            code_lines.append("wp = cq.Workplane(inPlane=custom_plane)")
            code_lines.extend(_build_sketch_lines(profile))

            if data["type"] == 1:      # 对称拉伸
                extrude_code = f"wp.extrude({extent_one}, both=True)"
            elif data["type"] == 2:    # 双向不等
                code_lines.append(f"extent_one = wp.extrude({extent_one})")
                code_lines.append("wp = cq.Workplane(inPlane=custom_plane)")  # 重新绘制
                code_lines.extend(_build_sketch_lines(profile))
                code_lines.append(f"extent_two = wp.extrude({extent_two})")
                extrude_code = f"extent_one.union(extent_two)"
            else:                      # 单向
                extrude_code = f"wp.extrude({extent_one})" if extent_one else f"wp.extrude({extent_two})"

            if operation == 0:         # new body
                if index == 0:
                    code_lines.append(f"result_{index} = {extrude_code}")
                else:
                    code_lines.append(f"result_{index} = result_{index - 1}.union({extrude_code})")
            elif operation == 2:       # cut
                if index > 0:
                    code_lines.append(f"result_{index} = result_{index - 1}.cut({extrude_code})")
                else:
                    code_lines.append(f"result_{index} = {extrude_code}  # Initial object for subtraction")
            elif operation == 3:       # intersect
                if index > 0:
                    code_lines.append(f"result_{index} = result_{index - 1}.intersect({extrude_code})")
                else:
                    code_lines.append(f"result_{index} = {extrude_code}  # Initial object for subtraction")
            else:                      # join/union
                if index > 0:
                    code_lines.append(f"result_{index} = result_{index - 1}.union({extrude_code})")
                else:
                    code_lines.append(f"result_{index} = {extrude_code}  # Initial object for subtraction")
            results.append(f"result_{index}")

        elif "Fillet" in name:
            edges = [tuple(e) for e in data["edge"]]
            radius = data["radius"]
            code_lines.append(f"\n# {name}")
            if len(edges) > 1:
                shapes = []
                for i, edge in enumerate(edges):
                    code_lines.append(f"result_{index}_{i} = result_{index - 1}.edges(cq.NearestToPointSelector({edge})).fillet({radius})")
                    shapes.append(f"result_{index}_{i}")
                code_lines.append(f"result_{index} = reduce(lambda a, b: a.intersect(b), [{', '.join(shapes)}])")
            else:
                code_lines.append(f"result_{index} = result_{index - 1}.edges(cq.NearestToPointSelector({edges[0]})).fillet({radius})")

        elif "Chamfer" in name:
            edges = [tuple(e) for e in data["edge"]]
            d1 = data["distance_one"]; d2 = data["distance_two"]; t = data["chamfer_type"]
            code_lines.append(f"\n# {name}")
            if len(edges) > 1:
                shapes = []
                for i, edge in enumerate(edges):
                    if t == 0:
                        code_lines.append(f"result_{index}_{i} = result_{index - 1}.edges(cq.NearestToPointSelector({edge})).chamfer({d1})")
                    else:
                        code_lines.append(f"result_{index}_{i} = result_{index - 1}.edges(cq.NearestToPointSelector({edge})).chamfer({d1}, {d2})")
                    shapes.append(f"result_{index}_{i}")
                code_lines.append(f"result_{index} = reduce(lambda a, b: a.intersect(b), [{', '.join(shapes)}])")
            else:
                if t == 0:
                    code_lines.append(f"result_{index} = result_{index - 1}.edges(cq.NearestToPointSelector({edges[0]})).chamfer({d1})")
                else:
                    code_lines.append(f"result_{index} = result_{index - 1}.edges(cq.NearestToPointSelector({edges[0]})).chamfer({d1}, {d2})")

    if include_export and results:
        code_lines.append(f"{results[-1]}.val().exportStl('output.stl')")
    return "\n".join(code_lines)

# ===================== 主流程 =====================
def main():
    master = pd.read_csv(MASTER_CSV)
    ops    = pd.read_csv(OPS_CSV)
    master.columns = [c.lower() for c in master.columns]
    ops.columns    = [c.lower() for c in ops.columns]
    if "group_index" not in master.columns or "group_index" not in ops.columns:
        raise KeyError("两个CSV都需要包含列 `group_index`。")
    if "previous_op" not in ops.columns:
        raise KeyError("ops.csv 需要包含列 `previous_op`。")

    merged = master.merge(ops[["group_index","previous_op"]], on="group_index", how="left")

    # 先构建“已完成”的 group_index 集合
    candidate_gids = merged["group_index"].astype(str).tolist()
    done_gids = build_existing_gid_set(candidate_gids, OUT_DIR, OUT_INDEX_CSV)

    rows = []

    # 使用 itertuples 提升性能
    for r in tqdm(merged.itertuples(index=False), total=len(merged),
                    desc="Processing groups", ncols=100, miniters=200, mininterval=0.5):
        gid = str(getattr(r, "group_index"))
        prev_ops_str = str(getattr(r, "previous_op", "") or "")

        # ---- 断点续跑：已存在则跳过 ----
        if gid in done_gids:
            rows.append({
                "group_index": gid,
                "prev_code_path": _gid_to_filename(gid, OUT_DIR),
                "prev_code_sha10": "",                 # 此处可留空；如需可读文件后再算
                "missing_ops": "skipped_exists",
                "json_path": "",
                "n_ops": ""
            })
            continue

        if not prev_ops_str.strip():
            rows.append({"group_index": gid, "prev_code_path": "", "missing_ops": "no_previous_ops", "json_path": ""})
            continue

        ops_list = [norm_name(x) for x in prev_ops_str.split("|") if norm_name(x)]
        if not ops_list:
            rows.append({"group_index": gid, "prev_code_path": "", "missing_ops": "empty_after_split", "json_path": ""})
            continue

        json_path = find_json_by_group_index_fast(gid)
        if not json_path:
            rows.append({"group_index": gid, "prev_code_path": "", "missing_ops": "json_not_found", "json_path": ""})
            continue

        parsed_all = parse_json_fast(json_path)

        # 标准化 name 索引（一次性）
        name2op: Dict[str, Dict] = {}
        for d in parsed_all:
            k = norm_name(d["name"])
            name2op[k] = d
            name2op[k.replace("_", " ")] = d  # 兼容下划线/空格差异

        filtered_ops, missing = [], []
        for nm in ops_list:
            hit = name2op.get(nm)
            if hit is None:
                alt = nm.replace("_", " ")
                hit = name2op.get(norm_name(alt))
            if hit is not None:
                filtered_ops.append(hit)
            else:
                missing.append(nm)

        if not filtered_ops:
            rows.append({"group_index": gid, "prev_code_path": "", "missing_ops": ";".join(missing) or "all_missing", "json_path": json_path})
            continue

        # —— 注释插入器（可关） ——
        if INSERT_NL_COMMENTS:
            group_base = STEP_TAIL_RE.sub("", gid.strip())
            m = STEP_TAIL_RE.search(gid)
            cur_step = int(m.group(1)) if m else 10**9

            group_dir = os.path.join(STEP_FILES_DIR, group_base.replace("/", os.sep))
            gi_path   = os.path.join(group_dir, "group_info.txt")
            step2idxs, op2step = parse_group_info_txt_fast(gi_path)

            canonical_gid_list = []
            for step in sorted(step2idxs):
                if step < cur_step:
                    canonical_gid = resolve_canonical_group_index_fast(f"{group_base}/step{step}")
                    canonical_gid_list.append(canonical_gid)

            step_prompts = get_step_prompt(canonical_gid_list)
            seen_steps = set()
            def step_comment_inserter(op_name_norm: str):
                st = op2step.get(op_name_norm)
                if st is None or st in seen_steps:
                    return None
                seen_steps.add(st)
                txt = step_prompts.get(str(st), "")
                return f"## step{st}: {txt.strip()}" if txt else f"## step{st}:"

            code = generate_code(filtered_ops, include_export=False, step_comment_inserter=step_comment_inserter)
        else:
            code = generate_code(filtered_ops, include_export=False)

        out_py = os.path.join(OUT_DIR, f"{gid.replace('/','_')}.py")

        # 内容哈希一致则跳过写盘，省 I/O
        need_write = True
        if os.path.exists(out_py) and not OVERWRITE:
            need_write = False
        elif os.path.exists(out_py) and OVERWRITE:
            try:
                with open(out_py, "r", encoding="utf-8") as rf:
                    old = rf.read()
                if sha10(old) == sha10(code):
                    need_write = False
            except:
                need_write = True

        if need_write:
            with open(out_py, "w", encoding="utf-8") as f:
                f.write(code)

        rows.append({
            "group_index": gid,
            "prev_code_path": out_py if need_write or os.path.exists(out_py) else "",
            "prev_code_sha10": sha10(code),
            "missing_ops": ";".join(missing),
            "json_path": json_path,
            "n_ops": len(filtered_ops)
        })

    pd.DataFrame(rows).to_csv(OUT_INDEX_CSV, index=False)
    print(f"[DONE] 预处理完成，共生成 {sum(bool(r.get('prev_code_path')) for r in rows)} 条。Index: {OUT_INDEX_CSV}")

if __name__ == "__main__":
    main()
