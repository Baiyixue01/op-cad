import re
from typing import Optional, Dict, Tuple

# —— 面方向到“基准平面 + 偏移常量”的映射（用于去依赖 faces(">X")）——
FACE_MAP = {
    ">X": ("YZ", "XMAX"), "<X": ("YZ", "XMIN"),
    ">Y": ("XZ", "YMAX"), "<Y": ("XZ", "YMIN"),
    ">Z": ("XY", "ZMAX"), "<Z": ("XY", "ZMIN"),
}

# —— 通用数字正则 —— #
NUM = r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?'


# =========================
#        小工具函数
# =========================
def _to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _strip_booleans(src: str) -> str:
    """清除所有显式布尔调用（只留几何链）。"""
    # 全局清除所有 .cut/.union/.intersect/.combine(...)
    return re.sub(r'\.(cut|union|intersect|combine)\s*\([^)]*\)', '', src, flags=re.I)


def _std_lhs_to_model(src: str, model_var: str) -> str:
    """左值标准化：把任意 LHS 去掉并包成 model_result_k = (...)."""
    s = src.strip()
    s = re.sub(r'^\s*\w+\s*=\s*', '', s)  # 先去 LHS
    if s.startswith("(") and s.endswith(")"):
        return f"{model_var} = {s}"
    return f"{model_var} = (\n{s}\n)"


def _replace_result_prefix_with_base(src: str, force_base: str) -> str:
    """
    把 result_i.*.workplane(...)、faces("..").workplane(...) 之类的“依赖起点”
    统一替换为 cq.Workplane("<base>")。内部先去 LHS，匹配才不会失手。
    """
    s = re.sub(r'^\s*\w+\s*=\s*', '', src.strip())  # 去掉 LHS
    # 情形1：result_i.<任意链>.workplane(...)
    s = re.sub(
        r'^\s*result_\d+(?:\.[A-Za-z_]\w*\([^()]*\))*\.workplane\([^()]*\)',
        f'cq.Workplane("{force_base}")',
        s
    )
    # 情形2：开头就是 faces("..").workplane(...)
    s = re.sub(
        r'^\s*faces\(\s*"(>|<)[XYZ]"\s*\)\.workplane\([^()]*\)',
        f'cq.Workplane("{force_base}")',
        s
    )
    return s


def _faces_to_offset_workplane(src: str, bbox: Optional[Dict[str, float]], meta: Dict) -> str:
    """
    尝试将 faces("±Axis")[.workplane()] 替换为 Workplane(BASE).workplane(offset=CONST)。
    若无法替换，保留原写法，但 meta 标注 used_faces/partial。
    会替换所有出现位置。
    """
    # both: faces("..").workplane() 和裸 faces("..")
    pattern = re.compile(r'faces\(\s*"(>|<)[XYZ]"\s*\)(?:\.workplane\(\s*\))?', flags=re.I)

    def _repl(m: re.Match) -> str:
        meta["used_faces"] = True
        sign = m.group(1) + m.group(0)[7].upper()  # 组装如 >X
        # 更稳：直接从匹配文本提取符号和轴
        txt = m.group(0)
        g = re.search(r'"(>|<)([XYZ])"', txt, re.I)
        if not g:
            meta["de_dependency"] = "partial"
            return m.group(0)
        sign_axis = g.group(1).upper() + g.group(2).upper()  # e.g. >X
        if bbox and sign_axis in FACE_MAP:
            base, off = FACE_MAP[sign_axis]
            meta["de_dependency"] = "done"
            return f'cq.Workplane("{base}").workplane(offset={off})'
        else:
            meta["de_dependency"] = "partial"
            return m.group(0)

    return pattern.sub(_repl, src)


# =========================
#    孔类统一：circle + extrude（循环替换）
# =========================
def _replace_hole_family_with_circle_extrude(
    src: str, default_depth: float, meta: Dict
) -> str:
    """
    将 hole/cboreHole/cskHole 统一为 circle(d/2).extrude(depth)，只保留几何。
    解析不到 depth 时用 default_depth。循环直到不再命中。
    """
    changed = True
    s = src
    while changed:
        changed = False
        # hole(d[, depth])
        m = re.search(rf'\.hole\s*\(\s*({NUM})(?:\s*,\s*({NUM}))?', s, re.I)
        if m:
            d = _to_float(m.group(1), 0.0)
            depth = _to_float(m.group(2), default_depth) if m.group(2) else default_depth
            meta.update({"hybrid": "hole", "depth_used": depth})
            s = re.sub(r'\.hole\s*\([^\)]*\)', f'.circle({d/2}).extrude({depth})', s, count=1)
            changed = True
            continue

        # cboreHole(d, Dcbore, depth, ...)
        m = re.search(rf'\.cboreHole\s*\(\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})', s, re.I)
        if m:
            d = _to_float(m.group(1), 0.0)
            depth = _to_float(m.group(3), default_depth)
            meta.update({"hybrid": "cbore", "depth_used": depth})
            s = re.sub(r'\.cboreHole\s*\([^\)]*\)', f'.circle({d/2}).extrude({depth})', s, count=1)
            changed = True
            continue

        # cskHole(d, Dcsk, ..., depth?) —— 取第1个为 d，最后一个数当 depth（缺省用默认）
        m = re.search(r'\.cskHole\s*\(\s*([^\)]*)\)', s, re.I)
        if m:
            nums = re.findall(NUM, m.group(1))
            if nums:
                d = _to_float(nums[0], 0.0)
                depth = _to_float(nums[-1], default_depth) if len(nums) > 1 else default_depth
                meta.update({"hybrid": "csk", "depth_used": depth})
                s = re.sub(r'\.cskHole\s*\([^\)]*\)', f'.circle({d/2}).extrude({depth})', s, count=1)
                changed = True
                continue
    return s


# =========================
#        cutBlind / cutThruAll （循环替换）
# =========================
def _replace_cut_hybrids_with_extrude(
    src: str, bbox: Optional[Dict[str, float]], default_depth: float, thruall_factor: float, meta: Dict
) -> str:
    s = src
    # cutBlind(until) → extrude(|until|)
    while True:
        m = re.search(rf'\.cutBlind\s*\(\s*({NUM})\s*\)', s, re.I)
        if not m:
            break
        depth = abs(_to_float(m.group(1), default_depth))
        meta.update({"hybrid": "cutBlind", "depth_used": depth})
        s = re.sub(rf'\.cutBlind\s*\(\s*{NUM}\s*\)', f'.extrude({depth})', s, count=1)

    # cutThruAll(...) → extrude(THICK≈ bbox厚度最小边 * 系数)
    while True:
        if not re.search(r'\.cutThruAll\s*\(', s, re.I):
            break
        thick = default_depth
        if bbox:
            spans = []
            for lo, hi in (("XMIN", "XMAX"), ("YMIN", "YMAX"), ("ZMIN", "ZMAX")):
                if lo in bbox and hi in bbox:
                    spans.append(abs(bbox[hi] - bbox[lo]))
            if spans:
                thick = max(default_depth, min(spans) * thruall_factor)
        meta.update({"hybrid": "cutThruAll", "depth_used": thick, "thruall": True})
        s = re.sub(r'\.cutThruAll\s*\([^\)]*\)', f'.extrude({thick})', s, count=1)

    return s


# =========================
#        核心函数
# =========================
def decouple_step_to_geom_only(
    code: str,
    bbox: Optional[Dict[str, float]],
    k: int,
    force_base: str = "XY",
    default_depth: float = 2e-3,
    thruall_factor: float = 1.5,
) -> Tuple[str, Dict]:
    """
    只返回“独立几何体”：
        model_result_k = <纯几何建模链>
    - 删除/忽略所有布尔（cut/union/intersect/combine）（全局）
    - cutBlind/cutThruAll/hole/cboreHole/cskHole → 统一转为 circle + extrude(...) 或 extrude(THICK)（循环替换）
    - faces(">X")、result_i.workplane(...) 等 → 强制替换为 cq.Workplane("<force_base>") 或 Workplane(BASE).workplane(offset=CONST)
    返回 (geom_code, meta)
    """
    meta = {
        "used_faces": False,
        "de_dependency": "none",
        "hybrid": None,          # 最近一次识别到的 hybrid 类型：hole/cbore/csk/cutBlind/cutThruAll
        "depth_used": None,
        "thruall": False,
    }

    model_var = f"model_result_{k}"
    src = code.strip()

    # (0) 先把明显的 result_i.*.workplane(...) / faces(..).workplane(...) 前缀砍成基准面
    src = _replace_result_prefix_with_base(src, force_base=force_base)

    # (1) faces 去依赖（尽量去：所有出现位置）
    src = _faces_to_offset_workplane(src, bbox, meta)

    # (2) Hybrid → 纯几何（循环）
    src = _replace_cut_hybrids_with_extrude(src, bbox, default_depth, thruall_factor, meta)
    src = _replace_hole_family_with_circle_extrude(src, default_depth, meta)

    # (3) 删除所有显式布尔（我们只要几何；全局替换）
    src = _strip_booleans(src)

    # (4) 左值标准化为 model_result_k（并包裹为严格赋值）
    src = re.sub(r'^\s*result_\d+\.\s*(?=cq\.Workplane\(")', '', src)
    src = _std_lhs_to_model(src, model_var)

    # (5) 双保险：几何块中不应再出现布尔（再扫一次）
    src = _strip_booleans(src)

    return src.strip(), meta


# =========================
#         示例
# =========================
if __name__ == "__main__":
    bbox = {"XMAX": 0.040, "XMIN": 0.000, "YMAX": 0.020, "YMIN": 0.000, "ZMAX": 0.010, "ZMIN": 0.000}

    raw = 'result_1 = (result_0.faces(">X").workplane().polarArray(radius=0.012, startAngle=0, angle=360, count=3).circle(0.003).cutBlind(-0.001))'
    geom, meta = decouple_step_to_geom_only(raw, bbox=bbox, k=1, force_base="XY")

    print("=== GEOMETRY ONLY ===\n", geom)
    print("=== META ===\n", meta)
