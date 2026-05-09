import re
from typing import Tuple

_BOOLEAN_OPS = ("union", "cut", "intersect")

def _find_last_boolean_assignment(code: str) -> Tuple[int, str, int, str]:
    """
    查找最后一次形如:
        <lhs> = <something>.<op>(<arg...>)
    的布尔操作。
    返回: (line_idx, lhs_var, op_start_pos_global, op_name)
    若未找到，返回 (-1, "", -1, "")
    """
    lines = code.splitlines()
    src = "\n".join(lines)

    last = (-1, "", -1, "")
    # 先粗找 .op( 的最后位置
    best_pos = -1
    best_op = ""
    for op in _BOOLEAN_OPS:
        idx = src.rfind(f".{op}(")
        if idx > best_pos:
            best_pos = idx
            best_op = op
    if best_pos < 0:
        return (-1, "", -1, "")

    # 定位所属行 & LHS
    # 找该位置所在的行号
    acc = 0
    line_idx = 0
    for i, line in enumerate(lines):
        nxt = acc + len(line) + 1
        if acc <= best_pos < nxt:
            line_idx = i
            break
        acc = nxt

    line = lines[line_idx]

    # 解析 LHS（= 左边变量）
    # 允许带空格:   result_2   =   result_1.union(...)
    m_lhs = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)$", line)
    if not m_lhs:
        return (-1, "", -1, "")
    lhs = m_lhs.group(1)

    return (line_idx, lhs, best_pos, best_op)

def _extract_call_argument(src: str, lparen_idx: int) -> str:
    """
    从 src[lparen_idx] 指向 '(' 开始，提取与之匹配的最外层括号里的参数字符串。
    处理嵌套括号。
    """
    if lparen_idx < 0 or lparen_idx >= len(src) or src[lparen_idx] != "(":
        return ""
    depth = 0
    buf = []
    i = lparen_idx
    while i < len(src):
        ch = src[i]
        if ch == "(":
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
            else:
                buf.append(ch)
        else:
            if depth >= 1:
                buf.append(ch)
        i += 1
    return "".join(buf).strip()

import re
from typing import Tuple

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[1] if "```" in s[3:] else s[3:]
    if s.endswith("```"):
        s = s[: s.rfind("```")]
    return s.strip()

def _last_assigned_var(code_block: str, default: str = "shape") -> str:
    """获取代码块中最后一次赋值的变量名。"""
    guess = None
    for l in reversed(code_block.splitlines()):
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=", l)
        if m:
            guess = m.group(1)
            break
    return guess or default

# def build_iso_code(previous_code: str, generated_code: str, iso_export_path: str) -> Tuple[str, str]:
#     """
#     简化版方案1：
#     - 直接把 previous_code 原封贴在 shape 段前面；
#     - 不做净化，只要不重复导出即可；
#     - 只导出 shape（或最后赋值变量）；
#     """
#     def _strip_fences(s: str) -> str:
#         s = s.strip()
#         if s.startswith("```"):
#             s = s.split("```", 1)[1] if "```" in s[3:] else s[3:]
#         if s.endswith("```"):
#             s = s[: s.rfind("```")]
#         return s.strip()

#     def _last_assigned_var(code_block: str, default: str = "shape") -> str:
#         for l in reversed(code_block.splitlines()):
#             m = re.match(r"\s*([A-Za-z_]\w*)\s*=", l)
#             if m:
#                 return m.group(1)
#         return default

#     src = _strip_fences(generated_code)
#     lines = src.splitlines()
#     shape_idx = next((i for i, l in enumerate(lines) if re.match(r"^\s*#\s*shape\s*$", l, re.I)), -1)
#     bool_idx  = next((i for i, l in enumerate(lines) if re.match(r"^\s*#\s*bool\s*$",  l, re.I)), -1)

#     shape_block = ""
#     if 0 <= shape_idx < bool_idx:
#         shape_block = "\n".join(lines[shape_idx + 1: bool_idx]).strip()

#     lhs = _last_assigned_var(shape_block)

#     # 拼接：上一步 + 当前 shape + 导出
#     iso_code = (
#         previous_code.rstrip() + "\n\n"
#         + "# === SHAPE ===\n"
#         + shape_block + "\n\n"
#         + "# === ISO export ===\n"
#         + f"cq.exporters.export(shape, r\"{iso_export_path}\")"
#     )
#     return iso_code, lhs


def build_iso_code(previous_code: str, generated_code: str, iso_export_path: str, first_step: bool) -> Tuple[str, str]:
    """
    构建“单步（isolated）导出”代码：
    变量选择优先级：
      1) 精确命名为 'shape' 的最后一次赋值
      2) 以 'shape' 开头（如 shape0, shape_iso, shape_tmp）的最后一次赋值
      3) 若都没有，则取最后一个出现的赋值变量
    导出时自动兼容 Workplane/Shape：优先尝试 .val()，失败则直接导出对象本身。
    额外保险：
      - 若存在 #bool，保证 #bool 及其后内容绝不输出
      - 若最终块的最后一行以 result_<数字> 开头，则删除该行（不管 RHS 是什么）
    返回：(合并后的源码, 选中的 lhs 变量名)
    """
    def _strip_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # 去掉 ```python ... ``` 包裹
            first_close = s.find("```", 3)
            if first_close != -1:
                s = s[3:first_close] if "```" not in s[first_close + 3:] else s.split("```", 1)[1]
        if s.endswith("```"):
            s = s[: s.rfind("```")]
        return s.strip()

    def _collect_lhs(code_block: str):
        lhs_all = []
        for l in code_block.splitlines():
            m = re.match(r"\s*([A-Za-z_]\w*)\s*=", l)
            if m:
                lhs_all.append(m.group(1))
        return lhs_all

    def _pick_lhs_by_priority(lhs_all):
        # 1) 优先精确 'shape'
        for v in reversed(lhs_all):
            if v == "shape":
                return v
        # 2) 其次以 'shape' 开头
        for v in reversed(lhs_all):
            if v.startswith("shape"):
                return v
        # 3) 最后一个赋值变量
        return lhs_all[-1] if lhs_all else "shape"

    def _drop_trailing_result_line(block: str) -> str:
        ls = block.splitlines()
        # 去掉末尾空行
        while ls and ls[-1].strip() == "":
            ls.pop()
        # 如果最后一行以 result_<digits> 开头，删掉整行（不管后面是什么）
        if ls and re.match(r"^\s*result_\d+\b", ls[-1]):
            ls.pop()
            while ls and ls[-1].strip() == "":
                ls.pop()
        return "\n".join(ls).rstrip()

    src = _strip_fences(generated_code)
    lines = src.splitlines()

    # 找到 #bool，并“全局裁掉” #bool 及其后内容（保证不会输出）
    bool_idx = next((i for i, l in enumerate(lines) if re.match(r"^\s*#\s*bool\s*$", l, re.I)), -1)
    lines_upto_bool = lines[:bool_idx] if bool_idx >= 0 else lines

    # 在裁剪后的范围内找 #shape
    shape_idx = next((i for i, l in enumerate(lines_upto_bool) if re.match(r"^\s*#\s*shape\s*$", l, re.I)), -1)

    if shape_idx >= 0:
        # 从 #shape 后到 #bool 前（或文件末尾）作为 shape_block
        shape_block = "\n".join(lines_upto_bool[shape_idx + 1:]).strip()
    else:
        # 没有 #shape，就用 “#bool 之前” 的所有内容
        shape_block = "\n".join(lines_upto_bool).strip()

    # 最后一行如果是 result_<digits>... 则删除
    shape_block = _drop_trailing_result_line(shape_block)

    # 如果裁剪后为空，兜底仍然只取 bool 前部分（避免泄漏）
    if not shape_block.strip():
        shape_block = _drop_trailing_result_line("\n".join(lines_upto_bool).strip())

    lhs_candidates = _collect_lhs(shape_block if shape_block.strip() else "\n".join(lines_upto_bool))
    lhs = _pick_lhs_by_priority(lhs_candidates)

    head = (
        f"import cadquery as cq\n"
        f"from cadquery import Plane, Vector, Workplane\n"
        f"from functools import reduce\n"
    )

    export_tail = f"""
# === ISO export ===
def _export_any(obj, path):
    try:
        cq.exporters.export(obj.val(), path)
    except Exception:
        cq.exporters.export(obj, path)

_export_any({lhs}, r"{iso_export_path}")
""".rstrip()

    parts = []
    # if (previous_code or "").strip():
    #     parts.append(previous_code.rstrip())
    parts.append(head)
    parts.append("# === SHAPE ===\n")
    parts.append(shape_block.rstrip() if shape_block.strip() else "\n".join(lines_upto_bool).rstrip())
    parts.append(export_tail)

    merged = "\n\n".join(parts).rstrip() + "\n"
    return merged, lhs

import re
from typing import Tuple

def build_integrated_code(
    previous_code: str,
    generated_code: str,
    full_export_path: str,
    first_step: bool
) -> Tuple[str, str]:
    """
    构建“整体验证”代码：
    1. 优先导出最后一次赋值的 result；
    2. 否则导出最后一次赋值的 result_n；
    3. 否则导出最后一个出现的赋值变量；
    4. 若无任何赋值，则兜底为 'result'。
    """
    lhs_all = []
    for line in generated_code.splitlines():
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
        if m:
            lhs_all.append(m.group(1))

    lhs = None
    # 1️⃣ 优先最后一个 result
    for v in reversed(lhs_all):
        if v == "result":
            lhs = v
            break

    # 2️⃣ 否则最后一个 result_n
    if lhs is None:
        for v in reversed(lhs_all):
            if re.fullmatch(r"result_\d+", v):
                lhs = v
                break

    # 3️⃣ 否则最后一个赋值变量
    if lhs is None and lhs_all:
        lhs = lhs_all[-1]

    # 4️⃣ 兜底
    if lhs is None:
        lhs = "result"

    # 生成导出语句
    head = ""
    if first_step:
        head = (
            f"import cadquery as cq\n"
            f"from cadquery import Plane, Vector, Workplane\n"
            f"from functools import reduce\n"
        )
    export_tail = f"""
# === INTEGRATED export ===
def _export_any(obj, path):
    try:
        cq.exporters.export(obj.val(), path)
    except Exception:
        cq.exporters.export(obj, path)

_export_any({lhs}, r"{full_export_path}")
""".rstrip()
    
    parts = []
    if first_step:
        parts.append(head)
    if previous_code and previous_code.strip():
        parts.append(previous_code.rstrip())
    if generated_code and generated_code.strip():
        parts.append("# === GENERATED CODE ===\n")
        parts.append(generated_code.rstrip())
    parts.append(export_tail.rstrip())

    merged = "\n\n".join(parts) + "\n"
    return merged, lhs

if __name__ == "__main__":
    previous_code = """
"""
    generated_code = """
narrower_cylinder = cq.Workplane(custom_plane).circle(0.015).extrude(0.01)
"""
    merged, lhs = build_iso_code(previous_code, generated_code, "111",True)
    print(merged, lhs)
    pass