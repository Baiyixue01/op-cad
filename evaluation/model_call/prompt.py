import re
from textwrap import dedent
from typing import List, Optional, Dict, Literal

def _detect_last_result_var(code: str) -> str:
    """从前序代码中自动检测最后一个 result_x 变量名；没有则返回 default。"""
    matches = re.findall(r"\b(result_\d+)\b\s*=", code)
    return matches[-1] if matches else None

from textwrap import dedent
import re
from typing import Dict, List, Optional, Literal

def build_incremental_cq_prompt(
    previous_code: str,
    operation_instruction: str,
    link_mode: Optional[Literal["inplace", "append_new"]] = None,
    current_var_name: Optional[str] = None,
    next_var_name: Optional[str] = None,
    images: Optional[List[Dict]] = None,
    image_prompt: Optional[str] = None,
    allow_comments: bool = False,
    add_size_guidelines: bool = True,
    op_kind: Optional[str] = None,
    few_shots: Optional[List[Dict]] = None,
) -> str:
    """
    构建“增量式 CadQuery 代码生成”Prompt（风格对齐精简版），功能保持不变：
    - 自动识别 fillet/chamfer 类 modify 操作与普通 shape-then-bool 操作
    - 自动推断当前变量名与 next 变量名（用于“温和建议”）
    - 支持 images / image_prompt / few_shots / 尺寸建议 / 注释开关
    """
    # ---------------- 识别操作类型 ----------------
    opk = (op_kind or "").lower()
    is_modify = any(k in opk for k in ["fillet", "chamfer"])  # chamfer / fillet / chamfer_fillet
    # ------------------------------------------------

    # 1) 解析“最近一步变量名”
    cur_var = current_var_name or _detect_last_result_var(previous_code)

    # 2) 计算默认 next_var（用于提示；不强制）
    if cur_var:
        m = re.match(r"(result_)(\d+)$", cur_var)
        auto_next = f"{m.group(1)}{int(m.group(2)) + 1}" if m else "result_0"
    else:
        auto_next = "result_0"
    if not next_var_name:
        next_var_name = auto_next

    # 3) 图片块（可选）
    image_block = ""
    if images:
        lines = []
        for idx, im in enumerate(images, 1):
            src = im.get("url") or im.get("path")
            if not src:
                continue
            cap = im.get("caption", "")
            lines.append(f"{idx}. {src}" + (f" — {cap}" if cap else ""))
        if lines:
            image_block = "\n### Images (optional)\n" + "\n".join(lines) + "\n"

    img_prompt_block = f"\n### Image Guidance\n{image_prompt}\n" if image_prompt else ""

    # 4) Hard Requirements（风格对齐：短、硬、可执行）
    hard_reqs = [
        "Output **only** the new CadQuery code snippet (no extra text).",
        "If previous code exists: do not recreate the base model or redefine previous variables unnecessarily.",
        "The code must be directly executable when appended to the context.",
        "Keep naming consistent with the previous steps.",
    ]
    if allow_comments:
        hard_reqs.append("Comments are allowed but must be concise and purely code-adjacent.")
    else:
        hard_reqs.append("Do not include comments or explanations.")

    if add_size_guidelines:
        hard_reqs += [
            "If dimensions are unspecified, choose reasonable proportions (e.g., diameter ≈ 30%–60% of local feature diameter).",
            "For cuts from a top face, choose a depth that removes the intended material **without** penetrating unintended base layers.",
        ]

    hard_reqs_block = "\n".join([f"{i+1}. {r}" for i, r in enumerate(hard_reqs)])

    # 5) Linking Suggestion（不强制，保持原功能：给出温和建议）
    if not previous_code.strip():
        linking_note = dedent("""
        ### Linking Notice
        - This is the **first modeling step**.
        - Follow the required output format. If a boolean result is not applicable, assign `result = shape` in **#bool**.
        """).strip()
    else:
        # link_mode 仅用于“建议措辞”，不改变约束逻辑（保持原功能）
        if link_mode == "inplace" and cur_var:
            assign_hint = f"- Prefer updating **{cur_var}** in-place (suggestion only)."
        else:
            assign_hint = f"- Prefer assigning the new result to **{next_var_name}** (suggestion only)."

        linking_note = dedent(f"""
        ### Linking Suggestion
        - Treat **{cur_var or 'the latest result variable'}** as the current solid.
        {assign_hint}
        """).strip()

    # 6) Rules blocks（风格对齐：条款化 + 强制输出格式）
    plane_usage_rules = dedent("""
    ### Workplane and Face Selection Rules (MANDATORY)
    - **Before modeling operation, you must explicitly define a new workplane to ensure geometric consistency.**
    - **NEVER** use `.faces()` or `.face()` to select faces or workplanes.
    - Always construct workplanes **explicitly** using `Plane` and `Vector`.

    Example:
    ```python
    from cadquery import Plane, Vector
    wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    ```
    """).strip()

    # 非 modify：Shape-then-Bool（保持原功能：引用 cur_var/next_var 的 union/cut 形式）
    shape_bool_rules = dedent(f"""
    ### Shape-then-Bool Rules (ENFORCED)
    - You MUST output exactly two sections in this exact order: **#shape** then **#bool**.
    - Under **#shape**: build required feature(s) as **independent solid(s)** without referencing **{cur_var if cur_var else 'previous results'}**.
      - If multiple bodies are created (e.g., shape_1, shape_2, ...), you MUST union them into a single solid and assign to shape:
        ```python
        shape = shape_1.union(shape_2)...union(shape_n)
        ```
    - Under **#bool**: include exactly **one** boolean statement combining the current solid and `shape`:
      - `result = result_n.union(shape)` OR `result = result_n.cut(shape)`
      - `result = shape` is ONLY allowed when this is the first step (no previous code).
    """).strip()

    # modify：fillet/chamfer（保持原功能：禁止 edges(...).fillet(...) 链式；要求两步变量）
    modify_rules = dedent("""
    ### Edge-Selection and Application Rules (MANDATORY for fillet/chamfer)
    - Before applying any fillet or chamfer, you MUST reset the workplane coordinate system exactly as follows (no modifications allowed):
        wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    - You **MUST** split the operation into two distinct steps using sequential variable names (e.g., `edges_1`, `edges_2`...).
        - **Step 1: Select Edges.**
        Select edges from the *current* result (e.g., `result_0`) and assign the **selection workplane** to a new variable (e.g., `edges_1`).
        - **Step 2: Apply Operation.**
        Call `.fillet()` or `.chamfer()` on the **selected edges** (e.g., `edges_1`) and assign the **modified shape** to another new variable (e.g., `shape_1`).
    - Finally, if there are multiple modified shapes, combine all modified shapes using intersection to form the final result, if there is only a single modified shape, assign it directly to `result`.
    Example:
    ```python
    wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    edges_1 = result_0.edges(cq.NearestToPointSelector((x,y,z)))
    shape_1 = edges_1.fillet(fillet_radius)
    edges_2 = result_1.edges(cq.NearestToPointSelector((x,y,z)))
    shape_2 = edges_2.chamfer(chamfer_distance_1, chamfer_distance_2)
    result = reduce(lambda a, b: a.intersect(b), [shape_1, shape_2])
    ```
    """).strip()

    # 7) Few-shots（保持原功能：可选示例，但整体风格与主体一致）
    few_shots_block = ""
    if few_shots:
        ex_blocks = []
        for i, ex in enumerate(few_shots, 1):
            ex_prev = (ex.get("prev_code") or "").strip()
            ex_instr = (ex.get("instruction") or "").strip()
            ex_ans = (ex.get("answer") or "").strip()
            block = ["### Example (Do not copy variable names/numbers)"]
            if ex_prev:
                block.append("**Context**")
                block.append("```python\n" + ex_prev + "\n```")
            if ex_instr:
                block.append("**Instruction**\n" + ex_instr)
            if ex_ans:
                block.append("**Output**")
                block.append("```python\n" + ex_ans + "\n```")
            ex_blocks.append("\n".join(block))
        few_shots_block = "\n\n".join(ex_blocks)

    # ---- 组装 Prompt（风格对齐：Role / Context / Instruction / Rules / Output Format） ----
    sections = []
    sections.append(
        "### Role\n"
        "You are an expert CAD modeling assistant specialized in CadQuery.\n"
        "Generate ONLY the incremental CadQuery code needed to perform the requested operation, "
        "as a continuation of the provided previous code context."
    )

    if few_shots_block:
        sections.append(few_shots_block)

    sections.append(f"""### Context (already executed Python code)
```python
{previous_code if previous_code.strip() else '# No previous code — this is the first modeling step.'}
```""")

    sections.append(f"""### Instruction
Perform the following operation **as a continuation** of the existing model:
> {operation_instruction}
""")

    if image_block:
        sections.append(image_block)
    if img_prompt_block:
        sections.append(img_prompt_block)

    sections.append(linking_note)

    # plane rules：保持原逻辑：非 modify 才追加（fillet/chamfer 自己的规则里已包含 wp ）
    if not is_modify:
        sections.append(plane_usage_rules)

    sections.append("### Hard Requirements\n" + hard_reqs_block)

    if is_modify:
        sections.append(modify_rules)
        sections.append(dedent("""
        ### Output Format (STRICT)
        ```python
        wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
        edges_1 = result_m.edges({edge_selection})
        result_n = edges_1.{operation}(...)
        edges_2 = result_n.edges({edge_selection})
        result_{n+1} = edges_2.{operation}(...)
        ...
        ```
        """).strip())
    else:
        sections.append(shape_bool_rules)
        sections.append(dedent("""
        ### Output Format (STRICT)
        ```python
        #shape
        {generated_shape_code}
        #bool
        {generated_bool_code}
        ```
        """).strip())

    return dedent("\n\n".join(sections)).strip()





def build_incremental_cq_prompt_infer(
    previous_code: str,
    operation_instruction: str,
    link_mode: Optional[Literal["inplace", "append_new"]] = None,
    current_var_name: Optional[str] = None,
    next_var_name: Optional[str] = None,
    images: Optional[List[Dict]] = None,
    image_prompt: Optional[str] = None,
    allow_comments: bool = False,
    add_size_guidelines: bool = True,
    op_kind: Optional[str] = None,
    few_shots: Optional[List[Dict]] = None,
) -> str:
    """
    构建用于推理阶段（inference）的 Alpaca 风格 Prompt。
    - 简化结构，去掉硬性规则与多层约束，保持和 SFT 训练一致。
    - 模型只看到 instruction + input，不出现 Output 或元规则。
    """
    import re
    from textwrap import dedent

    # 检测当前 result 变量名
    cur_var = current_var_name or re.findall(r"(result_\d+)", previous_code or "")[-1] if previous_code else None
    next_var_name = next_var_name or (
        f"result_{int(cur_var.split('_')[-1]) + 1}" if cur_var and cur_var.startswith("result_") else "result_0"
    )

    # few-shot 样例（若提供）
    few_shot_block = ""
    if few_shots:
        blocks = []
        for ex in few_shots:
            ex_prev = ex.get("prev_code", "").strip()
            ex_instr = ex.get("instruction", "").strip()
            ex_ans = (ex.get("answer", "") or "").strip()
            block = dedent(f"""
            ### Example
            Instruction:
            {ex_instr}

            Input:
            ```python
            {ex_prev}
            ```
            
            Output:
            ```python
            {ex_ans}
            ```
            """)
            blocks.append(block.strip())
        few_shot_block = "\n\n".join(blocks) + "\n\n"

    # 拼装核心部分（Alpaca结构）
    prompt = dedent(f"""
    {few_shot_block}Below is an instruction that describes a CAD modeling operation.
    Given the previous CadQuery code, generate the next incremental modeling step.

    ### Instruction:
    {operation_instruction}

    ### Input:
    ```python
    {previous_code if previous_code.strip() else '# No previous code — start from scratch.'}
    ```

    ### Output:
    """).strip()

    return prompt