import re
from textwrap import dedent
from typing import List, Optional, Dict, Literal

def _detect_last_result_var(code: str) -> str:
    """从前序代码中自动检测最后一个 result_x 变量名；没有则返回 default。"""
    matches = re.findall(r"\b(result_\d+)\b\s*=", code)
    return matches[-1] if matches else None

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
    构建“增量式 CadQuery 代码生成”Prompt。
    - link_mode=None       ：不强制如何命名结果变量，仅提供温和建议
    - link_mode="inplace"  ：建议基于最近变量继续，并覆盖同名变量（非强制）
    - link_mode="append_new": 建议写入 next_var_name（非强制；未给则自动 current+1）
    """
    # ---------------- 识别操作类型 ----------------
    opk = (op_kind or "").lower()
    is_modify = any(k in opk for k in ["fillet", "chamfer"])  # chamfer / fillet / chamfer_fillet
    # ------------------------------------------------
    
    # 1) 解析“最近一步变量名”
    cur_var = current_var_name or _detect_last_result_var(previous_code)

    # 2) 计算默认 next_var（仅在 append_new 建议需要）
    auto_next = None
    if cur_var:
        m = re.match(r"(result_)(\d+)$", cur_var)
        auto_next = f"{m.group(1)}{int(m.group(2)) + 1}"
    else:
        auto_next = "result_0"
    if not next_var_name:
        next_var_name = auto_next

    # 3) 图片块
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

    # 4) 硬性要求（不含变量命名强制）
    hard_reqs = [
"Output **only** the new CadQuery code snippet (no extra text).",
"If previous code exists: do not recreate the base model or redefine previous variables unnecessarily.",
"The code must be directly executable when appended to the context.",
"Keep naming consistent with the previous steps.",
"Do not include comments or explanations.",
]
    if add_size_guidelines:
        hard_reqs += [
            "If dimensions are unspecified, choose reasonable proportions (e.g., diameter ≈ 30%–60% of local feature diameter).",
            "For cuts from a top face, use a depth that removes the intended material **without** penetrating unintended base layers.",
        ]
    if not allow_comments:
        hard_reqs.append("Do **not** include comments or explanations.")
    else:
        hard_reqs.append("Comments are allowed but keep them concise.")

    # 5) 变量衔接
    if not previous_code.strip():
        linking_note = (
        "### Linking notice\n"
        "- This is the **first modeling step**.\n"
        "- After building `shape` (if applicable), set `result = shape` in the **#bool** section."
    )
    else:
        linking_note = dedent(f"""
        ### Linking Suggestion
        - Treat **{cur_var}** as the current solid to operate on.
        - Assigning the new result to **{next_var_name}** (e.g., `{next_var_name} = ...`) to keep a clear history.
        """).strip()

    hard_reqs_block = "\n".join([f"{i+1}. {r}" for i, r in enumerate(hard_reqs)])
    
    shape_bool_rules = dedent(f"""
    ### Shape-then-Bool Rules (ENFORCED)
- In **#shape**: Build the required feature(s) as **independent solid(s)** without referencing **{cur_var if cur_var else 'previous results'}**; if multiple bodies are created, **union** them into a single solid and assign to **shape**.
    - In **#bool**: Apply **only** one of **union** or **cut** between **{cur_var}** (the current solid) and **shape**.
         - The result assignment **must** follow one of:
           - `{next_var_name} = {cur_var}.union(shape)`
           - `{next_var_name} = {cur_var}.cut(shape)`
           - `result = shape (This form is ONLY allowed if this is the first step)`
    """).strip()

    # plane_usage_rules = dedent("""
    # ### Workplane and Face Selection Rules (MANDATORY)
    # - **NEVER** use `.faces()` or `.face()` to select faces or workplanes.
    # - **NEVER** use string shortcuts like `"XY"`, `"XZ"`, or `"YZ"` to define workplanes.
    # - Always construct workplanes **explicitly** using `Plane` and `Vector`.
    # Example:
    # ```python
    # from cadquery import Plane, Vector
    # normal_vector = Vector(0.0, 0.0, 1.0)
    # x_dir = Vector(1.0, 0.0, 0.0)
    # origin = Vector(0.0, 0.0, 0.0)
    # custom_plane = Plane(origin=origin, normal=normal_vector, xDir=x_dir)
    # wp = cq.Workplane(custom_plane)
    # ```
    # - Any generated code containing .faces(, .face(, or "XY"/"XZ"/"YZ" strings will be rejected.
    # """)
    
    plane_usage_rules = dedent("""
    ### Workplane and Face Selection Rules (MANDATORY)
    - **Before modeling operation, you must explicitly define a new workplane to ensure geometric consistency.**
    - **NEVER** use `.faces()` or `.face()` to select faces or workplanes.
    - Always construct workplanes **explicitly** using `Plane` and `Vector`.
    Example:
    ```python
    from cadquery import Plane, Vector
    wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    """)


    modify_rules = dedent("""
    ### Edge-Selection and Application Rules (MANDATORY for fillet/chamfer)
    -Before applying any fillet or chamfer, you MUST reset the workplane coordinate system exactly as follows (no modifications allowed):
        wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    - **NEVER** chain `.fillet()` or `.chamfer()` directly after an edge selector (e.g., `.edges(...).fillet(...)`).
    - You **MUST** split the operation into two distinct steps using sequential variable names (e.g., `edges_1`, `edges_2`...).

    - **Step 1: Select Edges.**
      Select edges from the *current* result (e.g., `result_0`) and assign the **selection workplane** to a new variable (e.g., `edges_1`).
    - **Step 2: Apply Operation.**
      Create the *next* result (e.g., `result_1`) by calling `.fillet()` or `.chamfer()` on the **selected edges** (e.g., `edges_0`).

    Example:
    ```python
    wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    # --- Fillet Operation (Following modify_rules) ---
    # Step 1: Select edges from result_0 and assign to edges_1
    edges_1 = result_0.edges(cq.NearestToPointSelector((x,y,z)))
    
    # Step 2: Apply operation on edges_1, to create result_1
    result_1 = edges_1.fillet(fillet_radius)

    # --- Chamfer Operation (Continuing the sequence) ---
    # Step 1: Select edges from result_1 and assign to edges_2
    edges_2 = result_1.edges(cq.NearestToPointSelector((x,y,z)))
    
    # Step 2: Apply operation on edges_2, to create result_2
    result_2 = edges_2.chamfer(chamfer_distance_1, chamfer_distance_2)
    ```
    """)


# ---- 组装 Prompt ----
    sections = []
    sections.append("### Role\nYou are an expert CAD modeling assistant specialized in CadQuery.\nGenerate ONLY the incremental CadQuery code needed to perform the requested operation, as a continuation of the provided previous code context.")
    if few_shots:
        ex_blocks = []
        for i, ex in enumerate(few_shots, 1):
            label = ex.get("label", f"example_{i}")
            ex_prev = ex.get("prev_code", "").strip()
            ex_instr = ex.get("instruction", "").strip()
            ex_ans = (ex.get("answer", "") or "").strip()

            block = [f"#### Example(Do not copy numbers/variable names from examples)"]
            if ex_prev:
                block.append("**Previous code**")
                block.append("```python\n" + ex_prev + "\n```")
            if ex_instr:
                block.append("**Instruction**\n" + ex_instr)
            if ex_ans:
                block.append("**Output**")
                block.append("```python\n" + ex_ans + "\n```")
            ex_blocks.append("\n".join(block))
        sections.append("\n\n".join(ex_blocks))
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
    if not is_modify:
        sections.append(plane_usage_rules)  
    sections.append("### Hard Requirements\n" + hard_reqs_block)

    if is_modify:
        sections.append(modify_rules)
        sections.append(dedent(f"""
### Output Format (STRICT)
```python
#edges select

{{generated_edges_select_code}}
#operation
{{generated_operation_code}}
                               
#edges select
{{generated_edges_select_code}}
#operation
{{generated_operation_code}}                        
...
""").strip())
    else:
        sections.append(shape_bool_rules)
        sections.append(dedent(f"""
### Output Format (STRICT)
```python
#shape
{{generated_shape_code}}
#bool
{{generated_bool_code}}
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