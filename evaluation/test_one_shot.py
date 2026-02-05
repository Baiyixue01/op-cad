# --- imports（替换你现有的 * 导入） ---
import argparse, pandas as pd

# 1) 用“模块别名”方式导入 evaluation（关键）
import evaluation.evaluation as ev

# 2) 只从 prompt.py 拿到需要的公共函数
from evaluation.model_call.prompt import build_incremental_cq_prompt


def _load_prompts_map_from(path: str):
    """不复制逻辑，直接复用 ev 模块中的实现 & 全局变量"""
    ev.PROMPTS_CSV = path
    return ev._load_prompts_map()    # 注意：明确经 ev 调用


def preview_oneshot_prompt(
    pid: str,
    oneshot_csv: str,
    meta_csv: str,
    bool_csv: str,
    prompts_csv: str,
    pre_code_dir: str,
    cop_pre_code_dir: str,
    mode: str = "cop",
    next_var_name: str = "result",
):
    # ===== 把“全局配置”写回 ev 模块命名空间（关键）=====
    ev.ONESHOT_ON   = True
    ev.ONESHOT_CSV  = oneshot_csv
    ev.META_CSV     = meta_csv
    ev.BOOL_CSV     = bool_csv
    ev.PRE_CODE_DIR = pre_code_dir
    ev.COP_PRE_CODE_DIR = cop_pre_code_dir

    cop_mode = (mode.lower() == "cop")

    # 载入 prompts
    pmap = _load_prompts_map_from(prompts_csv)
    if pid not in pmap:
        raise KeyError(f"{pid} 不在 {prompts_csv} 的 group_index 列中")

    op_kind = str(pmap[pid].get("op", "")).lower()
    instruction = pmap[pid]["prompt_text"]

    # 前序代码（从 ev 模块取函数）
    prev_code = ev._load_prev_code_from_dir(pid, ev.COP_PRE_CODE_DIR if cop_mode else ev.PRE_CODE_DIR)

    # 构建 one-shot 示例（从 ev 模块取函数）
    few = ev._build_few_shot_for_pid(pid, pmap, cop_mode)
    few_list = [few] if few else None

    # 组装最终 Prompt
    prompt = build_incremental_cq_prompt(
        previous_code=prev_code,
        operation_instruction=instruction,
        link_mode=None,
        next_var_name=next_var_name,
        allow_comments=False,
        add_size_guidelines=True,
        op_kind=op_kind,
        few_shots=few_list,
    )

    # 打印
    print("\n================= One-shot 预览 =================")
    print(f"[PID]     {pid}")
    print(f"[Mode]    {'COP' if cop_mode else 'STD'}")
    print(f"[Op kind] {op_kind}")
    print(f"[Oneshot] {oneshot_csv}")
    if few:
        print(f"[Picked]  {few.get('label')}")
        try:
            df = pd.read_csv(oneshot_csv)
            eg = df[df["picked_as"] == few.get("label")].iloc[0]
            print(f"[Example PID] {eg['group_index']}")
        except Exception:
            pass
    else:
        print("[Oneshot] 未找到匹配 example，Prompt 将不含 few-shot")

    print("=============== FINAL PROMPT BEGIN ===============")
    print(prompt)
    print("=============== FINAL PROMPT  END  ===============\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("preview one-shot prompt")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--oneshot", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--bool", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--pre-code-dir", required=True)
    ap.add_argument("--cop-pre-code-dir", required=True)
    ap.add_argument("--mode", choices=["std","cop"], default="cop")
    ap.add_argument("--next-var", default="result")
    args = ap.parse_args()

    preview_oneshot_prompt(
        pid=args.pid,
        oneshot_csv=args.oneshot,
        meta_csv=args.meta,
        bool_csv=args.bool,
        prompts_csv=args.prompts,
        pre_code_dir=args.pre_code_dir,
        cop_pre_code_dir=args.cop_pre_code_dir,
        mode=args.mode,
        next_var_name=args.next_var,
    )
