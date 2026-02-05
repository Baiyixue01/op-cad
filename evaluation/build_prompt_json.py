#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import pandas as pd

# ====== 引用你现有代码中的函数 ======
# 确保能 import 到下列函数；若同文件，直接粘贴函数体亦可
from model_call.prompt import build_incremental_cq_prompt
from evaluation import _load_prev_code_from_dir
def main():
    ap = argparse.ArgumentParser(description="构建 SiliconFlow 批量推理 JSONL（从 prompts.csv ）")
    ap.add_argument("--prompts-csv", default = "/home/baiyixue/project/op-cad/data/prompt.csv", help="至少包含 group_index,prompt_text,op")
    ap.add_argument("--out-jsonl", default="/home/baiyixue/project/op-cad/data/input.json", help="输出 JSONL 路径")
    ap.add_argument("--mode", choices=["std","cop"], default="std", help="选择前序代码目录")
    ap.add_argument("--pre-code-dir", default="./data/pre_code")
    ap.add_argument("--cop-pre-code-dir", default="./data/pre_code_cop")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V3",
                    help="deepseek-ai/DeepSeek-V3 | deepseek-ai/DeepSeek-R1 | Qwen/QwQ-32B | ...")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--stream", action="store_true", default=True)
    ap.add_argument("--k-per-sample", type=int, default=1, help="每个样本生成几条请求（一般=1），>1 时 custom_id 加 -k{i}")
    ap.add_argument("--system", default="You are an expert CadQuery assistant. Return only executable CadQuery code.",
                    help="system 提示词")
    ap.add_argument("--append-custom-suffix", default="", help="给 custom_id 追加后缀标记（可选）")
    args = ap.parse_args()

    df = pd.read_csv(args.prompts_csv)
    df.columns = [c.lower() for c in df.columns]
    need = {"group_index","prompt_text","op"}
    if not need.issubset(df.columns):
        raise KeyError(f"prompts.csv 需要列：{need}")

    code_dir = args.cop_pre_code_dir if args.mode == "cop" else args.pre_code_dir
    rows = []

    for _, r in df.iterrows():
        pid = str(r["group_index"]).strip()
        op_kind = str(r["op"]).strip()
        instr = str(r["prompt_text"])

        # 前序代码（允许为空，内部会处理 step0）
        prev_code = _load_prev_code_from_dir(pid, code_dir)

        # 拼装 user 内容（你的严格规范已包含在 prompt 构造里）
        user_prompt = build_incremental_cq_prompt(
            previous_code=prev_code,
            operation_instruction=instr,
            op_kind=op_kind,
            link_mode=None,
            allow_comments=False,
            add_size_guidelines=True,
        )

        # K>1 时复制多条（通常评测只要一条即可）
        for k in range(args.k_per_sample):
            custom_id = f"{pid}-k{k}" if args.k_per_sample > 1 else pid
            if args.append_custom_suffix:
                custom_id = f"{custom_id}-{args.append_custom_suffix}"

            body = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": args.system},
                    {"role": "user",   "content": user_prompt}
                ],
                "stream": bool(args.stream),
                "max_tokens": args.max_tokens
            }
            # DeepSeek-R1 建议带 thinking_budget
            if "DeepSeek-R1" in args.model or args.model.endswith("/DeepSeek-R1"):
                body["thinking_budget"] = 32768

            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            rows.append(line)

    # 写 JSONL
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] 写出 {len(rows)} 行 → {args.out_jsonl}")

if __name__ == "__main__":
    main()
