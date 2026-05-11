#!/usr/bin/env python3
"""Local HF inference runner for highlight-embedding ablations.

This script is intentionally independent from the vLLM/API path: highlight
embeddings are injected as soft tokens via ``inputs_embeds``.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_call.highlight_paths import resolve_highlight_embedding_path  # noqa: E402
from model_call.prompt import build_incremental_cq_prompt  # noqa: E402
from model_call.call_model import _extract_code_from_text  # noqa: E402


class HighlightProjector(nn.Module):
    """Project one highlight vector into a sequence of soft prompt tokens."""

    def __init__(self, input_dim: int, hidden_size: int, num_soft_tokens: int):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size * num_soft_tokens),
        )

    def forward(self, z_highlight: torch.Tensor) -> torch.Tensor:
        soft_tokens = self.net(z_highlight)
        return soft_tokens.view(-1, self.num_soft_tokens, self.hidden_size)


@dataclass
class GenConfig:
    max_new_tokens: int
    max_input_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


def _resolve_dtype(name: str) -> torch.dtype:
    name = (name or "bf16").lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _load_prev_code(group_index: str, base_dir: Optional[str]) -> str:
    if not base_dir or re.search(r"/step0$", str(group_index)):
        return ""

    parts = str(group_index).split("/")
    fname = "_".join(parts) + ".py"
    candidates = [
        Path(base_dir).joinpath(*parts[:-1], fname) if len(parts) > 1 else Path(base_dir) / fname,
        Path(base_dir) / fname,
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return ""


def _read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"group_index", "prompt_text"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return rows


def _load_split_ids(path: Optional[str], key: str) -> Optional[set[str]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = data.get(key, data if isinstance(data, list) else None)
    if ids is None:
        raise ValueError(f"split key {key!r} not found in {path}")
    return {str(x) for x in ids}


def _prepare_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _read_rows(args.prompts_csv)
    split_ids = _load_split_ids(args.split_json, args.split_key)
    if split_ids is not None:
        rows = [r for r in rows if str(r.get("group_index")) in split_ids]
    if args.start:
        rows = rows[args.start :]
    if args.limit:
        rows = rows[: args.limit]

    tasks: List[Dict[str, Any]] = []
    for row in rows:
        pid = str(row["group_index"])
        step_match = re.search(r"step(\d+)", pid)
        step_num = int(step_match.group(1)) if step_match else 0
        prev_code = _load_prev_code(pid, args.pre_code_dir)
        if row.get("prev_code_path") and Path(str(row["prev_code_path"])).is_file():
            prev_code = Path(str(row["prev_code_path"])).read_text(encoding="utf-8")

        baseline_prompt = build_incremental_cq_prompt(
            previous_code=prev_code,
            operation_instruction=str(row["prompt_text"]),
            current_var_name=f"result{step_num - 1} " if step_num > 0 else None,
            next_var_name="result",
            allow_comments=False,
            op_kind=str(row.get("op", "")).lower(),
            use_highlight_embedding=False,
        )
        highlight_prompt = build_incremental_cq_prompt(
            previous_code=prev_code,
            operation_instruction=str(row["prompt_text"]),
            current_var_name=f"result{step_num - 1} " if step_num > 0 else None,
            next_var_name="result",
            allow_comments=False,
            op_kind=str(row.get("op", "")).lower(),
            use_highlight_embedding=True,
        )
        emb_path = resolve_highlight_embedding_path(row, embed_dir=args.embed_dir)
        tasks.append(
            {
                "pid": pid,
                "row": row,
                "baseline_prompt": baseline_prompt,
                "highlight_prompt": highlight_prompt,
                "embedding_path": emb_path,
            }
        )
    return tasks


class LocalGenerator:
    def __init__(self, args: argparse.Namespace, device: str):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.dtype = _resolve_dtype(args.precision)

        reserved_soft_tokens = args.num_soft_tokens if args.mode in ("highlight", "both") else 0
        safe_max_input_tokens = args.max_model_len - args.max_new_tokens - reserved_soft_tokens
        if safe_max_input_tokens <= 0:
            raise ValueError(
                f"Invalid length budget: max_model_len={args.max_model_len}, "
                f"max_new_tokens={args.max_new_tokens}, "
                f"num_soft_tokens={reserved_soft_tokens}"
            )
        effective_max_input_tokens = min(args.max_input_tokens, safe_max_input_tokens)

        self.gen = GenConfig(
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=effective_max_input_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            model_max_length=effective_max_input_tokens,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.apply_chat_template = args.apply_chat_template

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device.startswith("cuda"):
            load_kwargs["device_map"] = {"": self.device}
        if args.attn_impl:
            load_kwargs["attn_implementation"] = args.attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
        if args.lora_adapter:
            self.model = PeftModel.from_pretrained(self.model, args.lora_adapter)
        if not device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

        self.projector = None
        if args.mode in ("highlight", "both"):
            hidden = self.model.get_input_embeddings().embedding_dim
            self.projector = HighlightProjector(
                args.highlight_embed_dim,
                hidden,
                args.num_soft_tokens,
            ).to(device=self.device, dtype=self.dtype)
            payload = torch.load(args.projector_checkpoint, map_location=self.device, weights_only=False)
            state_dict = payload["projector"] if isinstance(payload, dict) and "projector" in payload else payload
            self.projector.load_state_dict(state_dict)
            self.projector.eval()

    def _format_prompts(self, prompts: List[str]) -> List[str]:
        if not self.apply_chat_template:
            return prompts
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    def _decode_new_text(self, generated: torch.Tensor, prompt_len: int) -> List[str]:
        if generated.shape[1] > prompt_len:
            new_tokens = generated[:, prompt_len:]
        else:
            new_tokens = generated
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def _tokenize(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        formatted = self._format_prompts(prompts)
        tok = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.gen.max_input_tokens,
        )
        return {k: v.to(self.device) for k, v in tok.items()}

    @torch.inference_mode()
    def generate_baseline(self, prompts: List[str]) -> List[Dict[str, Any]]:
        tok = self._tokenize(prompts)
        prompt_len = int(tok["input_ids"].shape[1])
        gen_kwargs = {
            "max_new_tokens": self.gen.max_new_tokens,
            "do_sample": self.gen.do_sample,
            "top_p": self.gen.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.gen.do_sample:
            gen_kwargs["temperature"] = self.gen.temperature
        out = self.model.generate(**tok, **gen_kwargs)
        texts = self._decode_new_text(out, prompt_len)
        return [
            {
                "text": t,
                "code": _extract_code_from_text(t),
                "input_tokens": prompt_len,
                "output_tokens": int(out.shape[1] - prompt_len) if out.shape[1] > prompt_len else int(out.shape[1]),
            }
            for t in texts
        ]

    @torch.inference_mode()
    def generate_highlight(self, prompts: List[str], embedding_paths: List[str]) -> List[Dict[str, Any]]:
        if self.projector is None:
            raise RuntimeError("highlight generation requested without a loaded projector")
        tok = self._tokenize(prompts)

        zs = []
        for path in embedding_paths:
            z = np.load(path)
            zs.append(torch.tensor(z, dtype=torch.float32))
        z_t = torch.stack(zs, dim=0).to(self.device)
        soft_tokens = self.projector(z_t.to(dtype=self.dtype))
        text_embeds = self.model.get_input_embeddings()(tok["input_ids"])
        inputs_embeds = torch.cat([soft_tokens, text_embeds], dim=1)
        soft_mask = torch.ones(
            (len(prompts), soft_tokens.shape[1]),
            dtype=tok["attention_mask"].dtype,
            device=self.device,
        )
        attention_mask = torch.cat([soft_mask, tok["attention_mask"]], dim=1)
        prompt_len = int(inputs_embeds.shape[1])

        gen_kwargs = {
            "max_new_tokens": self.gen.max_new_tokens,
            "do_sample": self.gen.do_sample,
            "top_p": self.gen.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.gen.do_sample:
            gen_kwargs["temperature"] = self.gen.temperature
        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        texts = self._decode_new_text(out, prompt_len)
        return [
            {
                "text": t,
                "code": _extract_code_from_text(t),
                "input_tokens": prompt_len,
                "output_tokens": int(out.shape[1] - prompt_len) if out.shape[1] > prompt_len else int(out.shape[1]),
            }
            for t in texts
        ]


def _batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _clear_cuda_cache(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _run_with_oom_fallback(
    gen_fn,
    batch_items: List[Dict[str, Any]],
    prompts: List[str],
    embedding_paths: Optional[List[str]],
    mode: str,
    device: str,
) -> List[Dict[str, Any]]:
    try:
        if embedding_paths is None:
            return gen_fn(prompts)
        return gen_fn(prompts, embedding_paths)
    except torch.cuda.OutOfMemoryError as exc:
        _clear_cuda_cache(device)
        if len(batch_items) == 1:
            return [
                {
                    "text": "",
                    "code": "",
                    "input_tokens": None,
                    "output_tokens": None,
                    "err": f"oom_single: {type(exc).__name__}: {exc}",
                }
            ]

        results = []
        for prompt_idx, _item in enumerate(batch_items):
            try:
                if embedding_paths is None:
                    sub_result = gen_fn([prompts[prompt_idx]])[0]
                else:
                    sub_result = gen_fn([prompts[prompt_idx]], [embedding_paths[prompt_idx]])[0]
                results.append(sub_result)
            except Exception as sub_exc:
                _clear_cuda_cache(device)
                results.append(
                    {
                        "text": "",
                        "code": "",
                        "input_tokens": None,
                        "output_tokens": None,
                        "err": f"failed_single: {type(sub_exc).__name__}: {sub_exc}",
                    }
                )
        return results
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        _clear_cuda_cache(device)
        return [
            {
                "text": "",
                "code": "",
                "input_tokens": None,
                "output_tokens": None,
                "err": f"oom_runtime: {type(exc).__name__}: {exc}",
            }
            for _ in batch_items
        ]


def _worker(rank: int, device: str, tasks: List[Dict[str, Any]], args: argparse.Namespace, out_path: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    gen = LocalGenerator(args, device)
    total = len(tasks)
    done = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for batch in _batched(tasks, args.batch_size):
            if args.mode in ("baseline", "both"):
                baseline_prompts = [x["baseline_prompt"] for x in batch]
                preds = _run_with_oom_fallback(
                    gen.generate_baseline,
                    batch,
                    baseline_prompts,
                    None,
                    "baseline",
                    device,
                )
                for task, pred in zip(batch, preds):
                    err = pred.pop("err", "")
                    f.write(json.dumps(_make_record(task, "baseline", pred, err), ensure_ascii=False) + "\n")
                f.flush()

            if args.mode in ("highlight", "both"):
                valid = [x for x in batch if x["embedding_path"]]
                missing = [x for x in batch if not x["embedding_path"]]
                for task in missing:
                    f.write(
                        json.dumps(
                            _make_record(task, "highlight", None, "embedding_missing"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                for sub in _batched(valid, args.batch_size):
                    highlight_prompts = [x["highlight_prompt"] for x in sub]
                    embedding_paths = [str(x["embedding_path"]) for x in sub]
                    preds = _run_with_oom_fallback(
                        gen.generate_highlight,
                        sub,
                        highlight_prompts,
                        embedding_paths,
                        "highlight",
                        device,
                    )
                    for task, pred in zip(sub, preds):
                        err = pred.pop("err", "")
                        f.write(json.dumps(_make_record(task, "highlight", pred, err), ensure_ascii=False) + "\n")
                f.flush()

            done += len(batch)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            print(
                f"[worker {rank} {device}] {done}/{total}  {rate:.2f} samples/s  elapsed={elapsed:.1f}s",
                flush=True,
            )


def _make_record(
    task: Dict[str, Any],
    mode: str,
    pred: Optional[Dict[str, Any]],
    err: str,
) -> Dict[str, Any]:
    row = task["row"]
    pred = pred or {}
    return {
        "group_index": task["pid"],
        "mode": mode,
        "op": row.get("op", ""),
        "embedding_path": task.get("embedding_path"),
        "err": err,
        "code": pred.get("code", ""),
        "raw_text": pred.get("text", ""),
        "input_tokens": pred.get("input_tokens"),
        "output_tokens": pred.get("output_tokens"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts-csv", required=True)
    p.add_argument("--pre-code-dir", default=None)
    p.add_argument("--embed-dir", default=None)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--split-json", default=None)
    p.add_argument("--split-key", default="test")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)

    p.add_argument("--base-model", required=True)
    p.add_argument("--lora-adapter", default="")
    p.add_argument("--projector-checkpoint", default="")
    p.add_argument("--highlight-embed-dim", type=int, default=1536)
    p.add_argument("--num-soft-tokens", type=int, default=16)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--apply-chat-template", action="store_true")

    p.add_argument("--mode", choices=["baseline", "highlight", "both"], default="both")
    p.add_argument("--devices", default="cuda:0", help="Comma-separated devices, e.g. cuda:0,cuda:1 or cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-input-tokens", type=int, default=32768,
                   help="Tokenizer truncation length (also sets tokenizer.model_max_length).")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["", "sdpa", "flash_attention_2", "eager"],
                   help="Set transformers attn_implementation; pick flash_attention_2 if installed.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in ("highlight", "both") and not args.projector_checkpoint:
        raise SystemExit("--projector-checkpoint is required for highlight/both mode.")
    if args.mode in ("highlight", "both") and not args.embed_dir:
        print("[WARN] --embed-dir is empty; highlight mode will only work for rows with embedding_path.")
    tasks = _prepare_tasks(args)
    if not tasks:
        raise SystemExit("No tasks selected.")

    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if not devices:
        devices = ["cuda:0" if torch.cuda.is_available() else "cpu"]

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] tasks={len(tasks)} mode={args.mode} devices={devices} batch_size={args.batch_size}")
    print("[RUN] one model copy is loaded per device; avoid multiple workers on the same GPU.")

    if len(devices) == 1:
        _worker(0, devices[0], tasks, args, args.out_jsonl)
        return

    mp.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory(prefix="local_highlight_") as tmpdir:
        procs = []
        part_paths = []
        for rank, device in enumerate(devices):
            shard = tasks[rank:: len(devices)]
            part = str(Path(tmpdir) / f"part_{rank}.jsonl")
            part_paths.append(part)
            proc = mp.Process(target=_worker, args=(rank, device, shard, args, part))
            proc.start()
            procs.append(proc)

        failed = False
        for proc in procs:
            proc.join()
            failed = failed or proc.exitcode != 0
        if failed:
            raise SystemExit("At least one worker failed.")

        with open(args.out_jsonl, "w", encoding="utf-8") as out:
            for part in part_paths:
                if Path(part).is_file():
                    out.write(Path(part).read_text(encoding="utf-8"))
    print(f"[DONE] wrote {args.out_jsonl}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[DONE] elapsed={time.time() - t0:.1f}s")
