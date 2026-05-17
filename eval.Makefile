PYTHON := /data/baiyixue/envs/trl/bin/python
SCRIPT := /home/baiyixue/project/op-cad/reward/evaluation.py
LOCAL_INFER_SCRIPT := /home/baiyixue/project/op-cad/reward/local_highlight_infer.py

# stage2 预计算向量（在 rlcad 上经 `octo run rlcad ls ...` 确认）：
#   pred: .../stage2_code_decoder/outputs/embeddings/pred
#   gt:   .../stage2_code_decoder/outputs/embeddings/gt
# 注意：不在仓库根 `jepa-cad-stage2-train/outputs`，而在 stage2_code_decoder 下。
STAGE2_EMBED_DIR := /home/baiyixue/project/jepa-cad-stage2-train/stage2_code_decoder/outputs/embeddings/gt

# Local highlight-embedding ablation. Override these on the make command line.
LOCAL_BASE_MODEL ?= /data/baiyixue/inference_model/Qwen2.5-Coder-3B-Instruct-q2-stage2
LOCAL_LORA_ADAPTER ?=
LOCAL_PROJECTOR_CKPT ?= /home/baiyixue/project/jepa-cad-stage2-train/stage2_code_decoder/outputs/stage2_qwen25_coder_q2_gt_embed/checkpoints/latest.pt
LOCAL_MODE ?= both
LOCAL_DEVICES ?= cuda:0
LOCAL_BATCH_SIZE ?= 1
LOCAL_LIMIT ?= 20
LOCAL_NPROC ?= 1
LOCAL_MAX_NEW_TOKENS ?= 2048
LOCAL_MAX_INPUT_TOKENS ?= 32768
LOCAL_MAX_MODEL_LEN ?= 32768
LOCAL_ATTN_IMPL ?= sdpa
LOCAL_PRECISION ?= bf16
LOCAL_APPLY_CHAT_TEMPLATE ?= 1
LOCAL_PROMPTS_CSV ?= /home/baiyixue/project/flowcad/data/prompt.csv
LOCAL_PRE_CODE_DIR ?= /home/baiyixue/project/flowcad/data/pre_code
LOCAL_SPLIT_JSON ?= /home/baiyixue/project/flowcad/data/split_result_filtered.json
LOCAL_SPLIT_KEY ?= test
LOCAL_OUT_JSONL ?= /data/baiyixue/CAD/inference_result/local_highlight_ablation.jsonl

ifneq ($(strip $(LOCAL_LORA_ADAPTER)),)
LOCAL_LORA_ARG := --lora-adapter $(LOCAL_LORA_ADAPTER)
endif

ifeq ($(LOCAL_APPLY_CHAT_TEMPLATE),1)
LOCAL_CHAT_TEMPLATE_ARG := --local-apply-chat-template
endif

.PHONY: run gemini Qwen3-vl-vision Qwen2.5-3b-coder Qwen2.5-3b-coder-highlight Qwen2.5-3b-coder-highlight-full build-full-sequence-test local-highlight repair

# 用脚本默认参数
run:
	$(PYTHON) $(SCRIPT)

Qwen3-vl-vision:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code_cop \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/before_picture \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/flowcad/data/split_result.json \
		--visual-mode \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model Qwen3-VL-8B \
		--nproc 64

Qwen2.5-3b-coder:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code_cop \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/nopicture \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/flowcad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model Qwen2.5-Coder-3B-Instruct \
		--nproc 64

Qwen2.5-3b-coder-highlight:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code_cop \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/highlight_q2 \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \single
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/flowcad/data/split_result_filtered.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model Qwen2.5-Coder-3B-q2 \
		--highlight-embedding \
		--embed-dir $(STAGE2_EMBED_DIR) \
		--nproc 64

FULL_SEQUENCE_JSONL ?= /data/baiyixue/CAD/inference_result/full_sequence_test.jsonl
FULL_SEQUENCE_OUT ?= /data/baiyixue/CAD/inference_result/full_sequence_eval
FULL_SEQUENCE_LIMIT ?= 0
FULL_SEQUENCE_K ?= 1

build-full-sequence-test:
	$(PYTHON) /home/baiyixue/project/op-cad/reward/build_full_sequence_test.py \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--split-json /home/baiyixue/project/flowcad/data/split_result_filtered.json \
		--split-key test \
		--pre-code-cop-dir /home/baiyixue/project/flowcad/data/pre_code_cop \
		--out-jsonl $(FULL_SEQUENCE_JSONL) \
		--limit $(FULL_SEQUENCE_LIMIT)

Qwen2.5-3b-coder-highlight-full:
	$(PYTHON) /home/baiyixue/project/op-cad/reward/evaluation_full.py \
		--full-jsonl $(FULL_SEQUENCE_JSONL) \
		--out-root $(FULL_SEQUENCE_OUT) \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--k $(FULL_SEQUENCE_K) \
		--limit $(FULL_SEQUENCE_LIMIT) \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model Qwen2.5-Coder-3B-q3 \
		--highlight-embedding \
		--embed-dir /home/baiyixue/project/jepa-cad-stage2-train/stage2_code_decoder/outputs/embeddings/pred

local-highlight:
	@test -n "$(LOCAL_BASE_MODEL)" || (echo "Set LOCAL_BASE_MODEL=/path/to/base_model"; exit 1)
	@test -n "$(LOCAL_PROJECTOR_CKPT)" || (echo "Set LOCAL_PROJECTOR_CKPT=/path/to/projector_checkpoint.pt"; exit 1)
	$(PYTHON) $(SCRIPT) --mode std \
		--prompts-csv $(LOCAL_PROMPTS_CSV) \
		--pre-code-dir $(LOCAL_PRE_CODE_DIR) \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code_cop \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/local_highlight_eval \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--embed-dir $(STAGE2_EMBED_DIR) \
		--split-json $(LOCAL_SPLIT_JSON) \
		--split-key $(LOCAL_SPLIT_KEY) \
		--limit $(LOCAL_LIMIT) \
		--gen-mode local-highlight \
		--provider local \
		--highlight-embedding \
		--local-base-model $(LOCAL_BASE_MODEL) \
		--local-lora-adapter "$(LOCAL_LORA_ADAPTER)" \
		--local-projector-ckpt $(LOCAL_PROJECTOR_CKPT) \
		--local-devices $(LOCAL_DEVICES) \
		--local-max-new-tokens $(LOCAL_MAX_NEW_TOKENS) \
		--local-max-input-tokens $(LOCAL_MAX_INPUT_TOKENS) \
		--local-max-model-len $(LOCAL_MAX_MODEL_LEN) \
		--local-attn-impl $(LOCAL_ATTN_IMPL) \
		--local-precision $(LOCAL_PRECISION) \
		$(LOCAL_CHAT_TEMPLATE_ARG) \
		--nproc $(LOCAL_NPROC)

# ===== 修正模式 =====
repair:
	@echo "🚀 启动修正模式..."
	${PYTHON} ${SCRIPT} \
		--repair-csv /home/baiyixue/project/op-cad/inference_results/gpt-4/std/repair_list.csv \
		--mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider http \
		--http-model gpt-4 \
		--nproc 100 \
		--resume
