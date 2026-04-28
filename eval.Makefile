PYTHON := /data/baiyixue/envs/trl/bin/python
SCRIPT := /home/baiyixue/project/op-cad/reward/evaluation.py

.PHONY: run gemini

# 用脚本默认参数
run:
	$(PYTHON) $(SCRIPT)

Llama-3.1-8B_sft_noco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/noco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b_sft \
		--nproc 32

Llama-3.1-8B_sft_pco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_comments \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_comments \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/pco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b_sft \
		--nproc 16

Llama-3.1-8B_sft_fco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/fco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b_sft \
		--nproc 16

Qwen3-8B-coop-sft-full-260326-noco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/noco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen3-8B-coop-sft-full-260326 \
		--nproc 32

Qwen3-8B-coop-sft-full-260326-fco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/fco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen3-8B-coop-sft-full-260326 \
		--nproc 16

Qwen3-8B-coop-sft-full-260326-toy:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/toy \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen3-8B-coop-sft-full-260326 \
		--nproc 16

Llama-3.1-8B_sft_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/toy \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b_sft \
		--nproc 16

Llama-3.1-8B_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/toy \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-pc-dir /data/baiyixue/CAD/step_files_pc_2048_normalized \
		--gt-full-pc-dir /data/baiyixue/CAD/op_oriented_step_pc_2048_normalized \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b \
		--nproc 16

Llama-3.1-8B_noco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/noco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-pc-dir /data/baiyixue/CAD/step_files_pc_2048_normalized \
		--gt-full-pc-dir /data/baiyixue/CAD/op_oriented_step_pc_2048_normalized \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b \
		--nproc 16

Llama-3.1-8B_fco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/fco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-pc-dir /data/baiyixue/CAD/step_files_pc_2048_normalized \
		--gt-full-pc-dir /data/baiyixue/CAD/op_oriented_step_pc_2048_normalized \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b \
		--nproc 16

Llama-3.1-8B_test:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/test \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-pc-dir /data/baiyixue/CAD/step_files_pc_2048_normalized \
		--gt-full-pc-dir /data/baiyixue/CAD/op_oriented_step_pc_2048_normalized \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b \
		--nproc 16



Qwen3-8B-noco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/noco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen3-8B \
		--nproc 32

Qwen3-8B-fco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/fco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen3-8B \
		--nproc 32

Qwen3-8B-toy:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/toy \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen3-8B \
		--nproc 32


Qwen2.5-7B-noco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_only \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/noco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen2.5-7B \
		--nproc 32

Qwen2.5-7B-fco:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_acted_with_all_comments \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/fco \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen2.5-7B \
		--nproc 32

Qwen2.5-7B-toy:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--full-pre-code-dir /home/baiyixue/project/flowcad/data/pre_code \
		--meta-csv /home/baiyixue/project/flowcad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/flowcad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/toy \
		--prompts-csv /home/baiyixue/project/flowcad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step \
		--dedup-csv /home/baiyixue/project/flowcad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/flowcad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/toy_test.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2 \
		--http-model Qwen2.5-7B \
		--nproc 32


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
