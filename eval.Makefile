PYTHON := /data/baiyixue/envs/trl/bin/python
SCRIPT := /home/baiyixue/project/op-cad/reward/evaluation.py

.PHONY: run gemini

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
