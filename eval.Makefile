PYTHON := /home/baiyixue/miniforge3/envs/op-cad/bin/python
SCRIPT := /home/baiyixue/project/op-cad/evaluation/evaluation.py

.PHONY: run gemini

# 用脚本默认参数
run:
	$(PYTHON) $(SCRIPT)

# 带参数的目标
gemini_test:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/test/inference \
		--prompts-csv /home/baiyixue/project/op-cad/test/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--gen-mode api \
		--no-resume


gemini_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/test_subset.json \
		--split-key test \
		--provider openai \
		--nproc 128 \
		--resume

gemini_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/test_subset.json \
		--split-key test \
		--provider openai \
		--nproc 128 \
		--resume

gpt4o_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /data/baiyixue/CAD/inference_result/main/gpt-4-BDATA \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/test_subset.json \
		--split-key test \
		--gen-mode api \
		--provider http \
		--http-model gpt-4 \
		--nproc 32 \
		--resume

gpt4o_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /data/baiyixue/CAD/inference_result/main/gpt-4-BDATA \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_scale_short \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop_scale_short \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/test_subset.json \
		--split-key test \
		--gen-mode api \
		--provider http \
		--http-model gpt-4 \
		--nproc 32 \
		--resume

qwen_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider http \
		--http-model Qwen \
		--nproc 128 \
		--resume



deepseek_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/test_subset.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model deepseek-ai/DeepSeek-V3 \
		--nproc 16 \

deepseek_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/test_subset.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model deepseek-ai/DeepSeek-V3 \
		--nproc 16 \

qwen3-coder_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
		--nproc

qwen3-coder_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
		--nproc


qwen3_std_lab1:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/spilt_for_qwen/unrun_part1.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--thinking \
		--vllm-endpoint-key port1\
		--http-model Qwen_Qwen3-8B \
		--nproc




deepseek_r1_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/test_subset.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model deepseek-ai/DeepSeek-R1 \
		--nproc 16 \

llama70_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port4\
		--http-model Llama-3.1-70B-Instruct \
		--nproc

llama70_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port4\
		--http-model Llama-3.1-70B-Instruct \
		--nproc 4 \


qwen3_cop_wo_thinking:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port0\
		--http-model Qwen_Qwen3-8B_WOT \
		--nproc


deepseekr1_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/test_subset.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model deepseek-ai/DeepSeek-R1 \
		--nproc 16 \





qwen3_cop_1:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/qwen1 \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_10_parts/unruncop_part1.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--thinking \
		--vllm-endpoint-key port1\
		--http-model Qwen_Qwen3-8B \
		--nproc

qwen3_cop_2:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_10_parts/unruncop_part2.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--thinking \
		--vllm-endpoint-key port5\
		--http-model Qwen_Qwen3-8B \
		--nproc

qwen3_cop_3:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--thinking \
		--vllm-endpoint-key port2\
		--http-model Qwen_Qwen3-8B \
		--nproc 128 \



qwen3_cop_hpc1:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/spilt_for_qwen/unruncop_part1.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port3\
		--http-model Qwen_Qwen3-8B \
		--thinking \
		--nproc 128 \



qwen3-30_std_os:
	$(PYTHON) $(SCRIPT) --mode std \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model Qwen/Qwen3-30B-A3B \
		--nproc 32

qwen3-30_cop_os:
	$(PYTHON) $(SCRIPT) --mode cop \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model Qwen/Qwen3-30B-A3B \
		--nproc 32


gemini_std_os:
	$(PYTHON) $(SCRIPT) --mode std \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--provider openai \
		--nproc 128 \
		--resume

gemini_cop_os:
	$(PYTHON) $(SCRIPT) --mode cop \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--provider openai \
		--nproc 128 \
		--resume

llama70_cop_os:
	$(PYTHON) $(SCRIPT) --mode cop \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model Llama-3.1-70B-Instruct \
		--nproc

llama70_std_os:
	$(PYTHON) $(SCRIPT) --mode std \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model Llama-3.1-70B-Instruct \
		--nproc

llama-3.1-8b-instruct_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_scale_short \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop_scale_short \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/main \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model Llama-3.1-8B-Instruct-BDATA-lora \
		--nproc 16


llama-3.1-8b-instruct_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_scale_short \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop_scale_short \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/main \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model Llama-3.1-8B-Instruct-BDATA-lora \
		--nproc 16


qwen3_cop_wo_thinking:
	$(PYTHON) $(SCRIPT) --mode cop \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_scale_short \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop_scale_short \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/main \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port0\
		--http-model Qwen_Qwen3-8B-BDATA \
		--nproc 16

qwen3_std_os:
	$(PYTHON) $(SCRIPT) --mode std \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--thinking \
		--vllm-endpoint-key port1\
		--http-model Qwen_Qwen3-8B \
		--nproc 64

qwen3_cop_os:
	$(PYTHON) $(SCRIPT) --mode cop \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--thinking \
		--vllm-endpoint-key port1\
		--http-model Qwen_Qwen3-8B \
		--nproc 64

gpt4o_std_os:
	$(PYTHON) $(SCRIPT) --mode std \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
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
		--nproc 32 \
		--resume

gpt4o_cop_os:
	$(PYTHON) $(SCRIPT) --mode cop \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
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
		--nproc 32
		--resume

qwen3-coder_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
		--nproc 32


qwen3-coder_cop_os:
	$(PYTHON) $(SCRIPT) --mode cop \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
		--nproc 32


deepseek_std_os:
	$(PYTHON) $(SCRIPT) --mode std \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model deepseek-ai/DeepSeek-V3 \
		--nproc 16


deepseek_cop_os:
	$(PYTHON) $(SCRIPT) --mode cop \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model deepseek-ai/DeepSeek-V3 \
		--nproc 16


deepseekr1_cop_os:
	$(PYTHON) $(SCRIPT) --mode cop \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model deepseek-ai/DeepSeek-R1 \
		--nproc 16 \

deepseekr1_std_os:
	$(PYTHON) $(SCRIPT) --mode std \
		--oneshot \
		--oneshot-csv /home/baiyixue/project/op-cad/data/one_shot_test/one_shot_filled.csv \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv \
		--out-root /home/baiyixue/project/op-cad/inference_one_shot \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/one_shot_test/sampled_for_one_shot.json \
		--split-key test \
		--gen-mode api \
		--provider siliconflow \
		--http-model deepseek-ai/DeepSeek-R1 \
		--nproc 16 \

qwen3vl_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model Qwen3-VL-8B-Instruct \
		--nproc 64

qwen3vl_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model Qwen3-VL-8B-Instruct \
		--nproc 64

qwen3vl_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /home/baiyixue/project/op-cad/inference_results \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2\
		--http-model Qwen3-VL-30B-A3B-Instruct \
		--nproc 64

qwen30vl_cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root  /data/baiyixue/CAD/inference_result/main \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2\
		--http-model Qwen3-VL-30B-A3B-Instruct \
		--nproc 64
qwen30vl_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /data/baiyixue/CAD/inference_result/main \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2\
		--http-model Qwen3-VL-30B-A3B-Instruct \
		--nproc 64

qwen3lorastd_std_wo_thinking:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /data/baiyixue/CAD/inference_result/main \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_scale_short \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop_scale_short \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port-1\
		--http-model Qwen_Qwen3-8B-LORA-STD-BDATA \
		--nproc 64

qwen3lorastd_cop_wo_thinking:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /data/baiyixue/CAD/inference_result/main \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_scale_short \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop_scale_short \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port-1\
		--http-model Qwen_Qwen3-8B-LORA-STD-BDATA \
		--nproc 64


qwen3lorastd_std_wo_thinking_plane:
	$(PYTHON) $(SCRIPT) --mode std \
		--out-root /data/baiyixue/CAD/inference_result/prompt \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_scale_short \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop_scale_short \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port-1\
		--http-model Qwen_Qwen3-8B-LORA-STD-BDATA \
		--nproc 64

qwen3lorastd_cop_wo_thinking_plane:
	$(PYTHON) $(SCRIPT) --mode cop \
		--out-root /data/baiyixue/CAD/inference_result/prompt \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_scale_short \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop_scale_short \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files \
		--op-orient-dir /data/baiyixue/CAD/op_orientated_step \
		--dedup-csv /home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_index.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port-1\
		--http-model Qwen_Qwen3-8B-LORA-STD-BDATA \
		--nproc 64


llama-3.1-8b-std:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/op-cad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/Bdata_v2 \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files_sketch \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step_sketch \
		--dedup-csv /home/baiyixue/project/op-cad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/small_subset.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model Llama-3.1-8B-Instruct \
		--nproc 8


llama-3.1-8b-cop:
	$(PYTHON) $(SCRIPT) --mode cop \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/op-cad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/Bdata_v2 \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files_sketch \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step_sketch \
		--dedup-csv /home/baiyixue/project/op-cad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/small_subset.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model Llama-3.1-8B-Instruct \
		--nproc 8

qwen3_std_wo_thinking:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/op-cad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/Bdata_v2 \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files_sketch \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step_sketch \
		--dedup-csv /home/baiyixue/project/op-cad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/small_subset.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port2\
		--http-model Qwen_Qwen3-8B \
		--nproc 16

Llama-3.1-8B_full_std:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/op-cad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/Bdata_v2 \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files_sketch \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step_sketch \
		--dedup-csv /home/baiyixue/project/op-cad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/small_subset.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1\
		--http-model llama_3.1_8b_coop_sft_full \
		--nproc 32

Llama-3.1-8B_full_std_full:
	$(PYTHON) $(SCRIPT) --mode std \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/op-cad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/main \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files_sketch \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step_sketch \
		--dedup-csv /home/baiyixue/project/op-cad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b_coop_sft_full \
		--nproc 32

Llama-3.1-8B_full_cop_full:
	$(PYTHON) $(SCRIPT) --mode cop \
		--pre-code-dir /home/baiyixue/project/op-cad/data/pre_code \
		--cop-pre-code-dir /home/baiyixue/project/op-cad/data/pre_code_cop \
		--meta-csv /home/baiyixue/project/op-cad/data/data_indication_out.csv \
		--bool-csv /home/baiyixue/project/op-cad/data/bool.csv \
		--out-root /data/baiyixue/CAD/inference_result/main \
		--prompts-csv /home/baiyixue/project/op-cad/data/prompt.csv \
		--gt-image-dir /data/baiyixue/CAD/screenshots \
		--gt-single-step-dir /data/baiyixue/CAD/step_files_sketch \
		--op-orient-dir /data/baiyixue/CAD/op_oriented_step_sketch \
		--dedup-csv /home/baiyixue/project/op-cad/data/dedup.csv \
		--gt-edges-dir /home/baiyixue/project/op-cad/data/gt_edges_json \
		--split-json /home/baiyixue/project/op-cad/data/split_result.json \
		--split-key test \
		--gen-mode api \
		--provider vllm \
		--vllm-endpoint-key port1 \
		--http-model llama_3.1_8b_coop_sft_full \
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
