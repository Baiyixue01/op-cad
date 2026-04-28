source /data/baiyixue/envs/vllm-py310/bin/activate
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m vllm.entrypoints.openai.api_server \
# --model /data/baiyixue/inference_model/Qwen3-8B \
# --tensor-parallel-size 8 \
# --hf-overrides '{"rope_parameters":{"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}' \
# --max-model-len 131072 \
# --port 8001 \
# --dtype bfloat16 \
# --gpu-memory-utilization 0.9 \
# --served-model-name Qwen3-8B

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m vllm.entrypoints.openai.api_server \
#   --model /data/baiyixue/inference_model/Qwen2.5-7B \
#   --tensor-parallel-size 8 \
#   --hf-overrides '{"rope_parameters":{"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}' \
#   --max-model-len 131072 \
#   --port 8001 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.9 \
#   --served-model-name Qwen2.5-7B

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m vllm.entrypoints.openai.api_server \
  --model /data/baiyixue/inference_model/llama_3.1_8b_coop_sft_full \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 128000 \
  --port 8001 \
  --served-model-name tool_agent

