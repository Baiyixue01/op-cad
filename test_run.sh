source /data/baiyixue/envs/vllm-py310/bin/activate
cd /home/baiyixue/project/op-cad
make -f eval.Makefile Qwen2.5-7B-noco
make -f eval.Makefile Qwen2.5-7B-fco
make -f eval.Makefile Qwen2.5-7B-toy