#!/usr/bin/env bash
set -euo pipefail

IDLE_SECONDS=300
CHECK_INTERVAL=5
UTIL_THRESHOLD=5

idle_time=0

is_all_gpu_idle() {
    local utils
    mapfile -t utils < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    if [ ${#utils[@]} -eq 0 ]; then
        return 1
    fi

    for u in "${utils[@]}"; do
        u="$(echo "$u" | tr -d '[:space:]')"
        if [ -z "$u" ]; then
            return 1
        fi
        if [ "$u" -gt "$UTIL_THRESHOLD" ]; then
            return 1
        fi
    done

    return 0
}

kill_current_vllm() {
    echo "[INFO] 所有 GPU 持续空闲 ${IDLE_SECONDS}s，准备 kill 当前 vLLM"

    pkill -f "vllm.entrypoints.openai.api_server" || true
    pkill -f "vllm serve" || true

    sleep 3
    echo "[INFO] 当前 vLLM 已尝试结束"
}

start_new_pipeline() {
    echo "[INFO] 启动新的 run_vllm.sh"
    bash run_vllm.sh > vllm.log 2>&1 &

    echo "[INFO] 等待 300 秒让 vLLM 完成启动"
    sleep 300

    echo "[INFO] 启动 test_run.sh"
    bash test_run.sh > test.log 2>&1
}

while true; do
    if is_all_gpu_idle; then
        idle_time=$((idle_time + CHECK_INTERVAL))
        echo "[INFO] 所有 GPU 低利用率持续 ${idle_time}/${IDLE_SECONDS}s"
    else
        if [ "$idle_time" -ne 0 ]; then
            echo "[INFO] 检测到 GPU 重新活跃，空闲计时清零"
        fi
        idle_time=0
    fi

    if [ "$idle_time" -ge "$IDLE_SECONDS" ]; then
        kill_current_vllm
        start_new_pipeline
        exit 0
    fi

    sleep "$CHECK_INTERVAL"
done