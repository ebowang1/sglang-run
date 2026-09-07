#!/usr/bin/env bash
# head(29.209.104.16) 上的一键总控：等待实例就绪 -> 健康检查 -> 跑 KV load 实验(+可选 profile)。
# 前提：两台已各自 `bash /vllm-workspace/sglang-run/start_server.sh 0` / `... 1`。
#
# 用法：
#   bash run_all.sh                    # 等就绪 + 只验证 metrics
#   bash run_all.sh --profile          # 等就绪 + 抓 1 轮 profile
#   bash run_all.sh --profile --repeat 5
#
# 透传给 kv_profile_auto.py 的参数：把 run_all.sh 之后的所有参数原样传下去。

set -uo pipefail

LOG="${LOG:-/tmp/glm5.log}"
PORT="${PORT:-30000}"
AUTO="${AUTO:-/vllm-workspace/sglang-run/kv_profile_auto.py}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # 最长等 30 分钟（含 DeepGEMM warmup）

cd /vllm-workspace && source .sglang_venv/bin/activate
mkdir -p /tmp/sgprof

# 1) 等待就绪：优先看 ready 标志，其次探 /health
echo "[wait] 等待实例就绪（最长 ${READY_TIMEOUT}s）..."
t0=$(date +%s)
while true; do
    if grep -q "fired up and ready to roll" "${LOG}" 2>/dev/null; then
        echo "[wait] 检测到 ready 标志"; break
    fi
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)
    if [ "${code}" = "200" ]; then
        echo "[wait] /health 200"; break
    fi
    now=$(date +%s)
    if [ $((now - t0)) -gt "${READY_TIMEOUT}" ]; then
        echo "[wait] 超时未就绪，最近日志："; tail -n 20 "${LOG}" 2>/dev/null; exit 1
    fi
    sleep 5
done

# 2) 关键参数确认（metrics 必须开）
echo "[check] 生效参数："
grep -oE 'enable_metrics=(True|False)|enable_hierarchical_cache=(True|False)|hicache_io_backend=.?[a-z]+.?|chunked_prefill_size=[0-9]+' "${LOG}" | sort -u | head

# 3) 健康探测一条推理
echo "[check] 冒烟推理："
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/GLM-5-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' \
  -o /dev/null -w "  chat http=%{http_code}\n"

# 4) 确认自动化脚本在位
if [ ! -f "${AUTO}" ]; then
    echo "[err] 未找到 ${AUTO}，请先创建（见教程 §第三章 完整脚本）"; exit 1
fi

# 5) 跑实验：把本脚本之后的参数原样透传
#    未显式给 --tag 时，自动用时间戳
ARGS="$*"
case "${ARGS}" in
  *--tag*) : ;;
  *) ARGS="--tag auto-$(date +%H%M%S) ${ARGS}" ;;
esac

echo "[run] python ${AUTO} ${ARGS}"
python "${AUTO}" ${ARGS}
