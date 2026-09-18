#!/usr/bin/env bash
# 启动 GLM-5.1 双机单实例（每台各执行一次）。
# 用法：
#   head  (29.209.104.16):  bash start_server.sh 0
#   worker(29.209.114.88):  bash start_server.sh 1
# 说明：node-rank 由第 1 个参数给出；其余参数两台一致。

set -euo pipefail

NODE_RANK="${1:?用法: bash start_server.sh <node_rank 0|1>}"
HEAD_ADDR="${HEAD_ADDR:-29.209.104.16}"
MODEL="${MODEL:-/data/models/GLM-5-FP8}"
LOG="${LOG:-/tmp/glm5.log}"

# --- 可调容量参数（环境变量覆盖，不传用默认）---
# L1（GPU KV pool）：调大 MEM_FRACTION 扩大单卡可放的 token 数（55040 是 0.60 时的值）。
#   例：MEM_FRACTION=0.75 bash start_server.sh 0
# L2（host DRAM 池）：HICACHE_RATIO 是 L1 的倍数，调大更不易把命中前缀挤出 host。
#   例：HICACHE_RATIO=3 bash start_server.sh 0
# 两者可同时给：MEM_FRACTION=0.75 HICACHE_RATIO=3 bash start_server.sh 0
MEM_FRACTION="${MEM_FRACTION:-0.60}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"

# --- 清代理 + 进 venv ---
unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY all_proxy ALL_PROXY || true
cd /vllm-workspace && source .sglang_venv/bin/activate

# --- NVSHMEM / DeepEP 环境（§1.5，跨机 EP 必需）---
export LD_PRELOAD=/usr/local/nvshmem/lib/libnvshmem_host.so.3
export NVSHMEM_ENABLE_NIC_PE_MAPPING=0
export NVSHMEM_HCA_LIST=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
unset NVSHMEM_HCA_PE_MAPPING || true
export NVSHMEM_IB_GID_INDEX=3

# --- 若已在跑，先清理 ---
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 2

echo "[start] node_rank=${NODE_RANK} head=${HEAD_ADDR} model=${MODEL} -> ${LOG}"
echo "[start] L1 mem-fraction-static=${MEM_FRACTION}  L2 hicache-ratio=${HICACHE_RATIO}"
nohup python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --trust-remote-code \
    --tp 16 --dp 4 --enable-dp-attention \
    --ep-size 16 --moe-a2a-backend deepep \
    --attention-backend nsa \
    --nnodes 2 --node-rank "${NODE_RANK}" --dist-init-addr "${HEAD_ADDR}:5000" \
    --enable-hierarchical-cache \
    --hicache-ratio "${HICACHE_RATIO}" \
    --hicache-io-backend kernel \
    --page-size 64 \
    --mem-fraction-static "${MEM_FRACTION}" \
    --chunked-prefill-size 16384 \
    --enable-metrics \
    --enable-cache-report \
    --enable-request-time-stats-logging \
    --max-running-requests 128 \
    --host 0.0.0.0 --port 30000 \
    > "${LOG}" 2>&1 &

echo "[start] launched, pid=$!  tail -f ${LOG}"
