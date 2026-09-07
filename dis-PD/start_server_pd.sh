#!/usr/bin/env bash
# 启动 GLM-5 四机 32 卡 PD 分离（2 Prefill + 2 Decode，每台各执行一次）。
#
# 用法：
#   P0 (29.209.115.238):  bash start_server_pd.sh prefill 0
#   P1 (29.209.114.88) :  bash start_server_pd.sh prefill 1
#   D0 (29.209.104.16) :  bash start_server_pd.sh decode  0
#   D1 (29.209.105.143):  bash start_server_pd.sh decode  1
#
# ⚠️ 启动顺序：Prefill 组必须先于 Decode 组（D 组要连 P 组 head 的 8998 bootstrap）。
#    worker 可以先起（会等本组 head）。
#
# 与共置版 start_server.sh 的差异：
#   1) 新增 role 参数，决定 --disaggregation-mode / API 端口 / dist-init-addr；
#   2) 新增 PD 三件套：--disaggregation-transfer-backend / -ib-device / -bootstrap-port；
#   3) HiCache 只在 prefill 组开（KV load 实验必须发生在 P 侧，D 侧开会污染归因）；
#   4) mem-fraction 按角色分开默认：P=0.60（KV 传完即释放，易制造 host 命中）
#                                   D=0.85（要长期驻留 KV，决定最大并发）。
#
# 常用变量组合（对应教程 §2.4 三流隔离对照实验）：
#   组 A 基线   ：HICACHE=off  bash start_server_pd.sh prefill 0
#   组 B 全量   ：             bash start_server_pd.sh prefill 0        # 默认
#   组 C EP隔离 ：IB_DEV=mlx5_bond_8 bash start_server_pd.sh prefill 0
#   扩容 L1/L2  ：MEM_FRACTION=0.75 HICACHE_RATIO=3 bash start_server_pd.sh prefill 0

set -euo pipefail

ROLE="${1:?用法: bash start_server_pd.sh <prefill|decode> <node_rank 0|1>}"
NODE_RANK="${2:?用法: bash start_server_pd.sh <prefill|decode> <node_rank 0|1>}"

P_HEAD="${P_HEAD:-29.209.115.238}"      # Prefill 组 head
D_HEAD="${D_HEAD:-29.209.104.16}"       # Decode  组 head
MODEL="${MODEL:-/data/models/GLM-5-FP8}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
IB_DEV="${IB_DEV:-mlx5_bond_1}"         # ★变量：换 mlx5_bond_8 做 §2.4 组 C 隔离对照
XFER_BACKEND="${XFER_BACKEND:-mooncake}"

# --- 可调容量参数（环境变量覆盖）---
#   HICACHE      : on|off，仅 prefill 生效（off = §2.4 组 A 基线）
#   HICACHE_RATIO: L2 (host DRAM) = L1 的倍数，仅 prefill 生效
#   MEM_FRACTION : L1 (GPU KV pool) 占比，按角色有不同默认值
HICACHE="${HICACHE:-on}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"

case "${ROLE}" in
  prefill)
    PORT="${PORT:-30000}"
    HEAD="${P_HEAD}"
    LOG="${LOG:-/tmp/prefill.log}"
    MEM_FRACTION="${MEM_FRACTION:-0.60}"
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-600}"
    ;;
  decode)
    PORT="${PORT:-30001}"
    HEAD="${D_HEAD}"
    LOG="${LOG:-/tmp/decode.log}"
    MEM_FRACTION="${MEM_FRACTION:-0.85}"
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-600}"
    ;;
  *) echo "[err] role 只能是 prefill 或 decode，收到: ${ROLE}"; exit 1 ;;
esac

# --- 清代理 + 进 venv（代理会劫持 bootstrap 注册，必须清）---
unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY all_proxy ALL_PROXY || true
cd /vllm-workspace && source .sglang_venv/bin/activate

# --- NVSHMEM / DeepEP 环境（跨机 EP 必需，四台都要）---
export LD_PRELOAD=/usr/local/nvshmem/lib/libnvshmem_host.so.3
export NVSHMEM_ENABLE_NIC_PE_MAPPING=0
export NVSHMEM_HCA_LIST=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
unset NVSHMEM_HCA_PE_MAPPING || true
export NVSHMEM_IB_GID_INDEX=3

# --- 若已在跑，先清理 ---
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 2

# --- 仅 prefill 组挂 HiCache 参数 ---
HICACHE_ARGS=()
if [ "${ROLE}" = "prefill" ] && [ "${HICACHE}" = "on" ]; then
    HICACHE_ARGS=(--enable-hierarchical-cache
                  --hicache-ratio "${HICACHE_RATIO}"
                  --hicache-io-backend "${HICACHE_IO_BACKEND:-kernel}"
                  --enable-cache-report)
fi

# --- 仅 prefill 组需要 chunked-prefill 调优 ---
EXTRA_ARGS=()
if [ "${ROLE}" = "prefill" ]; then
    EXTRA_ARGS=(--chunked-prefill-size "${CHUNKED_PREFILL:-16384}")
fi

echo "[start] role=${ROLE} node_rank=${NODE_RANK} dist-init=${HEAD}:5000 port=${PORT} -> ${LOG}"
echo "[start] mem-fraction=${MEM_FRACTION} hicache=${HICACHE}(ratio=${HICACHE_RATIO})"
echo "[start] pd: backend=${XFER_BACKEND} ib=${IB_DEV} bootstrap=${BOOTSTRAP_PORT}"

nohup python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --trust-remote-code \
    --tp 16 --dp 4 --enable-dp-attention \
    --ep-size 16 --moe-a2a-backend deepep \
    --attention-backend nsa \
    --nnodes 2 --node-rank "${NODE_RANK}" --dist-init-addr "${HEAD}:5000" \
    --disaggregation-mode "${ROLE}" \
    --disaggregation-transfer-backend "${XFER_BACKEND}" \
    --disaggregation-ib-device "${IB_DEV}" \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    "${HICACHE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --page-size 64 \
    --mem-fraction-static "${MEM_FRACTION}" \
    --enable-metrics \
    --enable-request-time-stats-logging \
    --max-running-requests 128 \
    --host 0.0.0.0 --port "${PORT}" \
    > "${LOG}" 2>&1 &

echo "[start] launched, pid=$!  tail -f ${LOG}"
echo "[start] 成功标志: 'The server is fired up and ready to roll!'"
