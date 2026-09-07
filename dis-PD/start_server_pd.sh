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
#
# ★坏卡剔除（BAD_HCA）：
#   某些机器个别 HCA 的 index3 RoCE v2 GID 缺失/全零（体检方法见教程 §1.4.1：
#   grep -H . /sys/class/infiniband/mlx5_bond_*/ports/1/gids/3）。
#   把坏卡编号用逗号传给 BAD_HCA，脚本会自动从 NVSHMEM_HCA_LIST 与 NCCL_IB_HCA
#   两套白名单中剔除，并校验 --disaggregation-ib-device 未落在坏卡上。
#   实测结论：8 个 GPU 共享 6 张好卡即可跑通 DeepEP internode，
#   因此【不需要】改 --tp/--dp/--ep-size，也【不需要】设 CUDA_VISIBLE_DEVICES。
#
#   例（D0=16 与 D1=143 坏 bond_2 / bond_5）：
#     BAD_HCA=2,5 bash start_server_pd.sh decode 0     # 在 16 上
#     BAD_HCA=2,5 bash start_server_pd.sh decode 1     # 在 143 上
#   也可写全名：BAD_HCA=mlx5_bond_2,mlx5_bond_5

set -euo pipefail

ROLE="${1:?用法: bash start_server_pd.sh <prefill|decode> <node_rank 0|1>}"
NODE_RANK="${2:?用法: bash start_server_pd.sh <prefill|decode> <node_rank 0|1>}"

P_HEAD="${P_HEAD:-29.209.115.238}"      # Prefill 组 head
D_HEAD="${D_HEAD:-29.209.104.16}"       # Decode  组 head
MODEL="${MODEL:-/data/models/GLM-5-FP8}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
IB_DEV="${IB_DEV:-mlx5_bond_1}"         # ★变量：换 mlx5_bond_8 做 §2.4 组 C 隔离对照
XFER_BACKEND="${XFER_BACKEND:-mooncake}"
ALL_HCA_IDS="${ALL_HCA_IDS:-1 2 3 4 5 6 7 8}"   # 本机数据面 HCA 全集（mlx5_bond_N）
BAD_HCA="${BAD_HCA:-}"                  # ★坏卡编号，逗号分隔，如 "2,5"

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

# --- ★按 BAD_HCA 生成好卡白名单（NVSHMEM 与 NCCL 两套都要，缺一不可）---
# 说明：只从白名单剔除坏卡，GPU 仍用满 8 个（实测 8 GPU 共享 6 张好卡可跑通 EP）。
is_bad_hca () {   # $1 = HCA 序号；返回 0=坏卡，1=好卡
    local id="$1" b
    if [ -z "${BAD_HCA}" ]; then
        return 1
    fi
    for b in $(echo "${BAD_HCA}" | tr ',' ' '); do
        b="${b#mlx5_bond_}"                       # 兼容传全名 mlx5_bond_2
        if [ "${b}" = "${id}" ]; then
            return 0
        fi
    done
    return 1
}

NVSHMEM_LIST=""; NCCL_LIST=""; GOOD_CNT=0
for i in ${ALL_HCA_IDS}; do
    dev="mlx5_bond_${i}"
    if is_bad_hca "${i}"; then
        echo "[hca] skip ${dev} (BAD_HCA)"
        continue
    fi
    NVSHMEM_LIST="${NVSHMEM_LIST:+${NVSHMEM_LIST},}${dev}:1"
    NCCL_LIST="${NCCL_LIST:+${NCCL_LIST},}${dev}"
    GOOD_CNT=$((GOOD_CNT + 1))
done

if [ "${GOOD_CNT}" -eq 0 ]; then
    echo "[err] 好卡数为 0，请检查 BAD_HCA / ALL_HCA_IDS"; exit 1
fi

# --disaggregation-ib-device 必须落在好卡上，否则 KV 传输握手失败或退化 socket
if is_bad_hca "${IB_DEV#mlx5_bond_}"; then
    echo "[err] IB_DEV=${IB_DEV} 在 BAD_HCA(${BAD_HCA}) 列表中，PD KV 传输会失败。"
    echo "      请显式指定一张好卡，例如: IB_DEV=mlx5_bond_1"; exit 1
fi

# --- NVSHMEM / DeepEP 环境（跨机 EP 必需，四台都要）---
export LD_PRELOAD=/usr/local/nvshmem/lib/libnvshmem_host.so.3
export NVSHMEM_ENABLE_NIC_PE_MAPPING=0
export NVSHMEM_HCA_LIST="${NVSHMEM_LIST}"
unset NVSHMEM_HCA_PE_MAPPING || true
export NVSHMEM_IB_GID_INDEX="${GID_INDEX:-3}"

# --- NCCL 同步收窄到同一批好卡（NCCL_IB_HCA 只管 NCCL，不影响 NVSHMEM）---
export NCCL_IB_HCA="${NCCL_LIST}"
export NCCL_IB_GID_INDEX="${GID_INDEX:-3}"

# --- ★DeepEP 超时放宽（实测必需）---
# 背景：deep_ep/buffer.py 中 CPU 侧默认超时仅 100s（下限也是 100）。
#   首次跑真实请求时会撞上 DeepGEMM JIT 冷编译（日志提示"通常 10-20 分钟"），
#   部分 rank 仍在编译、未进入 dispatch，已到达的 rank 等不到就报
#   "RuntimeError: DeepEP error: timeout (dispatch CPU)" → scheduler exit -3 → SIGQUIT。
#   实测：13:20:30 进 JIT，13:22:11 报错，间隔 101s，正好卡在 100s 阈值。
# 治本仍是 §预编译（见脚本末尾提示），此处放宽只是兜底。
export DEEPEP_NUM_CPU_TIMEOUT_SECONDS="${DEEPEP_CPU_TIMEOUT:-1800}"
export DEEPEP_NUM_GPU_TIMEOUT_CYCLES_IN_BILLIONS="${DEEPEP_GPU_TIMEOUT:-800}"

echo "[hca] 好卡 ${GOOD_CNT} 张  NVSHMEM_HCA_LIST=${NVSHMEM_HCA_LIST}"
echo "[hca] NCCL_IB_HCA=${NCCL_IB_HCA}  GID_INDEX=${GID_INDEX:-3}"
echo "[deepep] CPU_TIMEOUT=${DEEPEP_NUM_CPU_TIMEOUT_SECONDS}s  GPU_TIMEOUT=${DEEPEP_NUM_GPU_TIMEOUT_CYCLES_IN_BILLIONS}G cycles"

# --- ★好卡 GID 自检（GID_CHECK=0 可跳过）---
# 目的：坏卡若漏进白名单，会在 EP 建 QP 时报 ibv_modify_qp failed / remote_gid=::，
#      在这里提前拦住比等 watchdog 超时 300s 更省时间。
if [ "${GID_CHECK:-1}" = "1" ]; then
    bad_found=""
    for d in $(echo "${NCCL_IB_HCA}" | tr ',' ' '); do
        f="/sys/class/infiniband/${d}/ports/1/gids/${GID_INDEX:-3}"
        if [ ! -e "${f}" ]; then
            bad_found="${bad_found} ${d}(MISSING)"; continue
        fi
        g=$(cat "${f}" 2>/dev/null)
        if [ "${g}" = "0000:0000:0000:0000:0000:0000:0000:0000" ]; then
            bad_found="${bad_found} ${d}(ZERO)"
        fi
    done
    if [ -n "${bad_found}" ]; then
        echo "[err] 白名单中仍存在坏卡:${bad_found}"
        echo "      请把它们加入 BAD_HCA 后重试，例如: BAD_HCA=2,5 bash $0 ${ROLE} ${NODE_RANK}"
        exit 1
    fi
    echo "[hca] GID 自检通过（index ${GID_INDEX:-3} 全部有效）"
fi

# --- 若已在跑，先彻底清理 ---
# ★必须清干净：残留进程会导致 Prometheus 指标重复注册
#   （ValueError: Duplicated timeseries in CollectorRegistry），新实例直接启动失败。
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang::" 2>/dev/null || true
pkill -9 -f "port ${PORT}" 2>/dev/null || true
sleep 3
if ss -lntp 2>/dev/null | grep -qE ":(${PORT}|5000|${BOOTSTRAP_PORT})\b"; then
    echo "[warn] 端口仍被占用，再等 5s..."
    ss -lntp 2>/dev/null | grep -E ":(${PORT}|5000|${BOOTSTRAP_PORT})\b"
    pkill -9 -f sglang 2>/dev/null || true
    sleep 5
fi

# --- 仅 prefill 组挂 HiCache 参数 ---
HICACHE_ARGS=()
if [ "${ROLE}" = "prefill" ] && [ "${HICACHE}" = "on" ]; then
    HICACHE_ARGS=(--enable-hierarchical-cache
                  --hicache-ratio "${HICACHE_RATIO}"
                  --hicache-io-backend "${HICACHE_IO_BACKEND:-kernel}"
                  --enable-cache-report)
fi

# --- 仅 prefill 组需要 chunked-prefill 调优 ---
# 默认 4096（与 SGLang 默认一致）；需要拉大单卡 prefill token 数时显式指定，例如：
#   CHUNKED_PREFILL=16384 bash start_server_pd.sh prefill 0
EXTRA_ARGS=()
if [ "${ROLE}" = "prefill" ]; then
    EXTRA_ARGS=(--chunked-prefill-size "${CHUNKED_PREFILL:-4096}")
fi

echo "[start] role=${ROLE} node_rank=${NODE_RANK} dist-init=${HEAD}:5000 port=${PORT} -> ${LOG}"
echo "[start] mem-fraction=${MEM_FRACTION} hicache=${HICACHE}(ratio=${HICACHE_RATIO})"
echo "[start] pd: backend=${XFER_BACKEND} ib=${IB_DEV} bootstrap=${BOOTSTRAP_PORT}"
if [ "${ROLE}" = "prefill" ]; then
    echo "[start] chunked-prefill-size=${CHUNKED_PREFILL:-4096}"
fi

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
echo ""
echo "[hint] 若日志出现 'DeepEP error: timeout (dispatch CPU)' 且伴随 DeepGEMM JIT 编译，"
echo "       说明 JIT 冷编译超过了 DeepEP 等待窗口。治本方式（每台跑一次，之后缓存复用）："
echo "       python3 -m sglang.compile_deep_gemm --model ${MODEL} \\"
echo "           --tp 16 --dp 4 --ep-size 16 --attention-backend nsa --trust-remote-code"
