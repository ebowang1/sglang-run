#!/usr/bin/env bash
# 启动 GLM-5 四机 32 卡 PD 分离（2 Prefill + 2 Decode）—— ★decode/hisparse profile 专用版。
#
# 与 dis-PD-prefill/start_server_pd.sh 的差异（仅 decode 相关，部署部分一字不改）：
#   1) decode 侧【默认开启 hisparse】（HISPARSE 默认 on，而非 off）——本目录实验主体；
#   2) 打印段强调 decode profile 相关提示（swap-in / EP 冲突 / MEM_FRACTION 与显存）；
#   3) 其余（NVSHMEM/坏卡剔除/PD 三件套/GID 自检/DeepEP 超时/清理逻辑）与 prefill 版完全一致。
#
# 用法：
#   P0 (29.209.115.238):  bash start_server_pd.sh prefill 0
#   P1 (29.209.114.88) :  bash start_server_pd.sh prefill 1
#   D0 (29.209.104.16) :  bash start_server_pd.sh decode  0      # 默认已开 hisparse
#   D1 (29.209.105.143):  bash start_server_pd.sh decode  1
#
# ⚠️ 启动顺序：Prefill 组必须先于 Decode 组（D 组要连 P 组 head 的 8998 bootstrap）。
#
# ★本目录实验目标：decode 侧开 hisparse、压高 batch，抓 EP low-latency 通信
#   与 hisparse swap-in/swap-out 在【默认流】上的冲突。代码依据（sglang/srt）：
#     swap-in ：nsa_backend.py forward_decode → swap_in_selected_pages(layer_id)
#               【每 decode step × 每层(78) × 每请求】，在默认流、且 page_table_1
#               是 attention 直接输入 → 阻塞、无法被掩盖。
#     swap-out：schedule_batch.py prepare_for_decode → _eager_backup_previous_token
#               【每 decode step 每请求 1 token×78 层】，也在默认流。
#   ⚠️ 开 hisparse 会强制 NSA backend=flashmla_sparse（server_args.py:1450），
#      A/B 基线组必须显式 NSA_DECODE_BACKEND=flashmla_sparse 对齐 backend，
#      否则测到的是 "backend 差异" 而非 "swap 开销"。
#
# ★命中规律（决定能否观测到 swap-in，见 factory.py:_parse_sparse_config）：
#     top_k                2048        每层取多少 token 参与 sparse attention
#     device_buffer_size   2*top_k     GPU 端 buffer 容量，必须 >= top_k
#     host_to_device_ratio 2           host pool = device buffer 的倍数
#   seq_len <= device_buffer_size 时全部预载(几乎无 swap-in)；
#   超过则每次 top-k 查找都 miss → 必须 host load（swap-in 流量最大）。
#   ★所以压测端 input_len 必须 > device_buffer_size，否则 profile 里看不到 swap-in。
#
# ★显存提醒（呼应此前 host-pin OOM）：
#   hisparse 的 req_to_device_buffer / req_to_host_pool 形状是 (max_num_reqs, ...)，
#   batch 越大占用越高；host pool 随 MEM_FRACTION 线性涨（单 rank 55GB × 8 = 440GB pin）。
#   若启动即报 cudaErrorInvalidValue，先降 MEM_FRACTION（如 0.6）或查容器 ulimit -l。
#
#   用法：
#     bash start_server_pd.sh decode 0                                  # 默认开 hisparse
#     HISPARSE_TOPK=2048 HISPARSE_DEV_BUF=4096 bash start_server_pd.sh decode 0
#     HISPARSE_CONFIG='{"top_k":512,"device_buffer_size":16384}' bash start_server_pd.sh decode 0
#     # A/B 基线组（关 hisparse，但对齐 backend，用于纯净对照）：
#     HISPARSE=off NSA_DECODE_BACKEND=flashmla_sparse bash start_server_pd.sh decode 0

set -euo pipefail

ROLE="${1:?用法: bash start_server_pd.sh <prefill|decode> <node_rank 0|1>}"
NODE_RANK="${2:?用法: bash start_server_pd.sh <prefill|decode> <node_rank 0|1>}"

P_HEAD="${P_HEAD:-29.209.115.238}"      # Prefill 组 head
D_HEAD="${D_HEAD:-29.209.104.16}"       # Decode  组 head
MODEL="${MODEL:-/data/models/GLM-5-FP8}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
IB_DEV="${IB_DEV:-mlx5_bond_1}"         # ★变量：换 mlx5_bond_8 做 EP 隔离对照
XFER_BACKEND="${XFER_BACKEND:-mooncake}"
ALL_HCA_IDS="${ALL_HCA_IDS:-1 2 3 4 5 6 7 8}"   # 本机数据面 HCA 全集（mlx5_bond_N）
BAD_HCA="${BAD_HCA:-}"                  # ★坏卡编号，逗号分隔，如 "2,5"

# --- 可调容量参数（环境变量覆盖）---
HICACHE="${HICACHE:-on}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"

# --- ★HiSparse 参数（仅 decode 生效）---
#   ★本目录默认 on（与 prefill 版唯一的默认值差异）
#   HISPARSE        : on|off，默认 on
#   HISPARSE_TOPK   : top_k，默认 2048
#   HISPARSE_DEV_BUF: device_buffer_size，默认 2*top_k（必须 >= top_k）
#   HISPARSE_H2D    : host_to_device_ratio，默认 2
#   HISPARSE_CONFIG : 直接给完整 JSON，给了则忽略上面三个分项
#   NSA_DECODE_BACKEND / NSA_PREFILL_BACKEND: 显式指定 NSA backend（做纯净 A/B 用）
HISPARSE="${HISPARSE:-on}"
HISPARSE_TOPK="${HISPARSE_TOPK:-2048}"
HISPARSE_DEV_BUF="${HISPARSE_DEV_BUF:-}"
HISPARSE_H2D="${HISPARSE_H2D:-2}"
HISPARSE_CONFIG="${HISPARSE_CONFIG:-}"

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

# --- ★DeepEP 超时放宽（实测必需，兜底 DeepGEMM JIT 冷编译）---
export DEEPEP_NUM_CPU_TIMEOUT_SECONDS="${DEEPEP_CPU_TIMEOUT:-1800}"
export DEEPEP_NUM_GPU_TIMEOUT_CYCLES_IN_BILLIONS="${DEEPEP_GPU_TIMEOUT:-800}"

echo "[hca] 好卡 ${GOOD_CNT} 张  NVSHMEM_HCA_LIST=${NVSHMEM_HCA_LIST}"
echo "[hca] NCCL_IB_HCA=${NCCL_IB_HCA}  GID_INDEX=${GID_INDEX:-3}"
echo "[deepep] CPU_TIMEOUT=${DEEPEP_NUM_CPU_TIMEOUT_SECONDS}s  GPU_TIMEOUT=${DEEPEP_NUM_GPU_TIMEOUT_CYCLES_IN_BILLIONS}G cycles"

# --- ★好卡 GID 自检（GID_CHECK=0 可跳过）---
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

# --- 若已在跑，先彻底清理（残留进程会导致 Prometheus 指标重复注册）---
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
EXTRA_ARGS=()
if [ "${ROLE}" = "prefill" ]; then
    EXTRA_ARGS=(--chunked-prefill-size "${CHUNKED_PREFILL:-4096}")
fi

# --- ★HiSparse 参数（仅 decode 生效）---
HISPARSE_ARGS=()
HISPARSE_DESC="off"
if [ "${ROLE}" = "decode" ] && [ "${HISPARSE}" = "on" ]; then
    if [ -n "${HISPARSE_CONFIG}" ]; then
        HS_CFG="${HISPARSE_CONFIG}"
    else
        # device_buffer_size 默认 2*top_k（代码默认），且必须 >= top_k
        if [ -z "${HISPARSE_DEV_BUF}" ]; then
            HISPARSE_DEV_BUF=$((HISPARSE_TOPK * 2))
        fi
        if [ "${HISPARSE_DEV_BUF}" -lt "${HISPARSE_TOPK}" ]; then
            echo "[err] HISPARSE_DEV_BUF(${HISPARSE_DEV_BUF}) 必须 >= HISPARSE_TOPK(${HISPARSE_TOPK})"
            exit 1
        fi
        HS_CFG="{\"top_k\":${HISPARSE_TOPK},\"device_buffer_size\":${HISPARSE_DEV_BUF},\"host_to_device_ratio\":${HISPARSE_H2D}}"
    fi
    HISPARSE_ARGS=(--enable-hisparse --hisparse-config "${HS_CFG}")
    HISPARSE_DESC="on cfg=${HS_CFG}"
fi

# --- NSA backend 显式指定（做 hisparse A/B 时用于对齐基线）---
NSA_ARGS=()
if [ -n "${NSA_DECODE_BACKEND:-}" ]; then
    NSA_ARGS+=(--nsa-decode-backend "${NSA_DECODE_BACKEND}")
fi
if [ -n "${NSA_PREFILL_BACKEND:-}" ]; then
    NSA_ARGS+=(--nsa-prefill-backend "${NSA_PREFILL_BACKEND}")
fi

echo "[start] role=${ROLE} node_rank=${NODE_RANK} dist-init=${HEAD}:5000 port=${PORT} -> ${LOG}"
echo "[start] mem-fraction=${MEM_FRACTION} hicache=${HICACHE}(ratio=${HICACHE_RATIO})"
echo "[start] pd: backend=${XFER_BACKEND} ib=${IB_DEV} bootstrap=${BOOTSTRAP_PORT}"
if [ "${ROLE}" = "prefill" ]; then
    echo "[start] chunked-prefill-size=${CHUNKED_PREFILL:-4096}"
else
    echo "[start] ★hisparse=${HISPARSE_DESC}"
fi
if [ ${#NSA_ARGS[@]} -gt 0 ]; then
    echo "[start] nsa: ${NSA_ARGS[*]}"
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
    "${HISPARSE_ARGS[@]}" \
    "${NSA_ARGS[@]}" \
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
echo "       说明 JIT 冷编译超过了 DeepEP 等待窗口，治本是预编译 DeepGEMM。"
if [ "${ROLE}" = "decode" ]; then
    echo ""
    echo "[hint] ★decode hisparse profile 实验（本目录主体）："
    echo "       实验 B(默认): bash \$0 decode ${NODE_RANK}                              # 已开 hisparse"
    echo "       基线 A      : HISPARSE=off NSA_DECODE_BACKEND=flashmla_sparse bash \$0 decode ${NODE_RANK}"
    echo "       ★A 组必须显式对齐 backend，否则测到的是 backend 差异而非 swap 开销"
    echo "       ★profile 打 Decode head(30001)；关注 TPOT；trace 关键字："
    echo "         swap-in: load_cache_to_device_buffer_mla / swap_in_selected_pages"
    echo "         EP     : internode::dispatch / combine / low_latency / notify_dispatch"
    echo "       ★压测端 input_len 必须 > device_buffer_size 才能观测到 swap-in"
fi
