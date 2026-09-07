#!/usr/bin/env bash
# P0（Prefill head）上的一键总控（PD 分离版）：
#   等 Prefill / Decode / Router 三方就绪 -> 参数与容量确认 -> 冒烟验 KV 链路 -> 跑 KV load 实验(+可选 profile)。
#
# 前提：
#   1) 四台已各自 bash start_server_pd.sh <role> <node_rank>；
#   2) P0 已 bash start_router.sh。
#
# 用法：
#   bash run_all_pd.sh                                   # 只验证 metrics（快）
#   bash run_all_pd.sh --profile                          # 抓 1 轮 profile（P 侧）
#   bash run_all_pd.sh --profile --repeat 5               # 连抓 5 轮
#   bash run_all_pd.sh --prefix 16000 --new-tokens 19556 --flush-tokens 700000   # ~45% 命中
#   KV_POOL_GB=5.14 KV_POOL_SLOT=55040 bash run_all_pd.sh --profile             # 校准 H2D 换算
#
# 与共置版 run_all.sh 的三处差异：
#   1) 要等三方（P head / D head / Router）而不是一方；
#   2) 冒烟推理打 Router(8000)，这样才真正验证 P->D KV 链路；参数确认读 prefill.log；
#   3) 用 SGL_CHAT_URL / SGL_SERVER_URL 把「发请求地址」与「抓指标地址」解耦后传给 python。

set -uo pipefail

P_HEAD="${P_HEAD:-29.209.115.238}"
D_HEAD="${D_HEAD:-29.209.104.16}"
P_PORT="${P_PORT:-30000}"
D_PORT="${D_PORT:-30001}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
PLOG="${PLOG:-/tmp/prefill.log}"
DLOG="${DLOG:-/tmp/decode.log}"
RLOG="${RLOG:-/tmp/router.log}"
AUTO="${AUTO:-/vllm-workspace/sglang-run/kv_profile_auto.py}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # 最长等 30 分钟（含 DeepGEMM warmup）

cd /vllm-workspace && source .sglang_venv/bin/activate
mkdir -p /tmp/sgprof

wait_health () {   # $1=名字  $2=base_url
    local name="$1" url="$2" t0 now code
    t0=$(date +%s)
    while true; do
        code=$(curl -s -o /dev/null -w "%{http_code}" "${url}/health" 2>/dev/null || echo 000)
        if [ "${code}" = "200" ]; then
            echo "[wait] ${name} ready (200)  用时 $(( $(date +%s) - t0 ))s"; return 0
        fi
        now=$(date +%s)
        if [ $((now - t0)) -gt "${READY_TIMEOUT}" ]; then
            echo "[wait] ${name} 超时未就绪 (last http=${code})"; return 1
        fi
        sleep 5
    done
}

# ---------- 1) 等三方就绪（顺序：P -> D -> Router）----------
echo "[wait] 等待 Prefill / Decode / Router 就绪（各最长 ${READY_TIMEOUT}s）..."
wait_health "prefill" "http://${P_HEAD}:${P_PORT}"      || { echo "--- ${PLOG} ---"; tail -n 30 "${PLOG}" 2>/dev/null; exit 1; }
wait_health "decode"  "http://${D_HEAD}:${D_PORT}"      || { echo "--- ${DLOG} ---"; tail -n 30 "${DLOG}" 2>/dev/null; exit 1; }
wait_health "router"  "http://127.0.0.1:${ROUTER_PORT}" || { echo "--- ${RLOG} ---"; tail -n 30 "${RLOG}" 2>/dev/null; exit 1; }

# ---------- 2) 关键参数确认（metrics / hicache / PD 三件套必须开）----------
echo "[check] Prefill 生效参数："
grep -oE 'enable_metrics=(True|False)|enable_cache_report=(True|False)|enable_request_time_stats_logging=(True|False)|enable_hierarchical_cache=(True|False)|hicache_io_backend=.?[a-z]+.?|chunked_prefill_size=[0-9]+|disaggregation_mode=.?[a-z]+.?|disaggregation_transfer_backend=.?[a-z]+.?|page_size=[0-9]+' \
    "${PLOG}" | sort -u | sed 's/^/  /' | head -20

# PD 分离特有：确认 KV 传输没有退化成 socket（退化后 TTFT 数据完全不可用）
if grep -qiE 'fallback.*socket|socket.*fallback|transfer backend.*socket' "${PLOG}" 2>/dev/null; then
    echo "  [warn] ★prefill.log 中出现 socket 退化迹象，请检查 --disaggregation-ib-device 名称"
fi

echo "[check] Prefill L1 KV pool（用于 H2D 换算，请与 KV_POOL_GB / KV_POOL_SLOT 对齐）："
grep -oE 'KV Cache is allocated. #tokens: [0-9]+, KV size: [0-9.]+ GB' "${PLOG}" | sort -u | sed 's/^/  /' | head -3
echo "  当前换算常数: KV_POOL_GB=${KV_POOL_GB:-5.14}  KV_POOL_SLOT=${KV_POOL_SLOT:-55040}"

# ---------- 3) 冒烟推理：必须打 Router，验证 P->D KV 链路真的通 ----------
echo "[check] 冒烟推理（走 Router，验证 PD KV 链路）："
curl -s "http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/GLM-5-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}' \
  -o /tmp/smoke.json -w "  chat http=%{http_code}  total=%{time_total}s\n"
echo -n "  resp: "; head -c 200 /tmp/smoke.json; echo
if ! grep -q '"content"' /tmp/smoke.json 2>/dev/null; then
    echo "  [err] 冒烟未拿到正常回复 => P->D KV 链路可能没通，先查："
    echo "        grep -iE 'bootstrap|WAITING_TIMEOUT|mooncake|transfer' ${DLOG} | tail -30"
    echo "        grep -iE 'bootstrap|socket|fallback|mooncake'          ${PLOG} | tail -30"
    exit 1
fi

# ---------- 4) NIC 计数器基线（可选，用于 §2.1.4 的 P->D 传输量交叉验证）----------
if [ "${NIC_COUNTER:-1}" = "1" ]; then
    echo "[check] NIC 计数器基线（单位 4B lane，×4 得字节）："
    for d in ${NIC_DEVS:-mlx5_bond_1 mlx5_bond_8}; do
        x=$(cat "/sys/class/infiniband/$d/ports/1/counters/port_xmit_data" 2>/dev/null || echo n/a)
        r=$(cat "/sys/class/infiniband/$d/ports/1/counters/port_rcv_data"  2>/dev/null || echo n/a)
        echo "  ${d}: xmit=${x} rcv=${r}"
    done
fi

# ---------- 5) 确认自动化脚本在位 ----------
if [ ! -f "${AUTO}" ]; then
    echo "[err] 未找到 ${AUTO}"
    echo "      请把 PD 分离目录下的 kv_profile_auto.py 放到 /vllm-workspace/sglang-run/"
    exit 1
fi

# ---------- 6) 跑实验：把本脚本之后的参数原样透传 ----------
ARGS="$*"
case "${ARGS}" in
  *--tag*) : ;;
  *) ARGS="--tag pd-$(date +%H%M%S) ${ARGS}" ;;
esac

# ★关键：请求走 Router(8000)，metrics / start_profile 走 Prefill head(30000)
export SGL_CHAT_URL="http://127.0.0.1:${ROUTER_PORT}"
export SGL_SERVER_URL="http://${P_HEAD}:${P_PORT}"
export KV_POOL_GB="${KV_POOL_GB:-5.14}"
export KV_POOL_SLOT="${KV_POOL_SLOT:-55040}"

echo "[run] SGL_CHAT_URL=${SGL_CHAT_URL}   SGL_SERVER_URL=${SGL_SERVER_URL}"
echo "[run] python ${AUTO} ${ARGS}"
python "${AUTO}" ${ARGS}

# ---------- 7) NIC 计数器收尾（与 §4 基线做差）----------
if [ "${NIC_COUNTER:-1}" = "1" ]; then
    echo ""
    echo "[check] NIC 计数器收尾（与上面基线做差 ×4 得字节）："
    for d in ${NIC_DEVS:-mlx5_bond_1 mlx5_bond_8}; do
        x=$(cat "/sys/class/infiniband/$d/ports/1/counters/port_xmit_data" 2>/dev/null || echo n/a)
        r=$(cat "/sys/class/infiniband/$d/ports/1/counters/port_rcv_data"  2>/dev/null || echo n/a)
        echo "  ${d}: xmit=${x} rcv=${r}"
    done
fi
