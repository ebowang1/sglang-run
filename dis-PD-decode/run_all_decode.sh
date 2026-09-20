#!/usr/bin/env bash
# P0（Prefill head）上的一键总控 —— ★decode/hisparse profile 版。
#   等 P/D/Router 三方就绪 -> 确认 hisparse 生效 -> 冒烟验 KV 链路 -> 跑 decode 并发压测(+可选 profile)。
#
# 前提：
#   1) 四台已各自 bash ../dis-PD-prefill/start_server_pd.sh <role> <node_rank>；
#      D 侧要测 hisparse 则加 HISPARSE=on（启动脚本已统一到 dis-PD-prefill 目录）；
#   2) P0 已 bash start_router.sh。
#
# ★与 dis-PD-prefill/run_all_pd.sh 的差异：
#   1) 参数确认 grep 的是 enable_hisparse / nsa_decode_backend（而非 hicache），确认 hisparse 真开了；
#   2) SGL_SERVER_URL 指向 ★Decode head(30001)，让 metrics/profile 落 decode 侧；请求仍走 Router；
#   3) 去掉 KV load 的 flush/命中率校准，改为透传 decode 压测参数（batch/max-tokens/input-len）；
#   4) 保留 NIC 计数器基线（EP 通信量交叉验证仍有用）。
#
# 用法：
#   bash run_all_decode.sh                                          # 默认单组 batch=64 +（无profile，仅metrics）
#   bash run_all_decode.sh --profile                               # 单组 batch=64 抓 profile
#   bash run_all_decode.sh --profile --batch 32                    # 指定单组 batch
#   bash run_all_decode.sh --profile --batch 8,16,32,64,128        # 扫 batch
#   bash run_all_decode.sh --profile --batch 64 --max-tokens 512 --input-len 16384
#   # A/B：off 组需在 D 侧用 HISPARSE=off NSA_DECODE_BACKEND=flashmla_sparse 重启后再跑，用 --tag 区分：
#   bash run_all_decode.sh --profile --batch 64 --tag hs-on
#   bash run_all_decode.sh --profile --batch 64 --tag hs-off

set -uo pipefail

P_HEAD="${P_HEAD:-29.209.115.238}"
D_HEAD="${D_HEAD:-29.209.104.16}"
P_PORT="${P_PORT:-30000}"
D_PORT="${D_PORT:-30001}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
PLOG="${PLOG:-/tmp/prefill.log}"
DLOG="${DLOG:-/tmp/decode.log}"
RLOG="${RLOG:-/tmp/router.log}"
AUTO="${AUTO:-/vllm-workspace/sglang-run/dis-PD-decode/decode_ep_profile.py}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # 最长等 30 分钟（含 DeepGEMM warmup）

cd /vllm-workspace && source .sglang_venv/bin/activate
mkdir -p /tmp/sgprof_decode

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

# ---------- 2) ★关键参数确认：hisparse 是否真的在 decode 侧生效 ----------
echo "[check] Decode 生效参数（确认 hisparse 开启 + backend 对齐）："
grep -oE 'enable_hisparse=(True|False)|hisparse_config=[^ ]+|nsa_decode_backend=.?[a-z_]+.?|attention_backend=.?[a-z]+.?|disaggregation_mode=.?[a-z]+.?|page_size=[0-9]+|max_running_requests=[0-9]+' \
    "${DLOG}" | sort -u | sed 's/^/  /' | head -20

if grep -qE 'enable_hisparse=True' "${DLOG}" 2>/dev/null; then
    echo "  [ok] ★hisparse 已在 decode 侧启用"
else
    echo "  [warn] ★未在 ${DLOG} 检测到 enable_hisparse=True！"
    echo "         若你是想测 hisparse ON，请确认 D 侧用 HISPARSE=on ../dis-PD-prefill/start_server_pd.sh 启动。"
    echo "         若你是在跑 A/B 的 OFF 基线组，请忽略本警告。"
fi

# 确认 decode 侧 backend（开 hisparse 会强制 flashmla_sparse）
grep -oE 'flashmla_sparse' "${DLOG}" 2>/dev/null | head -1 | sed 's/^/  nsa backend: /' || true

# PD 分离特有：确认 KV 传输没有退化成 socket
if grep -qiE 'fallback.*socket|socket.*fallback|transfer backend.*socket' "${DLOG}" "${PLOG}" 2>/dev/null; then
    echo "  [warn] ★日志出现 socket 退化迹象，请检查 --disaggregation-ib-device 名称"
fi

# ---------- 3) 冒烟推理：走 Router，验证 P->D KV 链路真的通 ----------
echo "[check] 冒烟推理（走 Router，验证 PD KV 链路 + decode 出 token）："
curl -s "http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/GLM-5-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0}' \
  -o /tmp/smoke_decode.json -w "  chat http=%{http_code}  total=%{time_total}s\n"
echo -n "  resp: "; head -c 200 /tmp/smoke_decode.json; echo
if ! grep -q '"content"' /tmp/smoke_decode.json 2>/dev/null; then
    echo "  [err] 冒烟未拿到正常回复 => P->D KV 链路可能没通，先查："
    echo "        grep -iE 'bootstrap|WAITING_TIMEOUT|mooncake|transfer' ${DLOG} | tail -30"
    echo "        grep -iE 'bootstrap|socket|fallback|mooncake'          ${PLOG} | tail -30"
    exit 1
fi

# ---------- 4) NIC 计数器基线（EP 通信量交叉验证；NIC_COUNTER=0 可关）----------
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
    echo "      请把 decode_ep_profile.py 放到 /vllm-workspace/sglang-run/dis-PD-decode/"
    exit 1
fi

# ---------- 6) 跑实验：把本脚本之后的参数原样透传 ----------
ARGS="$*"
case "${ARGS}" in
  *--tag*) : ;;
  *) ARGS="--tag dec-$(date +%H%M%S) ${ARGS}" ;;
esac

# ★关键：请求走 Router(8000)，metrics/profile 走 ★Decode head(30001)
export SGL_CHAT_URL="http://127.0.0.1:${ROUTER_PORT}"
export SGL_SERVER_URL="http://${D_HEAD}:${D_PORT}"

echo "[run] SGL_CHAT_URL=${SGL_CHAT_URL}   SGL_SERVER_URL=${SGL_SERVER_URL} (★Decode head)"
echo "[run] python ${AUTO} ${ARGS}"
python "${AUTO}" ${ARGS}

# ---------- 7) NIC 计数器收尾（与 §4 基线做差）----------
if [ "${NIC_COUNTER:-1}" = "1" ]; then
    echo ""
    echo "[check] NIC 计数器收尾（与上面基线做差 ×4 得字节，估算本轮 EP 通信量）："
    for d in ${NIC_DEVS:-mlx5_bond_1 mlx5_bond_8}; do
        x=$(cat "/sys/class/infiniband/$d/ports/1/counters/port_xmit_data" 2>/dev/null || echo n/a)
        r=$(cat "/sys/class/infiniband/$d/ports/1/counters/port_rcv_data"  2>/dev/null || echo n/a)
        echo "  ${d}: xmit=${x} rcv=${r}"
    done
fi
