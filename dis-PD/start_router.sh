#!/usr/bin/env bash
# 起 PD Router（mini-lb），只在 P0（Prefill head）执行。
#
# 用法：
#   bash start_router.sh
#   P_HEAD=29.209.115.238 D_HEAD=29.209.104.16 bash start_router.sh
#
# 说明：
#   - Router 只对接两组的 head（P0:30000 / D0:30001），组内 worker 不暴露给 Router；
#   - Router 是唯一对外入口：所有实验请求必须打 :8000，
#     打 30000 只会走 prefill、拿不到完整 TTFT，且 KV 会堆在 transfer queue；
#   - 前提：Prefill 组与 Decode 组均已 ready（否则 Router 起来也是 503）。

set -uo pipefail

P_HEAD="${P_HEAD:-29.209.115.238}"
D_HEAD="${D_HEAD:-29.209.104.16}"
P_PORT="${P_PORT:-30000}"
D_PORT="${D_PORT:-30001}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
LOG="${LOG:-/tmp/router.log}"

unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY all_proxy ALL_PROXY || true
cd /vllm-workspace && source .sglang_venv/bin/activate

# 清理旧 Router
pkill -f "launch_sglang.py router" 2>/dev/null || true
pkill -f "mini_lb"                 2>/dev/null || true
sleep 1

echo "[router] prefill=http://${P_HEAD}:${P_PORT}  decode=http://${D_HEAD}:${D_PORT}  -> :${ROUTER_PORT}"

nohup python launch_sglang.py router \
    --prefill "http://${P_HEAD}:${P_PORT}" \
    --decode  "http://${D_HEAD}:${D_PORT}" \
    --host 0.0.0.0 --port "${ROUTER_PORT}" --mini-lb \
    --run > "${LOG}" 2>&1 &
sleep 5

# launch_sglang.py 的 router 在部分 sglang-router 版本会报
# --mini-lb / --pd-disaggregation 参数冲突，此时回退原生 mini_lb。
code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${ROUTER_PORT}/health" 2>/dev/null || echo 000)
if [ "${code}" != "200" ]; then
    echo "[router] launch_sglang.py router 未就绪(http=${code})，回退原生 mini_lb"
    tail -n 15 "${LOG}" 2>/dev/null
    pkill -f "launch_sglang.py router" 2>/dev/null || true
    sleep 1
    nohup python -m sglang.srt.disaggregation.mini_lb \
        --prefill "http://${P_HEAD}:${P_PORT}" \
        --decode  "http://${D_HEAD}:${D_PORT}" \
        --host 0.0.0.0 --port "${ROUTER_PORT}" >> "${LOG}" 2>&1 &
    sleep 5
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${ROUTER_PORT}/health" 2>/dev/null || echo 000)
fi

echo "[router] health http=${code}   tail -f ${LOG}"
[ "${code}" = "200" ] || { echo "[router] 仍未就绪，请检查 P/D 两组是否 ready"; exit 1; }
