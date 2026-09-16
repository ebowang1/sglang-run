#!/usr/bin/env python3
"""
decode 侧 hisparse profile 实验脚本（PD 分离版）—— 抓 EP 通信与 swap-in/swap-out 冲突。

★与 dis-PD-prefill/kv_profile_auto.py 的本质区别（后者测的是 Prefill 侧 KV load）：
  ┌────────────┬──────────────────────────┬────────────────────────────────┐
  │ 维度       │ 旧 kv_profile_auto        │ 本脚本 decode_ep_profile        │
  ├────────────┼──────────────────────────┼────────────────────────────────┤
  │ 发请求     │ 串行（decode batch 恒=1） │ ★并发线程池（压高 decode batch）│
  │ max_tokens │ 1（几乎无 decode）        │ 可配，默认 256（覆盖多 step）   │
  │ profile位置│ Prefill head(30000)       │ ★Decode head(30001)            │
  │ 圈窗口     │ round2 前几条 hit(H2D时刻)│ ★稳态 decode 阶段(warmup后)    │
  │ 判定指标   │ cached/load_back(HiCache) │ ★TPOT / decode 吞吐            │
  │ trace 关键字│ KV load / P->D           │ ★EP dispatch/combine + swap-in │
  └────────────┴──────────────────────────┴────────────────────────────────┘

为什么要并发压 batch：
  EP low-latency 的 dispatch/combine 通信量随 batch 线性涨；hisparse swap-in 也随
  batch×层数(78) 涨，两者都在 decode 关键路径（默认流）上。batch=1 时两边都很小，
  timeline 很干净、看不到冲突。必须让 N 条请求【同时在飞】把 decode batch 顶起来。

为什么 input_len 必须 > device_buffer_size：
  hisparse 命中规律：seq_len <= device_buffer_size 全部预载(几乎无 swap-in)；
  超过才每次 top-k miss → host load。device_buffer_size 默认 2*top_k=4096，
  所以 --input-len 默认给 8192（>4096），确保能观测到 swap-in 流量。

profile 窗口的圈法（关键）：
  decode batch 需要时间填满（请求陆续到达 D 侧、prefill 完成后才进 decode）。
  所以先并发发一批请求做 warmup，sleep --warmup-sec 等 batch 稳定在高位，
  再 POST /start_profile，采 --profile-sec 秒，再 /stop_profile。
  这样窗口精确落在【高 batch decode 稳态】上。

默认【只测一组】（单 batch、hisparse on）。多组是可选：
  - 扫 batch    : --batch 8,16,32,64,128   （传多个值才扫，单值就是单组）
  - A/B 对照    : 需要在【服务端】切 hisparse on/off 分别重启，本脚本只管发压/采集；
                  用 --tag 区分两次采集的 trace 即可（on 组 tag=hs-on，off 组 tag=hs-off）。

用法：
  # 由 run_all_decode.sh 调用时会自动注入 SGL_CHAT_URL / SGL_SERVER_URL
  python decode_ep_profile.py --tag dec-001 --batch 64 --profile           # 默认单组
  python decode_ep_profile.py --tag dec-scan --batch 8,16,32,64 --profile  # 扫 batch
  python decode_ep_profile.py --tag dec-long --batch 64 --max-tokens 512 --input-len 16384 --profile

  # 手动指定地址（不经 run_all_decode.sh）：请求走 Router，profile 打 Decode head
  SGL_CHAT_URL=http://127.0.0.1:8000 \
  SGL_SERVER_URL=http://29.209.104.16:30001 \
  python decode_ep_profile.py --tag dec-manual --batch 64 --profile
"""

import argparse
import json
import os
import random
import string
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ★请求走 Router(8000)；指标/profile 走 Decode head(30001)——与 prefill 版相反
CHAT_BASE = os.environ.get("SGL_CHAT_URL", "http://127.0.0.1:8000")       # /v1/chat/completions
URL_BASE = os.environ.get("SGL_SERVER_URL", "http://127.0.0.1:30001")     # ★Decode head: metrics/profile
CHAT_URL = CHAT_BASE + "/v1/chat/completions"
METRICS_URL = URL_BASE + "/metrics"
MODEL = os.environ.get("SGL_MODEL", "/data/models/GLM-5-FP8")

# decode 侧关心的指标（TPOT / 运行中请求数 / 生成吞吐）
METRIC_KEYS = (
    "running_reqs",                       # 当前在跑请求数≈decode batch
    "gen_throughput",                     # 生成吞吐 token/s
    "num_running_reqs",
    "inter_token_latency_seconds",        # ITL/TPOT 相关
    "e2e_request_latency_seconds",
    "decode",                             # 兜底匹配 decode 相关行
)


def rand_tag(n=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_prompt(input_len, uniq):
    """构造 input_len 个 token 的输入。
    用【单字符词】保证近似一词一 token（GLM tokenizer 对字母数字混合串会拆碎）。
    uniq 放最前，保证各请求前缀不同、不互相命中缓存（每条都真正走 decode）。"""
    return f"{uniq} " + " ".join(["a"] * max(1, input_len))


def post_chat(content, max_tokens, timeout=1800):
    """发一条 chat（阻塞）。返回 (elapsed_ms, http_code, n_completion_tokens)。"""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,            # ★强制生成满 max_tokens，保证 decode 步数一致
    }).encode()
    req = urllib.request.Request(
        CHAT_URL, data=body, headers={"Content-Type": "application/json"}
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        return (time.time() - t) * 1000.0, e.code, 0
    except Exception as e:
        return (time.time() - t) * 1000.0, -1, 0
    n_tok = 0
    try:
        j = json.loads(data)
        n_tok = j.get("usage", {}).get("completion_tokens", 0)
    except Exception:
        pass
    return (time.time() - t) * 1000.0, code, n_tok


def scrape_metrics():
    """抓 Decode head 的 /metrics，只保留关心的行。"""
    out = {}
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=30) as r:
            text = r.read().decode()
    except Exception as e:
        print(f"[warn] 抓 metrics 失败({METRICS_URL}): {e}")
        return out
    for line in text.splitlines():
        if not line.startswith("sglang:"):
            continue
        if not any(k in line for k in METRIC_KEYS):
            continue
        try:
            key, val = line.rsplit(" ", 1)
            out[key] = float(val)
        except ValueError:
            continue
    return out


def show_metrics_snapshot(title):
    snap = scrape_metrics()
    print(f"  ---- {title} ----")
    # 只挑几个最有代表性的打印
    for k in sorted(snap.keys()):
        if any(x in k for x in ("running_reqs", "gen_throughput")):
            print(f"    {k} = {snap[k]:.2f}")
    return snap


def post_json(path, payload):
    """向 Decode head POST 一个 json。返回 (status, text)。"""
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    req = urllib.request.Request(
        URL_BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:
        return -1, str(e)


def start_profile(output_dir, prefix):
    """手动模式起 profile（不带 num_steps，由后续 stop_profile 停）。
    ★打到 Decode head，采 CPU+GPU，覆盖 EP kernel 与 swap-in kernel。"""
    payload = {
        "output_dir": output_dir,
        "activities": ["CPU", "GPU"],
        "profile_prefix": prefix,
    }
    code, msg = post_json("/start_profile", payload)
    print(f"[profiler] start_profile(D head) -> {code} {msg.strip()[:120]}")
    return code == 200


def stop_profile():
    code, msg = post_json("/stop_profile", {})
    print(f"[profiler] stop_profile(D head)  -> {code} {msg.strip()[:120]}")
    return code == 200


def run_batch_group(args, batch, run_tag):
    """跑一个 batch 组：并发压出 decode 稳态，圈 profile 窗口。"""
    print(f"\n########## TAG={run_tag} BATCH={batch} "
          f"MAX_TOKENS={args.max_tokens} INPUT_LEN={args.input_len} "
          f"PROFILE={args.profile} ##########")
    print(f"  [PD] chat -> {CHAT_URL}")
    print(f"  [PD] metrics/profile -> {URL_BASE}  (★Decode head)")
    if args.input_len <= args.device_buffer_size:
        print(f"  [warn] ★input_len({args.input_len}) <= device_buffer_size"
              f"({args.device_buffer_size})：seq 未超 buffer，几乎不会触发 swap-in，"
              f"trace 里可能看不到 swap 开销。建议 --input-len > {args.device_buffer_size}")

    # 并发线程池：持续维持 batch 条在飞。为了让 decode 稳态足够长，
    # 每条请求 max_tokens 拉满；总提交条数 = batch * request_multiple，
    # 保证采集窗口内一直有请求在 decode（前面的还没结束，后面的补上）。
    n_total = batch * max(1, args.request_multiple)
    stop_flag = threading.Event()
    results = []
    results_lock = threading.Lock()

    def worker(idx):
        if stop_flag.is_set():
            return
        prompt = make_prompt(args.input_len, f"{run_tag}-{idx}-{rand_tag()}")
        dt, code, ntok = post_chat(prompt, args.max_tokens)
        with results_lock:
            results.append((dt, code, ntok))

    pool = ThreadPoolExecutor(max_workers=batch)

    # ---- 阶段1：warmup —— 先并发打满，等 decode batch 稳定在高位 ----
    print(f"== warmup: 并发提交 {n_total} 条 (并发度={batch})，等 batch 填满 ==")
    futures = [pool.submit(worker, i) for i in range(n_total)]
    # 轮询 running_reqs，等它爬到接近 batch 或超时
    t0 = time.time()
    while time.time() - t0 < args.warmup_sec:
        snap = scrape_metrics()
        running = max([v for k, v in snap.items() if "running_reqs" in k] + [0])
        if running >= batch * args.warmup_ratio:
            print(f"  [warmup] running_reqs={running:.0f} 已达 {args.warmup_ratio:.0%}×batch，"
                  f"用时 {time.time()-t0:.1f}s")
            break
        time.sleep(1.0)
    else:
        snap = scrape_metrics()
        running = max([v for k, v in snap.items() if "running_reqs" in k] + [0])
        print(f"  [warmup] 到时 {args.warmup_sec}s，running_reqs={running:.0f}"
              f"（可能 batch 未完全填满，可增大 --request-multiple 或 --warmup-sec）")

    # ---- 阶段2：稳态采集 —— start_profile -> 采 profile_sec 秒 -> stop_profile ----
    before = show_metrics_snapshot("metrics before(采集窗口起点)")
    if args.profile:
        os.makedirs(args.output_dir, exist_ok=True)
        prof_prefix = f"{run_tag}-b{batch}"
        if not start_profile(args.output_dir, prof_prefix):
            print("[profiler] start 失败，跳过本组 profile")
        else:
            print(f"== profiling {args.profile_sec}s（稳态 decode，含 EP + swap-in）==")
            time.sleep(args.profile_sec)
            stop_profile()
    else:
        print(f"== 采集 {args.profile_sec}s metrics（未开 profile）==")
        time.sleep(args.profile_sec)
    after = show_metrics_snapshot("metrics after(采集窗口终点)")

    # ---- 阶段3：收尾，等剩余请求跑完（或直接不等，交给下一组前的清理）----
    print("  [drain] 等待在飞请求结束（最多 --drain-sec）...")
    stop_flag.set()   # 阻止尚未开始的 worker 再发新请求
    drain_t0 = time.time()
    done = 0
    for fu in as_completed(futures, timeout=None):
        done += 1
        if time.time() - drain_t0 > args.drain_sec:
            print(f"  [drain] 到时 {args.drain_sec}s，已完成 {done}/{len(futures)}，不再等待")
            break
    pool.shutdown(wait=False)

    # ---- 统计 ----
    ok = [r for r in results if r[1] == 200]
    if ok:
        lat = sorted(r[0] for r in ok)
        toks = [r[2] for r in ok if r[2] > 0]
        avg_tpot = None
        if toks:
            # 粗略 TPOT ≈ (端到端 - 常数TTFT) / tokens，这里只给端到端/生成token的粗值
            per = [r[0] / r[2] for r in ok if r[2] > 0]
            avg_tpot = sum(per) / len(per)
        print(f"  [result] 成功 {len(ok)}/{len(results)}  "
              f"e2e(ms) min={lat[0]:.0f} p50={lat[len(lat)//2]:.0f} max={lat[-1]:.0f}")
        if avg_tpot:
            print(f"  [result] 粗略 每token端到端 ≈ {avg_tpot:.1f} ms/tok "
                  f"(含TTFT，仅横向比 on/off 用，不是纯 TPOT)")
    else:
        print(f"  [result] ★无成功请求（{len(results)} 条全失败），检查 Router/D 侧是否 ready")

    # metrics 差分（gen_throughput 取窗口内均值参考）
    def pick(snap, key):
        return max([v for k, v in snap.items() if key in k] + [0])
    print(f"  [metrics] running_reqs: before={pick(before,'running_reqs'):.0f} "
          f"after={pick(after,'running_reqs'):.0f}")
    print(f"  [metrics] gen_throughput: before={pick(before,'gen_throughput'):.0f} "
          f"after={pick(after,'gen_throughput'):.0f} tok/s")

    if args.profile:
        print(f"\n[trace] ★Decode 组两台各看: "
              f"find {args.output_dir} -name '{prof_prefix}*.trace.json.gz'")
        print(f"        D0 是 TP 0-7，D1 是 TP 8-15，两台都要取")
        print(f"[trace] trace 中重点搜索：")
        print(f"        swap-in : load_cache_to_device_buffer_mla / swap_in_selected_pages")
        print(f"        swap-out: map_last_loc_to_buffer / _eager_backup_previous_token")
        print(f"        EP      : internode::dispatch / combine / notify_dispatch / low_latency")
        print(f"        ★看 EP kernel 与 swap-in kernel 是否在【同一条流/时间段】互相阻塞")
    return len(ok) > 0


def main():
    ap = argparse.ArgumentParser(
        description="decode 侧 hisparse profile 实验（并发压 batch + 稳态采集）")
    ap.add_argument("--tag", required=True, help="本轮标签（trace 前缀/多组区分）")

    # ★核心可配参数
    ap.add_argument("--batch", default="64",
                    help="目标并发数(=decode batch)。单值=单组(默认)；"
                         "逗号多值=扫描，如 8,16,32,64,128")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="每条请求 decode 长度（越长稳态越久，默认256）")
    ap.add_argument("--input-len", type=int, default=8192,
                    help="每条请求输入 token 数。★必须 > device_buffer_size 才触发 swap-in，"
                         "默认8192(>默认buffer 4096)")
    ap.add_argument("--device-buffer-size", type=int, default=4096,
                    help="仅用于告警提示：与服务端 hisparse device_buffer_size 对齐（默认2*top_k=4096）")

    # profile 窗口控制
    ap.add_argument("--profile", action="store_true", help="是否抓 profile（Decode head）")
    ap.add_argument("--output-dir", default="/tmp/sgprof_decode", help="trace 输出目录")
    ap.add_argument("--warmup-sec", type=float, default=30.0,
                    help="warmup 最长等待秒数（等 decode batch 填满）")
    ap.add_argument("--warmup-ratio", type=float, default=0.8,
                    help="running_reqs 达到 ratio×batch 视为填满，提前结束 warmup")
    ap.add_argument("--profile-sec", type=float, default=10.0,
                    help="稳态采集时长(秒)，profile 窗口=这段时间")
    ap.add_argument("--request-multiple", type=int, default=4,
                    help="总提交条数 = batch × 该值，保证采集窗口内持续有请求在 decode")
    ap.add_argument("--drain-sec", type=float, default=60.0,
                    help="采集后等在飞请求结束的最长秒数")

    # 多组
    ap.add_argument("--repeat", type=int, default=1, help="每个 batch 点重复轮数")
    ap.add_argument("--sleep-between", type=float, default=8.0,
                    help="组与组之间间隔(秒)，让上一组请求清空、batch 归零")
    args = ap.parse_args()

    batch_list = [int(x) for x in str(args.batch).split(",") if x.strip()]
    is_scan = len(batch_list) > 1
    print(f"[plan] batch 点: {batch_list}  {'(扫描模式)' if is_scan else '(单组模式)'}  "
          f"repeat={args.repeat}")

    results = []
    for bi, batch in enumerate(batch_list):
        for k in range(args.repeat):
            if len(batch_list) == 1 and args.repeat == 1:
                run_tag = args.tag
            else:
                run_tag = f"{args.tag}-b{batch}-r{k:02d}"
            ok = run_batch_group(args, batch, run_tag)
            results.append((run_tag, batch, ok))
            # 组间等待：让 decode batch 归零，避免串扰
            if not (bi == len(batch_list) - 1 and k == args.repeat - 1):
                print(f"  [gap] 组间等待 {args.sleep_between}s（等 batch 归零）...")
                time.sleep(args.sleep_between)

    print("\n==== 汇总 ====")
    for tag, batch, ok in results:
        print(f"  {tag} (batch={batch}): {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
