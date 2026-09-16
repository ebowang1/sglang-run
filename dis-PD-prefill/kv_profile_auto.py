#!/usr/bin/env python3
"""
KV load 一键实验脚本 —— **PD 分离版**（单进程自动编排，无需两个终端 / 手动 Enter）。

与共置版的唯一区别（其余逻辑、参数语义完全一致，便于两组实验直接对齐）：
  ★1) 「发请求地址」与「抓指标/抓 profile 地址」解耦：
        - 请求      -> Router (8000)      SGL_CHAT_URL
        - metrics   -> Prefill head(30000) SGL_SERVER_URL
        - start/stop_profile -> Prefill head(30000)
      PD 分离下把请求直接打 30000 只会走 prefill、拿不到完整 TTFT，
      且 KV 无人接收会堆在 transfer queue。
  ★2) H2D 换算常数（L1 KV pool 的 GB / slot 数）改为环境变量可覆盖：
        KV_POOL_GB / KV_POOL_SLOT
      PD 分离下 P 组的 #tokens 与共置很可能不同，必须按 /tmp/prefill.log 的
      "KV Cache is allocated. #tokens: N, KV size: S GB" 校准。

流程：
  1) round1 fill：用唯一前缀建立 KV（L1 + write-through 到 L2）
  2) flush     ：随机请求把目标前缀挤出 L1（L2 保留）
  3) metrics_before 快照
  4)（--profile 时）脚本内 POST /start_profile（手动模式，不带 num_steps）
  5) round2 hit：
       - profile 模式：只发前 --profile-hits 条（每个 DP 副本首命中，含 H2D），
         发完 settle 后 POST /stop_profile；其余 hit 在关闭 profile 后补发（仅用于 metrics）
       - 非 profile 模式：发满 --round-n 条
  6) metrics_after 快照 + 差分判定
  7) 打印本轮产物路径

为什么这样设计：
  - KV load 只在每个 DP 副本的第 1 条 round2 请求发生，所以 profile 只需圈前 dp 条；
  - 用脚本内 /start_profile + /stop_profile 精确圈定窗口，
    彻底弃用 `sglang.profiler --num-steps`（其"服务端自数 step"会与外部发请求节奏
    错位，曾导致 RemoteDisconnected 甚至 watchdog 杀实例）。

⚠️ PD 分离注意事项：
  - [fill]/[hit] 打印的时延是**端到端**（含 decode 首 token + P->D KV 传输），
    比共置版天然偏大。不要与共置版直接比；要比就比 /metrics 的 forward_duration。
  - P 组把 KV 传给 D 组后即释放 L1，前缀更快落 L2，因此 --flush-tokens 若沿用
    共置值可能"过度 flush"、把前缀连 L2 一起挤掉（表现为 Δcached_host=0）。
    PD 下建议从共置值的 1/2 起步逐步上调，找到 Δcached_host>0 且 Δload_back>0 的最小量。

用法：
  # 由 run_all_pd.sh 调用时会自动注入 SGL_CHAT_URL / SGL_SERVER_URL
  python kv_profile_auto.py --tag pd-001                 # 只 metrics（近100%命中）
  python kv_profile_auto.py --tag pd-prof-001 --profile  # metrics + profile(前4条)
  python kv_profile_auto.py --tag batch --profile --repeat 5
  # 控制命中率（命中率≈prefix/(prefix+new-tokens)）：45% 命中，验证计算/通信重叠
  python kv_profile_auto.py --tag hr45 --profile --prefix 16000 --new-tokens 19556

  # 手动指定地址（不经 run_all_pd.sh）
  SGL_CHAT_URL=http://127.0.0.1:8000 \
  SGL_SERVER_URL=http://29.209.115.238:30000 \
  python kv_profile_auto.py --tag pd-manual --profile
"""

import argparse
import json
import os
import random
import string
import time
import urllib.request
import urllib.error

# ★PD 分离：发请求走 Router(8000)，指标/profile 走 Prefill head(30000)
URL_BASE = os.environ.get("SGL_SERVER_URL", "http://127.0.0.1:30000")   # metrics / start_profile
CHAT_BASE = os.environ.get("SGL_CHAT_URL", "http://127.0.0.1:8000")     # /v1/chat/completions
CHAT_URL = CHAT_BASE + "/v1/chat/completions"
METRICS_URL = URL_BASE + "/metrics"
MODEL = os.environ.get("SGL_MODEL", "/data/models/GLM-5-FP8")

# ★PD 分离：L1 KV pool 换算常数，按 /tmp/prefill.log 校准
#   grep -oE 'KV Cache is allocated. #tokens: [0-9]+, KV size: [0-9.]+ GB' /tmp/prefill.log
KV_POOL_GB = float(os.environ.get("KV_POOL_GB", "5.14"))
KV_POOL_SLOT = float(os.environ.get("KV_POOL_SLOT", "55040"))

METRIC_KEYS = (
    "cached_tokens_total",
    "load_back_tokens_total",
    "load_back_duration_seconds_sum",
    "load_back_duration_seconds_count",
    "evicted_tokens_total",
    "hicache_host_used_tokens",
)


def post_chat(content, max_tokens=1, timeout=600):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        CHAT_URL, data=body, headers={"Content-Type": "application/json"}
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
    return (time.time() - t) * 1000.0, code


def scrape_metrics():
    """返回 dict：{full_metric_line_key: float_value}，只保留关心的指标行。
    ★PD 分离：抓的是 Prefill head 的 /metrics（KV load 只发生在 P 侧）。"""
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
        # 形如: sglang:xxx{labels...} 12345.0
        try:
            key, val = line.rsplit(" ", 1)
            out[key] = float(val)
        except ValueError:
            continue
    return out


def sum_by_metric(snapshot, metric_name, label_filter=None):
    """把某指标（可能有多 rank/多 label）的值求和。label_filter 是子串。"""
    total = 0.0
    for key, val in snapshot.items():
        if metric_name not in key:
            continue
        if label_filter and label_filter not in key:
            continue
        total += val
    return total


def diff_metrics(before, after, prefix=16000, new_tokens=0):
    print("\n==== metrics 差分（after - before）====")

    def show(name, label=None, tag=""):
        b = sum_by_metric(before, name, label)
        a = sum_by_metric(after, name, label)
        print(f"  {name}{tag}: {b:.0f} -> {a:.0f}   Δ={a-b:+.0f}")
        return a - b

    d_host = show("cached_tokens_total", 'cache_source="host"', ' (host)')
    d_dev = show("cached_tokens_total", 'cache_source="device"', ' (device)')
    d_load = show("load_back_tokens_total")
    show("load_back_duration_seconds_count")
    show("load_back_duration_seconds_sum")
    d_evict = show("evicted_tokens_total")
    d_hostused = show("hicache_host_used_tokens")

    print("\n==== 判定 ====")
    ok_host = d_host > 0
    ok_load = d_load > 0
    print(f"  host L2 命中 (Δcache_source=host>0): {'YES' if ok_host else 'NO'}")
    print(f"  发生 KV load (Δload_back_tokens>0) : {'YES' if ok_load else 'NO'}")
    if ok_host and ok_load:
        # 逻辑 H2D 估算：B_slot = KV_POOL_GB / KV_POOL_SLOT
        # （load_back 含主KV+indexer 两组件）
        gb = d_load * KV_POOL_GB / KV_POOL_SLOT
        print(f"  => 确认 KV load 有效。逻辑 H2D ≈ {gb:.2f} GB "
              f"(Δload_back_tokens={d_load:.0f} × {KV_POOL_GB}GB/{KV_POOL_SLOT:.0f})")
        # cached_host 才是"命中的前缀 token 数"；load_back 因主KV+indexer两组件通常约为其 2 倍
        n_seg = d_host / prefix if prefix else 0
        print(f"  => host 命中前缀段数 ≈ {n_seg:.1f}（Δcached_host={d_host:.0f} / prefix={prefix}）；"
              f"load_back={d_load:.0f}≈命中×2（主KV+indexer 两组件各计一次）")
        # 实际命中率：命中前缀 token / (命中 + 新算)
        if new_tokens > 0:
            print(f"  => 目标命中率≈{prefix/(prefix+new_tokens)*100:.0f}%（prefix/(prefix+new)）；"
                  f"新 token 会真正过 attention+MoE，可观察 KV load 与计算/EP 的重叠")
        # ★PD 分离新增：P->D KV 传输量估算（本请求完整 context 都要推给 D 组）
        b_slot_kb = KV_POOL_GB * 1024 * 1024 / KV_POOL_SLOT
        pd_gb = (prefix + new_tokens) * b_slot_kb / 1024 / 1024
        print(f"  => ★P->D KV 传输 ≈ {pd_gb:.2f} GB / 请求 / DP副本 "
              f"((prefix+new)={prefix+new_tokens} × {b_slot_kb:.1f}KB)；"
              f"注意：即使 prefix 100% 命中，KV 仍要完整过一遍 RDMA 给 D 组，"
              f"故 KV load 与 P->D 传输【同步放大】")
    else:
        print("  => 本轮未确认有效 KV load，profile（若抓了）可能没覆盖回载，"
              "建议检查 RUN_TAG 是否唯一 / flush 是否足够 / 采集时机。")
        # ★PD 分离特有误因：flush 过量把 L2 也挤空
        if d_evict > 0 and not ok_host:
            print("  => ★PD 分离常见误因：flush 过度。P 组 KV 传给 D 后即释放 L1，"
                  "前缀落 L2 更快，同样的 --flush-tokens 在 PD 下可能连 L2 一起挤掉。")
            print(f"     当前 hicache_host_used_tokens Δ={d_hostused:+.0f}"
                  f"（接近 0 或为负说明 L2 被挤空）")
            print("     对策：把 --flush-tokens 降到共置实验值的 1/2 起步，逐步上调，"
                  "找到 Δcached_host>0 且 Δload_back>0 的最小 flush 量。")
    return ok_host and ok_load


def post_json(path, payload):
    """向服务端 POST 一个 json，返回 (status, text)。
    ★PD 分离：走 URL_BASE = Prefill head，所以 /start_profile、/stop_profile
    自动打到 P 侧，符合"KV load 只在 P 侧"的预期。"""
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
    """手动模式起 profile：不带 num_steps，由脚本后续 stop_profile 停止。
    不传 profile_by_stage（by-stage 依赖 num_steps 自数停止，会与手动 stop 冲突）；
    round2 用 max_tokens=1，基本只有 prefill。"""
    payload = {
        "output_dir": output_dir,
        "activities": ["CPU", "GPU"],
        "profile_prefix": prefix,
    }
    code, msg = post_json("/start_profile", payload)
    print(f"[profiler] start_profile -> {code} {msg.strip()[:120]}")
    return code == 200


def stop_profile():
    code, msg = post_json("/stop_profile", {})
    print(f"[profiler] stop_profile  -> {code} {msg.strip()[:120]}")
    return code == 200


def make_prefix(run_tag, prefix_tokens):
    # RUN_TAG 放最前，保证每轮冷填充
    return f"{run_tag} " + " ".join(["a"] * prefix_tokens)


def make_hit_body(prefix, run_tag, i, new_tokens):
    """hit 请求 = 共享前缀 + 新 token 段。
    new_tokens=0 时退化为原来的近 100% 命中（只加短 tag）。
    新 token 必须用【单字符】词（如 'b'），保证近似一词一 token；
    若用 'u<tag>i<idx>w<j>' 这类字母数字混合串，会被 GLM tokenizer 拆碎，
    例如 19556 词会膨胀到 238434 token、超过 context length 202752。
    新 token 段各请求相同也不影响：前缀命中(触发KV load) + 新token重算(触发计算/EP)
    才是目的；profile 只圈前 4 条，各 DP 副本首命中独立。"""
    if new_tokens <= 0:
        return prefix + f" QUESTION-hit-{i}"
    tail = " ".join(["b"] * new_tokens)
    return prefix + " " + tail


def run_once(args, run_tag):
    prefix = make_prefix(run_tag, args.prefix)
    total = args.prefix + args.new_tokens
    exp_hit = args.prefix / total if total else 0
    flush_desc = (f"FLUSH_TOKENS={args.flush_tokens}(长请求×{args.flush_req_len})"
                  if args.flush_tokens > 0 else f"FLUSH_N={args.flush_n}(旧模式×2048)")
    print(f"\n########## TAG={run_tag} PREFIX={args.prefix} NEW={args.new_tokens} "
          f"(总长≈{total}, 预期命中率≈{exp_hit*100:.0f}%) "
          f"ROUND_N={args.round_n} {flush_desc} "
          f"PROFILE={args.profile} PROFILE_HITS={args.profile_hits} ##########")
    print(f"  [PD] chat -> {CHAT_URL}")
    print(f"  [PD] metrics/profile -> {URL_BASE}")

    # ---- round1: fill（也带新 token，保证与 hit 前缀一致地建立 L1/L2）----
    print("== round1: fill cache ==")
    fill = []
    for i in range(args.round_n):
        dt, _ = post_chat(prefix + f" QUESTION-fill-{i}")
        fill.append(dt)
    fill.sort()
    print(f"  [fill] n={len(fill)} min={fill[0]:.1f} "
          f"median={fill[len(fill)//2]:.1f} max={fill[-1]:.1f} ms")
    print("  [PD] 注意：以上时延为端到端（含 decode 首 token + P->D KV 传输），"
          "不可与共置版直接比")

    # ---- flush ----
    # flush 目的：往 L1 灌入足够多的新 KV，用 LRU 把目标前缀挤出 L1（L2 保留）。
    # 关键是【灌入的总 token 量】要 > L1 容量，而不是请求条数。
    # 优化：用【少量长请求】(每条 flush_req_len)替代原来的【多量短请求】(520×2048)，
    #      固定开销(网络/调度/tokenize)从 O(条数) 降下来，flush 明显更快。
    # 每条 flush 请求加唯一随机 tag 放最前，保证彼此不命中前缀（各占新 L1）。
    # ★PD 分离：P 组 KV 传完即释放 L1，前缀落 L2 更快，flush 过量会连 L2 一起挤掉。
    #   本值需重新标定，不能直接复用共置实验的 --flush-tokens。
    if args.flush_tokens > 0:
        req_len = max(1, args.flush_req_len)
        n_flush = (args.flush_tokens + req_len - 1) // req_len
        print(f"== flush (总量≈{args.flush_tokens} tokens = {n_flush} 条 × {req_len}/条, 长请求快速flush) ==")
    else:
        # 兼容旧模式：flush_n 条 × 2048
        req_len = 2048
        n_flush = args.flush_n
        print(f"== flush ({n_flush} junk reqs × {req_len}, 旧模式) ==")
    for i in range(n_flush):
        # 唯一 tag 放最前，保证每条互不命中；其后用单字符 'z' 词凑长度(近似一词一token)
        tag = f"flush{run_tag}n{i}r{random.randint(0, 1_000_000)}"
        junk = tag + " " + " ".join(["z"] * req_len)
        dt, _ = post_chat(junk)
    print("  flush done")

    # ---- metrics before ----
    before = scrape_metrics()

    hit = []
    if args.profile:
        # 只把 profile 窗口精确圈在前 profile_hits 条（每个 DP 副本首命中，含 H2D 回载）
        n_prof = min(args.profile_hits, args.round_n)
        print(f"== round2 (profiled): 前 {n_prof} 条 hit，圈在 profile 窗口内 ==")
        if not start_profile(args.output_dir, run_tag):
            print("[profiler] start 失败，跳过本轮 profile")
        for i in range(n_prof):
            dt, _ = post_chat(make_hit_body(prefix, run_tag, i, args.new_tokens))
            hit.append(dt)
        time.sleep(args.profile_settle)   # 等 transfer kernel 落地
        stop_profile()
        print(f"  profiled first {n_prof}: {[round(x) for x in hit]} ms")
        # 其余 hit 在 profile 关闭后补发（用于 metrics 统计，不进 trace）
        for i in range(n_prof, args.round_n):
            dt, _ = post_chat(make_hit_body(prefix, run_tag, i, args.new_tokens))
            hit.append(dt)
    else:
        print("== round2: hit (expect host hit + KV load) ==")
        for i in range(args.round_n):
            dt, _ = post_chat(make_hit_body(prefix, run_tag, i, args.new_tokens))
            hit.append(dt)

    hit_sorted = sorted(hit)
    print(f"  [hit] n={len(hit)} min={hit_sorted[0]:.1f} "
          f"median={hit_sorted[len(hit)//2]:.1f} max={hit_sorted[-1]:.1f} ms")
    print(f"  首4条(各DP首命中): {[round(x) for x in hit[:4]]}")

    # ---- metrics after + 判定 ----
    after = scrape_metrics()
    ok = diff_metrics(before, after, args.prefix, args.new_tokens)

    if args.profile:
        print(f"\n[trace] ★P 组两台各看: find {args.output_dir} -name '{run_tag}*.trace.json.gz'")
        print(f"        P0 是 TP 0-7，P1 是 TP 8-15，两台都要取")
        print(f"[trace] trace 中重点搜索：")
        print(f"        load_stream / hicache_transfer_per_layer / transfer_kernel_impl  ← KV load")
        print(f"        internode::dispatch / combine / notify_dispatch                  ← EP")
        print(f"        mooncake / kv_send / send_kvcache / transfer                     ← ★P->D KV 传输")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="KV load 一键实验（PD 分离版：metrics + 可选 profile）")
    ap.add_argument("--tag", required=True, help="本轮标签（多轮时作为前缀）")
    ap.add_argument("--prefix", type=int, default=16000, help="共享前缀词数（命中部分）")
    ap.add_argument("--new-tokens", type=int, default=0,
                    help="hit 请求在前缀后拼接的唯一新 token 数（控制命中率）；"
                         "命中率≈prefix/(prefix+new-tokens)，0=近100%%命中")
    ap.add_argument("--round-n", type=int, default=32, help="fill/hit 各多少请求")
    ap.add_argument("--flush-n", type=int, default=160,
                    help="[旧模式] flush 短请求数（每条 2048）；仅当 --flush-tokens=0 时生效")
    ap.add_argument("--flush-tokens", type=int, default=0,
                    help="[新模式/推荐] flush 目标总灌入 token 量（>0 时启用长请求快速flush，"
                         "自动算条数=flush-tokens/flush-req-len）；需 > L1 容量×dp 才能挤出目标前缀。"
                         "★PD 分离下需重新标定，建议从共置值的 1/2 起步")
    ap.add_argument("--flush-req-len", type=int, default=16000,
                    help="新模式下每条 flush 长请求的 token 数（默认16000，"
                         "平衡固定开销与单条时延；必须 ≤ 单请求能装进 L1 的上限）")
    ap.add_argument("--dp", type=int, default=4, help="DP 数（profile 时默认圈 dp 条 hit）")
    ap.add_argument("--profile", action="store_true", help="是否抓 profile（P 侧）")
    ap.add_argument("--output-dir", default="/tmp/sgprof", help="trace 输出目录")
    ap.add_argument("--profile-hits", type=int, default=4,
                    help="profile 窗口内只发前 N 条 hit（默认=dp，每个DP副本首命中，含H2D）")
    ap.add_argument("--profile-settle", type=float, default=2.0,
                    help="发完 profile 内的 hit 后、stop_profile 前等待 kernel 落地(秒)。"
                         "★PD 分离下 KV send 在 forward 之后，建议 ≥2.0 以覆盖 P->D 传输")
    ap.add_argument("--repeat", type=int, default=1, help="连续跑多少轮")
    ap.add_argument("--sleep-between", type=float, default=3.0,
                    help="多轮之间间隔(秒)")
    args = ap.parse_args()

    if args.profile:
        os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for k in range(args.repeat):
        run_tag = args.tag if args.repeat == 1 else f"{args.tag}-{k:03d}"
        ok = run_once(args, run_tag)
        results.append((run_tag, ok))
        if k < args.repeat - 1:
            time.sleep(args.sleep_between)

    print("\n==== 汇总 ====")
    for tag, ok in results:
        print(f"  {tag}: KV load {'确认' if ok else '未确认'}")


if __name__ == "__main__":
    main()
