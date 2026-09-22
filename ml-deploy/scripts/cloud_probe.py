"""现网探针：只打 Render 公开端点 /healthz（无 token），量冷启动与公网稳态延迟。

背景：README"免费档脾气"里的 1/5 折算与 p99 300ms 承诺是推算值，本脚本用实测代替拍脑袋。
纪律：鉴权端点不打（token 只在 Render 环境变量，不落盘）；总请求 ~30 发，对免费档无压力。
用法：python scripts/cloud_probe.py [--base https://news-classifier-service-jsfq.onrender.com]
产物：cloud_probe_results.json
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DEFAULT = "https://news-classifier-service-jsfq.onrender.com"
WARM_N, WARM_GAP = 25, 0.8


def get(url: str, timeout: float = 150.0):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, body, (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, "", (time.perf_counter() - t0) * 1000
    except OSError as e:  # URLError / socket.timeout / 断连都算"这一发失败"，但不算崩溃
        return None, repr(e), (time.perf_counter() - t0) * 1000


def parsed_json(body: str):
    """休眠中的免费档会返回 HTML 错误页，此时没有 model_loaded 可读（返回 None 让调用方保留上一发）。"""
    try:
        return json.loads(body)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    args = ap.parse_args()
    url = args.base.rstrip("/") + "/healthz"

    st, body, ms = get(url)  # 第一发：若空闲休眠，这就是冷启动实测
    cold = {"status": st, "ms": round(ms), "body": body[:200]}
    warm_ms = []
    statuses = []
    loaded = None
    for _ in range(WARM_N):
        time.sleep(WARM_GAP)
        st, body, ms = get(url, timeout=20)
        statuses.append(st)
        warm_ms.append(round(ms))
        payload = parsed_json(body)
        if isinstance(payload, dict):
            loaded = payload.get("model_loaded", loaded)
    warm = {"n": len(warm_ms), "all_200": set(statuses) == {200},
            "model_loaded": loaded,
            "p50_ms": int(statistics.median(warm_ms)),
            "p95_ms": int(sorted(warm_ms)[int(0.95 * len(warm_ms)) - 1]),
            "max_ms": max(warm_ms), "samples": warm_ms}
    out = {"cold_first_hit": cold, "warm": warm,
           "note": "冷启动能否成立取决于探针前空闲是否已超15分钟休眠阈值"}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    dest = Path(__file__).resolve().parents[1] / "cloud_probe_results.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
