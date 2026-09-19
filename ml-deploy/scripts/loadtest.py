"""ml-deploy 本地压测：三档并发打 /v1/classify，量 RPS、延迟分位数、错误率。
用法（先起服务：run_server.bat 或 uvicorn deploy.service:app --app-dir src --port 8000）：
    python scripts/loadtest.py [--host 127.0.0.1:8000]
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

TEXTS = [
    "Space shuttle discovery is back. NASA announced the next flight schedule.",
    "The Maple Leafs defeated the Rangers 4-2 in an overtime thriller tonight.",
    "New graphics card benchmarks show a 30% uplift in ray tracing performance.",
    "The SEC commissioned a study on corporate accounting fraud last quarter.",
]
LOCK = threading.Lock()
LAT = []
ERRS = []


def worker(host: str, n: int, wid: int, batch: int = 1) -> None:
    for i in range(n):
        texts = [TEXTS[(wid + i + j) % len(TEXTS)] for j in range(batch)]
        body = json.dumps({"texts": texts}).encode()
        req = urllib.request.Request(f"http://{host}/v1/classify", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
            dt = (time.perf_counter() - t0) * 1000
            with LOCK:
                LAT.append(dt)
        except urllib.error.URLError as e:
            with LOCK:
                ERRS.append(f"{type(e).__name__}: {e}")


def pct(xs, p):
    ys = sorted(xs)
    return ys[min(int(len(ys) * p / 100), len(ys) - 1)]


def run_level(host: str, conc: int, per_thread: int, batch: int = 1) -> dict:
    LAT.clear()
    ERRS.clear()
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(host, per_thread, i, batch)) for i in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    lat = list(LAT)
    return {"concurrency": conc, "batch": batch, "requests": len(lat) + len(ERRS),
            "ok": len(lat), "errors": len(ERRS),
            "rps": round((len(lat) + len(ERRS)) / wall, 1),
            "p50_ms": round(pct(lat, 50), 1) if lat else None,
            "p95_ms": round(pct(lat, 95), 1) if lat else None,
            "p99_ms": round(pct(lat, 99), 1) if lat else None,
            "wall_s": round(wall, 2),
            "error_sample": ERRS[:2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:8000")
    args = ap.parse_args()
    with urllib.request.urlopen(f"http://{args.host}/ready", timeout=10) as r:
        assert json.loads(r.read())["status"] == "ready", "模型未就绪"
    results = [run_level(args.host, c, n, b) for c, n, b in
               [(1, 60, 1), (8, 40, 1), (32, 25, 1), (8, 10, 16)]]
    for r in results:
        print(f"并发{r['concurrency']:>3} 批{r['batch']:>3}  {r['rps']:>6} RPS "
              f"({r['rps'] * r['batch']:>7.1f} 条/秒)  "
              f"p50 {r['p50_ms']}ms  p95 {r['p95_ms']}ms  p99 {r['p99_ms']}ms  "
              f"错误 {r['errors']}/{r['requests']}")
    out = {"levels": results}
    Path(__file__).resolve().parents[1].joinpath("loadtest_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果已存 loadtest_results.json")


if __name__ == "__main__":
    main()
