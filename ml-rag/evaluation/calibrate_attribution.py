"""v10·A 校准：给"数字有据性核验器"体检——它在生产答案上报的可疑数字，
到底是真幻觉还是算术派生/换写的误报？全走 v7.1 缓存，零重付（缺缓存的题直接跳过）。
判据：核验器要对已有 L3b 判决的题给出与人工可核实的"编造"信号大体一致，
误报率太高（动辄把派生数字标红）就不配进 ask.py 主流程。
用法：python evaluation/calibrate_attribution.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.attribution import ungrounded_numbers
from rag.config import INDEX_DIR
from rag.pipeline import build_context, retrieve_hybrid
from rag.store import VectorStore

CACHE = json.loads((INDEX_DIR / "answer_cache.json").read_text(encoding="utf-8"))
E2E = HERE / "evaluation" / "multihop50_e2e_v7.json"
K = 4


def cache_key(q: str, hits) -> str:
    prompt = f"{build_context(hits, 'RRF分')}\n\n问题：{q}"
    return hashlib.md5(("rag|" + prompt).encode("utf-8")).hexdigest()


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    rows = json.loads(E2E.read_text(encoding="utf-8"))["per_question"]
    t0 = time.time()
    ok = skip = 0
    per_q = []
    for item in rows:
        hits = retrieve_hybrid(store, item["q"], k=K)
        key = cache_key(item["q"], hits)
        answer = CACHE.get(key)
        if answer is None:
            skip += 1
            continue
        ok += 1
        flagged = ungrounded_numbers(answer, hits, item["q"])
        per_q.append((item["q"], item["faithful"], len(flagged), flagged))
    n_flagged = sum(1 for *_, k, _ in per_q if k)
    total_flags = sum(k for *_, k, _ in per_q)
    print(f"核验器校准：可用 {ok} 题 / 缺缓存跳过 {skip} 题（走 v7.1 缓存零费）")
    print(f"报告可疑数字的题：{n_flagged}/{ok}（{100*n_flagged//max(ok,1)}%），"
          f"共标 {total_flags} 个数字，人均 {total_flags/max(ok,1):.1f}\n")
    print("—— 按 L3b 有据性升序（分低=判卷官已认为有编造，核验器该在这类题上响）——")
    for q, faith, k, flags in sorted(per_q, key=lambda r: r[1])[:18]:
        tag = f"标[{'] ['.join(flags)}]" if flags else "无标"
        print(f"  L3b={faith:>4} | {k}个 | {q[:34]}… | {tag}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
