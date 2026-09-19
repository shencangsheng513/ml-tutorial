"""L1 召回压测：80+ 题压力集上对比 纯向量 vs BM25+向量 hybrid 的 top-4 命中率。
零生成费（只花嵌入调用，且走 embed 缓存）。用法：python evaluation/stress_recall.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.pipeline import retrieve, retrieve_hybrid
from rag.store import VectorStore

SETS = ["qa_stress80.json", "qa_stress.json", "qa_v3.json", "qa_eval.json"]


def hit(item: dict, found) -> bool:
    docs = item.get("docs") or [item.get("doc")]
    return any(any(d in c.doc for c, _ in found) for d in docs)


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    report = {}
    for name in SETS:
        p = HERE / "evaluation" / name
        if not p.exists():
            continue
        qs = json.loads(p.read_text(encoding="utf-8"))
        rows = []
        for item in qs:
            v = hit(item, retrieve(store, item["q"], k=4))
            h = hit(item, retrieve_hybrid(store, item["q"], k=4))
            rows.append({"q": item["q"], "vec": v, "hybrid": h})
        nv = sum(r["vec"] for r in rows)
        nh = sum(r["hybrid"] for r in rows)
        report[name] = {"n": len(qs), "vec": nv, "hybrid": nh,
                        "misses": [r for r in rows if not (r["vec"] and r["hybrid"])]}
        print(f"{name:20s} 向量 {nv}/{len(qs)}  hybrid {nh}/{len(qs)}")
        for r in rows:
            if r["vec"] != r["hybrid"]:
                who = "仅向量中" if r["vec"] else "仅hybrid中"
                print(f"    [{who}] {r['q'][:50]}")
        for r in report[name]["misses"]:
            print(f"    [双双落空] {r['q'][:50]}")
    (HERE / "evaluation" / "stress_recall.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已存 evaluation/stress_recall.json")


if __name__ == "__main__":
    main()
