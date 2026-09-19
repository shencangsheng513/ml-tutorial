"""v4 L1 A/B（大样本）：20 道 LLM 生成的跨文档多跳题上比三臂检索——
直接 hybrid vs 分解+槽位保证(reserve) vs 分解+纯多查询RRF(plain)。
判据：最差名次 MRR@6 + 所需双文档全进 top-4。成本 = 每题 1 次分解调用 + 子查询嵌入（均落盘缓存）。
用法：python evaluation/run_multihop20_ab.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.bm25 import BM25, rrf_fuse
from rag.config import INDEX_DIR
from rag.decompose import retrieve_decomposed
from rag.embeddings import embed_texts
from rag.store import VectorStore

QA = json.loads((HERE / "evaluation" / "qa_multihop20.json").read_text(encoding="utf-8"))


def worst_rank(docs_in_order: list[str], need: list[str]) -> int:
    return max(next((r + 1 for r, d in enumerate(docs_in_order) if n in d), 99) for n in need)


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    bm = BM25([c.text for c in store.chunks])
    qvecs = embed_texts([it["q"] for it in QA], verbose=False)
    mrr = {"hybrid": 0.0, "decomp_res": 0.0, "decomp_pl": 0.0}
    hit4 = {k: 0 for k in mrr}
    rows = []
    for it, qv in zip(QA, qvecs):
        vec = [i for i, _ in store.search_all(qv, k=6)]
        bml = [i for i, _ in bm.search(it["q"], k=6)]
        hy = [i for i, _ in sorted(rrf_fuse([vec, bml]).items(), key=lambda x: -x[1])[:6]]
        res = [i for i, _ in retrieve_decomposed(store, it["q"], k=6, cand=8, _bm=bm, reserve=True)]
        pla = [i for i, _ in retrieve_decomposed(store, it["q"], k=6, cand=8, _bm=bm, reserve=False)]
        row = {"q": it["q"][:40], "docs": it["docs"]}
        for name, idxs in [("hybrid", hy), ("decomp_res", res), ("decomp_pl", pla)]:
            w = worst_rank([store.chunks[i].doc for i in idxs], it["docs"])
            mrr[name] += 1.0 / w
            hit4[name] += w <= 4
            row[name] = w
        rows.append(row)
        print(f"  v{row['hybrid']:>3} r{row['decomp_res']:>3} p{row['decomp_pl']:>3}  {row['q']}…")

    n = len(QA)
    out = {"summary": {"mrr6": {k: round(v / n, 4) for k, v in mrr.items()},
                       "all_top4": hit4, "n": n}, "per_question": rows}
    (HERE / "evaluation" / "multihop20_ab_v4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMRR@6 (n={n}):", out["summary"]["mrr6"], "| 双文档全进top-4:", hit4)


if __name__ == "__main__":
    main()
