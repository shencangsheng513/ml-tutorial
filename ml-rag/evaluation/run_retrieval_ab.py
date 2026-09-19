"""L1-v3 检索 A/B（离线为主）：向量 vs BM25 vs RRF混合，单跳13题+多跳4题。
判据：所需文档全部出现在 top-k（多跳题要求两篇都在——这是对"跳"的真实考验）。
只有 4 道新题需要 embedding API（旧题命中缓存）。
用法：python evaluation/run_retrieval_ab.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.bm25 import BM25, rrf_fuse
from rag.config import INDEX_DIR
from rag.embeddings import embed_texts
from rag.store import VectorStore

QA13 = json.loads((HERE / "evaluation" / "qa_eval.json").read_text(encoding="utf-8"))
QA4 = json.loads((HERE / "evaluation" / "qa_v3.json").read_text(encoding="utf-8"))
QA6 = json.loads((HERE / "evaluation" / "qa_stress.json").read_text(encoding="utf-8"))
ITEMS = [{"q": it["q"], "docs": [it["doc"]], "grp": "单跳"} for it in QA13] + \
        [{"q": it["q"], "docs": it["docs"], "grp": "多跳", "multi": True} for it in QA4] + \
        [{"q": it["q"], "docs": it["docs"], "grp": "压力"} for it in QA6]


def doc_in(chunk_docs: list[str], need: str) -> bool:
    return any(need in d for d in chunk_docs)


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    bm = BM25([c.text for c in store.chunks])
    qtexts = [it["q"] for it in ITEMS]
    qvecs = embed_texts(qtexts, verbose=True)

    results: dict[str, int] = {}
    mrr: dict[str, float] = {}
    per = []
    for it, qv in zip(ITEMS, qvecs):
        vec_idx = [i for i, _ in store.search_all(qv, k=6)]
        bm_idx = [i for i, _ in bm.search(it["q"], k=6)]
        hy_idx = [i for i, _ in sorted(rrf_fuse([vec_idx, bm_idx]).items(), key=lambda x: -x[1])[:6]]
        row = {"q": it["q"][:36], "grp": it["grp"]}
        for name, idxs in [("vector", vec_idx), ("bm25", bm_idx), ("hybrid", hy_idx)]:
            docs = [store.chunks[i].doc for i in idxs]
            for kk in (4, 6):
                ok = all(doc_in(docs[:kk], d) for d in it["docs"])
                key = f"{it['grp']}/{name}@{kk}"
                results[key] = results.get(key, 0) + ok
                row[f"{name}{kk}"] = "✓" if ok else "✗"
            # MRR：所需文档中最靠后的那个的首次出现名次（多跳取最差）
            worst = 0
            for d in it["docs"]:
                pos = next((r + 1 for r, docn in enumerate(docs) if d in docn), 99)
                worst = max(worst, pos)
            mrr[f"{it['grp']}/{name}"] = mrr.get(f"{it['grp']}/{name}", 0.0) + 1.0 / worst
        per.append(row)
        print(f"  [{it['grp']}] v{row['vector4']} b{row['bm254']} h{row['hybrid4']} | "
              f"v{row['vector6']} b{row['bm256']} h{row['hybrid6']}  {row['q']}…")

    print("\n命中统计（多跳/压力按所需文档全中计）：")
    mrr_n = {}
    for grp, n in [("单跳", 13), ("多跳", 4), ("压力", 6)]:
        line = f"  {grp}({n}题) "
        for kk in (4, 6):
            line += f" top-{kk}: v {results.get(f'{grp}/vector@{kk}', 0)}" \
                    f" b {results.get(f'{grp}/bm25@{kk}', 0)}" \
                    f" h {results.get(f'{grp}/hybrid@{kk}', 0)}   "
        for name in ("vector", "bm25", "hybrid"):
            mrr_n[f"{grp}/{name}"] = round(mrr[f"{grp}/{name}"] / n, 4)
        line += " | MRR@6: " + " ".join(f"{name[0]} {mrr_n[f'{grp}/{name}']:.3f}"
                                        for name in ("vector", "bm25", "hybrid"))
        print(line)
    out = HERE / "evaluation" / "retrieval_ab_v3.json"
    out.write_text(json.dumps({"summary": results, "mrr6": mrr_n, "per_question": per},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细 -> {out.name}")


if __name__ == "__main__":
    main()
