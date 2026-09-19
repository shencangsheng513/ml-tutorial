"""L1-v4 多跳 A/B：直接 hybrid 检索 vs LLM 查询分解检索（qa_v3 的 4 道多跳题）。
判据：最差名次 MRR@6（与 v3 同口径）+ 所需文档全进 top-4，另做一轮端到端判卷对比。
API 花费：4 次分解 + 若干子查询嵌入（走缓存）+ 新上下文触发的生成/判卷。
用法：python evaluation/run_decompose_ab.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.bm25 import BM25, rrf_fuse
from rag.config import INDEX_DIR
from rag.decompose import retrieve_decomposed, subqueries
from rag.embeddings import embed_texts
from rag.judge import judge_pair
from rag.pipeline import answer_with_context, build_context
from rag.store import VectorStore

QA4 = json.loads((HERE / "evaluation" / "qa_v3.json").read_text(encoding="utf-8"))


def rank_of_first(docs_in_order: list[str], need: str) -> int:
    return next((r + 1 for r, d in enumerate(docs_in_order) if need in d), 99)


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    bm = BM25([c.text for c in store.chunks])
    qvecs = embed_texts([it["q"] for it in QA4], verbose=False)

    rows = []
    mrr = {"hybrid": 0.0, "decompose": 0.0}
    hit = {"hybrid4": 0, "decompose4": 0}
    for it, qv in zip(QA4, qvecs):
        vec = [i for i, _ in store.search_all(qv, k=6)]
        bml = [i for i, _ in bm.search(it["q"], k=6)]
        hy_full = sorted(rrf_fuse([vec, bml]).items(), key=lambda x: -x[1])
        hy = [i for i, _ in hy_full[:6]]
        dc_pairs = retrieve_decomposed(store, it["q"], k=6, cand=8, _bm=bm)
        dc = [i for i, _ in dc_pairs]
        row = {"q": it["q"][:40], "subqs": subqueries(it["q"])[1:]}
        for name, idxs in [("hybrid", hy), ("decompose", dc)]:
            docs = [store.chunks[i].doc for i in idxs]
            worst = max(rank_of_first(docs, d) for d in it["docs"])
            mrr[name] += 1.0 / worst
            if worst <= 4:
                hit[f"{name}4"] += 1
            row[f"{name}_mrr"] = round(1.0 / worst, 4)
        # 端到端：两版上下文各自生成答案，一次判卷（奇偶交换位置，grounded 用分解版上下文）
        hits_h = [(store.chunks[i], s) for i, s in hy_full[:4]]
        hits_d = [(store.chunks[i], s) for i, s in dc_pairs[:4]]
        ans_h = answer_with_context(it["q"], hits_h, score_label="融合分")
        ans_d = answer_with_context(it["q"], hits_d, score_label="融合分")
        ctx = build_context(hits_d, score_label="融合分")
        swap = len(rows) % 2
        j = judge_pair(it["q"], it["gold"], ans_d if swap else ans_h,
                       ans_h if swap else ans_d, context=ctx)
        a, b = (j["A"], j["B"]) if swap else (j["B"], j["A"])
        row.update(decomp=a, direct=b, judge_reason=j.get("reason", "")[:60])
        print(f"  h{row['hybrid_mrr']:.2f} d{row['decompose_mrr']:.2f} | 分解{a} vs 直接{b} | {row['q']}…")
        rows.append(row)

    out = {"summary": {"mrr6": {k: round(v / len(QA4), 4) for k, v in mrr.items()},
                       "all_docs_top4": hit, "n": len(QA4)},
           "per_question": rows}
    (HERE / "evaluation" / "decompose_ab_v4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMRR@6（与 v3 同口径）:", out["summary"]["mrr6"], "| 双文档进top-4:", hit)
    print("-> decompose_ab_v4.json")


if __name__ == "__main__":
    main()
