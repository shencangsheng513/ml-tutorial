"""v5 L1 大盘：50 道跨文档多跳题上重跑 v4 三臂（hybrid / 分解+reserve / 分解+plain），
并给 reserve-vs-hybrid 的胜负做精确二项检验（McNemar 思路：只看 discordant 题）。
用法：python evaluation/run_multihop50_ab.py   （成本=新题分解调用+嵌入，均有缓存）
"""
import json
import math
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
from run_multihop20_ab import worst_rank

QA = json.loads((HERE / "evaluation" / "qa_multihop50.json").read_text(encoding="utf-8"))


def binom_two_sided(w: int, d: int) -> float:
    """H0: p=0.5 的双侧精确检验——d 次 discordant 里 w 次偏 reserve。"""
    if d == 0:
        return 1.0
    tail = sum(math.comb(d, k) for k in range(min(w, d - w) + 1)) / 2 ** d
    return min(1.0, 2 * tail)


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    bm = BM25([c.text for c in store.chunks])
    qvecs = embed_texts([it["q"] for it in QA], verbose=False)
    mrr = {"hybrid": 0.0, "decomp_res": 0.0, "decomp_pl": 0.0}
    hit4 = {k: 0 for k in mrr}
    disc = {"res_better": 0, "hyb_better": 0, "tie": 0}
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
        if row["decomp_res"] < row["hybrid"]:
            disc["res_better"] += 1
        elif row["decomp_res"] > row["hybrid"]:
            disc["hyb_better"] += 1
        else:
            disc["tie"] += 1
        rows.append(row)
        print(f"  v{row['hybrid']:>3} r{row['decomp_res']:>3} p{row['decomp_pl']:>3}  {row['q']}…")

    n = len(QA)
    d = disc["res_better"] + disc["hyb_better"]
    p = binom_two_sided(max(disc["res_better"], disc["hyb_better"]), d)
    out = {"summary": {"mrr6": {k: round(v / n, 4) for k, v in mrr.items()},
                       "all_top4": hit4, "n": n,
                       "reserve_vs_hybrid": {**disc, "discordant": d, "p_two_sided": round(p, 4)}},
           "per_question": rows}
    (HERE / "evaluation" / "multihop50_ab_v5.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMRR@6 (n={n}):", out["summary"]["mrr6"])
    print("双文档全进top-4:", hit4)
    print(f"reserve vs hybrid: 分解优 {disc['res_better']} / 直接优 {disc['hyb_better']} / 平 {disc['tie']}"
          f"  双侧精确检验 p={p:.3f}")


if __name__ == "__main__":
    main()
