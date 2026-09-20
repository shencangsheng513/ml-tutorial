"""v7：50 题多跳端到端判卷——v5 只测了检索层 L1，判卷层（答案质量）一直是空白。
链路 = 生产主路径：hybrid 检索 top-4 → RAG 作答 vs 裸考 → L3a 配对盲评 + L3b 有据性。
gold 要点来自 build_multihop50_gold.py（引文式、逐片段回查语料验证过）。
分层看：both-docs-in-top-4 的题 vs 检索没覆盖全的题，答案质量差多少——
"多跳是检索短板"到底会不会传导成答案短板，这一跑见分晓。
用法：python evaluation/run_multihop50_e2e.py   （先跑 scripts/build_multihop50_gold.py）
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.judge import judge_faithful, judge_pair
from rag.pipeline import answer_bare, answer_with_context, build_context, retrieve_hybrid
from rag.store import VectorStore

K = 4
GOLD = HERE / "evaluation" / "qa_multihop50_gold.json"
OUT = HERE / "evaluation" / "multihop50_e2e_v7.json"


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    data = json.loads(GOLD.read_text(encoding="utf-8"))
    items = data["kept"]
    print(f"50 题多跳 e2e：留 {len(items)} 题（废 {len(data['dropped'])}），"
          f"索引 {len(store.chunks)} 块，检索走 hybrid 主路径\n")
    t0 = time.time()
    rows = []
    for i, item in enumerate(items):
        found = retrieve_hybrid(store, item["q"], k=K)
        docs_hit = {c.doc for c, _ in found}
        need = set(item["docs"])
        both = need <= docs_hit
        a_rag = answer_with_context(item["q"], found, score_label="RRF分")
        a_bare = answer_bare(item["q"])
        x, y = (a_bare, a_rag) if i % 2 else (a_rag, a_bare)  # 奇偶换位防位置偏置
        j = judge_pair(item["q"], item["gold"], x, y)
        s_rag, s_bare = (j["B"], j["A"]) if i % 2 else (j["A"], j["B"])
        f = judge_faithful(item["q"], item["gold"], a_rag,
                           build_context(found, "RRF分"))
        viols = f.get("violations", []) or []
        rows.append({"q": item["q"], "docs": item["docs"], "both_in_top4": both,
                     "rag": s_rag, "bare": s_bare, "faithful": float(f["score"]),
                     "violations": viols, "reason": j.get("reason", "")})
        mark = "✓" if both else "✗"
        print(f"  [{i+1:02d}] 检索{mark} 配对 RAG {s_rag:.1f} vs 裸考 {s_bare:.1f}"
              f" | 有据性 {f['score']:.1f}（无据 {len(viols)}）  {item['q'][:30]}")

    n = len(rows)
    hit = [r for r in rows if r["both_in_top4"]]
    miss = [r for r in rows if not r["both_in_top4"]]

    def agg(rs):
        if not rs:
            return "—", "—", "—", 0
        def mean(k):
            return sum(r[k] for r in rs) / len(rs)
        w = sum(r["rag"] > r["bare"] for r in rs)
        l = sum(r["rag"] < r["bare"] for r in rs)
        return (f"{mean('rag'):.2f}", f"{mean('bare'):.2f}", f"{mean('faithful'):.2f}",
                f"{w}/{len(rs)-w-l}/{l}")

    print(f"\n全部 {n} 题:      RAG {agg(hit+miss)[0]}  裸考 {agg(hit+miss)[1]}  "
          f"有据性 {agg(hit+miss)[2]}  胜/平/负 {agg(hit+miss)[3]}")
    print(f"检索双中 {len(hit)} 题: RAG {agg(hit)[0]}  裸考 {agg(hit)[1]}  "
          f"有据性 {agg(hit)[2]}  胜/平/负 {agg(hit)[3]}")
    print(f"检索缺腿 {len(miss)} 题: RAG {agg(miss)[0]}  裸考 {agg(miss)[1]}  "
          f"有据性 {agg(miss)[2]}  胜/平/负 {agg(miss)[3]}")
    print(f"总耗时 {time.time()-t0:.0f}s")
    OUT.write_text(json.dumps({
        "mode": "v7：50 题多跳 e2e，hybrid 主路径 + L3a 配对 + L3b 有据性，gold=引文式",
        "index_chunks": len(store.chunks), "kept": n, "dropped": len(data["dropped"]),
        "summary": {"all": agg(hit+miss), "both_hit": agg(hit), "miss": agg(miss)},
        "per_question": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已存 {OUT.name}")


if __name__ == "__main__":
    main()
