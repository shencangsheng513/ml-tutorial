"""判卷官校准验证（v3）：用当前锚定式 CRITERIA 重判 v2 的缓存答案。
- 若 13 题全部命中 judge_cache → 说明 v2 已用过这套规则仍满分（饱和是判卷官的问题，需升级量表）；
- 若大量未命中 → v2 是旧规则判的，本次重判本身就是校准实验。
答案文本全部走 answer_cache，零生成费；判卷按需 13 次以内调用。
用法：python evaluation/rescore_judge.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.judge import _cache_load, judge_pair
from rag.pipeline import answer_bare, answer_with_context, retrieve
from rag.store import VectorStore

QA = json.loads((HERE / "evaluation" / "qa_eval.json").read_text(encoding="utf-8"))


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    scores, new_calls = [], 0
    for i, item in enumerate(QA):
        found = retrieve(store, item["q"], k=4)
        a_rag = answer_with_context(item["q"], found)
        a_bare = answer_bare(item["q"])
        if i % 2:  # 与 run_eval.py 相同的奇偶换位
            x, y = a_bare, a_rag
        else:
            x, y = a_rag, a_bare
        before = set(_cache_load().keys())
        ctx = "\n\n".join(f"【资料{i+1}·{c.doc}·{c.heading}】\n{c.text}"
                          for i, (c, _) in enumerate(found))
        j = judge_pair(item["q"], item["gold"], x, y, context=ctx)
        new_calls += len(set(_cache_load().keys()) - before)
        s_rag = j["B"] if i % 2 else j["A"]
        s_bare = j["A"] if i % 2 else j["B"]
        scores.append((s_rag, s_bare))
        print(f"  [{i+1:02d}] RAG {s_rag} vs 裸考 {s_bare}")
    rag_scores = [a for a, _ in scores]
    print(f"\n本次实际调用判卷 {new_calls} 次（其余命中缓存）")
    print(f"RAG 分数分布：{sorted(rag_scores)}  满分占比 {sum(1 for v in rag_scores if v == 10)}/13")
    out = HERE / "evaluation" / "rescore_v3.json"
    out.write_text(json.dumps({"mode": "grounded(锚定+有据性)", "new_calls": new_calls,
                               "scores": [{"rag": a, "bare": b} for a, b in scores]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
