"""RAG 评估三层：
  L1 检索命中率（零成本）：top-k 里有没有标准答案所在文档
  L2 有上下文 vs 裸考：同 13 题各答一遍
  L3 LLM-as-judge 双盲评分（位置奇偶交换 A/B，防位置偏置）
用法：python evaluation/run_eval.py   （需先 build_index）
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.judge import judge_pair
from rag.pipeline import answer_bare, answer_with_context, retrieve
from rag.store import VectorStore

QA = json.loads((HERE / "evaluation" / "qa_eval.json").read_text(encoding="utf-8"))
K = 4


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    t0 = time.time()

    print(f"L1) 检索命中率（top-{K} 含标准出处文档）")
    hits = 0
    rows = []
    for item in QA:
        found = retrieve(store, item["q"], k=K)
        ok = any(item["doc"] in c.doc for c, _ in found)
        hits += ok
        rows.append((item, found, ok))
        mark = "✓" if ok else "✗"
        print(f"  {mark} {item['q'][:34]}…  ->  {found[0][0].doc}§{found[0][0].heading[:18]}")
    print(f"  命中率 {hits}/{len(QA)} = {hits/len(QA):.0%}\n")

    print("L2+L3) RAG vs 裸考，judge 双盲")
    rag_wins = bare_wins = ties = 0
    rag_sum = bare_sum = 0.0
    detail = []
    for i, (item, found, _) in enumerate(rows):
        a_rag = answer_with_context(item["q"], found)
        a_bare = answer_bare(item["q"])
        if i % 2:  # 交换位置：A=bare, B=rag
            j = judge_pair(item["q"], item["gold"], a_bare, a_rag)
            s_rag, s_bare = j["B"], j["A"]
        else:
            j = judge_pair(item["q"], item["gold"], a_rag, a_bare)
            s_rag, s_bare = j["A"], j["B"]
        rag_wins += s_rag > s_bare
        bare_wins += s_rag < s_bare
        ties += s_rag == s_bare
        rag_sum += s_rag
        bare_sum += s_bare
        detail.append({"q": item["q"], "rag": s_rag, "bare": s_bare, "reason": j.get("reason", "")})
        print(f"  [{i+1:02d}] RAG {s_rag:.1f} vs 裸考 {s_bare:.1f}  {j.get('reason','')[:56]}")

    n = len(QA)
    print(f"\n结论：RAG 均分 {rag_sum/n:.2f}  裸考均分 {bare_sum/n:.2f}  "
          f"胜/平/负 = {rag_wins}/{ties}/{bare_wins}（{n} 题）")
    print(f"总耗时 {time.time()-t0:.0f}s")
    (HERE / "evaluation" / "eval_results.json").write_text(
        json.dumps({"hit_rate": f"{hits}/{len(QA)}", "per_question": detail},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("明细已存 evaluation/eval_results.json")


if __name__ == "__main__":
    main()
