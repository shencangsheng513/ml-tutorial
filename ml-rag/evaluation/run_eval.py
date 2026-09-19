"""RAG 评估：
  L1 检索命中率（零成本）：top-4 里有没有标准答案所在文档
  L2 有上下文 vs 裸考：同题各答一遍（答案走 answer_cache，判卷官迭代不重新生成）
  L3a LLM-as-judge 配对盲评（无资料、对称规则）：正确性/覆盖/可操作
  L3b 有据性判卷（RAG 答案单独对检索到的原始资料核幻觉，探针验证过的机制）
用法：python evaluation/run_eval.py   （需先 build_index）
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.judge import judge_faithful, judge_pair
from rag.pipeline import answer_bare, answer_with_context, build_context, retrieve
from rag.store import VectorStore

K = 4


def load_questions() -> list[dict]:
    items = []
    for name in ("qa_eval.json", "qa_v3.json"):
        p = HERE / "evaluation" / name
        for it in json.loads(p.read_text(encoding="utf-8")):
            if "docs" not in it:
                it["docs"] = [it["doc"]]
            items.append(it)
    return items


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    qa = load_questions()
    t0 = time.time()

    print(f"L1) 检索命中率（top-{K} 含标准出处文档，共 {len(qa)} 题）")
    hits = 0
    rows = []
    for item in qa:
        found = retrieve(store, item["q"], k=K)
        ok = any(any(d in c.doc for c, _ in found) for d in item["docs"])
        hits += ok
        rows.append((item, found, ok))
        mark = "✓" if ok else "✗"
        print(f"  {mark} {item['q'][:34]}…  ->  {found[0][0].doc}§{found[0][0].heading[:18]}")
    print(f"  命中率 {hits}/{len(qa)} = {hits/len(qa):.0%}\n")

    print("L2+L3) RAG vs 裸考（L3a 配对盲评）+ RAG 幻觉审计（L3b 有据性）")
    rag_wins = bare_wins = ties = 0
    rag_sum = bare_sum = faith_sum = 0.0
    viol_total = 0
    detail = []
    for i, (item, found, _) in enumerate(rows):
        a_rag = answer_with_context(item["q"], found)
        a_bare = answer_bare(item["q"])
        x, y = (a_bare, a_rag) if i % 2 else (a_rag, a_bare)  # 奇偶换位防位置偏置
        j = judge_pair(item["q"], item["gold"], x, y)
        s_rag, s_bare = (j["B"], j["A"]) if i % 2 else (j["A"], j["B"])
        f = judge_faithful(item["q"], item["gold"], a_rag, build_context(found))
        s_faith = float(f["score"])
        viols = f.get("violations", []) or []
        rag_wins += s_rag > s_bare
        bare_wins += s_rag < s_bare
        ties += s_rag == s_bare
        rag_sum += s_rag
        bare_sum += s_bare
        faith_sum += s_faith
        viol_total += len(viols)
        detail.append({"q": item["q"], "rag": s_rag, "bare": s_bare,
                       "faithful": s_faith, "violations": viols,
                       "fact_check": j.get("fact_check", {}),
                       "reason": j.get("reason", "")})
        print(f"  [{i+1:02d}] 配对 RAG {s_rag:.1f} vs 裸考 {s_bare:.1f} | 有据性 {s_faith:.1f}"
              f"（无据断言 {len(viols)}）  {j.get('reason', '')[:36]}")

    n = len(qa)
    print(f"\n结论：L3a RAG 均分 {rag_sum/n:.2f}  裸考均分 {bare_sum/n:.2f}  "
          f"胜/平/负 = {rag_wins}/{ties}/{bare_wins}（{n} 题）")
    print(f"     L3b RAG 有据性均分 {faith_sum/n:.2f}  无据断言共 {viol_total} 处"
          f"（0 = 判卷官零容忍且答案全部有据）")
    print(f"总耗时 {time.time()-t0:.0f}s")
    (HERE / "evaluation" / "eval_results.json").write_text(
        json.dumps({"hit_rate": f"{hits}/{n}", "mode": "v4：L3a对称配对 + L3b有据性",
                    "per_question": detail},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("明细已存 evaluation/eval_results.json")


if __name__ == "__main__":
    main()
