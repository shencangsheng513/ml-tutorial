"""v8：缺腿→幻觉 的针对性缓解消融——基线提示词 vs 基线+诚实护栏。
v7.1 证明检索缺腿不掉分、只掉有据性（9.45→8.30），本实验测一句提示词能不能补回来。
口径：同一 42 题、同一 hybrid 检索结果；基线臂全部走 v7.1 缓存（零重付），护栏臂新调用。
主指标 L3b 有据性（按双中/缺腿分组做配对符号检验）；副指标 L3a 配对——看"明说不足"会不会牺牲完整性。
用法：python evaluation/run_coverage_guard_ab.py   （先跑 scripts/build_multihop50_gold.py）
"""
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.judge import judge_faithful, judge_pair
from rag.pipeline import SYSTEM_RAG, answer_with_context, build_context, retrieve_hybrid
from rag.store import VectorStore

GUARD = (SYSTEM_RAG + "若资料只覆盖问题的一部分（例如比较题只给了一边的资料），"
         "只答有据的部分，没据的部分明确写\"资料不足\"，不要凭常识补全。")
K = 4
GOLD = HERE / "evaluation" / "qa_multihop50_gold.json"
OUT = HERE / "evaluation" / "coverage_guard_ab_v8.json"


def sign_p(w: int, l: int) -> float:
    """双侧精确符号检验：平局不计入 n。"""
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    items = json.loads(GOLD.read_text(encoding="utf-8"))["kept"]
    print(f"v8 护栏消融：{len(items)} 题，基线臂走缓存、护栏臂新调用\n")
    t0 = time.time()
    rows = []
    for i, item in enumerate(items):
        found = retrieve_hybrid(store, item["q"], k=K)
        both = set(item["docs"]) <= {c.doc for c, _ in found}
        ctx = build_context(found, "RRF分")
        a_base = answer_with_context(item["q"], found, score_label="RRF分")
        a_guard = answer_with_context(item["q"], found, score_label="RRF分", system=GUARD)
        f_base = judge_faithful(item["q"], item["gold"], a_base, ctx)
        f_guard = judge_faithful(item["q"], item["gold"], a_guard, ctx)
        x, y = (a_base, a_guard) if i % 2 else (a_guard, a_base)  # 奇偶换位防位置偏置
        j = judge_pair(item["q"], item["gold"], x, y)
        s_guard, s_base = (j["B"], j["A"]) if i % 2 else (j["A"], j["B"])
        trig = "资料不足" in a_guard
        rows.append({"q": item["q"], "both_in_top4": both, "triggered": trig,
                     "faith_base": float(f_base["score"]), "faith_guard": float(f_guard["score"]),
                     "pair_guard": s_guard, "pair_base": s_base,
                     "v_base": len(f_base.get("violations", []) or []),
                     "v_guard": len(f_guard.get("violations", []) or [])})
        print(f"  [{i+1:02d}] {'双中' if both else '缺腿'} 触发护栏{'✓' if trig else '-'}  "
              f"有据性 {f_base['score']:.1f}→{f_guard['score']:.1f}（无据 {len(f_base.get('violations', []) or [])}→{len(f_guard.get('violations', []) or [])}）  "
              f"完整性 {s_base:.1f}→{s_guard:.1f}  {item['q'][:24]}")

    def agg(rs, label):
        if not rs:
            return
        fb = sum(r["faith_base"] for r in rs) / len(rs)
        fg = sum(r["faith_guard"] for r in rs) / len(rs)
        w = sum(r["faith_guard"] > r["faith_base"] for r in rs)
        l = sum(r["faith_guard"] < r["faith_base"] for r in rs)
        pw = sum(r["pair_guard"] > r["pair_base"] for r in rs)
        pl = sum(r["pair_guard"] < r["pair_base"] for r in rs)
        trig = sum(r["triggered"] for r in rs)
        print(f"{label} n={len(rs)}: 有据性 {fb:.2f}→{fg:.2f} 护栏胜/平/负 {w}/{len(rs)-w-l}/{l} "
              f"p={sign_p(w, l):.3f} | 完整性 {pl and ''}{pw}/{len(rs)-pw-pl}/{pl} | 触发 {trig}")
        return {"n": len(rs), "faith_base": fb, "faith_guard": fg,
                "faith_w": w, "faith_l": l, "faith_p": sign_p(w, l),
                "pair_w": pw, "pair_l": pl, "triggered": trig}

    hit = [r for r in rows if r["both_in_top4"]]
    miss = [r for r in rows if not r["both_in_top4"]]
    print()
    summary = {"all": agg(rows, "全部  "), "both_hit": agg(hit, "双中  "),
               "miss": agg(miss, "缺腿  ")}
    print(f"总耗时 {time.time()-t0:.0f}s")
    OUT.write_text(json.dumps({
        "mode": "v8 缺腿诚实护栏消融：SYSTEM_RAG vs +护栏，同检索同题集，主指标 L3b",
        "guard_prompt": GUARD, "summary": summary, "per_question": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已存 {OUT.name}")


if __name__ == "__main__":
    main()
