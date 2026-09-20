"""v11：换变量兑现——检索深度 k=4 vs k=6 整配置消融。
v8/v9 把生成端措辞封死，欠的债认给"检索端覆盖"；预检显示缺腿 20 题里 k=6 能翻正 10 题双中。
本轮同题库同 SYSTEM_RAG 只换 k：基线臂(k=4)答案+判卷全走 v7.1 缓存零费，k=6 臂新调用。
主指标：L3b 有据性（分组配对符号检验）+ v10 数字核验器报警数（零费）；
副指标：L3a 完整性 parity 配对——防"资料塞多了稀释"式倒退。
判决纪律：测不出增益 k 就不进主流程，维持 4。
用法：python evaluation/run_depth6_ab.py
"""
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.attribution import ungrounded_numbers
from rag.config import INDEX_DIR
from rag.judge import judge_faithful, judge_pair
from rag.pipeline import answer_with_context, build_context, retrieve_hybrid
from rag.store import VectorStore

GOLD = HERE / "evaluation" / "qa_multihop50_gold.json"
OUT = HERE / "evaluation" / "depth6_ab_v11.json"


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    items = json.loads(GOLD.read_text(encoding="utf-8"))["kept"]
    print(f"v11 检索深度消融：{len(items)} 题，k=4 基线走缓存、k=6 臂新调用\n")
    t0 = time.time()
    rows = []
    for i, item in enumerate(items):
        hits4 = retrieve_hybrid(store, item["q"], k=4)
        hits6 = retrieve_hybrid(store, item["q"], k=6)
        docs = set(item["docs"])
        both4, both6 = docs <= {c.doc for c, _ in hits4}, docs <= {c.doc for c, _ in hits6}
        ctx4, ctx6 = build_context(hits4, "RRF分"), build_context(hits6, "RRF分")
        a4 = answer_with_context(item["q"], hits4, score_label="RRF分")
        a6 = answer_with_context(item["q"], hits6, score_label="RRF分")
        f4 = judge_faithful(item["q"], item["gold"], a4, ctx4)
        f6 = judge_faithful(item["q"], item["gold"], a6, ctx6)
        x, y = (a4, a6) if i % 2 else (a6, a4)  # 奇偶换位防位置偏置
        j = judge_pair(item["q"], item["gold"], x, y)
        s6, s4 = (j["B"], j["A"]) if i % 2 else (j["A"], j["B"])
        g4, g6 = ungrounded_numbers(a4, hits4, item["q"]), ungrounded_numbers(a6, hits6, item["q"])
        rows.append({"q": item["q"], "both4": both4, "both6": both6,
                     "faith4": float(f4["score"]), "faith6": float(f6["score"]),
                     "viol4": len(f4.get("violations") or []), "viol6": len(f6.get("violations") or []),
                     "pair6": s6, "pair4": s4, "flags4": g4, "flags6": g6})
        flip = "翻正" if (not both4 and both6) else ("双中" if both4 else "仍缺")
        print(f"  [{i+1:02d}] {flip} 有据 {f4['score']:.1f}→{f6['score']:.1f} 无据 {len(f4.get('violations') or [])}→{len(f6.get('violations') or [])}"
              f" 完整 {s4:.1f}→{s6:.1f} 核验 {len(g4)}→{len(g6)}  {item['q'][:20]}")

    groups = {"全部": rows, "双中(k4)": [r for r in rows if r["both4"]],
              "缺腿(k4)": [r for r in rows if not r["both4"]]}

    def agg(rs, label):
        if not rs:
            return None
        n = len(rs)
        f4s = sum(r["faith4"] for r in rs) / n
        f6s = sum(r["faith6"] for r in rs) / n
        w = sum(r["faith6"] > r["faith4"] for r in rs)
        l = sum(r["faith6"] < r["faith4"] for r in rs)
        pw = sum(r["pair6"] > r["pair4"] for r in rs)
        pl = sum(r["pair6"] < r["pair4"] for r in rs)
        v4 = sum(r["viol4"] for r in rs)
        v6 = sum(r["viol6"] for r in rs)
        fl4 = sum(len(r["flags4"]) for r in rs)
        fl6 = sum(len(r["flags6"]) for r in rs)
        q4 = sum(bool(r["flags4"]) for r in rs)
        q6 = sum(bool(r["flags6"]) for r in rs)
        print(f"{label} n={n}: 有据性 {f4s:.2f}→{f6s:.2f} 胜/平/负 {w}/{n-w-l}/{l} p={sign_p(w, l):.3f}"
              f" | 完整性 {pw}/{n-pw-pl}/{pl} | 无据断言 {v4}→{v6} | 核验器 {fl4}({q4}题)→{fl6}({q6}题)")
        return {"n": n, "faith4": f4s, "faith6": f6s, "faith_w": w, "faith_l": l,
                "faith_p": sign_p(w, l), "pair_w": pw, "pair_l": pl,
                "viol4": v4, "viol6": v6, "flags4": fl4, "flags6": fl6,
                "flagged_q4": q4, "flagged_q6": q6}

    print()
    summary = {k: agg(v, k.ljust(5)) for k, v in groups.items()}
    print(f"总耗时 {time.time()-t0:.0f}s")
    OUT.write_text(json.dumps({
        "mode": "v11 检索深度消融：k=4(基线走缓存) vs k=6(新调用)，同题库同提示词",
        "summary": summary, "per_question": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"明细已存 {OUT.name}")


if __name__ == "__main__":
    main()
