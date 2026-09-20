"""v12 阶段2（付费）：把 42 题多跳金库搬到 5 倍规模索引上重跑端到端，检验 v7 结论的外部效度。
两条判决问题：①"缺腿扣有据性"在规模下是放大还是被稀释淹没；②检索掉到 66% 后 RAG 还压得住裸考吗。
基线（小索引臂）答案+判卷全走 v7 缓存零费；只有扩库臂的 42 答案 + 42 有据性 + 42 配对是新调用。
裸考臂索引无关、必然命中缓存：付前先数缓存键，付后再数，新增条数即真实开销（自评纪律）。
用法：python evaluation/run_scale_e2e.py
"""
import json
import math
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.attribution import ungrounded_numbers
from rag.config import INDEX_DIR
from rag.judge import judge_faithful, judge_pair
from rag.pipeline import answer_bare, answer_with_context, build_context, retrieve_hybrid
from rag.store import VectorStore

EXPANDED = HERE / "index_expanded"
GOLD = HERE / "evaluation" / "qa_multihop50_gold.json"
BASE = HERE / "evaluation" / "multihop50_e2e_v7.json"
OUT = HERE / "evaluation" / "scale_e2e_v12.json"
ANSWER_CACHE = INDEX_DIR / "answer_cache.json"
JUDGE_CACHE = INDEX_DIR / "judge_cache.json"


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cache_sizes() -> tuple[int, int]:
    def n(p):
        return len(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else 0
    return n(ANSWER_CACHE), n(JUDGE_CACHE)


def main() -> None:
    store = VectorStore.load(EXPANDED)
    base_rows = {r["q"]: r for r in json.loads(BASE.read_text(encoding="utf-8"))["per_question"]}
    items = json.loads(GOLD.read_text(encoding="utf-8"))["kept"]
    a0, j0 = cache_sizes()
    print(f"扩库 e2e：{len(items)} 题 × 索引 {len(store.chunks)} 块；"
          f"缓存现有 答案 {a0} / 判卷 {j0} 条\n")
    t0 = time.time()
    rows = []
    for i, item in enumerate(items):
        b = base_rows[item["q"]]
        hits = retrieve_hybrid(store, item["q"], k=4)
        both = set(item["docs"]) <= {c.doc for c, _ in hits}
        a_exp = answer_with_context(item["q"], hits, score_label="RRF分")
        a_bare = answer_bare(item["q"])
        f = judge_faithful(item["q"], item["gold"], a_exp, build_context(hits, "RRF分"))
        x, y = (a_bare, a_exp) if i % 2 else (a_exp, a_bare)   # 奇偶换位防位置偏置
        j = judge_pair(item["q"], item["gold"], x, y)
        s_exp, s_bare = (j["B"], j["A"]) if i % 2 else (j["A"], j["B"])
        rows.append({"q": item["q"], "both": both, "both_v7": b["both_in_top4"],
                     "faith_exp": float(f["score"]), "faith_v7": float(b["faithful"]),
                     "viol_exp": len(f.get("violations") or []),
                     "rag_exp": s_exp, "rag_v7": b["rag"], "bare": s_bare,
                     "flags_exp": ungrounded_numbers(a_exp, hits, item["q"]),
                     "reason": re.sub(r"(?<![A-Za-z0-9])[AB](?![A-Za-z0-9])",
                                      lambda m: "B" if m.group() == "A" else "A",
                                      j.get("reason", "")) if i % 2 else j.get("reason", "")})
        r = rows[-1]
        print(f"  [{i+1:02d}] {'双中' if both else '缺腿'}"
              f"{'（v7双中→现缺腿）' if b['both_in_top4'] and not both else ''}"
              f" 有据 {r['faith_v7']:.1f}→{r['faith_exp']:.1f} 无据 {r['viol_exp']}"
              f" 配对 RAG {s_exp:.1f}/裸 {s_bare:.1f}（v7 RAG {r['rag_v7']:.1f}）  {item['q'][:22]}")

    def mean(rs, k):
        return sum(r[k] for r in rs) / len(rs) if rs else float("nan")

    groups = {"全部": rows, "双中(扩库)": [r for r in rows if r["both"]],
              "缺腿(扩库)": [r for r in rows if not r["both"]]}
    summary = {}
    for label, rs in groups.items():
        if not rs:
            continue
        w = sum(r["rag_exp"] > r["bare"] for r in rs)
        l = sum(r["rag_exp"] < r["bare"] for r in rs)
        wn = sum(r["faith_exp"] < r["faith_v7"] for r in rs)   # 规模后退步题数
        better = sum(r["faith_exp"] > r["faith_v7"] for r in rs)
        summary[label] = {"n": len(rs), "faith_exp": round(mean(rs, "faith_exp"), 2),
                          "faith_v7": round(mean(rs, "faith_v7"), 2),
                          "rag_exp": round(mean(rs, "rag_exp"), 2), "bare": round(mean(rs, "bare"), 2),
                          "pair": f"{w}/{len(rs)-w-l}/{l}", "pair_p": round(sign_p(w, l), 4),
                          "faith_down": f"{wn}↓/{better}↑", "faith_p": round(sign_p(better, wn), 4),
                          "flagged": sum(1 for r in rs if r["flags_exp"])}
        s = summary[label]
        print(f"\n{label} {s['n']} 题：有据性 {s['faith_v7']}→{s['faith_exp']}"
              f"（退化 {s['faith_down']}，p={s['faith_p']}）| 配对 RAG {s['rag_exp']} vs 裸考 {s['bare']}"
              f" 胜/平/负 {s['pair']} p={s['pair_p']} | 数字核验报警 {s['flagged']} 题")
    both_r = [r for r in rows if r["both"]]
    miss_r = [r for r in rows if not r["both"]]
    gap = mean(both_r, "faith_exp") - mean(miss_r, "faith_exp")
    summary["缺腿惩罚"] = {"exp": round(gap, 2), "v7": round(
        mean([r for r in rows if r["both_v7"]], "faith_v7")
        - mean([r for r in rows if not r["both_v7"]], "faith_v7"), 2)}
    print(f"\n缺腿惩罚（双中均值 - 缺腿均值）：v7 {summary['缺腿惩罚']['v7']} -> 扩库 {summary['缺腿惩罚']['exp']}")

    a1, j1 = cache_sizes()
    print(f"缓存增量：答案 +{a1-a0}、判卷 +{j1-j0}（= 实际付费调用数，裸考应 +0）")
    OUT.write_text(json.dumps({"mode": "v12 阶段2：42 题多跳金库 × 扩库索引 e2e，基线=v7 小索引（缓存零费）",
                               "index_chunks": len(store.chunks), "summary": summary,
                               "paid": {"answers": a1 - a0, "judges": j1 - j0},
                               "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已存 {OUT.name}（{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
