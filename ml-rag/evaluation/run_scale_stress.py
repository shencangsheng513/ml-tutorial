"""v12 阶段1（零费）：语料扩到 5 倍后重跑检索，看"小语料测不动"的判决会不会被规模推翻。
三臂对照拆变量：生产索引(旧 md 快照) → 仅文档对照(当前 md) → 扩库(md+代码)，
第一跳只反映语料新鲜度，第二跳才是纯规模效应；门槛判定走第二跳。
三套指标一起看，避免饱和指标掩盖差异：hit@4（大盘口径）+ MRR@8（有区分度）+ top-8 里代码块占比。
42 题多跳金库另算双中/缺腿翻转，作为是否付阶段2（端到端）的门槛。
用法：python evaluation/run_scale_stress.py
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.pipeline import retrieve, retrieve_hybrid
from rag.store import VectorStore

EXPANDED = HERE / "index_expanded"
CTRL = HERE / "index_control"
SETS = ["qa_stress80.json", "qa_stress.json", "qa_v3.json", "qa_eval.json"]
GOLD = HERE / "evaluation" / "qa_multihop50_gold.json"
OUT = HERE / "evaluation" / "scale_stress_v12.json"
K_FULL = 8  # 取前8算 MRR，前4即大盘口径


def golds(item) -> list[str]:
    return item.get("docs") or [item.get("doc")]


def is_code(chunk) -> bool:
    return chunk.doc.endswith(".py")


def first_gold_rank(found, gold_list: list[str]):
    """金标文档第一次出现在第几位（1 起），没有则 None。"""
    for r, (c, _) in enumerate(found, 1):
        if any(d in c.doc for d in gold_list):
            return r
    return None


def eval_set(store: VectorStore, items: list) -> dict:
    """一个题集在一份索引上的检索表现：hit@4 / MRR@8 / top-8 里代码块占比。"""
    hits_vec = [retrieve(store, it["q"], k=K_FULL) for it in items]
    retrieve_hybrid._cache = {}          # BM25 缓存在函数属性上，跨索引必须重置
    hits_hyb = [retrieve_hybrid(store, it["q"], k=K_FULL) for it in items]

    per = []
    for it, hv, hh in zip(items, hits_vec, hits_hyb):
        g = golds(it)
        rv, rh = first_gold_rank(hv, g), first_gold_rank(hh, g)
        per.append({
            "q": it["q"],
            "vec_hit4": rv is not None and rv <= 4,
            "hyb_hit4": rh is not None and rh <= 4,
            "vec_mrr8": 1.0 / rv if rv else 0.0,
            "hyb_mrr8": 1.0 / rh if rh else 0.0,
            "hyb_code_share": sum(1 for c, _ in hh if is_code(c)) / len(hh),
            "vec_code_share": sum(1 for c, _ in hv if is_code(c)) / len(hv),
        })
    n = len(per)
    return {"n": n,
            "vec_hit4": sum(r["vec_hit4"] for r in per),
            "hyb_hit4": sum(r["hyb_hit4"] for r in per),
            "vec_mrr8": round(sum(r["vec_mrr8"] for r in per) / n, 3),
            "hyb_mrr8": round(sum(r["hyb_mrr8"] for r in per) / n, 3),
            "vec_code": round(sum(r["vec_code_share"] for r in per) / n, 3),
            "hyb_code": round(sum(r["hyb_code_share"] for r in per) / n, 3),
            "per": per}


def multihop_view(store: VectorStore, items: list) -> list[dict]:
    """42 题多跳金库：双中（docs 全在 top-4）口径，与 v7.1/v8/v11 分组一致。"""
    retrieve_hybrid._cache = {}
    out = []
    for it in items:
        top4 = retrieve_hybrid(store, it["q"], k=4)
        got = {c.doc for c, _ in top4}
        out.append({"q": it["q"], "both": set(it["docs"]) <= got,
                    "code_share": sum(1 for c, _ in top4 if is_code(c)) / len(top4)})
    return out


def main() -> None:
    arms = {"生产(旧快照)": VectorStore.load(INDEX_DIR)}
    if CTRL.exists():
        arms["对照(仅文档)"] = VectorStore.load(CTRL)
    arms["扩库(+代码)"] = VectorStore.load(EXPANDED)
    base, last = next(iter(arms)), list(arms)[-1]
    print("索引：" + "、".join(f"{k} {len(v.matrix)} 块" for k, v in arms.items()) + "\n")
    t0 = time.time()

    report: dict = {"arms": {k: len(v.matrix) for k, v in arms.items()}}
    for name in SETS:
        p = HERE / "evaluation" / name
        if not p.exists():
            continue
        items = json.loads(p.read_text(encoding="utf-8"))
        res = {k: eval_set(v, items) for k, v in arms.items()}
        report[name] = {k: {kk: vv for kk, vv in r.items() if kk != "per"} for k, r in res.items()}
        line = " | ".join(f"{k} hit {r['vec_hit4']}/{r['hyb_hit4']} MRR {r['vec_mrr8']}/{r['hyb_mrr8']}"
                          for k, r in res.items())
        print(f"{name:20s} 题数 {res[base]['n']:3d} | {line}"
              f" | 扩库代码占比 {res[last]['vec_code']}/{res[last]['hyb_code']}")

    items = json.loads(GOLD.read_text(encoding="utf-8"))["kept"]
    mv = {k: multihop_view(v, items) for k, v in arms.items()}
    both = {k: sum(m["both"] for m in rows) for k, rows in mv.items()}
    scale_src = "对照(仅文档)" if "对照(仅文档)" in mv else base
    mo, mn = mv[scale_src], mv[last]
    fixed = [m["q"] for m, n_ in zip(mo, mn) if not m["both"] and n_["both"]]
    broke = [m["q"] for m, n_ in zip(mo, mn) if m["both"] and not n_["both"]]
    report["multihop42"] = {"both": both, "vs_arm": scale_src,
                            "fixed": fixed, "broke": broke,
                            "code_share_new": round(sum(m["code_share"] for m in mn) / len(mn), 3),
                            "rows": {k: [{"q": m["q"], "both": m["both"]} for m in rows]
                                     for k, rows in mv.items()}}
    print("\n42 题多跳双中：" + " -> ".join(f"{k} {both[k]}" for k in arms) +
          f"（{scale_src}→{last} 翻正 {len(fixed)}，翻坏 {len(broke)}）")
    for q in fixed:
        print(f"    [翻正] {q[:44]}")
    for q in broke:
        print(f"    [翻坏] {q[:44]}")

    gate = (len(fixed) + len(broke)) >= 3 or any(
        abs(report[s][last][m] - report[s][scale_src][m]) >= 3
        for s in SETS if s in report for m in ("vec_hit4", "hyb_hit4"))
    report["gate"] = {"pay_phase2": bool(gate), "vs_arm": scale_src,
                      "rule": "双中翻转(翻正+翻坏)>=3 或 任一大盘 hit@4 差>=3 题才付端到端"}
    print(f"\n阶段2 门槛：{'付（检索确有实质变化）' if gate else '不付（检索无实质变化，规模不改判决）'}"
          f"  依据 {report['gate']['rule']}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已存 evaluation/scale_stress_v12.json（{time.time()-t0:.0f}s，全程零 API 费）")


if __name__ == "__main__":
    main()
