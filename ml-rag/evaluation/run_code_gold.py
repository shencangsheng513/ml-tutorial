"""v12 阶段3（零费/低费）：以 .py 为金标的题集 —— 补上 v12 阶段1 欠的另一半。
阶段1 的题集金标全是 markdown，扩库只可能有害，所以那轮测的是"抗干扰"不是"值不值"。
本轮把问题反过来问：题目本身就问代码，金标就是那个 .py 文件。
设计取舍：
  · 题面手写（账号 chat 不可用，且人写的题零费、金标绝对可靠），
    但"人写的题"同样要过逐字闸门——引用不逐字、金标不在库里的一律废题并公布废题率（v7 的教训）。
  · 匹配口径从阶段1 的"子串命中"收紧为"整条相对路径相等"，避免 train.py 这类重名误判命中。
  · 仅文档索引(md-only)在代码题上按构造必近 0，留着它是仪器自检：它若不为 0，说明金标判定漏了。
用法：python evaluation/run_code_gold.py --dry   # 只过闸门+算嵌入缓存命中，零 API
      python evaluation/run_code_gold.py         # 闸门通过后才会真跑检索
"""
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.embeddings import _CACHE_FILE, _md5
from rag.pipeline import retrieve, retrieve_hybrid
from rag.store import VectorStore

EXPANDED = HERE / "index_expanded"
CTRL = HERE / "index_control"
BANK = HERE / "evaluation" / "qa_code_gold.json"
OUT = HERE / "evaluation" / "code_gold_v12.json"
PREV = HERE / "evaluation" / "scale_stress_v12.json"
K_FULL = 8


def norm(doc: str) -> str:
    return doc.replace("\\", "/")


def chunks_by_doc(store: VectorStore) -> dict[str, list[str]]:
    """doc -> 该文档全部块正文（闸门要拿索引里真实存在的文本来核对，而不是信我的记忆）。"""
    bag: dict[str, list[str]] = {}
    for c in store.chunks:
        bag.setdefault(norm(c.doc), []).append(c.text)
    return bag


def gate(items: list[dict], store: VectorStore) -> tuple[list[dict], list[dict], list[dict]]:
    """逐字闸门：金标文档必须在库里、每条 quote 必须逐字出现在该文档中。
    另报"quote 也出现在别的文档"的题——金标不唯一会让 hit@4 虚高。"""
    bag = chunks_by_doc(store)
    kept, dropped, leaky = [], [], []
    for it in items:
        gold = norm(it["doc"])
        if gold not in bag:
            dropped.append({"q": it["q"], "why": "金标不在扩库索引里", "doc": gold})
            continue
        flat_gold = _flat("\n".join(bag[gold]))
        miss = [q for q in it["quote"] if _flat(q) not in flat_gold]
        if miss:
            dropped.append({"q": it["q"], "why": "引用不逐字", "doc": gold, "miss": miss})
            continue
        elsewhere = [d for d, texts in bag.items() if d != gold
                     and any(_flat(q) in _flat("\n".join(texts)) for q in it["quote"])]
        if elsewhere:
            leaky.append({"q": it["q"], "elsewhere": elsewhere})
        kept.append(it)
    return kept, dropped, leaky


def _flat(s: str) -> str:
    return " ".join(s.split())


def first_gold_rank(found: list, gold: str):
    """金标文档第一次出现在第几位（1 起）。整条路径相等才算，重名文件不许蒙对。"""
    for r, (c, _) in enumerate(found, 1):
        if norm(c.doc) == gold:
            return r
    return None


def eval_arm(store: VectorStore, items: list[dict]) -> dict:
    hits_vec = [retrieve(store, it["q"], k=K_FULL) for it in items]
    retrieve_hybrid._cache = {}          # BM25 缓存挂在函数属性上，跨索引必须重置
    hits_hyb = [retrieve_hybrid(store, it["q"], k=K_FULL) for it in items]

    def ctx_hit(found, quotes) -> bool:
        """块级口径：top-4 里有任意一块含该题的引用原文（跨文件也算答对了这个事实）。
        与严格 hit@4 不是包含关系——严格看的是"那份文档进没进"，块级看的是"那句话进没进"；
        同一文档常有多块，进了 top-4 的可能正好是不含引用的那一块。"""
        return any(any(q in _flat(c.text) for q in quotes) for c, _ in found[:4])

    per = []
    for it, hv, hh in zip(items, hits_vec, hits_hyb, strict=True):
        rv, rh = first_gold_rank(hv, norm(it["doc"])), first_gold_rank(hh, norm(it["doc"]))
        qs = [_flat(q) for q in it["quote"]]
        per.append({"q": it["q"], "doc": norm(it["doc"]),
                    "vec_rank": rv, "hyb_rank": rh,
                    "vec_hit4": rv is not None and rv <= 4,
                    "hyb_hit4": rh is not None and rh <= 4,
                    "vec_ctx4": ctx_hit(hv, qs), "hyb_ctx4": ctx_hit(hh, qs),
                    "vec_mrr8": 1.0 / rv if rv else 0.0,
                    "hyb_mrr8": 1.0 / rh if rh else 0.0})
    n = len(per)
    mean = lambda k: round(sum(r[k] for r in per) / n, 3)
    cnt = lambda k: sum(r[k] for r in per)
    return {"n": n, "vec_hit4": cnt("vec_hit4"), "hyb_hit4": cnt("hyb_hit4"),
            "vec_ctx4": cnt("vec_ctx4"), "hyb_ctx4": cnt("hyb_ctx4"),
            "vec_mrr8": mean("vec_mrr8"), "hyb_mrr8": mean("hyb_mrr8"),
            "vec_top1": sum(r["vec_rank"] == 1 for r in per),
            "hyb_top1": sum(r["hyb_rank"] == 1 for r in per),
            "per": per}


def embed_precheck(items: list[dict]) -> tuple[int, int, int]:
    """零费预检：题面里有多少条已在嵌入缓存里（=不用付）。返回 (总题数, 命中数, 缓存条目数)。"""
    cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8")) if _CACHE_FILE.exists() else {}
    return len(items), sum(_md5(it["q"]) in cache for it in items), len(cache)


def main() -> None:
    dry = "--dry" in sys.argv
    if not EXPANDED.exists():
        sys.exit(f"缺扩库索引 {EXPANDED}，先跑 scripts/build_index_expanded.py")
    expanded = VectorStore.load(EXPANDED)
    items = json.loads(BANK.read_text(encoding="utf-8"))
    print(f"题集 {len(items)} 题 | 扩库 {len(expanded.matrix)} 块")

    kept, dropped, leaky = gate(items, expanded)
    print(f"逐字闸门：留 {len(kept)} / 废 {len(dropped)}（废题率 {len(dropped) / len(items):.0%}）")
    for d in dropped:
        print(f"    [废] {d['why']} | {d['q'][:40]} | {d.get('miss', d.get('doc'))}")
    for l in leaky:
        print(f"    [金标不唯一] {l['q'][:40]} ← 也出现在 {l['elsewhere']}")
    if not kept:
        sys.exit("闸门后无题可评，未发起任何 API 调用")

    n, hit, cache_before = embed_precheck(kept)
    print(f"\n嵌入缓存预检：{hit}/{n} 题面已缓存 → 预计付费调用 {n - hit} 次")
    if dry:
        print("--dry：到此为止，未调用 API")
        return

    t0 = time.time()
    arms = {"扩库(+代码)": expanded}
    if CTRL.exists():
        arms["对照(仅文档)"] = VectorStore.load(CTRL)
    res = {k: eval_arm(v, kept) for k, v in arms.items()}
    print()
    for k, r in res.items():
        print(f"{k:12s} n={r['n']:2d} | 严格 hit@4 向量 {r['vec_hit4']} / hybrid {r['hyb_hit4']}"
              f" | 答案块 top-4 向量 {r['vec_ctx4']} / hybrid {r['hyb_ctx4']}"
              f" | MRR@8 {r['vec_mrr8']} / {r['hyb_mrr8']} | top1 {r['vec_top1']} / {r['hyb_top1']}")

    matrix = {}
    if PREV.exists():
        prev = json.loads(PREV.read_text(encoding="utf-8"))
        for s in ("qa_stress80.json", "qa_stress.json", "qa_v3.json", "qa_eval.json"):
            if s not in prev:
                continue
            row = {}
            for arm in ("对照(仅文档)", "扩库(+代码)"):
                a = prev[s].get(arm)
                if a:
                    row[arm] = {"vec_hit4": a["vec_hit4"], "hyb_hit4": a["hyb_hit4"]}
            matrix[s] = row
    print("\n2×2（行=题库金标类型，列=索引内容；数字为 hit@4 hybrid/向量）")
    for s, row in matrix.items():
        cells = " | ".join(f"{a}: {v['hyb_hit4']}/{v['vec_hit4']}" for a, v in row.items())
        print(f"  markdown 金标 {s:20s} {cells}")
    for a, r in res.items():
        print(f"  代码     金标 {'qa_code_gold.json':20s} {a}: {r['hyb_hit4']}/{r['vec_hit4']}")

    _, _, cache_after = embed_precheck(kept)
    paid = cache_after - cache_before
    out = {"n_bank": len(items), "kept": len(kept), "dropped": dropped, "leaky": leaky,
           "embed_expected": n - hit, "embed_paid": paid,
           "arms": {k: {kk: vv for kk, vv in r.items() if kk != "per"} for k, r in res.items()},
           "per_question": {k: r["per"] for k, r in res.items()},
           "md_gold_reference": matrix, "index_blocks": {k: len(v.matrix) for k, v in arms.items()}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 evaluation/code_gold_v12.json（{time.time() - t0:.0f}s，"
          f"付费嵌入 预计 {n - hit} / 实际缓存增量 {paid}）")


if __name__ == "__main__":
    main()

