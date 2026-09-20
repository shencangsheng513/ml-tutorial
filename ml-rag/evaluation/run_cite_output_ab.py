"""v9：输出端结构化——逐句标注【资料i】出处，让"没据"无处藏。
v8 证明软提示（"没据就写资料不足"）管不住缺腿幻觉；本实验换硬约束：
要求每句末尾挂出处编号，未挂/挂错的句子在离线归因核验里现形（句中数字必须能在被引块里逐字找到）。
口径：同 42 题同检索，基线臂走 v7/v7.1 缓存，引用臂新调用；主指标 L3b 有据性（分组配对符号检验），
副指标 L3a 完整性 + 离线归因率（数字级出处核验，零成本）。
用法：python evaluation/run_cite_output_ab.py
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

from rag.config import INDEX_DIR
from rag.judge import judge_faithful, judge_pair
from rag.pipeline import answer_with_context, build_context, retrieve_hybrid
from rag.store import VectorStore

SYSTEM_CITE = (
    "你是这个机器学习教程仓库的助教。把回答拆成短句，每句末尾必须标注该句依据的【资料】编号，"
    "如【资料2】或【资料1】【资料4】。句中的数字必须与所标资料的原文一字不差。"
    "写不出出处的句子就不要写，宁可短不可编。用中文，不客套。")
K = 4
GOLD = HERE / "evaluation" / "qa_multihop50_gold.json"
OUT = HERE / "evaluation" / "cite_output_ab_v9.json"


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def attribution_check(answer: str, hits) -> tuple[int, int, int]:
    """离线归因核验：返回 (带数字句总数, 可归因数, 出处编号有效句数)。
    可归因 = 句中每个数字（≥2位或带小数）能在其标注的资料块原文里找到。"""
    texts = {i + 1: c.text for i, (c, _) in enumerate(hits)}
    num_claim = num_ok = idx_ok = 0
    for sent in re.split(r"(?<=[。；])", answer):
        refs = [int(n) for n in re.findall(r"【资料(\d+)】", sent)]
        if not refs:
            continue
        if all(r in texts for r in refs):
            idx_ok += 1
        nums = re.findall(r"\d+\.\d+|\d{2,}", sent)
        if not nums:
            continue
        num_claim += 1
        src = "".join(texts[r] for r in refs if r in texts)
        if all(n in src for n in nums):
            num_ok += 1
    return num_claim, num_ok, idx_ok


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    items = json.loads(GOLD.read_text(encoding="utf-8"))["kept"]
    print(f"v9 逐句出处消融：{len(items)} 题，基线臂走缓存、引用臂新调用\n")
    t0 = time.time()
    rows = []
    for i, item in enumerate(items):
        found = retrieve_hybrid(store, item["q"], k=K)
        both = set(item["docs"]) <= {c.doc for c, _ in found}
        ctx = build_context(found, "RRF分")
        a_base = answer_with_context(item["q"], found, score_label="RRF分")
        a_cite = answer_with_context(item["q"], found, score_label="RRF分",
                                     system=SYSTEM_CITE)
        f_base = judge_faithful(item["q"], item["gold"], a_base, ctx)
        f_cite = judge_faithful(item["q"], item["gold"], a_cite, ctx)
        x, y = (a_base, a_cite) if i % 2 else (a_cite, a_base)  # 奇偶换位防位置偏置
        j = judge_pair(item["q"], item["gold"], x, y)
        s_cite, s_base = (j["B"], j["A"]) if i % 2 else (j["A"], j["B"])
        n_cl, n_ok, n_idx = attribution_check(a_cite, found)
        n_sent = len(re.findall(r"【资料\d+】", a_cite))
        rows.append({"q": item["q"], "both_in_top4": both,
                     "faith_base": float(f_base["score"]), "faith_cite": float(f_cite["score"]),
                     "pair_cite": s_cite, "pair_base": s_base,
                     "cited_sents": n_sent, "num_claims": n_cl, "num_ok": n_ok,
                     "idx_ok": n_idx})
        print(f"  [{i+1:02d}] {'双中' if both else '缺腿'} 有据性 {f_base['score']:.1f}→{f_cite['score']:.1f}  "
              f"完整性 {s_base:.1f}→{s_cite:.1f}  归因 {n_ok}/{n_cl} 标句{n_sent}  {item['q'][:20]}")

    hit = [r for r in rows if r["both_in_top4"]]
    miss = [r for r in rows if not r["both_in_top4"]]

    def agg(rs, label):
        if not rs:
            return None
        fb = sum(r["faith_base"] for r in rs) / len(rs)
        fc = sum(r["faith_cite"] for r in rs) / len(rs)
        w = sum(r["faith_cite"] > r["faith_base"] for r in rs)
        l = sum(r["faith_cite"] < r["faith_base"] for r in rs)
        pw = sum(r["pair_cite"] > r["pair_base"] for r in rs)
        pl = sum(r["pair_cite"] < r["pair_base"] for r in rs)
        cl = sum(r["num_claims"] for r in rs)
        ok = sum(r["num_ok"] for r in rs)
        print(f"{label} n={len(rs)}: 有据性 {fb:.2f}→{fc:.2f} 胜/平/负 {w}/{len(rs)-w-l}/{l} "
              f"p={sign_p(w, l):.3f} | 完整性 {pw}/{len(rs)-pw-pl}/{pl} | 归因率 {ok}/{cl}")
        return {"n": len(rs), "faith_base": fb, "faith_cite": fc, "faith_w": w,
                "faith_l": l, "faith_p": sign_p(w, l), "pair_w": pw, "pair_l": pl,
                "num_claims": cl, "num_ok": ok}

    print()
    summary = {"all": agg(rows, "全部  "), "both_hit": agg(hit, "双中  "),
               "miss": agg(miss, "缺腿  ")}
    print(f"总耗时 {time.time()-t0:.0f}s")
    OUT.write_text(json.dumps({
        "mode": "v9 逐句出处标注消融：SYSTEM_RAG vs SYSTEM_CITE，主指标 L3b+离线归因率",
        "cite_prompt": SYSTEM_CITE, "summary": summary, "per_question": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已存 {OUT.name}")


if __name__ == "__main__":
    main()
