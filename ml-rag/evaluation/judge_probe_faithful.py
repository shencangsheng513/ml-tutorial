"""L3b 有据性判卷的再探针：配对判卷（v3 结论）对"编造声明"不如对"改数字"敏感，
本探针检验 judge_faithful 是否补上这个盲区——每题干净版/污染版各判一次，
看 violations 有没有点名注入的那句资料外断言（而不只是篡改的数字）。
用法：python evaluation/judge_probe_faithful.py   （3 题 × 2 次判卷 = 6 次调用）
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from judge_probe import PROBES, QA  # 复用同一组注入错误
from rag.config import INDEX_DIR
from rag.judge import judge_faithful
from rag.pipeline import answer_with_context, retrieve
from rag.store import VectorStore

# 每题的"定罪关键词"：篡改后的数字 + 编造句里资料绝无仅有的专有词
FAKE_KWS = ["等级保护", "Redis", "DGX"]


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    rows = []
    both_hit = 0
    for (qi, old, new, fake), fake_kw in zip(PROBES, FAKE_KWS):
        item = QA[qi]
        found = retrieve(store, item["q"], k=4)
        good = answer_with_context(item["q"], found)
        bad = good.replace(old, new, 1) + f"\n补充说明：{fake}"
        ctx = "\n\n".join(f"【资料{i+1}·{c.doc}·{c.heading}】\n{c.text}"
                          for i, (c, _) in enumerate(found))
        j_good = judge_faithful(item["q"], item["gold"], good, ctx)
        j_bad = judge_faithful(item["q"], item["gold"], bad, ctx)
        viols = " | ".join(j_bad.get("violations", []))
        hit_num = new in viols
        hit_fake = fake_kw in viols or fake_kw.lower() in viols.lower()
        if hit_num and hit_fake:
            both_hit += 1
        print(f"[probe {qi+1}] 干净 {j_good['score']} → 污染 {j_bad['score']}  "
              f"抓数字{'✓' if hit_num else '✗'} 抓编造{'✓' if hit_fake else '✗'}")
        print(f"    violations: {viols[:120]}")
        rows.append({"q_idx": qi, "clean": j_good["score"], "polluted": j_bad["score"],
                     "hit_digit": hit_num, "hit_fabrication": hit_fake,
                     "violations": j_bad.get("violations", [])})
    out = {"both_hit": both_hit, "n": len(rows),
           "digit_only": sum(1 for r in rows if r["hit_digit"] and not r["hit_fabrication"]),
           "rows": rows}
    print(f"\nL3b 双抓（数字+编造都点名）：{both_hit}/{len(rows)}")
    (HERE / "evaluation" / "judge_probe_faithful_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
