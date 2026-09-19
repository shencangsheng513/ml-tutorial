"""判卷官灵敏度探针：用"已知坏答案"校准 LLM-judge。
方法：取 v2 缓存的真 RAG 答案，人工注入两类错误——
  ① 改一个关键数字（0.4229→0.4329 这类）；② 加一句资料里根本没有的斩钉截铁断言。
如果 judge 给"污染版"照样打 10 → 判卷官失灵（量表再严也只是摆设）；
如果显著扣分 → 10/13 满分是真·有据性，v2 的 RAG 答案质量确实过硬。
用法：python evaluation/judge_probe.py   （3 题 × 1 次判卷 = 3 次调用）
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.judge import judge_pair
from rag.pipeline import answer_with_context, retrieve
from rag.store import VectorStore

QA = json.loads((HERE / "evaluation" / "qa_eval.json").read_text(encoding="utf-8"))

# (题号, 原文片段 -> 篡改片段, 注入的资料外断言)
PROBES = [
    (0, "0.4229", "0.4329", "该模型已通过国家信息安全等级保护三级认证。"),
    (7, "8504", "8505", "应用后端使用 Redis 做会话缓存，命中率 98%。"),
    (9, "0.9798", "0.9978", "CNN 训练时使用了 NVIDIA DGX 集群，成本约 3000 美元。"),
]


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    rows = []
    for qi, old, new, fake in PROBES:
        item = QA[qi]
        found = retrieve(store, item["q"], k=4)
        good = answer_with_context(item["q"], found)
        if old not in good:
            print(f"[probe {qi+1}] 跳过：缓存答案里没有锚点 {old}")
            continue
        bad = good.replace(old, new, 1) + f"\n补充说明：{fake}"
        ctx = "\n\n".join(f"【资料{i+1}·{c.doc}·{c.heading}】\n{c.text}"
                          for i, (c, _) in enumerate(found))
        j = judge_pair(item["q"], item["gold"], good, bad, context=ctx)
        s_good, s_bad = j["A"], j["B"]
        verdict = "抓到 ✓" if s_bad < s_good else "没抓到 ✗"
        print(f"[probe {qi+1}] 原题「{item['q'][:24]}…」 干净版 {s_good} vs 污染版 {s_bad}  {verdict}")
        print(f"    扣分理由：{j.get('reason', '')[:80]}")
        rows.append({"q_idx": qi, "clean": s_good, "polluted": s_bad,
                     "deductions_B": j.get("deductions", {}).get("B", [])})
    caught = sum(1 for r in rows if r["polluted"] < r["clean"])
    print(f"\n判卷官灵敏度：{caught}/{len(rows)} 抓到注入错误")
    (HERE / "evaluation" / "judge_probe_results.json").write_text(
        json.dumps({"caught": caught, "n": len(rows), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
