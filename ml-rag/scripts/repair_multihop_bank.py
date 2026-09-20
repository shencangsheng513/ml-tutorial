"""v7.1 题库正向修复：18 道废题按原文档对在当前语料（58 块）上重新出题，替换回 qa_multihop50.json。
废而不修只算诊断——修完再走 build_multihop50_gold.py（老 32 题走缓存零费）+ e2e，看满编 50 的成色。
用法：python scripts/repair_multihop_bank.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.pipeline import _cached_chat
from rag.store import VectorStore

QA = HERE / "evaluation" / "qa_multihop50.json"
GOLD = HERE / "evaluation" / "qa_multihop50_gold.json"


def doc_snippets(store: VectorStore) -> dict[str, str]:
    by_doc: dict[str, list[str]] = {}
    for c in store.chunks:
        by_doc.setdefault(c.doc, []).append(c.text)
    return {d: "\n".join(t)[:2400] for d, t in sorted(by_doc.items())}


def gen(snips, doc_a, doc_b, attempt, used):
    prompt = (f"资料甲（《{doc_a}》）：\n{snips[doc_a]}\n\n"
              f"资料乙（《{doc_b}》）：\n{snips[doc_b]}\n\n"
              "出一道必须同时依据甲、乙两篇资料才能答完整的中文多跳问题"
              "（例如比较两者的做法/数字/教训）。问题里出现的数字、名词必须能在资料里逐字找到。"
              '\n只输出 JSON：{"q": "问题，不超过60字"}')
    key = hashlib.md5(f"mh50fix|{doc_a}|{doc_b}|{attempt}".encode()).hexdigest()
    raw = _cached_chat(key, [{"role": "user", "content": prompt}])
    m = re.search(r'"q"\s*:\s*"([^"]+)"', raw)
    q = m.group(1) if m else None
    return q if q and len(q) >= 10 and q not in used else None


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    snips = doc_snippets(store)
    bank = json.loads(QA.read_text(encoding="utf-8"))
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    kept_q = {k["q"] for k in gold["kept"]}
    drop_by_q = {d["q"]: d for d in gold["dropped"]}
    used = {it["q"] for it in bank}
    fixed = 0
    for i, it in enumerate(bank):
        if it["q"] in kept_q or it["q"] not in drop_by_q:
            continue
        doc_a, doc_b = it["docs"]
        if doc_a not in snips or doc_b not in snips:
            print(f"  [{i+1:02d}] 跳过：文档对不在语料 {doc_a}×{doc_b}")
            continue
        for attempt in range(6):
            q = gen(snips, doc_a, doc_b, attempt, used)
            if q:
                print(f"  [{i+1:02d}] 换题：{it['q'][:30]}…\n         -> {q}")
                bank[i] = {"q": q, "docs": [doc_a, doc_b]}
                used.add(q)
                fixed += 1
                break
        else:
            print(f"  [{i+1:02d}] 6 次尝试未出新题，保留废题待人工")
    QA.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n修复 {fixed}/{len(gold['dropped'])}，题库共 {len(bank)} 题 -> {QA.name}")


if __name__ == "__main__":
    main()
