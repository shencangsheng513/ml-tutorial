"""v4 题集扩容：生成 20 道跨文档多跳题（L1 检索口径用，不考答题）。
每题：从已入库文档里选一对 → 把两篇的代表片段喂给 LLM → 要求出一道"必须同时依据两篇才答得全"的题。
确定性种子可复现；LLM 输出走 answer 缓存，重跑零重复计费。成本约 20 次生成调用。
用法：python scripts/build_multihop20.py
"""
import itertools
import json
import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.config import INDEX_DIR
from rag.pipeline import _cached_chat
from rag.store import VectorStore

OUT = HERE / "evaluation" / "qa_multihop20.json"
N = 20


def doc_snippets(store: VectorStore) -> dict[str, str]:
    by_doc: dict[str, list[str]] = {}
    for c in store.chunks:
        by_doc.setdefault(c.doc, []).append(c.text)
    return {d: "\n".join(t)[:2400] for d, t in sorted(by_doc.items())}


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    snips = doc_snippets(store)
    docs = list(snips)
    rng = random.Random(42)
    pairs = list(itertools.combinations(docs, 2))
    rng.shuffle(pairs)
    items = []
    for doc_a, doc_b in pairs:
        if len(items) >= N:
            break
        prompt = (f"资料甲（《{doc_a}》）：\n{snips[doc_a]}\n\n"
                  f"资料乙（《{doc_b}》）：\n{snips[doc_b]}\n\n"
                  "出一道必须同时依据甲、乙两篇资料才能答完整的中文多跳问题"
                  "（例如比较两者的做法/数字/教训）。"
                  '只输出 JSON：{"q": "问题，不超过60字"}')
        raw = _cached_chat("multihop20|" + doc_a + "|" + doc_b,
                           [{"role": "user", "content": prompt}])
        m = re.search(r'"q"\s*:\s*"([^"]+)"', raw)
        if not m or len(m.group(1)) < 10:
            print(f"  跳过（输出无效）：{doc_a} × {doc_b}")
            continue
        items.append({"q": m.group(1), "docs": [doc_a, doc_b]})
        print(f"  [{len(items)}] {m.group(1)[:48]}…  <- {doc_a} × {doc_b}")
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(items)} 题 -> {OUT.name}")


if __name__ == "__main__":
    main()
