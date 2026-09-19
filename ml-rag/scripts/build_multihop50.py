"""v5 题集扩容：20 → 50 道跨文档多跳题（L1 检索口径），验证 v4 的 +9% 分解增益是否稳。
10 篇文档只有 C(10,2)=45 对，第一轮沿用 build_multihop20 的缓存键（老题零重付），
不足 50 时进入第二轮"换角度"（比较维度写进 prompt，产生新缓存键）。
用法：python scripts/build_multihop50.py
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

OUT = HERE / "evaluation" / "qa_multihop50.json"
N = 50
ANGLES = ["", "\n本轮比较角度：只聚焦两者【方法论/流程步骤】的差异，别问具体指标数字。"]


def doc_snippets(store: VectorStore) -> dict[str, str]:
    by_doc: dict[str, list[str]] = {}
    for c in store.chunks:
        by_doc.setdefault(c.doc, []).append(c.text)
    return {d: "\n".join(t)[:2400] for d, t in sorted(by_doc.items())}


def gen_pair(snips, doc_a, doc_b, angle, cache_i):
    prompt = (f"资料甲（《{doc_a}》）：\n{snips[doc_a]}\n\n"
              f"资料乙（《{doc_b}》）：\n{snips[doc_b]}\n\n"
              "出一道必须同时依据甲、乙两篇资料才能答完整的中文多跳问题"
              "（例如比较两者的做法/数字/教训）。" + angle +
              '\n只输出 JSON：{"q": "问题，不超过60字"}')
    raw = _cached_chat(f"multihop20|{doc_a}|{doc_b}" + (f"|{cache_i}" if cache_i else ""),
                       [{"role": "user", "content": prompt}])
    m = re.search(r'"q"\s*:\s*"([^"]+)"', raw)
    return m.group(1) if m and len(m.group(1)) >= 10 else None


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    snips = doc_snippets(store)
    rng = random.Random(42)
    pairs = list(itertools.combinations(sorted(snips), 2))
    rng.shuffle(pairs)
    items, seen_q = [], set()
    for rnd in range(2):
        for doc_a, doc_b in pairs:
            if len(items) >= N:
                break
            q = gen_pair(snips, doc_a, doc_b, ANGLES[rnd], rnd)
            if not q or q in seen_q:
                print(f"  跳过（无效/重复）：{doc_a} × {doc_b} 第{rnd+1}轮")
                continue
            seen_q.add(q)
            items.append({"q": q, "docs": [doc_a, doc_b]})
            print(f"  [{len(items)}] {q[:48]}…  <- {doc_a} × {doc_b}")
        if len(items) < N:
            print(f"  第{rnd+1}轮结束仅 {len(items)} 题，继续下一轮")
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(items)} 题 -> {OUT.name}")


if __name__ == "__main__":
    main()
