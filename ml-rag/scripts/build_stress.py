"""压力题生成：让 LLM 按每篇文档出 9-10 道"答案就在文中"的短题，凑 ~80 题检索压力集。
只用于 L1 召回压测（不考答题），成本 = 每文档 1 次生成调用。
增量幂等：结果存 evaluation/qa_stress80.json，重跑只补没有的文档。
用法：python scripts/build_stress.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from build_index import collect_docs
from rag.config import CHAT_MODEL
from rag.pipeline import _client

OUT = HERE / "evaluation" / "qa_stress80.json"

PROMPT = """你是出卷人。基于下面这篇机器学习教程文档，出 {n} 道检索测试题。
要求：
- 每题的答案必须能在文中找到，优先考具体数字、端口、文件名、结论短句（如"0.4229 是什么"）；
- 题干短（10-25 字）、彼此不重复、不要出"本文讲了什么"这类泛题；
- 只输出 JSON 数组，每项形如 {{"q": "题干"}}，不要输出任何解释。
文档《{doc}》：
{body}"""


def gen_for(doc: str, body: str, n: int = 9) -> list[dict]:
    r = _client().chat.completions.create(
        model=CHAT_MODEL, temperature=0.3, max_tokens=900,
        messages=[{"role": "user", "content": PROMPT.format(n=n, doc=doc, body=body[:6000])}])
    raw = r.choices[0].message.content or ""
    s, e = raw.find("["), raw.rfind("]")
    if s < 0 or e <= s:
        raise ValueError(f"{doc}: 生成结果里没有 JSON 数组")
    items = json.loads(raw[s:e + 1])
    return [{"q": it["q"].strip(), "docs": [doc]} for it in items if it.get("q", "").strip()][:n]


def main() -> None:
    done: dict[str, list[dict]] = {}
    if OUT.exists():
        for row in json.loads(OUT.read_text(encoding="utf-8")):
            done.setdefault(row["docs"][0], []).append(row)
    docs = collect_docs()
    for rel, body in docs.items():
        if rel in done:
            continue
        rows = gen_for(rel, body)
        done[rel] = rows
        OUT.write_text(json.dumps([x for v in done.values() for x in v],
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {rel}: +{len(rows)} 题（累计 {sum(len(v) for v in done.values())}）")
    print(f"压力集已存 {OUT.name}")


if __name__ == "__main__":
    main()
