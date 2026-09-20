"""v7 前置：给 50 题多跳集补"引文式 gold 要点"——判卷官（L3a/L3b）没有标准要点就是裸判。
硬规则：要点必须是资料原句（截取处用 … 连接），离线逐片段回查语料验证，查无此句即废题；
两篇资料合起来也答不了的题标 answerable=false 一并废掉——顺便检测题库腐烂（v6 换索引后题源可能已改）。
用法：python scripts/build_multihop50_gold.py   （50 次 LLM 调用，走缓存重跑零费）
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
OUT = HERE / "evaluation" / "qa_multihop50_gold.json"
DOC_CAP = 12000
FRAG_MIN = 6  # 短于此的引用片段没有回查价值，视为无效


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def doc_texts(store: VectorStore) -> dict[str, str]:
    by_doc: dict[str, list[str]] = {}
    for c in store.chunks:
        by_doc.setdefault(c.doc, []).append(c.text)
    return {d: "\n".join(t)[:DOC_CAP] for d, t in by_doc.items()}


def extract_json(raw: str) -> dict | None:
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return None


def verify(gold: list[str], corpus_norm: str) -> list[str]:
    """返回验证失败的片段列表：每个引用片段（… 分隔）必须逐字存在于语料。"""
    bad = []
    for point in gold:
        for frag in re.split(r"…|\.\.\.", point):
            f = norm(frag)
            if len(f) >= FRAG_MIN and f not in corpus_norm:
                bad.append(frag.strip()[:30])
    return bad


def ask_gold(it, texts, suffix, cache_tag):
    doc_a, doc_b = it["docs"]
    prompt = (f"资料甲（《{doc_a}》）：\n{texts[doc_a]}\n\n"
              f"资料乙（《{doc_b}》）：\n{texts[doc_b]}\n\n"
              f"问题：{it['q']}\n"
              "任务：只依据以上资料，给出回答此问题所需的\"标准要点\"（3-5 条）。\n"
              "硬规则：每条要点必须直接引用资料原句（可截取关键半句，多处引用之间用 … 连接），"
              "不得改写数字、不得引入资料外信息。若两篇资料合起来也答不了此问题，"
              'answerable 填 false。\n只输出 JSON：{"answerable": true, "gold": ["要点1", "…"]}'
              + suffix)
    key = hashlib.md5((f"gold50{cache_tag}|" + it["q"]).encode("utf-8")).hexdigest()
    raw = _cached_chat(key, [{"role": "user", "content": prompt}])
    out = extract_json(raw) or {}
    return out.get("gold") or [] if out.get("answerable") else None


def try_gold(it, texts, suffix="", cache_tag=""):
    """返回 (gold, bad_frags)：gold=None 表示答不了；bad_frags 非空表示引文回查失败。"""
    doc_a, doc_b = it["docs"]
    gold = ask_gold(it, texts, suffix, cache_tag)
    if gold is None:
        return None, []
    corpus_norm = norm(texts[doc_a] + texts[doc_b])
    return gold, verify(gold, corpus_norm)


def main() -> None:
    store = VectorStore.load(INDEX_DIR)
    texts = doc_texts(store)
    items = json.loads(QA.read_text(encoding="utf-8"))
    kept, dropped = [], []
    retry_pool = []
    for i, it in enumerate(items):
        if it["docs"][0] not in texts or it["docs"][1] not in texts:
            dropped.append({"q": it["q"], "why": "题源文档已不在语料"})
            print(f"  [{i+1:02d}] ✗ 题源缺失  {it['q'][:36]}")
            continue
        gold, bad = try_gold(it, texts)
        if gold is None:
            dropped.append({"q": it["q"], "why": "资料答不了/无要点"})
            print(f"  [{i+1:02d}] ✗ 答不了    {it['q'][:36]}")
        elif bad:
            retry_pool.append((i, it, bad))
            print(f"  [{i+1:02d}] ↻ 引文回查失败，进重试  {it['q'][:36]}")
        else:
            kept.append({"q": it["q"], "docs": it["docs"], "gold": gold})
            print(f"  [{i+1:02d}] ✓ {len(gold)} 要点  {it['q'][:36]}")

    print(f"\n第二轮（带失败片段反馈的逐字重试）：{len(retry_pool)} 题")
    for i, it, bad in retry_pool:
        suffix = ("\n\n上一轮你给出的要点里，以下片段在资料中逐字查不到，全部作废重做："
                  + "；".join(bad[:3]) +
                  "\n这次必须从资料里逐字复制句子（连字符、空格、标点都不许动），"
                  "不要自加\"甲中：\"之类前缀。")
        gold, bad2 = try_gold(it, texts, suffix, "r2")
        if gold is None or bad2:
            dropped.append({"q": it["q"], "why": "引文回查失败(重试后)",
                            "bad_frags": bad2})
            print(f"  [{i+1:02d}] ✗ 仍造假    {it['q'][:36]}")
        else:
            kept.append({"q": it["q"], "docs": it["docs"], "gold": gold})
            print(f"  [{i+1:02d}] ✓ 救回 {len(gold)} 要点  {it['q'][:36]}")
    kept.sort(key=lambda r: [x["q"] for x in items].index(r["q"]))
    OUT.write_text(json.dumps({"kept": kept, "dropped": dropped},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n留 {len(kept)} / 废 {len(dropped)} -> {OUT.name}")


if __name__ == "__main__":
    main()
