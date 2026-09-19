"""多跳查询分解：LLM 把跨文档问题拆成可独立检索的子问题，逐个子问题混合检索后做覆盖优先融合。

v3 的结论：多跳题上 vec/bm25/hybrid 全方法 MRR@6 同为 0.438——第二篇文档在单查询排名里
埋得太深，检索器本身修不动。分解把"两跳"拆成两次"单跳"，每个子查询保证一个头部槽位。
"""
import hashlib
import json
import re

from .bm25 import rrf_fuse
from .config import CHAT_MODEL, INDEX_DIR
from .embeddings import embed_texts
from .pipeline import _client
from .store import VectorStore

_DECOMP_CACHE = INDEX_DIR / "decompose_cache.json"
_PREFIX_RE = re.compile(r"^[\s\-•*0-9.、)）]+")


def parse_subqueries(raw: str) -> list[str]:
    """LLM 输出按行清洗：去编号前缀、去空行、去重、最多 3 条。纯函数可离线测。"""
    seen: set[str] = set()
    out: list[str] = []
    for line in raw.splitlines():
        s = _PREFIX_RE.sub("", line).strip().rstrip("：:")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:3]


def subqueries(question: str) -> list[str]:
    """原问题永远排第一（保底退化为 v3 的 hybrid）；分解结果走磁盘缓存。"""
    cache = json.loads(_DECOMP_CACHE.read_text(encoding="utf-8")) if _DECOMP_CACHE.exists() else {}
    key = hashlib.md5(question.encode("utf-8")).hexdigest()
    if key not in cache:
        prompt = ("把下面的问题拆成2-3个可以独立去文档库检索的子问题；"
                  "如果问题本身只需一条资料就能回答，就原样返回。"
                  "每行一个子问题，不要编号、不要解释。\n问题：" + question)
        r = _client().chat.completions.create(
            model=CHAT_MODEL, temperature=0, max_tokens=200,
            messages=[{"role": "user", "content": prompt}])
        parts = parse_subqueries(r.choices[0].message.content.strip())
        cache[key] = parts
        _DECOMP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _DECOMP_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    parts = [q for q in cache[key] if q]
    return [question] + [q for q in parts if q != question]


def retrieve_decomposed(store: VectorStore, question: str, k: int = 4, cand: int = 8,
                        _bm=None, reserve: bool = True) -> list[tuple[int, float]]:
    """返回 [(chunk索引, 融合分)]，长度 ≤k。

    reserve=True：每个子问题先保一个 top-1 槽位（覆盖保证），其余按 RRF 补满；
    reserve=False：纯多查询 RRF（让共识块自然上浮），两档供 A/B。
    """
    if not hasattr(retrieve_decomposed, "_cache"):
        retrieve_decomposed._cache = {}
    if _bm is None:
        if "bm" not in retrieve_decomposed._cache:
            from .bm25 import BM25
            retrieve_decomposed._cache["bm"] = BM25([c.text for c in store.chunks])
        _bm = retrieve_decomposed._cache["bm"]
    lists: list[list[int]] = []
    for q in subqueries(question):
        qv = embed_texts([q], verbose=False)[0]
        vec = [i for i, _ in store.search_all(qv, k=cand)]
        bm = [i for i, _ in _bm.search(q, k=cand)]
        hy = [i for i, _ in sorted(rrf_fuse([vec, bm]).items(), key=lambda x: -x[1])]
        lists.append(hy)
    picked: list[int] = []
    if reserve:
        for hy in lists:  # 每个子查询保一个 top-1 槽位
            if hy and hy[0] not in picked:
                picked.append(hy[0])
    pool = sorted(rrf_fuse(lists).items(), key=lambda x: -x[1])
    for i, _ in pool:
        if len(picked) >= k:
            break
        if i not in picked:
            picked.append(i)
    score = dict(pool)
    return [(i, score.get(i, 0.0)) for i in picked[:k]]
