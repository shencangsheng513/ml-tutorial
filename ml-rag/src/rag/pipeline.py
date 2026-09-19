"""检索 + 生成 的两条路：有上下文（RAG）与裸考（无上下文），供对照实验用。"""
import hashlib
import json

from openai import OpenAI

from .config import CHAT_MODEL, INDEX_DIR, load_credentials
from .embeddings import embed_texts
from .store import VectorStore


def _client() -> OpenAI:
    key, base = load_credentials()
    return OpenAI(api_key=key, base_url=base, timeout=120)


_ANSWER_CACHE = INDEX_DIR / "answer_cache.json"


def _cached_chat(prompt_key: str, messages) -> str:
    """答案落盘缓存：换判卷官重评时不重新生成，控制变量也省钱。"""
    cache = json.loads(_ANSWER_CACHE.read_text(encoding="utf-8")) if _ANSWER_CACHE.exists() else {}
    if prompt_key in cache:
        return cache[prompt_key]
    r = _client().chat.completions.create(model=CHAT_MODEL, temperature=0, max_tokens=800,
                                          messages=messages)
    out = r.choices[0].message.content.strip()
    cache[prompt_key] = out
    _ANSWER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _ANSWER_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out


def retrieve(store: VectorStore, question: str, k: int = 4):
    qv = embed_texts([question], verbose=False)[0]
    return store.search(qv, k=k)


def retrieve_hybrid(store: VectorStore, question: str, k: int = 4, cand: int = 8):
    """向量 + BM25 双路召回，RRF 名次融合。两路各取前 cand，融合后取前 k。"""
    if not hasattr(retrieve_hybrid, "_cache"):
        retrieve_hybrid._cache = {}
    _bm25_cache = retrieve_hybrid._cache
    from .bm25 import BM25, rrf_fuse

    if "idx" not in _bm25_cache:
        _bm25_cache["idx"] = BM25([c.text for c in store.chunks])
    qv = embed_texts([question], verbose=False)[0]
    vec_rank = [i for i, _ in store.search_all(qv, k=cand)]
    bm_rank = [i for i, _ in _bm25_cache["idx"].search(question, k=cand)]
    fused = sorted(rrf_fuse([vec_rank, bm_rank]).items(), key=lambda x: -x[1])[:k]
    return [(store.chunks[i], s) for i, s in fused]


SYSTEM_RAG = ("你是这个机器学习教程仓库的助教。回答中的关键数字与结论必须出自【资料】且一字不差；"
              "在此前提下充分展开：解释背景、给出实操建议，像给学生讲课一样完整。"
              "只有资料与问题完全无关时才说不知道。用中文，不客套。")
SYSTEM_BARE = "你是机器学习助教。直接回答，用中文。"


def build_context(hits, score_label: str = "相似度") -> str:
    return "\n\n".join(f"【资料{i+1}·{c.doc}·{c.heading}·{score_label}{s:.2f}】\n{c.text}"
                       for i, (c, s) in enumerate(hits))


def answer_with_context(question: str, hits, score_label: str = "相似度") -> str:
    prompt = f"{build_context(hits, score_label)}\n\n问题：{question}"
    key = hashlib.md5(("rag|" + prompt).encode("utf-8")).hexdigest()
    return _cached_chat(key, [{"role": "system", "content": SYSTEM_RAG},
                              {"role": "user", "content": prompt}])


def answer_bare(question: str) -> str:
    key = hashlib.md5(("bare|" + question).encode("utf-8")).hexdigest()
    return _cached_chat(key, [{"role": "system", "content": SYSTEM_BARE},
                              {"role": "user", "content": question}])
