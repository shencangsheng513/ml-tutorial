"""检索 + 生成 的两条路：有上下文（RAG）与裸考（无上下文），供对照实验用。"""
from openai import OpenAI

from .config import CHAT_MODEL, load_credentials
from .embeddings import embed_texts
from .store import VectorStore


def _client() -> OpenAI:
    key, base = load_credentials()
    return OpenAI(api_key=key, base_url=base, timeout=120)


def retrieve(store: VectorStore, question: str, k: int = 4):
    qv = embed_texts([question], verbose=False)[0]
    return store.search(qv, k=k)


SYSTEM_RAG = ("你是这个机器学习教程仓库的助教。回答中的关键数字与结论必须出自【资料】且一字不差；"
              "在此前提下充分展开：解释背景、给出实操建议，像给学生讲课一样完整。"
              "只有资料与问题完全无关时才说不知道。用中文，不客套。")
SYSTEM_BARE = "你是机器学习助教。直接回答，用中文。"


def answer_with_context(question: str, hits) -> str:
    ctx = "\n\n".join(f"【资料{i+1}·{c.doc}·{c.heading}·相似度{s:.2f}】\n{c.text}"
                      for i, (c, s) in enumerate(hits))
    r = _client().chat.completions.create(
        model=CHAT_MODEL, temperature=0, max_tokens=800,
        messages=[{"role": "system", "content": SYSTEM_RAG},
                  {"role": "user", "content": f"{ctx}\n\n问题：{question}"}])
    return r.choices[0].message.content.strip()


def answer_bare(question: str) -> str:
    r = _client().chat.completions.create(
        model=CHAT_MODEL, temperature=0, max_tokens=800,
        messages=[{"role": "system", "content": SYSTEM_BARE},
                  {"role": "user", "content": question}])
    return r.choices[0].message.content.strip()
