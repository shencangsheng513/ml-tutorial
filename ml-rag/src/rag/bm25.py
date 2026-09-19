"""纯 Python BM25(Okapi) + 中文友好分词 + RRF 融合检索。零依赖，32 块规模够用。

分词策略：ASCII 连续段按词、CJK 连续段按字符二元组（bigram）——
不引 jieba 是因为词汇表就 9 篇 md，bigram 已足够区分"房价/归回"这类词对。
"""
import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group()
        if s[0].isascii():
            toks.append(s.lower())
        elif len(s) == 1:
            toks.append(s)  # 单字也保留，避免单字查询查不到
        else:
            toks.extend(s[i:i + 2] for i in range(len(s) - 1))
    return toks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.doc_toks]
        self.lens = [len(t) for t in self.doc_toks]
        n = max(len(docs), 1)
        self.avgdl = sum(self.lens) / n
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        self.idf = {w: math.log((n - c + 0.5) / (c + 0.5) + 1) for w, c in df.items()}

    def search(self, query: str, k: int = 8) -> list[tuple[int, float]]:
        scores: defaultdict[int, float] = defaultdict(float)
        for w in tokenize(query):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1))
                    scores[i] += idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


def rrf_fuse(rankings: list[list[int]], c: float = 60.0) -> dict[int, float]:
    """Reciprocal Rank Fusion：只看名次不看量纲，天然免疫 BM25 分与余弦分不可比的问题。"""
    pool: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            pool[idx] += 1.0 / (c + r + 1)
    return dict(pool)
