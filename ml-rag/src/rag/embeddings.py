"""Embedding 客户端 + 磁盘缓存：同一文本永远只嵌一次（重建索引零成本）。"""
import hashlib
import json
import time

from openai import OpenAI

from .config import EMBED_MODEL, INDEX_DIR, load_credentials

_CACHE_FILE = INDEX_DIR / "embed_cache.json"


def _client() -> OpenAI:
    key, base = load_credentials()
    return OpenAI(api_key=key, base_url=base, timeout=60)


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def embed_texts(texts: list[str], batch: int = 10, verbose: bool = True) -> list[list[float]]:
    cache: dict[str, list[float]] = {}
    if _CACHE_FILE.exists():
        cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    todo = [t for t in texts if _md5(t) not in cache]
    if todo:
        cli = _client()
        if verbose:
            print(f"  embedding {len(todo)} 条新文本（{len(texts) - len(todo)} 条命中缓存）")
        for i in range(0, len(todo), batch):
            grp = todo[i:i + batch]
            for attempt in range(3):
                try:
                    r = cli.embeddings.create(model=EMBED_MODEL, input=grp)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            for t, d in zip(grp, r.data):
                cache[_md5(t)] = d.embedding
            if verbose:
                print(f"  embedded {min(i + batch, len(todo))}/{len(todo)}", end="\r")
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[_md5(t)] for t in texts]
