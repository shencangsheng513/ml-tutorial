"""离线单测：不花一分钱 API 费的核心逻辑（切块 + 向量库）。"""
import numpy as np
import pytest

from rag.chunker import MAX_CHARS, chunk_markdown
from rag.store import VectorStore


def test_section_split_by_heading():
    md = "# t\n\nintro text\n\n## A\n\naaa\n\n## B\n\nbbb"
    chunks = chunk_markdown(md, "x.md")
    headings = [c.heading for c in chunks]
    assert "A" in headings and "B" in headings


def test_long_section_is_split_with_limit():
    long_sec = "。".join(["这是一句足够长的中文用来把段落撑爆" * 3] * 60)
    md = f"## 大节\n\n{long_sec}"
    chunks = chunk_markdown(md, "x.md")
    assert len(chunks) >= 2
    assert all(len(c.text) <= MAX_CHARS * 1.6 for c in chunks)  # 句子边界容忍小幅超


def test_empty_doc_yields_nothing():
    assert chunk_markdown("", "x.md") == []


def test_store_cosine_ranking_basic():
    store = VectorStore()
    texts = ["cat", "dog", "car"]
    vecs = [[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]]  # 手造单位近似向量
    from rag.chunker import Chunk
    store.add([Chunk("d", "h", t, i) for i, t in enumerate(texts)], vecs)
    hits = store.search([1, 0, 0], k=2)
    assert hits[0][0].text == "cat"
    assert hits[1][0].text == "dog"


def test_store_save_load_roundtrip(tmp_path):
    from rag.chunker import Chunk
    store = VectorStore()
    store.add([Chunk("a.md", "H", "hello", 0)], [[1.0, 2.0]])
    store.save(tmp_path)
    other = VectorStore.load(tmp_path)
    assert other.chunks[0].doc == "a.md"
    assert np.allclose(other.matrix[0] / np.linalg.norm(other.matrix[0]),
                       [1, 2] / np.linalg.norm([1, 2]), atol=1e-5)


def test_load_corrupted_index_raises(tmp_path):
    VectorStore().save(tmp_path)
    (tmp_path / "chunks.jsonl").write_text('{"doc":"x"}\n', encoding="utf-8")
    with pytest.raises((TypeError, AssertionError)):
        VectorStore.load(tmp_path)
