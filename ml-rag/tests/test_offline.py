"""离线单测：不花一分钱 API 费的核心逻辑（切块 + 向量库）。"""
import numpy as np
import pytest

from rag.bm25 import BM25, rrf_fuse, tokenize
from rag.chunker import MAX_CHARS, Chunk, chunk_markdown
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
    store.add([Chunk("d", "h", t, i) for i, t in enumerate(texts)], vecs)
    hits = store.search([1, 0, 0], k=2)
    assert hits[0][0].text == "cat"
    assert hits[1][0].text == "dog"


def test_store_save_load_roundtrip(tmp_path):
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


def test_tokenize_cjk_bigrams_and_ascii():
    toks = tokenize("算256乘以37的HTTP结果")
    assert "256" in toks and "37" in toks and "http" in toks
    assert "乘以" in toks and "以3" not in toks  # 数字打断 CJK 连续段，bigram 只在汉字间产生
    assert tokenize("好") == ["好"]  # 单字保留


def test_bm25_ranks_matching_doc_first_and_unknown_scores_zero():
    docs = ["容器健康检查 HEALTHCHECK 探针配置", "Kaggle 公榜过拟合教训", "完全无关的菜谱"]
    bm = BM25(docs)
    hits = bm.search("HEALTHCHECK 探针", k=3)
    assert hits[0][0] == 0
    assert bm.search("不存在的词条zzz") == []


def test_bm25_length_normalization_prefers_short_exact_doc():
    long_doc = "回归" + "。填充文字" * 50
    bm = BM25(["回归分析入门", long_doc])
    assert bm.search("回归", k=2)[0][0] == 0  # 同 tf 下短文档 b 惩罚小


def test_rrf_fuse_agreement_beats_single_list_rank1():
    fused = rrf_fuse([[5, 7], [7, 5]], c=60)
    assert abs(fused[5] - fused[7]) < 1e-12  # 两榜名次互换 → 总分相同
    both = rrf_fuse([[3, 9], [9, 3]], c=60)
    only_one = rrf_fuse([[7], []], c=60)
    assert min(both.values()) > max(only_one.values())  # 双榜命中 > 单榜第一


def test_rrf_fuse_empty_rankings():
    assert rrf_fuse([[], []]) == {}
