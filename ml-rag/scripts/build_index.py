"""建库 CLI：扫描教程仓库所有 md -> 切块 -> 嵌入 -> 存向量索引。
用法：python scripts/build_index.py   （在 ml-rag/ 目录下）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunker import chunk_markdown
from rag.config import INDEX_DIR, KB_GLOBS, ROOT
from rag.embeddings import embed_texts
from rag.store import VectorStore


def collect_docs() -> dict[str, str]:
    seen: dict[str, str] = {}
    for g in KB_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if p.is_file() and "node_modules" not in str(p) and ".venv" not in str(p):
                seen[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    return seen


def main() -> None:
    docs = collect_docs()
    chunks = []
    for rel, text in docs.items():
        chunks.extend(chunk_markdown(text, rel))
    print(f"知识库：{len(docs)} 个文档 -> {len(chunks)} 块，"
          f"共 {sum(len(c.text) for c in chunks)/1000:.0f}k 字符")
    vecs = embed_texts([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vecs)
    store.save(INDEX_DIR)
    print(f"索引已存 {INDEX_DIR}（{len(store.matrix)}×{store.matrix.shape[1]}）")


if __name__ == "__main__":
    main()
