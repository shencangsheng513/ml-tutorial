"""v12 扩库：现有 10 篇 md + 仓库真实 .py（ml-rag 自身除外）→ 4 倍规模语料，索引落 index_expanded/。
只作规模压力测试用，生产 index/ 完全不动。--dry 只切块报数、不嵌入（零费）。
用法：python scripts/build_index_expanded.py --dry   /   python scripts/build_index_expanded.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunker import chunk_code, chunk_markdown
from rag.config import ROOT
from rag.embeddings import embed_texts
from rag.store import VectorStore

OUT_DIR = Path(__file__).resolve().parents[1] / "index_expanded"
CTRL_DIR = Path(__file__).resolve().parents[1] / "index_control"
# 纳入的代码项目（排除 ml-rag 自身，免得把评估仪器当知识、自指导致混淆）
CODE_DIRS = ["ml-text", "ml-vision", "ml-timeseries", "ml-car-price",
             "kaggle-sim", "mlflow-lab", "ml-deploy"]
SKIP_PARTS = {".venv", "node_modules", "__pycache__", ".pytest_cache"}


def collect_markdown() -> dict[str, str]:
    seen: dict[str, str] = {}
    for g in ["*.md", "*/README.md", "ml-text/*.md", "ml-vision/*.md", "ml-timeseries/*.md",
              "ml-car-price/*.md", "kaggle-sim/*.md", "mlflow-lab/*.md"]:
        for p in sorted(ROOT.glob(g)):
            if p.is_file() and not SKIP_PARTS & set(p.parts):
                seen[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    return seen


def collect_code() -> dict[str, str]:
    seen: dict[str, str] = {}
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            if p.is_file() and not SKIP_PARTS & set(p.relative_to(ROOT).parts):
                seen[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    return seen


def build_chunks() -> list:
    chunks = []
    for rel, text in collect_markdown().items():
        chunks.extend(chunk_markdown(text, rel))
    for rel, text in collect_code().items():
        chunks.extend(chunk_code(text, rel))
    return chunks


def main() -> None:
    md = collect_markdown()
    code = collect_code()
    chunks = build_chunks()
    n_md = sum(len(chunk_markdown(t, r)) for r, t in md.items())
    print(f"文档 {len(md)} 篇→{n_md} 块；代码 {len(code)} 文件→{len(chunks) - n_md} 块；"
          f"合计 {len(chunks)} 块 / {sum(len(c.text) for c in chunks) / 1000:.0f}k 字符")
    if "--dry" in sys.argv:
        print("(--dry) 只切块未嵌入，上面块数即成本预估")
        return
    if "--md-only" in sys.argv:
        mchunks = [c for rel, text in md.items() for c in chunk_markdown(text, rel)]
        store = VectorStore()
        store.add(mchunks, embed_texts([c.text for c in mchunks]))
        CTRL_DIR.mkdir(exist_ok=True)
        store.save(CTRL_DIR)
        print(f"对照索引（仅文档）已存 {CTRL_DIR.name}/（{len(store.matrix)}×{store.matrix.shape[1]}）")
        return
    vecs = embed_texts([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vecs)
    OUT_DIR.mkdir(exist_ok=True)
    store.save(OUT_DIR)
    print(f"扩库索引已存 {OUT_DIR.name}/（{len(store.matrix)}×{store.matrix.shape[1]}）")


if __name__ == "__main__":
    main()
