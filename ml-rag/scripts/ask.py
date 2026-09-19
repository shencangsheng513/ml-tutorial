"""问答 CLI：python scripts/ask.py "问题" —— 混合检索 top4 资料，RAG 作答并给出处。
默认 hybrid（90 题大盘 top-4 命中 87.8% vs 纯向量 81.1%）；--vec 回退纯向量。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import INDEX_DIR
from rag.pipeline import answer_with_context, retrieve, retrieve_hybrid
from rag.store import VectorStore


def main() -> None:
    if len(sys.argv) < 2 or not (INDEX_DIR / "vectors.npy").exists():
        sys.exit("用法：python scripts/ask.py '问题'（先跑 build_index.py）")
    vec_only = "--vec" in sys.argv[1:]
    q = " ".join(a for a in sys.argv[1:] if a != "--vec")
    store = VectorStore.load(INDEX_DIR)
    hits = (retrieve(store, q, k=4) if vec_only
            else retrieve_hybrid(store, q, k=4))
    print("引用来源：")
    for c, s in hits:
        print(f"  [{s:.3f}] {c.doc} § {c.heading}")
    print("\n" + answer_with_context(q, hits, score_label="向量分" if vec_only else "RRF分"))


if __name__ == "__main__":
    main()
