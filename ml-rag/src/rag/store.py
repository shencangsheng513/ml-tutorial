"""最小向量库：L2 归一 + 余弦检索，JSONL+NPZ 持久化。千级块规模够用且零依赖。"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], vecs: list[list[float]]) -> None:
        m = np.asarray(vecs, dtype=np.float32)
        m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        self.chunks.extend(chunks)
        self.matrix = np.vstack([self.matrix, m]) if self.matrix.size else m

    def search(self, qvec: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        q = np.asarray(qvec, dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.matrix)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in self.chunks)

    @classmethod
    def load(cls, d: Path) -> "VectorStore":
        s = cls()
        s.matrix = np.load(d / "vectors.npy")
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            s.chunks = [Chunk(**json.loads(line)) for line in f]
        assert len(s.chunks) == len(s.matrix), "索引与块数不一致，请重建"
        return s
