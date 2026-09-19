"""分块策略消融（收官之十六·方法线）：主索引"标题感知·700字"从 v1 用到现在从未被 A/B。
三策略各建一份索引，在 90 题压力大盘（qa_stress80）上比 向量/hybrid 的 hit@4 与 MRR@6：
  A heading700  基线：chunk_markdown（生产代码原样）
  B fixed700    砍掉结构感知：整篇文档 700 字定窗直切（重叠 80，与基线同尺寸）
  C para350     砍一半粒度：标题感知保留，段落边界 ≤350 小块、无重叠
公平性：B/C 与 A 只差"怎么切"，嵌入模型/检索链路（hybrid 参数 cand=8、k=4，与 ask.py 一致）不变。
零 LLM 生成费；A 全命中嵌入缓存，B/C 各需重嵌（一次性）。用法：python evaluation/run_chunk_ab.py
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from rag.bm25 import BM25, rrf_fuse
from rag.chunker import Chunk, chunk_markdown
from rag.config import KB_GLOBS, ROOT
from rag.embeddings import embed_texts
from rag.store import VectorStore

AB_DIR = HERE / "index_ab"


def _collect_docs() -> dict[str, str]:
    seen: dict[str, str] = {}
    for g in KB_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if p.is_file() and "node_modules" not in str(p) and ".venv" not in str(p):
                seen[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    return seen


def chunk_fixed(md: str, doc: str, size: int = 700, overlap: int = 80) -> list[Chunk]:
    """B：无视标题，定窗直切——测"结构感知"到底买不买账。"""
    text = md.strip()
    out: list[Chunk] = []
    i = 0
    while True:
        piece = text[i:i + size].strip()
        if piece:
            out.append(Chunk(doc, "(fixed)", piece, len(out)))
        if i + size >= len(text):
            break
        i += size - overlap
    return out


def chunk_para350(md: str, doc: str, size: int = 350) -> list[Chunk]:
    """C：标题感知保留，但更细——先按空行分段凑到 ≤size，单段超长再按句号硬切，无重叠。"""
    sections: list[tuple[str, list[str]]] = []
    heading, buf = "(intro)", []
    for line in md.splitlines():
        if line.startswith("## "):
            if any(l.strip() for l in buf):
                sections.append((heading, buf))
            heading, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if any(l.strip() for l in buf):
        sections.append((heading, buf))

    chunks: list[Chunk] = []

    def emit(h: str, t: str) -> None:
        t = t.strip()
        if t:
            chunks.append(Chunk(doc, h, t, len(chunks)))

    for h, lines in sections:
        block = "\n".join(lines).strip()
        if not block:
            continue
        cur = ""
        for p in [x.strip() for x in block.split("\n\n") if x.strip()]:
            if cur and len(cur) + len(p) > size:
                emit(h, cur)
                cur = p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
            while len(cur) > size:  # 单段本身就超长
                cut = next((j + 1 for j in range(size, len(cur)) if cur[j] == "。"), size)
                emit(h, cur[:cut])
                cur = cur[cut:]
        emit(h, cur)
    return chunks


STRATEGIES = {
    "A_heading700": lambda docs: [c for d, t in docs.items() for c in chunk_markdown(t, d)],
    "B_fixed700": lambda docs: [c for d, t in docs.items() for c in chunk_fixed(t, d)],
    "C_para350": lambda docs: [c for d, t in docs.items() for c in chunk_para350(t, d)],
}


def build(name: str, chunk_fn) -> VectorStore:
    idx = AB_DIR / name
    if (idx / "vectors.npy").exists():
        return VectorStore.load(idx)
    chunks = chunk_fn(_collect_docs())
    vecs = embed_texts([c.text for c in chunks], verbose=False)
    store = VectorStore()
    store.add(chunks, vecs)
    store.save(idx)
    med = statistics.median(len(c.text) for c in chunks)
    print(f"  [{name}] 建库 {len(chunks)} 块 / 中位 {med:.0f} 字")
    return store


def evaluate(store: VectorStore, qs: list[dict]) -> dict:
    hit_v = hit_h = 0
    mrr_v = mrr_h = 0.0
    rows: list[dict] = []
    bm = BM25([c.text for c in store.chunks])
    qvecs = embed_texts([it["q"] for it in qs], verbose=False)

    def rank(order, need):
        return next((r + 1 for r, i in enumerate(order)
                     if any(d in store.chunks[i].doc for d in need)), 99)

    for item, qv in zip(qs, qvecs):
        need = item["docs"]
        vr8 = store.search_all(qv, k=8)
        rv = rank([i for i, _ in vr8[:6]], need)
        hit_v += rv <= 4
        mrr_v += 1 / rv if rv <= 6 else 0.0
        # hybrid 路：与 ask.py/生产同参（两路各 cand=8，RRF 融合后取前 6 打分）
        br = [i for i, _ in bm.search(item["q"], k=8)]
        fused = sorted(rrf_fuse([[i for i, _ in vr8], br]).items(), key=lambda x: -x[1])
        rh = rank([i for i, _ in fused[:6]], need)
        hit_h += rh <= 4
        mrr_h += 1 / rh if rh <= 6 else 0.0
        rows.append({"q": item["q"][:36], "vec": rv, "hyb": rh})
    n = len(qs)
    return {"chunks": len(store.chunks), "vec_hit4": round(hit_v / n, 4),
            "hyb_hit4": round(hit_h / n, 4), "vec_mrr6": round(mrr_v / n, 4),
            "hyb_mrr6": round(mrr_h / n, 4), "per_q": rows}


def evaluate_multihop(store: VectorStore, qs: list[dict]) -> list[int]:
    """多跳视角：双文档全进 top-4 才算赢——大块在文档级 hit 上的水分在这里挤掉。"""
    from run_multihop20_ab import worst_rank
    bm = BM25([c.text for c in store.chunks])
    qvecs = embed_texts([it["q"] for it in qs], verbose=False)
    out = []
    for item, qv in zip(qs, qvecs):
        vr8 = [i for i, _ in store.search_all(qv, k=8)]
        br = [i for i, _ in bm.search(item["q"], k=8)]
        fused = sorted(rrf_fuse([vr8, br]).items(), key=lambda x: -x[1])[:6]
        out.append(worst_rank([store.chunks[i].doc for i, _ in fused], item["docs"]))
    return out


def main() -> None:
    from run_multihop50_ab import binom_two_sided
    qs = json.loads((HERE / "evaluation" / "qa_stress80.json").read_text(encoding="utf-8"))
    mh = json.loads((HERE / "evaluation" / "qa_multihop50.json").read_text(encoding="utf-8"))
    print(f"题集 {len(qs)} 问（qa_stress80）+ {len(mh)} 问多跳 · 知识库 {len(_collect_docs())} 文档")
    results, mh_worst = {}, {}
    for name, fn in STRATEGIES.items():
        store = build(name, fn)
        results[name] = evaluate(store, qs)
        mh_worst[name] = evaluate_multihop(store, mh)
        res = {k: v for k, v in results[name].items() if k != "per_q"}
        top4 = sum(w <= 4 for w in mh_worst[name])
        print(f"  [{name}] {res} | 多跳双文档进top-4: {top4}/{len(mh)}")
    base = results["A_heading700"]["per_q"]
    sig = {}
    for name, res in results.items():
        if name == "A_heading700":
            continue
        pairs = [(a["hyb"], b["hyb"]) for a, b in zip(base, res["per_q"])]
        w = sum(nb < na for na, nb in pairs)   # X 严格优于 A 的题数
        l = sum(nb > na for na, nb in pairs)
        sig[name + "_stress"] = {"X_better": w, "A_better": l,
                                 "p": round(binom_two_sided(w, w + l), 4)}
        mp = list(zip(mh_worst["A_heading700"], mh_worst[name]))
        mw = sum(b < a for a, b in mp)
        ml = sum(b > a for a, b in mp)
        sig[name + "_multihop"] = {"X_better": mw, "A_better": ml,
                                   "p": round(binom_two_sided(mw, mw + ml), 4)}
        print(f"  符号检验 {name} vs A：压力题 {w}:{l} p={sig[name + '_stress']['p']} · "
              f"多跳题 {mw}:{ml} p={sig[name + '_multihop']['p']}")
    out = {"n": len(qs), "retrieval": "hybrid=RRF(cand8) 与 ask.py 一致；MRR@6；"
           "压力题为文档级命中，多跳题要求双文档全进 top-4",
           "significance": sig,
           "multihop50_top4": {k: sum(w <= 4 for w in v) for k, v in mh_worst.items()},
           "results": results}
    (HERE / "evaluation" / "chunk_ab_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已存 evaluation/chunk_ab_results.json")


if __name__ == "__main__":
    main()
