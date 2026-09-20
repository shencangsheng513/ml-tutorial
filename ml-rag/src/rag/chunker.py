"""Markdown 感知切块：按 ## 标题分段，超长的按句子边界二次切，保留重叠。
代码切块 chunk_code：按顶层 def/class 分段，超长按行边界二次切（规模压力实验用）。"""
from dataclasses import dataclass

MAX_CHARS = 700
OVERLAP_CHARS = 80


@dataclass
class Chunk:
    doc: str          # 源文件相对路径
    heading: str      # 所属标题
    text: str
    idx: int


def _split_long(paragraph_block: str, doc: str, heading: str, start_i: int) -> list[Chunk]:
    out, buf = [], ""
    sentences = paragraph_block.replace("\n", " ").split("。")
    for s in sentences:
        piece = s + "。"
        if len(buf) + len(piece) > MAX_CHARS and buf:
            out.append(Chunk(doc, heading, buf.strip(), start_i + len(out)))
            buf = buf[-OVERLAP_CHARS:] + piece
        else:
            buf += piece
    if buf.strip():
        out.append(Chunk(doc, heading, buf.strip(), start_i + len(out)))
    return out


def chunk_markdown(md_text: str, doc: str) -> list[Chunk]:
    """以 '## ' 为一级切分点；无标题的前导内容归入 '(intro)'。"""
    sections: list[tuple[str, list[str]]] = []
    current_heading, buf = "(intro)", []
    for line in md_text.splitlines():
        if line.startswith("## "):
            if any(l.strip() for l in buf):
                sections.append((current_heading, buf))
            current_heading, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if any(l.strip() for l in buf):
        sections.append((current_heading, buf))

    chunks: list[Chunk] = []
    for heading, lines in sections:
        block = "\n".join(lines).strip()
        if not block:
            continue
        if len(block) <= MAX_CHARS:
            chunks.append(Chunk(doc, heading, block, len(chunks)))
        else:
            chunks.extend(_split_long(block, doc, heading, len(chunks)))
    return chunks


def _split_code_lines(block: str, doc: str, heading: str, start_i: int) -> list[Chunk]:
    """代码块按行累加到 ~MAX_CHARS 再切，不重叠（代码不像散文那样靠重叠保语义）。"""
    out, buf = [], ""
    for line in block.splitlines():
        if len(buf) + len(line) + 1 > MAX_CHARS and buf:
            out.append(Chunk(doc, heading, buf.rstrip(), start_i + len(out)))
            buf = ""
        buf += line + "\n"
    if buf.strip():
        out.append(Chunk(doc, heading, buf.rstrip(), start_i + len(out)))
    return out


def chunk_code(py_text: str, doc: str) -> list[Chunk]:
    """按顶层 def/class 切块（缩进行归当前块），前导 import/模块 docstring 归 '(module)'。"""
    blocks: list[tuple[str, list[str]]] = []
    current, buf = "(module)", []
    for line in py_text.splitlines():
        stripped = line.strip()
        is_top = line[:1] not in (" ", "\t") and stripped.startswith(("def ", "class ", "@"))
        if is_top:
            if any(l.strip() for l in buf):
                blocks.append((current, buf))
            head = (stripped.split("(")[0].replace("def ", "").replace("class ", "")
                    .lstrip("@").strip().rstrip(":"))
            current, buf = head or "@", [line]
        else:
            buf.append(line)
    if any(l.strip() for l in buf):
        blocks.append((current, buf))

    chunks: list[Chunk] = []
    for head, lines in blocks:
        block = "\n".join(lines).rstrip()
        if not block.strip():
            continue
        if len(block) <= MAX_CHARS:
            chunks.append(Chunk(doc, head, block, len(chunks)))
        else:
            chunks.extend(_split_code_lines(block, doc, head, len(chunks)))
    return chunks
