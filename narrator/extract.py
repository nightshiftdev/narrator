"""Turn md / pdf / txt into a Document of clean, ordered blocks."""
from __future__ import annotations

import re
from pathlib import Path

from .document import Block, Document


# --------------------------------------------------------------------------- md

_MD_INLINE = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),      # image -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),       # link -> label
    (re.compile(r"`([^`]+)`"), r"\1"),                   # inline code
    (re.compile(r"\*\*\*([^*]+)\*\*\*"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)"), r"\1"),
    (re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)"), r"\1"),
    (re.compile(r"~~([^~]+)~~"), r"\1"),
    (re.compile(r"<[^>]+>"), ""),                        # stray html
]


def _md_inline(s: str) -> str:
    for pat, rep in _MD_INLINE:
        s = pat.sub(rep, s)
    return s.strip()


def extract_markdown(text: str, title_hint: str) -> Document:
    blocks: list[Block] = []
    title = title_hint
    lines = text.splitlines()
    i, n = 0, len(lines)
    para: list[str] = []

    def flush_para() -> None:
        nonlocal para
        if para:
            body = _md_inline(" ".join(x.strip() for x in para))
            if body:
                blocks.append(Block("para", body, index=len(blocks)))
            para = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            flush_para()
            i += 1
            body = []
            while i < n and not lines[i].strip().startswith(fence):
                body.append(lines[i])
                i += 1
            i += 1
            blocks.append(Block("code", "\n".join(body), index=len(blocks)))
            continue

        # yaml front matter
        if i == 0 and stripped == "---":
            i += 1
            while i < n and lines[i].strip() != "---":
                m = re.match(r"title:\s*(.+)", lines[i].strip())
                if m:
                    title = m.group(1).strip().strip("\"'")
                i += 1
            i += 1
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        # atx heading
        m = re.match(r"(#{1,6})\s+(.*)", stripped)
        if m:
            flush_para()
            level = len(m.group(1))
            body = _md_inline(m.group(2)).rstrip("#").strip()
            kind = "title" if level == 1 and not blocks else ("chapter" if level <= 2 else "heading")
            if kind == "title":
                title = body
            blocks.append(Block(kind, body, level=level, index=len(blocks)))
            i += 1
            continue

        # setext heading
        if i + 1 < n and re.fullmatch(r"=+|-{2,}", lines[i + 1].strip()) and stripped:
            flush_para()
            level = 1 if lines[i + 1].strip()[0] == "=" else 2
            body = _md_inline(stripped)
            kind = "title" if level == 1 and not blocks else "chapter"
            if kind == "title":
                title = body
            blocks.append(Block(kind, body, level=level, index=len(blocks)))
            i += 2
            continue

        # horizontal rule
        if re.fullmatch(r"(\*\s*){3,}|(-\s*){3,}|(_\s*){3,}", stripped):
            flush_para()
            i += 1
            continue

        # block quote
        if stripped.startswith(">"):
            flush_para()
            body = []
            while i < n and lines[i].strip().startswith(">"):
                body.append(lines[i].strip().lstrip(">").strip())
                i += 1
            text_ = _md_inline(" ".join(x for x in body if x))
            if text_:
                blocks.append(Block("quote", text_, index=len(blocks)))
            continue

        # list item
        m = re.match(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)", line)
        if m:
            flush_para()
            indent = len(m.group(1)) // 2
            body = [m.group(2)]
            i += 1
            # continuation lines
            while i < n and lines[i].strip() and not re.match(
                r"^\s*(?:[-*+]|\d+[.)])\s+|^#{1,6}\s|^>", lines[i]
            ):
                body.append(lines[i].strip())
                i += 1
            item = _md_inline(" ".join(body))
            if item:
                blocks.append(Block("list_item", item, level=indent, index=len(blocks)))
            continue

        # table row -> skip quietly (reading pipe tables aloud is noise)
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_para()
            while i < n and lines[i].strip().startswith("|"):
                i += 1
            continue

        para.append(line)
        i += 1

    flush_para()
    return Document(title=title, blocks=blocks)


# -------------------------------------------------------------------------- pdf

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_PAGE_NUM = re.compile(r"^\s*(?:page\s+)?[ivxlcdm\d]{1,6}\s*$", re.I)


def extract_pdf(path: Path) -> Document:
    import pymupdf as fitz

    doc = fitz.open(path)
    title = (doc.metadata or {}).get("title") or path.stem
    title = title.strip() or path.stem

    # Collect per-page text with font sizes so we can spot headings.
    blocks: list[Block] = []
    sizes: list[float] = []
    page_spans: list[list[tuple[str, float, int]]] = []

    for pno, page in enumerate(doc):
        spans: list[tuple[str, float, int]] = []
        data = page.get_text("dict")
        for blk in data.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", []))
                if not text.strip():
                    continue
                size = max((s.get("size", 0.0) for s in line.get("spans", [])), default=0.0)
                spans.append((text, size, pno))
                sizes.append(size)
        page_spans.append(spans)

    body_size = _mode(sizes) if sizes else 10.0
    # Merge lines into paragraphs; promote clearly larger lines to headings.
    buf: list[str] = []
    buf_page = 0

    def flush() -> None:
        nonlocal buf
        if buf:
            text = " ".join(buf)
            text = _HYPHEN_BREAK.sub(r"\1\2", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 1:
                blocks.append(Block("para", text, index=len(blocks), source_page=buf_page))
            buf = []

    # Designed PDFs often stack duplicate text layers (drop shadows, outlines);
    # left alone they make the narrator stutter.
    prev_line = ""
    for spans in page_spans:
        for text, size, pno in spans:
            s = text.strip()
            if not s or _PAGE_NUM.fullmatch(s):
                continue
            collapsed = " ".join(s.split())
            if collapsed == prev_line:
                continue
            # "PAWEL KIJOWSKI" immediately after "PAWEL" is the same layer twice
            if prev_line and (collapsed.startswith(prev_line + " ")
                              or prev_line.startswith(collapsed + " ")):
                continue
            # a line that is its own halves repeated: "certify that certify that"
            words = collapsed.split()
            half = len(words) // 2
            if half and words[:half] == words[half:half * 2] and len(words) % 2 == 0:
                collapsed = " ".join(words[:half])
                s = collapsed
            prev_line = collapsed
            is_heading = size > body_size * 1.15 and len(s) < 120
            if is_heading:
                flush()
                buf_page = pno
                level = 1 if size > body_size * 1.5 else 2
                kind = "title" if level == 1 and not blocks else "chapter"
                blocks.append(Block(kind, s, level=level, index=len(blocks), source_page=pno))
                continue
            if not buf:
                buf_page = pno
            buf.append(s)
            # a line ending in sentence punctuation and short is likely a para end
            if s.endswith((".", "!", "?", '"', "”")) and len(" ".join(buf)) > 200:
                flush()
    flush()
    doc.close()
    return Document(title=title, blocks=blocks)


def _mode(values: list[float]) -> float:
    counts: dict[float, int] = {}
    for v in values:
        k = round(v, 1)
        counts[k] = counts.get(k, 0) + 1
    return max(counts, key=lambda k: counts[k])


# -------------------------------------------------------------------------- txt


def extract_txt(text: str, title_hint: str) -> Document:
    blocks: list[Block] = []
    for chunk in re.split(r"\n\s*\n", text):
        body = re.sub(r"\s+", " ", chunk).strip()
        if not body:
            continue
        # ALL-CAPS or very short standalone lines read as headings
        if len(body) < 80 and "\n" not in chunk.strip() and (
            body.isupper() or re.match(r"^(chapter|part|book)\b", body, re.I)
        ):
            blocks.append(Block("chapter", body.title() if body.isupper() else body,
                                level=1, index=len(blocks)))
        else:
            blocks.append(Block("para", body, index=len(blocks)))
    return Document(title=title_hint, blocks=blocks)


# ------------------------------------------------------------------------ entry


def load(path: str | Path) -> Document:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    hint = re.sub(r"[_-]+", " ", p.stem).strip().title()

    if suffix == ".pdf":
        doc = extract_pdf(p)
    elif suffix in (".md", ".markdown", ".mdown"):
        doc = extract_markdown(p.read_text(encoding="utf-8", errors="replace"), hint)
    else:
        doc = extract_txt(p.read_text(encoding="utf-8", errors="replace"), hint)

    doc.source = str(p)
    for i, b in enumerate(doc.blocks):
        b.index = i
    return doc
