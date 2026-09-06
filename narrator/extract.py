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


def _pdf_lines(path: Path) -> list[tuple[str, float, int]]:
    """(text, font size, page) for every line, in reading order.

    pdfium reports a font size and a *loose* box per character. The loose box
    is line-based rather than glyph-based, so its top edge is stable across
    letters of different height — that is what makes it usable for grouping
    characters back into lines.
    """
    import ctypes

    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    out: list[tuple[str, float, int]] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for pno in range(len(pdf)):
            page = pdf[pno]
            tp = page.get_textpage()
            cur: list[str] = []
            cur_size = 0.0
            cur_top: float | None = None
            box = raw.FS_RECTF()

            for i in range(tp.count_chars()):
                ch = chr(raw.FPDFText_GetUnicode(tp, i))
                if ch in "\r\n":
                    continue
                size = raw.FPDFText_GetFontSize(tp, i)
                raw.FPDFText_GetLooseCharBox(tp, i, ctypes.byref(box))
                top = round(box.top, 1)
                if cur_top is not None and abs(top - cur_top) > 2:
                    text = "".join(cur).strip()
                    if text:
                        out.append((text, round(cur_size, 1), pno))
                    cur, cur_size = [], 0.0
                cur.append(ch)
                cur_size = max(cur_size, size)
                cur_top = top

            text = "".join(cur).strip()
            if text:
                out.append((text, round(cur_size, 1), pno))
    finally:
        pdf.close()
    return out


def _pdf_title(path: Path) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        meta = pdf.get_metadata_dict() or {}
    except Exception:
        meta = {}
    finally:
        pdf.close()
    return (meta.get("Title") or "").strip() or path.stem


def extract_pdf(path: Path) -> Document:
    title = _pdf_title(path)
    lines = _pdf_lines(path)
    sizes = [size for _, size, _ in lines]
    body_size = _mode(sizes) if sizes else 10.0

    # A bare number is only a page number when it sits alone at the top or
    # bottom of a page. In the middle of a page it is content — "5" in
    # "0.5 Continuing Education Units; 5 Professional Development Hours".
    edge: set[int] = set()
    by_page: dict[int, list[int]] = {}
    for idx, (_, _, pno) in enumerate(lines):
        by_page.setdefault(pno, []).append(idx)
    for idxs in by_page.values():
        edge.add(idxs[0])
        edge.add(idxs[-1])

    blocks: list[Block] = []
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

    for idx, (text, size, pno) in enumerate(lines):
        s = text.strip()
        if not s:
            continue
        if idx in edge and _PAGE_NUM.fullmatch(s):
            continue
        collapsed = " ".join(s.split())
        if collapsed == prev_line:
            continue
        # Only treat a prefix repeat as a stacked layer when the shared part
        # is substantial — otherwise a legitimately short line ("5", "and")
        # gets swallowed by the sentence that happens to follow it.
        shorter = min(collapsed, prev_line, key=len)
        if (prev_line and len(shorter) >= 8
                and (collapsed.startswith(prev_line + " ")
                     or prev_line.startswith(collapsed + " "))):
            continue
        words = collapsed.split()
        half = len(words) // 2
        if half and words[:half] == words[half:half * 2] and len(words) % 2 == 0:
            collapsed = " ".join(words[:half])
            s = collapsed
        prev_line = collapsed

        if size > body_size * 1.15 and len(s) < 120:
            flush()
            level = 1 if size > body_size * 1.5 else 2
            kind = "title" if level == 1 and not blocks else "chapter"
            blocks.append(Block(kind, s, level=level, index=len(blocks), source_page=pno))
            continue

        if not buf:
            buf_page = pno
        buf.append(s)
        if s.endswith((".", "!", "?", '"', "\u201d")) and len(" ".join(buf)) > 200:
            flush()

    flush()
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
