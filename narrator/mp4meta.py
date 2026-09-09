"""Write audiobook metadata into an m4b: title, author, cover art.

afconvert encodes the audio and writes almost nothing else, so a finished
file shows up in a player as an untitled recording by nobody. Everything a
reader sees -- the name, the author, the cover, and the fact that this is an
audiobook rather than a song -- lives in iTunes-style atoms under
moov/udta/meta/ilst, and has to be added afterwards.

Unlike chapter marks, a cover is far too large to hide in the padding after
moov, so moov grows and mdat moves. Every chunk offset in every stco table is
corrected by exactly that shift; get this wrong and the file still opens, then
plays silence.
"""
from __future__ import annotations

import struct
from pathlib import Path

from .mp4chapters import CONTAINERS, _box, _find, _full, _parse


def _data(payload: bytes, type_code: int) -> bytes:
    """An iTunes 'data' box: 1 = UTF-8 text, 13 = JPEG, 14 = PNG, 21 = int."""
    return _box(b"data", struct.pack(">II", type_code, 0) + payload)


def _text_atom(name: bytes, value: str) -> bytes:
    return _box(name, _data(value.encode("utf-8"), 1))


def _int_atom(name: bytes, value: int, width: int = 1) -> bytes:
    return _box(name, _data(value.to_bytes(width, "big"), 21))


def build_ilst(title: str = "", author: str = "", album: str = "",
               cover: bytes | None = None, cover_png: bool = True,
               genre: str = "Audiobook", comment: str = "") -> bytes:
    items = b""
    if title:
        items += _text_atom(b"\xa9nam", title)
    if author:
        items += _text_atom(b"\xa9ART", author)      # artist
        items += _text_atom(b"aART", author)         # album artist
        items += _text_atom(b"\xa9wrt", author)      # composer/author
    if album or title:
        items += _text_atom(b"\xa9alb", album or title)
    if genre:
        items += _text_atom(b"\xa9gen", genre)
    if comment:
        items += _text_atom(b"\xa9cmt", comment)
    # stik 2 marks the file as an audiobook, which is what makes a player
    # remember the position and shelve it with books rather than music.
    items += _int_atom(b"stik", 2)
    items += _box(b"pgap", _data(b"\x01", 21))       # gapless
    if cover:
        items += _box(b"covr", _data(cover, 14 if cover_png else 13))
    return _box(b"ilst", items)


def _meta_box(ilst: bytes) -> bytes:
    hdlr = _full(b"hdlr", 0, 0,
                 b"\x00" * 4 + b"mdir" + b"appl" + b"\x00" * 9)
    return _full(b"meta", 0, 0, hdlr + ilst)


def _shift_offsets(data: bytearray, boxes: list[dict], delta: int) -> int:
    """Add delta to every chunk offset. Returns how many were changed."""
    n = 0
    for b in boxes:
        if b["type"] in (b"stco", b"co64"):
            pos = b["start"] + b["head"] + 4          # skip version/flags
            count = struct.unpack_from(">I", data, pos)[0]
            pos += 4
            wide = b["type"] == b"co64"
            for _ in range(count):
                if wide:
                    v = struct.unpack_from(">Q", data, pos)[0]
                    struct.pack_into(">Q", data, pos, v + delta)
                    pos += 8
                else:
                    v = struct.unpack_from(">I", data, pos)[0]
                    struct.pack_into(">I", data, pos, v + delta)
                    pos += 4
                n += 1
        if b["children"]:
            n += _shift_offsets(data, b["children"], delta)
    return n


def write_tags(path: Path | str, *, title: str = "", author: str = "",
               album: str = "", cover_path: Path | str | None = None,
               comment: str = "") -> str:
    path = Path(path)
    data = bytearray(path.read_bytes())
    boxes = _parse(bytes(data), 0, len(data))
    moov = _find(boxes, b"moov")
    if moov is None:
        raise ValueError("no moov box")

    cover = None
    is_png = True
    if cover_path:
        cover_path = Path(cover_path)
        cover = cover_path.read_bytes()
        is_png = cover[:8] == b"\x89PNG\r\n\x1a\n"

    ilst = build_ilst(title=title, author=author, album=album, cover=cover,
                      cover_png=is_png, comment=comment)
    meta = _meta_box(ilst)

    # Replace udta wholesale, but keep anything already in it except an old
    # meta box -- the chapter list lives there and must survive.
    udta = _find(moov["children"], b"udta")
    keep = b""
    if udta is not None:
        for child in udta["children"]:
            if child["type"] != b"meta":
                keep += bytes(data[child["start"]:child["start"] + child["size"]])
    new_udta = _box(b"udta", keep + meta)

    body = bytearray(data[moov["start"] + 8:moov["start"] + moov["size"]])
    if udta is not None:
        lo = udta["start"] - (moov["start"] + 8)
        body = body[:lo] + bytearray(new_udta) + body[lo + udta["size"]:]
    else:
        body += new_udta
    new_moov = struct.pack(">I", len(body) + 8) + b"moov" + bytes(body)

    tail_start = moov["start"] + moov["size"]
    free = next((b for b in boxes
                 if b["type"] == b"free" and b["start"] == tail_start), None)
    absorb = free["size"] if free else 0
    if free:
        tail_start += free["size"]

    growth = len(new_moov) - moov["size"] - absorb
    if growth < 0:                       # keep the leftover as padding
        pad = -growth
        new_moov += struct.pack(">I", pad) + b"free" + b"\x00" * (pad - 8)
        growth = 0

    out = bytearray(data[:moov["start"]]) + bytearray(new_moov) \
        + bytearray(data[tail_start:])

    if growth:
        # mdat has moved by exactly `growth`; every chunk offset must follow.
        shifted = _parse(bytes(out), 0, len(out))
        moved = _shift_offsets(out, [b for b in shifted if b["type"] == b"moov"],
                               growth)
    else:
        moved = 0
    path.write_bytes(bytes(out))
    return (f"tagged: title={title!r} author={author!r}"
            f"{' cover ' + str(len(cover) // 1024) + 'KB' if cover else ''}; "
            f"moov grew {growth} bytes, {moved} chunk offsets corrected")
