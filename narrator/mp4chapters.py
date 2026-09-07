"""Embed chapter marks into an m4b, without ffmpeg or Homebrew.

macOS ships afconvert, which encodes AAC but writes no chapter atoms, so the
marks have to be added afterwards. Two formats are written, because players
disagree about which one to read:

* a QuickTime chapter *track* - a text track the audio track points at with a
  'chap' reference. This is what Apple Books and iTunes read.
* a Nero 'chpl' atom in moov/udta, which VLC, Plex, Audiobookshelf, Kodi and
  most Android players read.

The file is grown in place: an m4b written by afconvert carries a 'free' box
after moov, so the new boxes are taken out of that padding and mdat never
moves. Chunk offsets in stco therefore stay valid and need no rewriting,
which is what makes this safe to do to a finished 50 MB file.
"""
from __future__ import annotations

import struct
from pathlib import Path

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"edts"}


def _parse(data: bytes, start: int, end: int) -> list[dict]:
    """Shallow box list for the range, recursing into container boxes."""
    out = []
    pos = start
    while pos + 8 <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        typ = data[pos + 4:pos + 8]
        head = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            head = 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            break
        box = {"type": typ, "start": pos, "size": size, "head": head,
               "children": []}
        if typ in CONTAINERS:
            box["children"] = _parse(data, pos + head, pos + size)
        out.append(box)
        pos += size
    return out


def _find(boxes: list[dict], *path: bytes) -> dict | None:
    cur = boxes
    box = None
    for want in path:
        box = next((b for b in cur if b["type"] == want), None)
        if box is None:
            return None
        cur = box["children"]
    return box


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + typ + payload


def _full(typ: bytes, version: int, flags: int, payload: bytes) -> bytes:
    return _box(typ, struct.pack(">B3s", version, flags.to_bytes(3, "big")) + payload)


# ------------------------------------------------------------------ builders


def _chpl(chapters: list[tuple[str, float]]) -> bytes:
    """Nero chapter list. Times are in 100-nanosecond units."""
    body = struct.pack(">I", 0) + struct.pack(">B", len(chapters))
    for title, start in chapters:
        name = title.encode("utf-8")[:255]
        body += struct.pack(">Q", int(round(start * 10_000_000)))
        body += struct.pack(">B", len(name)) + name
    return _full(b"chpl", 1, 0, body)


def _text_samples(chapters: list[tuple[str, float]]) -> tuple[bytes, list[int]]:
    """QuickTime text samples: a 16-bit length followed by the bytes."""
    blob = b""
    sizes = []
    for title, _ in chapters:
        name = title.encode("utf-8")[:2000]
        sample = struct.pack(">H", len(name)) + name
        blob += sample
        sizes.append(len(sample))
    return blob, sizes


def _text_trak(track_id: int, timescale: int, duration: int,
               chapters: list[tuple[str, float]], sizes: list[int],
               base_offset: int) -> bytes:
    """A minimal text track holding one sample per chapter."""
    n = len(chapters)

    # A chapter track is not presented on its own, so it is not enabled;
    # only "in movie" is set. AVFoundation ignores a track that claims to be
    # a playable text track.
    tkhd = _full(b"tkhd", 0, 0x2,
                 struct.pack(">II", 0, 0)               # created, modified
                 + struct.pack(">I", track_id)
                 + struct.pack(">I", 0)                 # reserved
                 + struct.pack(">I", duration)
                 + struct.pack(">II", 0, 0)             # reserved
                 + struct.pack(">hh", 0, 0)             # layer, alternate group
                 + struct.pack(">hH", 0, 0)             # volume, reserved
                 + struct.pack(">9i", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0,
                               0x40000000)
                 + struct.pack(">II", 0, 0))            # width, height

    mdhd = _full(b"mdhd", 0, 0,
                 struct.pack(">IIII", 0, 0, timescale, duration)
                 + struct.pack(">HH", 0x15C7, 0))       # 'eng', quality
    hdlr = _full(b"hdlr", 0, 0,
                 b"\x00" * 4 + b"text" + b"\x00" * 12 + b"Chapters\x00")

    starts = [int(round(s * timescale)) for _, s in chapters]
    deltas = [starts[i + 1] - starts[i] for i in range(n - 1)] + \
             [max(1, duration - starts[-1])]
    stts = _full(b"stts", 0, 0, struct.pack(">I", n)
                 + b"".join(struct.pack(">II", 1, d) for d in deltas))

    # QuickTime 'text' sample entry, 52 bytes after the box header. The layout
    # is fixed; AVFoundation rejects the track if it does not match.
    text_entry = (
        b"\x00" * 6                       # reserved
        + struct.pack(">H", 1)            # data reference index
        + struct.pack(">i", 0)            # display flags
        + struct.pack(">i", 1)            # text justification: centred
        + struct.pack(">HHH", 0, 0, 0)    # background colour
        + struct.pack(">hhhh", 0, 0, 0, 0)  # default text box
        + struct.pack(">II", 0, 0)        # reserved
        + struct.pack(">h", 0)            # font number
        + struct.pack(">h", 0)            # font face
        + struct.pack(">b", 0)            # reserved
        + struct.pack(">h", 0)            # reserved
        + struct.pack(">HHH", 0, 0, 0)    # foreground colour
        + b"\x00"                         # font name, empty pascal string
    )
    stsd = _full(b"stsd", 0, 0, struct.pack(">I", 1)
                 + _box(b"text", text_entry))

    stsc = _full(b"stsc", 0, 0, struct.pack(">I", 1)
                 + struct.pack(">III", 1, 1, 1))
    stsz = _full(b"stsz", 0, 0, struct.pack(">II", 0, n)
                 + b"".join(struct.pack(">I", s) for s in sizes))
    offsets, run = [], base_offset
    for s in sizes:
        offsets.append(run)
        run += s
    stco = _full(b"stco", 0, 0, struct.pack(">I", n)
                 + b"".join(struct.pack(">I", o) for o in offsets))

    dinf = _box(b"dinf", _full(b"dref", 0, 0, struct.pack(">I", 1)
                               + _full(b"url ", 0, 1, b"")))
    gmin = _full(b"gmin", 0, 0, struct.pack(">HHHHHH", 0, 0x8000, 0x8000,
                                            0x8000, 0, 0))
    gmhd = _box(b"gmhd", gmin)
    stbl = _box(b"stbl", stsd + stts + stsc + stsz + stco)
    minf = _box(b"minf", gmhd + dinf + stbl)
    mdia = _box(b"mdia", mdhd + hdlr + minf)
    return _box(b"trak", tkhd + mdia)


def embed(path: Path | str, chapters: list[tuple[str, float]]) -> str:
    """Add chapter marks to an m4b in place. Returns a short report."""
    path = Path(path)
    data = bytearray(path.read_bytes())
    boxes = _parse(bytes(data), 0, len(data))

    moov = _find(boxes, b"moov")
    if moov is None:
        raise ValueError("no moov box; not an MP4 file")
    mvhd = _find(moov["children"], b"mvhd")
    trak = next((b for b in moov["children"] if b["type"] == b"trak"), None)
    if mvhd is None or trak is None:
        raise ValueError("missing mvhd or trak")

    version = data[mvhd["start"] + 8]
    off = mvhd["start"] + 12
    if version == 1:
        timescale = struct.unpack_from(">I", data, off + 16)[0]
        duration = struct.unpack_from(">Q", data, off + 20)[0]
    else:
        timescale = struct.unpack_from(">I", data, off + 8)[0]
        duration = struct.unpack_from(">I", data, off + 12)[0]

    tkhd = _find(trak["children"], b"tkhd")
    audio_id = struct.unpack_from(">I", data, tkhd["start"] + 20)[0]
    next_id = audio_id + 1

    # text samples go in their own mdat appended at the end of the file
    blob, sizes = _text_samples(chapters)
    text_mdat_off = len(data) + 8
    new_mdat = _box(b"mdat", blob)

    text_trak = _text_trak(next_id, timescale, duration, chapters, sizes,
                           text_mdat_off)
    tref = _box(b"tref", _box(b"chap", struct.pack(">I", next_id)))
    chpl = _chpl(chapters)

    # Grow moov out of the free box that follows it, so mdat never moves.
    free = next((b for b in boxes
                 if b["type"] == b"free" and b["start"] > moov["start"]), None)
    growth = len(text_trak) + len(tref) + len(chpl)
    udta = _find(moov["children"], b"udta")
    if udta is None:
        growth += 8
    if free is None or free["size"] < growth + 8:
        raise ValueError(f"not enough free padding: need {growth}, "
                         f"have {free['size'] if free else 0}")

    # Rebuild moov with the additions, then shrink the free box to match.
    moov_body = bytearray(data[moov["start"] + 8:moov["start"] + moov["size"]])
    rel_trak_end = trak["start"] + trak["size"] - (moov["start"] + 8)
    rel_trak_start = trak["start"] - (moov["start"] + 8)

    trak_bytes = bytearray(moov_body[rel_trak_start:rel_trak_end])
    trak_bytes[0:4] = struct.pack(">I", len(trak_bytes) + len(tref))
    trak_bytes += tref                      # tref goes at the end of trak

    if udta is not None:
        rel_udta_start = udta["start"] - (moov["start"] + 8)
        rel_udta_end = rel_udta_start + udta["size"]
        udta_bytes = bytearray(moov_body[rel_udta_start:rel_udta_end])
        udta_bytes[0:4] = struct.pack(">I", len(udta_bytes) + len(chpl))
        udta_bytes += chpl
    else:
        rel_udta_start = rel_udta_end = len(moov_body)
        udta_bytes = bytearray(_box(b"udta", chpl))

    lo, hi = sorted([(rel_trak_start, rel_trak_end, trak_bytes),
                     (rel_udta_start, rel_udta_end, udta_bytes)],
                    key=lambda t: t[0])
    rebuilt = (moov_body[:lo[0]] + lo[2] + moov_body[lo[1]:hi[0]] + hi[2]
               + moov_body[hi[1]:])
    rebuilt += text_trak                    # new track at the end of moov

    new_moov = struct.pack(">I", len(rebuilt) + 8) + b"moov" + bytes(rebuilt)
    actual_growth = len(new_moov) - moov["size"]
    new_free_size = free["size"] - actual_growth
    if new_free_size < 8:
        raise ValueError("free box too small after rebuild")
    new_free = struct.pack(">I", new_free_size) + b"free" + \
        b"\x00" * (new_free_size - 8)

    out = (bytes(data[:moov["start"]]) + new_moov + new_free
           + bytes(data[free["start"] + free["size"]:]) + new_mdat)
    path.write_bytes(out)
    return (f"{len(chapters)} chapters embedded "
            f"(QuickTime chapter track + Nero chpl); "
            f"file {len(data):,} -> {len(out):,} bytes")
