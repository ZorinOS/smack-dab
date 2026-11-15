"""Compare hinted and unhinted rendering of every glyph at small sizes.

Flags glyphs where hinting closes or opens a counter (hole count of the
thresholded bitmap changes) or changes the amount of ink by a large factor.

usage: qa_hinting.py hinted.ttf
"""
import sys
import numpy as np
import freetype
from scipy import ndimage
from fontTools.ttLib import TTFont

path = sys.argv[1]
font = TTFont(path)
order = font.getGlyphOrder()
face = freetype.Face(path)


def bitmap(gid, ppem, hinted):
    face.set_pixel_sizes(0, ppem)
    flags = freetype.FT_LOAD_RENDER | (freetype.FT_LOAD_TARGET_MONO if hinted else freetype.FT_LOAD_NO_HINTING)
    if hinted:
        flags |= freetype.FT_LOAD_MONOCHROME
    face.load_glyph(gid, flags)
    bm = face.glyph.bitmap
    if not bm.rows or not bm.width:
        return None
    if bm.pixel_mode == freetype.FT_PIXEL_MODE_MONO:
        rows = []
        for r in range(bm.rows):
            row = bm.buffer[r * bm.pitch:(r + 1) * bm.pitch]
            bits = np.unpackbits(np.array(row, np.uint8))[:bm.width]
            rows.append(bits)
        return np.array(rows, bool)
    a = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.pitch)[:, :bm.width]
    return a >= 128


def holes(b):
    pad = np.pad(~b, 1, constant_values=True)
    lab, n = ndimage.label(pad)
    return n - 1


flags = {}
for ppem in (13, 16, 20, 28):
    for gid, name in enumerate(order):
        if font["glyf"][name].numberOfContours == 0:
            continue
        h, u = bitmap(gid, ppem, True), bitmap(gid, ppem, False)
        if h is None or u is None:
            if (h is None) != (u is None):
                flags.setdefault(name, []).append(f"{ppem}px: vanishes")
            continue
        hh, hu = holes(h), holes(u)
        ia, ua = h.sum(), u.sum()
        msg = []
        if hh < hu:
            msg.append(f"counter closed ({hu}->{hh})")
        if ua and (ia / ua > 1.6 or ia / ua < 0.55) and ua > 12:
            msg.append(f"ink x{ia / ua:.2f}")
        if msg:
            flags.setdefault(name, []).append(f"{ppem}px: " + ", ".join(msg))
print(f"glyphs flagged: {len(flags)} of {len(order)}")
for n, m in sorted(flags.items(), key=lambda kv: -len(kv[1])):
    print(f"  {n}: {'; '.join(m)}")
