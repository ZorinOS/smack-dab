"""Check every pair of letters/punctuation for collisions after layout.

Shapes each pair with HarfBuzz (kerning on), then measures the narrowest white
gap between the two glyphs' outlines row by row. Reports pairs that touch or
overlap, separately for kerned pairs and for pairs with no kerning at all.

usage: qa_pairs.py font.ttf [min_gap] [--plain]   (--plain: letters without accents only)
"""
import sys
import unicodedata
import numpy as np
import uharfbuzz as hb
from fontTools.ttLib import TTFont
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fontops  # noqa: E402

path = sys.argv[1]
MIN_GAP = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else 0.0
font = TTFont(path)
glyf = font["glyf"]
order = font.getGlyphOrder()
cmap = font.getBestCmap()
hbf = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
Y = np.arange(-450, 2100, 10.0)


def rows(name):
    c, e, f = glyf[name].getCoordinates(glyf)
    if not len(c):
        return None
    polys = fontops.nonzero_polys(np.array(c, float), list(e), [x & 1 for x in f])
    L = np.full(len(Y), np.nan)
    R = np.full(len(Y), np.nan)
    for i, y in enumerate(Y):
        ink = fontops._scan_gaps(polys, y)
        if ink:
            L[i], R[i] = ink[0][0], ink[-1][1]
    return L, R


PLAIN = "--plain" in sys.argv
chars = [c for c in sorted(cmap)
         if unicodedata.category(chr(c))[0] in "LNP" and c < 0x2000
         and (not PLAIN or len(unicodedata.normalize("NFD", chr(c))) == 1)
         and (0x20 < c < 0x250 or 0x400 <= c < 0x530 or c in (0x2013, 0x2014))]
prof = {}
for c in chars:
    p = rows(cmap[c])
    if p is not None:
        prof[c] = p
chars = [c for c in chars if c in prof]
kerned, plain = [], []
for a in chars:
    La, Ra = prof[a]
    for b in chars:
        buf = hb.Buffer()
        buf.add_codepoints([a, b])
        buf.guess_segment_properties()
        hb.shape(hbf, buf, {})
        if len(buf.glyph_infos) != 2:
            continue
        ga, gb = order[buf.glyph_infos[0].codepoint], order[buf.glyph_infos[1].codepoint]
        if ga != cmap[a] or gb != cmap[b]:
            continue                       # substituted (ligature, case form...)
        adv = buf.glyph_positions[0].x_advance
        kern = adv - font["hmtx"][ga][0]
        Lb, Rb = prof[b]
        d = Lb + adv - Ra
        if not np.isfinite(d).any():
            continue
        j = int(np.nanargmin(d))
        gap = d[j]
        if gap < MIN_GAP:
            zone = "body" if -300 < Y[j] < 1650 else "accents"
            (kerned if kern else plain).append((gap, chr(a) + chr(b), kern, zone))
kerned.sort()
plain.sort()
print(f"characters: {len(chars)}, pairs: {len(chars) ** 2}")
print(f"kerned pairs closer than {MIN_GAP:.0f} units: {len(kerned)}")
for g, s, k, z in kerned[:60]:
    print(f"   {s}  gap {g:.0f}  kern {k}  ({z})")
body = [p for p in plain if p[3] == "body"]
print(f"unkerned pairs closer than {MIN_GAP:.0f} units: {len(plain)} "
      f"({len(body)} touching in the letter bodies, the rest only where accents meet)")
print("   body: " + " ".join(f"{s}({g:.0f})" for g, s, k, z in body[:150]))
