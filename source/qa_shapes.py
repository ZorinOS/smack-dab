"""Shape-consistency QA against the source designs.

1. Openness: share of white inside each letter (counters, notches, gaps between
   strokes, measured row by row between the outermost strokes) relative to the
   same letter in its source font. Letters that close up much more than their
   peers are flagged.
2. Tilt: straight edges that are slightly off vertical/horizontal (0.6-8 deg)
   in the output but not in the source -- the signature of distorted junctions.

usage: python3 -I qa_shapes.py SmackDab-unhinted.ttf Anton-Regular.ttf Oswald-Bold-700.ttf
"""
import os
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontops  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402

S, A, O = (TTFont(p) for p in sys.argv[1:4])
sc, ac, oc = S.getBestCmap(), A.getBestCmap(), O.getBestCmap()


def outline(f, name, pre=1.0):
    g = f["glyf"]
    c, e, fl = g[name].getCoordinates(g)
    return np.array(c, float) * pre, list(e), [x & 1 for x in fl]


def openness(pts, ends, flags):
    a = fontops.raster_outline(pts, ends, flags, upx=4.0)
    white = 0
    for r in range(a.shape[0]):
        rr = fontops.runs(a[r])
        if len(rr) > 1:
            white += sum(rr[i + 1][0] - rr[i][1] for i in range(len(rr) - 1))
    return white / max(1, a.sum() + white)


def tilted(pts, ends, flags, min_len=80):
    n, start = 0, 0
    for e in ends:
        P, F = pts[start:e + 1], flags[start:e + 1]
        m = len(P)
        for i in range(m):
            j = (i + 1) % m
            if F[i] and F[j]:
                v = P[j] - P[i]
                L = np.hypot(*v)
                if L >= min_len:
                    ang = np.degrees(np.arctan2(abs(v[1]), abs(v[0])))
                    off = min(ang, 90 - ang)
                    if 0.6 < off < 8:
                        n += 1
        start = e + 1
    return n


rows = []
for cp, name in sorted(sc.items()):
    ch = chr(cp)
    if unicodedata.category(ch) not in ("Lu", "Ll") or S["glyf"][name].isComposite():
        continue
    if cp in ac:
        src = outline(A, ac[cp])
        label = "Anton"
    elif cp in oc:
        src = outline(O, oc[cp], 2.048)
        label = "Oswald"
    else:
        continue
    out = outline(S, name)
    o_src, o_out = openness(*src), openness(*out)
    t_src, t_out = tilted(*src), tilted(*out)
    rows.append((ch, cp, label, o_src, o_out, t_src, t_out))

ratios = np.array([r[4] / r[3] for r in rows if r[3] > 0.02])
med, q1 = np.median(ratios), np.percentile(ratios, 25)
print(f"openness kept vs source: median {med:.2f}, lower quartile {q1:.2f}")
flag = sorted((r for r in rows if r[3] > 0.02 and r[4] / r[3] < med - 0.15), key=lambda r: r[4] / r[3])
print(f"letters that close up much more than their peers ({len(flag)}):")
for r in flag:
    print(f"   {r[0]} U+{r[1]:04X} ({r[2]}) keeps {r[4] / r[3]:.2f} of its white")
tl = [r for r in rows if r[6] > r[5]]
print(f"letters with new slightly-off-axis straight edges ({len(tl)}):")
for r in tl:
    print(f"   {r[0]} U+{r[1]:04X} ({r[2]}) {r[5]} -> {r[6]}")
