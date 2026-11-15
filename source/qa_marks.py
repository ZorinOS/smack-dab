"""End-to-end check of combining-mark placement with HarfBuzz.

For every encoded letter, shape <letter + combining mark> for common marks
above and below and check, column by column, that the mark's ink doesn't run
into the letter's ink and that it sits over the letter horizontally.

usage: qa_marks.py font.ttf [--all]
"""
import sys
import unicodedata
import numpy as np
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.basePen import BasePen
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fontops  # noqa: E402

path = sys.argv[1]
font = TTFont(path)
gs = font.getGlyphSet()
order = font.getGlyphOrder()
hbf = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
cmap = font.getBestCmap()


def polys(name):
    g = font["glyf"][name]
    c, e, f = g.getCoordinates(font["glyf"])
    if not len(c):
        return []
    return fontops.flatten(np.array(c, float), list(e), [x & 1 for x in f], steps=6)


cache = {}


def ink(name):
    if name not in cache:
        cache[name] = polys(name)
    return cache[name]


def columns(ps, X):
    """Highest and lowest ink of polygons ps on the vertical lines X."""
    top = np.full(len(X), np.nan)
    bot = np.full(len(X), np.nan)
    for P in ps:
        b = np.roll(P, -1, axis=0)
        xa, ya, xb, yb = P[:, 0], P[:, 1], b[:, 0], b[:, 1]
        lo, hi = np.minimum(xa, xb), np.maximum(xa, xb)
        M = (X[:, None] >= lo[None, :]) & (X[:, None] <= hi[None, :])
        if not M.any():
            continue
        ddx = xb - xa
        vert = np.abs(ddx) < 1e-9
        t = (X[:, None] - xa[None, :]) / np.where(vert, 1.0, ddx)[None, :]
        Y = ya[None, :] + t * (yb - ya)[None, :]
        Yt = np.where(vert[None, :], np.maximum(ya, yb)[None, :], Y)
        Yb = np.where(vert[None, :], np.minimum(ya, yb)[None, :], Y)
        tmax = np.where(M, Yt, -np.inf).max(axis=1)
        tmin = np.where(M, Yb, np.inf).min(axis=1)
        top = np.fmax(top, np.where(np.isfinite(tmax), tmax, np.nan))
        bot = np.fmin(bot, np.where(np.isfinite(tmin), tmin, np.nan))
    return top, bot


def span_y(ps, x0, x1, top):
    best = None
    for P in ps:
        n = len(P)
        for i in range(n):
            (xa, ya), (xb, yb) = P[i], P[(i + 1) % n]
            lo, hi = max(min(xa, xb), x0), min(max(xa, xb), x1)
            if lo > hi:
                continue
            ys = (ya, yb) if xa == xb else tuple(ya + (x - xa) / (xb - xa) * (yb - ya) for x in (lo, hi))
            v = max(ys) if top else min(ys)
            if best is None or (v > best if top else v < best):
                best = v
    return best


ABOVE = [0x0301, 0x0300, 0x0308, 0x0304, 0x0302, 0x030C, 0x0306, 0x0307, 0x030B, 0x0303]
BELOW = [0x0323, 0x0331, 0x0324, 0x0325, 0x0326]
problems = {}
ALL = "--all" in sys.argv
letters = [c for c in sorted(cmap) if unicodedata.category(chr(c)) in ("Lu", "Ll", "Lt", "Lo")
           and (ALL or len(unicodedata.normalize("NFD", chr(c))) == 1)]   # --all: precomposed too
for cp in letters:
    for m in ABOVE + BELOW:
        buf = hb.Buffer()
        buf.add_codepoints([cp, m])
        buf.guess_segment_properties()
        hb.shape(hbf, buf, {})
        infos, pos = buf.glyph_infos, buf.glyph_positions
        if len(infos) != 2:
            continue                      # composed by ccmp etc.
        b, k = order[infos[0].codepoint], order[infos[1].codepoint]
        bx = 0
        mx = pos[0].x_advance + pos[1].x_offset
        my = pos[1].y_offset
        mp = [P + (mx, my) for P in ink(k)]
        bp = [P + (bx, 0) for P in ink(b)]
        if not mp or not bp:
            continue
        allm = np.vstack(mp)
        allb = np.vstack(bp)
        x0, x1 = allm[:, 0].min(), allm[:, 0].max()
        bx0, bx1 = allb[:, 0].min(), allb[:, 0].max()
        cx = (x0 + x1) / 2
        issue = None
        X = np.arange(np.floor(x0), x1 + 1, 5.0)
        mt, mb = columns(mp, X)
        bt, bb = columns(bp, X)
        if not (bx0 - 20 <= cx <= bx1 + 20):
            issue = f"off to the side (mark centre {cx:.0f}, letter {bx0:.0f}..{bx1:.0f})"
        elif np.vstack(ink(k))[:, 1].max() >= 300:     # judged by the accent HarfBuzz produced
            d = mb - bt                        # white between letter top and accent bottom
            if np.nanmin(d, initial=np.inf) < -2:
                j = int(np.nanargmin(d))
                issue = f"overlaps letter at x={X[j]:.0f} (accent bottom {mb[j]:.0f} < letter top {bt[j]:.0f})"
        else:
            d = bb - mt
            if np.nanmin(d, initial=np.inf) < -2:
                j = int(np.nanargmin(d))
                issue = f"overlaps letter at x={X[j]:.0f} (accent top {mt[j]:.0f} > letter bottom {bb[j]:.0f})"
        if issue:
            problems.setdefault((b, k), (chr(cp) + chr(m), issue))
            continue

by_base = {}
for (b, k), (s, issue) in problems.items():
    by_base.setdefault(b, []).append((s, k, issue))
print(f"letters tested: {len(letters)}, marks: {len(ABOVE) + len(BELOW)}; failing letter/mark pairs: {len(problems)}")
for b, lst in sorted(by_base.items(), key=lambda kv: -len(kv[1])):
    print(f"  {b} ({len(lst)}): " + "; ".join(f"{s} [{k}] {i}" for s, k, i in lst[:3]))
