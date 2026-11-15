"""Two combining accents on one letter: each accent must clear the letter and
the other accent (HarfBuzz shaping, outlines rasterised in place).

usage: qa_stack.py font.ttf
"""
import sys
import numpy as np
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.basePen import BasePen
from PIL import Image, ImageDraw

path = sys.argv[1]
font = TTFont(path)
gs = font.getGlyphSet()
order = font.getGlyphOrder()
hbf = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
UPX = 6.0                                     # font units per pixel


class Flat(BasePen):
    def __init__(self, gs):
        super().__init__(gs)
        self.polys, self.cur = [], []

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, a, b, c):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, 7)[1:]:
            mt = 1 - t
            self.cur.append(tuple(mt**3 * np.array(p0) + 3 * mt * mt * t * np.array(a)
                                  + 3 * mt * t * t * np.array(b) + t**3 * np.array(c)))

    def _qCurveToOne(self, a, b):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, 6)[1:]:
            mt = 1 - t
            self.cur.append(tuple(mt * mt * np.array(p0) + 2 * mt * t * np.array(a) + t * t * np.array(b)))

    def _closePath(self):
        if len(self.cur) > 2:
            self.polys.append(self.cur)
        self.cur = []
    _endPath = _closePath


_cache = {}


def polys(name):
    if name not in _cache:
        p = Flat(gs)
        gs[name].draw(p)
        _cache[name] = p.polys
    return _cache[name]


def mask(name, dx, dy, W, H, x0, y0):
    acc = np.zeros((H, W), bool)
    for poly in polys(name):
        img = Image.new("1", (W, H), 0)
        ImageDraw.Draw(img).polygon([((x + dx - x0) / UPX, (y0 - (y + dy)) / UPX) for x, y in poly], fill=1)
        acc ^= np.array(img, bool)
    return acc


def check(text):
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hbf, buf, {})
    items, x = [], 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        items.append((order[info.codepoint], x + pos.x_offset, pos.y_offset))
        x += pos.x_advance
    x0, y0 = -1200, 3200
    W, H = int(4000 / UPX), int(4400 / UPX)
    ms = [mask(n, dx, dy, W, H, x0, y0) for n, dx, dy in items]
    # 1 px dilation: touching counts as a clash
    from scipy import ndimage
    bad = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            if (ndimage.binary_dilation(ms[i]) & ms[j]).any():
                bad.append((items[i][0], items[j][0]))
    return bad, [n for n, _, _ in items]


cmap = font.getBestCmap()
bases = [c for c in "aeinouxyzbdhlgpqAEHIMNOUXZАЕИОУЮЯаеиоуюяыэ" if ord(c) in cmap]
above = [m for m in "́̀̂̃̄̆̈̌̉̊̇̋" if ord(m) in cmap]
below = [m for m in "̧̨̣̱̤̥̦" if ord(m) in cmap]
seqs = []
for b in bases:
    for m1 in above:
        for m2 in above:
            if m1 != m2:
                seqs.append(b + m1 + m2)
        for m2 in below:
            seqs.append(b + m2 + m1)
fails = []
for s in seqs:
    bad, names = check(s)
    if bad:
        fails.append((s, bad))
print(f"sequences: {len(seqs)}; clashing: {len(fails)}")
from collections import Counter
print("by mark pair:", Counter(tuple(f"U+{ord(c):04X}" for c in s[1:]) for s, _ in fails).most_common(15))
print("by base:", Counter(s[0] for s, _ in fails).most_common(20))
for s, bad in fails[:30]:
    print("  ", s, [f"U+{ord(c):04X}" for c in s[1:]], bad)

ATTACHED = ("uni0327", "uni0328", "cedilla", "ogonek")
real = []
for s, bad in fails:
    keep = [(a, b) for a, b in bad if not any(b.startswith(x) for x in ATTACHED)]
    if keep:
        real.append((s, keep))
print(f"excluding cedilla/ogonek meeting their letter: {len(real)} clashing")
kinds = Counter()
for s, bad in real:
    for a, b in bad:
        kinds[(a if a in ("uni030B.case", "uni030B", "uni030A.case", "uni030A") or "comb" in a or a.startswith("uni03") else "LETTER", b)] += 1
print(kinds.most_common(30))
