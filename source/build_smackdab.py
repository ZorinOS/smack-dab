#!/usr/bin/env python3
"""Build Smack Dab: Anton re-proportioned to a reference font's metrics.

Inputs
  * Anton-Regular.ttf (SIL OFL 1.1): the only source of outlines, OpenType
    layout, and glyph data.
  * impact_metrics.json: numbers only (advance widths, horizontal ink
    extents, line metrics), produced by measure_impact.py.

Method (per glyph)
  * Vertical: uniform scale so cap height / x-height land on the reference
    zones. Lowercase ascenders are lengthened by a rigid shift above a "cut"
    placed where only straight stems cross, so no curve or horizontal stroke
    is distorted.
  * Horizontal: scale + x-only emboldening (FreeType's EmboldenXY algorithm)
    chosen so the ink width matches the reference while vertical stems keep a
    uniform target weight and counters never collapse.
  * Advance widths are set exactly to the reference values, so layout is
    identical. Composites, mark anchors and kerning are re-derived from the
    transformed Anton geometry.

usage: python3 -I build_smackdab.py Anton-Regular.ttf impact_metrics.json out.ttf
"""
import json
import math
import statistics
import sys
import unicodedata
from datetime import datetime, timezone

import os

import freetype
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontops  # noqa: E402
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent, GlyphCoordinates

ANTON, METRICS, OUT = sys.argv[1:4]

# ---------------------------------------------------------------- parameters
K_STEM = 1.17          # stem weight relative to Anton's, where the reference gives none (symbols)
COUNTER_KEEP = 0.55    # never let a scaled counter shrink below this fraction
STEM_KEEP = 0.55       # never thin a stem below this fraction
H_MIN, H_MAX = 0.5, 1.9

ref = json.load(open(METRICS))
RCH = {int(k, 16): v for k, v in ref["chars"].items()}
# Where the reference's ink placement would make text collide, only its
# advance width (all that layout depends on) is kept and the ink is placed
# sensibly instead:
#  · U+00B7 sits at the far right of its advance and fuses with the next
#    letter ("col·lecció" reads "col łecció"): centred instead;
#  ∕ U+2215, a maths operator, spans 786 units in a 256-unit advance and runs
#    into both neighbours: drawn steeper, overhanging 60 units each side.
#    (The fraction slash U+2044 overhangs by design, to build 1⁄2, as in Anton.)
for _cp, _how in ((0xB7, "centre"), (0x2215, "steep")):
    _r = RCH.get(_cp)
    if _r and "xMin" in _r:
        _w = _r["xMax"] - _r["xMin"]
        if _how == "centre":
            _r["xMin"] = round((_r["adv"] - _w) / 2)
            _r["xMax"] = _r["xMin"] + _w
        else:
            _r["xMin"], _r["xMax"] = -60, _r["adv"] + 60
font = TTFont(ANTON)
src = TTFont(ANTON)                 # untouched copy for measurements
glyf, hmtx = font["glyf"], font["hmtx"]
order = font.getGlyphOrder()
cmap = font.getBestCmap()
rev = {}
for cp, g in cmap.items():
    rev.setdefault(g, cp)
gdef_cls = font["GDEF"].table.GlyphClassDef.classDefs

CAP_A, XH_A = font["OS/2"].sCapHeight, font["OS/2"].sxHeight      # 1760, 1500
CAP_R, XH_R = ref["zones"]["capHeight"], ref["zones"]["xHeight"]  # 1619, 1327
VC = CAP_R / CAP_A
VX = XH_R / XH_A
ASC_SHIFT = CAP_R - VX * CAP_A      # lengthen lowercase ascenders to cap height
print(f"VC={VC:.4f} VX={VX:.4f} ascender shift={ASC_SHIFT:.1f}")

SPACING_ACCENTS = {"grave", "acute", "circumflex", "tilde", "macron", "breve",
                   "dotaccent", "dieresis", "ring", "cedilla", "hungarumlaut",
                   "ogonek", "caron", "uni02C9"}


def basename(g):
    return g.split(".")[0]


def is_markish(g):
    return (gdef_cls.get(g) == 3 or basename(g) in SPACING_ACCENTS
            or (hmtx[g][0] == 0 and glyf[g].numberOfContours != 0))


def base_char(g):
    """Best-guess character a glyph represents (for classification)."""
    if g in rev:
        return chr(rev[g])
    b = basename(g)
    if b in font.getReverseGlyphMap() and b in rev:
        return chr(rev[b])
    if b.startswith("uni") and len(b) == 7:
        try:
            return chr(int(b[3:], 16))
        except ValueError:
            pass
    if len(b) == 1:
        return b
    first = b.split("_")[0]
    if first in rev:
        return chr(rev[first])
    return None


def vclass(g):
    """Vertical-map class: 'lc' lowercase, 'uc' everything cap-height based,
    'mk' lowercase-size marks, 'mkc' cap-size (.case) marks."""
    if is_markish(g):
        return "mkc" if ".case" in g else "mk"
    if g.endswith(".sc") or ".sc." in g:
        return "lc"
    if any(s in g for s in (".sups", ".subs", ".sinf", ".numr", ".dnom")):
        return "uc"
    ch = base_char(g)
    if ch and unicodedata.category(ch) == "Ll":
        return "lc"
    return "uc"


# ------------------------------------------------------------ raster helpers
ft_face = freetype.Face(ANTON)
FT_PPEM = 1024
ft_face.set_char_size(FT_PPEM * 64, 0, 72, 72)
U = font["head"].unitsPerEm / FT_PPEM            # font units per pixel
FT_FLAGS = freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_BITMAP


def render(face, gid):
    face.load_glyph(gid, FT_FLAGS)
    bm = face.glyph.bitmap
    a = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.width) if bm.rows else np.zeros((0, 0))
    return a > 127, face.glyph.bitmap_left, face.glyph.bitmap_top


def runs(row):
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(starts, ends))


def stroke_stats(g):
    """Typical vertical-stem thickness (horizontal runs), smallest horizontal
    counter, and typical perpendicular stroke width (distance-transform ridge)."""
    a, left, top = render(ft_face, order.index(g))
    if a.size == 0:
        return None, None, None
    rows = a.shape[0]
    ink_runs, gaps = [], []
    for frac in np.linspace(0.18, 0.82, 13):
        r = int(rows * frac)
        rr = runs(a[r])
        ink_runs += [(e - s) for s, e in rr]
        gaps += [(rr[i + 1][0] - rr[i][1]) for i in range(len(rr) - 1)]
    W = a.shape[1]
    stems = [w for w in ink_runs if w < 0.7 * W] or ink_runs or [W]
    stem = float(np.median(stems)) * U
    gap = float(np.percentile(gaps, 25)) * U if gaps else math.inf
    dt = ndimage.distance_transform_edt(np.pad(a, 1))
    ridge = (dt > 1) & (dt >= ndimage.maximum_filter(dt, size=3))
    sw = 2 * float(np.median(dt[ridge])) * U if ridge.any() else stem
    return stem, gap, sw


# --------------------------------------------------------------- outline ops
def contours_of(gl):
    coords = np.array(gl.coordinates, dtype=float)
    return coords, list(gl.endPtsOfContours), list(gl.flags)


def embolden_x(pts, ends, xstrength):
    """Port of FreeType's FT_Outline_EmboldenXY with ystrength = 0.
    Right-facing edges move by xstrength, left-facing edges stay."""
    if abs(xstrength) < 1e-9:
        return pts
    xs = xstrength / 2.0
    # orientation by total signed area of control polygon
    area, first = 0.0, 0
    for last in ends:
        p = pts[first:last + 1]
        area += np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        first = last + 1
    truetype = area < 0
    out_pts = pts.copy()
    first = 0
    for last in ends:
        idx = list(range(first, last + 1))
        n = len(idx)
        P = pts[first:last + 1]
        for i in range(n):
            # previous / next distinct points
            j = (i - 1) % n
            while j != i and np.allclose(P[j], P[i]):
                j = (j - 1) % n
            k = (i + 1) % n
            while k != i and np.allclose(P[k], P[i]):
                k = (k + 1) % n
            if j == i or k == i:
                continue
            vin = P[i] - P[j]
            vout = P[k] - P[i]
            l_in, l_out = np.hypot(*vin), np.hypot(*vout)
            vin, vout = vin / l_in, vout / l_out
            d = vin @ vout
            if d > -0.9375:
                d += 1.0
                sx = vin[1] + vout[1]
                if truetype:
                    sx = -sx
                q = vout[0] * vin[1] - vout[1] * vin[0]
                if truetype:
                    q = -q
                l = min(l_in, l_out)
                if xs * q <= l * d:
                    sx = sx * xs / d
                else:
                    sx = sx * l / q
            else:
                sx = 0.0
            out_pts[first + i, 0] = P[i, 0] + xs + sx
        first = last + 1
    return out_pts


def find_cut(coords, ends, flags):
    """For lowercase glyphs with an ascender: lowest y above the x-height where
    only straight segments cross. Returns (cut, band_fallback)."""
    spans, first = [], 0
    for last in ends:
        ys = coords[first:last + 1, 1]
        spans.append((ys.min(), ys.max()))
        first = last + 1
    if not any(lo < XH_A - 100 and hi > XH_A + 150 for lo, hi in spans):
        return None, False
    ys = sorted(set(np.round(coords[:, 1]).astype(int)))
    cands = [(a, b) for a, b in zip(ys, ys[1:]) if b - a >= 8 and b > XH_A + 3 and a < CAP_A - 40]
    for a, b in cands:
        cut = max(a, XH_A) + (b - max(a, XH_A)) / 2
        if b - max(a, XH_A) < 8:
            continue
        clean, first = True, 0
        for last in ends:
            idx = list(range(first, last + 1))
            for m in range(len(idx)):
                p, q = idx[m], idx[(m + 1) % len(idx)]
                if (coords[p, 1] - cut) * (coords[q, 1] - cut) < 0:
                    if not (flags[p] & 1 and flags[q] & 1):
                        clean = False
            first = last + 1
        if clean:
            return cut, False
    return None, True


def ymap_array(ys, cls, cut=None, band=False):
    ys = np.asarray(ys, dtype=float)
    if cls in ("uc", "mkc"):
        return ys * VC
    if cls == "mk":
        return ys * VX
    out = ys * VX
    if cut is not None:
        out = out + ASC_SHIFT * (ys > cut)
    elif band:
        lo, hi = XH_A, CAP_A
        mid = (ys > lo) & (ys <= hi)
        out = np.where(mid, XH_R + (ys - lo) * (CAP_R - XH_R) / (hi - lo), out)
        out = np.where(ys > hi, CAP_R + (ys - hi) * VX, out)
    return out


# ---------------------------------------------------------------- transforms
T = {}        # simple glyph name -> dict(h, delta, tx, cls, cut, band, adv)


def _y_room(y, cls, ystr):
    """Stretch the mapped heights so that, after the top-facing edges move by
    ystr (bottom-facing edges stay), the letter still spans baseline to cap
    height / x-height and parts above it keep their place."""
    t = -np.asarray(ystr, float) * np.ones(len(y))
    if not np.any(t):
        return np.asarray(y, float)
    Z = XH_R if cls == "lc" else CAP_R
    y = np.array(y, float)
    body = (y > 0) & (y <= Z)
    up = y > Z                       # (decided before the body is stretched)
    y[body] = y[body] * (Z + t[body]) / Z
    y[up] += t[up]
    return y


def apply_simple(g, h, delta, cls, cut, band, ystr=0.0):
    """Scale horizontally by h, map vertical zones, then change stem weight by
    delta (and horizontal-stroke weight by ystr): each edge moves right (up)
    in proportion to how much it faces right (up), FreeType-style, so round
    counters keep their height, with exact miter joins, so straight edges stay
    straight; loops are resolved (nonzero)."""
    gl = src["glyf"][g]
    coords, ends, flags = contours_of(gl)
    pts = np.column_stack([coords[:, 0] * h, ymap_array(coords[:, 1], cls, cut, band)])
    if abs(delta) < 0.5 and abs(ystr) < 0.5:
        return pts, list(ends), [f & 1 for f in flags]
    ys = ystr
    if ystr < 0:                     # thinning: notches keep their depth
        ys = ystr * fontops.notch_weights(pts, list(ends), [f & 1 for f in flags], radius=max(60.0, -2 * ystr))
    pts[:, 1] = _y_room(pts[:, 1], cls, ys)
    pts = fontops.embolden_xy_v2(pts, ends, delta, ys)
    return fontops.clean_nonzero(pts, ends, flags)


def fit_simple(g, target_w, stem, gap, sw, cls, cut, band, kind="alnum", target_stem=None):
    """Choose h (horizontal scale) and delta (stem-weight change) so the ink
    width matches target_w while vertical stems become K_STEM x Anton's
    (measured as horizontal stroke runs), within limits that keep counters
    open; then make sure no counter or opening between strokes has closed."""
    gl = src["glyf"][g]
    WA = gl.xMax - gl.xMin
    HA = gl.yMax - gl.yMin
    stem = stem or WA
    if stem >= 0.7 * WA and HA < 0.6 * WA:
        # horizontal bars (dashes, minus, equals, underscore): stretching
        # sideways distorts nothing, so match the width with pure scaling
        return target_w / WA, 0.0

    def measure(h, d):
        p = apply_simple(g, h, d, cls, cut, band)[0]
        return float(np.ptp(p[:, 0]))
    h, delta = fontops.fit_width(WA, target_w, stem, gap, sw, target_stem or K_STEM * stem, kind, measure)
    # closure guard: no counter may close (or appear) and no parts may merge,
    # compared with the same glyph at this scale without weight change
    def same_topology(d):
        return fontops.topology(*apply_simple(g, h, d, cls, cut, band))
    if delta > 0:
        base = fontops.topology(*apply_simple(g, h, 0.0, cls, cut, band))
        lo_d, hi_d = 0.0, delta
        cand = same_topology(delta)
        if (cand[0], cand[2]) != (base[0], base[2]):
            for _ in range(7):
                mid = (lo_d + hi_d) / 2
                m = same_topology(mid)
                if (m[0], m[2]) == (base[0], base[2]):
                    lo_d = mid
                else:
                    hi_d = mid
            print(f"   closure guard: {g} delta {delta:.0f} -> {lo_d:.0f}")
            delta = lo_d
    return h, delta


def store_simple(g, h, delta, cls, cut, band, xmin_target, adv, center=None, ystr=0.0):
    pts, ends, flags = apply_simple(g, h, delta, cls, cut, band, ystr)
    if center is not None:   # place ink centre at the reference ink centre
        tx = center - (pts[:, 0].max() + pts[:, 0].min()) / 2
    else:
        tx = xmin_target - pts[:, 0].min()
    pts[:, 0] += tx
    ng = Glyph()
    ng.numberOfContours = len(ends)
    ng.coordinates = GlyphCoordinates([(int(round(x)), int(round(y))) for x, y in pts])
    ng.flags = list(flags)
    ng.endPtsOfContours = list(ends)
    ng.program = ttProgram.Program()
    ng.program.fromBytecode(b"")
    glyf[g] = ng
    T[g] = dict(h=h, delta=delta, tx=tx, cls=cls, cut=cut, band=band, adv=int(round(adv)), ystr=ystr)


# ---- classify simple glyphs
simple = [g for g in order if not glyf[g].isComposite() and glyf[g].numberOfContours > 0]
fitted, marks, others = [], [], []
for g in simple:
    if is_markish(g):
        marks.append(g)
    elif g in rev and rev[g] in RCH and "xMin" in RCH[rev[g]]:
        fitted.append(g)
    else:
        others.append(g)

stats = {g: stroke_stats(g) for g in fitted + others}
cuts = {}
for g in fitted + others:
    if vclass(g) == "lc":
        c, ends, fl = contours_of(src["glyf"][g])
        cuts[g] = find_cut(c, ends, fl)
    else:
        cuts[g] = (None, False)

# letters drawn as base letter + stroke/slash/bar: reuse the base transform
DERIVED = {"Oslash": "O", "oslash": "o", "Lslash": "L", "lslash": "l", "Dcroat": "D",
           "dcroat": "d", "Eth": "D", "Hbar": "H", "hbar": "h", "Tbar": "T", "tbar": "t",
           "uni0180": "b", "uni0197": "I", "uni0268": "i", "uni024C": "R", "uni024D": "r",
           "uni01E4": "G", "uni01E5": "g", "uni0246": "E", "uni0247": "e", "uni023B": "C",
           "uni023C": "c", "uni0248": "J", "uni0249": "j", "uni024E": "Y", "uni024F": "y",
           "uni2C65": "a", "uni023A": "A", "uni0244": "U", "uni0289": "u", "uni01B5": "Z",
           "uni01B6": "z", "uni0243": "B", "uni2C66": "t"}


def kind_of(g):
    ch = base_char(g)
    return "alnum" if ch and unicodedata.category(ch)[0] in "LN" else "sym"


# ---- stroke weights from the reference (numbers measured by measure_impact.py)
# Vertical stems: the reference's own stem weight for each letter and figure
# (where it is within 15% of Anton's weight x K_STEM; elsewhere, e.g. for
# symbols, K_STEM). Horizontal strokes (arches, bowl tops and bottoms, bars):
# moved toward the reference's weight by changing the y weight, where two
# measures agree (median and lower quartile of the strokes crossed by the middle
# columns), at most 60 units thinner or 30 heavier, never thinning a glyph's
# thinnest horizontal by more than 45%, and never changing its topology.
STEM_RATIO = {}                  # vertical class -> [target / (K_STEM x Anton)]
YSTR = {}                        # vertical class -> [ystr]


def stem_target(g, stem, r):
    if not stem or kind_of(g) != "alnum" or not r or not r.get("stem"):
        return None
    ratio = r["stem"] / (K_STEM * stem)
    if not 0.85 <= ratio <= 1.15:
        return None
    STEM_RATIO.setdefault(vclass(g), []).append(ratio)
    return float(r["stem"])


def hbar_stats(pts, ends, flags):
    a = fontops.raster_outline(pts, ends, flags, upx=4.0)
    h, w = a.shape
    vr = []
    for fr in np.linspace(0.3, 0.7, 9):
        col = a[:, int(w * fr)]
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        vr += list((np.where(d == -1)[0] - np.where(d == 1)[0]) * 4.0)
    vr = [v for v in vr if v < 0.6 * h * 4.0]
    if not vr:
        return None
    return float(np.median(vr)), float(np.percentile(vr, 25)), float(min(vr))


def _shape(top):
    return top[0], top[2]            # counters and pieces (not the counters' area)


def y_ok(g, h, d, cls, cut, band, ystr, base_top):
    return _shape(fontops.topology(*apply_simple(g, h, d, cls, cut, band, ystr))) == _shape(base_top)


# diagonal strokes (their vertical runs are not horizontal strokes), dots, flags
Y_SKIP = set("AKMNVWXYkvwxy1247ijrIJ")


def y_eligible(g):
    """Full-size letters and figures: not superiors, ordinals, fraction
    figures or other size variants, which keep their weight."""
    ch = base_char(g)
    return (ch is not None and "." not in g and ch not in Y_SKIP
            and unicodedata.category(ch) in ("Lu", "Ll", "Nd"))


def fit_ystr(g, h, d, cls, cut, band, r):
    if not y_eligible(g):
        return 0.0
    base = apply_simple(g, h, d, cls, cut, band)
    st0 = hbar_stats(*base)
    if st0 is None or "hbar" not in r or "hbar_lq" not in r:
        return 0.0
    m0, q0, mn0 = st0
    t1, t2 = m0 - r["hbar"], q0 - r["hbar_lq"]
    if t1 > 8 and t2 > 8:
        t = min(t1, t2, 60.0, 0.45 * mn0)
    elif t1 < -8 and t2 < -8:
        t = max(t1, t2, -30.0)
    else:
        YSTR.setdefault(cls, []).append(0.0)
        return 0.0
    top0 = fontops.topology(*base)
    # stroke edges move by ~85% of t once the room is made: correct once
    st1 = hbar_stats(*apply_simple(g, h, d, cls, cut, band, -t))
    if st1 is not None and abs(m0 - st1[0]) > 4:
        t = t * min(1.4, max(0.7, t / (m0 - st1[0])))
        t = min(t, 60.0, 0.45 * mn0) if t > 0 else max(t, -30.0)
    for _ in range(4):
        if y_ok(g, h, d, cls, cut, band, -t, top0):
            YSTR.setdefault(cls, []).append(-t)
            return -t
        t *= 0.5
    YSTR.setdefault(cls, []).append(0.0)
    return 0.0


# ---- pass 1: glyphs whose reference metrics are known
derived_later = []
for g in fitted:
    if g in DERIVED and DERIVED[g] in fitted:
        derived_later.append(g)
        continue
    r = RCH[rev[g]]
    cls = vclass(g)
    cut, band = cuts[g]
    stem, gap, sw = stats[g]
    tw = r["xMax"] - r["xMin"]
    h, d = fit_simple(g, tw, stem, gap, sw, cls, cut, band, kind_of(g), target_stem=stem_target(g, stem, r))
    ys = fit_ystr(g, h, d, cls, cut, band, r) if kind_of(g) == "alnum" else 0.0
    want_center = (r["xMax"] + r["xMin"]) / 2
    store_simple(g, h, d, cls, cut, band, r["xMin"], r["adv"], center=want_center, ystr=ys)
print("stem targets / (K_STEM x Anton):", {k: round(statistics.median(v), 3) for k, v in STEM_RATIO.items()},
      "| y weight change, median:", {k: round(statistics.median(v), 1) for k, v in YSTR.items()})


def place_like_base(g, b, adv_new, cls, cut, band):
    """Transform g exactly like letter b (same scale and stem weight) and keep
    it aligned with b's x-map, re-centred if g's advance differs from b's."""
    tb = T[b]
    shift = (adv_new - tb["adv"]) / 2
    ys = tb.get("ystr", 0.0) if y_eligible(g) else 0.0
    if ys and _shape(fontops.topology(*apply_simple(g, tb["h"], tb["delta"], cls, cut, band, ys))) != \
            _shape(fontops.topology(*apply_simple(g, tb["h"], tb["delta"], cls, cut, band))):
        ys = 0.0
    pts, _, _ = apply_simple(g, tb["h"], tb["delta"], cls, cut, band, ys)
    store_simple(g, tb["h"], tb["delta"], cls, cut, band,
                 pts[:, 0].min() + tb["tx"] + shift, adv_new, ystr=ys)


def squeeze_overhang(g, b, lo, hi):
    """Keep a derived letter's extras (bars, slashes) within [lo, hi] by
    compressing only the parts outside its base letter's ink, so the base
    letter's stems stay exactly as they are."""
    gl = glyf[g]
    co = np.array(gl.coordinates, float)
    x0, x1 = co[:, 0].min(), co[:, 0].max()
    if x0 >= lo and x1 <= hi:
        return
    bco = np.array(glyf[b].coordinates, float)
    shift = (T[g]["adv"] - T[b]["adv"]) / 2
    bx0, bx1 = bco[:, 0].min() + shift, bco[:, 0].max() + shift
    xs = co[:, 0].copy()
    if x0 < lo < bx0:
        m = xs < bx0
        xs[m] = bx0 - (bx0 - xs[m]) * (bx0 - lo) / (bx0 - x0)
    if x1 > hi > bx1:
        m = xs > bx1
        xs[m] = bx1 + (xs[m] - bx1) * (hi - bx1) / (x1 - bx1)
    co[:, 0] = xs
    gl.coordinates = GlyphCoordinates([(int(round(x)), int(round(y))) for x, y in co])


for g in derived_later:
    cut, band = cuts[g]
    r = RCH[rev[g]]
    place_like_base(g, DERIVED[g], r["adv"], vclass(g), cut, band)
    squeeze_overhang(g, DERIVED[g], min(r["xMin"], 0) - 10, max(r["xMax"], r["adv"]) + 10)


def med(vals, default=1.0):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else default


def letters(cat):
    return [g for g in fitted if unicodedata.category(chr(rev[g])) == cat and rev[g] < 0x80]


H_GEN = {
    "Lu": med(T[g]["h"] for g in letters("Lu")),
    "Ll": med(T[g]["h"] for g in letters("Ll")),
    "Nd": med(T[g]["h"] for g in fitted if unicodedata.category(chr(rev[g])) == "Nd"),
}
H_GEN["other"] = med(T[g]["h"] for g in fitted)
# accents keep the size they were tuned at: the horizontal scale lowercase
# letters get with K_STEM x Anton's stems (the reference's lighter stems
# change how width is split between scale and weight, not the accents)
_hm = []
for g in letters("Ll"):
    _r = RCH[rev[g]]
    _st = stats[g]
    if g in DERIVED or _st[0] is None:
        continue
    _hm.append(fit_simple(g, _r["xMax"] - _r["xMin"], *_st, vclass(g), *cuts[g], kind_of(g))[0])
H_MARK = med(_hm, H_GEN["Ll"])
print("accent scale", round(H_MARK, 3))
# spacing (sidebearing) scale between Anton and the reference
sb_ratios = []
for g in letters("Lu") + letters("Ll"):
    a_adv, a_lsb = src["hmtx"][g]
    a_rsb = a_adv - src["glyf"][g].xMax
    r = RCH[rev[g]]
    r_sb = r["xMin"] + (r["adv"] - r["xMax"])
    if a_lsb + a_rsb > 60 and r_sb > 0:
        sb_ratios.append(r_sb / (a_lsb + a_rsb))
SB = med(sb_ratios)
print("H_GEN", {k: round(v, 3) for k, v in H_GEN.items()}, "SB", round(SB, 3))


def hgen_for(g):
    ch = base_char(g)
    cat = unicodedata.category(ch) if ch else ""
    if g.endswith(".sc") or ".sc." in g:
        return H_GEN["Lu"]
    return H_GEN.get(cat, H_GEN["other"])


# ---- pass 2: other simple glyphs (not in the reference): letters take the
# class-wide stem ratio and y weight change of pass 1, so they stay even with
# the letters the reference has
KR = {k: statistics.median(v) for k, v in STEM_RATIO.items()}
YS = {k: statistics.median(v) for k, v in YSTR.items()}
for g in others:
    cls = vclass(g)
    cut, band = cuts[g]
    stem, gap, sw = stats[g]
    a_adv, a_lsb = src["hmtx"][g]
    gl = src["glyf"][g]
    WA = gl.xMax - gl.xMin
    b = basename(g)
    tol = 2 if g.endswith(".case") else 0      # (Anton's braceright.case is 1 unit off)
    if b in T and b != g and abs(src["hmtx"][b][0] - a_adv) <= tol:
        # stylistic/case variant of a known glyph: same scale/weight and advance
        place_like_base(g, b, T[b]["adv"], cls, cut, band)
        continue
    if g in DERIVED and DERIVED[g] in T:
        db = DERIVED[g]
        place_like_base(g, db, T[db]["adv"] + (a_adv - src["hmtx"][db][0]) * SB, cls, cut, band)
        squeeze_overhang(g, db, -40, T[g]["adv"] + 40)
        continue
    h = hgen_for(g)
    ch = base_char(g)
    letter = (ch is not None and unicodedata.category(ch) in ("Lu", "Ll") and "." not in g)
    y_ok_g = letter and y_eligible(g)
    k = K_STEM * (KR.get(cls, 1.0) if letter else 1.0)
    tw = h * WA + (k - h) * (stem or 0)
    h, d = fit_simple(g, tw, stem, gap, sw, cls, cut, band, kind_of(g), target_stem=k * stem if stem else None)
    ys = YS.get(cls, 0.0) if y_ok_g else 0.0
    if ys:
        base = apply_simple(g, h, d, cls, cut, band)
        st0 = hbar_stats(*base)
        if st0 is None or st0[0] > 400:          # no horizontal strokes (diagonals)
            ys = 0.0
        else:
            ys = max(ys, -0.45 * st0[2])
            if not y_ok(g, h, d, cls, cut, band, ys, fontops.topology(*base)):
                ys = 0.0
    pts, _, _ = apply_simple(g, h, d, cls, cut, band, ys)
    w = pts[:, 0].max() - pts[:, 0].min()
    lsb = max(a_lsb * SB, -40.0)
    rsb = max((a_adv - gl.xMax) * SB, -40.0)
    store_simple(g, h, d, cls, cut, band, lsb, lsb + w + rsb, ystr=ys)

# ---- marks and spacing accents: uniform scale, no emboldening
for g in marks:
    cls = vclass(g)
    gl = src["glyf"][g]
    a_adv = src["hmtx"][g][0]
    pts, ends, flags = apply_simple(g, H_MARK, 0.0, cls, None, False)
    if g in rev and rev[g] in RCH:
        adv = RCH[rev[g]]["adv"]
        w = pts[:, 0].max() - pts[:, 0].min()
        xmin = (adv - w) / 2
    elif a_adv:
        adv = a_adv * H_MARK
        xmin = gl.xMin * H_MARK
    else:
        adv = 0
        xmin = gl.xMin * H_MARK
    store_simple(g, H_MARK, 0.0, cls, None, False, xmin, adv)


# ----------------------------------------------------------- point mapping
def ymap_glyph(g, y):
    t = T[g]
    return float(ymap_array([y], t["cls"], t["cut"], t["band"])[0])


C = {}   # composite name -> layout info


def map_point(g, x, y):
    """Map a point in Anton's glyph g coordinates to Smack Dab coordinates."""
    if g in T:
        t = T[g]
        X = t["h"] * x + t["delta"] / 2 + t["tx"]
        if "xmap" in t:              # glyph re-proportioned after fitting (J)
            xs, ys = t["xmap"]
            X = float(np.interp(X, xs, ys)) if xs[0] <= X <= xs[-1] else X
        return X, ymap_glyph(g, y)
    c = C[g]
    if c["kind"] == "based":
        b, dxn, dyn = c["base"]
        X, Y = map_point(b, x, y)
        return X + dxn, Y + dyn
    if c["kind"] == "ligature":
        pcs = c["pieces"]
        for i, (cg, dx0, adv0, dxn) in enumerate(pcs):
            if x < dx0 + adv0 or i == len(pcs) - 1:
                X, Y = map_point(cg, x - dx0, y)
                return X + dxn, Y
    r, b = c["affine"]
    return r * x + b, float(ymap_array([y], c["cls"])[0])


src_gs = src.getGlyphSet()


def ink_span_at(g, y0, y1):
    """x-extent of Anton glyph g's ink between heights y0 and y1, or None."""
    c, e, f = src["glyf"][g].getCoordinates(src["glyf"])
    if not len(c):
        return None
    xs = []
    for poly in fontops.flatten(np.array(c, float), list(e), [x & 1 for x in f], steps=8):
        n = len(poly)
        for i in range(n):
            (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
            lo, hi = max(min(ya, yb), y0), min(max(ya, yb), y1)
            if lo > hi:
                continue
            if ya == yb:
                xs += [xa, xb]
                continue
            for yy in (lo, hi):
                xs.append(xa + (yy - ya) / (yb - ya) * (xb - xa))
    return (min(xs), max(xs)) if xs else None


def ink_bounds(g):
    p = BoundsPen(src_gs)
    src_gs[g].draw(p)
    return p.bounds


def write_simple(g, pts, flags, ends):
    ng = Glyph()
    ng.numberOfContours = len(ends)
    ng.coordinates = GlyphCoordinates([(int(round(x)), int(round(y))) for x, y in pts])
    ng.flags = list(flags)
    ng.endPtsOfContours = list(ends)
    ng.program = ttProgram.Program()
    ng.program.fromBytecode(b"")
    glyf[g] = ng


# ----------------------------------------- J: a hook that stops, like j's
# Impact's J is much narrower than Anton's (581 vs 860 units of ink). Squeezing
# Anton's J, whose hook curls back up, left either a thin stem or a cramped hook.
# J (and the small-cap J) are therefore rebuilt from Smack Dab's own dotless j
# (Anton's design): a full-weight stem whose foot curls left and stops. The
# hook is scaled like the rest of the letter (cap height / x-height) and
# lengthened to reach the reference's left ink edge. The x-remap is recorded so
# accents placed through J (Ĵ) follow the stem.
def _ink_runs(name, frac):
    c, e, f = glyf[name].getCoordinates(glyf)
    c = np.array(c, float)
    return fontops._scan_gaps(fontops.nonzero_polys(c, list(e), [x & 1 for x in f]), frac * c[:, 1].max())


def hooked_J(g, stem_ref, keep_right=True):
    old = np.array(glyf[g].coordinates, float)
    o_x0, o_x1 = old[:, 0].min(), old[:, 0].max()
    y_bot, height = old[:, 1].min(), old[:, 1].max()
    o_runs = _ink_runs(g, 0.6)
    o_s0, o_s1 = o_runs[-1]
    stem = _ink_runs(stem_ref, 0.35)[0]
    stem = stem[1] - stem[0]
    ext = stem * (171.0 / 410.0)             # hook length, in proportion to the stem
    if keep_right:                           # J: right edge and left ink edge as fitted
        x1 = o_x1
        ext = x1 - stem - o_x0
    x0 = o_x0
    x1 = x0 + ext + stem
    c, e, f = glyf["uni0237"].getCoordinates(glyf)
    c = np.array(c, float)
    f = [x & 1 for x in f]
    (s0, s1), = fontops._scan_gaps(fontops.nonzero_polys(c, list(e), f), 0.5 * c[:, 1].max())
    jl, jb, jt = c[:, 0].min(), c[:, 1].min(), c[:, 1].max()
    ky = height / jt                         # cap (or small-cap) height / x-height
    ycut = 150.0                             # the hook lies below this in the j
    X = np.where(c[:, 0] >= s0, x1 - stem + (c[:, 0] - s0) * stem / (s1 - s0),
                 x0 + (c[:, 0] - jl) * (x1 - stem - x0) / (s0 - jl))
    yc = y_bot + (ycut - jb) * ky
    Y = np.where(c[:, 1] <= ycut, y_bot + (c[:, 1] - jb) * ky,
                 yc + (c[:, 1] - ycut) * (height - yc) / (jt - ycut))
    write_simple(g, np.c_[X, Y], f, list(e))
    T[g]["adv"] = T[g]["adv"] - (o_x1 - x1)  # keep the right side bearing
    T[g]["xmap"] = ([o_x0, o_s0, o_s1], [x0, x1 - stem, x1])
    print(f"{g}: hooked J, stem {o_s1 - o_s0:.0f} -> {stem:.0f}, ink {x0:.0f}..{x1:.0f}, adv {T[g]['adv']:.0f}")


hooked_J("J", "H")
if "j.sc" in glyf.keys() and "h.sc" in glyf.keys():
    hooked_J("j.sc", "h.sc", keep_right=False)


# ---- hairline slits: where a stroke's emboldened edge runs up against a deep
# notch of Anton's (the shoulder of h n m, the joins of a b d), a white
# hairline wedge can remain. Rows where such a dead-end channel is narrower
# than 30 units are filled (counters, closed at both ends, are left alone).
slit_filled = []
for g in list(T):
    if g in marks or glyf[g].isComposite() or not glyf[g].numberOfContours:
        continue
    c, e, f = glyf[g].getCoordinates(glyf)
    p2, e2, f2, area = fontops.fill_narrow_channels(np.array(c, float), list(e), [x & 1 for x in f],
                                                     w_max=30.0, min_len=40.0)
    if area > 0:
        write_simple(g, p2, f2, e2)
        slit_filled.append(g)
print("hairline slits filled:", len(slit_filled), slit_filled[:30])


# ------------------------------------------- bars kept inside the reference ink
# The bars of Ħ and ł keep Anton's length, which reaches 20-30 units past the
# reference's ink on both sides (ĦĦ, łł joined into one line). Only the bar
# ends are drawn in, to the reference's ink edges; the stems don't move.
def _cp_of(g):
    return rev.get(g)


for g in ("Hbar", "lslash"):
    r = RCH.get(_cp_of(g), {})
    if g not in glyf.keys() or "xMin" not in r or glyf[g].isComposite():
        continue
    c, e, f = glyf[g].getCoordinates(glyf)
    pts, e, f = np.array(c, float), list(e), [x & 1 for x in f]
    for side, lim in (("L", r["xMin"]), ("R", r["xMax"])):
        pr = fontops.side_protrusion(pts, e, f, side, lim)
        if pr and pr[0] is not None:
            pts = fontops.squeeze_side(pts, side, pr[0], lim)
            print(f"{g}: bar end {side} {pr[1]:.0f} -> {lim} (from {pr[0]:.0f})")
    write_simple(g, pts, f, e)

# x: the upper right arm reached 20 units further out than the reference's x
# while y v w keep their reference edges, so "xy", "xv", "xw" touched at the
# x-height. The arm is leaned in by that much at its tip, nothing below the
# crossing moves.
def _outline(name):
    c, e, f = glyf[name].getCoordinates(glyf)
    return np.array(c, float), list(e), [x & 1 for x in f]


def _edge_at(name, y, side):
    return fontops.edge_at(*_outline(name), y, side)


MIN_WHITE = 8                            # white kept between t / x and a following v w y
if "x" in glyf.keys() and not glyf["x"].isComposite():
    pts, e, f = _outline("x")
    top = pts[:, 1].max()
    mid_y, mid_x = 0.5 * top, 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    span = 0.25 * np.ptp(pts[:, 0])
    x_tr = _edge_at("x", top - 8, "R")
    lean = 0.0
    for nb in ("v", "w", "y"):
        tl = _edge_at(nb, top - 8, "L")
        if x_tr is not None and tl is not None:
            lean = max(lean, MIN_WHITE - (T["x"]["adv"] - x_tr + tl))
    if lean > 0:
        pts = fontops.lean_in(pts, "R", lean, top, mid_y, mid_x, span)
        write_simple("x", pts, f, e)
        print(f"x: upper right arm leaned in by {lean:.0f}")

# t ŧ: the crossbar ends at the advance (the reference's ink edge is the foot),
# so "ty", "tv", "tw" touched where the bar meets the arm of y v w. The bar's
# right end is drawn in until MIN_WHITE units of white remain.
for g in ("t", "tbar"):
    if g not in glyf.keys() or glyf[g].isComposite():
        continue
    pts, e, f = _outline(g)
    top = pts[:, 1].max()
    rows = [(y, _edge_at(g, y, "R")) for y in np.arange(0.5 * top, top, 4.0)]
    rows = [(y, r) for y, r in rows if r is not None]
    vals, cnt = np.unique(np.round(np.array([r for _, r in rows]) / 2) * 2, return_counts=True)
    stem_r = float(vals[cnt.argmax()])
    bar = [y for y, r in rows if r > stem_r + 20]
    if not bar:
        continue
    y0, y1 = min(bar) - 6, max(bar) + 6     # rows are 4 units apart: reach the bar's corners
    need = 0.0
    for nb in ("v", "w", "y"):
        gap = fontops.pair_gap(pts, e, f, T[g]["adv"], *_outline(nb), y0, y1)
        if gap is not None:
            need = max(need, MIN_WHITE - gap)
    if need > 0:
        ext = max(r for y, r in rows if y0 <= y <= y1)
        pts = fontops.squeeze_side(pts, "R", stem_r, ext - need, y0, y1)
        write_simple(g, pts, f, e)
        print(f"{g}: crossbar end {ext:.0f} -> {ext - need:.0f}")

# ---------------------------------------------------------------- composites
comps = [g for g in order if glyf[g].isComposite()]
for g in comps:
    gl = glyf[g]
    cl = gl.components
    a_adv = src["hmtx"][g][0]
    target_adv = RCH[rev[g]]["adv"] if g in rev and rev[g] in RCH else None
    baselike = [c for c in cl if not is_markish(c.glyphName)]
    # ligature: base-like components side by side at cumulative advances
    cum, lig = 0, len(baselike) >= 2 and len(baselike) == len(cl)
    if lig:
        for c in cl:
            if c.x != cum or c.y != 0:
                lig = False
                break
            cum += src["hmtx"][c.glyphName][0]
    if lig:
        pieces, cum_new = [], 0
        for c in cl:
            pieces.append((c.glyphName, c.x, src["hmtx"][c.glyphName][0], cum_new))
            cum_new += T[c.glyphName]["adv"]
        adv = target_adv if target_adv is not None else cum_new
        off = int(round((adv - cum_new) / 2))
        pieces = [(n, dx0, a0, dxn + off) for n, dx0, a0, dxn in pieces]
        C[g] = dict(kind="ligature", pieces=pieces, adv=adv)
        for c, p in zip(cl, pieces):
            c.x, c.y = p[3], 0
        continue
    first = cl[0]
    # a variant drawn as its base letter moved up or down (hyphen.case,
    # guillemotleft.case, quoteleft.sc ...) keeps its base's advance and width
    lifted = len(cl) == 1 and first.y != 0
    if not is_markish(first.glyphName) and first.x == 0 and (first.y == 0 or lifted):
        b = first.glyphName
        badv_a = src["hmtx"][b][0]
        if target_adv is not None:
            adv = target_adv
        elif a_adv == badv_a:
            adv = T[b]["adv"]
        else:
            adv = T[b]["adv"] + (a_adv - badv_a) * hgen_for(b)
        shift = int(round((adv - T[b]["adv"]) / 2)) if a_adv == badv_a else 0
        # where the reference puts the base inside a wider composite (e.g. the
        # L of Ľ keeps its own left edge and the extra width holds the caron)
        rg, rb = RCH.get(rev.get(g)), RCH.get(rev.get(b))
        if (len(cl) > 1 and target_adv is not None and rg and rb and "xMin" in rg and "xMin" in rb
                and rg["xMin"] >= rb["xMin"] - 2 and abs(adv - T[b]["adv"]) > 4):
            shift = int(round(rg["xMin"] - rb["xMin"]))
        dyn = int(round(first.y * (VC if vclass(g) in ("uc", "mkc") else VX))) if lifted else 0
        C[g] = dict(kind="based", base=(b, shift, dyn), adv=int(round(adv)))
        first.x, first.y = shift, dyn
        attach = cl[1:]
    else:
        bx = ink_bounds(g)
        cls = vclass(g)
        if target_adv is not None and "xMin" in RCH[rev[g]]:
            rr = RCH[rev[g]]
            r = (rr["xMax"] - rr["xMin"]) / (bx[2] - bx[0])
            b0 = rr["xMin"] - r * bx[0]
            adv = target_adv
        else:
            r = hgen_for(g)
            b0 = src["hmtx"][g][1] * SB - r * bx[0]
            adv = b0 + r * bx[2] + (a_adv - bx[2]) * SB
        C[g] = dict(kind="general", affine=(r, b0), cls=cls, adv=int(round(adv)))
        attach = cl
    # position attachments by mapping their reference point through the frame
    for c in attach:
        bb = ink_bounds(c.glyphName)
        if bb is None:
            continue
        c_y_src = c.y
        cx = (bb[0] + bb[2]) / 2
        cy_mid = (bb[1] + bb[3]) / 2 + c.y
        yref = bb[3] if cy_mid > XH_A else bb[1]
        if C[g]["kind"] == "based":
            X, Y = map_point(g, cx + c.x, yref + c.y)
            # an accent standing beside the base (caron of Ľ ď ť) keeps its gap
            # to the base's emboldened edge instead of following the centre map
            b, shift, _ = C[g]["base"]
            span = ink_span_at(b, bb[1] + c.y, bb[3] + c.y)
            if span is not None and b in T and bb[0] + c.x >= span[1] - 5:
                tb = T[b]
                gap = bb[0] + c.x - span[1]
                left_new = tb["h"] * span[1] + tb["delta"] + tb["tx"] + shift + tb["h"] * gap
                ta = T.get(c.glyphName, dict(h=1.0, tx=0.0, delta=0.0))
                # stay inside the advance (or the reference's own ink edge),
                # giving up at most 40% of the gap for it
                rr = RCH.get(rev.get(g), {})
                lim = max(C[g]["adv"], rr.get("xMax", C[g]["adv"]))
                over = left_new + ta["h"] * (bb[2] - bb[0]) + ta["delta"] - lim
                if over > 0:
                    left_new -= min(over, 0.4 * tb["h"] * max(gap, 0.0))
                c.x = int(round(left_new - (ta["h"] * bb[0] + ta["tx"])))
                c.y = int(round(Y - map_point(c.glyphName, cx, yref)[1]))
                # Anton tops the caron of ľ ď Ľ level with the stem; the accent's
                # own vertical map set it 60 units lower, where its tip met the
                # arm of a following y v w at the x-height. Keep it level.
                bsrc = ink_bounds(b)
                if bsrc is not None and abs(bb[3] + c_y_src - bsrc[3]) <= 10:
                    gb_, ga_ = glyf[b], glyf[c.glyphName]
                    gb_.recalcBounds(glyf)
                    ga_.recalcBounds(glyf)
                    c.y = int(round(gb_.yMax + (bb[3] + c_y_src - bsrc[3]) - ga_.yMax))
                continue
        else:
            r, b0 = C[g]["affine"]
            X = r * (cx + c.x) + b0
            Y = float(ymap_array([yref + c.y], C[g]["cls"])[0])
        Xa, Ya = map_point(c.glyphName, cx, yref)
        c.x, c.y = int(round(X - Xa)), int(round(Y - Ya))

for g in comps:
    adv = C[g]["adv"]
    cl = glyf[g].components
    for c in cl:
        # USE_MY_METRICS is only valid when the composite really takes that
        # component's metrics unchanged
        if c.flags & 0x200 and not (c.x == 0 and T.get(c.glyphName, {}).get("adv") == adv):
            c.flags &= ~0x200

# ---------------------------------------- fractions: slash centred in its gap
# The emboldened figures are wider than Anton's while the reference fixes the
# advance; Anton's slash offsets then leave one side jammed (⅛ ⅜ ⅝ ⅞). The
# slash is moved so the white on both sides of it is equal.
def _row_extents(name, dx, dy, step=8):
    c, e, f = glyf[name].getCoordinates(glyf)
    if not len(c):
        return {}
    polys = fontops.nonzero_polys(np.array(c, float) + (dx, dy), list(e), [x & 1 for x in f])
    out = {}
    for y in range(-800, 2400, step):
        ink = fontops._scan_gaps(polys, y)
        if ink:
            out[y] = (ink[0][0], ink[-1][1])
    return out


def _hgap(L, R):
    ys = set(L) & set(R)
    return min(R[y][0] - L[y][1] for y in ys) if ys else None


frac_log = []
for g in comps:
    cl = glyf[g].components
    if (len(cl) == 3 and cl[0].glyphName.endswith(".numr") and basename(cl[1].glyphName) == "fraction"
            and cl[2].glyphName.endswith(".dnom")):
        N, F, D = (_row_extents(c.glyphName, c.x, c.y) for c in cl)
        rr = RCH.get(rev.get(g), {})
        if "xMin" in rr and N and D:      # figures no more than 40 units past the reference's ink
            dn = int(round(max(0.0, rr["xMin"] - 40 - min(v[0] for v in N.values()))))
            dd = int(round(min(0.0, rr["xMax"] + 40 - max(v[1] for v in D.values()))))
            cl[0].x += dn
            cl[2].x += dd
            N = {y: (u + dn, v + dn) for y, (u, v) in N.items()}
            D = {y: (u + dd, v + dd) for y, (u, v) in D.items()}
        a, b = _hgap(N, F), _hgap(F, D)
        if a is None or b is None:
            continue
        shift = int(round((b - a) / 2))
        cl[1].x += shift
        frac_log.append(f"{g} {a:.0f}/{b:.0f}->{a + shift:.0f}")
print("fractions balanced:", ", ".join(frac_log))

# ------------------------------------- keep accented capitals inside win ascent
TOP_LIMIT = ref["os2"]["usWinAscent"] - 16      # GDI clips above usWinAscent
for g in order:
    glyf[g].recalcBounds(glyf) if not glyf[g].isComposite() else None
tucked = []


def placed(c):
    cg = glyf[c.glyphName]
    return cg.yMin + c.y, cg.yMax + c.y


BOT_LIMIT = -ref["os2"]["usWinDescent"] + 16


# composites: accents tucked inside the clip box and kept near the advance
for g in comps:
    if C[g]["kind"] != "based":
        continue
    base = glyf[glyf[g].components[0].glyphName]
    if base.numberOfContours <= 0:
        continue
    Bt = base.yMax if base.yMax < CAP_R + 100 else CAP_R + 15   # e.g. slashed O
    rr = RCH.get(rev.get(g), {})
    res = fontops.tuck_parts(fontops.composite_parts(glyf, g), C[g]["adv"], Bt, TOP_LIMIT, BOT_LIMIT,
                             debug=g if (g in os.environ.get("TUCK_DEBUG", "").split(",") or os.environ.get("TUCK_DEBUG") == "*") else None,
                             xlim=(rr["xMin"], rr["xMax"]) if "xMin" in rr else None)
    if res is not None:
        write_simple(g, *res)
        tucked.append(g)

# simple glyphs with accents drawn in (e.g. U+01D9, U+1E66)
for g in simple:
    gl = glyf[g]
    if g in marks or gl.yMax <= TOP_LIMIT:
        continue
    tentative = CAP_R if vclass(g) == "uc" else XH_R
    parts = fontops.contour_parts(glyf, g, tentative)
    body = [p for p in parts if p["role"] == "base"]
    if not body:
        continue
    base_top = max(p["pts"][:, 1].max() for p in body)
    res = fontops.tuck_parts(parts, T[g]["adv"], base_top, TOP_LIMIT, BOT_LIMIT)
    if res is not None:
        write_simple(g, *res)
        glyf[g].recalcBounds(glyf)
        tucked.append(g)

# simple (non-mark) glyphs with descenders past the clip box, e.g. Eng
for g in simple:
    gl = glyf[g]
    if g not in marks and gl.yMin < BOT_LIMIT:
        f = max(0.7, BOT_LIMIT / gl.yMin)
        gl.coordinates = GlyphCoordinates([(x, int(round(y * f)) if y < 0 else y) for x, y in gl.coordinates])
        gl.recalcBounds(glyf)
        tucked.append((g, "descender", round(f, 2)))

# standalone marks hanging below the clip box (cedilla, ogonek, ...): compress
# toward their top (attachment point) so composites and GPOS stay aligned
for g in marks:
    gl = glyf[g]
    if (gl.yMin < BOT_LIMIT and gl.yMax < XH_R and gl.numberOfContours == 2 and not gl.isComposite()
            and fontops.topology(np.array(gl.coordinates, float), list(gl.endPtsOfContours),
                                 [x & 1 for x in gl.flags])[0] == 1):      # a real ring (one hole), not two dots
        # ring below: scaled evenly (never squashed) and hung clear of the
        # letter, like the rings of the precomposed Ḁ ḁ
        top_new = -30
        sc = (top_new - BOT_LIMIT) / (gl.yMax - gl.yMin)
        cx = (gl.xMin + gl.xMax) / 2
        gl.coordinates = GlyphCoordinates([(int(round(cx + (x - cx) * sc)), int(round(top_new + (y - gl.yMax) * sc)))
                                           for x, y in gl.coordinates])
        gl.recalcBounds(glyf)
        tucked.append((g, "ring", round(sc, 2)))
        continue
    if gl.yMin < BOT_LIMIT and gl.yMax < XH_R:
        top = gl.yMax
        f = max(0.5, (top - BOT_LIMIT) / (top - gl.yMin))
        gl.coordinates = GlyphCoordinates([(x, int(round(top + (y - top) * f) if y < top else y))
                                           for x, y in gl.coordinates])
        gl.recalcBounds(glyf)
        tucked.append((g, "mark", round(f, 2)))
print(f"tucked accents in {len(tucked)} glyphs")

# Spacing accents typed on their own (U+00A8, U+02C6 ...) keep Anton's accent
# size inside composites, but the copy the cmap points to is narrowed to the
# reference ink width so it doesn't overhang its advance.
spacing_copies = {}
for g in marks:
    if g not in rev or rev[g] not in RCH or "xMin" not in RCH[rev[g]] or glyf[g].isComposite():
        continue
    r = RCH[rev[g]]
    gl = glyf[g]
    gl.recalcBounds(glyf)
    if gl.xMin >= -40 and gl.xMax <= r["adv"] + 40:
        continue
    tw = min(r["xMax"] - r["xMin"], r["adv"] + 40 - 40)
    sx = tw / max(1, gl.xMax - gl.xMin)
    cx = (gl.xMin + gl.xMax) / 2
    new = g + ".spacing"
    co = [((x - cx) * sx + r["adv"] / 2, y) for x, y in gl.coordinates]
    write_simple(new, co, gl.flags, gl.endPtsOfContours)
    T[new] = dict(T[g])
    T[new]["adv"] = r["adv"]
    for t in font["cmap"].tables:
        if t.isUnicode() and rev[g] in t.cmap:
            t.cmap[rev[g]] = new
    gdef_cls[new] = 1          # spacing glyph, not a mark (HarfBuzz zeroes mark widths)
    spacing_copies[g] = new
print("spacing accent copies:", list(spacing_copies.values()))

# ------------------------------------------------- empty glyphs and advances
for g in order:
    if g in T or g in C:
        continue
    a_adv = src["hmtx"][g][0]
    if g in rev and rev[g] in RCH:
        adv = RCH[rev[g]]["adv"]
    elif g == order[0]:
        adv = ref["notdefAdvance"]
    else:
        sp_ratio = RCH[0x20]["adv"] / src["hmtx"][font.getBestCmap()[0x20]][0]
        adv = a_adv * sp_ratio
    T[g] = dict(h=1, delta=0, tx=0, cls="uc", cut=None, band=False, adv=int(round(adv)))

for g in order:
    gl = glyf[g]
    gl.program = ttProgram.Program()
    gl.program.fromBytecode(b"")
for g in order:
    glyf[g].recalcBounds(glyf)
    adv = C[g]["adv"] if g in C else T[g]["adv"]
    lsb = glyf[g].xMin if glyf[g].numberOfContours != 0 else 0
    hmtx[g] = (int(round(adv)), lsb)


# ------------------------------------------------------------- extra glyphs
def add_glyph(name, cp, glyph, adv):
    glyf[name] = glyph            # also appends to the shared glyph order
    glyph.recalcBounds(glyf)
    hmtx[name] = (adv, glyph.xMin if glyph.numberOfContours else 0)
    for t in font["cmap"].tables:
        if t.isUnicode():
            t.cmap[cp] = name
    gdef_cls[name] = 1


def composite(parts):
    g = Glyph()
    g.numberOfContours = -1
    g.components = []
    for name, x, y in parts:
        c = GlyphComponent()
        c.glyphName, c.x, c.y, c.flags = name, int(round(x)), int(round(y)), 0x4
        g.components.append(c)
    return g


def simple_from(src_name, fx):
    """New simple glyph from a transformed copy of an existing one."""
    s = glyf[src_name]
    coords = [fx(x, y) for x, y in s.coordinates]
    g = Glyph()
    g.numberOfContours = s.numberOfContours
    g.coordinates = GlyphCoordinates([(int(round(x)), int(round(y))) for x, y in coords])
    g.endPtsOfContours = list(s.endPtsOfContours)
    g.flags = list(s.flags)
    g.program = ttProgram.Program()
    g.program.fromBytecode(b"")
    return g


def reverse_contours(g):
    pts, fl, ends, start = list(g.coordinates), list(g.flags), g.endPtsOfContours, 0
    npts, nfl = [], []
    for e in ends:
        seg, sf = pts[start:e + 1], fl[start:e + 1]
        seg = [seg[0]] + seg[1:][::-1]
        sf = [sf[0]] + sf[1:][::-1]
        npts += seg
        nfl += sf
        start = e + 1
    g.coordinates = GlyphCoordinates(npts)
    g.flags = nfl


cmap_now = font.getBestCmap()
if 0x203C in RCH and 0x203C not in cmap_now:              # double exclamation
    ex = cmap_now[0x21]
    adv = RCH[0x203C]["adv"]
    w1 = hmtx[ex][0]
    add_glyph("exclamdbl", 0x203C, composite([(ex, (adv - 2 * w1) / 2, 0), (ex, (adv - 2 * w1) / 2 + w1, 0)]), adv)
us = cmap_now[0x5F]
ub = glyf[us]
if 0x203E in RCH and 0x203E not in cmap_now:              # overline
    adv = RCH[0x203E]["adv"]
    dy = CAP_R + 120 - ub.yMax
    g = simple_from(us, lambda x, y: (x + (adv - hmtx[us][0]) / 2, y + dy))
    add_glyph("overline", 0x203E, g, adv)
if 0x2017 in RCH and 0x2017 not in cmap_now:              # double low line
    adv = RCH[0x2017]["adv"]
    th = ub.yMax - ub.yMin
    off = (adv - hmtx[us][0]) / 2
    add_glyph("underscoredbl", 0x2017,
              composite([(us, off, 0), (us, off, int(round(th * 1.9)))]), adv)
if 0x201B in RCH and 0x201B not in cmap_now:              # reversed-9 quote
    q = cmap_now[0x2019]
    adv = RCH[0x201B]["adv"]
    qa = hmtx[q][0]
    g = simple_from(q, lambda x, y: (qa - x + (adv - qa) / 2, y))
    reverse_contours(g)
    add_glyph("quotereversed", 0x201B, g, adv)

# ------------------------------------------------------------------- GPOS
gpos = font["GPOS"].table


def fix_anchor(a, g):
    x, y = map_point(g, a.XCoordinate, a.YCoordinate)
    a.Format = 1
    a.XCoordinate, a.YCoordinate = int(round(x)), int(round(y))
    for attr in ("AnchorPoint", "XDeviceTable", "YDeviceTable"):
        if hasattr(a, attr):
            delattr(a, attr)


kern_subtables, single_subtables = [], []
for lk in gpos.LookupList.Lookup:
    for st in lk.SubTable:
        if lk.LookupType == 9:
            st = st.ExtSubTable
        t = st.LookupType if hasattr(st, "LookupType") else lk.LookupType
        if t == 4:
            for g, rec in zip(st.MarkCoverage.glyphs, st.MarkArray.MarkRecord):
                fix_anchor(rec.MarkAnchor, g)
            for g, rec in zip(st.BaseCoverage.glyphs, st.BaseArray.BaseRecord):
                for a in rec.BaseAnchor:
                    if a is not None:
                        fix_anchor(a, g)
        elif t == 5:
            for g, rec in zip(st.MarkCoverage.glyphs, st.MarkArray.MarkRecord):
                fix_anchor(rec.MarkAnchor, g)
            for g, att in zip(st.LigatureCoverage.glyphs, st.LigatureArray.LigatureAttach):
                for comp in att.ComponentRecord:
                    for a in comp.LigatureAnchor:
                        if a is not None:
                            fix_anchor(a, g)
        elif t == 6:
            for g, rec in zip(st.Mark1Coverage.glyphs, st.Mark1Array.MarkRecord):
                fix_anchor(rec.MarkAnchor, g)
            for g, rec in zip(st.Mark2Coverage.glyphs, st.Mark2Array.Mark2Record):
                for a in rec.Mark2Anchor:
                    if a is not None:
                        fix_anchor(a, g)
        elif t == 2:
            kern_subtables.append(st)
        elif t == 1:
            single_subtables.append(st)

for st in single_subtables:            # cpsp: scale spacing values
    vals = st.Value if isinstance(st.Value, list) else [st.Value]
    for v in vals:
        for attr in ("XPlacement", "XAdvance"):
            if hasattr(v, attr):
                setattr(v, attr, int(round(getattr(v, attr) * SB)))

# ---- small caps Anton leaves out: í Í (no iacute.sc, so "smcp" left í in
# lower case among small capitals) and the Turkish dotted i (i.loclTRK, İ).
# Built like Anton's other accented small caps: i.sc with the accent its
# siblings use (acute of eacute.sc, dot of edotaccent.sc), centred on the ink.
gsub = font["GSUB"].table


def _single_maps(tag):
    out = []
    for fr in gsub.FeatureList.FeatureRecord:
        if fr.FeatureTag == tag:
            for li in fr.Feature.LookupListIndex:
                for st in gsub.LookupList.Lookup[li].SubTable:
                    st = getattr(st, "ExtSubTable", st)
                    if hasattr(st, "mapping"):
                        out.append(st.mapping)
    return out


def _ink(name):
    g_ = glyf[name]
    g_.recalcBounds(glyf)
    return g_.xMin, g_.xMax


sc_added = []
for new, sib, lc, uc in (("iacute.sc", "eacute.sc", "iacute", "Iacute"),
                         ("idotaccent.sc", "edotaccent.sc", "i.loclTRK", "Idotaccent")):
    if new in glyf.keys() or "i.sc" not in glyf.keys() or sib not in glyf.keys() or lc not in glyf.keys():
        continue
    acc = [c for c in glyf[sib].components[1:]]
    if len(acc) != 1:
        continue
    a = acc[0]
    bx0, bx1 = _ink("i.sc")
    ax0, ax1 = _ink(a.glyphName)
    x = (bx0 + bx1) / 2 - (ax0 + ax1) / 2
    glyf[new] = composite([("i.sc", 0, 0), (a.glyphName, x, a.y)])
    glyf[new].recalcBounds(glyf)
    hmtx[new] = (hmtx["i.sc"][0], glyf[new].xMin)
    gdef_cls[new] = 1
    for m in _single_maps("smcp"):
        m[lc] = new
    for m in _single_maps("c2sc"):
        if uc in glyf.keys():
            m[uc] = new
    # kerning: same classes as i.sc
    for st in kern_subtables:
        if st.Format != 2:
            continue
        if "i.sc" in st.Coverage.glyphs and new not in st.Coverage.glyphs:
            st.Coverage.glyphs.append(new)
            st.ClassDef1.classDefs[new] = st.ClassDef1.classDefs.get("i.sc", 0)
        if "i.sc" in st.ClassDef2.classDefs:
            st.ClassDef2.classDefs[new] = st.ClassDef2.classDefs["i.sc"]
    sc_added.append(new)
if sc_added:
    font.setGlyphOrder(font.getGlyphOrder())
    for st in kern_subtables:          # keep coverage in glyph order
        if st.Format == 2:
            gid = {g_: i for i, g_ in enumerate(font.getGlyphOrder())}
            st.Coverage.glyphs.sort(key=lambda n: gid[n])
print("small caps added:", sc_added)

# ---- GSUB: drop features that would change advances by default ('liga':
# ligature advances must equal the sum of their parts)
gsub = font["GSUB"].table
# also frac/numr/dnom: HarfBuzz applies them automatically around U+2044
# FRACTION SLASH, which would change advances where the reference does nothing
DROP = {"liga", "frac", "numr", "dnom"}
keep = [i for i, fr in enumerate(gsub.FeatureList.FeatureRecord) if fr.FeatureTag not in DROP]
remap = {old: new for new, old in enumerate(keep)}
gsub.FeatureList.FeatureRecord = [gsub.FeatureList.FeatureRecord[i] for i in keep]
gsub.FeatureList.FeatureCount = len(keep)
for sr in gsub.ScriptList.ScriptRecord:
    langs = [sr.Script.DefaultLangSys] + [l.LangSys for l in sr.Script.LangSysRecord]
    for ls in langs:
        if ls is None:
            continue
        ls.FeatureIndex = [remap[i] for i in ls.FeatureIndex if i in remap]
        ls.FeatureCount = len(ls.FeatureIndex)

# save intermediate (needed to measure profiles of the new outlines)
font.save(OUT + ".tmp.ttf")
new_face = freetype.Face(OUT + ".tmp.ttf")
new_face.set_char_size(FT_PPEM * 64, 0, 72, 72)
new_gid = {g: i for i, g in enumerate(TTFont(OUT + ".tmp.ttf").getGlyphOrder())}
src_gid = {g: i for i, g in enumerate(src.getGlyphOrder())}


def profile(face, gid):
    a, left, top = render(face, gid)
    prof = {}
    for r in range(a.shape[0]):
        xs = np.where(a[r])[0]
        if xs.size:
            prof[top - r] = ((left + xs[0]) * U, (left + xs[-1] + 1) * U)
    return prof


_pcache = {}


def prof(which, g):
    key = (which, g)
    if key not in _pcache:
        _pcache[key] = profile(src_face_ if which == "A" else new_face,
                               (src_gid if which == "A" else new_gid)[g])
    return _pcache[key]


src_face_ = freetype.Face(ANTON)
src_face_.set_char_size(FT_PPEM * 64, 0, 72, 72)


def min_gap(which, L, R):
    pl, pr = prof(which, L), prof(which, R)
    advL = (src["hmtx"] if which == "A" else hmtx)[L][0]
    common = set(pl) & set(pr)
    if not common:
        return None
    return min((advL - pl[y][1]) + pr[y][0] for y in common)


# global spacing ratio from identical-letter pairs
ratios = []
for g in letters("Lu") + letters("Ll"):
    dA, dS = min_gap("A", g, g), min_gap("S", g, g)
    if dA and dS and dA > 40:
        ratios.append(dS / dA)
S_SP = med(ratios)
print(f"spacing ratio {S_SP:.3f}")


def retarget(L, R, v):
    if not v:
        return 0
    dA, dS = min_gap("A", L, R), min_gap("S", L, R)
    if dA is None or dS is None:
        return int(round(v * S_SP))
    k = S_SP * (dA + v) - dS
    if v < 0:
        k = max(min(k, 0), 2 * S_SP * v)
    else:
        k = min(max(k, 0), 2 * S_SP * v)
    return int(round(k))


def rep(glyphs):
    return sorted(glyphs, key=lambda n: (len(n), n))[0]


nk = 0
for st in kern_subtables:
    if st.Format == 1:
        for L, ps in zip(st.Coverage.glyphs, st.PairSet):
            for pv in ps.PairValueRecord:
                v1 = pv.Value1
                if v1 is not None and getattr(v1, "XAdvance", 0):
                    v1.XAdvance = retarget(L, pv.SecondGlyph, v1.XAdvance)
                    nk += 1
    else:
        cls1, cls2 = {}, {}
        for g in st.Coverage.glyphs:
            cls1.setdefault(st.ClassDef1.classDefs.get(g, 0), []).append(g)
        for g, c in st.ClassDef2.classDefs.items():
            cls2.setdefault(c, []).append(g)
        for c1, rec in enumerate(st.Class1Record):
            for c2, r2 in enumerate(rec.Class2Record):
                v1 = r2.Value1
                if v1 is None or not getattr(v1, "XAdvance", 0):
                    continue
                if c1 not in cls1 or c2 not in cls2:
                    v1.XAdvance = int(round(v1.XAdvance * S_SP))
                    continue
                v1.XAdvance = retarget(rep(cls1[c1]), rep(cls2[c2]), v1.XAdvance)
                nk += 1
print("kerning values retargeted:", nk)

# --------------------------------------------------------- font-wide tables
for tag in ("DSIG", "fpgm", "prep", "cvt ", "gasp", "hdmx", "LTSH", "VDMX"):
    if tag in font:
        del font[tag]

hh = font["hhea"]
hh.ascent, hh.descent, hh.lineGap = ref["hhea"]["ascent"], ref["hhea"]["descent"], ref["hhea"]["lineGap"]
os2 = font["OS/2"]
ro = ref["os2"]
os2.sTypoAscender, os2.sTypoDescender, os2.sTypoLineGap = ro["sTypoAscender"], ro["sTypoDescender"], ro["sTypoLineGap"]
os2.usWinAscent, os2.usWinDescent = ro["usWinAscent"], ro["usWinDescent"]
os2.fsSelection = (os2.fsSelection & ~(1 << 7)) if not ro["useTypoMetrics"] else (os2.fsSelection | (1 << 7))
os2.fsSelection = (os2.fsSelection | (1 << 6)) & ~0b100001   # REGULAR, not italic/bold
for k in ("yStrikeoutSize", "yStrikeoutPosition", "ySubscriptXSize", "ySubscriptYSize",
          "ySubscriptYOffset", "ySuperscriptXSize", "ySuperscriptYSize", "ySuperscriptYOffset",
          "usWidthClass", "usWeightClass"):
    setattr(os2, k, ro[k])
os2.ySubscriptXOffset = os2.ySuperscriptXOffset = 0
os2.sxHeight = int(round(XH_R))
os2.sCapHeight = int(round(CAP_R))
os2.achVendID = "NONE"
# PANOSE: Latin text, normal sans, heavy, condensed; remaining digits "any"
p = os2.panose
p.bFamilyType, p.bSerifStyle, p.bWeight, p.bProportion = 2, 11, 8, 6
p.bContrast = p.bStrokeVariation = p.bArmStyle = p.bLetterForm = p.bMidline = p.bXHeight = 0
# xAvgCharWidth: OS/2 v1 definition (weighted lowercase + space), from our widths
WEIGHTS = dict(zip("abcdefghijklmnopqrstuvwxyz ", [64, 14, 27, 35, 100, 20, 14, 42, 63, 3, 6, 35, 20,
                                                    56, 56, 17, 4, 49, 56, 71, 31, 10, 18, 3, 18, 2, 166]))
cm = font.getBestCmap()
os2.xAvgCharWidth = int(sum(hmtx[cm[ord(c)]][0] * w for c, w in WEIGHTS.items()) / 1000)
os2.recalcUnicodeRanges(font)
os2.recalcCodePageRanges(font)
os2.usDefaultChar, os2.usBreakChar = 0, 32
post = font["post"]
post.underlinePosition = ref["post"]["underlinePosition"]
post.underlineThickness = ref["post"]["underlineThickness"]
head = font["head"]
head.fontRevision = 1.103        # bump with every release: font caches key on it
head.flags &= ~(1 << 4)
now = int((datetime.now(timezone.utc) - datetime(1904, 1, 1, tzinfo=timezone.utc)).total_seconds())
head.created = head.modified = now

# ------------------------------------------------------------------ naming
name = font["name"]
name.names = []
COPY = ("Copyright 2020 The Anton Project Authors (https://github.com/googlefonts/AntonFont.git). "
        "Smack Dab modifications Copyright 2026 The Smack Dab Project Authors.")
NAMES = {
    0: COPY,
    1: "Smack Dab",
    2: "Regular",
    3: "1.103;NONE;SmackDab-Regular",
    4: "Smack Dab Regular",
    5: "Version 1.103",
    6: "SmackDab-Regular",
    9: "Vernon Adams (Anton); Smack Dab adaptation",
    10: "Smack Dab is a modified version of Anton, re-proportioned to the widths and "
        "vertical metrics of a classic condensed display face so it can stand in for it "
        "without changing line breaks.",
    13: "This Font Software is licensed under the SIL Open Font License, Version 1.1. "
        "This license is available with a FAQ at: https://openfontlicense.org",
    14: "https://openfontlicense.org",
}
for nid, s in NAMES.items():
    name.setName(s, nid, 3, 1, 0x409)

font.save(OUT)
import os
os.remove(OUT + ".tmp.ttf")
print("saved", OUT, "glyphs:", len(font.getGlyphOrder()))
json.dump({g: {k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in t.items()}
           for g, t in T.items()}, open(OUT + ".transforms.json", "w"), indent=0, default=float)
