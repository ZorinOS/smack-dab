"""Complete combining-mark support.

Anton only gives accent anchors to letters it also ships precomposed, and its
mark lookups cover different letters (H takes lowercase accents but not the
capital forms the font switches to after capitals). Combining accents typed
after ~200 letters (M, Q, Ð, Þ, Ŋ, Ɛ, Ɔ, Ə, ...) therefore landed beside the
letter, and with one anchor shared by every accent, wide accents hit slashes,
hooks and descenders that a narrow accent clears. This step:

* anchors accent forms that are in no mark lookup (capital double acute, ...)
  like their siblings;
* gives every letter (precomposed ones and small caps included) that lacks one
  an anchor in each mark-to-base lookup, derived from its outline;
* splits each accent group into narrow, wide and attached (cedilla, ogonek)
  classes, and solves every letter's anchor for each class from the actual
  outlines: the accent must clear the letter by its own designed gap (at most
  GAP units), with the smallest
  sideways slide or lift, inside the Windows clip lines where that is possible
  (otherwise clear of the letter even if a legacy GDI app clips it);
* lets Cyrillic (and any other) capitals switch to the capital accent forms.

usage: complete_marks.py in.ttf out.ttf
"""
import sys
import unicodedata

import numpy as np
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import otTables as ot

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fontops  # noqa: E402

src, dst = sys.argv[1:3]
font = TTFont(src)
glyf = font["glyf"]
hmtx = font["hmtx"]
gdef = font["GDEF"].table.GlyphClassDef.classDefs
cmap = font.getBestCmap()
rev = {g: c for c, g in cmap.items()}
gid = font.getGlyphID
lookups = font["GPOS"].table.LookupList.Lookup
CAP, XH = font["OS/2"].sCapHeight, font["OS/2"].sxHeight
CEIL = font["OS/2"].usWinAscent - 16
FLOOR = -font["OS/2"].usWinDescent + 16
GAP = 12                         # clearance between an accent and its letter (at most)
NARROW = 520                     # accents up to this width form the narrow class
ATTACHED = ("uni0327", "uni0328", "cedilla", "ogonek")   # meant to touch the letter

_poly_cache = {}


def polys(name):
    if name not in _poly_cache:
        c, e, f = glyf[name].getCoordinates(glyf)
        _poly_cache[name] = [] if not len(c) else \
            fontops.flatten(np.array(c, float), list(e), [x & 1 for x in f], steps=6)
    return _poly_cache[name]


def extent_y(ps, x0, x1, top):
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


def local_top(ps, x, ymax):
    """Ink top under an accent centred at x (accents are about 500 wide); where
    that is a low patch between strokes (Ǉ, ꞷ), the letter's own top."""
    y = extent_y(ps, x - 250, x + 250, True)
    return ymax if y is None or y < 0.6 * ymax else y


def anchor(x, y):
    a = ot.Anchor()
    a.Format = 1
    a.XCoordinate, a.YCoordinate = int(round(x)), int(round(y))
    return a


def top_anchor(name):
    ps = polys(name)
    if not ps:
        return None
    pts = np.vstack(ps)
    y0, y1 = max(pts[:, 1].min(), 0.0), pts[:, 1].max()
    upper = pts[pts[:, 1] >= (y0 + y1) / 2]
    x = (upper[:, 0].min() + upper[:, 0].max()) / 2
    y = local_top(ps, x, y1)
    for zone in (CAP, XH):                 # overshoots sit at the zone, like Anton's
        if abs(y - zone) <= 30:
            y = zone
    return anchor(x, y)


def bottom_anchor(name):
    ps = polys(name)
    if not ps:
        return None
    pts = np.vstack(ps)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    # columns of the letter that stand on the baseline (no descender below them)
    xs = np.arange(x0, x1 + 1, 10.0)
    best, run = None, None
    for x in xs:
        b = extent_y(ps, x - 5, x + 5, False)
        if b is not None and b >= -60:
            run = (run[0], x) if run else (x, x)
            if best is None or run[1] - run[0] > best[1] - best[0]:
                best = run
        else:
            run = None
    if best is None or best[1] - best[0] < 150:
        x = (x0 + x1) / 2
        return anchor(x, extent_y(ps, x - 150, x + 150, False) or 0)
    return anchor((best[0] + best[1]) / 2, 0)


# 0. soft-dotted letters lose their dot under an accent: Anton swaps i j for
# dotless forms before a top mark (ccmp), but not Cyrillic і ј (Ukrainian and
# Serbian stress marks: і́ showed dot and acute) nor the precomposed į ị that
# normalization produces from i + ogonek / dot below. Dotless forms of į ị are
# made by dropping the dot; і ј use Anton's dotless i j like i j do.
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphCoordinates  # noqa: E402
from fontTools.ttLib.tables import ttProgram  # noqa: E402
import copy  # noqa: E402


def dotless_copy(name):
    g = glyf[name]
    new = name + ".dotless"
    if g.isComposite():
        keep = [copy.deepcopy(c) for c in g.components
                if c.glyphName not in ("uni0307", "dotaccent", "dotaccentcomb")]
        if len(keep) == len(g.components):
            return None
        ng = Glyph()
        ng.numberOfContours = -1
        ng.components = keep
    else:
        c, e, f = g.getCoordinates(glyf)
        c = np.array(c, float)
        pts, fl, ends, st = [], [], [], 0
        for en in e:
            seg = c[st:en + 1]
            if not seg[:, 1].min() >= XH - 60:          # not the dot above the x-height
                pts += [tuple(x) for x in seg]
                fl += list(f[st:en + 1])
                ends.append(len(pts) - 1)
            st = en + 1
        if len(ends) == len(e):
            return None
        ng = Glyph()
        ng.numberOfContours = len(ends)
        ng.coordinates = GlyphCoordinates([(int(round(x)), int(round(y))) for x, y in pts])
        ng.flags = [x & 1 for x in fl]
        ng.endPtsOfContours = ends
        ng.program = ttProgram.Program()
        ng.program.fromBytecode(b"")
    glyf[new] = ng
    ng.recalcBounds(glyf)
    hmtx[new] = (hmtx[name][0], ng.xMin)
    gdef[new] = 1
    return new


soft_dotted = {}
for cp, like in ((0x0456, "i"), (0x0458, "j")):
    if cp in cmap:
        soft_dotted[cmap[cp]] = like                   # resolved to like's dotless form below
for cp in (0x012F, 0x1ECB):
    if cp in cmap:
        d = dotless_copy(cmap[cp])
        if d:
            soft_dotted[cmap[cp]] = d
font.getReverseGlyphMap(rebuild=True)


def is_letter(g):
    if gdef.get(g, 1) == 3 or not glyf[g].numberOfContours:
        return False
    if g.endswith(".dotless") and g[:-8] in rev:
        return True
    if g in rev:
        return unicodedata.category(chr(rev[g])) in ("Lu", "Ll", "Lt", "Lo")
    if g.endswith(".sc"):                  # small caps of letters
        b = g[:-3]
        return b in rev and unicodedata.category(chr(rev[b])) in ("Lu", "Ll", "Lt", "Lo")
    return False


letters = [g for g in font.getGlyphOrder() if is_letter(g)]


def mark_subtables():
    for i, lk in enumerate(lookups):
        if lk.LookupType != 4:
            continue
        for st in lk.SubTable:
            yield i, getattr(st, "ExtSubTable", st)


def is_below(m):
    c = glyf[m].getCoordinates(glyf)[0]
    return bool(len(c)) and np.array(c)[:, 1].max() < 300


# 1. accent forms that no mark lookup positions --------------------------------
in_lookup = {}
for i, st in mark_subtables():
    for m, rec in zip(st.MarkCoverage.glyphs, st.MarkArray.MarkRecord):
        in_lookup.setdefault(m, []).append((i, st, rec))
new_marks = []
for m in sorted(g for g, c in gdef.items() if c == 3):
    if m in in_lookup or not glyf[m].numberOfContours or is_below(m):
        continue
    ref = "acutecomb.case" if m.endswith(".case") else "acutecomb"
    if ref not in in_lookup:
        continue
    li, st, rrec = in_lookup[ref][-1]
    rc, mc = np.array(glyf[ref].getCoordinates(glyf)[0]), np.array(glyf[m].getCoordinates(glyf)[0])
    rec = ot.MarkRecord()
    rec.Class = rrec.Class
    rec.MarkAnchor = anchor(rrec.MarkAnchor.XCoordinate + (mc[:, 0].mean() - rc[:, 0].mean()),
                            rrec.MarkAnchor.YCoordinate + (mc[:, 1].min() - rc[:, 1].min()))
    pairs = dict(zip(st.MarkCoverage.glyphs, st.MarkArray.MarkRecord))
    pairs[m] = rec
    order = sorted(pairs, key=gid)
    st.MarkCoverage.glyphs = order
    st.MarkArray.MarkRecord = [pairs[g] for g in order]
    st.MarkArray.MarkCount = len(order)
    new_marks.append(m)

# 2. anchors for letters that have none; marks below out of top lookups ---------
added, dropped_below, groups = {}, [], []
for i, st in mark_subtables():
    if st.ClassCount != 1:
        continue                           # the special lookup (cedilla, horn, overlays)
    below = all(is_below(m) for m in st.MarkCoverage.glyphs if glyf[m].numberOfContours)
    if not below:
        # a mark below listed in a top lookup (Anton's dot below, anchored at
        # -450) jumps above any letter that has no bottom anchor: remove it
        keep = [(m, r) for m, r in zip(st.MarkCoverage.glyphs, st.MarkArray.MarkRecord)
                if not glyf[m].numberOfContours or not is_below(m)]
        dropped_below += [m for m in st.MarkCoverage.glyphs if m not in dict(keep)]
        st.MarkCoverage.glyphs = [m for m, _ in keep]
        st.MarkArray.MarkRecord = [r for _, r in keep]
        st.MarkArray.MarkCount = len(keep)
    recs = dict(zip(st.BaseCoverage.glyphs, st.BaseArray.BaseRecord))
    n = 0
    for g in letters:
        if g in recs:
            continue
        a = bottom_anchor(g) if below else top_anchor(g)
        if a is None:
            continue
        r = ot.BaseRecord()
        r.BaseAnchor = [a]
        recs[g] = r
        n += 1
    added[i] = ("below" if below else "above", n)
    groups.append((i, st, below, recs))

# 2b. ligatures (ﬂ, and the discretionary c_h c_t s_t T_h): the top-accent
# lookup carries Anton's dot below too, which then lands on top; bottom accents
# get their own per-component anchors at the baseline instead
lig_top = lig_bot = None
for lk in lookups:
    if lk.LookupType != 5:
        continue
    for st in lk.SubTable:
        st = getattr(st, "ExtSubTable", st)
        ms = [m for m in st.MarkCoverage.glyphs if glyf[m].numberOfContours]
        if sum(is_below(m) for m in ms) > len(ms) / 2:      # (Anton lists the macron here too)
            lig_bot = st
        else:
            lig_top = st
if lig_top is not None:
    keep = [(m, r) for m, r in zip(lig_top.MarkCoverage.glyphs, lig_top.MarkArray.MarkRecord)
            if not glyf[m].numberOfContours or not is_below(m)]
    dropped_below += [m for m in lig_top.MarkCoverage.glyphs if m not in dict(keep)]
    lig_top.MarkCoverage.glyphs = [m for m, _ in keep]
    lig_top.MarkArray.MarkRecord = [r for _, r in keep]
    lig_top.MarkArray.MarkCount = len(keep)
if lig_top is not None and lig_bot is not None:
    attach = dict(zip(lig_bot.LigatureCoverage.glyphs, lig_bot.LigatureArray.LigatureAttach))
    cls_marks = {}
    for m, r in zip(lig_bot.MarkCoverage.glyphs, lig_bot.MarkArray.MarkRecord):
        if glyf[m].numberOfContours:
            cls_marks.setdefault(r.Class, []).append(is_below(m))
    below_cls = [bool(cls_marks.get(c)) and all(cls_marks[c]) for c in range(lig_bot.ClassCount)]
    for lig, la in zip(lig_top.LigatureCoverage.glyphs, lig_top.LigatureArray.LigatureAttach):
        if lig in attach:
            continue
        new = ot.LigatureAttach()
        new.ComponentRecord = []
        for cr in la.ComponentRecord:
            x = next((a.XCoordinate for a in cr.LigatureAnchor if a is not None), None)
            rec = ot.ComponentRecord()
            rec.LigatureAnchor = [anchor(x, 0) if x is not None and below_cls[c] else None
                                  for c in range(lig_bot.ClassCount)]
            new.ComponentRecord.append(rec)
        new.ComponentCount = len(new.ComponentRecord)
        attach[lig] = new
    order = sorted(attach, key=gid)
    lig_bot.LigatureCoverage.glyphs = order
    lig_bot.LigatureArray.LigatureAttach = [attach[g] for g in order]
    lig_bot.LigatureArray.LigatureCount = len(order)

# 3. narrow / wide / attached classes, anchors solved from the outlines ---------
G0, GSTEP = -1500, 10
GRID = np.arange(G0, 4000, GSTEP, dtype=float)
D = np.arange(-1000, 1001, GSTEP, dtype=float)


def columns(ps, X, dx=0.0, dy=0.0):
    """Highest and lowest ink of polygons ps (shifted by dx, dy) on columns X."""
    top = np.full(len(X), np.nan)
    bot = np.full(len(X), np.nan)
    for P in ps:
        a = P + (dx, dy)
        b = np.roll(a, -1, axis=0)
        xa, ya, xb, yb = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
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


_base_cache = {}


def base_cols(g):
    """Letter profile, widened by one column each side so that an edge falling
    between two sample columns is not missed."""
    if g not in _base_cache:
        top, bot = columns(polys(g), GRID)
        pad_t = np.pad(top, 1, constant_values=np.nan)
        pad_b = np.pad(bot, 1, constant_values=np.nan)
        top = np.fmax(np.fmax(pad_t[:-2], pad_t[1:-1]), pad_t[2:])
        bot = np.fmin(np.fmin(pad_b[:-2], pad_b[1:-1]), pad_b[2:])
        _base_cache[g] = (top, bot)
    return _base_cache[g]


def envelope(marks):
    """Lowest and highest ink, relative to the mark anchor, over a set of marks."""
    lo = np.full(len(D), np.nan)
    hi = np.full(len(D), np.nan)
    for m, rec in marks:
        t, b = columns(polys(m), D, -rec.MarkAnchor.XCoordinate, -rec.MarkAnchor.YCoordinate)
        hi, lo = np.fmax(hi, t), np.fmin(lo, b)
    return lo, hi


def solve(g, pref, env, below):
    """Anchor for letter g and an accent class with envelope env: the least
    slide + lift from pref that keeps every accent of the class GAP units
    clear of the letter, preferring placements inside the clip lines."""
    lo, hi = env
    has = ~np.isnan(lo)
    if not has.any():
        return pref, "keep"
    top, bot = base_cols(g)
    px, py = pref.XCoordinate, pref.YCoordinate
    adv = hmtx[g][0]
    ink = np.vstack(polys(g))
    width = np.ptp(ink[:, 0])
    W = int(min(300, 0.3 * width) // GSTEP) * GSTEP
    ex0, ex1 = D[has].min(), D[has].max()
    over0 = max(0, -(px + ex0)) + max(0, px + ex1 - adv)
    # a clip line already crossed at the original anchor (tall stacked accents)
    # is the limit, so only placements that cross it further count as outside
    ceil_eff = max(CEIL, py + np.nanmax(hi))
    floor_eff = min(FLOOR, py + np.nanmin(lo))
    best_in, best_out = None, None
    for s in range(-W, W + 1, GSTEP):
        ax = px + s
        idx = np.round((ax + D - G0) / GSTEP).astype(int)
        ok = has & (idx >= 0) & (idx < len(GRID))
        if below:
            col = bot[idx[ok]]
            gap = np.minimum(GAP, np.maximum(-hi[ok], 0))      # never more than the accent's own gap
            need = np.nanmin(col - gap - hi[ok]) if np.isfinite(col).any() else np.inf
            ay = min(py, need)
            lift = py - ay
            edge = ay + np.nanmin(lo)
            inside = edge >= floor_eff
            spill = floor_eff - edge
        else:
            col = top[idx[ok]]
            gap = np.minimum(GAP, np.maximum(lo[ok], 0))
            need = np.nanmax(col + gap - lo[ok]) if np.isfinite(col).any() else -np.inf
            ay = max(py, need)
            lift = ay - py
            edge = ay + np.nanmax(hi)
            inside = edge <= ceil_eff
            spill = edge - ceil_eff
        over = max(0, -(ax + ex0)) + max(0, ax + ex1 - adv)
        if s and over > over0 + 20:
            continue                       # don't slide accents off the letter
        cost = abs(s) + lift
        cand = (cost, ax, ay)
        if inside:
            if best_in is None or cand < best_in:
                best_in = cand
        elif best_out is None or (spill + cost) < best_out[0]:
            best_out = (spill + cost, ax, ay)
    pick, how = (best_in, "in") if best_in is not None else (best_out, "out")
    if pick is None or (pick[1] == px and pick[2] == py):
        return pref, "keep"
    return anchor(pick[1], pick[2]), how


stats = {"keep": 0, "in": 0, "out": 0}
moved_out = []
for i, st, below, recs in groups:
    marks = list(zip(st.MarkCoverage.glyphs, st.MarkArray.MarkRecord))
    kinds = {}
    for m, rec in marks:
        if any(m.startswith(p) for p in ATTACHED):
            kinds[m] = 2
        elif not glyf[m].numberOfContours:
            kinds[m] = 0
        else:
            c = np.array(glyf[m].getCoordinates(glyf)[0])
            kinds[m] = 0 if np.ptp(c[:, 0]) <= NARROW else 1
    envs = {k: envelope([(m, r) for m, r in marks if kinds[m] == k]) for k in (0, 1)}
    for m, rec in marks:
        rec.Class = kinds[m]
    st.ClassCount = 3
    for g, r in recs.items():
        pref = r.BaseAnchor[0]
        anchors = [pref, pref, anchor(pref.XCoordinate, pref.YCoordinate)]
        if is_letter(g):
            for k in (0, 1):
                a, how = solve(g, pref, envs[k], below)
                anchors[k] = a if a is not pref else anchor(pref.XCoordinate, pref.YCoordinate)
                stats[how] += 1
                if how == "out":
                    moved_out.append(g)
        else:
            anchors[1] = anchor(pref.XCoordinate, pref.YCoordinate)
        r.BaseAnchor = anchors
    order = sorted(recs, key=gid)
    st.BaseCoverage.glyphs = order
    st.BaseArray.BaseRecord = [recs[g] for g in order]
    st.BaseArray.BaseCount = len(order)

# 4. capitals switch accents to their capital forms (ccmp context lists only Latin)
caps = [g for g in letters if g in rev and unicodedata.category(chr(rev[g])) in ("Lu", "Lt")]
n_caps = 0
for lk in font["GSUB"].table.LookupList.Lookup:
    if lk.LookupType != 6:
        continue
    for st in lk.SubTable:
        st = getattr(st, "ExtSubTable", st)
        if getattr(st, "Format", 0) != 3 or len(st.BacktrackCoverage) != 1:
            continue
        cov = st.BacktrackCoverage[0]
        if not ("A" in cov.glyphs and "H" in cov.glyphs and "a" not in cov.glyphs):
            continue                       # not the "after a capital" context
        missing = [g for g in caps if g not in cov.glyphs]
        cov.glyphs = sorted(set(cov.glyphs) | set(missing), key=gid)
        n_caps += len(missing)

# 5. soft-dotted letters join Anton's i/j-before-accent context (see 0.)
gsub_lk = font["GSUB"].table.LookupList.Lookup
n_soft = 0
for lk in gsub_lk:
    if lk.LookupType != 6:
        continue
    for st in lk.SubTable:
        st = getattr(st, "ExtSubTable", st)
        if getattr(st, "Format", 0) != 3 or st.BacktrackCoverage or len(st.InputCoverage) != 1:
            continue
        cov = st.InputCoverage[0]
        if not ("i" in cov.glyphs and "j" in cov.glyphs):
            continue
        for rec in st.SubstLookupRecord:
            for sub in gsub_lk[rec.LookupListIndex].SubTable:
                sub = getattr(sub, "ExtSubTable", sub)
                m = getattr(sub, "mapping", None)
                if m is None or "i" not in m:
                    continue
                for g, tgt in soft_dotted.items():
                    m[g] = m.get(tgt, tgt) if tgt in ("i", "j") else tgt
        cov.glyphs = sorted(set(cov.glyphs) | set(soft_dotted), key=gid)
        n_soft += 1

# 6. stacking accents typed one after another (mark-to-mark) -----------------
# Anton's mkmk lookups mix sides (a ring above after a dieresis below was hung
# from the dieresis, under the letter) and leave out many accents (a double
# acute had nothing to stack on, so the next accent landed on top of it). They
# are rebuilt: one lookup for accents above, one for accents below; every
# accent of that side can carry and be carried, and each carrier's anchor puts
# the next accent STACK_GAP units clear of it.
STACK_GAP = 48
base_anchor_of = {}                      # mark -> its attachment anchor (from mark-to-base)
for i, st in mark_subtables():
    for m, rec in zip(st.MarkCoverage.glyphs, st.MarkArray.MarkRecord):
        base_anchor_of.setdefault(m, (rec.MarkAnchor.XCoordinate, rec.MarkAnchor.YCoordinate))


def ink_y(m):
    c = glyf[m].getCoordinates(glyf)[0]
    c = np.array(c, float)
    return (c[:, 1].min(), c[:, 1].max()) if len(c) else None


def ccc(m):
    """Canonical combining class of the accent a mark glyph draws (230 above,
    220 below, 202 attached, 1 overlay)."""
    b = m.split(".")[0]
    cp = rev.get(b)
    if cp is None and b.startswith("uni") and len(b) >= 7:
        try:
            cp = int(b[3:7], 16)
        except ValueError:
            cp = None
    if cp is None:
        cp = {"acutecomb": 0x301, "gravecomb": 0x300, "tildecomb": 0x303, "hookabovecomb": 0x309,
              "dotbelowcomb": 0x323}.get(b)
    return unicodedata.combining(chr(cp)) if cp else 0


stackable = [m for m in base_anchor_of if ink_y(m) is not None and ccc(m) not in (0, 1)]
above_m = sorted([m for m in stackable if not is_below(m)], key=gid)
below_m = sorted([m for m in stackable if is_below(m) and not m.startswith(ATTACHED)], key=gid)
below_carriers = sorted(set(below_m) | {m for m in stackable if is_below(m)}, key=gid)


def mkmk_subtable(marks1, marks2, up):
    st = ot.MarkMarkPos()
    st.Format = 1
    st.ClassCount = 1
    st.Mark1Coverage = ot.Coverage()
    st.Mark1Coverage.glyphs = list(marks1)
    st.Mark1Array = ot.Mark1Array()
    st.Mark1Array.MarkRecord = []
    for m in marks1:              # an accent hangs from its ink edge (bottom above, top below)
        r = ot.MarkRecord()
        r.Class = 0
        r.MarkAnchor = anchor(base_anchor_of[m][0], ink_y(m)[0] if up else ink_y(m)[1])
        st.Mark1Array.MarkRecord.append(r)
    st.Mark1Array.MarkCount = len(marks1)
    reach = 0.0
    st.Mark2Coverage = ot.Coverage()
    st.Mark2Coverage.glyphs = list(marks2)
    st.Mark2Array = ot.Mark2Array()
    st.Mark2Array.Mark2Record = []
    for m in marks2:
        lo, hi = ink_y(m)
        x = base_anchor_of[m][0]
        y = hi + STACK_GAP + reach if up else lo - STACK_GAP - reach
        r = ot.Mark2Record()
        r.Mark2Anchor = [anchor(x, y)]
        st.Mark2Array.Mark2Record.append(r)
    st.Mark2Array.MarkCount = len(marks2)
    return st


gpos_t = font["GPOS"].table
mkmk_idx = sorted({i for fr in gpos_t.FeatureList.FeatureRecord if fr.FeatureTag == "mkmk"
                   for i in fr.Feature.LookupListIndex})
if len(mkmk_idx) >= 2:
    up_lk, down_lk = gpos_t.LookupList.Lookup[mkmk_idx[0]], gpos_t.LookupList.Lookup[mkmk_idx[1]]
    for lk, sub in ((up_lk, mkmk_subtable(above_m, above_m, True)),
                    (down_lk, mkmk_subtable(below_m, below_carriers, False))):
        lk.LookupType = 6
        lk.LookupFlag = 0
        lk.SubTable = [sub]
        lk.SubTableCount = 1
    keep = set(mkmk_idx[:2])
    for fr in gpos_t.FeatureList.FeatureRecord:
        if fr.FeatureTag == "mkmk":
            fr.Feature.LookupListIndex = [i for i in fr.Feature.LookupListIndex if i in keep]
            fr.Feature.LookupCount = len(fr.Feature.LookupListIndex)
print(f"mark-to-mark rebuilt: {len(above_m)} accents above, {len(below_m)} below")

font.save(dst)
print("soft-dotted letters given dotless forms before accents:", soft_dotted, "contexts:", n_soft)
print("marks anchored:", new_marks, "| below-marks removed from top lookups:", sorted(set(dropped_below)))
print("anchors added:", added, "| capitals added to the accent-case context:", n_caps)
print("anchors solved per class: kept", stats["keep"], "| moved inside clip lines", stats["in"],
      "| moved, past a clip line (no room inside):", stats["out"], sorted(set(moved_out))[:40])
