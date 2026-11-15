"""Shared outline helpers for the Smack Dab build."""
import math

import freetype
import numpy as np
from scipy import ndimage

FT_FLAGS = freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_BITMAP


def runs(row):
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def render(face, gid):
    face.load_glyph(gid, FT_FLAGS)
    bm = face.glyph.bitmap
    if not bm.rows:
        return np.zeros((0, 0), bool), 0, 0
    a = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.width)
    return a > 127, face.glyph.bitmap_left, face.glyph.bitmap_top


def open_face(path, upm, ppem=1024):
    face = freetype.Face(path)
    face.set_char_size(ppem * 64, 0, 72, 72)
    return face, upm / ppem


def stroke_stats(face, gid, upx):
    """(vertical-stem thickness, smallest horizontal counter, perpendicular
    stroke width) in font units."""
    a, _, _ = render(face, gid)
    if a.size == 0:
        return None, None, None
    rows = a.shape[0]
    ink, gaps = [], []
    edge_gaps = []
    for frac in np.linspace(0.04, 0.96, 24):
        rr = runs(a[min(rows - 1, int(rows * frac))])
        g = [(rr[i + 1][0] - rr[i][1]) for i in range(len(rr) - 1)]
        if 0.18 <= frac <= 0.82:
            ink += [(e - s) for s, e in rr]
            gaps += g
        else:
            edge_gaps += g          # counters near the top/bottom (hooks, tails)
    W = a.shape[1]
    stems = [w for w in ink if w < 0.7 * W] or ink or [W]
    stem = float(np.median(stems)) * upx
    gap = float(np.percentile(gaps, 25)) * upx if gaps else math.inf
    if edge_gaps:
        gap = min(gap, float(np.percentile(edge_gaps, 25)) * upx)
    dt = ndimage.distance_transform_edt(np.pad(a, 1))
    ridge = (dt > 1) & (dt >= ndimage.maximum_filter(dt, size=3))
    sw = 2 * float(np.median(dt[ridge])) * upx if ridge.any() else stem
    return stem, gap, sw


def embolden_xy(pts, ends, xstr, ystr):
    """Port of FreeType's FT_Outline_EmboldenXY. Strengths may be scalars or
    per-point arrays. Right-facing edges move right by xstr and up-facing
    edges move up by ystr; left/bottom-facing edges stay put."""
    n_all = len(pts)
    xs_all = np.broadcast_to(np.asarray(xstr, float), (n_all,)) / 2.0
    ys_all = np.broadcast_to(np.asarray(ystr, float), (n_all,)) / 2.0
    if not np.any(xs_all) and not np.any(ys_all):
        return pts
    area, first = 0.0, 0
    for last in ends:
        p = pts[first:last + 1]
        area += np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        first = last + 1
    truetype = area < 0
    out = pts.copy()
    first = 0
    for last in ends:
        P = pts[first:last + 1]
        n = len(P)
        for i in range(n):
            xs, ys = xs_all[first + i], ys_all[first + i]
            j = (i - 1) % n
            while j != i and np.allclose(P[j], P[i]):
                j = (j - 1) % n
            k = (i + 1) % n
            while k != i and np.allclose(P[k], P[i]):
                k = (k + 1) % n
            if j == i or k == i:
                continue
            vin, vout = P[i] - P[j], P[k] - P[i]
            l_in, l_out = np.hypot(*vin), np.hypot(*vout)
            vin, vout = vin / l_in, vout / l_out
            d = vin @ vout
            sx = sy = 0.0
            if d > -0.9375:
                d += 1.0
                sx, sy = vin[1] + vout[1], vin[0] + vout[0]
                if truetype:
                    sx = -sx
                else:
                    sy = -sy
                q = vout[0] * vin[1] - vout[1] * vin[0]
                if truetype:
                    q = -q
                l = min(l_in, l_out)
                sx = sx * xs / d if xs * q <= l * d else sx * l / q
                sy = sy * ys / d if ys * q <= l * d else sy * l / q
            out[first + i, 0] = P[i, 0] + xs + sx
            out[first + i, 1] = P[i, 1] + ys + sy
        first = last + 1
    return out


def fit_width(WA, target_w, stem, gap, sw, target_stem, kind, measure):
    """Choose horizontal scale h and x-emboldening delta so that the ink width
    is target_w and vertical stems approach target_stem, within limits that
    keep counters open and stroke weight even. `measure(h, delta)` returns
    the actual resulting ink width (used for final correction)."""
    stem = stem or WA
    sw = sw or stem
    gap = gap if gap is not None else math.inf
    ratio = target_w / WA if WA else 1.0
    if kind == "alnum":
        hmin, hmax, dlim = 0.55, 1.45, 0.45
    else:
        hmin, hmax, dlim = max(0.45, min(0.8, ratio)), 1.3, 0.3
    single = WA < 1.25 * stem
    h = (target_stem / stem) if single else (target_w - target_stem) / (WA - stem)
    delta = 0.0
    for _ in range(6):
        h = min(max(h, hmin), hmax)
        dmax = min(0.45 * h * gap, dlim * h * sw)
        dmin = -min(0.45, dlim) * h * sw
        if not single:
            dmax = min(dmax, (target_stem - h * stem) + 0.18 * target_stem)
            dmin = max(dmin, (target_stem - h * stem) - 0.18 * target_stem)
            dmin = min(dmin, dmax)
        delta = min(max(target_w - h * WA, dmin), dmax)
        h_new = min(max((target_w - delta) / WA, hmin), hmax) if WA else h
        if abs(h_new - h) < 1e-4:
            break
        h = h_new
    want = h * WA + delta
    for _ in range(3):
        w = measure(h, delta)
        if abs(w - want) < 1.0:
            break
        delta += want - w
    return h, delta


# ------------------------------------------------------- outline rasterizing
def flatten(pts, ends, flags, steps=6):
    """TrueType quadratic contours -> list of polygons (N x 2 arrays)."""
    polys, start = [], 0
    pts = np.asarray(pts, float)
    for end in ends:
        P, F = pts[start:end + 1], [f & 1 for f in flags[start:end + 1]]
        start = end + 1
        n = len(P)
        if n < 2:
            continue
        seq = []
        for i in range(n):
            seq.append((P[i], F[i]))
            if not F[i] and not F[(i + 1) % n]:
                seq.append(((P[i] + P[(i + 1) % n]) / 2, 1))
        k = next((i for i, s in enumerate(seq) if s[1]), None)
        if k is None:
            continue
        seq = seq[k:] + seq[:k]
        m, i, poly = len(seq), 0, [seq[0][0]]
        while i < m:
            p0, nxt = seq[i][0], seq[(i + 1) % m]
            if nxt[1]:
                poly.append(nxt[0])
                i += 1
            else:
                c, p2 = nxt[0], seq[(i + 2) % m][0]
                for t in np.linspace(0, 1, steps + 1)[1:]:
                    poly.append((1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t * t * p2)
                i += 2
        polys.append(np.array(poly))
    return polys


def nonzero_polys(pts, ends, flags):
    """Resolve an outline under the nonzero rule (what TrueType renderers do)
    into clean, non-overlapping polygons."""
    import pathops
    path = pathops.Path(fillType=pathops.FillType.WINDING)
    pen = path.getPen()
    for poly in flatten(pts, ends, flags):
        pen.moveTo(tuple(poly[0]))
        for q in poly[1:]:
            pen.lineTo(tuple(q))
        pen.closePath()
    path.simplify(fix_winding=True, clockwise=True)
    out = []
    for contour in path.contours:
        poly = []
        for verb, p in contour:
            poly += [tuple(x) for x in p]
        if len(poly) >= 3:
            out.append(np.array(poly, float))
    return out


def poly_area(p):
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def topology(pts, ends, flags):
    """(number of counters, total counter area, number of outer pieces) under
    nonzero fill."""
    polys = nonzero_polys(pts, ends, flags)
    if not polys:
        return 0, 0.0, 0
    areas = [poly_area(p) for p in polys]
    sign = np.sign(max(areas, key=abs))
    holes = [abs(a) for a in areas if np.sign(a) != sign]
    return len(holes), float(sum(holes)), len(areas) - len(holes)


def hole_list(pts, ends, flags):
    polys = nonzero_polys(pts, ends, flags)
    if not polys:
        return []
    areas = [poly_area(p) for p in polys]
    sign = np.sign(max(areas, key=abs))
    return sorted((abs(a) for a in areas if np.sign(a) != sign), reverse=True)


def raster_outline(pts, ends, flags, upx=4.0, pad=3):
    """Raster (nonzero rule) of an outline at `upx` font units per pixel."""
    from PIL import Image, ImageDraw
    polys = nonzero_polys(pts, ends, flags)
    if not polys:
        return np.zeros((1, 1), bool)
    allp = np.vstack(polys)
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    W, H = int((x1 - x0) / upx) + 2 * pad + 2, int((y1 - y0) / upx) + 2 * pad + 2
    acc = np.zeros((H, W), bool)
    for poly in polys:
        img = Image.new("1", (W, H), 0)
        ImageDraw.Draw(img).polygon([((x - x0) / upx + pad, (y1 - y) / upx + pad) for x, y in poly], fill=1)
        acc ^= np.array(img, bool)
    return acc


def _runlen(a, axis):
    """Length of the ink run through each pixel along an axis."""
    b = a if axis == 1 else a.T
    out = np.zeros(b.shape, np.int32)
    for r in range(b.shape[0]):
        for s, e in runs(b[r]):
            out[r, s:e] = e - s
    return out if axis == 1 else out.T


def weight_metrics(a, upx=4.0):
    """sw_v: perpendicular thickness of non-horizontal strokes (stems and
    diagonals); f_diag: share of those that are diagonal; topology."""
    a = np.pad(a, 1)
    dt = ndimage.distance_transform_edt(a)
    ridge = (dt > 1.5) & (dt >= ndimage.maximum_filter(dt, size=3))
    hr, vr = _runlen(a, 1), _runlen(a, 0)
    vert = ridge & (hr <= 1.15 * vr)            # stems and diagonals up to ~45 degrees
    sw_v = float(2 * np.median(dt[vert]) * upx) if vert.sum() >= 8 else None
    diag = vert & (hr > 2 * dt * 1.25)          # horizontal cut much wider than stroke
    f_diag = float(diag.sum() / max(1, vert.sum()))
    return dict(sw_v=sw_v, f_diag=f_diag)


def outline_metrics(pts, ends, flags, upx=4.0):
    a = raster_outline(pts, ends, flags, upx)
    m = weight_metrics(a, upx)
    m["holes"], m["hole_area"], m["comps"] = topology(pts, ends, flags)
    m["hole_list"] = hole_list(pts, ends, flags)
    # interior white per pixel row (counters, notches, apertures): number of
    # gaps between strokes and their total width
    interior = np.zeros_like(a)
    for r in range(a.shape[0]):
        rr = runs(a[r])
        if len(rr) > 1:
            interior[r, rr[0][1]:rr[-1][0]] = ~a[r, rr[0][1]:rr[-1][0]]
    lab, n = ndimage.label(interior)
    areas = np.sort(np.bincount(lab.ravel())[1:])[::-1] * upx * upx if n else np.zeros(0)
    m["interior"] = float(areas.sum())
    m["regions"] = areas
    m["ink_area"] = float(a.sum()) * upx * upx
    return m


def shapes_ok(m, base, h, area_keep=0.45):
    """No counter/notch/aperture closes or shrinks below area_keep (scaled)."""
    if m["holes"] != base["holes"] or m["comps"] > base["comps"]:
        return False
    # every counter keeps >= 40% of its scaled area
    for hb, hc in zip(base["hole_list"], m["hole_list"]):
        if hc < 0.4 * hb * h:
            return False
    if m["interior"] < 0.4 * base["interior"] * h:
        return False
    # every white region (counter, notch, mouth, gap between strokes) must
    # survive with >= 35% of its scaled area: wedge tips may fill, but no
    # opening may close
    sig = [x for x in base["regions"] if x >= max(0.004 * base["ink_area"], 2000)]
    cand = m["regions"]
    for i, ab in enumerate(sig):
        if i >= len(cand) or cand[i] < 0.35 * ab * h:
            return False
    return True


def solve_weight(apply_fn, ends, flags, WA, target_w, sw_target, h0, d0, hmin, hmax,
                 base, upx=4.0, area_keep=0.45, max_iter=12, width_cap=None, return_err=False,
                 extra_ok=None):
    """Find (h, delta) so that the outline's perpendicular stem/diagonal weight
    (sw_v) reaches sw_target while its ink width stays at target_w (h is
    solved from the width; with target_w None, h is fixed at h0). Candidates
    that close/merge counters, shrink them below `area_keep` of their scaled
    size, or overshoot the width are rejected. Bracketing + bisection."""
    kappa = 1.0
    lo, hi = None, None              # delta too light / too heavy (or invalid)
    best = None
    d = d0

    def evaluate(d):
        nonlocal kappa
        h = h0
        for _ in range(3):           # solve h for the width at this delta
            if target_w is not None and WA:
                h = min(max((target_w - kappa * d) / WA, hmin), hmax)
            pts = apply_fn(h, d)
            w = float(np.ptp(pts[:, 0]))
            if target_w is None or abs(d) < 8 or not (hmin < h < hmax):
                break
            k_new = min(max((w - h * WA) / d, 0.3), 1.6)
            if abs(k_new - kappa) < 0.02:
                break
            kappa = k_new
        m = outline_metrics(pts, ends, flags, upx)
        ok = (m["sw_v"] is not None and shapes_ok(m, base, h, area_keep)
              and (target_w is None or w <= (width_cap or 1.04 * target_w + 8))
              and (extra_ok is None or extra_ok(pts)))
        return h, m["sw_v"], ok

    for _ in range(max_iter):
        h, sw, ok = evaluate(d)
        if ok:
            err = sw_target - sw
            if best is None or abs(err) < abs(best[2]):
                best = (h, d, err)
            if abs(err) < 0.015 * sw_target:
                break
            if err > 0:
                lo = (d, sw) if lo is None or d > lo[0] else lo
            else:
                hi = (d, sw) if hi is None or d < hi[0] else hi
        else:
            hi = (d, None) if hi is None or d < hi[0] else hi
        if lo is not None and hi is not None:
            if hi[1] is not None and hi[1] != lo[1]:
                t = (sw_target - lo[1]) / (hi[1] - lo[1])
                d = lo[0] + min(max(t, 0.2), 0.8) * (hi[0] - lo[0])
            else:
                d = (lo[0] + hi[0]) / 2
            if abs(hi[0] - lo[0]) < 2:
                break
        elif lo is not None:
            d = lo[0] + max(20.0, (sw_target - lo[1]) / 0.6)
        else:
            d = (hi[0] - max(20.0, 0.25 * abs(hi[0]))) if hi[1] is None else hi[0] - max(20.0, (hi[1] - sw_target) / 0.6)
    if best is None:            # nothing valid: keep the source shape (no emboldening)
        h_safe = h0
        if target_w is not None and WA:
            h_safe = min(max(target_w / WA, hmin), hmax)
        return (h_safe, 0.0, None) if return_err else (h_safe, 0.0)
    return (best[0], best[1], best[2] / sw_target) if return_err else (best[0], best[1])


def solve_weight_flex(apply_fn, ends, flags, WA, target_w, adv, sw_target, h0, d0, hmin, hmax, base,
                      extra_ok=None):
    """solve_weight; if the weight can't be reached at the reference ink width
    (design conflict, e.g. a much wider source letter), allow the ink to use up
    to 60% of the remaining sidebearing space."""
    h, d, err = solve_weight(apply_fn, ends, flags, WA, target_w, sw_target, h0, d0, hmin, hmax,
                             base, return_err=True, extra_ok=extra_ok)
    if target_w is None or adv is None or err is None or err < 0.04 or adv <= target_w:
        # solve_weight tolerates up to 4% (+8 units) of extra width; where the
        # weight was reached, take the width back to the target exactly (the
        # letter is centred on the reference's ink, so extra width would land
        # on both sides, as on Ђ)
        if target_w is not None and WA:
            h1 = h
            for _ in range(3):
                w = float(np.ptp(apply_fn(h1, d)[:, 0]))
                if abs(w - target_w) <= 2:
                    break
                h1 = min(max((target_w - (w - h1 * WA)) / WA, hmin), hmax)
            if h1 != h:
                p = apply_fn(h1, d)
                m = outline_metrics(p, ends, flags)
                if shapes_ok(m, base, h1) and (extra_ok is None or extra_ok(p)):
                    h = h1
        return h, d
    cap = target_w + 0.6 * (adv - target_w)
    h2, d2, err2 = solve_weight(apply_fn, ends, flags, WA, cap, sw_target, h, d, hmin, hmax,
                                base, return_err=True, width_cap=cap + 4, extra_ok=extra_ok)
    return (h2, d2) if err2 is not None and err2 < err - 0.02 else (h, d)


def fill_slits(pts, ends, flags, w_min=24.0, max_angle_deg=20.0):
    """Fill the hairline tip of very narrow notches (crotches): where two
    straight edges meet in a concave angle < max_angle, replace the tip with a
    flat end exactly w_min wide. Returns new (pts, ends, flags)."""
    pts = np.asarray(pts, float)
    out_p, out_f, out_e, start = [], [], [], 0
    for end in ends:
        P, F = pts[start:end + 1], list(flags[start:end + 1])
        n = len(P)
        for i in range(n):
            a, p, b = P[i - 1], P[i], P[(i + 1) % n]
            done = False
            if F[i] & 1 and F[i - 1] & 1 and F[(i + 1) % n] & 1:
                u, v = a - p, b - p
                lu, lv = np.hypot(*u), np.hypot(*v)
                if lu > 1 and lv > 1:
                    cross = (p - a)[0] * (b - p)[1] - (p - a)[1] * (b - p)[0]
                    ang = np.degrees(np.arccos(np.clip(u @ v / (lu * lv), -1, 1)))
                    if cross > 0 and 0 < ang < max_angle_deg:
                        t = w_min / (2 * np.tan(np.radians(ang) / 2))
                        if t < 0.8 * min(lu, lv):
                            out_p += [p + u / lu * t, p + v / lv * t]
                            out_f += [1, 1]
                            done = True
            if not done:
                out_p.append(p)
                out_f.append(F[i])
        out_e.append(len(out_p) - 1)
        start = end + 1
    return np.array(out_p), out_e, out_f


def tuck_parts(parts, adv, base_top, top_limit, bot_limit, ov=40.0, debug=None, xlim=None):
    """Keep accents inside the Windows clip box and close to the advance.

    parts: list of dicts {pts (N,2 float), flags, ends, role 'base'|'att'}.
    Vertical: an accent group above the base that exceeds top_limit first
    loses up to 40% of its gap, then is compressed (to 60%, 50% if stacked);
    marks hanging below bot_limit are compressed toward their top or lifted.
    Horizontal: each accent must lie within [-ov, adv+ov] (or the base's own
    extent); it is shifted, or for multi-part marks (dieresis) the parts are
    moved closer together, or as a last resort narrowed.
    Returns merged (pts, flags, ends) or None if nothing changed."""
    base = [p for p in parts if p["role"] == "base"]
    att = [p for p in parts if p["role"] == "att"]
    if not base or not att:
        return None

    def ext(p):
        return p["pts"][:, 0].min(), p["pts"][:, 1].min(), p["pts"][:, 0].max(), p["pts"][:, 1].max()
    allb = np.vstack([p["pts"] for p in base])
    lo, hi = min(-ov, allb[:, 0].min()), max(adv + ov, allb[:, 0].max())
    above = [p for p in att if ext(p)[1] >= base_top - 60]
    above_ids = {id(p) for p in above}
    below = [p for p in att if id(p) not in above_ids and (ext(p)[1] + ext(p)[3]) / 2 < 0]
    changed = False

    def base_ink_y(x0, x1, top):
        """Highest (top=True) or lowest point of the base's outline between
        x0 and x1, measured along its edges (flat bars have no points there)."""
        best = None
        for p in base:
            for poly in flatten(p["pts"], p["ends"], p["flags"], steps=8):
                n = len(poly)
                for i in range(n):
                    (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
                    lo_, hi_ = max(min(xa, xb), x0), min(max(xa, xb), x1)
                    if lo_ > hi_:
                        continue
                    if xa == xb:
                        cand = (ya, yb)
                    else:
                        cand = tuple(ya + (xx - xa) / (xb - xa) * (yb - ya) for xx in (lo_, hi_))
                    v = max(cand) if top else min(cand)
                    if best is None or (v > best if top else v < best):
                        best = v
        return best

    if above:
        ab, at = min(ext(p)[1] for p in above), max(ext(p)[3] for p in above)
        if at > top_limit:
            gap = max(ab - base_top, 0)
            red = min(at - top_limit, 0.4 * gap)
            new_gap, rest = gap - red, at - top_limit - red
            floor = 0.5 if len(above) >= 2 else 0.6
            f = max(floor, (at - ab - rest) / (at - ab)) if rest > 0 else 1.0
            for p in above:
                p["pts"][:, 1] = base_top + new_gap + (p["pts"][:, 1] - ab) * f
            changed = True
        # keep the accent clear of base parts that rise into its span (the
        # slash of Ø, a horn): at least 30 units of white
        ab, at = min(ext(p)[1] for p in above), max(ext(p)[3] for p in above)
        ax0, ax1 = min(ext(p)[0] for p in above), max(ext(p)[2] for p in above)
        ceil = base_ink_y(ax0 - 10, ax1 + 10, True)
        if ceil is not None and ab < ceil + 30 and at > ceil:
            # first try sliding the accent sideways off the obstacle (up to a
            # third of its width), as Anton does beside the slash of Ø
            thr, xs = ab - 30, []
            for p in base:
                for poly in flatten(p["pts"], p["ends"], p["flags"], steps=8):
                    n = len(poly)
                    for i in range(n):
                        (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
                        if max(ya, yb) < thr:
                            continue
                        for (x_, y_) in ((xa, ya), (xb, yb)):
                            if y_ >= thr:
                                xs.append(x_)
                        if (ya - thr) * (yb - thr) < 0:
                            xs.append(xa + (thr - ya) / (yb - ya) * (xb - xa))
            xs = [x_ for x_ in xs if ax0 - 10 <= x_ <= ax1 + 10]
            if xs:
                ox0, ox1 = min(xs), max(xs)
                s_ = (ox0 - 30 - ax1) if (ox0 + ox1) / 2 > (ax0 + ax1) / 2 else (ox1 + 30 - ax0)
                if abs(s_) <= (ax1 - ax0) / 3 and lo <= ax0 + s_ and ax1 + s_ <= hi:
                    for p in above:
                        p["pts"][:, 0] += s_
                    ax0, ax1 = ax0 + s_, ax1 + s_
                    ceil = base_ink_y(ax0 - 10, ax1 + 10, True)
                    changed = True
                    if debug:
                        print(f"[tuck {debug}] above: slid {s_:.0f} off base ink at x {ox0:.0f}..{ox1:.0f}")
        if ceil is not None and ab < ceil + 30 and at > ceil:
            lift = ceil + 30 - ab
            room = top_limit - at
            if lift <= room:
                for p in above:
                    p["pts"][:, 1] += lift
            else:
                nb = ceil + 30
                f = max(0.5, (top_limit - nb) / (at - ab))
                for p in above:
                    p["pts"][:, 1] = nb + (p["pts"][:, 1] - ab) * f
            if debug:
                print(f"[tuck {debug}] above: cleared base ink at {ceil:.0f} (accent bottom was {ab:.0f})")
            changed = True
    if below:
        bb, bt = min(ext(p)[1] for p in below), max(ext(p)[3] for p in below)
        bx0 = min(ext(p)[0] for p in below)
        bx1 = max(ext(p)[2] for p in below)
        floor_y = base_ink_y(bx0 - 20, bx1 + 20, False)       # letter's bottom above the mark
        if floor_y is None:
            floor_y = allb[:, 1].min()
        # a closed mark (ring below) is never squashed: it is scaled evenly and
        # set clear of the letter, even where the source lets it touch
        ring = (len(below) == 1 and len(below[0]["ends"]) >= 2
                and topology(below[0]["pts"], below[0]["ends"], below[0]["flags"])[0] >= 1)
        # cedilla/ogonek: hangs from the letter and must stay joined to it
        hanging = not ring and bt >= floor_y - 30
        if debug:
            print(f"[tuck {debug}] below bb={bb:.0f} bt={bt:.0f} x={bx0:.0f}..{bx1:.0f} floor={floor_y:.0f} "
                  f"bot_limit={bot_limit:.0f} ring={ring} hanging={hanging}")
        if hanging:
            top_new = max(bt, floor_y + 10)                  # overlap the letter a little
            sc = min(1.0, (top_new - bot_limit) / (bt - bb))
            if top_new != bt or sc < 1.0:
                band = np.vstack([p["pts"] for p in below])
                band = band[band[:, 1] > bt - 0.2 * (bt - bb)]
                ax = (band[:, 0].min() + band[:, 0].max()) / 2   # where it joins the letter
                for p in below:
                    p["pts"][:, 0] = ax + (p["pts"][:, 0] - ax) * sc
                    p["pts"][:, 1] = top_new + (p["pts"][:, 1] - bt) * sc
                changed = True
        elif bb < bot_limit:
            avail = (floor_y - 20) - bot_limit
            if bt + (bot_limit - bb) < floor_y - 20:      # room to lift it intact
                for p in below:
                    p["pts"][:, 1] += bot_limit - bb
            elif avail >= 0.45 * (bt - bb):                # dot, comma, ring: shrink to fit
                sc = avail / (bt - bb)
                cx = (bx0 + bx1) / 2
                for p in below:
                    p["pts"][:, 0] = cx + (p["pts"][:, 0] - cx) * sc
                    p["pts"][:, 1] = bot_limit + (p["pts"][:, 1] - bb) * sc
            elif bt > bot_limit + 40:
                f = max(0.5, (bt - bot_limit) / (bt - bb))
                for p in below:
                    p["pts"][:, 1] = bt + (p["pts"][:, 1] - bt) * f
            else:
                for p in below:
                    p["pts"][:, 1] += bot_limit - bb
            changed = True
    for p in att:
        x0, _, x1, _ = ext(p)
        if x0 >= lo - 0.5 and x1 <= hi + 0.5:
            continue
        W, R = x1 - x0, hi - lo
        if W <= R:
            p["pts"][:, 0] += (lo - x0) if x0 < lo else (hi - x1)
        else:
            # contour-wise centres: move separate pieces (dots) closer
            spans, start = [], 0
            for e in p["ends"]:
                seg = p["pts"][start:e + 1, 0]
                spans.append((start, e, seg.min(), seg.max()))
                start = e + 1
            cmid = (lo + hi) / 2
            gmid = (x0 + x1) / 2
            if len(spans) >= 2:
                cs = [(a + b) / 2 for _, _, a, b in spans]
                piece = max(b - a for _, _, a, b in spans)
                sep = max(cs) - min(cs)
                s = (R - piece) / sep if sep > 0 else 1.0
                if sep * s - piece >= 0.3 * piece:
                    for (st, en, a, b), c in zip(spans, cs):
                        p["pts"][st:en + 1, 0] += cmid + (c - gmid) * s - c
                    continue
            p["pts"][:, 0] = cmid + (p["pts"][:, 0] - gmid) * (R / W)
        changed = True
    # accents above the letter kept off the neighbours. A one-piece accent may
    # use the reference's ink box (xlim) or the advance less 15 units, whichever
    # is wider on that side; past it, it moves inward (up to 25 units, without
    # crossing the other side), then is narrowed (to 85% at most). Accents in
    # separate pieces (dieresis, double acute) keep to the reference's ink box:
    # the pieces are moved closer together, never squashed.
    if xlim is not None and above:
        ax0 = min(ext(p)[0] for p in above)
        ax1 = max(ext(p)[2] for p in above)
        pieces = []                      # contours grouped where their x-spans overlap
        for p in above:
            st = 0
            for en in p["ends"]:
                a, b = p["pts"][st:en + 1, 0].min(), p["pts"][st:en + 1, 0].max()
                for pc in pieces:
                    if a <= pc["b"] and b >= pc["a"]:
                        pc["segs"].append((p, st, en))
                        pc["a"], pc["b"] = min(pc["a"], a), max(pc["b"], b)
                        break
                else:
                    pieces.append(dict(a=a, b=b, segs=[(p, st, en)]))
                st = en + 1
        multi = len(pieces) > 1 and max(pc["b"] - pc["a"] for pc in pieces) < 0.45 * (ax1 - ax0)
        lo_, hi_ = (xlim[0], xlim[1]) if multi else (min(xlim[0], 15.0), max(xlim[1], adv - 15.0))
        over_l, over_r = lo_ - ax0, ax1 - hi_
        # only accents standing over the letter (not the apostrophe of ŉ)
        over_base = allb[:, 0].min() <= (ax0 + ax1) / 2 <= allb[:, 0].max()
        if over_base and (over_l > 2 or over_r > 2):
            c0, half = (ax0 + ax1) / 2, (ax1 - ax0) / 2
            if over_r <= 0:
                sh = min(over_l, -over_r, 25.0)
            elif over_l <= 0:
                sh = -min(over_r, -over_l, 25.0)
            else:
                sh = min(max((lo_ + hi_) / 2 - c0, -25.0), 25.0)
            c1 = c0 + sh
            s = max(0.85, min(1.0, (hi_ - c1) / half, (c1 - lo_) / half))
            for pc in pieces:
                pm = (pc["a"] + pc["b"]) / 2
                for p, st, en in pc["segs"]:
                    if multi:
                        p["pts"][st:en + 1, 0] += c1 + (pm - c0) * s - pm
                    else:
                        p["pts"][st:en + 1, 0] = c1 + (p["pts"][st:en + 1, 0] - c0) * s
            if debug:
                print(f"[tuck {debug}] accent {ax0:.0f}..{ax1:.0f} into {lo_:.0f}..{hi_:.0f}: centre {c0:.0f}->{c1:.0f}, x{s:.2f}")
            changed = True
    if not changed:
        return None
    pts, flags, ends = [], [], []
    for p in parts:
        n0 = len(pts)
        pts += [tuple(x) for x in p["pts"]]
        flags += [f & 1 for f in p["flags"]]
        ends += [e + n0 for e in p["ends"]]
    return np.array(pts), flags, ends


def composite_parts(glyf, name):
    """Decompose a composite into parts (first component = base)."""
    parts = []
    for i, c in enumerate(glyf[name].components):
        cg = glyf[c.glyphName]
        if cg.numberOfContours == 0:
            continue
        co, en, fl = cg.getCoordinates(glyf)
        pts = np.array(co, float) + (c.x, c.y)
        parts.append(dict(pts=pts, flags=list(fl), ends=list(en), role="base" if i == 0 else "att"))
    return parts


def contour_parts(glyf, name, body_top):
    """Split a simple glyph into body contours and accent contours above."""
    g = glyf[name]
    co = np.array(g.coordinates, float)
    parts, start = [], 0
    for e in g.endPtsOfContours:
        pts = co[start:e + 1].copy()
        parts.append(dict(pts=pts, flags=list(g.flags[start:e + 1]), ends=[e - start],
                          role="att" if pts[:, 1].min() >= body_top - 60 else "base"))
        start = e + 1
    return parts


def drop_tiny_contours(glyph, min_area=600.0):
    """Remove pinhole/speck contours left over from outline operations."""
    co = np.array(glyph.coordinates, float)
    keep_p, keep_f, keep_e, start = [], [], [], 0
    for e in glyph.endPtsOfContours:
        seg, fl = co[start:e + 1], list(glyph.flags[start:e + 1])
        polys = flatten(seg, [len(seg) - 1], fl)
        a = abs(poly_area(polys[0])) if polys else 0
        if a >= min_area:
            keep_p += [tuple(x) for x in seg]
            keep_f += fl
            keep_e.append(len(keep_p) - 1)
        start = e + 1
    return keep_p, keep_f, keep_e, len(keep_e) != len(glyph.endPtsOfContours)


# ------------------------------------------- exact (morphological) emboldening
def tt_segments(pts, ends, flags):
    """TrueType contours -> list of contours, each a list of segments
    ('L', p0, p1) or ('Q', p0, c, p1) with implied on-curve points resolved."""
    pts = np.asarray(pts, float)
    out, start = [], 0
    for end in ends:
        P, F = pts[start:end + 1], [f & 1 for f in flags[start:end + 1]]
        start = end + 1
        n = len(P)
        if n < 2:
            continue
        seq = []
        for i in range(n):
            seq.append((P[i], F[i]))
            if not F[i] and not F[(i + 1) % n]:
                seq.append(((P[i] + P[(i + 1) % n]) / 2, 1))
        k = next((i for i, s in enumerate(seq) if s[1]), None)
        if k is None:
            continue
        seq = seq[k:] + seq[:k]
        m, i, segs = len(seq), 0, []
        while i < m:
            p0, nxt = seq[i][0], seq[(i + 1) % m]
            if nxt[1]:
                segs.append(("L", p0, nxt[0]))
                i += 1
            else:
                segs.append(("Q", p0, nxt[0], seq[(i + 2) % m][0]))
                i += 2
        out.append(segs)
    return out


def _split_monotone(seg, axis):
    """Split a quadratic at its extremum along `axis` (0=x, 1=y)."""
    if seg[0] != "Q":
        return [seg]
    _, p0, c, p1 = seg
    den = p0[axis] - 2 * c[axis] + p1[axis]
    if abs(den) < 1e-9:
        return [seg]
    t = (p0[axis] - c[axis]) / den
    if not (1e-4 < t < 1 - 1e-4):
        return [seg]
    a, b = p0 + (c - p0) * t, c + (p1 - c) * t
    m = a + (b - a) * t
    return [("Q", p0, a, m), ("Q", m, b, p1)]


def _path_from_segments(contours):
    import pathops
    path = pathops.Path(fillType=pathops.FillType.WINDING)
    pen = path.getPen()
    for segs in contours:
        pen.moveTo(tuple(segs[0][1]))
        for s in segs:
            if s[0] == "L":
                pen.lineTo(tuple(s[2]))
            else:
                pen.qCurveTo(tuple(s[2]), tuple(s[3]))
        pen.closePath()
    return path


def _sweep_path(contours, v):
    """Union of the regions swept by every outline segment moving by vector v."""
    import pathops
    axis = 1 if abs(v[0]) > abs(v[1]) else 0       # split monotone across the sweep
    path = pathops.Path(fillType=pathops.FillType.WINDING)
    pen = path.getPen()
    v = np.asarray(v, float)
    for segs in contours:
        for seg0 in segs:
            for s in _split_monotone(seg0, axis):
                if s[0] == "L":
                    p0, p1 = s[1], s[2]
                    poly = [p0, p1, p1 + v, p0 + v]
                    area = 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))
                    if abs(area) < 1e-6:
                        continue
                    if area < 0:
                        poly = poly[::-1]
                    pen.moveTo(tuple(poly[0]))
                    for q in poly[1:]:
                        pen.lineTo(tuple(q))
                    pen.closePath()
                else:
                    _, p0, c, p1 = s
                    # orientation from the control polygon of the closed ribbon
                    ring = [p0, c, p1, p1 + v, c + v, p0 + v]
                    area = 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:] + ring[:1]))
                    if abs(area) < 1e-6:
                        continue
                    if area > 0:
                        pen.moveTo(tuple(p0))
                        pen.qCurveTo(tuple(c), tuple(p1))
                        pen.lineTo(tuple(p1 + v))
                        pen.qCurveTo(tuple(c + v), tuple(p0 + v))
                        pen.closePath()
                    else:
                        pen.moveTo(tuple(p0 + v))
                        pen.qCurveTo(tuple(c + v), tuple(p1 + v))
                        pen.lineTo(tuple(p1))
                        pen.qCurveTo(tuple(c), tuple(p0))
                        pen.closePath()
    path.simplify(fix_winding=True, clockwise=True)
    return path


def _path_to_tt(path):
    """pathops Path -> (pts, ends, flags) TrueType quadratic contours."""
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.pens.cu2quPen import Cu2QuPen
    pen = TTGlyphPen(None)
    path.draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=False))
    g = pen.glyph()
    if g.numberOfContours <= 0:
        return np.zeros((0, 2)), [], []
    return np.array(g.coordinates, float), list(g.endPtsOfContours), [f & 1 for f in g.flags]


def morph_bold(pts, ends, flags, dx=0.0, dy=0.0, dy_zone=None):
    """Exact stroke-weight change. dx > 0: every right-facing edge moves right by
    dx (left-facing edges stay); dx < 0: right-facing edges move left by |dx|.
    dy > 0 likewise moves up-facing edges up. dy_zone=(y0, y1) restricts the
    vertical change to that band. Straight edges stay straight; corners stay
    crisp; narrow notches get shallower instead of turning into slivers."""
    import pathops
    if abs(dx) <= 0.5 and abs(dy) <= 0.5:
        return np.asarray(pts, float), list(ends), list(flags)
    contours = tt_segments(pts, ends, flags)
    if not contours:
        return np.asarray(pts, float), list(ends), list(flags)
    base = _path_from_segments(contours)
    base.simplify(fix_winding=True, clockwise=True)
    cur = base
    if abs(dx) > 0.5:
        sw = _sweep_path(contours, (dx, 0.0))
        cur = pathops.op(cur, sw, pathops.PathOp.UNION if dx > 0 else pathops.PathOp.DIFFERENCE,
                         fix_winding=True, clockwise=True)
    if abs(dy) > 0.5:
        # continue from the exact TrueType form of the intermediate outline so
        # the sweep and the shape share identical edges (no hairline cracks)
        p2, e2, f2 = _path_to_tt(cur)
        c2 = tt_segments(p2, e2, f2)
        cur = _path_from_segments(c2)
        cur.simplify(fix_winding=True, clockwise=True)
        sw = _sweep_path(c2, (0.0, dy))
        if dy_zone is not None:
            box = pathops.Path()
            bp = box.getPen()
            y0, y1 = dy_zone
            bp.moveTo((-5000, y0)); bp.lineTo((5000, y0)); bp.lineTo((5000, y1)); bp.lineTo((-5000, y1)); bp.closePath()
            sw = pathops.op(sw, box, pathops.PathOp.INTERSECTION, fix_winding=True)
        cur = pathops.op(cur, sw, pathops.PathOp.UNION if dy > 0 else pathops.PathOp.DIFFERENCE,
                         fix_winding=True, clockwise=True)
    return _path_to_tt(cur)


def embolden_xy_v2(pts, ends, xstr, ystr):
    """FreeType-style anisotropic emboldening (each edge moves in proportion to
    how much it faces right/up, so round counters keep their height), but with
    exact miter joins: the short-segment limiter that made edges tilt is only
    applied at very sharp turns. Strengths may be scalars or per-point arrays.
    Result should be passed through nonzero cleanup (removeOverlaps)."""
    n_all = len(pts)
    xs_all = np.broadcast_to(np.asarray(xstr, float), (n_all,)) / 2.0
    ys_all = np.broadcast_to(np.asarray(ystr, float), (n_all,)) / 2.0
    if not np.any(xs_all) and not np.any(ys_all):
        return np.asarray(pts, float)
    pts = np.asarray(pts, float)
    area, first = 0.0, 0
    for last in ends:
        p = pts[first:last + 1]
        area += np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        first = last + 1
    truetype = area < 0
    out = pts.copy()
    first = 0
    for last in ends:
        P = pts[first:last + 1]
        n = len(P)
        for i in range(n):
            xs, ys = xs_all[first + i], ys_all[first + i]
            j = (i - 1) % n
            while j != i and np.allclose(P[j], P[i]):
                j = (j - 1) % n
            k = (i + 1) % n
            while k != i and np.allclose(P[k], P[i]):
                k = (k + 1) % n
            if j == i or k == i:
                continue
            vin, vout = P[i] - P[j], P[k] - P[i]
            l_in, l_out = np.hypot(*vin), np.hypot(*vout)
            vin, vout = vin / l_in, vout / l_out
            d = vin @ vout
            sx = sy = 0.0
            if d > -0.9375:
                d += 1.0
                sx, sy = vin[1] + vout[1], vin[0] + vout[0]
                if truetype:
                    sx = -sx
                else:
                    sy = -sy
                q = vout[0] * vin[1] - vout[1] * vin[0]
                if truetype:
                    q = -q
                l = min(l_in, l_out)
                # exact miter joins, except at sharp *outer* corners (where an
                # unlimited miter would grow a spike) and near-reversals
                limit = (d < 0.5 and q < 0) or d < 0.0625
                sx = sx * xs / d if (not limit or xs * q <= l * d) else sx * l / q
                sy = sy * ys / d if (not limit or ys * q <= l * d) else sy * l / q
            out[first + i, 0] = P[i, 0] + xs + sx
            out[first + i, 1] = P[i, 1] + ys + sy
        first = last + 1
    return out


def clean_nonzero(pts, ends, flags):
    """Resolve self-overlaps/loops under the nonzero rule into clean contours.
    Output keeps the TrueType convention (outer contours clockwise), so the
    result still overlaps cleanly with un-processed parts such as accents."""
    import pathops
    contours = tt_segments(pts, ends, flags)
    if not contours:
        return np.asarray(pts, float), list(ends), list(flags)
    path = _path_from_segments(contours)
    path.simplify(fix_winding=True, clockwise=True)
    return _path_to_tt(path)


def openness(pts, ends, flags, upx=4.0):
    """Share of white between the outermost strokes, row by row (counters,
    notches, apertures, gaps between strokes)."""
    a = raster_outline(pts, ends, flags, upx)
    w = 0
    for r in range(a.shape[0]):
        rr = runs(a[r])
        if len(rr) > 1:
            w += sum(rr[i + 1][0] - rr[i][1] for i in range(len(rr) - 1))
    return w / max(1, a.sum() + w)


def _scan_gaps(polys, y):
    """Ink intervals of nonzero-filled polygons on the horizontal line y."""
    xs = []
    for P in polys:
        n = len(P)
        for i in range(n):
            (xa, ya), (xb, yb) = P[i], P[(i + 1) % n]
            if (ya <= y < yb) or (yb <= y < ya):
                xs.append((xa + (y - ya) / (yb - ya) * (xb - xa), 1 if yb > ya else -1))
    xs.sort()
    ink, wind, start = [], 0, None
    for x, d in xs:
        prev = wind
        wind += d
        if prev == 0 and wind != 0:
            start = x
        elif prev != 0 and wind == 0:
            ink.append((start, x))
    return ink


def fill_narrow_channels(pts, ends, flags, w_max=36.0, min_len=40.0, step=4.0):
    """Close the hairline part of narrow dead-end white channels between two
    strokes (e.g. the crotch of an emboldened У): rows where the gap between two
    ink runs is narrower than w_max, counted from the closed end of the channel,
    are filled with ink. Channels closed at both ends (counters) are left
    alone. Returns (pts, ends, flags, filled_area)."""
    import pathops
    pts = np.asarray(pts, float)
    polys = nonzero_polys(pts, ends, flags)
    if not polys:
        return pts, list(ends), list(flags), 0.0
    allp = np.vstack(polys)
    ys = np.arange(allp[:, 1].min() + step / 2, allp[:, 1].max(), step)
    rows, inks = [], []
    for y in ys:
        ink = _scan_gaps(polys, y)
        inks.append(ink)
        rows.append([(ink[i][1], ink[i + 1][0]) for i in range(len(ink) - 1)])
    fills = []

    def capped(k, gap):          # ink spans the whole gap on row k: a closed end
        return 0 <= k < len(rows) and any(a <= gap[0] + 1.0 and b >= gap[1] - 1.0 for a, b in inks[k])

    def overlap(a, b):          # tolerant: a channel starts as a zero-width gap
        return a[0] <= b[1] + 1.0 and b[0] <= a[1] + 1.0

    for direction in (1, -1):              # closed at the bottom / at the top
        rng = range(len(rows)) if direction == 1 else range(len(rows) - 1, -1, -1)
        for k in rng:
            for gap in rows[k]:
                prev_k = k - direction
                if 0 <= prev_k < len(rows) and any(overlap(gap, g) for g in rows[prev_k]):
                    continue                    # not the end of a channel
                if not capped(prev_k, gap):
                    continue                    # open to the outside, not a crotch
                if gap[1] - gap[0] >= w_max:
                    continue
                run, kk, cur = [], k, gap
                while 0 <= kk < len(rows):
                    nxt = [g for g in rows[kk] if overlap(g, cur)]
                    if len(nxt) != 1:
                        break
                    cur = nxt[0]
                    if cur[1] - cur[0] >= w_max:
                        break
                    run.append((ys[kk], cur))
                    kk += direction
                else:
                    run = []                    # ran off the glyph: not a crotch
                if not run or not (0 <= kk < len(rows)) or not any(overlap(cur, g) for g in rows[kk]):
                    continue                    # never widens: a counter or a hairline loop
                if len(run) * step < min_len:
                    continue
                left = [(g[0] - 1.0, y) for y, g in run]
                right = [(g[1] + 1.0, y) for y, g in run]
                y_end = run[0][0] - direction * step / 2      # reach the closed end
                y_top = run[-1][0] + direction * step / 2
                left = [(run[0][1][0] - 1.0, y_end)] + left + [(run[-1][1][0] - 1.0, y_top)]
                right = [(run[0][1][1] + 1.0, y_end)] + right + [(run[-1][1][1] + 1.0, y_top)]
                fills.append(left + right[::-1])
    if not fills:
        return pts, list(ends), list(flags), 0.0
    acc = _path_from_segments(tt_segments(pts, ends, flags))
    acc.simplify(fix_winding=True, clockwise=True)
    area = 0.0
    for F in fills:
        part = pathops.Path()
        pen = part.getPen()
        pen.moveTo(tuple(F[0]))
        for q in F[1:]:
            pen.lineTo(tuple(q))
        pen.closePath()
        part.simplify(fix_winding=True, clockwise=True)
        area += abs(poly_area(np.array(F)))
        acc = pathops.op(acc, part, pathops.PathOp.UNION, fix_winding=True, clockwise=True)
    p, e, f = _path_to_tt(acc)
    return p, e, f, area


def tidy_points(pts, ends, flags, min_seg=2.0):
    """Drop duplicate consecutive points and on-curve points closer than
    min_seg to the previous on-curve point (no off-curve between them).
    Returns (pts, ends, flags, n_removed)."""
    pts = np.asarray(pts, float)
    out_p, out_f, out_e, removed, st = [], [], [], 0, 0
    for en in ends:
        P, F = pts[st:en + 1], list(flags[st:en + 1])
        st = en + 1
        keep = list(range(len(P)))
        changed = True
        while changed and len(keep) > 3:
            changed = False
            for j in range(len(keep)):
                a, b = keep[j - 1], keep[j]
                same = np.allclose(P[a], P[b])
                short = (F[a] & 1) and (F[b] & 1) and np.hypot(*(P[a] - P[b])) < min_seg
                if same or short:
                    drop = b if (F[a] & 1 or not F[b] & 1) else a
                    keep.remove(drop)
                    removed += 1
                    changed = True
                    break
        out_p.extend(P[k] for k in keep)
        out_f.extend(F[k] for k in keep)
        out_e.append(len(out_p) - 1)
    return np.array(out_p), out_e, out_f, removed


def side_protrusion(pts, ends, flags, side, limit, step=4.0):
    """Where the ink on one side ('L' or 'R') passes `limit`, return
    (inner, extreme, y0, y1): the stroke edge the protrusion grows from
    (see below; None if the whole height passes the limit), the extreme edge,
    and the band's vertical extent. None if nothing passes."""
    pts = np.asarray(pts, float)
    polys = nonzero_polys(pts, list(ends), [f & 1 for f in flags])
    rows = []
    for y in np.arange(pts[:, 1].min() + step / 2, pts[:, 1].max(), step):
        ink = _scan_gaps(polys, y)
        if ink:
            rows.append((y, ink[0][0] if side == "L" else ink[-1][1]))
    if not rows:
        return None
    rows = np.array(rows)
    over = (limit - rows[:, 1]) if side == "L" else (rows[:, 1] - limit)
    if over.max() <= 0:
        return None
    rest = rows[over <= 0, 1]
    inner = None
    if len(rest):
        # the stroke the protrusion grows from: a straight stem's edge if one
        # runs along most of the rest of the height, else the furthest edge
        vals, cnt = np.unique(np.round(rest / 2) * 2, return_counts=True)
        if cnt.max() >= 0.25 * len(rest):
            inner = float(vals[cnt.argmax()])
        else:
            inner = rest.min() if side == "L" else rest.max()
    ext = rows[:, 1].min() if side == "L" else rows[:, 1].max()
    band = rows[over > 0, 0]
    return inner, ext, band.min(), band.max()


def squeeze_side(pts, side, inner, target, y0=-1e9, y1=1e9):
    """Compress, horizontally and toward `inner`, everything lying beyond
    `inner` on one side (between y0 and y1) so the extreme lands on `target`
    (a bar end, a tail): piecewise linear in x, the rest untouched."""
    pts = np.array(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    band = (y >= y0) & (y <= y1)
    if not band.any():
        return pts
    if side == "R":
        ext = x[band].max()
        if ext <= target or target <= inner:
            return pts
        m = band & (x > inner)
        x[m] = inner + (x[m] - inner) * (target - inner) / (ext - inner)
    else:
        ext = x[band].min()
        if ext >= target or target >= inner:
            return pts
        m = band & (x < inner)
        x[m] = inner - (inner - x[m]) * (inner - target) / (inner - ext)
    return pts


def lean_in(pts, side, amount, y_tip, y_base, x_from, ramp):
    """Lean a diagonal stroke's end inward: points beyond x_from (fading in
    over `ramp` units) move toward the glyph by `amount` at y_tip, fading
    linearly to nothing at y_base. Straight strokes stay straight."""
    pts = np.array(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    wy = np.clip((y - y_base) / (y_tip - y_base), 0, 1)
    if side == "R":
        x -= amount * wy * np.clip((x - x_from) / ramp, 0, 1)
    else:
        x += amount * wy * np.clip((x_from - x) / ramp, 0, 1)
    return pts


def edge_at(pts, ends, flags, y, side):
    """Outermost ink edge on one side at height y (None if no ink there)."""
    ink = _scan_gaps(nonzero_polys(np.asarray(pts, float), list(ends), [f & 1 for f in flags]), y)
    return (ink[0][0] if side == "L" else ink[-1][1]) if ink else None


def pair_gap(pa, ea, fa, adv_a, pb, eb, fb, y0, y1, step=4.0):
    """Smallest white between glyph a and glyph b set after it, over y0..y1."""
    best = None
    for y in np.arange(y0, y1 + 0.1, step):
        r, l = edge_at(pa, ea, fa, y, "R"), edge_at(pb, eb, fb, y, "L")
        if r is not None and l is not None:
            g = adv_a + l - r
            best = g if best is None or g < best else best
    return best


def notch_weights(pts, ends, flags, radius=100.0, min_turn_deg=35.0):
    """Per-point weights (0..1) that fall to 0 at sharp concave corners of the
    ink (notches: where n's arch meets its stem, the waist of 3) and rise back
    to 1 over `radius` units. Used to keep notches as deep as they were when
    strokes are thinned (TrueType: ink lies right of the direction of travel,
    so a left turn is concave)."""
    pts = np.asarray(pts, float)
    notches, st = [], 0
    for en in ends:
        P = pts[st:en + 1]
        F = flags[st:en + 1]
        on = [i for i in range(len(P)) if F[i] & 1]
        n = len(on)
        if n >= 3:
            for k in range(n):
                a, b, c = P[on[k - 1]], P[on[k]], P[on[(k + 1) % n]]
                # tangents from the neighbouring points (off-curve included)
                i = on[k]
                pa = P[i - 1] if i > 0 else P[-1]
                pc = P[(i + 1) % len(P)]
                vin, vout = b - pa, pc - b
                li, lo = np.hypot(*vin), np.hypot(*vout)
                if li < 1e-6 or lo < 1e-6:
                    continue
                cross = vin[0] * vout[1] - vin[1] * vout[0]
                ang = np.degrees(np.arccos(np.clip((vin @ vout) / (li * lo), -1, 1)))
                if cross > 0 and ang >= min_turn_deg:
                    notches.append(b)
        st = en + 1
    if not notches:
        return np.ones(len(pts))
    N = np.array(notches)
    d = np.sqrt(((pts[:, None, :] - N[None, :, :]) ** 2).sum(-1)).min(1)
    return np.clip(d / radius, 0.0, 1.0)


def snap_flats(pts, ends, flags, zones, tol=2.0):
    """Move flat (horizontal) edges lying within `tol` units of a zone height
    (baseline, x-height, cap height) onto it: rounding in the fitting left a
    few letters 1-2 units off (Cyrillic lowercase at 1328 against Latin 1327,
    V X at 1620). Returns (pts, number of points moved)."""
    pts = np.array(pts, float)
    moved, st = 0, 0
    for en in ends:
        idx = list(range(st, en + 1))
        on = [i for i in idx if flags[i] & 1]
        n = len(on)
        for k in range(n):
            a, b = on[k], on[(k + 1) % n]
            # a straight segment: consecutive on-curve points in the contour
            if b != a + 1 and not (a == en and b == st):
                continue
            ya, yb = pts[a, 1], pts[b, 1]
            if abs(ya - yb) > 0.5 or abs(pts[a, 0] - pts[b, 0]) < 20:
                continue
            for z in zones:
                if 0 < abs(ya - z) <= tol:
                    pts[a, 1] = pts[b, 1] = z
                    moved += 2
                    break
        st = en + 1
    return pts, moved
