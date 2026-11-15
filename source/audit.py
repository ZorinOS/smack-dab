"""Error audit for Smack Dab.

usage: python3 -I audit.py SmackDab-unhinted.ttf Anton-Regular.ttf Oswald-Bold-700.ttf impact_metrics.json
"""
import json
import os
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontops  # noqa: E402
import freetype  # noqa: E402
import pathops  # noqa: E402
import uharfbuzz as hb  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402

sd_p, anton_p, osw_p, metrics_p = sys.argv[1:5]
S, A, O = TTFont(sd_p), TTFont(anton_p), TTFont(osw_p)
ref = {int(k, 16): v for k, v in json.load(open(metrics_p))["chars"].items()}
sg, ag, og = S["glyf"], A["glyf"], O["glyf"]
sc, oc = S.getBestCmap(), O.getBestCmap()
rev = {g: c for c, g in sc.items()}
issues = {}


def report(kind, item):
    issues.setdefault(kind, []).append(item)


def outline(glyf, name, pre=1.0):
    c, e, f = glyf[name].getCoordinates(glyf)
    return np.array(c, float) * pre, list(e), [x & 1 for x in f]


def area(pts, ends, flags, fill):
    p = pathops.Path(fillType=fill)
    pen = p.getPen()
    for poly in fontops.flatten(pts, ends, flags, steps=10):
        pen.moveTo(tuple(poly[0]))
        for q in poly[1:]:
            pen.lineTo(tuple(q))
        pen.closePath()
    p.simplify(fix_winding=True)
    return abs(p.area)


# 1. outline integrity -------------------------------------------------------
marks = {g for g, c in S["GDEF"].table.GlyphClassDef.classDefs.items() if c == 3}
# accents keep Anton's size or shrink (to fit the clip box); one that grew
# was mangled (e.g. two dots mistaken for a ring)
_upm_ratio = S["head"].unitsPerEm / A["head"].unitsPerEm
for name in sorted(marks):
    if name in A.getGlyphOrder() and sg[name].numberOfContours and A["glyf"][name].numberOfContours:
        sp, ap = np.array(outline(sg, name)[0]), np.array(outline(ag, name)[0])
        for axis, label in ((0, "wider"), (1, "taller")):
            s_ext, a_ext = np.ptp(sp[:, axis]), np.ptp(ap[:, axis]) * _upm_ratio
            if a_ext > 0 and s_ext > 1.15 * a_ext * (1.0 if axis == 0 else 1.0):
                report("accent larger than Anton's", f"{name} {label}: {s_ext:.0f} vs {a_ext:.0f}")
for name in S.getGlyphOrder():
    g = sg[name]
    if g.numberOfContours == 0:
        continue
    pts, ends, flags = outline(sg, name)      # composites come out decomposed
    if not g.isComposite():
        nz = area(pts, ends, flags, pathops.FillType.WINDING)
        eo = area(pts, ends, flags, pathops.FillType.EVEN_ODD)
        if abs(nz - eo) > 30:
            report("overlapping/self-intersecting outline", f"{name} ({nz - eo:.0f} units²)")
        signed = sum(fontops.poly_area(p) for p in fontops.flatten(pts, ends, flags, steps=4))
        if signed > 0:
            report("contours run anticlockwise (TrueType wants clockwise)", name)
    # counters vs the source design
    single = g.isComposite() and len(g.components) == 1 and g.components[0].glyphName
    if name in A.getGlyphOrder() and A["glyf"][name].numberOfContours != 0:
        src = fontops.topology(*outline(ag, name))       # Anton (composites decomposed)
    elif single and single in A.getGlyphOrder():         # look-alike reusing an Anton letter
        src = fontops.topology(*outline(ag, single))
    elif name in rev and rev[name] in oc:
        src = fontops.topology(*outline(og, oc[rev[name]], 2.048))
    else:
        src = None
    if src is not None:
        now = fontops.topology(pts, ends, flags)
        if now[0] != src[0]:
            report("counter count differs from source design", f"{name}: {src[0]} -> {now[0]}")
        if now[2] != src[2]:      # accents/parts that merged or came apart
            report("separate pieces differ from source design", f"{name}: {src[2]} -> {now[2]}")
        if src[1] > 0 and now[1] < 0.3 * src[1]:
            report("counter nearly closed", f"{name}: area {src[1]:.0f} -> {now[1]:.0f}")
    if g.isComposite():
        continue
    # tiny contours and spikes
    start = 0
    for e in ends:
        cp = pts[start:e + 1]
        if len(cp) >= 3 and abs(fontops.poly_area(cp)) < 400:
            report("tiny contour", f"{name} contour at {cp[0].round().tolist()}")
        n = len(cp)
        for i in range(n):
            a, b, c = cp[i - 1], cp[i], cp[(i + 1) % n]
            v1, v2 = b - a, c - b
            l1, l2 = np.hypot(*v1), np.hypot(*v2)
            if l1 > 25 and l2 > 25 and (v1 @ v2) / (l1 * l2) < -0.985:
                report("spike (outline doubles back)", f"{name} at {b.round().tolist()}")
                break
        start = e + 1

# 2. clipping (all mapped glyphs) -------------------------------------------
wa, wd = S["OS/2"].usWinAscent, -S["OS/2"].usWinDescent
for cp, name in sc.items():
    g = sg[name]
    if g.numberOfContours == 0:
        continue
    g.recalcBounds(sg)
    if g.yMax > wa or g.yMin < wd:
        report("extends past Windows clip box", f"{chr(cp)} U+{cp:04X} ({g.yMin}..{g.yMax})")

# 3. ink far outside the advance -------------------------------------------
for cp, name in sc.items():
    g = sg[name]
    if g.numberOfContours == 0 or name in marks or S["hmtx"][name][0] == 0:
        continue
    adv, lsb = S["hmtx"][name]
    rsb = adv - g.xMax
    lim_l = min(-60, ref[cp]["xMin"] - 40) if cp in ref and "xMin" in ref[cp] else -60
    lim_r = min(-60, ref[cp]["adv"] - ref[cp]["xMax"] - 40) if cp in ref and "xMax" in ref[cp] else -60
    if lsb < lim_l or rsb < lim_r:
        report("ink overhangs advance (collision risk)", f"{chr(cp)} U+{cp:04X} lsb {lsb} rsb {rsb}")

# 4. kerning collisions ------------------------------------------------------
face = freetype.Face(sd_p)
face.set_char_size(512 * 64, 0, 72, 72)
gid = {g: i for i, g in enumerate(S.getGlyphOrder())}
_pc = {}


def prof(name):
    if name not in _pc:
        a, left, top = fontops.render(face, gid[name])
        p = {}
        for r in range(a.shape[0]):
            xs = np.where(a[r])[0]
            if xs.size:
                p[top - r] = ((left + xs[0]) * 4, (left + xs[-1] + 1) * 4)
        _pc[name] = p
    return _pc[name]


def gap(L, R, kern):
    pl, pr = prof(L), prof(R)
    common = set(pl) & set(pr)
    if not common:
        return None
    return min(S["hmtx"][L][0] + kern + pr[y][0] - pl[y][1] for y in common)


letters = {n for c, n in sc.items() if unicodedata.category(chr(c))[0] in "LNP"}
checked = 0
for lk in S["GPOS"].table.LookupList.Lookup:
    if lk.LookupType != 2:
        continue
    seen = set()  # pairs matched by an earlier subtable (first match wins)
    for st in lk.SubTable:
        if st.Format == 1:
            for L, ps in zip(st.Coverage.glyphs, st.PairSet):
                for pv in ps.PairValueRecord:
                    if (L, pv.SecondGlyph) in seen:
                        continue
                    seen.add((L, pv.SecondGlyph))
                    k = getattr(pv.Value1, "XAdvance", 0) if pv.Value1 else 0
                    if k < 0 and L in letters and pv.SecondGlyph in letters:
                        g_ = gap(L, pv.SecondGlyph, k)
                        checked += 1
                        if g_ is not None and g_ < -8:
                            report("kerning makes glyphs collide", f"{L}+{pv.SecondGlyph} kern {k} overlap {-g_}")
        else:
            c1 = {}
            for gname in st.Coverage.glyphs:
                c1.setdefault(st.ClassDef1.classDefs.get(gname, 0), []).append(gname)
            c2 = {}
            for gname, c in st.ClassDef2.classDefs.items():
                c2.setdefault(c, []).append(gname)
            for i, rec in enumerate(st.Class1Record):
                for j, r2 in enumerate(rec.Class2Record):
                    k = getattr(r2.Value1, "XAdvance", 0) if r2.Value1 else 0
                    if k >= 0 or i not in c1 or j not in c2:
                        continue
                    for L in [x for x in c1[i] if x in letters][:12]:
                        for R in [x for x in c2[j] if x in letters][:12]:
                            if (L, R) in seen:
                                continue
                            g_ = gap(L, R, k)
                            checked += 1
                            if g_ is not None and g_ < -8:
                                report("kerning makes glyphs collide", f"{L}+{R} kern {k} overlap {-g_}")
print(f"(kerning: {checked} pairs checked)")

# 5. Cyrillic shape sanity vs Oswald (catches wrong mappings / broken glyphs)
ofa = freetype.Face(osw_p)
ofa.set_char_size(256 * 64, 0, 72, 72)
sfa = freetype.Face(sd_p)
sfa.set_char_size(256 * 64, 0, 72, 72)
ogid = {g: i for i, g in enumerate(O.getGlyphOrder())}


def norm_raster(face, gi):
    a, _, _ = fontops.render(face, gi)
    if a.size == 0:
        return None
    from PIL import Image
    return np.array(Image.fromarray(a.astype(np.uint8) * 255).resize((48, 64))) > 127


ious = []
for cp in sorted(c for c in sc if 0x400 <= c < 0x530 and c in oc):
    a = norm_raster(sfa, gid[sc[cp]])
    b = norm_raster(ofa, ogid[oc[cp]])
    if a is None or b is None:
        continue
    iou = (a & b).sum() / max(1, (a | b).sum())
    ious.append((iou, chr(cp)))
    if iou < 0.55 and cp not in (0x04BE, 0x04BF):     # Ҿ ҿ: Oswald's stray diaeresis removed on purpose
        report("Cyrillic glyph differs strongly from Oswald's (check mapping)", f"{chr(cp)} U+{cp:04X} IoU {iou:.2f}")
print("(Cyrillic shape similarity to Oswald, lowest:", [(c, round(i, 2)) for i, c in sorted(ious)[:6]], ")")

# 6. combining accents on Cyrillic letters -------------------------------
hf = hb.Font(hb.Face(hb.Blob.from_file_path(sd_p)))
for base in "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёђѓєѕіјљњћќўџ":
    for mark in ("́", "̀", "̈"):
        buf = hb.Buffer()
        buf.add_str(base + mark)
        buf.guess_segment_properties()
        hb.shape(hf, buf, {})
        infos, pos = buf.glyph_infos, buf.glyph_positions
        if len(infos) != 2:
            continue
        bn, mn = S.getGlyphOrder()[infos[0].codepoint], S.getGlyphOrder()[infos[1].codepoint]
        bg_, mg_ = sg[bn], sg[mn]
        bg_.recalcBounds(sg)
        mg_.recalcBounds(sg)
        mx0 = mg_.xMin + pos[1].x_offset + pos[0].x_advance
        mx1 = mg_.xMax + pos[1].x_offset + pos[0].x_advance
        my0 = mg_.yMin + pos[1].y_offset
        cx = (mx0 + mx1) / 2
        if (pos[1].x_offset == 0 and pos[1].y_offset == 0) or not (bg_.xMin <= cx <= bg_.xMax) or my0 < bg_.yMax - 40:
            report("combining accent badly placed", f"{base}+U+{ord(mark):04X} offset ({pos[1].x_offset},{pos[1].y_offset})")

# 7. same-width alternates: turning on case-sensitive forms, the slashed zero
# or a language's forms must not change a glyph's width
gsub = S["GSUB"].table
for fr in gsub.FeatureList.FeatureRecord:
    if fr.FeatureTag not in ("case", "zero", "locl"):
        continue
    for li in fr.Feature.LookupListIndex:
        for st in gsub.LookupList.Lookup[li].SubTable:
            st = getattr(st, "ExtSubTable", st)
            for a, b in (getattr(st, "mapping", None) or {}).items():
                if S["hmtx"][a][0] != S["hmtx"][b][0]:
                    report("alternate changes the advance", f"{fr.FeatureTag}: {a} {S['hmtx'][a][0]} -> {b} {S['hmtx'][b][0]}")

# 8. flat edges 1-2 units off the baseline, x-height or cap height
_zones = [0] + [max(S["glyf"][n].getCoordinates(S["glyf"])[0], key=lambda q: q[1])[1] for n in ("dotlessi", "H")]
_gd = S["GDEF"].table.GlyphClassDef.classDefs
for n in S.getGlyphOrder():
    g = S["glyf"][n]
    if g.isComposite() or g.numberOfContours <= 0 or _gd.get(n) == 3:
        continue
    c = g.getCoordinates(S["glyf"])[0]
    fl = g.flags
    for i in range(len(c) - 1):
        if (fl[i] & 1) and (fl[i + 1] & 1) and c[i][1] == c[i + 1][1] and abs(c[i][0] - c[i + 1][0]) >= 20:
            if any(0 < abs(c[i][1] - z) <= 2 for z in _zones):
                report("flat edge just off the baseline / x-height / cap height", f"{n} at y={c[i][1]}")
                break

print()
if not issues:
    print("NO ISSUES FOUND")
for k, v in issues.items():
    print(f"## {k}: {len(v)}")
    for item in v[:40]:
        print("   ", item)
    if len(v) > 40:
        print(f"    ... and {len(v) - 40} more")
