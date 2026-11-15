"""Relax kerning pairs whose glyphs would touch or overlap.

Checks every class pair (up to 24 members per class) and every glyph pair in
the kern lookups; colliding pairs get a pair-specific exception in the leading
glyph-pair subtable so they keep at least MIN_GAP units of clearance.
usage: python3 -I fix_kern_collisions.py in.ttf out.ttf
"""
import os
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontops  # noqa: E402
import freetype  # noqa: E402
from fontTools.otlLib.builder import buildPairPosGlyphsSubtable, buildValue  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402

MIN_GAP = 12
src, dst = sys.argv[1:3]
f = TTFont(src)
cmap = f.getBestCmap()
order = f.getGlyphOrder()
gid = {g: i for i, g in enumerate(order)}
face = freetype.Face(src)
face.set_char_size(512 * 64, 0, 72, 72)
upx = f["head"].unitsPerEm / 512
hmtx = f["hmtx"]
_pc = {}


def prof(name):
    if name not in _pc:
        a, left, top = fontops.render(face, gid[name])
        p = {}
        for r in range(a.shape[0]):
            xs = np.where(a[r])[0]
            if xs.size:
                p[top - r] = ((left + xs[0]) * upx, (left + xs[-1] + 1) * upx)
        _pc[name] = p
    return _pc[name]


def gap(L, R, k):
    pl, pr = prof(L), prof(R)
    common = set(pl) & set(pr)
    if not common:
        return None
    return min(hmtx[L][0] + k + pr[y][0] - pl[y][1] for y in common)


chars = {n for c, n in cmap.items() if unicodedata.category(chr(c))[0] in "LNPS"}
fixes = {}
for lk in f["GPOS"].table.LookupList.Lookup:
    if lk.LookupType != 2:
        continue
    glyph_sub = lk.SubTable[0] if lk.SubTable[0].Format == 1 else None
    existing = {}
    if glyph_sub is not None:
        for L, ps in zip(glyph_sub.Coverage.glyphs, glyph_sub.PairSet):
            for pv in ps.PairValueRecord:
                existing[(L, pv.SecondGlyph)] = getattr(pv.Value1, "XAdvance", 0) if pv.Value1 else 0
    for st in lk.SubTable:
        if st.Format == 1:
            for L, ps in zip(st.Coverage.glyphs, st.PairSet):
                for pv in ps.PairValueRecord:
                    k = getattr(pv.Value1, "XAdvance", 0) if pv.Value1 else 0
                    if k < 0 and L in chars and pv.SecondGlyph in chars:
                        g_ = gap(L, pv.SecondGlyph, k)
                        if g_ is not None and g_ < MIN_GAP:
                            fixes[(L, pv.SecondGlyph)] = k + (MIN_GAP - g_)
        else:
            c1, c2 = {}, {}
            for g in st.Coverage.glyphs:
                c1.setdefault(st.ClassDef1.classDefs.get(g, 0), []).append(g)
            for g, c in st.ClassDef2.classDefs.items():
                c2.setdefault(c, []).append(g)
            for i, rec in enumerate(st.Class1Record):
                for j, r2 in enumerate(rec.Class2Record):
                    k = getattr(r2.Value1, "XAdvance", 0) if r2.Value1 else 0
                    if k >= 0 or i not in c1 or j not in c2:
                        continue
                    for L in [x for x in c1[i] if x in chars][:24]:
                        for R in [x for x in c2[j] if x in chars][:24]:
                            if (L, R) in existing:
                                continue
                            g_ = gap(L, R, k)
                            if g_ is not None and g_ < MIN_GAP:
                                fixes[(L, R)] = min(0, int(round(k + (MIN_GAP - g_))))
    if fixes:
        pairs = {k: (buildValue({"XAdvance": v}), None) for k, v in existing.items()}
        for k, v in fixes.items():
            pairs[k] = (buildValue({"XAdvance": int(v)}), None)
        sub = buildPairPosGlyphsSubtable(pairs, f.getReverseGlyphMap())
        if glyph_sub is not None:
            lk.SubTable[0] = sub
        else:
            lk.SubTable.insert(0, sub)
        lk.SubTableCount = len(lk.SubTable)
print(f"kerning collisions relaxed: {len(fixes)}", sorted(fixes)[:12])
f.save(dst)
