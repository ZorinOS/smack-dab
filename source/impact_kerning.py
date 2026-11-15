"""Kerning identical to the reference's.

The reference kerns with a legacy 'kern' table (and no GPOS), which HarfBuzz
(LibreOffice, browsers), DirectWrite and GDI apply by default. Anton's own
kerning, re-fitted to Smack Dab's shapes, kerned different pairs by different
amounts, so kerned text (LibreOffice, browsers) shifted against the
reference. This step:

* kerns every pair of characters the reference kerns, by exactly the
  reference's amount (from impact_metrics.json, by character), both in the
  GPOS 'kern' feature and in a legacy 'kern' table for apps that read only
  that;
* keeps Anton's kerning only for pairs involving a character the reference
  doesn't have (extended Latin, most Cyrillic, small caps ...), where no
  layout can shift: two characters the reference shares are never kerned
  unless the reference kerns them.

usage: impact_kerning.py in.ttf impact_metrics.json out.ttf
"""
import copy
import json
import sys

from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables._k_e_r_n import KernTable_format_0

src, metrics, dst = sys.argv[1:4]
font = TTFont(src)
ref = json.load(open(metrics))
cmap = font.getBestCmap()
gid = font.getGlyphID

shared = {cmap[int(k, 16)] for k in ref["chars"] if int(k, 16) in cmap}   # glyphs both fonts have
pairs = {}
missing = 0
for a, b, v in ref["kern"]:
    ca, cb = int(a, 16), int(b, 16)
    if ca in cmap and cb in cmap:
        pairs[(cmap[ca], cmap[cb])] = v
    else:
        missing += 1

gpos = font["GPOS"].table
kern_lookups = sorted({i for fr in gpos.FeatureList.FeatureRecord if fr.FeatureTag == "kern"
                       for i in fr.Feature.LookupListIndex})
if not kern_lookups:
    raise SystemExit("no kern lookup to extend")


def restrict(st):
    """Anton's pair subtable, limited to pairs with a glyph outside `shared`.
    Returns a list of subtables (a class subtable is split by first glyph)."""
    if st.Format == 1:
        cov, sets = [], []
        for g, ps in zip(st.Coverage.glyphs, st.PairSet):
            recs = [r for r in ps.PairValueRecord if not (g in shared and r.SecondGlyph in shared)]
            if recs:
                ps.PairValueRecord = recs
                ps.PairValueCount = len(recs)
                cov.append(g)
                sets.append(ps)
        if not cov:
            return []
        st.Coverage.glyphs = cov
        st.PairSet = sets
        st.PairSetCount = len(sets)
        return [st]
    out = []
    first_new = [g for g in st.Coverage.glyphs if g not in shared]
    first_shared = [g for g in st.Coverage.glyphs if g in shared]
    if first_new:
        a = copy.deepcopy(st)
        a.Coverage.glyphs = first_new
        out.append(a)
    if first_shared:
        b = copy.deepcopy(st)
        b.Coverage.glyphs = first_shared
        # second glyphs the reference shares fall to class 0 (no kerning)
        b.ClassDef2.classDefs = {g: c for g, c in b.ClassDef2.classDefs.items() if g not in shared}
        for rec in b.Class1Record:             # and class 0 kerns nothing
            r0 = rec.Class2Record[0]
            for attr in ("Value1", "Value2"):
                v = getattr(r0, attr, None)
                if v is not None:
                    for f in ("XPlacement", "YPlacement", "XAdvance", "YAdvance"):
                        if hasattr(v, f):
                            setattr(v, f, 0)
        out.append(b)
    return out


def value(v):
    vr = ot.ValueRecord()
    vr.XAdvance = v
    return vr


# 1. the reference's pairs, first in the kern lookup
first = {}
for (a, b), v in pairs.items():
    first.setdefault(a, []).append((b, v))
st = ot.PairPos()
st.Format = 1
st.ValueFormat1, st.ValueFormat2 = 4, 0          # XAdvance only
st.Coverage = ot.Coverage()
st.Coverage.glyphs = sorted(first, key=gid)
st.PairSet = []
for a in st.Coverage.glyphs:
    ps = ot.PairSet()
    ps.PairValueRecord = []
    for b, v in sorted(first[a], key=lambda t: gid(t[0])):
        r = ot.PairValueRecord()
        r.SecondGlyph, r.Value1 = b, value(v)
        ps.PairValueRecord.append(r)
    ps.PairValueCount = len(ps.PairValueRecord)
    st.PairSet.append(ps)
st.PairSetCount = len(st.PairSet)

# 2. Anton's pairs, restricted; the reference's subtable goes first
n_before = n_after = 0
for i, li in enumerate(kern_lookups):
    lk = gpos.LookupList.Lookup[li]
    subs = []
    for s in lk.SubTable:
        s = getattr(s, "ExtSubTable", s)
        n_before += 1
        subs += restrict(s)
    if i == 0:
        subs = [st] + subs
    if lk.LookupType == 9:          # rewrap as extension subtables
        wrapped = []
        for s in subs:
            e = ot.ExtensionPos()
            e.Format, e.ExtensionLookupType, e.ExtSubTable = 1, 2, s
            wrapped.append(e)
        subs = wrapped
    lk.SubTable = subs
    lk.SubTableCount = len(subs)
    n_after += len(subs)

# 3. legacy 'kern' table with the same pairs, for apps that read only that
kt = newTable("kern")
kt.version = 0
sub = KernTable_format_0()
sub.version, sub.coverage, sub.format, sub.tupleIndex = 0, 1, 0, None
sub.kernTable = dict(pairs)
kt.kernTables = [sub]
font["kern"] = kt

font.save(dst)
print(f"reference kerning: {len(pairs)} pairs ({missing} with characters Smack Dab lacks); "
      f"Anton's kerning kept only for characters the reference lacks "
      f"({n_before} -> {n_after - 1} subtables)")
