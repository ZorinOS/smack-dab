"""Extract *numeric metrics only* from the reference font into a JSON file.

What is read: per-character advance widths and ink extents (horizontal
bounding box), the font-wide vertical/line metrics needed for layout
compatibility, the kerning pairs (by character, so text kerns identically),
and per-character stroke weights (vertical-stem and horizontal-stroke
thickness, as numbers). No outlines, hinting code or names are extracted, and
the Smack Dab build reads only this JSON.

usage: python3 -I measure_impact.py Impact.ttf impact_metrics.json
"""
import json
import sys

import numpy as np
import freetype

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont


def main(src, dst):
    f = TTFont(src)
    gs = f.getGlyphSet()
    cmap = f.getBestCmap()
    chars = {}
    for cp, gname in sorted(cmap.items()):
        pen = BoundsPen(gs)
        gs[gname].draw(pen)
        adv = f["hmtx"][gname][0]
        entry = {"adv": adv}
        if pen.bounds:
            entry["xMin"], _, entry["xMax"], _ = (round(v) for v in pen.bounds)
        chars[f"{cp:04X}"] = entry

    def ink(ch):
        pen = BoundsPen(gs)
        gs[cmap[ord(ch)]].draw(pen)
        return pen.bounds

    # kerning: the legacy 'kern' table's pairs, by character
    by_glyph = {}
    for cp, gname in cmap.items():
        by_glyph.setdefault(gname, []).append(cp)
    kern = []
    if "kern" in f:
        for sub in f["kern"].kernTables:
            if getattr(sub, "format", 0) != 0 or not (sub.coverage & 1):
                continue
            for (a, b), v in sorted(sub.kernTable.items()):
                for ca in by_glyph.get(a, []):
                    for cb in by_glyph.get(b, []):
                        kern.append([f"{ca:04X}", f"{cb:04X}", int(v)])

    # stroke weights, measured on an unhinted rendering (font units): the
    # typical vertical-stem thickness (horizontal ink runs across the middle
    # rows) and horizontal-stroke thickness (vertical ink runs down the middle
    # columns, stems excluded)
    ppem = 512
    face = freetype.Face(src)
    face.set_char_size(ppem * 64, 0, 72, 72)
    u = head_upm = f["head"].unitsPerEm
    u = head_upm / ppem
    gid = {g: i for i, g in enumerate(f.getGlyphOrder())}

    def runs(v):
        d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
        return list(np.where(d == -1)[0] - np.where(d == 1)[0])
    for cp, gname in sorted(cmap.items()):
        if chr(cp).isspace():
            continue
        face.load_glyph(gid[gname], freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_RENDER
                        | freetype.FT_LOAD_NO_BITMAP)
        bm = face.glyph.bitmap
        if not bm.rows or not bm.width:
            continue
        a = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.width) > 127
        h, w = a.shape
        hr = [r for fr in np.linspace(0.18, 0.82, 13) for r in runs(a[int(h * fr)])]
        hr = [r for r in hr if r < 0.7 * w] or hr
        vr = [r for fr in np.linspace(0.3, 0.7, 9) for r in runs(a[:, int(w * fr)])]
        vr = [r for r in vr if r < 0.6 * h]
        e = chars[f"{cp:04X}"]
        if hr:
            e["stem"] = round(float(np.median(hr)) * u)
        if vr:
            e["hbar"] = round(float(np.median(vr)) * u)
            e["hbar_lq"] = round(float(np.percentile(vr, 25)) * u)

    hh, os2, post, head = f["hhea"], f["OS/2"], f["post"], f["head"]
    out = {
        "unitsPerEm": head.unitsPerEm,
        "hhea": {"ascent": hh.ascent, "descent": hh.descent, "lineGap": hh.lineGap},
        "os2": {
            "sTypoAscender": os2.sTypoAscender,
            "sTypoDescender": os2.sTypoDescender,
            "sTypoLineGap": os2.sTypoLineGap,
            "usWinAscent": os2.usWinAscent,
            "usWinDescent": os2.usWinDescent,
            "useTypoMetrics": bool(os2.fsSelection & (1 << 7)),
            "yStrikeoutSize": os2.yStrikeoutSize,
            "yStrikeoutPosition": os2.yStrikeoutPosition,
            "ySubscriptXSize": os2.ySubscriptXSize,
            "ySubscriptYSize": os2.ySubscriptYSize,
            "ySubscriptYOffset": os2.ySubscriptYOffset,
            "ySuperscriptXSize": os2.ySuperscriptXSize,
            "ySuperscriptYSize": os2.ySuperscriptYSize,
            "ySuperscriptYOffset": os2.ySuperscriptYOffset,
            "usWidthClass": os2.usWidthClass,
            "usWeightClass": os2.usWeightClass,
        },
        "post": {"underlinePosition": post.underlinePosition,
                 "underlineThickness": post.underlineThickness},
        # reference heights (measured from ink extents)
        "zones": {
            "capHeight": round(ink("H")[3]),
            "xHeight": round(ink("x")[3]),
            "ascender": round(ink("h")[3]),
            "descender": round(ink("p")[1]),
        },
        "notdefAdvance": f["hmtx"][f.getGlyphOrder()[0]][0],
        "chars": chars,
        "kern": kern,
    }
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dst}: {len(chars)} characters, {len(kern)} kerning pairs")


if __name__ == "__main__":
    main(*sys.argv[1:3])
