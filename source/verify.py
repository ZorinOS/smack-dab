"""Verify Smack Dab against the reference font's metrics.

usage: python3 -I verify.py Impact.ttf SmackDab-Regular.ttf SmackDab-unhinted.ttf
"""
import random
import sys

import freetype
import uharfbuzz as hb
from fontTools.ttLib import TTFont

ref_p, sd_p, unh_p = sys.argv[1:4]
R, S, SU = TTFont(ref_p), TTFont(sd_p), TTFont(unh_p)
rc, sc = R.getBestCmap(), S.getBestCmap()
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)


# 1. advance widths for every shared character
shared = sorted(set(rc) & set(sc))
bad = [(f"U+{c:04X}", R["hmtx"][rc[c]][0], S["hmtx"][sc[c]][0]) for c in shared
       if R["hmtx"][rc[c]][0] != S["hmtx"][sc[c]][0]]
check(not bad, f"advance widths identical for all {len(shared)} shared characters {bad[:5]}")
missing = sorted(set(rc) - set(sc))
print(f"     reference characters not in Smack Dab: {len(missing)} "
      f"(Greek {sum(0x370 <= c < 0x400 for c in missing)}, Cyrillic {sum(0x400 <= c < 0x530 for c in missing)}, "
      f"other {sum(not (0x370 <= c < 0x530) for c in missing)})")

# 2. vertical / line metrics
pairs = [("UPM", R["head"].unitsPerEm, S["head"].unitsPerEm)]
for t, attrs in (("hhea", ("ascent", "descent", "lineGap")),
                 ("OS/2", ("sTypoAscender", "sTypoDescender", "sTypoLineGap", "usWinAscent", "usWinDescent",
                           "yStrikeoutPosition", "yStrikeoutSize")),
                 ("post", ("underlinePosition", "underlineThickness"))):
    for a in attrs:
        pairs.append((f"{t}.{a}", getattr(R[t], a), getattr(S[t], a)))
pairs.append(("USE_TYPO_METRICS", bool(R["OS/2"].fsSelection & 128), bool(S["OS/2"].fsSelection & 128)))
for n, a, b in pairs:
    check(a == b, f"{n}: {a} vs {b}")

# 3. hinting did not change metrics
check(all(S["hmtx"][g] == SU["hmtx"][g] for g in S.getGlyphOrder()), "hinted hmtx == unhinted hmtx")


# 4. HarfBuzz shaping: total advance of real strings
def hbfont(p):
    blob = hb.Blob.from_file_path(p)
    return hb.Font(hb.Face(blob))


HR, HS = hbfont(ref_p), hbfont(sd_p)


def width(f, text, feats):
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(f, buf, feats)
    return sum(p.x_advance for p in buf.glyph_positions)


chars = "".join(chr(c) for c in shared if c > 32 and chr(c).isprintable())
random.seed(7)
samples = [chars, "The quick brown fox jumps over the lazy dog. 0123456789", "1\u20442 3\u20444 10\u2044100",
           "Съешь же ещё этих мягких французских булок, да выпей чаю. ЁЖИК Ђорђе Љубљана Ґанок Їжак",
           "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich",
           "Płóćź śnieżnych łąk — „cytat” «guillemets» fi fl ffi ffl IJ ij",
           "Příliš žluťoučký kůň úpěl ďábelské ódy", "Árvíztűrő tükörfúrógép",
           "office affine waffle fjord (2025) 50% $9.99 #1 @home & co."]
samples += ["".join(random.choice(chars + "     ") for _ in range(60)) for _ in range(2000)]
samples += [chr(c) for c in shared]
nokern = {"kern": False}
diff = [t for t in samples if width(HR, t, nokern) != width(HS, t, nokern)]
check(not diff, f"HarfBuzz widths identical (kerning off) for {len(samples)} strings {diff[:3]}")


def positions(f, text, feats):
    """Where each glyph is drawn (pen position + offset) and the end position."""
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(f, buf, feats)
    out, x = [], 0
    for p in buf.glyph_positions:
        out.append(x + p.x_offset)
        x += p.x_advance
    return out + [x]


# kerning on (the default in browsers, LibreOffice, DirectWrite): every glyph
# lands where the reference puts it. Each string a+x+a+y+... covers every pair
# (a, x) and (x, a) of characters both fonts share.
lorem = ("Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae "
         "pellentesque sem placerat. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus.")
pair_strings = [c + c.join(chars) + c for c in chars]
diff_k = [t for t in samples[:9] + [lorem] + pair_strings if positions(HR, t, {}) != positions(HS, t, {})]
check(not diff_k, f"HarfBuzz glyph positions identical with kerning on (default), all {len(chars)}^2 "
      f"character pairs and sample text {[t[:12] for t in diff_k[:3]]}")

er, es = HR.get_font_extents("ltr"), HS.get_font_extents("ltr")
check((er.ascender, er.descender, er.line_gap) == (es.ascender, es.descender, es.line_gap),
      f"HarfBuzz line extents {er.ascender, er.descender, er.line_gap} vs {es.ascender, es.descender, es.line_gap}")

# 5. clipping: no shared character extends past the Windows clip box
#    (usWinAscent/usWinDescent) further than the reference's own glyph does
glyf, rglyf = S["glyf"], R["glyf"]
wa, wd = S["OS/2"].usWinAscent, -S["OS/2"].usWinDescent
worse = []
for c in shared:
    g, r = glyf[sc[c]], rglyf[rc[c]]
    if not g.numberOfContours or not hasattr(g, "yMax"):
        continue
    r_top = getattr(r, "yMax", 0) if r.numberOfContours else 0
    r_bot = getattr(r, "yMin", 0) if r.numberOfContours else 0
    if (g.yMax > wa and g.yMax > r_top) or (g.yMin < wd and g.yMin < r_bot):
        worse.append((chr(c), g.yMin, g.yMax, r_bot, r_top))
check(not worse, f"no shared character clips where the reference doesn't {worse[:8]}")

# 6. lsb == xMin
lsb_bad = [g for g in S.getGlyphOrder() if glyf[g].numberOfContours and S["hmtx"][g][1] != glyf[g].xMin]
check(not lsb_bad, f"hmtx lsb == glyph xMin {lsb_bad[:5]}")

# 7. grid-fitted (GDI-style) advances at 9-30 ppem vs the reference's device widths
face = freetype.Face(sd_p)
hd = R["hdmx"].hdmx if "hdmx" in R else {}
same = n = 0
for ppem in sorted(p for p in hd if p <= 30):
    face.set_pixel_sizes(0, ppem)
    for c in shared:
        face.load_char(chr(c), freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_TARGET_MONO)
        n += 1
        same += (face.glyph.advance.x // 64) == hd[ppem][rc[c]]
if n:
    print(f"INFO pixel-snapped advances equal to reference device widths at 9-30 ppem: {same}/{n} = {same / n:.1%}")

# 8. zones
def top(g):
    return glyf[g].yMax
print(f"     cap height {top(sc[ord('H')])}, x-height {top(sc[ord('x')])}, ascender {top(sc[ord('h')])}, "
      f"descender {glyf[sc[ord('p')]].yMin}")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
