#!/usr/bin/env python3
"""Stage 2: extend Smack Dab with Cyrillic (and a few extras) from Oswald Bold.

* Letters that look identical to Latin ones (А В Е К М Н О Р С Т Х а е о р с
  у х ...) reuse Smack Dab's own Anton-derived glyphs as components.
* Precomposed letters (Й Ё Ў Ӂ ...) are built from their base letter plus
  Smack Dab's (Anton's) combining marks, placed with Anton's mark anchors.
* All other letters take Oswald Bold outlines and re-shape them to Anton's
  proportions and weight: vertical zones mapped onto Smack Dab's cap height,
  x-height, ascender and descender; stems and bars thickened with FreeType-style
  emboldening to Smack Dab's measured stroke weights; widths fitted to the
  reference advance/ink widths.

usage: python3 -I add_cyrillic.py SmackDab-unhinted.ttf Oswald-Bold.ttf impact_metrics.json out.ttf
"""
import collections
import json
import os
import statistics
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontops  # noqa: E402
from fontTools.otlLib.builder import buildPairPosGlyphsSubtable, buildValue  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402
from fontTools.ttLib.tables import otTables as ot  # noqa: E402
from fontTools.ttLib.tables import ttProgram  # noqa: E402
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent, GlyphCoordinates  # noqa: E402

SD_PATH, OSW_PATH, METRICS, OUT = sys.argv[1:5]
SD, O = TTFont(SD_PATH), TTFont(OSW_PATH)
ref = json.load(open(METRICS))
RCH = {int(k, 16): v for k, v in ref["chars"].items()}
glyf, hmtx = SD["glyf"], SD["hmtx"]
sd_cmap, o_cmap = SD.getBestCmap(), O.getBestCmap()
o_glyf = O["glyf"]
PRE = SD["head"].unitsPerEm / O["head"].unitsPerEm
WIN_A, WIN_D = SD["OS/2"].usWinAscent, SD["OS/2"].usWinDescent
TOP_LIMIT, BOT_LIMIT = WIN_A - 16, -WIN_D + 16


# ------------------------------------------------------------ measurements
def measure(path, cmap, upm):
    face, upx = fontops.open_face(path, upm, ppem=upm)
    gid = {g: i for i, g in enumerate(TTFont(path).getGlyphOrder())}

    def r(ch):
        return fontops.render(face, gid[cmap[ord(ch)]])
    m = {}
    a, _, top = r("H")
    m["H_stem"] = np.mean([e - s for s, e in fontops.runs(a[a.shape[0] // 4])])
    m["cap"] = top
    a, _, top = r("n")
    m["n_stem"] = np.mean([e - s for s, e in fontops.runs(a[int(a.shape[0] * 0.7)])])
    a, _, top = r("x")
    m["xflat"] = top
    a, _, top = r("E")
    m["E_arm"] = np.mean([e - s for s, e in fontops.runs(a[:, a.shape[1] * 3 // 4])])
    a, _, top = r("o")
    m["o_tb"] = np.mean([e - s for s, e in fontops.runs(a[:, a.shape[1] // 2])])
    m["xover"] = top
    _, _, m["asc"] = r("h")
    a, _, top = r("p")
    m["desc"] = top - a.shape[0]
    return m


ms = measure(SD_PATH, sd_cmap, SD["head"].unitsPerEm)
mo = {k: v * PRE for k, v in measure(OSW_PATH, o_cmap, O["head"].unitsPerEm).items()}
print("Smack Dab:", {k: round(float(v)) for k, v in ms.items()})
print("Oswald (x%.3f):" % PRE, {k: round(float(v)) for k, v in mo.items()})

# capitals: cap height and bar weight both land on Smack Dab's
v_uc = (ms["cap"] - ms["E_arm"]) / (mo["cap"] - mo["E_arm"])
ystr_uc = ms["E_arm"] - v_uc * mo["E_arm"]
# lowercase body: x-height and round-stroke weight
v_lc = (ms["xflat"] - ms["o_tb"]) / (mo["xflat"] - mo["o_tb"])
ystr_lc = ms["o_tb"] - v_lc * mo["o_tb"]
if ystr_lc < 0:
    v_lc, ystr_lc = ms["xflat"] / mo["xflat"], 0.0
# lowercase ascender zone: compress to Anton's short ascenders, re-thicken
T_band = 0.85 * ms["o_tb"]
y_xo = v_lc * mo["xover"] + ystr_lc
s_band = max(0.2, (ms["asc"] - y_xo - T_band) / (mo["asc"] - mo["xover"] - mo["o_tb"]))
ystr_band = T_band - s_band * mo["o_tb"]
# lowercase descender zone (straight tails: like p/q)
v_d = ms["desc"] / mo["desc"]
ystr_d = max(0.0, ms["o_tb"] - v_d * mo["o_tb"])
# hooked descenders go deeper in Anton (cf. its eng, about -405)
HOOK_DEPTH = 400.0
v_dh = min(v_lc, HOOK_DEPTH / -mo["desc"])
ystr_dh = max(0.0, ms["o_tb"] - v_dh * mo["o_tb"])
print(f"v_uc={v_uc:.3f} ystr_uc={ystr_uc:.0f} | v_lc={v_lc:.3f} ystr_lc={ystr_lc:.0f} | "
      f"band s={s_band:.3f} ystr={ystr_band:.0f} | v_d={v_d:.3f} ystr_d={ystr_d:.0f}")
S_UC, S_LC = float(ms["H_stem"]), float(ms["n_stem"])


def vmap(ys, cls, has_asc, hook=False):
    ys = np.asarray(ys, float)
    if cls == "uc":
        return v_uc * ys, np.full(len(ys), ystr_uc)
    vd, sd = (v_dh, ystr_dh) if hook else (v_d, ystr_d)
    Y = np.where(ys < 0, vd * ys, v_lc * ys)
    st = np.where(ys < 0, sd, ystr_lc)
    if has_asc:
        top_body = v_lc * mo["xover"]
        band = ys > mo["xover"]
        Y = np.where(band, top_body + s_band * (ys - mo["xover"]), Y)
        st = np.where(band, ystr_band, st)
    return Y, st


# ------------------------------------------------------------- what to add
CYR = lambda c: 0x400 <= c < 0x530  # noqa: E731
EXTRA = {0x20AE, 0x20B4, 0x20B8, 0x212A, 0x212B}
todo = sorted(c for c in o_cmap if c not in sd_cmap and (CYR(c) or c in EXTRA))
print("characters to add:", len(todo))

HOMO = {0x0410: "A", 0x0412: "B", 0x0415: "E", 0x041A: "K", 0x041C: "M", 0x041D: "H",
        0x041E: "O", 0x0420: "P", 0x0421: "C", 0x0422: "T", 0x0425: "X", 0x0405: "S",
        0x0406: "I", 0x0408: "J", 0x04C0: "I", 0x051A: "Q", 0x051C: "W", 0x04AE: "Y",
        0x0430: "a", 0x0435: "e", 0x043E: "o", 0x0440: "p", 0x0441: "c", 0x0443: "y",
        0x0445: "x", 0x0455: "s", 0x0456: "i", 0x0458: "j", 0x04BB: "h", 0x051B: "q",
        0x051D: "w", 0x04CF: "l", 0x212A: "K", 0x212B: "Å", 0x04D4: "Æ",
        0x04D5: "æ", 0x04D8: "Ə", 0x04D9: "ə", 0x04E8: "Ɵ", 0x04E9: "ɵ"}

# kerning / anchor look-alikes: (left side like, right side like)
ALIKE = {
    "Б": ("E", "B"), "Г": ("E", "F"), "Ґ": ("E", "F"), "Д": (None, None), "Ж": ("X", "K"),
    "З": (None, "B"), "И": ("H", "H"), "Л": (None, "H"), "П": ("H", "H"), "У": ("Y", "Y"),
    "Ф": ("O", "O"), "Ц": ("H", None), "Ч": ("H", "H"), "Ш": ("H", "H"), "Щ": ("H", None),
    "Ъ": ("T", "B"), "Ы": ("H", "H"), "Ь": ("H", "B"), "Э": (None, "O"), "Ю": ("H", "O"),
    "Я": ("O", "H"), "Є": ("O", "C"), "Ђ": ("T", None), "Ћ": ("T", "H"), "Љ": (None, "B"),
    "Њ": ("H", "B"), "Џ": ("H", "H"),
    "б": ("b", "o"), "в": ("n", "o"), "г": ("n", "r"), "ґ": ("n", "r"), "ж": ("x", "k"),
    "з": (None, "o"), "и": ("n", "n"), "к": ("n", "k"), "л": (None, "n"), "м": (None, "n"),
    "н": ("n", "n"), "п": ("n", "n"), "ф": ("o", "o"), "ц": ("n", None), "ч": ("n", "n"),
    "ш": ("n", "n"), "щ": ("n", None), "ъ": (None, "o"), "ы": ("n", "n"), "ь": ("n", "o"),
    "э": (None, "o"), "ю": ("n", "o"), "я": ("o", "n"), "є": ("o", "c"), "ђ": ("n", None),
    "ћ": ("n", "n"), "љ": (None, "o"), "њ": ("n", "o"), "џ": ("n", "n"),
}


def alike_of(c):
    ch = chr(c)
    if ch in ALIKE:
        return ALIKE[ch]
    name = unicodedata.name(ch, "")
    if " WITH " in name:                      # e.g. ZE WITH DESCENDER -> ZE
        try:
            b = unicodedata.lookup(name.split(" WITH ")[0])
            if b in ALIKE:
                return ALIKE[b]
            if ord(b) in HOMO:
                return (HOMO[ord(b)], HOMO[ord(b)])
        except KeyError:
            pass
    nfd = unicodedata.normalize("NFD", ch)
    if len(nfd) > 1 and nfd[0] != ch:
        b = nfd[0]
        if b in ALIKE:
            return ALIKE[b]
        if ord(b) in HOMO:
            return (HOMO[ord(b)], HOMO[ord(b)])
    return (None, None)


def vcls(c):
    return "lc" if unicodedata.category(chr(c)) == "Ll" else "uc"


def kind(c):
    return "alnum" if unicodedata.category(chr(c))[0] in "LN" else "sym"


new_names = {}            # cp -> new glyph name
adv_of = {}               # new glyph name -> advance
anchor_alike = {}         # new glyph -> Latin glyph used for mark anchors


def gname(c):
    n = f"uni{c:04X}"
    assert n not in SD.getGlyphOrder()
    return n


def add_glyph(name, glyph, adv, c):
    glyf[name] = glyph                # appends to the shared glyph order
    glyph.recalcBounds(glyf)
    hmtx[name] = (int(round(adv)), glyph.xMin if glyph.numberOfContours else 0)
    adv_of[name] = int(round(adv))
    for t in SD["cmap"].tables:
        if t.isUnicode():
            t.cmap[c] = name
    new_names[c] = name


def composite(parts):
    g = Glyph()
    g.numberOfContours = -1
    g.components = []
    for n, x, y in parts:
        comp = GlyphComponent()
        comp.glyphName, comp.x, comp.y, comp.flags = n, int(round(x)), int(round(y)), 0x4
        g.components.append(comp)
    return g


def simple(coords, flags, ends):
    g = Glyph()
    g.numberOfContours = len(ends)
    g.coordinates = GlyphCoordinates([(int(round(x)), int(round(y))) for x, y in coords])
    g.flags = list(flags)
    g.endPtsOfContours = list(ends)
    g.program = ttProgram.Program()
    g.program.fromBytecode(b"")
    return g


def bounds(name):
    g = glyf[name]
    g.recalcBounds(glyf)
    return g.xMin, g.yMin, g.xMax, g.yMax


# ------------------------------------------------- 1. Oswald-derived outlines
o_face, o_upx = fontops.open_face(OSW_PATH, O["head"].unitsPerEm)
o_gid = {g: i for i, g in enumerate(O.getGlyphOrder())}


def nfd_plan(c):
    """(base_char, [combining marks]) if c decomposes onto available parts."""
    nfd = unicodedata.normalize("NFD", chr(c))
    if len(nfd) < 2:
        return None
    base, marks = nfd[0], nfd[1:]
    if not all(ord(m) in sd_cmap for m in marks):
        return None
    return base, marks


homo_plan, comp_plan, outline_plan = {}, {}, []
for c in todo:
    if c in HOMO and ord(HOMO[c]) in sd_cmap:
        homo_plan[c] = sd_cmap[ord(HOMO[c])]
        continue
    p = nfd_plan(c)
    if p:
        base, marks = p
        lat = HOMO.get(ord(base))
        if lat:
            pre = unicodedata.normalize("NFC", lat + marks)
            if len(pre) == 1 and ord(pre) in sd_cmap:
                homo_plan[c] = sd_cmap[ord(pre)]
                continue
        if ord(base) in sd_cmap or ord(base) in todo:
            comp_plan[c] = p
            continue
    outline_plan.append(c)
print(f"look-alikes {len(homo_plan)}, composed {len(comp_plan)}, from Oswald outlines {len(outline_plan)}")

# stroke-weight targets: Oswald's relative weights, pulled toward Anton's much
# more even colour (Anton's letters deviate from stem weight ~40% as much)
def raw_outline(name):
    coords, ends, flags = o_glyf[name].getCoordinates(o_glyf)
    return np.array(coords, float) * PRE, list(ends), [f & 1 for f in flags]


def sd_outline(name):
    coords, ends, flags = glyf[name].getCoordinates(glyf)
    return np.array(coords, float), list(ends), list(flags)


# Stroke-weight targets from Latin look-alikes.  "t" (transfer): convert the
# letter's Oswald weight the same way its Latin analogs were converted from
# Oswald to Smack Dab (so e.g. round letters follow o, stem letters follow n/H).
# "m" (match): diagonal letters take the weight of the Latin letters they
# resemble directly (Oswald thins these far more than Anton does).
ANALOG = {
    "Б": ("t", "BE"), "Г": ("t", "FE"), "Ґ": ("t", "F"), "Д": ("t", "AH"), "Ж": ("m", "KX", 0.95),
    "З": ("t", "BS"), "И": ("t", "N"), "Л": ("t", "AH"), "П": ("t", "H"), "У": ("m", "Y", 1.0),
    "Ф": ("t", "OI"), "Ц": ("t", "HU"), "Ч": ("t", "HU"), "Ш": ("t", "H"), "Щ": ("t", "H"),
    "Ъ": ("t", "BT"), "Ы": ("t", "BI"), "Ь": ("t", "B"), "Э": ("t", "CO"), "Ю": ("t", "OH"),
    "Я": ("t", "R"), "Ђ": ("t", "Th"), "Ћ": ("t", "Th"), "Љ": ("t", "AB"), "Њ": ("t", "HB"),
    "Џ": ("t", "H"), "Є": ("t", "C"), "Ѣ": ("t", "BT"), "Ѳ": ("t", "O"), "Ѵ": ("m", "V", 1.0),
    "Ѧ": ("t", "A"), "Ѫ": ("m", "X", 0.9),
    "б": ("t", "ob"), "в": ("t", "bo"), "г": ("t", "n"), "ґ": ("t", "n"), "д": ("t", "na"),
    "ж": ("m", "kx", 0.95), "з": ("t", "so"), "и": ("t", "nu"), "к": ("m", "k", 1.0), "л": ("t", "n"),
    "м": ("t", "mn"), "н": ("t", "n"), "п": ("t", "n"), "т": ("t", "n"), "ф": ("t", "on"),
    "ц": ("t", "nu"), "ч": ("t", "n"), "ш": ("t", "nm"), "щ": ("t", "nm"), "ъ": ("t", "bn"),
    "ы": ("t", "bn"), "ь": ("t", "b"), "э": ("t", "co"), "ю": ("t", "on"), "я": ("t", "nb"),
    "ђ": ("t", "h"), "ћ": ("t", "h"), "љ": ("t", "bn"), "њ": ("t", "bn"), "џ": ("t", "n"),
    "є": ("t", "ce"), "ѣ": ("t", "b"), "ѳ": ("t", "o"), "ѵ": ("m", "v", 1.0), "ѧ": ("t", "a"),
    "ѫ": ("m", "x", 0.9), "ү": ("m", "vy", 1.0),
    "\u20ae": ("t", "T"), "\u20b4": ("t", "S"), "\u20b8": ("t", "T"),
}
DIAGONAL_LATIN = set("KXYVWkxyvw")


def analog_of(c):
    ch = chr(c)
    if ch in ANALOG:
        return ANALOG[ch]
    name = unicodedata.name(ch, "")
    for b in ([unicodedata.lookup(name.split(" WITH ")[0])] if " WITH " in name else []) + \
             [unicodedata.normalize("NFD", ch)[0]]:
        try:
            if b in ANALOG:
                return ANALOG[b]
            lat = HOMO.get(ord(b))
            if lat and len(lat) == 1:
                return ("m", lat, 1.0) if lat in DIAGONAL_LATIN else ("t", lat)
        except (KeyError, TypeError):
            continue
    return ("t", "HO") if vcls(c) == "uc" else ("t", "no")


_swc = {}


def sw_sd(ch):
    if ("S", ch) not in _swc:
        _swc[("S", ch)] = fontops.outline_metrics(*sd_outline(sd_cmap[ord(ch)]))["sw_v"]
    return _swc[("S", ch)]


def sw_osw(ch):
    if ("O", ch) not in _swc:
        _swc[("O", ch)] = fontops.outline_metrics(*raw_outline(o_cmap[ord(ch)]))["sw_v"]
    return _swc[("O", ch)]


OPEN_FLOOR = 0.85
_opc = {}


def open_sd(ch):
    if ("S", ch) not in _opc:
        _opc[("S", ch)] = fontops.openness(*sd_outline(sd_cmap[ord(ch)]))
    return _opc[("S", ch)]


def open_osw(ch):
    if ("O", ch) not in _opc:
        _opc[("O", ch)] = fontops.openness(*raw_outline(o_cmap[ord(ch)]))
    return _opc[("O", ch)]


def openness_floor(c, src_open):
    """Minimum openness: OPEN_FLOOR x what the Latin look-alikes imply, i.e.
    the letter may not clog up more than its Latin analogs did."""
    lat = analog_of(c)[1]
    ratios = [open_sd(x) / open_osw(x) for x in lat if open_osw(x) > 0.005]
    if not ratios or src_open < 0.01:
        return None
    return OPEN_FLOOR * src_open * float(np.mean(ratios))


def weight_target(c, src_sw):
    a = analog_of(c)
    if a[0] == "m":
        return a[2] * float(np.mean([sw_sd(x) for x in a[1]]))
    return src_sw * float(np.mean([sw_sd(x) / sw_osw(x) for x in a[1]]))


def fit_oswald(c, target_w, adv, fixed_h=None):
    on = o_cmap[c]
    coords, ends, flags = raw_outline(on)
    cls, k = vcls(c), kind(c)
    has_asc = cls == "lc" and coords[:, 1].max() > mo["xover"] + 60
    hook = cls == "lc" and any(not f and y < 0.3 * mo["desc"] for (x, y), f in zip(coords, flags))
    Y, ystr = vmap(coords[:, 1], cls, has_asc, hook)
    stem, gap, sw = fontops.stroke_stats(o_face, o_gid[on], o_upx)
    stem = (stem or 0) * PRE
    WA = float(np.ptp(coords[:, 0]))

    def apply(h, d):     # tilt-free offset; loops are resolved in the final cleanup
        return fontops.embolden_xy_v2(np.column_stack([coords[:, 0] * h, Y]), ends, d, ystr)

    src = fontops.outline_metrics(coords, ends, flags)
    tstem = S_UC if cls == "uc" else S_LC
    if src["sw_v"] is None or stem == 0:
        h = fixed_h or (min(max(target_w / WA, 0.55), 1.45) if target_w else 1.0)
        return apply(h, 0.0), flags, ends, h, 0.0
    sw_target = weight_target(c, src["sw_v"])
    floor = openness_floor(c, fontops.openness(coords, ends, flags))

    def open_ok(p):
        return floor is None or fontops.openness(p, ends, flags) >= floor
    base = fontops.outline_metrics(apply(1.0, 0.0), ends, flags)
    hmin, hmax = (0.55, 1.45) if k == "alnum" else (0.45, 1.3)
    if fixed_h is not None:
        h0 = fixed_h
    else:
        h0 = min(max((target_w - tstem) / max(1.0, WA - stem), hmin), hmax)
    d0 = tstem - h0 * stem
    h, d = fontops.solve_weight_flex(apply, ends, flags, WA, target_w, adv, sw_target, h0, d0,
                                     hmin, hmax, base, extra_ok=open_ok)
    return apply(h, d), flags, ends, h, d


fit_log = {}
for c in outline_plan:
    if c in RCH and "xMin" in RCH[c]:
        r = RCH[c]
        pts, flags, ends, h, d = fit_oswald(c, r["xMax"] - r["xMin"], r["adv"])
        pts[:, 0] += (r["xMin"] + r["xMax"]) / 2 - (pts[:, 0].max() + pts[:, 0].min()) / 2
        fit_log[c] = (h, d, True)
        add_glyph(gname(c), simple(pts, flags, ends), r["adv"], c)
    else:
        fit_log[c] = None

# generic transform for characters the reference doesn't have
H_GEN = {cl: statistics.median([fit_log[c][0] for c in fit_log if fit_log[c] and vcls(c) == cl] or [1.0])
         for cl in ("uc", "lc")}
sbr = []
for c, v in fit_log.items():
    if v:
        o_adv, o_lsb = O["hmtx"][o_cmap[c]]
        o_rsb = o_adv - o_glyf[o_cmap[c]].xMax
        r = RCH[c]
        if o_lsb + o_rsb > 20:
            sbr.append((r["xMin"] + r["adv"] - r["xMax"]) / ((o_lsb + o_rsb) * PRE))
SB_O = statistics.median(sbr) if sbr else 1.0
print("generic h:", {k: round(v, 3) for k, v in H_GEN.items()}, "sidebearing ratio", round(SB_O, 3))
for c in outline_plan:
    if fit_log[c] is not None:
        continue
    on = o_cmap[c]
    pts, flags, ends, h, d = fit_oswald(c, None, None, fixed_h=H_GEN[vcls(c)])
    o_adv, o_lsb = O["hmtx"][on]
    o_rsb = (o_adv - o_glyf[on].xMax) * PRE * SB_O
    lsb = o_lsb * PRE * SB_O
    pts[:, 0] += lsb - pts[:, 0].min()
    adv = pts[:, 0].max() + o_rsb
    fit_log[c] = (h, d, False)
    add_glyph(gname(c), simple(pts, flags, list(ends)), adv, c)

# feet and tails kept inside the reference's ink box.  Where the weight needed
# more width than the reference ink (fit_oswald's allowance), the letter was
# centred, so Л's kicked foot and the tails of Ц ц reach 30 units past the
# reference's ink: АЛ КЛ ХЛ and ЦА overlapped at the baseline.  Only the foot or
# tail is drawn in, horizontally and toward the stroke it grows from.  The
# diagonal legs of Ж К end 12-14 units past it (АЖ ЖА КЖ touched): their feet
# lean in, fading out toward the middle of the letter.  Done before accented
# letters are composed from these.
def _pts(n):
    cc, e, f = glyf[n].getCoordinates(glyf)
    return np.array(cc, float), list(e), [x & 1 for x in f]


# (any letter fitted from Oswald: bar ends like Ђ's, tails, kicked feet; only
# where the part past the line is upright, i.e. not a diagonal stroke)
for c in sorted(fit_log):
    n = new_names.get(c)
    r = RCH.get(c, {})
    if n is None or "xMin" not in r or not fit_log[c] or glyf[n].isComposite():
        continue
    pts, e, f = _pts(n)
    for side in ("L", "R"):
        lim = r["xMin"] if side == "L" else r["xMax"]
        pr = fontops.side_protrusion(pts, e, f, side, lim)
        if not pr or pr[0] is None or abs(pr[1] - lim) <= 6 or pr[3] - pr[2] >= 0.4 * np.ptp(pts[:, 1]):
            continue
        # upright: the outer edge stays within a third of the overhang across the band
        ys = np.arange(pr[2], pr[3] + 0.1, 4.0)
        xs = [fontops.edge_at(pts, e, f, y, side) for y in ys]
        xs = [x for x in xs if x is not None]
        over = abs(pr[1] - lim)
        if not xs or abs(lim - np.median(xs)) < 0.66 * over:
            continue
        pts = fontops.squeeze_side(pts, side, pr[0], lim)
        print(f"{chr(c)}: {side} end {pr[1]:.0f} -> {lim} (band y {pr[2]:.0f}..{pr[3]:.0f}, from x {pr[0]:.0f})")
    glyf[n] = simple(pts, f, e)
for c, sides in ((0x0416, "LR"), (0x041A, "R")):
    n = new_names.get(c)
    r = RCH.get(c, {})
    if n is None or "xMin" not in r or glyf[n].isComposite():
        continue
    pts, e, f = _pts(n)
    x0, x1, y0, y1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    cx, w = (x0 + x1) / 2, x1 - x0
    for side in sides:
        foot = fontops.edge_at(pts, e, f, y0 + 4, side)
        over = (r["xMin"] - foot) if side == "L" else (foot - r["xMax"])
        if foot is None or over <= 2:
            continue
        x_from = cx + 0.1 * w if side == "R" else cx - 0.1 * w
        pts = fontops.lean_in(pts, side, over, y0, y0 + 0.45 * (y1 - y0), x_from, 0.15 * w)
        print(f"{chr(c)}: {side} foot leaned in by {over:.0f}")
    glyf[n] = simple(pts, f, e)

# ------------------------------------- 1b. currency signs with bars/slashes
# Oswald draws ₮ ₸ ₴ as a body (its T, or its S mirrored) plus separate thin
# rectangular bars.  Emboldening the merged outline swells the bars into each
# other, so these are rebuilt from Smack Dab's own T and S with bars sized from
# Anton's T crossbar, keeping Oswald's bar/gap proportions.
import pathops  # noqa: E402


def union_glyph(contours):
    """contours: list of point lists (straight-sided) or (pts, ends, flags)
    TrueType outlines -> one clean simple glyph (clockwise, no overlaps)."""
    acc = None
    for item in contours:      # each part normalised on its own, then united
        if isinstance(item, tuple):
            part = fontops._path_from_segments(fontops.tt_segments(*item))
        else:
            part = pathops.Path()
            pen = part.getPen()
            pen.moveTo(tuple(item[0]))
            for p in item[1:]:
                pen.lineTo(tuple(p))
            pen.closePath()
        part.simplify(fix_winding=True, clockwise=True)
        acc = part if acc is None else pathops.op(acc, part, pathops.PathOp.UNION,
                                                  fix_winding=True, clockwise=True)
    p, e, f = fontops._path_to_tt(acc)
    return simple(p, f, e)


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]


T_adv = hmtx["T"][0]
tc, te, tf = glyf["T"].getCoordinates(glyf)
tc = np.array(tc, float)
txs, tys = sorted(set(tc[:, 0])), sorted(set(tc[:, 1]))
assert len(txs) == 4 and len(tys) == 3, "expected a plain rectangular T"
TX0, SX0, SX1, TX1 = txs
CAP, BAR = tys[2], tys[2] - tys[1]          # crossbar thickness (311 at UPM 2048)
T_outline = (tc, list(te), [x & 1 for x in tf])

if 0x20B8 in outline_plan:       # ₸ tenge: two crossbars over a stem
    bt, gap = round(0.87 * BAR), round(0.35 * BAR)
    y1 = CAP - bt - gap
    add_glyph(new_names[0x20B8], union_glyph([
        rect(TX0, CAP - bt, TX1, CAP), rect(TX0, y1 - bt, TX1, y1), rect(SX0, 0, SX1, y1 - bt + 1)]),
        T_adv, 0x20B8)
if 0x20AE in outline_plan:       # ₮ tugrik: T with two rising slashes over the stem
    t, step = round(0.61 * BAR), round(0.61 * BAR) + round(0.35 * BAR)
    slope = 129 / 359                                   # Oswald's slash angle
    inset = 0.0757 * (TX1 - TX0)
    xa, xb = TX0 + inset, TX1 - inset
    mid_x, mid_y = (xa + xb) / 2, 0.4966 * CAP
    sl = []
    for cy in (mid_y - step / 2, mid_y + step / 2):
        ya, yb = cy - slope * (mid_x - xa), cy + slope * (xb - mid_x)
        sl.append([(xa, ya - t / 2), (xa, ya + t / 2), (xb, yb + t / 2), (xb, yb - t / 2)])
    assert max(p[1] for p in sl[1]) < CAP - BAR - 60
    add_glyph(new_names[0x20AE], union_glyph([T_outline] + sl), T_adv, 0x20AE)
if 0x20B4 in outline_plan:       # ₴ hryvnia: mirrored S with two bars
    S_adv = hmtx["S"][0]
    sc_, se, sf = glyf["S"].getCoordinates(glyf)
    sc_ = np.array(sc_, float)
    sc_[:, 0] = S_adv - sc_[:, 0]                       # mirror (direction fixed by union)
    bt = gap = round(0.48 * BAR)
    mid = 0.5037 * CAP
    bx0, bx1 = sc_[:, 0].min() - 40, sc_[:, 0].max() + 40
    add_glyph(new_names[0x20B4], union_glyph([
        (sc_, list(se), [x & 1 for x in sf]),
        rect(bx0, mid - gap / 2 - bt, bx1, mid - gap / 2), rect(bx0, mid + gap / 2, bx1, mid + gap / 2 + bt)]),
        S_adv, 0x20B4)

# Ҹ ҹ (che with vertical stroke): Oswald adds a separate thin stroke to its Ч ч.
# Emboldened as one outline the stroke swelled to fill the counter, so they are
# rebuilt from Smack Dab's own Ч ч, widened, with a stroke of Oswald's
# proportion (stroke/stem) and white gaps like Anton's own counters.
def vranges(pts, ends, flags, x):
    """Vertical ink intervals of an outline on the line x."""
    pts = np.asarray(pts, float)[:, ::-1]
    return fontops._scan_gaps(fontops.nonzero_polys(pts, ends, flags), x)


# (base, stroke/stem, white gap each side, stroke ends beyond the bowl in
#  Oswald units below/above, Oswald body height)
CHE_STROKE = {0x04B8: (0x0427, 0.45, 110.0, 125.0, 141.0, 810.0),
              0x04B9: (0x0447, 0.42, 92.0, 90.0, 108.0, 578.0)}
for cp, (bcp, ratio, gap_w, ext_lo, ext_hi, o_h) in CHE_STROKE.items():
    if cp not in outline_plan or bcp not in new_names:
        continue
    bn = new_names[bcp]
    bc, be, bf = glyf[bn].getCoordinates(glyf)
    bc, be, bf = np.array(bc, float), list(be), [x & 1 for x in bf]
    top = bc[:, 1].max()
    ink = fontops._scan_gaps(fontops.nonzero_polys(bc, be, bf), 0.8 * top)
    assert len(ink) == 2, (cp, ink)
    (a0, a1), (s0, s1) = ink
    stem = ((a1 - a0) + (s1 - s0)) / 2
    w = ratio * stem
    dx = max(0.0, w + 2 * gap_w - (s0 - a1))
    mid = (a1 + s0) / 2
    bc[bc[:, 0] > mid, 0] += dx                     # open the counter
    cx = mid + dx / 2
    bowl = vranges(bc, be, bf, cx)
    assert len(bowl) == 1, (cp, bowl)
    v = top / o_h
    y0, y1 = bowl[0][0] - ext_lo * v, min(bowl[0][1] + ext_hi * v, top - 0.12 * top)
    add_glyph(new_names[cp], union_glyph([(bc, be, bf), rect(cx - w / 2, y0, cx + w / 2, y1)]),
              hmtx[bn][0] + dx, cp)
    print(f"{chr(cp)}: built from {chr(bcp)} widened by {dx:.0f}, stroke {w:.0f} wide")

# Ҿ ҿ are Ҽ ҽ with a descender, without a diaeresis; Oswald draws two dots on
# them by mistake.  Contours lying wholly above the body of Ҽ ҽ are removed.
for cp, body in ((0x04BE, 0x04BC), (0x04BF, 0x04BD)):
    if cp not in new_names or body not in new_names:
        continue
    top = np.array(glyf[new_names[body]].getCoordinates(glyf)[0])[:, 1].max()
    c, e, f = glyf[new_names[cp]].getCoordinates(glyf)
    c = np.array(c, float)
    keep_p, keep_f, keep_e, st = [], [], [], 0
    for en in e:
        if c[st:en + 1, 1].min() < top - 10:
            keep_p.extend(c[st:en + 1])
            keep_f.extend(f[st:en + 1])
            keep_e.append(len(keep_p) - 1)
        st = en + 1
    if len(keep_e) < len(e):
        add_glyph(new_names[cp], simple(keep_p, [x & 1 for x in keep_f], keep_e), hmtx[new_names[cp]][0], cp)
        print(f"{chr(cp)}: removed {len(e) - len(keep_e)} stray dot contours (Oswald)")

# ----------------------------------------------------- 2. look-alike letters
homo_src = {}          # look-alike glyph -> (Latin glyph, x offset)


def flat_parts(name, dx, dy=0):
    """Component list for `name` placed at (dx, dy) without nesting."""
    g = glyf[name]
    if g.isComposite():
        return [(c.glyphName, c.x + dx, c.y + dy) for c in g.components]
    return [(name, dx, dy)]


for c, lat in homo_plan.items():
    ladv = hmtx[lat][0]
    adv = RCH[c]["adv"] if c in RCH else ladv
    dx = (adv - ladv) / 2
    nm = gname(c)
    add_glyph(nm, composite(flat_parts(lat, dx)), adv, c)
    homo_src[nm] = (lat, dx)

# К Ќ reuse K Ḱ, whose leg ends 13 units past the reference's (narrower) К:
# Кл КХ touched.  Their own copy of the outline has the foot leaned in, like Ж.
for c in (0x041A, 0x040C):
    nm = new_names.get(c)
    r = RCH.get(c, {})
    if nm is None or "xMax" not in r:
        continue
    cc, e, f = glyf[nm].getCoordinates(glyf)
    pts, e, f = np.array(cc, float), list(e), [x & 1 for x in f]
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    foot = fontops.edge_at(pts, e, f, y0 + 4, "R")
    if foot is None or foot - r["xMax"] <= 2:
        continue
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    pts = fontops.lean_in(pts, "R", foot - r["xMax"], y0, y0 + 0.45 * (y1 - y0),
                          (x0 + x1) / 2 + 0.1 * (x1 - x0), 0.15 * (x1 - x0))
    glyf[nm] = simple(pts, f, e)
    print(f"{chr(c)}: R foot leaned in by {foot - r['xMax']:.0f}")

# ------------------------------------- 3. mark anchors (Anton's placement)
gpos = SD["GPOS"].table
markbase = []
for lk in gpos.LookupList.Lookup:
    if lk.LookupType == 4:
        markbase += lk.SubTable


def mark_info(m):
    for st in markbase:
        if m in st.MarkCoverage.glyphs:
            rec = st.MarkArray.MarkRecord[st.MarkCoverage.glyphs.index(m)]
            return st, rec.Class, (rec.MarkAnchor.XCoordinate, rec.MarkAnchor.YCoordinate)
    return None, None, None


def base_anchor(st, cls, b):
    if b in st.BaseCoverage.glyphs:
        a = st.BaseArray.BaseRecord[st.BaseCoverage.glyphs.index(b)].BaseAnchor[cls]
        if a is not None:
            return a.XCoordinate, a.YCoordinate
    return None


def alike_anchor(st, cls, newg, c):
    """Anchor for a new glyph, transferred from a similar Latin letter by its
    relative position in the ink box."""
    up = vcls(c) == "uc"
    xmn, ymn, xmx, ymx = bounds(newg)
    cands = ["H", "E", "O", "I"] if up else (["h", "b", "k", "l"] if ymx > 1450 else ["n", "o", "u", "e"])
    for lat in cands:
        lg = sd_cmap[ord(lat)]
        a = base_anchor(st, cls, lg)
        if a:
            lx0, _, lx1, _ = bounds(lg)
            fx = (a[0] - lx0) / max(1, lx1 - lx0)
            _, ly0, _, ly1 = bounds(lg)
            y = a[1]
            if y > ly1 - 300:                    # top-type anchor
                y += max(0, ymx - ly1)
            elif y < ly0 + 100:                  # bottom-type anchor
                y += min(0, ymn - ly0)
            return xmn + fx * (xmx - xmn), y
    return None


def anchor_for(st, cls, base, c_base):
    a = base_anchor(st, cls, base)
    if a is None and base in homo_src:            # look-alike: Latin anchors
        lat, dx = homo_src[base]
        a2 = base_anchor(st, cls, lat)
        if a2:
            return a2[0] + dx, a2[1]
    if a is None and base in adv_of:
        g = glyf[base]
        if g.isComposite() and len(g.components) == 1:        # look-alike
            comp = g.components[0]
            a2 = base_anchor(st, cls, comp.glyphName)
            if a2:
                return a2[0] + comp.x, a2[1] + comp.y
        a = alike_anchor(st, cls, base, c_base)
    return a


def tuck(name, c=None):
    """Keep composite accents inside the clip box and near the advance (and,
    where the reference has the letter, inside its ink box)."""
    g = glyf[name]
    if not g.isComposite():
        return
    base = glyf[g.components[0].glyphName]
    base.recalcBounds(glyf)
    Bt = base.yMax if base.yMax < ms["cap"] + 100 else ms["cap"] + 15
    r = RCH.get(c, {}) if c is not None else {}
    res = fontops.tuck_parts(fontops.composite_parts(glyf, name), hmtx[name][0], Bt, TOP_LIMIT, BOT_LIMIT,
                             xlim=(r["xMin"], r["xMax"]) if "xMin" in r else None)
    if res is not None:
        glyf[name] = simple(*res)


for c, (base, marks) in comp_plan.items():
    cb = ord(base)
    bg = sd_cmap.get(cb) or new_names.get(cb)
    if bg is None:
        continue
    up = unicodedata.category(chr(c)) == "Lu"
    badv = hmtx[bg][0]
    adv = RCH[c]["adv"] if c in RCH else badv
    shift = (adv - badv) / 2
    parts = flat_parts(bg, shift)
    stack_top = None
    for m in marks:
        mg = sd_cmap[ord(m)]
        if up and mg + ".case" in SD.getGlyphOrder():
            mg = mg + ".case"
        st, mcls, ma = mark_info(mg)
        ba = anchor_for(st, mcls, bg, cb) if st else None
        mb = bounds(mg)
        if ba is None:
            x0, y0, x1, y1 = bounds(bg)
            ba, ma = ((x0 + x1) / 2, y1 + 40), ((mb[0] + mb[2]) / 2, mb[1])
        dx, dy = ba[0] - ma[0] + shift, ba[1] - ma[1]
        if stack_top is not None and mb[1] + dy < stack_top + 30 and mb[1] > 0:
            dy = stack_top + 30 - mb[1]
        stack_top = mb[3] + dy if mb[1] > 0 else stack_top
        parts.append((mg, dx, dy))
    name = gname(c)
    add_glyph(name, composite(parts), adv, c)
    tuck(name, c)
    glyf[name].recalcBounds(glyf)
    hmtx[name] = (hmtx[name][0], glyf[name].xMin)

# ------------------------------------------------------------------ 4. GDEF
gdef = SD["GDEF"].table.GlyphClassDef.classDefs
for n in new_names.values():
    gdef[n] = 1

SD.getReverseGlyphMap(rebuild=True)
# --------------------------------- 5. mark-to-base anchors for new letters
def composed_anchor(st, cls, n, c):
    b = comp_plan[c][0]
    bn = sd_cmap.get(ord(b)) or new_names.get(ord(b))
    if bn is None:
        return None
    a = anchor_for(st, cls, bn, ord(b))
    if a is None:
        return None
    _, by0, _, by1 = bounds(bn)
    _, ny0, _, ny1 = bounds(n)
    dx = glyf[n].components[0].x if glyf[n].isComposite() else (hmtx[n][0] - hmtx[bn][0]) / 2
    y = a[1] + (ny1 - by1 if a[1] > by1 - 300 else (ny0 - by0 if a[1] < by0 + 100 else 0))
    return a[0] + dx, y


for st in markbase:
    for c, n in sorted(new_names.items(), key=lambda kv: SD.getGlyphID(kv[1])):
        if unicodedata.category(chr(c))[0] != "L":
            continue
        anchors = []
        for cls in range(st.ClassCount):
            a = composed_anchor(st, cls, n, c) if c in comp_plan else anchor_for(st, cls, n, c)
            if a is None:
                anchors.append(None)
            else:
                an = ot.Anchor()
                an.Format, an.XCoordinate, an.YCoordinate = 1, int(round(a[0])), int(round(a[1]))
                anchors.append(an)
        if not any(anchors):
            continue
        rec = ot.BaseRecord()
        rec.BaseAnchor = anchors
        st.BaseCoverage.glyphs.append(n)
        st.BaseArray.BaseRecord.append(rec)
        st.BaseArray.BaseCount = len(st.BaseArray.BaseRecord)

# ----------------------------------------------------------- 6. kerning
side = {}   # new glyph -> (left-side Latin glyph, right-side Latin glyph)
for c, n in new_names.items():
    if c in homo_plan:
        lat = homo_plan[c]
        side[n] = (lat, lat)
        continue
    la, ra = alike_of(c)
    side[n] = (sd_cmap.get(ord(la)) if la else None, sd_cmap.get(ord(ra)) if ra else None)
    if c in comp_plan:                       # inherit from the base letter
        b = comp_plan[c][0]
        bn = sd_cmap.get(ord(b)) or new_names.get(ord(b))
        if bn in side:
            side[n] = side[bn]
        elif bn:
            side[n] = (bn, bn)

for lk in gpos.LookupList.Lookup:
    if lk.LookupType != 2:
        continue
    pairs = {}
    for st in list(lk.SubTable):
        if st.Format == 1:
            for L, ps in zip(st.Coverage.glyphs, st.PairSet):
                for pv in ps.PairValueRecord:
                    v = getattr(pv.Value1, "XAdvance", 0) if pv.Value1 else 0
                    if not v:
                        continue
                    for n, (lft, rgt) in side.items():
                        if rgt == L:
                            pairs[(n, pv.SecondGlyph)] = v
                        if lft == pv.SecondGlyph:
                            pairs[(L, n)] = v
            for n1, (l1, r1) in side.items():          # new-new pairs
                if r1 in st.Coverage.glyphs:
                    ps = st.PairSet[st.Coverage.glyphs.index(r1)]
                    for pv in ps.PairValueRecord:
                        v = getattr(pv.Value1, "XAdvance", 0) if pv.Value1 else 0
                        for n2, (l2, r2) in side.items():
                            if v and l2 == pv.SecondGlyph:
                                pairs[(n1, n2)] = v
        else:
            cd1, cd2 = st.ClassDef1.classDefs, st.ClassDef2.classDefs
            cov = set(st.Coverage.glyphs)
            for n, (lft, rgt) in side.items():
                if rgt in cov:
                    cd1[n] = cd1.get(rgt, 0)
                    st.Coverage.glyphs.append(n)
                if lft is not None and lft in cd2:
                    cd2[n] = cd2[lft]
            st.Coverage.glyphs.sort(key=SD.getGlyphID)
    if pairs:
        vr = {k: (buildValue({"XAdvance": v}), None) for k, v in pairs.items()}
        sub = buildPairPosGlyphsSubtable(vr, SD.getReverseGlyphMap())
        lk.SubTable.insert(0, sub)
        lk.SubTableCount = len(lk.SubTable)
    print("kerning pairs added (glyph-pair subtable):", len(pairs))

# ------------------------------------------- 7. script list: add 'cyrl'
for tag in ("GSUB", "GPOS"):
    t = SD[tag].table
    if any(sr.ScriptTag == "cyrl" for sr in t.ScriptList.ScriptRecord):
        continue
    dflt = next(sr for sr in t.ScriptList.ScriptRecord if sr.ScriptTag == "DFLT")
    sr = ot.ScriptRecord()
    sr.ScriptTag = "cyrl"
    sr.Script = ot.Script()
    ls = ot.LangSys()
    ls.LookupOrder = None
    ls.ReqFeatureIndex = 0xFFFF
    ls.FeatureIndex = list(dflt.Script.DefaultLangSys.FeatureIndex)
    ls.FeatureCount = len(ls.FeatureIndex)
    sr.Script.DefaultLangSys = ls
    sr.Script.LangSysRecord = []
    sr.Script.LangSysCount = 0
    t.ScriptList.ScriptRecord.append(sr)
    t.ScriptList.ScriptRecord.sort(key=lambda r: r.ScriptTag)
    t.ScriptList.ScriptCount = len(t.ScriptList.ScriptRecord)

# ------------------------------------------------------------- 8. tables
for n in new_names.values():
    g = glyf[n]
    g.recalcBounds(glyf)
    hmtx[n] = (hmtx[n][0], g.xMin if g.numberOfContours else 0)
# ------------------------------------------- 9. outline hygiene
# merge self-overlaps (from emboldening) and overlapping parts of decomposed
# accented glyphs into clean, non-overlapping contours (nonzero rule)
from fontTools.ttLib import removeOverlaps  # noqa: E402
simple_names = [n for n in SD.getGlyphOrder() if not glyf[n].isComposite() and glyf[n].numberOfContours > 0]
n_slit = 0
for n in simple_names:          # fill hairline crotch tips (< 24 units wide)
    g = glyf[n]
    pts, ends, fl = fontops.fill_slits(np.array(g.coordinates, float), list(g.endPtsOfContours), list(g.flags))
    if len(pts) != len(g.coordinates):
        glyf[n] = simple(pts, fl, ends)
        n_slit += 1
removeOverlaps.removeOverlaps(SD, glyphNames=simple_names, removeHinting=True)
# long hairline crotch slits in the adapted letters (e.g. У): emboldening pushed
# two strokes to within a few units of each other over a long stretch
slit_fixed = []
added = set(new_names.values())
for n in simple_names:
    if n not in added:
        continue                # Anton's own letters are left as designed
    g = glyf[n]
    c, e, f = g.getCoordinates(glyf)
    p, e2, f2, a = fontops.fill_narrow_channels(np.array(c, float), list(e), [x & 1 for x in f], min_len=150)
    if a > 0:
        glyf[n] = simple(p, f2, e2)
        slit_fixed.append(n)
print("hygiene: long crotch slits closed in", slit_fixed)
# duplicate consecutive points and on-curve points < 2 units apart
tidied = []
for n in simple_names:
    g = glyf[n]
    if g.isComposite() or g.numberOfContours <= 0:
        continue
    c, e, f = g.getCoordinates(glyf)
    p, e2, f2, k = fontops.tidy_points(np.array(c, float), list(e), [x & 1 for x in f])
    if k:
        glyf[n] = simple(p, f2, e2)
        tidied.append(n)
print("hygiene: duplicate/near-duplicate points removed in", len(tidied), "glyphs")
n_tiny = 0
for n in simple_names:          # drop pinholes/specks left by outline operations
    kp, kf, ke, changed = fontops.drop_tiny_contours(glyf[n])
    if changed:
        glyf[n] = simple(kp, kf, ke)
        n_tiny += 1
print(f"hygiene: crotch tips filled in {n_slit} glyphs, pinholes removed in {n_tiny}")
# flat edges 1-2 units off the baseline, x-height or cap height go onto it
# (the heights of the flat tops of Latin ı and H)
ZONES = [0] + [max(glyf[n].getCoordinates(glyf)[0], key=lambda q: q[1])[1] for n in ("dotlessi", "H")]
# round overshoots as deep as the Latin o / O's: Oswald's rounds overshoot
# further (З 1642 against O's 1634, э 1351 against o's 1341).  Only the band
# past the x-height / cap height (or below the baseline) is scaled, and only
# on letters whose ink goes no further than an overshoot there.
def _ext(n):
    c = np.array(glyf[n].getCoordinates(glyf)[0], float)
    return c[:, 1].min(), c[:, 1].max()


OVER = {"lc": _ext("o"), "uc": _ext("O")}
n_over = 0
for c, n in sorted(new_names.items()):
    g = glyf[n]
    if not fit_log.get(c) or g.isComposite() or g.numberOfContours <= 0:
        continue
    cls = "uc" if unicodedata.category(chr(c)) == "Lu" else "lc"
    Z = ZONES[2] if cls == "uc" else ZONES[1]
    lo_ref, hi_ref = OVER[cls]
    cc, e, f = g.getCoordinates(glyf)
    pts = np.array(cc, float)
    lo, hi = pts[:, 1].min(), pts[:, 1].max()
    changed = False
    if Z + 4 <= hi <= Z + 40 and abs(hi - hi_ref) > 2:        # a round top, not a flat one
        m = pts[:, 1] > Z
        pts[m, 1] = Z + (pts[m, 1] - Z) * (hi_ref - Z) / (hi - Z)
        changed = True
    if -40 <= lo <= -4 and abs(lo - lo_ref) > 2:
        m = pts[:, 1] < 0
        pts[m, 1] = pts[m, 1] * lo_ref / lo
        changed = True
    if changed:
        glyf[n] = simple(pts, [x & 1 for x in f], list(e))
        n_over += 1
print(f"hygiene: overshoots matched to Latin o/O in {n_over} glyphs")
n_snap = 0
for n in SD.getGlyphOrder():
    g = glyf[n]
    if g.isComposite() or g.numberOfContours <= 0 or gdef.get(n) == 3:
        continue
    c, e, f = g.getCoordinates(glyf)
    p, k = fontops.snap_flats(np.array(c, float), list(e), [x & 1 for x in f], ZONES)
    if k:
        glyf[n] = simple(p, [x & 1 for x in f], list(e))
        n_snap += 1
print(f"hygiene: flat edges put onto heights {ZONES} in {n_snap} glyphs")
for n in SD.getGlyphOrder():
    g = glyf[n]
    g.recalcBounds(glyf)
    hmtx[n] = (hmtx[n][0], g.xMin if g.numberOfContours else 0)

name = SD["name"]
name.setName("Copyright 2020 The Anton Project Authors (https://github.com/googlefonts/AntonFont.git). "
             "Copyright 2016 The Oswald Project Authors (https://github.com/googlefonts/OswaldFont). "
             "Smack Dab modifications Copyright 2026 The Smack Dab Project Authors.", 0, 3, 1, 0x409)
desc = name.getDebugName(10)
if desc and "Oswald" not in desc:
    name.setName(desc + " Cyrillic is adapted from Oswald Bold.", 10, 3, 1, 0x409)
# drop glyphs nothing can reach (cmap, GSUB or components), inherited from Anton
from fontTools import subset  # noqa: E402
opts = subset.Options()
opts.layout_features = ["*"]
opts.layout_scripts = ["*"]
opts.name_IDs = ["*"]
opts.name_languages = ["*"]
opts.name_legacy = True
opts.glyph_names = True
opts.notdef_glyph = True
opts.notdef_outline = True
opts.recommended_glyphs = True
opts.hinting = False
opts.legacy_kern = True
opts.drop_tables = []
opts.prune_unicode_ranges = False
opts.passthrough_tables = True
n_before = len(SD.getGlyphOrder())
sub = subset.Subsetter(opts)
sub.populate(unicodes=SD.getBestCmap().keys())
sub.subset(SD)
print(f"unreachable glyphs removed: {n_before - len(SD.getGlyphOrder())}")
os2 = SD["OS/2"]
os2.recalcUnicodeRanges(SD)
os2.recalcCodePageRanges(SD)
# Δ Σ Ω π sit in the Greek block but the font has no Greek: don't claim the
# Greek range, or software choosing fonts by it shows empty boxes for Greek text
os2.ulUnicodeRange1 &= ~(1 << 7)

# Azerbaijani, Crimean Tatar, Kazakh and Tatar (Latin) need Turkish's dotted i
# (İ in small caps / case mapping); Anton gives it to Turkish only
gsub = SD["GSUB"].table
feats = gsub.FeatureList.FeatureRecord
latn = next(r for r in gsub.ScriptList.ScriptRecord if r.ScriptTag == "latn").Script
trk = next(l for l in latn.LangSysRecord if l.LangSysTag == "TRK ").LangSys
trk_locl = [i for i in trk.FeatureIndex if feats[i].FeatureTag == "locl"]
for l in latn.LangSysRecord:
    if l.LangSysTag in ("AZE ", "CRT ", "KAZ ", "TAT "):
        have = {feats[i].FeatureTag for i in l.LangSys.FeatureIndex}
        if "locl" not in have and trk_locl:
            l.LangSys.FeatureIndex = sorted(l.LangSys.FeatureIndex + trk_locl)
            l.LangSys.FeatureCount = len(l.LangSys.FeatureIndex)
SD.save(OUT)
print("saved", OUT, "glyphs:", len(SD.getGlyphOrder()), "added:", len(new_names))
json.dump({f"{c:04X}": [float(v[0]), float(v[1]), v[2]] for c, v in fit_log.items() if v},
          open(OUT + ".oswald_fit.json", "w"))
