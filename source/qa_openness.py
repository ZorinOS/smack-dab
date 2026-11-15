"""Relative openness of Cyrillic letters vs their Latin look-alikes:

R = [open_SD(cyr)/open_SD(lat)] / [open_Oswald(cyr)/open_Oswald(lat)].
R ~ 1: the letter was treated like its Latin analog; R << 1: clogged more.

usage: python3 -I qa_openness.py SmackDab.ttf Oswald-Bold-700.ttf
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontops
from fontTools.ttLib import TTFont
S, O = TTFont(sys.argv[1]), TTFont(sys.argv[2])
def op(f, ch, pre=1.0):
    g = f["glyf"]; n = f.getBestCmap()[ord(ch)]
    c, e, fl = g[n].getCoordinates(g)
    a = fontops.raster_outline(np.array(c, float) * pre, list(e), [x & 1 for x in fl], upx=4.0)
    w = 0
    for r in range(a.shape[0]):
        rr = fontops.runs(a[r])
        if len(rr) > 1:
            w += sum(rr[i + 1][0] - rr[i][1] for i in range(len(rr) - 1))
    return w / max(1, a.sum() + w)
PAIRS = [("Ж", "KX"), ("У", "YV"), ("Л", "AH"), ("Д", "AH"), ("Я", "R"), ("Ц", "U"), ("Ш", "U"), ("Щ", "U"),
         ("Ђ", "T"), ("Ћ", "T"), ("Љ", "B"), ("Њ", "B"), ("И", "N"), ("Б", "B"), ("Ъ", "B"), ("Ы", "B"),
         ("Ю", "O"), ("Э", "C"), ("Ф", "O"), ("Ч", "U"), ("Г", "F"), ("П", "U"), ("Ь", "B"), ("Є", "C"),
         ("ж", "kx"), ("ц", "u"), ("ш", "u"), ("щ", "u"), ("ђ", "h"), ("ћ", "h"), ("в", "b"), ("я", "n"),
         ("л", "n"), ("д", "n"), ("э", "c"), ("ю", "o"), ("и", "u"), ("к", "k"), ("б", "o"), ("ф", "o"),
         ("з", "s"), ("ы", "b"), ("ь", "b"), ("ч", "u"), ("п", "n"), ("г", "r"), ("т", "n"), ("м", "n"),
         ("н", "n"), ("є", "c"), ("ъ", "b"), ("љ", "b"), ("њ", "b"), ("џ", "u")]
rows = []
for cy, lat in PAIRS:
    sd = op(S, cy) / np.mean([op(S, x) for x in lat])
    ow = op(O, cy, 2.048) / np.mean([op(O, x, 2.048) for x in lat])
    rows.append((sd / ow, cy, lat))
rows.sort()
for r, cy, lat in rows:
    print(f"   {cy} vs {lat:3} R={r:.2f}" + ("   <-- clogged relative to Latin" if r < 0.8 else ""))
