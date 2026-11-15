"""Full Smack Dab pipeline.

usage: python3 -I make_fonts.py Anton-Regular.ttf "Oswald[wght].ttf" impact_metrics.json OUTDIR

1. build_smackdab.py  Anton -> Smack Dab Latin (reference metrics applied)
2. prepare_oswald.py  Oswald variable font -> static Bold instance
3. add_cyrillic.py    add Cyrillic adapted from Oswald Bold
4. complete_marks.py  combining accents for every letter (anchors Anton lacks)
5. fix_kern_collisions.py   relax Anton's kerning pairs that would make glyphs touch
6. impact_kerning.py  the reference's kerning pairs, exactly; Anton's kerning
                      kept only for characters the reference lacks
7. ttfautohint        hinting; then WOFF2
"""
import os
import subprocess
import sys

import ttfautohint
from fontTools.ttLib import TTFont

anton, oswald_vf, metrics, outdir = sys.argv[1:5]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(outdir, exist_ok=True)
tmp = os.path.join(outdir, "_work")
os.makedirs(tmp, exist_ok=True)
latin = os.path.join(tmp, "SmackDab-latin-unhinted.ttf")
osw = os.path.join(tmp, "Oswald-Bold-700.ttf")
unhinted = os.path.join(outdir, "SmackDab-Regular-unhinted.ttf")
hinted = os.path.join(outdir, "SmackDab-Regular.ttf")


def run(script, *args):
    subprocess.run([sys.executable, "-I", os.path.join(here, script), *args], check=True)


run("build_smackdab.py", anton, metrics, latin)
run("prepare_oswald.py", oswald_vf, osw)
cyr = os.path.join(tmp, "SmackDab-cyr-unhinted.ttf")
run("add_cyrillic.py", latin, osw, metrics, cyr)
marks = os.path.join(tmp, "SmackDab-marks-unhinted.ttf")
run("complete_marks.py", cyr, marks)
kerncol = os.path.join(tmp, "SmackDab-kern-unhinted.ttf")
run("fix_kern_collisions.py", marks, kerncol)
run("impact_kerning.py", kerncol, metrics, unhinted)
ttfautohint.ttfautohint(in_file=unhinted, out_file=hinted, windows_compatibility=True,
                        no_info=True, hint_composites=False)
f = TTFont(hinted)
f.flavor = "woff2"
f.save(os.path.join(outdir, "SmackDab-Regular.woff2"))
print("done:", hinted)
