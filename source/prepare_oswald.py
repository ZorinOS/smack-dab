"""Make a static, overlap-free Oswald Bold (wght=700) from the variable font,
which covers more Cyrillic than the static Bold TTF.

usage: python3 -I prepare_oswald.py "Oswald[wght].ttf" out.ttf
"""
import sys

from fontTools.ttLib import TTFont, removeOverlaps
from fontTools.varLib import instancer

vf = TTFont(sys.argv[1])
inst = instancer.instantiateVariableFont(vf, {"wght": 700}, updateFontNames=False)
removeOverlaps.removeOverlaps(inst)
inst.save(sys.argv[2])
print("wrote", sys.argv[2])
