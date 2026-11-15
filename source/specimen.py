"""Side-by-side specimen: reference vs Smack Dab, HarfBuzz (raqm) layout.
Bars under each line show the set width (identical bars = no layout shift).

usage: python3 -I specimen.py ref.ttf SmackDab.ttf out.png
"""
import sys

from PIL import Image, ImageDraw, ImageFont

ref, sd, out = sys.argv[1:4]
LINES = [
    "HAMBURGEFONSTIV hamburgefonstiv",
    "The quick brown fox jumps over the lazy dog.",
    "Zwölf Boxkämpfer jagen Viktor — ÅÉÎÕÜ ąčęłńśżő",
    "0123456789 !?&%@#$€£ (12:30) “Quotes” «guillemets»",
    "СЪЕШЬ ЖЕ ЕЩЁ ЭТИХ МЯГКИХ БУЛОК",
    "Съешь же ещё этих мягких французских булок",
    "Ђорђе, Љубљана, Їжак, Ґанок, Ўладзімір, Қазақстан",
]
size = 84
pad, label_w = 24, 230
lh = int(size * 1.32)
fr = ImageFont.truetype(ref, size, layout_engine=ImageFont.Layout.RAQM)
fs = ImageFont.truetype(sd, size, layout_engine=ImageFont.Layout.RAQM)
lab = ImageFont.truetype(sd, 30)
probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
W = int(max(probe.textlength(t, font=fs, features=["-kern"]) for t in LINES)) + label_w + 2 * pad
img = Image.new("RGB", (W, pad + len(LINES) * (2 * lh + 26)), "white")
d = ImageDraw.Draw(img)
y = pad
for t in LINES:
    for name, f, col in (("Impact", fr, (180, 30, 30)), ("Smack Dab", fs, (30, 60, 170))):
        d.text((pad, y + size * 0.35), name, font=lab, fill=col)
        d.text((label_w, y), t, font=f, fill="black", features=["-kern"])
        w = d.textlength(t, font=f, features=["-kern"])
        d.rectangle([label_w, y + int(size * 1.18), label_w + w, y + int(size * 1.18) + 5], fill=col)
        y += lh
    y += 26
img.save(out)
print("saved", out, img.size)
