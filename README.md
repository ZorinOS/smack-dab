# Smack Dab

Smack Dab is a heavy condensed display typeface that is metric-compatible with Impact®.
It is a modified version of [Anton](https://github.com/googlefonts/AntonFont) (Latin), extended
with Cyrillic adapted from [Oswald](https://github.com/googlefonts/OswaldFont) Bold. Both are
licensed under the SIL Open Font License 1.1, and so is Smack Dab (see `OFL.txt`).

## Files

* `fonts/SmackDab-Regular.ttf` – hinted (ttfautohint) TrueType, for desktop use.
* `fonts/SmackDab-Regular.woff2` – the same font for the web.
* `specimen.png` – Impact vs Smack Dab, with set widths shown as bars.
* `source/` – the complete, reproducible build and its quality checks.

## Character set

* **Latin:** everything Anton covers (Western, Central and Eastern European, Vietnamese,
  African Latin, IPA additions).
* **Cyrillic:** Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Macedonian and many other
  languages (Kazakh, Tatar, Bashkir, Chuvash, Abkhaz and more): 241 characters, including all
  Cyrillic in Impact. Also ₮ ₴ ₸.

## How to build

Build (Python 3, `pip install -r source/requirements.txt`):

```
python3 -I source/measure_impact.py Impact.ttf source/impact_metrics.json   # optional: re-measure
python3 -I source/make_fonts.py Anton-Regular.ttf "Oswald[wght].ttf" source/impact_metrics.json build/
```

## Trademark note

Impact is a trademark of its owner. It is used here only to describe compatibility; the font
does not use the name. Smack Dab is not affiliated with or endorsed by the owners of Impact.
This README is not legal advice.
