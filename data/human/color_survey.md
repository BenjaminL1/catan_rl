# Colonist 1v1 player-colour survey

Measured from real ThePhantom `high`-tier footage by `scripts/color_survey.py` — **no HSV value here is invented**; every range is derived from the on-frame measurements in `color_survey_raw.json` (HUD seat-avatar rings + on-board pieces). This is the calibration source for widening `openings.PALETTE` / `_HUD_RING` beyond GREEN+BLACK (the Tier-5 `hud_unreadable` NO-GO).

- **Videos sampled:** 33 (`high` tier, setup-window frames) — **67 frames**.
- **Calibrated threshold:** a colour seen in **>= 4 distinct videos** is harvestable; fewer -> `low_sample` (harvest-excluded until more games, not a guessed range).
- **Hue convention:** OpenCV RGB->HSV hue 0..179; a hue range [lo, hi] with lo > hi wraps the 0/180 seam (RED only).

## Colour histogram (how often each identity was seen)

| identity | kind | videos | ring n | piece n | status |
|---|---|---|---|---|---|
| BLACK | achromatic | 24 | 49 | 43 | calibrated |
| RED | chromatic | 22 | 44 | 44 | calibrated |
| WHITE | achromatic | 9 | 20 | 20 | calibrated |
| PURPLE | chromatic | 4 | 8 | 8 | calibrated |
| BLUE | chromatic | 2 | 4 | 4 | LOW-SAMPLE (excluded) |
| ORANGE | chromatic | 2 | 4 | 4 | LOW-SAMPLE (excluded) |
| GREEN | chromatic | 1 | 2 | 2 | LOW-SAMPLE (excluded) |
| LIGHTGREEN | chromatic | 1 | 3 | 3 | LOW-SAMPLE (excluded) |

## Per-colour derived HSV ranges (OpenCV H 0..179, S/V 0..255)

| identity | ring H | ring S | ring V | piece H | piece S | piece V | tile_subtract |
|---|---|---|---|---|---|---|---|
| BLACK | [0, 179] | [0, 60] | [34, 130] | [0, 179] | [0, 60] | [0, 89] | False |
| RED | [169, 4] | [88, 255] | [107, 255] | [174, 9] | [121, 255] | [107, 255] | True |
| WHITE | [0, 179] | [0, 55] | [145, 255] | [0, 179] | [0, 55] | [195, 255] | False |
| PURPLE | [129, 142] | [80, 156] | [121, 181] | [136, 147] | [152, 255] | [107, 253] | False |
| BLUE | [102, 112] | [97, 252] | [118, 213] | [100, 108] | [112, 255] | [106, 255] | True |
| ORANGE | [8, 19] | [89, 246] | [112, 255] | [11, 24] | [113, 254] | [111, 255] | True |
| GREEN | [60, 70] | [100, 247] | [109, 230] | [57, 70] | [140, 255] | [102, 255] | True |
| LIGHTGREEN | [91, 101] | [88, 228] | [119, 250] | [90, 99] | [113, 255] | [109, 229] | True |

## Measured spread (non-overlap proof)

Chromatic identities are separated by **hue** (boundaries placed at the Voronoi midpoints between adjacent measured cluster means, so no two hue ranges share a value); achromatic BLACK/WHITE are separated from every chromatic colour by **saturation** (S <= ~90 vs chromatic S floors >= ~90) and from each other by **value** (BLACK V <= 130 vs WHITE V >= 145). The unit test `test_color_survey.py` enforces pairwise non-overlap of the 3-D HSV boxes for both ring and piece.

- **BLACK** (24 vids, 49 ring samples): achromatic; ring sat/val med 9.0/83.0.
- **RED** (22 vids, 44 ring samples): ring hue circular-mean **176.0** (concentration 0.998), piece hue circular-mean **2.0**; ring sat/val med 187.0/219.0.
- **WHITE** (9 vids, 20 ring samples): achromatic; ring sat/val med 12.5/175.5.
- **PURPLE** (4 vids, 8 ring samples): ring hue circular-mean **135.9** (concentration 0.999), piece hue circular-mean **141.6**; ring sat/val med 122.5/154.0.
- **BLUE** (2 vids, 4 ring samples): ring hue circular-mean **106.5** (concentration 1.0), piece hue circular-mean **102.5**; ring sat/val med 180.5/176.0.
- **ORANGE** (2 vids, 4 ring samples): ring hue circular-mean **13.6** (concentration 1.0), piece hue circular-mean **17.5**; ring sat/val med 207.5/224.5.
- **GREEN** (1 vids, 2 ring samples): ring hue circular-mean **65.0** (concentration 1.0), piece hue circular-mean **63.5**; ring sat/val med 204.0/180.0.
- **LIGHTGREEN** (1 vids, 3 ring samples): ring hue circular-mean **95.9** (concentration 1.0), piece hue circular-mean **95.8**; ring sat/val med 143.0/212.0.

## Tile / background hue collisions (`tile_subtract`)

Board saturated-pixel hue histogram (mean fraction per 10-wide bin, across all sampled frames) — the reference for which piece colours collide with a same-hued tile or the sea background:

| hue bin | fraction | |
|---|---|---|
| 0-10 | 0.038 | brick / red-hills |
| 10-20 | 0.07 | brick / wheat |
| 20-30 | 0.129 | wheat |
| 30-40 | 0.06 | sheep / pasture |
| 40-50 | 0.011 |  |
| 50-60 | 0.013 | sheep |
| 60-70 | 0.046 | wood / forest |
| 70-80 | 0.012 | wood |
| 90-100 | 0.047 | upper-green |
| 100-110 | 0.577 | **SEA BACKGROUND** |

Resulting flags:

- **BLACK**: `tile_subtract=False` — dark pieces (val <75) vs grey ore tiles / the robber; the existing hex-centre exclusion handles the robber, no baseline subtraction.
- **RED**: `tile_subtract=True` — piece hue wraps the 0/180 seam and overlaps the brick/red-hills tile band (measured hue 0-20 = ~11% of saturated board pixels). Red pieces are highly saturated (measured piece sat median ~248, p5 ~176), but this survey measures tile HUE only, not tile saturation, so a saturation-floor separation from brick cannot be verified from the data; apply the empty-baseline tile subtraction conservatively (fail-closed).
- **WHITE**: `tile_subtract=False` — white pieces (val >200, sat <45) share their signature with white number tokens / vertex borders / port glyphs; the hex-centre + lattice-snap guards discriminate, not a hue subtraction.
- **PURPLE**: `tile_subtract=False` — piece hue (~141) has no same-hued board tile (hue 130-150 is empty).
- **BLUE**: `tile_subtract=True` — CRITICAL: the board sits on a blue sea background (hue 100-110 = 58% of saturated board pixels), the same hue as blue pieces; needs the sea/tile subtraction or a tight sat+val gate.
- **ORANGE**: `tile_subtract=True` — piece hue (16-19) sits inside the brick+wheat tile band (hue 10-30, ~20% of saturated board pixels) with overlapping saturation; needs the empty-baseline tile subtraction.
- **GREEN**: `tile_subtract=True` — the known green_tile_suppressed collision: piece hue (62-65) matches the forest/pasture tiles (hue 60-70); needs the empty-baseline subtraction.
- **LIGHTGREEN**: `tile_subtract=True` — piece hue (~96) overlaps the upper-green tile band (hue 90-100).

## Excluded / low-sample colours

These appeared in too few distinct games to calibrate a trustworthy range; their measured ranges are recorded (and kept pairwise non-overlapping so a fail-closed reader never *mislabels* them) but they are **harvest-excluded** until more games are gathered — a game whose seat is one of these should still emit a typed `hud_unreadable`/`player_colors_invalid` rejection, never a guess:

- **BLUE** — only 2 video(s).
- **ORANGE** — only 2 video(s).
- **GREEN** — only 1 video(s).
- **LIGHTGREEN** — only 1 video(s).
