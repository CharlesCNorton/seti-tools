# (883607) 2016 TA56: a candidate dark comet on a Jupiter-family orbit

`dark_comet.tex` and `dark_comet.pdf` are the paper; `cfht2007_cutouts.pdf` is its Figure 2. `make_numbers.py` prints every
number in the text and tables from the files in `data/`, the pgfplots code of Figures 1 and 3, Table S1 as
`data/table_S1_screen.csv`, and Figure 2 from the CFHT cutouts.

## Data

| File | Contents |
|------|----------|
| `data/sbdb_A2.json` | the JPL Small-Body Database query of 2026 October 16: `sbdb_query.api` with `sb-kind=a`, `sb-cdata={"AND":["A2\|DF"]}` (every asteroid whose orbit solution includes A2), `full-prec=true` and the fields of `yarkovsky_FIELDS` in `setitools/solar_system.py` |
| `data/yarkovsky_sign.json` | the screen: each term against the radiation limit and the DAMIT spin poles |
| `data/table_S1_screen.csv` | Table S1: all 611 asteroids of the screen |
| `data/2016_TA56/obs80.txt` | the MPC observations of 2016 TA56 used in the fits, as retrieved on 2026 October 16 |
| `data/2016_TA56/sbdb.json` | JPL orbit solution 23 with its covariance and physical parameters |
| `data/2016_TA56/neowise.json` | the NEOWISE single-exposure sources matched to the MPC's WISE positions, with Horizons geometry |
| `data/2016_TA56/neowise_detections.csv` | Table S2: the twenty NEOWISE detections, their W1 and W2 photometry, quality flags and use in the fit |
| `data/2016_TA56/thermal.json`, `thermal_variants.json` | the NEATM fits (Table 4) and the fit under other assumptions |
| `data/2016_TA56/refit.json` | the ASSIST orbit fits (Table 2, Figure 1) and each apparition against the gravity-only orbit of the others |
| `data/2016_TA56/robust.json` | night jackknife and bootstrap, clipping and weighting variants, and the 2030 prediction ellipses (Table 5) |
| `data/2016_TA56/blockcov.json` | A2 under the night-level covariance, and the error shared by the 2007 night that would remove it |
| `data/2016_TA56/precovery_20070914.json`, `frames_20070914.json` | the 2007 CFHT frames measured against Gaia DR3, and each against both orbits (Table 3) |
| `data/2016_TA56/field_20070914.json` | the known asteroids around the 2007 track from SkyBoT |
| `data/2016_TA56/locus_20070914.json` | the positions in each 2007 frame that orbits fitted to the later apparitions allow, A2 held at each value |
| `data/2016_TA56/census_20070914.json` | the moving sources found in the whole 2007 frames, the injected copies recovered, and the flux along the locus (Figure 3) |
| `data/2016_TA56/forecast.json`, `ephemeris_2030.csv` | the 2030 apparition: the A2 it would measure, and Table S3, daily positions of both orbits with their ellipses |

## Commands

Each result comes from `python -I seti.py solar_system <approach> <mode>`; the darkcomet modes take the designation, here
`"2016 TA56"`.

| Result | Command |
|--------|---------|
| the screen | `yarkovsky fetch`, `yarkovsky analyse`, `yarkovsky record` |
| observations, NEOWISE matches, JPL solution | `darkcomet fetch` |
| NEATM fit, its variants and Table S2 | `darkcomet thermal`, `darkcomet variants` |
| orbit fits, robustness, covariance | `darkcomet refit`, `darkcomet robust`, `darkcomet blockcov` (ASSIST with the DE440 and sb441-n16 ephemeris files) |
| archival images of the track | `darkcomet images` (CADC SSOIS) |
| the 2007 frames | `darkcomet precovery --night 2007-09-14`, `darkcomet frames --night 2007-09-14`, `darkcomet field --night 2007-09-14` |
| the 2007 locus and census | `darkcomet locus --night 2007-09-14`, then `darkcomet census --night 2007-09-14` with the sixteen frames from the CADC archive in `cfht_20070914/full` |
| 2030 | `darkcomet forecast` |
| record in `coverage.db` | `darkcomet record` |
