# The eight Breakthrough Listen signals of interest of Ma et al. (2023) traced through their sessions

`ml_signals.tex` and `ml_signals.pdf` are the note; `mlc8_session.pdf` is its Figure 1. `make_numbers.py` prints every
number in the text and Table 1 from the files in `data/`, and draws Figure 1 from `data/MLc8_waterfall.npz`.

## Data

| File | Contents |
|------|----------|
| `data/MLc1.jsonl` to `data/MLc8.jsonl` | each signal's line (reference MJD, Hz from the tabulated frequency, drift, and the S/N of its ON tracks), then one record per scan of its session: target, cadence, ON or OFF, the power on the line with its parallel-track null, and the tracks above S/N 8 within 25 kHz |
| `data/table.json` | Table 1: each signal's line, its power on the line in the ON and OFF scans of its cadence, the OFF scans' combined significance, and the scans pointed at other stars that hold it |
| `data/listing_<session>.json` | the mid-resolution scans of each of the six sessions in the Breakthrough Listen pipeline directory |
| `data/MLc8_waterfall.npz` | every scan of AGBT16A_999_213 over the band MLc8's line crosses, unit-noise spectra as float16 with their MJDs |
| `data/stars.json` | the five stars' SIMBAD positions |

## Commands

Each result comes from `python -I seti.py breakthrough soi <mode>`, reading the 2.84 Hz products (`gpuspec.0000.h5`)
of the Breakthrough Listen open data by byte range.

| Result | Command |
|--------|---------|
| the traces | `soi trace` (all eight; `--tag MLc8` for one) |
| Table 1 | `soi table` |
| Figure 1's spectra | `soi waterfall --tag MLc8` |
| the store's record | `soi record` |
