"""Sensitivities of the photometric paper: flash-detector scatter of Kepler short-cadence, TESS two-minute and 20 s samples as 5-sigma amplitudes and reflector areas, glint durations, and nightside limits against intercepted starlight."""
import csv
import glob
import io
import json
import math
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.argv = sys.argv[:1]
from setitools import core, science  # noqa: E402
from setitools import flashes, transits  # noqa: E402

RSUN = 6.957e8
AU = 1.495978707e11


def results(name):
    p = os.path.join(HERE, "data", name)
    return p if os.path.exists(p) else core.results(name)


def results_glob(pattern):
    """Files in data/ matching the pattern, else in the results directory."""
    return sorted(glob.glob(os.path.join(HERE, "data", pattern))) or sorted(glob.glob(core.results(pattern)))


def scatter_kepler(n=40, seed=3):
    """Robust scatter about a 30-minute running median, one short-cadence file each of n Kepler hosts."""
    from astropy.io import fits
    from scipy.ndimage import median_filter
    random.seed(seed)
    kics = [re.search(r'"kic": "(\d+)"', ln).group(1) for ln in open(results_glob("kepler_glints_*.jsonl")[0])]
    out = []
    for kic in random.sample(kics, min(3 * n, len(kics))):
        k = int(kic)
        prefix = "kepler/public/lightcurves/%04d/%09d/" % (k // 100000, k)
        b = core.get(flashes.kepler_S3 + "?list-type=2&prefix=" + prefix, timeout=60, retries=2, soft=True)
        keys = re.findall(r"<Key>([^<]*_slc\.fits)</Key>", b.decode()) if b else []
        fb = core.get(flashes.kepler_S3 + keys[0], timeout=120, retries=2, soft=True) if keys else None
        if not fb:
            continue
        with fits.open(io.BytesIO(fb)) as h:
            d = h[1].data
            t, f, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["SAP_QUALITY"], int)
            kp = h[0].header.get("KEPMAG")
        good = (q == 0) & np.isfinite(f) & np.isfinite(t)
        if good.sum() < 1000:
            continue
        r = f[good] / median_filter(f[good], size=31, mode="nearest") - 1.0
        out.append(dict(kic=k, kepmag=kp, sig_ppm=1e6 * float(1.4826 * np.median(np.abs(r - np.median(r))))))
        if len(out) >= n:
            break
    return out


def scatter_toi(n=40, seed=3):
    """Robust scatter of the two-minute flash detector, one sector each of n TOI hosts."""
    random.seed(seed)
    ts = [t for t in json.load(open(os.path.join(flashes.toi_ROOT, "targets.json"))) if t["sectors"]]
    out = []
    for t in random.sample(ts, min(3 * n, len(ts))):
        b = core.spoc_bytes(t["tic"], t["sectors"][0])
        if not b:
            continue
        try:
            tt, f, _ = core.read(b, columns=("PDCSAP_FLUX",), normalise=False)
        except Exception:
            continue
        if len(tt) < 1000:
            continue
        _, sig = science.flash_events(tt, f, flashes.toi_CAD, quiet=4)
        out.append(dict(tic=t["tic"], tmag=t["tmag"], sig_ppm=1e6 * float(sig)))
        if len(out) >= n:
            break
    return out


def scatter_tess20(n=150, seed=1):
    """Robust scatter of the 20 s flash detector over n cached light curves."""
    from astropy.io import fits
    files = sorted(glob.glob(os.path.join(flashes.tess20_DL, "*fast-lc.fits")))
    rng = np.random.default_rng(seed)
    out = []
    for f in (files[i] for i in rng.choice(len(files), min(n, len(files)), replace=False)):
        try:
            with fits.open(f) as h:
                d = h[1].data
                r = flashes.tess20_flashes(np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"], int))
                tic = int(h[0].header["TICID"])
        except Exception:
            continue
        if r is not None:
            out.append(dict(tic=tic, sig_ppm=1e6 * float(r[2])))
    return out


def glints():
    print("== glint sensitivity")
    for name, rows, rstar in (("Kepler short cadence", scatter_kepler(), 1.0), ("TESS two-minute", scatter_toi(), 0.8),
                              ("TESS 20 s", scatter_tess20(), 0.35)):
        s = np.array([x["sig_ppm"] for x in rows])
        amp = 5 * np.median(s) * 1e-6
        area = amp * math.pi * (rstar * RSUN) ** 2
        print("%s: %d curves, per-cadence scatter median %.0f ppm (10-90%% %.0f-%.0f); 5 sigma flash %.2f per cent; a perfect flat "
              "reflector of %.1e m2 (%.0f km square) around a star of %.2f solar radii" % (
                  name, len(s), np.median(s), *np.percentile(s, [10, 90]), 100 * amp, area, math.sqrt(area) / 1e3, rstar))
    for a_au in (0.05, 0.1, 1.0):
        P = 365.25 * a_au ** 1.5
        print("a mirror whose orientation follows its orbit at %.2f au (P %.1f d) sweeps its reflection of a solar disc across the "
              "observer in %.0f minutes" % (a_au, P, RSUN * P * 86400 / (2 * math.pi * a_au * AU) / 60))


def nightside():
    print("== nightside limits against intercepted starlight")
    toi = {r["TOI"]: r for r in csv.DictReader(open(os.path.join(transits.nightside_ROOT, "..", "tess_toi", "toi.csv"), encoding="utf-8", errors="replace"))} \
        if os.path.exists(os.path.join(transits.nightside_ROOT, "..", "tess_toi", "toi.csv")) else \
        {r["TOI"]: r for r in csv.DictReader(open(os.path.join(core.DATA, "tess_toi", "toi.csv"), encoding="utf-8", errors="replace"))}
    rows = []
    for p in results_glob("phase_curves_*.jsonl"):
        for line in open(p):
            d = json.loads(line)
            for t in d["tois"]:
                r = toi.get(t["toi"])
                if not r:
                    continue
                try:
                    S, R, T = float(r["Planet Insolation (Earth Flux)"]), float(r["Stellar Radius (R_Sun)"]), float(r["Stellar Eff Temp (K)"])
                    Rp = float(r["Planet Radius (R_Earth)"])
                except ValueError:
                    continue
                L = R ** 2 * (T / 5772.0) ** 4
                a_au = math.sqrt(L / S)
                rows.append(dict(toi=t["toi"], f_int_ppm=1e6 * (Rp * 6371.0 / (2 * a_au * 1.495978707e8)) ** 2,
                                 lim_ppm=3 * t["e_night_ppm"], night_W=3 * t["e_night_ppm"] * 1e-6 * L * 3.828e26))
    f = np.array([x["f_int_ppm"] for x in rows])
    lim = np.array([x["lim_ppm"] for x in rows])
    W = np.array([x["night_W"] for x in rows])
    print("%d planets: intercepted starlight (R_p/2a)^2 median %.1f ppm (%.1f-%.1f); 3 sigma nightside limit median %.0f ppm; "
          "limit over intercepted median %.0f (min %.1f); non-stellar power the limits allow median %.1e W (%.1e-%.1e)"
          % (len(rows), np.median(f), f.min(), f.max(), np.median(lim), np.median(lim / f), (lim / f).min(), np.median(W), W.min(), W.max()))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    nightside()
    glints()
