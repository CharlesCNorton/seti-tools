"""Numbers of the survey-laser paper: surviving LAMOST features with heights and mask bits, and per-star line-flux limits (Table 2) from the rvs, galah, apogee, desi and lamost results and cached magnitudes."""
import csv
import glob
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.argv = sys.argv[:1]
from setitools import core, science  # noqa: E402
from setitools import lasers  # noqa: E402

PC = 3.0857e18
# Vega zero points (erg/cm^2/s/A) at pivot wavelengths: Gaia BP, G, RP (SVO GAIA3) and 2MASS H (Cohen et al. 2003)
ZP = {"BP": (5110.0, 4.08e-9), "G": (6218.0, 2.50e-9), "RP": (7769.0, 1.27e-9), "H": (16620.0, 1.133e-10)}
# matched-filter thresholds in noise units
THR = {"rvs": 5.0, "galah": 5.5, "apogee": 5.0, "desi": 5.0, "lamost": 6.0}
GALAH_CAMS = {"B": (4810.0, 0.046, "snr_px_ccd1"), "V": (5760.0, 0.055, "snr_px_ccd2"), "R": (6610.0, 0.063, "snr_px_ccd3"),
              "I": (7740.0, 0.074, "snr_px_ccd4")}
DESI_CAMS = {"b": (4800.0, 2500.0, "sn_b"), "r": (6800.0, 3500.0, "sn_r"), "z": (8800.0, 4500.0, "sn_z")}


def results(name):
    """Result file from data/, else the results directory."""
    p = os.path.join(HERE, "data", name)
    return p if os.path.exists(p) else core.results(name)


def chunks(xs, n=60):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def fetch_mags():
    """Gaia magnitudes of the five samples and 2MASS H of the APOGEE stars from the Astro Data Lab, cached in RESULTS."""
    p = results("survey_mags.json")
    if os.path.exists(p):
        M = json.load(open(p))
        M["twomass"] = {k.strip(): v for k, v in M.get("twomass", {}).items()}
        return M
    out = {}

    def gaia(ids):
        got = {}
        for ch in chunks(sorted(set(ids))):
            q = ("SELECT source_id, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, grvs_mag, parallax FROM "
                 "gaia_dr3.gaia_source WHERE source_id IN (%s)" % ",".join(ch))
            for r in core.datalab(q):
                got[str(r["source_id"])] = r
        return got
    rvs = [r["source_id"] for r in csv.DictReader(open(core.results("rvs_sample.csv")))]
    galah = [r["gaiadr3_source_id"] for r in csv.DictReader(open(core.results("galah_sample.csv")))]
    desi = [r["source_id"] for r in csv.DictReader(open(core.results("desi_sample.csv"))) if r["source_id"]]
    lamost = list(json.load(open(os.path.join(lasers.lamost_ROOT, "targets.json"))).keys())
    ap = {r["APOGEE"]: r for r in csv.DictReader(open(os.path.join(core.DATA, "apogee_100pc_dwarfs.csv")))}
    apg = [r["GaiaEDR3"] for r in ap.values() if r.get("GaiaEDR3")]
    for name, ids in (("rvs", rvs), ("galah", galah), ("desi", desi), ("lamost", lamost), ("apogee", apg)):
        out[name] = gaia(ids)
        print("%s: %d of %d stars with Gaia magnitudes" % (name, len(out[name]), len(set(ids))), flush=True)
    hm = {}
    for ch in chunks([a[2:] for a in ap if a.startswith("2M")]):
        q = "SELECT designation, h_m, j_m, k_m FROM twomass.psc WHERE designation IN (%s)" % ",".join("'%s'" % x for x in ch)
        for r in core.datalab(q):
            hm[r["designation"].strip()] = r
    out["twomass"] = hm
    json.dump(out, open(core.results("survey_mags.json"), "w"))
    return out


def flam(mags, lam):
    """F_lambda (erg/cm^2/s/A) at lam (A), log-linear through the available bands, extrapolated beyond."""
    pts = sorted((ZP[b][0], ZP[b][1] * 10 ** (-0.4 * m)) for b, m in mags.items() if m is not None and np.isfinite(m))
    if not pts:
        return None
    if len(pts) == 1:
        return float(pts[0][1])
    x = np.log10([p[0] for p in pts])
    y = np.log10([p[1] for p in pts])
    i = min(max(int(np.searchsorted(x, math.log10(lam))) - 1, 0), len(pts) - 2)
    slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return float(10 ** (y[i] + slope * (math.log10(lam) - x[i])))


def gain(sigma_px):
    k = np.arange(-int(4 * sigma_px) - 1, int(4 * sigma_px) + 2)
    return float(math.sqrt(np.sum(np.exp(-0.5 * (k / sigma_px) ** 2) ** 2)))


def limit(k, sig_pix, sigma_px, lam, R, F):
    """Unresolved-line flux at k times the matched-filter noise: k sig_pix over the filter gain, times the Gaussian integral, times the continuum."""
    return k * sig_pix / gain(sigma_px) * math.sqrt(2 * math.pi) * (lam / R / 2.3548) * F


def row(sid, d_pc, F, lam, extra=None):
    d = d_pc * PC
    return dict(id=str(sid), d_pc=d_pc, F=F, L=4 * math.pi * d ** 2 * F * 1e-7, P10=F * d ** 2 * (lam * 1e-8 / 1000.0) ** 2 * 1e-7,
                **(extra or {}))


def gmags(table, sid):
    r = table.get(str(sid))
    if not r:
        return None, None

    def f(k):
        try:
            return float(r[k])
        except (TypeError, ValueError, KeyError):
            return None
    return dict(BP=f("phot_bp_mean_mag"), G=f("phot_g_mean_mag"), RP=f("phot_rp_mean_mag")), f("parallax")


def limits(M):
    out = {}
    z = np.load(core.results("rvs_laser_resid.npz"), allow_pickle=True)
    rows = []
    sp = 858.0 / 11500.0 / 2.3548 / 0.01
    for sid, r in zip(z["ids"], z["resid"]):
        ok = np.isfinite(r)
        mags, plx = gmags(M["rvs"], sid)
        if ok.sum() < 500 or not mags or not plx or plx <= 0:
            continue
        x = r[ok]
        sig = 1.4826 * np.median(np.abs(x - np.median(x)))
        F = flam(mags, 8580.0)
        if F:
            rows.append(row(sid, 1000.0 / plx, limit(THR["rvs"], sig, sp, 8580.0, 11500.0, F), 8580.0))
    out["rvs"] = rows
    rows = []
    for r in csv.DictReader(open(core.results("galah_sample.csv"))):
        mags, plx = gmags(M["galah"], r["gaiadr3_source_id"])
        if not mags or not plx or plx <= 0:
            continue
        for cam, (lam, step, col) in GALAH_CAMS.items():
            try:
                snr = float(r[col])
            except (ValueError, KeyError):
                continue
            F = flam(mags, lam)
            if snr > 0 and F:
                rows.append(row(r["gaiadr3_source_id"], 1000.0 / plx, limit(THR["galah"], 1.0 / snr, lam / 28000.0 / 2.3548 / step, lam, 28000.0, F), lam, dict(cam=cam)))
    out["galah"] = rows
    rows = []
    sp = science.lsf_sigma_px(22500.0, dlog10=6e-6)
    for r in csv.DictReader(open(core.results("apogee_sample.csv"))):
        tm = M["twomass"].get(r["apogee_id"][2:])
        try:
            snr, plx = float(r["snr"]), float(r["gaiaedr3_parallax"])
            H = float(tm["h_m"])
        except (TypeError, ValueError, KeyError):
            continue
        if snr > 0 and plx > 0:
            F = ZP["H"][1] * 10 ** (-0.4 * H)
            rows.append(row(r["apogee_id"], 1000.0 / plx, limit(THR["apogee"], 1.0 / snr, sp, 16000.0, 22500.0, F), 16000.0))
    out["apogee"] = rows
    rows = []
    for r in csv.DictReader(open(core.results("desi_sample.csv"))):
        mags, _ = gmags(M["desi"], r["source_id"]) if r["source_id"] else (None, None)
        try:
            plx = float(r["parallax"])
        except ValueError:
            continue
        if not mags or plx <= 0:
            continue
        for cam, (lam, Rc, col) in DESI_CAMS.items():
            try:
                snr = float(r[col])
            except (ValueError, KeyError):
                continue
            F = flam(mags, lam)
            if snr > 0 and F:
                rows.append(row(r["targetid"], 1000.0 / plx, limit(THR["desi"], 1.0 / snr, lam / Rc / 2.3548 / 0.8, lam, Rc, F), lam, dict(cam=cam)))
    out["desi"] = rows
    # LAMOST per-exposure S/N from the headers of 1500 random cached observations
    from astropy.io import fits
    random.seed(5)
    files = sorted(glob.glob(os.path.join(lasers.lamost_ROOT, "fits", "*.fits.gz")))
    targets = json.load(open(os.path.join(lasers.lamost_ROOT, "targets.json")))
    by_obs = {str(o): sid for sid, t in targets.items() for o in t["obsids"]}
    rows = []
    for f in random.sample(files, min(1500, len(files))):
        sid = by_obs.get(os.path.basename(f).split(".")[0])
        if not sid:
            continue
        mags, _ = gmags(M["lamost"], sid)
        try:
            plx = float(targets[sid]["plx"])
        except (TypeError, ValueError):
            continue
        if not mags or plx <= 0:
            continue
        try:
            with fits.open(f) as h:
                for x in h[1:]:
                    if x.name[:2] not in ("B-", "R-") or not x.header.get("SNR") or float(x.header["SNR"]) <= 0:
                        continue
                    lam = 5150.0 if x.name[0] == "B" else 6550.0
                    w = np.asarray(x.data["WAVELENGTH"][0], float)
                    step = float(np.median(np.diff(w)))
                    F = flam(mags, lam)
                    if F:
                        rows.append(row(sid, 1000.0 / plx, limit(THR["lamost"], 1.0 / float(x.header["SNR"]), lam / 7500.0 / 2.3548 / step, lam, 7500.0, F), lam, dict(arm=x.name[0])))
        except Exception:
            continue
    out["lamost"] = rows
    json.dump(out, open(core.results("survey_limits.json"), "w"))
    return out


def table_limits(out):
    print("== Table 2 rows")
    names = {"rvs": "Gaia DR3 RVS", "galah": "GALAH DR4", "apogee": "APOGEE DR17", "desi": "DESI DR1", "lamost": "LAMOST DR10 MRS"}
    for k, rows in out.items():
        if not rows:
            print("%s: no limits (sample or magnitudes missing)" % names[k])
            continue
        F = np.array([r["F"] for r in rows])
        L = np.array([r["L"] for r in rows])
        P = np.array([r["P10"] for r in rows])
        d = np.array([r["d_pc"] for r in rows])
        near = d < 20
        i = int(np.argmin(d))
        print(r"%s & $%d$ & $%.1f\times10^{%d}$ & $%.1f\times10^{%d}$ & $%.0f\times10^{%d}$ & $%d$ & $%.2g$ ($%.1f$\,pc) \\"
              % (names[k], len({r["id"] for r in rows}), *sci(np.median(F)), *sci(np.median(L)), *sci(np.median(P), 0), near.sum(),
                 P[i], d[i]))
        print("   %% 10-90 per cent of F %.1e-%.1e; within 20 pc isotropic median %.1e W, 10 m %.1e W" % (
            *np.percentile(F, [10, 90]), np.median(L[near]) if near.any() else 0, np.median(P[near]) if near.any() else 0))


def sci(x, digits=1):
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return (round(m, digits), e)


def table_lamost():
    """Table 1 rows: LAMOST features passing every test, with height over the continuum and pipeline masks."""
    from astropy.io import fits
    d = json.load(open(results("lamost_laser.json")))
    targets = json.load(open(os.path.join(lasers.lamost_ROOT, "targets.json")))
    left = [p for p in d["persistent"] if not p["why"]]
    print("== Table 1 rows (%d features)" % len(left))
    for p in sorted(left, key=lambda p: p["rest"]):
        t = targets.get(p["star"], {})
        heights = []
        for obsid in p["obsids"]:
            try:
                h = fits.open(os.path.join(lasers.lamost_ROOT, "fits", "%s.fits.gz" % obsid))
            except Exception:
                continue
            for x in h[1:]:
                if not x.name.startswith(p["arm"] + "-"):
                    continue
                w = np.asarray(x.data["WAVELENGTH"][0], float)
                fl = np.asarray(x.data["FLUX"][0], float)
                j = int(np.argmin(np.abs(w - p["obs"])))
                cont = np.median(fl[max(j - 150, 0):j + 150])
                heights.append(fl[j] / cont if cont else float("nan"))
        print(r"%s & %s & %.2f & %.2f & %d/%d & %d & %.1f \\  %% peak/continuum up to %.2f in %d exposures" % (
            p["star"], p["arm"], p["rest"], p["obs"], p["n"], t.get("exposures", 0), p["nights"], p["z"],
            max(heights) if heights else float("nan"), len(heights)))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    table_lamost()
    M = fetch_mags()
    table_limits(limits(M))
