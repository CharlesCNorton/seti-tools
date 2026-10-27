"""Laser lines in archival optical and infrared spectra of nearby stars."""
import argparse
import collections
import csv
import glob
import io
import json
import math
import os
import re
import struct
import sys
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import nnls

from setitools import core, science
from setitools.core import num, quiet_warnings, sep_arcsec


# --------------------------------------------------------------------------------------------------------------

rvs_DOC = r"""Laser lines in Gaia DR3 RVS spectra of nearby FGK stars: neighbour-model residuals above all absorption-like peaks, vetted for stellar cores, sharing and width.

    python -I seti.py lasers rvs [--k 25] [--min-snr 4]
    python -I seti.py lasers rvs fetch [--plx 20]"""

rvs_HERE = core.RESULTS
rvs_W = np.round(np.arange(846.0, 870.0001, 0.01), 2)
rvs_CA = (849.802, 854.209, 866.214)
rvs_PASCHEN = (846.72, 850.25, 854.54, 859.82, 866.50)
rvs_C = 299792.458
rvs_SIG_PX = 858.0 / 11500.0 / 2.3548 / 0.01
rvs_USE = (rvs_W > 846.4) & (rvs_W < 869.6)
rvs_SEARCH = (rvs_W > 846.6) & (rvs_W < 869.4)


def rvs_local_sigma(r, ok, half=100, step=20):
    """Robust residual scatter around each sample from valid samples within half."""
    centres = np.arange(0, len(r), step)
    vals = []
    for c in centres:
        m = ok[max(c - half, 0):c + half + 1]
        x = r[max(c - half, 0):c + half + 1][m]
        vals.append(1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 20 else np.nan)
    vals = np.array(vals)
    good = np.isfinite(vals)
    return np.interp(np.arange(len(r)), centres[good], vals[good])


def rvs_load_sample():
    return {r["source_id"]: r for r in csv.DictReader(open(os.path.join(rvs_HERE, "rvs_sample.csv")))}


def rvs_load_spec(path):
    f = np.full(len(rvs_W), np.nan)
    e = np.full(len(rvs_W), np.nan)
    meta = {}
    for r in csv.DictReader(open(path)):
        i = int(round((float(r["wavelength"]) - 846.0) / 0.01))
        if r["flux"]:
            f[i] = float(r["flux"])
            e[i] = float(r["flux_error"]) if r["flux_error"] else np.nan
        meta = dict(transits=int(r["combined_transits"] or 0), ccds=int(r["combined_ccds"] or 0),
                    deblended=int(r["deblended_ccds"] or 0))
    return f, e, meta


def rvs_fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def rvs_peaks(snr, thr, sep=10):
    idx = np.nonzero((snr > thr) & rvs_SEARCH)[0]
    out = []
    for i in idx[np.argsort(-snr[idx])]:
        if all(abs(i - j) >= sep for j in out):
            out.append(int(i))
    return out


def rvs_gauss_width(r, i, half=15):
    """Best Gaussian sigma (samples) of the residual around sample i."""
    lo, hi = max(i - half, 0), min(i + half + 1, len(r))
    x = np.arange(lo, hi) - i
    y = r[lo:hi]
    ok = np.isfinite(y)
    best, bs = np.inf, np.nan
    for s in np.arange(1.0, 9.01, 0.25):
        g = np.exp(-0.5 * (x / s) ** 2)
        a = np.dot(g[ok], y[ok]) / np.dot(g[ok], g[ok])
        chi = np.sum((y[ok] - a * g[ok]) ** 2)
        if chi < best:
            best, bs = chi, s
    return bs


def rvs_main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--min-snr", type=float, default=4.0)
    a = p.parse_args()
    sample = rvs_load_sample()
    ids, F, E, META = [], [], [], []
    for path in sorted(glob.glob(os.path.join(rvs_HERE, "rvs_dl", "RVS_Gaia_DR3_*.csv"))):
        sid = os.path.basename(path)[13:-4]
        if sid not in sample:
            continue
        f, e, m = rvs_load_spec(path)
        if np.isfinite(f[rvs_USE]).mean() < 0.95:
            continue
        ids.append(sid)
        F.append(f)
        E.append(e)
        META.append(m)
    F, E = np.array(F), np.array(E)
    n = len(ids)
    s = [sample[i] for i in ids]
    col = np.array([rvs_fnum(r["bp_rp"]) for r in s])
    mg = np.array([rvs_fnum(r["phot_g_mean_mag"]) + 5 * np.log10(rvs_fnum(r["parallax"]) / 100.0) for r in s])
    rv = np.array([rvs_fnum(r["radial_velocity"]) for r in s])
    print("%d spectra" % n, flush=True)
    x = (rvs_W - 858.0) / 12.0
    poly = np.array([x ** k * sgn for k in range(4) for sgn in (1.0, -1.0)])
    kern = np.exp(-0.5 * (np.arange(-13, 14) / rvs_SIG_PX) ** 2)
    allpk = []
    resid = {}
    for i in range(n):
        d = np.hypot((col - col[i]) / 0.05, (mg - mg[i]) / 0.3)
        d[i] = np.inf
        nb = np.argsort(d)[:a.k]
        basis = np.nan_to_num(F[nb], nan=1.0)
        ok = rvs_USE & np.isfinite(F[i])
        A = np.vstack([basis, poly]).T
        coef, _ = nnls(A[ok], F[i][ok])
        model0 = A @ coef
        d1 = np.gradient(model0)
        d2 = np.gradient(d1)
        d1 /= max(np.abs(d1[ok]).max(), 1e-9)
        d2 /= max(np.abs(d2[ok]).max(), 1e-9)
        A = np.vstack([basis, poly, d1, -d1, d2, -d2]).T
        coef, _ = nnls(A[ok], F[i][ok])
        model = A @ coef
        tmpl = basis.T @ coef[:len(nb)] / max(coef[:len(nb)].sum(), 1e-9)
        r = np.where(ok, F[i] - model, 0.0)
        sig = np.maximum(rvs_local_sigma(r, ok), 1e-5)
        w = np.where(ok, 1.0 / sig ** 2, 0.0)
        num = np.convolve(r * w, kern, mode="same")
        den = np.convolve(w, kern ** 2, mode="same")
        snr = num / np.sqrt(np.maximum(den, 1e-30))
        amp = num / np.maximum(den, 1e-30)
        resid[ids[i]] = r
        for sign in (1, -1):
            for j in rvs_peaks(sign * snr, a.min_snr):
                env = "line" if tmpl[j] < 0.95 else "continuum"
                lam = float(rvs_W[j])
                tags = []
                if any(abs(lam - c) < 0.25 for c in rvs_CA):
                    tags.append("CaII")
                if any(abs(lam - c) < 0.20 for c in rvs_PASCHEN):
                    tags.append("Paschen")
                allpk.append(dict(sid=ids[i], sign=sign, lam=lam, snr=round(float(sign * snr[j]), 2),
                                  amp=float(sign * amp[j]), ew_pm=round(float(sign * amp[j]) * rvs_SIG_PX * 0.01 *
                                                                         np.sqrt(2 * np.pi) * 1000, 3),
                                  env=env, tmpl=round(float(tmpl[j]), 3), tags=tags,
                                  lam_obs=lam * (1 + rv[i] / rvs_C) if np.isfinite(rv[i]) else np.nan,
                                  width=round(rvs_gauss_width(sign * r, j) / rvs_SIG_PX, 2),
                                  deblended=META[i]["deblended"], transits=META[i]["transits"],
                                  clean=bool(rvs_fnum(s[i]["ruwe"]) < 1.4 and s[i]["non_single_star"] in ("0", "")
                                             and META[i]["deblended"] <= 0.5 * max(META[i]["ccds"], 1))))
    span = float(rvs_W[rvs_SEARCH][-1] - rvs_W[rvs_SEARCH][0])
    for sign in (1, -1):
        grp = [q for q in allpk if q["sign"] == sign]
        ref = [q for q in grp if q["snr"] > 5.0]
        rate = len(ref) * 0.04 / span
        limit = max(2.0, rate + 4.0 * np.sqrt(rate))
        rest = np.array([q["lam"] for q in ref])
        obs = np.array([q["lam_obs"] for q in ref])
        sids = np.array([q["sid"] for q in ref])
        for q in grp:
            other = sids != q["sid"]
            q["rest_twins"] = int(np.sum(other & (np.abs(rest - q["lam"]) < 0.02)))
            q["obs_twins"] = (int(np.sum(other & (np.abs(obs - q["lam_obs"]) < 0.02)))
                              if np.isfinite(q["lam_obs"]) else 0)
            q["twin_limit"] = round(float(limit), 1)
            q["isolated"] = bool(not q["tags"] and q["rest_twins"] <= limit and q["obs_twins"] <= limit
                                 and 0.6 <= q["width"] <= 1.6 and q["clean"])
        print("%s peaks: %d, chance twins per 0.04 nm %.1f, shared above %.1f"
              % ("emission-like" if sign > 0 else "absorption-like", len(grp), rate, limit))
    json.dump(allpk, open(os.path.join(rvs_HERE, "rvs_laser_peaks.json"), "w"))
    np.savez_compressed(os.path.join(rvs_HERE, "rvs_laser_resid.npz"), ids=np.array(ids),
                        resid=np.array([resid[i] for i in ids]))
    null = {}
    for env in ("continuum", "line"):
        negs = sorted((q["snr"] for q in allpk if q["sign"] < 0 and q["env"] == env and q["isolated"]), reverse=True)
        null[env] = negs[0] if negs else 0.0
        print("%s: isolated absorption-like peaks, strongest %s" % (env, ", ".join("%.1f" % v for v in negs[:5])))
        for thr in (4, 5, 6, 8, 10):
            npos = sum(1 for q in allpk if q["sign"] > 0 and q["env"] == env and q["isolated"] and q["snr"] > thr)
            nneg = sum(1 for q in allpk if q["sign"] < 0 and q["env"] == env and q["isolated"] and q["snr"] > thr)
            print("  SNR>%d: isolated emission-like %d, absorption-like %d" % (thr, npos, nneg))
    cands = sorted((q for q in allpk if q["sign"] > 0 and q["isolated"]), key=lambda q: -q["snr"])
    print("strongest isolated emission-like peaks:")
    for q in cands[:25]:
        print("  %s%s  %.2f nm  SNR %.1f  EW %.2f pm  %s (template %.2f)  width %.2f  deblended %d/%d transits"
              % ("ABOVE-NULL " if q["snr"] > null[q["env"]] else "", q["sid"], q["lam"], q["snr"], q["ew_pm"],
                 q["env"], q["tmpl"], q["width"], q["deblended"], q["transits"]))


# fetch mode

rvs_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"

rvs_DL = "https://gea.esac.esa.int/data-server/data"

rvs_OUT = os.path.join(rvs_HERE, "rvs_dl")

def rvs_fetch_main():
    p = argparse.ArgumentParser()
    p.add_argument("--plx", type=float, default=20.0)
    a = p.parse_args()
    os.makedirs(rvs_OUT, exist_ok=True)
    q = ("SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp, grvs_mag, radial_velocity, "
         "rvs_spec_sig_to_noise, vbroad, ruwe, rv_template_teff, rv_template_logg, rv_template_fe_h, "
         "non_single_star FROM gaiadr3.gaia_source WHERE parallax > %g AND has_rvs = 'True' "
         "AND bp_rp BETWEEN 0.55 AND 1.6" % a.plx)
    d = urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q)).encode()
    t = urllib.request.urlopen(rvs_TAP, data=d, timeout=300).read().decode()
    open(os.path.join(rvs_HERE, "rvs_sample.csv"), "w").write(t)
    ids = [line.split(",")[0] for line in t.strip().splitlines()[1:]]
    todo = [i for i in ids if not os.path.exists(os.path.join(rvs_OUT, "RVS_Gaia_DR3_%s.csv" % i))]
    print("%d stars, %d to fetch" % (len(ids), len(todo)), flush=True)
    for k in range(0, len(todo), 50):
        batch = todo[k:k + 50]
        u = rvs_DL + "?" + urllib.parse.urlencode(dict(RETRIEVAL_TYPE="RVS", RELEASE="Gaia DR3",
                                                   DATA_STRUCTURE="DATAMODEL_STANDARD", FORMAT="csv",
                                                   ID=",".join("Gaia DR3 " + i for i in batch)))
        for attempt in range(4):
            try:
                b = urllib.request.urlopen(u, timeout=300).read()
                break
            except Exception as e:
                print("  retry %d (%s)" % (attempt + 1, str(e)[:80]), flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            continue
        if not b.startswith(b"PK"):
            if len(batch) == 1 and b.startswith(b"source_id"):
                open(os.path.join(rvs_OUT, "RVS_Gaia_DR3_%s.csv" % batch[0]), "wb").write(b)
            continue
        z = zipfile.ZipFile(io.BytesIO(b))
        for n in z.namelist():
            base = os.path.basename(n)
            if base.startswith("RVS_Gaia_DR3_") and base.endswith(".csv"):
                open(os.path.join(rvs_OUT, base), "wb").write(z.read(n))
        print("  %d/%d" % (min(k + 50, len(todo)), len(todo)), flush=True)


def rvs_cli():
    if len(sys.argv) > 1 and sys.argv[1] == 'fetch':
        sys.argv.pop(1)
        rvs_fetch_main()
    else:
        rvs_main()


# --------------------------------------------------------------------------------------------------------------

eso_DOC = r"""Laser lines in ESO HARPS, ESPRESSO, UVES, FEROS (and SOPHIE) spectra: recurring barycentric peaks across epochs and steady peaks against twin-star fits, with absorption nulls.

    python -I seti.py lasers eso [--inst HARPS|ESPRESSO|UVES|FEROS|SOPHIE] [--root DIR] [--stars HIP103039 ...] [--snr 6]
    python -I seti.py lasers eso fetch --n 12 "HIP 103039" "HIP 22762" ...
    python -I seti.py lasers eso record [--inst ...] [--tag _fgk]       (the run's groups and peaks into the coverage store)
    python -I seti.py lasers eso cores [--record]       (the unresolved peaks inside chromospheric and Balmer cores, every run)"""

eso_C = 299792.458
eso_ROOT = core.data("harps_dl")
eso_CHROMO = (3933.66, 3968.47, 3970.07, 4101.73, 4340.46, 4861.33, 5875.62, 5889.95, 5895.92, 6562.80, 3889.05,
          3835.38, 3797.90, 8498.02, 8542.09, 8662.14)
# range (A), resolving power, grid step (A), detector gaps and site per spectrograph
eso_INST = {"HARPS": dict(lo=3800.0, hi=6900.0, R=115000.0, step=0.01, gaps=((5290.0, 5350.0),), site="LaSilla"),
        "ESPRESSO": dict(lo=3800.0, hi=7880.0, R=140000.0, step=0.01, gaps=(), site="Paranal"),
        "UVES": dict(lo=3300.0, hi=10400.0, R=45000.0, step=0.025, gaps=(), site="Paranal"),
        "FEROS": dict(lo=3550.0, hi=9200.0, R=48000.0, step=0.025, gaps=(), site="LaSilla"),
        "SOPHIE": dict(lo=3880.0, hi=6940.0, R=75000.0, step=0.01, gaps=(), site="OHP")}
eso_GRID = np.round(np.arange(3800.0, 6900.0, 0.01), 3)
eso_R = 115000.0
eso_STEP = 0.01
eso_GAPS = eso_INST["HARPS"]["gaps"]
eso_SITE = "LaSilla"
eso_TELLURIC = ((5880.0, 5990.0), (6270.0, 6330.0), (6460.0, 6600.0), (6860.0, 6960.0), (7160.0, 7340.0),
            (7590.0, 7700.0), (8120.0, 8350.0), (8950.0, 9850.0))
eso_AIRGLOW = (4046.56, 4358.33, 5197.90, 5200.26, 5460.74, 5577.34, 5769.60, 5790.66, 5889.95, 5895.92, 6300.30,
           6363.78)


def eso_set_instrument(name):
    """Set the grid, resolving power, gaps and site for one spectrograph."""
    global eso_GRID, eso_R, eso_STEP, eso_GAPS, eso_SITE
    p = eso_INST[name]
    eso_STEP, eso_R, eso_GAPS, eso_SITE = p["step"], p["R"], p["gaps"], p["site"]
    eso_GRID = np.round(np.arange(p["lo"], p["hi"], eso_STEP), 3)


def eso_load(path):
    """Flux and error on GRID (air, barycentric) with the epoch's BERV (km/s)."""
    from astropy.io import fits
    with fits.open(path) as h:
        H = h[0].header
        if "OHP DRS BERV" in H:
            return sophie_load(h)
        t = h[1].data
        names = [n.upper() for n in t.columns.names]
        col = {n.upper(): n for n in t.columns.names}
        wname = "WAVE_AIR" if "WAVE_AIR" in names else "WAVE"
        ucd = str(h[1].header.get("TUCD%d" % (names.index(wname) + 1), "obs.atmos"))
        w = np.asarray(t[col[wname]][0], float)
        f = np.asarray(t[col["FLUX"]][0] if "FLUX" in col else t[t.columns.names[1]][0], float)
        e = np.asarray(t[col["ERR"]][0], float) if "ERR" in col else np.full(len(f), np.nan)
        spec = str(H.get("SPECSYS", h[1].header.get("SPECSYS", ""))).upper()
        mjd = float(H.get("MJD-OBS", np.nan)) + float(H.get("EXPTIME", 0)) / 172800.0
        berv = np.nan
        for k in ("HIERARCH ESO DRS BERV", "HIERARCH ESO QC BERV", "HIERARCH ESO DRS BERV CORR"):
            if k in H:
                berv = float(H[k])
                break
        if not np.isfinite(berv):
            berv = science.barycentric_correction(float(H.get("RA")), float(H.get("DEC")), mjd, eso_SITE)
        meta = dict(berv=berv, mjd=float(H.get("MJD-OBS", np.nan)), snr=float(H.get("SNR", np.nan)),
                    dpr=str(H.get("HIERARCH ESO DPR TYPE", "")), specsys=spec, file=os.path.basename(path))
    if w.max() < 100:
        w = w * 10.0 if w.max() > 10 else w * 1e4
    if "obs.atmos" not in ucd and wname != "WAVE_AIR":
        w = science.vac_to_air(w)
    if not spec.startswith("BARY"):
        w = w * (1.0 + berv / eso_C)
    fi = np.interp(eso_GRID, w, f, left=np.nan, right=np.nan)
    ei = np.interp(eso_GRID, w, e, left=np.nan, right=np.nan)
    return fi, ei, meta


def eso_px(angstrom):
    """Grid samples spanning a wavelength interval."""
    return max(int(round(angstrom / eso_STEP)), 1)


def eso_envelope(f, block=None, q=90):
    """Running upper envelope: block percentiles over 5 A, interpolated and smoothed."""
    block = block or eso_px(5.0)
    n = len(f)
    cen, val = [], []
    for s in range(0, n, block):
        x = f[s:s + block]
        x = x[np.isfinite(x)]
        if len(x) > block // 4:
            cen.append(s + block // 2)
            val.append(np.percentile(x, q))
    if not cen:
        return np.full(n, np.nan)
    env = np.interp(np.arange(n), cen, val)
    k = np.ones(5) / 5.0
    return np.convolve(np.pad(env, 2, mode="edge"), k, mode="valid")


def eso_block_mad(res, block=None, floor=2e-3):
    """Blockwise robust residual scatter over 5 A, interpolated."""
    block = block or eso_px(5.0)
    n = len(res)
    cen, val = [], []
    for s in range(0, n, block):
        x = res[s:s + block]
        x = x[np.isfinite(x)]
        if len(x) > block // 4:
            cen.append(s + block // 2)
            val.append(1.4826 * np.median(np.abs(x - np.median(x))))
    if not cen:
        return np.full(n, np.inf)
    return np.maximum(np.interp(np.arange(n), cen, val), floor)


def eso_smooth_ratio(r, size=None):
    from scipy.ndimage import median_filter
    x = np.where(np.isfinite(r), r, 1.0)
    return median_filter(x, size=size or (eso_px(3.0) | 1), mode="nearest")


def eso_matched(res, sig):
    """Matched-filter significance and amplitude in 100 A chunks."""
    snr = np.zeros_like(res)
    amp = np.zeros_like(res)
    w = np.where(np.isfinite(res) & np.isfinite(sig) & (sig > 0), 1.0 / np.maximum(sig, 1e-9) ** 2, 0.0)
    r0 = np.where(w > 0, res, 0.0)
    ch = eso_px(100.0)
    for s in range(0, len(eso_GRID), ch):
        lam = eso_GRID[min(s + ch // 2, len(eso_GRID) - 1)]
        sp = lam / eso_R / 2.3548 / eso_STEP
        x = np.arange(-int(4 * sp) - 1, int(4 * sp) + 2)
        g = np.exp(-0.5 * (x / sp) ** 2)
        lo, hi = max(s - len(x), 0), min(s + ch + len(x), len(eso_GRID))
        num = np.convolve(r0[lo:hi] * w[lo:hi], g, mode="same")
        den = np.convolve(w[lo:hi], g * g, mode="same")
        a, b = s - lo, s - lo + min(ch, len(eso_GRID) - s)
        snr[s:s + ch] = (num / np.sqrt(np.maximum(den, 1e-30)))[a:b]
        amp[s:s + ch] = (num / np.maximum(den, 1e-30))[a:b]
    return snr, amp


def eso_width_ratio(res, j):
    lam = eso_GRID[j]
    sp = lam / eso_R / 2.3548 / eso_STEP
    half = int(5 * sp) + 2
    lo, hi = max(j - half, 0), min(j + half + 1, len(res))
    x = np.arange(lo, hi) - j
    y = res[lo:hi]
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    best, bs = np.inf, np.nan
    for s in np.arange(0.4, 4.01, 0.1) * sp:
        g = np.exp(-0.5 * (x / s) ** 2)
        a = np.dot(g[ok], y[ok]) / np.dot(g[ok], g[ok])
        chi = np.sum((y[ok] - a * g[ok]) ** 2)
        if chi < best:
            best, bs = chi, s
    return bs / sp


def eso_peaks(snr, thr, sep=None):
    """Samples above thr, strongest first, at least sep (0.2 A) apart."""
    sep = sep or eso_px(0.2)
    idx = np.nonzero(snr > thr)[0]
    out = []
    blocked = np.zeros(len(snr), bool)
    for i in idx[np.argsort(-snr[idx])]:
        if not blocked[i]:
            out.append(int(i))
            blocked[max(i - sep + 1, 0):i + sep] = True
        if len(out) > 2000:
            break
    return out


def eso_narrow(q):
    """Unresolved peak: 0.6-1.6 times the instrumental width, and at least 0.75 of it at 10 sigma."""
    return 0.6 <= q["width"] <= 1.6 and not (q["snr"] >= 10.0 and q["width"] < 0.75)


def eso_steady_clean(q, R):
    """Steady peak no filter explains: unresolved, off chromospheric lines, unshared, clear of sky and star lines, with BERV spanning a resolution element."""
    return (eso_narrow(q) and not q["chromospheric"] and q["shared"] == 0 and not q.get("sky")
            and q.get("depth", 0.0) <= 0.1 and not q.get("second_fibre")
            and q.get("berv_spread", np.inf) >= eso_C / R)


def eso_star_spectra(d):
    """A star's spectra on the grid, one per exposure, keeping the higher S/N of duplicates."""
    got = []
    for path in sorted(glob.glob(os.path.join(d, "*.fits"))):
        try:
            f, e, m = eso_load(path)
        except Exception:
            continue
        if np.isfinite(m["berv"]):
            got.append((f, e, m))
    got.sort(key=lambda x: -(x[2]["snr"] if np.isfinite(x[2]["snr"]) else 0.0))
    keep, seen = [], set()
    for f, e, m in got:
        key = round(m["mjd"], 4) if np.isfinite(m["mjd"]) else m["file"]
        if key not in seen:
            seen.add(key)
            keep.append((f, e, m))
    keep.sort(key=lambda x: x[2]["file"])
    return np.array([k[0] for k in keep]), np.array([k[1] for k in keep]), [k[2] for k in keep]


def eso_valid_mask(F, margin=None):
    """Samples usable in every epoch, away from gaps, zero fills and strong telluric bands."""
    from scipy.ndimage import binary_dilation
    margin = margin or eso_px(1.0)
    bad = np.any(~np.isfinite(F) | (F <= 0), axis=0)
    bad = binary_dilation(bad, structure=np.ones(2 * margin + 1, bool))
    for lo, hi in eso_GAPS:
        bad |= (eso_GRID > lo) & (eso_GRID < hi)
    for lo, hi in eso_TELLURIC:
        bad |= (eso_GRID > lo) & (eso_GRID < hi)
    return ~bad


def eso_register(F, E, M):
    """Each epoch registered to the others by cross-correlation, applied above a tenth of a resolution element."""
    import warnings
    keep = eso_valid_mask(F)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N = np.array([np.where(keep, f / eso_envelope(f), np.nan) for f in F])
        for k in range(len(F)):
            M[k]["dv"] = 0.0
            others = np.nanmedian(np.delete(N, k, axis=0), axis=0)
            m = np.isfinite(N[k]) & np.isfinite(others)
            if m.sum() < eso_px(100.0):
                continue
            v = science.velocity_offset(N[k], others, eso_GRID, mask=m, span=30.0, step=0.05, coarse=0.5)
            if abs(v) > 0.1 * eso_C / eso_R:
                F[k] = science.to_rest(F[k], eso_GRID, v)
                E[k] = science.to_rest(E[k], eso_GRID, v)
                M[k]["dv"] = float(v)
    return F, E, M


def eso_intermittent(name, F, E, M, thr):
    F, E, M = eso_register(F, E, M)
    keep = eso_valid_mask(F)
    N = []
    for f in F:
        N.append(np.where(keep, f / eso_envelope(f), np.nan))
    N = np.array(N)
    T = np.nanmedian(N, axis=0)
    found, resids = [], []
    Tf = np.nan_to_num(T, nan=1.0)
    d1 = np.gradient(Tf)
    d2 = np.gradient(d1)
    envT = eso_envelope(np.where(np.isfinite(T), T, np.nan))
    depth = 1.0 - T / envT
    for k in range(len(N)):
        ratio = eso_smooth_ratio(N[k] / T)
        y = N[k] / ratio
        res = np.full(len(eso_GRID), np.nan)
        ch = eso_px(20.0)
        for s in range(0, len(eso_GRID), ch):
            sl = slice(s, s + ch)
            ok = np.isfinite(y[sl]) & np.isfinite(T[sl])
            if ok.sum() < ch // 10:
                continue
            A = np.vstack([np.ones(ok.sum()), Tf[sl][ok], d1[sl][ok], d2[sl][ok]]).T
            coef, *_ = np.linalg.lstsq(A, y[sl][ok], rcond=None)
            fit = coef[0] + coef[1] * Tf[sl] + coef[2] * d1[sl] + coef[3] * d2[sl]
            res[sl] = np.where(ok, y[sl] - fit, np.nan)
        sig = eso_block_mad(res)
        res = np.where(np.isfinite(res) & (T > 0.05) & (F[k] > 0) & (sig < 0.05), res, np.nan)
        resids.append((res / sig).astype(np.float32))
        snr, amp = eso_matched(res, sig)
        for sign in (1, -1):
            for j in eso_peaks(sign * snr, thr):
                wr = eso_width_ratio(sign * res, j)
                found.append(dict(star=name, epoch=k, sign=sign, lam=float(eso_GRID[j]), snr=float(sign * snr[j]),
                                  amp=float(sign * amp[j]), width=float(wr), berv=M[k]["berv"], mjd=M[k]["mjd"],
                                  dpr=M[k]["dpr"], lam_topo=float(eso_GRID[j] * (1.0 + M[k]["dv"] / eso_C)
                                                                  / (1.0 + M[k]["berv"] / eso_C)),
                                  depth=float(depth[j]) if np.isfinite(depth[j]) else float("nan")))
    return found, T, resids


def eso_telluric_map(acc, cnt):
    """Observatory wavelengths where summed epoch residuals vary beyond their surroundings: the telluric map."""
    from scipy.ndimage import binary_dilation, median_filter
    v = acc / np.maximum(cnt, 1)
    base = median_filter(np.nan_to_num(v, nan=0.0), size=eso_px(5.0) | 1, mode="nearest")
    tell = (cnt >= 4) & (v > 4.0 * np.maximum(base, 1e-6))
    return binary_dilation(tell, structure=np.ones(eso_px(0.1) | 1, bool))


def eso_groups(peaks_, key, tol_kms=None):
    """Peaks within tol_kms of neighbours in two or more epochs."""
    tol_kms = tol_kms or 0.9 * 115000.0 / eso_R
    pts = sorted(peaks_, key=lambda q: q[key])
    out, cur = [], []
    for q in pts:
        if cur and (q[key] - cur[-1][key]) / q[key] * eso_C > tol_kms:
            out.append(cur)
            cur = []
        cur.append(q)
    if cur:
        out.append(cur)
    return [g for g in out if len({q["epoch"] for q in g}) >= 2]


def eso_pair_correlations(X):
    """Pairwise Pearson correlation of rest-frame stacks over 5000-6400 A by masked matrix products."""
    def corr(sel):
        V = np.array([np.where(np.isfinite(x[sel]), x[sel], 0.0) for x in X], dtype=float)
        M = np.array([np.isfinite(x[sel]) for x in X], dtype=float)
        n = M @ M.T
        sx = V @ M.T
        sxx = (V * V) @ M.T
        sxy = V @ V.T
        with np.errstate(all="ignore"):
            c = (n * sxy - sx * sx.T) / np.sqrt((n * sxx - sx ** 2) * (n * sxx.T - sx.T ** 2))
        return np.nan_to_num(c, nan=-1.0), n
    band = (eso_GRID > 5000) & (eso_GRID < 6400)
    c, n = corr(band)
    short = n < eso_px(100.0)
    if short.any():
        c2, n2 = corr(np.ones(len(eso_GRID), bool))
        c = np.where(short, np.where(n2 > eso_px(100.0), c2, -1.0), c)
    return c


def eso_rest_shift(T, ref):
    """Velocity (km/s) of T relative to ref by cross-correlation, refined to 0.05 km/s."""
    m = (eso_GRID > 5000) & (eso_GRID < 6400) & np.isfinite(T) & np.isfinite(ref)
    if m.sum() < eso_px(100.0):
        m = np.isfinite(T) & np.isfinite(ref)
    if m.sum() < eso_px(10.0):
        return 0.0
    return science.velocity_offset(T, ref, eso_GRID, mask=m, span=150.0, step=0.05, coarse=0.5)


def eso_main():
    p = argparse.ArgumentParser()
    p.add_argument("--stars", nargs="*")
    p.add_argument("--snr", type=float, default=6.0)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--root", default=eso_ROOT)
    p.add_argument("--tag", default="")
    p.add_argument("--inst", choices=tuple(eso_INST), default="HARPS")
    a = p.parse_args()
    eso_set_instrument(a.inst)
    tag = a.tag or ("" if a.inst == "HARPS" else "_" + a.inst.lower())
    vtol = 115000.0 / eso_R  # velocity tolerances scale with the resolution element
    dirs = sorted(d for d in glob.glob(os.path.join(a.root, "*")) if os.path.isdir(d))
    if a.stars:
        dirs = [d for d in dirs if os.path.basename(d) in a.stars]
    temps, metas, inter, per_star, pooled = {}, {}, [], [], []
    acc, cnt = np.zeros(len(eso_GRID)), np.zeros(len(eso_GRID))
    for d in dirs:
        name = os.path.basename(d)
        F, E, M = eso_star_spectra(d)
        if len(F) < 3:
            print("%s: %d usable spectra, skipped" % (name, len(F)), flush=True)
            continue
        found, T, resids = eso_intermittent(name, F, E, M, a.snr)
        temps[name], metas[name] = T, M
        for k, r in enumerate(resids):
            rt = np.interp(eso_GRID * (1.0 + M[k]["berv"] / eso_C) / (1.0 + M[k].get("dv", 0.0) / eso_C), eso_GRID, r,
                           left=np.nan, right=np.nan)
            ok = np.isfinite(rt)
            acc[ok] += rt[ok] ** 2
            cnt[ok] += 1
        per_star.append((name, len(F), M, found))
        print("%s: %d spectra searched" % (name, len(F)), flush=True)
    tell = eso_telluric_map(acc, cnt)
    np.savez_compressed(core.results("laser_telluric%s.npz") % tag, grid=eso_GRID, telluric=tell, var=acc / np.maximum(cnt, 1))
    print("telluric map: %.1f per cent of the grid" % (100 * tell.mean()), flush=True)

    def telluric_residual(lam, M):
        for m in M:
            j = int(round((lam * (1.0 + m.get("dv", 0.0) / eso_C) / (1.0 + m["berv"] / eso_C) - eso_GRID[0])
                          / eso_STEP))
            if 0 <= j < len(eso_GRID) and tell[j]:
                return True
        return False

    for name, nf, M, found in per_star:
        narrow = [q for q in found if eso_narrow(q)]
        topo = eso_groups([q for q in narrow if q["sign"] > 0], "lam_topo")
        topo_lams = [float(np.mean([q["lam_topo"] for q in g])) for g in topo]

        def terrestrial(q):
            return (any(abs(q["lam_topo"] - x) / x * eso_C < 1.8 * vtol for x in topo_lams)
                    or any(abs(q["lam_topo"] - x) / x * eso_C < 3.0 * vtol for x in eso_AIRGLOW))

        good = [q for q in narrow if q["sign"] > 0 and not terrestrial(q)]
        negs = [q for q in narrow if q["sign"] < 0]

        def strength(g):
            v = sorted((q["snr"] for q in g), reverse=True)
            return v[1]

        bary = [g for g in eso_groups(good, "lam") if np.ptp([q["berv"] for q in g]) >= eso_C / eso_R]
        n_tell = sum(1 for g in bary if telluric_residual(float(np.mean([q["lam"] for q in g])), M))
        bary = [g for g in bary if not telluric_residual(float(np.mean([q["lam"] for q in g])), M)]
        nbary = [g for g in eso_groups(negs, "lam") if np.ptp([q["berv"] for q in g]) >= eso_C / eso_R
                 and not telluric_residual(float(np.mean([q["lam"] for q in g])), M)]
        null = sorted((strength(g) for g in nbary), reverse=True)
        print("%s: %d spectra, %d narrow emission-like peaks (%d terrestrial), %d recurring in the barycentric frame "
              "(%d more on another epoch's telluric line); null: %d recurring absorption-like groups, strongest %s"
              % (name, nf, len(good), sum(1 for q in narrow if q["sign"] > 0) - len(good), len(bary), n_tell,
                 len(nbary), ", ".join("%.1f" % v for v in null[:3])), flush=True)
        for g in topo[:6]:
            print("    terrestrial %.3f A (observatory frame) in %d epochs, SNR up to %.1f"
                  % (np.mean([q["lam_topo"] for q in g]), len({q["epoch"] for q in g}), max(q["snr"] for q in g)))
        pooled.extend(null)
        # groups whose epochs two or more other groups share are seeing-dependent line-depth residuals
        sets = [frozenset(q["epoch"] for q in g) for g in bary]
        for g, es in zip(bary, sets):
            g[0]["common_mode"] = sum(1 for e in sets if e == es) >= 3
        for g in bary:
            lam = float(np.mean([q["lam"] for q in g]))
            dep = float(np.nanmax([q["depth"] for q in g]))
            chromo = any(abs(lam - c) < 1.5 for c in eso_CHROMO)
            inter.append(dict(star=name, lam_bary=lam, epochs=sorted({q["epoch"] for q in g}), n_epochs=nf,
                              strength=strength(g), snr=[round(q["snr"], 1) for q in g],
                              width=[round(q["width"], 2) for q in g], berv_spread=float(np.ptp([q["berv"] for q in g])),
                              template_depth=dep, in_stellar_line=bool(dep > 0.1), chromospheric=bool(chromo),
                              lam_topo=[round(float(q["lam_topo"]), 4) for q in g],
                              dv=[round(M[q["epoch"]].get("dv", 0.0), 2) for q in g],
                              common_mode=bool(g[0].get("common_mode"))))
    # null: the sample's strongest absorption-like group
    top = max(pooled) if pooled else 0.0
    for r in inter:
        r["above_null"] = bool(r["strength"] > top)
        if r["above_null"]:
            print("    RECURRING %s %.3f A barycentric in epochs %s of %d, SNR %s, width %s, BERV spread %.1f km/s, "
                  "template depth %.2f%s%s" % (r["star"], r["lam_bary"], r["epochs"], r["n_epochs"], r["snr"],
                                                r["width"], r["berv_spread"], r["template_depth"],
                                                " (inside a stellar line)" if r["in_stellar_line"] else "",
                                                " (chromospheric line)" if r["chromospheric"] else ""), flush=True)
    # groups meeting other stars' in the observatory frame are sky or instrument
    tl = np.array([q for r in inter for q in r["lam_topo"]], dtype=float)
    ts = np.array([r["star"] for r in inter for _ in r["lam_topo"]], dtype=object)
    order = np.argsort(tl)
    tl, ts = tl[order], ts[order]
    for r in inter:
        near = set()
        for x in r["lam_topo"]:
            w = 1.8 * vtol * x / eso_C
            for j in range(np.searchsorted(tl, x - w, "left"), np.searchsorted(tl, x + w, "right")):
                if ts[j] != r["star"] and abs(tl[j] - x) / x * eso_C < 1.8 * vtol:
                    near.add(ts[j])
        r["stars_topo"] = len(near)
    shared = [r for r in inter if r["above_null"] and r["stars_topo"] >= 2]
    print("recurring groups above the sample's strongest absorption-like group (%.1f, of %d): %d; %d of them meet two "
          "or more other stars' groups in the observatory frame; %d share their epochs with two or more other groups "
          "of their star" % (top, len(pooled), sum(1 for r in inter if r["above_null"]), len(shared),
                             sum(1 for r in inter if r["above_null"] and r["common_mode"])))
    for r in shared:
        print("    shared in the observatory frame: %s %.3f A (%d other stars)" % (r["star"], r["lam_bary"],
                                                                                r["stars_topo"]))
    json.dump(inter, open(core.results("harps_intermittent%s.json") % tag, "w"))
    names = sorted(temps)
    if len(names) < 3:
        return
    # stacks to rest by strong-line cores, else aligned to the first star
    ref = temps[names[0]]
    v_ref = science.line_velocity(ref, eso_GRID)
    v_ref = v_ref if np.isfinite(v_ref) else 0.0
    rest, vels = {}, {}
    for n in names:
        v = science.line_velocity(temps[n], eso_GRID)
        if not np.isfinite(v):
            v = eso_rest_shift(temps[n], ref) + v_ref
        rest[n], vels[n] = science.to_rest(temps[n], eso_GRID, v), v

    def sky(n, lam):
        # peaks on the telluric map or airglow in any epoch are set aside
        for m in metas[n]:
            lo = lam * (1.0 + vels[n] / eso_C) * (1.0 + m.get("dv", 0.0) / eso_C) / (1.0 + m["berv"] / eso_C)
            j = int(round((lo - eso_GRID[0]) / eso_STEP))
            if (0 <= j < len(eso_GRID) and tell[j]) or any(abs(lo - x) / x * eso_C < 3.0 * vtol for x in eso_AIRGLOW):
                return True
        return False
    C = eso_pair_correlations([rest[n] for n in names])
    allpk = []
    for i, n in enumerate(names):
        others = [m for m in names if m != n]
        corr = [C[i, j] for j in range(len(names)) if j != i]
        nb = [others[i] for i in np.argsort(corr)[::-1][:a.k]]
        ok = np.isfinite(rest[n]) & np.all([np.isfinite(rest[m]) for m in nb], axis=0)
        x = (eso_GRID - 0.5 * (eso_GRID[0] + eso_GRID[-1])) / (0.5 * (eso_GRID[-1] - eso_GRID[0]))
        poly = [x ** k * s for k in range(4) for s in (1, -1)]
        A = np.vstack([rest[m] for m in nb] + poly).T
        model = np.full(len(eso_GRID), np.nan)
        ch = eso_px(200.0)
        for s in range(0, len(eso_GRID), ch):
            sl = slice(s, s + ch)
            m_ok = ok[sl]
            if m_ok.sum() < eso_px(10.0):
                continue
            Asl = np.nan_to_num(A[sl])
            coef, _ = nnls(Asl[m_ok], rest[n][sl][m_ok])
            m0 = Asl @ coef
            g1 = np.gradient(m0)
            g2 = np.gradient(g1)
            g1 /= max(np.abs(g1[m_ok]).max(), 1e-9)
            g2 /= max(np.abs(g2[m_ok]).max(), 1e-9)
            A2 = np.hstack([Asl, np.vstack([g1, -g1, g2, -g2]).T])
            coef, _ = nnls(A2[m_ok], rest[n][sl][m_ok])
            model[sl] = A2 @ coef
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(~ok, structure=np.ones(eso_px(3.0) | 1, bool))
        res = np.where(~edge & np.isfinite(model) & (model > 0.05), rest[n] / model - 1.0, np.nan)
        mdepth = 1.0 - model / eso_envelope(np.where(np.isfinite(model), model, np.nan))
        sig = eso_block_mad(res, floor=1e-3)
        res = np.where(sig < 0.05, res, np.nan)
        snr, amp = eso_matched(res, sig)
        for sign in (1, -1):
            for j in eso_peaks(sign * snr, a.snr):
                lam = float(eso_GRID[j])
                allpk.append(dict(star=n, sign=sign, lam_rest=lam, snr=float(sign * snr[j]), amp=float(sign * amp[j]),
                                  width=float(eso_width_ratio(sign * res, j)), neighbours=nb, vel=float(vels[n]),
                                  berv_spread=float(np.ptp([m["berv"] for m in metas[n]])),
                                  chromospheric=bool(any(abs(lam - c) < 1.5 for c in eso_CHROMO)),
                                  sky=sky(n, lam), depth=float(mdepth[j]) if np.isfinite(mdepth[j]) else 0.0))
    for sign in (1, -1):
        pk = sorted((q for q in allpk if q["sign"] == sign), key=lambda q: q["lam_rest"])
        lams = np.array([q["lam_rest"] for q in pk])
        for q in pk:
            w = 1.8 * vtol * q["lam_rest"] / eso_C
            lo, hi = np.searchsorted(lams, q["lam_rest"] - w, "left"), np.searchsorted(lams, q["lam_rest"] + w, "right")
            q["shared"] = sum(1 for o in pk[lo:hi] if o["star"] != q["star"]
                              and abs(o["lam_rest"] - q["lam_rest"]) / q["lam_rest"] * eso_C < 1.8 * vtol)
    clean = [q for q in allpk if eso_steady_clean(q, eso_R)]
    negs = sorted((q["snr"] for q in clean if q["sign"] < 0), reverse=True)
    above = [q for q in clean if q["sign"] > 0 and (not negs or q["snr"] > negs[0])]
    print("steady search: isolated narrow absorption-like residuals (null), strongest %s; %d emission-like peaks above "
          "them" % (", ".join("%.1f" % v for v in negs[:5]), len(above)))
    for q in sorted((q for q in allpk if q["sign"] > 0), key=lambda q: -q["snr"])[:30]:
        labels = []
        if q["chromospheric"]:
            labels.append("chromospheric")
        if q["shared"]:
            labels.append("shared x%d" % q["shared"])
        if q["sky"]:
            labels.append("sky")
        if q["depth"] > 0.1:
            labels.append("inside a stellar line")
        if not eso_narrow(q):
            labels.append("width %.2f" % q["width"])
        if not labels and negs and q["snr"] > negs[0]:
            labels.append("ABOVE-NULL")
        print("  %s %.3f A rest  SNR %.1f  amp %.3f  width %.2f  %s" % (q["star"], q["lam_rest"], q["snr"], q["amp"],
                                                                     q["width"], ", ".join(labels)))
    for q in above:
        print("  ABOVE-NULL %s %.3f A rest  SNR %.1f  amp %.3f  width %.2f" % (q["star"], q["lam_rest"], q["snr"],
                                                                            q["amp"], q["width"]))
    json.dump(allpk, open(core.results("harps_steady%s.json") % tag, "w"))


eso_IDS = {"HARPS": "laser_harps", "ESPRESSO": "laser_espresso", "UVES": "laser_uves", "FEROS": "laser_feros",
           "SOPHIE": "laser_sophie"}


def eso_cores_main(argv):
    """Unresolved, unshared, sky-clear steady peaks inside chromospheric and Balmer cores, with core dips as null."""
    p = argparse.ArgumentParser(prog="seti.py lasers eso cores")
    p.add_argument("--record", action="store_true")
    a = p.parse_args(argv)
    runs = (("HARPS", ""), ("HARPS", "_fgk"), ("ESPRESSO", "_espresso"), ("FEROS", "_feros"), ("UVES", "_uves"))
    rows, total, above_all = [], 0, []
    for inst, tag in runs:
        path = core.results("harps_steady%s.json" % tag)
        if not os.path.exists(path):
            continue
        steady = json.load(open(path))
        R = eso_INST[inst]["R"]
        in_core = [q for q in steady if q["chromospheric"] and eso_narrow(q) and q["shared"] == 0 and not q.get("sky")
                and q.get("berv_spread", np.inf) >= eso_C / R]
        neg = sorted((q["snr"] for q in in_core if q["sign"] < 0), reverse=True)
        pos = [q for q in in_core if q["sign"] > 0]
        above = [q for q in pos if not neg or q["snr"] > neg[0]]
        for q in above:
            line = min(eso_CHROMO, key=lambda c: abs(c - q["lam_rest"]))
            q["line"], q["offset_kms"] = line, (q["lam_rest"] - line) / line * eso_C
        above_all += [(inst, q) for q in above]
        stars = sorted({q["star"] for q in steady})
        total += len(stars)
        print("%s%s: %d stars; %d unresolved emission-like and %d absorption-like core peaks; %d above the strongest "
              "absorption-like" % (inst, tag, len(stars), len(pos), len(neg), len(above)))
        for q in above:
            print("  %s %.3f A, %+.0f km/s from %.2f, SNR %.1f, amplitude %.3f, template depth %.2f"
                  % (q["star"], q["lam_rest"], q["offset_kms"], q["line"], q["snr"], q["amp"], q.get("depth", 0.0)))
        for s in stars:
            mine = [q for q in above if q["star"] == s]
            rows.append(dict(target=eso_target(s), stat=max((q["snr"] for q in mine), default=None),
                             unit="S/N of the strongest unresolved core peak above the core null",
                             vet=("in a Balmer or He I line %.0f km/s from its centre, where the template dips by %.0f per cent"
                                  % (abs(mine[0]["offset_kms"]), 100 * mine[0].get("depth", 0.0))) if mine and mine[0]["line"] not in
                                 (3933.66, 3968.47, 5889.95, 5895.92) else None,
                             note="%s%s" % (inst, tag)))
    if a.record:
        core.record_search(
            "laser_line_cores", "Lasers in line cores", archive="ESO archive, HARPS, ESPRESSO, FEROS and UVES Phase 3 spectra",
            selection="the stars of the laser-line runs", code=os.path.abspath(__file__), entry="eso cores", n_targets=total,
            params=dict(lines=list(eso_CHROMO)), null_summary="absorption-like unresolved peaks inside the same cores",
            ledger="Lasers in line cores",
            result=("%d unresolved core peaks above the core null, all in Balmer and He I lines %.0f-%.0f km/s from their "
                    "centres, where the neighbour template dips by %.0f-%.0f per cent; none in the Ca II H and K or Na D cores"
                    % (len(above_all), min(abs(q["offset_kms"]) for _, q in above_all), max(abs(q["offset_kms"]) for _, q in above_all),
                       100 * min(q.get("depth", 0.0) for _, q in above_all), 100 * max(q.get("depth", 0.0) for _, q in above_all)))
                   if above_all else "no unresolved core peak above the core null")
        core.clear_targets("laser_line_cores")
        core.record_targets("laser_line_cores", rows)
        print("recorded %d rows" % len(rows))


eso_LON = {"LaSilla": -70.73, "Paranal": -70.40, "OHP": 5.71}


def eso_nights(root, q):
    """Whether a steady peak lives in one night: tallest night at 5 sigma, every other 3 sigma below."""
    if not os.path.isdir(os.path.join(root, q["star"])):
        raise FileNotFoundError("no spectra of %s under %s" % (q["star"], root))
    F, E, M = eso_star_spectra(os.path.join(root, q["star"]))
    lb = q["lam_rest"] * (1 + q["vel"] / eso_C)
    j = int(round((lb - eso_GRID[0]) / eso_STEP))
    w, core = eso_px(1.0), max(eso_px(lb / eso_R), 1)
    lon = eso_LON.get(eso_SITE, 0.0)
    by = {}
    for f, m in zip(F, M):
        if j - w - core < 0 or j + core + w + 1 > len(f) or not np.isfinite(m["mjd"]):
            continue
        seg = f[j - w - core:j + core + w + 1]
        if np.all(np.isfinite(seg)):
            by.setdefault(int(math.floor(m["mjd"] + lon / 360.0 - 0.5)), []).append(seg / np.median(seg))
    heights = []
    for night, segs in sorted(by.items()):
        s_ = np.mean(segs, axis=0)
        side = np.concatenate([s_[:w], s_[-w:]])
        cont = np.median(side)
        mad = 1.4826 * np.median(np.abs(side - cont))
        if mad > 0:
            heights.append((night, float(np.max(s_[w:w + 2 * core + 1]) - cont), float(mad)))
    if len(heights) < 2:
        return dict(nights=len(heights), one_night=False, heights=heights)
    top = max(heights, key=lambda x: x[1] / x[2])
    others = [x for x in heights if x is not top]
    one = ((top[1] / top[2] >= 5.0 and all(x[1] < top[1] - 3.0 * math.hypot(x[2], top[2]) for x in others))
           or (top[1] / top[2] >= 3.0 and all(x[1] < x[2] for x in others)))
    return dict(nights=len(heights), one_night=bool(one), heights=heights)


def eso_target(name):
    """Coverage identifier of an ESO star directory."""
    if name.upper().startswith("HIP"):
        return "hip:" + name[3:].strip()
    return "gaia_dr3:" + name.replace("GaiaDR3", "").strip()


def eso_record_main(argv):
    """Record one spectrograph's run: recurring groups above null by explaining filter, and steady peaks above null."""
    p = argparse.ArgumentParser(prog="seti.py lasers eso record")
    p.add_argument("--inst", choices=tuple(eso_INST), default="HARPS")
    p.add_argument("--tag", default="")
    p.add_argument("--selection", default="")
    p.add_argument("--root", default=None, help="the star directories the search read (default: the archive cache "
                   "for the instrument)")
    a = p.parse_args(argv)
    eso_set_instrument(a.inst)
    tag = a.tag or ("" if a.inst == "HARPS" else "_" + a.inst.lower())
    inter = json.load(open(core.results("harps_intermittent%s.json") % tag))
    steady = json.load(open(core.results("harps_steady%s.json") % tag))
    stars = sorted({r["star"] for r in inter} | {q["star"] for q in steady})

    def why(r):
        for key, label in (("stars_topo", "shared in the observatory frame"), ("common_mode", "epoch common mode"),
                           ("chromospheric", "chromospheric line"), ("in_stellar_line", "inside a stellar line"),
                           ("second_fibre", "light in the sky fibre")):
            v = r.get(key)
            if (v >= 2 if key == "stars_topo" else v):
                return label
        return None
    above = [r for r in inter if r.get("above_null")]
    left = [r for r in above if why(r) is None]
    clean = [q for q in steady if eso_steady_clean(q, eso_INST[a.inst]["R"])]
    top = max([q["snr"] for q in clean if q["sign"] < 0] or [0.0])
    st_above = [q for q in clean if q["sign"] > 0 and q["snr"] > top]
    root = a.root or {"HARPS": eso_ROOT, "SOPHIE": core.data("sophie")}.get(a.inst, core.results("tc_eso", a.inst))
    for q in st_above:
        q["night_test"] = eso_nights(root, q)
    # a residual of the neighbour model rises by at most the model's own depth
    st_dip = [q for q in st_above if not q["night_test"]["one_night"] and q["amp"] <= 1.3 * q.get("depth", 0.0)]
    st_left = [q for q in st_above if not q["night_test"]["one_night"] and q not in st_dip]
    st_one = len(st_above) - len(st_left) - len(st_dip)
    # regions crowded with other stars' residuals mark line-strength differences
    lam_all = np.array([q["lam_rest"] for q in steady])
    snr_all = np.array([q["snr"] for q in steady])
    pos_all = np.array([q["sign"] > 0 for q in steady])
    star_all = np.array([str(q["star"]) for q in steady])
    for q in st_above:
        m = (np.abs(lam_all - q["lam_rest"]) < 0.6) & (snr_all >= 10) & pos_all & (star_all != str(q["star"]))
        q["crowd"] = len(set(star_all[m].tolist()))
    st_crowd = [q for q in st_left if q["crowd"] >= 5]
    st_left = [q for q in st_left if q["crowd"] < 5]
    from collections import Counter
    reasons = Counter(why(r) for r in above if why(r))
    sid = eso_IDS[a.inst] + ("" if a.tag in ("", "_" + a.inst.lower()) else a.tag)
    core.record_search(
        sid, "Laser lines, %s" % a.inst, archive=("OHP SOPHIE archive, s1d spectra" if a.inst == "SOPHIE" else
                                                  "ESO archive, %s Phase 3 spectra" % a.inst),
        selection=a.selection or "stars with three or more epochs", code=os.path.abspath(__file__), entry="eso",
        n_targets=len(stars), params=dict(snr=6.0, R=eso_INST[a.inst]["R"], width_ratio=[0.6, 1.6]),
        null_summary="recurring absorption-like groups across the sample; isolated absorption-like steady residuals "
                     "(strongest %.1f)" % top,
        sensitivity="unresolved lines at 6 sigma of each epoch's residual",
        ledger="Laser lines, %s" % ("ESPRESSO, UVES, FEROS" if a.inst in ("ESPRESSO", "UVES", "FEROS") else a.inst),
        result="%d recurring groups above the null, explained: %s; %d left; %d steady peaks above the null, %d of "
               "them in one night's exposures alone, %d within the neighbour model's own dip and %d where five or more "
               "other stars carry model residuals within 0.6 A; %d left"
               % (len(above), ", ".join("%s %d" % kv for kv in reasons.most_common()) or "none", len(left),
                  len(st_above), st_one, len(st_dip), len(st_crowd), len(st_left)))
    core.clear_targets(sid)
    core.record_targets(sid, [dict(target=eso_target(s), stat=sum(1 for r in left if r["star"] == s)
                                       + sum(1 for q in st_left if q["star"] == s), unit="unexplained features")
                                  for s in stars])
    print("%s: %d stars; %d recurring groups above the null (%s), %d left; %d steady peaks above the null"
          % (sid, len(stars), len(above), dict(reasons), len(left), len(st_left)))
    for r in left:
        print("  group %s %.3f A epochs %s of %d SNR %s" % (r["star"], r["lam_bary"], r["epochs"], r["n_epochs"], r["snr"]))
    for q in st_left:
        print("  steady %s %.3f A SNR %.1f width %.2f; nightly heights %s"
              % (q["star"], q["lam_rest"], q["snr"], q["width"],
                 ", ".join("%.1f" % (h / e) for _, h, e in q["night_test"]["heights"])))
    for q in st_dip:
        print("  model dip: %s %.3f A SNR %.1f, amplitude %.4f against the model's depth %.4f"
              % (q["star"], q["lam_rest"], q["snr"], q["amp"], q["depth"]))
    for q in st_crowd:
        print("  crowded: %s %.3f A SNR %.1f, %d other stars with residuals at S/N 10 or more within 0.6 A"
              % (q["star"], q["lam_rest"], q["snr"], q["crowd"]))
    for q in st_above:
        if q["night_test"]["one_night"]:
            print("  one night: %s %.3f A SNR %.1f; nightly heights %s" % (
                q["star"], q["lam_rest"], q["snr"], ", ".join("%.1f" % (h / e) for _, h, e in q["night_test"]["heights"])))


# fetch mode

eso_TAP = "https://archive.eso.org/tap_obs/sync"

eso_SIMBAD = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

eso_OUT = core.data("harps_dl")

def eso_simbad_pos(name):
    q = ("SELECT basic.ra, basic.dec FROM basic JOIN ident ON ident.oidref=basic.oid WHERE ident.id='%s'" % name)
    d = urllib.parse.urlencode(dict(request="doQuery", lang="adql", format="json", query=q)).encode()
    return json.loads(urllib.request.urlopen(eso_SIMBAD, data=d, timeout=90).read())["data"][0]

def eso_fetch_main():
    p = argparse.ArgumentParser()
    p.add_argument("stars", nargs="+")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", default=eso_OUT)
    a = p.parse_args()
    for name in a.stars:
        ra, dec = eso_simbad_pos(name)
        q = ("SELECT dp_id, t_min, snr, target_name FROM ivoa.ObsCore WHERE obs_collection='HARPS' AND "
             "dataproduct_type='spectrum' AND CONTAINS(POINT('ICRS',s_ra,s_dec),CIRCLE('ICRS',%f,%f,0.0083))=1"
             % (ra, dec))
        u = eso_TAP + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
        rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(u, timeout=180).read().decode())))
        rows.sort(key=lambda r: -float(r["snr"] or 0))
        chosen, nights = [], set()
        for r in rows:
            night = int(float(r["t_min"]) - 0.5)
            if night not in nights:
                chosen.append(r)
                nights.add(night)
            if len(chosen) >= a.n:
                break
        for r in rows:
            if len(chosen) >= a.n:
                break
            if r not in chosen:
                chosen.append(r)
        d = os.path.join(a.out, name.replace(" ", ""))
        os.makedirs(d, exist_ok=True)
        got = 0
        for r in chosen:
            path = os.path.join(d, r["dp_id"].replace(":", "_") + ".fits")
            if os.path.exists(path):
                got += 1
                continue
            for attempt in range(3):
                try:
                    b = urllib.request.urlopen("https://dataportal.eso.org/dataportal_new/file/" + r["dp_id"],
                                               timeout=300).read()
                    if b[:6] == b"SIMPLE":
                        open(path, "wb").write(b)
                        got += 1
                    break
                except Exception:
                    time.sleep(10 * (attempt + 1))
        print("%s: %d spectra in archive, %d on disk" % (name, len(rows), got), flush=True)


def eso_cli():
    if len(sys.argv) > 1 and sys.argv[1] == 'fetch':
        sys.argv.pop(1)
        eso_fetch_main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'record':
        eso_record_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'cores':
        eso_cores_main(sys.argv[2:])
    else:
        eso_main()


# --------------------------------------------------------------------------------------------------------------

sophie_DOC = r"""Laser lines in SOPHIE spectra of dwarfs within 50 pc via the ESO search, with fibre B sky checks.

    python -I seti.py lasers sophie select [--n 12]
    python -I seti.py lasers sophie fetch [--threads 4]
    python -I seti.py lasers sophie second-fibre"""

sophie_ROOT = os.environ.get("SOPHIE_ROOT", core.data("sophie"))
sophie_URL = "http://atlas.obs-hp.fr/sophie/sophie.cgi"
sophie_HEAD = core.UA
sophie_ROW = re.compile(r"J(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)\s+S\s+E\s+(\d+)\s+"
                        r"(?:\d+\s+)?(\d{4}-\d{2}-\d{2})\s+(HR|HE)\s+(?:([A-Z]+)\s+)?([\d.]+)\s+(-?\d+)")


def sophie_load(h):
    """eso_load for a SOPHIE s1d file: fibre A on the grid."""
    H = h[0].header
    d = np.asarray(h[0].data, float)
    w = H["CRVAL1"] + (np.arange(len(d)) + 1 - H["CRPIX1"]) * H["CDELT1"]
    t = str(H.get("OHP OBS DATE START", ""))
    mjd = float(H.get("OHP DRS BJD", np.nan)) - 2400000.5
    meta = dict(berv=float(H["OHP DRS BERV"]), mjd=mjd, snr=float(H.get("OHP DRS CAL EXT SN26", np.nan)),
                dpr=str(H.get("OHP DPR TYPE", "")), specsys="BARYCENT", file=os.path.basename(h.filename() or t))
    fi = np.interp(eso_GRID, w, d, left=np.nan, right=np.nan)
    return fi, np.full(len(eso_GRID), np.nan), meta


def sophie_listing():
    """Public SOPHIE spectra listing as (ra, dec, seq, date, mode, fibre_b, sn26), cached."""
    path = os.path.join(sophie_ROOT, "catalogue_all.txt")
    if not os.path.exists(path):
        os.makedirs(sophie_ROOT, exist_ok=True)
        u = sophie_URL + "?n=sophie&a=t&ob=ra,seq&c=o&o=%25"
        t = urllib.request.urlopen(urllib.request.Request(u, headers=sophie_HEAD), timeout=900).read()
        open(path, "wb").write(t)
    out = []
    for line in open(path, encoding="latin-1"):
        m = sophie_ROW.search(line)
        if not m or line.startswith("#"):
            continue
        g = m.groups()
        ra = 15.0 * (int(g[0]) + int(g[1]) / 60.0 + float(g[2]) / 3600.0)
        dec = (1 if g[3] == "+" else -1) * (int(g[4]) + int(g[5]) / 60.0 + float(g[6]) / 3600.0)
        out.append(dict(ra=ra, dec=dec, seq=g[7], date=g[8], mode=g[9], fibre_b=g[10] or "", sn26=int(g[12])))
    return out


def sophie_select_main(n):
    from collections import defaultdict
    spec = [r for r in sophie_listing() if r["mode"] == "HR"]
    stars = core.gaia("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp FROM "
                          "gaiadr3.gaia_source WHERE parallax > 20 AND dec > -45 AND phot_g_mean_mag < 14 AND "
                          "bp_rp BETWEEN 0.5 AND 4.0")
    cells = defaultdict(list)
    for s in stars:
        if float(s["phot_g_mean_mag"]) + 5 * math.log10(float(s["parallax"]) / 100.0) <= 2.0:
            continue
        dec = float(s["dec"]) - num(s["pmdec"], 0.0) / 3.6e6 * 16.0
        ra = float(s["ra"]) - num(s["pmra"], 0.0) / 3.6e6 * 16.0 / math.cos(math.radians(dec))
        s["ra2000"], s["dec2000"] = ra, dec
        cells[(int(ra * 4), int((dec + 90) * 4))].append(s)
    by = defaultdict(list)
    for r in spec:
        best = None
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for s in cells.get((int(r["ra"] * 4) + i, int((r["dec"] + 90) * 4) + j), ()):
                    d = sep_arcsec(r["ra"], r["dec"], s["ra2000"], s["dec2000"])
                    if d <= 15.0 and (best is None or float(s["phot_g_mean_mag"]) < float(best["phot_g_mean_mag"])):
                        best = s
        if best is not None:
            by[str(best["source_id"])].append(r)
    sample = {}
    for sid, rows in by.items():
        nights = {}
        for r in sorted(rows, key=lambda r: -r["sn26"]):
            nights.setdefault(r["date"], r)
        if len(nights) < 3:
            continue
        chosen = sorted(nights.values(), key=lambda r: -r["sn26"])[:n]
        rest = [r for r in sorted(rows, key=lambda r: -r["sn26"]) if r not in chosen]
        chosen += rest[:max(0, n - len(chosen))]
        sample[sid] = [dict(seq=r["seq"], date=r["date"], fibre_b=r["fibre_b"], sn26=r["sn26"]) for r in chosen]
    json.dump(sample, open(os.path.join(sophie_ROOT, "sample.json"), "w"), indent=0)
    print("%d high-resolution spectra; %d nearby dwarfs matched; %d observed on three or more nights, %d spectra chosen"
          % (len(spec), len(by), len(sample), sum(len(v) for v in sample.values())))


def sophie_fetch_one(job):
    sid, seq = job
    path = os.path.join(sophie_ROOT, sid, "%s_s1d.fits" % seq)
    if os.path.exists(path):
        return True
    u = sophie_URL + "?c=i&a=mime:application/fits&o=sophie:[s1d,%s]" % seq
    for attempt in range(3):
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers=sophie_HEAD), timeout=300).read()
            if b[:6] != b"SIMPLE":
                return False
            open(path + ".tmp", "wb").write(b)
            os.replace(path + ".tmp", path)
            return True
        except Exception:
            time.sleep(10 * (attempt + 1))
    return False


def sophie_fetch_main(threads):
    sample = json.load(open(os.path.join(sophie_ROOT, "sample.json")))
    jobs = []
    for sid, rows in sample.items():
        os.makedirs(os.path.join(sophie_ROOT, sid), exist_ok=True)
        jobs += [(sid, r["seq"]) for r in rows]
    t0, ok = time.time(), 0
    with ThreadPoolExecutor(threads) as ex:
        for k, got in enumerate(ex.map(sophie_fetch_one, jobs)):
            ok += bool(got)
            if (k + 1) % 200 == 0:
                print("%d of %d spectra, %d on disk, %.0f s" % (k + 1, len(jobs), ok, time.time() - t0), flush=True)
    print("%d of %d spectra on disk" % (ok, len(jobs)))


def sophie_fibre_b(path):
    """Fibre B (wavelength, flux) in the barycentric frame and its content type."""
    from astropy.io import fits
    with fits.open(path) as h:
        if len(h) < 2 or h[1].data is None:
            return None
        H = h[1].header
        d = np.asarray(h[1].data, float)
        kind = str(h[0].header.get("OHP DPR TYPE", "")).split(",")
    w = H["CRVAL1"] + (np.arange(len(d)) + 1 - H["CRPIX1"]) * H["CDELT1"]
    return w, d, kind[1] if len(kind) > 1 else ""


def sophie_peak_b(fb, lam, thr=4.0):
    """Peak significance within two resolution elements of lam in fibre B against 4 A scatter."""
    w, d, _ = fb
    m = (w > lam - 2.0) & (w < lam + 2.0) & np.isfinite(d)
    if m.sum() < 50:
        return 0.0
    x, y = w[m], d[m]
    base = np.median(y)
    s = 1.4826 * np.median(np.abs(y - base))
    near = np.abs(x - lam) < 2 * lam / 75000.0
    return float((y[near].max() - base) / s) if near.any() and s > 0 else 0.0


def sophie_fibre_b_light(path, lam):
    """Whether a sky-watching fibre B holds a 4-sigma peak at lam; lamp epochs excluded."""
    fb = sophie_fibre_b(path)
    if fb is None or fb[2] != "SKY":
        return 0.0, False
    s = sophie_peak_b(fb, lam)
    return s, s >= 4.0


def sophie_second_fibre_main():
    tag = "_sophie"
    inter_p = core.results("harps_intermittent%s.json") % tag
    steady_p = core.results("harps_steady%s.json") % tag
    inter = json.load(open(inter_p))
    steady = json.load(open(steady_p)) if os.path.exists(steady_p) else []
    eso_set_instrument("SOPHIE")
    files = {}

    def star_files(star):
        if star not in files:
            files[star] = [m["file"] for m in eso_star_spectra(os.path.join(sophie_ROOT, star))[2]]
        return files[star]
    marked = 0
    for r in inter:
        if not r.get("above_null"):
            continue
        got = [sophie_fibre_b_light(os.path.join(sophie_ROOT, r["star"], star_files(r["star"])[e]), r["lam_bary"])
               for e in r["epochs"]]
        r["second_fibre_sigma"] = round(max(g[0] for g in got), 1)
        r["second_fibre"] = any(g[1] for g in got)
        marked += r["second_fibre"]
    clean = [q for q in steady if eso_steady_clean(q, eso_INST["SOPHIE"]["R"])]
    top = max([q["snr"] for q in clean if q["sign"] < 0] or [0.0])
    for q in clean:
        if q["sign"] < 0 or q["snr"] <= top:
            continue
        lam = q["lam_rest"] * (1.0 + q.get("vel", 0.0) / eso_C)
        q["second_fibre"] = any(sophie_fibre_b_light(os.path.join(sophie_ROOT, q["star"], f), lam)[1]
                                for f in star_files(q["star"]))
    json.dump(inter, open(inter_p, "w"))
    if steady:
        json.dump(steady, open(steady_p, "w"))
    print("%d groups above the null; %d with light at their wavelength in fibre B" %
          (sum(1 for r in inter if r.get("above_null")), marked))


def sophie_cli():
    p = argparse.ArgumentParser(prog="seti.py lasers sophie")
    p.add_argument("mode", choices=("select", "fetch", "second-fibre"))
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.mode == "select":
        sophie_select_main(a.n)
    elif a.mode == "fetch":
        sophie_fetch_main(a.threads)
    else:
        sophie_second_fibre_main()


# --------------------------------------------------------------------------------------------------------------

galah_DOC = r"""Laser lines in GALAH DR4 spectra of sun-like dwarfs against twin-star fits, with observatory- and rest-frame sharing and same-exposure fibre checks. The served spectra are on barycentric wavelengths with the star's velocity left in, so each is shifted to rest by the catalogue's rv_comp_1 before the twins are fitted.

`exposures` reads each candidate in the other fibres of its exposure at its observatory wavelength; `check` reads it in the same fibre's other runs of the night at that wavelength and in the star's other spectra at its rest wavelength; `raw` reads each candidate those leave in the raw HERMES frames of its observation from the AAT archive (hermes_raw/<date>/): the fibre's trace from the flat, the fibre less its neighbours exposure by exposure, light the fibre holds in every exposure against excesses one exposure holds, and each excess's place and shape against the trace and the arc lines.

    python -I seti.py lasers galah fetch        (downloads galah_dl/<sobject_id>_<camera>.fits)
    python -I seti.py lasers galah search
    python -I seti.py lasers galah exposures
    python -I seti.py lasers galah check
    python -I seti.py lasers galah raw
    python -I seti.py lasers galah record"""

galah_HERE = core.RESULTS
galah_DL = os.path.join(galah_HERE, "galah_dl")
galah_CAMS = ("B", "V", "R", "I")
galah_C = 299792.458
galah_R_RES = 28000.0
galah_MASK = ((7590.0, 7700.0),)


def galah_fetch_one(args):
    sid, cam = args
    path = os.path.join(galah_DL, "%s_%s.fits" % (sid, cam))
    if os.path.exists(path):
        return True
    u = ("https://datacentral.org.au/vo/slink/links?ID=%s&DR=galah_dr4&IDX=0&FILT=%s&RESPONSEFORMAT=fits" % (sid, cam))
    os.makedirs(galah_DL, exist_ok=True)
    for attempt in range(3):
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers=core.UA), timeout=120).read()
        except Exception:
            time.sleep(5 * (attempt + 1))
            continue
        if b[:6] == b"SIMPLE":
            open(path, "wb").write(b)
            return True
        return False
    return False


def galah_load(sid, cam):
    from astropy.io import fits
    path = os.path.join(galah_DL, "%s_%s.fits" % (sid, cam))
    if not os.path.exists(path):
        return None
    with fits.open(path) as h:
        H = h[0].header
        d = np.asarray(h[0].data, float)
        w = H["CRVAL1"] + (np.arange(len(d)) + 1 - H["CRPIX1"]) * H["PC1_1"] * H["CDELT1"]
        meta = dict(mjd=float(H.get("MJD-BEG", np.nan)) + float(H.get("EXPTIME", 0)) / 172800.0, rv=float(H.get("RV", np.nan)),
                    ra=float(H.get("RA", np.nan)), dec=float(H.get("DEC", np.nan)))
    return w, d, meta


def galah_grids():
    """Common rest-frame grid per camera."""
    g = {"B": (4715.0, 4900.0, 0.046), "V": (5649.0, 5870.0, 0.0547), "R": (6480.0, 6735.0, 0.0632),
         "I": (7586.0, 7884.0, 0.0736)}
    return {c: np.arange(lo, hi, st) for c, (lo, hi, st) in g.items()}


def galah_baryc(meta):
    """Barycentric correction (km/s) at Siding Spring."""
    return science.barycentric_correction(meta["ra"], meta["dec"], meta["mjd"], "SSO")


def galah_matched(res, sig, sp):
    k = np.arange(-int(4 * sp) - 1, int(4 * sp) + 2)
    g = np.exp(-0.5 * (k / sp) ** 2)
    w = np.where(np.isfinite(res) & np.isfinite(sig) & (sig > 0), 1.0 / np.maximum(sig, 1e-9) ** 2, 0.0)
    r0 = np.where(w > 0, res, 0.0)
    num = np.convolve(r0 * w, g, mode="same")
    den = np.convolve(w, g * g, mode="same")
    return num / np.sqrt(np.maximum(den, 1e-30)), num / np.maximum(den, 1e-30)


def galah_width_ratio(res, j, sp):
    half = int(5 * sp) + 2
    lo, hi = max(j - half, 0), min(j + half + 1, len(res))
    x = np.arange(lo, hi) - j
    y = res[lo:hi]
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    best, bs = np.inf, np.nan
    for s in np.arange(0.4, 4.01, 0.1) * sp:
        gg = np.exp(-0.5 * (x / s) ** 2)
        a = np.dot(gg[ok], y[ok]) / np.dot(gg[ok], gg[ok])
        chi = np.sum((y[ok] - a * gg[ok]) ** 2)
        if chi < best:
            best, bs = chi, s
    return bs / sp


def galah_block_mad(res, block=200, floor=2e-3):
    n = len(res)
    cen, val = [], []
    for s in range(0, n, block):
        x = res[s:s + block]
        x = x[np.isfinite(x)]
        if len(x) > block // 4:
            cen.append(s + block // 2)
            val.append(1.4826 * np.median(np.abs(x - np.median(x))))
    if not cen:
        return np.full(n, np.inf)
    return np.maximum(np.interp(np.arange(n), cen, val), floor)


def galah_search_part(job, thr=5.5, k=20):
    """Peaks for one camera and one slice of the spectra."""
    cam, part, nparts = job
    quiet_warnings()
    rows = list(csv.DictReader(open(os.path.join(galah_HERE, "galah_sample.csv"))))
    G = galah_grids()
    params = np.array([[float(r["teff"]) / 100.0, float(r["logg"]) / 0.1, float(r["fe_h"] or 0) / 0.1] for r in rows])
    allpk = []
    for cam in (cam,):
        grid = G[cam]
        F, metas, ok_rows = [], [], []
        for i, r in enumerate(rows):
            x = galah_load(r["sobject_id"], cam)
            if x is None:
                continue
            w, d, meta = x
            # the served wavelengths are barycentric with the star's velocity left in; the catalogue's velocity takes them to rest
            rv = core.num(r["rv_comp_1"])
            if rv is None:
                continue
            fi = np.interp(grid, w / (1.0 + rv / galah_C), d, left=np.nan, right=np.nan)
            for lo, hi in galah_MASK:
                fi[(grid > lo) & (grid < hi)] = np.nan
            F.append(fi)
            metas.append(meta)
            ok_rows.append(i)
        F = np.array(F)
        P = params[ok_rows]
        sp = grid.mean() / galah_R_RES / 2.3548 / (grid[1] - grid[0])
        x = (grid - grid.mean()) / np.ptp(grid)
        poly = [x ** p * s for p in range(3) for s in (1, -1)]
        for n in range(part, len(F), nparts):
            dd = np.hypot(np.hypot(P[:, 0] - P[n, 0], P[:, 1] - P[n, 1]), P[:, 2] - P[n, 2])
            dd[n] = np.inf
            same_star = [m for m in range(len(F)) if rows[ok_rows[m]]["gaiadr3_source_id"] == rows[ok_rows[n]]["gaiadr3_source_id"]]
            dd[same_star] = np.inf
            nb = np.argsort(dd)[:k]
            B = np.nan_to_num(F[nb], nan=1.0)
            ok = np.isfinite(F[n])
            if ok.sum() < len(grid) * 0.5:
                continue
            A = np.vstack([B] + poly).T
            coef, _ = nnls(A[ok], F[n][ok])
            m0 = A @ coef
            g1 = np.gradient(m0)
            g2 = np.gradient(g1)
            g1 /= max(np.abs(g1[ok]).max(), 1e-9)
            g2 /= max(np.abs(g2[ok]).max(), 1e-9)
            A2 = np.hstack([A, np.vstack([g1, -g1, g2, -g2]).T])
            coef, _ = nnls(A2[ok], F[n][ok])
            model = A2 @ coef
            tmpl = B.T @ coef[:len(nb)] / max(coef[:len(nb)].sum(), 1e-9)
            res = np.where(ok, F[n] - model, np.nan)
            sig = galah_block_mad(res)
            snr, amp = galah_matched(res, sig, sp)
            vb = galah_baryc(metas[n])
            # header RV is the barycentric correction; the star's RV comes from the catalogue
            try:
                rv = float(rows[ok_rows[n]]["rv_comp_1"])
            except ValueError:
                rv = 0.0
            for sign in (1, -1):
                s_ = sign * snr
                idx = np.nonzero((s_ > thr) & ok)[0]
                chosen = []
                for j in idx[np.argsort(-s_[idx])]:
                    if all(abs(j - c) >= 10 for c in chosen):
                        chosen.append(int(j))
                    if len(chosen) > 200:
                        break
                for j in chosen:
                    lam = float(grid[j])
                    allpk.append(dict(sid=rows[ok_rows[n]]["sobject_id"], star=rows[ok_rows[n]]["gaiadr3_source_id"],
                                      cam=cam, sign=sign, lam=lam, snr=round(float(s_[j]), 2),
                                      amp=float(sign * amp[j]), tmpl=round(float(tmpl[j]), 3),
                                      width=round(float(galah_width_ratio(sign * res, j, sp)), 2),
                                      lam_topo=lam * (1.0 + rv / galah_C) * (1.0 - vb / galah_C), mjd=metas[n]["mjd"]))
    return allpk


def galah_search(nparts=4, workers=min(16, core.MAX_PROCS)):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(c, p, nparts) for c in galah_CAMS for p in range(nparts)]
    allpk = []
    with ProcessPoolExecutor(workers) as ex:
        for pk in ex.map(galah_search_part, jobs):
            allpk.extend(pk)
    print("%d peaks from %d jobs" % (len(allpk), len(jobs)), flush=True)
    return allpk


def galah_vet(allpk):
    for sign in (1, -1):
        grp = [q for q in allpk if q["sign"] == sign]
        rest = np.array([q["lam"] for q in grp])
        topo = np.array([q["lam_topo"] for q in grp])
        stars = np.array([q["star"] for q in grp])
        for q in grp:
            other = stars != q["star"]
            tol = q["lam"] / galah_R_RES / 2.0
            q["rest_twins"] = int(np.sum(other & (np.abs(rest - q["lam"]) < tol)))
            q["topo_twins"] = int(np.sum(other & (np.abs(topo - q["lam_topo"]) < tol)))
    for q in allpk:
        q["isolated"] = bool(q["rest_twins"] <= 1 and q["topo_twins"] <= 1 and 0.6 <= q["width"] <= 1.6)
    out = {}
    for cam in galah_CAMS:
        for env, cond in (("continuum", lambda q: q["tmpl"] >= 0.95), ("line", lambda q: q["tmpl"] < 0.95)):
            negs = sorted((q["snr"] for q in allpk if q["cam"] == cam and q["sign"] < 0 and q["isolated"] and cond(q)),
                          reverse=True)
            out[(cam, env)] = negs[0] if negs else 0.0
            npos = sum(1 for q in allpk if q["cam"] == cam and q["sign"] > 0 and q["isolated"] and cond(q))
            print("camera %s %-9s: isolated emission-like %d, absorption-like %d, strongest absorption-like %s"
                  % (cam, env, npos, len(negs), ", ".join("%.1f" % v for v in negs[:3])))
    cands = [q for q in allpk if q["sign"] > 0 and q["isolated"]
             and q["snr"] > out[(q["cam"], "continuum" if q["tmpl"] >= 0.95 else "line")]]
    print("%d isolated emission-like peaks above the null" % len(cands))
    for q in sorted(cands, key=lambda q: -q["snr"])[:30]:
        print("  %s (Gaia DR3 %s) camera %s %.3f A rest, %.3f A observatory, S/N %.1f, width %.2f, template %.2f"
              % (q["sid"], q["star"], q["cam"], q["lam"], q["lam_topo"], q["snr"], q["width"], q["tmpl"]))
    json.dump(allpk, open(os.path.join(galah_HERE, "galah_peaks.json"), "w"))


def galah_main():
    os.makedirs(galah_DL, exist_ok=True)
    if sys.argv[1] == "fetch":
        rows = list(csv.DictReader(open(os.path.join(galah_HERE, "galah_sample.csv"))))
        jobs = [(r["sobject_id"], c) for r in rows for c in galah_CAMS]
        with ThreadPoolExecutor(16) as ex:
            got = sum(ex.map(galah_fetch_one, jobs))
        print("%d of %d files" % (got, len(jobs)))
    else:
        galah_vet(galah_search())


# exposures mode

def galah_tap(q):
    """Rows of ADQL on Data Central's TAP service."""
    return core.tap("datacentral", q)

def galah_highpass(w, d, width=1.5):
    from scipy.ndimage import median_filter
    step = np.median(np.diff(w))
    base = median_filter(d, size=max(3, int(width / step)) | 1, mode="nearest")
    hp = d / np.maximum(base, 1e-3) - 1.0
    return hp

def galah_exposures_main():
    pk = json.load(open(core.results("galah_peaks.json")))
    negmax = {}
    for cam in "BVRI":
        for env in ("continuum", "line"):
            f = (lambda q: q["tmpl"] >= 0.95) if env == "continuum" else (lambda q: q["tmpl"] < 0.95)
            v = [q["snr"] for q in pk if q["cam"] == cam and q["sign"] < 0 and q["isolated"] and f(q)]
            negmax[(cam, env)] = max(v) if v else 0
    cands = [q for q in pk if q["sign"] > 0 and q["isolated"]
             and q["snr"] > negmax[(q["cam"], "continuum" if q["tmpl"] >= 0.95 else "line")]]
    out = []
    for q in sorted(cands, key=lambda q: -q["snr"]):
        pre = q["sid"][:12]
        rows = galah_tap("SELECT sobject_id, rv_comp_1 FROM galah_dr4.mainspectable WHERE sobject_id BETWEEN %s000 AND %s999"
                   % (pre, pre))
        rows = [r for r in rows if r["sobject_id"] != q["sid"] and r["rv_comp_1"]][:60]
        os.makedirs(galah_DL, exist_ok=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(galah_fetch_one, [(r["sobject_id"], q["cam"]) for r in rows]))
        vb = None
        hits, vals, n = 0, [], 0
        lam_o = q["lam_topo"]
        for r in rows:
            x = galah_load(r["sobject_id"], q["cam"])
            if x is None:
                continue
            w, d, meta = x
            if vb is None:
                vb = galah_baryc(meta)
            # barycentric wavelengths to the observatory's
            wo = w * (1.0 - vb / galah_C)
            hp = galah_highpass(w, d)
            m = np.abs(wo - lam_o) < lam_o / galah_R_RES
            if not m.any():
                continue
            n += 1
            sig = 1.4826 * np.median(np.abs(hp - np.median(hp)))
            vals.append(float(hp[m].max()))
            if hp[m].max() > 5 * sig:
                hits += 1
        rec = dict(sid=q["sid"], cam=q["cam"], lam=q["lam"], lam_obs=lam_o, snr=q["snr"], tmpl=q["tmpl"],
                   fibres=n, fibres_with_peak=hits, median_highpass=round(float(np.median(vals)) if vals else float("nan"), 4))
        out.append(rec)
        print("%s camera %s %.3f A rest (%.3f observatory) S/N %.1f template %.2f: %d of %d fibres in the same exposure "
              "peak there, median high-pass %+.4f" % (q["sid"], q["cam"], q["lam"], lam_o, q["snr"], q["tmpl"], hits, n,
                                                     rec["median_highpass"]), flush=True)
    json.dump(out, open(core.results("galah_exposure_check.json"), "w"), indent=0)


def galah_candidates(pk):
    """The null per camera and environment (the strongest isolated absorption-like S/N) and the isolated emission-like peaks above it."""
    null = {}
    for cam in galah_CAMS:
        for env in ("continuum", "line"):
            v = [q["snr"] for q in pk if q["cam"] == cam and q["sign"] < 0 and q["isolated"] and (q["tmpl"] >= 0.95) == (env == "continuum")]
            null[(cam, env)] = max(v) if v else 0.0
    return null, [q for q in pk if q["sign"] > 0 and q["isolated"] and q["snr"] > null[(q["cam"], "continuum" if q["tmpl"] >= 0.95 else "line")]]


def galah_peak_at(w, d, lam, tol):
    """The highest high-pass excess within tol (A) of lam over the spectrum's robust scatter, or None where it does not reach."""
    m = np.abs(w - lam) < tol
    if not m.any():
        return None
    hp = galah_highpass(w, d)
    sig = 1.4826 * np.nanmedian(np.abs(hp - np.nanmedian(hp)))
    return float(np.nanmax(hp[m]) / sig) if sig > 0 else None


def galah_check_main():
    """Each candidate in the same fibre's other runs of its night at the same observatory wavelength, where a detector or fibre defect recurs, and in the star's other GALAH spectra at the same rest wavelength, where a line of the star recurs."""
    pk = json.load(open(core.results("galah_peaks.json")))
    exp = {(r["sid"], r["cam"], round(r["lam"], 2)): r for r in json.load(open(core.results("galah_exposure_check.json")))} \
        if os.path.exists(core.results("galah_exposure_check.json")) else {}
    null, cands = galah_candidates(pk)
    out = []
    for q in sorted(cands, key=lambda q: -q["snr"]):
        tol = q["lam"] / galah_R_RES
        day, pivot = int(q["sid"][:6]), int(q["sid"][-3:])
        fib = galah_tap("SELECT sobject_id FROM galah_dr4.mainspectable WHERE sobject_id BETWEEN %d AND %d AND MOD(sobject_id, 1000) = %d"
                        % (day * 10 ** 9, day * 10 ** 9 + 999999999, pivot))
        fib = [r["sobject_id"] for r in fib if r["sobject_id"] != q["sid"]]
        ep = [r for r in galah_tap("SELECT sobject_id, rv_comp_1 FROM galah_dr4.mainspectable WHERE gaiadr3_source_id = %s" % q["star"])
              if r["sobject_id"] != q["sid"] and core.num(r["rv_comp_1"]) is not None]
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(galah_fetch_one, [(s, q["cam"]) for s in fib] + [(r["sobject_id"], q["cam"]) for r in ep]))
        f_snr = []
        for s in fib:
            x = galah_load(s, q["cam"])
            if x is not None:
                w, d, meta = x
                v = galah_peak_at(w * (1.0 - galah_baryc(meta) / galah_C), d, q["lam_topo"], tol)
                if v is not None:
                    f_snr.append(v)
        e_snr = []
        for r in ep:
            x = galah_load(r["sobject_id"], q["cam"])
            if x is not None:
                w, d, meta = x
                v = galah_peak_at(w / (1.0 + float(r["rv_comp_1"]) / galah_C), d, q["lam"], tol)
                if v is not None:
                    e_snr.append(v)
        e = exp.get((q["sid"], q["cam"], round(q["lam"], 2)), {})
        rec = dict(sid=q["sid"], star=q["star"], cam=q["cam"], lam=q["lam"], lam_obs=q["lam_topo"], snr=q["snr"], width=q["width"],
                   tmpl=q["tmpl"], exposure_fibres=e.get("fibres"), exposure_peaks=e.get("fibres_with_peak"),
                   fibre_runs=len(f_snr), fibre_peaks=int(sum(v > 5 for v in f_snr)), fibre_max=max(f_snr) if f_snr else None,
                   epochs=len(e_snr), epoch_peaks=int(sum(v > 5 for v in e_snr)), epoch_max=max(e_snr) if e_snr else None)
        rec["vet"] = galah_vets(rec)
        out.append(rec)
        print("%s (Gaia DR3 %s) camera %s %.2f A rest, %.2f observatory, S/N %.0f: same exposure %s of %s fibres; same fibre %d of %d "
              "runs that night (max %s); the star's other spectra %d of %d (max %s); %s" % (
                  q["sid"], q["star"], q["cam"], q["lam"], q["lam_topo"], q["snr"], rec["exposure_peaks"], rec["exposure_fibres"],
                  rec["fibre_peaks"], rec["fibre_runs"], "%.1f" % rec["fibre_max"] if rec["fibre_max"] is not None else "-",
                  rec["epoch_peaks"], rec["epochs"], "%.1f" % rec["epoch_max"] if rec["epoch_max"] is not None else "-",
                  "; ".join(rec["vet"]) or "unexplained"), flush=True)
    json.dump(out, open(core.results("galah_candidate_check.json"), "w"), indent=1)


def galah_vets(rec, raw=None):
    """Vet codes for a checked candidate: peaks in two or more other fibres of the exposure, in the same fibre on another run of the night, a star seen again without it, or a peak narrower than three quarters of the instrumental profile at 10 sigma, which light through the spectrograph cannot make (the unresolved-line rule of the ESO search); and from the raw frames (galah_raw_one), no light the fibre holds in every exposure near the peak's column while one exposure holds an excess there, or in a block of one exposure the excess there off the fibre's trace."""
    v = []
    if (rec.get("exposure_peaks") or 0) >= 2:
        v.append("neighbouring-fibres")
    if rec["fibre_peaks"] >= 1:
        v.append("fibre-fixed")
    if rec["epochs"] and not rec["epoch_peaks"]:
        v.append("single-epoch")
    if rec["snr"] >= 10.0 and rec["width"] < 0.75:
        v.append("detector-pixel")
    if raw and raw.get("runs"):
        if len(raw["runs"]) >= 2 and raw["single"] and not any(all(p["light"]) for p in raw["persistent"]):
            v.append("cosmic-ray")
        if len(raw["runs"]) >= 2 and any(not any(p["light"]) and max(e["fwhm_x"] for e in p["events"]) < 0.6 * float(np.median(
                [a["fwhm_x"] for a in raw["arc_lines"] if abs(a["dy"]) < 1] or [3.0])) for p in raw["persistent"]):
            v.append("detector-pixel")
        if len(raw["runs"]) == 1 and raw["persistent"]:
            top = max(raw["persistent"], key=lambda p: p["min_snr"])
            if abs(top["events"][0]["dy"]) > 1.0:
                v.append("off-trace")
    return v


# the raw HERMES frames of a candidate: its observation block from the AAT archive, the fibre's trace and its neighbours', and
# whether the excess near the peak's column is light the fibre holds in every exposure or charge that one exposure holds

galah_RAW = os.path.join(galah_HERE, "hermes_raw")
galah_NCOL = 4096
galah_GAIN = 2.75
galah_EV_PAIR = 3.65
# camera I of 2017-01-02, fibre 47: the raw column, observatory wavelength and dispersion where the reduction places the charge
# spot of 170102001601044, from which a peak's column elsewhere is predicted to a few hundred pixels
galah_RAW_ANCHOR = (2578.3, 7791.79, 0.0735)
galah_RAW_HALF = 400


def galah_raw_run(r):
    m = re.search(r"(\d{4})\.f(?:i)?ts$", r["filename"], re.I)
    return int(m.group(1)) if m else -1


def galah_raw_plate(r):
    return (r["fits_header"] if isinstance(r["fits_header"], dict) else {}).get("SOURCE")


def galah_raw_block(rows, first):
    """Camera IR's (E2V8-IR) science runs of the observation that begins at run FIRST (the first object run from there and those of the same field after it), with the flat and the arc of the same plate nearest them, before or after."""
    run = {galah_raw_run(r): r for r in rows if r["detector"] == "E2V8-IR"}
    sci, obj = [], None
    for k in sorted(k for k in run if k >= first):
        r = run[k]
        if r["ndf_class"] == "MFOBJECT":
            if obj is None:
                obj = r["OBJECT"]
            if r["OBJECT"] != obj:
                break
            sci.append(r)
        elif sci:
            break
    if not sci:
        return [], [], []
    k0, k1, pl = galah_raw_run(sci[0]), galah_raw_run(sci[-1]), galah_raw_plate(sci[0])

    def nearest(cls):
        c = [k for k in run if run[k]["ndf_class"] == cls and galah_raw_plate(run[k]) == pl and (k0 - 4 <= k < k0 or k1 < k <= k1 + 4)]
        return [run[min(c, key=lambda k: min(abs(k - k0), abs(k - k1)))]] if c else []
    return nearest("MFARC"), nearest("MFFFF"), sci


def galah_raw_load(path):
    """A raw HERMES frame less its overscan's bias, row by row, and its fibre table."""
    from astropy.io import fits
    h = fits.open(path)
    a = np.asarray(h[0].data, float)
    return a[:, :galah_NCOL] - np.median(a[:, galah_NCOL + 4:], axis=1)[:, None], h[1].data


def galah_raw_peaks(flat):
    from scipy.signal import find_peaks
    prof = np.median(flat[:, galah_NCOL // 2 - 20:galah_NCOL // 2 + 20], axis=1)
    return find_peaks(prof, distance=5, prominence=0.02 * np.percentile(prof, 99))[0]


def galah_raw_walk(pk, start=1):
    """{fibre: y at the detector's centre}, walking the slit from the flat's first trace as fibre START: each next fibre one spacing on, plus the slitlet gap after every tenth fibre, the spacing and the gap following the traces already placed; a fibre with no trace within a third of a spacing of where it is expected is dark."""
    d = np.diff(pk)
    sp = float(np.median(d[:8]))
    big = d[(d > 1.8 * sp) & (d < 3.0 * sp)]
    gap = float(np.median(big) - sp) if len(big) else 13.0
    pos, last_f, last_y, i = {start: int(pk[0])}, start, float(pk[0]), 1
    recent = [sp]
    for f in range(start + 1, 401):
        if i >= len(pk):
            break
        s = float(np.median(recent[-8:]))
        bounds = (f - 1) // 10 - (last_f - 1) // 10
        pred = last_y + (f - last_f) * s + bounds * gap
        if abs(pk[i] - pred) < s / 3:
            if f - last_f == 1 and bounds == 0:
                recent.append(pk[i] - last_y)
            elif f - last_f == 1 and bounds == 1:
                gap = 0.7 * gap + 0.3 * (pk[i] - last_y - s)
            pos[f], last_f, last_y = int(pk[i]), f, float(pk[i])
            i += 1
        elif pk[i] < pred - s / 3:
            i += 1
    return pos


def galah_raw_trace(flat, y0, half=4, step=32):
    """The trace through (centre, y0) in the flat: centroids every STEP columns, a cubic in x."""
    xs, ys = [], []
    for direction in (1, -1):
        y, x = float(y0), galah_NCOL // 2
        while 40 <= x < galah_NCOL - 40:
            yi = int(round(y))
            win = np.median(flat[yi - half:yi + half + 1, x - 8:x + 8], axis=1)
            win = win - win.min()
            if win.sum() > 0:
                c = float(np.sum(win * np.arange(-half, half + 1)) / win.sum())
                if abs(c) < 2:
                    y = yi + c
                    xs.append(x)
                    ys.append(y)
            x += direction * step
    p = np.polyfit(np.array(xs) - galah_NCOL / 2, ys, 3)
    return lambda x: np.polyval(p, np.asarray(x, float) - galah_NCOL / 2)


def galah_raw_extract(img, ytr, half=3):
    x = np.arange(galah_NCOL)
    rows = np.clip(np.round(ytr(x)).astype(int)[None, :] + np.arange(-half, half + 1)[:, None], 0, img.shape[0] - 1)
    return img[rows, x[None, :]].sum(axis=0)


def galah_raw_gauss(v):
    """Centre (from the middle) and FWHM of a profile by a Gaussian with a constant."""
    from scipy.optimize import curve_fit
    x = np.arange(len(v)) - (len(v) - 1) / 2.0
    try:
        p, _ = curve_fit(lambda x, a, m, s, c: a * np.exp(-0.5 * ((x - m) / s) ** 2) + c, x, v,
                         p0=[float(v.max()), float(x[np.argmax(v)]), 1.5, 0.0], maxfev=20000)
        return float(p[1]), float(2.3548 * abs(p[2]))
    except Exception:
        return np.nan, np.nan


def galah_raw_event(img, ytr, x0, search=2, h=6):
    """The brightest excess within SEARCH columns of x0 near the trace, each row's continuum taken from columns 12-40 px away: its centre, FWHM along and across the dispersion, offset from the trace's centre (fitted to the same rows' continuum), the trace's FWHM there, and its charge in electrons and in the energy that releases it in silicon."""
    yi, xi = int(round(float(ytr(x0)))), int(round(x0))
    st = img[yi - 14:yi + 15, xi - 40:xi + 41]
    cont = np.median(np.concatenate([st[:, :28], st[:, -28:]], axis=1), axis=1)
    r = st - cont[:, None]
    sub = r[10:19, 40 - search:40 + search + 1]
    j = np.unravel_index(np.argmax(sub), sub.shape)
    py, px = j[0] + 10, j[1] + 40 - search
    box = r[py - h:py + h + 1, px - h:px + h + 1]
    mx, fx = galah_raw_gauss(box.sum(axis=0))
    my, fy = galah_raw_gauss(box.sum(axis=1))
    bg = np.median(img[yi - 40:yi + 41, xi - 40:xi + 41])
    ty, tf = galah_raw_gauss((cont - min(bg, cont.min()))[8:21])
    e = float(box.sum()) * galah_GAIN
    return dict(x=float(xi - 40 + px + mx), y=float(yi - 14 + py + my), dy=float(py - 14 + my - ty), fwhm_x=fx, fwhm_y=fy, trace_fwhm=tf,
                peak_adu=float(r[py, px]), electrons=e, mev=e * galah_EV_PAIR / 1e6)


def galah_raw_arcs(arc, ytr, x0, n=10):
    """The brightest unsaturated arc lines within 600 px of x0 on the trace, each measured as an event."""
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    e = galah_raw_extract(arc, ytr)
    lo, hi = max(60, int(x0) - 600), min(galah_NCOL - 60, int(x0) + 600)
    seg = e[lo:hi] - median_filter(e[lo:hi], size=41, mode="nearest")
    pk = find_peaks(seg, distance=15, prominence=20 * 1.4826 * np.median(np.abs(seg - np.median(seg))))[0]
    out = []
    for k in sorted(pk, key=lambda k: -seg[k]):
        ev = galah_raw_event(arc, ytr, lo + k, search=3)
        if ev["peak_adu"] < 40000 and np.isfinite(ev["fwhm_x"]):
            out.append(ev)
        if len(out) == n:
            break
    return out


def galah_raw_differences(img, flat, ytr, ntr):
    """The target less the median of its neighbours, in flux per unit throughput (where the night sky is common) and continuum- normalised (where the telluric absorption is), each high-passed in units of its scatter; and the first as a share of the target's continuum."""
    from scipy.ndimage import median_filter

    def per_throughput(tr):
        e, ef = galah_raw_extract(img, tr), galah_raw_extract(flat, tr)
        return e / np.where(ef > 0.05 * np.nanmedian(ef), ef, np.nan)

    def z(u):
        u = np.nan_to_num(u)
        hp = u - median_filter(u, size=51, mode="nearest")
        s = 1.4826 * np.median(np.abs(hp - np.median(hp)))
        return (hp / s if s > 0 else hp * 0), hp

    def cont(v):
        return median_filter(np.nan_to_num(v), size=51, mode="nearest")
    a = per_throughput(ytr)
    A = [per_throughput(t) for t in ntr]
    z1, hp1 = z(a - np.nanmedian(A, axis=0))
    ca = cont(a)
    z2, _ = z(a / np.where(ca > 0, ca, np.nan) - np.nanmedian([v / np.where(cont(v) > 0, cont(v), np.nan) for v in A], axis=0))
    return np.minimum(z1, z2), hp1 / np.where(ca > 0, ca, np.nan)


def galah_raw_light(ev, arcs):
    """Within 1 px of the trace's centre, where the arc lines fall, and as wide as light along and across the dispersion."""
    on = [a["fwhm_x"] for a in arcs if abs(a["dy"]) < 1]
    fx = float(np.median(on)) if on else 3.0
    return bool(abs(ev["dy"]) <= 1.0 and ev["fwhm_x"] >= 0.6 * fx and 0.6 * ev["trace_fwhm"] <= ev["fwhm_y"] <= 1.6 * ev["trace_fwhm"])


def galah_raw_one(sid, lam_obs):
    """One camera-I candidate in its raw frames: the excesses the fibre holds and its lit neighbours within six fibres do not, above 4 sigma at one column in every exposure (persistent) and above 6 sigma in one exposure with the others below 3 (single), within GALAH_RAW_HALF columns of the anchor's prediction, each with its shape against the arc lines'."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import find_peaks
    date = "20%s-%s-%s" % (sid[0:2], sid[2:4], sid[4:6])
    arcs, flats, sci = galah_raw_block(core.aat_night(date), int(sid[6:10]))
    if not (arcs and flats and sci):
        return dict(sid=sid, lam_obs=lam_obs, error="the archive holds no complete block from run %s" % sid[6:10])
    paths = core.aat_fetch(arcs + flats + sci, os.path.join(galah_RAW, sid[:6]))
    arc = galah_raw_load(paths[0])[0]
    flat, tab = galah_raw_load(paths[1])
    imgs = [galah_raw_load(p)[0] for p in paths[2:]]
    types = np.asarray(tab["TYPE"]).astype(str)
    fibre = int(np.nonzero(np.asarray(tab["PIVOT"]) == int(sid[-3:]))[0][0]) + 1
    pos = galah_raw_walk(galah_raw_peaks(flat))
    if fibre not in pos:
        return dict(sid=sid, lam_obs=lam_obs, error="fibre %d has no trace in the flat" % fibre)
    # the walk's placement checked by the fibre table's magnitudes against the light of the placed fibres
    stack = np.sum(imgs, axis=0)
    col = np.median(stack[:, 1848:2248], axis=1)
    lit = {f: float(np.median(col[y - 1:y + 2]) - np.median(col)) for f, y in pos.items()}
    P = [f for f in pos if types[f - 1] == "P" and lit[f] > 0]
    rho = float(np.corrcoef([float(tab["MAGNITUDE"][f - 1]) for f in P], [np.log10(lit[f]) for f in P])[0, 1])
    ytr = galah_raw_trace(flat, pos[fibre])
    ntr = [galah_raw_trace(flat, pos[f]) for f in pos if f != fibre and abs(f - fibre) <= 6 and types[f - 1] in ("P", "S")]
    xr = galah_RAW_ANCHOR[0] + (lam_obs - galah_RAW_ANCHOR[1]) / galah_RAW_ANCHOR[2]
    lo, hi = int(max(60, xr - galah_RAW_HALF)), int(min(galah_NCOL - 60, xr + galah_RAW_HALF))
    arcl = galah_raw_arcs(arc, ytr, galah_NCOL / 2.0)
    Z = [galah_raw_differences(img, flat, ytr, ntr) for img in imgs]
    M = [maximum_filter1d(z, 3) for z, rel in Z]
    win = (np.arange(galah_NCOL) >= lo) & (np.arange(galah_NCOL) < hi)
    persistent, single = [], []
    for k in find_peaks(np.where(win, np.min(M, axis=0), 0), height=4, distance=5)[0]:
        evs = [galah_raw_event(img, ytr, k) for img in imgs]
        persistent.append(dict(x=int(k), min_snr=float(np.min(M, axis=0)[k]), rel=[float(rel[k]) for z, rel in Z],
                               light=[galah_raw_light(ev, arcl) for ev in evs], events=evs))
    if len(imgs) > 1:
        for i, (z, rel) in enumerate(Z):
            for k in find_peaks(np.where(win, z, 0), height=6, distance=5)[0]:
                if any(M[j][k] >= 3 for j in range(len(imgs)) if j != i):
                    continue
                ev = galah_raw_event(imgs[i], ytr, k)
                single.append(dict(run=os.path.basename(paths[2 + i]), x=int(k), snr=float(z[k]), rel=float(rel[k]),
                                   light=galah_raw_light(ev, arcl), event=ev))
    return dict(sid=sid, lam_obs=lam_obs, fibre=fibre, rho=rho, runs=[os.path.basename(p) for p in paths[2:]],
                exposed=[r["EXPOSED"] for r in sci], window=[lo, hi], arc_lines=arcl, persistent=persistent, single=single)


def galah_raw_census(sid, span=20, x_min=1300):
    """How often an excess one exposure holds on a fibre's trace has the shape of light: across the lit fibres within SPAN of the candidate's, each against its own neighbours, the single-exposure excesses beyond the A band and the share of them light-shaped."""
    from scipy.signal import find_peaks
    date = "20%s-%s-%s" % (sid[0:2], sid[2:4], sid[4:6])
    arcs, flats, sci = galah_raw_block(core.aat_night(date), int(sid[6:10]))
    paths = core.aat_fetch(arcs + flats + sci, os.path.join(galah_RAW, sid[:6]))
    arc = galah_raw_load(paths[0])[0]
    flat, tab = galah_raw_load(paths[1])
    imgs = [galah_raw_load(p)[0] for p in paths[2:]]
    types = np.asarray(tab["TYPE"]).astype(str)
    target = int(np.nonzero(np.asarray(tab["PIVOT"]) == int(sid[-3:]))[0][0]) + 1
    pos = galah_raw_walk(galah_raw_peaks(flat))
    lit = [f for f in pos if types[f - 1] in ("P", "S")]
    tr = {f: galah_raw_trace(flat, pos[f]) for f in lit if abs(f - target) <= span + 7}
    arcl = galah_raw_arcs(arc, tr[target], galah_NCOL / 2.0)
    fibres = [f for f in lit if types[f - 1] == "P" and abs(f - target) <= span]
    n = light = 0
    for f in fibres:
        Z = [galah_raw_differences(img, flat, tr[f], [tr[g] for g in lit if g != f and abs(g - f) <= 6 and g in tr])[0] for img in imgs]
        for i, z in enumerate(Z):
            zz = z.copy()
            zz[:x_min] = 0
            zz[-60:] = 0
            for k in find_peaks(zz, height=6, distance=5)[0]:
                if any(Z[j][k - 1:k + 2].max() >= 3 for j in range(len(Z)) if j != i):
                    continue
                n += 1
                light += galah_raw_light(galah_raw_event(imgs[i], tr[f], k), arcl)
    cols = (galah_NCOL - 60 - x_min) * len(fibres)
    return dict(fibres=len(fibres), events=n, light=light, per_window=2.0 * galah_RAW_HALF * n / cols,
                light_per_window=2.0 * galah_RAW_HALF * light / cols)


def galah_raw_main():
    """Each candidate the catalogue checks leave unexplained, in its raw frames; a census beside a candidate whose window holds a light-shaped excess in one exposure."""
    chk = json.load(open(core.results("galah_candidate_check.json")))
    path = core.results("galah_raw_check.json")
    done = {(r["sid"], round(r["lam_obs"], 2)): r for r in json.load(open(path))} if os.path.exists(path) else {}
    for c in sorted(chk, key=lambda c: -c["snr"]):
        key = (c["sid"], round(c["lam_obs"], 2))
        if galah_vets(c) or key in done:
            continue
        r = galah_raw_one(c["sid"], c["lam_obs"])
        if any(s["light"] for s in r.get("single", [])):
            r["census"] = galah_raw_census(c["sid"])
        done[key] = r
        json.dump(list(done.values()), open(path, "w"), indent=1, default=float)
        if "error" in r:
            print("%s %.2f A: %s" % (c["sid"], c["lam_obs"], r["error"]), flush=True)
            continue
        print("%s %.2f A: %d exposures %s, fibre %d (placement r %.2f); window x %d-%d: in every exposure %s; in one exposure %s%s; vets %s" % (
            c["sid"], c["lam_obs"], len(r["runs"]), r["exposed"], r["fibre"], r["rho"], r["window"][0], r["window"][1],
            ["x %d %.0f sigma %s" % (p["x"], p["min_snr"], "light" if all(p["light"]) else "%+.1f px" % p["events"][0]["dy"]) for p in r["persistent"]
             ][:4], ["%s x %d %.0f sigma%s" % (s["run"], s["x"], s["snr"], " light-shaped" if s["light"] else "") for s in r["single"]][:6],
            "; census %d events in %d fibres, %d light-shaped, %.2f light-shaped expected in a window" % (
                r["census"]["events"], r["census"]["fibres"], r["census"]["light"], r["census"]["light_per_window"]) if "census" in r else "",
            galah_vets(c, r)), flush=True)


def galah_record():
    rows = list(csv.DictReader(open(os.path.join(galah_HERE, "galah_sample.csv"))))
    pk = json.load(open(core.results("galah_peaks.json")))
    null, cands = galah_candidates(pk)
    chk = {(r["sid"], r["cam"], round(r["lam"], 2)): r for r in json.load(open(core.results("galah_candidate_check.json")))}
    rp = core.results("galah_raw_check.json")
    raw = {(r["sid"], round(r["lam_obs"], 2)): r for r in json.load(open(rp))} if os.path.exists(rp) else {}
    best = {}
    for q in pk:
        if q["sign"] > 0 and q["isolated"] and q["snr"] > best.get(q["star"], (0,))[0]:
            best[q["star"]] = (q["snr"], q)
    stars = sorted({r["gaiadr3_source_id"] for r in rows})
    vetted = {}
    for q in cands:
        c = chk.get((q["sid"], q["cam"], round(q["lam"], 2)))
        vetted.setdefault(q["star"], []).append((q, galah_vets(c, raw.get((c["sid"], round(c["lam_obs"], 2)))) if c else []))
    left = [s for s, v in vetted.items() if any(not codes for q, codes in v)]
    core.record_search(
        "laser_galah", "Laser lines, GALAH DR4", archive="GALAH DR4 (Data Central SODA spectra, cameras B, V, R and I)", release="DR4",
        selection="%d spectra of %d dwarfs within 100 pc (galah_sample.csv)" % (len(rows), len(stars)),
        code=os.path.abspath(__file__), entry="galah", n_targets=len(stars), n_unit="stars",
        params=dict(threshold_snr=5.5, neighbours=20, resolution=galah_R_RES, isolated="at most one other star at the rest or the "
                    "observatory wavelength, width 0.6-1.6 of the instrument's", unresolved="at least 0.75 of the instrument's width at 10 sigma"),
        null_summary="per camera and environment the strongest isolated absorption-like peak: %s" % ", ".join(
            "%s %s %.1f" % (c, e, v) for (c, e), v in sorted(null.items())),
        sensitivity="an unresolved emission line above the null in a spectrum shifted to the star's rest frame by the catalogue's velocity",
        ledger="Laser lines, GALAH DR4",
        result="%d isolated emission-like peaks above the null in %d stars, all checked in the other fibres of their exposure, the same "
               "fibre's other runs of the night, the star's other spectra and against the instrument's width, and those left in their raw "
               "HERMES exposures; %d stars with a peak no check explains" % (len(cands), len(vetted), len(left)),
        outputs=[core.results("galah_peaks.json"), core.results("galah_exposure_check.json"), core.results("galah_candidate_check.json")]
        + ([rp] if raw else []))
    core.clear_targets("laser_galah")
    recs = []
    for s in stars:
        b = best.get(s)
        v = vetted.get(s, [])
        codes = sorted({c for q, cs in v for c in cs}) if v and all(cs for q, cs in v) else []
        note = "; ".join("%s camera %s %.2f A S/N %.0f%s" % (q["sid"], q["cam"], q["lam"], q["snr"], ": " + ", ".join(cs) if cs else ": unexplained")
                         for q, cs in v) if v else None
        recs.append(dict(target="gaia_dr3:%s" % s, stat=round(b[0], 2) if b else None, unit="S/N of the strongest isolated emission-like peak",
                         vet="; ".join(codes) or None, note=note))
    core.record_targets("laser_galah", recs)
    print("recorded %d stars; %d with candidates, %d unexplained" % (len(recs), len(vetted), len(left)))


def galah_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "exposures":
        sys.argv.pop(1)
        galah_exposures_main()
    elif mode == "check":
        galah_check_main()
    elif mode == "raw":
        galah_raw_main()
    elif mode == "record":
        galah_record()
    else:
        galah_main()


# --------------------------------------------------------------------------------------------------------------

apogee_DOC = r"""Steady laser lines in APOGEE DR17 combined spectra of nearby dwarfs against twin-star fits, with dips as null and sharing and airglow vetoes.

    python -I seti.py lasers apogee pack
    python -I seti.py lasers apogee search [--k 12] [--thr 5] [--procs 2]
    python -I seti.py lasers apogee vet [--top 30]"""


apogee_HERE = core.RESULTS
apogee_DL = os.path.join(apogee_HERE, "apogee_dl")
apogee_OUT = os.path.join(apogee_HERE, "apogee_combined")
apogee_NPIX = 8575
apogee_LAM = 10 ** (4.179 + 6e-6 * np.arange(apogee_NPIX))
apogee_SIG_PX = science.lsf_sigma_px(22500.0, dlog10=6e-6)
# masked pixel bits; PERSIST_MED and PERSIST_LOW kept and reported
apogee_BAD = sum(1 << b for b in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14))
apogee_PERSIST = (1 << 10) | (1 << 11)
# peaks under subtracted sky above this fraction of the star's flux are airglow
apogee_SKY_LIMIT = 0.3


def apogee_pack():
    """Combined spectrum, error and mask of each apStar file as .npy arrays."""
    from astropy.io import fits
    os.makedirs(apogee_OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(apogee_DL, "apStar-dr17-*.fits")))
    aids, F, E, M, S, meta = [], [], [], [], [], []
    for p in files:
        try:
            with fits.open(p) as h:
                H = h[0].header
                f = np.asarray(h[1].data, np.float32)
                e = np.asarray(h[2].data, np.float32)
                m = np.asarray(h[3].data, np.int32)
                sk = np.asarray(h[4].data, np.float32)
                if f.ndim == 1:
                    f, e, m, sk = f[None], e[None], m[None], sk[None]
                nv = int(H.get("NVISITS", 1))
                rows = range(2, 2 + nv) if f.shape[0] >= nv + 2 else range(0, 1)
                # brightest removed sky emission relative to the visit's median flux
                skyfrac = np.max([sk[r] / max(float(np.median(f[r][f[r] > 0])) if np.any(f[r] > 0) else 1.0, 1e-9)
                                  for r in rows], axis=0)
                vrel = []
                for i in range(1, nv + 1):
                    vh, b = H.get("VHELIO%d" % i, H.get("VRAD%d" % i)), H.get("BC%d" % i)
                    vrel.append(round(float(vh) - float(b), 3) if vh is not None and b is not None else None)
        except Exception:
            continue
        if f.shape[1] != apogee_NPIX:
            continue
        aids.append(os.path.basename(p)[12:-5])
        F.append(f[0])
        E.append(e[0])
        M.append(m[0])
        S.append(skyfrac.astype(np.float32))
        meta.append(dict(nv=nv, snr=float(H.get("SNR", np.nan)), vhelio=float(H.get("VHELIO", np.nan)),
                         teff=float(H.get("RV_TEFF", np.nan)), logg=float(H.get("RV_LOGG", np.nan)),
                         feh=float(H.get("RV_FEH", np.nan)), vrel=vrel))
    np.save(os.path.join(apogee_OUT, "flux.npy"), np.array(F))
    np.save(os.path.join(apogee_OUT, "err.npy"), np.array(E))
    np.save(os.path.join(apogee_OUT, "mask.npy"), np.array(M))
    np.save(os.path.join(apogee_OUT, "skyfrac.npy"), np.array(S))
    json.dump(dict(aids=aids, meta=meta), open(os.path.join(apogee_OUT, "meta.json"), "w"))
    print("%d combined spectra packed" % len(aids), flush=True)


def apogee_chips(good):
    """Pixel ranges of the three detectors."""
    frac = good.mean(0) > 0.5
    out, s = [], None
    for j, v in enumerate(frac):
        if v and s is None:
            s = j
        if not v and s is not None:
            if j - s > 500:
                out.append((s, j))
            s = None
    if s is not None and apogee_NPIX - s > 500:
        out.append((s, apogee_NPIX))
    return out


def apogee_normalise(F, good, ranges):
    N = np.full(F.shape, np.nan, np.float32)
    for i in range(len(F)):
        for lo, hi in ranges:
            f = np.where(good[i, lo:hi], F[i, lo:hi], np.nan)
            env = science.envelope(f, block=100, q=90.0, smooth=3)
            N[i, lo:hi] = f / env
    return N


def apogee_neighbours(N, good, k):
    """k most similar spectra per spectrum on the GPU."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = torch.as_tensor(good, dtype=torch.float32, device=dev)
    X = torch.as_tensor(np.nan_to_num(N, nan=0.0), dtype=torch.float32, device=dev) * W
    X2 = X * X
    overlap = W @ W.T
    d = (X2 @ W.T + W @ X2.T - 2.0 * X @ X.T) / overlap.clamp(min=1.0)
    d[overlap < 0.5 * W.sum(1, keepdim=True)] = float("inf")
    d.fill_diagonal_(float("inf"))
    return torch.topk(d, k, largest=False).indices.cpu().numpy()


def apogee_fill(row, ok):
    """Comparison spectrum with bad samples interpolated."""
    if ok.all():
        return row
    x = np.arange(len(row))
    if ok.sum() < 2:
        return np.ones_like(row)
    return np.interp(x, x[ok], row[ok])


apogee__S = {}


def apogee__init(k):
    apogee__S["F"] = np.load(os.path.join(apogee_OUT, "flux.npy"), mmap_mode="r")
    apogee__S["E"] = np.load(os.path.join(apogee_OUT, "err.npy"), mmap_mode="r")
    apogee__S["M"] = np.load(os.path.join(apogee_OUT, "mask.npy"), mmap_mode="r")
    apogee__S["N"] = np.load(os.path.join(apogee_OUT, "norm.npy"), mmap_mode="r")
    apogee__S["G"] = np.load(os.path.join(apogee_OUT, "good.npy"), mmap_mode="r")
    apogee__S["SKY"] = np.load(os.path.join(apogee_OUT, "skyfrac.npy"), mmap_mode="r")
    apogee__S["NB"] = np.load(os.path.join(apogee_OUT, "neighbours.npy"))[:, :k]
    apogee__S["R"] = json.load(open(os.path.join(apogee_OUT, "chips.json")))


def apogee_one(args):
    """Peaks of both signs in one star's twin-model residual."""
    i, thr = args
    quiet_warnings()
    N, G, nb = apogee__S["N"], apogee__S["G"], apogee__S["NB"][i]
    t = np.asarray(N[i], float)
    g = np.asarray(G[i], bool)
    B = np.array([apogee_fill(np.asarray(N[j], float), np.asarray(G[j], bool)) for j in nb])
    model = np.full(apogee_NPIX, np.nan)
    tmpl = np.full(apogee_NPIX, np.nan)
    for lo, hi in apogee__S["R"]:
        for s in range(lo, hi, 128):
            a, b = max(s - 64, lo), min(s + 192, hi)
            m = g[a:b] & np.isfinite(t[a:b])
            if m.sum() < 80:
                continue
            mod, tp = science.twin_model(np.nan_to_num(t[a:b], nan=1.0), B[:, a:b], m, poly_order=2)
            c0, c1 = s, min(s + 128, hi)
            model[c0:c1] = mod[c0 - a:c1 - a]
            tmpl[c0:c1] = tp[c0 - a:c1 - a]
    res = np.where(g & np.isfinite(model) & (model > 0.05), t / model - 1.0, np.nan)
    ferr = np.asarray(apogee__S["E"][i], float) / np.maximum(np.asarray(apogee__S["F"][i], float), 1e-9)
    sig = np.maximum(np.where(np.isfinite(ferr), ferr, np.inf), science.block_mad(res, block=200, floor=1e-3))
    snr, amp = science.matched_filter(res, sig, apogee_SIG_PX)
    near_bad = np.convolve((~g).astype(float), np.ones(5), mode="same") > 0
    mk = np.asarray(apogee__S["M"][i], np.int64)
    out = []
    for sign in (1, -1):
        for j in science.find_peaks(sign * snr, thr, sep=6, mask=~near_bad & np.isfinite(snr), limit=300):
            out.append(dict(i=int(i), sign=sign, pix=int(j), lam=round(float(apogee_LAM[j]), 3), snr=round(float(sign * snr[j]), 2),
                            amp=round(float(sign * amp[j]), 5),
                            width=round(science.width_ratio(sign * res, j, apogee_SIG_PX), 2),
                            spiky=bool(science.profile_is_spiky(np.nan_to_num(sign * res), j, apogee_SIG_PX, base=0.0)),
                            tmpl=round(float(tmpl[j]), 3) if np.isfinite(tmpl[j]) else None,
                            persist=bool(np.any(mk[max(j - 2, 0):j + 3] & apogee_PERSIST)),
                            sky=round(float(np.nanmax(apogee__S["SKY"][i, max(j - 2, 0):j + 3])), 3)))
    scat = float(np.nanmedian(sig[np.isfinite(res)])) if np.isfinite(res).any() else float("nan")
    return dict(i=int(i), peaks=out, scatter=scat, used=int(np.isfinite(res).sum()))


def apogee_search(k, thr, procs, limit=None):
    meta = json.load(open(os.path.join(apogee_OUT, "meta.json")))
    aids = meta["aids"]
    F = np.load(os.path.join(apogee_OUT, "flux.npy"))
    E = np.load(os.path.join(apogee_OUT, "err.npy"))
    M = np.load(os.path.join(apogee_OUT, "mask.npy"))
    medE = np.nanmedian(np.where(F > 0, E, np.nan), axis=1)
    good = ((M.astype(np.int64) & apogee_BAD) == 0) & (F > 0) & np.isfinite(E) & (E > 0) & (E < 50 * medE[:, None])
    ranges = apogee_chips(good)
    print("%d stars; detector ranges %s" % (len(aids), [(int(a), int(b)) for a, b in ranges]), flush=True)
    N = apogee_normalise(F, good, ranges)
    good &= np.isfinite(N)
    np.save(os.path.join(apogee_OUT, "norm.npy"), N)
    np.save(os.path.join(apogee_OUT, "good.npy"), good)
    json.dump([[int(a), int(b)] for a, b in ranges], open(os.path.join(apogee_OUT, "chips.json"), "w"))
    nb = apogee_neighbours(N, good, max(k, 20))
    np.save(os.path.join(apogee_OUT, "neighbours.npy"), nb)
    del F, E, M, N
    idx = list(range(len(aids)))[:limit] if limit else list(range(len(aids)))
    jobs = [(i, thr) for i in idx]
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs, initializer=apogee__init, initargs=(k,)) as pool:
            res = pool.map(apogee_one, jobs, chunksize=8)
    else:
        apogee__init(k)
        res = [apogee_one(j) for j in jobs]
    peaks = [dict(q, aid=aids[q["i"]]) for r in res for q in r["peaks"]]
    for sign in (1, -1):
        sel = [q for q in peaks if q["sign"] == sign and not q["spiky"] and 0.6 <= q["width"] <= 1.6]
        pix = np.array([q["pix"] for q in sel])
        st = np.array([q["i"] for q in sel])
        for q in sel:
            q["shared"] = int(np.sum((np.abs(pix - q["pix"]) <= 1) & (st != q["i"])))
    for q in peaks:
        q.setdefault("shared", -1)
        q["clean"] = bool(not q["spiky"] and 0.6 <= q["width"] <= 1.6 and 0 <= q["shared"] <= 2
                          and not q["sky"] > apogee_SKY_LIMIT)
        q["env"] = "continuum" if (q["tmpl"] or 0) >= 0.95 else "line"
    json.dump(dict(peaks=peaks, stars=[dict(aid=aids[r["i"]], scatter=r["scatter"], used=r["used"]) for r in res]),
              open(os.path.join(apogee_HERE, "apogee_steady.json"), "w"))
    print("%d stars searched, %d peaks" % (len(res), len(peaks)))
    for env in ("continuum", "line"):
        neg = sorted((q["snr"] for q in peaks if q["sign"] < 0 and q["clean"] and q["env"] == env), reverse=True)
        pos = sorted((q for q in peaks if q["sign"] > 0 and q["clean"] and q["env"] == env), key=lambda q: -q["snr"])
        top = neg[0] if neg else 0.0
        print("%s: clean absorption-like %d (strongest %s), clean emission-like %d"
              % (env, len(neg), ", ".join("%.1f" % v for v in neg[:5]), len(pos)))
        for t in (5, 6, 8, 10, 15):
            print("  S/N>%d: emission-like %d, absorption-like %d"
                  % (t, sum(1 for q in pos if q["snr"] > t), sum(1 for v in neg if v > t)))
        for q in pos[:15]:
            band = " telecom C band" if 15300 < q["lam"] < 15650 else ""
            print("  %s%s %.3f A S/N %.1f amp %.4f width %.2f template %.3f%s%s"
                  % ("ABOVE-NULL " if q["snr"] > top else "", q["aid"], q["lam"], q["snr"], q["amp"], q["width"],
                     q["tmpl"] or 0, " persistence" if q["persist"] else "", band))


def apogee_vet(top):
    """Per-visit amplitude, removed sky and telluric spectra, and observatory wavelength of each emission peak above the null."""
    from astropy.io import fits
    apogee__init(12)
    meta = json.load(open(os.path.join(apogee_OUT, "meta.json")))
    d = json.load(open(os.path.join(apogee_HERE, "apogee_steady.json")))
    peaks = d["peaks"]
    null = {env: max([q["snr"] for q in peaks if q["sign"] < 0 and q["clean"] and q["env"] == env] or [0.0])
            for env in ("continuum", "line")}
    cands = sorted((q for q in peaks if q["sign"] > 0 and q["clean"]), key=lambda q: -q["snr"] / null[q["env"]])[:top]
    out = []
    for q in cands:
        i, j = q["i"], q["pix"]
        lo, hi = next((a, b) for a, b in apogee__S["R"] if a <= j < b)
        a, b = max(j - 128, lo), min(j + 128, hi)
        t = np.asarray(apogee__S["N"][i], float)
        g = np.asarray(apogee__S["G"][i], bool)
        B = np.array([apogee_fill(np.asarray(apogee__S["N"][n], float), np.asarray(apogee__S["G"][n], bool)) for n in apogee__S["NB"][i]])
        m = g[a:b] & np.isfinite(t[a:b]) & (np.abs(np.arange(a, b) - j) > 4)
        model, _ = science.twin_model(np.nan_to_num(t[a:b], nan=1.0), B[:, a:b], m, poly_order=2)
        path = os.path.join(apogee_DL, "apStar-dr17-%s.fits" % q["aid"])
        with fits.open(path) as h:
            nv = int(h[0].header.get("NVISITS", 1))
            rows = range(2, 2 + nv) if h[1].data.ndim == 2 and h[1].data.shape[0] >= nv + 2 else range(0, 1)
            V = [np.asarray(h[1].data[r], float) for r in rows]
            Ev = [np.asarray(h[2].data[r], float) for r in rows]
            Mv = [np.asarray(h[3].data[r], np.int64) for r in rows]
            Sv = [np.asarray(h[4].data[r], float) for r in rows]
            Tv = [np.asarray(h[6].data[r], float) for r in rows]
        vrel = meta["meta"][i]["vrel"]
        visits = []
        k_ = np.arange(-int(4 * apogee_SIG_PX) - 1, int(4 * apogee_SIG_PX) + 2)
        gk = np.exp(-0.5 * (k_ / apogee_SIG_PX) ** 2)
        for k, (v, e, mk, sky, tel) in enumerate(zip(V, Ev, Mv, Sv, Tv)):
            seg, es = v[a:b], e[a:b]
            ok = m & (seg > 0) & np.isfinite(es) & ((mk[a:b] & apogee_BAD) == 0)
            if ok.sum() < 60 or not np.all((seg[j - a + k_] > 0)):
                visits.append(None)
                continue
            x = np.arange(a, b) - j
            A = np.vstack([model, model * x / 128.0]).T
            c, *_ = np.linalg.lstsq(A[ok], seg[ok], rcond=None)
            fit = A @ c
            r = seg / fit - 1.0
            noise = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok])))
            amp = float(np.dot(r[j - a + k_], gk) / np.dot(gk, gk))
            snr_k = amp / (noise / np.sqrt(np.dot(gk, gk)))
            visits.append(dict(amp=round(amp, 4), snr=round(float(snr_k), 1),
                               sky=round(float(np.max(sky[j - 2:j + 3]) / max(np.median(seg[ok]), 1e-9)), 4),
                               tell=round(float(1.0 - np.min(tel[j - 2:j + 3])) if np.all(tel[j - 2:j + 3] > 0) else -1.0, 3),
                               lam_obs=round(apogee_LAM[j] * (1 + vrel[k] / science.C), 3) if k < len(vrel) and vrel[k] is not None else None,
                               flag=bool(np.any(mk[j - 2:j + 3] & apogee_BAD))))
        rec = dict(q, null=null[q["env"]], visits=visits)
        out.append(rec)
        amps = [v["amp"] for v in visits if v]
        print("%s %.3f A S/N %.1f (null %.1f) amp %.4f template %.3f%s" % (q["aid"], q["lam"], q["snr"], null[q["env"]],
              q["amp"], q["tmpl"] or 0, " persistence" if q["persist"] else ""))
        print("    visit amplitudes %s" % ["%.3f(%.0f)" % (v["amp"], v["snr"]) if v else "-" for v in visits])
        print("    sky/star %s  telluric depth %s" % ([v["sky"] if v else "-" for v in visits],
                                                      [v["tell"] if v else "-" for v in visits]))
        print("    observatory wavelengths %s" % [v["lam_obs"] if v else "-" for v in visits])
        if amps:
            print("    spread of visit amplitudes %.4f, median %.4f, %d of %d visits above half the combined amplitude"
                  % (np.std(amps), np.median(amps), sum(1 for x in amps if x > 0.5 * q["amp"]), len(amps)), flush=True)
    json.dump(out, open(os.path.join(apogee_HERE, "apogee_steady_vet.json"), "w"))


def apogee_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("pack", "search", "vet"))
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--thr", type=float, default=5.0)
    p.add_argument("--procs", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.mode == "pack":
        apogee_pack()
    elif a.mode == "vet":
        apogee_vet(a.top)
    else:
        apogee_search(a.k, a.thr, a.procs, a.limit or None)


def apogee_cli():
    apogee_main()


# --------------------------------------------------------------------------------------------------------------

desi_DOC = r"""Laser lines in DESI DR1 spectra of stars within 100 pc against twin fits, with dips as null and sky-prevalent pixels removed.

    python -I seti.py lasers desi sample
    python -I seti.py lasers desi fetch [--batch 400]
    python -I seti.py lasers desi search [--k 12] [--thr 5] [--procs 2]"""


desi_HERE = core.RESULTS
desi_DL = os.path.join(desi_HERE, "desi_dl")
desi_OUT = os.path.join(desi_HERE, "desi_combined")
desi_NPIX = 7781
desi_LAM = 3600.0 + 0.8 * np.arange(desi_NPIX)
# night-sky and street-light lines (vacuum A)
desi_SKY = (5578.89, 6302.05, 6365.54, 5891.58, 5897.56, 4047.71, 4359.56, 5462.27, 5771.21, 5792.27)
# lines active M dwarfs show or twins disagree on (vacuum A)
desi_STELLAR = (6564.61, 4862.68, 4341.68, 4102.89, 3971.20, 3890.15, 3934.78, 3969.59, 8500.36, 8544.44, 8664.52,
           5877.25, 6679.99, 7067.14, 4472.73, 5891.58, 5897.56, 8185.51, 8197.07, 7667.01, 7701.08, 8809.18, 7774.08,
           8448.68, 9548.59, 9231.55, 9017.39, 8865.22, 8752.88, 8667.40, 8600.75, 8547.73, 8469.58, 8439.08, 8415.63)


def desi_summary():
    """Peaks against the null by exposures per coadd, stellar-line peaks set aside."""
    nexp = {int(k): v[0] for k, v in json.load(open(os.path.join(desi_HERE, "desi_numexp.json"))).items()}
    rows = {int(r["targetid"]): r for r in csv.DictReader(open(os.path.join(desi_HERE, "desi_sample.csv")))}
    pk = json.load(open(os.path.join(desi_HERE, "desi_laser.json")))["peaks"]
    for q in pk:
        q["stellar"] = bool(any(abs(q["lam"] - s) < 3.0 for s in desi_STELLAR))
        q["nexp"] = nexp.get(q["targetid"], 0)
    for label, lo, hi in (("one exposure", 1, 1), ("two exposures", 2, 2), ("three or more", 3, 999)):
        nstar = sum(1 for t, n in nexp.items() if lo <= n <= hi)
        for env in ("continuum", "line"):
            sel = [q for q in pk if q["clean"] and not q["stellar"] and q["env"] == env and lo <= q["nexp"] <= hi]
            pos = sorted((q for q in sel if q["sign"] > 0), key=lambda q: -q["snr"])
            neg = sorted((q["snr"] for q in sel if q["sign"] < 0), reverse=True)
            top = neg[0] if neg else 0.0
            print("%-14s (%4d stars) %-9s emission-like S/N>8 %3d, >15 %3d; absorption-like S/N>8 %3d, >15 %3d; "
                  "strongest absorption %.1f" % (label, nstar, env, sum(q["snr"] > 8 for q in pos),
                                                 sum(q["snr"] > 15 for q in pos), sum(v > 8 for v in neg),
                                                 sum(v > 15 for v in neg), top))
            for q in pos[:5]:
                if q["snr"] > top:
                    r = rows[q["targetid"]]
                    print("    ABOVE-NULL %d (%d exp, Teff %s, %.0f pc) %.2f A S/N %.1f amp %.3f width %.2f"
                          % (q["targetid"], q["nexp"], r["teff"][:4], 1000.0 / float(r["parallax"]), q["lam"], q["snr"],
                             q["amp"], q["width"]))
    json.dump(dict(peaks=pk), open(os.path.join(desi_HERE, "desi_laser.json"), "w"))


def desi_tap(q):
    u = "https://datalab.noirlab.edu/tap/sync?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv",
                                                                             QUERY=q))
    raw = urllib.request.urlopen(urllib.request.Request(u, headers=core.UA), timeout=900).read()
    return list(csv.DictReader(io.StringIO(raw.decode())))


def desi_sample():
    # Postgres sorts NaN high; parallax needs an upper bound
    rows = desi_tap("SELECT m.targetid, m.source_id, m.target_ra AS ra, m.target_dec AS dec, g.parallax, g.phot_g_mean_mag, "
               "g.bp_rp, m.teff, m.logg, m.feh, m.rv_adop, m.sn_b, m.sn_r, m.sn_z, m.survey, m.program, m.healpix "
               "FROM desi_dr1.mws AS m JOIN gaia_dr3.gaia_source AS g "
               "ON m.source_id = g.source_id WHERE g.parallax BETWEEN 10 AND 800 AND m.zcat_primary = 't' "
               "AND m.rvs_warn = 0 AND m.sn_r > 20")
    with open(os.path.join(desi_HERE, "desi_sample.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("%d stars" % len(rows))


def desi_fetch(batch):
    from sparcl.client import SparclClient
    os.makedirs(desi_DL, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(desi_HERE, "desi_sample.csv"))))
    tids = [int(r["targetid"]) for r in rows]
    have = set()
    for p in glob.glob(os.path.join(desi_DL, "batch_*.npz")):
        have |= set(int(x) for x in np.load(p)["targetid"])
    todo = [t for t in tids if t not in have]
    print("%d stars, %d to fetch" % (len(tids), len(todo)), flush=True)
    c = SparclClient()
    k0 = len(glob.glob(os.path.join(desi_DL, "batch_*.npz")))
    for k in range(0, len(todo), batch):
        part = todo[k:k + batch]
        found = c.find(outfields=["sparcl_id", "targetid", "specprimary"],
                       constraints={"targetid": part, "data_release": ["DESI-DR1"]}, limit=batch * 4)
        ids = [r["sparcl_id"] for r in found.records if r.get("specprimary")]
        got = c.retrieve(uuid_list=ids, include=["targetid", "flux", "ivar", "mask", "model", "wave_sigma"],
                         dataset_list=["DESI-DR1"], limit=len(ids) + 10)
        recs = [r for r in got.records if len(r["flux"]) == desi_NPIX]
        np.savez_compressed(os.path.join(desi_DL, "batch_%04d.npz" % (k0 + k // batch)),
                            targetid=np.array([int(r["targetid"]) for r in recs], np.int64),
                            flux=np.array([r["flux"] for r in recs], np.float32),
                            ivar=np.array([r["ivar"] for r in recs], np.float32),
                            mask=np.array([r["mask"] for r in recs], np.int32),
                            model=np.array([r["model"] for r in recs], np.float32),
                            wave_sigma=np.array([r["wave_sigma"] for r in recs], np.float32))
        print("  %d/%d (%d spectra in this batch)" % (min(k + batch, len(todo)), len(todo), len(recs)), flush=True)


desi__S = {}


def desi__init():
    for name in ("norm", "good", "flux", "ivar", "sigma", "skyfrac"):
        desi__S[name] = np.load(os.path.join(desi_OUT, name + ".npy"), mmap_mode="r")
    desi__S["NB"] = np.load(os.path.join(desi_OUT, "neighbours.npy"))


def desi_one(args):
    i, thr, k = args
    quiet_warnings()
    N, G = desi__S["norm"], desi__S["good"]
    t = np.asarray(N[i], float)
    g = np.asarray(G[i], bool)
    nb = desi__S["NB"][i][:k]
    B = np.array([science.fill_gaps(np.asarray(N[j], float), np.asarray(G[j], bool)) for j in nb])
    model = np.full(desi_NPIX, np.nan)
    tmpl = np.full(desi_NPIX, np.nan)
    for s in range(0, desi_NPIX, 128):
        a, b = max(s - 64, 0), min(s + 192, desi_NPIX)
        m = g[a:b] & np.isfinite(t[a:b])
        if m.sum() < 80:
            continue
        mod, tp = science.twin_model(np.nan_to_num(t[a:b], nan=1.0), B[:, a:b], m, poly_order=2)
        c1 = min(s + 128, desi_NPIX)
        model[s:c1] = mod[s - a:c1 - a]
        tmpl[s:c1] = tp[s - a:c1 - a]
    res = np.where(g & np.isfinite(model) & (model > 0.05), t / model - 1.0, np.nan)
    flux = np.asarray(desi__S["flux"][i], float)
    ivar = np.asarray(desi__S["ivar"][i], float)
    ferr = np.where(ivar > 0, 1.0 / np.sqrt(np.maximum(ivar, 1e-30)) / np.maximum(np.abs(flux), 1e-9), np.inf)
    sig = np.maximum(ferr, science.block_mad(res, block=200, floor=1e-3))
    sp_all = np.asarray(desi__S["sigma"][i], float) / 0.8
    snr = np.zeros(desi_NPIX)
    amp = np.zeros(desi_NPIX)
    for s in range(0, desi_NPIX, 1000):
        sp = float(np.median(sp_all[s:s + 1000]))
        a, b = max(s - 20, 0), min(s + 1020, desi_NPIX)
        z1, a1 = science.matched_filter(res[a:b], sig[a:b], sp)
        snr[s:s + 1000] = z1[s - a:s - a + min(1000, desi_NPIX - s)]
        amp[s:s + 1000] = a1[s - a:s - a + min(1000, desi_NPIX - s)]
    near_bad = np.convolve((~g).astype(float), np.ones(5), mode="same") > 0
    out = []
    for sign in (1, -1):
        for j in science.find_peaks(sign * snr, thr, sep=5, mask=~near_bad & np.isfinite(snr), limit=300):
            sp = sp_all[j]
            out.append(dict(i=int(i), sign=sign, pix=int(j), lam=round(float(desi_LAM[j]), 2), snr=round(float(sign * snr[j]), 2),
                            amp=round(float(sign * amp[j]), 5), width=round(science.width_ratio(sign * res, j, sp), 2),
                            spiky=bool(science.profile_is_spiky(np.nan_to_num(sign * res), j, sp, base=0.0)),
                            tmpl=round(float(tmpl[j]), 3) if np.isfinite(tmpl[j]) else None,
                            sky=bool(any(abs(desi_LAM[j] - s_) < 2.5 for s_ in desi_SKY))))
    return dict(i=int(i), peaks=out, used=int(np.isfinite(res).sum()))


def desi_search(k, thr, procs):
    rows = {int(r["targetid"]): r for r in csv.DictReader(open(os.path.join(desi_HERE, "desi_sample.csv")))}
    T, F, IV, M, S = [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(desi_DL, "batch_*.npz"))):
        z = np.load(p)
        T.append(z["targetid"])
        F.append(z["flux"])
        IV.append(z["ivar"])
        M.append(z["mask"])
        S.append(z["wave_sigma"])
    tid = np.concatenate(T)
    F, IV, M, S = np.vstack(F), np.vstack(IV), np.vstack(M), np.vstack(S)
    good = (IV > 0) & (M == 0) & np.isfinite(F) & (F > 0)
    N = np.full(F.shape, np.nan, np.float32)
    for i in range(len(F)):
        f = np.where(good[i], F[i], np.nan)
        N[i] = f / science.envelope(f, block=100, q=90.0, smooth=3)
    good &= np.isfinite(N)
    os.makedirs(desi_OUT, exist_ok=True)
    for name, arr in (("norm", N), ("good", good), ("flux", F), ("ivar", IV), ("sigma", S)):
        np.save(os.path.join(desi_OUT, name + ".npy"), arr)
    np.save(os.path.join(desi_OUT, "skyfrac.npy"), np.zeros((1, 1), np.float32))
    nb = science.nearest_spectra(N, good, max(k, 20))
    np.save(os.path.join(desi_OUT, "neighbours.npy"), nb)
    print("%d spectra; neighbours found" % len(F), flush=True)
    del F, IV, M, S, N
    jobs = [(i, thr, k) for i in range(len(tid))]
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs, initializer=desi__init) as pool:
            res = pool.map(desi_one, jobs, chunksize=8)
    else:
        desi__init()
        res = [desi_one(j) for j in jobs]
    peaks = [dict(q, targetid=int(tid[q["i"]])) for r in res for q in r["peaks"]]
    # prevalence: spectra with a peak within 1 pixel against the median over 100
    for sign in (1, -1):
        sel = [q for q in peaks if q["sign"] == sign and not q["spiky"] and 0.6 <= q["width"] <= 1.6]
        hist = np.bincount([q["pix"] for q in sel], minlength=desi_NPIX).astype(float)
        near = np.convolve(hist, np.ones(3), mode="same")
        base = np.array([np.median(near[max(j - 100, 0):j + 101]) for j in range(desi_NPIX)])
        for q in sel:
            e = max(base[q["pix"]], 0.5)
            q["shared"] = round(float(near[q["pix"]] - 1), 1)
            q["shared_limit"] = round(float(e + 4 * np.sqrt(e) + 2), 1)
    for q in peaks:
        q.setdefault("shared", -1)
        q.setdefault("shared_limit", 0)
        q["clean"] = bool(not q["spiky"] and 0.6 <= q["width"] <= 1.6 and 0 <= q["shared"] <= q["shared_limit"]
                          and not q["sky"])
        q["env"] = "continuum" if (q["tmpl"] or 0) >= 0.95 else "line"
    json.dump(dict(peaks=peaks), open(os.path.join(desi_HERE, "desi_laser.json"), "w"))
    print("%d spectra searched, %d peaks" % (len(res), len(peaks)))
    for env in ("continuum", "line"):
        neg = sorted((q["snr"] for q in peaks if q["sign"] < 0 and q["clean"] and q["env"] == env), reverse=True)
        pos = sorted((q for q in peaks if q["sign"] > 0 and q["clean"] and q["env"] == env), key=lambda q: -q["snr"])
        top = neg[0] if neg else 0.0
        print("%s: clean absorption-like %d (strongest %s), clean emission-like %d"
              % (env, len(neg), ", ".join("%.1f" % v for v in neg[:5]), len(pos)))
        for t in (5, 6, 8, 10, 15):
            print("  S/N>%d: emission-like %d, absorption-like %d"
                  % (t, sum(1 for q in pos if q["snr"] > t), sum(1 for v in neg if v > t)))
        for q in pos[:15]:
            r = rows.get(q["targetid"], {})
            print("  %s%d (Gaia DR3 %s, %.0f pc, Teff %s) %.2f A S/N %.1f amp %.4f width %.2f template %.3f"
                  % ("ABOVE-NULL " if q["snr"] > top else "", q["targetid"], r.get("source_id"),
                     1000.0 / float(r.get("parallax", "nan") or "nan"), (r.get("teff") or "?")[:4], q["lam"], q["snr"],
                     q["amp"], q["width"], q["tmpl"] or 0))


def desi_exposures(row):
    """Every exposure of one target from its HEALPix file: per camera wavelength, flux, ivar, mask, MJD, fibre, petal."""
    from astropy.io import fits
    hp = int(row["healpix"])
    name = "spectra-%s-%s-%d.fits.gz" % (row["survey"], row["program"], hp)
    path = os.path.join(desi_HERE, "desi_vet", name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = ("https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/%s/%s/%d/%d/%s"
               % (row["survey"], row["program"], hp // 100, hp, name))
        with urllib.request.urlopen(urllib.request.Request(url, headers=core.UA), timeout=1800) as r, \
                open(path + ".tmp", "wb") as fh:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                fh.write(b)
        os.replace(path + ".tmp", path)
    out = {}
    with fits.open(path) as h:
        fm = h["FIBERMAP"].data
        rows = np.nonzero(fm["TARGETID"] == int(row["targetid"]))[0]
        out["mjd"] = np.asarray(fm["MJD"][rows], float) if "MJD" in fm.columns.names else np.full(len(rows), np.nan)
        out["fiber"] = np.asarray(fm["FIBER"][rows])
        out["expid"] = np.asarray(fm["EXPID"][rows]) if "EXPID" in fm.columns.names else np.arange(len(rows))
        for cam in ("B", "R", "Z"):
            out[cam] = (np.asarray(h[cam + "_WAVELENGTH"].data, float), np.asarray(h[cam + "_FLUX"].data[rows], float),
                        np.asarray(h[cam + "_IVAR"].data[rows], float), np.asarray(h[cam + "_MASK"].data[rows]))
    return out


def desi_vet(top, targets=None):
    """Per-exposure amplitude and observatory wavelength of the strongest clean peaks."""
    rows = {int(r["targetid"]): r for r in csv.DictReader(open(os.path.join(desi_HERE, "desi_sample.csv")))}
    peaks = json.load(open(os.path.join(desi_HERE, "desi_laser.json")))["peaks"]
    null = {env: max([q["snr"] for q in peaks if q["sign"] < 0 and q["clean"] and q["env"] == env] or [0.0])
            for env in ("continuum", "line")}
    if targets:
        want = [(int(t.split(":")[0]), float(t.split(":")[1])) for t in targets]
        cands = [min((q for q in peaks if q["targetid"] == t and q["sign"] > 0), key=lambda q, lam=lam: abs(q["lam"] - lam))
                 for t, lam in want]
    else:
        cands = sorted((q for q in peaks if q["sign"] > 0 and q["clean"]),
                       key=lambda q: -q["snr"] / max(null[q["env"]], 1))[:top]
    from scipy.ndimage import median_filter
    report = []
    for q in cands:
        r = rows[q["targetid"]]
        try:
            ex = desi_exposures(r)
        except Exception as e:
            print("%d: exposures unavailable (%s)" % (q["targetid"], str(e)[:80]))
            continue
        lam0 = q["lam"]
        amps = []
        for cam in ("B", "R", "Z"):
            w, F, IV, M = ex[cam]
            if not (w[0] + 5 < lam0 < w[-1] - 5):
                continue
            j = int(np.argmin(np.abs(w - lam0)))
            for k in range(len(F)):
                f = F[k]
                base = median_filter(f, size=31, mode="nearest")
                seg = (f[j - 2:j + 3] - base[j - 2:j + 3])
                err = 1.0 / np.sqrt(np.maximum(IV[k][j - 2:j + 3], 1e-30))
                bad = bool(np.any(M[k][j - 2:j + 3]))
                a = float(np.sum(seg / err ** 2) / np.sum(1 / err ** 2)) / max(float(np.median(base[j - 50:j + 50])), 1e-9)
                s = float(np.sum(seg / err ** 2) / np.sqrt(np.sum(1 / err ** 2)))
                mjd = ex["mjd"][k]
                bc = (float(science.barycentric_correction(float(r.get("ra") or 0), float(r.get("dec") or 0), mjd, "KPNO"))
                      if np.isfinite(mjd) and r.get("ra") else float("nan"))
                amps.append(dict(cam=cam, expid=int(ex["expid"][k]), mjd=round(float(mjd), 4), fiber=int(ex["fiber"][k]),
                                 amp=round(a, 4), snr=round(s, 1), masked=bad,
                                 lam_obs=round(lam0 / (1 + bc / 299792.458), 2) if np.isfinite(bc) else None))
        report.append(dict(q, null=null[q["env"]], exposures=amps))
        print("%d (Gaia DR3 %s, %.0f pc, Teff %s) %.2f A S/N %.1f (null %.1f) amp %.4f width %.2f"
              % (q["targetid"], r["source_id"], 1000.0 / float(r["parallax"]), r["teff"][:4], lam0, q["snr"],
                 null[q["env"]], q["amp"], q["width"]))
        for e in amps:
            print("    %s exp %d MJD %.3f fibre %d: amp %+.4f S/N %+.1f%s, observatory %s A"
                  % (e["cam"], e["expid"], e["mjd"], e["fiber"], e["amp"], e["snr"], " masked" if e["masked"] else "",
                     e["lam_obs"]), flush=True)
    json.dump(report, open(os.path.join(desi_HERE, "desi_laser_vet.json"), "w"))


def desi_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("sample", "fetch", "search", "summary", "vet"))
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--targets", nargs="*", help="TARGETID:WAVELENGTH pairs to vet")
    p.add_argument("--batch", type=int, default=400)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--thr", type=float, default=5.0)
    p.add_argument("--procs", type=int, default=2)
    a = p.parse_args()
    if a.mode == "sample":
        desi_sample()
    elif a.mode == "fetch":
        desi_fetch(a.batch)
    elif a.mode == "vet":
        desi_vet(a.top, a.targets)
    elif a.mode == "summary":
        desi_summary()
    else:
        desi_search(a.k, a.thr, a.procs)


def desi_cli():
    quiet_warnings()
    desi_main()


# --------------------------------------------------------------------------------------------------------------

lamost_DOC = r"""Laser lines in LAMOST medium-resolution exposures of stars within 100 pc: 6-sigma unresolved peaks persisting across exposures, vetted for sharing, fibres and frame.

    python -I seti.py lasers lamost targets
    python -I seti.py lasers lamost fetch [--threads 4]
    python -I seti.py lasers lamost search
    python -I seti.py lasers lamost judge
    python -I seti.py lasers lamost neighbours
    python -I seti.py lasers lamost record"""

lamost_ROOT = os.environ.get("LAMOST_ROOT", core.data("lamost_mrs"))
lamost_HERE = core.RESULTS
lamost_TAP = "https://www.lamost.org/dr10/v2.0/voservice/tap/sync"
lamost_FITS = "https://www.lamost.org/dr10/v2.0/medspectrum/fits/%s"
lamost_C_KMS = 299792.458
lamost_R = 7500.0
lamost_Z_MIN = 6.0
lamost_WIN_A = 3.0


def lamost_tap(q):
    """Rows of ADQL on LAMOST's TAP service by POST."""
    import time
    import urllib.request
    from astropy.io.votable import parse_single_table
    data = urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", QUERY=q)).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(lamost_TAP, data=data, headers=dict(core.UA, **{"Content-Type": "application/x-www-form-urlencoded"}))
            b = urllib.request.urlopen(req, timeout=300).read()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    t = parse_single_table(io.BytesIO(b)).to_table()
    return [dict(zip(t.colnames, [x.item() if hasattr(x, "item") else x for x in row])) for row in t]


def lamost_targets():
    os.makedirs(lamost_ROOT, exist_ok=True)
    # Gaia returns at most 50000 rows; RA slices
    gaia = []
    for ra0 in range(0, 360, 30):
        gaia += core.gaia("SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp, radial_velocity FROM "
                              "gaiadr3.gaia_source WHERE parallax > 10 AND parallax_over_error > 10 AND dec > -15 AND "
                              "phot_g_mean_mag < 15 AND ra >= %d AND ra < %d" % (ra0, ra0 + 30))
    by = {r["source_id"]: r for r in gaia}
    ids = list(by)
    rows = []
    for k in range(0, len(ids), 400):
        q = ("SELECT obsid, gaia_source_id, band, coadd, snr, lmjm, lmjd, rv_br1, rv_br_flag FROM public.med_catalogue "
             "WHERE gaia_source_id IN (%s)" % ",".join("'%s'" % i for i in ids[k:k + 400]))
        try:
            rows += lamost_tap(q)
        except Exception as e:
            print("chunk %d: %s" % (k, str(e)[:80]), flush=True)
        if k % 8000 == 0:
            print("%d of %d Gaia stars queried, %d MRS rows" % (k, len(ids), len(rows)), flush=True)
    stars = {}
    for r in rows:
        s = str(r["gaia_source_id"])
        g = by.get(s, {})
        x = stars.setdefault(s, dict(gaia=s, plx=g.get("parallax"), g=g.get("phot_g_mean_mag"), bp_rp=g.get("bp_rp"),
                                     rv_gaia=g.get("radial_velocity"), obsids=set(), exposures=0, rv=[]))
        x["obsids"].add(str(r["obsid"]))
        if int(r["coadd"] or 0) == 0:
            x["exposures"] += 1
        try:
            if float(r["rv_br1"]) > -9000:
                x["rv"].append(float(r["rv_br1"]))
        except (TypeError, ValueError):
            pass
    for x in stars.values():
        x["obsids"] = sorted(x["obsids"])
        x["rv"] = float(np.median(x["rv"])) if x["rv"] else None
    json.dump(stars, open(os.path.join(lamost_ROOT, "targets.json"), "w"))
    print("%d Gaia stars within 100 pc (dec > -15, G < 15); %d with MRS spectra, %d observations, %d single-exposure rows"
          % (len(ids), len(stars), len({o for x in stars.values() for o in x["obsids"]}),
             sum(x["exposures"] for x in stars.values())))


def lamost_fetch(threads):
    from concurrent.futures import ThreadPoolExecutor
    stars = json.load(open(os.path.join(lamost_ROOT, "targets.json")))
    obs = sorted({o for x in stars.values() for o in x["obsids"]})
    os.makedirs(os.path.join(lamost_ROOT, "fits"), exist_ok=True)

    def one(o):
        p = os.path.join(lamost_ROOT, "fits", o + ".fits.gz")
        if os.path.exists(p):
            return 1
        b = core.get(lamost_FITS % o, timeout=120, retries=3, soft=True, pause=2)
        if not b or b[:2] != b"\x1f\x8b":
            return 0
        open(p + ".tmp", "wb").write(b)
        os.replace(p + ".tmp", p)
        return 1
    n = 0
    with ThreadPoolExecutor(threads) as ex:
        for k, r in enumerate(ex.map(one, obs)):
            n += r
            if k % 2000 == 0:
                print("%d of %d observations fetched" % (n, len(obs)), flush=True)
    print("%d of %d observation files on disk" % (n, len(obs)))


def lamost_exposures(path):
    """Single exposures of one observation: (arm, lmjm, wavelength, flux, sigma, mask, helio_rv, fibre)."""
    from astropy.io import fits
    out = []
    with fits.open(path) as h:
        hrv = h[0].header.get("HELIO_RV")
        fibre = "%s-%s" % (h[0].header.get("SPID"), h[0].header.get("FIBERID"))
        for e in h[1:]:
            name = str(e.name)
            if name.startswith("COADD") or "-" not in name:
                continue
            d = e.data[0]
            cols = e.columns.names
            w = np.asarray(d["WAVELENGTH"], float)
            f = np.asarray(d["FLUX"], float)
            iv = np.asarray(d["IVAR"], float)
            m = np.asarray(d["PIXMASK"] if "PIXMASK" in cols else d["ANDMASK"] if "ANDMASK" in cols else np.zeros(len(w)),
                           int)
            out.append((name[0], name[2:], w, f, np.where(iv > 0, 1 / np.sqrt(np.maximum(iv, 1e-30)), np.inf), m, hrv,
                        fibre))
    return out


def lamost_peaks_of(w, f, s, m):
    from scipy.ndimage import maximum_filter1d, percentile_filter
    from scipy.optimize import curve_fit
    ok = np.isfinite(f) & np.isfinite(s) & (s < np.inf) & (m == 0)
    if ok.sum() < 500:
        return []
    w, f, s = w[ok], f[ok], s[ok]
    step = float(np.median(np.diff(w)))
    win = max(int(lamost_WIN_A / step) | 1, 15)
    env = percentile_filter(f, 90, size=win, mode="nearest")
    z = (f - env) / s
    half = max(int(lamost_WIN_A / step), 10)
    left = maximum_filter1d(np.concatenate([np.full(half, -np.inf), f[:-half]]), size=half, origin=(half - 1) // 2)
    right = maximum_filter1d(np.concatenate([f[half:], np.full(half, -np.inf)]), size=half, origin=-(half // 2))
    top = np.maximum(left, right)
    out = []
    for i in np.where((z >= lamost_Z_MIN) & (z >= np.roll(z, 1)) & (z >= np.roll(z, -1)))[0]:
        if i < 6 or i > len(w) - 7 or not f[i] - top[i] > 3 * s[i]:
            continue
        inst = w[i] / lamost_R
        x, y = w[i - 6:i + 7], (f - env)[i - 6:i + 7]
        try:
            p, _ = curve_fit(lambda x, a, mu, sg: a * np.exp(-0.5 * ((x - mu) / sg) ** 2), x, y,
                             p0=[y[6], w[i], inst / 2.355], sigma=s[i - 6:i + 7], maxfev=2000)
        except Exception:
            continue
        fw = 2.355 * abs(p[2])
        if 0.5 * inst <= fw <= 1.5 * inst and abs(p[1] - w[i]) < inst:
            out.append(dict(lam=float(p[1]), z=float(z[i]), fwhm=float(fw), inst=float(inst)))
    return out


def lamost_scan_obs(job):
    quiet_warnings()
    obsid, star, rv = job
    p = os.path.join(lamost_ROOT, "fits", obsid + ".fits.gz")
    if not os.path.exists(p):
        return []
    try:
        exps = lamost_exposures(p)
    except Exception:
        return []
    out = []
    for arm, lmjm, w, f, s, m, hrv, fibre in exps:
        lo, hi = float(np.nanmin(w)), float(np.nanmax(w))
        for pk in lamost_peaks_of(w, f, s, m):
            out.append(dict(pk, star=star, obsid=obsid, arm=arm, lmjm=lmjm, hrv=hrv, fibre=fibre,
                            edge=min(pk["lam"] - lo, hi - pk["lam"]), topo=pk["lam"] / (1 + (hrv or 0.0) / lamost_C_KMS),
                            rest=pk["lam"] / (1 + (rv or 0.0) / lamost_C_KMS)))
    return out


def lamost_search():
    from concurrent.futures import ProcessPoolExecutor
    stars = json.load(open(os.path.join(lamost_ROOT, "targets.json")))
    jobs = [(o, s, x["rv"] if x["rv"] is not None else (float(x["rv_gaia"]) if x["rv_gaia"] not in (None, "") else 0.0))
            for s, x in stars.items() for o in x["obsids"]]
    with ProcessPoolExecutor(int(os.environ.get("LAMOST_PROCS", "4"))) as ex:
        feats = [f for fs in ex.map(lamost_scan_obs, jobs, chunksize=16) for f in fs]
    print("%d observations of %d stars; %d narrow peaks at %.0f sigma in single exposures"
          % (len(jobs), len(stars), len(feats), lamost_Z_MIN), flush=True)
    json.dump(dict(observations=len(jobs), stars=len(stars), peaks=feats), open(os.path.join(lamost_HERE, "lamost_peaks.json"), "w"))
    lamost_judge()


def lamost_poisson_sf(n, mu):
    """Poisson P(X >= n) for mean mu."""
    if n <= 0:
        return 1.0
    term, cdf = math.exp(-mu), 0.0
    for k in range(n):
        cdf += term
        term *= mu / (k + 1)
    return max(0.0, 1.0 - cdf)


def lamost_judge():
    """Persistent features tested for sharing with other stars in rest and observed frames beyond Poisson chance at p 1e-3."""
    d = json.load(open(os.path.join(lamost_HERE, "lamost_peaks.json")))
    feats = d["peaks"]
    by = {}
    for f in feats:
        by.setdefault((f["star"], f["arm"]), []).append(f)
    pers = []
    for (star, arm), fs in by.items():
        fs.sort(key=lambda f: f["rest"])
        used = set()
        for k, f in enumerate(fs):
            if k in used:
                continue
            grp = [j for j, g in enumerate(fs) if abs(g["rest"] - f["rest"]) < f["inst"] / 3]
            exps = {(fs[j]["obsid"], fs[j]["lmjm"]) for j in grp}
            if len(exps) >= 2:
                used.update(grp)
                pers.append(dict(star=star, arm=arm, rest=float(np.mean([fs[j]["rest"] for j in grp])),
                                 obs=float(np.mean([fs[j]["lam"] for j in grp])), n=len(exps),
                                 nights=len({fs[j]["obsid"] for j in grp}), z=max(fs[j]["z"] for j in grp),
                                 inst=f["inst"]))
    print("%d persistent features in %d stars" % (len(pers), len({p["star"] for p in pers})), flush=True)
    for p in pers:
        grp = [g for g in by[(p["star"], p["arm"])] if abs(g["rest"] - p["rest"]) < p["inst"] / 3]
        p["topo"] = float(np.mean([g["topo"] for g in grp]))
        p["edge"] = float(min(g["edge"] for g in grp))
        p["fibres"] = sorted({g["fibre"] for g in grp})
        p["obsids"] = sorted({g["obsid"] for g in grp})
    # sharing against the density 1-20 A either side
    for frame, key, kms in (("rest", "rest", 15.0), ("topo", "topo", 5.0)):
        w = np.array([f[key] for f in feats])
        st = np.array([f["star"] for f in feats])
        o = np.argsort(w)
        w, st = w[o], st[o]
        for p in pers:
            centre = p[key]
            tol = centre * kms / lamost_C_KMS
            lo, hi = np.searchsorted(w, [centre - tol, centre + tol])
            n = len(set(st[lo:hi]) - {p["star"]})
            side = set(st[np.searchsorted(w, centre - 20.0):np.searchsorted(w, centre - 1.0)]) | \
                set(st[np.searchsorted(w, centre + 1.0):np.searchsorted(w, centre + 20.0)])
            mu = len(side - {p["star"]}) * (2 * tol / 38.0)
            p["n_" + frame], p["mu_" + frame], p["p_" + frame] = n, mu, lamost_poisson_sf(n, mu)
    # same or neighbouring fibres, and the star on other nights, in the instrument frame
    by_spec = {}
    for f in feats:
        by_spec.setdefault((f["fibre"].split("-")[0], f["arm"]), []).append(f)
    stars = json.load(open(os.path.join(lamost_ROOT, "targets.json")))
    for p in pers:
        mine = {(fb.split("-")[0], int(fb.split("-")[1])) for fb in p["fibres"] if fb.split("-")[1].isdigit()}
        same, near = set(), set()
        for spec, fno in mine:
            for f in by_spec.get((spec, p["arm"]), []):
                if f["star"] == p["star"] or not f["fibre"].split("-")[1].isdigit():
                    continue
                df = abs(int(f["fibre"].split("-")[1]) - fno)
                if df == 0 and abs(f["topo"] - p["topo"]) < 0.5:
                    same.add(f["star"])
                elif df <= 6 and abs(f["topo"] - p["topo"]) < 8.0:
                    near.add(f["star"])
        p["fibre_mates"], p["detector_mates"] = sorted(same), sorted(near - same)
        own = [f for f in by[(p["star"], p["arm"])] if f["obsid"] not in p["obsids"]
               and abs(f["topo"] - p["topo"]) < p["inst"] / 6 and abs(f["rest"] - p["rest"]) >= p["inst"] / 3]
        p["other_nights_topo"] = sorted({f["obsid"] for f in own})
        # members on nights with distinguishable heliocentric corrections
        grp = [g for g in by[(p["star"], p["arm"])] if abs(g["rest"] - p["rest"]) < p["inst"] / 3]
        hrvs = [g["hrv"] or 0.0 for g in grp]
        p["hrv_spread"] = float(max(hrvs) - min(hrvs))
        p["spread_star"] = float(np.ptp([g["rest"] for g in grp]))
        p["spread_topo"] = float(np.ptp([g["topo"] for g in grp]))
        # the catalogue counts both arms' exposures
        p["duty"] = float(p["n"] / max(int(stars.get(p["star"], {}).get("exposures") or 2 * p["n"]) / 2.0, p["n"]))
        # other stars in the same exposures within 1 A (sky, satellites, meteors)
        lmjms = {g["lmjm"] for g in grp}
        p["exposure_mates"] = sorted({f["star"] for f in feats if f["lmjm"] in lmjms and f["star"] != p["star"]
                                      and f["arm"] == p["arm"] and abs(f["topo"] - p["topo"]) < 1.0})
    for p in pers:
        why = []
        if p["edge"] < 10.0:
            why.append("within %.1f A of the arm's end" % p["edge"])
        if p["p_rest"] < 1e-3:
            why.append("shared in the stellar frame (%d other stars against %.1f by chance)" % (p["n_rest"], p["mu_rest"]))
        if p["p_topo"] < 1e-3:
            why.append("shared in the instrument's frame (%d other stars against %.1f by chance)"
                       % (p["n_topo"], p["mu_topo"]))
        if p["fibre_mates"]:
            why.append("the same fibre shows it in %d other stars" % len(p["fibre_mates"]))
        if p["detector_mates"]:
            why.append("fibres within 6 on the same spectrograph show it in %d other stars" % len(p["detector_mates"]))
        if p["other_nights_topo"]:
            why.append("fixed in the instrument's frame across %d other nights" % len(p["other_nights_topo"]))
        if p["exposure_mates"]:
            why.append("other stars in the same exposures show it (%d)" % len(p["exposure_mates"]))
        if p["hrv_spread"] >= 8.0 and p["spread_topo"] < 0.5 * p["spread_star"]:
            why.append("its nights agree in the instrument's frame (%.3f A) and not in the star's (%.3f A)"
                       % (p["spread_topo"], p["spread_star"]))
        p["why"] = why
    left = sorted([p for p in pers if not p["why"]], key=lambda p: -p["z"])
    # chance rate from same-star pair separations of 3-30 A scaled to the window
    near_pairs, far_pairs = 0, 0
    for fs in by.values():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                if (fs[i]["obsid"], fs[i]["lmjm"]) != (fs[j]["obsid"], fs[j]["lmjm"]):
                    s = abs(fs[i]["rest"] - fs[j]["rest"]) / (fs[i]["inst"] / 3)
                    near_pairs += s < 1
                    far_pairs += 10 <= s < 100
    chance = far_pairs / 90.0
    print("two-exposure coincidences within a third of a resolution element: %d, of which chance makes %.1f"
          % (near_pairs, chance))
    counts = {}
    for p in pers:
        for w_ in p["why"]:
            k = ("edge" if w_.startswith("within") else "stellar" if "stellar" in w_ else "instrument frame"
                 if "instrument's frame (" in w_ else "fibre" if "fibre" in w_ else "other nights")
            counts[k] = counts.get(k, 0) + 1
    print("explained: %s; %d left:" % (", ".join("%s %d" % kv for kv in sorted(counts.items())), len(left)))
    for p in left[:60]:
        print("  Gaia DR3 %s %s %.3f A (instrument %.3f), %d exposures on %d nights, peak %.1f sigma; other stars %d "
              "(%.1f expected) in the stellar frame, %d (%.1f) in the instrument's; fibres %s"
              % (p["star"], p["arm"], p["rest"], p["topo"], p["n"], p["nights"], p["z"], p["n_rest"], p["mu_rest"],
                 p["n_topo"], p["mu_topo"], ",".join(p["fibres"])))
    json.dump(dict(observations=d["observations"], stars=d["stars"], peaks=len(feats), persistent=pers,
                   near_pairs=int(near_pairs), chance_pairs=float(chance)),
              open(os.path.join(lamost_HERE, "lamost_laser.json"), "w"), indent=1)


def lamost_excess(w, f, s, m, centre, half=0.5):
    """Largest excess over the 3 A running 90th percentile within half of centre."""
    from scipy.ndimage import percentile_filter
    ok = np.isfinite(f) & np.isfinite(s) & (s < np.inf) & (m == 0)
    if ok.sum() < 500:
        return float("nan")
    w, f, s = w[ok], f[ok], s[ok]
    step = float(np.median(np.diff(w)))
    env = percentile_filter(f, 90, size=max(int(lamost_WIN_A / step) | 1, 15), mode="nearest")
    sel = np.abs(w - centre) < half
    return float(np.max((f[sel] - env[sel]) / s[sel])) if sel.any() else float("nan")


def lamost_neighbours():
    """Judged features against fibres within 8 on the same spectrograph and exposures: shared light is cross-talk or sky."""
    r = json.load(open(os.path.join(lamost_HERE, "lamost_laser.json")))
    peaks = json.load(open(os.path.join(lamost_HERE, "lamost_peaks.json")))["peaks"]
    os.makedirs(os.path.join(lamost_ROOT, "neighbours"), exist_ok=True)
    for p in r["persistent"]:
        if p["why"]:
            continue
        grp = [g for g in peaks if g["star"] == p["star"] and g["arm"] == p["arm"] and abs(g["rest"] - p["rest"]) < p["inst"] / 3]
        p["neighbours"] = []
        for obsid in sorted({g["obsid"] for g in grp}):
            meta = lamost_tap("SELECT planid, spid, fiberid, lmjd FROM public.med_catalogue WHERE obsid = %s" % obsid)
            if not meta:
                continue
            m = meta[0]
            rows = lamost_tap("SELECT obsid, fiberid FROM public.med_catalogue WHERE planid = '%s' AND lmjd = %s AND spid = %s "
                       "AND fiberid BETWEEN %d AND %d" % (m["planid"], m["lmjd"], m["spid"], int(m["fiberid"]) - 8,
                                                          int(m["fiberid"]) + 8))
            nb = [dict(obsid=o) for o in sorted({str(x["obsid"]) for x in rows} - {str(obsid)})]
            lmjms = {g["lmjm"] for g in grp if g["obsid"] == obsid}
            for n in nb:
                path = os.path.join(lamost_ROOT, "neighbours", "%s.fits.gz" % n["obsid"])
                if not os.path.exists(path):
                    b = core.get(lamost_FITS % n["obsid"], timeout=120, retries=3, soft=True, pause=2)
                    if not b or b[:2] != b"\x1f\x8b":
                        continue
                    open(path, "wb").write(b)
                try:
                    exps = lamost_exposures(path)
                except Exception:
                    continue
                for arm, lmjm, w, f, s, mk, hrv, fibre in exps:
                    if arm == p["arm"] and lmjm in lmjms:
                        z = lamost_excess(w / (1 + (hrv or 0.0) / lamost_C_KMS), f, s, mk, p["topo"])
                        p["neighbours"].append(dict(obsid=n["obsid"], fibre=fibre, lmjm=lmjm, z=z))
        zs = [x["z"] for x in p["neighbours"] if np.isfinite(x["z"])]
        print("Gaia DR3 %s %s %.3f A (instrument %.3f): %d neighbouring-fibre exposures, largest excess there %.1f sigma "
              "(the star's own peak %.1f)" % (p["star"], p["arm"], p["rest"], p["topo"], len(zs), max(zs) if zs else
                                               float("nan"), p["z"]), flush=True)
        if zs and max(zs) >= 5.0:
            p["why"].append("fibres beside it on the same plate show it in the same exposures (%.1f sigma)" % max(zs))
    json.dump(r, open(os.path.join(lamost_HERE, "lamost_laser.json"), "w"), indent=1)


def lamost_exposure_check():
    """Judged features against every exposure: split peaks, cosmic rays, bad pixels, and single-night instrument features."""
    from scipy.ndimage import percentile_filter
    r = json.load(open(os.path.join(lamost_HERE, "lamost_laser.json")))
    peaks = json.load(open(os.path.join(lamost_HERE, "lamost_peaks.json")))["peaks"]
    stars = json.load(open(os.path.join(lamost_ROOT, "targets.json")))
    star_obs = {}
    for pk in peaks:
        star_obs.setdefault(pk["star"], set()).add(pk["obsid"])
    for sid, info in stars.items():
        if isinstance(info, dict):
            for o in info.get("obsids", []):
                star_obs.setdefault(sid, set()).add(str(o))
    cache = {}

    def exps_of(obsid):
        if obsid not in cache:
            p = os.path.join(lamost_ROOT, "fits", obsid + ".fits.gz")
            try:
                cache[obsid] = lamost_exposures(p) if os.path.exists(p) else []
            except Exception:
                cache[obsid] = []
        return cache[obsid]

    def read(w, f, s, m, centre, half):
        """Strongest unmasked pixel near centre: wavelength, excess, nearby masks, sigma ratio, lowest flux nearby."""
        fin = np.isfinite(f) & np.isfinite(s) & (s < np.inf)
        ok = fin & (m == 0)
        if ok.sum() < 500:
            return None
        wc, fc, sc = w[ok], f[ok], s[ok]
        step = float(np.median(np.diff(wc)))
        env = percentile_filter(fc, 90, size=max(int(lamost_WIN_A / step) | 1, 15), mode="nearest")
        sel = np.abs(wc - centre) < half
        if not sel.any():
            return None
        z = (fc - env) / sc
        i = int(np.where(sel)[0][np.argmax(z[sel])])
        med_s = float(np.median(sc[max(i - 40, 0):i + 40]))
        j = int(np.argmin(np.abs(w - wc[i])))                      # the same pixel in the raw array
        lo, hi = max(j - 5, 0), min(j + 6, len(w))
        raw = np.where(fin[lo:hi], f[lo:hi], 0.0) / max(float(env[i]), 1e-9)
        # excess width against the instrumental FWHM
        ex = fc - env
        half_ex, a, b = 0.5 * ex[i], i, i
        while a > 0 and ex[a - 1] > half_ex:
            a -= 1
        while b < len(ex) - 1 and ex[b + 1] > half_ex:
            b += 1
        return dict(lam=float(wc[i]), z=float(z[i]), masked=bool((m[max(j - 3, 0):j + 4] != 0).any()),
                    sig_ratio=float(sc[i] / med_s) if med_s > 0 else 1.0, floor=float(np.min(raw)),
                    width=b - a + 1, fwhm_px=float(wc[i] / lamost_R / step))

    counts = {}
    for p in r["persistent"]:
        if p["why"]:
            continue
        found = []
        for obsid in p["obsids"]:
            for a, lmjm, w, f, s, m, hrv, fibre in exps_of(obsid):
                if a == p["arm"]:
                    q = read(w, f, s, m, p["obs"], 1.2)
                    if q and q["z"] >= 4:
                        found.append(q)
        centre = float(np.median([q["lam"] for q in found])) if found else p["obs"]
        spread = float(max(q["lam"] for q in found) - min(q["lam"] for q in found)) if len(found) > 1 else 0.0
        nights = []
        for obsid in sorted(star_obs.get(p["star"], set()) | set(p["obsids"])):
            zs, hits = [], []
            for a, lmjm, w, f, s, m, hrv, fibre in sorted((e for e in exps_of(obsid) if e[0] == p["arm"]), key=lambda e: e[1]):
                q = read(w, f, s, m, centre, 0.4)
                zs.append(q["z"] if q else float("nan"))
                if q and q["z"] >= 4:
                    hits.append(q)
            good = [z for z in zs if np.isfinite(z)]
            nights.append(dict(obsid=obsid, n=len(good), hit=len(hits), zmax=max(good) if good else float("nan"),
                               masked=any(q["masked"] for q in hits), sig_ratio=max([q["sig_ratio"] for q in hits] or [1.0]),
                               floor=min([q["floor"] for q in hits] or [1.0])))
        p["peak_pixel"], p["peak_spread"], p["nights_check"] = centre, spread, nights
        det = [n for n in nights if n["hit"] > 0]
        others = [n for n in nights if n["hit"] == 0 and n["n"] >= 2]
        why = []
        if spread > p["inst"] / 2:
            why.append("its exposures' peaks lie %.2f A apart, more than half a resolution element: not one feature" % spread)
        single = [n for n in det if n["n"] >= 3 and n["hit"] <= max(1, n["n"] // 3)]
        if single and not any(n["hit"] >= 2 for n in det):
            why.append("in %s of its nights' %s exposures with the others flat: cosmic rays"
                       % ("+".join(str(n["hit"]) for n in det), "+".join(str(n["n"]) for n in det)))
        if any(n["masked"] or n["sig_ratio"] >= 3 or n["floor"] < 0.3 for n in det):
            why.append("masked, variance-inflated or dead pixels at the peak")
        if found and all(q["width"] < 0.5 * q["fwhm_px"] for q in found):
            why.append("narrower than the instrument (%d pixels above half the excess against an instrumental FWHM of %.1f): detector pixels, not light"
                       % (max(q["width"] for q in found), found[0]["fwhm_px"]))
        if len(det) == 1 and det[0]["hit"] >= max(2, det[0]["n"] - 1) and others and all(not (n["zmax"] >= 3) for n in others):
            why.append("in every exposure of one night and in none of the star's %d other nights" % len(others))
        p["why"].extend(why)
        for k in (why or ["left"]):
            key = k.split(":")[0].split(" with ")[0][:48]
            counts[key] = counts.get(key, 0) + 1
        print("Gaia DR3 %s %s %.3f A: peak pixel %.3f (spread %.2f); nights %s; %s"
              % (p["star"], p["arm"], p["rest"], centre, spread,
                 ", ".join("%s %d/%d%s" % (n["obsid"], n["hit"], n["n"], "*" if n["obsid"] in p["obsids"] else "") for n in nights),
                 "; ".join(why) or "left"), flush=True)
    print("exposure check: " + "; ".join("%s %d" % kv for kv in sorted(counts.items())))
    json.dump(r, open(os.path.join(lamost_HERE, "lamost_laser.json"), "w"), indent=1)


def lamost_record():
    """Record one row per star with its strongest persistent feature and explanation."""
    r = json.load(open(os.path.join(lamost_HERE, "lamost_laser.json")))
    stars = json.load(open(os.path.join(lamost_ROOT, "targets.json")))
    best = {}
    for p in r["persistent"]:
        if p["star"] not in best or p["z"] > best[p["star"]]["z"]:
            best[p["star"]] = p
    left = [p for p in r["persistent"] if not p["why"]]
    core.record_search(
        "laser_lamost_mrs", "Laser lines, LAMOST medium resolution",
        archive="LAMOST medium-resolution survey single exposures", release="DR10 v2.0",
        selection="Gaia DR3 stars with parallax above 10 mas, dec above -15 deg and G below 15 in the MRS catalogue",
        code="seti.py lasers lamost", params=dict(z_min=lamost_Z_MIN, window_a=lamost_WIN_A, persistence="two exposures within a third of "
                                            "a resolution element", stellar_kms=15, instrument_kms=5, edge_a=10,
                                            fibre_a=0.5, p_share=1e-3),
        n_targets=len(stars),
        null_summary="%d observations, %d narrow single-exposure peaks at %.0f sigma, %d persistent features in %d "
                     "stars, %d left after the edge, sharing, fibre and instrument-frame filters"
                     % (r["observations"], r["peaks"], lamost_Z_MIN, len(r["persistent"]),
                        len({p["star"] for p in r["persistent"]}), len(left)),
        sensitivity="an unresolved line 6 sigma above the local continuum in two 20-minute exposures, R 7500, "
                    "495-535 and 630-680 nm",
        result="%d unexplained persistent features against %.0f two-exposure coincidences that "
                                    "chance makes" % (len(left), r.get("chance_pairs", float("nan"))),
        ledger="Laser lines, LAMOST medium resolution")
    core.clear_targets("laser_lamost_mrs")
    rows = []
    for s in stars:
        p = best.get(s)
        rows.append(dict(target="gaia_dr3:%s" % s, stat=float(p["z"]) if p else None, unit="sigma" if p else None,
                         vet=("explained" if p["why"] else "unexplained") if p else None,
                         note=("%s %.3f A: %s" % (p["arm"], p["rest"], "; ".join(p["why"]) or "left")) if p else None))
    core.record_targets("laser_lamost_mrs", rows)
    print("recorded")


def lamost_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("targets", "fetch", "search", "judge", "neighbours", "exposures", "record"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.mode == "targets":
        lamost_targets()
    elif a.mode == "fetch":
        lamost_fetch(a.threads)
    elif a.mode == "judge":
        lamost_judge()
    elif a.mode == "neighbours":
        lamost_neighbours()
    elif a.mode == "exposures":
        lamost_exposure_check()
    elif a.mode == "record":
        lamost_record()
    else:
        lamost_search()


def lamost_cli():
    quiet_warnings()
    lamost_main()


# --------------------------------------------------------------------------------------------------------------

carmenes_DOC = r"""Laser lines in CARMENES DR1 M-dwarf spectra: persistent unresolved peak groups across epochs, vetted by sharing, forced photometry in other stars, and twin fits.

    python -I seti.py lasers carmenes stars
    python -I seti.py lasers carmenes stream --list SHARE.json --out carmenes_<part>.jsonl [--k 8] [--procs 2]
    python -I seti.py lasers carmenes vet carmenes_*.jsonl
    python -I seti.py lasers carmenes forced carmenes_*.jsonl [--procs 4]     (the cached spectra under CARMENES_ROOT)
    python -I seti.py lasers carmenes refit [--procs 4]                        (the last stage again on carmenes_forced.json)
    python -I seti.py lasers carmenes record carmenes_*.jsonl"""

carmenes_ROOT = os.environ.get("CARMENES_ROOT", core.data("carmenes"))
carmenes_HERE = core.RESULTS
carmenes_PAGE = "https://carmenes.cab.inta-csic.es/gto/jsp/dr1Public.jsp"
carmenes_ZIP = "https://carmenes.cab.inta-csic.es/gto/getDR1DataPublic.action?id=%s_VIS.zip"
carmenes_UA = dict(core.UA, **{"Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
carmenes_C = 299792.458
carmenes_R = 94600.0
carmenes_SIG_PX = carmenes_C / carmenes_R / 2.3548 / 1.26  # instrumental sigma in pixels (1.26 km/s per pixel)
carmenes_EDGE = 100  # pixels dropped at each order end
carmenes_THR = 6.0
carmenes_WIN = 41
carmenes_PCT = 80


def carmenes_stars():
    """DR1 targets: Karmn identifier and name."""
    os.makedirs(carmenes_ROOT, exist_ok=True)
    t = urllib.request.urlopen(urllib.request.Request(carmenes_PAGE, headers=carmenes_UA), timeout=120).read().decode("utf-8", "replace")
    ids = re.findall(r"id=\s*(J[0-9]{5}[+-][0-9]{3}[A-Z]?)_VIS\.zip", t.replace("%2B", "+"))
    names = {}
    for k, nm in re.findall(r"<td[^>]*>\s*(J[0-9]{5}[+-][0-9]{3}[A-Z]?)\s*</td>\s*<td[^>]*>\s*([^<]+?)\s*</td>", t,
                            flags=re.I):
        names[k] = nm
    out = [dict(karmn=k, name=names.get(k, "")) for k in dict.fromkeys(ids)]
    json.dump(out, open(os.path.join(carmenes_ROOT, "stars.json"), "w"), indent=1)
    print("%d stars" % len(out))


def carmenes_stream_zip(karmn, k):
    """First k car-*-vis_A.fits members of a star's zip read from the stream."""
    r = urllib.request.urlopen(urllib.request.Request(carmenes_ZIP % karmn.replace("+", "%2B"), headers=carmenes_UA), timeout=300)

    def readn(n):
        out = bytearray()
        while len(out) < n:
            c = r.read(min(n - len(out), 1 << 20))
            if not c:
                break
            out += c
        return bytes(out)

    got = []
    try:
        while len(got) < k:
            h = readn(30)
            if len(h) < 30 or h[:4] != b"PK\x03\x04":
                break
            _, _, flag, meth, _, _, _, csize, _, nl, el = struct.unpack("<IHHHHHIIIHH", h)
            if flag & 8:
                raise IOError("zip member without sizes in its local header")
            name = readn(nl).decode("utf-8", "replace")
            readn(el)
            data = readn(csize)
            if name.endswith("vis_A.fits"):
                got.append((name, zlib.decompress(data, -15) if meth == 8 else data))
    finally:
        r.close()
    return got


def carmenes_load(raw):
    from astropy.io import fits
    with fits.open(io.BytesIO(raw)) as f:
        h = f[0].header
        return dict(spec=f["SPEC"].data.astype(np.float32), sig=f["SIG"].data.astype(np.float32),
                    wave=f["WAVE"].data.astype(np.float64), berv=float(h.get("HIERARCH CARACAL BERV", np.nan)),
                    rv=float(h.get("HIERARCH CARACAL SERVAL RV", np.nan)), mjd=float(h.get("MJD-OBS", np.nan)),
                    airmass=float(h.get("AIRMASS", np.nan)), exptime=float(h.get("EXPTIME", np.nan)))


def carmenes_fetch_star(karmn, k):
    """A star's first k spectra, cached as one npz."""
    d = os.path.join(carmenes_ROOT, "cache")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "%s.npz" % karmn)
    if os.path.exists(p):
        z = dict(np.load(p))  # each member read once
        n = len(z["mjd"])
        if n >= k or int(z["complete"]):
            return [dict(spec=z["spec"][i], sig=z["sig"][i], wave=z["wave"][i], berv=float(z["berv"][i]),
                         rv=float(z["rv"][i]), mjd=float(z["mjd"][i]), airmass=float(z["airmass"][i]),
                         exptime=float(z["exptime"][i]), name=str(z["names"][i])) for i in range(min(n, k))]
    for attempt in range(4):
        try:
            members = carmenes_stream_zip(karmn, k)
            break
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))
            print("%s: retry after %s" % (karmn, str(e)[:80]), flush=True)
    eps = []
    for name, raw in members:
        try:
            x = carmenes_load(raw)
        except Exception:
            continue
        x["name"] = name
        eps.append(x)
    if eps:
        np.savez_compressed(p, spec=np.array([e["spec"] for e in eps]), sig=np.array([e["sig"] for e in eps]),
                            wave=np.array([e["wave"] for e in eps]), berv=[e["berv"] for e in eps],
                            rv=[e["rv"] for e in eps], mjd=[e["mjd"] for e in eps],
                            airmass=[e["airmass"] for e in eps], exptime=[e["exptime"] for e in eps],
                            names=[e["name"] for e in eps], complete=int(len(members) < k))
    return eps


def carmenes_peaks(ep):
    """Unresolved peaks of one spectrum: [(order, pixel, wavelength A, height SIG, sigma px, rel)]."""
    from scipy.ndimage import percentile_filter
    from scipy.optimize import curve_fit
    out = []
    S, E, W = ep["spec"], ep["sig"], ep["wave"]
    for o in range(S.shape[0]):
        x, s, w = S[o].astype(float), E[o].astype(float), W[o]
        ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
        if ok.sum() < 1000:
            continue
        xf = np.where(ok, x, np.nanmedian(x[ok]))
        env = percentile_filter(xf, carmenes_PCT, size=carmenes_WIN, mode="nearest")
        snr = np.nanmedian(x[ok] / s[ok])
        if not np.isfinite(snr) or snr < 10:
            continue
        z = np.where(ok, (x - env) / np.where(ok, s, 1.0), 0.0)
        cand = np.nonzero((z > carmenes_THR) & ok)[0]
        cand = cand[(cand > carmenes_EDGE) & (cand < len(x) - carmenes_EDGE)]
        done = set()
        for i in cand:
            if i in done:
                continue
            j0 = max(i - 5, 0)
            seg = z[j0:i + 6]
            i = j0 + int(np.argmax(seg))
            if i in done:
                continue
            done.update(range(i - 5, i + 6))
            if z[i - 1] < 1.5 or z[i + 1] < 1.5:
                continue
            xx = np.arange(i - 6, i + 7)
            yy = (x[xx] - env[xx]) / env[i]
            ee = s[xx] / env[i]
            try:
                p, _ = curve_fit(lambda t, a, m, sg, b: a * np.exp(-0.5 * ((t - m) / sg) ** 2) + b, xx, yy,
                                 p0=(yy[6], i, carmenes_SIG_PX, 0.0), sigma=ee, maxfev=2000)
            except Exception:
                continue
            a, m, sg, b = p
            sg = abs(sg)
            if a <= 0 or abs(m - i) > 1.5 or not (0.6 * carmenes_SIG_PX <= sg <= 1.6 * carmenes_SIG_PX):
                continue
            lam = float(np.interp(m, np.arange(len(w)), w))
            out.append((o, int(i), lam, float(z[i]), float(sg), float(a)))
    return out


def carmenes_group(cands, tol=1.5):
    """Peak groups within tol km/s in the star's frame."""
    cands = sorted(cands, key=lambda c: c["rest"])
    groups, cur = [], []
    for c in cands:
        if cur and (c["rest"] - cur[-1]["rest"]) / c["rest"] * carmenes_C > tol:
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)
    return [g for g in groups if len({c["ep"] for c in g}) >= 2]


def carmenes_persistent(groups):
    """Groups whose epochs span a resolution element in the observatory frame."""
    out = []
    for g in groups:
        obs = [c["obs"] for c in g]
        span = (max(obs) - min(obs)) / np.mean(obs) * carmenes_C
        out.append(dict(rest=float(np.mean([c["rest"] for c in g])), obs_span_kms=float(span),
                        epochs=sorted({c["ep"] for c in g}), z=[round(c["z"], 1) for c in g],
                        separated=bool(span >= carmenes_C / carmenes_R)))
    return out


def carmenes_one_star(args):
    karmn, k = args
    quiet_warnings()
    try:
        eps = carmenes_fetch_star(karmn, k)
    except Exception as e:
        return dict(karmn=karmn, error=str(e)[:200])
    if not eps:
        return dict(karmn=karmn, n=0)
    rv = float(np.nanmedian([e["rv"] for e in eps]))
    vel = carmenes_frame(karmn, rv, len(eps))
    if karmn not in _carmenes_dv:
        dv = carmenes_register_star(karmn)[1]
        vel = [rv + (dv[i] if i < len(dv) else 0.0) for i in range(len(eps))]
    cands = []
    for ie, ep in enumerate(eps):
        for o, i, lam, z, sg, a in carmenes_peaks(ep):
            cands.append(dict(ep=ie, order=o, px=i, obs=lam, z=z, sig_px=sg, amp=a))
    r = dict(karmn=karmn, n=len(eps), rv=rv, berv=[round(e["berv"], 3) for e in eps], mjd=[e["mjd"] for e in eps],
             snr=[float(np.nanmedian(e["spec"][40:50] / e["sig"][40:50])) for e in eps],
             cands=[dict(c, obs=round(c["obs"], 4), z=round(c["z"], 2), sig_px=round(c["sig_px"], 2),
                         amp=round(c["amp"], 4)) for c in cands])
    return carmenes_groups_in_frame(r, vel)


def carmenes_groups_in_frame(r, vel):
    """Peaks in the star's frame, grouped, with persistent groups and the random-offset chance count."""
    cands =[dict(c, rest=round(c["obs"] * (1 + r["berv"][c["ep"]] / carmenes_C) / (1 + vel[c["ep"]] / carmenes_C),
                                4)) for c in r["cands"]]
    groups = carmenes_group(cands)
    rng = np.random.default_rng(zlib.crc32(r["karmn"].encode()))
    nulls = []
    for _ in range(20):
        off = {ie: rng.uniform(30, 300) * rng.choice((-1, 1)) for ie in range(r["n"])}
        sh = [dict(c, rest=c["rest"] * (1 + off[c["ep"]] / carmenes_C)) for c in cands]
        nulls.append(sum(1 for g in carmenes_persistent(carmenes_group(sh)) if g["separated"]))
    per = carmenes_persistent(groups)
    return dict(r, vel=[round(v, 3) for v in vel], cands=cands, persistent=per,
                n_separated=sum(1 for g in per if g["separated"]), null_separated=float(np.mean(nulls)))


# telluric-free orders for registration (vacuum ranges)
carmenes_CLEAN = ((5500, 6260), (6340, 6850), (6970, 7150), (7350, 7580), (7730, 8090), (8420, 8880))
carmenes_REGISTER = os.path.join(core.RESULTS, "carmenes_register.json")
_carmenes_dv = None


def carmenes_highpass(f, w):
    """High-passed spectrum: f over blockwise median, minus one, clipped at five deviations."""
    nb = max(len(f) // w, 1)
    cen = (np.arange(nb) + 0.5) * len(f) / nb
    med = np.array([np.median(b) for b in np.array_split(f, nb)])
    h = f / np.interp(np.arange(len(f)), cen, med) - 1.0
    mad = 1.4826 * np.median(np.abs(h - np.median(h)))
    return np.clip(h, -5 * mad, 5 * mad)


def carmenes_register_star(karmn, vmax=40.0, step=0.5):
    """Each epoch's velocity offset (km/s) from the median by cross-correlation over telluric-free orders."""
    p = os.path.join(carmenes_ROOT, "cache", "%s.npz" % karmn)
    if not os.path.exists(p):
        return karmn, []
    z = dict(np.load(p))  # each member read once
    n = len(z["mjd"])
    if n < 2:
        return karmn, [0.0] * n
    rv0 = float(np.nanmedian(z["rv"]))
    W, S, berv = z["wave"], z["spec"], z["berv"]
    shifts = np.arange(-int(vmax / step), int(vmax / step) + 1)
    ccf = np.zeros((n, len(shifts)))
    for o in range(W.shape[1]):
        if not any(lo < float(np.nanmedian(W[0][o])) < hi for lo, hi in carmenes_CLEAN):
            continue
        rest = [W[i][o] * (1 + float(berv[i]) / carmenes_C) / (1 + rv0 / carmenes_C) for i in range(n)]
        if not all(np.isfinite(r[carmenes_EDGE]) and np.isfinite(r[-carmenes_EDGE]) for r in rest):
            continue
        lo = max(r[carmenes_EDGE] for r in rest)
        hi = min(r[-carmenes_EDGE] for r in rest)
        if hi - lo < 20:
            continue
        g = np.exp(np.arange(np.log(lo), np.log(hi), step / carmenes_C))
        F = np.full((n, len(g)), np.nan)
        for i in range(n):
            x = S[i][o].astype(float)
            ok = np.isfinite(x) & np.isfinite(rest[i])
            if ok.sum() < 1000:
                continue
            F[i] = carmenes_highpass(np.interp(g, rest[i][ok], x[ok]), int(3.0 / (g[len(g) // 2] * step / carmenes_C)))
        for i in range(n):
            if not np.isfinite(F[i]).all():
                continue
            others = [j for j in range(n) if j != i and np.isfinite(F[j]).all()]
            if not others:
                continue
            T = np.median(F[others], axis=0)
            m = len(g)
            norm = np.sqrt(np.dot(T, T) * np.dot(F[i], F[i]))
            if not norm > 0:
                continue
            for k, sft in enumerate(shifts):
                if sft >= 0:
                    ccf[i, k] += float(np.dot(T[:m - sft], F[i][sft:])) / norm
                else:
                    ccf[i, k] += float(np.dot(T[-sft:], F[i][:m + sft])) / norm
    dv = []
    for i in range(n):
        k = int(np.argmax(ccf[i]))
        if not ccf[i].any() or k in (0, len(shifts) - 1):
            dv.append(0.0)
            continue
        a, b, c = ccf[i, k - 1], ccf[i, k], ccf[i, k + 1]
        frac = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        dv.append(float((shifts[k] + frac) * step))
    return karmn, dv


def carmenes_frame(karmn, rv0, n):
    """Velocities moving a star's epochs to its frame."""
    global _carmenes_dv
    if _carmenes_dv is None:
        _carmenes_dv = json.load(open(carmenes_REGISTER)) if os.path.exists(carmenes_REGISTER) else {}
    e = _carmenes_dv.get(karmn)
    dv, ab = (e.get("dv") or [], e.get("abs") or 0.0) if isinstance(e, dict) else (e or [], 0.0)
    return [rv0 + ab + (dv[i] if i < len(dv) else 0.0) for i in range(n)]


def carmenes_velocities(karmn, z):
    """carmenes_frame for cached epochs."""
    return carmenes_frame(karmn, float(np.nanmedian(z["rv"])), len(z["mjd"]))


# reference stars with literature-consistent velocities (GJ 436, Wolf 359) set each star's absolute frame
carmenes_REFS = ("J11421+267", "J10564+070")
_carmenes_refs = None


def carmenes_star_spectrum(karmn, step=0.5):
    """Epochs in the star's frame, high-passed and median-combined on one log grid: (grid, flux) or None."""
    import warnings
    p = os.path.join(carmenes_ROOT, "cache", "%s.npz" % karmn)
    if not os.path.exists(p):
        return None
    z = dict(np.load(p))
    vel = carmenes_velocities(karmn, z)
    W, S, berv = z["wave"], z["spec"], z["berv"]
    g = np.exp(np.arange(np.log(carmenes_CLEAN[0][0]), np.log(carmenes_CLEAN[-1][1]), step / carmenes_C))
    clean = np.zeros(len(g), bool)
    for a, b in carmenes_CLEAN:
        clean |= (g > a) & (g < b)
    rows = []
    for i in range(len(z["mjd"])):
        f = np.full(len(g), np.nan)
        for o in range(W.shape[1]):
            rest = W[i][o] * (1 + float(berv[i]) / carmenes_C) / (1 + vel[i] / carmenes_C)
            x = S[i][o].astype(float)
            ok = np.isfinite(x) & np.isfinite(rest)
            ok[:carmenes_EDGE] = False
            ok[-carmenes_EDGE:] = False
            if ok.sum() < 1000:
                continue
            sel = (g > rest[ok][0]) & (g < rest[ok][-1]) & clean & np.isnan(f)
            if sel.sum() < 1000:
                continue
            f[sel] = carmenes_highpass(np.interp(g[sel], rest[ok], x[ok]),
                                       int(3.0 / (float(np.median(g[sel])) * step / carmenes_C)))
        rows.append(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return g, np.nanmedian(np.array(rows), axis=0)


def carmenes_absolute_star(karmn, vmax=200.0, step=0.5):
    """Offset (km/s) of a star's frame by cross-correlation with the references; 0 for references or weak peaks."""
    global _carmenes_refs
    if karmn in carmenes_REFS:
        return karmn, 0.0, 1.0, karmn
    if _carmenes_refs is None:
        _carmenes_refs = {r: carmenes_star_spectrum(r, step) for r in carmenes_REFS}
    sp = carmenes_star_spectrum(karmn, step)
    if sp is None:
        return karmn, 0.0, 0.0, None
    sv = np.nan_to_num(sp[1])
    sm = np.isfinite(sp[1])
    best = (0.0, 0.0, None)
    shifts = np.arange(-int(vmax / step), int(vmax / step) + 1)
    for name, ref in _carmenes_refs.items():
        if ref is None:
            continue
        rv_, rm = np.nan_to_num(ref[1]), np.isfinite(ref[1])
        m = len(sv)
        c = np.zeros(len(shifts))
        for k, t in enumerate(shifts):
            a, b = (sv[t:], rv_[:m - t]) if t >= 0 else (sv[:m + t], rv_[-t:])
            ma, mb = (sm[t:], rm[:m - t]) if t >= 0 else (sm[:m + t], rm[-t:])
            both = ma & mb
            den = np.sqrt(np.sum(a[both] ** 2) * np.sum(b[both] ** 2))
            c[k] = np.sum(a[both] * b[both]) / den if den > 0 else 0.0
        k = int(np.argmax(c))
        if 0 < k < len(c) - 1 and c[k] > best[1]:
            d = c[k - 1] - 2 * c[k] + c[k + 1]
            frac = 0.5 * (c[k - 1] - c[k + 1]) / d if d != 0 else 0.0
            best = (float((shifts[k] + frac) * step), float(c[k]), name)
    if best[1] < 0.15:
        return karmn, 0.0, best[1], best[2]
    return karmn, best[0], best[1], best[2]


def carmenes_register(paths, procs):
    """Register every star and write carmenes_register.json."""
    from concurrent.futures import ProcessPoolExecutor
    global _carmenes_dv
    res = carmenes_load_results(paths)
    out = {}
    with ProcessPoolExecutor(procs) as ex:
        for karmn, dv in ex.map(carmenes_register_star, [r["karmn"] for r in res], chunksize=2):
            out[karmn] = [round(d, 3) for d in dv]
    json.dump(out, open(carmenes_REGISTER, "w"), indent=0)
    _carmenes_dv = None
    # each star's frame against the references
    ab = {}
    with ProcessPoolExecutor(procs) as ex:
        for karmn, off, peak, ref in ex.map(carmenes_absolute_star, [r["karmn"] for r in res], chunksize=2):
            ab[karmn] = (off, peak, ref)
    full = {k: dict(dv=v, abs=round(ab[k][0], 3), ccf=round(ab[k][1], 3), ref=ab[k][2]) for k, v in out.items()}
    json.dump(full, open(carmenes_REGISTER, "w"), indent=0)
    _carmenes_dv = None
    far = sorted(((k, v) for k, v in full.items() if abs(v["abs"]) > 3.0), key=lambda kv: -abs(kv[1]["abs"]))
    print("%d stars placed against the references; %d more than 3 km/s from the recorded velocity, %d with a peak "
          "under 0.15 left at it" % (len(full), len(far), sum(1 for v in full.values() if v["ccf"] < 0.15)))
    for k, v in far[:40]:
        print("  %s %+.2f km/s (peak %.2f against %s)" % (k, v["abs"], v["ccf"], v["ref"]))
    big = {k: v for k, v in out.items() if v and max(abs(d) for d in v) > 1.5}
    print("%d stars registered; %d have an epoch more than 1.5 km/s from the median recorded velocity's frame"
          % (len(out), len(big)))
    for k, v in sorted(big.items(), key=lambda kv: -max(abs(d) for d in kv[1]))[:30]:
        print("  %s %s" % (k, v))


def carmenes_regroup(paths, out):
    """Groups and chance rate recomputed in registered frames."""
    global _carmenes_dv
    _carmenes_dv = json.load(open(carmenes_REGISTER))
    res = carmenes_load_results(paths)
    path = os.path.join(carmenes_HERE, out)
    with open(path, "w") as fh:
        for r in res:
            r = carmenes_groups_in_frame(r, carmenes_frame(r["karmn"], r["rv"], r["n"]))
            fh.write(json.dumps(r) + "\n")
    print("%d stars regrouped into %s: %d persistent frame-separated groups against %.1f by chance"
          % (len(res), path, sum(r["n_separated"] for r in carmenes_load_results([path])),
             sum(r["null_separated"] for r in carmenes_load_results([path]))))


def carmenes_stream(share, out, k, procs):
    from concurrent.futures import ProcessPoolExecutor
    ss = json.load(open(share or os.path.join(carmenes_ROOT, "stars.json")))
    os.makedirs(carmenes_HERE, exist_ok=True)
    path = os.path.join(carmenes_HERE, out)
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)["karmn"] for line in open(path) if line.strip()}
    todo = [s["karmn"] for s in ss if s["karmn"] not in done]
    print("%d stars to search (%d done)" % (len(todo), len(done)), flush=True)
    with ProcessPoolExecutor(procs) as ex, open(path, "a") as fh:
        for r in ex.map(carmenes_one_star, [(x, k) for x in todo], chunksize=1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("%s: %s spectra, %d peaks, %s persistent and separated (chance %.2f)%s"
                  % (r["karmn"], r.get("n"), len(r.get("cands", [])), r.get("n_separated"),
                     r.get("null_separated", float("nan")), "; " + r["error"] if "error" in r else ""), flush=True)


def carmenes_load_results(paths):
    seen, res = set(), []
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r["karmn"] not in seen and r.get("n"):
                    seen.add(r["karmn"])
                    res.append(r)
    return res


class carmenes_Recurrence:
    """All stars' peaks in one frame for counting recurrences against local chance."""

    def __init__(self, res, key):
        pts = sorted((c[key], r["karmn"]) for r in res for c in r["cands"])
        self.lam = np.array([p[0] for p in pts])
        self.who = [p[1] for p in pts]

    def count(self, lam, star, tol):
        lo = np.searchsorted(self.lam, lam * (1 - tol / carmenes_C))
        hi = np.searchsorted(self.lam, lam * (1 + tol / carmenes_C), side="right")
        return len({self.who[j] for j in range(lo, hi)} - {star})

    def test(self, lam, star, tol=1.5, offsets=(-50, -40, -30, -20, -10, 10, 20, 30, 40, 50)):
        """(stars at lam, mean at offsets, recurring): recurring at four Poisson deviations and at least 3."""
        n = self.count(lam, star, tol)
        bg = float(np.mean([self.count(lam * (1 + d / carmenes_C), star, tol) for d in offsets]))
        return n, bg, bool(n >= 3 and n > bg + 4 * np.sqrt(bg + 1))


def carmenes_vet(paths):
    res = carmenes_load_results(paths)
    obs = carmenes_Recurrence(res, "obs")
    rest = carmenes_Recurrence(res, "rest")
    n_sep = sum(r["n_separated"] for r in res)
    n_null = sum(r["null_separated"] for r in res)
    print("%d stars, %d spectra, %d single-epoch peaks; persistent and frame-separated groups %d against %.1f by chance"
          % (len(res), sum(r["n"] for r in res), sum(len(r["cands"]) for r in res), n_sep, n_null), flush=True)
    left, why = [], {}
    for r in res:
        for g in r["persistent"]:
            if not g["separated"]:
                continue
            mem = [c for c in r["cands"] if abs(c["rest"] - g["rest"]) / g["rest"] * carmenes_C < 1.5]
            o = [obs.test(c["obs"], r["karmn"]) for c in mem]
            s = rest.test(g["rest"], r["karmn"])
            if sum(x[2] for x in o) * 2 >= len(o):
                why["airglow or instrument"] = why.get("airglow or instrument", 0) + 1
            elif s[2]:
                why["stellar line"] = why.get("stellar line", 0) + 1
            else:
                left.append(dict(karmn=r["karmn"], n=r["n"], rest=g["rest"], epochs=g["epochs"], z=g["z"],
                                 obs_span_kms=g["obs_span_kms"], rest_recurrence=s[:2],
                                 obs_recurrence=[x[:2] for x in o], members=mem))
    print("explained: %s; left %d" % (why, len(left)))
    for x in sorted(left, key=lambda x: -len(x["epochs"]))[:60]:
        print("  %s rest %.3f A (vacuum), epochs %s of %d, z %s, observatory span %.1f km/s, other stars at it %d "
              "(local %.1f)" % (x["karmn"], x["rest"], x["epochs"], x["n"], x["z"], x["obs_span_kms"],
                                x["rest_recurrence"][0], x["rest_recurrence"][1]))
    json.dump(left, open(os.path.join(carmenes_HERE, "carmenes_left.json"), "w"), indent=1)


carmenes_OFFSETS = (-50, -40, -30, -20, -10, 10, 20, 30, 40, 50)


def carmenes_forced_star(args):
    """Largest excess (SIG) in any cached spectrum at each rest wavelength and its offsets."""
    from scipy.ndimage import percentile_filter
    karmn, rest = args
    p = os.path.join(carmenes_ROOT, "cache", "%s.npz" % karmn)
    shifts = np.array((0.0,) + carmenes_OFFSETS)
    best = np.full((len(rest), len(shifts)), -np.inf)
    if not os.path.exists(p):
        return karmn, best
    z = dict(np.load(p))  # each member read once
    vel = carmenes_velocities(karmn, z)
    lam = np.outer(np.asarray(rest, float), 1 + shifts / carmenes_C)
    for i in range(len(z["mjd"])):
        S, E, W, berv = z["spec"][i], z["sig"][i], z["wave"][i], float(z["berv"][i])
        obs = lam * (1 + vel[i] / carmenes_C) / (1 + berv / carmenes_C)
        for o in range(S.shape[0]):
            x, s, w = S[o].astype(float), E[o].astype(float), W[o]
            ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
            if ok.sum() < 1000:
                continue
            sel = (obs > w[carmenes_EDGE]) & (obs < w[-carmenes_EDGE])
            if not sel.any():
                continue
            env = percentile_filter(np.where(ok, x, np.nanmedian(x[ok])), carmenes_PCT, size=carmenes_WIN, mode="nearest")
            zz = np.where(ok, (x - env) / np.where(ok, s, 1.0), -np.inf)
            j = np.searchsorted(w, obs[sel])
            v = np.max([zz[np.clip(j + d, 0, len(w) - 1)] for d in (-2, -1, 0, 1, 2)], axis=0)
            best[sel] = np.maximum(best[sel], v)
    return karmn, best


def carmenes_forced(paths, procs, thr=4.0):
    """Groups measured in every other star: stellar when three or more stars exceed thr beyond the offset count."""
    from concurrent.futures import ProcessPoolExecutor
    res = carmenes_load_results(paths)
    left = json.load(open(os.path.join(carmenes_HERE, "carmenes_left.json")))
    rest = [x["rest"] for x in left]
    hits = np.zeros((len(left), 1 + len(carmenes_OFFSETS)), int)
    holders = [[] for _ in left]
    with ProcessPoolExecutor(procs) as ex:
        for karmn, best in ex.map(carmenes_forced_star, [(r["karmn"], rest) for r in res], chunksize=2):
            above = best > thr
            for k, x in enumerate(left):
                if x["karmn"] != karmn:
                    hits[k] += above[k]
                    if above[k, 0]:
                        holders[k].append(karmn)
    out = []
    for k, x in enumerate(left):
        n, bg = int(hits[k, 0]), float(np.mean(hits[k, 1:]))
        x = dict(x, forced_stars=n, forced_local=bg, forced_holders=holders[k][:20])
        x["stellar"] = bool(n >= 3 and n > bg + 4 * np.sqrt(bg + 1))
        out.append(x)
    kept = [x for x in out if not x["stellar"]]
    from collections import Counter
    print("%d groups measured in %d stars; %d held by other stars above %g sigma; %d left in %d stars"
          % (len(out), len(res), len(out) - len(kept), thr, len(kept), len({x["karmn"] for x in kept})))
    print("left by star: " + ", ".join("%s %d" % kv for kv in Counter(x["karmn"] for x in kept).most_common(25)))
    # members must stand 3 SIG above every pixel within 2.5 A
    by = {}
    for x in kept:
        by.setdefault(x["karmn"], []).append(x)
    with ProcessPoolExecutor(procs) as ex:
        for karmn, flags in ex.map(carmenes_above_star, [(k, v) for k, v in by.items()]):
            for x, f in zip(by[karmn], flags):
                x["above_maxima"] = f
    for x in out:
        x.setdefault("above_maxima", None)
    kept = [x for x in kept if x["above_maxima"] and sum(x["above_maxima"]) * 2 >= len(x["above_maxima"])]
    print("%d of them stand above the spectrum's maxima within 2.5 A in half their epochs or more, in %d stars"
          % (len(kept), len({x["karmn"] for x in kept})))
    print("left by star: " + ", ".join("%s %d" % kv for kv in Counter(x["karmn"] for x in kept).most_common(25)))
    # peculiar-spectrum features checked in the five nearest spectral twins
    with ProcessPoolExecutor(procs) as ex:
        prof = dict(ex.map(carmenes_profile, [r["karmn"] for r in res]))
    prof = {k: v for k, v in prof.items() if v is not None}
    names = sorted(prof)
    M = np.array([prof[k] for k in names])
    jobs = {}
    for x in kept:
        if x["karmn"] not in prof:
            continue
        d = np.nanstd(M - prof[x["karmn"]], axis=1)
        twins = [names[j] for j in np.argsort(d) if names[j] != x["karmn"]][:5]
        x["twins"] = twins
        for t in twins:
            jobs.setdefault(t, []).append(x["rest"])
    with ProcessPoolExecutor(procs) as ex:
        tw = dict(ex.map(carmenes_twin_star, list(jobs.items())))
    for x in kept:
        hits = np.zeros(1 + len(carmenes_OFFSETS))
        for t in x.get("twins", []):
            hits += tw[t][x["rest"]]
        x["twin_hits"], x["twin_local"] = int(hits[0]), float(np.mean(hits[1:]))
        x["twin_shared"] = bool(hits[0] >= 2 and hits[0] >= x["twin_local"] + 2)
    left2 = [x for x in kept if not x.get("twin_shared")]
    print("%d of them show in two or more of their star's five nearest spectral twins; %d left in %d stars"
          % (len(kept) - len(left2), len(left2), len({x["karmn"] for x in left2})))
    print("left by star: " + ", ".join("%s %d" % kv for kv in Counter(x["karmn"] for x in left2).most_common(25)))
    # remaining groups against twin fits and the star's other maxima
    carmenes_finish(out, carmenes_twin_stage(left2, procs))
    json.dump(out, open(os.path.join(carmenes_HERE, "carmenes_forced.json"), "w"), indent=1)


def carmenes_finish(out, left):
    ids = {id(x) for x in left}
    for x in out:
        x["left"] = id(x) in ids
    for x in sorted(left, key=lambda x: -len(x["epochs"]))[:40]:
        print("  %s rest %.3f A, epochs %s of %d, z %s, other stars %d (local %.1f)"
              % (x["karmn"], x["rest"], x["epochs"], x["n"], x["z"], x["forced_stars"], x["forced_local"]))


def carmenes_refit(procs):
    """Rerun carmenes_forced's last stage on stored groups."""
    path = os.path.join(carmenes_HERE, "carmenes_forced.json")
    out = json.load(open(path))
    groups = [x for x in out if x.get("twins") and x.get("above_maxima")
              and sum(x["above_maxima"]) * 2 >= len(x["above_maxima"]) and not x.get("twin_shared")]
    print("%d groups from the earlier stages" % len(groups))
    carmenes_finish(out, carmenes_twin_stage(groups, procs))
    json.dump(out, open(path, "w"), indent=1)


def carmenes_profile(karmn):
    """(karmn, log median flux per order relative to the median order), tracking spectral type."""
    p = os.path.join(carmenes_ROOT, "cache", "%s.npz" % karmn)
    if not os.path.exists(p):
        return karmn, None
    z = dict(np.load(p))  # each member read once
    S = z["spec"].astype(float)
    with np.errstate(all="ignore"):
        med = np.nanmedian(S[:, :, 500:-500], axis=2)  # epochs x orders
        v = np.log(np.nanmedian(med, axis=0))
    v = v - np.nanmedian(v)
    return karmn, v if np.isfinite(v).sum() > 20 else None


def carmenes_twin_star(args):
    """(karmn, {wavelength: 3-SIG local-maximum flags at the wavelength and offsets})."""
    karmn, rests = args
    p = os.path.join(carmenes_ROOT, "cache", "%s.npz" % karmn)
    out = {r: np.zeros(1 + len(carmenes_OFFSETS)) for r in rests}
    if not os.path.exists(p):
        return karmn, out
    z = dict(np.load(p))  # each member read once
    vel = carmenes_velocities(karmn, z)
    shifts = np.array((0.0,) + carmenes_OFFSETS)
    for r in set(rests):
        for i in range(len(z["mjd"])):
            W, berv = z["wave"][i], float(z["berv"][i])
            for k, s in enumerate(shifts):
                if out[r][k]:
                    continue
                lam = r * (1 + s / carmenes_C) * (1 + vel[i] / carmenes_C) / (1 + berv / carmenes_C)
                for o in range(W.shape[0]):
                    w = W[o]
                    if not (w[carmenes_EDGE] < lam < w[-carmenes_EDGE]):
                        continue
                    x, e = z["spec"][i][o].astype(float), z["sig"][i][o].astype(float)
                    j = int(np.searchsorted(w, lam))
                    j = j - 2 + int(np.nanargmax(x[j - 2:j + 3])) if np.isfinite(x[j - 2:j + 3]).any() else j
                    half = int(round(2.5 / max(abs(w[min(j + 1, len(w) - 1)] - w[j]), 1e-6)))
                    nb = np.r_[x[max(j - half, 0):max(j - 3, 0)], x[j + 4:j + half + 1]]
                    nb = nb[np.isfinite(nb)]
                    if len(nb) and np.isfinite(x[j]) and e[j] > 0 and x[j] - nb.max() > 3 * e[j]:
                        out[r][k] = 1
    return karmn, out


def carmenes_rest_template(karmn, lam, half=4.0):
    """Rest-frame template around lam (+-half A) median-combined over epochs, or None."""
    import warnings
    p = os.path.join(carmenes_ROOT, "cache", "%s.npz" % karmn)
    if not os.path.exists(p):
        return None
    z = dict(np.load(p))  # each member read once
    vel = carmenes_velocities(karmn, z)
    grid = lam + np.arange(-half, half + 0.0001, 0.01)
    rows = []
    for i in range(len(z["mjd"])):
        W, berv = z["wave"][i], float(z["berv"][i])
        best = None
        for o in range(W.shape[0]):
            rest = W[o] * (1 + berv / carmenes_C) / (1 + vel[i] / carmenes_C)
            inside = (grid > rest[carmenes_EDGE]) & (grid < rest[-carmenes_EDGE])
            if inside.any() and (best is None or inside.sum() > best[0].sum()):
                best = (inside, o, rest)
        if best is None:
            continue
        inside, o, rest = best
        x = z["spec"][i][o].astype(float)
        ok = np.isfinite(x)
        f = np.full(grid.shape, np.nan)
        f[inside] = np.interp(grid[inside], rest[ok], x[ok])
        rows.append(f / np.nanmedian(f[inside]))
    if not rows:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return grid, np.nanmedian(np.array(rows), axis=0)


def carmenes_twin_fit(x, s, rest, i, temps):
    """Excess (SIG) of pixel i over a non-negative fit of twin templates and a linear continuum over +-3 A; None when underconstrained."""
    from scipy.optimize import nnls
    idx = np.where(np.abs(rest - rest[i]) < 3.0)[0]
    r = rest[idx]
    u = (r - rest[i]) / 3.0
    A = np.vstack([np.interp(r, tg, tf, left=np.nan, right=np.nan) for tg, tf in temps]
                  + [np.ones_like(u), -np.ones_like(u), u, -u]).T
    xx, ss = x[idx], s[idx]
    good = np.isfinite(xx) & np.isfinite(ss) & (ss > 0) & np.all(np.isfinite(A), axis=1)
    k = int(np.searchsorted(idx, i))
    fit = good & (np.abs(r - rest[i]) > 0.15)
    if fit.sum() < 50 or k >= len(idx) or idx[k] != i or not good[k]:
        return None
    scale = np.nanmedian(xx[fit])
    coef, _ = nnls(A[fit], xx[fit] / scale)
    return float((xx[k] / scale - A[k] @ coef) * scale / ss[k])


def carmenes_segment_star(args):
    """Per member: twin-fit excess and the same excess at the order's other local maxima 4-25 A away."""
    from scipy.ndimage import maximum_filter1d
    karmn, groups = args
    p = os.path.join(carmenes_ROOT, "cache", "%s.npz" % karmn)
    if not os.path.exists(p):
        return karmn, [None] * len(groups)
    z = dict(np.load(p))  # each member read once
    vel = carmenes_velocities(karmn, z)
    out = []
    for g in groups:
        temps = [carmenes_rest_template(t, g["rest"], half=30.0) for t in g.get("twins", [])]
        temps = [t for t in temps if t is not None]
        res = []
        for c in g["members"]:
            if not temps:
                res.append(None)
                continue
            S, E, W = z["spec"][c["ep"]], z["sig"][c["ep"]], z["wave"][c["ep"]]
            x, s, w = S[c["order"]].astype(float), E[c["order"]].astype(float), W[c["order"]]
            rest = w * (1 + float(z["berv"][c["ep"]]) / carmenes_C) / (1 + vel[c["ep"]] / carmenes_C)
            i = int(c["px"])
            e = carmenes_twin_fit(x, s, rest, i, temps)
            if e is None:
                res.append(None)
                continue
            step = abs(w[min(i + 1, len(w) - 1)] - w[i]) or 0.02
            half = max(int(round(2.5 / step)), 1)
            xf = np.where(np.isfinite(x), x, -np.inf)
            d = np.abs(rest - rest[i])
            peaks = np.where((maximum_filter1d(xf, 2 * half + 1, mode="nearest") == xf) & np.isfinite(x)
                             & (d >= 4.0) & (d <= 25.0))[0]
            peaks = peaks[(peaks >= carmenes_EDGE) & (peaks < len(x) - carmenes_EDGE)]
            ctrl = [v for v in (carmenes_twin_fit(x, s, rest, j, temps) for j in peaks) if v is not None]
            res.append([round(e, 2), [round(v, 2) for v in ctrl]])
        out.append(res)
    return karmn, out


def carmenes_twin_pass(m):
    """Member passes at 5 SIG above the twin fit, above every control, and 5 deviations above their median."""
    if not m:
        return False
    e, ctrl = m
    if e <= 5.0 or (ctrl and e <= max(ctrl)):
        return False
    if len(ctrl) >= 5:
        med = float(np.median(ctrl))
        sd = max(1.4826 * float(np.median(np.abs(np.array(ctrl) - med))), 1.0)
        return (e - med) / sd > 5.0
    return True


def carmenes_twin_stage(groups, procs):
    """Groups passing the twin test in at least half their epochs."""
    from concurrent.futures import ProcessPoolExecutor
    from collections import Counter
    by = {}
    for x in groups:
        by.setdefault(x["karmn"], []).append(x)
    with ProcessPoolExecutor(procs) as ex:
        for karmn, fits in ex.map(carmenes_segment_star, list(by.items())):
            for x, f in zip(by[karmn], fits):
                x["twin_fit"] = f
    kept = [x for x in groups if x.get("twin_fit")
            and sum(carmenes_twin_pass(m) for m in x["twin_fit"]) * 2 >= len(x["twin_fit"])]
    print("%d of them stand no further above their twins' fit than the star's other local maxima within 25 A, in half "
          "their epochs or more; %d left in %d stars" % (len(groups) - len(kept), len(kept),
                                                        len({x["karmn"] for x in kept})))
    print("left by star: " + ", ".join("%s %d" % kv for kv in Counter(x["karmn"] for x in kept).most_common(25)))
    return kept


def carmenes_above_star(args):
    """Per member: whether its pixel stands 3 SIG above every pixel within 2.5 A."""
    karmn, groups = args
    p = os.path.join(carmenes_ROOT, "cache", "%s.npz" % karmn)
    if not os.path.exists(p):
        return karmn, [None] * len(groups)
    z = dict(np.load(p))  # each member read once
    out = []
    for g in groups:
        flags = []
        for c in g["members"]:
            S, E, W = z["spec"][c["ep"]], z["sig"][c["ep"]], z["wave"][c["ep"]]
            x, s, w = S[c["order"]].astype(float), E[c["order"]].astype(float), W[c["order"]]
            i = int(c["px"])
            half = int(round(2.5 / max(abs(w[min(i + 1, len(w) - 1)] - w[i]), 1e-6)))
            lo, hi = max(i - half, 0), min(i + half + 1, len(x))
            nb = np.r_[x[lo:max(i - 3, lo)], x[min(i + 4, hi):hi]]
            nb = nb[np.isfinite(nb)]
            flags.append(bool(len(nb) and np.isfinite(x[i]) and s[i] > 0 and x[i] - nb.max() > 3 * s[i]))
        out.append(flags)
    return karmn, out


def carmenes_record(paths):
    """Record the run and each star's result; limit as the equivalent width (mA) of a THR-SIG line at 825 nm."""
    res = carmenes_load_results(paths)
    left = [x for x in json.load(open(os.path.join(carmenes_HERE, "carmenes_forced.json"))) if x.get("left")]
    by = {}
    for x in left:
        by.setdefault(x["karmn"], []).append(x)
    def ew(snr):
        return carmenes_THR / snr * math.sqrt(2 * math.pi) * 8250.0 / carmenes_R / 2.3548 * 1000.0
    lims = [ew(np.median(r["snr"])) for r in res if np.median(r["snr"]) > 0]
    n_sep = sum(r["n_separated"] for r in res)
    n_null = sum(r["null_separated"] for r in res)
    core.record_search(
        "laser_carmenes_vis", "Laser lines, CARMENES visible channel", archive="CARMENES DR1 public visible-channel "
        "spectra (Ribas et al. 2023)", release="DR1, 2016-2020", selection="the first %d spectra of each DR1 star"
        % max(r["n"] for r in res), code=os.path.abspath(__file__), entry="carmenes", n_targets=len(res),
        params=dict(threshold=carmenes_THR, envelope_window_px=carmenes_WIN, envelope_percentile=carmenes_PCT, group_kms=1.5,
                    recurrence="3 or more other stars and 4 Poisson deviations above the local rate"),
        null_summary="%d persistent frame-separated groups against %.0f after random velocity offsets between epochs"
                     % (n_sep, n_null),
        sensitivity="median %.1f mA equivalent width at 825 nm for a line %g SIG above the envelope"
                    % (np.median(lims), carmenes_THR),
        ledger="Laser lines in CARMENES spectra of nearby M dwarfs",
        result="%d persistent groups left after the airglow, instrument and stellar-line filters, the forced "
               "measurement in the other stars and the fit of the star's spectral twins" % len(left))
    core.clear_targets("laser_carmenes_vis")
    core.record_targets("laser_carmenes_vis", [
        dict(target="carmenes:%s" % r["karmn"], stat=len(by.get(r["karmn"], [])), unit="persistent groups left",
             upper_limit=round(ew(np.median(r["snr"])), 2) if np.median(r["snr"]) > 0 else None,
             note="%d spectra, BERV %.0f to %.0f km/s" % (r["n"], min(r["berv"]), max(r["berv"])))
        for r in res])
    print("recorded %d stars" % len(res))


def carmenes_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("stars", "stream", "register", "regroup", "vet", "forced", "refit", "record"))
    p.add_argument("paths", nargs="*")
    p.add_argument("--list")
    p.add_argument("--out", default="carmenes.jsonl")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--procs", type=int, default=2)
    a = p.parse_args()
    if a.mode == "stars":
        carmenes_stars()
    elif a.mode == "stream":
        carmenes_stream(a.list, a.out, a.k, a.procs)
    elif a.mode == "forced":
        carmenes_forced(a.paths, a.procs)
    elif a.mode == "refit":
        carmenes_refit(a.procs)
    elif a.mode == "register":
        carmenes_register(a.paths, a.procs)
    elif a.mode == "regroup":
        carmenes_regroup(a.paths, a.out)
    elif a.mode == "record":
        carmenes_record(a.paths)
    else:
        carmenes_vet(a.paths)


def carmenes_cli():
    quiet_warnings()
    carmenes_main()


# --------------------------------------------------------------------------------------------------------------

skyfibre_DOC = r"""Laser lines in SDSS BOSS and eBOSS sky fibres: unresolved 6-sigma peaks off airglow and unshared, against dips through the same tests, with fibre-fixed, galaxy, two-sign, counterpart and exposure vetting.

    python -I seti.py lasers skyfibre search [--threads 6] [--limit N] [--part a:b/n]
    python -I seti.py lasers skyfibre summary
    python -I seti.py lasers skyfibre record
    python -I seti.py lasers skyfibre galaxies      (fibres whose lines fall at one redshift, with Legacy Surveys and DESI)
    python -I seti.py lasers skyfibre rest          (clean 8-sigma peaks against Legacy Surveys and DESI)
    python -I seti.py lasers skyfibre inject [--limit 40]
    python -I seti.py lasers skyfibre exposures [--threads 4]      (each clean 8-sigma line frame by frame)"""

skyfibre_BASE = "https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/"
skyfibre_ROOT = os.environ.get("SKYFIBRE_ROOT", core.data("skyfibre"))
skyfibre_SNR = 6.0
skyfibre_LSF_PX = 1.1  # instrumental Gaussian sigma (pixels of 1e-4 dex)
skyfibre_MAXRANGES = 100  # Apache returns the whole file beyond MaxRanges
skyfibre_LAYOUT = {}


def skyfibre_get(url, ranges=None, timeout=120):
    """Bytes of a URL or {start: bytes} of several ranges over the thread's kept connection."""
    if not ranges:
        return core.get(url, timeout=timeout)
    parts = core.get_ranges(url, [(a, b - 1) for a, b in ranges], timeout=timeout)
    return {a: p for (a, _), p in zip(ranges, parts)}


def skyfibre_hdus(url, n):
    """Header and data offset of the first n HDUs, predicted from the previous layout and checked."""
    from astropy.io import fits

    def parse(raw, off):
        """(header, data offset, next offset) of the HDU at off, or None before END."""
        k = next((j for j in range(0, len(raw) - 79, 80) if raw[j:j + 8] == b"END     "), None)
        if k is None:
            return None
        h = fits.Header.fromstring(raw[:k + 80].decode("ascii", "replace"))
        size = abs(h.get("BITPIX", 8)) // 8
        for a in range(1, h.get("NAXIS", 0) + 1):
            size *= h["NAXIS%d" % a]
        size = (size + h.get("PCOUNT", 0)) if h.get("NAXIS", 0) else 0
        data = off + (k // 2880 + 1) * 2880
        return h, data, data + ((size + 2879) // 2880) * 2880

    def read(off, blocks):
        raw = b""
        while True:
            raw += next(iter(skyfibre_get(url, [(off + len(raw), off + len(raw) + 2880 * blocks)]).values()))
            p = parse(raw, off)
            if p is not None:
                return p

    hdus = [(0,) + read(0, 16)]
    d0 = hdus[0][3] - hdus[0][2]
    guess = skyfibre_LAYOUT.get(n)
    if guess:
        starts = [hdus[0][3]]
        for hl, same, dl in guess[:-1]:
            starts.append(starts[-1] + hl + (d0 if same else dl))
        try:
            got = skyfibre_get(url, [(a, a + 2880 * 4) for a in starts])
        except core.RangeError:
            got = {}
        for a in starts:
            raw = got.get(a, b"")
            p = parse(raw, a) if a == hdus[-1][3] and raw.startswith(b"XTENSION") else None
            if p is None:
                break
            hdus.append((a,) + p)
    while len(hdus) < n:
        a = hdus[-1][3]
        hdus.append((a,) + read(a, 4))
    skyfibre_LAYOUT[n] = [(d - a, e - d == d0, e - d) for a, h, d, e in hdus[1:]]
    return [(h, d) for a, h, d, e in hdus]


def skyfibre_rows(url, blocks, rowbytes, dtype):
    """Image HDU rows by multi-range requests: one {row: array} per block."""
    spans = []
    for b, (off, rows) in enumerate(blocks):
        rows = sorted(set(int(r) for r in rows))
        i = 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
                j += 1
            spans.append((b, off, rows[i], rows[j] + 1))
            i = j + 1
    out = [{} for _ in blocks]
    count = rowbytes // np.dtype(dtype).itemsize
    for i in range(0, len(spans), skyfibre_MAXRANGES):
        part = spans[i:i + skyfibre_MAXRANGES]
        res = skyfibre_get(url, [(off + r0 * rowbytes, off + r1 * rowbytes) for b, off, r0, r1 in part])
        for b, off, r0, r1 in part:
            raw = res[off + r0 * rowbytes]
            for r in range(r0, r1):
                out[b][r] = np.frombuffer(raw, dtype=dtype, count=count, offset=(r - r0) * rowbytes)
    return out


def skyfibre_plugmap(url, hd):
    """Plug-map table of an spPlate from its HDU offsets."""
    from astropy.io import fits
    hp, off5 = hd[5]
    size = hp["NAXIS1"] * hp["NAXIS2"] + hp.get("PCOUNT", 0)
    tab = next(iter(skyfibre_get(url, [(off5, off5 + size)]).values()))
    blob = fits.PrimaryHDU().header.tostring().encode() + hp.tostring().encode() + tab + b"\0" * ((2880 - len(tab) % 2880) % 2880)
    return fits.open(io.BytesIO(blob))[1].data


def skyfibre_read(url):
    """Wavelengths, plug map and sky-fibre flux, ivar and sky-model rows of an spPlate; None with fewer than four sky fibres."""
    hd = skyfibre_hdus(url, 7)
    h0 = hd[0][0]
    npix = h0["NAXIS1"]
    lam = 10 ** (h0["COEFF0"] + h0["COEFF1"] * np.arange(npix))
    pm = skyfibre_plugmap(url, hd)
    objtype = np.array([str(x).strip().upper() for x in pm["OBJTYPE"]])
    sky = np.nonzero(objtype == "SKY")[0]
    if len(sky) < 4:
        return None
    rb = npix * 4
    flux, ivar, skym = skyfibre_rows(url, [(hd[0][1], sky), (hd[1][1], sky), (hd[6][1], sky)], rb, ">f4")
    return dict(h0=h0, lam=lam, pm=pm, sky=sky, flux=np.array([flux[r] for r in sky], float),
                ivar=np.array([ivar[r] for r in sky], float), skym=np.array([skym[r] for r in sky], float))


skyfibre_COMMON = os.environ.get("SKYFIBRE_COMMON", "0") == "1"  # subtract each pixel's median over fibres
skyfibre_SHARED_MIN = int(os.environ.get("SKYFIBRE_SHARED_MIN", "1"))  # other fibres at 3 sigma marking a peak shared


def skyfibre_detect(d, common=None, shared_min=None):
    """Peaks of both signs on one file's sky fibres with labels, shared counts, airglow mask and good-pixel map."""
    from scipy.ndimage import median_filter
    common = skyfibre_COMMON if common is None else common
    shared_min = skyfibre_SHARED_MIN if shared_min is None else shared_min
    sky, flux, ivar = d["sky"], d["flux"], d["ivar"]
    npix = len(d["lam"])
    med_sky = np.median(d["skym"], axis=0)
    cont = median_filter(med_sky, size=101, mode="nearest")
    on_line = med_sky > 3.0 * np.maximum(cont, 1e-3)
    on_line = np.convolve(on_line.astype(float), np.ones(5), "same") > 0
    k = np.arange(-4, 5)
    w = np.exp(-0.5 * (k / skyfibre_LSF_PX) ** 2)
    Z = np.zeros((len(sky), npix))
    G = np.zeros((len(sky), npix), bool)
    for i in range(len(sky)):
        iv = ivar[i]
        good = (iv > 0) & np.isfinite(flux[i])  # masked pixels have zero ivar
        z = np.where(good, flux[i] * np.sqrt(np.where(good, iv, 0.0)), 0.0)
        s = 1.4826 * np.median(np.abs(z[good] - np.median(z[good]))) if good.sum() > 100 else 1.0
        Z[i] = z / max(s, 1e-3)
        G[i] = good
    if common:
        cm = np.median(np.where(G, Z, np.nan), axis=0)
        Z = np.where(G, Z - np.nan_to_num(cm)[None, :], 0.0)
    num_ = np.array([np.convolve(Z[i], w[::-1], "same") for i in range(len(sky))])
    den = np.sqrt(np.array([np.convolve(G[i].astype(float), (w ** 2)[::-1], "same") for i in range(len(sky))]))
    snr = np.where(den > 0.5 * math.sqrt(np.sum(w ** 2)), num_ / np.maximum(den, 1e-6), 0.0)
    out = []
    for sign in (1, -1):
        v = sign * snr
        for i in range(len(sky)):
            pk = np.nonzero((v[i] >= skyfibre_SNR) & (v[i] >= np.roll(v[i], 1)) & (v[i] > np.roll(v[i], -1)))[0]
            for j in pk:
                if j < 5 or j > npix - 6:
                    continue
                pos = np.clip(sign * Z[i, j - 4:j + 5], 0, None)
                width = math.sqrt(np.sum(pos * k ** 2) / max(np.sum(pos), 1e-9))
                labels = []
                if not G[i, j - 1:j + 2].all():
                    labels.append("masked")
                if on_line[j]:
                    labels.append("airglow")
                others = np.delete(np.arange(len(sky)), i)
                n_shared = int(np.sum(np.max(sign * snr[others, j - 1:j + 2], axis=1) >= 3.0))
                if n_shared >= shared_min:
                    labels.append("shared")
                if not 0.6 * skyfibre_LSF_PX <= width <= 2.0 * skyfibre_LSF_PX:
                    labels.append("width %.2f" % width)
                out.append((sign, i, int(j), float(v[i, j]), float(width), labels, n_shared))
    return out, on_line, G


def skyfibre_plate(plate):
    """Search one plate's sky fibres into ROOT/plates/<plate>.json; returns its summary."""
    out_p = os.path.join(skyfibre_ROOT, "plates", "%s.json" % plate)
    if os.path.exists(out_p):
        return json.load(open(out_p))
    d = skyfibre_get(skyfibre_BASE + "%s/" % plate).decode("utf-8", "replace")
    names = sorted(set(re.findall(r'href="(spPlate-%s-\d+\.fits)"' % plate, d)))
    res = dict(plate=plate, files=[], peaks=[], n_neg=[0, 0, 0], n_pos=[0, 0, 0], n_sky=0, shared_min=skyfibre_SHARED_MIN)
    for name in names:
        d = skyfibre_read(skyfibre_BASE + "%s/%s" % (plate, name))
        if d is None:
            continue
        peaks, _, _ = skyfibre_detect(d)
        pm, sky, lam, h0 = d["pm"], d["sky"], d["lam"], d["h0"]
        res["n_sky"] += len(sky)
        res["files"].append(name)
        for sign, i, j, v, width, labels, n_shared in peaks:
            key = "n_pos" if sign > 0 else "n_neg"
            clean = not labels
            res[key][0] += 1
            res[key][1] += int(clean)
            if clean and v >= 8:
                res[key][2] += 1
            if not [x for x in labels if x != "shared"]:
                fib = int(pm["FIBERID"][sky[i]]) if "FIBERID" in pm.names else int(sky[i]) + 1
                res["peaks"].append(dict(file=name, fiber=fib, ra=float(pm["RA"][sky[i]]), dec=float(pm["DEC"][sky[i]]),
                                         lam=float(lam[j]), snr=v, width=width, sign=sign, n_shared=n_shared,
                                         flux=float(sign * np.sum(d["flux"][i][j - 3:j + 4]) * lam[j] * 2.302585 * h0["COEFF1"])))
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump(res, open(out_p, "w"))
    return res


def skyfibre_clean(p, sign=1, shared_min=None):
    """Whether a stored peak is clean for the sign and sharing rule."""
    shared_min = skyfibre_SHARED_MIN if shared_min is None else shared_min
    return p.get("sign", 1) == sign and p.get("n_shared", 0) < shared_min


def skyfibre_inject(n_plates=40, mults=(1.0, 1.5, 2.0, 3.0, 5.0), seed=1, variants=((False, 1),)):
    """Completeness of injected instrumental-profile lines per detector variant, with losses by test."""
    rng = np.random.default_rng(seed)
    plates = skyfibre_plates()
    chosen = [plates[int(i)] for i in np.linspace(0, len(plates) - 1, n_plates)]
    k = np.arange(-4, 5)
    w = np.exp(-0.5 * (k / skyfibre_LSF_PX) ** 2)
    gain = math.sqrt(float(np.sum(w ** 2)))  # matched-filter S/N of a unit line
    tally = {(var, m): dict(n=0, found=0, lost={}, false=0, fibres=0) for var in variants for m in mults}
    for pl in chosen:
        d0 = skyfibre_get(skyfibre_BASE + "%s/" % pl).decode("utf-8", "replace")
        names = sorted(set(re.findall(r'href="(spPlate-%s-\d+\.fits)"' % pl, d0)))
        if not names:
            continue
        try:
            d = skyfibre_read(skyfibre_BASE + "%s/%s" % (pl, names[0]))
        except Exception as e:
            print("plate %s: %s" % (pl, str(e)[:100]), flush=True)
            continue
        if d is None:
            continue
        _, on_line, G = skyfibre_detect(d)
        npix = len(d["lam"])
        for m in mults:
            dm = dict(d, flux=d["flux"].copy())
            where = []
            for i in range(len(d["sky"])):
                ok = np.nonzero(G[i] & ~on_line)[0]
                ok = ok[(ok > 10) & (ok < npix - 10)]
                if len(ok) == 0:
                    continue
                j = int(rng.choice(ok))
                sig = 1.0 / math.sqrt(d["ivar"][i][j])
                dm["flux"][i][j - 4:j + 5] += m * skyfibre_SNR / gain * sig * w
                where.append((i, j))
            for var in variants:
                peaks, _, _ = skyfibre_detect(dm, common=var[0], shared_min=var[1])
                t = tally[(var, m)]
                injected = {(i, j) for i, j in where}
                t["false"] += sum(1 for p in peaks if p[0] > 0 and not p[5] and not any(p[1] == i and abs(p[2] - j) <= 1 for i, j in injected))
                t["fibres"] += len(d["sky"])
                for i, j in where:
                    t["n"] += 1
                    hit = [p for p in peaks if p[0] > 0 and p[1] == i and abs(p[2] - j) <= 1]
                    if hit and not hit[0][5]:
                        t["found"] += 1
                    else:
                        why = (hit[0][5][0].split()[0] if hit and hit[0][5] else "below threshold")
                        t["lost"][why] = t["lost"].get(why, 0) + 1
        print("plate %s: %s" % (pl, "; ".join("%s%d: %s" % ("c" if var[0] else "", var[1], ", ".join("%.1fx %d/%d" % (m, tally[(var, m)]["found"], tally[(var, m)]["n"]) for m in mults)) for var in variants)), flush=True)
    out = {"%s_%d_%.1f" % ("common" if var[0] else "plain", var[1], m): dict(t, recovered=t["found"] / max(t["n"], 1))
           for (var, m), t in tally.items()}
    json.dump(out, open(core.results("skyfibre_inject.json"), "w"), indent=1)
    for var in variants:
        for m in mults:
            t = tally[(var, m)]
            print("%s, shared at %d: %.1f times the threshold: %d of %d recovered (%.2f); lost to %s; other clean peaks %.2f per fibre"
                  % ("common mode removed" if var[0] else "plain", var[1], m, t["found"], t["n"], t["found"] / max(t["n"], 1),
                     ", ".join("%s %d" % kv for kv in sorted(t["lost"].items())), t["false"] / max(t["fibres"], 1)))
    return out


def skyfibre_plates():
    p = os.path.join(skyfibre_ROOT, "plates.json")
    if os.path.exists(p):
        return json.load(open(p))
    d = skyfibre_get(skyfibre_BASE).decode("utf-8", "replace")
    plates = sorted(set(re.findall(r'href="(\d{4,5})/"', d)), key=int)
    os.makedirs(skyfibre_ROOT, exist_ok=True)
    json.dump(plates, open(p, "w"))
    return plates


def skyfibre_search(threads, limit, part=None):
    plates = skyfibre_plates()
    if limit:
        plates = plates[:limit]
    if part:
        span, n = part.split("/")
        a, b = (int(x) for x in span.split(":"))
        plates = [p for i, p in enumerate(plates) if a <= i % int(n) < b]
    print("%d plates" % len(plates), flush=True)
    done = [0]

    def one(pl):
        try:
            r = skyfibre_plate(pl)
        except Exception as e:
            print("plate %s: %s" % (pl, str(e)[:120]), flush=True)
            return None
        done[0] += 1
        if done[0] % 25 == 0:
            print("%d plates searched" % done[0], flush=True)
        return r
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, plates))
    skyfibre_summary()


def skyfibre_summary():
    """Counts, clean peaks and dips across plates with the fibre-fixed test applied."""
    files = glob.glob(os.path.join(skyfibre_ROOT, "plates", "*.json"))
    R = [json.load(open(f)) for f in files]
    stored = [p for r in R for p in r["peaks"]]
    if all(r.get("shared_min", 1) == skyfibre_SHARED_MIN for r in R) or not any("sign" in p for p in stored):
        pos = np.sum([r["n_pos"] for r in R], axis=0)
        neg = np.sum([r["n_neg"] for r in R], axis=0)
    else:
        # counts under another sharing rule from the stored peaks
        pos = np.array([sum(r["n_pos"][0] for r in R), sum(1 for p in stored if skyfibre_clean(p, 1)),
                        sum(1 for p in stored if skyfibre_clean(p, 1) and p["snr"] >= 8)])
        neg = np.array([sum(r["n_neg"][0] for r in R), sum(1 for p in stored if skyfibre_clean(p, -1)),
                        sum(1 for p in stored if skyfibre_clean(p, -1) and p["snr"] >= 8)])
    print("%d plates, %d sky-fibre spectra; peaks at 6 sigma: emission-like %d (clean %d, clean at 8 sigma %d), "
          "absorption-like %d (clean %d, clean at 8 sigma %d)" % (len(R), sum(r["n_sky"] for r in R), pos[0], pos[1], pos[2],
                                                                neg[0], neg[1], neg[2]))
    pk = sorted((p for p in stored if skyfibre_clean(p, 1)), key=lambda p: -p["snr"])
    dips = sorted((p for p in stored if skyfibre_clean(p, -1)), key=lambda p: -p["snr"])
    for group in (pk, dips):
        for p in group:
            plate = p["file"].split("-")[1]
            p["fixed"] = sorted({q["file"].split("-")[1] for q in group if q["fiber"] == p["fiber"] and q["file"].split("-")[1] != plate
                                 and abs(math.log10(q["lam"] / p["lam"])) < 3e-4})
    for p in pk[:30]:
        print("  %s fibre %d (%.5f %+.5f): %.2f A, SNR %.1f, width %.2f px, flux %.2e erg/s/cm2"
              % (p["file"], p["fiber"], p["ra"], p["dec"], p["lam"], p["snr"], p["width"], p["flux"] * 1e-17))
    if dips:
        print("dips: %d clean, %d at 8 sigma, fixed to a fibre %d and %d" % (len(dips), sum(1 for p in dips if p["snr"] >= 8),
                                                                         sum(1 for p in dips if p["fixed"]),
                                                                         sum(1 for p in dips if p["fixed"] and p["snr"] >= 8)))
    json.dump(dict(n_plates=len(R), n_all=len(skyfibre_plates()), n_sky=int(sum(r["n_sky"] for r in R)), pos=pos.tolist(),
                   neg=neg.tolist(), shared_min=skyfibre_SHARED_MIN, peaks=pk, dips=dips),
              open(core.results("skyfibre_summary.json"), "w"), indent=0)


skyfibre_LINES = {"[OII]3727": 3728.48, "Hbeta": 4862.68, "[OIII]4959": 4960.29, "[OIII]5007": 5008.24,
                  "Halpha": 6564.61, "[NII]6583": 6585.27, "[SII]6716": 6718.29, "[SII]6731": 6732.67, "Lyalpha": 1215.67}  # vacuum


def skyfibre_redshift(lams, tol=4e-4):
    """Common galaxy redshift of two or more lines, or None."""
    names = list(skyfibre_LINES)
    for i in range(len(lams)):
        for j in range(i + 1, len(lams)):
            for a in names:
                for b in names:
                    if a == b:
                        continue
                    za, zb = lams[i] / skyfibre_LINES[a] - 1, lams[j] / skyfibre_LINES[b] - 1
                    if za > -2e-3 and abs(za - zb) < tol * (1 + za):  # Galactic nebulae within 600 km/s of rest
                        return "%s and %s at z %.4f" % (a, b, (za + zb) / 2)
    return None


def skyfibre_classify(peaks):
    """Each clean line with its galaxy identification and the plates where the same fibre holds it at another sky position."""
    by_spec, by_fibre = {}, {}
    for p in peaks:
        by_spec.setdefault((p["file"], p["fiber"]), []).append((p["lam"], p["flux"]))
        by_fibre.setdefault(p["fiber"], []).append(p)
    for p in peaks:
        found = skyfibre_galaxy(by_spec[(p["file"], p["fiber"])])
        p["galaxy"] = "%s at z %.4f" % (" and ".join(found[1]), found[0]) if found else None
        plate = p["file"].split("-")[1]
        p["fixed"] = sorted({q["file"].split("-")[1] for q in by_fibre[p["fiber"]] if q["file"].split("-")[1] != plate
                             and abs(math.log10(q["lam"] / p["lam"])) < 3e-4 and skyfibre_sep(p, q) > 1.5})
    return peaks


def skyfibre_sep(p, q):
    """Arcsec between two lines' fibre positions."""
    return math.hypot((q["ra"] - p["ra"]) * math.cos(math.radians(p["dec"])), q["dec"] - p["dec"]) * 3600.0


def skyfibre_both_signs(peaks, dips):
    """Mark lines whose spectrum also holds an unfixed clean line of the other sign."""
    have = {1: {(p["file"], p["fiber"]) for p in peaks if not p["fixed"]},
            -1: {(p["file"], p["fiber"]) for p in dips if not p["fixed"]}}
    for sign, group in ((1, peaks), (-1, dips)):
        for p in group:
            p["both"] = (p["file"], p["fiber"]) in have[-sign]
    return peaks, dips


def skyfibre_frames(lines, sign):
    """Each line's exposure verdict, None where unread."""
    path = core.results("skyfibre_exposures.json")
    ex = json.load(open(path)) if os.path.exists(path) else {}
    for p in lines:
        p["frames"] = ex.get("%s %d %.3f %+d" % (p["file"], p["fiber"], p["lam"], sign), {}).get("verdict")
    return bool(ex)


def skyfibre_counterparts(lines, cache):
    """(lines, covered, with a source at the fibre, with one 30 arcsec north)."""
    keys = ["%.5f %+.5f" % (p["ra"], p["dec"]) for p in lines]
    have = [k for k in keys if k in cache]
    return len(keys), len(have), sum(1 for k in have if cache[k]["on"]), sum(1 for k in have if cache[k]["off"])


def skyfibre_rest():
    """Clean 8-sigma peaks matched to Legacy Surveys at the fibre and a control, and to DESI, into skyfibre_rest.json."""
    d = json.load(open(core.results("skyfibre_summary.json")))
    peaks = skyfibre_classify(d["peaks"])
    rest = [p for p in peaks if p["snr"] >= 8 and not p["galaxy"] and not p["fixed"]]

    def match(p):
        r = 1.5 / 3600
        out = dict(p)
        for key, dec in (("ls", p["dec"]), ("ls_off", p["dec"] + 30.0 / 3600)):
            rows = core.datalab("SELECT ra, dec, type, mag_g, mag_r, mag_z FROM ls_dr10.tractor WHERE q3c_radial_query(ra, dec, %.6f, %.6f, %.6f) = 't'"
                                    % (p["ra"], dec, r))
            out[key] = [{k: x[k] for k in ("type", "mag_g", "mag_r", "mag_z")} for x in rows]
        de = core.datalab("SELECT z, zwarn, spectype FROM desi_dr1.zpix WHERE q3c_radial_query(mean_fiber_ra, mean_fiber_dec, %.6f, %.6f, %.6f) = 't'"
                              % (p["ra"], p["dec"], r))
        out["desi"] = [dict(z=num(x["z"]), zwarn=int(num(x["zwarn"])), spectype=x["spectype"]) for x in de]
        return out
    rest = core.pmap(match, rest, threads=6)
    withls, off = [p for p in rest if p["ls"]], [p for p in rest if p["ls_off"]]
    print("%d leftover lines at 8 sigma: a Legacy Surveys DR10 source within 1.5 arcsec at %d (median r %.2f), at %d of the positions 30 arcsec north; a DESI DR1 spectrum at %d"
          % (len(rest), len(withls), np.median([num(p["ls"][0]["mag_r"]) for p in withls if num(p["ls"][0]["mag_r"]) is not None]) if withls else 0.0, len(off),
             sum(1 for p in rest if p["desi"])))
    for p in sorted(rest, key=lambda p: -p["snr"])[:12]:
        print("  %s fibre %d (%.5f %+.5f): %.2f A, S/N %.1f, width %.2f px, flux %.2e erg/s/cm2; LS %s" % (
            p["file"], p["fiber"], p["ra"], p["dec"], p["lam"], p["snr"], p["width"], p["flux"] * 1e-17,
            ", ".join("%s r %s" % (x["type"], x["mag_r"]) for x in p["ls"]) or "none"))
    json.dump(dict(n_rest=len(rest), n_ls=len(withls), n_off=len(off), rest=rest), open(core.results("skyfibre_rest.json"), "w"), indent=1)


# readings of single positions: (RA, Dec, wavelength or None for every line, vet)
skyfibre_FOLLOWUP = (
    (213.5584, 53.6108, 4732.6, "background galaxy: Ly alpha at z 2.893, recurring at this position on 54 plate-nights "
                                "and alone in the stack of its 94 spectra, 287 km/s wide"),
    (133.6088, 8.8632, None, "planetary nebula Abell 31 (PN G219.1+31.2): its spectrum from [Ne III] 3869 to [S III] 9531 "
                             "at zero redshift fills the fibre"),
)


def skyfibre_record():
    """Record the plates, peaks against dips through the same tests, and every clean 8-sigma peak with its vet."""
    if os.path.isdir(os.path.join(skyfibre_ROOT, "plates")):
        skyfibre_summary()
    d = json.load(open(core.results("skyfibre_summary.json")))
    pos, neg = d["pos"], d["neg"]
    n_all = d.get("n_all") or len(skyfibre_plates())
    peaks = skyfibre_classify(d["peaks"])
    top = [p for p in peaks if p["snr"] >= 8]
    gal = sum(1 for p in top if p["galaxy"] and not p["fixed"])
    fixed = sum(1 for p in top if p["fixed"])
    dips = skyfibre_classify(d.get("dips", []))
    skyfibre_both_signs(peaks, dips)
    if dips:
        # dips through the fibre-fixed, redshift and two-sign tests too
        neg = [neg[0], sum(1 for p in dips if not p["fixed"] and not p["galaxy"] and not p["both"]),
               sum(1 for p in dips if not p["fixed"] and not p["galaxy"] and not p["both"] and p["snr"] >= 8)]
        print("dips fixed to a fibre: %d of %d clean, %d of %d at 8 sigma; at a galaxy redshift %d" % (
            sum(1 for p in dips if p["fixed"]), len(dips), sum(1 for p in dips if p["fixed"] and p["snr"] >= 8),
            sum(1 for p in dips if p["snr"] >= 8), sum(1 for p in dips if p["galaxy"] and not p["fixed"])))
    rest = [p for p in top if not p["galaxy"] and not p["fixed"]]
    rest_d = [p for p in dips if p["snr"] >= 8 and not p["galaxy"] and not p["fixed"]]
    both = (sum(1 for p in rest if p["both"]), sum(1 for p in rest_d if p["both"]))
    rest, rest_d = [p for p in rest if not p["both"]], [p for p in rest_d if not p["both"]]
    framed = skyfibre_frames(top, 1) and skyfibre_frames(rest_d, -1)
    cpath = core.results("skyfibre_counterparts.json")
    cp = ""

    def net(lp, ld, cache):
        # counterpart excess over control counted on both signs and scaled to all lines
        (n_p, h_p, on_p, off_p), (n_d, h_d, on_d, off_d) = skyfibre_counterparts(lp, cache), skyfibre_counterparts(ld, cache)
        g_p, g_d = (on_p - off_p) * n_p / max(h_p, 1), (on_d - off_d) * n_d / max(h_d, 1)
        sd = math.sqrt(n_p + n_d + (on_p + off_p) * (n_p / max(h_p, 1)) ** 2 + (on_d + off_d) * (n_d / max(h_d, 1)) ** 2)
        return (on_p, h_p, off_p, on_d, h_d, off_d, g_p, n_p - g_p, n_d - g_d, (n_p - g_p) - (n_d - g_d), sd)
    if os.path.exists(cpath) and dips:
        cache = json.load(open(cpath))
        cp = ("; %d peaks and %d dips sit in spectra that hold clean lines of both signs; a Legacy Surveys source "
              "within 1.5 arcsec at %d of the %d other peaks the catalogue covers against %d at positions 30 arcsec "
              "north, and at %d of %d dips against %d, so %.0f of the peaks are single lines of galaxies; net of them "
              "%.0f peaks against %.0f dips, an excess of %+.0f +- %.0f" % (both + net(rest, rest_d, cache)))
        if framed:
            v = collections.Counter(p["frames"] or "not read" for p in rest)
            vd = collections.Counter(p["frames"] or "not read" for p in rest_d)
            cp += ("; read exposure by exposure, the peaks hold %s and the dips %s, and of those in every exposure, "
                   "%d of %d peaks covered have a source against %d at the control and %d of %d dips against %d, so "
                   "%.0f peaks are galaxies' lines; net of them %.0f persistent peaks against %.0f dips, an excess of "
                   "%+.0f +- %.0f" % ((", ".join("%s %d" % kv for kv in v.most_common()), ", ".join("%s %d" % kv for kv in vd.most_common()))
                                      + net([p for p in rest if p["frames"] == "persistent"],
                                            [p for p in rest_d if p["frames"] == "persistent"], cache)))
    thr = np.array([6 * p["flux"] / p["snr"] for p in peaks]) * 1e-17
    inj = ""
    ip = core.results("skyfibre_inject.json")
    if os.path.exists(ip):
        r = json.load(open(ip))
        key = "plain_%d_" % skyfibre_SHARED_MIN
        got = {float(k[len(key):]): v["recovered"] for k, v in r.items() if k.startswith(key)}
        if got:
            inj = "; injected lines recovered as clean peaks: %s" % ", ".join("%.2f at %.1f times the threshold" % (got[m], m) for m in sorted(got))
    print("%d clean emission peaks (%d at 8 sigma): fibre-fixed %d (%d), galaxy lines %d (%d at 8 sigma), the rest %d (%d) "
          "against %d absorption-like at 8 sigma; 6 sigma line flux median %.1e erg/s/cm2 (10-90 per cent %.1e-%.1e)%s"
          % (len(peaks), len(top), sum(1 for p in peaks if p["fixed"]), fixed,
             sum(1 for p in peaks if p["galaxy"] and not p["fixed"]), gal,
             sum(1 for p in peaks if not p["galaxy"] and not p["fixed"] and not p["both"]), len(rest), neg[2], np.median(thr),
             *np.percentile(thr, [10, 90]), cp))
    sid = "sdss_sky_fibres" if skyfibre_SHARED_MIN == 1 else "sdss_sky_fibres_shared%d" % skyfibre_SHARED_MIN
    core.record_search(
        sid, "Lines in sky fibres" + ("" if skyfibre_SHARED_MIN == 1 else ", %d-fibre rule" % skyfibre_SHARED_MIN),
        archive="SDSS DR17 BOSS and eBOSS spPlate files (v5_13_2)",
        release="%d of %d plates" % (d["n_plates"], n_all), selection="every sky fibre of the plates searched (%d spectra)" % d["n_sky"],
        code=os.path.abspath(__file__), entry="skyfibre", n_targets=d["n_sky"], n_unit="sky-fibre spectra",
        outputs=[core.results(x) for x in ("skyfibre_summary.json", "skyfibre_inject.json", "skyfibre_counterparts.json")],
        params=dict(snr=skyfibre_SNR, lsf_px=skyfibre_LSF_PX, width=[0.6, 2.0], airglow=3.0, shared_sigma=3.0,
                    shared_min=skyfibre_SHARED_MIN, common=skyfibre_COMMON, fixed_px=3),
        null_summary="absorption-like peaks of the same shape put through the same tests, the test against other "
                     "plates, the redshift test and the two-sign test included: %d at 6 sigma, %d clean, %d clean at 8 sigma"
                     % tuple(neg),
        sensitivity="an unresolved line at 6 sigma of a sky fibre's noise, median %.1e erg/s/cm2 (10-90 per cent %.1e-%.1e)%s"
                    % (np.median(thr), *np.percentile(thr, [10, 90]), inj),
        ledger="Lines in sky fibres",
        result="emission-like peaks at 6 sigma %d, %d clean, %d clean at 8 sigma; of the clean 8 sigma peaks %d recur in "
               "the same fibre on other plates and %d are galaxy lines (two or more at one redshift), and %d remain "
               "against %d absorption-like put through the same tests%s"
               % (pos[0], pos[1], pos[2], fixed, gal, len(rest), neg[2], cp))
    core.clear_targets(sid)

    def vet(p):
        if p["fixed"]:
            return "fibre-fixed: the same fibre holds this line on plates %s" % ", ".join(p["fixed"])
        if p["galaxy"]:
            return "background galaxy: %s" % p["galaxy"]
        if p["both"]:
            return "a spectrum with clean lines of both signs"
        for ra, dec, lam, note in skyfibre_FOLLOWUP:
            if skyfibre_sep(p, dict(ra=ra, dec=dec)) < 1.5 and (lam is None or abs(p["lam"] - lam) < 2.0):
                return note
        return {"one exposure": "one exposure of the frames carries it: a cosmic ray",
                "one camera": "one camera of the two covering it carries it"}.get(p.get("frames"))
    core.record_targets(sid, [dict(
        target="sky:ra %.5f dec %+.5f" % (p["ra"], p["dec"]), stat=p["snr"], unit="S/N of an unresolved line",
        vet=vet(p), note="%s fibre %d, %.2f A, width %.2f px%s" % (
            p["file"], p["fiber"], p["lam"], p["width"], ", frames: %s" % p["frames"] if p.get("frames") else "")) for p in top])
    print("recorded %d plates, %d clean peaks at 8 sigma" % (d["n_plates"], len(top)))


skyfibre_STRONG = ("[OII]3727", "Hbeta", "[OIII]5007", "Halpha", "Lyalpha")


def skyfibre_lines(lams, tol=4e-4):
    """Redshift where most of a fibre's lines fall on galaxy lines (at least two, one strong), or None."""
    best = None
    names = list(skyfibre_LINES)
    for i in range(len(lams)):
        for j in range(i + 1, len(lams)):
            for a in names:
                for b in names:
                    if a == b:
                        continue
                    za, zb = lams[i] / skyfibre_LINES[a] - 1, lams[j] / skyfibre_LINES[b] - 1
                    if za <= -2e-3 or abs(za - zb) >= tol * (1 + za):
                        continue
                    z = (za + zb) / 2
                    hit = sorted({n for lam in lams for n in names if abs(lam / skyfibre_LINES[n] - 1 - z) < tol * (1 + z)})
                    if not any(n in skyfibre_STRONG for n in hit):
                        continue
                    if best is None or len(hit) > len(best[1]):
                        best = (z, hit)
    return best


skyfibre_RANGE = (3600.0, 10400.0)


def skyfibre_galaxy(lines):
    """skyfibre_lines with galaxy flux ratios required: [O III] 4959 with 5007 at 1.5-5, H alpha with a strong line."""
    lams = sorted(lam for lam, _ in lines)
    found = skyfibre_lines(lams)
    if not found:
        return None
    z, hit = found

    def flux(name):
        lam0 = skyfibre_LINES[name] * (1 + z)
        return max((f for lam, f in lines if abs(lam / lam0 - 1) < 4e-4), default=None)
    if "[OIII]4959" in hit:
        in_range = skyfibre_RANGE[0] < skyfibre_LINES["[OIII]5007"] * (1 + z) < skyfibre_RANGE[1]
        if "[OIII]5007" not in hit and in_range:
            return None
        if "[OIII]5007" in hit:
            r = flux("[OIII]5007") / max(flux("[OIII]4959"), 1e-9)
            if not 1.5 <= r <= 5.0:
                return None
    if "Halpha" in hit and not any(n in hit for n in ("[OII]3727", "Hbeta", "[OIII]5007", "Lyalpha")):
        return None
    return found


def skyfibre_galaxies():
    """Fibres at one redshift with Legacy Surveys and DESI matches, into skyfibre_galaxies.json."""
    d = json.load(open(core.results("skyfibre_summary.json")))
    by = {}
    for p in d["peaks"]:
        by.setdefault((p["file"], p["fiber"]), []).append(p)
    gals = []
    for (f, fib), ps in sorted(by.items()):
        found = skyfibre_galaxy([(p["lam"], p["flux"]) for p in ps])
        if not found:
            continue
        z, lines = found
        plate, mjd = re.match(r"spPlate-(\d+)-(\d+)\.fits", f).groups()
        o3 = [p for p in ps if abs(p["lam"] / skyfibre_LINES["[OIII]5007"] - 1 - z) < 4e-4 * (1 + z)]
        gals.append(dict(plate=int(plate), mjd=int(mjd), fiber=fib, ra=ps[0]["ra"], dec=ps[0]["dec"], z=z, lines=lines,
                         n_peaks=len(ps), snr=max(p["snr"] for p in ps), oiii_flux=o3[0]["flux"] * 1e-17 if o3 else None))

    def match(g):
        r = 1.5 / 3600
        ls = core.datalab("SELECT ra, dec, type, mag_g, mag_r, mag_z, shape_r FROM ls_dr10.tractor WHERE "
                              "q3c_radial_query(ra, dec, %.6f, %.6f, %.6f) = 't'" % (g["ra"], g["dec"], r))
        de = core.datalab("SELECT mean_fiber_ra, mean_fiber_dec, z, zwarn, spectype, zcat_primary FROM desi_dr1.zpix "
                              "WHERE q3c_radial_query(mean_fiber_ra, mean_fiber_dec, %.6f, %.6f, %.6f) = 't'"
                              % (g["ra"], g["dec"], r))
        cosd = math.cos(math.radians(g["dec"]))
        for x in ls:
            x["sep"] = math.hypot((float(x["ra"]) - g["ra"]) * cosd, float(x["dec"]) - g["dec"]) * 3600
        ls = sorted(ls, key=lambda x: x["sep"])
        de = [x for x in de if str(x.get("zcat_primary")).lower() in ("true", "t", "1")] or de
        return dict(g, ls=[{k: x[k] for k in ("type", "mag_g", "mag_r", "mag_z", "shape_r", "sep")} for x in ls],
                    desi=[dict(z=num(x["z"]), zwarn=int(num(x["zwarn"])), spectype=x["spectype"]) for x in de])
    gals = core.pmap(match, gals, threads=6)
    json.dump(dict(n_plates=d["n_plates"], n_sky=d["n_sky"], galaxies=gals), open(core.results("skyfibre_galaxies.json"), "w"),
              indent=1)
    zs = np.array([g["z"] for g in gals])
    withls = [g for g in gals if g["ls"]]
    agree = [g for g in gals if any(x["zwarn"] == 0 and abs(x["z"] - g["z"]) < 0.002 * (1 + g["z"]) for x in g["desi"])]
    print("%d galaxies in %d sky spectra of %d plates: z %.3f-%.3f (median %.3f); a Legacy Surveys DR10 source within 1.5 "
          "arcsec for %d, median r %.2f; a DESI DR1 redshift for %d, agreeing for %d"
          % (len(gals), d["n_sky"], d["n_plates"], zs.min(), zs.max(), np.median(zs), len(withls),
             np.median([num(g["ls"][0]["mag_r"]) for g in withls if num(g["ls"][0]["mag_r"]) is not None]),
             sum(1 for g in gals if g["desi"]), len(agree)))
    for g in sorted(gals, key=lambda g: g["z"]):
        ls = g["ls"][0] if g["ls"] else None
        print("  %d-%d-%d %.5f %+.5f z %.4f %s; S/N %.1f; LS %s; DESI %s"
              % (g["plate"], g["mjd"], g["fiber"], g["ra"], g["dec"], g["z"], "+".join(g["lines"]), g["snr"],
                 "%s r %s at %.1f arcsec" % (ls["type"], ls["mag_r"], ls["sep"]) if ls else "none",
                 ", ".join("z %.4f (%s, zwarn %d)" % (x["z"], x["spectype"], x["zwarn"]) for x in g["desi"]) or "none"))
    return gals


skyfibre_FRAME_SIG_PX = 1.2  # instrumental profile in calibrated-frame pixels


def skyfibre_frame_snr(flux, ivar, mask, lam, lam0):
    """(S/N, mask, covered) of an instrument-width line at lam0 in one frame; None outside it."""
    j = int(np.argmin(np.abs(lam - lam0)))
    if j < 40 or j > len(lam) - 41 or abs(lam[j] - lam0) > 2 * abs(lam[j + 1] - lam[j]):
        return None
    k = np.arange(-40, 41)
    f, iv, m = flux[j - 40:j + 41], ivar[j - 40:j + 41], mask[j - 40:j + 41]
    good = iv > 0
    side = good & (np.abs(k) > 5)
    cont = float(np.median(f[side])) if side.any() else 0.0
    best = (0.0, 0)
    for c in range(-3, 4):
        w = np.exp(-0.5 * ((k - c) / skyfibre_FRAME_SIG_PX) ** 2) * (np.abs(k - c) <= 4)
        den = math.sqrt(max(float(np.sum(w ** 2 * iv * good)), 1e-30))
        s = float(np.sum(w * (f - cont) * iv * good)) / den
        if abs(s) > abs(best[0]):
            best = (s, c)
    # covered when three or more pixels carry data; cosmic-ray frames keep only some
    covered = bool(np.sum(good[36 + best[1]:45 + best[1]]) >= 3)
    return round(best[0], 2), int(m[40 + best[1]]), covered


def skyfibre_line_frames(file, fiber, lam0, sign=1):
    """A sky-fibre line read in every covering spCFrame: [(camera, exposure, S/N, mask, covered)]."""
    plate = file.split("-")[1]
    h0 = skyfibre_hdus(skyfibre_BASE + "%s/%s" % (plate, file), 1)[0][0]
    exps = sorted({h0[k] for k in h0.keys() if k.startswith("EXPID")})
    spec, row = (1 if fiber <= 500 else 2), (fiber - 1) % 500
    out = []
    for e in exps:
        cam, expid = e.split("-")[:2]
        if not cam.endswith(str(spec)):
            continue
        url = skyfibre_BASE + "%s/spCFrame-%s-%s.fits" % (plate, cam, expid)
        hd = skyfibre_hdus(url, 4)
        npix = hd[0][0]["NAXIS1"]
        wl = skyfibre_rows(url, [(hd[3][1], [row])], npix * 8, ">f8")[0][row]
        lam = 10 ** wl if wl.max() < 10 else wl
        if not lam.min() + 20 < lam0 < lam.max() - 20:
            continue
        fl = skyfibre_rows(url, [(hd[0][1], [row])], npix * 8, ">f8")[0][row].astype(float)
        iv = skyfibre_rows(url, [(hd[1][1], [row])], npix * 4, ">f4")[0][row].astype(float)
        mk = skyfibre_rows(url, [(hd[2][1], [row])], npix * 4, ">i4")[0][row]
        m = skyfibre_frame_snr(sign * fl, iv, mk, lam, lam0)
        if m:
            out.append((cam, expid) + m)
    return out


def skyfibre_frames_verdict(frames):
    """Verdict of a line over its frames: one exposure, one camera, persistent, one frame, no frame or unclear."""
    cov = [f for f in frames if f[4]]
    if len(cov) < 2:
        return "one frame" if cov else "no frame"
    s = np.array([max(f[2], 0.0) for f in cov])
    if s.max() >= 5 and s.max() ** 2 / max(float(np.sum(s ** 2)), 1e-9) > 0.8:
        return "one exposure"
    cams = {f[0][0] for f in cov}
    if len(cams) == 2:
        sb = max(f[2] for f in cov if f[0][0] == "b")
        sr = max(f[2] for f in cov if f[0][0] == "r")
        if (sb >= 4 and sr < 2) or (sr >= 4 and sb < 2):
            return "one camera"
    if np.sum(s >= 2.5) >= 2:
        return "persistent"
    return "unclear"


def skyfibre_exposures(threads=4, snr_min=8.0):
    """Read every clean line at snr_min frame by frame into skyfibre_exposures.json."""
    d = json.load(open(core.results("skyfibre_summary.json")))
    path = core.results("skyfibre_exposures.json")
    done = json.load(open(path)) if os.path.exists(path) else {}
    todo = []
    for sign, group in ((1, skyfibre_classify(d["peaks"])), (-1, skyfibre_classify(d.get("dips", [])))):
        for p in group:
            if p["snr"] >= snr_min and not p["fixed"]:
                key = "%s %d %.3f %+d" % (p["file"], p["fiber"], p["lam"], sign)
                if key not in done:
                    todo.append((key, p, sign))
    print("%d lines to read, %d read before" % (len(todo), len(done)), flush=True)

    def one(item):
        key, p, sign = item
        try:
            fr = skyfibre_line_frames(p["file"], p["fiber"], p["lam"], sign)
        except Exception as e:
            print("  %s: %s" % (key, repr(e)[:120]), flush=True)
            return key, None
        return key, dict(frames=fr, verdict=skyfibre_frames_verdict(fr), snr=p["snr"], galaxy=bool(p["galaxy"]))
    n = 0
    with ThreadPoolExecutor(threads) as ex:
        for key, res in ex.map(one, todo):
            if res is not None:
                done[key] = res
            n += 1
            if n % 50 == 0 or n == len(todo):
                json.dump(done, open(path, "w"))
                print("%d of %d" % (n, len(todo)), flush=True)
    json.dump(done, open(path, "w"))
    for sign in (1, -1):
        c = collections.Counter(v["verdict"] for k, v in done.items() if k.endswith("%+d" % sign) and not v["galaxy"])
        g = collections.Counter(v["verdict"] for k, v in done.items() if k.endswith("%+d" % sign) and v["galaxy"])
        print("%s: %s; galaxy lines %s" % ("peaks" if sign > 0 else "dips", dict(c), dict(g)))


def skyfibre_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("search", "summary", "record", "galaxies", "rest", "inject", "exposures"))
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--part", default=None, help="a:b/n: the plates whose position in the list modulo n lies in [a, b)")
    a = p.parse_args()
    if a.mode == "search":
        skyfibre_search(a.threads, a.limit, a.part)
    elif a.mode == "record":
        skyfibre_record()
    elif a.mode == "galaxies":
        skyfibre_galaxies()
    elif a.mode == "rest":
        skyfibre_rest()
    elif a.mode == "inject":
        skyfibre_inject(n_plates=a.limit or 40, variants=((False, 1), (False, 2), (True, 1), (True, 2)))
    elif a.mode == "exposures":
        skyfibre_exposures(threads=min(a.threads, 4))
    else:
        skyfibre_summary()


telecom_DOC = r"""Telecom-band laser lines: the cached NIRPS and SPIRou spectra of the shoreline programme's M dwarfs, unresolved emission two to six samples wide above six local deviations in the O band (1260-1330 nm) and the C and L bands (1530-1625 nm), Paschen beta masked, recurring within 1.5 km/s in the star's frame on three or more nights whose barycentric corrections spread over 3 km/s, so a feature fixed at the telescope cannot recur, against the absorption-like dips the same test finds, the mirror; a recurring peak whose wavelengths scatter less in the telescope's frame than in the star's is sky emission the subtraction left, and one seen in more than one star sits on the stars' shared molecular pseudo-continuum rather than on one star.

    python -I seti.py lasers telecom [record] [--sig 6] [--tol-kms 1.5] [--min-nights 3] [--min-berv-spread 3] [--edge 30]"""

telecom_WINDOWS = ((12600.0, 13300.0), (15300.0, 16250.0))
telecom_MASK = ((12815.0, 12822.0),)  # Paschen beta in air, in emission in active M dwarfs


def telecom_features(w, f, sign, sig, edge):
    """Unresolved features of one sign above SIG local deviations in the windows: (wavelength, significance)."""
    from scipy.ndimage import median_filter
    out = []
    for lo, hi in telecom_WINDOWS:
        m = (w > lo) & (w < hi) & np.isfinite(f)
        if m.sum() < 500:
            continue
        ww, ff = w[m], f[m]
        r = sign * (ff / median_filter(ff, size=41, mode="nearest") - 1.0)
        n = len(r)
        sd = 1.4826 * median_filter(np.abs(r - median_filter(r, size=401, mode="nearest")), size=401, mode="nearest")
        sd = np.where(sd > 0.2 * np.nanmedian(sd), sd, np.inf)
        # zero-filled stretches repeat a value exactly
        z = np.where(np.r_[False, np.diff(ff) == 0], 0.0, r / sd)
        for i in np.nonzero((z > sig) & (z >= np.roll(z, 1)) & (z >= np.roll(z, -1)))[0]:
            if i < 3 or i > n - 4 or any(a < ww[i] < b for a, b in telecom_MASK) or not lo + edge < ww[i] < hi - edge:
                continue
            j0, j1 = i, i
            while j0 > 0 and z[j0 - 1] > z[i] / 2:
                j0 -= 1
            while j1 < n - 1 and z[j1 + 1] > z[i] / 2:
                j1 += 1
            # a line through the spectrograph spans two to six samples above half its height; one sample is a hit on the detector
            if 2 <= j1 - j0 + 1 <= 6:
                out.append((float(ww[i]), float(z[i])))
    return out


def telecom_main(record=False, sig=6.0, tol_kms=1.5, min_nights=3, min_berv_spread=3.0, edge=30.0):
    from astropy.io import fits
    from setitools import biosignatures, composition
    c = 299792.458
    root = os.path.join(biosignatures.shoreline_ROOT, "spectra")
    groups = collections.defaultdict(list)
    for inst in ("NIRPS", "SPIRou"):
        for p in sorted(glob.glob(os.path.join(root, inst, "*.fits"))):
            try:
                obj = str(fits.getheader(p).get("OBJECT", "")).strip().upper()
            except Exception:
                continue
            groups[(inst, re.sub(r"[ _-]", "", obj))].append(p)
    res, n_dips, n_spec = [], 0, sum(len(v) for v in groups.values())
    for (inst, obj), paths in sorted(groups.items()):
        if len(paths) < min_nights:
            continue
        loaded = []
        for p in paths:
            try:
                loaded.append(biosignatures.shoreline_load(p))
            except Exception:
                continue
        if not loaded:
            continue
        vsys = composition.companions_vsys(loaded)
        found = {+1: [], -1: []}
        for w, f, e, meta in loaded:
            rv = meta.get("rv") if meta.get("rv") is not None else vsys
            berv = meta.get("berv") or 0.0
            for sign in (+1, -1):
                for lam, z in telecom_features(np.asarray(w, float), np.asarray(f, float), sign, sig, edge):
                    found[sign].append(dict(rest=lam / (1.0 + rv / c), obs=lam / (1.0 + berv / c), z=z, night=math.floor(meta["mjd"] + 0.3), berv=berv))
        rec = {}
        for sign in (+1, -1):
            runs = []
            for x in sorted(found[sign], key=lambda x: x["rest"]):
                if runs and (x["rest"] - runs[-1][-1]["rest"]) / x["rest"] * c < tol_kms:
                    runs[-1].append(x)
                else:
                    runs.append([x])
            # scatter of the feature's wavelength in the star's frame and in the telescope's, in km/s
            rec[sign] = [dict(rest=float(np.median([x["rest"] for x in g])), nights=len({x["night"] for x in g}), z=max(x["z"] for x in g),
                              spread=max(x["berv"] for x in g) - min(x["berv"] for x in g),
                              sd_star=float(np.std([x["rest"] for x in g]) / g[0]["rest"] * c), sd_tel=float(np.std([x["obs"] for x in g]) / g[0]["obs"] * c))
                         for g in runs if len({x["night"] for x in g}) >= min_nights]
            rec[sign] = [r for r in rec[sign] if r["spread"] > min_berv_spread]
        for r in rec[+1]:
            res.append(dict(inst=inst, obj=obj, **r))
        n_dips += len(rec[-1])
        print("%-7s %-16s %3d spectra: recurring peaks %d, recurring dips %d" % (inst, obj[:16], len(paths), len(rec[+1]), len(rec[-1])), flush=True)
    # a peak at one rest wavelength in two or more stars belongs to the M dwarfs' shared spectrum
    # a peak that holds its wavelength better at the telescope than in the star is sky emission left by the subtraction
    for r in res:
        others = {(x["inst"], x["obj"]) for x in res if abs(x["rest"] - r["rest"]) / r["rest"] * c < 3 * tol_kms and x["obj"] != r["obj"]}
        r["vet"] = "atmospheric-line" if r["sd_tel"] <= r["sd_star"] else "shared-feature" if others else None
    left = [r for r in res if r["vet"] is None]
    print("%d spectra in %d groups; %d recurring peaks against %d recurring dips; %d steadier at the telescope than in the star, %d seen in more than one star, %d left" % (
        n_spec, len(groups), len(res), n_dips, sum(r["vet"] == "atmospheric-line" for r in res), sum(r["vet"] == "shared-feature" for r in res), len(left)))
    for r in left:
        print("  %s %s: %.3f A in the star's frame, %d nights, %.1f sigma, BERV spread %.1f km/s, scatter %.2f km/s in the star and %.2f at the telescope" % (
            r["inst"], r["obj"], r["rest"], r["nights"], r["z"], r["spread"], r["sd_star"], r["sd_tel"]))
    if record:
        # one-star peaks are the photometric tail when the dips, which no laser makes, outnumber them
        mirror = len(left) <= n_dips
        core.record_run(
            "telecom_band_lasers", "Telecom-band laser lines", "ESO Phase 3 (NIRPS) and the CADC (CFHT SPIRou): the shoreline programme's cached spectra",
            "%d spectra of %d instrument-star groups" % (n_spec, len(groups)), os.path.abspath(__file__), "telecom", len(groups), "instrument-star groups",
            dict(sig=sig, tol_kms=tol_kms, min_nights=min_nights, min_berv_spread=min_berv_spread, edge=edge, windows=telecom_WINDOWS),
            dict(peaks=len(res), dips=n_dips), "an unresolved feature above %g local deviations recurring on %d nights in the star's frame" % (sig, min_nights),
            "%d recurring peaks against %d recurring dips; %d steadier at the telescope than in the star, %d in more than one star; %d left" % (
                len(res), n_dips, sum(r["vet"] == "atmospheric-line" for r in res), sum(r["vet"] == "shared-feature" for r in res), len(left)),
            [dict(target="%s#%.3f" % (r["obj"], r["rest"]), stat=round(r["z"], 1), unit="sigma, the strongest night", vet=r["vet"] or ("chance-rate" if mirror else None),
                  note="%s; %d nights; BERV spread %.1f km/s; scatter %.2f km/s in the star, %.2f at the telescope" % (
                      r["inst"], r["nights"], r["spread"], r["sd_star"], r["sd_tel"])) for r in res])
        print("recorded")


APPROACHES = {
    "telecom": (lambda: core.run_cli(telecom_main), telecom_DOC),
    "skyfibre": (skyfibre_cli, skyfibre_DOC),
    "rvs": (rvs_cli, rvs_DOC),
    "eso": (eso_cli, eso_DOC),
    "sophie": (sophie_cli, sophie_DOC),
    "galah": (galah_cli, galah_DOC),
    "apogee": (apogee_cli, apogee_DOC),
    "desi": (desi_cli, desi_DOC),
    "lamost": (lamost_cli, lamost_DOC),
    "carmenes": (carmenes_cli, carmenes_DOC),
}


