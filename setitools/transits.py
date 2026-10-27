"""Transits and timing: occulter shapes, timing steps, nightside light and satellite belts."""
import argparse
import collections
import csv
import io
import json
import math
import os
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from setitools import core, science
from setitools.science import profile, shape_stats
from setitools.core import num, quiet_warnings


# --------------------------------------------------------------------------------------------------------------

shapes_kepler_DOC = r"""Occulter shapes in Kepler's not-transit-like KOIs (Arnold 2005): stable folded profiles no trapezoid fits, without a secondary eclipse or a pulsator's sinusoid.

    python -I seti.py transits shapes_kepler kois
    python -I seti.py transits shapes_kepler fetch [--threads 8] [--list SHARE.json]
    python -I seti.py transits shapes_kepler analyse [--plots 20] [--list SHARE.json] [--out kepler_shapes.json] [--procs N]"""

shapes_kepler_HERE = core.RESULTS
shapes_kepler_DL = os.environ.get("SETI_KEPLER", core.data("kepler"))
shapes_kepler_UA = core.UA


def shapes_kepler_get(url, timeout=120):
    """Bytes of a URL; None on 404 or after three failures."""
    return core.get(url, timeout=timeout, retries=3, soft=True, pause=0)


def shapes_kepler_kois():
    rows = core.exoplanets("SELECT kepid, kepoi_name, koi_period, koi_time0bk, koi_duration, koi_depth, koi_model_snr, "
                               "koi_steff, koi_srad, koi_kepmag FROM cumulative WHERE koi_fpflag_nt = 1 AND koi_fpflag_ss = 0 "
                               "AND koi_fpflag_co = 0 AND koi_fpflag_ec = 0")
    json.dump(rows, open(os.path.join(shapes_kepler_HERE, "kepler_ntl.json"), "w"))
    print("%d KOIs flagged only as not transit-like, on %d stars" % (len(rows), len({r["kepid"] for r in rows})))


def shapes_kepler_fetch_star(kepid):
    k9 = "%09d" % int(kepid)
    d = os.path.join(shapes_kepler_DL, k9)
    if os.path.isdir(d) and os.listdir(d):
        return len(os.listdir(d))
    lst = shapes_kepler_get("https://stpubdata.s3.amazonaws.com/?list-type=2&prefix=kepler/public/lightcurves/%s/%s/" % (k9[:4], k9))
    if not lst:
        return 0
    keys = re.findall(r"<Key>([^<]*_llc\.fits)</Key>", lst.decode())
    os.makedirs(d, exist_ok=True)
    n = 0
    for k in keys:
        b = shapes_kepler_get("https://stpubdata.s3.amazonaws.com/" + k)
        if b and b.startswith(b"SIMPLE"):
            open(os.path.join(d, k.split("/")[-1]), "wb").write(b)
            n += 1
    return n


def shapes_kepler_fetch(threads, share=None):
    rows = json.load(open(share or os.path.join(shapes_kepler_HERE, "kepler_ntl.json")))
    ids = sorted({r["kepid"] for r in rows})
    os.makedirs(shapes_kepler_DL, exist_ok=True)
    with ThreadPoolExecutor(threads) as ex:
        got = list(ex.map(shapes_kepler_fetch_star, ids))
    print("%d stars, %d with light curves, %d quarter files" % (len(ids), sum(1 for g in got if g), sum(got)))


def shapes_kepler_load(kepid):
    from astropy.io import fits
    d = os.path.join(shapes_kepler_DL, "%09d" % int(kepid))
    T, F = [], []
    for p in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        try:
            with fits.open(os.path.join(d, p)) as h:
                x = h[1].data
                t, f, q = np.asarray(x["TIME"], float), np.asarray(x["PDCSAP_FLUX"], float), np.asarray(x["SAP_QUALITY"], int)
        except Exception:
            continue
        ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
        if ok.sum() < 100:
            continue
        T.append(t[ok])
        F.append(f[ok] / np.median(f[ok]))
    if not T:
        return None
    return np.concatenate(T), np.concatenate(F)


def shapes_kepler_analyse_one(r):
    """Folded-profile statistics of one KOI, or None."""
    quiet_warnings()
    try:
        P, t0, dur = float(r["koi_period"]), float(r["koi_time0bk"]), float(r["koi_duration"]) / 24.0
    except (TypeError, ValueError):
        return None
    lc = shapes_kepler_load(r["kepid"])
    if lc is None or P <= 0 or dur <= 0:
        return None
    s = shape_stats(lc[0], lc[1], P, t0, dur, 0.0204)
    return dict(koi=r["kepoi_name"], kepid=r["kepid"], **s) if s else None


def shapes_kepler_analyse(nplots, share=None, outname="kepler_shapes.json", procs=1):
    rows = json.load(open(share or os.path.join(shapes_kepler_HERE, "kepler_ntl.json")))
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pool:
            out = [o for o in pool.map(shapes_kepler_analyse_one, rows, chunksize=2) if o]
    else:
        out = [o for o in map(shapes_kepler_analyse_one, rows) if o]
    json.dump(out, open(os.path.join(shapes_kepler_HERE, outname), "w"))
    good = [o for o in out if o["snr"] >= 15 and o["stable"] > 0.8 and o["odd_even"] > 0.8 and
            not (o["secondary"] > 0.3 * o["depth"])]
    print("%d KOIs folded; %d periodic and stable at S/N 15 or more with no secondary eclipse" % (len(out), len(good)))
    good.sort(key=lambda o: -o["red_chi"])
    for o in good[:25]:
        print("  %s P %.4f d dur %.1f h S/N %.0f: trapezoid reduced chi2 %.1f, asymmetry %+.2f, dips %d, halves r %.2f, "
              "odd/even r %.2f, ingress fraction %.2f" % (o["koi"], o["period"], o["dur_h"], o["snr"], o["red_chi"],
                                                          o["asym"], o["dips"], o["stable"], o["odd_even"], o["ingress_frac"]))
    if nplots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(shapes_kepler_HERE, "kepler_plots"), exist_ok=True)
        for o in good[:nplots]:
            r = next(x for x in rows if x["kepoi_name"] == o["koi"])
            t, f = shapes_kepler_load(r["kepid"])
            P, t0, dur = float(r["koi_period"]), float(r["koi_time0bk"]), float(r["koi_duration"]) / 24.0
            x, y, e = profile(t, f, P, t0, dur, nb=120)
            plt.figure(figsize=(6, 3))
            plt.errorbar(x * 24, y * 1e6, e * 1e6, fmt=".", ms=3)
            plt.xlabel("hours from mid-event")
            plt.ylabel("ppm")
            plt.title("%s  P %.4f d" % (o["koi"], P))
            plt.tight_layout()
            plt.savefig(os.path.join(shapes_kepler_HERE, "kepler_plots", o["koi"] + ".png"), dpi=90)
            plt.close()


def shapes_kepler_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("kois", "fetch", "analyse"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--plots", type=int, default=20)
    p.add_argument("--list", default=None, help="JSON list of KOI records (one share)")
    p.add_argument("--out", default="kepler_shapes.json")
    p.add_argument("--procs", type=int, default=1)
    a = p.parse_args()
    if a.mode == "kois":
        shapes_kepler_kois()
    elif a.mode == "fetch":
        shapes_kepler_fetch(a.threads, a.list)
    else:
        shapes_kepler_analyse(a.plots, a.list, a.out, a.procs)


def shapes_kepler_cli():
    quiet_warnings()
    shapes_kepler_main()


# --------------------------------------------------------------------------------------------------------------

shapes_tess_DOC = r"""Occulter shapes in TESS false-positive and false-alarm TOIs whose comments name no binary, blend or offset: the Kepler shape test on SPOC, TESS-SPOC or QLP folds.

    python -I seti.py transits shapes_tess fetch [--tmag 11.5]
    python -I seti.py transits shapes_tess ffi
    python -I seti.py transits shapes_tess analyse [--plots 20]"""

shapes_tess_HERE = core.RESULTS

shapes_tess_ROOT = os.environ.get("TOI_ROOT", core.data("tess_toi"))
shapes_tess_EXPLAINED = ("eb", "neb", "offset", "centroid", "secondary", "odd", "sb1", "sb2", " sb", "binary", "blend", "v-shape",
             "companion", "nearby")


def shapes_tess_rejects(tmag):
    rows = list(csv.DictReader(open(os.path.join(shapes_tess_ROOT, "toi.csv"), encoding="utf-8", errors="replace")))
    out = []
    for r in rows:
        if r["TFOPWG Disposition"] not in ("FP", "FA"):
            continue
        if any(k in (r["Comments"] or "").lower() for k in shapes_tess_EXPLAINED):
            continue
        try:
            P, t0, dur, tm = float(r["Period (days)"]), float(r["Epoch (BJD)"]), float(r["Duration (hours)"]), float(r["TESS Mag"])
        except ValueError:
            continue
        if P <= 0 or dur <= 0 or tm > tmag:
            continue
        sectors = [int(s) for s in re.findall(r"\d+", r["Sectors"] or "")]
        out.append(dict(toi=r["TOI"], tic=int(r["TIC ID"]), period=P, t0=t0 - 2457000.0, dur_d=dur / 24.0, tmag=tm,
                        depth_ppm=r["Depth (ppm)"], sectors=sectors, comment=r["Comments"], disp=r["TFOPWG Disposition"]))
    return out


def shapes_tess_fetch_one(job):
    tic, s = job
    d = os.path.join(shapes_tess_ROOT, "lc", str(tic))
    n = 0
    for k in core.spoc_keys(tic, s):
        p = os.path.join(d, os.path.basename(k))
        if os.path.exists(p):
            n += 1
            continue
        b = core.get(core.S3 + k, timeout=300, retries=3, soft=True, pause=2)
        if b and b.startswith(b"SIMPLE"):
            os.makedirs(d, exist_ok=True)
            open(p + ".tmp", "wb").write(b)
            os.replace(p + ".tmp", p)
            n += 1
    return n


def shapes_tess_fetch(tmag):
    rs = shapes_tess_rejects(tmag)
    json.dump(rs, open(os.path.join(shapes_tess_ROOT, "rejects.json"), "w"), indent=1)
    jobs = sorted({(r["tic"], s) for r in rs for s in r["sectors"]})
    print("%d rejected TOIs with no stated binary, blend or offset at Tmag %.1f or brighter; %d star-sectors to try"
          % (len(rs), tmag, len(jobs)), flush=True)
    with ThreadPoolExecutor(6) as ex:
        n = sum(ex.map(shapes_tess_fetch_one, jobs))
    print("%d two-minute light curves on disk" % n)


def shapes_tess_fetch_ffi_one(job):
    """One star-sector's FFI light curve: TESS-SPOC, else QLP."""
    tic, s = job
    d = os.path.join(shapes_tess_ROOT, "lc", str(tic))
    for _, u in core.ffi_urls(tic, s):
        p = os.path.join(d, os.path.basename(u))
        if os.path.exists(p):
            return 1
        b = core.get(u, timeout=120, retries=2, soft=True, pause=1)
        if b and b.startswith(b"SIMPLE"):
            os.makedirs(d, exist_ok=True)
            open(p + ".tmp", "wb").write(b)
            os.replace(p + ".tmp", p)
            return 1
    return 0


def shapes_tess_fetch_ffi():
    """FFI light curves for rejects without two-minute photometry."""
    rs = json.load(open(os.path.join(shapes_tess_ROOT, "rejects.json")))
    bare = [r for r in rs if not any(p.endswith("-s_lc.fits") for p in
                                     (os.listdir(os.path.join(shapes_tess_ROOT, "lc", str(r["tic"])))
                                      if os.path.isdir(os.path.join(shapes_tess_ROOT, "lc", str(r["tic"]))) else []))]
    jobs = sorted({(r["tic"], s) for r in bare for s in r["sectors"]})
    print("%d rejects without two-minute photometry; %d star-sectors to try" % (len(bare), len(jobs)), flush=True)
    with ThreadPoolExecutor(4) as ex:
        n = sum(ex.map(shapes_tess_fetch_ffi_one, jobs))
    print("%d full-frame-image light curves on disk" % n)


def shapes_tess_load(tic):
    """Normalised photometry from every sector (SPOC two-minute, else TESS-SPOC PDCSAP or QLP KSPSAP): times, fluxes, cadence."""
    from astropy.io import fits
    d = os.path.join(shapes_tess_ROOT, "lc", str(tic))
    names = sorted(os.listdir(d)) if os.path.isdir(d) else []
    if any(p.endswith("-s_lc.fits") for p in names):
        names = [p for p in names if p.endswith("-s_lc.fits")]
    T, F, C = [], [], []
    for p in names:
        try:
            with fits.open(os.path.join(d, p)) as h:
                x = h[1].data
                col = "KSPSAP_FLUX" if "hlsp_qlp" in p else "PDCSAP_FLUX"
                # QLP detrended flux: KSPSAP_FLUX early, DET_FLUX later
                col = next(c for c in (col, "DET_FLUX", "SAP_FLUX") if c in x.columns.names)
                t, f, q = np.asarray(x["TIME"], float), np.asarray(x[col], float), np.asarray(x["QUALITY"], int)
        except Exception:
            continue
        ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
        if ok.sum() < 500:
            continue
        T.append(t[ok])
        F.append(f[ok] / np.median(f[ok]))
        C.append(float(np.median(np.diff(t[ok]))))
    if not T:
        return None
    return np.concatenate(T), np.concatenate(F), max(C)


def shapes_tess_analyse_one(r):
    quiet_warnings()
    lc = shapes_tess_load(r["tic"])
    if lc is None:
        return None
    s = science.shape_stats(lc[0], lc[1], r["period"], r["t0"], r["dur_d"], lc[2])
    # events with three or more in-transit points; sparser events give sampling-driven shapes
    ep = np.round((lc[0] - r["t0"]) / r["period"])
    intr = np.abs(lc[0] - r["t0"] - ep * r["period"]) < r["dur_d"] / 2
    u, c = np.unique(ep[intr], return_counts=True)
    return dict(toi=r["toi"], tic=r["tic"], tmag=r["tmag"], comment=r["comment"], disp=r["disp"],
                cadence_min=round(lc[2] * 1440, 1), events=int((c >= 3).sum()), **s) if s else None


def shapes_tess_analyse(nplots):
    rs = json.load(open(os.path.join(shapes_tess_ROOT, "rejects.json")))
    from multiprocessing import Pool
    with Pool(int(os.environ.get("TOI_PROCS", "2"))) as pool:
        out = [o for o in pool.map(shapes_tess_analyse_one, rs, chunksize=2) if o]
    json.dump(out, open(os.path.join(shapes_tess_HERE, "tess_shapes.json"), "w"), indent=1)
    good = [o for o in out if o["snr"] >= 15 and o["stable"] > 0.8 and o["odd_even"] > 0.8 and
            not (o["secondary"] > 0.3 * o["depth"]) and o["events"] >= 3]
    print("%d rejected TOIs folded (%d from full-frame images); %d periodic and stable over three or more events at S/N 15 "
          "or more with no secondary eclipse" % (len(out), sum(o["cadence_min"] > 3 for o in out), len(good)))
    good.sort(key=lambda o: -o["red_chi"])
    for o in good[:30]:
        print("  TOI %s (TIC %d, T %.1f, %s, %g-min) P %.4f d dur %.1f h S/N %.0f: trapezoid reduced chi2 %.1f, asymmetry "
              "%+.2f, dips %d, halves r %.2f, odd/even r %.2f, ingress fraction %.2f; %s"
              % (o["toi"], o["tic"], o["tmag"], o["disp"], o["cadence_min"], o["period"], o["dur_h"], o["snr"], o["red_chi"],
                 o["asym"], o["dips"], o["stable"], o["odd_even"], o["ingress_frac"], (o["comment"] or "")[:80]))
    if nplots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(shapes_tess_HERE, "tess_plots"), exist_ok=True)
        for o in good[:nplots]:
            r = next(x for x in rs if x["toi"] == o["toi"])
            t, f, cad = shapes_tess_load(r["tic"])
            x, y, e = science.profile(t, f, r["period"], r["t0"], r["dur_d"], nb=120)
            plt.figure(figsize=(6, 3))
            plt.errorbar(x * 24, y * 1e6, e * 1e6, fmt=".", ms=3)
            plt.xlabel("hours from mid-event")
            plt.ylabel("ppm")
            plt.title("TOI %s  P %.4f d" % (o["toi"], r["period"]))
            plt.tight_layout()
            plt.savefig(os.path.join(shapes_tess_HERE, "tess_plots", "TOI%s.png" % o["toi"]), dpi=90)
            plt.close()


def shapes_tess_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "ffi", "analyse"))
    p.add_argument("--tmag", type=float, default=11.5)
    p.add_argument("--plots", type=int, default=20)
    a = p.parse_args()
    if a.mode == "fetch":
        shapes_tess_fetch(a.tmag)
    elif a.mode == "ffi":
        shapes_tess_fetch_ffi()
    else:
        shapes_tess_analyse(a.plots)


def shapes_tess_cli():
    shapes_tess_main()


# --------------------------------------------------------------------------------------------------------------

ttv_DOC = r"""Held phase steps in Kepler transit times (Holczer et al. 2016): steps beating the best sinusoid by 10 in chi-square at permutation FAP below 0.01, away from quarter boundaries.

    python -I seti.py transits ttv"""

ttv_HERE = core.RESULTS
ttv_MIN_N = 20
ttv_MIN_SIDE = 6
# Kepler quarter starts (BJD - 2454833, Q1-Q17), where rolls move targets to other CCDs
ttv_QUARTERS = (120.5, 169.5, 260.2, 352.4, 443.5, 539.4, 630.2, 735.4, 808.5, 906.8, 1001.2, 1099.4, 1182.0, 1274.1,
            1373.0, 1472.4, 1559.2, 1591.0)


def ttv_fit(A, y, w):
    p, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    return p, float(np.sum(w * (y - A @ p) ** 2))


def ttv_step_scan(n, y, w):
    best = (0.0, None, 0.0)
    base = np.vstack([np.ones_like(n), n]).T
    _, chi0 = ttv_fit(base, y, w)
    order = np.argsort(n)
    for k in range(ttv_MIN_SIDE, len(n) - ttv_MIN_SIDE + 1):
        h = (n >= n[order][k]).astype(float)
        p, chi = ttv_fit(np.vstack([np.ones_like(n), n, h]).T, y, w)
        if chi0 - chi > best[0]:
            best = (chi0 - chi, n[order][k], float(p[2]))
    return chi0, best


def ttv_sine_chi(n, y, w, per):
    from astropy.timeseries import LombScargle
    x = n.astype(float)
    if len(x) < 10:
        return None
    fr, pw = LombScargle(x, y, 1 / np.sqrt(w)).autopower(minimum_frequency=1 / (2 * np.ptp(x) + 1),
                                                         maximum_frequency=0.5, samples_per_peak=5)
    f = fr[int(np.argmax(pw))]
    A = np.vstack([np.ones_like(x), x, np.sin(2 * np.pi * f * x), np.cos(2 * np.pi * f * x)]).T
    return ttv_fit(A, y, w)[1], 1 / f


def ttv_main():
    cats = core.vizier("J/ApJS/225/9")
    t2 = {str(r["KOI"]): r for r in cats[0]}
    t3 = cats[1]
    by = {}
    for r in t3:
        flags = [r[c] for c in ("f_O-C", "Out", "Over")]
        if any(str(x).strip() not in ("", "--", "0", "None", "nan") for x in flags):
            continue
        try:
            oc, e = float(r["O-C"]), float(r["e_O-C"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(oc) and np.isfinite(e) and e > 0):
            continue
        by.setdefault(str(r["KOI"]), []).append((int(r["N"]), float(r["tn"]), oc, e))
    rng = np.random.default_rng(3)
    res = []
    for koi, rows in by.items():
        if len(rows) < ttv_MIN_N:
            continue
        a = np.array(rows)
        n, tn, y, e = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w = 1 / e ** 2
        chi0, (dchi, n0, step) = ttv_step_scan(n, y, w)
        sc = ttv_sine_chi(n, y, w, None)
        chi_sin = sc[0] if sc else chi0
        t_step = float(tn[np.argmin(np.abs(n - n0))]) if n0 is not None else None
        res.append(dict(koi=koi, kic=str(t2.get(koi, {}).get("KIC", "")) if koi in t2 else "", n=len(rows), chi0=chi0,
                        dchi_step=dchi, n0=n0, t_step=t_step, step_min=step, chi_step=chi0 - dchi, chi_sin=chi_sin,
                        sin_period=sc[1] if sc else None, dof=len(rows) - 3))
    # permutation FAPs for steps beating the sinusoid
    strong = [r for r in res if r["chi_sin"] - r["chi_step"] >= 10]
    for r in strong:
        a = np.array(by[r["koi"]])
        n, y, e = a[:, 0], a[:, 2], a[:, 3]
        w = 1 / e ** 2
        cnt = 0
        for _ in range(300):
            perm = rng.permutation(len(y))
            cnt += ttv_step_scan(n, y[perm], w[perm])[1][0] >= r["dchi_step"]
        r["fap"] = (cnt + 1) / 301.0
        r["quarter_edge"] = bool(r["t_step"] is not None and min(abs(r["t_step"] - q) for q in ttv_QUARTERS) < 2.0)
    json.dump(res, open(os.path.join(ttv_HERE, "kepler_ttv_steps.json"), "w"), indent=1)
    keep = [r for r in strong if r["fap"] < 0.01 and not r["quarter_edge"]]
    print("%d KOIs with %d or more good transit times; %d steps beat the best sinusoid by chi2 10; %d at FAP below 0.01, "
          "%d of them at a quarter boundary; %d left" % (len(res), ttv_MIN_N, len(strong),
                                                          sum(1 for r in strong if r["fap"] < 0.01),
                                                          sum(1 for r in strong if r["fap"] < 0.01 and r["quarter_edge"]),
                                                          len(keep)))
    for r in sorted(keep, key=lambda r: -(r["chi_sin"] - r["chi_step"]))[:30]:
        print("  KOI %s (KIC %s): %d transits, step %+.2f min at transit %d (BKJD %.1f), chi2 linear %.0f, step %.0f, "
              "sinusoid %.0f (P %.0f transits), FAP %.3f" % (r["koi"], r["kic"], r["n"], r["step_min"], r["n0"],
                                                               r["t_step"], r["chi0"], r["chi_step"], r["chi_sin"],
                                                               r["sin_period"] or 0, r["fap"]))


def ttv_cli():
    quiet_warnings()
    ttv_main()


impulse_DOC = r"""Impulsive period changes in Kepler transit times: two-segment kinks beating line, quadratic and sinusoid by evidence 10 and every sinusoid-plus-AR(1) null, at 5 sigma, away from quarter boundaries.

    python -I seti.py transits impulse analyse
    python -I seti.py transits impulse record"""

impulse_OUT = "impulsive_period_changes.json"
impulse_MIN_N, impulse_MIN_SIDE, impulse_DRAWS, impulse_EVIDENCE, impulse_SIGMA = 20, 6, 60, 10.0, 5.0


def impulse_kink_scan(n, y, w):
    """Two-segment continuous ephemeris over breaks with MIN_SIDE transits each side: chi-square, break, period change (min per transit) and error."""
    order = np.argsort(n)
    best = None
    for k in range(impulse_MIN_SIDE, len(n) - impulse_MIN_SIDE + 1):
        n0 = n[order][k]
        A = np.vstack([np.ones_like(n), n, np.clip(n - n0, 0, None)]).T
        Aw = A * np.sqrt(w)[:, None]
        p, *_ = np.linalg.lstsq(Aw, y * np.sqrt(w), rcond=None)
        chi = float(np.sum(w * (y - A @ p) ** 2))
        if best is None or chi < best[0]:
            cov = np.linalg.pinv(Aw.T @ Aw) * max(chi / max(len(n) - 3, 1), 1.0)
            best = (chi, float(n0), float(p[2]), float(math.sqrt(max(cov[2, 2], 0))))
    return best


def impulse_models(n, y, w):
    """Chi-square of line, quadratic and best sinusoid, with the sinusoid's frequency and amplitude."""
    x = n.astype(float)
    base = np.vstack([np.ones_like(x), x]).T
    _, chi_lin = ttv_fit(base, y, w)
    _, chi_quad = ttv_fit(np.vstack([np.ones_like(x), x, x ** 2]).T, y, w)
    from astropy.timeseries import LombScargle
    fr, pw = LombScargle(x, y, 1 / np.sqrt(w)).autopower(minimum_frequency=1 / (2 * np.ptp(x) + 1), maximum_frequency=0.5, samples_per_peak=5)
    f = float(fr[int(np.argmax(pw))])
    A = np.vstack([np.ones_like(x), x, np.sin(2 * np.pi * f * x), np.cos(2 * np.pi * f * x)]).T
    p, chi_sin = ttv_fit(A, y, w)
    return chi_lin, chi_quad, chi_sin, f, float(math.hypot(p[2], p[3]))


def impulse_analyse():
    cats = core.vizier("J/ApJS/225/9")
    t2 = {str(r["KOI"]): r for r in cats[0]}
    by = {}
    for r in cats[1]:
        if any(str(r[c]).strip() not in ("", "--", "0", "None", "nan") for c in ("f_O-C", "Out", "Over")):
            continue
        try:
            oc, e = float(r["O-C"]), float(r["e_O-C"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(oc) and np.isfinite(e) and e > 0:
            by.setdefault(str(r["KOI"]), []).append((int(r["N"]), float(r["tn"]), oc, e))
    rng = np.random.default_rng(9)
    res = []
    for koi, rows in by.items():
        if len(rows) < impulse_MIN_N:
            continue
        a = np.array(sorted(rows))
        n, tn, y, e = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w = 1 / e ** 2
        chi_lin, chi_quad, chi_sin, f, amp = impulse_models(n, y, w)
        chi_k, n0, dp, e_dp = impulse_kink_scan(n, y, w)
        lnN = math.log(len(n))
        ev_sin, ev_quad = (chi_sin - chi_k) - lnN, (chi_quad - chi_k) - lnN
        per = float(t2[koi]["Per"]) if koi in t2 and "Per" in t2[koi].colnames else None
        r = dict(koi=koi, kic=str(t2[koi]["KIC"]) if koi in t2 else "", n=len(n), period=per, chi_lin=chi_lin, chi_quad=chi_quad, chi_sin=chi_sin, sin_period=1 / f, sin_amp=amp,
                 chi_kink=chi_k, n0=n0, t_break=float(tn[np.argmin(np.abs(n - n0))]), dP_min=dp, e_dP_min=e_dp, sigma=abs(dp) / e_dp if e_dp > 0 else 0.0,
                 evidence_sin=ev_sin, evidence_quad=ev_quad, dof=len(n) - 3)
        r["quarter_edge"] = min(abs(r["t_break"] - q) for q in ttv_QUARTERS) < 2.0
        res.append(r)
    # null for kinks beating both smooth models: best sinusoid at random phase plus AR(1) noise
    cand = [r for r in res if r["evidence_sin"] >= impulse_EVIDENCE and r["evidence_quad"] >= impulse_EVIDENCE and r["sigma"] >= impulse_SIGMA]
    for r in cand:
        a = np.array(sorted(by[r["koi"]]))
        n, y, e = a[:, 0], a[:, 2], a[:, 3]
        w = 1 / e ** 2
        x = n.astype(float)
        resid = y - np.polyval(np.polyfit(x, y, 1, w=np.sqrt(w)), x)
        s = float(np.sqrt(np.mean(resid ** 2)))
        rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if len(resid) > 3 else 0.0
        rho = min(max(rho, 0.0), 0.95)
        f = 1 / r["sin_period"]
        draws = []
        for _ in range(impulse_DRAWS):
            ph = rng.uniform(0, 2 * np.pi)
            noise = np.zeros(len(x))
            noise[0] = rng.normal(0, s)
            for i in range(1, len(x)):
                noise[i] = rho * noise[i - 1] + rng.normal(0, s * math.sqrt(1 - rho ** 2))
            ys = r["sin_amp"] * np.sin(2 * np.pi * f * x + ph) + noise
            _, cq, cs, _, _ = impulse_models(n, ys, w)
            ck = impulse_kink_scan(n, ys, w)[0]
            draws.append(min(cs - ck, cq - ck) - math.log(len(n)))
        r["null_max"] = float(max(draws))
        r["null_median"] = float(np.median(draws))
        r["fap"] = float(np.mean(np.array(draws) >= min(r["evidence_sin"], r["evidence_quad"])))
        # leave-one-out evidence: a kink carried by one transit is that transit's error
        loo = []
        for i in range(len(x)):
            m = np.ones(len(x), bool)
            m[i] = False
            _, cq, cs, _, _ = impulse_models(n[m], y[m], w[m])
            ck = impulse_kink_scan(n[m], y[m], w[m])[0]
            loo.append(min(cs - ck, cq - ck) - math.log(int(m.sum())))
        r["evidence_loo"] = float(min(loo))
        r["loo_point"] = int(n[int(np.argmin(loo))])
    keep = [r for r in cand if r["fap"] == 0.0 and not r["quarter_edge"] and r["evidence_loo"] >= impulse_EVIDENCE]
    print("%d KOIs with %d or more good transit times; %d kinks with evidence above %.0f over both the sinusoid and the quadratic at %.0f sigma; %d beat every null draw, %d of them away from a quarter boundary and "
          "above %.0f with any one transit left out" % (len(res), impulse_MIN_N, len(cand), impulse_EVIDENCE, impulse_SIGMA, sum(1 for r in cand if r["fap"] == 0.0), len(keep), impulse_EVIDENCE))
    for r in sorted(cand, key=lambda r: -min(r["evidence_sin"], r["evidence_quad"]))[:25]:
        print("  KOI %s (KIC %s): %d transits, P %s d, period change %+.3f +- %.3f min at transit %.0f (BKJD %.1f%s), evidence over sinusoid %.1f (P %.0f transits, amplitude %.1f min), over quadratic %.1f; "
              "null max %.1f, FAP %.3f; with transit %d left out %.1f" % (
                  r["koi"], r["kic"], r["n"], "%.2f" % r["period"] if r["period"] else "-", r["dP_min"], r["e_dP_min"], r["n0"], r["t_break"], ", quarter edge" if r["quarter_edge"] else "",
                  r["evidence_sin"], r["sin_period"], r["sin_amp"], r["evidence_quad"], r["null_max"], r["fap"], r["loo_point"], r["evidence_loo"]))
    json.dump(dict(n=len(res), n_cand=len(cand), n_keep=len(keep), rows=res, params=dict(min_n=impulse_MIN_N, min_side=impulse_MIN_SIDE, draws=impulse_DRAWS, evidence=impulse_EVIDENCE, sigma=impulse_SIGMA)),
              open(core.results(impulse_OUT), "w"), indent=0)


def impulse_record():
    d = json.load(open(core.results(impulse_OUT)))
    cand = [r for r in d["rows"] if "fap" in r]
    keep = [r for r in cand if r["fap"] == 0.0 and not r["quarter_edge"] and r.get("evidence_loo", 0) >= d["params"]["evidence"]]
    core.record_search(
        "impulsive_period_changes", "Impulsive period changes", archive="Holczer et al. (2016) Kepler transit times (J/ApJS/225/9)",
        selection="KOIs with %d or more good transit times (%d)" % (d["params"]["min_n"], d["n"]), code=os.path.abspath(__file__), entry="impulse", n_targets=d["n"], params=d["params"],
        null_summary="each candidate's best sinusoid at a random phase plus AR(1) noise of its own residual scatter, %d draws, scanned for a kink the same way" % d["params"]["draws"],
        sensitivity="a period change at 5 sigma of the fit, about 1 min per transit for a typical series; 2 s on a 10 d planet is 0.07 m/s",
        ledger="Impulsive period changes",
        result="%d kinks with evidence above %.0f over both smooth models at %.0f sigma; %d beat every null draw, lie away from a quarter boundary and keep their evidence with any one transit left out%s%s" % (
            d["n_cand"], d["params"]["evidence"], d["params"]["sigma"], len(keep),
            (": " + "; ".join("KOI %s (%+.2f min at BKJD %.0f, P_sin %.0f transits)" % (r["koi"], r["dP_min"], r["t_break"], r["sin_period"]) for r in keep[:8])) if keep else "",
            ("; the kinks one transit carries: " + "; ".join("KOI %s (evidence %.0f, %.0f without transit %d)" % (r["koi"], min(r["evidence_sin"], r["evidence_quad"]), r["evidence_loo"], r["loo_point"])
                                                           for r in cand if r["fap"] == 0.0 and r not in keep)) if any(r["fap"] == 0.0 and r not in keep for r in cand) else ""))
    core.clear_targets("impulsive_period_changes")
    core.record_targets("impulsive_period_changes", [dict(target="koi:K%s" % r["koi"].zfill(8) if "." in r["koi"] else "koi:%s" % r["koi"], stat=min(r["evidence_sin"], r["evidence_quad"]),
                                                              p=r.get("fap"), unit="kink evidence over the sinusoid and the quadratic",
                                                              vet=("quarter boundary" if r["quarter_edge"] else ("beats every null draw" + ("" if r in keep else ", one transit carries it") if r.get("fap") == 0.0 else "within the null")) if "fap" in r else None,
                                                              note="%d transits, period change %+.3f +- %.3f min at transit %.0f" % (r["n"], r["dP_min"], r["e_dP_min"], r["n0"])) for r in d["rows"]])
    print("recorded %d KOIs" % d["n"])


def impulse_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": impulse_analyse, "record": impulse_record}[mode]()


# --------------------------------------------------------------------------------------------------------------

cepheids_DOC = r"""Phase steps in LMC Cepheid times of maximum (Poleski 2008; Learned et al. 2008): held steps beating a light-travel sinusoid, against random walks of each star's scatter.

    python -I seti.py transits cepheids analyse
    python -I seti.py transits cepheids record"""

cepheids_ROOT = os.environ.get("CEPHEID_ROOT", core.data("cepheid"))
cepheids_HERE = core.RESULTS
cepheids_MIN_SIDE = 4
cepheids_MAX_STEP = 0.3
cepheids_SOURCES = ("O", "R", "V")
# steps within this many days of a source's first or last maximum are source offsets (OGLE-II and -III share a code)
cepheids_BOUND_DAYS = 30.0
# step dates shared at three times the uniform rate within 15 days are instrumental
cepheids_SHARED = 3.0


def cepheids_stars():
    out = []
    for line in open(os.path.join(cepheids_ROOT, "poleski_list.dat")):
        f = line.split()
        if len(f) < 4:
            continue
        out.append(dict(id=f[0], p=float(f[1]), ep=float(f[2]), mode=f[3], remarks=f[4] if len(f) > 4 else ""))
    return out


def cepheids_maxima(sid):
    path = os.path.join(cepheids_ROOT, "max", sid + ".max")
    if not os.path.exists(path):
        return None
    a = [ln.split() for ln in open(path) if ln.strip()]
    t = np.array([float(x[0]) for x in a])
    e = np.array([float(x[1]) for x in a])
    s = np.array([x[2] for x in a])
    o = np.argsort(t)
    return t[o], e[o], s[o]


def cepheids_design(t, s, t0, p, extra=None):
    """Design columns: per-source offsets (O as reference), linear and quadratic in cycle number."""
    n = np.round((t - t0) / p)
    cols = [np.ones_like(t), n, n * n]
    for src in cepheids_SOURCES[1:]:
        if np.any(s == src):
            cols.append((s == src).astype(float))
    if extra is not None:
        cols += extra
    return np.vstack(cols).T, n


def cepheids_wls(a, y, w):
    aw = a * w[:, None]
    coef, *_ = np.linalg.lstsq(aw, y * w, rcond=None)
    r = y - a @ coef
    return coef, float(np.sum((r * w) ** 2)), r


def cepheids_analyse_star(st, rng):
    m = cepheids_maxima(st["id"])
    if m is None or len(m[0]) < 2 * cepheids_MIN_SIDE + 2:
        return None
    t, e, s = m
    e = np.maximum(e, 1e-4)
    w = 1.0 / e
    p = st["p"]
    t0 = t[0]
    # O-C against the catalogue period, then the natural model
    a, n = cepheids_design(t, s, t0, p)
    oc = t - (t0 + n * p)
    _, chi0, r0 = cepheids_wls(a, oc, w)
    dof0 = len(t) - a.shape[1]
    best = (0.0, None, None)
    for k in range(cepheids_MIN_SIDE, len(t) - cepheids_MIN_SIDE + 1):
        if t[k] - t[k - 1] < 1e-6:
            continue
        step = (t >= t[k]).astype(float)
        a2, _ = cepheids_design(t, s, t0, p, [step])
        coef, chi, _ = cepheids_wls(a2, oc, w)
        gain = chi0 - chi
        if gain > best[0]:
            best = (gain, float(t[k]), float(coef[-1]))
    span = (t[-1] - t[0]) / 365.25
    bounds = [float(x) for src in set(s) for x in (t[s == src].min(), t[s == src].max())]
    near_bound = best[1] is not None and min(abs(best[1] - b) for b in bounds) <= cepheids_BOUND_DAYS
    sin_best = 0.0
    for porb in np.exp(np.linspace(np.log(1.0), np.log(30.0), 120)) * 365.25:
        ph = 2 * np.pi * t / porb
        a3, _ = cepheids_design(t, s, t0, p, [np.sin(ph), np.cos(ph)])
        _, chi, _ = cepheids_wls(a3, oc, w)
        sin_best = max(sin_best, chi0 - chi)
    # null: random walk with the unexplained residual variance plus white noise at the errors
    rms = float(np.sqrt(np.mean(r0 ** 2)))
    red = math.sqrt(max(rms ** 2 - float(np.mean(e ** 2)), 0.0))
    sims = []
    for _ in range(20):
        rw = np.cumsum(rng.normal(0, 1, len(t)))
        rw = (rw - rw.mean()) / max(rw.std(), 1e-9) * red + rng.normal(0, 1, len(t)) * e
        _, chis, _ = cepheids_wls(a, rw, w)
        g = 0.0
        for k in range(cepheids_MIN_SIDE, len(t) - cepheids_MIN_SIDE + 1):
            step = (t >= t[k]).astype(float)
            a2, _ = cepheids_design(t, s, t0, p, [step])
            _, chi, _ = cepheids_wls(a2, rw, w)
            g = max(g, chis - chi)
        sims.append(g / max(chis / max(dof0, 1), 1e-9))
    return dict(id=st["id"], p=p, mode=st["mode"], remarks=st["remarks"], n=len(t), span_yr=span,
                chi2_dof=chi0 / max(dof0, 1), step_gain=best[0], step_t=best[1], step_d=best[2],
                step_frac=abs(best[2]) / p if best[2] is not None else None, sin_gain=sin_best,
                step_rel=best[0] / max(chi0 / max(dof0, 1), 1e-9), rw_rel=float(np.median(sims)),
                rw_rel_max=float(np.max(sims)), rms_d=rms, near_bound=bool(near_bound))


def cepheids_analyse():
    from concurrent.futures import ProcessPoolExecutor
    ss = cepheids_stars()
    with ProcessPoolExecutor(4) as ex:
        res = [r for r in ex.map(cepheids_one, ss, chunksize=20) if r]
    rel = np.array([r["step_rel"] for r in res])
    rw = np.array([r["rw_rel"] for r in res])
    print("%d Cepheids with at least %d maxima; median %d maxima over %.1f years; median chi2/dof of the natural model %.2f"
          % (len(res), 2 * cepheids_MIN_SIDE + 2, np.median([r["n"] for r in res]), np.median([r["span_yr"] for r in res]),
             np.median([r["chi2_dof"] for r in res])))
    print("step improvement in units of the residual variance: median %.1f, 99th percentile %.1f; the same for a random "
          "walk of each star's power: median %.1f, 99th percentile %.1f" % (np.median(rel), np.percentile(rel, 99),
                                                                           np.median(rw), np.percentile(rw, 99)))
    strong, keep = cepheids_select(res)
    print("steps beating a companion orbit twice over, every random-walk trial of the star, and 25 times the residual "
          "variance, with size below %.1f periods: %d; away from a data source's first or last maximum and from step "
          "dates shared by other stars: %d" % (cepheids_MAX_STEP, len(strong), len(keep)))
    for r in strong:
        if r not in keep:
            print("  %s step at HJD %.1f: %s" % (r["id"], r["step_t"], r["why"]))
    for r in sorted(keep, key=lambda r: -r["step_rel"])[:30]:
        print("  %s %-2s P %.4f d, %d maxima over %.1f yr, chi2/dof %.1f: step of %+.4f d (%.3f P) at HJD %.1f, gain %.0f "
              "(%.0f residual variances; random walk up to %.0f), orbit gain %.0f%s"
              % (r["id"], r["mode"], r["p"], r["n"], r["span_yr"], r["chi2_dof"], r["step_d"], r["step_frac"], r["step_t"],
                 r["step_gain"], r["step_rel"], r["rw_rel_max"], r["sin_gain"], "; remarks " + r["remarks"] if r["remarks"] else ""))
    json.dump(res, open(os.path.join(cepheids_HERE, "cepheid_oc.json"), "w"), indent=1)


def cepheids_select(res):
    """Strong steps, and those surviving the source-boundary and shared-date filters, with reasons in 'why'."""
    sig = np.array([r["step_t"] for r in res if r["step_t"] and r["step_rel"] > r["rw_rel_max"]
                    and r["step_frac"] < cepheids_MAX_STEP])
    rate = (len(sig) - 1) * 30.0 / max(np.ptp(sig), 1.0)
    strong = [r for r in res if r["step_frac"] is not None and r["step_frac"] < cepheids_MAX_STEP
              and r["step_gain"] > 2 * r["sin_gain"] and r["step_rel"] > max(r["rw_rel_max"], 25.0)]
    keep = []
    for r in strong:
        why = []
        if r["near_bound"]:
            why.append("within %d days of a data source's first or last maximum" % cepheids_BOUND_DAYS)
        n = int(np.sum(np.abs(sig - r["step_t"]) <= 15)) - 1
        if n >= cepheids_SHARED * rate:
            why.append("%d other stars step within 15 days (%.1f expected)" % (n, rate))
        r["why"] = "; ".join(why)
        if not why:
            keep.append(r)
    return strong, keep


def cepheids_one(st):
    rng = np.random.default_rng(abs(hash(st["id"])) % (2 ** 32))
    try:
        return cepheids_analyse_star(st, rng)
    except Exception:
        return None


def cepheids_record():
    res = json.load(open(os.path.join(cepheids_HERE, "cepheid_oc.json")))
    strong, keep = cepheids_select(res)
    why = {r["id"]: r["why"] for r in strong}
    rows = [dict(target="ogle:" + r["id"], stat=r["step_rel"], note="step %+.4f d at HJD %.1f; random walk up to %.0f; "
                 "orbit gain %.0f" % (r["step_d"] or 0, r["step_t"] or 0, r["rw_rel_max"], r["sin_gain"]),
                 vet=why.get(r["id"]) or None) for r in res]
    core.record_search("cepheid_phase_steps", "Cepheid phase coding", archive="OGLE-III OCVS (Poleski 2008 times of maximum)",
                           release="MACHO 1992-1999, OGLE 1997-2008", selection="LMC Cepheids with times of maximum",
                           code=os.path.abspath(__file__), entry="cepheids", params=dict(min_side=cepheids_MIN_SIDE, max_step=cepheids_MAX_STEP),
                           n_targets=len(rows), result="%d steps beyond the companion and random-walk models" % len(keep),
                           ledger="Cepheid phase coding")
    core.record_targets("cepheid_phase_steps", rows)
    print("recorded %d Cepheids" % len(rows))


def cepheids_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("analyse", "record"))
    a = p.parse_args()
    {"analyse": cepheids_analyse, "record": cepheids_record}[a.mode]()


def cepheids_cli():
    cepheids_main()


# --------------------------------------------------------------------------------------------------------------

nightside_DOC = r"""Nightside light of TESS ultra-short-period rocky planets (P below 1 d, R below 2 R_earth): phase-curve flux at phase 0 less the eclipse mean.

    python -I seti.py transits nightside targets
    python -I seti.py transits nightside stream --list TARGETS.json --out phase_curves_<part>.jsonl [--procs N]
    python -I seti.py transits nightside report phase_curves_*.jsonl
    python -I seti.py transits nightside record phase_curves_*.jsonl"""

nightside_ROOT = os.environ.get("PHASE_CURVES_ROOT", core.data("phase_curves"))
nightside_HERE = core.RESULTS
nightside_TOI_CSV = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
nightside_BTJD = 2457000.0
nightside_CAD = 120.0 / 86400.0


def nightside_targets():
    os.makedirs(nightside_ROOT, exist_ok=True)
    raw = core.get(nightside_TOI_CSV, timeout=300, retries=3).decode("utf-8", "replace")
    by = {}
    for r in csv.DictReader(io.StringIO(raw)):
        if (r.get("TFOPWG Disposition") or "").strip() in ("FP", "FA"):
            continue
        P, ep, dur = num(r.get("Period (days)")), num(r.get("Epoch (BJD)")), num(r.get("Duration (hours)"))
        rad = num(r.get("Planet Radius (R_Earth)"))
        if not P or not ep or not dur or not rad or P >= 1.0 or rad >= 2.0:
            continue
        tic = int(r["TIC ID"])
        x = by.setdefault(tic, dict(tic=tic, tmag=num(r.get("TESS Mag")), sectors=set(), tois=[]))
        x["sectors"] |= {int(s) for s in re.findall(r"\d+", r.get("Sectors") or "") if int(s) < 200}
        x["tois"].append(dict(toi=r["TOI"], period=P, t0=ep - nightside_BTJD, dur_h=dur, radius=rad,
                              teq=num(r.get("Planet Equil Temp (K)")), disp=r.get("TFOPWG Disposition")))
    out = [dict(x, sectors=sorted(x["sectors"])) for x in by.values() if x["sectors"]]
    json.dump(out, open(os.path.join(nightside_ROOT, "targets.json"), "w"))
    print("%d hosts of %d ultra-short-period rocky TOIs, %d star-sectors" % (len(out), sum(len(x["tois"]) for x in out),
                                                                             sum(len(x["sectors"]) for x in out)))


def nightside_running_median(t, f, width):
    import pandas as pd
    s = pd.Series(f, index=pd.to_datetime(t, unit="D"))
    return s.rolling("%dmin" % int(width * 1440), center=True, min_periods=10).median().to_numpy()


def nightside_measure(t, f, toi):
    P, t0, half = toi["period"], toi["t0"], toi["dur_h"] / 48.0
    ph = ((t - t0) / P) % 1.0
    w = half / P + 0.01
    tr = (ph < w) | (ph > 1 - w)
    ec = np.abs(ph - 0.5) < half / P
    oot = ~tr & ~ec
    if ec.sum() < 50 or oot.sum() < 500:
        return None
    x = 2 * np.pi * ph[oot]
    A = np.vstack([np.ones(oot.sum()), np.cos(x), np.sin(x), np.cos(2 * x)]).T
    coef, *_ = np.linalg.lstsq(A, f[oot], rcond=None)
    res = f[oot] - A @ coef
    # red-noise-aware scatter: 30-minute bins of the residual, scaled back to one point
    nb = 15
    k = len(res) // nb
    bscat = np.std(res[:k * nb].reshape(k, nb).mean(axis=1)) if k > 10 else np.std(res) / math.sqrt(nb)
    sig_pt = max(np.std(res), bscat * math.sqrt(nb))
    cov = np.linalg.inv(A.T @ A) * sig_pt ** 2
    E = float(np.mean(f[ec]))
    eE = sig_pt / math.sqrt(ec.sum())
    v0 = np.array([1, 1, 0, 1.0])
    v5 = np.array([1, -1, 0, 1.0])
    night = float(v0 @ coef - E)
    day = float(v5 @ coef - E)
    return dict(toi=toi["toi"], period=P, radius=toi["radius"], teq=toi.get("teq"), n_eclipse=int(ec.sum()),
                n_oot=int(oot.sum()), night_ppm=night * 1e6, e_night_ppm=float(math.sqrt(v0 @ cov @ v0 + eE ** 2)) * 1e6,
                depth_ppm=day * 1e6, e_depth_ppm=float(math.sqrt(v5 @ cov @ v5 + eE ** 2)) * 1e6,
                amp_ppm=float(math.hypot(coef[1], coef[2])) * 1e6, scatter_ppm=float(np.std(res)) * 1e6)


def nightside_one_star(t):
    quiet_warnings()
    tt, ff, used = [], [], []
    for s in t["sectors"]:
        b = core.spoc_bytes(t["tic"], s)
        if not b:
            continue
        try:
            x, f, _ = core.read(b, columns=("PDCSAP_FLUX",), normalise=False)
        except Exception:
            continue
        ok = np.isfinite(f)
        x, f = x[ok], f[ok]
        if len(x) < 1000:
            continue
        f = f / np.median(f)
        intr = np.zeros(len(x), bool)
        for toi in t["tois"]:
            ph = ((x - toi["t0"]) / toi["period"]) % 1.0
            w = toi["dur_h"] / 48.0 / toi["period"] + 0.01
            intr |= (ph < w) | (ph > 1 - w)
        width = max(1.5, 3 * max(toi["period"] for toi in t["tois"]))
        g = np.where(intr, np.nan, f)
        trend = nightside_running_median(x, g, width)
        f = f / trend
        r = f - 1
        mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
        keep = np.isfinite(f) & (np.abs(r) < 5 * mad)
        tt.append(x[keep])
        ff.append(f[keep])
        used.append(s)
    if not tt:
        return dict(tic=t["tic"], sectors=[], tois=[])
    x, f = np.concatenate(tt), np.concatenate(ff)
    out = [m for m in (nightside_measure(x, f, toi) for toi in t["tois"]) if m]
    return dict(tic=t["tic"], tmag=t.get("tmag"), sectors=used, days=float(len(x) * nightside_CAD), tois=out)


def nightside_stream(share, out, procs):
    from concurrent.futures import ProcessPoolExecutor
    ts = json.load(open(share or os.path.join(nightside_ROOT, "targets.json")))
    path = os.path.join(nightside_HERE, out)
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)["tic"] for line in open(path) if line.strip()}
    todo = [t for t in ts if t["tic"] not in done]
    print("%d stars to fold (%d done)" % (len(todo), len(done)), flush=True)
    with ProcessPoolExecutor(procs) as ex, open(path, "a") as fh:
        for r in ex.map(nightside_one_star, todo, chunksize=1):
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()


def nightside_load(paths):
    seen, res = set(), []
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r["tic"] not in seen:
                    seen.add(r["tic"])
                    res.append(r)
    return res


def nightside_report(paths):
    res = nightside_load(paths)
    tois = [dict(k, tic=r["tic"]) for r in res for k in r["tois"]]
    z = np.array([k["night_ppm"] / k["e_night_ppm"] for k in tois if k["e_night_ppm"] > 0])
    print("%d hosts, %d planets measured; nightside flux over its error: median %+.2f, spread %.2f; above +3: %d, below "
          "-3: %d" % (len(res), len(tois), np.median(z), 1.4826 * np.median(np.abs(z - np.median(z))), np.sum(z > 3),
                      np.sum(z < -3)))
    lim = sorted(k["night_ppm"] + 3 * k["e_night_ppm"] for k in tois)
    print("3-sigma nightside limits: best %.0f ppm, median %.0f ppm" % (lim[0], lim[len(lim) // 2]))
    for k in sorted(tois, key=lambda k: -k["night_ppm"] / max(k["e_night_ppm"], 1e-9))[:12]:
        print("  TIC %d TOI %s P %.3f d, %.2f Re: nightside %+.0f +- %.0f ppm, eclipse %+.0f +- %.0f ppm, scatter %.0f ppm"
              % (k["tic"], k["toi"], k["period"], k["radius"], k["night_ppm"], k["e_night_ppm"], k["depth_ppm"],
                 k["e_depth_ppm"], k["scatter_ppm"]))
    return res, tois


def nightside_record(paths):
    res, tois = nightside_report(paths)
    z = [k["night_ppm"] / k["e_night_ppm"] for k in tois if k["e_night_ppm"] > 0]
    lim = sorted(k["night_ppm"] + 3 * k["e_night_ppm"] for k in tois)
    core.record_search(
        "tess_usp_nightside", "Phase curves of rocky planets", archive="TESS SPOC two-minute light curves",
        release="sectors 1-%d" % max(s for r in res for s in r["sectors"]),
        selection="TOIs that are neither false positives nor false alarms, period below 1 d, radius below 2 Earth radii",
        code="seti.py transits nightside", params=dict(detrend_days=1.5, clip=5, model="c0 + a cos + b sin + c cos 2x"),
        n_targets=len(tois),
        null_summary="nightside flux over its error: %d above +3, %d below -3 of %d" % (sum(x > 3 for x in z),
                                                                                       sum(x < -3 for x in z), len(z)),
        sensitivity="3-sigma nightside limits from %.0f ppm (median %.0f ppm) of the star's flux" % (lim[0],
                                                                                                     lim[len(lim) // 2]),
        result="no nightside flux above 3 sigma; median 3-sigma limit %.0f ppm of the star's flux"
                            % lim[len(lim) // 2], ledger="Phase curves of rocky planets")
    core.clear_targets("tess_usp_nightside")
    core.record_targets("tess_usp_nightside", [
        dict(target="tic:%d" % k["tic"], stat=float(k["night_ppm"]), unit="nightside flux, ppm of the star",
             upper_limit=float(k["night_ppm"] + 3 * k["e_night_ppm"]),
             note="TOI %s, P %.3f d, %.2f Re, eclipse %+.0f +- %.0f ppm" % (k["toi"], k["period"], k["radius"],
                                                                          k["depth_ppm"], k["e_depth_ppm"]))
        for k in tois])
    print("recorded")


def nightside_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("targets", "stream", "report", "record"))
    p.add_argument("paths", nargs="*")
    p.add_argument("--list")
    p.add_argument("--out", default="phase_curves.jsonl")
    p.add_argument("--procs", type=int, default=2)
    a = p.parse_args()
    if a.mode == "targets":
        nightside_targets()
    elif a.mode == "stream":
        nightside_stream(a.list, a.out, a.procs)
    elif a.mode == "record":
        nightside_record(a.paths)
    else:
        nightside_report(a.paths)


def nightside_cli():
    quiet_warnings()
    nightside_main()


# --------------------------------------------------------------------------------------------------------------

exobelts_DOC = r"""Clarke exobelts in Kepler transits (Socas-Navarro 2018): shoulders to k Rp/v beyond contact, k 3-90 within half the Hill radius, against random phases; vetted at z 5 with light-curve contacts and times.

    python -I seti.py transits exobelts targets [--kepmag 15]
    python -I seti.py transits exobelts targets --long [--kepmag 16]   (candidates and confirmed planets below 2 Earth radii beyond 30 d around FGK dwarfs, where tides leave the planet unlocked and its synchronous orbit lies inside the Hill sphere, into ROOT/targets_long.json)
    python -I seti.py transits exobelts stream --list TARGETS.json --out exobelts_<part>.jsonl [--procs 2]
    python -I seti.py transits exobelts report exobelts_*.jsonl
    python -I seti.py transits exobelts vet KIC [KIC ...] [--list TARGETS.json]   (contacts and times from the light curve)
    python -I seti.py transits exobelts record exobelts_*.jsonl [--long]   (with vet's exobelts_vet.json, into the coverage store; --long the run of targets --long)"""

exobelts_ROOT = os.environ.get("EXOBELT_ROOT", core.data("exobelts"))
exobelts_HERE = core.RESULTS
exobelts_S3 = "https://stpubdata.s3.amazonaws.com/"
exobelts_KS = (3, 4, 6, 8, 11, 15, 20, 30, 45, 60, 90)
exobelts_MARGIN_H = 0.3  # hours excluded beyond each contact: half a cadence plus contact error
exobelts_NULLS = 40


def exobelts_num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def exobelts_targets(kepmag, long=False):
    os.makedirs(exobelts_ROOT, exist_ok=True)
    kois = core.exoplanets("SELECT kepid, kepoi_name, koi_disposition, koi_period, koi_time0bk, koi_duration, "
                               "koi_ror, koi_dor, koi_impact, koi_prad, koi_kepmag, koi_smass, koi_steff, koi_slogg FROM cumulative")
    holc = {}
    cats = core.vizier("J/ApJS/225/9")
    for r in cats[1]:
        flags = [r[c] for c in ("f_O-C", "Out", "Over")]
        if any(str(x).strip() not in ("", "--", "0", "None", "nan") for x in flags):
            continue
        tn, oc = exobelts_num(r["tn"]), exobelts_num(r["O-C"])
        if tn is None or oc is None:
            continue
        # Holczer's times are BJD - 2454900; Kepler's are BJD - 2454833
        holc.setdefault("K%08.2f" % float(r["KOI"]), []).append(tn + 67.0 + oc / 1440.0)
    by = {}
    for k in kois:
        by.setdefault(str(k["kepid"]), []).append(k)
    sel = {}
    for kic, ks in by.items():
        planets = []
        for k in ks:
            P, t0, dur, ror, dor = (exobelts_num(k[c]) for c in ("koi_period", "koi_time0bk", "koi_duration", "koi_ror",
                                                         "koi_dor"))
            if None in (P, t0, dur, ror, dor) or dor <= 1 or (exobelts_num(k["koi_kepmag"]) or 99) > kepmag:
                continue
            if long:
                # planets tides leave unlocked: candidates and confirmed below 2 Earth radii beyond 30 d around FGK dwarfs
                teff, logg, rp = (exobelts_num(k[c]) for c in ("koi_steff", "koi_slogg", "koi_prad"))
                if (k["koi_disposition"] not in ("CONFIRMED", "CANDIDATE") or P < 30 or rp is None or rp >= 2
                        or teff is None or not 4000 <= teff <= 6500 or (logg or 0) < 4.0):
                    continue
            elif k["koi_disposition"] != "CONFIRMED":
                continue
            planets.append(dict(koi=k["kepoi_name"], period=P, t0=t0, dur_h=dur, ror=ror, dor=dor,
                                impact=exobelts_num(k["koi_impact"]), prad=exobelts_num(k["koi_prad"]), smass=exobelts_num(k["koi_smass"]),
                                times=sorted(holc.get(k["kepoi_name"], [])) or None))
        if planets:
            mask = [dict(koi=k["kepoi_name"], period=exobelts_num(k["koi_period"]), t0=exobelts_num(k["koi_time0bk"]),
                         dur_h=exobelts_num(k["koi_duration"])) for k in ks if k["koi_disposition"] != "FALSE POSITIVE"
                    and None not in (exobelts_num(k["koi_period"]), exobelts_num(k["koi_time0bk"]), exobelts_num(k["koi_duration"]))]
            sel[kic] = dict(kic=kic, kepmag=exobelts_num(ks[0]["koi_kepmag"]), planets=planets, mask=mask)
    from concurrent.futures import ThreadPoolExecutor

    def listing(kic):
        k9 = "%09d" % int(kic)
        b = core.get(exobelts_S3 + "?list-type=2&prefix=kepler/public/lightcurves/%s/%s/" % (k9[:4], k9), timeout=60,
                    retries=3, soft=True, pause=1)
        return kic, re.findall(r"<Key>([^<]*_llc\.fits)</Key>", b.decode()) if b else []
    out = []
    with ThreadPoolExecutor(8) as ex:
        for kic, keys in ex.map(listing, sorted(sel)):
            if keys:
                out.append(dict(sel[kic], files=keys))
    json.dump(out, open(os.path.join(exobelts_ROOT, "targets_long.json" if long else "targets.json"), "w"))
    print("%d %s on %d stars of Kp %.1f or brighter; %d long-cadence files; %d planets with measured transit times" % (
        sum(len(t["planets"]) for t in out), "planets below 2 Earth radii beyond 30 d" if long else "confirmed planets", len(out),
        kepmag, sum(len(t["files"]) for t in out), sum(1 for t in out for p in t["planets"] if p["times"])))


def exobelts_centres(p, t):
    """Nearest transit centre to each time, Holczer's measured time where available."""
    n = np.round((t - p["t0"]) / p["period"]).astype(int)
    tc = p["t0"] + n * p["period"]
    if p["times"]:
        meas = np.array(p["times"])
        ne = np.round((meas - p["t0"]) / p["period"]).astype(int)
        lut = dict(zip(ne.tolist(), meas.tolist()))
        tc = np.array([lut.get(int(a), b) for a, b in zip(n, tc)])
    return tc


def exobelts_hill_k(p):
    """Hill radius in planetary radii: mass from radius (Chen and Kipping 2017; Jovian above 14.3 R_earth), star mass from the catalogue or 1 M_sun."""
    r = p["prad"] or 1.0
    m = r ** 3.58 if r < 1.23 else 2.10 * (r / 1.23) ** 1.70 if r < 14.3 else 318.0
    ms = p.get("smass") or 1.0
    return p["dor"] / p["ror"] * (m * 3.003e-6 / (3 * ms)) ** (1 / 3.0)


def exobelts_measure(t, f, p):
    """Shoulder depths (ppm) and significance against random phases for each k up to half the Hill radius."""
    P = p["period"]
    half = p["dur_h"] / 2.0
    rv = p["ror"] / p["dor"] * P * 24.0 / (2 * math.pi)  # hours for the planet to move one planetary radius
    x = (t - exobelts_centres(p, t)) * 24.0  # hours from the nearest transit centre
    kmax = 0.5 * exobelts_hill_k(p)
    out = []
    for k in exobelts_KS:
        d = k * rv
        base = max(d, 2.0)
        if k > kmax or d < exobelts_MARGIN_H + 0.5 or half + d + base + 1 > P * 24 / 4:
            continue

        def stat(c):
            """Baseline minus shoulder flux (both, before, after) about phase-time c (hours)."""
            y = x - c
            pre = (y > -half - d) & (y < -half - exobelts_MARGIN_H)
            post = (y > half + exobelts_MARGIN_H) & (y < half + d)
            bpre = (y > -half - d - base - 0.25) & (y < -half - d - 0.25)
            bpost = (y > half + d + 0.25) & (y < half + d + base + 0.25)
            if min(pre.sum(), post.sum(), bpre.sum(), bpost.sum()) < 5:
                return None
            b = np.mean(f[bpre | bpost])
            return b - np.mean(f[pre | post]), np.mean(f[bpre]) - np.mean(f[pre]), np.mean(f[bpost]) - np.mean(f[post])
        s = stat(0.0)
        if s is None:
            continue
        rng = np.random.default_rng(int(p["period"] * 1e6) % 2 ** 32)
        offs = [o for o in rng.uniform(0.1, 0.9, 4 * exobelts_NULLS) * P * 24 if abs(o - P * 12) > 2 * (half + d + base)
                and P * 24 - o > 2 * (half + d + base) and o > 2 * (half + d + base)][:exobelts_NULLS]
        nul = [v for v in (stat(o) for o in offs) if v]
        if len(nul) < 10:
            continue
        nb = np.array([v[0] for v in nul])
        npre = np.array([v[1] for v in nul])
        npost = np.array([v[2] for v in nul])
        sd, sdp, sdq = np.std(nb), np.std(npre), np.std(npost)
        out.append(dict(k=k, shoulder_h=round(d, 3), ppm=round(s[0] * 1e6, 1), z=round(s[0] / sd, 2) if sd > 0 else None,
                        z_pre=round(s[1] / sdp, 2) if sdp > 0 else None, z_post=round(s[2] / sdq, 2) if sdq > 0 else None,
                        sd_ppm=round(sd * 1e6, 1), lim3_ppm=round((max(s[0], 0) + 3 * sd) * 1e6, 1),
                        null_z=[round(v / sd, 2) for v in nb] if sd > 0 else [], n_null=len(nul)))
    return out


def exobelts_one_star(t, files):
    from astropy.io import fits
    T, F = [], []
    for path in files:
        try:
            with fits.open(path) as h:
                d = h[1].data
                tt, ff, qq = (np.asarray(d[c], float) for c in ("TIME", "PDCSAP_FLUX", "SAP_QUALITY"))
        except Exception:
            continue
        ok = np.isfinite(tt) & np.isfinite(ff) & (qq == 0)
        if ok.sum() < 200:
            continue
        T.append(tt[ok])
        F.append(ff[ok] / np.median(ff[ok]))
    if not T:
        return dict(kic=t["kic"], planets=[])
    tt, ff = np.concatenate(T), np.concatenate(F)
    o = np.argsort(tt)
    tt, ff = tt[o], ff[o]
    res = []
    for p in t["planets"]:
        P = p["period"]
        rv = p["ror"] / p["dor"] * P * 24.0 / (2 * math.pi)
        kmax = max([k for k in exobelts_KS if k <= 0.5 * exobelts_hill_k(p)] or [exobelts_KS[0]])
        reach = p["dur_h"] / 2.0 + kmax * rv + max(kmax * rv, 2.0) + 1.0
        keep = np.ones(len(tt), bool)
        for m in t["mask"]:
            if m["koi"] == p["koi"]:
                continue
            ph = ((tt - m["t0"]) / m["period"] + 0.5) % 1.0 - 0.5
            keep &= np.abs(ph * m["period"] * 24.0) > m["dur_h"] / 2.0 + 1.0
        x = (tt - exobelts_centres(p, tt)) * 24.0
        quiet = keep & (np.abs(x) > reach)
        if quiet.sum() < 1000:
            continue
        g = exobelts_detrend(tt, ff, quiet)
        sel = keep & np.isfinite(g)
        nt = len(np.unique(np.round((tt[sel & (np.abs(x) < reach)] - p["t0"]) / P)))
        r = exobelts_measure(tt[sel], g[sel], p)
        res.append(dict(koi=p["koi"], period=P, dur_h=p["dur_h"], ror=p["ror"], prad=p["prad"], rv_h=round(rv, 4),
                        hill_k=round(exobelts_hill_k(p), 1), n_transits=int(nt), ttv_times=bool(p["times"]), ks=r))
    return dict(kic=t["kic"], kepmag=t["kepmag"], n_points=int(len(tt)), planets=res)


def exobelts_detrend(t, f, quiet, width=2.0):
    """Flux over a running median of the quiet points (width days), interpolated across the rest."""
    import pandas as pd
    s = pd.Series(np.where(quiet, f, np.nan), index=pd.to_datetime(t, unit="D"))
    med = s.rolling("%dmin" % int(width * 1440), center=True, min_periods=20).median().to_numpy()
    ok = np.isfinite(med)
    if ok.sum() < 100:
        return np.full(len(f), np.nan)
    med = np.interp(t, t[ok], med[ok])
    return f / med


def exobelts_stream_star(t):
    import shutil
    tmp = os.path.join(exobelts_ROOT, "tmp_%s" % t["kic"])
    os.makedirs(tmp, exist_ok=True)
    local = []
    for k in t["files"]:
        p = os.path.join(tmp, os.path.basename(k))
        if not os.path.exists(p):
            b = core.get(exobelts_S3 + k, timeout=300, retries=3, soft=True, pause=2)
            if not b or not b.startswith(b"SIMPLE"):
                continue
            open(p, "wb").write(b)
        local.append(p)
    try:
        return exobelts_one_star(t, local)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def exobelts_stream(share, out, procs):
    from concurrent.futures import ProcessPoolExecutor
    ts = json.load(open(share or os.path.join(exobelts_ROOT, "targets.json")))
    os.makedirs(exobelts_HERE, exist_ok=True)
    path = os.path.join(exobelts_HERE, out)
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)["kic"] for line in open(path) if line.strip()}
    todo = [t for t in ts if t["kic"] not in done]
    print("%d stars to search (%d done)" % (len(todo), len(done)), flush=True)
    with ProcessPoolExecutor(procs) as ex, open(path, "a") as fh:
        for r in ex.map(exobelts_stream_star, todo, chunksize=1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def exobelts_own_times(t, f, p, tmpl):
    """Each transit's centre (days) by least-squares shift of the trapezoid template within a third of the duration, one-minute steps."""
    depth, dur, ing = tmpl
    pred = exobelts_centres(p, t)
    out = {}
    for c in np.unique(np.round(pred, 6)):
        near = np.abs(t - c) < dur
        if near.sum() < 10 or (np.abs(t - c) < dur / 2).sum() < 4:
            continue
        shifts = np.arange(-dur / 3, dur / 3 + 1e-9, 1.0 / 1440)
        chi = [np.sum((f[near] - 1.0 - depth * science.trapezoid(t[near] - c, 1.0, dur, ing, s)) ** 2)
               for s in shifts]
        out[int(round((c - p["t0"]) / p["period"]))] = c + shifts[int(np.argmin(chi))]
    return out


def exobelts_vet(listing, kics):
    """Shoulder test for the named stars with centres and duration from the light curve: timing or duration artefacts vanish, a belt stays."""
    import shutil
    ts = {t["kic"]: t for t in json.load(open(listing))}
    path = core.results("exobelts_vet.json")
    done = json.load(open(path)) if os.path.exists(path) else []
    for kic in kics:
        t = ts[str(kic)]
        tmp = os.path.join(exobelts_ROOT, "vet_%s" % kic)
        os.makedirs(tmp, exist_ok=True)
        files = []
        for k in t["files"]:
            lc = os.path.join(tmp, os.path.basename(k))
            if not os.path.exists(lc):
                b = core.get(exobelts_S3 + k, timeout=300, retries=3, soft=True, pause=2)
                if not b or not b.startswith(b"SIMPLE"):
                    continue
                open(lc, "wb").write(b)
            files.append(lc)
        try:
            base = exobelts_one_star(t, files)
            for p in t["planets"]:
                q = [x for x in base["planets"] if x["koi"] == p["koi"]]
                if not q or not q[0]["ks"]:
                    continue
                b0 = max((k for k in q[0]["ks"] if k["z"] is not None), key=lambda k: k["z"])
                tt, ff = exobelts_series(files, t, p)
                if tt is None:
                    continue
                p2 = dict(p)
                for _ in range(3):
                    x, y, e = exobelts_fold(tt, ff, p2)
                    best, _ = science.fit_trapezoid(x, y, e, p2["dur_h"] / 24.0)
                    own = exobelts_own_times(tt, ff, p2, (best[1], best[2], best[3]))
                    p2 = dict(p2, times=sorted(own.values()), dur_h=best[2] * 24.0)
                x2, y2, e2 = exobelts_fold(tt, ff, p2)
                best2, _ = science.fit_trapezoid(x2, y2, e2, p2["dur_h"] / 24.0)
                p2["dur_h"] = best2[2] * 24.0
                r2 = {k["k"]: k for k in exobelts_measure(tt, ff, p2)}
                oc = np.array([own[n] - (p["t0"] + n * p["period"]) for n in sorted(own)]) * 1440.0
                k2 = r2.get(b0["k"])
                print("KIC %s %s: catalogue duration %.2f h, fitted %.2f h, depth %.0f ppm; own times of %d transits, "
                      "O-C robust spread %.1f min; k %d: z %.1f with catalogue contacts, %s with fitted ones (best %s)"
                      % (kic, p["koi"], q[0]["dur_h"], p2["dur_h"], best2[1] * 1e6, len(own),
                         1.4826 * np.median(np.abs(oc - np.median(oc))), b0["k"], b0["z"],
                         "%.1f" % k2["z"] if k2 and k2["z"] is not None else "unmeasured",
                         max((k["z"] for k in r2.values() if k["z"] is not None), default=None)), flush=True)
                row = dict(kic=str(kic), koi=p["koi"], k=b0["k"], z=b0["z"], z_pre=b0["z_pre"], z_post=b0["z_post"],
                           dur_cat_h=q[0]["dur_h"], dur_fit_h=round(p2["dur_h"], 2), n_own=len(own),
                           oc_spread_min=round(float(1.4826 * np.median(np.abs(oc - np.median(oc)))), 1) if len(oc) else None,
                           z_fitted=k2["z"] if k2 else None)
                for label, key, pp in (("catalogue contacts", "z_matched", p),
                                       ("fitted contacts and own times", "z_matched_fitted", p2)):
                    rt, rf, rk = exobelts_series(files, t, pp, raw=True)
                    zm = exobelts_matched_z(rt, rf, rk, pp, b0["k"])
                    row[key] = round(zm[0], 2) if zm[0] is not None else None
                    print("    k %d, %s, against a null detrended as the transit is: z %s, %s ppm (%d null phases)"
                          % (b0["k"], label, "%.1f" % zm[0] if zm[0] is not None else "unmeasured",
                             "%.0f" % zm[1] if zm[1] is not None else "-", zm[2]), flush=True)
                done = [x for x in done if not (x["kic"] == row["kic"] and x["koi"] == row["koi"])] + [row]
                json.dump(done, open(path, "w"), indent=1)
                print("    O-C (min): %s" % " ".join("%.0f" % v for v in oc - np.median(oc)), flush=True)
                print("    profile (h, ppm): %s" % " ".join("%.1f:%.0f" % (a * 24, b * 1e6) for a, b in zip(x2, y2)
                                                         if np.isfinite(b)), flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def exobelts_matched_z(tt, raw, keep, p, k):
    """Shoulder statistic at k against random phases detrended with the transit's gap: (z, ppm, null phases)."""
    P = p["period"]
    half = p["dur_h"] / 2.0
    rv = p["ror"] / p["dor"] * P * 24.0 / (2 * math.pi)
    d = k * rv
    base = max(d, 2.0)
    reach = half + d + base + 1.0
    x = (tt - exobelts_centres(p, tt)) * 24.0

    def stat(c):
        g = exobelts_detrend(tt, raw, keep & (np.abs(x - c) > reach))
        y = x - c
        pre = keep & (y > -half - d) & (y < -half - exobelts_MARGIN_H)
        post = keep & (y > half + exobelts_MARGIN_H) & (y < half + d)
        bb = keep & (((y > -half - d - base - 0.25) & (y < -half - d - 0.25)) | ((y > half + d + 0.25) & (y < half + d + base + 0.25)))
        if min(pre.sum(), post.sum(), bb.sum()) < 5 or not np.isfinite(g[pre | post | bb]).all():
            return None
        return np.mean(g[bb]) - np.mean(g[pre | post])
    s = stat(0.0)
    rng = np.random.default_rng(int(P * 1e6) % 2 ** 32)
    offs = [o for o in rng.uniform(0.1, 0.9, 4 * exobelts_NULLS) * P * 24 if abs(o - P * 12) > 2 * reach
            and P * 24 - o > 2 * reach and o > 2 * reach][:exobelts_NULLS]
    nul = np.array([v for v in (stat(o) for o in offs) if v is not None])
    if s is None or len(nul) < 10:
        return None, None, len(nul)
    return float(s / np.std(nul)), float(s * 1e6), len(nul)


def exobelts_series(files, t, p, raw=False):
    """One planet's detrended light curve with other transits masked; raw=True gives undetrended flux and the usable mask."""
    from astropy.io import fits
    T, F = [], []
    for path in files:
        try:
            with fits.open(path) as h:
                d = h[1].data
                tt, ff, qq = (np.asarray(d[c], float) for c in ("TIME", "PDCSAP_FLUX", "SAP_QUALITY"))
        except Exception:
            continue
        ok = np.isfinite(tt) & np.isfinite(ff) & (qq == 0)
        if ok.sum() >= 200:
            T.append(tt[ok])
            F.append(ff[ok] / np.median(ff[ok]))
    if not T:
        return None, None
    tt, ff = np.concatenate(T), np.concatenate(F)
    o = np.argsort(tt)
    tt, ff = tt[o], ff[o]
    P = p["period"]
    rv = p["ror"] / p["dor"] * P * 24.0 / (2 * math.pi)
    kmax = max([k for k in exobelts_KS if k <= 0.5 * exobelts_hill_k(p)] or [exobelts_KS[0]])
    reach = p["dur_h"] / 2.0 + kmax * rv + max(kmax * rv, 2.0) + 1.0
    keep = np.ones(len(tt), bool)
    for m in t["mask"]:
        if m["koi"] != p["koi"]:
            ph = ((tt - m["t0"]) / m["period"] + 0.5) % 1.0 - 0.5
            keep &= np.abs(ph * m["period"] * 24.0) > m["dur_h"] / 2.0 + 1.0
    if raw:
        return tt, ff, keep
    x = (tt - exobelts_centres(p, tt)) * 24.0
    g = exobelts_detrend(tt, ff, keep & (np.abs(x) > reach))
    sel = keep & np.isfinite(g)
    return tt[sel], g[sel]


def exobelts_fold(t, f, p, nb=80):
    """Binned profile (days from centre, flux - 1, error) over +-1.5 durations."""
    x = t - exobelts_centres(p, t)
    w = 1.5 * p["dur_h"] / 24.0
    edges = np.linspace(-w, w, nb + 1)
    k = np.digitize(x, edges) - 1
    y = np.array([np.median(f[k == i]) - 1.0 if np.sum(k == i) >= 2 else np.nan for i in range(nb)])
    e = np.array([max(1.2533 * np.std(f[k == i]), 1e-5) / np.sqrt(np.sum(k == i)) if np.sum(k == i) >= 2 else np.nan
                  for i in range(nb)])
    return 0.5 * (edges[1:] + edges[:-1]), y, e


def exobelts_report(paths):
    seen, res = set(), []
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r["kic"] not in seen:
                    seen.add(r["kic"])
                    res.append(r)
    pl = [dict(q, kic=r["kic"]) for r in res for q in r["planets"] if q["ks"]]
    best = []
    for q in pl:
        ok = [k for k in q["ks"] if k["z"] is not None]
        if ok:
            b = max(ok, key=lambda k: k["z"])
            best.append((q, b))
    trials = sum(len(q["ks"]) for q in pl)
    allz = np.array([k["z"] for q in pl for k in q["ks"] if k["z"] is not None])
    nulls = np.array([v for q in pl for k in q["ks"] for v in k["null_z"]])
    print("%d stars, %d planets with a measurement, %d planet-k tests; shoulder z above 5 with both sides above 2: %d; "
          "fraction of null z above 5: %.4f (expected among the tests %.1f)"
          % (len(res), len(pl), trials, sum(1 for q in pl for k in q["ks"] if (k["z"] or 0) > 5
                                             and (k["z_pre"] or 0) > 2 and (k["z_post"] or 0) > 2),
             float(np.mean(nulls > 5)) if len(nulls) else float("nan"), float(np.mean(nulls > 5)) * len(allz)))
    lims = sorted(k["lim3_ppm"] for q in pl for k in q["ks"] if k["k"] == 6)
    if lims:
        print("3-sigma shoulder limits at k = 6: best %.0f ppm, median %.0f ppm" % (lims[0], lims[len(lims) // 2]))
    for q, b in sorted(best, key=lambda x: -x[1]["z"])[:20]:
        print("  KIC %s %s P %.3f d Rp/R* %.3f, %d transits%s: k %d (%.2f h) %.0f +- %.0f ppm, z %.1f (before %.1f, after "
              "%.1f)" % (q["kic"], q["koi"], q["period"], q["ror"], q["n_transits"], ", measured times" if q["ttv_times"]
                         else "", b["k"], b["shoulder_h"], b["ppm"], b["sd_ppm"], b["z"], b["z_pre"] or 0,
                         b["z_post"] or 0))
    return res, pl


def exobelts_record(paths, long=False):
    """Record each planet's strongest shoulder and 3-sigma limit at k = 6, with vet's remeasurements; a shoulder survives above z 5 in both. With long, the run of the planets below 2 Earth radii beyond 30 d (targets --long)."""
    run = "exobelts_kepler_long" if long else "exobelts_kepler"
    res, pl = exobelts_report(paths)
    vet = {(v["kic"], v["koi"]): v for v in json.load(open(core.results("exobelts_vet.json")))}
    rows, survivors, explained = [], [], 0
    for q in pl:
        ok = [k for k in q["ks"] if k["z"] is not None]
        if not ok:
            continue
        b = max(ok, key=lambda k: k["z"])
        k6 = [k for k in q["ks"] if k["k"] == 6]
        v = vet.get((str(q["kic"]), q["koi"]))
        note = "%s, P %.3f d, %d transits; strongest shoulder k %d (%.2f h) %.0f ppm at z %.1f" % (
            q["koi"], q["period"], q["n_transits"], b["k"], b["shoulder_h"], b["ppm"], b["z"])
        vcode = None
        if b["z"] > 4:
            if v is None:
                survivors.append(q["koi"])
                note += "; not yet measured against the matched null"
            elif min(v["z_matched"] or 0, v["z_matched_fitted"] or 0) >= 5:
                survivors.append(q["koi"])
                note += "; matched-null z %.1f and %.1f" % (v["z_matched"], v["z_matched_fitted"])
            else:
                explained += 1
                vcode = ("shoulder from the detrending gap and the transit's timing and duration: against a null "
                         "detrended as the transit is, z %s with catalogue contacts and %s with the light curve's "
                         "(duration %.1f h against the catalogue's %.1f h, transit times spread %s min)"
                         % (v["z_matched"], v["z_matched_fitted"], v["dur_fit_h"], v["dur_cat_h"], v["oc_spread_min"]))
        rows.append(dict(target="kic:%s" % q["kic"], stat=b["z"], vet=vcode, note=note,
                         upper_limit=k6[0]["lim3_ppm"] if k6 else None,
                         unit="shoulder z against random phases; limit: 3-sigma shoulder depth (ppm) at k = 6"))
    lims = sorted(r["upper_limit"] for r in rows if r["upper_limit"] is not None)
    trials = sum(len(q["ks"]) for q in pl)
    core.record_search(
        run, "Clarke exobelts beyond 30 days" if long else "Clarke exobelts", archive="MAST Kepler long-cadence PDCSAP light "
        "curves; NASA Exoplanet Archive cumulative KOI table; Holczer et al. 2016 transit times", release="DR25",
        selection=("candidate and confirmed Kepler planets below 2 Earth radii beyond 30 d around FGK dwarfs (Teff 4000-6500 K, "
                   "log g 4 or above) on stars of Kp 16 or brighter with long-cadence data") if long else
                  "confirmed Kepler planets on stars of Kp 15 or brighter with long-cadence data",
        code=os.path.abspath(__file__), entry="exobelts", n_targets=len(rows),
        params=dict(k=list(exobelts_KS), margin_h=exobelts_MARGIN_H, null_phases=exobelts_NULLS, stage_two_z=4,
                    survive_z=5),
        null_summary="%d stars, %d planets, %d planet-k tests; %d planets above z 4 measured again against a null "
                     "detrended as the transit is, %d of them explained" % (len(res), len(rows), trials,
                                                                           sum(1 for r in rows if r["stat"] > 4),
                                                                           explained),
        sensitivity="3-sigma shoulder depth at k = 6: best %.0f ppm, median %.0f ppm" % (lims[0], lims[len(lims) // 2]),
        ledger="Clarke exobelts beyond 30 days" if long else "Clarke exobelts", n_unit="planets",
        result="no shoulder survives" if not survivors else "%d shoulders survive: %s" % (len(survivors),
                                                                                         ", ".join(survivors)))
    core.clear_targets(run)
    core.record_targets(run, rows)
    print("recorded %d planets; survivors %s" % (len(rows), survivors or "none"))


def exobelts_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("targets", "stream", "report", "vet", "record"))
    p.add_argument("paths", nargs="*")
    p.add_argument("--kepmag", type=float, default=15.0)
    p.add_argument("--list")
    p.add_argument("--out", default="exobelts.jsonl")
    p.add_argument("--procs", type=int, default=2)
    p.add_argument("--long", action="store_true")
    a = p.parse_args()
    if a.mode == "targets":
        exobelts_targets(a.kepmag, a.long)
    elif a.mode == "stream":
        exobelts_stream(a.list, a.out, a.procs)
    elif a.mode == "vet":
        exobelts_vet(a.list or os.path.join(exobelts_ROOT, "targets.json"), a.paths)
    elif a.mode == "record":
        exobelts_record(a.paths, a.long)
    else:
        exobelts_report(a.paths)


def exobelts_cli():
    quiet_warnings()
    exobelts_main()


# --------------------------------------------------------------------------------------------------------------

commensurability_DOC = r"""Exact commensurability: Exoplanet Archive pairs on p:q (q up to 7) within errors, counts below |delta| against the 1e-3 to 1e-2 density extrapolated.

    python -I seti.py transits commensurability search
    python -I seti.py transits commensurability record"""

commensurability_OUT = "commensurability.json"


def commensurability_search():
    from fractions import Fraction
    q = ("select hostname,pl_name,pl_orbper,pl_orbpererr1,pl_orbpererr2,sy_pnum,discoverymethod,tran_flag,gaia_dr3_id "
         "from pscomppars where sy_pnum>1 and pl_orbper is not null")
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode({"query": q, "format": "csv"})
    rows = list(csv.DictReader(io.StringIO(core.get(url, timeout=300).decode())))
    by = {}
    for r in rows:
        by.setdefault(r["hostname"], []).append(r)
    ratios = sorted({Fraction(p, k) for k in range(1, 8) for p in range(k + 1, 3 * k + 1)
                     if math.gcd(p, k) == 1 and p <= 9})

    def err(p):
        e = [abs(float(x)) for x in (p["pl_orbpererr1"], p["pl_orbpererr2"]) if x not in ("", None)]
        return max(e) if e else float("nan")
    pairs = []
    for h, ps in by.items():
        for a in ps:
            for c in ps:
                P1, P2 = float(a["pl_orbper"]), float(c["pl_orbper"])
                if not P2 > P1:
                    continue
                r = P2 / P1
                best = min(ratios, key=lambda f: abs(r / float(f) - 1))
                e1, e2 = err(a), err(c)
                pairs.append(dict(host=h, gaia=(a["gaia_dr3_id"] or "").replace("Gaia DR3 ", "").strip(),
                                  inner=a["pl_name"], outer=c["pl_name"], P1=P1, P2=P2,
                                  res="%d:%d" % (best.numerator, best.denominator), delta=r / float(best) - 1,
                                  sd=math.hypot(e1 / P1, e2 / P2), transit=a["tran_flag"] == "1" and c["tran_flag"] == "1"))
    d = np.array([abs(p["delta"]) for p in pairs])
    dens = np.sum((d > 1e-3) & (d < 1e-2)) / 9e-3
    counts = {"%.0e" % lim: [int(np.sum(d < lim)), float(dens * lim)] for lim in (1e-6, 1e-5, 1e-4, 1e-3)}
    pairs.sort(key=lambda p: abs(p["delta"]))
    print("%d planets in %d multi-planet systems; %d pairs" % (len(rows), len(by), len(pairs)))
    for lim, (n, e) in counts.items():
        print("  |delta| < %s: %d pairs against %.2f for a smooth distribution" % (lim, n, e))
    for p in pairs[:12]:
        print("  %-16s %-18s %-18s %s  delta %+.2e +- %.1e (%.1f sigma from exact)"
              % (p["host"][:16], p["inner"][:18], p["outer"][:18], p["res"], p["delta"], p["sd"],
                 abs(p["delta"]) / p["sd"] if p["sd"] > 0 else float("nan")))
    json.dump(dict(n_planets=len(rows), n_systems=len(by), counts=counts, pairs=pairs),
              open(core.results(commensurability_OUT), "w"), indent=0)


def commensurability_record():
    d = json.load(open(core.results(commensurability_OUT)))
    best = {}
    for p in d["pairs"]:
        key = ("gaia_dr3:" + p["gaia"]) if p["gaia"] else ("host:" + p["host"])
        if key not in best or abs(p["delta"]) < abs(best[key]["delta"]):
            best[key] = p
    c = d["counts"]
    core.record_search(
        "exact_commensurability", "Exact commensurability", archive="NASA Exoplanet Archive pscomppars",
        selection="every pair of planets in multi-planet systems", code=os.path.abspath(__file__),
        entry="commensurability", n_targets=len(best), params=dict(q_max=7, p_max=9),
        null_summary="density of |delta| between 1e-3 and 1e-2 carried to smaller values",
        sensitivity="the period errors: median %.1e in delta" % float(np.nanmedian([p["sd"] for p in d["pairs"]])),
        ledger="Exact commensurability",
        result="no pair within 1e-5 of a small-integer ratio (%.2f expected); %d within 1e-4 against %.1f, the "
               "closest in the librating chains of TOI-1136, Kepler-223 and HD 110067"
               % (c["1e-05"][1], c["1e-04"][0], c["1e-04"][1]))
    core.clear_targets("exact_commensurability")
    core.record_targets("exact_commensurability", [
        dict(target=k, stat=abs(p["delta"]), unit="fractional distance of the closest pair from p:q",
             note="%s / %s near %s, delta %+.2e +- %.1e" % (p["inner"], p["outer"], p["res"], p["delta"], p["sd"]))
        for k, p in best.items()])
    print("recorded %d systems" % len(best))


def commensurability_cli():
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        commensurability_record()
    else:
        commensurability_search()


# --------------------------------------------------------------------------------------------------------------

nightheat_DOC = r"""Nightside heat on bare rocks: published two-sigma nightside limits (LHS 3844 b, GJ 367 b) as power limits P < 2 pi R^2 sigma T^4.

    python -I seti.py transits nightheat [record] [--sigma 2]"""

nightheat_PLANETS = (
    dict(name="LHS 3844 b", target="gaia_dr3:6385548541499112448", r_earth=1.303, mode="tb1",
         t1=710.0, lam_um=4.5, ref="Kreidberg et al. 2019, Nature 573, 87"),
    dict(name="GJ 367 b", target="gaia_dr3:5412250540681250560", r_earth=0.699, mode="ratio", ratio=4e-6, ratio_err=8e-6,
         lam_um=8.0, t_star=3522.0, r_star_sun=0.457, ref="Zhang et al. 2024, ApJL 961, L44"),
)


def nightheat_main(record=False, sigma=2.0):
    h, c, k, sb = 6.626e-34, 2.998e8, 1.381e-23, 5.670e-8

    def x_of(t, lam):
        return h * c / (lam * 1e-6 * k * t)
    out = []
    for p in nightheat_PLANETS:
        lam = p["lam_um"]
        if p["mode"] == "tb1":
            # the limit at SIGMA is that multiple of the one-sigma Planck occupancy
            occ = sigma / math.expm1(x_of(p["t1"], lam))
            t2 = h * c / (lam * 1e-6 * k * math.log1p(1.0 / occ))
        else:
            rr = (p["r_earth"] * 6.371e6 / (p["r_star_sun"] * 6.957e8)) ** 2
            occ_star = 1.0 / math.expm1(x_of(p["t_star"], lam))
            occ = (p["ratio"] + sigma * p["ratio_err"]) / rr * occ_star
            t2 = h * c / (lam * 1e-6 * k * math.log1p(1.0 / occ))
        r = p["r_earth"] * 6.371e6
        power = 2 * math.pi * r * r * sb * t2 ** 4
        out.append(dict(p, t2=t2, power=power))
        print("%s: nightside below %.0f K at %g sigma, so below %.1e W over the night hemisphere (%s)"
              % (p["name"], t2, sigma, power, p["ref"]))
    if not record:
        return
    core.record_search(
        "nightside_heat", "Nightside heat on bare rocks", archive="published phase curves (Spitzer, JWST MIRI)",
        selection="airless tidally locked planets with published nightside limits", code=os.path.abspath(__file__), entry="nightheat",
        n_targets=len(out), params=dict(sigma=sigma), sensitivity="%g-sigma nightside brightness temperature" % sigma, ledger="Nightside heat on bare rocks",
        result="nightside power below %s at %g sigma; Kardashev-I powers of 1e16-1e17 W are out of reach"
               % (", ".join("%.0e W on %s" % (o["power"], o["name"]) for o in out), sigma))
    core.clear_targets("nightside_heat")
    core.record_targets("nightside_heat", [dict(target=o["target"], upper_limit=o["power"], unit="W over the night hemisphere",
                                                    note="T_night < %.0f K (%g sigma), %s" % (o["t2"], sigma, o["ref"])) for o in out])
    print("recorded")


stability_DOC = r"""Station-kept planetary systems: multi-planet systems unstable within 1 per cent of their age in over 99 per cent of error draws, phases fixed by transit and periastron times; SPOCK for three or more planets (its regressor where 1e9 orbits outlast 1 per cent of the age), Hill stability then REBOUND for two; stable twins observed with the same errors as the null.

    python -I seti.py transits stability fetch
    python -I seti.py transits stability limits                           (each planet's mass and eccentricity limit flags, for the vets)
    python -I seti.py transits stability run [--procs 12] [--part K N]     (needs spock and rebound; 100 draws a system, 1000 once 90 per cent are unstable)
    python -I seti.py transits stability twins [--procs 12] [--twins 10]   (stable draws of every system over 99 per cent unstable, observed again with its errors)
    python -I seti.py transits stability long [--procs 12] [--draws 20]    (those systems' draws integrated directly with WHFast to 1e8 orbits of the inner planet)
    python -I seti.py transits stability record"""

stability_ROOT = os.environ.get("STABILITY_ROOT", core.data("spock"))
stability_ME = 3.003489e-6
stability_DRAWS = (100, 1000)  # draws a system, the second count once stability_MORE of the first are unstable
stability_MORE = 0.9
stability_FLAG = 0.99  # unstable fraction of the draws a flag needs
stability_AGE = 1.0  # Gyr where the archive gives no age
stability_NBODY = 1e6  # orbits of the inner planet a Hill-unstable two-planet draw is integrated for
stability_STEPS = 2e7  # and at most this many steps
stability_REG = 100  # regressor draws where 1e9 orbits of the inner planet outlast 1 per cent of the age
stability_TWIN = 0.05  # flagged fraction of a system's stable twins that explains its flag
stability_LONG = 1e8  # orbits of the inner planet a flagged system's direct integrations run for
stability_LONG_STEPS = 2e9  # and at most this many steps
stability_MODEL = {}


def stability_fetch():
    os.makedirs(stability_ROOT, exist_ok=True)
    q = ("select pl_name,hostname,sy_pnum,pl_orbper,pl_orbpererr1,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_bmassprov,"
         "pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_orblper,pl_orbtper,pl_tranmid,pl_orbincl,st_mass,st_masserr1,"
         "st_masserr2,st_age,st_ageerr1,st_ageerr2,discoverymethod,pl_controv_flag from pscomppars where sy_pnum>=2")
    b = core.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(dict(query=q, format="csv")), timeout=600)
    open(os.path.join(stability_ROOT, "multis.csv"), "wb").write(b if isinstance(b, bytes) else b.encode("utf-8"))
    print("fetched")


def stability_limits():
    """Each planet's mass and eccentricity limit flags (1 an upper limit, -1 a lower), into ROOT/limits.csv."""
    q = "select pl_name,pl_bmasselim,pl_orbeccenlim from pscomppars where sy_pnum>=2"
    b = core.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(dict(query=q, format="csv")), timeout=600)
    open(os.path.join(stability_ROOT, "limits.csv"), "wb").write(b if isinstance(b, bytes) else b.encode("utf-8"))
    print("fetched limit flags")


def stability_jobs():
    rows = list(csv.DictReader(open(os.path.join(stability_ROOT, "multis.csv"), encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["hostname"], []).append(r)
    jobs = []
    for host, g in by.items():
        g = sorted([r for r in g if num(r["pl_orbper"]) and num(r["pl_bmasse"])], key=lambda r: num(r["pl_orbper"]))
        if len(g) >= 2 and not any((num(r["pl_controv_flag"]) or 0) > 0 for r in g):
            jobs.append((host, g))
    return sorted(jobs, key=lambda j: len(j[1]))


def stability_measured(g):
    """Star mass, the age a flag is judged against (lower 1 sigma), and each planet's period, mass, eccentricity, periastron argument, transit and periastron times, with the errors the draws take above and below."""
    r0 = g[0]
    ms, age = num(r0["st_mass"]) or 1.0, num(r0["st_age"])
    pl = []
    for r in g:
        m, e = num(r["pl_bmasse"]), num(r["pl_orbeccen"])
        mu = abs(num(r["pl_bmasseerr1"]) or 0.2 * m)
        eu = abs(num(r["pl_orbeccenerr1"]) or 0.0) if e is not None else 0.0
        pl.append(dict(P=num(r["pl_orbper"]), m=m, m_up=mu, m_lo=abs(num(r["pl_bmasseerr2"]) or mu), e=e or 0.0, e_up=eu,
                       e_lo=abs(num(r["pl_orbeccenerr2"]) or eu) if e is not None else 0.0, w=num(r["pl_orblper"]),
                       tc=num(r["pl_tranmid"]), tp=num(r["pl_orbtper"])))
    su = abs(num(r0["st_masserr1"]) or 0.05 * ms)
    return dict(ms=ms, ms_up=su, ms_lo=abs(num(r0["st_masserr2"]) or su), planets=pl, age=age,
                age_used=max(age - abs(num(r0["st_ageerr2"]) or 0.0), 0.01) if age else stability_AGE)


def stability_split(rng, x, up, lo, n):
    """n draws about x, up and lo the standard deviations above and below."""
    z = rng.standard_normal(n)
    return x + np.where(z > 0, up, lo) * z


def stability_draws(s, n, rng, centre=None):
    """n draws of the star mass and of each planet's mass, eccentricity, pericentre and node longitudes, inclination and mean longitude at a common epoch: a transit puts the planet at true longitude pi/2, towards the observer, a periastron time fixes the mean anomaly, and neither leaves it random; centre replaces the measured star mass, masses and eccentricities."""
    pl = s["planets"]
    c = centre or dict(ms=s["ms"], m=[p["m"] for p in pl], e=[p["e"] for p in pl])
    ms = np.maximum(stability_split(rng, c["ms"], s["ms_up"], s["ms_lo"], n), 0.05)
    ts = [t for t in (p["tc"] if p["tc"] is not None else p["tp"] for p in pl) if t is not None]
    t0 = float(np.median(ts)) if ts else 0.0
    cols = []
    for i, p in enumerate(pl):
        e = np.clip(stability_split(rng, c["e"][i], p["e_up"], p["e_lo"], n), 0.0, 0.9)
        pom = np.full(n, math.radians(p["w"])) if p["w"] is not None else rng.uniform(0, 2 * math.pi, n)
        if p["tc"] is not None:
            E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan((math.pi / 2 - pom) / 2))
            M = E - e * np.sin(E) + 2 * math.pi * (t0 - p["tc"]) / p["P"]
        elif p["tp"] is not None:
            M = np.full(n, 2 * math.pi * (t0 - p["tp"]) / p["P"])
        else:
            M = rng.uniform(0, 2 * math.pi, n)
        cols.append(dict(m=np.maximum(stability_split(rng, c["m"][i], p["m_up"], p["m_lo"], n), 0.05 * p["m"]), e=e, pomega=pom,
                         l=pom + M, Omega=rng.uniform(0, 2 * math.pi, n), inc=np.radians(rng.uniform(0, 1, n))))
    return ms, cols


def stability_sim(ms, P, cols, k):
    import rebound
    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.add(m=float(ms[k]))
    for Pi, c in zip(P, cols):
        sim.add(m=float(c["m"][k]) * stability_ME, P=Pi, e=float(c["e"][k]), inc=float(c["inc"][k]), Omega=float(c["Omega"][k]),
                pomega=float(c["pomega"][k]), l=float(c["l"][k]))
    sim.move_to_com()
    return sim


def stability_crossing(sim):
    """Adjacent orbits whose apocentre and pericentre overlap."""
    ps = [sim.particles[i] for i in range(1, sim.N)]
    return any(b.a * (1 - b.e) < a.a * (1 + a.e) for a, b in zip(ps[:-1], ps[1:]))


def stability_hill(sim):
    """Gladman (1993): a coplanar pair that can never approach closely."""
    s, a, b = sim.particles[0], sim.particles[1], sim.particles[2]
    m1, m2 = a.m / s.m, b.m / s.m
    al, lam = m1 + m2, math.sqrt(b.a / a.a)
    lhs = al ** -3 * (m1 + m2 / lam ** 2) * (m1 * math.sqrt(1 - a.e ** 2) + m2 * math.sqrt(1 - b.e ** 2) * lam) ** 2
    return lhs > 1 + 3 ** (4 / 3) * m1 * m2 / al ** (4 / 3)


def stability_nbody(sim, tmax):
    """True when a pair comes within its mutual Hill radius or loses a planet within tmax days (WHFast, at most stability_STEPS)."""
    import rebound
    s, a, b = sim.particles[0], sim.particles[1], sim.particles[2]
    sim.integrator = "whfast"
    sim.dt = min(p.P * (1 - p.e) ** 1.5 for p in (a, b)) / 20
    # the distance applies to the star as well, so it stays inside the inner planet's pericentre
    sim.exit_min_distance = min(((a.m + b.m) / (3 * s.m)) ** (1 / 3) * (a.a + b.a) / 2, 0.5 * a.a * (1 - a.e))
    sim.exit_max_distance = 100 * b.a * (1 + b.e)
    try:
        sim.integrate(min(tmax, stability_STEPS * sim.dt), exact_finish_time=0)
    except (rebound.Encounter, rebound.Escape):
        return True
    return False


def stability_eval(s, n, rng, centre=None, keep=20):
    """Each of n draws' chance of instability within its horizon (SPOCK's 1e9 orbits for three or more planets, the N-body window for two, certainty where orbits cross), the crossing count, and up to keep draws judged stable (twin centres)."""
    P = [p["P"] for p in s["planets"]]
    ms, cols = stability_draws(s, n, rng, centre)
    win = 0.01 * s["age_used"] * 365.25e9
    u, idx, cross = np.zeros(n), [], 0
    for k in range(n):
        sim = stability_sim(ms, P, cols, k)
        if stability_crossing(sim):
            u[k], cross = 1.0, cross + 1
        elif len(P) == 2:
            u[k] = 0.0 if stability_hill(sim) else float(stability_nbody(sim, min(stability_NBODY * P[0], win)))
        else:
            idx.append(k)
    if idx:
        try:
            u[idx] = 1 - np.asarray(stability_MODEL["fc"].predict_stable([stability_sim(ms, P, cols, k) for k in idx], n_jobs=1), float)
        except Exception:
            for k in idx:
                try:
                    u[k] = 1 - float(stability_MODEL["fc"].predict_stable(stability_sim(ms, P, cols, k), n_jobs=1))
                except Exception:
                    u[k] = np.nan
    stable = [dict(ms=float(ms[k]), m=[float(c["m"][k]) for c in cols], e=[float(c["e"][k]) for c in cols])
              for k in range(n) if u[k] < 0.5][:keep]
    return u, cross, stable


def stability_regress(s, rng, centre=None):
    """SPOCK's regressor over stability_REG draws: the mean chance each goes unstable within 1 per cent of the age."""
    if "dr" not in stability_MODEL:
        import torch
        torch.set_num_threads(1)
        from spock import DeepRegressor
        stability_MODEL["dr"] = DeepRegressor()
    P = [p["P"] for p in s["planets"]]
    ms, cols = stability_draws(s, stability_REG, rng, centre)
    win = 0.01 * s["age_used"] * 365.25e9
    u = []
    for k in range(stability_REG):
        sim = stability_sim(ms, P, cols, k)
        if stability_crossing(sim):
            u.append(1.0)
            continue
        try:
            t = stability_MODEL["dr"].predict_instability_time(sim, samples=1000, return_samples=True, Ncpus=1)[-1]
            u.append(float(np.mean(np.asarray(t) < win)))
        except Exception:
            pass
    return float(np.mean(u)) if u else None


def stability_system(job, centre=None, seed=0, keep=20):
    """A system's fraction of error draws unstable within 1 per cent of its age: stability_DRAWS draws, the regressor's estimate where SPOCK's 1e9 orbits outlast that and the classifier leaves the system over stability_FLAG."""
    import zlib
    host, g = job
    s = stability_measured(g)
    pl = s["planets"]
    rng = np.random.default_rng(zlib.crc32(host.encode()) + seed)
    u, cross, stable = stability_eval(s, stability_DRAWS[0], rng, centre, keep)
    if np.nanmean(u) >= stability_MORE:
        u2, c2, st2 = stability_eval(s, stability_DRAWS[1] - stability_DRAWS[0], rng, centre, keep - len(stable))
        u, cross, stable = np.concatenate([u, u2]), cross + c2, stable + st2
    a = [(s["ms"] * (p["P"] / 365.25) ** 2) ** (1 / 3) for p in pl]
    rec = dict(host=host, n=len(pl), periods=[p["P"] for p in pl], masses=[p["m"] for p in pl], ecc=[p["e"] for p in pl],
               ecc_err=[p["e_up"] for p in pl], prov=sorted({r["pl_bmassprov"] for r in g}),
               method=sorted({r["discoverymethod"] for r in g}), age=s["age"], age_used=s["age_used"],
               phased=sum(1 for p in pl if p["tc"] is not None or p["tp"] is not None), draws=len(u), crossing=cross,
               U=float(np.nanmean(u)), horizon="spock" if len(pl) >= 3 else "nbody", stable=stable,
               min_hill=float(min((a2 - a1) / (((pl[i]["m"] + pl[i + 1]["m"]) * stability_ME / (3 * s["ms"])) ** (1 / 3) * (a1 + a2) / 2)
                                  for i, (a1, a2) in enumerate(zip(a[:-1], a[1:])))))
    if len(pl) >= 3 and rec["U"] > stability_FLAG and 1e9 * pl[0]["P"] > 0.01 * s["age_used"] * 365.25e9:
        rec["U_1e9"], rec["U"], rec["horizon"] = rec["U"], stability_regress(s, rng, centre), "regressor"
    return rec


def stability_init():
    quiet_warnings()
    import warnings
    warnings.filterwarnings("ignore")
    from spock import FeatureClassifier
    stability_MODEL["fc"] = FeatureClassifier()


def stability_path(kind, host):
    return os.path.join(stability_ROOT, kind, "%s.json" % "".join(ch if ch.isalnum() else "_" for ch in host))


def stability_task(job):
    path = stability_path("systems", job[0])
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(stability_system(job), open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    return json.load(open(path))


def stability_run(part=None, procs=12):
    """Every system of the part in a pool, the many-planet systems first, one JSON a system under ROOT/systems (resumable)."""
    import time
    from multiprocessing import Pool
    jobs = stability_jobs()
    if part:
        jobs = jobs[part[0]::part[1]]
    jobs = sorted(jobs, key=lambda j: -len(j[1]))
    t0 = time.time()
    with Pool(procs, initializer=stability_init) as pool:
        for i, r in enumerate(pool.imap_unordered(stability_task, jobs)):
            if (r["U"] or 0) > stability_FLAG or (i + 1) % 25 == 0:
                print("%d of %d, %.0f min: %s, %d planets, %d draws, unstable %s (%s)"
                      % (i + 1, len(jobs), (time.time() - t0) / 60, r["host"], r["n"], r["draws"],
                         "%.3f" % r["U"] if r["U"] is not None else "-", r["horizon"]), flush=True)
    print("%d systems done" % len(jobs))


def stability_search_task(job):
    """Stable draws of a flagged system the run kept too few of, from 4000 more draws."""
    import zlib
    s = stability_measured(job[1])
    return job[0], stability_eval(s, 4000, np.random.default_rng(zlib.crc32(job[0].encode()) + 1), None, 50)[2]


def stability_twin_task(task):
    """One stable twin: a stable draw observed again with the system's errors, and its unstable fraction by the same procedure."""
    import zlib
    host, g, t, X = task
    s = stability_measured(g)
    pl = s["planets"]
    rng = np.random.default_rng(zlib.crc32(host.encode()) + 7919 * (t + 1))

    def obs(x, up, lo):
        return float(stability_split(rng, x, up, lo, 1)[0])
    Y = dict(ms=max(obs(X["ms"], s["ms_up"], s["ms_lo"]), 0.05),
             m=[max(obs(X["m"][i], p["m_up"], p["m_lo"]), 0.05 * p["m"]) for i, p in enumerate(pl)],
             e=[min(max(obs(X["e"][i], p["e_up"], p["e_lo"]), 0.0), 0.9) for i, p in enumerate(pl)])
    return host, stability_system((host, g), centre=Y, seed=104729 * (t + 1), keep=0)["U"]


def stability_twins(procs=12, twins=10):
    """The null for every system over stability_FLAG unstable: twins, each a stable draw observed again, under ROOT/twins."""
    from multiprocessing import Pool
    jobs = dict(stability_jobs())
    recs = [json.load(open(stability_path("systems", h))) for h in jobs if os.path.exists(stability_path("systems", h))]
    fl = [r for r in recs if (r["U"] or 0) > stability_FLAG and not os.path.exists(stability_path("twins", r["host"]))]
    with Pool(procs, initializer=stability_init) as pool:
        extra = dict(pool.imap_unordered(stability_search_task, [(r["host"], jobs[r["host"]]) for r in fl if len(r["stable"]) < twins]))
        stable = {r["host"]: r["stable"] + extra.get(r["host"], []) for r in fl}
        tasks = [(h, jobs[h], t, st[t % len(st)]) for h, st in stable.items() if st for t in range(twins)]
        got = {h: [] for h in stable}
        for h, U in pool.imap_unordered(stability_twin_task, tasks):
            got[h].append(U)
            if len(got[h]) == twins:
                os.makedirs(os.path.dirname(stability_path("twins", h)), exist_ok=True)
                json.dump(dict(host=h, U=got[h], stable_draws=len(stable[h])), open(stability_path("twins", h), "w"))
    for h, st in stable.items():
        if not st:
            os.makedirs(os.path.dirname(stability_path("twins", h)), exist_ok=True)
            json.dump(dict(host=h, U=[], stable_draws=0), open(stability_path("twins", h), "w"))
    print("%d flagged systems: %d with twins, %d with no stable draw" % (len(fl), sum(1 for v in stable.values() if v),
                                                                          sum(1 for v in stable.values() if not v)))


def stability_long_task(task):
    """One draw of a flagged system integrated directly (WHFast) until two planets come within the smallest mutual Hill radius, a planet leaves, or stability_LONG orbits of the inner planet (1 per cent of the age, stability_LONG_STEPS steps if sooner): the instability time in days or None, and the window."""
    import zlib
    import rebound
    host, g, k = task
    s = stability_measured(g)
    P = [p["P"] for p in s["planets"]]
    ms, cols = stability_draws(s, 1, np.random.default_rng(zlib.crc32(host.encode()) + 31337 * (k + 1)))
    sim = stability_sim(ms, P, cols, 0)
    if stability_crossing(sim):
        return host, 0.0, 0.0
    ps = [sim.particles[i] for i in range(1, sim.N)]
    sim.integrator = "whfast"
    sim.dt = min(p.P * (1 - p.e) ** 1.5 for p in ps) / 20
    rh = min(((a.m + b.m) / (3 * sim.particles[0].m)) ** (1 / 3) * (a.a + b.a) / 2 for a, b in zip(ps[:-1], ps[1:]))
    sim.exit_min_distance = min(rh, 0.5 * ps[0].a * (1 - ps[0].e))
    sim.exit_max_distance = 100 * ps[-1].a * (1 + ps[-1].e)
    tmax = min(stability_LONG * P[0], 0.01 * s["age_used"] * 365.25e9, stability_LONG_STEPS * sim.dt)
    try:
        sim.integrate(tmax, exact_finish_time=0)
    except (rebound.Encounter, rebound.Escape):
        return host, float(sim.t), tmax
    return host, None, tmax


def stability_long(procs=12, draws=20):
    """Direct integrations of every system over stability_FLAG unstable, draws each, under ROOT/nbody (a check on SPOCK)."""
    from multiprocessing import Pool
    jobs = dict(stability_jobs())
    recs = [json.load(open(stability_path("systems", h))) for h in jobs if os.path.exists(stability_path("systems", h))]
    fl = [r["host"] for r in recs if (r["U"] or 0) > stability_FLAG and not os.path.exists(stability_path("nbody", r["host"]))]
    got = {h: [] for h in fl}
    with Pool(procs) as pool:
        for h, t, w in pool.imap_unordered(stability_long_task, [(h, jobs[h], k) for h in fl for k in range(draws)]):
            got[h].append((t, w))
            if len(got[h]) == draws:
                os.makedirs(os.path.dirname(stability_path("nbody", h)), exist_ok=True)
                json.dump(dict(host=h, t=[t for t, w in got[h]], window=[w for t, w in got[h]],
                               unstable=sum(1 for t, w in got[h] if t is not None) / draws), open(stability_path("nbody", h), "w"))
    print("%d flagged systems integrated, %d draws each" % (len(fl), draws))


def stability_vet(r, g=None, lim=None):
    """Why a flagged system may be mismeasured rather than maintained, or None; g are its archive rows, lim each planet's (mass, eccentricity) limit flags."""
    why = []
    if "M-R relationship" in r["prov"]:
        why.append("masses estimated from radii")
    if any(e is not None and e > 0.1 for e in r["ecc_err"]):
        why.append("eccentricity errors above 0.1")
    if g:
        ms, e0 = num(g[0]["st_mass"]) or 1.0, [num(x["pl_orbeccen"]) or 0.0 for x in g]
        a0 = [(ms * (num(x["pl_orbper"]) / 365.25) ** 2) ** (1 / 3) for x in g]
        if any(a2 * (1 - e2) < a1 * (1 + e1) for a1, a2, e1, e2 in zip(a0[:-1], a0[1:], e0[:-1], e0[1:])):
            why.append("orbits that cross at the archive's own values")
        if any((num(x["pl_orbeccen"]) or 0.0) > 0 and num(x["pl_orbeccenerr1"]) is None for x in g):
            why.append("eccentricities given without errors")
        if lim and any(lim.get(x["pl_name"], (0, 0))[0] == 1 for x in g):
            why.append("masses that are upper limits")
        if lim and any(lim.get(x["pl_name"], (0, 0))[1] == 1 for x in g):
            why.append("eccentricities that are upper limits")
    for p1, p2 in zip(r["periods"][:-1], r["periods"][1:]):
        x = p2 / p1
        if any(abs(x / (j / k) - 1) < 0.01 for j in range(2, 8) for k in range(1, j)):
            why.append("a pair within 1 per cent of a small-integer period ratio")
            break
    if r.get("twin_flagged") is not None and r["twin_flagged"] >= stability_TWIN:
        why.append("%d per cent of its stable twins flagged as well" % round(100 * r["twin_flagged"]))
    return "; ".join(why) or None


def stability_record():
    jobs = dict(stability_jobs())
    recs = [json.load(open(stability_path("systems", h))) for h in jobs if os.path.exists(stability_path("systems", h))]
    lim = {}
    if os.path.exists(os.path.join(stability_ROOT, "limits.csv")):
        lim = {x["pl_name"]: (int(num(x["pl_bmasselim"]) or 0), int(num(x["pl_orbeccenlim"]) or 0))
               for x in csv.DictReader(open(os.path.join(stability_ROOT, "limits.csv"), encoding="utf-8"))}
    for r in recs:
        if os.path.exists(stability_path("twins", r["host"])):
            tw = json.load(open(stability_path("twins", r["host"])))
            r["twins"], r["stable_draws"] = len(tw["U"]), tw["stable_draws"]
            r["twin_flagged"] = float(np.mean([(u or 0) > stability_FLAG for u in tw["U"]])) if tw["U"] else None
        if os.path.exists(stability_path("nbody", r["host"])):
            r["nbody"] = json.load(open(stability_path("nbody", r["host"])))["unstable"]
    fl = sorted([r for r in recs if (r["U"] or 0) > stability_FLAG], key=lambda r: -r["U"])
    for r in fl:
        r["vet"] = stability_vet(r, jobs[r["host"]], lim)
    left = [r for r in fl if not r["vet"]]
    for r in fl:
        print("  %-24s %d planets, %d draws, unstable %.4f (%s), crossing %d, twins %s, direct %s; %s"
              % (r["host"], r["n"], r["draws"], r["U"], r["horizon"], r["crossing"],
                 "%d of %d flagged" % (round(r["twin_flagged"] * r["twins"]), r["twins"]) if r.get("twin_flagged") is not None
                 else "none, no stable draw" if r.get("twins") == 0 else "not run",
                 "%.2f unstable" % r["nbody"] if r.get("nbody") is not None else "not run", r["vet"] or "OPEN"))
    tw = [r for r in fl if r.get("twin_flagged") is not None]
    core.record_search(
        "exoplanet_station_kept", "Station-kept planetary systems", archive="NASA Exoplanet Archive composite parameters",
        selection="%d multi-planet systems with periods and masses, %d of three or more planets, %d with transit or periastron "
                  "times for every planet" % (len(recs), sum(1 for r in recs if r["n"] >= 3), sum(1 for r in recs if r["phased"] == r["n"])),
        code=os.path.abspath(__file__), entry="stability", n_targets=len(recs),
        params=dict(draws=list(stability_DRAWS), more=stability_MORE, flag=stability_FLAG, age_missing_gyr=stability_AGE,
                    nbody_orbits=stability_NBODY, nbody_steps=stability_STEPS, regressor_draws=stability_REG, twin=stability_TWIN,
                    resonance=0.01, ecc_err=0.1),
        null_summary="stable draws of each flagged system observed again with its errors and drawn by the same procedure: %d of "
                     "%d flagged systems with twins have %d per cent or more of them flagged; %d flagged systems kept no stable "
                     "draw" % (sum(1 for r in tw if r["twin_flagged"] >= stability_TWIN), len(tw), round(100 * stability_TWIN),
                               sum(1 for r in fl if r.get("twins") == 0)),
        sensitivity="a system over 99 per cent of whose error draws go unstable within 1 per cent of its age (lower 1 sigma, 1 Gyr "
                    "where none is given): SPOCK's probability over 1e9 orbits of the inner planet, its regressor where that "
                    "outlasts 1 per cent of the age, Hill stability and 1e6 orbits of WHFast for two planets",
        ledger="Station-kept planetary systems",
        result="%d systems flagged (%d of three or more planets); %d explained by their measurements or their twins; %d open%s"
               % (len(fl), sum(1 for r in fl if r["n"] >= 3), len(fl) - len(left), len(left),
                  (": " + ", ".join(r["host"] for r in left)) if left else ""))
    core.clear_targets("exoplanet_station_kept")
    core.record_targets("exoplanet_station_kept", [dict(
        target="host:%s" % r["host"], stat=r["U"], unit="fraction of error draws unstable within 1 per cent of the age",
        vet=r.get("vet") if r in fl else None,
        note="%s%d planets, %d draws (%s), crossing in %d, minimum mutual Hill separation %.1f, age used %.2f Gyr%s"
             % ("flagged and unexplained; " if r in fl and not r.get("vet") else "", r["n"], r["draws"], r["horizon"], r["crossing"],
                r["min_hill"], r["age_used"],
                "; %.2f of direct integrations unstable" % r["nbody"] if r.get("nbody") is not None else "")) for r in recs])
    print("recorded %d systems; %d flagged, %d open" % (len(recs), len(fl), len(left)))


def stability_cli():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "limits", "run", "twins", "long", "record"))
    p.add_argument("--part", type=int, nargs=2, default=None)
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--twins", type=int, default=10)
    p.add_argument("--draws", type=int, default=20)
    a = p.parse_args()
    if a.mode == "fetch":
        stability_fetch()
    elif a.mode == "limits":
        stability_limits()
    elif a.mode == "run":
        stability_run(a.part, a.procs)
    elif a.mode == "twins":
        stability_twins(a.procs, a.twins)
    elif a.mode == "long":
        stability_long(a.procs, a.draws)
    else:
        stability_record()


hollow_DOC = r"""Hollow planets: densities below 1 g/cm^3 with escape parameter below 10 around hosts of 3 Gyr or more, young planets as the natural envelope.

    python -I seti.py transits hollow [record] [--lam 10] [--age-gyr 3] [--density 1.0] [--mass-snr 3]"""


def hollow_main(record=False, lam=10.0, age_gyr=3.0, density=1.0, mass_snr=3.0):
    p = core.data("hollow", "planets.csv")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        q = ("select pl_name,hostname,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_bmassprov,pl_rade,pl_radeerr1,pl_eqt,st_age,"
             "st_ageerr1,st_ageerr2,discoverymethod,pl_controv_flag from pscomppars where pl_bmasse is not null and "
             "pl_rade is not null and pl_eqt is not null")
        b = core.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(dict(query=q, format="csv")), timeout=600)
        open(p, "wb").write(b if isinstance(b, bytes) else b.encode("utf-8"))
    G, mH, kB, ME, RE = 6.674e-11, 1.6735e-27, 1.380649e-23, 5.972e24, 6.371e6
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        M, R, T, em = num(r["pl_bmasse"]), num(r["pl_rade"]), num(r["pl_eqt"]), num(r["pl_bmasseerr1"])
        if not (M and R and T and em) or r["pl_bmassprov"] != "Mass" or M / em < mass_snr:
            continue
        dens = 5.514 * M / R ** 3
        if dens >= density:
            continue
        out.append(dict(name=r["pl_name"], lam=G * M * ME * mH / (kB * T * R * RE), dens=dens, age=num(r["st_age"]), M=M, R=R, T=T))
    old = [o for o in out if o["age"] is not None and o["age"] >= age_gyr]
    cand = [o for o in old if o["lam"] < lam]
    young = sorted(out, key=lambda o: o["lam"])[:6]
    print("%d measured planets below %g g/cm^3; %d around stars of %g Gyr or more, lowest Lambda %.1f (%s); below %g: %d; the lowest "
          "overall: %s" % (len(out), density, len(old), age_gyr, min(o["lam"] for o in old), min(old, key=lambda o: o["lam"])["name"], lam, len(cand),
                           ", ".join("%s (%.1f, %s Gyr)" % (o["name"], o["lam"], o["age"]) for o in young)))
    if record:
        lo = min(old, key=lambda o: o["lam"])
        core.record_search(
            "exoplanet_hollow", "Hollow planets around old stars", archive="NASA Exoplanet Archive composite parameters",
            selection="%d planets with measured masses at %g sigma and bulk density below %g g/cm^3" % (len(out), mass_snr, density),
            code=os.path.abspath(__file__), entry="hollow", n_targets=len(out), params=dict(lam=lam, age_gyr=age_gyr, density=density, mass_snr=mass_snr),
            null_summary="the young planets: the lowest Lambda overall belongs to %s" % ", ".join(
                "%s (%s Gyr)" % (o["name"], o["age"]) for o in young[:4]),
            sensitivity="Lambda below %g with a %g sigma mass, around a star of %g Gyr or more" % (lam, mass_snr, age_gyr),
            ledger="Hollow planets around old stars",
            result="%d around stars of %g Gyr or more, %d below Lambda %g; the lowest is %s at %.1f" % (len(old), age_gyr, len(cand), lam, lo["name"], lo["lam"]))
        core.clear_targets("exoplanet_hollow")
        core.record_targets("exoplanet_hollow", [dict(
            target="planet:%s" % o["name"], stat=o["lam"], unit="escape parameter Lambda",
            note="density %.2f g/cm^3, host age %s Gyr" % (o["dens"], o["age"])) for o in out])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------

escape_DOC = r"""Transits faster than escape velocity: Kepler TCEs and TOIs whose duration is below 0.8 of the bound-orbit floor and below the smallest floor the errors allow.

    python -I seti.py transits escape fetch
    python -I seti.py transits escape analyse
    python -I seti.py transits escape check KIC [TCE]     (the light curve at every epoch of one event, with quality flags)
    python -I seti.py transits escape record"""

escape_ROOT = os.environ.get("ESCAPE_ROOT", core.data("escape_transits"))
escape_OUT = "escape_transits.json"
escape_SNR = 15.0
escape_RATIO = 0.8
escape_S3 = "https://stpubdata.s3.amazonaws.com/"
escape_G, escape_MSUN, escape_RSUN = 6.674e-11, 1.989e30, 6.957e8


def escape_fetch():
    os.makedirs(escape_ROOT, exist_ok=True)
    jobs = (("dr25_tce.csv", "SELECT kepid, tce_plnt_num, tce_period, tce_time0bk, tce_duration, tce_depth, tce_impact, tce_model_snr, "
                             "tce_max_mult_ev, tce_prad, tce_ror, tce_dor, tce_steff, tce_slogg, tce_sradius, tce_rogue_flag FROM q1_q17_dr25_tce"),
            ("koi_cumulative.csv", "SELECT kepid, kepoi_name, koi_disposition, koi_pdisposition, koi_score, koi_fpflag_nt, koi_fpflag_ss, "
                                   "koi_fpflag_co, koi_fpflag_ec, koi_period, koi_duration, koi_duration_err1, koi_impact, koi_impact_err1, "
                                   "koi_impact_err2, koi_depth, koi_prad, koi_ror, koi_dor, koi_model_snr, koi_srad, koi_smass, koi_steff, "
                                   "koi_slogg, koi_count, koi_num_transits FROM cumulative"),
            ("toi.csv", "SELECT toi, tid, tfopwg_disp, pl_orbper, pl_trandurh, pl_trandurherr1, pl_trandep, pl_rade, st_rad, st_raderr1, "
                        "st_teff, st_logg, st_tmag, st_dist, ra, dec FROM toi"))
    for name, q in jobs:
        p = os.path.join(escape_ROOT, name)
        if os.path.exists(p):
            continue
        rows = core.exoplanets(q)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("%s: %d rows" % (name, len(rows)))
    p = os.path.join(escape_ROOT, "berger2020_table2.ecsv")
    if not os.path.exists(p):
        from astroquery.vizier import Vizier
        t = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/AJ/159/280/table2")[0]
        t.write(p, format="ascii.ecsv", overwrite=True)
        print("Berger et al. (2020) table 2: %d stars" % len(t))


def escape_floor_hours(R, M, b, k):
    """Shortest bound full transit (hours) across a star of R (R_sun) and M (M_sun) at impact parameter b and radius ratio k."""
    if R is None or M is None or R <= 0 or M <= 0:
        return None
    v = math.sqrt(2 * escape_G * M * escape_MSUN / (R * escape_RSUN))
    chord = 2 * R * escape_RSUN * math.sqrt(max((1 + k) ** 2 - b ** 2, 0.0))
    return chord / v / 3600.0


def escape_analyse():
    from astropy.table import Table
    t = Table.read(os.path.join(escape_ROOT, "berger2020_table2.ecsv"), format="ascii.ecsv")
    star = {}
    for r in t:
        star[int(r["KIC"])] = dict(R=float(r["Rad"]), R_lo=float(r["e_Rad"]), R_hi=float(r["E_Rad"]), M=float(r["Mass"]),
                                   M_lo=float(r["e_Mass"]), M_hi=float(r["E_Mass"]), logg=float(r["logg"]), teff=float(r["Teff"]))
    kois = {}
    for r in csv.DictReader(open(os.path.join(escape_ROOT, "koi_cumulative.csv"))):
        kois.setdefault(int(r["kepid"]), []).append(r)
    tces = list(csv.DictReader(open(os.path.join(escape_ROOT, "dr25_tce.csv"))))
    n_tce, by_kic = {}, {}
    for r in tces:
        n_tce[int(r["kepid"])] = n_tce.get(int(r["kepid"]), 0) + 1
        by_kic.setdefault(int(r["kepid"]), []).append(r)
    eb_path = core.data("kepler_eb_kirk2016.json")
    ebs = {int(k): v for k, v in json.load(open(eb_path)).items()} if os.path.exists(eb_path) else {}
    rows = []
    for r in tces:
        kic = int(r["kepid"])
        s = star.get(kic)
        snr, dur, b, k = num(r["tce_model_snr"]), num(r["tce_duration"]), num(r["tce_impact"]), num(r["tce_ror"])
        if s is None or snr is None or dur is None or b is None or k is None or num(r["tce_rogue_flag"]):
            continue
        b, k = min(max(b, 0.0), 1.0), max(k, 0.0)
        f_b = escape_floor_hours(s["R"], s["M"], b, k)
        f_0 = escape_floor_hours(s["R"], s["M"], 0.0, k)
        f_min = escape_floor_hours(max(s["R"] - s["R_lo"], 0.01), s["M"] + s["M_hi"], b, k)
        koi = None
        for q in kois.get(kic, []):
            if num(q["koi_period"]) and abs(num(q["koi_period"]) - num(r["tce_period"])) < 1e-3 * num(r["tce_period"]):
                koi = q
        row = dict(kic=kic, tce=int(r["tce_plnt_num"]), period=num(r["tce_period"]), duration_h=dur, b=b, k=k, snr=snr,
                   R=s["R"], M=s["M"], logg=s["logg"], teff=s["teff"], floor_b=f_b, floor_0=f_0, floor_min=f_min,
                   ratio_b=dur / f_b, ratio_0=dur / f_0, ratio_min=dur / f_min, koi=koi["kepoi_name"] if koi else None,
                   disposition=koi["koi_disposition"] if koi else None,
                   flags=("".join(("N" if num(koi["koi_fpflag_nt"]) else "", "S" if num(koi["koi_fpflag_ss"]) else "",
                                   "C" if num(koi["koi_fpflag_co"]) else "", "E" if num(koi["koi_fpflag_ec"]) else "")) if koi else None),
                   koi_duration_h=num(koi["koi_duration"]) if koi else None, koi_impact=num(koi["koi_impact"]) if koi else None,
                   depth_ppm=num(r["tce_depth"]), n_tce=n_tce[kic], eb_period=ebs.get(kic))
        # natural readings: giant host, hot-star pulsator, catalogued EB, Robovetter flag, or a one- or two-cadence event to check
        same_period = sum(1 for q in by_kic[kic] if abs(float(q["tce_period"]) - row["period"]) < 1e-3 * row["period"])
        if row["logg"] < 4.0:
            row["reading"] = "giant host"
        elif row["eb_period"]:
            row["reading"] = "eclipsing binary (Kirk et al. 2016, P %.3f d)%s" % (row["eb_period"], ", shallow: blended" if (row["depth_ppm"] or 0) < 2000 else "")
        elif row["n_tce"] >= 5:
            row["reading"] = "star with %d events: variability" % row["n_tce"]
        elif row["teff"] > 6500 and row["n_tce"] >= 3:
            row["reading"] = "hot star with %d events: pulsator" % row["n_tce"]
        elif same_period >= 2:
            row["reading"] = "two events at one period: an eclipsing binary%s" % (", shallow: blended" if (row["depth_ppm"] or 0) < 2000 else "")
        elif row["flags"] and row["flags"].strip():
            row["reading"] = "KOI false-positive flags %s" % row["flags"]
        else:
            row["reading"] = "unexplained by the catalogues: light curve to check"
        rows.append(row)
    good = [x for x in rows if x["snr"] >= escape_SNR]
    ratios = np.array([x["ratio_b"] for x in good])
    print("%d DR25 TCEs with Berger et al. (2020) hosts, %d at S/N %.0f or more: duration over the floor at the fitted geometry, "
          "percentiles 1/5/50/95: %s; below 1: %d; below %.1f: %d; below %.1f within the errors too: %d"
          % (len(rows), len(good), escape_SNR, np.percentile(ratios, [1, 5, 50, 95]).round(2), int((ratios < 1).sum()),
             escape_RATIO, int((ratios < escape_RATIO).sum()), escape_RATIO,
             sum(1 for x in good if x["ratio_b"] < escape_RATIO and x["ratio_min"] < escape_RATIO)))
    dwarfs = [x for x in good if x["logg"] >= 4.0]
    print("  dwarfs (log g >= 4): %d, below 1: %d; giants: %d, below 1: %d" % (
        len(dwarfs), sum(1 for x in dwarfs if x["ratio_b"] < 1), len(good) - len(dwarfs), sum(1 for x in good if x["logg"] < 4.0 and x["ratio_b"] < 1)))
    cands = sorted((x for x in good if x["ratio_b"] < escape_RATIO and x["ratio_min"] < escape_RATIO), key=lambda x: x["ratio_b"])
    readings = {}
    for x in cands:
        x["clean"] = x["reading"].startswith("unexplained")
        readings[x["reading"].split(" (")[0].split(":")[0]] = readings.get(x["reading"].split(" (")[0].split(":")[0], 0) + 1
    print("  readings: " + "; ".join("%s %d" % kv for kv in sorted(readings.items(), key=lambda kv: -kv[1])))
    for x in (y for y in cands if y["logg"] >= 4.0):
        print("  KIC %d TCE %d%s: P %.3f d, T14 %.2f h, depth %.0f ppm, b %.2f, k %.3f, S/N %.0f; host R %.2f M %.2f log g %.2f Teff %.0f; floor %.2f h "
              "(b 0: %.2f, errors: %.2f); ratio %.2f (b 0: %.2f); %s %s; %s" % (
                  x["kic"], x["tce"], " = %s" % x["koi"] if x["koi"] else "", x["period"], x["duration_h"], x["depth_ppm"] or 0, x["b"], x["k"], x["snr"],
                  x["R"], x["M"], x["logg"], x["teff"], x["floor_b"], x["floor_0"], x["floor_min"], x["ratio_b"], x["ratio_0"],
                  x["disposition"] or "not a KOI", "flags %s" % x["flags"] if x["flags"] else "", x["reading"]))
    # TOIs at b = 0, mass from radius: M = R^1.25 below 1.5 R_sun, R^0.6 above
    tois = []
    for r in csv.DictReader(open(os.path.join(escape_ROOT, "toi.csv"), encoding="utf-8", errors="replace")):
        R, dur, prad = num(r["st_rad"]), num(r["pl_trandurh"]), num(r["pl_rade"])
        if not R or not dur or R <= 0:
            continue
        M = R ** 1.25 if R < 1.5 else 1.5 ** 1.25 * (R / 1.5) ** 0.6
        k = (prad or 0.0) * 6371.0 / (R * 695700.0)
        f0 = escape_floor_hours(R, M, 0.0, k)
        row = dict(toi=r["toi"], tic=r["tid"], disp=r["tfopwg_disp"], period=num(r["pl_orbper"]), duration_h=dur, R=R, M=M, k=k,
                   logg=num(r["st_logg"]), teff=num(r["st_teff"]), floor_0=f0, ratio_0=dur / f0)
        # natural readings: giant host, published false positive, sub-day pulsator or contact binary, else check the light curve
        if R >= 3.0:
            row["reading"] = "giant host"
        elif row["disp"] in ("FP", "FA"):
            row["reading"] = "false positive or alarm (%s)" % row["disp"]
        elif row["period"] and row["period"] < 1.0 and (row["teff"] or 0) > 6300:
            row["reading"] = "sub-day period on a hot star: pulsator"
        elif row["period"] and row["period"] < 1.0:
            row["reading"] = "sub-day period: contact binary or pulsator"
        else:
            row["reading"] = "no impact parameter in the TOI table: a grazing geometry is not excluded"
        tois.append(row)
    rt = np.array([x["ratio_0"] for x in tois])
    print("%d TOIs with a host radius and a duration: duration over the b = 0 floor, percentiles 1/5/50/95: %s; below 1: %d; below %.1f: %d"
          % (len(tois), np.percentile(rt, [1, 5, 50, 95]).round(2), int((rt < 1).sum()), escape_RATIO, int((rt < escape_RATIO).sum())))
    tc = sorted((x for x in tois if x["ratio_0"] < escape_RATIO), key=lambda x: x["ratio_0"])
    tr = {}
    for x in tc:
        tr[x["reading"]] = tr.get(x["reading"], 0) + 1
    print("  readings: " + "; ".join("%s %d" % kv for kv in sorted(tr.items(), key=lambda kv: -kv[1])))
    for x in [y for y in tc if y["reading"].startswith("no impact")][:12]:
        print("  TOI %s (TIC %s, %s): P %s d, T14 %.2f h; host R %.2f (M %.2f, log g %s, Teff %s); floor %.2f h; ratio %.2f" % (
            x["toi"], x["tic"], x["disp"], "%.3f" % x["period"] if x["period"] else "-", x["duration_h"], x["R"], x["M"],
            "%.2f" % x["logg"] if x["logg"] else "-", "%.0f" % x["teff"] if x["teff"] else "-", x["floor_0"], x["ratio_0"]))
    json.dump(dict(n_tce=len(rows), n_good=len(good), snr=escape_SNR, ratio=escape_RATIO,
                   percentiles=dict(zip(("p1", "p5", "p50", "p95"), [float(v) for v in np.percentile(ratios, [1, 5, 50, 95])])),
                   below_1=int((ratios < 1).sum()), candidates=cands, n_toi=len(tois),
                   toi_percentiles=dict(zip(("p1", "p5", "p50", "p95"), [float(v) for v in np.percentile(rt, [1, 5, 50, 95])])),
                   toi_candidates=tc), open(core.results(escape_OUT), "w"), indent=1)


def escape_check(kic, tce_num):
    """Kepler long-cadence flux and quality flags within two hours of each predicted epoch, normalised by quarter: lone flagged cadence or recurring transit."""
    import io
    from astropy.io import fits
    r = next(x for x in csv.DictReader(open(os.path.join(escape_ROOT, "dr25_tce.csv"))) if int(x["kepid"]) == kic and int(x["tce_plnt_num"]) == tce_num)
    P, t0, dur = float(r["tce_period"]), float(r["tce_time0bk"]), float(r["tce_duration"]) / 24.0
    prefix = "kepler/public/lightcurves/%04d/%09d/" % (kic // 100000, kic)
    b = core.get(escape_S3 + "?list-type=2&prefix=" + prefix, timeout=60, retries=2)
    keys = re.findall(r"<Key>([^<]*_llc\.fits)</Key>", b.decode())
    T, F, Q = [], [], []
    for key in keys:
        fb = core.get(escape_S3 + key, timeout=120, retries=2, soft=True)
        if not fb:
            continue
        with fits.open(io.BytesIO(fb)) as h:
            d = h[1].data
            f = np.asarray(d["PDCSAP_FLUX"], float)
            T.append(np.asarray(d["TIME"], float))
            F.append(f / np.nanmedian(f))
            Q.append(np.asarray(d["SAP_QUALITY"], int))
    T, F, Q = (np.concatenate(x) for x in (T, F, Q))
    ok = np.isfinite(T) & np.isfinite(F)
    print("KIC %d TCE %d: P %.4f d, epoch BKJD %.3f, duration %.2f h; %d quarters, %d cadences" % (kic, tce_num, P, t0, dur * 24, len(keys), ok.sum()))
    for n in range(int((np.nanmin(T[ok]) - t0) / P) - 1, int((np.nanmax(T[ok]) - t0) / P) + 2):
        tc = t0 + n * P
        near = ok & (np.abs(T - tc) < 2.0 / 24)
        if not near.any():
            continue
        print("  epoch %+d (BKJD %.3f): flux within 2 h %s; quality flags %s" % (
            n, tc, " ".join("%.3f" % v for v in F[near]), sorted({int(q) for q in Q[near]})))


escape_SHAPES = "escape_toi_shapes.json"


def escape_trapezoid(x, delta, t14, t23, exp):
    """Trapezoid transit (relative flux) at times x from mid-transit, averaged over exposure exp."""
    sub = np.linspace(-0.5, 0.5, 9)
    xx = x[:, None] + exp[:, None] * sub[None, :]
    a = np.abs(xx)
    ramp = np.clip((t14 / 2 - a) / max((t14 - t23) / 2, 1e-6), 0.0, 1.0)
    return 1 - delta * np.mean(ramp, axis=1)


def escape_fit_shape(x, f, e, exp, t14_guess, ratio=None):
    """Least-squares trapezoid (depth, T14, T23/T14, shift) of folded transits, T23/T14 held at ratio when given: (chi2, parameters)."""
    from scipy.optimize import least_squares

    def resid(p):
        d, t14, r, s = p if ratio is None else (p[0], p[1], ratio, p[2])
        return (f - escape_trapezoid(x - s, d, t14, r * t14, exp)) / e
    best = None
    for t14 in (0.6 * t14_guess, t14_guess, 1.6 * t14_guess):
        for r0 in ((0.1, 0.5, 0.8) if ratio is None else (ratio,)):
            depth0 = max(1 - float(np.median(f[np.abs(x) < 0.3 * t14])), 1e-5)
            p0 = [depth0, t14, r0, 0.0] if ratio is None else [depth0, t14, 0.0]
            lo = [0, 0.2 * t14_guess, 0, -0.5 * t14_guess] if ratio is None else [0, 0.2 * t14_guess, -0.5 * t14_guess]
            hi = [0.5, 5 * t14_guess, 0.98, 0.5 * t14_guess] if ratio is None else [0.5, 5 * t14_guess, 0.5 * t14_guess]
            try:
                r = least_squares(resid, np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9), bounds=(lo, hi))
            except ValueError:
                continue
            c = float(np.sum(r.fun ** 2))
            if best is None or c < best[0]:
                best = (c, list(r.x) if ratio is None else [r.x[0], r.x[1], ratio, r.x[2]])
    return best


def escape_shapes():
    """Trapezoid fits of folded TESS transits of sub-floor TOIs without b: the smallest flat-bottom fraction within Delta chi2 9 bounds b and the slowest crossing; a V shape leaves a grazing chord."""
    d = json.load(open(core.results(escape_OUT)))
    cands = [x for x in d["toi_candidates"] if x["reading"].startswith("no impact")]
    eph = {r["toi"]: r for r in core.exoplanets("SELECT toi, pl_tranmid, pl_orbper, pl_trandurh, ra, dec FROM toi WHERE toi IN (%s)"
                                                       % ", ".join(x["toi"] for x in cands))}

    def one(x):
        from astropy.io import fits
        e = eph[x["toi"]]
        P, t0, dur = float(e["pl_orbper"]), float(e["pl_tranmid"]) - 2457000.0, float(e["pl_trandurh"]) / 24.0
        half = max(3 * dur, 2.0 / 24)
        X, F, E, EXP, used = [], [], [], [], []
        for s in core.sectors(float(e["ra"]), float(e["dec"])):
            src, b = "spoc", core.spoc_bytes(int(x["tic"]), s)
            if not b:
                src, b = core.ffi_bytes(int(x["tic"]), s)
            if not b:
                continue
            with fits.open(io.BytesIO(b)) as h:
                exp = float(h[1].header.get("TIMEDEL") or 0.0208333)
            t, f, _ = core.read(b)
            if len(t) < 100:
                continue
            n_used = 0
            for n in range(int(np.floor((t.min() - t0) / P)), int(np.ceil((t.max() - t0) / P)) + 1):
                dt = t - (t0 + n * P)
                w = np.abs(dt) < half
                side = w & (np.abs(dt) > max(1.5 * dur, dur + 0.5 / 24))
                if (side & (dt < 0)).sum() < 3 or (side & (dt > 0)).sum() < 3 or (w & (np.abs(dt) < 0.5 * dur)).sum() < 1:
                    continue
                a, c = np.polyfit(dt[side], f[side], 1)
                r = f[w] / (a * dt[w] + c)
                sd = 1.4826 * np.median(np.abs(r[side[w]] - np.median(r[side[w]])))
                X.append(dt[w] * 24)
                F.append(r)
                E.append(np.full(w.sum(), max(sd, 1e-6)))
                EXP.append(np.full(w.sum(), exp * 24))
                n_used += 1
            if n_used:
                used.append((s, src, n_used, round(exp * 1440, 1)))
        if not X:
            return dict(toi=x["toi"], tic=x["tic"], sectors=[], verdict="no light curve")
        X, F, E, EXP = (np.concatenate(v) for v in (X, F, E, EXP))
        best = escape_fit_shape(X, F, E, EXP, dur * 24)
        vshape = escape_fit_shape(X, F, E, EXP, dur * 24, ratio=0.0)
        out = dict(toi=x["toi"], tic=x["tic"], disp=x["disp"], P=P, T14_toi=dur * 24, R=x["R"], M=x["M"], floor_0=x["floor_0"],
                   sectors=used, n_points=int(len(X)), chi2=round(best[0], 1), dof=int(len(X) - 4),
                   depth=best[1][0], T14=best[1][1], ratio23=best[1][2], shift_h=best[1][3], dchi2_v=round(vshape[0] - best[0], 1))
        # smallest flat-bottom fraction within Delta chi2 9
        rmin = 0.0
        if out["dchi2_v"] > 9:
            lo, hi = 0.0, out["ratio23"]
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if escape_fit_shape(X, F, E, EXP, dur * 24, ratio=mid)[0] - best[0] > 9:
                    lo = mid
                else:
                    hi = mid
            rmin = hi
        k = math.sqrt(max(out["depth"], 0.0))
        out["k"], out["ratio23_min"] = k, rmin
        snr = out["depth"] / (float(np.median(E)) / math.sqrt(max(int(np.sum(np.abs(X - out["shift_h"]) < out["T14"] / 2)), 1)))
        out["snr"] = round(snr, 1)
        if rmin <= 0:
            out["b_max"] = 1 + k
            out["verdict"] = "grazing allowed by the light curve: a V shape within Delta chi2 %.1f of the best (T14 %.2f h, S/N %.0f)" % (out["dchi2_v"], out["T14"], snr)
        else:
            b2 = ((1 - k) ** 2 - rmin ** 2 * (1 + k) ** 2) / (1 - rmin ** 2)
            out["b_max"] = math.sqrt(max(b2, 0.0))
            v_esc = math.sqrt(2 * escape_G * x["M"] * escape_MSUN / (x["R"] * escape_RSUN))
            chord = 2 * x["R"] * escape_RSUN * math.sqrt(max((1 + k) ** 2 - out["b_max"] ** 2, 0.0))
            out["v_over_esc"] = chord / (out["T14"] * 3600) / v_esc
            out["verdict"] = ("flat-bottomed (T23/T14 %.2f, at least %.2f), b at most %.2f: the slowest crossing is %.2f of the escape speed (T14 %.2f h, S/N %.0f)"
                              % (out["ratio23"], rmin, out["b_max"], out["v_over_esc"], out["T14"], snr))
        print("TOI %s (TIC %s, %s): %s" % (x["toi"], x["tic"], x["disp"], out["verdict"]), flush=True)
        return out
    res = core.pmap(one, cands, threads=2)
    json.dump(res, open(core.results(escape_SHAPES), "w"), indent=1)


def escape_shape_reading(f, c):
    """Reading of TOI c from fit f: no transit below S/N 7, bound duration at b = 0, V shape (grazing), flat bottom within 1/RATIO of escape, else None."""
    if not f or f.get("verdict") == "no light curve":
        return None
    if (f.get("snr") or 0) < 7:
        return "transit not recovered at the TOI ephemeris (S/N %.1f in %d sectors)" % (f.get("snr") or 0, len(f["sectors"]))
    R = c["R"]
    M = 10 ** (c["logg"] - 4.438) * R ** 2 if c.get("logg") else c["M"]
    v_esc = math.sqrt(2 * escape_G * M * escape_MSUN / (R * escape_RSUN))
    k = f["k"]

    def speed(b):
        return 2 * R * escape_RSUN * math.sqrt(max((1 + k) ** 2 - b ** 2, 0.0)) / (f["T14"] * 3600) / v_esc
    told = "T14 %.2f h in the light curve against %.2f in the TOI table, host %.2f Rsun and %.2f Msun%s" % (
        f["T14"], f["T14_toi"], R, M, " from its log g" if c.get("logg") else "")
    if speed(0.0) <= 1 / escape_RATIO:
        return "the light curve's duration allows a bound orbit at any impact parameter: %s, %.2f of the escape speed at b 0" % (told, speed(0.0))
    if f["ratio23_min"] <= 0:
        return "a V shape the light curve allows: a grazing geometry is not excluded (%s)" % told
    if speed(f["b_max"]) <= 1 / escape_RATIO:
        return "the light curve allows b up to %.2f, where the crossing is %.2f of the escape speed: a grazing geometry is not excluded (%s)" % (
            f["b_max"], speed(f["b_max"]), told)
    return None


# Kepler events whose light curve escape_check read
escape_FOLLOWUP = {7292626: "a single cadence flagged for cosmic rays with the other epochs flat"}
escape_FAST = "flat-bottomed and faster than the escape speed at every impact parameter the light curve allows"


def escape_record():
    d = json.load(open(core.results(escape_OUT)))
    clean = [x for x in d["candidates"] if x.get("clean")]
    for x in d["candidates"]:
        x["reading"] = escape_FOLLOWUP.get(x["kic"], x["reading"])
    shapes = {f["toi"]: f for f in json.load(open(core.results(escape_SHAPES)))} if os.path.exists(core.results(escape_SHAPES)) else {}
    told = collections.Counter()
    for x in d["toi_candidates"]:
        f = shapes.get(x["toi"])
        if not x["reading"].startswith("no impact") or not f:
            continue
        if f.get("verdict") == "no light curve":
            # no public light curve: the catalogue's reading stands
            told["no TESS light curve in the public stores"] += 1
            continue
        x["reading"] = escape_shape_reading(f, x) or escape_FAST
        told[next((label for key, label in (("the light curve's duration", "a duration a bound orbit gives at any impact parameter"),
                                            ("a V shape", "a V shape, which leaves a grazing chord"),
                                            ("the light curve allows b", "a flat bottom whose largest allowed impact parameter keeps the crossing below the escape speed"),
                                            ("transit not recovered", "no transit at the TOI ephemeris"))
                   if x["reading"].startswith(key)), "faster than the escape speed at every allowed impact parameter")] += 1
    shape_note = ("; the TESS light curves of the %d without an impact parameter: %s" % (
        sum(told.values()), ", ".join("%d %s" % (n, k) for k, n in told.most_common()))) if told else ""
    core.record_search(
        "escape_transits", "Transits faster than escape velocity",
        archive="NASA Exoplanet Archive: Kepler DR25 TCE table, cumulative KOI table, TOI table; Berger et al. (2020) Gaia-Kepler stellar properties",
        release="DR25; KOI and TOI tables as served", selection="every DR25 threshold-crossing event with a Berger et al. (2020) host; every TOI with a host radius",
        code=os.path.abspath(__file__), entry="escape", n_targets=d["n_good"] + d["n_toi"],
        params=dict(snr_min=d["snr"], ratio_flag=d["ratio"], mass_radius_toi="M = R^1.25 below 1.5 R_sun"),
        null_summary="duration over the escape-velocity floor for the rest of the events: percentiles %s (Kepler), %s (TOI at b = 0)" % (
            d["percentiles"], d["toi_percentiles"]),
        sensitivity="a full transit shorter than 0.8 of 37.5 min sqrt(1 - b^2) R^1.5 M^-0.5 at S/N 15",
        ledger="Transits faster than escape velocity",
        result="%d Kepler events at S/N %.0f: %d below the floor at the fitted geometry, %d below %.1f of it within the host errors as well, "
               "%d of those KOI candidates without centroid, eclipse or ephemeris flags; %d TOIs: %d below %.1f of the b = 0 floor%s" % (
                   d["n_good"], d["snr"], d["below_1"], len(d["candidates"]), d["ratio"], len(clean), d["n_toi"],
                   len(d["toi_candidates"]), d["ratio"], shape_note))
    core.clear_targets("escape_transits")
    rows = [dict(target="kic:%d" % x["kic"], stat=x["ratio_b"], unit="transit duration over the escape-velocity floor",
                 vet=x["reading"] + ((" (KOI %s, %s)" % (x["koi"], x["disposition"])) if x["koi"] else ""),
                 note="TCE %d, P %.3f d, T14 %.2f h, depth %.0f ppm, b %.2f, host R %.2f M %.2f log g %.2f, %d events on the star" % (
                     x["tce"], x["period"], x["duration_h"], x["depth_ppm"] or 0, x["b"], x["R"], x["M"], x["logg"], x["n_tce"]))
            for x in d["candidates"]]
    rows += [dict(target="tic:%s" % x["tic"], stat=x["ratio_0"], unit="transit duration over the escape-velocity floor at b = 0",
                  vet=None if x["reading"] == escape_FAST else "%s (TOI %s, %s)" % (x["reading"], x["toi"], x["disp"]),
                  note="%sP %s d, T14 %.2f h, host R %.2f, M %.2f from the radius" % (
                      "%s (TOI %s, %s); " % (x["reading"], x["toi"], x["disp"]) if x["reading"] == escape_FAST else "",
                      "%.3f" % x["period"] if x["period"] else "-", x["duration_h"], x["R"], x["M"]))
             for x in d["toi_candidates"]]
    core.record_targets("escape_transits", rows)
    print("recorded %d candidates" % len(rows))


def escape_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if mode == "check":
        escape_check(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
        return
    {"fetch": escape_fetch, "analyse": escape_analyse, "shapes": escape_shapes, "record": escape_record}[mode]()


aimed_DOC = r"""Aimed transits: excess of Kepler KOIs at b below 0.05 and 0.1 against DR25 injection fractions, all planets and the habitable zone.

    python -I seti.py transits aimed analyse
    python -I seti.py transits aimed record"""

aimed_ROOT = os.environ.get("AIMED_ROOT", core.data("aimed_transits"))
aimed_INJ = "https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj1_plti.txt"
aimed_OUT = "aimed_transits.json"
aimed_SNR, aimed_BERR = 20.0, 0.1
aimed_BINS = ((0.0, 0.05), (0.05, 0.1), (0.1, 0.5))


def aimed_analyse():
    from astropy.io import ascii as asc
    os.makedirs(aimed_ROOT, exist_ok=True)
    p = os.path.join(aimed_ROOT, "kplr_dr25_inj1_plti.txt")
    if not os.path.exists(p):
        open(p, "wb").write(core.get(aimed_INJ, timeout=900, retries=2))
    inj = asc.read(p, format="ipac")
    rec = inj[(inj["Recovered"] == 1) & (inj["EB_injection"] == 0) & (inj["Offset_from_source"] == 0)]
    r_b = np.array(rec["r_b"], float)
    i_b = np.array(rec["i_b"], float)
    mes = np.array(rec["Measured_MES"], float)
    ok = np.isfinite(r_b) & np.isfinite(i_b) & (mes >= aimed_SNR)
    print("%d injected planets on their own stars, %d recovered, %d with fitted b at MES %.0f or more; injected b: %.3f below 0.05, %.3f below 0.1, "
          "%.3f in 0.1-0.5 (a uniform population gives 0.05, 0.10, 0.40)" % (
              len(inj), len(rec), ok.sum(), aimed_SNR, np.mean(i_b[ok] < 0.05), np.mean(i_b[ok] < 0.1), np.mean((i_b[ok] >= 0.1) & (i_b[ok] < 0.5))))
    inj_frac = {str(b): float(np.mean((r_b[ok] >= b[0]) & (r_b[ok] < b[1]))) for b in aimed_BINS}
    print("  recovered (fitted) b of those: " + ", ".join("%s: %.3f" % kv for kv in inj_frac.items()))
    kois = []
    for r in csv.DictReader(open(os.path.join(escape_ROOT, "koi_cumulative.csv"))):
        b, e1, e2, snr = num(r["koi_impact"]), num(r["koi_impact_err1"]), num(r["koi_impact_err2"]), num(r["koi_model_snr"])
        if b is None or snr is None or r["koi_disposition"] not in ("CANDIDATE", "CONFIRMED"):
            continue
        err = max(abs(e1 or 9), abs(e2 or 9))
        kois.append(dict(koi=r["kepoi_name"], kic=int(r["kepid"]), b=b, err=err, snr=snr, disp=r["koi_disposition"], insol=num(r["koi_insol"]),
                         prad=num(r["koi_prad"]), period=num(r["koi_period"]), srad=num(r["koi_srad"])))
    # no error cut: one pipeline fitted both, and a cut selects the best-constrained central and grazing fits
    good = [k for k in kois if k["snr"] >= aimed_SNR]
    hz = [k for k in good if k["insol"] is not None and 0.32 <= k["insol"] <= 1.78]
    out = dict(n_inj=int(ok.sum()), inj_frac=inj_frac, n_koi=len(kois), n_good=len(good), n_hz=len(hz), samples={})
    for name, sample in (("all", good), ("habitable zone", hz), ("other", [k for k in good if k not in hz]),
                         ("all, b error below %.1f" % aimed_BERR, [k for k in good if k["err"] < aimed_BERR])):
        bs = np.array([k["b"] for k in sample])
        counts = {str(b): int(((bs >= b[0]) & (bs < b[1])).sum()) for b in aimed_BINS}
        rows = {}
        for b in aimed_BINS:
            f = inj_frac[str(b)]
            expected = len(sample) * f
            obs = counts[str(b)]
            # binomial significance of the count against the injections' fraction
            sig = (obs - expected) / math.sqrt(expected * (1 - f)) if expected > 0 else float("nan")
            rows[str(b)] = dict(observed=obs, expected=expected, sigma=sig)
        out["samples"][name] = dict(n=len(sample), counts=counts, tests=rows)
        print("%s: %d planets at S/N %.0f; against the injections' fractions: %s" % (
            name, len(sample), aimed_SNR, "; ".join("%s: %d observed, %.1f expected (%+.1f sigma)" % (k, v["observed"], v["expected"], v["sigma"]) for k, v in rows.items())))
    low = sorted((k for k in hz if k["b"] < 0.05), key=lambda k: k["b"])
    print("habitable-zone planets below b 0.05: " + ", ".join("%s (b %.3f +- %.3f, %.1f R_earth, %.0f d)" % (k["koi"], k["b"], k["err"], k["prad"] or 0, k["period"] or 0) for k in low))
    out["hz_low"] = low
    # b errors of the whole sample's central fits
    errs = np.array([k["err"] for k in good if k["b"] < 0.05])
    out["low_all"] = dict(n=len(errs), median_err=float(np.median(errs)), above_013=int((errs > 0.13).sum()))
    print("all planets below b 0.05: %d, median b error %.2f, %d with errors above 0.13" % (len(errs), np.median(errs), (errs > 0.13).sum()))
    json.dump(out, open(core.results(aimed_OUT), "w"), indent=1)


def aimed_record():
    d = json.load(open(core.results(aimed_OUT)))
    s = d["samples"]
    core.record_search(
        "aimed_transits", "Aimed transits", archive="NASA Exoplanet Archive: cumulative KOI table; Kepler DR25 injected-planet recoveries (kplr_dr25_inj1_plti)",
        release="cumulative table as served; DR25 injections", selection="KOI candidates and confirmed planets at S/N 20 or more",
        code=os.path.abspath(__file__), entry="aimed", n_targets=d["n_good"], params=dict(snr_min=aimed_SNR, bins=aimed_BINS),
        null_summary="the fitted impact parameters of %d recovered injected planets with uniform b at MES 20 or more: fractions %s" % (d["n_inj"], d["inj_frac"]),
        sensitivity="an excess at b below 0.05 of a few per cent of the sample, 3 sigma at about 35 planets",
        ledger="Aimed transits",
        # the constrained-b test decides; the whole sample's excess is lower-bound fits with large errors
        result="no excess of central transits among planets whose impact parameter is measured to %.1f; "
               "the excess below 0.05 in the whole sample is carried by fits at b's lower bound, %d of its %d with errors "
               "above 0.13 (median %.2f); " % (aimed_BERR, d["low_all"]["above_013"], d["low_all"]["n"], d["low_all"]["median_err"])
               + "; ".join("%s: %d planets, b below 0.05: %d against %.1f (%+.1f sigma), 0.05-0.1: %d against %.1f (%+.1f sigma), 0.1-0.5: %d against %.1f (%+.1f sigma)" % (
                   name, v["n"], *(v["tests"][str(b)][k] for b in aimed_BINS for k in ("observed", "expected", "sigma")))
                   for name, v in sorted(s.items(), key=lambda kv: not kv[0].startswith("all, b error"))))
    core.clear_targets("aimed_transits")
    core.record_targets("aimed_transits", [dict(target="koi:" + k["koi"], stat=k["b"], unit="impact parameter", vet="habitable-zone planet at b below 0.05, within the fit's own pile-up",
                                                    note="b %.3f +- %.3f, %.1f R_earth, %.0f d" % (k["b"], k["err"], k["prad"] or 0, k["period"] or 0)) for k in d["hz_low"]])
    print("recorded")


def aimed_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": aimed_analyse, "record": aimed_record}[mode]()


synchronous_DOC = r"""Stellar-synchronous orbits beyond tidal reach: planets past 10 d with P_orb at 1, 2 or 1/2 P_rot within 0.02 against shuffled rotation periods, and spot phase locking at the transits.

    python -I seti.py transits synchronous analyse
    python -I seti.py transits synchronous lock [KOI ...]
    python -I seti.py transits synchronous record"""

synchronous_ROOT = os.environ.get("SYNCHRONOUS_ROOT", core.data("synchronous"))
synchronous_OUT = "synchronous_orbits.json"
synchronous_PMIN, synchronous_TOL, synchronous_TEFF_BIN, synchronous_SHUFFLES = 10.0, 0.02, 500.0, 2000


def synchronous_load():
    """KOIs with a rotation period: McQuillan et al. (2013) unflagged first, Mazeh et al. (2015) for the rest."""
    from astropy.table import Table
    os.makedirs(synchronous_ROOT, exist_ok=True)
    kp = os.path.join(synchronous_ROOT, "koi_cumulative.csv")
    if not os.path.exists(kp):
        rows = core.exoplanets("SELECT kepid, kepoi_name, koi_disposition, koi_period, koi_time0bk, koi_duration, koi_depth, koi_prad, "
                                   "koi_steff, koi_srad, koi_smass, koi_count FROM cumulative")
        with open(kp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    rot = {}
    for cat, name in (("J/ApJ/801/3", "mazeh2015"), ("J/ApJ/775/L11", "mcquillan2013")):
        p = os.path.join(synchronous_ROOT, cat.replace("/", "_") + "_0.ecsv")
        if not os.path.exists(p):
            core.vizier(cat)[0].write(p, format="ascii.ecsv", overwrite=True)
        t = Table.read(p, format="ascii.ecsv")
        for r in t:
            prot = num(r["Prot"])
            if not prot or prot <= 0:
                continue
            if name == "mcquillan2013" and str(r["Flag"]) not in ("--", "", "None"):
                continue
            if name == "mazeh2015" and str(r["D"]) != "1":
                continue
            rvar = num(r["Rvar"])
            if name == "mazeh2015" and rvar:
                rvar = rvar / 1000.0  # ppm there, mmag in McQuillan
            rot[int(r["KIC"])] = dict(prot=prot, e_prot=num(r["e_Prot"]), rvar=rvar, teff_rot=num(r["Teff"]), source=name)
    kois = []
    for r in csv.DictReader(open(kp)):
        if r["koi_disposition"] == "FALSE POSITIVE" or not num(r["koi_period"]):
            continue
        s = rot.get(int(r["kepid"]))
        if not s:
            continue
        kois.append(dict(kic=int(r["kepid"]), koi=r["kepoi_name"], disposition=r["koi_disposition"], P=float(r["koi_period"]), t0=num(r["koi_time0bk"]),
                         dur=num(r["koi_duration"]), depth=num(r["koi_depth"]), prad=num(r["koi_prad"]), teff=num(r["koi_steff"]) or s["teff_rot"], **s))
    return kois


def synchronous_deltas(P, prot):
    return dict(d1=abs(P / prot - 1), d2=abs(P / (2 * prot) - 1), dh=abs(2 * P / prot - 1))


def synchronous_analyse():
    kois = synchronous_load()
    far = [k for k in kois if k["P"] >= synchronous_PMIN]
    for k in kois:
        k.update(synchronous_deltas(k["P"], k["prot"]))
        k["dmin"] = min(k["d1"], k["d2"], k["dh"])
        k["match"] = k["dmin"] < synchronous_TOL
    hits = [k for k in far if k["match"]]
    # null: rotation periods shuffled within temperature bins, orbital periods fixed
    rng = np.random.default_rng(3)
    teff = np.array([k["teff"] or 5500.0 for k in far])
    prot = np.array([k["prot"] for k in far])
    P = np.array([k["P"] for k in far])
    bins = np.floor(teff / synchronous_TEFF_BIN).astype(int)
    counts = []
    for _ in range(synchronous_SHUFFLES):
        sh = prot.copy()
        for b in np.unique(bins):
            idx = np.where(bins == b)[0]
            sh[idx] = prot[rng.permutation(idx)]
        d = np.minimum.reduce([np.abs(P / sh - 1), np.abs(P / (2 * sh) - 1), np.abs(2 * P / sh - 1)])
        counts.append(int(np.sum(d < synchronous_TOL)))
    counts = np.array(counts)
    print("%d KOIs with rotation periods (%d from McQuillan et al. 2013, %d from Mazeh et al. 2015), %d with P_orb of %.0f d or more: %d within %.2f of 1:1, 2:1 or 1:2 "
          "against %.1f +- %.1f with the rotation periods shuffled within %.0f K bins (p %.3f)" % (
              len(kois), sum(1 for k in kois if k["source"] == "mcquillan2013"), sum(1 for k in kois if k["source"] == "mazeh2015"), len(far), synchronous_PMIN,
              len(hits), synchronous_TOL, counts.mean(), counts.std(), synchronous_TEFF_BIN, float(np.mean(counts >= len(hits)))))
    for k in sorted(hits, key=lambda k: k["dmin"]):
        ratio = "1:1" if k["d1"] == k["dmin"] else ("2:1" if k["d2"] == k["dmin"] else "1:2")
        print("  %-10s %s KIC %d: P_orb %.3f d, P_rot %.3f +- %s d (%s, amplitude %s mmag), %s off by %.4f; depth %s ppm, R_p %s, Teff %s" % (
            k["koi"], k["disposition"][:9], k["kic"], k["P"], k["prot"], "%.3f" % k["e_prot"] if k["e_prot"] else "-", k["source"], "%.2f" % k["rvar"] if k["rvar"] else "-",
            ratio, k["dmin"], "%.0f" % k["depth"] if k["depth"] else "-", "%.2f" % k["prad"] if k["prad"] else "-", "%.0f" % k["teff"] if k["teff"] else "-"))
    near = [k for k in kois if k["P"] < synchronous_PMIN and k["match"]]
    print("  inside %.0f d, where tides can synchronise: %d of %d match" % (synchronous_PMIN, len(near), len(kois) - len(far)))
    json.dump(dict(n=len(kois), n_far=len(far), n_hits=len(hits), null_mean=float(counts.mean()), null_sd=float(counts.std()), p=float(np.mean(counts >= len(hits))),
                   kois=kois, params=dict(pmin=synchronous_PMIN, tol=synchronous_TOL, teff_bin=synchronous_TEFF_BIN, shuffles=synchronous_SHUFFLES)),
              open(core.results(synchronous_OUT), "w"), indent=0)


def synchronous_curve(kic):
    """Kepler long-cadence light curve, quarters concatenated and each normalised."""
    cache = os.path.join(synchronous_ROOT, "lc")
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, "%d.npz" % kic)
    if os.path.exists(p):
        z = np.load(p)
        return z["t"], z["f"]
    ts, fs = [], []
    for key in core.kepler_keys(kic, short=False):
        try:
            t, f, _ = core.read(core.get(escape_S3 + key, timeout=300, retries=2))
        except Exception as e:
            print("  KIC %d %s: %s" % (kic, key.split("/")[-1], str(e)[:60]), flush=True)
            continue
        ts.append(t)
        fs.append(f)
    if not ts:
        return np.array([]), np.array([])
    t, f = np.concatenate(ts), np.concatenate(fs)
    o = np.argsort(t)
    t, f = t[o], f[o]
    np.savez(p, t=t, f=f)
    return t, f


def synchronous_phases(t, f, prot, epochs, mask=None, nrot=3.0):
    """Phase (cycles), amplitude and point count of a sinusoid at prot in nrot-rotation windows about each epoch, cadences within mask of a transit excluded."""
    out = []
    for e in epochs:
        w = (t > e - nrot * prot / 2) & (t < e + nrot * prot / 2)
        if mask is not None:
            w &= mask
        if w.sum() < 0.5 * nrot * prot / (29.4 / 1440):
            out.append(None)
            continue
        ph = 2 * np.pi * (t[w] - e) / prot
        A = np.column_stack([np.ones(w.sum()), np.sin(ph), np.cos(ph), (t[w] - e) / prot])
        c = np.linalg.lstsq(A, f[w], rcond=None)[0]
        out.append(dict(phase=float((math.atan2(c[2], c[1]) / (2 * np.pi)) % 1.0), amp=float(math.hypot(c[1], c[2])), n=int(w.sum())))
    return out


def synchronous_lock(names):
    d = json.load(open(core.results(synchronous_OUT)))
    hits = [k for k in d["kois"] if k["match"] and k["P"] >= d["params"]["pmin"]]
    if names:
        hits = [k for k in hits if k["koi"] in names or k["koi"].split(".")[0] in names or str(k["kic"]) in names]
    rng = np.random.default_rng(5)
    others = [k for k in d["kois"] if not k["match"]]
    out = []
    for k in hits:
        t, f = synchronous_curve(k["kic"])
        if not len(t):
            print("%s: no light curve" % k["koi"])
            continue
        n0 = math.ceil((t.min() - k["t0"]) / k["P"])
        epochs = [k["t0"] + n * k["P"] for n in range(n0, n0 + int((t.max() - t.min()) / k["P"]) + 1)]
        epochs = [e for e in epochs if t.min() < e < t.max()]
        mask = np.ones(len(t), bool)
        for e in epochs:
            mask &= np.abs(t - e) > 1.5 * (k["dur"] or 3.0) / 24
        ph = synchronous_phases(t, f, k["prot"], epochs, mask)
        have = [(e, p) for e, p in zip(epochs, ph) if p]
        if len(have) < 6:
            print("%s: %d windows with data" % (k["koi"], len(have)))
            continue
        phases = np.array([p["phase"] for _, p in have])
        # scatter about the circular mean and drift over the baseline
        mean = math.atan2(np.sin(2 * np.pi * phases).mean(), np.cos(2 * np.pi * phases).mean()) / (2 * np.pi)
        dev = ((phases - mean + 0.5) % 1.0) - 0.5
        scatter = float(np.sqrt(np.mean(dev ** 2)))
        # the same for the 40 nearest-period rotators at this planet's epochs
        pool = sorted(others, key=lambda o: abs(math.log(o["prot"] / k["prot"])))[:40]
        ctrl = []
        for o in [pool[i] for i in rng.choice(len(pool), min(8, len(pool)), replace=False)]:
            to, fo = synchronous_curve(o["kic"])
            if not len(to):
                continue
            pho = synchronous_phases(to, fo, o["prot"], [e for e in epochs if to.min() < e < to.max()])
            pp = np.array([p["phase"] for p in pho if p])
            if len(pp) < 6:
                continue
            m = math.atan2(np.sin(2 * np.pi * pp).mean(), np.cos(2 * np.pi * pp).mean()) / (2 * np.pi)
            ctrl.append(float(np.sqrt(np.mean((((pp - m + 0.5) % 1.0) - 0.5) ** 2))))
        print("%s KIC %d: P_orb %.3f d, P_rot %.3f d: %d transit windows over %.0f d, rotational phase at the transits scatters by %.3f cycle (amplitude median %.4f) "
              "against %s for %d rotators at like periods" % (k["koi"], k["kic"], k["P"], k["prot"], len(have), have[-1][0] - have[0][0], scatter,
                                                               float(np.median([p["amp"] for _, p in have])), ", ".join("%.3f" % c for c in sorted(ctrl)) or "-", len(ctrl)))
        out.append(dict(koi=k["koi"], kic=k["kic"], P=k["P"], prot=k["prot"], n_windows=len(have), span=have[-1][0] - have[0][0], scatter=scatter,
                        amp=float(np.median([p["amp"] for _, p in have])), controls=ctrl, phases=[(float(e), p["phase"], p["amp"]) for e, p in have]))
    json.dump(out, open(core.results("synchronous_lock.json"), "w"), indent=0)


def synchronous_record():
    d = json.load(open(core.results(synchronous_OUT)))
    lp = core.results("synchronous_lock.json")
    locks = {x["koi"]: x for x in json.load(open(lp))} if os.path.exists(lp) else {}
    hits = [k for k in d["kois"] if k["match"] and k["P"] >= d["params"]["pmin"]]
    core.record_search(
        "synchronous_orbits", "Stellar-synchronous orbits beyond the tidal radius", archive="cumulative KOI table (NASA Exoplanet Archive); KOI rotation periods of McQuillan et al. (2013, J/ApJ/775/L11) and Mazeh et al. (2015, J/ApJ/801/3)",
        selection="KOI candidates and confirmed planets whose host has a rotation period (%d), those with P_orb of %.0f d or more (%d)" % (d["n"], d["params"]["pmin"], d["n_far"]),
        code=os.path.abspath(__file__), entry="synchronous", n_targets=d["n"], params=d["params"],
        null_summary="rotation periods shuffled among the hosts within %.0f K bins %d times: %.1f +- %.1f matches" % (d["params"]["teff_bin"], d["params"]["shuffles"], d["null_mean"], d["null_sd"]),
        sensitivity="P_orb within %.0f per cent of P_rot, 2 P_rot or P_rot/2; the phase of the rotational modulation at the transits to about 0.05 cycle over four years" % (100 * d["params"]["tol"]),
        ledger="Stellar-synchronous orbits beyond the tidal radius",
        result="%d matches within %.2f beyond %.0f d against %.1f +- %.1f shuffled (p %.2f)%s" % (
            d["n_hits"], d["params"]["tol"], d["params"]["pmin"], d["null_mean"], d["null_sd"], d["p"],
            ("; phase at the transits: " + "; ".join("%s %.3f cycle against %s" % (x["koi"], x["scatter"], "-".join("%.2f" % c for c in (min(x["controls"]), max(x["controls"]))) if x["controls"] else "no controls") for x in locks.values())) if locks else ""))
    core.clear_targets("synchronous_orbits")
    core.record_targets("synchronous_orbits", [dict(target="koi:%s" % k["koi"], stat=k["dmin"], unit="smallest |P_orb/P_rot - 1| over 1:1, 2:1 and 1:2",
                                                        vet=("beyond %.0f d, matched%s" % (d["params"]["pmin"], (", phase scatter %.3f cycle" % locks[k["koi"]]["scatter"]) if k["koi"] in locks else "")) if k in hits
                                                        else ("inside %.0f d, where tides synchronise" % d["params"]["pmin"] if k["match"] else None),
                                                        note="P_orb %.3f d, P_rot %.3f d (%s), Teff %s" % (k["P"], k["prot"], k["source"], "%.0f" % k["teff"] if k["teff"] else "-")) for k in d["kois"]])
    print("recorded %d KOIs" % d["n"])


def synchronous_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if mode == "lock":
        synchronous_lock(sys.argv[2:])
    else:
        {"analyse": synchronous_analyse, "record": synchronous_record}[mode]()


lightweight_DOC = r"""Opaque lightweight eclipsers: Gaia DR3 eclipses deeper than 0.75 mag under 30 d whose RVS amplitude is below 0.1 of the least stellar-mass floor.

    python -I seti.py transits lightweight fetch
    python -I seti.py transits lightweight analyse
    python -I seti.py transits lightweight record"""

lightweight_ROOT = os.environ.get("LIGHTWEIGHT_ROOT", core.data("eclipsers"))
lightweight_OUT = "lightweight_eclipsers.json"
lightweight_DEPTH, lightweight_PMAX, lightweight_NRV, lightweight_RATIO, lightweight_CYCLES = 0.75, 30.0, 10, 0.1, 5.0
lightweight_MS = ((3000, 0.25, 0.12), (3500, 0.42, 0.37), (4000, 0.60, 0.60), (4500, 0.70, 0.70), (5000, 0.80, 0.82), (5500, 0.90, 0.92), (5800, 1.00, 1.00),
                  (6000, 1.10, 1.08), (6500, 1.32, 1.25), (7000, 1.55, 1.45), (7500, 1.75, 1.65), (8000, 1.95, 1.85), (9000, 2.3, 2.2), (10000, 2.7, 2.6))


def lightweight_fetch():
    """Deep short-period vari_eclipsing_binary systems, then their gaia_source rows in id batches."""
    os.makedirs(lightweight_ROOT, exist_ok=True)
    eb = core.tap("gaia", "SELECT source_id, frequency, model_type, derived_primary_ecl_depth, derived_primary_ecl_depth_error, derived_primary_ecl_duration, "
                 "derived_secondary_ecl_depth, derived_secondary_ecl_duration, reduced_chi2, global_ranking FROM gaiadr3.vari_eclipsing_binary "
                 "WHERE derived_primary_ecl_depth > %g AND frequency > %g" % (lightweight_DEPTH, 1.0 / lightweight_PMAX), maxrec=400000, timeout=1800)
    print("%d eclipsers deeper than %.2f mag under %.0f d" % (len(eb), lightweight_DEPTH, lightweight_PMAX), flush=True)
    src = core.tap_by_ids("gaia", [r["source_id"] for r in eb],
                         "SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, grvs_mag, teff_gspphot, ruwe, non_single_star, rv_nb_transits, rv_time_duration, "
                         "rv_amplitude_robust, rv_renormalised_gof, rv_chisq_pvalue, vbroad, radial_velocity, radial_velocity_error FROM gaiadr3.gaia_source "
                         "WHERE rv_nb_transits >= %d AND source_id IN (%%s)" % lightweight_NRV)
    rows = []
    for r in eb:
        s = src.get(str(r["source_id"]))
        if s:
            rows.append(dict(r, **s))
    json.dump(rows, open(os.path.join(lightweight_ROOT, "gaia_deep_eclipsers.json"), "w"))
    print("%d with %d or more RVS transits" % (len(rows), lightweight_NRV), flush=True)


def lightweight_ms(teff):
    """Main-sequence radius and mass (solar) at a temperature, interpolated."""
    t, r, m = zip(*lightweight_MS)
    return float(np.interp(teff, t, r)), float(np.interp(teff, t, m))


def lightweight_analyse():
    rows = json.load(open(os.path.join(lightweight_ROOT, "gaia_deep_eclipsers.json")))
    out = []
    for r in rows:
        P = 1.0 / float(r["frequency"])
        depth, A, n, span, teff = num(r["derived_primary_ecl_depth"]), num(r["rv_amplitude_robust"]), int(num(r["rv_nb_transits"]) or 0), num(r["rv_time_duration"]), num(r["teff_gspphot"])
        if P > lightweight_PMAX or depth is None or depth < lightweight_DEPTH or A is None or n < lightweight_NRV or not teff:
            continue
        f = 1 - 10 ** (-0.4 * depth)  # fraction of light lost at mid-eclipse
        R1, M1 = lightweight_ms(teff)
        R2 = math.sqrt(f) * R1  # dark body covering that fraction of the disc
        teff2 = float(np.interp(R2, [x[1] for x in lightweight_MS], [x[0] for x in lightweight_MS]))
        M2 = lightweight_ms(teff2)[1]  # least natural mass at that radius
        G, Msun = 6.674e-11, 1.989e30
        K_min = (2 * math.pi * G / (P * 86400.0)) ** (1 / 3) * M2 * Msun / ((M1 + M2) * Msun) ** (2 / 3) / 1000.0
        ratio = (A / 2) / K_min
        out.append(dict(source_id=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"]), P=P, depth=depth, f=f, teff=teff, R1=R1, M1=M1, R2=R2, M2=M2, K_min=K_min, A=A, n_rv=n, span=span,
                        cycles=(span or 0) / P, ratio=ratio, G=num(r["phot_g_mean_mag"]), grvs=num(r["grvs_mag"]), bprp=num(r["bp_rp"]), ruwe=num(r["ruwe"]), vbroad=num(r["vbroad"]),
                        gof=num(r["rv_renormalised_gof"]), p_const=num(r["rv_chisq_pvalue"]), model=r["model_type"], rank=num(r["global_ranking"]), chi2=num(r["reduced_chi2"]),
                        nss=int(num(r["non_single_star"]) or 0), secondary=num(r["derived_secondary_ecl_depth"])))
    ratios = np.array([x["ratio"] for x in out])
    cand = [x for x in out if x["ratio"] < lightweight_RATIO and x["cycles"] >= lightweight_CYCLES]
    print("%d eclipsers deeper than %.2f mag under %.0f d with %d or more RVS transits: observed over least semi-amplitude median %.2f, 10th percentile %.2f; %d below %.1f with %.0f or more cycles spanned" % (
        len(out), lightweight_DEPTH, lightweight_PMAX, lightweight_NRV, np.median(ratios), np.percentile(ratios, 10), len(cand), lightweight_RATIO, lightweight_CYCLES))
    for x in sorted(cand, key=lambda x: x["ratio"])[:25]:
        print("  Gaia DR3 %s (%.4f %+.4f): P %.3f d, depth %.2f mag (%.0f per cent), Teff %.0f K, R1 %.2f M1 %.2f, eclipser R %.2f Rsun needs %.2f Msun and K %.1f km/s; RVS peak to peak %.1f km/s over %d transits (%.0f cycles), ratio %.3f; G %.1f G_RVS %.1f vbroad %s RV gof %s p %s; model %s rank %.2f%s" % (
            x["source_id"], x["ra"], x["dec"], x["P"], x["depth"], 100 * x["f"], x["teff"], x["R1"], x["M1"], x["R2"], x["M2"], x["K_min"], x["A"], x["n_rv"], x["cycles"], x["ratio"], x["G"] or 0, x["grvs"] or 0,
            "%.0f" % x["vbroad"] if x["vbroad"] else "-", "%.1f" % x["gof"] if x["gof"] is not None else "-", "%.2f" % x["p_const"] if x["p_const"] is not None else "-", x["model"], x["rank"] or 0,
            ", secondary %.2f mag" % x["secondary"] if x["secondary"] else ""))
    json.dump(dict(n=len(out), n_cand=len(cand), ratio_median=float(np.median(ratios)) if len(ratios) else None, rows=out,
                   params=dict(depth=lightweight_DEPTH, pmax=lightweight_PMAX, nrv=lightweight_NRV, ratio=lightweight_RATIO, cycles=lightweight_CYCLES)), open(core.results(lightweight_OUT), "w"), indent=0)


def lightweight_record():
    d = json.load(open(core.results(lightweight_OUT)))
    cand = [x for x in d["rows"] if x["ratio"] < d["params"]["ratio"] and x["cycles"] >= d["params"]["cycles"]]
    core.record_search(
        "lightweight_eclipsers", "Opaque lightweight eclipsers", archive="Gaia DR3 vari_eclipsing_binary and gaia_source (RVS)", release="Gaia DR3",
        selection="eclipses deeper than %.2f mag at periods under %.0f d with %d or more RVS transits (%d)" % (d["params"]["depth"], d["params"]["pmax"], d["params"]["nrv"], d["n"]),
        code=os.path.abspath(__file__), entry="lightweight", n_targets=d["n"], params=d["params"],
        null_summary="the ratio of the observed RVS semi-amplitude to the least a natural eclipser of the depth's size allows: median %.2f over the sample" % (d["ratio_median"] or 0),
        sensitivity="a ratio below %.1f: an eclipser of stellar size with under a tenth of the least stellar mass's pull" % d["params"]["ratio"], ledger="Opaque lightweight eclipsers",
        result="%d of %d below %.1f with %.0f or more cycles spanned%s" % (len(cand), d["n"], d["params"]["ratio"], d["params"]["cycles"],
                                                                      (": " + "; ".join("Gaia DR3 %s (P %.2f d, %.0f per cent, ratio %.3f)" % (x["source_id"], x["P"], 100 * x["f"], x["ratio"]) for x in sorted(cand, key=lambda x: x["ratio"])[:8])) if cand else ""))
    core.clear_targets("lightweight_eclipsers")
    core.record_targets("lightweight_eclipsers", [dict(target="gaia_dr3:%s" % x["source_id"], stat=x["ratio"], unit="observed RVS semi-amplitude over the least a natural eclipser allows",
                                                            vet=None, note="P %.3f d, depth %.2f mag, Teff %.0f K, K_min %.1f km/s, RVS peak to peak %.1f over %d transits" % (x["P"], x["depth"], x["teff"], x["K_min"], x["A"], x["n_rv"])) for x in d["rows"]])
    print("recorded %d eclipsers" % d["n"])


def lightweight_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": lightweight_fetch, "analyse": lightweight_analyse, "record": lightweight_record}[mode]()


roche_DOC = r"""Transits inside the fluid Roche limit: Gaia main-sequence variables faster than 8.6 cycles a day where a grey box beats a sinusoid in G, BP and RP or ZTF g and r.

    python -I seti.py transits roche select
    python -I seti.py transits roche fetch_gaia [--threads 4]
    python -I seti.py transits roche fetch [--threads 4]       (ZTF, for the box-like ones)
    python -I seti.py transits roche analyse [--procs N] [--nsim 2000]      (each band against its sinusoid null)
    python -I seti.py transits roche ztf [--threads 4]      (the grey boxes beyond the null, in ZTF g and r)
    python -I seti.py transits roche record"""

roche_ROOT = os.environ.get("ROCHE_ROOT", core.data("roche"))
roche_OUT = "roche_transits.json"
roche_FMIN, roche_AMIN, roche_NMIN, roche_GMAX, roche_DUTY = 8.6, 0.02, 20, 17.5, 0.15
roche_MS = ((0.5, 3.9), (0.7, 4.8), (0.9, 5.6), (1.1, 6.3), (1.3, 7.0), (1.5, 7.7), (1.8, 8.6), (2.1, 9.6), (2.5, 10.9), (3.0, 12.4))


def roche_ms(bprp):
    """Main-sequence absolute G at a BP-RP colour, interpolated."""
    c, m = zip(*roche_MS)
    return float(np.interp(bprp, c, m))


def roche_select():
    os.makedirs(roche_ROOT, exist_ok=True)
    var = json.load(open(os.path.join(roche_ROOT, "vari_short_fast.json")))
    src = {r["source_id"]: r for r in json.load(open(os.path.join(roche_ROOT, "gaia_source_fast.json")))}
    out = []
    for v in var:
        s = src.get(v["source_id"])
        f, amp, n = num(v["frequency"]), num(v["amplitude_estimate"]), int(num(v["number_of_fov_transits"]) or 0)
        if not s or f is None or f < roche_FMIN or (amp or 0) < roche_AMIN or n < roche_NMIN:
            continue
        plx, poe, G, bprp = num(s["parallax"]), num(s["parallax_over_error"]), num(s["phot_g_mean_mag"]), num(s["bp_rp"])
        if not plx or plx <= 0 or (poe or 0) < 5 or G is None or bprp is None or G > roche_GMAX or float(s["dec"]) < -28:
            continue
        MG = G + 5 * math.log10(plx / 100.0)
        if bprp < 0.5 or bprp > 3.0 or not (roche_ms(bprp) - 0.8 <= MG <= roche_ms(bprp) + 1.0):
            continue
        if (num(s["ruwe"]) or 9) > 1.4 or (num(s["ipd_frac_multi_peak"]) or 0) >= 2:
            continue
        if min(abs(f / 4.0 - round(f / 4.0)) * 4.0, abs(f / 4.0 - round(f / 4.0)) * 4.0) < 0.05:
            continue  # Gaia's 6-hour spin and its harmonics
        out.append(dict(source_id=v["source_id"], ra=float(s["ra"]), dec=float(s["dec"]), G=G, bprp=bprp, MG=MG, plx=plx, frequency=f, P_h=24.0 / f, amp=amp, n_transits=n,
                        teff=num(s["teff_gspphot"]), ruwe=num(s["ruwe"]), excess=num(s["phot_bp_rp_excess_factor"]), nss=int(num(s["non_single_star"]) or 0)))
    out.sort(key=lambda x: -x["amp"])
    json.dump(out, open(os.path.join(roche_ROOT, "candidates.json"), "w"), indent=0)
    print("%d fast variables with source rows; %d main-sequence candidates faster than %.1f cycles a day with amplitude above %.2f mag (%d brighter than G 16)" % (
        len(var), len(out), roche_FMIN, roche_AMIN, sum(1 for x in out if x["G"] < 16)))


roche_LC = ("https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE%%20%.6f%%20%.6f%%200.000417"
            "&BANDNAME=g,r&NOBS_MIN=30&BAD_CATFLAGS_MASK=32768&FORMAT=csv")


def roche_fetch_one(x):
    """One star's ZTF DR23 g and r light curves from a 1.5 arcsec cone into ROOT/lc; 1 when it has epochs."""
    p = os.path.join(roche_ROOT, "lc", "%s.csv" % x["source_id"])
    if os.path.exists(p):
        return 1
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        text = core.get(roche_LC % (x["ra"], x["dec"]), timeout=300, retries=2).decode("utf-8", "replace")
    except Exception as e:
        print("  %s: light curves failed: %s" % (x["source_id"], str(e)[:80]), flush=True)
        return 0
    rows = text.splitlines()
    head = rows[0] if rows and rows[0].startswith("oid") else "oid,expid,hjd,mjd,mag,magerr,catflags,filtercode"
    lines = [ln for ln in rows[1:] if ln.strip()] if rows and rows[0].startswith("oid") else []
    open(p, "w", encoding="utf-8").write("\n".join([head] + lines) + "\n")
    return 1 if lines else 0


def roche_fetch(threads=4):
    """ZTF DR23 g and r light curves per candidate, one request a star."""
    cand = json.load(open(os.path.join(roche_ROOT, "candidates.json")))
    got = sum(core.pmap(roche_fetch_one, cand, threads=threads))
    print("%d of %d candidates with light curves" % (got, len(cand)))


roche_DL = "https://gea.esac.esa.int/data-server/data?RETRIEVAL_TYPE=EPOCH_PHOTOMETRY&ID=Gaia+DR3+%s&FORMAT=CSV&VALID_DATA=false"


def roche_fetch_gaia(threads=4):
    """Gaia DR3 epoch photometry per candidate by DataLink, brightest first, unfiltered with flags."""
    cand = sorted(json.load(open(os.path.join(roche_ROOT, "candidates.json"))), key=lambda x: x["G"])
    d = os.path.join(roche_ROOT, "gaia_epoch")
    os.makedirs(d, exist_ok=True)

    def one(x):
        p = os.path.join(d, "%s.csv" % x["source_id"])
        if os.path.exists(p):
            return 1
        b = core.get(roche_DL % x["source_id"], timeout=120, retries=2, soft=True, pause=3)
        if not b:
            return 0
        open(p, "wb").write(b)
        return 1
    got = sum(core.pmap(one, cand, threads=threads))
    print("%d of %d candidates with Gaia epoch photometry" % (got, len(cand)))


def roche_gaia_series(path):
    """{band: (days, mag)} from a DataLink table without photometry-rejected transits; variability-rejected ones kept as possible eclipses."""
    import csv
    import io
    t = open(path, encoding="utf-8", errors="replace").read()
    rows = list(csv.DictReader(io.StringIO("\n".join(ln for ln in t.splitlines() if not ln.startswith("#")))))
    out = {}
    for band, tc, mc in (("G", "g_transit_time", "g_transit_mag"), ("BP", "bp_obs_time", "bp_mag"), ("RP", "rp_obs_time", "rp_mag")):
        b = [r for r in rows if num(r.get(mc)) is not None and num(r.get(tc)) is not None and r.get("rejected_by_photometry", "false") != "true"]
        out[band] = (np.array([float(r[tc]) for r in b]), np.array([float(r[mc]) for r in b]))
    return out


def roche_fold(t, m, P):
    """Fold at P: chi-square gains of a one-harmonic sinusoid and the best box (duty up to roche_DUTY) over a constant, box depth and duty, out-of-transit scatter."""
    ph = (t / P) % 1.0
    w = np.ones_like(m)
    mean = np.average(m, weights=w)
    chi0 = np.sum((m - mean) ** 2)
    A = np.column_stack([np.ones_like(ph), np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
    c = np.linalg.lstsq(A, m, rcond=None)[0]
    chi_sin = np.sum((m - A @ c) ** 2)
    best = (chi0, 0.0, 0.0, 0.0)
    for duty in (0.03, 0.05, 0.08, 0.11, roche_DUTY):
        for phase0 in np.arange(0, 1, duty / 3):
            inside = ((ph - phase0) % 1.0) < duty
            if inside.sum() < 5 or (~inside).sum() < 10:
                continue
            m_in, m_out = m[inside].mean(), m[~inside].mean()
            chi = np.sum((m[inside] - m_in) ** 2) + np.sum((m[~inside] - m_out) ** 2)
            if chi < best[0]:
                best = (chi, m_in - m_out, duty, float(np.std(m[~inside])))
    return dict(gain_sin=chi0 - chi_sin, gain_box=chi0 - best[0], depth=best[1], duty=best[2], scatter_out=best[3], amp_sin=2 * math.hypot(c[1], c[2]), n=len(m))


def roche_ztf_series(p, strict=True):
    """{band: (mjd, mag)} of a ZTF light-curve file, catflags 0 only, or below 32768 (no bad-image bit) when not strict."""
    if not os.path.exists(p):
        return {}
    rows = [ln.split(",") for ln in open(p, encoding="utf-8").read().splitlines()[1:] if ln.strip()]
    head = open(p, encoding="utf-8").readline().strip().split(",")
    if not rows or "mag" not in head:
        return {}
    im, it, ic, ifl = head.index("mag"), head.index("mjd"), head.index("filtercode"), head.index("catflags")
    out = {}
    for band in ("zg", "zr"):
        sel = [r for r in rows if r[ic] == band and (r[ifl] == "0" if strict else int(r[ifl]) < 32768)]
        out[band] = (np.array([float(r[it]) for r in sel]), np.array([float(r[im]) for r in sel]))
    return out


def roche_series(x):
    """Light curves {band: (t, mag)} and the achromaticity pair: Gaia G, BP, RP where on disk, else ZTF g and r."""
    p = os.path.join(roche_ROOT, "gaia_epoch", "%s.csv" % x["source_id"])
    if os.path.exists(p):
        try:
            return "gaia", roche_gaia_series(p), ("BP", "RP"), 25
        except Exception:
            return "gaia", {}, ("BP", "RP"), 25
    p = os.path.join(roche_ROOT, "lc", "%s.csv" % x["source_id"])
    if not os.path.exists(p):
        return None, {}, None, 0
    return "ztf", roche_ztf_series(p), ("zg", "zr"), 40


def roche_clip(t, m):
    """Drop brightenings beyond 5 MAD and dimmings beyond 1.5 mag."""
    med = np.median(m)
    mad = 1.4826 * np.median(np.abs(m - med))
    keep = (m - med > -(5 * mad + 0.05)) & (m - med < 1.5)
    return t[keep], m[keep]


roche_FACTORS = (1.0, 2.0, 0.5)
roche_P = 1e-3  # lead band's p-value against the sinusoid null


def roche_chi(ph, X):
    """Chi-square of the constant, the one-harmonic sinusoid and the best box (duty up to roche_DUTY) for each column of X."""
    n = len(ph)
    A = np.column_stack([np.ones(n), np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
    chi_sin = ((X - A @ np.linalg.lstsq(A, X, rcond=None)[0]) ** 2).sum(0)
    T, T2 = X.sum(0), (X ** 2).sum(0)
    chi0 = T2 - T ** 2 / n
    best = chi0.copy()
    for duty in (0.03, 0.05, 0.08, 0.11, roche_DUTY):
        M = ((ph[None, :] - np.arange(0, 1, duty / 3)[:, None]) % 1.0) < duty
        c = M.sum(1)
        ok = (c >= 5) & (n - c >= 10)
        if ok.any():
            M, c = M[ok].astype(float), c[ok][:, None]
            S, Q = M @ X, M @ (X ** 2)
            best = np.minimum(best, (Q - S ** 2 / c + (T2 - Q) - (T - S) ** 2 / (n - c)).min(0))
    return chi0, chi_sin, best


def roche_null(t, m, P, nsim, seed):
    """Box advantage over the sinusoid, n ln(chi_sin / chi_box) at the best period factor, and its p-value against the best sinusoid plus the star's resampled residuals."""
    m = m - np.median(m)
    phs = [(t / (P * k)) % 1.0 for k in roche_FACTORS]

    def advantage(ph, X):
        chi0, chi_sin, chi_box = roche_chi(ph, X)
        return len(m) * np.log(np.maximum(chi_sin, 1e-12) / np.maximum(chi_box, 1e-12))
    obs = max(float(advantage(ph, m[:, None])[0]) for ph in phs)
    ph = min(phs, key=lambda ph: float(roche_chi(ph, m[:, None])[1][0]))
    A = np.column_stack([np.ones(len(m)), np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
    fit = A @ np.linalg.lstsq(A, m, rcond=None)[0]
    rng = np.random.default_rng(seed)
    above, done = 0, 0
    # ten times the draws when fewer than ten reach the observed advantage
    while done < nsim or (above < 10 and done < 10 * nsim):
        k = min(2000, 10 * nsim - done)
        X = fit[:, None] + (m - fit)[rng.integers(0, len(m), size=(len(m), k))]
        above += int(np.sum(np.max([advantage(p2, X) for p2 in phs], axis=0) >= obs))
        done += k
    return obs, (1 + above) / (1 + done)


def roche_star(x, nsim=2000):
    """One candidate's folds per band at the Gaia period, its double and half, with each band's sinusoid null."""
    src, series, pair, nmin = roche_series(x)
    if not series:
        return None
    res = {}
    for band, (t, m) in series.items():
        if len(m) < nmin:
            continue
        t, m = roche_clip(t, m)
        if len(m) < nmin:
            continue
        fits = {k: roche_fold(t, m, x["P_h"] / 24.0 * k) for k in roche_FACTORS}
        k = max(fits, key=lambda k: max(fits[k]["gain_sin"], fits[k]["gain_box"]))
        adv, p = roche_null(t, m, x["P_h"] / 24.0, nsim, int(x["source_id"]) % 2 ** 32 + len(res))
        res[band] = dict(fits[k], period_factor=k, advantage=adv, p_box=p)
    if not res:
        return None
    lead = "G" if "G" in res else max(res, key=lambda b: res[b]["n"])
    r0 = res[lead]
    boxy = r0["gain_box"] > 1.2 * r0["gain_sin"] and r0["depth"] > 3 * r0["scatter_out"] / math.sqrt(max(r0["n"] * r0["duty"], 1)) and \
        all(r["gain_box"] >= r["gain_sin"] for b, r in res.items() if b != lead)
    a, b = pair
    grey = (a in res and b in res and res[a]["depth"] > 0 and res[b]["depth"] > 0
            and abs(res[a]["depth"] - res[b]["depth"]) < 0.3 * max(res[a]["depth"], res[b]["depth"]))
    sig = r0["p_box"] < roche_P and all(r["p_box"] < 0.05 for bb, r in res.items() if bb != lead)
    return dict(x, source=src, lead=lead, bands={bb: {k2: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k2, v in r.items()}
                                                 for bb, r in res.items()},
                boxy=bool(boxy), grey=bool(grey and boxy), significant=bool(sig and boxy))


def roche_analyse(procs=1, nsim=2000):
    """Every candidate's folds and sinusoid nulls on procs processes into RESULTS/roche_transits.json."""
    import functools
    from concurrent.futures import ProcessPoolExecutor
    cand = json.load(open(os.path.join(roche_ROOT, "candidates.json")))
    with ProcessPoolExecutor(procs) as ex:
        out = [x for x in ex.map(functools.partial(roche_star, nsim=nsim), cand, chunksize=16) if x]
    found = [x for x in out if x["grey"] and x["significant"]]
    print("%d candidates with light curves (%d Gaia, %d ZTF): %d box-like, %d of those beyond the sinusoid null (p below %g in the lead band, "
          "%.1f expected by chance there), %d with equal depths in the two colours as well" % (
              len(out), sum(x["source"] == "gaia" for x in out), sum(x["source"] == "ztf" for x in out), sum(x["boxy"] for x in out),
              sum(x["significant"] for x in out), roche_P, roche_P * len(out), len(found)))
    for x in sorted(found, key=lambda x: x["bands"][x["lead"]]["p_box"])[:30]:
        b = x["bands"]
        print("  Gaia DR3 %s (%.4f %+.4f): G %.1f BP-RP %.2f M_G %.1f, P %.3f h x %s, amp %.3f; " % (
            x["source_id"], x["ra"], x["dec"], x["G"], x["bprp"], x["MG"], x["P_h"], b[x["lead"]]["period_factor"], x["amp"]) +
              "; ".join("%s box %.3f mag duty %.2f, advantage %.1f, p %.1e (n %d)" % (k, r["depth"], r["duty"], r["advantage"], r["p_box"], r["n"])
                        for k, r in b.items()))
    json.dump(dict(n_cand=len(cand), n_lc=len(out), n_boxy=int(sum(x["boxy"] for x in out)), n_grey=int(sum(x["grey"] for x in out)),
                   n_sig=int(sum(x["significant"] for x in out)), n_found=len(found), rows=out,
                   params=dict(fmin=roche_FMIN, amin=roche_AMIN, nmin=roche_NMIN, gmax=roche_GMAX, duty=roche_DUTY, nsim=nsim, p=roche_P)),
              open(core.results(roche_OUT), "w"), indent=0)


roche_ZTF = "roche_ztf.json"


def roche_ztf(threads=4):
    """The grey boxes beyond the null in ZTF: best box from 0.45 to 2.2 Gaia periods, depths in g and r, secondary eclipse, odd and even depths, sinusoid's fit."""
    from astropy.timeseries import BoxLeastSquares
    d = json.load(open(core.results(roche_OUT)))
    found = [x for x in d["rows"] if x["grey"] and x["significant"]]
    core.pmap(roche_fetch_one, found, threads=threads)
    out = {}
    for x in found:
        path = os.path.join(roche_ROOT, "lc", "%s.csv" % x["source_id"])
        series = {b: roche_clip(t, m) for b, (t, m) in roche_ztf_series(path).items() if len(m) >= 40}
        flagged = not series
        if flagged:
            # every epoch flagged: those without the bad-image bit
            series = {b: roche_clip(t, m) for b, (t, m) in roche_ztf_series(path, strict=False).items() if len(m) >= 40}
        P0 = x["P_h"] / 24.0
        if not series:
            out[x["source_id"]] = dict(verdict=None, note="no ZTF light curve")
            print("Gaia DR3 %s: no ZTF light curve" % x["source_id"], flush=True)
            continue
        band = max(series, key=lambda b: len(series[b][1]))
        t, m = series[band]
        durations = P0 * np.array([0.02, 0.04, 0.07, 0.1])
        periods = np.exp(np.linspace(np.log(0.45 * P0), np.log(2.2 * P0), 30000))
        bls = BoxLeastSquares(t, -(m - np.median(m)))
        pw = bls.power(periods, durations, objective="snr")
        i = int(np.argmax(pw.power))
        P, dur, t0 = float(pw.period[i]), float(pw.duration[i]), float(pw.transit_time[i])
        st = bls.compute_stats(P, dur, t0)
        depth, err = float(st["depth"][0]), float(st["depth"][1])
        sec = float(st["depth_phased"][0]) / max(float(st["depth_phased"][1]), 1e-9)
        odd, even = st["depth_odd"], st["depth_even"]
        oe = abs(float(odd[0]) - float(even[0])) / max(math.hypot(float(odd[1]), float(even[1])), 1e-9)
        other = {}
        for b2, (t2, m2) in series.items():
            if b2 != band:
                try:
                    st2 = BoxLeastSquares(t2, -(m2 - np.median(m2))).compute_stats(P, dur, t0)
                except ValueError:
                    continue  # no point of this band inside the eclipse
                other[b2] = (float(st2["depth"][0]), float(st2["depth"][1]))
        sine = float(st["harmonic_delta_log_likelihood"])
        told = "ZTF %s%s box %.3f +- %.3f mag at P %.4f h (%.2f Gaia periods), duration %.0f min, %d points; secondary %.1f sigma, odd-even %.1f sigma%s" % (
            band, " (flagged epochs, catflags below 32768)" if flagged else "", depth, err, 24 * P, P / P0, 1440 * dur, len(m), sec, oe,
            "".join("; %s box %.3f +- %.3f" % (b2, dd, ee) for b2, (dd, ee) in other.items()))
        if depth / max(err, 1e-9) < 7:
            verdict = "not in ZTF at the Gaia period: Gaia's period or depth"
        elif sine > 0:
            verdict = "sinusoidal in ZTF, a pulsator's light curve"
        elif sec > 3:
            verdict = "eclipsing binary in ZTF: a secondary eclipse at phase 0.5"
        elif oe > 3:
            verdict = "eclipsing binary in ZTF: odd and even eclipses differ"
        elif any(abs(dd - depth) > 0.3 * max(dd, depth) and abs(dd - depth) > 3 * math.hypot(ee, err) for dd, ee in other.values()):
            verdict = "eclipsing binary in ZTF: the eclipse differs between g and r"
        else:
            verdict = None
        out[x["source_id"]] = dict(verdict=verdict, note=told, period_h=24 * P, depth=depth, err=err, secondary_sigma=sec, odd_even_sigma=oe, other=other)
        print("Gaia DR3 %s (G %.1f, M_G %.1f, BP-RP %.2f, Gaia P %.3f h): %s; %s" % (
            x["source_id"], x["G"], x["MG"], x["bprp"], x["P_h"], told, verdict or "a grey box in ZTF too"), flush=True)
    json.dump(out, open(core.results(roche_ZTF), "w"), indent=1)


def roche_vet(x):
    """Reading of one light curve: a sinusoid, a box no more than the sinusoid null reaches, an eclipse of two colours, or None for a grey box."""
    r = x["bands"][x["lead"]]
    if not x["boxy"]:
        return "sinusoidal at the Gaia period, a pulsator's light curve"
    if not x["significant"]:
        return "box-like only as far as a pulsator's sinusoid with the star's own noise reaches (p %.2g)" % r["p_box"]
    if not x["grey"]:
        return "box-like beyond the sinusoid null with unequal depths in the two colours, an eclipsing binary of stars of two colours"
    return roche_ZTF_READ.get(x["source_id"], {}).get("verdict")


roche_ZTF_READ = {}


def roche_record():
    d = json.load(open(core.results(roche_OUT)))
    if os.path.exists(core.results(roche_ZTF)):
        roche_ZTF_READ.update(json.load(open(core.results(roche_ZTF))))
    found = [x for x in d["rows"] if x["grey"] and x["significant"]]
    core.record_search(
        "roche_transits", "Transiting bodies inside the fluid Roche limit", archive="Gaia DR3 vari_short_timescale, gaia_source and epoch photometry; ZTF DR23 light curves (IRSA)", release="Gaia DR3, ZTF DR23",
        selection="main-sequence hosts faster than %.1f cycles a day with amplitudes above %.2f mag and %d or more transits: %d candidates, %d with light curves" % (d["params"]["fmin"], d["params"]["amin"], d["params"]["nmin"], d["n_cand"], d["n_lc"]),
        code=os.path.abspath(__file__), entry="roche", n_targets=d["n_lc"], n_unit="stars", params=d["params"],
        null_summary="each band's best sinusoid at the Gaia period, its double or half, plus its own residuals resampled at the same epochs, %d draws or ten times that "
                     "when fewer than ten reach the box's advantage: p below %g in the lead band and 0.05 in the others" % (d["params"]["nsim"], d["params"]["p"]),
        sensitivity="a box depth of 3 sigma of the out-of-transit scatter over the in-transit points, about 1 per cent at G 16", ledger="Transiting bodies inside the fluid Roche limit",
        result="%d box-like, %d of them beyond the sinusoid null (%.1f expected by chance in the lead band), %d with equal depths in the two colours%s" % (
            d["n_boxy"], d["n_sig"], d["params"]["p"] * d["n_lc"], d["n_found"],
            ("; in ZTF g and r " + ", ".join("%d %s" % (n, k) for k, n in collections.Counter(
                (r["verdict"] or ("without a ZTF light curve" if r["note"] == "no ZTF light curve" else "a grey box in ZTF too")).split(":")[0]
                for r in roche_ZTF_READ.values()).most_common())) if roche_ZTF_READ else
            (": " + "; ".join("Gaia DR3 %s (P %.2f h)" % (x["source_id"], x["P_h"]) for x in found[:8])) if found else ""))
    core.clear_targets("roche_transits")
    core.record_targets("roche_transits", [dict(target="gaia_dr3:%s" % x["source_id"], stat=x["bands"][x["lead"]]["advantage"],
                                                p=x["bands"][x["lead"]]["p_box"], unit="n ln(chi2 sinusoid / chi2 box)",
                                                vet=roche_vet(x), note=("box-like beyond the sinusoid null with equal depths in the two colours; " if x["grey"] and x["significant"] else "")
                                                + "P %.3f h, amp %.3f, G %.1f" % (x["P_h"], x["amp"], x["G"])
                                                + ("; " + roche_ZTF_READ[x["source_id"]]["note"] if x["source_id"] in roche_ZTF_READ else "")) for x in d["rows"]])
    print("recorded %d light curves" % d["n_lc"])


def roche_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["select", "fetch", "fetch_gaia", "analyse", "ztf", "record"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--nsim", type=int, default=2000)
    a = p.parse_args()
    if a.mode == "fetch":
        roche_fetch(a.threads)
    elif a.mode == "fetch_gaia":
        roche_fetch_gaia(a.threads)
    elif a.mode == "analyse":
        roche_analyse(a.procs, a.nsim)
    elif a.mode == "ztf":
        roche_ztf(a.threads)
    else:
        {"select": roche_select, "record": roche_record}[a.mode]()


# --------------------------------------------------------------------------------------------------------------

depthgrowth_DOC = r"""Transit-depth growth from Kepler to TESS: KOI-TOI pairs deeper in TESS at 5 sigma with duration held, the shallower side as null; vetted in SAP and PDCSAP light curves.

    python -I seti.py transits depthgrowth fetch
    python -I seti.py transits depthgrowth analyse
    python -I seti.py transits depthgrowth vet
    python -I seti.py transits depthgrowth record"""

depthgrowth_ROOT = os.environ.get("DEPTHGROWTH_ROOT", core.data("depthgrowth"))
depthgrowth_OUT = "depth_growth.json"
depthgrowth_SEP = 6.0  # arcsec
depthgrowth_DP = 2e-3  # fractional period agreement
depthgrowth_Z = 5.0
depthgrowth_DUR = 0.2


def depthgrowth_fetch():
    os.makedirs(depthgrowth_ROOT, exist_ok=True)
    # dispositions filtered here; the archive's firewall refuses some clause combinations
    kois = [k for k in core.exoplanets("SELECT kepid, kepoi_name, kepler_name, koi_disposition, koi_period, koi_duration, koi_duration_err1, koi_depth, koi_depth_err1, "
                                           "koi_impact, koi_model_snr, koi_kepmag, ra, dec FROM cumulative WHERE koi_disposition != 'FALSE POSITIVE'")
            if k["koi_disposition"] in ("CONFIRMED", "CANDIDATE") and num(k["koi_depth"])]
    tois = core.exoplanets("SELECT toi, tid, tfopwg_disp, ra, dec, st_tmag, pl_orbper, pl_orbpererr1, pl_trandurh, pl_trandurherr1, pl_trandep, pl_trandeperr1, pl_rade FROM toi "
                               "WHERE pl_trandep IS NOT NULL AND ra BETWEEN 275 AND 305 AND dec BETWEEN 30 AND 60")  # the archive's firewall refuses '<'
    json.dump(dict(kois=kois, tois=tois), open(os.path.join(depthgrowth_ROOT, "tables.json"), "w"))
    print("%d KOIs with depths, %d TOIs with depths in the Kepler field's box" % (len(kois), len(tois)))


def depthgrowth_analyse():
    d = json.load(open(os.path.join(depthgrowth_ROOT, "tables.json")))
    kois = [k for k in d["kois"] if num(k["ra"]) is not None and num(k["koi_period"])]
    kra = np.radians(np.array([float(k["ra"]) for k in kois]))
    kdec = np.radians(np.array([float(k["dec"]) for k in kois]))
    pairs = []
    for t in d["tois"]:
        ra, dec, P = num(t["ra"]), num(t["dec"]), num(t["pl_orbper"])
        if ra is None or dec is None or not P:
            continue
        sep = np.degrees(np.arccos(np.clip(np.sin(kdec) * math.sin(math.radians(dec)) + np.cos(kdec) * math.cos(math.radians(dec)) * np.cos(kra - math.radians(ra)), -1, 1))) * 3600
        for i in np.where(sep < depthgrowth_SEP)[0]:
            k = kois[i]
            Pk = float(k["koi_period"])
            if abs(P - Pk) / Pk > depthgrowth_DP:
                continue
            dk, ek = num(k["koi_depth"]), num(k["koi_depth_err1"])
            dt_, et = num(t["pl_trandep"]), num(t["pl_trandeperr1"])
            if not dk or not dt_ or not ek or not et:
                continue
            dur_k, dur_t = num(k["koi_duration"]), num(t["pl_trandurh"])
            pairs.append(dict(toi=t["toi"], tic=t["tid"], koi=k["kepoi_name"], kepler=k["kepler_name"], koi_disp=k["koi_disposition"], toi_disp=t["tfopwg_disp"],
                              P=Pk, sep=float(sep[i]), depth_k=dk, err_k=abs(ek), depth_t=dt_, err_t=abs(et), z=(dt_ - dk) / math.hypot(ek, et), ratio=dt_ / dk,
                              dur_k=dur_k, dur_t=dur_t, dur_ratio=(dur_t / dur_k) if dur_k and dur_t else None, impact=num(k["koi_impact"]),
                              kepmag=num(k["koi_kepmag"]), tmag=num(t["st_tmag"]), ra=ra, dec=dec))
    pairs.sort(key=lambda p: -p["z"])
    ratio = np.array([p["ratio"] for p in pairs])
    med = float(np.median(ratio)) if len(ratio) else float("nan")
    rs = float(1.4826 * np.median(np.abs(ratio - med))) if len(ratio) else float("nan")
    deeper = [p for p in pairs if p["z"] >= depthgrowth_Z and p["dur_ratio"] and abs(p["dur_ratio"] - 1) <= depthgrowth_DUR]
    shallower = [p for p in pairs if p["z"] <= -depthgrowth_Z and p["dur_ratio"] and abs(p["dur_ratio"] - 1) <= depthgrowth_DUR]
    print("%d KOI-TOI pairs; TESS/Kepler depth ratio median %.3f, robust scatter %.3f; deeper in TESS at %.0f sigma with the duration held: %d; shallower (the null side): %d"
          % (len(pairs), med, rs, depthgrowth_Z, len(deeper), len(shallower)))
    if deeper:
        types = core.simbad_otypes(["TIC %s" % p["tic"] for p in deeper])
        for p in deeper:
            p["otypes"] = sorted(types.get("TIC %s" % p["tic"], []))
    for p in deeper[:40]:
        print("  TOI %s = %s %s (%s; TOI %s): P %.4f d, depth %.0f +- %.0f ppm in Kepler, %.0f +- %.0f in TESS (%.2fx, %+.1f sigma), duration %.2f h against %.2f, b %s, Kp %s, Tmag %s, SIMBAD %s" % (
            p["toi"], p["koi"], p["kepler"] or "", p["koi_disp"], p["toi_disp"], p["P"], p["depth_k"], p["err_k"], p["depth_t"], p["err_t"], p["ratio"], p["z"],
            p["dur_t"] or 0, p["dur_k"] or 0, p["impact"], p["kepmag"], p["tmag"], ",".join(p.get("otypes", []))[:60]))
    json.dump(dict(pairs=pairs, deeper=deeper, shallower=shallower, median_ratio=med, scatter=rs,
                   params=dict(sep=depthgrowth_SEP, dp=depthgrowth_DP, z=depthgrowth_Z, dur=depthgrowth_DUR)), open(core.results(depthgrowth_OUT), "w"), indent=1)


depthgrowth_VET = "depth_growth_vet.json"
depthgrowth_BKJD, depthgrowth_BTJD = 2454833.0, 2457000.0  # Kepler and TESS BJD zero points


depthgrowth_T10 = 15000.0  # e-/s at TESS magnitude 10 (TESS Instrument Handbook)


def depthgrowth_lc(b, cols):
    """({column: (time, flux over median)}, header) with flagged cadences dropped; header holds CROWDSAP and, for TESS, the target's share of the SAP flux."""
    from astropy.io import fits
    with fits.open(io.BytesIO(b)) as h:
        x, crowd, flfr, tmag = h[1].data, h[1].header.get("CROWDSAP"), h[1].header.get("FLFRCSAP"), h[0].header.get("TESSMAG")
        names = x.columns.names
        t = np.asarray(x["TIME"], float)
        q = np.asarray(x["QUALITY"] if "QUALITY" in names else x["SAP_QUALITY"], int)
        out, sap = {}, None
        for c in cols:
            if c in names:
                f = np.asarray(x[c], float)
                ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
                if ok.sum() > 50:
                    out[c] = (t[ok], f[ok] / np.median(f[ok]))
                    if c == "SAP_FLUX":
                        sap = float(np.median(f[ok]))
    share = None
    if sap and flfr and tmag is not None and sap > 0:
        share = depthgrowth_T10 * 10 ** (-0.4 * (float(tmag) - 10)) * float(flfr) / sap
    return out, dict(crowd=float(crowd) if crowd is not None else None, share=share, sap=sap)


def depthgrowth_transits(t, f, P, t0, dur):
    """[(epoch, depth, error)] per transit: central-half mean below a line fitted 0.75-2.5 durations either side."""
    out = []
    for n in range(int(np.floor((t.min() - t0) / P)), int(np.ceil((t.max() - t0) / P)) + 1):
        dt = t - (t0 + n * P)
        near = np.abs(dt) < 2.5 * dur
        dt, ff = dt[near], f[near]
        inn, side = np.abs(dt) < 0.25 * dur, np.abs(dt) > 0.75 * dur
        if inn.sum() < 1 or (side & (dt < 0)).sum() < 3 or (side & (dt > 0)).sum() < 3:
            continue
        a, b = np.polyfit(dt[side], ff[side], 1)
        r = ff / (a * dt + b)
        sd = 1.4826 * np.median(np.abs(r[side] - np.median(r[side])))
        out.append((n, float(1 - np.mean(r[inn])), float(sd * math.sqrt(1 / inn.sum() + 1 / side.sum()))))
    return out


def depthgrowth_combine(tr):
    """Weighted mean depth (ppm) of transits, error widened by their scatter."""
    if not tr:
        return None
    d, e = np.array([x[1] for x in tr]), np.maximum(np.array([x[2] for x in tr]), 1e-7)
    w = 1 / e ** 2
    m = float(np.sum(w * d) / np.sum(w))
    chi = float(np.sum(w * (d - m) ** 2) / max(len(d) - 1, 1))
    return dict(depth=round(1e6 * m, 1), err=round(1e6 * math.sqrt(max(chi, 1.0) / np.sum(w)), 1), n=len(d), chi2nu=round(chi, 2))


def depthgrowth_verdict(r):
    """Light-curve explanation of a pair or None: deeper side, PDCSAP-only growth is crowding or the TOI table; shallower side, neighbours' light or aperture light beyond the crowding model."""
    k, raw, cor = r["kepler"].get("PDCSAP_FLUX"), r["tess"].get("SAP_FLUX"), r["tess"].get("PDCSAP_FLUX")
    light = r["tess"].get("light")
    if not k or not raw:
        return None

    def z(x, frac=0.0):
        return (x["depth"] - k["depth"]) / math.sqrt(x["err"] ** 2 + k["err"] ** 2 + (frac * x["depth"]) ** 2)
    told = "TESS %.0f +- %.0f ppm uncorrected%s against Kepler's %.0f +- %.0f" % (
        raw["depth"], raw["err"], ", %.0f +- %.0f corrected" % (cor["depth"], cor["err"]) if cor else "", k["depth"], k["err"])
    if r["side"] == "deeper":
        if z(raw) > 3:
            return None
        if cor and z(cor) > 3:
            return "crowding correction: %s, CROWDSAP %s" % (told, r["tess_crowdsap"])
        return "TOI depth not in the light curves: %s" % told
    if z(raw) < -3 and (not cor or z(cor) >= -3):
        return "neighbours' light in TESS's aperture: %s, CROWDSAP %s" % (told, r["tess_crowdsap"])
    if z(raw) < -3 and light and abs(z(light, 0.1)) <= 3:
        shares = [s["share"] for s in r["sectors"] if s.get("share")]
        return ("neighbours' light in TESS's aperture beyond the crowding model: %s, the target giving %.2f-%.2f of the "
                "aperture's light by its TESS magnitude against CROWDSAP %s, over which its depth is %.0f +- %.0f"
                % (told, min(shares), max(shares), r["tess_crowdsap"], light["depth"], light["err"]))
    if z(raw) >= -3:
        return "TOI depth not in the light curves: %s" % told
    return None


def depthgrowth_vet():
    """Measure every pair in the light curves into RESULTS/depth_growth_vet.json."""
    d = json.load(open(core.results(depthgrowth_OUT)))
    pairs = [("deeper", p) for p in d["deeper"]] + [("shallower", p) for p in d["shallower"]]
    eph_t = {r["toi"]: r for r in core.exoplanets("SELECT toi, pl_tranmid FROM toi WHERE toi IN (%s)" % ", ".join(p["toi"] for _, p in pairs))}
    eph_k = {r["kepoi_name"]: r for r in core.exoplanets("SELECT kepoi_name, kepid, koi_time0bk FROM cumulative WHERE kepoi_name IN (%s)"
                                                             % ", ".join("'%s'" % p["koi"] for _, p in pairs))}
    out = []
    for side, p in pairs:
        P, dur = p["P"], p["dur_k"] / 24.0
        r = dict(side=side, toi=p["toi"], koi=p["koi"], kepler_name=p["kepler"], tic=p["tic"], depth_koi=p["depth_k"], depth_toi=p["depth_t"])
        series, crowd = {"PDCSAP_FLUX": [], "SAP_FLUX": []}, []
        for key in core.kepler_keys(int(eph_k[p["koi"]]["kepid"]), short=False):
            b = core.get(core.S3 + key, timeout=300, retries=3, soft=True, pause=2)
            if b and b.startswith(b"SIMPLE"):
                cols, c = depthgrowth_lc(b, tuple(series))
                crowd.append(c["crowd"])
                for name, v in cols.items():
                    series[name].append(v)
        t0 = float(eph_k[p["koi"]]["koi_time0bk"])
        r["kepler"] = {name: depthgrowth_combine([x for t, f in v for x in depthgrowth_transits(t, f, P, t0, dur)]) for name, v in series.items()}
        r["kepler_crowdsap"] = round(float(np.median([c for c in crowd if c is not None])), 3) if any(c is not None for c in crowd) else None
        t0 = float(eph_t[p["toi"]]["pl_tranmid"]) - depthgrowth_BTJD
        series, crowd, light, r["sectors"] = {}, [], [], []
        for s in core.sectors(p["ra"], p["dec"]):
            src, b = "spoc", core.spoc_bytes(int(p["tic"]), s)
            if not b:
                src, b = core.ffi_bytes(int(p["tic"]), s)
            if not b:
                continue
            cols, c = depthgrowth_lc(b, ("SAP_FLUX", "PDCSAP_FLUX"))
            row = dict(sector=s, source=src, crowdsap=c["crowd"], share=c["share"], sap_e_s=c["sap"])
            for name, v in cols.items():
                tr = depthgrowth_transits(v[0], v[1], P, t0, dur)
                row[name] = depthgrowth_combine(tr)
                series.setdefault(name, []).append(v)
                if name == "SAP_FLUX" and c["share"]:
                    # SAP depth over the target's share of the aperture light
                    light += [(n, d / c["share"], e / c["share"]) for n, d, e in tr]
            if c["crowd"] is not None:
                crowd.append(c["crowd"])
            r["sectors"].append(row)
        r["tess"] = {name: depthgrowth_combine([x for t, f in v for x in depthgrowth_transits(t, f, P, t0, dur)]) for name, v in series.items()}
        r["tess"]["light"] = depthgrowth_combine(light)
        r["tess_crowdsap"] = "%.2f-%.2f" % (min(crowd), max(crowd)) if crowd else None
        r["verdict"] = depthgrowth_verdict(r)
        print("%s %s (%s, TOI %s): Kepler %s; TESS %s over %d sectors; %s"
              % (side, p["koi"], p["kepler"] or "", p["toi"], r["kepler"].get("PDCSAP_FLUX"), r["tess"], len(r["sectors"]), r["verdict"] or "unexplained"), flush=True)
        out.append(r)
    json.dump(out, open(core.results(depthgrowth_VET), "w"), indent=1)


def depthgrowth_record():
    d = json.load(open(core.results(depthgrowth_OUT)))
    deeper = d["deeper"]
    vets = {r["koi"]: r for r in json.load(open(core.results(depthgrowth_VET)))} if os.path.exists(core.results(depthgrowth_VET)) else {}
    for r in vets.values():
        r["verdict"] = depthgrowth_verdict(r)

    def told(p):
        r = vets.get(p["koi"])
        if not r:
            return ""
        k, raw, cor = r["kepler"].get("PDCSAP_FLUX"), r["tess"].get("SAP_FLUX"), r["tess"].get("PDCSAP_FLUX")
        return "; light curves: Kepler %s, TESS %s uncorrected, %s corrected" % tuple(
            "%.0f +- %.0f ppm" % (x["depth"], x["err"]) if x else "-" for x in (k, raw, cor))
    side = {}
    for name in ("deeper", "shallower"):
        c = collections.Counter()
        for p in d[name]:
            r = vets.get(p["koi"])
            c[(r["verdict"].split(":")[0] if r["verdict"] else "unexplained") if r else "not measured"] += 1
        side[name] = ", ".join("%d %s" % (n, k) for k, n in c.most_common())
    core.record_search(
        "transit_depth_growth", "Transit-depth growth, Kepler to TESS",
        archive="NASA Exoplanet Archive: Kepler cumulative KOI table, TESS TOI table; Kepler and TESS light curves (STScI AWS mirror, MAST HLSP)",
        release="as served", selection="KOIs (confirmed or candidate) matched to TOIs within %.0f arcsec and %.1f per cent in period" % (d["params"]["sep"], 100 * d["params"]["dp"]),
        code=os.path.abspath(__file__), entry="depthgrowth", n_targets=len(d["pairs"]), n_unit="KOI-TOI pairs", params=d["params"],
        null_summary="the shallower side of the same comparison: %d pairs at %.0f sigma, measured in the light curves as the deeper side is" % (
            len(d["shallower"]), d["params"]["z"]),
        sensitivity="the catalogue depth errors combined; median TESS/Kepler ratio %.3f with robust scatter %.3f" % (d["median_ratio"], d["scatter"]),
        ledger="Transit-depth growth, Kepler to TESS",
        result="%d pairs; %d deeper in TESS at %.0f sigma with the duration held, in the light curves %s; %d shallower, %s" % (
            len(d["pairs"]), len(deeper), d["params"]["z"], side["deeper"], len(d["shallower"]), side["shallower"]))
    core.clear_targets("transit_depth_growth")
    core.record_targets("transit_depth_growth", [dict(target="tic:%s" % p["tic"], stat=p["z"], unit="TESS minus Kepler depth over the combined error",
                                                          vet=vets[p["koi"]]["verdict"] if p["koi"] in vets else None,
                                                          note="%s / TOI %s, %.2fx in the catalogues, durations %.2f and %.2f h%s" % (
                                                              p["koi"], p["toi"], p["ratio"], p["dur_t"] or 0, p["dur_k"] or 0, told(p)))
                                                     for p in d["pairs"]])
    print("recorded %d pairs" % len(d["pairs"]))


def depthgrowth_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": depthgrowth_fetch, "analyse": depthgrowth_analyse, "vet": depthgrowth_vet, "record": depthgrowth_record}[mode]()


# --------------------------------------------------------------------------------------------------------------
# planetary periods matched to ours, and periods in equal steps

earthyear_DOC = r"""Planets on Earth's year: NASA Exoplanet Archive composite planets whose period lies within 0.2 per cent of the sidereal year, a period chosen to match ours, against the density of periods between 300 and 430 d.

    python -I seti.py transits earthyear [record] [--tolerance 0.002] [--window-d 300 430]"""

YEAR_D = 365.25636


def earthyear_main(record=False, tolerance=0.002, window_d=[300.0, 430.0]):
    from scipy.stats import poisson
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbper, pl_orbpererr1, disc_method FROM pscomppars WHERE pl_orbper BETWEEN %g AND %g"
                           % (window_d[0], window_d[1]))
    P = np.array([float(r["pl_orbper"]) for r in rows])
    width = tolerance * YEAR_D
    expect = len(P) / (window_d[1] - window_d[0]) * 2 * width
    hits = [r for r in rows if abs(float(r["pl_orbper"]) - YEAR_D) < width]
    p = float(poisson.sf(len(hits) - 1, expect)) if hits else 1.0
    print("%d planets at 300-430 d; within 0.2 per cent of the sidereal year %d against %.2f expected (Poisson p %.2f): %s" % (
        len(P), len(hits), expect, p, ["%s %.4f d (%s)" % (r["pl_name"], float(r["pl_orbper"]), r["disc_method"]) for r in hits]))
    json.dump(dict(n=len(P), expect=expect, p=p, hits=hits), open(core.results("earthyear_planets.json"), "w"), indent=1)
    if record:
        core.record_search(
            "exoplanets_earth_year", "Planets on Earth's year", archive="NASA Exoplanet Archive pscomppars",
            selection="composite planets with periods of 300-430 d: %d" % len(P), code=os.path.abspath(__file__), entry="earthyear",
            n_targets=len(P), n_unit="planets", params=dict(tolerance=tolerance, window_d=window_d, year_d=YEAR_D),
            null_summary="the density of periods at 300-430 d: %.2f expected within the tolerance" % expect,
            sensitivity="a planet whose period matches the sidereal year to 0.2 per cent", ledger="Planets on Earth's year",
            result="%d against %.2f expected: %s" % (len(hits), expect, ", ".join(r["pl_name"] for r in hits) or "none"))
        core.clear_targets("exoplanets_earth_year")
        core.record_targets("exoplanets_earth_year", [dict(
            target="planet:%s" % r["pl_name"], stat=float(r["pl_orbper"]) / YEAR_D - 1, unit="fractional period offset from the sidereal year",
            vet="chance-rate" if p > 0.01 else None, note="%.4f d, %s" % (float(r["pl_orbper"]), r["disc_method"])) for r in hits])
        print("recorded")


steps_DOC = r"""Planetary systems in equal steps of period: consecutive triples in the NASA Exoplanet Archive's systems of three or more planets whose two period differences agree to 0.5 per cent, a spacing no resonance asks for, each read for a chain n : n+1 : n+2 (arithmetic and resonant at once); triples with equal period ratios are counted beside them.

    python -I seti.py transits steps [record] [--tolerance 0.005] [--chain-tol 0.02]"""


def steps_main(record=False, tolerance=0.005, chain_tol=0.02):
    rows = core.exoplanets("SELECT hostname, pl_name, pl_orbper FROM pscomppars WHERE sy_pnum >= 3 AND pl_orbper IS NOT NULL")
    sysd = {}
    for r in rows:
        sysd.setdefault(r["hostname"], []).append((float(r["pl_orbper"]), r["pl_name"]))
    out, geo, n = [], 0, 0
    for h, ps in sysd.items():
        ps.sort()
        for i in range(len(ps) - 2):
            (a, na), (b, nb), (c, nc) = ps[i:i + 3]
            n += 1
            geo += abs((c / b) - (b / a)) / (b / a) < tolerance
            if abs((c - b) - (b - a)) / (b - a) < tolerance:
                k = a / (b - a)
                chain = abs(k - round(k)) < chain_tol * max(round(k), 1)
                out.append(dict(host=h, planets=[na, nb, nc], periods=[a, b, c], k=k, chain="%d:%d:%d" % (round(k), round(k) + 1, round(k) + 2) if chain else None))
    print("%d consecutive triples in %d systems; equal steps to 0.5 per cent %d, equal ratios %d" % (n, len(sysd), len(out), geo))
    for o in out:
        print("  %s %s: %s, %s" % (o["host"], [round(x, 3) for x in o["periods"]], "chain " + o["chain"] if o["chain"] else "no integer chain",
                                    o["planets"]))
    json.dump(dict(n=n, systems=len(sysd), equal_ratios=geo, steps=out), open(core.results("period_steps.json"), "w"), indent=1)
    if record:
        core.record_search(
            "exoplanets_period_steps", "Planetary systems in equal period steps", archive="NASA Exoplanet Archive pscomppars",
            selection="systems of three or more planets: %d, %d consecutive triples" % (len(sysd), n), code=os.path.abspath(__file__),
            entry="steps", n_targets=n, n_unit="planet triples", params=dict(tolerance=tolerance, chain_tol=chain_tol),
            null_summary="triples with equal period ratios, the resonant chains: %d" % geo,
            sensitivity="three neighbouring planets whose period differences agree to 0.5 per cent", ledger="Planetary systems in equal period steps",
            result="%d triples, every one %s" % (len(out), "an n : n+1 : n+2 chain" if all(o["chain"] for o in out) else "read individually"))
        core.clear_targets("exoplanets_period_steps")
        core.record_targets("exoplanets_period_steps", [dict(
            target="system:%s" % o["host"], stat=o["k"], unit="first period over the step",
            vet="near-resonance" if o["chain"] else None, note="%s: %s d, chain %s" % (", ".join(o["planets"]), ", ".join("%.3f" % x for x in o["periods"]), o["chain"]))
            for o in out])
        print("recorded")


earthday_DOC = r"""Planets on Earth's day: NASA Exoplanet Archive composite planets whose period lies between the sidereal day less 0.2 per cent and the solar day plus 0.2 per cent, a period chosen to match Earth's turning, against the density of periods at 0.7-1.4 d, each read for its discovery method, since a ground-based survey's nights alias its periods at one day.

    python -I seti.py transits earthday [record] [--tolerance 0.002] [--window-d 0.7 1.4]"""

DAY_SIDEREAL = 0.99726957


def earthday_main(record=False, tolerance=0.002, window_d=[0.7, 1.4]):
    from scipy.stats import poisson
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbper, pl_orbpererr1, disc_method, disc_facility FROM pscomppars "
                           "WHERE pl_orbper BETWEEN %g AND %g" % (window_d[0], window_d[1]))
    lo, hi = DAY_SIDEREAL * (1 - tolerance), 1 + tolerance
    P = np.array([float(r["pl_orbper"]) for r in rows])
    expect = float(np.sum((P < lo) | (P > hi))) / ((window_d[1] - window_d[0]) - (hi - lo)) * (hi - lo)
    hits = sorted([r for r in rows if lo <= float(r["pl_orbper"]) <= hi], key=lambda r: float(r["pl_orbper"]))
    p = float(poisson.sf(len(hits) - 1, expect)) if hits else 1.0
    print("%d planets at 0.7-1.4 d; between %.5f and %.5f d %d against %.2f expected (Poisson p %.2f)" % (len(P), lo, hi, len(hits), expect, p))
    for r in hits:
        print("  %s: %.6f d +- %s (%s, %s)" % (r["pl_name"], float(r["pl_orbper"]), r["pl_orbpererr1"], r["disc_method"], r["disc_facility"]))
    json.dump(dict(n=len(P), expect=expect, p=p, hits=hits), open(core.results("earthday_planets.json"), "w"), indent=1)
    if record:
        core.record_search(
            "exoplanets_earth_day", "Planets on Earth's day", archive="NASA Exoplanet Archive pscomppars",
            selection="composite planets with periods of 0.7-1.4 d: %d" % len(P), code=os.path.abspath(__file__), entry="earthday",
            n_targets=len(P), n_unit="planets", params=dict(window_d=[lo, hi], tolerance=tolerance, range_d=window_d),
            null_summary="the density of periods at 0.7-1.4 d outside the window: %.2f expected inside it" % expect,
            sensitivity="a planet whose period matches the sidereal or solar day to 0.2 per cent", ledger="Planets on Earth's day",
            result="%d against %.2f expected: %s" % (len(hits), expect, ", ".join("%s (%s)" % (r["pl_name"], r["disc_facility"]) for r in hits) or "none"))
        core.clear_targets("exoplanets_earth_day")
        core.record_targets("exoplanets_earth_day", [dict(
            target="planet:%s" % r["pl_name"], stat=float(r["pl_orbper"]), unit="period, d",
            vet="chance-rate" if p > 0.01 else None, note="%s, %s" % (r["disc_method"], r["disc_facility"])) for r in hits])
        print("recorded")


sharedperiod_DOC = r"""Planets on one clock: pairs of planets in different systems whose periods, each measured to 5e-7 or better, agree to 1e-6 and within three times their combined error, a period standardised across systems, against the pairs within 1e-4 scaled to the tolerance, since a smooth distribution of periods gives pairs in proportion to it.

    python -I seti.py transits sharedperiod [record] [--precision 5e-7] [--tolerance 1e-6] [--sigma 3]"""


def sharedperiod_main(record=False, precision=5e-7, tolerance=1e-6, sigma=3.0):
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbper, pl_orbpererr1, pl_orbpererr2, disc_method FROM pscomppars "
                           "WHERE pl_orbper IS NOT NULL AND pl_orbpererr1 IS NOT NULL")
    good = []
    for r in rows:
        P, e = float(r["pl_orbper"]), max(abs(float(r["pl_orbpererr1"] or 0)), abs(float(r["pl_orbpererr2"] or 0)))
        if 0 < e < precision * P:
            good.append((P, e, r))
    good.sort(key=lambda t: t[0])
    t1, t2, t3 = tolerance, 10 * tolerance, 100 * tolerance
    counts, pairs = {t1: 0, t2: 0, t3: 0}, []
    for i, (P, e, r) in enumerate(good):
        for j in range(i + 1, len(good)):
            Q, f, s = good[j]
            d = (Q - P) / P
            if d > t3:
                break
            if r["hostname"] == s["hostname"]:
                continue
            for t in counts:
                counts[t] += d < t
            if d < t2:
                pairs.append(dict(a=r["pl_name"], b=s["pl_name"], pa=P, pb=Q, frac=d, sigma=(Q - P) / np.hypot(e, f)))
    expect = counts[t3] / 100.0
    within = [x for x in pairs if x["frac"] < t1 and x["sigma"] < sigma]
    print("%d planets with periods to %g; pairs in different systems within %g: %d, %g: %d, %g: %d against %.2f scaled; within "
          "%g combined errors at %g: %d" % (len(good), precision, t3, counts[t3], t2, counts[t2], t1, counts[t1], expect, sigma, t1, len(within)))
    for x in sorted(pairs, key=lambda x: x["frac"])[:15]:
        print("  %s %.7f d and %s %.7f d: %.1e apart, %.1f combined sigma" % (x["a"], x["pa"], x["b"], x["pb"], x["frac"], x["sigma"]))
    json.dump(dict(n=len(good), counts={str(k): v for k, v in counts.items()}, expect=expect, pairs=pairs), open(core.results("shared_periods.json"), "w"), indent=1)
    if record:
        core.record_search(
            "exoplanets_shared_periods", "Planets on one clock", archive="NASA Exoplanet Archive pscomppars",
            selection="composite planets with periods measured to 5e-7: %d" % len(good), code=os.path.abspath(__file__), entry="sharedperiod",
            n_targets=len(good), n_unit="planets", params=dict(precision=precision, tolerance=tolerance, sigma=sigma),
            null_summary="pairs in different systems within %g (%d) scaled to %g: %.2f" % (t3, counts[t3], t1, expect),
            sensitivity="two planets of different systems on one period to %g and within their errors" % t1,
            ledger="Planets on one clock",
            result="%d pairs within %g against %.2f, %d of them within %g combined errors" % (counts[t1], t1, expect, len(within), sigma))
        core.clear_targets("exoplanets_shared_periods")
        core.record_targets("exoplanets_shared_periods", [dict(
            target="planet:%s" % x["a"], stat=x["frac"], unit="fractional period difference",
            vet="chance-rate" if counts[t1] <= expect + 3 * np.sqrt(max(expect, 1)) else None,
            note="with %s, %.1f combined sigma" % (x["b"], x["sigma"])) for x in pairs])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------
# periods shared across the sky by pulsating stars, eclipsing binaries and binary orbits, and systems of identical planets


def clock_pairs(P, e, ra, dec, prec, min_sep=1.0, tol=1e-6, sigma=3.0):
    """Pairs of a sample on one clock: periods measured to PREC, pairs more than MIN_SEP deg apart (a star and a blended neighbour share a period) within TOL, ten and a hundred times it, and the pairs at TOL within SIGMA combined errors as (i, j, frac, sigma)."""
    ok = np.nonzero((e > 0) & (e < prec * P))[0]
    o = ok[np.argsort(P[ok])]
    Ps = P[o]
    n = np.searchsorted(Ps, Ps * (1 + 100 * tol), side="right") - np.arange(len(Ps)) - 1
    i = np.repeat(np.arange(len(Ps)), n)
    j = i + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1
    a, b = o[i], o[j]
    frac = (P[b] - P[a]) / P[a]
    r1, d1, r2, d2 = (np.radians(v) for v in (ra[a], dec[a], ra[b], dec[b]))
    sep = np.degrees(2 * np.arcsin(np.sqrt(np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2)))
    far = sep > min_sep
    counts = {t: int(((frac < t) & far).sum()) for t in (tol, 10 * tol, 100 * tol)}
    sig = (P[b] - P[a]) / np.hypot(e[a], e[b])
    m = far & (frac < tol) & (sig < sigma)
    return len(ok), counts, list(zip(a[m], b[m], frac[m], sig[m]))


def clock_main(record, sid, approach, archive, what, query, prec, entry, tol=1e-6, min_sep=1.0, sigma=3.0):
    """One clock test on a Gaia DR3 variability or orbit table: QUERY returns source_id, ra, dec, P and its error; pairs within TOL more than MIN_SEP deg apart within SIGMA combined errors."""
    rows = core.tap_async("gaia", query, maxrec=2000000)
    P = np.array([float(r["p"]) for r in rows])
    e = np.array([float(r["e_p"]) for r in rows])
    ra = np.array([float(r["ra"]) for r in rows])
    dec = np.array([float(r["dec"]) for r in rows])
    n, counts, pairs = clock_pairs(P, e, ra, dec, prec, min_sep, tol, sigma)
    expect = counts[100 * tol] / 100.0
    print("%d %s, %d with periods to %g; pairs more than %g deg apart within %g: %d, %g: %d, %g: %d against %.1f scaled; within "
          "%g combined errors at %g: %d" % (len(rows), what, n, prec, min_sep, 100 * tol, counts[100 * tol], 10 * tol, counts[10 * tol], tol, counts[tol],
                                             expect, sigma, tol, len(pairs)))
    for a, b, f, s in sorted(pairs, key=lambda x: x[2])[:10]:
        print("  Gaia DR3 %s %.8f d and %s %.8f d: %.1e apart, %.1f combined sigma" % (rows[a]["source_id"], P[a], rows[b]["source_id"], P[b], f, s))
    if record:
        excess = counts[tol] > expect + 3 * math.sqrt(max(expect, 1))
        core.record_run(
            sid, approach, archive, "%s with periods measured to %g: %d" % (what, prec, n), os.path.abspath(__file__), entry, n, what,
            dict(precision=prec, tolerance=tol, sigma=sigma, min_sep_deg=min_sep), "pairs within %g (%d) scaled to %g: %.1f" % (100 * tol, counts[100 * tol], tol, expect),
            "%s more than %g deg apart on one period to %g within their errors" % (what, min_sep, tol),
            "%d pairs within %g against %.1f, %d of them within %g combined errors" % (counts[tol], tol, expect, len(pairs), sigma),
            [dict(target="gaia_dr3:%s" % rows[a]["source_id"], stat=float(f), unit="fractional period difference",
                  vet=None if excess else "chance-rate", note="with Gaia DR3 %s, %.1f combined sigma" % (rows[b]["source_id"], s))
             for a, b, f, s in pairs])
        print("recorded")


rrlclock_DOC = r"""RR Lyrae on one clock: Gaia DR3 RR Lyrae (vari_rrlyrae) with fundamental periods measured to 1e-5, pairs more than 1 deg apart whose periods agree to 1e-6 and within three combined errors, against the pairs within 1e-4 scaled to the tolerance.

    python -I seti.py transits rrlclock [record] [--precision 1e-5] [--tolerance 1e-6] [--min-sep-deg 1] [--sigma 3]"""


def rrlclock_main(record=False, precision=1e-5, tolerance=1e-6, min_sep_deg=1.0, sigma=3.0):
    clock_main(record, "gaia_rrlyrae_clock", "RR Lyrae on one clock", "Gaia DR3 vari_rrlyrae", "RR Lyrae",
               "SELECT v.source_id, g.ra, g.dec, v.pf AS p, v.pf_error AS e_p FROM gaiadr3.vari_rrlyrae AS v JOIN gaiadr3.gaia_source AS g "
               "ON v.source_id = g.source_id WHERE v.pf_error > 0 AND v.pf_error < %g * v.pf" % precision, precision, "rrlclock", tolerance, min_sep_deg, sigma)


cepclock_DOC = r"""Cepheids on one clock: Gaia DR3 Cepheids (vari_cepheid) with fundamental periods measured to 1e-5, pairs more than 1 deg apart whose periods agree to 1e-6 and within three combined errors, against the pairs within 1e-4 scaled to the tolerance.

    python -I seti.py transits cepclock [record] [--precision 1e-5] [--tolerance 1e-6] [--min-sep-deg 1] [--sigma 3]"""


def cepclock_main(record=False, precision=1e-5, tolerance=1e-6, min_sep_deg=1.0, sigma=3.0):
    clock_main(record, "gaia_cepheid_clock", "Cepheids on one clock", "Gaia DR3 vari_cepheid", "Cepheids",
               "SELECT v.source_id, g.ra, g.dec, v.pf AS p, v.pf_error AS e_p FROM gaiadr3.vari_cepheid AS v JOIN gaiadr3.gaia_source AS g "
               "ON v.source_id = g.source_id WHERE v.pf_error > 0 AND v.pf_error < %g * v.pf" % precision, precision, "cepclock", tolerance, min_sep_deg, sigma)


ebclock_DOC = r"""Eclipsing binaries on one clock: Gaia DR3 eclipsing binaries (vari_eclipsing_binary) with orbital frequencies measured to 1e-6, pairs more than 1 deg apart whose periods agree to 1e-6 and within three combined errors, against the pairs within 1e-4 scaled to the tolerance.

    python -I seti.py transits ebclock [record] [--precision 1e-6] [--tolerance 1e-6] [--min-sep-deg 1] [--sigma 3]"""


def ebclock_main(record=False, precision=1e-6, tolerance=1e-6, min_sep_deg=1.0, sigma=3.0):
    clock_main(record, "gaia_eclipser_clock", "Eclipsing binaries on one clock", "Gaia DR3 vari_eclipsing_binary", "eclipsing binaries",
               "SELECT v.source_id, g.ra, g.dec, 1.0 / v.frequency AS p, v.frequency_error / (v.frequency * v.frequency) AS e_p "
               "FROM gaiadr3.vari_eclipsing_binary AS v JOIN gaiadr3.gaia_source AS g ON v.source_id = g.source_id "
               "WHERE v.frequency_error > 0 AND v.frequency_error < %g * v.frequency" % precision, precision, "ebclock", tolerance, min_sep_deg, sigma)


nssclock_DOC = r"""Binary orbits on one clock: Gaia DR3 two-body orbits (nss_two_body_orbit) with periods measured to 1e-5, pairs more than 1 deg apart whose periods agree to 1e-6 and within three combined errors, against the pairs within 1e-4 scaled to the tolerance.

    python -I seti.py transits nssclock [record] [--precision 1e-5] [--tolerance 1e-6] [--min-sep-deg 1] [--sigma 3]"""


def nssclock_main(record=False, precision=1e-5, tolerance=1e-6, min_sep_deg=1.0, sigma=3.0):
    clock_main(record, "gaia_orbit_clock", "Binary orbits on one clock", "Gaia DR3 nss_two_body_orbit", "binary orbits",
               "SELECT n.source_id, g.ra, g.dec, n.period AS p, n.period_error AS e_p FROM gaiadr3.nss_two_body_orbit AS n JOIN gaiadr3.gaia_source AS g "
               "ON n.source_id = g.source_id WHERE n.period_error > 0 AND n.period_error < %g * n.period" % precision, precision, "nssclock", tolerance, min_sep_deg, sigma)


uniform_DOC = r"""Systems built from identical planets: composite systems of three or more planets, every one with mass measured (not estimated from radius) and radius measured to 15 per cent, whose masses and whose radii are each consistent with one value: the chi-square of each about its error-weighted mean, flagged where both have p above 0.05.

    python -I seti.py transits uniform [record] [--precision 0.15] [--p-min 0.05]"""


def uniform_main(record=False, precision=0.15, p_min=0.05):
    from scipy.stats import chi2
    rows = pcp("hostname, pl_name, pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_rade, pl_radeerr1, pl_radeerr2, sy_pnum",
               "sy_pnum >= 3 AND pl_bmasse IS NOT NULL AND pl_rade IS NOT NULL AND pl_bmassprov = 'Mass'")
    sysd = {}
    for r in rows:
        if (rel_err(r, "pl_bmasse") or 1) <= precision and (rel_err(r, "pl_rade") or 1) <= precision:
            sysd.setdefault(r["hostname"], []).append(r)
    full = {h: ps for h, ps in sysd.items() if len(ps) >= 3 and len(ps) == int(ps[0]["sy_pnum"])}

    def one_value(ps, key):
        v = np.array([float(p[key]) for p in ps])
        e = np.array([rel_err(p, key) * float(p[key]) for p in ps])
        m = float(np.sum(v / e ** 2) / np.sum(1 / e ** 2))
        c = float(np.sum(((v - m) / e) ** 2))
        return m, c, float(chi2.sf(c, len(v) - 1))
    res = {h: (one_value(ps, "pl_bmasse"), one_value(ps, "pl_rade"), len(ps)) for h, ps in full.items()}
    hits = sorted(h for h, (m, r, n) in res.items() if m[2] > p_min and r[2] > p_min)
    print("%d systems of three or more planets with every mass and radius to 15 per cent; masses and radii both consistent with one value: %d %s" % (
        len(full), len(hits), hits))
    for h, (m, r, n) in sorted(res.items(), key=lambda kv: -min(kv[1][0][2], kv[1][1][2])):
        print("  %s (%d planets): mass chi-square %.0f (p %.2g), radius chi-square %.0f (p %.2g)" % (h, n, m[1], m[2], r[1], r[2]))
    if record:
        core.record_run(
            "exoplanets_identical_systems", "Systems built from identical planets", "NASA Exoplanet Archive pscomppars",
            "systems of three or more planets, every one with measured mass and radius to 15 per cent: %d" % len(full), os.path.abspath(__file__),
            "uniform", len(full), "planetary systems", dict(precision=precision, p_min=p_min), "chi-square of the masses and of the radii about their means",
            "a system whose planets all share one mass and radius within their errors", "%d; the most alike %s" % (len(hits), "; ".join(
                "%s (mass p %.1g, radius p %.1g)" % (h, m[2], r[2]) for h, (m, r, n) in sorted(res.items(), key=lambda kv: -min(kv[1][0][2], kv[1][1][2]))[:3])),
            [dict(target="system:%s" % h, stat=min(m[2], r[2]), unit="smaller p of one mass and one radius") for h, (m, r, n) in res.items()])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------
# planets past physical limits, copies of planets and systems, and period ratios that spell constants


def pcp(cols, where):
    """Rows of the Exoplanet Archive's composite planet table."""
    return core.exoplanets("SELECT %s FROM pscomppars WHERE %s" % (cols, where))


def rel_err(r, key):
    """The larger fractional error of a pscomppars value, or None."""
    v, e1, e2 = num(r.get(key)), num(r.get(key + "err1")), num(r.get(key + "err2"))
    if not v or e1 is None:
        return None
    return max(abs(e1), abs(e2 or 0)) / abs(v)


rocheplanets_DOC = r"""Planets inside their own Roche limit: composite planets with mass and radius measured to a third or better, the shortest period a fluid body of their density survives, 12.6 h (density / 1 g cm^-3)^-1/2 (Rappaport et al. 2013), against their period, flagged where the period is shorter at 3 sigma, which only a rigid structure survives.

    python -I seti.py transits rocheplanets [record] [--precision 0.333] [--sigma 3] [--roche-h 12.6]"""


def rocheplanets_main(record=False, precision=1 / 3.0, sigma=3.0, roche_h=12.6):
    rows = pcp("pl_name, pl_orbper, pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_rade, pl_radeerr1, pl_radeerr2",
               "pl_bmasse IS NOT NULL AND pl_rade IS NOT NULL AND pl_orbper IS NOT NULL AND pl_bmassprov = 'Mass'")
    out = []
    for r in rows:
        if num(r["pl_orbper"]) is None:
            continue
        em, er = rel_err(r, "pl_bmasse"), rel_err(r, "pl_rade")
        if em is None or er is None or em > precision or er > precision:
            continue
        rho = 5.514 * float(r["pl_bmasse"]) / float(r["pl_rade"]) ** 3
        erho = rho * math.hypot(em, 3 * er)
        pmin = roche_h * rho ** -0.5
        epmin = 0.5 * pmin * erho / rho
        p_h = float(r["pl_orbper"]) * 24
        out.append(dict(name=r["pl_name"], p_h=p_h, rho=rho, pmin=pmin, z=(pmin - p_h) / epmin if epmin > 0 else 0.0))
    inside = sorted([o for o in out if o["z"] > sigma], key=lambda o: -o["z"])
    print("%d planets with mass and radius to a third; inside the fluid Roche limit at 3 sigma: %d" % (len(out), len(inside)))
    for o in sorted(out, key=lambda o: -o["z"])[:8]:
        print("  %s: P %.2f h, density %.2f g/cm3, shortest fluid period %.2f h, %.1f sigma inside" % (o["name"], o["p_h"], o["rho"], o["pmin"], o["z"]))
    json.dump(out, open(core.results("roche_planets.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanets_inside_roche", "Planets inside their own Roche limit", "NASA Exoplanet Archive pscomppars",
            "planets with measured mass and radius to a third: %d" % len(out), os.path.abspath(__file__), "rocheplanets", len(out), "planets",
            dict(sigma=sigma, roche_h=roche_h, precision=precision), "each planet's own errors", "a planet orbiting faster than a fluid body of its density survives",
            "%d inside at 3 sigma; the nearest %s" % (len(inside), "; ".join("%s %.1f sigma" % (o["name"], o["z"]) for o in sorted(out, key=lambda o: -o["z"])[:3])),
            [dict(target="planet:%s" % o["name"], stat=o["z"], unit="sigma inside the fluid Roche limit") for o in inside])
        print("recorded")


copyplanets_DOC = r"""Copy planets in different systems: composite planets with mass and radius measured to 3 per cent, pairs in different systems agreeing to 1 per cent in radius, mass and period, against the pairs found with the masses shuffled among planets of like radius fifty times.

    python -I seti.py transits copyplanets [record] [--tol 0.01] [--precision 0.03] [--shuffles 50]"""


def copyplanets_main(record=False, tol=0.01, precision=0.03, shuffles=50):
    rows = pcp("pl_name, hostname, pl_orbper, pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_rade, pl_radeerr1, pl_radeerr2",
               "pl_bmasse IS NOT NULL AND pl_rade IS NOT NULL AND pl_bmassprov = 'Mass'")
    good = [r for r in rows if (rel_err(r, "pl_bmasse") or 1) <= precision and (rel_err(r, "pl_rade") or 1) <= precision]
    R = np.array([float(r["pl_rade"]) for r in good])
    M = np.array([float(r["pl_bmasse"]) for r in good])
    P = np.array([float(r["pl_orbper"]) for r in good])
    H = [r["hostname"] for r in good]

    def count(m):
        n = []
        for i in range(len(good)):
            for j in range(i + 1, len(good)):
                if H[i] != H[j] and abs(R[i] / R[j] - 1) < tol and abs(m[i] / m[j] - 1) < tol and abs(P[i] / P[j] - 1) < tol:
                    n.append((i, j))
        return n
    found = count(M)
    rng = np.random.default_rng(9)
    order = np.argsort(R)
    null = []
    for _ in range(shuffles):
        m = M.copy()
        for s in range(0, len(order), 10):
            blk = order[s:s + 10]
            m[blk] = m[rng.permutation(blk)]
        null.append(len(count(m)))
    print("%d planets with mass and radius to 3 per cent; copies in different systems %d; with masses shuffled among planets of like radius: "
          "mean %.2f" % (len(good), len(found), float(np.mean(null))))
    for i, j in found:
        print("  %s and %s: R %.2f/%.2f, M %.1f/%.1f, P %.3f/%.3f" % (good[i]["pl_name"], good[j]["pl_name"], R[i], R[j], M[i], M[j], P[i], P[j]))
    if record:
        core.record_run(
            "exoplanets_copies", "Copy planets in different systems", "NASA Exoplanet Archive pscomppars",
            "planets with mass and radius to 3 per cent: %d" % len(good), os.path.abspath(__file__), "copyplanets", len(good), "planets",
            dict(tol=tol, precision=precision, shuffles=shuffles), "masses shuffled among planets of like radius: mean %.2f" % float(np.mean(null)),
            "two planets of different systems alike to 1 per cent in radius, mass and period",
            "%d pairs against %.2f" % (len(found), float(np.mean(null))),
            [dict(target="planet:%s" % good[i]["pl_name"], note="copy of %s" % good[j]["pl_name"]) for i, j in found])
        print("recorded")


dayside_DOC = r"""Daysides hotter than their stars allow: every brightness temperature of the Exoplanet Archive's emission-spectroscopy table against the hottest dayside a planet can have from its star, T_eff (R*/a)^1/2 (2/3)^1/4 with zero albedo and no redistribution (pscomppars T_eff and a/R*), flagged where it exceeds that at 3 sigma.

    python -I seti.py transits dayside [record] [--sigma 3]"""


def dayside_main(record=False, sigma=3.0):
    em = core.exoplanets("SELECT plntname, centralwavelng, espbritemp, espbritemperr1, espbritemperr2, facility, instrument FROM emissionspec "
                         "WHERE espbritemp IS NOT NULL")
    host = {r["pl_name"]: r for r in pcp("pl_name, st_teff, pl_ratdor", "st_teff IS NOT NULL AND pl_ratdor IS NOT NULL")}
    out = []
    for r in em:
        h = host.get(r["plntname"])
        if not h:
            continue
        tmax = float(h["st_teff"]) / math.sqrt(float(h["pl_ratdor"])) * (2 / 3.0) ** 0.25
        t, e = float(r["espbritemp"]), abs(num(r["espbritemperr2"]) or num(r["espbritemperr1"]) or 0)
        out.append(dict(planet=r["plntname"], wl=num(r["centralwavelng"]), t=t, e=e, tmax=tmax, z=(t - tmax) / e if e > 0 else 0.0,
                        inst="%s %s" % (r["facility"], r["instrument"])))
    hot = sorted([o for o in out if o["z"] > sigma], key=lambda o: -o["z"])
    print("%d brightness temperatures with host temperature and a/R*; above the hottest possible dayside at 3 sigma: %d over %d planets" % (
        len(out), len(hot), len({o["planet"] for o in hot})))
    for o in hot[:15]:
        print("  %s at %s um: %.0f +- %.0f K against %.0f K (%.1f sigma), %s" % (o["planet"], o["wl"], o["t"], o["e"], o["tmax"], o["z"], o["inst"]))
    json.dump(out, open(core.results("dayside_heat.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanets_dayside_excess", "Daysides hotter than their stars allow", "NASA Exoplanet Archive emissionspec and pscomppars",
            "brightness temperatures with host parameters: %d" % len(out), os.path.abspath(__file__), "dayside", len(out), "measurements",
            dict(sigma=sigma), "the hottest dayside zero albedo and no redistribution give", "a dayside brighter than its star can heat it",
            "%d measurements of %d planets above at 3 sigma" % (len(hot), len({o["planet"] for o in hot})),
            [dict(target="planet:%s" % o["planet"], stat=o["z"], unit="sigma above the hottest possible dayside",
                  note="%.0f +- %.0f K at %s um against %.0f K, %s" % (o["t"], o["e"], o["wl"], o["tmax"], o["inst"])) for o in hot])
        print("recorded")


constants_DOC = r"""Period ratios that spell constants: every adjacent planet pair of the composite table's multi-planet systems, ratios within 1 part in 10,000 of pi, e, the golden ratio, sqrt 2 or sqrt 3, against the hits for 1000 random numbers between 1.2 and 3.5.

    python -I seti.py transits constants [record] [--tol 1e-4] [--draws 1000] [--draw-range 1.2 3.5]"""

CONSTANTS = {"pi": math.pi, "e": math.e, "golden ratio": (1 + math.sqrt(5)) / 2, "sqrt 2": math.sqrt(2), "sqrt 3": math.sqrt(3)}


def constants_main(record=False, tol=1e-4, draws=1000, draw_range=[1.2, 3.5]):
    rows = pcp("hostname, pl_name, pl_orbper", "sy_pnum >= 2 AND pl_orbper IS NOT NULL")
    sysd = {}
    for r in rows:
        sysd.setdefault(r["hostname"], []).append((float(r["pl_orbper"]), r["pl_name"]))
    ratios = []
    for ps in sysd.values():
        ps.sort()
        for (a, na), (b, nb) in zip(ps, ps[1:]):
            ratios.append((b / a, na, nb))
    r = np.array([x[0] for x in ratios])
    hits = [(k, x) for k, c in CONSTANTS.items() for x in ratios if abs(x[0] / c - 1) < tol]
    rng = np.random.default_rng(13)
    null = [int(np.sum(np.abs(r / c - 1) < tol)) for c in rng.uniform(draw_range[0], draw_range[1], draws)]
    exp5 = 5 * float(np.mean(null))
    print("%d adjacent pairs; within 1e-4 of a constant %d against %.2f for five random numbers: %s" % (
        len(ratios), len(hits), exp5, ["%s: %s/%s %.6f" % (k, x[2], x[1], x[0]) for k, x in hits]))
    if record:
        core.record_run(
            "exoplanets_constant_ratios", "Period ratios that spell constants", "NASA Exoplanet Archive pscomppars",
            "adjacent planet pairs: %d" % len(ratios), os.path.abspath(__file__), "constants", len(ratios), "planet pairs",
            dict(tol=tol, constants=list(CONSTANTS), draws=draws, draw_range=draw_range), "%d random numbers between %g and %g: %.3f hits each" % (
                draws, draw_range[0], draw_range[1], float(np.mean(null))),
            "two neighbouring planets whose period ratio equals a mathematical constant to 1e-4",
            "%d hits against %.2f" % (len(hits), exp5),
            [dict(target="planet:%s" % x[2], stat=x[0], unit="period ratio", vet="chance-rate" if len(hits) <= exp5 + 3 * math.sqrt(max(exp5, 1)) else None,
                  note="%s over %s against %s" % (x[2], x[1], k)) for k, x in hits])
        print("recorded")


hillpack_DOC = r"""Systems packed tighter than Hill stability: adjacent planets of the composite table with masses (measured or minimum) and semi-major axes around stars with masses, their separation in mutual Hill radii, ((m1+m2)/3M*)^1/3 (a1+a2)/2, flagged below 2 sqrt 3, Gladman's limit for two planets, each read for masses taken from a mass-radius relation.

    python -I seti.py transits hillpack [record] [--hill-limit 3.464] [--resonance-tol 0.01]"""


def hillpack_main(record=False, hill_limit=3.4641016151377544, resonance_tol=0.01):
    rows = pcp("hostname, pl_name, pl_orbsmax, pl_orbper, pl_bmasse, pl_bmassprov, st_mass, disc_method", "sy_pnum >= 2 AND pl_orbsmax IS NOT NULL "
               "AND pl_bmasse IS NOT NULL AND st_mass IS NOT NULL")
    sysd = {}
    for r in rows:
        sysd.setdefault(r["hostname"], []).append(r)
    out = []
    for h, ps in sysd.items():
        ps.sort(key=lambda r: float(r["pl_orbsmax"]))
        for a, b in zip(ps, ps[1:]):
            a1, a2 = float(a["pl_orbsmax"]), float(b["pl_orbsmax"])
            m = (float(a["pl_bmasse"]) + float(b["pl_bmasse"])) * 3.003e-6
            rh = (m / (3 * float(a["st_mass"]))) ** (1 / 3.0) * (a1 + a2) / 2
            vet = []
            if "Microlensing" in (a["disc_method"] or ""):
                vet.append("unconstrained-geometry")
            if "M-R" in (a["pl_bmassprov"] or "") + (b["pl_bmassprov"] or ""):
                vet.append("estimated-masses")
            if num(a["pl_orbper"]) and num(b["pl_orbper"]):
                ratio = num(b["pl_orbper"]) / num(a["pl_orbper"])
                if any(abs(ratio * q / p - 1) < resonance_tol for p in range(2, 7) for q in range(1, p)):
                    vet.append("near-resonance")
            out.append(dict(host=h, a=a["pl_name"], b=b["pl_name"], delta=(a2 - a1) / rh if rh > 0 else 99,
                            prov="%s/%s" % (a["pl_bmassprov"], b["pl_bmassprov"]), vet=vet))
    tight = sorted([o for o in out if o["delta"] < hill_limit], key=lambda o: o["delta"])
    print("%d adjacent pairs; closer than 2 sqrt 3 mutual Hill radii: %d" % (len(out), len(tight)))
    for o in tight[:15]:
        print("  %s / %s: %.2f mutual Hill radii, masses %s" % (o["a"], o["b"], o["delta"], o["prov"]))
    if record:
        core.record_run(
            "exoplanets_hill_packing", "Systems packed tighter than Hill stability", "NASA Exoplanet Archive pscomppars",
            "adjacent pairs with masses, semi-major axes and stellar masses: %d" % len(out), os.path.abspath(__file__), "hillpack", len(out),
            "planet pairs", dict(limit=hill_limit, resonance_tol=resonance_tol), "Gladman's two-planet Hill-stability limit", "two neighbouring planets closer than "
            "Hill stability allows", "%d pairs below 2 sqrt 3 mutual Hill radii" % len(tight),
            [dict(target="planet:%s" % o["a"], stat=o["delta"], unit="separation in mutual Hill radii", vet="; ".join(o["vet"]) or None,
                  note="with %s, masses %s" % (o["b"], o["prov"])) for o in tight])
        print("recorded")


coorbital_DOC = r"""Two planets on one orbit: planets of one system in the composite table whose periods agree to 1 per cent, co-orbitals no survey has confirmed.

    python -I seti.py transits coorbital [record] [--tol 0.01]"""


def coorbital_main(record=False, tol=0.01):
    rows = pcp("hostname, pl_name, pl_orbper, disc_method", "sy_pnum >= 2 AND pl_orbper IS NOT NULL")
    sysd = {}
    for r in rows:
        sysd.setdefault(r["hostname"], []).append(r)
    hits = []
    for h, ps in sysd.items():
        ps.sort(key=lambda r: float(r["pl_orbper"]))
        for a, b in zip(ps, ps[1:]):
            if float(b["pl_orbper"]) / float(a["pl_orbper"]) - 1 < tol:
                hits.append((a, b))
    print("%d systems of two or more planets; pairs within 1 per cent in period: %d %s" % (
        len(sysd), len(hits), ["%s %.4f d / %s %.4f d (%s)" % (a["pl_name"], float(a["pl_orbper"]), b["pl_name"], float(b["pl_orbper"]), a["disc_method"])
                               for a, b in hits]))
    if record:
        core.record_run(
            "exoplanets_coorbital", "Two planets on one orbit", "NASA Exoplanet Archive pscomppars", "systems of two or more planets: %d" % len(sysd),
            os.path.abspath(__file__), "coorbital", len(sysd), "planetary systems", dict(tol=tol), "co-orbital planets, none confirmed in nature",
            "two planets of one system with periods equal to 1 per cent", "%d pairs" % len(hits),
            [dict(target="planet:%s" % a["pl_name"], stat=float(b["pl_orbper"]) / float(a["pl_orbper"]) - 1, unit="fractional period difference",
                  note="with %s" % b["pl_name"]) for a, b in hits])
        print("recorded")


copysystems_DOC = r"""Copied planetary systems: systems of three or more planets in the composite table whose two consecutive period ratios both agree to 0.1 per cent with another system's, one architecture built twice, against the count with the second ratios shuffled among the systems.

    python -I seti.py transits copysystems [record] [--tol 1e-3] [--shuffles 50]"""


def copysystems_main(record=False, tol=1e-3, shuffles=50):
    rows = pcp("hostname, pl_name, pl_orbper", "sy_pnum >= 3 AND pl_orbper IS NOT NULL")
    sysd = {}
    for r in rows:
        sysd.setdefault(r["hostname"], []).append(float(r["pl_orbper"]))
    trip = []
    for h, ps in sysd.items():
        ps.sort()
        for i in range(len(ps) - 2):
            trip.append((h, ps[i + 1] / ps[i], ps[i + 2] / ps[i + 1]))

    def count(t):
        n = []
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                if t[i][0] != t[j][0] and abs(t[i][1] / t[j][1] - 1) < tol and abs(t[i][2] / t[j][2] - 1) < tol:
                    n.append((t[i], t[j]))
        return n
    found = count(trip)
    rng = np.random.default_rng(31)
    null = []
    for _ in range(shuffles):
        sec = [x[2] for x in trip]
        rng.shuffle(sec)
        null.append(len(count([(a, b, c) for (a, b, _), c in zip(trip, sec)])))
    print("%d consecutive triples; copies across systems %d against %.2f shuffled: %s" % (
        len(trip), len(found), float(np.mean(null)), [(a[0], b[0], round(a[1], 4), round(a[2], 4)) for a, b in found]))
    if record:
        core.record_run("exoplanets_copied_systems", "Copied planetary systems", "NASA Exoplanet Archive pscomppars", "consecutive triples: %d" % len(trip),
                        os.path.abspath(__file__), "copysystems", len(trip), "planet triples", dict(tol=tol, shuffles=shuffles),
                        "second ratios shuffled among systems: %.2f" % float(np.mean(null)), "two systems with the same two period ratios to 0.1 per cent",
                        "%d against %.2f" % (len(found), float(np.mean(null))),
                        [dict(target="system:%s" % a[0], note="copy of %s: ratios %.4f, %.4f" % (b[0], a[1], a[2]),
                              vet="near-resonance" if all(any(abs(x * q / p - 1) < 0.01 for p in range(2, 7) for q in range(1, p)) for x in a[1:]) else None)
                         for a, b in found])
        print("recorded")


mirrorplanets_DOC = r"""Planets brighter than a mirror: eclipse depths shortward of 1 um in the emission-spectroscopy table turned into geometric albedos with the composite table's a/R* and Rp/R*, A_g = depth (a/R*)^2 / (Rp/R*)^2, flagged above 1 at 3 sigma, more light than a perfect Lambert reflector returns.

    python -I seti.py transits mirrorplanets [record] [--albedo-max 1] [--sigma 3] [--wavelength-max-um 1]"""


def mirrorplanets_main(record=False, albedo_max=1.0, sigma=3.0, wavelength_max_um=1.0):
    em = core.exoplanets("SELECT plntname, centralwavelng, especlipdep, especlipdeperr1, especlipdeperr2, facility FROM emissionspec "
                         "WHERE centralwavelng < %g AND especlipdep IS NOT NULL" % wavelength_max_um)
    host = {r["pl_name"]: r for r in pcp("pl_name, pl_ratdor, pl_ratror, pl_eqt", "pl_ratdor IS NOT NULL AND pl_ratror IS NOT NULL")}
    out = []
    for r in em:
        h = host.get(r["plntname"])
        if not h:
            continue
        k = (float(h["pl_ratdor"]) / float(h["pl_ratror"])) ** 2
        # depths are tabulated in per cent
        d, e = float(r["especlipdep"]) / 100, abs(num(r["especlipdeperr1"]) or num(r["especlipdeperr2"]) or 0) / 100
        out.append(dict(planet=r["plntname"], wl=num(r["centralwavelng"]), ag=d * k, e=e * k, teq=num(h["pl_eqt"]), fac=r["facility"]))
    hits = sorted([o for o in out if o["e"] > 0 and o["ag"] - sigma * o["e"] > albedo_max], key=lambda o: -(o["ag"] - albedo_max) / o["e"])
    print("%d optical eclipse depths with geometry; geometric albedo above 1 at 3 sigma: %d %s" % (
        len(out), len(hits), [(o["planet"], o["wl"], round(o["ag"], 2), round(o["e"], 2), o["teq"]) for o in hits[:10]]))
    if record:
        core.record_run("exoplanets_mirror_albedo", "Planets brighter than a mirror", "NASA Exoplanet Archive emissionspec and pscomppars",
                        "eclipse depths shortward of 1 um with geometry: %d" % len(out), os.path.abspath(__file__), "mirrorplanets", len(out),
                        "measurements", dict(albedo=albedo_max, sigma=sigma, wavelength_max_um=wavelength_max_um), "a Lambert reflector's geometric albedo of 2/3, a perfect one's 1",
                        "an optical eclipse deeper than a perfect mirror gives", "%d above 1 at 3 sigma" % len(hits),
                        [dict(target="planet:%s" % o["planet"], stat=o["ag"], unit="geometric albedo", note="+- %.2f at %s um, T_eq %s K, %s" % (
                            o["e"], o["wl"], o["teq"], o["fac"])) for o in hits])
        print("recorded")


insidestar_DOC = r"""Planets orbiting inside their star or hotter than it: composite planets whose orbital period is shorter than a grazing orbit at the stellar surface allows, 2 pi (R*^3 / G M*)^1/2, or whose equilibrium temperature exceeds the star's effective temperature over sqrt 2, the hottest a body can be held by its star.

    python -I seti.py transits insidestar [record]"""


def insidestar_main(record=False):
    rows = pcp("pl_name, pl_orbper, st_rad, st_mass, st_teff, pl_eqt", "pl_orbper IS NOT NULL AND st_rad IS NOT NULL AND st_mass IS NOT NULL")
    inside, hot = [], []
    for r in rows:
        pmin = 2 * math.pi * math.sqrt((float(r["st_rad"]) * 6.957e8) ** 3 / (6.674e-11 * float(r["st_mass"]) * 1.989e30)) / 86400
        if float(r["pl_orbper"]) < pmin:
            inside.append((r["pl_name"], float(r["pl_orbper"]), pmin))
        if num(r["pl_eqt"]) and num(r["st_teff"]) and num(r["pl_eqt"]) > num(r["st_teff"]) / math.sqrt(2):
            hot.append((r["pl_name"], num(r["pl_eqt"]), num(r["st_teff"])))
    print("%d planets; orbiting inside their star %d %s; hotter than the star allows %d %s" % (len(rows), len(inside), inside[:10], len(hot), hot[:10]))
    if record:
        core.record_run("exoplanets_inside_star", "Planets orbiting inside their star or hotter than it", "NASA Exoplanet Archive pscomppars",
                        "planets with period, stellar radius and mass: %d" % len(rows), os.path.abspath(__file__), "insidestar", len(rows), "planets",
                        dict(), "the surface-grazing period and T_eff / sqrt 2", "a period or temperature no orbit around the star allows",
                        "%d inside the star, %d too hot" % (len(inside), len(hot)),
                        [dict(target="planet:%s" % n, stat=p / m, unit="period over the surface-grazing period", vet="published-solution") for n, p, m in inside] +
                        # a young companion glows from its own contraction, not its star's light
                        [dict(target="planet:%s" % n, stat=t / s, unit="T_eq over T_eff", vet="young-star",
                              note="a self-luminous young companion" if "Oph" in n else None) for n, t, s in hot])
        print("recorded")


ironplanets_DOC = r"""Planets denser than iron: transiting composite planets (whose radii are measured, not computed from their masses) with mass measured (not estimated from radius) and radius each to 20 per cent, against the radius of a pure-iron planet of their mass, R = 0.86 M^0.27 in Earth units (Zeng et al. 2016, core mass fraction 1), flagged smaller at 3 sigma with both errors carried, which no natural composition reaches.

    python -I seti.py transits ironplanets [record] [--precision 0.2] [--sigma 3] [--mass-max 30]"""


def ironplanets_main(record=False, precision=0.2, sigma=3.0, mass_max=30.0):
    # a transit measures the radius; the archive computes the others' radii from their masses
    rows = pcp("pl_name, hostname, pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_rade, pl_radeerr1, pl_radeerr2, disc_method",
               "pl_bmasse IS NOT NULL AND pl_rade IS NOT NULL AND pl_bmassprov = 'Mass' AND pl_bmasse < %g AND tran_flag = 1" % mass_max)
    out = []
    for r in rows:
        em, er = rel_err(r, "pl_bmasse"), rel_err(r, "pl_rade")
        if em is None or er is None or em > precision or er > precision:
            continue
        M, Rp = float(r["pl_bmasse"]), float(r["pl_rade"])
        rfe = 0.86 * M ** 0.27
        s = math.hypot(er * Rp, 0.27 * em * rfe)
        out.append(dict(name=r["pl_name"], M=M, R=Rp, rfe=rfe, z=(rfe - Rp) / s if s > 0 else 0.0, method=r["disc_method"]))
    dense = sorted([o for o in out if o["z"] > sigma], key=lambda o: -o["z"])
    print("%d planets below 30 Earth masses with mass and radius to 20 per cent; smaller than pure iron at 3 sigma: %d; the densest %s" % (
        len(out), len(dense), ["%s %.2f Me %.2f Re against %.2f (%.1f sigma)" % (o["name"], o["M"], o["R"], o["rfe"], o["z"]) for o in sorted(out, key=lambda o: -o["z"])[:5]]))
    if record:
        core.record_run("exoplanets_denser_than_iron", "Planets denser than iron", "NASA Exoplanet Archive pscomppars",
                        "planets below 30 Earth masses with measured mass and radius to 20 per cent: %d" % len(out), os.path.abspath(__file__), "ironplanets",
                        len(out), "planets", dict(precision=precision, sigma=sigma, mass_max=mass_max, relation="R = 0.86 M^0.27"), "a pure-iron planet's radius at the planet's mass",
                        "a planet smaller than iron allows", "%d at 3 sigma; the densest %s" % (len(dense), "; ".join(
                            "%s (%.1f sigma)" % (o["name"], o["z"]) for o in sorted(out, key=lambda o: -o["z"])[:3])),
                        [dict(target="planet:%s" % o["name"], stat=o["z"], unit="sigma below a pure-iron radius") for o in dense])
        print("recorded")


exactres_DOC = r"""Planets in exact resonance: adjacent planets of the composite table's multi-planet systems, both periods measured to 1e-5, whose period ratio equals a small-integer ratio p/q (q up to 4, p up to 9) to 1e-4 and within three combined errors, against the pairs within 1e-3 to 1e-2 of the same ratios scaled by the window's width, where migration in a disc leaves resonant pairs about a per cent wide of exact commensurability.

    python -I seti.py transits exactres [record] [--tolerance 1e-4] [--precision 1e-5] [--q-max 4] [--p-max 9] [--near-window 1e-3 1e-2]"""


def exactres_main(record=False, tolerance=1e-4, precision=1e-5, q_max=4, p_max=9, near_window=[1e-3, 1e-2]):
    rows = pcp("hostname, pl_name, pl_orbper, pl_orbpererr1, pl_orbpererr2", "sy_pnum >= 2 AND pl_orbper IS NOT NULL AND pl_orbpererr1 IS NOT NULL")
    sysd = {}
    for r in rows:
        P, e = float(r["pl_orbper"]), max(abs(float(r["pl_orbpererr1"] or 0)), abs(float(r["pl_orbpererr2"] or 0)))
        if 0 < e < precision * P:
            sysd.setdefault(r["hostname"], []).append((P, e, r["pl_name"]))
    ratios = sorted({(p, q) for q in range(1, q_max + 1) for p in range(q + 1, p_max + 1) if math.gcd(p, q) == 1}, key=lambda t: t[0] / t[1])
    exact, near, n = [], 0, 0
    for h, ps in sysd.items():
        ps.sort()
        for (P1, e1, n1), (P2, e2, n2) in zip(ps, ps[1:]):
            n += 1
            x = P2 / P1
            ex = x * math.hypot(e1 / P1, e2 / P2)
            for p, q in ratios:
                d = x / (p / q) - 1
                if abs(d) < tolerance and abs(x - p / q) < 3 * ex:
                    exact.append(dict(a=n1, b=n2, ratio="%d:%d" % (p, q), d=d, sig=abs(x - p / q) / ex if ex > 0 else 0.0))
                elif near_window[0] < abs(d) < near_window[1]:
                    near += 1
    expect = near * (2 * tolerance) / (2 * (near_window[1] - near_window[0]))
    print("%d adjacent pairs with both periods to 1e-5; within 1e-4 of a ratio p/q: %d against %.2f from the %d pairs at 1e-3 to 1e-2 %s" % (
        n, len(exact), expect, near, ["%s/%s %s %+.1e" % (x["a"], x["b"], x["ratio"], x["d"]) for x in exact]))
    if record:
        core.record_run("exoplanets_exact_resonance", "Planets in exact resonance", "NASA Exoplanet Archive pscomppars",
                        "adjacent pairs with both periods measured to 1e-5: %d" % n, os.path.abspath(__file__), "exactres", n, "planet pairs",
                        dict(tolerance=tolerance, precision=precision, q_max=q_max, p_max=p_max, near_window=near_window),
                        "pairs %g to %g from the ratios scaled by width: %.2f" % (near_window[0], near_window[1], expect),
                        "neighbouring planets locked exactly on a small-integer period ratio", "%d against %.2f" % (len(exact), expect),
                        [dict(target="planet:%s" % x["a"], stat=x["d"], unit="fractional offset from %s" % x["ratio"], note="with %s" % x["b"],
                              vet="chance-rate" if len(exact) <= expect + 3 * math.sqrt(max(expect, 1)) else None) for x in exact])
        print("recorded")


earthcopy_DOC = r"""Copies of Earth: composite planets with a measured radius (transiting) within 10 per cent of Earth's and a measured mass (not estimated from radius) within 20 per cent of Earth's, each measured to those precisions, receiving 0.75-1.25 of Earth's insolation from a star of 5300-6100 K; the nearest by mass, radius and insolation listed.

    python -I seti.py transits earthcopy [record] [--radius 0.1] [--mass 0.2] [--insolation 0.75 1.25] [--teff 5300 6100]"""


def earthcopy_main(record=False, radius=0.1, mass=0.2, insolation=[0.75, 1.25], teff=[5300.0, 6100.0]):
    rows = pcp("pl_name, pl_rade, pl_radeerr1, pl_radeerr2, pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_bmassprov, pl_insol, st_teff, tran_flag",
               "pl_rade BETWEEN 0.5 AND 2 AND pl_insol IS NOT NULL AND st_teff IS NOT NULL")
    hits, scored = [], []
    for r in rows:
        Rp, S, T = float(r["pl_rade"]), float(r["pl_insol"]), float(r["st_teff"])
        M = num(r["pl_bmasse"])
        measured = r["pl_bmassprov"] == "Mass" and str(r["tran_flag"]) == "1"
        d = math.sqrt((Rp - 1) ** 2 + ((M - 1) / 2 if M else 1) ** 2 + (math.log(S) if S > 0 else 9) ** 2 + ((T - 5772) / 1000.0) ** 2)
        scored.append((d, r["pl_name"], Rp, M, S, T, measured))
        if (measured and abs(Rp - 1) <= radius and M and abs(M - 1) <= mass and insolation[0] <= S <= insolation[1] and teff[0] <= T <= teff[1]
                and (rel_err(r, "pl_rade") or 1) <= radius and (rel_err(r, "pl_bmasse") or 1) <= mass):
            hits.append(r["pl_name"])
    near = sorted(scored)[:5]
    print("%d composite planets of 0.5-2 Earth radii with insolation and host temperature; copies of Earth: %d %s; the nearest %s" % (
        len(rows), len(hits), hits, ["%s (%.2f Re, %s Me, %.2f S, %.0f K, mass %s)" % (n, R, "%.2f" % M if M else "-", S, T, "measured" if m else "estimated")
                                     for d, n, R, M, S, T, m in near]))
    if record:
        core.record_run("exoplanets_earth_copies", "Copies of Earth", "NASA Exoplanet Archive pscomppars",
                        "planets of 0.5-2 Earth radii with insolation and host temperature: %d" % len(rows), os.path.abspath(__file__), "earthcopy", len(rows),
                        "planets", dict(radius=radius, mass=mass, insolation=insolation, teff=teff), "Earth's radius, mass and insolation and the Sun's temperature",
                        "a planet like Earth in size, mass and sunlight around a star like the Sun", "%d; the nearest %s" % (len(hits), "; ".join(
                            "%s (%.2f Re, %s Me, %.2f S, %.0f K)" % (n, R, "%.2f" % M if M else "-", S, T) for d, n, R, M, S, T, m in near[:3])),
                        [dict(target="planet:%s" % h) for h in hits])
        print("recorded")


randomtransiter_DOC = r"""The random transiter in TESS: HD 139139 (TIC 70652803), whose K2 light curve of 2017 showed 28 transit-like dips of 200-1600 ppm and 0.7-7 h with no period (Rappaport et al. 2019), in its TESS QLP light curve of sector 91 (200-s full-frame cadence, 2025 April-May). The detrended flux at quality 0 divided by a running median of one day; a dip is three or more consecutive cadences below three times the robust scatter with a mean depth beyond 300 ppm; the count against the 8.7 a sector that K2's rate gives, and the sector's threshold for dips of K2's durations from the binned noise.

    python -I seti.py transits randomtransiter [record] [--name "HD 139139"] [--tic 70652803] [--sector 91] [--k2-dips 28] [--k2-days 87] [--sigma 3] [--min-cadences 3] [--min-depth-ppm 300] [--trend-days 1]"""

RANDOM_TRANSITER = dict(name="HD 139139", tic=70652803, sector=91, k2_dips=28, k2_days=87.0)


def qlp_url(tic, sector):
    """The MAST HLSP path of a TESS QLP light curve: the sector and the TIC number zero-padded to sixteen digits in four groups."""
    s = "%016d" % tic
    return "https://archive.stsci.edu/hlsps/qlp/s%04d/%s/%s/%s/%s/hlsp_qlp_tess_ffi_s%04d-%s_tess_v01_llc.fits" % (sector, s[0:4], s[4:8], s[8:12], s[12:16], sector, s)


def randomtransiter_main(record=False, name="HD 139139", tic=70652803, sector=91, k2_dips=28, k2_days=87.0, sigma=3.0, min_cadences=3, min_depth_ppm=300.0,
                         trend_days=1.0):
    from astropy.io import fits
    d = core.data("tess_qlp")
    os.makedirs(d, exist_ok=True)
    url = qlp_url(tic, sector)
    path = os.path.join(d, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        open(path, "wb").write(core.get(url, timeout=300))
    with fits.open(path) as h:
        t = np.asarray(h[1].data["TIME"], float)
        f = np.asarray(h[1].data["DET_FLUX"], float)
        q = np.asarray(h[1].data["QUALITY"], int)
    ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
    t, f = t[ok], f[ok]
    step = float(np.median(np.diff(t)))
    half = int(round(trend_days / 2 / step))
    trend = np.array([np.median(f[max(0, i - half):i + half + 1]) for i in range(len(f))])
    r = f / trend - 1.0
    sig = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    low = r < -sigma * sig
    dips = []
    i = 0
    while i < len(r):
        if low[i]:
            j = i
            while j + 1 < len(r) and low[j + 1] and t[j + 1] - t[j] < 3 * step:
                j += 1
            depth = -float(np.mean(r[i:j + 1])) * 1e6
            if j - i + 1 >= min_cadences and depth > min_depth_ppm:
                dips.append(dict(t0=round(float(t[i]), 3), hours=round(float(t[j] - t[i] + step) * 24, 2), depth_ppm=round(depth, 0), n=j - i + 1))
            i = j + 1
        else:
            i += 1
    span = float(t[-1] - t[0])
    expected = k2_dips / k2_days * span
    # a dip of duration tau averages tau/step cadences: the depth it needs at the threshold
    thresh = {hrs: round(sigma * sig / math.sqrt(max(hrs / 24 / step, 1)) * 1e6) for hrs in (1, 3, 7)}
    print("sector %d: %d cadences over %.1f d, scatter %.0f ppm a cadence; dips %d %s against %.1f at the earlier rate; %g-sigma depth for 1, 3 and 7 h: %s ppm" % (
        sector, len(t), span, sig * 1e6, len(dips), dips, expected, sigma, thresh))
    if record:
        from scipy.stats import poisson
        sid = "hd139139_tess" if tic == RANDOM_TRANSITER["tic"] else "tic%d_s%d_dips" % (tic, sector)
        core.record_run(sid, "The random transiter in TESS", "TESS QLP light curve, sector %d (MAST HLSP)" % sector,
                        "%s, TIC %d: %d good cadences over %.1f d" % (name, tic, len(t), span), os.path.abspath(__file__), "randomtransiter", 1, "stars",
                        dict(sigma=sigma, min_cadences=min_cadences, min_depth_ppm=min_depth_ppm, trend_days=trend_days, k2_dips=k2_dips, k2_days=k2_days),
                        "the earlier rate, %d dips in %.0f d: %.1f expected" % (k2_dips, k2_days, expected), "%g-sigma depth for 1, 3 and 7 h dips: %s ppm" % (sigma, thresh),
                        "%d dips in the sector (chance of this few or fewer at the earlier rate %.3f)" % (len(dips), float(poisson.cdf(len(dips), expected))),
                        [dict(target="tic:%d" % tic, stat=len(dips), unit="dips in the sector", p=float(poisson.cdf(len(dips), expected)),
                              vet=None if dips else "not-measured" if thresh[3] > 800 else None, note="dips %s; thresholds %s" % (dips, thresh))])
        print("recorded")


circles_DOC = r"""Planets on perfect circles: planets of the NASA Exoplanet Archive's composite table with periods beyond 30 days, where tides no longer circularise an orbit, whose eccentricity is measured below 0.002 at its upper error, an orbit rounder than formation and scattering leave, against the eccentricities of the rest; an eccentricity fixed at zero by the fit carries no error or a limit flag and is set aside, and a planet in a compact resonant chain or around a companion-free star read for the solution that measured it.

    python -I seti.py transits circles [record] [--p-min-d 30] [--tol 0.002]"""


def circles_main(record=False, p_min_d=30.0, tol=0.002):
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbper, pl_orbeccen, pl_orbeccenerr1, pl_orbeccenerr2, pl_orbeccenlim, pl_orbsmax, pl_bmasse, "
                           "pl_bmassprov, discoverymethod, st_mass, st_rad, sy_pnum, pl_orbeccen_reflink FROM pscomppars WHERE pl_orbeccen IS NOT NULL AND "
                           "pl_orbper > %g" % p_min_d)
    meas, fixed, out = [], [], []
    for r in rows:
        e, up, lim = num(r["pl_orbeccen"]), num(r["pl_orbeccenerr1"]), num(r["pl_orbeccenlim"])
        if up is None or up <= 0 or (lim is not None and lim != 0):
            fixed.append(r)
            continue
        meas.append((r, e, up))
        if e + up < tol:
            ref = str(r["pl_orbeccen_reflink"] or "")
            ref = ref[ref.find(">") + 1:ref.find("</a>")] if "<a" in ref else ref
            out.append(dict(name=r["pl_name"], host=r["hostname"], P=num(r["pl_orbper"]), e=e, up=up, a=num(r["pl_orbsmax"]), m=num(r["pl_bmasse"]),
                            method=r["discoverymethod"], npl=num(r["sy_pnum"]), ref=ref, vet=None))
    es = sorted(e + up for r, e, up in meas)
    k = max(1, int(0.01 * len(es)))
    print("%d planets with periods beyond %g d and an eccentricity; %d carry a measured eccentricity (the rest fixed or limits); upper eccentricity below %g: %d; "
          "the lowest percentile of upper eccentricities sits at %.4f" % (len(rows), p_min_d, len(meas), tol, len(out), es[k - 1] if es else float("nan")))
    for o in sorted(out, key=lambda o: o["e"] + o["up"]):
        print("  %s: P %.2f d, e %.4f +%.4f, a %s au, %s Earth masses, %s, %s planets; %s" % (o["name"], o["P"], o["e"], o["up"], o["a"], o["m"], o["method"], o["npl"], o["ref"]))
    json.dump(dict(n=len(rows), measured=len(meas), hits=out), open(core.results("perfect_circles.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_perfect_circles", "Planets on perfect circles", "NASA Exoplanet Archive composite parameters (pscomppars)",
            "planets with periods beyond %g d and a measured eccentricity: %d of %d" % (p_min_d, len(meas), len(rows)), os.path.abspath(__file__), "circles",
            len(meas), "planets", dict(p_min_d=p_min_d, tol=tol), "the measured upper eccentricities themselves: the lowest percentile at %.4f" % (es[k - 1] if es else 0),
            "an orbit beyond the tidal reach measured rounder than %g at its upper error" % tol,
            "%d below %g: %s" % (len(out), tol, "; ".join("%s (%s)" % (o["name"], o["ref"]) for o in out) or "none"),
            [dict(target="planet:%s" % o["name"], stat=o["e"] + o["up"], unit="upper eccentricity", vet=o["vet"],
                  note="P %.2f d, e %.4f +%.4f, %s, %s" % (o["P"], o["e"], o["up"], o["method"], o["ref"])) for o in out])
        print("recorded")


wholedays_DOC = r"""Planets on whole days: planets of the NASA Exoplanet Archive's composite table with periods between 1.5 and 400 days known to better than 0.00002 days, those whose period sits within 0.00002 days (1.7 seconds) of a whole number of days, an orbit timed in Earth's solar days, against the same window at half-days and the count uniform fractions give; a transit survey's cadence and a ground-based survey's nightly sampling both alias at whole days, so the discovery method of each match is read.

    python -I seti.py transits wholedays [record] [--tol-d 0.00002] [--p-min-d 1.5] [--p-max-d 400]"""


def wholedays_main(record=False, tol_d=0.00002, p_min_d=1.5, p_max_d=400.0):
    from scipy.stats import poisson
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbper, pl_orbpererr1, pl_orbpererr2, discoverymethod, disc_facility, pl_rade, pl_bmasse, sy_pnum FROM pscomppars "
                           "WHERE pl_orbper BETWEEN %g AND %g AND pl_orbpererr1 IS NOT NULL" % (p_min_d, p_max_d))
    prec = [r for r in rows if 0 < max(abs(num(r["pl_orbpererr1"]) or 0), abs(num(r["pl_orbpererr2"]) or 0)) < tol_d]
    out, half = [], []
    for r in prec:
        P = num(r["pl_orbper"])
        n = round(P)
        o = dict(name=r["pl_name"], host=r["hostname"], P=P, n=int(n), off=P - n, err=max(abs(num(r["pl_orbpererr1"]) or 0), abs(num(r["pl_orbpererr2"]) or 0)),
                 method=r["discoverymethod"], facility=r["disc_facility"], r=num(r["pl_rade"]), m=num(r["pl_bmasse"]), npl=num(r["sy_pnum"]))
        if abs(P - n) < tol_d:
            out.append(o)
        elif abs(P - (math.floor(P) + 0.5)) < tol_d:
            half.append(o)
    exp = len(prec) * 2 * tol_d
    p = float(poisson.sf(len(out) - 1, exp)) if out else 1.0
    for o in out:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d planets with periods of %g-%g d; %d known to better than %g d; within %g d of a whole day: %d (%.2f expected from uniform fractions, Poisson p %.3g); "
          "at half-days: %d" % (len(rows), p_min_d, p_max_d, len(prec), tol_d, tol_d, len(out), exp, p, len(half)))
    for o in sorted(out, key=lambda o: abs(o["off"])):
        print("  %s: P %.6f d (%d, off by %+.6f, error %.6f), %s at %s, %s Earth radii, %s planets; %s" % (
            o["name"], o["P"], o["n"], o["off"], o["err"], o["method"], o["facility"], o["r"], o["npl"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(rows), precise=len(prec), hits=out, half=half), open(core.results("planets_whole_days.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_whole_days", "Planets on whole days", "NASA Exoplanet Archive composite parameters (pscomppars)",
            "planets with periods of %g-%g d known to better than %g d: %d of %d" % (p_min_d, p_max_d, tol_d, len(prec), len(rows)), os.path.abspath(__file__),
            "wholedays", len(prec), "planets", dict(tol_d=tol_d, p_d=[p_min_d, p_max_d]), "uniform fractions give %.2f in the window; the half-day window holds %d" % (
                exp, len(half)), "a period within %g d of a whole number of days at its measured precision" % tol_d,
            "%d at whole days against %.2f expected and %d at half-days (Poisson p %.3g)" % (len(out), exp, len(half), p),
            [dict(target="planet:%s" % o["name"], stat=o["off"], unit="days from a whole day", vet=o["vet"],
                  note="P %.6f d (%d), error %.6f, %s at %s" % (o["P"], o["n"], o["err"], o["method"], o["facility"])) for o in out])
        print("recorded")


spinsteps_DOC = r"""Planets in step with their star's spin: planets of the NASA Exoplanet Archive's composite table whose host carries a rotation period, those whose orbital period is a whole multiple, from 2 to 20, of the star's rotation period within 0.5 per cent, an orbit geared to the star's day as a made clock would be, against the same windows at half-integer multiples and the count uniform fractions give; a planet at 1:1 is the tidal case kept apart, and a rotation period that is itself an alias of the orbit (a star-spot signal at the planet's period) reads as a whole ratio by construction, so each match is read for how its rotation was measured.

    python -I seti.py transits spinsteps [record] [--tol-rel 0.005] [--n-max 20]"""


def spinsteps_main(record=False, tol_rel=0.005, n_max=20):
    from scipy.stats import poisson
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbper, st_rotp, st_rotperr1, st_rotp_reflink, pl_orbsmax, pl_bmasse, pl_rade, discoverymethod, st_teff FROM pscomppars "
                           "WHERE st_rotp IS NOT NULL AND st_rotp > 0 AND pl_orbper IS NOT NULL")
    out, half, n = [], [], 0
    for r in rows:
        P, rot = num(r["pl_orbper"]), num(r["st_rotp"])
        if not P or not rot:
            continue
        q = P / rot
        if q < 1.5 or q > n_max + 0.5:
            continue
        n += 1
        k = round(q)
        ref = str(r["st_rotp_reflink"] or "")
        ref = ref[ref.find(">") + 1:ref.find("</a>")] if "<a" in ref else ref
        o = dict(name=r["pl_name"], host=r["hostname"], P=P, rot=rot, ratio=q, n=int(k), rel=q / k - 1, ref=ref, method=r["discoverymethod"], teff=num(r["st_teff"]),
                 a=num(r["pl_orbsmax"]), vet=None)
        if abs(q / k - 1) < tol_rel:
            out.append(o)
        elif abs(q / (math.floor(q) + 0.5) - 1) < tol_rel:
            half.append(o)
    # a relative window of 2 tol_rel at ratio k spans 2 tol_rel k of the unit interval between whole ratios
    exp = 0.0
    for r in rows:
        P, rot = num(r["pl_orbper"]), num(r["st_rotp"])
        if P and rot and 1.5 <= P / rot <= n_max + 0.5:
            exp += 2 * tol_rel * round(P / rot)
    p = float(poisson.sf(len(out) - 1, exp)) if out else 1.0
    for o in out:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d planets with a host rotation period and a period ratio of 1.5-%g; within %g of a whole ratio: %d (%.2f expected from uniform fractions, Poisson p %.3g); at "
          "half-integers: %d" % (n, n_max + 0.5, tol_rel, len(out), exp, p, len(half)))
    for o in sorted(out, key=lambda o: abs(o["rel"])):
        print("  %s: P %.4f d over rotation %.3f d = %.4f (%d, off by %.2e), Teff %s, %s; rotation from %s; %s" % (
            o["name"], o["P"], o["rot"], o["ratio"], o["n"], o["rel"], o["teff"], o["method"], o["ref"], o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out, half=half), open(core.results("planet_spin_steps.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_spin_steps", "Planets in step with their star's spin", "NASA Exoplanet Archive composite parameters (pscomppars)",
            "planets whose host carries a rotation period, with period ratios of 1.5-%g: %d" % (n_max + 0.5, n), os.path.abspath(__file__), "spinsteps", n, "planets",
            dict(tol_rel=tol_rel, n_max=n_max), "uniform fractions give %.2f in the windows; the half-integer windows hold %d" % (exp, len(half)),
            "an orbital period at a whole multiple of the star's rotation within %g" % tol_rel,
            "%d against %.2f expected and %d at half-integers (Poisson p %.3g)" % (len(out), exp, len(half), p),
            [dict(target="planet:%s" % o["name"], stat=o["rel"], unit="fractional offset from the whole ratio", vet=o["vet"],
                  note="P %.4f d, rotation %.3f d (%s), ratio %.4f (%d)" % (o["P"], o["rot"], o["ref"], o["ratio"], o["n"])) for o in out])
        print("recorded")


longtransits_DOC = r"""Transits longer than the star allows: transiting planets of the NASA Exoplanet Archive's composite table with a transit duration, period, semi-major axis and stellar and planetary radii, whose duration exceeds by a quarter the longest a central transit of a circular orbit can last, (P / pi) arcsin((R_star + R_planet) / a); a chord longer than the star is an occulter larger than the planet, an orbit crossing at apastron or a star larger than catalogued, so each match is read for its eccentricity, its duration error and the sources of its duration and radius, since a radius of the Kepler Input Catalogue era or a duration and a radius from different solutions do not share one geometry, and an eccentricity fixed at zero without errors leaves an apastron crossing open.

    python -I seti.py transits longtransits [record] [--excess 1.25] [--err-max 0.2]"""


def longtransits_main(record=False, excess=1.25, err_max=0.2):
    import re
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbper, pl_orbsmax, pl_orbeccen, pl_orbeccenerr1, st_rad, pl_rade, pl_trandur, pl_trandurerr1, pl_trandurerr2, "
                           "pl_trandur_reflink, st_rad_reflink, discoverymethod FROM pscomppars WHERE pl_trandur IS NOT NULL AND pl_orbsmax IS NOT NULL AND st_rad IS NOT NULL "
                           "AND pl_rade IS NOT NULL AND pl_orbper IS NOT NULL")
    out, n = [], 0
    for r in rows:
        P, a, rs, rp, T = num(r["pl_orbper"]), num(r["pl_orbsmax"]), num(r["st_rad"]), num(r["pl_rade"]), num(r["pl_trandur"])
        if not all(v and v > 0 for v in (P, a, rs, rp, T)):
            continue
        x = (rs * 0.00465047 + rp * 4.26352e-5) / a
        if x >= 1:
            continue
        tmax = P * 24.0 / math.pi * math.asin(x)
        n += 1
        err = max(abs(num(r["pl_trandurerr1"]) or 0), abs(num(r["pl_trandurerr2"]) or 0))
        if T > excess * tmax:
            def strip(s):
                s = str(s or "")
                return s[s.find(">") + 1:s.find("</a>")] if "<a" in s else s
            e, eup = num(r["pl_orbeccen"]), num(r["pl_orbeccenerr1"])
            dur_ref, rad_ref = strip(r["pl_trandur_reflink"]).strip(), strip(r["st_rad_reflink"]).strip()
            # a stellar radius from the Kepler Input Catalogue era, or a duration and a radius drawn from different solutions, do not share one geometry
            old = bool(re.search(r"Morton et al\. 2016|KOI Table|Rowe et al\. 2014|Xie 2014|Steffen et al\. 2013|Hadden et al\. 2014", rad_ref))
            out.append(dict(name=r["pl_name"], P=P, a=a, rs=rs, rp=rp, T=T, err=err, tmax=tmax, ratio=T / tmax, e=e, eup=eup, dur_ref=dur_ref, rad_ref=rad_ref,
                            method=r["discoverymethod"],
                            vet=("unconstrained-geometry" if err > err_max * T or err == 0 else
                                 "single-record" if T / tmax > 2.5 and dur_ref == rad_ref and not old else
                                 "published-solution" if old or dur_ref != rad_ref else
                                 "loose-eccentricities" if e is not None and e + (eup or 0) > 0.1 else
                                 "unmeasured-eccentricities" if not eup else None)))
    left = [o for o in out if o["vet"] is None]
    count = {v: sum(o["vet"] == v for o in out) for v in ("unconstrained-geometry", "single-record", "published-solution", "loose-eccentricities", "unmeasured-eccentricities")}
    print("%d transiting planets with durations and geometry; longer than %g times the central circular maximum: %d; duration errors above %g of the duration or none "
          "listed %d, a duration off by itself %d, a radius of the Kepler Input Catalogue era or a duration and radius from different solutions %d, eccentric orbits %d, "
          "eccentricities fixed without errors %d; unexplained %d" % (n, excess, len(out), err_max, count["unconstrained-geometry"], count["single-record"],
                                                                     count["published-solution"], count["loose-eccentricities"], count["unmeasured-eccentricities"], len(left)))
    for o in sorted(out, key=lambda o: -o["ratio"]):
        print("  %s: duration %.2f +- %.2f h against %.2f h at most (%.2fx), P %.3f d, a %.4f au, R* %.2f, Rp %.2f Earth, e %s; duration from %s, radius from %s; %s" % (
            o["name"], o["T"], o["err"], o["tmax"], o["ratio"], o["P"], o["a"], o["rs"], o["rp"], o["e"], o["dur_ref"], o["rad_ref"], o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out), open(core.results("transits_too_long.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_long_transits", "Transits longer than the star allows", "NASA Exoplanet Archive composite parameters (pscomppars)",
            "transiting planets with a duration, period, semi-major axis and radii: %d" % n, os.path.abspath(__file__), "longtransits", n, "planets",
            dict(excess=excess, err_max=err_max), "each planet's own central circular maximum from its catalogued geometry",
            "a transit %g times longer than a central circular crossing of the star" % excess,
            "%d: %d with loose or missing duration errors, %d a duration off by itself, %d with a Kepler Input Catalogue radius or mixed solutions, %d on eccentric orbits, "
            "%d with eccentricities fixed without errors; %d unexplained" % (len(out), count["unconstrained-geometry"], count["single-record"], count["published-solution"],
                                                                          count["loose-eccentricities"], count["unmeasured-eccentricities"], len(left)),
            [dict(target="planet:%s" % o["name"], stat=o["ratio"], unit="duration over the central circular maximum", vet=o["vet"],
                  note="%.2f +- %.2f h against %.2f h; P %.3f d, a %.4f au, R* %.2f, e %s; %s; %s" % (o["T"], o["err"], o["tmax"], o["P"], o["a"], o["rs"], o["e"], o["dur_ref"],
                                                                                                    o["rad_ref"])) for o in out])
        print("recorded")


faceon_DOC = r"""Systems turned face-on to us: planets of the NASA Exoplanet Archive's composite table with an orbital inclination measured below 5 deg, an orbit we look straight down upon, which no transit and no radial velocity could find and which an astrometric or imaged orbit alone reveals; orbits point at random, so the share within 5 deg of face-on is 0.4 per cent of all orientations, and a surplus would mean systems arranged to face us. Each read for its inclination error and the method that measured it.

    python -I seti.py transits faceon [record] [--i-max-deg 5] [--err-max-deg 5]"""


def faceon_main(record=False, i_max_deg=5.0, err_max_deg=5.0):
    from scipy.stats import binom
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbincl, pl_orbinclerr1, pl_orbinclerr2, pl_orbincllim, pl_orbper, pl_bmassj, discoverymethod, pl_orbincl_reflink "
                           "FROM pscomppars WHERE pl_orbincl IS NOT NULL AND pl_orbinclerr1 IS NOT NULL AND discoverymethod NOT IN ('Transit', 'Transit Timing Variations')")
    meas = []
    for r in rows:
        i, up, lo = num(r["pl_orbincl"]), abs(num(r["pl_orbinclerr1"]) or 0), abs(num(r["pl_orbinclerr2"]) or 0)
        if i is None or up == 0 or (num(r["pl_orbincllim"]) or 0) != 0:
            continue
        # an inclination counts as face-on from either pole
        meas.append(dict(name=r["pl_name"], i=i, up=up, lo=lo, pole=min(i, abs(180 - i)), P=num(r["pl_orbper"]), m=num(r["pl_bmassj"]), method=r["discoverymethod"],
                         ref=(lambda s: s[s.find(">") + 1:s.find("</a>")] if "<a" in s else s)(str(r["pl_orbincl_reflink"] or ""))))
    hits = [m for m in meas if m["pole"] < i_max_deg]
    frac = 1 - math.cos(math.radians(i_max_deg))
    exp = len(meas) * frac
    p = float(binom.sf(len(hits) - 1, len(meas), frac)) if hits else 1.0
    for m in hits:
        m["vet"] = "unconstrained-geometry" if max(m["up"], m["lo"]) > err_max_deg else "chance-rate" if p > 0.01 else None
    print("%d non-transiting planets with a measured inclination; within %g deg of face-on: %d (%.2f expected from random orientations, binomial p %.3g)" % (
        len(meas), i_max_deg, len(hits), exp, p))
    for m in sorted(hits, key=lambda m: m["pole"]):
        print("  %s: i %.2f +%.2f -%.2f deg, P %s d, %s Jupiter masses, %s, %s; %s" % (m["name"], m["i"], m["up"], m["lo"], m["P"], m["m"], m["method"], m["ref"], m["vet"]))
    json.dump(dict(n=len(meas), hits=hits), open(core.results("face_on_systems.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_face_on", "Systems turned face-on to us", "NASA Exoplanet Archive composite parameters (pscomppars)",
            "non-transiting planets with a measured inclination: %d" % len(meas), os.path.abspath(__file__), "faceon", len(meas), "planets",
            dict(i_max_deg=i_max_deg, err_max_deg=err_max_deg), "random orientations give %.2f within %g deg of either pole" % (exp, i_max_deg),
            "an orbit within %g deg of face-on, measured to better than %g deg" % (i_max_deg, err_max_deg),
            "%d against %.2f expected (binomial p %.3g)" % (len(hits), exp, p),
            [dict(target="planet:%s" % m["name"], stat=m["pole"], unit="deg from face-on", vet=m["vet"],
                  note="i %.2f +%.2f -%.2f deg, P %s d, %s Jupiter masses, %s, %s" % (m["i"], m["up"], m["lo"], m["P"], m["m"], m["method"], m["ref"])) for m in hits])
        print("recorded")


wdrefuge_DOC = r"""Refuges at white dwarfs: planets of the NASA Exoplanet Archive's composite table whose host is a white dwarf (a spectral type beginning with D, or a radius below 0.05 solar at a temperature above 6000 K), each placed against the radius the host's giant progenitor reached, taken as 1 au for a star that left a white dwarf of 0.6 solar masses, since a planet now inside that radius would have been engulfed unless it was moved there afterwards or arrived from elsewhere; a planet parked close to a dead star is where a civilisation outliving its sun would settle. Each read for its discovery method, its period and the published account of its arrival, which gives each close companion a natural route: a brown dwarf drawn in through a common envelope, or a planet brought in by its outer stars.

    python -I seti.py transits wdrefuge [record] [--engulf-au 1.0]"""


def wdrefuge_main(record=False, engulf_au=1.0):
    rows = core.exoplanets("SELECT pl_name, hostname, st_spectype, st_teff, st_rad, st_mass, pl_orbper, pl_orbsmax, pl_bmassj, pl_rade, discoverymethod, disc_year, disc_refname "
                           "FROM pscomppars WHERE st_spectype LIKE 'D%' OR (st_rad < 0.05 AND st_teff > 6000)")
    out = []
    for r in rows:
        a = num(r["pl_orbsmax"])
        P = num(r["pl_orbper"])
        if a is None and P and num(r["st_mass"]):
            a = (num(r["st_mass"]) * (P / 365.25) ** 2) ** (1 / 3)
        ref = str(r["disc_refname"] or "")
        ref = ref[ref.find(">") + 1:ref.find("</a>")] if "<a" in ref else ref
        out.append(dict(name=r["pl_name"], host=r["hostname"], sp=(r["st_spectype"] or "").strip(), teff=num(r["st_teff"]), rs=num(r["st_rad"]), ms=num(r["st_mass"]), P=P, a=a,
                        mj=num(r["pl_bmassj"]), re=num(r["pl_rade"]), method=r["discoverymethod"], year=r["disc_year"], ref=ref,
                        vet="natural-source" if a is not None and a < engulf_au else "outside-selection"))
    # the discovery papers give each close companion a natural route: a brown dwarf drawn in through a common envelope, or a planet brought in by its outer stars
    inside = [o for o in out if o["vet"] == "natural-source"]
    print("%d planets at white dwarfs; inside the giant's reach of %g au: %d, beyond it or without an orbit: %d" % (len(out), engulf_au, len(inside), len(out) - len(inside)))
    for o in sorted(out, key=lambda o: (o["a"] is None, o["a"] or 0)):
        print("  %s (%s, %s, Teff %s, R %s): P %s d, a %s au, %s Jupiter masses, %s Earth radii, %s %s; %s; %s" % (
            o["name"], o["host"], o["sp"] or "-", o["teff"], o["rs"], o["P"], o["a"], o["mj"], o["re"], o["method"], o["year"], o["ref"], o["vet"] or "inside the giant's reach"))
    json.dump(out, open(core.results("white_dwarf_planets.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_white_dwarf_refuges", "Refuges at white dwarfs", "NASA Exoplanet Archive composite parameters (pscomppars)",
            "planets at white dwarfs: %d" % len(out), os.path.abspath(__file__), "wdrefuge", len(out), "planets", dict(engulf_au=engulf_au),
            "the planets beyond the giant's reach, which survived in place: %d" % (len(out) - len(inside)),
            "a planet inside the radius its host's giant phase reached", "%d of %d inside %g au: %s" % (
                len(inside), len(out), engulf_au, "; ".join("%s at %s au (%s)" % (o["name"], o["a"], o["method"]) for o in inside) or "none"),
            [dict(target="planet:%s" % o["name"], stat=o["a"], unit="semi-major axis, au", vet=o["vet"],
                  note="%s %s, Teff %s, P %s d, %s Jupiter masses, %s %s, %s" % (o["host"], o["sp"], o["teff"], o["P"], o["mj"], o["method"], o["year"], o["ref"])) for o in out])
        print("recorded")


deeptransits_DOC = r"""Transits deeper than the planet: transiting planets of the NASA Exoplanet Archive's composite table with a transit depth and a radius ratio, those whose depth exceeds the square of the radius ratio by half or more, a dip taking more light than the planet's disc can block, which limb darkening lifts by a fifth at most and a larger occulter would; each read for the sources of its depth and ratio, since the composite table draws them from different solutions, and for the depth's error.

    python -I seti.py transits deeptransits [record] [--excess 1.5] [--err-max 0.2]"""


def deeptransits_main(record=False, excess=1.5, err_max=0.2):
    rows = core.exoplanets("SELECT pl_name, pl_trandep, pl_trandeperr1, pl_trandeperr2, pl_ratror, pl_ratrorerr1, pl_trandep_reflink, pl_ratror_reflink, discoverymethod "
                           "FROM pscomppars WHERE pl_trandep IS NOT NULL AND pl_ratror IS NOT NULL AND pl_trandep > 0 AND pl_ratror > 0")

    def strip(s):
        s = str(s or "")
        return (s[s.find(">") + 1:s.find("</a>")] if "<a" in s else s).strip()
    out, n = [], 0
    for r in rows:
        d, k = num(r["pl_trandep"]), num(r["pl_ratror"])
        err = max(abs(num(r["pl_trandeperr1"]) or 0), abs(num(r["pl_trandeperr2"]) or 0))
        n += 1
        ratio = (d / 100.0) / (k * k)
        if ratio > excess:
            dref, kref = strip(r["pl_trandep_reflink"]), strip(r["pl_ratror_reflink"])
            out.append(dict(name=r["pl_name"], depth=d, err=err, k=k, ratio=ratio, dref=dref, kref=kref, method=r["discoverymethod"],
                            vet="unconstrained-geometry" if err == 0 or err > err_max * d else "published-solution" if dref != kref else None))
    left = [o for o in out if o["vet"] is None]
    print("%d transiting planets with a depth and a radius ratio; depth above %g times the ratio squared: %d, %d with loose or missing depth errors, %d with depth and ratio from "
          "different solutions; unexplained %d" % (n, excess, len(out), sum(o["vet"] == "unconstrained-geometry" for o in out), sum(o["vet"] == "published-solution" for o in out),
                                                     len(left)))
    for o in sorted(out, key=lambda o: -o["ratio"]):
        print("  %s: depth %.4f +- %.4f per cent against %.4f from the ratio %.4f (%.2fx); depth from %s, ratio from %s; %s" % (
            o["name"], o["depth"], o["err"], 100 * o["k"] ** 2, o["k"], o["ratio"], o["dref"], o["kref"], o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out), open(core.results("transits_too_deep.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_deep_transits", "Transits deeper than the planet", "NASA Exoplanet Archive composite parameters (pscomppars)",
            "transiting planets with a depth and a radius ratio: %d" % n, os.path.abspath(__file__), "deeptransits", n, "planets", dict(excess=excess, err_max=err_max),
            "each planet's own radius ratio squared, with limb darkening allowing a fifth more", "a dip %g times deeper than the planet's disc can make" % excess,
            "%d: %d with loose or missing depth errors, %d with depth and ratio from different solutions; %d unexplained" % (
                len(out), sum(o["vet"] == "unconstrained-geometry" for o in out), sum(o["vet"] == "published-solution" for o in out), len(left)),
            [dict(target="planet:%s" % o["name"], stat=o["ratio"], unit="depth over the ratio squared", vet=o["vet"],
                  note="depth %.4f +- %.4f per cent, ratio %.4f; %s; %s" % (o["depth"], o["err"], o["k"], o["dref"], o["kref"])) for o in out])
        print("recorded")


ttvabsent_DOC = r"""Resonant pairs without timing variations: pairs of Kepler planets in one system whose period ratio lies within 2 per cent of 2:1 or 3:2, where the neighbour's pull shifts each transit by minutes over the resonant cycle, read in the Holczer et al. (2016) transit-timing catalogue (VizieR J/ApJS/225/9) for the timing scatter and the timing-variation significance of each; a pair whose transits keep time to better than a fifth of the variation its masses demand is a system held on station, which the masses estimated from the radii (Chen and Kipping 2017) set, so each quiet pair is read for how low those masses could be.

    python -I seti.py transits ttvabsent [record] [--delta-max 0.02] [--quiet-ratio 0.2] [--pval-min 0.5]"""


def ttvabsent_main(record=False, delta_max=0.02, quiet_ratio=0.2, pval_min=0.5):
    tt = {str(r["KOI"]).strip(): r for r in core.tap("vizier", 'SELECT "KOI", "sigTT", "pval" FROM "J/ApJS/225/9/table4"')}
    per = {str(r["KOI"]).strip(): r for r in core.tap("vizier", 'SELECT "KOI", "KIC", "Per", "Dur", "Depth" FROM "J/ApJS/225/9/table2"')}
    arch = core.exoplanets("SELECT kepoi_name, kepid, koi_period, koi_prad, koi_smass, koi_srad FROM q1_q17_dr25_koi WHERE koi_disposition = 'CONFIRMED' OR koi_pdisposition = 'CANDIDATE'")
    info = {str(r["kepoi_name"]).replace("K", "").lstrip("0").strip(): r for r in arch}
    systems = {}
    for koi, r in per.items():
        if num(r["Per"]):
            systems.setdefault(str(r["KIC"]).strip(), []).append((koi, num(r["Per"])))

    def mass_earth(rp):
        # Chen and Kipping (2017): terran below 2 Earth radii, neptunian above
        return rp ** 3.58 / 1.0 if rp < 2.04 else 2.04 ** 3.58 * (rp / 2.04) ** (1 / 0.59)
    out, pairs = [], 0
    for kic, planets in systems.items():
        planets.sort(key=lambda x: x[1])
        for i in range(len(planets)):
            for j in range(i + 1, len(planets)):
                (k1, p1), (k2, p2) = planets[i], planets[j]
                for jres in (2, 3):
                    ratio = jres / (jres - 1)
                    delta = p2 / p1 / ratio - 1
                    if abs(delta) > delta_max:
                        continue
                    pairs += 1
                    a1, a2 = info.get(k1, {}), info.get(k2, {})
                    ms = num(a1.get("koi_smass")) or 1.0
                    r1, r2 = num(a1.get("koi_prad")), num(a2.get("koi_prad"))
                    if not r1 or not r2:
                        continue
                    m1, m2 = mass_earth(r1) * 3.003e-6 / ms, mass_earth(r2) * 3.003e-6 / ms
                    # Lithwick, Xie and Wu (2012): the amplitude of each planet's timing variation from its neighbour, in days
                    amp1 = p1 * m2 / (math.pi * jres ** (2 / 3) * (jres - 1) ** (1 / 3) * max(abs(delta), 1e-4)) * 1.5
                    amp2 = p2 * m1 / (math.pi * jres * max(abs(delta), 1e-4)) * 1.5
                    for koi, amp, other in ((k1, amp1, k2), (k2, amp2, k1)):
                        t = tt.get(koi)
                        if not t:
                            continue
                        sig, pval = num(t["sigTT"]), num(t["pval"])
                        if sig is None or pval is None:
                            continue
                        quiet = pval >= pval_min and sig * 1440.0 < quiet_ratio * amp * 1440.0
                        out.append(dict(koi=koi, other=other, kic=kic, res="%d:%d" % (jres, jres - 1), delta=delta, amp_min=amp * 1440.0, sig_min=sig, pval=pval, r=r1 if koi == k1 else r2,
                                        quiet=quiet, vet="estimated-masses" if quiet else None))
    quiet = [o for o in out if o["quiet"]]
    print("%d near-resonant pairs within %g of 2:1 or 3:2 among %d Kepler systems; planets whose timing scatter stays below %g of the variation their estimated masses demand with no "
          "detected variation: %d of %d" % (pairs, delta_max, len(systems), quiet_ratio, len(quiet), len(out)))
    for o in sorted(quiet, key=lambda o: o["sig_min"] / o["amp_min"]):
        print("  KOI %s with %s (%s, delta %+.4f): expected %.1f min, scatter %.2f min, variation p %.2f, radius %.2f Earth; %s" % (
            o["koi"], o["other"], o["res"], o["delta"], o["amp_min"], o["sig_min"], o["pval"], o["r"], o["vet"]))
    json.dump(dict(pairs=pairs, planets=out), open(core.results("resonant_pairs_quiet.json"), "w"), indent=1)
    if record:
        core.record_run(
            "kepler_resonant_quiet", "Resonant pairs without timing variations", "Holczer et al. (2016) Kepler transit timings (VizieR J/ApJS/225/9); Kepler DR25 KOI table",
            "planets in pairs within %g of 2:1 or 3:2 with timing measurements: %d" % (delta_max, len(out)), os.path.abspath(__file__), "ttvabsent", len(out), "planets",
            dict(delta_max=delta_max, quiet_ratio=quiet_ratio, pval_min=pval_min), "the variation the masses estimated from the radii demand, against the timing scatter",
            "a transit keeping time to a fifth of the variation its neighbour's estimated mass demands",
            "%d of %d planets keep time below a fifth of the expected variation; their masses rest on radii alone" % (len(quiet), len(out)),
            [dict(target="koi:%s" % o["koi"], stat=o["sig_min"] / o["amp_min"], unit="timing scatter over the expected variation", vet=o["vet"],
                  note="with KOI %s at %s (delta %+.4f), expected %.1f min, scatter %.2f min, variation p %.2f" % (o["other"], o["res"], o["delta"], o["amp_min"], o["sig_min"], o["pval"]))
             for o in quiet])
        print("recorded")


spinlaps_DOC = r"""Planets lapping their star's spin in whole steps: planets of the NASA Exoplanet Archive's composite table whose host carries a rotation period, those whose orbital period divides the star's rotation a whole number of times, from 2 to 20, within 0.5 per cent, a planet geared to complete so many orbits per stellar day as a made clock would, against the same windows at half-integer ratios and the count uniform fractions give; a rotation period that is itself an alias of the orbit reads as a whole ratio by construction, so each match is read for how its rotation was measured.

    python -I seti.py transits spinlaps [record] [--tol-rel 0.005] [--n-max 20]"""


def spinlaps_main(record=False, tol_rel=0.005, n_max=20):
    from scipy.stats import poisson
    rows = core.exoplanets("SELECT pl_name, hostname, pl_orbper, st_rotp, st_rotp_reflink, pl_bmasse, pl_rade, discoverymethod, st_teff FROM pscomppars "
                           "WHERE st_rotp IS NOT NULL AND st_rotp > 0 AND pl_orbper IS NOT NULL")
    out, half, n, exp = [], [], 0, 0.0
    for r in rows:
        P, rot = num(r["pl_orbper"]), num(r["st_rotp"])
        if not P or not rot:
            continue
        q = rot / P
        if q < 1.5 or q > n_max + 0.5:
            continue
        n += 1
        k = round(q)
        exp += 2 * tol_rel * k
        ref = str(r["st_rotp_reflink"] or "")
        ref = ref[ref.find(">") + 1:ref.find("</a>")] if "<a" in ref else ref
        o = dict(name=r["pl_name"], host=r["hostname"], P=P, rot=rot, ratio=q, n=int(k), rel=q / k - 1, ref=ref, method=r["discoverymethod"], teff=num(r["st_teff"]), vet=None)
        if abs(q / k - 1) < tol_rel:
            out.append(o)
        elif abs(q / (math.floor(q) + 0.5) - 1) < tol_rel:
            half.append(o)
    p = float(poisson.sf(len(out) - 1, exp)) if out else 1.0
    for o in out:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d planets with a host rotation period and a rotation over period of 1.5-%g; within %g of a whole ratio: %d (%.2f expected from uniform fractions, Poisson p %.3g); at "
          "half-integers: %d" % (n, n_max + 0.5, tol_rel, len(out), exp, p, len(half)))
    for o in sorted(out, key=lambda o: abs(o["rel"])):
        print("  %s: rotation %.3f d over P %.4f d = %.4f (%d, off by %.2e), Teff %s, %s; rotation from %s; %s" % (
            o["name"], o["rot"], o["P"], o["ratio"], o["n"], o["rel"], o["teff"], o["method"], o["ref"], o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out, half=half), open(core.results("planet_spin_laps.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_spin_laps", "Planets lapping their star's spin in whole steps", "NASA Exoplanet Archive composite parameters (pscomppars)",
            "planets whose host carries a rotation period, with rotation over period of 1.5-%g: %d" % (n_max + 0.5, n), os.path.abspath(__file__), "spinlaps", n, "planets",
            dict(tol_rel=tol_rel, n_max=n_max), "uniform fractions give %.2f in the windows; the half-integer windows hold %d" % (exp, len(half)),
            "an orbital period dividing the star's rotation a whole number of times within %g" % tol_rel,
            "%d against %.2f expected and %d at half-integers (Poisson p %.3g)" % (len(out), exp, len(half), p),
            [dict(target="planet:%s" % o["name"], stat=o["rel"], unit="fractional offset from the whole ratio", vet=o["vet"],
                  note="rotation %.3f d (%s), P %.4f d, ratio %.4f (%d)" % (o["rot"], o["ref"], o["P"], o["ratio"], o["n"])) for o in out])
        print("recorded")


translines_DOC = r"""Lines in transmission spectra: the NASA Exoplanet Archive's transmission-spectroscopy table (transitspec), each published spectrum taken in wavelength order, bins whose transit depth stands more than four combined errors, four times the spectrum's own bin-to-bin scatter and 2 per cent of the depth away from the mean of their two neighbours, in either sign; a bin deeper is an absorption line of the atmosphere, read against the sodium, potassium, hydrogen, helium, calcium, magnesium and lithium lines, and a bin shallower is light the planet adds at one wavelength while in transit, which an active star's chromospheric lines in the transit chord give and a lamp would; bins wider than 50 nm are broad features rather than lines.

    python -I seti.py transits translines [record] [--sigma 4] [--frac 0.02] [--wide-um 0.05] [--min-bins 5]"""

TRANSLINES_LINES = {"Ca II K": 0.39337, "Ca II H": 0.39685, "Ca I": 0.42268, "H beta": 0.48613, "Mg I b": 0.51733, "Na D": 0.58930, "H alpha": 0.65628, "Li I": 0.67078,
                    "K I": 0.76649, "K I red": 0.76990, "He I": 1.08330}


def translines_main(record=False, sigma=4.0, frac=0.02, wide_um=0.05, min_bins=5):
    rows = core.exoplanets("SELECT plntname, plntranreflink, facility, instrument, centralwavelng, bandwidth, plntransdep, plntransdeperr1, plntransdeperr2, plntransdeplim FROM transitspec "
                           "WHERE plntransdep IS NOT NULL AND centralwavelng IS NOT NULL")
    series = {}
    for r in rows:
        if num(r["plntransdeplim"]):
            continue
        e1, e2 = num(r["plntransdeperr1"]), num(r["plntransdeperr2"])
        err = (abs(e1) + abs(e2)) / 2 if e1 is not None and e2 is not None else abs(e1 if e1 is not None else e2) if (e1 is not None or e2 is not None) else None
        if not err:
            continue
        ref = str(r["plntranreflink"] or "")
        ref = ref[ref.find(">") + 1:ref.find("</a>")] if "<a" in ref else ref
        series.setdefault((str(r["plntname"]).strip(), ref, str(r["instrument"] or "").strip()), []).append(
            (float(r["centralwavelng"]), num(r["bandwidth"]) or 0.0, float(r["plntransdep"]), err))
    out, n_series, n_bins = [], 0, 0
    for (planet, ref, inst), pts in series.items():
        pts = sorted(set(pts))
        if len(pts) < min_bins:
            continue
        n_series += 1
        n_bins += len(pts)
        res = [pts[i][2] - 0.5 * (pts[i - 1][2] + pts[i + 1][2]) for i in range(1, len(pts) - 1)]
        scale = 1.4826 * float(np.median(np.abs(res))) if res else 0.0
        for i in range(1, len(pts) - 1):
            w, bw, d, e = pts[i]
            dev = res[i - 1]
            err = math.sqrt(e ** 2 + 0.25 * (pts[i - 1][3] ** 2 + pts[i + 1][3] ** 2))
            if abs(dev) > sigma * err and abs(dev) > sigma * scale and abs(dev) > frac * abs(d):
                half = max(bw / 2, 0.002)
                lines = [k for k, lam in TRANSLINES_LINES.items() if abs(lam - w) <= half]
                vet = ("broad-feature" if bw > wide_um else "atmospheric-line" if dev > 0 and lines else
                       "active-star" if dev < 0 and any(k.startswith(("H ", "Ca II")) for k in lines) else None)
                out.append(dict(planet=planet, ref=ref, inst=inst, um=w, bw=bw, depth=d, dev=dev, err=err, z=dev / err, scale=scale, lines=lines, vet=vet, n=len(pts)))
    left = [o for o in out if o["vet"] is None]
    print("%d spectra of %d bins; bins %g sigma and a %g fraction from their neighbours: %d (%d absorption at atmospheric lines, %d shallower at chromospheric lines, %d broad); "
          "unexplained %d" % (n_series, n_bins, sigma, frac, len(out), sum(o["vet"] == "atmospheric-line" for o in out), sum(o["vet"] == "active-star" for o in out),
                             sum(o["vet"] == "broad-feature" for o in out), len(left)))
    for o in sorted(out, key=lambda o: -abs(o["z"])):
        print("  %s (%s, %s): %.4f um (%.4f wide) depth %.4f, %+.4f from its neighbours (%.1f sigma, scatter %.4f), lines %s; %s" % (
            o["planet"], o["ref"], o["inst"], o["um"], o["bw"], o["depth"], o["dev"], o["z"], o["scale"], o["lines"] or "-", o["vet"] or "unexplained"))
    json.dump(dict(series=n_series, bins=n_bins, hits=out), open(core.results("transmission_lines.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_transmission_lines", "Lines in transmission spectra", "NASA Exoplanet Archive transmission spectroscopy table (transitspec)",
            "published spectra with %d or more bins: %d, %d bins" % (min_bins, n_series, n_bins), os.path.abspath(__file__), "translines", n_series, "spectra",
            dict(sigma=sigma, frac=frac, wide_um=wide_um, min_bins=min_bins), "each spectrum's own bin-to-bin scatter and the combined errors of the three bins",
            "a bin %g sigma and a %g fraction of the depth away from the mean of its neighbours" % (sigma, frac),
            "%d: %d absorption at atmospheric lines, %d shallower at chromospheric lines, %d broad; %d unexplained" % (
                len(out), sum(o["vet"] == "atmospheric-line" for o in out), sum(o["vet"] == "active-star" for o in out), sum(o["vet"] == "broad-feature" for o in out), len(left)),
            [dict(target="planet:%s" % o["planet"], stat=round(o["z"], 1), unit="sigma from the neighbouring bins", vet=o["vet"],
                  note="%s, %s, %.4f um, depth %.4f, %+.4f, lines %s" % (o["ref"], o["inst"], o["um"], o["depth"], o["dev"], o["lines"] or "none")) for o in out])
        print("recorded")


polar_DOC = r"""Planets on polar orbits: the NASA Exoplanet Archive's planetary systems table (ps), every planet with a projected obliquity, the measurement of smallest error taken for each, those whose sky-projected obliquity lies within 15 deg of 90 at an error under 15 deg, an orbit set square to the star's spin, against the share a uniform projected obliquity gives and the share within 15 deg of alignment; the measured obliquities pile up near 90 deg (Albrecht et al. 2021), a natural population feature with tidal and secular mechanisms proposed, against which each polar planet is read.

    python -I seti.py transits polar [record] [--half-width-deg 15] [--err-max-deg 15]"""


def polar_main(record=False, half_width_deg=15.0, err_max_deg=15.0):
    from scipy.stats import binom
    rows = core.exoplanets("SELECT pl_name, hostname, pl_projobliq, pl_projobliqerr1, pl_projobliqerr2, pl_trueobliq, pl_orbper, st_teff, pl_bmassj, pl_refname FROM ps "
                           "WHERE pl_projobliq IS NOT NULL")
    best = {}
    for r in rows:
        lam = num(r["pl_projobliq"])
        e1, e2 = num(r["pl_projobliqerr1"]), num(r["pl_projobliqerr2"])
        err = max(abs(e1 or 0), abs(e2 or 0)) if (e1 is not None or e2 is not None) else None
        if lam is None or err is None:
            continue
        name = str(r["pl_name"]).strip()
        ref = str(r["pl_refname"] or "")
        ref = ref[ref.find(">") + 1:ref.find("</a>")] if "<a" in ref else ref
        o = dict(name=name, host=str(r["hostname"]).strip(), lam=lam, err=err, fold=abs(((lam + 180) % 360) - 180), true=num(r["pl_trueobliq"]), P=num(r["pl_orbper"]), teff=num(r["st_teff"]),
                 mass=num(r["pl_bmassj"]), ref=ref)
        if name not in best or err < best[name]["err"]:
            best[name] = o
    pl = [o for o in best.values() if o["err"] < err_max_deg]
    polar = [o for o in pl if abs(o["fold"] - 90) < half_width_deg]
    aligned = [o for o in pl if o["fold"] < half_width_deg]
    retro = [o for o in pl if o["fold"] > 180 - half_width_deg]
    frac = 2 * half_width_deg / 180.0
    p = float(binom.sf(len(polar) - 1, len(pl), frac)) if polar else 1.0
    for o in polar:
        o["vet"] = "perpendicular-peak" if p < 0.01 else "chance-rate"
    print("%d planets with a projected obliquity to %g deg; within %g deg of 90: %d against %.1f for a uniform obliquity (binomial p %.3g); within %g deg of alignment %d, of "
          "anti-alignment %d" % (len(pl), err_max_deg, half_width_deg, len(polar), frac * len(pl), p, half_width_deg, len(aligned), len(retro)))
    for o in sorted(polar, key=lambda o: abs(o["fold"] - 90)):
        print("  %s: lambda %+.1f +- %.1f deg (true %s), P %.2f d, %s K, %s Mj, %s; %s" % (o["name"], o["lam"], o["err"], o["true"], o["P"] or 0, o["teff"], o["mass"], o["ref"], o["vet"]))
    json.dump(dict(n=len(pl), polar=polar, aligned=len(aligned), retrograde=len(retro)), open(core.results("polar_planets.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_polar_orbits", "Planets on polar orbits", "NASA Exoplanet Archive planetary systems table (ps)", "planets with a projected obliquity to %g deg: %d" % (err_max_deg, len(pl)),
            os.path.abspath(__file__), "polar", len(pl), "planets", dict(half_width_deg=half_width_deg, err_max_deg=err_max_deg),
            "a uniform projected obliquity gives %.1f within %g deg of 90; %d aligned and %d anti-aligned to the same width" % (frac * len(pl), half_width_deg, len(aligned), len(retro)),
            "a projected obliquity within %g deg of 90" % half_width_deg, "%d against %.1f (binomial p %.3g)" % (len(polar), frac * len(pl), p),
            [dict(target="planet:%s" % o["name"], stat=round(o["fold"], 1), unit="projected obliquity, deg", vet=o["vet"],
                  note="lambda %+.1f +- %.1f deg, true %s, P %.2f d, %s K, %s" % (o["lam"], o["err"], o["true"], o["P"] or 0, o["teff"], o["ref"])) for o in polar])
        print("recorded")


daylines_DOC = r"""Lines in dayside spectra: the NASA Exoplanet Archive's emission-spectroscopy table (emissionspec), each published eclipse spectrum taken in wavelength order, bins whose eclipse depth stands more than five combined errors, five times the spectrum's own bin-to-bin scatter and a tenth of the depth away from the mean of their two neighbours, in either sign; a bin brighter is light the dayside adds at one wavelength, which a molecular band does broadly and a lamp or laser would narrowly, and a bin darker is absorption; bins wider than 50 nm are broad features rather than lines, a bin, or a neighbour of a spike, whose depth falls below a fifth of the spectrum's median is a null entry of the table resting on one record, a spike within a tenth of the spectrum's range of either end sits where the detector's throughput fails, and a host that pulsates, by its SIMBAD type or the literature, stamps its own variations on every eclipse depth.

    python -I seti.py transits daylines [record] [--sigma 5] [--frac 0.1] [--wide-um 0.05] [--min-bins 5] [--edge-frac 0.1]"""


# hosts whose pulsations are published where SIMBAD's main type does not show them
DAYLINES_PULSATING = {"WASP-33": "delta Scuti pulsator (Herrero et al. 2011)"}


def daylines_main(record=False, sigma=5.0, frac=0.1, wide_um=0.05, min_bins=5, edge_frac=0.1):
    rows = core.exoplanets("SELECT plntname, plntreflink, instrument, centralwavelng, bandwidth, especlipdep, especlipdeperr1, especlipdeperr2, especlipdeplim FROM emissionspec "
                           "WHERE especlipdep IS NOT NULL AND centralwavelng IS NOT NULL")
    series = {}
    for r in rows:
        if num(r["especlipdeplim"]):
            continue
        e1, e2 = num(r["especlipdeperr1"]), num(r["especlipdeperr2"])
        err = (abs(e1) + abs(e2)) / 2 if e1 is not None and e2 is not None else abs(e1 if e1 is not None else e2) if (e1 is not None or e2 is not None) else None
        if not err:
            continue
        ref = str(r["plntreflink"] or "")
        ref = ref[ref.find(">") + 1:ref.find("</a>")] if "<a" in ref else ref
        series.setdefault((str(r["plntname"]).strip(), ref, str(r["instrument"] or "").strip()), []).append(
            (float(r["centralwavelng"]), num(r["bandwidth"]) or 0.0, float(r["especlipdep"]), err))
    out, n_series, n_bins = [], 0, 0
    for (planet, ref, inst), pts in series.items():
        pts = sorted(set(pts))
        if len(pts) < min_bins:
            continue
        n_series += 1
        n_bins += len(pts)
        lo, hi = pts[0][0], pts[-1][0]
        depths = {w: d for w, bw, d, e in pts}
        ws = [w for w, bw, d, e in pts]
        floor = 0.2 * float(np.median([d for w, bw, d, e in pts]))
        for o in science.spectrum_spikes(pts, sigma, frac):
            k = ws.index(o["um"])
            o.update(planet=planet, ref=ref, inst=inst, n=len(pts),
                     vet=("broad-feature" if o["bw"] > wide_um else "single-record" if min(depths[ws[k - 1]], depths[ws[k]], depths[ws[k + 1]]) < floor else
                          "detector-edge" if min(o["um"] - lo, hi - o["um"]) < edge_frac * (hi - lo) else None))
            out.append(o)
    hosts = sorted({o["planet"].rsplit(" ", 1)[0] for o in out if o["vet"] is None})
    types = {}
    if hosts:
        for r in core.simbad("SELECT i.id, b.otype FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join("'%s'" % h.replace("'", "''") for h in hosts)):
            types[str(r["id"]).strip()] = str(r["otype"]).strip()
    for o in out:
        host = o["planet"].rsplit(" ", 1)[0]
        if o["vet"] is None and (types.get(host, "") in ("dS*", "gD*", "RR*", "Ce*", "Pu*", "bC*", "RV*", "LP*", "Mi*", "PV*") or host in DAYLINES_PULSATING):
            o["vet"] = "variable-star"
            o["host_type"] = DAYLINES_PULSATING.get(host, types.get(host))
    left = [o for o in out if o["vet"] is None]
    print("%d eclipse spectra of %d bins; bins %g sigma and a %g fraction from their neighbours: %d (%d broad, %d on or beside a null entry, %d at a detector's edge, %d at pulsating hosts); "
          "unexplained %d" % (n_series, n_bins, sigma, frac, len(out), sum(o["vet"] == "broad-feature" for o in out), sum(o["vet"] == "single-record" for o in out),
                             sum(o["vet"] == "detector-edge" for o in out), sum(o["vet"] == "variable-star" for o in out), len(left)))
    for o in sorted(out, key=lambda o: -abs(o["z"])):
        print("  %s (%s, %s): %.4f um (%.4f wide) depth %.1f, %+.1f from its neighbours (%.1f sigma, scatter %.1f); %s" % (
            o["planet"], o["ref"], o["inst"], o["um"], o["bw"], o["depth"], o["dev"], o["z"], o["scale"], o["vet"] or "unexplained"))
    json.dump(dict(series=n_series, bins=n_bins, hits=out), open(core.results("dayside_lines.json"), "w"), indent=1)
    if record:
        core.record_run(
            "exoplanet_dayside_lines", "Lines in dayside spectra", "NASA Exoplanet Archive emission spectroscopy table (emissionspec)",
            "published eclipse spectra with %d or more bins: %d, %d bins" % (min_bins, n_series, n_bins), os.path.abspath(__file__), "daylines", n_series, "spectra",
            dict(sigma=sigma, frac=frac, wide_um=wide_um, min_bins=min_bins, edge_frac=edge_frac), "each spectrum's own bin-to-bin scatter and the combined errors of the three bins; "
            "null entries and the detectors' edges", "a bin %g sigma and a %g fraction of the depth away from the mean of its neighbours" % (sigma, frac),
            "%d: %d broad, %d on or beside a null entry, %d at a detector's edge, %d at pulsating hosts; %d unexplained" % (
                len(out), sum(o["vet"] == "broad-feature" for o in out), sum(o["vet"] == "single-record" for o in out), sum(o["vet"] == "detector-edge" for o in out),
                sum(o["vet"] == "variable-star" for o in out), len(left)),
            [dict(target="planet:%s" % o["planet"], stat=round(o["z"], 1), unit="sigma from the neighbouring bins", vet=o["vet"],
                  note="%s, %s, %.4f um (%.4f wide), depth %.1f, %+.1f" % (o["ref"], o["inst"], o["um"], o["bw"], o["depth"], o["dev"])) for o in out])
        print("recorded")


ttvzip_DOC = r"""Transit-timing residuals that compress: Kepler transit times of Holczer et al. (2016) for every KOI with thirty or more clean times, a line and the two strongest sinusoids removed, the residual divided by its errors, cut into four quantile symbols in transit order and compressed with zlib, against the same symbols shuffled and against first-order autoregressive surrogates at the residual's own lag-one correlation; a sequence below both nulls at the smallest p the draws allow carries structure beyond a line, two periods and red noise, and a published TTV system or an expected count of such sequences explains it.

    python -I seti.py transits ttvzip [record] [--min-n 30] [--draws 300] [--seed 31]"""


def ttvzip_main(record=False, min_n=30, draws=300, seed=31):
    import zlib
    from astropy.timeseries import LombScargle

    def fit(A, y, w):
        p = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        return y - A @ p

    def strip(x, y, w):
        cols, periods = [np.ones_like(x), x], []
        r = fit(np.vstack(cols).T, y, w)
        for _ in range(2):
            fr, pw = LombScargle(x, r, 1 / np.sqrt(w)).autopower(minimum_frequency=1 / (2 * np.ptp(x) + 1), maximum_frequency=0.5, samples_per_peak=5)
            f = float(fr[int(np.argmax(pw))])
            periods.append(1 / f)
            cols += [np.sin(2 * np.pi * f * x), np.cos(2 * np.pi * f * x)]
            r = fit(np.vstack(cols).T, y, w)
        return r, periods

    def code(z):
        q = np.quantile(z, [0.25, 0.5, 0.75])
        return len(zlib.compress(bytes(np.searchsorted(q, z).astype(np.uint8) + 65), 9))

    cats = core.vizier("J/ApJS/225/9")
    t2 = {str(r["KOI"]): r for r in cats[0]}
    by = {}
    for r in cats[1]:
        if any(str(r[c]).strip() not in ("", "--", "0", "None", "nan") for c in ("f_O-C", "Out", "Over")):
            continue
        oc, e = num(r["O-C"]), num(r["e_O-C"])
        if oc is not None and e and np.isfinite(oc) and np.isfinite(e) and e > 0:
            by.setdefault(str(r["KOI"]), []).append((int(r["N"]), oc, e))
    rng = np.random.default_rng(seed)
    res = []
    for koi, rows in by.items():
        if len(rows) < min_n:
            continue
        a = np.array(sorted(rows))
        r, periods = strip(a[:, 0], a[:, 1], 1 / a[:, 2] ** 2)
        z = r / a[:, 2]
        obs = code(z)
        sh = np.array([code(rng.permutation(z)) for _ in range(draws)])
        rho = min(max(float(np.corrcoef(z[:-1], z[1:])[0, 1]), 0.0), 0.95)
        ar = []
        for _ in range(draws):
            u = np.empty(len(z))
            u[0] = rng.normal()
            for i in range(1, len(z)):
                u[i] = rho * u[i - 1] + rng.normal() * math.sqrt(1 - rho ** 2)
            ar.append(code(u))
        res.append(dict(koi=koi, kic=str(t2[koi]["KIC"]) if koi in t2 else "", n=len(z), obs=obs, sh=float(sh.mean()), rho=rho, periods=periods,
                        p_sh=float((1 + np.sum(sh <= obs)) / (draws + 1)), p_ar=float((1 + np.sum(np.array(ar) <= obs)) / (draws + 1)),
                        chi2=float(np.sum(z ** 2) / max(len(z) - 6, 1))))
    floor = 1 / (draws + 1)
    both = [r for r in res if r["p_sh"] <= floor and r["p_ar"] <= floor]
    expected = len(res) * floor ** 2
    # residual structure a published TTV system explains: reduced chi2 well above one after the two sinusoids
    for r in res:
        r["vet"] = "timing-variation" if r in both and r["chi2"] > 3.0 else None
    rest = [r for r in both if r["vet"] is None]
    # the rest are the chance count when the Poisson tail at the expected number is not small
    p_rest = 1.0 - sum(math.exp(-expected) * expected ** i / math.factorial(i) for i in range(len(rest)))
    for r in rest:
        r["vet"] = "chance-rate" if p_rest > 0.01 else None
    frac = float(np.mean([r["p_sh"] < 0.01 for r in res])) if res else 0.0
    print("%d KOIs with %d or more clean transit times; below p %.4f on both nulls: %d (about %.2f by chance); p_sh below 0.01 in %.3f of them" % (
        len(res), min_n, floor, len(both), expected, frac))
    for r in both:
        print("  KOI %s (KIC %s) n %d: %d bytes against %.1f shuffled, lag-1 %.2f, periods %.1f and %.1f transits removed, reduced chi2 %.2f; %s" % (
            r["koi"], r["kic"], r["n"], r["obs"], r["sh"], r["rho"], r["periods"][0], r["periods"][1], r["chi2"], r["vet"] or "unexplained"))
    if record:
        core.record_run(
            "kepler_ttv_compressibility", "Transit-timing residuals that compress", "Holczer et al. (2016) Kepler transit times (VizieR J/ApJS/225/9)",
            "%d KOIs with %d or more clean transit times" % (len(res), min_n), os.path.abspath(__file__), "ttvzip", len(res), "KOIs",
            dict(min_n=min_n, draws=draws, seed=seed), dict(below_both=len(both), expected=expected, frac_p_sh_001=frac),
            "zlib length of four quantile symbols below every shuffle and every AR(1) surrogate (p %.4f)" % floor,
            "%d below both nulls at p %.4f against %.2f by chance; %.1f%% below 0.01 against shuffles alone; %d unexplained" % (
                len(both), floor, expected, 100 * frac, sum(r["vet"] is None for r in both)),
            [dict(target="koi:%s" % r["koi"], stat=r["obs"], p=max(r["p_sh"], r["p_ar"]), unit="zlib bytes of the quantile symbols", vet=r["vet"],
                  note="KIC %s; %d times; %.1f shuffled; lag-1 %.2f; reduced chi2 %.2f" % (r["kic"], r["n"], r["sh"], r["rho"], r["chi2"])) for r in both])
        print("recorded")


APPROACHES = {
    "ttvzip": (lambda: core.run_cli(ttvzip_main), ttvzip_DOC),
    "daylines": (lambda: core.run_cli(daylines_main), daylines_DOC),
    "polar": (lambda: core.run_cli(polar_main), polar_DOC),
    "translines": (lambda: core.run_cli(translines_main), translines_DOC),
    "spinlaps": (lambda: core.run_cli(spinlaps_main), spinlaps_DOC),
    "ttvabsent": (lambda: core.run_cli(ttvabsent_main), ttvabsent_DOC),
    "deeptransits": (lambda: core.run_cli(deeptransits_main), deeptransits_DOC),
    "wdrefuge": (lambda: core.run_cli(wdrefuge_main), wdrefuge_DOC),
    "faceon": (lambda: core.run_cli(faceon_main), faceon_DOC),
    "longtransits": (lambda: core.run_cli(longtransits_main), longtransits_DOC),
    "spinsteps": (lambda: core.run_cli(spinsteps_main), spinsteps_DOC),
    "wholedays": (lambda: core.run_cli(wholedays_main), wholedays_DOC),
    "circles": (lambda: core.run_cli(circles_main), circles_DOC),
    "randomtransiter": (lambda: core.run_cli(randomtransiter_main), randomtransiter_DOC),
    "earthcopy": (lambda: core.run_cli(earthcopy_main), earthcopy_DOC),
    "exactres": (lambda: core.run_cli(exactres_main), exactres_DOC),
    "ironplanets": (lambda: core.run_cli(ironplanets_main), ironplanets_DOC),
    "rrlclock": (lambda: core.run_cli(rrlclock_main), rrlclock_DOC),
    "cepclock": (lambda: core.run_cli(cepclock_main), cepclock_DOC),
    "ebclock": (lambda: core.run_cli(ebclock_main), ebclock_DOC),
    "nssclock": (lambda: core.run_cli(nssclock_main), nssclock_DOC),
    "uniform": (lambda: core.run_cli(uniform_main), uniform_DOC),
    "copysystems": (lambda: core.run_cli(copysystems_main), copysystems_DOC),
    "mirrorplanets": (lambda: core.run_cli(mirrorplanets_main), mirrorplanets_DOC),
    "insidestar": (lambda: core.run_cli(insidestar_main), insidestar_DOC),
    "rocheplanets": (lambda: core.run_cli(rocheplanets_main), rocheplanets_DOC),
    "copyplanets": (lambda: core.run_cli(copyplanets_main), copyplanets_DOC),
    "dayside": (lambda: core.run_cli(dayside_main), dayside_DOC),
    "constants": (lambda: core.run_cli(constants_main), constants_DOC),
    "hillpack": (lambda: core.run_cli(hillpack_main), hillpack_DOC),
    "coorbital": (lambda: core.run_cli(coorbital_main), coorbital_DOC),
    "earthyear": (lambda: core.run_cli(earthyear_main), earthyear_DOC),
    "earthday": (lambda: core.run_cli(earthday_main), earthday_DOC),
    "sharedperiod": (lambda: core.run_cli(sharedperiod_main), sharedperiod_DOC),
    "steps": (lambda: core.run_cli(steps_main), steps_DOC),
    "depthgrowth": (depthgrowth_cli, depthgrowth_DOC),
    "roche": (roche_cli, roche_DOC),
    "lightweight": (lightweight_cli, lightweight_DOC),
    "impulse": (impulse_cli, impulse_DOC),
    "synchronous": (synchronous_cli, synchronous_DOC),
    "aimed": (aimed_cli, aimed_DOC),
    "escape": (escape_cli, escape_DOC),
    "hollow": (lambda: core.run_cli(hollow_main), hollow_DOC),
    "stability": (stability_cli, stability_DOC),
    "nightheat": (lambda: core.run_cli(nightheat_main), nightheat_DOC),
    "shapes_kepler": (shapes_kepler_cli, shapes_kepler_DOC),
    "shapes_tess": (shapes_tess_cli, shapes_tess_DOC),
    "ttv": (ttv_cli, ttv_DOC),
    "cepheids": (cepheids_cli, cepheids_DOC),
    "nightside": (nightside_cli, nightside_DOC),
    "exobelts": (exobelts_cli, exobelts_DOC),
    "commensurability": (commensurability_cli, commensurability_DOC),
}


