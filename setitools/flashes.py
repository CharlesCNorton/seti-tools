"""Specular glints: brief flashes folded at a planet's or an asteroid's period."""
import argparse
import csv
import glob
import io
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from setitools import catalogues, core, science
from setitools.core import num, quiet_warnings


# --------------------------------------------------------------------------------------------------------------

tess20_DOC = r"""Glints in TESS 20 s light curves within 30 pc: 5-sigma flashes of at most three cadences tested for periodicity against draws from good cadences.

    python -I seti.py flashes tess20 targets
    python -I seti.py flashes tess20 fetch [--threads 8] [--per-star N] [--list SHARE.json]
    python -I seti.py flashes tess20 search [--list SHARE.json] [--out glints.json] [--procs N]
    python -I seti.py flashes tess20 merge     (merges the shares' glints_*.json and records the run in the coverage store)"""

tess20_HERE = core.RESULTS
tess20_DL = os.environ.get("SETI_TESS20", core.data("tess20"))
tess20_UA = core.UA


def tess20_get(url, timeout=120):
    """Bytes of a URL, or None when missing or still failing after three tries."""
    return core.get(url, timeout=timeout, retries=3, soft=True, pause=0)


def tess20_targets():
    lists = {}
    for s in range(27, 140):
        b = tess20_get("https://tess.mit.edu/public/target_lists/20s/all_targets_20s_S%03d_v1.csv" % s)
        if not b or not b.lstrip().startswith(b"#"):
            if s > 100:
                break
            continue
        rows = [ln for ln in b.decode(errors="replace").splitlines() if ln and not ln.startswith("#")]
        lists[s] = list(csv.DictReader(io.StringIO("\n".join(rows))))
    print("%d sectors with 20 s target lists, %d target entries" % (len(lists), sum(len(v) for v in lists.values())),
          flush=True)
    near = core.gaia("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp FROM "
                         "gaiadr3.gaia_source WHERE parallax > 33.33 AND phot_g_mean_mag < 16")
    ra0 = np.array([float(r["ra"]) - float(r["pmra"] or 0) / 3.6e6 * 16.0 / math.cos(math.radians(float(r["dec"])))
                    for r in near])
    de0 = np.array([float(r["dec"]) - float(r["pmdec"] or 0) / 3.6e6 * 16.0 for r in near])
    out = []
    for s, rows in lists.items():
        for t in rows:
            try:
                ra, dec = float(t["RA"]), float(t["Dec"])
            except (KeyError, ValueError):
                continue
            d = np.hypot((ra0 - ra) * math.cos(math.radians(dec)), de0 - dec) * 3600.0
            k = int(np.argmin(d))
            if d[k] < 10.0:
                g = near[k]
                out.append(dict(tic=int(t["TICID"]), sector=s, gaia=g["source_id"], sep=round(float(d[k]), 2),
                                dist_pc=round(1000.0 / float(g["parallax"]), 2), g=float(g["phot_g_mean_mag"]),
                                bp_rp=float(g["bp_rp"] or "nan")))
    json.dump(out, open(os.path.join(tess20_HERE, "glint_targets.json"), "w"))
    print("%d light curves of %d stars within 30 pc" % (len(out), len({o["gaia"] for o in out})))


def tess20_fetch_one(t):
    path = os.path.join(tess20_DL, "s%04d_%d_fast-lc.fits" % (t["sector"], t["tic"]))
    if os.path.exists(path):
        return True
    keys = core.spoc_keys(t["tic"], t["sector"], fast=True)
    if not keys:
        return False
    b = tess20_get(core.S3 + keys[0], timeout=300)
    if not b or not b.startswith(b"SIMPLE"):
        return False
    open(path + ".tmp", "wb").write(b)
    os.replace(path + ".tmp", path)
    return True


def tess20_fetch(threads, per_star, share=None):
    os.makedirs(tess20_DL, exist_ok=True)
    ts = json.load(open(share or os.path.join(tess20_HERE, "glint_targets.json")))
    if per_star and not share:
        by = {}
        for t in sorted(ts, key=lambda t: t["sector"]):
            by.setdefault(t["gaia"], []).append(t)
        ts = [t for v in by.values() for t in v[:per_star]]
    with ThreadPoolExecutor(threads) as ex:
        ok = list(ex.map(tess20_fetch_one, ts))
    print("%d of %d light curves on disk" % (sum(ok), len(ts)))


def tess20_flashes(t, f, q):
    good = (q == 0) & np.isfinite(f) & np.isfinite(t)
    t, f = t[good], f[good]
    if len(f) < 1000:
        return None
    from scipy.ndimage import median_filter
    base = median_filter(f, size=31, mode="nearest")
    r = f / base - 1.0
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    idx = np.nonzero(r > 5 * sig)[0]
    out, seen = [], set()
    for i in idx:
        if i in seen:
            continue
        j = i
        while j + 1 < len(r) and r[j + 1] > 5 * sig and t[j + 1] - t[j] < 30 / 86400.0:
            j += 1
        seen.update(range(i, j + 1))
        if j - i >= 3 or i < 4 or j > len(r) - 5:
            continue
        before, after = r[i - 4:i - 1], r[j + 2:j + 5]
        if np.all(np.abs(before) < 2 * sig) and np.all(np.abs(after) < 2 * sig) and \
                t[i] - t[i - 4] < 120 / 86400.0 and t[j + 4] - t[j] < 120 / 86400.0:
            k = i + int(np.argmax(r[i:j + 1]))
            out.append((float(t[k]), float(r[k] / sig), j - i + 1))
    return out, t, sig


def tess20_best_period(times, tol=40.0 / 86400.0):
    times = np.sort(np.asarray(times))
    best = (0, None)
    for a in range(len(times)):
        for b in range(a + 1, len(times)):
            for k in range(1, 6):
                P = (times[b] - times[a]) / k
                if P < 0.01:
                    continue
                ph = (times - times[a]) / P
                n = int(np.sum(np.abs(ph - np.round(ph)) * P < tol))
                if n > best[0]:
                    best = (n, P)
    return best


def tess20_one_curve(p):
    """Flashes in one light-curve file: (name, flashes, good times) or None."""
    from astropy.io import fits
    try:
        with fits.open(os.path.join(tess20_DL, p)) as h:
            d = h[1].data
            res = tess20_flashes(np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float),
                          np.asarray(d["QUALITY"], int))
    except Exception:
        return None
    if res is None:
        return None
    return p, res[0], res[1].astype(np.float32)


def tess20_search(share=None, out="glints.json", procs=1):
    ts = {(t["sector"], t["tic"]): t for t in json.load(open(share or os.path.join(tess20_HERE, "glint_targets.json")))}
    files = [p for p in sorted(os.listdir(tess20_DL)) if p.endswith("_fast-lc.fits")
             and (int(p.split("_")[0][1:]), int(p.split("_")[1])) in ts]
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pool:
            results = pool.map(tess20_one_curve, files, chunksize=4)
    else:
        results = [tess20_one_curve(p) for p in files]
    per_star = {}
    stats = dict(curves=0, flashes=0, hours=0.0)
    for item in results:
        if item is None:
            continue
        p, fl, tg = item
        s, tic = p.split("_")[0], int(p.split("_")[1])
        meta = ts.get((int(s[1:]), tic))
        stats["curves"] += 1
        stats["flashes"] += len(fl)
        stats["hours"] += len(tg) * 20.0 / 3600.0
        key = meta["gaia"] if meta else str(tic)
        e = per_star.setdefault(key, dict(meta=meta, flashes=[], good_times=[]))
        e["flashes"] += [dict(sector=int(s[1:]), t=x[0], snr=round(x[1], 1), n=x[2]) for x in fl]
        e["good_times"].append(tg)
    print("%d light curves, %.0f star-hours at 20 s, %d short symmetric flashes (%.2f per star-day)"
          % (stats["curves"], stats["hours"], stats["flashes"], stats["flashes"] / max(stats["hours"] / 24.0, 1e-9)))
    rng = np.random.default_rng(3)
    report = []
    for key, e in per_star.items():
        # the 60 strongest flashes carry any periodic family
        times = [x["t"] for x in sorted(e["flashes"], key=lambda x: -x["snr"])[:60]]
        if len(times) < 3:
            continue
        n, P = tess20_best_period(times)
        allt = np.concatenate(e["good_times"])
        sims = []
        for _ in range(200):
            sims.append(tess20_best_period(rng.choice(allt, size=len(times), replace=False))[0])
        p = (1 + sum(1 for v in sims if v >= n)) / 201.0
        report.append(dict(gaia=key, meta=e["meta"], n_flashes=len(times), best_n=n, period_d=P, p=p,
                           flashes=e["flashes"]))
    report.sort(key=lambda r: r["p"])
    stars_all = [dict(gaia=k, meta=e["meta"], n_flashes=len(e["flashes"]), flashes=e["flashes"]) for k, e in per_star.items()]
    json.dump(dict(stats=stats, stars=report, all=stars_all), open(os.path.join(tess20_HERE, out), "w"), default=float)
    print("%d stars with three or more flashes" % len(report))
    for r in report[:12]:
        m = r["meta"] or {}
        print("  Gaia DR3 %s (%.1f pc, G %.1f, BP-RP %.2f): %d flashes, %d on one period of %.4f d, p %.3f"
              % (r["gaia"], m.get("dist_pc", 0), m.get("g", 0), m.get("bp_rp", 0), r["n_flashes"], r["best_n"],
                 r["period_d"] or 0, r["p"]))


def tess20_merge():
    """Merge shares, compare periodicity p-values with uniform, record one row per star."""
    stats = dict(curves=0, flashes=0, hours=0.0)
    report, every = [], {}
    import glob
    for path in sorted(glob.glob(os.path.join(tess20_HERE, "glints_*.json"))):
        d = json.load(open(path))
        if "stats" not in d:
            continue
        for k in stats:
            stats[k] += d["stats"][k]
        report += d["stars"]
        for s in d["all"]:
            e = every.setdefault(s["gaia"], dict(n=0))
            e["n"] += s["n_flashes"]
    ps = np.array([r["p"] for r in report])
    null = {str(c): dict(n=int((ps <= c + 1e-9).sum()), expected=round(c * len(ps), 1)) for c in (0.005, 0.01, 0.02, 0.05)}
    print("%d light curves, %.0f star-hours, %d flashes (%.2f per star-day); %d stars with three or more flashes"
          % (stats["curves"], stats["hours"], stats["flashes"], stats["flashes"] / max(stats["hours"] / 24, 1e-9), len(ps)))
    for c, v in null.items():
        print("  p <= %s: %d stars against %.1f for uniform p" % (c, v["n"], v["expected"]))
    best = {r["gaia"]: r for r in report}
    rows = [dict(target="gaia_dr3:" + k, stat=best[k]["best_n"] if k in best else None,
                 p=best[k]["p"] if k in best else None,
                 note="%d flashes%s" % (e["n"], ", period %.5f d" % best[k]["period_d"] if k in best and best[k]["period_d"] else ""))
            for k, e in every.items()]
    db = core.connect()
    sid = "tess20_glints"
    core.record_search(sid, "Specular glints, TESS 20 s", archive="MAST TESS SPOC fast-lc (STScI AWS mirror)",
                           release="sectors 27 on", selection="20 s targets matched to Gaia DR3 stars within 30 pc; the "
                           "shares in share_glint_<part>.json", code=os.path.abspath(__file__), entry="tess20",
                           params=dict(flash_sigma=5, max_cadences=3, quiet_sigma=2, period_tol_s=40, draws=200,
                                       strongest=60),
                           n_targets=len(rows), null_summary=dict(stats=stats, p=null),
                           result="no star's flashes recur at a period beyond chance", ledger="Specular glints, TESS 20 s",
                           db=db)
    core.record_targets(sid, rows, db=db)
    db.close()
    print("recorded %d stars under %s" % (len(rows), sid))


def tess20_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("targets", "fetch", "search", "merge"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--per-star", type=int, default=0, help="light curves per star, earliest sectors first (0: all)")
    p.add_argument("--list", default=None, help="JSON list of target records to fetch or search (one share)")
    p.add_argument("--out", default="glints.json")
    p.add_argument("--procs", type=int, default=1)
    a = p.parse_args()
    {"targets": tess20_targets, "fetch": lambda: tess20_fetch(a.threads, a.per_star, a.list),
     "search": lambda: tess20_search(a.list, a.out, a.procs), "merge": tess20_merge}[a.mode]()


def tess20_cli():
    quiet_warnings()
    tess20_main()


# --------------------------------------------------------------------------------------------------------------

toi_DOC = r"""Glints at TOI periods in SPOC two-minute light curves: the busiest 2.5-cadence phase window against 400 draws.

    python -I seti.py flashes toi targets
    python -I seti.py flashes toi split N                       (shares toi_share<i>.json, balanced by sector count)
    python -I seti.py flashes toi stream --list SHARE.json --out toi_glints_<part>.jsonl [--procs N]
    python -I seti.py flashes toi report toi_glints_*.jsonl
    python -I seti.py flashes toi record toi_glints_*.jsonl"""

toi_ROOT = os.environ.get("TOI_ROOT", core.data("toi_glints"))
toi_HERE = core.RESULTS
toi_TOI_CSV = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
toi_CAD = 120.0 / 86400.0
toi_BTJD = 2457000.0


def toi_targets():
    os.makedirs(toi_ROOT, exist_ok=True)
    raw = core.get(toi_TOI_CSV, timeout=300, retries=3).decode("utf-8", "replace")
    by = {}
    for r in csv.DictReader(io.StringIO(raw)):
        if (r.get("TFOPWG Disposition") or "").strip() in ("FP", "FA"):
            continue
        P, ep, dur = num(r.get("Period (days)")), num(r.get("Epoch (BJD)")), num(r.get("Duration (hours)"))
        if not P or P <= 0 or not ep:
            continue
        tic = int(r["TIC ID"])
        x = by.setdefault(tic, dict(tic=tic, tmag=num(r.get("TESS Mag")), sectors=set(), tois=[]))
        x["sectors"] |= {int(s) for s in re.findall(r"\d+", r.get("Sectors") or "")}
        x["tois"].append(dict(toi=r["TOI"], period=P, t0=ep - toi_BTJD, dur_h=dur, disp=r.get("TFOPWG Disposition")))
    out = [dict(x, sectors=sorted(x["sectors"])) for x in by.values() if x["sectors"]]
    json.dump(out, open(os.path.join(toi_ROOT, "targets.json"), "w"))
    print("%d TOI hosts (false positives and false alarms left out), %d TOIs, %d star-sectors"
          % (len(out), sum(len(x["tois"]) for x in out), sum(len(x["sectors"]) for x in out)))


def toi_split(n):
    ts = sorted(json.load(open(os.path.join(toi_ROOT, "targets.json"))), key=lambda t: -len(t["sectors"]))
    for i in range(n):
        json.dump(ts[i::n], open(os.path.join(toi_ROOT, "toi_share%d.json" % i), "w"))
    print("%d shares of %s stars" % (n, ", ".join(str(len(ts[i::n])) for i in range(n))))


def toi_one_star(t):
    quiet_warnings()
    fl, goods, used = [], [], []
    for s in t["sectors"]:
        b = core.spoc_bytes(t["tic"], s)
        if not b:
            continue
        try:
            tt, f, _ = core.read(b, columns=("PDCSAP_FLUX",), normalise=False)
        except Exception:
            continue
        if len(tt) < 1000:
            continue
        ev, _ = science.flash_events(tt, f, toi_CAD, quiet=4)
        fl += ev
        goods.append(tt)
        used.append(s)
    if not goods:
        return dict(tic=t["tic"], sectors=[], days=0.0, n_flash=0, flashes=[], tois=[])
    good_t = np.concatenate(goods)
    rng = np.random.default_rng(t["tic"] % 100000)
    times = [x[0] for x in fl]
    res = []
    for k in t["tois"]:
        if len(times) >= 3:
            n0, fap = science.phase_cluster(times, good_t, k["period"], k["t0"], 2.5 * toi_CAD, rng)
        else:
            n0, fap = len(times), 1.0
        res.append(dict(k, n_cluster=n0, fap=fap))
    return dict(tic=t["tic"], sectors=used, days=float(len(good_t) * toi_CAD), n_flash=len(fl), flashes=fl[:300], tois=res)


def toi_stream(share, out, procs):
    from concurrent.futures import ProcessPoolExecutor
    ts = json.load(open(share or os.path.join(toi_ROOT, "targets.json")))
    path = os.path.join(toi_HERE, out)
    done = set()
    if os.path.exists(path):
        done = {json.loads(ln)["tic"] for ln in open(path) if ln.strip()}
    todo = [t for t in ts if t["tic"] not in done]
    print("%d stars to search (%d done)" % (len(todo), len(done)), flush=True)
    with ProcessPoolExecutor(procs) as ex, open(path, "a") as fh:
        for r in ex.map(toi_one_star, todo, chunksize=2):
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()


def toi_report(paths):
    seen, res = set(), []
    for p in paths:
        for ln in open(p):
            if ln.strip():
                r = json.loads(ln)
                if r["tic"] not in seen:
                    seen.add(r["tic"])
                    res.append(r)
    tested = [k for r in res for k in r["tois"] if r["n_flash"] >= 3]
    print("%d stars, %.0f days of two-minute cadence, %d flashes; %d TOIs on stars with three or more flashes"
          % (len(res), sum(r["days"] for r in res), sum(r["n_flash"] for r in res), len(tested)))
    for thr, label in ((0.01, "below 0.01"), (1 / 401.0 + 1e-9, "at the floor 1/401")):
        k = sum(1 for x in tested if x["fap"] < thr or (thr < 0.01 and x["fap"] <= thr))
        print("  TOIs with flashes clustered in phase at FAP %s: %d (at most %.1f expected)" % (label, k, thr * len(tested)))
    for r in res:
        for k in r["tois"]:
            if r["n_flash"] >= 3 and k["fap"] < 0.01:
                print("  TIC %d TOI %s P %.4f d: %d of %d flashes within 2.5 cadences of one phase, FAP %.4f"
                      % (r["tic"], k["toi"], k["period"], k["n_cluster"], r["n_flash"], k["fap"]))
    return res


def toi_record(paths):
    """Record one row per star with its lowest TOI false-alarm probability."""
    res = toi_report(paths)
    tested = [k for r in res for k in r["tois"] if r["n_flash"] >= 3]
    below = sum(1 for k in tested if k["fap"] < 0.01)
    floor = sum(1 for k in tested if k["fap"] <= 1 / 401.0 + 1e-9)
    core.record_search(
        "tess_toi_glints", "Specular glints at TESS planet-candidate periods",
        archive="TESS SPOC two-minute light curves (STScI AWS mirror)", release="sectors 1-%d" % max(
            s for r in res for s in r["sectors"]),
        selection="%d hosts of TOIs that are neither false positives nor false alarms (ExoFOP TOI table)" % len(res),
        code="seti.py flashes toi", params=dict(flash_sigma=5, max_cadences=3, clean_window=4, phase_window_cadences=2.5,
                                          null_draws=400),
        n_targets=len(res),
        null_summary="%d TOIs on stars with three or more flashes: %d below FAP 0.01 against at most %.1f, %d at the "
                     "1/401 floor against at most %.1f" % (len(tested), below, 0.01 * len(tested), floor,
                                                           len(tested) / 401.0),
        sensitivity="flashes of 5 robust deviations lasting 1-3 cadences (2-6 min); %.0f star-days, %d flashes"
                    % (sum(r["days"] for r in res), sum(r["n_flash"] for r in res)),
        result="no TOI period gathers flashes at one phase beyond chance",
        ledger="Specular glints, TESS TOI hosts")
    core.clear_targets("tess_toi_glints")
    core.record_targets("tess_toi_glints", [
        dict(target="tic:%s" % r["tic"], stat=float(r["n_flash"]), unit="flashes",
             p=min((k["fap"] for k in r["tois"]), default=None) if r["n_flash"] >= 3 else None,
             note="%.0f days in %d sectors; lowest TOI false-alarm probability over its folded periods"
                  % (r["days"], len(r["sectors"]))) for r in res])
    print("recorded")


def toi_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("targets", "split", "stream", "report", "record"))
    p.add_argument("paths", nargs="*")
    p.add_argument("--list")
    p.add_argument("--out", default="toi_glints.jsonl")
    p.add_argument("--procs", type=int, default=2)
    a = p.parse_args()
    if a.mode == "targets":
        toi_targets()
    elif a.mode == "split":
        toi_split(int(a.paths[0]) if a.paths else 2)
    elif a.mode == "stream":
        toi_stream(a.list, a.out, a.procs)
    elif a.mode == "record":
        toi_record(a.paths)
    else:
        toi_report(a.paths)


def toi_cli():
    quiet_warnings()
    toi_main()


# --------------------------------------------------------------------------------------------------------------

kepler_DOC = r"""Glints at KOI periods in Kepler short cadence: the busiest 2.5-cadence phase window against flash times drawn from good cadences.

    python -I seti.py flashes kepler targets
    python -I seti.py flashes kepler fetch [--threads 4] [--list SHARE.json]
    python -I seti.py flashes kepler search [--list SHARE.json] [--out kepler_glints.json] [--procs 2]
    python -I seti.py flashes kepler split          (two shares of the targets, share0.json and share1.json)
    python -I seti.py flashes kepler stream --list SHARE.json --out kepler_glints_<part>.jsonl [--procs 2]
    python -I seti.py flashes kepler report kepler_glints_*.jsonl"""

kepler_ROOT = os.environ.get("KEPLER_SC", core.data("kepler_sc"))
kepler_HERE = core.RESULTS
kepler_S3 = "https://stpubdata.s3.amazonaws.com/"
kepler_CAD = 58.85 / 86400.0


def kepler_targets():
    os.makedirs(kepler_ROOT, exist_ok=True)
    kois = core.exoplanets("SELECT kepid, kepoi_name, koi_disposition, koi_period, koi_time0bk, koi_duration, "
                               "koi_prad FROM cumulative WHERE koi_disposition IN ('CONFIRMED', 'CANDIDATE')")
    by = {}
    for k in kois:
        by.setdefault(str(k["kepid"]), []).append(dict(koi=k["kepoi_name"], period=float(k["koi_period"]),
                                                       t0=float(k["koi_time0bk"]), dur_h=float(k["koi_duration"] or 0),
                                                       prad=k["koi_prad"]))
    from concurrent.futures import ThreadPoolExecutor

    def listing(kic):
        k9 = "%09d" % int(kic)
        b = core.get(kepler_S3 + "?list-type=2&prefix=kepler/public/lightcurves/%s/%s/" % (k9[:4], k9), timeout=60,
                    retries=3, soft=True, pause=1)
        return kic, re.findall(r"<Key>([^<]*_slc\.fits)</Key>", b.decode()) if b else []
    out = []
    with ThreadPoolExecutor(8) as ex:
        for kic, keys in ex.map(listing, sorted(by)):
            if keys:
                out.append(dict(kic=kic, kois=by[kic], files=keys))
    json.dump(out, open(os.path.join(kepler_ROOT, "targets.json"), "w"))
    print("%d KOIs on %d stars; %d stars with short cadence, %d short-cadence files"
          % (len(kois), len(by), len(out), sum(len(t["files"]) for t in out)))


def kepler_fetch(threads, share=None):
    from concurrent.futures import ThreadPoolExecutor
    ts = json.load(open(share or os.path.join(kepler_ROOT, "targets.json")))
    keys = [k for t in ts for k in t["files"]]
    os.makedirs(os.path.join(kepler_ROOT, "slc"), exist_ok=True)

    def one(k):
        p = os.path.join(kepler_ROOT, "slc", os.path.basename(k))
        if os.path.exists(p):
            return 1
        b = core.get(kepler_S3 + k, timeout=300, retries=3, soft=True, pause=2)
        if not b or not b.startswith(b"SIMPLE"):
            return 0
        open(p + ".tmp", "wb").write(b)
        os.replace(p + ".tmp", p)
        return 1
    n = 0
    with ThreadPoolExecutor(threads) as ex:
        for i, r in enumerate(ex.map(one, keys)):
            n += r
            if i % 500 == 0:
                print("%d of %d files" % (n, len(keys)), flush=True)
    print("%d of %d short-cadence files on disk" % (n, len(keys)))


def kepler_flashes(t, f, q):
    good = (q == 0) & np.isfinite(f) & np.isfinite(t)
    t, f = t[good], f[good]
    if len(f) < 1000:
        return [], t
    from scipy.ndimage import median_filter
    r = f / median_filter(f, size=31, mode="nearest") - 1.0
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    out, seen = [], set()
    for i in np.nonzero(r > 5 * sig)[0]:
        if i in seen:
            continue
        j = i
        while j + 1 < len(r) and r[j + 1] > 5 * sig and t[j + 1] - t[j] < 1.5 * kepler_CAD:
            j += 1
        seen.update(range(i, j + 1))
        if j - i >= 3 or i < 5 or j > len(r) - 6:
            continue
        before, after = r[i - 5:i - 1], r[j + 2:j + 6]
        if np.all(np.abs(before) < 2 * sig) and np.all(np.abs(after) < 2 * sig) and \
                t[i] - t[i - 5] < 6 * kepler_CAD and t[j + 5] - t[j] < 6 * kepler_CAD:
            k = i + int(np.argmax(r[i:j + 1]))
            out.append((float(t[k]), float(r[k] / sig), int(j - i + 1)))
    return out, t


def kepler_phase_cluster(times, good_t, P, t0, rng, ntrial=400):
    """Largest flash count in a 2.5-cadence phase window and its false-alarm probability."""
    w = 2.5 * kepler_CAD / P

    def best(ts):
        ph = np.sort(((np.asarray(ts) - t0) / P) % 1.0)
        if len(ph) < 2:
            return len(ph)
        ext = np.concatenate([ph, ph + 1.0])
        j = np.searchsorted(ext, ext[:len(ph)] + w, side="right")
        return int(np.max(j - np.arange(len(ph))))
    n0 = best(times)
    null = [best(rng.choice(good_t, size=len(times), replace=False)) for _ in range(ntrial)]
    return n0, (1 + sum(x >= n0 for x in null)) / (ntrial + 1.0)


def kepler_one_star(t, slc_dir=None):
    quiet_warnings()
    from astropy.io import fits
    fl, goods = [], []
    for k in t["files"]:
        p = os.path.join(slc_dir or os.path.join(kepler_ROOT, "slc"), os.path.basename(k))
        if not os.path.exists(p):
            continue
        try:
            with fits.open(p) as h:
                d = h[1].data
                f, g = kepler_flashes(np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float),
                               np.asarray(d["SAP_QUALITY"] if "SAP_QUALITY" in d.columns.names else d["QUALITY"], int))
        except Exception:
            continue
        fl += f
        goods.append(g)
    if not goods:
        return None
    good_t = np.concatenate(goods)
    rng = np.random.default_rng(int(t["kic"]) % 100000)
    res = []
    times = [x[0] for x in fl]
    for k in t["kois"]:
        if len(times) >= 3 and k["period"] > 0:
            n0, fap = kepler_phase_cluster(times, good_t, k["period"], k["t0"], rng)
        else:
            n0, fap = len(times), 1.0
        res.append(dict(k, n_cluster=n0, fap=fap))
    return dict(kic=t["kic"], days=float(len(good_t) * kepler_CAD), n_flash=len(fl), flashes=fl[:200], kois=res)


def kepler_summary(res):
    """Counts against the null: KOIs on stars with three or more flashes; p floor 1/401."""
    allk = [k for r in res for k in r["kois"]]
    tested = [k for r in res for k in r["kois"] if r["n_flash"] >= 3 and k["period"] > 0]
    print("%d stars, %.0f days of short cadence, %d flashes; %d KOIs folded, %d on stars with three or more flashes" % (
        len(res), sum(r["days"] for r in res), sum(r["n_flash"] for r in res), len(allk), len(tested)))
    for thr in (0.01, 1 / 401.0 + 1e-9):
        print("  KOIs with flashes clustered in phase at FAP %s %.4f: %d (at most %.1f expected)"
              % ("below" if thr == 0.01 else "at the floor", thr, sum(1 for k in tested if k["fap"] < thr or
                                                                    (thr < 0.01 and k["fap"] <= thr)), thr * len(tested)))


def kepler_search(share=None, out="kepler_glints.json", procs=2):
    from concurrent.futures import ProcessPoolExecutor
    ts = json.load(open(share or os.path.join(kepler_ROOT, "targets.json")))
    with ProcessPoolExecutor(procs) as ex:
        res = [r for r in ex.map(kepler_one_star, ts, chunksize=2) if r]
    kepler_summary(res)
    for r in res:
        for k in r["kois"]:
            if k["fap"] < 0.01:
                print("  KIC %s %s P %.4f d: %d of %d flashes within 2.5 cadences of one phase, FAP %.4f"
                      % (r["kic"], k["koi"], k["period"], k["n_cluster"], r["n_flash"], k["fap"]))
    json.dump(res, open(os.path.join(kepler_HERE, out), "w"))


def kepler_stream_star(t):
    """Fetch, search and delete one star's short-cadence files; returns its JSONL line."""
    import shutil
    tmp = os.path.join(kepler_ROOT, "tmp_%s" % t["kic"])
    os.makedirs(tmp, exist_ok=True)
    local = []
    for k in t["files"]:
        p = os.path.join(tmp, os.path.basename(k))
        if not os.path.exists(p):
            b = core.get(kepler_S3 + k, timeout=300, retries=3, soft=True, pause=2)
            if not b or not b.startswith(b"SIMPLE"):
                continue
            open(p, "wb").write(b)
        local.append(p)
    r = kepler_one_star(dict(t, files=local), slc_dir=tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    return r


def kepler_stream(share, out, procs):
    """Stream every star of the list into out (JSONL), skipping stars already there."""
    from concurrent.futures import ProcessPoolExecutor
    ts = json.load(open(share or os.path.join(kepler_ROOT, "targets.json")))
    path = os.path.join(kepler_HERE, out)
    done = set()
    if os.path.exists(path):
        done = {json.loads(ln)["kic"] for ln in open(path) if ln.strip()}
    todo = [t for t in ts if t["kic"] not in done]
    print("%d stars to search (%d done)" % (len(todo), len(done)), flush=True)
    with ProcessPoolExecutor(procs) as ex, open(path, "a") as fh:
        for r in ex.map(kepler_stream_star, todo):
            if r:
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()


def kepler_report(paths):
    # a star counts once when two runs overlap
    seen, res = set(), []
    for p in paths:
        for ln in open(p):
            if ln.strip():
                r = json.loads(ln)
                if r["kic"] not in seen:
                    seen.add(r["kic"])
                    res.append(r)
    kepler_summary(res)
    for r in res:
        for k in r["kois"]:
            if k["fap"] < 0.01:
                print("  KIC %s %s P %.4f d: %d of %d flashes within 2.5 cadences of one phase, FAP %.4f"
                      % (r["kic"], k["koi"], k["period"], k["n_cluster"], r["n_flash"], k["fap"]))
    return res


def kepler_record(paths):
    """Record one row per star with its lowest KOI false-alarm probability."""
    res = kepler_report(paths)
    tested = [k for r in res for k in r["kois"] if r["n_flash"] >= 3 and k["period"] > 0]
    below = sum(1 for k in tested if k["fap"] < 0.01)
    floor = sum(1 for k in tested if k["fap"] <= 1 / 401.0 + 1e-9)
    core.record_search(
        "kepler_sc_glints", "Specular glints at KOI periods, Kepler short cadence",
        archive="MAST Kepler short-cadence light curves (STScI AWS mirror)", release="DR25 light curves",
        selection="%d hosts of confirmed and candidate KOIs with short cadence" % len(res), code="seti.py flashes kepler",
        params=dict(flash_sigma=5, max_cadences=3, clean_window=4, phase_window_cadences=2.5, null_draws=400),
        n_targets=len(res),
        null_summary="%d KOIs on stars with three or more flashes: %d below FAP 0.01 against at most %.1f, %d at the "
                     "1/401 floor against at most %.1f" % (len(tested), below, 0.01 * len(tested), floor,
                                                           len(tested) / 401.0),
        sensitivity="flashes of 5 robust deviations lasting 1-3 cadences (59-177 s); %.0f star-days, %d flashes"
                    % (sum(r["days"] for r in res), sum(r["n_flash"] for r in res)),
        result="no KOI period gathers flashes at one phase beyond chance",
        ledger="Specular glints, Kepler short cadence")
    core.clear_targets("kepler_sc_glints")
    core.record_targets("kepler_sc_glints", [
        dict(target="kic:%s" % r["kic"], stat=float(r["n_flash"]), unit="flashes",
             p=min((k["fap"] for k in r["kois"]), default=None),
             note="%.0f days; lowest KOI false-alarm probability over its folded periods" % r["days"]) for r in res])


def kepler_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("targets", "fetch", "search", "stream", "split", "report", "record"))
    p.add_argument("paths", nargs="*")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--list")
    p.add_argument("--out", default="kepler_glints.json")
    p.add_argument("--procs", type=int, default=2)
    a = p.parse_args()
    if a.mode == "targets":
        kepler_targets()
    elif a.mode == "fetch":
        kepler_fetch(a.threads, a.list)
    elif a.mode == "stream":
        kepler_stream(a.list, a.out, a.procs)
    elif a.mode == "split":
        # alternate stars into two shares by short-cadence volume
        ts = sorted(json.load(open(os.path.join(kepler_ROOT, "targets.json"))), key=lambda t: -len(t["files"]))
        for i in range(2):
            json.dump(ts[i::2], open(os.path.join(kepler_ROOT, "share%d.json" % i), "w"))
        print("two shares of %d and %d stars" % (len(ts[0::2]), len(ts[1::2])))
    elif a.mode == "report":
        kepler_report(a.paths)
    elif a.mode == "record":
        kepler_record(a.paths)
    else:
        kepler_search(a.list, a.out, a.procs)


def kepler_cli():
    quiet_warnings()
    kepler_main()


# --------------------------------------------------------------------------------------------------------------

asteroids_DOC = r"""Asteroid glints in ALCDEF: spikes above the rotational model recurring in phase, against draws and fades as null; injection recovery.

    python -I seti.py flashes asteroids parse
    python -I seti.py flashes asteroids inject [--groups 300]
    python -I seti.py flashes asteroids search [--procs 2] [--out alcdef_glints.json]"""

asteroids_ROOT = os.environ.get("ALCDEF_ROOT", core.data("alcdef"))
asteroids_LCDB = os.environ.get("LCDB", core.data("lcdb", "lc_summary_pub.txt"))
asteroids_HERE = core.RESULTS
# fractional period uncertainty per LCDB quality code
asteroids_FRAC = {"3": 1e-4, "3-": 3e-4, "2+": 1e-3, "2": 2e-3}


def asteroids_data_rows(path):
    """Row index of the first data line after '# Data' in a PDS4 CSV."""
    with open(path, encoding="latin-1") as f:
        for i, line in enumerate(f):
            if line.startswith("# Data"):
                return i + 1
    raise ValueError(path)


def asteroids_parse():
    import pandas as pd
    base = os.path.join(asteroids_ROOT, "gbo.ast.alcdef-database_V1_0", "data")
    pts = []
    for p in sorted(glob.glob(os.path.join(base, "*", "alcdef_lcdata-*.csv"))):
        d = pd.read_csv(p, skiprows=asteroids_data_rows(p), header=None, names=["num", "name", "mdid", "jd", "mag", "err"],
                        dtype={"num": str, "name": str}, encoding="latin-1", low_memory=False)
        pts.append(d[["mdid", "jd", "mag", "err"]])
    d = pd.concat(pts, ignore_index=True)
    d = d.sort_values(["mdid", "jd"], kind="stable")
    np.savez(os.path.join(asteroids_ROOT, "alcdef.npz"), mdid=d["mdid"].to_numpy(np.int64), jd=d["jd"].to_numpy(np.float64),
             mag=d["mag"].to_numpy(np.float32), err=d["err"].to_numpy(np.float32))
    cols = ["ID", "Submitter", "ObjNumber", "ObjName", "Desig", "Facility", "MPCCode", "Instrument", "Detector",
            "ObsLong", "ObsLat", "SessionDateTime", "ObjRA", "ObjDec", "Epoch", "Phase", "PABL", "PABB", "Filter",
            "MagBand", "CIBand", "CIObject", "CICorr", "DiffMags", "DiffZeroMag", "MagStd", "ExpJD", "Exposure"]
    meta = {}
    for p in sorted(glob.glob(os.path.join(base, "*", "alcdef_metadata-*.csv"))):
        m = pd.read_csv(p, skiprows=asteroids_data_rows(p), header=None, dtype=str, encoding="latin-1", low_memory=False)
        m = m.iloc[:, :len(cols)]
        m.columns = cols
        for r in m.itertuples(index=False):
            meta[int(r.ID)] = dict(num=r.ObjNumber, name=r.ObjName, desig=r.Desig, sub=r.Submitter, fac=r.Facility,
                                   tel=r.Instrument, when=r.SessionDateTime, phase=r.Phase, filt=r.Filter,
                                   band=r.MagBand, exp=r.Exposure, mpc=r.MPCCode)
    json.dump(meta, open(os.path.join(asteroids_ROOT, "sessions.json"), "w"))
    print("%d points in %d sessions (%d metadata records)" % (len(d), d["mdid"].nunique(), len(meta)))


def asteroids_lcdb():
    """LCDB number or name -> (period h, U code, diameter km)."""
    out = {}
    with open(asteroids_LCDB, encoding="latin-1") as f:
        lines = f.read().splitlines()
    for ln in lines[5:]:
        if len(ln) < 190:
            continue
        per, u = ln[145:158].strip(), ln[187:190].strip()
        if not per or u not in asteroids_FRAC:
            continue
        try:
            per = float(per)
        except ValueError:
            continue
        try:
            dia = float(ln[88:96])
        except ValueError:
            dia = float("nan")
        num, name = ln[0:10].strip(), ln[10:41].strip()
        rec = (per, u, dia)
        if num and num != "0":
            out["n" + num.lstrip("0")] = rec
        if name:
            out["s" + name.lower()] = rec
    return out


def asteroids_running_median(t, y, w):
    lo = np.searchsorted(t, t - w / 2)
    hi = np.searchsorted(t, t + w / 2, side="right")
    return np.array([np.median(y[a:b]) for a, b in zip(lo, hi)])


def asteroids_fourier_model(t, y, P, order):
    """Robust Fourier fit in rotational phase plus a linear term, or None."""
    ph = 2 * np.pi * (t - t[0]) / P
    cols = [np.ones_like(t), (t - t.mean())]
    for k in range(1, order + 1):
        cols += [np.cos(k * ph), np.sin(k * ph)]
    A = np.array(cols).T
    keep = np.ones(len(t), bool)
    for _ in range(5):
        if keep.sum() < A.shape[1] + 5:
            return None
        c, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        r = y - A @ c
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) < 3 * max(s, 1e-4)
        if (new == keep).all():
            break
        keep = new
    return A @ c


def asteroids_events(t, flag, gap):
    """Mean times of merged runs of flagged points."""
    idx = np.nonzero(flag)[0]
    out = []
    run = []
    for i in idx:
        if run and t[i] - t[run[-1]] > gap:
            out.append(float(np.mean(t[run])))
            run = []
        run.append(i)
    if run:
        out.append(float(np.mean(t[run])))
    return out


def asteroids_session_events(t, y, e, P_d):
    """Spike and fade times of one session, its spacing and robust scatter."""
    dt = float(np.median(np.diff(t)))
    span = t[-1] - t[0]
    model = None
    if span >= 0.8 * P_d:
        order = int(min(6, max(2, len(t) // 8)))
        model = asteroids_fourier_model(t, y, P_d, order)
    if model is None:
        w = float(np.clip(P_d / 10, 5 * dt, 1 / 24))
        model = asteroids_running_median(t, y, w)
    r = y - model
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    thr = np.maximum(np.maximum(5 * s, 3 * np.where(e > 0, e, 0)), 0.05)
    spikes = asteroids_events(t, r < -thr, 2.5 * dt)
    fades = asteroids_events(t, r > thr, 2.5 * dt)
    return spikes, fades, dt, float(s)


def asteroids_max_cluster(times, P, tol):
    if len(times) < 2:
        return len(times)
    ph = np.mod(np.asarray(times) / P, 1.0)
    d = np.abs(ph[:, None] - ph[None, :])
    d = np.minimum(d, 1 - d)
    return int((d < tol / P).sum(axis=1).max())


def asteroids_group_test(args):
    """One body's session group: largest phase cluster of spikes and fades against random draws."""
    key, P_d, frac, sess, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    allt = np.concatenate([s["t"] for s in sess])
    dt = float(np.median([s["dt"] for s in sess]))
    tol = max(0.75 * dt, 3 / 1440)
    out = dict(key=key, period_h=P_d * 24, n_sessions=len(sess), n_points=int(len(allt)), tol_min=tol * 1440,
               mdids=[s["mdid"] for s in sess])
    for kind in ("spikes", "fades"):
        ev = [x for s in sess for x in s[kind]]
        out["n_" + kind] = len(ev)
        best = None
        for mult in (1.0, 0.5):
            Pt = P_d * mult
            m = asteroids_max_cluster(ev, Pt, tol)
            if m < 3:
                continue
            draws = np.array([asteroids_max_cluster(rng.choice(allt, len(ev), replace=False), Pt, tol) for _ in range(2000)])
            p = float((np.sum(draws >= m) + 1) / (len(draws) + 1))
            if best is None or p < best[2]:
                best = (mult, m, p)
        if best:
            out[kind] = dict(mult=best[0], cluster=best[1], p=best[2])
    return out


def asteroids_load_sessions(keep_flux=False):
    """Sessions passing the cuts, grouped by body."""
    z = np.load(os.path.join(asteroids_ROOT, "alcdef.npz"))
    meta = json.load(open(os.path.join(asteroids_ROOT, "sessions.json")))
    per = asteroids_lcdb()
    mdid, jd, mag, err = z["mdid"], z["jd"], z["mag"], z["err"]
    cut = np.nonzero(np.diff(mdid))[0] + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(mdid)]])
    bodies = {}
    nsess = 0
    seen_t = {}
    ndup = 0
    for a, b in zip(starts, ends):
        if b - a < 20:
            continue
        m = meta.get(str(int(mdid[a])))
        if not m:
            continue
        num = str(m["num"]).strip().lstrip("0")
        rec = per.get("n" + num) if num and num not in ("-9", "-", "nan") else None
        if rec is None:
            rec = per.get("s" + str(m["name"]).strip().lower())
        if rec is None:
            continue
        t = jd[a:b]
        y = mag[a:b].astype(float)
        e = err[a:b].astype(float)
        ok = np.isfinite(t) & np.isfinite(y) & (y > -90) & (y < 90)
        t, y, e = t[ok], y[ok], e[ok]
        if len(t) < 20:
            continue
        o = np.argsort(t)
        t, y, e = t[o], y[o], e[o]
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, y, e = t[keep], y[keep], e[keep]
        if len(t) < 20 or np.median(np.diff(t)) > 15 / 1440:
            continue
        # ALCDEF repeats sessions under new IDs; mostly shared observation times mark copies
        key = "n" + num if num and num not in ("-9", "-", "nan") else "s" + str(m["name"]).strip().lower()
        stamps = set(np.round(t * 86400 / 2).astype(np.int64).tolist())
        held = seen_t.setdefault(key, set())
        if len(stamps & held) > 0.5 * len(stamps):
            ndup += 1
            continue
        held |= stamps
        P_d = rec[0] / 24.0
        spikes, fades, dt, s = asteroids_session_events(t, y, e, P_d)
        nsess += 1
        rec_s = dict(mdid=int(mdid[a]), t=t, dt=dt, spikes=spikes, fades=fades, scatter=s)
        if keep_flux:
            rec_s.update(y=y, e=e)
        bodies.setdefault(key, dict(P=P_d, frac=asteroids_FRAC[rec[1]], dia=rec[2], name=str(m["name"]).strip(),
                                    sess=[]))["sess"].append(rec_s)
    print("%d sessions on %d bodies with LCDB periods (%d duplicate sessions dropped)" % (nsess, len(bodies), ndup))
    return bodies


def asteroids_make_groups(bodies):
    """Session groups at most 10 d long within the period's phase tolerance."""
    groups = []
    seed = 0
    for key, bd in bodies.items():
        ss = sorted(bd["sess"], key=lambda s: s["t"][0])
        group = []
        for s in ss:
            if group:
                span = s["t"][-1] - group[0]["t"][0]
                tol = max(0.75 * np.median([g["dt"] for g in group + [s]]), 3 / 1440)
                if span > 10 or span * bd["frac"] > tol:
                    groups.append((key, bd["P"], bd["frac"], group, seed))
                    seed += 1
                    group = []
            group.append(s)
        if group:
            groups.append((key, bd["P"], bd["frac"], group, seed))
            seed += 1
    return groups


def asteroids_target_of(key, bd):
    return "mpc:" + (key[1:] if key.startswith("n") else bd["name"])


def asteroids_search(procs, outname, sensitivity=None):
    bodies = asteroids_load_sessions()
    groups = asteroids_make_groups(bodies)
    nspk = sum(len(s["spikes"]) for bd in bodies.values() for s in bd["sess"])
    nfad = sum(len(s["fades"]) for bd in bodies.values() for s in bd["sess"])
    tasks = [x for x in groups if sum(len(s["spikes"]) for s in x[3]) >= 3 or sum(len(s["fades"]) for s in x[3]) >= 3]
    print("%d spike events and %d fade events; %d session groups hold three or more of either" % (nspk, nfad, len(tasks)))
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pool:
            res = pool.map(asteroids_group_test, tasks, chunksize=4)
    else:
        res = [asteroids_group_test(x) for x in tasks]
    json.dump(res, open(os.path.join(asteroids_HERE, outname), "w"))
    summ = {}
    for kind in ("spikes", "fades"):
        ps = [r[kind]["p"] for r in res if kind in r]
        summ[kind] = dict(clusters=len(ps), p01=sum(p < 0.01 for p in ps), p001=sum(p < 0.001 for p in ps))
        print("%s: %d groups with a phase cluster of three or more; %d at p < 0.01, %d at p < 0.001"
              % (kind, len(ps), summ[kind]["p01"], summ[kind]["p001"]))
    top = sorted([r for r in res if "spikes" in r], key=lambda r: (r["spikes"]["p"], -r["spikes"]["cluster"]))[:15]
    for r in top:
        print("  %s P %.3f h: %d of %d spikes within %.1f min of one phase (x%.1f period), p %.4f; fades %s; %d sessions"
              % (r["key"], r["period_h"], r["spikes"]["cluster"], r["n_spikes"], r["tol_min"], r["spikes"]["mult"],
                 r["spikes"]["p"], r.get("fades", {}).get("p", "-"), r["n_sessions"]))
    asteroids_record(bodies, res, summ, sensitivity)


def asteroids_record(bodies, res, summ, sensitivity=None):
    """Record the run with one row per body."""
    best = {}
    for r in res:
        if "spikes" in r:
            best[r["key"]] = min(best.get(r["key"], 1.0), r["spikes"]["p"])
    rows = []
    for key, bd in bodies.items():
        n = sum(len(s["spikes"]) for s in bd["sess"])
        rows.append(dict(target=asteroids_target_of(key, bd), stat=n, p=best.get(key),
                         note="%d sessions, %d spike events, period %.4f h" % (len(bd["sess"]), n, bd["P"] * 24)))
    sid = "alcdef_glints_" + time.strftime("%Y%m%d")
    db = core.connect()
    core.record_search(sid, "Specular glints on asteroids", archive="ALCDEF (PDS SBN gbo.ast.alcdef-database)",
                           release="V1.0, 2021 snapshot; periods LCDB October 2023",
                           selection="sessions of 20 or more points at 15 min or less spacing on bodies with an LCDB "
                                     "period of U 2 or better; duplicate sessions dropped",
                           code=os.path.abspath(__file__), entry="asteroids",
                           params=dict(spike_sigma=5, spike_err=3, spike_mag=0.05, tol="max(0.75 spacing, 3 min)",
                                       group_days=10, draws=2000, folds=[1.0, 0.5]),
                           n_targets=len(rows), null_summary=summ, sensitivity=sensitivity,
                           result="no body with spikes recurring at its period at p < 0.01; fades give %d groups"
                                  % summ["fades"]["p01"],
                           ledger="Specular glints on asteroids", db=db)
    core.record_targets(sid, rows, db=db)
    db.close()
    print("recorded %d bodies under %s" % (len(rows), sid))


def asteroids_inject(n_groups, amps):
    """Recovery of a glint brightening one rotational phase every rotation."""
    bodies = asteroids_load_sessions(keep_flux=True)
    groups = asteroids_make_groups(bodies)
    rng = np.random.default_rng(7)
    ok = [g for g in groups if sum(s["t"][-1] - s["t"][0] for s in g[3]) >= 3 * g[1]]
    pick = [ok[i] for i in rng.choice(len(ok), min(n_groups, len(ok)), replace=False)]
    print("%d groups cover three or more rotations; injecting into %d" % (len(ok), len(pick)))
    out = {}
    for A in amps:
        hit = 0
        for key, P_d, frac, sess, seed in pick:
            phi0 = rng.random()
            new = []
            for s in sess:
                t, y, e = s["t"], s["y"].copy(), s["e"]
                k = np.arange(np.floor(t[0] / P_d - phi0) - 1, np.ceil(t[-1] / P_d - phi0) + 2)
                for tg in (k + phi0) * P_d:
                    j = int(np.argmin(np.abs(t - tg)))
                    if abs(t[j] - tg) < s["dt"] / 2:
                        y[j] -= A
                spikes, fades, dt, sc = asteroids_session_events(t, y, e, P_d)
                new.append(dict(mdid=s["mdid"], t=t, dt=dt, spikes=spikes, fades=fades, scatter=sc))
            r = asteroids_group_test((key, P_d, frac, new, seed))
            hit += "spikes" in r and r["spikes"]["p"] < 0.01
        out[A] = hit / len(pick)
        print("  glint of %.2f mag in one exposure per rotation: recovered at p < 0.01 in %.0f per cent"
              % (A, 100 * out[A]))
    json.dump(out, open(os.path.join(asteroids_HERE, "alcdef_inject.json"), "w"))
    return out


def asteroids_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("parse", "search", "inject"))
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--out", default="alcdef_glints.json")
    p.add_argument("--groups", type=int, default=300)
    a = p.parse_args()
    if a.mode == "parse":
        asteroids_parse()
    elif a.mode == "search":
        inj = os.path.join(asteroids_HERE, "alcdef_inject.json")
        asteroids_search(a.procs, a.out, json.load(open(inj)) if os.path.exists(inj) else None)
    else:
        asteroids_inject(a.groups, (0.05, 0.1, 0.2, 0.5))


def asteroids_cli():
    quiet_warnings()
    asteroids_main()


# --------------------------------------------------------------------------------------------------------------

tophat_DOC = r"""Top-hat brightenings in TESS 20 s light curves within 30 pc: sharp-edged boxes beating a flare template by BIC 10; common-mode and dip nulls.

    python -I seti.py flashes tophat [--procs 4]
    python -I seti.py flashes tophat record"""


def tophat_flare(t, t0, fwhm, amp):
    """Davenport et al. (2014) flare template in FWHM units."""
    x = (t - t0) / fwhm
    out = np.zeros_like(x)
    r = (x > -1) & (x <= 0)
    out[r] = 1 + 1.941 * x[r] - 0.175 * x[r] ** 2 - 2.246 * x[r] ** 3 - 1.125 * x[r] ** 4
    d = x > 0
    out[d] = 0.6890 * np.exp(-1.600 * x[d]) + 0.3030 * np.exp(-0.2783 * x[d])
    return amp * out


def tophat_one(p):
    from astropy.io import fits
    from scipy.ndimage import median_filter
    from scipy.optimize import least_squares
    try:
        with fits.open(os.path.join(tess20_DL, p)) as h:
            d = h[1].data
            t, f, q = (np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"], int))
            sector = int(h[0].header.get("SECTOR", 0))
    except Exception:
        return None
    good = (q == 0) & np.isfinite(f) & np.isfinite(t)
    t, f = t[good], f[good]
    if len(f) < 5000:
        return None
    base = median_filter(f, size=1081, mode="nearest")  # six hours of 20 s cadences
    r = f / base - 1.0
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    events = []
    cs = np.concatenate([[0.0], np.cumsum(r)])
    for w in (30, 60, 90, 135, 180):  # 10 to 60 min
        box = (cs[w:] - cs[:-w]) / w
        snr = box / (sig / math.sqrt(w))
        for sign in (1, -1):
            v = sign * snr
            idx = np.nonzero((v >= 7) & (v >= np.roll(v, 1)) & (v > np.roll(v, -1)))[0]
            for i in idx:
                tc = t[i + w // 2]
                if any(abs(e["t"] - tc) < 0.03 and e["sign"] == sign for e in events):
                    continue
                events.append(dict(t=float(tc), w=w, snr=float(v[i]), sign=sign, i0=int(i)))
    out = []
    for e in sorted(events, key=lambda e: -e["snr"])[:40]:
        a, b = max(e["i0"] - 3 * e["w"], 0), min(e["i0"] + 4 * e["w"], len(t))
        tt, rr = t[a:b], e["sign"] * r[a:b]
        if len(tt) < 2 * e["w"]:
            continue
        def box_model(pp):
            t1, t2, amp, c0 = pp
            return np.where((tt >= t1) & (tt <= t2), amp, 0.0) + c0
        t1g, t2g = t[e["i0"]], t[min(e["i0"] + e["w"], len(t) - 1)]
        best_box = None
        for d1 in np.linspace(-0.5, 0.5, 11) * e["w"] * 20 / 86400:
            for d2 in np.linspace(-0.5, 0.5, 11) * e["w"] * 20 / 86400:
                pp = [t1g + d1, t2g + d2, 0.0, 0.0]
                inside = (tt >= pp[0]) & (tt <= pp[1])
                if inside.sum() < 5:
                    continue
                pp[3] = float(np.median(rr[~inside])) if (~inside).sum() > 5 else 0.0
                pp[2] = float(np.mean(rr[inside]) - pp[3])
                chi = float(np.sum((rr - box_model(pp)) ** 2) / sig ** 2)
                if best_box is None or chi < best_box[0]:
                    best_box = (chi, pp)
        k = int(np.argmax(rr))
        res = least_squares(lambda pp: (tophat_flare(tt, pp[0], abs(pp[1]) + 1e-4, pp[2]) + pp[3] - rr) / sig,
                            [tt[k], e["w"] * 20 / 86400 / 3, float(rr[k]), 0.0], max_nfev=400)
        chi_f = float(np.sum(res.fun ** 2))
        n = len(tt)
        bic_box = best_box[0] + 4 * math.log(n)
        bic_flare = chi_f + 4 * math.log(n)
        # edge sharpness: cadences from 10 to 90 per cent of amplitude
        pp = best_box[1]
        def edge(t_edge, rising):
            j = int(np.searchsorted(tt, t_edge))
            seg = rr[max(j - 6, 0):j + 6]
            if len(seg) < 6 or pp[2] <= 0:
                return 99
            lo_, hi_ = pp[3] + 0.1 * pp[2], pp[3] + 0.9 * pp[2]
            between = np.sum((seg > lo_) & (seg < hi_))
            return int(between)
        out.append(dict(t=e["t"], width_min=e["w"] / 3.0, snr=e["snr"], sign=e["sign"], amp=float(pp[2]),
                        bic_box=bic_box, bic_flare=bic_flare, edges=(edge(pp[0], True), edge(pp[1], False)),
                        dur_min=float((pp[1] - pp[0]) * 1440)))
    return dict(file=p, sector=sector, sig=float(sig), events=out)


def tophat_main(procs):
    files = sorted(os.listdir(tess20_DL))
    files = [x for x in files if x.endswith("_fast-lc.fits")]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(procs) as ex:
        R = [r for r in ex.map(tophat_one, files, chunksize=8) if r]
    # common mode: events within 30 min on two or more other stars of the sector
    by = {}
    for r in R:
        for e in r["events"]:
            by.setdefault(r["sector"], []).append((e["t"], r["file"]))
    for r in R:
        for e in r["events"]:
            others = {fn for tt, fn in by.get(r["sector"], []) if abs(tt - e["t"]) < 30 / 1440 and fn != r["file"]}
            e["common"] = len(others)
    def tophat(e):
        return (e["bic_flare"] - e["bic_box"] > 10 and max(e["edges"]) <= 2 and e["common"] < 2)
    pos = [(r, e) for r in R for e in r["events"] if e["sign"] > 0]
    neg = [(r, e) for r in R for e in r["events"] if e["sign"] < 0]
    tp, tn = [x for x in pos if tophat(x[1])], [x for x in neg if tophat(x[1])]
    print("%d light curves; brightenings %d, dips %d at 7 sigma; top-hat brightenings %d, top-hat dips %d"
          % (len(R), len(pos), len(neg), len(tp), len(tn)))
    for r, e in sorted(tp, key=lambda x: -x[1]["snr"])[:25]:
        print("  %s t %.4f: %.0f min, SNR %.1f, amplitude %.4f, edges %s cadences, BIC box %.0f flare %.0f"
              % (r["file"], e["t"], e["dur_min"], e["snr"], e["amp"], e["edges"], e["bic_box"], e["bic_flare"]))
    json.dump(dict(n=len(R), pos=len(pos), neg=len(neg), tophat_pos=[dict(file=r["file"], **e) for r, e in tp],
                   tophat_neg=[dict(file=r["file"], **e) for r, e in tn]), open(os.path.join(tess20_HERE, "tophat_events.json"), "w"),
              indent=0)


def tophat_record():
    d = json.load(open(os.path.join(tess20_HERE, "tophat_events.json")))
    tg = {}
    for t in json.load(open(os.path.join(tess20_HERE, "glint_targets.json"))):
        tg[(int(t["sector"]), int(t["tic"]))] = t["gaia"]
    files = [x for x in os.listdir(tess20_DL) if x.endswith("_fast-lc.fits")]
    stars = {}
    for fn in files:
        m = re.match(r"s(\d+)_(\d+)_fast-lc\.fits", fn)
        g = tg.get((int(m.group(1)), int(m.group(2)))) if m else None
        if g:
            stars.setdefault(g, []).append(fn)
    best = {}
    for e in d["tophat_pos"]:
        m = re.match(r"s(\d+)_(\d+)_fast-lc\.fits", e["file"])
        g = tg.get((int(m.group(1)), int(m.group(2))))
        if g and (g not in best or e["snr"] > best[g]["snr"]):
            best[g] = e
    core.record_search(
        "tess20_tophat", "Top-hat flares", archive="MAST TESS SPOC 20 s light curves", selection="stars within 30 pc",
        code=os.path.abspath(__file__), entry="tophat", n_targets=len(stars), params=dict(snr=7, bic=10, max_edge_cadences=2),
        null_summary="dips of the same shape: %d" % len(d["tophat_neg"]),
        sensitivity="10-60 minute brightenings at 7 sigma of the binned noise",
        ledger="Top-hat flares",
        result="%d top-hat brightenings against %d top-hat dips among %d brightenings and %d dips at 7 sigma: the rate of "
               "instrumental steps" % (len(d["tophat_pos"]), len(d["tophat_neg"]), d["pos"], d["neg"]))
    core.clear_targets("tess20_tophat")
    core.record_targets("tess20_tophat", [dict(target="gaia_dr3:%s" % g, stat=best[g]["snr"] if g in best else None,
                                                   unit="S/N of the strongest top-hat brightening",
                                                   note=("%s t %.4f, %.0f min" % (best[g]["file"], best[g]["t"], best[g]["dur_min"]))
                                                   if g in best else None) for g in stars])
    print("recorded %d stars" % len(stars))


def tophat_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        tophat_record()
        return
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    tophat_main(a.procs)


clockwork_DOC = r"""Clockwork flares: de Jager H-test of Kepler (and TESS) flare times over 0.5-300 d against quarter-shuffled and pooled nulls, vetted for rotation, binaries and spacecraft timing.

    python -I seti.py flashes clockwork [search] [--procs 1]
    python -I seti.py flashes clockwork null [--procs 4]
    python -I seti.py flashes clockwork vet
    python -I seti.py flashes clockwork record"""

clockwork_MMAX = 20
clockwork_CAT = os.environ.get("CLOCKWORK_CAT", "kepler")
clockwork_FMAX = 5.0 if clockwork_CAT == "tess" else 2.0
clockwork_LABEL = "TIC" if clockwork_CAT == "tess" else "KIC"
clockwork_OUT = "clockwork_flares_tess.json" if clockwork_CAT == "tess" else "clockwork_flares.json"
clockwork_CADENCE = 2.0 / 1440 if clockwork_CAT == "tess" else 0.0204


def clockwork_h(t, freqs):
    """de Jager's H per frequency: max over m of Z2_m - 4m + 4, and its m."""
    n = len(t)
    best_h = np.full(len(freqs), -np.inf)
    best_m = np.zeros(len(freqs), int)
    for fb in range(0, len(freqs), 2000):
        fr = freqs[fb:fb + 2000]
        ph = 2 * np.pi * np.outer(fr, t)
        z = np.zeros(len(fr))
        bh = np.full(len(fr), -np.inf)
        bm = np.zeros(len(fr), int)
        for m in range(1, clockwork_MMAX + 1):
            z += (np.cos(m * ph).sum(axis=1) ** 2 + np.sin(m * ph).sum(axis=1) ** 2) * 2.0 / n
            h = z - 4 * m + 4
            up = h > bh
            bh[up] = h[up]
            bm[up] = m
        best_h[fb:fb + 2000] = bh
        best_m[fb:fb + 2000] = bm
    return best_h, best_m


def clockwork_data():
    if clockwork_CAT == "tess":
        p = core.data("tess_flares_yang2023.json")
        if not os.path.exists(p):
            from astroquery.vizier import Vizier
            t = Vizier(row_limit=-1, columns=["TIC", "TimeS"], timeout=900).get_catalogs("J/A+A/669/A15/table1")[0]
            json.dump([[int(r["TIC"]), float(r["TimeS"])] for r in t], open(p, "w"))
        by = {}
        for tic, ts in json.load(open(p)):
            by.setdefault(tic, []).append((ts, int((ts - 1325.29) // 13.7)))
        return by, {}
    p = core.data("kepler_flares_yang2019.json")
    if not os.path.exists(p):
        from astroquery.vizier import Vizier
        v = Vizier(row_limit=-1, columns=["**"], timeout=900)
        t2 = v.get_catalogs("J/ApJS/241/29/table2")[0]
        t1 = v.get_catalogs("J/ApJS/241/29/table1")[0]
        fl = [dict(kic=int(r["KIC"]), q=int(re.sub(r"\D", "", str(r["Q"])) or -1), begin=float(r["Begin"]), end=float(r["End"]))
              for r in t2]
        st = [dict(kic=int(r["KIC"]), prot=num(r["Prot"])) for r in t1]
        json.dump(dict(flares=fl, stars=st), open(p, "w"))
    d = json.load(open(p))
    by = {}
    for f in d["flares"]:
        by.setdefault(f["kic"], []).append((f["begin"], f["q"]))
    return by, {s["kic"]: s.get("prot") for s in d["stars"]}


def clockwork_star(args):
    kic, ev, seed = args
    ev = sorted(ev)
    t = np.array([e[0] for e in ev])
    if seed is not None:
        rng = np.random.default_rng(seed)
        q = np.array([e[1] for e in ev])
        for qq in set(q.tolist()):
            sel = q == qq
            t[sel] = rng.uniform(t[sel].min(), t[sel].max(), sel.sum())
    span = t.max() - t.min()
    freqs = np.arange(max(1 / 300.0, 3.0 / span), clockwork_FMAX, 1.0 / (4 * span))
    h, m = clockwork_h(t - t.min(), freqs)
    i = int(np.argmax(h))
    return dict(kic=kic, n=len(t), span=float(span), h=float(h[i]), period=float(1 / freqs[i]), m=int(m[i]), n_freq=len(freqs))


def clockwork_main(procs):
    by, prot = clockwork_data()
    jobs = [(k, v, None) for k, v in by.items() if len(v) >= 10]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(procs) as ex:
        res = list(ex.map(clockwork_star, jobs, chunksize=8))
    res.sort(key=lambda r: -r["h"])
    top = res[:40]
    with ProcessPoolExecutor(procs) as ex:
        sh = list(ex.map(clockwork_star, [(r["kic"], by[r["kic"]], s) for r in top for s in range(10)]))
    shmax = {}
    for r in sh:
        shmax.setdefault(r["kic"], []).append(r["h"])
    for r in res:
        r["prot"] = prot.get(r["kic"])
        rot = r["prot"]
        r["rotation"] = bool(rot) and any(abs(r["period"] * k / rot - 1) < 0.05 or abs(r["period"] / (rot * k) - 1) < 0.05
                                          for k in (1, 2, 3))
        if r["kic"] in shmax:
            r["shuffled"] = sorted(shmax[r["kic"]])
            r["p_shuffled"] = float(np.mean(np.array(shmax[r["kic"]]) >= r["h"]))
    print("%d stars with 10 or more flares" % len(res))
    for r in top[:25]:
        print("  " + clockwork_LABEL + " %d: %d flares over %.0f d; H %.1f at %.3f d (%d harmonics); Prot %s%s; shuffled best H %s"
              % (r["kic"], r["n"], r["span"], r["h"], r["period"], r["m"], r["prot"], " (rotation)" if r["rotation"] else "",
                 ", ".join("%.0f" % v for v in r["shuffled"][-3:])))
    json.dump(res, open(core.results(clockwork_OUT), "w"), indent=0)


clockwork_POOL = {}


def clockwork_pool_init():
    by, _ = clockwork_data()
    pool = {}
    for kic, ev in by.items():
        for t, q in ev:
            pool.setdefault(q, []).append((t, kic))
    clockwork_POOL.update({q: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for q, v in pool.items()})
    clockwork_POOL["by"] = by


def clockwork_pooled(args):
    """Best H with each flare time replaced by another star's from the same quarter."""
    kic, seed = args
    rng = np.random.default_rng(seed)
    ev = clockwork_POOL["by"][kic]
    t = np.empty(len(ev))
    for i, (t0, q) in enumerate(ev):
        times, owners = clockwork_POOL[q]
        others = times[owners != kic]
        t[i] = others[rng.integers(len(others))] if len(others) else t0
    span = t.max() - t.min()
    freqs = np.arange(max(1 / 300.0, 3.0 / span), clockwork_FMAX, 1.0 / (4 * span))
    return kic, float(clockwork_h(t - t.min(), freqs)[0].max())


def clockwork_null(procs, n_top=40, n_real=20):
    res = json.load(open(core.results(clockwork_OUT)))
    top = res[:n_top]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(procs, initializer=clockwork_pool_init) as ex:
        sh = list(ex.map(clockwork_pooled, [(r["kic"], s) for r in top for s in range(n_real)]))
    by, _ = clockwork_data()
    pooled = {}
    for k, h in sh:
        pooled.setdefault(k, []).append(h)
    for r in top:
        r["pooled"] = sorted(pooled[r["kic"]])
        r["p_pooled"] = float(np.mean(np.array(pooled[r["kic"]]) >= r["h"]))
        t = np.array(sorted(e[0] for e in by[r["kic"]]))
        ph = (t / r["period"]) % 1.0
        hist = np.histogram(ph, bins=50, range=(0, 1))[0]
        k = int(np.argmax(hist))
        d = np.abs(((ph - (k + 0.5) / 50) + 0.5) % 1.0 - 0.5)
        r["in_peak"] = int(np.sum(d < 0.02))
    print("the %d highest stars against %d draws of other stars' flare times in the same quarters:" % (n_top, n_real))
    for r in top:
        print("  " + clockwork_LABEL + " %-9d %4d flares H %7.1f at %8.3f d (%2d harmonics) %s; pooled best H %s; p %.2f; %d flares within 0.02 "
              "of the peak phase" % (r["kic"], r["n"], r["h"], r["period"], r["m"], "rotation" if r["rotation"] else "        ",
                                     ", ".join("%.0f" % v for v in r["pooled"][-3:]), r["p_pooled"], r["in_peak"]))
    json.dump(res, open(core.results(clockwork_OUT), "w"), indent=0)


clockwork_INSTRUMENT = ((("TESS orbit", 13.7), ("sector", 27.4)) if clockwork_CAT == "tess" else
                        (("Kepler orbit", 372.5), ("quarter", 93.0), ("monthly downlink", 31.0)))


def clockwork_vet():
    """Vet null-beating stars: distinct cycles in the peak, Kepler EB periods, spacecraft timing harmonics."""
    res = json.load(open(core.results(clockwork_OUT)))
    by, _ = clockwork_data()
    eb_p = core.data("tess_eb_prsa2022.json" if clockwork_CAT == "tess" else "kepler_eb_kirk2016.json")
    if not os.path.exists(eb_p):
        from astroquery.vizier import Vizier
        idc, cat = ("TIC", "J/ApJS/258/16/tess-ebs") if clockwork_CAT == "tess" else ("KIC", "J/AJ/151/68/catalog")
        t = Vizier(row_limit=-1, columns=[idc, "Per"], timeout=600).get_catalogs(cat)[0]
        json.dump({str(int(r[idc])): float(r["Per"]) for r in t if r["Per"] is not np.ma.masked}, open(eb_p, "w"))
    eb = json.load(open(eb_p))
    for r in res:
        if "p_pooled" not in r or r["p_pooled"] > 0 or r["rotation"]:
            continue
        t = np.array(sorted(e[0] for e in by[r["kic"]]))
        ph = (t / r["period"]) % 1.0
        k = int(np.argmax(np.histogram(ph, bins=50, range=(0, 1))[0]))
        d = np.abs(((ph - (k + 0.5) / 50) + 0.5) % 1.0 - 0.5)
        cyc = np.floor(t / r["period"])[d < 0.02]
        r["peak_cycles"] = int(len(set(cyc.tolist())))
        r["cycles_spanned"] = int(np.floor(t.max() / r["period"]) - np.floor(t.min() / r["period"]) + 1)
        pe = eb.get(str(r["kic"]))
        r["eb_period"] = pe
        r["eb"] = bool(pe) and any(abs(r["period"] * m / (pe * n) - 1) < 0.01 for m in (1, 2, 3) for n in (1, 2, 3))
        r["instrument"] = next((nm for nm, pi in clockwork_INSTRUMENT for n in (1, 2, 3, 4)
                                if abs(r["period"] / (pi / n) - 1) < 0.03 or abs(r["period"] / (pi * n) - 1) < 0.03), None)
        r["burst"] = r["peak_cycles"] < 4
        # period refined over +-0.2 per cent; a clock keeps the phase scatter below 1e-3
        fr = np.linspace(1 / r["period"] * (1 - 2e-3), 1 / r["period"] * (1 + 2e-3), 801)
        h = clockwork_h(t - t.min(), fr)[0]
        P = 1 / fr[int(np.argmax(h))]
        ph = ((t - t.min()) / P) % 1.0
        hist = np.histogram(ph, bins=100, range=(0, 1))[0]
        c = (int(np.argmax(np.convolve(np.r_[hist, hist[:4]], np.ones(5), "valid")[:100])) + 2.5) / 100
        dd = ((ph - c + 0.5) % 1.0) - 0.5
        r["period_refined"] = float(P)
        r["jitter"] = float(np.std(dd[np.abs(dd) < 0.03]))
        r["jitter_floor"] = clockwork_CADENCE / (P * math.sqrt(12))
        print("  " + clockwork_LABEL + " %-9d %4d flares at %8.3f d: peak fed by %d of %d cycles%s; EB %s%s; spacecraft timing %s; phase scatter "
              "%.4f (cadence floor %.4f)"
              % (r["kic"], r["n"], P, r["peak_cycles"], r["cycles_spanned"], " (burst)" if r["burst"] else "",
                 "%.4f d" % pe if pe else "no", " (match)" if r["eb"] else "", r["instrument"] or "no", r["jitter"],
                 r["jitter_floor"]))
    json.dump(res, open(core.results(clockwork_OUT), "w"), indent=0)


def clockwork_record():
    res = json.load(open(core.results(clockwork_OUT)))
    top = [r for r in res if "p_pooled" in r]
    beat = [r for r in top if r["p_pooled"] == 0.0 and not r["rotation"]]

    def vet(r):
        if r.get("rotation"):
            return "rotation period or a multiple"
        if "burst" not in r:
            return None
        if r["burst"]:
            return "a burst: the phase peak is fed by %d cycles" % r["peak_cycles"]
        if r["eb"]:
            return "eclipsing binary period"
        if r["instrument"]:
            return "spacecraft timing (%s)" % r["instrument"]
        return "phase scatter %.4f of the period, above the 1e-3 of a clock" % r["jitter"] if r["jitter"] >= 1e-3 else \
            "open: phase scatter %.4f of the period" % r["jitter"]
    clock = [r for r in beat if vet(r) and vet(r).startswith("open")]
    tess = clockwork_CAT == "tess"
    rid = "tess_clockwork_flares" if tess else "kepler_clockwork_flares"
    lab = clockwork_LABEL
    locked = [r for r in beat if vet(r) and vet(r).startswith("phase scatter")]
    jit = [r["jitter"] for r in beat if "jitter" in r]
    core.record_search(
        rid, "Clockwork flares",
        archive=("Yang et al. 2023 TESS flare catalogue (J/A+A/669/A15)" if tess else
                 "Yang and Liu 2019 Kepler flare catalogue (J/ApJS/241/29)"),
        selection="%s stars with 10 or more catalogued flares" % ("TESS" if tess else "Kepler"),
        code=os.path.abspath(__file__), entry="clockwork",
        n_targets=len(res), params=dict(harmonics=clockwork_MMAX, periods_d=[1 / clockwork_FMAX, 300], null_draws=20),
        null_summary="each of the %d highest stars' flare times replaced by other stars' flare times from the same %s, "
                     "20 times, which keeps the spacecraft's gaps" % (len(top), "TESS orbits" if tess else "quarters"),
        sensitivity="a train of flare-like events phase-locked within about 2.5 per cent of its period",
        ledger="Clockwork flares, TESS" if tess else "Clockwork flares",
        result="%d of the %d highest stars beat all 20 of their null draws away from rotation: %d burst%s, %d at spacecraft "
               "timing, %d eclipsing binaries, %d phase-locked with scatter %s of the period (%s %s)%s"
               % (len(beat), len(top), sum(1 for r in beat if r.get("burst")),
                  "" if sum(1 for r in beat if r.get("burst")) == 1 else "s",
                  sum(1 for r in beat if not r.get("burst") and r.get("instrument")),
                  sum(1 for r in beat if not r.get("burst") and r.get("eb")), len(locked),
                  "%.4f-%.4f" % (min(jit), max(jit)) if jit else "-", lab,
                  ", ".join("%d at %.3f d" % (r["kic"], r["period_refined"]) for r in locked) or "none",
                  "; none below the 1e-3 of a clock" if not clock else
                  "; below the 1e-3 of a clock: " + ", ".join("%s %d" % (lab, r["kic"]) for r in clock)))
    core.clear_targets(rid)
    core.record_targets(rid, [dict(
        target="%s:%d" % (lab.lower(), r["kic"]), stat=r["h"], p=r.get("p_pooled"),
        unit="best H over %.1f-300 d" % (1 / clockwork_FMAX), vet=vet(r),
        note="%d flares, best %.3f d with %d harmonics" % (r["n"], r.get("period_refined", r["period"]), r["m"])) for r in res])
    print("recorded %d stars" % len(res))


def clockwork_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        clockwork_record()
        return
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", choices=("search", "null", "vet"), default="search")
    p.add_argument("--procs", type=int, default=1)
    a = p.parse_args()
    if a.mode == "vet":
        clockwork_vet()
        return
    (clockwork_null if a.mode == "null" else clockwork_main)(a.procs)


levels_DOC = r"""Discrete flux levels in TESS 20 s light curves within 30 pc: sharp 7-sigma steps between three or more revisited levels; common-mode steps and flares removed.

    python -I seti.py flashes levels [--procs 4]
    python -I seti.py flashes levels record
    python -I seti.py flashes levels stream --targets FILE [--part k/n] [--procs 3]   (SPOC two-minute light curves of the star-sectors in FILE, a determinism targets list, read in memory, steps and levels into LEVELS_ROOT/levels2_<k>of<n>.jsonl)"""

levels_W = 15  # 30 min of 2-min bins
levels_SNR = 7.0
levels_DECAY = 0.5  # an up-step whose light falls by half its jump within the next window is a flare's rise
levels_DECAY_DAYS = 0.125  # steps down within three hours of a flare's rise are its decay


def levels_steps(tb, fb, w, gap, sharp_half=1):
    """Steps in a binned light curve: 7-sigma jumps between window medians, each with its sharpest transition, and the levels between them."""
    from scipy.ndimage import median_filter
    med = median_filter(fb, size=w, mode="nearest")
    delta = np.full(len(fb), np.nan)
    delta[w:-w] = med[w + w // 2:len(fb) - w + w // 2] - med[w - w // 2 - 1:len(fb) - w - w // 2 - 1]
    # no steps across data gaps
    for gi in np.nonzero(np.diff(tb) > gap)[0]:
        delta[max(gi - w, 0):gi + w + 1] = np.nan
    ok = np.isfinite(delta)
    if ok.sum() < 3 * w:
        return None
    sd = 1.4826 * np.median(np.abs(delta[ok] - np.median(delta[ok])))
    sig_bin = 1.4826 * np.median(np.abs(np.diff(fb))) / math.sqrt(2)
    steps = []
    order = np.argsort(-np.abs(np.nan_to_num(delta)))
    for i in order:
        if not ok[i] or abs(delta[i]) < levels_SNR * sd:
            break
        if any(abs(i - s["i"]) < 2 * w for s in steps):
            continue
        # sharpest consistent transition: ramps of 0-8 bins near i
        best = None
        for j in range(i - w // 2, i + w // 2 + 1):
            a = fb[max(j - w, 0):j]
            b = fb[j:j + w]
            if len(a) < 5 or len(b) < 5:
                continue
            for half in (0, 1, 2, 4, 8):
                lo, hi = j - half, j + half
                if lo - w < 0 or hi + w > len(fb):
                    continue
                pre, post = fb[lo - w:lo], fb[hi:hi + w]
                m1, m2 = float(np.median(pre)), float(np.median(post))
                ramp = np.linspace(m1, m2, hi - lo + 2)[1:-1] if hi > lo else np.array([])
                model = np.concatenate([np.full(len(pre), m1), ramp, np.full(len(post), m2)])
                seg = fb[lo - w:hi + w]
                chi = float(np.sum((seg - model) ** 2)) / sig_bin ** 2 - (len(seg) - 2 * w - 2 * half)
                if best is None or chi < best[0]:
                    best = (chi, j, half, m2 - m1)
        if best is None:
            continue
        # a flare rises sharply and decays: the light just after the transition against the window after it, over the jump
        j, k = best[1], max(3, w // 5)
        early = float(np.median(fb[j:j + k])) if j + k <= len(fb) else float("nan")
        late = float(np.median(fb[j + w:j + 2 * w])) if j + 2 * w <= len(fb) else float("nan")
        steps.append(dict(i=int(i), t=float(tb[j]), jump=float(best[3]), snr=float(abs(delta[i]) / sd),
                          half=int(best[2]), sharp=best[2] <= sharp_half,
                          decay=round((early - late) / best[3], 3) if best[3] and np.isfinite(early - late) else None))
    steps.sort(key=lambda s: s["t"])
    # segments between detected steps with their medians
    edges = [0] + [s["i"] for s in steps] + [len(fb)]
    segs = [dict(t0=float(tb[a]), t1=float(tb[b - 1]), level=float(np.median(fb[a:b])), n=int(b - a))
            for a, b in zip(edges[:-1], edges[1:]) if b - a >= 5]
    return dict(sd=float(sd), sig_bin=float(sig_bin), steps=steps, segs=segs)


def levels_one(p):
    from astropy.io import fits
    try:
        with fits.open(os.path.join(tess20_DL, p)) as h:
            d = h[1].data
            t, f, q = (np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"], int))
            sector = int(h[0].header.get("SECTOR", 0))
    except Exception:
        return None
    good = (q == 0) & np.isfinite(f) & np.isfinite(t)
    t, f = t[good], f[good]
    if len(f) < 5000:
        return None
    f = f / np.median(f)
    k = np.floor((t - t[0]) / (2.0 / 1440)).astype(int)
    n = np.bincount(k)
    keep = n >= 4
    tb = (np.bincount(k, t) / np.maximum(n, 1))[keep]
    fb = (np.bincount(k, f) / np.maximum(n, 1))[keep]
    r = levels_steps(tb, fb, levels_W, 10.0 / 1440)
    if r is None:
        return None
    for st in r["steps"]:
        st["ramp_min"] = 2 * st["half"] * 2.0
    return dict(file=p, sector=sector, **r)


levels_ROOT = os.environ.get("LEVELS_ROOT", core.data("levels2"))


def levels_one_spoc(x):
    """Steps and levels of one star-sector's SPOC two-minute light curve read in memory; None when it cannot be read."""
    try:
        b = core.spoc_bytes(x["tic"], x["sector"])
    except Exception:
        return None
    if not b:
        return dict(id=x["id"], status="no light curve")
    try:
        t, f, _ = core.read(b, columns=("PDCSAP_FLUX",))
    except Exception as e:
        return dict(id=x["id"], status="error", error=repr(e)[:160])
    if len(f) < 2000:
        return dict(id=x["id"], status="short")
    r = levels_steps(t, f, levels_W, 10.0 / 1440)
    if r is None:
        return dict(id=x["id"], status="short")
    for st in r["steps"]:
        st["ramp_min"] = 2 * st["half"] * 2.0
    return dict(id=x["id"], status="ok", tic=x["tic"], sector=x["sector"], cam=x.get("cam"), ccd=x.get("ccd"), plx=x.get("plx"), **r)


def levels_stream(targets_path, part=None, procs=3):
    """The star-sectors of a determinism targets list through levels_one_spoc on procs processes, one line each into ROOT/levels2_<k>of<n>.jsonl, past those any levels2 file under ROOT holds; the number left unread."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(levels_ROOT, exist_ok=True)
    targets = json.load(open(targets_path))
    k, n = (int(v) for v in part.split("/")) if part else (0, 1)
    done = set()
    for p in glob.glob(os.path.join(levels_ROOT, "levels2_*.jsonl")):
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                done.add(json.loads(ln)["id"])
            except (ValueError, KeyError):
                pass
    todo = [dict(x, id="tic:%d:s%d" % (x["tic"], x["sector"])) for x in targets[k::n]]
    todo = [x for x in todo if x["id"] not in done]
    path = os.path.join(levels_ROOT, "levels2_%dof%d.jsonl" % (k, n))
    print("%d star-sectors to read, %d read under the root" % (len(todo), len(done)), flush=True)
    t0, m, miss = time.time(), 0, 0
    with ProcessPoolExecutor(procs) as ex, open(path, "a", encoding="utf-8") as fh:
        for r in ex.map(levels_one_spoc, todo, chunksize=4):
            if r is None:
                miss += 1
                continue
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            m += 1
            if m % 200 == 0:
                fh.flush()
                print("%d of %d, %.1f a minute, %d to read again" % (m, len(todo), 60 * m / (time.time() - t0), miss), flush=True)
    print("%d star-sectors read, %d to read again" % (m, miss), flush=True)
    return miss


def levels_count(segs, real, sd, sig_bin, w, cad):
    """Levels of one light curve with the segments joined across rejected steps: (levels, levels revisited twice or more); a segment begins at a kept step when a real step's sharpest transition lies within half a window of its start."""
    # a segment starts where the window medians differ most, a real step at its sharpest transition, at most w // 2 bins away
    near = (w // 2 + 1) * cad
    times = [st["t"] for st in real]
    merged = []
    for sg in segs:
        if merged and not any(abs(sg["t0"] - t) <= near for t in times):
            m = merged[-1]
            m["level"] = (m["level"] * m["n"] + sg["level"] * sg["n"]) / (m["n"] + sg["n"])
            m["n"] += sg["n"]
            m["t1"] = sg["t1"]
        else:
            merged.append(dict(sg))
    tol = 3 * max(sd, sig_bin / math.sqrt(w))
    groups = []
    for v in sorted(m["level"] for m in merged):
        if groups and v - groups[-1][-1] < tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    centres = [float(np.mean(g)) for g in groups]
    seq = [int(np.argmin([abs(m["level"] - c) for c in centres])) for m in merged]
    visits = {g: sum(1 for a, b in zip([None] + seq[:-1], seq) if b == g and a != g) for g in set(seq)}
    return len(centres), sum(1 for v in visits.values() if v >= 2)


def levels_main(procs):
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(x for x in os.listdir(tess20_DL) if x.endswith("_fast-lc.fits"))
    with ProcessPoolExecutor(procs) as ex:
        res = [r for r in ex.map(levels_one, files, chunksize=4) if r]
    by_sector = {}
    for r in res:
        for s in r["steps"]:
            by_sector.setdefault(r["sector"], []).append((s["t"], r["file"]))
    out = []
    for r in res:
        for s in r["steps"]:
            others = {fn for tt, fn in by_sector[r["sector"]] if abs(tt - s["t"]) < 1.0 / 24 and fn != r["file"]}
            s["common"] = len(others) >= 2
        for s in r["steps"]:
            s["flare"] = s["jump"] > 0 and s["decay"] is not None and s["decay"] > levels_DECAY
        # the steps down within three hours after a flare's rise are its decay
        rises = [s["t"] for s in r["steps"] if s["flare"]]
        for s in r["steps"]:
            if s["jump"] < 0 and any(0 < s["t"] - t0 <= levels_DECAY_DAYS for t0 in rises):
                s["flare"] = True
        real = [s for s in r["steps"] if s["sharp"] and not s["common"] and not s["flare"]]
        nlev, revisited = levels_count(r["segs"], real, r["sd"], r["sig_bin"], levels_W, 2.0 / 1440)
        out.append(dict(file=r["file"], sector=r["sector"], n_steps=len(r["steps"]), n_sharp=sum(s["sharp"] for s in r["steps"]),
                        n_common=sum(s["common"] for s in r["steps"]), n_flare=sum(s["flare"] and s["sharp"] and not s["common"] for s in r["steps"]),
                        n_real=len(real), levels=nlev,
                        revisited=revisited, multi=revisited >= 3, steps=real[:20], sd=r["sd"]))
    n_multi = sum(o["multi"] for o in out)
    print("%d light curves; sharp steps %d, %d of them common-mode, %d of flares, rises that decay by half within the next window and the steps down after them; "
          "light curves with real sharp steps %d; with three or more levels each revisited %d"
          % (len(out), sum(o["n_sharp"] for o in out), sum(o["n_common"] for o in out), sum(o["n_flare"] for o in out),
             sum(1 for o in out if o["n_real"]), n_multi))
    up = sum(1 for o in out for s in o["steps"] if s["jump"] > 0)
    dn = sum(1 for o in out for s in o["steps"] if s["jump"] < 0)
    print("real sharp steps up %d, down %d" % (up, dn))
    for o in sorted(out, key=lambda o: (-o["revisited"], -o["n_real"]))[:25]:
        print("  %s: %d real sharp steps, %d levels, %d revisited; steps %s"
              % (o["file"][:40], o["n_real"], o["levels"], o["revisited"],
                 ", ".join("%.3f %+.4f (%.0f min)" % (s["t"], s["jump"], s["ramp_min"]) for s in o["steps"][:6])))
    json.dump(out, open(os.path.join(tess20_HERE, "levels.json"), "w"))


def levels_record():
    out = json.load(open(os.path.join(tess20_HERE, "levels.json")))
    tg = {}
    for t in json.load(open(os.path.join(tess20_HERE, "glint_targets.json"))):
        tg[(int(t["sector"]), int(t["tic"]))] = t["gaia"]
    best = {}
    for o in out:
        m = re.match(r"s(\d+)_(\d+)_fast-lc\.fits", o["file"])
        g = tg.get((int(m.group(1)), int(m.group(2)))) if m else None
        if g and (g not in best or o["revisited"] > best[g]["revisited"]):
            best[g] = o
    multi = [o for o in out if o["multi"]]
    core.record_search(
        "tess20_levels", "Flux in discrete levels", archive="MAST TESS SPOC 20 s light curves", selection="stars within 30 pc",
        code=os.path.abspath(__file__), entry="levels", n_targets=len(best),
        params=dict(bin_min=2, window_min=30, snr=levels_SNR, max_ramp_min=2, common_hours=1, level_sigma=3),
        null_summary="common-mode steps on three or more stars of a sector: %d, and flares, rises that lose half their jump within the next "
                     "half hour and the steps down within three hours after them: %d, of %d sharp steps"
                     % (sum(o["n_common"] for o in out), sum(o["n_flare"] for o in out), sum(o["n_sharp"] for o in out)),
        sensitivity="levels separated by 7 sigma of the 30-minute median noise, transitions of 2 minutes or less",
        ledger="Flux in discrete levels",
        result="%d light curves: %d with real sharp steps, %d with three or more levels each revisited%s"
               % (len(out), sum(1 for o in out if o["n_real"]), len(multi),
                  (": " + ", ".join(o["file"][:22] for o in multi[:8])) if multi else ""))
    core.clear_targets("tess20_levels")
    core.record_targets("tess20_levels", [dict(
        target="gaia_dr3:%s" % g, stat=float(o["revisited"]), unit="levels revisited at least twice",
        note="%s: %d real sharp steps, %d levels" % (o["file"], o["n_real"], o["levels"])) for g, o in best.items()])
    print("recorded %d stars" % len(best))


def levels_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        levels_record()
        return
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", choices=["stream"])
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--targets")
    p.add_argument("--part", default=None)
    a = p.parse_args()
    if a.mode == "stream":
        sys.exit(1 if levels_stream(a.targets, a.part, a.procs) else 0)
    levels_main(a.procs)


keplevels_DOC = r"""Discrete flux levels in Kepler long-cadence light curves of the DR25 stars: steps within one cadence between three or more revisited levels in a quarter; steps three other stars of the channel or twenty of the focal plane share removed.

    python -I seti.py flashes keplevels targets                    (the DR25 stellar table into ROOT/targets.json)
    python -I seti.py flashes keplevels stream [--part k/n] [--sub j/m] [--threads 4]   (each star's quarters read in memory, steps into ROOT/steps_<k>of<n>[_<j>of<m>].jsonl; exits 1 while stars remain unread)
    python -I seti.py flashes keplevels analyse                    (every part: channel-common and focal-plane steps removed, levels counted)
    python -I seti.py flashes keplevels vet                        (the stars with three or more revisited levels against the eclipsing-binary catalogue, their gravities, their planets' transits, SIMBAD and Gaia's variability classes)
    python -I seti.py flashes keplevels record"""

keplevels_ROOT = os.environ.get("KEPLEVELS_ROOT", core.data("keplevels"))
keplevels_W = 15  # cadences per window, 7.4 h
keplevels_CAD = 29.4244 / 1440  # long cadence, days
keplevels_COMMON = 3  # other stars of the channel stepping within 1.5 cadences
keplevels_PLANE = 20  # stars anywhere on the focal plane stepping within 1.5 cadences, three times the chance median of 6


def keplevels_targets():
    """DR25 stellar table: every KIC star with Kepler data, into ROOT/targets.json."""
    os.makedirs(keplevels_ROOT, exist_ok=True)
    rows = core.exoplanets("SELECT kepid, kepmag, teff, logg, radius FROM keplerstellar WHERE st_delivname = 'q1_q17_dr25_stellar'")
    json.dump([dict(kepid=int(r["kepid"]), kepmag=num(r["kepmag"]), teff=num(r["teff"]), logg=num(r["logg"]), radius=num(r["radius"]))
               for r in rows if num(r["kepid"])], open(os.path.join(keplevels_ROOT, "targets.json"), "w"))
    print("%d DR25 stars" % len(rows))


def keplevels_star(x):
    """One star's quarters read in memory: steps and levels of each quarter's PDCSAP light curve with its channel; None when the listing or any quarter fails, so the star is read again on the next pass."""
    from astropy.io import fits
    out = []
    keys = core.kepler_keys(x["kepid"], short=False)
    if not keys:
        return None
    for key in keys:
        b = core.get(core.S3 + key, timeout=300, retries=3, soft=True, pause=2)
        if not b or not b.startswith(b"SIMPLE"):
            return None
        try:
            with fits.open(io.BytesIO(b)) as h:
                d = h[1].data
                t, f, q = (np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["SAP_QUALITY"], int))
                quarter, channel = int(h[0].header.get("QUARTER", -1)), int(h[0].header.get("CHANNEL", -1))
        except Exception:
            continue
        good = (q == 0) & np.isfinite(f) & np.isfinite(t)
        t, f = t[good], f[good]
        if len(f) < 500:
            continue
        r = levels_steps(t, f / np.median(f), keplevels_W, 2.5 * keplevels_CAD)
        if r is None:
            continue
        out.append(dict(q=quarter, ch=channel, sd=round(r["sd"], 7), sig=round(r["sig_bin"], 7),
                        steps=[[round(st["t"], 5), round(st["jump"], 6), round(st["snr"], 1), st["half"]] for st in r["steps"]],
                        segs=[[round(sg["t0"], 4), round(sg["t1"], 4), round(sg["level"], 6), sg["n"]] for sg in r["segs"]]))
    return dict(kepid=x["kepid"], kepmag=x["kepmag"], quarters=out)


def keplevels_stream(part=None, threads=4, sub=None):
    """Stream a part of the DR25 stars (and a sub-part j/m of it), appending one line per star to ROOT/steps_<k>of<n>[_<j>of<m>].jsonl, past every star any steps file under ROOT holds with quarters; the number of stars left unread."""
    targets = json.load(open(os.path.join(keplevels_ROOT, "targets.json")))
    k, n = (int(v) for v in part.split("/")) if part else (0, 1)
    j, m_ = (int(v) for v in sub.split("/")) if sub else (0, 1)
    path = os.path.join(keplevels_ROOT, "steps_%dof%d%s.jsonl" % (k, n, "_%dof%d" % (j, m_) if sub else ""))
    done = set()
    for p in glob.glob(os.path.join(keplevels_ROOT, "steps_*of*.jsonl")):
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("quarters"):
                done.add(r["kepid"])
    todo = [x for x in targets[k::n][j::m_] if x["kepid"] not in done]
    print("%d stars to read, %d read before under the root" % (len(todo), len(done)), flush=True)
    t0, m, miss = time.time(), 0, 0
    with ThreadPoolExecutor(threads) as ex, open(path, "a", encoding="utf-8") as fh:
        for r in ex.map(keplevels_star, todo):
            if r is None:
                miss += 1
                continue
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            m += 1
            if m % 200 == 0:
                fh.flush()
                print("%d of %d, %.1f stars a minute, %d to read again" % (m, len(todo), 60 * m / (time.time() - t0), miss), flush=True)
    print("%d stars read, %d to read again" % (m, miss), flush=True)
    return miss


def keplevels_analyse():
    """Every part's steps: steps three other stars of the channel share within 1.5 cadences removed, levels counted per quarter."""
    best_row = {}
    for p in sorted(glob.glob(os.path.join(keplevels_ROOT, "steps_*of*.jsonl"))):
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r["kepid"] not in best_row or len(r["quarters"]) > len(best_row[r["kepid"]]["quarters"]):
                best_row[r["kepid"]] = r  # one row a star, the fullest where a star was read twice
    rows = list(best_row.values())
    index, plane = {}, {}
    for r in rows:
        for qd in r["quarters"]:
            for st in qd["steps"]:
                index.setdefault((qd["q"], qd["ch"]), []).append((st[0], r["kepid"]))
                if st[3] <= 1:
                    plane.setdefault(qd["q"], []).append(st[0])
    for v in index.values():
        v.sort()
    plane = {q: np.sort(np.array(v)) for q, v in plane.items()}
    out, n_steps, n_sharp, n_common, n_plane = [], 0, 0, 0, 0
    for r in rows:
        best = None
        for qd in r["quarters"]:
            ts = index.get((qd["q"], qd["ch"]), [])
            real = []
            for t, jump, snr, half in qd["steps"]:
                n_steps += 1
                if half > 1:
                    continue
                n_sharp += 1
                lo = np.searchsorted([x[0] for x in ts], t - 1.5 * keplevels_CAD)
                others = {kid for tt, kid in ts[lo:] if tt <= t + 1.5 * keplevels_CAD and kid != r["kepid"]}
                if len(others) >= keplevels_COMMON:
                    n_common += 1
                    continue
                pq = plane[qd["q"]]
                # the spacecraft's events step many stars across the focal plane at once
                if np.searchsorted(pq, t + 1.5 * keplevels_CAD, side="right") - np.searchsorted(pq, t - 1.5 * keplevels_CAD) - 1 >= keplevels_PLANE:
                    n_plane += 1
                    continue
                real.append(dict(t=t, jump=jump, snr=snr))
            segs = [dict(t0=a, t1=b, level=lv, n=nn) for a, b, lv, nn in qd["segs"]]
            nlev, revisited = levels_count(segs, real, qd["sd"], qd["sig"], keplevels_W, keplevels_CAD)
            row = dict(q=qd["q"], ch=qd["ch"], n_real=len(real), levels=nlev, revisited=revisited, steps=real[:12])
            if best is None or (revisited, len(real)) > (best["revisited"], best["n_real"]):
                best = row
        if best:
            out.append(dict(kepid=r["kepid"], kepmag=r["kepmag"], n_quarters=len(r["quarters"]), **best))
    multi = [o for o in out if o["revisited"] >= 3]
    up = sum(1 for o in out for st in o["steps"] if st["jump"] > 0)
    print("%d stars, %d quarters; steps %d, sharp %d, %d of those shared on the channel and %d by %d or more stars of the focal plane; "
          "stars with real sharp steps %d; with three or more levels each revisited in a quarter %d; real steps up %d, down %d"
          % (len(out), sum(o["n_quarters"] for o in out), n_steps, n_sharp, n_common, n_plane, keplevels_PLANE,
             sum(1 for o in out if o["n_real"]), len(multi), up, sum(1 for o in out for st in o["steps"]) - up))
    for o in sorted(multi, key=lambda o: (-o["revisited"], -o["n_real"]))[:25]:
        print("  KIC %d (Kp %s) Q%d ch%d: %d real sharp steps, %d levels, %d revisited; %s" % (
            o["kepid"], o["kepmag"], o["q"], o["ch"], o["n_real"], o["levels"], o["revisited"],
            ", ".join("%.3f %+.4f" % (st["t"], st["jump"]) for st in o["steps"][:6])))
    json.dump(dict(n_stars=len(out), n_steps=n_steps, n_sharp=n_sharp, n_common=n_common, n_plane=n_plane, rows=out,
                   params=dict(window=keplevels_W, snr=levels_SNR, common=keplevels_COMMON, plane=keplevels_PLANE)),
              open(core.results("keplevels.json"), "w"))


keplevels_GIANT_LOGG = 3.5  # giants' oscillations and granulation move the window medians in steps


def keplevels_vet():
    """Each star with three or more revisited levels against the Kepler eclipsing-binary catalogue (Kirk et al. 2016), the DR25 gravity, its KOIs' transit times, its SIMBAD type and Gaia DR3's variability class, into results/keplevels_vet.json."""
    from astroquery.vizier import Vizier
    d = json.load(open(core.results("keplevels.json")))
    multi = [o for o in d["rows"] if o["revisited"] >= 3]
    stars = {x["kepid"]: x for x in json.load(open(os.path.join(keplevels_ROOT, "targets.json")))}
    eb = set()
    for t in Vizier(columns=["KIC"], row_limit=-1).get_catalogs("J/AJ/151/68"):
        if "KIC" in t.colnames:
            eb |= {int(x) for x in t["KIC"]}
    ids = ",".join(str(o["kepid"]) for o in multi)
    kois = core.exoplanets("SELECT kepid, kepoi_name, kepler_name, koi_period, koi_time0bk, koi_duration FROM cumulative "
                           "WHERE koi_disposition IN ('CANDIDATE', 'CONFIRMED') AND kepid IN (%s)" % ids)
    sim = core.simbad("SELECT i.id AS kic, b.oid, b.main_id, b.otype FROM ident i JOIN basic b ON b.oid = i.oidref WHERE i.id IN (%s)"
                      % ",".join("'KIC %d'" % o["kepid"] for o in multi))
    by_kic = {int(x["kic"].split()[1]): x for x in sim}
    gaia_ids = {x["oidref"]: x["id"].split()[-1] for x in core.simbad(
        "SELECT oidref, id FROM ident WHERE oidref IN (%s) AND id LIKE 'Gaia DR3%%'" % ",".join(str(x["oid"]) for x in sim))} if sim else {}
    classes = {str(x["source_id"]): (x["best_class_name"], float(x["best_class_score"])) for x in core.gaia(
        "SELECT source_id, best_class_name, best_class_score FROM gaiadr3.vari_classifier_result WHERE source_id IN (%s)"
        % ",".join(gaia_ids.values()))} if gaia_ids else {}
    out = {}
    for o in multi:
        k, why, vet = o["kepid"], [], []
        x = by_kic.get(k)
        name = x["main_id"] if x else "KIC %d" % k
        if k in eb:
            vet.append("eclipsing-binary")
            why.append("in the Kepler eclipsing-binary catalogue")
        logg = stars.get(k, {}).get("logg")
        if logg is not None and logg < keplevels_GIANT_LOGG:
            vet.append("evolved")
            why.append("a giant (log g %.2f), whose oscillations and granulation step the window medians" % logg)
        for kr in (r for r in kois if int(r["kepid"]) == k):
            per, t0, dur = num(kr["koi_period"]), num(kr["koi_time0bk"]), num(kr["koi_duration"])
            if not (per and t0 and dur):
                continue
            near = [abs(((st["t"] - t0 + per / 2) % per) - per / 2) <= dur / 48 + 1.5 * keplevels_CAD for st in o["steps"]]
            if sum(near) >= len(near) / 2:
                vet.append("natural-source")
                why.append("the transits of %s at %.4f d" % (kr["kepler_name"] or kr["kepoi_name"], per))
        if x and x["otype"] in ("CV*", "DN*", "No*", "NL*"):
            vet.append("natural-type")
            why.append("the cataclysmic variable %s" % name)
        cl = classes.get(gaia_ids.get(x["oid"]) if x else None)
        if cl and cl[1] >= 0.5:
            vet.append("variable-star")
            why.append("Gaia DR3 classes it %s (%.2f)" % cl)
        elif x and x["otype"] in ("dS*", "gD*", "RR*", "Ce*", "Pu*", "LP*", "Mi*", "sr*"):
            vet.append("variable-star")
            why.append("a pulsating star (%s) in SIMBAD" % x["otype"])
        out[str(k)] = dict(name=name, vet=";".join(dict.fromkeys(vet)) or None, why="; ".join(why) or None, q=o["q"], ch=o["ch"],
                           n_real=o["n_real"], levels=o["levels"], revisited=o["revisited"], kepmag=o["kepmag"], logg=logg,
                           jumps=[round(st["jump"], 4) for st in o["steps"][:8]])
    left = [v for v in out.values() if not v["vet"]]
    print("%d stars with three or more revisited levels: %d eclipsing binaries, %d giants, %d planets' transits, %d cataclysmic variables, "
          "%d pulsators; %d left" % (len(out), sum("eclipsing-binary" in (v["vet"] or "") for v in out.values()),
                                     sum("evolved" in (v["vet"] or "") for v in out.values()),
                                     sum("natural-source" in (v["vet"] or "") for v in out.values()),
                                     sum("natural-type" in (v["vet"] or "") for v in out.values()),
                                     sum("variable-star" in (v["vet"] or "") for v in out.values()), len(left)))
    for v in left:
        print("  %s (Kp %s, log g %s) Q%d ch%d: %d real sharp steps, %d levels, %d revisited; jumps %s" % (
            v["name"], v["kepmag"], v["logg"], v["q"], v["ch"], v["n_real"], v["levels"], v["revisited"], v["jumps"]))
    json.dump(out, open(core.results("keplevels_vet.json"), "w"), indent=1)


def keplevels_record():
    d = json.load(open(core.results("keplevels.json")))
    vets = json.load(open(core.results("keplevels_vet.json")))
    multi = [o for o in d["rows"] if o["revisited"] >= 3]
    left = [o for o in multi if not vets[str(o["kepid"])]["vet"]]
    core.record_search(
        "kepler_levels", "Flux in discrete levels, Kepler", archive="Kepler DR25 long-cadence light curves (STScI AWS mirror)",
        release="DR25", selection="the DR25 stellar table: %d stars with light curves" % d["n_stars"],
        code=os.path.abspath(__file__), entry="keplevels", n_targets=d["n_stars"], n_unit="stars", params=d["params"],
        outputs=[core.results("keplevels.json"), core.results("keplevels_vet.json")],
        null_summary="steps that three or more other stars of the CCD channel share within 1.5 cadences: %d, and that %d or more stars of the "
                     "focal plane share: %d, of %d sharp steps" % (d["n_common"], d["params"]["plane"], d["n_plane"], d["n_sharp"]),
        sensitivity="levels separated by 7 sigma of the 7.4-hour median noise, transitions within one cadence",
        ledger="Flux in discrete levels, Kepler",
        result="%d stars: %d with real sharp steps, %d with three or more levels each revisited in a quarter, %d of them unexplained%s" % (
            d["n_stars"], sum(1 for o in d["rows"] if o["n_real"]), len(multi), len(left),
            (": " + ", ".join("KIC %d Q%d" % (o["kepid"], o["q"]) for o in left[:8])) if left else ""))
    core.clear_targets("kepler_levels")

    def vet(o):
        v = vets.get(str(o["kepid"]))
        return (v["vet"], v["why"]) if v else (None, None)
    core.record_targets("kepler_levels", [dict(target="kic:%d" % o["kepid"], stat=float(o["revisited"]), unit="levels revisited at least twice in a quarter",
                                                vet=vet(o)[0], note="Q%d channel %d: %d real sharp steps, %d levels%s" % (
                                                    o["q"], o["ch"], o["n_real"], o["levels"], "; " + vet(o)[1] if vet(o)[1] else ""))
                                           for o in d["rows"] if o["n_real"]])
    print("recorded %d stars, rows for the %d with real sharp steps" % (d["n_stars"], sum(1 for o in d["rows"] if o["n_real"])))


def keplevels_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["targets", "stream", "analyse", "vet", "record"])
    p.add_argument("--part", default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--sub", default=None)
    a = p.parse_args()
    if a.mode == "stream":
        sys.exit(1 if keplevels_stream(a.part, a.threads, a.sub) else 0)
    else:
        {"targets": keplevels_targets, "analyse": keplevels_analyse, "vet": keplevels_vet, "record": keplevels_record}[a.mode]()


determinism_DOC = r"""Determinism in light curves: each segment's residual, once the slow trend, up to three periodic signals and flares are removed, against IAAFT surrogates of the same power spectrum and flux distribution, at the native cadence and binned eightfold: the compressed length of its quantile symbols, its nonlinear prediction error and its time asymmetry; a candidate is more compressible or more predictable than its surrogates by 5 sigma, its asymmetry read beside it (flares carry asymmetry without predictability, chaos carries both), unshared by the stars of its sector and camera; a chaotic signal at the noise's level reaches 14 sigma in prediction binned, one at a third of it is not found.

    python -I seti.py flashes determinism targets --lists DIR [--plx 10]   (TESS two-minute star-sectors of the Gaia nearby stars beyond parallax plx, nearest first, into ROOT/targets.json; DIR holds the all_targets_S<nnn> lists as s<nnn>.csv)
    python -I seti.py flashes determinism stream [--part k/n] [--procs 8] [--gpu]   (SPOC two-minute light curves read in memory, one line a star-sector into ROOT/stats_<k>of<n>.jsonl)
    python -I seti.py flashes determinism local DIR [--procs 6] [--tag NAME]   (every FITS light curve under DIR, 20 s or Kepler short cadence, into ROOT/stats_<NAME>.jsonl)
    python -I seti.py flashes determinism kepler_targets   (every KOI host with a candidate or confirmed planet into ROOT/targets_kepler.json)
    python -I seti.py flashes determinism kepler [--procs 10]   (each host's long-cadence quarters read in memory, every segment on the CPU, into ROOT/stats_kepler.jsonl; exits 1 while stars remain unread)
    python -I seti.py flashes determinism check   (white and red noise, a chaotic signal, a coded on-off signal and flares through the same statistics)
    python -I seti.py flashes determinism report"""

determinism_ROOT = os.environ.get("DETERMINISM_ROOT", core.data("determinism"))
determinism_NSUR = 99  # surrogates a segment and level
determinism_ITERS = 40  # IAAFT iterations
determinism_LEVELS = 16  # quantile symbols for the compressed length
determinism_EMBED = 4  # embedding dimension of the nonlinear prediction
determinism_K = 4  # neighbours a prediction
determinism_QUERY = 3000  # predicted samples a series
determinism_LAGS = (1, 2, 4, 8, 16, 32)
determinism_BIN = 8  # cadences a sample at the coarse level
determinism_MIN_DAYS = 2.0
determinism_GAP = 0.25  # days; a longer gap splits a segment
determinism_FLARE = 5.0  # robust sigma


def determinism_segments(t, f, cad, seed=0):
    """Uniform segments of a light curve's residual: [(residual, info)] with the slow trend (half-day block medians), up to three periodic signals (each folded at its Lomb-Scargle peak and its phase-binned median subtracted) and flares (5 sigma brightenings and the half hour after) removed, flare and missing cadences refilled with noise of the residual's scatter."""
    from astropy.timeseries import LombScargle
    o = np.argsort(t)
    t, f = t[o], f[o]
    cut = np.nonzero(np.diff(t) > determinism_GAP)[0] + 1
    out = []
    rng = np.random.default_rng(seed)
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]):
        tt, ff = t[a:b], f[a:b]
        if len(tt) < 500 or tt[-1] - tt[0] < determinism_MIN_DAYS:
            continue
        pos = np.round((tt - tt[0]) / cad).astype(np.int64)
        n = int(pos[-1]) + 1
        x = np.full(n, np.nan)
        x[pos] = ff / np.median(ff) - 1.0
        fill = 1.0 - len(np.unique(pos)) / n
        if fill > 0.3:
            continue
        x = x - science.block_median(x, max(5, int(0.5 / cad)))
        grid = tt[0] + cad * np.arange(n)
        ok = np.isfinite(x)
        periods = []
        for _ in range(3):
            ts, xs = grid[ok], x[ok]
            if len(xs) > 20000:
                k = max(1, len(xs) // 20000)
                ts, xs = ts[::k], xs[::k]
            fmin, fmax = 2.0 / (grid[-1] - grid[0]), min(0.5 / cad, 100.0)
            freq = np.linspace(fmin, fmax, int(5 * (grid[-1] - grid[0]) * (fmax - fmin)))
            pw = LombScargle(ts, xs).power(freq, method="fast")
            j = int(np.argmax(pw))
            if not pw[j] > 30 * np.median(pw):
                break
            P = 1.0 / freq[j]
            ph = ((grid - grid[0]) / P) % 1.0
            nb = int(np.clip(ok.sum() / 50, 20, 200))
            k = (ph * nb).astype(int)
            med = np.array([np.nanmedian(x[k == i]) if np.any(ok & (k == i)) else 0.0 for i in range(nb)])
            x = x - np.interp(ph, (np.arange(nb) + 0.5) / nb, med, period=1.0)
            periods.append([round(float(P), 6), round(float(pw[j]), 4)])
        s = core.robust_sigma(x[ok])
        hot = ok & (x > determinism_FLARE * s)
        after = max(3, int(round(0.02 / cad)))
        mask = np.convolve(hot.astype(float), np.ones(after + 2), mode="full")[1:n + 1] > 0
        events = int(np.sum(np.diff(np.r_[0, hot.astype(int)]) == 1))
        bad = mask | ~ok
        x[bad] = rng.normal(0.0, s, int(bad.sum()))
        out.append((x, dict(t0=round(float(grid[0]), 4), t1=round(float(grid[-1]), 4), n=n, fill=round(float(fill), 4),
                            masked=round(float(mask.mean()), 4), periods=periods, flares=events, sigma=float(s))))
    return out


def determinism_iaaft(x, n, iters, rng):
    """n IAAFT surrogates of x: the amplitude spectrum imposed and the values rank-remapped in turn, ending on the exact values."""
    N = len(x)
    amp = np.abs(np.fft.rfft(x))
    sx = np.sort(x)
    Y = np.array([rng.permutation(x) for _ in range(n)])
    for _ in range(iters):
        F = np.fft.rfft(Y, axis=1)
        Y = np.fft.irfft(amp[None, :] * F / np.maximum(np.abs(F), 1e-30), N, axis=1)
        np.put_along_axis(Y, np.argsort(Y, axis=1), np.broadcast_to(sx, Y.shape), axis=1)
    return Y


def determinism_iaaft_torch(x, n, iters, seed, dev):
    """determinism_iaaft on a torch device."""
    import torch
    N = len(x)
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    amp = torch.abs(torch.fft.rfft(xt))
    sx = torch.sort(xt).values.expand(n, N).contiguous()
    g = torch.Generator(device=dev).manual_seed(seed)
    Y = xt[torch.argsort(torch.rand(n, N, generator=g, device=dev), dim=1)]
    for _ in range(iters):
        F = torch.fft.rfft(Y, dim=1)
        Y = torch.fft.irfft(amp[None, :] * F / F.abs().clamp_min(1e-30), N, dim=1)
        Y = torch.empty_like(Y).scatter_(1, torch.argsort(Y, dim=1), sx)
    return Y


def determinism_window(x):
    """Theiler window: the first lag where the autocorrelation falls below 1/e, within 5-50 samples."""
    y = x - x.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(y))))
    acf = np.fft.irfft(np.abs(np.fft.rfft(y, n)) ** 2, n)[:200]
    acf = acf / max(acf[0], 1e-30)
    below = np.nonzero(acf < 1 / math.e)[0]
    return int(np.clip(below[0] if len(below) else 50, 5, 50))


def determinism_asym(X):
    """Time asymmetry of each row: the root sum of squares over lags of <d^3>/<d^2>^1.5, d the lagged difference."""
    rows = []
    for L in determinism_LAGS:
        if L >= X.shape[1] // 10:
            break
        d = X[:, L:] - X[:, :-L]
        rows.append(np.mean(d ** 3, axis=1) / np.maximum(np.mean(d ** 2, axis=1), 1e-30) ** 1.5)
    return np.sqrt(np.sum(np.array(rows) ** 2, axis=0))


def determinism_compress(X, edges, threads=4):
    """zlib-compressed length (bytes) of each row's quantile symbols."""
    import zlib
    S = np.searchsorted(edges, X).astype(np.uint8)
    with ThreadPoolExecutor(threads) as ex:
        return np.array(list(ex.map(lambda s: len(zlib.compress(s.tobytes(), 9)), S)), float)


def determinism_npe(X, qidx, w):
    """Nonlinear prediction error of each row: the next value from the mean of the K nearest delay vectors more than w samples away, its RMS error over the row's scatter at the samples qidx."""
    from scipy.spatial import cKDTree
    m, k = determinism_EMBED, determinism_K
    out = []
    for x in X:
        N = len(x)
        E = np.stack([x[m - 1 - j:N - 1 - j] for j in range(m)], axis=1)
        target = x[m:]
        _, nb = cKDTree(E).query(E[qidx], k=k + 2 * w + 1)
        far = np.abs(nb - qidx[:, None]) > w
        rank = np.cumsum(far, axis=1)
        use = far & (rank <= k)
        cnt = use.sum(1)
        pred = np.where(use, target[np.minimum(nb, len(target) - 1)], 0.0).sum(1) / np.maximum(cnt, 1)
        good = cnt > 0
        out.append(math.sqrt(float(np.mean((target[qidx][good] - pred[good]) ** 2))) / max(float(np.std(x)), 1e-30))
    return np.array(out)


def determinism_npe_torch(X, qidx, w, dev, chunk=32):
    """determinism_npe on a torch device for every row at once: distances by chunks of queries, near-in-time vectors set to infinity."""
    import torch
    m, k = determinism_EMBED, determinism_K
    B, N = X.shape
    E = torch.stack([X[:, m - 1 - j:N - 1 - j] for j in range(m)], dim=2)
    target = X[:, m:]
    idx = torch.arange(E.shape[1], device=dev)
    qi = torch.as_tensor(qidx, device=dev)
    se = torch.zeros(B, device=dev, dtype=torch.float64)
    for c in range(0, len(qi), chunk):
        q = qi[c:c + chunk]
        d = torch.cdist(E[:, q], E)
        d.masked_fill_(((idx[None, :] - q[:, None]).abs() <= w)[None], float("inf"))
        nb = torch.topk(d, k, dim=2, largest=False).indices
        pred = torch.gather(target, 1, nb.reshape(B, -1)).reshape(B, len(q), k).mean(2)
        se += ((target[:, q] - pred) ** 2).sum(1).double()
    return (torch.sqrt(se / len(qi)) / X.std(dim=1, correction=0).double().clamp_min(1e-30)).cpu().numpy()


def determinism_level(x, seed, dev=None):
    """One series against its surrogates: (value, surrogate mean, surrogate scatter, z) of the compressed length, the prediction error and the asymmetry."""
    rng = np.random.default_rng(seed)
    w = determinism_window(x)
    N = len(x)
    qidx = np.sort(rng.choice(N - determinism_EMBED - 1, min(determinism_QUERY, N - determinism_EMBED - 1), replace=False))
    edges = np.quantile(x, np.arange(1, determinism_LEVELS) / determinism_LEVELS)
    if dev is not None:
        import torch
        Y = determinism_iaaft_torch(x, determinism_NSUR, determinism_ITERS, seed, dev)
        Xt = torch.cat([torch.as_tensor(x, dtype=torch.float32, device=dev)[None, :], Y])
        npe = determinism_npe_torch(Xt, qidx, w, dev)
        X = Xt.cpu().numpy().astype(np.float64)
    else:
        X = np.vstack([x[None, :], determinism_iaaft(x, determinism_NSUR, determinism_ITERS, rng)])
        npe = determinism_npe(X, qidx, w)
    stats = dict(comp=determinism_compress(X, edges), npe=npe, asym=determinism_asym(X))
    out = dict(w=w)
    for key, v in stats.items():
        mu, sd = float(np.mean(v[1:])), float(np.std(v[1:], ddof=1))
        out[key] = [round(float(v[0]), 6), round(mu, 6), round(sd, 6), round((float(v[0]) - mu) / max(sd, 1e-12), 2)]
    return out


def determinism_curve(t, f, cad, seed=0, dev=None):
    """Every segment of a light curve at the native cadence and binned determinism_BIN-fold, with z combined over segments."""
    segs = []
    for i, (x, info) in enumerate(determinism_segments(t, f, cad, seed)):
        info["native"] = determinism_level(x, seed + 101 * i, dev)
        nb = len(x) // determinism_BIN
        if nb >= 500:
            info["coarse"] = determinism_level(x[:nb * determinism_BIN].reshape(nb, determinism_BIN).mean(1), seed + 101 * i + 7, dev)
        segs.append(info)
    Z = {}
    for lev in ("native", "coarse"):
        zs = [s[lev] for s in segs if lev in s]
        if zs:
            Z[lev] = [round(sum(z[key][3] for z in zs) / math.sqrt(len(zs)), 2) for key in ("comp", "npe", "asym")]
    return dict(cad_s=round(cad * 86400, 2), segs=segs, Z=Z)


def determinism_lists(dirpath):
    """{(tic, sector): (camera, ccd, tmag, ra, dec)} from the two-minute target lists s<nnn>.csv under dirpath."""
    out = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "s[0-9][0-9][0-9].csv"))):
        sector = int(os.path.basename(p)[1:4])
        for line in open(p, encoding="utf-8", errors="replace"):
            v = line.strip().split(",")
            if len(v) == 6 and v[0].isdigit():
                out[(int(v[0]), sector)] = (int(v[1]), int(v[2]), num(v[3]), float(v[4]), float(v[5]))
    return out


def determinism_targets(lists, plx_min=10.0):
    """TESS two-minute star-sectors of the Gaia Catalogue of Nearby Stars' sources beyond parallax plx_min, matched within 3 arcsec at the nearest of epochs 2000, 2016 and 2021 and within 2.5 mag of G, nearest first, into ROOT/targets.json."""
    from scipy.spatial import cKDTree
    os.makedirs(determinism_ROOT, exist_ok=True)
    rows = catalogues.gcns("source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag", "parallax > %g" % plx_min, maxrec=600000, timeout=1800)
    print("%d nearby stars beyond parallax %g mas" % (len(rows), plx_min), flush=True)
    L = determinism_lists(lists)
    print("%d two-minute star-sectors in %s" % (len(L), lists), flush=True)
    pos = {}
    for (tic, sector), v in L.items():
        pos.setdefault(tic, (v[3], v[4], v[2]))
    tics = np.array(list(pos))
    radec = np.array([pos[x][:2] for x in tics])
    tree = cKDTree(science.unit_vector(radec[:, 0], radec[:, 1]).T)
    match = {}
    for r in rows:
        d, j = min(tree.query(science.unit_vector(*core.star_at(float(r["ra"]), float(r["dec"]), num(r["pmra"], 0.0),
                                                                num(r["pmdec"], 0.0), 2016.0, at)), k=1)
                   for at in (2000.0, 2016.0, 2021.0))
        if d < math.radians(3.0 / 3600) and abs((pos[tics[j]][2] or 99) - num(r["phot_g_mean_mag"], 99)) < 2.5:
            tic = int(tics[j])
            if tic not in match or float(r["parallax"]) > match[tic][1]:
                match[tic] = (r["source_id"], float(r["parallax"]), num(r["phot_g_mean_mag"]))
    out = [dict(tic=tic, sector=sector, gaia=match[tic][0], plx=round(match[tic][1], 3), G=match[tic][2], cam=v[0], ccd=v[1])
           for (tic, sector), v in L.items() if tic in match]
    out.sort(key=lambda r: (-r["plx"], r["tic"], r["sector"]))
    json.dump(out, open(os.path.join(determinism_ROOT, "targets.json"), "w"))
    print("%d star-sectors of %d stars" % (len(out), len({r["tic"] for r in out})), flush=True)


def determinism_one_spoc(x):
    """One star-sector's SPOC two-minute light curve read in memory and its statistics on the CPU; None when unreadable."""
    try:
        b = core.spoc_bytes(x["tic"], x["sector"], strict=True)
        if not b:
            return dict(x, status="no light curve")
        t, f, _ = core.read(b, columns=("PDCSAP_FLUX",))
        if len(t) < 1000:
            return dict(x, status="short")
        cad = float(np.median(np.diff(t)))
        if _DET_DEV.get("gpu"):
            return dict(x, status="segments", cad=cad, segs=[(s.astype(np.float32), info) for s, info in
                                                              determinism_segments(t, f, cad, seed=x["tic"] % 100000)])
        return dict(x, status="ok", **determinism_curve(t, f, cad, seed=x["tic"] % 100000))
    except Exception as e:
        return dict(x, status="error", error=repr(e)[:200])


_DET_DEV = {}


def determinism_pool_init(gpu):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _DET_DEV["gpu"] = gpu


def determinism_finish_gpu(r, dev):
    """Statistics on the GPU for a star-sector whose segments a worker prepared."""
    segs = []
    seed = r["tic"] % 100000
    for i, (x, info) in enumerate(r.pop("segs")):
        x = x.astype(np.float64)
        info["native"] = determinism_level(x, seed + 101 * i, dev)
        nb = len(x) // determinism_BIN
        if nb >= 500:
            info["coarse"] = determinism_level(x[:nb * determinism_BIN].reshape(nb, determinism_BIN).mean(1), seed + 101 * i + 7, dev)
        segs.append(info)
    Z = {}
    for lev in ("native", "coarse"):
        zs = [s[lev] for s in segs if lev in s]
        if zs:
            Z[lev] = [round(sum(z[key][3] for z in zs) / math.sqrt(len(zs)), 2) for key in ("comp", "npe", "asym")]
    r.update(status="ok", cad_s=round(r.pop("cad") * 86400, 2), segs=segs, Z=Z)
    return r


def determinism_done():
    done = set()
    for p in glob.glob(os.path.join(determinism_ROOT, "stats_*.jsonl")):
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("status") in ("ok", "no light curve", "short"):
                done.add(r["id"])
    return done


def determinism_stream(part=None, procs=8, gpu=False):
    """Stream a part of the target star-sectors: workers read each light curve in memory and prepare its segments (and on the CPU its statistics); with gpu the surrogates and predictions run in this process on the GPU, the workers started by spawn so that none inherits its CUDA context."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    targets = json.load(open(os.path.join(determinism_ROOT, "targets.json")))
    k, n = (int(v) for v in part.split("/")) if part else (0, 1)
    done = determinism_done()
    todo = [dict(x, id="tic:%d:s%d" % (x["tic"], x["sector"])) for x in targets[k::n]]
    todo = [x for x in todo if x["id"] not in done]
    dev = None
    if gpu:
        import torch
        dev = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.25)
    path = os.path.join(determinism_ROOT, "stats_%dof%d.jsonl" % (k, n))
    print("%d star-sectors to read, %d done under the root, statistics on %s" % (len(todo), len(done), dev or "the CPU"), flush=True)
    t0, m = time.time(), 0
    with ProcessPoolExecutor(procs, mp_context=multiprocessing.get_context("spawn"), initializer=determinism_pool_init,
                             initargs=(gpu,)) as ex, open(path, "a", encoding="utf-8") as fh:
        for r in ex.map(determinism_one_spoc, todo, chunksize=1):
            if r.get("status") == "segments":
                r = determinism_finish_gpu(r, dev)
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            m += 1
            if m % 50 == 0:
                fh.flush()
                print("%d of %d, %.1f a minute" % (m, len(todo), 60 * m / (time.time() - t0)), flush=True)
    print("%d star-sectors written" % m, flush=True)


def determinism_one_file(path):
    """One cached FITS light curve's statistics on the CPU."""
    name = os.path.basename(path)
    try:
        t, f, _ = core.read(open(path, "rb").read(), columns=("PDCSAP_FLUX",))
        if len(t) < 1000:
            return dict(id=name, status="short")
        cad = float(np.median(np.diff(t)))
        seed = int(re.sub(r"\D", "", name)[-5:] or 0)
        return dict(id=name, status="ok", **determinism_curve(t, f, cad, seed=seed))
    except Exception as e:
        return dict(id=name, status="error", error=repr(e)[:200])


def determinism_local(directory, procs=6, tag="local"):
    """Every FITS light curve under a directory through determinism_curve on procs processes, into ROOT/stats_<tag>.jsonl."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(determinism_ROOT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(directory, "**", "*.fits"), recursive=True))
    path = os.path.join(determinism_ROOT, "stats_%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8", errors="replace"):
            try:
                done.add(json.loads(ln)["id"])
            except (ValueError, KeyError):
                pass
    todo = [p for p in files if os.path.basename(p) not in done]
    print("%d light curves under %s, %d to do" % (len(files), directory, len(todo)), flush=True)
    t0, m = time.time(), 0
    with ProcessPoolExecutor(procs, initializer=determinism_pool_init, initargs=(False,)) as ex, open(path, "a", encoding="utf-8") as fh:
        for r in ex.map(determinism_one_file, todo, chunksize=1):
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            m += 1
            if m % 20 == 0:
                fh.flush()
                print("%d of %d, %.2f a minute" % (m, len(todo), 60 * m / (time.time() - t0)), flush=True)
    print("%d light curves written" % m, flush=True)


def determinism_kepler_targets():
    """Every KOI host with a candidate or confirmed planet, brightest first, into ROOT/targets_kepler.json."""
    os.makedirs(determinism_ROOT, exist_ok=True)
    rows = core.exoplanets("SELECT kepid, koi_kepmag, koi_disposition FROM cumulative")
    best = {}
    for r in rows:
        if r["koi_disposition"] in ("CANDIDATE", "CONFIRMED") and num(r["kepid"]):
            best[int(r["kepid"])] = num(r["koi_kepmag"], 99.0)
    out = [dict(kepid=k, kepmag=v) for k, v in sorted(best.items(), key=lambda kv: kv[1])]
    json.dump(out, open(os.path.join(determinism_ROOT, "targets_kepler.json"), "w"))
    print("%d KOI hosts" % len(out))


def determinism_one_kepler(x):
    """One Kepler star's long-cadence quarters read in memory, every segment of every quarter through the statistics on the CPU, z combined over them; None when the listing or a quarter fails, so the star is read again."""
    keys = core.kepler_keys(x["kepid"], short=False)
    if not keys:
        return None
    segs = []
    for i, key in enumerate(keys):
        b = core.get(core.S3 + key, timeout=300, retries=3, soft=True, pause=2)
        if not b or not b.startswith(b"SIMPLE"):
            return None
        try:
            t, f, _ = core.read(b, columns=("PDCSAP_FLUX",))
        except Exception:
            continue
        if len(t) < 1000:
            continue
        r = determinism_curve(t, f, float(np.median(np.diff(t))), seed=(x["kepid"] + 7919 * i) % 100000)
        for s in r["segs"]:
            s["file"] = os.path.basename(key)
        segs += r["segs"]
    Z = {}
    for lev in ("native", "coarse"):
        zs = [s[lev] for s in segs if lev in s]
        if zs:
            Z[lev] = [round(sum(z[k][3] for z in zs) / math.sqrt(len(zs)), 2) for k in ("comp", "npe", "asym")]
    return dict(id="kic:%d" % x["kepid"], kepid=x["kepid"], kepmag=x["kepmag"], status="ok", cad_s=1765.5, segs=segs, Z=Z)


def determinism_kepler(procs=10):
    """Every KOI host of ROOT/targets_kepler.json through determinism_one_kepler, one line a star into ROOT/stats_kepler.jsonl."""
    from concurrent.futures import ProcessPoolExecutor
    targets = json.load(open(os.path.join(determinism_ROOT, "targets_kepler.json")))
    done = determinism_done()
    todo = [x for x in targets if "kic:%d" % x["kepid"] not in done]
    path = os.path.join(determinism_ROOT, "stats_kepler.jsonl")
    print("%d KOI hosts to read, %d done" % (len(todo), len(targets) - len(todo)), flush=True)
    t0, m, miss = time.time(), 0, 0
    with ProcessPoolExecutor(procs, initializer=determinism_pool_init, initargs=(False,)) as ex, open(path, "a", encoding="utf-8") as fh:
        for r in ex.map(determinism_one_kepler, todo, chunksize=1):
            if r is None:
                miss += 1
                continue
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            m += 1
            if m % 20 == 0:
                fh.flush()
                print("%d of %d, %.1f stars a minute, %d to read again" % (m, len(todo), 60 * m / (time.time() - t0), miss), flush=True)
    print("%d stars written, %d to read again" % (m, miss), flush=True)
    return miss


def determinism_check(gpu=False):
    """The statistics on synthetic segments of two-minute cadence: white noise and red noise (AR(1)) as the null, a chaotic signal (the logistic map's orbit, held for ten cadences a step) and a coded on-off signal (a repeating 63-step maximal-length sequence) at a tenth of the noise and a third of it, and flares on red noise."""
    dev = None
    if gpu:
        import torch
        dev = torch.device("cuda")
    rng = np.random.default_rng(5)
    N = 9000
    ar = np.zeros(N)
    e = rng.normal(0, 1, N)
    for i in range(1, N):
        ar[i] = 0.9 * ar[i - 1] + e[i]
    ar /= ar.std()
    lm = np.empty(N // 10 + 1)
    lm[0] = 0.4
    for i in range(1, len(lm)):
        lm[i] = 3.99 * lm[i - 1] * (1 - lm[i - 1])
    chaos = np.repeat(lm - lm.mean(), 10)[:N] / lm.std()
    reg = [1, 0, 0, 0, 0, 0]
    seq = []
    for _ in range(63):
        seq.append(reg[-1])
        reg = [reg[0] ^ reg[-1]] + reg[:-1]
    code = np.tile(np.repeat(np.array(seq, float) * 2 - 1, 8), N // (63 * 8) + 1)[:N]
    flares = ar.copy()
    for t0 in rng.choice(N - 60, 12, replace=False):
        flares[t0:t0 + 60] += 8 * np.exp(-np.arange(60) / 8.0)
    cases = [("white noise", rng.normal(0, 1, N)), ("red noise", ar)]
    for amp in (0.33, 1.0, 3.0):
        cases += [("chaos at %.2f of the noise" % amp, rng.normal(0, 1, N) + amp * chaos),
                  ("code at %.2f of the noise" % amp, rng.normal(0, 1, N) + amp * code)]
    cases.append(("flares on red noise", flares))
    for name, x in cases:
        r = determinism_level(x, 11, dev)
        nb = len(x) // determinism_BIN
        c = determinism_level(x[:nb * determinism_BIN].reshape(nb, determinism_BIN).mean(1), 12, dev)
        print("%-27s z compressed, prediction, asymmetry: native %+7.2f %+7.2f %+7.2f; binned %+7.2f %+7.2f %+7.2f" % (
            name, r["comp"][3], r["npe"][3], r["asym"][3], c["comp"][3], c["npe"][3], c["asym"][3]), flush=True)


def determinism_report():
    """Every star-sector's combined z at both levels: the null's spread from the population, the candidates more compressible or more predictable by 5 sigma with their asymmetry, and the stars of the same sector and camera that share it."""
    rows = {}
    for p in sorted(glob.glob(os.path.join(determinism_ROOT, "stats_*.jsonl"))):
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("status") == "ok" and r.get("Z"):
                rows[r["id"]] = r
    R = list(rows.values())
    print("%d star-sectors with statistics" % len(R))
    out = []
    for lev in ("native", "coarse"):
        Z = np.array([r["Z"][lev] for r in R if lev in r["Z"]])
        if not len(Z):
            continue
        print("%s: %d; z median and robust scatter compressed %+.2f %.2f, prediction %+.2f %.2f, asymmetry %+.2f %.2f" % (
            lev, len(Z), np.median(Z[:, 0]), core.robust_sigma(Z[:, 0]), np.median(Z[:, 1]), core.robust_sigma(Z[:, 1]),
            np.median(Z[:, 2]), core.robust_sigma(Z[:, 2])))
        for thr in (3, 5, 8):
            print("   below -%d: compressed %d, prediction %d; asymmetry above %d: %d" % (
                thr, np.sum(Z[:, 0] < -thr), np.sum(Z[:, 1] < -thr), thr, np.sum(Z[:, 2] > thr)))
        for r in R:
            z = r["Z"].get(lev)
            if z and min(z[0], z[1]) < -5:
                out.append(dict(id=r["id"], level=lev, z=z, gaia=r.get("gaia"), plx=r.get("plx"), G=r.get("G"), cam=r.get("cam"),
                                sector=r.get("sector"), periods=[p for s in r["segs"] for p in s["periods"]],
                                flares=sum(s["flares"] for s in r["segs"])))
    for c in out:
        if c.get("sector") is not None:
            c["shared"] = sum(1 for o in out if o["id"] != c["id"] and o["level"] == c["level"] and o.get("sector") == c["sector"]
                              and o.get("cam") == c["cam"])
    out.sort(key=lambda c: min(c["z"][0], c["z"][1]))
    for c in out[:40]:
        print("  %-28s %-6s z %s plx %s G %s cam %s, %s others in its sector and camera; periods %s; flares %d" % (
            c["id"], c["level"], c["z"], c["plx"], c["G"], c["cam"], c.get("shared"), c["periods"][:3], c["flares"]))
    json.dump(dict(n=len(R), candidates=out), open(os.path.join(determinism_ROOT, "report.json"), "w"))


def determinism_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["targets", "stream", "local", "kepler_targets", "kepler", "check", "report"])
    p.add_argument("dir", nargs="?")
    p.add_argument("--lists")
    p.add_argument("--plx", type=float, default=10.0)
    p.add_argument("--part", default=None)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--tag", default="local")
    a = p.parse_args()
    if a.mode == "targets":
        determinism_targets(a.lists, a.plx)
    elif a.mode == "stream":
        determinism_stream(a.part, a.procs, a.gpu)
    elif a.mode == "local":
        determinism_local(a.dir, a.procs, a.tag)
    elif a.mode == "kepler_targets":
        determinism_kepler_targets()
    elif a.mode == "kepler":
        sys.exit(1 if determinism_kepler(a.procs) else 0)
    elif a.mode == "check":
        determinism_check(a.gpu)
    else:
        determinism_report()


psfflash_DOC = r"""PSF-shaped flashes in Kepler's cosmic-ray corrections: the flux the pipeline removed from single cadences of each target pixel file (the COSMIC_RAYS column), in cadences where three or more pixels of the optimal aperture were corrected together, correlated with the star's own pixel image; a flash from the star follows that image (correlation 0.9 or more, 0.2 above the image shifted by one pixel), a cosmic ray or its track does not; the null is the shifted image and the coincidences each pixel's own correction rate gives, events shared by other stars of the CCD channel at the same cadence are the spacecraft's, and light left in the next cadences marks a flare.

    python -I seti.py flashes psfflash targets   (every KOI host with a candidate or confirmed planet, brightest first, into ROOT/targets.json)
    python -I seti.py flashes psfflash stream [--procs 3]   (each star's long-cadence target pixel files read in memory, one line a star into ROOT/events.jsonl; exits 1 while stars remain unread)
    python -I seti.py flashes psfflash report
    python -I seti.py flashes psfflash record   (the PSF-shaped rate against the shifted-image null by stellar temperature; each star's PSF-shaped events its flares when the star is cooler than 5300 K, carries three or more or shows light in the next cadence, else cosmic rays shaped like its image by chance)"""

psfflash_ROOT = os.environ.get("PSFFLASH_ROOT", core.data("psfflash"))
psfflash_MIN_PIX = 3
psfflash_R = 0.9
psfflash_MARGIN = 0.2
psfflash_SHIFTS = ((1, 0), (-1, 0), (0, 1), (0, -1))
psfflash_TEFF_BINS = ((2500, 4000), (4000, 4800), (4800, 5300), (5300, 5800), (5800, 6300), (6300, 12000))
psfflash_FLARE_TEFF = 5300  # below it the PSF-shaped events outrun the null: cool dwarfs' flares
psfflash_FLARE_MIN = 3  # PSF-shaped events on one star, against about half an event the null gives a star over its quarters
psfflash_AFTER = 3.0  # light in the next cadence, in the light curve's sigma


def psfflash_targets():
    """Every KOI host with a candidate or confirmed planet, brightest first, into ROOT/targets.json."""
    os.makedirs(psfflash_ROOT, exist_ok=True)
    rows = core.exoplanets("SELECT kepid, koi_kepmag, koi_disposition FROM cumulative")
    best = {}
    for r in rows:
        if r["koi_disposition"] in ("CANDIDATE", "CONFIRMED") and num(r["kepid"]):
            best[int(r["kepid"])] = num(r["koi_kepmag"], 99.0)
    out = [dict(kepid=k, kepmag=v) for k, v in sorted(best.items(), key=lambda kv: kv[1])]
    json.dump(out, open(os.path.join(psfflash_ROOT, "targets.json"), "w"))
    print("%d KOI hosts" % len(out))


def psfflash_corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


def psfflash_quarter(raw):
    """Multi-pixel corrections of one gzipped long-cadence target pixel file: cadences where psfflash_MIN_PIX or more optimal-aperture pixels were corrected, each with its correlation to the median image and to the image shifted by one pixel, its share of the star's flux and the aperture flux in the next two cadences over the scatter; with the counts the per-pixel rates expect."""
    import gzip
    from astropy.io import fits
    with fits.open(io.BytesIO(gzip.decompress(raw))) as h:
        d = h[1].data
        quarter, channel = int(h[0].header.get("QUARTER", -1)), int(h[0].header.get("CHANNEL", -1))
        t, cad, q = np.asarray(d["TIME"], float), np.asarray(d["CADENCENO"], int), np.asarray(d["QUALITY"], int)
        F, C = np.asarray(d["FLUX"], float), np.asarray(d["COSMIC_RAYS"], float)
        ap = np.asarray(h[2].data, int)
    # cadences whose only flags are a cosmic ray in the optimal aperture (128), an impulsive outlier (1024) or a cosmic ray
    # in the collateral data (2048): every other flag marks the spacecraft's events
    good = ((q & ~(128 | 1024 | 2048)) == 0) & np.isfinite(t)
    t, cad, F, C = t[good], cad[good], F[good], np.nan_to_num(C[good])
    if len(t) < 500:
        return None
    with np.errstate(all="ignore"):
        M = np.nan_to_num(np.nanmedian(F, axis=0)).clip(0)
    core_ = ((ap & 2) > 0) & (M > 0)
    if core_.sum() < psfflash_MIN_PIX + 1:
        return None
    m = M[core_]
    total = float(m.sum())
    lc = np.nansum(np.where(core_[None, :, :], F, 0.0), axis=(1, 2))
    base = science.block_median(lc, 48)
    scat = core.robust_sigma(lc - base)
    hit = (C > 0) & core_[None, :, :]
    npix = hit.sum(axis=(1, 2))
    rate = hit.mean(axis=0)[core_]
    # cadences with psfflash_MIN_PIX or more corrected pixels if the pixels were hit independently at their own rates
    p = rate.clip(0, 1)
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for x in p:
        dist[1:] = dist[1:] * (1 - x) + dist[:-1] * x
        dist[0] *= 1 - x
    expect = float(len(t) * dist[psfflash_MIN_PIX:].sum())
    events = []
    shifted = [np.roll(M, s, axis=(0, 1))[core_] for s in psfflash_SHIFTS]
    for i in np.nonzero(npix >= psfflash_MIN_PIX)[0]:
        c = C[i][core_]
        r = psfflash_corr(c, m)
        rs = max(psfflash_corr(c, s) for s in shifted)
        after = [round(float((lc[j] - base[j]) / scat), 1) if j < len(lc) and scat > 0 else None for j in (i + 1, i + 2)]
        events.append(dict(cad=int(cad[i]), t=round(float(t[i]), 5), npix=int(npix[i]), r=round(r, 3), r_shift=round(rs, 3),
                           frac=round(float(c.sum()) / total, 6), after=after))
    return dict(q=quarter, ch=channel, n=int(len(t)), npix_ap=int(core_.sum()), n_hits=int(hit.sum()), n_multi=len(events),
                expect=round(expect, 3), scatter=round(float(scat / total), 7), events=events)


def psfflash_star(x):
    """One star's long-cadence target pixel files read in memory through psfflash_quarter; None when the listing or a file fails."""
    k9 = "%09d" % x["kepid"]
    keys = core.s3_keys("kepler/public/target_pixel_files/%s/%s/" % (k9[:4], k9), r"_lpd-targ\.fits\.gz$")
    if not keys:
        return None
    out = []
    for key in keys:
        b = core.get(core.S3 + key, timeout=300, retries=3, soft=True, pause=2)
        if not b:
            return None
        try:
            r = psfflash_quarter(b)
        except Exception:
            continue
        if r:
            out.append(r)
    return dict(kepid=x["kepid"], kepmag=x["kepmag"], quarters=out)


def psfflash_stream(procs=3):
    """Every target star through psfflash_star on procs processes, one line a star into ROOT/events.jsonl, past those read."""
    from concurrent.futures import ProcessPoolExecutor
    targets = json.load(open(os.path.join(psfflash_ROOT, "targets.json")))
    path = os.path.join(psfflash_ROOT, "events.jsonl")
    done = set()
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8", errors="replace"):
            try:
                done.add(json.loads(ln)["kepid"])
            except (ValueError, KeyError):
                pass
    todo = [x for x in targets if x["kepid"] not in done]
    print("%d stars to read, %d read" % (len(todo), len(done)), flush=True)
    t0, m, miss = time.time(), 0, 0
    with ProcessPoolExecutor(procs) as ex, open(path, "a", encoding="utf-8") as fh:
        for r in ex.map(psfflash_star, todo, chunksize=1):
            if r is None:
                miss += 1
                continue
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            m += 1
            if m % 10 == 0:
                fh.flush()
                print("%d of %d, %.1f stars a minute, %d to read again" % (m, len(todo), 60 * m / (time.time() - t0), miss), flush=True)
    print("%d stars read, %d to read again" % (m, miss), flush=True)
    return miss


def psfflash_report():
    """Multi-pixel corrections against the per-pixel expectation, PSF-shaped ones against the shifted image, shared cadences of a channel, and the flashes left with their after-light."""
    rows = [json.loads(ln) for ln in open(os.path.join(psfflash_ROOT, "events.jsonl"), encoding="utf-8")]
    nq = sum(len(r["quarters"]) for r in rows)
    multi = sum(qd["n_multi"] for r in rows for qd in r["quarters"])
    expect = sum(qd["expect"] for r in rows for qd in r["quarters"])
    ev = [dict(e, kepid=r["kepid"], kepmag=r["kepmag"], q=qd["q"], ch=qd["ch"]) for r in rows for qd in r["quarters"] for e in qd["events"]]
    shape = [e for e in ev if e["r"] >= psfflash_R and e["r"] - e["r_shift"] >= psfflash_MARGIN]
    null = [e for e in ev if e["r_shift"] >= psfflash_R and e["r_shift"] - e["r"] >= psfflash_MARGIN]
    by = {}
    for e in ev:
        by.setdefault((e["q"], e["ch"], e["cad"]), set()).add(e["kepid"])
    for e in shape:
        e["shared"] = len(by[(e["q"], e["ch"], e["cad"])]) - 1
    lone = [e for e in shape if e["shared"] == 0]
    print("%d stars, %d quarters; cadences with %d or more corrected aperture pixels %d against %.1f from the pixels' own rates; "
          "PSF-shaped %d, shaped like the shifted image instead %d; PSF-shaped and unshared on the channel %d"
          % (len(rows), nq, psfflash_MIN_PIX, multi, expect, len(shape), len(null), len(lone)))
    for e in sorted(lone, key=lambda e: -e["frac"])[:30]:
        print("  KIC %d (Kp %.1f) Q%d ch%d cadence %d BKJD %.4f: %d pixels, r %.3f against %.3f shifted, %.0f ppm of the star; "
              "next cadences %s sigma" % (e["kepid"], e["kepmag"], e["q"], e["ch"], e["cad"], e["t"], e["npix"], e["r"],
                                          e["r_shift"], e["frac"] * 1e6, e["after"]))
    json.dump(dict(n_stars=len(rows), n_quarters=nq, multi=multi, expect=expect, shaped=shape, null=len(null)),
              open(os.path.join(psfflash_ROOT, "report.json"), "w"))


def psfflash_teff(kepids):
    """DR25 stellar-table temperatures of the stars: the copy the Kepler levels search keeps, else the archive's table by id."""
    path = os.path.join(keplevels_ROOT, "targets.json")
    if os.path.exists(path):
        return {x["kepid"]: x["teff"] for x in json.load(open(path)) if x.get("teff")}
    teff = {}
    left = sorted(set(kepids))
    for i in range(0, len(left), 400):
        for r in core.exoplanets("SELECT kepid, teff FROM keplerstellar WHERE st_delivname = 'q1_q17_dr25_stellar' AND kepid IN (%s)"
                                 % ",".join(str(k) for k in left[i:i + 400])):
            if num(r["teff"]):
                teff[int(r["kepid"])] = num(r["teff"])
    return teff


def psfflash_record():
    """The PSF-shaped and shifted-image rates by stellar temperature, and each star's unshared PSF-shaped events judged as its flares or as cosmic rays shaped like its image."""
    rows = [json.loads(ln) for ln in open(os.path.join(psfflash_ROOT, "events.jsonl"), encoding="utf-8")]
    rep = json.load(open(os.path.join(psfflash_ROOT, "report.json")))
    teff = psfflash_teff([r["kepid"] for r in rows])
    nq = {r["kepid"]: len(r["quarters"]) for r in rows}
    ev = [dict(e, kepid=r["kepid"], kepmag=r["kepmag"], q=qd["q"], ch=qd["ch"]) for r in rows for qd in r["quarters"] for e in qd["events"]]
    shape = [e for e in ev if e["r"] >= psfflash_R and e["r"] - e["r_shift"] >= psfflash_MARGIN]
    null = [e for e in ev if e["r_shift"] >= psfflash_R and e["r_shift"] - e["r"] >= psfflash_MARGIN]
    ratios = []
    for lo, hi in psfflash_TEFF_BINS:
        ks = {k for k in nq if lo <= teff.get(k, -1) < hi}
        a = sum(1 for e in shape if e["kepid"] in ks)
        b = sum(1 for e in null if e["kepid"] in ks) / len(psfflash_SHIFTS)
        ratios.append("%d-%d K %.1f" % (lo, hi, a / b if b else float("nan")))
    by = {}
    for e in ev:
        by.setdefault((e["q"], e["ch"], e["cad"]), set()).add(e["kepid"])
    lone = [e for e in shape if len(by[(e["q"], e["ch"], e["cad"])]) == 1]
    per = {}
    for e in lone:
        per.setdefault(e["kepid"], []).append(e)
    out = []
    for k, es in per.items():
        t = teff.get(k)
        after = sum(1 for e in es if any(a is not None and a > psfflash_AFTER for a in e["after"]))
        flare = (t is not None and t < psfflash_FLARE_TEFF) or len(es) >= psfflash_FLARE_MIN or after > 0
        big = max(es, key=lambda e: e["frac"])
        out.append(dict(target="kic:%d" % k, stat=float(len(es)), unit="unshared PSF-shaped single-cadence corrections",
                        vet="flare" if flare else "cosmic-ray",
                        note="%s; %d of them with light in the next cadence; the largest %.0f ppm of the star, Q%d cadence %d; %s" % (
                            "Teff %d K" % t if t else "no Teff", after, big["frac"] * 1e6, big["q"], big["cad"],
                            "the star's flares" if flare else "a cosmic ray shaped like the star's image by chance, at the shifted-image rate")))
    n_flare = sum(1 for o in out if o["vet"] == "flare")
    core.record_search(
        "kepler_psf_flashes", "PSF-shaped flashes in pixel data",
        archive="Kepler DR25 long-cadence target pixel files (STScI AWS mirror)", release="DR25",
        selection="every KOI host with a candidate or confirmed planet: %d stars, %d quarters" % (rep["n_stars"], rep["n_quarters"]),
        code=os.path.abspath(__file__), entry="psfflash", n_targets=rep["n_stars"], n_unit="stars",
        outputs=[os.path.join(psfflash_ROOT, "events.jsonl"), os.path.join(psfflash_ROOT, "report.json")],
        params=dict(min_pixels=psfflash_MIN_PIX, r=psfflash_R, margin=psfflash_MARGIN, flare_teff=psfflash_FLARE_TEFF,
                    flare_min=psfflash_FLARE_MIN, after_sigma=psfflash_AFTER),
        null_summary="corrections shaped like the star's image shifted by one pixel, the most like of four shifts: %d, against %d PSF-shaped; "
                     "PSF-shaped over a quarter of the shifted-image count by temperature: %s" % (len(null), len(shape), ", ".join(ratios)),
        sensitivity="a brightening in one 29.4-minute cadence over three or more pixels of the optimal aperture that the pipeline removed as a cosmic ray",
        ledger="PSF-shaped flashes in pixel data",
        result="%d cadences with three or more corrected aperture pixels against %.0f from the pixels' own rates; %d PSF-shaped, %d of them "
               "unshared on the channel, on %d stars: %d stars' flares, %d at the chance rate" % (
                   rep["multi"], rep["expect"], len(shape), len(lone), len(out), n_flare, len(out) - n_flare))
    core.clear_targets("kepler_psf_flashes")
    core.record_targets("kepler_psf_flashes", out)
    print("recorded %d stars with unshared PSF-shaped events: %d flaring, %d at the chance rate; PSF-shaped over the null by temperature: %s"
          % (len(out), n_flare, len(out) - n_flare, ", ".join(ratios)))


def psfflash_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["targets", "stream", "report", "record"])
    p.add_argument("--procs", type=int, default=3)
    a = p.parse_args()
    if a.mode == "targets":
        psfflash_targets()
    elif a.mode == "stream":
        sys.exit(1 if psfflash_stream(a.procs) else 0)
    elif a.mode == "record":
        psfflash_record()
    else:
        psfflash_report()


etz_DOC = r"""Earth-transit-zone stars at opposition in TESS sectors 42-46: flashes within 6.5 h of opposition against the rest and off-ecliptic controls.

    python -I seti.py flashes etz [--threads 8]
    python -I seti.py flashes etz record"""

etz_ROOT = os.environ.get("ETZ_ROOT", core.data("etz"))
etz_SECTORS = tuple(int(x) for x in os.environ.get("ETZ_SECTORS", "42,43,44,45,46").split(","))
etz_RUN = "tess_etz_opposition" + ("" if etz_SECTORS == (42, 43, 44, 45, 46) else "_%d_%d" % (etz_SECTORS[0], etz_SECTORS[-1]))


def etz_sun_lon(jd):
    T = (jd - 2451545.0) / 36525
    L = (280.46646 + 36000.76983 * T) % 360
    M = math.radians((357.52911 + 35999.05029 * T) % 360)
    return (L + (1.914602 - 0.004817 * T) * math.sin(M) + 0.019993 * math.sin(2 * M)) % 360


def etz_opposition(lon, t0, t1):
    """BTJD in [t0, t1] when the Sun's longitude is the star's plus 180, or None."""
    ts = np.arange(t0, t1, 0.01)
    d = np.array([((etz_sun_lon(t + 2457000.0) - (lon + 180.0)) + 180) % 360 - 180 for t in ts])
    k = np.nonzero((d[:-1] <= 0) & (d[1:] > 0))[0]
    if not len(k):
        return None
    i = k[0]
    return float(ts[i] + 0.01 * (-d[i]) / (d[i + 1] - d[i]))


def etz_tasks():
    from scipy.spatial import cKDTree
    os.makedirs(etz_ROOT, exist_ok=True)
    gp = os.path.join(etz_ROOT, "gaia100pc_ecl5.json")
    if not os.path.exists(gp):
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = -1
        t = Gaia.launch_job_async("SELECT source_id, ra, dec, ecl_lon, ecl_lat, parallax, pmra, pmdec, phot_g_mean_mag FROM "
                                  "gaiadr3.gaia_source WHERE parallax > 10 AND parallax_over_error > 10 AND ABS(ecl_lat) < 5 "
                                  "AND phot_g_mean_mag < 15").get_results()
        json.dump([[str(r["source_id"]), float(r["ra"]), float(r["dec"]), float(r["ecl_lon"]), float(r["ecl_lat"]),
                    float(r["parallax"]), num(r["pmra"]) or 0.0, num(r["pmdec"]) or 0.0, float(r["phot_g_mean_mag"])] for r in t],
                  open(gp, "w"))
    g = json.load(open(gp))
    ra = np.array([x[1] for x in g])
    de = np.array([x[2] for x in g])
    ra0 = ra - np.array([x[6] for x in g]) / 3.6e6 * 16 / np.cos(np.radians(de))  # back to the TIC epoch 2000
    de0 = de - np.array([x[7] for x in g]) / 3.6e6 * 16

    def xyz(r, d):
        r, d = np.radians(r), np.radians(d)
        return np.vstack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]).T
    tree = cKDTree(xyz(ra0, de0))
    out = []
    for s in etz_SECTORS:
        tl = core.get("https://tess.mit.edu/public/target_lists/2m/all_targets_S%03d_v1.csv" % s, timeout=300, soft=True)
        if tl is None:
            tl = core.get("https://tess.mit.edu/wp-content/uploads/all_targets_S%03d_v1.csv" % s, timeout=300)
        rows = list(csv.reader([ln for ln in (tl.decode("latin-1") if isinstance(tl, bytes) else tl).splitlines()
                                if ln and not ln.startswith("#")]))
        ix = {h.strip(): i for i, h in enumerate(rows[0])}
        sc = core.get("https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_%d_lc.sh" % s, timeout=600)
        urls = {int(m.group(2)): m.group(1) for m in re.finditer(r"(https://\S+-s%04d-(\d{16})-\d{4}-s_lc\.fits)" % s,
                                                                 sc.decode() if isinstance(sc, bytes) else sc)}
        tic = np.array([int(r[ix["TICID"]]) for r in rows[1:]])
        d, i = tree.query(xyz(np.array([float(r[ix["RA"]]) for r in rows[1:]]), np.array([float(r[ix["Dec"]]) for r in rows[1:]])),
                          distance_upper_bound=np.radians(20 / 3600.0))
        for k in np.nonzero(np.isfinite(d))[0]:
            x = g[i[k]]
            if int(tic[k]) in urls and (abs(x[4]) < 0.27 or 1 < abs(x[4]) < 5):
                out.append(dict(sector=s, tic=int(tic[k]), gaia=x[0], ecl_lon=x[3], ecl_lat=x[4], g=x[8],
                                url=urls[int(tic[k])], etz=abs(x[4]) < 0.27))
    import random
    random.seed(4)
    ctl = [o for o in out if not o["etz"]]
    return [o for o in out if o["etz"]] + random.sample(ctl, min(1500, len(ctl)))


def etz_one(task):
    from astropy.io import fits
    from scipy.ndimage import median_filter
    p = os.path.join(etz_ROOT, "%d_%d.json" % (task["sector"], task["tic"]))
    if os.path.exists(p):
        return json.load(open(p))
    try:
        b = core.get(task["url"], timeout=300)
        with fits.open(io.BytesIO(b)) as h:
            d = h[1].data
            t, f, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"], int)
    except Exception as e:
        return dict(task, error=str(e)[:100])
    good = (q == 0) & np.isfinite(f) & np.isfinite(t)
    t, f = t[good], f[good]
    if len(t) < 2000:
        return dict(task, error="short")
    r = f / median_filter(f, size=61, mode="nearest") - 1.0
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    lone = (r > 5 * sig) & (np.r_[r[1:], 0] < 2 * sig) & (np.r_[0, r[:-1]] < 2 * sig)
    multi = (r > 5 * sig) & ~lone
    to = etz_opposition(task["ecl_lon"], t.min(), t.max())
    win = np.abs(t - to) < 6.5 / 24 if to is not None else np.zeros(len(t), bool)
    res = dict(task, t_opp=to, n=int(len(t)), n_win=int(win.sum()), sig=float(sig), lone=int(lone.sum()),
               lone_win=int((lone & win).sum()), multi=int(multi.sum()), multi_win=int((multi & win).sum()),
               max_win=float(np.max(r[win]) / sig) if win.any() else None, lone_times=[float(x) for x in t[lone & win]])
    json.dump(res, open(p, "w"))
    return res


def etz_summary(res):
    out = {}
    for lab, sel in (("etz", True), ("control", False)):
        rs = [r for r in res if "error" not in r and r["etz"] == sel and r["t_opp"] is not None and r["n_win"] > 100]
        out[lab] = dict(windows=len(rs), stars=len({r["gaia"] for r in rs}), lone_win=sum(r["lone_win"] for r in rs),
                        n_win=sum(r["n_win"] for r in rs), lone_out=sum(r["lone"] - r["lone_win"] for r in rs),
                        n_out=sum(r["n"] - r["n_win"] for r in rs), multi_win=sum(r["multi_win"] for r in rs),
                        multi_out=sum(r["multi"] - r["multi_win"] for r in rs), rows=rs)
    return out


def etz_main(threads):
    tasks = etz_tasks()
    print("%d light curves: %d transit-zone, %d control" % (len(tasks), sum(t["etz"] for t in tasks), sum(not t["etz"] for t in tasks)),
          flush=True)
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(etz_one, tasks))
    s = etz_summary(res)
    for lab in ("etz", "control"):
        x = s[lab]
        print("%s: %d windows on %d stars; lone flashes %d in %d window cadences against %d in %d outside (%.2f expected); "
              "brightenings %d in the windows against %.2f expected"
              % (lab, x["windows"], x["stars"], x["lone_win"], x["n_win"], x["lone_out"], x["n_out"],
                 x["lone_out"] / max(x["n_out"], 1) * x["n_win"], x["multi_win"], x["multi_out"] / max(x["n_out"], 1) * x["n_win"]))
    json.dump(res, open(core.results("etz_opposition.json"), "w"))


def etz_record():
    res = json.load(open(core.results("etz_opposition.json")))
    s = etz_summary(res)
    e, c = s["etz"], s["control"]
    exp_l = e["lone_out"] / max(e["n_out"], 1) * e["n_win"]
    exp_m = e["multi_out"] / max(e["n_out"], 1) * e["n_win"]
    core.record_search(
        etz_RUN, "Earth-transit-zone stars at opposition", archive="MAST TESS SPOC 2-minute light curves",
        release="sectors %d-%d" % (etz_SECTORS[0], etz_SECTORS[-1]), selection="Gaia DR3 stars within 100 pc and 0.27 deg of the ecliptic, G below 15, with "
                                           "2-minute data and their opposition inside the sector",
        code=os.path.abspath(__file__), entry="etz", n_targets=e["stars"], n_unit="stars",
        params=dict(window_h=6.5, lone_sigma=5, median_cadences=61, etz_deg=0.27, control_deg=[1, 5]),
        null_summary="the same stars outside the window (%.2f lone flashes and %.2f brightenings expected in it); control stars "
                     "1-5 deg off the ecliptic: %d lone flashes in %d window cadences" % (exp_l, exp_m, c["lone_win"], c["n_win"]),
        sensitivity="a one-cadence flash at 5 sigma of each light curve's 2-minute scatter",
        ledger="Earth-transit-zone stars at opposition",
        result="%d transit windows on %d stars: %d lone flashes against %.2f expected and %d brightenings against %.2f"
               % (e["windows"], e["stars"], e["lone_win"], exp_l, e["multi_win"], exp_m))
    core.clear_targets(etz_RUN)
    core.record_targets(etz_RUN, [dict(
        target="gaia_dr3:%s" % r["gaia"], stat=float(r["lone_win"]), unit="lone flashes within 6.5 h of opposition",
        note="sector %d, TIC %d, opposition BTJD %.3f, %d window cadences, peak %.1f sigma" % (r["sector"], r["tic"], r["t_opp"],
                                                                                            r["n_win"], r["max_win"] or 0))
        for r in e["rows"]])
    print("recorded")


def etz_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        etz_record()
        return
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    etz_main(a.threads)


nova_DOC = r"""Nova-ellipsoid timing: TESS flashes of stars within 100 pc at their arrival dates keyed to T CrB 1866 and 1946 (or another eruption), against the rest of their light curves.

    python -I seti.py flashes nova [--threads 8]
    python -I seti.py flashes nova record"""

nova_ROOT = os.environ.get("NOVA_ROOT", core.data("novae"))
# name -> (RA, Dec deg, parallax mas, [(shell, epoch yr)])
nova_EVENTS = {
    "tcrb": (239.87565, 25.92026, 1.091953, [("1946", 1946.108), ("1866", 1866.363)]),        # Gaia DR3 1220110705972528512
    "gkper": (52.80000, 43.90417, 2.26, [("gkper_1901", 1901.140)]),
    "v603aql": (282.22750, 0.58417, 3.17, [("v603aql_1918", 1918.433)]),
    "rrpic": (98.90042, -62.64000, 2.0, [("rrpic_1925", 1925.395)]),
    "dqher": (271.87625, 45.85917, 2.0, [("dqher_1934", 1934.945)]),
    "cppup": (122.94208, -35.35139, 1.27, [("cppup_1942", 1942.852)]),
    "v1500cyg": (317.90250, 48.15056, 0.67, [("v1500cyg_1975", 1975.658)]),
    "sn1572": (6.33958, 64.14083, 0.4, [("sn1572", 1572.877)]),
    "sn1604": (262.64992, -21.48228, 0.2, [("sn1604", 1604.773)]),
}
nova_EVENT = os.environ.get("NOVA_EVENT", "tcrb")
nova_TCRB = nova_EVENTS[nova_EVENT][:3]
nova_ERUPTIONS = tuple(nova_EVENTS[nova_EVENT][3])
nova_TAG = "" if nova_EVENT == "tcrb" else "_" + nova_EVENT


def nova_gaia():
    """Gaia DR3 stars within 100 pc, G below 13, parallax at 20 sigma, cached."""
    p = os.path.join(nova_ROOT, "gaia_100pc_g13.json")
    if os.path.exists(p):
        return json.load(open(p))
    rows = []
    for lo, hi in ((10, 11.5), (11.5, 13), (13, 15), (15, 18), (18, 25), (25, 1000)):
        rows += core.datalab("SELECT source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag FROM gaia_dr3.gaia_source "
                                 "WHERE parallax >= %g AND parallax < %g AND parallax_over_error > 20 AND phot_g_mean_mag < 13" % (lo, hi))
    out = [[str(r["source_id"]), float(r["ra"]), float(r["dec"]), float(r["parallax"]), float(r["parallax_error"]),
            float(r["phot_g_mean_mag"])] for r in rows]
    os.makedirs(nova_ROOT, exist_ok=True)
    json.dump(out, open(p, "w"))
    return out


def nova_targets():
    p = os.path.join(nova_ROOT, "nova_targets%s.json" % nova_TAG)
    if os.path.exists(p):
        return json.load(open(p))
    from astropy.table import Table
    from astroquery.xmatch import XMatch
    import astropy.units as u
    G = nova_gaia()
    sid = np.array([int(x[0]) for x in G], dtype=np.int64)
    ra, de = np.radians(np.array([x[1] for x in G])), np.radians(np.array([x[2] for x in G]))
    gmag = np.array([x[5] for x in G])
    plx, plx_err = np.array([x[3] for x in G]), np.array([x[4] for x in G])
    r = 1000 / plx
    sr = r * plx_err / plx

    def uv(a, d):
        return np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    dn = 1000 / nova_TCRB[2]
    un = uv(math.radians(nova_TCRB[0]), math.radians(nova_TCRB[1]))
    xs = (uv(ra, de) * r).T
    D = np.linalg.norm(xs - dn * un, axis=1) + r - dn
    cos = (xs @ un) / r
    rows = []
    for name, t0 in nova_ERUPTIONS:
        arr = t0 + D * 3.26156
        sig = np.abs(1 - cos) * sr * 3.26156
        for k in np.nonzero((arr > 2018.55) & (arr < 2026.5))[0]:
            rows.append((name, str(sid[k]), float(np.degrees(ra[k])), float(np.degrees(de[k])), float(gmag[k]),
                         float(arr[k]), float(sig[k])))
    if not rows:
        os.makedirs(nova_ROOT, exist_ok=True)
        json.dump([], open(p, "w"))
        return []
    t = Table(rows=[(i, x[2], x[3]) for i, x in enumerate(rows)], names=("idx", "ra", "dec"))
    m = XMatch.query(cat1=t, cat2="vizier:IV/39/tic82", max_distance=6 * u.arcsec, colRA1="ra", colDec1="dec")
    best = {}
    for x in m:
        i = int(x["idx"])
        dg = abs(float(x["Gmag"]) - rows[i][4]) if not np.ma.is_masked(x["Gmag"]) else 0.5
        if i not in best or (dg, float(x["angDist"])) < best[i][0]:
            best[i] = ((dg, float(x["angDist"])), int(x["TIC"]))
    out = [dict(shell=x[0], gaia=x[1], ra=x[2], dec=x[3], g=x[4], arrival=x[5], sig=x[6], tic=best[i][1])
           for i, x in enumerate(rows) if i in best and best[i][0][0] < 1.0]
    os.makedirs(nova_ROOT, exist_ok=True)
    json.dump(out, open(p, "w"))
    return out


def nova_year(stamp):
    import datetime
    d = datetime.datetime.strptime(stamp[:7], "%Y%j") + datetime.timedelta(hours=int(stamp[7:9]), minutes=int(stamp[9:11]))
    y0 = datetime.datetime(d.year, 1, 1)
    return d.year + (d - y0).total_seconds() / (datetime.datetime(d.year + 1, 1, 1) - y0).total_seconds()


def nova_one(t):
    from astropy.io import fits
    from scipy.ndimage import median_filter
    p = os.path.join(nova_ROOT, "lc", "%s_%d_%d.json" % (t["shell"], t["sector"], t["tic"]))
    if os.path.exists(p):
        return json.load(open(p))
    try:
        with fits.open(io.BytesIO(core.get(t["url"], timeout=300))) as h:
            d = h[1].data
            tt, f, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"], int)
    except Exception as e:
        return dict(t, error=str(e)[:80])
    ok = (q == 0) & np.isfinite(f) & np.isfinite(tt)
    tt, f = tt[ok], f[ok]
    if len(tt) < 2000:
        return dict(t, error="short")
    r = f / median_filter(f, size=61, mode="nearest") - 1.0
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    lone = (r > 5 * sig) & (np.r_[r[1:], 0] < 2 * sig) & (np.r_[0, r[:-1]] < 2 * sig)
    yr = 2014.0 + (tt + 2457000.0 - 2456658.5) / 365.25
    win = np.abs(yr - t["arrival"]) <= 2 * t["sig"]
    res = dict(t, n=int(len(tt)), n_win=int(win.sum()), lone=int(lone.sum()), lone_win=int((lone & win).sum()),
               lone_times=[float(x) for x in tt[lone & win]], max_win=float(np.max(r[win]) / sig) if win.any() else None)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, "w"))
    return res


def nova_main(threads):
    targets = nova_targets()
    urls, start = {}, {}
    for s in range(1, 100):
        p = os.path.join(nova_ROOT, "curl_%03d.sh" % s)
        if not os.path.exists(p):
            try:
                b = core.get("https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_%d_lc.sh" % s, timeout=600)
            except Exception:
                continue
            open(p, "wb").write(b if isinstance(b, bytes) else b.encode("utf-8"))
        txt = open(p, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"(https://\S+/(tess(\d{13})-s%04d-(\d{16})-\d{4}-s_lc\.fits))" % s, txt):
            urls[(s, int(m.group(4)))] = m.group(1)
            start.setdefault(s, nova_year(m.group(3)))
    sect = sorted(start)
    end = {s: (start[sect[i + 1]] if i + 1 < len(sect) else start[s] + 27.4 / 365.25) for i, s in enumerate(sect)}
    tasks = []
    for t in targets:
        for s in sect:
            if (s, t["tic"]) in urls:
                tasks.append(dict(t, sector=s, url=urls[(s, t["tic"])], t0=start[s], t1=end[s],
                                  in_window=bool(start[s] - 2 * t["sig"] <= t["arrival"] <= end[s] + 2 * t["sig"])))
    keep = {(t["shell"], t["tic"]) for t in tasks if t["in_window"]}
    tasks = [t for t in tasks if (t["shell"], t["tic"]) in keep]
    print("%d sectors; %d star-sectors of %d stars with a sector over their arrival" % (len(sect), len(tasks), len(keep)), flush=True)
    with ThreadPoolExecutor(threads) as ex:
        res = [r for r in ex.map(nova_one, tasks) if "error" not in r]
    json.dump(res, open(core.results("nova_ellipsoid%s.json" % nova_TAG), "w"))
    for name, _ in nova_ERUPTIONS:
        s = nova_summary(res, name)
        print("%s eruption: %d stars (G 6 or fainter) observed at arrival in %d light curves; capped lone flashes %d against %.1f "
              "from the same sectors; light curves with a flash %d against %.1f" % (name, s["stars"], s["lcs"], s["lone_win"],
                                                                                   s["expected"], s["any_obs"], s["any_exp"]))


def nova_summary(res, name, gmin=6.0, cap=3):
    """Window flashes against the out-of-window rate; G below 6 excluded, counts capped at 3 per light curve."""
    import collections
    rs = [r for r in res if r["shell"] == name]
    raw = dict(n=sum(r["lone_win"] for r in rs))
    rs = [r for r in rs if r["g"] >= gmin]
    n_s, l_s = collections.Counter(), collections.Counter()
    for r in rs:
        n_s[r["sector"]] += r["n"] - r["n_win"]
        l_s[r["sector"]] += min(r["lone"] - r["lone_win"], cap * max(1, round((r["n"] - r["n_win"]) / 15000)))
    win = [r for r in rs if r["n_win"] > 0]
    rate = {s_: (l_s[s_] / n_s[s_] if n_s[s_] else 0.0) for s_ in n_s}
    obs = sum(min(r["lone_win"], cap) for r in win)
    exp = sum(r["n_win"] * rate.get(r["sector"], 0.0) for r in win)
    any_obs = sum(1 for r in win if r["lone_win"] > 0)
    any_exp = sum(1 - math.exp(-r["n_win"] * rate.get(r["sector"], 0.0)) for r in win)
    return dict(stars=len({r["tic"] for r in win}), lcs=len(win), lone_win=obs, expected=exp, any_obs=any_obs, any_exp=any_exp,
                raw=raw["n"], rows=win)


def nova_record():
    res = json.load(open(core.results("nova_ellipsoid%s.json" % nova_TAG)))
    S = [(name, nova_summary(res, name)) for name, _ in nova_ERUPTIONS]
    run = "tcrb_nova_ellipsoid" if nova_EVENT == "tcrb" else "nova_ellipsoid_" + nova_EVENT
    core.record_search(
        run, "Nova ellipsoids", archive="MAST TESS SPOC 2-minute light curves",
        release="sectors 1-99", selection="Gaia DR3 stars within 100 pc, G below 13, on the %s ellipsoid during the TESS "
                                          "years, with 2-minute data over their arrival" % " or ".join(n for n, _ in nova_ERUPTIONS),
        code=os.path.abspath(__file__), entry="nova", n_targets=sum(s["stars"] for _, s in S),
        params=dict(window_sigma=2, lone_sigma=5, event=nova_EVENT, ra=nova_TCRB[0], dec=nova_TCRB[1], parallax_mas=nova_TCRB[2],
                    eruptions=[e for _, e in nova_ERUPTIONS]),
        null_summary="the same sectors' light curves outside their windows give the rate; stars brighter than G 6 are left out "
                     "and each light curve's count is capped at 3; with them, the raw window counts are %s"
                     % ", ".join("%d" % s["raw"] for _, s in S),
        sensitivity="a one-cadence flash at 5 sigma of each light curve's 2-minute scatter within the arrival window",
        ledger="Nova ellipsoids",
        result="; ".join("%s: %d stars at arrival, %d capped lone flashes against %.1f and %d light curves with a flash against %.1f"
                         % (name, s["stars"], s["lone_win"], s["expected"], s["any_obs"], s["any_exp"]) for name, s in S))
    core.clear_targets(run)
    core.record_targets(run, [dict(
        target="gaia_dr3:%s" % r["gaia"], stat=float(r["lone_win"]), unit="lone flashes within 2 sigma of the arrival",
        note="%s, arrival %.3f +- %.3f yr, sector %d, %d window cadences" % (r["shell"], r["arrival"], r["sig"], r["sector"],
                                                                           r["n_win"]))
        for _, s in S for r in s["rows"] if r["n_win"] > 0])
    print("recorded %s" % run)


def nova_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        nova_record()
        return
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    nova_main(a.threads)


clocks_DOC = r"""Shared clocks: ZTF periodic variables pairing in period within a fractional tolerance (counted as well at a tenth and three tenths of it) and in phase at a reference epoch, against the phases shuffled among the stars.

    python -I seti.py flashes clocks [record] [--eps 1e-5] [--phase 0.01] [--tref 58700] [--min-sep-arcsec 10] [--shuffles 20]"""


def clocks_main(record=False, eps=1e-5, phase=0.01, tref=58700.0, min_sep_arcsec=10.0, shuffles=20):
    from astropy.table import Table
    p = core.data("ztf_var", "ztf_periodic.fits")
    t = Table.read(p)
    P = np.array(np.ma.filled(t["Per"].astype(float), np.nan))
    T0 = np.array(np.ma.filled(t["T0"].astype(float), np.nan))
    ra = np.radians(np.array(t["RAJ2000"], float))
    de = np.radians(np.array(t["DEJ2000"], float))
    ok = np.isfinite(P) & np.isfinite(T0) & (P > 0)
    alias = np.zeros(len(P), bool)
    for base in (1.0, 0.99726957, 29.530589, 365.25):
        for n in range(1, 11):
            for k in range(1, 5):
                alias |= np.abs(P / (base * n / k) - 1) < 1e-3
    keep = np.nonzero(ok & ~alias)[0]
    lp = np.log(P[keep])
    o = np.argsort(lp)
    lp, idx = lp[o], keep[o]
    xyz = np.vstack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]).T

    def pairs(eps):
        out = []
        j = 0
        for i in range(len(lp)):
            j = max(j, i + 1)
            while j < len(lp) and lp[j] - lp[i] < eps:
                j += 1
            for k in range(i + 1, j):
                out.append((idx[i], idx[k]))
        return out
    counts = {}
    sep10 = 2 * math.sin(math.radians(min_sep_arcsec / 3600.0) / 2)
    for tol in (eps / 10, eps * 0.3, eps):
        pr = pairs(tol)
        a = np.array(pr, dtype=np.int64).reshape(-1, 2)
        far = np.linalg.norm(xyz[a[:, 0]] - xyz[a[:, 1]], axis=1) > sep10 if len(a) else np.zeros(0, bool)
        counts[tol] = int(far.sum())
        if tol == eps:
            base = a[far]
    print("%d variables with periods off the aliases (of %d); pairs within the fractional period tolerance: %s"
          % (len(keep), len(P), ", ".join("%.0e: %d (%.3g per unit)" % (e, c, c / e) for e, c in counts.items())), flush=True)
    ph = ((tref - T0) / P) % 1.0

    def coherent(phases):
        d = np.abs(phases[base[:, 0]] - phases[base[:, 1]])
        d = np.minimum(d, 1 - d)
        return base[d < phase]

    def groups(cp):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a_, b_ in cp:
            parent[find(int(a_))] = find(int(b_))
        g = {}
        for a_, b_ in cp:
            for x in (int(a_), int(b_)):
                g.setdefault(find(x), set()).add(x)
        return [sorted(v) for v in g.values()]
    cp = coherent(ph)
    gr = groups(cp)
    rng = np.random.default_rng(5)
    nulls = []
    for k in range(shuffles):
        sh = ph.copy()
        sh[keep] = rng.permutation(ph[keep])
        c2 = coherent(sh)
        g2 = groups(c2)
        nulls.append((len(c2), sum(1 for g in g2 if len(g) >= 3), max([len(g) for g in g2] or [0])))
    big = sorted([g for g in gr if len(g) >= 3], key=len, reverse=True)
    # an exactly repeated catalogue period is a grid, not a measurement
    from collections import Counter
    reps = Counter(np.round(P[keep], 6).tolist())
    grid = {round(float(P[g[0]]), 6) for g in big if reps[round(float(P[g[0]]), 6)] >= 10}
    print("in phase within %g at MJD %.0f: %d pairs, %d groups of 3 or more (largest %d); phases shuffled: pairs %s, groups %s, "
          "largest %s" % (phase, tref, len(cp), len(big), len(big[0]) if big else 0, [n[0] for n in nulls[:8]], [n[1] for n in nulls[:8]],
                          [n[2] for n in nulls[:8]]))
    for g in big[:10]:
        print("  group of %d at P %.6f d: %s" % (len(g), P[g[0]], ", ".join("%s (%s)" % (t["ID"][i], t["Type"][i]) for i in g[:6])))
    nongrid = [g for g in big if round(float(P[g[0]]), 6) not in grid]
    print("groups at a period the catalogue repeats exactly (its grid): %d; the largest of the rest %d, against the shuffled "
          "largest %d" % (len(big) - len(nongrid), len(nongrid[0]) if nongrid else 0, max(n[2] for n in nulls)))
    res = dict(n=len(keep), counts={str(k): v for k, v in counts.items()}, pairs=len(cp), groups=[[int(i) for i in g] for g in big],
               null_pairs=[n[0] for n in nulls], null_groups=[n[1] for n in nulls], null_largest=[n[2] for n in nulls])
    json.dump(res, open(core.results("shared_clocks.json"), "w"))
    if record:
        core.record_search(
            "ztf_shared_clocks", "Shared clocks", archive="ZTF periodic variables (Chen et al. 2020, J/ApJS/249/18)",
            selection="%d variables with periods off the n/k-day, sidereal-day, lunar and annual aliases" % len(keep),
            code=os.path.abspath(__file__), entry="clocks", n_targets=len(keep),
            params=dict(eps=eps, phase=phase, tref=tref, min_sep_arcsec=min_sep_arcsec, shuffles=shuffles),
            null_summary="phases shuffled among the stars, %d times: %.0f +- %.0f pairs in phase, groups of 3 or more %.1f, largest "
                         "%d" % (shuffles, np.mean([n[0] for n in nulls]), np.std([n[0] for n in nulls]), np.mean([n[1] for n in nulls]),
                                 max(n[2] for n in nulls)),
            sensitivity="three or more variables sharing a period to %g and a phase to %g cycle" % (eps, phase),
            ledger="Shared clocks",
            result="pairs within %s in period scale with the tolerance (%s per unit); in phase within %g: %d "
                   "pairs and %d groups of 3 or more, as shuffled phases give; groups above the shuffled largest (%d) sit at "
                   "periods the catalogue repeats exactly, its grid for long-period variables; the largest of the rest %d"
                   % (", ".join("%.3g" % e for e in counts), ", ".join("%.3g" % (c / e) for e, c in counts.items()), phase, len(cp), len(big),
                      max(n[2] for n in nulls), len(nongrid[0]) if nongrid else 0))
        core.clear_targets("ztf_shared_clocks")
        core.record_targets("ztf_shared_clocks", [dict(
            target="ztf:%s" % t["ID"][i], stat=float(len(g)), unit="variables sharing its period and phase",
            note="P %.6f d, %s" % (P[i], t["Type"][i])) for g in big for i in g])
        print("recorded")


superflare_DOC = r"""Superflare deficit among TESS planet hosts: fraction of flares above 1e34 erg against matched non-hosts, 2000 draws.

    python -I seti.py flashes superflare analyse
    python -I seti.py flashes superflare record"""

superflare_ROOT = os.environ.get("SUPERFLARE_ROOT", core.data("superflare"))
superflare_OUT = "superflare_deficit.json"
superflare_E1, superflare_E2, superflare_NMATCH = 1e33, 1e34, 10


def superflare_load():
    from astropy.table import Table
    os.makedirs(superflare_ROOT, exist_ok=True)
    hp = os.path.join(superflare_ROOT, "hosts.json")
    if not os.path.exists(hp):
        json.dump(dict(toi=core.exoplanets("SELECT tid, toi, tfopwg_disp, pl_orbper, pl_rade, st_teff, st_dist FROM toi"),
                       ps=core.exoplanets("SELECT tic_id, hostname, pl_name, pl_orbper, pl_rade, st_teff, sy_dist, discoverymethod FROM pscomppars")), open(hp, "w"))
    h = json.load(open(hp))
    hosts = {int(r["tid"]) for r in h["toi"] if r["tfopwg_disp"] in ("PC", "CP", "KP") and r["tid"]}
    hosts |= {int(str(r["tic_id"]).replace("TIC", "").strip()) for r in h["ps"] if r["tic_id"] and str(r["tic_id"]).replace("TIC", "").strip().isdigit()}
    for cat, tb in (("J/A+A/669/A15", "table1"), ("J/A+A/669/A15", "table2")):
        p = os.path.join(superflare_ROOT, "yang2023_%s.ecsv" % tb)
        if not os.path.exists(p):
            from astroquery.vizier import Vizier
            Vizier(row_limit=-1, columns=["**"], timeout=900).get_catalogs("%s/%s" % (cat, tb))[0].write(p, format="ascii.ecsv", overwrite=True)
    stars = {}
    for r in Table.read(os.path.join(superflare_ROOT, "yang2023_table2.ecsv"), format="ascii.ecsv"):
        stars[int(r["TIC"])] = dict(tic=int(r["TIC"]), teff=num(r["Teff"]), rad=num(r["Rad"]), dist=num(r["DistG"]) or num(r["DistT"]), nsec=num(r["NTESS"]),
                                    flare=str(r["Flare"]).strip().lower().startswith("y"), host=int(r["TIC"]) in hosts, E=[])
    for r in Table.read(os.path.join(superflare_ROOT, "yang2023_table1.ecsv"), format="ascii.ecsv"):
        tic, e = int(r["TIC"]), num(r["E"])
        if tic in stars and e and e > 0:
            stars[tic]["E"].append(e)
    return stars


def superflare_analyse():
    stars = superflare_load()
    flaring = [s for s in stars.values() if s["E"] and s["teff"] and s["rad"] and s["dist"]]
    hosts = [s for s in flaring if s["host"]]
    pool = [s for s in flaring if not s["host"]]
    print("%d stars, %d flaring with energies, %d of them planet hosts (of %d hosts in the sample); %d flaring non-hosts" % (
        len(stars), len(flaring), len(hosts), sum(1 for s in stars.values() if s["host"]), len(pool)))

    def key(s):
        return np.array([s["teff"] / 300.0, math.log10(s["rad"]) / 0.1, math.log10(s["dist"]) / 0.15, (s["nsec"] or 1) / 4.0])
    from scipy.spatial import cKDTree
    pk = np.array([key(s) for s in pool])
    tree = cKDTree(pk)
    matches = [tree.query(key(h), k=superflare_NMATCH)[1] for h in hosts]

    def fractions(group):
        e = np.concatenate([np.array(s["E"]) for s in group]) if group else np.array([])
        e = e[e >= superflare_E1]
        return (float(np.mean(e >= superflare_E2)) if len(e) else float("nan"), int(len(e)), float(np.mean([max(s["E"]) >= superflare_E2 for s in group])) if group else float("nan"))
    fh, nh, ph = fractions(hosts)
    rng = np.random.default_rng(4)
    draws, draws_star = [], []
    for _ in range(2000):
        ctrl = [pool[int(rng.choice(m))] for m in matches]
        f, _, p = fractions(ctrl)
        draws.append(f)
        draws_star.append(p)
    draws, draws_star = np.array(draws), np.array(draws_star)
    from scipy.stats import ks_2samp
    eh = np.concatenate([np.array(s["E"]) for s in hosts])
    eh = np.log10(eh[eh >= superflare_E1])
    ctrl_all = [pool[int(i)] for m in matches for i in m]
    ec = np.concatenate([np.array(s["E"]) for s in ctrl_all])
    ec = np.log10(ec[ec >= superflare_E1])
    ks = ks_2samp(eh, ec)
    print("flares above %.0e erg among those above %.0e: hosts %.3f (%d flares) against matched non-hosts %.3f +- %.3f (p of a deficit %.3f); stars with a flare above %.0e: "
          "hosts %.3f against %.3f +- %.3f (p %.3f); energies above %.0e erg: KS statistic %.3f, p %.3f, medians %.2f and %.2f" % (
              superflare_E2, superflare_E1, fh, nh, draws.mean(), draws.std(), float(np.mean(draws <= fh)), superflare_E2, ph, draws_star.mean(), draws_star.std(),
              float(np.mean(draws_star <= ph)), superflare_E1, ks.statistic, ks.pvalue, float(np.median(eh)), float(np.median(ec))))
    # hosts with their largest flares
    for s in sorted(hosts, key=lambda s: -max(s["E"]))[:8]:
        print("  TIC %d: Teff %.0f K, R %.2f Rsun, %.0f pc, %d flares, largest %.1e erg" % (s["tic"], s["teff"], s["rad"], s["dist"], len(s["E"]), max(s["E"])))
    json.dump(dict(n_stars=len(stars), n_flaring=len(flaring), n_hosts=len(hosts), n_pool=len(pool), frac_hosts=fh, n_flares_hosts=nh, frac_ctrl=float(draws.mean()), sd_ctrl=float(draws.std()),
                   p_deficit=float(np.mean(draws <= fh)), star_hosts=ph, star_ctrl=float(draws_star.mean()), star_sd=float(draws_star.std()), p_star=float(np.mean(draws_star <= ph)),
                   ks=float(ks.statistic), ks_p=float(ks.pvalue), params=dict(e1=superflare_E1, e2=superflare_E2, nmatch=superflare_NMATCH),
                   hosts=[dict(tic=s["tic"], teff=s["teff"], rad=s["rad"], dist=s["dist"], n=len(s["E"]), emax=max(s["E"]), frac=float(np.mean(np.array(s["E"])[np.array(s["E"]) >= superflare_E1] >= superflare_E2)) if any(e >= superflare_E1 for e in s["E"]) else None) for s in hosts]),
              open(core.results(superflare_OUT), "w"), indent=0)


def superflare_record():
    d = json.load(open(core.results(superflare_OUT)))
    core.record_search(
        "superflare_deficit", "Superflare deficit among planet hosts", archive="Yang et al. (2023) TESS flare catalogue (J/A+A/669/A15); TESS objects of interest and confirmed planets (NASA Exoplanet Archive)",
        selection="flaring TESS two-minute targets with flare energies (%d), the %d planet hosts among them against %d flaring non-hosts" % (d["n_flaring"], d["n_hosts"], d["n_pool"]),
        code=os.path.abspath(__file__), entry="superflare", n_targets=d["n_hosts"], params=d["params"],
        null_summary="each host's %d nearest non-hosts in temperature, radius, distance and sectors, drawn 2000 times: fraction of flares above %.0e among those above %.0e %.3f +- %.3f, stars with a flare above %.0e %.3f +- %.3f" % (
            d["params"]["nmatch"], d["params"]["e2"], d["params"]["e1"], d["frac_ctrl"], d["sd_ctrl"], d["params"]["e2"], d["star_ctrl"], d["star_sd"]),
        sensitivity="a halving of the superflare fraction at about 3 sigma for %d hosts with %d flares above %.0e erg" % (d["n_hosts"], d["n_flares_hosts"], d["params"]["e1"]),
        ledger="Superflare deficit among planet hosts",
        result="hosts' fraction of flares above %.0e erg %.3f (%d flares) against %.3f +- %.3f matched (p of a deficit %.2f); stars with a superflare %.3f against %.3f +- %.3f (p %.2f); KS on the energies %.3f (p %.2f)" % (
            d["params"]["e2"], d["frac_hosts"], d["n_flares_hosts"], d["frac_ctrl"], d["sd_ctrl"], d["p_deficit"], d["star_hosts"], d["star_ctrl"], d["star_sd"], d["p_star"], d["ks"], d["ks_p"]))
    core.clear_targets("superflare_deficit")
    core.record_targets("superflare_deficit", [dict(target="tic:%d" % h["tic"], stat=h["frac"], unit="fraction of the star's flares above 1e34 erg among those above 1e33",
                                                         note="Teff %.0f K, R %.2f Rsun, %.0f pc, %d flares, largest %.1e erg" % (h["teff"], h["rad"], h["dist"], h["n"], h["emax"])) for h in d["hosts"]])
    print("recorded %d hosts" % d["n_hosts"])


def superflare_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": superflare_analyse, "record": superflare_record}[mode]()


neliota_DOC = r"""NELIOTA lunar flashes: R-I colour temperature against impact range, duration against 0.5 s, and positional recurrence against permuted longitudes.

    python -I seti.py flashes neliota fetch
    python -I seti.py flashes neliota analyse
    python -I seti.py flashes neliota record"""

neliota_ROOT = os.environ.get("NELIOTA_ROOT", core.data("neliota"))
neliota_OUT = "neliota_flashes.json"
neliota_LAM_R, neliota_LAM_I = 0.64, 0.80  # Cousins R and I (um)
neliota_TMIN, neliota_TMAX, neliota_DURATION, neliota_PAIR = 1000.0, 8000.0, 0.5, 1.5
neliota_HEAD = dict(core.UA, **{"Accept": "*/*"})


def neliota_mag(s):
    """Magnitude m.mm(ee) as (value, error); (None, None) when empty."""
    m = re.match(r"\s*([\d.]+)\s*\((\d+)\)", s or "")
    if not m:
        return None, None
    v, e = m.group(1), m.group(2)
    dec = len(v.split(".")[1]) if "." in v else 0
    return float(v), float(e) * 10 ** (-dec)


def neliota_fetch():
    """NELIOTA flash tables (2020 PDF, 2024 HTML appendix): id, date, class, duration, R, I, latitude, longitude."""
    import fitz
    import html as htmlmod
    os.makedirs(neliota_ROOT, exist_ok=True)
    rows, seen = [], set()
    p = os.path.join(neliota_ROOT, "arxiv_1911.06101.pdf")
    if not os.path.exists(p):
        core.download("https://arxiv.org/pdf/1911.06101", p, headers=neliota_HEAD)
    text = re.sub(r"\s*\n\s*", " ", "".join(pg.get_text() for pg in fitz.open(p)))
    pat = re.compile(r"(?<![\d.])(\d{1,3}) (20\d\d \d\d \d\d) (\d\d:\d\d:\d\d\.\d+) (V|SC1|SC2) (\d+) ([\d.]+\(\d+\))(?: ([\d.]+\(\d+\)))? ([+−-]?\d+\.\d) ([+−-]?\d+\.\d)")
    for m in pat.finditer(text):
        fid, date, ut, val, dt, m1, m2, lat, lon = m.groups()
        if int(fid) in seen:
            continue
        seen.add(int(fid))
        if m2:
            (mR, eR), (mI, eI) = neliota_mag(m1), neliota_mag(m2)
        elif val == "SC2":
            (mR, eR), (mI, eI) = (None, None), neliota_mag(m1)
        else:
            (mR, eR), (mI, eI) = neliota_mag(m1), (None, None)
        rows.append(dict(id=int(fid), date=date + " " + ut, val=val, dur_ms=float(dt), mR=mR, eR=eR, mI=mI, eI=eI,
                         lat=float(lat.replace("−", "-")), lon=float(lon.replace("−", "-")), source="Liakos et al. 2020"))
    p = os.path.join(neliota_ROOT, "arxiv_2403.19613.html")
    if not os.path.exists(p):
        open(p, "wb").write(core.get("https://arxiv.org/html/2403.19613", headers=neliota_HEAD, timeout=120))
    page = open(p, encoding="utf-8", errors="replace").read()

    def cells(row):
        return [re.sub(r"\s+", " ", htmlmod.unescape(re.sub(r"<[^>]+>", " ", c))).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]

    def number(s):
        m = re.match(r"^[−-]?\s*[\d.]+", (s or "").replace("−", "-"))
        return float(m.group(0).replace(" ", "")) if m else None

    for tb in re.findall(r"<table.*?</table>", page, re.S):
        trs = re.findall(r"<tr[^>]*>(.*?)</tr>", tb, re.S)
        if not trs or "Date" not in trs[0] or "Lat" not in trs[0]:
            continue
        for tr in trs[1:]:
            c = cells(tr)
            for off in (0, 8):
                if len(c) >= off + 8 and re.match(r"^\d+$", c[off]) and int(c[off]) not in seen:
                    fid, date, val, dt, m1, m2, lat, lon = c[off:off + 8]
                    (mR, eR), (mI, eI) = neliota_mag(m1), neliota_mag(m2)
                    rows.append(dict(id=int(fid), date=date, val="V" if val.startswith("Val") else val, dur_ms=number(dt), mR=mR, eR=eR, mI=mI, eI=eI,
                                     lat=number(lat), lon=number(lon), source="Liakos et al. 2024"))
                    seen.add(int(fid))
    rows.sort(key=lambda r: r["id"])
    json.dump(rows, open(os.path.join(neliota_ROOT, "flashes.json"), "w"), indent=0)
    print("%d flashes, %d validated, %d with both bands, ids %d-%d" % (len(rows), sum(r["val"] == "V" for r in rows), sum(1 for r in rows if r["mR"] is not None and r["mI"] is not None),
                                                                       rows[0]["id"], rows[-1]["id"]))


def neliota_planck(lam_um, T):
    h, c, k = 6.626e-27, 2.998e10, 1.381e-16
    lam = lam_um * 1e-4
    return 2 * h * c ** 2 / lam ** 5 / math.expm1(h * c / (lam * k * T))


def neliota_temperature(ri):
    """Black-body temperature with R - I = ri (Vega as 9602 K); None outside 500-50000 K."""
    def colour(T):
        v = neliota_planck(neliota_LAM_R, 9602.0) / neliota_planck(neliota_LAM_I, 9602.0)
        return -2.5 * math.log10(neliota_planck(neliota_LAM_R, T) / neliota_planck(neliota_LAM_I, T) / v)
    lo, hi = 500.0, 50000.0
    if ri <= colour(hi) or ri >= colour(lo):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if colour(mid) > ri:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def neliota_pairs(lat, lon, limit_deg):
    """Index pairs of flashes within limit_deg on the lunar sphere."""
    la, lo = np.radians(lat), np.radians(lon)
    cosd = np.sin(la)[:, None] * np.sin(la)[None, :] + np.cos(la)[:, None] * np.cos(la)[None, :] * np.cos(lo[:, None] - lo[None, :])
    ang = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    i, j = np.where(np.triu(ang < limit_deg, 1))
    return list(zip(i.tolist(), j.tolist(), ang[i, j].tolist()))


def neliota_analyse():
    rows = json.load(open(os.path.join(neliota_ROOT, "flashes.json")))
    for r in rows:
        r["T"] = r["T_lo3"] = r["T_hi3"] = None
        if r["mR"] is not None and r["mI"] is not None:
            ri = r["mR"] - r["mI"]
            e = math.hypot(r["eR"] or 0.3, r["eI"] or 0.3)
            r["T"], r["T_hi3"], r["T_lo3"] = neliota_temperature(ri), neliota_temperature(ri - 3 * e), neliota_temperature(ri + 3 * e)
        r["hot"] = r["T_lo3"] is not None and r["T_lo3"] > neliota_TMAX
        r["cold"] = r["T_hi3"] is not None and r["T_hi3"] < neliota_TMIN
        r["long"] = (r["dur_ms"] or 0) > neliota_DURATION * 1000
    withT = [r for r in rows if r["T"]]
    T = np.array([r["T"] for r in withT])
    pos = [r for r in rows if r["lat"] is not None and r["lon"] is not None]
    lat, lon = np.array([r["lat"] for r in pos]), np.array([r["lon"] for r in pos])
    pairs = neliota_pairs(lat, lon, neliota_PAIR)
    rng = np.random.default_rng(4)
    null = np.array([len(neliota_pairs(lat, rng.permutation(lon), neliota_PAIR)) for _ in range(2000)])
    p_pairs = float((np.sum(null >= len(pairs)) + 1) / (len(null) + 1))
    hot, cold, long_ = [r for r in rows if r["hot"]], [r for r in rows if r["cold"]], [r for r in rows if r["long"]]
    print("%d flashes (%d validated), %d with colour temperatures: median %.0f K, 5th-95th %.0f-%.0f K; %d hotter than %.0f K at 3 sigma, %d cooler than %.0f, %d longer than %.1f s (longest %.0f ms)" % (
        len(rows), sum(r["val"] == "V" for r in rows), len(withT), np.median(T), np.percentile(T, 5), np.percentile(T, 95), len(hot), neliota_TMAX, len(cold), neliota_TMIN, len(long_), neliota_DURATION,
        max(r["dur_ms"] or 0 for r in rows)))
    print("pairs within %.1f deg on the Moon: %d against %.1f +- %.1f with the longitudes permuted (p %.3f)" % (neliota_PAIR, len(pairs), null.mean(), null.std(), p_pairs))
    for i, j, ang in sorted(pairs, key=lambda x: x[2]):
        a, b = pos[i], pos[j]
        print("  flashes %d and %d: %.2f deg apart at (%+.1f, %+.1f), %s and %s, %s/%s, T %s/%s K" % (
            a["id"], b["id"], ang, a["lat"], a["lon"], a["date"][:10], b["date"][:10], a["val"], b["val"], "%.0f" % a["T"] if a["T"] else "-", "%.0f" % b["T"] if b["T"] else "-"))
    for r in hot + cold:
        print("  flash %d (%s, %s): R %s I %s, T %.0f K (3 sigma range %s-%s), %s ms" % (r["id"], r["date"][:10], r["val"], r["mR"], r["mI"], r["T"], "%.0f" % r["T_lo3"] if r["T_lo3"] else "-",
                                                                                   "%.0f" % r["T_hi3"] if r["T_hi3"] else "-", r["dur_ms"]))
    json.dump(dict(n=len(rows), n_T=len(withT), T_median=float(np.median(T)), hot=[r["id"] for r in hot], cold=[r["id"] for r in cold], long=[r["id"] for r in long_],
                   pairs=[(pos[i]["id"], pos[j]["id"], ang) for i, j, ang in pairs], pairs_null=float(null.mean()), pairs_null_sd=float(null.std()), p_pairs=p_pairs, rows=rows,
                   params=dict(tmin=neliota_TMIN, tmax=neliota_TMAX, duration_s=neliota_DURATION, pair_deg=neliota_PAIR)), open(core.results(neliota_OUT), "w"), indent=0)


def neliota_record():
    d = json.load(open(core.results(neliota_OUT)))
    core.record_search(
        "neliota_flashes", "Non-thermal or recurrent lunar impact flashes", archive="NELIOTA flash lists of Liakos et al. (2020, 2024), from the papers' tables", release="2017-2023",
        selection="%d flashes, %d with R and I" % (d["n"], d["n_T"]), code=os.path.abspath(__file__), entry="neliota", n_targets=d["n"], params=d["params"],
        null_summary="pairs within %.1f deg against %.1f +- %.1f with the longitudes permuted among the flashes 2000 times" % (d["params"]["pair_deg"], d["pairs_null"], d["pairs_null_sd"]),
        sensitivity="a colour temperature outside %.0f-%.0f K at 3 sigma, a duration above %.1f s, or two flashes within %.1f deg" % (d["params"]["tmin"], d["params"]["tmax"], d["params"]["duration_s"], d["params"]["pair_deg"]),
        ledger="Non-thermal or recurrent lunar impact flashes",
        result="colour temperatures median %.0f K; %d hotter than the range at 3 sigma, %d cooler, %d longer than %.1f s; %d pairs within %.1f deg against %.1f (p %.2f)" % (
            d["T_median"], len(d["hot"]), len(d["cold"]), len(d["long"]), d["params"]["duration_s"], len(d["pairs"]), d["params"]["pair_deg"], d["pairs_null"], d["p_pairs"]))
    core.clear_targets("neliota_flashes")
    core.record_targets("neliota_flashes", [dict(target="neliota:%d" % r["id"], stat=r["T"], unit="colour temperature from R - I, K",
                                                     note="%s, %s, %s ms, (%s, %s)" % (r["date"][:10], r["val"], r["dur_ms"], r["lat"], r["lon"])) for r in d["rows"]])
    print("recorded %d flashes" % d["n"])


def neliota_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": neliota_fetch, "analyse": neliota_analyse, "record": neliota_record}[mode]()


# --------------------------------------------------------------------------------------------------------------
# flares together and alike, and stars steadier than their photons

YANG_FLARES = '"J/A+A/669/A15/table1"'
YANG_STARS = '"J/A+A/669/A15/table2"'


flaresync_DOC = r"""Flares firing together across the sky: the TESS flares of Yang et al. (2023) (VizieR J/A+A/669/A15), pairs on stars more than 10 deg apart whose peak times agree to 1 minute, against the pair rate at separations of 10-60 minutes scaled to the 2-minute window.

    python -I seti.py flashes flaresync [record] [--dt-min 1] [--sep-deg 10] [--control-min 10] [--control-max 60]"""


def flaresync_main(record=False, dt_min=1.0, sep_deg=10.0, control_min=10.0, control_max=60.0):
    fl = core.tap("vizier", "SELECT TIC, TimeP, E FROM %s" % YANG_FLARES, timeout=900)
    st = {str(r["TIC"]): (num(r["RAJ2000"]), num(r["DEJ2000"])) for r in core.tap("vizier", "SELECT TIC, RAJ2000, DEJ2000 FROM %s" % YANG_STARS, timeout=900)}
    fl = sorted([r for r in fl if num(r["TimeP"]) is not None and str(r["TIC"]) in st], key=lambda r: num(r["TimeP"]))
    t = np.array([num(r["TimeP"]) for r in fl])
    near, wide, zero, hits = 0, 0, 0, []
    m1, m10, m60 = dt_min / 1440.0, control_min / 1440.0, control_max / 1440.0
    for i in range(len(fl)):
        j = i + 1
        while j < len(fl) and t[j] - t[i] < m60:
            a, b = fl[i], fl[j]
            if str(a["TIC"]) != str(b["TIC"]) and core.sep_arcsec(*st[str(a["TIC"])], *st[str(b["TIC"])]) > sep_deg * 3600:
                dt = t[j] - t[i]
                if dt < m1:
                    near += 1
                    zero += dt * 86400 < 1
                    hits.append((a["TIC"], b["TIC"], dt * 86400))
                elif dt >= m10:
                    wide += 1
            j += 1
    expect = wide * dt_min / (control_max - control_min)
    print("%d flares on %d stars; pairs more than %g deg apart within %g min: %d (%d of them at one timestamp, the 2-minute cadence all stars "
          "share) against %.1f from the rate at %g-%g min" % (len(fl), len(st), sep_deg, dt_min, near, zero, expect, control_min, control_max))
    if record:
        core.record_run(
            "tess_flare_synchrony", "Flares firing together across the sky", "Yang et al. (2023) TESS flare catalogue (VizieR J/A+A/669/A15)",
            "flares with peak times on catalogued stars: %d" % len(fl), os.path.abspath(__file__), "flaresync", len(fl), "flares",
            dict(dt_min=dt_min, sep_deg=sep_deg, control_min=[control_min, control_max]), "pairs at %g-%g min scaled to the window: %.1f" % (
                control_min, control_max, expect), "flares on distant stars within %g min of each other" % dt_min, "%d pairs against %.1f" % (near, expect), [])
        print("recorded")


flarecopies_DOC = r"""Standardised flares: stars of the Yang et al. (2023) TESS flare catalogue with ten or more flares whose flare energies scatter by less than 5 per cent, where a star's flares spread over decades of energy.

    python -I seti.py flashes flarecopies [record] [--n-min 10] [--spread 0.05]"""


def flarecopies_main(record=False, n_min=10, spread=0.05):
    fl = core.tap("vizier", "SELECT TIC, E FROM %s" % YANG_FLARES, timeout=900)
    by = {}
    for r in fl:
        if num(r["E"]):
            by.setdefault(str(r["TIC"]), []).append(num(r["E"]))
    many = {k: np.array(v) for k, v in by.items() if len(v) >= n_min}
    # energies may be listed as logarithms; the spread is taken on the linear energy either way
    spreads = {k: float(np.std(10 ** v if np.median(v) < 100 else v) / np.mean(10 ** v if np.median(v) < 100 else v)) for k, v in many.items()}
    hits = sorted([(k, s) for k, s in spreads.items() if s < spread], key=lambda x: x[1])
    vals = np.array(list(spreads.values()))
    print("%d stars with %d or more flares; spread of energy median %.2f, lowest %.3f; below %g per cent: %d %s" % (
        len(many), n_min, float(np.median(vals)), float(vals.min()), 100 * spread, len(hits), hits[:10]))
    if record:
        core.record_run(
            "tess_standard_flares", "Standardised flares", "Yang et al. (2023) TESS flare catalogue (VizieR J/A+A/669/A15)",
            "stars with %d or more flares: %d" % (n_min, len(many)), os.path.abspath(__file__), "flarecopies", len(many), "stars", dict(spread=spread, n_min=n_min),
            "the stars' own energy spreads: median %.2f" % float(np.median(vals)), "a star whose flares all carry one energy to %g per cent" % (100 * spread),
            "%d below %g per cent; the lowest spread %.3f" % (len(hits), 100 * spread, float(vals.min())),
            [dict(target="tic:%s" % k, stat=s, unit="fractional spread of flare energy") for k, s in hits])
        print("recorded")


steady_DOC = r"""Stars steadier than photon noise allows: the Kepler DR25 stellar table's 6-hour combined differential photometric precision against the shot noise of the star's Kepler magnitude, 1.74e5 electrons per second at Kp 12 over 6 hours at a 92 per cent duty cycle (17 ppm at Kp 12), flagged below 0.8 of it.

    python -I seti.py flashes steady [record] [--fraction 0.8] [--hours 6] [--electrons-kp12 1.74e5] [--duty 0.92]

The hours name the CDPP column (3, 6 or 12 in the DR25 table)."""


def steady_main(record=False, fraction=0.8, hours=6.0, electrons_kp12=1.74e5, duty=0.92):
    cols = list(core.exoplanets("SELECT TOP 1 * FROM q1_q17_dr25_ks")[0])
    tag = "%02dp%d" % (int(hours), round((hours % 1) * 10))
    cd = next((c for c in cols if "cdpp" in c.lower() and tag in c), None)
    rows = core.exoplanets("SELECT kepid, kepmag, %s FROM q1_q17_dr25_ks WHERE %s IS NOT NULL AND kepmag IS NOT NULL" % (cd, cd))
    out = []
    for r in rows:
        kp, cdpp = float(r["kepmag"]), float(r[cd])
        shot = 1e6 / math.sqrt(electrons_kp12 * 10 ** (-0.4 * (kp - 12)) * hours * 3600 * duty)
        out.append((r["kepid"], kp, cdpp, cdpp / shot))
    hits = sorted([o for o in out if o[3] < fraction], key=lambda o: o[3])
    ratio = np.array([o[3] for o in out])
    print("%d Kepler stars with %g-hour CDPP (%s); noise over shot noise median %.2f; below %g: %d %s" % (
        len(out), hours, cd, float(np.median(ratio)), fraction, len(hits), [(o[0], o[1], o[2], round(o[3], 2)) for o in hits[:10]]))
    if record:
        core.record_run(
            "kepler_steadier_than_photons", "Stars steadier than photon noise allows", "NASA Exoplanet Archive Q1-Q17 DR25 Kepler stellar table",
            "stars with a %g-hour CDPP: %d" % (hours, len(out)), os.path.abspath(__file__), "steady", len(out), "stars",
            dict(fraction=fraction, hours=hours, electrons_kp12=electrons_kp12, duty=duty),
            "each star's shot noise from its Kepler magnitude", "a star whose measured noise falls below the photons' own",
            "%d below %g of the shot noise; median ratio %.2f" % (len(hits), fraction, float(np.median(ratio))),
            [dict(target="kic:%s" % o[0], stat=o[3], unit="CDPP over shot noise", vet="artefact",
                  note="Kp %.2f, %s ppm, below the photons' own noise: the pipeline's figure" % (o[1], o[2])) for o in hits[:500]])
        print("recorded")


sndeficit_DOC = r"""Galaxies that suppress supernovae: HECATE galaxies within 20 Mpc with a homogenised star-formation rate, each given the core-collapse supernovae its rate predicts over 1980-2015 at one per 100 solar masses formed, against the supernovae the Asiago catalogue (B/sn) lists in it, hosts matched within 1 arcmin; a galaxy whose supernovae fall short of its star formation beyond chance is either hiding them in dust or stopping them. A galaxy whose infrared star-formation rate exceeds its ultraviolet one by a factor of three, or whose infrared luminosity passes 2e10 solar, is dusty, and an active nucleus lends its infrared to the rate.

    python -I seti.py flashes sndeficit [record] [--d-max-mpc 20] [--sn-per-msun 0.01] [--year-min 1980] [--year-max 2015] [--host-arcmin 1] [--dust-ratio 3] [--ltir-max 10.3] [--p-max 0.01]"""


def sndeficit_main(record=False, d_max_mpc=20.0, sn_per_msun=0.01, year_min=1980, year_max=2015, host_arcmin=1.0, dust_ratio=3.0, ltir_max=10.3, p_max=0.01):
    from scipy.stats import poisson
    SN_YEARS = (year_min, year_max)
    gal = core.tap("vizier", 'SELECT PGC, NED, RAJ2000, DEJ2000, D, logSFRh, logSFRTIR, logSFRGSW, logLTIR, logMtot, Nuclear, AGNs17, AGNh FROM "J/MNRAS/506/1896/hecate" '
                             'WHERE D < %g AND logSFRh IS NOT NULL' % d_max_mpc)
    sne = core.tap("vizier", 'SELECT SN, Galaxy, Type, RAG, DEG FROM "B/sn/sncat" WHERE HRV < 3000 AND RAG IS NOT NULL')
    sne = [s for s in sne if SN_YEARS[0] <= int(s["SN"][:4]) <= SN_YEARS[1]]
    hosts = {}
    for m in core.xmatch([dict(source_id="s%d" % i, ra=float(s["RAG"]), dec=float(s["DEG"])) for i, s in enumerate(sne)], "J/MNRAS/506/1896/hecate", host_arcmin * 60.0):
        i = int(m["source_id"][1:])
        if i not in hosts or float(m["angDist"]) < hosts[i][1]:
            hosts[i] = (m["PGC"], float(m["angDist"]))
    count = {}
    for i, (pgc, _) in hosts.items():
        count[pgc] = count.get(pgc, 0) + 1
    years = SN_YEARS[1] - SN_YEARS[0] + 1
    out = []
    for g in gal:
        sfr = 10 ** float(g["logSFRh"])
        exp = sfr * sn_per_msun * years
        n = count.get(g["PGC"], 0)
        p = float(poisson.cdf(n, exp))
        tir, uv, ltir = core.num(g["logSFRTIR"]), core.num(g["logSFRGSW"]), core.num(g["logLTIR"])
        # a luminous infrared galaxy hides its supernovae in dust; an active nucleus lends its infrared to the rate
        dusty = (tir is not None and uv is not None and tir - uv > math.log10(dust_ratio)) or (ltir is not None and ltir > ltir_max)
        agn = bool((g["AGNs17"] or "").strip() or (g["AGNh"] or "").strip() or (g["Nuclear"] or "").strip() not in ("", "0"))
        out.append(dict(pgc=g["PGC"], name=(g["NED"] or "").strip(), d=float(g["D"]), sfr=round(sfr, 2), expected=round(exp, 1), seen=n, p=p, dusty=dusty, agn=agn,
                        vet="dust" if dusty else "background-galaxy" if agn else None))
    short = sorted([o for o in out if o["p"] < p_max], key=lambda o: o["p"])
    print("%d galaxies within %g Mpc with star-formation rates, %d supernovae of %d-%d matched to them; %d galaxies short of their supernovae at p below %g, %d of them dusty "
          "and %d more with active nuclei; the rest %s" % (len(gal), d_max_mpc, len(hosts), SN_YEARS[0], SN_YEARS[1], len(short), p_max, sum(1 for o in short if o["dusty"]),
                                                            sum(1 for o in short if o["agn"] and not o["dusty"]),
                                                            [(o["name"], o["d"], o["sfr"], o["expected"], o["seen"], round(o["p"], 4)) for o in short if o["vet"] is None]))
    for o in short:
        print("  %s (PGC %s, %.1f Mpc): %.1f Msun/yr, %.1f expected, %d seen, p %.4f%s%s" % (o["name"], o["pgc"], o["d"], o["sfr"], o["expected"], o["seen"], o["p"],
                                                                                             ", dusty" if o["dusty"] else "", ", active nucleus" if o["agn"] else ""))
    if record:
        core.record_run("supernova_deficits", "Galaxies that suppress supernovae", "HECATE (J/MNRAS/506/1896); Asiago supernova catalogue (B/sn) by CDS XMatch",
                        "galaxies within %g Mpc with star-formation rates: %d" % (d_max_mpc, len(gal)), os.path.abspath(__file__), "sndeficit", len(gal), "galaxies",
                        dict(sn_per_msun=sn_per_msun, years=SN_YEARS, host_arcmin=host_arcmin, dust_ratio=dust_ratio, ltir_max=ltir_max, d_max_mpc=d_max_mpc, p_max=p_max),
                        "Poisson counts at each galaxy's own rate",
                        "a galaxy with fewer supernovae than its star formation gives at p below %g" % p_max, "%d short, %d of them dusty and %d with active nuclei; %d unexplained" % (
                            len(short), sum(1 for o in short if o["dusty"]), sum(1 for o in short if o["agn"] and not o["dusty"]), sum(1 for o in short if o["vet"] is None)),
                        [dict(target="pgc:%s" % o["pgc"], stat=o["seen"], p=o["p"], unit="supernovae seen against %.1f expected" % o["expected"], vet=o["vet"],
                              note="%s, %.1f Mpc, %.1f Msun/yr" % (o["name"], o["d"], o["sfr"])) for o in short])
        print("recorded")


symflares_DOC = r"""Symmetric flares: the TESS flares of Yang et al. (2023) (VizieR J/A+A/669/A15) lasting 40 minutes or more, so that the two-minute cadence resolves their shape, whose rise from start to peak takes as long as the decay from peak to end within 10 per cent of the duration, or longer; a stellar flare rises impulsively and decays slowly, and a shape that runs backwards or stands symmetric is not a flare's. Each read for its amplitude and energy, and the star's flare count, since two flares in succession give one symmetric event and a catalogue's boundaries follow its own noise threshold.

    python -I seti.py flashes symflares [record] [--dur-min-d 0.028] [--sym-frac 0.1]"""


def symflares_main(record=False, dur_min_d=0.028, sym_frac=0.1):
    fl = core.tap("vizier", "SELECT TIC, TimeS, TimeP, TimeE, Dur, Amp, E FROM %s" % YANG_FLARES, timeout=900)
    per_star = {}
    for r in fl:
        per_star[str(r["TIC"])] = per_star.get(str(r["TIC"]), 0) + 1
    long_, sym, back = [], [], []
    for r in fl:
        ts, tp, te, d = num(r["TimeS"]), num(r["TimeP"]), num(r["TimeE"]), num(r["Dur"])
        if None in (ts, tp, te, d) or d < dur_min_d:
            continue
        rise, decay = tp - ts, te - tp
        o = dict(tic=str(r["TIC"]), start=ts, dur=d, rise=rise, decay=decay, ratio=rise / d, amp=num(r["Amp"]), e=num(r["E"]), nflares=per_star[str(r["TIC"])])
        long_.append(o)
        if rise > decay:
            back.append(o)
        elif abs(rise - decay) < sym_frac * d:
            sym.append(o)
    ratios = np.array([o["ratio"] for o in long_])
    print("%d flares, %d lasting %g d or more; rise over duration median %.2f, 90th percentile %.2f; symmetric within %g of the duration: %d; rising longer than "
          "decaying: %d" % (len(fl), len(long_), dur_min_d, np.median(ratios) if len(ratios) else 0, np.percentile(ratios, 90) if len(ratios) else 0, sym_frac, len(sym),
                            len(back)))
    for o in sorted(back + sym, key=lambda o: -o["ratio"])[:25]:
        print("  TIC %s at %.3f: %.3f d, rise %.3f, decay %.3f (rise %.2f of the duration), amplitude %s, energy %s, %d flares on the star" % (
            o["tic"], o["start"], o["dur"], o["rise"], o["decay"], o["ratio"], o["amp"], o["e"], o["nflares"]))
    json.dump(dict(n=len(fl), long=len(long_), symmetric=sym, backwards=back), open(core.results("symmetric_flares.json"), "w"), indent=1)
    if record:
        hits = back + sym
        core.record_run(
            "tess_symmetric_flares", "Symmetric flares", "Yang et al. (2023) TESS flare catalogue (VizieR J/A+A/669/A15)",
            "flares lasting %g d or more: %d of %d" % (dur_min_d, len(long_), len(fl)), os.path.abspath(__file__), "symflares", len(long_), "flares",
            dict(dur_min_d=dur_min_d, sym_frac=sym_frac), "the flares' own rise fractions: median %.2f, 90th percentile %.2f" % (
                np.median(ratios) if len(ratios) else 0, np.percentile(ratios, 90) if len(ratios) else 0),
            "a flare whose rise takes as long as its decay to %g of its duration" % sym_frac,
            "%d symmetric and %d rising longer than decaying, of %d, at every duration and on quiet and busy stars alike: the catalogue's event boundaries, not flare "
            "shapes" % (len(sym), len(back), len(long_)),
            [dict(target="tic:%s" % o["tic"], stat=o["ratio"], unit="rise as a fraction of the duration",
                  vet="event-boundary",
                  note="start %.3f, duration %.3f d, amplitude %s, energy %s, %d flares on the star" % (o["start"], o["dur"], o["amp"], o["e"], o["nflares"])) for o in hits])
        print("recorded")


flicker_DOC = r"""Nearby stars flickering by the minute: Gaia DR3 short-timescale variables (gaiadr3.vari_short_timescale) within 100 pc whose characteristic frequency exceeds 48 cycles a day, light changing within half an hour at a star next door, where a photosphere's own pulsations and spots are slower by far and only compact stars and close binaries keep such time; each read for its amplitude, its white-dwarf locus in colour and absolute magnitude, its RUWE and its SIMBAD type; a red dwarf (BP-RP above 2) the pipeline picks up flares, and its variogram timescale is the flares' rise rather than a period.

    python -I seti.py flashes flicker [record] [--plx-min 10] [--freq-min 48] [--red-colour 2.0]"""


def flicker_main(record=False, plx_min=10.0, freq_min=48.0, red_colour=2.0):
    rows = core.gaia_by_range("SELECT v.source_id, v.frequency, v.amplitude_estimate, v.number_of_fov_transits, g.parallax, g.phot_g_mean_mag, g.bp_rp, g.ruwe, "
                              "g.ipd_frac_multi_peak, g.ra, g.dec FROM gaiadr3.vari_short_timescale AS v JOIN gaiadr3.gaia_source AS g ON v.source_id = g.source_id "
                              "WHERE v.source_id BETWEEN {lo} AND {hi} AND g.parallax > %g AND g.parallax_over_error > 10" % plx_min, n=8, maxrec=100000)
    st = []
    for r in rows:
        f, plx, g, c = num(r["frequency"]), num(r["parallax"]), num(r["phot_g_mean_mag"]), num(r["bp_rp"])
        if f is None or plx is None or g is None:
            continue
        st.append(dict(id=str(r["source_id"]), f=f, period_min=1440.0 / f, amp=num(r["amplitude_estimate"]), n=num(r["number_of_fov_transits"]), plx=plx, g=g, c=c,
                       mg=g + 5 * math.log10(plx / 100.0), ruwe=num(r["ruwe"]), multi=num(r["ipd_frac_multi_peak"])))
    fast = [s for s in st if s["f"] > freq_min]
    sim = {}
    for k in range(0, len(fast), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in fast[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in fast:
        s["simbad"] = sim.get(s["id"])
        typ, sp = (s["simbad"] or (None, "", ""))[1], (s["simbad"] or (None, "", ""))[2]
        # the white-dwarf locus: fainter than the main sequence by five magnitudes at the colour
        wd = s["c"] is not None and s["mg"] > 5.0 * s["c"] + 7.0
        # a red dwarf the short-timescale pipeline picks up flares, and its variogram timescale is the flares' rise, not a period
        s["vet"] = ("white-dwarf" if wd or typ in ("WD*", "WD?", "CV*", "No*") or "D" in sp[:2] else
                    "natural-type" if typ not in ("", "*", "PM*", "HV*", "low-mass*") else
                    "blend" if (s["ruwe"] or 0) > 1.4 or (s["multi"] or 0) > 2 else
                    "flare" if (s["c"] or 0) > red_colour else None)
    left = [s for s in fast if s["vet"] is None]
    count = {v: sum(s["vet"] == v for s in fast) for v in ("white-dwarf", "natural-type", "blend", "flare")}
    print("%d short-timescale variables within %.0f pc, the fastest at %.1f cycles a day; faster than %g cycles a day: %d, %d on the white-dwarf locus or of compact type, "
          "%d of other known types, %d blended, %d red dwarfs whose timescale is their flares'; unexplained %d" % (
              len(st), 1000 / plx_min, max(s["f"] for s in st) if st else 0, freq_min, len(fast), count["white-dwarf"], count["natural-type"], count["blend"], count["flare"],
              len(left)))
    for s in sorted(fast, key=lambda s: -s["f"]):
        print("  Gaia DR3 %s: %.1f cycles a day (%.1f min), amplitude %s mag over %s transits, G %.2f, BP-RP %s, M_G %.2f, RUWE %s; SIMBAD %s; %s" % (
            s["id"], s["f"], s["period_min"], s["amp"], s["n"], s["g"], s["c"], s["mg"], s["ruwe"], s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), fast=fast), open(core.results("nearby_flicker.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_short_timescale_nearby", "Nearby stars flickering by the minute", "Gaia DR3 vari_short_timescale and gaia_source; SIMBAD",
            "short-timescale variables within %.0f pc: %d" % (1000 / plx_min, len(st)), os.path.abspath(__file__), "flicker", len(st), "stars",
            dict(plx_min=plx_min, freq_min=freq_min, red_colour=red_colour), "the white-dwarf locus, SIMBAD's types, the image flags and the flare stars' colour",
            "a characteristic frequency above %g cycles a day at a star within %.0f pc" % (freq_min, 1000 / plx_min),
            "%d: %d white dwarfs or compact types, %d other known types, %d blended, %d flaring red dwarfs; %d unexplained" % (
                len(fast), count["white-dwarf"], count["natural-type"], count["blend"], count["flare"], len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=s["f"], unit="cycles a day", vet=s["vet"],
                  note="period %.1f min, amplitude %s mag, G %.2f, BP-RP %s, M_G %.2f, RUWE %s; SIMBAD %s" % (s["period_min"], s["amp"], s["g"], s["c"], s["mg"], s["ruwe"],
                                                                                                           s["simbad"])) for s in fast])
        print("recorded")


unclassified_DOC = r"""Variables Gaia could not classify next door: Gaia DR3 variables (vari_summary) within 50 pc whose G range exceeds 0.3 mag and to which the DR3 classifier (vari_classifier_result) gave no class, light changing by a third at a star next door in a way none of the two dozen known kinds of variable fits; each read in SIMBAD for its type, for its RUWE and image flags and for the white-dwarf locus, since an eclipsing pair, a flare star or a young star often escapes the classifier on few transits, and a red dwarf's range is one flaring transit against the rest.

    python -I seti.py flashes unclassified [record] [--plx-min 20] [--range-min 0.3] [--red-colour 2.0]"""


def unclassified_main(record=False, plx_min=20.0, range_min=0.3, red_colour=2.0):
    rows = core.gaia_by_range("SELECT g.source_id, g.parallax, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.ipd_frac_multi_peak, v.num_selected_g_fov, v.range_mag_g_fov, "
                              "v.in_vari_classification_result, c.best_class_name, c.best_class_score FROM gaiadr3.gaia_source AS g JOIN gaiadr3.vari_summary AS v "
                              "ON g.source_id = v.source_id LEFT OUTER JOIN gaiadr3.vari_classifier_result AS c ON g.source_id = c.source_id WHERE g.source_id BETWEEN {lo} "
                              "AND {hi} AND g.parallax > %g AND g.parallax_over_error > 10 AND v.range_mag_g_fov > %g" % (plx_min, range_min), n=8, maxrec=100000)
    st = [dict(id=str(r["source_id"]), plx=num(r["parallax"]), g=num(r["phot_g_mean_mag"]), c=num(r["bp_rp"]), ruwe=num(r["ruwe"]), multi=num(r["ipd_frac_multi_peak"]),
               n=num(r["num_selected_g_fov"]), rng=num(r["range_mag_g_fov"]), cls=(r["best_class_name"] or "").strip() or None, score=num(r["best_class_score"])) for r in rows]
    un = [s for s in st if not s["cls"]]
    sim = {}
    for k in range(0, len(un), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in un[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in un:
        s["simbad"] = sim.get(s["id"])
        typ, sp = (s["simbad"] or (None, "", ""))[1], (s["simbad"] or (None, "", ""))[2]
        s["mg"] = s["g"] + 5 * math.log10(s["plx"] / 100.0) if s["g"] is not None and s["plx"] else None
        wd = s["c"] is not None and s["mg"] is not None and s["mg"] > 5.0 * s["c"] + 7.0
        # a red dwarf's range is one flaring transit against the rest
        s["vet"] = ("white-dwarf" if wd or typ in ("WD*", "WD?", "CV*", "No*") or "D" in sp[:2] else
                    "natural-type" if typ not in ("", "*", "PM*", "HV*", "low-mass*") else
                    "blend" if (s["ruwe"] or 0) > 1.4 or (s["multi"] or 0) > 2 else
                    "flare" if (s["c"] or 0) > red_colour else None)
    left = [s for s in un if s["vet"] is None]
    count = {v: sum(s["vet"] == v for s in un) for v in ("white-dwarf", "natural-type", "blend", "flare")}
    print("%d Gaia variables within %.0f pc with a G range above %g mag; without a class: %d, %d white dwarfs or compact, %d of other known types, %d blended, %d flaring red "
          "dwarfs; unexplained %d" % (len(st), 1000 / plx_min, range_min, len(un), count["white-dwarf"], count["natural-type"], count["blend"], count["flare"], len(left)))
    for s in sorted(un, key=lambda s: -(s["rng"] or 0)):
        print("  Gaia DR3 %s: range %.2f mag over %s transits, G %.2f, BP-RP %s, M_G %s, RUWE %s; SIMBAD %s; %s" % (
            s["id"], s["rng"], s["n"], s["g"], s["c"], "%.2f" % s["mg"] if s["mg"] is not None else "-", s["ruwe"], s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), unclassified=un), open(core.results("gaia_unclassified_nearby.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_unclassified_variables_nearby", "Variables Gaia could not classify next door", "Gaia DR3 vari_summary and vari_classifier_result with gaia_source; SIMBAD",
            "Gaia variables within %.0f pc with a G range above %g mag: %d" % (1000 / plx_min, range_min, len(st)), os.path.abspath(__file__), "unclassified", len(st), "stars",
            dict(plx_min=plx_min, range_min=range_min, red_colour=red_colour), "the %d classified variables of the same sample" % (len(st) - len(un)),
            "a variable of a third of a magnitude that no class of the DR3 classifier fits",
            "%d unclassified: %d white dwarfs or compact, %d of other known types, %d blended, %d flaring red dwarfs; %d unexplained" % (
                len(un), count["white-dwarf"], count["natural-type"], count["blend"], count["flare"], len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=s["rng"], unit="G range, mag", vet=s["vet"],
                  note="%s transits, G %.2f, BP-RP %s, RUWE %s; SIMBAD %s" % (s["n"], s["g"], s["c"], s["ruwe"], s["simbad"])) for s in un])
        print("recorded")


wdvary_DOC = r"""White dwarfs varying outside the pulsation strips: Gaia DR3 variables (vari_summary) within 100 pc on the white-dwarf locus (fainter than the main sequence by five magnitudes at their colour) with a G range above 0.1 mag, matched to Gentile Fusillo et al. (2021) for temperature, gravity and white-dwarf probability, those outside the three pulsation strips with their margins (DA 10,000-13,500 K, DB 21,000-31,000 K, GW Vir above 70,000 K); a white dwarf that varies without pulsating is eclipsed, spotted, magnetic or crossed by debris, and debris in orbit around a dead star is the one natural kind of structure the archive would show a made one against. Each read in SIMBAD for its type, including the magnetic types that vary with their rotation, for a companion in its BP/RP flux excess and for its RUWE.

    python -I seti.py flashes wdvary [record] [--plx-min 10] [--range-min 0.1] [--pwd-min 0.75]"""

WD_STRIPS = ((10000.0, 13500.0), (21000.0, 31000.0), (70000.0, 1e6))


def wdvary_main(record=False, plx_min=10.0, range_min=0.1, pwd_min=0.75):
    rows = core.gaia_by_range("SELECT g.source_id, g.parallax, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.phot_bp_rp_excess_factor, v.num_selected_g_fov, v.range_mag_g_fov "
                              "FROM gaiadr3.gaia_source AS g JOIN gaiadr3.vari_summary AS v ON g.source_id = v.source_id WHERE g.source_id BETWEEN {lo} AND {hi} "
                              "AND g.parallax > %g AND g.parallax_over_error > 10 AND v.range_mag_g_fov > %g AND g.bp_rp IS NOT NULL "
                              "AND g.phot_g_mean_mag + 5 * LOG10(g.parallax / 100) > 5 * g.bp_rp + 7" % (plx_min, range_min), n=8, maxrec=100000)
    ids = [str(r["source_id"]) for r in rows]
    wd = {}
    for k in range(0, len(ids), 300):
        for r in core.tap("vizier", 'SELECT GaiaEDR3, TeffH, loggH, TeffHe, Pwd, RUWE FROM "J/MNRAS/508/3877/maincat" WHERE GaiaEDR3 IN (%s)' % ",".join(ids[k:k + 300])):
            wd[str(r["GaiaEDR3"])] = r
    sim = {}
    for k in range(0, len(ids), 200):
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % x for x in ids[k:k + 200])):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    out = []
    for r in rows:
        w = wd.get(str(r["source_id"]))
        if not w or (num(w["Pwd"]) or 0) < pwd_min:
            continue
        teff = num(w["TeffH"])
        c = num(r["bp_rp"])
        ex = (num(r["phot_bp_rp_excess_factor"]) or 0) - (1.154360 + 0.033772 * c + 0.032277 * c * c if c < 0.5 else 1.162004 + 0.011464 * c + 0.049255 * c * c - 0.005879 * c ** 3)
        o = dict(id=str(r["source_id"]), plx=num(r["parallax"]), g=num(r["phot_g_mean_mag"]), c=c, ruwe=num(r["ruwe"]), excess=ex, n=num(r["num_selected_g_fov"]),
                 rng=num(r["range_mag_g_fov"]), teff=teff, logg=num(w["loggH"]), pwd=num(w["Pwd"]), simbad=sim.get(str(r["source_id"])))
        o["strip"] = teff is not None and any(lo <= teff <= hi for lo, hi in WD_STRIPS)
        typ, sp = (o["simbad"] or (None, "", ""))[1], (o["simbad"] or (None, "", ""))[2]
        # a magnetic white dwarf (P or H in its type) varies with its rotation as its spots and field cross the disc
        o["vet"] = ("natural-type" if o["strip"] or typ in ("CV*", "No*", "EB*", "**", "SB*", "Pu*", "V*", "WD?") or "ZZ" in sp or "+" in sp or "P" in sp or "H" in sp[1:] else
                    "companion" if abs(ex) > 0.2 or (o["ruwe"] or 0) > 1.4 else None)
        out.append(o)
    left = [o for o in out if o["vet"] is None]
    print("%d Gaia variables within %.0f pc on the white-dwarf locus with a G range above %g mag and a catalogued white dwarf; in a pulsation strip or of a known kind %d, "
          "with a companion's flux excess or RUWE %d; unexplained %d" % (len(out), 1000 / plx_min, range_min, sum(o["vet"] == "natural-type" for o in out),
                                                                        sum(o["vet"] == "companion" for o in out), len(left)))
    for o in sorted(out, key=lambda o: -(o["rng"] or 0)):
        print("  Gaia DR3 %s: range %.2f mag over %s transits, G %.2f, BP-RP %.2f, Teff %s K, log g %s, P_wd %s, excess %+.3f, RUWE %s; SIMBAD %s; %s" % (
            o["id"], o["rng"], o["n"], o["g"], o["c"], o["teff"], o["logg"], o["pwd"], o["excess"], o["ruwe"], o["simbad"], o["vet"] or "unexplained"))
    json.dump(out, open(core.results("white_dwarfs_varying.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_white_dwarfs_varying", "White dwarfs varying outside the pulsation strips", "Gaia DR3 vari_summary and gaia_source; Gentile Fusillo et al. (2021) white dwarfs "
            "(VizieR J/MNRAS/508/3877); SIMBAD", "catalogued white dwarfs within %.0f pc that Gaia flags variable with a G range above %g mag: %d" % (1000 / plx_min, range_min, len(out)),
            os.path.abspath(__file__), "wdvary", len(out), "white dwarfs", dict(plx_min=plx_min, range_min=range_min, pwd_min=pwd_min, strips_k=WD_STRIPS[:2]),
            "the pulsation strips, SIMBAD's types, the BP/RP flux excess and RUWE", "a white dwarf varying by a tenth of a magnitude outside every pulsation strip",
            "%d: %d in a strip or of a known kind, %d with a companion's signature; %d unexplained" % (
                len(out), sum(o["vet"] == "natural-type" for o in out), sum(o["vet"] == "companion" for o in out), len(left)),
            [dict(target="gaia_dr3:%s" % o["id"], stat=o["rng"], unit="G range, mag", vet=o["vet"],
                  note="%s transits, Teff %s K, log g %s, BP-RP %.2f, excess %+.3f, RUWE %s; SIMBAD %s" % (o["n"], o["teff"], o["logg"], o["c"], o["excess"], o["ruwe"], o["simbad"]))
             for o in out])
        print("recorded")


copyebs_DOC = r"""Copy eclipsing binaries: the Kepler eclipsing binary catalogue (Prsa et al. 2011, VizieR J/AJ/141/83), pairs of systems whose periods agree within the catalogue's own quantum of 0.00001 d, two clocks set alike, and among them the pairs whose temperature ratios and summed radii agree to 10 per cent as copies would, against the count the periods' own density gives in a window ten times wider and the pairs within 2 arcmin, where one binary's light leaks into another's aperture and the catalogue lists one signal twice.

    python -I seti.py flashes copyebs [record] [--tol-d 0.00001] [--like 0.1] [--blend-arcmin 2]"""


def copyebs_main(record=False, tol_d=0.00001, like=0.1, blend_arcmin=2.0):
    from scipy.stats import poisson
    rows = core.tap("vizier", 'SELECT KIC, Per, "T2/T1", "r1+r2", "_RA", "_DE" FROM "J/AJ/141/83/table1" WHERE Per > 0')
    st = sorted([dict(kic=str(r["KIC"]), P=num(r["Per"]), ra=num(r["_RA"]), dec=num(r["_DE"]), t21=num(r["T2/T1"]), rr=num(r["r1+r2"])) for r in rows if num(r["Per"])],
                key=lambda s: s["P"])
    P = np.array([s["P"] for s in st])

    def pairs_within(w):
        out = []
        for i in range(len(st)):
            j = i + 1
            while j < len(st) and P[j] - P[i] <= w:
                out.append((st[i], st[j]))
                j += 1
        return out
    pairs = pairs_within(tol_d)
    # the periods' own density: the pairs within a window ten times wider, scaled down
    exp = len(pairs_within(10 * tol_d)) / 10.0

    def alike(x, y):
        return x is not None and y is not None and abs(x - y) <= like * max(abs(x), abs(y), 1e-9)
    out = []
    for a, b in pairs:
        sep = core.sep_arcsec(a["ra"], a["dec"], b["ra"], b["dec"]) / 60.0
        same = alike(a["t21"], b["t21"]) and alike(a["rr"], b["rr"])
        out.append(dict(kic1=a["kic"], kic2=b["kic"], P1=a["P"], P2=b["P"], dP=abs(a["P"] - b["P"]), sep_arcmin=sep, t21=(a["t21"], b["t21"]), rr=(a["rr"], b["rr"]), copy=same,
                        vet="blend" if sep < blend_arcmin else "chance-rate" if not same else None))
    far = [o for o in out if o["vet"] != "blend"]
    copies = [o for o in far if o["copy"]]
    p = float(poisson.sf(len(far) - 1, exp)) if far else 1.0
    print("%d eclipsing binaries; pairs with periods within %g d: %d, %d within %g arcmin of each other (one signal in two apertures), %d apart on the sky against %.1f expected "
          "from the periods' own density (Poisson p %.3g); of those, alike in temperature ratio and summed radii to %g: %d" % (
              len(st), tol_d, len(out), sum(o["vet"] == "blend" for o in out), blend_arcmin, len(far), exp, p, like, len(copies)))
    for o in sorted(out, key=lambda o: o["dP"]):
        print("  KIC %s and KIC %s: P %.6f and %.6f d (%.1e d apart), %.1f arcmin, T2/T1 %s, r1+r2 %s; %s" % (
            o["kic1"], o["kic2"], o["P1"], o["P2"], o["dP"], o["sep_arcmin"], o["t21"], o["rr"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(st), pairs=out), open(core.results("copy_eclipsing_binaries.json"), "w"), indent=1)
    if record:
        core.record_run(
            "kepler_eb_copies", "Copy eclipsing binaries", "Kepler eclipsing binary catalogue (Prsa et al. 2011, VizieR J/AJ/141/83)", "eclipsing binaries with periods: %d" % len(st),
            os.path.abspath(__file__), "copyebs", len(st), "binaries", dict(tol_d=tol_d, like=like, blend_arcmin=blend_arcmin),
            "the periods' own density gives %.1f pairs within the window; pairs within %g arcmin are one signal in two apertures" % (exp, blend_arcmin),
            "two binaries far apart on the sky with periods within %g d and temperature ratios and summed radii alike to %g" % (tol_d, like),
            "%d pairs within the period window, %d of them blends, %d apart against %.1f expected (Poisson p %.3g); alike in temperature ratio and radii: %d" % (
                len(out), sum(o["vet"] == "blend" for o in out), len(far), exp, p, len(copies)),
            [dict(target="kic:%s" % o["kic1"], stat=o["dP"], unit="period difference from KIC %s, d" % o["kic2"], vet=o["vet"],
                  note="P %.6f and %.6f d, %.1f arcmin apart, T2/T1 %s, r1+r2 %s" % (o["P1"], o["P2"], o["sep_arcmin"], o["t21"], o["rr"])) for o in out])
        print("recorded")


ebwholedays_DOC = r"""Eclipsing binaries on whole days: the Kepler eclipsing binary catalogue (Prsa et al. 2011, VizieR J/AJ/141/83), periods under 50 days measured from space, where no day's cycle aliases a period, those within 0.0002 d of a whole number of days, a binary set to Earth's day, against the same windows at half-integers and the count the periods' own density gives; the catalogue's periods are quoted to 0.00001 d, so none lands on the grid by rounding.

    python -I seti.py flashes ebwholedays [record] [--tol-d 0.0002] [--p-max-d 50]"""


def ebwholedays_main(record=False, tol_d=0.0002, p_max_d=50.0):
    from scipy.stats import poisson
    rows = core.tap("vizier", 'SELECT KIC, Per, "T2/T1", "r1+r2" FROM "J/AJ/141/83/table1" WHERE Per > 0')
    ps = [dict(kic=str(r["KIC"]).strip(), P=float(r["Per"]), ratio=num(r["T2/T1"]), radii=num(r["r1+r2"])) for r in rows]
    ps = [p for p in ps if 0.5 < p["P"] < p_max_d]
    whole = [p for p in ps if abs(p["P"] - round(p["P"])) < tol_d and round(p["P"]) >= 1]
    half = [p for p in ps if abs(p["P"] - (math.floor(p["P"]) + 0.5)) < tol_d]
    exp = 2 * tol_d * len(ps)
    p = float(poisson.sf(len(whole) - 1, exp)) if whole else 1.0
    for w in whole:
        w["vet"] = "chance-rate" if p > 0.01 else None
    print("%d eclipsing binaries with periods under %g d; within %g d of a whole day: %d (%.2f expected from the periods' density, Poisson p %.3g); at half-days: %d" % (
        len(ps), p_max_d, tol_d, len(whole), exp, p, len(half)))
    for w in sorted(whole, key=lambda w: abs(w["P"] - round(w["P"]))):
        print("  KIC %s: P %.5f d (off %d by %.5f), T2/T1 %s, r1+r2 %s; %s" % (w["kic"], w["P"], round(w["P"]), w["P"] - round(w["P"]), w["ratio"], w["radii"], w["vet"] or "unexplained"))
    json.dump(dict(n=len(ps), whole=whole, half=len(half)), open(core.results("kepler_eb_whole_days.json"), "w"), indent=1)
    if record:
        core.record_run(
            "kepler_eb_whole_days", "Eclipsing binaries on whole days", "Kepler eclipsing binary catalogue (Prsa et al. 2011, VizieR J/AJ/141/83)",
            "eclipsing binaries with periods between 0.5 and %g d: %d" % (p_max_d, len(ps)), os.path.abspath(__file__), "ebwholedays", len(ps), "binaries", dict(tol_d=tol_d, p_max_d=p_max_d),
            "the periods' density gives %.2f in the windows; the half-day windows hold %d" % (exp, len(half)), "a period within %g d of a whole number of days" % tol_d,
            "%d against %.2f (Poisson p %.3g), %d at half-days" % (len(whole), exp, p, len(half)),
            [dict(target="kic:%s" % w["kic"], stat=round(w["P"] - round(w["P"]), 5), unit="d from the whole day", vet=w["vet"], note="P %.5f d, T2/T1 %s, r1+r2 %s" % (w["P"], w["ratio"], w["radii"]))
             for w in whole])
        print("recorded")


tesswholedays_DOC = r"""Rotation periods on whole days in TESS: the Fetherolf et al. (2023) catalogue of TESS rotation periods (VizieR J/ApJS/266/22), one period per star from space with no day's cycle to alias it, periods between 0.5 and 13 days measured to 0.005 d, those within 0.005 d of a whole number of days, a star turning on Earth's day, against the same windows at half-integers and the count the periods' own density gives.

    python -I seti.py flashes tesswholedays [record] [--tol-d 0.005] [--p-max-d 13] [--err-max-d 0.005]"""


def tesswholedays_main(record=False, tol_d=0.005, p_max_d=13.0, err_max_d=0.005):
    from scipy.stats import poisson
    from setitools.highenergy import dynamo_vizier
    t = dynamo_vizier("fetherolf")[0]
    best = {}
    for r in t:
        P, e = num(r["Pvar"]), num(r["e_Pvar"])
        tic = str(r["TIC"]).strip()
        if P is None or e is None or e > err_max_d or not 0.5 < P < p_max_d:
            continue
        if tic not in best or e < best[tic]["e"]:
            best[tic] = dict(tic=tic, P=P, e=e, sector=str(r["Sec"]).strip(), tmag=num(r["Tmag"]), teff=num(r["Teff"]))
    ps = list(best.values())
    whole = [p for p in ps if abs(p["P"] - round(p["P"])) < tol_d and round(p["P"]) >= 1]
    half = [p for p in ps if abs(p["P"] - (math.floor(p["P"]) + 0.5)) < tol_d]
    exp = 2 * tol_d * len(ps)
    p = float(poisson.sf(len(whole) - 1, exp)) if whole else 1.0
    for w in whole:
        w["vet"] = "chance-rate" if p > 0.01 else None
    print("%d stars with a TESS rotation period between 0.5 and %g d to %g d; within %g d of a whole day: %d (%.2f expected from the periods' density, Poisson p %.3g); at half-days: %d" % (
        len(ps), p_max_d, err_max_d, tol_d, len(whole), exp, p, len(half)))
    for w in sorted(whole, key=lambda w: abs(w["P"] - round(w["P"]))):
        print("  TIC %s: P %.4f +- %.4f d (off %d by %.4f), sector %s, Tmag %s, %s K; %s" % (w["tic"], w["P"], w["e"], round(w["P"]), w["P"] - round(w["P"]), w["sector"], w["tmag"], w["teff"],
                                                                                        w["vet"] or "unexplained"))
    json.dump(dict(n=len(ps), whole=whole, half=len(half)), open(core.results("tess_rotation_whole_days.json"), "w"), indent=1)
    if record:
        core.record_run(
            "tess_rotation_whole_days", "Rotation periods on whole days in TESS", "Fetherolf et al. (2023) TESS rotation periods (VizieR J/ApJS/266/22)",
            "stars with a rotation period between 0.5 and %g d to %g d: %d" % (p_max_d, err_max_d, len(ps)), os.path.abspath(__file__), "tesswholedays", len(ps), "stars",
            dict(tol_d=tol_d, p_max_d=p_max_d, err_max_d=err_max_d), "the periods' density gives %.2f in the windows; the half-day windows hold %d" % (exp, len(half)),
            "a rotation period within %g d of a whole number of days" % tol_d, "%d against %.2f (Poisson p %.3g), %d at half-days" % (len(whole), exp, p, len(half)),
            [dict(target="tic:%s" % w["tic"], stat=round(w["P"] - round(w["P"]), 4), unit="d from the whole day", vet=w["vet"], note="P %.4f +- %.4f d, sector %s, %s K" % (w["P"], w["e"], w["sector"], w["teff"]))
             for w in whole])
        print("recorded")


ebtwins_DOC = r"""Eclipsing twins to a part in a thousand: the Kepler eclipsing binary catalogue (Prsa et al. 2011, VizieR J/AJ/141/83), systems whose fitted temperature ratio T2/T1 lies within 0.001 of one, two stars alike to a tenth of a per cent, a pair made to one measure, against the count the ratios' own density within 0.05 of one gives; a fit that holds the ratio at exactly 1.000 has pegged a parameter at its boundary, so each is read for the exact value and for its summed radius.

    python -I seti.py flashes ebtwins [record] [--tol 0.001] [--band 0.05]"""


def ebtwins_main(record=False, tol=0.001, band=0.05):
    from scipy.stats import poisson
    rows = core.tap("vizier", 'SELECT KIC, Per, "T2/T1", "r1+r2" FROM "J/AJ/141/83/table1" WHERE "T2/T1" > 0')
    st = [dict(kic=str(r["KIC"]).strip(), P=num(r["Per"]), ratio=num(r["T2/T1"]), radii=num(r["r1+r2"])) for r in rows]
    st = [s for s in st if s["ratio"] is not None]
    near = [s for s in st if abs(s["ratio"] - 1) < band]
    twins = [s for s in st if abs(s["ratio"] - 1) < tol]
    pegged = [s for s in twins if s["ratio"] == 1.0]
    exp = len(near) * (2 * tol) / (2 * band)
    free = [s for s in twins if s["ratio"] != 1.0]
    p = float(poisson.sf(len(free) - 1, exp)) if free else 1.0
    for s in twins:
        s["vet"] = "fit-boundary" if s["ratio"] == 1.0 else ("chance-rate" if p > 0.01 else None)
    left = [s for s in twins if s["vet"] is None]
    print("%d eclipsing binaries with a temperature ratio, %d within %g of one; within %g: %d, %d of them at exactly 1.000 (the fit's boundary) and %d free against %.1f from the density "
          "(Poisson p %.3g); unexplained %d" % (len(st), len(near), band, tol, len(twins), len(pegged), len(free), exp, p, len(left)))
    for s in sorted(free, key=lambda s: abs(s["ratio"] - 1)):
        print("  KIC %s: T2/T1 %.4f, r1+r2 %s, P %s d; %s" % (s["kic"], s["ratio"], s["radii"], s["P"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), near=len(near), twins=twins), open(core.results("kepler_eb_twins.json"), "w"), indent=1)
    if record:
        core.record_run(
            "kepler_eb_twins", "Eclipsing twins to a part in a thousand", "Kepler eclipsing binary catalogue (Prsa et al. 2011, VizieR J/AJ/141/83)",
            "eclipsing binaries with a fitted temperature ratio: %d, %d within %g of one" % (len(st), len(near), band), os.path.abspath(__file__), "ebtwins", len(st), "binaries",
            dict(tol=tol, band=band), "the ratios' density within %g of one: %.1f expected within %g; fits pegged at exactly 1.000: %d" % (band, exp, tol, len(pegged)),
            "a temperature ratio within %g of one not pegged by the fit" % tol, "%d within %g, %d pegged at the boundary, %d free against %.1f (Poisson p %.3g); %d unexplained" % (
                len(twins), tol, len(pegged), len(free), exp, p, len(left)),
            [dict(target="kic:%s" % s["kic"], stat=round(s["ratio"], 4), unit="T2/T1", vet=s["vet"], note="r1+r2 %s, P %s d" % (s["radii"], s["P"])) for s in twins])
        print("recorded")


vascobelt_DOC = r"""VASCO transients and the geosynchronous belt: the Villarroel et al. (2020) lists of sources on POSS-I plates with no counterpart on later images (VizieR J/AJ/159/8), their declinations against the belt of geosynchronous satellites, which from Palomar stands at -5.44 deg, the band from -6.1 to -4.6 deg; a glint from something in that orbit on a 1950s plate would be a reflector there before the first launch, and the count in the band is taken against the band's share of the plates' sky, which spans the declinations the lists reach.

    python -I seti.py flashes vascobelt [record] [--belt-lo-deg -6.1] [--belt-hi-deg -4.6]"""


def vascobelt_main(record=False, belt_lo_deg=-6.1, belt_hi_deg=-4.6):
    from scipy.stats import binomtest
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    dec, n_tab = [], {}
    for t in core.vizier("J/AJ/159/8"):
        name = t.meta.get("name", "")
        if "DEJ2000" not in t.colnames:
            continue
        c = SkyCoord([str(x) for x in t["RAJ2000"]], [str(x) for x in t["DEJ2000"]], unit=(u.hourangle, u.deg), frame="icrs")
        dec += list(c.dec.deg)
        n_tab[name] = len(t)
    dec = np.array(dec)
    lo, hi = float(dec.min()), float(dec.max())
    share = (math.sin(math.radians(belt_hi_deg)) - math.sin(math.radians(belt_lo_deg))) / (math.sin(math.radians(hi)) - math.sin(math.radians(lo)))
    n = int(np.sum((dec >= belt_lo_deg) & (dec <= belt_hi_deg)))
    p = binomtest(n, len(dec), share, alternative="greater").pvalue if len(dec) else 1.0
    print("%d transient positions from %s spanning declinations %.1f to %.1f; in the belt's band %g to %g deg: %d against %.2f from the band's share of that sky (binomial p %.3g)" % (
        len(dec), n_tab, lo, hi, belt_lo_deg, belt_hi_deg, n, share * len(dec), p))
    if record:
        core.record_run(
            "vasco_geosynchronous_belt", "VASCO transients and the geosynchronous belt", "Villarroel et al. (2020) POSS-I transient candidates (VizieR J/AJ/159/8)",
            "transient positions: %d, spanning declinations %.1f to %.1f" % (len(dec), lo, hi), os.path.abspath(__file__), "vascobelt", len(dec), "positions",
            dict(belt_lo_deg=belt_lo_deg, belt_hi_deg=belt_hi_deg), "the band's share of the sky between the lists' extreme declinations: %.4f" % share,
            "transients crowding the declination of the geosynchronous belt seen from Palomar", "%d in the band against %.2f (binomial p %.3g)" % (n, share * len(dec), p), [])
        print("recorded")


twomessengers_DOC = r"""Two messengers at once: the IceCube alert tracks of IceCat-1 (Abbasi et al. 2023, VizieR J/ApJS/269/25) that arrived while TESS was observing for the Yang et al. (2023) flare catalogue (sectors 1-30, VizieR J/A+A/669/A15), the catalogue's stars inside each track's 90 per cent region, and their flares peaking within an hour of the neutrino, against the coincidences the same regions give with the alert times shifted by up to 100 days; a transmitter that fires a neutrino beam and lights its star, or a flare that makes neutrinos, would pair the two, and the stars within 100 pc are counted on their own.

    python -I seti.py flashes twomessengers [record] [--window-h 1] [--shifts 2000] [--near-pc 100]"""


def twomessengers_main(record=False, window_h=1.0, shifts=2000, near_pc=100.0):
    al = [t for t in core.vizier("J/ApJS/269/25") if t.meta.get("name", "").endswith("table2")][0]
    st = core.tap("vizier", "SELECT TIC, RAJ2000, DEJ2000, DistG FROM %s" % YANG_STARS, timeout=900)
    fl = core.tap("vizier", "SELECT TIC, TimeP FROM %s" % YANG_FLARES, timeout=900)
    times = {}
    for r in fl:
        if num(r["TimeP"]) is not None:
            times.setdefault(str(r["TIC"]), []).append(num(r["TimeP"]) + 56999.5)
    lo, hi = min(min(v) for v in times.values()), max(max(v) for v in times.values())
    stars = [(str(r["TIC"]), num(r["RAJ2000"]), num(r["DEJ2000"]), num(r["DistG"])) for r in st if str(r["TIC"]) in times and num(r["RAJ2000"]) is not None]
    ra_s = np.array([s[1] for s in stars])
    de_s = np.array([s[2] for s in stars])
    alerts = []
    for a in al:
        t = float(a["MJD"])
        if not lo <= t <= hi:
            continue
        ra, de = float(a["RAJ2000"]), float(a["DEJ2000"])
        dra = (ra_s - ra + 180) % 360 - 180
        dde = de_s - de
        inside = np.nonzero((dra <= float(a["E_RAJ2000"])) & (dra >= -float(a["e_RAJ2000"])) & (dde <= float(a["E_DEJ2000"])) & (dde >= -float(a["e_DEJ2000"])))[0]
        ft = [(stars[k][0], stars[k][3], tt) for k in inside for tt in times[stars[k][0]]]
        alerts.append(dict(alert=str(a["Alert"]).strip(), mjd=t, ra=ra, de=de, energy=num(a["Energy"]), n_stars=len(inside), flares=ft))
    w = window_h / 24.0

    def hits(shift, near=False):
        return sum(1 for a in alerts for tic, d, tt in a["flares"] if abs(tt - (a["mjd"] + shift)) < w and (not near or (d is not None and d < near_pc)))
    real, real_near = hits(0.0), hits(0.0, True)
    rng = np.random.default_rng(9)
    off = rng.uniform(-100, 100, shifts)
    off = off[np.abs(off) > 2 * w]
    null = np.array([hits(s) for s in off])
    null_near = np.array([hits(s, True) for s in off])
    p, p_near = float((null >= real).mean()) if real else 1.0, float((null_near >= real_near).mean()) if real_near else 1.0
    found = [(a["alert"], tic, d, round((tt - a["mjd"]) * 24, 2)) for a in alerts for tic, d, tt in a["flares"] if abs(tt - a["mjd"]) < w]
    print("%d IceCat-1 alerts between MJD %.0f and %.0f; %d flaring stars inside their regions (%d flares); flares within %g h: %d against %.3f from %d shifts (p %.3g); "
          "within %g pc: %d against %.3f (p %.3g); %s" % (len(alerts), lo, hi, sum(a["n_stars"] for a in alerts), sum(len(a["flares"]) for a in alerts), window_h, real,
                                                         null.mean(), len(off), p, near_pc, real_near, null_near.mean(), p_near, found))
    if record:
        core.record_run(
            "icecat_tess_flare_coincidence", "Two messengers at once", "IceCat-1 (VizieR J/ApJS/269/25); Yang et al. (2023) TESS flares (VizieR J/A+A/669/A15)",
            "IceCube alert tracks during TESS sectors 1-30: %d, with %d flaring stars inside their 90 per cent regions" % (len(alerts), sum(a["n_stars"] for a in alerts)),
            os.path.abspath(__file__), "twomessengers", len(alerts), "alerts", dict(window_h=window_h, shifts=shifts, near_pc=near_pc),
            "%d shifts of the alert times by up to 100 days: %.3f coincidences, %.3f within %g pc" % (len(off), null.mean(), null_near.mean(), near_pc),
            "a flare on a star inside a neutrino track's region within %g h of the neutrino" % window_h,
            "%d coincidences against %.3f (p %.3g), %d within %g pc against %.3f" % (real, null.mean(), p, real_near, near_pc, null_near.mean()),
            [dict(target="tic:%s" % tic, stat=dt_h, unit="hours from the neutrino", vet=None, note="%s, %s pc" % (alert, d)) for alert, tic, d, dt_h in found])
        print("recorded")


flareecho_DOC = r"""Flare echoes: every flare of a flare star in Kepler's one-minute photometry (GJ 1243 by default), or with --tess true of the nearest flare stars in TESS's twenty-second and two-minute photometry (Proxima Centauri, AU Mic, YZ CMi, EV Lac, AD Leo, GJ 1243, Wolf 359 and UV Ceti), or with --nearby true of every star within 30 pc brighter than Tmag 15 whose twenty-second light curves hold 30 flares, the light curve made relative to an hour's running median and then to nine minutes', each flare's five-minute impulse correlated with the light curve after it at every delay from 3 to 180 minutes and the flares averaged with weights of their impulse's signal-to-noise, every other peak at 5 widths masked, the secondary bursts of complex flares among them; a mirror, a ring or a swarm in orbit returns each flare as a faint copy after the light-travel delay of its own geometry, 17 minutes from 1 au straight back, where a disc or a shell spreads its echo over a range of delays and sympathetic flares come at any delay; the same correlation before each flare, where no echo can arrive, is the null, a delay that loses half its echo when one flare is left out rests on that flare, and VSX names the eclipsing binaries whose eclipse edges pass for flares.

    python -I seti.py flashes flareecho [record] [--kic 9726699] [--tess false] [--nearby false] [--min-flares 30] [--snr-min 10] [--tau-max 180] [--z-min 5] [--mask-snr 5]"""

FLAREECHO_TESS = {"Proxima Cen": 388857263, "AU Mic": 441420236, "YZ CMi": 266744225, "EV Lac": 154101678, "AD Leo": 95431305, "GJ 1243": 273589987,
                  "Wolf 359": 365006789, "UV Cet": 632499595}


# the high-pass spans nine minutes and the impulse five, wide enough to hold a flare's rise and peak whole
FLAREECHO_HP_MIN, FLAREECHO_IMPULSE_MIN = 9.0, 5.0


def flareecho_odd(minutes, cad):
    """An odd number of cadences spanning about the given minutes, three at least."""
    return max(3, int(round(minutes / cad)) | 1)


def flareecho_prepare(t, f, cad):
    """A light curve in minutes as (minutes, relative flux, its high-pass, noise, cadence): flux over an hour's running median, less the relative flux's running median over FLAREECHO_HP_MIN."""
    from scipy.ndimage import median_filter
    r = f / median_filter(f, size=flareecho_odd(60, cad), mode="nearest") - 1.0
    hp = r - median_filter(r, size=flareecho_odd(FLAREECHO_HP_MIN, cad), mode="nearest")
    return t, r, hp, 1.4826 * float(np.median(np.abs(r - np.median(r)))), cad


def flareecho_kepler(kic):
    """A Kepler star's short-cadence light curves, PDCSAP flux of quality 0, prepared per file."""
    from astropy.io import fits
    out = []
    for key in core.kepler_keys(kic, short=True):
        local = core.data("flareecho", os.path.basename(key))
        if not os.path.exists(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            body = core.get(core.S3 + key, timeout=600)
            open(local, "wb").write(body)
        with fits.open(local) as h:
            d = h[1].data
            t, f, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["SAP_QUALITY"])
        ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
        out.append(flareecho_prepare(t[ok] * 1440.0, f[ok], 0.98084))
    return out


def flareecho_tess(tic, sectors=range(1, 100), threads=8):
    """A TESS star's SPOC light curves, twenty-second where a sector has them and two-minute otherwise, PDCSAP flux of quality 0, prepared per sector."""
    from astropy.io import fits

    def one(sector):
        for fast in (True, False):
            local = core.data("flareecho", "tic%d_s%04d_%s.fits" % (tic, sector, "fast" if fast else "2min"))
            if not os.path.exists(local):
                keys = core.spoc_keys(tic, sector, fast)
                if not keys:
                    continue
                body = core.get(core.S3 + keys[0], timeout=600, retries=3, soft=True, pause=2)
                if not body or not body.startswith(b"SIMPLE"):
                    continue
                os.makedirs(os.path.dirname(local), exist_ok=True)
                open(local, "wb").write(body)
            with fits.open(local) as h:
                d = h[1].data
                t, f, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"])
            ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
            if ok.sum() < 1000:
                return None
            t = t[ok] * 1440.0
            return flareecho_prepare(t, f[ok], float(np.median(np.diff(t))))
        return None
    with ThreadPoolExecutor(threads) as ex:
        return [c for c in ex.map(one, sectors) if c is not None]


def flareecho_stack(curves, snr_min, tau_max, mask_snr=5.0):
    """The echo depth of each flare's impulse at every delay of whole minutes from -tau_max to tau_max, weighted over the flares, with their count; every peak at mask_snr widths masked, the secondary bursts of complex flares among them."""
    taus = np.arange(-tau_max, tau_max + 1)
    num_, den, n = np.zeros(len(taus)), np.zeros(len(taus)), 0
    for t, r, hp, sig, cad in curves:
        h = flareecho_odd(FLAREECHO_IMPULSE_MIN, cad) // 2
        # a flare: a local peak at snr_min noise widths with a neighbour above three
        peaks = [i for i in range(h, len(r) - h) if r[i] >= snr_min * sig and r[i] == r[i - h:i + h + 1].max() and max(r[i - 1], r[i + 1]) >= 3 * sig]
        busy = np.zeros(len(r), bool)
        for i in [i for i in range(h, len(r) - h) if r[i] >= mask_snr * sig and r[i] == r[i - h:i + h + 1].max()]:
            busy[max(i - int(3 / cad) - h, 0):i + int(6 / cad) + h + 1] = True
        # cadences on one grid, so that a delay counts in steps even across gaps
        step = np.round((t - t[0]) / cad).astype(int)
        where = np.full(step.max() + 1, -1)
        where[step] = np.arange(len(step))
        shift = np.round(taus / cad).astype(int)
        for i in peaks:
            g = hp[i - h:i + h + 1]
            gg = float(g @ g)
            if gg <= 0:
                continue
            n += 1
            w = gg / sig ** 2
            for k, tau in enumerate(taus):
                j = step[i] + shift[k]
                if abs(tau) < 3 or j - h < 0 or j + h + 1 > len(where):
                    continue
                idx = where[j - h:j + h + 1]
                if (idx < 0).any() or busy[idx].any():
                    continue
                num_[k] += w * float(g @ hp[idx]) / gg
                den[k] += w
    return taus, np.where(den > 0, num_ / np.maximum(den, 1e-30), np.nan), n


def flareecho_score(taus, eps, z_min):
    """The echo's strongest delay after the flares in widths of its spread before them, the largest before them, and the delays at z_min or more."""
    pre, post = (taus <= -3) & np.isfinite(eps), (taus >= 3) & np.isfinite(eps)
    med = float(np.median(eps[pre]))
    width = 1.4826 * float(np.median(np.abs(eps[pre] - med)))
    z = (eps - med) / max(width, 1e-12)
    k = int(np.nanargmax(np.where(post, z, -np.inf)))
    hits = [(int(taus[j]), float(eps[j]), float(z[j])) for j in np.flatnonzero(post & (z >= z_min))]
    return dict(tau=int(taus[k]), eps=float(eps[k]), z=float(z[k]), pre=float(np.nanmax(np.abs(z[pre]))), width=width, hits=hits)


def flareecho_nearby(min_flares, procs=8, tmag_max=15.0):
    """Every star of the twenty-second light curves within 30 pc (the discrete-level search's set) brighter than tmag_max with min_flares flares at ten noise widths, its sectors prepared, as (name, target, curves)."""
    import glob
    from concurrent.futures import ProcessPoolExecutor
    from astropy.io import fits
    files = [f for f in sorted(glob.glob(core.data("tess20", "*.fits"))) if (fits.getheader(f).get("TESSMAG") or 99) <= tmag_max]
    with ProcessPoolExecutor(procs) as ex:
        curves = list(ex.map(flareecho_file, files, chunksize=8))
    by = {}
    for path, c in zip(files, curves):
        if c is not None:
            by.setdefault(int(fits.getheader(path).get("TICID")), []).append(c)
    out = []
    for tic, cs in sorted(by.items()):
        n = 0
        for t, r, hp, sig, cad in cs:
            h = flareecho_odd(FLAREECHO_IMPULSE_MIN, cad) // 2
            n += sum(1 for i in range(h, len(r) - h) if r[i] >= 10 * sig and r[i] == r[i - h:i + h + 1].max() and max(r[i - 1], r[i + 1]) >= 3 * sig)
        if n >= min_flares:
            out.append(("TIC %d" % tic, "tic:%d" % tic, cs))
    return out


def flareecho_file(path):
    """One twenty-second light curve file prepared, or None."""
    from astropy.io import fits
    with fits.open(path) as h:
        d = h[1].data
        t, f, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"])
    ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
    if ok.sum() < 1000:
        return None
    t = t[ok] * 1440.0
    return flareecho_prepare(t, f[ok], float(np.median(np.diff(t))))


def flareecho_main(record=False, kic=9726699, tess=False, nearby=False, min_flares=30, snr_min=10.0, tau_max=180, z_min=5.0, mask_snr=5.0):
    if nearby:
        stars = flareecho_nearby(min_flares)
    elif tess:
        stars = [(name, "tic:%d" % tic, flareecho_tess(tic)) for name, tic in FLAREECHO_TESS.items()]
    else:
        stars = [("KIC %d" % kic, "kic:%d" % kic, flareecho_kepler(kic))]
    out = []
    for name, target, curves in stars:
        if not curves:
            print("  %s: no light curves" % name)
            continue
        taus, eps, n = flareecho_stack(curves, snr_min, tau_max, mask_snr)
        if n == 0 or not np.isfinite(eps).any():
            print("  %s: no flare at %g widths in %d curves" % (name, snr_min, len(curves)))
            continue
        sc = flareecho_score(taus, eps, z_min)
        sc.update(name=name, target=target, n=n, curves=len(curves), cadences=sorted({round(c[4], 2) for c in curves}), vet=None, loo=None)
        if sc["hits"] and sc["z"] > sc["pre"]:
            # an echo returns after every flare: one that falls by half with a single flare left out rests on that flare
            k = int(np.flatnonzero(taus == sc["tau"])[0])
            low = sc["eps"]
            for ci, (t, r, hp, sig, cad) in enumerate(curves):
                h = flareecho_odd(FLAREECHO_IMPULSE_MIN, cad) // 2
                for i in [i for i in range(h, len(r) - h) if r[i] >= snr_min * sig and r[i] == r[i - h:i + h + 1].max() and max(r[i - 1], r[i + 1]) >= 3 * sig]:
                    r2 = r.copy()
                    r2[i - h:i + h + 1] = np.minimum(r2[i - h:i + h + 1], 0.9 * snr_min * sig)
                    low = min(low, float(flareecho_stack(curves[:ci] + [(t, r2, hp, sig, cad)] + curves[ci + 1:], snr_min, tau_max, mask_snr)[1][k]))
            sc["loo"] = low
            if low < 0.5 * sc["eps"]:
                sc["vet"] = "single-epoch"
        elif sc["hits"]:
            sc["vet"] = "chance-rate"
        out.append(sc)
        if not nearby or (sc["hits"] and sc["z"] > sc["pre"]):
            print("%s: %d flares at %g noise widths or more in %d light curves (cadences %s min); echo spread per delay before the flares %.2g; strongest after them %.4f at %d min (%.1f widths), largest before them %.1f widths; %d delays at %g widths or more" % (
                name, n, snr_min, len(curves), ", ".join("%g" % c for c in sc["cadences"]), sc["width"], sc["eps"], sc["tau"], sc["z"], sc["pre"], len(sc["hits"]), z_min))
            for tau, e, zz in sc["hits"]:
                print("  delay %4d min: echo %.4f of the flare (%.1f widths)" % (tau, e, zz))
    # an eclipsing binary's eclipse edges pass the flare finder once its hour's median is taken out
    cand = [o for o in out if o["hits"] and o["z"] > o["pre"] and o["vet"] is None and o["target"].startswith("tic:")]
    if cand:
        pos = {}
        for o in cand:
            r_ = core.tap("simbad", "SELECT b.ra, b.dec FROM ident AS i JOIN basic AS b ON b.oid = i.oidref WHERE i.id = 'TIC %s'" % o["target"][4:])
            if r_:
                pos[o["target"]] = (float(r_[0]["ra"]), float(r_[0]["dec"]))
        vsx = {m["source_id"]: (m.get("Type") or "").strip() for m in core.xmatch([dict(source_id=k, ra=v[0], dec=v[1]) for k, v in pos.items()], "B/vsx/vsx", 5.0)} if pos else {}
        for o in cand:
            if vsx.get(o["target"], "").startswith("E"):
                o["vet"] = "eclipsing-binary"
    for o in out:
        if o["hits"] and o["z"] > o["pre"]:
            print("  %s: echo %.4f at %d min (%.1f widths, above the %.1f before the flares); with one flare left out as low as %s; %s" % (
                o["name"], o["eps"], o["tau"], o["z"], o["pre"], "%.4f" % o["loo"] if o["loo"] is not None else "-", o["vet"] or "LEFT"))
    if nearby:
        print("%d stars within 30 pc brighter than Tmag 15 with %d or more flares at %g widths; %d with a delay after their flares at %g widths above the largest before them, %d of them left; median spread per delay %.2g of the flare" % (
            len(out), min_flares, snr_min, sum(1 for o in out if o["hits"] and o["z"] > o["pre"]), z_min, sum(1 for o in out if o["hits"] and o["z"] > o["pre"] and o["vet"] is None),
            float(np.median([o["width"] for o in out])) if out else float("nan")))
    json.dump(out, open(core.results("flare_echoes_%s.json" % ("nearby" if nearby else "tess" if tess else kic)), "w"), indent=1)
    if record and out:
        core.record_run(
            "tess20_nearby_flare_echoes" if nearby else "tess_flare_echoes" if tess else "kepler_flare_echoes_%d" % kic, "Flare echoes",
            ("TESS SPOC twenty-second light curves of the stars within 30 pc" if nearby else "TESS SPOC twenty-second and two-minute" if tess else "Kepler short-cadence") + " PDCSAP light curves from the STScI AWS mirror",
            "; ".join("%s: %d flares" % (o["name"], o["n"]) for o in out), os.path.abspath(__file__), "flareecho", sum(o["n"] for o in out), "flares",
            dict(kic=kic, tess=tess, nearby=nearby, min_flares=min_flares, snr_min=snr_min, tau_max=tau_max, z_min=z_min, mask_snr=mask_snr),
            "the same correlation at delays before each star's flares, spread per delay %s of the flare" % ", ".join("%.2g" % o["width"] for o in out),
            "a copy of each flare's impulse at one delay of 3-%d minutes; %g widths is an echo of %s of the flare" % (tau_max, z_min, ", ".join("%.2g" % (z_min * o["width"]) for o in out)),
            "%d of %d stars with a delay at %g widths or more after their flares, %d of them above the largest before their flares, %d of those left once single flares and eclipses are read" % (
                sum(1 for o in out if o["hits"]), len(out), z_min, sum(1 for o in out if o["hits"] and o["z"] > o["pre"]),
                sum(1 for o in out if o["hits"] and o["z"] > o["pre"] and o["vet"] is None)),
            [dict(target=o["target"], stat=round(o["z"], 2), unit="widths of the echo spread before the flares, at the strongest delay",
                  vet=o["vet"] if o["vet"] else (None if o["z"] >= max(z_min, o["pre"]) else "chance-rate"),
                  note="%s: echo %.4f of the flare at %d min; spread %.2g; largest before the flares %.1f widths; %d flares%s" % (
                      o["name"], o["eps"], o["tau"], o["width"], o["pre"], o["n"], "; with one flare left out %.4f" % o["loo"] if o["loo"] is not None else ""))
             for o in out])
        print("recorded")

tones_DOC = r"""Minute tones at the nearest stars: every TESS twenty-second light curve the discrete-level search read (the stars within 30 pc), its hour-long variations and its flares removed, its Fourier amplitude read at every period from 1 to 60 minutes against its own running median; a beacon modulating a star or a swarm of regular structure sounds a pure tone that holds its frequency, where cool dwarfs pulsate too faintly to see at these periods, so a tone belongs to a pulsating white dwarf, delta Scuti or roAp star in the aperture, a hot star of its own, or the spacecraft, and the tone's amplitude in the flux-weighted centroid places the source on or off the target; a frequency other stars share within 0.05 cycles a day is the spacecraft's, a whole multiple of the light curve's strongest slow signal, up to the twentieth, is that signal's harmonic, and VSX and SIMBAD name eclipsing binaries and white dwarfs.

    python -I seti.py flashes tones [record] [--snr-min 6] [--p-min-min 1] [--p-max-min 60] [--offset-max 0.3]"""


def tones_one(path, f_lo, f_hi):
    """A twenty-second light curve's strongest tone between f_lo and f_hi cycles a day: frequency, amplitude, signal-to-noise against the running median of the amplitude spectrum, and the centroid's amplitude there in pixels per unit flux."""
    from astropy.io import fits
    from scipy.ndimage import median_filter
    with fits.open(path) as h:
        H, d = h[0].header, h[1].data
        t, f, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"])
        cad = np.asarray(d["CADENCENO"], int)
        cx, cy = np.asarray(d["MOM_CENTR1"], float), np.asarray(d["MOM_CENTR2"], float)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(cx) & np.isfinite(cy) & (q == 0)
    if ok.sum() < 20000:
        return None
    t, f, cad, cx, cy = t[ok], f[ok], cad[ok], cx[ok], cy[ok]
    r = f / np.median(f) - 1.0
    r = r - median_filter(r, size=181, mode="nearest")
    sig = 1.4826 * float(np.median(np.abs(r)))
    # a flare and the half-hour after it are taken out
    bad = np.zeros(len(r), bool)
    for i in np.flatnonzero(r > 5 * sig):
        bad[max(i - 3, 0):i + 90] = True
    cx = cx - median_filter(cx, size=181, mode="nearest")
    cy = cy - median_filter(cy, size=181, mode="nearest")
    n = int(cad.max() - cad.min() + 1)
    grid = np.zeros((3, n))
    k = cad - cad.min()
    for row, x in enumerate((r, cx, cy)):
        grid[row, k] = np.where(bad, 0.0, x)
    dt = float(np.median(np.diff(t)))
    A = np.abs(np.fft.rfft(grid, axis=1)) * 2 / max((~bad).sum(), 1)
    freq = np.fft.rfftfreq(n, d=dt)
    # the light curve's strongest slow signal, before the hour-long variations are taken out, of which a tone may be a harmonic
    raw = np.zeros(n)
    raw[k] = f / np.median(f) - 1.0
    R = np.abs(np.fft.rfft(raw - raw[k].mean() * (raw != 0)))
    slow = freq > 0.3
    f0 = float(freq[slow][np.argmax(R[slow])])
    band = (freq >= f_lo) & (freq <= f_hi)
    noise = median_filter(A[0], size=401, mode="nearest")
    snr = np.where(band, A[0] / np.maximum(noise, 1e-30), 0)
    j = int(np.argmax(snr))
    return dict(tic=int(H.get("TICID")), sector=int(H.get("SECTOR")), tmag=float(H.get("TESSMAG") or np.nan), radius=H.get("RADIUS"), ra=float(H.get("RA_OBJ")), dec=float(H.get("DEC_OBJ")),
                freq=float(freq[j]), amp=float(A[0][j]), snr=float(snr[j]), offset=float(np.hypot(A[1][j], A[2][j]) / max(A[0][j], 1e-12)), f0=f0)


def tones_main(record=False, snr_min=6.0, p_min_min=1.0, p_max_min=60.0, offset_max=0.3, procs=8):
    import glob
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(glob.glob(core.data("tess20", "*.fits")))
    f_lo, f_hi = 1440.0 / p_max_min, 1440.0 / p_min_min
    with ProcessPoolExecutor(procs) as ex:
        res = [x for x in ex.map(tones_one, files, [f_lo] * len(files), [f_hi] * len(files), chunksize=8) if x]
    tones = [x for x in res if x["snr"] >= snr_min]
    for x in tones:
        x["shared"] = sorted({y["tic"] for y in tones if y["tic"] != x["tic"] and abs(y["freq"] - x["freq"]) < 0.05})
        x["sectors_same"] = sorted({y["sector"] for y in tones if y["tic"] == x["tic"] and abs(y["freq"] - x["freq"]) < 0.05})
    stars = len({x["tic"] for x in res})
    # a tone from the target barely moves the centroid; a neighbour's moves it by its separation in pixels per unit of the target's flux
    for x in tones:
        x["vet"] = "detector-wide" if x["shared"] else ("blend" if x["offset"] > offset_max else None)
        # a harmonic of order up to 20 within two frequency bins of a sector (0.08 cycles a day)
        h = int(round(x["freq"] / x["f0"]))
        x["harmonic"] = h if 2 <= h <= 20 and abs(x["freq"] - h * x["f0"]) <= 0.08 else 0
    if tones:
        sb = {}
        for m in core.xmatch([dict(source_id=str(x["tic"]), ra=x["ra"], dec=x["dec"]) for x in tones], "I/355/gaiadr3", 5.0):
            if m["source_id"] not in sb or float(m["angDist"]) < sb[m["source_id"]][0]:
                sb[m["source_id"]] = (float(m["angDist"]), num(m.get("BP-RP")), num(m.get("Gmag")))
        for x in tones:
            g = sb.get(str(x["tic"]))
            x["bp_rp"], x["g"] = (g[1], g[2]) if g else (None, None)
            # A and F stars pulsate at these periods themselves, delta Scuti and roAp alike
            if x["vet"] is None and x["bp_rp"] is not None and x["bp_rp"] < 0.7:
                x["vet"] = "variable-star"
        # the rest by what the star is: a harmonic of its own slow light curve, an eclipsing binary in VSX, a white dwarf in SIMBAD, which pulsates at minutes
        rest = [x for x in tones if x["vet"] is None]
        vsx = {}
        if rest:
            for m in core.xmatch([dict(source_id=str(x["tic"]), ra=x["ra"], dec=x["dec"]) for x in rest], "B/vsx/vsx", 3.0):
                vsx[m["source_id"]] = (m.get("Type") or "").strip()
            otype = {r_["id"]: r_["otype"] for r_ in core.tap("simbad", "SELECT i.id, b.otype FROM ident AS i JOIN basic AS b ON b.oid = i.oidref WHERE i.id IN (%s)" % ",".join(
                "'TIC %d'" % x["tic"] for x in rest))}
            for x in rest:
                x["vsx"], x["otype"] = vsx.get(str(x["tic"]), ""), otype.get("TIC %d" % x["tic"], "")
                if x["otype"].startswith("WD") or x["vsx"].startswith("ZZ"):
                    x["vet"] = "variable-star"
                elif x["vsx"].startswith("E"):
                    x["vet"] = "eclipsing-binary"
                elif x["harmonic"]:
                    x["vet"] = "rotation"
    left = [x for x in tones if x["vet"] is None]
    print("%d twenty-second light curves of %d stars within 30 pc; tones at %g times the local amplitude between %g and %g minutes: %d, of which %d at frequencies other stars share, %d moving the centroid as a neighbour would, %d on stars that pulsate at minutes (hot stars and white dwarfs), %d harmonics of a slower signal or an eclipsing binary's, %d left" % (
        len(res), stars, snr_min, p_min_min, p_max_min, len(tones), sum(x["vet"] == "detector-wide" for x in tones), sum(x["vet"] == "blend" for x in tones),
        sum(x["vet"] == "variable-star" for x in tones), sum(x["vet"] in ("rotation", "eclipsing-binary") for x in tones), len(left)))
    for x in sorted(tones, key=lambda x: -x["snr"])[:25]:
        print("  TIC %-10d s%02d  %8.3f c/d (%6.2f min)  %7.1f ppm  S/N %5.1f  centroid %.3f px  BP-RP %s  harmonic %d  %s %s  %s" % (
            x["tic"], x["sector"], x["freq"], 1440 / x["freq"], 1e6 * x["amp"], x["snr"], x["offset"], "%.2f" % x["bp_rp"] if x.get("bp_rp") is not None else "-",
            x["harmonic"], x.get("vsx") or "", x.get("otype") or "", x["vet"] or "LEFT"))
    json.dump(dict(n=len(res), tones=tones), open(core.results("tess20_minute_tones.json"), "w"), indent=1)
    if record:
        core.record_run(
            "tess20_minute_tones", "Minute tones at the nearest stars", "TESS SPOC twenty-second PDCSAP light curves of the stars within 30 pc (the discrete-level search's set)",
            "%d light curves of %d stars" % (len(res), stars), os.path.abspath(__file__), "tones", len(res), "light curves",
            dict(snr_min=snr_min, p_min_min=p_min_min, p_max_min=p_max_min, offset_max=offset_max),
            "each amplitude spectrum's running median over 401 bins; a Rayleigh spectrum passes %g times its median once in %.0e bins" % (snr_min, math.exp(-math.log(2) * snr_min ** 2)),
            "a tone of 1-60 minutes at %g times the local amplitude" % snr_min,
            "%d tones: %d shared, %d off the target, %d on hot stars and white dwarfs, %d harmonics of a slower signal or an eclipsing binary's, %d left" % (
                len(tones), sum(x["vet"] == "detector-wide" for x in tones), sum(x["vet"] == "blend" for x in tones), sum(x["vet"] == "variable-star" for x in tones),
                sum(x["vet"] in ("rotation", "eclipsing-binary") for x in tones), len(left)),
            [dict(target="tic:%d" % x["tic"], stat=round(x["snr"], 1), unit="times the local amplitude", vet=x["vet"],
                  note="sector %d, %.3f c/d (%.2f min), %.1f ppm, centroid %.3f px per unit flux, BP-RP %s; harmonic %d of %.4f c/d; VSX %s, SIMBAD %s; shared with %s" % (
                      x["sector"], x["freq"], 1440 / x["freq"], 1e6 * x["amp"], x["offset"], "%.2f" % x["bp_rp"] if x.get("bp_rp") is not None else "-", x["harmonic"], x["f0"],
                      x.get("vsx") or "-", x.get("otype") or "-", ", ".join("TIC %d" % t for t in x["shared"][:5]) or "none")) for x in tones])
        print("recorded")


chirps_DOC = r"""Chirps at the nearest stars: every TESS twenty-second light curve the discrete-level search read (the stars within 30 pc, brighter than Tmag 15), its hour-long variations and its flares removed, cut into back-to-back windows of six hours for drifts up to 16 cycles a day per day and of an hour and a half beyond, each window's power at periods of 1-60 minutes, each frequency levelled by its median over the windows and each window by its median over the frequencies, and the windows summed along every straight frequency track whose drift lies between 1 and 60 cycles a day per day either way, as radio searches de-Doppler a drifting carrier; a beacon that sweeps its modulation, or a modulation whose frequency slides with a source's acceleration, gathers its power along one track, where stars hold their pulsation frequencies and a stationary tone belongs to the tone search; a sum's chance is its Gamma tail, a track that the same light curve's windows in shuffled order match belongs to bursts of power rather than to a sweep, a track that other stars of the same sector carry at p 1e-4 when read along it in absolute time, that the flux-weighted centroid carries as strongly as the flux, or that lies within 15 per cent in frequency and a factor of two in drift of such a track in any sector is the spacecraft's, one that vanishes once the band starts 8 cycles a day higher rides the star's own slower variability, and chirps laid into light curves measure the reach.

    python -I seti.py flashes chirps [record] [--rate-max 60] [--rate-min 1] [--p-max 1e-12] [--tmag-max 15] [--inject 0.0005] [--shuffles 100]"""


def chirps_prepare(path):
    """A twenty-second light curve as (TIC, sector, Tmag, days, relative flux): hour-long variations and flares with the half-hour after them removed, flares set to zero; None when too short."""
    from astropy.io import fits
    from scipy.ndimage import median_filter
    with fits.open(path) as h:
        H, d = h[0].header, h[1].data
        t, f, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"])
        cad = np.asarray(d["CADENCENO"], int)
    ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
    if ok.sum() < 20000:
        return None
    t, f, cad = t[ok], f[ok], cad[ok]
    r = f / np.median(f) - 1.0
    r = r - median_filter(r, size=181, mode="nearest")
    sig = 1.4826 * float(np.median(np.abs(r)))
    bad = np.zeros(len(r), bool)
    for i in np.flatnonzero(r > 5 * sig):
        bad[max(i - 3, 0):i + 90] = True
    n = int(cad.max() - cad.min() + 1)
    x, have = np.zeros(n), np.zeros(n, bool)
    x[cad - cad.min()] = np.where(bad, 0.0, r)
    have[cad - cad.min()] = ~bad
    dt = float(np.median(np.diff(t)))
    return int(H.get("TICID")), int(H.get("SECTOR")), float(H.get("TESSMAG") or 99), float(t[0]), dt, x, have


def chirps_spectrogram(x, have, dt, f_lo, f_hi, window_days):
    """Power of back-to-back windows at frequencies f_lo-f_hi (cycles a day), each frequency over its median in time and then each window over its median, with the windows' centres (days) and the frequency step; windows under 70 per cent filled left out."""
    w = int(round(window_days / dt))
    step = w
    freq = np.fft.rfftfreq(w, d=dt)
    band = (freq >= f_lo) & (freq <= f_hi)
    rows, centres = [], []
    hann = np.hanning(w)
    for a in range(0, len(x) - w + 1, step):
        if have[a:a + w].mean() < 0.7:
            continue
        p = np.abs(np.fft.rfft(x[a:a + w] * hann)) ** 2
        rows.append(p[band])
        centres.append((a + w / 2) * dt)
    P = np.array(rows)
    if len(P):
        # each frequency over its own median in time, as radio searches level each channel, so that steady lines and red noise drop out and a sweep, which passes each frequency briefly, stays
        P = P / np.maximum(np.median(P, axis=0, keepdims=True), 1e-30)
        P = P / np.maximum(np.median(P, axis=1, keepdims=True), 1e-30) * math.log(2)
    return P, np.array(centres), freq[band], 1.0 / window_days


def chirps_logp(acc, cnt):
    """The natural log of the chance that cnt unit exponentials sum to acc or more: the Gamma tail, by its leading term where the tail underflows."""
    from scipy.special import gammaln
    from scipy.stats import gamma
    lp = gamma.logsf(acc, np.maximum(cnt, 1))
    lead = -acc + (np.maximum(cnt, 1) - 1) * np.log(np.maximum(acc, 1e-300)) - gammaln(np.maximum(cnt, 1))
    return np.where(np.isfinite(lp), lp, lead)


def chirps_search(P, centres, df, rates):
    """Each straight track's summed power: for every drift rate the windows' power shifted to the track's frequency at the window's time, summed; the track of least chance as (log p, rate, bin)."""
    K, B = P.shape
    tc = centres - centres.mean()
    best = (np.inf, None, None)
    out = []
    for rate in rates:
        shift = np.round(rate * tc / df).astype(int)
        acc = np.zeros(B)
        cnt = np.zeros(B)
        for k in range(K):
            lo, hi = max(0, -shift[k]), min(B, B - shift[k])
            if hi > lo:
                acc[lo:hi] += P[k, lo + shift[k]:hi + shift[k]]
                cnt[lo:hi] += 1
        good = cnt >= max(3, K // 2)
        if not good.any():
            continue
        lp = np.where(good, chirps_logp(acc, cnt), 0.0)
        j = int(np.argmin(lp))
        out.append((float(lp[j]), float(rate), j, int(cnt[j])))
        if lp[j] < best[0]:
            best = (float(lp[j]), float(rate), j)
    return best, out


def chirps_one(args):
    """One light curve's strongest drifting track and strongest stationary one, as a dict, optionally with a chirp laid in first and with the strongest drifting track of each of SHUFFLES orderings of the windows."""
    path, f_lo, f_hi, rates, inject = args[:5]
    shuffles = args[5] if len(args) > 5 else 0
    c = chirps_prepare(path)
    if c is None:
        return None
    tic, sector, tmag, t0, dt, x, have = c
    if inject:
        amp, f0, rate = inject
        tt = np.arange(len(x)) * dt
        tm = tt.mean()
        x = x + np.where(have, amp * np.sin(2 * np.pi * (f0 * (tt - tm) + 0.5 * rate * (tt - tm) ** 2)), 0.0)
    best = dict(tic=tic, sector=sector, tmag=tmag, logp=0.0, rate=0.0, f=None, windows=0, logp_still=0.0, f_still=None, logp_shuffled=[], t_ref=None, window_days=None)
    rng = np.random.default_rng(tic % 100000)
    shuffled = np.zeros(shuffles)
    # each pass's windows are short enough that its fastest drift crosses no more than one frequency bin within a window
    for window_days, rs in rates:
        P, centres, freq, df = chirps_spectrogram(x, have, dt, f_lo, f_hi, window_days)
        if len(P) < 6:
            continue
        (lp, rate, j), _ = chirps_search(P, centres, df, rs)
        if j is not None and lp < best["logp"]:
            best.update(logp=lp, rate=rate, f=float(freq[j]), windows=len(P), t_ref=float(t0 + centres.mean()), window_days=window_days)
        # the windows' order shuffled keeps every burst of power and breaks every straight track
        for k in range(shuffles):
            shuffled[k] = min(shuffled[k], chirps_search(P[rng.permutation(len(P))], centres, df, rs)[0][0])
        if window_days == rates[0][0]:
            (lp0, _, j0), _ = chirps_search(P, centres, df, [0.0])
            best.update(logp_still=lp0, f_still=float(freq[j0]) if j0 is not None else None)
    best["logp_shuffled"] = shuffled.tolist()
    return best if best["windows"] else None


def chirps_track(args):
    """The chance of one light curve's power along a given track in absolute time (frequency f at time t_ref, drifting by rate), with the window length the track was found in."""
    path, f_lo, f_hi, f, rate, t_ref, window_days = args
    c = chirps_prepare(path)
    if c is None:
        return None
    tic, sector, tmag, t0, dt, x, have = c
    P, centres, freq, df = chirps_spectrogram(x, have, dt, f_lo, f_hi, window_days)
    if len(P) < 6:
        return None
    b = np.round((f + rate * (t0 + centres - t_ref) - freq[0]) / df).astype(int)
    ok = (b >= 0) & (b < P.shape[1])
    if ok.sum() < 6:
        return None
    acc = float(P[np.flatnonzero(ok), b[ok]].sum())
    return tic, float(chirps_logp(np.array([acc]), np.array([ok.sum()]))[0])


def chirps_centroid(args):
    """The chance of a light curve's flux and of each axis of its flux-weighted centroid along one track in absolute time, the flux's first."""
    from astropy.io import fits
    from scipy.ndimage import median_filter
    path, f_lo, f_hi, f, rate, t_ref, window_days = args
    with fits.open(path) as h:
        d = h[1].data
        t, fl, q = np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"])
        cad = np.asarray(d["CADENCENO"], int)
        cx, cy = np.asarray(d["MOM_CENTR1"], float), np.asarray(d["MOM_CENTR2"], float)
    ok = np.isfinite(t) & np.isfinite(fl) & np.isfinite(cx) & np.isfinite(cy) & (q == 0)
    t, fl, cad, cx, cy = t[ok], fl[ok], cad[ok], cx[ok], cy[ok]
    dt = float(np.median(np.diff(t)))
    out = []
    for v in (fl / np.median(fl) - 1, cx - np.median(cx), cy - np.median(cy)):
        v = v - median_filter(v, size=181, mode="nearest")
        n = int(cad.max() - cad.min() + 1)
        x, have = np.zeros(n), np.zeros(n, bool)
        x[cad - cad.min()] = v
        have[cad - cad.min()] = True
        P, centres, freq, df = chirps_spectrogram(x, have, dt, f_lo, f_hi, window_days)
        b = np.round((f + rate * (t[0] + centres - t_ref) - freq[0]) / df).astype(int)
        good = (b >= 0) & (b < P.shape[1])
        out.append(float(chirps_logp(np.array([float(P[np.flatnonzero(good), b[good]].sum())]), np.array([good.sum()]))[0]))
    return out


def chirps_prepare_key(path):
    """A twenty-second light curve file's (TIC, sector) from its header."""
    from astropy.io import fits
    H = fits.getheader(path)
    return int(H.get("TICID")), int(H.get("SECTOR"))


def chirps_main(record=False, rate_max=60.0, rate_min=1.0, p_max=1e-12, tmag_max=15.0, inject=0.0005, shuffles=100, procs=8):
    import glob
    from concurrent.futures import ProcessPoolExecutor
    from astropy.io import fits
    files = [f for f in sorted(glob.glob(core.data("tess20", "*.fits"))) if (fits.getheader(f).get("TESSMAG") or 99) <= tmag_max]
    f_lo, f_hi = 24.0, 1440.0
    # six-hour windows for drifts up to 16 cycles a day per day, an hour and a half beyond; the step keeps the track within half a bin over 25 days
    span = 25.0
    slow = [r for r in np.arange(-16.0, 16.0 + 1e-9, 4.0 / span / 2) if rate_min <= abs(r) <= min(16.0, rate_max)]
    fast = [r for r in np.arange(-64.0, 64.0 + 1e-9, 16.0 / span / 2) if 16.0 < abs(r) <= rate_max]
    rates = [(0.25, slow), (0.0625, fast)] if fast else [(0.25, slow)]
    with ProcessPoolExecutor(procs) as ex:
        res = [x for x in ex.map(chirps_one, [(f, f_lo, f_hi, rates, None) for f in files], chunksize=4) if x]
    lim = math.log(p_max)
    hits = [x for x in res if x["logp"] <= lim]
    if hits:
        # each light curve below the threshold read again with its windows shuffled
        byname = {chirps_prepare_key(f): f for f in files}
        with ProcessPoolExecutor(procs) as ex:
            again = list(ex.map(chirps_one, [(byname[(x["tic"], x["sector"])], f_lo, f_hi, rates, None, shuffles) for x in hits], chunksize=1))
        for x, y in zip(hits, again):
            x["logp_shuffled"] = y["logp_shuffled"] if y else []
    keys = {chirps_prepare_key(f): f for f in files} if hits else {}
    for x in hits:
        x["vet"] = "chance-rate" if x["logp_shuffled"] and min(x["logp_shuffled"]) <= x["logp"] else None
        x["shared"] = []
    # a track that other stars of the same sector carry at p 1e-4 belongs to the spacecraft: every other light curve of the sector read along it
    for x in [x for x in hits if x["vet"] is None]:
        others = [f for (tic, sec), f in keys.items() if sec == x["sector"] and tic != x["tic"]]
        with ProcessPoolExecutor(procs) as ex:
            got = [g for g in ex.map(chirps_track, [(f, f_lo, f_hi, x["f"], x["rate"], x["t_ref"], x["window_days"]) for f in others], chunksize=4) if g]
        x["shared"] = [tic for tic, lp in got if lp <= math.log(1e-4)]
        x["sector_read"] = len(got)
        if x["shared"]:
            x["vet"] = "detector-wide"
    rest = [x for x in hits if x["vet"] is None]
    for x in rest:
        path = keys[(x["tic"], x["sector"])]
        # pointing jitter moves the image: a track at least as strong in the centroid as in the flux is the spacecraft's
        fl, col, row = chirps_centroid((path, f_lo, f_hi, x["f"], x["rate"], x["t_ref"], x["window_days"]))
        x["centroid"] = (col, row)
        if min(col, row) <= fl:
            x["vet"] = "detector-wide"
            continue
        # a track that vanishes once the band starts 8 cycles a day higher rode the leakage of the star's own slower variability
        y = chirps_one((path, f_lo + 8.0, f_hi, rates, None, 20))
        x["above_floor"] = (y["logp"], min(y["logp_shuffled"])) if y else None
        if y and y["logp"] >= min(y["logp_shuffled"]):
            x["vet"] = "variable-star"
    # a track like a spacecraft track of any sector, within 15 per cent in frequency and a factor of two in drift of the same sign, joins its family
    craft = [y for y in hits if y["vet"] == "detector-wide"]
    for x in [x for x in hits if x["vet"] is None]:
        x["family"] = [(y["tic"], y["sector"]) for y in craft if abs(x["f"] / y["f"] - 1) < 0.15 and x["rate"] * y["rate"] > 0 and 0.5 < x["rate"] / y["rate"] < 2]
        if x["family"]:
            x["vet"] = "detector-wide"
    # the reach: chirps of 0.05 per cent at 300 cycles a day sliding 20 cycles a day per day laid into twenty light curves
    rng = np.random.default_rng(3)
    pick = [files[i] for i in rng.choice(len(files), size=min(20, len(files)), replace=False)]
    with ProcessPoolExecutor(procs) as ex:
        inj = [x for x in ex.map(chirps_one, [(f, f_lo, f_hi, rates, (inject, 300.0, 20.0)) for f in pick], chunksize=1) if x]
    found = sum(1 for x in inj if x["logp"] <= lim and abs(x["rate"] - 20.0) <= 1.0 and abs(x["f"] - 300.0) <= 16.0)
    lps = np.array([x["logp"] for x in res])
    print("%d twenty-second light curves of stars within 30 pc at Tmag %g or brighter; drifting tracks of %g-%g cycles a day per day between 1 and 60 minutes: median best log p %.1f, %d below p %g: %d matched by their own windows shuffled, %d the spacecraft's (shared in the sector, in the centroid or of a spacecraft family), %d riding a star's slower variability, %d left; a chirp of %.2f per cent at 300 c/d drifting 20 c/d per day is found in %d of %d light curves" % (
        len(res), tmag_max, rate_min, rate_max, float(np.median(lps)), len(hits), p_max, sum(x["vet"] == "chance-rate" for x in hits), sum(x["vet"] == "detector-wide" for x in hits),
        sum(x["vet"] == "variable-star" for x in hits), sum(x["vet"] is None for x in hits), 100 * inject, found, len(inj)))
    for x in sorted(res, key=lambda x: x["logp"])[:15]:
        print("  TIC %-10d s%02d Tmag %5.2f  log p %7.1f at %+6.1f c/d per day from %7.2f c/d; shuffled best %s; stationary log p %7.1f; %s" % (
            x["tic"], x["sector"], x["tmag"], x["logp"], x["rate"], x["f"] or 0, "%.1f" % min(x["logp_shuffled"]) if x.get("logp_shuffled") else "-", x["logp_still"],
            x.get("vet") or ("" if x["logp"] > lim else "LEFT")))
    json.dump(dict(n=len(res), hits=hits, injected=inj), open(core.results("tess20_chirps.json"), "w"), indent=1)
    if record:
        core.record_run(
            "tess20_chirps", "Chirps at the nearest stars", "TESS SPOC twenty-second PDCSAP light curves of the stars within 30 pc (the discrete-level search's set)",
            "%d light curves at Tmag %g or brighter" % (len(res), tmag_max), os.path.abspath(__file__), "chirps", len(res), "light curves",
            dict(rate_max=rate_max, rate_min=rate_min, p_max=p_max, tmag_max=tmag_max, inject=inject, shuffles=shuffles),
            "the Gamma tail of each track's summed power over its windows' medians; the median light curve's best track has log p %.1f" % float(np.median(lps)),
            "a track drifting %g-%g cycles a day per day at p %g; a chirp of %.2f per cent at 300 c/d drifting 20 c/d per day is found in %d of %d light curves" % (rate_min, rate_max, p_max, 100 * inject, found, len(inj)),
            "%d tracks below p %g: %d matched by their own windows shuffled, %d the spacecraft's, %d riding a star's slower variability, %d left" % (
                len(hits), p_max, sum(x["vet"] == "chance-rate" for x in hits), sum(x["vet"] == "detector-wide" for x in hits), sum(x["vet"] == "variable-star" for x in hits),
                sum(x["vet"] is None for x in hits)),
            [dict(target="tic:%d" % x["tic"], stat=round(x["logp"], 1), unit="natural log of the track's chance", vet=x["vet"],
                  note="sector %d, Tmag %.2f, %+.1f c/d per day from %.2f c/d over %d windows; shuffled best log p %s; centroid log p %s; band 8 c/d higher %s; shared with %s; family %s" % (
                      x["sector"], x["tmag"], x["rate"], x["f"] or 0, x["windows"], "%.1f" % min(x["logp_shuffled"]) if x.get("logp_shuffled") else "-",
                      "%.1f/%.1f" % x["centroid"] if x.get("centroid") else "-", "%.1f against %.1f shuffled" % x["above_floor"] if x.get("above_floor") else "-",
                      ", ".join("TIC %d" % t for t in x["shared"]) or "none", ", ".join("TIC %d s%d" % fs for fs in x.get("family", [])) or "none"))
             for x in hits])
        print("recorded")


APPROACHES = {
    "chirps": (lambda: core.run_cli(chirps_main), chirps_DOC),
    "tones": (lambda: core.run_cli(tones_main), tones_DOC),
    "flareecho": (lambda: core.run_cli(flareecho_main), flareecho_DOC),
    "twomessengers": (lambda: core.run_cli(twomessengers_main), twomessengers_DOC),
    "vascobelt": (lambda: core.run_cli(vascobelt_main), vascobelt_DOC),
    "ebtwins": (lambda: core.run_cli(ebtwins_main), ebtwins_DOC),
    "tesswholedays": (lambda: core.run_cli(tesswholedays_main), tesswholedays_DOC),
    "ebwholedays": (lambda: core.run_cli(ebwholedays_main), ebwholedays_DOC),
    "copyebs": (lambda: core.run_cli(copyebs_main), copyebs_DOC),
    "wdvary": (lambda: core.run_cli(wdvary_main), wdvary_DOC),
    "unclassified": (lambda: core.run_cli(unclassified_main), unclassified_DOC),
    "flicker": (lambda: core.run_cli(flicker_main), flicker_DOC),
    "symflares": (lambda: core.run_cli(symflares_main), symflares_DOC),
    "sndeficit": (lambda: core.run_cli(sndeficit_main), sndeficit_DOC),
    "flaresync": (lambda: core.run_cli(flaresync_main), flaresync_DOC),
    "flarecopies": (lambda: core.run_cli(flarecopies_main), flarecopies_DOC),
    "steady": (lambda: core.run_cli(steady_main), steady_DOC),
    "neliota": (neliota_cli, neliota_DOC),
    "superflare": (superflare_cli, superflare_DOC),
    "clocks": (lambda: core.run_cli(clocks_main), clocks_DOC),
    "nova": (nova_cli, nova_DOC),
    "etz": (etz_cli, etz_DOC),
    "levels": (levels_cli, levels_DOC),
    "keplevels": (keplevels_cli, keplevels_DOC),
    "determinism": (determinism_cli, determinism_DOC),
    "psfflash": (psfflash_cli, psfflash_DOC),
    "clockwork": (clockwork_cli, clockwork_DOC),
    "tophat": (tophat_cli, tophat_DOC),
    "tess20": (tess20_cli, tess20_DOC),
    "toi": (toi_cli, toi_DOC),
    "kepler": (kepler_cli, kepler_DOC),
    "asteroids": (asteroids_cli, asteroids_DOC),
}


