"""Laser lines and photon pulses in archival ultraviolet data of nearby stars."""
import argparse
import glob
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from setitools import core
from setitools.core import quiet_warnings


# --------------------------------------------------------------------------------------------------------------
# archive pointings carry catalogue positions of any epoch since 1950: match proper-motion tracks, nearest star within the aperture

def uv_nearby(plx):
    """SIMBAD stars above plx mas, planets excluded: main_id, ra, dec, pmra, pmdec, plx_value, sp_type, rvz_radvel, otype."""
    return core.simbad("SELECT main_id, ra, dec, pmra, pmdec, plx_value, sp_type, rvz_radvel, otype FROM basic "
                           "WHERE plx_value > %g AND otype NOT IN ('Pl', 'Pl?') AND ra IS NOT NULL" % plx)


def uv_track_dist(star, ra, dec, mjd):
    """Arcsec from pointings (deg, MJD) to a star's proper-motion track from 1950 to the observation or 2000, whichever is later."""
    ra0, de0 = float(star["ra"]), float(star["dec"])
    x = ((np.asarray(ra, float) - ra0 + 180.0) % 360.0 - 180.0) * math.cos(math.radians(de0)) * 3600.0
    y = (np.asarray(dec, float) - de0) * 3600.0
    mx, my = float(star.get("pmra") or 0) / 1000.0, float(star.get("pmdec") or 0) / 1000.0
    t1 = np.maximum((np.asarray(mjd, float) - 51544.5) / 365.25, 0.0)
    v2 = mx * mx + my * my
    t = np.clip((x * mx + y * my) / v2, -50.0, t1) if v2 > 0 else np.zeros_like(x)
    return np.hypot(x - mx * t, y - my * t)


def uv_owner(dists, tol):
    """Nearest star to a pointing from {name: track distance} within tol arcsec, or None."""
    if not dists:
        return None
    name = min(dists, key=dists.get)
    return name if dists[name] <= tol else None


def uv_groups(lams, tols):
    """Indices of ascending wavelengths grouped where neighbours lie within the lower one's tolerance."""
    if not len(lams):
        return []
    out, cur = [], [0]
    for i in range(1, len(lams)):
        if lams[i] - lams[i - 1] < tols[i - 1]:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    return out + [cur]


def uv_forced(items, lam_of, z_of, thr=5.0):
    """Stars whose forced measurement exceeds thr: lam_of(star) the wavelength in the star's frame, z_of(path, lam) the excess in noise units or None."""
    hit = set()
    for star, path in items:
        if star in hit:
            continue
        z = z_of(path, lam_of(star))
        if z is not None and z > thr:
            hit.add(star)
    return hit


def uv_resolve(names):
    """{target name: SIMBAD main_id} for names SIMBAD knows, a spaced component letter ('HD 48915 B') also tried joined."""
    var = {}
    for n in names:
        n = " ".join(str(n).split())
        for v in (n, re.sub(r"(\d) ([A-Ca-c])$", r"\1\2", n)):
            var.setdefault(v, n)
    out = {}
    keys = sorted(var)
    for k in range(0, len(keys), 50):
        q = ("SELECT i.id, b.main_id FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)"
             % ",".join("'%s'" % x.replace("'", "''") for x in keys[k:k + 50]))
        for r in core.simbad(q):
            out.setdefault(var.get(" ".join(str(r["id"]).split()), ""), " ".join(str(r["main_id"]).split()))
    out.pop("", None)
    return out


# --------------------------------------------------------------------------------------------------------------

starcat_DOC = r"""Laser lines in StarCAT HST/STIS echelle spectra (Ayres 2010): unresolved 6-sigma peaks persistent in two exposures of a setting, absent from other stars in either frame.

    python -I seti.py ultraviolet starcat fetch
    python -I seti.py ultraviolet starcat search"""

starcat_ROOT = os.environ.get("STARCAT_ROOT", core.data("starcat"))
starcat_HERE = core.RESULTS
starcat_BASE = "https://archive.stsci.edu/hlsps/starcat/fits/"
starcat_RES = {"E140M": 45800.0, "E230M": 30000.0, "E140H": 114000.0, "E230H": 114000.0}
starcat_AIRGLOW = (1215.67, 1302.17, 1304.86, 1306.03, 1355.60, 1358.51, 1199.55, 1200.22, 1200.71)
# chromospheric and transition-region lines whose self-reversed cores leave narrow peaks: C III, Si III, N V, C II, Si IV, C IV, He II, Mg II
starcat_CHROMOSPHERIC = (1175.71, 1206.50, 1238.82, 1242.80, 1334.53, 1335.71, 1393.76, 1402.77, 1548.20, 1550.77, 1640.4,
                 2795.53, 2796.35, 2802.70, 2803.53)
starcat_Z_MIN = 6.0
starcat_C_KMS = 299792.458
starcat_ENVELOPE_A = 1.0
starcat_ENVELOPE_PCT = 90


def starcat_get(u):
    return core.get(u, timeout=180, retries=3, soft=True, pause=2)


def starcat_fetch():
    os.makedirs(os.path.join(starcat_ROOT, "fits"), exist_ok=True)
    top = starcat_get(starcat_BASE).decode("utf-8", "replace")
    objs = [o for o in re.findall(r'href="([^"?/][^"]*)/"', top)]
    jobs = []

    def listing(o):
        t = starcat_get(starcat_BASE + o + "/")
        return [(o, f) for f in re.findall(r'href="(h_o[0-9a-z]+_spc\.fits)"', t.decode("utf-8", "replace"))] if t else []
    with ThreadPoolExecutor(6) as ex:
        for js in ex.map(listing, objs):
            jobs += js
    print("%d objects, %d single exposures" % (len(objs), len(jobs)), flush=True)

    def one(j):
        o, f = j
        d = os.path.join(starcat_ROOT, "fits", o)
        p = os.path.join(d, f)
        if os.path.exists(p):
            return 1
        b = starcat_get(starcat_BASE + o + "/" + f)
        if not b or not b.startswith(b"SIMPLE"):
            return 0
        os.makedirs(d, exist_ok=True)
        open(p + ".tmp", "wb").write(b)
        os.replace(p + ".tmp", p)
        return 1
    with ThreadPoolExecutor(6) as ex:
        n = sum(ex.map(one, jobs))
    print("%d exposures on disk" % n)


def starcat_load(path):
    """(wavelength, flux, error) of an exposure's good samples and its metadata; None for a setting outside RES."""
    from astropy.io import fits
    with fits.open(path) as h:
        hd = h[0].header
        d = h[1].data[0]
        w, f, e, dq = (np.asarray(d[k], float) for k in ("WAVE", "FLUX", "ERROR", "DQ"))
    mode = str(hd.get("OPT_ELEM", "")).strip()
    if mode not in starcat_RES:
        return None
    meta = dict(obj=str(hd.get("CATNAME", "")).strip(), mode=mode, cenwave=hd.get("CENWAVE"), mjd=hd.get("EXPSTART"),
                ra=hd.get("RA_TARG"), dec=hd.get("DEC_TARG"), file=os.path.basename(path), exptime=hd.get("EXPTIME"))
    good = (e > 0) & np.isfinite(f) & ((dq.astype(int) & (4 | 8 | 16 | 32 | 64 | 128 | 256)) == 0)
    return w[good], f[good], e[good], meta


def starcat_features(path):
    x = starcat_load(path)
    if x is None:
        return None
    w, f, e, meta = x
    if len(w) < 1000:
        return dict(meta, feats=[])
    return dict(meta, feats=starcat_detect(w, f, e, meta["mode"]))


def starcat_detect(w, f, e, mode):
    """Unresolved peaks: 6 sigma above the running upper envelope, 3 sigma above the brightest samples within 2.5 A, width 0.5-1.5 instrumental."""
    from scipy.optimize import curve_fit
    step = float(np.median(np.diff(w)))
    # upper envelope: a running median between dense absorption lines makes the windows read as peaks
    from scipy.ndimage import percentile_filter
    win = max(int(starcat_ENVELOPE_A / step) | 1, 31)
    cont = percentile_filter(f, starcat_ENVELOPE_PCT, size=win, mode="nearest")
    z = (f - cont) / e
    # and above the brightest neighbours within 2.5 A, which line-dense windows reach
    from scipy.ndimage import maximum_filter1d
    half = max(int(2.5 / step), 30)
    left = maximum_filter1d(np.concatenate([np.full(half, -np.inf), f[:-half]]), size=half, origin=(half - 1) // 2)
    right = maximum_filter1d(np.concatenate([f[half:], np.full(half, -np.inf)]), size=half, origin=-(half // 2))
    top = np.maximum(left, right)
    out = []
    cand = np.where((z >= starcat_Z_MIN) & (z >= np.roll(z, 1)) & (z >= np.roll(z, -1)))[0]
    for i in cand:
        if i < 8 or i > len(w) - 9:
            continue
        lam = w[i]
        if any(abs(lam - a) < 0.6 for a in starcat_AIRGLOW) or any(abs(lam - a) < 1.2 for a in starcat_CHROMOSPHERIC):
            continue
        if not f[i] - top[i] > 3 * e[i]:
            continue
        inst = lam / starcat_RES[mode]
        sl = slice(i - 8, i + 9)
        x, y, s = w[sl], (f - cont)[sl], e[sl]
        try:
            p, _ = curve_fit(lambda x, a, m, sg: a * np.exp(-0.5 * ((x - m) / sg) ** 2), x, y,
                             p0=[y[8], lam, inst / 2.355], sigma=s, maxfev=2000)
        except Exception:
            continue
        fwhm = 2.355 * abs(p[2])
        if 0.5 * inst <= fwhm <= 1.5 * inst and abs(p[1] - lam) < inst:
            out.append(dict(lam=float(p[1]), z=float(z[i]), fwhm=float(fwhm), inst=float(inst), flux=float(p[0])))
    return out


def starcat_resolve(objs):
    """SIMBAD main_id, parallax, spectral type and RV of StarCAT objects {name: (ra, dec)}: by identifier, else by position among stars."""
    def num(x):
        return float(x) if x not in (None, "") else None
    out = {}
    ident = {k: re.sub(r"^(HD|HR|HIP|GJ|GL|BD|CD|LHS|LTT|WD)(?=[0-9+-])", r"\1 ", k.replace("-", " "), flags=re.I)
             for k in objs}
    ids = sorted(set(ident.values()))
    for j in range(0, len(ids), 50):
        q = ("SELECT i.id, b.main_id, b.plx_value, b.sp_type, b.rvz_radvel FROM ident AS i JOIN basic AS b ON "
             "i.oidref = b.oid WHERE i.id IN (%s)" % ",".join("'%s'" % x.replace("'", "''") for x in ids[j:j + 50]))
        for r in core.simbad(q):
            key = " ".join(str(r["id"]).split()).lower()
            for k, v in ident.items():
                if key == " ".join(v.split()).lower():
                    out[k] = dict(main_id=" ".join(r["main_id"].split()), plx=num(r.get("plx_value")),
                                  sp=r.get("sp_type") or "", rv=num(r.get("rvz_radvel")))
    rest = {k: v for k, v in objs.items() if k not in out and v[0] is not None}
    m = core.xmatch([dict(source_id=k, ra=v[0], dec=v[1]) for k, v in rest.items()], "simbad", 10.0) if rest else []
    best = {}
    for r in m:
        if r.get("otype") in ("Pl", "Pl?") or "Planet" in (r.get("main_type") or ""):
            continue
        if r["source_id"] not in best or float(r["angDist"]) < float(best[r["source_id"]]["angDist"]):
            best[r["source_id"]] = r
    for k, r in best.items():
        out[k] = dict(main_id=" ".join(r["main_id"].split()), plx=num(r.get("plx")), sp=r.get("sp_type") or "",
                      rv=num(r.get("radvel")))
    return out


def starcat_forced_z(w, f, e, lam):
    """Largest excess over the upper envelope in noise units within two samples of lam; None without coverage."""
    from scipy.ndimage import percentile_filter
    j = int(np.searchsorted(w, lam))
    if j < 3 or j > len(w) - 4:
        return None
    step = float(np.median(np.diff(w[max(j - 50, 0):j + 50])))
    if abs(w[j] - lam) > 2 * step:
        return None
    lo, hi = max(j - 600, 0), min(j + 600, len(w))
    cont = percentile_filter(f[lo:hi], starcat_ENVELOPE_PCT, size=max(int(starcat_ENVELOPE_A / step) | 1, 31),
                             mode="nearest")
    k = j - lo
    return float(np.max((f[lo:hi][k - 2:k + 3] - cont[k - 2:k + 3]) / e[lo:hi][k - 2:k + 3]))


def starcat__forced(job):
    """Forced measurements of candidate features in one exposure: (object, [(z at rest wavelength, z at observed wavelength in the same setting)])."""
    quiet_warnings()
    path, rv, feats = job
    x = starcat_load(path)
    if x is None:
        return None
    w, f, e, meta = x
    out = []
    for p in feats:
        zr = starcat_forced_z(w, f, e, p["rest"] * (1 + rv / starcat_C_KMS))
        same = meta["mode"] == p["mode"] and meta["cenwave"] in p["cenwaves"]
        out.append((zr, starcat_forced_z(w, f, e, p["lam"]) if same else None))
    return meta["obj"], out


def starcat_search():
    from concurrent.futures import ProcessPoolExecutor
    paths = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(starcat_ROOT, "fits")) for f in fs if f.endswith("_spc.fits")]
    procs = int(os.environ.get("UV_PROCS", "2"))
    with ProcessPoolExecutor(procs) as ex:
        res = [r for r in ex.map(starcat_features, paths, chunksize=8) if r]
    print("%d exposures of %d objects; %d narrow peaks at %d sigma or more" % (
        len(res), len({r["obj"] for r in res}), sum(len(r["feats"]) for r in res), starcat_Z_MIN), flush=True)
    # persistent features per object and setting
    by = {}
    for r in res:
        by.setdefault((r["obj"], r["mode"]), []).append(r)
    pers = []
    for (obj, mode), rs in by.items():
        if len(rs) < 2:
            continue
        allf = sorted(((x, r["file"], r["cenwave"]) for r in rs for x in r["feats"]), key=lambda t: t[0]["lam"])
        for grp in uv_groups([x["lam"] for x, _, _ in allf], [x["inst"] / 3 for x, _, _ in allf]):
            files = {allf[j][1] for j in grp}
            if len(files) >= 2:
                xs = [allf[j][0] for j in grp]
                pers.append(dict(obj=obj, mode=mode, lam=float(np.mean([y["lam"] for y in xs])), n=len(files),
                                 of=len(rs), z=float(max(y["z"] for y in xs)), inst=xs[0]["inst"],
                                 cenwaves=sorted({allf[j][2] for j in grp}), ra=rs[0]["ra"], dec=rs[0]["dec"]))
    print("%d persistent features (two or more exposures in one setting) in %d objects" % (
        len(pers), len({p["obj"] for p in pers})), flush=True)
    names = starcat_resolve({r["obj"]: (r["ra"], r["dec"]) for r in res})
    rv = {k: v["rv"] or 0.0 for k, v in names.items()}
    for p in pers:
        p["rest"] = p["lam"] / (1 + rv.get(p["obj"], 0.0) / starcat_C_KMS)
        p["main_id"], p["sp"] = names.get(p["obj"], {}).get("main_id", p["obj"]), names.get(p["obj"], {}).get("sp", "")
    # other stars' peaks: rest frame (stellar), observed frame (instrument or sky)
    pk = [(x["lam"], r["obj"]) for r in res for x in r["feats"]]
    obsw = np.array([lam for lam, _ in pk])
    rest = np.array([lam / (1 + rv.get(o, 0.0) / starcat_C_KMS) for lam, o in pk])
    objs = np.array([o for _, o in pk])
    for p in pers:
        near_r = np.abs(rest - p["rest"]) / p["rest"] * starcat_C_KMS < 15.0
        near_o = np.abs(obsw - p["lam"]) / p["lam"] * starcat_C_KMS < 15.0
        p["stars_rest"] = len(set(objs[near_r]) - {p["obj"]})
        p["stars_obs"] = len(set(objs[near_o]) - {p["obj"]})
    left = [p for p in pers if p["stars_rest"] < 2 and p["stars_obs"] < 2]
    # forced measurements in other stars below detection: rest wavelength (stellar), observed wavelength in the same setting (instrument)
    obj_of = {r["file"]: r["obj"] for r in res}
    jobs = [(q, rv.get(obj_of.get(os.path.basename(q)), 0.0), left) for q in paths if os.path.basename(q) in obj_of]
    hits_r, hits_o = [set() for _ in left], [set() for _ in left]
    with ProcessPoolExecutor(procs) as ex:
        for out in ex.map(starcat__forced, jobs, chunksize=8):
            if not out:
                continue
            obj, zs = out
            for k, (zr, zo) in enumerate(zs):
                if obj != left[k]["obj"]:
                    if zr is not None and zr > 5.0:
                        hits_r[k].add(obj)
                    if zo is not None and zo > 5.0:
                        hits_o[k].add(obj)
    for k, p in enumerate(left):
        p["forced_rest"], p["forced_obs"] = sorted(hits_r[k]), sorted(hits_o[k])
    unique = sorted([p for p in left if len(p["forced_rest"]) < 2 and len(p["forced_obs"]) < 2], key=lambda p: -p["z"])
    print("%d persistent features in at most one other star in either frame; %d of them above 5 sigma in two or more "
          "other stars when measured there; %d left:" % (len(left), len(left) - len(unique), len(unique)))
    for p in unique[:60]:
        print("  %-18s %s %.3f A (rest %.3f), in %d of %d exposures, peak %.1f sigma, other stars %d/%d, measured %d/%d"
              % (p["obj"], p["mode"], p["lam"], p["rest"], p["n"], p["of"], p["z"], p["stars_rest"], p["stars_obs"],
                 len(p["forced_rest"]), len(p["forced_obs"])))
    json.dump(dict(exposures=len(res), objects=len({r["obj"] for r in res}), peaks=sum(len(r["feats"]) for r in res),
                   persistent=pers), open(os.path.join(starcat_HERE, "uv_laser.json"), "w"), indent=1)


def starcat_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "search"))
    a = p.parse_args()
    {"fetch": starcat_fetch, "search": starcat_search}[a.mode]()


def starcat_cli():
    quiet_warnings()
    starcat_main()


# --------------------------------------------------------------------------------------------------------------

cos_DOC = r"""Laser lines in HST/COS FUV spectra of stars within 10 pc: unresolved peaks at p 1e-9, persistent across FP-POS, absent from other stars and their summed spectra.

    python -I seti.py ultraviolet cos spectra      (STIS_ROOT event files; COS_PROCS processes)
    python -I seti.py ultraviolet cos search
    python -I seti.py ultraviolet cos identify     (NIST ASD lines, cached beside STIS_ROOT)"""

cos_ROOT = os.environ.get("STIS_ROOT", core.data("stis_tag"))
cos_HERE = core.RESULTS
cos_C_KMS = 299792.458
cos_BIN_PX = 2
cos_LSF_PX = 6.5
cos_AIRGLOW = ((1215.67, 2.0), (1302.17, 0.6), (1304.86, 0.6), (1306.03, 0.6), (1355.60, 0.6), (1358.51, 0.6),
           (1199.55, 0.6), (1200.22, 0.6), (1200.71, 0.6))
cos_STRONG = (1175.71, 1206.50, 1238.82, 1242.80, 1334.53, 1335.71, 1393.76, 1402.77, 1548.20, 1550.77, 1640.4)
cos_P_MAX = 1e-9
cos_DQ_BAD = 2 | 8 | 16 | 32 | 64 | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384
cos_APERTURE_TOL = 3.0  # arcsec; the primary science aperture is 2.5 arcsec across


def cos_owned(rs):
    """Spectra whose pointing holds a star within 10 pc, each given that star as 'star'."""
    inv = json.load(open(os.path.join(cos_ROOT, "inventory.json")))
    tags = {t["file"]: t for t in inv["tags"]}
    out = []
    for r in rs:
        tg = tags.get(r["file"])
        o = inv["obs"].get(tg["parent"], inv["obs"].get(tg["obsid"], {})) if tg else {}
        star = uv_owner(o.get("dist", {}), cos_APERTURE_TOL)
        if star:
            out.append(dict(r, star=star))
    return out


cos_CAL_TARGETS = ("DARK", "BIAS", "DEUTERIUM", "WAVE", "WAVEHITM", "WAVELINE", "FLAT", "CCDFLAT", "NONE")


def cos_left(p):
    """True for a persistent feature no two other stars show in either frame and absent from their summed spectra."""
    return p["stars_rest"] < 2 and p["stars_obs"] < 2 and p.get("template_p", 1.0) > 1e-5


def cos_science(hd):
    """False for calibration exposures (darks, flats, lamps) filed under a science visit."""
    et = str(hd.get("EXPTYPE", "EXTERNAL/SCI")).strip().upper()
    tn = str(hd.get("TARGNAME", "")).strip().upper()
    return et in ("EXTERNAL/SCI", "") and not any(tn.startswith(c) for c in cos_CAL_TARGETS)


def cos_spectrum(path):
    """Heliocentric count spectrum of one COS FUV corrtag segment: on-band and scaled background counts per two-pixel bin, with metadata."""
    from astropy.io import fits
    with fits.open(path) as h:
        hd = h[0].header
        if str(hd.get("DETECTOR", "")).strip() != "FUV":
            return None
        if not cos_science(hd):
            return None
        seg = str(hd.get("SEGMENT", "")).strip()[-1:]
        e = h["EVENTS"]
        eh, d = e.header, e.data
        if d is None or len(d) < 100:
            return None
        cols = d.columns.names
        t = np.asarray(d["TIME"], float)
        xf = np.asarray(d["XFULL"], float)
        yf = np.asarray(d["YFULL"], float)
        wl = np.asarray(d["WAVELENGTH"], float)
        good = (np.asarray(d["DQ"], int) & cos_DQ_BAD) == 0 if "DQ" in cols else np.ones(len(t), bool)
        if "PHA" in cols:
            pha = np.asarray(d["PHA"], int)
            good &= (pha >= int(eh.get("PHALOWR" + seg, 2))) & (pha <= int(eh.get("PHAUPPR" + seg, 23)))
        good &= wl > 0
        for i in range(1, 40):
            if "LMP_ON%d" % i in eh:
                on = float(eh["LMP_ON%d" % i])
                off = float(eh.get("LMPOFF%d" % i, on + 15.0))
                good &= ~((t >= on - 2.0) & (t <= off + 2.0))
        loc, hgt = eh.get("SP_LOC_" + seg), eh.get("SP_HGT_" + seg)
        if loc is None or hgt is None:
            return None
        slp = float(eh.get("SP_SLP_" + seg, 0.0) or 0.0)
        best = None
        for xr in (0.0, 8192.0):
            inb = np.abs(yf - (float(loc) + slp * (xf - xr))) <= float(hgt) / 2
            if best is None or (good & inb).sum() > (good & best).sum():
                best = inb
        on_band = good & best
        bk = np.zeros(len(t), bool)
        bh = 0.0
        for k in (1, 2):
            b, bhh = eh.get("B_BKG%d_%s" % (k, seg)), eh.get("B_HGT%d_%s" % (k, seg))
            if b is not None and bhh:
                bk |= np.abs(yf - float(b)) <= float(bhh) / 2
                bh += float(bhh)
        bk &= good
        vh = float(eh.get("V_HELIO", 0.0) or 0.0)
        meta = dict(file=os.path.basename(path), grating=str(hd.get("OPT_ELEM", "")).strip(), cenwave=hd.get("CENWAVE"),
                    fppos=hd.get("FPPOS"), segment=seg, target=str(hd.get("TARGNAME", "")).strip(),
                    mjd=float(eh.get("EXPSTART", 0) or 0), exptime=float(eh.get("EXPTIME", 0) or 0), v_helio=vh)
    if on_band.sum() < 200:
        return None
    edges = np.arange(0, 16384 + cos_BIN_PX, cos_BIN_PX)
    c_on = np.histogram(xf[on_band], edges)[0].astype(float)
    c_bk = np.histogram(xf[bk], edges)[0].astype(float) * (float(hgt) / bh if bh else 0.0)
    # dispersion: cubic wavelength against XFULL per segment from the band's events
    sel = on_band
    if sel.sum() > 50:
        cf = np.polyfit(xf[sel], wl[sel], 3)
    else:
        return None
    centres = 0.5 * (edges[:-1] + edges[1:])
    lam = np.polyval(cf, centres)
    lo, hi = np.percentile(xf[on_band], [0.5, 99.5])
    keep = (centres >= lo) & (centres <= hi)
    return dict(meta, lam=lam[keep].tolist(), on=c_on[keep].tolist(), bk=c_bk[keep].tolist(),
                disp=float(abs(np.polyval(np.polyder(cf), 8192.0))))


def cos_spectra():
    from concurrent.futures import ProcessPoolExecutor
    inv = json.load(open(os.path.join(cos_ROOT, "inventory.json")))
    obs = inv["obs"]
    files = [(os.path.join(cos_ROOT, "events", tg["file"]), obs.get(tg["parent"], obs.get(tg["obsid"], {})).get("stars", []))
             for tg in inv["tags"] if tg["file"].endswith(("_corrtag_a.fits", "_corrtag_b.fits"))]
    files = [(p, s) for p, s in files if os.path.exists(p)]
    out = []
    with ProcessPoolExecutor(int(os.environ.get("COS_PROCS", "4"))) as ex:
        for (p, s), r in zip(files, ex.map(cos__safe_spectrum, [p for p, _ in files], chunksize=4)):
            if r:
                r["stars"] = [" ".join(x.split()) for x in s]
                out.append(r)
    json.dump(out, open(os.path.join(cos_ROOT, "cos_spectra.json"), "w"))
    print("%d COS FUV segment spectra of %d targets" % (len(out), len({r["target"] for r in out})))


def cos__safe_spectrum(p):
    try:
        return cos_spectrum(p)
    except Exception as e:
        print(os.path.basename(p), "failed:", str(e)[:80], flush=True)
        return None


def cos_peaks(r, sign, p_max=cos_P_MAX):
    """Narrow peaks of one spectrum: (heliocentric wavelength, significance, width in pixels, net counts)."""
    from scipy.ndimage import percentile_filter
    from scipy.optimize import curve_fit
    from scipy.stats import poisson
    lam = np.array(r["lam"]) * (1 + sign * r["v_helio"] / cos_C_KMS)
    on, bk = np.array(r["on"]), np.array(r["bk"])
    if len(lam) < 200:
        return []
    s3 = np.convolve(on, np.ones(3), "same")
    b3 = np.convolve(bk, np.ones(3), "same")
    step = abs(float(np.median(np.diff(lam))))
    win = max(int(1.0 / step) * 2 + 1, 31)
    env = percentile_filter(s3, 90, size=win, mode="nearest")
    mu = np.maximum(env, b3 + 0.5)
    p = poisson.sf(s3 - 1, mu)
    cand = np.where((p < p_max) & (s3 >= np.roll(s3, 1)) & (s3 >= np.roll(s3, -1)))[0]
    out = []
    net = on - bk
    for i in cand:
        if i < 8 or i > len(lam) - 9:
            continue
        l0 = lam[i]
        if any(abs(l0 - a) < w for a, w in cos_AIRGLOW) or any(abs(l0 - a) < 1.2 for a in cos_STRONG):
            continue
        x = np.arange(-8, 9) * cos_BIN_PX
        y = net[i - 8:i + 9] - np.median(net[max(i - 40, 0):i + 40])
        try:
            q, _ = curve_fit(lambda x, a, m, sg: a * np.exp(-0.5 * ((x - m) / sg) ** 2), x, y,
                             p0=[max(y[8], 1.0), 0.0, cos_LSF_PX / 2.355], maxfev=2000)
        except Exception:
            continue
        fw = 2.355 * abs(q[2])
        if 0.5 * cos_LSF_PX <= fw <= 1.5 * cos_LSF_PX and abs(q[1]) < cos_LSF_PX:
            out.append(dict(lam=float(np.interp(i + q[1] / cos_BIN_PX, np.arange(len(lam)), lam)),
                            p=float(max(p[i], 1e-300)), fwhm_px=float(fw), net=float(q[0])))
    return out


def cos_search():
    rs = cos_owned(json.load(open(os.path.join(cos_ROOT, "cos_spectra.json"))))
    # CalCOS HELCORR already makes WAVELENGTH heliocentric
    best_sign = 0
    feats, loose = [], []
    for r in rs:
        for f in cos_peaks(r, best_sign, 1e-5):
            g = dict(f, file=r["file"], grating=r["grating"], fppos=r["fppos"], target=r["target"], disp=r["disp"],
                     star=r["star"])
            loose.append(g)
            if f["p"] < cos_P_MAX:
                feats.append(g)
    print("%d spectra of %d stars; %d narrow peaks at p < %.0e" % (len(rs), len({r["star"] for r in rs}), len(feats),
                                                                   cos_P_MAX), flush=True)
    by = {}
    for f in feats:
        by.setdefault((f["star"], f["grating"]), []).append(f)
    pers = []
    for (star, grating), fs in by.items():
        fs.sort(key=lambda f: f["lam"])
        used = set()
        for k, f in enumerate(fs):
            if k in used:
                continue
            tol = cos_BIN_PX * f["disp"] * 3 / 3.0
            grp = [j for j, g in enumerate(fs) if abs(g["lam"] - f["lam"]) < tol]
            files = {fs[j]["file"][:9] for j in grp}
            # persistence across FP-POS: a fixed wavelength moves on the detector
            if len(files) >= 2 and len({fs[j]["fppos"] for j in grp}) >= 2:
                used.update(grp)
                pers.append(dict(star=star, grating=grating, lam=float(np.mean([fs[j]["lam"] for j in grp])),
                                 n=len(files), p=min(fs[j]["p"] for j in grp),
                                 fppos=sorted({fs[j]["fppos"] for j in grp}), disp=f["disp"]))
    print("%d persistent features (two or more exposures) in %d stars" % (len(pers), len({p["star"] for p in pers})),
          flush=True)
    names = sorted({p["star"] for p in pers} | {g["star"] for g in loose})
    rv = {}
    for n in names:
        try:
            q = core.simbad("SELECT b.rvz_radvel FROM basic b JOIN ident i ON i.oidref = b.oid WHERE i.id = '%s'"
                                % n.replace("'", "''"))
            rv[n] = float(q[0]["rvz_radvel"]) if q and q[0]["rvz_radvel"] not in ("", None) else 0.0
        except Exception:
            rv[n] = 0.0
    for p in pers:
        p["rest"] = p["lam"] / (1 + rv.get(p["star"], 0.0) / cos_C_KMS)
    # other stars' peaks at p < 1e-5 within 15 km/s: rest frame (stellar), observed frame (instrument or sky)
    rest = np.array([g["lam"] / (1 + rv.get(g["star"], 0.0) / cos_C_KMS) for g in loose])
    obsw = np.array([g["lam"] for g in loose])
    star = np.array([g["star"] for g in loose])
    for p in pers:
        near_r = np.abs(rest - p["rest"]) / p["rest"] * cos_C_KMS < 15.0
        near_o = np.abs(obsw - p["lam"]) / p["lam"] * cos_C_KMS < 15.0
        p["stars_rest"] = len(set(star[near_r]) - {p["star"]})
        p["stars_obs"] = len(set(star[near_o]) - {p["star"]})
    # deep test: other stars summed in their rest frames, a peak at Poisson p < 1e-5 per resolution element over the running 90th percentile
    from scipy.stats import poisson
    grid = np.arange(1100.0, 1800.0, 0.01)
    stack = {}
    for r in rs:
        st = r["star"]
        lam = np.array(r["lam"]) * (1 + best_sign * r["v_helio"] / cos_C_KMS) / (1 + rv.get(st, 0.0) / cos_C_KMS)
        on = np.array(r["on"])
        idx = np.clip(np.searchsorted(grid, lam), 0, len(grid) - 1)
        h = stack.setdefault((r["grating"], st), np.zeros(len(grid)))
        np.add.at(h, idx, on)
    for p in pers:
        tot = np.zeros(len(grid))
        for (g, st), h in stack.items():
            if g == p["grating"] and st != p["star"]:
                tot += h
        res_el = 3 * cos_BIN_PX * p["disp"]
        k = max(int(round(res_el / 0.01)), 1)
        s_k = np.convolve(tot, np.ones(k), "same")
        j = int(np.searchsorted(grid, p["rest"]))
        lo, hi = max(j - 100, 0), min(j + 101, len(grid))
        env = np.percentile(s_k[lo:hi], 90)
        i0 = int(np.argmax(s_k[max(j - k, 0):j + k + 1])) + max(j - k, 0)
        p["template_counts"] = float(s_k[i0])
        p["template_p"] = float(poisson.sf(s_k[i0] - 1, max(env, 0.5)))
    unique = sorted([p for p in pers if cos_left(p)], key=lambda p: p["p"])
    print("%d persistent features also in the other stars' summed spectra (template p < 1e-5)"
          % sum(1 for p in pers if p["template_p"] <= 1e-5))
    from collections import Counter
    print("persistent features by star: " + ", ".join("%s %d" % kv for kv in Counter(p["star"] for p in pers).most_common()))
    print("%d persistent features in at most one other star in either frame and absent from the summed spectra:"
          % len(unique))
    for p in unique[:80]:
        print("  %-22s %s %.3f A (rest %.3f), in %d exposures (FP-POS %s), p %.1e, other stars %d/%d, other stars' "
              "sum %.0f counts (p %.1e)" % (p["star"][:22], p["grating"], p["lam"], p["rest"], p["n"],
                                           ",".join(str(x) for x in p["fppos"]), p["p"], p["stars_rest"],
                                           p["stars_obs"], p["template_counts"], p["template_p"]))
    json.dump(dict(sign=best_sign, spectra=len(rs), stars=len({r["star"] for r in rs}),
                   peaks=len(feats), persistent=pers), open(os.path.join(cos_HERE, "cos_laser.json"), "w"), indent=1)


cos_IONS = ("H I", "He II", "C I", "C II", "C III", "C IV", "N I", "N II", "N III", "N IV", "N V", "O I", "O II", "O III",
        "O IV", "O V", "Si I", "Si II", "Si III", "Si IV", "S I", "S II", "S III", "S IV", "Fe I", "Fe II", "Fe III",
        "Fe XII", "Fe XXI", "Mg I", "Mg II", "Al I", "Al II", "Al III", "Ca II", "Cl I", "Cl II", "P I", "P II", "Ni II",
        "Cr II", "Mn II", "Co II", "Zn II", "Ti II", "Ne V")


def cos_identify():
    """Persistent features within 15 km/s of NIST chromospheric and transition-region lines (intensity listed or A above 1e7 /s, low levels for neutral and singly ionised)."""
    lines = []
    for ion in cos_IONS:
        lines += core.nist_lines(ion, 1100, 1800, cache_dir=os.path.join(os.path.dirname(cos_ROOT), "nist"))

    def num(x):
        try:
            return float(str(x).strip("[]()+x? "))
        except ValueError:
            return float("nan")
    strong = []
    for ln in lines:
        stage = ln["ion"].split()[1]
        ei, aki = num(ln["ei"]), num(ln["aki"])
        if (ln["intens"].strip() or aki > 1e7) and (stage not in ("I", "II") or ei < 30000):
            strong.append(ln)
    strong.sort(key=lambda ln: ln["lam"])
    lam = np.array([ln["lam"] for ln in strong])
    d = json.load(open(os.path.join(cos_HERE, "cos_laser.json")))
    left = [p for p in d["persistent"] if cos_left(p)]
    rng = np.random.default_rng(1)
    r = rng.uniform(1150, 1750, 20000)
    k = np.searchsorted(lam, r).clip(1, len(lam) - 1)
    dv = np.minimum(np.abs(lam[k] - r), np.abs(lam[k - 1] - r)) / r * cos_C_KMS
    print("%d NIST lines, %d strong or low-excitation; a random wavelength lies within 2 and 15 km/s of one with "
          "probability %.2f and %.2f" % (len(lines), len(strong), np.mean(dv < 2), np.mean(dv < 15)))
    unid = []
    for p in sorted(left, key=lambda p: p["p"]):
        dv = (lam - p["rest"]) / p["rest"] * cos_C_KMS
        j = int(np.argmin(np.abs(dv)))
        p["id"] = "%s %.3f (%+.0f km/s)" % (strong[j]["ion"], strong[j]["lam"], dv[j]) if abs(dv[j]) < 15 else None
        if not p["id"]:
            unid.append(p)
        print("  %-14s %s rest %.3f, %d exposures, p %.0e: %s" % (p["star"][:14], p["grating"], p["rest"], p["n"], p["p"],
                                                                 p["id"] or "no line within 15 km/s"))
    print("%d of %d persistent features unique to one star have no line within 15 km/s" % (len(unid), len(left)))
    json.dump(dict(d, identified=left), open(os.path.join(cos_HERE, "cos_laser.json"), "w"), indent=1)


def cos_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("spectra", "search", "identify"))
    a = p.parse_args()
    {"spectra": cos_spectra, "search": cos_search, "identify": cos_identify}[a.mode]()


def cos_cli():
    quiet_warnings()
    cos_main()


# --------------------------------------------------------------------------------------------------------------

iue_DOC = r"""Laser lines in IUE high-dispersion spectra of stars within 10 pc: unresolved 6-sigma peaks persistent in two images of one camera and aperture, absent from other stars and the other aperture.

    python -I seti.py ultraviolet iue inventory
    python -I seti.py ultraviolet iue fetch [--threads 4]
    python -I seti.py ultraviolet iue search"""

iue_ROOT = os.environ.get("IUE_ROOT", core.data("iue"))
iue_HERE = core.RESULTS
iue_DL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:IUE/url/missions/iue/data/%s/%05d/%s.mxhi.gz"
iue_R = 10000.0
iue_Z_MIN = float(os.environ.get("IUE_ZMIN", "6.0"))
iue_N_MIN = int(os.environ.get("IUE_NMIN", "2"))
iue_C_KMS = 299792.458
iue_MATCH_ARCSEC = 30.0
iue_APERTURE_TOL = 10.0  # arcsec; the large aperture is 10 by 20 arcsec


def iue_owned(inv):
    """One inventory entry per image: the nearest star within the aperture tolerance."""
    by = {}
    for o in inv:
        by.setdefault(o["obs_id"], []).append(o)
    out = []
    for v in by.values():
        # archived coordinates err by up to 30 arcsec: the target name's star first
        named = [o for o in v if o.get("named")]
        o = min(named or v, key=lambda o: o["sep"])
        if named or o["sep"] <= iue_APERTURE_TOL:
            out.append(o)
    return out


def iue_inventory():
    os.makedirs(iue_ROOT, exist_ok=True)
    stars = uv_nearby(100)
    every, page = [], 1
    while True:
        d = hst_pulses_invoke({"service": "Mast.Caom.Filtered", "format": "json", "pagesize": 50000, "page": page,
                                "params": {"columns": "obsid,obs_id,instrument_name,target_name,t_min,s_ra,s_dec",
                                           "filters": [{"paramName": "obs_collection", "values": ["IUE"]},
                                                       {"paramName": "filters", "values": ["HIGH DISP"]}]}})
        rows = d.get("data", [])
        every += rows
        if len(rows) < 50000:
            break
        page += 1
    print("%d IUE high-dispersion observations" % len(every), flush=True)
    ora = np.array([float(x["s_ra"]) if x.get("s_ra") is not None else np.nan for x in every])
    ode = np.array([float(x["s_dec"]) if x.get("s_dec") is not None else np.nan for x in every])
    tmin = np.array([float(x["t_min"]) if x.get("t_min") is not None else 46000.0 for x in every])
    out = []
    for s in stars:
        sep = uv_track_dist(s, ora, ode, tmin)
        for j in np.where(sep < iue_MATCH_ARCSEC)[0]:
            x = every[j]
            out.append(dict(star=" ".join(s["main_id"].split()), rv=s.get("rvz_radvel"), obs_id=x["obs_id"],
                            camera=x["instrument_name"], target=x["target_name"], mjd=float(tmin[j]), sep=float(sep[j])))
    names = uv_resolve({o["target"] for o in out})
    for o in out:
        o["named"] = names.get(" ".join(str(o["target"]).split())) == o["star"]
    json.dump(out, open(os.path.join(iue_ROOT, "inventory.json"), "w"), indent=1)
    print("%d high-dispersion images of %d objects within 10 pc" % (len(out), len({o["star"] for o in out})))


def iue_url(obs_id):
    cam = "".join(c for c in obs_id if c.isalpha()).lower()
    num = int("".join(c for c in obs_id if c.isdigit()))
    return iue_DL % (cam, num // 1000 * 1000, obs_id.lower())


def iue_fetch(threads):
    from concurrent.futures import ThreadPoolExecutor
    inv = json.load(open(os.path.join(iue_ROOT, "inventory.json")))
    os.makedirs(os.path.join(iue_ROOT, "mxhi"), exist_ok=True)

    def one(o):
        p = os.path.join(iue_ROOT, "mxhi", o + ".mxhi.gz")
        if os.path.exists(p):
            return 1
        b = core.get(iue_url(o), timeout=180, retries=3, soft=True, pause=2)
        if not b or b[:2] != b"\x1f\x8b" or len(b) < 5000:
            return 0
        open(p + ".tmp", "wb").write(b)
        os.replace(p + ".tmp", p)
        return 1
    ids = sorted({o["obs_id"] for o in inv})
    with ThreadPoolExecutor(threads) as ex:
        n = sum(ex.map(one, ids))
    print("%d of %d MXHI files on disk" % (n, len(ids)))


def iue_orders(path):
    """[(wavelength, net, noise, abs_cal)] of an image's orders at QUALITY 0; ABS_CAL converts flux numbers to erg/cm2/s/A."""
    from astropy.io import fits
    out = []
    with fits.open(path) as h:
        d = h[1].data
        names = [c.upper() for c in d.columns.names]
        for row in d:
            n = int(row["NPOINTS"])
            if n < 100:
                continue
            w = float(row["WAVELENGTH"]) + float(row["DELTAW"]) * np.arange(n)
            f = np.asarray(row["NET"][:n], float)
            e = np.asarray(row["NOISE"][:n], float)
            q = np.asarray(row["QUALITY"][:n], int)
            cal = np.asarray(row["ABS_CAL"][:n], float) if "ABS_CAL" in names else np.full(n, np.nan)
            ok = (q == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
            if ok.sum() < 80:
                continue
            out.append((w[ok], f[ok], e[ok], cal[ok]))
    return out


def iue_detect(w, f, e):
    """Unresolved peaks of one order: 6 sigma above a 5 A running upper envelope and 3 sigma above the brightest samples within 2.5 A."""
    from scipy.ndimage import maximum_filter1d, percentile_filter
    from scipy.optimize import curve_fit
    out = []
    step = float(np.median(np.diff(w)))
    env = percentile_filter(f, 90, size=max(int(5.0 / step) | 1, 61), mode="nearest")
    z = (f - env) / e
    half = max(int(2.5 / step), 10)
    left = maximum_filter1d(np.concatenate([np.full(half, -np.inf), f[:-half]]), size=half, origin=(half - 1) // 2)
    right = maximum_filter1d(np.concatenate([f[half:], np.full(half, -np.inf)]), size=half, origin=-(half // 2))
    top = np.maximum(left, right)
    for i in np.where((z >= iue_Z_MIN) & (z >= np.roll(z, 1)) & (z >= np.roll(z, -1)))[0]:
        if i < 6 or i > len(w) - 7 or not f[i] - top[i] > 3 * e[i]:
            continue
        lam = w[i]
        if any(abs(lam - a) < 0.6 for a in starcat_AIRGLOW) or any(abs(lam - a) < 1.2 for a in starcat_CHROMOSPHERIC):
            continue
        inst = lam / iue_R
        x, y = w[i - 6:i + 7], (f - env)[i - 6:i + 7]
        try:
            p, _ = curve_fit(lambda x, a, m, sg: a * np.exp(-0.5 * ((x - m) / sg) ** 2), x, y,
                             p0=[y[6], lam, inst / 2.355], sigma=e[i - 6:i + 7], maxfev=2000)
        except Exception:
            continue
        fw = 2.355 * abs(p[2])
        if 0.5 * inst <= fw <= 1.5 * inst and abs(p[1] - lam) < inst:
            out.append(dict(lam=float(p[1]), z=float(z[i]), inst=float(inst)))
    return out


def iue_features(path):
    return [x for w, f, e, _ in iue_orders(path) for x in iue_detect(w, f, e)]


def iue_aperture(path):
    """LARGE or SMALL: the aperture whose spectrum an MXHI file holds."""
    from astropy.io import fits
    with fits.open(path) as h:
        return str(h[0].header.get("APERTURE", "")).strip().upper() or "?"


def iue_forced_z(path, lam):
    """Largest excess over the upper envelope in noise units within two pixels of lam; None without coverage."""
    from scipy.ndimage import percentile_filter
    best = None
    for w, f, e, _ in iue_orders(path):
        if w[0] + 1 < lam < w[-1] - 1:
            j = int(np.searchsorted(w, lam))
            step = float(np.median(np.diff(w)))
            env = percentile_filter(f, 90, size=max(int(5.0 / step) | 1, 61), mode="nearest")
            v = float(np.max((f[j - 2:j + 3] - env[j - 2:j + 3]) / e[j - 2:j + 3]))
            best = v if best is None else max(best, v)
    return best


def iue__feat(job):
    quiet_warnings()
    p = os.path.join(iue_ROOT, "mxhi", job["obs_id"] + ".mxhi.gz")
    if not os.path.exists(p):
        return job, None
    try:
        return job, (iue_features(p), iue_aperture(p))
    except Exception:
        return job, None


def iue_search():
    from concurrent.futures import ProcessPoolExecutor
    inv = iue_owned(json.load(open(os.path.join(iue_ROOT, "inventory.json"))))
    feats, n_read, apert, rvs = [], 0, {}, {}
    with ProcessPoolExecutor(int(os.environ.get("IUE_PROCS", "4"))) as ex:
        for job, res in ex.map(iue__feat, inv, chunksize=8):
            if res is None:
                continue
            fs, ap = res
            n_read += 1
            apert[(job["star"], job["obs_id"])] = (job["camera"], ap)
            rv = float(job["rv"]) if job.get("rv") not in (None, "") else 0.0
            rvs[job["star"]] = rv
            for f in fs:
                feats.append(dict(f, star=job["star"], camera=job["camera"], aperture=ap, obs_id=job["obs_id"],
                                  rest=f["lam"] / (1 + rv / iue_C_KMS)))
    print("%d images of %d objects read; %d narrow peaks at %.0f sigma" % (n_read, len({j["star"] for j in inv}),
                                                                          len(feats), iue_Z_MIN), flush=True)
    # persistence within one camera and aperture, each with its own detector position
    by = {}
    for f in feats:
        by.setdefault((f["star"], f["camera"], f["aperture"]), []).append(f)
    pers = []
    for (star, cam, ap), fs in by.items():
        fs.sort(key=lambda f: f["lam"])
        for grp in uv_groups([f["lam"] for f in fs], [f["inst"] / 3 for f in fs]):
            imgs = {fs[j]["obs_id"] for j in grp}
            if len(imgs) >= iue_N_MIN:
                pers.append(dict(star=star, camera=cam, aperture=ap, lam=float(np.mean([fs[j]["lam"] for j in grp])),
                                 rest=float(np.mean([fs[j]["rest"] for j in grp])), n=len(imgs),
                                 of=sum(1 for (s, o), (c, a) in apert.items() if s == star and c == cam and a == ap),
                                 z=max(fs[j]["z"] for j in grp), inst=fs[grp[0]]["inst"]))
    print("%d persistent features in %d stars" % (len(pers), len({p["star"] for p in pers})), flush=True)
    rest = np.array([f["rest"] for f in feats])
    obsw = np.array([f["lam"] for f in feats])
    star = np.array([f["star"] for f in feats])
    for p in pers:
        near_r = np.abs(rest - p["rest"]) / p["rest"] * iue_C_KMS < 15.0
        near_o = np.abs(obsw - p["lam"]) / p["lam"] * iue_C_KMS < 15.0
        p["stars_rest"] = len(set(star[near_r]) - {p["star"]})
        p["stars_obs"] = len(set(star[near_o]) - {p["star"]})
    left = sorted([p for p in pers if p["stars_rest"] < 2 and p["stars_obs"] < 2], key=lambda p: -p["z"])
    # a detector-position feature is absent through the other aperture
    for p in left:
        other = [o for (s, o), (c, a) in apert.items() if s == p["star"] and c == p["camera"] and a != p["aperture"]]
        zs = [z for z in (iue_forced_z(os.path.join(iue_ROOT, "mxhi", o + ".mxhi.gz"), p["lam"]) for o in other)
              if z is not None]
        p["other_aperture_z"] = [round(z, 1) for z in zs]
        p["aperture_fixed"] = bool(zs) and max(zs) < 2.0
    # forced measurements in other stars below detection: rest wavelength, same camera (stellar); observed wavelength, same camera and aperture (detector)
    for p in left:
        imgs = [(s, os.path.join(iue_ROOT, "mxhi", o + ".mxhi.gz"), a) for (s, o), (c, a) in apert.items()
                if s != p["star"] and c == p["camera"]]
        p["forced_rest"] = sorted(uv_forced([(s, q) for s, q, _ in imgs],
                                            lambda s: p["rest"] * (1 + rvs.get(s, 0.0) / iue_C_KMS), iue_forced_z))
        p["forced_obs"] = sorted(uv_forced([(s, q) for s, q, a in imgs if a == p["aperture"]], lambda s: p["lam"],
                                           iue_forced_z))
    kept = [p for p in left if not p["aperture_fixed"] and len(p["forced_rest"]) < 2 and len(p["forced_obs"]) < 2]
    print("%d persistent features in at most one other star in either frame; %d of them absent from the same star's "
          "images through the other aperture; %d above 5 sigma in two or more other stars when measured there; "
          "%d left:" % (len(left), sum(p["aperture_fixed"] for p in left),
                        sum(1 for p in left if not p["aperture_fixed"] and p not in kept), len(kept)))
    for p in kept[:60]:
        print("  %-20s %s %s %.3f A (rest %.3f), %d of %d images, peak %.1f sigma, other stars %d/%d, other aperture %s"
              % (p["star"][:20], p["camera"], p["aperture"], p["lam"], p["rest"], p["n"], p["of"], p["z"],
                 p["stars_rest"], p["stars_obs"], p["other_aperture_z"] or "none"))
    json.dump(dict(images=n_read, peaks=len(feats), persistent=pers), open(os.path.join(iue_HERE, "iue_laser.json"), "w"),
              indent=1)


def iue_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("inventory", "fetch", "search"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    {"inventory": iue_inventory, "fetch": lambda: iue_fetch(a.threads), "search": iue_search}[a.mode]()


def iue_cli():
    quiet_warnings()
    iue_main()


# --------------------------------------------------------------------------------------------------------------

fuse_DOC = r"""Laser lines in FUSE spectra (905-1187 A) of stars within 20 pc: point-source peaks at p 1e-6 recurring in a second channel, away from stellar and airglow lines and other stars' peaks.

    python -I seti.py ultraviolet fuse inventory [--plx 50]
    python -I seti.py ultraviolet fuse fetch
    python -I seti.py ultraviolet fuse search"""

fuse_ROOT = os.environ.get("FUSE_ROOT", core.data("fuse"))
fuse_HERE = core.RESULTS
fuse_MAST = "https://mast.stsci.edu/api/v0/invoke"
fuse_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="


def fuse_invoke(req):
    data = urllib.parse.urlencode({"request": json.dumps(req)}).encode()
    for attempt in range(3):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(
                fuse_MAST, data=data, headers=dict(core.UA, **{"Content-Type": "application/x-www-form-urlencoded"})), timeout=240).read())
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {"data": []}


def fuse_inventory(plx):
    os.makedirs(fuse_ROOT, exist_ok=True)
    stars = uv_nearby(plx)
    every, page = [], 1
    while True:
        d = fuse_invoke({"service": "Mast.Caom.Filtered", "format": "json", "pagesize": 5000, "page": page,
                    "params": {"columns": "obsid,obs_id,t_exptime,target_name,s_ra,s_dec,t_min",
                               "filters": [{"paramName": "obs_collection", "values": ["FUSE"]}]}})
        rows = d.get("data", [])
        every += rows
        if len(rows) < 5000:
            break
        page += 1
    print("%d FUSE observations; %d SIMBAD objects with parallax above %g mas" % (len(every), len(stars), plx), flush=True)
    ora = np.array([float(x["s_ra"]) if x.get("s_ra") is not None else np.nan for x in every])
    ode = np.array([float(x["s_dec"]) if x.get("s_dec") is not None else np.nan for x in every])
    tmin = np.array([float(x["t_min"]) if x.get("t_min") is not None else 52500.0 for x in every])
    obs = {}
    for s in stars:
        dist = uv_track_dist(s, ora, ode, tmin)
        for j in np.where(dist < 108.0)[0]:
            x = every[j]
            o = obs.setdefault(str(x["obsid"]), dict(x, stars=[]))
            o["stars"].append(dict(main_id=s["main_id"], plx=s["plx_value"], sp=s["sp_type"], otype=s["otype"],
                                   rv=s["rvz_radvel"], dist=round(float(dist[j]), 2)))
    files = []
    ids = list(obs)
    for k in range(0, len(ids), 50):
        p = fuse_invoke({"service": "Mast.Caom.Products", "format": "json", "pagesize": 20000,
                    "params": {"obsid": ",".join(ids[k:k + 50])}})
        for x in p.get("data", []):
            fn = x["productFilename"]
            if fn.endswith("00000all4ttagfcal.fit.gz") or fn.endswith("00000all4histfcal.fit.gz"):
                files.append(dict(file=fn, uri=x["dataURI"], size=int(x.get("size") or 0), obsid=str(x.get("obsID"))))
    json.dump(dict(obs=obs, files=files), open(os.path.join(fuse_ROOT, "inventory.json"), "w"), indent=1)
    names = {s["main_id"] for o in obs.values() for s in o["stars"]}
    print("%d FUSE observations of %d objects within %.0f pc; %d combined files, %.0f MB"
          % (len(obs), len(names), 1000.0 / plx, len(files), sum(f["size"] for f in files) / 1e6))


def fuse_fetch():
    inv = json.load(open(os.path.join(fuse_ROOT, "inventory.json")))
    d = os.path.join(fuse_ROOT, "all")
    os.makedirs(d, exist_ok=True)

    def one(f):
        p = os.path.join(d, f["file"])
        if os.path.exists(p):
            return 1
        b = core.get(fuse_DOWNLOAD + f["uri"], timeout=300, retries=3, soft=True, pause=5)
        if not b:
            return 0
        open(p + ".tmp", "wb").write(b)
        os.replace(p + ".tmp", p)
        return 1
    with ThreadPoolExecutor(4) as ex:
        n = sum(ex.map(one, inv["files"]))
    print("%d of %d combined files on disk" % (n, len(inv["files"])))


fuse_C_KMS = 299792.458
fuse_LSF_SIG = 0.026  # A, point-source line spread function (FWHM about 0.06 A)
fuse_THR = 5.0
fuse_TOL_KMS = 20.0  # km/s; channel wavelength zero points differ by up to this
fuse_APERTURE_TOL = 15.0  # arcsec; the large square aperture is 30 arcsec across


def fuse_owner(o):
    """Inventory entry of the star in an observation's aperture, or None."""
    st = {" ".join(s["main_id"].split()): s for s in o.get("stars") or []}
    name = uv_owner({k: s["dist"] for k, s in st.items()}, fuse_APERTURE_TOL)
    return dict(st[name], main_id=name) if name else None


def fuse_on_target(path):
    """False for observations that put the star at the reference point, outside every aperture."""
    from astropy.io import fits
    return str(fits.getheader(path, 0).get("APERTURE", "")).strip().upper() != "RFPT"


def fuse_channels(path):
    """{channel: (wave, flux, error)} of a combined file, pixels without an error dropped."""
    from astropy.io import fits
    out = {}
    with fits.open(path) as h:
        for e in h[1:]:
            d = e.data
            if d is None or len(d) < 500:
                continue
            w, f, s = (np.asarray(d[c], float) for c in ("WAVE", "FLUX", "ERROR"))
            ok = np.isfinite(w) & np.isfinite(f) & np.isfinite(s) & (s > 0)
            if ok.sum() > 500:
                out[e.name.strip()] = (w[ok], f[ok], s[ok])
    return out


def fuse_matched(w, f, s):
    """Point-source line amplitude over its error at every pixel, above a 2 A running 90th percentile."""
    from scipy.ndimage import percentile_filter
    step = float(np.median(np.diff(w)))
    sp = fuse_LSF_SIG / step
    x = np.arange(-int(4 * sp), int(4 * sp) + 1)
    k = np.exp(-0.5 * (x / sp) ** 2)
    r = f - percentile_filter(f, 90, size=int(2.0 / step) | 1, mode="nearest")
    iv = 1.0 / s ** 2
    num = np.convolve(r * iv, k[::-1], mode="same")
    den = np.convolve(iv, (k ** 2)[::-1], mode="same")
    a = num / np.maximum(den, 1e-300)
    err = 1.0 / np.sqrt(np.maximum(den, 1e-300))
    return a / err, r, err


def fuse_line_limit(w, f, s):
    """Median THR-sigma flux of an unresolved line over the channel, erg cm-2 s-1."""
    _, _, err = fuse_matched(w, f, s)
    return float(np.median(fuse_THR * err) * np.sqrt(2 * np.pi) * fuse_LSF_SIG)


def fuse_channel_peaks(w, f, s, thr):
    """[(wavelength, z, fitted sigma A, amplitude)] of point-source peaks above thr."""
    from scipy.optimize import curve_fit
    z, r, _ = fuse_matched(w, f, s)
    out = []
    step = float(np.median(np.diff(w)))
    edge = int(1.0 / step)
    cand = np.nonzero(z > thr)[0]
    cand = cand[(cand > edge) & (cand < len(w) - edge)]
    used = set()
    for i in cand:
        if i in used:
            continue
        j = i + int(np.argmax(z[i:i + 12]))
        used.update(range(j - 12, j + 13))
        sl = slice(max(j - 20, 0), j + 21)
        try:
            p, _ = curve_fit(lambda t, a, m, sg, b: a * np.exp(-0.5 * ((t - m) / sg) ** 2) + b, w[sl], r[sl],
                             p0=(r[j], w[j], fuse_LSF_SIG, 0.0), sigma=s[sl], maxfev=3000)
        except Exception:
            continue
        a, m, sg, b = p
        sg = abs(sg)
        if a > 0 and abs(m - w[j]) < 0.05 and 0.5 * fuse_LSF_SIG <= sg <= 2.0 * fuse_LSF_SIG:
            out.append((float(m), float(z[j]), float(sg), float(a)))
    return out


def fuse_search():
    inv = json.load(open(os.path.join(fuse_ROOT, "inventory.json")))
    rv = {}
    found, limits, n_read = [], {}, 0
    for f in inv["files"]:
        p = os.path.join(fuse_ROOT, "all", f["file"])
        own = fuse_owner(inv["obs"].get(f["obsid"]) or {})
        if own is None or not os.path.exists(p) or not fuse_on_target(p):
            continue
        n_read += 1
        star = own["main_id"]
        rv[star] = float(own["rv"]) if own.get("rv") not in (None, "") else 0.0
        ch = fuse_channels(p)
        lim = [fuse_line_limit(*v) for v in ch.values()]
        if lim:
            plx = float(own.get("plx") or "nan")
            limits.setdefault(star, []).append(dict(file=f["file"], flux=float(np.median(lim)), plx=plx))
        pk = {c: fuse_channel_peaks(*v, thr=fuse_THR) for c, v in ch.items()}
        low = {c: fuse_channel_peaks(*v, thr=3.0) for c, v in ch.items()}
        for c, lst in pk.items():
            for lam, z, sg, a in lst:
                cover = [d for d, (w, _, _) in ch.items() if d != c and w[0] + 1 < lam < w[-1] - 1]
                hits = [d for d in cover if any(abs(l2 - lam) / lam * fuse_C_KMS < fuse_TOL_KMS for l2, _, _, _ in low[d])]
                found.append(dict(file=f["file"], star=star, rv=rv.get(star, 0.0), channel=c, lam=lam, z=z, sig=sg,
                                  amp=a, cover=cover, confirmed=hits))
    multi = [x for x in found if x["confirmed"]]
    for x in multi:
        x["rest"] = x["lam"] / (1 + x["rv"] / fuse_C_KMS)
    # stars with a confirmed peak at the same wavelength, stellar and heliocentric frames
    for x in multi:
        x["stars_rest"] = len({y["star"] for y in multi if y["star"] != x["star"]
                               and abs(y["rest"] - x["rest"]) / x["rest"] * fuse_C_KMS < fuse_TOL_KMS})
        x["stars_obs"] = len({y["star"] for y in multi if y["star"] != x["star"]
                              and abs(y["lam"] - x["lam"]) / x["lam"] * fuse_C_KMS < fuse_TOL_KMS})
    left = [x for x in multi if x["stars_rest"] < 2 and x["stars_obs"] < 2]
    lam, strong = fuse_nist_strong()
    rng = np.random.default_rng(1)
    r = rng.uniform(910, 1185, 20000)
    k = np.searchsorted(lam, r).clip(1, len(lam) - 1)
    chance = float(np.mean(np.minimum(np.abs(lam[k] - r), np.abs(lam[k - 1] - r)) / r * fuse_C_KMS < fuse_TOL_KMS))
    for x in left:
        dv = (lam - x["rest"]) / x["rest"] * fuse_C_KMS
        j = int(np.argmin(np.abs(dv)))
        x["id"] = "%s %.3f (%+.0f km/s)" % (strong[j]["ion"], strong[j]["lam"], dv[j]) if abs(dv[j]) < fuse_TOL_KMS else None
        dh = (lam - x["lam"]) / x["lam"] * fuse_C_KMS
        j = int(np.argmin(np.abs(dh)))
        if x["id"] is None and abs(dh[j]) < fuse_TOL_KMS and strong[j]["ion"] in fuse_AIRGLOW:
            x["id"] = "airglow %s %.3f (%+.0f km/s)" % (strong[j]["ion"], strong[j]["lam"], dh[j])
    best = {}
    for star, ls in limits.items():
        x = min(ls, key=lambda x: x["flux"])
        d_cm = 1000.0 / x["plx"] * 3.0857e18 if x["plx"] > 0 else float("nan")
        best[star] = dict(flux=x["flux"], plx=x["plx"], lum=4 * np.pi * d_cm ** 2 * x["flux"], file=x["file"])
    fl = np.array([b["flux"] for b in best.values()])
    lu = np.array([b["lum"] for b in best.values() if np.isfinite(b["lum"])])
    print("%d objects: median %g-sigma unresolved-line limit %.1e erg cm-2 s-1, isotropic %.1e erg/s (%.1e W)"
          % (len(best), fuse_THR, np.median(fl), np.median(lu), np.median(lu) * 1e-7))
    json.dump(dict(files=n_read, objects=len(limits), found=found, left=left, chance=chance, limits=best),
              open(os.path.join(fuse_HERE, "fuse_laser.json"), "w"), indent=1)
    print("%d observations with a star in the aperture; %d point-source peaks above %g; %d confirmed in a second "
          "channel; %d not shared with two or more other stars; a random wavelength lies within %.0f km/s of a strong "
          "NIST line with probability %.2f" % (n_read, len(found), fuse_THR, len(multi), len(left), fuse_TOL_KMS, chance))
    seen = set()
    for x in sorted(left, key=lambda x: -x["z"]):
        key = (x["star"], round(x["rest"], 1))
        if key in seen:
            continue
        seen.add(key)
        print("  %-22s %s %.3f A (rest %.3f), z %.1f, sigma %.3f A, confirmed in %s of %s, other stars %d/%d: %s"
              % (x["star"], x["channel"], x["lam"], x["rest"], x["z"], x["sig"], x["confirmed"], x["cover"],
                 x["stars_rest"], x["stars_obs"], x["id"] or "no line within %.0f km/s" % fuse_TOL_KMS))
    print("%d distinct features have no strong line within %.0f km/s"
          % (len({(x["star"], round(x["rest"], 1)) for x in left if not x["id"]}), fuse_TOL_KMS))


fuse_IONS = ("H I", "He I", "He II", "C I", "C II", "C III", "C IV", "N I", "N II", "N III", "N IV", "N V", "O I", "O II",
        "O III", "O IV", "O V", "O VI", "Ne V", "Ne VI", "Si II", "Si III", "Si IV", "P II", "P IV", "P V", "S II",
        "S III", "S IV", "S V", "S VI", "Ar I", "Ar II", "Fe II", "Fe III", "Fe XVIII", "Fe XIX")
fuse_AIRGLOW = ("H I", "O I", "N I", "N II", "He I")


def fuse_nist_strong():
    """NIST's strong FUV lines of chromospheric, transition-region, coronal and airglow ions (intensity listed or A above 1e7 /s, low levels for neutral and singly ionised)."""
    lines = []
    for ion in fuse_IONS:
        lines += core.nist_lines(ion, 900, 1190, cache_dir=os.path.join(os.path.dirname(fuse_ROOT), "nist"))

    def num(x):
        try:
            return float(str(x).strip("[]()+x? "))
        except ValueError:
            return float("nan")
    strong = sorted((ln for ln in lines if (ln["intens"].strip() or num(ln["aki"]) > 1e7)
                     and (ln["ion"].split()[1] not in ("I", "II") or num(ln["ei"]) < 30000)), key=lambda ln: ln["lam"])
    return np.array([ln["lam"] for ln in strong]), strong


def fuse_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("inventory", "fetch", "search"))
    p.add_argument("--plx", type=float, default=50.0)
    a = p.parse_args()
    if a.mode == "inventory":
        fuse_inventory(a.plx)
    elif a.mode == "fetch":
        fuse_fetch()
    else:
        fuse_search()


def fuse_cli():
    quiet_warnings()
    fuse_main()


# --------------------------------------------------------------------------------------------------------------

stis_DOC = r"""Laser lines in post-StarCAT HST/STIS echelle x1d spectra (2004 July on) of stars whose tracks lie within 3 arcsec, through StarCAT's detector and tests.

    python -I seti.py ultraviolet stis inventory
    python -I seti.py ultraviolet stis fetch [--threads 4]
    python -I seti.py ultraviolet stis search                     (UV_PROCS processes)
    python -I seti.py ultraviolet stis identify                   (the features left, against NIST's lines)"""

stis_ROOT = os.environ.get("STIS_X1D_ROOT", core.data("stis_x1d"))
stis_HERE = core.RESULTS
stis_T0 = 53200.0  # 2004 July, after StarCAT's last exposures
stis_TOL = 3.0  # arcsec; the echelle apertures are 0.2 arcsec across
stis_TOL_NAMED = 15.0  # arcsec from the track of the star the target name resolves to
stis_NOT_STARS = ("QSO", "G", "GiG", "GiC", "GiP", "BiC", "Sy1", "Sy2", "Sy*", "AGN", "Bla", "BLL", "LIN", "SBG", "rG",
                  "PN", "SNR", "HII", "Cl*", "GlC", "OpC", "ISM", "MoC", "Pl", "Pl?", "LSB", "EmG", "H2G", "bCG", "IG")


def stis_inventory():
    os.makedirs(stis_ROOT, exist_ok=True)
    every, page = [], 1
    while True:
        d = hst_pulses_invoke({"service": "Mast.Caom.Filtered", "format": "json", "pagesize": 50000, "page": page,
                               "params": {"columns": "obsid,obs_id,instrument_name,filters,target_name,s_ra,s_dec,t_min",
                                          "filters": [{"paramName": "obs_collection", "values": ["HST"]},
                                                      {"paramName": "instrument_name",
                                                       "values": ["STIS/FUV-MAMA", "STIS/NUV-MAMA"]},
                                                      {"paramName": "filters", "values": list(starcat_RES)},
                                                      {"paramName": "t_min", "values": [{"min": stis_T0, "max": 99999}]}]}})
        rows = d.get("data", [])
        every += rows
        if len(rows) < 50000:
            break
        page += 1
    def finite(x):
        try:
            return math.isfinite(float(x))
        except (TypeError, ValueError):
            return False
    every = [r for r in every if finite(r.get("s_ra")) and finite(r.get("s_dec"))]
    pts = {}
    for r in every:
        pts.setdefault((round(float(r["s_ra"]), 5), round(float(r["s_dec"]), 5)), []).append(r)
    rows = [dict(source_id="%d" % i, ra=k[0], dec=k[1]) for i, k in enumerate(pts)]
    keys = list(pts)
    m = core.xmatch(rows, "simbad", 30.0)
    cands = {}
    for x in m:
        if (x.get("otype") or "").strip() in stis_NOT_STARS:
            continue
        cands.setdefault(int(x["source_id"]), []).append(x)
    # a resolved target name credits its star within stis_TOL_NAMED, since binary orbits carry stars off their tracks
    tnames = sorted({str(r["target_name"]) for r in every if r.get("target_name")})
    ident = {t: re.sub(r"^(HD|HR|HIP|GJ|GL|BD|CD|LHS|LTT|WD)(?=[0-9+-])", r"\1 ", t.replace("-", " "), flags=re.I)
             for t in tnames}
    named = uv_resolve(sorted(set(ident.values())))
    target_star = {t: named.get(" ".join(v.split())) for t, v in ident.items()}
    stars = sorted({s for s in target_star.values() if s})
    info = {}
    for j in range(0, len(stars), 50):
        q = ("SELECT main_id, ra, dec, pmra, pmdec, plx_value AS plx, sp_type, rvz_radvel AS radvel, otype FROM basic "
             "WHERE main_id IN (%s)" % ",".join("'%s'" % s.replace("'", "''") for s in stars[j:j + 50]))
        for x in core.simbad(q):
            if (x.get("otype") or "").strip() not in stis_NOT_STARS and x.get("ra") is not None:
                info[" ".join(str(x["main_id"]).split())] = x
    obs = []
    for i, k in enumerate(keys):
        best = None
        for x in cands.get(i, []) + [info[s] for s in {target_star.get(str(r.get("target_name"))) for r in pts[k]} if s in info]:
            nm = " ".join(str(x["main_id"]).split()) in {target_star.get(str(r.get("target_name"))) for r in pts[k]}
            for r in pts[k]:
                dist = float(uv_track_dist(dict(ra=x["ra"], dec=x["dec"], pmra=x.get("pmra") or 0, pmdec=x.get("pmdec") or 0),
                                           [float(r["s_ra"])], [float(r["s_dec"])], [float(r["t_min"])])[0])
                ok = dist <= (stis_TOL_NAMED if nm else stis_TOL)
                if ok and (best is None or (not nm, dist) < (not best[2], best[0])):
                    best = (dist, x, nm)
        if best is None:
            continue
        x = best[1]
        for r in pts[k]:
            obs.append(dict(r, star=" ".join(str(x["main_id"]).split()), otype=x.get("otype"), sp=x.get("sp_type"),
                            rv=x.get("radvel"), plx=x.get("plx"), dist=round(best[0], 2), named=best[2]))
    ids = sorted({str(o["obsid"]) for o in obs})
    files = []
    for k in range(0, len(ids), 50):
        p = hst_pulses_invoke({"service": "Mast.Caom.Products", "format": "json", "pagesize": 20000,
                               "params": {"obsid": ",".join(ids[k:k + 50])}})
        for x in p.get("data", []):
            if x["productFilename"].endswith("_x1d.fits"):
                files.append(dict(file=x["productFilename"], uri=x["dataURI"], size=int(x.get("size") or 0),
                                  obsid=str(x.get("obsID")), parent=str(x.get("parent_obsid"))))
    files = list({f["file"]: f for f in files}.values())
    json.dump(dict(obs=obs, files=files), open(os.path.join(stis_ROOT, "inventory.json"), "w"), indent=1)
    print("%d STIS echelle observations since 2004 July; %d of %d stars; %d x1d files, %.1f GB"
          % (len(every), len(obs), len({o["star"] for o in obs}), len(files), sum(f["size"] for f in files) / 1e9))


def stis_fetch(threads):
    inv = json.load(open(os.path.join(stis_ROOT, "inventory.json")))
    d = os.path.join(stis_ROOT, "x1d")
    os.makedirs(d, exist_ok=True)

    def one(f):
        p = os.path.join(d, f["file"])
        if os.path.exists(p):
            return 1
        b = core.get(hst_pulses_DOWNLOAD + f["uri"], timeout=300, retries=3, soft=True, pause=5)
        if not b or not b.startswith(b"SIMPLE"):
            return 0
        open(p + ".tmp", "wb").write(b)
        os.replace(p + ".tmp", p)
        return 1
    with ThreadPoolExecutor(threads) as ex:
        n = sum(ex.map(one, inv["files"]))
    print("%d of %d x1d files on disk" % (n, len(inv["files"])))


def stis_orders(path):
    """[(wavelength, flux, error)] of each echelle order at good pixels, with the exposure's metadata."""
    from astropy.io import fits
    out = []
    with fits.open(path) as h:
        hd = h[0].header
        meta = dict(mode=str(hd.get("OPT_ELEM", "")).strip(), cenwave=hd.get("CENWAVE"), file=os.path.basename(path),
                    rootname=str(hd.get("ROOTNAME", "")).strip().lower(), target=str(hd.get("TARGNAME", "")).strip())
        for e in h[1:]:
            if e.name != "SCI" or e.data is None:
                continue
            meta["mjd"] = e.header.get("EXPSTART", meta.get("mjd"))
            for row in e.data:
                w, f, s, dq = (np.asarray(row[c], float) for c in ("WAVELENGTH", "FLUX", "ERROR", "DQ"))
                good = (s > 0) & np.isfinite(f) & ((dq.astype(int) & (4 | 8 | 16 | 32 | 64 | 128 | 256)) == 0)
                if good.sum() > 200:
                    out.append((w[good], f[good], s[good]))
    return out, meta


def stis__feat(job):
    quiet_warnings()
    path, star = job
    try:
        od, meta = stis_orders(path)
    except Exception:
        return None
    if meta["mode"] not in starcat_RES:
        return None
    return dict(meta, star=star, feats=[x for w, f, e in od for x in starcat_detect(w, f, e, meta["mode"])])


def stis_forced(job):
    """(star, [(z at each rest wavelength, z at each observed wavelength in the same grating and central wavelength)]) for one exposure."""
    quiet_warnings()
    path, star, rv, feats = job
    try:
        od, meta = stis_orders(path)
    except Exception:
        return None
    out = []
    for p in feats:
        zr = zo = None
        for w, f, e in od:
            a = starcat_forced_z(w, f, e, p["rest"] * (1 + rv / starcat_C_KMS))
            zr = a if zr is None or (a is not None and a > zr) else zr
            if meta["mode"] == p["mode"] and meta["cenwave"] in p["cenwaves"]:
                b = starcat_forced_z(w, f, e, p["lam"])
                zo = b if zo is None or (b is not None and b > zo) else zo
        out.append((zr, zo))
    return star, out


def stis_search():
    from concurrent.futures import ProcessPoolExecutor
    inv = json.load(open(os.path.join(stis_ROOT, "inventory.json")))
    star_of = {str(o["obsid"]): o for o in inv["obs"]}
    jobs = []
    for f in inv["files"]:
        o = star_of.get(f["parent"]) or star_of.get(f["obsid"])
        p = os.path.join(stis_ROOT, "x1d", f["file"])
        if o and os.path.exists(p):
            jobs.append((p, o["star"]))
    rv = {o["star"]: float(o["rv"]) if o.get("rv") not in (None, "") else 0.0 for o in inv["obs"]}
    procs = int(os.environ.get("UV_PROCS", "2"))
    with ProcessPoolExecutor(procs) as ex:
        res = [r for r in ex.map(stis__feat, jobs, chunksize=4) if r]
    print("%d exposures of %d stars; %d narrow peaks at %d sigma or more" % (
        len(res), len({r["star"] for r in res}), sum(len(r["feats"]) for r in res), starcat_Z_MIN), flush=True)
    by = {}
    for r in res:
        by.setdefault((r["star"], r["mode"]), []).append(r)
    pers = []
    for (star, mode), rs in by.items():
        if len({r["rootname"] for r in rs}) < 2:
            continue
        allf = sorted(((x, r["rootname"], r["cenwave"]) for r in rs for x in r["feats"]), key=lambda t: t[0]["lam"])
        for grp in uv_groups([x["lam"] for x, _, _ in allf], [x["inst"] / 3 for x, _, _ in allf]):
            files = {allf[j][1] for j in grp}
            if len(files) >= 2:
                xs = [allf[j][0] for j in grp]
                lam = float(np.mean([y["lam"] for y in xs]))
                pers.append(dict(star=star, mode=mode, lam=lam, rest=lam / (1 + rv.get(star, 0.0) / starcat_C_KMS),
                                 n=len(files), of=len({r["rootname"] for r in rs}), z=float(max(y["z"] for y in xs)),
                                 inst=xs[0]["inst"], cenwaves=sorted({allf[j][2] for j in grp})))
    print("%d persistent features (two or more exposures in one grating) in %d stars"
          % (len(pers), len({p["star"] for p in pers})), flush=True)
    pk = [(x["lam"], r["star"]) for r in res for x in r["feats"]]
    obsw = np.array([lam for lam, _ in pk])
    rest = np.array([lam / (1 + rv.get(s, 0.0) / starcat_C_KMS) for lam, s in pk])
    who = np.array([s for _, s in pk])
    for p in pers:
        p["stars_rest"] = len(set(who[np.abs(rest - p["rest"]) / p["rest"] * starcat_C_KMS < 15.0]) - {p["star"]})
        p["stars_obs"] = len(set(who[np.abs(obsw - p["lam"]) / p["lam"] * starcat_C_KMS < 15.0]) - {p["star"]})
    left = [p for p in pers if p["stars_rest"] < 2 and p["stars_obs"] < 2]
    hits_r, hits_o = [set() for _ in left], [set() for _ in left]
    with ProcessPoolExecutor(procs) as ex:
        for out in ex.map(stis_forced, [(p, s, rv.get(s, 0.0), left) for p, s in jobs], chunksize=4):
            if not out:
                continue
            s, zs = out
            for k, (zr, zo) in enumerate(zs):
                if s != left[k]["star"]:
                    if zr is not None and zr > 5.0:
                        hits_r[k].add(s)
                    if zo is not None and zo > 5.0:
                        hits_o[k].add(s)
    for k, p in enumerate(left):
        p["forced_rest"], p["forced_obs"] = sorted(hits_r[k]), sorted(hits_o[k])
    unique = sorted([p for p in left if len(p["forced_rest"]) < 2 and len(p["forced_obs"]) < 2], key=lambda p: -p["z"])
    print("%d persistent features in at most one other star in either frame; %d of them above 5 sigma in two or more "
          "other stars when measured there; %d left:" % (len(left), len(left) - len(unique), len(unique)))
    for p in unique[:60]:
        print("  %-24s %s %.3f A (rest %.3f), in %d of %d exposures, peak %.1f sigma, other stars %d/%d"
              % (p["star"][:24], p["mode"], p["lam"], p["rest"], p["n"], p["of"], p["z"], p["stars_rest"], p["stars_obs"]))
    json.dump(dict(exposures=len(res), objects=len({r["star"] for r in res}), peaks=sum(len(r["feats"]) for r in res),
                   persistent=pers), open(os.path.join(stis_HERE, "stis_laser.json"), "w"), indent=1)


def stis_identify():
    """Surviving STIS features against NIST lines by cos identify's rule, with the chance a random wavelength lies as close."""
    lines = []
    for ion in cos_IONS:
        lines += core.nist_lines(ion, 1140, 3150, cache_dir=os.path.join(os.path.dirname(stis_ROOT), "nist"))

    def num(x):
        try:
            return float(str(x).strip("[]()+x? "))
        except ValueError:
            return float("nan")
    strong = sorted((ln for ln in lines if (ln["intens"].strip() or num(ln["aki"]) > 1e7)
                     and (ln["ion"].split()[1] not in ("I", "II") or num(ln["ei"]) < 30000)), key=lambda ln: ln["lam"])
    lam = np.array([ln["lam"] for ln in strong])
    path = os.path.join(stis_HERE, "stis_laser.json")
    d = json.load(open(path))
    left = [p for p in d["persistent"] if "forced_rest" in p and len(p["forced_rest"]) < 2 and len(p["forced_obs"]) < 2]
    r = np.random.default_rng(1).uniform(1150, 3100, 20000)
    k = np.searchsorted(lam, r).clip(1, len(lam) - 1)
    near = np.minimum(np.abs(lam[k] - r), np.abs(lam[k - 1] - r)) / r * cos_C_KMS
    print("%d NIST lines, %d strong or low-excitation; a random wavelength lies within 15 km/s of one with probability "
          "%.2f (FUV %.2f, NUV %.2f)" % (len(lines), len(strong), np.mean(near < 15), np.mean(near[r < 1700] < 15),
                                         np.mean(near[r >= 1700] < 15)))
    unid = []
    for p in sorted(left, key=lambda p: -p["z"]):
        dv = (lam - p["rest"]) / p["rest"] * cos_C_KMS
        j = int(np.argmin(np.abs(dv)))
        p["id"] = "%s %.3f (%+.0f km/s)" % (strong[j]["ion"], strong[j]["lam"], dv[j]) if abs(dv[j]) < 15 else None
        if not p["id"]:
            unid.append(p)
    by = {}
    for p in left:
        by.setdefault(p["star"], [0, 0])[0 if p["id"] else 1] += 1
    print("%d features left in %d stars; %d without a line within 15 km/s" % (len(left), len(by), len(unid)))
    for s, (a, b) in sorted(by.items(), key=lambda kv: -sum(kv[1])):
        print("  %-24s identified %d, unidentified %d" % (s[:24], a, b))
    for p in unid[:40]:
        print("  unidentified %-20s %s rest %.3f, in %d of %d exposures, %.1f sigma"
              % (p["star"][:20], p["mode"], p["rest"], p["n"], p["of"], p["z"]))
    json.dump(dict(d, identified=left), open(path, "w"), indent=1)


def stis_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("inventory", "fetch", "search", "identify"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.mode == "inventory":
        stis_inventory()
    elif a.mode == "fetch":
        stis_fetch(a.threads)
    elif a.mode == "identify":
        stis_identify()
    else:
        stis_search()


def stis_cli():
    quiet_warnings()
    stis_main()


# --------------------------------------------------------------------------------------------------------------

limits_DOC = r"""Sensitivity of the ultraviolet laser searches: nominal unresolved-line thresholds per exposure and injection recovery, a star's limit m90 times its second-best exposure.

    python -I seti.py ultraviolet limits starcat|iue|fuse|stis [--n 300] [--procs 4]
    python -I seti.py ultraviolet limits cos [--n 300] [--procs 4]      (where the COS spectra are, STIS_ROOT)
    python -I seti.py ultraviolet limits table                          (per-star limits, distances and powers)
    python -I seti.py ultraviolet limits record                         (the four searches and their limits, coverage store)"""

limits_HERE = core.RESULTS
limits_MULT = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
limits_PER = 6  # lines injected per exposure and multiple, 6 instrumental widths apart
limits_ROOT2PI = math.sqrt(2 * math.pi)


def limits_m_at(frac, mult, rec):
    """Multiple recovered with probability frac, interpolated in the recovery curve."""
    rec = np.maximum.accumulate(np.asarray(rec, float))
    if rec[-1] < frac:
        return float("inf")
    return float(np.interp(frac, rec, mult)) if rec[0] < frac else float(mult[0])


def limits_masked(lam, masks, widths):
    return any(abs(lam - a) < w for a, w in zip(masks, widths))


# ------------------------------------------------------------------------------------------------------- StarCAT

def limits_starcat_one(args):
    path, seed = args
    x = starcat_load(path)
    if x is None or len(x[0]) < 1000:
        return None
    w, f, e, meta = x
    R = starcat_RES[meta["mode"]]
    sig = w / R / 2.3548
    nom = starcat_Z_MIN * e * limits_ROOT2PI * sig
    free = np.array([not limits_masked(lam, starcat_AIRGLOW + starcat_CHROMOSPHERIC,
                                [0.6] * len(starcat_AIRGLOW) + [1.2] * len(starcat_CHROMOSPHERIC)) for lam in w])
    out = dict(file=meta["file"], obj=meta["obj"], mode=meta["mode"], nominal=float(np.median(nom[free])),
               lam_med=float(np.median(w)), ra=meta["ra"], dec=meta["dec"])
    if seed is None:
        return out
    rng = np.random.default_rng(seed)
    lo, hi = w[0] + 10, w[-1] - 10
    rec = []
    for m in limits_MULT:
        inj = np.zeros_like(f)
        lams = []
        for _ in range(limits_PER * 20):
            l0 = rng.uniform(lo, hi)
            j = int(np.searchsorted(w, l0))
            if j <= 0 or j >= len(w) or not free[j] or any(abs(l0 - q) < 6 * l0 / R for q in lams):
                continue
            lams.append(l0)
            inj += m * nom[j] / (limits_ROOT2PI * sig[j]) * np.exp(-0.5 * ((w - l0) / sig[j]) ** 2)
            if len(lams) == limits_PER:
                break
        found = starcat_detect(w, f + inj, e, meta["mode"])
        rec.append(sum(1 for l0 in lams if any(abs(q["lam"] - l0) < 0.5 * l0 / R for q in found)) / max(len(lams), 1))
    out["rec"] = rec
    return out


def limits_stis_one(args):
    """Nominal threshold and recovery of one post-2004 STIS exposure, one line per echelle order and multiple."""
    path, seed, star = args
    try:
        od, meta = stis_orders(path)
    except Exception:
        return None
    if meta["mode"] not in starcat_RES or not od:
        return None
    R = starcat_RES[meta["mode"]]
    noms, lams = [], []
    rng = np.random.default_rng(seed) if seed is not None else None
    hits, tries = np.zeros(len(limits_MULT)), np.zeros(len(limits_MULT))
    masks = starcat_AIRGLOW + starcat_CHROMOSPHERIC
    widths = [0.6] * len(starcat_AIRGLOW) + [1.2] * len(starcat_CHROMOSPHERIC)
    for w, f, e in od:
        sig = w / R / 2.3548
        nom = starcat_Z_MIN * e * limits_ROOT2PI * sig
        free = np.array([not limits_masked(lam, masks, widths) for lam in w])
        noms += list(nom[free])
        lams += list(w[free])
        if rng is None or len(w) < 200:
            continue
        edge = min(3.0, 0.25 * (w[-1] - w[0]))  # an E140H order spans about 4 A
        if edge <= 0:
            continue
        for k, m in enumerate(limits_MULT):
            l0 = rng.uniform(w[0] + edge, w[-1] - edge)
            j = int(np.searchsorted(w, l0))
            if j <= 0 or j >= len(w) or not free[j]:
                continue
            inj = m * nom[j] / (limits_ROOT2PI * sig[j]) * np.exp(-0.5 * ((w - l0) / sig[j]) ** 2)
            found = starcat_detect(w, f + inj, e, meta["mode"])
            tries[k] += 1
            hits[k] += any(abs(q["lam"] - l0) < 0.5 * l0 / R for q in found)
    if not noms:
        return None
    out = dict(file=meta["file"], star=star, mode=meta["mode"], nominal=float(np.median(noms)),
               lam_med=float(np.median(lams)))
    if rng is not None:
        out["rec"] = (hits / np.maximum(tries, 1)).tolist()
    return out


# ------------------------------------------------------------------------------------------------------------ IUE

def limits_iue_one(args):
    path, seed = args
    try:
        od = iue_orders(path)
    except Exception:
        return None
    if not od:
        return None
    R = iue_R
    noms, lam_all = [], []
    rng = np.random.default_rng(seed) if seed is not None else None
    hits = np.zeros(len(limits_MULT))
    tries = np.zeros(len(limits_MULT))
    for w, f, e, cal in od:
        okc = np.isfinite(cal) & (np.abs(f) > 5 * e)
        if okc.sum() < 10:
            continue
        conv = np.interp(w, w[okc], (cal / f)[okc])  # erg/cm2/s/A per flux number
        sig = w / R / 2.3548
        nom = iue_Z_MIN * e * np.abs(conv) * limits_ROOT2PI * sig
        free = np.array([not limits_masked(lam, starcat_AIRGLOW + starcat_CHROMOSPHERIC,
                                    [0.6] * len(starcat_AIRGLOW) + [1.2] * len(starcat_CHROMOSPHERIC)) for lam in w])
        noms += list(nom[free])
        lam_all += list(w[free])
        if rng is None or len(w) < 200:
            continue
        for k, m in enumerate(limits_MULT):
            l0 = rng.uniform(w[0] + 2, w[-1] - 2)
            j = int(np.searchsorted(w, l0))
            if j <= 0 or j >= len(w) or not free[j]:
                continue
            amp = m * iue_Z_MIN * e[j]  # peak in flux numbers
            found = iue_detect(w, f + amp * np.exp(-0.5 * ((w - l0) / sig[j]) ** 2), e)
            tries[k] += 1
            hits[k] += any(abs(q["lam"] - l0) < 0.5 * l0 / R for q in found)
    if not noms:
        return None
    out = dict(file=os.path.basename(path), nominal=float(np.median(noms)), lam_med=float(np.median(lam_all)))
    if rng is not None:
        out["rec"] = (hits / np.maximum(tries, 1)).tolist()
        out["tries"] = tries.tolist()
    return out


# ----------------------------------------------------------------------------------------------------------- FUSE

def limits_fuse_threshold(ch, errs, l0):
    """Unresolved-line flux at l0 reaching THR in the most sensitive channel and 3 sigma in the next; None under two channels."""
    cover = [c for c, (w, _, _) in ch.items() if w[0] + 1 < l0 < w[-1] - 1]
    if len(cover) < 2:
        return None, cover
    e = sorted(float(errs[c][min(int(np.searchsorted(ch[c][0], l0)), len(ch[c][0]) - 1)]) for c in cover)
    return max(fuse_THR * e[0], 3.0 * e[1]) * limits_ROOT2PI * fuse_LSF_SIG, cover


def limits_fuse_one(args):
    path, seed = args
    if not fuse_on_target(path):
        return None
    ch = fuse_channels(path)
    if not ch:
        return None
    errs = {c: fuse_matched(*v)[2] for c, v in ch.items()}
    grid = np.arange(910.0, 1187.0, 0.5)
    noms = [x for x in (limits_fuse_threshold(ch, errs, lam)[0] for lam in grid) if x is not None and np.isfinite(x)]
    if not noms:
        return None
    out = dict(file=os.path.basename(path), nominal=float(np.median(noms)), lam_med=1040.0)
    if seed is None:
        return out
    rng = np.random.default_rng(seed)
    rec = []
    for m in limits_MULT:
        hit = n = 0
        for _ in range(limits_PER):
            c0 = list(ch)[rng.integers(len(ch))]
            w0 = ch[c0][0]
            l0 = rng.uniform(w0[0] + 2, w0[-1] - 2)
            F, cover = limits_fuse_threshold(ch, errs, l0)
            if F is None or not np.isfinite(F):
                continue
            F = m * F
            det5, det3 = [], []
            for c in cover:
                w, f, s = ch[c]
                g = F / (limits_ROOT2PI * fuse_LSF_SIG) * np.exp(-0.5 * ((w - l0) / fuse_LSF_SIG) ** 2)
                if any(abs(x[0] - l0) / l0 * fuse_C_KMS < fuse_TOL_KMS for x in fuse_channel_peaks(w, f + g, s, thr=fuse_THR)):
                    det5.append(c)
                if any(abs(x[0] - l0) / l0 * fuse_C_KMS < fuse_TOL_KMS for x in fuse_channel_peaks(w, f + g, s, thr=3.0)):
                    det3.append(c)
            n += 1
            hit += any(any(d != c for d in det3) for c in det5)
        rec.append(hit / max(n, 1))
    out["rec"] = rec
    return out


# ------------------------------------------------------------------------------------------------------------ COS

def limits_cos_one(args):
    r, seed, sens = args
    from scipy.stats import poisson
    lam = np.array(r["lam"])
    on, bk = np.array(r["on"]), np.array(r["bk"])
    if len(lam) < 200 or r["exptime"] <= 0:
        return None
    from scipy.ndimage import percentile_filter
    s3 = np.convolve(on, np.ones(3), "same")
    b3 = np.convolve(bk, np.ones(3), "same")
    step = abs(float(np.median(np.diff(lam))))
    env = percentile_filter(s3, 90, size=max(int(1.0 / step) * 2 + 1, 31), mode="nearest")
    mu = np.maximum(env, b3 + 0.5)
    # 3-bin counts for p < P_MAX, scaled by the 72 per cent of the line in 6 pixels
    thr = poisson.isf(cos_P_MAX, mu) + 1
    sig_px = cos_LSF_PX / 2.3548
    frac = math.erf(3.0 / (sig_px * math.sqrt(2)))
    nline = np.maximum(thr - mu, 1.0) / frac
    key = "%s/%s" % (r["grating"], r["cenwave"])
    s = sens.get(key)
    if s is None:
        return None
    S = np.interp(lam, s["lam"], s["sens"], left=np.nan, right=np.nan)  # counts/s/pixel per erg/cm2/s/A
    k = r["disp"] / (r["exptime"] * S)  # erg/cm2/s per count
    free = np.array([not (any(abs(x - a) < w for a, w in cos_AIRGLOW) or any(abs(x - a) < 1.2
                                                                                  for a in cos_STRONG))
                     for x in lam]) & np.isfinite(k)
    if free.sum() < 100:
        return None
    nom = nline * k
    out = dict(file=r["file"], star=r["star"], grating=r["grating"], fppos=r["fppos"], nominal=float(np.median(nom[free])),
               lam_med=float(np.median(lam[free])))
    if seed is None:
        return out
    rng = np.random.default_rng(seed)
    rec = []
    px = np.arange(len(lam)) * cos_BIN_PX
    for m in limits_MULT:
        hit = n = 0
        for _ in range(limits_PER):
            j = int(rng.integers(20, len(lam) - 20))
            if not free[j]:
                continue
            prof = np.exp(-0.5 * ((px - px[j]) / sig_px) ** 2)
            prof = prof / prof.sum()
            add = rng.poisson(m * nline[j] * prof)
            rr = dict(r, on=(on + add).tolist())
            found = cos_peaks(rr, 0, cos_P_MAX)
            n += 1
            hit += any(abs(q["lam"] - lam[j]) < 3 * r["disp"] * cos_BIN_PX for q in found)
        rec.append(hit / max(n, 1))
    out["rec"] = rec
    return out


def limits_cos_sensitivity():
    """NET/FLUX per pixel against wavelength per grating and central wavelength, from one x1d each."""
    import urllib.request
    from astropy.io import fits
    root = os.environ.get("STIS_ROOT", core.data("stis_tag"))
    rs = cos_owned(json.load(open(os.path.join(root, "cos_spectra.json"))))
    path = os.path.join(root, "cos_sens.json")
    if os.path.exists(path):
        return json.load(open(path)), rs
    pick = {}
    for r in rs:
        pick.setdefault("%s/%s" % (r["grating"], r["cenwave"]), r["file"][:9])
    os.makedirs(os.path.join(root, "x1d"), exist_ok=True)
    sens = {}
    for key, rootname in pick.items():
        p = os.path.join(root, "x1d", rootname + "_x1d.fits")
        if not os.path.exists(p):
            u = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/%s_x1d.fits" % rootname
            try:
                b = urllib.request.urlopen(urllib.request.Request(u, headers=core.UA),
                                           timeout=300).read()
            except Exception:
                continue
            if not b.startswith(b"SIMPLE"):
                continue
            open(p, "wb").write(b)
        lam, sv = [], []
        with fits.open(p) as h:
            for row in h[1].data:
                w, fl, rate = (np.asarray(row[c], float) for c in ("WAVELENGTH", "FLUX", "NET"))
                ok = (fl > 0) & (rate > 0) & np.isfinite(fl) & np.isfinite(rate)
                lam += list(w[ok])
                sv += list((rate / fl)[ok])
        if lam:
            o = np.argsort(lam)
            lam, sv = np.array(lam)[o], np.array(sv)[o]
            # 51-pixel running median of the smooth ratio
            from scipy.ndimage import median_filter
            sens[key] = dict(lam=lam[::10].tolist(), sens=median_filter(sv, 51)[::10].tolist())
    json.dump(sens, open(path, "w"))
    return sens, rs


# ---------------------------------------------------------------------------------------------------------- driver

def limits_run(which, n, procs):
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(7)
    if which == "starcat":
        paths = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(starcat_ROOT, "fits"))
                       for f in fs if f.endswith("_spc.fits"))
        fn, items = limits_starcat_one, paths
    elif which == "iue":
        items = sorted(os.path.join(iue_ROOT, "mxhi", f) for f in os.listdir(os.path.join(iue_ROOT, "mxhi"))
                       if f.endswith(".gz"))
        fn = limits_iue_one
    elif which == "fuse":
        items = sorted(os.path.join(fuse_ROOT, "all", f) for f in os.listdir(os.path.join(fuse_ROOT, "all")))
        fn = limits_fuse_one
    elif which == "stis":
        inv = json.load(open(os.path.join(stis_ROOT, "inventory.json")))
        star_of = {str(o["obsid"]): o["star"] for o in inv["obs"]}
        items = sorted((os.path.join(stis_ROOT, "x1d", f["file"]), star_of.get(f["parent"]) or star_of.get(f["obsid"]))
                       for f in inv["files"] if os.path.exists(os.path.join(stis_ROOT, "x1d", f["file"])))
        fn = limits_stis_one
    else:
        sens, rs = limits_cos_sensitivity()
        items = rs
        fn = limits_cos_one
    inj = set(rng.choice(len(items), size=min(n, len(items)), replace=False).tolist())
    args = [(it, (int(rng.integers(1 << 30)) if i in inj else None)) for i, it in enumerate(items)]
    if which == "cos":
        args = [(it, s, sens) for it, s in args]
    elif which == "stis":
        args = [(it[0], s, it[1]) for it, s in args]
    with ProcessPoolExecutor(procs) as ex:
        res = [r for r in ex.map(fn, args, chunksize=4) if r]
    recs = np.array([r["rec"] for r in res if "rec" in r])
    curve = recs.mean(axis=0).tolist() if len(recs) else []
    # plateau: findable fraction of the band; the limit is 90 per cent of it
    plateau = float(max(curve)) if curve else None
    out = dict(set=which, mult=list(limits_MULT), recovery=curve, n_injected=int(len(recs)), plateau=plateau,
               m50=limits_m_at(0.5 * plateau, limits_MULT, curve) if curve else None,
               m90=limits_m_at(0.9 * plateau, limits_MULT, curve) if curve else None, exposures=res)
    json.dump(out, open(os.path.join(limits_HERE, "uv_limits_%s.json" % which), "w"))
    print("%s: %d exposures, median nominal limit %.2e erg cm-2 s-1; recovery at %s times nominal: %s; plateau %.2f; "
          "half and 90 per cent of it at %.2f and %.2f times nominal"
          % (which, len(res), np.median([r["nominal"] for r in res]), "/".join("%g" % m for m in limits_MULT),
             " ".join("%.2f" % x for x in curve), plateau or float("nan"), out["m50"] or float("nan"),
             out["m90"] or float("nan")))


limits_PC_M = 3.0857e16
limits_D_TX = 10.0  # transmitter aperture, m


def limits_star_of_file():
    """{file: SIMBAD main_id} for IUE images and FUSE observations with a star in the aperture."""
    out = {}
    for o in iue_owned(json.load(open(os.path.join(iue_ROOT, "inventory.json")))):
        out[o["obs_id"].lower() + ".mxhi.gz"] = o["star"]
    fi = json.load(open(os.path.join(fuse_ROOT, "inventory.json")))
    for f in fi["files"]:
        own = fuse_owner(fi["obs"].get(f["obsid"]) or {})
        if own:
            out[f["file"]] = own["main_id"]
    return out


def limits_table():
    """Per-star limits: m90 times the second most sensitive exposure's nominal limit (best observation for FUSE), with distance, isotropic luminosity and 10 m diffraction-limited transmitter power."""
    sets = {w: json.load(open(os.path.join(limits_HERE, "uv_limits_%s.json" % w)))
            for w in ("starcat", "iue", "cos", "fuse", "stis")
            if os.path.exists(os.path.join(limits_HERE, "uv_limits_%s.json" % w))}
    fmap = limits_star_of_file()
    plx, sp = {}, {}
    if "starcat" in sets:
        objs = {e["obj"]: (float(e["ra"]), float(e["dec"])) for e in sets["starcat"]["exposures"] if e.get("ra") is not None}
        for k, v in starcat_resolve(objs).items():
            plx[k] = v["plx"]
            sp[k] = "%s; %s" % (v["main_id"], v["sp"])
    names = set()
    for w in ("iue", "cos", "fuse", "stis"):
        for e in sets.get(w, {}).get("exposures", []):
            n = e.get("star") or fmap.get(e["file"])
            if n:
                names.add(n)
    names = sorted(names)
    for k in range(0, len(names), 50):
        q = ("SELECT i.id, b.plx_value, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)"
             % ",".join("'%s'" % n.replace("'", "''") for n in names[k:k + 50]))
        for r in core.simbad(q):
            n = " ".join(str(r["id"]).split())
            plx[n] = float(r["plx_value"]) if r.get("plx_value") not in (None, "") else None
            sp[n] = r.get("sp_type") or ""
    rows = []
    for which, s in sets.items():
        m90 = s["m90"]
        by = {}
        for e in s["exposures"]:
            star = e.get("obj") if which == "starcat" else e.get("star") or fmap.get(e["file"])
            if not star:
                continue
            setting = e.get("mode") or e.get("grating") or e["file"][:3]
            by.setdefault((star, setting), []).append(e)
        best = {}
        for (star, setting), es in by.items():
            v = sorted(x["nominal"] for x in es)
            lim = v[0] if which == "fuse" else (v[1] if len(v) >= 2 else None)
            if lim is None:
                continue
            lam = float(np.median([x["lam_med"] for x in es]))
            if star not in best or lim < best[star][0]:
                best[star] = (lim, setting, lam, len(es))
        for star, (lim, setting, lam, n) in best.items():
            F = lim * m90  # erg cm-2 s-1
            p = plx.get(star)
            d = 1000.0 / p * limits_PC_M if p and p > 0 else None
            L = 4 * math.pi * d ** 2 * F * 1e-3 if d else None  # W (erg cm-2 s-1 = 1e-3 W m-2)
            P = F * 1e-3 * d ** 2 * (lam * 1e-10 / limits_D_TX) ** 2 if d else None
            rows.append(dict(set=which, star=star, setting=setting, n=n, lam=lam, flux=F, plx=p,
                             sp=sp.get(star, ""), L_iso_W=L, P_10m_W=P))
    json.dump(dict(m90={w: s["m90"] for w, s in sets.items()}, rows=rows),
               open(os.path.join(limits_HERE, "uv_limits_table.json"), "w"), indent=1)
    for which in sets:
        rs = [r for r in rows if r["set"] == which]
        near = [r for r in rs if r["plx"] and r["plx"] > 100]
        print("%s: m90 %.2f; %d stars with two exposures, median %.1e erg cm-2 s-1; %d within 10 pc, median %.1e W "
              "isotropic, %.1e W through 10 m" % (which, sets[which]["m90"], len(rs), np.median([r["flux"] for r in rs]),
                                                 len(near), np.median([r["L_iso_W"] for r in near]) if near else np.nan,
                                                 np.median([r["P_10m_W"] for r in near]) if near else np.nan))


def limits_left(which, d):
    """Candidates a search's result file leaves after every test."""
    if which in ("starcat", "stis"):
        return [p for p in d["persistent"] if "forced_rest" in p and len(p["forced_rest"]) < 2
                and len(p["forced_obs"]) < 2]
    if which == "iue":
        return [p for p in d["persistent"] if "aperture_fixed" in p and not p["aperture_fixed"]
                and len(p["forced_rest"]) < 2 and len(p["forced_obs"]) < 2]
    if which == "cos":
        return [p for p in d["persistent"] if cos_left(p)]
    return d["left"]


limits_SEARCHES = (
    ("uv_laser_starcat", "starcat", "uv_laser.json", "Ultraviolet stellar spectra, STIS", "MAST HLSP StarCAT",
     "single exposures (Ayres 2010)", "every StarCAT exposure in E140M, E140H, E230M and E230H"),
    ("cos_laser_lines", "cos", "cos_laser.json", "Ultraviolet stellar spectra, COS", "MAST HST COS FUV corrtag",
     "CalCOS photon events", "COS FUV exposures with a SIMBAD star within 10 pc in the aperture"),
    ("iue_laser_lines", "iue", "iue_laser_6sig.json", "Ultraviolet stellar spectra, IUE", "MAST IUE NEWSIPS MXHI",
     "high dispersion", "images whose target is a SIMBAD star within 10 pc, by name or by its proper-motion track"),
    ("fuse_laser_lines", "fuse", "fuse_laser.json", "Ultraviolet stellar spectra, FUSE",
     "MAST FUSE CalFUSE combined channel spectra", "all4ttagfcal and all4histfcal",
     "observations with a SIMBAD star within 20 pc in the aperture, reference-point pointings left out"),
    ("uv_laser_stis", "stis", "stis_laser.json", "Ultraviolet stellar spectra, STIS after 2004", "MAST HST STIS x1d",
     "pipeline x1d, 2004 July on", "E140M, E140H, E230M and E230H exposures within 3 arcsec of a SIMBAD star's "
     "proper-motion track"))


limits_VETTED = {
    "starcat": "steady features of Arcturus, in every exposure covering them from 1998 to 2003, emission of the Mira R "
               "Leo (the strongest within 4 km/s of Fe II 2626.27) and C III] 1908.73 of the symbiotic EG And",
    "cos": "chromospheric lines of eps Eri within 2 km/s of Cr II 1330.93, C I 1468.41 and Si I 1769.79, steady from "
           "2015 to 2025",
    "iue": "the last were fixed to one aperture's camera position or lay 440 km/s redward of Si II 1808 and 1817 A in "
           "the brightest G and K dwarfs, absent from STIS",
    "fuse": "of 9 single-channel peaks 4 lie on Ly beta, C III and O VI and 5 reach 5.0-5.5 sigma with nothing in the "
            "overlapping channel",
    "stis": "all lie in twelve stars whose ultraviolet spectra are crowded with natural lines: cool supergiants and giants "
            "(alpha Ori, alpha Her, gamma Cru, beta Gem), T Tauri stars (TW Hya, RU Lup), eta Car, the Ap star GY And, "
            "beta Pic, two stars catalogued as [KT2007] 1181 and 1184, and Procyon, whose features are resolved peaks "
            "of its spectrum steady in every exposure"}


def limits_record():
    """Record each search's row and per-star limits."""
    t = json.load(open(os.path.join(limits_HERE, "uv_limits_table.json")))
    for sid, which, fn, approach, archive, release, selection in limits_SEARCHES:
        d = json.load(open(os.path.join(limits_HERE, fn)))
        s = json.load(open(os.path.join(limits_HERE, "uv_limits_%s.json" % which)))
        rows = [r for r in t["rows"] if r["set"] == which]
        near = [r for r in rows if r["plx"] and r["plx"] > 100]
        left = limits_left(which, d)
        if which == "fuse":
            null = "%d single-channel peaks, %d confirmed in a second channel" % (
                len(d["found"]), sum(1 for x in d["found"] if x["confirmed"]))
        else:
            null = "%d peaks, %d persistent, %d left after the cross-star, aperture and forced tests" % (
                d["peaks"], len(d["persistent"]), len(left))
        core.record_search(
            sid, approach, archive=archive, release=release, selection=selection, code=os.path.abspath(__file__),
            entry=which, n_targets=len(rows), null_summary=null, ledger="Ultraviolet stellar spectra",
            params=dict(plateau=s["plateau"], m50=s["m50"], m90=s["m90"], injected=s["n_injected"]),
            sensitivity="median per-star limit %.1e erg cm-2 s-1; %d stars within 10 pc, median %.1e W isotropic and "
                        "%.1e W through a 10 m aperture" % (np.median([r["flux"] for r in rows]), len(near),
                                                            np.median([r["L_iso_W"] for r in near]) if near else np.nan,
                                                            np.median([r["P_10m_W"] for r in near]) if near else np.nan),
            result="%d features pass every test; %s" % (len(left), limits_VETTED[which]))
        core.clear_targets(sid)
        core.record_targets(sid, [
            dict(target="simbad:%s" % (r["sp"].split(";")[0] if which == "starcat" and r.get("sp") else r["star"]),
                 upper_limit=r["flux"], unit="erg cm-2 s-1, unresolved line at 90 per cent of the recovery plateau",
                 note="%s, %.0f A; isotropic %.1e W; 10 m beam %.1e W" % (r["setting"], r["lam"], r["L_iso_W"] or np.nan,
                                                                         r["P_10m_W"] or np.nan))
            for r in rows])
        print("%s: %d stars, %d left" % (sid, len(rows), len(left)))


def limits_main():
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=("starcat", "iue", "fuse", "cos", "stis", "table", "record"))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    if a.which == "table":
        limits_table()
    elif a.which == "record":
        limits_record()
    else:
        limits_run(a.which, a.n, a.procs)


def limits_cli():
    quiet_warnings()
    limits_main()


# --------------------------------------------------------------------------------------------------------------

hst_pulses_DOC = r"""Ultraviolet laser pulses in STIS TIME-TAG and COS corrtag events of stars within 10 pc: same-bin photon clusters in 1 ms, 32 ms or 0.25 s beyond the local rate, Poisson FAP with trials.

    python -I seti.py ultraviolet hst_pulses inventory
    python -I seti.py ultraviolet hst_pulses fetch
    python -I seti.py ultraviolet hst_pulses search"""

hst_pulses_ROOT = os.environ.get("STIS_ROOT", core.data("stis_tag"))
hst_pulses_HERE = core.RESULTS
hst_pulses_MAST = "https://mast.stsci.edu/api/v0/invoke"
hst_pulses_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
hst_pulses_INSTRUMENTS = ["STIS/FUV-MAMA", "STIS/NUV-MAMA", "COS/FUV", "COS/NUV"]
hst_pulses_MIN_SPAN = 10.0
hst_pulses_LOCAL = 5.0
hst_pulses_SPOT = 30.0


def hst_pulses_invoke(req):
    data = urllib.parse.urlencode({"request": json.dumps(req)}).encode()
    for attempt in range(3):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(
                hst_pulses_MAST, data=data, headers=dict(core.UA, **{"Content-Type": "application/x-www-form-urlencoded"})),
                timeout=240).read())
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {"data": []}


def hst_pulses_inventory():
    os.makedirs(hst_pulses_ROOT, exist_ok=True)
    stars = uv_nearby(100)

    # one filtered MAST query, matched to the stars' tracks locally
    every, page = [], 1
    while True:
        d = hst_pulses_invoke({"service": "Mast.Caom.Filtered", "format": "json", "pagesize": 50000, "page": page,
                    "params": {"columns": "obsid,obs_id,instrument_name,t_exptime,target_name,s_ra,s_dec,t_min",
                               "filters": [{"paramName": "obs_collection", "values": ["HST"]},
                                           {"paramName": "instrument_name", "values": hst_pulses_INSTRUMENTS}]}})
        rows = d.get("data", [])
        every += rows
        if len(rows) < 50000:
            break
        page += 1
    print("%d HST observations with %s" % (len(every), ", ".join(hst_pulses_INSTRUMENTS)), flush=True)
    ora = np.array([float(x["s_ra"]) if x.get("s_ra") is not None else np.nan for x in every])
    ode = np.array([float(x["s_dec"]) if x.get("s_dec") is not None else np.nan for x in every])
    tmin = np.array([float(x["t_min"]) if x.get("t_min") is not None else 55000.0 for x in every])
    allobs = {}
    res = []
    for s in stars:
        dist = uv_track_dist(s, ora, ode, tmin)
        near = np.where(dist < 108.0)[0]
        res.append((s, near))
        for j in near:
            x = every[j]
            o = allobs.setdefault(str(x["obsid"]), dict(x, stars=[], dist={}))
            o["stars"].append(s["main_id"])
            o["dist"][" ".join(s["main_id"].split())] = round(float(dist[j]), 2)
    ids = list(allobs)
    tags = []
    for k in range(0, len(ids), 50):
        p = hst_pulses_invoke({"service": "Mast.Caom.Products", "format": "json", "pagesize": 20000,
                    "params": {"obsid": ",".join(ids[k:k + 50])}})
        for x in p.get("data", []):
            fn = x["productFilename"]
            if fn.endswith(("_tag.fits", "_corrtag_a.fits", "_corrtag_b.fits", "_corrtag.fits")):
                tags.append(dict(file=fn, uri=x["dataURI"], size=int(x.get("size") or 0), obsid=str(x.get("obsID")),
                                 parent=str(x.get("parent_obsid"))))
    uniq = {t["file"]: t for t in tags}
    json.dump(dict(obs=allobs, tags=list(uniq.values())), open(os.path.join(hst_pulses_ROOT, "inventory.json"), "w"), indent=1)
    print("%d HST STIS and COS MAMA observations near %d objects; %d event files, %.1f GB"
          % (len(allobs), sum(1 for s, near in res if len(near)), len(uniq), sum(t["size"] for t in uniq.values()) / 1e9))


def hst_pulses_fetch():
    inv = json.load(open(os.path.join(hst_pulses_ROOT, "inventory.json")))
    os.makedirs(os.path.join(hst_pulses_ROOT, "events"), exist_ok=True)

    def one(t):
        p = os.path.join(hst_pulses_ROOT, "events", t["file"])
        if os.path.exists(p):
            return 1
        b = core.get(hst_pulses_DOWNLOAD + t["uri"], timeout=600, retries=3, soft=True, pause=5)
        if not b or not b.startswith(b"SIMPLE"):
            return 0
        open(p + ".tmp", "wb").write(b)
        os.replace(p + ".tmp", p)
        return 1
    with ThreadPoolExecutor(4) as ex:
        n = sum(ex.map(one, inv["tags"]))
    print("%d of %d event files on disk" % (n, len(inv["tags"])))


hst_pulses_CAL_TARGETS = ("DARK", "BIAS", "DEUTERIUM", "WAVE", "WAVEHITM", "WAVELINE", "FLAT", "CCDFLAT", "NONE")


def hst_pulses_science(hd):
    """False for calibration exposures (darks, flats, lamps) filed under a science visit."""
    et = str(hd.get("EXPTYPE", "EXTERNAL/SCI")).strip().upper()
    tn = str(hd.get("TARGNAME", "")).strip().upper()
    return et in ("EXTERNAL/SCI", "") and not any(tn.startswith(c) for c in hst_pulses_CAL_TARGETS)


def hst_pulses_read_events(path):
    """Times (s), detector bin indices and test windows from a STIS _tag or COS _corrtag file."""
    from astropy.io import fits
    out = []
    with fits.open(path) as h:
        hd = h[0].header
        inst = str(hd.get("INSTRUME", "")).strip()
        if not hst_pulses_science(hd):
            return out
        for ext in h:
            if ext.name != "EVENTS" or ext.data is None or len(ext.data) == 0:
                continue
            d = ext.data
            cols = [c.upper() for c in d.columns.names]
            t = np.asarray(d["TIME"], float)
            if inst == "STIS":
                x, y = np.asarray(d["AXIS1"], float), np.asarray(d["AXIS2"], float)
                bx, by = 8, 8
                windows = (0.001, 0.032, 0.25)
                # merge MAMA double counts: events within 2 pixels and 250 us of the previous
                o = np.argsort(t, kind="stable")
                t, x, y = t[o], x[o], y[o]
                dup = np.zeros(len(t), bool)
                for lag in range(1, 7):
                    dup[lag:] |= (t[lag:] - t[:-lag] < 250e-6) & (np.abs(x[lag:] - x[:-lag]) <= 2) & \
                        (np.abs(y[lag:] - y[:-lag]) <= 2)
                t, x, y = t[~dup], x[~dup], y[~dup]
            else:
                x = np.asarray(d["XCORR"] if "XCORR" in cols else d["RAWX"], float)
                y = np.asarray(d["YCORR"] if "YCORR" in cols else d["RAWY"], float)
                fuv = str(hd.get("DETECTOR", "")).strip() == "FUV"
                bx, by = (16, 4) if fuv else (8, 8)
                good = np.ones(len(t), bool)
                if "DQ" in cols:
                    # COS DQ bits (Data Handbook table 2.19): 2 hot spot, 8 calibration, 16/1024 low response, 32 background, 64 burst, 128 bounds, 512 PHA, 2048 bad time, 4096 low PHA, 8192 gain sag, 16384 dark edge
                    good &= (np.asarray(d["DQ"], int) & (2 | 8 | 16 | 32 | 64 | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384)) == 0
                if fuv and "PHA" in cols:
                    pha = np.asarray(d["PHA"], int)
                    good &= (pha >= int(ext.header.get("PHALOWRA", 2))) & (pha <= int(ext.header.get("PHAUPPRA", 23)))
                if "WAVELENGTH" in cols:
                    # no wavelength: digitizer-edge rows outside the calibrated area
                    good &= np.asarray(d["WAVELENGTH"], float) > 0
                # aperture light only: events within the SP_LOC, SP_HGT extraction band in YFULL
                if "YFULL" in cols:
                    yf = np.asarray(d["YFULL"], float)
                    xf = np.asarray(d["XFULL"], float)
                    band = np.zeros(len(t), bool)
                    for s in "ABC":
                        loc, hgt = ext.header.get("SP_LOC_" + s), ext.header.get("SP_HGT_" + s)
                        if loc is None or hgt is None or float(loc) < 0:
                            continue
                        if fuv and str(hd.get("SEGMENT", "")).strip()[-1:] != s:
                            continue
                        slp = float(ext.header.get("SP_SLP_" + s, 0.0) or 0.0)
                        half = float(hgt) / 2 + 2
                        # slope reference column unrecorded: the trace is whichever of edge and centre puts more events in the band
                        best = None
                        for xr in (0.0, 8192.0 if fuv else 512.0):
                            inb = np.abs(yf - (float(loc) + slp * (xf - xr))) <= half
                            if best is None or inb.sum() > best.sum():
                                best = inb
                        band |= best
                    good &= band
                t, x, y = t[good], x[good], y[good]
                # TAGFLASH lamp intervals (LMP_ONn, LMPOFFn) cut with 2 s margins
                eh = ext.header
                flashes = [(float(eh["LMP_ON%d" % i]), float(eh.get("LMPOFF%d" % i, eh["LMP_ON%d" % i] + 15.0)))
                           for i in range(1, 40) if "LMP_ON%d" % i in eh]
                if flashes:
                    keep = np.ones(len(t), bool)
                    for on, off in flashes:
                        keep &= ~((t >= on - 2.0) & (t <= off + 2.0))
                    t, x, y = t[keep], x[keep], y[keep]
                windows = (0.032, 0.25)
            ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
            t, x, y = t[ok], x[ok], y[ok]
            b = (np.floor(x / bx).astype(np.int64) << 20) + np.floor(y / by).astype(np.int64)
            out.append(dict(inst=inst, target=str(hd.get("TARGNAME", "")).strip(), t=t, b=b, windows=windows,
                            det=str(hd.get("DETECTOR", "")).strip() + str(hd.get("SEGMENT", "")).strip(), bx=bx, by=by,
                            exptime=float(hd.get("TEXPTIME", 0) or (t.max() - t.min() if len(t) else 0))))
    return out


def hst_pulses_scan(ev):
    """Largest same-bin cluster per window, its bin's expectation and the false-alarm probability."""
    from scipy.stats import poisson
    t, b = ev["t"], ev["b"]
    if len(t) < 50:
        return None
    t0 = float(t.min())
    span = float(t.max()) - t0
    if span < hst_pulses_MIN_SPAN:
        return None
    # one sort key by bin then time, bins over 2 x SPOT seconds apart
    key_scale = 10.0 ** math.ceil(math.log10(span + 2 * hst_pulses_SPOT + 10.0))
    ub, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    key = inv.astype(np.float64) * key_scale + (t - t0)
    o = np.argsort(key)
    key, inv_s, ts = key[o], inv[o], t[o]
    tall = np.sort(t)
    # expectation: the larger of the mean rate scaled by the detector's LOCAL rate (flares, bursts) and the spot's own SPOT-second rate (airglow, hot pixels)
    near = np.searchsorted(tall, ts + hst_pulses_LOCAL, side="right") - np.searchsorted(tall, ts - hst_pulses_LOCAL, side="left")
    scale = near / (2 * hst_pulses_LOCAL * len(t) / span)
    rate = cnt[inv_s] / span * np.maximum(scale, 1e-3)
    own = np.searchsorted(key, key + hst_pulses_SPOT, side="right") - np.searchsorted(key, key - hst_pulses_SPOT, side="left")
    dur = np.minimum(ts + hst_pulses_SPOT, t0 + span) - np.maximum(ts - hst_pulses_SPOT, t0)
    res = {}
    for w in ev["windows"]:
        # COS 32 ms ticks: a one-tick window holds one tick
        k = np.searchsorted(key, key + w * (1 - 1e-6), side="right") - np.arange(len(key))
        mu = np.maximum(rate * w, (own - k) / np.maximum(dur - w, 1.0) * w)
        cand = np.where(k >= 2)[0]
        if len(cand) == 0:
            res[str(w)] = dict(k=1, mu=0.0, p=1.0, fap=1.0, t=None, detector=None, mean_detector=len(t) / span * w)
            continue
        # probability of the other k - 1 events at this bin's rate
        p = poisson.sf(k[cand] - 2, mu[cand])
        j = cand[int(np.argmin(p))]
        pmin = float(p.min())
        det = int(np.searchsorted(tall, ts[j] + w, side="right") - np.searchsorted(tall, ts[j], side="left"))
        fap = -math.expm1(len(t) * math.log1p(-min(pmin, 1 - 1e-16)))
        bj = int(ub[inv_s[j]])
        res[str(w)] = dict(k=int(k[j]), mu=float(mu[j]), p=pmin, fap=fap, t=float(ts[j]), detector=det,
                           mean_detector=len(t) / span * w, site=[ev["det"], (bj >> 20) * ev["bx"], (bj & 0xFFFFF) * ev["by"]])
    return dict(n=int(len(t)), span=span, windows=res)


def hst_pulses_scan_file(job):
    tg, stars = job
    p = os.path.join(hst_pulses_ROOT, "events", tg["file"])
    if not os.path.exists(p):
        return []
    try:
        evs = hst_pulses_read_events(p)
    except Exception as e:
        print(tg["file"], "unreadable:", str(e)[:60], flush=True)
        return []
    out = []
    for k, ev in enumerate(evs):
        r = hst_pulses_scan(ev)
        if r:
            out.append(dict(file=tg["file"], ext=k, inst=ev["inst"], target=ev["target"], stars=stars, **r))
    return out


def hst_pulses_search():
    from concurrent.futures import ProcessPoolExecutor
    inv = json.load(open(os.path.join(hst_pulses_ROOT, "inventory.json")))
    obs = inv["obs"]
    jobs = [(tg, obs.get(tg["parent"], obs.get(tg["obsid"], {})).get("stars", [])) for tg in inv["tags"]]
    with ProcessPoolExecutor(int(os.environ.get("STIS_PROCS", "2"))) as ex:
        rows = [r for rs in ex.map(hst_pulses_scan_file, jobs, chunksize=4) for r in rs]
    faps = [min(w["fap"] for w in r["windows"].values()) for r in rows]
    print("%d event lists (%d STIS, %d COS) of %d HST targets; %.1f million photons over %.1f days"
          % (len(rows), sum(r["inst"] == "STIS" for r in rows), sum(r["inst"] == "COS" for r in rows),
             len({r["target"] for r in rows}), sum(r["n"] for r in rows) / 1e6, sum(r["span"] for r in rows) / 86400))
    for thr in (1e-2, 1e-3, 1e-5):
        print("  lists with a same-spot cluster at false-alarm probability below %.0e: %d" % (thr, sum(f < thr for f in faps)))
    # explanations: a flare raises the whole detector; a site hosting two targets' strongest clusters is a hot pixel
    top = [(r, f, min(r["windows"].items(), key=lambda kv: kv[1]["fap"])) for r, f in zip(rows, faps) if f < 1e-2]
    sites = {}
    for r, f, w in top:
        sites.setdefault(tuple(w[1]["site"]), set()).add(r["target"])
    for r, f, w in top:
        x = w[1]
        rest, m = (x["detector"] or 0) - x["k"], x["mean_detector"]
        if rest > 3 * m + 5 * math.sqrt(m):
            r["why"] = "flare: %d other events detector-wide against a mean of %.1f" % (rest, m)
        elif len(sites[tuple(x["site"])]) > 1:
            r["why"] = "hot pixel: the same detector site (%s x %d y %d) hosts the strongest cluster of %s" % (
                x["site"][0], x["site"][1], x["site"][2], ", ".join(sorted(sites[tuple(x["site"])])))
        else:
            r["why"] = ""
    for r, f, w in sorted(top, key=lambda z: z[1])[:20]:
        print("  %s [%d] %s %-14s %s: %d photons over %.0f s; %s s window: %d in one spot (expected %.1e), FAP %.1e, at "
              "t %.4f with %d events detector-wide (mean %.2f)%s"
              % (r["file"], r["ext"], r["inst"], r["target"][:14], ",".join(s.strip() for s in r["stars"])[:30], r["n"],
                 r["span"], w[0], w[1]["k"], w[1]["mu"], w[1]["fap"], w[1]["t"] or 0, w[1]["detector"] or 0,
                 w[1]["mean_detector"], "; " + r["why"] if r["why"] else ""))
    left = [z for z in top if not z[0]["why"]]
    # smallest FAPs against the lists and windows tried
    trials = sum(len(r["windows"]) for r in rows)
    print("unexplained lists below 1e-3: %d (about %.1f expected from %d windows); below 1e-4: %d"
          % (sum(f < 1e-3 for r, f, w in left), trials * 1e-3, trials, sum(f < 1e-4 for r, f, w in left)))
    json.dump(rows, open(os.path.join(hst_pulses_HERE, "stis_pulses.json"), "w"), indent=1, default=str)


def hst_pulses_record():
    """Record per SIMBAD object from stis_pulses.json: lists, photons, hours and the smallest FAP left after flare and hot-pixel explanations."""
    rows = json.load(open(os.path.join(hst_pulses_HERE, "stis_pulses.json")))
    per = {}
    for r in rows:
        f = min(w["fap"] for w in r["windows"].values())
        for s in r["stars"] or [r["target"]]:
            s = " ".join(s.split())
            x = per.setdefault(s, dict(lists=0, n=0, span=0.0, fap=1.0, inst=set(), why=None))
            x["lists"] += 1
            x["n"] += r["n"]
            x["span"] += r["span"]
            x["inst"].add(r["inst"])
            if f < x["fap"]:
                x["fap"], x["why"] = f, r.get("why") or None
    out = [dict(target="simbad:" + s, stat=x["n"], p=x["fap"], unit="photons; p is the smallest list false-alarm probability",
                vet=x["why"], note="%d %s event lists over %.1f h" % (x["lists"], "+".join(sorted(x["inst"])), x["span"] / 3600))
           for s, x in per.items()]
    core.record_search("hst_photon_pulses", "HST photon-event pulses", archive="MAST HST STIS TIME-TAG and COS corrtag",
                           release="1997-2024", selection="SIMBAD objects with parallax above 100 mas within 0.03 deg",
                           code=os.path.abspath(__file__), entry="hst_pulses", params=dict(windows=[0.001, 0.032, 0.25], local=hst_pulses_LOCAL, spot=hst_pulses_SPOT,
                                                                        min_span=hst_pulses_MIN_SPAN),
                           n_targets=len(out), result="%d unexplained lists below FAP 1e-3 against %.1f expected; %d below 1e-4"
                           % (sum(1 for r in rows if not r.get("why") and min(w["fap"] for w in r["windows"].values()) < 1e-3),
                              1e-3 * sum(len(r["windows"]) for r in rows),
                              sum(1 for r in rows if not r.get("why") and min(w["fap"] for w in r["windows"].values()) < 1e-4)),
                           ledger="GALEX time-tagged photons")
    core.record_targets("hst_photon_pulses", out)
    print("recorded %d objects" % len(out))


def hst_pulses_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("inventory", "fetch", "search", "record"))
    a = p.parse_args()
    {"inventory": hst_pulses_inventory, "fetch": hst_pulses_fetch, "search": hst_pulses_search, "record": hst_pulses_record}[a.mode]()


def hst_pulses_cli():
    quiet_warnings()
    hst_pulses_main()


# --------------------------------------------------------------------------------------------------------------

galex_pulses_DOC = r"""Ultraviolet pulses from stars within 10 pc in GALEX time-tagged photons: largest counts in 5, 20 and 100 ms against the local rate, with control apertures and detector-wide bursts.

    python -I seti.py ultraviolet galex_pulses fetch [--band fuv|nuv]
    python -I seti.py ultraviolet galex_pulses search [--band fuv|nuv]
    python -I seti.py ultraviolet galex_pulses record [--band fuv|nuv]"""

galex_pulses_ROOT = os.environ.get("GALEX_ROOT", core.data("galex"))
galex_pulses_HERE = core.RESULTS
galex_pulses_R_SRC = 8.0
galex_pulses_R_CTRL = 60.0
galex_pulses_N_CTRL = 8
galex_pulses_WINDOWS = (0.005, 0.02, 0.1)
galex_pulses_LOCAL = 10.0
# band: photon-list suffix, directory of the cut apertures, results file
galex_pulses_BANDS = {"fuv": ("fd", "cut", "galex_pulses.json"), "nuv": ("nd", "cut_nuv", "galex_pulses_nuv.json")}


def galex_pulses_year_of(t):
    """GALEX time (seconds since 1980 January 6, the GPS epoch) as a decimal year."""
    return 1980.0 + (t + 5 * 86400.0) / 3.15576e7


def galex_pulses_targets():
    """Single-pointing eclipses and stars within 0.55 deg of each boresight, cached in ROOT/eclipse_stars.json."""
    cache = os.path.join(galex_pulses_ROOT, "eclipse_stars.json")
    if os.path.exists(cache):
        return {int(e): v for e, v in json.load(open(cache)).items()}
    import pyarrow.parquet as pq
    b = pq.read_table(os.path.join(galex_pulses_ROOT, "boresight.parquet")).to_pandas()
    m = pq.read_table(os.path.join(galex_pulses_ROOT, "metadata.parquet")).to_pandas().set_index("eclipse")
    stars = core.simbad("SELECT main_id, ra, dec, pmra, pmdec, plx_value, otype, sp_type FROM basic WHERE plx_value > "
                            "100 AND otype NOT IN ('Pl', 'Pl?') AND ra IS NOT NULL")
    ra0, de0 = np.radians(b["ra0"].values), np.radians(b["dec0"].values)
    by = {}
    for s in stars:
        a, d = math.radians(float(s["ra"])), math.radians(float(s["dec"]))
        cs = np.sin(d) * np.sin(de0) + np.cos(d) * np.cos(de0) * np.cos(ra0 - a)
        for e in set(b["eclipse"].values[np.degrees(np.arccos(np.clip(cs, -1, 1))) < 0.55]):
            e = int(e)
            if e in m.index and m.loc[e]["obstype"] != "AIS":
                by.setdefault(e, []).append(s)
    json.dump({str(e): v for e, v in by.items()}, open(cache, "w"))
    return by


def galex_pulses_column(key, name):
    """One photon-list column by byte range from the public bucket; None when absent."""
    import pyarrow.fs
    import pyarrow.parquet as pq
    fs = pyarrow.fs.S3FileSystem(anonymous=True, region="us-east-1", connect_timeout=60, request_timeout=600)
    for attempt in range(3):
        try:
            return pq.read_table("stpubdata/" + key, filesystem=fs, columns=[name]).column(name).to_numpy()
        except FileNotFoundError:
            return None
        except OSError as ex:
            if "NoSuchKey" in str(ex) or "does not exist" in str(ex):
                return None
            if attempt == 2:
                raise
            import time
            time.sleep(10)


def galex_pulses_extract(e, stars, band="fuv"):
    suffix, cut, _ = galex_pulses_BANDS[band]
    out = os.path.join(galex_pulses_ROOT, cut, "e%05d.npz" % e)
    if os.path.exists(out):
        return 1
    key = "galex/gap/galex_photon_list/e%05d/e%05d-%s.parquet" % (e, e, suffix)
    try:
        t = galex_pulses_column(key, "t")
        if t is None:
            return 0
        good = (galex_pulses_column(key, "flags") == 0) & ~galex_pulses_column(key, "mask").astype(bool)
        de = galex_pulses_column(key, "dec")
        good &= np.isfinite(de)
        tt_all = t[good]
        if not len(tt_all):
            return 0
        # keep photons in the stars' and controls' declination bands
        dy = galex_pulses_year_of(np.median(tt_all)) - 2000.0
        band_sel = np.zeros(len(de), bool)
        for s in stars:
            sde = float(s["dec"]) + float(s.get("pmdec") or 0) / 3.6e6 * dy
            band_sel |= np.abs(de - sde) < (galex_pulses_R_CTRL + 2 * galex_pulses_R_SRC) / 3600
        idx = np.nonzero(band_sel & good)[0]
        de = de[idx]
        ra = galex_pulses_column(key, "ra")[idx]
        tt = t[idx]
        del t, band_sel, good
    except Exception as ex:
        print("e%05d: %s" % (e, str(ex)[:120]), flush=True)
        return 0
    res = dict(eclipse=e, band=band, t0=float(tt_all.min()), t1=float(tt_all.max()), n_all=int(len(tt_all)))
    # detector-wide photons per 5 ms tick, for field-wide bursts
    ticks, counts = np.unique(np.round(tt_all / 0.005).astype(np.int64), return_counts=True)
    del tt_all
    res["tick_hist"] = np.bincount(counts)
    arrays = {}
    for i, s in enumerate(stars):
        sra = float(s["ra"]) + float(s.get("pmra") or 0) / 3.6e6 * dy / math.cos(math.radians(float(s["dec"])))
        sde = float(s["dec"]) + float(s.get("pmdec") or 0) / 3.6e6 * dy
        cd = math.cos(math.radians(sde))
        for j in range(galex_pulses_N_CTRL + 1):
            if j == 0:
                cra, cde = sra, sde
            else:
                ang = 2 * math.pi * (j - 1) / galex_pulses_N_CTRL
                cra, cde = sra + galex_pulses_R_CTRL / 3600 * math.cos(ang) / cd, sde + galex_pulses_R_CTRL / 3600 * math.sin(ang)
            sel = (np.abs(de - cde) < galex_pulses_R_SRC / 3600) & (np.hypot(((ra - cra + 180) % 360 - 180) * cd, de - cde) * 3600 < galex_pulses_R_SRC)
            arrays["s%d_a%d" % (i, j)] = tt[sel]
            if j == 0:
                arrays["tick_s%d" % i] = counts[np.searchsorted(ticks, np.round(tt[sel] / 0.005).astype(np.int64))]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(dict(res, tick_hist=res["tick_hist"].tolist(),
                                                  stars=[s["main_id"] for s in stars])), **arrays)
    return 1


def galex_pulses_fetch(band):
    by = galex_pulses_targets()
    print("%d single-pointing eclipses cover %d objects within 10 pc" % (len(by), len({s["main_id"] for v in by.values() for s in v})),
          flush=True)
    done = [0]

    def one(kv):
        r = galex_pulses_extract(kv[0], kv[1], band)
        done[0] += 1
        if done[0] % 10 == 0:
            print("%d of %d eclipses read" % (done[0], len(by)), flush=True)
        return r
    with ThreadPoolExecutor(int(os.environ.get("GALEX_THREADS", "3"))) as ex:
        n = sum(ex.map(one, sorted(by.items())))
    print("%d eclipses with %s photon lists extracted" % (n, band))


def galex_pulses_clusters(t, span):
    """Per window length: largest photon-started count, its local expectation and the FAP over the exposure."""
    from scipy.stats import poisson
    out = {}
    t = np.sort(t)
    for w in galex_pulses_WINDOWS:
        if len(t) < 2:
            out[w] = (len(t), 0.0, 1.0)
            continue
        k = np.searchsorted(t, t + w - 1e-9) - np.arange(len(t))
        i = int(np.argmax(k))
        n_local = np.searchsorted(t, t[i] + galex_pulses_LOCAL) - np.searchsorted(t, t[i] - galex_pulses_LOCAL)
        lam = max(n_local - k[i], 1) / (2 * galex_pulses_LOCAL) * w
        p = poisson.sf(k[i] - 1, lam)
        fap = -math.expm1(max(span / w, 1) * math.log1p(-min(p, 1 - 1e-16)))
        out[w] = (int(k[i]), float(lam), float(fap))
    return out


def galex_pulses_search(band):
    _, cut, result = galex_pulses_BANDS[band]
    files = sorted(os.listdir(os.path.join(galex_pulses_ROOT, cut)))
    rows, ctrl = [], []
    for fn in files:
        z = np.load(os.path.join(galex_pulses_ROOT, cut, fn))
        meta = json.loads(str(z["meta"]))
        span = meta["t1"] - meta["t0"]
        for i, name in enumerate(meta["stars"]):
            t = z["s%d_a0" % i]
            c = galex_pulses_clusters(t, span)
            best = min(c.values(), key=lambda x: x[2])
            rows.append(dict(star=name, eclipse=meta["eclipse"], n=int(len(t)), span=span, clusters={str(k): v for k, v in c.items()},
                             fap=best[2], tick_max=int(z["tick_s%d" % i].max()) if len(t) else 0,
                             tick_hist=meta["tick_hist"]))
            for j in range(1, galex_pulses_N_CTRL + 1):
                cc = galex_pulses_clusters(z["s%d_a%d" % (i, j)], span)
                ctrl.append(min(v[2] for v in cc.values()))
    ctrl = np.array(ctrl)
    print("%d star-eclipses (%d stars), median %d source photons over %.0f s; %d control apertures"
          % (len(rows), len({r["star"] for r in rows}), np.median([r["n"] for r in rows]), np.median([r["span"] for r in rows]),
             len(ctrl)))
    for thr in (1e-2, 1e-3, 1e-4):
        print("  false-alarm probability below %.0e: targets %d (%.1f expected from the controls' rate)"
              % (thr, sum(r["fap"] < thr for r in rows), len(rows) * np.mean(ctrl < thr)))
    for r in sorted(rows, key=lambda r: r["fap"])[:15]:
        c = r["clusters"]
        print("  %-24s e%05d: %d photons over %.0f s; 5 ms max %d (exp %.1e), 20 ms %d (exp %.1e), 100 ms %d (exp %.1e); "
              "FAP %.1e; busiest detector tick holding a source photon %d photons"
              % (r["star"], r["eclipse"], r["n"], r["span"], c["0.005"][0], c["0.005"][1], c["0.02"][0], c["0.02"][1],
                 c["0.1"][0], c["0.1"][1], r["fap"], r["tick_max"]))
    json.dump(dict(rows=rows, ctrl=ctrl.tolist()), open(os.path.join(galex_pulses_HERE, result), "w"), indent=1)


def galex_pulses_record(band):
    """Record one row per star with its lowest FAP over its eclipses."""
    _, _, result = galex_pulses_BANDS[band]
    r = json.load(open(os.path.join(galex_pulses_HERE, result)))
    rows, ctrl = r["rows"], np.array(r["ctrl"])
    best = {}
    for x in rows:
        if x["star"] not in best or x["fap"] < best[x["star"]]["fap"]:
            best[x["star"]] = dict(x, n_ecl=sum(1 for y in rows if y["star"] == x["star"]))
    run = "galex_%s_pulses" % band
    core.record_search(
        run, "GALEX time-tagged photons", archive="STScI S3 galex/gap/galex_photon_list (single-pointing eclipses)",
        release="GALEX %s photon lists" % band.upper(),
        selection="SIMBAD objects with parallax above 100 mas inside 0.55 deg of a single-pointing eclipse boresight",
        code="seti.py ultraviolet galex_pulses", params=dict(r_src=galex_pulses_R_SRC, n_ctrl=galex_pulses_N_CTRL, windows=list(galex_pulses_WINDOWS), local_s=galex_pulses_LOCAL),
        n_targets=len(best),
        null_summary=dict(below_1e2=sum(1 for x in rows if x["fap"] < 1e-2),
                          below_1e3=sum(1 for x in rows if x["fap"] < 1e-3),
                          expected_1e2=float(len(rows) * np.mean(ctrl < 1e-2)), controls=int(len(ctrl))),
        sensitivity="several photons within 5, 20 or 100 ms in an 8 arcsec aperture against the local rate",
        result="%d star-eclipses of %d stars: %d clusters at false-alarm probability below 0.01 against %.1f expected "
               "from %d control apertures" % (len(rows), len(best), sum(1 for x in rows if x["fap"] < 1e-2),
                                              len(rows) * np.mean(ctrl < 1e-2), len(ctrl)),
        ledger="GALEX time-tagged photons")
    core.clear_targets(run)
    core.record_targets(run, [
        dict(target="simbad:%s" % s, p=float(x["fap"]),
             note="best of %d eclipses: e%05d, %d photons over %.0f s; 5 ms max %d"
                  % (x["n_ecl"], x["eclipse"], x["n"], x["span"], x["clusters"]["0.005"][0])) for s, x in best.items()])
    print("recorded")


def galex_pulses_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "search", "record"))
    p.add_argument("--band", choices=tuple(galex_pulses_BANDS), default="fuv")
    a = p.parse_args()
    {"fetch": galex_pulses_fetch, "search": galex_pulses_search, "record": galex_pulses_record}[a.mode](a.band)


def galex_pulses_cli():
    quiet_warnings()
    galex_pulses_main()


# --------------------------------------------------------------------------------------------------------------

galex_census_DOC = r"""Bursts across the GALEX NUV detector as the false-positive population: 100 ms clusters per 16 arcsec cell, judged off streaks and local rises, with a detector-position test.

    python -I seti.py ultraviolet galex_census census [--eclipses FILE] [--part k/n]
    python -I seti.py ultraviolet galex_census judge
    python -I seti.py ultraviolet galex_census detector ECLIPSE RA DEC T [--fixed]
    python -I seti.py ultraviolet galex_census burstpos ECLIPSE
    python -I seti.py ultraviolet galex_census record"""

galex_census_ROOT = os.environ.get("GALEX_CENSUS_ROOT", core.data("galex_census"))
galex_census_KEY = "stpubdata/galex/gap/galex_photon_list/e%05d/e%05d-nd.parquet"
galex_census_W, galex_census_R, galex_census_FAP = 0.1, 16.0, 1e-3


def galex_census_photons(ecl, columns=("t", "ra", "dec")):
    """Good photons of one NUV eclipse from the public bucket (flags 0, unmasked, finite position)."""
    import pyarrow.fs
    import pyarrow.parquet as pq
    fs = pyarrow.fs.S3FileSystem(anonymous=True, region="us-east-1", connect_timeout=60, request_timeout=900)
    cols = list(dict.fromkeys(list(columns) + ["flags", "mask", "dec"]))
    tab = pq.read_table(galex_census_KEY % (ecl, ecl), filesystem=fs, columns=cols)
    good = ((tab.column("flags").to_numpy() == 0) & ~tab.column("mask").to_numpy().astype(bool)
            & np.isfinite(tab.column("dec").to_numpy()))
    return {c: tab.column(c).to_numpy()[good] for c in columns}


def galex_census_eclipse(ecl):
    """Census of one eclipse into ROOT/e<eclipse>.json: every cluster below the FAP threshold."""
    from scipy.stats import poisson
    W, R = galex_census_W, galex_census_R
    out_p = os.path.join(galex_census_ROOT, "e%05d.json" % ecl)
    if os.path.exists(out_p):
        return
    try:
        ph = galex_census_photons(ecl)
    except Exception as e:
        json.dump(dict(eclipse=ecl, error=str(e)[:200]), open(out_p, "w"))
        return
    t, ra, de = ph["t"], ph["ra"], ph["dec"]
    if len(t) < 1000:
        json.dump(dict(eclipse=ecl, error="too few photons"), open(out_p, "w"))
        return
    t0 = float(t.min())
    tr = t - t0
    span = float(tr.max())
    ra0, de0 = float(np.median(ra)), float(np.median(de))
    x = ((ra - ra0 + 180) % 360 - 180) * math.cos(math.radians(de0)) * 3600.0
    y = (de - de0) * 3600.0
    nwin = span / W
    hits, cells = [], {}
    for ox, oy in ((0, 0), (8, 0), (0, 8), (8, 8)):
        ix = np.floor((x + ox) / R).astype(np.int64)
        iy = np.floor((y + oy) / R).astype(np.int64)
        keyc = (ix - ix.min()) * 100000 + (iy - iy.min())
        order = np.lexsort((tr, keyc))
        kc, ts = keyc[order], tr[order]
        v = kc.astype(np.float64) * 4096.0 + ts
        cnt = np.searchsorted(v, v + W, side="left") - np.arange(len(v))
        uniq, inv, ncell = np.unique(kc, return_inverse=True, return_counts=True)
        rate = ncell / span
        mu = rate * W
        if (ox, oy) == (0, 0):
            h, _ = np.histogram(rate, bins=[0, 0.1, 0.3, 1, 3, 10, 1e9])
            cells = dict(zip(["<0.1", "0.1-0.3", "0.3-1", "1-3", "3-10", ">10"], h.tolist()))
        kthr = np.full(len(uniq), 10 ** 6, np.int64)
        for k in range(2, 600):
            ok = (kthr == 10 ** 6) & (poisson.sf(k - 1, mu) * nwin < galex_census_FAP)
            kthr[ok] = k
            if not (kthr == 10 ** 6).any():
                break
        for h_ in np.nonzero(cnt >= kthr[inv])[0]:
            j = order[h_]
            fap = float(poisson.sf(cnt[h_] - 1, mu[inv[h_]]) * nwin)
            # the cell's own rate within 10 s either side, the window left out
            lo = np.searchsorted(v, kc[h_] * 4096.0 + max(ts[h_] - 10.0, 0.0))
            hi = np.searchsorted(v, kc[h_] * 4096.0 + ts[h_] + 10.0)
            rate_loc = max(int(hi - lo - cnt[h_]), 1) / 20.0
            fap_loc = float(poisson.sf(cnt[h_] - 1, max(rate_loc, rate[inv[h_]] * 0.5) * W) * nwin)
            hits.append(dict(t=float(ts[h_]), n=int(cnt[h_]), x=float(x[j]), y=float(y[j]), ra=float(ra[j]),
                             dec=float(de[j]), rate=float(rate[inv[h_]]), rate_local=float(rate_loc), fap=fap,
                             fap_local=fap_loc))
    cl = []
    for h_ in sorted(hits, key=lambda d: d["fap"]):
        if not any(abs(c["t"] - h_["t"]) < 0.3 and math.hypot(c["x"] - h_["x"], c["y"] - h_["y"]) < 24 for c in cl):
            cl.append(h_)
    for c in cl:
        c["neighbours"] = sum(1 for o in cl if o is not c and abs(o["t"] - c["t"]) < 2.0
                              and math.hypot(o["x"] - c["x"], o["y"] - c["y"]) < 60.0)
    json.dump(dict(eclipse=ecl, t0=t0, span=span, n_photons=int(len(t)), cells=cells, clusters=cl), open(out_p, "w"))


def galex_census_eclipses(path=None):
    if path:
        return [int(e) for e in json.load(open(path))]
    rows = json.load(open(core.results("galex_pulses_nuv.json")))["rows"]
    return sorted({int(r["eclipse"]) for r in rows})


def galex_census_run(path=None, part=None):
    os.makedirs(galex_census_ROOT, exist_ok=True)
    ecls = galex_census_eclipses(path)
    k, n = (int(x) for x in part.split("/")) if part else (0, 1)
    for e in ecls[k::n]:
        t_ = time.time()
        galex_census_eclipse(e)
        print("e%05d done in %.0f s" % (e, time.time() - t_), flush=True)


def galex_census_judge():
    """Bursts of 7 or more photons in 100 ms on cells of 0.5-5 counts/s, off streaks and outside local rises."""
    n_ecl = n_err = streaks = flares = 0
    src_seconds = 0.0
    bursts = []
    for f in sorted(glob.glob(os.path.join(galex_census_ROOT, "e*.json"))):
        d = json.load(open(f))
        if "error" in d:
            n_err += 1
            continue
        n_ecl += 1
        src_seconds += (d["cells"].get("1-3", 0) + d["cells"].get("3-10", 0)) * d["span"]
        for c in d["clusters"]:
            if c["n"] < 7 or not 0.5 <= c["rate"] <= 5.0:
                continue
            if c.get("neighbours", 0) > 0:
                streaks += 1
            elif c.get("fap_local", 1.0) >= 1e-4:
                flares += 1
            else:
                bursts.append(dict(c, eclipse=d["eclipse"]))
    # one burst per cell and moment across the four grids
    uniq = []
    for b in sorted(bursts, key=lambda b: b["fap_local"]):
        if not any(b["eclipse"] == u["eclipse"] and abs(b["t"] - u["t"]) < 0.2 and abs(b["ra"] - u["ra"]) < 0.01
                   and abs(b["dec"] - u["dec"]) < 0.01 for u in uniq):
            uniq.append(b)
    print("%d eclipses judged (%d failed); %.3g source-seconds at 1-10 counts/s" % (n_ecl, n_err, src_seconds))
    print("clusters of 7+ photons in 100 ms on 0.5-5 counts/s cells: %d on streaks, %d within a local rise, %d isolated"
          % (streaks, flares, len(uniq)))
    print("isolated bursts per source-second: %.2g" % (len(uniq) / max(src_seconds, 1)))
    for b in uniq[:30]:
        print("  e%05d t %.2f n %d at RA %.5f Dec %+.5f, cell %.2f c/s, local %.2f c/s, fap %.1e, local fap %.1e"
              % (b["eclipse"], b["t"], b["n"], b["ra"], b["dec"], b["rate"], b["rate_local"], b["fap"], b["fap_local"]))
    json.dump(dict(n_ecl=n_ecl, src_seconds=src_seconds, streaks=streaks, flares=flares, bursts=uniq),
              open(os.path.join(galex_census_ROOT, "judged.json"), "w"))


def galex_census_boresight(tr, ra, de):
    """Boresight per photon from each second's median position (over 100 photons), 5 s running median, linear within the second; with cos(dec)."""
    from scipy.ndimage import median_filter
    sec = np.floor(tr).astype(int)
    ns = sec.max() + 1
    order = np.argsort(sec)
    st = np.searchsorted(sec[order], np.arange(ns))
    en = np.searchsorted(sec[order], np.arange(ns), side="right")
    bra, bde = np.full(ns, np.nan), np.full(ns, np.nan)
    for s in range(ns):
        idx = order[st[s]:en[s]]
        if len(idx) > 100:
            bra[s], bde[s] = np.median(ra[idx]), np.median(de[idx])
    ok = np.isfinite(bra)
    sb_ra = np.interp(np.arange(ns), np.nonzero(ok)[0], median_filter(bra[ok], 5))
    sb_de = np.interp(np.arange(ns), np.nonzero(ok)[0], median_filter(bde[ok], 5))
    fr = tr - sec
    bore_ra = sb_ra[sec] + (np.r_[sb_ra[1:], sb_ra[-1]][sec] - sb_ra[sec]) * fr
    bore_de = sb_de[sec] + (np.r_[sb_de[1:], sb_de[-1]][sec] - sb_de[sec]) * fr
    return bore_ra, bore_de, math.cos(math.radians(np.nanmedian(bde))), (sb_ra, sb_de)


def galex_census_detector(ecl, star_ra, star_de, t_burst, fixed=False):
    """A star's burst in detector coordinates, and that detector spot over the whole visit against the star."""
    ph = galex_census_photons(ecl)
    t, ra, de = ph["t"], ph["ra"], ph["dec"]
    tr = t - t.min()
    bore_ra, bore_de, cd, (sb_ra, sb_de) = galex_census_boresight(tr, ra, de)
    dx = ((ra - bore_ra + 180) % 360 - 180) * cd * 3600.0
    dy = (de - bore_de) * 3600.0
    print("visit %.0f s, %d photons; boresight dither range RA %.1f arcsec, Dec %.1f arcsec"
          % (tr.max(), len(t), (sb_ra.max() - sb_ra.min()) * cd * 3600, (sb_de.max() - sb_de.min()) * 3600))
    sp = np.hypot(np.diff(sb_ra) * cd * 3600, np.diff(sb_de) * 3600)
    print("boresight speed: median %.2f arcsec/s, 95th percentile %.2f" % (np.median(sp), np.percentile(sp, 95)))
    # star centroid: densest 2 arcsec cell within 60 arcsec, then a 6 arcsec mean
    x60 = ((ra - star_ra + 180) % 360 - 180) * cd * 3600
    y60 = (de - star_de) * 3600
    w60 = np.hypot(x60, y60) < 60
    H, xe, ye = np.histogram2d(x60[w60], y60[w60], bins=60, range=[[-60, 60], [-60, 60]])
    ix, iy = np.unravel_index(np.argmax(H), H.shape)
    cx, cy = (xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2
    m = w60 & (np.hypot(x60 - cx, y60 - cy) < 6)
    if not fixed:
        star_ra += np.mean(x60[m]) / 3600 / cd
        star_de += np.mean(y60[m]) / 3600
    print("star centroid RA %.5f Dec %.5f from %d photons within 6 arcsec" % (star_ra, star_de, m.sum()))
    r8 = np.hypot(((ra - star_ra + 180) % 360 - 180) * cd * 3600, (de - star_de) * 3600) < 8
    cand = np.nonzero(r8 & (tr > t_burst - 5) & (tr < t_burst + 5))[0]
    tt = tr[cand]
    best = None
    for x in tt:
        k = int(np.sum((tt >= x) & (tt < x + 0.1)))
        if best is None or k > best[0]:
            best = (k, x)
    t_best = best[1]
    sel = cand[(tt >= t_best) & (tt < t_best + 0.1)]
    px, py = dx[sel].mean(), dy[sel].mean()
    print("burst: %d photons from %.3f s; sky centre RA %.5f Dec %.5f; detector centre %.1f %.1f arcsec, spread %.1f arcsec"
          % (len(sel), t_best, ra[sel].mean(), de[sel].mean(), px, py, np.hypot(dx[sel] - px, dy[sel] - py).mean()))
    on_star = np.hypot(((ra - star_ra + 180) % 360 - 180) * cd * 3600, (de - star_de) * 3600) < 6
    on_spot = np.hypot(dx - px, dy - py) < 6
    star_det_x = ((star_ra - bore_ra + 180) % 360 - 180) * cd * 3600
    star_det_y = (star_de - bore_de) * 3600
    sep_t = np.hypot(star_det_x - px, star_det_y - py)
    print("star (sky, 6 arcsec): %d photons; detector spot (6 arcsec): %d photons; spot while the star is 20+ arcsec "
          "away: %d photons" % (on_star.sum(), on_spot.sum(), (on_spot & (sep_t > 20)).sum()))
    print("star track on the detector: x %.0f..%.0f, y %.0f..%.0f arcsec; burst spot %.1f %.1f; nearest approach of the "
          "track to the spot %.1f arcsec" % (star_det_x.min(), star_det_x.max(), star_det_y.min(), star_det_y.max(), px, py,
                                            sep_t.min()))

    def tops(mask, label):
        ts = np.sort(tr[mask])
        idx = np.argsort(tr[mask])
        cnt = np.searchsorted(ts, ts + 0.1) - np.arange(len(ts))
        seen = []
        for i in np.argsort(-cnt):
            if cnt[i] < 4:
                break
            if any(abs(ts[i] - s) < 0.2 for s in seen):
                continue
            seen.append(ts[i])
            j = np.nonzero(mask)[0][idx[i]]
            print("  %s: %d photons in 100 ms at %.3f s; star-spot separation on the detector %.1f arcsec"
                  % (label, cnt[i], ts[i], sep_t[j]))

    tops(on_spot, "detector spot")
    tops(on_star, "star")
    away = on_spot & (sep_t > 20)
    tops(away, "spot, star away")
    frac = np.mean(sep_t > 20)
    print("fraction of the visit the star is 20+ arcsec from the spot: %.2f; spot rate then %.3f c/s; background in "
          "6 arcsec %.3f c/s" % (frac, away.sum() / (frac * tr.max()), np.sum(np.hypot(dx - px - 120, dy - py) < 6) / tr.max()))


def galex_census_burstpos(ecl):
    """Detector position and radius of every judged burst of one eclipse."""
    d = json.load(open(os.path.join(galex_census_ROOT, "judged.json")))
    ph = galex_census_photons(ecl, ("t", "ra", "dec", "detrad"))
    t, ra, de, dr = ph["t"], ph["ra"], ph["dec"], ph["detrad"]
    tr = t - t.min()
    bore_ra, bore_de, cd, _ = galex_census_boresight(tr, ra, de)
    for b in sorted((b for b in d["bursts"] if b["eclipse"] == ecl), key=lambda b: b["t"]):
        w = ((tr >= b["t"] - 0.02) & (tr < b["t"] + 0.12)
             & (np.hypot(((ra - b["ra"] + 180) % 360 - 180) * cd * 3600, (de - b["dec"]) * 3600) < 12))
        dx = ((ra[w] - bore_ra[w] + 180) % 360 - 180) * cd * 3600
        dy = (de[w] - bore_de[w]) * 3600
        print("t %7.2f n %2d sky %.5f %+.5f -> detector %7.1f %7.1f arcsec (spread %.1f), detrad %.4f deg"
              % (b["t"], w.sum(), b["ra"], b["dec"], dx.mean(), dy.mean(),
                 np.hypot(dx - dx.mean(), dy - dy.mean()).mean(), np.median(dr[w])))


def galex_census_record():
    """Record the judged census: one row per isolated burst."""
    d = json.load(open(os.path.join(galex_census_ROOT, "judged.json")))
    b = d["bursts"]
    core.record_search(
        "galex_burst_census", "GALEX NUV burst census", archive="STScI S3 galex/gap/galex_photon_list (single-pointing eclipses)",
        release="%d NUV eclipses" % d["n_ecl"], selection="every 16 arcsec cell of the eclipses of the NUV pulse search",
        code=os.path.abspath(__file__), entry="galex_census", n_targets=len(b),
        params=dict(window_s=galex_census_W, cell_arcsec=galex_census_R, fap=galex_census_FAP, min_photons=7,
                    cell_rate=[0.5, 5.0], local_fap=1e-4),
        null_summary=dict(streaks=d["streaks"], local_rises=d["flares"], source_seconds=d["src_seconds"]),
        sensitivity="7 or more photons in 100 ms on cells of 0.5 to 5 counts/s",
        result="%d eclipses: %d isolated bursts of 7 or more photons in 100 ms (%.2g per source-second), %d clusters on "
               "satellite streaks and %d within local rises" % (d["n_ecl"], len(b), len(b) / max(d["src_seconds"], 1),
                                                               d["streaks"], d["flares"]),
        ledger="Ultraviolet pulses")
    core.clear_targets("galex_burst_census")
    core.record_targets("galex_burst_census", [dict(
        target="galex:e%05d t %.2f" % (x["eclipse"], x["t"]), stat=x["n"], p=x["fap_local"], unit="photons in 100 ms",
        note="RA %.5f Dec %+.5f, cell %.2f counts/s, local %.2f counts/s" % (x["ra"], x["dec"], x["rate"], x["rate_local"]))
        for x in b])
    print("recorded %d bursts" % len(b))


def galex_census_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("census", "judge", "detector", "burstpos", "record"))
    p.add_argument("args", nargs="*")
    p.add_argument("--eclipses", default=None)
    p.add_argument("--part", default=None)
    p.add_argument("--fixed", action="store_true")
    a = p.parse_args()
    if a.mode == "census":
        galex_census_run(a.eclipses, a.part)
    elif a.mode == "judge":
        galex_census_judge()
    elif a.mode == "detector":
        e, ra_, de_, tb = a.args
        galex_census_detector(int(e), float(ra_), float(de_), float(tb), a.fixed)
    elif a.mode == "burstpos":
        galex_census_burstpos(int(a.args[0]))
    else:
        galex_census_record()


# --------------------------------------------------------------------------------------------------------------
# ultraviolet from Earth's transit zone

etzuv_DOC = r"""Ultraviolet from Earth's transit zone: Gaia Catalogue of Nearby Stars objects within 0.264 deg of the ecliptic, which see Earth cross the Sun, carried to 2007.0, against GALEX GUVcat AIS (II/335) within 5 arcsec by CDS XMatch, a detection in either band; the share with an ultraviolet source in the zone against the share 1-5 deg from the ecliptic and against the zone moved 2 arcmin in eight directions. Matches read in SIMBAD.

    python -I seti.py ultraviolet etzuv [record] [--radius-arcsec 5] [--epoch 2007] [--catalogue II/335/galex_ais]

Another CDS catalogue of ultraviolet sources, with the epoch of its positions, is recorded under its own run."""


def etzuv_main(record=False, radius_arcsec=5.0, epoch=2007.0, catalogue="II/335/galex_ais"):
    from setitools.neighbourhood import etz_share, etz_vets
    r = etz_share(catalogue, radius_arcsec, epoch)
    vets = etz_vets(r["got"])
    mean = float(np.mean(r["null"]))
    print("%d zone stars and %d stars 1-5 deg away; %s within %g arcsec: zone %d, away %d (%.2f expected in the zone, p %.3g); zone moved 2 arcmin %s (mean %.1f); "
          "zone matches by SIMBAD type %s" % (r["zone"], r["near"], catalogue, radius_arcsec, len(r["got"]), r["far"], r["exp"], r["p"], r["null"], mean,
                                              sorted({(v[0] or (None, "none"))[1] for v in vets.values()})))
    if record:
        galex = catalogue == "II/335/galex_ais"
        core.record_run("etz_galex_stars" if galex else "etz_%s_stars" % catalogue.replace("/", "_"), "Ultraviolet from Earth's transit zone",
                        "Gaia Catalogue of Nearby Stars (external.gaiaedr3_gcns_main_1); %s by CDS XMatch; SIMBAD" % ("GALEX GUVcat AIS (II/335)" if galex else catalogue),
                        "stars within 100 pc and 0.264 deg of the ecliptic: %d" % r["zone"], os.path.abspath(__file__),
                        "etzuv", r["zone"], "stars", dict(zone_deg=0.264, control_deg=[1, 5], radius_arcsec=radius_arcsec, epoch=epoch, catalogue=catalogue),
                        "the share 1-5 deg away (%d of %d); the zone moved 2 arcmin: %s" % (r["far"], r["near"], r["null"]),
                        "ultraviolet from a star that sees Earth transit, beyond the rate of stars beside the zone",
                        "%d against %.2f from the share beside the zone (p %.3g) and a mean of %.1f moved" % (len(r["got"]), r["exp"], r["p"], mean),
                        [dict(target="gaia_dr3:%s" % n[1:], vet=vets[n][1] or ("chance-rate" if r["p"] > 0.01 else None), note="SIMBAD %s; NUV %s, FUV %s at %.1f arcsec" % (
                            vets[n][0], m.get("NUVmag"), m.get("FUVmag"), float(m["angDist"]))) for n, m in sorted(r["got"].items())])
        print("recorded")


bluest_DOC = r"""Nearby light bluer than any star: Gaia Catalogue of Nearby Stars objects (probability above 0.9) whose BP-RP lies below -0.65, bluer than the Rayleigh-Jeans tail of any photosphere gives in Gaia's bands, a spectrum rising toward the blue faster than heat allows, each read for its BP/RP flux excess against the single-star relation, its white-dwarf probability and its SIMBAD type; the hottest white dwarfs and subdwarfs reach -0.6, and a blend or a faint source's BP noise makes a bluer colour without a bluer star.

    python -I seti.py ultraviolet bluest [record] [--colour-max -0.65] [--excess-max 0.2]"""


def bluest_main(record=False, colour_max=-0.65, excess_max=0.2):
    from setitools import catalogues
    from setitools.core import num
    rows = catalogues.gcns("source_id, ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_bp_rp_excess_factor, ruwe, wd_prob, ipd_frac_multi_peak",
                           "gcns_prob > 0.9 AND phot_bp_mean_mag - phot_rp_mean_mag < %g" % colour_max, ranges=12)
    sim = {}
    ids = [str(r["source_id"]) for r in rows]
    for k in range(0, len(ids), 200):
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % x for x in ids[k:k + 200])):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    out = []
    for r in rows:
        c = num(r["phot_bp_mean_mag"]) - num(r["phot_rp_mean_mag"])
        # the single-star BP/RP excess (Riello et al. 2021) at the colour
        ex = (num(r["phot_bp_rp_excess_factor"]) or 0) - (1.154360 + 0.033772 * c + 0.032277 * c * c)
        o = dict(id=str(r["source_id"]), c=c, g=num(r["phot_g_mean_mag"]), plx=num(r["parallax"]), ex=ex, ruwe=num(r["ruwe"]), wd=num(r["wd_prob"]),
                 multi=num(r["ipd_frac_multi_peak"]), simbad=sim.get(str(r["source_id"])))
        typ = (o["simbad"] or (None, "", ""))[1]
        o["vet"] = ("artefact" if abs(ex) > excess_max or (o["multi"] or 0) > 2 or (o["ruwe"] or 0) > 1.4 else
                    "white-dwarf" if (o["wd"] or 0) > 0.5 or typ in ("WD*", "WD?", "HS*", "HS?", "sdB", "sdO") or "D" in (o["simbad"] or (None, "", ""))[2][:2] else
                    "natural-type" if typ not in ("", "*", "PM*", "HV*") else None)
        out.append(o)
    left = [o for o in out if o["vet"] is None]
    print("%d stars within 100 pc with BP-RP below %g: %d with a BP/RP flux excess beyond %g, several peaks or RUWE above 1.4, %d white dwarfs or hot subdwarfs, %d of other "
          "known types; unexplained %d" % (len(out), colour_max, sum(o["vet"] == "artefact" for o in out), excess_max, sum(o["vet"] == "white-dwarf" for o in out),
                                            sum(o["vet"] == "natural-type" for o in out), len(left)))
    for o in sorted(out, key=lambda o: o["c"]):
        print("  Gaia DR3 %s: BP-RP %.3f, G %.2f, parallax %.1f mas, excess %+.3f, RUWE %s, WD probability %s; SIMBAD %s; %s" % (
            o["id"], o["c"], o["g"], o["plx"], o["ex"], o["ruwe"], o["wd"], o["simbad"], o["vet"] or "unexplained"))
    json.dump(out, open(core.results("bluest_nearby.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gcns_bluer_than_stars", "Nearby light bluer than any star", "Gaia Catalogue of Nearby Stars (external.gaiaedr3_gcns_main_1); SIMBAD",
            "stars within 100 pc with BP-RP below %g: %d" % (colour_max, len(out)), os.path.abspath(__file__), "bluest", len(out), "sources",
            dict(colour_max=colour_max, excess_max=excess_max), "the BP/RP flux excess, the image-peak and RUWE flags, the white-dwarf probability and SIMBAD's types",
            "a source bluer than the Rayleigh-Jeans limit of any photosphere in Gaia's bands",
            "%d: %d flux-excess or blended artefacts, %d white dwarfs or hot subdwarfs, %d other known types; %d unexplained" % (
                len(out), sum(o["vet"] == "artefact" for o in out), sum(o["vet"] == "white-dwarf" for o in out), sum(o["vet"] == "natural-type" for o in out), len(left)),
            [dict(target="gaia_dr3:%s" % o["id"], stat=o["c"], unit="BP-RP", vet=o["vet"],
                  note="G %.2f, parallax %.1f mas, BP/RP excess %+.3f, RUWE %s, WD probability %s; SIMBAD %s" % (o["g"], o["plx"], o["ex"], o["ruwe"], o["wd"], o["simbad"]))
             for o in out])
        print("recorded")


APPROACHES = {
    "bluest": (lambda: core.run_cli(bluest_main), bluest_DOC),
    "etzuv": (lambda: core.run_cli(etzuv_main), etzuv_DOC),
    "starcat": (starcat_cli, starcat_DOC),
    "cos": (cos_cli, cos_DOC),
    "iue": (iue_cli, iue_DOC),
    "fuse": (fuse_cli, fuse_DOC),
    "stis": (stis_cli, stis_DOC),
    "limits": (limits_cli, limits_DOC),
    "hst_pulses": (hst_pulses_cli, hst_pulses_DOC),
    "galex_pulses": (galex_pulses_cli, galex_pulses_DOC),
    "galex_census": (galex_census_cli, galex_census_DOC),
}


