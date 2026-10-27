"""X-ray, gamma-ray, neutrino and millimetre signatures of nearby stars."""
import argparse
import csv
import glob
import gzip
import io
import json
import math
import os
import sys
import tarfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from setitools import catalogues, core, science
from setitools.catalogues import cold_dwarfs, shifted_matches
from setitools.core import num, quiet_warnings, sep_arcsec


# --------------------------------------------------------------------------------------------------------------

sources_DOC = r"""eRASS1 and 4FGL-DR4 sources on stars within 20 pc: log Lx/Lbol above -2.5, hard spectra, or gamma-ray ellipses beyond the area expectation.

    python -I seti.py highenergy sources"""


sources_HERE = core.RESULTS
sources_EPOCH = 2020.3
sources_BP_RP = np.array([0.0, 0.5, 0.82, 1.2, 1.8, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
sources_BC_G = np.array([-0.1, 0.05, 0.08, -0.05, -0.35, -0.8, -1.1, -1.4, -1.8, -2.3, -2.8])
sources_LSUN = 3.828e33
sources_PC = 3.0857e18


def sources_xmatch(rows, cat, radius=30.0):
    """CDS XMatch of epoch-propagated star positions against a VizieR catalogue."""
    return core.xmatch(rows, cat, radius, ra_col="ra_ep", dec_col="dec_ep", timeout=600)


def sources_fnum(x):
    return num(x, float("nan"))


def sources_main():
    stars = list(csv.DictReader(open(os.path.join(sources_HERE, "near20.csv"))))
    for r in stars:
        dt = sources_EPOCH - 2016.0
        dec = float(r["dec"])
        r["ra_ep"] = float(r["ra"]) + sources_fnum(r["pmra"] or 0) / 3.6e6 * dt / math.cos(math.radians(dec))
        r["dec_ep"] = dec + sources_fnum(r["pmdec"] or 0) / 3.6e6 * dt
    byid = {r["source_id"]: r for r in stars}
    out = dict(xray=[], hard=[], gamma=[])
    for key, cat in (("xray", "J/A+A/682/A34/erass1-m"), ("hard", "J/A+A/682/A34/erass1-h")):
        m = sources_xmatch(stars, cat)
        for x in m:
            s = byid[x["source_id"]]
            sep = sources_fnum(x["angDist"])
            perr = sources_fnum(x.get("posErr")) if x.get("posErr") else sources_fnum(x.get("RADEErr"))
            if not (sep <= 3.0 * math.hypot(perr if np.isfinite(perr) else 5.0, 1.0)):
                continue
            plx = sources_fnum(s["parallax"])
            d = 1000.0 / plx * sources_PC
            fx = sources_fnum(x["MLFlux1"])
            lx = 4 * math.pi * d * d * fx if np.isfinite(fx) else float("nan")
            mg = sources_fnum(s["phot_g_mean_mag"]) + 5 * math.log10(plx / 100.0)
            col = sources_fnum(s["bp_rp"])
            bc = float(np.interp(col, sources_BP_RP, sources_BC_G)) if np.isfinite(col) else -1.0
            sources_lbol = sources_LSUN * 10 ** (-0.4 * (mg + bc - 4.74))
            soft = sum(sources_fnum(x.get("MLFluxP%d" % k)) for k in (1, 2)) if key == "xray" else float("nan")
            hard = sum(sources_fnum(x.get("MLFluxP%d" % k)) for k in (3, 4)) if key == "xray" else float("nan")
            hr = (hard - soft) / (hard + soft) if np.isfinite(soft) and np.isfinite(hard) and hard + soft > 0 else float("nan")
            out[key].append(dict(source_id=s["source_id"], sep=round(sep, 2), pos_err=perr, fx=fx, lx=lx,
                                 log_lx_lbol=round(math.log10(lx / sources_lbol), 2) if lx > 0 else None, hr=round(hr, 2) if np.isfinite(hr) else None,
                                 g=sources_fnum(s["phot_g_mean_mag"]), bp_rp=col, plx=plx, ruwe=sources_fnum(s["ruwe"]),
                                 nss=s["non_single_star"], det_like=sources_fnum(x.get("DetLike0")), ext=sources_fnum(x.get("EXT")),
                                 mjd=sources_fnum(x.get("MJD")), ra=sources_fnum(x.get("RA_ICRS")), dec=sources_fnum(x.get("DE_ICRS")),
                                 name=x.get("IAUName")))
        print("%s: %d stars within 20 pc matched in %s" % (key, len(out[key]), cat), flush=True)
    # unassociated 4FGL-DR4 sources whose 95 per cent ellipse holds a star
    path = os.path.join(sources_HERE, "gll_psc_v35.fit")
    if not os.path.exists(path):
        b = urllib.request.urlopen("https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/gll_psc_v35.fit",
                                   timeout=300).read()
        open(path, "wb").write(b)
    from astropy.io import fits
    with fits.open(path) as h:
        t = h["LAT_Point_Source_Catalog"].data
        cls = np.char.strip(np.asarray(t["CLASS1"]).astype(str))
        un = (cls == "") | (cls == "unk")
        ra, dec = np.asarray(t["RAJ2000"], float)[un], np.asarray(t["DEJ2000"], float)[un]
        a, b_, pa = (np.asarray(t["Conf_95_SemiMajor"], float)[un], np.asarray(t["Conf_95_SemiMinor"], float)[un],
                     np.asarray(t["Conf_95_PosAng"], float)[un])
        names = np.asarray(t["Source_Name"]).astype(str)[un]
    ok = np.isfinite(a) & np.isfinite(b_)
    area = float(np.sum(np.pi * a[ok] * b_[ok]))
    expected = len(stars) * area / 41252.96
    for s in stars:
        sra, sdec = s["ra_ep"], s["dec_ep"]
        dx = (sra - ra) * np.cos(np.radians(sdec))
        dy = sdec - dec
        near = np.nonzero(ok & (np.abs(dy) < 1.0) & (np.abs(dx) < 1.0))[0]
        for k in near:
            th = np.radians(pa[k])
            u = dx[k] * np.sin(th) + dy[k] * np.cos(th)
            v = dx[k] * np.cos(th) - dy[k] * np.sin(th)
            r2 = (u / a[k]) ** 2 + (v / b_[k]) ** 2
            if r2 <= 1.0:
                out["gamma"].append(dict(source_id=s["source_id"], fgl=names[k], ellipse_r=round(float(np.sqrt(r2)), 2),
                                         a95_deg=float(a[k]), g=sources_fnum(s["phot_g_mean_mag"]), bp_rp=sources_fnum(s["bp_rp"]),
                                         plx=sources_fnum(s["parallax"])))
    print("gamma: %d unassociated 4FGL-DR4 sources, ellipses cover %.1f deg2; %d stars inside an ellipse, %.1f expected by "
          "chance" % (int(un.sum()), area, len(out["gamma"]), expected))
    json.dump(out, open(os.path.join(sources_HERE, "highenergy_near.json"), "w"), indent=0)
    xr = [x for x in out["xray"] if x["log_lx_lbol"] is not None]
    v = np.array([x["log_lx_lbol"] for x in xr])
    print("X-ray: log Lx/Lbol median %.2f, %d above -3.0, %d above -2.5" % (np.median(v), int((v > -3).sum()), int((v > -2.5).sum())))
    hrs = np.array([x["hr"] for x in xr if x["hr"] is not None])
    print("hardness (1-5 keV vs 0.2-1 keV): median %.2f, %d of %d above 0" % (np.median(hrs), int((hrs > 0).sum()), len(hrs)))
    for x in sorted(xr, key=lambda x: -x["log_lx_lbol"])[:12]:
        print("  Gaia DR3 %s G %.2f BP-RP %.2f %.1f pc: log Lx/Lbol %+.2f, HR %s, sep %.1f arcsec (err %.1f), RUWE %.2f, "
              "NSS %s" % (x["source_id"], x["g"], x["bp_rp"], 1000 / x["plx"], x["log_lx_lbol"], x["hr"], x["sep"],
                          x["pos_err"], x["ruwe"], x["nss"]))
    print("hard-band (2.3-5 keV) detections of stars within 20 pc:")
    for x in sorted(out["hard"], key=lambda x: -x["fx"]):
        print("  Gaia DR3 %s G %.2f BP-RP %.2f %.1f pc sep %.1f arcsec, F(0.2-2.3) %.2e" % (
            x["source_id"], x["g"], x["bp_rp"], 1000 / x["plx"], x["sep"], x["fx"]))
    for x in out["gamma"]:
        print("  gamma: Gaia DR3 %s G %.2f %.1f pc inside %s (r = %.2f of the ellipse, a95 %.3f deg)"
              % (x["source_id"], x["g"], 1000 / x["plx"], x["fgl"], x["ellipse_r"], x["a95_deg"]))


def sources_lbol(s):
    plx = sources_fnum(s["parallax"])
    mg = sources_fnum(s["phot_g_mean_mag"]) + 5 * math.log10(plx / 100.0)
    col = sources_fnum(s["bp_rp"])
    bc = float(np.interp(col, sources_BP_RP, sources_BC_G)) if np.isfinite(col) else -1.0
    return sources_LSUN * 10 ** (-0.4 * (mg + bc - 4.74))


def sources_systems():
    """Each eRASS1 source with its stars within 3 sigma: summed L_bol, Lx/Lbol, hardness, and SIMBAD types above -2.5."""
    d = json.load(open(os.path.join(sources_HERE, "highenergy_near.json")))
    stars = {r["source_id"]: r for r in csv.DictReader(open(os.path.join(sources_HERE, "near20.csv")))}
    names = sorted({x["name"] for x in d["xray"]})
    meta = {}
    for k in range(0, len(names), 100):
        q = ("SELECT IAUName, posErr, DetLike0, EXT, MLFlux1, MJD FROM \"J/A+A/682/A34/erass1-m\" WHERE IAUName IN (%s)"
             % ",".join("'%s'" % n for n in names[k:k + 100]))
        for r in core.tap("vizier", q):
            meta[r["IAUName"]] = r
    groups = {}
    for x in d["xray"]:
        groups.setdefault(x["name"], []).append(x)
    out = []
    for name, xs in groups.items():
        m = meta.get(name, {})
        perr = sources_fnum(m.get("posErr"))
        lim = 3.0 * math.hypot(perr if np.isfinite(perr) else 5.0, 1.0)
        members = [x for x in xs if x["sep"] <= lim]
        if not members:
            continue
        lb = sum(sources_lbol(stars[x["source_id"]]) for x in members)
        lx = members[0]["lx"]
        best = min(members, key=lambda x: x["sep"])
        out.append(dict(name=name, members=[x["source_id"] for x in members], nearest=best["source_id"],
                        sep=best["sep"], pos_err=perr, log_lx_lbol=round(math.log10(lx / lb), 2) if lx > 0 else None,
                        hr=best["hr"], det_like=sources_fnum(m.get("DetLike0")), ext=sources_fnum(m.get("EXT")),
                        g=[stars[x["source_id"]]["phot_g_mean_mag"] for x in members],
                        dist_pc=round(1000.0 / sources_fnum(stars[best["source_id"]]["parallax"]), 1)))
    v = np.array([o["log_lx_lbol"] for o in out if o["log_lx_lbol"] is not None])
    print("%d X-ray sources on %d systems within 20 pc; log Lx/Lbol median %.2f, %d above -3.0, %d above -2.5"
          % (len(out), len(out), np.median(v), int((v > -3).sum()), int((v > -2.5).sum())))
    hi = sorted((o for o in out if (o["log_lx_lbol"] or -9) > -2.5), key=lambda o: -o["log_lx_lbol"])
    for o in hi:
        sb = core.simbad("SELECT b.main_id, b.otype, b.sp_type FROM basic b JOIN ident i ON i.oidref=b.oid "
                             "WHERE i.id='Gaia DR3 %s'" % o["nearest"])
        o["simbad"] = sb[0] if sb else {}
        print("  %s %s (%s, %s) %.1f pc: log Lx/Lbol %+.2f, HR %s, %d member(s), sep %.1f arcsec (err %.1f), "
              "DET_LIKE %.0f" % (o["name"], o["simbad"].get("main_id", o["nearest"]), o["simbad"].get("otype", "?"),
                                 o["simbad"].get("sp_type", "?"), o["dist_pc"], o["log_lx_lbol"], o["hr"],
                                 len(o["members"]), o["sep"], o["pos_err"], o["det_like"]))
    json.dump(out, open(os.path.join(sources_HERE, "highenergy_systems.json"), "w"), indent=0)


def sources_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "systems":
        sources_systems()
    else:
        sources_main()


# --------------------------------------------------------------------------------------------------------------

chandra_DOC = r"""Periodic X-rays (6.5-2000 s) from stars within 10 pc in Chandra event lists: Z^2 with dither and frame frequencies masked; FAP below 1e-3 repeated in a second observation.

    python -I seti.py highenergy chandra fetch
    python -I seti.py highenergy chandra search
    python -I seti.py highenergy chandra record"""

chandra_ROOT = os.environ.get("XRAY_ROOT", core.data("xray_beacons"))
chandra_HERE = core.RESULTS
chandra_CLI = "https://cda.cfa.harvard.edu/csccli/"
chandra_DITHER = {"ACIS": (1000.0, 707.0), "HRC": (1087.0, 768.6)}
chandra_FRAME = 3.24104
chandra_P_MIN = 6.5
chandra_P_MAX = 2000.0
chandra_OVERSAMPLE = 5
chandra_MATCH = 3.0


def chandra_cli_query(q):
    u = chandra_CLI + "getProperties?" + urllib.parse.urlencode(dict(query=" ".join(q.split()), coordFormat="decimal",
                                                               nullAppearance="NaN", version="cur", outputFormat="tsv"))
    t = core.get(u, timeout=300, retries=3).decode("latin-1")
    if "Error executing query" in t:
        raise RuntimeError(t[:300])
    lines = [ln for ln in t.splitlines() if ln and not ln.startswith("#")]
    head = [h.strip() for h in lines[0].split("\t")]
    return [dict(zip(head, [v.strip() for v in ln.split("\t")])) for ln in lines[1:]]


def chandra_star_at(s, mjd):
    yrs = (mjd - 51544.5) / 365.25
    ra, dec = float(s["ra"]), float(s["dec"])
    pmra, pmdec = float(s.get("pmra") or 0), float(s.get("pmdec") or 0)
    return ra + pmra / 3.6e6 * yrs / math.cos(math.radians(dec)), dec + pmdec / 3.6e6 * yrs


def chandra_detections(s):
    """Per-observation CSC detections on the star at their epoch."""
    pm = math.hypot(float(s.get("pmra") or 0), float(s.get("pmdec") or 0)) / 1000.0
    radius = (pm * 25 + 10) / 60.0
    q = ("SELECT m.name, o.obsid, o.obi, o.region_id, s.ra, s.dec, o.theta, o.instrument, o.gti_mjd_obs, o.livetime, "
         "o.src_cnts_aper_b, o.src_cnts_aper_w, o.dither_warning_flag FROM master_source m, master_stack_assoc a, observation_source o, "
         "stack_observation_assoc b, stack_source s WHERE dbo.cone_distance(m.ra,m.dec,%.8f,%.8f)<=%.6f AND "
         "(m.name = a.name) AND (s.detect_stack_id = a.detect_stack_id and s.region_id = a.region_id) AND "
         "(s.detect_stack_id = b.detect_stack_id and s.region_id = b.region_id) AND (o.obsid = b.obsid and "
         "o.obi = b.obi and o.region_id = b.region_id)" % (float(s["ra"]), float(s["dec"]), radius))
    try:
        rows = chandra_cli_query(q)
    except Exception as e:
        print(s["main_id"], "query failed:", str(e)[:100], flush=True)
        return []
    out = {}
    for r in rows:
        try:
            mjd, ra, dec = float(r["gti_mjd_obs"]), float(r["ra"]), float(r["dec"])
        except ValueError:
            continue
        sra, sdec = chandra_star_at(s, mjd)
        d = sep_arcsec(sra, sdec, ra, dec)
        if d <= chandra_MATCH:
            key = (r["obsid"], r["obi"], r["region_id"])
            out[key] = dict(star=s["main_id"], name=r["name"], obsid=r["obsid"], obi=r["obi"], region=r["region_id"],
                            ra=ra, dec=dec, theta=float(r["theta"]) if r["theta"] != "NaN" else 0.0,
                            instrument=r["instrument"], mjd=mjd, offset=d, dither_warning=r["dither_warning_flag"])
    return list(out.values())


def chandra_retrieve(det):
    os.makedirs(os.path.join(chandra_ROOT, "regevt"), exist_ok=True)
    pkg = "%s.%s.%s/regevt3/" % (det["obsid"], det["obi"], det["region"])
    try:
        lst = json.loads(core.get(chandra_CLI + "browse?" + urllib.parse.urlencode(dict(packageset=pkg, version="cur")),
                                 timeout=120, retries=3).decode("ascii"))
        fn = lst[0]["filename"]
    except Exception as e:
        return dict(det, file=None, error="browse: " + str(e)[:80])
    path = os.path.join(chandra_ROOT, "regevt", fn + ".gz")
    if not os.path.exists(path):
        try:
            b = core.get(chandra_CLI + "retrieveFile?" + urllib.parse.urlencode(dict(filetype="regevt3", filename=fn, version="cur")),
                        timeout=300, retries=3)
        except Exception as e:
            return dict(det, file=None, error="retrieve: " + str(e)[:80])
        open(path + ".tmp", "wb").write(b)
        os.replace(path + ".tmp", path)
    return dict(det, file=path)


def chandra_fetch():
    os.makedirs(chandra_ROOT, exist_ok=True)
    stars = core.simbad("SELECT main_id, ra, dec, pmra, pmdec, plx_value, otype FROM basic WHERE plx_value > 100 "
                            "AND otype NOT IN ('Pl', 'Pl?') AND ra IS NOT NULL")
    cache_path = os.path.join(chandra_ROOT, "detections_by_star.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [s for s in stars if s["main_id"] not in cache]
    print("%d objects within 10 pc, %d still to query" % (len(stars), len(todo)), flush=True)

    def one_star(s):
        return s["main_id"], chandra_detections(s)
    with ThreadPoolExecutor(int(os.environ.get("XRAY_THREADS", "3"))) as ex:
        for k, (name, ds) in enumerate(ex.map(one_star, todo)):
            cache[name] = ds
            if k % 20 == 19 or ds:
                json.dump(cache, open(cache_path, "w"))
                print("%d/%d queried; %s: %d detections" % (k + 1, len(todo), name, len(ds)), flush=True)
    json.dump(cache, open(cache_path, "w"))
    dets = [d for ds in cache.values() for d in ds]
    uniq = {}
    for d in dets:
        uniq.setdefault((d["obsid"], d["obi"], d["region"]), d)
    dets = list(uniq.values())
    print("%d objects within 10 pc; %d per-observation CSC detections on them, %d stars" % (
        len(stars), len(dets), len({d["star"] for d in dets})), flush=True)
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(chandra_retrieve, dets))
    json.dump(got, open(os.path.join(chandra_ROOT, "detections.json"), "w"), indent=1)
    print("%d region event files on disk" % sum(1 for g in got if g.get("file")))


def chandra_events(det):
    from astropy.io import fits
    with fits.open(det["file"]) as h:
        e = h["EVENTS"]
        d, hd = e.data, e.header
        cols = [c.name.lower() for c in e.columns]
        ix, iy = cols.index("x") + 1, cols.index("y") + 1
        crval = (hd["TCRVL%d" % ix], hd["TCRVL%d" % iy])
        crpix = (hd["TCRPX%d" % ix], hd["TCRPX%d" % iy])
        cdelt = (hd["TCDLT%d" % ix], hd["TCDLT%d" % iy])
        t = np.asarray(d["time"], float)
        x, y = np.asarray(d["x"], float), np.asarray(d["y"], float)
        # tangent-plane event offsets from the source (arcsec)
        dec0 = math.radians(crval[1])
        xi = (x - crpix[0]) * cdelt[0] * 3600
        eta = (y - crpix[1]) * cdelt[1] * 3600
        sx = (det["ra"] - crval[0]) * math.cos(dec0) * 3600
        sy = (det["dec"] - crval[1]) * 3600
        r = np.hypot(xi - sx, eta - sy)
        gti = [(float(a), float(b)) for a, b in zip(h["GTI"].data["START"], h["GTI"].data["STOP"])] if "GTI" in h else None
        return t, r, hd.get("MJDREF", 50814.0), hd.get("TIMEZERO", 0.0), gti


def chandra_barycentre(t, mjdref, det):
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u
    src = SkyCoord(det["ra"], det["dec"], unit="deg")
    grid = np.linspace(t.min(), t.max(), 50)
    tt = Time(mjdref + grid / 86400.0, format="mjd", scale="tt",
              location=EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 0 * u.m))
    ltt = tt.light_travel_time(src, kind="barycentric").to(u.s).value
    return t + np.interp(t, grid, ltt)


def chandra_zsearch(t, dither):
    tspan = t.max() - t.min()
    if tspan < 3 * chandra_P_MIN or len(t) < 20:
        return None
    fmin, fmax = max(1.0 / chandra_P_MAX, 2.0 / tspan), 1.0 / chandra_P_MIN
    df = 1.0 / (chandra_OVERSAMPLE * tspan)
    f = np.arange(fmin, fmax, df)
    # dither combinations and ACIS frame sidebands masked within two resolution elements, harmonics included
    inst = [k / p for p in dither for k in (0.5, 1.5, 2.5)]
    for k in (0, 1, 2):
        for a in range(-5, 6):
            for b in range(-5, 6):
                inst.append(abs(k / chandra_FRAME + a / dither[0] + b / dither[1]))
    inst = np.array(sorted({round(x, 9) for x in inst if x > 0}))
    mask = np.ones(len(f), bool)
    for m in (1, 2):
        j = np.searchsorted(inst, m * f).clip(1, len(inst) - 1)
        near = np.minimum(np.abs(m * f - inst[j - 1]), np.abs(m * f - inst[j]))
        mask &= near > 2.0 / tspan
    f = f[mask]
    tt = t - t.min()
    z1 = np.empty(len(f))
    z2 = np.empty(len(f))
    for i in range(0, len(f), 2000):
        ph = 2 * np.pi * np.outer(f[i:i + 2000], tt)
        c1, s1 = np.cos(ph).sum(1), np.sin(ph).sum(1)
        c2, s2 = np.cos(2 * ph).sum(1), np.sin(2 * ph).sum(1)
        z1[i:i + 2000] = 2.0 / len(t) * (c1 ** 2 + s1 ** 2)
        z2[i:i + 2000] = z1[i:i + 2000] + 2.0 / len(t) * (c2 ** 2 + s2 ** 2)
    ntrial = max(1, int(len(f) / chandra_OVERSAMPLE))
    # red-noise scaling: statistics divided where their running median exceeds the white-noise median
    anchors = np.unique(np.searchsorted(f, np.geomspace(f[0], f[-1], 120)).clip(0, len(f) - 1))
    for z, med0 in ((z1, 2 * math.log(2)), (z2, 3.357)):
        meds = []
        for j in anchors:
            lo = min(np.searchsorted(f, f[j] * 0.75), max(j - 50, 0))
            hi = max(np.searchsorted(f, f[j] * 1.25), min(j + 50, len(f)))
            meds.append(np.median(z[lo:hi]))
        local = np.interp(np.arange(len(f)), anchors, meds)
        z *= np.minimum(1.0, med0 / np.maximum(local, 1e-9))
    j1, j2 = int(np.argmax(z1)), int(np.argmax(z2))
    from scipy.stats import chi2
    p1 = chi2.sf(z1[j1], 2)
    p2 = chi2.sf(z2[j2], 4)
    fap1 = -np.expm1(ntrial * np.log1p(-min(p1, 1 - 1e-16)))
    fap2 = -np.expm1(ntrial * np.log1p(-min(p2, 1 - 1e-16)))
    return dict(n=len(t), span=tspan, ntrial=ntrial, f1=float(f[j1]), z1=float(z1[j1]), fap1=float(fap1),
                f2=float(f[j2]), z2=float(z2[j2]), fap2=float(fap2))


def chandra_one(det):
    if not det.get("file"):
        return None
    try:
        t, r, mjdref, tz, gti = chandra_events(det)
    except Exception as e:
        return dict(det, error="read: " + str(e)[:80])
    rad = 2.0 if det["theta"] < 3.0 else 4.0
    t = t[r <= rad]
    if len(t) < 20:
        return dict(det, n=len(t), result=None)
    tb = chandra_barycentre(t, mjdref, det)
    inst = "HRC" if "HRC" in det["instrument"].upper() else "ACIS"
    return dict(det, result=chandra_zsearch(np.sort(tb), chandra_DITHER[inst]))


def chandra_search():
    dets = json.load(open(os.path.join(chandra_ROOT, "detections.json")))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        res = [r for r in ex.map(chandra_one, dets) if r]
    ok = [r for r in res if r.get("result")]
    print("%d detections searched (%d stars), %d with fewer than 20 source events; median %d events over %.0f ks"
          % (len(ok), len({r["star"] for r in ok}), sum(1 for r in res if not r.get("result")),
             np.median([r["result"]["n"] for r in ok]), np.median([r["result"]["span"] for r in ok]) / 1e3))
    faps = np.array([min(r["result"]["fap1"], r["result"]["fap2"]) for r in ok])
    print("best false-alarm probability per detection: below 0.01 in %d, below 1e-3 in %d (%.1f and %.1f expected)"
          % (np.sum(faps < 0.01), np.sum(faps < 1e-3), 0.02 * len(ok), 0.002 * len(ok)))
    hits = sorted([r for r in ok if min(r["result"]["fap1"], r["result"]["fap2"]) < 1e-3],
                  key=lambda r: min(r["result"]["fap1"], r["result"]["fap2"]))
    for r in hits:
        z = r["result"]
        best = (z["f1"], z["fap1"], "Z1") if z["fap1"] <= z["fap2"] else (z["f2"], z["fap2"], "Z2")
        # the same period in another observation
        again = []
        for o in ok:
            if o["star"] == r["star"] and o["obsid"] != r["obsid"]:
                res_f = 1.0 / min(o["result"]["span"], z["span"])
                for key in ("f1", "f2"):
                    if abs(o["result"][key] - best[0]) < res_f:
                        again.append(o["obsid"])
        print("  %-22s obsid %s %s theta %.1f': %d events over %.1f ks, %s peak at %.2f s, FAP %.1e; same period in %s"
              % (r["star"], r["obsid"], r["instrument"], r["theta"], z["n"], z["span"] / 1e3, best[2], 1.0 / best[0],
                 best[1], ", ".join(sorted(set(again))) or "no other observation"))
    json.dump(res, open(os.path.join(chandra_HERE, "xray_beacons.json"), "w"), indent=1)


def chandra_record():
    res = [r for r in json.load(open(os.path.join(chandra_HERE, "xray_beacons.json"))) if r.get("result")]
    best = {}
    for r in res:
        fap = min(r["result"]["fap1"], r["result"]["fap2"])
        if r["star"] not in best or fap < best[r["star"]][0]:
            best[r["star"]] = (fap, r)
    rows = [dict(target="simbad:" + " ".join(k.split()), p=v[0], note="best of %d observations: obsid %s, %d events, "
                 "period %.2f s" % (sum(1 for r in res if r["star"] == k), v[1]["obsid"], v[1]["result"]["n"],
                                    1.0 / v[1]["result"]["f1"])) for k, v in best.items()]
    core.record_search("xray_beacons_csc", "X-ray event lists", archive="Chandra Source Catalog (csccli, regevt3)",
                           release="CSC current", selection="SIMBAD objects with parallax above 100 mas",
                           code=os.path.abspath(__file__), entry="chandra", params=dict(p_min=chandra_P_MIN, p_max=chandra_P_MAX, match=chandra_MATCH),
                           n_targets=len(rows), result="%d detections searched; none below false-alarm probability 1e-3 after the dither, frame-sideband and red-noise filters" % len(res), ledger="X-ray event lists")
    core.record_targets("xray_beacons_csc", rows)
    print("recorded %d stars" % len(rows))


def chandra_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "search", "record"))
    a = p.parse_args()
    {"fetch": chandra_fetch, "search": chandra_search, "record": chandra_record}[a.mode]()


def chandra_cli():
    quiet_warnings()
    chandra_main()


# --------------------------------------------------------------------------------------------------------------

xmm_DOC = r"""Periodic X-rays (5-600 s) in XMM-Newton EPIC-pn series of stars within 20 pc: red-noise-normalised FFT with harmonics, FAP below 1e-3 in both halves.

    python -I seti.py highenergy xmm fetch
    python -I seti.py highenergy xmm search
    python -I seti.py highenergy xmm record"""

xmm_ROOT = os.environ.get("XMM_ROOT", core.data("xmm"))
xmm_HERE = core.RESULTS
xmm_AIO = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=%s&name=SRCTSR&level=PPS&instname=PN"
xmm_DT = 0.5
xmm_P_MIN, xmm_P_MAX = 5.0, 600.0
xmm_MJDREF = 50814.0
xmm_GZIP_MAGIC = bytes([0x1F, 0x8B])


def xmm_detections():
    path = os.path.join(xmm_ROOT, "detections.json")
    if not os.path.exists(path):
        q = ("SELECT obs_id, pps_srcnum, srcid, iauname, ra, dec, gaiadr3_source_id, gaiadr3_parallax, gaiadr3_gmag, "
             "pn_cts, pn_ontime, pn_submode, pn_rate, m1_cts, m2_cts, ep_cts, tseries, var_flag, var_prob, fvar, "
             "pn_pileup, sum_flag, mjd_first, mjd_last FROM xsa.v_epic_source_cat WHERE gaiadr3_parallax > 50 AND "
             "ep_cts > 300")
        json.dump(core.xsa(q), open(path, "w"))
    rows = json.load(open(path)) + xmm_nearest()
    seen, out = set(), []
    for r in rows:
        key = (r["obs_id"], r["pps_srcnum"])
        if r["obs_id"] and r["tseries"] == "true" and r["pps_srcnum"] and key not in seen:
            seen.add(key)
            out.append(r)
    return out


xmm_COLS = ("obs_id, pps_srcnum, srcid, iauname, ra, dec, gaiadr3_source_id, gaiadr3_parallax, gaiadr3_gmag, pn_cts, "
        "pn_ontime, pn_submode, pn_rate, m1_cts, m2_cts, ep_cts, tseries, var_flag, var_prob, fvar, pn_pileup, "
        "sum_flag, mjd_first, mjd_last")


def xmm_nearest():
    """Detections of stars within 10 pc missed by the catalogue's Gaia match: SIMBAD parallax above 100 mas, within 6 arcsec, over 300 counts."""
    path = os.path.join(xmm_ROOT, "nearest.json")
    if os.path.exists(path):
        return json.load(open(path))
    from setitools.core import mjd_to_year, sep_arcsec, star_at
    stars = core.simbad("SELECT main_id, ra, dec, pmra, pmdec, plx_value FROM basic WHERE plx_value > 100 AND "
                            "ra IS NOT NULL AND otype NOT IN ('Pl', 'Pl?')")
    out = []
    for s in stars:
        ra, dec = float(s["ra"]), float(s["dec"])
        pm = math.hypot(float(s["pmra"] or 0), float(s["pmdec"] or 0)) / 1000.0
        rad = (pm * 26 + 15) / 3600.0
        q = ("SELECT %s FROM xsa.v_epic_source_cat WHERE ep_cts > 300 AND 1 = CONTAINS(POINT('ICRS', ra, dec), "
             "CIRCLE('ICRS', %.7f, %.7f, %.6f))" % (xmm_COLS, ra, dec, rad))
        try:
            rows = core.xsa(q)
        except Exception:
            continue
        for r in rows:
            if not r["obs_id"] or not r["mjd_first"]:
                continue
            at = mjd_to_year(float(r["mjd_first"]))
            sra, sdec = star_at(ra, dec, float(s["pmra"] or 0), float(s["pmdec"] or 0), 2000.0, at)
            if sep_arcsec(sra, sdec, float(r["ra"]), float(r["dec"])) <= 6.0:
                out.append(dict(r, star=s["main_id"], plx=float(s["plx_value"])))
    json.dump(out, open(path, "w"))
    return out


def xmm_fetch():
    os.makedirs(os.path.join(xmm_ROOT, "tar"), exist_ok=True)
    obs = sorted({d["obs_id"] for d in xmm_detections()})
    for o in obs:
        p = os.path.join(xmm_ROOT, "tar", o + ".tar")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        ok = core.download(xmm_AIO % o, p, retries=3) and tarfile.is_tarfile(p)
        print(o, "fetched" if ok else "failed", flush=True)


def xmm_series(obs):
    """Every source's time series in an observation tar: {source: header, times, rates, exposure, background}."""
    out = {}
    p = os.path.join(xmm_ROOT, "tar", obs + ".tar")
    if not os.path.exists(p):
        return out
    from astropy.io import fits
    with tarfile.open(p) as tf:
        best = {}
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if "SRCTSR8" not in base:
                continue
            src = int(base.split("SRCTSR8")[1][:3], 16)
            expo = base[13:17]
            # combined X000 series first, else the longest exposure
            rank = (expo == "X000", m.size)
            if src not in best or rank > best[src][0]:
                best[src] = (rank, m)
        for src, (_, m) in best.items():
            raw = tf.extractfile(m).read()
            if raw[:2] == xmm_GZIP_MAGIC:
                raw = gzip.decompress(raw)
            with fits.open(io.BytesIO(raw)) as h:
                d = h["RATE"].data
                out[src] = dict(header=dict(h["RATE"].header), time=np.asarray(d["TIME"], float),
                                rate=np.asarray(d["RATE"], float), fracexp=np.asarray(d["FRACEXP"], float),
                                back=np.asarray(d["BACKV"], float) if "BACKV" in d.columns.names else None)
    return out


def xmm_power(t, x, t0, n):
    """Normalised FFT power on a regular grid: unit exponential under white noise."""
    from scipy.ndimage import median_filter
    g = np.zeros(n)
    k = np.clip(((t - t0) / xmm_DT).astype(int), 0, n - 1)
    np.add.at(g, k, x)
    filled = np.zeros(n)
    np.add.at(filled, k, 1.0)
    g = np.where(filled > 0, g / np.maximum(filled, 1), 0.0)
    g[filled > 0] -= g[filled > 0].mean()
    p = np.abs(np.fft.rfft(g)) ** 2
    f = np.fft.rfftfreq(n, xmm_DT)
    # running median 1/50 dex wide (at least 15 frequencies), interpolated in log-log to follow red noise
    m = len(f)
    anchors = np.unique(np.concatenate([np.arange(1, min(m, 100)),
                                        np.round(np.geomspace(1, m - 1, 2000)).astype(int)]))
    meds = []
    for i in anchors:
        w = max(7, int(0.0236 * i))
        meds.append(np.median(p[max(1, i - w):min(m, i + w + 1)]))
    med = np.empty(m)
    med[1:] = 10 ** np.interp(np.log10(np.arange(1, m)), np.log10(anchors), np.log10(np.maximum(meds, 1e-30)))
    med[0] = med[1]
    if m > 401:
        # local median kept for narrow lines
        med = np.maximum(med, median_filter(p, size=401, mode="nearest"))
    return f, p / np.maximum(med, 1e-30) * math.log(2), filled


def xmm_analyse(s, ra, dec):
    ok = (s["fracexp"] > 0.5) & np.isfinite(s["rate"])
    if ok.sum() < 1000:
        return None
    t = science.barycentre_seconds(s["time"][ok], xmm_MJDREF, ra, dec)
    t0 = t.min()
    span = t.max() - t0
    if span < 3 * xmm_P_MAX:
        return None
    n = int(span / xmm_DT) + 1
    f, pw, filled = xmm_power(t, s["rate"][ok], t0, n)
    _, pwin, _ = xmm_power(np.arange(n) * xmm_DT + t0, filled - filled.mean(), t0, n)
    pb = xmm_power(t, s["back"][ok], t0, n)[1] if s["back"] is not None else np.zeros_like(pw)
    # coarse native bins and their harmonics masked; search stops at 2.5 native bins
    dt_native = float(np.median(np.diff(np.sort(s["time"][ok]))))
    fmax = min(1 / xmm_P_MIN, 1 / (2.5 * dt_native))
    clean = (pwin < 5 * math.log(2) * 5) & (pb < 5 * math.log(2) * 5)
    df = 1.0 / span
    for k in range(1, int(f[-1] * dt_native) + 2):
        clean &= np.abs(f - k / dt_native) > 2 * df
    # periods must fit twenty times into the series
    band = (f >= max(1 / xmm_P_MAX, 20.0 / span)) & (f <= fmax)
    mask = band & clean
    # second harmonic summed where clean
    idx = np.arange(len(f))
    two = np.clip(2 * idx, 0, len(f) - 1)
    p2 = np.where((f[two] <= fmax) & clean[two], pw + pw[two], pw)
    nfreq = int(mask.sum())
    if nfreq < 10:
        return None
    j1 = int(np.argmax(np.where(mask, pw, -1)))
    j2 = int(np.argmax(np.where(mask, p2, -1)))
    fap1 = 1 - (1 - math.exp(-pw[j1])) ** nfreq
    # gamma(2) survival: (1 + x) e^-x
    fap2 = 1 - (1 - (1 + p2[j2]) * math.exp(-p2[j2])) ** nfreq
    halves = []
    for lo, hi in ((t0, t0 + span / 2), (t0 + span / 2, t0 + span)):
        sel = (t >= lo) & (t < hi)
        if sel.sum() < 500:
            halves.append(None)
            continue
        nh = int((hi - lo) / xmm_DT) + 1
        fh, ph, _ = xmm_power(t[sel], s["rate"][ok][sel], lo, nh)
        jj = int(np.argmin(np.abs(fh - f[j1])))
        halves.append(float(ph[max(jj - 1, 0):jj + 2].max()))
    return dict(span=span, nfreq=nfreq, f1=float(f[j1]), p1=float(pw[j1]), fap1=fap1, f2=float(f[j2]), p2=float(p2[j2]),
                fap2=fap2, halves=halves, df=1.0 / span, bin_s=dt_native)


def xmm_search():
    dets = xmm_detections()
    by = {}
    for d in dets:
        by.setdefault(d["obs_id"], []).append(d)
    res = []
    for obs, ds in sorted(by.items()):
        ss = xmm_series(obs)
        if not ss:
            print(obs, "no time series", flush=True)
            continue
        # other sources' best peaks for the shared-frequency veto
        others = {}
        for src, s in ss.items():
            ra = float(s["header"].get("SRC_RA", ds[0]["ra"]))
            dec = float(s["header"].get("SRC_DEC", ds[0]["dec"]))
            r = xmm_analyse(s, ra, dec)
            if r:
                others[src] = r
        for d in ds:
            src = int(d["pps_srcnum"])
            r = others.get(src)
            if not r:
                print(obs, src, "too short or too sparse", flush=True)
                continue
            shared = [o for o, x in others.items() if o != src and
                      (abs(x["f1"] - r["f1"]) < 2 * r["df"] or abs(x["f2"] - r["f2"]) < 2 * r["df"])]
            row = dict(d, **r, shared=shared)
            res.append(row)
            print("%s %s src %d: %.0f ks, best %.2f s FAP %.2g (two harmonics %.2f s FAP %.2g), halves %s%s"
                  % (obs, d["iauname"], src, r["span"] / 1000, 1 / r["f1"], r["fap1"], 1 / r["f2"], r["fap2"],
                     r["halves"], ", shared with sources %s" % shared if shared else ""), flush=True)
    json.dump(res, open(os.path.join(xmm_HERE, "xmm_beacons.json"), "w"), default=float, indent=1)
    n = len(res)
    for thr in (0.01, 1e-3):
        k = sum(1 for r in res if min(r["fap1"], r["fap2"]) < thr)
        print("observations of stars with a best peak below FAP %g: %d of %d (%.1f expected for two statistics)"
              % (thr, k, n, 2 * thr * n))
    for r in res:
        if min(r["fap1"], r["fap2"]) < 1e-3:
            both = all(h is not None and h > 5 for h in r["halves"])
            print("  candidate %s %s: %.2f s, FAP %.2g, both halves %s, shared %s"
                  % (r["obs_id"], r["iauname"], 1 / r["f1"], r["fap1"], both, r["shared"]))


def xmm_record():
    """Record one row per detection with its lowest false-alarm probability."""
    res = json.load(open(os.path.join(xmm_HERE, "xmm_beacons.json")))
    n = len(res)
    below = [sum(1 for r in res if min(r["fap1"], r["fap2"]) < thr) for thr in (1e-2, 1e-3)]
    core.record_search(
        "xray_beacons_xmm", "X-ray event lists", archive="XMM-Newton Science Archive, 5XMM-DR15 EPIC-pn source time "
        "series (SRCTSR)", release="5XMM-DR15",
        selection="detections with more than 300 EPIC counts of Gaia DR3 stars with parallax above 50 mas, and of SIMBAD "
                  "objects within 10 pc at their epoch positions", code="seti.py highenergy xmm",
        params=dict(dt_s=xmm_DT, p_min_s=xmm_P_MIN, p_max_s=xmm_P_MAX, min_cycles=20, window="1/50 dex, 15 frequencies"),
        n_targets=n,
        null_summary="%d observations of stars: %d below false-alarm probability 0.01 against %.1f, %d below 1e-3 against "
                     "%.1f" % (n, below[0], 2 * 0.01 * n, below[1], 2 * 1e-3 * n),
        sensitivity="a coherent period of 5-600 s in the EPIC-pn source rate, alone or with its second harmonic",
        result="no period beyond the noise" if not below[1] else "%d periods below 1e-3" % below[1],
        ledger="X-ray event lists")
    core.clear_targets("xray_beacons_xmm")
    rows = []
    for r in res:
        g = r.get("gaiadr3_source_id")
        tgt = "gaia_dr3:%s" % g if g not in (None, "", "None") else "simbad:%s" % r.get("star", r["iauname"])
        rows.append(dict(target=tgt, stat=float(1 / r["f1"]), unit="best period, s", p=float(min(r["fap1"], r["fap2"])),
                         note="%s obs %s, %.0f ks" % (r["iauname"], r["obs_id"], r["span"] / 1000)))
    core.record_targets("xray_beacons_xmm", rows)
    print("recorded")


def xmm_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "search", "record"))
    a = p.parse_args()
    if a.mode == "fetch":
        xmm_fetch()
    elif a.mode == "record":
        xmm_record()
    else:
        xmm_search()


def xmm_cli():
    quiet_warnings()
    xmm_main()


# --------------------------------------------------------------------------------------------------------------

icecube_DOC = r"""TeV neutrinos from systems within 10 pc in IceCube's ten-year sample: unbinned likelihood against RA scrambles; E^-2 flux and luminosity limits.

    python -I seti.py highenergy icecube [--pc 10] [--trials 1000]"""

icecube_ROOT = os.environ.get("ICECUBE_ROOT", core.data("icecube"))
icecube_HERE = core.RESULTS
icecube_R6 = np.radians(6.0)


def icecube_load_events():
    out = []
    for p in sorted(glob.glob(os.path.join(icecube_ROOT, "events", "*.tab"))):
        a = np.loadtxt(p, comments="#")
        out.append(a)
    a = np.vstack(out)
    return dict(mjd=a[:, 0], logE=a[:, 1], sig=np.radians(np.maximum(a[:, 2], 0.2)), ra=np.radians(a[:, 3]),
                dec=np.radians(a[:, 4]))


def icecube_unit(ra, dec):
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def icecube_background_density(dec):
    """Background density per steradian at each event's declination."""
    s = np.sin(dec)
    h, e = np.histogram(s, bins=100, range=(-1, 1))
    from scipy.ndimage import gaussian_filter1d
    h = gaussian_filter1d(h.astype(float), 1.0)
    dens = h / h.sum() / (e[1] - e[0]) / (2 * np.pi)
    idx = np.clip(np.digitize(s, e) - 1, 0, 99)
    return dens, e, dens[idx]


class icecube_Field:
    def __init__(self, ev):
        from scipy.spatial import cKDTree
        self.ev = ev
        self.N = len(ev["ra"])
        self.xyz = icecube_unit(ev["ra"], ev["dec"])
        self.tree = cKDTree(self.xyz)
        self.dens, self.edges, self.B = icecube_background_density(ev["dec"])

    def ts(self, ra, dec, want_limit=False):
        """TS, fitted ns and optional 90 per cent limit at one position."""
        c = icecube_unit(np.array(ra), np.array(dec))
        idx = np.array(self.tree.query_ball_point(c, 2 * np.sin(icecube_R6 / 2)), dtype=int)
        if len(idx) == 0:
            return 0.0, 0.0, None
        cosd = np.clip(self.xyz[idx] @ c, -1, 1)
        psi = np.arccos(cosd)
        sg = self.ev["sig"][idx]
        S = np.exp(-psi ** 2 / (2 * sg ** 2)) / (2 * np.pi * sg ** 2)
        return icecube_fit(S / self.B[idx] - 1.0, self.N, want_limit)


def icecube_fit(x, N, want_limit=False):
    """TS, ns and optional 90 per cent ns limit from near-event signal-to-background ratios among N events."""
    n_near = len(x)

    def llh(ns):
        return np.sum(np.log1p(ns / N * x)) + (N - n_near) * np.log1p(-ns / N)

    # Newton iterations on the concave log-likelihood, ns >= 0
    ns = 0.0
    g0 = np.sum(x) / N - (N - n_near) / N
    if g0 > 0:
        ns = 1.0
        for _ in range(50):
            d = 1 + ns / N * x
            g = np.sum(x / d) / N - (N - n_near) / (N - ns)
            h = -np.sum((x / d) ** 2) / N ** 2 - (N - n_near) / (N - ns) ** 2
            step = -g / h
            ns = max(ns + step, 0.0)
            if abs(step) < 1e-4:
                break
    L0, L1 = llh(0.0), llh(ns)
    ts = max(2 * (L1 - L0), 0.0)
    lim = None
    if want_limit:
        hi = max(ns, 1.0)
        while 2 * (L1 - llh(hi)) < 2.71 and hi < 5000:
            hi *= 1.5
        lo = ns
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if 2 * (L1 - llh(mid)) < 2.71:
                lo = mid
            else:
                hi = mid
        lim = 0.5 * (lo + hi)
    return ts, ns, lim


def icecube_acceptance(dec_deg):
    """Events per unit E^-2 normalisation (1 TeV-1 PeV) summed over seasons."""
    tot = 0.0
    seasons = ["IC40", "IC59", "IC79", "IC86_I", "IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI", "IC86_VII"]
    for s in seasons:
        ae = os.path.join(icecube_ROOT, "irfs", "%s_effectiveArea.tab" % (s if s in ("IC40", "IC59", "IC79", "IC86_I") else "IC86_II"))
        up = os.path.join(icecube_ROOT, "uptime", "%s_exp.tab" % s)
        if not (os.path.exists(ae) and os.path.exists(up)):
            continue
        a = np.loadtxt(ae, comments="#")
        u = np.loadtxt(up, comments="#")
        live = float(np.sum(u[:, 1] - u[:, 0])) * 86400.0
        sel = (a[:, 2] <= dec_deg) & (dec_deg < a[:, 3])
        e_lo, e_hi = 10 ** a[sel, 0] / 1e3, 10 ** a[sel, 1] / 1e3  # TeV
        keep = (e_hi > 1.0) & (e_lo < 1000.0)
        e_lo, e_hi = np.clip(e_lo[keep], 1.0, 1000.0), np.clip(e_hi[keep], 1.0, 1000.0)
        # E^-2 integral per energy bin (TeV^-1)
        tot += float(np.sum(a[sel][keep, 4] * (1 / e_lo - 1 / e_hi))) * live
    return tot


def icecube_systems(pc):
    rows = list(csv.DictReader(open(os.path.join(icecube_HERE, "near20.csv"), encoding="utf-8")))
    rows = [r for r in rows if float(r["parallax"]) >= 1000.0 / pc]
    rows.sort(key=lambda r: -float(r["parallax"]))
    out = []
    for r in rows:
        ra, dec = np.radians(float(r["ra"])), np.radians(float(r["dec"]))
        for s in out:
            if np.arccos(np.clip(icecube_unit(ra, dec) @ icecube_unit(s["ra"], s["dec"]), -1, 1)) < np.radians(1.0):
                s["members"].append(r["source_id"])
                break
        else:
            out.append(dict(ra=ra, dec=dec, d_pc=1000.0 / float(r["parallax"]), members=[r["source_id"]],
                            g=float(r["phot_g_mean_mag"] or 99)))
    return out


def icecube_main():
    p = argparse.ArgumentParser()
    p.add_argument("--pc", type=float, default=10.0)
    p.add_argument("--trials", type=int, default=1000)
    a = p.parse_args()
    ev = icecube_load_events()
    f = icecube_Field(ev)
    syst = icecube_systems(a.pc)
    print("%d events; %d systems within %.0f pc" % (f.N, len(syst), a.pc))
    rng = np.random.default_rng(11)
    res = []
    for s in syst:
        ts, ns, lim = f.ts(s["ra"], s["dec"], want_limit=True)
        null = []
        while len(null) < a.trials:
            ra = rng.uniform(0, 2 * np.pi)
            dra = (ra - s["ra"] + np.pi) % (2 * np.pi) - np.pi
            if abs(dra) * np.cos(s["dec"]) < np.radians(6.0):
                continue
            null.append(f.ts(ra, s["dec"])[0])
        null = np.array(null)
        pv = float((np.sum(null >= ts) + 1) / (len(null) + 1))
        acc = icecube_acceptance(np.degrees(s["dec"]))
        flux = lim / acc if acc > 0 else float("nan")  # TeV^-1 cm^-2 s^-1 at 1 TeV
        # energy flux over 1 TeV-1 PeV for E^-2
        eflux = flux * np.log(1000.0) * 1.602  # erg cm^-2 s^-1
        lum = 4 * np.pi * (s["d_pc"] * 3.0857e18) ** 2 * eflux  # erg s^-1
        res.append(dict(members=s["members"], ra=float(np.degrees(s["ra"])), dec=float(np.degrees(s["dec"])),
                        d_pc=s["d_pc"], ts=ts, ns=ns, ns90=lim, p=pv, flux90=flux, lum90_erg_s=lum))
    # stacked TS against RA-rotated catalogues
    stack = sum(r["ts"] for r in res)
    rot = []
    for _ in range(a.trials):
        d = rng.uniform(np.radians(10), 2 * np.pi - np.radians(10))
        rot.append(sum(f.ts((s["ra"] + d) % (2 * np.pi), s["dec"])[0] for s in syst))
    rot = np.array(rot)
    p_stack = float((np.sum(rot >= stack) + 1) / (len(rot) + 1))
    json.dump(dict(pc=a.pc, n_events=f.N, systems=res, stack_ts=stack, p_stack=p_stack),
              open(os.path.join(icecube_HERE, "icecube_stars_%gpc.json" % a.pc), "w"))
    ps = np.array([r["p"] for r in res])
    print("stacked TS %.2f, p %.3f against %d catalogue rotations" % (stack, p_stack, a.trials))
    print("systems at p <= 0.01: %d (%.1f expected), p <= 0.05: %d (%.1f expected)"
          % ((ps <= 0.01).sum(), 0.01 * len(ps), (ps <= 0.05).sum(), 0.05 * len(ps)))
    for r in sorted(res, key=lambda r: r["p"])[:10]:
        print("  Gaia DR3 %s (%.2f pc, dec %+.1f): TS %.2f, ns %.1f, p %.4f; ns90 %.1f, E^-2 flux90 %.2e /TeV/cm2/s, "
              "L90 %.1e erg/s" % (r["members"][0], r["d_pc"], r["dec"], r["ts"], r["ns"], r["p"], r["ns90"],
                                   r["flux90"], r["lum90_erg_s"]))
    lums = np.array([r["lum90_erg_s"] for r in res])
    print("median 90 per cent luminosity limit %.1e erg/s (range %.1e-%.1e)" % (np.median(lums), lums.min(), lums.max()))


def icecube_cli():
    quiet_warnings()
    icecube_main()


# --------------------------------------------------------------------------------------------------------------

planets_DOC = r"""TeV neutrinos tracking Jupiter, Saturn, Mars and Venus in IceCube's sample: planet-frame likelihood against time scrambles; flux and luminosity limits.

    python -I seti.py highenergy planets [--trials 1000]"""

planets_BODIES = ("jupiter", "saturn", "mars", "venus")


def planets_track(body, mjd_lo, mjd_hi):
    """Hourly RA, Dec (rad) and distance (AU) of a body between two MJDs."""
    from astropy.time import Time
    from astropy.coordinates import get_body, solar_system_ephemeris
    grid = np.arange(math.floor(mjd_lo) - 1, math.ceil(mjd_hi) + 1, 1 / 24.0)
    with solar_system_ephemeris.set("builtin"):
        c = get_body(body, Time(grid, format="mjd"))
    return grid, np.radians(c.ra.deg), np.radians(c.dec.deg), c.distance.au


def planets_at(track, mjd):
    """Track RA and Dec at times mjd, RA interpolated on the unit circle."""
    grid, ra, dec, _ = track
    x = np.interp(mjd, grid, np.cos(ra))
    y = np.interp(mjd, grid, np.sin(ra))
    return np.arctan2(y, x) % (2 * np.pi), np.interp(mjd, grid, dec)


def planets_ts(ev, B, ra_s, dec_s, want_limit=False):
    """TS, ns and 90 per cent limit for a moving source using each event's time-resolved distance."""
    cosd = np.sin(ev["dec"]) * np.sin(dec_s) + np.cos(ev["dec"]) * np.cos(dec_s) * np.cos(ev["ra"] - ra_s)
    psi = np.arccos(np.clip(cosd, -1, 1))
    idx = np.where(psi < icecube_R6)[0]
    if len(idx) == 0:
        return 0.0, 0.0, None
    sg = ev["sig"][idx]
    S = np.exp(-psi[idx] ** 2 / (2 * sg ** 2)) / (2 * np.pi * sg ** 2)
    return icecube_fit(S / B[idx] - 1.0, len(psi), want_limit)


def planets_main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1000)
    a = p.parse_args()
    ev = icecube_load_events()
    _, _, B = icecube_background_density(ev["dec"])
    mjd = ev["mjd"]
    sun = planets_track("sun", mjd.min(), mjd.max())
    rng = np.random.default_rng(13)
    res = {}
    print("%d events, %.1f years" % (len(mjd), (mjd.max() - mjd.min()) / 365.25), flush=True)
    for body in planets_BODIES:
        tr = planets_track(body, mjd.min(), mjd.max())
        ra_s, dec_s = planets_at(tr, mjd)
        ra_sun, dec_sun = planets_at(sun, mjd)
        elong = np.arccos(np.clip(np.sin(dec_s) * np.sin(dec_sun) + np.cos(dec_s) * np.cos(dec_sun) * np.cos(ra_s - ra_sun), -1, 1))
        keep = elong > np.radians(5.0)
        evk = {k: v[keep] for k, v in ev.items()}
        Bk = B[keep]
        ts, ns, lim = planets_ts(evk, Bk, ra_s[keep], dec_s[keep], want_limit=True)
        null = []
        for _ in range(a.trials):
            r2, d2 = planets_at(tr, rng.permutation(evk["mjd"]))
            null.append(planets_ts(evk, Bk, r2, d2)[0])
        null = np.array(null)
        pv = float((np.sum(null >= ts) + 1) / (len(null) + 1))
        # acceptance averaged along the track every ten days
        decs = np.degrees(tr[2][:: 24 * 10])
        acc = float(np.mean([icecube_acceptance(d) for d in decs]))
        flux = lim / acc if acc > 0 else float("nan")                    # TeV^-1 cm^-2 s^-1 at 1 TeV
        eflux = flux * np.log(1000.0) * 1.602                               # erg cm^-2 s^-1 over 1 TeV-1 PeV
        d_cm = float(np.mean(tr[3])) * 1.496e13
        lum = 4 * np.pi * d_cm ** 2 * eflux
        res[body] = dict(n_events=int(keep.sum()), ts=ts, ns=ns, ns90=lim, p=pv, null_median=float(np.median(null)), null_max=float(null.max()),
                         acceptance=acc, flux90=flux, eflux90_erg_cm2_s=eflux, lum90_erg_s=lum, mean_distance_au=float(np.mean(tr[3])))
        print("%s: %d events away from the Sun, TS %.2f, ns %.1f, p %.3f against %d time scrambles (null median %.2f, max %.2f); ns90 %.1f, E^-2 flux90 %.2e /TeV/cm2/s, "
              "energy flux %.1e erg/cm2/s, L90 %.1e erg/s (%.1e W) at %.1f AU" % (body, keep.sum(), ts, ns, pv, a.trials, np.median(null), null.max(), lim, flux, eflux, lum, lum * 1e-7, np.mean(tr[3])), flush=True)
    json.dump(dict(trials=a.trials, n_events=len(mjd), bodies=res), open(core.results("icecube_planets.json"), "w"), indent=1)


def planets_record():
    d = json.load(open(core.results("icecube_planets.json")))
    b = d["bodies"]
    core.record_search(
        "icecube_planets", "TeV neutrino sources moving with the planets", archive="IceCube ten-year point-source sample (2008-2018), %d tracks" % d["n_events"], release="IceCube 2021 release",
        selection="every track, with those taken while the planet lay within 5 deg of the Sun set aside; Jupiter, Saturn, Mars and Venus", code=os.path.abspath(__file__), entry="planets",
        n_targets=4, params=dict(trials=d["trials"], radius_deg=6.0, sun_exclusion_deg=5.0),
        null_summary="%d time scrambles per planet, the events fixed on the sky and the planet moved under them" % d["trials"],
        sensitivity="90 per cent E^-2 energy-flux limits over 1 TeV-1 PeV: " + ", ".join("%s %.1e erg/cm2/s" % (k, v["eflux90_erg_cm2_s"]) for k, v in b.items()),
        ledger="TeV neutrino sources moving with the planets",
        result="; ".join("%s TS %.2f p %.2f (ns %.1f, L90 %.1e W)" % (k, v["ts"], v["p"], v["ns"], v["lum90_erg_s"] * 1e-7) for k, v in b.items()))
    core.clear_targets("icecube_planets")
    core.record_targets("icecube_planets", [dict(target="planet:%s" % k, stat=v["ts"], unit="point-source TS in the planet's frame", p=v["p"],
                                                    note="ns %.1f, ns90 %.1f, L90 %.1e W at %.1f AU" % (v["ns"], v["ns90"], v["lum90_erg_s"] * 1e-7, v["mean_distance_au"])) for k, v in b.items()])
    print("recorded 4 planets")


def planets_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        planets_record()
    else:
        planets_main()


# --------------------------------------------------------------------------------------------------------------

lineonly_DOC = r"""Line-only X-ray sources in eRASS:3: 85 per cent of counts in 0.5-1 or 1-2 keV and every other band below 5 per cent at 3 sigma.

    python -I seti.py highenergy lineonly analyse
    python -I seti.py highenergy lineonly record"""

lineonly_ROOT = os.environ.get("LINEONLY_ROOT", core.data("erosita"))
lineonly_OUT = "erosita_line_only.json"
lineonly_BANDS = (("P1", "0.2-0.5"), ("P2", "0.5-1"), ("P3", "1-2"), ("P4", "2-5"))
lineonly_CTS, lineonly_FRAC, lineonly_OTHER = 80.0, 0.85, 0.05


def lineonly_analyse():
    from astropy.table import Table
    t = Table.read(os.path.join(lineonly_ROOT, "erass3_main.ecsv"), format="ascii.ecsv")
    R = np.column_stack([np.array(t["MLRate%s" % b], float) for b, _ in lineonly_BANDS])
    Elo = np.column_stack([np.array(t["e_MLRate%s" % b], float) for b, _ in lineonly_BANDS])
    Ehi = np.column_stack([np.array(t["E_MLRate%s" % b], float) for b, _ in lineonly_BANDS])
    C = np.column_stack([np.array(t["MLcts%s" % b], float) for b, _ in lineonly_BANDS])
    ext = np.array(t["Ext"], float)
    R, Elo, Ehi, C = np.nan_to_num(R), np.nan_to_num(Elo), np.nan_to_num(Ehi), np.nan_to_num(C)
    Elo, Ehi = np.abs(Elo), np.abs(Ehi)
    tot = R.sum(axis=1)
    cts = C.sum(axis=1)
    point = (np.nan_to_num(ext) == 0) & (cts >= lineonly_CTS) & (tot > 0)
    print("%d sources, %d point sources with %.0f or more counts in the four bands" % (len(t), point.sum(), lineonly_CTS), flush=True)
    F = R / np.maximum(tot, 1e-12)[:, None]
    Flo = np.clip((R - 3 * Elo), 0, None) / np.maximum(tot, 1e-12)[:, None]
    Fhi = (R + 3 * Ehi) / np.maximum(tot, 1e-12)[:, None]
    out = {b: [] for b, _ in lineonly_BANDS}
    for i in np.where(point)[0]:
        k = int(np.argmax(F[i]))
        others = [j for j in range(4) if j != k]
        if Flo[i, k] >= lineonly_FRAC and all(Fhi[i, j] <= lineonly_OTHER for j in others):
            out[lineonly_BANDS[k][0]].append(dict(name=str(t["IAUName"][i]).strip(), ra=float(t["RA_ICRS"][i]), dec=float(t["DE_ICRS"][i]), glat=float(t["GLAT"][i]),
                                                  counts=float(cts[i]), fraction=float(F[i, k]), fraction_lo=float(Flo[i, k]), others_hi=float(max(Fhi[i, j] for j in others)),
                                                  detlike=float(t["DetLike0"][i]), band=lineonly_BANDS[k][1], rates=[float(x) for x in R[i]]))
    maxf = F[point].max(axis=1)
    print("band-fraction maxima among the point sources: median %.2f, 90th %.2f, 99th %.2f; single-band sources at %.0f per cent (3 sigma) with the rest below %.0f per cent: %s" % (
        np.median(maxf), np.percentile(maxf, 90), np.percentile(maxf, 99), 100 * lineonly_FRAC, 100 * lineonly_OTHER, {lineonly_BANDS[k][1] + " keV": len(out[lineonly_BANDS[k][0]]) for k in range(4)}), flush=True)
    # middle-band candidates named through SIMBAD
    middle = out["P2"] + out["P3"]
    for x in sorted(middle, key=lambda x: -x["counts"]):
        try:
            s = core.simbad("SELECT main_id, otype, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', %f, %f))*3600 AS sep FROM basic WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %f, %f, 0.005)) = 1 ORDER BY sep" % (x["ra"], x["dec"], x["ra"], x["dec"]))
            x["simbad"] = "; ".join("%s (%s, %.0f arcsec)" % (r["main_id"], r["otype"], float(r["sep"])) for r in s[:3]) or "none within 18 arcsec"
        except Exception as e:
            x["simbad"] = "query failed: %s" % str(e)[:40]
        print("  %s (%.4f %+.4f, b %+.1f): %.0f counts, %.0f per cent in %s keV (lower %.2f, others below %.2f), rates %s; %s" % (
            x["name"], x["ra"], x["dec"], x["glat"], x["counts"], 100 * x["fraction"], x["band"], x["fraction_lo"], x["others_hi"], ["%.3f" % r for r in x["rates"]], x["simbad"]), flush=True)
    json.dump(dict(n=len(t), n_point=int(point.sum()), counts={k: len(v) for k, v in out.items()}, maxf_percentiles=[float(np.percentile(maxf, p)) for p in (50, 90, 99)],
                   middle=middle, soft=out["P1"][:200], hard=out["P4"][:200], params=dict(cts=lineonly_CTS, frac=lineonly_FRAC, other=lineonly_OTHER)),
              open(core.results(lineonly_OUT), "w"), indent=0)


def lineonly_record():
    d = json.load(open(core.results(lineonly_OUT)))
    core.record_search(
        "erosita_line_only", "Line-only X-ray sources in eROSITA DR1", archive="eROSITA DR2 eRASS:3 main catalogue (VizieR J/A+A/712/A171)", release="eROSITA-DE DR2",
        selection="%d point sources with %.0f or more counts in 0.2-5 keV of %d" % (d["n_point"], d["params"]["cts"], d["n"]), code=os.path.abspath(__file__), entry="lineonly",
        n_targets=d["n_point"], params=d["params"], null_summary="the band-fraction maximum over the point sources: median %.2f, 90th percentile %.2f, 99th %.2f" % tuple(d["maxf_percentiles"]),
        sensitivity="%.0f per cent of the counts in one band at 3 sigma with every other band below %.0f per cent" % (100 * d["params"]["frac"], 100 * d["params"]["other"]), ledger="Line-only X-ray sources in eROSITA DR1",
        result="single-band sources: %s; the middle-band ones: %s" % (", ".join("%d in %s" % (v, k) for k, v in d["counts"].items()),
                                                                       "; ".join("%s (%s keV, %.0f counts: %s)" % (x["name"], x["band"], x["counts"], x.get("simbad", "")) for x in d["middle"][:10]) or "none"))
    core.clear_targets("erosita_line_only")
    core.record_targets("erosita_line_only", [dict(target="erosita:%s" % x["name"], stat=x["fraction"], unit="fraction of the counts in the %s keV band" % x["band"], vet=x.get("simbad"),
                                                       note="%.0f counts, others below %.2f" % (x["counts"], x["others_hi"])) for x in d["middle"]])
    print("recorded %d middle-band sources" % len(d["middle"]))


def lineonly_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": lineonly_analyse, "record": lineonly_record}[mode]()


# --------------------------------------------------------------------------------------------------------------

mm_DOC = r"""SPT-3G mm-flaring stars without X-ray or chromospheric activity, checked in eRASS1, 4XMM, CSC, Gaia, GALEX and SIMBAD.

    python -I seti.py highenergy mm check
    python -I seti.py highenergy mm record"""

mm_ROOT = os.environ.get("MMFLARE_ROOT", core.data("mm_flares"))
mm_HERE = core.RESULTS
mm_CAT = "https://pole.uchicago.edu/public/data/tandoi24/spt3g_2019_2022_flare_star_catalog.csv"


def mm_stars():
    rows = list(csv.DictReader(open(os.path.join(mm_ROOT, "spt3g_flare_catalog.csv"), encoding="utf-8")))
    u = {}
    for r in rows:
        u.setdefault(r["dr3_source_id"], []).append(r)
    return u


def mm_check():
    u = mm_stars()
    ids = list(u)
    g = {r["source_id"]: r for r in core.gaia(
        "SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax, g.ruwe, g.phot_g_mean_mag, g.bp_rp, "
        "g.non_single_star, a.activityindex_espcs, a.activityindex_espcs_uncertainty, a.teff_gspphot, a.logg_gspphot, "
        "a.classprob_dsc_combmod_star FROM gaiadr3.gaia_source AS g LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS a "
        "ON g.source_id = a.source_id WHERE g.source_id IN (%s)" % ",".join(ids))}
    # positions at 2020.5 for eROSITA-era catalogues, 2016.0 otherwise
    pos = [dict(source_id=k, ra=float(g[k]["ra"]), dec=float(g[k]["dec"])) for k in ids if k in g]
    xr = {}
    for cat, rad, name in (("J/A+A/682/A34/erass1-m", 15.0, "eRASS1"), ("IX/69/xmm4d13s", 10.0, "4XMM-DR13"),
                           ("IX/70/csc21mas", 5.0, "CSC 2.1"), ("II/335/gal_ais", 5.0, "GALEX")):
        try:
            m = core.xmatch(pos, cat, rad)
        except Exception as e:
            print("%s: %s" % (name, str(e)[:80]), flush=True)
            m = []
        for r in m:
            xr.setdefault(r["source_id"], {}).setdefault(name, []).append(r)
        print("%s: %d of %d stars matched within %.0f arcsec" % (name, len({r["source_id"] for r in m}), len(pos), rad),
              flush=True)
    sim = {}
    for r in core.xmatch(pos, "simbad", 3.0):
        sim.setdefault(r["source_id"], r)
    res = []
    for k in ids:
        f = u[k]
        x = g.get(k, {})
        xm = xr.get(k, {})
        s = sim.get(k, {})
        plx = num(x.get("parallax")) or num(f[0]["parallax"])
        d = 1000.0 / plx if plx and plx > 0 else None
        fx = None
        if "eRASS1" in xm:
            e = min(xm["eRASS1"], key=lambda r: float(r["angDist"]))
            fx = num(e.get("Flux") or e.get("ML_FLUX_1") or e.get("FluxP"))
        nuv = None
        if "GALEX" in xm:
            gx = min(xm["GALEX"], key=lambda r: float(r["angDist"]))
            nuv = num(gx.get("NUVmag"))
        gm = num(x.get("phot_g_mean_mag"))
        bprp = num(x.get("bp_rp"))
        res.append(dict(
            source_id=k, flares=len(f), s150_max=max(num(r["150_flux"]) or 0 for r in f), d_pc=d, g=gm, bp_rp=bprp,
            abs_g=gm + 5 * math.log10(plx / 100.0) if gm and plx and plx > 0 else None, ruwe=num(x.get("ruwe")),
            nss=x.get("non_single_star"), act=num(x.get("activityindex_espcs")),
            act_err=num(x.get("activityindex_espcs_uncertainty")), teff=num(x.get("teff_gspphot")),
            rosat=bool(f[0]["2rxs_id"]), erass1=fx, erass1_n=len(xm.get("eRASS1", [])), xmm=len(xm.get("4XMM-DR13", [])),
            csc=len(xm.get("CSC 2.1", [])), nuv=nuv, nuv_g=(nuv - gm) if nuv is not None and gm else None,
            spt_binary=f[0]["binary"] == "True", simbad=s.get("main_id"), otype=s.get("main_type") or s.get("otype"),
            sptype=s.get("sp_type")))
    json.dump(res, open(os.path.join(mm_HERE, "mm_flares.json"), "w"), indent=1)
    quiet = [r for r in res if not r["rosat"] and not r["erass1_n"] and not r["xmm"] and not r["csc"]]
    print("%d flaring stars; %d without 2RXS; %d without any X-ray source (eRASS1, 4XMM, CSC as well)"
          % (len(res), sum(not r["rosat"] for r in res), len(quiet)))
    for r in sorted(res, key=lambda r: (r["rosat"] or bool(r["erass1_n"]), r["bp_rp"] or 0)):
        if r["rosat"]:
            continue
        print("  Gaia DR3 %s %s %-12s %s: G %.2f BP-RP %.2f M_G %s d %s pc RUWE %s; %d flare(s), S150 %.0f mJy; eRASS1 %s, "
              "4XMM %d, CSC %d; Ca IRT index %s; NUV-G %s; SPT binary flag %s; SIMBAD %s"
              % (r["source_id"], (r["simbad"] or "")[:22], (r["otype"] or "")[:12], (r["sptype"] or "")[:8], r["g"] or 0,
                 r["bp_rp"] or 0, "%.2f" % r["abs_g"] if r["abs_g"] is not None else "-",
                 "%.0f" % r["d_pc"] if r["d_pc"] else "-", "%.2f" % r["ruwe"] if r["ruwe"] else "-", r["flares"],
                 r["s150_max"], "%.2e" % r["erass1"] if r["erass1"] else ("match" if r["erass1_n"] else "none"),
                 r["xmm"], r["csc"], "%.3f +- %.3f" % (r["act"], r["act_err"] or 0) if r["act"] is not None else "-",
                 "%.2f" % r["nuv_g"] if r["nuv_g"] is not None else "-", r["spt_binary"], r["simbad"] is not None))


def mm_record():
    res = json.load(open(os.path.join(mm_HERE, "mm_flares.json")))
    rows = []
    for r in res:
        if r["rosat"] or r["erass1_n"] or r["xmm"] or r["csc"]:
            vet = "X-ray source (%s)" % ", ".join(n for n, v in (("2RXS", r["rosat"]), ("eRASS1", r["erass1_n"]),
                                                                ("4XMM", r["xmm"]), ("CSC", r["csc"])) if v)
        else:
            # eRASS1 limit about 5e-14 erg/s/cm^2 in 0.2-2.3 keV
            lim = 4 * math.pi * (r["d_pc"] * 3.086e18) ** 2 * 5e-14 if r["d_pc"] else None
            vet = "no X-ray source; %s %s dwarf with emission lines, eRASS1 limit log Lx %.1f above an active ultracool " \
                  "dwarf's" % (r["sptype"] or "", "ultracool" if (r["bp_rp"] or 0) > 3.5 else "late-M",
                               math.log10(lim) if lim else float("nan"))
        rows.append(dict(target="gaia_dr3:" + r["source_id"], stat=r["s150_max"], unit="mJy at 150 GHz, brightest flare",
                         vet=vet, note="%d SPT-3G flare(s); %s" % (r["flares"], r["simbad"] or "")))
    core.record_search("spt3g_quiet_flares", "Millimetre flares on inactive stars",
                           archive="SPT-3G flare star catalogue (Tandoi et al. 2024); eRASS1, 4XMM-DR13, CSC 2.1, Gaia DR3",
                           release="2019-2022", selection="the 66 Gaia DR3 stars of the 111 SPT-3G flares",
                           code=os.path.abspath(__file__), entry="mm", n_targets=len(rows),
                           result="every flaring star has an X-ray source except four M5-M8 emission-line dwarfs at "
                                  "37-49 pc, below eRASS1's limit", ledger="Millimetre transients")
    core.record_targets("spt3g_quiet_flares", rows)
    print("recorded %d stars" % len(rows))


def mm_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("check", "record"))
    a = p.parse_args()
    if a.mode == "check":
        if not os.path.exists(os.path.join(mm_ROOT, "spt3g_flare_catalog.csv")):
            os.makedirs(mm_ROOT, exist_ok=True)
            open(os.path.join(mm_ROOT, "spt3g_flare_catalog.csv"), "wb").write(core.get(mm_CAT, timeout=60, retries=3))
        mm_check()
    else:
        mm_record()


def mm_cli():
    quiet_warnings()
    mm_main()


# --------------------------------------------------------------------------------------------------------------

doublets_DOC = r"""Cosmic-ray multiplets in Auger data: pairs and triplets within 2 deg and days, against sidereal-preserving time scrambles; matched to stars within 100 pc.

    python -I seti.py highenergy doublets [record] [--set phase1|sd1500|sd750|inclined]"""

doublets_URL = "https://zenodo.org/api/records/6759610/files/Anisotropy_%20ApJS_2022.zip/content"
doublets_MEMBER = "Anisotropy_ ApJS_2022/Data/AugerApJS2022_Yr_JD_UTC_Th_Ph_RA_Dec_E_Expo.dat"


doublets_OPEN = "https://zenodo.org/records/10488964/files/summary.zip?download=1"
doublets_OPEN_FILES = {"sd1500": "summary/dataSummarySD1500.csv", "sd750": "summary/dataSummarySD750.csv",
                       "inclined": "summary/dataSummaryInclined.csv"}


def doublets_events(which="phase1"):
    if which != "phase1":
        import zipfile
        p = core.data("auger", "opendata_summary.zip")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(core.get(doublets_OPEN, timeout=600))
        rows = list(csv.DictReader(io.StringIO(zipfile.ZipFile(p).read(doublets_OPEN_FILES[which]).decode("utf-8"))))
        rows = [r for r in rows if num(r["sd_ra"]) is not None and num(r["sd_dec"]) is not None and num(r["gpstime"])]
        t = np.array([float(r["gpstime"]) for r in rows]) / 86400.0 + 44244.0 - 40587.0  # GPS epoch as Unix days
        ra = np.radians([float(r["sd_ra"]) for r in rows])
        dec = np.radians([float(r["sd_dec"]) for r in rows])
        e = np.array([num(r["sd_energy"]) or 0.0 for r in rows])
        xyz = np.vstack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).T
        return t, xyz, e, np.degrees(ra), np.degrees(dec)
    p = core.data("auger", "AugerApJS2022_events.dat")
    if not os.path.exists(p):
        import zipfile
        os.makedirs(os.path.dirname(p), exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(core.get(doublets_URL, timeout=600)))
        open(p, "wb").write(z.read(doublets_MEMBER))
    rows = [ln.split() for ln in open(p) if ln.strip() and not ln.startswith("#")]
    t = np.array([float(r[2]) for r in rows]) / 86400.0
    ra = np.radians([float(r[5]) for r in rows])
    dec = np.radians([float(r[6]) for r in rows])
    e = np.array([float(r[7]) for r in rows])
    xyz = np.vstack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).T
    return t, xyz, e, np.degrees(ra), np.degrees(dec)


def doublets_count(t, close, dt):
    """Angular pairs closer than dt days."""
    return int(np.sum(np.abs(t[close[:, 0]] - t[close[:, 1]]) < dt))


def doublets_triplets(t, xyz, nbr, dt, rad):
    """Triplets within dt days and rad of their mean direction."""
    n = 0
    found = []
    for i, js in enumerate(nbr):
        js = [j for j in js if j > i and abs(t[j] - t[i]) < dt]
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                j, k = js[a], js[b]
                if abs(t[j] - t[k]) >= dt:
                    continue
                m = xyz[i] + xyz[j] + xyz[k]
                m /= np.linalg.norm(m)
                if min(m @ xyz[i], m @ xyz[j], m @ xyz[k]) > math.cos(rad):
                    n += 1
                    found.append((i, j, k))
    return n, found


def doublets_main(record=False, n_perm=10000, which="phase1"):
    from scipy.spatial import cKDTree
    t, xyz, e, ra, dec = doublets_events(which)
    tree = cKDTree(xyz)
    rad = math.radians(2.0)
    chord = 2 * math.sin(rad / 2)
    close = np.array(sorted(tree.query_pairs(chord)))
    nbr = tree.query_ball_point(xyz, 2 * math.sin(rad))
    obs = dict(pairs10=doublets_count(t, close, 10.0), pairs1=doublets_count(t, close, 1.0))
    obs["triplets30"], trip = doublets_triplets(t, xyz, nbr, 30.0, rad)
    rng = np.random.default_rng(1)
    null = {k: [] for k in obs}
    sid = 0.99726957
    for k in range(n_perm):
        # events moved by whole sidereal days, keeping exposure and freeing time
        tq = rng.permutation(t)
        tp = t + np.round((tq - t) / sid) * sid
        null["pairs10"].append(doublets_count(tp, close, 10.0))
        null["pairs1"].append(doublets_count(tp, close, 1.0))
        if k < 1000:
            null["triplets30"].append(doublets_triplets(tp, xyz, nbr, 30.0, rad)[0])
    p = {k: float((np.sum(np.array(null[k]) >= obs[k]) + 1) / (len(null[k]) + 1)) for k in obs}
    print("%d events (%s), %d angular pairs within 2 deg" % (len(t), which, len(close)))
    for k in obs:
        print("  %-11s observed %3d, sidereal-shifted mean %.2f (95th percentile %.0f), p %.4f"
              % (k, obs[k], np.mean(null[k]), np.percentile(null[k], 95), p[k]))
    mult = []
    for i, j in close:
        if abs(t[i] - t[j]) < 10:
            mult.append(dict(kind="pair", idx=[int(i), int(j)], dt_days=float(abs(t[i] - t[j]))))
    for i, j, k in trip:
        mult.append(dict(kind="triplet", idx=[int(i), int(j), int(k)], dt_days=float(max(t[[i, j, k]]) - min(t[[i, j, k]]))))
    # stars within 100 pc in each 2 deg circle; with many multiplets only same-day pairs and triplets
    many = len(mult) > 60
    for m in mult:
        c = xyz[m["idx"]].mean(axis=0)
        c /= np.linalg.norm(c)
        m["ra"], m["dec"] = math.degrees(math.atan2(c[1], c[0])) % 360, math.degrees(math.asin(c[2]))
        m["energies"] = [float(e[i]) for i in m["idx"]]
        m["dates_mjd"] = [float(t[i] + 40587.0) for i in m["idx"]]
        # only Phase One is matched; lower energies deflect too far
        m["stars"] = (doublets_stars(m["ra"], m["dec"])[:5] if which == "phase1" and (not many or m["kind"] == "triplet"
                                                                                        or m["dt_days"] < 1) else [])
        print("  %s at RA %.1f Dec %+.1f, %.2f d apart, %s EeV, MJD %s; stars within 100 pc in 2 deg: %s"
              % (m["kind"], m["ra"], m["dec"], m["dt_days"], ", ".join("%.0f" % x for x in m["energies"]),
                 ", ".join("%.1f" % x for x in m["dates_mjd"]),
                 "; ".join("Gaia DR3 %s at %.1f pc" % (s[1], s[0]) for s in m["stars"]) or "none"))
    out = dict(n=len(t), n_angular_pairs=len(close), observed=obs, p=p, n_perm=n_perm,
               null_mean={k: float(np.mean(v)) for k, v in null.items()}, multiplets=mult)
    out["set"] = which
    json.dump(out, open(core.results("auger_doublets%s.json" % ("" if which == "phase1" else "_" + which)), "w"), indent=1)
    if record:
        doublets_record(out)


def doublets_stars(ra, dec):
    """Gaia DR3 stars within 100 pc inside 2 deg, nearest first: (pc, source_id)."""
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    q = ("SELECT source_id, parallax FROM gaiadr3.gaia_source WHERE parallax > 10 AND parallax_over_error > 10 AND "
         "1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.5f, %.5f, 2.0))" % (ra, dec))
    for attempt in range(3):
        try:
            t = Gaia.launch_job(q).get_results()
            return sorted((round(1000.0 / float(r["parallax"]), 1), str(r["source_id"])) for r in t)
        except Exception:
            pass
    return []


def doublets_record(out):
    which = out.get("set", "phase1")
    rid = "auger_doublets" if which == "phase1" else "auger_doublets_" + which
    core.record_search(
        rid, "Cosmic-ray doublets",
        archive=("Pierre Auger Observatory Phase One events above 32 EeV" if which == "phase1" else
                 "Pierre Auger Observatory open data 2024, %s" % {"sd1500": "1500 m array", "sd750": "750 m array",
                                                                   "inclined": "inclined showers"}[which]),
        release=("Zenodo 10.5281/zenodo.6504276 (2004-2020)" if which == "phase1" else "Zenodo 10.5281/zenodo.10488964"),
        selection="all %d events" % out["n"],
        code=os.path.abspath(__file__), entry="doublets", n_targets=out["n"],
        params=dict(radius_deg=2, pair_days=[1, 10], triplet_days=30, permutations=out.get("n_perm"),
                    null="whole sidereal days to another event's date"),
        null_summary="; ".join("%s %d against a sidereal-shifted mean of %.2f (p %.4f)" % (k, out["observed"][k], out["null_mean"][k], out["p"][k])
                               for k in out["observed"]),
        sensitivity="two events above 32 EeV within 2 deg in 10 days: about 6e-9 J/m^2 over the array",
        ledger="Cosmic-ray doublets" if which == "phase1" else "Cosmic-ray doublets, further sets",
        result="pairs within 2 deg: %d within 10 days and %d within a day, triplets within 30 days %d, against sidereal-shifted "
               "means of %.1f, %.1f and %.2f" % (out["observed"]["pairs10"], out["observed"]["pairs1"], out["observed"]["triplets30"],
                                                 out["null_mean"]["pairs10"], out["null_mean"]["pairs1"], out["null_mean"]["triplets30"]))
    core.clear_targets(rid)
    core.record_targets(rid, [dict(
        target="sky:ra %.2f dec %+.2f" % (m["ra"], m["dec"]), stat=m["dt_days"], unit="days between members",
        note="%s, %s EeV, MJD %s; stars within 100 pc in 2 deg: %s"
             % (m["kind"], ", ".join("%.0f" % x for x in m["energies"]), ", ".join("%.2f" % x for x in m["dates_mjd"]),
                ", ".join("Gaia DR3 %s" % s[1] for s in m["stars"]) or "none")) for m in out["multiplets"]])
    print("recorded")


def doublets_cli():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", choices=("search", "record"), default="search")
    p.add_argument("--set", default="phase1", choices=("phase1", "sd1500", "sd750", "inclined"))
    p.add_argument("--perm", type=int, default=10000)
    a = p.parse_args()
    doublets_main(record=a.mode == "record", n_perm=a.perm, which=a.set)


dynamo_DOC = r"""Activity without a dynamo: eRASS1 X-rays above the rotation-activity relation at Ro above 1, and V-LoTSS circular polarisation at inactive slow rotators.

    python -I seti.py highenergy dynamo xray [record]
    python -I seti.py highenergy dynamo polarised [record]"""

dynamo_ROOT = os.environ.get("DYNAMO_ROOT", core.data("dynamo_dead"))
dynamo_OUT = "dynamo_dead.json"
dynamo_VIZIER = {"hamstar": "J/A+A/684/A121", "vlotss": "J/A+A/670/A124", "fetherolf": "J/ApJS/268/4", "ztf": "J/ApJS/249/18"}
dynamo_ERASS_LIMIT = 5e-14  # erg/s/cm^2, eRASS1 limit 0.2-2.3 keV
dynamo_RESID = 1.5
# dwarf colour-temperature (Pecaut and Mamajek 2013) and bolometric corrections (Andrae et al. 2018)
dynamo_BPRP_TEFF = ((0.4, 7200), (0.5, 6900), (0.65, 6400), (0.82, 5800), (1.0, 5300), (1.2, 4900), (1.5, 4400), (2.0, 3900), (2.5, 3500), (3.0, 3200), (3.5, 3000), (4.0, 2800))
dynamo_TEFF_BC = ((7200, 0.10), (6500, 0.09), (6000, 0.08), (5800, 0.07), (5500, 0.03), (5000, -0.10), (4500, -0.30), (4000, -0.60), (3500, -1.20), (3000, -2.00), (2800, -2.40))


def dynamo_teff(bprp):
    x, y = zip(*dynamo_BPRP_TEFF)
    return float(np.interp(bprp, x, y))


def dynamo_bc(teff):
    x, y = zip(*sorted(dynamo_TEFF_BC))
    return float(np.interp(teff, x, y))


def dynamo_tauc(teff):
    """Convective turnover time (d), Cranmer and Saar (2011)."""
    return 314.24 * math.exp(-teff / 1952.5 - (teff / 6250.0) ** 18) + 0.002


def dynamo_expected(ro):
    """Expected L_X/L_bol: 10^-3.13 below Ro 0.13, Ro^-2.5 above."""
    return 10 ** -3.13 if ro <= 0.13 else 10 ** -3.13 * (ro / 0.13) ** -2.5


def dynamo_lbol(G, bprp, plx):
    """L_bol (L_sun) from G, BP-RP and parallax (mas)."""
    teff = dynamo_teff(bprp)
    MG = G + 5 * math.log10(plx) - 10
    Mbol = MG + dynamo_bc(teff)
    return 10 ** (0.4 * (4.74 - Mbol)), teff


def dynamo_vizier(key):
    """VizieR tables of a catalogue, cached as ECSV."""
    from astropy.table import Table
    paths = sorted(glob.glob(os.path.join(dynamo_ROOT, "%s_*.ecsv" % key)))
    if not paths:
        os.makedirs(dynamo_ROOT, exist_ok=True)
        for i, t in enumerate(core.vizier(dynamo_VIZIER[key])):
            t.write(os.path.join(dynamo_ROOT, "%s_%d.ecsv" % (key, i)), format="ascii.ecsv", overwrite=True)
        paths = sorted(glob.glob(os.path.join(dynamo_ROOT, "%s_*.ecsv" % key)))
    return [Table.read(p, format="ascii.ecsv") for p in paths]


def dynamo_rotation():
    """Rotation periods by Gaia source and by position (TESS, ZTF)."""
    by_id, pos = {}, []
    p = os.path.join(dynamo_ROOT, "gaia_rotation.csv")
    if not os.path.exists(p):
        try:
            rows = core.tap("gaia", "SELECT source_id, best_rotation_period, best_rotation_period_error, num_segments FROM gaiadr3.vari_rotation_modulation",
                           timeout=1800, maxrec=600000)
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        except Exception as e:
            print("Gaia rotation table not fetched (%s); TESS and ZTF periods only" % str(e)[:80])
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if num(r["best_rotation_period"]):
                by_id[int(r["source_id"])] = (num(r["best_rotation_period"]), "Gaia DR3")
    t = dynamo_vizier("fetherolf")[0]
    for r in t:
        if r["Pvar"] == r["Pvar"] and float(r["Pvar"]) > 0:
            pos.append((float(r["RAJ2000"]), float(r["DEJ2000"]), float(r["Pvar"]), "TESS (Fetherolf et al. 2023)"))
    zp = sorted(glob.glob(os.path.join(dynamo_ROOT, "ztf_*.ecsv")))
    zf = core.data("ztf_var", "ztf_periodic.fits")
    if not zp and os.path.exists(zf):
        zp = [zf]
    if zp:
        from astropy.table import Table
        for path in zp:
            t = Table.read(path, format="ascii.ecsv") if path.endswith(".ecsv") else Table.read(path)
            cols = {c.lower(): c for c in t.colnames}
            ck, ra_c, de_c, ty = cols.get("per"), cols.get("raj2000"), cols.get("dej2000"), cols.get("type")
            if not (ck and ra_c and de_c):
                continue
            for r in t:
                typ = str(r[ty]) if ty else ""
                # rotational variables only (BY Dra, RS CVn)
                if typ and typ.upper() not in ("BYDRA", "RSCVN", "BY DRA", "RS CVN", "ROT"):
                    continue
                if r[ck] == r[ck] and float(r[ck]) > 0:
                    pos.append((float(r[ra_c]), float(r[de_c]), float(r[ck]), "ZTF (Chen et al. 2020, %s)" % typ))
    return by_id, pos


def dynamo_match_pos(pos, ra, dec, radius=3.0):
    """Nearest position-indexed period within radius arcsec."""
    if not hasattr(dynamo_match_pos, "tree") or dynamo_match_pos.n != len(pos):
        from scipy.spatial import cKDTree
        a = np.radians([[p[0], p[1]] for p in pos])
        xyz = np.column_stack([np.cos(a[:, 1]) * np.cos(a[:, 0]), np.cos(a[:, 1]) * np.sin(a[:, 0]), np.sin(a[:, 1])])
        dynamo_match_pos.tree, dynamo_match_pos.n = cKDTree(xyz), len(pos)
    v = np.array([math.cos(math.radians(dec)) * math.cos(math.radians(ra)), math.cos(math.radians(dec)) * math.sin(math.radians(ra)), math.sin(math.radians(dec))])
    d, i = dynamo_match_pos.tree.query(v)
    return pos[i] if 2 * math.asin(d / 2) * 206265.0 <= radius else None


def dynamo_stars():
    """HamStar counterparts with L_X, L_bol, Teff and rotation period where known."""
    ham = dynamo_vizier("hamstar")[0]
    by_id, pos = dynamo_rotation()
    out = []
    for r in ham:
        if int(r["Coronal"]) != 1 or int(r["CtpRank"]) != 1:
            continue
        plx, G, bprp, fx = (float(r[k]) if r[k] == r[k] else None for k in ("plx", "Gmag", "BP-RP", "FX"))
        if not plx or plx <= 0.5 or G is None or bprp is None or not fx or fx <= 0:
            continue
        lbol, teff = dynamo_lbol(G, bprp, plx)
        d_cm = 3.0857e18 * 1000.0 / plx
        lx = 4 * math.pi * d_cm ** 2 * fx  # erg/s
        rx = lx / (lbol * 3.828e33)
        sid = int(r["CtpId"]) if r["CtpId"] == r["CtpId"] else None
        rot = by_id.get(sid) or dynamo_match_pos(pos, float(r["RAJ2000"]), float(r["DEJ2000"]))
        if rot and len(rot) == 4:
            rot = (rot[2], rot[3])
        out.append(dict(ero=str(r["ERO"]), gaia=sid, ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"]), G=G, bprp=bprp, plx=plx, teff=teff, lbol=lbol,
                        fx=fx, lx=lx, rx=rx, simbad=str(r["SName"]) if r["SName"] == r["SName"] else "", otype=str(r["SOType"]) if r["SOType"] == r["SOType"] else "",
                        p_coronal=float(r["p-coronal"]), prot=rot[0] if rot else None, prot_src=rot[1] if rot else None))
    return out


dynamo_TEFF = (3000.0, 6000.0)  # convective envelopes only; late-F stars keep a basal corona


def dynamo_xray(record=False):
    stars = dynamo_stars()
    with_p = [s for s in stars if s["prot"] and dynamo_TEFF[0] <= s["teff"] <= dynamo_TEFF[1]]
    for s in with_p:
        s["tauc"] = dynamo_tauc(s["teff"])
        s["ro"] = s["prot"] / s["tauc"]
        s["resid"] = math.log10(s["rx"]) - math.log10(dynamo_expected(s["ro"]))
    res_fast = np.array([s["resid"] for s in with_p if s["ro"] < 1.0])
    slow = [s for s in with_p if s["ro"] >= 1.0]
    # Gaia periods beyond 10 d are not trusted
    cands = sorted((s for s in slow if s["resid"] >= dynamo_RESID and not (s["prot_src"] == "Gaia DR3" and s["prot"] > 10.0)), key=lambda s: -s["resid"])
    print("  Ro at or above 1 with a Gaia period above 10 d, not counted: %d" % sum(1 for s in slow if s["prot_src"] == "Gaia DR3" and s["prot"] > 10.0))
    print("%d HamStar coronal counterparts with a distance and colour; %d at %.0f-%.0f K with a rotation period (%s); Ro below 1: %d, residual scatter %.2f dex, "
          "fraction above %.1f dex %.4f; Ro at or above 1: %d, above %.1f dex: %d" % (
              len(stars), len(with_p), dynamo_TEFF[0], dynamo_TEFF[1], ", ".join("%s %d" % (k, v) for k, v in sorted(((k, sum(1 for s in with_p if s["prot_src"] == k)) for k in {s["prot_src"] for s in with_p}), key=lambda kv: -kv[1])),
              int((np.array([s["ro"] for s in with_p]) < 1).sum()), float(np.std(res_fast)) if len(res_fast) else 0, dynamo_RESID,
              float(np.mean(res_fast >= dynamo_RESID)) if len(res_fast) else 0, len(slow), dynamo_RESID, len(cands)))
    for s in cands[:40]:
        print("  %s Gaia %s: G %.2f BP-RP %.2f, %.0f pc, Teff %.0f, L_bol %.3f; P %.1f d (%s), tau_c %.1f, Ro %.2f; L_X %.2e, L_X/L_bol %.1e, residual %+.2f; SIMBAD %s %s" % (
            s["ero"], s["gaia"], s["G"], s["bprp"], 1000 / s["plx"], s["teff"], s["lbol"], s["prot"], s["prot_src"], s["tauc"], s["ro"], s["lx"], s["rx"], s["resid"], s["simbad"], s["otype"]))
    out = dict(n=len(stars), n_rot=len(with_p), n_fast=int(len(res_fast)), scatter_fast=float(np.std(res_fast)) if len(res_fast) else None,
               frac_fast_above=float(np.mean(res_fast >= dynamo_RESID)) if len(res_fast) else None, n_slow=len(slow), candidates=cands, threshold=dynamo_RESID)
    json.dump(out, open(core.results(dynamo_OUT.replace(".json", "_xray.json")), "w"), indent=1)
    if record:
        core.record_search(
            "dynamo_dead_xray", "Coronal X-rays without a dynamo", archive="eRASS1 HamStar (Freund et al. 2024); Gaia DR3 rotation modulation; TESS (Fetherolf et al. 2023); ZTF (Chen et al. 2020)",
            release="eRASS1; Gaia DR3; VizieR tables", selection="HamStar coronal counterparts (rank 1) with a parallax above 0.5 mas, a colour and a rotation period",
            code=os.path.abspath(__file__), entry="dynamo xray", n_targets=len(with_p), params=dict(residual_dex=dynamo_RESID, ro_min=1.0, relation="Wright et al. 2011, beta -2.5"),
            null_summary="residual scatter %.2f dex at Ro below 1, fraction above %.1f dex %.4f" % (out["scatter_fast"] or 0, dynamo_RESID, out["frac_fast_above"] or 0),
            sensitivity="L_X/L_bol 30 times the relation at Ro above 1; eRASS1 reaches 5e-14 erg/s/cm2", ledger="Coronal X-rays without a dynamo",
            result="%d stars with periods, %d at Ro above 1, %d of them %.1f dex above the relation: %s" % (
                len(with_p), len(slow), len(cands), dynamo_RESID, "; ".join("%s (%s, Ro %.1f, %+.1f dex)" % (s["simbad"] or s["ero"], s["otype"] or "no type", s["ro"], s["resid"]) for s in cands[:12])))
        core.clear_targets("dynamo_dead_xray")
        core.record_targets("dynamo_dead_xray", [dict(target="gaia_dr3:%s" % s["gaia"], stat=s["resid"], unit="log L_X/L_bol above the rotation-activity relation",
                                                          vet="SIMBAD %s %s; period from %s" % (s["simbad"], s["otype"] or "no type", s["prot_src"]),
                                                          note="P %.1f d, Ro %.2f, L_X %.2e erg/s, L_bol %.3f L_sun" % (s["prot"], s["ro"], s["lx"], s["lbol"])) for s in cands])
        print("recorded")


dynamo_RACS_SRC = "https://arxiv.org/e-print/2102.01801"


def dynamo_racs():
    """RACS circular-polarisation stars (Pritchard et al. 2021) from the table source."""
    import io
    import re
    import tarfile
    p = os.path.join(dynamo_ROOT, "racs_table2.tex")
    if not os.path.exists(p):
        b = core.get(dynamo_RACS_SRC, timeout=300, retries=2)
        t = tarfile.open(fileobj=io.BytesIO(b), mode="r:*")
        for n in t.getnames():
            if n.endswith("table2.tex"):
                open(p, "wb").write(t.extractfile(n).read())
    rows, section = [], ""
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.search(r"multicolumn\{10\}\{c\}\{([^}]*)\}", line)
        if m:
            section = m.group(1)
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 9 or not re.match(r"\d\d:\d\d:\d\d", cells[2]):
            continue

        def clean(s):
            return re.sub(r"[\$~\\,{}]", "", s).replace("--", "-").strip()

        def val(s):
            m2 = re.search(r"[-+]?\d*\.?\d+", clean(s).replace("+", ""))
            return float(m2.group(0)) if m2 else None
        ra = cells[2].split(":")
        dec = clean(cells[3]).split(":")
        sign = -1.0 if dec[0].startswith("-") else 1.0
        rows.append(dict(name=clean(cells[0]).replace("~", " "), sptype=clean(cells[1]), section=section,
                         ra=15 * (float(ra[0]) + float(ra[1]) / 60 + float(ra[2]) / 3600),
                         dec=sign * (abs(float(dec[0])) + float(dec[1]) / 60 + float(dec[2]) / 3600),
                         plx=val(cells[5]), s888=val(cells[6]), frac_pol=val(cells[7])))
    return rows


def dynamo_polarised(record=False):
    from astropy.coordinates import SkyCoord
    v = dynamo_vizier("vlotss")[0]
    ham = {int(r["CtpId"]): r for r in dynamo_vizier("hamstar")[0] if r["CtpId"] == r["CtpId"] and int(r["Coronal"]) == 1}
    ham_pos = [(float(r["RAJ2000"]), float(r["DEJ2000"]), r) for r in ham.values()]
    by_id, pos = dynamo_rotation()
    rows = []
    def gaia_id(x):
        s = str(x).strip().split()[-1] if x == x and str(x).strip() else ""
        return int(s) if s.isdigit() else None
    sources = [dict(name=str(r["Name"]), survey="V-LoTSS", vlotss=str(r["VLoTSS"]).strip(), gaia=gaia_id(r["Gaia"]), ra=float(r["RAGdeg"]) if r["RAGdeg"] == r["RAGdeg"] else float(r["RAJ2000"]),
                    dec=float(r["DEGdeg"]) if r["DEGdeg"] == r["DEGdeg"] else float(r["DEJ2000"]),
                    plx=float(r["plx"]) if r["plx"] == r["plx"] else None, G=float(r["Gmag"]) if r["Gmag"] == r["Gmag"] else None,
                    bprp=float(r["BP-RP"]) if r["BP-RP"] == r["BP-RP"] else None, vI=float(r["vI"]) if r["vI"] == r["vI"] else None,
                    simbad=str(r["SimbadName"]), sptype="") for r in v if "PSR" not in str(r["Name"])]
    sources += [dict(name=r["name"], survey="RACS (%s)" % r["section"].lower(), gaia=None, ra=r["ra"], dec=r["dec"], plx=r["plx"], G=None, bprp=None,
                     vI=abs(r["frac_pol"]) if r["frac_pol"] is not None else None, simbad=r["sptype"], sptype=r["sptype"]) for r in dynamo_racs()]
    for s in sources:
        name, sid, ra, dec, plx, G, bprp = s["name"], s["gaia"], s["ra"], s["dec"], s["plx"], s["G"], s["bprp"]
        c = SkyCoord(ra, dec, unit="deg", frame="icrs").galactic
        in_de = c.l.deg > 180.0
        h = ham.get(sid) if sid else None
        if h is None:
            # eRASS1 counterpart by position for RACS stars without Gaia ids
            near = [x for x in ham_pos if abs(x[1] - dec) < 0.002 and abs(x[0] - ra) * math.cos(math.radians(dec)) < 0.002]
            h = near[0][2] if near else None
            if h is not None and (G is None or bprp is None or not plx):
                G, bprp, plx = (float(h[k]) if h[k] == h[k] else None for k in ("Gmag", "BP-RP", "plx"))
        lbol, teff = dynamo_lbol(G, bprp, plx) if plx and plx > 0 and G is not None and bprp is not None else (None, None)
        if h is not None and lbol:
            lx = 4 * math.pi * (3.0857e18 * 1000.0 / plx) ** 2 * float(h["FX"])
            rx, rx_lim = lx / (lbol * 3.828e33), None
        elif in_de and lbol:
            rx, rx_lim = None, 4 * math.pi * (3.0857e18 * 1000.0 / plx) ** 2 * dynamo_ERASS_LIMIT / (lbol * 3.828e33)
        else:
            rx, rx_lim = None, None
        rot = (by_id.get(sid) if sid else None) or dynamo_match_pos(pos, ra, dec)
        if rot and len(rot) == 4:
            rot = (rot[2], rot[3])
        rows.append(dict(name=name, survey=s["survey"], vlotss=s.get("vlotss"), gaia=sid, G=G, bprp=bprp, plx=plx, teff=teff, lbol=lbol, vI=s["vI"], in_erass_de=bool(in_de),
                         rx=rx, rx_limit=rx_lim, prot=rot[0] if rot else None, prot_src=rot[1] if rot else None, simbad=s["simbad"]))
    flagged = [x for x in rows if x["vI"] and x["vI"] >= 30 and x["prot"] and x["prot"] > 20 and ((x["rx"] is not None and x["rx"] < 1e-5) or (x["rx_limit"] is not None and x["rx_limit"] < 1e-5))]
    print("%d V-LoTSS stars with Gaia counterparts; %d in the eRASS1 half of the sky, %d with eRASS1 detections, %d with rotation periods; %d above 30 per cent polarisation at "
          "L_X/L_bol below 1e-5 and periods above 20 d" % (len(rows), sum(1 for x in rows if x["in_erass_de"]), sum(1 for x in rows if x["rx"] is not None),
                                                           sum(1 for x in rows if x["prot"]), len(flagged)))
    for x in sorted(rows, key=lambda x: -(x["vI"] or 0)):
        print("  %-22s %-28s V/I %5.1f%%; %s pc, Teff %s; L_X/L_bol %s; P %s (%s); eRASS1 sky %s; %s" % (
            x["name"], x["survey"], x["vI"] or 0, "%.0f" % (1000 / x["plx"]) if x["plx"] else "-", "%.0f" % x["teff"] if x["teff"] else "-",
            ("%.1e" % x["rx"]) if x["rx"] is not None else ("< %.1e" % x["rx_limit"] if x["rx_limit"] is not None else "-"),
            "%.1f d" % x["prot"] if x["prot"] else "-", x["prot_src"] or "-", "yes" if x["in_erass_de"] else "no", x["simbad"]))
    json.dump(dict(stars=rows, flagged=flagged), open(core.results(dynamo_OUT.replace(".json", "_polarised.json")), "w"), indent=1)
    if record:
        core.record_search(
            "dynamo_dead_polarised", "Circular polarisation at inactive stars", archive="V-LoTSS (Callingham et al. 2023); RACS circular polarisation (Pritchard et al. 2021); eRASS1 HamStar; Gaia DR3, TESS and ZTF rotation periods",
            release="V-LoTSS as on VizieR; RACS table 2; eRASS1", selection="V-LoTSS stars and RACS circularly polarised stars", code=os.path.abspath(__file__), entry="dynamo polarised",
            n_targets=len(rows), params=dict(vI_min=30, prot_min_d=20, rx_max=1e-5, erass_limit=dynamo_ERASS_LIMIT),
            null_summary="the V-LoTSS stars' own X-ray and rotation distribution", sensitivity="0.5 mJy at 144 MHz; L_X/L_bol to 1e-5 where eRASS1 covers and the star is within about 50 pc",
            ledger="Circular polarisation at inactive stars",
            result="%d stars, %d with rotation periods, %d with eRASS1 detections and %d limits; flagged %d: %s" % (
                len(rows), sum(1 for x in rows if x["prot"]), sum(1 for x in rows if x["rx"] is not None), sum(1 for x in rows if x["rx_limit"] is not None), len(flagged),
                ", ".join(x["name"] for x in flagged) or "none"))
        core.clear_targets("dynamo_dead_polarised")
        # a V-LoTSS source without a Gaia DR3 counterpart is keyed by its V-LoTSS identifier, a RACS star by its name
        core.record_targets("dynamo_dead_polarised", [dict(target=("gaia_dr3:%s" % x["gaia"]) if x["gaia"] else ("vlotss:" + x["vlotss"]) if x["vlotss"] else "star:" + x["name"], stat=x["vI"], unit="circular polarisation per cent",
                                                               vet="flagged" if x in flagged else ("active or rotating (L_X/L_bol %s, P %s)" % (
                                                                   ("%.1e" % x["rx"]) if x["rx"] is not None else "unknown", ("%.1f d" % x["prot"]) if x["prot"] else "unknown")),
                                                               note="%s; %s" % (x["name"], x["simbad"])) for x in rows])
        print("recorded")


dynamo_CEILING = 1e-2
dynamo_CEILING_TYPES = ("CV", "Nova", "WD", "YSO", "TTau", "Orion", "Ae", "Be", "AGN", "QSO", "Sy", "HII", "Pulsar", "XB", "LMXB", "HMXB", "Symb", "Em", "Ae*", "Be*")
dynamo_CEILING_RADIUS = 20.0  # arcsec; SIMBAD places X-ray sources 5-15 arcsec from their Gaia counterparts


def dynamo_simbad_near(ra, dec, radius=dynamo_CEILING_RADIUS):
    """SIMBAD objects within radius, nearest first: (main_id, otype, sep arcsec)."""
    q = ("SELECT main_id, otype, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', %f, %f))*3600 AS sep FROM basic "
         "WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %f, %f, %f)) = 1" % (ra, dec, ra, dec, radius / 3600.0))
    try:
        return sorted(((str(r["main_id"]), str(r["otype"]).strip(), float(r["sep"])) for r in core.simbad(q)), key=lambda x: x[2])
    except Exception as e:
        print("  SIMBAD failed at %.5f %+.5f: %s" % (ra, dec, str(e)[:60]))
        return []


def dynamo_ceiling(record=False):
    """HamStar counterparts above the coronal ceiling L_X/L_bol 1e-2, vetted by SIMBAD types within 20 arcsec, colour and match probability."""
    stars = dynamo_stars()
    rx = np.array([s["rx"] for s in stars])
    print("%d HamStar coronal counterparts: log L_X/L_bol percentiles 50/90/99/99.9: %s; above 1e-3: %d, above 1e-2: %d, above 1e-1: %d" % (
        len(stars), np.percentile(np.log10(rx), [50, 90, 99, 99.9]).round(2), int((rx > 1e-3).sum()), int((rx > dynamo_CEILING).sum()), int((rx > 1e-1).sum())))
    high = sorted((s for s in stars if s["rx"] > dynamo_CEILING), key=lambda s: -s["rx"])
    for s in high:
        s["natural_type"] = any(k.lower() in (s["otype"] or "").lower() for k in dynamo_CEILING_TYPES)
        # types within the ceiling radius read too, since SIMBAD keeps X-ray positions as separate objects
        near = dynamo_simbad_near(s["ra"], s["dec"])
        typed = [n for n in near if any(k.lower() in n[1].lower() for k in dynamo_CEILING_TYPES)]
        # most telling type first, then nearest
        typed.sort(key=lambda n: (min(i for i, k in enumerate(dynamo_CEILING_TYPES) if k.lower() in n[1].lower()), n[2]))
        s["near"] = "; ".join("%s (%s, %.0f arcsec)" % n for n in near[:3])
        if typed and not s["natural_type"]:
            s["natural_type"], s["otype"], s["simbad"], s["near_sep"] = True, typed[0][1], s["simbad"] or typed[0][0], typed[0][2]
        # M_G against the main sequence 1.7 + 3.6 (BP-RP)
        s["MG"] = s["G"] + 5 * math.log10(s["plx"]) - 10
        s["dwarf"] = 0.4 < s["bprp"] < 3.5 and abs(s["MG"] - (1.7 + 3.6 * s["bprp"])) < 1.2
    types = {}
    for s in high:
        types[s["otype"] or "none"] = types.get(s["otype"] or "none", 0) + 1
    print("  SIMBAD types of those above the ceiling: " + ", ".join("%s %d" % kv for kv in sorted(types.items(), key=lambda kv: -kv[1])[:20]))
    left = [s for s in high if not s["natural_type"] and s["dwarf"] and s["p_coronal"] >= 0.5]
    print("  main-sequence by Gaia, no cataclysmic, young, degenerate or extragalactic type, counterpart probability 0.5 or more: %d" % len(left))
    for s in left[:40]:
        print("  %s Gaia %s %s (%s): G %.2f BP-RP %.2f M_G %.2f, %.0f pc, Teff %.0f; F_X %.1e, L_X %.2e, L_X/L_bol %.2e, p %.2f" % (
            s["ero"], s["gaia"], s["simbad"], s["otype"] or "no type", s["G"], s["bprp"], s["MG"], 1000 / s["plx"], s["teff"], s["fx"], s["lx"], s["rx"], s["p_coronal"]))
    json.dump(dict(n=len(stars), above_1e3=int((rx > 1e-3).sum()), above_ceiling=len(high), left=left, types=types), open(core.results("coronal_ceiling.json"), "w"), indent=1)
    if record:
        core.record_search(
            "coronal_ceiling", "X-ray output above the coronal ceiling", archive="eRASS1 HamStar (Freund et al. 2024); Gaia DR3",
            release="eRASS1", selection="HamStar coronal counterparts (rank 1) with a parallax above 0.5 mas and a colour", code=os.path.abspath(__file__), entry="dynamo ceiling",
            n_targets=len(stars), params=dict(ceiling=dynamo_CEILING, types_set_aside=dynamo_CEILING_TYPES),
            null_summary="the saturated relation at 1e-3 with its 0.3 dex scatter and the flare duty cycle", sensitivity="L_X/L_bol above 1e-2 in the eRASS1 average",
            ledger="X-ray output above the coronal ceiling",
            result="%d counterparts, %d above 1e-3, %d above 1e-2; %d of those main-sequence by Gaia without a cataclysmic, young, degenerate or extragalactic type: %s" % (
                len(stars), int((rx > 1e-3).sum()), len(high), len(left), "; ".join("%s (%s, %.1e)" % (s["simbad"] or s["ero"], s["otype"] or "no type", s["rx"]) for s in left[:10])))
        core.clear_targets("coronal_ceiling")
        core.record_targets("coronal_ceiling", [dict(target="gaia_dr3:%s" % s["gaia"], stat=s["rx"], unit="L_X/L_bol from the eRASS1 flux and Gaia",
                                                        vet=("natural type %s" % s["otype"]) if s["natural_type"] else ("not main-sequence by Gaia" if not s["dwarf"] else ("counterpart probability %.2f" % s["p_coronal"] if s["p_coronal"] < 0.5 else "above the ceiling: to vet")),
                                                        note="%s%s; G %.2f BP-RP %.2f M_G %.2f %.0f pc" % (s["simbad"], (" at %.0f arcsec" % s["near_sep"]) if s.get("near_sep") else "",
                                                                                                           s["G"], s["bprp"], s["MG"], 1000 / s["plx"])) for s in high])
        print("recorded")


def dynamo_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "xray"
    rec = "record" in sys.argv[2:]
    {"xray": dynamo_xray, "polarised": dynamo_polarised, "ceiling": dynamo_ceiling}[mode](record=rec)


# --------------------------------------------------------------------------------------------------------------
# bursts and X-rays from places that should be dark

grbstars_DOC = r"""Gamma-ray bursts from nearby stars: Swift's GRB table (HEASARC swiftgrb) positions with an XRT localisation, each against the sources of the Gaia Catalogue of Nearby Stars within 20 pc carried to the burst's year, a star inside the larger of 5 arcsec and three times the XRT error, against the same count with the bursts moved by 1 and 2 deg in four directions.

    python -I seti.py highenergy grbstars [record] [--plx-mas 50] [--min-arcsec 5] [--sigma 3] [--shifts-deg 1 2]"""


def grbstars_main(record=False, plx_mas=50.0, min_arcsec=5.0, sigma=3.0, shifts_deg=[1.0, 2.0]):
    grbs = [r for r in core.heasarc("SELECT name, ra, dec, xrt_pos_err, start_time FROM swiftgrb WHERE xrt_pos_err IS NOT NULL")
            if num(r["ra"]) is not None and num(r["xrt_pos_err"])]
    stars = catalogues.gcns("source_id, ra, dec, pmra, pmdec, parallax", "parallax > %g" % plx_mas)
    sra = np.radians([float(s["ra"]) for s in stars])
    sde = np.radians([float(s["dec"]) for s in stars])

    def inside(dra=0.0, dde=0.0):
        out = []
        for g in grbs:
            a, d = math.radians(num(g["ra"]) + dra), math.radians(num(g["dec"]) + dde)
            cs = np.sin(d) * np.sin(sde) + np.cos(d) * np.cos(sde) * np.cos(sra - a)
            sep = np.degrees(np.arccos(np.clip(cs, -1, 1))) * 3600
            r = max(min_arcsec, sigma * num(g["xrt_pos_err"]))
            for j in np.nonzero(sep < r + 60)[0]:
                s = stars[j]
                # the star where it stood at the burst, from its proper motion since 2016
                yr = 2016.0
                try:
                    yr = float(str(g["start_time"])[:4]) + 0.5
                except ValueError:
                    pass
                ra2, de2 = core.star_at(float(s["ra"]), float(s["dec"]), num(s["pmra"], 0.0), num(s["pmdec"], 0.0), 2016.0, yr)
                if core.sep_arcsec(ra2, de2, math.degrees(a), math.degrees(d)) < r:
                    out.append((g["name"], s["source_id"]))
        return out
    hits = inside()
    null = [len(inside(dx, dy)) for k in shifts_deg for dx, dy in ((k, 0), (-k, 0), (0, k), (0, -k))]
    print("%d Swift bursts with XRT positions, %d sources within 20 pc; a star inside the error circle: %d %s; moved bursts %s" % (
        len(grbs), len(stars), len(hits), hits, null))
    if record:
        core.record_run(
            "swift_grb_nearby_stars", "Gamma-ray bursts from nearby stars", "HEASARC swiftgrb; Gaia Catalogue of Nearby Stars",
            "Swift bursts with XRT positions: %d; sources within 20 pc: %d" % (len(grbs), len(stars)), os.path.abspath(__file__), "grbstars",
            len(grbs), "bursts", dict(radius="max(%g arcsec, %g XRT sigma)" % (min_arcsec, sigma), plx_mas=plx_mas, shifts_deg=shifts_deg),
            "the bursts moved %s deg: %s" % (shifts_deg, null),
            "a gamma-ray burst localised by XRT onto a star within 20 pc", "%d against a mean of %.2f moved" % (len(hits), float(np.mean(null))),
            [dict(target="gaia_dr3:%s" % s, note="inside %s" % g) for g, s in hits])
        print("recorded")


coldwdx_DOC = r"""X-rays from cold white dwarfs: Gentile Fusillo et al. (2021) white dwarfs (VizieR J/MNRAS/508/3877) within 100 pc with a white-dwarf probability above 0.75 and hydrogen-model temperatures below 8000 K, too cool for a corona or photospheric X-rays, against eRASS1's main catalogue (J/A+A/682/A34/erass1-m) by CDS XMatch within three times each source's positional error, against the same match with the white dwarfs moved 2 arcmin in eight directions; each match read for another Gaia star inside the X-ray error circle, a companion where its parallax agrees with the white dwarf's.

    python -I seti.py highenergy coldwdx [record] [--teff-max 8000] [--pwd 0.75] [--sigma 3] [--shift-arcmin 2]"""


def coldwdx_main(record=False, teff_max=8000.0, pwd=0.75, sigma=3.0, shift_arcmin=2.0, xmatch_arcsec=30.0):
    wds = catalogues.white_dwarfs("WDJname, RA_ICRS, DE_ICRS, Plx, TeffH, Pwd", pwd=pwd, teff_max=teff_max)
    pos = [dict(source_id=w["WDJname"], ra=float(w["RA_ICRS"]), dec=float(w["DE_ICRS"])) for w in wds]

    def match(dra=0.0, dde=0.0):
        up = []
        for p in pos:
            ra, de = p["ra"] + dra / max(math.cos(math.radians(p["dec"])), 0.05), p["dec"] + dde
            if abs(de) > 90:
                ra, de = ra + 180, math.copysign(180, de) - de
            up.append(dict(p, ra=ra % 360, dec=de))
        return [(m["source_id"], float(m["angDist"]), m["IAUName"], float(m["RA_ICRS"]), float(m["DE_ICRS"]))
                for m in core.xmatch(up, "J/A+A/682/A34/erass1-m", xmatch_arcsec)]
    d = shift_arcmin / 60.0
    runs = [match()] + [match(x, y) for x, y in ((d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, d), (d, -d), (-d, -d))]
    # XMatch returns the catalogue's first hundred columns, without the positional error
    names = sorted({x[2] for run in runs for x in run})
    pe = {}
    for k in range(0, len(names), 200):
        for r in core.tap("vizier", 'SELECT IAUName, posErr FROM "J/A+A/682/A34/erass1-m" WHERE IAUName IN (%s)' % ", ".join(
                "'%s'" % n for n in names[k:k + 200])):
            pe[r["IAUName"]] = num(r["posErr"])
    hits = [(w, s, pe[n], n, ra, de) for w, s, n, ra, de in runs[0] if pe.get(n) and s < sigma * pe[n]]
    null = [sum(1 for w, s, n, ra, de in run if pe.get(n) and s < sigma * pe[n]) for run in runs[1:]]
    # a star sharing the X-ray error circle is the natural source: the white dwarf's companion at its parallax, or a neighbour in the beam
    by = {w["WDJname"].strip(): w for w in wds}
    why = {}
    for w, s, e, n, ra, de in hits:
        wd = by[w]
        near = core.tap("gaia", "SELECT source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag, bp_rp FROM gaiadr3.gaia_source WHERE "
                                "1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f))" % (ra, de, max(sigma * e, 5.0) / 3600.0))
        others = [g for g in near if core.sep_arcsec(float(g["ra"]), float(g["dec"]), float(wd["RA_ICRS"]), float(wd["DE_ICRS"])) > 1.0]
        comp = [g for g in others if num(g["parallax"]) and abs(float(g["parallax"]) - float(wd["Plx"])) < 3 * (num(g["parallax_error"]) or 1) + 1]
        why[w] = ("companion", comp) if comp else ("blend", others) if others else (None, [])
    print("%d white dwarfs within 100 pc below 8000 K; within 3 sigma of an eRASS1 source: %d; moved 2 arcmin %s" % (len(wds), len(hits), null))
    for w, s, e, n, ra, de in hits:
        v, g = why[w]
        print("  %s: %s at %.1f arcsec (error %.1f); %s %s" % (w, n, s, e, v or "no other Gaia star in the circle",
                                                             [(x["source_id"], x["phot_g_mean_mag"], x["bp_rp"], x["parallax"]) for x in g]))
    if record:
        core.record_run(
            "erass1_cold_white_dwarfs", "X-rays from cold white dwarfs", "Gentile Fusillo et al. (2021) white dwarfs; eRASS1 main catalogue by CDS XMatch",
            "white dwarfs within 100 pc below 8000 K: %d" % len(wds), os.path.abspath(__file__), "coldwdx", len(wds), "white dwarfs",
            dict(teff_k=teff_max, pwd=pwd, sigma=sigma, shift_arcmin=shift_arcmin), "the white dwarfs moved %g arcmin: %s" % (shift_arcmin, null),
            "an eRASS1 source on a white dwarf too cool to emit X-rays",
            "%d against a mean of %.2f moved" % (len(hits), float(np.mean(null))),
            [dict(target="wd:%s" % w, stat=sep, unit="arcsec", note="%s, error %.1f arcsec%s" % (n, e, "; Gaia DR3 %s in the circle" % ", ".join(
                x["source_id"] for x in why[w][1]) if why[w][1] else ""),
                  vet=why[w][0] or ("chance-rate" if len(hits) <= np.mean(null) + 3 * math.sqrt(max(np.mean(null), 1)) else None))
             for w, sep, e, n, ra, de in hits])
        print("recorded")


wdmass_DOC = r"""White dwarfs above the Chandrasekhar mass: Gentile Fusillo et al. (2021) white dwarfs (VizieR J/MNRAS/508/3877) within 100 pc with a white-dwarf probability above 0.75 whose hydrogen-model and helium-model masses both exceed 1.40 solar masses at 3 sigma, more than electron degeneracy holds; each read for its fit's chi-square and temperature.

    python -I seti.py highenergy wdmass [record] [--mass-msun 1.4] [--sigma 3] [--pwd 0.75]"""


def wdmass_main(record=False, mass_msun=1.40, sigma=3.0, pwd=0.75):
    rows = catalogues.white_dwarfs("WDJname, Plx, Pwd, GmagCorr, TeffH, MassH, e_MassH, chisqH, MassHe, e_MassHe, chisqHe", pwd=pwd, where="MassH IS NOT NULL")
    over = [r for r in rows if num(r["MassH"]) - sigma * (num(r["e_MassH"]) or 9) > mass_msun and num(r["MassHe"]) is not None
            and num(r["MassHe"]) - sigma * (num(r["e_MassHe"]) or 9) > mass_msun]
    top = sorted(rows, key=lambda r: -(num(r["MassH"]) or 0))[:5]
    print("%d white dwarfs within 100 pc with hydrogen-model masses; above 1.40 solar masses at 3 sigma in both models: %d; the heaviest %s" % (
        len(rows), len(over), ["%s %s +- %s (He %s), Teff %s, chi2 %s" % (r["WDJname"].strip(), r["MassH"], r["e_MassH"], r["MassHe"], r["TeffH"], r["chisqH"]) for r in top]))
    if record:
        core.record_run("wd_above_chandrasekhar", "White dwarfs above the Chandrasekhar mass", "Gentile Fusillo et al. (2021) white dwarfs (VizieR J/MNRAS/508/3877)",
                        "white dwarfs within 100 pc with hydrogen-model masses: %d" % len(rows), os.path.abspath(__file__), "wdmass", len(rows), "white dwarfs",
                        dict(mass_msun=mass_msun, sigma=sigma, pwd=pwd), "the Chandrasekhar mass", "a white dwarf heavier than degeneracy holds",
                        "%d; the heaviest %s at %s solar masses" % (len(over), top[0]["WDJname"].strip(), top[0]["MassH"]),
                        [dict(target="wd:%s" % r["WDJname"].strip(), stat=num(r["MassH"]), unit="solar masses", note="He-model %s, Teff %s, chi2 %s" % (
                            r["MassHe"], r["TeffH"], r["chisqH"])) for r in over])
        print("recorded")


coldbdx_DOC = r"""X-rays from cold brown dwarfs: the T and Y dwarfs of the Kirkpatrick et al. (2021) 20 pc census with Spitzer astrometry, carried to 2020.0, against eRASS1's main catalogue by CDS XMatch within 15 arcsec, against the same match with the dwarfs moved 2 arcmin in eight directions; a body below 1300 K has no corona.

    python -I seti.py highenergy coldbdx [record] [--radius-arcsec 15] [--epoch 2020]"""


def coldbdx_main(record=False, radius_arcsec=15.0, epoch=2020.0):
    bd = cold_dwarfs(epoch)
    hits, null = shifted_matches(bd, "J/A+A/682/A34/erass1-m", radius_arcsec)
    names = sorted({m["source_id"] for m in hits})
    print("%d T and Y dwarfs; eRASS1 sources within 15 arcsec: %d %s; moved 2 arcmin %s" % (len(bd), len(names), [
        (m["source_id"], m.get("IAUName"), round(float(m["angDist"]), 1)) for m in hits], null))
    if record:
        mean = float(np.mean(null))
        core.record_run("erass1_cold_brown_dwarfs", "X-rays from cold brown dwarfs", "Kirkpatrick et al. (2021) 20 pc census (J/ApJS/253/7); eRASS1 main "
                        "catalogue by CDS XMatch", "T and Y dwarfs with Spitzer astrometry: %d" % len(bd), os.path.abspath(__file__), "coldbdx", len(bd),
                        "brown dwarfs", dict(radius_arcsec=radius_arcsec, epoch=epoch), "the dwarfs moved 2 arcmin: %s" % null, "an X-ray source on a brown dwarf too cold for a corona",
                        "%d against a mean of %.2f moved" % (len(names), mean),
                        [dict(target="bd:%s" % n, vet="chance-rate" if len(names) <= mean + 3 * math.sqrt(max(mean, 1)) else None) for n in names])
        print("recorded")


coldbduv_DOC = r"""Ultraviolet from cold brown dwarfs: the T and Y dwarfs of the Kirkpatrick et al. (2021) 20 pc census with Spitzer astrometry, carried to 2007.0, against GALEX's all-sky survey (GUVcat AIS, II/335) by CDS XMatch within 5 arcsec, a detection in either band, against the same match with the dwarfs moved 2 arcmin in eight directions; a body below 1300 K gives no ultraviolet GALEX can see.

    python -I seti.py highenergy coldbduv [record] [--radius-arcsec 5] [--epoch 2007]"""


def coldbduv_main(record=False, radius_arcsec=5.0, epoch=2007.0):
    bd = cold_dwarfs(epoch)
    hits, null = shifted_matches(bd, "II/335/galex_ais", radius_arcsec)
    names = sorted({m["source_id"] for m in hits})
    print("%d T and Y dwarfs; GALEX sources within 5 arcsec: %d %s; moved 2 arcmin %s" % (len(bd), len(names), [
        (m["source_id"], m.get("NUVmag"), m.get("FUVmag"), round(float(m["angDist"]), 1)) for m in hits], null))
    if record:
        mean = float(np.mean(null))
        core.record_run("galex_cold_brown_dwarfs", "Ultraviolet from cold brown dwarfs", "Kirkpatrick et al. (2021) 20 pc census (J/ApJS/253/7); GALEX "
                        "GUVcat AIS (II/335) by CDS XMatch", "T and Y dwarfs with Spitzer astrometry: %d" % len(bd), os.path.abspath(__file__), "coldbduv",
                        len(bd), "brown dwarfs", dict(radius_arcsec=radius_arcsec, epoch=epoch), "the dwarfs moved 2 arcmin: %s" % null,
                        "ultraviolet from a brown dwarf below 1300 K", "%d against a mean of %.2f moved" % (len(names), mean),
                        [dict(target="bd:%s" % n, vet="chance-rate" if len(names) <= mean + 3 * math.sqrt(max(mean, 1)) else None) for n in names])
        print("recorded")


coldwduv_DOC = r"""Ultraviolet from the coldest white dwarfs: Gentile Fusillo et al. (2021) white dwarfs within 100 pc with a white-dwarf probability above 0.75 and hydrogen-model temperatures below 4500 K, carried from 2016.0 to 2007.0, against GALEX GUVcat AIS (II/335) by CDS XMatch within 5 arcsec, a detection in either band, against the same match with the white dwarfs moved 2 arcmin in eight directions.

    python -I seti.py highenergy coldwduv [record] [--teff-max 4500] [--radius-arcsec 5] [--epoch 2007]"""


def coldwduv_main(record=False, teff_max=4500.0, radius_arcsec=5.0, epoch=2007.0):
    wds = catalogues.white_dwarfs("WDJname, RA_ICRS, DE_ICRS, pmRA, pmDE, TeffH, GmagCorr", teff_max=teff_max)
    pos = []
    for w in wds:
        ra, de = core.star_at(float(w["RA_ICRS"]), float(w["DE_ICRS"]), num(w["pmRA"], 0.0), num(w["pmDE"], 0.0), 2016.0, epoch)
        pos.append(dict(name=w["WDJname"].strip(), ra=ra, dec=de))
    hits, null = shifted_matches(pos, "II/335/galex_ais", radius_arcsec)
    names = sorted({m["source_id"] for m in hits})
    mean = float(np.mean(null))
    print("%d white dwarfs within 100 pc below 4500 K; GALEX sources within 5 arcsec: %d %s; moved 2 arcmin %s (mean %.1f)" % (
        len(pos), len(names), [(m["source_id"], m.get("NUVmag"), m.get("FUVmag"), round(float(m["angDist"]), 1)) for m in hits], null, mean))
    if record:
        core.record_run("galex_coldest_white_dwarfs", "Ultraviolet from the coldest white dwarfs", "Gentile Fusillo et al. (2021) white dwarfs; GALEX GUVcat AIS "
                        "(II/335) by CDS XMatch", "white dwarfs within 100 pc below 4500 K: %d" % len(pos), os.path.abspath(__file__), "coldwduv", len(pos),
                        "white dwarfs", dict(radius_arcsec=radius_arcsec, epoch=epoch, teff_k=teff_max), "the white dwarfs moved 2 arcmin: %s" % null,
                        "ultraviolet from a white dwarf too cold to give it", "%d against a mean of %.1f moved" % (len(names), mean),
                        [dict(target="wd:%s" % n, vet="chance-rate" if len(names) <= mean + 3 * math.sqrt(max(mean, 1)) else None) for n in names])
        print("recorded")


coldwdgamma_DOC = r"""Gamma rays from cold white dwarfs: Gentile Fusillo et al. (2021) white dwarfs within 100 pc with a white-dwarf probability above 0.75 and hydrogen-model temperatures below 8000 K, inside the 95 per cent ellipse of a 4FGL-DR4 source, against the same count with the white dwarfs moved 1 deg in eight directions; a white dwarf cold enough to have no corona, accretion or wind gives no GeV photons. A source the catalogue associates with another object is that object's.

    python -I seti.py highenergy coldwdgamma [record] [--teff-max 8000] [--shift-deg 1]"""


def fgl_sources():
    """4FGL-DR4 (gll_psc_v35): name, position, 95 per cent ellipse and first association and class of every source."""
    path = os.path.join(sources_HERE, "gll_psc_v35.fit")
    if not os.path.exists(path):
        open(path, "wb").write(core.get("https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/gll_psc_v35.fit", timeout=300))
    from astropy.io import fits
    with fits.open(path) as h:
        t = h["LAT_Point_Source_Catalog"].data
        return dict(name=np.char.strip(np.asarray(t["Source_Name"]).astype(str)), ra=np.asarray(t["RAJ2000"], float),
                    dec=np.asarray(t["DEJ2000"], float), a=np.asarray(t["Conf_95_SemiMajor"], float), b=np.asarray(t["Conf_95_SemiMinor"], float),
                    pa=np.asarray(t["Conf_95_PosAng"], float), cls=np.char.strip(np.asarray(t["CLASS1"]).astype(str)),
                    assoc=np.char.strip(np.asarray(t["ASSOC1"]).astype(str)))


def fgl_inside(fgl, ra, dec):
    """Indices of the 4FGL sources whose 95 per cent ellipse holds the position."""
    dx = (ra - fgl["ra"] + 180) % 360 - 180
    dx = dx * math.cos(math.radians(dec))
    dy = dec - fgl["dec"]
    near = np.nonzero(np.isfinite(fgl["a"]) & (np.abs(dy) < 1.0) & (np.abs(dx) < 1.0))[0]
    th = np.radians(fgl["pa"][near])
    u = dx[near] * np.sin(th) + dy[near] * np.cos(th)
    v = dx[near] * np.cos(th) - dy[near] * np.sin(th)
    return near[(u / fgl["a"][near]) ** 2 + (v / fgl["b"][near]) ** 2 <= 1.0]


def coldwdgamma_main(record=False, teff_max=8000.0, shift_deg=1.0):
    wds = catalogues.white_dwarfs("WDJname, RA_ICRS, DE_ICRS, TeffH", teff_max=teff_max)
    fgl = fgl_sources()
    un = (fgl["cls"] == "") | (fgl["cls"] == "unk")

    def count(dra=0.0, dde=0.0, keep=False):
        out = []
        for w in wds:
            ra, de = float(w["RA_ICRS"]), float(w["DE_ICRS"]) + dde
            ra += dra / max(math.cos(math.radians(de)), 0.05)
            if abs(de) > 90:
                ra, de = ra + 180, math.copysign(180, de) - de
            out += [(w["WDJname"].strip(), int(k)) for k in fgl_inside(fgl, ra % 360, de)]
        return out
    at = count()
    null = [sum(1 for w, k in count(x * shift_deg, y * shift_deg) if un[k]) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))]
    mean = float(np.mean(null))
    free = [(w, k) for w, k in at if un[k]]
    owned = [(w, k) for w, k in at if not un[k]]
    print("%d white dwarfs within 100 pc below 8000 K; inside an unassociated 4FGL-DR4 ellipse: %d %s; moved 1 deg %s (mean %.1f); inside an "
          "associated one: %d %s" % (len(wds), len(free), [(w, fgl["name"][k], round(float(fgl["a"][k]) * 60, 1)) for w, k in free], null, mean,
                                     len(owned), [(w, fgl["name"][k], fgl["cls"][k], fgl["assoc"][k]) for w, k in owned]))
    if record:
        chance = len(free) <= mean + 3 * math.sqrt(max(mean, 1))
        core.record_run("fermi_cold_white_dwarfs", "Gamma rays from cold white dwarfs", "Gentile Fusillo et al. (2021) white dwarfs; Fermi-LAT 4FGL-DR4 "
                        "(gll_psc_v35)", "white dwarfs within 100 pc below 8000 K: %d" % len(wds), os.path.abspath(__file__), "coldwdgamma", len(wds),
                        "white dwarfs", dict(teff_k=teff_max, ellipse="95 per cent", shift_deg=shift_deg), "the white dwarfs moved %g deg: %s" % (shift_deg, null),
                        "GeV gamma rays from a white dwarf with no corona, accretion or wind",
                        "%d inside an unassociated ellipse against a mean of %.1f moved; %d inside an associated one" % (len(free), mean, len(owned)),
                        [dict(target="wd:%s" % w, vet="chance-rate" if chance else None, note=str(fgl["name"][k])) for w, k in free] +
                        [dict(target="wd:%s" % w, vet="natural-source", note="%s, associated with %s (%s)" % (fgl["name"][k], fgl["assoc"][k], fgl["cls"][k]))
                         for w, k in owned])
        print("recorded")


nearbat_DOC = r"""Hard X-rays from the nearest stars: Gaia Catalogue of Nearby Stars objects within 20 pc against the Swift-BAT 105-month (J/ApJS/235/4) and 150-month (J/A+A/709/A12) sources at 14-195 keV, inside a radius of 30 arcmin over the signal-to-noise (at least 1 arcmin; 4.5 arcmin for the 150-month sources), against the same count with the stars moved 1 deg in eight directions; a star's corona gives soft X-rays, and a steady source of hard ones at a quiet star would need another engine. A source whose counterpart the catalogue names is that counterpart's.

    python -I seti.py highenergy nearbat [record] [--plx-mas 50] [--shift-deg 1]"""


def nearbat_main(record=False, plx_mas=50.0, shift_deg=1.0):
    stars = catalogues.gcns("source_id, ra, dec, parallax", "parallax > %g" % plx_mas)
    bat = [dict(name="SWIFT " + r["Swift"].strip(), ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"]), r=max(1.0, 30.0 / float(r["S/N"])) / 60,
                ctp=(r["CName"] or "").strip(), typ=(r["Type"] or "").strip())
           for r in core.tap("vizier", 'SELECT Swift, RAJ2000, DEJ2000, "S/N", CName, Type FROM "J/ApJS/235/4/table3"')]
    bat += [dict(name="BAT " + r["BAT"].strip(), ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"]), r=num(r["R95"], 4.5) / 60, ctp=(r["Ctp"] or "").strip(),
                 typ=(r["Type"] or "").strip()) for r in core.tap("vizier", 'SELECT BAT, RAJ2000, DEJ2000, R95, Ctp, Type FROM "J/A+A/709/A12/catalog"')]
    bra, bde, brad = (np.array([b[k] for b in bat]) for k in ("ra", "dec", "r"))

    def inside(dra=0.0, dde=0.0):
        out = []
        for s in stars:
            de = float(s["dec"]) + dde
            ra = float(s["ra"]) + dra / max(math.cos(math.radians(de)), 0.05)
            if abs(de) > 90:
                ra, de = ra + 180, math.copysign(180, de) - de
            c = np.sin(np.radians(de)) * np.sin(np.radians(bde)) + np.cos(np.radians(de)) * np.cos(np.radians(bde)) * np.cos(np.radians(ra % 360 - bra))
            sep = np.degrees(np.arccos(np.clip(c, -1, 1)))
            out += [(s["source_id"], int(k), float(sep[k]) * 60) for k in np.nonzero(sep < brad)[0]]
        return out
    at = inside()
    null = [len({s for s, k, d in inside(x * shift_deg, y * shift_deg)}) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))]
    mean = float(np.mean(null))
    # the catalogue's counterpart: the star itself (named in SIMBAD under the counterpart's name) or another object
    sim = {}
    for s, k, d in at:
        if bat[k]["ctp"]:
            for r in core.simbad("SELECT b.main_id, b.otype FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id = '%s'" % bat[k]["ctp"].replace("'", "''")):
                sim[k] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip())
    own = {}
    for s, k, d in at:
        for r in core.simbad("SELECT b.main_id FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id = 'Gaia DR3 %s'" % s):
            own[s] = " ".join(str(r["main_id"]).split())
    rows = []
    for s, k, d in at:
        b = bat[k]
        same = b["ctp"] and k in sim and own.get(s) == sim[k][0]
        vet = "flare" if same else "background-galaxy" if b["typ"][:2] in ("Sy", "Bl", "BL", "FS", "QS", "LI", "Ga", "AG") or "Sy" in b["typ"] else (
            "natural-source" if b["ctp"] else None)
        rows.append(dict(star=s, simbad=own.get(s), bat=b["name"], sep_arcmin=round(d, 1), ctp=b["ctp"], typ=b["typ"], ctp_simbad=sim.get(k), vet=vet))
    left = [r for r in rows if r["vet"] is None]
    print("%d stars within 20 pc; inside a Swift-BAT error circle: %d stars %s; moved 1 deg %s (mean %.1f); without a named counterpart: %d" % (
        len(stars), len({r["star"] for r in rows}), [(r["simbad"], r["bat"], r["sep_arcmin"], r["ctp"], r["typ"], r["vet"]) for r in rows], null, mean, len(left)))
    if record:
        chance = len({r["star"] for r in left}) <= mean + 3 * math.sqrt(max(mean, 1))
        core.record_run("bat_nearest_stars", "Hard X-rays from the nearest stars", "Gaia Catalogue of Nearby Stars; Swift-BAT 105-month (J/ApJS/235/4) and "
                        "150-month (J/A+A/709/A12) catalogues; SIMBAD", "objects within 20 pc: %d" % len(stars), os.path.abspath(__file__), "nearbat",
                        len(stars), "stars", dict(radius_arcmin="30/(S/N), at least 1; R95 for 150-month", shift_deg=shift_deg, band_kev=[14, 195], plx_mas=plx_mas),
                        "the stars moved %g deg: %s" % (shift_deg, null), "steady 14-195 keV emission from a star within 20 pc",
                        "%d stars inside a BAT circle against a mean of %.1f moved; %d without a named counterpart" % (len({r["star"] for r in rows}), mean, len(left)),
                        [dict(target="gaia_dr3:%s" % r["star"], vet=r["vet"] or ("chance-rate" if chance else None),
                              note="%s; %s at %.1f arcmin, counterpart %s (%s)" % (r["simbad"], r["bat"], r["sep_arcmin"], r["ctp"] or "none", r["typ"] or "-"))
                         for r in rows])
        print("recorded")


etzxray_DOC = r"""X-rays from Earth's transit zone: Gaia Catalogue of Nearby Stars objects within 0.264 deg of the ecliptic, which see Earth cross the Sun, in the eRASS1 half of the sky (Galactic longitude 180-360 deg), carried to 2020.0, against eRASS1 main-catalogue sources (J/A+A/682/A34) within 15 arcsec by CDS XMatch; the share with an X-ray source in the zone against the share 1-5 deg from the ecliptic and against the zone moved 2 arcmin in eight directions. Matches read in SIMBAD.

    python -I seti.py highenergy etzxray [record] [--radius-arcsec 15] [--epoch 2020]"""


def etzxray_main(record=False, radius_arcsec=15.0, epoch=2020.0):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from setitools.neighbourhood import etz_share, etz_stars, etz_vets
    st = etz_stars()
    every = st["zone"] + st["near"]
    lon = SkyCoord(np.array([s["ra"] for s in every]) * u.deg, np.array([s["dec"] for s in every]) * u.deg).galactic.l.deg
    west = {s["id"] for s, x in zip(every, lon) if x >= 180}
    r = etz_share("J/A+A/682/A34/erass1-m", radius_arcsec, epoch, keep=lambda s: s["id"] in west)
    vets = etz_vets(r["got"])
    mean = float(np.mean(r["null"]))
    print("%d zone stars and %d stars 1-5 deg away in the eRASS1 half; eRASS1 within 15 arcsec: zone %d %s, away %d (%.2f expected in the zone, p %.3g); "
          "zone moved 2 arcmin %s (mean %.1f)" % (r["zone"], r["near"], len(r["got"]), [(n, vets[n][0], vets[n][1]) for n in sorted(r["got"])], r["far"],
                                                  r["exp"], r["p"], r["null"], mean))
    if record:
        core.record_run("etz_erass1_stars", "X-rays from Earth's transit zone", "Gaia Catalogue of Nearby Stars (external.gaiaedr3_gcns_main_1); eRASS1 main "
                        "catalogue (J/A+A/682/A34) by CDS XMatch; SIMBAD", "stars within 100 pc and 0.264 deg of the ecliptic at Galactic longitude 180-360 deg: %d"
                        % r["zone"], os.path.abspath(__file__), "etzxray", r["zone"], "stars", dict(zone_deg=0.264, control_deg=[1, 5], radius_arcsec=radius_arcsec,
                                                                                                     epoch=epoch), "the share 1-5 deg away (%d of %d); the zone moved 2 arcmin: %s" % (r["far"], r["near"], r["null"]),
                        "X-rays from a star that sees Earth transit, beyond the rate of stars beside the zone",
                        "%d against %.2f from the share beside the zone (p %.3g) and a mean of %.1f moved" % (len(r["got"]), r["exp"], r["p"], mean),
                        [dict(target="gaia_dr3:%s" % n[1:], vet=vets[n][1] or ("chance-rate" if r["p"] > 0.01 else None), note="SIMBAD %s; eRASS1 %s at %.1f arcsec" % (
                            vets[n][0], m.get("IAUName"), float(m["angDist"]))) for n, m in sorted(r["got"].items())])
        print("recorded")


deadstars_DOC = r"""Dead stars with living companions: Gaia DR3 astrometric and astrometric-spectroscopic binaries (binary_masses) whose unseen companion outweighs 1.4 solar masses at its lower bound with a flux ratio below 0.05 or none, and the published black holes Gaia BH1, BH2 and BH3; a neutron star or black hole in a wide orbit round an ordinary star is a power source, and a swarm feeding on it glows warm. AllWISE within 3 arcsec by CDS XMatch for W1-W3 redder than 0.3 mag with W3 measured to 0.2 mag, and eRASS1 within 15 arcsec, each against the stars moved 2 arcmin in eight directions; matches read in SIMBAD, for AllWISE's contamination and extension flags, for SIMBAD's young objects within 1 arcmin, and for the hot dust of an emission-line star (W1-W2 above 0.5) in the plane.

    python -I seti.py highenergy deadstars [record] [--m2-min 1.4] [--fluxratio-max 0.05] [--w13-min 0.3] [--allwise-arcsec 3] [--erass1-arcsec 15]"""

GAIA_BLACK_HOLES = {"4373465352415301632": "Gaia BH1", "5870569352746779008": "Gaia BH2", "4318465066420528000": "Gaia BH3"}


def deadstars_main(record=False, m2_min=1.4, fluxratio_max=0.05, w13_min=0.3, allwise_arcsec=3.0, erass1_arcsec=15.0):
    rows = core.tap_async("gaia", "SELECT b.source_id, b.m1, b.m2_lower, b.m2_upper, b.fluxratio, b.combination_method, g.ra, g.dec, g.parallax, g.phot_g_mean_mag, "
                                  "g.bp_rp FROM gaiadr3.binary_masses AS b JOIN gaiadr3.gaia_source AS g ON b.source_id = g.source_id WHERE b.m2_lower > %g AND "
                                  "(b.fluxratio < %g OR b.fluxratio IS NULL) AND b.combination_method IN ('Orbital+M1', 'AstroSpectroSB1+M1', 'Orbital+SB1+M1')"
                          % (m2_min, fluxratio_max), maxrec=5000)
    have = {r["source_id"] for r in rows}
    for sid, name in GAIA_BLACK_HOLES.items():
        if sid not in have:
            g = core.tap("gaia", "SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp FROM gaiadr3.gaia_source WHERE source_id = %s" % sid)[0]
            rows.append(dict(g, m1=None, m2_lower=None, m2_upper=None, fluxratio=None, combination_method=name))
    pos = [dict(name=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"])) for r in rows]
    warm = lambda m: (num(m.get("W3mag")) is not None and num(m.get("e_W3mag")) is not None and num(m["e_W3mag"]) < 0.2 and num(m.get("W1mag")) is not None
                      and num(m["W1mag"]) - num(m["W3mag"]) > w13_min)
    ir, ir_null = shifted_matches(pos, "II/328/allwise", allwise_arcsec, keyfn=warm)
    xr, xr_null = shifted_matches(pos, "J/A+A/682/A34/erass1-m", erass1_arcsec)
    hit = {}
    for m in ir:
        hit.setdefault(m["source_id"], {})["W1-W3"] = round(num(m["W1mag"]) - num(m["W3mag"]), 2)
        hit[m["source_id"]]["W1-W2"] = round(num(m["W1mag"]) - num(m["W2mag"]), 2) if num(m.get("W2mag")) is not None else None
        hit[m["source_id"]]["flags"] = ((m.get("ccf") or "").strip(), (m.get("ex") or "").strip())
    for m in xr:
        hit.setdefault(m["source_id"], {})["eRASS1"] = m.get("IAUName")
    # the dust of a young or emission-line star: its region's young objects within 1 arcmin, and the Galactic plane
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    young = ("Y*O", "Y*?", "TT*", "HII", "MoC", "DNe", "RNe", "HH", "out", "smm", "mm", "Cl*", "OpC", "cor", "Ae*", "Be*", "Or*", "EmO", "PoC", "SFR", "Cld", "glb", "Em*", "pA*")
    for n, h in hit.items():
        r = next(r for r in rows if r["source_id"] == n)
        kinds = core.simbad_types_near(float(r["ra"]), float(r["dec"]), 1 / 60.0)
        h["young_nearby"] = {k: v for k, v in kinds.items() if k in young}
        h["b"] = round(float(SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg).galactic.b.deg), 1)
    sim = {}
    if hit:
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % n for n in hit)):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    by = {r["source_id"]: r for r in rows}
    act = ("BY*", "RS*", "Er*", "Fl*", "TT*", "Y*O", "Y*?", "EB*", "**", "CV*", "Ro*", "El*", "Em*", "X", "Ae*", "Be*", "a2*", "AB*", "Mi*", "sr*", "LP*", "RG*", "s*r", "s*b", "SB*")
    out = []
    for n, h in hit.items():
        s = sim.get(n)
        vet = ("artefact" if h.get("flags") and (h["flags"][0].strip("0") or h["flags"][1] not in ("", "0")) else
               "active-star" if s and s[1] in act else
               "young-star" if h.get("young_nearby") else
               # hot dust of a Be or Herbig star: W1-W2 above 0.5 beside a W3 excess, in the plane where such stars live
               "dust" if h.get("W1-W2") is not None and h["W1-W2"] > 0.5 and abs(h["b"]) < 10 else None)
        out.append(dict(id=n, what=by[n]["combination_method"], m2=by[n]["m2_lower"], plx=by[n]["parallax"], g=by[n]["phot_g_mean_mag"], found=h, simbad=s, vet=vet))
    print("%d stars with an unseen companion above 1.4 solar masses (with Gaia BH1-3); warm W1-W3 within 3 arcsec: %d against moved %s; eRASS1 within 15 arcsec: %d against "
          "moved %s" % (len(rows), len(ir), ir_null, len(xr), xr_null))
    for o in out:
        print("  Gaia DR3 %s (%s, M2 above %s Msun, parallax %s, G %s): %s; SIMBAD %s; %s" % (o["id"], o["what"], o["m2"], o["plx"], o["g"], o["found"], o["simbad"], o["vet"]))
    if record:
        core.record_run("gaia_dead_companions", "Dead stars with living companions", "Gaia DR3 binary_masses and gaia_source; Gaia BH1-3; AllWISE (II/328) and eRASS1 "
                        "(J/A+A/682/A34) by CDS XMatch; SIMBAD", "stars with an unseen companion above 1.4 solar masses: %d" % len(rows), os.path.abspath(__file__),
                        "deadstars", len(rows), "stars", dict(m2_min=m2_min, fluxratio_max=fluxratio_max, w13_min=w13_min, radius_arcsec=dict(allwise=allwise_arcsec, erass1=erass1_arcsec)),
                        "the stars moved 2 arcmin: warm %s, X-ray %s" % (ir_null, xr_null), "waste heat or X-rays from a swarm feeding on a dead star",
                        "%d warm and %d X-ray matches against means of %.1f and %.1f moved; %d unexplained" % (len(ir), len(xr), float(np.mean(ir_null)), float(np.mean(xr_null)), sum(1 for o in out if o["vet"] is None)),
                        [dict(target="gaia_dr3:%s" % o["id"], vet=o["vet"], note="%s; %s; SIMBAD %s" % (o["what"], o["found"], o["simbad"])) for o in out])
        print("recorded")


gwequal_DOC = r"""Identical black holes merging: every compact-binary merger in the GWOSC event list (GWTC-1 to GWTC-5) with source-frame component masses and their 90 per cent bounds; the mass difference in units of the combined error, and the mass ratio of the events whose primary mass is known to 10 per cent. Manufactured objects of one design would merge at a mass ratio of exactly one beyond what the population's own preference for near-equal masses gives; the convention that the primary is the heavier piles posteriors at a ratio of one, so the test rests on the best-measured events.

    python -I seti.py highenergy gwequal [record] [--primary-error-max 0.1] [--ratio-error-max 0.05]"""


def gwequal_main(record=False, primary_error_max=0.1, ratio_error_max=0.05):
    ev = json.loads(core.get("https://gwosc.org/eventapi/json/GWTC/", timeout=120))["events"]
    rows = []
    for e in ev.values():
        m1, m2 = e.get("mass_1_source"), e.get("mass_2_source")
        b = [e.get(k) for k in ("mass_1_source_lower", "mass_1_source_upper", "mass_2_source_lower", "mass_2_source_upper")]
        if m1 is None or m2 is None or None in b:
            continue
        s1, s2 = (abs(b[0]) + abs(b[1])) / 2, (abs(b[2]) + abs(b[3])) / 2
        rows.append(dict(name=e["commonName"], cat=e["catalog.shortName"], m1=m1, m2=m2, s1=s1, s2=s2, dz=(m1 - m2) / math.hypot(s1, s2), q=m2 / m1,
                         sq=(m2 / m1) * math.hypot(s1 / m1, s2 / m2), frac=abs(b[0]) / m1))
    tight = sorted((r for r in rows if r["frac"] < primary_error_max), key=lambda r: r["dz"])
    within = sum(1 for r in rows if abs(r["dz"]) < 1)
    exact = [r for r in tight if r["q"] + r["sq"] >= 1 and r["sq"] < ratio_error_max]
    print("%d mergers with component masses; %d with the primary to 10 per cent; mass difference within one combined error for %d of %d (%.0f per cent); among the "
          "best measured, mass ratios %s; at one within a ratio error below 0.05: %d %s" % (
              len(rows), len(tight), within, len(rows), 100.0 * within / len(rows), [(r["name"], round(r["q"], 2), round(r["sq"], 2)) for r in tight],
              len(exact), [r["name"] for r in exact]))
    if record:
        core.record_run("gwtc_equal_masses", "Identical black holes merging", "GWOSC event list (GWTC-1 to GWTC-5)", "mergers with source-frame component masses: %d" % len(rows),
                        os.path.abspath(__file__), "gwequal", len(rows), "mergers", dict(primary_error_max=primary_error_max, ratio_error_max=ratio_error_max),
                        "the population's own pile-up at a ratio of one: %d of %d within one combined error" % (within, len(rows)),
                        "a merger whose components are equal to a few per cent beyond the population's preference",
                        "%d of %d best-measured events at a ratio of one within an error below 0.05" % (len(exact), len(tight)),
                        [dict(target="gw:%s" % r["name"], stat=round(r["q"], 3), unit="mass ratio", vet="natural-source" if r["q"] + r["sq"] < 1 else "chance-rate",
                              note="%s; %.1f and %.1f Msun, ratio %.2f +- %.2f" % (r["cat"], r["m1"], r["m2"], r["q"], r["sq"])) for r in tight])
        print("recorded")


tevnear_DOC = r"""TeV light at the nearest stars: the 1LHAASO catalogue's point-like sources (extension 0, 90 sources in all) without an association, against Gaia Catalogue of Nearby Stars objects within 20 pc inside each source's position error plus 0.1 deg, against the same count with the sources moved 2 deg in eight directions; nothing within 20 pc makes TeV photons but a flare star's corona cannot, so a star inside a TeV error circle is a candidate emitter at the energies an engine would reach. A source the catalogue associates is its counterpart's.

    python -I seti.py highenergy tevnear [record] [--plx-mas 50] [--extra-deg 0.1] [--shift-deg 2]"""


def tevnear_main(record=False, plx_mas=50.0, extra_deg=0.1, shift_deg=2.0):
    src = core.tap("vizier", 'SELECT "1LHAASO", "f_1LHAASO", Comp, RAJ2000, DEJ2000, ePos, r39, TS, Assoc FROM "J/ApJS/271/25/catalog"')
    stars = catalogues.gcns("source_id, ra, dec, parallax, phot_g_mean_mag", "parallax > %g AND gcns_prob > 0.9" % plx_mas)
    sra, sde = np.array([float(s["ra"]) for s in stars]), np.array([float(s["dec"]) for s in stars])
    point = [r for r in src if num(r["r39"]) == 0]
    free = [r for r in point if not (r["Assoc"] or "").strip()]

    def inside(rows, dra=0.0, dde=0.0):
        out = []
        for r in rows:
            ra, de = float(r["RAJ2000"]) + dra / max(math.cos(math.radians(float(r["DEJ2000"]))), 0.05), float(r["DEJ2000"]) + dde
            if abs(de) > 90:
                ra, de = ra + 180, math.copysign(180, de) - de
            rad = (num(r["ePos"]) or 0.3) + extra_deg
            c = np.sin(np.radians(de)) * np.sin(np.radians(sde)) + np.cos(np.radians(de)) * np.cos(np.radians(sde)) * np.cos(np.radians(ra % 360 - sra))
            sep = np.degrees(np.arccos(np.clip(c, -1, 1)))
            out += [(r["1LHAASO"].strip(), stars[k]["source_id"], round(float(sep[k]), 2)) for k in np.nonzero(sep < rad)[0]]
        return out
    at = inside(free)
    null = [len(inside(free, x * shift_deg, y * shift_deg)) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))]
    owned = inside([r for r in point if (r["Assoc"] or "").strip()])
    print("%d 1LHAASO sources, %d point-like, %d of them unassociated; stars within 20 pc inside an unassociated circle: %d %s against moved %s; inside an associated one: %d %s" % (
        len(src), len(point), len(free), len(at), at, null, len(owned), owned))
    if record:
        mean = float(np.mean(null))
        core.record_run("lhaaso_nearest_stars", "TeV light at the nearest stars", "1LHAASO catalogue (J/ApJS/271/25); Gaia Catalogue of Nearby Stars",
                        "point-like unassociated 1LHAASO sources: %d; stars within 20 pc: %d" % (len(free), len(stars)), os.path.abspath(__file__), "tevnear", len(free),
                        "TeV sources", dict(extension=0, radius="ePos + %g deg" % extra_deg, shift_deg=shift_deg, plx_mas=plx_mas), "the sources moved %g deg: %s" % (shift_deg, null),
                        "a star within 20 pc inside the error circle of a TeV source nothing else explains", "%d against a mean of %.1f moved; %d inside associated sources" % (
                            len(at), mean, len(owned)),
                        [dict(target="gaia_dr3:%s" % g, vet="chance-rate" if len(at) <= mean + 3 * math.sqrt(max(mean, 1)) else None, note="%s at %.2f deg" % (n, sep)) for n, g, sep in at] +
                        [dict(target="gaia_dr3:%s" % g, vet="natural-source", note="%s at %.2f deg, associated" % (n, sep)) for n, g, sep in owned])
        print("recorded")


bursts511_DOC = r"""Bursts peaking at the electron's mass: Fermi GBM bursts (HEASARC fermigbrst) whose fluence Band fit gives a peak energy measured to 10 per cent; the count within 2 per cent of 511 keV against the count a smooth run of the peak-energy distribution gives, from the bursts within 0.3 dex either side, and the same for the Comptonised fit. Annihilation radiation peaks at the electron's mass; an engine burning antimatter would send it, and no burst of a collapsing star prefers that energy.

    python -I seti.py highenergy bursts511 [record] [--window 0.02] [--error-max 0.1] [--smooth-dex 0.3]"""

ELECTRON_KEV = 510.999


def bursts511_main(record=False, window=0.02, error_max=0.1, smooth_dex=0.3):
    rows = core.heasarc("SELECT name, trigger_name, t90, flnc_band_epeak, flnc_band_epeak_pos_err, flnc_band_epeak_neg_err, flnc_comp_epeak, flnc_comp_epeak_pos_err, "
                   "flnc_comp_epeak_neg_err FROM fermigbrst WHERE flnc_band_epeak > 0 OR flnc_comp_epeak > 0")
    out = {}
    for key in ("band", "comp"):
        ep = np.array([core.num(r["flnc_%s_epeak" % key]) or np.nan for r in rows])
        err = np.array([max(abs(core.num(r["flnc_%s_epeak_pos_err" % key]) or np.nan), abs(core.num(r["flnc_%s_epeak_neg_err" % key]) or np.nan)) for r in rows])
        ok = np.isfinite(ep) & np.isfinite(err) & (ep > 0) & (err < error_max * ep)
        le = np.log10(ep[ok])
        at = np.sum(np.abs(np.log10(ep[ok] / ELECTRON_KEV)) < math.log10(1 + window))
        # the smooth expectation: the density of peak energies within smooth_dex of the line, spread over the window
        near = np.sum(np.abs(le - math.log10(ELECTRON_KEV)) < smooth_dex)
        exp = near * (2 * math.log10(1 + window)) / (2 * smooth_dex)
        from scipy.stats import poisson
        names = [r["name"] for r, k in zip(rows, ok) if k and abs(math.log10((core.num(r["flnc_%s_epeak" % key]) or 1) / ELECTRON_KEV)) < math.log10(1 + window)]
        out[key] = dict(n=int(ok.sum()), at=int(at), exp=float(exp), p=float(poisson.sf(at - 1, exp)) if at else 1.0, names=names)
        print("%s fit: %d bursts with the peak to %g per cent; %d within %g per cent of 511 keV against %.1f from the neighbouring %g dex (p %.3f) %s" % (
            key, out[key]["n"], 100 * error_max, at, 100 * window, exp, 2 * smooth_dex, out[key]["p"], names[:12]))
    if record:
        core.record_run("gbm_bursts_511kev", "Bursts peaking at the electron's mass", "Fermi GBM burst catalogue (HEASARC fermigbrst)",
                        "bursts with a Band peak energy to 10 per cent: %d; Comptonised: %d" % (out["band"]["n"], out["comp"]["n"]), os.path.abspath(__file__), "bursts511",
                        out["band"]["n"], "bursts", dict(window=window, error_max=error_max, smooth_dex=smooth_dex), "the smooth run of peak energies within %g dex of the line" % smooth_dex,
                        "an excess of bursts peaking at 511 keV", "; ".join("%s %d against %.1f (p %.3f)" % (k, v["at"], v["exp"], v["p"]) for k, v in out.items()),
                        [dict(target="grb:%s" % n, vet="chance-rate" if out[k]["p"] > 0.01 else None, note="%s fit peak within 2 per cent of 511 keV" % k)
                         for k, v in out.items() for n in v["names"]])
        print("recorded")


grbrepeat_DOC = r"""Gamma-ray bursts repeating at one position: Swift bursts (HEASARC swiftgrb) with XRT positions, pairs within the sum of their position errors plus 5 arcsec, against the pairs the sky density gives at random; nothing natural makes two gamma-ray bursts at one spot except a magnetar's giant flares in a nearby galaxy, which carry a host name in the catalogue.

    python -I seti.py highenergy grbrepeat [record] [--extra-arcsec 5] [--default-err 5]"""


def grbrepeat_main(record=False, extra_arcsec=5.0, default_err=5.0):
    rows = [r for r in core.heasarc("SELECT name, xrt_ra, xrt_dec, xrt_pos_err, galaxy_name, redshift, duration FROM swiftgrb WHERE xrt_ra IS NOT NULL AND xrt_dec IS NOT NULL")
            if core.num(r["xrt_ra"]) is not None and core.num(r["xrt_dec"]) is not None]
    ra = np.radians(np.array([float(r["xrt_ra"]) for r in rows]))
    de = np.radians(np.array([float(r["xrt_dec"]) for r in rows]))
    err = np.array([core.num(r["xrt_pos_err"]) or default_err for r in rows])
    xyz = np.stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)], axis=1)
    pairs = []
    for i in range(len(rows)):
        c = np.clip(xyz[i + 1:] @ xyz[i], -1, 1)
        sep = np.degrees(np.arccos(c)) * 3600
        tol = err[i] + err[i + 1:] + extra_arcsec
        for j in np.nonzero(sep < tol)[0]:
            pairs.append((rows[i]["name"], rows[i + 1 + j]["name"], round(float(sep[j]), 1), rows[i]["galaxy_name"], rows[i + 1 + j]["galaxy_name"]))
    # chance pairs: every pair's chance of falling within its tolerance on an isotropic sky
    tol_all = err[:, None] + err[None, :] + extra_arcsec
    iu = np.triu_indices(len(rows), 1)
    expected = float(np.sum(np.pi * np.radians(tol_all[iu] / 3600) ** 2 / (4 * np.pi)))
    print("%d Swift bursts with XRT positions; %d pairs within their position errors %s against %.4f by chance" % (len(rows), len(pairs), pairs, expected))
    if record:
        from scipy.stats import poisson
        core.record_run("swift_repeating_bursts", "Gamma-ray bursts repeating at one position", "Swift GRB catalogue (HEASARC swiftgrb), XRT positions",
                        "bursts with XRT positions: %d" % len(rows), os.path.abspath(__file__), "grbrepeat", len(rows), "bursts",
                        dict(tolerance="xrt_pos_err sum + %g arcsec" % extra_arcsec, default_err_arcsec=default_err),
                        "pairs on an isotropic sky: %.4f" % expected, "two gamma-ray bursts at one position", "%d pairs against %.4f (p %.3g)" % (
                            len(pairs), expected, float(poisson.sf(len(pairs) - 1, expected)) if pairs else 1.0),
                        [dict(target="grb:%s" % a, vet="natural-source" if (ga or gb) else None, note="with %s at %.1f arcsec; hosts %s %s" % (b, sep, ga, gb))
                         for a, b, sep, ga, gb in pairs])
        print("recorded")


coldbdw3_DOC = r"""Twelve-micron light at cold brown dwarfs: the T and Y dwarfs of the Kirkpatrick et al. (2021) 20 pc census with Spitzer astrometry, carried to 2010.5, against AllWISE (II/328) by CDS XMatch within 5 arcsec for a W3 detection at S/N 3, against the same match with the dwarfs moved 2 arcmin in eight directions, and against the share of field sources of the dwarfs' own W1 and W2 that the catalogue gives a W3 at S/N 3, since a source it holds at W1 and W2 is measured at W3 whether or not anything is there; a body below 1300 K gives nothing at 12 um that AllWISE reaches, so a detection beyond that share is something warmer and larger around it.

    python -I seti.py highenergy coldbdw3 [record] [--radius-arcsec 5] [--epoch 2010.5] [--w3-err-max 0.36] [--field-w1 14 18] [--field-w12-min 1.2]"""


def coldbdw3_main(record=False, radius_arcsec=5.0, epoch=2010.5, w3_err_max=0.36, field_w1=[14.0, 18.0], field_w12_min=1.2):
    bd = cold_dwarfs(epoch)
    hits, null = shifted_matches(bd, "II/328/allwise", radius_arcsec,
                                 keyfn=lambda m: num(m.get("W3mag")) is not None and num(m.get("e_W3mag")) is not None and num(m["e_W3mag"]) < w3_err_max)
    names = sorted({m["source_id"] for m in hits})
    mean = float(np.mean(null))
    # the catalogue's own habit: a faint red source it holds at W1 and W2 carries a W3 at S/N 3 in a fixed share of cases, measured on field sources of the dwarfs' colours
    rng = np.random.default_rng(3)
    up = []
    for k, (ra0, de0) in enumerate(((30, 40), (150, 20), (200, -30), (330, -45), (90, 60), (250, 45), (10, -60), (120, -10))):
        for i in range(150):
            up.append(dict(source_id="p%d_%d" % (k, i), ra=ra0 + rng.uniform(-1, 1) / math.cos(math.radians(de0)), dec=de0 + rng.uniform(-1, 1)))
    field = {}
    for m in core.xmatch(up, "II/328/allwise", 90.0):
        w1, w2 = num(m.get("W1mag")), num(m.get("W2mag"))
        if w1 is not None and w2 is not None and field_w1[0] <= w1 <= field_w1[1] and w1 - w2 > field_w12_min:
            field[m["AllWISE"]] = num(m.get("e_W3mag")) is not None and num(m["e_W3mag"]) < w3_err_max
    share = sum(field.values()) / max(len(field), 1)
    from scipy.stats import binom
    p = float(binom.sf(len(names) - 1, len(bd), share)) if names else 1.0
    print("%d T and Y dwarfs; AllWISE W3 at S/N 3 within 5 arcsec: %d; moved 2 arcmin %s (mean %.1f); field sources of the dwarfs' colours carry one in %.3f of cases "
          "(%d of %d), so %.1f expected (p %.3f) %s" % (len(bd), len(names), null, mean, share, sum(field.values()), len(field), share * len(bd), p,
                                                         [(m["source_id"], m.get("W3mag"), m.get("e_W3mag"), m.get("W1mag"), m.get("ccf")) for m in hits][:8]))
    if record:
        core.record_run("allwise_w3_cold_brown_dwarfs", "Twelve-micron light at cold brown dwarfs", "Kirkpatrick et al. (2021) 20 pc census; AllWISE (II/328) by CDS XMatch",
                        "T and Y dwarfs with Spitzer astrometry: %d" % len(bd), os.path.abspath(__file__), "coldbdw3", len(bd), "brown dwarfs",
                        dict(radius_arcsec=radius_arcsec, epoch=epoch, w3_err_max=w3_err_max, field_w1=field_w1, field_w12_min=field_w12_min),
                        "the share of field sources of the dwarfs' colours carrying a W3 at S/N 3: %.3f (%d of %d); the dwarfs moved 2 arcmin: %s" % (
                            share, sum(field.values()), len(field), null), "12 um light at a body too cold to give it",
                        "%d against %.1f at the field share (p %.3f)" % (len(names), share * len(bd), p),
                        [dict(target="bd:%s" % n, vet="chance-rate" if p > 0.01 else None, note="a W3 the catalogue gives faint red sources in %.0f per cent of cases" % (100 * share))
                         for n in names])
        print("recorded")


gammapeak_DOC = r"""Gamma rays peaked like a line: Fermi-LAT point sources (HEASARC's copy of the 4FGL catalogue) detected above 10 sigma whose log-parabola spectrum curves with beta above 0.8 at a curvature significance above 4, a spectrum so sharply peaked that a single energy, which a particle annihilation or a made beam would give, fits better than any power law or cutoff, against the catalogue's own curvatures; each read for its association and source class, since pulsars and blazars curve by nature, and for its analysis flags.

    python -I seti.py highenergy gammapeak [record] [--sig-min 10] [--beta-min 0.8] [--curve-min 4]"""


def gammapeak_main(record=False, sig_min=10.0, beta_min=0.8, curve_min=4.0):
    rows = core.heasarc("SELECT name, ra, dec, lii, bii, detection_significance, spectrum_type, pl_index, lp_index, lp_beta, lp_beta_error, lp_curve_significance, lp_epeak, "
                        "plec_curve_significance, variability_index, source_type, assoc_name, analysis_flags FROM fermilpsc WHERE detection_significance > %g AND lp_beta "
                        "IS NOT NULL" % sig_min, 20000)
    betas = np.array([num(r["lp_beta"]) for r in rows if num(r["lp_beta"]) is not None])
    hits = [r for r in rows if (num(r["lp_beta"]) or 0) > beta_min and (num(r["lp_curve_significance"]) or 0) > curve_min]
    out = []
    for r in hits:
        cls = str(r["source_type"] or "").strip()
        flags = str(r["analysis_flags"] or "").strip()
        out.append(dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), b=num(r["bii"]), sig=num(r["detection_significance"]), beta=num(r["lp_beta"]),
                        beta_err=num(r["lp_beta_error"]), curve=num(r["lp_curve_significance"]), epeak=num(r["lp_epeak"]), spec=str(r["spectrum_type"] or "").strip(),
                        var=num(r["variability_index"]), cls=cls, assoc=str(r["assoc_name"] or "").strip(), flags=flags,
                        vet="natural-type" if cls else "artefact" if flags not in ("", "0") else None))
    left = [o for o in out if o["vet"] is None]
    print("%d sources above %g sigma with a log-parabola curvature; beta median %.2f, 99th percentile %.2f; beta above %g at %g sigma of curvature: %d, %d of a known class, "
          "%d flagged; unexplained %d" % (len(rows), sig_min, float(np.median(betas)) if len(betas) else 0, float(np.percentile(betas, 99)) if len(betas) else 0, beta_min,
                                           curve_min, len(out), sum(o["vet"] == "natural-type" for o in out), sum(o["vet"] == "artefact" for o in out), len(left)))
    for o in sorted(out, key=lambda o: -(o["beta"] or 0)):
        print("  %s (b %+.1f): %.1f sigma, beta %.2f +- %s at %.1f sigma, peak %s MeV, %s, variability %s, class %s %s, flags %s; %s" % (
            o["name"], o["b"] or 0, o["sig"], o["beta"], o["beta_err"], o["curve"], o["epeak"], o["spec"], o["var"], o["cls"] or "-", o["assoc"] or "", o["flags"] or "-",
            o["vet"] or "unexplained"))
    json.dump(out, open(core.results("gamma_peaked_spectra.json"), "w"), indent=1)
    if record:
        core.record_run(
            "fermi_peaked_spectra", "Gamma rays peaked like a line", "Fermi-LAT 4FGL source catalogue (HEASARC fermilpsc)",
            "sources above %g sigma with a log-parabola curvature: %d" % (sig_min, len(rows)), os.path.abspath(__file__), "gammapeak", len(rows), "sources",
            dict(sig_min=sig_min, beta_min=beta_min, curve_min=curve_min), "the catalogue's curvatures: median %.2f, 99th percentile %.2f" % (
                float(np.median(betas)) if len(betas) else 0, float(np.percentile(betas, 99)) if len(betas) else 0),
            "a spectrum curving with beta above %g at %g sigma, peaked within a factor of a few in energy" % (beta_min, curve_min),
            "%d: %d of a known class, %d flagged; %d unexplained" % (len(out), sum(o["vet"] == "natural-type" for o in out), sum(o["vet"] == "artefact" for o in out), len(left)),
            [dict(target="fermi:%s" % o["name"], stat=o["beta"], unit="log-parabola beta", vet=o["vet"],
                  note="%.1f sigma, curvature %.1f sigma, peak %s MeV, class %s %s, variability %s, flags %s" % (
                      o["sig"], o["curve"], o["epeak"], o["cls"] or "-", o["assoc"], o["var"], o["flags"] or "-")) for o in out])
        print("recorded")


grbfrb_DOC = r"""Gamma rays with fast radio bursts: the Fermi GBM burst catalogue (HEASARC fermigbrst) against CHIME/FRB Catalog 1, pairs of a gamma-ray burst and a radio burst within 100 seconds of each other whose positions overlap within the gamma-ray error circle plus a degree, a message sent twice at once in two bands, against the pairs the two catalogues' rates give by chance in the same window; each read for the gamma-ray burst's duration and error circle.

    python -I seti.py highenergy grbfrb [record] [--window-s 100] [--extra-deg 1]"""


def grbfrb_main(record=False, window_s=100.0, extra_deg=1.0):
    from scipy.stats import poisson
    grbs = core.heasarc("SELECT name, ra, dec, error_radius, trigger_time, t90 FROM fermigbrst WHERE trigger_time > 0", 20000)
    frbs = catalogues.frb_catalog1("Name, RpName, RAJ2000, DEJ2000, MJD400, MJDinf, DMfitb")
    g = [dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]) or 0.0, t=num(r["trigger_time"]), t90=num(r["t90"])) for r in grbs]
    g = [x for x in g if x["t"] and x["ra"] is not None]
    f = [dict(name=str(r["Name"]).strip(), rep=str(r["RpName"] or "").strip().replace("-9999", ""), ra=num(r["RAJ2000"]), dec=num(r["DEJ2000"]),
              t=num(r["MJDinf"]) or num(r["MJD400"]), dm=num(r["DMfitb"])) for r in frbs]
    f = [x for x in f if x["t"] and x["ra"] is not None]
    w = window_s / 86400.0
    gt = np.array([x["t"] for x in g])
    order = np.argsort(gt)
    gt = gt[order]
    g = [g[i] for i in order]
    close, overlap = [], []
    for x in f:
        lo, hi = np.searchsorted(gt, x["t"] - w), np.searchsorted(gt, x["t"] + w)
        for i in range(lo, hi):
            y = g[i]
            sep = core.sep_arcsec(x["ra"], x["dec"], y["ra"], y["dec"]) / 3600.0
            pair = dict(frb=x["name"], grb=y["name"], dt_s=(x["t"] - y["t"]) * 86400.0, sep_deg=sep, err_deg=y["err"], t90=y["t90"], dm=x["dm"], rep=x["rep"])
            close.append(pair)
            if sep <= y["err"] + extra_deg:
                overlap.append(pair)
    span = (gt.max() - gt.min()) if len(gt) > 1 else 1.0
    frbs_in = [x for x in f if gt.min() <= x["t"] <= gt.max()]
    exp_time = len(frbs_in) * len(gt) * 2 * w / span
    # the chance a random gamma-ray error circle plus the margin covers a given radio position
    cover = float(np.mean([(1 - math.cos(math.radians(min(y["err"] + extra_deg, 180.0)))) / 2 for y in g]))
    exp = exp_time * cover
    p = float(poisson.sf(len(overlap) - 1, exp)) if overlap else 1.0
    for o in overlap:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d gamma-ray bursts over %.1f years and %d radio bursts, %d inside the gamma-ray span; pairs within %g s: %d (%.2f expected), overlapping in position: %d (%.3f expected, "
          "Poisson p %.3g)" % (len(g), span / 365.25, len(f), len(frbs_in), window_s, len(close), exp_time, len(overlap), exp, p))
    for o in sorted(close, key=lambda o: abs(o["dt_s"])):
        print("  %s and %s: %+.1f s apart, %.1f deg apart against an error circle of %.1f deg, T90 %s s, DM %s%s; %s" % (
            o["frb"], o["grb"], o["dt_s"], o["sep_deg"], o["err_deg"], o["t90"], o["dm"], " (repeater %s)" % o["rep"] if o["rep"] else "",
            o.get("vet") or ("overlapping" if o in overlap else "apart on the sky")))
    json.dump(dict(n_grb=len(g), n_frb=len(f), close=close, overlap=overlap), open(core.results("grb_frb_pairs.json"), "w"), indent=1)
    if record:
        core.record_run(
            "fermi_chime_coincidences", "Gamma rays with fast radio bursts", "Fermi GBM burst catalogue (HEASARC fermigbrst); CHIME/FRB Catalog 1",
            "radio bursts inside the gamma-ray catalogue's span: %d against %d gamma-ray bursts" % (len(frbs_in), len(g)), os.path.abspath(__file__), "grbfrb", len(frbs_in), "bursts",
            dict(window_s=window_s, extra_deg=extra_deg), "the two catalogues' rates give %.2f pairs in time and %.3f in time and position by chance" % (exp_time, exp),
            "a gamma-ray burst within %g s of a radio burst inside its error circle plus %g deg" % (window_s, extra_deg),
            "%d pairs in time, %d also overlapping in position against %.3f expected (Poisson p %.3g)" % (len(close), len(overlap), exp, p),
            [dict(target="frb:%s" % o["frb"], stat=o["dt_s"], unit="seconds after %s" % o["grb"], vet=o.get("vet") if o in overlap else "off-source",
                  note="%.1f deg apart against %.1f deg, T90 %s s, DM %s" % (o["sep_deg"], o["err_deg"], o["t90"], o["dm"])) for o in close])
        print("recorded")


nocorona_DOC = r"""Spinning stars without coronae: Gaia DR3 stars within 20 pc with a rotation period below 5 days (vari_rotation_modulation), whose dynamos saturate and whose X-rays reach 1e28 erg/s and more, in the half of the sky eROSITA's first survey released (Galactic longitude 180-360 deg), carried to 2020.0 and matched to the eRASS1 main catalogue (VizieR J/A+A/682/A34) within 15 arcsec; a fast rotator without X-rays has a corona switched off, which a star does not do and a shell around it would. Each unmatched star read for its brightness and colour, since the faintest M dwarfs sit at the survey's limit, and in SIMBAD.

    python -I seti.py highenergy nocorona [record] [--plx-min 50] [--period-max-d 5] [--radius-arcsec 15]"""


def nocorona_main(record=False, plx_min=50.0, period_max_d=5.0, radius_arcsec=15.0):
    rows = core.gaia_by_range("SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax, g.phot_g_mean_mag, g.bp_rp, g.l, v.best_rotation_period, v.best_rotation_period_error "
                              "FROM gaiadr3.gaia_source AS g JOIN gaiadr3.vari_rotation_modulation AS v ON g.source_id = v.source_id WHERE g.source_id BETWEEN {lo} AND {hi} "
                              "AND g.parallax > %g AND g.parallax_over_error > 10 AND v.best_rotation_period < %g AND g.l > 180" % (plx_min, period_max_d), n=4, maxrec=50000)
    st = [dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), pmra=num(r["pmra"]) or 0.0, pmdec=num(r["pmdec"]) or 0.0, plx=num(r["parallax"]),
               g=num(r["phot_g_mean_mag"]), c=num(r["bp_rp"]), l=num(r["l"]), period=num(r["best_rotation_period"])) for r in rows]
    up = []
    for s in st:
        ra, de = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, 2020.0)
        up.append(dict(source_id=s["id"], ra=ra % 360, dec=de))
    got = {}
    for k in range(0, len(up), 5000):
        for m in core.xmatch(up[k:k + 5000], "J/A+A/682/A34/erass1-m", radius_arcsec):
            if m["source_id"] not in got or float(m["angDist"]) < got[m["source_id"]][0]:
                got[m["source_id"]] = (float(m["angDist"]), m.get("IAUName"))
    missing = [s for s in st if s["id"] not in got]
    sim = {}
    for k in range(0, len(missing), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in missing[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in missing:
        s["simbad"] = sim.get(s["id"])
        # the X-rays of a saturated dwarf scale with its light: a star fainter than G 14 at 20 pc sits at the survey's limit
        s["vet"] = "not-measured" if (s["g"] or 0) > 14 else None
    left = [s for s in missing if s["vet"] is None]
    print("%d stars within %.0f pc with rotation periods below %g d in the eROSITA-DE sky; with an eRASS1 source within %g arcsec: %d; without: %d, %d of them fainter than G 14 at "
          "the survey's limit; unexplained %d" % (len(st), 1000 / plx_min, period_max_d, radius_arcsec, len(got), len(missing), sum(s["vet"] == "not-measured" for s in missing),
                                                  len(left)))
    for s in sorted(missing, key=lambda s: s["g"] or 0):
        print("  Gaia DR3 %s: P %.2f d, G %.2f, BP-RP %s, %.1f pc, l %.1f; SIMBAD %s; %s" % (s["id"], s["period"], s["g"], s["c"], 1000 / s["plx"], s["l"], s["simbad"],
                                                                                           s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), detected=len(got), missing=missing), open(core.results("rotators_without_xrays.json"), "w"), indent=1)
    if record:
        core.record_run(
            "erass1_rotators_without_xrays", "Spinning stars without coronae", "Gaia DR3 vari_rotation_modulation and gaia_source; eRASS1 main catalogue (VizieR J/A+A/682/A34) by "
            "CDS XMatch; SIMBAD", "stars within %.0f pc with rotation periods below %g d in the eROSITA-DE sky: %d" % (1000 / plx_min, period_max_d, len(st)),
            os.path.abspath(__file__), "nocorona", len(st), "stars", dict(plx_min=plx_min, period_max_d=period_max_d, radius_arcsec=radius_arcsec, epoch=2020.0),
            "the %d rotators the survey detects; the faintest M dwarfs at its limit" % len(got), "a fast rotator within %.0f pc without an eRASS1 source" % (1000 / plx_min),
            "%d of %d undetected, %d of them at the survey's limit; %d unexplained" % (len(missing), len(st), sum(s["vet"] == "not-measured" for s in missing), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=s["period"], unit="rotation period, d; no eRASS1 source", vet=s["vet"],
                  note="G %.2f, BP-RP %s, %.1f pc; SIMBAD %s" % (s["g"], s["c"], 1000 / s["plx"], s["simbad"])) for s in missing])
        print("recorded")


nugrb_DOC = r"""Neutrinos with gamma-ray bursts: the IceCube ten-year point-source event list (HEASARC icecubepsc) above 10 TeV against the Fermi GBM burst catalogue, a neutrino within 100 seconds of a gamma-ray burst and inside the two error circles plus a degree, two messengers sent together, against the pairs the event rate gives by chance over the same windows and circles; each pair read for the neutrino's energy and error circle and the burst's duration.

    python -I seti.py highenergy nugrb [record] [--energy-min-gev 10000] [--window-s 100] [--extra-deg 1]"""


def nugrb_main(record=False, energy_min_gev=10000.0, window_s=100.0, extra_deg=1.0):
    from scipy.stats import poisson
    # the service returns at most 100,000 rows a query, so the list is read in five spans of time
    bounds = core.heasarc("SELECT MIN(time) AS a, MAX(time) AS b FROM icecubepsc", 5)[0]
    edges = np.linspace(float(bounds["a"]) - 1, float(bounds["b"]) + 1, 6)
    nus = []
    for a, b in zip(edges[:-1], edges[1:]):
        nus += core.heasarc("SELECT event_number, time, ra, dec, error_radius, event_energy FROM icecubepsc WHERE event_energy > %g AND time >= %.5f AND time < %.5f" % (
            energy_min_gev, a, b), 100000)
    grbs = core.heasarc("SELECT name, ra, dec, error_radius, trigger_time, t90 FROM fermigbrst WHERE trigger_time > 0", 20000)
    nu = sorted([dict(n=str(r["event_number"]), t=num(r["time"]), ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]) or 0.0, e=num(r["event_energy"])) for r in nus
                 if num(r["time"]) and num(r["ra"]) is not None], key=lambda x: x["t"])
    nt = np.array([x["t"] for x in nu])
    span_lo, span_hi = nt.min(), nt.max()
    g = [dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]) or 0.0, t=num(r["trigger_time"]), t90=num(r["t90"])) for r in grbs]
    g = [x for x in g if x["t"] and x["ra"] is not None and span_lo <= x["t"] <= span_hi]
    w = window_s / 86400.0
    close, overlap, cover = [], [], 0.0
    for y in g:
        lo, hi = np.searchsorted(nt, y["t"] - w), np.searchsorted(nt, y["t"] + w)
        for i in range(lo, hi):
            x = nu[i]
            sep = core.sep_arcsec(x["ra"], x["dec"], y["ra"], y["dec"]) / 3600.0
            r = min(x["err"] + y["err"] + extra_deg, 180.0)
            pair = dict(grb=y["name"], nu=x["n"], dt_s=(x["t"] - y["t"]) * 86400.0, sep_deg=sep, circle_deg=r, energy_gev=x["e"], nu_err=x["err"], grb_err=y["err"], t90=y["t90"])
            close.append(pair)
            cover += (1 - math.cos(math.radians(r))) / 2
            if sep <= r:
                overlap.append(pair)
    rate = len(nu) / (span_hi - span_lo)
    exp_time = len(g) * rate * 2 * w
    p = float(poisson.sf(len(overlap) - 1, cover)) if overlap else 1.0
    for o in overlap:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d neutrinos above %g GeV over %.1f years and %d gamma-ray bursts in that span; neutrinos within %g s of a burst: %d (%.1f expected from the rate), inside the circles "
          "too: %d (%.3f expected from the circles' sky fraction, Poisson p %.3g)" % (len(nu), energy_min_gev, (span_hi - span_lo) / 365.25, len(g), window_s, len(close), exp_time,
                                                                                     len(overlap), cover, p))
    for o in sorted(overlap, key=lambda o: o["sep_deg"] / o["circle_deg"]):
        print("  %s and neutrino %s: %+.1f s, %.1f deg apart within a %.1f deg circle, %.0f GeV, errors %.2f and %.2f deg, T90 %s s; %s" % (
            o["grb"], o["nu"], o["dt_s"], o["sep_deg"], o["circle_deg"], o["energy_gev"], o["nu_err"], o["grb_err"], o["t90"], o["vet"] or "unexplained"))
    json.dump(dict(n_nu=len(nu), n_grb=len(g), close=len(close), overlap=overlap), open(core.results("neutrino_grb_pairs.json"), "w"), indent=1)
    if record:
        core.record_run(
            "icecube_fermi_coincidences", "Neutrinos with gamma-ray bursts", "IceCube ten-year point-source events (HEASARC icecubepsc); Fermi GBM burst catalogue (HEASARC fermigbrst)",
            "gamma-ray bursts inside the neutrino list's span: %d against %d neutrinos above %g GeV" % (len(g), len(nu), energy_min_gev), os.path.abspath(__file__), "nugrb", len(g),
            "bursts", dict(energy_min_gev=energy_min_gev, window_s=window_s, extra_deg=extra_deg),
            "the event rate gives %.1f neutrinos in the windows and %.3f inside the circles by chance" % (exp_time, cover),
            "a neutrino above %g GeV within %g s of a burst inside the two error circles plus %g deg" % (energy_min_gev, window_s, extra_deg),
            "%d in the windows, %d inside the circles against %.3f expected (Poisson p %.3g)" % (len(close), len(overlap), cover, p),
            [dict(target="grb:%s" % o["grb"], stat=o["dt_s"], unit="seconds to neutrino %s" % o["nu"], vet=o["vet"],
                  note="%.1f deg apart within %.1f deg, %.0f GeV, T90 %s s" % (o["sep_deg"], o["circle_deg"], o["energy_gev"], o["t90"])) for o in overlap])
        print("recorded")


nuetz_DOC = r"""Neutrinos from Earth's transit zone: the IceCube ten-year point-source events above 10 TeV (HEASARC icecubepsc) by ecliptic latitude, the count within the 0.262 deg half-width of the zone from which Earth is seen to transit the Sun, where a sender who has found us would aim, against the mean and scatter of the twenty strips of the same width on either side out to 5.5 deg, which share the zone's declinations and so the detector's acceptance; the zone's count at 10 and 100 TeV, with its highest-energy events listed.

    python -I seti.py highenergy nuetz [record] [--energy-min-gev 10000] [--half-width-deg 0.262] [--band-deg 5.5] [--high-gev 100000]"""


def nuetz_main(record=False, energy_min_gev=10000.0, half_width_deg=0.262, band_deg=5.5, high_gev=100000.0):
    bounds = core.heasarc("SELECT MIN(time) AS a, MAX(time) AS b FROM icecubepsc", 5)[0]
    edges = np.linspace(float(bounds["a"]) - 1, float(bounds["b"]) + 1, 6)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        rows += core.heasarc("SELECT event_number, time, ra, dec, error_radius, event_energy FROM icecubepsc WHERE event_energy > %g AND time >= %.5f AND time < %.5f" % (
            energy_min_gev, a, b), 100000)
    ev = [dict(n=str(r["event_number"]), t=num(r["time"]), ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]) or 0.0, e=num(r["event_energy"])) for r in rows]
    ev = [x for x in ev if x["ra"] is not None and x["dec"] is not None and x["e"]]
    lam, beta = science.ecliptic_lon_lat([x["ra"] for x in ev], [x["dec"] for x in ev])
    for x, b, l in zip(ev, beta, lam):
        x["beta"] = abs(float(b))
        x["lam"] = float(l)
    nstrip = int(round(band_deg / half_width_deg)) - 1
    out = {}
    for label, emin in (("low", energy_min_gev), ("high", high_gev)):
        out[label] = dict(emin=emin, **science.zone_excess([x["beta"] for x in ev if x["e"] >= emin], half_width_deg, band_deg))
    top = sorted([x for x in ev if x["beta"] < half_width_deg], key=lambda x: -x["e"])[:10]
    p_low = out["low"]["p"]
    for x in top:
        x["vet"] = "chance-rate" if p_low > 0.01 and out["high"]["p"] > 0.01 else None
    print("%d events above %g GeV; in the transit zone (|beta| < %g deg): %d against %.1f +- %.1f in the %d strips beside it (z %+.2f, Poisson p %.3g); above %g GeV: %d against "
          "%.2f +- %.2f (z %+.2f, p %.3g)" % (len(ev), energy_min_gev, half_width_deg, out["low"]["zone"], out["low"]["mean"], out["low"]["sd"], nstrip, out["low"]["z"], p_low,
                                                 high_gev, out["high"]["zone"], out["high"]["mean"], out["high"]["sd"], out["high"]["z"], out["high"]["p"]))
    for x in top:
        print("  event %s: %.0f GeV at RA %.2f Dec %+.2f (ecliptic %.2f, %+.3f), error %.2f deg, MJD %.3f; %s" % (
            x["n"], x["e"], x["ra"], x["dec"], x["lam"], x["beta"], x["err"], x["t"], x["vet"] or "unexplained"))
    json.dump(dict(n=len(ev), bands=out, top=top), open(core.results("neutrinos_transit_zone.json"), "w"), indent=1)
    if record:
        core.record_run(
            "icecube_transit_zone", "Neutrinos from Earth's transit zone", "IceCube ten-year point-source events (HEASARC icecubepsc)",
            "events above %g GeV: %d, %d of them in the zone" % (energy_min_gev, len(ev), out["low"]["zone"]), os.path.abspath(__file__), "nuetz", len(ev), "events",
            dict(energy_min_gev=energy_min_gev, half_width_deg=half_width_deg, band_deg=band_deg, high_gev=high_gev),
            "the %d strips of the zone's width beside it: %.1f +- %.1f above %g GeV, %.2f +- %.2f above %g GeV" % (
                nstrip, out["low"]["mean"], out["low"]["sd"], energy_min_gev, out["high"]["mean"], out["high"]["sd"], high_gev),
            "an excess of neutrinos from the band of sky that sees Earth transit the Sun",
            "%d in the zone above %g GeV (z %+.2f, p %.3g) and %d above %g GeV (z %+.2f, p %.3g)" % (
                out["low"]["zone"], energy_min_gev, out["low"]["z"], p_low, out["high"]["zone"], high_gev, out["high"]["z"], out["high"]["p"]),
            [dict(target="icecube_event:%s" % x["n"], stat=x["e"], unit="GeV", vet=x["vet"],
                  note="RA %.2f Dec %+.2f, ecliptic latitude %+.3f deg, error %.2f deg, MJD %.3f" % (x["ra"], x["dec"], x["beta"], x["err"], x["t"])) for x in top])
        print("recorded")


grbetz_DOC = r"""Gamma-ray bursts from Earth's transit zone: the Swift GRB table (HEASARC swiftgrb, arcminute positions) and the Fermi GBM bursts (fermigbrst) with a statistical error radius below 1 deg, by ecliptic latitude, the count within the 0.262 deg half-width of the zone from which Earth is seen to transit the Sun against the mean and scatter of the twenty strips of that width on either side out to 5.5 deg, a flash aimed at a world that has been seen; each burst in the zone listed with its duration.

    python -I seti.py highenergy grbetz [record] [--err-max-deg 1] [--half-width-deg 0.262] [--band-deg 5.5]"""


def grbetz_main(record=False, err_max_deg=1.0, half_width_deg=0.262, band_deg=5.5):
    cats = {}
    sw = core.heasarc("SELECT name, ra, dec, bat_t90, trigger_time FROM swiftgrb WHERE ra IS NOT NULL", 20000)
    cats["Swift"] = [dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), err=0.05, t90=num(r["bat_t90"]), t=str(r["trigger_time"] or "")) for r in sw]
    gb = core.heasarc("SELECT name, ra, dec, error_radius, t90, trigger_time FROM fermigbrst WHERE error_radius < %g" % err_max_deg, 20000)
    cats["GBM"] = [dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]), t90=num(r["t90"]), t=num(r["trigger_time"])) for r in gb]
    out, inzone = {}, []
    for label, bursts in cats.items():
        bursts[:] = [b for b in bursts if b["ra"] is not None and b["dec"] is not None]
        lam, beta = science.ecliptic_lon_lat([b["ra"] for b in bursts], [b["dec"] for b in bursts])
        for b, l, be in zip(bursts, lam, beta):
            b["lam"], b["beta"], b["cat"] = float(l), abs(float(be)), label
        out[label] = dict(n=len(bursts), **science.zone_excess([b["beta"] for b in bursts], half_width_deg, band_deg))
        inzone += [b for b in bursts if b["beta"] < half_width_deg]
    chance = all(v["p"] > 0.01 for v in out.values())
    for b in inzone:
        b["vet"] = "chance-rate" if chance else None
    print("; ".join("%s: %d bursts, %d in the zone against %.2f +- %.2f beside it (z %+.2f, Poisson p %.3g)" % (
        k, v["n"], v["zone"], v["mean"], v["sd"], v["z"], v["p"]) for k, v in out.items()))
    for b in sorted(inzone, key=lambda b: b["beta"]):
        print("  %s (%s): RA %.3f Dec %+.3f, ecliptic latitude %+.3f deg, error %.2f deg, T90 %s s; %s" % (b["name"], b["cat"], b["ra"], b["dec"], b["beta"], b["err"], b["t90"],
                                                                                                     b["vet"] or "unexplained"))
    json.dump(dict(bands=out, zone=inzone), open(core.results("grbs_transit_zone.json"), "w"), indent=1)
    if record:
        core.record_run(
            "grb_transit_zone", "Gamma-ray bursts from Earth's transit zone", "Swift GRB table (HEASARC swiftgrb); Fermi GBM burst catalogue (fermigbrst)",
            "Swift bursts: %d; GBM bursts with an error radius below %g deg: %d" % (out["Swift"]["n"], err_max_deg, out["GBM"]["n"]), os.path.abspath(__file__), "grbetz",
            sum(v["n"] for v in out.values()), "bursts", dict(err_max_deg=err_max_deg, half_width_deg=half_width_deg, band_deg=band_deg),
            "the twenty strips of the zone's width beside it: Swift %.2f +- %.2f, GBM %.2f +- %.2f" % (out["Swift"]["mean"], out["Swift"]["sd"], out["GBM"]["mean"], out["GBM"]["sd"]),
            "an excess of bursts from the band of sky that sees Earth transit the Sun",
            "Swift %d in the zone (z %+.2f, p %.3g), GBM %d (z %+.2f, p %.3g)" % (out["Swift"]["zone"], out["Swift"]["z"], out["Swift"]["p"], out["GBM"]["zone"], out["GBM"]["z"],
                                                                                 out["GBM"]["p"]),
            [dict(target="grb:%s" % b["name"], stat=round(b["beta"], 4), unit="ecliptic latitude, deg", vet=b["vet"],
                  note="%s, RA %.3f Dec %+.3f, error %.2f deg, T90 %s s" % (b["cat"], b["ra"], b["dec"], b["err"], b["t90"])) for b in inzone])
        print("recorded")


nudoublets_DOC = r"""Two neutrinos from one spot: the IceCube ten-year point-source events above 100 TeV (HEASARC icecubepsc), pairs within one hour and within the sum of their angular errors plus half a degree, a source firing twice where a diffuse sky fires once, against the pairs the same events give with their times shuffled among them twenty times; each doublet listed with its energies and errors.

    python -I seti.py highenergy nudoublets [record] [--energy-min-gev 100000] [--window-h 1] [--extra-deg 0.5] [--draws 20]"""


def nudoublets_main(record=False, energy_min_gev=100000.0, window_h=1.0, extra_deg=0.5, draws=20):
    from scipy.stats import poisson
    bounds = core.heasarc("SELECT MIN(time) AS a, MAX(time) AS b FROM icecubepsc", 5)[0]
    edges = np.linspace(float(bounds["a"]) - 1, float(bounds["b"]) + 1, 6)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        rows += core.heasarc("SELECT event_number, time, ra, dec, error_radius, event_energy FROM icecubepsc WHERE event_energy > %g AND time >= %.5f AND time < %.5f" % (
            energy_min_gev, a, b), 100000)
    ev = [dict(n=str(r["event_number"]), t=num(r["time"]), ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]) or 0.0, e=num(r["event_energy"])) for r in rows]
    ev = [x for x in ev if x["t"] and x["ra"] is not None and x["dec"] is not None]
    t = np.array([x["t"] for x in ev])
    ra = np.radians([x["ra"] for x in ev])
    dec = np.radians([x["dec"] for x in ev])
    err = np.array([x["err"] for x in ev])
    xyz = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)
    w = window_h / 24.0

    def pairs(times):
        order = np.argsort(times)
        ts = times[order]
        found = []
        for a in range(len(ts)):
            b = a + 1
            while b < len(ts) and ts[b] - ts[a] < w:
                i, j = order[a], order[b]
                sep = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(xyz[i], xyz[j]))))))
                if sep <= err[i] + err[j] + extra_deg:
                    found.append((i, j, sep))
                b += 1
        return found

    real = pairs(t)
    rng = np.random.default_rng(7)
    null = [len(pairs(rng.permutation(t))) for _ in range(draws)]
    exp = float(np.mean(null))
    p = float(poisson.sf(len(real) - 1, exp)) if real else 1.0
    out = [dict(a=ev[i]["n"], b=ev[j]["n"], sep_deg=sep, dt_min=abs(ev[i]["t"] - ev[j]["t"]) * 1440, e_a=ev[i]["e"], e_b=ev[j]["e"], err_a=ev[i]["err"], err_b=ev[j]["err"],
                ra=ev[i]["ra"], dec=ev[i]["dec"], vet="chance-rate" if p > 0.01 else None) for i, j, sep in real]
    print("%d events above %g GeV; pairs within %g h and the summed errors plus %g deg: %d against %.1f +- %.1f with the times shuffled (Poisson p %.3g)" % (
        len(ev), energy_min_gev, window_h, extra_deg, len(real), exp, float(np.std(null)), p))
    for o in sorted(out, key=lambda o: o["sep_deg"] - o["err_a"] - o["err_b"]):
        print("  %s and %s: %.2f deg apart (errors %.2f and %.2f), %.1f min, %.0f and %.0f GeV at RA %.2f Dec %+.2f; %s" % (
            o["a"], o["b"], o["sep_deg"], o["err_a"], o["err_b"], o["dt_min"], o["e_a"], o["e_b"], o["ra"], o["dec"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(ev), null=null, pairs=out), open(core.results("neutrino_doublets.json"), "w"), indent=1)
    if record:
        core.record_run(
            "icecube_doublets", "Two neutrinos from one spot", "IceCube ten-year point-source events (HEASARC icecubepsc)", "events above %g GeV: %d" % (energy_min_gev, len(ev)),
            os.path.abspath(__file__), "nudoublets", len(ev), "events", dict(energy_min_gev=energy_min_gev, window_h=window_h, extra_deg=extra_deg, draws=draws),
            "the pairs the same events give with their times shuffled %d times: %.1f +- %.1f" % (draws, exp, float(np.std(null))),
            "two events within %g h and the summed errors plus %g deg" % (window_h, extra_deg), "%d against %.1f (Poisson p %.3g)" % (len(real), exp, p),
            [dict(target="icecube_event:%s" % o["a"], stat=round(o["sep_deg"], 2), unit="deg to its partner", vet=o["vet"],
                  note="with %s, %.1f min apart, %.0f and %.0f GeV, errors %.2f and %.2f deg" % (o["b"], o["dt_min"], o["e_a"], o["e_b"], o["err_a"], o["err_b"])) for o in out])
        print("recorded")


copygrbs_DOC = r"""Copy gamma-ray bursts: the Fermi GBM burst catalogue (HEASARC fermigbrst), pairs of bursts from directions more than 10 deg apart alike to 2 per cent in T90 and fluence and to 5 per cent in T50 and the Band peak energy, one explosion played twice where nature draws each from broad distributions, against the pairs found after T90 is scaled by factors that keep its distribution and break coincidence.

    python -I seti.py highenergy copygrbs [record] [--tol-time 0.02] [--tol-flux 0.02] [--tol-shape 0.05] [--sep-min-deg 10]"""


def copygrbs_main(record=False, tol_time=0.02, tol_flux=0.02, tol_shape=0.05, sep_min_deg=10.0):
    from scipy.stats import poisson
    rows = core.heasarc("SELECT name, ra, dec, error_radius, t90, t50, fluence, flnc_band_epeak, trigger_time FROM fermigbrst WHERE t90 > 0 AND t50 > 0 AND fluence > 0 AND "
                        "flnc_band_epeak > 0", 20000)
    g = [dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), t90=num(r["t90"]), t50=num(r["t50"]), fl=num(r["fluence"]), ep=num(r["flnc_band_epeak"]),
              t=num(r["trigger_time"])) for r in rows]
    g = [x for x in g if None not in (x["ra"], x["dec"], x["t90"], x["t50"], x["fl"], x["ep"])]
    t90 = np.log(np.array([x["t90"] for x in g]))
    t50 = np.log(np.array([x["t50"] for x in g]))
    fl = np.log(np.array([x["fl"] for x in g]))
    ep = np.log(np.array([x["ep"] for x in g]))
    ra = np.radians([x["ra"] for x in g])
    dec = np.radians([x["dec"] for x in g])
    xyz = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)
    far = xyz @ xyz.T < math.cos(math.radians(sep_min_deg))

    def count(a):
        m = (np.abs(a[:, None] - a[None, :]) < tol_time) & (np.abs(fl[:, None] - fl[None, :]) < tol_flux) & (np.abs(t50[:, None] - t50[None, :]) < tol_shape) & \
            (np.abs(ep[:, None] - ep[None, :]) < tol_shape) & far
        return np.argwhere(np.triu(m, 1))

    real = count(t90)
    null = [len(count(t90 + math.log(f))) for f in (1.37, 1.73, 2.21, 0.73, 0.58, 0.45)]
    exp = float(np.mean(null))
    p = float(poisson.sf(len(real) - 1, exp)) if len(real) else 1.0
    out = [dict(a=g[i]["name"], b=g[j]["name"], t90=(g[i]["t90"], g[j]["t90"]), t50=(g[i]["t50"], g[j]["t50"]), fl=(g[i]["fl"], g[j]["fl"]), ep=(g[i]["ep"], g[j]["ep"]),
                sep_deg=math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(xyz[i], xyz[j])))))), vet="chance-rate" if p > 0.01 else None) for i, j in real]
    print("%d bursts with T90, T50, fluence and a Band peak; pairs more than %g deg apart alike to %g in T90 and fluence and %g in T50 and peak energy: %d against %.1f (%s) with "
          "T90 rescaled (Poisson p %.3g)" % (len(g), sep_min_deg, tol_time, tol_shape, len(real), exp, null, p))
    for o in out:
        print("  %s and %s: T90 %.2f/%.2f s, T50 %.2f/%.2f s, fluence %.2e/%.2e, Epeak %.0f/%.0f keV, %.0f deg apart; %s" % (
            o["a"], o["b"], o["t90"][0], o["t90"][1], o["t50"][0], o["t50"][1], o["fl"][0], o["fl"][1], o["ep"][0], o["ep"][1], o["sep_deg"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(g), null=null, pairs=out), open(core.results("grb_copies.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gbm_copy_bursts", "Copy gamma-ray bursts", "Fermi GBM burst catalogue (HEASARC fermigbrst)", "bursts with T90, T50, fluence and a Band peak energy: %d" % len(g),
            os.path.abspath(__file__), "copygrbs", len(g), "bursts", dict(tol_time=tol_time, tol_flux=tol_flux, tol_shape=tol_shape, sep_min_deg=sep_min_deg),
            "the pairs after T90 is rescaled by six factors: %s" % null, "two bursts from different directions alike in duration, fluence and spectrum",
            "%d against %.1f (Poisson p %.3g)" % (len(real), exp, p),
            [dict(target="grb:%s" % o["a"], stat=round(o["sep_deg"], 1), unit="deg to its copy", vet=o["vet"],
                  note="with %s: T90 %.2f/%.2f s, fluence %.2e/%.2e, Epeak %.0f/%.0f keV" % (o["b"], o["t90"][0], o["t90"][1], o["fl"][0], o["fl"][1], o["ep"][0], o["ep"][1]))
             for o in out])
        print("recorded")


grbtogether_DOC = r"""Gamma-ray bursts arriving together from different directions: the Fermi GBM burst catalogue (HEASARC fermigbrst), pairs of bursts triggered within 10 s whose positions lie farther apart than the sum of their error radii and 5 deg, two explosions timed together across the sky, against the count a Poisson arrival of the same bursts gives, N(N-1) times the window over the span.

    python -I seti.py highenergy grbtogether [record] [--window-s 10] [--extra-deg 5]"""


def grbtogether_main(record=False, window_s=10.0, extra_deg=5.0):
    from scipy.stats import poisson
    rows = core.heasarc("SELECT name, ra, dec, error_radius, trigger_time, t90 FROM fermigbrst WHERE trigger_time > 0", 20000)
    g = [dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]) or 0.0, t=num(r["trigger_time"]), t90=num(r["t90"])) for r in rows]
    g = sorted([x for x in g if x["t"] and x["ra"] is not None and x["dec"] is not None], key=lambda x: x["t"])
    w = window_s / 86400.0
    close, apart = [], []
    for a in range(len(g)):
        b = a + 1
        while b < len(g) and g[b]["t"] - g[a]["t"] < w:
            sep = core.sep_arcsec(g[a]["ra"], g[a]["dec"], g[b]["ra"], g[b]["dec"]) / 3600.0
            pair = dict(a=g[a]["name"], b=g[b]["name"], dt_s=(g[b]["t"] - g[a]["t"]) * 86400.0, sep_deg=sep, err_a=g[a]["err"], err_b=g[b]["err"], t90_a=g[a]["t90"], t90_b=g[b]["t90"])
            close.append(pair)
            if sep > g[a]["err"] + g[b]["err"] + extra_deg:
                apart.append(pair)
            b += 1
    span = g[-1]["t"] - g[0]["t"]
    exp = len(g) * (len(g) - 1) * w / span
    p = float(poisson.sf(len(apart) - 1, exp)) if apart else 1.0
    for o in apart:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d bursts over %.1f years; pairs within %g s: %d, from directions apart beyond their errors and %g deg: %d against %.2f for Poisson arrivals (p %.3g)" % (
        len(g), span / 365.25, window_s, len(close), extra_deg, len(apart), exp, p))
    for o in close:
        print("  %s and %s: %.1f s apart, %.1f deg apart (errors %.1f and %.1f), T90 %s and %s s; %s" % (
            o["a"], o["b"], o["dt_s"], o["sep_deg"], o["err_a"], o["err_b"], o["t90_a"], o["t90_b"], o.get("vet") or ("the same event" if o not in apart else "unexplained")))
    json.dump(dict(n=len(g), close=close, apart=apart, expected=exp), open(core.results("grbs_together.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gbm_bursts_together", "Gamma-ray bursts arriving together from different directions", "Fermi GBM burst catalogue (HEASARC fermigbrst)", "bursts with a trigger time: %d" % len(g),
            os.path.abspath(__file__), "grbtogether", len(g), "bursts", dict(window_s=window_s, extra_deg=extra_deg), "Poisson arrivals of the same bursts: %.2f pairs" % exp,
            "two bursts within %g s from directions apart beyond their errors and %g deg" % (window_s, extra_deg), "%d against %.2f (p %.3g); %d pairs within the window in all" % (
                len(apart), exp, p, len(close)),
            [dict(target="grb:%s" % o["a"], stat=round(o["dt_s"], 1), unit="s to its partner", vet=o["vet"],
                  note="with %s, %.1f deg apart, errors %.1f and %.1f deg" % (o["b"], o["sep_deg"], o["err_a"], o["err_b"])) for o in apart])
        print("recorded")


fermimoved_DOC = r"""Gamma-ray sources that moved between catalogues: Fermi-LAT sources carrying the same association in the four-year 3FGL and the fourteen-year 4FGL-DR4 (HEASARC fermi3fgl and fermilpsc), the separation of the two positions against the sum in quadrature of their 95 per cent semi-major axes, those apart by more than twice that sum, a source that has shifted on the sky between 2012 and 2022 beyond its errors, which a point at a fixed distance never does and a thing in motion would; each read for extension and for its place in the Galactic plane, since a source confused with its neighbours or an extended one shifts as the model changes.

    python -I seti.py highenergy fermimoved [record] [--factor 2] [--plane-deg 2]"""


def fermimoved_main(record=False, factor=2.0, plane_deg=2.0):
    old = core.heasarc("SELECT name, ra, dec, semi_major_axis_95, assoc_name_1, detection_significance FROM fermi3fgl WHERE assoc_name_1 IS NOT NULL", 10000)
    new = core.heasarc("SELECT name, ra, dec, semi_major_axis_95, assoc_name, assoc_name_alt, extended_source_name, detection_significance FROM fermilpsc WHERE assoc_name IS NOT NULL", 20000)

    def key(v):
        return " ".join(str(v or "").split()).lower()
    by = {}
    for r in new:
        for k in (key(r["assoc_name"]), key(r["assoc_name_alt"])):
            if k:
                by.setdefault(k, r)
    out, n = [], 0
    for r in old:
        k = key(r["assoc_name_1"])
        if not k or k not in by or num(r["ra"]) is None:
            continue
        m = by[k]
        if num(m["ra"]) is None:
            continue
        n += 1
        sep = core.sep_arcsec(num(r["ra"]), num(r["dec"]), num(m["ra"]), num(m["dec"])) / 3600.0
        err = math.hypot(num(r["semi_major_axis_95"]) or 0, num(m["semi_major_axis_95"]) or 0)
        if err and sep > factor * err:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            b = float(SkyCoord(ra=num(m["ra"]) * u.deg, dec=num(m["dec"]) * u.deg, frame="icrs").galactic.b.deg)
            out.append(dict(old=str(r["name"]).strip(), new=str(m["name"]).strip(), assoc=" ".join(str(r["assoc_name_1"]).split()), sep_deg=sep, err_deg=err, ratio=sep / err, b=b,
                            extended=str(m["extended_source_name"] or "").strip(), sig=(num(r["detection_significance"]), num(m["detection_significance"])),
                            vet="diffuse-emission" if str(m["extended_source_name"] or "").strip() else "crowding" if abs(b) < plane_deg else None))
    left = [o for o in out if o["vet"] is None]
    print("%d associations shared by 3FGL and 4FGL-DR4; positions apart by more than %g times the combined 95 per cent axes: %d (%d extended in 4FGL, %d within %g deg of the Galactic "
          "plane); unexplained %d" % (n, factor, len(out), sum(o["vet"] == "diffuse-emission" for o in out), sum(o["vet"] == "crowding" for o in out), plane_deg, len(left)))
    for o in sorted(out, key=lambda o: -o["ratio"]):
        print("  %s -> %s (%s): %.3f deg apart against %.3f (%.1f times), b %+.1f, significances %s%s; %s" % (
            o["old"], o["new"], o["assoc"], o["sep_deg"], o["err_deg"], o["ratio"], o["b"], o["sig"], ", extended " + o["extended"] if o["extended"] else "", o["vet"] or "unexplained"))
    json.dump(dict(n=n, moved=out), open(core.results("fermi_moved_sources.json"), "w"), indent=1)
    if record:
        core.record_run(
            "fermi_3fgl_4fgl_moved", "Gamma-ray sources that moved between catalogues", "Fermi-LAT 3FGL (HEASARC fermi3fgl) and 4FGL-DR4 (fermilpsc)",
            "associations shared by the two catalogues: %d" % n, os.path.abspath(__file__), "fermimoved", n, "sources", dict(factor=factor, plane_deg=plane_deg),
            "the catalogues' own 95 per cent error axes in quadrature; the extended sources of 4FGL; the confusion of the Galactic plane",
            "two positions of one association apart by more than %g times their combined errors" % factor,
            "%d, %d extended, %d in the plane; %d unexplained" % (len(out), sum(o["vet"] == "diffuse-emission" for o in out), sum(o["vet"] == "crowding" for o in out), len(left)),
            [dict(target="fermi:%s" % o["new"], stat=round(o["ratio"], 1), unit="separation over the combined 95 per cent axes", vet=o["vet"],
                  note="%s from %s, %.3f deg against %.3f, significances %s%s" % (o["assoc"], o["old"], o["sep_deg"], o["err_deg"], o["sig"], ", extended " + o["extended"] if o["extended"] else ""))
             for o in out])
        print("recorded")


frbgalaxies_DOC = r"""Bursts from the nearest galaxies: CHIME/FRB Catalog 1 bursts against the large galaxies within 10 Mpc north of -11 deg, each burst placed within a galaxy's optical radius plus the catalogue's half-degree localisation, a burst from a galaxy close enough that its light could be answered, against the count the bursts' own density within 10 deg of each galaxy gives; FRB 20200120E in M81's globular cluster is the published case, and each match is read against it.

    python -I seti.py highenergy frbgalaxies [record] [--loc-deg 0.5] [--annulus-deg 10]"""

# the large galaxies within 10 Mpc that CHIME can see: name, RA, Dec (deg) and the optical radius in arcmin
FRBGALAXIES = [("M31", 10.685, 41.269, 95.0), ("M33", 23.462, 30.660, 35.0), ("M81", 148.888, 69.065, 13.0), ("M82", 148.970, 69.680, 6.0), ("IC 342", 56.702, 68.096, 11.0),
               ("NGC 6946", 308.718, 60.154, 6.0), ("M94", 192.721, 41.120, 6.0), ("M64", 194.182, 21.683, 5.0), ("M101", 210.802, 54.349, 14.0), ("M51", 202.470, 47.195, 6.0),
               ("NGC 2403", 114.214, 65.603, 11.0), ("NGC 4236", 184.175, 69.462, 11.0), ("Maffei 1", 40.192, 59.654, 2.0), ("Maffei 2", 40.479, 59.604, 3.0),
               ("NGC 4449", 187.046, 44.094, 3.0), ("M106", 184.740, 47.304, 9.0), ("NGC 4631", 190.533, 32.541, 8.0), ("NGC 891", 35.639, 42.349, 7.0),
               ("NGC 4244", 184.374, 37.807, 8.0), ("NGC 4258", 184.740, 47.304, 9.0), ("NGC 3077", 150.830, 68.734, 3.0), ("NGC 4214", 183.913, 36.327, 4.0),
               ("NGC 2976", 146.814, 67.916, 3.0), ("NGC 5204", 202.401, 58.418, 3.0), ("NGC 5474", 211.256, 53.663, 2.0), ("Holmberg II", 124.771, 70.721, 4.0),
               ("NGC 2366", 112.229, 69.205, 4.0), ("NGC 4605", 189.998, 61.610, 3.0), ("NGC 672", 26.975, 27.433, 4.0), ("IC 1727", 26.875, 27.333, 3.0),
               ("NGC 925", 36.820, 33.579, 5.0), ("NGC 3109", 150.780, -26.160, 10.0), ("Sextans A", 152.753, -4.693, 3.0), ("Sextans B", 150.000, 5.333, 3.0),
               ("Leo A", 149.860, 30.746, 3.0), ("NGC 3738", 173.952, 54.524, 1.5), ("NGC 4395", 186.454, 33.547, 7.0), ("NGC 5238", 203.738, 51.613, 1.0),
               ("UGC 8508", 203.325, 54.912, 1.0), ("NGC 5585", 214.951, 56.729, 3.0), ("NGC 6503", 267.360, 70.144, 4.0), ("NGC 7793", 359.458, -32.591, 5.0),
               ("NGC 3621", 169.568, -32.814, 6.0), ("NGC 247", 11.786, -20.760, 11.0), ("NGC 3184", 154.570, 41.424, 4.0), ("M63", 198.956, 42.029, 6.0),
               ("NGC 4826", 194.182, 21.683, 5.0), ("NGC 5055", 198.956, 42.029, 6.0), ("NGC 4490", 187.651, 41.644, 3.0), ("NGC 4485", 187.661, 41.702, 1.0)]


def frbgalaxies_main(record=False, loc_deg=0.5, annulus_deg=10.0):
    from scipy.stats import poisson
    frbs = catalogues.frb_catalog1("Name, RpName, RAJ2000, DEJ2000, DMfitb")
    f = [dict(name=str(r["Name"]).strip(), rep=str(r["RpName"] or "").strip().replace("-9999", ""), ra=num(r["RAJ2000"]), dec=num(r["DEJ2000"]), dm=num(r["DMfitb"])) for r in frbs]
    f = [x for x in f if x["ra"] is not None and x["dec"] is not None]
    seen = set()
    f = [x for x in f if not (x["rep"] and x["rep"] in seen or seen.add(x["rep"] or x["name"]))]
    gals = [g for g in FRBGALAXIES if g[2] > -11]
    out, exp_total = [], 0.0
    for name, ra, dec, rad in gals:
        r = rad / 60.0 + loc_deg
        seps = np.array([core.sep_arcsec(ra, dec, x["ra"], x["dec"]) / 3600.0 for x in f])
        inside = [f[i] for i in np.nonzero(seps < r)[0]]
        ring = int(np.sum((seps >= r) & (seps < annulus_deg)))
        exp = ring * (1 - math.cos(math.radians(r))) / (math.cos(math.radians(r)) - math.cos(math.radians(annulus_deg)))
        exp_total += exp
        for x in inside:
            out.append(dict(galaxy=name, frb=x["name"], rep=x["rep"], sep_deg=float(core.sep_arcsec(ra, dec, x["ra"], x["dec"]) / 3600.0), radius_deg=r, dm=x["dm"], expected=exp,
                            vet="natural-source" if x["rep"] and "20200120E" in x["rep"] else None))
    p = float(poisson.sf(len(out) - 1, exp_total)) if out else 1.0
    for o in out:
        if o["vet"] is None and p > 0.01:
            o["vet"] = "chance-rate"
    left = [o for o in out if o["vet"] is None]
    print("%d sources against %d galaxies within 10 Mpc north of -11 deg; bursts inside a galaxy's radius plus %g deg: %d against %.2f from the density within %g deg (Poisson p %.3g); "
          "unexplained %d" % (len(f), len(gals), loc_deg, len(out), exp_total, annulus_deg, p, len(left)))
    for o in sorted(out, key=lambda o: o["sep_deg"] / o["radius_deg"]):
        print("  %s at %s: %.2f deg of %.2f, DM %s%s; %s" % (o["frb"], o["galaxy"], o["sep_deg"], o["radius_deg"], o["dm"], ", repeater " + o["rep"] if o["rep"] else "", o["vet"] or "unexplained"))
    json.dump(dict(n=len(f), galaxies=len(gals), expected=exp_total, matches=out), open(core.results("frbs_nearest_galaxies.json"), "w"), indent=1)
    if record:
        core.record_run(
            "chime_nearest_galaxies", "Bursts from the nearest galaxies", "CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59); the large galaxies within 10 Mpc",
            "sources: %d; galaxies north of -11 deg: %d" % (len(f), len(gals)), os.path.abspath(__file__), "frbgalaxies", len(f), "sources", dict(loc_deg=loc_deg, annulus_deg=annulus_deg),
            "the bursts' density within %g deg of each galaxy: %.2f expected inside" % (annulus_deg, exp_total), "a burst inside a nearby galaxy's radius and the localisation",
            "%d against %.2f (Poisson p %.3g); %d unexplained" % (len(out), exp_total, p, len(left)),
            [dict(target="frb:%s" % o["frb"], stat=round(o["sep_deg"], 3), unit="deg from the galaxy", vet=o["vet"], note="%s, radius %.2f deg, DM %s" % (o["galaxy"], o["radius_deg"], o["dm"]))
             for o in out])
        print("recorded")


grbgalaxies_DOC = r"""Gamma-ray bursts from the nearest galaxies: the Swift GRB table (HEASARC swiftgrb, arcminute positions) and the Fermi GBM bursts with a statistical error radius below 1 deg against the large galaxies within 10 Mpc, a burst inside a galaxy's optical radius plus its own error, against the count the bursts' density within 10 deg of each galaxy gives; giant flares of magnetars in nearby galaxies are the published cases, GRB 200415A in NGC 253 and GRB 231115A in M82, and each match is read against them and against the burst's duration.

    python -I seti.py highenergy grbgalaxies [record] [--err-max-deg 1] [--annulus-deg 10]"""

GRBGALAXIES_KNOWN = {"GRB200415A": "a magnetar giant flare in NGC 253 (Svinkin et al. 2021)", "GRB 200415A": "a magnetar giant flare in NGC 253 (Svinkin et al. 2021)",
                     "GRB231115650": "a magnetar giant flare in M82 (Mereghetti et al. 2024)", "GRB 231115A": "a magnetar giant flare in M82 (Mereghetti et al. 2024)"}


def grbgalaxies_main(record=False, err_max_deg=1.0, annulus_deg=10.0):
    from scipy.stats import poisson
    bursts = []
    for r in core.heasarc("SELECT name, ra, dec, bat_t90 FROM swiftgrb WHERE ra IS NOT NULL", 20000):
        bursts.append(dict(name=str(r["name"]).strip(), cat="Swift", ra=num(r["ra"]), dec=num(r["dec"]), err=0.05, t90=num(r["bat_t90"])))
    for r in core.heasarc("SELECT name, ra, dec, error_radius, t90 FROM fermigbrst WHERE error_radius < %g" % err_max_deg, 20000):
        bursts.append(dict(name=str(r["name"]).strip(), cat="GBM", ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]) or 0.0, t90=num(r["t90"])))
    bursts = [b for b in bursts if b["ra"] is not None and b["dec"] is not None]
    out, exp_total = [], 0.0
    for name, ra, dec, rad in FRBGALAXIES:
        seps = np.array([core.sep_arcsec(ra, dec, b["ra"], b["dec"]) / 3600.0 for b in bursts])
        radii = np.array([rad / 60.0 + b["err"] for b in bursts])
        inside = [bursts[i] for i in np.nonzero(seps < radii)[0]]
        r0 = rad / 60.0 + 0.3
        ring = int(np.sum((seps >= r0) & (seps < annulus_deg)))
        exp = ring * (1 - math.cos(math.radians(r0))) / (math.cos(math.radians(r0)) - math.cos(math.radians(annulus_deg)))
        exp_total += exp
        for b in inside:
            out.append(dict(galaxy=name, burst=b["name"], cat=b["cat"], sep_deg=float(core.sep_arcsec(ra, dec, b["ra"], b["dec"]) / 3600.0), radius_deg=rad / 60.0 + b["err"], t90=b["t90"],
                            vet="natural-source" if b["name"] in GRBGALAXIES_KNOWN else None, note=GRBGALAXIES_KNOWN.get(b["name"])))
    p = float(poisson.sf(len(out) - 1, exp_total)) if out else 1.0
    for o in out:
        if o["vet"] is None and p > 0.01:
            o["vet"] = "chance-rate"
    left = [o for o in out if o["vet"] is None]
    print("%d bursts against %d galaxies within 10 Mpc; bursts inside a galaxy's radius plus their error: %d against %.2f from the density within %g deg (Poisson p %.3g); unexplained %d" % (
        len(bursts), len(FRBGALAXIES), len(out), exp_total, annulus_deg, p, len(left)))
    for o in sorted(out, key=lambda o: o["sep_deg"] / o["radius_deg"]):
        print("  %s (%s) at %s: %.2f deg of %.2f, T90 %s s%s; %s" % (o["burst"], o["cat"], o["galaxy"], o["sep_deg"], o["radius_deg"], o["t90"], ", " + o["note"] if o["note"] else "",
                                                                   o["vet"] or "unexplained"))
    json.dump(dict(n=len(bursts), expected=exp_total, matches=out), open(core.results("grbs_nearest_galaxies.json"), "w"), indent=1)
    if record:
        core.record_run(
            "grb_nearest_galaxies", "Gamma-ray bursts from the nearest galaxies", "Swift GRB table (HEASARC swiftgrb); Fermi GBM burst catalogue (fermigbrst); the large galaxies within 10 Mpc",
            "Swift bursts and GBM bursts with errors below %g deg: %d" % (err_max_deg, len(bursts)), os.path.abspath(__file__), "grbgalaxies", len(bursts), "bursts",
            dict(err_max_deg=err_max_deg, annulus_deg=annulus_deg), "the bursts' density within %g deg of each galaxy: %.2f expected inside" % (annulus_deg, exp_total),
            "a burst inside a nearby galaxy's radius and its own error", "%d against %.2f (Poisson p %.3g); %d unexplained" % (len(out), exp_total, p, len(left)),
            [dict(target="grb:%s" % o["burst"], stat=round(o["sep_deg"], 3), unit="deg from the galaxy", vet=o["vet"],
                  note="%s, %s, radius %.2f deg, T90 %s s%s" % (o["galaxy"], o["cat"], o["radius_deg"], o["t90"], ", " + o["note"] if o["note"] else "")) for o in out])
        print("recorded")


hardgamma_DOC = r"""Gamma rays harder than any shock: Fermi-LAT 4FGL-DR4 sources (HEASARC fermilpsc) above 10 sigma whose power-law photon index lies below 1.3 at its upper error, a spectrum rising in energy flux more steeply than any shock accelerator or jet produces, which a monoenergetic or line-like emitter would mimic; each read for its association, since a pulsar's exponentially cut-off spectrum fits a hard power law below the cutoff and a hard blazar's peak does the same, and for its curvature significance.

    python -I seti.py highenergy hardgamma [record] [--index-max 1.3] [--sig-min 10]"""


def hardgamma_main(record=False, index_max=1.3, sig_min=10.0):
    rows = core.heasarc("SELECT name, ra, dec, pl_index, pl_index_error, lp_curve_significance, plec_curve_significance, source_type, assoc_name, detection_significance, extended_source_name "
                        "FROM fermilpsc WHERE detection_significance > %g AND pl_index IS NOT NULL" % sig_min, 20000)
    n_all = len(rows)
    hits = []
    for r in rows:
        idx, e = num(r["pl_index"]), num(r["pl_index_error"]) or 0.0
        if idx is None or idx + e >= index_max:
            continue
        cls = str(r["source_type"] or "").strip().lower()
        hits.append(dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), index=idx, err=e, curve=max(num(r["lp_curve_significance"]) or 0, num(r["plec_curve_significance"]) or 0),
                         cls=cls, assoc=" ".join(str(r["assoc_name"] or "").split()), sig=num(r["detection_significance"]), extended=str(r["extended_source_name"] or "").strip(),
                         vet=("natural-type" if cls in ("psr", "msp", "pwn", "snr", "spp", "hmb", "lmb", "bin", "glc", "sfr", "nov", "gal", "css", "bcu", "bll", "fsrq", "agn", "rdg", "sey",
                                                        "nlsy1", "ssrq", "sbg", "unk") and cls else "diffuse-emission" if str(r["extended_source_name"] or "").strip() else None)))
    left = [h for h in hits if h["vet"] is None]
    print("%d sources above %g sigma with a power-law index; index below %g at its upper error: %d (%d of associated classes, %d extended); unexplained %d" % (
        n_all, sig_min, index_max, len(hits), sum(h["vet"] == "natural-type" for h in hits), sum(h["vet"] == "diffuse-emission" for h in hits), len(left)))
    for h in sorted(hits, key=lambda h: h["index"]):
        print("  %s: index %.2f +- %.2f, curvature %.1f sigma, class %s, %s, %.1f sigma%s; %s" % (
            h["name"], h["index"], h["err"], h["curve"], h["cls"] or "-", h["assoc"] or "no association", h["sig"], ", extended " + h["extended"] if h["extended"] else "", h["vet"] or "unexplained"))
    json.dump(dict(n=n_all, hits=hits), open(core.results("hard_gamma_sources.json"), "w"), indent=1)
    if record:
        core.record_run(
            "fermi_hard_spectra", "Gamma rays harder than any shock", "Fermi-LAT 4FGL-DR4 (HEASARC fermilpsc)", "sources above %g sigma with a power-law index: %d" % (sig_min, n_all),
            os.path.abspath(__file__), "hardgamma", n_all, "sources", dict(index_max=index_max, sig_min=sig_min), "the catalogue's associations and extended sources",
            "a photon index below %g at its upper error" % index_max, "%d: %d of associated classes, %d extended; %d unexplained" % (
                len(hits), sum(h["vet"] == "natural-type" for h in hits), sum(h["vet"] == "diffuse-emission" for h in hits), len(left)),
            [dict(target="fermi:%s" % h["name"], stat=round(h["index"], 2), unit="photon index", vet=h["vet"],
                  note="+- %.2f, curvature %.1f sigma, class %s, %s, %.1f sigma" % (h["err"], h["curve"], h["cls"] or "-", h["assoc"] or "no association", h["sig"])) for h in hits])
        print("recorded")


gammaflares_DOC = r"""Gamma-ray flares from the nearest stars: the second Fermi All-sky Variability Analysis catalogue of flaring sources (HEASARC fermi2favs, 518 sources with their error radii) against the Gaia Catalogue of Nearby Stars within 20 pc, a flare of GeV light from a star next door, which no quiet star makes and a beam or an engine would; against the same stars moved 2 deg in eight directions, and each match read for the flaring source's own counterpart, since nearly every flaring source is a blazar behind the star.

    python -I seti.py highenergy gammaflares [record] [--plx-min 50] [--extra-deg 0.1]"""


def gammaflares_main(record=False, plx_min=50.0, extra_deg=0.1):
    from scipy.stats import poisson
    flares = [dict(name=str(r["name"]).strip(), ra=num(r["ra"]), dec=num(r["dec"]), err=num(r["error_radius"]) or 0.5, n=num(r["total_flares"]), kind=str(r["source_type"] or "").strip(),
                   counterpart=" ".join(str(r["gamma_ctrprt_name"] or "").split()), other=" ".join(str(r["other_ctrprt_name"] or "").split()))
              for r in core.heasarc("SELECT name, ra, dec, error_radius, total_flares, source_type, gamma_ctrprt_name, other_ctrprt_name FROM fermi2favs", 2000)]
    flares = [f for f in flares if f["ra"] is not None and f["dec"] is not None]
    stars = [dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), plx=num(r["parallax"]), g=num(r["phot_g_mean_mag"])) for r in
             catalogues.gcns("source_id, ra, dec, parallax, phot_g_mean_mag", "gcns_prob > 0.9 AND parallax > %g" % plx_min, ranges=4)]

    def count(dra, ddec):
        hits = []
        for s in stars:
            ra, dec = s["ra"] + dra / max(math.cos(math.radians(s["dec"])), 0.01), s["dec"] + ddec
            if abs(dec) > 90:
                continue
            for f in flares:
                if core.sep_arcsec(ra, dec, f["ra"], f["dec"]) / 3600.0 <= f["err"] + extra_deg:
                    hits.append((s, f))
        return hits
    real = count(0.0, 0.0)
    null = [len(count(2 * math.cos(k * math.pi / 4), 2 * math.sin(k * math.pi / 4))) for k in range(8)]
    exp = float(np.mean(null))
    p = float(poisson.sf(len(real) - 1, exp)) if real else 1.0
    out = [dict(star=s["id"], g=s["g"], d_pc=1000 / s["plx"] if s["plx"] else None, flare=f["name"], sep_deg=core.sep_arcsec(s["ra"], s["dec"], f["ra"], f["dec"]) / 3600.0, err_deg=f["err"],
                flares=f["n"], kind=f["kind"], counterpart=f["counterpart"] or f["other"],
                vet="natural-type" if (f["counterpart"] or f["other"]) else ("chance-rate" if p > 0.01 else None)) for s, f in real]
    left = [o for o in out if o["vet"] is None]
    print("%d flaring sources and %d stars within %.0f pc; stars inside a flaring source's error circle: %d against %.1f with the stars moved 2 deg (%s; Poisson p %.3g); %d sources with a "
          "counterpart of their own; unexplained %d" % (len(flares), len(stars), 1000 / plx_min, len(real), exp, null, p, sum(o["vet"] == "natural-type" for o in out), len(left)))
    for o in sorted(out, key=lambda o: o["sep_deg"] / o["err_deg"]):
        print("  Gaia DR3 %s (G %.1f, %.1f pc) in %s: %.2f deg of %.2f, %s flares, %s, counterpart %s; %s" % (
            o["star"], o["g"] or 0, o["d_pc"] or 0, o["flare"], o["sep_deg"], o["err_deg"], o["flares"], o["kind"] or "-", o["counterpart"] or "none", o["vet"] or "unexplained"))
    json.dump(dict(n_flares=len(flares), n_stars=len(stars), null=null, matches=out), open(core.results("gamma_flares_nearby_stars.json"), "w"), indent=1)
    if record:
        core.record_run(
            "fava_nearest_stars", "Gamma-ray flares from the nearest stars", "Fermi All-sky Variability Analysis second catalogue (HEASARC fermi2favs); Gaia Catalogue of Nearby Stars",
            "flaring sources: %d; stars within %.0f pc: %d" % (len(flares), 1000 / plx_min, len(stars)), os.path.abspath(__file__), "gammaflares", len(stars), "stars",
            dict(plx_min=plx_min, extra_deg=extra_deg), "the stars moved 2 deg in eight directions: %s" % null, "a star within %.0f pc inside a flaring source's error circle" % (1000 / plx_min),
            "%d against %.1f (Poisson p %.3g); %d with the source's own counterpart; %d unexplained" % (len(real), exp, p, sum(o["vet"] == "natural-type" for o in out), len(left)),
            [dict(target="gaia_dr3:%s" % o["star"], stat=round(o["sep_deg"], 3), unit="deg from the flaring source", vet=o["vet"],
                  note="%s, error %.2f deg, %s flares, %s, counterpart %s" % (o["flare"], o["err_deg"], o["flares"], o["kind"] or "-", o["counterpart"] or "none")) for o in out])
        print("recorded")


barlagrange_DOC = r"""The Lagrange points of the Galactic bar: the bar's L4 and L5, on its minor axis at the corotation radius (6.5 kpc, the bar's major axis 27 deg from the Sun's line to the centre, with the Sun at 8.178 kpc), the only large-scale stable points in the Galaxy, where a builder wanting a fixed place would settle; the unassociated Fermi-LAT 4FGL-DR4 sources within 2 deg of each point's line of sight, leaving out the confused ('c') sources and those flagged as soft Galactic unassociated (flag 14), which mark defects of the diffuse model, against the same count toward 2 deg windows along the inner plane (|l| up to 60 deg) and toward other points of the corotation ring; a point on a spiral arm's tangent, where the line of sight runs along a whole arm, is read as that.

    python -I seti.py highenergy barlagrange [record] [--radius-deg 2] [--r-cr-kpc 6.5] [--bar-angle-deg 27]"""

ARM_TANGENTS = {"Scutum": 31.0, "Sagittarius": 50.0, "Crux-Centaurus": 310.0, "Norma": 328.0, "Perseus start": 336.0, "3-kpc near": 338.0}


def barlagrange_main(record=False, radius_deg=2.0, r_cr_kpc=6.5, bar_angle_deg=27.0):
    from scipy.stats import poisson
    r0 = 8.178
    phi = math.radians(bar_angle_deg)
    rows = core.heasarc("SELECT name, lii, bii, source_type, analysis_flags FROM fermilpsc WHERE ABS(bii) < %g" % (radius_deg + 1))
    unas = [(float(r["lii"]), float(r["bii"])) for r in rows if not (r.get("source_type") or "").strip() and not int(r["analysis_flags"] or 0) & 8192 and not r["name"].strip().endswith("c")]

    def dl(a, b):
        return abs((a - b + 180) % 360 - 180)

    def los(theta):
        # the point at Galactocentric angle theta from the bar's near end, seen from the Sun; x toward the centre, y toward l = 90
        x = r0 + r_cr_kpc * (-math.cos(phi) * math.cos(theta) + math.sin(phi) * math.sin(theta))
        y = r_cr_kpc * (math.sin(phi) * math.cos(theta) + math.cos(phi) * math.sin(theta))
        return math.degrees(math.atan2(y, x)) % 360, math.hypot(x, y)

    def count(l0):
        return sum(1 for l, b in unas if core.sep_arcsec(l0, 0.0, l, b) < radius_deg * 3600)
    pts = {"L4": los(math.pi / 2), "L5": los(-math.pi / 2)}
    ring = []
    for k in range(72):
        th = 2 * math.pi * k / 72
        if min(abs(math.sin(th)), abs(math.cos(th))) < math.sin(math.radians(15)):
            continue
        l, d = los(th)
        if min(dl(l, pts["L4"][0]), dl(l, pts["L5"][0])) >= 2 * radius_deg:
            ring.append(count(l))
    plane = [count(l) for l in np.arange(0, 360, 2 * radius_deg) if dl(l, 0) <= 60 and min(dl(l, pts["L4"][0]), dl(l, pts["L5"][0])) >= 2 * radius_deg]
    ring, plane = np.array(ring), np.array(plane)
    out = {}
    for k, (l, d) in pts.items():
        n = count(l)
        tangent = min(ARM_TANGENTS, key=lambda t: dl(l, ARM_TANGENTS[t]))
        p = float(poisson.sf(n - 1, plane.mean())) if n else 1.0
        out[k] = dict(l=round(l, 2), d_kpc=round(d, 2), n=n, plane_mean=float(plane.mean()), plane_max=int(plane.max()), p=p, ring_share=float((ring >= n).mean()),
                      tangent=tangent if dl(l, ARM_TANGENTS[tangent]) < 2 * radius_deg + 1 else None)
        out[k]["vet"] = "arm-tangent" if p < 0.01 and out[k]["tangent"] else None
    print("%d clean unassociated 4FGL sources within %g deg of the plane; inner-plane windows: %d, mean %.2f, max %d; ring windows: %d, median %.1f" % (
        len(unas), radius_deg + 1, len(plane), plane.mean(), plane.max(), len(ring), float(np.median(ring))))
    for k, v in out.items():
        print("  %s at l %.2f deg, %.1f kpc: %d (Poisson p %.2g against the plane's mean, ring share at or above %.2f); nearest arm tangent %s; %s" % (
            k, v["l"], v["d_kpc"], v["n"], v["p"], v["ring_share"], v["tangent"], v["vet"] or "unexplained" if v["p"] < 0.01 else "no excess"))
    if record:
        core.record_run(
            "fermi_bar_lagrange_points", "The Lagrange points of the Galactic bar", "Fermi-LAT 4FGL-DR4 (HEASARC fermilpsc)",
            "unassociated sources within %g deg of the plane, without confused or flag-14 sources: %d" % (radius_deg + 1, len(unas)), os.path.abspath(__file__), "barlagrange", 2, "points",
            dict(radius_deg=radius_deg, r_cr_kpc=r_cr_kpc, bar_angle_deg=bar_angle_deg),
            "%d windows along the inner plane, mean %.2f; %d other lines of sight to the corotation ring" % (len(plane), plane.mean(), len(ring)),
            "unassociated gamma-ray sources crowding the bar's L4 or L5",
            "; ".join("%s at l %.1f deg %d against the plane's mean %.2f (p %.2g)%s" % (k, v["l"], v["n"], v["plane_mean"], v["p"], ", on the %s arm's tangent" % v["tangent"] if v["vet"] else "")
                      for k, v in out.items()),
            [dict(target="galactic:l%.1f_b0" % v["l"], stat=v["n"], unit="clean unassociated sources within %g deg" % radius_deg, p=v["p"], vet=v["vet"],
                  note="the bar's %s at %.1f kpc; the plane's mean %.2f, max %d" % (k, v["d_kpc"], v["plane_mean"], v["plane_max"])) for k, v in out.items() if v["p"] < 0.01])
        print("recorded")


tadoublets_DOC = r"""Cosmic-ray doublets, northern sky: the 72 Telescope Array events above 57 EeV of its first five years (Abbasi et al. 2014, VizieR J/ApJ/790/L21), pairs within 2 deg arriving within 10 days and within a day, and triplets within 30 days, against 5000 shuffles of the arrival times among the events; a source firing particles in bursts would repeat on the sky within days, which the northern hotspot's steady excess does not.

    python -I seti.py highenergy tadoublets [record] [--sep-deg 2] [--shuffles 5000]"""


def tadoublets_main(record=False, sep_deg=2.0, shuffles=5000):
    from datetime import datetime
    t = core.vizier("J/ApJ/790/L21")[0]
    ev = [(datetime.fromisoformat(str(r["Arrival"]).strip()).timestamp() / 86400.0, float(r["RAJ2000"]), float(r["DEJ2000"]), float(r["E"])) for r in t]
    n = len(ev)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if core.sep_arcsec(ev[i][1], ev[i][2], ev[j][1], ev[j][2]) < sep_deg * 3600]
    close = {i: set() for i in range(n)}
    for i, j in pairs:
        close[i].add(j)
        close[j].add(i)
    tri = [(i, j, k) for i, j in pairs for k in close[i] & close[j] if k > j]

    def stats(tt):
        d10 = sum(1 for i, j in pairs if abs(tt[i] - tt[j]) < 10)
        d1 = sum(1 for i, j in pairs if abs(tt[i] - tt[j]) < 1)
        t30 = sum(1 for i, j, k in tri if max(tt[i], tt[j], tt[k]) - min(tt[i], tt[j], tt[k]) < 30)
        return d10, d1, t30
    tt = np.array([e[0] for e in ev])
    real = stats(tt)
    rng = np.random.default_rng(11)
    null = np.array([stats(rng.permutation(tt)) for _ in range(shuffles)])
    p = [float((null[:, k] >= real[k]).mean()) for k in range(3)]
    found = [(str(t[i]["Arrival"]).strip(), str(t[j]["Arrival"]).strip(), round(abs(tt[i] - tt[j]), 2)) for i, j in pairs if abs(tt[i] - tt[j]) < 10]
    print("%d events, %d pairs within %g deg, %d triplets; pairs within 10 days %d against %.2f (p %.3g), within a day %d against %.2f (p %.3g); triplets within 30 days %d against %.2f (p %.3g); %s" % (
        n, len(pairs), sep_deg, len(tri), real[0], null[:, 0].mean(), p[0], real[1], null[:, 1].mean(), p[1], real[2], null[:, 2].mean(), p[2], found))
    if record:
        core.record_run(
            "ta_doublets", "Cosmic-ray doublets, northern sky", "Telescope Array five-year events above 57 EeV (VizieR J/ApJ/790/L21)", "events: %d; pairs within %g deg: %d" % (n, sep_deg, len(pairs)),
            os.path.abspath(__file__), "tadoublets", n, "events", dict(sep_deg=sep_deg, shuffles=shuffles),
            "%d shuffles of the arrival times: %.2f pairs within 10 days, %.2f within a day, %.2f triplets within 30 days" % (shuffles, null[:, 0].mean(), null[:, 1].mean(), null[:, 2].mean()),
            "cosmic rays repeating from one direction within days", "pairs within 10 days %d against %.2f (p %.3g), within a day %d against %.2f; triplets within 30 days %d against %.2f" % (
                real[0], null[:, 0].mean(), p[0], real[1], null[:, 1].mean(), real[2], null[:, 2].mean()), [])
        print("recorded")


unasclass_DOC = r"""What a pile of unassociated gamma-ray sources is: the 4FGL-DR4 sources associated with pulsars, with blazars and with supernova remnants, nebulae and star-forming regions train a random forest on spectral shape and variability (power-law index, log-parabola curvature and its significance, peak energy and variability index), and the unassociated sources within a window of the plane, neither confused nor flagged as soft Galactic unassociated, are classed by it against the same classing of the clean unassociated sources elsewhere within 5 deg of the plane; their X-ray counterparts in eRASS1 within 1 arcmin of the 95 per cent ellipse's centre are counted the same way; a pile of young pulsars along an arm's tangent looks like pulsars, a defect of the diffuse model like none of the trained classes.

    python -I seti.py highenergy unasclass [record] [--l-deg 312.07] [--b-deg 0] [--radius-deg 2]"""


def unasclass_main(record=False, l_deg=312.07, b_deg=0.0, radius_deg=2.0):
    from sklearn.ensemble import RandomForestClassifier
    rows = core.heasarc("SELECT name, ra, dec, lii, bii, source_type, analysis_flags, pl_index, lp_beta, lp_curve_significance, lp_epeak, variability_index, detection_significance "
                        "FROM fermilpsc")
    groups = {"pulsar": ("PSR", "MSP"), "blazar": ("BLL", "FSRQ", "BCU", "RDG", "AGN"), "remnant or nebula": ("SNR", "PWN", "SPP", "SFR", "HMB", "BIN")}

    def feats(r):
        v = [num(r.get(k)) for k in ("pl_index", "lp_beta", "lp_curve_significance", "lp_epeak", "variability_index")]
        if None in v or v[3] <= 0 or v[4] <= 0:
            return None
        return [v[0], v[1], min(v[2], 30.0), math.log10(v[3]), math.log10(v[4])]
    X, y = [], []
    for r in rows:
        t = (r.get("source_type") or "").strip().upper()
        for g, codes in groups.items():
            if t in codes:
                f = feats(r)
                if f:
                    X.append(f)
                    y.append(g)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=41, class_weight="balanced").fit(np.array(X), np.array(y))
    clean = [r for r in rows if not (r.get("source_type") or "").strip() and not int(r["analysis_flags"] or 0) & 8192 and not r["name"].strip().endswith("c") and abs(float(r["bii"])) < 5]
    inwin = [r for r in clean if core.sep_arcsec(l_deg, b_deg, float(r["lii"]), float(r["bii"])) < radius_deg * 3600]
    rest = [r for r in clean if r not in inwin]

    def classify(rs):
        fs = [(r, feats(r)) for r in rs]
        fs = [(r, f) for r, f in fs if f]
        if not fs:
            return [], {}
        pr = clf.predict_proba(np.array([f for _, f in fs]))
        lab = [(r, clf.classes_[int(np.argmax(p))], float(np.max(p))) for (r, _), p in zip(fs, pr)]
        share = {c: sum(1 for _, l, _ in lab if l == c) / len(lab) for c in clf.classes_}
        return lab, share
    lw, sw = classify(inwin)
    lr, sr = classify(rest)

    def xray(rs):
        up = [dict(source_id=k, ra=float(r["ra"]), dec=float(r["dec"])) for k, r in enumerate(rs) if float(r["lii"]) > 180]
        if not up:
            return 0, 0
        hit = {int(m["source_id"]) for m in core.xmatch(up, "J/A+A/682/A34/erass1-m", 60.0)}
        return len(hit), len(up)
    xw, xr = xray(inwin), xray(rest)
    print("classifier trained on %d associated sources; the %d clean unassociated sources within %g deg of l %.2f, b %.2f: %s; the %d elsewhere within 5 deg of the plane: %s; eRASS1 within "
          "1 arcmin: %d of %d in the window, %d of %d elsewhere in eRASS1's half of the sky" % (
              len(y), len(lw), radius_deg, l_deg, b_deg, {k: round(v, 2) for k, v in sw.items()}, len(lr), {k: round(v, 2) for k, v in sr.items()}, xw[0], xw[1], xr[0], xr[1]))
    for r, lab, p in lw:
        print("  %s (l %.2f, b %+.2f): index %s, curvature %s at %s sigma, peak %s MeV, variability %s; %s (%.2f)" % (
            r["name"].strip(), float(r["lii"]), float(r["bii"]), r["pl_index"], r["lp_beta"], r["lp_curve_significance"], r["lp_epeak"], r["variability_index"], lab, p))
    if record:
        core.record_run(
            "fermi_unassociated_pile_classes", "What a pile of unassociated gamma-ray sources is", "Fermi-LAT 4FGL-DR4 (HEASARC fermilpsc); eRASS1 by CDS XMatch",
            "clean unassociated sources within %g deg of l %.2f, b %.2f: %d" % (radius_deg, l_deg, b_deg, len(lw)), os.path.abspath(__file__), "unasclass", len(lw), "sources",
            dict(l_deg=l_deg, b_deg=b_deg, radius_deg=radius_deg), "a random forest trained on %d associated pulsars, blazars and remnants or nebulae; the %d clean unassociated sources elsewhere "
            "within 5 deg of the plane: %s" % (len(y), len(lr), {k: round(v, 2) for k, v in sr.items()}), "a pile unlike the plane's other unassociated sources",
            "classed %s against %s elsewhere; eRASS1 counterparts %d of %d against %d of %d" % ({k: round(v, 2) for k, v in sw.items()}, {k: round(v, 2) for k, v in sr.items()},
                                                                                                xw[0], xw[1], xr[0], xr[1]),
            [dict(target="fermi:%s" % r["name"].strip(), stat=round(p, 2), unit="class probability", vet=None, note="classed %s" % lab) for r, lab, p in lw])
        print("recorded")


hgpsoverlap_DOC = r"""Gamma-ray points inside TeV glows: the clean unassociated 4FGL-DR4 sources within a window of the plane against the sources of the H.E.S.S. Galactic Plane Survey (VizieR J/A+A/612/A1), a point counted inside a TeV source when it lies within twice the source's Gaussian width (or 0.1 deg for a point-like one) of its centre, against the same share among the clean unassociated sources elsewhere in the survey's footprint (longitude 250-65 deg through zero, latitude within 3 deg); points that the catalogue's model splits out of an extended glow sit inside it more often than the rest.

    python -I seti.py highenergy hgpsoverlap [record] [--l-deg 312.07] [--b-deg 0] [--radius-deg 2] [--width-factor 2]"""


def hgpsoverlap_main(record=False, l_deg=312.07, b_deg=0.0, radius_deg=2.0, width_factor=2.0):
    from scipy.stats import fisher_exact
    tabs = core.vizier("J/A+A/612/A1")
    hg = None
    for t in tabs:
        cols = set(t.colnames)
        if {"GLON", "GLAT"} <= cols and any(c.lower().startswith("size") or c.startswith("sigma") or c.startswith("Size") for c in cols):
            hg = t
            break
    if hg is None:
        hg = max(tabs, key=len)
    scol = next((c for c in hg.colnames if c.lower().startswith("size")), None)
    ncol = next((c for c in hg.colnames if c.upper().startswith("HGPS") or c.lower() in ("name", "source_name", "source")), hg.colnames[0])
    src = []
    for r in hg:
        size = num(r[scol]) if scol else None
        src.append((float(r["GLON"]), float(r["GLAT"]), max(width_factor * (size or 0.0), 0.1), str(r[ncol]).strip()))
    rows = core.heasarc("SELECT name, lii, bii, source_type, analysis_flags FROM fermilpsc WHERE ABS(bii) < 3")
    clean = [r for r in rows if not (r.get("source_type") or "").strip() and not int(r["analysis_flags"] or 0) & 8192 and not r["name"].strip().endswith("c")]

    def infoot(l):
        return l >= 250 or l <= 65

    def inside(l, b):
        return [s[3] for s in src if core.sep_arcsec(l, b, s[0], s[1]) < s[2] * 3600]
    win = [r for r in clean if core.sep_arcsec(l_deg, b_deg, float(r["lii"]), float(r["bii"])) < radius_deg * 3600]
    rest = [r for r in clean if r not in win and infoot(float(r["lii"]))]
    wi = [(r["name"].strip(), inside(float(r["lii"]), float(r["bii"]))) for r in win]
    ri = [inside(float(r["lii"]), float(r["bii"])) for r in rest]
    nw, nr = sum(1 for _, x in wi if x), sum(1 for x in ri if x)
    p = float(fisher_exact([[nw, len(wi) - nw], [nr, len(ri) - nr]], alternative="greater")[1])
    print("%d H.E.S.S. sources (size column %s); window's clean unassociated sources inside a TeV source: %d of %d; elsewhere in the footprint: %d of %d (Fisher p %.3g)" % (
        len(src), scol, nw, len(wi), nr, len(ri), p))
    for n, x in wi:
        print("  %s: %s" % (n, x or "outside every TeV source"))
    if record:
        core.record_run(
            "fermi_unassociated_in_tev_glows", "Gamma-ray points inside TeV glows", "Fermi-LAT 4FGL-DR4 (HEASARC fermilpsc); H.E.S.S. Galactic Plane Survey (VizieR J/A+A/612/A1)",
            "clean unassociated sources within %g deg of l %.2f, b %.2f: %d" % (radius_deg, l_deg, b_deg, len(wi)), os.path.abspath(__file__), "hgpsoverlap", len(wi), "sources",
            dict(l_deg=l_deg, b_deg=b_deg, radius_deg=radius_deg, width_factor=width_factor), "the clean unassociated sources elsewhere in the survey's footprint: %d of %d inside a TeV source" % (nr, len(ri)),
            "a pile of gamma-ray points inside a TeV glow beyond the plane's rate", "%d of %d inside against %d of %d elsewhere (Fisher p %.3g)" % (nw, len(wi), nr, len(ri), p),
            [dict(target="fermi:%s" % n, stat=len(x), unit="TeV sources containing it", vet=None, note=", ".join(x) or "outside every TeV source") for n, x in wi])
        print("recorded")


gasunder_DOC = r"""Gas under a pile of gamma-ray points: the clean unassociated 4FGL-DR4 sources within a window of the plane, the dust column at each from the Schlegel, Finkbeiner and Davis map (the IRSA dust service's value at the position), against control points at the same latitudes 3-15 deg along the plane on either side of the window; gamma rays from cosmic rays striking gas that the catalogue's model misses appear as points on the gas, so points sitting on more gas than their latitude's share point to unmodelled diffuse emission.

    python -I seti.py highenergy gasunder [record] [--l-deg 312.07] [--b-deg 0] [--radius-deg 2]"""


def sfd_dust(l, b):
    """E(B-V) of the Schlegel, Finkbeiner and Davis map at a Galactic position, from the IRSA dust service."""
    import re
    t = core.get("https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr=%.4f+%.4f+gal&regSize=2.0" % (l, b), timeout=60).decode("utf-8", "replace")
    m = re.search(r"<refPixelValueSFD>\s*([0-9.]+)", t)
    return float(m.group(1)) if m else None


def gasunder_main(record=False, l_deg=312.07, b_deg=0.0, radius_deg=2.0):
    from scipy.stats import mannwhitneyu
    rows = core.heasarc("SELECT name, lii, bii, source_type, analysis_flags FROM fermilpsc WHERE ABS(bii) < %g" % (abs(b_deg) + radius_deg + 1))
    clean = [r for r in rows if not (r.get("source_type") or "").strip() and not int(r["analysis_flags"] or 0) & 8192 and not r["name"].strip().endswith("c")]
    win = [r for r in clean if core.sep_arcsec(l_deg, b_deg, float(r["lii"]), float(r["bii"])) < radius_deg * 3600]
    pts = []
    for r in win:
        l, b = float(r["lii"]), float(r["bii"])
        d = sfd_dust(l, b)
        ctrl = [sfd_dust((l + s) % 360, b) for s in list(range(-15, -2)) + list(range(3, 16))]
        ctrl = [c for c in ctrl if c is not None]
        pts.append(dict(name=r["name"].strip(), l=l, b=b, dust=d, ctrl_median=float(np.median(ctrl)), rank=float(np.mean(np.array(ctrl) < d)) if d is not None else None))
    ranks = [p["rank"] for p in pts if p["rank"] is not None]
    p_val = float(mannwhitneyu(ranks, np.linspace(0.02, 0.98, 50)).pvalue) if len(ranks) > 2 else None
    print("%d clean unassociated sources within %g deg of l %.2f, b %.2f; dust at each against 26 points at its latitude along the plane: median rank %.2f (%s)" % (
        len(pts), radius_deg, l_deg, b_deg, float(np.median(ranks)) if ranks else float("nan"), "p %.3g against uniform ranks" % p_val if p_val is not None else "too few"))
    for p in pts:
        print("  %s (l %.2f, b %+.2f): E(B-V) %s against a median %.2f at its latitude, rank %s" % (p["name"], p["l"], p["b"], p["dust"], p["ctrl_median"], p["rank"]))
    if record:
        core.record_run(
            "fermi_pile_dust", "Gas under a pile of gamma-ray points", "Fermi-LAT 4FGL-DR4 (HEASARC fermilpsc); the Schlegel, Finkbeiner and Davis dust map (IRSA dust service)",
            "clean unassociated sources within %g deg of l %.2f, b %.2f: %d" % (radius_deg, l_deg, b_deg, len(pts)), os.path.abspath(__file__), "gasunder", len(pts), "sources",
            dict(l_deg=l_deg, b_deg=b_deg, radius_deg=radius_deg), "26 points at each source's latitude 3-15 deg along the plane", "gamma-ray points sitting on more gas than their latitude's share",
            "median rank of the sources' dust among their controls %.2f%s" % (float(np.median(ranks)) if ranks else float("nan"), ", p %.3g against uniform ranks" % p_val if p_val is not None else ""),
            [dict(target="fermi:%s" % p["name"], stat=p["rank"], unit="rank of the dust column among 26 points at the same latitude", vet=None,
                  note="E(B-V) %s against a median %.2f" % (p["dust"], p["ctrl_median"])) for p in pts])
        print("recorded")


rosatgone_DOC = r"""X-ray sources that went out: the 2RXS catalogue of the ROSAT All-Sky Survey (1990-1991) in the half of the sky eROSITA-DE's first survey covers (Galactic longitude 180-360 deg), point sources at existence likelihood 10 or more and 0.05 ct/s or brighter, ten times eRASS1's limit, a degree clear of the half's edges, against the eRASS1 main catalogue (2019-2020) within 120 arcsec, the reach of a faint 2RXS position; eRASS1 covers its half whole, so a source with no counterpart faded by an order of magnitude in thirty years; each read for SIMBAD's type within 90 arcsec (flare stars caught in a flare, galaxy clusters whose gas eRASS1 catalogues as one extended source, variable AGN, X-ray binaries and cataclysmic variables in quiescence, transients), a Gaia DR3 star brighter than G 13 or within 500 pc within 20 arcsec, WISE AGN colours (W1-W2 above 0.8), a source of 10 ct/s within half a degree, three or more missing detections within half a degree (one extended source cut up), extent, and the ROSAT light curve's own variability; what is left is the catalogue's spurious tail when every such source lies below likelihood 20, against the share of the sample there.

    python -I seti.py highenergy rosatgone [record] [--exi-min 10] [--crate-min 0.05] [--match-arcsec 120] [--edge-deg 1] [--simbad-arcsec 90] [--gaia-arcsec 20] [--bright-ct 10] [--bright-deg 0.5] [--clump-deg 0.5] [--spur-exi 20]"""

# SIMBAD's types as CDS XMatch writes them, long names and short codes alike
ROSATGONE_CLUSTER = ("ClG", "GtowardsCl", "BrightestCG", "GroupG", "GrG", "CGG", "SCG", "C?G", "SuperClG", "SNRemnant", "SNR")
ROSATGONE_VARIABLE = ("AGN", "QSO", "Seyfert", "Sy1", "Sy2", "SyG", "BLLac", "Blazar", "BLL", "Bla", "LINER", "LIN", "XrayB", "XB", "LXB", "HXB", "CataclyV", "CV", "Nova", "No*",
                      "SN", "GRB", "TDE", "Pulsar", "Psr")


def rosatgone_kind(t):
    """A SIMBAD type read as a galaxy cluster or remnant, a star, a variable extragalactic or accreting source, or none of these."""
    if any(t.startswith(k) for k in ROSATGONE_CLUSTER):
        return "cluster"
    if any(k in t for k in ROSATGONE_VARIABLE):
        return "variable"
    if t.endswith("*") or t.startswith("Star") or t in ("**", "*", "Star?"):
        return "star"
    return "other" if t else None


def rosatgone_main(record=False, exi_min=10.0, crate_min=0.05, match_arcsec=120.0, edge_deg=1.0, simbad_arcsec=90.0, gaia_arcsec=20.0, bright_ct=10.0, bright_deg=0.5, clump_deg=0.5, spur_exi=20.0):
    path = core.data("rosatgone", "cat2rxs.tsv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cols = "2RXS,ExiML,CRate,e_CRate,RAJ2000,DEJ2000,GLON,GLAT,Ext,FAll,Fluxp,sigExc"
        open(path, "wb").write(core.get("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A%2BA/588/A103/cat2rxs&-out.max=unlimited&-out=" + cols, timeout=900))
    lines = [l for l in open(path, encoding="utf-8", errors="replace").read().splitlines() if l and not l.startswith("#")]
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[3:]]
    src = []
    for r in rows:
        ra, dec, l, exi, cr = (num(r.get(k)) for k in ("RAJ2000", "DEJ2000", "GLON", "ExiML", "CRate"))
        if None in (ra, dec, l, exi, cr) or not 180.0 + edge_deg <= l < 360.0 - edge_deg or exi < exi_min or cr < crate_min:
            continue
        src.append(dict(source_id=r["2RXS"].strip(), ra=ra, dec=dec, cr=cr, ecr=num(r.get("e_CRate")), exi=exi, ext=num(r.get("Ext")) or 0.0,
                        flux=num(r.get("Fluxp")), sigexc=num(r.get("sigExc"))))
    near = {}
    for k in range(0, len(src), 20000):
        for m in core.xmatch(src[k:k + 20000], "J/A+A/682/A34/erass1-m", match_arcsec):
            d = float(m["angDist"])
            near[m["source_id"]] = min(near.get(m["source_id"], 1e9), d)
    gone = [s for s in src if near.get(s["source_id"], 1e9) > match_arcsec]
    covered = gone
    print("%d 2RXS point sources in the eRASS1 half at likelihood %g and %g ct/s or brighter; %d without an eRASS1 source within %g arcsec" % (
        len(src), exi_min, crate_min, len(gone), match_arcsec), flush=True)
    sim, gaia = {}, {}
    for m in core.xmatch(covered, "simbad", simbad_arcsec):
        if m["source_id"] not in sim or float(m["angDist"]) < sim[m["source_id"]][0]:
            sim[m["source_id"]] = (float(m["angDist"]), (m.get("main_id") or "").strip(), (m.get("main_type") or m.get("otype") or "").strip())
    for m in core.xmatch(covered, "I/355/gaiadr3", gaia_arcsec):
        g, plx = num(m.get("Gmag")), num(m.get("Plx"))
        # a bright star, or any star within 500 pc, of which M dwarfs flare by an order of magnitude
        if g is not None and (g < 13 or (plx or 0) > 2.0) and (m["source_id"] not in gaia or g < gaia[m["source_id"]]):
            gaia[m["source_id"]] = g
    agn = {m["source_id"] for m in core.xmatch(covered, "II/328/allwise", gaia_arcsec)
           if num(m.get("W1mag")) is not None and num(m.get("W2mag")) is not None and num(m["W1mag"]) - num(m["W2mag"]) > 0.8}
    # a source ROSAT saw at bright_ct or more spoils its surroundings; three or more missing detections within clump_deg are one extended source cut up
    bright = [x for x in src if x["cr"] >= bright_ct]
    for s in covered:
        s["bright"] = next((b["source_id"] for b in bright if b is not s and sep_arcsec(b["ra"], b["dec"], s["ra"], s["dec"]) < bright_deg * 3600), None)
        s["clump"] = sum(1 for x in covered if sep_arcsec(x["ra"], x["dec"], s["ra"], s["dec"]) < clump_deg * 3600)
    for s in covered:
        s["simbad"] = sim.get(s["source_id"])
        t = s["simbad"][2] if s["simbad"] else ""
        s["gaia_g"] = gaia.get(s["source_id"])
        s["vet"] = ("bright-neighbour" if s["bright"] else
                    "diffuse-emission" if s["ext"] > 0 or s["clump"] >= 3 or rosatgone_kind(t) == "cluster" else
                    "flare" if rosatgone_kind(t) == "star" or s["gaia_g"] is not None else
                    "natural-type" if rosatgone_kind(t) == "variable" or s["source_id"] in agn else
                    "variable-star" if s["sigexc"] and s["sigexc"] > 3 else
                    "natural-type" if t else None)
    # the rest are spurious detections when they all sit at the catalogue's lowest likelihoods, which few of the sample do
    low = sum(x["exi"] < spur_exi for x in src) / max(len(src), 1)
    rest = [s for s in covered if s["vet"] is None]
    p_low = low ** sum(s["exi"] < spur_exi for s in rest) if rest else 1.0
    if rest and all(s["exi"] < spur_exi for s in rest) and p_low < 1e-3:
        for s in rest:
            s["vet"] = "artefact"
    print("%d left by the other vets, %d of them below likelihood %g, where %.0f per cent of the sample lies (p %.1e)" % (
        len(rest), sum(s["exi"] < spur_exi for s in rest), spur_exi, 100 * low, p_low))
    left = [s for s in covered if s["vet"] is None]
    counts = {}
    for s in covered:
        counts[s["vet"]] = counts.get(s["vet"], 0) + 1
    print("vets %s; %d unexplained" % (counts, len(left)))
    for s in sorted(left, key=lambda s: -s["cr"])[:30]:
        print("  2RXS %s: %.3f +- %.3f ct/s at RA %.4f Dec %+.4f, likelihood %.0f" % (s["source_id"], s["cr"], s["ecr"] or 0, s["ra"], s["dec"], s["exi"]))
    if record:
        core.record_run(
            "rosat_erass1_gone", "X-ray sources that went out", "2RXS (VizieR J/A+A/588/A103) and eRASS1 main (J/A+A/682/A34/erass1-m) by CDS XMatch; SIMBAD; Gaia DR3",
            "%d 2RXS point sources at likelihood %g and %g ct/s or brighter at Galactic longitude 180-360 deg" % (len(src), exi_min, crate_min), os.path.abspath(__file__), "rosatgone",
            len(src), "sources", dict(exi_min=exi_min, crate_min=crate_min, match_arcsec=match_arcsec, edge_deg=edge_deg, simbad_arcsec=simbad_arcsec, gaia_arcsec=gaia_arcsec, bright_ct=bright_ct, bright_deg=bright_deg, clump_deg=clump_deg, spur_exi=spur_exi),
            dict(gone=len(gone), vets=counts), "a 2RXS source ten times eRASS1's limit with no eRASS1 source within %g arcsec" % match_arcsec,
            "%d of %d without an eRASS1 counterpart: %s; %d unexplained" % (len(covered), len(src), ", ".join("%d %s" % (v, k) for k, v in sorted(counts.items(), key=lambda x: -x[1]) if k), len(left)),
            [dict(target="2rxs:%s" % s["source_id"], stat=round(s["cr"], 3), unit="ROSAT ct/s, absent from eRASS1", vet=s["vet"],
                  note="likelihood %.0f; SIMBAD %s; Gaia G %s" % (s["exi"], s["simbad"], "%.1f" % s["gaia_g"] if s["gaia_g"] is not None else "-"))
             for s in covered])
        print("recorded")


APPROACHES = {
    "rosatgone": (lambda: core.run_cli(rosatgone_main), rosatgone_DOC),
    "gasunder": (lambda: core.run_cli(gasunder_main), gasunder_DOC),
    "hgpsoverlap": (lambda: core.run_cli(hgpsoverlap_main), hgpsoverlap_DOC),
    "unasclass": (lambda: core.run_cli(unasclass_main), unasclass_DOC),
    "tadoublets": (lambda: core.run_cli(tadoublets_main), tadoublets_DOC),
    "barlagrange": (lambda: core.run_cli(barlagrange_main), barlagrange_DOC),
    "gammaflares": (lambda: core.run_cli(gammaflares_main), gammaflares_DOC),
    "hardgamma": (lambda: core.run_cli(hardgamma_main), hardgamma_DOC),
    "grbgalaxies": (lambda: core.run_cli(grbgalaxies_main), grbgalaxies_DOC),
    "frbgalaxies": (lambda: core.run_cli(frbgalaxies_main), frbgalaxies_DOC),
    "fermimoved": (lambda: core.run_cli(fermimoved_main), fermimoved_DOC),
    "grbtogether": (lambda: core.run_cli(grbtogether_main), grbtogether_DOC),
    "copygrbs": (lambda: core.run_cli(copygrbs_main), copygrbs_DOC),
    "nudoublets": (lambda: core.run_cli(nudoublets_main), nudoublets_DOC),
    "grbetz": (lambda: core.run_cli(grbetz_main), grbetz_DOC),
    "nuetz": (lambda: core.run_cli(nuetz_main), nuetz_DOC),
    "nugrb": (lambda: core.run_cli(nugrb_main), nugrb_DOC),
    "nocorona": (lambda: core.run_cli(nocorona_main), nocorona_DOC),
    "grbfrb": (lambda: core.run_cli(grbfrb_main), grbfrb_DOC),
    "gammapeak": (lambda: core.run_cli(gammapeak_main), gammapeak_DOC),
    "coldbdw3": (lambda: core.run_cli(coldbdw3_main), coldbdw3_DOC),
    "grbrepeat": (lambda: core.run_cli(grbrepeat_main), grbrepeat_DOC),
    "bursts511": (lambda: core.run_cli(bursts511_main), bursts511_DOC),
    "tevnear": (lambda: core.run_cli(tevnear_main), tevnear_DOC),
    "gwequal": (lambda: core.run_cli(gwequal_main), gwequal_DOC),
    "deadstars": (lambda: core.run_cli(deadstars_main), deadstars_DOC),
    "etzxray": (lambda: core.run_cli(etzxray_main), etzxray_DOC),
    "nearbat": (lambda: core.run_cli(nearbat_main), nearbat_DOC),
    "coldwdgamma": (lambda: core.run_cli(coldwdgamma_main), coldwdgamma_DOC),
    "coldwduv": (lambda: core.run_cli(coldwduv_main), coldwduv_DOC),
    "coldbduv": (lambda: core.run_cli(coldbduv_main), coldbduv_DOC),
    "coldbdx": (lambda: core.run_cli(coldbdx_main), coldbdx_DOC),
    "wdmass": (lambda: core.run_cli(wdmass_main), wdmass_DOC),
    "grbstars": (lambda: core.run_cli(grbstars_main), grbstars_DOC),
    "coldwdx": (lambda: core.run_cli(coldwdx_main), coldwdx_DOC),
    "dynamo": (dynamo_cli, dynamo_DOC),
    "doublets": (doublets_cli, doublets_DOC),
    "sources": (sources_cli, sources_DOC),
    "chandra": (chandra_cli, chandra_DOC),
    "xmm": (xmm_cli, xmm_DOC),
    "lineonly": (lineonly_cli, lineonly_DOC),
    "planets": (planets_cli, planets_DOC),
    "icecube": (icecube_cli, icecube_DOC),
    "mm": (mm_cli, mm_DOC),
}


