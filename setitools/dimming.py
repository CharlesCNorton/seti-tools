"""Dimming and dips of nearby stars from the Harvard plates to Gaia."""
import argparse
import contextlib
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from setitools import core
from setitools.core import cstar, gaia_g_scatter, num, quiet_warnings, running


# --------------------------------------------------------------------------------------------------------------

hipparcos_DOC = r"""Dips in Hipparcos epoch photometry within 100 pc: 5 robust deviations and 0.03 mag, shared by the pass partner, vetted for variability, binarity and satellite events.

    python -I seti.py dimming hipparcos"""

hipparcos_ROOT = os.environ.get("HIP_ROOT", core.data("hipparcos"))
hipparcos_HERE = core.RESULTS
hipparcos_BAD = 8 | 16 | 32 | 64 | 128 | 256
hipparcos_PAIR_D = 0.02
hipparcos_SIMBAD_EXPLAINED = ("EB*", "SB*", "**", "BY*", "RS*", "Er*", "Fl*", "Ro*", "V*", "dS*", "gD*", "El*", "Pu*", "Ae*",
                    "TT*", "Y*O", "WU*", "Al*", "bL*", "RR*", "Ce*", "LP*", "Mi*", "sr*", "a2*", "Be*", "Em*")


def hipparcos_read_hep():
    stars, cur = {}, None
    with gzip.open(os.path.join(hipparcos_ROOT, "hep.gz"), "rt", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 10:
                hip = int(parts[0])
                # fields: HIP | component | V-I | Ntr | Nh and Hpmag | e_Hp | Hpmax | Hpmin and period | T0 | VType | flags
                cur = dict(hip=hip, comp=parts[1].strip(), vtype=parts[9].strip() if len(parts) > 9 else "", t=[], m=[],
                           e=[], q=[])
                stars[hip] = cur
            elif cur is not None and len(parts) >= 4:
                try:
                    cur["t"].append(float(parts[0]))
                    cur["m"].append(float(parts[1]))
                    cur["e"].append(float(parts[2]))
                    cur["q"].append(int(parts[3]))
                except ValueError:
                    continue
    return stars


def hipparcos_main():
    hep = hipparcos_read_hep()
    g = core.gaia("SELECT h.original_ext_source_id AS hip, g.source_id, g.parallax, g.bp_rp, g.ruwe, "
                      "g.phot_g_mean_mag, g.phot_variable_flag FROM gaiadr3.hipparcos2_best_neighbour AS h JOIN "
                      "gaiadr3.gaia_source AS g ON h.source_id = g.source_id WHERE g.parallax > 10")
    near = {int(r["hip"]): r for r in g}
    print("%d Hipparcos stars with epoch photometry; %d within 100 pc through Gaia DR3" % (len(hep), len(near)), flush=True)
    res = []
    for hip, gr in near.items():
        s = hep.get(hip)
        if not s or len(s["t"]) < 40:
            continue
        t, m, e, q = (np.array(s[k]) for k in ("t", "m", "e", "q"))
        ok = (q & hipparcos_BAD) == 0
        t, m, e = t[ok], m[ok], e[ok]
        if len(t) < 40:
            continue
        med = float(np.median(m))
        sd = float(1.4826 * np.median(np.abs(m - med)))
        thr = max(5 * sd, 0.03)
        dip = m - med > thr
        if not dip.any():
            continue
        i = int(np.argmax(m - med))
        partners = np.where((np.abs(t - t[i]) < hipparcos_PAIR_D) & (np.arange(len(t)) != i))[0]
        shared = bool(partners.size) and bool(np.any(m[partners] - med > 0.5 * (m[i] - med)))
        recur = int(np.sum(m - med > 0.5 * (m[i] - med))) - 1
        dip_t = [float(x) for x in t[m - med > thr]]
        res.append(dict(hip=hip, gaia=gr["source_id"], d_pc=1000 / float(gr["parallax"]), bp_rp=gr["bp_rp"], dip_t=dip_t,
                        ruwe=float(gr["ruwe"]) if gr["ruwe"] else None, gvar=gr["phot_variable_flag"], vtype=s["vtype"],
                        n=len(t), sd=sd, depth=float(m[i] - med), t_dip=float(t[i]) + 2440000.0, n_dip=int(dip.sum()),
                        partners=int(partners.size), shared=shared, recur=recur))
    sim = {}
    # SIMBAD types by HIP identifier
    ids = [r["hip"] for r in res]
    for k in range(0, len(ids), 300):
        q = ("SELECT i.id, b.main_id, b.otype FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)"
             % ",".join("'HIP %d'" % h for h in ids[k:k + 300]))
        for r in core.simbad(q):
            sim[int(r["id"].split()[-1])] = (r["main_id"], r["otype"])
    # simultaneous dips across the catalogue are the satellite's attitude events
    allt = []
    for s_ in hep.values():
        if len(s_["t"]) < 40:
            continue
        t_, m_, q_ = (np.array(s_[k]) for k in ("t", "m", "q"))
        ok_ = (q_ & hipparcos_BAD) == 0
        if ok_.sum() < 40:
            continue
        mm = m_[ok_]
        med_ = np.median(mm)
        sd_ = 1.4826 * np.median(np.abs(mm - med_))
        allt += list(t_[ok_][mm - med_ > max(5 * sd_, 0.03)])
    allt = np.sort(np.array(allt))
    print("%d dipping transits over the whole catalogue" % len(allt), flush=True)
    for r in res:
        r["date_shared"] = int(np.searchsorted(allt, r["t_dip"] - 2440000.0 + 0.02) -
                               np.searchsorted(allt, r["t_dip"] - 2440000.0 - 0.02)) -             sum(1 for x in r["dip_t"] if abs(x - (r["t_dip"] - 2440000.0)) < 0.02)
    # VSX names eclipsing binaries that SIMBAD files as plain stars
    gpos = {x["source_id"]: x for x in core.gaia("SELECT source_id, ra, dec FROM gaiadr3.gaia_source WHERE source_id "
                                                     "IN (%s)" % ",".join(str(r["gaia"]) for r in res))}
    vsx = {}
    pts = [dict(source_id=str(r["hip"]), ra=float(gpos[str(r["gaia"])]["ra"]), dec=float(gpos[str(r["gaia"])]["dec"]))
           for r in res if str(r["gaia"]) in gpos]
    for k in range(0, len(pts), 2000):
        for m in core.xmatch(pts[k:k + 2000], "B/vsx/vsx", 30.0):
            vsx.setdefault(int(m["source_id"]), "%s %s" % (m.get("Name"), m.get("Type")))
    for r in res:
        nm, ot = sim.get(r["hip"], ("", ""))
        r["simbad"], r["otype"] = nm, ot
        why = []
        if r["vtype"] in ("P", "D"):
            why.append("Hipparcos type %s" % r["vtype"])
        if r["hip"] in vsx:
            why.append("VSX %s" % vsx[r["hip"]])
        if r["ruwe"] and r["ruwe"] > 1.4:
            why.append("RUWE %.1f" % r["ruwe"])
        if any(str(ot).startswith(x) for x in hipparcos_SIMBAD_EXPLAINED):
            why.append("SIMBAD %s" % ot)
        if r["date_shared"] >= 3:
            why.append("%d other catalogue stars dip within half an hour" % r["date_shared"])
        if not r["shared"]:
            why.append("the pass partner 20.6 minutes away %s" % ("undimmed" if r["partners"] else "missing"))
        r["why"] = why
    json.dump(res, open(os.path.join(hipparcos_HERE, "hip_dippers.json"), "w"), indent=1, default=str)
    left = [r for r in res if not r["why"]]
    print("%d stars within 100 pc have a transit 5 robust deviations (0.03 mag) faint; %d unexplained" % (len(res), len(left)))
    from collections import Counter
    print("  " + ", ".join("%s %d" % kv for kv in Counter(r["why"][0].split(" (")[0].split(" %")[0] for r in res
                                                            if r["why"]).most_common(8)))
    for r in sorted(left, key=lambda r: -r["depth"] / max(r["sd"], 0.005))[:30]:
        print("  HIP %d %-22s %-6s %5.1f pc BP-RP %.2f RUWE %.2f Hip type %s: %d transits, deepest %.3f mag (scatter %.3f) "
              "on JD %.2f, %d dipping, partner shares %s, recurs %d, Gaia variable %s"
              % (r["hip"], r["simbad"][:22], r["otype"][:6], r["d_pc"], float(r["bp_rp"] or 0), r["ruwe"] or 0, r["vtype"],
                 r["n"], r["depth"], r["sd"], r["t_dip"], r["n_dip"], r["shared"], r["recur"], r["gvar"]))


def hipparcos_cli():
    quiet_warnings()
    hipparcos_main()


# --------------------------------------------------------------------------------------------------------------

gaia_DOC = r"""Dips in Gaia DR3 epoch photometry of skewed variables within 100 pc, recurring, shared by both fields of view and BP/RP, aperiodic; survivors checked in TESS.

    python -I seti.py dimming gaia fetch
    python -I seti.py dimming gaia analyse"""

gaia_ROOT = os.environ.get("DIPPER_ROOT", core.data("gaia_dippers"))
gaia_HERE = core.RESULTS
gaia_NATURAL = {"ECL", "EP", "YSO", "RS", "CEP", "RR", "LPV", "MICROLENSING", "AGN", "SN", "ELL"}
gaia_DL = "https://gea.esac.esa.int/data-server/data?RETRIEVAL_TYPE=EPOCH_PHOTOMETRY&ID=Gaia+DR3+%s&FORMAT=CSV&VALID_DATA=true"


def gaia_candidates():
    base = core.gaia("SELECT source_id, parallax, phot_g_mean_mag, bp_rp, ruwe FROM gaiadr3.gaia_source WHERE "
                         "parallax > 10 AND parallax_over_error > 10 AND phot_variable_flag = 'VARIABLE'")
    by = {r["source_id"]: r for r in base}
    ids = list(by)
    out = []
    for k in range(0, len(ids), 400):
        q = ("SELECT v.source_id, v.num_selected_g_fov, v.median_mag_g_fov, v.range_mag_g_fov, v.skewness_mag_g_fov, "
             "v.mad_mag_g_fov, c.best_class_name, c.best_class_score FROM gaiadr3.vari_summary AS v LEFT OUTER JOIN "
             "gaiadr3.vari_classifier_result AS c ON v.source_id = c.source_id WHERE v.source_id IN (%s)" % ",".join(ids[k:k + 400]))
        for r in core.gaia(q):
            r.update(by[r["source_id"]])
            out.append(r)
    dip = [r for r in out if (num(r["skewness_mag_g_fov"]) or 0) > 1.5 and (num(r["range_mag_g_fov"]) or 0) > 0.05
           and (num(r["num_selected_g_fov"]) or 0) >= 20]
    return len(base), dip


gaia_VIZ = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def gaia_vizier_epochs(sid):
    """Per-transit epoch table from VizieR's I/355/epphot, with DataLink column names."""
    import urllib.parse
    q = ('SELECT TimeG, Gmag, TimeBP, BPmag, TimeRP, RPmag, GrVFlag, BPrVFlag, RPrVFlag FROM "I/355/epphot" '
         "WHERE Source = %s" % sid)
    b = core.get(gaia_VIZ + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q)),
                timeout=120, retries=3, soft=True, pause=3)
    if not b:
        return None
    rows = list(csv.DictReader(io.StringIO(b.decode("utf-8", "replace"))))
    if not rows:
        return None
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["g_transit_time", "g_transit_mag", "bp_obs_time", "bp_mag", "rp_obs_time", "rp_mag",
                "variability_flag_g_reject", "variability_flag_bp_reject", "variability_flag_rp_reject"])
    for r in rows:
        w.writerow([r["TimeG"], r["Gmag"], r["TimeBP"], r["BPmag"], r["TimeRP"], r["RPmag"]] +
                   ["true" if r[k] == "1" else "false" for k in ("GrVFlag", "BPrVFlag", "RPrVFlag")])
    return out.getvalue().encode()


def gaia_fetch_one(sid):
    p = os.path.join(gaia_ROOT, "epoch", sid + ".csv")
    if os.path.exists(p):
        return 1
    # the unfiltered table carries the flags gaia_series applies
    b = (core.get(gaia_DL % sid, timeout=120, retries=1, soft=True, pause=3)
         or core.get(gaia_DL.replace("VALID_DATA=true", "VALID_DATA=false") % sid, timeout=120, retries=1, soft=True, pause=3)
         or gaia_vizier_epochs(sid))
    if not b:
        return 0
    open(p, "wb").write(b)
    return 1


def gaia_fetch():
    os.makedirs(os.path.join(gaia_ROOT, "epoch"), exist_ok=True)
    n, dip = gaia_candidates()
    keep = [r for r in dip if (r.get("best_class_name") or "") not in gaia_NATURAL]
    json.dump(dict(n_variable=n, dip=dip, keep=keep), open(os.path.join(gaia_ROOT, "candidates.json"), "w"), indent=1)
    print("%d variable sources within 100 pc; %d dip-skewed; %d after the classifier's natural classes" % (n, len(dip), len(keep)),
          flush=True)
    with ThreadPoolExecutor(4) as ex:
        got = sum(ex.map(gaia_fetch_one, [r["source_id"] for r in keep]))
    print("%d epoch photometry files" % got)


def gaia_series(sid):
    p = os.path.join(gaia_ROOT, "epoch", sid + ".csv")
    t = open(p, encoding="utf-8", errors="replace").read()
    rows = list(csv.DictReader(io.StringIO("\n".join(ln for ln in t.splitlines() if not ln.startswith("#")))))
    out = {}
    # one row per field-of-view transit with G, BP and RP
    for band, tc, mc, rej in (("G", "g_transit_time", "g_transit_mag", "variability_flag_g_reject"),
                              ("BP", "bp_obs_time", "bp_mag", "variability_flag_bp_reject"),
                              ("RP", "rp_obs_time", "rp_mag", "variability_flag_rp_reject")):
        b = [r for r in rows if num(r.get(mc)) is not None and num(r.get(tc)) is not None
             and r.get("rejected_by_photometry", "false") != "true" and r.get(rej, "false") != "true"]
        out[band] = (np.array([float(r[tc]) for r in b]), np.array([float(r[mc]) for r in b]))
    return out


def gaia_explain(x):
    """Natural cause of a star's dips, or an empty string."""
    if x["dips"] < 2:
        return "one dipping epoch (a single-transit outlier)"
    if not x.get("spans", True):
        return "every deep dip is one transit with its neighbours 106 minutes away undimmed (single-transit outliers)"
    if x["ruwe"] and x["ruwe"] > 1.4:
        return "RUWE %.1f: a close binary measured differently at each scan angle (period %.0f d)" % (x["ruwe"], x["period"])
    if x["power"] >= 0.3:
        return "periodic at %.3f d (power %.2f): an eclipsing binary" % (x["period"], x["power"])
    if x.get("share") is not None and x["share"] < 0.5:
        return "G-only dips: BP and RP in the same transits share %.0f per cent of the G drop" % (100 * x["share"])
    if x["dcol"] is not None and x["dcol"] > 0.05:
        return "redder in the dips by %.2f mag: dust or a cool companion" % x["dcol"]
    return ""


def gaia_analyse():
    from astropy.timeseries import LombScargle
    c = json.load(open(os.path.join(gaia_ROOT, "candidates.json")))
    res = []
    for r in c["keep"]:
        try:
            s = gaia_series(r["source_id"])
        except Exception:
            continue
        t, g = s["G"]
        if len(g) < 20:
            continue
        med = np.median(g)
        sd = 1.4826 * np.median(np.abs(g - med))
        dips = g > med + 4 * max(sd, 0.003)
        tb, bp = s["BP"]
        tr, rp = s["RP"]
        dcol = None
        share = None
        if len(bp) > 10 and len(rp) > 10:
            # BP and RP colour and share of each G dip; an occulter dims all three
            col, sh = [], []
            for tt, dg in zip(t[dips], g[dips] - med):
                i, j = np.argmin(np.abs(tb - tt)), np.argmin(np.abs(tr - tt))
                if abs(tb[i] - tt) < 0.001 and abs(tr[j] - tt) < 0.001:
                    col.append(bp[i] - rp[j])
                    sh.append(((bp[i] - np.median(bp)) + (rp[j] - np.median(rp))) / 2 / dg)
            base = np.median(bp) - np.median(rp)
            dcol = float(np.median(col) - base) if col else None
            share = float(np.median(sh)) if sh else None
        fr, pw = LombScargle(t, g).autopower(minimum_frequency=1 / 500.0, maximum_frequency=5.0, samples_per_peak=5)
        k = int(np.argmax(pw))
        # the deepest dip must recur at half depth or span neighbouring transits (fields of view 106 min apart)
        depth = g - med
        i_max = int(np.argmax(depth))
        deep_idx = np.where(depth > 0.5 * depth[i_max])[0]
        deep = int(len(deep_idx))
        spans = False
        for i in deep_idx:
            near = np.abs(t - t[i]) < 0.3
            near[i] = False
            spans = spans or bool(np.any(dips & near))
        mg = float(r["phot_g_mean_mag"]) + 5 * math.log10(float(r["parallax"]) / 100.0)
        res.append(dict(sid=r["source_id"], cls=r.get("best_class_name") or "", n=len(g), dips=int(dips.sum()),
                        deep=deep, spans=spans,
                        depth=float(np.max(g) - med), sd=float(sd), dcol=dcol, share=share, period=float(1 / fr[k]), power=float(pw[k]),
                        abs_g=mg, bp_rp=num(r["bp_rp"]), ruwe=num(r["ruwe"]), d_pc=1000 / float(r["parallax"])))
    for x in res:
        x["why"] = gaia_explain(x)
    left = [x for x in res if not x["why"]]
    print("%d light curves read; dipping epochs (4 robust deviations below the median) in %d; %d without a natural "
          "explanation" % (len(res), sum(1 for x in res if x["dips"] > 0), len(left)))
    from collections import Counter
    print("  " + ", ".join("%s %d" % kv for kv in Counter(x["why"].split(" (")[0] for x in res if x["why"]).most_common()))
    for x in sorted(left, key=lambda x: -x["depth"])[:30]:
        print("  Gaia DR3 %s %-10s %5.1f pc M_G %5.2f BP-RP %5.2f RUWE %4.1f: %d epochs, %d dipping, deepest %.3f mag "
              "(scatter %.3f), BP-RP change in dips %s, best period %.3f d (power %.2f)"
              % (x["sid"], x["cls"], x["d_pc"], x["abs_g"], x["bp_rp"] or 0, x["ruwe"] or 0, x["n"], x["dips"], x["depth"],
                 x["sd"], "%+.3f" % x["dcol"] if x["dcol"] is not None else "-", x["period"], x["power"]))
    json.dump(res, open(os.path.join(gaia_HERE, "gaia_dippers.json"), "w"), indent=1)


def gaia_tess():
    """TESS check of unexplained dippers: deepest dip per sector against scatter and a box search."""
    from astropy.timeseries import BoxLeastSquares
    res = json.load(open(os.path.join(gaia_HERE, "gaia_dippers.json")))
    for x in [x for x in res if not x["why"]]:
        g = core.gaia("SELECT ra, dec FROM gaiadr3.gaia_source WHERE source_id = %s" % x["sid"])[0]
        tic, lcs = core.curves(float(g["ra"]), float(g["dec"]))
        print("Gaia DR3 %s (TIC %s, Gaia dips to %.3f mag): %d TESS sectors" % (x["sid"], tic, x["depth"], len(lcs)),
              flush=True)
        if not lcs:
            continue
        from scipy.ndimage import median_filter
        half = 1 - 10 ** (-0.4 * x["depth"] / 2)
        for s, src, t, f in lcs:
            sm = median_filter(f, size=5, mode="nearest")
            sd = 1.4826 * np.median(np.abs(f - np.median(f)))
            # separate dips at least half the Gaia depth
            low = sm < 1 - half
            runs = int(np.sum(low[1:] & ~low[:-1]) + (1 if low[0] else 0))
            # box search from 0.3 d to half the sector
            pg = BoxLeastSquares(t, f).power(np.geomspace(0.3, max(np.ptp(t) / 2, 0.6), 4000), [0.05, 0.1, 0.2])
            k = int(np.argmax(pg.power))
            print("  S%d %s: %d points, scatter %.4f, deepest 5-point dip %.4f at BTJD %.2f, %d dips below half the Gaia "
                  "depth; box search %.4f d, depth %.4f, S/N %.1f"
                  % (s, src, len(f), sd, 1 - sm.min(), t[int(np.argmin(sm))], runs, pg.period[k], pg.depth[k],
                     pg.depth_snr[k]), flush=True)


def gaia_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "tess"))
    a = p.parse_args()
    {"fetch": gaia_fetch, "analyse": gaia_analyse, "tess": gaia_tess}[a.mode]()


def gaia_cli():
    quiet_warnings()
    gaia_main()


# --------------------------------------------------------------------------------------------------------------

hip_gaia_DOC = r"""Hipparcos-to-Gaia dimming within 100 pc: G - Hp against BP-RP, 5 robust deviations fainter, vetted for variability, companions, binarity and saturation; rotation in TESS.

    python -I seti.py dimming hip_gaia
    python -I seti.py dimming hip_gaia rotation    (the TESS step alone, on the stars the saved run left unexplained)
    python -I seti.py dimming hip_gaia record      (the coverage store's line for the saved run)"""

hip_gaia_HERE = core.RESULTS
hip_gaia_EDGE = 25.0
hip_gaia_CSTAR = 0.015
hip_gaia_SIMBAD_VAR = ("V*", "Ro*", "BY*", "RS*", "Er*", "Fl*", "EB*", "El*", "Pu*", "dS*", "gD*", "Ae*", "TT*", "Y*O", "WU*",
              "Al*", "bL*", "RR*", "Ce*", "LP*", "Mi*", "sr*", "a2*", "Be*", "Em*", "Ir*", "Or*", "Pe*", "WD*", "ZZ*")


def hip_gaia_main():
    g = core.gaia("SELECT h.original_ext_source_id AS hip, g.source_id, g.ra, g.dec, g.parallax, g.phot_g_mean_mag, "
                      "g.bp_rp, g.ruwe, g.phot_variable_flag, g.phot_g_mean_flux, g.phot_g_mean_flux_error, "
                      "g.phot_g_n_obs, g.phot_bp_rp_excess_factor, g.ipd_frac_multi_peak FROM gaiadr3.hipparcos2_best_neighbour AS h JOIN gaiadr3.gaia_source AS g ON "
                      "h.source_id = g.source_id WHERE g.parallax > 10 AND g.phot_g_mean_mag > 6 AND g.bp_rp IS NOT NULL")
    hip = {}
    from astroquery.vizier import Vizier
    v = Vizier(columns=["HIP", "Hpmag", "e_Hpmag", "Hpscat", "HvarType", "MultFlag", "Ncomp", "CCDM"], row_limit=-1)
    t = v.query_constraints(catalog="I/239/hip_main", Plx=">9")[0]
    for r in t:
        hip[int(r["HIP"])] = dict(hp=float(r["Hpmag"]), e_hp=float(r["e_Hpmag"]),
                                  hpscat=float(r["Hpscat"]) if r["Hpscat"] is not np.ma.masked else float("nan"),
                                  var=str(r["HvarType"]).strip(), mult=str(r["MultFlag"]).strip().replace("--", ""),
                                  ncomp=int(r["Ncomp"]) if r["Ncomp"] is not np.ma.masked else 1)
    rows = []
    for r in g:
        h = hip.get(int(r["hip"]))
        if not h or not np.isfinite(h["hp"]):
            continue
        f, fe, n = float(r["phot_g_mean_flux"]), float(r["phot_g_mean_flux_error"]), int(r["phot_g_n_obs"])
        rows.append(dict(hip=int(r["hip"]), gaia=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"]),
                         g=float(r["phot_g_mean_mag"]), bp_rp=float(r["bp_rp"]), ruwe=float(r["ruwe"] or 0),
                         gvar=r["phot_variable_flag"], gscat=gaia_g_scatter(f, fe, n),
                         cstar=cstar(float(r["phot_bp_rp_excess_factor"]), float(r["bp_rp"])),
                         multipeak=int(r["ipd_frac_multi_peak"] or 0), **h))
    print("%d stars within 100 pc with G above 6 and a Hipparcos Hp" % len(rows), flush=True)
    # Gaia neighbours within 20 arcsec summed into Hipparcos's 38 arcsec field; light beyond 10 arcsec bounds the residual
    pts = [dict(source_id=str(r["gaia"]), ra=r["ra"], dec=r["dec"]) for r in rows]
    extra = {}
    for k in range(0, len(pts), 3000):
        for m in core.xmatch(pts[k:k + 3000], "I/355/gaiadr3", hip_gaia_EDGE):
            if str(m.get("Source")) == m["source_id"] or not m.get("Gmag"):
                continue
            extra.setdefault(m["source_id"], []).append((float(m["Gmag"]), float(m["angDist"])))

    def summed(r, lo, hi):
        return -2.5 * math.log10(10 ** (-0.4 * r["g"]) + sum(10 ** (-0.4 * x[0]) for x in r["nb"] if lo < x[1] <= hi))

    for r in rows:
        r["nb"] = extra.get(str(r["gaia"]), [])
        r["neighbours"] = sum(1 for x in r["nb"] if x[0] < r["g"] + 5)
        r["g_sum"] = summed(r, 0, 20.0)
        r["d"] = r["g_sum"] - r["hp"]
        r["d_core"] = summed(r, 0, 10.0) - r["hp"]
        r["d_all"] = summed(r, 0, hip_gaia_EDGE) - r["hp"]
    const = [r for r in rows if r["var"] in ("", "C") and r["ruwe"] < 1.4 and r["neighbours"] == 0]
    cc = np.sort([r["bp_rp"] for r in const])
    med, sd = running(np.array([r["bp_rp"] for r in const]), np.array([r["d"] for r in const]),
                      [r["bp_rp"] for r in rows])
    for r, m, s in zip(rows, med, sd):
        # colour term extrapolated with fewer than 20 constant stars within 0.1 mag
        r["colour_support"] = int(np.searchsorted(cc, r["bp_rp"] + 0.1) - np.searchsorted(cc, r["bp_rp"] - 0.1))
        r["resid"] = r["d"] - m
        r["resid_range"] = sorted((r["d_core"] - m, r["d_all"] - m))
        r["sd"] = math.hypot(max(s, 0.003), r["e_hp"] or 0.002)
        r["z"] = r["resid"] / r["sd"]
    # photometric scatter against the median at the star's magnitude, both missions
    for key, mag in (("gscat", "g"), ("hpscat", "hp")):
        ok = [r for r in rows if np.isfinite(r[key])]
        m, _ = running(np.array([r[mag] for r in ok]), np.array([r[key] for r in ok]), [r[mag] for r in ok], half=150)
        for r, mm in zip(ok, m):
            r[key + "_ratio"] = r[key] / mm
    odd = [r for r in rows if abs(r["z"]) >= 5]
    print("%d stars changed by 5 robust deviations or more; SIMBAD and WDS for them" % len(odd), flush=True)
    sim = {int(k.split()[-1]): v for k, v in core.simbad_otypes(["HIP %d" % r["hip"] for r in odd]).items()}
    wds = {}
    opts = [dict(source_id=str(r["hip"]), ra=r["ra"], dec=r["dec"]) for r in odd]
    for k in range(0, len(opts), 2000):
        for m in core.xmatch(opts[k:k + 2000], "B/wds/wds", 20.0):
            wds.setdefault(int(m["source_id"]), []).append(m)
    for r in odd:
        why = []
        if r["var"] not in ("", "C"):
            why.append("Hipparcos variability type %s" % r["var"])
        if r.get("hpscat_ratio", 0) >= 3:
            why.append("Hipparcos scatter %.3f mag, %.1f times the median" % (r["hpscat"], r["hpscat_ratio"]))
        if r.get("gscat_ratio", 0) >= 3:
            why.append("Gaia G scatter %.3f mag, %.1f times the median" % (r["gscat"], r["gscat_ratio"]))
        if r["gvar"] == "VARIABLE":
            why.append("Gaia variable")
        types = sorted(x for x in sim.get(r["hip"], ()) if x in hip_gaia_SIMBAD_VAR)
        if types:
            why.append("SIMBAD %s" % ",".join(types))
        if r["ruwe"] > 1.4:
            why.append("RUWE %.1f" % r["ruwe"])
        if r["colour_support"] < 20:
            why.append("colour term extrapolated (%d constant stars within 0.1 mag of BP-RP %.2f)"
                       % (r["colour_support"], r["bp_rp"]))
        # a companion lost by the G fit raises C* by about the lost fraction
        frac = 1 - 10 ** (-0.4 * r["resid"])
        if r["z"] > 0 and r["cstar"] > hip_gaia_CSTAR and r["cstar"] >= 0.5 * frac:
            why.append("Gaia's G misses a close companion that BP and RP collect (C* %.3f, %d per cent of transits "
                       "with two image peaks)" % (r["cstar"], r["multipeak"]))
        lo, hi = r["resid_range"]
        if hi > lo and (lo <= 0 <= hi or min(abs(lo), abs(hi)) / r["sd"] < 5):
            why.append("Gaia neighbours 10-%d arcsec out, where Hipparcos summed part of their light, span "
                       "%+.3f to %+.3f mag" % (hip_gaia_EDGE, lo, hi))
        if r["mult"] in ("G", "O", "V", "X"):
            why.append("Hipparcos %s solution" % {"G": "acceleration", "O": "orbital", "V": "variability-induced",
                                                   "X": "stochastic"}[r["mult"]])
        # WDS companions within 20 arcsec and 5 mag unmatched by Gaia, allowed 0.5 mag brighter
        missing = 0.0
        for m in wds.get(r["hip"], []):
            comp = str(m.get("Comp") or "").strip()
            if comp and not comp.startswith("A"):
                continue
            s = num(m.get("sep2"))
            s = s if s is not None else num(m.get("sep1"))
            m1, m2 = num(m.get("mag1")), num(m.get("mag2"))
            if s is None or m1 is None or m2 is None or s > 20 or m2 - m1 > 5:
                continue
            if any(abs(a - s) < 1.5 + 0.1 * s for _, a in r["nb"]):
                continue
            missing += 10 ** (-0.4 * (m2 - m1 - 0.5))
            why_w = "WDS companion %.1f arcsec, %.1f mag fainter, not in Gaia" % (s, m2 - m1)
        if missing and r["z"] > 0:
            r["resid_wds"] = r["resid"] - 2.5 * math.log10(1 + missing)
            if r["resid_wds"] / r["sd"] < 5:
                why.append(why_w + " (accounts for up to %.3f mag)" % (2.5 * math.log10(1 + missing)))
        r["why"] = why
    for r in rows:
        r.pop("nb", None)
    resid = np.array([r["resid"] for r in const])
    print("G minus Hp after the colour term: constant single stars' robust spread %.4f mag" %
          (1.4826 * np.median(np.abs(resid - np.median(resid)))))
    hip_gaia_rotation([r for r in odd if not r["why"]])
    json.dump(rows, open(os.path.join(hip_gaia_HERE, "hip_gaia_dimming.json"), "w"), indent=1, default=str)
    hip_gaia_summarise(rows)


def hip_gaia_rotation(left):
    """Rotational modulation in TESS (power 0.5 below 15 d, range 0.5 per cent) names spotted stars."""
    for r in left:
        _, lcs = core.curves(r["ra"], r["dec"])
        m = core.modulation(lcs)
        r["tess"] = m
        if m and m["power"] >= 0.5 and m["range"] >= 0.005:
            r["why"].append("spotted rotator: TESS period %.2f d, 2-98 percentile range %.3f over %d sectors"
                            % (m["period"], m["range"], m["sectors"]))


def hip_gaia_summarise(rows):
    from collections import Counter
    odd = [r for r in rows if "why" in r]
    for label, sel in (("fainter", [r for r in odd if r["z"] >= 5]), ("brighter", [r for r in odd if r["z"] <= -5])):
        left = [r for r in sel if not r["why"]]
        print("%d stars %s in Gaia than Hipparcos predicts by 5 robust deviations; %d without a natural cause"
              % (len(sel), label, len(left)))
        print("  first causes: " + ", ".join("%s %d" % kv for kv in Counter(
            r["why"][0].split(" %")[0].split(" (")[0].split(":")[0].split(" 0")[0].split(" 1")[0].split(" 2")[0]
            for r in sel if r["why"]).most_common(10)))
        for r in sorted(left, key=lambda r: -abs(r["z"]))[:25]:
            print("  HIP %d (Gaia DR3 %s) G %.2f BP-RP %.2f: %+.3f mag (%.1f sigma), Hp %.3f +- %.3f, %d neighbours, "
                  "RUWE %.2f, scatter ratios Hip %.1f Gaia %.1f, TESS %s"
                  % (r["hip"], r["gaia"], r["g"], r["bp_rp"], r["resid"], r["z"], r["hp"], r["e_hp"], r["neighbours"],
                     r["ruwe"], r.get("hpscat_ratio", float("nan")), r.get("gscat_ratio", float("nan")), r.get("tess")))


def hip_gaia_record():
    """Record the run from hip_gaia_dimming.json."""
    rows = json.load(open(os.path.join(hip_gaia_HERE, "hip_gaia_dimming.json")))
    const = [r for r in rows if r["var"] in ("", "C") and r["ruwe"] < 1.4 and r["neighbours"] == 0]
    dim = [r for r in rows if r["z"] >= 5]
    bri = [r for r in rows if r["z"] <= -5]
    spread = 1.4826 * float(np.median(np.abs(np.array([r["resid"] for r in const]) -
                                             np.median([r["resid"] for r in const]))))
    core.record_search(
        "hip_gaia_dimming", "Secular dimming between Hipparcos (1991.25) and Gaia DR3 (2016.0)",
        archive="Hipparcos main catalogue I/239 (Hp, Hpscat, HvarType, MultFlag); Gaia DR3 gaia_source and "
                "hipparcos2_best_neighbour; WDS B/wds; SIMBAD; TESS-SPOC",
        release="Hipparcos 1997; Gaia DR3",
        selection="Gaia DR3 Hipparcos cross-match, parallax above 10 mas, G above 6, with BP-RP",
        code="seti.py dimming hip_gaia",
        params=dict(sigma=5, colour_window=200, neighbours_arcsec=[10, 20, hip_gaia_EDGE], cstar=hip_gaia_CSTAR, scatter_ratio=3,
                    tess_power=0.5, tess_range=0.005),
        n_targets=len(rows),
        null_summary="%d stars fainter in Gaia by 5 robust deviations, %d without a natural cause; %d brighter, %d of "
                     "them without one" % (len(dim), sum(1 for r in dim if not r["why"]), len(bri),
                                           sum(1 for r in bri if not r["why"])),
        sensitivity="constant single stars G minus Hp spread %.4f mag after the colour term" % spread,
        result="no unexplained dimming" if not any(not r["why"] for r in dim) else
        "%d unexplained dimmings" % sum(1 for r in dim if not r["why"]),
        ledger="Dimming between Hipparcos and Gaia")
    core.clear_targets("hip_gaia_dimming")
    core.record_targets("hip_gaia_dimming", [
        dict(target="hip:%d" % r["hip"], stat=float(r["z"]), unit="robust deviations of G - Hp",
             vet="explained" if r.get("why") else None,
             note="%+.4f mag%s" % (r["resid"], ("; " + r["why"][0]) if r.get("why") else "")) for r in rows])
    print("recorded %d stars" % len(rows))


def hip_gaia_cli():
    quiet_warnings()
    if sys.argv[1:] == ["record"]:
        hip_gaia_record()
    elif sys.argv[1:] == ["rotation"]:
        rows = json.load(open(os.path.join(hip_gaia_HERE, "hip_gaia_dimming.json")))
        hip_gaia_rotation([r for r in rows if "why" in r and not r["why"]])
        json.dump(rows, open(os.path.join(hip_gaia_HERE, "hip_gaia_dimming.json"), "w"), indent=1, default=str)
        hip_gaia_summarise(rows)
    else:
        hip_gaia_main()


# --------------------------------------------------------------------------------------------------------------

dasch_DOC = r"""Century trends of nearby sun-like stars on DASCH plates: Theil-Sen slopes against comparisons, coherent across plate series, with injected-fading recovery.

    python -I seti.py dimming dasch trends --list dasch_targets.csv [--workers 6]
    python -I seti.py dimming dasch series SOURCE_ID [SOURCE_ID ...]
    python -I seti.py dimming dasch judge [--sigma 4]
    python -I seti.py dimming dasch inject [--stars 30] [--sizes 0.2,0.4,0.6,1.0]"""

dasch_HERE = core.RESULTS
dasch_WORK = os.path.join(dasch_HERE, "dasch_work")
dasch_OUT = os.path.join(dasch_HERE, "dasch_trends.jsonl")
dasch_TARGETS = os.path.join(dasch_HERE, "dasch_targets.csv")
# target lists a session may come from
dasch_TARGET_LISTS = (dasch_TARGETS, os.path.join(dasch_HERE, "dasch_hz_targets.csv"))
dasch_MIN_PLATES, dasch_MIN_COMPS, dasch_MAX_SHIFT = 300, 4, 60.0


def dasch__dev():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dasch_theil_sen(x, y):
    """Theil-Sen slope on the GPU."""
    import torch
    dev = dasch__dev()
    xt = torch.as_tensor(np.asarray(x, np.float64) - np.mean(x), dtype=torch.float32, device=dev)
    yt = torch.as_tensor(np.asarray(y, np.float64), dtype=torch.float32, device=dev)
    i, j = torch.triu_indices(len(xt), len(xt), 1, device=dev)
    dx = xt[j] - xt[i]
    keep = dx.abs() > 1e-6
    sl = (yt[j] - yt[i])[keep] / dx[keep]
    return float(sl.median()), 0.0


def dasch_slope_stats(t, m, nboot=200, seed=1):
    """Theil-Sen slope (mag per century) and bootstrap error on the GPU."""
    import torch
    dev = dasch__dev()
    yr = (np.asarray(t, np.float64) - np.mean(t)) / 100.0
    s, _ = dasch_theil_sen(yr, m)
    xt = torch.as_tensor(yr, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(np.asarray(m, np.float64), dtype=torch.float32, device=dev)
    n = len(xt)
    g = torch.Generator(device=dev).manual_seed(seed)
    i, j = torch.triu_indices(n, n, 1, device=dev)
    bs = []
    for _ in range(nboot):
        k = torch.randint(0, n, (n,), generator=g, device=dev)
        xb, yb = xt[k], yt[k]
        dx = xb[j] - xb[i]
        keep = dx.abs() > 1e-6
        bs.append(float(((yb[j] - yb[i])[keep] / dx[keep]).median()))
    return s, float(np.std(bs))


def dasch_clean(lc):
    """Times, calibrated magnitudes and series of plates passing the standard rejections."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        lc.apply_standard_rejections()
    mag = np.asarray(lc["magcal_magdep"].filled(np.nan) if hasattr(lc["magcal_magdep"], "filled")
                     else lc["magcal_magdep"], float)
    rej = np.asarray(lc["reject"], int)
    t = lc["time"].jyear
    ser = np.asarray(lc["series"]).astype(str)
    ok = np.isfinite(mag) & (rej == 0)
    return t[ok], mag[ok], ser[ok]


def dasch_comparison_indices(rc, sep, i0):
    """Up to ten reference stars within 1.0 mag and 0.5 colour of the target, nearest first."""
    mag0, col0 = float(rc["stdmag"][i0]), float(rc["color"][i0])
    cand = [i for i in range(len(rc)) if i != i0 and np.isfinite(rc["stdmag"][i]) and np.isfinite(rc["color"][i])
            and abs(float(rc["stdmag"][i]) - mag0) < 1.0 and abs(float(rc["color"][i]) - col0) < 0.5]
    return sorted(cand, key=lambda i: sep[i])[:10]


def dasch_session_curves(sid):
    """Target and comparison curves from the star's cached daschlab session."""
    quiet_warnings()
    import daschlab
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    known = {}
    for path in dasch_TARGET_LISTS:
        if os.path.exists(path):
            known.update({x["source_id"]: x for x in csv.DictReader(open(path))})
    r = known[sid]
    ra, dec = float(r["ra"]), float(r["dec"])
    daschlab.REFCAT_RADIUS = 1800 * u.arcsec
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sess = daschlab.open_session(os.path.join(dasch_WORK, sid + "_r1800"))
        sess.select_target(ra_deg=ra, dec_deg=dec)
        sess.select_refcat("apass")
        rc = sess.refcat()
    sep = rc["pos"].separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec
    i0 = int(np.argmin(sep))
    curves = {}
    for i in [i0] + dasch_comparison_indices(rc, sep, i0):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                curves[i] = dasch_clean(sess.lightcurve(i))
        except Exception:
            continue
    return curves.pop(i0), curves


# ------------------------------------------------------------------------------------------------------------ trends


def dasch_one(job):
    """Slopes of one target and its comparisons from a fresh daschlab session."""
    sid, ra, dec = job
    quiet_warnings()
    import daschlab
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    d = os.path.join(dasch_WORK, sid + "_r1800")
    daschlab.REFCAT_RADIUS = 1800 * u.arcsec
    try:
        for attempt in range(3):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    sess = daschlab.open_session(d)
                    sess.select_target(ra_deg=ra, dec_deg=dec)
                    sess.select_refcat("apass")
                    rc = sess.refcat()
                break
            except Exception:
                if attempt == 2:
                    raise
                import shutil
                import time as _t
                shutil.rmtree(d, ignore_errors=True)
                _t.sleep(10 * (attempt + 1))
        pos = rc["pos"]
        sep = pos.separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec
        i0 = int(np.argmin(sep))
        if sep[i0] > 20:
            return dict(sid=sid, error="no refcat match within 20 arcsec (%.1f)" % sep[i0])
        mag0, col0 = float(rc["stdmag"][i0]), float(rc["color"][i0])
        cand = dasch_comparison_indices(rc, sep, i0)
        res = {}
        for i in [i0] + cand:
            lc = None
            for attempt in range(3):
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        lc = sess.lightcurve(i)
                    break
                except Exception:
                    import time as _t
                    _t.sleep(5 * (attempt + 1))
            if lc is None:
                continue
            t, m, ser = dasch_clean(lc)
            if len(t) < 50 or np.ptp(t) < 50:
                continue
            s, e = dasch_slope_stats(t, m)
            per = {}
            for name in set(ser):
                k = ser == name
                if k.sum() >= 30 and np.ptp(t[k]) >= 20:
                    per[name] = round(dasch_theil_sen(t[k] / 100.0, m[k])[0], 3)
            res[i] = dict(n=int(len(t)), span=[round(float(t.min()), 1), round(float(t.max()), 1)],
                          slope=round(s, 4), err=round(e, 4), rms=round(float(np.std(m - np.median(m))), 3),
                          series=per)
        if i0 not in res:
            return dict(sid=sid, error="target light curve too short", mag=mag0)
        comps = [res[i]["slope"] for i in cand if i in res]
        if len(comps) < 3:
            return dict(sid=sid, target=res[i0], mag=mag0, color=col0, error="fewer than 3 comparison light curves")
        med = float(np.median(comps))
        spread = float(1.4826 * np.median(np.abs(np.array(comps) - med)))
        diff = res[i0]["slope"] - med
        sig = diff / max(np.hypot(spread, res[i0]["err"]), 1e-3)
        return dict(sid=sid, mag=mag0, color=col0, target=res[i0], comps=comps, comp_median=round(med, 4),
                    comp_spread=round(spread, 4), diff=round(diff, 4), sigma=round(float(sig), 2))
    except Exception as e:
        return dict(sid=sid, error=str(e)[:200])


def dasch_trends_main(argv):
    p = argparse.ArgumentParser(prog="seti.py dimming dasch trends")
    p.add_argument("--list", required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(argv)
    done = set()
    if os.path.exists(dasch_OUT):
        done = {json.loads(ln)["sid"] for ln in open(dasch_OUT) if ln.strip()}
    jobs = [(r["source_id"], float(r["ra"]), float(r["dec"])) for r in csv.DictReader(open(a.list))
            if r["source_id"] not in done]
    print("%d stars to measure" % len(jobs), flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(dasch_OUT, "a") as fh:
        for r in ex.map(dasch_one, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "sigma" in r:
                print("%s B %.2f slope %+.3f mag/century, comparisons %+.3f +- %.3f, difference %+.3f (%.1f sigma), "
                      "%d plates" % (r["sid"], r["mag"], r["target"]["slope"], r["comp_median"], r["comp_spread"],
                                     r["diff"], r["sigma"], r["target"]["n"]), flush=True)
            else:
                print("%s: %s" % (r["sid"], r.get("error")), flush=True)


# ------------------------------------------------------------------------------------------------------------ series


def dasch_series_rows(target, comps):
    """Per-series target slope, comparison median and spread, and difference (mag per century)."""
    t0, m0, s0 = target
    rows = []
    for name in sorted(set(s0)):
        k = s0 == name
        if k.sum() < 30 or np.ptp(t0[k]) < 20:
            continue
        st = dasch_theil_sen(t0[k] / 100.0, m0[k])[0]
        cs = []
        for t, m, s in comps:
            kk = s == name
            if kk.sum() >= 30 and np.ptp(t[kk]) >= 20:
                cs.append(dasch_theil_sen(t[kk] / 100.0, m[kk])[0])
        if cs:
            med = float(np.median(cs))
            rows.append(dict(series=name, n=int(k.sum()), span=(float(t0[k].min()), float(t0[k].max())), target=st,
                             comp=med, spread=float(1.4826 * np.median(np.abs(np.array(cs) - med))), ncomp=len(cs),
                             diff=st - med))
    return rows


def dasch_analyse(sid):
    """Per-series rows and decade medians of the same-plate difference."""
    target, curves = dasch_session_curves(sid)
    comps = list(curves.values())
    rows = dasch_series_rows(target, comps)
    t0, m0, s0 = target

    def key(t):
        return np.round(t, 5)
    comp_on = {}
    for t, m, _ in comps:
        for tt, mm in zip(t, m):
            comp_on.setdefault(key(tt), []).append(mm)
    diff = np.array([m - np.median(comp_on[key(t)]) if len(comp_on.get(key(t), [])) >= 2 else np.nan
                     for t, m in zip(t0, m0)])
    decades = {}
    for name in sorted(set(s0)):
        k = (s0 == name) & np.isfinite(diff)
        if k.sum() >= 30:
            decades[name] = [(d, float(np.median(diff[k & (t0 >= d) & (t0 < d + 10)])),
                              int((k & (t0 >= d) & (t0 < d + 10)).sum()))
                             for d in range(1880, 2000, 10) if (k & (t0 >= d) & (t0 < d + 10)).sum() >= 8]
    return rows, decades


def dasch_coherent(rows, sign):
    """Coherence of differential slopes across series: common slope at 3 sigma with p above 0.01 over at least two series."""
    from scipy.stats import chi2
    big = [r for r in rows if r["n"] >= 100 and r["ncomp"] >= 3]
    if len(big) < 2:
        return False, float("nan"), float("nan"), float("nan"), len(big)
    d = np.array([r["diff"] for r in big])
    w = 1.0 / np.maximum(np.array([r["spread"] for r in big]), 0.05) ** 2
    D = float(np.sum(d * w) / np.sum(w))
    sD = float(1.0 / np.sqrt(np.sum(w)))
    p = float(chi2.sf(np.sum((d - D) ** 2 * w), len(big) - 1))
    return bool(p > 0.01 and D * sign / sD >= 3.0), D, sD, p, len(big)


def dasch_series_main(argv):
    for sid in argv:
        rows, decades = dasch_analyse(sid)
        print("Gaia DR3 %s" % sid)
        for r in rows:
            print("  series %-4s %4d plates %.0f-%.0f: target %+.3f, comparisons %+.3f (%d stars, spread %.3f), "
                  "difference %+.3f" % (r["series"], r["n"], r["span"][0], r["span"][1], r["target"], r["comp"],
                                        r["ncomp"], r["spread"], r["diff"]))
        print("  target minus comparisons on the same plates, median by decade (plates):")
        for name, cells in decades.items():
            print("    %-4s %s" % (name, "  ".join("%ds %+.3f (%d)" % c for c in cells)))


# ------------------------------------------------------------------------------------------------------------- judge


def dasch_proper_motions(ids):
    """Total proper motion (arcsec/yr) of Gaia DR3 sources, cached."""
    path = os.path.join(dasch_HERE, "dasch_pm.json")
    pm = json.load(open(path)) if os.path.exists(path) else {}
    todo = [i for i in ids if i not in pm]
    if todo:
        for k in range(0, len(todo), 400):
            rows = core.gaia("SELECT source_id, pmra, pmdec FROM gaiadr3.gaia_source WHERE source_id IN (%s)"
                                 % ",".join(todo[k:k + 400]))
            for r in rows:
                pm[r["source_id"]] = float(np.hypot(float(r["pmra"] or 0), float(r["pmdec"] or 0))) / 1000.0
        json.dump(pm, open(path, "w"))
    return pm


def dasch_moving_blend(sid):
    """Neighbours within 120 arcsec and 2.5 mag whose separation changes over 5 arcsec in 1890-1990, mimicking trends on patrol plates."""
    import math
    g = core.gaia("SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
                      "source_id = %s" % sid)
    if not g:
        return []
    t = g[0]
    ra, dec, gm = float(t["ra"]), float(t["dec"]), float(t["phot_g_mean_mag"])
    c = math.cos(math.radians(dec))
    near = core.gaia("SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
                         "CONTAINS(POINT(ra, dec), CIRCLE(%.7f, %.7f, %.5f)) = 1 AND phot_g_mean_mag < %.2f AND "
                         "source_id != %s" % (ra, dec, 120.0 / 3600, gm + 2.5, sid))

    def at(x, yr):
        return ((float(x["ra"]) * c * 3600 + float(x["pmra"] or 0) / 1000 * (yr - 2016)),
                float(x["dec"]) * 3600 + float(x["pmdec"] or 0) / 1000 * (yr - 2016))
    out = []
    for n in near:
        s = [math.hypot(at(t, y)[0] - at(n, y)[0], at(t, y)[1] - at(n, y)[1]) for y in (1890, 1990)]
        if abs(s[1] - s[0]) > 5.0:
            out.append("G %.1f neighbour %s from %.0f to %.0f arcsec" % (float(n["phot_g_mean_mag"]),
                                                                          "closing" if s[1] < s[0] else "opening", s[0], s[1]))
    return out


def dasch_judge_main(argv):
    p = argparse.ArgumentParser(prog="seti.py dimming dasch judge")
    p.add_argument("--sigma", type=float, default=4.0)
    a = p.parse_args(argv)
    rows = [json.loads(ln) for ln in open(dasch_OUT) if ln.strip()]
    total = len(list(csv.DictReader(open(dasch_TARGETS))))
    meas = [r for r in rows if "sigma" in r]
    pm = dasch_proper_motions([r["sid"] for r in meas])
    d = np.array([r["diff"] for r in meas])
    pop = float(1.4826 * np.median(np.abs(d - np.median(d))))
    judged, flagged = [], []
    for r in meas:
        field = float(np.hypot(r["comp_spread"], r["target"]["err"]))
        r["sigma_pop"] = r["diff"] / max(field, pop)
        why = []
        if r["target"]["n"] < dasch_MIN_PLATES:
            why.append("%d plates" % r["target"]["n"])
        if len(r["comps"]) < dasch_MIN_COMPS:
            why.append("%d comparisons" % len(r["comps"]))
        if pm.get(r["sid"], 0.0) * 120.0 > dasch_MAX_SHIFT:
            why.append("moves %.0f arcsec in 120 yr" % (pm[r["sid"]] * 120.0))
        (flagged if why else judged).append((r, why))
    print("%d of %d targets processed, %d measured; sample spread of differential slopes %.3f mag/century; %d judged, "
          "%d not judged" % (len(rows), total, len(meas), pop, len(judged), len(flagged)))
    hits = [r for r, _ in judged if abs(r["sigma_pop"]) >= a.sigma]
    print("judged trends at %.1f sigma or more: %d" % (a.sigma, len(hits)))
    for r in sorted(hits, key=lambda r: -abs(r["sigma_pop"])):
        rows_s, _ = dasch_analyse(r["sid"])
        ok, D, sD, p, ns = dasch_coherent(rows_s, np.sign(r["diff"]))
        blend = dasch_moving_blend(r["sid"])
        tag = "" if ok and not blend else ("blend changing with proper motion (%s): " % "; ".join(blend)) if blend \
            else "series incoherent: "
        print("  %sGaia DR3 %s B %.2f differential %+.3f mag/century (%+.1f sigma), %d plates; common series slope "
              "%+.3f +- %.3f over %d series, heterogeneity p %.3g; by series %s"
              % (tag, r["sid"], r["mag"], r["diff"], r["sigma_pop"], r["target"]["n"],
                 D, sD, ns, p, ", ".join("%s %+.2f (spread %.2f, %d plates)" % (x["series"], x["diff"], x["spread"],
                                                                                x["n"]) for x in rows_s)))
    for r, why in sorted(flagged, key=lambda x: -abs(x[0]["sigma_pop"]))[:5]:
        print("  not judged: Gaia DR3 %s differential %+.3f (%s)" % (r["sid"], r["diff"], ", ".join(why)))


# ------------------------------------------------------------------------------------------------------------ inject


def dasch_judge_injected(target, comps, pop):
    """Trend judgement of one target curve: slope, significance and recovery; None with fewer than four comparisons."""
    t, m, s = target
    sl, err = dasch_slope_stats(t, m, nboot=100)
    cs = [dasch_slope_stats(c[0], c[1], nboot=1)[0] for c in comps if len(c[0]) >= 50 and np.ptp(c[0]) >= 50]
    if len(cs) < 4:
        return None
    med = float(np.median(cs))
    spread = float(1.4826 * np.median(np.abs(np.array(cs) - med)))
    diff = sl - med
    sig = diff / max(np.hypot(spread, err), pop)
    ok = abs(sig) >= 4.0
    if ok:
        rows = [dict(series=r["series"], n=r["n"], ncomp=r["ncomp"], diff=r["diff"], spread=r["spread"])
                for r in dasch_series_rows(target, comps)]
        ok = dasch_coherent(rows, np.sign(diff))[0]
    return dict(diff=diff, sigma=sig, recovered=bool(ok))


def dasch_inject_main(argv):
    quiet_warnings()
    p = argparse.ArgumentParser(prog="seti.py dimming dasch inject")
    p.add_argument("--stars", type=int, default=30)
    p.add_argument("--sizes", default="0.2,0.4,0.6,1.0")
    a = p.parse_args(argv)
    rows = [json.loads(ln) for ln in open(dasch_OUT) if ln.strip()]
    pm = json.load(open(os.path.join(dasch_HERE, "dasch_pm.json")))
    meas = [r for r in rows if "sigma" in r]
    d = np.array([r["diff"] for r in meas])
    pop = float(1.4826 * np.median(np.abs(d - np.median(d))))
    judged = [r for r in meas if r["target"]["n"] >= 300 and len(r["comps"]) >= 4 and pm.get(r["sid"], 0) * 120 <= 60
              and abs(r["diff"]) < 2 * pop]
    rng = np.random.default_rng(5)
    pick = [judged[i] for i in rng.choice(len(judged), size=min(a.stars, len(judged)), replace=False)]
    sizes = [float(x) for x in a.sizes.split(",")]
    res = {s: [] for s in sizes}
    for r in pick:
        try:
            target, curves = dasch_session_curves(r["sid"])
        except Exception:
            continue
        comps = list(curves.values())
        t, m, s = target
        tm = 0.5 * (t.min() + t.max())
        for size in sizes:
            j = dasch_judge_injected((t, m + size * (t - tm) / 100.0, s), comps, pop)
            if j:
                res[size].append(j["recovered"])
        print("%s: %s" % (r["sid"], ", ".join("%.1f %s" % (sz, res[sz][-1] if res[sz] else "-") for sz in sizes)),
              flush=True)
    print("sample spread %.3f mag per century; recovered fraction by injected fading:" % pop)
    for size in sizes:
        v = res[size]
        print("  %.1f mag per century: %d of %d (%.0f per cent)" % (size, sum(v), len(v), 100.0 * sum(v) / max(len(v), 1)))


dasch_MODES = {"trends": dasch_trends_main, "series": dasch_series_main, "judge": dasch_judge_main, "inject": dasch_inject_main}


def dasch_cli():
    if len(sys.argv) < 2 or sys.argv[1] not in dasch_MODES:
        print(__doc__)
        sys.exit(2)
    dasch_MODES[sys.argv[1]](sys.argv[2:])


# --------------------------------------------------------------------------------------------------------------

tess_DOC = r"""Slow TESS dimming the pipeline removes: six-hour bins of cotrended SAP flux, runs 0.1 per cent deep over a day, vetted against background, centroid and rotation.

    python -I seti.py dimming tess "HD 193664"
    python -I seti.py dimming tess background "ra dec" SECTOR START-END [START-END ...]
    python -I seti.py dimming tess vet "HIP 45963" [...]   (names looked up in slow_targets_coords.txt)"""


def tess_bin6(t, f, hours=6.0):
    w = hours / 24.0
    edges = np.arange(t.min(), t.max() + w, w)
    idx = np.digitize(t, edges)
    bt, bf = [], []
    for i in np.unique(idx):
        m = idx == i
        if m.sum() > 30:
            bt.append(t[m].mean())
            bf.append(np.median(f[m]))
    return np.array(bt), np.array(bf)


def tess_edge_mask(bt, after=1.5, before=0.5, gap=0.5):
    """Bins near segment starts and ends where thermal settling and cotrending bend the flux."""
    bt = np.asarray(bt)
    starts = np.r_[bt[0], bt[1:][np.diff(bt) > gap]]
    ends = np.r_[bt[:-1][np.diff(bt) > gap], bt[-1]]
    m = np.zeros(len(bt), bool)
    for s in starts:
        m |= (bt >= s) & (bt < s + after)
    for e in ends:
        m |= (bt > e - before) & (bt <= e)
    return m


def tess_bright_sky(lc, bt, limit=1.3):
    """Six-hour bins with background above limit times the sector median."""
    q = lc.remove_nans()
    t, b = q.time.value, np.asarray(q.sap_bkg.value, float)
    b = b / np.nanmedian(b)
    w = 0.125
    return np.array([np.nanmedian(b[(t >= x - w) & (t < x + w)]) > limit if np.any((t >= x - w) & (t < x + w))
                     else False for x in bt])


def tess_runs(bt, bf, thr=1e-3, min_days=1.0, exclude=None):
    low = (bf < 1.0 - thr) & ~tess_edge_mask(bt)
    if exclude is not None:
        low &= ~exclude
    out, i = [], 0
    while i < len(low):
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(low) and low[j + 1]:
            j += 1
        if bt[j] - bt[i] >= min_days - 0.25:
            out.append((bt[i], bt[j], float(1.0 - bf[i:j + 1].min())))
        i = j + 1
    return out


def tess_cotrended(lc):
    """Six-hour bins of SAP flux corrected by single-scale cotrending vectors."""
    from lightkurve.correctors import CBVCorrector
    raw = lc[np.isfinite(np.asarray(lc.flux.value, float))].copy()
    raw.flux, raw.flux_err = raw.sap_flux, raw.sap_flux_err
    raw = raw.remove_nans()
    cc = CBVCorrector(raw, interpolate_cbvs=True)
    cor = cc.correct_gaussian_prior(cbv_type=["SingleScale"], cbv_indices=[np.arange(1, 5)], alpha=1e-4)
    return tess_bin6(cor.time.value, cor.flux.value / np.nanmedian(cor.flux.value))


def tess_neighbours_dip(lc, s, e, n=5, radius_deg=1.5):
    """Median depth over [s, e] among other two-minute targets on the same CCD."""
    import lightkurve as lk
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sector, cam, ccd = lc.meta.get("SECTOR"), lc.meta.get("CAMERA"), lc.meta.get("CCD")
    c = SkyCoord(lc.meta["RA_OBJ"] * u.deg, lc.meta["DEC_OBJ"] * u.deg)
    sr = lk.search_lightcurve(c, radius=radius_deg * 3600, mission="TESS", author="SPOC", exptime=120,
                              sector=sector)
    depths = []
    for row in sr[1:n * 3]:
        if len(depths) >= n:
            break
        try:
            o = row.download(download_dir=core.data("lc"))
            if o.meta.get("CAMERA") != cam or o.meta.get("CCD") != ccd or o.meta.get("TICID") == lc.meta.get("TICID"):
                continue
            bt, bf = tess_cotrended(o)
            m = (bt >= s) & (bt <= e)
            if m.sum():
                depths.append(float(1.0 - bf[m].min()))
        except Exception:
            continue
    return depths


def tess_main():
    p = argparse.ArgumentParser()
    p.add_argument("star")
    a = p.parse_args()
    import lightkurve as lk
    target = a.star
    parts = a.star.split()
    if len(parts) == 2 and all(p.replace(".", "").replace("-", "").isdigit() for p in parts):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        target = SkyCoord(float(parts[0]) * u.deg, float(parts[1]) * u.deg)
    sr = lk.search_lightcurve(target, radius=10, mission="TESS", author="SPOC", exptime=120)
    lcs = sr.download_all(download_dir=core.data("lc")) if len(sr) else []
    print("%s: %d sectors" % (a.star, len(lcs)), flush=True)
    for lc in lcs:
        sector = lc.meta.get("SECTOR")
        try:
            bt, bf = tess_cotrended(lc)
        except Exception as e:
            print("  sector %s: no cotrending vectors (%s)" % (sector, str(e)[:60]), flush=True)
            continue
        pd = lc.remove_nans().normalize()
        pt, pf = tess_bin6(pd.time.value, pd.flux.value.astype(float))
        r_raw, r_pdc = tess_runs(bt, bf), tess_runs(pt, pf)
        print("  sector %s: raw-cotrended spread %.0f ppm, slow dims %d (pipeline %d)"
              % (sector, np.std(bf) * 1e6, len(r_raw), len(r_pdc)), flush=True)
        for s, e, d in r_raw:
            seen = any(not (e2 < s or s2 > e) for s2, e2, _ in r_pdc)
            nd = tess_neighbours_dip(lc, s, e)
            shared = len(nd) > 0 and np.median(nd) > 0.5 * d
            verdict = ("shared by %d neighbours (systematic)" % len(nd) if shared else
                       "UNIQUE to the star among %d neighbours" % len(nd))
            print("    BTJD %.2f-%.2f, %.1f days, depth %.0f ppm, %s, %s"
                  % (s, e, e - s + 0.25, d * 1e6, "also in pipeline" if seen else "removed by pipeline", verdict),
                  flush=True)


def tess_background_main():
    """Six-hour bins around each window beside background and centroid."""
    import lightkurve as lk
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra, dec = map(float, sys.argv[1].split())
    sector = int(sys.argv[2])
    sr = lk.search_lightcurve(SkyCoord(ra * u.deg, dec * u.deg), radius=10, mission="TESS", author="SPOC",
                              exptime=120, sector=sector)
    lc = sr.download(download_dir=core.data("lc"))
    bt, bf = tess_cotrended(lc)
    q = lc.remove_nans()
    t = q.time.value
    bkg = np.asarray(q.sap_bkg.value, float)
    cx = np.asarray(q.centroid_col.value, float)
    cy = np.asarray(q.centroid_row.value, float)
    _, bb = tess_bin6(t, bkg / np.nanmedian(bkg))
    _, bx = tess_bin6(t, (cx - np.nanmedian(cx)) * 1e3)
    _, by = tess_bin6(t, (cy - np.nanmedian(cy)) * 1e3)
    n = min(len(bt), len(bb))
    r = np.corrcoef(bf[:n], bb[:n])[0, 1]
    print("sector %d: flux-background correlation over the sector %.2f" % (sector, r))
    for w in sys.argv[3:]:
        s, e = map(float, w.split("-"))
        print(" window %.2f-%.2f" % (s, e))
        for i in range(n):
            if s - 1.0 <= bt[i] <= e + 1.0:
                print("  %.2f  flux %+6.0f ppm  bkg x%.2f  centroid %+5.0f,%+5.0f mpix%s"
                      % (bt[i], (bf[i] - 1) * 1e6, bb[i], bx[i], by[i], "  <" if s <= bt[i] <= e else ""))


def tess_simbad_type(name):
    q = ("SELECT basic.otype, basic.sp_type FROM basic JOIN ident ON ident.oidref=basic.oid "
         "WHERE ident.id='%s'" % name)
    data = urllib.parse.urlencode(dict(request="doQuery", lang="adql", format="json", query=q)).encode()
    try:
        d = json.loads(urllib.request.urlopen("https://simbad.cds.unistra.fr/simbad/sim-tap/sync", data=data,
                                              timeout=60).read())
        return d["data"][0] if d["data"] else ["?", "?"]
    except Exception:
        return ["?", "?"]


def tess_detrend_rotation(bt, bf):
    from astropy.timeseries import LombScargle
    ls = LombScargle(bt, bf, nterms=2)
    freq, power = ls.autopower(minimum_frequency=1 / 13.0, maximum_frequency=2.0, samples_per_peak=20)
    k = int(np.argmax(power))
    model = ls.model(bt, freq[k])
    return 1.0 / freq[k], float(power[k]), float(np.ptp(model)), bf - model + 1.0


def tess_vet_main():
    """Dims left after removing rotational modulation."""
    coords = dict(line.strip().split("|") for line in open(core.results("slow_targets_coords.txt")) if "|" in line)
    import lightkurve as lk
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    for name in sys.argv[1:]:
        ra, dec = map(float, coords[name].split())
        otype, sp = tess_simbad_type(name)
        sr = lk.search_lightcurve(SkyCoord(ra * u.deg, dec * u.deg), radius=10, mission="TESS", author="SPOC",
                                  exptime=120)
        lcs = sr.download_all(download_dir=core.data("lc")) if len(sr) else []
        print("=== %s  SIMBAD %s  %s  %d sectors" % (name, otype, sp, len(lcs)), flush=True)
        for lc in lcs:
            try:
                bt, bf = tess_cotrended(lc)
            except Exception as e:
                print("  sector %s: %s" % (lc.meta.get("SECTOR"), str(e)[:60]), flush=True)
                continue
            period, power, amp, resid = tess_detrend_rotation(bt, bf)
            sig = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            left = [r for r in tess_runs(bt, resid, exclude=tess_bright_sky(lc, bt)) if r[2] > 3.0 * sig]
            print("  sector %s: period %.2f d, power %.2f, amplitude %.0f ppm, residual spread %.0f ppm, "
                  "dims left %d" % (lc.meta.get("SECTOR"), period, power, amp * 1e6, np.std(resid) * 1e6,
                                    len(left)), flush=True)
            for s, e, d in left:
                print("    RESIDUAL DIM BTJD %.2f-%.2f, %.1f days, depth %.0f ppm" % (s, e, e - s + 0.25, d * 1e6),
                      flush=True)


def tess_cli():
    quiet_warnings()
    modes = {"background": tess_background_main, "vet": tess_vet_main}
    if len(sys.argv) > 1 and sys.argv[1] in modes:
        modes[sys.argv.pop(1)]()
    else:
        tess_main()


# --------------------------------------------------------------------------------------------------------------

neowise_DOC = r"""Decade NEOWISE brightening of stars within 15 pc against same-frame references, W2 over W1 at 5 sigma; joint test with Hipparcos-to-Gaia dimming by Fisher's method.

    python -I seti.py dimming neowise fetch [--threads 3] [--sample hip_gaia]
    python -I seti.py dimming neowise analyse [--sample hip_gaia]
    python -I seti.py dimming neowise joint"""

neowise_ROOT = os.environ.get("NEOWISE_ROOT", core.data("neowise"))
neowise_HERE = core.RESULTS
neowise_RADIUS = 60.0
neowise_MATCH = 2.0
neowise_COLS = "ra, dec, mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qual_frame, cc_flags, ph_qual, nb, saa_sep, moon_masked"


def neowise_root(sample):
    return neowise_ROOT if sample == "nearest" else core.data("neowise_" + sample)


def neowise_result(sample):
    return os.path.join(neowise_HERE, "neowise_trends.json" if sample == "nearest" else "neowise_trends_%s.json" % sample)


def neowise_targets(sample="nearest"):
    if sample == "hip_gaia":
        ids = [r["gaia"] for r in json.load(open(core.results("hip_gaia_dimming.json")))]
        rows = []
        for i in range(0, len(ids), 2000):
            rows += core.gaia("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag FROM "
                                  "gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(ids[i:i + 2000]))
    else:
        rows = core.gaia("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag FROM gaiadr3.gaia_source "
                             "WHERE parallax > 66.67 AND parallax_over_error > 10")
    probe = [dict(source_id=r["source_id"], ra=float(r["ra"]) - float(r["pmra"] or 0) / 3.6e6 * 5.5 /
                  math.cos(math.radians(float(r["dec"]))), dec=float(r["dec"]) - float(r["pmdec"] or 0) / 3.6e6 * 5.5)
             for r in rows]
    w = {}
    for m in core.xmatch(probe, "II/328/allwise", 3.0):
        if m["source_id"] not in w or float(m["angDist"]) < float(w[m["source_id"]]["angDist"]):
            w[m["source_id"]] = m
    out = []
    for r in rows:
        m = w.get(r["source_id"])
        try:
            w1 = float(m["W1mag"]) if m else None
        except ValueError:
            w1 = None
        if w1 is not None and 8.0 <= w1 <= 12.5:
            out.append(dict(r, w1=w1))
    return out


def neowise_fetch_one(s, root=None):
    p = os.path.join(root or neowise_ROOT, "cones", "%s.json" % s["source_id"])
    if os.path.exists(p):
        return 1
    q = ("SELECT %s FROM neowiser_p1bs_psd WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f)) = 1"
         % (neowise_COLS, float(s["ra"]), float(s["dec"]), (neowise_RADIUS + 15 * math.hypot(float(s["pmra"] or 0), float(s["pmdec"] or 0)) / 1000) / 3600))
    for attempt in range(3):
        try:
            rows = core.irsa(q)
            break
        except Exception:
            rows = None
    if rows is None:
        return 0
    json.dump(rows, open(p, "w"))
    return 1


def neowise_fetch(threads, sample="nearest"):
    root = neowise_root(sample)
    os.makedirs(os.path.join(root, "cones"), exist_ok=True)
    tp = os.path.join(root, "targets.json")
    ts = json.load(open(tp)) if os.path.exists(tp) else neowise_targets(sample)
    json.dump(ts, open(tp, "w"))
    print("%d Gaia DR3 stars (%s) with AllWISE W1 8-12.5" % (len(ts), sample), flush=True)
    done = 0
    with ThreadPoolExecutor(threads) as ex:
        for i, ok in enumerate(ex.map(lambda s: neowise_fetch_one(s, root), ts)):
            done += ok
            if (i + 1) % 200 == 0:
                print("%d of %d cones fetched" % (done, len(ts)), flush=True)
    print("%d cones on disk" % done)


def neowise_analyse_one(s, root=None):
    p = os.path.join(root or neowise_ROOT, "cones", "%s.json" % s["source_id"])
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    if not rows:
        return None
    ra = np.array([float(r["ra"]) for r in rows])
    de = np.array([float(r["dec"]) for r in rows])
    mjd = np.array([float(r["mjd"]) for r in rows])
    w1 = np.array([num(r["w1mpro"]) or np.nan for r in rows])
    w2 = np.array([num(r["w2mpro"]) or np.nan for r in rows])
    ok = np.array([r["qual_frame"] not in ("0",) and r["cc_flags"][:2] == "00" and (num(r["saa_sep"]) or 0) > 5
                   and r["moon_masked"][:2] == "00" for r in rows])
    yrs = (mjd - 57388.0) / 365.25  # years from 2016.0
    pra = float(s["ra"]) + float(s["pmra"] or 0) / 3.6e6 * yrs / math.cos(math.radians(float(s["dec"])))
    pde = float(s["dec"]) + float(s["pmdec"] or 0) / 3.6e6 * yrs
    sep_t = np.hypot((ra - pra) * math.cos(math.radians(float(s["dec"]))), de - pde) * 3600
    tgt = ok & (sep_t < neowise_MATCH)
    if tgt.sum() < 40:
        return dict(sid=s["source_id"], error="%d target detections" % int(tgt.sum()))
    visit = np.floor((mjd - 56600.0) / 182.6).astype(int)
    # references: persistent field sources brighter than W1 14.5 on a 2 arcsec grid
    ref = ok & (sep_t > 8) & (w1 < 14.5)
    key = np.round(ra[ref] * 3600 * math.cos(math.radians(float(s["dec"]))) / 2).astype(np.int64) * 100000 + \
        np.round(de[ref] * 3600 / 2).astype(np.int64)
    out = dict(sid=s["source_id"], n=int(tgt.sum()), w1=float(np.nanmedian(w1[tgt])), w2=float(np.nanmedian(w2[tgt])))
    for band, mag in (("W1", w1), ("W2", w2)):
        vis = sorted(set(visit[tgt]))
        tv, dv = [], []
        rv_idx = np.where(ref)[0]
        # reference offsets from their medians, per visit
        offs = {}
        for k in np.unique(key):
            sel = rv_idx[key == k]
            if len(sel) < 30:
                continue
            m0 = np.nanmedian(mag[sel])
            for v in set(visit[sel]):
                vv = sel[visit[sel] == v]
                offs.setdefault(v, []).append(np.nanmedian(mag[vv]) - m0)
        for v in vis:
            sel = tgt & (visit == v)
            if sel.sum() < 4 or v not in offs or len(offs[v]) < 1:
                continue
            tv.append(np.median(yrs[sel]))
            dv.append(np.nanmedian(mag[sel]) - np.median(offs[v]))
        if len(tv) < 8:
            out[band] = None
            continue
        tv, dv = np.array(tv), np.array(dv)
        A = np.vstack([np.ones_like(tv), tv]).T
        coef, *_ = np.linalg.lstsq(A, dv, rcond=None)
        resid = dv - A @ coef
        sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        se = max(sd, 1e-3) / math.sqrt(max(np.sum((tv - tv.mean()) ** 2), 1e-9))
        out[band] = dict(slope_decade=float(coef[1] * 10), err_decade=float(se * 10), visits=len(tv),
                         scatter=float(sd), refs=int(np.median([len(offs[v]) for v in offs])))
    return out


def neowise_analyse(sample="nearest"):
    root = neowise_root(sample)
    ts = json.load(open(os.path.join(root, "targets.json")))
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    with ProcessPoolExecutor(int(os.environ.get("NEOWISE_PROCS", "2"))) as ex:
        res = [r for r in ex.map(partial(neowise_analyse_one, root=root), ts, chunksize=4) if r]
    good = [r for r in res if r.get("W1") and r.get("W2") and np.isfinite(r["W1"]["slope_decade"])
            and np.isfinite(r["W2"]["slope_decade"])]
    print("%d of %d stars with differential W1 and W2 light curves (median %d visits)" % (
        len(good), len(ts), np.median([r["W1"]["visits"] for r in good])))
    # slopes judged against stars within 0.5 mag in W1 to remove NEOWISE's brightness-dependent drift
    w1s = np.array([r["w1"] for r in good])
    for band in ("W1", "W2"):
        s = np.array([r[band]["slope_decade"] for r in good])
        for i, r in enumerate(good):
            near = np.abs(w1s - r["w1"]) < 0.5
            near[i] = False
            if near.sum() < 5:
                near = np.argsort(np.abs(w1s - r["w1"]))[1:8]
            med = float(np.median(s[near]))
            sc = float(1.4826 * np.median(np.abs(s[near] - med)))
            r[band]["excess"] = r[band]["slope_decade"] - med
            r[band]["z"] = r[band]["excess"] / math.hypot(r[band]["err_decade"], sc)
    for band in ("W1", "W2"):
        z = np.array([r[band]["z"] for r in good])
        print("%s slope against stars of the same brightness, in units of error and scatter: median %+.2f, robust "
              "spread %.2f" % (band, np.median(z), 1.4826 * np.median(np.abs(z - np.median(z)))))
    bright = [r for r in good if r["W2"]["z"] < -5 and r["W2"]["excess"] < r["W1"]["excess"]]
    print("brightening in W2 at 5 sigma beyond stars of the same brightness, more than in W1: %d" % len(bright))
    for r in sorted(good, key=lambda r: r["W2"]["z"])[:15]:
        print("  Gaia DR3 %s W1 %.2f W2 %.2f: W1 %+.3f (%+.1f sigma), W2 %+.3f (%+.1f sigma) mag/decade beyond stars of "
              "the same brightness, over %d visits (%d refs)"
              % (r["sid"], r["w1"], r["w2"], r["W1"]["excess"], r["W1"]["z"], r["W2"]["excess"], r["W2"]["z"],
                 r["W2"]["visits"], r["W2"]["refs"]))
    json.dump(res, open(neowise_result(sample), "w"), indent=1)


def neowise_joint(draws=2000, seed=7):
    """Fisher combination of optical-dimming and W2-brightening p-values, null from re-pairing within 0.5 mag in W1."""
    from scipy.stats import norm
    hg = {r["gaia"]: r for r in json.load(open(core.results("hip_gaia_dimming.json")))}
    nw = [r for r in json.load(open(neowise_result("hip_gaia")))
          if r.get("W2") and r.get("W1") and np.isfinite(r["W2"].get("z", np.nan)) and r["sid"] in hg]
    z_opt = np.array([hg[r["sid"]]["z"] for r in nw])
    z_ir = np.array([-r["W2"]["z"] for r in nw])  # brightening positive
    w1 = np.array([r["w1"] for r in nw])
    ok = np.isfinite(z_opt) & np.isfinite(z_ir)
    nw, z_opt, z_ir, w1 = [x for x, k in zip(nw, ok) if k], z_opt[ok], z_ir[ok], w1[ok]
    # each run's z scaled by its own spread
    so = 1.4826 * np.median(np.abs(z_opt - np.median(z_opt)))
    si = 1.4826 * np.median(np.abs(z_ir - np.median(z_ir)))
    zo, zi = (z_opt - np.median(z_opt)) / so, (z_ir - np.median(z_ir)) / si

    def fisher(a, b):
        return -2.0 * (norm.logsf(a) + norm.logsf(b))
    x = fisher(zo, zi)
    rng = np.random.default_rng(seed)
    order = np.argsort(w1)
    null = []
    for _ in range(draws):
        perm = order.copy()
        # shuffle within brightness blocks of at least 20 stars
        edges = [0]
        for j in range(1, len(order)):
            if w1[order[j]] - w1[order[edges[-1]]] > 0.5 and j - edges[-1] >= 20:
                edges.append(j)
        edges.append(len(order))
        for s0, s1 in zip(edges[:-1], edges[1:]):
            blk = perm[s0:s1].copy()
            rng.shuffle(blk)
            perm[s0:s1] = blk
        zi_p = np.empty_like(zi)
        zi_p[order] = zi[perm]
        null.append(np.sort(fisher(zo, zi_p))[::-1][:20])
    null = np.array(null)
    top = np.argsort(-x)[:20]
    print("%d stars with a Hipparcos-to-Gaia deviation and a W2 trend; robust spreads %.2f (optical) and %.2f (W2)"
          % (len(x), so, si))
    out = []
    for rank, i in enumerate(top):
        p_rank = float(np.mean(null[:, rank] >= x[i]))
        p_any = float(np.mean(null[:, 0] >= x[i]))
        r = nw[i]
        out.append(dict(sid=r["sid"], hip=hg[r["sid"]]["hip"], fisher=float(x[i]), z_opt=float(z_opt[i]),
                        z_w2=float(-z_ir[i]), w1=float(w1[i]), p_rank=p_rank, p_any=p_any,
                        opt_mag=float(hg[r["sid"]]["resid"]), w2_slope=r["W2"]["excess"]))
        print("  HIP %-7d Gaia DR3 %s: Fisher %.1f, optical %+.1f sigma (%+.3f mag fainter), W2 %+.1f sigma "
              "(%+.3f mag/decade); chance of a value this high at rank %d %.3f, anywhere %.3f"
              % (hg[r["sid"]]["hip"], r["sid"], x[i], z_opt[i], hg[r["sid"]]["resid"], -z_ir[i], r["W2"]["excess"],
                 rank + 1, p_rank, p_any))
    json.dump(dict(n=len(x), top=out, draws=draws), open(core.results("dimming_ir_joint.json"), "w"), indent=1)


def neowise_joint_record():
    """Record the joint statistic: the top twenty stars with Fisher value and chance rate."""
    d = json.load(open(core.results("dimming_ir_joint.json")))
    top = d["top"]
    best = min(top, key=lambda r: r["p_rank"])
    core.record_search(
        "dimming_ir_joint", "Dimming with mid-infrared brightening", archive="Hipparcos and Gaia DR3 photometry; NEOWISE-R single exposures (IRSA)",
        release="the Hipparcos-to-Gaia dimming run and the NEOWISE W2 trends of the same stars", selection="stars carrying both a Hipparcos-to-Gaia deviation and a W2 trend",
        code=os.path.abspath(__file__), entry="neowise joint", n_targets=d["n"], params=dict(draws=d.get("draws", 2000), block_mag=0.5, block_min=20),
        null_summary="the W2 trends re-paired among stars within 0.5 mag in W1, which keeps both marginal distributions",
        sensitivity="one-sided optical and W2 deviations combined by Fisher's method; each robustly scaled to its own run",
        ledger="Dimming with mid-infrared brightening",
        result="%d stars; the highest Fisher value %.0f (HIP %d, %.1f sigma fainter in Gaia, W2 %s at %.1f sigma) arises by chance at its rank %.0f per cent of the time; the lowest chance at any rank %.2f (HIP %d), one in twenty ranks" % (
            d["n"], top[0]["fisher"], top[0]["hip"], top[0]["z_opt"], "fading" if top[0]["z_w2"] > 0 else "brightening", abs(top[0]["z_w2"]),
            100 * top[0]["p_rank"], best["p_rank"], best["hip"]))
    core.clear_targets("dimming_ir_joint")
    core.record_targets("dimming_ir_joint", [dict(target="gaia_dr3:%s" % r["sid"], stat=r["fisher"], p=r["p_rank"], unit="Fisher's combined statistic",
                                                      vet="within the re-paired null" if r["p_rank"] > 0.01 else None,
                                                      note="HIP %d; Gaia fainter than Hipparcos at %+.1f sigma (%+.3f mag); W2 trend %+.1f sigma (%+.3f mag/decade, positive fading)" % (
                                                          r["hip"], r["z_opt"], r["opt_mag"], r["z_w2"], r["w2_slope"])) for r in top])
    print("recorded")


def neowise_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "joint", "joint_record"))
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--sample", choices=("nearest", "hip_gaia"), default="nearest")
    a = p.parse_args()
    if a.mode == "fetch":
        neowise_fetch(a.threads, a.sample)
    elif a.mode == "analyse":
        neowise_analyse(a.sample)
    elif a.mode == "joint_record":
        neowise_joint_record()
    else:
        neowise_joint()


def neowise_cli():
    quiet_warnings()
    neowise_main()


# --------------------------------------------------------------------------------------------------------------

alerts_DOC = r"""Gaia Science Alerts matched to stars within 100 pc, excluding flares, CV outbursts and accretion.

    python -I seti.py dimming alerts"""

alerts_ROOT = os.environ.get("ALERTS_ROOT", core.data("gaia_alerts"))


def alerts_main():
    rows = list(csv.reader(open(os.path.join(alerts_ROOT, "alerts.csv"), encoding="utf-8", errors="replace")))
    hdr = [h.strip().lstrip("#") for h in rows[0]]
    al = [dict(zip(hdr, r)) for r in rows[1:] if len(r) >= 8]
    pts = [dict(source_id=a["Name"], ra=float(a["RaDeg"]), dec=float(a["DecDeg"])) for a in al]
    m = []
    for k in range(0, len(pts), 5000):
        m += core.xmatch(pts[k:k + 5000], "I/355/gaiadr3", 2.0)
    near = {}
    for x in m:
        try:
            plx = float(x.get("Plx") or 0)
        except ValueError:
            plx = 0.0
        if plx > 10:
            near.setdefault(x["source_id"], []).append(x)
    by = {a["Name"]: a for a in al}
    out = []
    for n, xs in near.items():
        a, x = by[n], min(xs, key=lambda x: float(x["angDist"]))
        out.append(dict(name=n, date=a["Date"], cls=a["Class"], comment=a["Comment"], alert=a["AlertMag"],
                        hist=a["HistoricMag"], gaia=x["Source"], plx=float(x["Plx"]), g=x.get("Gmag"),
                        bprp=x.get("BP-RP"), sep=float(x["angDist"])))
    json.dump(out, open(os.path.join(alerts_ROOT, "near.json"), "w"), indent=1)
    print("%d alerts, %d on Gaia DR3 stars with parallax above 10 mas" % (len(al), len(out)))
    for o in sorted(out, key=lambda o: -o["plx"]):
        print("  %s %s %5.1f pc G %s BP-RP %s: %s from %s, class %s, %s" % (
            o["name"], o["date"][:10], 1000 / o["plx"], o["g"], o["bprp"], o["alert"], o["hist"], o["cls"], o["comment"]))


def alerts_record():
    """Record one row per alert on a nearby star with the team's comment."""
    near = json.load(open(os.path.join(alerts_ROOT, "near.json")))
    core.record_search(
        "gaia_alerts_near", "Gaia Science Alerts on stars within 100 pc",
        archive="Gaia photometric science alerts (Cambridge), CDS XMatch with Gaia DR3", release="alerts 2014-2025",
        selection="alerts within 2 arcsec of a Gaia DR3 source with parallax above 10 mas", code="seti.py dimming alerts",
        params=dict(radius_arcsec=2, parallax_mas=10), n_targets=len(near),
        null_summary="flares on red and brown dwarfs, two dwarf-nova outbursts, a young-star flare, a flare star's slow "
                     "rise and a distant dwarf nova behind a spurious parallax",
        sensitivity="alert thresholds of about 0.3-1 mag against the historic mean",
        result="no unexplained brightening of a nearby star", ledger="Gaia Science Alerts on nearby stars")
    core.clear_targets("gaia_alerts_near")
    core.record_targets("gaia_alerts_near", [
        dict(target="gaia_dr3:%s" % o["gaia"], stat=float(o["hist"]) - float(o["alert"]) if o["hist"] and o["alert"]
             else None, unit="mag brighter than the historic mean", vet="explained_by_alert_team",
             note="%s %s: %s" % (o["name"], o["date"][:10], o["comment"] or o["cls"])) for o in near])


def alerts_cli():
    if sys.argv[1:] == ["record"]:
        alerts_record()
    else:
        alerts_main()


# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------

occulters_DOC = r"""Gaia DR3 astrometric noise of long-occultation stars (VVV-WIT-08, -07, Gaia17bpp, TYC 2505-672-1, ASASSN-21qj, J1407; EE Cep control) ranked among field stars matched in G and colour within a cone that doubles until it holds the control count.

    python -I seti.py dimming occulters [record] [--stars NAME ...] [--controls NAME ...] [--dg 0.3] [--dcolour 0.2] [--n-controls 100] [--radius-deg 0.15]"""

occulters_STARS = ("VVV-WIT-08", "VVV-WIT-07", "Gaia17bpp", "TYC 2505-672-1", "ASASSN-21qj", "V1400 Cen", "EE Cep")
occulters_COLS = ("source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, parallax_error, ruwe, astrometric_excess_noise, "
                  "astrometric_excess_noise_sig, ipd_gof_harmonic_amplitude, ipd_frac_multi_peak, astrometric_params_solved")


def occulters_main(record=False, stars=occulters_STARS, controls=("EE Cep",), dg=0.3, dcolour=0.2, n_controls=100, radius_deg=0.15):
    out = []
    for name in stars:
        q = ("SELECT i.id FROM ident AS i JOIN ident AS j ON i.oidref = j.oidref WHERE j.id = '%s' AND i.id LIKE 'Gaia DR3 %%'" % name)
        ids = [str(r["id"]).split()[-1] for r in core.simbad(q)]
        if not ids:
            print("%s: no Gaia DR3 identifier in SIMBAD" % name)
            out.append(dict(name=name, gaia=None))
            continue
        # Astro Data Lab's indexed Gaia DR3 copy
        g = core.datalab("SELECT %s FROM gaia_dr3.gaia_source WHERE source_id = %s" % (occulters_COLS, ids[0]))
        if not g:
            out.append(dict(name=name, gaia=None))
            continue
        t = {k: num(v) for k, v in g[0].items()}
        # the cone doubles until it holds the control count
        ctl, radius = [], radius_deg
        while len(ctl) < n_controls and radius <= 5.0:
            ctl = core.datalab("SELECT %s FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, %.6f, %.6f, %.3f) = 't' "
                                   "AND phot_g_mean_mag BETWEEN %.3f AND %.3f AND bp_rp BETWEEN %.3f AND %.3f AND "
                                   "astrometric_params_solved >= 31 AND source_id != %s LIMIT 400"
                                   % (occulters_COLS, t["ra"], t["dec"], radius, t["phot_g_mean_mag"] - dg, t["phot_g_mean_mag"] + dg,
                                      (t["bp_rp"] or 0) - dcolour, (t["bp_rp"] or 0) + dcolour, ids[0]))
            radius *= 2.0
        ctl = [{k: num(v) for k, v in c.items()} for c in ctl]

        def rank(key):
            vals = [c[key] for c in ctl if c[key] is not None]
            return (sum(1 for v in vals if v >= (t[key] or 0)) / len(vals) if vals else None, len(vals),
                    float(np.median(vals)) if vals else None)
        r_ruwe, n, med_ruwe = rank("ruwe")
        r_aen, _, med_aen = rank("astrometric_excess_noise")
        out.append(dict(name=name, gaia=ids[0], g=t["phot_g_mean_mag"], bp_rp=t["bp_rp"], plx=t["parallax"], plx_err=t["parallax_error"],
                        ruwe=t["ruwe"], aen=t["astrometric_excess_noise"], aen_sig=t["astrometric_excess_noise_sig"],
                        ipd_amp=t["ipd_gof_harmonic_amplitude"], ipd_multi=t["ipd_frac_multi_peak"], n_ctl=n,
                        frac_ruwe=r_ruwe, frac_aen=r_aen, ctl_ruwe=med_ruwe, ctl_aen=med_aen))
        print("%-15s Gaia DR3 %s G %.2f: RUWE %.2f (controls' median %.2f, %.0f%% as high), excess noise %.2f mas at %.0f sigma "
              "(median %.2f, %.0f%% as high), multi-peak %.0f%%, %d controls"
              % (name, ids[0], t["phot_g_mean_mag"], t["ruwe"] or 0, med_ruwe or 0, 100 * (r_ruwe or 0), t["astrometric_excess_noise"] or 0,
                 t["astrometric_excess_noise_sig"] or 0, med_aen or 0, 100 * (r_aen or 0), t["ipd_frac_multi_peak"] or 0, n))
    json.dump(out, open(os.path.join(core.RESULTS, "occulters_gaia.json"), "w"), indent=1)
    if record:
        found = [x for x in out if x.get("gaia")]
        core.record_search(
            "long_occulters_gaia", "Long occulters' companions", archive="Gaia DR3 gaia_source; SIMBAD",
            release="Gaia DR3", selection=", ".join(stars), code=os.path.abspath(__file__), entry="occulters",
            n_targets=len(found), params=dict(radius_deg="%g, doubled until %d controls" % (radius_deg, n_controls), dg=dg, dcolour=dcolour, n_controls=400),
            null_summary="field stars matched in G and colour: the fraction with RUWE and excess noise as high",
            sensitivity="an unresolved orbit's photocentre motion above the astrometric excess noise of the controls, about "
                        "%s mas" % ", ".join("%.2f" % x["ctl_aen"] for x in found if x["ctl_aen"] is not None),
            ledger="Long occulters' companions",
            result="; ".join("%s RUWE %.2f (%.0f%% of controls as high), excess noise %.2f mas at %.0f sigma (%.0f%% as high)"
                             % (x["name"], x["ruwe"] or 0, 100 * (x["frac_ruwe"] or 0), x["aen"] or 0, x["aen_sig"] or 0,
                                100 * (x["frac_aen"] or 0)) for x in found))
        core.clear_targets("long_occulters_gaia")
        core.record_targets("long_occulters_gaia", [dict(
            target="gaia_dr3:" + x["gaia"], stat=x["ruwe"], p=x["frac_ruwe"], unit="RUWE; p the fraction of matched field stars as high",
            vet="disk-bearing companion established (control)" if x["name"] in controls else None,
            note="%s; G %.2f; excess noise %.2f mas at %.0f sigma, %.0f%% of %d controls as high; controls' median RUWE %.2f"
                 % (x["name"], x["g"], x["aen"] or 0, x["aen_sig"] or 0, 100 * (x["frac_aen"] or 0), x["n_ctl"], x["ctl_ruwe"] or 0))
            for x in found])
        print("recorded %d stars" % len(found))


achromatic_DOC = r"""Achromatic long-period modulation in ZTF: A_r/A_g near 1 in catalogued rotators (spots give 0.75-0.85), vetted for blends, binaries and young-star dust.

    python -I seti.py dimming achromatic select
    python -I seti.py dimming achromatic fetch [--threads N]
    python -I seti.py dimming achromatic analyse
    python -I seti.py dimming achromatic record"""

achromatic_ROOT = os.environ.get("ACHROMATIC_ROOT", core.data("achromatic"))
achromatic_CAT = os.environ.get("ACHROMATIC_CAT", core.data("dynamo_dead", "ztf_amplitudes.ecsv"))
achromatic_OUT = "achromatic_modulation.json"
achromatic_CLASSES = ("BYDra", "RSCVN")
achromatic_PMIN, achromatic_PMAX, achromatic_AMIN, achromatic_GMAX = 5.0, 500.0, 0.15, 18.0
achromatic_SELECT, achromatic_CONTROL, achromatic_NCONTROL = 0.90, (0.70, 0.90), 300
achromatic_GREY, achromatic_MAX, achromatic_PHASE, achromatic_NMIN = 0.97, 1.15, 0.05, 40
achromatic_W1W2 = 0.2
achromatic_LC = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID=%s&FORMAT=csv&BAD_CATFLAGS_MASK=32768"


def achromatic_select():
    from astropy.table import Table
    os.makedirs(achromatic_ROOT, exist_ok=True)
    t = Table.read(achromatic_CAT, format="ascii.ecsv")
    rows = []
    for r in t:
        ty = str(r["Type"])
        if ty not in achromatic_CLASSES:
            continue
        P, g, ag, ar = (float(r[k]) if r[k] == r[k] else None for k in ("Per", "gmag", "gAmp", "rAmp"))
        if not P or not ag or not ar or g is None or not (achromatic_PMIN <= P <= achromatic_PMAX) or ag < achromatic_AMIN or g > achromatic_GMAX:
            continue
        rows.append(dict(id=str(r["ID"]), ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"]), P=P, g=g, r=float(r["rmag"]) if r["rmag"] == r["rmag"] else None,
                         ag=ag, ar=ar, ratio=ar / ag, type=ty, ng=int(r["Ng"]), nr=int(r["Nr"])))
    rng = np.random.default_rng(3)
    cand = [x for x in rows if x["ratio"] >= achromatic_SELECT]
    pool = [x for x in rows if achromatic_CONTROL[0] <= x["ratio"] < achromatic_CONTROL[1]]
    ctrl = [pool[i] for i in rng.choice(len(pool), min(achromatic_NCONTROL, len(pool)), replace=False)]
    for x in cand:
        x["role"] = "candidate"
    for x in ctrl:
        x["role"] = "control"
    ratios = np.array([x["ratio"] for x in rows])
    json.dump(dict(n_class=len(rows), ratio_median=float(np.median(ratios)), ratio_p90=float(np.percentile(ratios, 90)), targets=cand + ctrl),
              open(os.path.join(achromatic_ROOT, "targets.json"), "w"), indent=0)
    print("%d rotational variables at %.0f-%.0f d, A_g %.2f or more, g %.0f or brighter: catalogue A_r/A_g median %.3f, 90th percentile %.3f; %d at %.2f or more and %d controls at %.2f-%.2f"
          % (len(rows), achromatic_PMIN, achromatic_PMAX, achromatic_AMIN, achromatic_GMAX, np.median(ratios), np.percentile(ratios, 90), len(cand), achromatic_SELECT,
             len(ctrl), achromatic_CONTROL[0], achromatic_CONTROL[1]))


def achromatic_fetch(threads=6):
    """ZTF DR23 g and r light curves of each target by object identifier."""
    d = json.load(open(os.path.join(achromatic_ROOT, "targets.json")))
    lc = os.path.join(achromatic_ROOT, "lc")
    os.makedirs(lc, exist_ok=True)

    def one(x):
        p = os.path.join(lc, x["id"] + ".csv")
        if os.path.exists(p):
            return 1
        q = ("SELECT oid, filtercode, ngoodobsrel FROM ztf_objects_dr23 WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f)) = 1"
             % (x["ra"], x["dec"], 1.5 / 3600))
        try:
            objs = core.tap("irsa", q, timeout=300)
        except Exception as e:
            print("  %s: objects query failed: %s" % (x["id"], str(e)[:80]), flush=True)
            return 0
        parts = []
        for o in objs:
            if o["filtercode"] in ("zg", "zr") and (num(o["ngoodobsrel"]) or 0) >= 10:
                try:
                    parts.append(core.get(achromatic_LC % o["oid"], timeout=300, retries=2).decode("utf-8", "replace"))
                except Exception as e:
                    print("  %s: light curve %s failed: %s" % (x["id"], o["oid"], str(e)[:80]), flush=True)
        lines = [ln for part in parts for ln in part.splitlines()[1:] if ln.strip()]
        head = parts[0].splitlines()[0] if parts else "oid,expid,hjd,mjd,mag,magerr,catflags,filtercode"
        open(p, "w", encoding="utf-8").write("\n".join([head] + lines) + "\n")
        return 1 if lines else 0
    got = core.pmap(one, d["targets"], threads=threads)
    print("%d of %d targets with light curves" % (sum(got), len(d["targets"])))


def achromatic_fit(t, m, e, P):
    """Mean plus two harmonics at period P with 4 sigma clipping: amplitude, error, phases, kept points, scatter; None below five points."""
    ph = (t / P) % 1.0
    A = np.column_stack([np.ones_like(ph), np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), np.sin(4 * np.pi * ph), np.cos(4 * np.pi * ph)])
    w = 1 / np.maximum(e, 0.01) ** 2
    keep = np.ones(len(t), bool)
    coef = None
    for _ in range(3):
        if keep.sum() < 6:
            return None
        Aw = A[keep] * np.sqrt(w[keep])[:, None]
        coef = np.linalg.lstsq(Aw, m[keep] * np.sqrt(w[keep]), rcond=None)[0]
        res = m - A @ coef
        s = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < 4 * max(s, 0.01)
    n = int(keep.sum())
    if n < 6:
        return None
    Aw = A[keep] * np.sqrt(w[keep])[:, None]
    coef = np.linalg.lstsq(Aw, m[keep] * np.sqrt(w[keep]), rcond=None)[0]
    res = (m - A @ coef)[keep]
    chi2 = float(np.sum(res ** 2 * w[keep]) / max(n - 5, 1))
    cov = np.linalg.pinv(Aw.T @ Aw) * max(chi2, 1.0)
    grid = np.linspace(0, 1, 400, endpoint=False)
    G = np.column_stack([np.ones_like(grid), np.sin(2 * np.pi * grid), np.cos(2 * np.pi * grid), np.sin(4 * np.pi * grid), np.cos(4 * np.pi * grid)])
    model = G @ coef
    draws = np.random.default_rng(5).multivariate_normal(coef, cov, 300)
    models = G @ draws.T
    return dict(amp=float(model.max() - model.min()), e_amp=float(np.std(models.max(0) - models.min(0))), delta=float(math.atan2(coef[2], coef[1]) / (2 * np.pi)),
                a1=float(math.hypot(coef[1], coef[2])), a2=float(math.hypot(coef[3], coef[4])), n=n, chi2=chi2, scatter=float(1.4826 * np.median(np.abs(res))))


def achromatic_read(path):
    """(t, mag, err) per band of one target file."""
    import csv
    out = {"zg": [], "zr": []}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            if r["filtercode"] in out and int(r["catflags"]) == 0:
                out[r["filtercode"]].append((float(r["mjd"]), float(r["mag"]), float(r["magerr"])))
        except (KeyError, ValueError):
            continue
    return {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v]), np.array([x[2] for x in v])) for k, v in out.items()}


def achromatic_analyse():
    d = json.load(open(os.path.join(achromatic_ROOT, "targets.json")))
    lc = os.path.join(achromatic_ROOT, "lc")
    rows = []
    for x in d["targets"]:
        p = os.path.join(lc, x["id"] + ".csv")
        if not os.path.exists(p):
            continue
        bands = achromatic_read(p)
        fg = achromatic_fit(*bands["zg"], x["P"]) if len(bands["zg"][0]) >= achromatic_NMIN else None
        fr = achromatic_fit(*bands["zr"], x["P"]) if len(bands["zr"][0]) >= achromatic_NMIN else None
        if not fg or not fr or fg["amp"] <= 0 or fr["amp"] <= 0:
            continue
        ratio = fr["amp"] / fg["amp"]
        e_ratio = ratio * math.hypot(fg["e_amp"] / fg["amp"], fr["e_amp"] / fr["amp"])
        dphase = ((fr["delta"] - fg["delta"]) + 0.5) % 1.0 - 0.5
        rows.append(dict(x, fit_g=fg, fit_r=fr, ratio_lc=ratio, e_ratio=e_ratio, dphase=dphase,
                         grey=(ratio - 3 * e_ratio >= achromatic_GREY and ratio <= achromatic_MAX and abs(dphase) <= achromatic_PHASE and fg["amp"] >= achromatic_AMIN)))
    ctrl = [r for r in rows if r["role"] == "control"]
    cand = [r for r in rows if r["role"] == "candidate"]
    grey = [r for r in cand if r["grey"]]
    print("%d targets fitted: %d controls with light-curve A_r/A_g median %.3f (robust scatter %.3f, catalogue median %.3f); %d candidates, %d grey at 3 sigma (ratio %.2f-%.2f, bands in phase)" % (
        len(rows), len(ctrl), np.median([r["ratio_lc"] for r in ctrl]) if ctrl else 0, 1.4826 * np.median(np.abs(np.array([r["ratio_lc"] for r in ctrl]) - np.median([r["ratio_lc"] for r in ctrl]))) if ctrl else 0,
        np.median([r["ratio"] for r in ctrl]) if ctrl else 0, len(cand), len(grey), achromatic_GREY, achromatic_MAX))
    # Gaia DR3 and AllWISE checks of grey candidates
    if grey:
        pos = [dict(source_id=i, ra=r["ra"], dec=r["dec"]) for i, r in enumerate(grey)]
        gaia, wise = {}, {}
        for m in core.xmatch(pos, "I/355/gaiadr3", 3.0, id_col="source_id", ra_col="ra", dec_col="dec", timeout=600):
            gaia.setdefault(int(m["source_id"]), []).append(m)
        for m in core.xmatch(pos, "II/328/allwise", 3.0, id_col="source_id", ra_col="ra", dec_col="dec", timeout=600):
            i = int(m["source_id"])
            if i not in wise or float(m["angDist"]) < float(wise[i]["angDist"]):
                wise[i] = m
        for i, r in enumerate(grey):
            gm = sorted(gaia.get(i, []), key=lambda m: float(m["angDist"]))
            w = wise.get(i)
            r["gaia"] = dict(n=len(gm), ruwe=num(gm[0].get("RUWE")) if gm else None, bprp=num(gm[0].get("BP-RP")) if gm else None, G=num(gm[0].get("Gmag")) if gm else None,
                             excess=num(gm[0].get("E(BP/RP)")) if gm else None, plx=num(gm[0].get("Plx")) if gm else None,
                             neighbour=([float(m["angDist"]) for m in gm[1:]] if len(gm) > 1 else []))
            # height above the single-star main sequence; an equal companion lifts a star by up to 0.75 mag
            gg = r["gaia"]
            gg["above_ms"] = (1.7 + 3.6 * gg["bprp"]) - (gg["G"] + 5 * math.log10(gg["plx"]) - 10) if gg["plx"] and gg["plx"] > 0 and gg["G"] is not None and gg["bprp"] is not None and 0.4 < gg["bprp"] < 3.2 else None
            r["wise"] = dict(w1w2=(num(w.get("W1mag")) - num(w.get("W2mag"))) if w and num(w.get("W1mag")) is not None and num(w.get("W2mag")) is not None else None) if w else None
            why = []
            if r["gaia"]["n"] > 1:
                why.append("a Gaia neighbour at %.1f arcsec" % min(r["gaia"]["neighbour"]))
            if r["gaia"]["ruwe"] and r["gaia"]["ruwe"] > 1.4:
                why.append("RUWE %.1f" % r["gaia"]["ruwe"])
            if gg["above_ms"] is not None and gg["above_ms"] >= 1.5:
                why.append("%.2f mag above the main sequence, an evolved star whose class pairs it with a companion that dilutes the spots' colour" % gg["above_ms"])
            elif gg["above_ms"] is not None and gg["above_ms"] >= 0.5:
                why.append("%.2f mag above the main sequence, an unresolved companion" % gg["above_ms"])
            if r["gaia"]["excess"] and r["gaia"]["bprp"] is not None and r["gaia"]["excess"] > 1.3 + 0.06 * r["gaia"]["bprp"] ** 2:
                why.append("BP/RP excess factor %.2f" % r["gaia"]["excess"])
            if r["wise"] and r["wise"]["w1w2"] is not None and r["wise"]["w1w2"] > achromatic_W1W2:
                why.append("W1-W2 %.2f" % r["wise"]["w1w2"])
            r["explained"] = "; ".join(why) or None
        left = [r for r in grey if not r["explained"]]
        print("  %d grey candidates: %d with a Gaia neighbour, high RUWE, a BP/RP excess, a seat above the main sequence or warm dust; %d left" % (len(grey), len(grey) - len(left), len(left)))
        for r in sorted(grey, key=lambda r: -(r["ratio_lc"] - 3 * r["e_ratio"])):
            print("  %s %s P %.2f d g %.1f: A_g %.3f A_r %.3f (catalogue %.3f %.3f), ratio %.3f +- %.3f, phase %+.3f, n %d/%d; Gaia %s RUWE %s BP-RP %s excess %s above MS %s; W1-W2 %s; %s" % (
                r["id"], r["type"], r["P"], r["g"], r["fit_g"]["amp"], r["fit_r"]["amp"], r["ag"], r["ar"], r["ratio_lc"], r["e_ratio"], r["dphase"], r["fit_g"]["n"], r["fit_r"]["n"],
                r["gaia"]["n"], "%.2f" % r["gaia"]["ruwe"] if r["gaia"]["ruwe"] else "-", "%.2f" % r["gaia"]["bprp"] if r["gaia"]["bprp"] is not None else "-",
                "%.2f" % r["gaia"]["excess"] if r["gaia"]["excess"] else "-", "%+.2f" % r["gaia"]["above_ms"] if r["gaia"]["above_ms"] is not None else "-",
                ("%.2f" % r["wise"]["w1w2"]) if r["wise"] and r["wise"]["w1w2"] is not None else "-", r["explained"] or "unexplained"))
    json.dump(dict(n=len(rows), n_control=len(ctrl), n_candidate=len(cand), n_grey=len(grey), rows=rows, n_class=d["n_class"],
                   params=dict(classes=achromatic_CLASSES, pmin=achromatic_PMIN, pmax=achromatic_PMAX, amin=achromatic_AMIN, gmax=achromatic_GMAX, select=achromatic_SELECT,
                               grey=achromatic_GREY, max=achromatic_MAX, phase=achromatic_PHASE, nmin=achromatic_NMIN)),
              open(core.results(achromatic_OUT), "w"), indent=0)


def achromatic_record():
    d = json.load(open(core.results(achromatic_OUT)))
    rows = d["rows"]
    ctrl = [r for r in rows if r["role"] == "control"]
    grey = [r for r in rows if r["role"] == "candidate" and r["grey"]]
    left = [r for r in grey if not r.get("explained")]
    p = d["params"]
    core.record_search(
        "ztf_achromatic", "Achromatic long-period modulation", archive="ZTF periodic variables (Chen et al. 2020, J/ApJS/249/18); ZTF DR23 light curves (IRSA); Gaia DR3 and AllWISE by CDS XMatch",
        release="ZTF DR23", selection="BY Dra and RS CVn variables at %.0f-%.0f d with g amplitudes of %.2f or more and g brighter than %.0f (%d), the %d with catalogue A_r/A_g of %.2f or more and %d controls at 0.7-0.9" % (
            p["pmin"], p["pmax"], p["amin"], p["gmax"], d["n_class"], d["n_candidate"], p["select"], d["n_control"]),
        code=os.path.abspath(__file__), entry="achromatic", n_targets=len(rows), params=p,
        null_summary="the controls' light-curve A_r/A_g: median %.3f, robust scatter %.3f; spots give 0.75-0.85" % (
            np.median([r["ratio_lc"] for r in ctrl]) if ctrl else 0, 1.4826 * np.median(np.abs(np.array([r["ratio_lc"] for r in ctrl]) - np.median([r["ratio_lc"] for r in ctrl]))) if ctrl else 0),
        sensitivity="a ratio of %.2f or more at 3 sigma, which the fits reach at errors of 0.02-0.05 for g brighter than 17" % p["grey"],
        ledger="Achromatic long-period modulation",
        result="%d light curves fitted: %d grey at 3 sigma, %d of them with a Gaia neighbour within 3 arcsec, RUWE above 1.4, a BP/RP excess, 0.5 mag or more above the main sequence or W1-W2 above %.1f, %d left%s" % (
            len(rows), len(grey), len(grey) - len(left), achromatic_W1W2, len(left), (": " + "; ".join("%s (%.2f +- %.2f at %.1f d)" % (r["id"], r["ratio_lc"], r["e_ratio"], r["P"]) for r in left[:10])) if left else ""))
    core.clear_targets("ztf_achromatic")
    core.record_targets("ztf_achromatic", [dict(target="ztf:%s" % r["id"], stat=r["ratio_lc"], unit="A_r/A_g from the folded light curves",
                                                   vet=(r.get("explained") or ("grey, unexplained" if r["grey"] else "chromatic")) if r["role"] == "candidate" else "control",
                                                   note="%s, P %.2f d, g %.1f, A_g %.3f, A_r %.3f, ratio %.3f +- %.3f, phase %+.3f" % (
                                                       r["type"], r["P"], r["g"], r["fit_g"]["amp"], r["fit_r"]["amp"], r["ratio_lc"], r["e_ratio"], r["dphase"])) for r in rows])
    print("recorded %d light curves" % len(rows))


def achromatic_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("select", "fetch", "analyse", "record"))
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    if a.mode == "fetch":
        achromatic_fetch(a.threads)
    else:
        {"select": achromatic_select, "analyse": achromatic_analyse, "record": achromatic_record}[a.mode]()


chromatic_DOC = r"""Chromatic-only variables: Gaia DR3 sources with BP and RP scatter but constant G, then BP-RP anticorrelation in epoch photometry.

    python -I seti.py dimming chromatic fetch
    python -I seti.py dimming chromatic analyse
    python -I seti.py dimming chromatic record"""

chromatic_ROOT = os.environ.get("CHROMATIC_ROOT", core.data("chromatic"))
chromatic_OUT = "chromatic_only_variables.json"
chromatic_GMAX, chromatic_NMIN, chromatic_SG, chromatic_SC, chromatic_Z = 16.0, 20, 0.003, 0.02, 5.0


def chromatic_fetch():
    os.makedirs(chromatic_ROOT, exist_ok=True)
    cols = ("source_id, num_selected_g_fov, mean_mag_g_fov, std_dev_mag_g_fov, mad_mag_g_fov, range_mag_g_fov, num_selected_bp, mean_mag_bp, std_dev_mag_bp, mad_mag_bp, num_selected_rp, "
            "mean_mag_rp, std_dev_mag_rp, mad_mag_rp, std_dev_over_rms_err_mag_g_fov, std_dev_over_rms_err_mag_bp, std_dev_over_rms_err_mag_rp, in_vari_classification_result, "
            "in_vari_eclipsing_binary, in_vari_rotation_modulation, in_vari_short_timescale")
    base = "FROM gaiadr3.vari_summary WHERE mean_mag_g_fov < %g AND num_selected_g_fov >= %d" % (chromatic_GMAX, chromatic_NMIN)
    p = os.path.join(chromatic_ROOT, "tight.json")
    if not os.path.exists(p):
        rows = core.tap("gaia", "SELECT %s %s AND std_dev_mag_g_fov < %g AND std_dev_mag_bp > %g AND std_dev_mag_rp > %g" % (cols, base, chromatic_SG, chromatic_SC, chromatic_SC), maxrec=100000, timeout=1200)
        json.dump(rows, open(p, "w"))
        print("%d candidates by the scatter cuts" % len(rows))
    p = os.path.join(chromatic_ROOT, "vari_summary_quiet_g.json")
    if not os.path.exists(p):
        rows = core.tap("gaia", "SELECT %s %s AND std_dev_mag_g_fov < 0.01" % (cols, base), maxrec=2000000, timeout=1800)
        json.dump(rows, open(p, "w"))
        print("%d sources quiet in G for the null" % len(rows))


def chromatic_ratio(r):
    sg, sb, sr = (num(r[k]) for k in ("std_dev_mag_g_fov", "std_dev_mag_bp", "std_dev_mag_rp"))
    if not sg or sb is None or sr is None or sg <= 0:
        return None
    return math.hypot(sb, sr) / sg


def chromatic_epochs(sid):
    """BP and RP magnitudes on transits holding both."""
    os.makedirs(os.path.join(gaia_ROOT, "epoch"), exist_ok=True)
    if not gaia_fetch_one(str(sid)):
        return None
    s = gaia_series(str(sid))
    tb, mb = s["BP"]
    tr, mr = s["RP"]
    if len(tb) < 8 or len(tr) < 8:
        return None
    # BP and RP of one transit are seconds apart
    j = np.searchsorted(tr, tb)
    j = np.clip(j, 0, len(tr) - 1)
    near = np.abs(tr[j] - tb) < 0.01
    if near.sum() < 8:
        return None
    return mb[near], mr[j][near], s["G"]


def chromatic_analyse():
    pool = json.load(open(os.path.join(chromatic_ROOT, "vari_summary_quiet_g.json")))
    tight = json.load(open(os.path.join(chromatic_ROOT, "tight.json")))
    # scatter-ratio distribution in G bins among G-quiet sources
    bins = np.arange(5.0, chromatic_GMAX + 0.5, 0.5)
    stats = {}
    for lo in bins:
        v = np.array([chromatic_ratio(r) for r in pool if num(r["mean_mag_g_fov"]) is not None and lo <= num(r["mean_mag_g_fov"]) < lo + 0.5])
        v = np.log10(v[np.isfinite(v) & (v > 0)]) if len(v) else v
        if len(v) >= 50:
            stats[lo] = (float(np.median(v)), float(1.4826 * np.median(np.abs(v - np.median(v)))), int(len(v)))
    cands = []
    for r in tight:
        g, rho = num(r["mean_mag_g_fov"]), chromatic_ratio(r)
        if g is None or rho is None:
            continue
        lo = math.floor(g / 0.5) * 0.5
        if lo not in stats:
            continue
        med, sd, n = stats[lo]
        z = (math.log10(rho) - med) / sd if sd > 0 else 0.0
        if z >= chromatic_Z:
            cands.append(dict(source_id=r["source_id"], G=g, s_g=num(r["std_dev_mag_g_fov"]), s_bp=num(r["std_dev_mag_bp"]), s_rp=num(r["std_dev_mag_rp"]), n_g=int(num(r["num_selected_g_fov"]) or 0),
                              ratio=rho, z=z, snr_g=num(r["std_dev_over_rms_err_mag_g_fov"]), snr_bp=num(r["std_dev_over_rms_err_mag_bp"]), snr_rp=num(r["std_dev_over_rms_err_mag_rp"]),
                              classified=str(r["in_vari_classification_result"]).lower() == "true", eclipsing=str(r["in_vari_eclipsing_binary"]).lower() == "true",
                              rotation=str(r["in_vari_rotation_modulation"]).lower() == "true", short=str(r["in_vari_short_timescale"]).lower() == "true"))
    print("%d sources quiet in G (scatter below 0.01) for the null in %d bins of G; %d pass the scatter cuts (G scatter below %.3f, BP and RP above %.2f), %d of them beyond the ratio's %.0f sigma tail at their G" % (
        len(pool), len(stats), len(tight), chromatic_SG, chromatic_SC, len(cands), chromatic_Z))
    # finalists' BP against RP over shared transits
    for c in sorted(cands, key=lambda c: -c["z"])[:60]:
        ep = chromatic_epochs(c["source_id"])
        if ep is None:
            c["corr"], c["n_pairs"] = None, 0
        else:
            mb, mr, (tg, mg) = ep
            c["corr"] = float(np.corrcoef(mb, mr)[0, 1]) if len(mb) > 3 else None
            c["n_pairs"] = int(len(mb))
            c["s_g_epochs"] = float(np.std(mg)) if len(mg) else None
            c["s_colour_epochs"] = float(np.std(mb - mr))
        c["kind"] = ("a neighbour in the windows (BP and RP vary together)" if c["corr"] is not None and c["corr"] > 0.3 else
                     ("chromatic only (BP against RP)" if c["corr"] is not None and c["corr"] < -0.3 else "undecided"))
        print("  Gaia DR3 %s: G %.2f, scatter G %.4f BP %.3f RP %.3f (%d transits), ratio %.0f, %.1f sigma; epochs %d pairs, BP-RP correlation %s, colour scatter %s; %s%s" % (
            c["source_id"], c["G"], c["s_g"], c["s_bp"], c["s_rp"], c["n_g"], c["ratio"], c["z"], c["n_pairs"], "%.2f" % c["corr"] if c["corr"] is not None else "-",
            "%.3f" % c["s_colour_epochs"] if c.get("s_colour_epochs") is not None else "-", c["kind"], ", eclipsing" if c["eclipsing"] else (", rotation" if c["rotation"] else "")))
    json.dump(dict(n_pool=len(pool), n_tight=len(tight), n_cand=len(cands), stats={str(k): v for k, v in stats.items()}, cands=cands,
                   params=dict(gmax=chromatic_GMAX, nmin=chromatic_NMIN, s_g=chromatic_SG, s_colour=chromatic_SC, z=chromatic_Z)), open(core.results(chromatic_OUT), "w"), indent=0)


def chromatic_record():
    d = json.load(open(core.results(chromatic_OUT)))
    c = d["cands"]
    anti = [x for x in c if x.get("corr") is not None and x["corr"] < -0.3]
    blend = [x for x in c if x.get("corr") is not None and x["corr"] > 0.3]
    core.record_search(
        "chromatic_only_variables", "Chromatic-only variables", archive="Gaia DR3 vari_summary; epoch photometry by DataLink or VizieR I/355/epphot", release="Gaia DR3",
        selection="variables brighter than G %.0f with %d or more transits: %d with G scatter below %.3f and BP and RP scatter above %.2f" % (d["params"]["gmax"], d["params"]["nmin"], d["n_tight"], d["params"]["s_g"], d["params"]["s_colour"]),
        code=os.path.abspath(__file__), entry="chromatic", n_targets=d["n_tight"], params=d["params"],
        null_summary="the ratio sqrt(s_BP^2 + s_RP^2)/s_G of %d sources quiet in G in half-magnitude bins, a candidate beyond its %.0f sigma tail" % (d["n_pool"], d["params"]["z"]),
        sensitivity="colour modulation of 1-2 per cent at G below 15 with G constant to 0.3 per cent", ledger="Chromatic-only variables",
        result="%d beyond the tail; of those with epoch photometry %d show BP and RP varying together (a neighbour in the windows) and %d against each other%s" % (
            d["n_cand"], len(blend), len(anti), (": " + "; ".join("Gaia DR3 %s (G %.1f, correlation %.2f)" % (x["source_id"], x["G"], x["corr"]) for x in anti[:8])) if anti else ""))
    core.clear_targets("chromatic_only_variables")
    core.record_targets("chromatic_only_variables", [dict(target="gaia_dr3:%s" % x["source_id"], stat=x["z"], unit="scatter-ratio excess over the quiet sources at the same G, sigma",
                                                              vet=x.get("kind"), note="G %.2f, scatter G %.4f BP %.3f RP %.3f, epoch correlation %s" % (x["G"], x["s_g"], x["s_bp"], x["s_rp"], "%.2f" % x["corr"] if x.get("corr") is not None else "-")) for x in c])
    print("recorded %d candidates" % d["n_cand"])


def chromatic_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": chromatic_fetch, "analyse": chromatic_analyse, "record": chromatic_record}[mode]()


# --------------------------------------------------------------------------------------------------------------
# Hipparcos stars Gaia cannot find

hipgone_DOC = r"""Hipparcos stars Gaia cannot find: stars of the Hipparcos new reduction at Hp 7-11 without a Gaia DR3 best neighbour, their positions carried from 1991.25 to 2016.0 with their Hipparcos proper motions and matched to Gaia DR3 within 10 arcsec by CDS XMatch; a star with no Gaia source within 10 arcsec of where it should be has gone since Hipparcos measured it, each read in SIMBAD for its type and a Gaia DR3 identifier of its own, and in Gaia DR3 within 30 arcsec and 1 mag of SIMBAD's position, where a Hipparcos motion that misses leaves the star.

    python -I seti.py dimming hipgone [record] [--hp-min 7] [--hp-max 11] [--radius-arcsec 10] [--near-arcsec 30] [--dmag 1]"""


def hipgone_main(record=False, hp_min=7.0, hp_max=11.0, radius_arcsec=10.0, near_arcsec=30.0, dmag=1.0):
    rows = core.tap_async("gaia", "SELECT h.hip, h.ra, h.dec, h.pm_ra, h.pm_de, h.plx, h.e_plx, h.hp_mag, h.b_v FROM public.hipparcos_newreduction AS h "
                                  "LEFT OUTER JOIN gaiadr3.hipparcos2_best_neighbour AS x ON h.hip = x.original_ext_source_id WHERE x.source_id IS NULL "
                                  "AND h.hp_mag BETWEEN %g AND %g" % (hp_min, hp_max), maxrec=50000)
    up = []
    for r in rows:
        ra, de = core.star_at(float(r["ra"]), float(r["dec"]), num(r["pm_ra"], 0.0), num(r["pm_de"], 0.0), 1991.25, 2016.0)
        up.append(dict(source_id="h%s" % r["hip"], ra=ra, dec=de))
    found = {}
    for k in range(0, len(up), 5000):
        for m in core.xmatch(up[k:k + 5000], "I/355/gaiadr3", radius_arcsec):
            if m["source_id"] not in found or float(m["angDist"]) < found[m["source_id"]]:
                found[m["source_id"]] = float(m["angDist"])
    gone = [r for r in rows if "h%s" % r["hip"] not in found]
    names = {}
    for k in range(0, len(gone), 200):
        ids = ",".join("'HIP %s'" % r["hip"] for r in gone[k:k + 200])
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type, b.oid, b.ra, b.dec FROM ident AS i JOIN basic AS b ON i.oidref = b.oid "
                             "WHERE i.id IN (%s)" % ids):
            names[str(s["id"]).split()[-1]] = dict(main=" ".join(str(s["main_id"]).split()), otype=str(s["otype"]).strip(), sp=s["sp_type"], oid=s["oid"],
                                                   ra=num(s["ra"]), dec=num(s["dec"]))
    gaia_of = {}
    oids = [str(v["oid"]) for v in names.values()]
    for k in range(0, len(oids), 200):
        for s in core.simbad("SELECT oidref, id FROM ident WHERE oidref IN (%s) AND id LIKE 'Gaia DR3 %%'" % ",".join(oids[k:k + 200])):
            gaia_of[str(s["oidref"])] = str(s["id"]).split()[-1]
    out = []
    for r in gone:
        n = names.get(str(r["hip"]), {})
        o = dict(hip=r["hip"], hp=num(r["hp_mag"]), plx=num(r["plx"]), e_plx=num(r["e_plx"]) if "e_plx" in r else None, main=n.get("main"),
                 otype=n.get("otype"), sp=n.get("sp"), gaia=gaia_of.get(str(n.get("oid"))), near=[])
        # where the Hipparcos motion misses, Gaia may still hold the star at SIMBAD's position
        if not o["gaia"] and n.get("ra") is not None:
            o["near"] = [g["source_id"] for g in core.tap("gaia", "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1 = CONTAINS("
                                                                  "POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f)) AND phot_g_mean_mag BETWEEN %.2f AND %.2f"
                                                                  % (n["ra"], n["dec"], near_arcsec / 3600.0, o["hp"] - dmag, o["hp"] + dmag))]
        out.append(o)
    with_gaia = [o for o in out if o["gaia"]]
    print("%d Hipparcos stars at Hp %g-%g without a Gaia DR3 best neighbour; a Gaia source within %g arcsec of the carried position for %d; none "
          "for %d, %d of which SIMBAD gives a Gaia DR3 identifier and %d of the rest a Gaia source within %g arcsec and %g mag at SIMBAD's position" % (
              len(rows), hp_min, hp_max, radius_arcsec, len(found), len(out), len(with_gaia), near_arcsec, dmag, sum(1 for o in out if o["near"])))
    for o in out[:40]:
        print("  HIP %s: Hp %.2f, parallax %s mas; SIMBAD %s (%s, %s), Gaia DR3 %s%s" % (o["hip"], o["hp"], o["plx"], o["main"], o["otype"], o["sp"],
                                                                                    o["gaia"], "; at SIMBAD's position %s" % o["near"] if o["near"] else ""))
    json.dump(dict(n=len(rows), matched=len(found), gone=out), open(core.results("hipparcos_gone.json"), "w"), indent=1)
    if record:
        core.record_run(
            "hipparcos_gone", "Hipparcos stars Gaia cannot find", "Hipparcos new reduction and Gaia DR3 (hipparcos2_best_neighbour; I/355/gaiadr3 by "
            "CDS XMatch); SIMBAD", "stars at Hp %g-%g without a Gaia DR3 best neighbour: %d" % (hp_min, hp_max, len(rows)), os.path.abspath(__file__),
            "hipgone", len(rows), "stars", dict(hp=[hp_min, hp_max], radius_arcsec=radius_arcsec, near_arcsec=near_arcsec, dmag=dmag, epochs=[1991.25, 2016.0]),
            "the Gaia DR3 sources within %g arcsec of each carried position" % radius_arcsec,
            "a star Hipparcos measured with no Gaia source where its motion takes it", "%d with no Gaia source within %g arcsec, %d of them with a "
            "Gaia DR3 identifier in SIMBAD" % (len(out), radius_arcsec, len(with_gaia)),
            [dict(target="hip:%s" % o["hip"], stat=o["hp"], unit="Hp mag", vet="artefact" if o["gaia"] else "astrometry" if o["near"] else None,
                  note="SIMBAD %s (%s, %s); Gaia DR3 %s" % (o["main"], o["otype"], o["sp"], o["gaia"] or ", ".join(o["near"]) or "none")) for o in out])
        print("recorded")


uniquevar_DOC = r"""Unique variables next door: GCVS stars of the unique type ('*', outside every class of the catalogue's scheme), carried from 2000.0 to 2016.0 with the GCVS proper motions and matched to Gaia DR3 within 10 arcsec by CDS XMatch, kept within 100 pc; a star near enough to be watched closely that varies as no class of variable does, each read in SIMBAD for its type and spectral type.

    python -I seti.py dimming uniquevar [record] [--var-type "*"] [--radius-arcsec 10] [--plx-min 10]"""

# SIMBAD types that name no class of variability
UNIQUEVAR_GENERIC = ("*", "V*", "PM*", "HV*")
# published accounts of unique variables whose SIMBAD type names no class
UNIQUEVAR_PUBLISHED = {"UU CrB": ("flare", "a superflare on 1980 May 21 while the star served as a comparison star (Schaefer, King & Deliyannis 2000)")}


def uniquevar_main(record=False, var_type="*", radius_arcsec=10.0, plx_min=10.0):
    rows = core.tap("vizier", "SELECT GCVS, RAJ2000, DEJ2000, VarType, magMax, Min1, flt, Period, SpType, pmRA, pmDE FROM \"B/gcvs/gcvs_cat\" "
                              "WHERE VarType LIKE '%%%s%%'" % var_type)
    up, by = [], {}
    for r in rows:
        if num(r["RAJ2000"]) is None:
            continue
        star, con = r["GCVS"].split()
        name = "V%d %s" % (int(star[1:]), con) if star[1:].isdigit() else "%s %s" % (star, con)
        ra, de = core.star_at(float(r["RAJ2000"]), float(r["DEJ2000"]), 1000 * num(r["pmRA"], 0.0), 1000 * num(r["pmDE"], 0.0), 2000.0, 2016.0)
        up.append(dict(source_id=name, ra=ra % 360, dec=de))
        by[name] = r
    best = {}
    for m in core.xmatch(up, "I/355/gaiadr3", radius_arcsec):
        if m["source_id"] not in best or float(m["angDist"]) < float(best[m["source_id"]]["angDist"]):
            best[m["source_id"]] = m
    near = sorted((k for k, m in best.items() if num(m["Plx"], 0.0) > plx_min), key=lambda k: -float(best[k]["Plx"]))
    simbad = {}
    for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)"
                         % ",".join("'Gaia DR3 %s'" % best[k]["Source"] for k in near)):
        simbad[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    out = []
    for k in near:
        r, m = by[k], best[k]
        main, otype, sp = simbad.get(m["Source"], (None, None, None))
        vet = "natural-type" if otype and otype not in UNIQUEVAR_GENERIC else UNIQUEVAR_PUBLISHED.get(k, (None, None))[0]
        out.append(dict(name=k, gaia=m["Source"], plx=float(m["Plx"]), G=num(m["Gmag"]), period=num(r["Period"]), amp="%s-%s %s" % (r["magMax"], r["Min1"], r["flt"].strip()),
                        sp=r["SpType"].strip(), main=main, otype=otype, simbad_sp=sp, vet=vet, why=UNIQUEVAR_PUBLISHED.get(k, (None, None))[1]))
    print("%d GCVS stars of type %s; %d matched in Gaia DR3; %d within %.0f pc:" % (len(rows), var_type, len(best), len(out), 1000 / plx_min))
    for o in out:
        print("  %s (Gaia DR3 %s): parallax %.1f mas, G %.2f, period %s d, %s, GCVS %s; SIMBAD %s (%s, %s) %s" % (
            o["name"], o["gaia"], o["plx"], o["G"], o["period"], o["amp"], o["sp"], o["main"], o["otype"], o["simbad_sp"], o["vet"]))
    if record:
        core.record_run(
            "gcvs_unique_near", "Unique variables next door", "GCVS (VizieR B/gcvs); Gaia DR3 (I/355/gaiadr3) by CDS XMatch; SIMBAD",
            "GCVS stars of type %s: %d" % (var_type, len(rows)), os.path.abspath(__file__), "uniquevar", len(rows), "stars",
            dict(type=var_type, radius_arcsec=radius_arcsec, parallax_mas=plx_min, epochs=[2000.0, 2016.0]), "none: each star read in SIMBAD",
            "a variable within %.0f pc that no class holds" % (1000 / plx_min), "%d within %.0f pc, %d without a specific SIMBAD type" % (
                len(out), 1000 / plx_min, sum(1 for o in out if not o["vet"])),
            [dict(target="gcvs:%s" % o["name"], stat=o["plx"], unit="parallax mas", vet=o["vet"],
                  note="Gaia DR3 %s; period %s d; GCVS %s; SIMBAD %s (%s, %s)%s" % (o["gaia"], o["period"], o["sp"], o["main"], o["otype"], o["simbad_sp"],
                                                                         "; " + o["why"] if o["why"] else "")) for o in out])
        print("recorded")


greysub_DOC = r"""Stars dimmed grey on thin-disc orbits: Gaia Catalogue of Nearby Stars objects (probability above 0.9, white-dwarf probability below 0.1) at BP-RP 0.6-3.0 with parallaxes at 20 sigma, RUWE below 1.4, one IPD peak and BP/RP excess within 0.1 of the single-star relation, lying more than 1.5 mag below the main-sequence locus of their colour while crossing the sky at under 50 km/s; a swarm covering three quarters of a star dims it without changing its colour, and the metal-poor subdwarfs that sit there naturally cross it at halo speeds. Each read in SIMBAD for its type, in GSP-Phot for a metallicity whose upper bound lies below -0.5, against the J-Ks of main-sequence stars of its colour, which collision-induced H2 absorption makes bluer in a metal-poor dwarf and a blue companion's light makes redder for the BP-RP, and in GALEX GUVcat AIS within 5 arcsec at 2007.0, where an NUV within 5 mag of G is a hot companion's.

    python -I seti.py dimming greysub [record] [--below-mag 1.5] [--tangential-kms 50] [--colour-min 0.6] [--colour-max 3.0] [--excess-max 0.1] [--uv-dmag 5]"""


def greysub_main(record=False, below_mag=1.5, tangential_kms=50.0, colour_min=0.6, colour_max=3.0, excess_max=0.1, uv_dmag=5.0):
    rows = core.gaia_by_range("SELECT g.source_id, g.parallax, g.parallax_error, g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.phot_bp_rp_excess_factor, "
                              "g.ruwe, g.ipd_frac_multi_peak, g.ra, g.dec, g.pmra, g.pmdec, g.j_m_2mass, g.j_msig_2mass, g.k_m_2mass, g.k_msig_2mass, a.mh_gspphot, "
                              "a.mh_gspphot_upper FROM external.gaiaedr3_gcns_main_1 AS g LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS a ON g.source_id = a.source_id "
                              "WHERE g.source_id BETWEEN {lo} AND {hi} AND g.gcns_prob > 0.9 AND g.wd_prob < 0.1 AND g.ruwe < 1.4 AND g.ipd_frac_multi_peak <= 2 "
                              "AND g.parallax > 20 * g.parallax_error", n=12, maxrec=400000)
    st = []
    for r in rows:
        g, bp, rp = num(r["phot_g_mean_mag"]), num(r["phot_bp_mean_mag"]), num(r["phot_rp_mean_mag"])
        if None in (g, bp, rp) or not colour_min <= bp - rp <= colour_max:
            continue
        c = bp - rp
        # the single-star BP/RP excess (Riello et al. 2021) for the colour
        ex = num(r["phot_bp_rp_excess_factor"]) - (1.154360 + 0.033772 * c + 0.032277 * c * c if c < 0.5 else 1.162004 + 0.011464 * c + 0.049255 * c * c - 0.005879 * c ** 3)
        speed = 4.74047 * math.hypot(float(r["pmra"]), float(r["pmdec"])) / float(r["parallax"])
        j, k, ej, ek = num(r["j_m_2mass"]), num(r["k_m_2mass"]), num(r["j_msig_2mass"]), num(r["k_msig_2mass"])
        jk = j - k if None not in (j, k, ej, ek) and ej < 0.05 and ek < 0.05 else None
        st.append(dict(id=r["source_id"], c=c, mg=g + 5 * math.log10(float(r["parallax"]) / 100), ex=ex, speed=speed, g=g, jk=jk, ra=float(r["ra"]),
                       dec=float(r["dec"]), pmra=float(r["pmra"]), pmdec=float(r["pmdec"]), mh=num(r["mh_gspphot"]), mh_up=num(r["mh_gspphot_upper"])))
    c = np.array([s["c"] for s in st])
    mg = np.array([s["mg"] for s in st])
    bins = np.floor(c / 0.05).astype(int)
    loc = {b: float(np.median(mg[bins == b])) for b in set(bins.tolist()) if np.sum(bins == b) >= 30}
    low = [s for s, b in zip(st, bins) if b in loc and s["mg"] - loc[b] > below_mag and abs(s["ex"]) < excess_max]
    # J-Ks against the locus of the main-sequence stars of the same colour: collision-induced H2 absorption makes a metal-poor dwarf bluer
    ms = [s for s, b in zip(st, bins) if b in loc and abs(s["mg"] - loc[b]) < 0.5 and s["jk"] is not None]
    mb = np.floor(np.array([s["c"] for s in ms]) / 0.1).astype(int)
    jkl = {}
    for b in set(mb.tolist()):
        v = np.array([s["jk"] for s, x in zip(ms, mb) if x == b])
        if len(v) >= 30:
            med = float(np.median(v))
            jkl[b] = (med, 1.4826 * float(np.median(np.abs(v - med))))
    for s in low:
        b = int(math.floor(s["c"] / 0.1))
        s["zjk"] = (s["jk"] - jkl[b][0]) / jkl[b][1] if s["jk"] is not None and b in jkl else None
    disc = [s for s in low if s["speed"] < tangential_kms]
    halo = [s for s in low if s["speed"] >= tangential_kms]
    zmed = lambda grp: float(np.median([s["zjk"] for s in grp if s["zjk"] is not None])) if any(s["zjk"] is not None for s in grp) else float("nan")
    sim = {}
    for k in range(0, len(disc), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in disc[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in disc:
        s["simbad"] = sim.get(s["id"])
        typ = (s["simbad"] or (None, "", ""))[1]
        sp = (s["simbad"] or (None, "", ""))[2]
        s["vet"] = ("white-dwarf" if typ in ("WD*", "WD?", "CV*", "No*") or "D" in sp[:2] else "metal-poor" if sp.startswith(("sd", "esd", "usd")) or typ == "sdM"
                    else "natural-type" if typ not in ("", "*", "PM*", "HV*", "low-mass*") else
                    "metal-poor" if (s["zjk"] is not None and s["zjk"] < -2) or (s["mh_up"] is not None and s["mh_up"] < -0.5) else
                    # J-Ks redder than the main sequence's at the star's BP-RP: a blue companion drags BP-RP blueward of the cool star's own
                    "white-dwarf" if s["zjk"] is not None and s["zjk"] > 3 else None)
    # a hot companion: ultraviolet within 5 mag of G, which an M dwarf's photosphere or chromosphere does not give
    rest = [s for s in disc if s["vet"] is None]
    if rest:
        up = []
        for s in rest:
            ra, de = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, 2007.0)
            up.append(dict(source_id=s["id"], ra=ra % 360, dec=de))
        uv = {}
        for m in core.xmatch(up, "II/335/galex_ais", 5.0):
            nuv = num(m.get("NUVmag"))
            if nuv is not None and (m["source_id"] not in uv or nuv < uv[m["source_id"]]):
                uv[m["source_id"]] = nuv
        for s in rest:
            s["nuv"] = uv.get(s["id"])
            if s["nuv"] is not None and s["nuv"] - s["g"] < uv_dmag:
                s["vet"] = "white-dwarf"
    left = [s for s in disc if s["vet"] is None]
    print("%d stars at BP-RP %g-%g; %d more than %g mag below the locus with clean photometry: %d slower than %g km/s across the sky, %d faster; median J-Ks "
          "against the main sequence %.1f widths for the slow and %.1f for the fast; unexplained %d:" % (
              len(st), colour_min, colour_max, len(low), below_mag, len(disc), tangential_kms, len(halo), zmed(disc), zmed(halo), len(left)))
    for s in sorted(disc, key=lambda s: -(s["mg"])):
        print("  Gaia DR3 %s: BP-RP %.2f, M_G %.2f, %.0f km/s, G %.1f, J-Ks %s widths, [M/H] %s, NUV %s; SIMBAD %s; %s" % (s["id"], s["c"], s["mg"], s["speed"],
              s["g"], "%.1f" % s["zjk"] if s["zjk"] is not None else "-", s["mh"], s.get("nuv"), s["simbad"], s["vet"]))
    if record:
        core.record_run("grey_dimmed_disc_stars", "Stars dimmed grey on thin-disc orbits", "Gaia Catalogue of Nearby Stars (external.gaiaedr3_gcns_main_1); SIMBAD",
                        "stars within 100 pc at BP-RP %g-%g with clean astrometry and photometry: %d" % (colour_min, colour_max, len(st)), os.path.abspath(__file__),
                        "greysub", len(st), "stars", dict(below_mag=below_mag, tangential_kms=tangential_kms, colour=[colour_min, colour_max], excess=excess_max, uv_dmag=uv_dmag),
                        "the same stars moving faster than %g km/s: %d" % (tangential_kms, len(halo)),
                        "a star %g mag fainter than its colour allows on a thin-disc orbit" % below_mag, "%d slower than %g km/s against %d faster, median J-Ks %.1f and %.1f "
                        "widths from the main sequence; %d unexplained" % (len(disc), tangential_kms, len(halo), zmed(disc), zmed(halo), len(left)),
                        [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["mg"] - loc[int(math.floor(s["c"] / 0.05))], 2), unit="mag below the locus", vet=s["vet"],
                              note="BP-RP %.2f, %.0f km/s, J-Ks %s widths from the main sequence, GSP-Phot [M/H] %s, NUV %s; SIMBAD %s" % (
                                  s["c"], s["speed"], "%.1f" % s["zjk"] if s["zjk"] is not None else "unmeasured", s["mh"], s.get("nuv"), s["simbad"])) for s in disc])
        print("recorded")


dr2gone_DOC = r"""Stars Gaia DR2 saw that DR3 cannot find: Gaia DR2 sources with parallaxes above 10 mas at 10 sigma and G below 17 that have no row in gaiadr3.dr2_neighbourhood, which lists every DR3 source near each DR2 source with the DR2 proper motion carried forward; a star that stopped shining between the two releases' data spans. Each is sought again in DR3 within 5 arcsec and 1 mag of its position carried to 2016.0, and the rest are read for DR2's duplicated-source flag and excess-noise significance and for a DR3 source within 10 arcsec more than a magnitude brighter, in whose image DR2's detection lay; the scan's candidates are kept in the results directory, so a rerun skips it.

    python -I seti.py dimming dr2gone [record] [--plx-min 10] [--plx-sigma 10] [--g-max 17] [--radius-arcsec 5] [--bright-arcsec 10]"""

dr2new_DOC = r"""Stars Gaia DR3 sees that DR2 did not: Gaia DR3 sources with parallaxes above 10 mas at 10 sigma, G below 17 and RUWE below 1.4 that have no row in gaiadr3.dr2_neighbourhood; a star that began shining between the releases. Each is sought in DR2 within 5 arcsec and 1 mag of its position carried back to 2015.5, and read for a DR3 neighbour within 2 arcsec, for a DR2 source within 10 arcsec more than a magnitude brighter (DR2 catalogued no second source inside a bright star's image), for a proper motion above 500 mas/yr (DR2's cross-match lost the fastest movers), for a two-parameter or duplicated solution, and for fewer than 13 visibility periods (too few in DR2's 22 months); the scan's candidates are kept in the results directory.

    python -I seti.py dimming dr2new [record] [--plx-min 10] [--plx-sigma 10] [--g-max 17] [--ruwe-max 1.4] [--radius-arcsec 5] [--bright-arcsec 10] [--fast-mas-yr 500] [--min-visibility 13]"""


def dr2_candidates(name, scan):
    """A release's sources with no counterpart in the other, from NAME in the results directory when a scan was kept there, else from SCAN(), whose count and rows are then kept."""
    path = core.results(name)
    if os.path.exists(path):
        d = json.load(open(path))
        return d["n_scanned"], d["rows"]
    n, rows = scan()
    json.dump(dict(n_scanned=n, rows=rows), open(path, "w"), indent=1)
    return n, rows


def dr2_neighbours(release, ra, dec, radius_arcsec, exclude=None):
    """Sources of a Gaia release (gaiadr2 or gaiadr3) within RADIUS_ARCSEC of a position, other than EXCLUDE: id, G, parallax and separation in arcsec."""
    rows = core.tap("gaia", "SELECT source_id, phot_g_mean_mag, parallax, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', %.6f, %.6f)) * 3600 AS sep "
                            "FROM %s.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f))"
                            % (ra, dec, release, ra, dec, radius_arcsec / 3600.0))
    return [dict(id=r["source_id"], g=num(r["phot_g_mean_mag"]), plx=num(r["parallax"]), sep=float(r["sep"])) for r in rows if r["source_id"] != exclude]


def dr2_same(near, g, radius_arcsec):
    """The other release's sources within RADIUS_ARCSEC and a magnitude of G: the star itself, placed differently."""
    return [b for b in near if b["sep"] < radius_arcsec and b["g"] is not None and abs(b["g"] - g) < 1]


def dr2_brighter(near, g):
    """The other release's sources more than a magnitude brighter than G, inside whose images a faint neighbour goes uncatalogued."""
    return [b for b in near if b["g"] is not None and b["g"] < g - 1]


def dr2gone_main(record=False, plx_min=10.0, plx_sigma=10.0, g_max=17.0, radius_arcsec=5.0, bright_arcsec=10.0):
    def scan():
        near = core.gaia_by_range("SELECT source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, phot_g_mean_mag, bp_rp, astrometric_excess_noise_sig, "
                                  "duplicated_source, astrometric_params_solved FROM gaiadr2.gaia_source WHERE source_id BETWEEN {lo} AND {hi} AND parallax > %g "
                                  "AND parallax_over_error > %g AND phot_g_mean_mag < %g" % (plx_min, plx_sigma, g_max), n=192, maxrec=50000)
        # the neighbourhood table lists every DR3 source near each DR2 source; a DR2 source with no row has nothing near it in DR3
        known = core.tap_by_ids("gaia", [r["source_id"] for r in near], "SELECT DISTINCT dr2_source_id AS source_id FROM gaiadr3.dr2_neighbourhood WHERE dr2_source_id IN (%s)")
        return len(near), [r for r in near if r["source_id"] not in known]
    n, rows = dr2_candidates("dr2gone_candidates.json", scan)
    print("%d DR2 sources within 100 pc at G below %g, %d without a DR3 neighbour" % (n, g_max, len(rows)), flush=True)
    out = []
    for r in rows:
        ra, de = core.star_at(float(r["ra"]), float(r["dec"]), num(r["pmra"]) or 0.0, num(r["pmdec"]) or 0.0, 2015.5, 2016.0)
        g = float(r["phot_g_mean_mag"])
        near = dr2_neighbours("gaiadr3", ra, de, bright_arcsec)
        same, bright = dr2_same(near, g, radius_arcsec), dr2_brighter(near, g)
        dup = str(r["duplicated_source"]).lower() in ("true", "1")
        noisy = (num(r["astrometric_excess_noise_sig"]) or 0) > 2
        if same:
            vet, why = "astrometry", "in DR3 within %g arcsec and a magnitude" % radius_arcsec
        elif bright:
            vet, why = "catalogue-dropped", "%.1f arcsec from a DR3 source %.1f mag brighter, which DR3 holds as one source with it" % (bright[0]["sep"], g - bright[0]["g"])
        elif dup or noisy:
            vet, why = "artefact", "a DR2 solution marked %s" % ("duplicated" if dup else "noisy")
        else:
            vet, why = None, "nothing in DR3 within %g arcsec at any magnitude" % bright_arcsec
        out.append(dict(id=r["source_id"], g=g, plx=float(r["parallax"]), bprp=num(r["bp_rp"]), vet=vet, why=why))
    left = [o for o in out if o["vet"] is None]
    print("%d DR2 sources without a DR3 neighbour: %d found again in DR3, %d inside a brighter DR3 source's image, %d DR2 artefacts, %d unexplained %s" % (
        len(out), sum(1 for o in out if o["vet"] == "astrometry"), sum(1 for o in out if o["vet"] == "catalogue-dropped"),
        sum(1 for o in out if o["vet"] == "artefact"), len(left), [(o["id"], o["g"], o["plx"], o["bprp"]) for o in left]))
    for o in out:
        print("  Gaia DR2 %s: G %.2f, parallax %.1f mas; %s" % (o["id"], o["g"], o["plx"], o["why"]))
    if record:
        core.record_run("gaia_dr2_vanished", "Stars Gaia DR2 saw that DR3 cannot find", "Gaia DR2 and DR3 (gaiadr3.dr2_neighbourhood)",
                        "DR2 sources with parallax above %g mas at %g sigma and G below %g: those without a DR3 neighbour, %d of %d" % (plx_min, plx_sigma, g_max, len(out), n),
                        os.path.abspath(__file__), "dr2gone", len(out), "Gaia sources",
                        dict(parallax_mas=plx_min, sigma=plx_sigma, g=g_max, radius_arcsec=radius_arcsec, bright_arcsec=bright_arcsec),
                        "none: each source sought again in DR3", "a nearby star of 2014-2016 with no light in 2014-2017", "%d unexplained of %d" % (len(left), len(out)),
                        [dict(target="gaia_dr2:%s" % o["id"], stat=o["g"], unit="G mag", vet=o["vet"], note="parallax %.1f mas, BP-RP %s; %s" % (o["plx"], o["bprp"], o["why"]))
                         for o in out])
        print("recorded")


def dr2new_main(record=False, plx_min=10.0, plx_sigma=10.0, g_max=17.0, ruwe_max=1.4, radius_arcsec=5.0, bright_arcsec=10.0, fast_mas_yr=500.0, min_visibility=13):
    def scan():
        near = core.gaia_by_range("SELECT source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, phot_g_mean_mag, bp_rp, ruwe, ipd_frac_multi_peak, "
                                  "astrometric_params_solved, duplicated_source, visibility_periods_used FROM gaiadr3.gaia_source WHERE source_id BETWEEN {lo} AND {hi} "
                                  "AND parallax > %g AND parallax_over_error > %g AND phot_g_mean_mag < %g AND ruwe < %g" % (plx_min, plx_sigma, g_max, ruwe_max),
                                  n=192, maxrec=50000)
        known = core.tap_by_ids("gaia", [r["source_id"] for r in near], "SELECT DISTINCT dr3_source_id AS source_id FROM gaiadr3.dr2_neighbourhood WHERE dr3_source_id IN (%s)")
        return len(near), [r for r in near if r["source_id"] not in known]
    n, rows = dr2_candidates("dr2new_candidates.json", scan)
    print("%d DR3 sources within 100 pc at G below %g with RUWE below %g, %d without a DR2 neighbour" % (n, g_max, ruwe_max, len(rows)), flush=True)
    out = []
    for r in rows:
        ra, de = core.star_at(float(r["ra"]), float(r["dec"]), num(r["pmra"]) or 0.0, num(r["pmdec"]) or 0.0, 2016.0, 2015.5)
        g = float(r["phot_g_mean_mag"])
        pm = math.hypot(num(r["pmra"]) or 0.0, num(r["pmdec"]) or 0.0)
        before = dr2_neighbours("gaiadr2", ra, de, bright_arcsec)
        same, bright = dr2_same(before, g, radius_arcsec), dr2_brighter(before, g)
        beside = dr2_neighbours("gaiadr3", float(r["ra"]), float(r["dec"]), 2.0, exclude=r["source_id"])
        two_par = str(r.get("astrometric_params_solved")) == "31"
        dup = str(r.get("duplicated_source")).lower() in ("true", "1")
        vis = int(num(r.get("visibility_periods_used")) or 0)
        if same:
            vet, why = "astrometry", "in DR2 within %g arcsec and a magnitude" % radius_arcsec
        elif beside:
            vet, why = "blend", "a DR3 source %.1f arcsec away at G %s" % (beside[0]["sep"], beside[0]["g"])
        elif bright:
            vet, why = "catalogue-dropped", "%.1f arcsec from a DR2 source %.1f mag brighter, inside whose image DR2 held no second source" % (bright[0]["sep"], g - bright[0]["g"])
        elif pm > fast_mas_yr:
            vet, why = "catalogue-dropped", "moving at %.0f mas/yr, beyond what DR2's cross-match followed" % pm
        elif two_par or dup:
            vet, why = "astrometry", "a %s DR3 solution" % ("duplicated" if dup else "two-parameter")
        elif vis < min_visibility:
            vet, why = "catalogue-dropped", "%d visibility periods in DR3's 34 months, too few for DR2's 22" % vis
        else:
            vet, why = None, "nothing in DR2 within %g arcsec within a magnitude or brighter; %d visibility periods, %.0f mas/yr" % (bright_arcsec, vis, pm)
        out.append(dict(id=r["source_id"], g=g, plx=float(r["parallax"]), bprp=num(r["bp_rp"]), vet=vet, why=why))
    left = [o for o in out if o["vet"] is None]
    counts = {v: sum(1 for o in out if o["vet"] == v) for v in ("astrometry", "blend", "catalogue-dropped")}
    print("%d DR3 sources without a DR2 neighbour: %d present in DR2, %d with a DR3 neighbour within 2 arcsec, %d dropped by DR2 (a bright neighbour, a fast mover "
          "or too few transits), %d unexplained %s" % (len(out), counts["astrometry"], counts["blend"], counts["catalogue-dropped"], len(left),
                                                        [(o["id"], o["g"], o["plx"], o["bprp"], o["why"]) for o in left]))
    if record:
        core.record_run("gaia_dr3_appeared", "Stars Gaia DR3 sees that DR2 did not", "Gaia DR2 and DR3 (gaiadr3.dr2_neighbourhood)",
                        "DR3 sources with parallax above %g mas at %g sigma, G below %g and RUWE below %g: those without a DR2 neighbour, %d of %d" % (
                            plx_min, plx_sigma, g_max, ruwe_max, len(out), n), os.path.abspath(__file__), "dr2new", len(out), "Gaia sources",
                        dict(parallax_mas=plx_min, sigma=plx_sigma, g=g_max, ruwe=ruwe_max, radius_arcsec=radius_arcsec, bright_arcsec=bright_arcsec,
                             fast_mas_yr=fast_mas_yr, min_visibility=min_visibility),
                        "none: each source sought in DR2", "a nearby star of 2014-2017 with no light in 2014-2016", "%d unexplained of %d" % (len(left), len(out)),
                        [dict(target="gaia_dr3:%s" % o["id"], stat=o["g"], unit="G mag", vet=o["vet"], note="parallax %.1f mas, BP-RP %s; %s" % (o["plx"], o["bprp"], o["why"]))
                         for o in out])
        print("recorded")


releasestep_DOC = r"""Stars that stepped a magnitude between Gaia releases: Gaia DR3 sources within 100 pc (parallax above 10 mas at 10 sigma) whose DR2 counterpart in gaiadr3.dr2_neighbourhood, within 0.1 arcsec, carries a mean G differing by more than a magnitude; DR2's 22 months lie inside DR3's 34, so a mean that moved a magnitude is a star that changed its light by a large factor during 2016-2017, grey across the whole G band. Each read for DR3's variability flag, in SIMBAD for its type and against the DR2 and DR3 transit counts, since a mean built on a handful of transits is noise; DR2's photometry of stars brighter than G 4 is saturated, and a source without a BP-RP colour sits in a bright neighbour's light, which DR2's window took in.

    python -I seti.py dimming releasestep [record] [--dmag 1.0] [--plx-min 10] [--match-mas 100] [--min-obs 20] [--bright-g 4]"""


def releasestep_main(record=False, dmag=1.0, plx_min=10.0, match_mas=100.0, min_obs=20, bright_g=4.0):
    rows = core.tap_async("gaia", "SELECT n.dr2_source_id, n.dr3_source_id, n.magnitude_difference, n.angular_distance, g.parallax, g.phot_g_mean_mag, g.phot_g_n_obs, "
                                  "g.phot_variable_flag, g.ra, g.dec, g.bp_rp, g.ruwe FROM gaiadr3.dr2_neighbourhood AS n JOIN gaiadr3.gaia_source AS g "
                                  "ON n.dr3_source_id = g.source_id WHERE g.parallax > %g AND g.parallax_over_error > 10 AND n.angular_distance < %g "
                                  "AND ABS(n.magnitude_difference) > %g" % (plx_min, match_mas, dmag), maxrec=100000)
    dr2 = {}
    ids = sorted({str(r["dr2_source_id"]) for r in rows})
    for k in range(0, len(ids), 500):
        for r in core.tap("gaia", "SELECT source_id, phot_g_mean_mag, phot_g_n_obs, phot_variable_flag FROM gaiadr2.gaia_source WHERE source_id IN (%s)" % ",".join(ids[k:k + 500])):
            dr2[str(r["source_id"])] = (num(r["phot_g_mean_mag"]), num(r["phot_g_n_obs"]), str(r["phot_variable_flag"] or ""))
    sim = {}
    d3 = sorted({str(r["dr3_source_id"]) for r in rows})
    for k in range(0, len(d3), 200):
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % x for x in d3[k:k + 200])):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    out = []
    for r in rows:
        g2, n2, v2 = dr2.get(str(r["dr2_source_id"]), (None, None, ""))
        o = dict(dr3=str(r["dr3_source_id"]), dr2=str(r["dr2_source_id"]), dG=num(r["magnitude_difference"]), g3=num(r["phot_g_mean_mag"]), g2=g2, n3=num(r["phot_g_n_obs"]),
                 n2=n2, plx=num(r["parallax"]), bprp=num(r["bp_rp"]), ruwe=num(r["ruwe"]), var3=str(r["phot_variable_flag"] or ""), var2=v2, sep=num(r["angular_distance"]),
                 simbad=sim.get(str(r["dr3_source_id"])))
        typ = (o["simbad"] or (None, "", ""))[1]
        # DR2's photometry of the brightest stars is saturated and wrong; a source without a BP-RP colour sits in a bright neighbour's light, which DR2's window took in
        o["vet"] = ("single-record" if (o["n2"] or 0) < min_obs or (o["n3"] or 0) < min_obs else
                    "variable-star" if o["var3"] == "VARIABLE" or o["var2"] == "VARIABLE" or typ in ("V*", "Mi*", "LP*", "Pu*", "RR*", "dS*", "Ce*", "EB*", "Fl*", "BY*", "RS*",
                                                                                                        "Er*", "TT*", "Y*O", "Or*", "CV*", "No*", "SN*", "Ro*", "El*", "a2*") else
                    "artefact" if (o["g3"] or 99) < bright_g else
                    "crowding" if o["bprp"] is None else
                    "blend" if (o["ruwe"] or 0) > 1.4 else None)
        out.append(o)
    left = [o for o in out if o["vet"] is None]
    count = {v: sum(o["vet"] == v for o in out) for v in ("single-record", "variable-star", "artefact", "crowding", "blend")}
    print("%d stars within %.0f pc whose DR2 mean G differs from DR3's by more than %g mag within %g mas; %d on few transits, %d flagged or typed variable, %d brighter than "
          "G %g where DR2 saturates, %d without a colour in a bright neighbour's light, %d with RUWE above 1.4; unexplained %d" % (
              len(out), 1000 / plx_min, dmag, match_mas, count["single-record"], count["variable-star"], count["artefact"], bright_g, count["crowding"], count["blend"], len(left)))
    for o in sorted(out, key=lambda o: -abs(o["dG"] or 0)):
        print("  Gaia DR3 %s: G %s (%s transits) against DR2 %s (%s transits), difference %+.2f, BP-RP %s, RUWE %s, flags %s/%s; SIMBAD %s; %s" % (
            o["dr3"], o["g3"], o["n3"], o["g2"], o["n2"], o["dG"] or 0, o["bprp"], o["ruwe"], o["var2"], o["var3"], o["simbad"], o["vet"] or "unexplained"))
    json.dump(out, open(core.results("gaia_release_steps.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_release_step", "Stars that stepped a magnitude between Gaia releases", "Gaia DR3 gaia_source and dr2_neighbourhood; Gaia DR2 gaia_source; SIMBAD",
            "stars within %.0f pc whose DR2 and DR3 mean G differ by more than %g mag: %d" % (1000 / plx_min, dmag, len(out)), os.path.abspath(__file__), "releasestep",
            len(out), "stars", dict(dmag=dmag, plx_min=plx_min, match_mas=match_mas, min_obs=min_obs, bright_g=bright_g),
            "the transit counts, the variability flags, SIMBAD's types, DR2's saturation and the missing colours of sources in a bright neighbour's light",
            "a star whose light changed by a magnitude between the releases' spans", "%d: %d variable, %d saturated in DR2, %d in a bright neighbour's light, %d blended; "
            "%d unexplained" % (len(out), count["variable-star"], count["artefact"], count["crowding"], count["blend"], len(left)),
            [dict(target="gaia_dr3:%s" % o["dr3"], stat=o["dG"], unit="DR2 minus DR3 mean G, mag", vet=o["vet"],
                  note="G %s on %s transits against DR2 %s on %s; flags %s/%s; SIMBAD %s" % (o["g3"], o["n3"], o["g2"], o["n2"], o["var2"], o["var3"], o["simbad"])) for o in out])
        print("recorded")


hipdim_DOC = r"""Stars that dimmed since Hipparcos: Hipparcos stars within 100 pc with a Gaia DR3 best neighbour, Gaia's G against Hipparcos's Hp through the sample's own running median of G - Hp in BP-RP, flagged where G is fainter than the median by 0.5 mag, a star that lost a third of its light between 1991 and 2016 without a change of colour recorded, and where it is brighter by as much, the mirror; each read for Hipparcos's scatter and variability, Gaia's variability flag and RUWE, SIMBAD's type and Gaia's saturation at the brightest magnitudes; Hipparcos measured a close pair's joint light where Gaia resolves the components, so a pair's name, a component letter or a Gaia neighbour within 10 arcsec and three magnitudes explains a step of up to 0.75 mag, and an Hp without a listed scatter rests on an incomplete record.

    python -I seti.py dimming hipdim [record] [--dmag 0.5] [--plx-min 10] [--bright-g 3] [--pair-arcsec 10]"""


def hipdim_main(record=False, dmag=0.5, plx_min=10.0, bright_g=3.0, pair_arcsec=10.0):
    import re
    rows = core.tap_async("gaia", "SELECT h.hip, h.hp_mag, h.b_v, h.s_hp, h.var, g.source_id, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.phot_variable_flag, g.parallax, g.ra, g.dec "
                                  "FROM public.hipparcos_newreduction AS h JOIN gaiadr3.hipparcos2_best_neighbour AS x ON h.hip = x.original_ext_source_id "
                                  "JOIN gaiadr3.gaia_source AS g ON x.source_id = g.source_id WHERE g.parallax > %g AND h.hp_mag IS NOT NULL" % plx_min, maxrec=200000)
    st = [dict(hip=r["hip"], hp=num(r["hp_mag"]), bv=num(r["b_v"]), scat=num(r["s_hp"]), hvar=str(r["var"] or "").strip(), id=str(r["source_id"]), g=num(r["phot_g_mean_mag"]),
               c=num(r["bp_rp"]), ruwe=num(r["ruwe"]), gvar=str(r["phot_variable_flag"] or ""), plx=num(r["parallax"]), ra=float(r["ra"]), dec=float(r["dec"])) for r in rows]
    st = [s for s in st if s["g"] is not None and s["c"] is not None and s["hp"] is not None]
    for s in st:
        s["d"] = s["g"] - s["hp"]
    bins = np.floor(np.array([s["c"] for s in st]) / 0.1).astype(int)
    d = np.array([s["d"] for s in st])
    for b in set(bins.tolist()):
        k = np.nonzero(bins == b)[0]
        med = float(np.median(d[k])) if len(k) >= 20 else None
        for i in k:
            st[i]["res"] = st[i]["d"] - med if med is not None else None
    dim = [s for s in st if s["res"] is not None and s["res"] > dmag]
    bright = [s for s in st if s["res"] is not None and s["res"] < -dmag]
    sim = {}
    both = dim + bright
    for k in range(0, len(both), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'HIP %s'" % s["hip"] for s in both[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    var_types = ("V*", "Mi*", "LP*", "Pu*", "RR*", "dS*", "Ce*", "EB*", "Fl*", "BY*", "RS*", "Er*", "TT*", "Y*O", "Or*", "CV*", "No*", "Ro*", "El*", "a2*", "gD*", "bC*", "sr*", "cC*")
    for s in both:
        s["simbad"] = sim.get(str(s["hip"]))
        typ = (s["simbad"] or (None, "", ""))[1]
        main = (s["simbad"] or ("", "", ""))[0] or ""
        # Hipparcos measured a close pair's joint light where Gaia resolves the components: a pair name, a component letter or a neighbour within 10 arcsec
        s["vet"] = ("artefact" if s["g"] < bright_g else
                    "single-record" if not s["scat"] else
                    "variable-star" if s["gvar"] == "VARIABLE" or s["hvar"] in ("P", "U", "M", "R") or (s["scat"] or 0) > 0.1 or typ in var_types else
                    "binary" if (s["ruwe"] or 0) > 1.4 or typ in ("**", "SB*", "EB*") or re.search(r"[A-D]$", main) else None)
    for s in both:
        if s["vet"] is None:
            near = core.tap("gaia", "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %s, %s, %.5f)) "
                                    "AND source_id <> %s AND phot_g_mean_mag < %.2f" % (s["ra"], s["dec"], pair_arcsec / 3600.0, s["id"], s["g"] + 3))
            s["near"] = ["%s (G %.1f)" % (r["source_id"], float(r["phot_g_mean_mag"])) for r in near]
            if s["near"]:
                s["vet"] = "binary"
    left = [s for s in both if s["vet"] is None]
    print("%d Hipparcos stars within %.0f pc with Gaia G; G - Hp against the running median in BP-RP: fainter by %g mag %d, brighter by as much %d; variables %d, "
          "binaries %d, brighter than G %g %d; unexplained %d" % (len(st), 1000 / plx_min, dmag, len(dim), len(bright), sum(s["vet"] == "variable-star" for s in both),
                                                                  sum(s["vet"] == "binary" for s in both), bright_g, sum(s["vet"] == "artefact" for s in both), len(left)))
    for s in sorted(both, key=lambda s: -abs(s["res"])):
        print("  HIP %s (Gaia DR3 %s): G %.2f, Hp %.2f, BP-RP %.2f, G - Hp %+.2f from the median; Hp scatter %s, Hipparcos %s, Gaia %s, RUWE %s; SIMBAD %s; %s" % (
            s["hip"], s["id"], s["g"], s["hp"], s["c"], s["res"], s["scat"], s["hvar"] or "-", s["gvar"], s["ruwe"], s["simbad"], s["vet"] or ("unexplained, %s" % (
                "dimmed" if s["res"] > 0 else "brightened"))))
    json.dump(dict(n=len(st), dim=dim, bright=bright), open(core.results("hipparcos_dimmed.json"), "w"), indent=1)
    if record:
        core.record_run(
            "hipparcos_gaia_dimming", "Stars that dimmed since Hipparcos", "Hipparcos new reduction and Gaia DR3 (hipparcos2_best_neighbour); SIMBAD",
            "Hipparcos stars within %.0f pc with Gaia G: %d" % (1000 / plx_min, len(st)), os.path.abspath(__file__), "hipdim", len(st), "stars",
            dict(dmag=dmag, plx_min=plx_min, bright_g=bright_g, pair_arcsec=pair_arcsec), "the stars brighter in G than the median by as much: %d" % len(bright),
            "a loss of %g mag in G against Hp, grey across the two bands" % dmag,
            "%d fainter against %d brighter; %d variables, %d binaries, %d saturated; %d unexplained" % (
                len(dim), len(bright), sum(s["vet"] == "variable-star" for s in both), sum(s["vet"] == "binary" for s in both), sum(s["vet"] == "artefact" for s in both), len(left)),
            [dict(target="hip:%s" % s["hip"], stat=s["res"], unit="G - Hp less the median at the colour, mag", vet=s["vet"],
                  note="Gaia DR3 %s; G %.2f, Hp %.2f, BP-RP %.2f; Hp scatter %s, Hipparcos %s, Gaia %s, RUWE %s; SIMBAD %s; neighbours %s" % (
                      s["id"], s["g"], s["hp"], s["c"], s["scat"], s["hvar"] or "-", s["gvar"], s["ruwe"], s["simbad"], s.get("near", "-"))) for s in both])
        print("recorded")


colourchange_DOC = r"""Stars that changed colour since Hipparcos: the Hipparcos new reduction (van Leeuwen 2007) matched to Gaia DR3 (hipparcos2_best_neighbour), stars with a B-V measured to 0.03 mag, the Gaia BP-RP of each star against the run of BP-RP with Hipparcos B-V across the whole sample, those more than five robust widths, five times the B-V error carried through the run's slope and 0.2 mag off the run, a star whose colour in 1991 was not its colour in 2016, which no star does on its own and a star under a changing screen or a changing lamp would; each read for its Hipparcos scatter, since a variable star carries a colour of its own epoch, for RUWE, since a binary mixes two colours, for saturation in Gaia's photometry, and for its spectral class, which sides with one colour or the other where the Hipparcos B-V, drawn from many ground sources, is the record that stands alone.

    python -I seti.py dimming colourchange [record] [--widths 5] [--min-mag 0.2] [--bv-min -0.3] [--bv-max 2.0] [--g-sat 6] [--e-bv-max 0.03] [--slope 2] [--class-tol 0.45]"""


# the colours of the spectral classes, B-V and BP-RP, for the arbitration of a disagreement by the SIMBAD type
COLOURCHANGE_CLASS = {"O": (-0.30, -0.45), "B": (-0.15, -0.20), "A": (0.10, 0.15), "F": (0.40, 0.60), "G": (0.65, 0.90), "K": (1.05, 1.35), "M": (1.50, 2.30)}


def colourchange_main(record=False, widths=5.0, min_mag=0.2, bv_min=-0.3, bv_max=2.0, g_sat=6.0, e_bv_max=0.03, slope=2.0, class_tol=0.45):
    rows = core.tap_async("gaia", "SELECT h.hip, h.b_v, h.e_b_v, h.hp_mag, h.s_hp, h.plx, g.source_id, g.bp_rp, g.phot_g_mean_mag, g.ruwe, g.ipd_frac_multi_peak, "
                                  "g.phot_bp_rp_excess_factor FROM public.hipparcos_newreduction AS h JOIN gaiadr3.hipparcos2_best_neighbour AS n ON h.hip = n.original_ext_source_id "
                                  "JOIN gaiadr3.gaia_source AS g ON n.source_id = g.source_id WHERE h.b_v IS NOT NULL AND h.b_v <> 0 AND h.e_b_v IS NOT NULL AND h.e_b_v < %g AND "
                                  "g.bp_rp IS NOT NULL AND h.b_v BETWEEN %g AND %g" % (e_bv_max, bv_min, bv_max), maxrec=200000)
    st = [dict(hip=int(r["hip"]), id=str(r["source_id"]), bv=float(r["b_v"]), ebv=float(r["e_b_v"]), hp=num(r["hp_mag"]), shp=num(r["s_hp"]), plx=num(r["plx"]), c=float(r["bp_rp"]),
               g=num(r["phot_g_mean_mag"]), ruwe=num(r["ruwe"]), multi=num(r["ipd_frac_multi_peak"]), excess=num(r["phot_bp_rp_excess_factor"])) for r in rows]
    bv = np.array([s["bv"] for s in st])
    c = np.array([s["c"] for s in st])
    bins = np.floor(bv / 0.05).astype(int)
    for b in set(bins.tolist()):
        k = np.nonzero(bins == b)[0]
        med = float(np.median(c[k])) if len(k) >= 30 else None
        wid = 1.4826 * float(np.median(np.abs(c[k] - med))) if med is not None else None
        for i in k:
            st[i]["r"] = (float(c[i] - med), wid) if med is not None else None
    off = [s for s in st if s["r"] and abs(s["r"][0]) > max(widths * s["r"][1], min_mag, widths * slope * s["ebv"])]
    sim = {}
    for k in range(0, len(off), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'HIP %d'" % s["hip"] for s in off[k:k + 200])):
            sim[int(str(r["id"]).split()[-1])] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in off:
        s["simbad"] = sim.get(s["hip"])
        typ = (s["simbad"] or (None, "", ""))[1]
        sp = (s["simbad"] or (None, "", ""))[2].lstrip("kgm ").upper()
        cls = COLOURCHANGE_CLASS.get(sp[:1]) if sp else None
        s["class"] = sp[:2] if sp else None
        hip_wrong = cls is not None and abs(s["c"] - cls[1]) < class_tol and abs(s["bv"] - cls[0]) > class_tol
        s["vet"] = ("artefact" if (s["g"] or 99) < g_sat or (s["excess"] or 0) > 1.3 + 0.06 * s["c"] ** 2 else "variable-star" if (s["shp"] or 0) > 0.05 or typ.endswith("*") and typ[:-1] in
                    ("V", "Pu", "dS", "RR", "Ce", "Mi", "LP", "sr", "Er", "RS", "BY", "Ro", "Fl", "Ir", "Or", "EB", "El", "bC", "gD", "RV", "SN", "No", "CV", "XB", "Sy", "Be", "LB", "WR")
                    else "binary" if (s["ruwe"] or 0) > 1.4 or (s["multi"] or 0) > 2 or typ in ("**", "SB*", "EB*") else "single-record" if hip_wrong else None)
    left = [s for s in off if s["vet"] is None]
    print("%d Hipparcos stars with B-V and Gaia BP-RP; off the colour run by %g widths and %g mag: %d (%d saturated or with BP/RP excess, %d variable, %d binaries, %d whose spectral "
          "class sides with Gaia against the Hipparcos B-V); unexplained %d" % (len(st), widths, min_mag, len(off), sum(s["vet"] == "artefact" for s in off),
                                                                              sum(s["vet"] == "variable-star" for s in off), sum(s["vet"] == "binary" for s in off),
                                                                              sum(s["vet"] == "single-record" for s in off), len(left)))
    for s in sorted(off, key=lambda s: -abs(s["r"][0]) / s["r"][1]):
        if s["vet"] is None or len(off) <= 40:
            print("  HIP %d: B-V %.3f +- %.3f, BP-RP %.3f, %+.3f off the run (width %.3f), Hp %s scatter %s, G %s, RUWE %s, multi %s, excess %s; SIMBAD %s; %s" % (
                s["hip"], s["bv"], s["ebv"], s["c"], s["r"][0], s["r"][1], s["hp"], s["shp"], s["g"], s["ruwe"], s["multi"], s["excess"], s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), off=[dict((k, v) for k, v in s.items() if k != "r") | dict(off=s["r"][0], width=s["r"][1]) for s in off]), open(core.results("colour_change_hipparcos.json"), "w"), indent=1)
    if record:
        core.record_run(
            "hipparcos_gaia_colour_change", "Stars that changed colour since Hipparcos", "Hipparcos new reduction (public.hipparcos_newreduction) and Gaia DR3 by hipparcos2_best_neighbour; SIMBAD",
            "Hipparcos stars with B-V between %g and %g to %g mag and a Gaia BP-RP: %d" % (bv_min, bv_max, e_bv_max, len(st)), os.path.abspath(__file__), "colourchange", len(st), "stars",
            dict(widths=widths, min_mag=min_mag, bv=[bv_min, bv_max], g_sat=g_sat, e_bv_max=e_bv_max, slope=slope, class_tol=class_tol),
            "the run of BP-RP with B-V across the sample in bins of 0.05; the B-V errors; Hipparcos scatter, RUWE, SIMBAD types",
            "a BP-RP more than %g widths and %g mag from the run at the star's B-V" % (widths, min_mag),
            "%d: %d saturated or with excess, %d variable, %d binaries, %d with the Hipparcos B-V against the spectral class; %d unexplained" % (
                len(off), sum(s["vet"] == "artefact" for s in off), sum(s["vet"] == "variable-star" for s in off), sum(s["vet"] == "binary" for s in off),
                sum(s["vet"] == "single-record" for s in off), len(left)),
            [dict(target="hip:%d" % s["hip"], stat=round(s["r"][0], 3), unit="mag off the BP-RP run", vet=s["vet"],
                  note="B-V %.3f, BP-RP %.3f, Hp scatter %s, G %s, RUWE %s; SIMBAD %s" % (s["bv"], s["c"], s["shp"], s["g"], s["ruwe"], s["simbad"])) for s in off])
        print("recorded")


irchanged_DOC = r"""Stars whose infrared changed since 2MASS: Gaia Catalogue of Nearby Stars objects (probability above 0.9) brighter than G 15 with 2MASS Ks measured to 0.03 mag and fainter than Ks 4, the G-Ks of each star against the run of G-Ks with BP-RP across the sample, those more than five robust widths and 0.3 mag off the run, a star whose 2 um light in 1998 does not match its optical light in 2016, which a variable star or a blend gives and a changing screen would; each read for Gaia's variability flag and RUWE, for a brighter Gaia neighbour within the 2MASS beam, for its image peaks and in SIMBAD, where white dwarfs, young stars with discs and subdwarfs carry colours of their own off the main run.

    python -I seti.py dimming irchanged [record] [--widths 5] [--min-mag 0.3] [--g-max 15] [--ks-sat 4] [--beam-arcsec 6]"""


def irchanged_main(record=False, widths=5.0, min_mag=0.3, g_max=15.0, ks_sat=4.0, beam_arcsec=6.0):
    from setitools import catalogues
    rows = catalogues.gcns("source_id, ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe, ipd_frac_multi_peak, k_m_2mass, k_msig_2mass",
                           "gcns_prob > 0.9 AND phot_g_mean_mag < %g AND k_msig_2mass < 0.03 AND k_m_2mass > %g" % (g_max, ks_sat), ranges=12)
    st = []
    for r in rows:
        bp, rp, k, g = num(r["phot_bp_mean_mag"]), num(r["phot_rp_mean_mag"]), num(r["k_m_2mass"]), num(r["phot_g_mean_mag"])
        if None in (bp, rp, k, g):
            continue
        st.append(dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), plx=num(r["parallax"]), g=g, c=bp - rp, k=k, gk=g - k, ruwe=num(r["ruwe"]), multi=num(r["ipd_frac_multi_peak"])))
    c = np.array([s["c"] for s in st])
    gk = np.array([s["gk"] for s in st])
    bins = np.floor(c / 0.05).astype(int)
    for b in set(bins.tolist()):
        kk = np.nonzero(bins == b)[0]
        med = float(np.median(gk[kk])) if len(kk) >= 30 else None
        wid = 1.4826 * float(np.median(np.abs(gk[kk] - med))) if med is not None else None
        for i in kk:
            st[i]["r"] = (float(gk[i] - med), wid) if med is not None else None
    off = [s for s in st if s["r"] and abs(s["r"][0]) > max(widths * s["r"][1], min_mag)]
    var, near = {}, {}
    for k in range(0, len(off), 500):
        ids = ",".join(s["id"] for s in off[k:k + 500])
        for r in core.tap("gaia", "SELECT source_id, phot_variable_flag, non_single_star FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ids):
            var[str(r["source_id"])] = (str(r["phot_variable_flag"] or "").strip(), num(r["non_single_star"]))
    gmag = {s["id"]: s["g"] for s in off}
    for k in range(0, len(off), 5000):
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in off[k:k + 5000]], "I/355/gaiadr3", beam_arcsec):
            other, g = str(m.get("Source") or "").strip(), num(m.get("Gmag"))
            if float(m["angDist"]) > 0.5 and other != m["source_id"] and g is not None and g < gmag[m["source_id"]] + 2:
                near.setdefault(m["source_id"], []).append((other, g))
    sim = {}
    for k in range(0, len(off), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in off[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in off:
        s["var"], s["near"], s["simbad"] = var.get(s["id"], ("", 0)), near.get(s["id"], []), sim.get(s["id"])
        typ = (s["simbad"] or (None, "", ""))[1]
        sp = (s["simbad"] or (None, "", ""))[2].lower()
        s["vet"] = ("blend" if s["near"] or (s["multi"] or 0) > 2 else "white-dwarf" if typ == "WD*" or sp.startswith("d") and sp[1:2] in "abcqzox" else
                    "young-star" if typ in ("TT*", "Y*?", "Y*O", "Or*", "pr*") else "metal-poor" if typ == "LM*" or "sd" in sp else
                    "variable-star" if s["var"][0] == "VARIABLE" or (typ.endswith("*") and typ[:-1] in
                    ("V", "Pu", "dS", "RR", "Ce", "Mi", "LP", "sr", "Er", "RS", "BY", "Ro", "Fl", "Ir", "Or", "EB", "El", "bC", "gD", "RV", "CV", "Sy")) else
                    "binary" if (s["ruwe"] or 0) > 1.4 or (s["var"][1] or 0) > 0 or typ in ("**", "SB*", "EB*") else None)
    left = [s for s in off if s["vet"] is None]
    print("%d stars within 100 pc with clean 2MASS Ks and Gaia colours; off the G-Ks run by %g widths and %g mag: %d (%d blended within %g arcsec or with image peaks, %d white dwarfs, "
          "%d young stars, %d subdwarfs, %d variable, %d binaries); unexplained %d" % (len(st), widths, min_mag, len(off), sum(s["vet"] == "blend" for s in off), beam_arcsec,
                                                                                   sum(s["vet"] == "white-dwarf" for s in off), sum(s["vet"] == "young-star" for s in off),
                                                                                   sum(s["vet"] == "metal-poor" for s in off), sum(s["vet"] == "variable-star" for s in off),
                                                                                   sum(s["vet"] == "binary" for s in off), len(left)))
    for s in sorted(off, key=lambda s: -abs(s["r"][0]) / s["r"][1]):
        if s["vet"] is None or len(off) <= 40:
            print("  Gaia DR3 %s: BP-RP %.2f, G %.2f, Ks %.2f, G-Ks %+.2f off the run (width %.3f), RUWE %s, multi %s, variable %s, neighbours %s; SIMBAD %s; %s" % (
                s["id"], s["c"], s["g"], s["k"], s["r"][0], s["r"][1], s["ruwe"], s["multi"], s["var"][0] or "-", s["near"] or "-", s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), off=[dict((k, v) for k, v in s.items() if k != "r") | dict(off=s["r"][0], width=s["r"][1]) for s in off]), open(core.results("ir_change_2mass_gaia.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gcns_2mass_gaia_ir_change", "Stars whose infrared changed since 2MASS", "Gaia Catalogue of Nearby Stars with its 2MASS photometry (external.gaiaedr3_gcns_main_1); Gaia DR3; SIMBAD",
            "stars within 100 pc brighter than G %g with Ks to 0.03 mag fainter than %g: %d" % (g_max, ks_sat, len(st)), os.path.abspath(__file__), "irchanged", len(st), "stars",
            dict(widths=widths, min_mag=min_mag, g_max=g_max, ks_sat=ks_sat, beam_arcsec=beam_arcsec), "the run of G-Ks with BP-RP in bins of 0.05; Gaia's neighbours, variability flag, RUWE and image peaks; SIMBAD",
            "a G-Ks more than %g widths and %g mag from the run at the star's colour" % (widths, min_mag),
            "%d: %d blends, %d white dwarfs, %d young stars, %d subdwarfs, %d variable, %d binaries; %d unexplained" % (
                len(off), sum(s["vet"] == "blend" for s in off), sum(s["vet"] == "white-dwarf" for s in off), sum(s["vet"] == "young-star" for s in off),
                sum(s["vet"] == "metal-poor" for s in off), sum(s["vet"] == "variable-star" for s in off), sum(s["vet"] == "binary" for s in off), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["r"][0], 3), unit="mag off the G-Ks run", vet=s["vet"],
                  note="BP-RP %.2f, G %.2f, Ks %.2f, RUWE %s, variable %s, neighbours %s; SIMBAD %s" % (s["c"], s["g"], s["k"], s["ruwe"], s["var"][0] or "-", s["near"] or "none", s["simbad"]))
             for s in off])
        print("recorded")


candles_DOC = r"""Standard candles that are not standard: the Cepheids of Gaia DR3's vari_cepheid in the Large and Small Magellanic Clouds, at distance moduli of 18.477 (Pietrzyński et al. 2019) and 18.977 (Graczyk et al. 2020), and with parallaxes at ten sigma in the Galaxy, and the RR Lyrae of vari_rrlyrae with such parallaxes, each on the period-Wesenheit relation W = G - 1.90 (BP - RP) of its class, mode and galaxy fitted with three-sigma clipping, where a star more than five widths of the relation (its scatter and the parallax's error together) brighter than its period allows carries light its pulsation does not give; each such star read for a companion in its astrometry (RUWE above 1.4 or a non-single-star solution), for a neighbour's light (a corrected BP/RP excess above 0.2, or a Gaia source within 2 arcsec at most 3 mag fainter), in the Clouds against OGLE-IV's Cepheids within 1 arcsec (Soszyński et al. 2015), whose class, mode and period show where Gaia's differ, whose I-band Wesenheit relation shows whether the star is bright in OGLE's sharper photometry too and whose remarks name eclipsing companions and blended pairs of Cepheids, in the Galaxy against Gaia's general variability classifier, and everywhere for its pulsation amplitude, since light that does not pulsate dilutes it: a star whose G amplitude lies below the 16th percentile of its peers' (the same class, mode and galaxy within 0.05 in log P) and returns within their 98th percentile when scaled by the light it carries in G beyond its group's G-band relation (a red companion brightens W by more than it brightens G) carries a companion's light, as does one whose G amplitude lies below its peers' median while its ratio of BP to RP amplitude lies outside their central 68 per cent, since a companion redder than the pulsator dilutes the RP amplitude more than the BP and a bluer one the reverse.

    python -I seti.py dimming candles [record] [--widths 5] [--clip 3] [--plx-sigma 10] [--min-group 20] [--cstar-max 0.2] [--blend-arcsec 2] [--blend-dmag 3] [--amp-dlogp 0.05] [--amp-low 0.16] [--amp-high 0.98]"""

# centre, radius (deg) and distance modulus of the Magellanic Clouds
CANDLES_CLOUDS = {"LMC": (80.894, -69.756, 8.0, 18.477), "SMC": (13.187, -72.829, 4.0, 18.977)}
# Lindegren et al. (2021)'s global parallax zero point, mas
CANDLES_PLX_ZP = -0.017


def candles_fit(x, y, s, clip):
    """A line fitted to Y against X by least squares with CLIP-sigma clipping, S each point's own error: coefficients, intrinsic scatter and the kept mask."""
    keep = np.ones(len(x), bool)
    for _ in range(10):
        c = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(c, x)
        sig = core.robust_sigma(r[keep])
        new = np.abs(r) < clip * np.sqrt(sig ** 2 + s ** 2)
        if (new == keep).all():
            break
        keep = new
    sig_int = math.sqrt(max(core.robust_sigma(r[keep]) ** 2 - float(np.median(s[keep] ** 2)), 0.0))
    return c, sig_int, keep


CANDLES_OGLE = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/%s/%s/%s"


def candles_ogle():
    """OGLE-IV's Cepheids of the Magellanic Clouds (classical, anomalous and type II): position, class, mode, period, mean I and V and OGLE's remark, by cloud."""
    out = {}
    for cloud in ("LMC", "SMC"):
        stars = {}
        for kind in ("cep", "acep", "t2cep"):
            for line in core.get(CANDLES_OGLE % (cloud.lower(), kind, "ident.dat"), timeout=120).decode("latin-1").splitlines():
                t = line.split()
                if len(t) < 4:
                    continue
                h, m, s = (float(x) for x in t[2].split(":"))
                d, dm, ds = (abs(float(x)) for x in t[3].split(":"))
                dec = (d + dm / 60 + ds / 3600) * (-1 if t[3].startswith("-") else 1)
                stars[t[0]] = dict(id=t[0], kind=kind, mode=t[1], ra=15 * (h + m / 60 + s / 3600), dec=dec)
            files = {"cep": ("cepF.dat", "cep1O.dat"), "acep": ("acepF.dat", "acep1O.dat"), "t2cep": ("t2cep.dat",)}[kind]
            for f in files:
                for line in core.get(CANDLES_OGLE % (cloud.lower(), kind, f), timeout=120).decode("latin-1").splitlines():
                    t = line.split()
                    if len(t) >= 4 and t[0] in stars:
                        stars[t[0]].update(I=num(t[1]) if 0 < (num(t[1]) or 0) < 30 else None, V=num(t[2]) if 0 < (num(t[2]) or 0) < 30 else None, P=num(t[3]))
            remarks = core.get(CANDLES_OGLE % (cloud.lower(), kind, "remarks.txt"), timeout=120, soft=True) or b""
            for line in remarks.decode("latin-1").splitlines():
                t = line.split(None, 1)
                if len(t) == 2 and t[0] in stars:
                    stars[t[0]]["remark"] = t[1].strip()
        out[cloud] = list(stars.values())
    return out


def candles_ogle_read(o, ogle, radius_arcsec=1.0):
    """A Cloud pulsator read in OGLE-IV: its OGLE class, mode and period against Gaia's, and its place on OGLE's own I-band Wesenheit relation for its mode."""
    rows = ogle[o["galaxy"]]
    ra, dec = np.array([r["ra"] for r in rows]), np.array([r["dec"] for r in rows])
    d = np.hypot((ra - o["ra"]) * math.cos(math.radians(o["dec"])), dec - o["dec"]) * 3600
    k = int(np.argmin(d))
    if d[k] > radius_arcsec:
        return dict(verdict="absent")
    m = rows[k]
    gaia_mode = {"FUNDAMENTAL": "F", "FIRST_OVERTONE": "1O"}.get(o["mode"], o["mode"])
    out = dict(id=m["id"], kind=m["kind"], mode=m["mode"], P=m.get("P"), remark=m.get("remark", ""))
    if m["kind"] != {"DCEP": "cep", "ACEP": "acep", "T2CEP": "t2cep"}.get(o["cls"], "?"):
        out["verdict"] = "class"
    elif m["mode"] != gaia_mode:
        out["verdict"] = "mode"
    elif not m.get("P") or abs(o["P"] / m["P"] - 1) > 0.01:
        out["verdict"] = "period"
    else:
        same = [r for r in rows if r["kind"] == m["kind"] and r["mode"] == m["mode"] and r.get("P") and r.get("I") and r.get("V")]
        x = np.array([math.log10(r["P"]) for r in same])
        y = np.array([r["I"] - 1.55 * (r["V"] - r["I"]) for r in same])
        c, sig, keep = candles_fit(x, y, np.zeros(len(x)), 3.0)
        out["w_resid"] = float(m["I"] - 1.55 * (m["V"] - m["I"]) - np.polyval(c, math.log10(m["P"]))) if m.get("I") and m.get("V") else None
        out["verdict"] = "normal" if out["w_resid"] is not None and out["w_resid"] > -5 * sig else "bright"
    return out


def candles_main(record=False, widths=5.0, clip=3.0, plx_sigma=10.0, min_group=20, cstar_max=0.2, blend_arcsec=2.0, blend_dmag=3.0, amp_dlogp=0.05, amp_low=0.16,
                 amp_high=0.98):
    cols = ("g.ra, g.dec, g.parallax, g.parallax_error, g.ruwe, g.non_single_star, g.phot_bp_rp_excess_factor, g.phot_g_mean_mag, g.bp_rp, v.source_id, v.pf, v.p1_o, "
            "v.int_average_g, v.int_average_bp, v.int_average_rp, v.peak_to_peak_g, v.peak_to_peak_bp, v.peak_to_peak_rp")
    ceph = core.tap_async("gaia", "SELECT %s, v.type_best_classification AS cls, v.mode_best_classification AS mode FROM gaiadr3.vari_cepheid AS v JOIN gaiadr3.gaia_source AS g "
                                  "ON v.source_id = g.source_id" % cols, maxrec=100000)
    # an RR Lyrae with a parallax at ten sigma lies within a few kpc, brighter than G 14, which keeps the join within the archive's time
    rrl = core.tap_async("gaia", "SELECT %s, v.best_classification AS cls, v.metallicity FROM gaiadr3.vari_rrlyrae AS v JOIN gaiadr3.gaia_source AS g ON v.source_id = g.source_id "
                                 "WHERE v.int_average_g < 14 AND g.parallax_over_error > %g" % (cols, plx_sigma), maxrec=100000)
    stars = []
    for kind, rows in (("Cepheid", ceph), ("RR Lyrae", rrl)):
        for r in rows:
            g, bp, rp = num(r["int_average_g"]), num(r["int_average_bp"]), num(r["int_average_rp"])
            if None in (g, bp, rp):
                continue
            cls = str(r["cls"] or "").strip()
            if kind == "Cepheid":
                mode = str(r["mode"] or "").strip()
                p = num(r["pf"]) if mode == "FUNDAMENTAL" else num(r["p1_o"]) if mode == "FIRST_OVERTONE" else None
            else:
                mode = "FUNDAMENTAL" if cls == "RRab" else "FIRST_OVERTONE" if cls == "RRc" else ""
                p = num(r["pf"]) if mode == "FUNDAMENTAL" else num(r["p1_o"]) if mode == "FIRST_OVERTONE" else None
            if not p or p <= 0:
                continue
            ra, dec = float(r["ra"]), float(r["dec"])
            gal, mu, s_mu = None, None, 0.0
            for name, (cra, cdec, rad, dm) in CANDLES_CLOUDS.items():
                if kind == "Cepheid" and core.sep_arcsec(ra, dec, cra, cdec) < rad * 3600:
                    gal, mu = name, dm
            plx, eplx = num(r["parallax"]), num(r["parallax_error"])
            if gal is None:
                if not plx or not eplx or plx / eplx < plx_sigma:
                    continue
                pc = plx - CANDLES_PLX_ZP
                gal, mu, s_mu = "Galaxy", 5 * math.log10(100.0 / pc), 5 / math.log(10) * eplx / pc
            w = g - 1.90 * (bp - rp)
            stars.append(dict(source_id=str(r["source_id"]), kind=kind, cls=cls, mode=mode, galaxy=gal, logp=math.log10(p), P=p, W=w - mu, MG=g - mu, s=s_mu, G=g, ra=ra, dec=dec,
                              ruwe=num(r["ruwe"]), nss=int(num(r["non_single_star"]) or 0), amp=num(r["peak_to_peak_g"]),
                              ratio=num(r["peak_to_peak_bp"]) / num(r["peak_to_peak_rp"]) if num(r["peak_to_peak_bp"]) and num(r["peak_to_peak_rp"]) else None,
                              cstar=core.cstar(r["phot_bp_rp_excess_factor"], r["bp_rp"]) if num(r["phot_bp_rp_excess_factor"]) and num(r["bp_rp"]) is not None else None))
    groups = {}
    for o in stars:
        groups.setdefault((o["galaxy"], o["kind"], o["cls"], o["mode"]), []).append(o)
    bright, summary = [], []
    for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < min_group:
            continue
        x, y, s = (np.array([o[k] for o in members]) for k in ("logp", "W", "s"))
        c, sig_int, keep = candles_fit(x, y, s, clip)
        for o in members:
            o["resid"] = o["W"] - float(np.polyval(c, o["logp"]))
            o["width"] = math.sqrt(sig_int ** 2 + o["s"] ** 2)
            if o["resid"] < -widths * o["width"]:
                bright.append(o)
        # light that does not pulsate dilutes the amplitude: the star's own G amplitude against its peers' at the same period, and that amplitude scaled up by the
        # light the star carries in G beyond the group's G-band relation, since a red companion brightens W by more than it brightens G
        cg, _, _ = candles_fit(x, np.array([o["MG"] for o in members]), s, clip)
        for o in members:
            if o["resid"] >= -widths * o["width"]:
                continue
            o["dG"] = o["MG"] - float(np.polyval(cg, o["logp"]))
            if not o["amp"]:
                continue
            peers = np.array([m["amp"] for m in members if m is not o and m["amp"] and abs(m["logp"] - o["logp"]) < amp_dlogp])
            if len(peers) >= 10:
                o["amp_q"] = float(np.mean(peers <= o["amp"]))
                o["amp_qu"] = float(np.mean(peers <= o["amp"] * 10 ** (-0.4 * min(o["dG"], 0.0))))
            # a companion redder than the pulsator dilutes the RP amplitude more than the BP amplitude, a bluer one the reverse
            ratios = np.array([m["ratio"] for m in members if m is not o and m["ratio"] and abs(m["logp"] - o["logp"]) < amp_dlogp])
            if o["ratio"] and len(ratios) >= 10:
                o["ratio_q"] = float(np.mean(ratios <= o["ratio"]))
        summary.append(dict(group=" ".join(key), n=len(members), slope=float(c[0]), zero=float(c[1]), scatter=sig_int,
                            bright=sum(1 for o in members if o.get("resid", 0) < -widths * o["width"])))
    ogle = candles_ogle()
    galactic = [o["source_id"] for o in bright if o["galaxy"] == "Galaxy"]
    general = {}
    if galactic:
        for r in core.tap("gaia", "SELECT source_id, best_class_name, best_class_score FROM gaiadr3.vari_classifier_result WHERE source_id IN (%s)" % ",".join(galactic)):
            general[str(r["source_id"])] = (str(r["best_class_name"]).strip(), num(r["best_class_score"]))
    for o in bright:
        q = "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.7f, %.7f, %g)) AND source_id <> %s" % (
            o["ra"], o["dec"], blend_arcsec / 3600.0, o["source_id"])
        near = [num(n["phot_g_mean_mag"]) for n in core.tap("gaia", q)]
        o["neighbours"] = sum(1 for m in near if m is not None and m < o["G"] + blend_dmag)
        o["ogle"] = candles_ogle_read(o, ogle) if o["galaxy"] in CANDLES_CLOUDS else None
        o["general"] = general.get(o["source_id"])
        o["diluted"] = o.get("amp_q") is not None and o["amp_q"] < amp_low and o["amp_qu"] <= amp_high
        o["coloured"] = o.get("amp_q") is not None and o["amp_q"] < 0.5 and o.get("ratio_q") is not None and (o["ratio_q"] < amp_low or o["ratio_q"] > 1 - amp_low)
        remark = ((o["ogle"] or {}).get("remark") or "").lower()
        want = "RR" if o["kind"] == "RR Lyrae" else "CEP"
        o["vet"] = ("binary" if (o["ruwe"] or 0) > 1.4 or o["nss"] or "binary" in remark else
                    "blend" if o["neighbours"] or (o["cstar"] is not None and o["cstar"] > cstar_max) or "double" in remark or "blend" in remark else
                    "published-solution" if o["ogle"] and o["ogle"]["verdict"] in ("class", "mode", "period") else
                    "blend" if (o["ogle"] and o["ogle"]["verdict"] == "normal") or o["diluted"] or o["coloured"] else
                    "published-solution" if o["general"] and o["general"][0] != want else None)
    left = [o for o in bright if o["vet"] is None]
    print("%d pulsators on a relation of their class, mode and galaxy (groups of %d or more): %s" % (
        sum(g["n"] for g in summary), min_group, "; ".join("%s %d, slope %.2f, scatter %.3f, bright by %g widths %d" % (
            g["group"], g["n"], g["slope"], g["scatter"], widths, g["bright"]) for g in summary)))
    print("brighter than %g widths: %d; a companion in the astrometry or OGLE's remarks %d, a neighbour's light in Gaia, on OGLE's normal relation or in the "
          "amplitudes %d (diluted %d, a companion's colour in the BP to RP amplitude ratio %d), a class, mode or period Gaia mistook %d; unexplained %d" % (
              widths, len(bright), sum(o["vet"] == "binary" for o in bright), sum(o["vet"] == "blend" for o in bright), sum(1 for o in bright if o["diluted"]),
              sum(1 for o in bright if o["coloured"]), sum(o["vet"] == "published-solution" for o in bright), len(left)))
    for o in sorted(bright, key=lambda o: o["resid"] / o["width"])[:120]:
        og = o["ogle"] or {}
        print("  Gaia DR3 %s %s %s %s in the %s: P %.4f d, %+.2f mag (%.1f widths), %s in G, RUWE %s, NSS %d, C* %s, %d neighbours; G amplitude %s at percentile %s of its "
              "peers, %s scaled, BP/RP amplitude ratio %s at percentile %s; OGLE %s %s %s P %s (%s%s)%s%s; %s" % (
                  o["source_id"], o["kind"], o["cls"], o["mode"], o["galaxy"], o["P"], o["resid"], o["resid"] / o["width"],
                  "%+.2f" % o["dG"] if o.get("dG") is not None else "-", o["ruwe"], o["nss"],
                  "%.2f" % o["cstar"] if o["cstar"] is not None else "-", o["neighbours"], o["amp"], "%.2f" % o["amp_q"] if o.get("amp_q") is not None else "-",
                  "%.2f" % o["amp_qu"] if o.get("amp_qu") is not None else "-", "%.3f" % o["ratio"] if o["ratio"] else "-",
                  "%.2f" % o["ratio_q"] if o.get("ratio_q") is not None else "-", og.get("id", "-"), og.get("kind", ""), og.get("mode", ""), og.get("P"),
                  og.get("verdict", "-"), ", W_I %+.2f" % og["w_resid"] if og.get("w_resid") is not None else "", "; '%s'" % og["remark"] if og.get("remark") else "",
                  "; Gaia's classifier %s %.2f" % o["general"] if o["general"] else "", o["vet"] or "unexplained"))
    json.dump(dict(groups=summary, bright=bright), open(core.results("standard_candles.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_standard_candles", "Standard candles that are not standard", "Gaia DR3 vari_cepheid and vari_rrlyrae with gaia_source",
            "Cepheids of the Magellanic Clouds and Cepheids and RR Lyrae with parallaxes at %g sigma, on relations of %d or more: %d" % (
                plx_sigma, min_group, sum(g["n"] for g in summary)), os.path.abspath(__file__), "candles", sum(g["n"] for g in summary), "pulsators",
            dict(widths=widths, clip=clip, plx_sigma=plx_sigma, min_group=min_group, cstar_max=cstar_max, blend_arcsec=blend_arcsec, blend_dmag=blend_dmag,
                 amp_dlogp=amp_dlogp, amp_low=amp_low, amp_high=amp_high),
            "each period-Wesenheit relation's own scatter with the parallax's error", "a pulsator %g widths brighter than its period allows" % widths,
            "%d brighter than %g widths: %d with a companion in the astrometry or OGLE's remarks, %d with a neighbour's light in Gaia, on OGLE's normal relation or in "
            "the amplitudes (%d diluted, %d with a companion's colour in the BP to RP amplitude ratio), %d in a class, mode or period Gaia mistook; %d unexplained" % (
                len(bright), widths, sum(o["vet"] == "binary" for o in bright), sum(o["vet"] == "blend" for o in bright), sum(1 for o in bright if o["diluted"]),
                sum(1 for o in bright if o["coloured"]), sum(o["vet"] == "published-solution" for o in bright), len(left)),
            [dict(target="gaia_dr3:%s" % o["source_id"], stat=round(o["resid"], 3), unit="mag from the period-Wesenheit relation", vet=o["vet"],
                  note="%s %s %s in the %s, P %.4f d, %.1f widths, %s mag in G, RUWE %s, NSS %d, C* %s, %d neighbours within %g arcsec; G amplitude %s at percentile %s "
                       "of its peers, %s scaled, BP/RP amplitude ratio at percentile %s; OGLE %s %s %s P %s (%s)%s%s" % (
                      o["kind"], o["cls"], o["mode"], o["galaxy"], o["P"], o["resid"] / o["width"], "%+.2f" % o["dG"] if o.get("dG") is not None else "-",
                      o["ruwe"], o["nss"],
                      "%.2f" % o["cstar"] if o["cstar"] is not None else "-", o["neighbours"], blend_arcsec, o["amp"],
                      "%.2f" % o["amp_q"] if o.get("amp_q") is not None else "-", "%.2f" % o["amp_qu"] if o.get("amp_qu") is not None else "-",
                      "%.2f" % o["ratio_q"] if o.get("ratio_q") is not None else "-",
                      (o["ogle"] or {}).get("id", "-"), (o["ogle"] or {}).get("kind", ""), (o["ogle"] or {}).get("mode", ""), (o["ogle"] or {}).get("P"),
                      (o["ogle"] or {}).get("verdict", "-"), "; '%s'" % o["ogle"]["remark"] if (o["ogle"] or {}).get("remark") else "",
                      "; Gaia's classifier %s %.2f" % o["general"] if o["general"] else "")) for o in bright])
        print("recorded")


overbright_DOC = r"""Stars brighter than themselves: Gaia DR3 stars within 100 pc with parallaxes at twenty sigma whose RVS spectra give GSP-Spec temperatures of 4000-5500 K and dwarf gravities, log g above 4.2, single in Gaia's astrometry (RUWE below 1.4, no non-single-star solution), their absolute G against the main sequence fitted to the same stars in temperature and metallicity; two equal stars unresolved stand 0.753 mag above it, so a star brighter than that by three widths of the sequence carries light no second star of its kind gives, a reflector's or a young star's, and each is read for GSP-Spec's own flags on the fit (any of the thirteen parameter flags at 2 or above), for the MK type Hipparcos gives it through Gaia's best-neighbour table (a luminosity class of I to IV marks a giant or subgiant whose gravity the RVS fit missed), for its activity index (Gaia's Ca II infrared triplet) and for its RV amplitude.

    python -I seti.py dimming overbright [record] [--plx-sigma 20] [--teff 4000 5500] [--logg-min 4.2] [--twin 0.753] [--widths 3] [--activity-max 0.05]"""


def overbright_evolved(sp):
    """Whether an MK type names a giant or subgiant: a luminosity class of I to IV other than IV-V, or Mount Wilson's g and sg prefixes."""
    s = (sp or "").strip()
    if s.startswith(("sg", "g")):
        return True
    m = re.match(r"(?:sd|d)?[OBAFGKM][0-9.]*(?:/[OBAFGKM]?[0-9.]*)?(III|II|IV|Ia|Ib|I)?([-/](?:IV|V))?", s)
    return bool(m and m.group(1) and not (m.group(1) == "IV" and m.group(2) and m.group(2)[1:] == "V"))


def overbright_main(record=False, plx_sigma=20.0, teff=[4000.0, 5500.0], logg_min=4.2, twin=0.753, widths=3.0, activity_max=0.05):
    # the Gaia Catalogue of Nearby Stars drives the join, since a selection on gaia_source by parallax runs past the archive's time
    rows = core.tap_async("gaia", "SELECT g.source_id, g.parallax, g.parallax_error, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.non_single_star, g.rv_amplitude_robust, "
                                  "g.rv_nb_transits, a.teff_gspspec, a.logg_gspspec, a.mh_gspspec, a.flags_gspspec, a.activityindex_espcs FROM external.gaiaedr3_gcns_main_1 AS n "
                                  "JOIN gaiadr3.astrophysical_parameters AS a ON n.source_id = a.source_id JOIN gaiadr3.gaia_source AS g ON g.source_id = n.source_id "
                                  "WHERE a.teff_gspspec BETWEEN %g AND %g AND a.mh_gspspec IS NOT NULL AND a.logg_gspspec IS NOT NULL" % (teff[0], teff[1]), maxrec=200000)
    rows = [r for r in rows if (num(r["parallax"]) or 0) > 10 and (num(r["parallax"]) or 0) > plx_sigma * (num(r["parallax_error"]) or 1e9)]
    out = []
    for r in rows:
        t, lg, mh, plx = num(r["teff_gspspec"]), num(r["logg_gspspec"]), num(r["mh_gspspec"]), num(r["parallax"])
        if None in (t, lg, mh, plx) or num(r["phot_g_mean_mag"]) is None:
            continue
        out.append(dict(source_id=str(r["source_id"]), teff=t, logg=lg, mh=mh, MG=float(r["phot_g_mean_mag"]) + 5 * math.log10(plx / 100.0), bp_rp=num(r["bp_rp"]),
                        ruwe=num(r["ruwe"]), nss=int(num(r["non_single_star"]) or 0), rvamp=num(r["rv_amplitude_robust"]), act=num(r["activityindex_espcs"]),
                        flags=str(r["flags_gspspec"] or "")))
    dwarfs = [o for o in out if o["logg"] > logg_min]
    single = [o for o in dwarfs if (o["ruwe"] or 0) < 1.4 and not o["nss"]]
    x = np.array([[1.0, (o["teff"] - 4750) / 1000, ((o["teff"] - 4750) / 1000) ** 2, ((o["teff"] - 4750) / 1000) ** 3, o["mh"]] for o in single])
    y = np.array([o["MG"] for o in single])
    keep = np.ones(len(y), bool)
    for _ in range(10):
        c, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
        res = y - x @ c
        sig = core.robust_sigma(res[keep])
        new = np.abs(res) < 3 * sig
        if (new == keep).all():
            break
        keep = new
    for o, d in zip(single, res):
        o["dM"] = float(d)
    edges = np.arange(-2.0, 1.01, 0.25)
    hist = [int(np.sum((res >= a) & (res < a + 0.25))) for a in edges[:-1]]
    over = [o for o in single if o["dM"] < -(twin + widths * sig)]
    mk = {}
    if over:
        for r in core.tap("gaia", "SELECT b.source_id, h.sptype FROM gaiadr3.hipparcos2_best_neighbour AS b JOIN public.hipparcos AS h ON h.hip = b.original_ext_source_id "
                                  "WHERE b.source_id IN (%s)" % ",".join(o["source_id"] for o in over)):
            mk[str(r["source_id"])] = str(r["sptype"] or "").strip()
    for o in over:
        o["mk"] = mk.get(o["source_id"], "")
        o["vet"] = ("fit-boundary" if any(ch.isdigit() and int(ch) >= 2 for ch in o["flags"][:13]) else
                    "evolved" if overbright_evolved(o["mk"]) else
                    "young-star" if o["act"] is not None and o["act"] > activity_max else
                    "binary" if o["rvamp"] is not None and o["rvamp"] > 5 else None)
    left = [o for o in over if o["vet"] is None]
    print("%d stars within 100 pc with GSP-Spec temperatures of %g-%g K; %d with log g above %g, %d of them single in the astrometry; the main sequence in temperature and "
          "metallicity leaves a scatter of %.3f mag (metallicity term %+.2f mag/dex); offsets in 0.25 mag bins from -2: %s; brighter than the twin limit by %g widths "
          "(%.2f mag): %d, flagged fits %d, giants and subgiants by MK type %d, active %d, RV-variable %d; unexplained %d" % (
              len(out), teff[0], teff[1], len(dwarfs), logg_min, len(single), sig, c[4], hist, widths, twin + widths * sig, len(over),
              sum(o["vet"] == "fit-boundary" for o in over), sum(o["vet"] == "evolved" for o in over), sum(o["vet"] == "young-star" for o in over),
              sum(o["vet"] == "binary" for o in over), len(left)))
    for o in sorted(over, key=lambda o: o["dM"])[:30]:
        print("  Gaia DR3 %s: %.0f K, log g %.2f, [M/H] %+.2f, M_G %.2f, %+.2f mag from the sequence, RUWE %s, RV amplitude %s, activity %s, flags %s, MK %s; %s" % (
            o["source_id"], o["teff"], o["logg"], o["mh"], o["MG"], o["dM"], o["ruwe"], o["rvamp"], o["act"], o["flags"][:13], o["mk"] or "-", o["vet"] or "unexplained"))
    json.dump(dict(scatter=sig, coef=[float(v) for v in c], hist=hist, over=over), open(core.results("overbright_dwarfs.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_overbright_dwarfs", "Stars brighter than themselves", "Gaia DR3 gaia_source with GSP-Spec astrophysical parameters",
            "single dwarfs of %g-%g K within 100 pc with GSP-Spec parameters and parallaxes at %g sigma: %d" % (teff[0], teff[1], plx_sigma, len(single)), os.path.abspath(__file__),
            "overbright", len(single), "stars", dict(plx_sigma=plx_sigma, teff=teff, logg_min=logg_min, twin=twin, widths=widths, activity_max=activity_max),
            "the main sequence's own scatter of %.3f mag in temperature and metallicity; offsets in 0.25 mag bins from -2: %s" % (sig, hist),
            "a single-looking dwarf brighter than an unresolved equal twin by %g widths" % widths,
            "%d brighter than %.2f mag: %d with GSP-Spec's fit flagged, %d giants or subgiants by their MK type, %d active, %d RV-variable; %d unexplained" % (
                len(over), twin + widths * sig, sum(o["vet"] == "fit-boundary" for o in over), sum(o["vet"] == "evolved" for o in over),
                sum(o["vet"] == "young-star" for o in over), sum(o["vet"] == "binary" for o in over), len(left)),
            [dict(target="gaia_dr3:%s" % o["source_id"], stat=round(o["dM"], 3), unit="mag from the main sequence", vet=o["vet"],
                  note="%.0f K, log g %.2f, [M/H] %+.2f, M_G %.2f, RUWE %s, RV amplitude %s, activity %s, GSP-Spec flags %s, Hipparcos MK %s" % (
                      o["teff"], o["logg"], o["mh"], o["MG"], o["ruwe"], o["rvamp"], o["act"], o["flags"][:13], o["mk"] or "-")) for o in over])
        print("recorded")


APPROACHES = {
    "overbright": (lambda: core.run_cli(overbright_main), overbright_DOC),
    "candles": (lambda: core.run_cli(candles_main), candles_DOC),
    "irchanged": (lambda: core.run_cli(irchanged_main), irchanged_DOC),
    "colourchange": (lambda: core.run_cli(colourchange_main), colourchange_DOC),
    "hipdim": (lambda: core.run_cli(hipdim_main), hipdim_DOC),
    "releasestep": (lambda: core.run_cli(releasestep_main), releasestep_DOC),
    "dr2gone": (lambda: core.run_cli(dr2gone_main), dr2gone_DOC),
    "dr2new": (lambda: core.run_cli(dr2new_main), dr2new_DOC),
    "greysub": (lambda: core.run_cli(greysub_main), greysub_DOC),
    "uniquevar": (lambda: core.run_cli(uniquevar_main), uniquevar_DOC),
    "hipgone": (lambda: core.run_cli(hipgone_main), hipgone_DOC),
    "chromatic": (chromatic_cli, chromatic_DOC),
    "achromatic": (achromatic_cli, achromatic_DOC),
    "occulters": (lambda: core.run_cli(occulters_main), occulters_DOC),
    "hipparcos": (hipparcos_cli, hipparcos_DOC),
    "gaia": (gaia_cli, gaia_DOC),
    "hip_gaia": (hip_gaia_cli, hip_gaia_DOC),
    "dasch": (dasch_cli, dasch_DOC),
    "tess": (tess_cli, tess_DOC),
    "neowise": (neowise_cli, neowise_DOC),
    "alerts": (alerts_cli, alerts_DOC),
}


