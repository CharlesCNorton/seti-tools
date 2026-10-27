"""Waste heat: infrared excess, enclosures and dark companions."""
import argparse
import csv
import io
import json
import math
import os
import random
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from setitools import catalogues, core, science
from setitools.core import ms_luminosity, num, quiet_warnings, sep_arcsec


# --------------------------------------------------------------------------------------------------------------

excess_DOC = r"""Mid-infrared excess (Ks-W3, Ks-W4 above 0.25 mag at 5 sigma) and NEOWISE trends of Breakthrough Listen 2016 L-band targets.

    python -I seti.py infrared excess [--types FGK] [--neowise]"""

excess_SIMBAD = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
excess_IRSA = "https://irsa.ipac.caltech.edu/TAP/sync"


def excess_tap(url, q, fmt="json", timeout=180):
    data = urllib.parse.urlencode(dict(request="doQuery", lang="ADQL", format=fmt, query=q)).encode()
    return urllib.request.urlopen(url, data=data, timeout=timeout).read()


def excess_simbad_rows(hips):
    q = ("SELECT ident.id, basic.ra, basic.dec, basic.pmra, basic.pmdec, basic.otype, basic.nbref "
         "FROM basic JOIN ident ON ident.oidref=basic.oid WHERE ident.id IN (%s)"
         % ",".join("'%s'" % h for h in hips))
    return {r[0]: r[1:] for r in json.loads(excess_tap(excess_SIMBAD, q))["data"]}


def excess_allwise(ra, dec):
    """Nearest AllWISE source within 6 arcsec, or None."""
    return core.allwise_nearest(ra, dec, 6.0)


def excess_neowise_slopes(ra, dec):
    q = ("SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qual_frame FROM neowiser_p1bs_psd WHERE "
         "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f))=1" % (ra, dec, 3 / 3600.0))
    from astropy.io.votable import parse_single_table
    t = parse_single_table(io.BytesIO(excess_tap(excess_IRSA, q, fmt="votable", timeout=600))).to_table(use_names_over_ids=True)
    t = t[np.asarray(t["qual_frame"], float) > 0]
    out = {}
    for b in ("w1", "w2"):
        m = np.asarray(t[b + "mpro"], float)
        j = np.asarray(t["mjd"], float)
        ok = np.isfinite(m)
        if ok.sum() < 20:
            continue
        yr = (j[ok] - j[ok].min()) / 365.25
        mm = m[ok]
        # epoch medians; NEOWISE visits every six months
        ep = np.round(yr * 2).astype(int)
        ey = np.array([yr[ep == e].mean() for e in np.unique(ep)])
        em = np.array([np.median(mm[ep == e]) for e in np.unique(ep)])
        if len(ey) < 6:
            continue
        p, cov = np.polyfit(ey, em, 1, cov=True)
        out[b] = (p[0] * 1000.0, np.sqrt(cov[0, 0]) * 1000.0, len(ey), float(np.ptp(em)) * 1000.0)
    return out


def excess_main():
    p = argparse.ArgumentParser()
    p.add_argument("--types", default="FGK")
    p.add_argument("--neowise", action="store_true")
    a = p.parse_args()
    stars = [s for s in json.load(open(core.results("session_stars.json")))
             if (s["sp"] or "?")[:1].upper() in a.types]
    sim = excess_simbad_rows([s["hip"] for s in stars])
    print("%d stars of types %s" % (len(stars), a.types), flush=True)
    for s in stars:
        r = sim.get(s["hip"])
        if not r:
            continue
        ra, dec, pmra, pmdec, otype, nbref = r
        dt = 2010.5 - 2000.0
        ra10 = ra + (pmra or 0) * dt / 3.6e6 / np.cos(np.radians(dec))
        dec10 = dec + (pmdec or 0) * dt / 3.6e6
        try:
            w = excess_allwise(ra10, dec10)
        except Exception as e:
            print("%s: AllWISE query failed (%s)" % (s["hip"], str(e)[:60]), flush=True)
            continue
        if w is None:
            print("%s: no AllWISE source" % s["hip"], flush=True)
            continue
        k = w["k_m_2mass"] if w["k_m_2mass"] is not None else np.nan
        flags = []
        for band, i in (("w3", 2), ("w4", 3)):
            m, sg, snr = float(w[band + "mpro"]), float(w[band + "sigmpro"] or np.nan), float(w[band + "snr"])
            q = str(w["ph_qual"])[i]
            cc = str(w["cc_flags"])[i]
            ex = k - m
            sig = np.hypot(sg, 0.03)
            if q == "A" and cc == "0" and snr >= 5 and ex > 0.25 and ex / sig >= 5:
                flags.append("%s excess Ks-%s=%.2f (%.1f sigma)" % (band.upper(), band.upper(), ex, ex / sig))
        line = "%s %s %.1f pc  %s  refs %s  Ks %.2f W3 %.2f W4 %.2f  qual %s cc %s" % (
            s["hip"], s["sp"], s["pc"], otype, nbref, k, float(w["w3mpro"]), float(w["w4mpro"]),
            w["ph_qual"], w["cc_flags"])
        if flags:
            print("EXCESS " + line + "  " + "; ".join(flags), flush=True)
        else:
            print("  " + line, flush=True)
        if a.neowise:
            try:
                sl = excess_neowise_slopes(ra + (pmra or 0) * 19.0 / 3.6e6 / np.cos(np.radians(dec)),
                                    dec + (pmdec or 0) * 19.0 / 3.6e6)
            except Exception as e:
                print("    NEOWISE query failed (%s)" % str(e)[:60], flush=True)
                continue
            for b, (sl_, se, n, rng) in sl.items():
                tag = "TREND" if abs(sl_) > 10 and abs(sl_) > 4 * se else "     "
                print("    %s %s slope %+.1f +- %.1f mmag/yr over %d epochs, range %.0f mmag"
                      % (tag, b.upper(), sl_, se, n, rng), flush=True)


def excess_cli():
    excess_main()


# --------------------------------------------------------------------------------------------------------------

hz_DOC = r"""Stacked W1-W3 and W1-W4 excess of habitable-zone hosts within 100 pc over Gaia-matched controls, bounding the mean 300 K covering fraction.

    python -I seti.py infrared hz fetch
    python -I seti.py infrared hz analyse
    python -I seti.py infrared hz record"""

hz_ROOT = os.environ.get("HZ_ROOT", core.data("hz_stack"))
hz_HERE = core.RESULTS
hz_HZ = (0.32, 1.78)
hz_D_MAX = 100.0
hz_N_CTRL = 25
hz_N_COMPARE = 250
hz_WISE_EPOCH = 2010.5


def hz_hosts():
    rows = core.tap("exoplanets", "SELECT pl_name, hostname, ra, dec, sy_dist, pl_insol, st_teff FROM pscomppars "
                                 "WHERE sy_dist < %g" % hz_D_MAX)
    by = {}
    for r in rows:
        by.setdefault(r["hostname"], []).append(r)
    hz, other = [], []
    for name, ps in by.items():
        ins = [num(p["pl_insol"]) for p in ps]
        h = dict(host=name, ra=num(ps[0]["ra"]), dec=num(ps[0]["dec"]), teff=num(ps[0]["st_teff"]),
                 planets=[p["pl_name"] for p in ps])
        if any(i is not None and hz_HZ[0] <= i <= hz_HZ[1] for i in ins):
            hz.append(h)
        elif all(i is not None and not (hz_HZ[0] / 3 <= i <= hz_HZ[1] * 3) for i in ins):
            other.append(h)
    return hz, other


def hz_gaia_of(hs):
    m = core.xmatch([dict(source_id=h["host"], ra=h["ra"], dec=h["dec"]) for h in hs], "I/355/gaiadr3", 10.0)
    best = {}
    for r in m:
        if num(r["Gmag"]) is not None and (r["source_id"] not in best or num(r["Gmag"]) < num(best[r["source_id"]]["Gmag"])):
            best[r["source_id"]] = r
    out = []
    for h in hs:
        r = best.get(h["host"])
        if not r or not num(r["Plx"]) or num(r["BP-RP"]) is None:
            continue
        h.update(source_id=r["Source"], gra=num(r["RAdeg"]), gdec=num(r["DEdeg"]), pmra=num(r["pmRA"]) or 0.0,
                 pmdec=num(r["pmDE"]) or 0.0, plx=num(r["Plx"]), g=num(r["Gmag"]), bp_rp=num(r["BP-RP"]),
                 ruwe=num(r["RUWE"]))
        out.append(h)
    return out


def hz_pool():
    """Control pool: Gaia DR3 parallax above 9 mas at S/N 10, RUWE below 1.4, G below 17."""
    path = os.path.join(hz_ROOT, "pool.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows = core.gaia("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp, b FROM "
                         "gaiadr3.gaia_source WHERE parallax > 9 AND parallax_over_error > 10 AND ruwe < 1.4 AND "
                         "phot_g_mean_mag < 17 AND bp_rp IS NOT NULL")
    out = [dict(source_id=r["source_id"], gra=float(r["ra"]), gdec=float(r["dec"]), pmra=num(r["pmra"]) or 0.0,
                pmdec=num(r["pmdec"]) or 0.0, plx=float(r["parallax"]), g=float(r["phot_g_mean_mag"]),
                bp_rp=float(r["bp_rp"]), b=float(r["b"])) for r in rows]
    json.dump(out, open(path, "w"))
    return out


def hz_controls(h, P, rng, exclude):
    """Up to N_CTRL pool stars matched in BP-RP, G, parallax and |b|."""
    from astropy.coordinates import SkyCoord
    hb = abs(float(SkyCoord(ra=h["gra"], dec=h["gdec"], unit="deg").galactic.b.deg))
    m = ((np.abs(P["bp_rp"] - h["bp_rp"]) < 0.03) & (np.abs(P["g"] - h["g"]) < 0.25) & (P["plx"] > h["plx"] / 1.3)
         & (P["plx"] < h["plx"] * 1.3) & (np.abs(np.abs(P["b"]) - hb) < 15.0))
    idx = [i for i in np.where(m)[0] if P["ids"][i] not in exclude]
    rng.shuffle(idx)
    return [P["rows"][i] for i in idx[:hz_N_CTRL]]


def hz_at_wise(s):
    dt = hz_WISE_EPOCH - 2016.0
    return (s["gra"] + s["pmra"] / 3.6e6 * dt / max(math.cos(math.radians(s["gdec"])), 1e-6),
            s["gdec"] + s["pmdec"] / 3.6e6 * dt)


def hz_fetch():
    os.makedirs(hz_ROOT, exist_ok=True)
    hz, other = hz_hosts()
    random.seed(1)
    other = random.sample(other, min(hz_N_COMPARE, len(other)))
    hz, other = hz_gaia_of(hz), hz_gaia_of(other)
    for h in hz:
        h["group"] = "hz"
    for h in other:
        h["group"] = "other"
    allh = hz + other
    rows = hz_pool()
    P = dict(rows=rows, ids=[r["source_id"] for r in rows], bp_rp=np.array([r["bp_rp"] for r in rows]),
             g=np.array([r["g"] for r in rows]), plx=np.array([r["plx"] for r in rows]), b=np.array([r["b"] for r in rows]))
    exclude = {h["source_id"] for h in allh}
    rng = random.Random(2)
    for h in allh:
        h["controls"] = hz_controls(h, P, rng, exclude)
    stars = []
    for h in allh:
        stars.append(dict(source_id=h["source_id"], s=h))
        stars += [dict(source_id=c["source_id"], s=c) for c in h["controls"]]
    probe = []
    for x in stars:
        ra, dec = hz_at_wise(x["s"])
        probe.append(dict(source_id=x["source_id"], ra=ra, dec=dec))
    wise = {}
    for k in range(0, len(probe), 5000):
        for r in core.xmatch(probe[k:k + 5000], "II/328/allwise", 3.0):
            if r["source_id"] not in wise or float(r["angDist"]) < float(wise[r["source_id"]]["angDist"]):
                wise[r["source_id"]] = r
    keep = ("W1mag", "W2mag", "W3mag", "W4mag", "e_W1mag", "e_W2mag", "e_W3mag", "e_W4mag", "Kmag", "e_Kmag", "ccf", "ex",
            "qph", "angDist")
    for h in allh:
        h["wise"] = {k: wise[h["source_id"]].get(k) for k in keep} if h["source_id"] in wise else None
        for c in h["controls"]:
            c["wise"] = {k: wise[c["source_id"]].get(k) for k in keep} if c["source_id"] in wise else None
    json.dump(allh, open(os.path.join(hz_ROOT, "hz_stack.json"), "w"), indent=1)
    print("%d habitable-zone hosts and %d comparison hosts within %d pc with Gaia DR3 counterparts; %d controls; "
          "AllWISE for %d of %d stars" % (len(hz), len(other), hz_D_MAX, sum(len(h["controls"]) for h in allh), len(wise),
                                         len(stars)))


def hz_reference(w):
    """Colour reference band: W1 when unsaturated (8 mag or fainter), else Ks."""
    w1 = num((w or {}).get("W1mag"))
    return "W1" if w1 is not None and w1 >= 8.0 else "K"


def hz_colour(w, band, ref):
    """Reference minus W3 or W4 with error, or None when a band fails the cuts."""
    if not w:
        return None
    q, c = str(w.get("qph") or "UUUU"), str(w.get("ccf") or "0000")
    wb, eb = num(w.get("W%dmag" % (band + 1))), num(w.get("e_W%dmag" % (band + 1)))
    if ref == "W1":
        r, er = num(w.get("W1mag")), num(w.get("e_W1mag"))
        if r is None or r < 8.0 or q[0] not in "AB" or c[0] != "0":
            return None
    else:
        r, er = num(w.get("Kmag")), num(w.get("e_Kmag"))
        if r is None or r < 4.5:
            return None
    if None in (wb, eb, er) or str(w.get("ex")) != "0" or q[band] not in "AB" or c[band] != "0":
        return None
    return r - wb, math.hypot(er, eb)


def hz_excess(target, others, band):
    ref = hz_reference(target)
    t = hz_colour(target, band, ref)
    cs = [hz_colour(o, band, ref) for o in others]
    cs = np.array([x[0] for x in cs if x])
    if t is None or len(cs) < 5:
        return None
    med = float(np.median(cs))
    sd = 1.4826 * float(np.median(np.abs(cs - med)))
    sd = max(sd, 0.01)
    return t[0] - med, math.hypot(sd * math.sqrt(1 + 1.0 / len(cs)), t[1]), len(cs)


def hz_wmean(xs):
    if not xs:
        return None, None
    w = np.array([1.0 / s ** 2 for _, s in xs])
    v = np.array([x for x, _ in xs])
    return float(np.sum(w * v) / np.sum(w)), float(1.0 / math.sqrt(np.sum(w)))


def hz_planck_ratio(lam_um, t1, t2):
    """B_nu(t1) / B_nu(t2) at lam_um."""
    x = 14387.77 / lam_um
    return math.expm1(x / t2) / math.expm1(x / t1)


def hz_swarm_gain(lam_um, teff, t_swarm=300.0):
    """Flux added at lam_um by a t_swarm swarm intercepting a teff photosphere, relative to the photosphere."""
    return hz_planck_ratio(lam_um, t_swarm, teff) * (teff / t_swarm) ** 4


def hz_galaxies_near(ra, dec, radius=10.0):
    """Extended Pan-STARRS DR2 sources within radius: background galaxies in the W3 and W4 beams."""
    import csv
    import io
    u = ("https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv?ra=%.6f&dec=%.6f&radius=%.5f"
         "&nDetections.gte=5&columns=[objID,raMean,decMean,iMeanPSFMag,iMeanKronMag,rMeanPSFMag]" % (ra, dec, radius / 3600))
    try:
        rows = list(csv.DictReader(io.StringIO(core.get(u, timeout=120, retries=3).decode())))
    except Exception:
        return None
    out = []
    for r in rows:
        ip, ik = num(r["iMeanPSFMag"]), num(r["iMeanKronMag"])
        if ip and ik and ip > 0 and ik > 0 and ip - ik > 0.05:
            sep = math.hypot((float(r["raMean"]) - ra) * math.cos(math.radians(dec)), float(r["decMean"]) - dec) * 3600
            out.append(dict(sep=sep, i_psf=ip, i_kron=ik, r=num(r["rMeanPSFMag"])))
    return out


def hz_analyse():
    allh = json.load(open(os.path.join(hz_ROOT, "hz_stack.json")))
    out = {}
    for band, lam in ((2, 12.0), (3, 22.0)):
        res = {"hz": [], "other": []}
        null = []
        for h in allh:
            cw = [c["wise"] for c in h["controls"]]
            e = hz_excess(h["wise"], cw, band)
            if e:
                res[h["group"]].append(dict(host=h["host"], dx=e[0], sd=e[1], n=e[2], teff=h.get("teff")))
            for i, c in enumerate(h["controls"]):
                e = hz_excess(c["wise"], cw[:i] + cw[i + 1:], band)
                if e:
                    null.append((e[0], e[1]))
        line = {}
        for g in ("hz", "other"):
            m, s = hz_wmean([(x["dx"], x["sd"]) for x in res[g]])
            line[g] = (m, s, len(res[g]))
        nm, ns = hz_wmean(null)
        zs = np.array([x / s for x, s in null])
        print("W1/Ks-W%d (%d um): habitable-zone hosts %+.4f +- %.4f mag (%d), other hosts %+.4f +- %.4f (%d), controls "
              "against their siblings %+.4f +- %.4f (%d; z beyond 3: %.2f per cent)"
              % (band + 1, lam, line["hz"][0], line["hz"][1], line["hz"][2], line["other"][0], line["other"][1],
                 line["other"][2], nm, ns, len(null), 100.0 * np.mean(zs > 3)))
        # each host's excess as a 300 K covering fraction, less the controls' offset
        fs = []
        for x in res["hz"]:
            g = hz_swarm_gain(lam, x["teff"] or 4000.0)
            lo, hi = 10 ** ((x["dx"] - nm) / 2.5) - 1, 10 ** ((x["dx"] - nm + x["sd"]) / 2.5) - 1
            fs.append((lo / g, max(hi - lo, 1e-9) / g))
        fm, fsd = hz_wmean(fs)
        fbound = max(fm, 0.0) + 2 * fsd
        print("  mean covering fraction of a 300 K swarm over the habitable-zone hosts %.1e +- %.1e; 95 per cent bound %.1e"
              % (fm, fsd, fbound))
        top = sorted(res["hz"], key=lambda x: -x["dx"] / x["sd"])[:8]
        print("  largest habitable-zone host excesses: " + "; ".join(
            "%s %+.3f (%.1f sigma, %d controls)" % (x["host"], x["dx"], x["dx"] / x["sd"], x["n"]) for x in top))
        allg = {h["host"]: h for h in allh}
        for x in top:
            if x["dx"] / x["sd"] < 2.5:
                break
            h = allg[x["host"]]
            gal = hz_galaxies_near(h["gra"], h["gdec"])
            x["galaxies"] = gal
            print("    %s: %s" % (x["host"], "extended Pan-STARRS source(s) within 10 arcsec: " + "; ".join(
                "%.1f\" r %.1f, i PSF-Kron %.2f" % (gg["sep"], gg["r"] or 0, gg["i_psf"] - gg["i_kron"]) for gg in gal)
                if gal else "no extended Pan-STARRS source within 10 arcsec" if gal is not None else "Pan-STARRS query failed"))
        lim = (line["hz"][0] - nm) + 2 * math.hypot(line["hz"][1], ns)
        f = fbound
        out["W%d" % (band + 1)] = dict(hz=line["hz"], other=line["other"], null=(nm, ns, len(null)), bound_mag=lim,
                                       covering_fraction=f, hosts=res)
    json.dump(out, open(os.path.join(hz_HERE, "hz_stack.json"), "w"), indent=1)


def hz_record():
    out = json.load(open(os.path.join(hz_HERE, "hz_stack.json")))
    allh = {h["host"]: h for h in json.load(open(os.path.join(hz_ROOT, "hz_stack.json")))}
    rows = {}
    for band in ("W3", "W4"):
        for g in ("hz", "other"):
            for x in out[band]["hosts"][g]:
                sid = allh[x["host"]]["source_id"]
                r = rows.setdefault(sid, dict(target="gaia_dr3:" + sid, note=x["host"] + " (" + g + ")"))
                r["note"] += "; W1/Ks-%s excess %+.3f (%.1f sigma)" % (band, x["dx"], x["dx"] / x["sd"])
                if x.get("galaxies"):
                    r["vet"] = "background galaxy in the beam: " + "; ".join(
                        "Pan-STARRS extended source %.1f arcsec away, r %.1f" % (gg["sep"], gg["r"] or 0) for gg in x["galaxies"])
                if band == "W3":
                    r["stat"] = x["dx"] / x["sd"]
    w3 = out["W3"]
    core.record_search("hz_stack_wise", "Stacking at habitable-zone hosts", archive="NASA Exoplanet Archive, Gaia DR3, AllWISE",
                           release="pscomppars as of the run; AllWISE", selection="planet hosts within %d pc" % hz_D_MAX,
                           code=os.path.abspath(__file__), entry="hz", params=dict(hz=hz_HZ, n_ctrl=hz_N_CTRL, n_compare=hz_N_COMPARE),
                           n_targets=len(rows), null_summary=dict(W3=w3["null"], W4=out["W4"]["null"]),
                           result="W1-W3 mean excess of habitable-zone hosts %+.4f +- %.4f mag (%d), other hosts %+.4f +- %.4f; "
                                  "mean covering fraction of a 300 K swarm below %s"
                                  % (w3["hz"][0], w3["hz"][1], w3["hz"][2], w3["other"][0], w3["other"][1],
                                     "%.1e" % w3["covering_fraction"] if w3["covering_fraction"] else "-"),
                           ledger="Stacking at habitable-zone hosts")
    core.record_targets("hz_stack_wise", list(rows.values()))
    print("recorded %d hosts" % len(rows))


def hz_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "record"))
    a = p.parse_args()
    {"fetch": hz_fetch, "analyse": hz_analyse, "record": hz_record}[a.mode]()


def hz_cli():
    quiet_warnings()
    hz_main()


# --------------------------------------------------------------------------------------------------------------

pma_DOC = r"""Proper-motion-anomaly companions within 50 pc: Ks-W3 and Ks-W4 excess against neighbours bounds reprocessed light at 200-600 K; excesses re-read in WISE All-Sky.

    python -I seti.py infrared pma
    python -I seti.py infrared pma record"""

pma_HERE = core.RESULTS
pma_VIZ = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
pma_H, pma_KB, pma_CL, pma_SIGMA = 6.62607015e-34, 1.380649e-23, 2.99792458e8, 5.670374419e-8
pma_LSUN, pma_PC, pma_MJ_MSUN = 3.828e26, 3.0857e16, 1047.57
pma_ZP = {"W3": (12.08e-6, 31.674), "W4": (22.19e-6, 8.363)}
pma_KS_ZP = 666.7
pma_TEMPS = (200.0, 300.0, 600.0)


def pma_viz(q):
    b = core.get(pma_VIZ + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q)),
                timeout=300, retries=3)
    return list(csv.DictReader(io.StringIO(b.decode("utf-8", "replace"))))


def pma_lum(m):
    return ms_luminosity(m)


def pma_bb_jy(lam, T, L_sun, d_pc):
    nu = pma_CL / lam
    bnu = 2 * pma_H * nu ** 3 / pma_CL ** 2 / math.expm1(pma_H * nu / (pma_KB * T))
    return bnu * L_sun * pma_LSUN / (4 * pma_SIGMA * T ** 4 * (d_pc * pma_PC) ** 2) / 1e-26


pma_COLS = ("HIP", "GaiaEDR3", "Name", "RAdeg", "DEdeg", "DMS", "W", "Gmag", "Kmag", "e_Kmag", "SpType", "PlxG3", "RUWE",
            "snrPMaH2EG3a", "M1", "M23au", "M25au", "M210au", "M230au", "_RA", "_DE")
pma_RELEASE_DIFF = 0.05  # mag; AllWISE minus All-Sky beyond the sample's 5-95 per cent range
# readings the catalogues alone do not give
pma_FOLLOWUP = {"20591": "known DA white-dwarf companion (56 Per B)",
                "106694": "Ks-W3 +0.16 within the M0 photospheric colour range"}


def pma_sample():
    """Kervella et al. 2022 stars within 50 pc with an anomaly at S/N 5, via VizieR TAP or ASU."""
    try:
        return pma_viz("SELECT %s FROM \"J/A+A/657/A7/tablea1\" WHERE PlxG3 > 20 AND snrPMaH2EG3a > 5"
                       % ", ".join('"%s"' % c if c.startswith("_") else c for c in pma_COLS))
    except OSError:
        from astroquery.vizier import Vizier
        t = Vizier(columns=list(pma_COLS), column_filters={"PlxG3": ">20", "snrPMaH2EG3a": ">5"}, row_limit=-1).get_catalogs(
            "J/A+A/657/A7/tablea1")[0]
        return [{c: ("" if np.ma.is_masked(r[c]) else str(r[c])) for c in pma_COLS if c in t.colnames} for r in t]


def pma_main():
    rows = pma_sample()
    sel = []
    for r in rows:
        ms = [num(r[k]) for k in ("M23au", "M25au", "M210au", "M230au")]
        ms = [m for m in ms if m]
        if not ms:
            continue
        m2 = min(ms) / pma_MJ_MSUN
        if m2 >= 0.3:
            r.update(m2=m2, d_pc=1000.0 / float(r["PlxG3"]), ra=num(r["RAdeg"]) or num(r["_RA"]),
                     dec=num(r["DEdeg"]) or num(r["_DE"]))
            if r["ra"] is not None and r["dec"] is not None:
                sel.append(r)
    print("%d Hipparcos stars within 50 pc with a proper-motion anomaly at S/N 5; %d with a companion of at least 0.3 "
          "solar masses everywhere in 3-30 au" % (len(rows), len(sel)), flush=True)
    pos = [dict(source_id=r["HIP"], ra=r["ra"], dec=r["dec"]) for r in sel]
    wise, allsky = {}, {}
    for cat, into in (("II/328/allwise", wise), ("II/311/wise", allsky)):
        for m in core.xmatch(pos, cat, 6.0):
            if m["source_id"] not in into or float(m["angDist"]) < float(into[m["source_id"]]["angDist"]):
                into[m["source_id"]] = m
    out = []
    for r in sel:
        w = wise.get(r["HIP"])
        if not w:
            continue
        # Kervella's own 2MASS K; AllWISE's associated source can be a fainter neighbour
        ks = num(r["Kmag"]) if num(r["Kmag"]) is not None else num(w.get("Kmag"))
        eks = num(r["e_Kmag"]) or 0.05
        if eks > 0.5:
            # 9.99 marks an unusable K
            ks = None
        gm = num(r["Gmag"])
        rec = dict(hip=r["HIP"], name=r["Name"], gaia=r["GaiaEDR3"], sptype=r["SpType"], d_pc=r["d_pc"], m1=num(r["M1"]),
                   m2=r["m2"], snr=num(r["snrPMaH2EG3a"]), ruwe=num(r["RUWE"]), dms=r["DMS"], ks=ks, e_ks=eks,
                   g_ks=(gm - ks) if gm is not None and ks is not None else None, L2=pma_lum(r["m2"]),
                   qph=w.get("qph"), ccf=w.get("ccf"))
        s = allsky.get(r["HIP"]) or {}
        for band, (lam, f0) in pma_ZP.items():
            mag, err = num(w.get(band + "mag")), num(w.get("e_" + band + "mag"))
            rec[band] = mag
            rec[band + "_err"] = err
            rec["ks_" + band] = (ks - mag) if ks is not None and mag is not None else None
            rec[band + "_allsky"] = num(s.get(band + "mag"))
            rec["ks_" + band + "_allsky"] = (ks - rec[band + "_allsky"]) if ks is not None and rec[band + "_allsky"] is not None else None
            for T in pma_TEMPS:
                rec["%s_pred_%d" % (band, T)] = pma_bb_jy(lam, T, rec["L2"], r["d_pc"])
        out.append(rec)
    # photospheric colours: median of the 15 nearest in G - Ks, AllWISE and All-Sky
    for band in ("W3", "W4"):
        for tag in ("", "_allsky"):
            ok = [o for o in out if o["ks_" + band + tag] is not None and o["g_ks"] is not None]
            jk = np.array([o["g_ks"] for o in ok])
            c = np.array([o["ks_" + band + tag] for o in ok])
            for o in ok:
                near = np.argsort(np.abs(jk - o["g_ks"]))[:16]
                near = [i for i in near if ok[i] is not o][:15]
                base = float(np.median(c[near]))
                spread = float(1.4826 * np.median(np.abs(c[near] - base)))
                o[band + "_excess_mag" + tag] = o["ks_" + band + tag] - base
                if tag:
                    continue
                err = math.sqrt(max(spread, 0.03) ** 2 + (o[band + "_err"] or 0.05) ** 2 + o["e_ks"] ** 2)
                o[band + "_excess_sig"] = o[band + "_excess_mag"] / err
                o[band + "_excess_err"] = err
    for band in ("W3", "W4"):
        lam, f0 = pma_ZP[band]
        for o in out:
            if o.get(band + "_excess_sig") is None:
                continue
            phot = f0 * 10 ** (-0.4 * (o[band] + o[band + "_excess_mag"]))
            meas = f0 * 10 ** (-0.4 * o[band])
            # 3-sigma ceiling when no excess is measured
            ex = max(meas - phot, 0.0) if o[band + "_excess_sig"] >= 3 else \
                meas * (10 ** (0.4 * 3 * o[band + "_excess_err"]) - 1)
            o[band + "_excess_jy"] = ex
            for T in pma_TEMPS:
                o["%s_frac_%d" % (band, T)] = ex / o["%s_pred_%d" % (band, T)]
    for o in out:
        if (o.get("W3_excess_sig") or 0) >= 3 or (o.get("W4_excess_sig") or 0) >= 3:
            o["reading"] = pma_reading(o)
    json.dump(out, open(os.path.join(pma_HERE, "pma_dark.json"), "w"), indent=1)
    lim = [min(o.get("W3_frac_300", 9e9), o.get("W4_frac_300", 9e9)) for o in out]
    lim = [x for x in lim if x < 9e9]
    print("%d with AllWISE photometry; reprocessed fraction of the least companion luminosity at 300 K: median %.2e, "
          "largest %.2e (%d with a usable K); %d below 1 per cent" % (len(out), np.median(lim), np.max(lim), len(lim),
                                                             sum(x < 0.01 for x in lim)))
    exc = [o for o in out if "reading" in o]
    d = [o[b] - o[b + "_allsky"] for o in out for b in ("W3",) if o[b] is not None and o[b + "_allsky"] is not None]
    print("infrared excess at 3 sigma in W3 or W4: %d; AllWISE minus All-Sky W3 over the sample: median %+.3f, 5-95 per "
          "cent %s" % (len(exc), np.median(d), np.percentile(d, [5, 95]).round(3)))
    for o in sorted(exc, key=lambda o: -max(o.get("W3_excess_sig") or 0, o.get("W4_excess_sig") or 0)):
        print("  HIP %s %-18s %-10s %5.1f pc M2 >= %.2f (PMa S/N %.0f, RUWE %s): Ks-W3 excess %+.2f (%.1f sigma), Ks-W4 "
              "excess %+.2f (%.1f sigma); fraction at 300 K %.1e; qph %s ccf %s; %s"
              % (o["hip"], o["name"][:18], (o["sptype"] or "")[:10], o["d_pc"], o["m2"], o["snr"] or 0,
                 "%.2f" % o["ruwe"] if o["ruwe"] else "-", o.get("W3_excess_mag") or 0, o.get("W3_excess_sig") or 0,
                 o.get("W4_excess_mag") or 0, o.get("W4_excess_sig") or 0,
                 min(o.get("W3_frac_300", 9), o.get("W4_frac_300", 9)), o["qph"], o["ccf"], o["reading"] or "unexplained"))


pma_CCF = {"d": "a diffraction spike", "h": "a scattered-light halo", "o": "an optical ghost", "p": "a latent image"}


def pma_reading(o):
    """Explanation of a 3-sigma excess: contamination flag, follow-up reading, or release-dependent photometry; None otherwise."""
    for i, band in ((2, "W3"), (3, "W4")):
        f = (o["ccf"] or "0000")[i:i + 1]
        if f in pma_CCF and (o.get(band + "_excess_sig") or 0) >= 3:
            return "%s flagged as %s (ccf %s)" % (band, pma_CCF[f], o["ccf"])
    if o["hip"] in pma_FOLLOWUP:
        return pma_FOLLOWUP[o["hip"]]
    for band in ("W3", "W4"):
        a, s, xs = o[band], o[band + "_allsky"], o.get(band + "_excess_mag_allsky")
        if (o.get(band + "_excess_sig") or 0) >= 3 and s is not None and xs is not None and abs(a - s) > pma_RELEASE_DIFF \
                and xs < 3 * o[band + "_excess_err"]:
            return ("release-dependent photometry: AllWISE %s %.3f against All-Sky %.3f, where the All-Sky excess is %+.3f mag"
                    % (band, a, s, xs))
    return None


def pma_record():
    out = json.load(open(os.path.join(pma_HERE, "pma_dark.json")))
    lim = [min(o.get("W3_frac_300", 9e9), o.get("W4_frac_300", 9e9)) for o in out]
    lim = [x for x in lim if x < 9e9]
    exc = [o for o in out if "reading" in o]
    left = [o for o in exc if not o["reading"]]
    core.record_search(
        "pma_dark_companions", "Dark companions, longer periods",
        archive="Kervella et al. 2022 (VizieR J/A+A/657/A7), AllWISE, WISE All-Sky", release="Hipparcos-Gaia EDR3",
        selection="Hipparcos stars within 50 pc with a proper-motion anomaly at S/N 5 and a companion of at least 0.3 Msun everywhere in 3-30 au",
        code=os.path.abspath(__file__), entry="pma", n_targets=len(out), n_unit="stars",
        params=dict(m2_min=0.3, snr=5, temps=list(pma_TEMPS), release_diff=pma_RELEASE_DIFF),
        null_summary="the Ks-W3 and Ks-W4 colours of the 15 sample stars nearest in G-Ks",
        sensitivity="3 sigma of the colour against its neighbours, median %.2f per cent of the companion's light at 300 K" % (100 * np.median(lim)),
        ledger="Dark companions",
        result="median bound %.2f per cent of the companion luminosity at 300 K, %d of %d below 1 per cent; %d excesses at 3 sigma: %s%s" % (
            100 * np.median(lim), sum(x < 0.01 for x in lim), len(lim), len(exc),
            "; ".join("%s, %s" % (o["name"], o["reading"]) for o in exc if o["reading"]),
            "; unexplained: %s" % ", ".join(o["name"] for o in left) if left else ""))
    core.clear_targets("pma_dark_companions")
    rows = []
    for o in out:
        if not o.get("gaia"):
            continue
        fr = min(o.get("W3_frac_300", 9e9), o.get("W4_frac_300", 9e9))
        note = "HIP %s %s, M2 >= %.2f Msun at 3-30 au (PMa S/N %.0f), %.1f pc" % (o["hip"], o["name"], o["m2"], o["snr"] or 0, o["d_pc"])
        if "reading" in o:
            note = "Ks-W3 %+.2f (%.1f sigma), Ks-W4 %+.2f (%.1f sigma); %s" % (
                o.get("W3_excess_mag") or 0, o.get("W3_excess_sig") or 0, o.get("W4_excess_mag") or 0, o.get("W4_excess_sig") or 0, note)
        rows.append(dict(target="gaia_dr3:%s" % o["gaia"], upper_limit=fr if fr < 9e9 else None,
                         unit="fraction of the least companion luminosity re-radiated at 300 K", vet=o.get("reading"), note=note))
    core.record_targets("pma_dark_companions", rows)
    print("recorded %d stars" % len(rows))


def pma_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        pma_record()
    else:
        pma_main()


# --------------------------------------------------------------------------------------------------------------

dark_DOC = r"""Gaia DR3 dark companions within 200 pc: W3 and W4 against a 300 K re-emission of a main-sequence companion's luminosity.

    python -I seti.py infrared dark"""

dark_MASS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4])
dark_MG = np.array([11.8, 10.7, 9.6, 8.6, 7.5, 6.6, 5.9, 5.2, 4.6, 3.6, 3.0])
dark_H, dark_KB, dark_CL, dark_SIGMA = 6.62607015e-34, 1.380649e-23, 2.99792458e8, 5.670374419e-8
dark_LSUN, dark_PC = 3.828e26, 3.0857e16
dark_ZP = {"w3": (12.08e-6, 31.674), "w4": (22.19e-6, 8.363)}


def dark_mg_of_mass(m):
    return float(np.interp(math.log(m), np.log(dark_MASS), dark_MG))


def dark_lum(m):
    return m ** 4 if m >= 0.43 else 0.23 * m ** 2.3


def dark_bb_jy(lam, T, L_sun, d_pc):
    nu = dark_CL / lam
    bnu = 2 * dark_H * nu ** 3 / dark_CL ** 2 / (math.exp(dark_H * nu / (dark_KB * T)) - 1.0)
    fnu = bnu * L_sun * dark_LSUN / (4 * dark_SIGMA * T ** 4 * (d_pc * dark_PC) ** 2)
    return fnu / 1e-26


def dark_wise(ra, dec):
    q = ("SELECT w3mpro, w3sigmpro, w3snr, w4mpro, w4sigmpro, w4snr, ph_qual, cc_flags FROM allwise_p3as_psd WHERE "
         "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f))=1" % (ra, dec, 3 / 3600.0))
    u = "https://irsa.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(dict(QUERY=q, FORMAT="csv"))
    for attempt in range(3):
        try:
            rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(u, timeout=120).read().decode())))
            return rows[0] if rows else None
        except Exception:
            pass
    return None


def dark_main():
    rows = list(csv.DictReader(open(core.results("dark_companions.csv"))))
    dark = []
    for r in rows:
        m1, m2 = float(r["m1"] or "nan"), float(r["m2"] or "nan")
        if not (m1 > 0 and m2 > 0):
            continue
        fr = r["fluxratio_upper"] or r["fluxratio"]
        if not fr:
            continue
        expected = 10 ** (-0.4 * (dark_mg_of_mass(m2) - dark_mg_of_mass(m1)))
        if float(fr) < expected / 3.0:
            r.update(expected_fluxratio=expected, d_pc=1000.0 / float(r["parallax"]))
            dark.append(r)
    print("%d systems; %d with an optically dark companion of >= 0.5 Msun" % (len(rows), len(dark)), flush=True)
    with ThreadPoolExecutor(8) as ex:
        ws = list(ex.map(lambda r: dark_wise(float(r["ra"]), float(r["dec"])), dark))
    out = []
    for r, w in zip(dark, ws):
        m2 = float(r["m2"])
        L = dark_lum(m2)
        rec = dict(source_id=r["source_id"], m1=float(r["m1"]), m2=m2, d_pc=round(r["d_pc"], 1), L_pred=round(L, 3),
                   method=r["combination_method"])
        for band, (lam, f0) in dark_ZP.items():
            pred = dark_bb_jy(lam, 300.0, L, r["d_pc"])
            rec[band + "_pred_jy"] = pred
            if w is None:
                rec[band + "_ratio"] = None
                continue
            mag = float(w[band + "mpro"]) if w[band + "mpro"] else float("nan")
            snr = float(w[band + "snr"]) if w[band + "snr"] else 0.0
            meas = f0 * 10 ** (-0.4 * mag)
            rec[band + "_meas_jy"] = meas
            rec[band + "_detected"] = bool(snr >= 3)
            rec[band + "_ratio"] = meas / pred
        out.append(rec)
    json.dump(out, open(core.results("dark_companion_ds.json"), "w"), indent=0)
    ok = [o for o in out if o.get("w3_ratio") is not None]
    print("%d dark companions with an AllWISE counterpart" % len(ok))
    ok.sort(key=lambda o: -(o["w3_ratio"] or 0))
    print("highest measured/predicted W3 (1.0 = the whole companion reprocessed at 300 K):")
    for o in ok[:15]:
        print("  Gaia DR3 %s  M2 %.2f  %.0f pc  W3 ratio %.3f (%s)  W4 ratio %.3f (%s)"
              % (o["source_id"], o["m2"], o["d_pc"], o["w3_ratio"], "det" if o["w3_detected"] else "limit",
                 o["w4_ratio"], "det" if o["w4_detected"] else "limit"))
    lim = np.array([o["w3_ratio"] for o in ok])
    print("median W3 ratio %.4f; %d of %d below 0.1" % (np.median(lim), int(np.sum(lim < 0.1)), len(lim)))


def dark_cli():
    dark_main()


# --------------------------------------------------------------------------------------------------------------

enclosed_DOC = r"""Fully enclosed stars: moving, cold AllWISE point sources (W3 below 8, W2-W3 above 1.5) without Gaia counterparts, vetted for young stars and clouds.

    python -I seti.py infrared enclosed
    python -I seti.py infrared enclosed record"""

enclosed_ROOT = os.environ.get("ENCLOSED_ROOT", core.data("enclosed"))
enclosed_YOUNG = ("Y*O", "Y*?", "TT*", "TT?", "Or*", "Ae*", "HH", "out", "smm", "cor", "MoC", "DNe", "glb", "bub", "HII", "C?*",
         "Em*", "FU*", "FIR")


def enclosed_f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def enclosed_cached(name, fn):
    p = os.path.join(enclosed_ROOT, name)
    if os.path.exists(p):
        return json.load(open(p))
    r = fn()
    json.dump(r, open(p, "w"))
    return r


def enclosed_main():
    os.makedirs(enclosed_ROOT, exist_ok=True)
    src = enclosed_cached("allwise_cold.json", lambda: core.tap_async(
        "irsa", "SELECT designation, ra, dec, w1mpro, w1sigmpro, w1snr, w2mpro, w2sigmpro, w2snr, w3mpro, w3sigmpro, "
                "w3snr, w4mpro, w4snr, ph_qual, cc_flags, ext_flg, nb, na, pmra, sigpmra, pmdec, sigpmdec, glat FROM "
                "allwise_p3as_psd WHERE w3mpro < 8 AND w3snr > 10 AND w2mpro - w3mpro > 1.5 AND ext_flg = 0"))
    pts = [dict(source_id=x["designation"], ra=float(x["ra"]), dec=float(x["dec"])) for x in src]

    def xm(cat):
        out = []
        for k in range(0, len(pts), 10000):
            out += core.xmatch(pts[k:k + 10000], cat, 3.0)
        return out
    cw = enclosed_cached("catwise.json", lambda: xm("II/365/catwise"))
    ga = enclosed_cached("gaia.json", lambda: xm("I/355/gaiadr3"))
    best = {}
    for x in cw:
        if x["source_id"] not in best or enclosed_f(x["angDist"]) < enclosed_f(best[x["source_id"]]["angDist"]):
            best[x["source_id"]] = x
    ngaia = Counter(x["source_id"] for x in ga)
    moving = []
    for s in src:
        c = best.get(s["designation"])
        if not c:
            continue
        pm = math.hypot(enclosed_f(c["pmRA"]), enclosed_f(c["pmDE"]))
        e = math.hypot(enclosed_f(c["e_pmRA"]), enclosed_f(c["e_pmDE"])) / math.sqrt(2)
        if pm > 0.04 and pm > 5 * e:
            moving.append(dict(s, pm=pm, e_pm=e, ng=ngaia.get(s["designation"], 0)))
    dark = [s for s in moving if s["ng"] == 0]
    clean = [s for s in dark if s["cc_flags"][:3] == "000" and enclosed_f(s["w2mpro"]) > 7 and enclosed_f(s["w1mpro"]) > 8
             and s["ph_qual"][1] == "A"]
    print("%d AllWISE point sources with W3 < 8 (S/N > 10) and W2-W3 > 1.5; %d in CatWISE2020; %d moving above "
          "40 mas/yr at 5 sigma; %d without a Gaia source within 3 arcsec; %d with clean flags below saturation"
          % (len(src), len(best), len(moving), len(dark), len(clean)))
    print("  Galactic latitude of the clean ones: %s" % dict(Counter(int(abs(enclosed_f(s["glat"])) // 5 * 5) for s in clean)))
    left = []
    for s in clean:
        q = ("SELECT b.otype FROM basic AS b WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', %f, %f, 0.5)) "
             "= 1" % (float(s["ra"]), float(s["dec"])))
        types = Counter(x["otype"] for x in core.simbad(q))
        s["young_30arcmin"] = sum(v for k, v in types.items() if k in enclosed_YOUNG)
        if s["young_30arcmin"] >= 5:
            continue
        left.append(s)
    print("  away from star-forming clouds (fewer than 5 young stars, cores or clouds in SIMBAD within 30 arcmin): %d, "
          "%d of them within 5 deg of the Galactic plane" % (len(left), sum(1 for s in left if abs(enclosed_f(s["glat"])) < 5)))
    for s in left:
        print("    %s b %+.1f W1 %s W2 %s W3 %s W4 %s pm %.3f +- %.3f arcsec/yr, %d young objects within 30 arcmin"
              % (s["designation"], enclosed_f(s["glat"]), s["w1mpro"], s["w2mpro"], s["w3mpro"], s["w4mpro"], s["pm"],
                 s["e_pm"], s["young_30arcmin"]))
    json.dump(dict(moving=len(moving), dark=len(dark), clean=clean, left=left), open(os.path.join(enclosed_ROOT, "result.json"),
                                                                                     "w"), default=str)


def enclosed_record():
    """Record one row per clean, moving, optically dark source."""
    r = json.load(open(os.path.join(enclosed_ROOT, "result.json")))
    core.record_search(
        "enclosed_stars", "Fully enclosed stars: cold moving infrared sources without optical counterparts",
        archive="AllWISE (IRSA asynchronous TAP), CatWISE2020 (CDS II/365/catwise), Gaia DR3, SIMBAD",
        release="AllWISE, CatWISE2020, Gaia DR3", selection="AllWISE point sources with W3 < 8 at S/N 10 and W2-W3 > 1.5",
        code="seti.py infrared enclosed", params=dict(w3_max=8, w2w3_min=1.5, pm_min_mas=40, pm_sigma=5, gaia_radius_arcsec=3,
                                              young_within_30arcmin=5),
        n_targets=len(r["clean"]),
        null_summary="%d move at 5 sigma, %d have no Gaia source, %d have clean flags below saturation, and %d of them "
                     "lie away from young stars, cores and clouds" % (r["moving"], r["dark"], len(r["clean"]),
                                                                      len(r["left"])),
        sensitivity="a 300 K shell around 0.003-1 L_sun within 100 pc (W3 below 8, W2 unsaturated, CatWISE motion above "
                    "40 mas/yr)",
        result="no fully enclosed star" if not r["left"] else "%d candidates" % len(r["left"]), ledger="Fully enclosed stars")
    core.clear_targets("enclosed_stars")
    core.record_targets("enclosed_stars", [
        dict(target="allwise:%s" % s["designation"], stat=float(s["pm"]), unit="CatWISE proper motion, arcsec/yr",
             vet="star_forming_region" if s["young_30arcmin"] >= 5 else None,
             note="%d young stars, cores or clouds within 30 arcmin" % s["young_30arcmin"]) for s in r["clean"]])


def enclosed_cli():
    quiet_warnings()
    if sys.argv[1:] == ["record"]:
        enclosed_record()
    else:
        enclosed_main()


# --------------------------------------------------------------------------------------------------------------

cold_DOC = r"""Cold shells (30-100 K) at stars within 50 pc from AKARI FIS: modified black-body fits and chance rates from source density.

    python -I seti.py infrared cold"""


cold_HERE = core.RESULTS
cold_BP_RP = np.array([0.0, 0.5, 0.82, 1.2, 1.8, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
cold_BC_G = np.array([-0.1, 0.05, 0.08, -0.05, -0.35, -0.8, -1.1, -1.4, -1.8, -2.3, -2.8])
# dwarf sequence M_G against BP-RP (Pecaut and Mamajek)
cold_SEQ_C = np.array([0.4, 0.6, 0.82, 1.0, 1.2, 1.5, 1.84, 2.2, 2.5, 2.8, 3.2, 3.6, 4.0])
cold_SEQ_M = np.array([2.6, 3.6, 4.7, 5.6, 6.3, 7.1, 8.0, 9.1, 10.0, 10.9, 12.0, 13.2, 14.4])
cold_LSUN, cold_PC, cold_H, cold_KB, cold_C = 3.828e33, 3.0857e18, 6.626e-27, 1.381e-16, 2.998e10
cold_BANDS = {"S65": 65e-4, "S90": 90e-4, "S140": 140e-4, "S160": 160e-4}


def cold_xmatch(rows, cat, radius):
    """CDS XMatch of the stars against a VizieR catalogue."""
    return core.xmatch(rows, cat, radius)


def cold_mbb(lam_cm, T, beta=1.0):
    nu = cold_C / lam_cm
    return nu ** (3 + beta) / (math.exp(cold_H * nu / (cold_KB * T)) - 1.0)


def cold_fit(fluxes):
    """Temperature and normalisation of a beta 1 modified black body through the bands (Jy)."""
    bands = [(cold_BANDS[k], v) for k, v in fluxes.items()]
    best = None
    for T in np.arange(15.0, 300.0, 1.0):
        model = np.array([cold_mbb(lam, T) for lam, _ in bands])
        obs = np.array([v for _, v in bands])
        a = np.dot(model, obs) / np.dot(model, model)
        chi = np.sum((obs - a * model) ** 2 / obs ** 2)
        if best is None or chi < best[0]:
            best = (chi, T, a)
    return best[1], best[2]


def cold_lir(T, a, d_cm):
    """Luminosity (erg/s) of the fitted modified black body at distance d."""
    nus = np.logspace(10.5, 14.5, 4000)
    fnu = a * nus ** 4 / (np.exp(cold_H * nus / (cold_KB * T)) - 1.0) * 1e-23
    return 4 * math.pi * d_cm ** 2 * np.trapezoid(fnu, nus)


def cold_main():
    stars = core.gaia("SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp, ruwe FROM gaiadr3.gaia_source "
                          "WHERE parallax > 20 AND bp_rp BETWEEN 0.4 AND 4.0 AND phot_g_mean_mag < 15")
    keep = []
    for s in stars:
        plx, g, col = float(s["parallax"]), float(s["phot_g_mean_mag"]), float(s["bp_rp"])
        mg = g + 5 * math.log10(plx / 100.0)
        if abs(mg - float(np.interp(col, cold_SEQ_C, cold_SEQ_M))) < 1.5:
            keep.append(s)
    print("%d main-sequence Gaia DR3 stars within 50 pc" % len(keep), flush=True)
    m = cold_xmatch(keep, "II/298/fis", 20.0)
    byid = {s["source_id"]: s for s in keep}
    out = []
    for x in m:
        s = byid[x["source_id"]]
        fl = {}
        for k in cold_BANDS:
            q = x.get("q_" + k) or x.get("q_" + k.replace("S", "F")) or ""
            v = x.get(k)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v > 0 and (q in ("3", "") or q.isdigit() and int(q) >= 3):
                fl[k] = v
        if "S90" not in fl:
            continue
        plx = float(s["parallax"])
        d = 1000.0 / plx * cold_PC
        mg = float(s["phot_g_mean_mag"]) + 5 * math.log10(plx / 100.0)
        lb = cold_LSUN * 10 ** (-0.4 * (mg + float(np.interp(float(s["bp_rp"]), cold_BP_RP, cold_BC_G)) - 4.74))
        if len(fl) >= 2:
            T, a = cold_fit(fl)
        else:
            T = 50.0
            a = fl["S90"] / cold_mbb(cold_BANDS["S90"], T)
        f = cold_lir(T, a, d) / lb
        out.append(dict(source_id=s["source_id"], g=float(s["phot_g_mean_mag"]), bp_rp=float(s["bp_rp"]),
                        d_pc=round(1000.0 / plx, 1), sep=float(x["angDist"]), fluxes=fl, T=float(T), frac=f,
                        akari=x.get("AKARI-FIS") or x.get("AKARI")))
    # chance alignments from FIS density within 1 deg times the match area
    print("%d stars with a 90 um detection within 20 arcsec" % len(out))
    out.sort(key=lambda o: -o["frac"])
    for o in out[:25]:
        sb = core.simbad("SELECT b.main_id, b.otype, b.sp_type FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
                             "WHERE i.id = 'Gaia DR3 %s'" % o["source_id"])
        o["simbad"] = sb[0] if sb else {}
        print("  %s (%s, %s) %.1f pc G %.2f: %.0f K, fraction %.1e, sep %.1f arcsec, %s"
              % (o["simbad"].get("main_id", o["source_id"]), o["simbad"].get("otype", "?"),
                 o["simbad"].get("sp_type", "?"), o["d_pc"], o["g"], o["T"], o["frac"], o["sep"],
                 ", ".join("%s %.2f Jy" % kv for kv in o["fluxes"].items())), flush=True)
    json.dump(out, open(os.path.join(cold_HERE, "cold_shells.json"), "w"), indent=0)


def cold_cli():
    quiet_warnings()
    cold_main()


# --------------------------------------------------------------------------------------------------------------

colder_DOC = r"""Colder shells (10-30 K) at stars within 50 pc from Planck PCCS2, PGCC and AKARI 140/160 um, against shifted-position chance matches.

    python -I seti.py infrared colder
    python -I seti.py infrared colder record"""

colder_BANDS = {"S140": 140e-4, "S160": 160e-4, "P857": 349.8e-4, "P545": 550.1e-4, "P353": 849.3e-4}
colder_CATS = {"pccs857": ("J/A+A/594/A26/pccs857", 150.0), "pccs545": ("J/A+A/594/A26/pccs545", 150.0),
               "pccs353": ("J/A+A/594/A26/pccs353", 150.0), "pgcc": ("J/A+A/594/A28/pgcc", 150.0),
               "fis": ("II/298/fis", 20.0)}


def colder_stars():
    stars = core.gaia("SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp, ruwe FROM gaiadr3.gaia_source "
                          "WHERE parallax > 20 AND bp_rp BETWEEN 0.4 AND 4.0 AND phot_g_mean_mag < 15")
    keep = []
    for s in stars:
        plx, g, col = float(s["parallax"]), float(s["phot_g_mean_mag"]), float(s["bp_rp"])
        mg = g + 5 * math.log10(plx / 100.0)
        if abs(mg - float(np.interp(col, cold_SEQ_C, cold_SEQ_M))) < 1.5:
            keep.append(s)
    return keep


def colder_flux(x, cat):
    """Far-infrared fluxes (Jy) of a match from PCCS2, PGCC and FIS."""
    out = {}

    def f(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) and v > 0 else None
    if cat.startswith("pccs"):
        band = cat[4:]
        v = f(x.get("DetFlux" + band))
        if v:
            out["P" + band] = v / 1000.0
    elif cat == "pgcc":
        for b in ("857", "545", "353"):
            v = f(x.get("F%sC" % b))
            if v:
                out["P" + b] = v
    else:
        for k in ("S140", "S160"):
            q = x.get("q_" + k) or ""
            v = f(x.get(k))
            if v and (q == "" or (q.isdigit() and int(q) >= 3)):
                out[k] = v
    return out


def colder_fit(fl, beta=1.5):
    bands = [(colder_BANDS[k], v) for k, v in fl.items()]
    best = None
    for T in np.arange(6.0, 60.0, 0.5):
        model = np.array([cold_mbb(lam, T, beta) for lam, _ in bands])
        obs = np.array([v for _, v in bands])
        a = np.dot(model, obs) / np.dot(model, model)
        chi = np.sum((obs - a * model) ** 2 / obs ** 2)
        if best is None or chi < best[0]:
            best = (chi, T, a)
    return best[1], best[2]


def colder_lir(T, a, d_cm, beta=1.5):
    nus = np.logspace(10.0, 14.0, 4000)
    fnu = a * nus ** (3 + beta) / (np.exp(cold_H * nus / (cold_KB * T)) - 1.0) * 1e-23
    return 4 * math.pi * d_cm ** 2 * np.trapezoid(fnu, nus)


def colder_main():
    keep = colder_stars()
    print("%d main-sequence Gaia DR3 stars within 50 pc" % len(keep), flush=True)
    byid = {s["source_id"]: s for s in keep}
    offsets = [(0.5 * math.cos(k * math.pi / 4), 0.5 * math.sin(k * math.pi / 4)) for k in range(8)]
    found, chance = {}, {}
    for cat, (vz, rad) in colder_CATS.items():
        m = core.xmatch(keep, vz, rad)
        for x in m:
            fl = colder_flux(x, cat)
            if not fl:
                continue
            o = found.setdefault(x["source_id"], dict(source_id=x["source_id"], fluxes={}, cats={}))
            o["cats"][cat] = round(float(x["angDist"]), 1)
            for k, v in fl.items():
                o["fluxes"].setdefault(k, v)
        n_sh = []
        for dra, dde in offsets:
            sh = [dict(source_id=s["source_id"], ra=(float(s["ra"]) + dra / max(math.cos(math.radians(float(s["dec"]))), 0.05)) % 360.0,
                       dec=max(-89.99, min(89.99, float(s["dec"]) + dde))) for s in keep]
            n_sh.append(sum(1 for x in core.xmatch(sh, vz, rad) if colder_flux(x, cat)))
        chance[cat] = (float(np.mean(n_sh)), float(np.std(n_sh)))
        print("  %s within %.0f arcsec: %d stars against %.1f +- %.1f at shifted positions"
              % (cat, rad, sum(1 for o in found.values() if cat in o["cats"]), chance[cat][0], chance[cat][1]), flush=True)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    out = []
    for sid, o in found.items():
        s = byid[sid]
        plx = float(s["parallax"])
        d = 1000.0 / plx * cold_PC
        mg = float(s["phot_g_mean_mag"]) + 5 * math.log10(plx / 100.0)
        lb = cold_LSUN * 10 ** (-0.4 * (mg + float(np.interp(float(s["bp_rp"]), cold_BP_RP, cold_BC_G)) - 4.74))
        if len(o["fluxes"]) >= 2:
            T, a = colder_fit(o["fluxes"])
        else:
            T = 15.0
            k, v = next(iter(o["fluxes"].items()))
            a = v / cold_mbb(colder_BANDS[k], T, 1.5)
        c = SkyCoord(float(s["ra"]) * u.deg, float(s["dec"]) * u.deg)
        o.update(g=float(s["phot_g_mean_mag"]), bp_rp=float(s["bp_rp"]), d_pc=round(1000.0 / plx, 1), T=float(T),
                 frac=float(colder_lir(T, a, d) / lb), glat=round(float(c.galactic.b.deg), 1))
        out.append(o)
    out.sort(key=lambda o: -o["frac"])
    for o in out[:30]:
        sb = core.simbad("SELECT b.main_id, b.otype, b.sp_type FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
                             "WHERE i.id = 'Gaia DR3 %s'" % o["source_id"])
        o["simbad"] = sb[0] if sb else {}
        print("  %s (%s, %s) %.1f pc, b %+.1f: %.1f K, fraction %.1e; %s; %s"
              % (o["simbad"].get("main_id", o["source_id"]), o["simbad"].get("otype", "?"), o["simbad"].get("sp_type", "?"),
                 o["d_pc"], o["glat"], o["T"], o["frac"], ", ".join("%s %.2f Jy" % kv for kv in o["fluxes"].items()),
                 ", ".join("%s %.0f\"" % kv for kv in o["cats"].items())), flush=True)
    json.dump(dict(n_stars=len(keep), chance=chance, matches=out), open(os.path.join(cold_HERE, "colder_shells.json"), "w"),
              indent=0)


def colder_record():
    """Record every star searched with the light fraction a match would carry."""
    d = json.load(open(os.path.join(cold_HERE, "colder_shells.json")))
    stars = colder_stars()
    m = {o["source_id"]: o for o in d["matches"]}
    ch = d["chance"]
    n_by = {c: sum(1 for o in d["matches"] if c in o["cats"]) for c in ch}
    core.record_search(
        "colder_shells", "Cold shells, 10-30 K", archive="Planck PCCS2 857/545/353 GHz, Planck PGCC, AKARI FIS BSC",
        selection="Gaia DR3 main-sequence stars within 50 pc", code=os.path.abspath(__file__), entry="colder",
        n_targets=len(stars), params=dict(radius_planck_arcsec=150, radius_fis_arcsec=20, beta=1.5),
        null_summary="the same match at eight positions shifted by 0.5 deg: " + ", ".join(
            "%s %.1f +- %.1f" % (c, v[0], v[1]) for c, v in ch.items()),
        sensitivity="about 1 Jy at 350 um: a 10 K shell carrying a solar luminosity out to 50 pc",
        ledger="Cold shells, 10-30 K",
        result="matches at the chance rate in every catalogue (%s); %d matched stars would need more than their own "
               "luminosity, cold clumps and cirrus in the beam, mostly within 20 deg of the plane"
               % (", ".join("%s %d" % (c, n_by[c]) for c in ch), sum(1 for o in d["matches"] if o["frac"] > 1)))
    core.clear_targets("colder_shells")
    rows = []
    for s_ in stars:
        o = m.get(s_["source_id"])
        if o is None:
            rows.append(dict(target="gaia_dr3:%s" % s_["source_id"], stat=None, unit="fraction of the star's light at 10-30 K"))
            continue
        vet = ("more than the star's luminosity: a cloud in the beam" if o["frac"] > 1 else
               "a cold clump in the beam" if "pgcc" in o["cats"] else "a compact source at the chance rate")
        rows.append(dict(target="gaia_dr3:%s" % s_["source_id"], stat=o["frac"], unit="fraction of the star's light at 10-30 K",
                         vet=vet, note="%.1f K; %s; %s; b %+.1f" % (o["T"], ", ".join("%s %.2f Jy" % kv for kv in o["fluxes"].items()),
                                                                    ", ".join("%s %.0f arcsec" % kv for kv in o["cats"].items()), o["glat"])))
    core.record_targets("colder_shells", rows)
    print("recorded %d stars, %d matched" % (len(rows), len(m)))


def colder_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        colder_record()
    else:
        colder_main()


# --------------------------------------------------------------------------------------------------------------

massless_DOC = r"""Light without mass: Gaia photocentre orbits within 100 pc whose dark-companion K the HARPS and HIRES velocities exclude, then WISE colours.

    python -I seti.py infrared massless
    python -I seti.py infrared massless record"""

massless_TYPES = ("Orbital", "OrbitalTargetedSearch", "OrbitalTargetedSearchValidated", "OrbitalAlternative",
                  "OrbitalAlternativeValidated")
massless_AU_DAY = 1731.456836


def massless_campbell(A, B, F, G):
    """Semi-major axis (mas) and inclination (deg) from Thiele-Innes constants."""
    p = math.hypot(A + G, B - F)
    q = math.hypot(A - G, B + F)
    a = 0.5 * (p + q)
    cosi = (p - q) / (p + q) if p + q > 0 else 1.0
    return a, math.degrees(math.acos(max(-1.0, min(1.0, cosi))))


def massless_nu(t, P, tp, e):
    M = 2 * np.pi * ((t - tp) / P % 1.0)
    E = M.copy()
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def massless_kfit(t, v, ev, P, e, tp):
    """Linear fit of offset, K cos w and K sin w at fixed periastron: K, error, jitter."""
    nu = massless_nu(t, P, tp, e)
    X = np.vstack([np.ones_like(t), np.cos(nu) + e, -np.sin(nu)]).T
    jit = 0.0
    for _ in range(3):
        w = 1.0 / (ev ** 2 + jit ** 2)
        Xw = X * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(Xw, v * np.sqrt(w), rcond=None)
        res = v - X @ coef
        dof = max(len(t) - 3, 1)
        jit = math.sqrt(max(np.sum(res ** 2) / dof - np.mean(ev ** 2), 0.0))
    cov = np.linalg.pinv(Xw.T @ Xw)
    C, S = coef[1], coef[2]
    K = math.hypot(C, S)
    sk = math.sqrt(max((C ** 2 * cov[1, 1] + S ** 2 * cov[2, 2] + 2 * C * S * cov[1, 2]) / max(K ** 2, 1e-12), 0.0))
    return K, sk, jit


def massless_main():
    from astroquery.vizier import Vizier
    q = ("SELECT o.source_id, o.nss_solution_type, o.period, o.period_error, o.t_periastron, o.eccentricity, "
         "o.a_thiele_innes, o.b_thiele_innes, o.f_thiele_innes, o.g_thiele_innes, o.parallax AS plx, o.significance, "
         "g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp, m.m1, m.m2, m.fluxratio, m.combination_method "
         "FROM gaiadr3.nss_two_body_orbit AS o JOIN gaiadr3.gaia_source AS g ON g.source_id = o.source_id "
         "LEFT JOIN gaiadr3.binary_masses AS m ON m.source_id = o.source_id "
         "WHERE o.parallax > 10 AND o.nss_solution_type IN (%s)" % ",".join("'%s'" % t for t in massless_TYPES))
    orbits = core.gaia(q)
    print("%d Gaia DR3 astrometric orbits within 100 pc" % len(orbits), flush=True)
    lists = []
    vz = Vizier(columns=["**"], row_limit=-1, timeout=600)
    t1 = vz.get_catalogs("J/A+A/636/A74/list")[0]
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c1 = SkyCoord(t1["RAJ2000"], t1["DEJ2000"], unit=(u.hourangle, u.deg))
    lists.append(("harps", [str(n) for n in t1["Name"]], c1))
    t2 = vz.get_catalogs("J/AJ/153/208/targets")[0]
    ra2, de2 = np.array(t2["_RA"], float), np.array(t2["_DE"], float)
    fin = np.isfinite(ra2) & np.isfinite(de2)
    c2 = SkyCoord(ra2[fin] * u.deg, de2[fin] * u.deg)
    lists.append(("hires", [str(n) for n, k in zip(t2["Name"], fin) if k], c2))
    co = SkyCoord(np.array([float(o["ra"]) for o in orbits]) * u.deg, np.array([float(o["dec"]) for o in orbits]) * u.deg)
    matched = {}
    for inst, names, cc in lists:
        idx, sep, _ = co.match_to_catalog_sky(cc)
        for i, (j, s) in enumerate(zip(idx, sep.arcsec)):
            if s < 6.0:
                matched.setdefault(i, []).append((inst, names[j]))
    print("%d orbits with HARPS or HIRES velocities" % len(matched), flush=True)
    out = []
    for i, srcs in matched.items():
        o = orbits[i]
        P, e, tp = float(o["period"]), float(o["eccentricity"] or 0.0), float(o["t_periastron"]) + 2457389.0
        a0, inc = massless_campbell(*(float(o[k + "_thiele_innes"]) for k in "abfg"))
        plx = float(o["plx"])
        a1 = a0 / plx
        Kp = 2 * math.pi * a1 * math.sin(math.radians(inc)) / (P * math.sqrt(1 - e * e)) * massless_AU_DAY * 1000.0
        t, v, ev = [], [], []
        for inst, name in srcs:
            if inst == "harps":
                r = vz.query_constraints(catalog="J/A+A/636/A74/rvbank", Name=name)
                if r:
                    r = r[0]
                    good = np.isfinite(np.array(r["RVdrs"], float))
                    t += list(np.array(r["BJD"], float)[good])
                    vv = np.array(r["RVdrs"], float)[good]
                    v += list(vv - np.median(vv) if len(vv) else vv)
                    ev += list(np.array(r["e_RVdrs"], float)[good])
            else:
                r = vz.query_constraints(catalog="J/AJ/153/208/table1", Name=name)
                if r:
                    r = r[0]
                    vv = np.array(r["RV"], float)
                    t += list(np.array(r["BJD"], float))
                    v += list(vv - np.median(vv))
                    ev += list(np.array(r["e_RV"], float))
        t, v, ev = np.array(t), np.array(v), np.array(ev)
        if len(t) < 8:
            continue
        ev = np.where(np.isfinite(ev) & (ev > 0), ev, np.nanmedian(ev[ev > 0]) if np.any(ev > 0) else 2.0)
        K0, s0, jit = massless_kfit(t, v, ev, P, e, tp)
        kmax = 0.0
        for ph in np.linspace(0, 1, 48, endpoint=False):
            K, sk, _ = massless_kfit(t, v, ev, P, e, tp + ph * P)
            kmax = max(kmax, K + 3 * sk)
        rec = dict(source_id=o["source_id"], type=o["nss_solution_type"], P=P, e=e, a0_mas=a0, inc=inc, plx=plx,
                   K_pred=Kp, K_at_gaia_phase=K0, sK=s0, K_max3=kmax, jitter=jit, n_rv=len(t),
                   span_d=float(t.max() - t.min()), rv=[s[0] + ":" + s[1] for s in srcs], g=float(o["phot_g_mean_mag"]),
                   m1=o.get("m1"), m2=o.get("m2"), fluxratio=o.get("fluxratio"), method=o.get("combination_method"),
                   ra=float(o["ra"]), dec=float(o["dec"]))
        out.append(rec)
    excl = [r for r in out if r["K_max3"] < 0.3 * r["K_pred"] and r["span_d"] > 0.5 * r["P"]]
    print("%d orbits with 8 or more velocities; %d whose velocities allow under 0.3 of the predicted K at any phase"
          % (len(out), len(excl)), flush=True)
    for r in excl:
        w = dark_wise(r["ra"], r["dec"])
        r["wise"] = w
        sb = core.simbad("SELECT b.main_id, b.otype, b.sp_type FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
                             "WHERE i.id = 'Gaia DR3 %s'" % r["source_id"])
        r["simbad"] = sb[0] if sb else {}
    for r in sorted(out, key=lambda r: r["K_max3"] / max(r["K_pred"], 1e-9))[:25]:
        print("  %s %s (%s): P %.1f d, a0 %.2f mas, i %.0f, predicted K %.0f m/s; velocities %d over %.0f d allow %.1f m/s "
              "(%.2f of predicted), K at Gaia's phase %.1f +- %.1f, jitter %.1f; %s"
              % (r.get("simbad", {}).get("main_id", "Gaia DR3 " + str(r["source_id"])), r["type"], ",".join(r["rv"]), r["P"],
                 r["a0_mas"], r["inc"], r["K_pred"], r["n_rv"], r["span_d"], r["K_max3"], r["K_max3"] / r["K_pred"],
                 r["K_at_gaia_phase"], r["sK"], r["jitter"],
                 ("W3 %s W4 %s" % (r["wise"].get("w3mpro"), r["wise"].get("w4mpro"))) if r.get("wise") else ""), flush=True)
    json.dump(out, open(core.results("massless_orbits.json"), "w"), indent=0, default=str)


wdcold_DOC = r"""Cold W3/W4 excess at white dwarfs within 100 pc with W1 and W2 photospheric, against shifted-position chance matches.

    python -I seti.py infrared wdcold
    python -I seti.py infrared wdcold record"""


def wdcold_bb_ratio(lam_um, teff, lam_ref_um=0.62):
    """Black-body F_nu(lam)/F_nu(lam_ref)."""
    def bnu(lam):
        nu = 2.998e14 / lam
        x = 6.626e-34 * nu / (1.381e-23 * teff)
        return nu ** 3 / math.expm1(min(x, 700.0))
    return bnu(lam_um) / bnu(lam_ref_um)


def wdcold_main():
    from astroquery.vizier import Vizier
    v = Vizier(columns=["GaiaEDR3", "RA_ICRS", "DE_ICRS", "Plx", "pmRA", "pmDE", "Pwd", "Gmag", "TeffH", "TeffHe"],
               column_filters={"Plx": ">10", "Pwd": ">0.75"}, row_limit=-1, timeout=600)
    t = v.get_catalogs("J/MNRAS/508/3877/maincat")[0]
    rows = []
    for r in t:
        try:
            ra, de = float(r["RA_ICRS"]), float(r["DE_ICRS"])
            pmra, pmde = float(r["pmRA"] or 0), float(r["pmDE"] or 0)
        except (TypeError, ValueError):
            continue
        ra10 = ra - pmra / 3.6e6 * 5.5 / max(math.cos(math.radians(de)), 0.05)
        de10 = de - pmde / 3.6e6 * 5.5
        teff = r["TeffH"] if r["TeffH"] is not np.ma.masked else r["TeffHe"]
        rows.append(dict(source_id=str(r["GaiaEDR3"]), ra=ra10, dec=de10, plx=float(r["Plx"]),
                         g=float(r["Gmag"]) if r["Gmag"] is not np.ma.masked else np.nan,
                         teff=float(teff) if teff is not np.ma.masked else np.nan))
    print("%d white dwarfs within 100 pc" % len(rows), flush=True)

    def match(rr):
        m = core.xmatch(rr, "II/328/allwise", 3.0)
        best = {}
        for x in m:
            if x["source_id"] not in best or float(x["angDist"]) < float(best[x["source_id"]]["angDist"]):
                best[x["source_id"]] = x
        return best
    w = match(rows)
    shifted = [dict(r, ra=r["ra"] + 30 / 3600.0 / max(math.cos(math.radians(r["dec"])), 0.05)) for r in rows]
    w_null = match(shifted)
    zp = {"W1": (3.4, 309.54), "W2": (4.6, 171.787), "W3": (12.0, 31.674), "W4": (22.0, 8.363)}

    def judge(r, x):
        out = {}
        g_jy = 3631.0 * 10 ** (-0.4 * r["g"]) if np.isfinite(r["g"]) else np.nan
        for b, (lam, f0) in zp.items():
            try:
                mag, err = float(x[b + "mag"]), float(x["e_" + b + "mag"])
            except (TypeError, ValueError, KeyError):
                continue
            f = f0 * 10 ** (-0.4 * mag)
            ef = f * 0.921 * err
            pred = g_jy * wdcold_bb_ratio(lam, r["teff"]) if np.isfinite(r["teff"]) else np.nan
            out[b] = dict(f=f, ef=ef, pred=pred, z=(f - pred) / ef if ef > 0 and np.isfinite(pred) else np.nan)
        return out
    cands, n_w3 = [], 0
    for r in rows:
        x = w.get(r["source_id"])
        if x is None:
            continue
        j = judge(r, x)
        cold = any(j.get(b, {}).get("z", -1) >= 3 for b in ("W3", "W4"))
        warm = any(j.get(b, {}).get("z", -1) >= 3 for b in ("W1", "W2"))
        n_w3 += int("W3" in j and j["W3"]["f"] > 3 * j["W3"]["ef"])
        if cold and not warm:
            cands.append(dict(r, sep=float(x["angDist"]), bands={b: {k: float(v) for k, v in d.items()} for b, d in j.items()},
                              allwise=x.get("AllWISE")))
    null_cold = 0
    for r in shifted:
        x = w_null.get(r["source_id"])
        if x is None:
            continue
        j = judge(r, x)
        if any(j.get(b, {}).get("z", -1) >= 3 for b in ("W3", "W4")) and not any(j.get(b, {}).get("z", -1) >= 3 for b in ("W1", "W2")):
            null_cold += 1
    print("%d matched to AllWISE within 3 arcsec, %d with a W3 detection; cold excess without near-infrared excess: %d, "
          "against %d at positions shifted by 30 arcsec" % (len(w), n_w3, len(cands), null_cold), flush=True)
    for c in sorted(cands, key=lambda c: -max(c["bands"].get("W3", {}).get("z", 0), c["bands"].get("W4", {}).get("z", 0)))[:25]:
        sb = core.simbad("SELECT b.main_id, b.otype FROM basic AS b JOIN ident AS i ON i.oidref = b.oid WHERE i.id = 'Gaia DR3 %s'"
                             % c["source_id"])
        c["simbad"] = sb[0] if sb else {}
        print("  %s (%s) %.1f pc, %.0f K, sep %.1f\": %s" % (c["simbad"].get("main_id", c["source_id"]), c["simbad"].get("otype", "?"),
                                                            1000 / c["plx"], c["teff"], c["sep"],
                                                            ", ".join("%s %.2g mJy (pred %.2g, z %.1f)" % (b, d["f"] * 1e3, d["pred"] * 1e3, d["z"])
                                                                      for b, d in c["bands"].items())), flush=True)
    json.dump(dict(n=len(rows), matched=len(w), w3=n_w3, cold=cands, null=null_cold), open(core.results("wd_cold_excess.json"), "w"),
              indent=0, default=str)


wdcold_VETS = {
    "5934185854227111808": "an extended AllWISE source 2.5 arcsec off, in the Galactic plane (b -2.1)",
    "3455921181049073280": "the 12 um flux is a flat patch over 30 arcsec with no point source at the star (b +3.6): cirrus in the beam",
    "2229875964145781504": "W3 in 1 of 38 single exposures",
    "1859440674157056128": "W3 in 2 of 21 single exposures",
    "5219215228420645248": "AllWISE flags the W3 flux as an optical ghost",
}


def wdcold_record():
    d = json.load(open(core.results("wd_cold_excess.json")))
    core.record_search(
        "wd_cold_excess", "Cold excess around white dwarfs", archive="Gentile Fusillo et al. 2021 (J/MNRAS/508/3877); AllWISE",
        selection="white dwarfs within 100 pc with P_WD above 0.75", code=os.path.abspath(__file__), entry="wdcold",
        n_targets=d["n"], params=dict(radius_arcsec=3.0, sigma=3.0),
        null_summary="the same match 30 arcsec away: %d" % d["null"],
        sensitivity="a 300 K enclosure carrying a white dwarf's whole luminosity at about 20 pc",
        ledger="Cold excess around white dwarfs",
        result="%d of %d matched white dwarfs exceed their photosphere in W3 or W4 at 3 sigma with W1 and W2 photospheric; the "
               "strongest are an extended source in the Galactic plane, a W3 optical ghost, single-exposure W3 detections and "
               "LSPM J0543+3637, whose 12 um flux is cirrus with no point source at the star" % (len(d["cold"]), d["matched"]))
    core.clear_targets("wd_cold_excess")
    rows = []
    for c in d["cold"]:
        z = max(c["bands"].get("W3", {}).get("z", 0), c["bands"].get("W4", {}).get("z", 0))
        rows.append(dict(target="gaia_dr3:%s" % c["source_id"], stat=z, unit="sigma of the W3 or W4 excess over the photosphere",
                         vet=wdcold_VETS.get(c["source_id"]),
                         note="%.1f pc, %.0f K, %.1f arcsec" % (1000 / c["plx"], c["teff"], c["sep"])))
    core.record_targets("wd_cold_excess", rows)
    print("recorded %d" % len(rows))


def wdcold_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        wdcold_record()
    else:
        wdcold_main()


missingflux_DOC = r"""Missing bolometric flux in Kepler giants: Gaia luminosity below the seismic 4 pi R^2 sigma T^4, against the excess tail.

    python -I seti.py infrared missingflux [record] [--ruwe-max 1.4] [--plx-snr 5] [--blend-lsun 3] [--blend-numax 150]"""


def missingflux_main(record=False, ruwe_max=1.4, plx_snr=5.0, blend_lsun=3.0, blend_numax=150.0):
    from astroquery.vizier import Vizier
    yu = Vizier(row_limit=-1, columns=["KIC", "Teff", "NoCorR", "numax"], timeout=900).get_catalogs("J/ApJS/236/42/giants")[0]
    b2 = Vizier(row_limit=-1, columns=["KIC", "Lum"], timeout=900).get_catalogs("J/AJ/159/280/table2")[0]
    b1 = Vizier(row_limit=-1, columns=["KIC", "plx", "e_plx", "RUWE", "Ncomp"], timeout=900).get_catalogs("J/AJ/159/280/table1")[0]
    B2 = {int(r["KIC"]): r for r in b2}
    B1 = {int(r["KIC"]): r for r in b1}

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")
    rows = []
    for r in yu:
        k = int(r["KIC"])
        if k not in B2 or k not in B1:
            continue
        R, T, logL = f(r["NoCorR"]), f(r["Teff"]), f(B2[k]["Lum"])  # log10(L/Lsun)
        if not (R > 0 and T > 0 and np.isfinite(logL)):
            continue
        ok = f(B1[k]["RUWE"]) < ruwe_max and f(B1[k]["plx"]) / f(B1[k]["e_plx"]) > plx_snr and not f(B1[k]["Ncomp"]) > 1
        rows.append(dict(kic=k, R=R, T=T, L=10 ** logL, ratio=10 ** logL / (R * R * (T / 5772.0) ** 4), ok=ok, numax=f(r["numax"])))
    good = [x for x in rows if x["ok"] and x["ratio"] > 0]
    lr = np.log10([x["ratio"] for x in good])
    med = float(np.median(lr))
    s = float(1.4826 * np.median(np.abs(lr - med)))
    z = (lr - med) / s
    counts = {t: (int(np.sum(z < -t)), int(np.sum(z > t))) for t in (3, 4, 5, 6)}
    print("%d clean giants; log10(L_Gaia/L_seismic) median %+.3f, robust spread %.3f dex" % (len(good), med, s))
    for t, (lo, hi) in counts.items():
        print("  deficit beyond %d sigma: %d; excess beyond %d sigma: %d" % (t, lo, t, hi))
    blends = [x for x, zz in zip(good, z) if zz < -6 and x["L"] < blend_lsun and x["numax"] < blend_numax]
    print("  of the deficits beyond 6 sigma, %d have a dwarf's luminosity (below %g Lsun) and a giant's numax (below %g uHz)" % (len(blends), blend_lsun, blend_numax))
    if not record:
        return
    core.record_search(
        "missing_flux_giants", "Missing bolometric flux", archive="Yu et al. 2018 (J/ApJS/236/42); Berger et al. 2020 (J/AJ/159/280)",
        selection="Kepler red giants, single, RUWE below %g, parallax S/N above %g" % (ruwe_max, plx_snr), code=os.path.abspath(__file__), entry="missingflux",
        n_targets=len(good), params=dict(radius="NoCorR", ruwe_max=ruwe_max, plx_snr=plx_snr, blend_lsun=blend_lsun, blend_numax=blend_numax),
        null_summary="the excess side of the same distribution: %s" % ", ".join("%d sigma %d/%d" % (t, a, b) for t, (a, b) in counts.items()),
        sensitivity="a deficit of %.0f per cent at 3 sigma" % (100 * (1 - 10 ** (-3 * s))),
        ledger="Missing bolometric flux",
        result="deficits as common as excesses at every threshold (beyond 4 sigma %d against %d); the extreme deficits have a dwarf's "
               "luminosity and a giant's oscillations: background giants in foreground dwarfs' apertures" % counts[4])
    core.clear_targets("missing_flux_giants")
    core.record_targets("missing_flux_giants", [dict(target="kic:%d" % x["kic"], stat=float(zz), unit="z of log10(L_Gaia/L_seismic)",
                                                         vet=("a dwarf's luminosity with a giant's numax: a background giant in the aperture"
                                                              if zz < -6 and x["L"] < blend_lsun and x["numax"] < blend_numax else None))
                                                    for x, zz in zip(good, z)])
    print("recorded")


massless_VETS = {
    "HD 80883": "a companion 0.6-0.7 arcsec away (WDS 09197-5929, found 2021) blends with the star in Gaia's window",
    "HD 163441": "a companion 0.5 arcsec away (WDS 17558+1148 Aa,Ab, found 2022) blends with the star in Gaia's window",
    "HD 105113B": "4 arcsec from HD 105113 A, 2.2 mag brighter (WDS 12061-3258 AB)",
    "HD 217850": "RUWE 35 and a period beyond Gaia DR3's 34-month baseline: a long-period companion the velocities show as an 87 m/s drift",
    "HD 223238": "a 61.3-day period beside Gaia's 63-day scanning-law precession, RUWE 1.19, no companion within 4 arcsec brighter than 17.5 mag",
}


def massless_record():
    """Record each orbit's allowed K over the predicted dark-companion K with vets."""
    d = json.load(open(core.results("massless_orbits.json")))
    excl = [r for r in d if r["K_max3"] < 0.3 * r["K_pred"]]
    core.record_search(
        "massless_orbits", "Light without mass", archive="Gaia DR3 nss_two_body_orbit; HARPS RVBank (Trifonov et al. 2020); "
        "HIRES (Butler et al. 2017); AllWISE", selection="astrometric orbits within 100 pc of stars with 8 or more velocities",
        code=os.path.abspath(__file__), entry="massless", n_targets=len(d),
        params=dict(k_ratio=0.3, sigma=3, phases=48),
        null_summary="the velocities' own fit at every periastron phase of the orbit's period and eccentricity",
        sensitivity="velocity amplitudes of 6 m/s to a few km/s at the orbit's period",
        ledger="Light without mass",
        result="%d of %d orbits have velocities that allow under 0.3 of the predicted amplitude; the close companions of HD 80883, "
               "HD 163441 and HD 105113B, a long-period companion of HD 217850, HD 223238's scanning-law period and two stars "
               "whose velocities scatter by 0.5-0.9 km/s; none has warm W3 excess" % (len(excl), len(d)))
    core.clear_targets("massless_orbits")
    rows = []
    for r in d:
        name = r.get("simbad", {}).get("main_id", "") if isinstance(r.get("simbad"), dict) else ""
        name = " ".join(name.split())
        vet = massless_VETS.get(name)
        if vet is None and r["K_max3"] < 0.3 * r["K_pred"] and r["jitter"] > 300:
            vet = "velocities scatter by %.0f m/s: blended lines of a double star" % r["jitter"]
        rows.append(dict(target="gaia_dr3:%s" % r["source_id"], stat=r["K_max3"] / r["K_pred"],
                         unit="largest velocity amplitude allowed over the dark-companion prediction", vet=vet,
                         note="%s %s, P %.1f d, a0 %.2f mas, predicted K %.0f m/s, allowed %.1f m/s from %d velocities over %.0f d"
                              % (name or "", r["type"], r["P"], r["a0_mas"], r["K_pred"], r["K_max3"], r["n_rv"], r["span_d"])))
    core.record_targets("massless_orbits", rows)
    print("recorded %d orbits, %d excluded by their velocities" % (len(rows), len(excl)))


def massless_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        massless_record()
    else:
        massless_main()


# --------------------------------------------------------------------------------------------------------------

vasco_DOC = r"""VASCO vanished POSS I sources with warm infrared counterparts: red W1-W2, off the ecliptic, fixed red CatWISE source, no Legacy Surveys source to r 24.

    python -I seti.py infrared vasco fetch
    python -I seti.py infrared vasco select"""

vasco_ROOT = os.environ.get("VASCO_ROOT", core.data("vasco"))
vasco_HERE = core.RESULTS
vasco_SVO = "http://svocats.cab.inta-csic.es/%s/cs.php?RA=180.000000&DEC=0.000000&SR=180.000000&VERB=2&format=ascii"
vasco_EPS = math.radians(23.4392911)


def vasco_fetch():
    os.makedirs(vasco_ROOT, exist_ok=True)
    for cat in ("vanish-possi", "vanish-neowise", "vanish-onlyir"):
        b = urllib.request.urlopen(urllib.request.Request(vasco_SVO % cat, headers=core.UA),
                                   timeout=1800).read()
        open(os.path.join(vasco_ROOT, cat + ".txt"), "wb").write(b)
        print(cat, b.count(b"\n") - 1, "rows")


def vasco_load(cat):
    t = [ln for ln in open(os.path.join(vasco_ROOT, cat + ".txt"), encoding="utf-8", errors="replace").read().splitlines()
         if ln.strip()]
    head = t[0].lstrip("#").split(",")
    return [dict(zip(head, ln.split(","))) for ln in t[1:]]


def vasco_ecl_lat(ra, dec):
    a, d = math.radians(ra), math.radians(dec)
    return math.degrees(math.asin(math.sin(d) * math.cos(vasco_EPS) - math.cos(d) * math.sin(vasco_EPS) * math.sin(a)))


def vasco_red_rows():
    out = []
    for r in vasco_load("vanish-neowise"):
        q = r.get("ph_qual") or "UU"
        w1, w2, s2 = num(r["w1mpro"]), num(r["w2mpro"]), num(r["w2sigmpro"])
        if w2 is None or s2 is None or s2 >= 0.2 or q[1:2] not in ("A", "B"):
            continue
        if q[0] in ("A", "B") and w1 is not None and w1 - w2 <= 1.5:
            continue
        out.append(dict(src="neowise", ra=float(r["RA"]), dec=float(r["DEC"]), w1=w1, w2=w2, q=q))
    for r in vasco_load("vanish-onlyir"):
        c1, c2 = num(r["W1mproPM_CATWISE"]), num(r["W2mproPM_CATWISE"])
        if c2 is not None and (c1 is None or c1 - c2 > 1.5):
            out.append(dict(src="onlyir", ra=float(r["RA"]), dec=float(r["DEC"]), w1=c1, w2=c2, q="catwise"))
    return out


def vasco_legacy(ra, dec):
    """Legacy Surveys sources within 30 arcsec with separations, or a string outside DR10 and DR9."""
    for table in ("ls_dr10.tractor", "ls_dr9.tractor"):
        q = ("SELECT ra, dec, type, mag_r, mag_z FROM %s WHERE q3c_radial_query(ra, dec, %.6f, %.6f, %.6f) = 't'"
             % (table, ra, dec, 30.0 / 3600))
        try:
            rows = core.datalab(q)
        except Exception as e:
            return "query failed: " + str(e)[:60]
        if rows and "ra" in rows[0]:
            for x in rows:
                x["sep"] = math.hypot((float(x["ra"]) - ra) * math.cos(math.radians(dec)), float(x["dec"]) - dec) * 3600
            return rows
    return "outside the Legacy Surveys footprint"


def vasco_select():
    rows = vasco_red_rows()
    off = [r for r in rows if abs(vasco_ecl_lat(r["ra"], r["dec"])) > 20.0]
    print("%d red infrared-only counterparts (W2 S/N 5, W1-W2 above 1.5 or W1 undetected); %d beyond 20 deg of the "
          "ecliptic" % (len(rows), len(off)))
    for i, r in enumerate(off):
        r["source_id"] = str(i)
    cw = {}
    for k in range(0, len(off), 5000):
        for m in core.xmatch(off[k:k + 5000], "II/365/catwise", 3.0):
            if m["source_id"] not in cw or float(m["angDist"]) < float(cw[m["source_id"]]["angDist"]):
                cw[m["source_id"]] = m
    fixed = []
    for r in off:
        m = cw.get(r["source_id"])
        if not m:
            continue
        w1, w2 = num(m.get("W1mproPM")), num(m.get("W2mproPM"))
        pmra, pmde = num(m.get("pmRA")), num(m.get("pmDE"))
        epm = num(m.get("e_pmRA")) or 1.0
        pm = math.hypot(pmra or 0, pmde or 0) * 1000.0 if pmra is not None else None
        if w2 is None or (w1 is not None and w1 - w2 <= 1.5):
            continue
        if pm is not None and pm > 100 and pm > 3 * epm * 1000.0:
            continue
        fixed.append(dict(r, cw_name=m.get("Name"), cw_w1=w1, cw_w2=w2, pm_mas=pm, sep=float(m["angDist"])))
    print("%d with a fixed CatWISE2020 counterpart within 3 arcsec that is itself red (W1-W2 above 1.5)" % len(fixed))
    left = []
    for r in fixed:
        r["ls"] = vasco_legacy(r["ra"], r["dec"])
        if not isinstance(r["ls"], list):
            left.append(r)
        elif not [x for x in r["ls"] if x["sep"] <= 2.0 and (num(x["mag_r"]) or 99) < 24]:
            left.append(r)
    print("%d with no Legacy Surveys source brighter than r = 24 within 2 arcsec (or outside its footprint)" % len(left))
    for r in left:
        print("  %s %10.5f %+9.5f (beta %+.1f): NEOWISE/IR W1 %s W2 %s (%s); CatWISE %s W1 %s W2 %s, pm %s mas/yr; %s"
              % (r["src"], r["ra"], r["dec"], vasco_ecl_lat(r["ra"], r["dec"]), r["w1"], r["w2"], r["q"], r["cw_name"],
                 r["cw_w1"], r["cw_w2"], "%.0f" % r["pm_mas"] if r["pm_mas"] is not None else "-",
                 r["ls"] if isinstance(r["ls"], str) else "no Legacy Surveys source to r = 24 within 2 arcsec (%d within 30)"
                 % len(r["ls"])))
    json.dump(dict(red=len(rows), off=len(off), fixed=fixed, left=left), open(os.path.join(vasco_HERE, "vasco_ir.json"), "w"),
              indent=1, default=str)


def vasco_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "select"))
    a = p.parse_args()
    {"fetch": vasco_fetch, "select": vasco_select}[a.mode]()


def vasco_cli():
    quiet_warnings()
    vasco_main()


# --------------------------------------------------------------------------------------------------------------

spherex_DOC = r"""SPHEREx 0.75-5 um spectra of candidates by PSF fit on byte-range blocks, against matched controls at 2-3, 3-4 and 4-5 um.

    python -I seti.py infrared spherex spec SOURCE_ID ... | @FILE              (SPHX_OUT, SPHX_THREADS)
    python -I seti.py infrared spherex controls [--targets CSV --col COLUMN] [--out JSON] [--threads N] [--top 15] [--radius 1.5]
    python -I seti.py infrared spherex compare [--targets sphx_targets.json] [--table CSV --idcol COLUMN]
    python -I seti.py infrared spherex fitexcess [--targets JSON]
    python -I seti.py infrared spherex blends [--targets JSON]          (Legacy Surveys deblended W1 and W2 of each star)"""

spherex_HERE = core.RESULTS
spherex_SPHX = os.environ.get("SPHX_OUT", os.path.join(spherex_HERE, "sphx"))
spherex_TARGETS = os.path.join(spherex_HERE, "sphx_targets.json")
# Project Hephaistos candidates and W26
spherex_HEPHAISTOS = {"A": "3496509309189181184", "B": "4843191593270342656", "C": "4649396037451459712",
              "D": "2660349163149053824", "E": "3190232820489766656", "F": "2956570141274256512",
              "G": "2644370304260053376", "H3a": "2437221214075471744", "H3b": "3854090071297359616",
              "H3c": "651765552072217216", "W26": "2981818948336660224"}
spherex_EDGES = np.r_[np.arange(0.75, 2.5, 0.1), np.arange(2.5, 5.01, 0.25)]
spherex_RANGES = ((2.0, 3.0), (3.0, 4.0), (4.0, 5.0))


def spherex_gaia(query):
    """Gaia rows with retries; empty when the archive keeps failing."""
    for attempt in range(4):
        try:
            return core.gaia(query)
        except Exception:
            time.sleep(15 * (attempt + 1))
    return []


# -------------------------------------------------------------------------------------------------------------- spec


def spherex_track(r):
    """Position of a Gaia DR3 row at an MJD along its proper motion."""
    ra0, dec0 = float(r["ra"]), float(r["dec"])
    pmra, pmdec = float(r.get("pmra") or 0), float(r.get("pmdec") or 0)

    def at(mjd):
        dt = (float(mjd) - 57388.5) / 365.25  # J2016.0
        return ra0 + pmra / 3.6e6 * dt / np.cos(np.radians(dec0)), dec0 + pmdec / 3.6e6 * dt
    return at


def spherex_measure(row, at):
    """One image's measurement: wavelength, flux, error (mJy), time, detector; None off detector or flagged."""
    ra, dec = at(row["t"])
    try:
        px = core.spherex_pixels(row["uri"], ra, dec)
    except Exception as e:
        return e
    m = science.spherex_point_pixels(px) if px is not None else None
    return None if m is None else dict(m, t=row["t"], band=row["band"])


def spherex_star(sid, at, out_dir, threads=8, most=1500):
    """Measure one star in every (or most) SPHEREx images, writing out_dir/SID.csv."""
    path = os.path.join(out_dir, sid + ".csv")
    if os.path.exists(path):
        return "%s: already measured" % sid
    t0 = time.time()
    rows = sorted(core.spherex_images(*at(61000.0)), key=lambda r: float(r["t"]))
    if len(rows) > most:
        rows = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, most)]
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(lambda r: spherex_measure(r, at), rows))
    errs = [r for r in res if isinstance(r, Exception)]
    good = sorted((r for r in res if isinstance(r, dict)), key=lambda r: r["wave"])
    if not rows or len(errs) > 0.2 * len(rows):
        why = Counter("%s: %s" % (type(e).__name__, str(e).split(".fits")[-1].strip(" :")[:80]) for e in errs)
        return "%s: %d of %d images failed at the archive (%s); not written" % (
            sid, len(errs), len(rows), "; ".join("%d %s" % (v, k) for k, v in why.most_common(3)) if errs
            else "no image list")
    with open(path + ".tmp", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["wave", "flux", "err", "t", "band"])
        wr.writeheader()
        for r in good:
            wr.writerow(r)
    os.replace(path + ".tmp", path)
    return "%s: %d images, %d measurements, %d failed, %.0f s" % (sid, len(rows), len(good), len(errs),
                                                                  time.time() - t0)


def spherex_spec_main(argv):
    os.makedirs(spherex_SPHX, exist_ok=True)
    ids = []
    for x in argv:
        ids += open(x[1:]).read().split() if x.startswith("@") else [x]
    for sid in ids:
        if os.path.exists(os.path.join(spherex_SPHX, sid + ".csv")):
            print("%s: already measured" % sid, flush=True)
            continue
        r = spherex_gaia("SELECT ra, dec, pmra, pmdec FROM gaiadr3.gaia_source WHERE source_id = %s" % sid)
        if not r:
            print("%s: Gaia query failed" % sid, flush=True)
            continue
        print(spherex_star(sid, spherex_track(r[0]), spherex_SPHX, int(os.environ.get("SPHX_THREADS", "8"))),
              flush=True)


# ---------------------------------------------------------------------------------------------------------- controls


def spherex_controls_main(argv):
    ap = argparse.ArgumentParser(prog="seti.py infrared spherex controls")
    ap.add_argument("--targets")
    ap.add_argument("--col", default="Gaia DR3 source ID")
    ap.add_argument("--out", default=spherex_TARGETS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--top", type=int, default=15, help="Gaia candidates tried per target")
    ap.add_argument("--radius", type=float, default=1.5, help="search radius in degrees")
    a = ap.parse_args(argv)
    targets = spherex_HEPHAISTOS
    if a.targets:
        targets = {"T%02d" % i: r[a.col] for i, r in enumerate(csv.DictReader(open(a.targets, encoding="utf-8")))}
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    lock = threading.Lock()

    def one(item):
        lab, sid = item
        if lab in out and len(out[lab]["controls"]) >= 2:
            return
        t = spherex_gaia("SELECT ra, dec, phot_g_mean_mag, bp_rp, parallax FROM gaiadr3.gaia_source WHERE source_id=%s" % sid)
        if not t:
            print(lab, sid, "Gaia query failed", flush=True)
            return
        ra, dec, g, c, p = (float(t[0][k]) for k in ("ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax"))
        cs = spherex_gaia("SELECT TOP %d source_id, ra, dec FROM gaiadr3.gaia_source WHERE "
                  "CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',%f,%f,%f))=1 AND source_id<>%s AND ABS(bp_rp-%f)<0.12 "
                  "AND ABS(phot_g_mean_mag-%f)<0.8 AND ruwe<1.4 AND parallax BETWEEN %f AND %f"
                  % (a.top, ra, dec, a.radius, sid, c, g, min(p / 1.6, p * 1.6), max(p / 1.6, p * 1.6)))
        ctrl = []
        for x in cs:
            try:
                w = core.allwise_nearest(float(x["ra"]), float(x["dec"]), 6.0)
            except Exception:
                continue
            if w is not None and float(w["w3snr"] or 0) < 3 and str(w["cc_flags"])[:2] == "00":
                ctrl.append(x["source_id"])
            if len(ctrl) >= 2:
                break
        with lock:
            out[lab] = dict(sid=sid, g=g, bp_rp=c, plx=p, controls=ctrl)
            json.dump(out, open(a.out, "w"), indent=1)
        print(lab, sid, "G %.2f bp_rp %.2f plx %.2f" % (g, c, p), "controls", ctrl, flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, targets.items()))


# ----------------------------------------------------------------------------------------------------------- compare


def spherex_binned(sid):
    path = os.path.join(spherex_SPHX, sid + ".csv")
    if not os.path.exists(path):
        return None
    r = list(csv.DictReader(open(path)))
    w = np.array([float(x["wave"]) for x in r])
    f = np.array([float(x["flux"]) for x in r])
    e = np.array([float(x["err"]) for x in r])
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    w, f, e = w[ok], f[ok], e[ok]
    cen, val, err = [], [], []
    for lo, hi in zip(spherex_EDGES[:-1], spherex_EDGES[1:]):
        m = (w >= lo) & (w < hi)
        if m.sum() >= 3:
            cen.append(0.5 * (lo + hi))
            val.append(np.median(f[m]))
            err.append(1.25 * np.std(f[m]) / np.sqrt(m.sum()))
    cen, val, err = np.array(cen), np.array(val), np.array(err)
    sel = (cen > 1.0) & (cen < 1.6)
    if sel.sum() < 2:
        return None
    norm = np.median(val[sel])
    return cen, val / norm, err / norm


def spherex_excess(tb, cbs):
    """Weighted mean excess over the control mean per range: [(value, error)]."""
    cen = tb[0]
    ctrl = np.array([np.interp(cen, c[0], c[1]) for c in cbs])
    cerr = np.array([np.interp(cen, c[0], c[2]) for c in cbs])
    cm = ctrl.mean(0)
    ce = np.sqrt((cerr ** 2).sum(0)) / len(cbs)
    ratio = tb[1] / cm
    rerr = ratio * np.hypot(tb[2] / tb[1], ce / cm)
    out = []
    for lo, hi in spherex_RANGES:
        m = (cen > lo) & (cen < hi) & np.isfinite(rerr) & (rerr > 0)
        if m.sum() == 0:
            out.append((np.nan, np.nan))
            continue
        wgt = 1.0 / rerr[m] ** 2
        out.append((float(np.sum((ratio[m] - 1.0) * wgt) / wgt.sum()), float(1.0 / np.sqrt(wgt.sum()))))
    return out, cen, ratio


def spherex_fitexcess_main(argv):
    """Temperature and fractional luminosity of each excess from 1.6-5 um ratios, with a 3.3 um PAH residual test."""
    p = argparse.ArgumentParser(prog="seti.py infrared spherex fitexcess")
    p.add_argument("--targets", default=spherex_TARGETS)
    a = p.parse_args(argv)
    T = json.load(open(a.targets))
    teff = {r["source_id"]: float(r["teff_gspphot"]) for r in core.gaia(
        "SELECT source_id, teff_gspphot FROM gaiadr3.astrophysical_parameters WHERE source_id IN (%s)"
        % ",".join(t["sid"] for t in T.values())) if r["teff_gspphot"] not in ("", None)}
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23

    def bl(lam_um, temp):
        x = h * c / (lam_um * 1e-6 * k * temp)
        return 1.0 / ((lam_um ** 5) * np.expm1(np.minimum(x, 700)))
    grid = np.geomspace(150, 3000, 80)
    out = []
    for lab, t in T.items():
        tb = spherex_binned(t["sid"])
        cb = [x for x in (spherex_binned(cc) for cc in t["controls"]) if x is not None]
        if tb is None or not cb:
            continue
        ex, cen, ratio = spherex_excess(tb, cb)
        ctrl = np.array([np.interp(cen, cc[0], cc[1]) for cc in cb])
        cerr = np.array([np.interp(cen, cc[0], cc[2]) for cc in cb])
        rerr = ratio * np.hypot(tb[2] / tb[1], np.sqrt((cerr ** 2).sum(0)) / len(cb) / ctrl.mean(0))
        m = (cen > 1.6) & np.isfinite(rerr) & (rerr > 0)
        x, y, e = cen[m], ratio[m] - 1.0, np.maximum(rerr[m], 0.01)
        ts = teff.get(t["sid"], 5500.0)
        best = None
        for temp in grid:
            shape = (ts / temp) ** 4 * bl(x, temp) / bl(x, ts)
            w = 1 / e ** 2
            f = float(np.sum(w * y * shape) / np.sum(w * shape ** 2))
            chi = float(np.sum(w * (y - f * shape) ** 2))
            if best is None or chi < best[0]:
                best = (chi, temp, f, shape)
        chi, temp, f, shape = best
        pah = (x > 3.2) & (x < 3.5)
        resid = float(np.sum((y - f * shape)[pah] / e[pah] ** 2) / np.sum(1 / e[pah] ** 2)) if pah.any() else float("nan")
        rerr_p = float(1 / np.sqrt(np.sum(1 / e[pah] ** 2))) if pah.any() else float("nan")
        out.append(dict(lab=lab, sid=t["sid"], teff=ts, T=float(temp), f=f, chi2=chi, dof=int(m.sum() - 2),
                        pah=resid, pah_err=rerr_p, ex45=ex[2][0], ex45_err=ex[2][1]))
        print("%-4s %s Teff %4.0f: excess at %4.0f K holding %.1e of the star's light, chi2 %.1f for %d; 3.3 um residual "
              "%+.3f +- %.3f" % (lab, t["sid"], ts, temp, f, chi, m.sum() - 2, resid, rerr_p))
    json.dump(out, open(os.path.join(os.path.dirname(a.targets), "ch53_fitexcess.json"), "w"), indent=1)
    sig = [o for o in out if o["ex45"] / o["ex45_err"] >= 3]
    tt = np.array([o["T"] for o in sig])
    print("%d with a 4-5 um excess at 3 sigma: excess temperature below 400 K in %d, 400-1000 K in %d, above 1000 K in %d"
          % (len(sig), int((tt < 400).sum()), int(((tt >= 400) & (tt <= 1000)).sum()), int((tt > 1000).sum())))


spherex_W2_SIGMA = 6.4 / 2.3548  # arcsec, W2 PSF sigma


def spherex_ls_sources(ra, dec, radius=8.0):
    """Legacy Surveys model sources within radius with W1 and W2 forced fluxes."""
    w = radius / 3600.0
    wr = w / max(np.cos(np.radians(dec)), 0.01)
    for tab in ("ls_dr10.tractor", "ls_dr9.tractor"):
        rows = core.datalab(
            "SELECT ra, dec, type, ref_cat, ref_id, flux_r, flux_w1, flux_w2, flux_ivar_w1, flux_ivar_w2, shape_r FROM %s "
            "WHERE ra BETWEEN %.7f AND %.7f AND dec BETWEEN %.7f AND %.7f" % (tab, ra - wr, ra + wr, dec - w, dec + w))
        if rows:
            for r in rows:
                r["sep"] = float(np.hypot((float(r["ra"]) - ra) * np.cos(np.radians(dec)), float(r["dec"]) - dec) * 3600)
            return tab, [r for r in rows if r["sep"] <= radius]
    return None, []


spherex_UNWISE_PX = 2.75  # arcsec per unWISE pixel


def spherex_unwise_shift(ra, dec, radius=3.0):
    """unWISE W2-minus-W1 centroid offset (arcsec, east, north) with errors, or None."""
    w = radius / 3600.0
    wr = w / max(np.cos(np.radians(dec)), 0.01)
    rows = core.datalab("SELECT ra, dec, x_w1, x_w2, y_w1, y_w2, dx_w1, dx_w2, dy_w1, dy_w2, flux_w1, flux_w2 "
                            "FROM unwise_dr1.object WHERE ra BETWEEN %.7f AND %.7f AND dec BETWEEN %.7f AND %.7f"
                            % (ra - wr, ra + wr, dec - w, dec + w))
    best = None
    for r in rows:
        try:
            vals = [float(r[k]) for k in ("x_w1", "x_w2", "y_w1", "y_w2", "dx_w1", "dx_w2", "dy_w1", "dy_w2")]
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(vals)):
            continue
        sep = np.hypot((float(r["ra"]) - ra) * np.cos(np.radians(dec)), float(r["dec"]) - dec) * 3600
        if best is None or sep < best[0]:
            best = (sep, vals)
    if best is None:
        return None
    x1, x2, y1, y2, ex1, ex2, ey1, ey2 = best[1]
    # unWISE x grows west, y north
    return dict(east=-(x2 - x1) * spherex_UNWISE_PX, north=(y2 - y1) * spherex_UNWISE_PX,
                err=float(np.hypot(np.hypot(ex1, ex2), np.hypot(ey1, ey2)) / np.sqrt(2) * spherex_UNWISE_PX))


def spherex_blends_main(argv):
    """Each target's W1 and W2 light divided among Legacy Surveys sources within 8 arcsec: own ratio against controls and neighbours' share."""
    p = argparse.ArgumentParser(prog="seti.py infrared spherex blends")
    p.add_argument("--targets", default=spherex_TARGETS)
    a = p.parse_args(argv)
    T = json.load(open(a.targets))
    fits = {}
    fp = os.path.join(os.path.dirname(a.targets), "ch53_fitexcess.json")
    if os.path.exists(fp):
        fits = {o["sid"]: o for o in json.load(open(fp))}

    def measure(sid):
        g = spherex_gaia("SELECT ra, dec FROM gaiadr3.gaia_source WHERE source_id = %s" % sid)
        if not g:
            return None
        ra, dec = float(g[0]["ra"]), float(g[0]["dec"])
        shift = spherex_unwise_shift(ra, dec)
        tab, rows = spherex_ls_sources(ra, dec)
        if not rows:
            return dict(sid=sid, tab=None, shift=shift)
        star = [r for r in rows if str(r.get("ref_id") or "") == str(sid)] or \
               sorted([r for r in rows if r["sep"] < 1.5], key=lambda r: r["sep"])[:1]
        if not star:
            return dict(sid=sid, tab=tab, star=None)
        s = star[0]
        f1, f2 = float(s["flux_w1"] or "nan"), float(s["flux_w2"] or "nan")
        e2 = 1 / np.sqrt(float(s["flux_ivar_w2"])) if float(s["flux_ivar_w2"] or 0) > 0 else float("nan")
        nb = [r for r in rows if r is not s]
        share = sum(float(r["flux_w2"] or 0) * np.exp(-r["sep"] ** 2 / (4 * spherex_W2_SIGMA ** 2)) for r in nb)
        near = min(nb, key=lambda r: r["sep"]) if nb else None
        return dict(sid=sid, tab=tab, shift=shift, w1=f1, w2=f2, w2_err=e2, ratio=f2 / f1 if f1 > 0 else float("nan"),
                    neighbour_share=share / f2 if f2 > 0 else float("nan"), n_neighbours=len(nb),
                    nearest=dict(sep=round(near["sep"], 2), type=near["type"], flux_r=near["flux_r"],
                                 flux_w2=near["flux_w2"]) if near else None)
    out, ctrl_shifts = [], []
    for lab, t in T.items():
        m = measure(t["sid"]) or dict(sid=t["sid"], tab=None, shift=None)
        cs = [x for x in (measure(c) for c in t["controls"]) if x]
        ctrl_shifts += [x["shift"] for x in cs if x.get("shift")]
        fx = fits.get(t["sid"], {})
        rec = dict(m, lab=lab, covered="ratio" in m, spherex_ex45=fx.get("ex45"), spherex_T=fx.get("T"))
        sh = m.get("shift")
        txt = "W2-W1 centroid %+.3f E %+.3f N +- %.3f arcsec" % (sh["east"], sh["north"], sh["err"]) if sh else \
              "no unWISE match"
        if "ratio" in m:
            cr = [x["ratio"] for x in cs if "ratio" in x and np.isfinite(x["ratio"]) and x["ratio"] > 0.1]
            rec["control_ratio"] = float(np.median(cr)) if cr else float("nan")
            rec["own_w2_excess"] = m["ratio"] / rec["control_ratio"] - 1 if cr else float("nan")
            txt += ("; own W2/W1 %.3f vs controls %.3f, excess the star holds %+.2f; neighbours within 8\" %d, their "
                    "share of a fit at the star %.2f" % (m["ratio"], rec["control_ratio"], rec["own_w2_excess"],
                                                         m["n_neighbours"], m["neighbour_share"]))
        else:
            txt += "; outside the Legacy Surveys"
        out.append(rec)
        print("%-4s %s: %s; SPHEREx 4-5 um excess %s" % (lab, t["sid"], txt, "%.2f" % fx["ex45"]
                                                         if fx.get("ex45") is not None else "--"))
    # controls' offsets set the floor
    c_mag = np.array([np.hypot(s["east"], s["north"]) for s in ctrl_shifts])
    floor = float(np.percentile(c_mag, 68)) if len(c_mag) else 0.0
    print("controls: %d W2-W1 centroid offsets, 68th percentile %.3f arcsec, largest %.3f"
          % (len(c_mag), floor, c_mag.max() if len(c_mag) else float("nan")))
    for r in out:
        s = r.get("shift")
        if s:
            r["shift_sig"] = float(np.hypot(s["east"], s["north"]) / np.hypot(s["err"], floor))
    big = [r for r in out if r.get("shift_sig", 0) > 3]
    print("%d targets whose W2 centroid stands 3 sigma or more from their W1 centroid: %s"
          % (len(big), ", ".join("%s %.2f\" (%.1f sigma)" % (r["lab"], np.hypot(r["shift"]["east"], r["shift"]["north"]),
                                                             r["shift_sig"]) for r in big)))
    json.dump(dict(targets=out, control_shift_68=floor), open(os.path.join(
        os.path.dirname(a.targets), os.path.basename(a.targets).replace(".json", "") + "_blends.json"), "w"), indent=1)


def spherex_compare_main(argv):
    p = argparse.ArgumentParser(prog="seti.py infrared spherex compare")
    p.add_argument("--targets", default=spherex_TARGETS)
    p.add_argument("--table")
    p.add_argument("--idcol", default="Gaia DR3 source ID")
    a = p.parse_args(argv)
    T = json.load(open(a.targets))
    pred = {}
    if a.table:
        for r in csv.DictReader(open(a.table, encoding="utf-8")):
            try:
                pred[r[a.idcol]] = 10 ** (0.4 * (float(r["hatW2_pred"]) - float(r["mag_w2_vg"]))) - 1.0
            except (KeyError, ValueError):
                pass
    rows, null = [], []
    for lab, t in T.items():
        tb = spherex_binned(t["sid"])
        cb = [x for x in (spherex_binned(c) for c in t["controls"]) if x is not None]
        if tb is None or not cb:
            print("%-4s %s: spectra not ready (target %s, %d controls)" % (lab, t["sid"], tb is not None, len(cb)))
            continue
        ex, cen, ratio = spherex_excess(tb, cb)
        if len(cb) >= 2:
            null.append(spherex_excess(cb[0], [cb[1]])[0])
        rows.append(dict(lab=lab, sid=t["sid"], n_ctrl=len(cb), ex=ex, w2=pred.get(t["sid"])))
        line = "; ".join("%.0f-%.0f um %+.3f +- %.3f (%.1f sigma)" % (lo, hi, v, e, v / e) for (lo, hi), (v, e) in zip(spherex_RANGES, ex))
        w2 = "; WISE W2 excess %+.3f" % pred[t["sid"]] if t["sid"] in pred else ""
        print("%-4s %s (%d controls): %s%s" % (lab, t["sid"], len(cb), line, w2))
        print("      ratio by bin: " + " ".join("%.2f:%.2f" % (c, r) for c, r in zip(cen, ratio) if c > 2.4))
    if null:
        nv = np.array([[v for v, e in n] for n in null])
        ne = np.array([[e for v, e in n] for n in null])
        tv = np.array([[v for v, e in r["ex"]] for r in rows])
        te = np.array([[e for v, e in r["ex"]] for r in rows])
        for k, (lo, hi) in enumerate(spherex_RANGES):
            # control-pair scatter beyond formal errors added to every error
            mad = 1.4826 * np.nanmedian(np.abs(nv[:, k] - np.nanmedian(nv[:, k])))
            floor = float(np.sqrt(max(mad ** 2 - np.nanmedian(ne[:, k]) ** 2, 0.0))) if len(nv) >= 5 else 0.0
            tz = tv[:, k] / np.hypot(te[:, k], floor)
            nz = nv[:, k] / np.hypot(ne[:, k], floor)
            for r, z in zip(rows, tz):
                r.setdefault("z_floor", []).append(float(z))
            print("%.0f-%.0f um: systematic floor %.3f from %d control pairs; targets above 3 sigma %d of %d, control "
                  "pairs %d of %d; median target excess %+.3f" % (lo, hi, floor, len(nv), np.sum(tz > 3), len(tz),
                                                                  np.sum(nz > 3), len(nz), np.nanmedian(tv[:, k])))
    json.dump(rows, open(os.path.splitext(a.targets)[0] + "_compare.json", "w"), indent=1)


spherex_MODES = {"spec": spherex_spec_main, "controls": spherex_controls_main, "compare": spherex_compare_main,
                 "fitexcess": spherex_fitexcess_main, "blends": spherex_blends_main}


# --------------------------------------------------------------------------------------------------------------

volume_DOC = r"""SPHEREx spectra of every unsaturated star within a distance: unresolved lines and 3-5 um excesses against neighbours, vetted for blends and companions.

    python -I seti.py infrared volume sample [--plx 40] [--ks 9.5] [--gfaint 15]
    python -I seti.py infrared volume spec [--part K/N] [--threads 8]        (SPHX_VOLUME, default results/sphx_volume)
    python -I seti.py infrared volume lines [--z 5] [--half 3]
    python -I seti.py infrared volume excess [--k 15]
    python -I seti.py infrared volume vet
    python -I seti.py infrared volume record"""

volume_SAMPLE = core.results("sphx_volume.json")
volume_DIR = os.environ.get("SPHX_VOLUME", core.results("sphx_volume"))


def volume_sample_main(argv):
    ap = argparse.ArgumentParser(prog="seti.py infrared volume sample")
    ap.add_argument("--plx", type=float, default=40.0)
    ap.add_argument("--ks", type=float, default=9.5)
    ap.add_argument("--gfaint", type=float, default=15.0)
    a = ap.parse_args(argv)
    rows = spherex_gaia("SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag AS g, bp_rp, ruwe "
                        "FROM gaiadr3.gaia_source WHERE parallax > %g AND parallax_over_error > 10" % a.plx)
    by = {r["source_id"]: r for r in rows}
    ids = list(by)
    for i in range(0, len(ids), 400):
        q = ("SELECT x.source_id, t.j_m, t.h_m, t.ks_m, t.ph_qual FROM gaiadr3.tmass_psc_xsc_best_neighbour AS x "
             "JOIN gaiadr3.tmass_psc_xsc_join AS j ON j.clean_tmass_psc_xsc_oid = x.clean_tmass_psc_xsc_oid "
             "JOIN gaiadr1.tmass_original_valid AS t ON t.designation = j.original_psc_source_id "
             "WHERE x.source_id IN (%s)" % ",".join(ids[i:i + 400]))
        for r in spherex_gaia(q):
            by[r["source_id"]].update(r)
    keep = [r for r in rows if (r.get("ks_m") and float(r["ks_m"]) >= a.ks)
            or (not r.get("ks_m") and r["g"] and float(r["g"]) > a.gfaint)]
    json.dump(keep, open(volume_SAMPLE, "w"), indent=0)
    print("%d Gaia DR3 sources with parallax above %g mas; %d kept (Ks %g or fainter, or no 2MASS match and G above "
          "%g)" % (len(rows), a.plx, len(keep), a.ks, a.gfaint))


def volume_spec_main(argv):
    ap = argparse.ArgumentParser(prog="seti.py infrared volume spec")
    ap.add_argument("--part", default="0/1", help="K/N: every Nth star of the sample from the Kth")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    k, n = (int(v) for v in a.part.split("/"))
    os.makedirs(volume_DIR, exist_ok=True)
    sample = json.load(open(volume_SAMPLE))
    for r in sample[k::n]:
        print(spherex_star(r["source_id"], spherex_track(r), volume_DIR, a.threads), flush=True)


# SPHEREx resolving power per detector and H, He lines (um)
volume_R = {"SPHEREx-D1": 39.0, "SPHEREx-D2": 41.0, "SPHEREx-D3": 41.0, "SPHEREx-D4": 35.0, "SPHEREx-D5": 110.0,
            "SPHEREx-D6": 130.0}
volume_KNOWN = (0.9549, 1.0052, 1.0833, 1.0941, 1.2822, 1.8756, 1.9451, 2.1661, 2.6259, 3.0392, 3.2970, 3.7406,
                4.0523, 4.6538)


def volume_read(sid):
    p = os.path.join(volume_DIR, sid + ".csv")
    if not os.path.exists(p):
        return None
    r = list(csv.DictReader(open(p)))
    if len(r) < 50:
        return None
    a = {k: np.array([float(x[k]) for x in r]) for k in ("wave", "flux", "err", "t")}
    a["band"] = np.array([x["band"] for x in r])
    ok = np.isfinite(a["flux"]) & np.isfinite(a["err"]) & (a["err"] > 0)
    return {k: v[ok] for k, v in a.items()}


def volume_line_scan(a):
    """Stacked line significance at every half-resolution step, whole span and each half."""
    out = []
    tmid = np.median(a["t"])
    for b, R in volume_R.items():
        m = a["band"] == b
        if m.sum() < 20:
            continue
        w, f, e, t = a["wave"][m], a["flux"][m], a["err"][m], a["t"][m]
        o = np.argsort(w)
        w, f, e, t = w[o], f[o], e[o], t[o]
        cont = np.array([np.median(f[np.abs(w - x) < 1.5 * x / R]) for x in w])
        z = (f - cont) / e
        s = 1.4826 * np.median(np.abs(z - np.median(z)))
        z = z / max(s, 1.0)
        # one-sided running median within 1.5 resolution elements of detector ends
        lam = w[0] * (1 + 1.5 / R)
        while lam < w[-1] * (1 - 1.5 / R):
            on = np.abs(w - lam) < 0.5 * lam / R
            if on.sum() >= 4:
                h1, h2 = on & (t < tmid), on & (t >= tmid)
                out.append(dict(band=b, lam=float(lam), n=int(on.sum()), z=float(z[on].sum() / np.sqrt(on.sum())),
                                z1=float(z[h1].sum() / np.sqrt(h1.sum())) if h1.sum() >= 2 else float("nan"),
                                z2=float(z[h2].sum() / np.sqrt(h2.sum())) if h2.sum() >= 2 else float("nan")))
            lam += 0.5 * lam / R
    return out


def volume_lines_main(argv):
    ap = argparse.ArgumentParser(prog="seti.py infrared volume lines")
    ap.add_argument("--z", type=float, default=5.0)
    ap.add_argument("--half", type=float, default=3.0)
    a = ap.parse_args(argv)
    sample = json.load(open(volume_SAMPLE))
    hits, null, n = [], [], 0
    for r in sample:
        d = volume_read(r["source_id"])
        if d is None:
            continue
        n += 1
        for q in volume_line_scan(d):
            for sign, keep in ((1, hits), (-1, null)):
                if sign * q["z"] >= a.z and sign * q["z1"] >= a.half and sign * q["z2"] >= a.half:
                    keep.append(dict(q, star=r["source_id"], sign=sign))
    # a feature in three or more stars is instrumental or zodiacal
    for q in hits + null:
        R = volume_R[q["band"]]
        q["shared"] = len({o["star"] for o in hits + null if o["star"] != q["star"]
                           and abs(o["lam"] - q["lam"]) < q["lam"] / R})
        q["known"] = any(abs(q["lam"] - k) < k / R for k in volume_KNOWN)
    left = [q for q in hits if q["shared"] < 3 and not q["known"]]
    nleft = [q for q in null if q["shared"] < 3 and not q["known"]]
    print("%d stars; emission-like features at z %.0f with both halves at %.0f: %d (%d after shared and known lines), "
          "absorption-like %d (%d)" % (n, a.z, a.half, len(hits), len(left), len(null), len(nleft)))
    for q in sorted(left, key=lambda q: -q["z"])[:40]:
        print("  %s %.4f um (%s, %d measurements) z %.1f, halves %.1f and %.1f"
              % (q["star"], q["lam"], q["band"][-2:], q["n"], q["z"], q["z1"], q["z2"]))
    json.dump(dict(hits=hits, null=null), open(core.results("sphx_volume_lines.json"), "w"))


def volume_binned(d, sel=None):
    """Median binned flux normalised over 1.0-1.6 um."""
    w, f = (d["wave"], d["flux"]) if sel is None else (d["wave"][sel], d["flux"][sel])
    cen, val = [], []
    for lo, hi in zip(spherex_EDGES[:-1], spherex_EDGES[1:]):
        m = (w >= lo) & (w < hi)
        if m.sum() >= 3:
            cen.append(0.5 * (lo + hi))
            val.append(np.median(f[m]))
    cen, val = np.array(cen), np.array(val)
    norm = np.median(val[(cen > 1.0) & (cen < 1.6)]) if ((cen > 1.0) & (cen < 1.6)).sum() >= 2 else np.nan
    return cen, val / norm


def volume_excess_main(argv):
    ap = argparse.ArgumentParser(prog="seti.py infrared volume excess")
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args(argv)
    sample = {r["source_id"]: r for r in json.load(open(volume_SAMPLE))}
    spec, halves = {}, {}
    for sid in sample:
        d = volume_read(sid)
        if d is None:
            continue
        c, v = volume_binned(d)
        if not np.isfinite(v).any():
            continue
        tmid = np.median(d["t"])
        spec[sid] = np.interp(spherex_EDGES[:-1] + np.diff(spherex_EDGES) / 2, c, v, left=np.nan, right=np.nan)
        halves[sid] = [np.interp(spherex_EDGES[:-1] + np.diff(spherex_EDGES) / 2, *volume_binned(d, s), left=np.nan,
                                 right=np.nan) for s in (d["t"] < tmid, d["t"] >= tmid)]
    cen = spherex_EDGES[:-1] + np.diff(spherex_EDGES) / 2
    ids = [s for s in spec if sample[s].get("bp_rp") and sample[s].get("g")]
    feat = np.array([[float(sample[s]["bp_rp"]), (float(sample[s]["g"]) + 5 * np.log10(float(sample[s]["parallax"]) / 100.0))
                      / 2.0] for s in ids])
    S = np.array([spec[s] for s in ids])

    def excess(x, ref, lo, hi):
        m = (cen > lo) & (cen < hi) & np.isfinite(x) & np.isfinite(ref)
        return float(np.mean(x[m] / ref[m] - 1.0)) if m.sum() >= 2 else np.nan
    rows = []
    for i, s in enumerate(ids):
        d = np.hypot(*(feat - feat[i]).T)
        d[i] = np.inf
        nb = np.argsort(d)[:a.k]
        ref = np.nanmedian(S[nb], axis=0)
        row = dict(star=s, bp_rp=feat[i][0])
        for lo, hi in ((3.0, 4.0), (4.0, 5.0)):
            ex = excess(S[i], ref, lo, hi)
            # neighbours' own excesses set the scatter
            own = [excess(S[j], np.nanmedian(S[[k for k in nb if k != j]], axis=0), lo, hi) for j in nb]
            sd = 1.4826 * np.nanmedian(np.abs(np.array(own) - np.nanmedian(own)))
            row["ex_%g" % lo] = ex
            row["z_%g" % lo] = ex / sd if sd > 0 else np.nan
            row["halves_%g" % lo] = [excess(h, ref, lo, hi) / sd if sd > 0 else np.nan for h in halves[s]]
        rows.append(row)
    for lo in (3.0, 4.0):
        k = "z_%g" % lo
        z = np.array([r[k] for r in rows])
        print("%.0f-%.0f um: %d stars, robust spread of z %.2f; above 5 with both halves above 3: %d" % (
            lo, lo + 1, np.isfinite(z).sum(), 1.4826 * np.nanmedian(np.abs(z - np.nanmedian(z))),
            sum(1 for r in rows if r[k] > 5 and all(h > 3 for h in r["halves_%g" % lo]))))
        for r in sorted((r for r in rows if r[k] > 5), key=lambda r: -r[k])[:20]:
            print("  %s BP-RP %.2f excess %+.3f z %.1f halves %s" % (r["star"], r["bp_rp"], r["ex_%g" % lo], r[k],
                                                                    ["%.1f" % h for h in r["halves_%g" % lo]]))
    json.dump(rows, open(core.results("sphx_volume_excess.json"), "w"), default=float)


# stars whose 3-5 um excess the literature explains, with sources
volume_EXPLAINED = {
    "2660358032257156736": "G 29-38, a dusty white dwarf: silicate emission (Reach et al. 2005)",
}


def volume_blends(sid, row, radius=12.0):
    """Gaia neighbours within two SPHEREx pixels able to carry the 3-5 um light."""
    ra, dec, g = float(row["ra"]), float(row["dec"]), float(row["g"])
    out = []
    for x in spherex_gaia("SELECT source_id, ra, dec, phot_g_mean_mag AS g, bp_rp, parallax FROM gaiadr3.gaia_source "
                          "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f))"
                          % (ra, dec, radius / 3600.0)):
        if str(x["source_id"]) == sid or not x["g"]:
            continue
        gx, c = float(x["g"]), float(x["bp_rp"]) if x["bp_rp"] else 0.0
        if gx < g + 3.0 or (c > 2.5 and gx < g + 6.0):
            out.append("%.1f arcsec, G %.1f, BP-RP %.2f" % (sep_arcsec(ra, dec, float(x["ra"]), float(x["dec"])), gx, c))
    return out


def volume_vet_main(argv):
    """Vet excess stars against blends, RUWE and literature readings."""
    sample = {r["source_id"]: r for r in json.load(open(volume_SAMPLE))}
    path = core.results("sphx_volume_excess.json")
    rows = json.load(open(path))
    for r in rows:
        r.pop("vet", None)
        hot = [lo for lo in (3, 4) if r["z_%d" % lo] > 5 and all(h > 3 for h in r["halves_%d" % lo])]
        if not hot:
            continue
        s = sample[r["star"]]
        blends = volume_blends(r["star"], s)
        if blends:
            r["vet"] = "blend: " + "; ".join(blends)
        elif s.get("ruwe") and float(s["ruwe"]) > 1.4:
            r["vet"] = "unresolved companion: RUWE %.2f" % float(s["ruwe"])
        elif r["star"] in volume_EXPLAINED:
            r["vet"] = volume_EXPLAINED[r["star"]]
        print("%s BP-RP %.2f z(3-4) %.1f z(4-5) %.1f: %s" % (r["star"], r["bp_rp"], r["z_3"], r["z_4"],
                                                              r.get("vet") or "UNEXPLAINED"))
    json.dump(rows, open(path, "w"), default=float)


def volume_record_main(argv):
    rows = json.load(open(core.results("sphx_volume_excess.json")))
    lines = json.load(open(core.results("sphx_volume_lines.json")))
    hot = [r for r in rows if any(r["z_%d" % lo] > 5 and all(h > 3 for h in r["halves_%d" % lo]) for lo in (3, 4))]
    left = [r for r in hot if not r.get("vet")]
    ll = [q for q in lines["hits"] if q["shared"] < 3 and not q["known"]]
    nl = [q for q in lines["null"] if q["shared"] < 3 and not q["known"]]
    core.record_search(
        "sphx_volume", "SPHEREx spectra of every unsaturated star within 25 pc", archive="SPHEREx images (IRSA)",
        selection="Gaia DR3 parallax above 40 mas, Ks 9.5 or fainter or no 2MASS match and G above 15",
        code=os.path.abspath(__file__), entry="volume", n_targets=len(rows),
        params=dict(line_z=5, line_half=3, excess_z=5, excess_half=3, neighbours=15),
        null_summary="%d absorption-like line features past the shared and known-line filters" % len(nl),
        sensitivity="3-4 and 4-5 um flux against the 15 nearest stars in colour and absolute magnitude",
        ledger="SPHEREx at every star",
        result="%d emission-like line features past the filters; %d stars with a 3-5 um excess, %d explained, %d left"
               % (len(ll), len(hot), len(hot) - len(left), len(left)))
    core.clear_targets("sphx_volume")
    core.record_targets("sphx_volume", [dict(target="gaia_dr3:%s" % r["star"],
                                                 stat=max(r["z_3"], r["z_4"]) if np.isfinite(max(r["z_3"], r["z_4"]))
                                                 else None, unit="3-5 um excess z", vet=r.get("vet"))
                                            for r in rows])
    print("%d stars; %d line features, %d excess stars of which %d explained" % (len(rows), len(ll), len(hot),
                                                                                len(hot) - len(left)))


volume_MODES = {"sample": volume_sample_main, "spec": volume_spec_main, "lines": volume_lines_main,
                "excess": volume_excess_main, "vet": volume_vet_main, "record": volume_record_main}


def volume_cli():
    quiet_warnings()
    if len(sys.argv) < 2 or sys.argv[1] not in volume_MODES:
        print(volume_DOC)
        sys.exit(2)
    volume_MODES[sys.argv[1]](sys.argv[2:])


def spherex_cli():
    quiet_warnings()
    if len(sys.argv) < 2 or sys.argv[1] not in spherex_MODES:
        print(__doc__)
        sys.exit(2)
    spherex_MODES[sys.argv[1]](sys.argv[2:])


# --------------------------------------------------------------------------------------------------------------

sbf_DOC = r"""Surface-brightness-fluctuation distances beyond other methods in Cosmicflows-4, the far tail against the near.

    python -I seti.py infrared sbf [record] [--sigma 4] [--dmag 0.3] [--group-depth 0.1]"""

sbf_OTHERS = ("DMsnIa", "DMfp", "DMtf", "DMsnII", "DMtrgb", "DMceph", "DMmas")
sbf_VET = {10123: "NGC 1023: its group's other moduli place it at 2.3 Mpc against its own 10-11 Mpc",
           44182: "M64: the dust lane of the Black Eye over the bulge", 42038: "NGC 4565: an edge-on spiral's dust lane",
           13671: "ESO 358-56: a Fornax galaxy 0.5 mag behind the cluster's group modulus",
           13321: "NGC 1381: an edge-on Fornax S0 0.5 mag behind the cluster's group modulus"}


def sbf_fetch():
    p = core.data("cf4_sbf.json")
    if not os.path.exists(p):
        from astroquery.vizier import Vizier
        v = Vizier(row_limit=-1, columns=["**"], timeout=900)
        g = v.get_catalogs("J/ApJ/944/94/table2")[0]
        grp = v.get_catalogs("J/ApJ/944/94/groups")[0]

        def f(x):
            return None if x is np.ma.masked or x is None else float(x)
        gal = [dict(pgc=int(r["PGC"]), grp=int(r["1PGC"]), sbf=f(r["DMsbf"]), e_sbf=f(r["e_DMsbf"]), ra=f(r["RAJ2000"]),
                    dec=f(r["DEJ2000"]), vcmb=f(r["Vcmb"]), **{k: f(r[k]) for k in sbf_OTHERS},
                    **{"e_" + k: f(r["e_" + k]) for k in sbf_OTHERS}) for r in g]
        groups = {int(r["1PGC"]): dict(n=int(r["Ngal"]), **{k: f(r[k]) for k in ("DMsnIa", "DMfp", "DMtf", "DMsnII")},
                                       **{"e_" + k: f(r["e_" + k]) for k in ("DMsnIa", "DMfp", "DMtf", "DMsnII")}) for r in grp}
        json.dump(dict(gal=gal, groups=groups), open(p, "w"))
    return json.load(open(p))


def sbf_main(record=False, sigma=4.0, dmag=0.3, group_depth=0.1):
    d = sbf_fetch()
    groups = {int(k): v for k, v in d["groups"].items()}
    rows = []
    for gl in d["gal"]:
        if gl["sbf"] is None or not gl["e_sbf"]:
            continue
        own = [(gl[k], gl["e_" + k]) for k in sbf_OTHERS if gl[k] is not None and gl["e_" + k]]
        src = "galaxy"
        if not own:
            gr = groups.get(gl["grp"], {})
            own = [(gr[k], math.hypot(gr["e_" + k], group_depth)) for k in ("DMsnIa", "DMfp", "DMtf", "DMsnII")
                   if gr.get(k) is not None and gr.get("e_" + k)]
            src = "group"
        if not own:
            continue
        w = np.array([1 / e ** 2 for _, e in own])
        m = float(np.sum(w * np.array([x for x, _ in own])) / w.sum())
        e = float(1 / math.sqrt(w.sum()))
        diff = gl["sbf"] - m
        z = diff / math.hypot(gl["e_sbf"], e)
        rows.append(dict(pgc=gl["pgc"], ra=gl["ra"], dec=gl["dec"], sbf=gl["sbf"], other=m, diff=diff, z=z, src=src,
                         methods=[k for k in sbf_OTHERS if gl[k] is not None] if src == "galaxy" else ["group"]))
    z = np.array([r["z"] for r in rows])
    far = [r for r in rows if r["z"] > sigma and r["diff"] > dmag]
    near = [r for r in rows if r["z"] < -sigma and r["diff"] < -dmag]
    print("%d SBF galaxies with another distance (%d in the galaxy, %d through its group); median difference %+.3f mag, "
          "robust z scatter %.2f" % (len(rows), sum(r["src"] == "galaxy" for r in rows), sum(r["src"] == "group" for r in rows),
                                     float(np.median([r["diff"] for r in rows])), 1.4826 * float(np.median(np.abs(z - np.median(z))))))
    print("too far (fluctuations too weak) by %g mag at %g sigma: %d; too near: %d" % (dmag, sigma, len(far), len(near)))
    for r in sorted(far, key=lambda r: -r["z"]) + sorted(near, key=lambda r: r["z"]):
        print("  PGC %d at %.4f %+.4f: SBF %.2f against %.2f (%s %s), %+.2f mag, %+.1f sigma"
              % (r["pgc"], r["ra"], r["dec"], r["sbf"], r["other"], r["src"], ",".join(r["methods"]), r["diff"], r["z"]))
    json.dump(dict(rows=rows, far=far, near=near), open(core.results("sbf_deficit.json"), "w"), indent=1)
    if record:
        core.record_search(
            "cf4_sbf_deficit", "Surface-brightness-fluctuation deficit", archive="Cosmicflows-4 (J/ApJ/944/94)",
            selection="%d galaxies with an SBF modulus and another method in the galaxy or its group" % len(rows),
            code=os.path.abspath(__file__), entry="sbf", n_targets=len(rows), params=dict(z=sigma, mag=dmag, group_depth=group_depth),
            null_summary="the near tail, where globular clusters and backgrounds push: %d" % len(near),
            sensitivity="fluctuations %.0f per cent too weak (%g mag) at %g sigma of the combined errors" % (100 * (1 - 10 ** (-0.4 * dmag)), dmag, sigma),
            ledger="Surface-brightness-fluctuation deficit",
            result="%d galaxies too far by SBF against %d too near; the far ones %s"
                   % (len(far), len(near), "; ".join(sbf_VET.get(r["pgc"], "PGC %d" % r["pgc"]) for r in far)))
        core.clear_targets("cf4_sbf_deficit")
        core.record_targets("cf4_sbf_deficit", [dict(
            target="pgc:%d" % r["pgc"], stat=r["z"], unit="SBF minus other moduli, sigma", vet=sbf_VET.get(r["pgc"]),
            note="%+.2f mag against %s" % (r["diff"], r["src"] if r["src"] == "group" else ",".join(r["methods"]))) for r in rows])
        print("recorded")


udg_DOC = r"""Mid-infrared sources at SMUDGes ultra-diffuse galaxy centroids without Gaia stars, against offset-position chance rates.

    python -I seti.py infrared udg [record] [--radius-arcsec 3] [--star-arcsec 2] [--snr 5] [--w1w2 0.5] [--control-offset-arcsec 60]"""


def udg_match(ra, dec, cat, radius):
    from astropy.table import Table
    from astroquery.xmatch import XMatch
    import astropy.units as u
    t = Table(dict(idx=np.arange(len(ra)), ra=np.asarray(ra), dec=np.asarray(dec)))
    out = []
    for k in range(0, len(t), 5000):
        out.append(XMatch.query(cat1=t[k:k + 5000], cat2=cat, max_distance=radius * u.arcsec, colRA1="ra", colDec1="dec"))
    from astropy.table import vstack
    return vstack(out) if out else None


def udg_main(record=False, radius_arcsec=3.0, star_arcsec=2.0, snr=5.0, w1w2=0.5, control_offset_arcsec=60.0):
    p = core.data("smudges_dr9.json")
    if not os.path.exists(p):
        from astroquery.vizier import Vizier
        t = Vizier(row_limit=-1, columns=["SMDG", "RAJ2000", "DEJ2000", "Re", "mu0g"], timeout=900).get_catalogs("J/ApJS/267/27/table2")[0]
        json.dump([[str(r["SMDG"]), float(r["RAJ2000"]), float(r["DEJ2000"]), num(r["Re"]), num(r["mu0g"])] for r in t], open(p, "w"))
    g = json.load(open(p))
    ra = np.array([x[1] for x in g])
    dec = np.array([x[2] for x in g])
    res = {}
    for lab, dr in (("udg", 0.0), ("control", control_offset_arcsec)):
        rr, dd = ra, dec + dr / 3600.0
        w = udg_match(rr, dd, "vizier:II/328/allwise", radius_arcsec)
        s = udg_match(rr, dd, "vizier:I/355/gaiadr3", star_arcsec)
        stars = set(int(i) for i in s["idx"]) if s is not None and len(s) else set()
        best = {}
        for r in w:
            i = int(r["idx"])
            if i not in best or r["angDist"] < best[i]["angDist"]:
                best[i] = r
        det = []
        for i, r in best.items():
            def f(c):
                v = r[c]
                return None if np.ma.is_masked(v) else float(v)
            e3, e4 = f("e_W3mag"), f("e_W4mag")
            s3, s4 = (1.0857 / e3 if e3 else None), (1.0857 / e4 if e4 else None)
            w1, w2 = f("W1mag"), f("W2mag")
            if ((s3 or 0) >= snr or (s4 or 0) >= snr):
                det.append(dict(i=i, w1=w1, w2=w2, w3=f("W3mag"), w4=f("W4mag"), s3=s3, s4=s4, sep=float(r["angDist"]),
                                agn=w1 is not None and w2 is not None and w1 - w2 >= w1w2, star=i in stars))
        clean = [x for x in det if not x["agn"] and not x["star"]]
        res[lab] = dict(n=len(g), matched=len(best), det=len(det), clean=clean)
        print("%s: %d positions, %d AllWISE sources within %g arcsec, %d at %g sigma in W3 or W4, %d of them neither AGN-coloured "
              "nor a Gaia star" % (lab, len(g), len(best), radius_arcsec, len(det), snr, len(clean)), flush=True)
    for x in sorted(res["udg"]["clean"], key=lambda x: -max(x["s3"] or 0, x["s4"] or 0))[:20]:
        gg = g[x["i"]]
        print("  %s %.5f %+.5f Re %s mu0g %s: W1 %s W2 %s W3 %s (%.1f) W4 %s (%s), %.1f arcsec"
              % (gg[0], gg[1], gg[2], gg[3], gg[4], x["w1"], x["w2"], x["w3"], x["s3"] or 0, x["w4"], x["s4"], x["sep"]))
    json.dump(dict(res=res, ids=[g[x["i"]][0] for x in res["udg"]["clean"]]), open(core.results("udg_waste_heat.json"), "w"), default=str)
    if record:
        u, c = res["udg"], res["control"]
        core.record_search(
            "smudges_waste_heat", "Waste heat in ultra-diffuse galaxies", archive="SMUDGes DR9 (J/ApJS/267/27); AllWISE and "
            "Gaia DR3 through CDS XMatch", selection="all %d SMUDGes galaxies" % u["n"], code=os.path.abspath(__file__),
            entry="udg", n_targets=u["n"], params=dict(radius_arcsec=radius_arcsec, star_arcsec=star_arcsec, snr=snr, w1w2=w1w2,
                                                       control_offset_arcsec=control_offset_arcsec),
            null_summary="positions %g arcsec away: %d clean W3 or W4 sources" % (control_offset_arcsec, len(c["clean"])),
            sensitivity="a W3 or W4 source at %g sigma at the centroid (about 1 mJy at 12 um, 6 mJy at 22 um at 5 sigma)" % snr,
            ledger="Waste heat in ultra-diffuse and HI-dark galaxies",
            result="%d centroid sources at %g sigma in W3 or W4 that are neither AGN-coloured nor Gaia stars, against %d at "
                   "the offset positions" % (len(u["clean"]), snr, len(c["clean"])))
        core.clear_targets("smudges_waste_heat")
        ids = {x["i"]: x for x in u["clean"]}
        core.record_targets("smudges_waste_heat", [dict(
            target="smudges:%s" % gg[0], stat=float(max(ids[i]["s3"] or 0, ids[i]["s4"] or 0)) if i in ids else 0.0,
            unit="W3 or W4 S/N of a clean centroid source") for i, gg in enumerate(g)])
        print("recorded")


debs_DOC = r"""Missing light in DEBCat detached binaries: Gaia G fainter than the summed component luminosities by 0.3 mag at 3 sigma.

    python -I seti.py infrared debs fetch
    python -I seti.py infrared debs analyse
    python -I seti.py infrared debs record"""

debs_ROOT = os.environ.get("DEBS_ROOT", core.data("debcat"))
debs_URL = "https://www.astro.keele.ac.uk/jkt/debcat/"
debs_OUT = "debs_missing_flux.json"
debs_TEFF_MAX, debs_DMAX, debs_DEFICIT, debs_RUWE = 8000.0, 1000.0, 0.3, 1.4
debs_TEFF_BC = ((8000, 0.12), (7200, 0.10), (6500, 0.09), (6000, 0.08), (5800, 0.07), (5500, 0.03), (5000, -0.10), (4500, -0.30), (4000, -0.60), (3500, -1.20), (3000, -2.00), (2800, -2.40))
# dwarf intrinsic B-V against Teff (Pecaut and Mamajek 2013)
debs_TEFF_BV = ((10000, 0.00), (8500, 0.14), (8000, 0.19), (7500, 0.27), (7000, 0.34), (6500, 0.44), (6000, 0.60), (5800, 0.65), (5500, 0.74), (5000, 0.90), (4500, 1.10), (4000, 1.42), (3500, 1.55), (3000, 1.75))
debs_AG_PER_EBV = 2.7


def debs_parse(html):
    """DEBCat rows: name, period, V, B-V, types and per-component mass, radius, log g, log Teff, log L with errors."""
    import re

    def val(s):
        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", s or "")
        return float(m.group(0)) if m else None

    def lines(c):
        return [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", p)).replace("&plusmn;", "+-").replace("&nbsp;", "").strip() for p in re.split(r"<BR>", c, flags=re.I)]

    def pair(c):
        return [(val(p.split("+-")[0]), val(p.split("+-")[1]) if "+-" in p else None) for p in lines(c) if p]
    out = []
    for r in re.findall(r"<TR[^>]*>(.*?)</TR>", html, re.S):
        cells = re.findall(r"<TD[^>]*>(.*?)</TD>", r, re.S)
        if len(cells) < 10:
            continue
        name = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cells[0])).strip()
        V = pair(cells[2])
        out.append(dict(name=name, P=val(cells[1]), V=V[0][0] if V else None, bv=V[1][0] if len(V) > 1 else None, sptype=[s for s in lines(cells[3]) if s],
                        M=pair(cells[4]), R=pair(cells[5]), logg=pair(cells[6]), logT=pair(cells[7]), logL=pair(cells[8]), feh=pair(cells[9])))
    return out


def debs_fetch():
    import re
    os.makedirs(debs_ROOT, exist_ok=True)
    p = os.path.join(debs_ROOT, "debcat.json")
    if not os.path.exists(p):
        json.dump(debs_parse(core.get(debs_URL, timeout=120).decode("utf-8", "replace")), open(p, "w"), indent=1)
    d = json.load(open(p))
    sp = os.path.join(debs_ROOT, "simbad.json")
    got = json.load(open(sp)) if os.path.exists(sp) else {}
    greek = {"&alpha;": "alf", "&beta;": "bet", "&gamma;": "gam", "&delta;": "del", "&epsilon;": "eps", "&zeta;": "zet", "&eta;": "eta", "&theta;": "tet", "&iota;": "iot",
             "&kappa;": "kap", "&lambda;": "lam", "&mu;": "mu.", "&nu;": "nu.", "&xi;": "ksi", "&omicron;": "omi", "&pi;": "pi.", "&rho;": "rho", "&sigma;": "sig", "&tau;": "tau",
             "&upsilon;": "ups", "&phi;": "phi", "&chi;": "chi", "&psi;": "psi", "&omega;": "ome"}

    def norm(s):
        return re.sub(r"\s+", "", s).lower()

    def variants(n):
        out = [n, "V* " + n, n.replace("BD +", "BD+").replace("CD -", "CD-").replace("CPD -", "CPD-")]
        for k, v in greek.items():
            if k in n:
                out = ["* " + n.replace(k, v).replace(" 2 ", "02 "), "* " + n.replace(k, v)] + out
        if n.startswith("OGLE"):
            out = [n.replace("OGLE-", "OGLE "), n.replace("OGLE ", "OGLE-")] + out
        if n.startswith("HV "):
            out = ["Cl* NGC 2070 HV " + n[3:]] + out
        return out
    missing = [x["name"] for x in d if x["name"] not in got]
    alts = {norm(v): n for n in missing for v in variants(n)}
    keys = sorted({v for n in missing for v in variants(n)})
    for k in range(0, len(keys), 60):
        q = ("SELECT i.id AS name, b.main_id, b.ra, b.dec, b.otype, b.sp_type FROM ident i JOIN basic b ON i.oidref = b.oid WHERE i.id IN (%s)"
             % ",".join("'%s'" % x.replace("'", "''") for x in keys[k:k + 60]))
        for r in core.simbad(q):
            n = alts.get(norm(r["name"]))
            if n and n not in got:
                got[n] = r
    json.dump(got, open(sp, "w"), indent=1)
    rows = [dict(source_id=i, ra=float(r["ra"]), dec=float(r["dec"]), name=n) for i, (n, r) in enumerate(got.items()) if r.get("ra") not in (None, "")]
    gx = {}
    for cat, key in (("I/355/gaiadr3", "gaia"), ("I/355/paramp", "paramp")):
        m = core.xmatch(rows, cat, 3.0, id_col="source_id", ra_col="ra", dec_col="dec", timeout=600)
        for x in m:
            i = int(x["source_id"])
            if i not in gx or key not in gx[i] or float(x["angDist"]) < float(gx[i][key]["angDist"]):
                gx.setdefault(i, {})[key] = x
    out = {rows[i]["name"]: v for i, v in gx.items()}
    json.dump(out, open(os.path.join(debs_ROOT, "gaia.json"), "w"), indent=1)
    print("%d systems, %d resolved in SIMBAD, %d with Gaia DR3 within 3 arcsec" % (len(d), len(got), len(out)))


def debs_bc(teff):
    x, y = zip(*sorted(debs_TEFF_BC))
    return float(np.interp(teff, x, y))


def debs_bv0(teff):
    x, y = zip(*sorted(debs_TEFF_BV))
    return float(np.interp(teff, x, y))


def debs_analyse():
    d = json.load(open(os.path.join(debs_ROOT, "debcat.json")))
    gaia = json.load(open(os.path.join(debs_ROOT, "gaia.json")))
    simbad = json.load(open(os.path.join(debs_ROOT, "simbad.json")))
    rows = []
    for s in d:
        g = gaia.get(s["name"], {}).get("gaia")
        if not g or len(s["logL"]) < 2 or len(s["logT"]) < 2 or any(v[0] is None for v in s["logL"][:2] + s["logT"][:2]):
            continue
        plx, eplx, G = num(g.get("Plx")), num(g.get("e_Plx")), num(g.get("Gmag"))
        if not plx or plx <= 0 or G is None:
            continue
        pp = gaia[s["name"]].get("paramp") or {}
        ag = num(pp.get("AG")) if pp else None
        teffs = [10 ** v[0] for v in s["logT"][:2]]
        L = [10 ** v[0] for v in s["logL"][:2]]
        eL = [math.log(10) * (v[1] or 0.05) * lv for v, lv in zip(s["logL"][:2], L)]
        # summed G flux of the components at the distance
        mbol = [4.74 - 2.5 * math.log10(lv) for lv in L]
        MG = [mb - debs_bc(t) for mb, t in zip(mbol, teffs)]
        MG_sum = -2.5 * math.log10(sum(10 ** (-0.4 * m) for m in MG))
        dist = 1000.0 / plx
        # extinction: larger of the light-weighted colour excess and Gaia A_G
        wts = [10 ** (-0.4 * m) for m in MG]
        bv0 = sum(w * debs_bv0(t) for w, t in zip(wts, teffs)) / sum(wts)
        ebv = max(0.0, s["bv"] - bv0) if s.get("bv") is not None else None
        ag_bv = debs_AG_PER_EBV * ebv if ebv is not None else None
        a_g = max(ag or 0.0, ag_bv or 0.0)
        e_ag = 0.15 if ag_bv is not None and ag_bv >= (ag or 0.0) else (0.05 if ag is not None else (0.1 if dist < 300 else 0.3))
        G_pred = MG_sum + 5 * math.log10(dist) - 5 + a_g
        # errors from luminosities, parallax, correction and extinction
        var = sum((1.0857 * e / lv) ** 2 * (lv / sum(L)) ** 2 for e, lv in zip(eL, L)) + (2.1715 * (eplx or 0.1) / plx) ** 2 + 2 * 0.05 ** 2 + e_ag ** 2
        err = math.sqrt(var)
        deficit = G - G_pred
        ruwe = num(g.get("RUWE"))
        rows.append(dict(name=s["name"], main_id=simbad.get(s["name"], {}).get("main_id"), otype=simbad.get(s["name"], {}).get("otype"), P=s["P"], teff=teffs, L=L, dist=dist, plx=plx,
                         G=G, G_pred=G_pred, ag=ag, ag_bv=ag_bv, bv=s.get("bv"), bv0=bv0, deficit=deficit, err=err, z=deficit / err, hot=max(teffs) > debs_TEFF_MAX, far=dist > debs_DMAX,
                         ruwe=ruwe, bprp=num(g.get("BP-RP")), astrometric=ruwe is not None and ruwe <= debs_RUWE))
    clean = [r for r in rows if not r["hot"] and not r["far"] and r["astrometric"]]
    dz = np.array([r["deficit"] for r in clean])
    print("%d systems with Gaia parallaxes and both components' luminosities; %d cooler than %.0f K within %.0f pc at RUWE %.1f or less: deficit G_obs - G_pred median %+.3f, "
          "robust scatter %.3f mag; beyond %+.1f mag at 3 sigma: %d too faint, %d too bright" % (len(rows), len(clean), debs_TEFF_MAX, debs_DMAX, debs_RUWE, np.median(dz),
                                                                                                 1.4826 * np.median(np.abs(dz - np.median(dz))), debs_DEFICIT,
                                                                                                 sum(1 for r in clean if r["deficit"] > debs_DEFICIT and r["z"] > 3), sum(1 for r in clean if r["deficit"] < -debs_DEFICIT and r["z"] < -3)))
    for r in sorted(clean, key=lambda r: -r["z"])[:15]:
        print("  %-20s %s (%s): P %.2f d, Teff %.0f + %.0f K, L %.2f + %.2f, %.0f pc; G %.2f, predicted %.2f (A_G %s, from B-V %s %s); deficit %+.2f +- %.2f (%+.1f sigma); RUWE %s" % (
            r["name"], r["main_id"] or "", r["otype"] or "", r["P"] or 0, r["teff"][0], r["teff"][1], r["L"][0], r["L"][1], r["dist"], r["G"], r["G_pred"],
            "%.2f" % r["ag"] if r["ag"] is not None else "-", "%.2f" % r["bv"] if r["bv"] is not None else "-", "%.2f" % r["ag_bv"] if r["ag_bv"] is not None else "-",
            r["deficit"], r["err"], r["z"], "%.2f" % r["ruwe"] if r["ruwe"] else "-"))
    json.dump(dict(n=len(rows), n_clean=len(clean), median=float(np.median(dz)), scatter=float(1.4826 * np.median(np.abs(dz - np.median(dz)))), rows=rows,
                   teff_max=debs_TEFF_MAX, dmax=debs_DMAX, deficit=debs_DEFICIT, ruwe=debs_RUWE), open(core.results(debs_OUT), "w"), indent=1)


def debs_record():
    d = json.load(open(core.results(debs_OUT)))
    clean = [r for r in d["rows"] if not r["hot"] and not r["far"] and r["astrometric"]]
    faint = [r for r in clean if r["deficit"] > d["deficit"] and r["z"] > 3]
    bright = [r for r in clean if r["deficit"] < -d["deficit"] and r["z"] < -3]
    core.record_search(
        "debs_missing_flux", "Missing bolometric flux, eclipsing binaries", archive="DEBCat (Southworth); SIMBAD; Gaia DR3 by CDS XMatch", release="DEBCat as served",
        selection="detached eclipsing binaries with both components' luminosities, Gaia parallaxes at RUWE %.1f or less, components below %.0f K and within %.0f pc" % (d["ruwe"], d["teff_max"], d["dmax"]),
        code=os.path.abspath(__file__), entry="debs", n_targets=len(clean), params=dict(teff_max=d["teff_max"], dmax=d["dmax"], deficit=d["deficit"], ruwe=d["ruwe"], ag_per_ebv=debs_AG_PER_EBV),
        null_summary="deficit G_obs - G_pred median %+.3f, robust scatter %.3f mag over %d systems" % (d["median"], d["scatter"], len(clean)),
        sensitivity="a deficit of %.1f mag at 3 sigma, 30 per cent of the summed surface emission" % d["deficit"], ledger="Missing bolometric flux",
        result="%d systems: %d too faint by %.1f mag at 3 sigma (%s), %d too bright (%s)" % (
            len(clean), len(faint), d["deficit"], ", ".join("%s %+.2f" % (r["name"], r["deficit"]) for r in faint[:8]), len(bright), ", ".join("%s %+.2f" % (r["name"], r["deficit"]) for r in bright[:8])))
    core.clear_targets("debs_missing_flux")
    core.record_targets("debs_missing_flux", [dict(target="simbad:" + (r["main_id"] or r["name"]), stat=r["deficit"], unit="G observed minus G predicted from the components' surfaces, mag",
                                                       vet=("too faint" if r in faint else ("too bright: third light or blend" if r in bright else "within the spread")) + ("; hot" if r["hot"] else "") + ("; beyond 1 kpc" if r["far"] else "")
                                                           + ("" if r["astrometric"] else "; RUWE above %.1f, parallax not trusted" % d["ruwe"]),
                                                       note="%s, P %.2f d, %.0f pc, Teff %.0f and %.0f K, deficit %+.2f +- %.2f" % (r["name"], r["P"] or 0, r["dist"], r["teff"][0], r["teff"][1], r["deficit"], r["err"])) for r in d["rows"]])
    print("recorded %d systems" % len(d["rows"]))


def debs_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": debs_fetch, "analyse": debs_analyse, "record": debs_record}[mode]()


bluefloor_DOC = r"""Point sources bluer than Rayleigh-Jeans in Euclid Q1 (I_E - H_E below -2.3 at 3 sigma), vetted by flags, blending and Gaia colour.

    python -I seti.py infrared bluefloor analyse
    python -I seti.py infrared bluefloor record"""

bluefloor_ROOT = os.environ.get("BLUEFLOOR_ROOT", core.data("euclid_blue"))
bluefloor_OUT = "euclid_bluefloor.json"
bluefloor_LAMBDA = dict(vis=0.72, y=1.08, j=1.37, h=1.77)  # pivot wavelengths (um)
bluefloor_CUT = -2.3


def bluefloor_rj(a, b):
    """Rayleigh-Jeans F_a/F_b (F_nu proportional to nu^2)."""
    return (bluefloor_LAMBDA[b] / bluefloor_LAMBDA[a]) ** 2


def bluefloor_analyse():
    rows = json.load(open(os.path.join(bluefloor_ROOT, "mer_blue.json")))
    out = []
    for r in rows:
        f = {b: num(r["flux_%s_%s" % (b, "psf" if b == "vis" else "templfit")]) for b in ("vis", "y", "j", "h")}
        e = {b: num(r["fluxerr_%s_%s" % (b, "psf" if b == "vis" else "templfit")]) for b in ("vis", "y", "j", "h")}
        if any(v is None or v <= 0 for v in f.values()) or any(v is None or v <= 0 for v in e.values()):
            continue
        c = -2.5 * math.log10(f["vis"] / f["h"])
        ec = 1.0857 * math.hypot(e["vis"] / f["vis"], e["h"] / f["h"])
        flags = [int(num(r.get("flag_%s" % b)) or 0) for b in ("vis", "y", "j", "h")]
        super_rj = f["y"] / f["h"] > bluefloor_rj("y", "h") and f["j"] / f["h"] > bluefloor_rj("j", "h") and f["vis"] / f["y"] > bluefloor_rj("vis", "y")
        out.append(dict(object_id=r["object_id"], tileid=r["tileid"], ra=float(r["ra"]), dec=float(r["dec"]), flux=f, err=e, colour=c, e_colour=ec,
                        snr_min=min(f[b] / e[b] for b in f), flags=flags, point_like=num(r.get("point_like_prob")), spurious=num(r.get("spurious_prob")),
                        blended=num(r.get("blended_prob")), variable=int(num(r.get("variable_flag")) or 0), mu_max=num(r.get("mu_max")), mumax_minus_mag=num(r.get("mumax_minus_mag")),
                        gaia=str(r.get("gaia_id") or "").strip() or None, super_rj=super_rj,
                        candidate=c + 3 * ec < bluefloor_CUT and super_rj and not any(flags) and (num(r.get("blended_prob")) or 0) < 0.1 and not int(num(r.get("variable_flag")) or 0)))
    cols = np.array([x["colour"] for x in out])
    cand = [x for x in out if x["candidate"]]
    print("%d point sources at S/N above 10 in VIS, Y, J and H with I_E - H_E below the Rayleigh-Jeans -1.95: colour median %.2f, 1st percentile %.2f; %d below %.1f, %d of them at 3 sigma with NISP fluxes "
          "falling faster than nu^2, clean flags, unblended and not variable" % (len(out), np.median(cols), np.percentile(cols, 1), int(np.sum(cols < bluefloor_CUT)), bluefloor_CUT, len(cand)))
    if cand:
        pos = [dict(source_id=i, ra=x["ra"], dec=x["dec"]) for i, x in enumerate(cand)]
        gaia = {}
        for m in core.xmatch(pos, "I/355/gaiadr3", 2.0, id_col="source_id", ra_col="ra", dec_col="dec", timeout=600):
            i = int(m["source_id"])
            if i not in gaia or float(m["angDist"]) < float(gaia[i]["angDist"]):
                gaia[i] = m
        for i, x in enumerate(cand):
            g = gaia.get(i)
            x["gaia_match"] = dict(source=g.get("Source"), G=num(g.get("Gmag")), bprp=num(g.get("BP-RP")), plx=num(g.get("Plx")), pm=math.hypot(num(g.get("pmRA")) or 0, num(g.get("pmDE")) or 0), sep=float(g["angDist"])) if g else None
            x["explained"] = ("Gaia counterpart at BP-RP %.2f, a thermal colour the Euclid fluxes contradict, so a photometric defect" % x["gaia_match"]["bprp"]) if x["gaia_match"] and x["gaia_match"]["bprp"] is not None and x["gaia_match"]["bprp"] > 0.0 else None
            print("  %s tile %s (%.5f %+.5f): I_E %.2f, I_E - H_E %.2f +- %.2f, Y/J/H %.1f/%.1f/%.1f uJy, S/N min %.0f, point-like %.2f, mu_max %s; Gaia %s; %s" % (
                x["object_id"], x["tileid"], x["ra"], x["dec"], 23.9 - 2.5 * math.log10(x["flux"]["vis"]), x["colour"], x["e_colour"], x["flux"]["y"], x["flux"]["j"], x["flux"]["h"], x["snr_min"],
                x["point_like"] or 0, "%.1f" % x["mu_max"] if x["mu_max"] else "-", ("G %.1f BP-RP %s plx %s mas pm %.0f" % (x["gaia_match"]["G"] or 0, "%.2f" % x["gaia_match"]["bprp"] if x["gaia_match"]["bprp"] is not None else "-",
                                                                                                               "%.2f" % x["gaia_match"]["plx"] if x["gaia_match"]["plx"] is not None else "-", x["gaia_match"]["pm"])) if x["gaia_match"] else "none",
                x["explained"] or "unexplained"))
    json.dump(dict(n=len(out), n_below=int(np.sum(cols < bluefloor_CUT)), n_cand=len(cand), cands=cand, params=dict(cut=bluefloor_CUT, pivots=bluefloor_LAMBDA)),
              open(core.results(bluefloor_OUT), "w"), indent=0, default=str)


def bluefloor_record():
    d = json.load(open(core.results(bluefloor_OUT)))
    cand = d["cands"]
    left = [x for x in cand if not x.get("explained")]
    core.record_search(
        "euclid_bluefloor", "Sources bluer than the Rayleigh-Jeans limit", archive="Euclid Q1 MER catalogue (IRSA); Gaia DR3 by CDS XMatch", release="Euclid Q1",
        selection="point-like VIS detections with clean flags at S/N above 10 in VIS, Y, J and H and F_VIS above six times F_H (%d)" % d["n"], code=os.path.abspath(__file__), entry="bluefloor",
        n_targets=d["n"], params=d["params"], null_summary="the colour distribution of the selected point sources against the Rayleigh-Jeans line at -1.95",
        sensitivity="I_E - H_E below %.1f at three sigma, a flux ratio above 8.3 where no thermal source exceeds 6.0" % d["params"]["cut"], ledger="Euclid Q1 sources bluer than the Rayleigh-Jeans limit",
        result="%d sources below %.1f, %d of them at 3 sigma with super-Rayleigh-Jeans NISP fluxes and clean flags, %d with a Gaia counterpart whose colour contradicts the Euclid fluxes, %d left" % (
            d["n_below"], d["params"]["cut"], d["n_cand"], len(cand) - len(left), len(left)))
    core.clear_targets("euclid_bluefloor")
    core.record_targets("euclid_bluefloor", [dict(target="euclid:%s" % x["object_id"], stat=x["colour"], unit="I_E - H_E (AB)", vet=x.get("explained"),
                                                       note="tile %s, I_E %.2f, colour %.2f +- %.2f, Gaia %s" % (x["tileid"], 23.9 - 2.5 * math.log10(x["flux"]["vis"]), x["colour"], x["e_colour"],
                                                                                                                 x["gaia_match"]["source"] if x.get("gaia_match") else "none")) for x in cand])
    print("recorded %d candidates" % len(cand))


def bluefloor_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": bluefloor_analyse, "record": bluefloor_record}[mode]()


wddust_DOC = r"""Warm dust without pollution at DESI white dwarfs: unWISE excess at 5 sigma with 600-2000 K colours and Ca II K below 50 mA, against matched controls.

    python -I seti.py infrared wddust fetch
    python -I seti.py infrared wddust analyse
    python -I seti.py infrared wddust spectra
    python -I seti.py infrared wddust record"""

wddust_ROOT = os.environ.get("WDDUST_ROOT", core.data("desi_wd"))
wddust_OUT = "wd_dust_pollution.json"
wddust_ZP = dict(w1=309.54, w2=171.79)  # Vega zero points (Jy)
wddust_LAM = dict(w1=3.368, w2=4.618)                  # um
wddust_TMIN, wddust_TMAX, wddust_SIGMA, wddust_EW = 600.0, 2000.0, 5.0, 50.0
wddust_CAK, wddust_MGII = 3934.78, 4482.3  # vacuum (A)


def wddust_fetch():
    os.makedirs(wddust_ROOT, exist_ok=True)
    p = os.path.join(wddust_ROOT, "wd_unwise_gaia.json")
    if os.path.exists(p):
        print("have", p)
        return
    q = ("SELECT z.id, z.targetid, z.mean_fiber_ra, z.mean_fiber_dec, z.subtype, z.zwarn, u.flux_w1, u.dflux_w1, u.flux_w2, u.dflux_w2, u.fracflux_w1, u.fracflux_w2, "
         "u.flags_unwise_w1, u.flags_unwise_w2, xu.distance AS d_unwise, g.source_id, g.parallax, g.parallax_error, g.phot_g_mean_mag, g.bp_rp, g.ruwe, "
         "g.phot_bp_rp_excess_factor, xg.distance AS d_gaia FROM desi_dr1.zpix AS z JOIN desi_dr1.x1p5__zpix__unwise_dr1__object AS xu ON xu.id1 = z.id "
         "JOIN unwise_dr1.object AS u ON u.unwise_objid = xu.id2 JOIN desi_dr1.x1p5__zpix__gaia_dr3__gaia_source AS xg ON xg.id1 = z.id "
         "JOIN gaia_dr3.gaia_source AS g ON g.source_id = xg.id2 WHERE z.spectype = 'STAR' AND z.subtype = 'WD' AND z.zcat_primary = 't'")
    rows = core.datalab(q)
    json.dump(rows, open(p, "w"))
    print("%d rows" % len(rows))


def wddust_planck_ratio(T):
    """Black-body F_nu(W2)/F_nu(W1) at T."""
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    def b(lam_um):
        nu = c / (lam_um * 1e-6)
        return nu ** 3 / math.expm1(h * nu / (k * T))
    return b(wddust_LAM["w2"]) / b(wddust_LAM["w1"])


def wddust_temperature(ratio):
    """Black-body temperature with W2/W1 ratio, by bisection over 200-5000 K; None outside."""
    lo, hi = 200.0, 5000.0
    if ratio >= wddust_planck_ratio(lo) or ratio <= wddust_planck_ratio(hi):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if wddust_planck_ratio(mid) > ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wddust_stars():
    rows = json.load(open(os.path.join(wddust_ROOT, "wd_unwise_gaia.json")))
    by = {}
    for r in rows:
        s = by.setdefault(r["targetid"], dict(targetid=r["targetid"], ra=float(r["mean_fiber_ra"]), dec=float(r["mean_fiber_dec"]), zwarn=int(num(r["zwarn"]) or 0), source_id=r["source_id"],
                                              plx=num(r["parallax"]), e_plx=num(r["parallax_error"]), G=num(r["phot_g_mean_mag"]), bprp=num(r["bp_rp"]), ruwe=num(r["ruwe"]),
                                              excess_factor=num(r["phot_bp_rp_excess_factor"]), d_gaia=num(r["d_gaia"]), n_unwise=0, w={}))
        s["n_unwise"] += 1
        for b in ("w1", "w2"):
            f, df = num(r["flux_%s" % b]), num(r["dflux_%s" % b])
            if f and df and f > 0 and (b not in s["w"] or f > s["w"][b]["flux"]):
                s["w"][b] = dict(flux=f, dflux=df, frac=num(r["fracflux_%s" % b]), flags=int(num(r["flags_unwise_%s" % b]) or 0), sep=num(r["d_unwise"]))
    return list(by.values())


def wddust_analyse():
    stars = wddust_stars()
    ok = [s for s in stars if s["plx"] and s["e_plx"] and s["plx"] / s["e_plx"] > 5 and s["G"] is not None and s["bprp"] is not None and (s["d_gaia"] or 9) < 1.0
          and "w1" in s["w"] and s["w"]["w1"]["flux"] / s["w"]["w1"]["dflux"] > 5 and "w2" in s["w"] and s["w"]["w2"]["flux"] / s["w"]["w2"]["dflux"] > 3]
    for s in ok:
        for b in ("w1", "w2"):
            s["w"][b]["mag"] = 22.5 - 2.5 * math.log10(s["w"][b]["flux"])
            s["w"][b]["e_mag"] = 1.0857 * s["w"][b]["dflux"] / s["w"][b]["flux"]
    # photosphere locus: running median of G - W in BP-RP bins with robust scatter
    bins = np.arange(-0.8, 1.6, 0.1)
    locus = {}
    for b in ("w1", "w2"):
        x = np.array([s["bprp"] for s in ok])
        y = np.array([s["G"] - s["w"][b]["mag"] for s in ok])
        med, sd, ctr = [], [], []
        for lo in bins:
            m = (x >= lo) & (x < lo + 0.1)
            if m.sum() >= 15:
                ctr.append(lo + 0.05)
                med.append(float(np.median(y[m])))
                sd.append(float(1.4826 * np.median(np.abs(y[m] - np.median(y[m])))))
        locus[b] = (np.array(ctr), np.array(med), np.array(sd))
    cands = []
    for s in ok:
        exc = {}
        for b in ("w1", "w2"):
            ctr, med, sd = locus[b]
            if not (ctr[0] - 0.05 <= s["bprp"] <= ctr[-1] + 0.05):
                exc = None
                break
            mu, sc = float(np.interp(s["bprp"], ctr, med)), float(np.interp(s["bprp"], ctr, sd))
            d = (s["G"] - s["w"][b]["mag"]) - mu
            exc[b] = dict(excess_mag=d, sigma=d / math.hypot(s["w"][b]["e_mag"], sc), photosphere_mag=s["G"] - mu)
        if not exc:
            continue
        s["excess"] = exc
        if all(exc[b]["sigma"] >= wddust_SIGMA for b in exc):
            # excess flux per band and its black body
            df = {}
            for b in ("w1", "w2"):
                f_obs = wddust_ZP[b] * 10 ** (-0.4 * s["w"][b]["mag"])
                f_phot = wddust_ZP[b] * 10 ** (-0.4 * exc[b]["photosphere_mag"])
                df[b] = f_obs - f_phot
            s["t_excess"] = wddust_temperature(df["w2"] / df["w1"]) if df["w1"] > 0 and df["w2"] > 0 else None
            s["blend"] = (s["w"]["w1"]["frac"] or 1) < 0.7 or (s["w"]["w2"]["frac"] or 1) < 0.7 or (s["ruwe"] or 0) > 1.4 or s["n_unwise"] > 2
            s["warm"] = s["t_excess"] is not None and wddust_TMIN <= s["t_excess"] <= wddust_TMAX
            if s["warm"] and not s["blend"]:
                cands.append(s)
    n_exc = sum(1 for s in ok if s.get("excess") and all(s["excess"][b]["sigma"] >= wddust_SIGMA for b in s["excess"]))
    print("%d DESI white dwarfs with unWISE W1 and W2 and Gaia parallaxes at 5 sigma; locus scatter in G - W1 %.2f mag; %d exceed the photosphere at %.0f sigma in both bands, %d of them with excess black bodies at %.0f-%.0f K and unblended" % (
        len(ok), np.median(locus["w1"][2]), n_exc, wddust_SIGMA, len(cands), wddust_TMIN, wddust_TMAX))
    for s in sorted(cands, key=lambda s: -s["excess"]["w2"]["sigma"])[:30]:
        print("  targetid %s Gaia %s (%.5f %+.5f): G %.2f BP-RP %.2f plx %.1f mas; excess W1 %.2f mag (%.0f sigma) W2 %.2f (%.0f sigma), T %.0f K; fracflux %.2f/%.2f RUWE %.2f" % (
            s["targetid"], s["source_id"], s["ra"], s["dec"], s["G"], s["bprp"], s["plx"], s["excess"]["w1"]["excess_mag"], s["excess"]["w1"]["sigma"], s["excess"]["w2"]["excess_mag"],
            s["excess"]["w2"]["sigma"], s["t_excess"], s["w"]["w1"]["frac"] or 0, s["w"]["w2"]["frac"] or 0, s["ruwe"] or 0))
    # controls: three nearest excess-free white dwarfs in G and BP-RP
    plain = [s for s in ok if s.get("excess") and all(abs(s["excess"][b]["sigma"]) < 2 for b in s["excess"])]
    ctrl = {}
    for s in cands:
        d = sorted(plain, key=lambda p: math.hypot((p["G"] - s["G"]) / 0.5, (p["bprp"] - s["bprp"]) / 0.1))[:3]
        for p in d:
            ctrl[p["targetid"]] = p
    json.dump(dict(n=len(ok), n_excess=n_exc, n_cand=len(cands), locus={b: [list(map(float, v)) for v in locus[b]] for b in locus},
                   cands=[{k: v for k, v in s.items()} for s in cands], controls=[{k: v for k, v in p.items()} for p in ctrl.values()],
                   params=dict(sigma=wddust_SIGMA, tmin=wddust_TMIN, tmax=wddust_TMAX, ew_ma=wddust_EW)), open(core.results(wddust_OUT), "w"), indent=0, default=str)


def wddust_ew(wave, flux, ivar, centre, model=None, half=4.0, cont=((-35.0, -12.0), (12.0, 35.0))):
    """Equivalent width (mA, positive in absorption) of a line against the template or side continuum, with its error."""
    m = np.zeros(len(wave), bool)
    for a, b in cont:
        m |= (wave >= centre + a) & (wave <= centre + b)
    m &= ivar > 0
    if model is not None:
        m &= model > 0
    if m.sum() < 10:
        return None, None
    if model is not None:
        p = np.polyfit(wave[m], flux[m] / model[m], 1, w=np.sqrt(ivar[m]) * model[m])
        c = np.polyval(p, wave) * model
    else:
        p = np.polyfit(wave[m], flux[m], 1, w=np.sqrt(ivar[m]))
        c = np.polyval(p, wave)
    w = (wave >= centre - half) & (wave <= centre + half) & (ivar > 0) & (c > 0)
    if w.sum() < 4:
        return None, None
    dl = float(np.median(np.diff(wave[w])))
    ew = float(np.sum((1 - flux[w] / c[w]) * dl)) * 1000.0
    err = float(math.sqrt(np.sum(dl ** 2 / (ivar[w] * c[w] ** 2)))) * 1000.0
    return ew, err


def wddust_spectra():
    d = json.load(open(core.results(wddust_OUT)))
    targets = {s["targetid"]: ("candidate", s) for s in d["cands"]}
    targets.update({s["targetid"]: ("control", s) for s in d["controls"]})
    ids = list(targets)
    out = {}
    for k in range(0, len(ids), 100):
        for rec in core.desi_spectra([int(t) for t in ids[k:k + 100]]):
            t = str(rec["targetid"])
            wave = np.linspace(3600.0, 9824.0, 7781)
            flux, ivar = np.asarray(rec["flux"], float), np.asarray(rec["ivar"], float)
            model = np.asarray(rec["model"], float) if rec.get("model") is not None else None
            ca, e_ca = wddust_ew(wave, flux, ivar, wddust_CAK, model)
            mg, e_mg = wddust_ew(wave, flux, ivar, wddust_MGII, model)
            out[t] = dict(role=targets[t][0], ca_k=ca, e_ca_k=e_ca, mg_ii=mg, e_mg_ii=e_mg)
    for role in ("candidate", "control"):
        rows = [v for v in out.values() if v["role"] == role and v["ca_k"] is not None]
        polluted = [v for v in rows if v["ca_k"] - 3 * v["e_ca_k"] > wddust_EW]
        clean = [v for v in rows if v["ca_k"] + 3 * v["e_ca_k"] < wddust_EW]
        print("%s white dwarfs with spectra: %d; Ca II K above %.0f mA at 3 sigma in %d, below it at 3 sigma in %d, undecided %d" % (
            role, len(rows), wddust_EW, len(polluted), len(clean), len(rows) - len(polluted) - len(clean)))
    for t, v in out.items():
        if v["role"] == "candidate":
            print("  %s: Ca II K %s mA, Mg II %s mA" % (t, "%.0f +- %.0f" % (v["ca_k"], v["e_ca_k"]) if v["ca_k"] is not None else "-", "%.0f +- %.0f" % (v["mg_ii"], v["e_mg_ii"]) if v["mg_ii"] is not None else "-"))
    d["spectra"] = out
    json.dump(d, open(core.results(wddust_OUT), "w"), indent=0, default=str)


def wddust_record():
    d = json.load(open(core.results(wddust_OUT)))
    sp = d.get("spectra", {})
    def tally(role):
        rows = [v for v in sp.values() if v["role"] == role and v["ca_k"] is not None]
        return len(rows), sum(1 for v in rows if v["ca_k"] - 3 * v["e_ca_k"] > d["params"]["ew_ma"]), sum(1 for v in rows if v["ca_k"] + 3 * v["e_ca_k"] < d["params"]["ew_ma"])
    nc, pc, cc = tally("candidate")
    nk, pk, ck = tally("control")
    core.record_search(
        "desi_wd_dust_pollution", "Warm dust without pollution at white dwarfs", archive="DESI DR1 zpix white dwarfs with unWISE and Gaia DR3 through the Astro Data Lab; DESI spectra through SPARCL", release="DESI DR1",
        selection="white dwarfs with unWISE W1 and W2 detections and Gaia parallaxes at 5 sigma (%d)" % d["n"], code=os.path.abspath(__file__), entry="wddust", n_targets=d["n"], params=d["params"],
        null_summary="the photosphere locus of G - W1 and G - W2 against BP - RP from the white dwarfs themselves; %d excess-free white dwarfs matched in G and BP - RP as the pollution control" % len(d["controls"]),
        sensitivity="an excess of %.0f sigma in both bands at 600-2000 K; Ca II K to about %.0f mA at 3 sigma" % (d["params"]["sigma"], d["params"]["ew_ma"]), ledger="Warm dust without pollution at white dwarfs",
        result="%d excess white dwarfs at %.0f sigma in W1 and W2, %d warm (600-2000 K) and unblended; of %d with spectra %d show Ca II K above %.0f mA and %d lie below it, against %d of %d controls above and %d below" % (
            d["n_excess"], d["params"]["sigma"], d["n_cand"], nc, pc, d["params"]["ew_ma"], cc, pk, nk, ck))
    core.clear_targets("desi_wd_dust_pollution")
    core.record_targets("desi_wd_dust_pollution", [dict(target="gaia_dr3:%s" % s["source_id"], stat=s["excess"]["w2"]["sigma"], unit="W2 excess over the photosphere locus, sigma",
                                                            vet=("Ca II K %.0f +- %.0f mA" % (sp[s["targetid"]]["ca_k"], sp[s["targetid"]]["e_ca_k"])) if s["targetid"] in sp and sp[s["targetid"]]["ca_k"] is not None else None,
                                                            note="DESI %s, G %.2f, BP-RP %.2f, excess W1 %.2f W2 %.2f mag, T %.0f K" % (s["targetid"], s["G"], s["bprp"], s["excess"]["w1"]["excess_mag"], s["excess"]["w2"]["excess_mag"], s["t_excess"])) for s in d["cands"]])
    print("recorded %d candidates" % d["n_cand"])


def wddust_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": wddust_fetch, "analyse": wddust_analyse, "spectra": wddust_spectra, "record": wddust_record}[mode]()


# --------------------------------------------------------------------------------------------------------------
# infrared deficits, one-filter sources, 12 um changes and swollen dwarfs

irdeficit_DOC = r"""Stars dimmer in the infrared than their spectra allow: Gaia DR3 stars within 100 pc with RVS spectroscopic temperatures (teff_gspspec) whose first ten GSP-Spec flags are 0 or 1, A_G below 0.05 and an unsaturated AllWISE W1 (8-15 mag, photometric quality A) through Gaia's AllWISE best neighbours, G - W1 against the running median at their spectroscopic temperature, the colour temperature from that locus, and stars whose colour temperature exceeds the spectroscopic one by 500 K at 5 sigma, an infrared deficit that a shield absorbing the star's infrared would make; each read for an optical temperature, GSP-Phot's or that of BP-RP on its own locus, nearer the infrared colour's than the spectrum's.

    python -I seti.py infrared irdeficit [record] [--dteff-k 500] [--sigma 5] [--w1-min 8] [--w1-max 15] [--ag-max 0.05] [--ruwe-max 1.4] [--gspspec-flag-max 1]"""


def irdeficit_main(record=False, dteff_k=500.0, sigma=5.0, w1_min=8.0, w1_max=15.0, ag_max=0.05, ruwe_max=1.4, gspspec_flag_max=1):
    rows = core.tap_async("gaia", "SELECT g.source_id, g.phot_g_mean_mag, g.bp_rp, g.parallax, g.ruwe, a.teff_gspspec, a.flags_gspspec, a.teff_gspphot, "
                                  "a.ag_gspphot, w.w1mpro, w.w1mpro_error, w.ph_qual "
                                  "FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a ON g.source_id = a.source_id "
                                  "JOIN gaiadr3.allwise_best_neighbour AS x ON g.source_id = x.source_id JOIN gaiadr1.allwise_original_valid AS w "
                                  "ON x.allwise_oid = w.allwise_oid WHERE g.parallax > 10 AND g.parallax_over_error > 10 AND a.teff_gspspec IS NOT NULL "
                                  "AND w.w1mpro BETWEEN %g AND %g" % (w1_min, w1_max), maxrec=500000)
    # GSP-Spec's first ten flags grade broadening, velocity, flux noise and extrapolation; above 1 the temperature pins to its grid
    rows = [r for r in rows if str(r["ph_qual"])[:1] == "A" and (num(r["ag_gspphot"]) or 0) < ag_max and (num(r["ruwe"]) or 9) < ruwe_max
            and len(str(r["flags_gspspec"] or "")) >= 10 and max(str(r["flags_gspspec"])[:10]) <= str(gspspec_flag_max)]
    T = np.array([float(r["teff_gspspec"]) for r in rows])
    C = np.array([float(r["phot_g_mean_mag"]) - float(r["w1mpro"]) for r in rows])
    B = np.array([num(r["bp_rp"]) if num(r["bp_rp"]) is not None else np.nan for r in rows])

    def locus(X):
        cen, med = [], []
        for lo in np.arange(3000, 7100, 100):
            s = (T >= lo) & (T < lo + 100) & np.isfinite(X)
            if s.sum() >= 20:
                cen.append(lo + 50)
                med.append(float(np.median(X[s])))
        cen, med = np.array(cen), np.array(med)
        # colours fall with temperature; the colour temperature reads the locus backwards
        order = np.argsort(med)
        return cen, med, np.interp(X, med[order], cen[order], left=np.nan, right=np.nan)
    cen, med, tcol = locus(C)
    tbp = locus(B)[2]
    resid = C - np.interp(T, cen, med)
    s = core.robust_sigma(resid[np.isfinite(resid)])
    hot = sorted([(i, tcol[i] - T[i]) for i in range(len(rows)) if np.isfinite(tcol[i]) and tcol[i] - T[i] > dteff_k and resid[i] < -sigma * s],
                 key=lambda x: -x[1])
    # an optical temperature, GSP-Phot's or the BP-RP colour's, siding with G - W1 leaves the RVS temperature the odd one out
    tp = {i: num(rows[i]["teff_gspphot"]) for i, d in hot}
    odd = {i for i, d in hot if any(t and np.isfinite(t) and abs(t - tcol[i]) < abs(t - T[i]) for t in (tp[i], tbp[i]))}
    print("%d stars with clean RVS temperatures and clean W1 within 100 pc; G-W1 scatter about the locus %.3f mag; colour temperature %g K above "
          "the spectroscopic at %g sigma: %d, %d of them with an optical temperature nearer the colour's" % (len(rows), s, dteff_k, sigma, len(hot), len(odd)))
    for i, d in hot:
        print("  Gaia DR3 %s: RVS %.0f K, G - W1 %.0f K, BP-RP %.0f K, GSP-Phot %s K, G - W1 %.2f mag below the locus" % (
            rows[i]["source_id"], T[i], tcol[i], tbp[i], "%.0f" % tp[i] if tp[i] else "-", -float(resid[i])))
    if record:
        core.record_run(
            "gaia_infrared_deficit", "Stars dimmer in the infrared than their spectra allow", "Gaia DR3 astrophysical_parameters and AllWISE best neighbours",
            "stars within 100 pc with clean RVS temperatures and clean W1: %d" % len(rows), os.path.abspath(__file__), "irdeficit", len(rows), "stars",
            dict(dteff_k=dteff_k, sigma=sigma, gspspec_flags=gspspec_flag_max, w1=[w1_min, w1_max], ag_max=ag_max, ruwe_max=ruwe_max),
            "the G - W1 locus at the spectroscopic temperature, scatter %.3f mag" % s,
            "an infrared deficit of %g K in colour temperature" % dteff_k, "%d stars, %d with an optical temperature nearer the colour's" % (len(hot), len(odd)),
            [dict(target="gaia_dr3:%s" % rows[i]["source_id"], stat=float(d), unit="colour minus spectroscopic temperature, K",
                  vet="published-solution" if i in odd else None, note="G - W1 %.2f mag below the locus; BP-RP temperature %.0f K, GSP-Phot %s K" % (
                      -float(resid[i]), tbp[i], "%.0f" % tp[i] if tp[i] else "-")) for i, d in hot])
        print("recorded")


onefilter_DOC = r"""Sources shining in one filter only: Pan-STARRS1 DR1 mean objects (VizieR II/349) detected five or more times in one filter and never in the other four, brighter than the depth of the neighbouring filters by more than any stellar colour bridges (g or r below 19, i below 18.5, z below 18, y below 17), a monochromatic source in a broadband survey.

    python -I seti.py infrared onefilter [record] [--limits 19 19 18.5 18 17] [--n-min 5]

The limits are the g, r, i, z and y magnitudes in that order; each filter's answer is kept in the results directory, under a name carrying the limit and count when they differ from the defaults."""

ONEFILTER = {"g": 19.0, "r": 19.0, "i": 18.5, "z": 18.0, "y": 17.0}


def onefilter_main(record=False, limits=(19.0, 19.0, 18.5, 18.0, 17.0), n_min=5):
    lims = dict(zip("grizy", limits))
    hits, failed = [], []
    for f, lim in lims.items():
        # each filter's selection reads the whole table, so its answer is kept and a later run starts at the next filter
        p = core.results("onefilter_%s.json" % f if (lim, n_min) == (ONEFILTER[f], 5) else "onefilter_%s_%g_%d.json" % (f, lim, n_min))
        if os.path.exists(p):
            hits += [dict(r, band=f) for r in json.load(open(p))]
            continue
        others = [o for o in lims if o != f]
        q = ('SELECT TOP 5000 objID, RAJ2000, DEJ2000, %smag, N%s, Qual FROM "II/349/ps1" WHERE N%s >= %d AND %s AND %smag < %.1f' % (
            f, f, f, n_min, " AND ".join("N%s = 0" % o for o in others), f, lim))
        try:
            got = core.tap_async("vizier", q, maxrec=5000, limit=21600)
        except Exception as ex:
            failed.append((f, repr(ex)[:120]))
            continue
        json.dump(got, open(p, "w"))
        hits += [dict(r, band=f) for r in got]
    print("one-filter sources: %d %s; queries failed: %s" % (len(hits), [(h["band"], h["objID"], h["%smag" % h["band"]]) for h in hits[:20]], failed))
    json.dump(dict(hits=hits, failed=failed), open(core.results("onefilter.json"), "w"), indent=1, default=str)
    if record and not failed:
        core.record_run(
            "ps1_one_filter", "Sources shining in one filter only", "Pan-STARRS1 DR1 (VizieR II/349)", "mean objects detected five or more times in "
            "one filter and in no other, brighter than %s" % lims, os.path.abspath(__file__), "onefilter", len(hits), "sources",
            dict(limits=lims, n_min=n_min), "no stellar colour reaches the limits", "a source seen in one Pan-STARRS filter alone",
            "%d sources" % len(hits), [dict(target="ps1:%s" % h["objID"], stat=num(h["%smag" % h["band"]]), unit="%s mag" % h["band"]) for h in hits])
        print("recorded")


iraswise_DOC = r"""Dwarfs whose 12 um output changed between 1983 and 2010: Gaia DR3 stars within 100 pc at G below 8 with RUWE below 1.4, the IRAS Point Source Catalog (VizieR II/125) at 12 um of high quality within 45 arcsec of the position in 1983 and AllWISE W3 (A quality, no contamination flag, unsaturated beyond 3.8 mag) within 3 arcsec of the position in 2010, the IRAS flux as a magnitude (zero point 28.3 Jy) against W3, flagged where they part by 1 mag from the sample's median offset, and read for a Gaia star within 45 arcsec and 2 mag in G, whose light IRAS's beam takes in and W3's does not.

    python -I seti.py infrared iraswise [record] [--g-max 8] [--ruwe-max 1.4] [--iras-arcsec 45] [--wise-arcsec 3] [--dmag 1] [--blend-dmag 2]"""


def iraswise_main(record=False, g_max=8.0, ruwe_max=1.4, iras_arcsec=45.0, wise_arcsec=3.0, dmag=1.0, blend_dmag=2.0):
    from setitools import neighbourhood
    R, c = neighbourhood.nearby_astrometry()
    sel = [i for i in range(len(R)) if c["phot_g_mean_mag"][i] < g_max and (c["ruwe"][i] or 9) < ruwe_max]
    up83, up10 = [], []
    for i in sel:
        r = R[i]
        for yr, up in ((1983.5, up83), (2010.5, up10)):
            ra, de = core.star_at(float(r["ra"]), float(r["dec"]), num(r["pmra"], 0.0), num(r["pmdec"], 0.0), 2016.0, yr)
            up.append(dict(source_id=r["source_id"], ra=ra, dec=de))
    iras = {}
    for m in core.xmatch(up83, "II/125/main", iras_arcsec):
        if str(m.get("q_Fnu_12")) == "3" and num(m.get("Fnu_12")):
            if m["source_id"] not in iras or float(m["angDist"]) < iras[m["source_id"]][1]:
                iras[m["source_id"]] = (num(m["Fnu_12"]), float(m["angDist"]))
    wise = {}
    for m in core.xmatch(up10, "II/328/allwise", wise_arcsec):
        q, cc, w3 = str(m.get("qph") or "UUUU"), str(m.get("ccf") or "0000"), num(m.get("W3mag"))
        if w3 and w3 > 3.8 and q[2:3] == "A" and cc[2:3] == "0":
            wise[m["source_id"]] = w3
    both = [(s, -2.5 * math.log10(iras[s][0] / 28.3), wise[s]) for s in iras if s in wise]
    d = np.array([w - m for s, m, w in both])
    off = float(np.median(d)) if len(d) else 0.0
    hits = sorted([(s, m, w, w - m - off) for s, m, w in both if abs(w - m - off) > dmag], key=lambda x: -abs(x[3]))
    # IRAS's 12 um beam spans arcminutes and W3's 6.5 arcsec: a star within 45 arcsec and 2 mag in G adds its light to IRAS alone
    at = {R[i]["source_id"]: i for i in sel}
    blend = {}
    for s, m, w, x in hits:
        i = at[s]
        blend[s] = [r["source_id"] for r in core.tap("gaia", "SELECT source_id FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
                                                             "CIRCLE('ICRS', %s, %s, %.5f)) AND source_id <> %s AND phot_g_mean_mag < %.2f" % (
                                                                 R[i]["ra"], R[i]["dec"], iras_arcsec / 3600.0, s, c["phot_g_mean_mag"][i] + blend_dmag))] if x > 0 else []
    print("%d stars within 100 pc at G below %g; IRAS 12 um of high quality %d, clean W3 %d, both %d; W3 - IRAS median %.2f, robust scatter %.2f; "
          "parted by %g mag: %d %s" % (len(sel), g_max, len(iras), len(wise), len(both), off, core.robust_sigma(d) if len(d) else 0, dmag, len(hits),
                                       [(s, round(x, 2), blend[s]) for s, m, w, x in hits[:10]]))
    if record:
        core.record_run(
            "iras_wise_12um", "Dwarfs whose 12 um output changed between 1983 and 2010", "IRAS PSC (VizieR II/125) and AllWISE (II/328) by CDS XMatch; Gaia DR3",
            "stars within 100 pc at G below %g with RUWE below %g: %d, %d measured in both" % (g_max, ruwe_max, len(sel), len(both)), os.path.abspath(__file__),
            "iraswise", len(both), "stars", dict(dmag=dmag, iras_q=3, iras_arcsec=iras_arcsec, wise_arcsec=wise_arcsec, blend_dmag=blend_dmag),
            "the sample's median W3 - IRAS offset %.2f mag, scatter %.2f" % (off, core.robust_sigma(d) if len(d) else 0),
            "a 12 um change of %g mag between IRAS and WISE" % dmag, "%d parted by %g mag" % (len(hits), dmag),
            [dict(target="gaia_dr3:%s" % s, stat=x, unit="W3 - IRAS 12 um less the median, mag", vet="blend" if blend[s] else None,
                  note="IRAS beam shared with Gaia DR3 %s" % ", ".join(blend[s]) if blend[s] else None) for s, m, w, x in hits])
        print("recorded")


bigdwarfs_DOC = r"""Dwarfs larger than their light allows: the JMMC Measured Stellar Diameters Catalogue (VizieR II/345) limb-darkened diameters, matched to Gaia DR3 within 5 arcsec by CDS XMatch, against the angular diameter Gaia's FLAME radius gives at its parallax, 9.30e-3 R(R_sun) parallax(mas) mas, for dwarfs (GSP-Phot log g above 4) within 100 pc, flagged where the measured diameter exceeds 1.2 times the predicted at 3 sigma, an envelope around the star.

    python -I seti.py infrared bigdwarfs [record] [--ratio 1.2] [--sigma 3] [--logg-min 4] [--plx-min 10] [--radius-arcsec 5]"""


def bigdwarfs_main(record=False, ratio=1.2, sigma=3.0, logg_min=4.0, plx_min=10.0, radius_arcsec=5.0):
    jm = core.tap("vizier", 'SELECT SimbadName, "_RA", "_DE", LDdiam, e_LDdiam, Band, Method FROM "II/345/jmdc" WHERE LDdiam > 0 AND e_LDdiam > 0')
    up = [dict(source_id=str(k), ra=float(r["_RA"]), dec=float(r["_DE"])) for k, r in enumerate(jm)]
    gm = {}
    for m in core.xmatch(up, "I/355/gaiadr3", radius_arcsec):
        if num(m.get("Plx")) and (m["source_id"] not in gm or (num(m.get("Gmag")) or 99) < (num(gm[m["source_id"]].get("Gmag")) or 99)):
            gm[m["source_id"]] = m
    ids = sorted({str(m["Source"]) for m in gm.values()})
    ap = core.tap_by_ids("gaia", ids, "SELECT source_id, radius_flame, radius_flame_lower, radius_flame_upper, logg_gspphot FROM "
                                      "gaiadr3.astrophysical_parameters WHERE source_id IN (%s)")
    out = []
    for k, m in gm.items():
        a = ap.get(str(m["Source"]))
        r = jm[int(k)]
        if not a or num(a["radius_flame"]) is None or (num(a["logg_gspphot"]) or 0) <= logg_min or num(m["Plx"]) <= plx_min:
            continue
        R, lo, hi = num(a["radius_flame"]), num(a["radius_flame_lower"]) or 0, num(a["radius_flame_upper"]) or 0
        pred = 9.30e-3 * R * num(m["Plx"])
        epred = 9.30e-3 * (hi - lo) / 2 * num(m["Plx"])
        th, e = num(r["LDdiam"]), num(r["e_LDdiam"])
        z = (th - ratio * pred) / math.hypot(e, ratio * epred) if math.hypot(e, epred) > 0 else 0
        out.append(dict(name=r["SimbadName"], gaia=m["Source"], th=th, e=e, pred=pred, z=z, ratio=th / pred, band=r["Band"]))
    big = sorted([o for o in out if o["z"] > sigma], key=lambda o: -o["z"])
    rat = np.array([o["ratio"] for o in out])
    print("%d measured diameters of dwarfs within %.0f pc; measured over predicted median %.3f, robust scatter %.3f; above %g at %g sigma: %d %s" % (
        len(out), 1000 / plx_min, float(np.median(rat)) if len(rat) else 0, core.robust_sigma(rat) if len(rat) else 0, ratio, sigma, len(big),
        [(o["name"], round(o["ratio"], 2), round(o["z"], 1)) for o in big[:10]]))
    if record:
        core.record_run(
            "jmdc_swollen_dwarfs", "Dwarfs larger than their light allows", "JMMC Measured Stellar Diameters Catalogue (VizieR II/345); Gaia DR3 FLAME",
            "measured diameters of dwarfs within %.0f pc: %d" % (1000 / plx_min, len(out)), os.path.abspath(__file__), "bigdwarfs", len(out), "measurements",
            dict(ratio=ratio, sigma=sigma, logg_min=logg_min, plx_min=plx_min, radius_arcsec=radius_arcsec),
            "the sample's measured-over-predicted ratio, median %.3f" % (float(np.median(rat)) if len(rat) else 0),
            "a dwarf measured %.0f per cent larger than its light and temperature allow" % (100 * (ratio - 1)), "%d at %g sigma" % (len(big), sigma),
            [dict(target="gaia_dr3:%s" % o["gaia"], stat=o["ratio"], unit="measured over predicted diameter", note="%s, %.3f +- %.3f mas against %.3f, %s" % (
                o["name"], o["th"], o["e"], o["pred"], o["band"])) for o in big])
        print("recorded")


etzwarm_DOC = r"""Warm stars in Earth's transit zone: Gaia Catalogue of Nearby Stars objects (probability above 0.9, not white dwarfs, RUWE below 1.4, one IPD peak) with WISE W1 and a W3 or W4 detection at S/N 10 or 5, whose W1-W3 or W1-W4 colour lies more than five robust widths and three errors above the locus of single stars of the same BP-RP, and still does with AllWISE's plain profile W1 in place of the catalogue's proper-motion-fitted W1, which runs faint for bright and fast stars; the share of such stars within 0.264 deg of the ecliptic, where Earth's transits across the Sun are seen, against the share 1-5 deg away. A watcher of Earth's transits with a civilisation's waste heat would raise the share in the zone.

With `--planet NAME` the zone is that planet's transit zone, the Sun's angular radius at the planet's distance either side of its orbital plane, where a watcher sees the giant planet rather than Earth, and the control band is measured from that plane.

    python -I seti.py infrared etzwarm [record] [--planet jupiter] [--zone-deg 0.264] [--control-min-deg 1] [--control-max-deg 5] [--widths 5] [--errors 3] [--w3-err-max 0.1] [--w4-err-max 0.2]"""


def etzwarm_main(record=False, planet="", zone_deg=0.264, control_min_deg=1.0, control_max_deg=5.0, widths=5.0, errors=3.0, w3_err_max=0.1, w4_err_max=0.2):
    if planet:
        from setitools.neighbourhood import PLANET_ORBITS
        planet = planet.lower()
        a, inc, node = PLANET_ORBITS[planet]
        zone_deg = math.degrees(math.asin(0.0046505 / a))
        eps, i, o = math.radians(23.4392911), math.radians(inc), math.radians(node)
        normal = (math.sin(i) * math.sin(o), -math.sin(i) * math.cos(o), math.cos(i))

    def latitude(ra_deg, dec_deg):
        if not planet:
            return abs(vasco_ecl_lat(ra_deg, dec_deg))
        ra, de = math.radians(ra_deg), math.radians(dec_deg)
        x, y, z = math.cos(de) * math.cos(ra), math.cos(de) * math.sin(ra), math.sin(de)
        ye, ze = y * math.cos(eps) + z * math.sin(eps), -y * math.sin(eps) + z * math.cos(eps)
        return abs(math.degrees(math.asin(max(-1.0, min(1.0, x * normal[0] + ye * normal[1] + ze * normal[2])))))
    rows = catalogues.gcns("source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe, ipd_frac_multi_peak, w1mpro_pm_wise, "
                           "w1sigmpro_pm_wise, w3mpro_wise, w3sigmpro_wise, w4mpro_wise, w4sigmpro_wise",
                           "gcns_prob > 0.9 AND wd_prob < 0.1 AND ruwe < 1.4 AND ipd_frac_multi_peak <= 2 AND w1mpro_pm_wise IS NOT NULL", ranges=12)
    st = []
    for r in rows:
        bprp = num(r["phot_bp_mean_mag"]) - num(r["phot_rp_mean_mag"]) if num(r["phot_bp_mean_mag"]) and num(r["phot_rp_mean_mag"]) else None
        if bprp is None:
            continue
        st.append(dict(id=r["source_id"], beta=latitude(float(r["ra"]), float(r["dec"])), bprp=bprp, w1=num(r["w1mpro_pm_wise"]),
                       s1=num(r["w1sigmpro_pm_wise"]), w3=num(r["w3mpro_wise"]), s3=num(r["w3sigmpro_wise"]), w4=num(r["w4mpro_wise"]),
                       s4=num(r["w4sigmpro_wise"]), g=num(r["phot_g_mean_mag"]), plx=num(r["parallax"]), ra=float(r["ra"]), dec=float(r["dec"]),
                       pmra=num(r["pmra"]) or 0.0, pmdec=num(r["pmdec"]) or 0.0))
    warm, loci = {}, {}
    for band, lim in (("3", w3_err_max), ("4", w4_err_max)):
        use = [s for s in st if s["w" + band] is not None and s["s" + band] is not None and s["s" + band] < lim and s["s1"] is not None]
        col = np.array([s["w1"] - s["w" + band] for s in use])
        x = np.array([s["bprp"] for s in use])
        bins = np.floor(x / 0.1).astype(int)
        for b in set(bins.tolist()):
            k = np.nonzero(bins == b)[0]
            if len(k) < 20:
                continue
            med = float(np.median(col[k]))
            wid = 1.4826 * float(np.median(np.abs(col[k] - med)))
            for i in k:
                s = use[i]
                err = math.hypot(s["s1"], s["s" + band])
                if col[i] - med > max(widths * wid, errors * err):
                    warm.setdefault(s["id"], {})[band] = round(float(col[i] - med), 2)
                    loci[(s["id"], band)] = (med, wid, err)
    # the catalogue's W1 is AllWISE's proper-motion fit, which runs up to a magnitude faint of the plain profile W1 for bright and fast stars:
    # each warm star's colour taken again with the plain W1 at its 2010.5 position, against the same locus
    by_id = {s["id"]: s for s in st}
    pts = [dict(source_id=str(i), ra=core.star_at(by_id[i]["ra"], by_id[i]["dec"], by_id[i]["pmra"], by_id[i]["pmdec"], 2016.0, 2010.5)[0] % 360,
                dec=core.star_at(by_id[i]["ra"], by_id[i]["dec"], by_id[i]["pmra"], by_id[i]["pmdec"], 2016.0, 2010.5)[1]) for i in warm]
    plain = {}
    for k in range(0, len(pts), 2000):
        for m in core.xmatch(pts[k:k + 2000], "II/328/allwise", 3.0):
            w1 = num(m.get("W1mag"))
            if w1 is not None and (m["source_id"] not in plain or float(m["angDist"]) < plain[m["source_id"]][0]):
                plain[m["source_id"]] = (float(m["angDist"]), w1)
    fitted = 0
    for i in list(warm):
        p = plain.get(str(i))
        if p is None:
            continue
        s = by_id[i]
        still = {band: round(float(p[1] - s["w" + band] - loci[(i, band)][0]), 2) for band in warm[i]
                 if p[1] - s["w" + band] - loci[(i, band)][0] > max(widths * loci[(i, band)][1], errors * loci[(i, band)][2])}
        if not still:
            del warm[i]
            fitted += 1
        else:
            warm[i] = still
    print("%d warm by the catalogue's proper-motion-fitted W1 lose their excess with AllWISE's plain W1" % fitted)
    zone = [s for s in st if s["beta"] < zone_deg and (s["w3"] is not None or s["w4"] is not None)]
    near = [s for s in st if control_min_deg <= s["beta"] < control_max_deg and (s["w3"] is not None or s["w4"] is not None)]
    wz = [s for s in zone if s["id"] in warm]
    wn = sum(1 for s in near if s["id"] in warm)
    frac = wn / max(len(near), 1)
    exp = frac * len(zone)
    sim = {}
    for k in range(0, len(wz), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in wz[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    from scipy.stats import poisson
    p = float(poisson.sf(len(wz) - 1, exp))
    who = (planet or "earth").capitalize()
    print("%s's transit zone, %.3f deg either side of its orbital plane: %d stars with W1 and W3 or W4; in the zone %d, %d warm (%s); %g-%g deg away %d, %d warm (share %.4f, %.2f "
          "expected in the zone, p %.3g)" % (who, zone_deg, len(st), len(zone), len(wz), [(s["id"], sim.get(s["id"]), warm[s["id"]], round(s["g"], 1)) for s in wz],
                                               control_min_deg, control_max_deg, len(near), wn, frac, exp, p))
    if record:
        core.record_run("etz_warm_stars" if not planet else "%s_zone_warm_stars" % planet, "Warm stars in %s's transit zone" % who,
                        "Gaia Catalogue of Nearby Stars with its WISE photometry (external.gaiaedr3_gcns_main_1); "
                        "SIMBAD", "stars within 100 pc with W1 and W3 or W4, %g deg of %s's orbital plane: %d" % (zone_deg, who, len(zone)), os.path.abspath(__file__), "etzwarm",
                        len(zone), "stars", dict(planet=planet, zone_deg=zone_deg, control_deg=[control_min_deg, control_max_deg], widths=widths, errors=errors,
                                                 err_max=dict(w3=w3_err_max, w4=w4_err_max)),
                        "the share %g-%g deg from the plane: %d of %d" % (control_min_deg, control_max_deg, wn, len(near)),
                        "a raised share of warm stars among those that see %s transit" % who,
                        "%d warm in the zone against %.2f expected (p %.3g)" % (len(wz), exp, p),
                        [dict(target="gaia_dr3:%s" % s["id"], stat=max(warm[s["id"]].values()), unit="mag above the locus",
                              vet="natural-type" if sim.get(s["id"]) and sim[s["id"]][1] not in ("*", "PM*", "HV*") else ("chance-rate" if p > 0.01 else None),
                              note="SIMBAD %s; excess %s" % (sim.get(s["id"]), warm[s["id"]])) for s in wz])
        print("recorded")


w2only_DOC = r"""Light at 4.6 um alone: Gaia Catalogue of Nearby Stars objects (probability above 0.9, white-dwarf probability below 0.1, RUWE below 1.4, one IPD peak) at BP-RP 0.5-2.5 with WISE W1 8-14, W2 above 7.5 and W1, W2 and W3 measured to 0.05, 0.05 and 0.1 mag, whose W1-W2 lies more than five robust widths and five errors above the single-star locus at their BP-RP while W1-W3 stays within two widths of its own; dust warm enough to shine at 4.6 um shines at 12 um as well, and an emitter in one band does not. The stars as far below the W1-W2 locus are the mirror null. Each candidate read in AllWISE for its contamination and variability flags and in SIMBAD.

    python -I seti.py infrared w2only [record] [--widths 5] [--errors 5] [--w3-widths 2] [--colour-min 0.5] [--colour-max 2.5] [--w1-min 8] [--w1-max 14] [--w2-min 7.5]"""


def w2only_main(record=False, widths=5.0, errors=5.0, w3_widths=2.0, colour_min=0.5, colour_max=2.5, w1_min=8.0, w1_max=14.0, w2_min=7.5):
    rows = catalogues.gcns("source_id, parallax, phot_bp_mean_mag, phot_rp_mean_mag, w1mpro_pm_wise, w1sigmpro_pm_wise, w2mpro_pm_wise, w2sigmpro_pm_wise, "
                           "w3mpro_wise, w3sigmpro_wise, ra, dec",
                           "gcns_prob > 0.9 AND wd_prob < 0.1 AND ruwe < 1.4 AND ipd_frac_multi_peak <= 2 AND w1mpro_pm_wise BETWEEN %g AND %g AND w2mpro_pm_wise > %g "
                           "AND w1sigmpro_pm_wise < 0.05 AND w2sigmpro_pm_wise < 0.05 AND w3sigmpro_wise < 0.1" % (w1_min, w1_max, w2_min), ranges=12)
    st = []
    for r in rows:
        bp, rp = num(r["phot_bp_mean_mag"]), num(r["phot_rp_mean_mag"])
        if bp is None or rp is None or not colour_min <= bp - rp <= colour_max or num(r["w3mpro_wise"]) is None:
            continue
        w1, w2, w3 = float(r["w1mpro_pm_wise"]), float(r["w2mpro_pm_wise"]), float(r["w3mpro_wise"])
        st.append(dict(id=r["source_id"], c=bp - rp, c12=w1 - w2, c13=w1 - w3, e12=math.hypot(float(r["w1sigmpro_pm_wise"]), float(r["w2sigmpro_pm_wise"])),
                       ra=float(r["ra"]), dec=float(r["dec"])))
    bins = np.floor(np.array([s["c"] for s in st]) / 0.1).astype(int)
    for key in ("c12", "c13"):
        val = np.array([s[key] for s in st])
        for b in set(bins.tolist()):
            k = np.nonzero(bins == b)[0]
            med = float(np.median(val[k])) if len(k) >= 30 else None
            wid = 1.4826 * float(np.median(np.abs(val[k] - med))) if med is not None else None
            for i in k:
                st[i]["r" + key] = (val[i] - med, wid) if med is not None else None
    up = [s for s in st if s["rc12"] and s["rc13"] and s["rc12"][0] > max(widths * s["rc12"][1], errors * s["e12"]) and abs(s["rc13"][0]) < w3_widths * s["rc13"][1]]
    down = [s for s in st if s["rc12"] and s["rc13"] and -s["rc12"][0] > max(widths * s["rc12"][1], errors * s["e12"]) and abs(s["rc13"][0]) < w3_widths * s["rc13"][1]]
    flags = {}
    if up:
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in up], "II/328/allwise", 3.0):
            if m["source_id"] not in flags or float(m["angDist"]) < flags[m["source_id"]][0]:
                flags[m["source_id"]] = (float(m["angDist"]), (m.get("ccf") or "").strip(), (m.get("ex") or "").strip(), (m.get("var") or m.get("Var") or "").strip())
    sim = {}
    for k in range(0, len(up), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in up[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in up:
        f = flags.get(s["id"])
        s["simbad"] = sim.get(s["id"])
        s["flags"] = f
        s["vet"] = ("artefact" if f and (f[1].strip("0") or f[2] not in ("", "0")) else
                    "natural-type" if s["simbad"] and s["simbad"][1] not in ("*", "PM*", "HV*", "low-mass*") else None)
    left = [s for s in up if s["vet"] is None]
    print("%d stars; W1-W2 %g widths above the locus with W1-W3 on its own: %d; as far below: %d; unexplained %d" % (len(st), widths, len(up), len(down), len(left)))
    for s in sorted(up, key=lambda s: -s["rc12"][0] / s["rc12"][1]):
        print("  Gaia DR3 %s: BP-RP %.2f, W1-W2 +%.3f (width %.3f), W1-W3 %+.3f; AllWISE %s; SIMBAD %s; %s" % (
            s["id"], s["c"], s["rc12"][0], s["rc12"][1], s["rc13"][0], s["flags"], s["simbad"], s["vet"]))
    if record:
        core.record_run("w2_only_excess", "Light at 4.6 um alone", "Gaia Catalogue of Nearby Stars with its WISE photometry (external.gaiaedr3_gcns_main_1); "
                        "AllWISE (II/328) by CDS XMatch; SIMBAD", "stars within 100 pc at BP-RP %g-%g with clean W1, W2 and W3: %d" % (colour_min, colour_max, len(st)),
                        os.path.abspath(__file__), "w2only", len(st), "stars", dict(widths=widths, errors=errors, w3_widths=w3_widths, colour=[colour_min, colour_max],
                                                                                   w1=[w1_min, w1_max], w2_min=w2_min),
                        "the stars as far below the W1-W2 locus: %d" % len(down), "a star brighter at 4.6 um than its 3.4 and 12 um light allow",
                        "%d above against %d below; %d unexplained" % (len(up), len(down), len(left)),
                        [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["rc12"][0] / s["rc12"][1], 1), unit="widths above the W1-W2 locus",
                              vet=s["vet"] or ("chance-rate" if len(up) <= len(down) + 3 * math.sqrt(max(len(down), 1)) else None),
                              note="AllWISE %s; SIMBAD %s" % (s["flags"], s["simbad"])) for s in up])
        print("recorded")


w3only_DOC = r"""Light at 12 um alone: Gaia Catalogue of Nearby Stars objects (probability above 0.9, white-dwarf probability below 0.1, RUWE below 1.4, one IPD peak) at BP-RP 0.5-2.5 with WISE W1 8-14 and W1, W3 and W4 measured to 0.05, 0.1 and 0.2 mag, whose W1-W3 lies more than five robust widths and five errors above the single-star locus at their BP-RP while W1-W4 stays within two widths of its own; warm dust that shines at 12 um shines at 22 um as well, and an emitter confined to the 12 um band does not, though the silicate feature at 10 um can lift W3 alone. The stars as far below the W1-W3 locus are the mirror null. Each candidate read in AllWISE at its 2010.5 position for its contamination and variability flags and its plain profile W1, against which the catalogue's proper-motion-fitted W1 runs faint for bright and fast stars, and in SIMBAD; a W3 fainter than 11 mag is an AllWISE faint detection biased bright, a W1-W4 excess at three errors is dust after all, and a second Gaia source within 10 arcsec and five magnitudes shares the 12 um beam.

    python -I seti.py infrared w3only [record] [--widths 5] [--errors 5] [--w4-widths 2] [--colour-min 0.5] [--colour-max 2.5] [--w1-min 8] [--w1-max 14] [--w3-faint 11] [--beam-arcsec 10]"""


def w3only_main(record=False, widths=5.0, errors=5.0, w4_widths=2.0, colour_min=0.5, colour_max=2.5, w1_min=8.0, w1_max=14.0, w3_faint=11.0, beam_arcsec=10.0):
    rows = catalogues.gcns("source_id, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, w1mpro_pm_wise, w1sigmpro_pm_wise, w3mpro_wise, w3sigmpro_wise, "
                           "w4mpro_wise, w4sigmpro_wise, ra, dec, pmra, pmdec",
                           "gcns_prob > 0.9 AND wd_prob < 0.1 AND ruwe < 1.4 AND ipd_frac_multi_peak <= 2 AND w1mpro_pm_wise BETWEEN %g AND %g "
                           "AND w1sigmpro_pm_wise < 0.05 AND w3sigmpro_wise < 0.1 AND w4sigmpro_wise < 0.2" % (w1_min, w1_max), ranges=12)
    st = []
    for r in rows:
        bp, rp = num(r["phot_bp_mean_mag"]), num(r["phot_rp_mean_mag"])
        if bp is None or rp is None or not colour_min <= bp - rp <= colour_max or num(r["w3mpro_wise"]) is None or num(r["w4mpro_wise"]) is None:
            continue
        w1, w3, w4 = float(r["w1mpro_pm_wise"]), float(r["w3mpro_wise"]), float(r["w4mpro_wise"])
        st.append(dict(id=r["source_id"], c=bp - rp, g=num(r["phot_g_mean_mag"]), w1=w1, w3=w3, c13=w1 - w3, c14=w1 - w4,
                       e13=math.hypot(float(r["w1sigmpro_pm_wise"]), float(r["w3sigmpro_wise"])), e14=math.hypot(float(r["w1sigmpro_pm_wise"]), float(r["w4sigmpro_wise"])),
                       ra=float(r["ra"]), dec=float(r["dec"]), pmra=num(r["pmra"]) or 0.0, pmdec=num(r["pmdec"]) or 0.0))
    bins = np.floor(np.array([s["c"] for s in st]) / 0.1).astype(int)
    for key in ("c13", "c14"):
        val = np.array([s[key] for s in st])
        for b in set(bins.tolist()):
            k = np.nonzero(bins == b)[0]
            med = float(np.median(val[k])) if len(k) >= 30 else None
            wid = 1.4826 * float(np.median(np.abs(val[k] - med))) if med is not None else None
            for i in k:
                st[i]["r" + key] = (val[i] - med, wid) if med is not None else None
    up = [s for s in st if s["rc13"] and s["rc14"] and s["rc13"][0] > max(widths * s["rc13"][1], errors * s["e13"]) and abs(s["rc14"][0]) < w4_widths * s["rc14"][1]]
    down = [s for s in st if s["rc13"] and s["rc14"] and -s["rc13"][0] > max(widths * s["rc13"][1], errors * s["e13"]) and abs(s["rc14"][0]) < w4_widths * s["rc14"][1]]
    flags = {}
    if up:
        # AllWISE at each star's 2010.5 position, its flags and its plain profile W1
        pts = []
        for s in up:
            ra10, de10 = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, 2010.5)
            pts.append(dict(source_id=s["id"], ra=ra10 % 360, dec=de10))
        for m in core.xmatch(pts, "II/328/allwise", 3.0):
            if m["source_id"] not in flags or float(m["angDist"]) < flags[m["source_id"]][0]:
                flags[m["source_id"]] = (float(m["angDist"]), (m.get("ccf") or "").strip(), (m.get("ex") or "").strip(), (m.get("var") or m.get("Var") or "").strip(),
                                         num(m.get("W1mag")))
    sim = {}
    for k in range(0, len(up), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in up[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in up:
        f = flags.get(s["id"])
        s["simbad"] = sim.get(s["id"])
        s["flags"] = f[:4] if f else None
        # the excess taken again with AllWISE's plain profile W1, since the catalogue's proper-motion-fitted W1 runs faint for bright and fast stars
        s["w1_plain"] = f[4] if f else None
        plain_gone = s["w1_plain"] is not None and s["rc13"][0] + (s["w1_plain"] - s["w1"]) <= max(widths * s["rc13"][1], errors * s["e13"])
        # AllWISE W3 beyond 11 mag is a faint detection biased bright; a W4 excess at three errors is dust, which shines at 22 um as well
        s["vet"] = ("artefact" if (f and (f[1].strip("0") or f[2] not in ("", "0"))) or s["w3"] > w3_faint or plain_gone else
                    "dust" if s["rc14"][0] > 3 * s["e14"] else
                    "natural-type" if s["simbad"] and s["simbad"][1] not in ("*", "PM*", "HV*", "low-mass*") else None)
    # a second Gaia source inside the 6.5 arcsec W3 beam, within five magnitudes, shares the 12 um flux
    for s in up:
        s["near"] = []
        if s["vet"] is None and s["g"] is not None:
            s["near"] = ["%s (G %.1f)" % (r["source_id"], float(r["phot_g_mean_mag"])) for r in core.tap(
                "gaia", "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f)) "
                        "AND source_id <> %s AND phot_g_mean_mag < %.2f" % (s["ra"], s["dec"], beam_arcsec / 3600.0, s["id"], s["g"] + 5))]
            if s["near"]:
                s["vet"] = "blend"
    left = [s for s in up if s["vet"] is None]
    print("%d stars; W1-W3 %g widths above the locus with W1-W4 on its own: %d; as far below: %d; unexplained %d" % (len(st), widths, len(up), len(down), len(left)))
    for s in sorted(up, key=lambda s: -s["rc13"][0] / s["rc13"][1]):
        print("  Gaia DR3 %s: BP-RP %.2f, W3 %.2f, W1-W3 +%.3f (width %.3f), W1-W4 %+.3f (%.1f errors); AllWISE %s; SIMBAD %s; neighbours %s; %s" % (
            s["id"], s["c"], s["w3"], s["rc13"][0], s["rc13"][1], s["rc14"][0], s["rc14"][0] / s["e14"], s["flags"], s["simbad"], s["near"], s["vet"]))
    if record:
        core.record_run("w3_only_excess", "Light at 12 um alone", "Gaia Catalogue of Nearby Stars with its WISE photometry (external.gaiaedr3_gcns_main_1); "
                        "AllWISE (II/328) by CDS XMatch; SIMBAD", "stars within 100 pc at BP-RP %g-%g with clean W1, W3 and W4: %d" % (colour_min, colour_max, len(st)),
                        os.path.abspath(__file__), "w3only", len(st), "stars", dict(widths=widths, errors=errors, w4_widths=w4_widths, colour=[colour_min, colour_max],
                                                                                   w1=[w1_min, w1_max], w3_faint=w3_faint, beam_arcsec=beam_arcsec),
                        "the stars as far below the W1-W3 locus: %d" % len(down), "a star brighter at 12 um than its 3.4 and 22 um light allow",
                        "%d above against %d below; %d unexplained" % (len(up), len(down), len(left)),
                        [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["rc13"][0] / s["rc13"][1], 1), unit="widths above the W1-W3 locus",
                              vet=s["vet"] or ("chance-rate" if len(up) <= len(down) + 3 * math.sqrt(max(len(down), 1)) else None),
                              note="W3 %.2f, W1-W4 %+.2f at %.1f errors; AllWISE %s; SIMBAD %s; neighbours %s" % (
                                  s["w3"], s["rc14"][0], s["rc14"][0] / s["e14"], s["flags"], s["simbad"], s["near"] or "none")) for s in up])
        print("recorded")


ksexcess_DOC = r"""Light at 2 um alone: Gaia Catalogue of Nearby Stars objects (probability above 0.9, white-dwarf probability below 0.1, RUWE below 1.4, one IPD peak) at BP-RP 0.5-3.0 with 2MASS J, H and Ks and WISE W1 measured to 0.03 and 0.05 mag, whose J-Ks lies more than five robust widths and five errors above the single-star locus at their BP-RP while J-W1 stays within two widths of its own; a cool companion or hot dust that brightens Ks brightens W1 as well, and a source confined to 2 um does not. The stars as far below the J-Ks locus are the mirror null. Each candidate read in SIMBAD and in 2MASS for its quality, read and extended-source flags.

    python -I seti.py infrared ksexcess [record] [--widths 5] [--errors 5] [--w1-widths 2] [--colour-min 0.5] [--colour-max 3.0]"""


def ksexcess_main(record=False, widths=5.0, errors=5.0, w1_widths=2.0, colour_min=0.5, colour_max=3.0):
    rows = catalogues.gcns("source_id, phot_bp_mean_mag, phot_rp_mean_mag, j_m_2mass, j_msig_2mass, h_m_2mass, k_m_2mass, k_msig_2mass, w1mpro_pm_wise, w1sigmpro_pm_wise, ra, dec",
                           "gcns_prob > 0.9 AND wd_prob < 0.1 AND ruwe < 1.4 AND ipd_frac_multi_peak <= 2 AND j_msig_2mass < 0.03 AND k_msig_2mass < 0.03 "
                           "AND w1sigmpro_pm_wise < 0.05", ranges=12)
    st = []
    for r in rows:
        bp, rp, j, k, w1 = num(r["phot_bp_mean_mag"]), num(r["phot_rp_mean_mag"]), num(r["j_m_2mass"]), num(r["k_m_2mass"]), num(r["w1mpro_pm_wise"])
        if None in (bp, rp, j, k, w1) or not colour_min <= bp - rp <= colour_max:
            continue
        st.append(dict(id=r["source_id"], c=bp - rp, j=j, k=k, cjk=j - k, cjw=j - w1, ejk=math.hypot(float(r["j_msig_2mass"]), float(r["k_msig_2mass"])),
                       ra=float(r["ra"]), dec=float(r["dec"])))
    bins = np.floor(np.array([s["c"] for s in st]) / 0.1).astype(int)
    for key in ("cjk", "cjw"):
        val = np.array([s[key] for s in st])
        for b in set(bins.tolist()):
            kk = np.nonzero(bins == b)[0]
            med = float(np.median(val[kk])) if len(kk) >= 30 else None
            wid = 1.4826 * float(np.median(np.abs(val[kk] - med))) if med is not None else None
            for i in kk:
                st[i]["r" + key] = (val[i] - med, wid) if med is not None else None
    up = [s for s in st if s["rcjk"] and s["rcjw"] and s["rcjk"][0] > max(widths * s["rcjk"][1], errors * s["ejk"]) and abs(s["rcjw"][0]) < w1_widths * s["rcjw"][1]]
    down = [s for s in st if s["rcjk"] and s["rcjw"] and -s["rcjk"][0] > max(widths * s["rcjk"][1], errors * s["ejk"]) and abs(s["rcjw"][0]) < w1_widths * s["rcjw"][1]]
    flags = {}
    if up:
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in up], "II/246/out", 3.0):
            if m["source_id"] not in flags or float(m["angDist"]) < flags[m["source_id"]][0]:
                flags[m["source_id"]] = (float(m["angDist"]), (m.get("Qfl") or "").strip(), (m.get("Rfl") or "").strip(), (m.get("X") or "").strip())
    sim = {}
    for k in range(0, len(up), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in up[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in up:
        f = flags.get(s["id"])
        s["simbad"] = sim.get(s["id"])
        s["flags"] = f
        # 2MASS quality below AAA or an extended-source flag; a Ks read from a different scan than J (read flags) is not a colour
        s["vet"] = ("artefact" if f and (f[1] != "AAA" or f[3] not in ("", "0")) else
                    "natural-type" if s["simbad"] and s["simbad"][1] not in ("*", "PM*", "HV*", "low-mass*") else None)
    left = [s for s in up if s["vet"] is None]
    print("%d stars; J-Ks %g widths above the locus with J-W1 on its own: %d; as far below: %d; unexplained %d" % (len(st), widths, len(up), len(down), len(left)))
    for s in sorted(up, key=lambda s: -s["rcjk"][0] / s["rcjk"][1]):
        print("  Gaia DR3 %s: BP-RP %.2f, Ks %.2f, J-Ks +%.3f (width %.3f), J-W1 %+.3f; 2MASS %s; SIMBAD %s; %s" % (
            s["id"], s["c"], s["k"], s["rcjk"][0], s["rcjk"][1], s["rcjw"][0], s["flags"], s["simbad"], s["vet"]))
    if record:
        core.record_run("ks_only_excess", "Light at 2 um alone", "Gaia Catalogue of Nearby Stars with its 2MASS and WISE photometry (external.gaiaedr3_gcns_main_1); "
                        "2MASS (II/246) by CDS XMatch; SIMBAD", "stars within 100 pc at BP-RP %g-%g with clean J, Ks and W1: %d" % (colour_min, colour_max, len(st)),
                        os.path.abspath(__file__), "ksexcess", len(st), "stars", dict(widths=widths, errors=errors, w1_widths=w1_widths, colour=[colour_min, colour_max]),
                        "the stars as far below the J-Ks locus: %d" % len(down), "a star brighter at 2 um than its 1.2 and 3.4 um light allow",
                        "%d above against %d below; %d unexplained" % (len(up), len(down), len(left)),
                        [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["rcjk"][0] / s["rcjk"][1], 1), unit="widths above the J-Ks locus",
                              vet=s["vet"] or ("chance-rate" if len(up) <= len(down) + 3 * math.sqrt(max(len(down), 1)) else None),
                              note="Ks %.2f, J-W1 %+.2f; 2MASS %s; SIMBAD %s" % (s["k"], s["rcjw"][0], s["flags"], s["simbad"])) for s in up])
        print("recorded")


warmgalaxies_DOC = r"""Galaxies warm without star formation: HECATE galaxies within 150 Mpc with a GSWLC stellar mass and ultraviolet-optical star-formation rate and with WISE 22 um and IRAS far-infrared luminosities, the quiescent ones (specific star-formation rate below 1e-11 per year) whose 22 um luminosity exceeds a tenth of a solar luminosity per solar mass and lies within a factor of three of their far-infrared luminosity; star-forming dust at 20-40 K peaks at 60-100 um and old stars warm little dust, so mid-infrared heat without young stars or cold dust is what structures at room temperature around a galaxy's stars would radiate. Each read for an active nucleus in HECATE's three flags and for an edge-on disc, whose dust lanes hide the young stars.

    python -I seti.py infrared warmgalaxies [record] [--d-max-mpc 150] [--ssfr-max -11] [--heat-min -1.0] [--warm-min -0.5] [--edge-on-deg 70]"""


def warmgalaxies_main(record=False, d_max_mpc=150.0, ssfr_max=-11.0, heat_min=-1.0, warm_min=-0.5, edge_on_deg=70.0):
    rows = core.tap("vizier", 'SELECT PGC, NED, RAJ2000, DEJ2000, D, i, logLTIR, logLFIR, logL22um, logL12um, logMGSW, logSFRGSW, logSFRTIR, Nuclear, AGNs17, AGNh, W3mag, W4mag '
                              'FROM "J/MNRAS/506/1896/hecate" WHERE D < %g AND logSFRGSW IS NOT NULL AND logMGSW IS NOT NULL AND logL22um IS NOT NULL AND logLFIR IS NOT NULL'
                    % d_max_mpc, timeout=900)
    gal = []
    for r in rows:
        m, sfr, l22, lfir = num(r["logMGSW"]), num(r["logSFRGSW"]), num(r["logL22um"]), num(r["logLFIR"])
        if None in (m, sfr, l22, lfir):
            continue
        # HECATE's luminosities are in erg/s; a solar luminosity is 3.828e33 erg/s
        l22, lfir = l22 - 33.583, lfir - 33.583
        gal.append(dict(pgc=r["PGC"], name=(r["NED"] or "").strip(), d=num(r["D"]), inc=num(r["i"]), m=m, ssfr=sfr - m, heat=l22 - m, warm=l22 - lfir, l22=l22, lfir=lfir,
                        ltir=num(r["logLTIR"]), agn=bool((r["AGNs17"] or "").strip() or (r["AGNh"] or "").strip() or (r["Nuclear"] or "").strip() not in ("", "0")),
                        w34=(num(r["W3mag"]) - num(r["W4mag"])) if num(r["W3mag"]) is not None and num(r["W4mag"]) is not None else None))
    quiet = [g for g in gal if g["ssfr"] < ssfr_max]
    forming = [g for g in gal if g["ssfr"] >= ssfr_max]
    heat_q = np.array([g["heat"] for g in quiet])
    warm_q = np.array([g["warm"] for g in quiet])
    hits = [g for g in quiet if g["heat"] > heat_min and g["warm"] > warm_min]
    for g in hits:
        g["vet"] = "background-galaxy" if g["agn"] else "dust-lane" if g["inc"] is not None and g["inc"] > edge_on_deg else None
    left = [g for g in hits if g["vet"] is None]
    print("%d galaxies within %g Mpc with masses, ultraviolet-optical star-formation rates, 22 um and far-infrared luminosities; quiescent (specific rate below 1e%g): %d, "
          "whose 22 um heat per stellar mass runs median %.2f (99th percentile %.2f) and 22 um over far-infrared median %.2f (99th %.2f); above %g per solar mass and within "
          "%g dex of the far-infrared: %d, %d with active nuclei, %d edge-on; unexplained %d" % (
              len(gal), d_max_mpc, ssfr_max, len(quiet), float(np.median(heat_q)) if len(heat_q) else 0, float(np.percentile(heat_q, 99)) if len(heat_q) else 0,
              float(np.median(warm_q)) if len(warm_q) else 0, float(np.percentile(warm_q, 99)) if len(warm_q) else 0, heat_min, warm_min, len(hits),
              sum(g["vet"] == "background-galaxy" for g in hits), sum(g["vet"] == "dust-lane" for g in hits), len(left)))
    for g in sorted(hits, key=lambda g: -g["heat"]):
        print("  %s (PGC %s, %s Mpc): log M %.2f, specific rate %.2f, 22 um per mass %.2f, 22 um over far-infrared %+.2f, W3-W4 %s, inclination %s; %s" % (
            g["name"], g["pgc"], g["d"], g["m"], g["ssfr"], g["heat"], g["warm"], "%.2f" % g["w34"] if g["w34"] is not None else "-", g["inc"], g["vet"] or "unexplained"))
    json.dump(dict(n=len(gal), quiet=len(quiet), hits=hits), open(core.results("warm_quiet_galaxies.json"), "w"), indent=1)
    if record:
        core.record_run(
            "hecate_warm_quiet", "Galaxies warm without star formation", "HECATE (J/MNRAS/506/1896) with its GSWLC, WISE and IRAS luminosities",
            "quiescent galaxies within %g Mpc with masses, rates, 22 um and far-infrared luminosities: %d of %d" % (d_max_mpc, len(quiet), len(gal)), os.path.abspath(__file__),
            "warmgalaxies", len(quiet), "galaxies", dict(d_max_mpc=d_max_mpc, ssfr_max=ssfr_max, heat_min=heat_min, warm_min=warm_min, edge_on_deg=edge_on_deg),
            "the quiescent galaxies' own 22 um heat per mass (median %.2f, 99th percentile %.2f) and 22 um over far-infrared (median %.2f, 99th %.2f); the %d star-forming "
            "galaxies apart" % (float(np.median(heat_q)) if len(heat_q) else 0, float(np.percentile(heat_q, 99)) if len(heat_q) else 0,
                                float(np.median(warm_q)) if len(warm_q) else 0, float(np.percentile(warm_q, 99)) if len(warm_q) else 0, len(forming)),
            "a quiescent galaxy radiating a tenth of a solar luminosity per solar mass at 22 um with little cold dust",
            "%d: %d with active nuclei, %d edge-on; %d unexplained" % (len(hits), sum(g["vet"] == "background-galaxy" for g in hits), sum(g["vet"] == "dust-lane" for g in hits),
                                                                     len(left)),
            [dict(target="pgc:%s" % g["pgc"], stat=g["heat"], unit="log 22 um luminosity per stellar mass, solar units", vet=g["vet"],
                  note="%s, %s Mpc, log M %.2f, specific rate %.2f, 22 um over far-infrared %+.2f, inclination %s" % (g["name"], g["d"], g["m"], g["ssfr"], g["warm"], g["inc"]))
             for g in hits])
        print("recorded")


irdim_DOC = r"""Infrared dimming next door: Gaia Catalogue of Nearby Stars objects within 20 pc with an AllWISE W1 between 8 and 15 mag, carried to 2015.0 and matched to CatWISE2020 (II/365) within 3 arcsec by CDS XMatch, whose profile W1 changed by more than 0.3 mag between the 2010 AllWISE mean (II/328, matched at the 2010.5 position) and the 2010-2018 CatWISE mean, a star whose 3.4 um light fell or rose by a third over the decade while nothing in the optical is asked; each read for its Gaia variability flag, its RUWE and image flags, its SIMBAD type and the AllWISE and CatWISE flags, with the stars that brightened as the mirror; CatWISE saturates brighter than W1 8.5, a match beyond 1.5 arcsec at a star next door is another source, and a star moving faster than a second a year is split between the epochs of CatWISE's coadds; a star still unexplained is read in NEOWISE's single exposures of 2014-2024 at its own position on each date, and one holding AllWISE's W1 there to 0.15 mag did not fade, CatWISE's proper-motion-fitted W1 of a fast star having moved instead.

    python -I seti.py infrared irdim [record] [--plx-min 50] [--dmag 0.3] [--radius-arcsec 3] [--saturation 8.5] [--match-arcsec 1.5] [--fast-mas-yr 1000] [--neowise-tol 0.15]"""


def irdim_main(record=False, plx_min=50.0, dmag=0.3, radius_arcsec=3.0, saturation=8.5, match_arcsec=1.5, fast_mas_yr=1000.0, neowise_tol=0.15):
    rows = catalogues.gcns("source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe, ipd_frac_multi_peak, w1mpro_pm_wise, "
                           "w1sigmpro_pm_wise", "gcns_prob > 0.9 AND parallax > %g AND w1mpro_pm_wise BETWEEN 8 AND 15 AND w1sigmpro_pm_wise < 0.05" % plx_min, ranges=12)
    # AllWISE's profile magnitude at 2010.5 and CatWISE's at 2015.0, each matched at the star's position for its epoch
    def placed(epoch):
        out = []
        for r in rows:
            ra, de = core.star_at(float(r["ra"]), float(r["dec"]), num(r["pmra"]) or 0.0, num(r["pmdec"]) or 0.0, 2016.0, epoch)
            out.append(dict(source_id=str(r["source_id"]), ra=ra % 360, dec=de))
        return out
    allw, cat = {}, {}
    for k in range(0, len(rows), 5000):
        for m in core.xmatch(placed(2010.5)[k:k + 5000], "II/328/allwise", radius_arcsec):
            w1 = num(m.get("W1mag"))
            if w1 is not None and (m["source_id"] not in allw or float(m["angDist"]) < allw[m["source_id"]][0]):
                allw[m["source_id"]] = (float(m["angDist"]), w1, num(m.get("e_W1mag")), (m.get("ccf") or "").strip())
        for m in core.xmatch(placed(2015.0)[k:k + 5000], "II/365/catwise", radius_arcsec):
            w1 = num(m.get("W1mproPM"))
            if w1 is not None and (m["source_id"] not in cat or float(m["angDist"]) < cat[m["source_id"]][0]):
                cat[m["source_id"]] = (float(m["angDist"]), w1, num(m.get("e_W1mproPM")), (m.get("ab_flags") or m.get("abFlags") or "").strip())
    st = []
    for r in rows:
        a, c = allw.get(str(r["source_id"])), cat.get(str(r["source_id"]))
        if not a or not c:
            continue
        st.append(dict(id=str(r["source_id"]), w1_all=a[1], w1_cat=c[1], d=c[1] - a[1], e=math.hypot(a[2] or 0, c[2] or 0), sep=c[0], flags=(a[3] + "/" + c[3]).strip("/"),
                       g=num(r["phot_g_mean_mag"]), ruwe=num(r["ruwe"]), multi=num(r["ipd_frac_multi_peak"]), var="", pm=math.hypot(num(r["pmra"]) or 0.0, num(r["pmdec"]) or 0.0),
                       ra=float(r["ra"]), dec=float(r["dec"]), pmra=num(r["pmra"]) or 0.0, pmdec=num(r["pmdec"]) or 0.0, neowise=None,
                       c=(num(r["phot_bp_mean_mag"]) - num(r["phot_rp_mean_mag"])) if num(r["phot_bp_mean_mag"]) is not None and num(r["phot_rp_mean_mag"]) is not None else None))
    d = np.array([s["d"] for s in st])
    off = float(np.median(d)) if len(d) else 0.0
    dim = [s for s in st if s["d"] - off > dmag]
    bright = [s for s in st if s["d"] - off < -dmag]
    both = dim + bright
    sim = {}
    for k in range(0, len(both), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in both[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
        for r in core.tap("gaia", "SELECT source_id, phot_variable_flag FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(s["id"] for s in both[k:k + 200])):
            for s in both:
                if s["id"] == str(r["source_id"]):
                    s["var"] = str(r["phot_variable_flag"] or "")
    var_types = ("V*", "Mi*", "LP*", "Pu*", "RR*", "dS*", "Ce*", "EB*", "Fl*", "BY*", "RS*", "Er*", "TT*", "Y*O", "Or*", "CV*", "No*", "Ro*", "El*", "a2*", "sr*", "Em*")
    for s in both:
        s["simbad"] = sim.get(s["id"])
        typ = (s["simbad"] or (None, "", ""))[1]
        # CatWISE saturates brighter than W1 8.5; a match beyond 1.5 arcsec at a star next door is another source, and a star moving faster than a second a year
        # is split between the epochs of CatWISE's coadds, each half carrying part of its light
        s["vet"] = ("variable-star" if s["var"] == "VARIABLE" or typ in var_types else
                    "blend" if (s["ruwe"] or 0) > 1.4 or (s["multi"] or 0) > 2 or typ in ("**", "SB*") else
                    "artefact" if s["flags"].replace("/", "").strip("0") or s["w1_all"] < saturation else
                    "astrometry" if s["sep"] > match_arcsec or s["pm"] > fast_mas_yr else None)
    # the rest against NEOWISE's single exposures of 2014-2024, each read at the star's own position on its date: a star whose 3.4 um light holds at
    # AllWISE's level there did not fade, and CatWISE's proper-motion-fitted W1 of a fast star is what moved
    for s in both:
        if s["vet"] is not None:
            continue
        ra19, de19 = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, 2019.5)
        found = core.tap("irsa", "SELECT mjd, ra, dec, w1mpro, qual_frame FROM neowiser_p1bs_psd WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f)) = 1"
                         % (ra19 % 360, de19, 9 / 3600.0), timeout=900)
        w = []
        for x in found:
            if num(x["w1mpro"]) is None or (num(x["qual_frame"]) or 0) <= 0:
                continue
            pr, pd = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, 2000.0 + (float(x["mjd"]) - 51544.5) / 365.25)
            if math.hypot((float(x["ra"]) - pr) * math.cos(math.radians(pd)), float(x["dec"]) - pd) * 3600 < 2.0:
                w.append(float(x["w1mpro"]))
        s["neowise"] = (len(w), float(np.median(w))) if w else (0, None)
        if len(w) >= 10 and abs(s["neowise"][1] - s["w1_all"]) < neowise_tol:
            s["vet"] = "artefact"
    left = [s for s in both if s["vet"] is None]
    count = {v: sum(s["vet"] == v for s in both) for v in ("variable-star", "blend", "artefact", "astrometry")}
    print("%d stars within %.0f pc with AllWISE and CatWISE W1; CatWISE minus AllWISE median %+.3f, robust scatter %.3f; fainter by %g: %d, brighter by as much: %d; variables %d, "
          "blends %d, flagged or saturated %d, matched to another source %d; unexplained %d" % (len(st), 1000 / plx_min, off, core.robust_sigma(d) if len(d) else 0, dmag,
                                                                                               len(dim), len(bright), count["variable-star"], count["blend"], count["artefact"],
                                                                                               count["astrometry"], len(left)))
    for s in sorted(both, key=lambda s: -abs(s["d"] - off)):
        print("  Gaia DR3 %s: W1 %.3f (AllWISE) to %.3f (CatWISE), %+.3f +- %.3f at %.1f arcsec, flags %s; G %s, BP-RP %s, RUWE %s, %s, %.0f mas/yr; SIMBAD %s; %s" % (
            s["id"], s["w1_all"], s["w1_cat"], s["d"] - off, s["e"], s["sep"], s["flags"] or "-", s["g"], s["c"], s["ruwe"], s["var"], s["pm"], s["simbad"],
            s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), offset=off, dim=dim, bright=bright), open(core.results("infrared_dimming_nearby.json"), "w"), indent=1)
    if record:
        core.record_run(
            "allwise_catwise_w1_change", "Infrared dimming next door", "Gaia Catalogue of Nearby Stars (external.gaiaedr3_gcns_main_1); AllWISE (II/328) and "
            "CatWISE2020 (II/365) by CDS XMatch; SIMBAD", "stars within %.0f pc with W1 8-15 matched in AllWISE and CatWISE: %d" % (1000 / plx_min, len(st)),
            os.path.abspath(__file__), "irdim", len(st), "stars", dict(plx_min=plx_min, dmag=dmag, radius_arcsec=radius_arcsec, saturation=saturation,
                                                                       match_arcsec=match_arcsec, fast_mas_yr=fast_mas_yr, epochs=[2010.5, 2015.0]),
            "the stars that brightened by as much: %d; the sample's offset %+.3f and scatter %.3f" % (len(bright), off, core.robust_sigma(d) if len(d) else 0),
            "a change of %g mag at 3.4 um between the AllWISE and CatWISE means" % dmag,
            "%d fainter against %d brighter; %d variables, %d blends, %d flagged or saturated, %d matched to another source; %d unexplained" % (
                len(dim), len(bright), count["variable-star"], count["blend"], count["artefact"], count["astrometry"], len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["d"] - off, 3), unit="CatWISE minus AllWISE W1 less the offset, mag", vet=s["vet"],
                  note="W1 %.3f to %.3f +- %.3f at %.1f arcsec, flags %s; G %s, RUWE %s, %s, %.0f mas/yr; SIMBAD %s%s" % (
                      s["w1_all"], s["w1_cat"], s["e"], s["sep"], s["flags"] or "-", s["g"], s["ruwe"], s["var"], s["pm"], s["simbad"],
                      "; NEOWISE 2014-2024: %d single exposures, median W1 %.3f" % s["neowise"] if s.get("neowise") and s["neowise"][1] is not None else "")) for s in both])
        print("recorded")


hidark_DOC = r"""Waste heat in HI-dark galaxies: the ALFALFA HI sources of Leisman et al. (2017) (VizieR J/ApJ/842/133) whose optical counterparts are low-surface-brightness ghosts, gas with hardly a star, matched to AllWISE within 10 arcsec, a source at 5 sigma in W3 or W4 where so few stars could warm dust and a civilisation drawing on the gas would radiate, against the same positions moved 2 arcmin in eight directions and against a Gaia star or an AGN colour (W1-W2 above 0.5) at the match.

    python -I seti.py infrared hidark [record] [--radius-arcsec 10] [--snr 5] [--w1w2 0.5]"""


def hidark_main(record=False, radius_arcsec=10.0, snr=5.0, w1w2=0.5):
    from scipy.stats import poisson
    from setitools.catalogues import shifted_matches
    rows = core.tap("vizier", 'SELECT AGC, RAJ2000, DEJ2000, cz, Dist, logMHI, mug0, gMAG, rh FROM "J/ApJ/842/133/table1"')
    pos = [dict(name="a%s" % r["AGC"], ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"])) for r in rows if num(r["RAJ2000"]) is not None]
    hits, null = shifted_matches(pos, "II/328/allwise", radius_arcsec)

    def warm(m):
        e3, e4 = num(m.get("e_W3mag")), num(m.get("e_W4mag"))
        return (e3 is not None and 1.0857 / e3 >= snr) or (e4 is not None and 1.0857 / e4 >= snr)
    got = {}
    for m in hits:
        if warm(m) and (m["source_id"] not in got or float(m["angDist"]) < float(got[m["source_id"]]["angDist"])):
            got[m["source_id"]] = m
    stars = set()
    if got:
        up = [p for p in pos if p["name"] in got]
        stars = {m["source_id"] for m in core.xmatch([dict(source_id=p["name"], ra=p["ra"], dec=p["dec"]) for p in up], "I/355/gaiadr3", radius_arcsec)}
    # the moved positions count every match, warm or not; the warm share among the matches at the true positions scales them
    raw = len({m["source_id"] for m in hits})
    exp = (float(np.mean(null)) if len(null) else 0.0) * (len(got) / raw if raw else 0.0)
    out = []
    for n, m in got.items():
        w1, w2 = num(m.get("W1mag")), num(m.get("W2mag"))
        agn = w1 is not None and w2 is not None and w1 - w2 >= w1w2
        out.append(dict(agc=n[1:], sep=float(m["angDist"]), w1=w1, w2=w2, w3=num(m.get("W3mag")), w4=num(m.get("W4mag")), e3=num(m.get("e_W3mag")), e4=num(m.get("e_W4mag")),
                        agn=agn, star=n in stars, vet="background-galaxy" if agn else "stellar-blend" if n in stars else None))
    p = float(poisson.sf(len(out) - 1, exp)) if out and exp > 0 else 1.0
    for o in out:
        if o["vet"] is None and p > 0.01:
            o["vet"] = "chance-rate"
    left = [o for o in out if o["vet"] is None]
    print("%d HI-dark galaxies; AllWISE matches within %g arcsec: %d, of them at %g sigma in W3 or W4: %d (the positions moved 2 arcmin give %s matches, mean %.1f, so %.1f "
          "warm; p %.3g); %d AGN-coloured, %d Gaia stars; unexplained %d" % (len(pos), radius_arcsec, raw, snr, len(out), null, float(np.mean(null)) if len(null) else 0, exp, p,
                                                                             sum(o["agn"] for o in out), sum(o["star"] for o in out), len(left)))
    for o in sorted(out, key=lambda o: o["sep"]):
        print("  AGC %s: W1 %s W2 %s W3 %s (+-%s) W4 %s (+-%s) at %.1f arcsec, %s%s; %s" % (o["agc"], o["w1"], o["w2"], o["w3"], o["e3"], o["w4"], o["e4"], o["sep"],
                                                                                       "AGN colour" if o["agn"] else "galaxy colour", ", Gaia star" if o["star"] else "",
                                                                                       o["vet"] or "unexplained"))
    json.dump(dict(n=len(pos), null=null, hits=out), open(core.results("hidark_waste_heat.json"), "w"), indent=1)
    if record:
        core.record_run(
            "alfalfa_hidark_waste_heat", "Waste heat in HI-dark galaxies", "Leisman et al. (2017) ALFALFA HI-dark galaxies (VizieR J/ApJ/842/133); AllWISE (II/328) and Gaia DR3 "
            "by CDS XMatch", "HI-dark galaxies: %d" % len(pos), os.path.abspath(__file__), "hidark", len(pos), "galaxies",
            dict(radius_arcsec=radius_arcsec, snr=snr, w1w2=w1w2), "the positions moved 2 arcmin in eight directions: %s matches, scaled by the warm share %d of %d" % (
                null, len(out), raw),
            "a W3 or W4 source at %g sigma within %g arcsec of the counterpart" % (snr, radius_arcsec),
            "%d against %.1f expected from the moved positions (p %.3g); %d AGN-coloured, %d Gaia stars; %d unexplained" % (
                len(out), exp, p, sum(o["agn"] for o in out), sum(o["star"] for o in out), len(left)),
            [dict(target="agc:%s" % o["agc"], stat=max(1.0857 / o["e3"] if o["e3"] else 0, 1.0857 / o["e4"] if o["e4"] else 0), unit="W3 or W4 S/N", vet=o["vet"],
                  note="W1 %s W2 %s W3 %s W4 %s at %.1f arcsec" % (o["w1"], o["w2"], o["w3"], o["w4"], o["sep"])) for o in out])
        print("recorded")


missingdwarfs_DOC = r"""Missing bolometric flux, dwarfs: the JMMC Measured Stellar Diameters Catalogue (VizieR II/345) limb-darkened diameters of dwarfs (GSP-Phot log g above 4) within 100 pc, matched to Gaia DR3 within 5 arcsec by CDS XMatch, each diameter and parallax giving a radius, each radius and GSP-Phot temperature a luminosity, against Gaia's FLAME luminosity from the star's light; a star shining less than its surface and temperature allow by 20 per cent at 3 sigma has light missing, as a shade around it would take, and the stars shining more by as much are the mirror. Each read for its SIMBAD type, since a companion within the interferometer's field adds light or area.

    python -I seti.py infrared missingdwarfs [record] [--deficit 0.2] [--sigma 3] [--logg-min 4] [--plx-min 10] [--radius-arcsec 5]"""


def missingdwarfs_main(record=False, deficit=0.2, sigma=3.0, logg_min=4.0, plx_min=10.0, radius_arcsec=5.0):
    jm = core.tap("vizier", 'SELECT SimbadName, "_RA", "_DE", LDdiam, e_LDdiam, Band, Method FROM "II/345/jmdc" WHERE LDdiam > 0 AND e_LDdiam > 0')
    up = [dict(source_id=str(k), ra=float(r["_RA"]), dec=float(r["_DE"])) for k, r in enumerate(jm)]
    gm = {}
    for m in core.xmatch(up, "I/355/gaiadr3", radius_arcsec):
        if num(m.get("Plx")) and (m["source_id"] not in gm or (num(m.get("Gmag")) or 99) < (num(gm[m["source_id"]].get("Gmag")) or 99)):
            gm[m["source_id"]] = m
    ids = sorted({str(m["Source"]) for m in gm.values()})
    ap = core.tap_by_ids("gaia", ids, "SELECT source_id, teff_gspphot, teff_gspphot_lower, teff_gspphot_upper, logg_gspphot, lum_flame, lum_flame_lower, lum_flame_upper "
                                      "FROM gaiadr3.astrophysical_parameters WHERE source_id IN (%s)")
    out = []
    for k, m in gm.items():
        a = ap.get(str(m["Source"]))
        r = jm[int(k)]
        if not a or num(a["teff_gspphot"]) is None or num(a["lum_flame"]) is None or (num(a["logg_gspphot"]) or 0) <= logg_min or num(m["Plx"]) <= plx_min:
            continue
        th, eth, plx, eplx = num(r["LDdiam"]), num(r["e_LDdiam"]), num(m["Plx"]), num(m.get("e_Plx")) or 0.0
        T, Tlo, Thi = num(a["teff_gspphot"]), num(a["teff_gspphot_lower"]) or num(a["teff_gspphot"]), num(a["teff_gspphot_upper"]) or num(a["teff_gspphot"])
        L, Llo, Lhi = num(a["lum_flame"]), num(a["lum_flame_lower"]) or num(a["lum_flame"]), num(a["lum_flame_upper"]) or num(a["lum_flame"])
        # radius in solar units from the angular diameter and the parallax (a solar radius at 1 kpc subtends 9.30e-3 mas of diameter), luminosity from radius and temperature
        R = th / (9.30e-3 * plx)
        pred = R * R * (T / 5772.0) ** 4
        e_pred = pred * math.sqrt((2 * eth / th) ** 2 + (2 * eplx / plx) ** 2 + (4 * (Thi - Tlo) / (2 * T)) ** 2)
        e_l = (Lhi - Llo) / 2
        ratio = L / pred
        z = (pred * (1 - deficit) - L) / math.hypot(e_pred * (1 - deficit), e_l) if (e_pred or e_l) else 0.0
        zm = (L - pred * (1 + deficit)) / math.hypot(e_pred * (1 + deficit), e_l) if (e_pred or e_l) else 0.0
        out.append(dict(name=str(r["SimbadName"]).strip(), gaia=str(m["Source"]), theta=th, R=R, T=T, pred=pred, L=L, ratio=ratio, z=z, zm=zm, band=r["Band"], method=r["Method"]))
    deficits = [o for o in out if o["z"] > sigma]
    excess = [o for o in out if o["zm"] > sigma]
    sim = {}
    both = deficits + excess
    for k in range(0, len(both), 200):
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % o["gaia"] for o in both[k:k + 200])):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    for o in both:
        o["simbad"] = sim.get(o["gaia"])
        typ = (o["simbad"] or (None, "", ""))[1]
        o["vet"] = "binary" if typ in ("**", "SB*", "EB*") else "natural-type" if typ not in ("", "*", "PM*", "HV*") else None
    rat = np.array([o["ratio"] for o in out])
    left = [o for o in deficits if o["vet"] is None]
    print("%d dwarfs within %.0f pc with a measured diameter, a GSP-Phot temperature and a FLAME luminosity; Gaia over predicted luminosity median %.3f, robust scatter %.3f; "
          "short by %.0f per cent at %g sigma: %d, over by as much: %d; unexplained deficits %d" % (
              len(out), 1000 / plx_min, float(np.median(rat)) if len(rat) else 0, core.robust_sigma(rat) if len(rat) else 0, 100 * deficit, sigma, len(deficits), len(excess),
              len(left)))
    for o in sorted(both, key=lambda o: o["ratio"]):
        print("  %s (Gaia DR3 %s): %.3f mas, R %.3f, Teff %.0f K, predicted %.3f against FLAME %.3f Lsun (ratio %.2f, %s %.1f sigma), %s %s; SIMBAD %s; %s" % (
            o["name"], o["gaia"], o["theta"], o["R"], o["T"], o["pred"], o["L"], o["ratio"], "short" if o["z"] > sigma else "over", max(o["z"], o["zm"]), o["band"], o["method"],
            o["simbad"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(out), deficits=deficits, excess=excess), open(core.results("missing_flux_dwarfs.json"), "w"), indent=1)
    if record:
        core.record_run(
            "jmdc_missing_flux_dwarfs", "Missing bolometric flux, dwarfs", "JMMC Measured Stellar Diameters Catalogue (VizieR II/345); Gaia DR3 GSP-Phot and FLAME; SIMBAD",
            "dwarfs within %.0f pc with a measured diameter, a temperature and a FLAME luminosity: %d" % (1000 / plx_min, len(out)), os.path.abspath(__file__), "missingdwarfs",
            len(out), "stars", dict(deficit=deficit, sigma=sigma, logg_min=logg_min, plx_min=plx_min, radius_arcsec=radius_arcsec),
            "the stars over-luminous by as much: %d; the sample's ratio, median %.3f, scatter %.3f" % (len(excess), float(np.median(rat)) if len(rat) else 0,
                                                                                                    core.robust_sigma(rat) if len(rat) else 0),
            "a star shining %.0f per cent less than its measured surface and temperature allow, at %g sigma" % (100 * deficit, sigma),
            "%d short against %d over; %d unexplained" % (len(deficits), len(excess), len(left)),
            [dict(target="gaia_dr3:%s" % o["gaia"], stat=o["ratio"], unit="FLAME over predicted luminosity", vet=o["vet"],
                  note="%s, %.3f mas, R %.3f, Teff %.0f, %s %s; SIMBAD %s" % (o["name"], o["theta"], o["R"], o["T"], o["band"], o["method"], o["simbad"])) for o in both])
        print("recorded")


hyadeswarm_DOC = r"""Warm stars in the Hyades: the Hyades members of Hunt and Reffert (2023) (VizieR J/A+A/673/A114, probability above 0.5) with BP-RP 0.5-2.5, matched to AllWISE within 3 arcsec at 2010.5, whose W1-W3 lies more than five robust widths and five errors above the cluster's own locus at their colour, mid-infrared heat at a star of the nearest cluster, which a settled cluster would show in many of its stars; the share set against the field's 0.8 per cent from the same test on the stars within 100 pc. Each candidate read for its AllWISE flags and its SIMBAD type.

    python -I seti.py infrared hyadeswarm [record] [--cluster Melotte_25] [--prob-min 0.5] [--widths 5] [--errors 5] [--field-share 0.0077] [--w3-err-max 0.2]"""


def cluster_warm_stars(cluster, prob_min=0.5, widths=5.0, errors=5.0, w3_err_max=0.2):
    """Members of CLUSTER (Hunt and Reffert 2023) at BP-RP 0.5-2.5 with clean AllWISE W1 and W3, and those warm above the cluster's own W1-W3 locus, vetted by flags and SIMBAD type."""
    rows = core.tap("vizier", 'SELECT GaiaDR3, RA_ICRS, DE_ICRS, pmRA, pmDE, Plx, Gmag, "BP-RP", Prob FROM "J/A+A/673/A114/members" WHERE Name = \'%s\' AND Prob > %g' % (cluster, prob_min))
    st = [dict(id=str(r["GaiaDR3"]), ra=float(r["RA_ICRS"]), dec=float(r["DE_ICRS"]), pmra=num(r["pmRA"]) or 0.0, pmdec=num(r["pmDE"]) or 0.0, g=num(r["Gmag"]), c=num(r["BP-RP"]))
          for r in rows if num(r["BP-RP"]) is not None and 0.5 <= num(r["BP-RP"]) <= 2.5]
    up = []
    for s in st:
        ra, de = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, 2010.5)
        up.append(dict(source_id=s["id"], ra=ra % 360, dec=de))
    w = {}
    for k in range(0, len(up), 5000):
        for m in core.xmatch(up[k:k + 5000], "II/328/allwise", 3.0):
            w1, w3, e1, e3 = num(m.get("W1mag")), num(m.get("W3mag")), num(m.get("e_W1mag")), num(m.get("e_W3mag"))
            if None not in (w1, w3, e1, e3) and e3 < w3_err_max and (m["source_id"] not in w or float(m["angDist"]) < w[m["source_id"]]["sep"]):
                w[m["source_id"]] = dict(sep=float(m["angDist"]), c13=w1 - w3, e13=math.hypot(e1, e3), ccf=(m.get("ccf") or "").strip(), ex=(m.get("ex") or "").strip())
    st = [dict(s, **w[s["id"]]) for s in st if s["id"] in w]
    bins = np.floor(np.array([s["c"] for s in st]) / 0.2).astype(int)
    val = np.array([s["c13"] for s in st])
    for b in set(bins.tolist()):
        k = np.nonzero(bins == b)[0]
        med = float(np.median(val[k])) if len(k) >= 10 else None
        wid = 1.4826 * float(np.median(np.abs(val[k] - med))) if med is not None else None
        for i in k:
            st[i]["r"] = (val[i] - med, wid) if med is not None else None
    up_ = [s for s in st if s["r"] and s["r"][0] > max(widths * s["r"][1], errors * s["e13"])]
    sim = {}
    for k in range(0, len(up_), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in up_[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in up_:
        s["simbad"] = sim.get(s["id"])
        s["vet"] = ("artefact" if s["ccf"].strip("0") or s["ex"] not in ("", "0") else
                    "natural-type" if s["simbad"] and s["simbad"][1] not in ("*", "PM*", "HV*", "low-mass*") else None)
    return st, up_


def hyadeswarm_main(record=False, cluster="Melotte_25", prob_min=0.5, widths=5.0, errors=5.0, field_share=0.0077, w3_err_max=0.2):
    from scipy.stats import binom
    st, up_ = cluster_warm_stars(cluster, prob_min, widths, errors, w3_err_max)
    p = float(binom.sf(len(up_) - 1, len(st), field_share)) if up_ else 1.0
    for s in up_:
        if s["vet"] is None and p > 0.01:
            s["vet"] = "chance-rate"
    left = [s for s in up_ if s["vet"] is None]
    print("%d members of %s at BP-RP 0.5-2.5 with clean W1 and W3; W1-W3 %g widths above the cluster's locus: %d, a share of %.2f per cent against the field's %.2f (binomial p %.3g); "
          "%d flagged, %d of known types; unexplained %d" % (len(st), cluster, widths, len(up_), 100 * len(up_) / max(len(st), 1), 100 * field_share, p,
                                                              sum(s["vet"] == "artefact" for s in up_), sum(s["vet"] == "natural-type" for s in up_), len(left)))
    for s in sorted(up_, key=lambda s: -s["r"][0] / s["r"][1]):
        print("  Gaia DR3 %s: BP-RP %.2f, G %.2f, W1-W3 +%.3f (width %.3f, error %.3f), flags %s/%s; SIMBAD %s; %s" % (
            s["id"], s["c"], s["g"], s["r"][0], s["r"][1], s["e13"], s["ccf"] or "-", s["ex"] or "-", s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), warm=up_), open(core.results("hyades_warm_stars.json"), "w"), indent=1)
    if record:
        core.record_run(
            "hyades_warm_stars", "Warm stars in the Hyades", "Hunt and Reffert (2023) cluster members (VizieR J/A+A/673/A114); AllWISE (II/328) by CDS XMatch; SIMBAD",
            "%s members at BP-RP 0.5-2.5 with clean W1 and W3: %d" % (cluster, len(st)), os.path.abspath(__file__), "hyadeswarm", len(st), "stars",
            dict(cluster=cluster, prob_min=prob_min, widths=widths, errors=errors, field_share=field_share, w3_err_max=w3_err_max), "the field's share of %.2f per cent from the same test within 100 pc" % (
                100 * field_share), "a share of stars with 12 um excess above the field's", "%d of %d (%.2f per cent) against the field's %.2f (binomial p %.3g); %d flagged, "
            "%d of known types, %d unexplained" % (len(up_), len(st), 100 * len(up_) / max(len(st), 1), 100 * field_share, p, sum(s["vet"] == "artefact" for s in up_),
                                                  sum(s["vet"] == "natural-type" for s in up_), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["r"][0] / s["r"][1], 1), unit="widths above the cluster's W1-W3 locus", vet=s["vet"],
                  note="BP-RP %.2f, G %.2f, flags %s/%s; SIMBAD %s" % (s["c"], s["g"], s["ccf"] or "-", s["ex"] or "-", s["simbad"])) for s in up_])
        print("recorded")


w1only_DOC = r"""Light at 3.4 um alone: Gaia Catalogue of Nearby Stars objects (probability above 0.9, white-dwarf probability below 0.1, RUWE below 1.4, one IPD peak) at BP-RP 0.5-2.5 with 2MASS Ks and WISE W1, W2 and W3 measured to 0.03, 0.05, 0.05 and 0.1 mag, whose Ks-W1 lies more than five robust widths and five errors above the single-star locus at their BP-RP while Ks-W2 stays within two widths of its own; hot dust that brightens 3.4 um brightens 4.6 um more, so light confined to the W1 band is no photosphere's and no dust's, though the 3.3 um feature of aromatic grains lifts W1 alone where a nebula lies behind. The stars as far below the Ks-W1 locus are the mirror null. Each candidate read in AllWISE for its flags and its plain profile W1, since the catalogue's proper-motion-fitted W1 mis-fits some fast movers, in 2MASS for its quality flags and in SIMBAD.

    python -I seti.py infrared w1only [record] [--widths 5] [--errors 5] [--w2-widths 2] [--colour-min 0.5] [--colour-max 2.5]"""


def w1only_main(record=False, widths=5.0, errors=5.0, w2_widths=2.0, colour_min=0.5, colour_max=2.5):
    rows = catalogues.gcns("source_id, phot_bp_mean_mag, phot_rp_mean_mag, k_m_2mass, k_msig_2mass, w1mpro_pm_wise, w1sigmpro_pm_wise, w2mpro_pm_wise, w2sigmpro_pm_wise, ra, dec",
                           "gcns_prob > 0.9 AND wd_prob < 0.1 AND ruwe < 1.4 AND ipd_frac_multi_peak <= 2 AND k_msig_2mass < 0.03 AND w1sigmpro_pm_wise < 0.05 "
                           "AND w2sigmpro_pm_wise < 0.05 AND w1mpro_pm_wise > 8", ranges=12, maxrec=200000)
    st = []
    for r in rows:
        bp, rp, k, w1, w2 = num(r["phot_bp_mean_mag"]), num(r["phot_rp_mean_mag"]), num(r["k_m_2mass"]), num(r["w1mpro_pm_wise"]), num(r["w2mpro_pm_wise"])
        if None in (bp, rp, k, w1, w2) or not colour_min <= bp - rp <= colour_max:
            continue
        st.append(dict(id=r["source_id"], c=bp - rp, k=k, w1=w1, ck1=k - w1, ck2=k - w2, ek1=math.hypot(float(r["k_msig_2mass"]), float(r["w1sigmpro_pm_wise"])),
                       ra=float(r["ra"]), dec=float(r["dec"])))
    bins = np.floor(np.array([s["c"] for s in st]) / 0.1).astype(int)
    for key in ("ck1", "ck2"):
        val = np.array([s[key] for s in st])
        for b in set(bins.tolist()):
            kk = np.nonzero(bins == b)[0]
            med = float(np.median(val[kk])) if len(kk) >= 30 else None
            wid = 1.4826 * float(np.median(np.abs(val[kk] - med))) if med is not None else None
            for i in kk:
                st[i]["r" + key] = (val[i] - med, wid) if med is not None else None
    up = [s for s in st if s["rck1"] and s["rck2"] and s["rck1"][0] > max(widths * s["rck1"][1], errors * s["ek1"]) and abs(s["rck2"][0]) < w2_widths * s["rck2"][1]]
    down = [s for s in st if s["rck1"] and s["rck2"] and -s["rck1"][0] > max(widths * s["rck1"][1], errors * s["ek1"]) and abs(s["rck2"][0]) < w2_widths * s["rck2"][1]]
    flags, plain, tm = {}, {}, {}
    if up:
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in up], "II/328/allwise", 3.0):
            if m["source_id"] not in flags or float(m["angDist"]) < flags[m["source_id"]][0]:
                flags[m["source_id"]] = (float(m["angDist"]), (m.get("ccf") or "").strip(), (m.get("ex") or "").strip())
                plain[m["source_id"]] = num(m.get("W1mag"))
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in up], "II/246/out", 3.0):
            if m["source_id"] not in tm or float(m["angDist"]) < tm[m["source_id"]][0]:
                tm[m["source_id"]] = (float(m["angDist"]), (m.get("Qfl") or "").strip(), (m.get("X") or "").strip())
    sim = {}
    for k in range(0, len(up), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in up[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in up:
        f, q = flags.get(s["id"]), tm.get(s["id"])
        s["simbad"] = sim.get(s["id"])
        s["flags"] = f
        s["tmass"] = q
        # the proper-motion-fitted W1 of the nearby-star catalogue against AllWISE's plain profile W1: an excess the plain magnitude lacks is the fit's
        w1p = plain.get(s["id"])
        s["plain_excess"] = (s["k"] - w1p) - (s["ck1"] - s["rck1"][0]) if w1p is not None else None
        # a star with no AllWISE source within 3 arcsec of its 2016 position has moved out of the match, and its fitted W1 belongs to the same mis-fit
        s["vet"] = ("artefact" if (f and (f[1].strip("0") or f[2] not in ("", "0"))) or (q and (q[1] != "AAA" or q[2] not in ("", "0"))) or
                    (s["plain_excess"] is not None and s["plain_excess"] < widths * s["rck1"][1]) else
                    "astrometry" if f is None else
                    "natural-type" if s["simbad"] and s["simbad"][1] not in ("*", "PM*", "HV*", "low-mass*") else None)
    left = [s for s in up if s["vet"] is None]
    print("%d stars; Ks-W1 %g widths above the locus with Ks-W2 on its own: %d; as far below: %d; unexplained %d" % (len(st), widths, len(up), len(down), len(left)))
    for s in sorted(up, key=lambda s: -s["rck1"][0] / s["rck1"][1]):
        print("  Gaia DR3 %s: BP-RP %.2f, W1 %.2f, Ks-W1 +%.3f (width %.3f; with the plain W1 %s), Ks-W2 %+.3f; AllWISE %s; 2MASS %s; SIMBAD %s; %s" % (
            s["id"], s["c"], s["w1"], s["rck1"][0], s["rck1"][1], "%+.3f" % s["plain_excess"] if s["plain_excess"] is not None else "-", s["rck2"][0], s["flags"], s["tmass"],
            s["simbad"], s["vet"]))
    if record:
        core.record_run("w1_only_excess", "Light at 3.4 um alone", "Gaia Catalogue of Nearby Stars with its 2MASS and WISE photometry (external.gaiaedr3_gcns_main_1); "
                        "AllWISE (II/328) by CDS XMatch; SIMBAD", "stars within 100 pc at BP-RP %g-%g with clean Ks, W1 and W2: %d" % (colour_min, colour_max, len(st)),
                        os.path.abspath(__file__), "w1only", len(st), "stars", dict(widths=widths, errors=errors, w2_widths=w2_widths, colour=[colour_min, colour_max]),
                        "the stars as far below the Ks-W1 locus: %d" % len(down), "a star brighter at 3.4 um than its 2.2 and 4.6 um light allow",
                        "%d above against %d below; %d unexplained" % (len(up), len(down), len(left)),
                        [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["rck1"][0] / s["rck1"][1], 1), unit="widths above the Ks-W1 locus",
                              vet=s["vet"] or ("chance-rate" if len(up) <= len(down) + 3 * math.sqrt(max(len(down), 1)) else None),
                              note="W1 %.2f, Ks-W2 %+.2f, excess with the plain W1 %s; AllWISE %s; 2MASS %s; SIMBAD %s" % (
                                  s["w1"], s["rck2"][0], "%+.3f" % s["plain_excess"] if s["plain_excess"] is not None else "-", s["flags"], s["tmass"], s["simbad"])) for s in up])
        print("recorded")


methane_DOC = r"""Methane in a star's light: Gaia Catalogue of Nearby Stars objects (probability above 0.9, white-dwarf probability below 0.1) at BP-RP 0.5-3.0 with WISE W1 and W2 measured to 0.05 mag, fainter than W1 8, where WISE is unsaturated, whose W1-W2 exceeds +0.8, the colour methane absorption at 3.4 um gives a T dwarf and no photosphere above 1500 K can show, a warm star's light carrying a cold body's band; each read in AllWISE for its flags, in Gaia for a second source within the 6 arcsec WISE beam and in SIMBAD, since a T-dwarf companion in the beam, a blend or a contaminated W1 gives the colour naturally.

    python -I seti.py infrared methane [record] [--w1w2-min 0.8] [--w1-min 8] [--colour-min 0.5] [--colour-max 3.0] [--beam-arcsec 8]"""


def methane_main(record=False, w1w2_min=0.8, w1_min=8.0, colour_min=0.5, colour_max=3.0, beam_arcsec=8.0):
    rows = catalogues.gcns("source_id, ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe, ipd_frac_multi_peak, w1mpro_pm_wise, w1sigmpro_pm_wise, "
                           "w2mpro_pm_wise, w2sigmpro_pm_wise",
                           "gcns_prob > 0.9 AND wd_prob < 0.1 AND w1sigmpro_pm_wise < 0.05 AND w2sigmpro_pm_wise < 0.05 AND w1mpro_pm_wise > %g AND w1mpro_pm_wise - w2mpro_pm_wise > %g" % (w1_min, w1w2_min),
                           ranges=12)
    st = []
    for r in rows:
        bp, rp = num(r["phot_bp_mean_mag"]), num(r["phot_rp_mean_mag"])
        if bp is None or rp is None or not colour_min <= bp - rp <= colour_max:
            continue
        st.append(dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), plx=num(r["parallax"]), g=num(r["phot_g_mean_mag"]), c=bp - rp,
                       w1=num(r["w1mpro_pm_wise"]), w2=num(r["w2mpro_pm_wise"]), ruwe=num(r["ruwe"]), multi=num(r["ipd_frac_multi_peak"])))
    flags = {}
    if st:
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in st], "II/328/allwise", 3.0):
            if m["source_id"] not in flags or float(m["angDist"]) < flags[m["source_id"]][0]:
                flags[m["source_id"]] = (float(m["angDist"]), (m.get("ccf") or "").strip(), (m.get("ex") or "").strip(), num(m.get("W1mag")), num(m.get("W2mag")))
    sim = {}
    for k in range(0, len(st), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in st[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in st:
        f = flags.get(s["id"])
        s["flags"] = f
        s["simbad"] = sim.get(s["id"])
        s["plain"] = (f[3] - f[4]) if f and f[3] is not None and f[4] is not None else None
        s["near"] = ["%s (G %.1f)" % (r["source_id"], float(r["phot_g_mean_mag"])) for r in core.tap(
            "gaia", "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f)) AND source_id <> %s "
                    "AND phot_g_mean_mag < %.2f" % (s["ra"], s["dec"], beam_arcsec / 3600.0, s["id"], (s["g"] or 20) + 6))] if s["g"] is not None else []
        typ = (s["simbad"] or (None, "", ""))[1]
        bad = (f and (f[1].strip("0") or f[2] not in ("", "0"))) or (s["plain"] is not None and s["plain"] < w1w2_min)
        s["vet"] = ("artefact" if bad else "natural-type" if typ not in ("", "*", "PM*", "HV*", "low-mass*") else "blend" if s["near"] else
                    "artefact" if (s["ruwe"] or 0) > 1.4 or (s["multi"] or 0) > 2 else None)
    left = [s for s in st if s["vet"] is None]
    print("%d stars within 100 pc at BP-RP %g-%g fainter than W1 %g with W1-W2 above %g: %d flagged, blended in Gaia or without the colour in AllWISE's plain magnitudes, %d with a Gaia neighbour in the "
          "beam, %d of known types; unexplained %d" % (len(st), colour_min, colour_max, w1_min, w1w2_min, sum(s["vet"] == "artefact" for s in st), sum(s["vet"] == "blend" for s in st),
                                                       sum(s["vet"] == "natural-type" for s in st), len(left)))
    for s in sorted(st, key=lambda s: -(s["w1"] - s["w2"])):
        print("  Gaia DR3 %s: BP-RP %.2f, G %.2f, W1-W2 %+.2f (plain %s), RUWE %s, AllWISE %s, neighbours %s; SIMBAD %s; %s" % (
            s["id"], s["c"], s["g"], s["w1"] - s["w2"], "%+.2f" % s["plain"] if s["plain"] is not None else "-", s["ruwe"], s["flags"], s["near"] or "-", s["simbad"],
            s["vet"] or "unexplained"))
    json.dump(st, open(core.results("methane_in_starlight.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gcns_methane_colour", "Methane in a star's light", "Gaia Catalogue of Nearby Stars with its WISE photometry (external.gaiaedr3_gcns_main_1); AllWISE (II/328) by CDS "
            "XMatch; Gaia DR3; SIMBAD", "stars within 100 pc at BP-RP %g-%g fainter than W1 %g with clean W1 and W2 and W1-W2 above %g: %d" % (colour_min, colour_max, w1_min, w1w2_min, len(st)),
            os.path.abspath(__file__), "methane", len(st), "stars", dict(w1w2_min=w1w2_min, w1_min=w1_min, colour=[colour_min, colour_max], beam_arcsec=beam_arcsec),
            "AllWISE's flags and plain magnitudes, Gaia's image flags and neighbours, SIMBAD's types", "a warm star whose W1-W2 shows a T dwarf's methane band",
            "%d: %d artefacts or blends in the photometry, %d with a neighbour in the beam, %d of known types; %d unexplained" % (
                len(st), sum(s["vet"] == "artefact" for s in st), sum(s["vet"] == "blend" for s in st), sum(s["vet"] == "natural-type" for s in st), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["w1"] - s["w2"], 3), unit="W1-W2", vet=s["vet"],
                  note="BP-RP %.2f, G %.2f, plain W1-W2 %s, RUWE %s, AllWISE %s, neighbours %s; SIMBAD %s" % (
                      s["c"], s["g"], s["plain"], s["ruwe"], s["flags"], s["near"] or "none", s["simbad"])) for s in st])
        print("recorded")


roomheat_DOC = r"""Waste heat at room temperature: Gaia Catalogue of Nearby Stars objects (probability above 0.9, white-dwarf probability below 0.1, RUWE below 1.4, one IPD peak) with 2MASS Ks and WISE W3 and W4 measured to 0.03, 0.1 and 0.2 mag, those whose Ks-W3 and Ks-W4 both stand five robust widths and five errors above the single-star locus at their BP-RP, the colour temperature of the excess from the ratio of its 12 and 22 um fluxes, and the count between 280 and 320 K against the bins of that width on either side; debris discs span a continuum of temperatures where habitats and machinery radiate at the temperature of water, and each star is read in AllWISE for its flags.

With `--sunlike true` the colours are those of Sun-like stars, BP-RP 0.65-1.0, the stars every searcher looks at first, recorded as its own run.

    python -I seti.py infrared roomheat [record] [--widths 5] [--errors 5] [--t-lo 280] [--t-hi 320] [--colour-min 0.5] [--colour-max 2.5] [--sunlike]"""

ROOMHEAT_W3 = (11.56e-6, 31.674)
ROOMHEAT_W4 = (22.09e-6, 8.363)


def roomheat_planck_ratio(t):
    """The ratio of Planck intensities at W4 to W3 at temperature T."""
    h, c, k = 6.62607e-34, 2.99792e8, 1.380649e-23
    def b(lam):
        return (1 / lam ** 3) / math.expm1(h * c / (lam * k * t))
    return b(ROOMHEAT_W4[0]) / b(ROOMHEAT_W3[0])


def roomheat_temperature(ratio):
    """The temperature whose Planck W4/W3 ratio equals RATIO, by bisection between 20 and 3000 K."""
    lo, hi = 20.0, 3000.0
    if ratio >= roomheat_planck_ratio(lo) or ratio <= roomheat_planck_ratio(hi):
        return None
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if roomheat_planck_ratio(mid) > ratio:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def roomheat_main(record=False, widths=5.0, errors=5.0, t_lo=280.0, t_hi=320.0, colour_min=0.5, colour_max=2.5, sunlike=False):
    if sunlike:
        colour_min, colour_max = 0.65, 1.0
    from scipy.stats import poisson
    rows = catalogues.gcns("source_id, ra, dec, phot_bp_mean_mag, phot_rp_mean_mag, k_m_2mass, k_msig_2mass, w3mpro_wise, w3sigmpro_wise, w4mpro_wise, w4sigmpro_wise",
                           "gcns_prob > 0.9 AND wd_prob < 0.1 AND ruwe < 1.4 AND ipd_frac_multi_peak <= 2 AND k_msig_2mass < 0.03 AND w3sigmpro_wise < 0.1 "
                           "AND w4sigmpro_wise < 0.2", ranges=12)
    st = []
    for r in rows:
        bp, rp, k, w3, w4 = num(r["phot_bp_mean_mag"]), num(r["phot_rp_mean_mag"]), num(r["k_m_2mass"]), num(r["w3mpro_wise"]), num(r["w4mpro_wise"])
        if None in (bp, rp, k, w3, w4) or not colour_min <= bp - rp <= colour_max:
            continue
        st.append(dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), c=bp - rp, k=k, w3=w3, w4=w4, ck3=k - w3, ck4=k - w4,
                       ek3=math.hypot(float(r["k_msig_2mass"]), float(r["w3sigmpro_wise"])), ek4=math.hypot(float(r["k_msig_2mass"]), float(r["w4sigmpro_wise"]))))
    bins = np.floor(np.array([s["c"] for s in st]) / 0.1).astype(int)
    for key in ("ck3", "ck4"):
        val = np.array([s[key] for s in st])
        for b in set(bins.tolist()):
            kk = np.nonzero(bins == b)[0]
            med = float(np.median(val[kk])) if len(kk) >= 30 else None
            wid = 1.4826 * float(np.median(np.abs(val[kk] - med))) if med is not None else None
            for i in kk:
                st[i]["r" + key] = (val[i] - med, wid) if med is not None else None
    ex = [s for s in st if s["rck3"] and s["rck4"] and s["rck3"][0] > max(widths * s["rck3"][1], errors * s["ek3"]) and s["rck4"][0] > max(widths * s["rck4"][1], errors * s["ek4"])]
    for s in ex:
        f3 = ROOMHEAT_W3[1] * (10 ** (-0.4 * s["w3"]) - 10 ** (-0.4 * (s["w3"] + s["rck3"][0])))
        f4 = ROOMHEAT_W4[1] * (10 ** (-0.4 * s["w4"]) - 10 ** (-0.4 * (s["w4"] + s["rck4"][0])))
        s["f3_mjy"], s["f4_mjy"] = f3 * 1000, f4 * 1000
        s["t"] = roomheat_temperature(f4 / f3) if f3 > 0 and f4 > 0 else None
    flags = {}
    if ex:
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in ex], "II/328/allwise", 3.0):
            if m["source_id"] not in flags or float(m["angDist"]) < flags[m["source_id"]][0]:
                flags[m["source_id"]] = (float(m["angDist"]), (m.get("ccf") or "").strip(), (m.get("ex") or "").strip())
    clean = []
    for s in ex:
        f = flags.get(s["id"])
        s["flags"] = f
        s["artefact"] = bool(f and (f[1].strip("0") or f[2] not in ("", "0")))
        if s["t"] and not s["artefact"]:
            clean.append(s)
    ts = np.array([s["t"] for s in clean])
    width = t_hi - t_lo
    room = int(np.sum((ts >= t_lo) & (ts < t_hi)))
    beside = [int(np.sum((ts >= t_lo - width) & (ts < t_lo))), int(np.sum((ts >= t_hi) & (ts < t_hi + width)))]
    exp = float(np.mean(beside))
    p = float(poisson.sf(room - 1, exp)) if room > exp else 1.0
    for s in clean:
        s["vet"] = ("chance-rate" if p > 0.01 else None) if t_lo <= s["t"] < t_hi else "dust"
    hist = {int(lo): int(np.sum((ts >= lo) & (ts < lo + 100))) for lo in range(0, 1000, 100)}
    print("%d stars; excess at both 12 and 22 um: %d, %d of them unflagged with a temperature; by hundreds of kelvin: %s; at %g-%g K: %d against %.1f in the bins beside (%s; Poisson "
          "p %.3g)" % (len(st), len(ex), len(clean), hist, t_lo, t_hi, room, exp, beside, p))
    for s in sorted(clean, key=lambda s: s["t"]):
        if t_lo - width <= s["t"] < t_hi + width:
            print("  Gaia DR3 %s: BP-RP %.2f, %.0f K from %.1f and %.1f mJy of excess at 12 and 22 um (Ks-W3 +%.2f, Ks-W4 +%.2f); %s" % (
                s["id"], s["c"], s["t"], s["f3_mjy"], s["f4_mjy"], s["rck3"][0], s["rck4"][0], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), excess=[dict((k, v) for k, v in s.items() if k not in ("rck3", "rck4")) for s in ex], hist=hist), open(core.results("room_temperature_heat.json"), "w"),
              indent=1)
    if record:
        core.record_run(
            "gcns_room_temperature_sunlike" if sunlike else "gcns_room_temperature_excess", "Dyson spheres built for our eyes" if sunlike else "Waste heat at room temperature", "Gaia Catalogue of Nearby Stars with its 2MASS and WISE photometry (external.gaiaedr3_gcns_main_1); "
            "AllWISE (II/328) by CDS XMatch", "stars within 100 pc at BP-RP %g-%g with clean Ks, W3 and W4: %d; with excess at both 12 and 22 um: %d" % (colour_min, colour_max, len(st), len(ex)),
            os.path.abspath(__file__), "roomheat", len(st), "stars", dict(widths=widths, errors=errors, t_lo=t_lo, t_hi=t_hi, colour=[colour_min, colour_max]),
            "the temperature bins of the same width on either side: %s; the excesses by hundreds of kelvin: %s" % (beside, hist),
            "an excess of excess temperatures between %g and %g K" % (t_lo, t_hi), "%d against %.1f (Poisson p %.3g)" % (room, exp, p),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["t"]), unit="K, excess colour temperature", vet=s["vet"],
                  note="BP-RP %.2f, excess %.1f and %.1f mJy at 12 and 22 um, Ks-W3 +%.2f, Ks-W4 +%.2f" % (s["c"], s["f3_mjy"], s["f4_mjy"], s["rck3"][0], s["rck4"][0]))
             for s in clean if t_lo - width <= s["t"] < t_hi + width])
        print("recorded")


clusterswarm_DOC = r"""Warm stars in the nearest clusters: every Hunt and Reffert (2023) cluster or moving group within 250 pc with 100 or more members, each member of probability above 0.5 at BP-RP 0.5-2.5 matched to AllWISE at 2010.5, those whose W1-W3 lies five robust widths and five errors above the cluster's own locus, each cluster's warm share against the field's 0.77 per cent from the same test within 100 pc, a colony that has settled a cluster and warmed its stars; each warm star read in AllWISE for its flags and in SIMBAD, the warm stars of clusters younger than 30 Myr being the disc-bearing young stars such clusters hold, and a cluster's share judged against the number of clusters tested.

    python -I seti.py infrared clusterswarm [record] [--dist-max-pc 250] [--members-min 100] [--prob-min 0.5] [--widths 5] [--errors 5] [--field-share 0.0077] [--w3-err-max 0.2] [--young-logage 7.5]"""


def clusterswarm_main(record=False, dist_max_pc=250.0, members_min=100, prob_min=0.5, widths=5.0, errors=5.0, field_share=0.0077, w3_err_max=0.2, young_logage=7.5):
    from scipy.stats import binom
    cl = core.tap("vizier", 'SELECT Name, Type, N, dist50, logAge50 FROM "J/A+A/673/A114/clusters" WHERE dist50 < %g AND N >= %d AND Type IN (\'o\', \'m\') ORDER BY dist50' % (
        dist_max_pc, members_min))
    out, rows_all = [], []
    for c in cl:
        name = str(c["Name"]).strip()
        st, warm = cluster_warm_stars(name, prob_min, widths, errors, w3_err_max)
        if not st:
            continue
        p = float(binom.sf(len(warm) - 1, len(st), field_share)) if warm else 1.0
        o = dict(name=name, kind=str(c["Type"]).strip(), members=int(num(c["N"]) or 0), dist=num(c["dist50"]), logage=num(c["logAge50"]), n=len(st), warm=len(warm), p=p,
                 share=len(warm) / len(st))
        out.append(o)
        for s in warm:
            if s["vet"] is None and (o["logage"] or 9) < young_logage:
                s["vet"] = "young-star"
            elif s["vet"] is None and p * len(cl) > 0.01:
                s["vet"] = "chance-rate"
            rows_all.append(dict(cluster=name, **dict((k, v) for k, v in s.items() if k != "r"), widths=round(s["r"][0] / s["r"][1], 1)))
    hot = [o for o in out if o["p"] * len(out) < 0.01]
    left = [s for s in rows_all if s["vet"] is None]
    print("%d clusters within %g pc with %d or more members, %d with members tested; warm shares against the field's %.2f per cent, significant over the %d tested: %d; warm stars %d "
          "(%d flagged, %d of known types, %d in clusters under 30 Myr); unexplained %d" % (
              len(cl), dist_max_pc, members_min, len(out), 100 * field_share, len(out), len(hot), len(rows_all), sum(s["vet"] == "artefact" for s in rows_all),
              sum(s["vet"] == "natural-type" for s in rows_all), sum(s["vet"] == "young-star" for s in rows_all), len(left)))
    for o in sorted(out, key=lambda o: o["p"]):
        print("  %s (%s, %.0f pc, log age %s): %d of %d warm, %.2f per cent (binomial p %.3g)%s" % (
            o["name"], o["kind"], o["dist"], o["logage"], o["warm"], o["n"], 100 * o["share"], o["p"], ", beyond chance over the clusters tested" if o in hot else ""))
    for s in left:
        print("    unexplained Gaia DR3 %s in %s: BP-RP %.2f, G %.2f, %.1f widths; SIMBAD %s" % (s["id"], s["cluster"], s["c"], s["g"], s["widths"], s["simbad"]))
    json.dump(dict(clusters=out, warm=rows_all), open(core.results("clusters_warm_stars.json"), "w"), indent=1)
    if record:
        core.record_run(
            "clusters_warm_stars", "Warm stars in the nearest clusters", "Hunt and Reffert (2023) clusters and members (VizieR J/A+A/673/A114); AllWISE (II/328) by CDS XMatch; SIMBAD",
            "clusters within %g pc with %d or more members: %d, %d with tested members" % (dist_max_pc, members_min, len(cl), len(out)), os.path.abspath(__file__), "clusterswarm",
            sum(o["n"] for o in out), "stars", dict(dist_max_pc=dist_max_pc, members_min=members_min, prob_min=prob_min, widths=widths, errors=errors, field_share=field_share,
                                                   w3_err_max=w3_err_max, young_logage=young_logage),
            "the field's share of %.2f per cent, judged over %d clusters" % (100 * field_share, len(out)), "a cluster whose share of stars with 12 um excess exceeds the field's",
            "%d clusters beyond chance; %d warm stars, %d flagged, %d of known types, %d in clusters under 30 Myr, %d unexplained" % (
                len(hot), len(rows_all), sum(s["vet"] == "artefact" for s in rows_all), sum(s["vet"] == "natural-type" for s in rows_all),
                sum(s["vet"] == "young-star" for s in rows_all), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=s["widths"], unit="widths above the cluster's W1-W3 locus", vet=s["vet"],
                  note="%s; BP-RP %.2f, G %.2f, flags %s/%s; SIMBAD %s" % (s["cluster"], s["c"], s["g"], s["ccf"] or "-", s["ex"] or "-", s["simbad"])) for s in rows_all])
        print("recorded")


oldsunlike_DOC = r"""Waste heat at the oldest Sun-like neighbours: GALAH DR4 dwarfs within 200 pc at 5300-6200 K and log g above 4.0 with a clean spectroscopic fit and an isochrone age, matched to AllWISE at 2010.5, those whose W1-W3 lies five robust widths and five errors above the locus at their temperature, the warm share of the oldest tenth against the rest, since the oldest Sun-like stars have had longest to be settled and warmed; each warm old star read in AllWISE for its flags and in SIMBAD.

    python -I seti.py infrared oldsunlike [record] [--plx-min 5] [--teff-min 5300] [--teff-max 6200] [--logg-min 4] [--old-fraction 0.1] [--widths 5] [--errors 5] [--w3-err-max 0.2]"""


def oldsunlike_main(record=False, plx_min=5.0, teff_min=5300.0, teff_max=6200.0, logg_min=4.0, old_fraction=0.1, widths=5.0, errors=5.0, w3_err_max=0.2):
    from scipy.stats import fisher_exact
    rows = core.tap_async("datacentral", "SELECT sobject_id, gaiadr3_source_id, ra, dec, teff, logg, age, parallax FROM galah_dr4.mainstartable WHERE flag_sp = 0 AND parallax > %g AND "
                                         "teff BETWEEN %g AND %g AND logg > %g AND age IS NOT NULL" % (plx_min, teff_min, teff_max, logg_min), maxrec=200000)
    st = {}
    for r in rows:
        gid = str(r["gaiadr3_source_id"] or "").strip()
        if gid and gid not in st:
            st[gid] = dict(id=gid, ra=float(r["ra"]), dec=float(r["dec"]), teff=float(r["teff"]), logg=num(r["logg"]), age=float(r["age"]), plx=num(r["parallax"]))
    st = list(st.values())
    w = {}
    for k in range(0, len(st), 5000):
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in st[k:k + 5000]], "II/328/allwise", 5.0):
            w1, w3, e1, e3 = num(m.get("W1mag")), num(m.get("W3mag")), num(m.get("e_W1mag")), num(m.get("e_W3mag"))
            if None not in (w1, w3, e1, e3) and e3 < w3_err_max and (m["source_id"] not in w or float(m["angDist"]) < w[m["source_id"]]["sep"]):
                w[m["source_id"]] = dict(sep=float(m["angDist"]), c13=w1 - w3, e13=math.hypot(e1, e3), ccf=(m.get("ccf") or "").strip(), ex=(m.get("ex") or "").strip())
    st = [dict(s, **w[s["id"]]) for s in st if s["id"] in w]
    bins = np.floor(np.array([s["teff"] for s in st]) / 100.0).astype(int)
    val = np.array([s["c13"] for s in st])
    for b in set(bins.tolist()):
        k = np.nonzero(bins == b)[0]
        med = float(np.median(val[k])) if len(k) >= 20 else None
        wid = 1.4826 * float(np.median(np.abs(val[k] - med))) if med is not None else None
        for i in k:
            st[i]["r"] = (val[i] - med, wid) if med is not None else None
    ages = sorted(s["age"] for s in st)
    cut = ages[int((1 - old_fraction) * len(ages))] if ages else 0
    old = [s for s in st if s["age"] >= cut]
    young = [s for s in st if s["age"] < cut]
    warm = [s for s in st if s["r"] and s["r"][0] > max(widths * s["r"][1], errors * s["e13"])]
    warm_old = [s for s in warm if s["age"] >= cut]
    sim = {}
    for k in range(0, len(warm_old), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in warm_old[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    a, b = len(warm_old), len(old) - len(warm_old)
    c, d = len(warm) - len(warm_old), len(young) - (len(warm) - len(warm_old))
    p = float(fisher_exact([[a, b], [c, d]], alternative="greater")[1]) if old and young else 1.0
    for s in warm_old:
        s["simbad"] = sim.get(s["id"])
        s["vet"] = ("artefact" if s["ccf"].strip("0") or s["ex"] not in ("", "0") else "natural-type" if s["simbad"] and s["simbad"][1] not in ("*", "PM*", "HV*", "low-mass*") else
                    "chance-rate" if p > 0.01 else None)
    left = [s for s in warm_old if s["vet"] is None]
    print("%d GALAH dwarfs within %.0f pc at %g-%g K with ages and clean W1 and W3; the oldest tenth is older than %.1f Gyr (%d stars); warm: %d of the oldest against %d of the "
          "%d younger (Fisher p %.3g); %d flagged, %d of known types; unexplained %d" % (len(st), 1000 / plx_min, teff_min, teff_max, cut, len(old), a, c, len(young), p,
                                                                                       sum(s["vet"] == "artefact" for s in warm_old), sum(s["vet"] == "natural-type" for s in warm_old),
                                                                                       len(left)))
    for s in sorted(warm_old, key=lambda s: -s["r"][0] / s["r"][1]):
        print("  Gaia DR3 %s: %.1f Gyr, %.0f K, W1-W3 +%.3f (width %.3f, error %.3f), flags %s/%s; SIMBAD %s; %s" % (
            s["id"], s["age"], s["teff"], s["r"][0], s["r"][1], s["e13"], s["ccf"] or "-", s["ex"] or "-", s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), cut=cut, old=len(old), warm_old=[dict((k, v) for k, v in s.items() if k != "r") for s in warm_old], warm_young=c),
              open(core.results("oldest_sunlike_warm.json"), "w"), indent=1)
    if record:
        core.record_run(
            "galah_oldest_sunlike_warm", "Waste heat at the oldest Sun-like neighbours", "GALAH DR4 mainstartable (Data Central); AllWISE (II/328) by CDS XMatch; SIMBAD",
            "GALAH dwarfs within %.0f pc at %g-%g K with ages and clean W1 and W3: %d, the oldest tenth %d" % (1000 / plx_min, teff_min, teff_max, len(st), len(old)),
            os.path.abspath(__file__), "oldsunlike", len(st), "stars", dict(plx_min=plx_min, teff=[teff_min, teff_max], logg_min=logg_min, old_fraction=old_fraction, widths=widths,
                                                                           errors=errors, w3_err_max=w3_err_max),
            "the warm share of the younger nine tenths: %d of %d" % (c, len(young)), "a raised share of 12 um excess among the oldest Sun-like stars",
            "%d warm of the oldest %d against %d of %d younger (Fisher p %.3g); %d flagged, %d of known types, %d unexplained" % (
                a, len(old), c, len(young), p, sum(s["vet"] == "artefact" for s in warm_old), sum(s["vet"] == "natural-type" for s in warm_old), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["r"][0] / s["r"][1], 1), unit="widths above the W1-W3 locus", vet=s["vet"],
                  note="%.1f Gyr, %.0f K, flags %s/%s; SIMBAD %s" % (s["age"], s["teff"], s["ccf"] or "-", s["ex"] or "-", s["simbad"])) for s in warm_old])
        print("recorded")


wisemissing_DOC = r"""Stars WISE could not see: Gaia Catalogue of Nearby Stars objects (probability above 0.9) brighter than G 15 that carry no WISE counterpart in the catalogue's own match, each carried to 2010.5 and sought again in AllWISE (II/328) within 3 arcsec by CDS XMatch; a star within 100 pc shines at W1 brighter than 13 and WISE saw every such star, so a star it did not see was hidden in 2010 or is confused; each unmatched star read for saturation, since WISE's detectors saturate on stars brighter than W1 8.5, estimated from G and colour, and the catalogue drops or displaces them, for a brighter Gaia neighbour within 10 arcsec, whose glare and blending swallow it, for a star 2 mag brighter within an arcminute or brighter than G 4 within three or V 3 within ten, whose halo and diffraction spikes hide it, in SIMBAD, in unWISE (II/363), whose deeper coadds list what AllWISE's extraction dropped, and for the Galactic plane, where confusion thins the catalogue.

    python -I seti.py infrared wisemissing [record] [--g-max 15] [--radius-arcsec 3] [--beam-arcsec 10] [--w1-sat 8.5] [--halo-arcsec 60] [--halo-mag 2] [--unwise-epoch 2015.4] [--glare-arcsec 180] [--glare-g 4] [--plane-deg 10] [--far-glare-arcsec 600] [--far-glare-g 3]"""


def wisemissing_main(record=False, g_max=15.0, radius_arcsec=3.0, beam_arcsec=10.0, w1_sat=8.5, halo_arcsec=60.0, halo_mag=2.0, unwise_epoch=2015.4, glare_arcsec=180.0,
                     glare_g=4.0, plane_deg=10.0, far_glare_arcsec=600.0, far_glare_g=3.0):
    rows = catalogues.gcns("source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe, ipd_frac_multi_peak, w1mpro_pm_wise",
                           "gcns_prob > 0.9 AND phot_g_mean_mag < %g" % g_max, ranges=12)
    n_all = len(rows)
    st = [dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), pmra=num(r["pmra"]) or 0.0, pmdec=num(r["pmdec"]) or 0.0, plx=num(r["parallax"]), g=num(r["phot_g_mean_mag"]),
               c=(num(r["phot_bp_mean_mag"]) - num(r["phot_rp_mean_mag"])) if num(r["phot_bp_mean_mag"]) and num(r["phot_rp_mean_mag"]) else None, ruwe=num(r["ruwe"]),
               multi=num(r["ipd_frac_multi_peak"])) for r in rows if num(r["w1mpro_pm_wise"]) is None]
    up = []
    for s in st:
        ra, de = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, 2010.5)
        up.append(dict(source_id=s["id"], ra=ra % 360, dec=de))
    seen = set()
    for k in range(0, len(up), 5000):
        seen |= {m["source_id"] for m in core.xmatch(up[k:k + 5000], "II/328/allwise", radius_arcsec)}
    gone = [s for s in st if s["id"] not in seen]
    near = {}
    gmag = {s["id"]: s["g"] for s in gone}
    for k in range(0, len(gone), 5000):
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in gone[k:k + 5000]], "I/355/gaiadr3", beam_arcsec):
            other, g = str(m.get("Source") or "").strip(), num(m.get("Gmag"))
            if float(m["angDist"]) > 0.5 and other != m["source_id"] and g is not None and g < gmag[m["source_id"]] + 1:
                near.setdefault(m["source_id"], []).append((other, g))
    halo = {}
    for k in range(0, len(gone), 5000):
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in gone[k:k + 5000]], "I/355/gaiadr3", halo_arcsec):
            other, g = str(m.get("Source") or "").strip(), num(m.get("Gmag"))
            if float(m["angDist"]) > beam_arcsec and other != m["source_id"] and g is not None and g < gmag[m["source_id"]] - halo_mag:
                if m["source_id"] not in halo or g < halo[m["source_id"]][1]:
                    halo[m["source_id"]] = (other, g, round(float(m["angDist"]), 1))
    sim = {}
    for k in range(0, len(gone), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in gone[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    unwise = {}
    later = []
    for s in gone:
        ra, de = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, unwise_epoch)
        later.append(dict(source_id=s["id"], ra=ra % 360, dec=de))
    for k in range(0, len(later), 5000):
        for m in core.xmatch(later[k:k + 5000], "II/363/unwise", radius_arcsec):
            if m["source_id"] not in unwise or float(m["angDist"]) < unwise[m["source_id"]][0]:
                unwise[m["source_id"]] = (round(float(m["angDist"]), 2), num(m.get("FW1")), num(m.get("FW2")))
    glare = {}
    for k in range(0, len(gone), 5000):
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in gone[k:k + 5000]], "I/355/gaiadr3", glare_arcsec):
            other, g = str(m.get("Source") or "").strip(), num(m.get("Gmag"))
            if other != m["source_id"] and g is not None and g < glare_g and (m["source_id"] not in glare or g < glare[m["source_id"]][1]):
                glare[m["source_id"]] = (other, g, round(float(m["angDist"]), 1))
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    lat = SkyCoord(ra=[s["ra"] for s in gone] * u.deg, dec=[s["dec"] for s in gone] * u.deg, frame="icrs").galactic.b.deg if gone else []
    for s, b in zip(gone, lat):
        s["b"] = float(b)
    for s in gone:
        s["near"] = near.get(s["id"], [])
        s["halo"] = halo.get(s["id"]) or glare.get(s["id"])
        s["unwise"] = unwise.get(s["id"])
        if not s["halo"] and not s["near"] and not s["unwise"]:
            # the glare of a star of the first magnitudes reaches ten arcminutes, beyond the cross-match service's radius, and such stars are often absent from Gaia, so SIMBAD's V is read
            for r in core.simbad("SELECT b.main_id, f.V, DISTANCE(POINT('ICRS', b.ra, b.dec), POINT('ICRS', %.6f, %.6f)) * 3600 AS d FROM basic AS b JOIN allfluxes AS f ON "
                                 "f.oidref = b.oid WHERE f.V < %g AND CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', %.6f, %.6f, %.5f)) = 1" % (
                                     s["ra"], s["dec"], far_glare_g, s["ra"], s["dec"], far_glare_arcsec / 3600.0)):
                if float(r["d"]) > beam_arcsec:
                    s["halo"] = (" ".join(str(r["main_id"]).split()), num(r["V"]), round(float(r["d"]), 1))
        s["simbad"] = sim.get(s["id"])
        typ = (s["simbad"] or (None, "", ""))[1]
        s["w1_est"] = s["g"] - (0.5 + 1.3 * (s["c"] if s["c"] is not None else 1.0))
        s["vet"] = ("artefact" if s["w1_est"] < w1_sat else "blend" if s["near"] or (s["multi"] or 0) > 2 else "bright-neighbour" if s["halo"] else
                    "natural-type" if typ in ("**", "SB*", "EB*") else "astrometry" if (s["ruwe"] or 0) > 1.4 else "catalogue-dropped" if s["unwise"] else
                    "crowding" if abs(s["b"]) < plane_deg else None)
    left = [s for s in gone if s["vet"] is None]
    print("%d stars within 100 pc brighter than G %g, %d without a WISE counterpart in the catalogue, %d of them unmatched in AllWISE within %g arcsec at 2010.5 (%d brighter than W1 %g by their "
          "colour, saturated in WISE, %d with a brighter or equal Gaia neighbour within %g arcsec or several image peaks, %d in the halo of a star %g mag brighter within %g arcsec, "
          "%d double by type, %d of poor astrometry, %d present in unWISE's deeper coadds and so dropped by AllWISE's extraction, %d within %g deg of the Galactic plane, where "
          "confusion thins the catalogue); unexplained %d" % (
              n_all, g_max, len(st), len(gone), radius_arcsec, sum(s["vet"] == "artefact" for s in gone), w1_sat, sum(s["vet"] == "blend" for s in gone), beam_arcsec,
              sum(s["vet"] == "bright-neighbour" for s in gone), halo_mag, halo_arcsec, sum(s["vet"] == "natural-type" for s in gone), sum(s["vet"] == "astrometry" for s in gone),
              sum(s["vet"] == "catalogue-dropped" for s in gone), sum(s["vet"] == "crowding" for s in gone), plane_deg, len(left)))
    for s in sorted(gone, key=lambda s: s["g"]):
        if s["vet"] is None or len(gone) <= 40:
            print("  Gaia DR3 %s: G %.2f, BP-RP %s, W1 about %.1f, %.0f pc, RUWE %s, multi %s, neighbours %s, halo %s, unWISE %s; SIMBAD %s; %s" % (
                s["id"], s["g"], s["c"], s["w1_est"], 1000 / s["plx"] if s["plx"] else 0, s["ruwe"], s["multi"], s["near"] or "-", s["halo"] or "-", s["unwise"] or "-", s["simbad"],
                s["vet"] or "unexplained"))
    json.dump(dict(n=n_all, no_wise=len(st), gone=gone), open(core.results("stars_wise_missing.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gcns_wise_missing", "Stars WISE could not see", "Gaia Catalogue of Nearby Stars (external.gaiaedr3_gcns_main_1); AllWISE (II/328) by CDS XMatch; Gaia DR3; SIMBAD",
            "stars within 100 pc brighter than G %g: %d, %d without a WISE counterpart in the catalogue" % (g_max, n_all, len(st)), os.path.abspath(__file__), "wisemissing", n_all, "stars",
            dict(g_max=g_max, radius_arcsec=radius_arcsec, beam_arcsec=beam_arcsec, w1_sat=w1_sat, halo_arcsec=halo_arcsec, halo_mag=halo_mag, unwise_epoch=unwise_epoch,
                 glare_arcsec=glare_arcsec, glare_g=glare_g, plane_deg=plane_deg, far_glare_arcsec=far_glare_arcsec, far_glare_g=far_glare_g),
            "WISE's saturation; Gaia's neighbours within the WISE beam and image peaks; the halos of brighter stars within an arcminute; SIMBAD's types; RUWE; unWISE's coadds",
            "a nearby star with no AllWISE source at its 2010.5 position", "%d unmatched: %d saturated, %d blended, %d in a brighter star's halo, %d double, %d of poor astrometry, "
            "%d in unWISE and so dropped by AllWISE, %d in the Galactic plane's confusion; %d unexplained" % (
                len(gone), sum(s["vet"] == "artefact" for s in gone), sum(s["vet"] == "blend" for s in gone), sum(s["vet"] == "bright-neighbour" for s in gone),
                sum(s["vet"] == "natural-type" for s in gone), sum(s["vet"] == "astrometry" for s in gone), sum(s["vet"] == "catalogue-dropped" for s in gone),
                sum(s["vet"] == "crowding" for s in gone), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=s["g"], unit="G, no AllWISE source", vet=s["vet"],
                  note="BP-RP %s, %.0f pc, RUWE %s, neighbours %s, halo %s, unWISE %s; SIMBAD %s" % (s["c"], 1000 / s["plx"] if s["plx"] else 0, s["ruwe"], s["near"] or "none",
                                                                                              s["halo"] or "none", s["unwise"] or "none", s["simbad"])) for s in gone])
        print("recorded")


gcdust_DOC = r"""Warm dust in globular clusters: the Galactic globular clusters of the Harris catalogue (VizieR VII/202) with a half-light radius, the AllWISE sources within that radius of each centre whose W1-W3 exceeds 1 mag at W3 signal-to-noise 10, warm dust in systems that hold no gas, against the density of such sources in the annulus from three to five half-light radii; a cluster's own asymptotic-giant stars shed dust and give the natural excess, so each excess cluster is read for the warm sources' brightness, which an AGB star at the cluster's distance sets.

    python -I seti.py infrared gcdust [record] [--colour-min 1.0] [--w3-err-max 0.1]"""


def gcdust_main(record=False, colour_min=1.0, w3_err_max=0.1):
    from scipy.stats import poisson
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    t = core.vizier("VII/202")[0]
    out = []
    for r in t:
        rh = core.num(r["Rh"])
        if rh is None or rh <= 0:
            continue
        c = SkyCoord(str(r["RAJ2000"]), str(r["DEJ2000"]), unit=(u.hourangle, u.deg), frame="icrs")
        ra, dec = c.ra.deg, c.dec.deg
        rows = core.tap("vizier", 'SELECT RAJ2000, DEJ2000, W1mag, W3mag, e_W3mag, ccf FROM "II/328/allwise" WHERE 1 = CONTAINS(POINT(\'ICRS\', RAJ2000, DEJ2000), CIRCLE(\'ICRS\', %.5f, %.5f, %.5f)) '
                                  'AND e_W3mag < %g AND W1mag - W3mag > %g' % (ra, dec, 5 * rh / 60.0, w3_err_max, colour_min), timeout=300)
        inside, ring = 0, 0
        w3 = []
        for s in rows:
            if (str(s["ccf"]) or "").strip("0"):
                continue
            d = core.sep_arcsec(ra, dec, float(s["RAJ2000"]), float(s["DEJ2000"])) / 60.0
            if d < rh:
                inside += 1
                w3.append(float(s["W3mag"]))
            elif 3 * rh <= d < 5 * rh:
                ring += 1
        exp = ring / 16.0
        p = float(poisson.sf(inside - 1, exp)) if inside else 1.0
        out.append(dict(name=str(r["ID"]).strip(), other=str(r["Name"]).strip(), rh=rh, dist_kpc=core.num(r["Rsun"]), feh=core.num(r["[Fe/H]"]), inside=inside, ring=ring, expected=exp, p=p,
                        w3_brightest=min(w3) if w3 else None, vet=None))
    hot = [o for o in out if o["p"] * len(out) < 0.01]
    for o in hot:
        # an AGB star of the cluster shines at W3 about 6 + 5 log10(d / 1 kpc) mag, within 3 mag of which the excess is the cluster's own giants
        o["vet"] = "evolved" if o["w3_brightest"] is not None and o["dist_kpc"] and o["w3_brightest"] > 3 + 5 * math.log10(o["dist_kpc"]) else None
    left = [o for o in hot if o["vet"] is None]
    print("%d clusters with a half-light radius; clusters with more warm sources inside it than the annulus gives, beyond chance over the %d: %d (%d with the warm sources at their own giants' "
          "brightness); unexplained %d" % (len(out), len(out), len(hot), sum(o["vet"] == "evolved" for o in hot), len(left)))
    for o in sorted(hot, key=lambda o: o["p"]):
        print("  %s (%s, %.1f kpc, [Fe/H] %s): %d warm sources within %.2f arcmin against %.1f (p %.2g), brightest W3 %.1f; %s" % (
            o["name"], o["other"], o["dist_kpc"] or 0, o["feh"], o["inside"], o["rh"], o["expected"], o["p"], o["w3_brightest"] or 0, o["vet"] or "unexplained"))
    json.dump(out, open(core.results("globular_warm_dust.json"), "w"), indent=1)
    if record:
        core.record_run(
            "allwise_globular_warm_dust", "Warm dust in globular clusters", "Harris (2010) globular clusters (VizieR VII/202); AllWISE (II/328)", "clusters with a half-light radius: %d" % len(out),
            os.path.abspath(__file__), "gcdust", len(out), "clusters", dict(colour_min=colour_min, w3_err_max=w3_err_max),
            "the annulus from three to five half-light radii; the clusters' own AGB stars by brightness", "a cluster with more warm sources inside its half-light radius than its surroundings give",
            "%d clusters in excess over the %d, %d of them at their giants' brightness; %d unexplained" % (len(hot), len(out), sum(o["vet"] == "evolved" for o in hot), len(left)),
            [dict(target="cluster:%s" % o["name"], stat=o["inside"], unit="warm sources within the half-light radius", vet=o["vet"],
                  note="%s, %.1f kpc, against %.1f (p %.2g), brightest W3 %s" % (o["other"], o["dist_kpc"] or 0, o["expected"], o["p"], o["w3_brightest"])) for o in hot])
        print("recorded")


diskvoids_DOC = r"""Debris-disk voids: the WISE excess stars of Cotten and Song (2016, VizieR J/ApJS/225/15) matched to the Gaia DR3 stars within 150 pc brighter than G 11 and bluer than BP-RP 1.5, their Tycho-2 parent population, and every 15 pc sphere about a parent star scanned for fewer excess stars than the colour-binned excess fraction of its members predicts; the deepest deficit's Poisson tail is set against the deepest one that 200 shuffles of the excess labels within colour bins give, so the scan's own trials set the threshold; a civilisation that mines its planetesimal belts over a region would leave its stars of every type without excess.

    python -I seti.py infrared diskvoids [record] [--radius-pc 15] [--shuffles 200] [--min-expected 3]"""


def diskvoids_main(record=False, radius_pc=15.0, shuffles=200, min_expected=3.0):
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.stats import poisson
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = core.tap("gaia", "SELECT source_id, ra, dec, parallax, pmra, pmdec, bp_rp, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE parallax > 6.6667 AND parallax_over_error > 10 "
                         "AND phot_g_mean_mag < 11 AND bp_rp < 1.5")
    ra = np.array([float(r["ra"]) for r in g])
    de = np.array([float(r["dec"]) for r in g])
    plx = np.array([float(r["parallax"]) for r in g])
    c = np.array([float(r["bp_rp"]) for r in g])
    pmra = np.array([float(r["pmra"] or 0) for r in g])
    pmde = np.array([float(r["pmdec"] or 0) for r in g])
    ra0 = ra - 16 * pmra / 3.6e6 / np.cos(np.radians(de))
    de0 = de - 16 * pmde / 3.6e6
    ex = []
    for t in core.vizier("J/ApJS/225/15"):
        if t.meta.get("name", "").split("/")[-1] in ("table3", "table4"):
            sc = SkyCoord([str(x) for x in t["RAJ2000"]], [str(x) for x in t["DEJ2000"]], unit=(u.hourangle, u.deg))
            ex += list(zip(sc.ra.deg, sc.dec.deg))
    tree0 = cKDTree(np.stack([np.cos(np.radians(de0)) * np.cos(np.radians(ra0)), np.cos(np.radians(de0)) * np.sin(np.radians(ra0)), np.sin(np.radians(de0))], 1))
    lab = np.zeros(len(g), bool)
    xe = np.array([[math.cos(math.radians(d)) * math.cos(math.radians(a)), math.cos(math.radians(d)) * math.sin(math.radians(a)), math.sin(math.radians(d))] for a, d in ex])
    dist, idx = tree0.query(xe, distance_upper_bound=2 * math.sin(math.radians(3 / 3600) / 2))
    lab[idx[np.isfinite(dist)]] = True
    d = 1000 / plx
    xyz = np.stack([d * np.cos(np.radians(de)) * np.cos(np.radians(ra)), d * np.cos(np.radians(de)) * np.sin(np.radians(ra)), d * np.sin(np.radians(de))], 1)
    nb = cKDTree(xyz).query_ball_point(xyz, radius_pc)
    rows = np.concatenate([np.full(len(v), i) for i, v in enumerate(nb)])
    cols = np.concatenate([np.array(v) for v in nb])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(g), len(g)))
    bins = np.digitize(c, np.arange(-0.2, 1.6, 0.1))

    def frac(labels):
        f = np.zeros(bins.max() + 1)
        for b in np.unique(bins):
            m = bins == b
            f[b] = labels[m].mean()
        return f[bins]

    def scan(labels):
        obs, exp = A @ labels.astype(float), A @ frac(labels)
        ok = exp >= min_expected
        p = np.ones(len(g))
        p[ok] = poisson.cdf(obs[ok], exp[ok])
        k = int(np.argmin(p))
        return float(p[k]), k, obs[k], exp[k]
    real = scan(lab)
    rng = np.random.default_rng(13)
    null = []
    for _ in range(shuffles):
        sh = lab.copy()
        for b in np.unique(bins):
            m = np.nonzero(bins == b)[0]
            sh[m] = rng.permutation(lab[m])
        null.append(scan(sh)[0])
    null = np.array(null)
    p_glob = float((null <= real[0]).mean())
    k = real[1]
    print("%d parent stars within 150 pc, %d of the %d excess stars matched (%.2f per cent); deepest deficit in a %g pc sphere: %d against %.1f (Poisson %.3g) about Gaia DR3 %s at %.0f pc, "
          "RA %.1f, Dec %.1f; shuffles reach it %.2f of the time (median deepest %.3g)" % (len(g), lab.sum(), len(ex), 100 * lab.mean(), radius_pc, real[2], real[3], real[0], g[k]["source_id"],
                                                                                         d[k], ra[k], de[k], p_glob, float(np.median(null))))
    if record:
        core.record_run(
            "cotten_song_debris_voids", "Debris-disk voids", "Cotten and Song (2016) WISE excess stars (VizieR J/ApJS/225/15); Gaia DR3",
            "Gaia DR3 stars within 150 pc at G below 11 and BP-RP below 1.5: %d, %d of them excess stars" % (len(g), lab.sum()), os.path.abspath(__file__), "diskvoids", len(g), "stars",
            dict(radius_pc=radius_pc, shuffles=shuffles, min_expected=min_expected), "%d shuffles of the excess labels within colour bins: median deepest deficit %.3g" % (shuffles, float(np.median(null))),
            "a %g pc region whose stars lack excess beyond the colour-binned rate" % radius_pc,
            "deepest deficit %d against %.1f (Poisson %.3g) about Gaia DR3 %s at %.0f pc, reached by %.2f of the shuffles" % (real[2], real[3], real[0], g[k]["source_id"], d[k], p_glob), [])
        print("recorded")


coldcomp_DOC = r"""Computation at the cosmic floor: the SPT-SZ survey's sources at 95, 150 and 220 GHz (Everett et al. 2020, 2,500 square degrees), each source's two spectral indices set against the track a body between 2.8 and 10 K draws, seen against the microwave background it hides: a source on that track within two errors, rising from 95 to 150 GHz as synchrotron does not and staying below an index of 2 from 150 to 220 GHz as dust does not, by two errors each, is a candidate, those with no counterpart in SUMSS, AT20G, PMN, IRAS, AKARI, WISE or ROSAT and no extent kept, and the chance of a natural spectrum passing found by drawing each source's best natural model through its own errors; computing lets its heat out as close to the background as it can, since every bit erased costs kT ln 2, and a solar luminosity let out at 4 K peaks near 235 GHz and gives 6.8 Jy at 150 GHz from a kiloparsec, where dust at any temperature keeps rising and synchrotron falls.

    python -I seti.py infrared coldcomp [record] [--t-min 2.8] [--t-max 10] [--sigmas 2] [--dust-alpha 2] [--draws 200]"""

COLDCOMP_NU = np.array([95.0, 150.0, 220.0]) * 1e9
COLDCOMP_TCMB = 2.7255


def coldcomp_planck(nu, t):
    """The shape of the Planck intensity at frequencies nu (Hz) and temperature t (K), without its constants."""
    return nu ** 3 / np.expm1(4.799243e-11 * nu / t)


def coldcomp_shapes(t_min, t_max):
    """The spectral families as arrays of shapes at the three SPT frequencies: the cold body against the background, the power law, dust and the background's own fluctuation."""
    nu = COLDCOMP_NU
    x = 4.799243e-11 * nu / COLDCOMP_TCMB
    return dict(cold=[coldcomp_planck(nu, t) - coldcomp_planck(nu, COLDCOMP_TCMB) for t in np.arange(t_min, t_max + 1e-9, 0.05)],
                temps=np.arange(t_min, t_max + 1e-9, 0.05),
                power=[(nu / 1.5e11) ** a for a in np.arange(-3.0, 6.0, 0.02)],
                dust=[nu ** 1.5 * coldcomp_planck(nu, t) for t in np.arange(5.0, 60.0, 0.25)],
                cmb=[nu ** 4 * np.exp(x) / np.expm1(x) ** 2])


def coldcomp_fit(s, e, shapes):
    """The least chi-square over a family of shapes for each row of fluxes, each shape scaled by its best amplitude: chi-squares, shape indices and amplitudes, one per row."""
    s = np.atleast_2d(s)
    w = 1 / e ** 2
    m = np.array(shapes)
    a = (s * w) @ m.T / ((m * m) @ w)
    c = ((s[:, None, :] - a[:, :, None] * m[None]) ** 2 * w).sum(axis=2)
    k = np.argmin(c, axis=1)
    rows = np.arange(len(s))
    return c[rows, k], k, a[rows, k]


def coldcomp_test(s, e, sh, sigmas=2.0, dust_alpha=2.0):
    """For each row of three fluxes, whether its two spectral indices lie on the cold body's track within the errors while rising from 95 to 150 GHz and staying below dust's index from 150 to 220 GHz, each by SIGMAS errors; with the fits' chi-squares and the cold temperature."""
    s = np.atleast_2d(s)
    l1, l2 = math.log(150 / 95), math.log(220 / 150)
    with np.errstate(divide="ignore", invalid="ignore"):
        a1, a2 = np.log(s[:, 1] / s[:, 0]) / l1, np.log(s[:, 2] / s[:, 1]) / l2
        e1 = np.hypot(e[1] / s[:, 1], e[0] / s[:, 0]) / l1
        e2 = np.hypot(e[2] / s[:, 2], e[1] / s[:, 1]) / l2
    track = np.array(sh["cold"])
    t1, t2 = np.log(track[:, 1] / track[:, 0]) / l1, np.log(track[:, 2] / track[:, 1]) / l2
    d2 = ((a1[:, None] - t1[None]) / e1[:, None]) ** 2 + ((a2[:, None] - t2[None]) / e2[:, None]) ** 2
    near = np.nanmin(d2, axis=1)
    # synchrotron falls or stays flat from 95 to 150 GHz and dust rises with an index near 3 from 150 to 220 GHz
    ok = (s > 0).all(axis=1) & (a1 - sigmas * e1 > 0) & (a2 + sigmas * e2 < dust_alpha) & (near <= sigmas ** 2 * 1.5)
    c_cold, k, _ = coldcomp_fit(s, e, sh["cold"])
    return np.nan_to_num(ok, nan=False).astype(bool), dict(cold=c_cold, t=sh["temps"][k], power=coldcomp_fit(s, e, sh["power"])[0],
                                                             dust=coldcomp_fit(s, e, sh["dust"])[0], cmb=coldcomp_fit(s, e, sh["cmb"])[0], a1=a1, a2=a2, e1=e1, e2=e2)


def coldcomp_main(record=False, t_min=2.8, t_max=10.0, sigmas=2.0, dust_alpha=2.0, draws=200):
    from scipy.stats import chi2
    rows = core.tap("vizier", 'SELECT * FROM "J/ApJ/900/55/table4"')
    sh = coldcomp_shapes(t_min, t_max)
    rng = np.random.default_rng(1)
    src, cands, expected = [], [], 0.0
    for r in rows:
        f = [num(r["S%d-best" % b]) for b in (95, 150, 220)]
        hi = [num(r["B_S%d-best" % b]) for b in (95, 150, 220)]
        lo = [num(r["b_S%d-best" % b]) for b in (95, 150, 220)]
        if None in f + hi + lo:
            continue
        s = np.array(f)
        e = np.maximum((np.array(hi) - np.array(lo)) / 2, 1e-3)
        ok, fits = coldcomp_test(s, e, sh, sigmas, dust_alpha)
        ok, fits = bool(ok[0]), {k: float(v[0]) for k, v in fits.items()}
        # the chance of a natural spectrum passing: the better of the power law and dust drawn through the source's errors
        nat = min((coldcomp_fit(s, e, sh["power"]), "power"), (coldcomp_fit(s, e, sh["dust"]), "dust"), key=lambda z: z[0][0][0])
        model = nat[0][2][0] * np.array(sh[nat[1]][nat[0][1][0]])
        expected += float(np.mean(coldcomp_test(model + rng.normal(0, e, (draws, 3)), e, sh, sigmas, dust_alpha)[0]))
        o = dict(id=r["SPT-S"].strip(), ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"]), s=f, e=e.tolist(), sn=[num(r["SN%d" % b]) for b in (95, 150, 220)],
                 type=r["Type"].strip(), ctpts=r["Ctpts"].strip(), extd=r["Extd"].strip(), cut=r["Cut"].strip(), z=num(r["z"]), fits=fits, cand=ok)
        src.append(o)
        if ok:
            cands.append(o)
    from scipy.stats import poisson
    clean = [o for o in cands if not o["ctpts"] and o["extd"] != "yes"]
    # a set the null fills as often as observed explains its clean members; counterparts, extent and the survey's star list name the rest
    by_chance = poisson.sf(len(cands) - 1, expected) > 0.01 if cands else True
    for o in cands:
        o["vet"] = ("chance-rate" if by_chance else None) if o in clean else "natural-source"
    print("%d SPT-SZ sources with three deboosted fluxes; on the cold body's track, rising from 95 to 150 GHz and below dust's index from 150 to 220 GHz: %d against %.1f by chance (Poisson p %.3g); %d with no counterpart and no extent" % (
        len(src), len(cands), expected, poisson.sf(len(cands) - 1, expected) if cands else 1.0, len(clean)))
    for o in sorted(cands, key=lambda o: o["fits"]["cold"]):
        print("  SPT-S %s  %8.3f %7.3f  %6.1f %6.1f %6.1f mJy (S/N %s)  alpha %.2f+-%.2f, %.2f+-%.2f  T %.2f K  chi2 cold %.2f power %.1f dust %.1f  %s ctpts %s extd %s cut %s z %s" % (
            o["id"], o["ra"], o["dec"], o["s"][0], o["s"][1], o["s"][2], "/".join("%.1f" % x for x in o["sn"]), o["fits"]["a1"], o["fits"]["e1"], o["fits"]["a2"], o["fits"]["e2"],
            o["fits"]["t"], o["fits"]["cold"], o["fits"]["power"], o["fits"]["dust"], o["type"], o["ctpts"] or "-", o["extd"], o["cut"] or "-", o["z"]))
    json.dump(dict(n=len(src), expected=expected, candidates=cands), open(core.results("spt_cold_computing.json"), "w"), indent=1, default=float)
    if record:
        core.record_run(
            "spt_cold_computing", "Computation at the cosmic floor", "SPT-SZ multiband point-source catalogue at 95, 150 and 220 GHz (Everett et al. 2020, VizieR J/ApJ/900/55)",
            "SPT-SZ sources with deboosted fluxes in all three bands: %d over 2,500 square degrees" % len(src), os.path.abspath(__file__), "coldcomp", len(src), "sources",
            dict(t_min=t_min, t_max=t_max, sigmas=sigmas, dust_alpha=dust_alpha, draws=draws),
            "each source's best power law or dust spectrum drawn %d times through its own errors: %.1f expected to pass" % (draws, expected),
            "spectral indices on the track of a 2.8-10 K body within %g errors, rising from 95 to 150 GHz and below %g from 150 to 220 GHz by as much; a solar luminosity let out at 4 K gives 6.8 Jy at 150 GHz from 1 kpc" % (sigmas, dust_alpha),
            "%d pass against %.1f by chance; %d with no counterpart and no extent" % (len(cands), expected, len(clean)),
            [dict(target="spt:%s" % o["id"], stat=round(o["fits"]["t"], 2), unit="K, temperature of the cold body", vet=o["vet"],
                  note="%.1f, %.1f and %.1f mJy at 95, 150 and 220 GHz; indices %.2f+-%.2f and %.2f+-%.2f; SPT type %s; counterparts %s; cut %s" % (
                      o["s"][0], o["s"][1], o["s"][2], o["fits"]["a1"], o["fits"]["e1"], o["fits"]["a2"], o["fits"]["e2"], o["type"], o["ctpts"] or "none", o["cut"] or "none"))
             for o in cands])
        print("recorded")


pvedge_DOC = r"""A silicon photovoltaic edge: a star wrapped in silicon collectors absorbs below 1.1 um and passes the J band, so RP-J rises at fixed BP-RP while J-Ks stays on its locus; Gaia Catalogue of Nearby Stars members at BP-RP 0.5-3.0 with RUWE below 1.4, one image peak and J and Ks measured to 0.03 mag, RP-J more than five locus widths and five errors above the running median at their colour with J-Ks within two widths of its own, against the stars as far below; each read for 2MASS quality and contamination flags, a Gaia neighbour within 4 arcsec and 6 mag inside the 2MASS beam, and the height above the main sequence in M_G at the same colour, where an unresolved twin sits 0.4-0.75 mag above it; RP-J at fixed BP-RP also rises with metallicity, so a photometric excess alone does not separate a collector from a metal-rich star.

    python -I seti.py infrared pvedge [record] [--widths 5] [--jk-widths 2] [--emax 0.03] [--beam-arcsec 4] [--twin-mag 0.4]"""


def pvedge_main(record=False, widths=5.0, jk_widths=2.0, emax=0.03, beam_arcsec=4.0, twin_mag=0.4):
    rows = catalogues.gcns("source_id, ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, j_m_2mass, j_msig_2mass, k_m_2mass, k_msig_2mass, ruwe, ipd_frac_multi_peak, pmra, pmdec",
                           "gcns_prob > 0.9 AND wd_prob < 0.1 AND ruwe < 1.4 AND ipd_frac_multi_peak <= 2 AND j_msig_2mass < %g AND k_msig_2mass < %g AND j_m_2mass > 5" % (emax, emax), ranges=12)
    st = []
    for r in rows:
        bp, rp, g, j, k, plx = (num(r[c]) for c in ("phot_bp_mean_mag", "phot_rp_mean_mag", "phot_g_mean_mag", "j_m_2mass", "k_m_2mass", "parallax"))
        if None in (bp, rp, g, j, k, plx) or not 0.5 <= bp - rp <= 3.0:
            continue
        st.append(dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), c=bp - rp, g=g, mg=g + 5 * math.log10(plx / 100), rpj=rp - j, jk=j - k,
                       e=math.hypot(float(r["j_msig_2mass"]), 0.003), pm=math.hypot(num(r["pmra"]) or 0, num(r["pmdec"]) or 0)))
    bins = np.floor(np.array([s["c"] for s in st]) / 0.05).astype(int)
    for key in ("rpj", "jk", "mg"):
        v = np.array([s[key] for s in st])
        for b in set(bins.tolist()):
            i = np.nonzero(bins == b)[0]
            med = float(np.median(v[i]))
            wid = 1.4826 * float(np.median(np.abs(v[i] - med)))
            for t in i:
                st[t]["r" + key] = (v[t] - med, wid) if len(i) >= 30 else None
    ok = [s for s in st if s["rrpj"] and s["rjk"]]
    up = [s for s in ok if s["rrpj"][0] > max(widths * s["rrpj"][1], widths * s["e"]) and abs(s["rjk"][0]) < jk_widths * s["rjk"][1]]
    down = [s for s in ok if -s["rrpj"][0] > max(widths * s["rrpj"][1], widths * s["e"]) and abs(s["rjk"][0]) < jk_widths * s["rjk"][1]]
    flags = {}
    for k0 in range(0, len(up), 2000):
        for m in core.xmatch([dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in up[k0:k0 + 2000]], "II/246/out", 5.0):
            if m["source_id"] not in flags or float(m["angDist"]) < flags[m["source_id"]][0]:
                flags[m["source_id"]] = (float(m["angDist"]), str(m.get("Qfl") or m.get("Qflg") or "").strip(), str(m.get("Cfl") or m.get("Cflg") or "").strip())
    near = {}
    for k0 in range(0, len(up), 50):
        chunk = up[k0:k0 + 50]
        where = " OR ".join("1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f))" % (s["ra"], s["dec"], beam_arcsec / 3600) for s in chunk)
        for r in core.tap("gaia", "SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE " + where):
            for s in chunk:
                if str(r["source_id"]) != s["id"] and sep_arcsec(float(r["ra"]), float(r["dec"]), s["ra"], s["dec"]) < beam_arcsec \
                        and (num(r["phot_g_mean_mag"]) is None or num(r["phot_g_mean_mag"]) < s["g"] + 6):
                    near.setdefault(s["id"], []).append(num(r["phot_g_mean_mag"]))
    for s in up:
        f = flags.get(s["id"])
        s["tmass"] = f
        s["near"] = near.get(s["id"], [])
        s["mg_off"] = -s["rmg"][0] if s.get("rmg") else None
        s["vet"] = ("blend" if s["near"] else "artefact" if f and (f[1][:1] != "A" or f[1][2:3] != "A" or f[2].strip("0")) else
                    "binary" if s["mg_off"] is not None and s["mg_off"] > twin_mag else None)
    left = [s for s in up if s["vet"] is None]
    print("%d stars at BP-RP 0.5-3.0 with clean J and Ks; RP-J above by %g widths with J-Ks on its locus: %d; as far below: %d; "
          "%d blends, %d with 2MASS flags, %d above the main sequence by %g mag; %d left" % (
              len(ok), widths, len(up), len(down), sum(s["vet"] == "blend" for s in up), sum(s["vet"] == "artefact" for s in up),
              sum(s["vet"] == "binary" for s in up), twin_mag, len(left)))
    for s in sorted(left, key=lambda s: -s["rrpj"][0] / s["rrpj"][1])[:30]:
        print("  Gaia DR3 %s BP-RP %.2f G %.2f RP-J +%.3f (%.1f widths) J-Ks %+.3f, M_G %+.2f from the sequence, pm %.0f mas/yr" % (
            s["id"], s["c"], s["g"], s["rrpj"][0], s["rrpj"][0] / s["rrpj"][1], s["rjk"][0], s["mg_off"] or 0, s["pm"]))
    if record:
        core.record_run(
            "pv_edge_rp_j", "A silicon photovoltaic edge", "Gaia Catalogue of Nearby Stars with its 2MASS photometry; 2MASS PSC (II/246) by CDS XMatch; Gaia DR3",
            "%d single-looking stars at BP-RP 0.5-3.0 with J and Ks to %g mag" % (len(ok), emax), os.path.abspath(__file__), "pvedge", len(ok), "stars",
            dict(widths=widths, jk_widths=jk_widths, emax=emax, beam_arcsec=beam_arcsec, twin_mag=twin_mag), dict(above=len(up), below=len(down)),
            "RP-J %g locus widths and errors above the running median at the star's colour with J-Ks on its locus" % widths,
            "%d above against %d below: %d blends, %d with 2MASS flags, %d twins above the main sequence; %d left, which the metallicity dependence of RP-J at fixed BP-RP "
            "leaves undecided by photometry" % (len(up), len(down), sum(s["vet"] == "blend" for s in up), sum(s["vet"] == "artefact" for s in up),
                                                sum(s["vet"] == "binary" for s in up), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["rrpj"][0] / s["rrpj"][1], 1), unit="RP-J locus widths above the median", vet=s["vet"],
                  note="BP-RP %.2f, G %.2f, RP-J +%.3f, J-Ks %+.3f, M_G %s from the sequence, 2MASS %s, neighbours %s" % (
                      s["c"], s["g"], s["rrpj"][0], s["rjk"][0], "%+.2f" % s["mg_off"] if s["mg_off"] is not None else "-", s["tmass"], s["near"] or "none"))
             for s in up])
        print("recorded")


APPROACHES = {
    "pvedge": (lambda: core.run_cli(pvedge_main), pvedge_DOC),
    "coldcomp": (lambda: core.run_cli(coldcomp_main), coldcomp_DOC),
    "diskvoids": (lambda: core.run_cli(diskvoids_main), diskvoids_DOC),
    "gcdust": (lambda: core.run_cli(gcdust_main), gcdust_DOC),
    "wisemissing": (lambda: core.run_cli(wisemissing_main), wisemissing_DOC),
    "oldsunlike": (lambda: core.run_cli(oldsunlike_main), oldsunlike_DOC),
    "clusterswarm": (lambda: core.run_cli(clusterswarm_main), clusterswarm_DOC),
    "roomheat": (lambda: core.run_cli(roomheat_main), roomheat_DOC),
    "methane": (lambda: core.run_cli(methane_main), methane_DOC),
    "w1only": (lambda: core.run_cli(w1only_main), w1only_DOC),
    "hyadeswarm": (lambda: core.run_cli(hyadeswarm_main), hyadeswarm_DOC),
    "missingdwarfs": (lambda: core.run_cli(missingdwarfs_main), missingdwarfs_DOC),
    "hidark": (lambda: core.run_cli(hidark_main), hidark_DOC),
    "irdim": (lambda: core.run_cli(irdim_main), irdim_DOC),
    "warmgalaxies": (lambda: core.run_cli(warmgalaxies_main), warmgalaxies_DOC),
    "ksexcess": (lambda: core.run_cli(ksexcess_main), ksexcess_DOC),
    "w3only": (lambda: core.run_cli(w3only_main), w3only_DOC),
    "w2only": (lambda: core.run_cli(w2only_main), w2only_DOC),
    "etzwarm": (lambda: core.run_cli(etzwarm_main), etzwarm_DOC),
    "irdeficit": (lambda: core.run_cli(irdeficit_main), irdeficit_DOC),
    "onefilter": (lambda: core.run_cli(onefilter_main), onefilter_DOC),
    "iraswise": (lambda: core.run_cli(iraswise_main), iraswise_DOC),
    "bigdwarfs": (lambda: core.run_cli(bigdwarfs_main), bigdwarfs_DOC),
    "wddust": (wddust_cli, wddust_DOC),
    "bluefloor": (bluefloor_cli, bluefloor_DOC),
    "debs": (debs_cli, debs_DOC),
    "udg": (lambda: core.run_cli(udg_main), udg_DOC),
    "sbf": (lambda: core.run_cli(sbf_main), sbf_DOC),
    "excess": (excess_cli, excess_DOC),
    "hz": (hz_cli, hz_DOC),
    "pma": (pma_cli, pma_DOC),
    "dark": (dark_cli, dark_DOC),
    "enclosed": (enclosed_cli, enclosed_DOC),
    "cold": (cold_cli, cold_DOC),
    "colder": (colder_cli, colder_DOC),
    "massless": (massless_cli, massless_DOC),
    "wdcold": (wdcold_cli, wdcold_DOC),
    "missingflux": (lambda: core.run_cli(missingflux_main), missingflux_DOC),
    "vasco": (vasco_cli, vasco_DOC),
    "spherex": (spherex_cli, spherex_DOC),
    "volume": (volume_cli, volume_DOC),
}


