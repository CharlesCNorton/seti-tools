"""Composition and spectral-modulation anomalies: elements without stable isotopes, periodic modulation, industrial gases."""
import argparse
import csv
import glob
import io
import json
import math
import os
import re
import sys
import urllib.parse
import zlib

import numpy as np

from setitools import core, science
from setitools.core import quiet_warnings


# --------------------------------------------------------------------------------------------------------------

elements_DOC = r"""Tc, Pm, Pa, Pu, Am and Cm lines in stacked HARPS, UVES, FEROS and ESPRESSO spectra of nearby sun-like stars, fitted against twin stacks and random-wavelength nulls.

    python -I seti.py composition elements census
    python -I seti.py composition elements harps_sample
    python -I seti.py composition elements fetch HARPS|UVES|FEROS|ESPRESSO [--max-stars 120]
    python -I seti.py composition elements search HARPS|UVES|FEROS|ESPRESSO [ELEMENT ...] [--inject STAR EW_mA] [--restack]
    python -I seti.py composition elements check HARPS|UVES|FEROS|ESPRESSO
    python -I seti.py composition elements record"""

elements_C = 299792.458
elements_ROOT = core.data("tc_eso")
elements_GRID = np.round(np.arange(4100.0, 5600.0, 0.01), 2)
elements_RES = {"HARPS": 115000.0, "UVES": 45000.0, "FEROS": 48000.0, "ESPRESSO": 140000.0}
elements_SITE_NAME = {"HARPS": "LaSilla", "FEROS": "LaSilla", "UVES": "Paranal", "ESPRESSO": "Paranal"}
elements_BALMER = (4101.73, 4340.46, 4861.33)
elements_NAMES = {"Tc": "technetium", "Pm": "promethium", "Pa": "protactinium", "Pu": "plutonium", "Am": "americium",
                  "Cm": "curium"}
elements_P_LIMIT = 0.001
# (air wavelength A, log gf, lower level cm^-1): Tc I from Palmeri et al. 2005, Pm II from Fivet et al. 2007
elements_LINES = {
    "Pm": [(4137.95, 0.30, 2950.31), (4157.86, 0.36, 1983.52), (4186.03, 0.32, 1133.45), (4297.78, 0.08, 0.00),
           (4325.92, -0.54, 4000.15), (4332.05, -0.58, 2950.31), (4340.53, -0.37, 1983.52), (4342.12, -0.90, 1133.45),
           (4381.88, -0.71, 446.45), (4432.51, -0.70, 1602.97), (4525.20, -0.73, 5017.77), (4615.87, -0.76, 1602.97),
           (4651.93, -0.67, 2666.76), (5546.08, -0.26, 6131.47), (5556.88, -0.03, 9119.34), (5558.39, -0.94, 8041.72),
           (5561.73, -0.04, 7040.66), (5576.02, -0.17, 5332.36)],
    "Tc": [(4238.19, -0.575, 0.0), (4262.27, -0.377, 0.0), (4297.06, -0.237, 0.0)],
}
# short-lived actinides, singly ionised at 5800 K; NIST arc intensities stand in for gf, lower levels taken as ground
elements_LINES.update({
    "Pa": [(4217.23, 0.0, 0.0), (4248.08, 0.0, 0.0), (4176.18, -0.52, 0.0), (4291.34, -0.52, 0.0), (4601.43, -0.52, 0.0)],
    "Pu": [(lam, 0.0, 0.0) for lam in (4105.95, 4129.93, 4141.20, 4159.39, 4178.28, 4189.90, 4190.06, 4196.20, 4229.77,
                                       4254.76, 4273.34, 4289.08, 4337.18, 4352.71, 4379.91, 4385.35, 4393.93, 4441.65,
                                       4468.54, 4472.79, 4493.78, 4504.91, 4536.15)],
    "Am": [(4509.45, -0.30, 0.0), (4575.59, -0.30, 0.0), (4324.57, -0.70, 0.0), (4441.36, -0.70, 0.0),
           (4699.70, -0.70, 0.0), (4872.22, -0.70, 0.0), (4188.12, -1.00, 0.0), (4593.31, -1.00, 0.0),
           (5020.96, -1.00, 0.0)],
    "Cm": [(4207.66, 0.0, 0.0)],
})


def elements_line_set(element, teff=5800.0):
    kt = 0.69504 * teff
    rows = [(lam, (lam / 4500.0) ** 2 * 10 ** gf * np.exp(-e / kt)) for lam, gf, e in elements_LINES[element]]
    rows = [(lam, w) for lam, w in rows if all(abs(lam - b) > 3.0 for b in elements_BALMER)]
    top = max(w for _, w in rows)
    return [(lam, w / top) for lam, w in rows if w >= 0.1 * top]


def elements_census():
    """UVES, FEROS and ESPRESSO 1D spectra of Gaia DR3 sun-like stars within 50 pc, into elements_census.json."""
    rows = core.gaia("SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp, radial_velocity FROM "
                         "gaiadr3.gaia_source WHERE parallax > 20 AND bp_rp BETWEEN 0.62 AND 1.0 AND dec < 30 AND "
                         "phot_g_mean_mag < 10 AND ruwe < 1.4")
    print("%d stars" % len(rows), flush=True)

    def count(r):
        q = ("SELECT obs_collection, COUNT(*) AS n FROM ivoa.ObsCore WHERE obs_collection IN ('UVES','FEROS',"
             "'ESPRESSO') AND dataproduct_type='spectrum' AND CONTAINS(POINT('ICRS',s_ra,s_dec),CIRCLE('ICRS',%f,%f,"
             "0.0083))=1 GROUP BY obs_collection" % (float(r["ra"]), float(r["dec"])))
        try:
            r["eso"] = {x["obs_collection"]: int(x["n"]) for x in core.tap("eso", q)}
        except Exception:
            r["eso"] = None
        return r
    out = core.pmap(count, rows, threads=8)
    json.dump(out, open(core.results("elements_census.json"), "w"))
    for inst in ("UVES", "FEROS", "ESPRESSO"):
        print("%s: %d stars with two or more spectra"
              % (inst, sum(1 for r in out if r["eso"] and r["eso"].get(inst, 0) >= 2)))
    print("%d stars whose count failed" % sum(1 for r in out if r["eso"] is None))


def elements_harps_sample():
    """HARPS sample: Gaia DR3 sun-like stars within 30 pc with five or more HARPS spectra, into harps_sample.json."""
    rows = core.gaia("SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp, radial_velocity FROM "
                         "gaiadr3.gaia_source WHERE parallax > 33.3 AND bp_rp BETWEEN 0.62 AND 1.0 AND dec < 20 AND "
                         "phot_g_mean_mag < 9 AND ruwe < 1.4")

    def count(r):
        q = ("SELECT COUNT(*) AS n FROM ivoa.ObsCore WHERE obs_collection='HARPS' AND dataproduct_type='spectrum' AND "
             "CONTAINS(POINT('ICRS',s_ra,s_dec),CIRCLE('ICRS',%f,%f,0.0083))=1" % (float(r["ra"]), float(r["dec"])))
        try:
            r["harps"] = int(core.tap("eso", q)[0]["n"])
        except Exception:
            r["harps"] = -1
        return r
    rows = [r for r in core.pmap(count, rows, threads=8) if r["harps"] >= 5 and r.get("radial_velocity")]
    refs = {}
    for k in range(0, len(rows), 200):
        ids = ",".join("'Gaia DR3 %s'" % r["source_id"] for r in rows[k:k + 200])
        for x in core.simbad("SELECT ident.id, basic.nbref FROM basic JOIN ident ON ident.oidref = basic.oid "
                                 "WHERE ident.id IN (%s)" % ids):
            refs[x["id"].split()[-1]] = x["nbref"]
    sel = [dict(r, refs=refs[r["source_id"]]) for r in rows
           if refs.get(r["source_id"]) is not None and refs[r["source_id"]] < 200]
    json.dump(sel, open(core.results("harps_sample.json"), "w"))
    print("%d stars" % len(sel))


def elements_fetch(inst, max_stars, threads=4):
    """Highest-S/N spectra covering 4220-4310 A of the nearest qualifying stars (one a night for HARPS)."""
    if inst == "HARPS":
        rows = json.load(open(core.results("harps_sample.json")))
    else:
        rows = [r for r in json.load(open(core.results("elements_census.json")))
                if r["eso"] and r["eso"].get(inst, 0) >= 2 and r.get("radial_velocity")]
        rows = sorted(rows, key=lambda r: -float(r["parallax"]))[:max_stars]

    def one(r):
        d = os.path.join(elements_ROOT, inst, r["source_id"])
        os.makedirs(d, exist_ok=True)
        try:
            found = sorted(core.eso_spectra(float(r["ra"]), float(r["dec"]), inst, covers=(4.22e-7, 4.31e-7)),
                           key=lambda x: -float(x["snr"] or 0))
        except Exception:
            return 0
        if inst == "HARPS":
            nights, first, rest = set(), [], []
            for x in found:
                night = int(float(x["t_min"]) - 0.5)
                (rest if night in nights else first).append(x)
                nights.add(night)
            found = (first + rest)[:6]
        got = 0
        for x in found[:6 if inst == "HARPS" else 4]:
            path = os.path.join(d, x["dp_id"].replace(":", "_") + ".fits")
            try:
                if os.path.exists(path) or core.eso_download(x["dp_id"], path):
                    got += 1
            except Exception:
                continue
        json.dump(dict(rv=float(r["radial_velocity"]), bp_rp=float(r["bp_rp"]), plx=float(r["parallax"]),
                       g=float(r["phot_g_mean_mag"]), ra=float(r["ra"]), dec=float(r["dec"])),
                  open(os.path.join(d, "star.json"), "w"))
        return got
    got = core.pmap(one, rows, threads=threads)
    print("%s: %d stars, %d spectra" % (inst, len(rows), sum(got)), flush=True)


def elements_baryc(inst, ra, dec, mjd):
    """Barycentric correction (km/s) at the instrument's site."""
    return science.barycentric_correction(ra, dec, mjd, elements_SITE_NAME[inst])


def elements_load(inst, path):
    """Flux on GRID in air wavelengths, barycentric frame."""
    from astropy.io import fits
    with fits.open(path) as h:
        H = h[0].header
        t = h[1].data
        names = [n.upper() for n in t.columns.names]
        wname = "WAVE_AIR" if "WAVE_AIR" in names else "WAVE"
        ucd = str(h[1].header.get("TUCD%d" % (names.index(wname) + 1), "obs.atmos"))
        w = np.asarray(t[t.columns.names[names.index(wname)]][0], float)
        f = np.asarray(t[t.columns.names[names.index("FLUX")] if "FLUX" in names else t.columns.names[1]][0], float)
        spec = str(H.get("SPECSYS", h[1].header.get("SPECSYS", ""))).upper()
        if w.max() < 100:
            w = w * 10.0 if w.max() > 10 else w * 1e4
        if "obs.atmos" not in ucd:
            w = science.vac_to_air(w)
        if not spec.startswith("BARY"):
            mjd = float(H.get("MJD-OBS")) + float(H.get("EXPTIME", 0)) / 172800.0
            w = w * (1.0 + elements_baryc(inst, float(H.get("RA")), float(H.get("DEC")), mjd) / elements_C)
    f = np.where(f > 0, f, np.nan)
    fi = np.interp(elements_GRID, w, f, left=np.nan, right=np.nan)
    if inst == "HARPS":
        fi[(elements_GRID > 5290.0) & (elements_GRID < 5350.0)] = np.nan
    return fi


def elements_file_owners(inst):
    """{file: star} for files fetched for two stars of a wide pair, assigned to the star nearest the pointing."""
    from astropy.io import fits
    files = {}
    for p in glob.glob(os.path.join(elements_ROOT, inst, "*", "*.fits")):
        files.setdefault(os.path.basename(p), []).append(p)
    shared = {f: ps for f, ps in files.items() if len(ps) > 1}
    if not shared:
        return {}
    pos = {}
    for s in {os.path.basename(os.path.dirname(p)) for ps in shared.values() for p in ps}:
        meta = json.load(open(os.path.join(elements_ROOT, inst, s, "star.json")))
        if "ra" not in meta:
            meta.update(core.gaia("SELECT ra, dec FROM gaiadr3.gaia_source WHERE source_id = %s" % s)[0])
        pos[s] = (float(meta["ra"]), float(meta["dec"]))
    out = {}
    for f, ps in shared.items():
        h = fits.getheader(ps[0])
        ra, dec = float(h["RA"]), float(h["DEC"])
        stars = [os.path.basename(os.path.dirname(p)) for p in ps]
        out[f] = min(stars, key=lambda s: ((pos[s][0] - ra) * np.cos(np.radians(dec))) ** 2 + (pos[s][1] - dec) ** 2)
    return out


def elements_stacks(inst, restack=False):
    """{star: (rest-frame stack, n spectra, line velocity, Gaia RV)} from envelope-normalised median stacks; noisy stacks dropped."""
    cache = os.path.join(elements_ROOT, "stacks_%s.npz" % inst)
    if os.path.exists(cache) and not restack:
        z = np.load(cache)
        return {str(n): (z["stacks"][i].astype(float), int(z["nspec"][i]), float(z["v"][i]), float(z["rv"][i]))
                for i, n in enumerate(z["names"])}
    owner = elements_file_owners(inst)
    out = {}
    for d in sorted(glob.glob(os.path.join(elements_ROOT, inst, "*"))):
        meta = os.path.join(d, "star.json")
        if not os.path.exists(meta):
            continue
        N = []
        for p in sorted(glob.glob(os.path.join(d, "*.fits"))):
            if owner.get(os.path.basename(p), os.path.basename(d)) != os.path.basename(d):
                continue
            try:
                f = elements_load(inst, p)
            except Exception:
                continue
            if np.isfinite(f).mean() >= 0.15:
                N.append(f / science.envelope(f, block=300, q=92.0, smooth=1))
        if not N:
            continue
        T = np.nanmedian(np.array(N), axis=0)
        seg = T[(elements_GRID > 4150) & (elements_GRID < 4550)]
        if np.nanmedian(seg) <= 0 or 1.4826 * np.nanmedian(np.abs(np.diff(seg))) / np.sqrt(2) / np.nanmedian(seg) > 0.1:
            continue
        v = science.line_velocity(T, elements_GRID)
        if np.isfinite(v):
            out[os.path.basename(d)] = (science.to_rest(T, elements_GRID, v), len(N), v,
                                        float(json.load(open(meta))["rv"]))
    names = sorted(out)
    np.savez_compressed(cache, names=np.array(names), stacks=np.array([out[n][0] for n in names], np.float32),
                        nspec=np.array([out[n][1] for n in names]), v=np.array([out[n][2] for n in names]),
                        rv=np.array([out[n][3] for n in names]))
    return out


def elements_twins(T, names, k=6):
    """Each star's k most correlated stacks and its highest correlation."""
    out = {}
    for n in names:
        corr = np.full(len(names), -2.0)
        for i, m in enumerate(names):
            if m != n:
                ok = np.isfinite(T[n]) & np.isfinite(T[m])
                corr[i] = np.corrcoef(T[n][ok], T[m][ok])[0, 1] if ok.sum() > 5000 else -1.0
        out[n] = ([names[i] for i in np.argsort(corr)[::-1][:k]], float(corr.max()))
    return out


def elements_inject(T, lines, R, ew):
    """Stacks with Gaussian absorption lines added, strongest at ew mA, others by weight."""
    out = T.copy()
    for lam0, w in lines:
        sp = lam0 / R / 2.3548
        out *= 1.0 - w * ew / 1000.0 / (sp * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((elements_GRID - lam0) / sp) ** 2)
    return out


def elements_local_flux(T, lam, R):
    """Stack flux at a line weighted by the instrumental profile over +-3 sigma."""
    sp = lam / R / 2.3548
    m = np.abs(elements_GRID - lam) < 3 * sp
    g = np.exp(-0.5 * ((elements_GRID[m] - lam) / sp) ** 2)
    y = T[m]
    ok = np.isfinite(y)
    return float(np.dot(g[ok], y[ok]) / g[ok].sum()) if ok.mean() > 0.9 else float("nan")


def elements_score(z, ew, local, w, null):
    """Element score, p and the star's limit (mA in the strongest line)."""
    zc, p = science.combined_line_score(z, w, null)
    # an injected line shifts z by its equivalent width times the local stack flux
    ok = np.isfinite(z) & (z != 0) & np.isfinite(local)
    if not np.isfinite(zc) or not ok.any():
        return None
    thr = science.combined_line_threshold(w[ok], null, elements_P_LIMIT)
    gain = np.sum(w[ok] ** 2 * local[ok] * z[ok] / ew[ok]) / np.sqrt(np.sum(w[ok] ** 2))
    return zc, p, int(ok.sum()), max(zc - thr, 0.0) / gain


def elements_search(inst, elements, inject=None, restack=False):
    """Each star's score and limit per element against 400 random-wavelength nulls; lines scattering beyond 1.5 times the null are dropped."""
    st = elements_stacks(inst, restack)
    names = sorted(st)
    T = {n: st[n][0] for n in names}
    print("%s: %d stars with usable stacks" % (inst, len(names)), flush=True)
    twins = elements_twins(T, names)
    sets = {el: elements_line_set(el) for el in elements}
    avoid = [lam for ls in elements_LINES.values() for lam, _, _ in ls] + list(elements_BALMER)
    R = elements_RES[inst]
    meas = []
    for n in ([inject[0]] if inject else names):
        tw, r = twins[n]
        if r < 0.9:
            continue
        B = [T[m] for m in tw]
        cov = elements_GRID[np.isfinite(T[n])]
        lo, hi = cov.min() + 10, cov.max() - 10
        rng = np.random.default_rng(zlib.crc32(n.encode()))
        centres = [c for c in rng.uniform(lo, hi, 2000) if all(abs(c - x) > 1.0 for x in avoid)][:400]
        null = np.array([science.line_absorption(T[n], B, elements_GRID, c, R)[0] for c in centres])
        null = null[np.isfinite(null)]
        row = dict(star=n, r=r, null=null, lines={})
        for el in elements:
            x = (inject[1][el] if isinstance(inject[1], dict) else float(inject[1])) if inject else 0.0
            y = elements_inject(T[n], sets[el], R, x) if inject else T[n]
            m = [science.line_absorption(y, B, elements_GRID, lam, R) if lo <= lam <= hi else (np.nan, np.nan)
                 for lam, _ in sets[el]]
            row["lines"][el] = (np.array([a for a, _ in m]), np.array([b for _, b in m]),
                                np.array([elements_local_flux(T[n], lam, R) for lam, _ in sets[el]]))
        meas.append(row)
    null_sd = float(np.median([np.std(row["null"]) for row in meas])) if meas else np.nan
    stable = {}
    for el in elements:
        if inject:
            stable[el] = np.array(json.load(open(core.results("element_%s_%s.json" % (el, inst))))["stable"])
            continue
        keep = []
        for i in range(len(sets[el])):
            zi = np.array([row["lines"][el][0][i] for row in meas])
            zi = zi[np.isfinite(zi)]
            keep.append(len(zi) < 10 or 1.4826 * np.median(np.abs(zi - np.median(zi))) <= 1.5 * null_sd)
        stable[el] = np.array(keep)
    out = {el: [] for el in elements}
    for row in meas:
        for el in elements:
            z, ew, local = row["lines"][el]
            w = np.array([x for _, x in sets[el]])
            s = elements_score(np.where(stable[el], z, np.nan), ew, local, w, row["null"])
            if s is None:
                continue
            n = row["star"]
            out[el].append(dict(star=n, z=[round(float(x), 2) if np.isfinite(x) else None for x in z],
                                ew_ma=[round(float(x), 2) if np.isfinite(x) else None for x in ew],
                                z_comb=round(s[0], 2), p=s[1], n_lines=s[2], n_spectra=st[n][1], twin_r=round(row["r"], 4),
                                null_sd=round(float(np.std(row["null"])), 2), v=round(st[n][2], 2),
                                rv_gaia=round(st[n][3], 2), ew_limit_ma=round(s[3], 3)))
    if inject:
        return out
    # a low-p star is vetted for absorption shared by three or more stars and for features wider than an unresolved line
    step = elements_GRID[1] - elements_GRID[0]
    for el in elements:
        w = np.array([x for _, x in sets[el]])
        for r in out[el]:
            if r["p"] >= 0.01:
                continue
            i = min((w[j] * r["z"][j], j) for j in range(len(w)) if r["z"][j] is not None and stable[el][j])[1]
            lam = sets[el][i][0]
            r["dominant"], r["shared"] = lam, sum(1 for o in out[el] if o is not r and o["z"][i] is not None
                                                  and o["z"][i] <= 0.5 * r["z"][i])
            res = science.line_absorption(T[r["star"]], [T[m] for m in twins[r["star"]][0]], elements_GRID, lam, R,
                                          residual=True)[2]
            r["width"] = (round(science.width_ratio(-res, len(res) // 2, lam / R / 2.3548 / step), 2)
                          if res is not None else None)
            if r["shared"] >= 3:
                r["vet"] = "absorption at %.2f A shared by %d other stars" % (lam, r["shared"])
            elif r["width"] and r["width"] > 1.6:
                r["vet"] = "absorption at %.2f A %.1f times as wide as an unresolved line" % (lam, r["width"])
    for el in elements:
        res = sorted(out[el], key=lambda x: x["p"])
        json.dump(dict(lines=sets[el], stable=[bool(k) for k in stable[el]], results=res),
                  open(core.results("element_%s_%s.json" % (el, inst)), "w"))
        pv = np.array([x["p"] for x in res])
        print("%s %s: %d stars; %d below p 0.01 (%.1f expected); median limit %.2f mA in the strongest line; lines left "
              "out %s; lowest:" % (el, inst, len(res), int((pv < 0.01).sum()), 0.01 * len(res),
                                   np.median([x["ew_limit_ma"] for x in res]),
                                   [lam for (lam, _), k in zip(sets[el], stable[el]) if not k]), flush=True)
        for x in res[:3]:
            print("  Gaia DR3 %s z %s combined %+.2f p %.5f (%d lines, %d spectra, twin r %.4f)%s"
                  % (x["star"], x["z"], x["z_comb"], x["p"], x["n_lines"], x["n_spectra"], x["twin_r"],
                     "; " + x["vet"] if x.get("vet") else ""))
        for i, (lam, _) in enumerate(sets[el]):
            zi = np.array([x["z"][i] for x in res if x["z"][i] is not None])
            if len(zi):
                print("  line %.2f: %d stars, z median %+.2f, robust sd %.2f (null sd median %.2f)%s"
                      % (lam, len(zi), np.median(zi), 1.4826 * np.median(np.abs(zi - np.median(zi))), null_sd,
                         "" if stable[el][i] else ", left out"))


def elements_check(inst, n=12):
    """Recovery p of lines injected at each star's limit into n stacks, into element_check_<inst>.json."""
    res = {el: {r["star"]: r for r in json.load(open(core.results("element_%s_%s.json" % (el, inst))))["results"]}
           for el in elements_NAMES}
    stars = sorted(set.intersection(*[set(v) for v in res.values()]))
    rows = []
    for star in [stars[i] for i in np.linspace(0, len(stars) - 1, min(n, len(stars))).astype(int)]:
        got = elements_search(inst, list(elements_NAMES), inject=(star, {el: res[el][star]["ew_limit_ma"]
                                                                          for el in elements_NAMES}))
        for el, r in got.items():
            if r:
                rows.append(dict(star=star, element=el, ew_ma=res[el][star]["ew_limit_ma"], p=r[0]["p"],
                                 z_comb=r[0]["z_comb"]))
                print("%s %s %s: %.3f mA injected, p %.5f" % (inst, star, el, rows[-1]["ew_ma"], rows[-1]["p"]),
                      flush=True)
    json.dump(rows, open(core.results("element_check_%s.json" % inst), "w"))
    p = np.array([r["p"] for r in rows])
    print("%s: %d injections at the limit, p median %.4f, 10-90 per cent %.4f-%.4f"
          % (inst, len(p), np.median(p), *np.percentile(p, [10, 90])))


elements_APPROACH = {"Tc": "Technetium", "Pm": "Promethium (Pm II)", "Pa": "Short-lived actinides (Pa II)",
                     "Pu": "Short-lived actinides (Pu II)", "Am": "Short-lived actinides (Am II)",
                     "Cm": "Short-lived actinides (Cm II)"}
elements_LEDGER = {"Tc": "Technetium", "Pm": "Promethium (Pm II)"}


def elements_record():
    """Record each element and instrument run with every star's score and limit."""
    for el, name in elements_NAMES.items():
        for inst in elements_RES:
            path = core.results("element_%s_%s.json" % (el, inst))
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            res = d["results"]
            pv = np.array([r["p"] for r in res])
            lim = np.array([r["ew_limit_ma"] for r in res])
            low = min(res, key=lambda r: r["p"])
            sid = "%s_%s" % (name, inst.lower())
            selection = ("the Gaia DR3 stars within 30 pc with BP-RP 0.62-1.0, G < 9, RUWE < 1.4, declination below "
                         "+20, five or more HARPS spectra, a Gaia radial velocity and fewer than 200 SIMBAD references; "
                         "up to six spectra a star, one a night" if inst == "HARPS" else
                         "the 120 nearest Gaia DR3 stars within 50 pc with BP-RP 0.62-1.0, G < 10, RUWE < 1.4, "
                         "declination below +30, two or more %s spectra and a Gaia radial velocity; up to four "
                         "spectra a star covering 4220-4310 A" % inst)
            core.record_search(
                sid, elements_APPROACH[el], archive="ESO archive, %s" % inst, release="Phase 3 1D spectra",
                selection=selection + "; stacks with a comparison stack correlating above 0.9",
                code=os.path.abspath(__file__), entry="elements", n_targets=len(res),
                params=dict(lines=[lam for lam, _ in d["lines"]], weights=[round(w, 3) for _, w in d["lines"]],
                            twins=6, core_a=0.25, window_a=8.0, resolution=elements_RES[inst],
                            null="400 random wavelengths of the same star", p_limit=elements_P_LIMIT),
                null_summary="%d stars: %d below p 0.01 (%.1f expected), %d of them explained, %d below p 0.1 (%.1f "
                             "expected)" % (len(res), int((pv < 0.01).sum()), 0.01 * len(res),
                                            sum(1 for r in res if r.get("vet")), int((pv < 0.1).sum()),
                                            0.1 * len(res)),
                sensitivity="equivalent width of the strongest line at p = %g: median %.2f mA, 10-90 per cent "
                            "%.2f-%.2f mA" % (elements_P_LIMIT, np.median(lim), *np.percentile(lim, [10, 90])),
                ledger=elements_LEDGER.get(el, "(G) Technetium, promethium and short-lived actinides"),
                result="lowest p %.4f (Gaia DR3 %s) among %d stars; %d below p 0.01 against %.1f expected"
                       % (low["p"], low["star"], len(res), int((pv < 0.01).sum()), 0.01 * len(res)))
            core.clear_targets(sid)
            core.record_targets(sid, [dict(target="gaia_dr3:%s" % r["star"], stat=r["z_comb"], p=r["p"],
                                               upper_limit=r["ew_limit_ma"],
                                               unit="combined line score; limit in mA, equivalent width of the "
                                                    "strongest line at p = %g" % elements_P_LIMIT, vet=r.get("vet"),
                                               note="%d lines, %d spectra" % (r["n_lines"], r["n_spectra"]))
                                          for r in res])
            print("recorded %s: %d stars" % (sid, len(res)))


def elements_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("census", "harps_sample", "fetch", "search", "check", "record"))
    p.add_argument("inst", nargs="?", choices=tuple(elements_RES))
    p.add_argument("elements", nargs="*")
    p.add_argument("--inject", nargs=2, metavar=("STAR", "EW_mA"))
    p.add_argument("--restack", action="store_true")
    p.add_argument("--max-stars", type=int, default=120)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    bad = [e for e in a.elements if e not in elements_LINES]
    if bad:
        p.error("unknown elements %s (choose from %s)" % (bad, ", ".join(elements_LINES)))
    if a.mode == "census":
        elements_census()
    elif a.mode == "harps_sample":
        elements_harps_sample()
    elif a.mode == "fetch":
        elements_fetch(a.inst, a.max_stars, a.threads)
    elif a.mode == "search" and a.inject:
        got = elements_search(a.inst, a.elements or list(elements_NAMES), (a.inject[0], float(a.inject[1])), a.restack)
        for el, r in got.items():
            print("%s %s, %s mA injected into %s: %s" % (el, a.inst, a.inject[1], a.inject[0], r[0] if r else "not measured"))
    elif a.mode == "search":
        elements_search(a.inst, a.elements or list(elements_NAMES), None, a.restack)
    elif a.mode == "check":
        elements_check(a.inst)
    else:
        elements_record()


def elements_cli():
    quiet_warnings()
    elements_main()


# --------------------------------------------------------------------------------------------------------------

borra_DOC = r"""Borra and Trottier (2016) 6.07 THz spectral modulation tested in independent LAMOST and DESI spectra and same-plate SDSS controls.

    python -I seti.py composition borra table|fetch|controls|measure|rayleigh|stack|record [--all-pixels]"""

borra_ROOT = os.environ.get("BORRA_ROOT", core.data("borra"))
borra_HERE = core.RESULTS
borra_PDF = "https://arxiv.org/pdf/1610.03031"
borra_T0 = 1.6465e-13
borra_C_AS = 2.99792458e18
borra_SDSS = "https://data.sdss.org/sas/dr17/sdss/spectro/redux/%s/spectra/lite/%04d/spec-%04d-%05d-%04d.fits"
borra_LAMOST_TAP = "https://www.lamost.org/dr10/v2.0/voservice/tap/sync"
borra_LAMOST_FITS = "https://www.lamost.org/dr10/v2.0/spectrum/fits/%s"


def borra_table():
    import fitz
    os.makedirs(borra_ROOT, exist_ok=True)
    path = os.path.join(borra_ROOT, "borra2016.pdf")
    if not os.path.exists(path):
        open(path, "wb").write(core.get(borra_PDF, timeout=120))
    text = "\n".join(p.get_text() for p in fitz.open(path))
    toks = [t.strip() for t in text[text.index("Stars detected in the SDSS"):].split("\n") if t.strip()]
    # two rows print g or SNR with a decimal comma
    pat = [r"^\d{3,4}$", r"^\d{1,4}$", r"^5\d{4}$", r"^-?\d+\.\d+$", r"^-?\d+\.\d+$", r"^[OBAFGKM]\d(\.\d)?[IVab]*$",
           r"^\d+[.,]\d+$", r"^\d+[.,]\d+$"]
    rows, i = [], 0
    while i <= len(toks) - 8:
        w = toks[i:i + 8]
        if all(re.match(p, x) for p, x in zip(pat, w)):
            rows.append(dict(plate=int(w[0]), fiber=int(w[1]), mjd=int(w[2]), ra=float(w[3]), dec=float(w[4]), sp=w[5],
                             g=float(w[6].replace(",", ".")), snr=float(w[7].replace(",", "."))))
            i += 8
        else:
            i += 1
    json.dump(rows, open(os.path.join(borra_ROOT, "table1.json"), "w"), indent=1)
    print("%d stars read from Table 1" % len(rows))


def borra_lamost_tap(q):
    from astropy.io.votable import parse_single_table
    b = core.get(borra_LAMOST_TAP, data=urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", QUERY=q,
                                                              FORMAT="votable")).encode(), timeout=120)
    t = parse_single_table(io.BytesIO(b)).to_table()
    return [dict(zip(t.colnames, [r[c] for c in t.colnames])) for r in t]


def borra_fetch():
    stars = json.load(open(os.path.join(borra_ROOT, "table1.json")))
    d = os.path.join(borra_ROOT, "spec")
    os.makedirs(d, exist_ok=True)
    for k, s in enumerate(stars):
        tag = "%04d-%05d-%04d" % (s["plate"], s["mjd"], s["fiber"])
        p = os.path.join(d, "sdss_%s.fits" % tag)
        if not os.path.exists(p):
            for run in ("26", "103", "104"):
                b = core.get(borra_SDSS % (run, s["plate"], s["plate"], s["mjd"], s["fiber"]), timeout=120, retries=2, soft=True)
                if b and b.startswith(b"SIMPLE"):
                    open(p, "wb").write(b)
                    break
        s["lamost"] = []
        try:
            rows = borra_lamost_tap("SELECT obsid, snrg, class, subclass FROM public.catalogue WHERE 1 = CONTAINS(POINT('ICRS', "
                              "ra, dec), CIRCLE('ICRS', %.6f, %.6f, 0.000833))" % (s["ra"], s["dec"]))
        except Exception as e:
            print("LAMOST query failed for %s: %s" % (tag, str(e)[:120]), flush=True)
            rows = []
        for r in rows:
            o = str(r["obsid"])
            q = os.path.join(d, "lamost_%s.fits" % o)
            if not os.path.exists(q):
                b = core.get(borra_LAMOST_FITS % o, timeout=120, retries=2, soft=True)
                if b and b[:2] == b"\x1f\x8b":
                    import gzip
                    b = gzip.decompress(b)
                if b and b.startswith(b"SIMPLE"):
                    open(q, "wb").write(b)
            if os.path.exists(q):
                s["lamost"].append(o)
        s["desi"] = []
        try:
            # Data Lab lacks CONTAINS here: box, then 1.5 arcsec radius
            r0 = 1.5 / 3600.0
            dra = r0 / max(np.cos(np.radians(s["dec"])), 1e-3)
            rows = core.datalab("SELECT targetid, mean_fiber_ra, mean_fiber_dec FROM desi_dr1.zpix WHERE mean_fiber_ra "
                                    "BETWEEN %.7f AND %.7f AND mean_fiber_dec BETWEEN %.7f AND %.7f AND zcat_primary = 't'"
                                    % (s["ra"] - dra, s["ra"] + dra, s["dec"] - r0, s["dec"] + r0))
            rows = [r for r in rows if np.hypot((float(r["mean_fiber_ra"]) - s["ra"]) * np.cos(np.radians(s["dec"])),
                                                float(r["mean_fiber_dec"]) - s["dec"]) < r0]
        except Exception as e:
            print("DESI query failed for %s: %s" % (tag, str(e)[:120]), flush=True)
            rows = []
        s["desi"] = sorted({str(r["targetid"]) for r in rows})
        if k % 20 == 0:
            print("%d of %d stars: SDSS %s, LAMOST %d, DESI %d" % (k + 1, len(stars), os.path.exists(p), len(s["lamost"]),
                                                                  len(s["desi"])), flush=True)
    ids = sorted({t for s in stars for t in s["desi"]})
    if ids:
        recs = core.desi_spectra([int(x) for x in ids], include=("targetid", "wavelength", "flux", "ivar"))
        for r in recs:
            np.savez_compressed(os.path.join(d, "desi_%s.npz" % r["targetid"]), wave=np.asarray(r["wavelength"]),
                                flux=np.asarray(r["flux"]), ivar=np.asarray(r["ivar"]))
    json.dump(stars, open(os.path.join(borra_ROOT, "stars.json"), "w"), indent=1)
    print("SDSS %d, LAMOST stars %d (%d spectra), DESI stars %d" % (
        sum(os.path.exists(os.path.join(d, "sdss_%04d-%05d-%04d.fits" % (s["plate"], s["mjd"], s["fiber"]))) for s in stars),
        sum(1 for s in stars if s["lamost"]), sum(len(s["lamost"]) for s in stars), sum(1 for s in stars if s["desi"])))


def borra_read(path):
    """(vacuum wavelength A, flux, ivar) of an SDSS, LAMOST or saved DESI spectrum."""
    if path.endswith(".npz"):
        z = np.load(path)
        return z["wave"], z["flux"], z["ivar"]
    from astropy.io import fits
    with fits.open(path) as h:
        if "COADD" in [x.name for x in h]:
            c = h["COADD"].data
            col = {n.lower(): n for n in c.columns.names}
            if "loglam" in col:
                return (10 ** np.asarray(c[col["loglam"]], float), np.asarray(c[col["flux"]], float),
                        np.asarray(c[col["ivar"]], float))
            # LAMOST DR10 v2.0: one row of vector columns in HDU COADD
            return (np.asarray(c[col["wavelength"]][0], float), np.asarray(c[col["flux"]][0], float),
                    np.asarray(c[col["ivar"]][0], float))
        if h[0].data is not None and np.ndim(h[0].data) == 2:
            a = np.asarray(h[0].data, float)
            return a[2], a[0], a[1]
        c = h[1].data
        return (np.asarray(c["WAVELENGTH"][0], float), np.asarray(c["FLUX"][0], float),
                np.asarray(c["IVAR"][0], float))


def borra_modulation(w, f, iv, t0=borra_T0):
    """Fourier modulus at t0 of the equal-frequency spectrum, in robust deviations over 0.7-1.3 t0."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(f) & np.isfinite(w) & (iv > 0)
    if ok.sum() < 500:
        return None
    w, f = w[ok], f[ok]
    nu = borra_C_AS / w
    o = np.argsort(nu)
    nu, f = nu[o], f[o]
    grid = np.linspace(nu[0], nu[-1], len(nu))
    g = np.interp(grid, nu, f)
    g = g - median_filter(g, size=151, mode="nearest")
    m = np.abs(np.fft.rfft(g - g.mean()))
    t = np.fft.rfftfreq(len(g), grid[1] - grid[0])
    j = int(np.argmin(np.abs(t - t0)))
    sel = (t > 0.7 * t0) & (t < 1.3 * t0) & (np.abs(np.arange(len(t)) - j) > 1)
    if sel.sum() < 10 or j < 2:
        return None
    med = np.median(m[sel])
    sd = 1.4826 * np.median(np.abs(m[sel] - med))
    peak = m[max(j - 1, 0):j + 2].max()
    # null: the same statistic at times offset by 10 per cent
    nulls = []
    for f_ in (0.85, 0.9, 1.1, 1.15):
        jj = int(np.argmin(np.abs(t - f_ * t0)))
        nulls.append(float((m[max(jj - 1, 0):jj + 2].max() - med) / sd))
    return dict(z=float((peak - med) / sd), null=nulls, bin_s=float(t[1]))


def borra_measure():
    stars = json.load(open(os.path.join(borra_ROOT, "stars.json")))
    d = os.path.join(borra_ROOT, "spec")
    out = []
    for s in stars:
        r = dict(s)
        p = os.path.join(d, "sdss_%04d-%05d-%04d.fits" % (s["plate"], s["mjd"], s["fiber"]))
        r["sdss"] = borra_modulation(*borra_read(p)) if os.path.exists(p) else None
        r["lamost_z"] = [m for m in (borra_modulation(*borra_read(os.path.join(d, "lamost_%s.fits" % o))) for o in s["lamost"]) if m]
        r["desi_z"] = [m for m in (borra_modulation(*borra_read(os.path.join(d, "desi_%s.npz" % t))) for t in s["desi"]
                                   if os.path.exists(os.path.join(d, "desi_%s.npz" % t))) if m]
        out.append(r)
    json.dump(out, open(os.path.join(borra_HERE, "borra_modulation.json"), "w"), indent=1)
    for name, key in (("SDSS", "sdss"), ("LAMOST", "lamost_z"), ("DESI", "desi_z")):
        ms = [r[key] for r in out if r[key]] if key == "sdss" else [m for r in out for m in r[key]]
        if not ms:
            print("%s: no spectra" % name)
            continue
        z = np.array([m["z"] for m in ms])
        nz = np.array([x for m in ms for x in m["null"]])
        print("%s: %d spectra; at t the median statistic is %.2f and %d exceed 3; at the offset times %.2f and %.3f of "
              "them exceed 3" % (name, len(z), np.median(z), np.sum(z > 3), np.median(nz), np.mean(nz > 3)))


def borra_controls(per_plate=2):
    """SDSS spectra of other FGK stars on the same plates, g 16-18.6."""
    stars = json.load(open(os.path.join(borra_ROOT, "table1.json")))
    d = os.path.join(borra_ROOT, "spec")
    os.makedirs(d, exist_ok=True)
    mine = {(s["plate"], s["mjd"], s["fiber"]) for s in stars}
    out = []
    for plate, mjd in sorted({(s["plate"], s["mjd"]) for s in stars}):
        q = ("SELECT TOP 12 s.plate, s.mjd, s.fiberid, s.subclass, p.g FROM SpecObj AS s JOIN PhotoObj AS p ON "
             "s.bestobjid = p.objid WHERE s.plate = %d AND s.mjd = %d AND s.class = 'STAR' AND (s.subclass LIKE 'F%%' "
             "OR s.subclass LIKE 'G%%' OR s.subclass LIKE 'K%%') AND p.g BETWEEN 16 AND 18.6" % (plate, mjd))
        try:
            rows = core.sdss_sql(q)
        except Exception:
            continue
        rows = [r for r in rows if (int(r["plate"]), int(r["mjd"]), int(r["fiberid"])) not in mine][:per_plate]
        for r in rows:
            pl, mj, fi = int(r["plate"]), int(r["mjd"]), int(r["fiberid"])
            p = os.path.join(d, "ctrl_%04d-%05d-%04d.fits" % (pl, mj, fi))
            if not os.path.exists(p):
                for run in ("26", "103", "104"):
                    b = core.get(borra_SDSS % (run, pl, pl, mj, fi), timeout=120, retries=2, soft=True)
                    if b and b.startswith(b"SIMPLE"):
                        open(p, "wb").write(b)
                        break
            if os.path.exists(p):
                out.append(dict(plate=pl, mjd=mj, fiber=fi, sp=r["subclass"], g=float(r["g"])))
    json.dump(out, open(os.path.join(borra_ROOT, "controls.json"), "w"), indent=1)
    print("%d control spectra from %d plates" % (len(out), len({(c["plate"], c["mjd"]) for c in out})))


def borra_stack():
    """Mean normalised Fourier modulus over 1.3-2.0e-13 s per spectrum set, and the time of its maximum."""
    from scipy.ndimage import median_filter
    d = os.path.join(borra_ROOT, "spec")
    tt = np.arange(1.30e-13, 2.0e-13, 0.0025e-13)
    sets = {"SDSS, the list": sorted(borra_glob_files(d, "sdss_")), "SDSS, same-plate controls": sorted(borra_glob_files(d, "ctrl_")),
            "LAMOST, the list": sorted(borra_glob_files(d, "lamost_")), "DESI, the list": sorted(borra_glob_files(d, "desi_"))}
    res = {}
    for name, fs in sets.items():
        acc, n = np.zeros(len(tt)), 0
        for f in fs:
            w, fl, iv = borra_read(f)
            ok = np.isfinite(fl) & np.isfinite(w) & (iv > 0)
            if ok.sum() < 500:
                continue
            nu = borra_C_AS / w[ok]
            o = np.argsort(nu)
            nu, g0 = nu[o], fl[ok][o]
            grid = np.linspace(nu[0], nu[-1], len(nu))
            g = np.interp(grid, nu, g0)
            h = g - median_filter(g, size=151, mode="nearest")
            m = np.abs(np.fft.rfft(h - h.mean()))
            t = np.fft.rfftfreq(len(h), grid[1] - grid[0])
            m = m / np.median(m[(t > 0.7e-13) & (t < 3e-13)])
            acc += np.interp(tt, t, m)
            n += 1
        if not n:
            continue
        acc /= n
        j0 = int(np.argmin(np.abs(tt - borra_T0)))
        jm = int(np.argmax(acc))
        res[name] = dict(n=n, t_max=float(tt[jm]), at_max=float(acc[jm]), at_t0=float(acc[j0]), median=float(np.median(acc)))
        print("%s: %d spectra; mean modulus peaks at %.4e s (%.2f), %.2f at %.4e s, median %.2f"
              % (name, n, tt[jm], acc[jm], acc[j0], borra_T0, np.median(acc)))
    json.dump(res, open(os.path.join(borra_HERE, "borra_stack.json"), "w"), indent=1)


def borra_rayleigh(w, f, iv, smooth_hz, t0=borra_T0, offsets=(0.8, 0.9, 1.1, 1.2), all_pixels=False):
    """Borra and Trottier's statistic: modulus at t0 over the Rayleigh sigma below 5.385e-13 s; None for short spectra."""
    from scipy.ndimage import uniform_filter1d
    ok = np.isfinite(f) & np.isfinite(w) & ((iv > 0) | all_pixels)
    if ok.sum() < 500:
        return None
    nu = borra_C_AS / w[ok]
    o = np.argsort(nu)
    # intensity per unit frequency
    nu, g0 = nu[o], (f[ok] * w[ok] ** 2)[o]
    grid = np.linspace(nu[0], nu[-1], len(nu))
    g = np.interp(grid, nu, g0)
    dnu = grid[1] - grid[0]
    h = g - uniform_filter1d(g, size=max(int(round(smooth_hz / dnu)), 3), mode="nearest")
    m = np.abs(np.fft.rfft(h))
    t = np.fft.rfftfreq(len(h), dnu)
    box = (t > 0) & (t < 5.385e-13)
    sigma = np.sqrt(np.sum(m[box] ** 2) / (2.0 * box.sum()))

    def at(tt):
        j = int(np.argmin(np.abs(t - tt)))
        return float(m[max(j - 1, 1):j + 2].max() / sigma)
    return dict(snr=at(t0), null=[at(x * t0) for x in offsets], bin_s=float(t[1]))


def borra_rayleigh_main(all_pixels=False):
    """Borra and Trottier's statistic at t0 in all sets for 1-32 THz smoothing."""
    d = os.path.join(borra_ROOT, "spec")
    sets = {"SDSS, the list": sorted(borra_glob_files(d, "sdss_")), "SDSS, same-plate controls": sorted(borra_glob_files(d, "ctrl_")),
            "LAMOST, the list": sorted(borra_glob_files(d, "lamost_")), "DESI, the list": sorted(borra_glob_files(d, "desi_"))}
    spectra = {name: [borra_read(f) for f in fs] for name, fs in sets.items()}
    res = {}
    for sm in (1e12, 2e12, 4e12, 8e12, 16e12, 32e12):
        for name, sp in spectra.items():
            rs = [r for r in (borra_rayleigh(*x, smooth_hz=sm, all_pixels=all_pixels) for x in sp) if r]
            if not rs:
                continue
            z = np.array([r["snr"] for r in rs])
            nz = np.array([x for r in rs for x in r["null"]])
            res["%s, %g Hz" % (name, sm)] = dict(n=len(z), above5=int(np.sum(z > 5)), median=float(np.median(z)),
                                                 null_above5=float(np.mean(nz > 5)), null_median=float(np.median(nz)))
            print("smoothing %.0f THz, %s: %d spectra, SNR at t0 median %.2f, %d above 5 (%.3f); at offset times "
                  "median %.2f, %.3f above 5" % (sm / 1e12, name, len(z), np.median(z), np.sum(z > 5), np.mean(z > 5),
                                                 np.median(nz), np.mean(nz > 5)), flush=True)
    json.dump(res, open(os.path.join(borra_HERE, "borra_rayleigh%s.json" % ("_allpix" if all_pixels else "")), "w"), indent=1)


def borra_record(smooth_hz=8e12):
    """Record the run and each star's SNR at t0."""
    d = os.path.join(borra_ROOT, "spec")
    stars = json.load(open(os.path.join(borra_ROOT, "stars.json")))
    rows, z_list, z_null = [], [], []
    for s in stars:
        tag = "%04d-%05d-%04d" % (s["plate"], s["mjd"], s["fiber"])
        p = os.path.join(d, "sdss_%s.fits" % tag)
        r = borra_rayleigh(*borra_read(p), smooth_hz=smooth_hz) if os.path.exists(p) else None
        other = [borra_rayleigh(*borra_read(os.path.join(d, "lamost_%s.fits" % o)), smooth_hz=smooth_hz) for o in s["lamost"]]
        other += [borra_rayleigh(*borra_read(os.path.join(d, "desi_%s.npz" % t)), smooth_hz=smooth_hz) for t in s["desi"]
                  if os.path.exists(os.path.join(d, "desi_%s.npz" % t))]
        other = [x for x in other if x]
        if r:
            z_list.append(r["snr"])
            z_null += r["null"]
        rows.append(dict(target="sdss:%s" % tag, stat=r["snr"] if r else None, unit="Fourier modulus at t0 over the "
                         "Rayleigh sigma (Borra and Trottier's SNR)",
                         note="%s; LAMOST %d and DESI %d spectra, highest SNR at t0 %s"
                              % (s["sp"], len(s["lamost"]), len(s["desi"]),
                                 "%.2f" % max(x["snr"] for x in other) if other else "none")))
    ctrl = [borra_rayleigh(*borra_read(f), smooth_hz=smooth_hz) for f in sorted(borra_glob_files(d, "ctrl_"))]
    zc = [x["snr"] for x in ctrl if x]
    lam = [borra_rayleigh(*borra_read(f), smooth_hz=smooth_hz) for f in sorted(borra_glob_files(d, "lamost_"))]
    des = [borra_rayleigh(*borra_read(f), smooth_hz=smooth_hz) for f in sorted(borra_glob_files(d, "desi_"))]
    stk = json.load(open(os.path.join(borra_HERE, "borra_stack.json")))
    core.record_search(
        "borra_modulation", "Periodic spectral modulation", archive="SDSS DR17 lite spectra, LAMOST DR10 v2.0 "
        "low-resolution spectra, DESI DR1 (SPARCL)", release="SDSS DR17, LAMOST DR10 v2.0, DESI DR1",
        selection="Borra and Trottier 2016 Table 1 (228 rows); LAMOST within 3 arcsec, DESI within 1.5 arcsec; two F, G "
                  "or K stars of g 16-18.6 from each plate as controls",
        code=os.path.abspath(__file__), entry="borra", n_targets=len(stars),
        params=dict(t0_s=borra_T0, smoothing_hz=smooth_hz, rayleigh_box_s=5.385e-13, snr_threshold=5.0,
                    smoothing_tried_hz=[1e12, 2e12, 4e12, 8e12, 16e12, 32e12]),
        null_summary="SNR at 0.8, 0.9, 1.1 and 1.2 t0: median %.2f, %.3f above 5; same-plate controls at t0: median "
                     "%.2f, %d of %d above 5" % (np.median(z_null), np.mean(np.array(z_null) > 5), np.median(zc),
                                                 int(np.sum(np.array(zc) > 5)), len(zc)),
        sensitivity="the paper's detection threshold, SNR 5, in the same SDSS spectra",
        ledger="Periodic spectral modulation",
        result="none of the %d SDSS spectra of the list reaches SNR 5 at t0 (median %.2f; controls %.2f), with a boxcar "
               "continuum of 1 to 32 THz; their LAMOST (%d) and DESI (%d) spectra show nothing at t0; the mean modulus of "
               "every set peaks at %.2e s" % (len(z_list), np.median(z_list), np.median(zc), sum(1 for x in lam if x),
                                              sum(1 for x in des if x), stk["SDSS, the list"]["t_max"]))
    core.clear_targets("borra_modulation")
    core.record_targets("borra_modulation", rows)
    print("recorded %d stars" % len(rows))


def borra_glob_files(d, prefix):
    return [os.path.join(d, f) for f in os.listdir(d) if f.startswith(prefix)]


def borra_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("table", "fetch", "controls", "measure", "stack", "rayleigh", "record"))
    p.add_argument("--all-pixels", action="store_true")
    a = p.parse_args()
    if a.mode == "rayleigh":
        borra_rayleigh_main(a.all_pixels)
    else:
        {"table": borra_table, "fetch": borra_fetch, "controls": borra_controls, "measure": borra_measure, "stack": borra_stack,
         "record": borra_record}[a.mode]()


def borra_cli():
    quiet_warnings()
    borra_main()


# --------------------------------------------------------------------------------------------------------------

gases_DOC = r"""SF6, NF3, CF4, CFC-11 and CFC-12 band depths in K2-18 b's JWST MIRI transmission spectrum (Luque et al. 2025), in scale heights.

    python -I seti.py composition gases"""

gases_ROOT = os.environ.get("K218B_ROOT", core.data("k218b"))
gases_HERE = core.RESULTS
gases_FILES = ("SPARTA_K2-18b_MIRI_LRS_R70.csv", "exoTEDRF_K2-18b_MIRI_LRS_R70.csv", "Eureka_K2-18b_MIRI_LRS_R70.csv")
gases_GASES = {"SF6": (10.55,), "NF3": (9.69, 11.03), "CF4": (7.82,), "CFC-11": (9.22, 11.81), "CFC-12": (8.68, 9.13, 10.85)}
gases_FWHM = 0.15
gases_PPM_PER_H = 34.0


def gases_load(name):
    rows = list(csv.DictReader(open(os.path.join(gases_ROOT, name))))
    w = np.array([float(r["wavelength"]) for r in rows])
    hw = np.array([float(r["bin_width"]) for r in rows])
    d = np.array([float(r["dppm"]) for r in rows])
    if "dppm_err" in rows[0]:
        e = np.array([float(r["dppm_err"]) for r in rows])
    else:
        e = 0.5 * (np.array([float(r["dppm_err_neg"]) for r in rows]) + np.array([float(r["dppm_err_pos"]) for r in rows]))
    return w, hw, d, e


def gases_band(w, hw, centres):
    """Gaussian bands averaged over each bin."""
    from scipy.special import erf
    s = gases_FWHM / 2.355
    out = np.zeros(len(w))
    for c in centres:
        lo, hi = (w - hw - c) / (np.sqrt(2) * s), (w + hw - c) / (np.sqrt(2) * s)
        out += 0.5 * (erf(hi) - erf(lo)) * s * np.sqrt(2 * np.pi) / (2 * hw)
    return out


def gases_main():
    res = []
    for f in gases_FILES:
        w, hw, d, e = gases_load(f)
        x = (w - 8.5) / 3.5
        for gas, cen in gases_GASES.items():
            if not any(w.min() <= c <= w.max() for c in cen):
                continue
            A = np.vstack([np.ones_like(x), x, x ** 2, gases_band(w, hw, cen)]).T
            W = 1 / e ** 2
            cov = np.linalg.inv(A.T @ (A * W[:, None]))
            p = cov @ (A.T @ (W * d))
            chi = float(np.sum(W * (d - A @ p) ** 2))
            res.append(dict(reduction=f.split("_")[0], gas=gas, depth=float(p[3]), err=float(np.sqrt(cov[3, 3])),
                            chi2=chi, dof=len(d) - 4))
    print("band depth in ppm (positive is absorption) and 3-sigma upper limit in scale heights:")
    for gas in gases_GASES:
        rs = [r for r in res if r["gas"] == gas]
        if not rs:
            continue
        line = "; ".join("%s %+.0f +- %.0f" % (r["reduction"], r["depth"], r["err"]) for r in rs)
        lim = max(max(r["depth"], 0) + 3 * r["err"] for r in rs) / gases_PPM_PER_H
        print("  %-7s %s; at most %.1f scale heights" % (gas, line, lim))
    json.dump(res, open(os.path.join(gases_HERE, "k218b_gases.json"), "w"), indent=1)


def gases_cli():
    gases_main()


# --------------------------------------------------------------------------------------------------------------

fission_DOC = r"""Fission-yield humps in GALAH DR4 abundances: Mo, Ru over Zr and Nd, Sm over Ba, La with Eu flat, against neighbours; the mirror pattern as null.

    python -I seti.py composition fission [record] [--neighbours 200] [--snr-min 40]

The cached table is fetched at S/N 40, so a lower floor needs the cache removed."""

fission_ELS = ("sr", "y", "zr", "mo", "ru", "ba", "la", "ce", "nd", "sm", "eu")


def fission_main(record=False, neighbours=200, snr_min=40.0):
    from scipy.spatial import cKDTree
    p = core.data("galah_dr4_ncapture.json")
    if not os.path.exists(p):
        cols = ", ".join(["sobject_id", "gaiadr3_source_id", "teff", "logg", "fe_h", "snr_px_ccd3"] +
                         ["%s_fe, e_%s_fe, flag_%s_fe" % (e, e, e) for e in fission_ELS])
        rows = core.tap_async("datacentral", "SELECT %s FROM galah_dr4.mainstartable WHERE flag_sp = 0 AND snr_px_ccd3 > 40 AND "
                             "flag_mo_fe = 0 AND flag_ru_fe = 0 AND flag_zr_fe = 0 AND flag_nd_fe = 0 AND flag_ba_fe = 0 AND "
                             "flag_eu_fe = 0" % cols)
        json.dump(rows, open(p, "w"))
    rows = json.load(open(p))

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan
    rows = [r for r in rows if f(r["snr_px_ccd3"]) > snr_min]
    X = np.array([[f(r[e + "_fe"]) for e in fission_ELS] for r in rows])
    F = np.array([[f(r["flag_" + e + "_fe"]) for e in fission_ELS] for r in rows])
    P = np.array([[f(r["teff"]) / 1000.0, f(r["logg"]), 2 * f(r["fe_h"])] for r in rows])
    X[(F != 0) | ~np.isfinite(X)] = np.nan
    _, nb = cKDTree(P).query(P, k=neighbours + 1)
    D = X - np.array([np.nanmedian(X[nb[i][1:]], axis=0) for i in range(len(rows))])
    ix = {e: j for j, e in enumerate(fission_ELS)}

    def mean(es):
        return np.nanmean(np.vstack([D[:, ix[e]] for e in es]), axis=0)
    A = mean(["mo", "ru"]) - D[:, ix["zr"]]
    B = mean(["nd", "sm"]) - mean(["ba", "la"])
    C = D[:, ix["eu"]] - mean(["nd", "sm"])

    def rob(v):
        w = v[np.isfinite(v)]
        med = np.median(w)
        s = 1.4826 * np.median(np.abs(w - med))
        return (v - med) / s, s
    (zA, sA), (zB, sB), (zC, sC) = rob(A), rob(B), rob(C)
    Z, Zm = (zA + zB - zC) / np.sqrt(3), (-zA - zB + zC) / np.sqrt(3)
    good = np.isfinite(Z)
    counts = {t: (int(np.sum(Z[good] > t)), int(np.sum(Zm[good] > t))) for t in (3, 4, 5, 6)}
    teff, logg = P[:, 0] * 1000, P[:, 1]
    dwarf = (teff > 4800) & (teff < 6500) & (logg > 3.8)
    print("%d stars (%d FGK dwarfs); spreads A %.3f, B %.3f, C %.3f dex" % (good.sum(), dwarf.sum(), sA, sB, sC))
    for t, (a, b) in counts.items():
        print("  combined z above %d: fission-like %d, mirror %d" % (t, a, b))
    order = np.argsort(-np.where(good, Z, -99))
    for i in order[:10]:
        r = rows[i]
        print("  %s Gaia DR3 %s: z %.1f (A %+.1f, B %+.1f, C %+.1f), Teff %.0f, logg %.2f" % (r["sobject_id"], r["gaiadr3_source_id"],
                                                                                         Z[i], zA[i], zB[i], zC[i], teff[i], logg[i]))
    if not record:
        return
    core.record_search(
        "fission_galah", "Fission-product abundances", archive="GALAH DR4 mainstartable (Data Central)",
        selection="stars with clean Zr, Mo, Ru, Ba, Nd and Eu at S/N above %g" % snr_min, code=os.path.abspath(__file__), entry="fission",
        n_targets=int(good.sum()), params=dict(neighbours=neighbours, snr_min=snr_min),
        null_summary="the mirror pattern: %s" % ", ".join("z>%d %d" % (t, b) for t, (a, b) in counts.items()),
        sensitivity="0.1 dex in the Mo/Ru and Nd/Sm contrasts; diluted by the deep envelopes of giants",
        ledger="Fission-product abundances",
        result="fission-like patterns rarer than the mirror at every threshold (z above 5: %d against %d); the strongest are "
               "single-element outliers in cool giants and M dwarfs; %d FGK dwarfs carry Mo and Ru" % (counts[5][0], counts[5][1], dwarf.sum()))
    core.clear_targets("fission_galah")

    def target(r):
        # GALAH DR4 gives -1 for a star without a Gaia DR3 source, which is keyed by its GALAH identifier instead
        g = str(r["gaiadr3_source_id"] or "").strip()
        return "gaia_dr3:" + g if g.isdigit() and int(g) > 0 else "galah:%s" % r["sobject_id"]
    core.record_targets("fission_galah", [dict(target=target(rows[i]), stat=float(Z[i]), unit="combined fission-pattern z",
                                                   note="Teff %.0f logg %.2f" % (teff[i], logg[i])) for i in np.nonzero(good)[0]])
    print("recorded")


cno_DOC = r"""CN-processed surfaces on unevolved metal-rich APOGEE dwarfs: [C/N] 5 sigma below the Teff-[Fe/H] locus, vetted for transfer, cluster origin, binarity and the Teff edge.

    python -I seti.py composition cno [record] [--sigma 5] [--ce-max 0.3] [--na-max 0.3] [--rv-scatter-max 1] [--teff-edge 4900] [--vsini-max 15] [--n-control 600]"""

cno_TAP = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def cno_fetch():
    p = core.data("apogee_dr17_dwarfs_cn_k.csv")
    if not os.path.exists(p):
        q = ('SELECT "APOGEE", "RAJ2000", "DEJ2000", "Teff", "logg", "[Fe/H]", "[C/Fe]", "e_[C/Fe]", "f_[C/Fe]", "[N/Fe]", '
             '"e_[N/Fe]", "f_[N/Fe]", "[Na/Fe]", "f_[Na/Fe]", "[Ce/Fe]", "f_[Ce/Fe]", "s_HRV", "Nvis", "SNR", "Vsini", '
             '"GaiaEDR3", "plx", "Ksmag", "AFlag" FROM "III/286/catalog" WHERE "logg" > 4.2 AND "Teff" > 4800 AND "Teff" < 6500 '
             'AND "[Fe/H]" > -0.3 AND "SNR" > 70 AND "[C/Fe]" IS NOT NULL AND "[N/Fe]" IS NOT NULL')
        u = cno_TAP + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q, MAXREC=2000000))
        b = core.get(u, timeout=1800)
        open(p, "wb").write(b if isinstance(b, bytes) else b.encode("utf-8"))
    return list(csv.DictReader(open(p, encoding="utf-8")))


def cno_gaia_binary(ids):
    """{source_id: (binary, note)} from Gaia DR3 RUWE, non_single_star and RV chi-square p."""
    from astroquery.gaia import Gaia
    out = {}
    ids = [i for i in dict.fromkeys(ids) if i and i.strip().isdigit()]
    for k in range(0, len(ids), 400):
        q = ("SELECT source_id, ruwe, non_single_star, rv_chisq_pvalue FROM gaiadr3.gaia_source WHERE source_id IN (%s)"
             % ",".join(ids[k:k + 400]))
        for r in Gaia.launch_job(q).get_results():
            ruwe = float(r["ruwe"]) if r["ruwe"] is not np.ma.masked else None
            nss = int(r["non_single_star"]) if r["non_single_star"] is not np.ma.masked else 0
            pv = float(r["rv_chisq_pvalue"]) if r["rv_chisq_pvalue"] is not np.ma.masked else None
            flag = (ruwe is not None and ruwe > 1.4) or nss > 0 or (pv is not None and pv < 0.01)
            out[str(r["source_id"])] = (bool(flag), "RUWE %s, NSS %d%s" % ("%.2f" % ruwe if ruwe is not None else "-", nss,
                                                                          ", RV p %.1e" % pv if pv is not None and pv < 0.01 else ""))
    return out


def cno_mk_ms(teff):
    """Main-sequence M_K at solar metallicity (Pecaut and Mamajek 2013) at Teff."""
    t = [4800, 5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600]
    m = [4.20, 3.95, 3.70, 3.45, 3.25, 3.05, 2.85, 2.65, 2.50, 2.35]
    return float(np.interp(teff, t, m))


def cno_main(record=False, sigma=5.0, ce_max=0.3, na_max=0.3, rv_scatter_max=1.0, teff_edge=4900.0, vsini_max=15.0, n_control=600):
    from setitools.core import num
    rows = cno_fetch()
    keep = []
    for r in rows:
        c, n, fe, te = num(r["[C/Fe]"]), num(r["[N/Fe]"]), num(r["[Fe/H]"]), num(r["Teff"])
        if None in (c, n, fe, te) or (r["f_[C/Fe]"] or "0").strip() not in ("0", "") or (r["f_[N/Fe]"] or "0").strip() not in ("0", ""):
            continue
        keep.append(dict(id=r["APOGEE"], ra=num(r["RAJ2000"]), dec=num(r["DEJ2000"]), teff=te, logg=num(r["logg"]), feh=fe,
                         cn=c - n, c=c, n=n, e_c=num(r["e_[C/Fe]"]) or 0.05, e_n=num(r["e_[N/Fe]"]) or 0.05,
                         na=num(r["[Na/Fe]"]), ce=num(r["[Ce/Fe]"]), srv=num(r["s_HRV"]), nvis=num(r["Nvis"]),
                         vsini=num(r["Vsini"]), gaia=r["GaiaEDR3"], plx=num(r["plx"]), ks=num(r["Ksmag"])))
    T = np.array([k["teff"] for k in keep])
    F = np.array([k["feh"] for k in keep])
    Y = np.array([k["cn"] for k in keep])
    # locus: running median in 100 K Teff and 0.1 dex [Fe/H] cells, smoothed
    ti = np.clip(((T - 4800) // 100).astype(int), 0, 16)
    fi = np.clip(((F + 0.3) // 0.1).astype(int), 0, 8)
    med = np.full((17, 9), np.nan)
    for a in range(17):
        for b in range(9):
            sel = (np.abs(ti - a) <= 1) & (np.abs(fi - b) <= 1)
            if sel.sum() >= 30:
                med[a, b] = np.median(Y[sel])
    loc = med[ti, fi]
    res = Y - loc
    ok = np.isfinite(res)
    sig = 1.4826 * np.median(np.abs(res[ok] - np.median(res[ok])))
    z = res / sig
    print("%d APOGEE DR17 dwarfs fetched, %d with unflagged C and N; [C/N] locus scatter %.3f dex" % (len(rows), len(keep), sig))
    low = [i for i in np.argsort(z) if ok[i] and z[i] < -sigma]
    high = [i for i in np.argsort(-z) if ok[i] and z[i] > sigma]
    print("below the locus at %g sigma (nitrogen-rich): %d; above it (carbon-rich, the mirror): %d" % (sigma, len(low), len(high)))
    out = []
    for i in low:
        k = keep[i]
        why = []
        if k["ce"] is not None and k["ce"] > ce_max:
            why.append("s-process [Ce/Fe] %+.2f" % k["ce"])
        if k["na"] is not None and k["na"] > na_max:
            why.append("[Na/Fe] %+.2f" % k["na"])
        if k["srv"] is not None and k["nvis"] and k["nvis"] > 1 and k["srv"] > rv_scatter_max:
            why.append("RV scatter %.1f km/s" % k["srv"])
        if k["teff"] < teff_edge:
            why.append("Teff edge")
        if k["vsini"] and k["vsini"] > vsini_max:
            why.append("vsini %.0f" % k["vsini"])
        # dwarfs lie within 1 mag of the main sequence in M_K
        mk = k["ks"] + 5 * np.log10(k["plx"] / 100.0) if k["plx"] and k["plx"] > 0 and k["ks"] is not None else None
        k["mk"] = float(mk) if mk is not None else None
        ms = cno_mk_ms(k["teff"])
        if mk is not None and mk < ms - 1.0:
            why.append("M_K %.1f, %.1f mag above the main sequence" % (mk, ms - mk))
        out.append(dict(k, z=float(z[i]), locus=float(loc[i]), vet="; ".join(why) or None))
    # candidate binarity against a random draw of the same dwarfs
    rng = np.random.default_rng(3)
    ctrl = [keep[i] for i in rng.choice(len(keep), size=min(n_control, len(keep)), replace=False)]
    gb = cno_gaia_binary([o["gaia"] for o in out] + [k["gaia"] for k in ctrl])
    base = np.mean([gb.get(k["gaia"]) is not None and gb[k["gaia"]][0] for k in ctrl if gb.get(k["gaia"]) is not None])
    for o in out:
        b = gb.get(o["gaia"])
        if b is not None and b[0]:
            o["vet"] = "; ".join(x for x in (o["vet"], "Gaia binary (%s)" % b[1]) if x)
    print("Gaia DR3 binarity (RUWE above 1.4, a non-single-star solution or RV variability at p 0.01): candidates %d of %d, "
          "random dwarfs %.0f per cent" % (sum(1 for o in out if gb.get(o["gaia"]) and gb[o["gaia"]][0]), len(out), 100 * base))
    for o in out[:40]:
        print("  %s Teff %4.0f logg %.2f [Fe/H] %+.2f [C/N] %+.2f (locus %+.2f, %.1f sigma) [C/Fe] %+.2f [N/Fe] %+.2f; %s"
              % (o["id"], o["teff"], o["logg"], o["feh"], o["cn"], o["locus"], o["z"], o["c"], o["n"], o["vet"] or "no natural flag"))
    json.dump(dict(n=len(keep), sigma=float(sig), low=out, n_high=len(high)), open(core.results("cno_dwarfs.json"), "w"), indent=1)
    if record:
        clean = [o for o in out if not o["vet"]]
        core.record_search(
            "apogee_cno_dwarfs", "CNO-processed unevolved dwarfs", archive="APOGEE DR17 (VizieR III/286)",
            selection="%d dwarfs: log g above 4.2, Teff 4800-6500 K, [Fe/H] above -0.3, S/N above 70, unflagged C and N" % len(keep),
            code=os.path.abspath(__file__), entry="cno", n_targets=len(keep),
            params=dict(sigma_dex=round(float(sig), 3), threshold=-sigma, teff=[4800, 6500], logg=4.2, feh=-0.3, ce_max=ce_max, na_max=na_max,
                        rv_scatter_max=rv_scatter_max, teff_edge=teff_edge, vsini_max=vsini_max, n_control=n_control),
            null_summary="the carbon-rich mirror: %d stars %g sigma above the locus against %d below" % (len(high), sigma, len(out)),
            sensitivity="[C/N] %.2f dex below the locus, the nitrogen doubling of an envelope stripped by a few tenths of a solar mass"
                        % (sigma * sig),
            ledger="CNO-processed unevolved dwarfs",
            result="%d dwarfs %g sigma nitrogen-rich against %d carbon-rich; %d are Gaia binaries against %.0f per cent of "
                   "random dwarfs; %d carry s-process, sodium, companion, rotation, evolution or Teff-edge flags, %d none"
                   % (len(out), sigma, len(high), sum(1 for o in out if o["vet"] and "Gaia binary" in o["vet"]), 100 * base,
                      len(out) - len(clean), len(clean)))
        core.clear_targets("apogee_cno_dwarfs")
        core.record_targets("apogee_cno_dwarfs", [dict(
            target="2mass:%s" % o["id"].replace("2M", ""), stat=o["z"], unit="sigma below the [C/N] dwarf locus", vet=o["vet"],
            note="Teff %.0f, log g %.2f, [Fe/H] %+.2f, [C/N] %+.2f against %+.2f" % (o["teff"], o["logg"], o["feh"], o["cn"], o["locus"]))
            for o in out])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------
# lithium: an element old cool dwarfs have burned

lithium_DOC = r"""Lithium in old cool dwarfs: GALAH DR4 dwarfs below 5300 K (log g above 4.3) with a clean A(Li) above 2.0 and an isochrone age above 6 Gyr, an element such a star has burned, read against their Gaia DR3 speeds (young stars move slowly, old ones fast, and isochrone ages of K dwarfs are loose) and the lithium mass a 0.04 M_sun convective envelope needs to hold it.

`refit` measures the Li I 6708 A line again in each star's red-arm spectrum: the spectrum's velocity from H-alpha, Fe I 6705.10 and 6710.32 and Ca I 6717.69, then a Gaussian at the doublet's centre on a local linear continuum at each width the star's line may have (the instrument's, that broadened by GALAH's vsini, and the star's own Ca I 6717.69 width), its error the larger of the formal one and the scatter of the same fit at 40 control wavelengths of the star's own spectrum; GALAH's line is confirmed at any width, absent where every width excludes it at 3 sigma, weaker, or unconstrained by the spectrum. Lithium-poor stars of the same selection matched in S/N, read against their rich partners' claimed widths, give the rule's false confirmations, and the same stars with a line of the claimed width injected at the rotational width measure the recovery; each spectrum's velocity against the catalogue's barycentric velocity and correction gives the frame the served spectra are in.

    python -I seti.py composition lithium [refit|record]"""

lithium_C = 299792.458
# H-alpha, Fe I 6705.10 and 6710.32 and Ca I 6717.69 (air, A): the spectrum's velocity
lithium_REF = (6562.80, 6705.10, 6710.32, 6717.69)
# the 7Li doublet's gf-weighted centre (6707.761 and 6707.912 A at log gf -0.002 and -0.303)
lithium_LINE = 6707.81
# the instrumental profile at R 28000 with the doublet's spread (A); the doublet alone spreads its 2:1 components by 0.071 A
lithium_SIGMA = 0.13
lithium_DOUBLET = 0.071
# Ca I 6717.69 A: the star's own line width, rotation included
lithium_WIDTH_LINE = 6717.69
# strong lines of a K dwarf at 6600-6700 A, kept 2 A from the control wavelengths
lithium_AVOID = lithium_REF + (6707.8, 6678.0, 6643.6, 6663.4, 6625.0, 6634.1, 6696.0, 6699.1)


def lithium_main(record=False):
    import math
    num = core.num
    rows = core.tap("datacentral", "SELECT sobject_id, gaiadr3_source_id, teff, logg, fe_h, a_li, age FROM galah_dr4.mainstartable "
                                   "WHERE teff < 5300 AND logg > 4.3 AND flag_a_li = 0 AND age > 6", timeout=600)
    rich = [r for r in rows if num(r["a_li"]) is not None and float(r["a_li"]) > 2.0 and r["gaiadr3_source_id"]]
    g = core.tap_by_ids("gaia", [r["gaiadr3_source_id"] for r in rich], "SELECT source_id, parallax, pmra, pmdec, radial_velocity, ruwe "
                                                                         "FROM gaiadr3.gaia_source WHERE source_id IN (%s)")
    # a 0.04 M_sun envelope, hydrogen mass fraction 0.7: lithium mass (kg) at abundance A
    m_env, m_h, m_li = 0.04 * 1.989e30, 1.6735e-27, 7.016 * 1.6605e-27
    out = []
    for r in rich:
        x = g.get(r["gaiadr3_source_id"])
        v = None
        if x and num(x["parallax"]) and float(x["parallax"]) > 0:
            vt = 4.74047 * math.hypot(float(x["pmra"]), float(x["pmdec"])) / float(x["parallax"])
            rv = num(x["radial_velocity"])
            v = math.hypot(vt, rv) if rv is not None else vt
        a = float(r["a_li"])
        out.append(dict(gaia=r["gaiadr3_source_id"], sobject=r["sobject_id"], teff=float(r["teff"]), feh=num(r["fe_h"]), a_li=a,
                        age=float(r["age"]), v=v, ruwe=num(x["ruwe"]) if x else None,
                        li_kg=10 ** (a - 12) * 0.7 * m_env / m_h * m_li))
    slow = [o for o in out if o["v"] is not None and o["v"] < 20]
    fast = sorted([o for o in out if o["v"] is not None and o["v"] > 60], key=lambda o: -o["a_li"])
    top = max(out, key=lambda o: o["a_li"]) if out else None
    print("%d old cool dwarfs with a clean A(Li); %d above 2.0, %d of them at young-disc speeds (below 20 km/s), %d at thick-disc speeds "
          "(above 60 km/s); the highest A(Li) %.2f needs %.1e kg of lithium in the envelope" % (
              len(rows), len(out), len(slow), len(fast), top["a_li"] if top else 0, top["li_kg"] if top else 0))
    for o in fast:
        print("  Gaia DR3 %s: A(Li) %.2f, %.0f K, [Fe/H] %+.2f, age %.1f, %.0f km/s, RUWE %.2f" % (
            o["gaia"], o["a_li"], o["teff"], o["feh"] or 0, o["age"], o["v"], o["ruwe"] or 0))
    json.dump(dict(n=len(rows), rich=out), open(core.results("lithium_old_dwarfs.json"), "w"), indent=1)
    if record:
        # the refit's verdict on each star's line, when it has run
        fit = {}
        if os.path.exists(core.results("lithium_refit.json")):
            fit = {r["gaiadr3_source_id"]: r for r in json.load(open(core.results("lithium_refit.json")))["rich"] if "verdict" in r}

        def vet(o):
            f = fit.get(o["gaia"])
            if f and (f["verdict"] == "absent" or (f["verdict"] == "weaker" and f["a_li_refit"] < 2.0)):
                return "artefact"
            if f and f["verdict"] == "unconstrained":
                return None
            codes = []
            if (f and f.get("young")) or (o["v"] is not None and o["v"] < 20):
                codes.append("young-star")
            if f and f.get("double"):
                codes.append("binary")
            return "; ".join(codes) or None

        def note(o):
            f = fit.get(o["gaia"])
            s = "%.0f K, [Fe/H] %+.2f, isochrone age %.1f Gyr, %s km/s" % (o["teff"], o["feh"] or 0, o["age"],
                                                                        "%.0f" % o["v"] if o["v"] is not None else "-")
            if f:
                s += "; Li I 6708 refitted at %.0f +- %.0f mA against GALAH's %.0f: %s" % (f["ew_fit"], f["e"], float(f["ew_li"]), f["verdict"])
                if f["verdict"] == "unconstrained":
                    s += ", the spectrum (S/N %.0f) too noisy to test the line" % float(f["snr_px_ccd3"] or 0)
                elif f.get("young") or f.get("double"):
                    s += "; %s" % ", ".join(f.get("young", []) + f.get("double", []))
            return s
        n_fit = {k: sum(1 for f in fit.values() if f["verdict"] == k) for k in ("confirmed", "weaker", "absent", "unconstrained")}
        core.record_search(
            "galah_lithium_old_dwarfs", "Lithium in old cool dwarfs", archive="GALAH DR4 (Data Central) catalogue and red-arm spectra; "
            "Gaia DR3 gaia_source", release="DR4",
            selection="dwarfs below 5300 K, log g above 4.3, clean A(Li), isochrone age above 6 Gyr: %d" % len(rows),
            code=os.path.abspath(__file__), entry="lithium", n_targets=len(rows), n_unit="stars",
            params=dict(teff_k=5300, logg=4.3, a_li=2.0, age_gyr=6, slow_kms=20, fast_kms=60, envelope_msun=0.04,
                        line_a=lithium_LINE, sigma_a=lithium_SIGMA, control_wavelengths=40),
            null_summary="kinematic ages: stars at young-disc speeds, where the isochrone age misleads; the line refitted in each "
                         "spectrum against lithium-poor stars of the same S/N and injected lines",
            sensitivity="A(Li) above 2.0 at a K dwarf's surface, %.0e kg of lithium in its envelope at the threshold" % (
                10 ** (2.0 - 12) * 0.7 * m_env / m_h * m_li),
            ledger="Lithium in old cool dwarfs",
            result="%d above 2.0, %d at young-disc speeds, %d at thick-disc speeds; the highest, A(Li) %.2f, needs %.0e kg of lithium%s" % (
                len(out), len(slow), len(fast), top["a_li"] if top else 0, top["li_kg"] if top else 0,
                "; refitted: %d confirmed, %d weaker, %d absent, %d unconstrained" % (
                    n_fit["confirmed"], n_fit["weaker"], n_fit["absent"], n_fit["unconstrained"]) if fit else ""))
        core.clear_targets("galah_lithium_old_dwarfs")
        core.record_targets("galah_lithium_old_dwarfs", [dict(
            target="gaia_dr3:%s" % o["gaia"], stat=o["a_li"], unit="A(Li)", vet=vet(o), note=note(o)) for o in out])
        print("recorded")


def lithium_spectrum(sobject):
    """GALAH DR4 red-arm spectrum: wavelength (air, A) and normalised flux, the star's lines at its observed velocity."""
    from astropy.io import fits
    h = fits.open(io.BytesIO(core.galah_spectrum(sobject, "R")))
    H, f = h[0].header, np.asarray(h[0].data, float)
    return H["CRVAL1"] + H.get("PC1_1", 1.0) * H.get("CDELT1", 1.0) * (np.arange(len(f)) + 1 - H["CRPIX1"]), f


def lithium_velocity(lam, f):
    """The velocity (km/s) that puts the reference lines deepest: a 0.5 km/s grid over +-300 km/s, then 0.05 km/s."""
    def depth(v):
        return sum(np.interp(c * (1 + v / lithium_C), lam, f) for c in lithium_REF)
    v0 = min(np.arange(-300.0, 300.01, 0.5), key=depth)
    return float(min(np.arange(v0 - 2.0, v0 + 2.01, 0.05), key=depth))


def lithium_fit(lam, f, centre, sigma=lithium_SIGMA):
    """A Gaussian of fixed width at a fixed centre on a linear continuum within 1.5 A: equivalent width and its formal error (mA)."""
    from scipy.optimize import least_squares
    w = (np.abs(lam - centre) < 1.5) & np.isfinite(f)
    x, y = lam[w] - centre, f[w]
    if len(y) < 20:
        return np.nan, np.nan

    def resid(p):
        return (p[0] + p[1] * x) * (1 - p[2] * np.exp(-0.5 * (x / sigma) ** 2)) - y
    r = least_squares(resid, [np.percentile(y, 90), 0.0, 0.1])
    s2 = float(np.sum(r.fun ** 2) / max(len(y) - 3, 1))
    try:
        e = float(np.sqrt(np.linalg.inv(r.jac.T @ r.jac)[2, 2] * s2))
    except np.linalg.LinAlgError:
        e = np.nan
    k = np.sqrt(2 * np.pi) * sigma * 1000.0
    return float(r.x[2] * k), e * k


def lithium_width(lam, f, centre):
    """The star's line width (A): a Gaussian of free width and depth fitted to Ca I 6717.69 at its velocity, which carries the instrument, the star's rotation and its turbulence."""
    from scipy.optimize import least_squares
    w = (np.abs(lam - centre) < 1.5) & np.isfinite(f)
    x, y = lam[w] - centre, f[w]
    if len(y) < 20:
        return np.nan

    def resid(p):
        return (p[0] + p[1] * x) * (1 - p[2] * np.exp(-0.5 * (x / p[3]) ** 2)) - y
    r = least_squares(resid, [np.percentile(y, 90), 0.0, 0.3, 0.13], bounds=([0, -np.inf, 0, 0.06], [np.inf, np.inf, 1, 0.8]))
    return float(r.x[3]) if r.x[2] > 0.05 else np.nan


def lithium_inject(lam, f, centre, ew_ma, sigma=lithium_SIGMA):
    """The spectrum with a line of the given equivalent width and width added at the centre, no deeper than 0.7, wider where the equivalent width demands it, as a saturated line is."""
    sigma = max(sigma, ew_ma / 1000.0 / (np.sqrt(2 * np.pi) * 0.7))
    d = ew_ma / 1000.0 / (np.sqrt(2 * np.pi) * sigma)
    return f * (1 - d * np.exp(-0.5 * ((lam - centre) / sigma) ** 2))


def lithium_widths(row, lam, f, s):
    """The widths (A) a lithium line of this star may have: the instrument's with the doublet's spread, that broadened by GALAH's vsini (a rotation profile's standard deviation is 0.47 of its half-width), and the star's own Ca I 6717.69 width with the doublet's spread, each within 0.13-0.8 A."""
    out = {"instrument": lithium_SIGMA}
    vs = core.num(row.get("vsini"))
    if vs is not None and np.isfinite(vs):
        out["rotation"] = float(np.clip(np.hypot(lithium_SIGMA, 0.47 * lithium_LINE * vs / lithium_C), lithium_SIGMA, 0.8))
    w = lithium_width(lam, f, lithium_WIDTH_LINE * s)
    if np.isfinite(w):
        out["calcium"] = float(np.clip(np.hypot(w, lithium_DOUBLET), lithium_SIGMA, 0.8))
    return out


def lithium_measure(row, inject=None):
    """One spectrum: its velocity, then at each width `lithium_widths` allows the Li line's equivalent width with the larger of its formal error and the robust scatter of the same fit at 40 control wavelengths of the star's own spectrum; with `inject` (mA) the widths recovered from a line of that equivalent width added at the rotational width."""
    try:
        lam, f = lithium_spectrum(row["sobject_id"])
    except Exception as ex:
        return dict(row, error=repr(ex)[:160])
    v = lithium_velocity(lam, f)
    s = 1 + v / lithium_C
    rng = np.random.default_rng(int(row["sobject_id"]) % 2 ** 32)
    controls = [c for c in rng.uniform(6600.0, 6700.0, 400) if all(abs(c - a) > 2.0 for a in lithium_AVOID)][:40]
    widths = lithium_widths(row, lam, f, s)
    fits = {}
    for name, sg in widths.items():
        ew, ef = lithium_fit(lam, f, lithium_LINE * s, sg)
        null = np.array([x for x in (lithium_fit(lam, f, c * s, sg)[0] for c in controls) if np.isfinite(x)])
        en = core.robust_sigma(null) if len(null) >= 10 else np.nan
        fits[name] = dict(sigma=sg, ew=ew, e=float(np.nanmax([ef, en])), e_formal=ef, e_null=float(en),
                          null_median=float(np.median(null)) if len(null) else np.nan)
    out = dict(row, v_fit=v, fits=fits)
    if inject:
        sg = widths.get("rotation", lithium_SIGMA)
        fi = lithium_inject(lam, f, lithium_LINE * s, inject, sg)
        out["ew_injected"] = float(inject)
        out["recovered"] = {k: lithium_fit(lam, fi, lithium_LINE * s, x["sigma"])[0] for k, x in fits.items()}
    return out


def lithium_verdict(m, ew=None, recovered=False):
    """GALAH's line (or `ew`, mA) at each width: confirmed (detected at 3 sigma and at least half the width or within 3 sigma of it), weaker (detected, the width excluded at 3 sigma and under half), absent (not detected, the width excluded at 3 sigma) or unconstrained; the star's verdict is confirmed at any width, else absent at every width, else weaker at any, else unconstrained. With `recovered` the widths measured after injection stand in for the fitted ones."""
    g = float(m["ew_li"]) if ew is None else ew
    each = {}
    for k, x in m["fits"].items():
        val = m["recovered"][k] if recovered else x["ew"]
        if not (np.isfinite(val) and np.isfinite(x["e"])):
            continue
        excluded = g - val > 3 * x["e"]
        if val >= 3 * x["e"]:
            each[k] = "confirmed" if val >= 0.5 * g or not excluded else "weaker"
        else:
            each[k] = "absent" if excluded else "unconstrained"
    if not each:
        return "unmeasured"
    v = set(each.values())
    return ("confirmed" if "confirmed" in v else "absent" if v == {"absent"} else "weaker" if "weaker" in v else "unconstrained")


def lithium_young(m):
    """Signs of youth in a K dwarf: a young-disc speed (below 20 km/s), rotation that a K dwarf keeps for about its first gigayear (vsini 5 km/s or more, a period under 8 days at 0.8 R_sun), H-alpha filled in by emission (GALAH's residual equivalent width below -0.3 A) or the RVS Ca II triplet activity index above 0.05 nm."""
    why = []
    if m.get("v_gaia") is not None and m["v_gaia"] < 20:
        why.append("young-disc speed %.0f km/s" % m["v_gaia"])
    vs = core.num(m.get("vsini"))
    if vs is not None and vs >= 5:
        why.append("vsini %.0f km/s" % vs)
    ha = core.num(m.get("ew_h_alpha"))
    if ha is not None and ha < -0.3:
        why.append("H-alpha in emission (%.2f A)" % ha)
    if m.get("activity") is not None and m["activity"] > 0.05:
        why.append("Ca II triplet activity %.2f nm" % m["activity"])
    return why


def lithium_double(m):
    """Signs of a companion: RUWE above 1.4, Gaia images doubled in more than a tenth of transits, or RVS velocities that vary (chi-square p below 0.01)."""
    why = []
    if (m.get("ruwe") or 0) > 1.4:
        why.append("RUWE %.1f" % m["ruwe"])
    if (m.get("multi_peak") or 0) > 10:
        why.append("double images in %d per cent of transits" % m["multi_peak"])
    if m.get("rv_p") is not None and m["rv_p"] < 0.01:
        why.append("velocity varies (p %.1e)" % m["rv_p"])
    return why


def lithium_best(m):
    """The width at which the line stands highest over its error: (name, equivalent width, error, sigma)."""
    k = max(m["fits"], key=lambda k: m["fits"][k]["ew"] / m["fits"][k]["e"] if np.isfinite(m["fits"][k]["ew"] / m["fits"][k]["e"]) else -np.inf)
    x = m["fits"][k]
    return k, x["ew"], x["e"], x["sigma"]


def lithium_refit():
    from concurrent.futures import ThreadPoolExecutor
    if not os.path.exists(core.results("lithium_old_dwarfs.json")):
        lithium_main()
    kin = {o["gaia"]: o for o in json.load(open(core.results("lithium_old_dwarfs.json")))["rich"]}
    cols = "sobject_id, gaiadr3_source_id, teff, logg, fe_h, a_li, ew_li, age, snr_px_ccd3, rv_comp_1, v_bary_eff, vsini, nn_li_fe, ew_h_alpha"
    rich = core.tap("datacentral", "SELECT %s FROM galah_dr4.mainstartable WHERE teff < 5300 AND logg > 4.3 AND age > 6 "
                                   "AND flag_a_li = 0 AND a_li > 2.0" % cols, timeout=600)
    rich = [r for r in rich if r["gaiadr3_source_id"] in kin]
    poor = core.tap("datacentral", "SELECT %s FROM galah_dr4.mainstartable WHERE teff < 5300 AND logg > 4.3 AND age > 6 "
                                   "AND flag_sp = 0 AND ew_li < 20" % cols, timeout=600)
    # one lithium-poor star of the nearest S/N for each rich one, without replacement
    pool = sorted(poor, key=lambda r: float(r["snr_px_ccd3"] or 0))
    snr = np.array([float(r["snr_px_ccd3"] or 0) for r in pool])
    used, controls = set(), []
    for r in sorted(rich, key=lambda r: -float(r["snr_px_ccd3"] or 0)):
        for j in np.argsort(np.abs(snr - float(r["snr_px_ccd3"] or 0))):
            if j not in used and pool[j]["gaiadr3_source_id"] not in kin:
                used.add(j)
                controls.append(dict(pool[j], inject_for=r["sobject_id"], inject_ew=float(r["ew_li"])))
                break
    print("%d rich stars, %d lithium-poor stars of the selection, %d S/N-matched controls" % (len(rich), len(poor), len(controls)), flush=True)
    with ThreadPoolExecutor(6) as ex:
        rich_m = list(ex.map(lithium_measure, rich))
        ctrl_m = list(ex.map(lambda r: lithium_measure(r, inject=r["inject_ew"]), controls))
    ok_r = [m for m in rich_m if m.get("fits")]
    ok_c = [m for m in ctrl_m if m.get("fits")]
    for m in ok_r:
        m["verdict"] = lithium_verdict(m)
        m["width_name"], m["ew_fit"], m["e"], m["sigma"] = lithium_best(m)
        # on the linear part of the curve of growth the abundance moves with the log of the width
        m["a_li_refit"] = float(m["a_li"]) + float(np.log10(max(m["ew_fit"], m["e"]) / float(m["ew_li"])))
        k = kin[m["gaiadr3_source_id"]]
        m["v_gaia"], m["ruwe"] = k["v"], k["ruwe"]
    # what makes a lithium-rich K dwarf young or double: Gaia DR3 image doubling, velocity scatter and the RVS activity index
    ids = [m["gaiadr3_source_id"] for m in ok_r]
    gx = core.tap_by_ids("gaia", ids, "SELECT source_id, ipd_frac_multi_peak, rv_chisq_pvalue FROM gaiadr3.gaia_source WHERE source_id IN (%s)")
    ax = core.tap_by_ids("gaia", ids, "SELECT source_id, activityindex_espcs FROM gaiadr3.astrophysical_parameters WHERE source_id IN (%s)")
    for m in ok_r:
        x, a = gx.get(m["gaiadr3_source_id"], {}), ax.get(m["gaiadr3_source_id"], {})
        m["multi_peak"], m["rv_p"] = core.num(x.get("ipd_frac_multi_peak")), core.num(x.get("rv_chisq_pvalue"))
        m["activity"] = core.num(a.get("activityindex_espcs"))
        m["young"] = lithium_young(m)
        m["double"] = lithium_double(m)
    for m in ok_c:
        # a lithium-poor star read against the claimed width of its rich partner: the rule's false confirmations
        m["verdict"] = lithium_verdict(m, ew=m["inject_ew"])
        m["verdict_injected"] = lithium_verdict(m, ew=m["inject_ew"], recovered=True)
        m["width_name"], m["ew_fit"], m["e"], m["sigma"] = lithium_best(m)
    counts = {k: sum(m["verdict"] == k for m in ok_r) for k in ("confirmed", "weaker", "absent", "unconstrained", "unmeasured")}
    print("rich: %d refitted of %d: %s" % (len(ok_r), len(rich), counts))
    for name in ("instrument", "rotation", "calcium"):
        w = [m["fits"][name]["sigma"] for m in ok_r if name in m["fits"]]
        print("  %s widths: %d stars, median %.3f A (5th-95th %.3f-%.3f)" % (name, len(w), np.median(w), *np.percentile(w, [5, 95])))
    c_ew = np.array([m["fits"]["instrument"]["ew"] for m in ok_c])
    c_e = np.array([m["fits"]["instrument"]["e"] for m in ok_c])
    print("controls: %d refitted; at the instrumental width their line reads %.1f mA median (robust scatter %.1f), %d above 3 sigma; "
          "errors median %.1f mA; against their rich partners' widths %s" % (
              len(ok_c), np.median(c_ew), core.robust_sigma(c_ew), int(np.sum(c_ew > 3 * c_e)), np.median(c_e),
              {k: sum(m["verdict"] == k for m in ok_c) for k in ("confirmed", "weaker", "absent", "unconstrained")}))
    rec = np.array([m["recovered"]["rotation"] / m["ew_injected"] for m in ok_c if "rotation" in m.get("recovered", {})])
    print("lines of the partners' widths injected at the rotational width: recovered over injected at that width median %.2f "
          "(quartiles %.2f-%.2f); verdicts %s" % (np.median(rec), *np.percentile(rec, [25, 75]),
                                                  {k: sum(m["verdict_injected"] == k for m in ok_c) for k in ("confirmed", "weaker", "absent", "unconstrained")}))
    # the frame: velocity found against the catalogue's barycentric velocity and the barycentric correction
    allm = [m for m in ok_r + ok_c if m.get("rv_comp_1") and m.get("v_bary_eff")]
    dv = np.array([m["v_fit"] - float(m["rv_comp_1"]) for m in allm])
    vb = np.array([float(m["v_bary_eff"]) for m in allm])
    good = np.abs(dv + vb * 0.5) < 15
    slope, icpt = np.polyfit(vb[good], dv[good], 1)
    print("frame: spectrum velocity minus rv_comp_1 = %.2f %+.3f v_bary_eff over %d spectra (slope -1 for topocentric, 0 for barycentric); "
          "residual scatter %.2f km/s" % (icpt, slope, int(good.sum()), core.robust_sigma(dv[good] - (icpt + slope * vb[good]))))
    def line(m):
        return ("  Gaia DR3 %s: GALAH %.0f mA (A(Li) %.2f); refitted %s; %.0f K, S/N %.0f, %s km/s, vsini %.0f, H-alpha EW %.2f, %s" % (
            m["gaiadr3_source_id"], float(m["ew_li"]), float(m["a_li"]),
            ", ".join("%.0f +- %.0f mA at %.2f A" % (x["ew"], x["e"], x["sigma"]) for x in m["fits"].values()),
            float(m["teff"]), float(m["snr_px_ccd3"] or 0), "%.0f" % m["v_gaia"] if m["v_gaia"] is not None else "-",
            core.num(m["vsini"], np.nan), core.num(m["ew_h_alpha"], np.nan), m["verdict"]))
    for v in ("confirmed", "weaker", "absent", "unconstrained"):
        s = sorted([m for m in ok_r if m["verdict"] == v], key=lambda m: -float(m["ew_li"]))
        print("%s (%d), S/N median %.0f, vsini median %.0f:" % (v, len(s), np.median([float(m["snr_px_ccd3"] or 0) for m in s]) if s else 0,
                                                                np.nanmedian([core.num(m["vsini"], np.nan) for m in s]) if s else 0))
        for m in s[:10]:
            print(line(m))
    fast = sorted([m for m in ok_r if m["v_gaia"] is not None and m["v_gaia"] > 60], key=lambda m: -float(m["ew_li"]))
    print("at thick-disc speeds (%d):" % len(fast))
    for m in fast:
        print(line(m) + "; %s" % ", ".join(m["young"] + m["double"]))
    real = [m for m in ok_r if m["verdict"] in ("confirmed", "weaker")]
    left = [m for m in real if not (m["young"] or m["double"])]
    print("lines in the spectra (%d): signs of youth in %d, of a companion in %d, neither in %d" % (
        len(real), sum(bool(m["young"]) for m in real), sum(bool(m["double"]) for m in real), len(left)))
    for m in left:
        print(line(m))
    json.dump(dict(rich=rich_m, controls=ctrl_m, frame=dict(intercept=float(icpt), slope=float(slope), n=int(good.sum()))),
              open(core.results("lithium_refit.json"), "w"), indent=1, default=float)


def lithium_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if mode == "refit":
        lithium_refit()
    else:
        lithium_main(record=mode == "record")


# --------------------------------------------------------------------------------------------------------------
# misfit: spectra no stellar model fits

misfit_DOC = r"""Spectra no stellar model fits: GALAH DR4 dwarfs within 100 pc (parallax above 10 mas, log g above 3.8) at S/N above 100 in the red arm, the 0.5 per cent whose spectral fit has the highest chi-square, light that is not a single star's, each read for a second velocity component, fast rotation, emission in H-alpha, GALAH's own flags and a companion in Gaia DR3 (RUWE, doubled images, velocity scatter, a non-single-star solution).

    python -I seti.py composition misfit [record] [--top 0.005] [--snr 100] [--plx-mas 10] [--logg-min 3.8] [--vsini-kms 20] [--halpha-ew -0.3]"""

MISFIT_TOP = 0.005


def misfit_main(record=False, top=MISFIT_TOP, snr=100.0, plx_mas=10.0, logg_min=3.8, vsini_kms=20.0, halpha_ew=-0.3):
    rows = core.tap("datacentral", "SELECT sobject_id, gaiadr3_source_id, teff, logg, fe_h, chi2_sp, flag_sp, snr_px_ccd3, vsini, rv_comp_1, rv_comp_2, "
                                   "sb2_rv_50, ew_h_alpha, parallax FROM galah_dr4.mainstartable WHERE parallax > %g AND logg > %g AND snr_px_ccd3 > %g"
                                   % (plx_mas, logg_min, snr), timeout=600)
    rows = [r for r in rows if core.num(r["chi2_sp"]) is not None]
    chi = np.array([float(r["chi2_sp"]) for r in rows])
    cut = float(np.quantile(chi, 1 - top))
    top = sorted([r for r in rows if float(r["chi2_sp"]) >= cut], key=lambda r: -float(r["chi2_sp"]))
    g = core.tap_by_ids("gaia", [r["gaiadr3_source_id"] for r in top if r["gaiadr3_source_id"]],
                        "SELECT source_id, ruwe, ipd_frac_multi_peak, rv_chisq_pvalue, non_single_star FROM gaiadr3.gaia_source WHERE source_id IN (%s)")
    out = []
    for r in top:
        x = g.get(r["gaiadr3_source_id"], {})
        why = []
        if core.num(r["rv_comp_2"]) is not None or core.num(r["sb2_rv_50"]) is not None:
            why.append("second velocity component")
        if (core.num(r["vsini"]) or 0) > vsini_kms:
            why.append("vsini %.0f km/s" % float(r["vsini"]))
        if (core.num(r["ew_h_alpha"]) or 0) < halpha_ew:
            why.append("H-alpha in emission (%.2f A)" % float(r["ew_h_alpha"]))
        if (core.num(x.get("ruwe")) or 0) > 1.4 or (core.num(x.get("ipd_frac_multi_peak")) or 0) > 10 or (core.num(x.get("non_single_star")) or 0) > 0 \
                or (core.num(x.get("rv_chisq_pvalue")) is not None and core.num(x.get("rv_chisq_pvalue")) < 0.01):
            why.append("companion in Gaia (RUWE %s, double images %s%%, NSS %s, RV p %s)" % (x.get("ruwe"), x.get("ipd_frac_multi_peak"),
                                                                                         x.get("non_single_star"), x.get("rv_chisq_pvalue")))
        if int(core.num(r["flag_sp"]) or 0) != 0:
            why.append("GALAH flag_sp %d" % int(core.num(r["flag_sp"])))
        out.append(dict(sobject=r["sobject_id"], gaia=r["gaiadr3_source_id"], chi2=float(r["chi2_sp"]), teff=core.num(r["teff"]), logg=core.num(r["logg"]),
                        snr=core.num(r["snr_px_ccd3"]), d=1000.0 / float(r["parallax"]), why=why))
    left = [o for o in out if not o["why"]]
    print("%d GALAH dwarfs within 100 pc at S/N above 100; chi-square median %.2f, the top %.1f per cent above %.2f: %d stars, %d with no "
          "binary, rotation, emission, flag or companion" % (len(rows), float(np.median(chi)), 100 * top, cut, len(out), len(left)))
    for o in out[:25]:
        print("  Gaia DR3 %s: chi2 %.2f, %.0f K, log g %.2f, %.0f pc, S/N %.0f; %s" % (o["gaia"], o["chi2"], o["teff"] or 0, o["logg"] or 0, o["d"], o["snr"] or 0,
                                                                           "; ".join(o["why"]) or "nothing"))
    json.dump(dict(n=len(rows), cut=cut, top=out), open(core.results("misfit_spectra.json"), "w"), indent=1)
    if record:
        def vet(o):
            v = []
            if any(w.startswith(("second", "companion")) for w in o["why"]):
                v.append("binary")
            if any(w.startswith("vsini") for w in o["why"]):
                v.append("rotation")
            if any(w.startswith("H-alpha") for w in o["why"]):
                v.append("active-star")
            return "; ".join(v) or None
        core.record_search(
            "galah_misfit_spectra", "Spectra no stellar model fits", archive="GALAH DR4 (Data Central); Gaia DR3 gaia_source", release="DR4",
            selection="dwarfs within 100 pc at S/N above 100 in the red arm: %d" % len(rows), code=os.path.abspath(__file__), entry="misfit",
            n_targets=len(rows), n_unit="stars", params=dict(top=top, snr=snr, plx_mas=plx_mas, logg=logg_min, vsini_kms=vsini_kms, halpha_ew=halpha_ew),
            null_summary="the fit's chi-square across the same stars: median %.2f, the top %.1f per cent above %.2f" % (float(np.median(chi)), 100 * top, cut),
            sensitivity="a nearby dwarf whose high-S/N spectrum no GALAH model fits and that is not a binary, a fast rotator or an active star",
            ledger="Spectra no stellar model fits",
            result="%d stars in the top %.1f per cent of chi-square, %d of them with no binary, rotation, emission, flag or companion" % (len(out), 100 * top, len(left)))
        core.clear_targets("galah_misfit_spectra")
        core.record_targets("galah_misfit_spectra", [dict(
            target="gaia_dr3:%s" % o["gaia"], stat=o["chi2"], unit="GALAH spectral-fit chi-square", vet=vet(o),
            note="%.0f K, log g %.2f, %.0f pc, S/N %.0f; %s" % (o["teff"] or 0, o["logg"] or 0, o["d"], o["snr"] or 0, "; ".join(o["why"]) or "nothing found"))
            for o in out if o["gaia"]])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------
# spectra the surveys could not classify, and abundances and rotation no star should show


def gaia_near(ids, plx=10.0):
    """{source_id: gaia row} for the ids with a parallax above plx at 10 sigma."""
    g = core.tap_by_ids("gaia", [i for i in ids if i], "SELECT source_id, parallax, parallax_over_error, phot_g_mean_mag, bp_rp, ruwe FROM "
                                                       "gaiadr3.gaia_source WHERE source_id IN (%s)")
    return {k: v for k, v in g.items() if (core.num(v["parallax"]) or 0) > plx and (core.num(v["parallax_over_error"]) or 0) > 10}


sdssunknown_DOC = r"""Nearby stars SDSS could not classify: SDSS DR17 primary spectra (SkyServer SpecObj) classed STAR with a nonzero redshift warning or no subclass at a median S/N above 20, matched to Gaia DR3 within 2 arcsec by CDS XMatch and kept within 100 pc at 10 sigma, each read in SIMBAD for a white dwarf or cataclysmic variable, which SDSS's stellar templates cover sparsely.

    python -I seti.py composition sdssunknown [record] [--snr 20] [--plx-mas 10] [--sigma 10] [--radius-arcsec 2]"""


def sdssunknown_main(record=False, snr=20.0, plx_mas=10.0, sigma=10.0, radius_arcsec=2.0):
    rows = core.sdss_sql("SELECT specObjID, ra, dec, subClass, zWarning, snMedian FROM SpecObj WHERE class = 'STAR' AND sciencePrimary = 1 AND "
                         "snMedian > %g AND (zWarning <> 0 OR subClass = '' OR subClass IS NULL)" % snr)
    # XMatch reads a numeric id column as a double, so the spectra travel under their row number
    up = [dict(source_id="s%d" % k, ra=float(r["ra"]), dec=float(r["dec"])) for k, r in enumerate(rows)]
    m = {}
    for x in core.xmatch(up, "I/355/gaiadr3", radius_arcsec) if up else []:
        plx = core.num(x.get("Plx"))
        if plx and plx > plx_mas and plx > sigma * (core.num(x.get("e_Plx")) or 1e9):
            m[x["source_id"]] = x
    by = {"s%d" % k: r for k, r in enumerate(rows)}
    hits = [dict(by[k], gaia=v.get("Source"), plx=core.num(v["Plx"]), g=core.num(v.get("Gmag"))) for k, v in m.items()]
    # a subclass is SDSS's own classification and the warning concerns the redshift fit (4 a weak best fit, 16 outlying pixels); white dwarfs,
    # which SDSS's templates cover sparsely, take the nearest template of any class
    from setitools import neighbourhood
    names = neighbourhood.simbad_names([str(h["gaia"]) for h in hits])

    def vet(h):
        t = names.get(str(h["gaia"]), ("-", "-"))[1]
        if t.startswith("WD") or t.startswith("CV"):
            return "white-dwarf"
        return "published-classification" if (h["subClass"] or "").strip() else None
    typed = [h for h in hits if (h["subClass"] or "").strip()]
    print("%d SDSS stellar spectra at S/N above 20 with a warning or no subclass; within 100 pc: %d, %d of them with a subclass, %d white dwarfs "
          "or cataclysmic variables in SIMBAD" % (len(rows), len(hits), len(typed), sum(vet(h) == "white-dwarf" for h in hits)))
    for h in sorted(hits, key=lambda h: h["subClass"] or ""):
        print("  SDSS %s: subclass %s, warning %s, %.1f mas, G %s; SIMBAD %s (%s)" % (h["specObjID"], h["subClass"] or "none", h["zWarning"], h["plx"],
                                                                                h["g"], *names.get(str(h["gaia"]), ("-", "-"))))
    if record:
        core.record_run(
            "sdss_unclassified_nearby", "Nearby stars SDSS could not classify", "SDSS DR17 SpecObj (SkyServer); Gaia DR3 by CDS XMatch",
            "primary stellar spectra at S/N above 20 with a warning or no subclass: %d" % len(rows), os.path.abspath(__file__), "sdssunknown", len(rows),
            "spectra", dict(snr=snr, plx_mas=plx_mas, sigma=sigma, radius_arcsec=radius_arcsec), "the stellar spectra of the same S/N the pipeline classified", "a nearby star whose SDSS spectrum "
            "the pipeline could neither fit nor type", "%d within 100 pc, %d of them typed by SDSS, %d white dwarfs or cataclysmic variables in SIMBAD" % (
                len(hits), len(typed), sum(vet(h) == "white-dwarf" for h in hits)),
            [dict(target="gaia_dr3:%s" % h["gaia"], vet=vet(h), note="SDSS %s, subclass %s, warning %s, %.1f mas; SIMBAD %s (%s)" % (
                h["specObjID"], h["subClass"] or "none", h["zWarning"], h["plx"], *names.get(str(h["gaia"]), ("-", "-")))) for h in hits],
            release="DR17")
        print("recorded")


lamostunknown_DOC = r"""Nearby stars LAMOST labelled unknown: LAMOST DR10 low-resolution spectra (public.catalogue) of class Unknown with r-band S/N above 50, their Gaia DR3 identifiers kept within 100 pc at 10 sigma.

    python -I seti.py composition lamostunknown [record] [--snr 50] [--plx-mas 10]"""


def lamostunknown_main(record=False, snr=50.0, plx_mas=10.0):
    cols = list(core.tap("lamost", "SELECT TOP 1 * FROM public.catalogue", fmt="votable")[0])
    gid = next((c for c in cols if "gaia" in c.lower() and "source" in c.lower()), None)
    rows = core.tap("lamost", "SELECT obsid, ra, dec, class, subclass, snrr%s FROM public.catalogue WHERE class = 'Unknown' AND snrr > %g" % (
        ", " + gid if gid else "", snr), fmt="votable", timeout=900)
    if gid:
        near = gaia_near([str(r[gid]) for r in rows], plx_mas)
        hits = [dict(r, gaia=str(r[gid])) for r in rows if str(r[gid]) in near]
    else:
        up = [dict(source_id=str(r["obsid"]), ra=float(r["ra"]), dec=float(r["dec"])) for r in rows]
        mm = {x["source_id"]: x for x in core.xmatch(up, "I/355/gaiadr3", 2.0) if (core.num(x.get("Plx")) or 0) > plx_mas}
        hits = [dict(r, gaia=mm[str(r["obsid"])].get("Source")) for r in rows if str(r["obsid"]) in mm]
    print("%d LAMOST spectra classed Unknown at r-band S/N above 50; within 100 pc: %d %s" % (
        len(rows), len(hits), [(h["obsid"], h["gaia"], h["snrr"]) for h in hits[:15]]))
    if record:
        core.record_run(
            "lamost_unknown_nearby", "Nearby stars LAMOST labelled unknown", "LAMOST DR10 LRS general catalogue; Gaia DR3",
            "spectra classed Unknown at r-band S/N above 50: %d" % len(rows), os.path.abspath(__file__), "lamostunknown", len(rows), "spectra",
            dict(snrr=snr, plx_mas=plx_mas), "the classed spectra of the same S/N", "a nearby star whose good LAMOST spectrum the pipeline could not type",
            "%d within 100 pc" % len(hits), [dict(target="gaia_dr3:%s" % h["gaia"], stat=core.num(h["snrr"]), unit="r-band S/N",
                                                 note="LAMOST obsid %s" % h["obsid"]) for h in hits])
        print("recorded")


desiunknown_DOC = r"""Bright nearby stars DESI fitted badly: DESI DR1 healpix redshifts (desi_dr1.zpix) of spectral type STAR with a nonzero ZWARN, joined to the Milky Way Survey's Gaia identifiers (desi_dr1.mws), kept within 100 pc at 10 sigma and G below 16.

    python -I seti.py composition desiunknown [record] [--plx-mas 10] [--g-max 16]"""


def desiunknown_main(record=False, plx_mas=10.0, g_max=16.0):
    rows = core.datalab("SELECT z.targetid, z.zwarn, z.deltachi2, m.source_id, m.rr_spectype FROM desi_dr1.zpix AS z JOIN desi_dr1.mws AS m "
                        "ON z.targetid = m.targetid WHERE z.spectype = 'STAR' AND z.zwarn > 0 AND m.source_id > 0")
    near = gaia_near([str(r["source_id"]) for r in rows], plx_mas)
    hits = [dict(r, g=core.num(near[str(r["source_id"])]["phot_g_mean_mag"])) for r in rows if str(r["source_id"]) in near
            and (core.num(near[str(r["source_id"])]["phot_g_mean_mag"]) or 99) < g_max]
    print("%d DESI stellar spectra with a fit warning and a Gaia identifier; within 100 pc at G below 16: %d %s" % (
        len(rows), len(hits), [(h["targetid"], h["source_id"], h["zwarn"], h["g"]) for h in hits[:15]]))
    if record:
        core.record_run(
            "desi_badfit_nearby", "Bright nearby stars DESI fitted badly", "DESI DR1 (Astro Data Lab) zpix and mws; Gaia DR3",
            "stellar spectra with a nonzero ZWARN and a Gaia identifier: %d" % len(rows), os.path.abspath(__file__), "desiunknown", len(rows),
            "spectra", dict(plx_mas=plx_mas, g=g_max), "the stellar spectra fitted without a warning", "a bright nearby star whose DESI spectrum no template fits",
            "%d within 100 pc at G below 16" % len(hits), [dict(target="gaia_dr3:%s" % h["source_id"], stat=core.num(h["zwarn"]), unit="ZWARN bits",
                                                                note="DESI target %s" % h["targetid"]) for h in hits])
        print("recorded")


def galah_dwarfs(cols, where):
    return core.tap("datacentral", "SELECT sobject_id, gaiadr3_source_id, teff, logg, fe_h, %s FROM galah_dr4.mainstartable WHERE logg > 4 AND "
                                   "flag_sp = 0 AND %s" % (cols, where), timeout=900)


zinc_DOC = r"""Zinc far above iron: GALAH DR4 dwarfs (log g above 4, clean fits) with clean zinc at [Fe/H] between -0.3 and +0.3 and [Zn/Fe] above +0.8, the signature of a galvanised alloy, against the scatter of the same stars' [Zn/Fe].

    python -I seti.py composition zinc [record] [--zn-fe 0.8] [--feh-range -0.3 0.3] [--snr-min 25]"""


def zinc_main(record=False, zn_fe=0.8, feh_range=[-0.3, 0.3], snr_min=25.0):
    rows = galah_dwarfs("zn_fe, e_zn_fe, snr_px_ccd1", "flag_zn_fe = 0 AND fe_h BETWEEN %g AND %g" % (feh_range[0], feh_range[1]))
    z = np.array([float(r["zn_fe"]) for r in rows])
    hits = sorted([r for r in rows if float(r["zn_fe"]) > zn_fe], key=lambda r: -float(r["zn_fe"]))
    print("%d clean dwarfs at near-solar metallicity; [Zn/Fe] median %+.2f, robust scatter %.2f; above +0.8: %d %s" % (
        len(rows), float(np.median(z)), core.robust_sigma(z), len(hits), [(r["gaiadr3_source_id"], r["zn_fe"], r["e_zn_fe"]) for r in hits[:10]]))
    if record:
        core.record_run(
            "galah_zinc_excess", "Zinc far above iron", "GALAH DR4 (Data Central)", "clean dwarfs at [Fe/H] -0.3 to +0.3 with clean zinc: %d" % len(rows),
            os.path.abspath(__file__), "zinc", len(rows), "stars", dict(zn_fe=zn_fe, feh_range=feh_range, snr_min=snr_min), "the sample's own [Zn/Fe]: median %+.2f, robust scatter %.2f" % (
                float(np.median(z)), core.robust_sigma(z)), "a dwarf with [Zn/Fe] above +0.8 at solar metallicity", "%d above +0.8" % len(hits),
            [dict(target="gaia_dr3:%s" % r["gaiadr3_source_id"], stat=float(r["zn_fe"]), unit="[Zn/Fe]",
                  vet="artefact" if float(r["snr_px_ccd1"] or 0) < snr_min else None,
                  note="+- %s, %.0f K, blue-arm S/N %.0f, where the zinc lines lie" % (r["e_zn_fe"], float(r["teff"]), float(r["snr_px_ccd1"] or 0)))
             for r in hits])
        print("recorded")


mined_DOC = r"""Dwarfs that look mined: GALAH DR4 dwarfs (log g above 4, clean fits) with clean silicon and aluminium, [Si/Fe] and [Al/Fe] both below -0.4 at [Fe/H] above -0.3, rock-forming elements gone with iron intact, against the scatter of the same stars' ratios.

    python -I seti.py composition mined [record] [--si-fe -0.4] [--al-fe -0.4] [--feh-min -0.3]"""


def mined_main(record=False, si_fe=-0.4, al_fe=-0.4, feh_min=-0.3):
    rows = galah_dwarfs("si_fe, al_fe, e_si_fe, e_al_fe", "flag_si_fe = 0 AND flag_al_fe = 0 AND fe_h > %g" % feh_min)
    si, al = np.array([float(r["si_fe"]) for r in rows]), np.array([float(r["al_fe"]) for r in rows])
    hits = [r for r in rows if float(r["si_fe"]) < si_fe and float(r["al_fe"]) < al_fe]
    print("%d clean dwarfs; [Si/Fe] scatter %.2f, [Al/Fe] scatter %.2f; both below -0.4: %d %s" % (
        len(rows), core.robust_sigma(si), core.robust_sigma(al), len(hits), [(r["gaiadr3_source_id"], r["si_fe"], r["al_fe"]) for r in hits[:10]]))
    if record:
        core.record_run(
            "galah_mined_dwarfs", "Dwarfs that look mined", "GALAH DR4 (Data Central)", "clean dwarfs at [Fe/H] above -0.3 with clean Si and Al: %d" % len(rows),
            os.path.abspath(__file__), "mined", len(rows), "stars", dict(si_fe=si_fe, al_fe=al_fe, feh_min=feh_min), "the sample's scatter: [Si/Fe] %.2f, [Al/Fe] %.2f" % (
                core.robust_sigma(si), core.robust_sigma(al)), "a dwarf missing silicon and aluminium with iron intact", "%d with both below -0.4" % len(hits),
            [dict(target="gaia_dr3:%s" % r["gaiadr3_source_id"], stat=float(r["si_fe"]), unit="[Si/Fe]", note="[Al/Fe] %s" % r["al_fe"]) for r in hits])
        print("recorded")


dibnear_DOC = r"""Interstellar bands within 50 pc: GALAH DR4 stars with parallaxes above 20 mas, their fitted 5780 A diffuse-interstellar-band equivalent widths against the scatter of the same fit across them, where the Local Bubble holds too little gas for the band, and against stars at 0.5-1 kpc, where it shows.

    python -I seti.py composition dibnear [record] [--plx-mas 20] [--sigma 5] [--ew-a 0.02] [--far-plx-mas 1 2]"""


def dibnear_main(record=False, plx_mas=20.0, sigma=5.0, ew_a=0.02, far_plx_mas=[1.0, 2.0]):
    near = core.tap("datacentral", "SELECT sobject_id, gaiadr3_source_id, parallax, teff, ew_dib5780, sigma_dib5780, rv_dib5780 FROM galah_dr4.mainstartable "
                                   "WHERE parallax > %g AND ew_dib5780 IS NOT NULL" % plx_mas, timeout=900)
    far = core.tap("datacentral", "SELECT ew_dib5780 FROM galah_dr4.mainstartable WHERE parallax BETWEEN %g AND %g AND ew_dib5780 IS NOT NULL"
                                  % (far_plx_mas[0], far_plx_mas[1]), timeout=900)
    e = np.array([float(r["ew_dib5780"]) for r in near])
    f = np.array([float(r["ew_dib5780"]) for r in far])
    s = core.robust_sigma(e)
    hits = sorted([r for r in near if float(r["ew_dib5780"]) > max(sigma * s, ew_a)], key=lambda r: -float(r["ew_dib5780"]))
    print("%d stars within 50 pc with a fitted band: median %.4f A, robust scatter %.4f; at 0.5-1 kpc (%d stars) median %.4f; within 50 pc above "
          "5 sigma and 20 mA: %d %s" % (len(near), float(np.median(e)), s, len(far), float(np.median(f)) if len(f) else float("nan"), len(hits),
                                       [(r["gaiadr3_source_id"], r["ew_dib5780"], r["parallax"]) for r in hits[:10]]))
    if record:
        core.record_run(
            "galah_dib_within_50pc", "Interstellar bands within 50 pc", "GALAH DR4 (Data Central)", "stars within 50 pc with a fitted 5780 A band: %d" % len(near),
            os.path.abspath(__file__), "dibnear", len(near), "stars", dict(plx_mas=plx_mas, sigma=sigma, ew_a=ew_a, far_plx_mas=far_plx_mas), "the fit's scatter within 50 pc (%.4f A) and its "
            "median at 0.5-1 kpc (%.4f A)" % (s, float(np.median(f)) if len(f) else float("nan")), "a 5780 A band within 50 pc",
            "%d above 5 sigma and 20 mA, %d of them cooler than 4500 K, where the star's own lines fill the band" % (
                len(hits), sum(float(r["teff"] or 0) < 4500 for r in hits)),
            [dict(target="gaia_dr3:%s" % r["gaiadr3_source_id"], stat=float(r["ew_dib5780"]), unit="EW, A",
                  vet="stellar-blend" if float(r["teff"] or 0) < 4500 else None, note="%s mas, %.0f K" % (r["parallax"], float(r["teff"] or 0)))
             for r in hits])
        print("recorded")


spinbreak_DOC = r"""Stars spinning faster than they hold together: GALAH DR4 dwarfs (log g above 4, clean fits) whose v sin i exceeds the break-up speed sqrt(GM/R), 436.7 km/s (M/R)^1/2 in solar units, with GALAH's mass and the radius from its luminosity and temperature.

    python -I seti.py composition spinbreak [record]"""


def spinbreak_main(record=False):
    rows = galah_dwarfs("vsini, e_vsini, mass, lbol", "vsini IS NOT NULL AND mass IS NOT NULL AND lbol IS NOT NULL")
    out = []
    for r in rows:
        R = math.sqrt(float(r["lbol"])) * (5772.0 / float(r["teff"])) ** 2
        vb = 436.7 * math.sqrt(float(r["mass"]) / R)
        out.append((float(r["vsini"]) / vb, r, vb))
    out.sort(key=lambda x: -x[0])
    hits = [x for x in out if x[0] > 1]
    print("%d dwarfs with v sin i, mass and luminosity; above break-up %d; the fastest fractions %s" % (
        len(out), len(hits), [(x[1]["gaiadr3_source_id"], round(x[0], 3), x[1]["vsini"], round(x[2])) for x in out[:5]]))
    if record:
        core.record_run(
            "galah_breakup_rotation", "Stars spinning faster than they hold together", "GALAH DR4 (Data Central)",
            "clean dwarfs with v sin i, mass and luminosity: %d" % len(out), os.path.abspath(__file__), "spinbreak", len(out), "stars", dict(),
            "the break-up speed from each star's mass and radius", "a dwarf rotating above break-up", "%d above; the fastest at %.2f of break-up" % (
                len(hits), out[0][0] if out else 0), [dict(target="gaia_dr3:%s" % x[1]["gaiadr3_source_id"], stat=x[0], unit="v sin i over break-up")
                                                      for x in hits])
        print("recorded")


apogeebad_DOC = r"""Nearby dwarfs APOGEE could not fit: SDSS DR17 APOGEE stars (aspcapStar joined to apogeeStar on SkyServer) with Gaia EDR3 parallaxes above 10 mas, combined S/N above 100, Doppler-fit temperatures of 4000-6500 K and gravities above 4.0, whose ASPCAP flag carries STAR_BAD; a spectrum no stellar model fits at that S/N carries something stars do not. Each is explained by a second set of lines (MULTIPLE_SUSPECT, two or more cross-correlation components, or velocities scattering by 1 km/s), by rotation, by a temperature or gravity outside the calibrated range of the pipeline, or by its SIMBAD type.

    python -I seti.py composition apogeebad [record] [--plx-mas 10] [--snr 100] [--teff 4000 6500] [--logg-min 4] [--vsini-kms 20]"""

# ASPCAPFLAG bits of DR17
APOGEE_BITS = {16: "TEFF_BAD", 17: "LOGG_BAD", 18: "VMICRO_BAD", 19: "M_H_BAD", 20: "ALPHA_M_BAD", 23: "STAR_BAD", 24: "CHI2_BAD", 25: "COLORTE_BAD",
               26: "ROTATION_BAD", 27: "SN_BAD", 31: "NO_ASPCAP_RESULT", 33: "NO_GRID", 36: "FERRE_FAIL", 40: "PROBLEM_TARGET", 41: "MULTIPLE_SUSPECT"}


def apogeebad_main(record=False, plx_mas=10.0, snr=100.0, teff=[4000.0, 6500.0], logg_min=4.0, vsini_kms=20.0):
    rows = core.sdss_sql("SELECT a.apogee_id, s.gaiaedr3_source_id, s.gaiaedr3_parallax, s.snr, s.n_components, s.vscatter, s.nvisits, s.starflag, s.rv_teff, "
                         "s.rv_logg, s.rv_feh, a.aspcapflag, a.vsini, a.aspcap_chi2 FROM aspcapStar a JOIN apogeeStar s ON a.apstar_id = s.apstar_id "
                         "WHERE s.gaiaedr3_parallax > %g AND s.snr > %g AND s.rv_teff BETWEEN %g AND %g AND s.rv_logg > %g" % (plx_mas, snr, teff[0], teff[1], logg_min))
    bad = [r for r in rows if int(float(r["aspcapflag"])) >> 23 & 1]
    sim = {}
    ids = [r["gaiaedr3_source_id"] for r in bad if r["gaiaedr3_source_id"] not in ("", "0", "-1")]
    for k in range(0, len(ids), 200):
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % x for x in ids[k:k + 200])):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    out = []
    for r in bad:
        f = int(float(r["aspcapflag"]))
        bits = [n for b, n in sorted(APOGEE_BITS.items()) if f >> b & 1]
        s = sim.get(r["gaiaedr3_source_id"])
        vet = ("binary" if "MULTIPLE_SUSPECT" in bits or int(float(r["n_components"])) >= 2 or (int(r["nvisits"]) > 1 and float(r["vscatter"]) > 1) else
               "rotation" if "ROTATION_BAD" in bits or float(r["vsini"]) > vsini_kms else
               "fit-boundary" if {"TEFF_BAD", "LOGG_BAD", "NO_GRID", "COLORTE_BAD"} & set(bits) else
               "natural-type" if s and s[1] not in ("*", "PM*", "HV*") else None)
        out.append(dict(apogee=r["apogee_id"], gaia=r["gaiaedr3_source_id"], snr=float(r["snr"]), teff=float(r["rv_teff"]), feh=float(r["rv_feh"]),
                        ncomp=int(float(r["n_components"])), bits=bits, simbad=s, vet=vet))
    # Gaia's own sign of a companion: a wobble beyond a single star's (RUWE) or a radial velocity that swings
    rest = [o for o in out if o["vet"] is None and o["gaia"] not in ("", "0", "-1")]
    if rest:
        gaia = {r["source_id"]: r for r in core.tap("gaia", "SELECT source_id, ruwe, rv_amplitude_robust, radial_velocity_error FROM gaiadr3.gaia_source "
                                                                  "WHERE source_id IN (%s)" % ",".join(o["gaia"] for o in rest))}
        for o in rest:
            g = gaia.get(o["gaia"])
            if g and ((core.num(g["ruwe"]) or 0) > 1.4 or (core.num(g["rv_amplitude_robust"]) or 0) > 5 * max(core.num(g["radial_velocity_error"]) or 1, 1)):
                o["vet"] = "binary"
                o["bits"] = o["bits"] + ["Gaia RUWE %s, RV amplitude %s km/s" % (g["ruwe"], g["rv_amplitude_robust"])]
    left = [o for o in out if o["vet"] is None]
    from collections import Counter
    print("%d nearby dwarfs at S/N above 100; %d STAR_BAD, explained %s; unexplained %d %s" % (
        len(rows), len(out), dict(Counter(o["vet"] for o in out)), len(left), [(o["apogee"], o["gaia"], o["snr"], o["teff"], o["feh"], o["bits"], o["simbad"]) for o in left]))
    if record:
        core.record_run("apogee_star_bad_dwarfs", "Nearby dwarfs APOGEE could not fit", "SDSS DR17 APOGEE (aspcapStar, apogeeStar) on SkyServer; SIMBAD",
                        "dwarfs with parallax above 10 mas, S/N above 100, 4000-6500 K, log g above 4.0: %d" % len(rows), os.path.abspath(__file__), "apogeebad",
                        len(rows), "stars", dict(plx_mas=plx_mas, snr=snr, teff=teff, logg=logg_min, vsini_kms=vsini_kms, flag="STAR_BAD"), "none: each failure read for its cause",
                        "a high-S/N dwarf spectrum no stellar model fits", "%d STAR_BAD (%s); %d unexplained" % (
                            len(out), ", ".join("%s %d" % (k, v) for k, v in Counter(o["vet"] or "none" for o in out).most_common()), len(left)),
                        [dict(target="gaia_dr3:%s" % o["gaia"] if o["gaia"] not in ("", "0", "-1") else "apogee:%s" % o["apogee"], stat=o["snr"], unit="S/N",
                              vet=o["vet"], note="%s; %.0f K, [Fe/H] %.2f, %d components; %s; SIMBAD %s" % (o["apogee"], o["teff"], o["feh"], o["ncomp"],
                                                                                                   " ".join(o["bits"]), o["simbad"])) for o in out])
        print("recorded")


przybylski_DOC = r"""Przybylski's star re-measured: HD 101065, the one star with published lines of promethium and the short-lived actinides (Cowley et al. 2004; Gopka et al. 2008), in the six highest-S/N HARPS spectra and the four highest-S/N UVES spectra of the ESO archive. Each instrument's envelope-normalised median stack, put in the star's rest frame, is read at the toolkit's Tc I, Pm II, Pa II, Pu II, Am II and Cm II lines for the weighted depth at the lines, against the same measure at 2000 random line sets of the same size drawn from the covered range, which in this star's forest of rare-earth lines is the chance of coincidences; a true identification sits deeper than chance. A set deeper than chance is read line by line in NIST's line list for the star's own species within 0.08 A and scored again without the blended lines.

    python -I seti.py composition przybylski [record] [--n-harps 6] [--n-uves 4] [--draws 2000] [--blend-tol-a 0.08]"""

PRZYBYLSKI = dict(ra=174.4043373, dec=-46.7096883, teff=6600.0)
# the species that fill this star's spectrum: the iron peak, the lanthanides singly and doubly ionised, thorium and uranium
PRZYBYLSKI_SPECIES = ("Fe I;Fe II;Ti II;Cr II;V II;Mn II;Ni I;Sr II;Y II;Zr II;Ba II;La II;Ce II;Pr II;Nd II;Sm II;Eu II;Gd II;Tb II;Dy II;Ho II;Er II;Tm II;Yb II;"
                      "Lu II;Hf II;Th II;U II;Ce III;Pr III;Nd III;Sm III;Eu III;Gd III;Tb III;Dy III;Ho III;Er III;Tm III;Yb III")


def przybylski_blends(lam, tol=0.08):
    """NIST ASD lines of the star's own species within tol A of a wavelength (air): (species, wavelength)."""
    import html as htmllib
    q = urllib.parse.urlencode(dict(spectra=PRZYBYLSKI_SPECIES, limits_type=0, low_w=round(lam - tol, 3), upp_w=round(lam + tol, 3), unit=0, submit="Retrieve Data", de=0,
                                    format=1, line_out=0, en_unit=0, output=0, bibrefs=1, page_size=15, show_obs_wl=1, show_calc_wl=1, unc_out=1, order_out=0, max_low_enrg="",
                                    show_av=2, max_upp_enrg="", tsb_value=0, min_str="", A_out=0, intens_out="on", max_str="", allowed_out=1, forbid_out=1, min_accur="",
                                    min_intens="", conf_out="on", term_out="on", enrg_out="on", J_out="on"))
    body = htmllib.unescape(re.sub(r"<[^>]+>", "", core.get("https://physics.nist.gov/cgi-bin/ASD/lines1.pl?" + q, timeout=60).decode("utf-8", "replace")))
    out = []
    for ln in body.splitlines():
        m = re.match(r"\s*([A-Z][a-z]? I{1,3})\s*\|(.*)", ln)
        if m:
            w = re.search(r"(\d{4}\.\d+)", m.group(2))
            out.append((m.group(1), float(w.group(1)) if w else None))
    return out


def przybylski_stack(inst, n_spec):
    """The star's envelope-normalised median stack from its highest-S/N spectra of the instrument, in its rest frame."""
    d = os.path.join(elements_ROOT, inst, "HD101065")
    os.makedirs(d, exist_ok=True)
    found = sorted(core.eso_spectra(PRZYBYLSKI["ra"], PRZYBYLSKI["dec"], inst, covers=(4.22e-7, 4.31e-7)), key=lambda x: -float(x["snr"] or 0))[:n_spec]
    N = []
    for x in found:
        path = os.path.join(d, x["dp_id"].replace(":", "_") + ".fits")
        if os.path.exists(path) or core.eso_download(x["dp_id"], path):
            f = elements_load(inst, path)
            if np.isfinite(f).mean() >= 0.15:
                N.append(f / science.envelope(f, block=300, q=92.0, smooth=1))
    T = np.nanmedian(np.array(N), axis=0)
    v = science.line_velocity(T, elements_GRID)
    return science.to_rest(T, elements_GRID, v) if np.isfinite(v) else T, len(N), v


def przybylski_main(record=False, n_harps=6, n_uves=4, draws=2000, blend_tol_a=0.08):
    rng = np.random.default_rng(101065)
    out = []
    for inst, n_spec in (("HARPS", n_harps), ("UVES", n_uves)):
        T, n, v = przybylski_stack(inst, n_spec)
        R = elements_RES[inst]
        cov = elements_GRID[np.isfinite(T)]
        lo, hi = cov.min() + 10, cov.max() - 10
        depth = lambda lam: 1.0 - elements_local_flux(T, lam, R)
        # the star's line forest: the depth at random wavelengths
        forest = np.array([depth(c) for c in rng.uniform(lo, hi, draws)])
        forest = forest[np.isfinite(forest)]
        for el in elements_NAMES:
            lines = [(lam, w) for lam, w in elements_line_set(el, PRZYBYLSKI["teff"]) if lo <= lam <= hi]
            if not lines:
                continue
            w = np.array([x for _, x in lines])
            d = np.array([depth(lam) for lam, _ in lines])
            ok = np.isfinite(d)
            score = float(np.dot(w[ok], d[ok]) / np.sqrt(np.sum(w[ok] ** 2)))
            null = np.array([np.dot(w[ok], rng.choice(forest, ok.sum())) / np.sqrt(np.sum(w[ok] ** 2)) for _ in range(draws)])
            p = float(np.mean(null >= score))
            row = dict(inst=inst, el=el, n_lines=int(ok.sum()), n_spectra=n, score=round(score, 3), p=p, depths=[round(float(x), 3) for x in d],
                       forest=round(float(np.mean(forest > 0.2)), 3), blends={})
            # a set deeper than chance is read line by line against the star's own species: the score again without the lines they blend
            if p < 0.05:
                blended = {}
                for lam, _ in lines:
                    hits = przybylski_blends(lam, blend_tol_a)
                    if hits:
                        blended[lam] = hits[:4]
                row["blends"] = {str(k): v for k, v in blended.items()}
                free = np.array([lam not in blended for lam, _ in lines]) & ok
                if free.any():
                    s2 = float(np.dot(w[free], d[free]) / np.sqrt(np.sum(w[free] ** 2)))
                    null2 = np.array([np.dot(w[free], rng.choice(forest, free.sum())) / np.sqrt(np.sum(w[free] ** 2)) for _ in range(draws)])
                    row["n_free"], row["score_free"], row["p_free"] = int(free.sum()), round(s2, 3), float(np.mean(null2 >= s2))
            out.append(row)
    print("Przybylski's star: %s" % "; ".join("%s %s %d lines in %d spectra, weighted depth %.3f, p %.3f" % (o["inst"], o["el"], o["n_lines"], o["n_spectra"], o["score"], o["p"])
                                               for o in out))
    for o in out:
        print("  %s %s: depths %s (random wavelengths deeper than 0.2: %.0f per cent)%s" % (o["inst"], o["el"], o["depths"], 100 * o["forest"],
              "; blends %s; without them %d lines, depth %s, p %s" % (o["blends"], o.get("n_free", 0), o.get("score_free"), o.get("p_free")) if o["blends"] else ""))
    if record:
        low = min(out, key=lambda o: o["p"])
        core.record_run("przybylski_actinides", "Przybylski's star re-measured", "ESO archive, HARPS and UVES Phase 3 spectra of HD 101065",
                        "the six highest-S/N HARPS and four highest-S/N UVES spectra covering 4220-4310 A", os.path.abspath(__file__), "przybylski", 1, "stars",
                        dict(teff=PRZYBYLSKI["teff"], draws=draws, spectra=dict(HARPS=n_harps, UVES=n_uves), blend_tol_a=blend_tol_a,
                             lines={o["el"]: o["n_lines"] for o in out if o["inst"] == "HARPS"}),
                        "%d random line sets of the same size from the same stack: the coincidence rate of the star's line forest" % draws,
                        "absorption at the lines of promethium and the actinides beyond the forest's coincidences",
                        "lowest p %.3f (%s %s); %s" % (low["p"], low["inst"], low["el"], ", ".join("%s %s p %.3f%s" % (o["inst"], o["el"], o["p"],
                                                                                                    " (%.3f without blended lines)" % o["p_free"] if "p_free" in o else "") for o in out)),
                        [dict(target="hd:101065", stat=o["score"], p=o["p"], unit="weighted line depth, %s %s" % (o["inst"], o["el"]),
                              vet="chance-rate" if o["p"] > 0.01 else "stellar-blend" if o.get("p_free", 0) > 0.01 else None,
                              note="%d lines, %d spectra; depths %s; blends %s; without them p %s" % (o["n_lines"], o["n_spectra"], o["depths"], o["blends"], o.get("p_free"))) for o in out])
        print("recorded")


haquiet_DOC = r"""H-alpha emission at quiet Sun-like neighbours: Gaia DR3's emission-line classifier (ESP-ELS) marks the stars whose BP/RP spectra show H-alpha in emission; those within 100 pc at BP-RP 0.5-1.5, the colours of F, G and K dwarfs, each read for the signs of an ordinary cause: an eRASS1 X-ray source within 15 arcsec (a corona), a Gaia line broadening above 15 km/s (a fast rotator), RUWE above 1.4 or a non-single-star solution (a binary), a SIMBAD type of a young, active or interacting star, and a place more than 2 mag under the main sequence of its colour (a white dwarf's light on a red dwarf's). Emission on a slow, quiet, single Sun-like star has no chromosphere to come from.

    python -I seti.py composition haquiet [record] [--plx-mas 10] [--colour 0.5 1.5] [--radius-arcsec 15] [--vbroad-kms 15] [--ruwe-max 1.4] [--under-mag 2]"""


def haquiet_main(record=False, plx_mas=10.0, colour=[0.5, 1.5], radius_arcsec=15.0, vbroad_kms=15.0, ruwe_max=1.4, under_mag=2.0):
    rows = core.gaia_by_range("SELECT g.source_id, g.ra, g.dec, g.parallax, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.non_single_star, g.vbroad, a.classlabel_espels, "
                              "a.ew_espels_halpha, a.ew_espels_halpha_uncertainty, a.teff_gspphot FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a "
                              "ON g.source_id = a.source_id WHERE g.source_id BETWEEN {lo} AND {hi} AND g.parallax > %g AND a.classlabel_espels IS NOT NULL "
                              "AND g.bp_rp BETWEEN %g AND %g" % (plx_mas, colour[0], colour[1]), n=24, maxrec=20000)
    from setitools.catalogues import shifted_matches
    pos = [dict(name=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"])) for r in rows]
    xr = {m["source_id"] for m in shifted_matches(pos, "J/A+A/682/A34/erass1-m", radius_arcsec)[0]} if pos else set()
    sim = {}
    if rows:
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % r["source_id"] for r in rows)):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    active = ("BY*", "RS*", "Er*", "Fl*", "TT*", "TT?", "Y*O", "Y*?", "EB*", "SB*", "**", "CV*", "Sy*", "Em*", "Ro*", "El*", "pr*", "Or*", "Ae*", "X", "AB*", "RG*")
    out = []
    for r in rows:
        s = sim.get(r["source_id"])
        # a star more than 2 mag under the main sequence of its colour is a white dwarf's light added to a red dwarf's, whose emission is the red dwarf's
        mg = float(r["phot_g_mean_mag"]) + 5 * math.log10(float(r["parallax"]) / 100)
        under = mg - (2.0 + 3.8 * float(r["bp_rp"]))
        vet = ("active-star" if r["source_id"] in xr else "binary" if (core.num(r["ruwe"]) or 0) > ruwe_max or (core.num(r["non_single_star"]) or 0) > 0 else
               "rotation" if (core.num(r["vbroad"]) or 0) > vbroad_kms else "natural-type" if s and s[1] in active else "white-dwarf" if under > under_mag else None)
        out.append(dict(id=r["source_id"], g=core.num(r["phot_g_mean_mag"]), bprp=core.num(r["bp_rp"]), mg=round(mg, 2), teff=core.num(r["teff_gspphot"]), cls=r["classlabel_espels"],
                        ew=core.num(r["ew_espels_halpha"]), e_ew=core.num(r["ew_espels_halpha_uncertainty"]), ruwe=core.num(r["ruwe"]), vbroad=core.num(r["vbroad"]), xray=r["source_id"] in xr,
                        simbad=s, vet=vet))
    left = [o for o in out if o["vet"] is None]
    print("%d emission-line stars within 100 pc at BP-RP 0.5-1.5; unexplained %d" % (len(out), len(left)))
    for o in out:
        print("  Gaia DR3 %s: G %.1f, M_G %.1f, BP-RP %.2f, %s K, %s, H-alpha EW %s +- %s A, RUWE %s, vbroad %s, X-ray %s; SIMBAD %s; %s" % (
            o["id"], o["g"], o["mg"], o["bprp"], o["teff"], o["cls"], o["ew"], o["e_ew"], o["ruwe"], o["vbroad"], o["xray"], o["simbad"], o["vet"]))
    if record:
        core.record_run("gaia_halpha_quiet_dwarfs", "H-alpha emission at quiet Sun-like neighbours", "Gaia DR3 ESP-ELS classes and gaia_source; eRASS1 by CDS XMatch; SIMBAD",
                        "emission-line stars within 100 pc at BP-RP 0.5-1.5: %d" % len(out), os.path.abspath(__file__), "haquiet", len(out), "stars",
                        dict(plx_mas=plx_mas, colour=colour, radius_arcsec=radius_arcsec, vbroad_kms=vbroad_kms, ruwe=ruwe_max, under_mag=under_mag),
                        "none: each star read for a corona, rotation, a companion and its type",
                        "H-alpha in emission on a slow, quiet, single Sun-like star", "%d stars, %d unexplained %s" % (len(out), len(left), [o["id"] for o in left]),
                        [dict(target="gaia_dr3:%s" % o["id"], stat=o["ew"], unit="H-alpha equivalent width, A (negative in emission)", vet=o["vet"],
                              note="%s K, %s; RUWE %s, vbroad %s, X-ray %s; SIMBAD %s" % (o["teff"], o["cls"], o["ruwe"], o["vbroad"], o["xray"], o["simbad"])) for o in out])
        print("recorded")


evenheavy_DOC = r"""Stars enriched evenly in every heavy element: GALAH DR4 stars with clean measurements of all eleven neutron-capture elements Sr, Y, Zr, Mo, Ru, Ba, La, Ce, Nd, Sm and Eu, those whose [X/Fe] stand above +0.4 dex in every one with a spread below 0.1 dex across the eleven, a dose of heavy elements in one flat measure, which neither the s-process (humps at Ba and Sr) nor the r-process (Eu over Ba) gives; the stars depleted as evenly are the mirror null, and the spread the abundance errors alone give is read from the whole sample.

    python -I seti.py composition evenheavy [record] [--mean-min 0.4] [--spread-max 0.1]"""


def evenheavy_main(record=False, mean_min=0.4, spread_max=0.1):
    p = core.data("galah_dr4_ncapture.json")
    if not os.path.exists(p):
        fission_main()
    rows = json.load(open(p))

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan
    X = np.array([[f(r[e + "_fe"]) for e in fission_ELS] for r in rows])
    F = np.array([[f(r["flag_" + e + "_fe"]) for e in fission_ELS] for r in rows])
    E = np.array([[f(r["e_" + e + "_fe"]) for e in fission_ELS] for r in rows])
    ok = np.all(np.isfinite(X) & (F == 0), axis=1)
    mean, spread = np.nanmean(X, axis=1), np.nanstd(X, axis=1)
    err = np.sqrt(np.nanmean(E ** 2, axis=1))
    rich = ok & (mean > mean_min) & (spread < spread_max)
    poor = ok & (mean < -mean_min) & (spread < spread_max)
    anyrich = ok & (mean > mean_min)
    print("%d stars, %d with all eleven elements clean; spread across the elements median %.3f against a typical error of %.3f; enriched above +%g with a spread below %g: %d "
          "of %d so enriched; depleted as evenly: %d" % (len(rows), int(ok.sum()), float(np.median(spread[ok])) if ok.any() else 0, float(np.median(err[ok])) if ok.any() else 0,
                                                       mean_min, spread_max, int(rich.sum()), int(anyrich.sum()), int(poor.sum())))
    out = []
    for i in np.nonzero(rich)[0]:
        r = rows[i]
        out.append(dict(sobject=str(r["sobject_id"]), gaia=str(r["gaiadr3_source_id"]), mean=float(mean[i]), spread=float(spread[i]), teff=f(r["teff"]), logg=f(r["logg"]),
                        feh=f(r["fe_h"]), pattern={e: round(float(X[i, j]), 2) for j, e in enumerate(fission_ELS)}, vet=None))
        print("  %s Gaia DR3 %s: mean %+.2f, spread %.3f, Teff %.0f, log g %.2f, [Fe/H] %+.2f; %s" % (out[-1]["sobject"], out[-1]["gaia"], out[-1]["mean"], out[-1]["spread"],
                                                                                                  out[-1]["teff"], out[-1]["logg"], out[-1]["feh"], out[-1]["pattern"]))
    json.dump(dict(n=int(ok.sum()), hits=out, poor=int(poor.sum())), open(core.results("even_heavy_enrichment.json"), "w"), indent=1)
    if record:
        core.record_run(
            "galah_even_heavy", "Stars enriched evenly in every heavy element", "GALAH DR4 mainstartable (Data Central)",
            "stars with all eleven neutron-capture elements clean: %d" % int(ok.sum()), os.path.abspath(__file__), "evenheavy", int(ok.sum()), "stars",
            dict(mean_min=mean_min, spread_max=spread_max, elements=list(fission_ELS)),
            "the stars depleted as evenly: %d; the sample's spread across the elements, median %.3f, against a typical error of %.3f" % (
                int(poor.sum()), float(np.median(spread[ok])) if ok.any() else 0, float(np.median(err[ok])) if ok.any() else 0),
            "every heavy element raised by the same %g dex or more to within %g dex" % (mean_min, spread_max),
            "%d of %d stars enriched above +%g carry a flat pattern; %d depleted as evenly" % (int(rich.sum()), int(anyrich.sum()), mean_min, int(poor.sum())),
            [dict(target="gaia_dr3:%s" % o["gaia"], stat=o["spread"], unit="spread of [X/Fe] across eleven heavy elements, dex", vet=o["vet"],
                  note="mean %+.2f, Teff %.0f, log g %.2f, [Fe/H] %+.2f; %s" % (o["mean"], o["teff"], o["logg"], o["feh"], o["pattern"])) for o in out if o["gaia"] not in ("", "None")])
        print("recorded")


ceriumonly_DOC = r"""Cerium without carbon in dwarfs: the unevolved metal-rich APOGEE DR17 dwarfs of the CNO test (log g above 4.2, Teff 4800-6500 K, [Fe/H] above -0.3, S/N above 70) with unflagged carbon and cerium, those with [Ce/Fe] above +0.5 dex while [C/Fe] stays at or below zero; the s-process that makes cerium in an asymptotic-giant companion carries carbon with it, so a dwarf dosed with the heavy element alone has it from elsewhere. Each read for a Gaia binary (RUWE, a non-single-star solution or radial-velocity variability), its Teff edge and its rotation, against the dwarfs with cerium and carbon together.

    python -I seti.py composition ceriumonly [record] [--ce-min 0.5] [--c-max 0.0]"""


def ceriumonly_main(record=False, ce_min=0.5, c_max=0.0):
    from setitools.core import num
    rows = cno_fetch()
    keep = []
    for r in rows:
        c, ce, te = num(r["[C/Fe]"]), num(r["[Ce/Fe]"]), num(r["Teff"])
        if None in (c, ce, te) or (r["f_[C/Fe]"] or "0").strip() not in ("0", "") or (r["f_[Ce/Fe]"] or "0").strip() not in ("0", ""):
            continue
        keep.append(dict(id=r["APOGEE"], teff=te, logg=num(r["logg"]), feh=num(r["[Fe/H]"]), c=c, ce=ce, n=num(r["[N/Fe]"]), vsini=num(r["Vsini"]), gaia=r["GaiaEDR3"],
                         srv=num(r["s_HRV"]), nvis=num(r["Nvis"])))
    rich = [k for k in keep if k["ce"] > ce_min]
    alone = [k for k in rich if k["c"] <= c_max]
    together = [k for k in rich if k["c"] > c_max]
    gb = cno_gaia_binary([k["gaia"] for k in alone]) if alone else {}
    for k in alone:
        b = gb.get(k["gaia"])
        k["binary"] = b
        k["vet"] = ("binary" if b is not None and b[0] else "binary" if k["srv"] is not None and k["nvis"] and k["nvis"] > 1 and k["srv"] > 1.0 else
                    "fit-boundary" if k["teff"] < 4900 else "rotation" if k["vsini"] and k["vsini"] > 15 else None)
    left = [k for k in alone if k["vet"] is None]
    cs = np.array([k["c"] for k in rich])
    print("%d dwarfs with unflagged carbon and cerium; [Ce/Fe] above %+g: %d, their [C/Fe] median %+.2f; with [C/Fe] at or below %+g: %d, %d Gaia or velocity binaries, "
          "%d at the Teff edge, %d fast rotators; unexplained %d" % (len(keep), ce_min, len(rich), float(np.median(cs)) if len(cs) else 0, c_max, len(alone),
                                                                   sum(k["vet"] == "binary" for k in alone), sum(k["vet"] == "fit-boundary" for k in alone),
                                                                   sum(k["vet"] == "rotation" for k in alone), len(left)))
    for k in sorted(alone, key=lambda k: -k["ce"]):
        print("  %s: [Ce/Fe] %+.2f, [C/Fe] %+.2f, [N/Fe] %s, Teff %.0f, log g %.2f, [Fe/H] %+.2f, vsini %s, RV scatter %s; Gaia %s; %s" % (
            k["id"], k["ce"], k["c"], k["n"], k["teff"], k["logg"], k["feh"], k["vsini"], k["srv"], (k["binary"] or (None, "-"))[1], k["vet"] or "unexplained"))
    json.dump(dict(n=len(keep), rich=len(rich), alone=alone), open(core.results("cerium_without_carbon.json"), "w"), indent=1)
    if record:
        core.record_run(
            "apogee_cerium_only", "Cerium without carbon in dwarfs", "APOGEE DR17 (VizieR III/286); Gaia DR3",
            "unevolved metal-rich dwarfs with unflagged carbon and cerium: %d" % len(keep), os.path.abspath(__file__), "ceriumonly", len(keep), "dwarfs",
            dict(ce_min=ce_min, c_max=c_max), "the %d dwarfs with [Ce/Fe] above %+g carry [C/Fe] of %+.2f at the median; %d hold cerium and carbon together" % (
                len(rich), ce_min, float(np.median(cs)) if len(cs) else 0, len(together)),
            "a dwarf with [Ce/Fe] above %+g and [C/Fe] at or below %+g" % (ce_min, c_max),
            "%d: %d binaries, %d at the Teff edge, %d fast rotators; %d unexplained" % (len(alone), sum(k["vet"] == "binary" for k in alone),
                                                                                       sum(k["vet"] == "fit-boundary" for k in alone), sum(k["vet"] == "rotation" for k in alone),
                                                                                       len(left)),
            [dict(target="2mass:%s" % k["id"].replace("2M", ""), stat=k["ce"], unit="[Ce/Fe]", vet=k["vet"],
                  note="[C/Fe] %+.2f, [N/Fe] %s, Teff %.0f, log g %.2f, [Fe/H] %+.2f, vsini %s, RV scatter %s, Gaia %s" % (
                      k["c"], k["n"], k["teff"], k["logg"], k["feh"], k["vsini"], k["srv"], (k["binary"] or (None, "-"))[1])) for k in alone])
        print("recorded")


rvjump_DOC = r"""Stars that changed velocity between GALAH and Gaia: GALAH DR4 stars with a clean fit whose spectroscopic radial velocity (rv_comp_1) and Gaia DR3's differ by more than 30 km/s with both measured to 2 km/s, a star whose line-of-sight speed changed by more than any single-star process allows between the surveys' epochs, which an orbit about a companion does and a driven thing would; each read for a second spectral component in GALAH, for a Gaia non-single-star orbit, radial-velocity amplitude, RUWE and image peaks, for its rotation, since a broad-lined star gives both surveys a poor velocity, for a GALAH temperature far from Gaia's, which marks a fibre on a different star, for a Gaia velocity resting on few transits, and for a third epoch in RAVE DR6 or LAMOST DR5 apart from either, which settles a velocity that moves.

    python -I seti.py composition rvjump [record] [--jump-kms 30] [--err-max-kms 2] [--vsini-max 50] [--teff-mismatch-k 600] [--transits-min 5] [--third-kms 10]"""


def rvjump_main(record=False, jump_kms=30.0, err_max_kms=2.0, vsini_max=50.0, teff_mismatch_k=600.0, transits_min=5, third_kms=10.0):
    from setitools.core import num
    rows = core.tap_async("datacentral", "SELECT sobject_id, gaiadr3_source_id, ra, dec, teff, logg, vsini, rv_comp_1, e_rv_comp_1, rv_comp_2, rv_gaia_dr3, e_rv_gaia_dr3 FROM "
                                         "galah_dr4.mainstartable WHERE flag_sp = 0 AND rv_comp_1 IS NOT NULL AND rv_gaia_dr3 IS NOT NULL AND e_rv_comp_1 < %g AND e_rv_gaia_dr3 < %g" % (
                                             err_max_kms, err_max_kms), maxrec=1000000)
    n_all = len(rows)
    st = {}
    for r in rows:
        rv, grv = num(r["rv_comp_1"]), num(r["rv_gaia_dr3"])
        gid = str(r["gaiadr3_source_id"] or "").strip()
        if rv is None or grv is None or not gid or abs(rv - grv) < jump_kms:
            continue
        if gid not in st or abs(rv - grv) > abs(st[gid]["rv"] - st[gid]["grv"]):
            st[gid] = dict(id=gid, sobject=str(r["sobject_id"]), rv=rv, erv=num(r["e_rv_comp_1"]), grv=grv, egrv=num(r["e_rv_gaia_dr3"]), rv2=num(r["rv_comp_2"]), teff=num(r["teff"]),
                           logg=num(r["logg"]), vsini=num(r["vsini"]))
    st = list(st.values())
    nss, amp = {}, {}
    for k in range(0, len(st), 500):
        ids = ",".join(s["id"] for s in st[k:k + 500])
        for r in core.tap("gaia", "SELECT source_id, nss_solution_type FROM gaiadr3.nss_two_body_orbit WHERE source_id IN (%s)" % ids):
            nss.setdefault(str(r["source_id"]), []).append(str(r["nss_solution_type"]))
        for r in core.tap("gaia", "SELECT source_id, rv_amplitude_robust, rv_nb_transits, rv_chisq_pvalue, non_single_star, ruwe, ipd_frac_multi_peak, teff_gspphot FROM "
                                  "gaiadr3.gaia_source WHERE source_id IN (%s)" % ids):
            amp[str(r["source_id"])] = (num(r["rv_amplitude_robust"]), num(r["rv_nb_transits"]), num(r["rv_chisq_pvalue"]), num(r["non_single_star"]), num(r["ruwe"]),
                                        num(r["ipd_frac_multi_peak"]), num(r["teff_gspphot"]))
    pos = {}
    for k in range(0, len(st), 500):
        for r in core.tap("gaia", "SELECT source_id, ra, dec FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(s["id"] for s in st[k:k + 500])):
            pos[str(r["source_id"])] = (float(r["ra"]), float(r["dec"]))
    third = {}
    up = [dict(source_id=i, ra=p[0], dec=p[1]) for i, p in pos.items()]
    for cat, key, ekey, label in (("III/283/ravedr6", "HRV", "e_HRV", "RAVE DR6"), ("V/164/dr5", "rv", "e_rv", "LAMOST DR5")):
        for m in core.xmatch(up, cat, 2.0):
            v = num(m.get(key))
            if v is not None:
                third.setdefault(m["source_id"], []).append((label, v, num(m.get(ekey))))
    for s in st:
        s["nss"] = nss.get(s["id"], [])
        s["amp"] = amp.get(s["id"])
        s["third"] = third.get(s["id"], [])
        s["third_apart"] = any((e or 0) < third_kms and (abs(v - s["rv"]) > third_kms + 3 * (e or 0) or abs(v - s["grv"]) > third_kms + 3 * (e or 0)) for lab, v, e in s["third"])
        a = s["amp"] or (None, None, None, None, None, None, None)
        s["ruwe"], s["gaia_teff"] = a[4], a[6]
        s["vet"] = ("binary" if s["rv2"] is not None or s["nss"] or (a[3] or 0) > 0 or (a[0] or 0) > jump_kms / 2 or (a[2] is not None and a[2] < 0.01) or (a[4] or 0) > 1.4
                    or (a[5] or 0) > 2 or s["third_apart"] else "artefact" if (s["vsini"] or 0) > vsini_max else
                    "misidentified" if a[6] is not None and s["teff"] is not None and abs(a[6] - s["teff"]) > teff_mismatch_k else
                    "single-epoch" if (a[1] or 0) < transits_min else None)
    left = [s for s in st if s["vet"] is None]
    print("%d GALAH stars with both velocities to %g km/s; apart by more than %g km/s: %d (%d binaries by a second component, a Gaia orbit, Gaia's amplitude, variability flag, RUWE "
          "or image peaks, %d broad-lined, %d with temperatures apart by %g K, %d on fewer than %d Gaia transits); unexplained %d" % (
              n_all, err_max_kms, jump_kms, len(st), sum(s["vet"] == "binary" for s in st), sum(s["vet"] == "artefact" for s in st), sum(s["vet"] == "misidentified" for s in st),
              teff_mismatch_k, sum(s["vet"] == "single-epoch" for s in st), transits_min, len(left)))
    for s in sorted(st, key=lambda s: -abs(s["rv"] - s["grv"])):
        if s["vet"] != "binary" or len(st) <= 40:
            print("  Gaia DR3 %s: GALAH %.1f +- %.1f, Gaia %.1f +- %.1f km/s, third epochs %s, second component %s, Gaia orbits %s, amplitude %s, vsini %s, %s K (Gaia %s); %s" % (
                s["id"], s["rv"], s["erv"], s["grv"], s["egrv"], s["third"] or "-", s["rv2"], s["nss"], s["amp"], s["vsini"], s["teff"], s["gaia_teff"], s["vet"] or "unexplained"))
    json.dump(dict(n=n_all, hits=st), open(core.results("rv_jumps_galah_gaia.json"), "w"), indent=1)
    if record:
        core.record_run(
            "galah_gaia_rv_jumps", "Stars that changed velocity between GALAH and Gaia", "GALAH DR4 mainstartable (Data Central); Gaia DR3 gaia_source and nss_two_body_orbit",
            "GALAH stars with a clean fit and both velocities to %g km/s: %d" % (err_max_kms, n_all), os.path.abspath(__file__), "rvjump", n_all, "stars",
            dict(jump_kms=jump_kms, err_max_kms=err_max_kms, vsini_max=vsini_max, teff_mismatch_k=teff_mismatch_k, transits_min=transits_min, third_kms=third_kms),
            "GALAH's second components, Gaia's orbits, radial-velocity amplitudes, variability flags, RUWE and image peaks; the two surveys' temperatures; Gaia's transit counts; a third epoch from RAVE DR6 or LAMOST DR5",
            "a radial velocity changed by more than %g km/s between the surveys" % jump_kms, "%d: %d binaries, %d broad-lined, %d with temperatures apart, %d on few Gaia transits; "
            "%d unexplained" % (len(st), sum(s["vet"] == "binary" for s in st), sum(s["vet"] == "artefact" for s in st), sum(s["vet"] == "misidentified" for s in st),
                                sum(s["vet"] == "single-epoch" for s in st), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["rv"] - s["grv"], 1), unit="km/s, GALAH minus Gaia", vet=s["vet"],
                  note="GALAH %.1f +- %.1f, Gaia %.1f +- %.1f, third epochs %s, second component %s, orbits %s, amplitude %s, vsini %s" % (
                      s["rv"], s["erv"], s["grv"], s["egrv"], s["third"] or "none", s["rv2"], s["nss"], (s["amp"] or (None,))[0], s["vsini"])) for s in st])
        print("recorded")


europium_DOC = r"""Europium far above iron in dwarfs: GALAH DR4 dwarfs (log g above 4.0) with a clean fit, a clean europium abundance measured to 0.2 dex and a spectrum of signal-to-noise above 40, those with [Eu/Fe] above 1.0, the r-process enrichment of the rarest halo stars found in the disc, where a star's own gas never carried it and a refinery of the heaviest elements would; each read for its metallicity, since the r-process-enhanced stars of the halo are metal-poor, for a second spectral component and a Gaia non-single-star flag, since a companion's transfer enriches, and for barium and lanthanum, since an s-process pattern marks that transfer.

    python -I seti.py composition europium [record] [--eu-min 1.0] [--err-max 0.2] [--snr-min 40] [--logg-min 4] [--metal-poor -1.0] [--ba-s 0.5]"""


def europium_main(record=False, eu_min=1.0, err_max=0.2, snr_min=40.0, logg_min=4.0, metal_poor=-1.0, ba_s=0.5):
    from setitools.core import num
    rows = core.tap_async("datacentral", "SELECT sobject_id, gaiadr3_source_id, teff, logg, fe_h, eu_fe, e_eu_fe, ba_fe, flag_ba_fe, la_fe, flag_la_fe, rv_comp_2, snr_px_ccd3 FROM "
                                         "galah_dr4.mainstartable WHERE flag_sp = 0 AND flag_eu_fe = 0 AND logg > %g AND snr_px_ccd3 > %g AND e_eu_fe < %g" % (logg_min, snr_min, err_max),
                          maxrec=1000000)
    n_all = len(rows)
    st = [dict(id=str(r["gaiadr3_source_id"] or "").strip(), sobject=str(r["sobject_id"]), teff=num(r["teff"]), logg=num(r["logg"]), feh=num(r["fe_h"]), eu=num(r["eu_fe"]), eeu=num(r["e_eu_fe"]),
               ba=num(r["ba_fe"]) if num(r["flag_ba_fe"]) == 0 else None, la=num(r["la_fe"]) if num(r["flag_la_fe"]) == 0 else None, rv2=num(r["rv_comp_2"]),
               snr=num(r["snr_px_ccd3"])) for r in rows]
    hits = [s for s in st if s["eu"] is not None and s["eu"] - s["eeu"] > eu_min]
    nss = {}
    ids = [s["id"] for s in hits if s["id"]]
    for k in range(0, len(ids), 500):
        for r in core.tap("gaia", "SELECT source_id, non_single_star, ruwe FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(ids[k:k + 500])):
            nss[str(r["source_id"])] = (num(r["non_single_star"]), num(r["ruwe"]))
    for s in hits:
        s["gaia"] = nss.get(s["id"])
        g = s["gaia"] or (0, 0)
        s["vet"] = ("metal-poor" if s["feh"] is not None and s["feh"] < metal_poor else
                    "binary" if s["rv2"] is not None or (g[0] or 0) > 0 or (g[1] or 0) > 1.4 else
                    "abundance-pattern" if (s["ba"] is not None and s["ba"] > ba_s) or (s["la"] is not None and s["la"] > ba_s) else None)
    left = [s for s in hits if s["vet"] is None]
    print("%d GALAH dwarfs with a clean europium abundance; [Eu/Fe] above %g at its lower bound: %d (%d metal-poor, %d with a companion, %d with an s-process pattern); unexplained %d" % (
        n_all, eu_min, len(hits), sum(s["vet"] == "metal-poor" for s in hits), sum(s["vet"] == "binary" for s in hits), sum(s["vet"] == "abundance-pattern" for s in hits), len(left)))
    for s in sorted(hits, key=lambda s: -s["eu"]):
        print("  Gaia DR3 %s: [Eu/Fe] %.2f +- %.2f, [Fe/H] %s, [Ba/Fe] %s, [La/Fe] %s, %s K, log g %s, S/N %s, second component %s, Gaia %s; %s" % (
            s["id"], s["eu"], s["eeu"], s["feh"], s["ba"], s["la"], s["teff"], s["logg"], s["snr"], s["rv2"], s["gaia"], s["vet"] or "unexplained"))
    json.dump(dict(n=n_all, hits=hits), open(core.results("europium_rich_dwarfs.json"), "w"), indent=1)
    if record:
        core.record_run(
            "galah_europium_dwarfs", "Europium far above iron in dwarfs", "GALAH DR4 mainstartable (Data Central); Gaia DR3",
            "dwarfs with a clean fit and a clean europium abundance to %g dex at S/N above %g: %d" % (err_max, snr_min, n_all), os.path.abspath(__file__), "europium", n_all, "stars",
            dict(eu_min=eu_min, err_max=err_max, snr_min=snr_min, logg_min=logg_min, metal_poor=metal_poor, ba_s=ba_s),
            "metallicity, second components, Gaia's non-single-star flag and RUWE, barium and lanthanum", "[Eu/Fe] above %g at its lower bound in a dwarf" % eu_min,
            "%d: %d metal-poor, %d with a companion, %d with an s-process pattern; %d unexplained" % (
                len(hits), sum(s["vet"] == "metal-poor" for s in hits), sum(s["vet"] == "binary" for s in hits), sum(s["vet"] == "abundance-pattern" for s in hits), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"] if s["id"].isdigit() else "galah:%s" % s["sobject"], stat=round(s["eu"], 2), unit="[Eu/Fe]", vet=s["vet"],
                  note="[Fe/H] %s, [Ba/Fe] %s, [La/Fe] %s, %s K, log g %s, S/N %s" % (s["feh"], s["ba"], s["la"], s["teff"], s["logg"], s["snr"])) for s in hits])
        print("recorded")


zirconium_DOC = r"""Zirconium alone: GALAH DR4 dwarfs (log g above 4.0) with a clean fit and clean zirconium, yttrium and barium abundances measured to 0.15 dex at signal-to-noise above 40, those with [Zr/Fe] above 0.8 while [Y/Fe] and [Ba/Fe] stay below 0.3, the metal of cladding and alloy raised where every process that makes zirconium makes its neighbours on the s-process path with it; each read for a second spectral component and Gaia's non-single-star flag, and for its temperature, since the weak zirconium lines blend with molecular bands below 5000 K and fade above 6500 K, where GALAH's zirconium is not trusted.

    python -I seti.py composition zirconium [record] [--zr-min 0.8] [--others-max 0.3] [--err-max 0.15] [--snr-min 40] [--logg-min 4] [--teff-min 5000] [--teff-max 6500]"""


def zirconium_main(record=False, zr_min=0.8, others_max=0.3, err_max=0.15, snr_min=40.0, logg_min=4.0, teff_min=5000.0, teff_max=6500.0):
    from setitools.core import num
    rows = core.tap_async("datacentral", "SELECT sobject_id, gaiadr3_source_id, teff, logg, fe_h, zr_fe, e_zr_fe, y_fe, e_y_fe, ba_fe, e_ba_fe, rv_comp_2, snr_px_ccd3 FROM "
                                         "galah_dr4.mainstartable WHERE flag_sp = 0 AND flag_zr_fe = 0 AND flag_y_fe = 0 AND flag_ba_fe = 0 AND logg > %g AND snr_px_ccd3 > %g AND "
                                         "e_zr_fe < %g AND e_y_fe < %g AND e_ba_fe < %g" % (logg_min, snr_min, err_max, err_max, err_max), maxrec=1000000)
    n_all = len(rows)
    st = [dict(id=str(r["gaiadr3_source_id"] or "").strip(), sobject=str(r["sobject_id"]), teff=num(r["teff"]), logg=num(r["logg"]), feh=num(r["fe_h"]), zr=num(r["zr_fe"]),
               ezr=num(r["e_zr_fe"]), y=num(r["y_fe"]), ba=num(r["ba_fe"]), rv2=num(r["rv_comp_2"]), snr=num(r["snr_px_ccd3"])) for r in rows]
    hits = [s for s in st if None not in (s["zr"], s["y"], s["ba"]) and s["zr"] - s["ezr"] > zr_min and s["y"] < others_max and s["ba"] < others_max]
    nss = {}
    ids = [s["id"] for s in hits if s["id"]]
    for k in range(0, len(ids), 500):
        for r in core.tap("gaia", "SELECT source_id, non_single_star, ruwe FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(ids[k:k + 500])):
            nss[str(r["source_id"])] = (num(r["non_single_star"]), num(r["ruwe"]))
    for s in hits:
        s["gaia"] = nss.get(s["id"])
        g = s["gaia"] or (0, 0)
        s["vet"] = ("binary" if s["rv2"] is not None or (g[0] or 0) > 0 or (g[1] or 0) > 1.4 else
                    "defective-spectrum" if s["teff"] is None or not teff_min <= s["teff"] <= teff_max else None)
    left = [s for s in hits if s["vet"] is None]
    print("%d GALAH dwarfs with clean zirconium, yttrium and barium; [Zr/Fe] above %g at its lower bound with [Y/Fe] and [Ba/Fe] below %g: %d (%d with a companion, %d outside %g-%g K); "
          "unexplained %d" % (n_all, zr_min, others_max, len(hits), sum(s["vet"] == "binary" for s in hits), sum(s["vet"] == "defective-spectrum" for s in hits), teff_min, teff_max,
                             len(left)))
    for s in sorted(hits, key=lambda s: -s["zr"]):
        print("  Gaia DR3 %s: [Zr/Fe] %.2f +- %.2f, [Y/Fe] %.2f, [Ba/Fe] %.2f, [Fe/H] %s, %s K, log g %s, S/N %s, second component %s, Gaia %s; %s" % (
            s["id"], s["zr"], s["ezr"], s["y"], s["ba"], s["feh"], s["teff"], s["logg"], s["snr"], s["rv2"], s["gaia"], s["vet"] or "unexplained"))
    json.dump(dict(n=n_all, hits=hits), open(core.results("zirconium_alone.json"), "w"), indent=1)
    if record:
        core.record_run(
            "galah_zirconium_alone", "Zirconium alone", "GALAH DR4 mainstartable (Data Central); Gaia DR3",
            "dwarfs with a clean fit and clean zirconium, yttrium and barium to %g dex at S/N above %g: %d" % (err_max, snr_min, n_all), os.path.abspath(__file__), "zirconium", n_all,
            "stars", dict(zr_min=zr_min, others_max=others_max, err_max=err_max, snr_min=snr_min, logg_min=logg_min, teff=[teff_min, teff_max]),
            "second components, Gaia's non-single-star flag and RUWE, the temperature range of clean zirconium lines",
            "[Zr/Fe] above %g with [Y/Fe] and [Ba/Fe] below %g" % (zr_min, others_max), "%d: %d with a companion, %d outside the temperature range; %d unexplained" % (
                len(hits), sum(s["vet"] == "binary" for s in hits), sum(s["vet"] == "defective-spectrum" for s in hits), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"] if s["id"].isdigit() else "galah:%s" % s["sobject"], stat=round(s["zr"], 2), unit="[Zr/Fe]", vet=s["vet"],
                  note="[Y/Fe] %.2f, [Ba/Fe] %.2f, [Fe/H] %s, %s K, log g %s, S/N %s" % (s["y"], s["ba"], s["feh"], s["teff"], s["logg"], s["snr"])) for s in hits])
        print("recorded")


limeteoritic_DOC = r"""Lithium above the meteoritic: GALAH DR4 dwarfs (log g above 4.0) with a clean fit and a clean lithium abundance measured to 0.15 dex at signal-to-noise above 40, those with A(Li) above 4.3 at the lower bound, a dex above the meteoritic 3.26 and beyond what the engulfment of a giant planet can add to a convective star, where a star fed refined lithium would stand; the stars between 3.6 and 4.3 are counted as within engulfment's reach; each read for its temperature, since the lithium line of a star above 6500 K is weak and blended, and for a second spectral component and a Gaia non-single-star flag, since a companion's spectrum can fill the line.

    python -I seti.py composition limeteoritic [record] [--li-min 4.3] [--err-max 0.15] [--snr-min 40] [--logg-min 4] [--teff-max 6500] [--engulf-min 3.6]"""


def limeteoritic_main(record=False, li_min=4.3, err_max=0.15, snr_min=40.0, logg_min=4.0, teff_max=6500.0, engulf_min=3.6):
    from setitools.core import num
    rows = core.tap_async("datacentral", "SELECT sobject_id, gaiadr3_source_id, teff, logg, fe_h, a_li, e_a_li_low, e_a_li_upp, a_li_upp_lim, rv_comp_2, snr_px_ccd3, vsini FROM "
                                         "galah_dr4.mainstartable WHERE flag_sp = 0 AND flag_a_li = 0 AND (a_li_upp_lim IS NULL OR a_li_upp_lim = 0) AND logg > %g AND snr_px_ccd3 > %g AND e_a_li_low < %g AND "
                                         "e_a_li_upp < %g" % (logg_min, snr_min, err_max, err_max), maxrec=1000000)
    n_all = len(rows)
    st = [dict(id=str(r["gaiadr3_source_id"] or "").strip(), sobject=str(r["sobject_id"]), teff=num(r["teff"]), logg=num(r["logg"]), feh=num(r["fe_h"]), li=num(r["a_li"]), eli=max(num(r["e_a_li_low"]) or 0, num(r["e_a_li_upp"]) or 0),
               rv2=num(r["rv_comp_2"]), snr=num(r["snr_px_ccd3"]), vsini=num(r["vsini"])) for r in rows]
    hits = [s for s in st if s["li"] is not None and s["li"] - s["eli"] > li_min]
    engulf = [s for s in st if s["li"] is not None and engulf_min < s["li"] - s["eli"] <= li_min]
    nss = {}
    ids = [s["id"] for s in hits if s["id"]]
    for k in range(0, len(ids), 500):
        for r in core.tap("gaia", "SELECT source_id, non_single_star, ruwe FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(ids[k:k + 500])):
            nss[str(r["source_id"])] = (num(r["non_single_star"]), num(r["ruwe"]))
    for s in hits:
        s["gaia"] = nss.get(s["id"])
        g = s["gaia"] or (0, 0)
        s["vet"] = ("defective-spectrum" if s["teff"] is None or s["teff"] > teff_max else "binary" if s["rv2"] is not None or (g[0] or 0) > 0 or (g[1] or 0) > 1.4 else None)
    left = [s for s in hits if s["vet"] is None]
    print("%d GALAH dwarfs with a clean lithium abundance; A(Li) above %g at the lower bound: %d (%d hotter than %g K, %d with a companion); between %g and %g, within engulfment's reach: %d; "
          "unexplained %d" % (n_all, li_min, len(hits), sum(s["vet"] == "defective-spectrum" for s in hits), teff_max, sum(s["vet"] == "binary" for s in hits), engulf_min, li_min,
                             len(engulf), len(left)))
    for s in sorted(hits, key=lambda s: -s["li"]):
        print("  Gaia DR3 %s: A(Li) %.2f +- %.2f, %s K, log g %s, [Fe/H] %s, vsini %s, S/N %s, second component %s, Gaia %s; %s" % (
            s["id"], s["li"], s["eli"], s["teff"], s["logg"], s["feh"], s["vsini"], s["snr"], s["rv2"], s["gaia"], s["vet"] or "unexplained"))
    json.dump(dict(n=n_all, hits=hits), open(core.results("lithium_above_meteoritic.json"), "w"), indent=1)
    if record:
        core.record_run(
            "galah_lithium_meteoritic", "Lithium above the meteoritic", "GALAH DR4 mainstartable (Data Central); Gaia DR3",
            "dwarfs with a clean fit and a clean lithium abundance to %g dex at S/N above %g: %d" % (err_max, snr_min, n_all), os.path.abspath(__file__), "limeteoritic", n_all, "stars",
            dict(li_min=li_min, err_max=err_max, snr_min=snr_min, logg_min=logg_min, teff_max=teff_max, engulf_min=engulf_min),
            "the reach of planet engulfment, a dex above the meteoritic; the temperature range of the lithium line; second components, Gaia's flags and RUWE",
            "A(Li) above %g at its lower bound in a dwarf" % li_min, "%d: %d too hot for the line, %d with a companion; %d between %g and %g within engulfment's reach; %d unexplained" % (
                len(hits), sum(s["vet"] == "defective-spectrum" for s in hits), sum(s["vet"] == "binary" for s in hits), len(engulf), engulf_min, li_min, len(left)),
            [dict(target="gaia_dr3:%s" % s["id"] if s["id"].isdigit() else "galah:%s" % s["sobject"], stat=round(s["li"], 2), unit="A(Li)", vet=s["vet"],
                  note="%s K, log g %s, [Fe/H] %s, vsini %s, S/N %s" % (s["teff"], s["logg"], s["feh"], s["vsini"], s["snr"])) for s in hits])
        print("recorded")


carbondwarfs_DOC = r"""Carbon stars on the main sequence next door: Gaia DR3's gold sample of carbon stars joined to gaia_source, those within 1 kpc at parallax over error above 5 whose absolute G places them on the main sequence, fainter than 4, a dwarf wearing an atmosphere of carbon it could not have made, since only an evolved star dredges carbon up; each read for RUWE and Gaia's non-single-star flag, since the known dwarf carbon stars carry the carbon of a companion that is now a white dwarf, and in SIMBAD.

    python -I seti.py composition carbondwarfs [record] [--plx-min 1] [--plx-sigma 5] [--mg-min 4]"""


def carbondwarfs_main(record=False, plx_min=1.0, plx_sigma=5.0, mg_min=4.0):
    from setitools.core import num
    rows = core.tap("gaia", "SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.non_single_star, g.teff_gspphot, g.logg_gspphot FROM "
                            "gaiadr3.gold_sample_carbon_stars AS c JOIN gaiadr3.gaia_source AS g ON c.source_id = g.source_id WHERE g.parallax > %g AND g.parallax_over_error > %g" % (plx_min, plx_sigma))
    n_gold = int(core.tap("gaia", "SELECT COUNT(*) AS n FROM gaiadr3.gold_sample_carbon_stars")[0]["n"])
    st = []
    for r in rows:
        plx, g = num(r["parallax"]), num(r["phot_g_mean_mag"])
        if plx is None or g is None:
            continue
        st.append(dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), plx=plx, g=g, mg=g + 5 * math.log10(plx / 100.0), c=num(r["bp_rp"]), ruwe=num(r["ruwe"]),
                       nss=num(r["non_single_star"]), teff=num(r["teff_gspphot"]), logg=num(r["logg_gspphot"])))
    dwarfs = [s for s in st if s["mg"] > mg_min]
    sim = {}
    for k in range(0, len(dwarfs), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in dwarfs[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in dwarfs:
        s["simbad"] = sim.get(s["id"])
        s["vet"] = "binary" if (s["ruwe"] or 0) > 1.4 or (s["nss"] or 0) > 0 else "companion"
    print("%d carbon stars in the gold sample, %d within %.0f pc at parallax over error above %g; on the main sequence (absolute G above %g): %d (%d with a companion shown by RUWE or Gaia's flag, "
          "%d with the vanished companion of a dwarf carbon star); unexplained 0" % (n_gold, len(st), 1000 / plx_min, plx_sigma, mg_min, len(dwarfs), sum(s["vet"] == "binary" for s in dwarfs),
                                                                                   sum(s["vet"] == "companion" for s in dwarfs)))
    for s in sorted(dwarfs, key=lambda s: -s["mg"]):
        print("  Gaia DR3 %s: absolute G %.2f, G %.2f, BP-RP %s, %.0f pc, RUWE %s, Gaia flag %s, %s K; SIMBAD %s; %s" % (
            s["id"], s["mg"], s["g"], s["c"], 1000 / s["plx"], s["ruwe"], s["nss"], s["teff"], s["simbad"], s["vet"]))
    json.dump(dict(n=len(st), dwarfs=dwarfs), open(core.results("carbon_dwarfs_nearby.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_carbon_dwarfs", "Carbon stars on the main sequence next door", "Gaia DR3 gold_sample_carbon_stars and gaia_source; SIMBAD",
            "gold-sample carbon stars within %.0f pc at parallax over error above %g: %d" % (1000 / plx_min, plx_sigma, len(st)), os.path.abspath(__file__), "carbondwarfs", len(st), "stars",
            dict(plx_min=plx_min, plx_sigma=plx_sigma, mg_min=mg_min), "RUWE and Gaia's non-single-star flag; the dwarf carbon stars' white-dwarf companions",
            "a carbon star on the main sequence", "%d: %d with a companion shown, %d of the dwarf carbon kind; 0 unexplained" % (
                len(dwarfs), sum(s["vet"] == "binary" for s in dwarfs), sum(s["vet"] == "companion" for s in dwarfs)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["mg"], 2), unit="absolute G", vet=s["vet"],
                  note="G %.2f, BP-RP %s, %.0f pc, RUWE %s, Gaia flag %s; SIMBAD %s" % (s["g"], s["c"], 1000 / s["plx"], s["ruwe"], s["nss"], s["simbad"])) for s in dwarfs])
        print("recorded")


singlesd_DOC = r"""Single hot subdwarfs in the thin disc: the known hot subdwarfs of Culpan et al. (2022, VizieR J/A+A/662/A40) of classes sdB, sdOB and sdO with a Gaia parallax at 5 sigma, a proper motion and a radial velocity from SDSS or LAMOST, their Galactic velocities, those on thin-disc orbits (within 50 km/s of the local standard of rest) that show no companion in their class, Gaia's RUWE, non-single-star flag or BP-RP, which a cool companion reddens; a helium-burning core stripped of its envelope needs a companion or a merger to lose it, and mergers come from the old population, so a young single one on a thin-disc orbit stands out; each counted against the binaries and the thick-disc singles of the same list.

    python -I seti.py composition singlesd [record] [--plx-sigma 5] [--v-lsr-max 50] [--bp-rp-max -0.25]"""


def singlesd_main(record=False, plx_sigma=5.0, v_lsr_max=50.0, bp_rp_max=-0.25):
    from astropy.coordinates import SkyCoord, Galactocentric
    import astropy.units as u
    t = [x for x in core.vizier("J/A+A/662/A40") if x.meta.get("name", "").endswith("knownhsd")][0]
    st = []
    for r in t:
        cls = str(r["SpClass"]).strip()
        if not cls.startswith("sd"):
            continue
        plx, eplx = core.num(r["Plx"]), core.num(r["e_Plx"])
        rv = core.num(r["RVSDSS"]) if core.num(r["RVSDSS"]) is not None else core.num(r["RVLAMOST"])
        if plx is None or not eplx or plx / eplx < plx_sigma or rv is None or core.num(r["pmRAGAIA"]) is None:
            continue
        st.append(dict(name=str(r["Name"]).strip(), id=str(r["GaiaEDR3"]).strip(), cls=cls, ra=float(r["RA_ICRS"]), dec=float(r["DE_ICRS"]), plx=plx, pmra=float(r["pmRAGAIA"]),
                       pmdec=float(r["pmDEGAIA"]), rv=rv, c=(core.num(r["BPGAIA"]) - core.num(r["RPGAIA"])) if core.num(r["BPGAIA"]) is not None and core.num(r["RPGAIA"]) is not None else None,
                       teff=core.num(r["Teff"]), logg=core.num(r["logg"])))
    c = SkyCoord(ra=[s["ra"] for s in st] * u.deg, dec=[s["dec"] for s in st] * u.deg, distance=[1000 / s["plx"] for s in st] * u.pc, pm_ra_cosdec=[s["pmra"] for s in st] * u.mas / u.yr,
                 pm_dec=[s["pmdec"] for s in st] * u.mas / u.yr, radial_velocity=[s["rv"] for s in st] * u.km / u.s)
    g = c.transform_to(Galactocentric())
    sun = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1e-6 * u.pc, pm_ra_cosdec=0 * u.mas / u.yr, pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s).transform_to(Galactocentric())
    vx, vy, vz = g.v_x.to(u.km / u.s).value, g.v_y.to(u.km / u.s).value, g.v_z.to(u.km / u.s).value
    sx, sy, sz = sun.v_x.to(u.km / u.s).value, sun.v_y.to(u.km / u.s).value, sun.v_z.to(u.km / u.s).value
    for k, s in enumerate(st):
        s["v_lsr"] = float(math.sqrt((vx[k] - sx + 11.1) ** 2 + (vy[k] - sy + 12.2) ** 2 + (vz[k] - sz + 7.25) ** 2))
    gaia = {}
    ids = [s["id"] for s in st if s["id"].isdigit()]
    for k in range(0, len(ids), 500):
        for r in core.tap("gaia", "SELECT source_id, ruwe, non_single_star, ipd_frac_multi_peak FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(ids[k:k + 500])):
            gaia[str(r["source_id"])] = (core.num(r["ruwe"]), core.num(r["non_single_star"]), core.num(r["ipd_frac_multi_peak"]))
    for s in st:
        q = gaia.get(s["id"], (None, None, None))
        s["ruwe"], s["nss"], s["multi"] = q
        s["binary"] = "+" in s["cls"] or (q[0] or 0) > 1.4 or (q[1] or 0) > 0 or (q[2] or 0) > 2 or (s["c"] is not None and s["c"] > bp_rp_max)
        s["thin"] = s["v_lsr"] < v_lsr_max
    singles = [s for s in st if not s["binary"]]
    thin_single = [s for s in singles if s["thin"]]
    thick_single = [s for s in singles if not s["thin"]]
    thin_binary = [s for s in st if s["binary"] and s["thin"]]
    for s in thin_single:
        s["vet"] = "evolved"
    print("%d known hot subdwarfs with parallax, motion and a velocity; %d show no companion, %d of those on thin-disc orbits against %d thick-disc singles and %d thin-disc binaries; "
          "the single thin-disc share %.2f against the binaries' %.2f" % (len(st), len(singles), len(thin_single), len(thick_single), len(thin_binary),
                                                                        len(thin_single) / max(len(singles), 1), len(thin_binary) / max(sum(1 for s in st if s["binary"]), 1)))
    for s in sorted(thin_single, key=lambda s: s["v_lsr"])[:15]:
        print("  %s (%s): %.0f km/s from the LSR, %s K, log g %s, BP-RP %s, RUWE %s; %s" % (s["name"], s["cls"], s["v_lsr"], s["teff"], s["logg"], s["c"], s["ruwe"], s["vet"]))
    json.dump(dict(n=len(st), thin_single=thin_single), open(core.results("single_hot_subdwarfs.json"), "w"), indent=1)
    if record:
        core.record_run(
            "hot_subdwarfs_single_thin_disc", "Single hot subdwarfs in the thin disc", "Culpan et al. (2022) known hot subdwarfs (VizieR J/A+A/662/A40); Gaia DR3",
            "known sdB, sdOB and sdO stars with parallax at %g sigma, proper motion and a radial velocity: %d" % (plx_sigma, len(st)), os.path.abspath(__file__), "singlesd", len(st), "stars",
            dict(plx_sigma=plx_sigma, v_lsr_max=v_lsr_max, bp_rp_max=bp_rp_max), "the binaries' thin-disc share %.2f; the thick-disc singles %d" % (
                len(thin_binary) / max(sum(1 for s in st if s["binary"]), 1), len(thick_single)),
            "single hot subdwarfs on thin-disc orbits beyond the binaries' share", "%d single thin-disc stars of %d singles, a share of %.2f against the binaries' %.2f; every one an evolved "
            "star the thin disc's own binaries and mergers can supply" % (len(thin_single), len(singles), len(thin_single) / max(len(singles), 1),
                                                                        len(thin_binary) / max(sum(1 for s in st if s["binary"]), 1)),
            [dict(target="gaia_dr3:%s" % s["id"] if s["id"].isdigit() else "star:%s" % s["name"], stat=round(s["v_lsr"], 1), unit="km/s from the LSR", vet=s["vet"],
                  note="%s, %s K, log g %s, BP-RP %s, RUWE %s" % (s["cls"], s["teff"], s["logg"], s["c"], s["ruwe"])) for s in thin_single])
        print("recorded")


worstfits_DOC = r"""Spectra no stellar model fits: the GALAH DR4 dwarfs (log g above 3.8) within 100 pc at signal-to-noise above 100 in the red arm, the worst 0.5 per cent of the spectral fit's chi-square over the median of its 250 K bin of temperature, each read for what spoils a stellar model: a second velocity component, rotation above 20 km/s, H-alpha in emission or filled in, a Gaia binary signal (RUWE above 1.4), or the fit's own flags; a spectrum carrying light no star makes would remain.

    python -I seti.py composition worstfits [record] [--snr-min 100] [--frac 0.005] [--teff-bin 250]"""


def worstfits_main(record=False, snr_min=100.0, frac=0.005, teff_bin=250.0):
    rows = core.tap("datacentral", "SELECT sobject_id, gaiadr3_source_id, teff, logg, fe_h, chi2_sp, flag_sp, vsini, rv_comp_1, rv_comp_2, sb2_rv_50, ew_h_alpha, res_h_alpha, ruwe, parallax, "
                                   "snr_px_ccd3 FROM galah_dr4.mainstartable WHERE parallax > 10 AND logg > 3.8 AND snr_px_ccd3 > %g AND chi2_sp IS NOT NULL" % snr_min)
    st = [dict(id=str(r["gaiadr3_source_id"]), sob=str(r["sobject_id"]), teff=core.num(r["teff"]), chi2=core.num(r["chi2_sp"]), flag=int(core.num(r["flag_sp"]) or 0), vsini=core.num(r["vsini"]),
               rv2=core.num(r["rv_comp_2"]), sb2=core.num(r["sb2_rv_50"]), ewha=core.num(r["ew_h_alpha"]), resha=core.num(r["res_h_alpha"]), ruwe=core.num(r["ruwe"]), feh=core.num(r["fe_h"]))
          for r in rows]
    st = [s for s in st if s["teff"] is not None and s["chi2"] is not None]
    seen = {}
    for s in st:
        if s["id"] not in seen or s["chi2"] < seen[s["id"]]["chi2"]:
            seen[s["id"]] = s
    st = list(seen.values())
    for b in {int(s["teff"] // teff_bin) for s in st}:
        grp = [s for s in st if int(s["teff"] // teff_bin) == b]
        med = float(np.median([s["chi2"] for s in grp]))
        for s in grp:
            s["rel"] = s["chi2"] / med if med > 0 else 0.0
    worst = sorted(st, key=lambda s: -s["rel"])[:max(1, int(round(frac * len(st))))]
    for s in worst:
        why = []
        if s["rv2"] is not None or s["sb2"] is not None:
            why.append("second component")
        if (s["vsini"] or 0) > 20:
            why.append("rotation %.0f km/s" % s["vsini"])
        if (s["ewha"] is not None and s["ewha"] < 0) or (s["resha"] or 0) > 0.1:
            why.append("H-alpha")
        if (s["ruwe"] or 0) > 1.4:
            why.append("RUWE %.1f" % s["ruwe"])
        if s["flag"]:
            why.append("flag_sp %d" % s["flag"])
        s["why"] = why
        s["vet"] = "binary" if ("second component" in why or any(w.startswith("RUWE") for w in why)) else ("active-star" if "H-alpha" in why or any(w.startswith("rotation") for w in why)
                                                                                                          else ("artefact" if s["flag"] else None))
    left = [s for s in worst if s["vet"] is None]
    print("%d GALAH DR4 dwarfs within 100 pc at S/N above %g; the worst %g per cent by temperature bin: %d; binary %d, activity or rotation %d, flagged fits %d, unexplained %d" % (
        len(st), snr_min, 100 * frac, len(worst), sum(s["vet"] == "binary" for s in worst), sum(s["vet"] == "active-star" for s in worst), sum(s["vet"] == "artefact" for s in worst), len(left)))
    for s in sorted(left, key=lambda s: -s["chi2"]):
        print("  Gaia DR3 %s (%s): %.0f K, [Fe/H] %s, chi2 %.2f (%.1f times its bin), v sin i %s, H-alpha EW %s residual %s, RUWE %s" % (s["id"], s["sob"], s["teff"], s["feh"], s["chi2"], s["rel"], s["vsini"], s["ewha"],
                                                                                                               s["resha"], s["ruwe"]))
    json.dump(worst, open(core.results("galah_worst_fits.json"), "w"), indent=1)
    if record:
        core.record_run(
            "galah_worst_fits_nearby", "Spectra no stellar model fits", "GALAH DR4 (Data Central galah_dr4.mainstartable)",
            "dwarfs within 100 pc at red-arm S/N above %g: %d" % (snr_min, len(st)), os.path.abspath(__file__), "worstfits", len(st), "stars", dict(snr_min=snr_min, frac=frac, teff_bin=teff_bin),
            "the chi-square over the median of its %g K temperature bin" % teff_bin, "a spectrum no stellar model fits, with no binary, activity or fit flag to account for it",
            "%d worst fits: %d binaries, %d active or fast rotators, %d flagged, %d unexplained" % (len(worst), sum(s["vet"] == "binary" for s in worst), sum(s["vet"] == "active-star" for s in worst),
                                                                                             sum(s["vet"] == "artefact" for s in worst), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["chi2"], 2), unit="chi-square of the spectral fit", vet=s["vet"], note="%.0f K; %s" % (s["teff"], ", ".join(s["why"]) or "nothing found"))
             for s in worst])
        print("recorded")


xpworst_DOC = r"""Gaia XP spectra the models fit worst: the Gaia DR3 stars within 100 pc with parallaxes at 10 sigma and a GSP-Phot solution, the GSP-Phot log-posterior of each against the median and spread of its bin of 0.1 mag in BP-RP and 1 mag in G fitted with the same model library, since the libraries fit unequally well, the worst 0.1 per cent read for what spoils a single-star fit: a companion (RUWE above 1.4, several image peaks or a non-single-star solution), emission lines (an ESP-ELS class), blended photometry (corrected BP/RP excess above 3 sigma), or an extinction no star within 100 pc carries (A0 above 0.5 mag), which marks the fit's own degeneracy; light no star makes would remain.

    python -I seti.py composition xpworst [record] [--frac 0.001]"""


def xpworst_main(record=False, frac=0.001):
    rows = core.tap("gaia", "SELECT g.source_id, g.bp_rp, g.phot_g_mean_mag, g.ruwe, g.ipd_frac_multi_peak, g.non_single_star, g.phot_bp_rp_excess_factor, a.logposterior_gspphot, "
                            "a.azero_gspphot, a.teff_gspphot, a.libname_gspphot, a.classlabel_espels FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a ON a.source_id = g.source_id "
                            "WHERE g.parallax > 10 AND g.parallax_over_error > 10 AND a.logposterior_gspphot IS NOT NULL AND g.bp_rp IS NOT NULL", timeout=1200)
    st = []
    for r in rows:
        c, gm = float(r["bp_rp"]), float(r["phot_g_mean_mag"])
        st.append(dict(id=r["source_id"], c=c, g=gm, lp=float(r["logposterior_gspphot"]), ruwe=core.num(r["ruwe"]), multi=core.num(r["ipd_frac_multi_peak"]), nss=core.num(r["non_single_star"]),
                       cstar=core.cstar(r["phot_bp_rp_excess_factor"], c) if core.num(r["phot_bp_rp_excess_factor"]) is not None else None, a0=core.num(r["azero_gspphot"]),
                       teff=core.num(r["teff_gspphot"]), els=(r.get("classlabel_espels") or "").strip(), lib=(r.get("libname_gspphot") or "").strip()))
    groups = {}
    for s in st:
        groups.setdefault((int(s["c"] // 0.1), int(s["g"]), s["lib"]), []).append(s)
    for grp in groups.values():
        lp = np.array([s["lp"] for s in grp])
        med, mad = float(np.median(lp)), float(np.median(np.abs(lp - np.median(lp)))) * 1.4826
        for s in grp:
            s["z"] = (s["lp"] - med) / mad if mad > 0 and len(grp) >= 20 else 0.0
    worst = sorted(st, key=lambda s: s["z"])[:max(1, int(round(frac * len(st))))]
    for s in worst:
        why = []
        if (s["ruwe"] or 0) > 1.4 or (s["multi"] or 0) > 2 or (s["nss"] or 0) > 0:
            why.append("companion")
        if s["els"]:
            why.append("emission %s" % s["els"])
        if s["cstar"] is not None and s["cstar"] > 3 * core.cstar_sigma(s["g"]):
            why.append("blended photometry")
        if (s["a0"] or 0) > 0.5:
            why.append("A0 %.2f" % s["a0"])
        s["why"] = why
        s["vet"] = ("binary" if "companion" in why else "active-star" if any(w.startswith("emission") for w in why) else "blend" if "blended photometry" in why
                    else "fit-degeneracy" if any(w.startswith("A0") for w in why) else None)
    left = [s for s in worst if s["vet"] is None]
    counts = {v: sum(s["vet"] == v for s in worst) for v in ("binary", "active-star", "blend", "fit-degeneracy")}
    print("%d stars within 100 pc with a GSP-Phot fit; the worst %g per cent against their colour and magnitude bin: %d; %s; unexplained %d" % (len(st), 100 * frac, len(worst), counts, len(left)))
    for s in left[:20]:
        print("  Gaia DR3 %s: BP-RP %.2f, G %.2f, %s, log-posterior %.1f (%.1f robust widths below its bin), Teff %s, A0 %s, RUWE %s" % (s["id"], s["c"], s["g"], s["lib"], s["lp"], -s["z"], s["teff"], s["a0"], s["ruwe"]))
    json.dump(worst, open(core.results("xp_worst_fits.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_xp_worst_fits_nearby", "Gaia XP spectra the models fit worst", "Gaia DR3 gaia_source and astrophysical_parameters (GSP-Phot, ESP-ELS)",
            "stars within 100 pc with parallaxes at 10 sigma and a GSP-Phot fit: %d" % len(st), os.path.abspath(__file__), "xpworst", len(st), "stars", dict(frac=frac),
            "the log-posterior's median and robust spread in bins of 0.1 mag in BP-RP and 1 mag in G for each model library", "an XP spectrum no single-star model fits with no companion, emission, blend or degeneracy behind it",
            "%d worst fits: %s; %d unexplained" % (len(worst), counts, len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(-s["z"], 1), unit="robust widths below the bin's log-posterior", vet=s["vet"],
                  note="BP-RP %.2f, G %.2f, Teff %s; %s" % (s["c"], s["g"], s["teff"], ", ".join(s["why"]) or "nothing found")) for s in worst])
        print("recorded")


xpmisfits_DOC = r"""What sets the worst-fitted nearby M dwarfs apart: the M dwarfs fitted with the MARCS library among the XP worst fits with no companion, emission, blend or extinction behind them, against 3000 stars of their BP-RP and G range within 100 pc whose GSP-Phot fits are ordinary, each set by its own excess in BP-G over the colour-matched median at its G-RP, where a white-dwarf companion adds blue light, its offset in absolute G from the main sequence, where a binary sits above and a metal-poor subdwarf below, its 2MASS J-Ks at fixed BP-RP, which collision-induced hydrogen absorption makes bluer in a metal-poor star, its tangential speed, its parallax's signal-to-noise, its Galactic latitude, its Gaia radial-velocity scatter and its GALEX ultraviolet detection; a property the misfits share and their peers lack names what the models miss; metallicity itself is read from spectra by the subdwarf-index test.

    python -I seti.py composition xpmisfits [record] [--controls 3000]"""


def xp_misfit_traits(worst, controls=3000):
    """The misfits' traits against peers of their colour and magnitude within 100 pc: BP-G at fixed G-RP, absolute G at fixed BP-RP and G-RP, J-Ks at fixed BP-RP, speed, radial-velocity scatter and GALEX detection."""
    from scipy.stats import mannwhitneyu, fisher_exact
    ids = [w["id"] for w in worst]
    cmin, cmax = min(w["c"] for w in worst) - 0.05, max(w["c"] for w in worst) + 0.05
    gmin, gmax = min(w["g"] for w in worst) - 0.5, max(w["g"] for w in worst) + 0.5
    cols = ("g.source_id, g.ra, g.dec, g.parallax, g.pmra, g.pmdec, g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp, g.ruwe, g.radial_velocity, "
            "g.rv_amplitude_robust, g.rv_chisq_pvalue, g.rv_nb_transits, g.phot_variable_flag, g.parallax_over_error, g.b, a.logposterior_gspphot")
    mine = core.tap("gaia", "SELECT %s FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a ON a.source_id = g.source_id WHERE g.source_id IN (%s)" % (cols, ",".join(ids)))
    peers = core.tap("gaia", "SELECT TOP %d %s FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a ON a.source_id = g.source_id WHERE g.parallax > 10 AND "
                             "g.parallax_over_error > 10 AND g.bp_rp BETWEEN %.3f AND %.3f AND g.phot_g_mean_mag BETWEEN %.2f AND %.2f AND a.libname_gspphot = 'MARCS' "
                             "AND g.random_index < 400000000 ORDER BY g.random_index" % (controls, cols, cmin, cmax, gmin, gmax))
    peers = [p for p in peers if p["source_id"] not in set(ids)]

    def feats(r):
        g, bp, rp, plx = (core.num(r[k]) for k in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "parallax"))
        if None in (g, bp, rp, plx):
            return None
        return dict(id=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"]), grp=g - rp, bpg=bp - g, c=bp - rp, mg=g + 5 * math.log10(plx / 100), g=g,
                    vt=4.74047 * math.hypot(core.num(r["pmra"]) or 0, core.num(r["pmdec"]) or 0) / plx, pmra=core.num(r["pmra"]) or 0.0, pmdec=core.num(r["pmdec"]) or 0.0,
                    rvamp=core.num(r["rv_amplitude_robust"]), rvp=core.num(r["rv_chisq_pvalue"]), ruwe=core.num(r["ruwe"]), var=r["phot_variable_flag"], lp=core.num(r["logposterior_gspphot"]),
                    plxsnr=core.num(r["parallax_over_error"]), absb=abs(core.num(r["b"]) or 0.0))
    m = [x for x in (feats(r) for r in mine) if x]
    c = [x for x in (feats(r) for r in peers) if x]
    q2 = ("SELECT x.source_id, t.j_m, t.ks_m FROM gaiadr3.tmass_psc_xsc_best_neighbour AS x JOIN gaiadr3.tmass_psc_xsc_join AS j ON j.clean_tmass_psc_xsc_oid = x.clean_tmass_psc_xsc_oid "
          "JOIN gaiadr1.tmass_original_valid AS t ON t.designation = j.original_psc_source_id WHERE x.source_id IN (%s)")
    jk = {}
    allids = [x["id"] for x in m + c]
    for k in range(0, len(allids), 400):
        for r in core.tap("gaia", q2 % ",".join(allids[k:k + 400])):
            if core.num(r["j_m"]) is not None and core.num(r["ks_m"]) is not None:
                jk[r["source_id"]] = core.num(r["j_m"]) - core.num(r["ks_m"])
    for x in m + c:
        x["jk"] = jk.get(x["id"])
    allp = c

    def resid(x, key, against, mags=False):
        near = sorted(allp, key=lambda p: abs(p[against] - x[against]))[:100]
        vals = np.array([p[key] for p in near])
        med = float(np.median(vals))
        return (x[key] - med) if mags else (x[key] - med) / (1.4826 * float(np.median(np.abs(vals - med))) or 1e-3)
    for group in (m, c):
        for x in group:
            x["blue"] = resid(x, "bpg", "grp")
            x["dmg"] = resid(x, "mg", "c")
            x["dmg_grp"] = resid(x, "mg", "grp")
            x["dmg_mag"] = resid(x, "mg", "grp", mags=True)
    jkp = [p for p in allp if p["jk"] is not None]
    for group in (m, c):
        for x in group:
            if x["jk"] is None:
                x["jkres"] = None
                continue
            near = sorted(jkp, key=lambda p: abs(p["c"] - x["c"]))[:100]
            vals = np.array([p["jk"] for p in near])
            med = float(np.median(vals))
            x["jkres"] = (x["jk"] - med) / (1.4826 * float(np.median(np.abs(vals - med))) or 1e-3)
    for x in m:
        x["vet"] = None
    out = {}
    for key, label in (("blue", "BP-G excess at fixed G-RP"), ("dmg", "absolute G offset at fixed BP-RP"), ("dmg_grp", "absolute G offset at fixed G-RP"), ("jkres", "J-Ks offset at fixed BP-RP"),
                       ("vt", "tangential speed, km/s"), ("plxsnr", "parallax signal-to-noise"), ("absb", "Galactic latitude, deg, unsigned")):
        a, b = np.array([x[key] for x in m if x[key] is not None]), np.array([x[key] for x in c if x[key] is not None])
        out[key] = dict(label=label, misfits=float(np.median(a)), peers=float(np.median(b)), p=float(mannwhitneyu(a, b).pvalue))
    def rvv(xs):
        return sum(1 for x in xs if x["rvp"] is not None and x["rvp"] < 0.01), sum(1 for x in xs if x["rvp"] is not None)
    rm, rc = rvv(m), rvv(c)
    out["rv"] = dict(label="radial velocity variable (chi-square p below 0.01)", misfits="%d of %d" % rm, peers="%d of %d" % rc,
                     p=float(fisher_exact([[rm[0], rm[1] - rm[0]], [rc[0], rc[1] - rc[0]]])[1]) if rm[1] and rc[1] else None)
    uv = {}
    for name, group in (("misfits", m), ("peers", c[:600])):
        up = [dict(source_id=x["id"], ra=x["ra"] - 8.5 * x["pmra"] / 3.6e6 / math.cos(math.radians(x["dec"])), dec=x["dec"] - 8.5 * x["pmdec"] / 3.6e6) for x in group]
        hit = {mm["source_id"] for mm in core.xmatch(up, "II/335/galex_ais", 10.0)}
        cov = {mm["source_id"] for mm in core.xmatch(up, "II/335/galex_ais", 180.0)}
        uv[name] = (len(hit), len(cov))
    out["uv"] = dict(label="GALEX AIS detection within 10 arcsec among covered stars", misfits="%d of %d" % uv["misfits"], peers="%d of %d" % uv["peers"],
                     p=float(fisher_exact([[uv["misfits"][0], uv["misfits"][1] - uv["misfits"][0]], [uv["peers"][0], uv["peers"][1] - uv["peers"][0]]])[1]) if uv["misfits"][1] and uv["peers"][1] else None)
    return m, c, out


def xpmisfits_main(record=False, controls=3000):
    worst = [w for w in json.load(open(core.results("xp_worst_fits.json"))) if w.get("lib") == "MARCS" and w.get("vet") is None]
    m, c, out = xp_misfit_traits(worst, controls)
    print("%d MARCS misfits with no companion, emission, blend or extinction behind them against %d peers of their colour and magnitude within 100 pc:" % (len(m), len(c)))
    for k, v in out.items():
        print("  %s: misfits %s, peers %s, p %s" % (v["label"], v["misfits"], v["peers"], v["p"]))
    for x in sorted(m, key=lambda x: -x["blue"]):
        print("  Gaia DR3 %s: BP-G excess %+.1f widths, absolute G offset %+.1f widths, J-Ks offset %s widths, %.0f km/s, RV p %s, RUWE %s, %s; %s" % (
            x["id"], x["blue"], x["dmg"], "%+.1f" % x["jkres"] if x["jkres"] is not None else "-", x["vt"], x["rvp"], x["ruwe"], x["var"], x["vet"] or "unexplained"))
    json.dump(dict(misfits=m, summary=out), open(core.results("xp_misfits_traits.json"), "w"), indent=1, default=str)
    if record:
        core.record_run(
            "gaia_xp_misfit_traits", "What sets the worst-fitted nearby M dwarfs apart", "Gaia DR3; GALEX AIS by CDS XMatch",
            "the MARCS misfits of gaia_xp_worst_fits_nearby with no companion, emission, blend or extinction behind them: %d" % len(m), os.path.abspath(__file__), "xpmisfits", len(m), "stars", dict(controls=controls),
            "%d stars of the same colour and magnitude within 100 pc with ordinary fits" % len(c), "a property the misfits share and their peers lack",
            "; ".join("%s: %s against %s (p %s)" % (v["label"], v["misfits"] if isinstance(v["misfits"], str) else "%.2f" % v["misfits"],
                                                   v["peers"] if isinstance(v["peers"], str) else "%.2f" % v["peers"], "%.3g" % v["p"] if v["p"] is not None else "-") for v in out.values()),
            [dict(target="gaia_dr3:%s" % x["id"], stat=round(x["dmg"], 2), unit="absolute G offset below the main sequence in robust widths", vet=x["vet"],
                  note="J-Ks offset %s widths, BP-G excess %+.1f widths, %.0f km/s" % ("%+.1f" % x["jkres"] if x["jkres"] is not None else "-", x["blue"], x["vt"])) for x in m])
        print("recorded")


misfitspectra_DOC = r"""Spectral classes of the worst-fitted nearby M dwarfs: the MARCS misfits the trait test examined, carried to 2014 and looked up in LAMOST DR5's low-resolution catalogue within 3 arcsec, and looked up in SDSS DR17's spectra within 3 arcsec of their 2005 positions, each class (dM, sdM, esdM or usdM in LAMOST's scheme, the subclass in SDSS's) read against the trait test's verdict; a subdwarf class confirms a metal-poor star, a dwarf class leaves its misfit open, and a pipeline type with no metallicity class decides neither.

    python -I seti.py composition misfitspectra [record] [--arcsec 3]"""


def misfitspectra_main(record=False, arcsec=3.0):
    m = json.load(open(core.results("xp_misfits_traits.json")))["misfits"]

    def carried(x, epoch):
        return x["ra"] + (epoch - 2016.0) * x["pmra"] / 3.6e6 / math.cos(math.radians(x["dec"])), x["dec"] + (epoch - 2016.0) * x["pmdec"] / 3.6e6
    lam = {}
    for x in m:
        ra, de = carried(x, 2014.0)
        rows = core.tap("vizier", "SELECT RAJ2000, DEJ2000, Class, SubClass, snrr FROM \"V/164/dr5\" WHERE 1 = CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), CIRCLE('ICRS', %.6f, %.6f, %.6f))" % (
            ra, de, arcsec / 3600.0), timeout=120)
        if rows:
            lam[x["id"]] = [(str(r.get("Class") or "").strip(), str(r.get("SubClass") or "").strip(), core.num(r.get("snrr"))) for r in rows]
    sd = {}
    for x in m:
        ra, de = carried(x, 2005.0)
        d = arcsec / 3600.0
        try:
            rows = core.sdss_sql("SELECT TOP 3 s.class, s.subClass, s.ra, s.dec, s.snMedian FROM SpecObj AS s WHERE s.ra BETWEEN %.6f AND %.6f AND s.dec BETWEEN %.6f AND %.6f" % (
                ra - d / math.cos(math.radians(de)), ra + d / math.cos(math.radians(de)), de - d, de + d))
        except Exception:
            rows = []
        if rows:
            sd[x["id"]] = [(str(r.get("class")), str(r.get("subClass")), core.num(r.get("snMedian"))) for r in rows]
    for x in m:
        x["lamost"], x["sdss"] = lam.get(x["id"], []), sd.get(x["id"], [])
        tags = " ".join(t[1] for t in x["lamost"] + x["sdss"])
        x["spec_sub"] = bool(re.search(r"\b(sd|esd|usd)M", tags))
        x["spec_dwarf"] = bool(re.search(r"\bdM", tags)) and not x["spec_sub"]
        x["spec_plain"] = bool(re.search(r"\bM\d", tags)) and not x["spec_sub"] and not x["spec_dwarf"]
    n_spec = sum(1 for x in m if x["lamost"] or x["sdss"])
    print("%d misfits; with a LAMOST DR5 or SDSS DR17 spectrum: %d; classed as subdwarfs %d, as dwarfs %d, by a pipeline type with no metallicity class %d" % (
        len(m), n_spec, sum(x["spec_sub"] for x in m), sum(x["spec_dwarf"] for x in m), sum(x["spec_plain"] for x in m)))
    for x in m:
        print("  Gaia DR3 %s (trait verdict %s): LAMOST %s; SDSS %s" % (x["id"], x.get("vet") or "open", x["lamost"] or "-", x["sdss"] or "-"))
    if record:
        core.record_run(
            "gaia_xp_misfit_spectra", "Spectral classes of the worst-fitted nearby M dwarfs", "LAMOST DR5 (VizieR V/164); SDSS DR17 SpecObj",
            "the misfits of gaia_xp_misfit_traits: %d" % len(m), os.path.abspath(__file__), "misfitspectra", len(m), "stars", dict(arcsec=arcsec),
            "the trait test's verdicts", "a spectral class that confirms or overturns the metal-poor verdict",
            "%d with spectra; subdwarf classes %d, dwarf classes %d, pipeline types with no metallicity class %d" % (
                n_spec, sum(x["spec_sub"] for x in m), sum(x["spec_dwarf"] for x in m), sum(x["spec_plain"] for x in m)),
            [dict(target="gaia_dr3:%s" % x["id"], stat=None, unit="spectral class", vet="metal-poor" if x["spec_sub"] else x.get("vet"),
                  note="LAMOST %s; SDSS %s" % (x["lamost"] or "-", x["sdss"] or "-")) for x in m])
        print("recorded")


subdwarfindex_DOC = r"""Subdwarf indices of the worst-fitted nearby M dwarfs: the misfits of the trait test that LAMOST DR5 or SDSS DR17 observed, their spectra fetched, put in air wavelengths at rest, and measured in the CaH2 (6814-6846 A), CaH3 (6960-6990 A) and TiO5 (7126-7135 A) bands against the 7042-7046 A pseudo-continuum, and the TiO-to-CaH ratio zeta of Lepine et al. (2013) classing each as a dwarf (zeta above 0.825), a subdwarf (0.5-0.825), an extreme subdwarf (0.2-0.5) or an ultra subdwarf; the bands measure metallicity directly, which the colours only infer.

    python -I seti.py composition subdwarfindex [record] [--arcsec 3]"""


def vac_to_air(w):
    """Air wavelengths from vacuum ones (Morton 2000), in angstroms."""
    s2 = (1e4 / w) ** 2
    return w / (1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2))


def mdwarf_indices(wave_air_rest, flux):
    """CaH2, CaH3, TiO5 and the zeta of Lepine et al. (2013) from a rest-frame spectrum in air wavelengths."""
    ok = np.isfinite(flux) & np.isfinite(wave_air_rest)
    w, f = wave_air_rest[ok], flux[ok]

    def band(lo, hi):
        # the band's mean flux on a fine grid, so a window narrower than a few pixels is still measured
        if w.size < 2 or lo < w.min() or hi > w.max():
            return float("nan")
        return float(np.mean(np.interp(np.linspace(lo, hi, 200), w, f)))
    cont = band(7042.0, 7046.0)
    cah2, cah3, tio5 = band(6814.0, 6846.0) / cont, band(6960.0, 6990.0) / cont, band(7126.0, 7135.0) / cont
    cah = cah2 + cah3
    tio5_sun = 0.622 - 1.906 * cah + 2.211 * cah ** 2 - 0.588 * cah ** 3
    zeta = (1 - tio5) / (1 - tio5_sun) if tio5_sun < 1 else float("nan")
    return cah2, cah3, tio5, zeta


def zeta_class(z):
    """The metallicity class of an M dwarf from its zeta (Lepine et al. 2007)."""
    if not np.isfinite(z):
        return "unmeasured"
    return "dwarf" if z > 0.825 else "subdwarf" if z > 0.5 else "extreme subdwarf" if z > 0.2 else "ultra subdwarf"


def subdwarfindex_main(record=False, arcsec=3.0):
    import gzip
    from astropy.io import fits
    m = json.load(open(core.results("xp_misfits_traits.json")))["misfits"]
    os.makedirs(core.data("mdwarf_spectra"), exist_ok=True)

    def carried(x, epoch):
        return x["ra"] + (epoch - 2016.0) * x["pmra"] / 3.6e6 / math.cos(math.radians(x["dec"])), x["dec"] + (epoch - 2016.0) * x["pmdec"] / 3.6e6
    out = []
    for x in m:
        spectra = []
        ra, de = carried(x, 2014.0)
        for r in core.tap("vizier", "SELECT ObsID, PlanId, spId, FiberId, LMJD, z, snrr FROM \"V/164/dr5\" WHERE 1 = CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), CIRCLE('ICRS', %.6f, %.6f, %.6f))" % (
                ra, de, arcsec / 3600.0), timeout=120):
            plan, lmjd, sp, fib = str(r["PlanId"]).strip(), int(float(r["LMJD"])), int(float(r["spId"])), int(float(r["FiberId"]))
            path = core.data("mdwarf_spectra", "spec-%d-%s_sp%02d-%03d.fits.gz" % (lmjd, plan, sp, fib))
            if not os.path.exists(path):
                try:
                    open(path, "wb").write(core.get("https://dr5.lamost.org/v3/spectrum/fits/%s" % str(r["ObsID"]).strip(), timeout=120))
                except Exception as e:
                    spectra.append(dict(source="LAMOST %s" % r["ObsID"], error=str(e)[:80]))
                    continue
            try:
                with fits.open(gzip.open(path)) as h:
                    d = h[0].data
                    flux, wave = np.array(d[0], float), np.array(d[2], float)
                z = core.num(r["z"]) or 0.0
                spectra.append(dict(source="LAMOST %s" % r["ObsID"], snr=core.num(r["snrr"]), idx=mdwarf_indices(vac_to_air(wave) / (1 + z), flux)))
            except Exception as e:
                spectra.append(dict(source="LAMOST %s" % r["ObsID"], error=str(e)[:80]))
        ra, de = carried(x, 2005.0)
        d = arcsec / 3600.0
        try:
            rows = core.sdss_sql("SELECT TOP 3 s.plate, s.mjd, s.fiberID, s.run2d, s.z, s.snMedian FROM SpecObj AS s WHERE s.ra BETWEEN %.6f AND %.6f AND s.dec BETWEEN %.6f AND %.6f" % (
                ra - d / math.cos(math.radians(de)), ra + d / math.cos(math.radians(de)), de - d, de + d))
        except Exception:
            rows = []
        for r in rows:
            plate, mjd, fib, run2d = int(r["plate"]), int(r["mjd"]), int(r["fiberID"]), str(r["run2d"]).strip()
            path = core.data("mdwarf_spectra", "spec-%04d-%05d-%04d.fits" % (plate, mjd, fib))
            if not os.path.exists(path):
                base = "sdss" if run2d in ("26", "103", "104") else "eboss"
                try:
                    open(path, "wb").write(core.get("https://data.sdss.org/sas/dr17/%s/spectro/redux/%s/spectra/lite/%04d/spec-%04d-%05d-%04d.fits" % (base, run2d, plate, plate, mjd, fib), timeout=120))
                except Exception as e:
                    spectra.append(dict(source="SDSS %d-%d-%d" % (plate, mjd, fib), error=str(e)[:80]))
                    continue
            with fits.open(path) as h:
                t = h[1].data
                wave, flux = 10 ** np.array(t["loglam"], float), np.array(t["flux"], float)
            spectra.append(dict(source="SDSS %d-%d-%d" % (plate, mjd, fib), snr=core.num(r["snMedian"]), idx=mdwarf_indices(vac_to_air(wave) / (1 + (core.num(r["z"]) or 0.0)), flux)))
        good = [s for s in spectra if "idx" in s and np.isfinite(s["idx"][3])]
        best = max(good, key=lambda s: s["snr"] or 0) if good else None
        out.append(dict(id=x["id"], trait=x.get("vet"), spectra=spectra, best=best, cls=zeta_class(best["idx"][3]) if best else None))
    meas = [o for o in out if o["cls"]]
    print("%d misfits; %d with a measurable LAMOST or SDSS spectrum: %s" % (len(out), len(meas), {c: sum(1 for o in meas if o["cls"] == c) for c in
                                                                                                    ("dwarf", "subdwarf", "extreme subdwarf", "ultra subdwarf")}))
    for o in out:
        if o["spectra"]:
            b = o["best"]
            print("  Gaia DR3 %s (trait %s): %s" % (o["id"], o["trait"] or "open", "%s S/N %s: CaH2 %.3f, CaH3 %.3f, TiO5 %.3f, zeta %.2f, %s" % (
                b["source"], b["snr"], b["idx"][0], b["idx"][1], b["idx"][2], b["idx"][3], o["cls"]) if b else o["spectra"]))
    json.dump(out, open(core.results("misfit_subdwarf_indices.json"), "w"), indent=1, default=str)
    if record:
        core.record_run(
            "gaia_xp_misfit_subdwarf_indices", "Subdwarf indices of the worst-fitted nearby M dwarfs", "LAMOST DR5 spectra (VizieR V/164 and the DR5 archive); SDSS DR17 spectra",
            "the misfits of gaia_xp_misfit_traits: %d, %d with a measurable spectrum" % (len(out), len(meas)), os.path.abspath(__file__), "subdwarfindex", len(meas), "stars", dict(arcsec=arcsec),
            "the zeta boundaries of Lepine et al. (2007): 0.825, 0.5 and 0.2", "a misfit whose bands show solar metallicity, which the subdwarf verdict cannot explain",
            "%s" % {c: sum(1 for o in meas if o["cls"] == c) for c in ("dwarf", "subdwarf", "extreme subdwarf", "ultra subdwarf")},
            [dict(target="gaia_dr3:%s" % o["id"], stat=round(o["best"]["idx"][3], 3), unit="zeta TiO/CaH", vet="metal-poor" if o["cls"] in ("subdwarf", "extreme subdwarf", "ultra subdwarf") else None,
                  note="%s, CaH2 %.3f, CaH3 %.3f, TiO5 %.3f, %s; trait verdict %s" % (o["best"]["source"], o["best"]["idx"][0], o["best"]["idx"][1], o["best"]["idx"][2], o["cls"], o["trait"] or "open"))
             for o in meas])
        print("recorded")


ghosts_DOC = r"""Ghost spectra: each rest-frame stack of the ESO laser-line runs (HARPS, ESPRESSO, FEROS and UVES, 4100-5600 A) covering 60 per cent of that range whose nearest twin correlates at 0.9, put on a grid uniform in velocity with its broad wings removed, its lines taken out by the scaled median of its most alike stars, and what remains correlated with the star's own lines at every shift from 1,000 to 60,000 km/s either way; a sail, a fleet or any surface moving fast through the system reflects the star's own light, which arrives as a faint copy of the star's lines shifted by its speed, where no natural body near a star moves at thousands of kilometres a second and a companion's lines sit within a few hundred; each correlation is in units of the star's own line strength, so that it reads as the copy's depth, and judged against its own noise, which grows as the overlap of the spectrum's coverage with itself shrinks, shifts leaving less than a fifth of it unread, a peak at a shift other stars' peaks share belongs to the spectra rather than to a body, and the depth a copy needs to stand out is measured by laying one into every stack.

    python -I seti.py composition ghosts [record] [--v-min 1000] [--v-max 60000] [--z-min 7] [--twins 6] [--inject 0.003] [--coverage 0.6] [--twin-r 0.9] [--shared-kms 30] [--min-overlap 0.2]"""

GHOSTS_DV = 1.0  # km/s per step of the velocity grid, a fifth of a line's width


def ghosts_lines(T):
    """A stack's line depths on a grid uniform in velocity, wings and continuum slopes removed by a running mean of 4 A, gaps set to zero."""
    from scipy.ndimage import uniform_filter1d
    L = 1.0 - np.asarray(T, float)
    L = np.where(np.isfinite(L), L, 0.0)
    L = L - uniform_filter1d(L, 401, mode="nearest")
    lng = np.arange(np.log(elements_GRID[0]), np.log(elements_GRID[-1]), GHOSTS_DV / elements_C)
    return np.interp(np.exp(lng), elements_GRID, L)


def ghosts_shifts(a, kmax):
    """A circular correlation's values at shifts from -kmax to kmax steps."""
    return np.concatenate([a[::-1][:kmax][::-1], a[:kmax + 1]])


def ghosts_xcorr(R, L, kmax):
    """The correlation of a residual with line depths shifted from -kmax to kmax steps (positive redward), in units of the lines' own strength over the pixels the two overlap, with that overlap as a share of the spectrum's coverage."""
    m = 2 * len(L)
    c = ghosts_shifts(np.fft.irfft(np.fft.rfft(R, m) * np.conj(np.fft.rfft(L, m)), m), kmax)
    cover = (L != 0).astype(float)
    F = np.fft.rfft(cover, m)
    share = np.maximum(ghosts_shifts(np.fft.irfft(F * np.conj(F), m), kmax), 0) / max(cover.sum(), 1.0)
    return c / (float(np.sum(L * L)) * np.maximum(share, 1e-9)), share


def ghosts_residual(Li, twins):
    """A star's line depths less the scaled median of its most alike stars' line depths."""
    M = np.median(twins, axis=0)
    return Li - float(Li @ M) / max(float(M @ M), 1e-12) * M


def ghosts_main(record=False, v_min=1000.0, v_max=60000.0, z_min=7.0, twins=6, inject=0.003, coverage=0.6, twin_r=0.9, shared_kms=30.0, min_overlap=0.2):
    from collections import Counter
    stacks = {}
    for inst in ("HARPS", "ESPRESSO", "FEROS", "UVES"):
        for n, (T, nspec, v, rv) in elements_stacks(inst).items():
            stacks["%s:%s" % (inst, n)] = (T, nspec)
    names = sorted(stacks)
    L = np.array([ghosts_lines(stacks[n][0]) for n in names])
    # a stack covering little of the range leaves too few lines to correlate
    keep = (L != 0).mean(axis=1) >= coverage
    names, L = [n for n, k in zip(names, keep) if k], L[keep]
    kmax = int(v_max / GHOSTS_DV)
    lag = np.arange(-kmax, kmax + 1) * GHOSTS_DV
    far = np.abs(lag) >= v_min
    D = L[:, ::10] / np.maximum(np.linalg.norm(L[:, ::10], axis=1, keepdims=True), 1e-12)
    sim = D @ D.T
    np.fill_diagonal(sim, -2)
    shift = int(20000 / GHOSTS_DV)
    out, rec = [], []
    for i, n in enumerate(names):
        ref = np.argsort(sim[i])[::-1][:twins]
        # a star whose nearest twin is unlike it keeps its own lines in the residual, which then echoes itself
        if sim[i, ref[0]] < twin_r:
            continue
        # the correlation's noise grows as the overlap of the two spectra shrinks, as one over its square root; shifts leaving less than a fifth of the coverage are not read
        c, share = ghosts_xcorr(ghosts_residual(L[i], L[ref]), L[i], kmax)
        grow = np.sqrt(share)
        use = far & (share >= min_overlap)
        c = c * grow
        width = 1.4826 * float(np.median(np.abs(c[use] - np.median(c[use]))))
        z = np.where(use, (c - np.median(c[use])) / max(width, 1e-12), -np.inf)
        j = int(np.argmax(z))
        out.append(dict(star=n, nspec=stacks[n][1], z=float(z[j]), v=float(lag[j]), depth=float(c[j] / grow[j]), width=width, twin=names[ref[0]], sim=float(sim[i, ref[0]])))
        # the same star with a copy of its own lines laid 20,000 km/s redward at depth INJECT
        g = L[i].copy()
        g[shift:] += inject * L[i][:-shift]
        cg = ghosts_xcorr(ghosts_residual(g, L[ref]), g, kmax)[0] * grow
        rec.append(float((cg[kmax + shift] - np.median(cg[use])) / max(width, 1e-12)))
    rec = np.array(rec)
    zs = np.array([o["z"] for o in out])
    hits = [o for o in out if o["z"] >= z_min]
    # a sail moves at its own speed, so a shift that other stars' strongest peaks share belongs to the spectra, not to a body
    for o in hits:
        o["shared"] = [p["star"] for p in out if p is not o and p["z"] >= 5 and abs(p["v"] - o["v"]) <= shared_kms]
        o["vet"] = "shared-feature" if o["shared"] else None
    inst = Counter(o["star"].split(":")[0] for o in out)
    print("%d stacks with a twin at r %g (%s); the strongest copy of each star's own lines between %g and %g km/s either way: median %.1f widths, the largest %.1f, %d at %g or more, %d of them at a shift other stars share; a copy of %.1f per cent depth at 20,000 km/s scores a median %.1f; the median width is %.2g of the star's line strength" % (
        len(out), twin_r, ", ".join("%s %d" % kv for kv in sorted(inst.items())), v_min, v_max, float(np.median(zs)), float(zs.max()), len(hits), z_min,
        sum(o["vet"] == "shared-feature" for o in hits), 100 * inject, float(np.median(rec)), float(np.median([o["width"] for o in out]))))
    for o in sorted(hits, key=lambda o: -o["z"]):
        print("  %-30s %3d spectra  z %5.1f at %+9.1f km/s, depth %.4f (width %.4f); shared with %s" % (o["star"], o["nspec"], o["z"], o["v"], o["depth"], o["width"], ", ".join(o["shared"]) or "none"))
    json.dump(dict(stars=out, injected=rec.tolist()), open(core.results("ghost_spectra.json"), "w"), indent=1)
    if record:
        core.record_run(
            "eso_ghost_spectra", "Ghost spectra", "ESO archive spectra of the laser-line runs: rest-frame stacks of HARPS, ESPRESSO, FEROS and UVES, 4100-5600 A",
            "stacks covering %g of the range with a twin at r %g: %d (%s)" % (coverage, twin_r, len(out), ", ".join("%s %d" % kv for kv in sorted(inst.items()))),
            os.path.abspath(__file__), "ghosts", len(out), "stars", dict(v_min=v_min, v_max=v_max, z_min=z_min, twins=twins, inject=inject, coverage=coverage, twin_r=twin_r, shared_kms=shared_kms),
            "each star's own correlation over all shifts beyond %g km/s, in robust widths; its strongest peak's median is %.1f" % (v_min, float(np.median(zs))),
            "a copy of the star's lines at %g-%g km/s either way; one of %.1f per cent depth scores a median %.1f, so about %.1f per cent stands at %g" % (
                v_min, v_max, 100 * inject, float(np.median(rec)), 100 * inject * z_min / max(float(np.median(rec)), 1e-9), z_min),
            "%d at %g widths or more, the largest %.1f; %d at a shift other stars share" % (len(hits), z_min, float(zs.max()), sum(o["vet"] == "shared-feature" for o in hits)),
            [dict(target="gaia_dr3:%s" % o["star"].split(":")[1], stat=round(o["z"], 1), unit="robust widths of the star's correlation", vet=o["vet"],
                  note="%s stack of %d spectra; copy at %+.0f km/s of depth %.4f; shared with %s" % (o["star"].split(":")[0], o["nspec"], o["v"], o["depth"], ", ".join(o["shared"]) or "none"))
             for o in hits])
        print("recorded")
    return out, hits, rec

combs_DOC = r"""Optical frequency combs: each rest-frame stack of the ESO laser-line runs (HARPS, ESPRESSO, FEROS and UVES, 4100-5600 A) covering 60 per cent of the range whose nearest twin correlates at 0.9, its lines taken out by the scaled median of its six most alike stars, the residual laid on a grid uniform in frequency and its Fourier power read at every line spacing from 20 GHz to 1 THz against the power's own running median, the twins' residuals read the same way; a mode-locked laser's light is hundreds of lines at one spacing, each too faint to notice alone, which add into one peak at the spacing and its harmonics, where the bands of molecules space their lines unevenly and the twins share them; a comb of known depth laid into every stack measures the reach.

    python -I seti.py composition combs [record] [--ghz-min 20] [--ghz-max 1000] [--ratio-min 30] [--twins 6] [--inject 0.002]"""


def combs_power(R, lnlam):
    """The Fourier power of a residual laid on a grid uniform in frequency, with the line spacing (GHz) of each Fourier bin."""
    nu = 299792.458e3 / np.exp(lnlam) * 1e10 / 1e9  # GHz, from wavelength in Angstrom
    o = np.argsort(nu)
    grid = np.linspace(nu[o][0], nu[o][-1], len(nu))
    x = np.interp(grid, nu[o], R[o]) * np.hanning(len(grid))
    p = np.abs(np.fft.rfft(x)) ** 2
    q = np.fft.rfftfreq(len(grid), d=grid[1] - grid[0])
    with np.errstate(divide="ignore"):
        return p, np.where(q > 0, 1.0 / q, np.inf)


def combs_inject(lnlam, spacing_ghz, depth, width_kms=3.0):
    """Line depths of a comb of unresolved lines at one spacing in frequency across the grid."""
    nu = 299792.458e3 / np.exp(lnlam) * 1e10 / 1e9
    lines = np.arange(nu.min() + spacing_ghz / 3, nu.max(), spacing_ghz)
    sig = width_kms / 299792.458 * nu
    k = np.clip(np.searchsorted(-nu, -lines), 0, len(nu) - 1)
    out = np.zeros(len(nu))
    for i, c in zip(k, lines):
        lo, hi = max(i - 30, 0), min(i + 30, len(nu))
        out[lo:hi] += depth * np.exp(-0.5 * ((nu[lo:hi] - c) / sig[lo:hi]) ** 2)
    return out


def combs_local(p):
    """Fourier power over its own running median across 201 bins, the level a comb's spike stands above."""
    from scipy.ndimage import median_filter
    return p / np.maximum(median_filter(p, size=201, mode="nearest"), 1e-30)


def combs_main(record=False, ghz_min=20.0, ghz_max=1000.0, ratio_min=30.0, twins=6, inject=0.002, coverage=0.6, twin_r=0.9):
    stacks = {}
    for inst in ("HARPS", "ESPRESSO", "FEROS", "UVES"):
        for n, (T, nspec, v, rv) in elements_stacks(inst).items():
            stacks["%s:%s" % (inst, n)] = (T, nspec)
    names = sorted(stacks)
    L = np.array([ghosts_lines(stacks[n][0]) for n in names])
    keep = (L != 0).mean(axis=1) >= coverage
    names, L = [n for n, k in zip(names, keep) if k], L[keep]
    lnlam = np.arange(np.log(elements_GRID[0]), np.log(elements_GRID[-1]), GHOSTS_DV / elements_C)[:L.shape[1]]
    D = L[:, ::10] / np.maximum(np.linalg.norm(L[:, ::10], axis=1, keepdims=True), 1e-12)
    sim = D @ D.T
    np.fill_diagonal(sim, -2)
    good = [i for i in range(len(names)) if sim[i].max() >= twin_r]
    refs = {i: np.argsort(sim[i])[::-1][:twins] for i in good}
    P, sp = {}, None
    for i in set(good) | {j for r in refs.values() for j in r}:
        R = ghosts_residual(L[i], L[refs[i]]) if i in refs else ghosts_residual(L[i], L[np.argsort(sim[i])[::-1][:twins]])
        P[i], sp = combs_power(R, lnlam)
    band = (sp >= ghz_min) & (sp <= ghz_max)
    comb = combs_inject(lnlam, 50.0, inject)
    Q = {i: combs_local(P[i]) for i in P}
    out = []
    for i in good:
        ratio = np.where(band, Q[i], 0)
        k = int(np.argmax(ratio))
        # the same star with a comb of 50 GHz spacing laid in at depth INJECT
        pg, _ = combs_power(ghosts_residual(L[i] + comb, L[refs[i]]), lnlam)
        kk = int(np.argmin(np.abs(sp - 50.0)))
        rec = float(np.max(combs_local(pg)[kk - 2:kk + 3]))
        h2 = int(np.argmin(np.abs(sp - sp[k] / 2))) if np.isfinite(sp[k]) else k
        out.append(dict(star=names[i], nspec=stacks[names[i]][1], ratio=float(ratio[k]), ghz=float(sp[k]), harmonic=float(np.max(Q[i][h2 - 1:h2 + 2])), rec=rec,
                        twins=float(np.median([np.max(Q[j][k - 1:k + 2]) for j in refs[i]]))))
    rs = np.array([o["ratio"] for o in out])
    rec = np.array([o["rec"] for o in out])
    hits = [o for o in out if o["ratio"] >= ratio_min]
    for o in hits:
        # a spacing other stars' strongest peaks share belongs to the instrument or the spectra
        o["shared"] = [p["star"] for p in out if p is not o and p["ratio"] >= ratio_min / 3 and abs(p["ghz"] / o["ghz"] - 1) < 0.01]
        o["vet"] = "shared-feature" if o["shared"] or o["twins"] >= ratio_min / 3 else None
    print("%d stacks; the strongest Fourier power of each residual over its own running median between %g GHz and %g THz of line spacing: median %.1f, the largest %.1f, %d at %g or more (%d shared with other stars or the twins); a comb of %.1f per cent depth at 50 GHz gives a median %.0f" % (
        len(out), ghz_min, ghz_max / 1000, float(np.median(rs)), float(rs.max()), len(hits), ratio_min, sum(o["vet"] == "shared-feature" for o in hits), 100 * inject, float(np.median(rec))))
    for o in sorted(out, key=lambda o: -o["ratio"])[:12]:
        print("  %-30s %3d spectra  power %6.1f times its level at %7.1f GHz, second harmonic %5.1f, twins %5.1f; shared %s" % (o["star"], o["nspec"], o["ratio"], o["ghz"], o["harmonic"], o["twins"], ", ".join(o.get("shared", [])) or "-"))
    json.dump(out, open(core.results("optical_combs.json"), "w"), indent=1)
    if record:
        core.record_run(
            "eso_optical_combs", "Optical frequency combs", "ESO archive spectra of the laser-line runs: rest-frame stacks of HARPS, ESPRESSO, FEROS and UVES, 4100-5600 A",
            "stacks covering %g of the range with a twin at r %g: %d" % (coverage, twin_r, len(out)), os.path.abspath(__file__), "combs", len(out), "stars",
            dict(ghz_min=ghz_min, ghz_max=ghz_max, ratio_min=ratio_min, twins=twins, inject=inject, coverage=coverage, twin_r=twin_r),
            "each residual's own running median of power over 201 Fourier bins; each star's strongest ratio has median %.1f" % float(np.median(rs)),
            "a residual's Fourier power at one line spacing of %g-%g GHz %g times its running median; a comb of %.1f per cent depth at 50 GHz gives a median %.0f" % (ghz_min, ghz_max, ratio_min, 100 * inject, float(np.median(rec))),
            "%d at %g or more, the largest %.1f; %d at a spacing other stars share" % (len(hits), ratio_min, float(rs.max()), sum(o["vet"] == "shared-feature" for o in hits)),
            [dict(target="gaia_dr3:%s" % o["star"].split(":")[1], stat=round(o["ratio"], 1), unit="residual power over the twins' at the strongest spacing", vet=o["vet"],
                  note="%s stack of %d spectra; spacing %.1f GHz, second harmonic %.1f, twins %.1f; shared with %s" % (o["star"].split(":")[0], o["nspec"], o["ghz"], o["harmonic"], o["twins"], ", ".join(o["shared"]) or "none"))
             for o in hits])
        print("recorded")


giants_DOC = r"""Planet-frame compositions of the giant planets: every archived high-resolution spectrum (ESO HARPS, ESPRESSO, NIRPS, CRIRES and UVES; CFHT SPIRou; Keck NIRSPEC) of the hosts of giant planets brighter than V 11, each spectrum placed on the planet's orbit from the archive's ephemeris (transit, secondary eclipse, dayside or other phase); the transits cross-correlated in the planet's frame with templates of nine atoms in the optical and of CO and H2O in the near-infrared, summed on a map over K_p and rest velocity and read against a sham of out-of-transit rows; reflected starlight of the planets without a transit, the star's own spectrum as the template; dayside emission in SPIRou spectra.

    python -I seti.py composition giants plan [--vmax 11]         archives queried, spectra placed on the orbits: ROOT/plan.json
    python -I seti.py composition giants fetch [--streams 4] [--planets NAME ...] [--instruments INST ...] [--min-in 4] [--night-per-in 2]
    python -I seti.py composition giants search [--planets NAME ...] [--instruments INST ...] [--species ...] [--sigma 4] [--temp K] [--sysrem 0] [--weight log|sqrt|linear] [--inject 0] [--no-whiten] [--no-shadow] [--redo]
    python -I seti.py composition giants reflected [--planets NAME ...] [--max-spectra 300] [--components 5] [--probe 1e-4] [--redo]
    python -I seti.py composition giants emission [--planets NAME ...] [--species CO H2O] [--sysrem 6] [--components 0] [--probe 3e-4] [--redo]
    python -I seti.py composition giants record [--sigma 4]"""

giants_ROOT = core.data("giants")


def giants_planets(vmax=11.0):
    """The giant planets (0.1 Jupiter masses or 0.5 Jupiter radii and more) of hosts brighter than V VMAX with their ephemerides, the time of inferior conjunction of a radial-velocity planet taken from its periastron and argument of periastron."""
    rows = core.exoplanets("SELECT pl_name, hostname, ra, dec, sy_dist, st_teff, st_rad, st_mass, st_radv, st_vsin, pl_bmassj, pl_radj, pl_ratror, pl_orbsmax, "
                           "pl_orbper, pl_orbeccen, pl_orblper, pl_orbtper, pl_tranmid, pl_trandur, pl_eqt, sy_vmag, sy_kmag, tran_flag FROM pscomppars WHERE "
                           "(pl_bmassj >= 0.1 OR pl_radj >= 0.5) AND sy_vmag < %g AND pl_orbper IS NOT NULL ORDER BY sy_vmag" % vmax)
    out = []
    # hosts the archive lists without a systemic velocity take SIMBAD's
    missing = sorted({r["hostname"] for r in rows if core.num(r["st_radv"]) is None})
    simbad = {}
    for i in range(0, len(missing), 100):
        names = "', '".join(n.replace("'", "''") for n in missing[i:i + 100])
        try:
            for s in core.tap("simbad", "SELECT ident.id, basic.rvz_radvel FROM basic JOIN ident ON ident.oidref = basic.oid WHERE ident.id IN ('%s')" % names):
                if core.num(s["rvz_radvel"]) is not None:
                    simbad[s["id"]] = core.num(s["rvz_radvel"])
        except Exception:
            pass
    for r in rows:
        p = {k: (r[k] if k in ("pl_name", "hostname") else core.num(r[k])) for k in r}
        if p["st_radv"] is None:
            p["st_radv"] = simbad.get(p["hostname"])
        if p["pl_tranmid"] is None and p["pl_orbtper"] is not None and p["pl_orblper"] is not None:
            # inferior conjunction follows periastron by (90 - omega) degrees of mean anomaly on a near-circular orbit
            p["pl_tranmid"] = p["pl_orbtper"] + p["pl_orbper"] * ((90.0 - p["pl_orblper"]) % 360.0) / 360.0
        if p["pl_tranmid"] is None:
            continue
        if p["pl_trandur"] is None and p["st_rad"] and p["pl_orbsmax"]:
            # a central transit's duration from the star's radius and the orbit's size
            p["pl_trandur"] = 24.0 * p["pl_orbper"] / math.pi * math.asin(min(p["st_rad"] * 0.00465047 / p["pl_orbsmax"], 1.0))
        out.append(p)
    return out


def giants_place(p, rows):
    """A planet's archived spectra by instrument with each spectrum's orbital phase and place: transit, eclipse, day (phase 0.3-0.7) or other."""
    tc, per = p["pl_tranmid"] - 2400000.5, p["pl_orbper"]
    half = (p["pl_trandur"] or 3.0) / 48.0
    out = {}
    for r in rows:
        mid = 0.5 * (r["t0"] + r["t1"])
        phase = ((mid - tc) / per) % 1.0
        d = min(phase, 1.0 - phase) * per
        place = "transit" if d < half and p.get("tran_flag") else "eclipse" if abs(phase - 0.5) * per < half and p.get("tran_flag") else "day" if 0.3 < phase < 0.7 else "other"
        out.setdefault(r["inst"], []).append(dict(r, phase=round(phase, 4), place=place))
    return out


def giants_plan(vmax=11.0):
    """The giant planets with their archived spectra placed on their orbits, into ROOT/plan.json, the archive answers kept in ROOT/archives.json."""
    os.makedirs(giants_ROOT, exist_ok=True)
    planets = giants_planets(vmax)
    hosts = {}
    for p in planets:
        hosts.setdefault(p["hostname"], dict(ra=p["ra"], dec=p["dec"]))
    apath = os.path.join(giants_ROOT, "archives.json")
    arch = json.load(open(apath)) if os.path.exists(apath) else {}
    todo = sorted(h for h in hosts if h not in arch)

    def one(name):
        try:
            return name, core.archive_spectra(hosts[name]["ra"], hosts[name]["dec"]), None
        except Exception as e:
            return name, None, str(e)[:200]
    errors = {}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        # answers are kept as they arrive, so a run cut short keeps what it had
        for i, (name, rows, err) in enumerate(ex.map(one, todo)):
            if rows is None:
                errors[name] = err
            else:
                arch[name] = rows
            if (i + 1) % 25 == 0:
                json.dump(arch, open(apath, "w"))
    json.dump(arch, open(apath, "w"))
    plan = []
    for p in planets:
        placed = giants_place(p, arch.get(p["hostname"], []))
        if placed:
            plan.append(dict(p, instruments=placed))
    json.dump(plan, open(os.path.join(giants_ROOT, "plan.json"), "w"), indent=1)
    print("%d giant planets of %d hosts brighter than V %g; %d hosts with archived spectra (%d archives not answering); %d planets with spectra"
          % (len(planets), len(hosts), vmax, sum(1 for h in hosts if arch.get(h)), len(errors), len(plan)))
    print("%-18s %-9s %7s %7s %5s %5s" % ("planet", "inst", "transit", "eclipse", "day", "other"))
    for p in plan:
        for inst, rows in sorted(p["instruments"].items()):
            c = {k: sum(1 for r in rows if r["place"] == k) for k in ("transit", "eclipse", "day", "other")}
            if c["transit"] or c["eclipse"] or c["day"] >= 5:
                print("%-18s %-9s %7d %7d %5d %5d" % (p["pl_name"], inst, c["transit"], c["eclipse"], c["day"], c["other"]))


giants_SPECIES = ("Fe I", "Fe II", "Ti I", "V I", "Cr I", "Na I", "Ca I", "Mg I", "K I")
giants_MOLECULES = ("H2O", "CO")
# the two bands: the optical spectrographs and the near-infrared ones, each with its grid's span (A) and the telluric bands left out
giants_BANDS = {"optical": dict(lo=3800.0, hi=7880.0, telluric=((6270.0, 6330.0), (6440.0, 6620.0), (6860.0, 6970.0), (7150.0, 7360.0), (7580.0, 7720.0)),
                                instruments=("ESPRESSO", "HARPS", "UVES")),
                "nir": dict(lo=9500.0, hi=25000.0, telluric=((11050.0, 11700.0), (13300.0, 14900.0), (17900.0, 19700.0), (24000.0, 25000.0)),
                            instruments=("SPIRou", "NIRPS", "CRIRES", "NIRSPEC"))}
# sky emission at the telescope's wavelengths (A, air): the sodium nightglow doublet, masked 6 km/s either side in every spectrum
giants_SKY = (5889.95, 5895.92)
giants_LOGGRID = {}


def giants_band(inst):
    """The band an instrument's spectra belong to."""
    return next(b for b, d in giants_BANDS.items() if inst in d["instruments"])


def giants_grid(band="optical"):
    """A band's common wavelength grid (A, air), uniform in velocity at 1 km/s."""
    if band not in giants_LOGGRID:
        lo, hi = giants_BANDS[band]["lo"], giants_BANDS[band]["hi"]
        n = int(np.log(hi / lo) * 299792.458)
        giants_LOGGRID[band] = lo * np.exp(np.arange(n) / 299792.458)
    return giants_LOGGRID[band]


giants_KURUCZ = os.path.join(giants_ROOT, "lines", "gfall08oct17.dat")
giants_CODES = {"Fe I": "26.00", "Fe II": "26.01", "Ti I": "22.00", "Ti II": "22.01", "V I": "23.00", "Cr I": "24.00", "Cr II": "24.01", "Mn I": "25.00",
                "Ni I": "28.00", "Co I": "27.00", "Na I": "11.00", "Ca I": "20.00", "Ca II": "20.01", "Mg I": "12.00", "K I": "19.00", "Sc II": "21.01",
                "Y II": "39.01", "Ba II": "56.01", "Sr II": "38.01", "Li I": "3.00"}


def giants_kurucz(species, band="optical"):
    """A species' lines within a band's span from Kurucz's gfall list (air wavelengths above 200 nm): wavelengths (A), log gf and lower-level energies (cm^-1), parsed once into ROOT/lines/<species>_<band>.npz."""
    cache = os.path.join(giants_ROOT, "lines", species.replace(" ", "_") + ("" if band == "optical" else "_" + band) + ".npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["lam"], d["loggf"], d["elow"]
    code = giants_CODES[species]
    lo, hi = giants_BANDS[band]["lo"], giants_BANDS[band]["hi"]
    lam, loggf, elow = [], [], []
    with open(giants_KURUCZ, encoding="ascii", errors="replace") as fh:
        for line in fh:
            if line[18:24].strip() != code:
                continue
            try:
                w, gf, e = float(line[0:11]) * 10.0, float(line[11:18]), abs(float(line[24:36]))
            except ValueError:
                # a line of the list's other layouts
                continue
            if lo <= w <= hi:
                lam.append(w)
                loggf.append(gf)
                elow.append(e)
    lam, loggf, elow = np.array(lam), np.array(loggf), np.array(elow)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, lam=lam, loggf=loggf, elow=elow)
    return lam, loggf, elow


giants_WEIGHT = "log"


giants_HITRAN = "https://hitran.org/lbl/api"
giants_HITRAN_ISO = {"H2O": 1, "CO": 26, "CH4": 32, "CO2": 7, "NH3": 45, "HCN": 23}


def giants_hitran(gas, lo_a=9500.0, hi_a=25000.0, step=100.0):
    """A gas's HITRAN lines of its main isotopologue between two wavelengths (A): wavenumber (cm^-1), intensity at 296 K and lower-level energy (cm^-1), read page by page per wavenumber range until a page comes back empty, kept in ROOT/lines/<gas>_hitran.npz."""
    import http.client
    import time
    import urllib.error
    import urllib.request
    cache = os.path.join(giants_ROOT, "lines", gas + "_hitran.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["nu"], d["sw"], d["elow"]

    def page(lo, hi):
        url = giants_HITRAN + "?" + urllib.parse.urlencode(dict(iso_ids_list=giants_HITRAN_ISO[gas], numin="%.6f" % lo, numax="%.6f" % hi, request_params="nu,sw,elower"))
        for attempt in range(4):
            try:
                body = urllib.request.urlopen(urllib.request.Request(url, headers=core.UA), timeout=300).read()
            except http.client.IncompleteRead as e:
                body = e.partial
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return np.zeros((0, 3))
                time.sleep(5 * (attempt + 1))
                continue
            except OSError:
                time.sleep(5 * (attempt + 1))
                continue
            rows = [r.split(",") for r in body.decode("ascii", "replace").split() if r.count(",") == 2]
            return np.array(rows[:-1] if len(rows) > 1 else rows, dtype=float).reshape(-1, 3)
        raise IOError("HITRAN did not answer for %s %g-%g" % (gas, lo, hi))

    def one(lo):
        hi, parts = lo + step, []
        while hi - lo > 1e-3:
            a = page(lo, hi)
            if not len(a):
                break
            parts.append(a)
            lo = a[-1, 0] + 1e-6
        return np.concatenate(parts) if parts else np.zeros((0, 3))
    starts = np.arange(1e8 / hi_a, 1e8 / lo_a, step)
    a = np.concatenate([x for x in core.pmap(one, starts, 4) if len(x)])
    a = np.unique(a, axis=0)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, nu=a[:, 0], sw=a[:, 1], elow=a[:, 2])
    return a[:, 0], a[:, 1], a[:, 2]


giants_KURUCZ_MOL = {}


def giants_kurucz_mol(gas, band="nir"):
    """A molecule's lines within a band's span from Kurucz's molecular lists (Goorvitch's CO, Partridge and Schwenke's water as Kurucz fixed it; wavelengths in nm, air above 200 nm, log gf and the lower level's energy in cm^-1), parsed once into ROOT/lines/<gas>_<band>.npz."""
    cache = os.path.join(giants_ROOT, "lines", gas + "_" + band + ".npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["lam"], d["loggf"], d["elow"]
    lo, hi = giants_BANDS[band]["lo"], giants_BANDS[band]["hi"]
    lam, loggf, elow = [], [], []
    with open(os.path.join(giants_ROOT, "lines", giants_KURUCZ_MOL[gas]), encoding="ascii", errors="replace") as fh:
        for line in fh:
            try:
                w = float(line[0:10]) * 10.0
            except ValueError:
                continue
            if lo <= w <= hi:
                try:
                    gf, e = float(line[10:17]), abs(float(line[22:32]))
                except ValueError:
                    continue
                lam.append(w)
                loggf.append(gf)
                elow.append(e)
    lam, loggf, elow = np.array(lam), np.array(loggf), np.array(elow)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, lam=lam, loggf=loggf, elow=elow)
    return lam, loggf, elow


giants_PRT = os.path.join(giants_ROOT, "opacities")
giants_PRT_NAMES = {"CO": "CO_main_iso", "H2O": "H2O_main_iso", "CH4": "CH4_main_iso", "CO2": "CO2_main_iso"}
giants_PRT_TEMPS = (81.14, 109.6, 148.0, 200.0, 270.0, 365.0, 493.0, 666.0, 900.0, 1215.0, 1641.0, 2217.0, 2995.0)


def giants_prt_array(path):
    """A petitRADTRANS line-by-line file as float64 values: a Fortran unformatted record's markers stripped where the size says there are some, else the raw array."""
    b = open(path, "rb").read()
    if (len(b) - 8) % 8 == 0 and len(b) % 8 != 0 or (len(b) >= 16 and int.from_bytes(b[:4], "little") == len(b) - 8):
        b = b[4:-4]
    return np.frombuffer(b, dtype="<f8")


def giants_template_prt(gas, t_k, band="nir", pressure="0.001000bar"):
    """A molecule's transmission template on a band's grid from petitRADTRANS's line-by-line cross-sections (wavelengths in cm, sigma in cm2 at the grid temperature nearest T_K and PRESSURE): the cross-section binned to the grid, each sample within three decades of the strongest weighted by its decade, the strongest one; air wavelengths."""
    from setitools import science
    folder = os.path.join(giants_PRT, giants_PRT_NAMES[gas])
    t_grid = min(giants_PRT_TEMPS, key=lambda t: abs(t - t_k))
    sig_file = next((f for f in os.listdir(folder) if f.startswith("sigma_") and "_%d.K_" % round(t_grid) in f and pressure in f), None)
    if sig_file is None:
        return np.zeros(len(giants_grid(band)))
    w = giants_prt_array(os.path.join(folder, "wlen.dat")) * 1e8
    s = giants_prt_array(os.path.join(folder, sig_file))
    n = min(len(w), len(s))
    w, s = science.vac_to_air(w[:n]), s[:n]
    g = giants_grid(band)
    lo, hi = giants_BANDS[band]["lo"], giants_BANDS[band]["hi"]
    keep = (w >= lo) & (w <= hi) & (s > 0)
    i = np.clip(np.round(np.log(w[keep] / g[0]) * 299792.458).astype(int), 0, len(g) - 1)
    binned = np.zeros(len(g))
    np.maximum.at(binned, i, s[keep])
    out = np.zeros(len(g))
    pos = binned > 0
    if not pos.any():
        return out
    out[pos] = np.clip(np.log10(binned[pos] / binned[pos].max()) + 3.0, 0.0, None) / 3.0
    return out


def giants_template(species, t_k, weight=None, band="optical"):
    """A species' transmission template on a band's grid at a temperature: an atom's lines from Kurucz's list where it is held (NIST's otherwise) weighted by gf exp(-E_low/kT), a molecule's from HITRAN's main isotopologue with its 296 K intensities carried to the temperature through the lower level's Boltzmann factor and the stimulated-emission term; each line a Gaussian of 3 km/s, the weights by decade of strength (log), their square root (sqrt) or capped at the 99.5th percentile (linear), the strongest one; air wavelengths."""
    from setitools import science
    g = giants_grid(band)
    lo, hi = giants_BANDS[band]["lo"], giants_BANDS[band]["hi"]
    out = np.zeros(len(g))
    # the lower level energy in cm^-1; Boltzmann's constant in eV/K, 1.2398e-4 eV per cm^-1 and the second radiation constant 1.4388 cm K
    k_ev = 8.617333e-5
    rows = []
    if species in giants_PRT_NAMES and os.path.isdir(os.path.join(giants_PRT, giants_PRT_NAMES[species])):
        return giants_template_prt(species, t_k, band)
    if species in giants_KURUCZ_MOL and os.path.exists(os.path.join(giants_ROOT, "lines", giants_KURUCZ_MOL[species])):
        lam, loggf, elow = giants_kurucz_mol(species, band)
        rows = list(zip(lam, 10.0 ** loggf * np.exp(-elow * 1.2398e-4 / (k_ev * t_k))))
    elif species in giants_HITRAN_ISO:
        nu, sw, elow = giants_hitran(species, lo, hi)
        s = sw * np.exp(-1.4388 * elow * (1.0 / t_k - 1.0 / 296.0)) * (1.0 - np.exp(-1.4388 * nu / t_k)) / (1.0 - np.exp(-1.4388 * nu / 296.0))
        rows = list(zip(science.vac_to_air(1e8 / nu), s))
    elif os.path.exists(giants_KURUCZ) and species in giants_CODES:
        for lam, loggf, elow in zip(*giants_kurucz(species, band)):
            rows.append((lam, 10.0 ** loggf * np.exp(-elow * 1.2398e-4 / (k_ev * t_k))))
    else:
        for ln in core.nist_lines(species, lo, hi):
            lam, aki, elow = core.num(ln.get("lam")), core.num(ln.get("aki")), core.num(ln.get("ei"))
            if lam and aki and elow is not None:
                rows.append((science.vac_to_air(np.array([lam]))[0], aki * lam ** 2 * np.exp(-elow * 1.2398e-4 / (k_ev * t_k))))
    if not rows:
        return out
    weight = weight or giants_WEIGHT
    if weight == "sqrt":
        rows = [(lam, np.sqrt(s)) for lam, s in rows]
    elif weight == "log":
        # saturated lines: every line within three decades of the strongest carries its decade's share, as the curve of growth flattens depth
        smax = max(s for _, s in rows)
        rows = [(lam, max(np.log10(s / smax) + 3.0, 0.0) / 3.0) for lam, s in rows]
    else:
        strengths = np.array([s for _, s in rows])
        keep = strengths >= 1e-4 * strengths.max()
        cap = np.percentile(strengths[keep], 99.5)
        rows = [(lam, min(s, cap)) for lam, s in rows]
    top = max(s for _, s in rows)
    g0 = g[0]
    for lam, s in rows:
        if s < 1e-3 * top:
            continue
        i = int(round(np.log(lam / g0) * 299792.458))
        a, b = max(i - 10, 0), min(i + 11, len(g))
        if b > a:
            v = (np.arange(a, b) - np.log(lam / g0) * 299792.458)
            out[a:b] += s / top * np.exp(-0.5 * (v / 3.0) ** 2)
    return out


def giants_residuals(spectra, vsys, whiten=True, band="optical"):
    """Each spectrum on the band's grid in the star's frame divided by its running median over 501 samples, then by the out-of-transit median spectrum, a cubic in wavelength taken out of each, the band's telluric regions and the samples beyond the spectrum NaN: (residuals, errors, each wavelength's out-of-transit scatter)."""
    from scipy.ndimage import median_filter
    g = giants_grid(band)
    F, E = [], []
    for w, f, e, meta, inside in spectra:
        wl = w / (1.0 + vsys / 299792.458)
        ok = np.isfinite(f) & (f > 0)
        fi = np.interp(g, wl[ok], f[ok], left=np.nan, right=np.nan)
        ei = np.interp(g, wl[ok], e[ok], left=np.nan, right=np.nan)
        cont = median_filter(np.nan_to_num(fi, nan=np.nanmedian(fi)), size=501, mode="nearest")
        for sky in giants_SKY:
            # the sky line's place on the grid in the star's frame, from its telescope wavelength through this spectrum's barycentric correction
            lam = sky * (1.0 + meta.get("berv", 0.0) / 299792.458) / (1.0 + vsys / 299792.458)
            i = int(round(np.log(lam / g[0]) * 299792.458))
            if 0 <= i < len(g):
                fi[max(i - 6, 0):i + 7] = np.nan
        with np.errstate(all="ignore"):
            F.append(fi / cont)
            E.append(ei / cont)
    F, E = np.array(F), np.array(E)
    ins = np.array([s[4] for s in spectra])
    master = np.nanmedian(F[~ins], axis=0)
    with np.errstate(all="ignore"):
        R, ER = F / master, E / master
    if band == "nir":
        # in the near infrared the median spectrum falls toward zero in the water bands, where the ratio would swamp each row's cubic, so the
        # telluric bands and the samples where the median lies at 0.2 or below are set aside before the fit
        R[:, ~(master > 0.2)] = np.nan
        for lo, hi in giants_BANDS[band]["telluric"]:
            R[:, (g > lo) & (g < hi)] = np.nan
    x = (np.arange(len(g)) - len(g) / 2.0) / len(g)
    for i in range(len(R)):
        good = np.isfinite(R[i])
        if good.sum() > 100:
            R[i] = R[i] / np.polyval(np.polyfit(x[good], R[i][good], 3), x)
    for lo, hi in giants_BANDS[band]["telluric"]:
        R[:, (g > lo) & (g < hi)] = np.nan
    R = R - 1.0
    # each wavelength's out-of-transit scatter, the weight of the inverse-variance cross-correlation, floored at a tenth of its median
    sd = np.nanstd(R[~ins], axis=0)
    floor = np.nanmedian(sd)
    sd = np.where(np.isfinite(sd) & (sd > 0), np.maximum(sd, 0.1 * floor), floor)
    return R, ER, (sd if whiten else np.ones_like(sd))


def giants_sysrem(R, sd, n=3):
    """SysRem (Tamuz et al. 2005) over the residual matrix: N rank-one systematics, each the product of a per-spectrum and a per-wavelength factor fitted by alternation weighted by each wavelength's scatter SD, taken out of every spectrum, NaN left as it is."""
    X = np.nan_to_num(R, nan=0.0)
    with np.errstate(all="ignore"):
        W = np.where(np.isfinite(R), 1.0 / np.maximum(np.broadcast_to(sd, R.shape), 1e-12) ** 2, 0.0)
    for _ in range(n):
        a = np.ones(X.shape[0])
        c = np.zeros(X.shape[1])
        for _ in range(6):
            with np.errstate(all="ignore"):
                c = (W * X * a[:, None]).sum(0) / np.maximum((W * a[:, None] ** 2).sum(0), 1e-30)
                a = (W * X * c[None, :]).sum(1) / np.maximum((W * c[None, :] ** 2).sum(1), 1e-30)
        X = X - a[:, None] * c[None, :]
    return np.where(np.isfinite(R), X, np.nan)


def giants_ccf(R, T, sd=None, vmax=300):
    """Inverse-variance cross-correlation of each residual row with a template over -VMAX..VMAX km/s (one sample a km/s on the grid), each sample weighted by its scatter SD, the sum normalised by the weighted template's norm so a row of unit noise gives unit scatter: absorption positive; on the GPU where torch has one, by FFT in numpy where torch is absent."""
    try:
        import torch
    except ImportError:
        return giants_ccf_fft(R, T, sd, vmax)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = R.shape[1]
    w = 1.0 / (sd if sd is not None else np.ones(n))
    # a non-finite sample, infinite as well as missing, counts as zero, since an infinite one would meet its zero weight as a NaN
    Rt = torch.as_tensor(np.where(np.isfinite(R), R * w[None, :], 0.0), dtype=torch.float32, device=dev)
    Wt = torch.as_tensor(np.where(np.isfinite(R).all(0), w, 0.0), dtype=torch.float32, device=dev)
    Tt = torch.as_tensor(T, dtype=torch.float32, device=dev)
    out = torch.zeros((R.shape[0], 2 * vmax + 1), device=dev)
    for j, v in enumerate(range(-vmax, vmax + 1)):
        if v >= 0:
            tw = Tt[:n - v] * Wt[v:]
            out[:, j] = -(Rt[:, v:] * tw).sum(1) / torch.sqrt((tw ** 2).sum().clamp(min=1e-30))
        else:
            tw = Tt[-v:] * Wt[:n + v]
            out[:, j] = -(Rt[:, :n + v] * tw).sum(1) / torch.sqrt((tw ** 2).sum().clamp(min=1e-30))
    return out.cpu().numpy()


def giants_ccf_fft(R, T, sd=None, vmax=300):
    """The same cross-correlation by FFT in numpy: each row, whitened and weighted, correlated with the template at every lag, and each lag normalised by the norm of the template under the weights it meets there, so that the result matches the torch form sample for sample."""
    n = R.shape[1]
    w = 1.0 / (sd if sd is not None else np.ones(n))
    Wt = np.where(np.isfinite(R).all(0), w, 0.0)
    X = np.where(np.isfinite(R), R * w[None, :], 0.0) * Wt[None, :]
    Tt = np.asarray(T, float)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    ft = np.conj(np.fft.rfft(Tt, m))
    den = np.fft.irfft(np.fft.rfft(Wt ** 2, m) * np.conj(np.fft.rfft(Tt ** 2, m)), m)
    # a correlation at lag k holds sum_j X[j + k] T[j]: lag k >= 0 at index k, lag k < 0 at index m + k
    pick = np.concatenate([np.arange(m - vmax, m), np.arange(0, vmax + 1)])
    norm = np.sqrt(np.maximum(den[pick], 1e-30))
    out = np.zeros((R.shape[0], 2 * vmax + 1))
    for i in range(R.shape[0]):
        out[i] = -np.fft.irfft(np.fft.rfft(X[i], m) * ft, m)[pick] / norm
    return out


def giants_shadow(ccf, vp_true, vsini, vmax=300):
    """The star's Doppler shadow taken out of each in-transit cross-correlation: a Gaussian fitted within the star's projected rotation (centre within +-VSINI, width from 3 km/s to half of it, 15 at least, amplitude within three times the window's largest value) with 12 km/s either side of the planet's own velocity left out of the fit, and subtracted; rows the fit cannot reach are left as they are."""
    from scipy.optimize import curve_fit
    v = np.arange(-vmax, vmax + 1, dtype=float)
    out = ccf.copy()
    wmax = max(15.0, vsini / 2.0)
    for k in range(ccf.shape[0]):
        fit = (np.abs(v) <= vsini + 20) & (np.abs(v - vp_true[k]) > 12)
        if fit.sum() < 10:
            continue
        y = ccf[k, fit]
        # the amplitude is held within three times the largest value in the fit window, so a window emptied by the exclusion cannot
        # return a Gaussian larger than anything the row holds
        amax = 3.0 * max(float(np.abs(y).max()), 1e-6)
        try:
            popt, _ = curve_fit(lambda x, a, c, s: a * np.exp(-0.5 * ((x - c) / s) ** 2), v[fit], y, p0=(min(max(y.max(), -amax), amax), v[fit][np.argmax(y)], min(6.0, wmax)),
                                bounds=([-amax, -vsini - 5, 3.0], [amax, vsini + 5, wmax]), maxfev=2000)
            out[k] -= popt[0] * np.exp(-0.5 * ((v - popt[1]) / popt[2]) ** 2)
        except Exception:
            pass
    return out


def giants_found(res, sigma):
    """Whether a species' result is a detection: at the threshold at the orbit's K_p, a sigma above anything the sham map reaches, with the map's own peak above 30 km/s in K_p and within 10 km/s of rest, and either inside the orbit's box or no more than a quarter above the orbit's value, as a short transit leaves K_p unresolved along a plateau; a shadow at the star's velocity leaking along the K_p axis passes neither, and a species whose template rests on one line, more than half of its weight within 10 km/s of one place, is left to the line tests, since a deep stellar line's own transit distortion sums along every trail."""
    peak = res["map_peak"]
    on_orbit = abs(peak["kp"] - res["kp_true"]) <= min(res.get("kp_half", 20.0), 60.0) or peak["snr"] <= 1.25 * res["snr"]
    many = res.get("top_line_share", 0.0) <= 0.5
    return bool(res["snr"] >= sigma and (res["sham"] is None or res["snr"] >= res["sham"] + 1.0) and peak["kp"] >= 30.0 and abs(peak["vrest"]) <= 10 and on_orbit and many)


def giants_top_line_share(T, sd=None):
    """The share of a template's weight, the sum of T^2, that lies within 10 km/s of its heaviest sample."""
    w2 = np.asarray(T, float) ** 2
    tot = float(w2.sum())
    if tot <= 0:
        return 0.0
    i = int(np.argmax(w2))
    return float(w2[max(i - 10, 0):i + 11].sum() / tot)


def giants_planet_frame(ccf, vp, w, kp_grid, vrest):
    """The in-transit cross-correlations summed in the planet's frame for each trial semi-amplitude: a map over K_p and rest velocity, each row's phases sin(2 pi phi) given by vp / kp_true as VP."""
    vmax = (ccf.shape[1] - 1) // 2
    out = np.zeros((len(kp_grid), len(vrest)))
    for i, kp in enumerate(kp_grid):
        for k in range(ccf.shape[0]):
            shift = kp * vp[k]
            v = vrest + shift
            out[i] += w[k] * np.interp(v, np.arange(-vmax, vmax + 1), ccf[k], left=0.0, right=0.0)
    return out / max(w.sum(), 1e-9)


def giants_search(planets=[], species=list(giants_SPECIES), sigma=4.0, temp=0.0, sysrem=0, weight="log", inject=0.0, whiten=True, shadow=True, redo=False, instruments=()):
    """Each planned transiting giant with four or more in-transit and out-of-transit spectra from HARPS or ESPRESSO on disk: residuals in the star's frame, each species' template cross-correlated with every spectrum, the in-transit sums over a K_p and rest-velocity map, and the peak at the orbit's K_p within 10 km/s of rest read against the map's scatter beyond 40 km/s; the strongest species per planet printed."""
    from setitools import biosignatures as bio
    plan = json.load(open(os.path.join(giants_ROOT, "plan.json")))
    kp_grid = np.arange(0.0, 301.0, 5.0)
    vrest = np.arange(-100.0, 101.0, 1.0)
    for p in plan:
        if planets and p["pl_name"] not in planets:
            continue
        for inst in ("ESPRESSO", "HARPS", "SPIRou", "NIRPS"):
            if instruments and inst not in instruments:
                continue
            band = giants_band(inst)
            # atoms in the optical, molecules in the near-infrared: the named species are kept to their band, the default list giving both
            pool = list(species) + (list(giants_MOLECULES) if species == list(giants_SPECIES) else [])
            wanted = [s for s in pool if (s in giants_HITRAN_ISO) == (band == "nir")]
            rows = p["instruments"].get(inst) or []
            rpath = os.path.join(giants_ROOT, "results", "%s.%s.json" % (re.sub(r"[^A-Za-z0-9+_.-]", "_", p["pl_name"]), inst))
            if not wanted or not rows or (os.path.exists(rpath) and not redo):
                continue
            per, tc, dur, k = p["pl_orbper"], p["pl_tranmid"] - 2400000.5, (p["pl_trandur"] or 3.0) / 24.0, p["pl_ratror"] or 0.1
            spectra = []
            for r in rows:
                path = os.path.join(giants_ROOT, "spectra", inst, re.sub(r"[^A-Za-z0-9_.-]", "_", r["id"]) + ".fits")
                if not os.path.exists(path):
                    continue
                try:
                    w, f, e, meta = bio.shoreline_load(path)
                except Exception:
                    continue
                wt, epoch = bio.shoreline_weight(r["t0"], r["t1"], tc, per, dur, k)
                spectra.append((w, f, e, dict(meta, weight=wt, epoch=epoch), wt > 0))
            n_in = sum(1 for s in spectra if s[4])
            if n_in < 4 or len(spectra) - n_in < 4:
                continue
            rvs = [s[3]["rv"] for s in spectra if s[3]["rv"] is not None]
            vsys = float(np.median(rvs)) if rvs else (p.get("st_radv") or 0.0)
            if inject:
                # the first species' lines laid into each in-transit spectrum's flux at depth INJECT along the planet's velocity, before anything else
                Tinj = giants_template(wanted[0], temp or p.get("pl_eqt") or 1500.0, weight, band)
                g = giants_grid(band)
                kp_inj = 2 * np.pi * (p["pl_orbsmax"] or 0.05) * 1.495978707e8 / (per * 86400.0)
                for i, (w, f, e, meta, inside) in enumerate(spectra):
                    if inside:
                        vp_i = kp_inj * np.sin(2 * np.pi * (meta["mjd"] - meta["epoch"]) / per)
                        tw = np.interp(w, g * (1.0 + (vsys + vp_i) / 299792.458), Tinj, left=0.0, right=0.0)
                        spectra[i] = (w, f * (1.0 - inject * meta["weight"] * tw), e, meta, inside)
            R, ER, sd = giants_residuals(spectra, vsys, whiten, band)
            if sysrem:
                R = giants_sysrem(R, sd, sysrem)
            ins = np.array([s[4] for s in spectra])
            phase = np.array([(s[3]["mjd"] - s[3]["epoch"]) / per for s in spectra])
            kp_true = 2 * np.pi * (p["pl_orbsmax"] or 0.05) * 1.495978707e8 / (per * 86400.0)
            vp_unit = np.sin(2 * np.pi * phase)
            wts = np.array([s[3]["weight"] for s in spectra])
            best, results, maps, sham_maps = None, [], [], []
            for sp in wanted:
                T = giants_template(sp, temp or p.get("pl_eqt") or 1500.0, weight, band)
                if not T.any():
                    continue
                ccf = giants_ccf(R, T, sd)
                # the star's own residual pattern, the median cross-correlation outside transit, taken out of every row
                ccf = ccf - np.median(ccf[~ins], axis=0)[None, :]
                # the sham: out-of-transit rows given the in-transit rows' phases and weights, and every step the in-transit rows get
                n_sh = min(int(ins.sum()), int((~ins).sum()))
                sham_rows = ccf[~ins][:n_sh].copy() if n_sh >= 4 else None
                # the star's shadow is an atomic-line feature of the optical band; a molecular template of a star without those lines has none
                if shadow and band == "optical":
                    ccf[ins] = giants_shadow(ccf[ins], kp_true * vp_unit[ins], p.get("st_vsin") or 15.0)
                    if sham_rows is not None:
                        sham_rows = giants_shadow(sham_rows, kp_true * vp_unit[ins][:n_sh], p.get("st_vsin") or 15.0)
                m = giants_planet_frame(ccf[ins], vp_unit[ins], wts[ins], kp_grid, vrest)
                m_out = giants_planet_frame(sham_rows, vp_unit[ins][:n_sh], wts[ins][:n_sh], kp_grid, vrest) if sham_rows is not None else None
                far = np.abs(vrest) > 40
                # the noise: the map's scatter away from rest, or the sham map's over its whole extent where that is larger
                noise = float(np.nanstd(m[:, far]))
                if m_out is not None:
                    noise = max(noise, float(np.nanstd(m_out)))
                # the orbit's box on the map: 10 km/s in rest velocity, and in K_p what moves the trail's ends by that much, 20 km/s at least
                # and 60 at most, never reaching below 30 km/s where the star's own pattern sits
                kp_half = min(max(20.0, 10.0 / max(float(np.abs(vp_unit[ins]).max()), 1e-3)), 60.0)
                near = (np.abs(kp_grid - kp_true) <= kp_half)[:, None] & (kp_grid >= 30.0)[:, None] & (np.abs(vrest) <= 10)[None, :]
                peak = float(np.nanmax(np.where(near, m, -np.inf)))
                i, j = np.unravel_index(int(np.nanargmax(m)), m.shape)
                res = dict(species=sp, snr=peak / noise if noise > 0 else 0.0, kp_true=kp_true, kp_half=kp_half, map_peak=dict(kp=float(kp_grid[i]), vrest=float(vrest[j]), snr=float(m[i, j] / noise) if noise > 0 else 0.0),
                           sham=float(np.nanmax(np.where(near, m_out, -np.inf)) / noise) if m_out is not None and noise > 0 else None,
                           top_line_share=giants_top_line_share(T, sd))
                # a detection stands at the threshold, a sigma above anything the sham map reaches, and as the map's own peak, so a shadow
                # at the star's velocity leaking along the K_p axis does not pass
                res["found"] = giants_found(res, sigma)
                results.append(res)
                maps.append(m)
                sham_maps.append(m_out)
                if best is None or (res["found"], res["snr"]) > (best["found"], best["snr"]):
                    best = res
                print("  %-12s %-9s %-5s S/N %5.1f at K_p %.0f (map peak %5.1f at K_p %.0f, v %+.0f; sham %s; top line %.2f)" % (
                    p["pl_name"], inst, sp, res["snr"], kp_true, res["map_peak"]["snr"], res["map_peak"]["kp"], res["map_peak"]["vrest"],
                    "%.1f" % res["sham"] if res["sham"] is not None else "-", res["top_line_share"]), flush=True)
            if best:
                det = [r["species"] for r in results if r["found"]]
                print("%-12s %-9s %d in, %d out: %s" % (p["pl_name"], inst, n_in, len(spectra) - n_in,
                                                       "DETECTED " + ", ".join("%s %.1f sigma" % (r["species"], r["snr"]) for r in results if r["found"]) if det
                                                       else "strongest %s at %.1f sigma, nothing detected" % (best["species"], best["snr"])), flush=True)
            if results:
                os.makedirs(os.path.join(giants_ROOT, "results"), exist_ok=True)
                # every species' map and sham map kept beside the result, for the figures
                np.savez_compressed(os.path.join(giants_ROOT, "results", "%s.%s.maps.npz" % (re.sub(r"[^A-Za-z0-9+_.-]", "_", p["pl_name"]), inst)),
                                    kp=kp_grid, vrest=vrest, species=np.array([r["species"] for r in results]), maps=np.array(maps), shams=np.array([x if x is not None else np.full_like(maps[0], np.nan) for x in sham_maps]))
                # a host rotating faster than 40 km/s or hotter than 7500 K leaves a shadow the Gaussian model does not reach: its map is kept but not read as a null
                caveat = ("fast rotator" if (p.get("st_vsin") or 0) > 40 else "hot host" if (p.get("st_teff") or 0) > 7500 else None)
                json.dump(dict(planet=p["pl_name"], inst=inst, band=band, n_in=n_in, n_out=len(spectra) - n_in, vsys=vsys, kp=kp_true, sigma=sigma, weight=weight,
                               sysrem=sysrem, temp=temp or p.get("pl_eqt") or 1500.0, caveat=caveat, species=results),
                          open(os.path.join(giants_ROOT, "results", "%s.%s.json" % (re.sub(r"[^A-Za-z0-9+_.-]", "_", p["pl_name"]), inst)), "w"), indent=1)


def giants_fetch(streams=4, planets=(), night_per_in=2.0, instruments=(), min_in=4):
    """The in-transit and eclipse spectra of every planned giant (or the named ones, or the named instruments' sets of MIN_IN spectra or more) with the same nights' other spectra nearest them up to NIGHT_PER_IN times their number, from ESO and the CADC, on STREAMS connections."""
    from setitools import biosignatures as bio
    plan = json.load(open(os.path.join(giants_ROOT, "plan.json")))
    want = []
    for p in plan:
        if planets and p["pl_name"] not in planets:
            continue
        for inst, rows in p["instruments"].items():
            if instruments and inst not in instruments:
                continue
            inside = [r for r in rows if r["src"] in ("ESO", "CADC") and r["place"] in ("transit", "eclipse")]
            if len(inside) < min_in:
                continue
            want += inside
            per, tc = p["pl_orbper"], p["pl_tranmid"] - 2400000.5
            for epoch in sorted({round((0.5 * (r["t0"] + r["t1"]) - tc) / per) for r in inside}):
                t_mid = tc + epoch * per
                n = sum(1 for r in inside if round((0.5 * (r["t0"] + r["t1"]) - tc) / per) == epoch)
                night = sorted((r for r in rows if r["src"] in ("ESO", "CADC") and r["place"] not in ("transit", "eclipse") and abs(0.5 * (r["t0"] + r["t1"]) - t_mid) < 0.5),
                               key=lambda r: abs(0.5 * (r["t0"] + r["t1"]) - t_mid))
                want += night[:max(int(night_per_in * n), 4)]
    seen, todo = set(), []
    for r in want:
        path = os.path.join(giants_ROOT, "spectra", r["inst"], re.sub(r"[^A-Za-z0-9_.-]", "_", r["id"]) + ".fits")
        if r["id"] not in seen and not os.path.exists(path):
            seen.add(r["id"])
            todo.append((r, path))

    def one(rp):
        r, path = rp
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if r["src"] == "ESO":
                return core.eso_download(r["id"], path)
            b = core.get(bio.shoreline_CADC_FILE + r["id"] + ".fits", timeout=600, soft=True)
            if b and b[:6] == b"SIMPLE":
                open(path + ".tmp", "wb").write(b)
                os.replace(path + ".tmp", path)
                return True
        except Exception:
            pass
        return False
    got = sum(1 for ok in core.pmap(one, todo, streams) if ok)
    print("%d spectra wanted, %d already here, %d fetched, %d failed" % (len(want), len(want) - len(todo), got, len(todo) - got))


def giants_record(sigma=4.0):
    """The searched planets' results in ROOT/results read into the coverage store: one row per planet and instrument, its target the planet's followed by # and the instrument, with its strongest species' significance at the orbit, the species detected, and a caveat where the host's rotation or heat left the star's shadow unmodelled or telluric residuals beyond the model; the reflected-light limits beside them, their targets followed by #reflected."""
    import glob
    rows, found, transiting = [], [], set()
    for path in sorted(glob.glob(os.path.join(giants_ROOT, "results", "*.json"))):
        r = json.load(open(path))
        if ".reflected." in path or ".emission." in path or "species" not in r:
            continue
        det = [s["species"] for s in r["species"] if giants_found(s, sigma)]
        best = max(r["species"], key=lambda s: s["snr"]) if r["species"] else None
        if best is None:
            continue
        found += [(r["planet"], r["inst"], d) for d in det]
        # a host rotating fast or hot leaves the star's shadow beyond the model; a sham map reaching ten sigma below zero marks telluric residuals beyond it
        caveat = (r["caveat"] + ", the star's shadow beyond the model") if r.get("caveat") else (
            "telluric residuals beyond the model" if any((s["sham"] or 0.0) < -10.0 for s in r["species"]) else None)
        transiting.add(r["planet"])
        rows.append(dict(target="planet:%s#%s" % (r["planet"], r["inst"]), stat=round(best["snr"], 1),
                         unit="sigma of the strongest species' cross-correlation at the orbit's K_p (%s, %d in-transit spectra)" % (r["inst"], r["n_in"]),
                         vet="not-measured" if caveat else None,
                         note="%s; %s; detected %s; species %s%s" % (r["inst"], r["band"], ", ".join(det) or "none",
                                                                     ", ".join("%s %.1f (sham %s)" % (s["species"], s["snr"], "%.1f" % s["sham"] if s["sham"] is not None else "-") for s in r["species"]),
                                                                     "; " + caveat if caveat else "")))
    # the reflected-light calibrations beside them, as limits on the planet-to-star contrast
    sets, reflected = len(rows), set()
    for path in sorted(glob.glob(os.path.join(giants_ROOT, "results", "*.reflected.json"))):
        r = json.load(open(path))
        if r.get("limit_3sigma") is None:
            continue
        reflected.add(r["planet"])
        rows.append(dict(target="planet:%s#reflected" % r["planet"], stat=round(r["snr"], 1), upper_limit=round(1e6 * r["limit_3sigma"], 1),
                         unit="ppm planet-to-star contrast at 3 sigma, reflected starlight (%d spectra, %.0f effective)" % (r["n"], r["n_eff"]),
                         vet="not-measured" if (r.get("retained") or 0) < 0.5 else None,
                         note="reflected light: %.1f sigma at the orbit (sham %.1f), map peak %.1f at K_p %.0f, v %+.0f; probe returned %.2f; contrast %+.0f ppm" % (
                             r["snr"], r["sham"], r["map_peak"]["snr"], r["map_peak"]["kp"], r["map_peak"]["vrest"], r.get("retained") or 0, 1e6 * (r.get("contrast") or 0))))
    core.record_run(
        "giants_planet_frame", "Planet-frame compositions of the giant planets",
        "ESO Phase 3 (HARPS, ESPRESSO, NIRPS, CRIRES, UVES), CADC (CFHT SPIRou), KOA (Keck NIRSPEC); Kurucz's gfall line list; HITRAN; NASA Exoplanet Archive pscomppars; SIMBAD",
        "every archived transit of the giant planets brighter than V 11 with four or more in-transit and out-of-transit spectra from one spectrograph",
        os.path.abspath(__file__), "giants", len(transiting | reflected), "planets", dict(sigma=sigma, weight=giants_WEIGHT),
        "the K_p and rest-velocity map's scatter away from rest and the sham map of out-of-transit rows given the transit's phases",
        "a species stands at %g sigma at the orbit's K_p, a sigma above the sham map, as the map's own peak" % sigma,
        "%d planet-instrument transit sets of %d planets and reflected light from %d; %d species detected in %d planet%s" % (
            sets, len(transiting), len(reflected), len(found), len({f[0] for f in found}), "" if len({f[0] for f in found}) == 1 else "s"), rows)
    print("%d planet-instrument transit sets of %d planets and %d reflected-light limits recorded; detections: %s" % (
        sets, len(transiting), len(reflected), ", ".join("%s %s %s" % f for f in found) or "none"))


companions_DOC = r"""Companions read out of their stars' spectra with the orbit left free: every archived spectrum of a star put on the grid in its own stellar frame, so the primary's lines cancel whatever its reflex motion, divided by their median and whitened; a cool companion's lines (an M dwarf's spectrum from the archive, or a species' lines) cross-correlated with every residual; and the trail summed along every orbit of a grid in period, phase and velocity amplitude. `search` and `test` read the summit against the summits of the same grid with the epochs shuffled; `survey` and `reanalyse` take the sky's pattern out of the rows, mask each row's telluric velocity, keep each amplitude to the periods a system of two solar masses allows and the sampling's own periods out, and read the summit against three nulls, the epochs shuffled, whole nights shuffled and nights shuffled within their season, the lowest significance kept.

    python -I seti.py composition companions plan                                              the archive's best-observed stars: ROOT/plan.json
    python -I seti.py composition companions test [--depth 0.001] [--period 7.3] [--k 40]     an injection into GJ 486's SPIRou spectra, recovered
    python -I seti.py composition companions search DIR [--template DIR2] [--inst SPIRou]       one star's spectra in DIR against a template star's in DIR2
    python -I seti.py composition companions survey [--inst HARPS] [--per-star 300] [--max-stars N] [--streams 4] [--keep] [--stars NAME ...] [--stars-file PATH] [--template DIR|FILE.npy] [--ccf-only]
    python -I seti.py composition companions reanalyse [--inst HARPS] [--components 5] [--part k/n]   the kept rows searched again"""


def companions_spectra(paths):
    """Spectra loaded with their own stellar velocities: (w, f, e, meta) per file, the velocity the header's where it has one."""
    from setitools import biosignatures as bio
    out = []
    for path in paths:
        try:
            w, f, e, meta = bio.shoreline_load(path)
        except Exception:
            continue
        out.append((w, f, e, meta))
    return out


def companions_vsys(spectra, fallback=None):
    """A series' systemic velocity (km/s): the median of the pipeline velocities where the products carry them, else FALLBACK where given, else SIMBAD's for the star nearest the spectra's position within 10 arcsec, else the median of the observing blocks' target velocities, else zero."""
    rvs = [s[3]["rv"] for s in spectra if s[3].get("rv") is not None]
    if rvs:
        return float(np.median(rvs))
    if fallback is not None:
        return float(fallback)
    pos = [(s[3]["ra"], s[3]["dec"]) for s in spectra if s[3].get("ra") is not None and s[3].get("dec") is not None]
    if pos:
        ra, dec = float(np.median([p[0] for p in pos])), float(np.median([p[1] for p in pos]))
        try:
            rows = core.simbad("SELECT b.rvz_radvel, DISTANCE(POINT('ICRS', b.ra, b.dec), POINT('ICRS', %f, %f)) AS d FROM basic AS b WHERE "
                               "CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', %f, %f, %f)) = 1 AND b.rvz_radvel IS NOT NULL ORDER BY d" % (ra, dec, ra, dec, 10.0 / 3600.0))
            if rows:
                return float(rows[0]["rvz_radvel"])
        except Exception:
            pass
    obs = [s[3]["rv_ob"] for s in spectra if s[3].get("rv_ob") is not None]
    return float(np.median(obs)) if obs else 0.0


def companions_align(F, master, max_kms=20):
    """Each row's velocity against the median spectrum (km/s): the whole-sample shift on the one km/s grid within MAX_KMS at which the row's lines match the median's best by least squares over the samples the median absorbs by 5 per cent or more, refined by a parabola through the best three."""
    use = np.isfinite(master) & (master < 0.95)
    M = np.where(use, master - 1.0, 0.0)
    X = np.nan_to_num(F - 1.0, nan=0.0)
    shifts = np.arange(-max_kms, max_kms + 1)
    out = np.zeros(F.shape[0])
    for i in range(F.shape[0]):
        cost = np.array([float(((np.roll(X[i], -s) - M)[use] ** 2).sum()) for s in shifts])
        k = int(np.argmin(cost))
        if 0 < k < len(shifts) - 1:
            a, b, c = cost[k - 1], cost[k], cost[k + 1]
            den = a - 2 * b + c
            out[i] = shifts[k] + (0.5 * (a - c) / den if den > 0 else 0.0)
        else:
            out[i] = shifts[k]
    return out


def companions_residuals(spectra, vsys, band, align=True, info=None):
    """Residuals on the band's grid: each spectrum in the primary's frame (its header velocity, else VSYS, then its own shift against the median spectrum when ALIGN) over its running median, divided by the median spectrum, a cubic taken out, telluric regions NaN; with each wavelength's scatter."""
    from scipy.ndimage import median_filter
    g = giants_grid(band)

    def onto(spectra, shifts):
        F = []
        for (w, f, e, meta), dv in zip(spectra, shifts):
            v = (meta["rv"] if meta.get("rv") is not None else vsys) + dv
            wl = w / (1.0 + v / 299792.458)
            ok = np.isfinite(f) & (f > 0)
            fi = np.interp(g, wl[ok], f[ok], left=np.nan, right=np.nan)
            cont = median_filter(np.nan_to_num(fi, nan=np.nanmedian(fi)), size=501, mode="nearest")
            with np.errstate(all="ignore"):
                F.append(fi / cont)
        return np.array(F)
    F = onto(spectra, np.zeros(len(spectra)))
    master = np.nanmedian(F, axis=0)
    if align:
        shifts = companions_align(F, master)
        moved = np.abs(shifts) >= 0.5
        if moved.any():
            F[moved] = onto([s for s, m in zip(spectra, moved) if m], shifts[moved])
            master = np.nanmedian(F, axis=0)
        if info is not None:
            info["vstar"] = np.array([(meta["rv"] if meta.get("rv") is not None else vsys) + (dv if mv else 0.0) for (_, _, _, meta), dv, mv in zip(spectra, shifts, moved)])
    with np.errstate(all="ignore"):
        R = F / master
    if band == "nir":
        # in the near infrared the median spectrum falls toward zero in the water bands, where the ratio would swamp each row's cubic, so the
        # telluric bands and the samples where the median lies at 0.2 or below are set aside before the fit
        R[:, ~(master > 0.2)] = np.nan
        for lo, hi in giants_BANDS[band]["telluric"]:
            R[:, (g > lo) & (g < hi)] = np.nan
    x = (np.arange(len(g)) - len(g) / 2.0) / len(g)
    for i in range(len(R)):
        good = np.isfinite(R[i])
        if good.sum() > 100:
            R[i] = R[i] / np.polyval(np.polyfit(x[good], R[i][good], 3), x)
    for lo, hi in giants_BANDS[band]["telluric"]:
        R[:, (g > lo) & (g < hi)] = np.nan
    R = R - 1.0
    sd = np.nanstd(R, axis=0)
    floor = np.nanmedian(sd)
    sd = np.where(np.isfinite(sd) & (sd > 0), np.maximum(sd, 0.1 * floor), floor)
    return R, sd, master


def companions_template(spectra, vsys, band):
    """A template star's absorption pattern on the band's grid: one minus its median normalised spectrum in its own frame, negative values zero."""
    _, _, master = companions_residuals(spectra, vsys, band)
    return np.where(np.isfinite(master), np.clip(1.0 - master, 0.0, None), 0.0)


def companions_unpattern(ccf, half=40):
    """The star's own pattern taken out of every cross-correlation row at its own strength: the rows' median shape, scaled to each row by least squares within HALF km/s of the star's velocity and subtracted over the whole row, then the rows' remaining median."""
    vmax = (ccf.shape[1] - 1) // 2
    pattern = np.median(ccf, axis=0)
    core = slice(vmax - half, vmax + half + 1)
    pc = pattern[core]
    scale = (ccf[:, core] * pc[None, :]).sum(1) / max(float((pc ** 2).sum()), 1e-30)
    out = ccf - scale[:, None] * pattern[None, :]
    return out - np.median(out, axis=0)[None, :]


def companions_trails(ccf, t, periods, phases=24, kgrid=np.arange(30.0, 155.0, 5.0), core_kms=25.0, tell=None, tell_kms=10.0, batch=256, m_max=None):
    """The trail sum over every orbit of the grid: for each period, phase and amplitude the mean of the cross-correlations read at K sin(2 pi (t/P + phase)), the readings within CORE_KMS of the star's own velocity left out, since the star's residual pattern lives there, and, given each row's telluric velocity TELL in the star's frame, those within TELL_KMS of it, since the sky's lines sweep through the star's frame once a year; with M_MAX (solar masses) each amplitude searched only up to the longest period a system of that mass allows it, 2 pi G M / K^3; BATCH periods at a time: the summit (period, phase, K, value) and the whole table over periods."""
    try:
        import torch
    except ImportError:
        raise ImportError("the trail search needs torch, which runs it on a GPU or the CPU")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n, width = ccf.shape
    vmax = (width - 1) // 2
    ph = np.arange(phases) / phases
    Cf = torch.as_tensor(np.ascontiguousarray(ccf), dtype=torch.float32, device=dev).reshape(-1)
    tt = torch.as_tensor(np.asarray(t, float), dtype=torch.float64, device=dev)
    pht = torch.as_tensor(ph, dtype=torch.float64, device=dev)
    kt = torch.as_tensor(np.asarray(kgrid, float), dtype=torch.float32, device=dev)
    base = (torch.arange(n, device=dev) * width)[None, :, None, None]
    tv = None if tell is None else torch.as_tensor(np.asarray(tell, float), dtype=torch.float32, device=dev)[None, :, None, None]
    P_all = np.asarray(periods, float)
    # the longest period at which each amplitude is possible: v^3 = 2 pi G M / P, with G M_sun = 1.32712e20 m^3 s^-2
    pmax = None if m_max is None else torch.as_tensor(2 * np.pi * 1.32712e20 * m_max / (np.asarray(kgrid, float) * 1e3) ** 3 / 86400.0, dtype=torch.float64, device=dev)
    table = np.full(len(P_all), -np.inf)
    where = np.zeros(len(P_all), dtype=np.int64)
    for s0 in range(0, len(P_all), batch):
        P = torch.as_tensor(P_all[s0:s0 + batch], dtype=torch.float64, device=dev)
        s = torch.sin(2 * np.pi * (tt[None, :, None] / P[:, None, None] + pht[None, None, :])).float()
        vel = kt[None, None, None, :] * s[..., None]
        # every row read at once by linear interpolation on the one km/s grid
        x = (vel + vmax).clamp(0, 2 * vmax - 1)
        j = torch.floor(x)
        frac = x - j
        idx = base + j.long()
        vals = Cf[idx] * (1.0 - frac) + Cf[idx + 1] * frac
        use = vel.abs() > core_kms
        if tv is not None:
            use = use & ((vel - tv).abs() > tell_kms)
        tot = (vals * use).sum(1) / use.sum(1).clamp(min=1)
        # a period the sampling keeps at one phase reads one velocity, not an orbit: the epochs' velocities must spread over the cycle
        tot = tot.masked_fill((s.std(1) < 0.45)[..., None], float("-inf"))
        if pmax is not None:
            tot = tot.masked_fill((P[:, None] > pmax[None, :])[:, None, :], float("-inf"))
        m, a = tot.reshape(tot.shape[0], -1).max(1)
        table[s0:s0 + len(P)] = m.cpu().numpy()
        where[s0:s0 + len(P)] = a.cpu().numpy()
    i = int(np.argmax(table))
    a, b = divmod(int(where[i]), len(kgrid))
    return (float(P_all[i]), float(ph[a]), float(kgrid[b]), float(table[i])), table


def companions_components(ccf, k):
    """The cross-correlation rows with their K leading principal components taken out, the shapes the star's own residual pattern takes from night to night, which a companion's trail, moving from row to row, does not share."""
    if not k:
        return ccf
    U, S, Vt = np.linalg.svd(ccf, full_matrices=False)
    return ccf - (U[:, :k] * S[:k]) @ Vt[:k]


def companions_periods(t):
    """The period grid for a series: uniform in frequency at a quarter of the span's resolution, as a periodogram is, from half the span (two cycles at least, since a longer period reads a slow drift of the star's pattern as an orbit) down to half a day."""
    span = max(float(t.max() - t.min()), 1.0)
    f = np.arange(2.0 / span, 2.0, 1.0 / (4 * span))
    return 1.0 / f[::-1]


def companions_search(spectra, template, band, periods=None, trials=5, vsys=0.0, components=5):
    """One star's spectra against a template: residuals, whitened cross-correlations with the static pattern removed and COMPONENTS principal components after it, the trail summit over the orbit grid, and the summits of TRIALS shufflings of the epochs as the null: (summit, null mean, null scatter, significance)."""
    R, sd, _ = companions_residuals(spectra, vsys, band)
    ccf = giants_ccf(R, template, sd)
    ccf = companions_components(companions_unpattern(ccf), components)
    t = np.array([s[3]["mjd"] for s in spectra])
    t = t - t.min()
    if periods is None:
        periods = companions_periods(t)
    best, table = companions_trails(ccf, t, periods)
    rng = np.random.default_rng(3)
    # the null on the same grid at its full density, so every alias the sampling allows is in it too
    null = np.array([companions_trails(ccf, rng.permutation(t), periods)[0][3] for _ in range(trials)])
    z = (best[3] - null.mean()) / max(null.std(), 1e-12)
    return best, float(null.mean()), float(null.std()), float(z), periods, table


def companions_test(depth=0.001, period=7.3, k=40.0):
    """An injection of GJ 486's own lines at DEPTH, moving on a PERIOD-day orbit of amplitude K, into its SPIRou spectra from the rocky-planet search, then the blind search with the star's own spectrum as the template, before and after the injection."""
    import glob
    from setitools import biosignatures as bio
    plan = json.load(open(os.path.join(bio.shoreline_ROOT, "plan.json")))
    p = next(p for p in plan if p["pl_name"] == "GJ 486 b")
    paths = [bio.shoreline_path(r) for r in p["instruments"]["SPIRou"]["inside"] + p["instruments"]["SPIRou"]["night"]]
    spectra = companions_spectra([x for x in paths if os.path.exists(x)])
    vsys = p.get("st_radv") or 19.1
    band = "nir"
    T = companions_template(spectra, vsys, band)
    print("%d spectra; template samples above 0.1: %d" % (len(spectra), int((T > 0.1).sum())), flush=True)
    best, nm, ns, z, periods, table = companions_search(spectra, T, band, vsys=vsys)
    print("blind, no injection: summit P %.2f d, K %.0f km/s, %.4f; null %.4f +- %.4f -> %.1f sigma" % (best[0], best[2], best[3], nm, ns, z), flush=True)
    g = giants_grid(band)
    t0 = min(s[3]["mjd"] for s in spectra)
    injected = []
    for w, f, e, meta in spectra:
        v = k * np.sin(2 * np.pi * (meta["mjd"] - t0) / period)
        vstar = meta["rv"] if meta.get("rv") is not None else vsys
        tw = np.interp(w, g * (1.0 + (vstar + v) / 299792.458), T, left=0.0, right=0.0)
        injected.append((w, f * (1.0 - depth * tw), e, meta))
    best, nm, ns, z, periods, table = companions_search(injected, T, band, vsys=vsys)
    print("injected depth %g at P %.2f d, K %.0f: summit P %.2f d, K %.0f km/s, %.4f; null %.4f +- %.4f -> %.1f sigma" % (depth, period, k, best[0], best[2], best[3], nm, ns, z), flush=True)


companions_ROOT = core.data("companions")


def companions_plan(min_spectra=100, min_span=365.0):
    """The archive's best-observed stars per spectrograph, those with MIN_SPECTRA spectra or more over MIN_SPAN days or more, into ROOT/plan.json: instrument, name, count and span."""
    out = []
    for inst in ("HARPS", "ESPRESSO", "NIRPS", "CRIRES", "UVES"):
        q = ("SELECT TOP 2000 target_name, COUNT(*) AS n, MIN(t_min) AS t0, MAX(t_max) AS t1, AVG(s_ra) AS ra, AVG(s_dec) AS dec FROM ivoa.ObsCore WHERE "
             "dataproduct_type = 'spectrum' AND instrument_name = '%s' GROUP BY target_name ORDER BY n DESC" % inst)
        for r in core.tap("eso", q, method="POST"):
            n, span = int(r["n"]), float(r["t1"]) - float(r["t0"])
            if n >= min_spectra and span >= min_span and r["target_name"].strip().upper() not in ("SUN", "MOON", "SKY"):
                out.append(dict(inst=inst, name=r["target_name"], n=n, span=round(span, 1), ra=core.num(r["ra"]), dec=core.num(r["dec"])))
    os.makedirs(companions_ROOT, exist_ok=True)
    json.dump(out, open(os.path.join(companions_ROOT, "plan.json"), "w"), indent=1)
    by = {}
    for o in out:
        by[o["inst"]] = by.get(o["inst"], 0) + 1
    print("%d star-instrument series: %s" % (len(out), ", ".join("%s %d" % kv for kv in sorted(by.items()))))


def companions_untellur(ccf, tell):
    """Each cross-correlation row less the sky's pattern: the rows moved into their telluric frames, their median there taken as the pattern, moved back to each row's frame, fitted to the row at its own strength by least squares and subtracted."""
    v = np.arange(ccf.shape[1], dtype=float) - (ccf.shape[1] - 1) // 2
    moved = np.array([np.interp(v, v - vt, row, left=np.nan, right=np.nan) for row, vt in zip(ccf, tell)])
    with np.errstate(all="ignore"):
        pattern = np.nanmedian(moved, axis=0)
    pattern = np.where(np.isfinite(pattern), pattern, 0.0)
    out = np.empty_like(ccf)
    for i, (row, vt) in enumerate(zip(ccf, tell)):
        back = np.interp(v, v + vt, pattern, left=0.0, right=0.0)
        out[i] = row - float((row * back).sum() / max((back * back).sum(), 1e-30)) * back
    return out


def companions_night_shuffle(t, rng):
    """The epochs with whole nights moved to other nights' dates, each night keeping its spectra's offsets from its start, so that what is tied to the hour of observing or to one night's conditions stays in the null."""
    night = np.floor(t + 0.3)
    bases, inv = np.unique(night, return_inverse=True)
    return bases[rng.permutation(len(bases))][inv] + (t - night)


def companions_season_shuffle(t, rng, width=20.0):
    """The epochs with whole nights moved only among nights within the same WIDTH days of the year, each night keeping its spectra's offsets from its start, so that what follows the season and the hour of observing stays in the null; a season with one night keeps it."""
    night = np.floor(t + 0.3)
    bases, inv = np.unique(night, return_inverse=True)
    season = np.floor((bases % 365.25) / width)
    moved = bases.copy()
    for s in np.unique(season):
        idx = np.where(season == s)[0]
        moved[idx] = bases[idx][rng.permutation(len(idx))]
    return moved[inv] + (t - night)


companions_KGRID = np.arange(40.0, 155.0, 5.0)
companions_CORE = 30.0
# the largest total mass of a system the search allows, in solar masses, which bounds each amplitude's period
companions_MMAX = 2.0
# the sampling's own periods: a summit near a day, its half and double, or near the year or its half is marked, not dropped
companions_ALIASES = ((0.5, 0.02), (0.99727, 0.02), (1.0, 0.02), (2.0, 0.02), (4.0, 0.02), (182.6, 0.08), (365.25, 0.08), (730.5, 0.08))


def companions_off_aliases(periods):
    """The periods away from the sampling's own: the summit is taken only there, since the day's and the year's rhythms outweigh a companion's trail at those periods and a companion on them cannot be told from the observing."""
    keep = np.ones(len(periods), dtype=bool)
    for p, w in companions_ALIASES:
        keep &= np.abs(periods / p - 1.0) >= w
    return periods[keep]


def companions_trail_result(ccf, t, components, trials=5, tell=None):
    """The trail summit of cleaned cross-correlation rows over the series' period grid, amplitudes from 40 km/s and the star's core to 30 km/s left out: the sky's pattern removed in its own frame and each row's telluric velocity TELL masked where given, then the summit against three nulls, the epochs shuffled, whole nights shuffled and nights shuffled within their season, the lowest significance kept; a summit at the sampling's own periods marked as an alias: the result fields for a star."""
    rows = companions_untellur(ccf, tell) if tell is not None else ccf
    c = companions_components(rows, components)
    tt = t - t.min()
    pmax = 2 * np.pi * 1.32712e20 * companions_MMAX / (companions_KGRID.min() * 1e3) ** 3 / 86400.0
    periods = companions_off_aliases(companions_periods(tt))
    periods = periods[periods <= pmax]
    run = lambda times: companions_trails(c, times, periods, kgrid=companions_KGRID, core_kms=companions_CORE, tell=tell, batch=512, m_max=companions_MMAX)
    best, table = run(tt)
    rng = np.random.default_rng(3)
    ep = np.array([run(rng.permutation(tt))[0][3] for _ in range(trials)])
    ni = np.array([run(companions_night_shuffle(tt, rng))[0][3] for _ in range(trials)])
    se = np.array([run(companions_season_shuffle(tt, rng))[0][3] for _ in range(trials)])
    zs = {name: (best[3] - x.mean()) / max(x.std(), 1e-12) for name, x in (("epochs", ep), ("nights", ni), ("seasons", se))}
    z_ep, z_ni, z_se = zs["epochs"], zs["nights"], zs["seasons"]
    null = {"epochs": ep, "nights": ni, "seasons": se}[min(zs, key=zs.get)]
    alias = next(("%g d" % p for p, w in companions_ALIASES if abs(best[0] / p - 1.0) < w), None)
    return dict(span=float(tt.max()), period=best[0], phase=best[1], k=best[2], value=best[3], null_mean=float(null.mean()), null_sd=float(null.std()),
                z=float(min(z_ep, z_ni, z_se)), z_epochs=float(z_ep), z_nights=float(z_ni), z_seasons=float(z_se), nights=int(len(np.unique(np.floor(tt + 0.3)))),
                alias=alias, components=components, telluric_mask=tell is not None, method=5)


def companions_tell(name, mjd, vsys, berv=None, vstar=None, instrument="HARPS"):
    """Each row's telluric velocity in the star's frame, the barycentric correction less the velocity the spectrum was put at: the correction from the kept rows where they carry it, else computed from the plan's position for the instrument's site; the velocity per row (VSTAR) where kept, else VSYS."""
    if berv is None or not np.all(np.isfinite(berv)):
        # names compared without spaces and underscores, which a star's tag turns its spaces into
        key = lambda s: re.sub(r"[\s_]", "", str(s)).upper()
        plan = {key(p["name"]): p for p in json.load(open(os.path.join(companions_ROOT, "plan.json"))) if p["inst"] == instrument}
        p = plan.get(key(name))
        if p is None or p.get("ra") is None:
            return None
        from setitools import science
        site = "Paranal" if instrument in ("ESPRESSO", "UVES", "CRIRES") else "LaSilla"
        berv = np.atleast_1d(science.barycentric_correction(p["ra"], p["dec"], np.asarray(mjd, float), site))
    return np.asarray(berv, float) - (np.asarray(vstar, float) if vstar is not None else float(vsys or 0.0))


def companions_reanalyse(components=5, instrument=None, part=None):
    """Every star whose cross-correlation rows are kept under ROOT/ccf searched again from those rows with COMPONENTS principal components taken out, the results under ROOT/results rewritten; PART (k, n) takes the stars whose tag hashes to k of n, so n searches can run side by side."""
    import glob
    import zlib
    os.makedirs(os.path.join(companions_ROOT, "results"), exist_ok=True)
    n = 0
    for path in sorted(glob.glob(os.path.join(companions_ROOT, "ccf", "*.npz"))):
        tag = os.path.basename(path)[:-4]
        if instrument and not tag.startswith(instrument + "_"):
            continue
        if part and zlib.crc32(tag.encode()) % part[1] != part[0]:
            continue
        rpath = os.path.join(companions_ROOT, "results", tag + ".json")
        old = json.load(open(rpath)) if os.path.exists(rpath) else (json.load(open(path[:-4] + ".json")) if os.path.exists(path[:-4] + ".json") else {})
        # a star whose note has not arrived is still being written or copied, and waits for the next pass
        if not old or (old.get("components") == components and old.get("method") == 5 and "z" in old):
            continue
        try:
            with np.load(path) as d:
                rows, mjd = d["ccf"].astype(np.float64), d["mjd"]
                berv = d["berv"] if "berv" in d.files else None
                vstar = d["vstar"] if "vstar" in d.files else None
        except Exception:
            continue
        tell = companions_tell(old.get("name") or tag[len(instrument or "HARPS") + 1:], mjd, old.get("vsys"), berv, vstar, instrument or "HARPS")
        res = companions_trail_result(rows, mjd, components, tell=tell)
        out = dict(old, n=int(rows.shape[0]), **res)
        json.dump(out, open(rpath, "w"), indent=1)
        print("%-20s %3d rows over %5.0f d on %d nights: summit P %8.2f d, K %3.0f km/s, %.2f; null %.2f +- %.2f -> %.1f sigma (epochs %.1f, nights %.1f, seasons %.1f)%s" % (
            out.get("name", tag), out["n"], out["span"], out["nights"], out["period"], out["k"], out["value"], out["null_mean"], out["null_sd"], out["z"],
            out["z_epochs"], out["z_nights"], out["z_seasons"], "; alias of %s" % out["alias"] if out["alias"] else ""), flush=True)
        n += 1
    print("%d stars searched again" % n)


def companions_survey(instrument="HARPS", per_star=300, max_stars=0, streams=4, template_dir=None, template_obj="GJ 367", template_vsys=47.9, keep=False, components=5, stars=(), ccf_only=False):
    """The plan's series of one instrument, most spectra first (or the named STARS in their order): up to PER_STAR spectra of each star spread over its span fetched from ESO, searched blind against the template with COMPONENTS principal components taken out of the cross-correlation rows, the summit, null and cross-correlation rows kept under ROOT/results and ROOT/ccf, and the spectra removed unless KEEP; a star with a result is passed over; with CCF_ONLY the rows are kept and the trail search left for later."""
    plan = [p for p in json.load(open(os.path.join(companions_ROOT, "plan.json"))) if p["inst"] == instrument]
    plan.sort(key=lambda p: -p["n"])
    if stars:
        order = {s.strip().upper().replace(" ", ""): i for i, s in enumerate(stars)}
        plan = sorted((p for p in plan if p["name"].strip().upper().replace(" ", "") in order), key=lambda p: order[p["name"].strip().upper().replace(" ", "")])
    band = giants_band(instrument)
    tpl_dir = template_dir or os.path.join(bio_shoreline_root(), "spectra", "HARPS")
    # a template kept as an array (.npy) is read as it is, which spares a node the template star's spectra
    T = np.load(tpl_dir) if str(tpl_dir).endswith(".npy") else companions_template(companions_spectra(companions_files(tpl_dir, template_obj)), template_vsys, band)
    os.makedirs(os.path.join(companions_ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(companions_ROOT, "ccf"), exist_ok=True)
    done = 0
    for p in plan:
        tag = re.sub(r"[^A-Za-z0-9+_.-]", "_", "%s_%s" % (instrument, p["name"]))
        rpath = os.path.join(companions_ROOT, "results", tag + ".json")
        if os.path.exists(rpath) or (ccf_only and os.path.exists(os.path.join(companions_ROOT, "ccf", tag + ".npz"))):
            continue
        # the stars another node has finished, from the list the coordinating node copies here
        finished = os.path.join(companions_ROOT, "ccf", "done.txt")
        if os.path.exists(finished) and tag in set(open(finished, encoding="utf-8").read().split()):
            continue
        rows = [r for r in core.archive_spectra(p["ra"], p["dec"]) if r["inst"] == instrument and (r.get("snr") or 0) > 30]
        rows.sort(key=lambda r: r["t0"])
        pick = rows[::max(len(rows) // per_star, 1)][:per_star]
        folder = os.path.join(companions_ROOT, "spectra", tag)
        os.makedirs(folder, exist_ok=True)

        def one(r):
            path = os.path.join(folder, re.sub(r"[^A-Za-z0-9_.-]", "_", r["id"]) + ".fits")
            if os.path.exists(path):
                return path
            try:
                return path if core.eso_download(r["id"], path) else None
            except Exception:
                # a product the archive refuses, proprietary or withdrawn, is passed over
                return None
        paths = [x for x in core.pmap(one, pick, streams) if x]
        spectra = companions_spectra(paths)
        if len(spectra) < 30:
            json.dump(dict(name=p["name"], inst=instrument, n=len(spectra), note="too few spectra"), open(rpath, "w"))
            continue
        vsys = companions_vsys(spectra)
        info = {}
        R, sd, _ = companions_residuals(spectra, vsys, band, info=info)
        ccf = companions_unpattern(giants_ccf(R, T, sd))
        t = np.array([s[3]["mjd"] for s in spectra])
        berv = np.array([s[3].get("berv") if s[3].get("berv") is not None else np.nan for s in spectra], float)
        vstar = info.get("vstar", np.full(len(spectra), vsys))
        np.savez_compressed(os.path.join(companions_ROOT, "ccf", tag + ".npz"), ccf=ccf.astype(np.float32), mjd=t, berv=berv, vstar=vstar)
        if ccf_only:
            json.dump(dict(name=p["name"], inst=instrument, n=len(spectra), vsys=vsys), open(os.path.join(companions_ROOT, "ccf", tag + ".json"), "w"))
            print("%-20s %3d spectra: rows kept" % (p["name"], len(spectra)), flush=True)
            if not keep:
                import shutil
                shutil.rmtree(folder, ignore_errors=True)
            done += 1
            if max_stars and done >= max_stars:
                break
            continue
        res = companions_trail_result(ccf, t, components, tell=berv - vstar if np.all(np.isfinite(berv)) else None)
        json.dump(dict(name=p["name"], inst=instrument, n=len(spectra), vsys=vsys, **res), open(rpath, "w"), indent=1)
        print("%-20s %3d spectra over %5.0f d: summit P %8.2f d, K %3.0f km/s, %.2f; null %.2f +- %.2f -> %.1f sigma" % (
            p["name"], len(spectra), res["span"], res["period"], res["k"], res["value"], res["null_mean"], res["null_sd"], res["z"]), flush=True)
        if not keep:
            import shutil
            shutil.rmtree(folder, ignore_errors=True)
        done += 1
        if max_stars and done >= max_stars:
            break
    print("%d stars searched" % done)


def bio_shoreline_root():
    """The rocky-planet search's data root, where the archive's M-dwarf spectra are kept."""
    from setitools import biosignatures as bio
    return bio.shoreline_ROOT


def companions_files(folder, obj):
    """The FITS files of a folder whose OBJECT header holds OBJ (spaces ignored, case ignored), all of them without OBJ."""
    import glob
    from astropy.io import fits
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.fits"))):
        if not obj or obj.replace(" ", "").upper() in str(fits.getheader(f).get("OBJECT", "")).replace(" ", "").upper():
            out.append(f)
    return out


def companions_cli():
    quiet_warnings()
    p = argparse.ArgumentParser(prog="seti.py composition companions")
    p.add_argument("mode", choices=("plan", "test", "search", "survey", "reanalyse"))
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--stars", nargs="*", default=[])
    p.add_argument("--stars-file")
    p.add_argument("--part", help="k/n: the re-search's share, by the stars' tags")
    p.add_argument("--ccf-only", action="store_true")
    p.add_argument("--per-star", type=int, default=300)
    p.add_argument("--max-stars", type=int, default=0)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--keep", action="store_true")
    p.add_argument("target", nargs="?")
    p.add_argument("--object", default="")
    p.add_argument("--vsys", type=float, default=None)
    p.add_argument("--template")
    p.add_argument("--template-object", default="")
    p.add_argument("--template-vsys", type=float, default=0.0)
    p.add_argument("--inst", default="SPIRou")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--inject", type=float, default=0.0)
    p.add_argument("--depth", type=float, default=0.001)
    p.add_argument("--period", type=float, default=7.3)
    p.add_argument("--k", type=float, default=40.0)
    a = p.parse_args()
    if a.stars_file:
        a.stars = [s.strip() for s in open(a.stars_file, encoding="utf-8") if s.strip()]
    if a.mode == "plan":
        companions_plan()
        return
    if a.mode == "survey":
        companions_survey(a.inst, a.per_star, a.max_stars, a.streams, a.template, a.template_object or "GJ 367", a.template_vsys or 47.9, a.keep, a.components, a.stars, a.ccf_only)
        return
    if a.mode == "reanalyse":
        companions_reanalyse(a.components, a.inst if a.inst != "SPIRou" else None, tuple(int(x) for x in a.part.split("/")) if a.part else None)
        return
    if a.mode == "test":
        companions_test(a.depth, a.period, a.k)
        return
    band = giants_band(a.inst)
    files = companions_files(a.target, a.object)
    spectra = companions_spectra(files[:a.max] if a.max else files)
    tpl = companions_spectra(companions_files(a.template, a.template_object)) if a.template else spectra
    T = companions_template(tpl, a.template_vsys, band)
    vsys = a.vsys if a.vsys is not None else companions_vsys(spectra)
    print("%d spectra of %s at %.1f km/s, template from %d spectra of %s: template samples above 0.1: %d" % (len(spectra), a.object or a.target, vsys, len(tpl), a.template_object or "itself", int((T > 0.1).sum())), flush=True)
    if a.inject:
        g = giants_grid(band)
        t0 = min(s[3]["mjd"] for s in spectra)
        injected = []
        for w, f, e, meta in spectra:
            v = a.k * np.sin(2 * np.pi * (meta["mjd"] - t0) / a.period)
            vstar = meta["rv"] if meta.get("rv") is not None else vsys
            tw = np.interp(w, g * (1.0 + (vstar + v) / 299792.458), T, left=0.0, right=0.0)
            injected.append((w, f * (1.0 - a.inject * tw), e, meta))
        spectra = injected
    best, nm, ns, z, periods, table = companions_search(spectra, T, band, vsys=vsys, components=a.components)
    print("%s: summit P %.2f d, phase %.2f, K %.0f km/s, %.4f; null %.4f +- %.4f -> %.1f sigma" % (
        "injected %g at P %.2f d, K %.0f" % (a.inject, a.period, a.k) if a.inject else "blind", best[0], best[1], best[2], best[3], nm, ns, z), flush=True)


# --------------------------------------------------------------------------------------------------------------

binaries_DOC = r"""Companions of single-lined binaries along their known orbits: each archived spectrum's velocity measured against a template of the primary and the primary's orbit fitted to those velocities; the spectra put on the grid in the primary's frame at their own velocities, divided by the series' median and cross-correlated by inverse variance with an M-dwarf template; the star's pattern, the sky's pattern and five principal components taken out of the rows; the rows summed along the track the orbit gives the companion over its semi-amplitude K2 and its systemic offset from the primary's, the summit read against the orbit's phase moved and whole nights shuffled, its false-alarm probability from ten thousand draws of each null and a Gumbel law fitted to them, for each binary and for the survey's binaries together, K2 bootstrapped over whole nights and over observing seasons with the scatter of the series' parts by date as its systematic uncertainty, and the masses from K1 and K2.

    python -I seti.py composition binaries plan [--min 20]                                     the ESO series of the Gaia DR3 and SB9 single-lined binaries: ROOT/plan.json
    python -I seti.py composition binaries template NAME [--inst HARPS] [--out FILE.npy]       a primary's line depths from its own spectra at its catalogue orbit's velocities
    python -I seti.py composition binaries template --spectra DIR --object NAME [NAME ...] --out FILE.npy   a star's line depths from its spectra in DIR in its own frame
    python -I seti.py composition binaries build NAME [NAME ...] [--inst HARPS] [--min-snr 10] [--per-star 300] [--keep] [--template FILE.npy] [--variant V] [--band LO HI]
                                                 [--inject RATIO --inject-k2 KMS [--inject-shift 0.25] [--inject-template FILE.npy]] [--orbit-frames]   each binary's rows: ROOT/rows or ROOT/rows_V
    python -I seti.py composition binaries search [NAME ...] [--inst HARPS] [--dv 1.0]          the summit over K2 at a fixed systemic offset: ROOT/results
    python -I seti.py composition binaries calibrate [NAME ...] [--inst HARPS] [--draws 10000] [--window 5]   the summits' false-alarm probabilities: ROOT/calibrate
    python -I seti.py composition binaries calibrate --aggregate [--inst HARPS]                the survey's binaries taken together: ROOT/calibrate/survey_<inst>.json
    python -I seti.py composition binaries measure NAME [--inst HARPS] [--incl DEG] [--k2-fixed KMS] [--dv-fixed KMS] [--variant V] [--components K] [--shift FRAC]
                                                 the peak over K2 and the offset, bootstrapped, with its systematic uncertainty: ROOT/measure
    python -I seti.py composition binaries record"""

binaries_ROOT = core.data("binaries")
# Gaia DR3 counts its times of periastron from the reference epoch J2016.0
binaries_J2016 = 2457389.0
binaries_CORE, binaries_TELL, binaries_COMPONENTS = 30.0, 10.0, 5


def binaries_radial(t, P, tp, e, w):
    """The orbit's radial-velocity shape at MJD T, cos(nu + omega) + e cos(omega), from Kepler's equation (omega in degrees)."""
    M = 2 * np.pi * (((np.asarray(t, float) - tp) / P) % 1.0)
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return np.cos(nu + np.radians(w)) + e * np.cos(np.radians(w))


def binaries_orbit(target):
    """A plan entry's catalogue orbit: (period, time of periastron in MJD, e, omega, K1, gamma)."""
    tp = target["t_peri"]
    if target["source"].startswith("Gaia"):
        tp = binaries_J2016 + tp
    elif tp < 1e5:
        tp = tp + 2400000.0
    return target["period"], tp - 2400000.5, target["e"] or 0.0, target["omega"] or 0.0, target["k1"], target["gamma"] or 0.0


def binaries_refit(t, v, P, tp, e, w, k1):
    """The orbit fitted to measured velocities V: a grid over period (within 0.4 per cent), phase, e and omega with K1 and gamma solved linearly at each point, then least squares with a soft loss; (P, tp, e, omega, K1, gamma, rms)."""
    from scipy.optimize import least_squares
    span = max(float(t.max() - t.min()), P)
    step = 0.02 * P / span * P
    ph = np.arange(0.0, 1.0, 0.01)
    best = (np.inf, P, tp, e, w)
    for eg in np.arange(0.0, 0.61, 0.05):
        for wg in np.arange(0.0, 360.0, 20.0):
            for Pg in np.arange(P * 0.996, P * 1.004 + step / 2, step):
                tpg = tp + ph[:, None] * Pg
                M = 2 * np.pi * (((t[None, :] - tpg) / Pg) % 1.0)
                E = M + eg * np.sin(M)
                for _ in range(12):
                    E = E - (E - eg * np.sin(E) - M) / (1.0 - eg * np.cos(E))
                nu = 2 * np.arctan2(np.sqrt(1 + eg) * np.sin(E / 2), np.sqrt(1 - eg) * np.cos(E / 2))
                s = np.cos(nu + np.radians(wg)) + eg * np.cos(np.radians(wg))
                sc = s - s.mean(1, keepdims=True)
                kk = (sc * (v - v.mean())[None, :]).sum(1) / np.maximum((sc * sc).sum(1), 1e-12)
                cost = np.where(kk > 0, np.log1p((v[None, :] - v.mean() - kk[:, None] * sc) ** 2).sum(1), np.inf)
                j = int(np.argmin(cost))
                if cost[j] < best[0]:
                    best = (float(cost[j]), float(Pg), float(tpg[j, 0]), float(eg), float(wg))
    P, tp, e, w = best[1], best[2], max(best[3], 0.01), best[4]
    se = np.sqrt(max(e, 1e-4))
    x0 = [P, tp, se * np.cos(np.radians(w)), se * np.sin(np.radians(w)), k1, float(np.median(v - k1 * binaries_radial(t, P, tp, e, w)))]
    unpack = lambda x: (x[0], x[1], min(x[2] ** 2 + x[3] ** 2, 0.95), float(np.degrees(np.arctan2(x[3], x[2]))), x[4], x[5])
    resid = lambda x: (lambda q: q[5] + q[4] * binaries_radial(t, q[0], q[1], q[2], q[3]) - v)(unpack(x))
    fit = least_squares(resid, x0, loss="soft_l1", f_scale=1.0,
                        bounds=([P * 0.995, -np.inf, -0.97, -0.97, 0.3 * k1, -np.inf], [P * 1.005, np.inf, 0.97, 0.97, 2.0 * k1, np.inf]))
    return unpack(fit.x) + (float(1.4826 * np.median(np.abs(resid(fit.x)))),)


def binaries_onto(spectra, v=None, band="optical"):
    """Each spectrum on the band's grid, moved by its velocity V where given, over its running median of 501 samples, the telluric regions NaN."""
    from scipy.ndimage import median_filter
    g = giants_grid(band)
    out = np.full((len(spectra), len(g)), np.nan)
    for i, (w, f, e, meta) in enumerate(spectra):
        wl = w / (1.0 + (0.0 if v is None else v[i]) / 299792.458)
        ok = np.isfinite(f) & (f > 0)
        fi = np.interp(g, wl[ok], f[ok], left=np.nan, right=np.nan)
        cont = median_filter(np.nan_to_num(fi, nan=np.nanmedian(fi)), size=501, mode="nearest")
        with np.errstate(all="ignore"):
            out[i] = fi / cont
    for lo, hi in giants_BANDS[band]["telluric"]:
        out[:, (g > lo) & (g < hi)] = np.nan
    return out


def binaries_velocities(spectra, T, vmax=200, band="optical"):
    """Each spectrum's barycentric velocity (km/s): the peak of its cross-correlation with the line depths T over +-VMAX, refined by a parabola through the three highest samples."""
    c = giants_ccf(binaries_onto(spectra, None, band) - 1.0, T, None, vmax)
    v = np.arange(-vmax, vmax + 1, dtype=float)
    out = np.zeros(len(c))
    for i, row in enumerate(c):
        k = int(np.argmax(row))
        den = row[k - 1] - 2 * row[k] + row[k + 1] if 0 < k < len(row) - 1 else 0.0
        out[i] = v[k] + (0.5 * (row[k - 1] - row[k + 1]) / den if den < 0 else 0.0)
    return out


def binaries_plan_entries(inst=None):
    """The plan's entries, of one instrument where given."""
    return [x for x in json.load(open(os.path.join(binaries_ROOT, "plan.json"))) if inst is None or x["inst"] == inst]


def binaries_target(name, inst):
    """A binary's plan entry by name, spaces and punctuation ignored."""
    key = lambda s: re.sub(r"[^A-Za-z0-9]", "", s).upper()
    return next(x for x in binaries_plan_entries(inst) if key(x["name"]) == key(name))


def binaries_tag(target):
    """The file tag of a binary's series: its instrument and name."""
    return re.sub(r"[^A-Za-z0-9+_.-]", "_", "%s_%s" % (target["inst"], target["name"]))


def binaries_plan(min_n=20):
    """The ESO archive's series of MIN_N or more spectra of one target from HARPS, ESPRESSO, NIRPS, UVES or FEROS, each matched within 10 arcsec to a Gaia DR3 single-lined orbit (SB1, SB1C, AstroSpectroSB1; G below 13) or else an SB9 single-lined orbit, with the mass function, the companion's least mass for a primary of one solar mass and its K2 at that mass: ROOT/plan.json."""
    series = []
    for inst in ("HARPS", "ESPRESSO", "NIRPS", "UVES", "FEROS"):
        q = ("SELECT TOP 20000 target_name, COUNT(*) AS n, MIN(t_min) AS t0, MAX(t_max) AS t1, AVG(s_ra) AS ra, AVG(s_dec) AS dec FROM ivoa.ObsCore "
             "WHERE dataproduct_type = 'spectrum' AND instrument_name = '%s' GROUP BY target_name ORDER BY n DESC" % inst)
        for r in core.tap("eso", q, method="POST"):
            n, ra, dec, name = int(r["n"]), core.num(r["ra"]), core.num(r["dec"]), (r["target_name"] or "").strip()
            if n >= min_n and ra is not None and dec is not None and name.upper() not in ("SUN", "MOON", "SKY", ""):
                series.append(dict(inst=inst, name=name, n=n, span=round(core.num(r["t1"]) - core.num(r["t0"]), 1), ra=ra, dec=dec))
    gaia = core.tap_async("gaia", "SELECT o.source_id, o.nss_solution_type, o.period, o.semi_amplitude_primary, o.eccentricity, o.t_periastron, o.arg_periastron, "
                          "o.center_of_mass_velocity, g.ra, g.dec, g.phot_g_mean_mag, g.parallax, g.bp_rp FROM gaiadr3.nss_two_body_orbit AS o "
                          "JOIN gaiadr3.gaia_source AS g ON g.source_id = o.source_id WHERE o.nss_solution_type IN ('SB1', 'SB1C', 'AstroSpectroSB1') "
                          "AND g.phot_g_mean_mag < 13", maxrec=2000000)
    sb9 = [r for r in core.tap("vizier", 'SELECT m.Seq, m.RAJ2000, m.DEJ2000, m.Name, o.Per, o.T0, o.e, o.omega, o.K1, o.K2, o.V0 FROM "B/sb9/main" AS m '
                                          'JOIN "B/sb9/orbits" AS o ON o.Seq = m.Seq') if not core.num(r["K2"])]

    def cells(rows, rkey, dkey, size=0.05):
        grid = {}
        for r in rows:
            rr, dd = core.num(r[rkey]), core.num(r[dkey])
            if rr is not None and dd is not None:
                grid.setdefault((int(dd // size), int(rr // size)), []).append((rr, dd, r))
        return grid

    def nearest(grid, ra, dec, size=0.05):
        best = None
        for i in (int(dec // size) - 1, int(dec // size), int(dec // size) + 1):
            for j in range(int(ra // size) - 2, int(ra // size) + 3):
                for rr, dd, r in grid.get((i, j % int(360 / size)), []):
                    d = math.degrees(math.acos(max(-1.0, min(1.0, math.sin(math.radians(dec)) * math.sin(math.radians(dd)) +
                                                             math.cos(math.radians(dec)) * math.cos(math.radians(dd)) * math.cos(math.radians(ra - rr)))))) * 3600
                    if d <= 10.0 and (best is None or d < best[0]):
                        best = (d, r)
        return best

    def least_mass(f):
        lo, hi = 0.0, 50.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if mid ** 3 / (1.0 + mid) ** 2 < f else (lo, mid)
        return 0.5 * (lo + hi)
    out, ggrid, sgrid = [], cells(gaia, "ra", "dec"), cells(sb9, "RAJ2000", "DEJ2000")
    for s in series:
        hit, orbit = nearest(ggrid, s["ra"], s["dec"]), None
        if hit and core.num(hit[1]["period"]) and core.num(hit[1]["semi_amplitude_primary"]):
            g = {k: (v if k in ("source_id", "nss_solution_type") else core.num(v)) for k, v in hit[1].items()}
            orbit = dict(source="Gaia DR3 " + g["nss_solution_type"], gaia=g["source_id"], sep_arcsec=round(hit[0], 1), period=g["period"], k1=g["semi_amplitude_primary"],
                         e=g["eccentricity"] or 0.0, t_peri=g["t_periastron"], omega=g["arg_periastron"], gamma=g["center_of_mass_velocity"], g_mag=g["phot_g_mean_mag"],
                         parallax=g["parallax"], bp_rp=g["bp_rp"])
        else:
            hit = nearest(sgrid, s["ra"], s["dec"])
            if hit and core.num(hit[1]["Per"]) and core.num(hit[1]["K1"]):
                o = hit[1]
                orbit = dict(source="SB9", sb9=o["Seq"], sb9_name=(o["Name"] or "").strip(), sep_arcsec=round(hit[0], 1), period=core.num(o["Per"]), k1=core.num(o["K1"]),
                             e=core.num(o["e"]) or 0.0, t_peri=core.num(o["T0"]), omega=core.num(o["omega"]), gamma=core.num(o["V0"]))
        if orbit is None:
            continue
        f = 1.0361e-7 * (1 - orbit["e"] ** 2) ** 1.5 * orbit["k1"] ** 3 * orbit["period"]
        m2 = least_mass(f)
        out.append(dict(s, **orbit, mass_function=f, m2_min=round(m2, 3), k2_at_m2_min=round(orbit["k1"] / max(m2, 1e-6), 1), cycles=round(s["span"] / orbit["period"], 1)))
    out.sort(key=lambda t: (-(t["n"] >= 30), -t["n"]))
    os.makedirs(binaries_ROOT, exist_ok=True)
    json.dump(out, open(os.path.join(binaries_ROOT, "plan.json"), "w"), indent=1)
    print("%d series of single-lined binaries of %d stars" % (len(out), len({(round(t["ra"], 2), round(t["dec"], 2)) for t in out})))


def binaries_fetch(target, min_snr=10.0, per_star=300, streams=4):
    """A binary's public spectra from its instrument above MIN_SNR, up to PER_STAR spread over the span, fetched into ROOT/spectra/<tag>: the loaded spectra and the number listed."""
    q = ("SELECT dp_id, instrument_name, t_min, snr FROM ivoa.ObsCore WHERE dataproduct_type = 'spectrum' AND "
         "CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', %.5f, %.5f, 0.01)) = 1" % (target["ra"], target["dec"]))
    listed = sorted((r for r in core.tap("eso", q) if r["instrument_name"] == target["inst"] and (core.num(r["snr"]) or 0) > min_snr), key=lambda r: float(r["t_min"]))
    folder = os.path.join(binaries_ROOT, "spectra", binaries_tag(target))
    os.makedirs(folder, exist_ok=True)

    def one(r):
        path = os.path.join(folder, re.sub(r"[^A-Za-z0-9_.-]", "_", r["dp_id"]) + ".fits")
        try:
            return path if os.path.exists(path) or core.eso_download(r["dp_id"], path) else None
        except Exception:
            return None
    paths = [x for x in core.pmap(one, listed[::max(len(listed) // per_star, 1)][:per_star], streams) if x]
    return companions_spectra(paths), len(listed)


def binaries_template(name, inst="HARPS", out=None, spectra=None, objects=()):
    """A primary's line depths, one minus the median of its spectra each put at its catalogue orbit's velocity: the template against which the velocities are measured, ROOT/templates/<tag>.npy unless OUT; given a folder of SPECTRA, the line depths of the star whose OBJECT headers hold any of OBJECTS, from its spectra in its own frame, as the M-dwarf template is made."""
    if spectra:
        files = sorted({f for obj in objects for f in companions_files(spectra, obj)})
        loaded = companions_spectra(files)
        T = companions_template(loaded, companions_vsys(loaded), giants_band(inst))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.save(out, T)
        print("%s: line depths from %d spectra, %d samples above 0.1: %s" % (" ".join(objects), len(loaded), int((T > 0.1).sum()), out))
        return
    target = binaries_target(name, inst)
    spectra, _ = binaries_fetch(target)
    P, tp, e, w, k1, gamma = binaries_orbit(target)
    v = gamma + k1 * binaries_radial(np.array([s[3]["mjd"] for s in spectra]), P, tp, e, w)
    with np.errstate(all="ignore"):
        master = np.nanmedian(binaries_onto(spectra, v, giants_band(inst)), axis=0)
    T = np.where(np.isfinite(master), np.clip(1.0 - master, 0.0, None), 0.0)
    out = out or os.path.join(binaries_ROOT, "templates", binaries_tag(target) + ".npy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, T)
    print("%s: line depths from %d spectra, %d samples above 0.1: %s" % (target["name"], len(spectra), int((T > 0.1).sum()), out))


def binaries_planck(w, t2, t1):
    """B(w, T2) / B(w, T1) at wavelengths W (A)."""
    return np.expm1(1.4388e8 / (np.asarray(w, float) * t1)) / np.expm1(1.4388e8 / (np.asarray(w, float) * t2))


def binaries_build(names, inst="HARPS", min_snr=10.0, per_star=300, streams=4, rv_template=None, template=None, keep=False, variant=None, band=None,
                   inject=0.0, inject_k2=None, inject_shift=0.25, inject_dv=1.0, inject_template=None, teff1=6000.0, teff2=3400.0, orbit_frames=False):
    """Each named binary's rows: its spectra's velocities measured against RV_TEMPLATE and the orbit fitted to them, each spectrum put in the primary's frame at its own velocity, the residuals cross-correlated with the M-dwarf TEMPLATE (zero outside BAND where given) and the star's pattern taken out, kept as ROOT/rows/<tag>.npz, or ROOT/rows_<VARIANT>, with a note; with ORBIT_FRAMES each spectrum is put at the plan orbit's velocity instead and left unaligned, as in the near infrared, where a solar-type primary's own lines are too few to place it; INJECT lays the lines of INJECT_TEMPLATE (the M-dwarf template unless given) into every spectrum at that V-band flux ratio, scaled as a TEFF2 black body against a TEFF1 one, along the orbit moved by INJECT_SHIFT of a period at INJECT_K2 and offset INJECT_DV."""
    rv_template = rv_template or os.path.join(binaries_ROOT, "templates", "rv.npy")
    template = template or os.path.join(binaries_ROOT, "templates", "mdwarf.npy")
    Trv, T = np.load(rv_template), np.load(template)
    Ti = np.load(inject_template) if inject_template else T
    g = giants_grid(giants_band(inst))
    Tc = np.where((g >= band[0]) & (g < band[1]), T, 0.0) if band else T
    folder = os.path.join(binaries_ROOT, "rows" + ("_" + variant if variant else ""))
    os.makedirs(folder, exist_ok=True)
    for name in names:
        target = binaries_target(name, inst)
        spectra, listed = binaries_fetch(target, min_snr, per_star, streams)
        if len(spectra) < 20:
            json.dump(dict(name=target["name"], inst=inst, n=len(spectra), listed=listed, note="too few spectra"), open(os.path.join(folder, binaries_tag(target) + ".json"), "w"))
            print("%s: %d spectra loaded of %d listed; too few" % (target["name"], len(spectra), listed), flush=True)
            continue
        t = np.array([s[3]["mjd"] for s in spectra])
        if orbit_frames:
            P, tp, e, w, k1, gamma = binaries_orbit(target)
            v, rms = gamma + k1 * binaries_radial(t, P, tp, e, w), 0.0
        else:
            v = binaries_velocities(spectra, Trv, band=giants_band(inst))
            P, tp, e, w, k1, gamma, rms = binaries_refit(t, v, *binaries_orbit(target)[:5])
        for s, x in zip(spectra, v):
            s[3]["rv"] = float(x)
        if inject:
            # a companion's light laid in along a moved orbit, so it falls away from any real companion's track
            v2 = gamma + inject_dv - inject_k2 * binaries_radial(t, P, tp + inject_shift * P, e, w)
            scale = lambda wl: binaries_planck(wl, teff2, teff1) / binaries_planck(5500.0, teff2, teff1)
            spectra = [(wl, fl * (1.0 - inject * scale(wl) * np.interp(wl, g * (1.0 + x / 299792.458), Ti, left=0.0, right=0.0)), er, meta)
                       for (wl, fl, er, meta), x in zip(spectra, v2)]
        info = {}
        R, sd, _ = companions_residuals(spectra, gamma, giants_band(inst), align=not orbit_frames, info=info)
        if "vstar" not in info:
            info["vstar"] = np.asarray(v, float)
        ccf = companions_unpattern(giants_ccf(R, Tc, sd))
        tag = binaries_tag(target)
        np.savez_compressed(os.path.join(folder, tag + ".npz"), ccf=ccf.astype(np.float32), mjd=t, berv=np.array([s[3]["berv"] for s in spectra], float),
                            vstar=np.asarray(info["vstar"], float), vpred=v)
        json.dump(dict(name=target["name"], inst=inst, n=len(spectra), listed=listed, vsys=gamma, template=os.path.basename(template), band=band, inject=inject,
                       inject_k2=inject_k2, inject_shift=inject_shift if inject else None, inject_dv=inject_dv if inject else None,
                       inject_template=os.path.basename(inject_template or template) if inject else None), open(os.path.join(folder, tag + ".json"), "w"))
        print("%s: %d spectra of %d listed; orbit P %.5f d, K1 %.3f km/s, e %.3f, omega %.1f, gamma %.2f; scatter %.3f km/s%s" % (
            target["name"], len(spectra), listed, P, k1, e, w, gamma, rms, "; injected at %g, K2 %g, shift %g" % (inject, inject_k2, inject_shift) if inject else ""), flush=True)
        if not keep:
            import shutil
            shutil.rmtree(os.path.join(binaries_ROOT, "spectra", tag), ignore_errors=True)


def binaries_rows(target, variant=None, shift=0.0):
    """A binary's kept rows (ROOT/rows, or ROOT/rows_<VARIANT>) with the primary's orbit fitted to their measured velocities, the companion's track taken along that orbit moved by SHIFT of a period: a dict."""
    d = np.load(os.path.join(binaries_ROOT, "rows" + ("_" + variant if variant else ""), binaries_tag(target) + ".npz"))
    c, t, berv, vstar, vpred = (d[k].astype(float) for k in ("ccf", "mjd", "berv", "vstar", "vpred"))
    P, tp, e, w, k1, gamma, rms = binaries_refit(t, vpred, *binaries_orbit(target)[:5])
    centre = gamma - float(np.median(vstar - k1 * binaries_radial(t, P, tp, e, w)))
    tp = tp + shift * P
    return dict(c=c, t=t, tell=berv - vstar, vstar=vstar, orbit=(P, tp, e, w, k1, gamma), rms=rms, shape=binaries_radial(t, P, tp, e, w),
                centre=centre, night=np.floor(t + 0.3), shift=shift)


def binaries_clean(b, idx, k=None):
    """The rows of IDX cleaned of the sky's pattern and K principal components (five unless given), with each row's weight, the inverse square of its robust scatter more than 50 km/s from the star."""
    cc = companions_components(companions_untellur(b["c"][idx], b["tell"][idx]), binaries_COMPONENTS if k is None else k)
    v = np.arange(cc.shape[1], dtype=float) - (cc.shape[1] - 1) // 2
    return cc, 1.0 / np.maximum(1.4826 * np.median(np.abs(cc[:, np.abs(v - b["centre"]) > 50]), axis=1), 1e-6) ** 2


def binaries_curve(b, idx, cc, wt, k2, dv0=0.0, offset=0.0, shape=None):
    """The weighted mean of the cleaned rows read along the companion's track for each K2, its systemic velocity the primary's moved by DV0 and read OFFSET from the track, readings near the star's lines or the sky's left out; -inf where fewer than a fifth of the rows, and ten, remain."""
    P, tp, e, w, k1, gamma = b["orbit"]
    shape = b["shape"][idx] if shape is None else shape
    dv = gamma + dv0 - np.asarray(k2, float)[:, None] * shape[None, :] - b["vstar"][idx][None, :] + offset
    vmax = (cc.shape[1] - 1) // 2
    x = dv + vmax
    xc = np.clip(x, 0, 2 * vmax - 1)
    j = np.floor(xc).astype(int)
    f = xc - j
    r = np.arange(cc.shape[0])[None, :]
    vals = cc[r, j] * (1 - f) + cc[r, j + 1] * f
    use = (x >= 0) & (x <= 2 * vmax - 1) & (np.abs(dv - b["centre"]) > binaries_CORE) & (np.abs(dv - b["tell"][idx][None, :]) > binaries_TELL)
    W = np.where(use, wt[None, :], 0.0)
    out = (W * vals).sum(1) / np.maximum(W.sum(1), 1e-30)
    return np.where(use.sum(1) >= max(10, len(idx) // 5), out, -np.inf)


def binaries_nullshape(b, idx, kind, rng):
    """One null draw's track shape for the rows IDX: the orbit's phase moved by a tenth to nine tenths of a period, and kept a tenth of a period from the real companion's track when the track is moved, or whole nights shuffled."""
    P, tp, e, w = b["orbit"][:4]
    t = b["t"][idx]
    if kind == "nights":
        return binaries_radial(companions_night_shuffle(t, rng), P, tp, e, w)
    s = b.get("shift", 0.0) % 1.0
    while True:
        u = rng.uniform(0.1, 0.9)
        if min(abs(u - s), 1.0 - abs(u - s)) >= 0.1:
            return binaries_radial(t + u * P, P, tp, e, w)


def binaries_summit(b, idx, cc, wt, grid, dv0, draws, rng):
    """The summit of the curve over GRID at offset DV0 against its summits with the orbit's phase moved by a tenth to nine tenths of a period and with whole nights shuffled: (index, value, significances, the stricter null's summits)."""
    val = binaries_curve(b, idx, cc, wt, grid, dv0)
    i = int(np.argmax(val))
    nulls, z = {}, {}
    for kind in ("phase", "nights"):
        tops = np.array([float(np.max(binaries_curve(b, idx, cc, wt, grid, dv0, shape=binaries_nullshape(b, idx, kind, rng)))) for _ in range(draws)])
        nulls[kind], z[kind] = tops, float((val[i] - tops.mean()) / max(tops.std(), 1e-12))
    return i, val, z, nulls[min(z, key=z.get)]


def binaries_local(b, idx, cc, wt, k2c, dvc, half_k, half_v, step):
    """The map's peak over K2 and the offset on a grid of STEP around (K2C, DVC), then on a grid of 0.05 km/s within half a km/s of it; NaN where too few rows reach the grid."""
    for hk, hv, s in ((half_k, half_v, step), (0.5, 0.5, 0.05)):
        ks, vs = np.arange(k2c - hk, k2c + hk + s / 2, s), np.arange(dvc - hv, dvc + hv + s / 2, s)
        mm = np.array([binaries_curve(b, idx, cc, wt, ks, x) for x in vs])
        if not np.isfinite(mm.max()):
            return np.nan, np.nan
        jj, ii = np.unravel_index(int(np.argmax(mm)), mm.shape)
        k2c, dvc = float(ks[ii]), float(vs[jj])
    return k2c, dvc


def binaries_parts(b, k):
    """The rows split by date into K parts of about equal numbers of spectra, whole nights kept together: a list of index arrays."""
    nights = np.unique(b["night"])
    counts = np.array([(b["night"] == n).sum() for n in nights])
    part = np.minimum((k * (np.cumsum(counts) - counts / 2.0) / len(b["t"])).astype(int), k - 1)
    return [np.where(np.isin(b["night"], nights[part == j]))[0] for j in range(k)]


def binaries_seasons(b, gap=90.0):
    """The rows grouped into observing seasons, runs of nights with no gap longer than GAP days: a list of index arrays."""
    nights = np.unique(b["night"])
    season = np.concatenate([[0], np.cumsum(np.diff(nights) > gap)])
    return [np.where(np.isin(b["night"], nights[season == s]))[0] for s in np.unique(season)]


def binaries_tail(obs, null):
    """An observed summit against null summits: its height in units of the nulls' scatter, the empirical false-alarm probability (one more than the draws as high over one more than the draws), the tail probability of a Gumbel law fitted to the nulls by their moments with its one-sided Gaussian equivalent, the generalised extreme-value shape fitted to them (zero for Gumbel's, negative for a lighter tail), and the highest null in units of their scatter."""
    from scipy import stats
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    m, s = float(null.mean()), float(null.std())
    p = science.gumbel_p(null, obs)
    return dict(z=(obs - m) / s, p_emp=(1 + int((null >= obs).sum())) / (len(null) + 1.0), p_gumbel=p, sigma_gumbel=float(stats.norm.isf(max(p, 1e-300))),
                xi=-float(stats.genextreme.fit(null)[0]), top=(float(null.max()) - m) / s, draws=len(null))


def binaries_masses(b, k2, incl=90.0):
    """The masses (solar masses) from the orbit and K2 at inclination INCL: (M1, M2)."""
    P, tp, e, w, k1, gamma = b["orbit"]
    f = 1.0361e-7 * (1 - e * e) ** 1.5 * P * (k1 + np.asarray(k2, float)) ** 2 / np.sin(np.radians(incl)) ** 3
    return f * k2, f * k1


def binaries_search(names=(), inst="HARPS", dv=1.0, draws=300, detect=5.0):
    """Each binary with kept rows (or the named ones) searched over K2 at a fixed systemic offset DV: the summit against the two nulls, the lower significance kept, and a detection's K2 bootstrapped over whole nights; ROOT/results/<tag>.json."""
    os.makedirs(os.path.join(binaries_ROOT, "results"), exist_ok=True)
    targets = [binaries_target(n, inst) for n in names] if names else [x for x in binaries_plan_entries(inst) if os.path.exists(os.path.join(binaries_ROOT, "rows", binaries_tag(x) + ".npz"))]
    for target in targets:
        tag = binaries_tag(target)
        if not os.path.exists(os.path.join(binaries_ROOT, "rows", tag + ".npz")):
            continue
        rng = np.random.default_rng(zlib.crc32(tag.encode()))
        b = binaries_rows(target)
        idx = np.arange(len(b["t"]))
        cc, wt = binaries_clean(b, idx)
        grid = np.arange(max(b["orbit"][4], 5.0), 250.001, 0.25)
        i, val, z, _ = binaries_summit(b, idx, cc, wt, grid, dv, draws, rng)
        m1, m2 = binaries_masses(b, grid[i])
        P, tp, e, w, k1, gamma = b["orbit"]
        res = dict(name=target["name"], inst=inst, gaia=target.get("gaia"), rows=len(idx), nights=int(len(np.unique(b["night"]))), span_days=float(b["t"].max() - b["t"].min()),
                   period=P, e=e, omega=w, k1=k1, gamma=gamma, rms_kms=b["rms"], dv=dv, k2=float(grid[i]), z_phase=z["phase"], z_nights=z["nights"], z=min(z.values()),
                   m1_sin3i=float(m1), m2_sin3i=float(m2))
        if res["z"] >= detect:
            nights = np.unique(b["night"])
            ks = np.arange(grid[i] - 6.0, grid[i] + 6.001, 0.25)
            boot = []
            for _ in range(draws):
                pick = np.concatenate([np.where(b["night"] == n)[0] for n in rng.choice(nights, size=len(nights), replace=True)])
                c2, w2 = binaries_clean(b, pick)
                boot.append(float(ks[int(np.argmax(binaries_curve(b, pick, c2, w2, ks, dv)))]))
            res["k2_16_50_84"] = [float(q) for q in np.percentile(boot, [16, 50, 84])]
        json.dump(res, open(os.path.join(binaries_ROOT, "results", tag + ".json"), "w"), indent=1)
        print("%-12s %3d rows: orbit P %.4f d K1 %.2f e %.3f, rms %.3f km/s; summit K2 %.2f km/s, %.1f sigma (phases %.1f, nights %.1f); m sin^3 i %.3f + %.3f" % (
            target["name"], res["rows"], P, k1, e, b["rms"], res["k2"], res["z"], z["phase"], z["nights"], m1, m2), flush=True)


def binaries_stem(target, variant=None, components=None, shift=0.0):
    """The file stem of a binary's analysis: its tag, then the rows' variant, the principal components where not the default and the track's shift."""
    return binaries_tag(target) + ("_" + variant if variant else "") + ("_pc%d" % components if components is not None else "") + ("_shift%g" % shift if shift else "")


def binaries_measure(name, inst="HARPS", draws=300, incl=90.0, k2_fixed=None, dv_fixed=1.0, variant=None, components=None, shift=0.0, window=5.0):
    """One binary measured with the offset free: the map over K2 and the offset with its peak within WINDOW km/s of zero offset refined and bootstrapped over whole nights and over observing seasons, the curve at the peak's offset against the nulls and at zero offset, the series in four parts by date (two below 60 spectra) and in alternate nights each measured alone, the larger of the parts' standard error (Bessel-corrected) and the season bootstrap's half 16-84 width as the systematic uncertainty of K2 and the night bootstrap's half 16-84 width as its statistical one, the profile in the companion's frame, the masses at INCL, and the rows read at a published K2_FIXED and offset DV_FIXED against 2000 null draws; VARIANT reads ROOT/rows_<VARIANT>, COMPONENTS takes that many principal components out, SHIFT moves the track by that fraction of a period; ROOT/measure/<stem>.json with the curve and profile as CSV and the map as an array."""
    target = binaries_target(name, inst)
    stem = binaries_stem(target, variant, components, shift)
    out = os.path.join(binaries_ROOT, "measure")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    b = binaries_rows(target, variant, shift)
    allrows = np.arange(len(b["t"]))
    cc, wt = binaries_clean(b, allrows, components)
    k1 = b["orbit"][4]
    grid = np.arange(max(k1, 5.0), 250.001, 0.25)
    k2m, dvs = np.arange(max(k1, 5.0), 160.001, 0.5), np.arange(-30.0, 30.001, 0.5)
    m = np.array([binaries_curve(b, allrows, cc, wt, k2m, x) for x in dvs])
    m = np.where(np.isfinite(m), m, np.nan)
    j, i = np.unravel_index(int(np.nanargmax(np.where(np.abs(dvs)[:, None] <= window, m, np.nan))), m.shape)
    far = (np.abs(k2m[None, :] - k2m[i]) > 25) | (np.abs(dvs[:, None] - dvs[j]) > 15)
    ref = m[far & np.isfinite(m)]
    snr = (m - np.median(ref)) / (1.4826 * np.median(np.abs(ref - np.median(ref))))
    np.savez_compressed(os.path.join(out, stem + "_map.npz"), k2=k2m, dv=dvs, snr=snr)
    k2p, dvp = binaries_local(b, allrows, cc, wt, float(k2m[i]), float(dvs[j]), 4.0, 3.0, 0.1)
    ii, val, z, worst = binaries_summit(b, allrows, cc, wt, grid, dvp, draws, rng)
    zc = (val - worst.mean()) / max(worst.std(), 1e-12)
    ok = np.isfinite(zc)
    np.savetxt(os.path.join(out, stem + "_k2curve.csv"), np.c_[grid[ok], zc[ok]], delimiter=",", header="k2,z", comments="", fmt="%.3f")
    i0, _, z0, _ = binaries_summit(b, allrows, cc, wt, grid, 0.0, draws, rng)
    nights = np.unique(b["night"])
    boot = []
    for _ in range(draws):
        pick = np.concatenate([np.where(b["night"] == n)[0] for n in rng.choice(nights, size=len(nights), replace=True)])
        c2, w2 = binaries_clean(b, pick, components)
        boot.append(binaries_local(b, pick, c2, w2, k2p, dvp, 6.0, 4.0, 0.25))
    boot = np.array(boot)
    boot = boot[np.isfinite(boot).all(1)]
    m1, m2 = binaries_masses(b, boot[:, 0], incl)
    pct = lambda x: [float(q) for q in np.percentile(x, [16, 50, 84])]
    parts = []
    for idx in binaries_parts(b, 4 if len(allrows) >= 60 else 2):
        c2, w2 = binaries_clean(b, idx, components)
        k2s, dv2 = binaries_local(b, idx, c2, w2, k2p, dvp, 6.0, 4.0, 0.25)
        zs = binaries_summit(b, idx, c2, w2, grid, dvp, max(draws // 3, 50), rng)[2]
        parts.append(dict(k2=k2s, dv=dv2, z=min(zs.values()), rows=len(idx), mjd=[float(b["t"][idx].min()), float(b["t"][idx].max())]))
    alt = np.isin(b["night"], nights[::2])
    splits = {}
    for label, idx in (("alternate nights, one set", np.where(alt)[0]), ("alternate nights, other set", np.where(~alt)[0])):
        c2, w2 = binaries_clean(b, idx, components)
        k2s, dv2 = binaries_local(b, idx, c2, w2, k2p, dvp, 6.0, 4.0, 0.25)
        zs = binaries_summit(b, idx, c2, w2, grid, dvp, max(draws // 3, 50), rng)[2]
        splits[label] = dict(k2=k2s, dv=dv2, z=min(zs.values()), rows=len(idx))
    seasons, sboot = binaries_seasons(b), []
    if len(seasons) >= 4:
        for _ in range(draws):
            pick = np.concatenate([seasons[s] for s in rng.choice(len(seasons), size=len(seasons), replace=True)])
            c2, w2 = binaries_clean(b, pick, components)
            sboot.append(binaries_local(b, pick, c2, w2, k2p, dvp, 6.0, 4.0, 0.25)[0])
        sboot = [x for x in sboot if np.isfinite(x)]
    pk = np.array([p["k2"] for p in parts])
    bk = pct(boot[:, 0])
    stat = 0.5 * (bk[2] - bk[0])
    # the parts' standard error as the EBLM analyses take it, or the season bootstrap's half width where that is larger or the parts fail
    by_parts = float(np.std(pk, ddof=1) / np.sqrt(len(pk))) if np.isfinite(pk).all() else np.nan
    by_seasons = 0.5 * (np.percentile(sboot, 84) - np.percentile(sboot, 16)) if sboot else np.nan
    syst = float(np.nanmax([by_parts, by_seasons])) if np.isfinite([by_parts, by_seasons]).any() else np.nan
    total = float(np.hypot(stat, syst)) if np.isfinite(syst) else np.nan
    mt = binaries_masses(b, [k2p - total, k2p, k2p + total], incl) if np.isfinite(total) else (np.full(3, np.nan), np.full(3, np.nan))
    offs = np.arange(-100.0, 100.001, 1.0)
    prof = np.array([binaries_curve(b, allrows, cc, wt, [k2p], dvp, o)[0] for o in offs])
    prof = np.where(np.isfinite(prof), prof, np.nan)
    side = np.abs(offs) > 30
    prof = (prof - np.nanmedian(prof[side])) / (1.4826 * np.nanmedian(np.abs(prof[side] - np.nanmedian(prof[side]))))
    ok = np.isfinite(prof)
    np.savetxt(os.path.join(out, stem + "_profile.csv"), np.c_[offs[ok], prof[ok]], delimiter=",", header="offset,snr", comments="", fmt="%.3f")
    P, tp, e, w, k1, gamma = b["orbit"]
    note = os.path.join(binaries_ROOT, "rows" + ("_" + variant if variant else ""), binaries_tag(target) + ".json")
    res = dict(name=target["name"], inst=inst, gaia=target.get("gaia"), variant=variant, components=binaries_COMPONENTS if components is None else components, shift=shift,
               window=window, rows_note=json.load(open(note)) if os.path.exists(note) else None, rows=len(allrows), nights=int(len(nights)),
               span_days=float(b["t"].max() - b["t"].min()), orbit=dict(period=P, t_peri_mjd=tp, e=e, omega=w, k1=k1, gamma=gamma, rms_kms=b["rms"]), peak=dict(k2=k2p, dv=dvp),
               peak_value=float(binaries_curve(b, allrows, cc, wt, [k2p], dvp)[0]),
               summit=dict(k2=float(grid[ii]), z_phase=z["phase"], z_nights=z["nights"], z=min(z.values())), summit_zero_offset=dict(k2=float(grid[i0]), z=min(z0.values())),
               bootstrap=dict(draws=draws, k2_16_50_84=bk, dv_16_50_84=pct(boot[:, 1])), incl=incl, m1_16_50_84=pct(m1), m2_16_50_84=pct(m2),
               parts=parts, splits=splits, seasons=len(seasons), season_bootstrap=dict(draws=len(sboot), k2_16_50_84=pct(sboot)) if sboot else None,
               k2_stat=stat, k2_sys_parts=by_parts, k2_sys_seasons=float(by_seasons), k2_sys=syst, k2_total=total, m1_total=[float(x) for x in mt[0]], m2_total=[float(x) for x in mt[1]])
    if k2_fixed:
        at = float(binaries_curve(b, allrows, cc, wt, [k2_fixed], dv_fixed)[0])
        fixed = []
        for n in range(2000):
            t = b["t"] + rng.uniform(0.1, 0.9) * P if n % 2 else companions_night_shuffle(b["t"], rng)
            fixed.append(float(binaries_curve(b, allrows, cc, wt, [k2_fixed], dv_fixed, shape=binaries_radial(t, P, tp, e, w))[0]))
        fixed = np.array([x for x in fixed if np.isfinite(x)])
        res["fixed"] = dict(k2=k2_fixed, dv=dv_fixed, z=float((at - fixed.mean()) / fixed.std()), as_high=int((fixed >= at).sum()), of=len(fixed))
    json.dump(res, open(os.path.join(out, stem + ".json"), "w"), indent=1)
    print("%s %s: peak K2 %.2f (%.2f-%.2f) +- %.2f (stat) +- %.2f (sys, %d parts) at offset %+.2f km/s, %.1f sigma (phases %.1f, nights %.1f); M1 %.4f M2 %.4f at i %.3f%s" % (
        inst, stem, k2p, bk[0], bk[2], stat, syst, len(parts), dvp, res["summit"]["z"], z["phase"], z["nights"], res["m1_16_50_84"][1], res["m2_16_50_84"][1], incl,
        "; at K2 %.2f: %.1f sigma, %d of %d nulls as high" % (k2_fixed, res["fixed"]["z"], res["fixed"]["as_high"], res["fixed"]["of"]) if k2_fixed else ""), flush=True)


def binaries_calibrate(names=(), inst="HARPS", draws=10000, dv=1.0, window=5.0, step=0.5):
    """Each binary with kept rows (or the named ones) against DRAWS draws of each null, the orbit's phase moved and whole nights shuffled: the summit over K2 at the fixed offset DV, and the summit over K2 and over offsets within WINDOW km/s of zero in steps of STEP, of the rows and of every draw, each observed summit set against its nulls by binaries_tail and the null of the larger Gumbel probability adopted; ROOT/calibrate/<tag>.json with the null summits as ROOT/calibrate/<tag>.npz."""
    out = os.path.join(binaries_ROOT, "calibrate")
    os.makedirs(out, exist_ok=True)
    targets = [binaries_target(n, inst) for n in names] if names else [x for x in binaries_plan_entries(inst) if os.path.exists(os.path.join(binaries_ROOT, "rows", binaries_tag(x) + ".npz"))]
    offs = np.arange(-window, window + step / 2, step)
    for target in targets:
        tag = binaries_tag(target)
        if not os.path.exists(os.path.join(binaries_ROOT, "rows", tag + ".npz")):
            continue
        rng = np.random.default_rng(zlib.crc32((tag + ":calibrate").encode()))
        b = binaries_rows(target)
        idx = np.arange(len(b["t"]))
        cc, wt = binaries_clean(b, idx)
        grid = np.arange(max(b["orbit"][4], 5.0), 250.001, 0.25)

        def summits(shape=None):
            fixed = float(np.max(binaries_curve(b, idx, cc, wt, grid, dv, shape=shape)))
            return fixed, max([fixed] + [float(np.max(binaries_curve(b, idx, cc, wt, grid, o, shape=shape))) for o in offs])
        obs = summits()
        if not np.isfinite(obs[0]):
            print("%-12s %3d rows: too few rows reach the track" % (target["name"], len(idx)), flush=True)
            continue
        nulls, res = {}, dict(name=target["name"], inst=inst, rows=len(idx), nights=int(len(np.unique(b["night"]))), draws=draws, dv=dv, window=window, step=step,
                              observed=dict(fixed=obs[0], free=obs[1]))
        for kind in ("phase", "nights"):
            x = np.array([summits(binaries_nullshape(b, idx, kind, rng)) for _ in range(draws)])
            x[~np.isfinite(x)] = np.nan
            nulls[kind + "_fixed"], nulls[kind + "_free"] = x[:, 0], x[:, 1]
            res[kind] = dict(fixed=binaries_tail(obs[0], x[:, 0]), free=binaries_tail(obs[1], x[:, 1]))
        res["adopted"] = {st: dict(null=max(("phase", "nights"), key=lambda k: res[k][st]["p_gumbel"]), **max((res[k][st] for k in ("phase", "nights")), key=lambda r: r["p_gumbel"]))
                          for st in ("fixed", "free")}
        np.savez_compressed(os.path.join(out, tag + ".npz"), **nulls)
        json.dump(res, open(os.path.join(out, tag + ".json"), "w"), indent=1)
        a, f = res["adopted"]["fixed"], res["adopted"]["free"]
        print("%-12s %3d rows: offset %+.1f: %.1f in the nulls' scatter, highest null %.1f, p %.2g (empirical %.2g), %.1f sigma; offset free within %g: %.1f, p %.2g, %.1f sigma" % (
            target["name"], len(idx), dv, a["z"], a["top"], a["p_gumbel"], a["p_emp"], a["sigma_gumbel"], window, f["z"], f["p_gumbel"], f["sigma_gumbel"]), flush=True)


def binaries_survey(inst="HARPS"):
    """The calibrated binaries of one instrument taken together: for each null and each draw the highest of the binaries' null summits, each in units of its own nulls' scatter, and each binary's observed summit in those units set against the highest values, empirically and by a Gumbel law fitted to them, beside its own Gumbel probability compounded over the number of binaries (Sidak); the larger of the two and of the two nulls adopted; ROOT/calibrate/survey_<inst>.json."""
    from scipy import stats
    folder = os.path.join(binaries_ROOT, "calibrate")
    tags = [binaries_tag(x) for x in binaries_plan_entries(inst) if os.path.exists(os.path.join(folder, binaries_tag(x) + ".npz"))]
    per = {t: json.load(open(os.path.join(folder, t + ".json"))) for t in tags}
    arrays = {t: np.load(os.path.join(folder, t + ".npz")) for t in tags}
    n, d = len(tags), min(r["draws"] for r in per.values())
    sigma = lambda p: float(stats.norm.isf(max(p, 1e-300)))
    res = dict(inst=inst, binaries=n, draws=d, highest={}, targets={per[t]["name"]: {} for t in tags})
    for st in ("fixed", "free"):
        for kind in ("phase", "nights"):
            Z = np.array([(arrays[t][kind + "_" + st][:d] - np.nanmean(arrays[t][kind + "_" + st])) / np.nanstd(arrays[t][kind + "_" + st]) for t in tags])
            top = np.nanmax(Z, axis=0)
            res["highest"]["%s_%s" % (kind, st)] = dict(mean=float(top.mean()), sd=float(top.std()), max=float(top.max()), xi=-float(stats.genextreme.fit(top)[0]))
            for t in tags:
                z, own = per[t][kind][st]["z"], per[t][kind][st]["p_gumbel"]
                r = dict(null=kind, z=z, p_emp=(1 + int((top >= z).sum())) / (d + 1.0), p_gumbel=science.gumbel_p(top, z),
                         sidak=float(-np.expm1(n * np.log1p(-min(own, 1.0 - 1e-16)))))
                r["p"] = max(r["p_gumbel"], r["sidak"])
                r["sigma"] = sigma(r["p"])
                prev = res["targets"][per[t]["name"]].get(st)
                if prev is None or r["p"] > prev["p"]:
                    res["targets"][per[t]["name"]][st] = r
    json.dump(res, open(os.path.join(folder, "survey_%s.json" % inst), "w"), indent=1)
    for name, r in sorted(res["targets"].items(), key=lambda kv: kv[1]["fixed"]["p"]):
        f, v = r["fixed"], r["free"]
        print("%-12s of %d: offset fixed p %.2g (highest of the survey's nulls %.2g, empirical %.2g; Sidak %.2g), %.1f sigma; offset free p %.2g, %.1f sigma" % (
            name, n, f["p"], f["p_gumbel"], f["p_emp"], f["sidak"], f["sigma"], v["p"], v["sigma"]))


def binaries_record(detect=5.0):
    """The searched binaries' results, calibrations and measurements written to the store: one row per binary and instrument, its target the primary's Gaia DR3 identifier followed by # and the instrument, its statistic the summit over K2 at the fixed offset in units of its nulls' scatter and its p the Gumbel law's false-alarm probability, with the probability across the survey's binaries in the note; a companion whose probability falls below DETECT sigma (one-sided) marked as the eclipsing binary's companion, one between 3 sigma and that a candidate, and a series too thin to search as not measured."""
    from scipy import stats
    limit = float(stats.norm.sf(detect))
    survey = {}
    for inst in ("HARPS", "ESPRESSO"):
        p = os.path.join(binaries_ROOT, "calibrate", "survey_%s.json" % inst)
        if os.path.exists(p):
            s = json.load(open(p))
            survey.update({(inst, name): dict(v, binaries=s["binaries"]) for name, v in s["targets"].items()})
    rows, found, candidates = [], [], []
    for target in binaries_plan_entries():
        tag = binaries_tag(target)
        load = lambda sub: json.load(open(os.path.join(binaries_ROOT, sub, tag + ".json"))) if os.path.exists(os.path.join(binaries_ROOT, sub, tag + ".json")) else None
        r, m, note, c = load("results"), load("measure"), load("rows"), load("calibrate")
        if r is None and note is None:
            continue
        key = "gaia_dr3:%s#%s" % (target.get("gaia"), target["inst"])
        if r is None or not np.isfinite(r["z"]) or c is None:
            rows.append(dict(target=key, stat=None, unit="summit over K2 in units of its nulls' scatter", vet="not-measured",
                             note="%s: %d spectra%s, too few to search" % (target["name"], (r or note)["rows" if r else "n"], " on %d nights" % r["nights"] if r else "")))
            continue
        a, f = c["adopted"]["fixed"], c["adopted"]["free"]
        hit = a["p_gumbel"] < limit
        found += [target["name"]] if hit else []
        candidates += [target["name"]] if not hit and a["sigma_gumbel"] >= 3.0 else []
        text = ("%s: P %.4f d, K1 %.2f km/s, e %.3f; summit K2 %.2f km/s at offset %+.1f km/s, %.1f in units of its nulls' scatter, the highest of %d null draws %.1f; "
                "false-alarm probability %.2g (%.1f sigma), empirical %.2g" % (target["name"], r["period"], r["k1"], r["e"], r["k2"], r["dv"], a["z"], 2 * c["draws"], a["top"],
                                                                                a["p_gumbel"], a["sigma_gumbel"], a["p_emp"]))
        s = survey.get((target["inst"], target["name"]))
        if s:
            text += ", across the survey's %d binaries %.2g (%.1f sigma)" % (s["binaries"], s["fixed"]["p"], s["fixed"]["sigma"])
        text += "; offset free within %g km/s %.2g (%.1f sigma)" % (c["window"], f["p_gumbel"], f["sigma_gumbel"])
        if m and np.isfinite(m.get("k2_total", np.nan)):
            text += "; offset free: K2 %.2f +- %.2f (stat) +- %.2f (sys) km/s at %+.2f km/s, masses %.3f + %.4f Msun at i %.2f" % (
                m["peak"]["k2"], m["k2_stat"], m["k2_sys"], m["peak"]["dv"], m["m1_total"][1], m["m2_total"][1], m["incl"])
        rows.append(dict(target=key, stat=round(a["z"], 1), p=float(a["p_gumbel"]),
                         unit="summit over K2 in units of its nulls' scatter, p from a Gumbel law fitted to %d draws of each null (%s, %d spectra)" % (c["draws"], target["inst"], r["rows"]),
                         vet="eclipsing-binary" if hit else None, note=text))
    n = len({x["target"].split("#")[0] for x in rows})
    core.record_run(
        "binaries_eblm_known_orbits", "Companions of single-lined binaries along their known orbits",
        "ESO Phase 3 (HARPS, ESPRESSO); Gaia DR3 non-single-star orbits; SB9",
        "the EBLM binaries with Gaia DR3 single-lined orbits and archived HARPS series of 20 or more spectra above S/N 10, periods below 100 d covered three times; TOI-1338's ESPRESSO series",
        os.path.abspath(__file__), "binaries", n, "binaries", dict(dv=1.0, components=binaries_COMPONENTS, core_kms=binaries_CORE, tell_kms=binaries_TELL, detect=detect, draws=10000, window_kms=5.0),
        "the summit over K2 against 10,000 draws of each null, the orbit's phase moved by 0.1 to 0.9 of a period and whole nights shuffled, by a Gumbel law fitted to the null summits, "
        "the null of the larger probability adopted; across the survey's binaries from the highest standardised null of each draw and by the per-binary probability compounded (Sidak)",
        "a companion stands when its false-alarm probability reaches %g sigma; with the companion's systemic offset free, its K2 bootstrapped over whole nights with the larger of the parts' "
        "standard error and the season bootstrap as its systematic uncertainty; checked against 3 and 8 principal components, each half of the band, a second M-dwarf template and "
        "companions injected into the spectra" % detect,
        "%d series of %d binaries; %d companions at %g sigma or more: %s; candidates between 3 and %g sigma: %s" % (
            len(rows), n, len(found), detect, ", ".join(sorted(found)) or "none", detect, ", ".join(sorted(candidates)) or "none"), rows)
    print("%d rows recorded; companions: %s; candidates: %s" % (len(rows), ", ".join(sorted(found)) or "none", ", ".join(sorted(candidates)) or "none"))


def binaries_cli():
    quiet_warnings()
    p = argparse.ArgumentParser(prog="seti.py composition binaries")
    p.add_argument("mode", choices=("plan", "template", "build", "search", "calibrate", "measure", "record"))
    p.add_argument("names", nargs="*")
    p.add_argument("--inst", default="HARPS")
    p.add_argument("--min", type=int, default=20)
    p.add_argument("--out")
    p.add_argument("--spectra", help="template: a folder of a star's spectra")
    p.add_argument("--object", nargs="+", default=[], help="template: the OBJECT headers of that star's spectra")
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--per-star", type=int, default=300)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--rv-template")
    p.add_argument("--template")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--variant", help="build, measure: the rows kept as ROOT/rows_<variant>")
    p.add_argument("--band", nargs=2, type=float, default=None, help="build: the template kept between these wavelengths (A)")
    p.add_argument("--inject", type=float, default=0.0, help="build: the V-band flux ratio of the companion laid into the spectra")
    p.add_argument("--inject-k2", type=float, default=None)
    p.add_argument("--inject-shift", type=float, default=0.25)
    p.add_argument("--inject-dv", type=float, default=1.0)
    p.add_argument("--inject-template")
    p.add_argument("--dv", type=float, default=1.0)
    p.add_argument("--draws", type=int, default=None, help="null draws: 300, or 10000 to calibrate")
    p.add_argument("--window", type=float, default=5.0, help="calibrate, measure: the offsets searched, km/s either side of zero")
    p.add_argument("--aggregate", action="store_true", help="calibrate: the survey's binaries taken together")
    p.add_argument("--detect", type=float, default=5.0)
    p.add_argument("--incl", type=float, default=90.0)
    p.add_argument("--k2-fixed", type=float, default=None)
    p.add_argument("--dv-fixed", type=float, default=1.0)
    p.add_argument("--components", type=int, default=None)
    p.add_argument("--shift", type=float, default=0.0, help="measure: the track moved by this fraction of a period")
    p.add_argument("--orbit-frames", action="store_true", help="build: each spectrum at the plan orbit's velocity, unaligned, as for the near infrared")
    a = p.parse_args()
    if a.mode == "plan":
        binaries_plan(a.min)
    elif a.mode == "template":
        binaries_template(a.names[0] if a.names else None, a.inst, a.out, a.spectra, a.object)
    elif a.mode == "build":
        binaries_build(a.names, a.inst, a.min_snr, a.per_star, a.streams, a.rv_template, a.template, a.keep, a.variant, a.band, a.inject, a.inject_k2, a.inject_shift,
                       a.inject_dv, a.inject_template, orbit_frames=a.orbit_frames)
    elif a.mode == "search":
        binaries_search(a.names, a.inst, a.dv, a.draws or 300, a.detect)
    elif a.mode == "calibrate":
        if a.aggregate:
            binaries_survey(a.inst)
        else:
            binaries_calibrate(a.names, a.inst, a.draws or 10000, a.dv, a.window)
    elif a.mode == "measure":
        binaries_measure(a.names[0], a.inst, a.draws or 300, a.incl, a.k2_fixed, a.dv_fixed, a.variant, a.components, a.shift, a.window)
    else:
        binaries_record(a.detect)


def giants_reflected(planets=(), max_spectra=300, streams=4, instruments=("HARPS", "ESPRESSO"), components=5, probe=1e-4, redo=False):
    """Reflected starlight of the named planets (or every non-transiting giant of the plan with an orbit): up to MAX_SPECTRA archived spectra of the host spread over its span, each on the grid in the star's frame over its running median and the series' median, whitened; the star's own absorption pattern as the template, its residual pattern taken out of each cross-correlation row at its own strength and COMPONENTS principal components of the rows taken out after it; the rows summed in the planet's frame over K_p and rest velocity, weighted by the Lambert phase function and only where the planet moves more than 25 km/s from the star; the value at the orbit's K_p against the map's scatter away from the star and a sham of the rows with their phases shuffled; the gain measured by laying a copy of the star's lines at contrast PROBE, dimmed by the phase function, into the same spectra along the planet's velocity and reading the map's rise at the orbit through the same cleaning, so the value and the 3 sigma limit are stated as contrasts; results under ROOT/results/<planet>.reflected.json."""
    plan = json.load(open(os.path.join(giants_ROOT, "plan.json")))
    kp_grid = np.arange(0.0, 301.0, 5.0)
    vrest = np.arange(-100.0, 101.0, 1.0)
    quiet = (kp_grid >= 60.0)[:, None] & (np.abs(vrest) > 40.0)[None, :]
    os.makedirs(os.path.join(giants_ROOT, "results"), exist_ok=True)

    def cleaned(R, sd, master):
        T = np.where(np.isfinite(master), np.clip(1.0 - master, 0.0, None), 0.0)
        c = companions_unpattern(giants_ccf(R, T, sd))
        if components:
            U, S, Vt = np.linalg.svd(c, full_matrices=False)
            c = c - (U[:, :components] * S[:components]) @ Vt[:components]
        return c, T
    for p in plan:
        if (planets and p["pl_name"] not in planets) or (not planets and p.get("tran_flag")):
            continue
        tag = re.sub(r"[^A-Za-z0-9+_.-]", "_", p["pl_name"])
        rpath = os.path.join(giants_ROOT, "results", tag + ".reflected.json")
        if os.path.exists(rpath) and not redo:
            continue
        rows = [r for inst in instruments for r in (p["instruments"].get(inst) or []) if r["src"] == "ESO" and (r.get("snr") or 0) > 50]
        if len(rows) < 30:
            continue
        rows.sort(key=lambda r: r["t0"])
        pick = rows[::max(len(rows) // max_spectra, 1)][:max_spectra]
        folder = os.path.join(giants_ROOT, "reflected", tag)
        os.makedirs(folder, exist_ok=True)

        def one(r):
            path = os.path.join(folder, re.sub(r"[^A-Za-z0-9_.-]", "_", r["id"]) + ".fits")
            if os.path.exists(path):
                return path
            try:
                return path if core.eso_download(r["id"], path) else None
            except Exception:
                # a product the archive refuses, proprietary or withdrawn, is passed over
                return None
        paths = [x for x in core.pmap(one, pick, streams) if x]
        spectra = companions_spectra(paths)
        if len(spectra) < 30:
            continue
        per, tc = p["pl_orbper"], p["pl_tranmid"] - 2400000.5
        kp_true = 2 * np.pi * (p["pl_orbsmax"] or 0.05) * 1.495978707e8 / (per * 86400.0)
        vsys = companions_vsys(spectra, p.get("st_radv"))
        band = giants_band(instruments[0])
        g = giants_grid(band)
        phase = np.array([((s[3]["mjd"] - tc) / per) % 1.0 for s in spectra])
        vp_unit = np.sin(2 * np.pi * phase)
        # Lambert's phase function at the phase angle, zero at inferior conjunction and pi at superior conjunction, where the lit face is full;
        # rows near the star's velocity left out
        alpha = 2.0 * np.pi * np.abs(phase - 0.5)
        lam = (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi
        w = np.where(np.abs(kp_true * vp_unit) > 25.0, lam, 0.0)
        if w.sum() <= 0:
            continue
        R, sd, master = companions_residuals(spectra, vsys, band)
        ccf, T = cleaned(R, sd, master)
        m = giants_planet_frame(ccf, vp_unit, w, kp_grid, vrest)
        # the probe: the star's own lines at contrast PROBE times the phase function, shifted by the planet's velocity, laid into each spectrum's flux
        laid = []
        for (wv, f, e, meta), vp, lam_k in zip(spectra, kp_true * vp_unit, lam):
            vstar = meta["rv"] if meta.get("rv") is not None else vsys
            tw = np.interp(wv, g * (1.0 + (vstar + vp) / 299792.458), T, left=0.0, right=0.0)
            laid.append((wv, f * (1.0 - probe * lam_k * tw), e, meta))
        Rp, sdp, masterp = companions_residuals(laid, vsys, band)
        ccfp, _ = cleaned(Rp, sdp, masterp)
        mp = giants_planet_frame(ccfp, vp_unit, w, kp_grid, vrest)
        i0, j0 = int(np.argmin(np.abs(kp_grid - kp_true))), int(np.argmin(np.abs(vrest)))
        gain = float(mp[i0, j0] - m[i0, j0]) / probe
        # the gain the whitened template would give with nothing lost: its norm times the phase function under the rows' weights
        ok = np.isfinite(R).all(0)
        gain_white = float(np.sqrt(((T / sd) ** 2)[ok].sum())) * float((w * lam).sum() / w.sum())
        rng = np.random.default_rng(5)
        shams = [giants_planet_frame(ccf, vp_unit[rng.permutation(len(vp_unit))], w, kp_grid, vrest) for _ in range(5)]
        noise = max(float(np.nanstd(m[quiet])), float(np.mean([np.nanstd(x[quiet]) for x in shams])))
        near = (np.abs(kp_grid - kp_true) <= 30)[:, None] & (kp_grid >= 30.0)[:, None] & (np.abs(vrest) <= 10)[None, :]
        val = float(np.nanmax(np.where(near, m, -np.inf)))
        above = np.where(kp_grid[:, None] >= 30.0, m, -np.inf)
        i, j = np.unravel_index(int(np.nanargmax(above)), m.shape)
        sham_max = float(np.mean([np.nanmax(np.where(near, x, -np.inf)) for x in shams]))
        res = dict(planet=p["pl_name"], n=len(spectra), n_eff=float(w.sum() ** 2 / (w ** 2).sum()), kp_true=kp_true, vsys=vsys, components=components, probe=probe,
                   snr=val / noise if noise > 0 else 0.0, sham=sham_max / noise if noise > 0 else 0.0,
                   map_peak=dict(kp=float(kp_grid[i]), vrest=float(vrest[j]), snr=float(m[i, j] / noise) if noise > 0 else 0.0), noise=noise, value=val,
                   orbit=float(m[i0, j0]), gain=gain, gain_white=gain_white, retained=gain / gain_white if gain_white > 0 else None,
                   contrast=float(m[i0, j0]) / gain if gain > 0 else None, limit_3sigma=3.0 * noise / gain if gain > 0 else None)
        json.dump(res, open(rpath, "w"), indent=1)
        print("%-14s %3d spectra (%.0f effective): reflected light at K_p %.0f: %.1f sigma (sham %.1f); map peak %.1f at K_p %.0f, v %+.0f; contrast %+.0f ppm, 3 sigma limit %.0f ppm; probe retained %.2f" % (
            p["pl_name"], len(spectra), res["n_eff"], kp_true, res["snr"], res["sham"], res["map_peak"]["snr"], res["map_peak"]["kp"], res["map_peak"]["vrest"],
            1e6 * (res["contrast"] or 0), 1e6 * (res["limit_3sigma"] or 0), res["retained"] or 0), flush=True)

def giants_emission(planets=(), max_spectra=300, streams=4, species=("CO", "H2O"), sysrem=6, components=0, probe=3e-4, redo=False):
    """Dayside thermal emission of the named planets (or every giant of the plan hotter than 1000 K with thirty or more SPIRou spectra): up to MAX_SPECTRA telluric-corrected SPIRou spectra from the CADC, those at the phases where the planet's dayside faces us and it moves more than 25 km/s from its star first; each on the near-infrared grid in the star's frame over its running median and the series' median, SYSREM systematics taken out of the residuals, whitened; each molecule's petitRADTRANS cross-section at the planet's temperature as the template; the rows' cross-correlations with the static pattern taken out and COMPONENTS principal components after it, summed in the planet's frame over K_p and rest velocity with the Lambert weights; the peak within 10 km/s of rest between 30 km/s and the edge-on amplitude against the map's scatter away from the star and shams with the phases shuffled; the gain from a copy of the template laid into the same spectra at contrast PROBE along the edge-on orbit through the same cleaning; results under ROOT/results/<planet>.emission.json."""
    from setitools import biosignatures as bio
    plan = json.load(open(os.path.join(giants_ROOT, "plan.json")))
    kp_grid = np.arange(0.0, 301.0, 5.0)
    vrest = np.arange(-100.0, 101.0, 1.0)
    band = "nir"
    g = giants_grid(band)
    os.makedirs(os.path.join(giants_ROOT, "results"), exist_ok=True)

    def cleaned(R, sd, T):
        c = companions_unpattern(giants_ccf(R, T, sd))
        if components:
            U, S, Vt = np.linalg.svd(c, full_matrices=False)
            c = c - (U[:, :components] * S[:components]) @ Vt[:components]
        return c
    for p in plan:
        if (planets and p["pl_name"] not in planets) or (not planets and (p.get("pl_eqt") or 0) < 1000):
            continue
        tag = re.sub(r"[^A-Za-z0-9+_.-]", "_", p["pl_name"])
        rpath = os.path.join(giants_ROOT, "results", tag + ".emission.json")
        if os.path.exists(rpath) and not redo:
            continue
        rows = [r for r in (p["instruments"].get("SPIRou") or []) if r["src"] == "CADC" and r["id"].endswith("t") and r["place"] != "transit"]
        if len(rows) < 30:
            continue
        per, tc = p["pl_orbper"], p["pl_tranmid"] - 2400000.5
        kp_true = 2 * np.pi * (p["pl_orbsmax"] or 0.05) * 1.495978707e8 / (per * 86400.0)

        def lambert(ph):
            alpha = 2.0 * np.pi * np.abs(ph - 0.5)
            return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi
        for r in rows:
            ph = ((0.5 * (r["t0"] + r["t1"]) - tc) / per) % 1.0
            r["weight"] = float(lambert(ph)) if abs(kp_true * np.sin(2 * np.pi * ph)) > 25.0 else 0.0
        pick = sorted((r for r in rows if r["weight"] > 0), key=lambda r: -r["weight"])[:max_spectra]
        if len(pick) < 30:
            continue
        folder = os.path.join(giants_ROOT, "emission", tag)
        os.makedirs(folder, exist_ok=True)

        def one(r):
            path = os.path.join(folder, re.sub(r"[^A-Za-z0-9_.-]", "_", r["id"]) + ".fits")
            if os.path.exists(path):
                return path
            try:
                b = core.get(bio.shoreline_CADC_FILE + r["id"] + ".fits", timeout=600, soft=True)
                if b and b[:6] == b"SIMPLE":
                    open(path + ".tmp", "wb").write(b)
                    os.replace(path + ".tmp", path)
                    return path
            except Exception:
                pass
            return None
        paths = [x for x in core.pmap(one, pick, streams) if x]
        spectra = companions_spectra(paths)
        if len(spectra) < 30:
            continue
        vsys = companions_vsys(spectra, p.get("st_radv"))
        phase = np.array([((s[3]["mjd"] - tc) / per) % 1.0 for s in spectra])
        vp_unit = np.sin(2 * np.pi * phase)
        lam = lambert(phase)
        w = np.where(np.abs(kp_true * vp_unit) > 25.0, lam, 0.0)
        if w.sum() <= 0:
            continue
        t_k = p.get("pl_eqt") or 1500.0
        # telluric residuals move with the barycentric correction, so the rows are not aligned on the star's lines here
        R, sd, master = companions_residuals(spectra, vsys, band, align=False)
        if sysrem:
            R = giants_sysrem(R, sd, sysrem)
        quiet = (kp_grid >= 30.0)[:, None] & (np.abs(vrest) > 40.0)[None, :]
        near = ((kp_grid >= 30.0) & (kp_grid <= kp_true + 30.0))[:, None] & (np.abs(vrest) <= 10)[None, :]
        i0, j0 = int(np.argmin(np.abs(kp_grid - kp_true))), int(np.argmin(np.abs(vrest)))
        rng = np.random.default_rng(5)
        perms = [rng.permutation(len(vp_unit)) for _ in range(5)]
        results = []
        for sp in species:
            T = giants_template(sp, t_k, band=band)
            if not T.any():
                continue
            ccf = cleaned(R, sd, T)
            m = giants_planet_frame(ccf, vp_unit, w, kp_grid, vrest)
            shams = [giants_planet_frame(ccf, vp_unit[q], w, kp_grid, vrest) for q in perms]
            noise = max(float(np.nanstd(m[quiet])), float(np.mean([np.nanstd(x[quiet]) for x in shams])))
            val = float(np.nanmax(np.where(near, m, -np.inf)))
            a, b = np.unravel_index(int(np.nanargmax(np.where(near, m, -np.inf))), m.shape)
            i, j = np.unravel_index(int(np.nanargmax(np.where(kp_grid[:, None] >= 30.0, m, -np.inf))), m.shape)
            sham_max = float(np.mean([np.nanmax(np.where(near, x, -np.inf)) for x in shams]))
            # the probe: the template's lines at contrast PROBE times the phase function along the edge-on orbit, laid into each spectrum's flux
            laid = []
            for (wv, f, e, meta), vp, lam_k in zip(spectra, kp_true * vp_unit, lam):
                vstar = meta["rv"] if meta.get("rv") is not None else vsys
                tw = np.interp(wv, g * (1.0 + (vstar + vp) / 299792.458), T, left=0.0, right=0.0)
                laid.append((wv, f * (1.0 - probe * lam_k * tw), e, meta))
            Rp, sdp, _ = companions_residuals(laid, vsys, band, align=False)
            if sysrem:
                Rp = giants_sysrem(Rp, sdp, sysrem)
            mp = giants_planet_frame(cleaned(Rp, sdp, T), vp_unit, w, kp_grid, vrest)
            gain = float(mp[i0, j0] - m[i0, j0]) / probe
            res = dict(species=sp, snr=val / noise if noise > 0 else 0.0, sham=sham_max / noise if noise > 0 else 0.0, at=dict(kp=float(kp_grid[a]), vrest=float(vrest[b])),
                       map_peak=dict(kp=float(kp_grid[i]), vrest=float(vrest[j]), snr=float(m[i, j] / noise) if noise > 0 else 0.0), noise=noise, value=val,
                       gain=gain, contrast=val / gain if gain > 0 else None, limit_3sigma=3.0 * noise / gain if gain > 0 else None)
            results.append(res)
            print("  %-12s SPIRou %-4s S/N %5.1f at K_p %.0f, v %+.0f (edge-on K_p %.0f; sham %.1f; map peak %.1f at K_p %.0f, v %+.0f); contrast %+.0f ppm, 3 sigma limit %.0f ppm, probe gain %.0f" % (
                p["pl_name"], sp, res["snr"], kp_grid[a], vrest[b], kp_true, res["sham"], res["map_peak"]["snr"], kp_grid[i], vrest[j], 1e6 * (res["contrast"] or 0), 1e6 * (res["limit_3sigma"] or 0), gain), flush=True)
        if results:
            json.dump(dict(planet=p["pl_name"], inst="SPIRou", n=len(spectra), n_eff=float(w.sum() ** 2 / (w ** 2).sum()), kp_true=kp_true, vsys=vsys, temp=t_k, sysrem=sysrem,
                           components=components, probe=probe, species=results), open(rpath, "w"), indent=1)


def giants_cli():
    quiet_warnings()
    p = argparse.ArgumentParser(prog="seti.py composition giants")
    p.add_argument("mode", choices=("plan", "fetch", "search", "record", "reflected", "emission"))
    p.add_argument("--max-spectra", type=int, default=300)
    p.add_argument("--vmax", type=float, default=11.0)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--planets", nargs="*", default=[])
    p.add_argument("--night-per-in", type=float, default=2.0)
    p.add_argument("--instruments", nargs="*", default=[])
    p.add_argument("--min-in", type=int, default=4)
    p.add_argument("--species", nargs="*", default=list(giants_SPECIES))
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--temp", type=float, default=0.0)
    # SysRem passes, principal components and the probe's contrast default per mode: search 0 passes, reflected 5 components at 1e-4,
    # emission 6 passes and no components at 3e-4
    p.add_argument("--sysrem", type=int, default=None)
    p.add_argument("--weight", choices=("linear", "sqrt", "log"), default="log")
    p.add_argument("--inject", type=float, default=0.0)
    p.add_argument("--no-whiten", action="store_true")
    p.add_argument("--no-shadow", action="store_true")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--components", type=int, default=None)
    p.add_argument("--probe", type=float, default=None)
    a = p.parse_args()
    pick = lambda value, default: default if value is None else value
    if a.mode == "plan":
        giants_plan(a.vmax)
    elif a.mode == "fetch":
        giants_fetch(a.streams, a.planets, a.night_per_in, a.instruments, a.min_in)
    elif a.mode == "record":
        giants_record(a.sigma)
    elif a.mode == "reflected":
        giants_reflected(a.planets, a.max_spectra, a.streams, components=pick(a.components, 5), probe=pick(a.probe, 1e-4), redo=a.redo)
    elif a.mode == "emission":
        giants_emission(a.planets, a.max_spectra, a.streams, species=tuple(a.species) if a.species != list(giants_SPECIES) else ("CO", "H2O"),
                        sysrem=pick(a.sysrem, 6), components=pick(a.components, 0), probe=pick(a.probe, 3e-4), redo=a.redo)
    else:
        giants_search(a.planets, a.species, a.sigma, a.temp, pick(a.sysrem, 0), a.weight, a.inject, not a.no_whiten, not a.no_shadow, a.redo, tuple(a.instruments))


APPROACHES = {
    "giants": (giants_cli, giants_DOC),
    "companions": (companions_cli, companions_DOC),
    "binaries": (binaries_cli, binaries_DOC),
    "combs": (lambda: core.run_cli(combs_main), combs_DOC),
    "ghosts": (lambda: core.run_cli(ghosts_main), ghosts_DOC),
    "subdwarfindex": (lambda: core.run_cli(subdwarfindex_main), subdwarfindex_DOC),
    "misfitspectra": (lambda: core.run_cli(misfitspectra_main), misfitspectra_DOC),
    "xpmisfits": (lambda: core.run_cli(xpmisfits_main), xpmisfits_DOC),
    "xpworst": (lambda: core.run_cli(xpworst_main), xpworst_DOC),
    "worstfits": (lambda: core.run_cli(worstfits_main), worstfits_DOC),
    "singlesd": (lambda: core.run_cli(singlesd_main), singlesd_DOC),
    "carbondwarfs": (lambda: core.run_cli(carbondwarfs_main), carbondwarfs_DOC),
    "limeteoritic": (lambda: core.run_cli(limeteoritic_main), limeteoritic_DOC),
    "zirconium": (lambda: core.run_cli(zirconium_main), zirconium_DOC),
    "europium": (lambda: core.run_cli(europium_main), europium_DOC),
    "rvjump": (lambda: core.run_cli(rvjump_main), rvjump_DOC),
    "ceriumonly": (lambda: core.run_cli(ceriumonly_main), ceriumonly_DOC),
    "evenheavy": (lambda: core.run_cli(evenheavy_main), evenheavy_DOC),
    "haquiet": (lambda: core.run_cli(haquiet_main), haquiet_DOC),
    "przybylski": (lambda: core.run_cli(przybylski_main), przybylski_DOC),
    "apogeebad": (lambda: core.run_cli(apogeebad_main), apogeebad_DOC),
    "sdssunknown": (lambda: core.run_cli(sdssunknown_main), sdssunknown_DOC),
    "lamostunknown": (lambda: core.run_cli(lamostunknown_main), lamostunknown_DOC),
    "desiunknown": (lambda: core.run_cli(desiunknown_main), desiunknown_DOC),
    "zinc": (lambda: core.run_cli(zinc_main), zinc_DOC),
    "mined": (lambda: core.run_cli(mined_main), mined_DOC),
    "dibnear": (lambda: core.run_cli(dibnear_main), dibnear_DOC),
    "spinbreak": (lambda: core.run_cli(spinbreak_main), spinbreak_DOC),
    "lithium": (lithium_cli, lithium_DOC),
    "misfit": (lambda: core.run_cli(misfit_main), misfit_DOC),
    "cno": (lambda: core.run_cli(cno_main), cno_DOC),
    "fission": (lambda: core.run_cli(fission_main), fission_DOC),
    "elements": (elements_cli, elements_DOC),
    "borra": (borra_cli, borra_DOC),
    "gases": (gases_cli, gases_DOC),
}


