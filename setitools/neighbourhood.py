"""Nearby stars as a population: co-moving pairs, transit-zone entrants, imaging motion and polarisation."""
import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys

import numpy as np

from setitools import catalogues, core
from setitools.core import num, quiet_warnings


# --------------------------------------------------------------------------------------------------------------

imaging_DOC = r"""NaCo direct-imaging candidates moving as neither bound companions nor background stars (chi-square above 25 against both).

    python -I seti.py neighbourhood imaging naco
    python -I seti.py neighbourhood imaging record"""

imaging_HERE = core.RESULTS
imaging_CHI2 = 25.0


def imaging_parallax_factors(ra, dec, when):
    """Parallactic displacement (f_alpha*, f_delta) per mas at the given times."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    p = get_body_barycentric("earth", Time(when))
    x, y, z = p.x.to("au").value, p.y.to("au").value, p.z.to("au").value
    a, d = math.radians(ra), math.radians(dec)
    return (x * math.sin(a) - y * math.cos(a), x * math.cos(a) * math.sin(d) + y * math.sin(a) * math.sin(d)
            - z * math.cos(d))


def imaging_hosts(names):
    """SIMBAD position, proper motion and parallax per host."""
    out = {}
    for k in range(0, len(names), 100):
        chunk = names[k:k + 100]
        q = ("SELECT i.id AS name, b.ra, b.dec, b.pmra, b.pmdec, b.plx_value FROM ident AS i JOIN basic AS b ON "
             "b.oid = i.oidref WHERE i.id IN (%s)" % ", ".join("'%s'" % n.replace("'", "''") for n in chunk))
        for r in core.simbad(q):
            out[" ".join(r["name"].split())] = r
    return out


def imaging_naco():
    cats = core.vizier("J/A+A/509/A52")
    t8, t23 = cats["J/A+A/509/A52/table8"], cats["J/A+A/509/A52/table23"]
    simbad_name = {str(r["Target"]).strip(): str(r["SimbadName"]).strip() for r in t23}
    groups = {}
    for r in t8:
        key = (str(r["Target"]).strip(), str(r["Cand"]).strip())
        groups.setdefault(key, []).append(r)
    names = sorted({simbad_name.get(k[0], k[0]) for k in groups})
    info = imaging_hosts([" ".join(n.split()) for n in names])
    # hosts unknown by name taken as the nearest object within 10 arcsec with motion and parallax
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    for r in t23:
        key = " ".join(str(r["SimbadName"]).strip().split()) or str(r["Target"]).strip()
        if key in info:
            continue
        c = SkyCoord(str(r["RAJ2000"]), str(r["DEJ2000"]), unit=(u.hourangle, u.deg))
        rows = core.simbad("SELECT main_id AS name, ra, dec, pmra, pmdec, plx_value, DISTANCE(POINT('ICRS', ra, dec), "
                               "POINT('ICRS', %.6f, %.6f)) AS d FROM basic WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
                               "CIRCLE('ICRS', %.6f, %.6f, 0.0028)) AND pmra IS NOT NULL AND plx_value IS NOT NULL "
                               "ORDER BY d" % (c.ra.deg, c.dec.deg, c.ra.deg, c.dec.deg))
        if rows:
            info[key] = rows[0]
    simbad_name = {k: " ".join(v.split()) or k for k, v in simbad_name.items()}
    res = []
    for (target, cand), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: str(r["Obs.Date"]))
        if len(rows) < 2:
            continue
        h = info.get(" ".join(simbad_name.get(target, target).split()))
        if not h or h["pmra"] is None or h["plx_value"] is None:
            res.append(dict(target=target, cand=cand, epochs=len(rows), note="host motion unknown"))
            continue
        ra, dec, pmra, pmdec, plx = (float(h[k]) for k in ("ra", "dec", "pmra", "pmdec", "plx_value"))
        dates = [str(r["Obs.Date"]) for r in rows]
        sep = np.array([float(r["Sep"]) for r in rows]) * 1000.0
        pa = np.radians([float(r["PA"]) for r in rows])
        esep = np.array([float(r["e_Sep"]) for r in rows]) * 1000.0
        epa = np.radians([float(r["e_PA"]) for r in rows])
        x, y = sep * np.sin(pa), sep * np.cos(pa)
        # RA and Dec errors from separation and angle errors
        ex = np.hypot(esep * np.sin(pa), sep * epa * np.cos(pa))
        ey = np.hypot(esep * np.cos(pa), sep * epa * np.sin(pa))
        from astropy.time import Time
        yr = Time(dates).decimalyear
        fa, fd = imaging_parallax_factors(ra, dec, dates)
        fa, fd = np.atleast_1d(fa), np.atleast_1d(fd)
        # a background star moves by minus the host's proper motion and parallax
        bx = x[0] - pmra * (yr - yr[0]) - plx * (fa - fa[0])
        by = y[0] - pmdec * (yr - yr[0]) - plx * (fd - fd[0])
        vx, vy = ex ** 2 + ex[0] ** 2, ey ** 2 + ey[0] ** 2
        chi_co = float(np.sum((x[1:] - x[0]) ** 2 / vx[1:] + (y[1:] - y[0]) ** 2 / vy[1:]))
        chi_bg = float(np.sum((x[1:] - bx[1:]) ** 2 / vx[1:] + (y[1:] - by[1:]) ** 2 / vy[1:]))
        span = float(yr[-1] - yr[0])
        motion = math.hypot(x[-1] - x[0], y[-1] - y[0]) / max(span, 1e-3)
        res.append(dict(target=target, cand=cand, epochs=len(rows), span_yr=span, sep_mas=float(sep[0]),
                        chi2_comoving=chi_co, chi2_background=chi_bg, own_motion_mas_yr=motion,
                        host_pm=[pmra, pmdec], host_plx=plx, status=str(rows[-1]["Status"]).strip(),
                        dmag=float(rows[0]["Dmag"]) if rows[0]["Dmag"] is not np.ma.masked else None))
    json.dump(res, open(os.path.join(imaging_HERE, "imaging_motion_naco.json"), "w"), indent=1)
    tested = [r for r in res if "chi2_comoving" in r]
    odd = [r for r in tested if r["chi2_comoving"] > imaging_CHI2 and r["chi2_background"] > imaging_CHI2]
    print("%d candidates with two or more epochs, %d with host motions; %d depart from both tracks at chi2 above %.0f"
          % (len(res), len(tested), len(odd), imaging_CHI2))
    for r in sorted(odd, key=lambda r: -min(r["chi2_comoving"], r["chi2_background"])):
        print("  %s %s: %d epochs over %.1f yr at %.0f mas, chi2 co-moving %.0f, background %.0f; the offset moved %.0f "
              "mas/yr against the host's (%.0f, %.0f) mas/yr; survey status %s"
              % (r["target"], r["cand"], r["epochs"], r["span_yr"], r["sep_mas"], r["chi2_comoving"],
                 r["chi2_background"], r["own_motion_mas_yr"], r["host_pm"][0], r["host_pm"][1], r["status"] or "-"))


def imaging_record():
    res = json.load(open(os.path.join(imaging_HERE, "imaging_motion_naco.json")))
    tested = [r for r in res if "chi2_comoving" in r]
    odd = [r for r in tested if r["chi2_comoving"] > imaging_CHI2 and r["chi2_background"] > imaging_CHI2]
    core.record_search(
        "naco_candidate_motion", "Direct-imaging survey rejects", archive="VLT/NaCo survey of young austral stars "
        "(VizieR J/A+A/509/A52)", release="Chauvin et al. 2010",
        selection="candidates with two or more epochs; host motions from SIMBAD", code="seti.py neighbourhood imaging",
        params=dict(chi2=imaging_CHI2), n_targets=len(tested),
        null_summary="%d of %d candidates depart from both the co-moving and the background track" % (len(odd), len(tested)),
        sensitivity="a few mas per year of own motion over the epochs, the separation and angle errors",
        result="FS1035 cc-1 lies on the background track in Gaia DR3 (one NaCo epoch off by 0.17 arcsec), 0ES1847 "
               "cc-8 sits within an epoch error of it, and 0ES1847 cc-4 moves 200 mas/yr of its own, too faint for Gaia, "
               "as a field brown dwarf or late M dwarf would",
        ledger="Direct-imaging survey rejects")
    core.clear_targets("naco_candidate_motion")
    core.record_targets("naco_candidate_motion", [
        dict(target="simbad:%s" % r["target"], stat=float(min(r["chi2_comoving"], r["chi2_background"])),
             unit="smaller chi2 of the two tracks", note="%s, %d epochs over %.1f yr" % (r["cand"], r["epochs"],
                                                                                      r["span_yr"]))
        for r in tested])
    print("recorded")


def imaging_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("naco", "record"))
    a = p.parse_args()
    imaging_naco() if a.mode == "naco" else imaging_record()


def imaging_cli():
    quiet_warnings()
    imaging_main()


# --------------------------------------------------------------------------------------------------------------

phase_space_DOC = r"""Anomalies shared by co-moving pairs (El-Badry et al. 2021) across coverage runs: rank correlations and joint top-1-per-cent pairs against permutations.

    python -I seti.py neighbourhood phase_space ids
    python -I seti.py neighbourhood phase_space pairs ELBADRY.npz SHIFT.npz
    python -I seti.py neighbourhood phase_space test
    python -I seti.py neighbourhood phase_space record"""

phase_space_ROOT = os.environ.get("PHASE_ROOT", core.data("phase_space"))
phase_space_HERE = core.RESULTS
phase_space_TOP = 0.99
phase_space_PERMS = 2000
# statistics that do not measure anomaly, and runs on bodies
phase_space_SKIP = {"radar_beam_stars", "reply_window_etz_bl", "pulsar_ir_allwise", "enclosed_stars", "dark_companions_wise",
        "pma_dark_companions", "sgl_parallax_racs", "sgl_parallax_racs_vlass", "pulsar_antipodes"}
# runs anomalous in both directions
phase_space_TWO_SIDED = {"hip_gaia_dimming", "neowise_decade_trends", "dasch_century_dimming", "dasch_hz_trends", "hz_stack_wise"}
# arcsec within which a run measures both members together
phase_space_BLEND_ARCSEC = {"hi4pi_star_frames": 972.0, "icecube_near10": 3600.0, "bl_rest_frame_tracks": 540.0,
                "highenergy_near20": 30.0, "hip_gaia_dimming": 38.0, "hipparcos_dippers": 38.0, "cold_shells_akari": 90.0,
                "spt3g_quiet_flares": 72.0, "tess20_glints": 60.0, "dasch_century_dimming": 20.0, "dasch_hz_trends": 20.0,
                "hz_stack_wise": 12.0, "neowise_decade_trends": 6.0, "spherex_contardo53": 12.0,
                "polarised_reflectors": 10.0, "vlass_quiet_radio_stars": 3.0, "laser_desi_dr1": 3.0, "laser_galah": 3.0,
                "laser_gaia_rvs": 3.0, "laser_harps": 3.0, "laser_lamost_mrs": 3.0, "gaia_dr3_dippers": 2.0,
                "gaia_alerts_near": 2.0}


def phase_space_ids():
    """{coverage target: Gaia DR3 source_id} for gaia_dr3, hip and simbad targets."""
    db = sqlite3.connect(core.COVERAGE)
    targets = sorted({t for (t,) in db.execute("SELECT DISTINCT target FROM coverage")})
    out = {t: t.split(":", 1)[1] for t in targets if t.startswith("gaia_dr3:")}
    hips = {t.split(":", 1)[1]: t for t in targets if t.startswith("hip:")}
    if hips:
        rows = core.gaia("SELECT original_ext_source_id, source_id FROM gaiadr3.hipparcos2_best_neighbour")
        for r in rows:
            h = str(r["original_ext_source_id"]).strip()
            if h in hips:
                out[hips[h]] = str(r["source_id"])
    names = {t.split(":", 1)[1]: t for t in targets if t.startswith("simbad:")}
    todo = sorted(names)
    for k in range(0, len(todo), 200):
        chunk = todo[k:k + 200]
        q = ("SELECT i1.id AS name, i2.id AS gaia FROM ident AS i1 JOIN ident AS i2 ON i1.oidref = i2.oidref "
             "WHERE i2.id LIKE 'Gaia DR3 %%' AND i1.id IN (%s)" % ", ".join("'%s'" % n.replace("'", "''") for n in chunk))
        norm = {" ".join(n.split()): names[n] for n in chunk}
        for r in core.simbad(q):
            t = norm.get(" ".join(r["name"].split()))
            if t:
                out[t] = r["gaia"].split()[-1]
    os.makedirs(phase_space_ROOT, exist_ok=True)
    json.dump(out, open(os.path.join(phase_space_ROOT, "ids.json"), "w"))
    print("%d targets: %d mapped to Gaia DR3 (%d gaia_dr3, %d of %d hip, %d of %d simbad), %d distinct stars"
          % (len(targets), len(out), sum(1 for t in out if t.startswith("gaia_dr3:")),
             sum(1 for t in out if t.startswith("hip:")), len(hips), sum(1 for t in out if t.startswith("simbad:")),
             len(names), len(set(out.values()))))


def phase_space_pairs(main_npz, shift_npz):
    """Pairs with both members among the mapped stars."""
    mapped = set(int(v) for v in json.load(open(os.path.join(phase_space_ROOT, "ids.json"))).values())
    keep = {}
    for label, path in (("main", main_npz), ("shift", shift_npz)):
        z = np.load(path)
        a, b = z["source_id1"].astype(np.int64), z["source_id2"].astype(np.int64)
        both = np.array([x in mapped and y in mapped for x, y in zip(a.tolist(), b.tolist())])
        r = z["R_chance_align"] if "R_chance_align" in z.files else np.full(len(a), np.nan)
        sep = z["sep_AU"]
        ang = z["pairdistance"] * 3600.0
        rows = [dict(a=str(x), b=str(y), sep_au=float(s), sep_arcsec=float(t), r=float(q)) for x, y, s, t, q in
                zip(a[both].tolist(), b[both].tolist(), sep[both].tolist(), ang[both].tolist(), r[both].tolist())]
        if label == "main":
            keep["bound"] = [x for x in rows if x["r"] < 0.1]
            keep["chance"] = [x for x in rows if x["r"] > 0.5]
        else:
            keep["shift"] = rows
        print("%s: %d pairs, %d with both members searched" % (label, len(a), len(rows)))
    json.dump(keep, open(os.path.join(phase_space_ROOT, "pairs.json"), "w"))
    print("bound %d, close chance alignments %d, shifted %d" % (len(keep["bound"]), len(keep["chance"]),
                                                                 len(keep["shift"])))


def phase_space_scores():
    """{run: {gaia id: percentile}}, 1 the most anomalous."""
    idmap = json.load(open(os.path.join(phase_space_ROOT, "ids.json")))
    db = sqlite3.connect(core.COVERAGE)
    raw = {}
    for run, tgt, stat, p in db.execute("SELECT search_id, target, stat, p FROM coverage"):
        if run in phase_space_SKIP or tgt not in idmap:
            continue
        if p is not None and p > 0:
            v = -np.log10(p)
        elif stat is not None:
            v = abs(stat) if run in phase_space_TWO_SIDED else stat
        else:
            continue
        g = idmap[tgt]
        d = raw.setdefault(run, {})
        d[g] = max(d.get(g, -np.inf), v)
    out = {}
    for run, d in raw.items():
        if len(d) < 20:
            continue
        g = list(d)
        v = np.array([d[x] for x in g])
        order = v.argsort(kind="stable")
        ranks = np.empty(len(v))
        ranks[order] = np.arange(1, len(v) + 1)
        # ties share their mean rank
        for val in np.unique(v):
            sel = v == val
            if sel.sum() > 1:
                ranks[sel] = ranks[sel].mean()
        out[run] = dict(zip(g, ranks / len(v)))
    return out


class phase_space_Layout:
    """All runs' percentiles in one array with same-run, cross-run and member index arrays for a pair set."""

    def __init__(self, sc, prs):
        self.runs = sorted(sc)
        self.flat, self.seg, where = [], [], {}
        for run in self.runs:
            start = len(self.flat)
            for g, u in sc[run].items():
                where.setdefault(g, []).append((run, len(self.flat)))
                self.flat.append(u)
            self.seg.append((start, len(self.flat)))
        self.flat = np.array(self.flat)
        sa, sb, ca, cb, kk, ka, kb = [], [], [], [], [], [], []
        self.blended = 0
        for k, pr in enumerate(prs):
            ea, eb = dict(where.get(pr["a"], [])), dict(where.get(pr["b"], []))
            for x, ix in ea.items():
                for y, iy in eb.items():
                    if x != y:
                        ca.append(ix)
                        cb.append(iy)
                    elif pr.get("sep_arcsec", np.inf) < phase_space_BLEND_ARCSEC.get(x, 0.0):
                        self.blended += 1
                        continue
                    else:
                        sa.append(ix)
                        sb.append(iy)
                    kk.append(k)
                    ka.append(ix)
                    kb.append(iy)
        self.n = len(prs)
        self.sa, self.sb, self.ca, self.cb, self.kk, self.ka, self.kb = (np.array(v, int) for v in
                                                                         (sa, sb, ca, cb, kk, ka, kb))

    def permuted(self, rng):
        out = self.flat.copy()
        for s, e in self.seg:
            out[s:e] = rng.permutation(out[s:e])
        return out

    def statistics(self, u):
        rho = lambda a, b: float(np.corrcoef(u[a], u[b])[0, 1]) if len(a) > 10 else float("nan")  # noqa: E731
        both = (u[self.ka] >= phase_space_TOP) & (u[self.kb] >= phase_space_TOP) if len(self.kk) else np.zeros(0, bool)
        return dict(n_same=len(self.sa), rho_same=rho(self.sa, self.sb), n_cross=len(self.ca),
                    rho_cross=rho(self.ca, self.cb), top_both=int(len(np.unique(self.kk[both]))),
                    blended=self.blended)


def phase_space_test():
    sc = phase_space_scores()
    prs = json.load(open(os.path.join(phase_space_ROOT, "pairs.json")))
    print("%d runs with ranked stars; %d stars ranked" % (len(sc), len({g for d in sc.values() for g in d})))
    rng = np.random.default_rng(55)
    res = {}
    for label in ("bound", "chance", "shift"):
        lay = phase_space_Layout(sc, prs[label])
        obs = lay.statistics(lay.flat)
        null = [lay.statistics(lay.permuted(rng)) for _ in range(phase_space_PERMS)]
        out = dict(obs, pairs=len(prs[label]))
        for key in ("rho_same", "rho_cross", "top_both"):
            vals = np.array([n[key] for n in null], float)
            vals = vals[np.isfinite(vals)]
            if np.isfinite(obs[key]) and len(vals):
                out[key + "_null_mean"] = float(vals.mean())
                out[key + "_null_sd"] = float(vals.std())
                out[key + "_p"] = float((np.sum(vals >= obs[key]) + 1) / (len(vals) + 1))
        res[label] = out
        print("%s pairs %d (%d same-run pairings inside a run's blend radius set aside): same run %d pairings, rho %.3f "
              "(null %.3f +- %.3f, p %.3g); other runs %d pairings, rho %.3f (null %.3f +- %.3f, p %.3g); both members "
              "in a top 1%%: %d (null %.1f +- %.1f, p %.3g)"
              % (label, len(prs[label]), out["blended"], out["n_same"], out["rho_same"], out.get("rho_same_null_mean", np.nan),
                 out.get("rho_same_null_sd", np.nan), out.get("rho_same_p", np.nan), out["n_cross"], out["rho_cross"],
                 out.get("rho_cross_null_mean", np.nan), out.get("rho_cross_null_sd", np.nan),
                 out.get("rho_cross_p", np.nan), out["top_both"], out.get("top_both_null_mean", np.nan),
                 out.get("top_both_null_sd", np.nan), out.get("top_both_p", np.nan)), flush=True)
    # bound pairs with both members in a run's top 1 per cent
    star_runs = {}
    for run, d in sc.items():
        for g, u in d.items():
            if u >= phase_space_TOP:
                star_runs.setdefault(g, []).append(run)
    res["top_pairs"] = []
    for pr in prs["bound"]:
        ok = [(x, y) for x in star_runs.get(pr["a"], []) for y in star_runs.get(pr["b"], [])
              if x != y or pr.get("sep_arcsec", np.inf) >= phase_space_BLEND_ARCSEC.get(x, 0.0)]
        if ok:
            res["top_pairs"].append(dict(pr, runs_a=sorted({x for x, y in ok}), runs_b=sorted({y for x, y in ok})))
    for pr in res["top_pairs"]:
        print("  Gaia DR3 %s (%s) and %s (%s), %.0f au" % (pr["a"], ", ".join(pr["runs_a"]), pr["b"],
                                                        ", ".join(pr["runs_b"]), pr["sep_au"]))
    json.dump(res, open(os.path.join(phase_space_HERE, "phase_space.json"), "w"), indent=1)


def phase_space_record():
    r = json.load(open(os.path.join(phase_space_HERE, "phase_space.json")))
    b = r["bound"]
    core.record_search(
        "phase_space_pairs", "Anomalies clustered in phase space",
        archive="the coverage store joined to El-Badry, Rix and Heintz (2021) wide binaries (Zenodo 4435257)",
        release="Gaia EDR3 pairs", selection="bound pairs (R below 0.1) with both members among the searched stars",
        code="seti.py neighbourhood phase_space", params=dict(top=phase_space_TOP, permutations=phase_space_PERMS, bound_r=0.1, chance_r=0.5),
        n_targets=b["pairs"],
        null_summary="same run: %d pairings, rho %.3f against %.3f +- %.3f; other runs: %d pairings, rho %.3f against "
                     "%.3f +- %.3f; both in a top 1 per cent: %d against %.1f +- %.1f"
                     % (b["n_same"], b["rho_same"], b.get("rho_same_null_mean", float("nan")),
                        b.get("rho_same_null_sd", float("nan")), b["n_cross"], b["rho_cross"],
                        b.get("rho_cross_null_mean", float("nan")), b.get("rho_cross_null_sd", float("nan")),
                        b["top_both"], b.get("top_both_null_mean", float("nan")), b.get("top_both_null_sd", float("nan"))),
        sensitivity="a shared anomaly in the top 1 per cent of any run among the bound pairs",
        result="no pair shares an anomaly beyond the permutations; the resolved same-run correlation is the likeness of "
               "co-natal stars, and across unrelated searches the pairs show none",
        ledger="Anomalies clustered in phase space")
    core.clear_targets("phase_space_pairs")
    core.record_targets("phase_space_pairs", [
        dict(target="gaia_dr3:%s" % x, note="bound partner of Gaia DR3 %s at %.0f au; top 1%% in %s"
                                            % (y, pr["sep_au"], ", ".join(runs)))
        for pr in r["top_pairs"] for x, y, runs in ((pr["a"], pr["b"], pr["runs_a"]), (pr["b"], pr["a"], pr["runs_b"]))])
    print("recorded")


def phase_space_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("ids", "pairs", "test", "record"))
    p.add_argument("paths", nargs="*")
    a = p.parse_args()
    if a.mode == "ids":
        phase_space_ids()
    elif a.mode == "pairs":
        phase_space_pairs(*a.paths)
    elif a.mode == "test":
        phase_space_test()
    else:
        phase_space_record()


def phase_space_cli():
    quiet_warnings()
    phase_space_main()


# --------------------------------------------------------------------------------------------------------------

etz_DOC = r"""Fresh Earth-transit-zone entrants within 100 pc from proper motion, joined to the coverage store and the BL target list.

    python -I seti.py neighbourhood etz [record] [--since 1000]"""

etz_HERE = core.RESULTS
etz_HALF = 0.2665  # deg, solar angular radius at 1 AU
etz_EPS = math.radians(23.4392911)  # J2000 obliquity


def etz_pm_beta(ra, dec, pmra, pmdec):
    """Proper motion in ecliptic latitude (mas/yr)."""
    a, d = math.radians(ra), math.radians(dec)
    # sin b = sin d cos e - cos d sin e sin a
    sinb = math.sin(d) * math.cos(etz_EPS) - math.cos(d) * math.sin(etz_EPS) * math.sin(a)
    cosb = math.sqrt(max(1 - sinb * sinb, 1e-12))
    dsinb_dd = math.cos(d) * math.cos(etz_EPS) + math.sin(d) * math.sin(etz_EPS) * math.sin(a)
    dsinb_da = -math.cos(d) * math.sin(etz_EPS) * math.cos(a)
    # pmra includes cos d
    mu_a = pmra / max(math.cos(d), 1e-9)
    return (dsinb_dd * pmdec + dsinb_da * mu_a) / cosb


def etz_main(record=False, since=1000.0):
    rows = core.gaia("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp, ecl_lat "
                         "FROM gaiadr3.gaia_source WHERE parallax > 10 AND parallax_over_error > 10 AND "
                         "ABS(ecl_lat) < 5 AND pmra IS NOT NULL")
    out = []
    for r in rows:
        b0, mub = float(r["ecl_lat"]), etz_pm_beta(float(r["ra"]), float(r["dec"]), float(r["pmra"]), float(r["pmdec"]))
        rate = mub / 3.6e6  # deg/yr
        if abs(rate) < 1e-12:
            continue
        # years from 2016.0 when b crosses +-HALF
        t1, t2 = (etz_HALF - b0) / rate, (-etz_HALF - b0) / rate
        t_in, t_out = min(t1, t2), max(t1, t2)
        out.append(dict(source_id=r["source_id"], d_pc=1000.0 / float(r["parallax"]), g=float(r["phot_g_mean_mag"] or 99),
                        bp_rp=float(r["bp_rp"] or 0), ecl_lat=b0, mu_beta=mub, entry=2016.0 + t_in, exit=2016.0 + t_out))
    inside = [x for x in out if x["entry"] <= 2016.0 <= x["exit"]]
    fresh = sorted([x for x in inside if x["entry"] >= 2016.0 - since], key=lambda x: -x["entry"])
    soon = sorted([x for x in out if 2016.0 < x["entry"] <= 2116.0], key=lambda x: x["entry"])
    print("%d Gaia DR3 stars within 100 pc and 5 deg of the ecliptic; %d see Earth transit now, %d of them entered after "
          "%d; %d enter within the next century" % (len(out), len(inside), len(fresh), 2016 - since, len(soon)))
    import sqlite3
    db = sqlite3.connect(core.COVERAGE)
    bl = {b.replace(" ", "").upper() for b in json.load(open(os.path.join(etz_HERE, "bl_targets.json")))}
    ids = [x["source_id"] for x in fresh[:300] + soon]
    hip = {}
    for k in range(0, len(ids), 400):
        q = ("SELECT source_id, original_ext_source_id FROM gaiadr3.hipparcos2_best_neighbour WHERE source_id IN (%s)"
             % ",".join(ids[k:k + 400]))
        for r in core.gaia(q):
            hip[r["source_id"]] = "HIP" + r["original_ext_source_id"]
    for label, group in (("entered", fresh[:25]), ("entering", soon[:25])):
        print("%s:" % label)
        for x in group:
            h = hip.get(x["source_id"])
            card = db.execute("SELECT COUNT(*) FROM coverage WHERE target IN (?, ?)",
                              ("gaia_dr3:" + x["source_id"], (h or "").replace("HIP", "hip:"))).fetchone()[0]
            print("  Gaia DR3 %-20s %5.1f pc G %5.2f BP-RP %4.2f: in the zone %7.0f to %7.0f%s%s; coverage rows %d"
                  % (x["source_id"], x["d_pc"], x["g"], x["bp_rp"], x["entry"], x["exit"], "; " + h if h else "",
                     " (BL)" if h and h.upper() in bl else "", card))
    json.dump(dict(fresh=fresh, soon=soon, hip=hip), open(os.path.join(etz_HERE, "etz_entrants.json"), "w"), indent=1)
    if record:
        covered = sum(1 for x in fresh + soon if db.execute("SELECT COUNT(*) FROM coverage WHERE target = ?", ("gaia_dr3:" + x["source_id"],)).fetchone()[0])
        core.record_run(
            "etz_fresh_entrants", "Fresh transit-zone entrants", "Gaia DR3; Hipparcos best neighbours; the coverage store; the Breakthrough Listen target list",
            "Gaia DR3 stars within 100 pc and 5 deg of the ecliptic: %d" % len(out), os.path.abspath(__file__), "etz", len(out), "stars", dict(since=since),
            "the stars' proper motion in ecliptic latitude", "stars that began to see Earth transit the Sun since %d or will within a century" % (2016 - since),
            "%d see Earth transit now, %d of them having entered since %d; %d enter within the next century; %d of the %d entered or entering carry rows in the store" % (
                len(inside), len(fresh), 2016 - since, len(soon), covered, len(fresh) + len(soon)), [])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------

polarised_DOC = r"""Polarised reflectors within 50 pc: linear polarisation 3 sigma above the distance-scaled neighbour median, filtered for dust, activity, companions and type.

    python -I seti.py neighbourhood polarised fetch
    python -I seti.py neighbourhood polarised analyse
    python -I seti.py neighbourhood polarised record"""

polarised_ROOT = os.environ.get("POLAR_ROOT", core.data("polar"))
polarised_HERE = core.RESULTS
polarised_PIIROLA = (("J/A+A/635/A46/table6", "Piirola UH88"), ("J/A+A/635/A46/table7", "Piirola T60"),
           ("J/A+A/635/A46/table8", "Piirola H127"))
polarised_PLANETPOL = "J/MNRAS/405/2570/pol"
polarised_HEILES = "II/226/catalog"
polarised_DEBRIS = ("J/ApJS/211/25/catalog", "J/A+A/555/A11/stars")
polarised_ACTIVITY = "J/A+A/616/A108/catalog"
polarised_ACTIVE_TYPES = {"BY*", "RS*", "Fl*", "Ro*", "Er*", "TT*", "Or*", "Y*O", "Y*?", "TT?"}
polarised_BINARY_TYPES = {"SB*", "EB*", "El*", "EB?", "SB?"}
polarised_DUSTY_TYPES = {"LP*", "Mi*", "C*", "S*", "AB*", "Be*", "Ae*", "Pe*", "Em*", "WR*", "a2*", "sg*", "s*b", "s*r", "s*y"}
polarised_MERGE_ARCSEC = 30.0
# interstellar bound 2.9 ppm/pc within 50 pc (Piirola et al. 2020)
polarised_ISM_BOUND = 2.9
# Local Leo Cold Cloud, 11-24 pc (Meyer 2006, Peek 2011)
polarised_LLCC = dict(l=(210.0, 232.0), b=(36.0, 50.0), d_min=11.0)
# Heiles catalogues reading above 0.03 per cent median carry offsets
polarised_REF_HIGH = 3.0e2
# night-to-night q, u at T60 (Piirola et al. 2020 Table 11, ppm)
polarised_VARIABLE = {"16673": [(-21, 31, 9), (-8, -6, 8)], "19373": [(-9, 5, 4), (0, 7, 7), (-3, 12, 9), (-10, 44, 8)],
            "34411": [(0, -3, 6), (11, 13, 12), (-17, 14, 5), (0, -19, 6), (-10, -3, 6)],
            "34721": [(22, 27, 7), (-9, -7, 8)],
            "43042": [(-3, -6, 7), (12, 16, 9), (11, 8, 6), (30, 19, 6), (18, 11, 7)],
            "43386": [(0, 0, 6), (13, 13, 7), (8, -27, 7), (0, 15, 7), (20, 0, 7)],
            "55575": [(-6, 4, 8), (39, -16, 9), (18, 17, 7), (5, 13, 7)],
            "58855": [(-10, -1, 5), (-10, 40, 5), (-31, 6, 7), (-29, -5, 7), (-9, -11, 10)],
            "59984": [(-16, -11, 11), (21, 12, 10), (37, 37, 7)], "83287": [(-8, -3, 6), (-14, -36, 6)],
            "89449": [(0, -9, 8), (2, -30, 6), (-11, -7, 8)], "115383": [(0, -19, 9), (18, 24, 8)],
            "164259": [(-3, 14, 7), (28, 19, 13)], "187691": [(-8, 0, 6), (7, 32, 8)],
            "205289": [(5, 20, 8), (6, -23, 8), (-36, 4, 11)], "206826": [(-6, -3, 5), (-29, -5, 7), (-16, -25, 7)],
            "207958": [(1, 16, 7), (-5, -14, 8), (-27, -15, 7), (-23, 10, 9), (-9, -8, 8), (12, 0, 6)],
            "215648": [(-10, -5, 5), (-18, -9, 8), (-28, -7, 8), (-3, -24, 7), (7, -16, 9), (7, 2, 6)]}
polarised_T60_STARS = 205
# within this angle of the neighbours' PA the polarisation follows the local field
polarised_PA_ALIGNED = 15.0


def polarised_tap_all(table, where=""):
    return core.tap("vizier", 'SELECT * FROM "%s"%s' % (table, (" WHERE " + where) if where else ""))


def polarised_measurements():
    out = []
    for table, src in polarised_PIIROLA:
        for r in polarised_tap_all(table):
            plx = num(r["plx"])
            out.append(dict(src=src, name="HD " + r["HD"].strip(), ra=num(r["RAJ2000"]), dec=num(r["DEJ2000"]),
                            p=num(r["p"]), e=num(r["e_p"]), pa=num(r["theta"]), v=num(r["Vmag"]),
                            d=1000.0 / plx if plx else None, sp=r["SpType"].strip()))
    for r in polarised_tap_all(polarised_PLANETPOL):
        out.append(dict(src="PlanetPol", name="HR " + r["HR"].strip(), ra=num(r["_RA"]), dec=num(r["_DE"]),
                        p=num(r["Pol"]), e=num(r["e_Pol"]), pa=num(r["theta"]), v=num(r["Vmag"]), d=num(r["Dist"]),
                        sp=r["SpType"].strip()))
    for r in polarised_tap_all(polarised_HEILES, "Dist > 0 AND Dist <= 50"):
        if num(r["Pol"]) is None or not num(r["e_Pol"]):
            continue
        name = ("HD " + str(int(num(r["HD"])))) if num(r["HD"]) else ("BD " + r["BD"]).strip()
        out.append(dict(src="Heiles", name=name, ra=num(r["RAJ2000"]), dec=num(r["DEJ2000"]), p=num(r["Pol"]) * 1e4,
                        e=num(r["e_Pol"]) * 1e4, pa=num(r["PA"]), v=num(r["Vmag"]), d=num(r["Dist"]),
                        sp=r["SpType"].strip(), ref=r["PolRefs"].strip()))
    return [m for m in out if m["ra"] is not None and m["p"] is not None and m["e"]]


def polarised_sep_deg(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def polarised_merge(ms):
    """Group one star's measurements within MERGE_ARCSEC and 1.5 mag in V."""
    stars = []
    for m in sorted(ms, key=lambda m: m["e"]):
        for s in stars:
            if polarised_sep_deg(s["ra"], s["dec"], m["ra"], m["dec"]) * 3600 < polarised_MERGE_ARCSEC and (
                    s["v"] is None or m["v"] is None or abs(s["v"] - m["v"]) < 1.5):
                s["meas"].append(m)
                break
        else:
            stars.append(dict(id="s%04d" % len(stars), ra=m["ra"], dec=m["dec"], v=m["v"], meas=[m]))
    return stars


def polarised_nearest(rows, key):
    polarised_best = {}
    for r in rows:
        if r["source_id"] not in polarised_best or key(r) < key(polarised_best[r["source_id"]]):
            polarised_best[r["source_id"]] = r
    return polarised_best


def polarised_fetch():
    os.makedirs(polarised_ROOT, exist_ok=True)
    stars = polarised_merge(polarised_measurements())
    pos = [dict(source_id=s["id"], ra=s["ra"], dec=s["dec"]) for s in stars]
    sim = polarised_nearest(core.xmatch(pos, "simbad", 15.0), lambda r: float(r["angDist"]))
    gaia = polarised_nearest(core.xmatch(pos, "I/355/gaiadr3", 20.0), lambda r: num(r["Gmag"]) or 99.0)
    wise = polarised_nearest(core.xmatch(pos, "II/328/allwise", 15.0), lambda r: num(r["W1mag"]) or 99.0)
    act = polarised_nearest(core.xmatch(pos, polarised_ACTIVITY, 15.0), lambda r: float(r["angDist"]))
    discs = [(num(r["_RA"]), num(r["_DE"])) for r in polarised_tap_all(polarised_DEBRIS[0])]
    discs += [(num(r["RAJ2000"]), num(r["DEJ2000"])) for r in polarised_tap_all(polarised_DEBRIS[1], "Ex = 'e'")]
    debris = {s["id"] for s in stars for ra, de in discs if ra is not None and polarised_sep_deg(s["ra"], s["dec"], ra, de) * 3600 < 30}
    planets = core.tap("exoplanets", "SELECT hostname, ra, dec, pl_name, pl_orbsmax, pl_bmassj FROM pscomppars")
    for s in stars:
        i = s["id"]
        s["simbad"] = {k: sim[i].get(k) for k in ("main_id", "main_type", "other_types", "sp_type", "plx", "V")} if i in sim else None
        s["gaia"] = {k: gaia[i].get(k) for k in ("Source", "Plx", "e_Plx", "RUWE", "Gmag", "BP-RP", "RV", "Vbroad")} if i in gaia else None
        s["wise"] = {k: wise[i].get(k) for k in ("W1mag", "W2mag", "W3mag", "W4mag", "e_W3mag", "e_W4mag", "Kmag", "e_Kmag",
                                                 "ccf", "qph", "ex")} if i in wise else None
        s["logrhk"] = num(act[i]["logRpHK"]) if i in act else None
        s["debris_listed"] = i in debris
        s["planets"] = [dict(name=p["pl_name"], a=num(p["pl_orbsmax"]), m=num(p["pl_bmassj"])) for p in planets
                        if num(p["ra"]) is not None and polarised_sep_deg(s["ra"], s["dec"], float(p["ra"]), float(p["dec"])) * 3600 < 30]
    json.dump(stars, open(os.path.join(polarised_ROOT, "stars.json"), "w"), indent=1)
    print("%d measurements of %d stars; SIMBAD %d, Gaia %d, AllWISE %d, log R'HK %d, listed debris discs %d"
          % (sum(len(s["meas"]) for s in stars), len(stars), len(sim), len(gaia), len(wise), len(act), len(debris)))


def polarised_distance(s):
    g = s.get("gaia") or {}
    if num(g.get("Plx")) and num(g["Plx"]) > 0:
        return 1000.0 / num(g["Plx"])
    sm = s.get("simbad") or {}
    if num(sm.get("plx")) and num(sm["plx"]) > 0:
        return 1000.0 / num(sm["plx"])
    ds = [m["d"] for m in s["meas"] if m["d"]]
    return ds[0] if ds else None


def polarised_spectral_type(s):
    sm = s.get("simbad") or {}
    return (sm.get("sp_type") or s["meas"][0]["sp"] or "").strip()


def polarised_evolved(sp):
    m = re.match(r"\s*[OBAFGKMSC][\d.]*\s*(?:[-/][OBAFGKM]?[\d.]+)?\s*((?:[IV]+)(?:[ab]+)?(?:[-/][IV]+)?)", sp)
    return bool(m) and re.match(r"[IV]+", m.group(1)).group(0) in ("I", "II", "III")


def polarised_photosphere_kw4(sp):
    return {"F": 0.05, "G": 0.06, "K": 0.12, "M": 0.30}.get(sp[:1], 0.05)


def polarised_ir_excess(s, sp):
    w = s.get("wise")
    if not w:
        return None
    q, c = str(w.get("qph") or "UUUU"), str(w.get("ccf") or "0000")
    w3, w4, e3, e4 = (num(w.get(k)) for k in ("W3mag", "W4mag", "e_W3mag", "e_W4mag"))
    k, ek = num(w.get("Kmag")), num(w.get("e_Kmag"))
    if w4 is None or q[3] not in "AB" or c[3] != "0":
        return None
    if w3 is not None and e3 and e4 and q[2] in "AB" and c[2] == "0":
        x = w3 - w4
        if x > 0.1 and x > 3 * math.hypot(e3, e4):
            return "W3-W4 %.2f" % x
    if k is not None and ek and e4:
        x = k - w4 - polarised_photosphere_kw4(sp)
        if x > 0.1 and x > 3 * math.hypot(ek, e4):
            return "Ks-W4 %.2f above the photosphere" % x
    return None


def polarised_explanations(s, high_refs=()):
    sp = polarised_spectral_type(s)
    sm = s.get("simbad") or {}
    types = set((sm.get("other_types") or "").split("|")) | {sm.get("otype") or ""}
    why = []
    b = polarised_best(s)
    if b.get("ref") in high_refs:
        why.append("source catalogue reads high (Heiles pattern %s)" % b["ref"])
    if s.get("d") and s["p"] - 3 * s["e"] <= polarised_ISM_BOUND * s["d"]:
        why.append("within the interstellar bound (%.1f ppm/pc)" % polarised_ISM_BOUND)
    if s.get("l") is not None and polarised_LLCC["l"][0] <= s["l"] <= polarised_LLCC["l"][1] and polarised_LLCC["b"][0] <= s["b"] <= polarised_LLCC["b"][1] \
            and s["d"] > polarised_LLCC["d_min"]:
        why.append("behind the Local Leo Cold Cloud")
    if "+" in sp:
        why.append("companion (composite spectrum %s)" % sp)
    ir = polarised_ir_excess(s, sp)
    if s["debris_listed"]:
        why.append("listed debris disc")
    if ir:
        why.append("infrared excess (%s)" % ir)
    if s["logrhk"] is not None and s["logrhk"] > -4.75:
        why.append("active (log R'HK %.2f)" % s["logrhk"])
    if types & polarised_ACTIVE_TYPES:
        why.append("active (%s)" % ",".join(sorted(types & polarised_ACTIVE_TYPES)))
    ruwe = num((s.get("gaia") or {}).get("RUWE"))
    if ruwe and ruwe > 1.4:
        why.append("companion (RUWE %.1f)" % ruwe)
    if types & polarised_BINARY_TYPES:
        why.append("companion (%s)" % ",".join(sorted(types & polarised_BINARY_TYPES)))
    if polarised_evolved(sp):
        why.append("evolved (%s)" % sp)
    if types & polarised_DUSTY_TYPES:
        why.append("emission, peculiar or dusty type (%s)" % ",".join(sorted(types & polarised_DUSTY_TYPES)))
    if sp[:1] in "OBA":
        why.append("early type (%s)" % sp)
    hj = [p for p in s["planets"] if p["a"] is not None and p["a"] < 0.1]
    if hj:
        why.append("close planet (%s)" % ", ".join(p["name"] for p in hj))
    return why


def polarised_debiased(p, e):
    return math.sqrt(max(p * p - e * e, 0.0))


def polarised_best(s):
    return min(s["meas"], key=lambda m: m["e"])


def polarised_high_references(stars):
    """Heiles source patterns reading high within 50 pc."""
    by = {}
    for s in stars:
        for m in s["meas"]:
            if m.get("ref") and m["d"] and m["d"] <= 50.0:
                by.setdefault(m["ref"], []).append((m["p"], m["e"]))
    out = {}
    for ref, v in by.items():
        if len(v) >= 8:
            mp, me = float(np.median([x[0] for x in v])), float(np.median([x[1] for x in v]))
            if mp > polarised_REF_HIGH and mp > 2 * me:
                out[ref] = (len(v), mp, me)
    return out


def polarised_galactic(ra, dec):
    from astropy.coordinates import SkyCoord
    g = SkyCoord(ra=ra, dec=dec, unit="deg").galactic
    return float(g.l.deg), float(g.b.deg)


def polarised_variability(stars, high_refs):
    """Chi-square of nightly q and u (Piirola Table 11) with the 205-star trial factor."""
    from scipy.stats import chi2
    by = {}
    for s in stars:
        for m in s["meas"]:
            if m["name"].startswith("HD "):
                by[m["name"][3:].strip()] = s
    out = []
    for hd, ms in polarised_VARIABLE.items():
        c = 0.0
        for k in (0, 1):
            w = np.array([1.0 / m[2] ** 2 for m in ms])
            x = np.array([m[k] for m in ms], float)
            c += float(np.sum(w * (x - np.sum(w * x) / np.sum(w)) ** 2))
        dof = 2 * (len(ms) - 1)
        p = float(chi2.sf(c, dof))
        s = by.get(hd)
        why = [w for w in polarised_explanations(s, high_refs) if not w.startswith("within the interstellar")] if s else []
        amp = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in ms for b in ms)
        out.append(dict(hd=hd, nights=len(ms), chi2=c, dof=dof, p=p, p_trials=min(1.0, p * polarised_T60_STARS), swing_ppm=amp,
                        why=why, d=s.get("d") if s else None, sp=polarised_spectral_type(s) if s else ""))
    return out


def polarised_analyse():
    stars = json.load(open(os.path.join(polarised_ROOT, "stars.json")))
    for s in stars:
        s["d"] = polarised_distance(s)
        b = polarised_best(s)
        s["p"], s["e"], s["pa"], s["src"] = b["p"], b["e"], b["pa"], b["src"]
        s["pdb"] = polarised_debiased(b["p"], b["e"])
        s["l"], s["b"] = polarised_galactic(s["ra"], s["dec"])
        s["old_single"] = b["src"] == "Heiles" and not any(m["src"] != "Heiles" for m in s["meas"])
    high_refs = polarised_high_references(stars)
    print("Heiles source patterns reading high within 50 pc: " + "; ".join(
        "%s (%d stars, median %.0f ppm, error %.0f)" % (r, *v) for r, v in high_refs.items()))
    stars = [s for s in stars if s["d"] and s["d"] <= 50.0]
    ref = [s for s in stars if s["src"] != "Heiles" and s["e"] <= 30]
    ra = np.radians([s["ra"] for s in ref])
    de = np.radians([s["dec"] for s in ref])
    for s in stars:
        a, d = math.radians(s["ra"]), math.radians(s["dec"])
        cs = np.sin(d) * np.sin(de) + np.cos(d) * np.cos(de) * np.cos(ra - a)
        sep = np.degrees(np.arccos(np.clip(cs, -1, 1)))
        for radius in (35.0, 45.0, 60.0):
            nb = [j for j in np.where(sep <= radius)[0] if ref[j]["id"] != s["id"]]
            if len(nb) >= 5:
                break
        v = np.array([ref[j]["pdb"] * min(max(s["d"] / ref[j]["d"], 0.2), 5.0) for j in nb])
        s["ism"] = float(np.median(v))
        s["ism_sd"] = float(1.4826 * np.median(np.abs(v - np.median(v))))
        s["n_nb"] = len(nb)
        w = np.array([ref[j]["pdb"] / ref[j]["e"] for j in nb])
        th = np.radians([2 * (ref[j]["pa"] or 0.0) for j in nb])
        s["ism_pa"] = float(np.degrees(math.atan2(np.sum(w * np.sin(th)), np.sum(w * np.cos(th)))) / 2) % 180
        s["z"] = (s["pdb"] - s["ism"]) / math.hypot(s["e"], s["ism_sd"])
        s["why"] = polarised_explanations(s, high_refs)
        dpa = abs(((s["pa"] or 0.0) - s["ism_pa"] + 90) % 180 - 90)
        if s["src"] != "Heiles" and dpa <= polarised_PA_ALIGNED:
            s["why"].append("aligned with the interstellar field (%.0f deg)" % dpa)
    beyond = [s for s in stars if s["z"] >= 3 and s["p"] >= 4 * s["e"]]
    left = sorted([s for s in beyond if not s["why"]], key=lambda s: (s["old_single"], -s["z"]))
    print("%d stars within 50 pc with polarimetry (%d in the ppm-level samples); %d polarised beyond the interstellar "
          "expectation; %d of them unexplained, %d of those resting on one photoelectric measurement from Heiles' "
          "compilation" % (len(stars), len(ref), len(beyond), len(left), sum(s["old_single"] for s in left)))
    counts = {}
    for s in beyond:
        for w in s["why"]:
            k = w.split(" (")[0]
            counts[k] = counts.get(k, 0) + 1
    print("explanations among them: " + ", ".join("%s %d" % kv for kv in sorted(counts.items(), key=lambda kv: -kv[1])))
    for s in left:
        sm = s.get("simbad") or {}
        dpa = abs(((s["pa"] or 0) - s["ism_pa"] + 90) % 180 - 90)
        print("  %-22s %-10s d %5.1f pc: p %7.1f +- %5.1f ppm (%s), interstellar %6.1f +- %5.1f (%d neighbours), z %5.1f, "
              "PA %5.1f vs %5.1f (diff %4.1f); other measurements: %s"
              % ((sm.get("main_id") or s["meas"][0]["name"])[:22], polarised_spectral_type(s)[:10], s["d"], s["p"], s["e"], s["src"],
                 s["ism"], s["ism_sd"], s["n_nb"], s["z"], s["pa"] or 0, s["ism_pa"], dpa,
                 "; ".join("%s %.0f+-%.0f" % (m["src"], m["p"], m["e"]) for m in s["meas"][1:]) or "none"))
    var = polarised_variability(stars, high_refs)
    print("night-to-night changes (Piirola et al. 2020 Table 11, %d stars observed at T60): %d of %d without a known cause"
          % (polarised_T60_STARS, sum(not v["why"] for v in var), len(var)))
    for v in sorted(var, key=lambda v: v["p"]):
        print("  HD %-7s %-12s %s: %d nights, chi2 %.1f/%d, p %.1e (%.2f after %d trials), largest swing %.0f ppm; %s"
              % (v["hd"], v["sp"][:12], "%.1f pc" % v["d"] if v["d"] else "-", v["nights"], v["chi2"], v["dof"], v["p"],
                 v["p_trials"], polarised_T60_STARS, v["swing_ppm"], "; ".join(v["why"]) or "no known cause"))
    json.dump(stars, open(os.path.join(polarised_HERE, "polarised.json"), "w"), indent=1)
    json.dump(var, open(os.path.join(polarised_HERE, "polarised_variable.json"), "w"), indent=1)


# follow-up of unexplained stars, appended to their notes
polarised_FOLLOWUP = {
    "gaia_dr3:5224975398040550400": (
        "HD 101805, one night's BVR: B 314 +- 18, V 188 +- 19, R 172 +- 19 ppm at 6.6-10.8 deg (Piirola et al. 2020 "
        "Table 9), whose authors read circumstellar small particles; a Serkowski curve fits only with lambda_max 0.245 um "
        "and K 1.15 (p 0.05) and fails with K tied to lambda_max (chi2 15.4 for 1 dof), bluer than the 350 +- 50 nm of "
        "Local Hot Bubble dust (Cotton et al. 2019), while a grey 122 ppm plus 188 ppm in lambda^-4 fits (chi2 0.9); the "
        "nine stars measured at the ppm level nearer than 35 pc within 22 deg read 12-45 ppm, five of them at 72-88 deg: "
        "I Car (HD 90589, 16 pc, 5.2 deg away) 17 +- 5 ppm at 88 deg and eta Cru (HD 105211, 20 pc, 10.8 deg) 17-21 ppm "
        "at 72-79 deg (HIPPI; Marshall et al. 2023, Cotton et al. 2017), beta Car (HD 80007, 34.7 pc, 12.2 deg) 24 +- 3 "
        "ppm at 74 deg (HIPPI; Cotton et al. 2016), HD 77370 (26 pc, 21.7 deg) 36 +- 6 at 86 deg and HD 44447 (32 pc, "
        "21.6 deg) 24 +- 9 at 88 deg (Piirola et al. 2020); HD 93372 (32 pc, 4.9 deg) reads 22 +- 12 ppm at 172 +- 19 "
        "deg, HD 71243 (19.5 pc, 12 deg) 43 +- 14 at 22 +- 10, gamma Cru (27 pc, 18.7 deg) 45 +- 4 at 122 and HD 43834 "
        "(10 pc, 19.8 deg) 12 +- 8 at 114; the heliospheric field (IBEX ribbon centre, Frisch et al. 2015 fit) projects "
        "to 50-58 deg here; a screen in front of HD 101805 would have to end within 2.7 pc of its sightline (HD 93372) "
        "and carry a field turned 63-79 deg from its surroundings'; the angles near 0 deg are alpha Mus (97 pc, 7.4 deg, "
        "146 +- 4 ppm at 0.7 deg, a B2 star whose polarisation Cotton et al. 2016 judge possibly intrinsic), HD 101132 "
        "(44 pc, 0.7 deg, 150 +- 80 ppm at 168 deg) and Gaia DR2 5860952474301981568 (49 pc, 10.1 deg, 1100 +- 300 ppm "
        "at 12 deg, photoelectric); beyond 100 pc the Chamaeleon clouds hold their field at 100-140 deg (Panopoulou et "
        "al. 2025 compilation); no excess in WISE W3/W4, AKARI 9 um or IRAS 12/25 um; SCANPI 60 um sigma 65 mJy and "
        "AKARI 90 um below 0.28 Jy (3 sigma) cap dust at L/L* < 2e-4 for 30-60 K and < 1e-4 at 80 K, while 223 ppm "
        "needs (4.5-22)e-4 of the light scattered; hot dust polarises its star by 30 ppm or less, more in the red "
        "(Marshall et al. 2016); the 5.4 arcsec common-proper-motion companion Gaia DR3 5224975398034733952 (G 14.7, "
        "the eROSITA source 1eRASS J114216.8-751341) gives 0.05 per cent of the light; Gaia DR3 FLAME age 3.9 "
        "(3.2-4.5) Gyr, 1.17 solar masses, constant radial velocity (-7.24 +- 0.13 km/s, p 0.53 over 14 transits); "
        "TESS 2-min sectors 11-65: 60-200 ppm variability at 4-11 d and no exocomet-shaped dip beyond the same-shaped "
        "brightenings; 20-s sectors 64-65: 28 single-cadence flashes in 1250 h against 21.8 at the survey's rate; "
        "Stromgren twins within 25 pc put E(b-y) at 0.003-0.006 against a fit scatter of 0.012; no public "
        "high-resolution spectrum (ESO, NOIRLab, MAST); Gemini/GHOST spectra of 2026 July 12 (GS-2026B-Q-134), public "
        "from 2027 July 12, measure the Na I D and Ca II H and K columns in front of it; Marshall et al. 2023 adopt it "
        "as an interstellar control for eta Cru"),
}


def polarised_record():
    stars = json.load(open(os.path.join(polarised_HERE, "polarised.json")))
    var = {v["hd"]: v for v in json.load(open(os.path.join(polarised_HERE, "polarised_variable.json")))}
    rows = []
    for s in stars:
        g = (s.get("gaia") or {}).get("Source")
        sm = (s.get("simbad") or {}).get("main_id")
        target = ("gaia_dr3:" + g) if g else ("simbad:" + " ".join(sm.split())) if sm else None
        if not target:
            continue
        beyond = s["z"] >= 3 and s["p"] >= 4 * s["e"]
        note = "p %.1f +- %.1f ppm (%s), interstellar %.1f" % (s["p"], s["e"], s["src"], s["ism"])
        hd = [m["name"][3:].strip() for m in s["meas"] if m["name"].startswith("HD ") and m["name"][3:].strip() in var]
        if hd:
            v = var[hd[0]]
            note += "; night-to-night q,u changes over %d nights, p %.1e (%.2f after %d trials)" % (
                v["nights"], v["p"], v["p_trials"], polarised_T60_STARS)
        if target in polarised_FOLLOWUP:
            note += "; " + polarised_FOLLOWUP[target]
        rows.append(dict(target=target, stat=s["z"], upper_limit=s["p"] + 3 * s["e"], unit="ppm, 3-sigma bound on p",
                         vet="; ".join(s["why"]) if beyond and s["why"] else None, note=note))
    left = [s for s in stars if s["z"] >= 3 and s["p"] >= 4 * s["e"] and not s["why"]]
    core.record_search("polarised_reflectors", "Polarised reflectors",
                           archive="VizieR (Piirola+ 2020, Bailey+ 2010, Heiles 2000) with SIMBAD, Gaia DR3, AllWISE",
                           release="catalogues as published", selection="stars within 50 pc with linear polarimetry",
                           code=os.path.abspath(__file__), entry="polarised", params=dict(z=3, snr=4, radius_deg=35, merge_arcsec=polarised_MERGE_ARCSEC),
                           n_targets=len(rows), result="%d polarised beyond the interstellar expectation with no "
                                                       "dust, activity, companion, evolution or early type" % len(left),
                           ledger="Polarised reflectors")
    core.record_targets("polarised_reflectors", rows)
    print("recorded %d stars" % len(rows))


def polarised_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "record"))
    a = p.parse_args()
    {"fetch": polarised_fetch, "analyse": polarised_analyse, "record": polarised_record}[a.mode]()


def polarised_cli():
    quiet_warnings()
    polarised_main()


# --------------------------------------------------------------------------------------------------------------

beams_DOC = r"""Antipodal nearby-star pairs within 20 pc whose mutual beams (0.25 deg or narrower) cover Earth, against randomised directions.

    python -I seti.py neighbourhood beams
    python -I seti.py neighbourhood beams record"""

beams_WIDTH = 0.25  # deg, widest beam counted


def beams_stars():
    rows = core.gaia("SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source "
                         "WHERE parallax > 50 AND parallax_over_error > 10")
    stars = [dict(id="gaia_dr3:%s" % r["source_id"], ra=float(r["ra"]), dec=float(r["dec"]), d=1000.0 / float(r["parallax"]),
                  pmra=float(r["pmra"] or 0), pmdec=float(r["pmdec"] or 0), mag=float(r["phot_g_mean_mag"] or 99))
             for r in rows]
    have = {str(r["original_ext_source_id"]) for r in core.gaia(
        "SELECT h.original_ext_source_id FROM gaiadr3.hipparcos2_best_neighbour AS h JOIN gaiadr3.gaia_source AS g "
        "ON g.source_id = h.source_id WHERE g.parallax > 40")}
    from astroquery.vizier import Vizier
    hip = Vizier(columns=["HIP", "RArad", "DErad", "Plx", "e_Plx", "pmRA", "pmDE", "Hpmag"], row_limit=-1,
                 column_filters={"Plx": ">50"}, timeout=300).get_catalogs("I/311/hip2")[0]
    from scipy.spatial import cKDTree
    gra, gde = np.radians([s["ra"] for s in stars]), np.radians([s["dec"] for s in stars])
    tree = cKDTree(np.vstack([np.cos(gde) * np.cos(gra), np.cos(gde) * np.sin(gra), np.sin(gde)]).T)
    for r in hip:
        if str(int(r["HIP"])) in have or not float(r["Plx"]) > 10 * float(r["e_Plx"]):
            continue
        # Hipparcos stars carried to 2016.0; one within 10 arcsec of a Gaia source is that source
        dec = float(r["DErad"]) + float(r["pmDE"]) / 3.6e6 * 24.75
        ra = float(r["RArad"]) + float(r["pmRA"]) / 3.6e6 * 24.75 / math.cos(math.radians(dec))
        v = [math.cos(math.radians(dec)) * math.cos(math.radians(ra)), math.cos(math.radians(dec)) * math.sin(math.radians(ra)),
             math.sin(math.radians(dec))]
        if tree.query_ball_point(v, 2 * math.sin(math.radians(10 / 3600.0) / 2)):
            continue
        stars.append(dict(id="hip:%d" % int(r["HIP"]), ra=ra, dec=dec, d=1000.0 / float(r["Plx"]),
                          pmra=float(r["pmRA"]), pmdec=float(r["pmDE"]), mag=float(r["Hpmag"])))
    return stars


def beams_pairs(stars, u, limit_deg=2.0):
    """Pairs missing opposition by under limit_deg, with the beam widths covering Earth."""
    from scipy.spatial import cKDTree
    tree = cKDTree(u)
    chord = 2 * math.sin(math.radians(limit_deg) / 2)
    out = []
    for i, js in enumerate(tree.query_ball_point(-u, chord)):
        for j in js:
            if j <= i:
                continue
            e = math.degrees(math.acos(max(-1.0, min(1.0, -float(np.dot(u[i], u[j]))))))
            dA, dB = stars[i]["d"], stars[j]["d"]
            out.append((i, j, e, 2 * e * dB / (dA + dB), 2 * e * dA / (dA + dB)))
    return out


def beams_main():
    stars = beams_stars()
    ra, de = np.radians([s["ra"] for s in stars]), np.radians([s["dec"] for s in stars])
    u = np.vstack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]).T
    pairs = beams_pairs(stars, u)
    keep = [p for p in pairs if min(p[3], p[4]) <= beams_WIDTH]
    rng = np.random.default_rng(11)
    null = []
    for _ in range(200):
        v = rng.normal(size=u.shape)
        v /= np.linalg.norm(v, axis=1)[:, None]
        null.append(sum(1 for p in beams_pairs(stars, v) if min(p[3], p[4]) <= beams_WIDTH))
    print("%d stars within 20 pc; %d pairs a beam of %.2f deg or narrower covers, against %.1f +- %.1f with directions "
          "randomised" % (len(stars), len(keep), beams_WIDTH, np.mean(null), np.std(null)), flush=True)
    db = sqlite3.connect(core.COVERAGE) if os.path.exists(getattr(core, "COVERAGE", "")) else None
    out = []
    for i, j, e, wA, wB in sorted(keep, key=lambda p: min(p[3], p[4])):
        rec = dict(e_deg=e, width_from_a=wA, width_from_b=wB, members=[])
        # alignment drift over a century from both proper motions
        mu = math.hypot((stars[i]["pmra"] + stars[j]["pmra"]), (stars[i]["pmdec"] + stars[j]["pmdec"])) / 3.6e6 * 100
        rec["drift_deg_per_century"] = mu
        for k in (i, j):
            s = stars[k]
            name = s["id"]
            if s["id"].startswith("gaia_dr3:"):
                sb = core.simbad("SELECT b.main_id, b.otype, b.sp_type FROM basic AS b JOIN ident AS t ON t.oidref = b.oid "
                                     "WHERE t.id = 'Gaia DR3 %s'" % s["id"].split(":")[1])
            else:
                sb = core.simbad("SELECT b.main_id, b.otype, b.sp_type FROM basic AS b JOIN ident AS t ON t.oidref = b.oid "
                                     "WHERE t.id = 'HIP %s'" % s["id"].split(":")[1])
            if sb:
                name = "%s (%s, %s)" % (sb[0]["main_id"], sb[0].get("otype"), sb[0].get("sp_type"))
            runs = [r[0] for r in db.execute("SELECT DISTINCT search_id FROM coverage WHERE target = ?", (s["id"],))] if db else []
            rec["members"].append(dict(id=s["id"], name=name, d_pc=round(s["d"], 2), mag=s["mag"], runs=runs))
        out.append(rec)
        a, b = rec["members"]
        print("  %s (%.1f pc) and %s (%.1f pc): %.3f deg from opposition; beam %.3f deg from the first, %.3f from the second; "
              "drift %.2f deg/century; searched %s / %s" % (a["name"], a["d_pc"], b["name"], b["d_pc"], e, wA, wB,
                                                         mu, ",".join(a["runs"]) or "nothing", ",".join(b["runs"]) or "nothing"),
              flush=True)
    json.dump(dict(n_stars=len(stars), null=[float(np.mean(null)), float(np.std(null))], pairs=out),
              open(core.results("beam_pairs.json"), "w"), indent=1)


def beams_record():
    d = json.load(open(core.results("beam_pairs.json")))
    core.record_search(
        "beam_pairs_20pc", "Beams between stars", archive="Gaia DR3; Hipparcos (I/311)",
        selection="stars within 20 pc at 10 sigma in parallax", code=os.path.abspath(__file__), entry="beams",
        n_targets=d["n_stars"], params=dict(width_deg=beams_WIDTH),
        null_summary="the same stars with randomised directions: %.1f +- %.1f pairs" % tuple(d["null"]),
        ledger="Beams between stars",
        result="%d pairs a beam of %.2f deg or narrower would cover, against %.1f +- %.1f by chance; the sightlines list "
               "for radio follow-up in the archives" % (len(d["pairs"]), beams_WIDTH, d["null"][0], d["null"][1]))
    rows = []
    for p in d["pairs"]:
        for k, m in enumerate(p["members"]):
            other = p["members"][1 - k]
            rows.append(dict(target=m["id"], stat=min(p["width_from_a"], p["width_from_b"]),
                             unit="full beam width (deg) that carries the pair's traffic across Earth",
                             note="opposite %s at %.1f pc, %.3f deg from opposition; drift %.2f deg/century"
                                  % (other["name"], other["d_pc"], p["e_deg"], p["drift_deg_per_century"])))
    core.clear_targets("beam_pairs_20pc")
    core.record_targets("beam_pairs_20pc", rows)
    print("recorded %d pair members" % len(rows))


def beams_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        beams_record()
    else:
        beams_main()


# --------------------------------------------------------------------------------------------------------------

joint_DOC = r"""Joint anomaly: rank-uniform p values per run, Sidak-combined by family, Fisher-combined across families, against family-shuffled nulls.

    python -I seti.py neighbourhood joint
    python -I seti.py neighbourhood joint record"""

joint_FAMILIES = (("elements", ("technetium_", "promethium_", "protactinium_", "plutonium_", "americium_", "curium_")),
                  ("glints", ("tess20_glints", "kepler_sc_glints", "tess_toi_glints", "alcdef_glints")),
                  ("photon_pulses", ("galex_nuv_pulses", "galex_fuv_pulses", "hst_photon_pulses")),
                  ("xray_periods", ("xray_beacons_",)),
                  ("radio_frames", ("bl_rest_frame_tracks", "bl_planet_frame")),
                  ("neutrinos", ("icecube_",)),
                  ("lpt", ("lpt_gaia",)))


def joint_family(run):
    for fam, prefixes in joint_FAMILIES:
        if any(run.startswith(p) for p in prefixes):
            return fam
    return run


def joint_main(draws=2000, seed=5):
    db = sqlite3.connect(core.COVERAGE)
    rows = db.execute("SELECT search_id, target, p FROM coverage WHERE p IS NOT NULL").fetchall()
    rng = np.random.default_rng(seed)
    runs = {}
    for s_, t, p in rows:
        runs.setdefault(s_, []).append((t, float(p)))
    fam = {}
    for s_, lst in runs.items():
        ps = np.array([p for _, p in lst])
        order = np.lexsort((rng.random(len(ps)), ps))
        rank = np.empty(len(ps))
        rank[order] = (np.arange(len(ps)) + 1) / (len(ps) + 1)
        for (t, _), r in zip(lst, rank):
            fam.setdefault(joint_family(s_), {}).setdefault(t, []).append(r)
    fp = {f: {t: 1 - (1 - min(v)) ** len(v) for t, v in d.items()} for f, d in fam.items()}
    targets = sorted({t for d in fp.values() for t in d})
    multi = [t for t in targets if sum(t in d for d in fp.values()) >= 2]
    fams = sorted(fp)
    print("%d runs with p values in %d families; %d targets, %d with two or more families"
          % (len(runs), len(fams), len(targets), len(multi)), flush=True)

    def stat(maps):
        out = {}
        for t in multi:
            ps = [maps[f][t] for f in fams if t in maps[f]]
            out[t] = -2.0 * float(np.sum(np.log(np.clip(ps, 1e-12, 1.0))))
        return out
    obs = stat(fp)
    from scipy.stats import chi2
    null_sorted = []
    for _ in range(draws):
        sh = {}
        for f, d in fp.items():
            ks = list(d)
            vs = np.array([d[k] for k in ks])
            rng.shuffle(vs)
            sh[f] = dict(zip(ks, vs))
        null_sorted.append(np.sort(np.array(list(stat(sh).values())))[::-1])
    null_sorted = np.array(null_sorted)
    top = sorted(obs, key=lambda t: -obs[t])[:20]
    out = []
    for rank_, t in enumerate(top):
        k = sum(t in fp[f] for f in fams)
        p_any = float(np.mean(null_sorted[:, 0] >= obs[t]))
        p_rank = float(np.mean(null_sorted[:, rank_] >= obs[t]))
        parts = ", ".join("%s %.3f" % (f, fp[f][t]) for f in fams if t in fp[f])
        out.append(dict(target=t, fisher=obs[t], families=k, p_fisher=float(chi2.sf(obs[t], 2 * k)), p_any=p_any,
                        p_rank=p_rank, parts=parts))
        print("  %-36s Fisher %.1f over %d families (chi2 p %.1e); chance of a value this high anywhere %.3f, at "
              "this rank %.3f; %s" % (t, obs[t], k, chi2.sf(obs[t], 2 * k), p_any, p_rank, parts), flush=True)
    json.dump(dict(n_runs=len(runs), families=fams, n_multi=len(multi), top=out), open(core.results("joint_anomaly.json"), "w"),
              indent=1)


def joint_record():
    d = json.load(open(core.results("joint_anomaly.json")))
    best = d["top"][0] if d["top"] else None
    core.record_search(
        "joint_anomaly", "Joint anomaly across methods", archive="the coverage store",
        selection="targets with p values from two or more families of runs", code=os.path.abspath(__file__), entry="joint",
        n_targets=d["n_multi"], params=dict(families=d["families"]),
        null_summary="each family's values shuffled among its own targets",
        ledger="Joint anomaly across methods",
        result=("the strongest combination, %s (%s), has chance %.3f of a value this high anywhere"
                % (best["target"], best["parts"], best["p_any"])) if best else "no target with two families")
    core.clear_targets("joint_anomaly")
    core.record_targets("joint_anomaly", [dict(target=r["target"], stat=r["fisher"], p=r["p_rank"],
                                                   unit="Fisher statistic over families", note=r["parts"]) for r in d["top"]])
    print("recorded")


def joint_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        joint_record()
    else:
        joint_main()


# --------------------------------------------------------------------------------------------------------------

herded_DOC = r"""Herded stars: APOGEE dwarfs within 100 pc moving together within 1 km/s and 10 pc but differing in [Fe/H].

    python -I seti.py neighbourhood herded [record] [--link-pc 10] [--link-kms 1] [--verr-kms 0.5] [--teff-band 600] [--spread-dex 0.2]"""


def herded_main(record=False, link_pc=10.0, link_kms=1.0, verr_kms=0.5, teff_band=600, spread_dex=0.2):
    import csv as _csv
    import urllib.parse
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from scipy.spatial import cKDTree
    p = core.data("apogee_100pc_dwarfs.csv")
    if not os.path.exists(p):
        q = ('SELECT "APOGEE", "RAJ2000", "DEJ2000", "plx", "e_plx", "pmRA", "e_pmRA", "pmDE", "e_pmDE", "HRV", "e_HRV", "s_HRV", '
             '"Nvis", "Teff", "logg", "[Fe/H]", "e_[Fe/H]", "f_[Fe/H]", "SNR", "GaiaEDR3", "Vsini" FROM "III/286/catalog" WHERE '
             '"plx" > 10 AND "plx"/"e_plx" > 20 AND "logg" > 3.5 AND "[Fe/H]" IS NOT NULL AND "HRV" IS NOT NULL AND "SNR" > 50')
        u_ = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?" + urllib.parse.urlencode(
            dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q, MAXREC=500000))
        b = core.get(u_, timeout=1800)
        open(p, "wb").write(b if isinstance(b, bytes) else b.encode("utf-8"))
    best = {}
    for r in _csv.DictReader(open(p, encoding="utf-8")):
        g = r["GaiaEDR3"].strip()
        if g and (g not in best or (num(r["SNR"]) or 0) > (num(best[g]["SNR"]) or 0)):
            best[g] = r
    R = [r for r in best.values() if (r["f_[Fe/H]"] or "0").strip() in ("0", "")]

    def col(k):
        return np.array([num(r[k]) if num(r[k]) is not None else np.nan for r in R], float)
    ra, de, plx, eplx = col("RAJ2000"), col("DEJ2000"), col("plx"), col("e_plx")
    pmra, pmde, rv, erv, srv = col("pmRA"), col("pmDE"), col("HRV"), col("e_HRV"), col("s_HRV")
    feh, teff = col("[Fe/H]"), col("Teff")
    c = SkyCoord(ra=ra * u.deg, dec=de * u.deg, distance=(1000 / plx) * u.pc, pm_ra_cosdec=pmra * u.mas / u.yr,
                 pm_dec=pmde * u.mas / u.yr, radial_velocity=rv * u.km / u.s).galactic
    c.representation_type = "cartesian"
    c.differential_type = "cartesian"
    X = np.vstack([c.u.to(u.pc).value, c.v.to(u.pc).value, c.w.to(u.pc).value]).T
    V = np.vstack([c.velocity.d_x.to(u.km / u.s).value, c.velocity.d_y.to(u.km / u.s).value, c.velocity.d_z.to(u.km / u.s).value]).T
    vt = 4.74047 * np.hypot(pmra, pmde) / plx
    sv = np.sqrt((vt * eplx / plx) ** 2 + np.nan_to_num(erv) ** 2 + np.where(np.nan_to_num(srv) > 0.5, srv, 0) ** 2)
    ok = np.isfinite(X).all(1) & np.isfinite(V).all(1) & (sv < verr_kms)
    idx = np.nonzero(ok)[0]
    pairs = [(idx[a], idx[b]) for a, b in cKDTree(X[idx]).query_pairs(link_pc) if np.linalg.norm(V[idx[a]] - V[idx[b]]) < link_kms]
    parent = {i: i for i in idx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = {}
    for a, b in pairs:
        for x in (a, b):
            groups.setdefault(find(x), set()).add(x)
    out = []
    for g in sorted((sorted(v) for v in groups.values()), key=len, reverse=True):
        fe, te = feh[g], teff[g]
        like, band = 0.0, None
        for lo in range(3400, 7400 - int(teff_band), 100):
            m = (te >= lo) & (te < lo + teff_band)
            if m.sum() >= 2 and fe[m].max() - fe[m].min() > like:
                like, band = float(fe[m].max() - fe[m].min()), (lo, lo + int(teff_band))
        out.append(dict(n=len(g), ids=[R[i]["APOGEE"] for i in g], gaia=[R[i]["GaiaEDR3"] for i in g], feh=[float(x) for x in fe],
                        teff=[float(x) for x in te], spread=float(fe.max() - fe.min()), spread_like=like, band=band,
                        vdisp=float(np.linalg.norm(np.std(V[g], axis=0))), size=float(np.max(np.linalg.norm(X[g] - X[g].mean(0), axis=1)))))
    print("%d dwarfs, %d with 3D velocity errors below %g km/s; %d pairs within %g pc and %g km/s in %d groups, %d of 3 or more"
          % (len(R), len(idx), verr_kms, len(pairs), link_pc, link_kms, len(out), sum(1 for o in out if o["n"] >= 3)))
    for o in out:
        if o["n"] >= 3 or o["spread_like"] > spread_dex:
            print("  n %d, %.1f pc, %.2f km/s: [Fe/H] %+.2f to %+.2f; like-with-like spread %.2f in Teff %s"
                  % (o["n"], o["size"], o["vdisp"], min(o["feh"]), max(o["feh"]), o["spread_like"], o["band"]))
    json.dump(out, open(core.results("herded_groups.json"), "w"), indent=1)
    if record:
        big = [o for o in out if o["spread_like"] > spread_dex]
        core.record_search(
            "apogee_herded_stars", "Herded stars", archive="APOGEE DR17 (VizieR III/286) with its Gaia EDR3 astrometry",
            selection="%d dwarfs within 100 pc, %d with 3D velocity errors below %g km/s" % (len(R), len(idx), verr_kms),
            code=os.path.abspath(__file__), entry="herded", n_targets=len(idx),
            params=dict(link_pc=link_pc, link_kms=link_kms, verr_kms=verr_kms, teff_band=teff_band, spread_dex=spread_dex),
            null_summary="coeval groups: the Hyades and the other co-moving groups found, whose like-with-like spreads measure "
                         "the pipeline's scatter",
            sensitivity="a group of two or more stars within %g pc and %g km/s whose [Fe/H] differs by %g dex at like Teff" % (link_pc, link_kms, spread_dex),
            ledger="Herded stars",
            result="%d groups (%d of 3 or more, the largest %d members, the Hyades); like-with-like spreads above %g dex: %d, %s"
                   % (len(out), sum(1 for o in out if o["n"] >= 3), out[0]["n"] if out else 0, spread_dex, len(big),
                      "; ".join("n %d at %d-%d K, %.2f dex" % (o["n"], o["band"][0], o["band"][1], o["spread_like"]) for o in big) or "none"))
        core.clear_targets("apogee_herded_stars")
        core.record_targets("apogee_herded_stars", [dict(
            target="gaia_dr3:%s" % g, stat=o["spread_like"], unit="like-with-like [Fe/H] spread of its group",
            note="group of %d within %.1f pc and %.2f km/s" % (o["n"], o["size"], o["vdisp"])) for o in out for g in o["gaia"]])
        print("recorded")


intrinsic_DOC = r"""Intrinsic polarisation of quiet dwarfs within 200 pc: Stokes residuals against same-distance neighbours (Panopoulou et al. 2025) at 5 sigma and 0.1 per cent.

    python -I seti.py neighbourhood intrinsic fetch
    python -I seti.py neighbourhood intrinsic analyse
    python -I seti.py neighbourhood intrinsic record"""

intrinsic_ROOT = os.environ.get("INTRINSIC_ROOT", core.data("polar_quiet"))
intrinsic_CAT = "J/ApJS/276/15"
intrinsic_OUT = "intrinsic_polarisation.json"
intrinsic_RADIUS_DEG, intrinsic_DIST_FRAC, intrinsic_NMIN = 3.0, 0.2, 5
intrinsic_PMIN, intrinsic_Z, intrinsic_ANGLE, intrinsic_DMAX = 0.001, 5.0, 30.0, 200.0  # fraction: 0.001 is 0.1 per cent
intrinsic_SYS = 0.001  # allowed inter-catalogue zero-point difference
# catalogue and SIMBAD findings per candidate; cloud-survey stars carry the cloud's polarisation
intrinsic_EXPLAINED = {
    6010997328529547904: "HD 142832, F3V at 158 pc in the Lupus 1 field of Alves and Franco (2006), at the cloud's distance, so the cloud's dust polarises it and not its nearer neighbours",
    4687221887455439616: "Dachs SMC 2-61, G5V at 134 pc with RUWE 8.9 and an X-ray counterpart, measured in a crowded SMC field (Lobo Gomes et al. 2015): a binary or a blend",
}
intrinsic_OPEN = {
    154235505416809344: "HD 31435, F9/G0 at 100 pc, 8-9 Gyr old at the turnoff: 5.48 +- 0.15 per cent at 47 deg in Johnson V on the Lowell 42-inch in 1977 November, listed by "
                        "name in Table 2 of Moneti et al. (1984), whose other 93 stars of that run read below 0.4 per cent within 130 pc and rise with reddening beyond; "
                        "no other polarimetry of the star exists in 135 VizieR catalogues; Gaia gives E(BP-RP) 0.00 and B-V 0.565 against an intrinsic 0.58, so an interstellar "
                        "origin needs ten times the dust in front of it; WISE sees the photosphere to 22 um, TESS, K2 and Hipparcos are flat to 0.03 per cent, RUWE 0.99 and the "
                        "proper-motion anomaly of 20 +- 32 m/s allow no companion, and the Harvard plates show no dimming around 1977 against matched stars on the same plates; "
                        "the angle matches the field's 46-66 deg behind the cloud, and HD 283904, a reddened A2 star of the same brightness 5 arcmin away at 214 pc with A_G 1.3, "
                        "would read near that value; Planck's 353 GHz polarisation at the position predicts 49-54 deg for starlight behind the column against the measured 47, "
                        "and 2.3-2.8 per cent for its amount, half the measured value; the star's Gaia XP spectrum against 40 unreddened twins is flat to 0.3 per cent from "
                        "400 to 700 nm, which leaves only a grey reflector, and Pan-STARRS profiles against stars of like brightness show no excess light at 2-13 arcsec, so with "
                        "WISE inside 10 AU and Gaia's image statistics between, a reflector has no distance left; a wrong star in 1977 on an efficiently polarising sightline; "
                        "one new measurement of both stars decides",
    6286369839598558592: "HD 127769, A7V at 186 pc: 1.3 per cent at 43 deg in the Heiles (2000) compilation against the field's 0.4 per cent at 87 deg; one modern measurement decides",
}


def intrinsic_fetch():
    from astropy.table import Table
    os.makedirs(intrinsic_ROOT, exist_ok=True)
    p = os.path.join(intrinsic_ROOT, "panopoulou_extended.ecsv")
    if not os.path.exists(p):
        t = core.vizier(intrinsic_CAT + "/table4")[0]
        t.write(p, format="ascii.ecsv", overwrite=True)
    t = Table.read(p, format="ascii.ecsv")
    # Gaia DR3 photometry, parallax and RUWE via XMatch, keeping the matching identifier
    seen, rows = set(), []
    for r in t:
        sid = int(r["GaiaEDR3"]) if r["GaiaEDR3"] == r["GaiaEDR3"] else 0
        if sid > 0 and sid not in seen and r["RAJ2000"] == r["RAJ2000"]:
            seen.add(sid)
            rows.append(dict(source_id=sid, ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"])))
    keep = {}
    for k in range(0, len(rows), 15000):
        for m in core.xmatch(rows[k:k + 15000], "I/355/gaiadr3", 2.0, id_col="source_id", ra_col="ra", dec_col="dec", timeout=900):
            sid = int(m["source_id"])
            same = str(m.get("Source")) == str(sid)
            if sid not in keep or (same and str(keep[sid].get("Source")) != str(sid)) or (not same and str(keep[sid].get("Source")) != str(sid) and float(m["angDist"]) < float(keep[sid]["angDist"])):
                keep[sid] = m
    cols = [c for c in ("source_id", "angDist", "Source", "Plx", "e_Plx", "pmRA", "pmDE", "Gmag", "BP-RP", "RUWE", "RV", "Teff", "logg") if keep and c in next(iter(keep.values()))]
    with open(os.path.join(intrinsic_ROOT, "gaia_xmatch.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep.values())
    print("%d stars, Gaia DR3 rows for %d, %d with the same identifier" % (len(rows), len(keep), sum(1 for s, m in keep.items() if str(m.get("Source")) == str(s))))


def intrinsic_stars():
    """One row per star: weighted q, u with errors, position, distance, G, flag, Gaia colour, M_G and RUWE."""
    from astropy.table import Table
    t = Table.read(os.path.join(intrinsic_ROOT, "panopoulou_extended.ecsv"), format="ascii.ecsv")
    gaia = {}
    gp = os.path.join(intrinsic_ROOT, "gaia_xmatch.csv")
    if os.path.exists(gp):
        for r in csv.DictReader(open(gp)):
            gaia[int(r["source_id"])] = dict(parallax=r["Plx"], bp_rp=r["BP-RP"], phot_g_mean_mag=r["Gmag"], ruwe=r["RUWE"], same=str(r["Source"]) == str(r["source_id"]))
    acc = {}
    for r in t:
        sid = int(r["GaiaEDR3"]) if r["GaiaEDR3"] == r["GaiaEDR3"] else 0
        q, u, eq, eu, d = (float(r[k]) if r[k] == r[k] else None for k in ("q", "u", "e_q", "e_u", "rpgeo"))
        if sid <= 0 or q is None or u is None or not eq or not eu or eq <= 0 or eu <= 0 or not d or d <= 0:
            continue
        a = acc.setdefault(sid, dict(gaia=sid, ra=float(r["RAGdeg"]) if r["RAGdeg"] == r["RAGdeg"] else float(r["RAJ2000"]),
                                     dec=float(r["DEGdeg"]) if r["DEGdeg"] == r["DEGdeg"] else float(r["DEJ2000"]), d=d,
                                     G=float(r["Gmag"]) if r["Gmag"] == r["Gmag"] else None, name=str(r["Name"]) if r["Name"] == r["Name"] else "",
                                     flagged=False, sq=0.0, su=0.0, wq=0.0, wu=0.0, n=0, refs=set()))
        if str(r["IntrPol"]) == "1":
            a["flagged"] = True
        a["refs"].add(int(r["RefID"]))
        a["sq"] += q / eq ** 2
        a["wq"] += 1 / eq ** 2
        a["su"] += u / eu ** 2
        a["wu"] += 1 / eu ** 2
        a["n"] += 1
    stars = []
    for a in acc.values():
        a["q"], a["u"] = a["sq"] / a["wq"], a["su"] / a["wu"]
        a["eq"], a["eu"] = 1 / math.sqrt(a["wq"]), 1 / math.sqrt(a["wu"])
        g = gaia.get(a["gaia"])
        if g and num(g["parallax"]) and num(g["parallax"]) > 0 and num(g["bp_rp"]) is not None and num(g["phot_g_mean_mag"]) is not None:
            a["bprp"] = num(g["bp_rp"])
            a["MG"] = num(g["phot_g_mean_mag"]) + 5 * math.log10(num(g["parallax"])) - 10
            a["ruwe"] = num(g["ruwe"])
            # main sequence M_G = 1.7 + 3.6 (BP-RP), F5 to M3
            a["dwarf"] = 0.4 < a["bprp"] < 3.2 and abs(a["MG"] - (1.7 + 3.6 * a["bprp"])) < 1.2
        else:
            a["bprp"], a["MG"], a["ruwe"], a["dwarf"] = None, None, None, None
        for k in ("sq", "su", "wq", "wu"):
            del a[k]
        a["refs"] = sorted(a["refs"])
        stars.append(a)
    return stars


def intrinsic_residuals(stars, shuffle=None):
    """Residual Stokes vector against neighbours; with shuffle, a random neighbour's values stand in."""
    from scipy.spatial import cKDTree
    ok = [s for s in stars if not s["flagged"]]
    a = np.radians([[s["ra"], s["dec"]] for s in ok])
    xyz = np.column_stack([np.cos(a[:, 1]) * np.cos(a[:, 0]), np.cos(a[:, 1]) * np.sin(a[:, 0]), np.sin(a[:, 1])])
    tree = cKDTree(xyz)
    chord = 2 * math.sin(math.radians(intrinsic_RADIUS_DEG) / 2)
    d = np.array([s["d"] for s in ok])
    q = np.array([s["q"] for s in ok])
    u = np.array([s["u"] for s in ok])
    eq = np.array([s["eq"] for s in ok])
    eu = np.array([s["eu"] for s in ok])
    refs = [set(s["refs"]) for s in ok]
    out = {}
    for i, s in enumerate(ok):
        nb = [j for j in tree.query_ball_point(xyz[i], chord) if j != i and abs(d[j] - d[i]) <= intrinsic_DIST_FRAC * d[i]]
        if len(nb) < intrinsic_NMIN:
            continue
        # neighbours from the star's own catalogues when enough
        same = [j for j in nb if refs[j] & refs[i]]
        within = len(same) >= intrinsic_NMIN
        nb = np.array(same if within else nb)
        sys2 = 0.0 if within else intrinsic_SYS ** 2
        wq, wu = 1 / (eq[nb] ** 2 + sys2 + 0.0001 ** 2), 1 / (eu[nb] ** 2 + sys2 + 0.0001 ** 2)
        mq, mu = float(np.sum(wq * q[nb]) / wq.sum()), float(np.sum(wu * u[nb]) / wu.sum())
        sq = float(math.sqrt(np.sum(wq * (q[nb] - mq) ** 2) / wq.sum()))
        su = float(math.sqrt(np.sum(wu * (u[nb] - mu) ** 2) / wu.sum()))
        if shuffle is not None:
            j = int(shuffle.choice(nb))
            qi, ui, eqi, eui = q[j], u[j], eq[j], eu[j]
        else:
            qi, ui, eqi, eui = q[i], u[i], eq[i], eu[i]
        dq, du = qi - mq, ui - mu
        z2 = dq ** 2 / (eqi ** 2 + sys2 + sq ** 2) + du ** 2 / (eui ** 2 + sys2 + su ** 2)
        p_res = math.hypot(dq, du)
        ang_res = 0.5 * math.degrees(math.atan2(du, dq)) % 180
        ang_nb = 0.5 * math.degrees(math.atan2(mu, mq)) % 180
        dang = abs((ang_res - ang_nb + 90) % 180 - 90)
        out[s["gaia"]] = dict(dq=float(dq), du=float(du), sq=sq, su=su, n=int(len(nb)), within=within, p_res=p_res, z=math.sqrt(z2), p_nb=math.hypot(mq, mu),
                              ang_res=ang_res, ang_nb=ang_nb, dang=dang)
    return out


def intrinsic_analyse():
    stars = intrinsic_stars()
    res = intrinsic_residuals(stars)
    by = {s["gaia"]: s for s in stars}
    z = np.array([r["z"] for r in res.values()])
    print("%d stars with measurements, %d unflagged with %d or more neighbours within %.0f deg and %.0f per cent in distance (%d judged within their own catalogues); "
          "residual z: median %.2f, above 3: %d (%.4f), above 5: %d" % (len(stars), len(res), intrinsic_NMIN, intrinsic_RADIUS_DEG, 100 * intrinsic_DIST_FRAC,
                                                                          sum(1 for r in res.values() if r["within"]), np.median(z), int((z > 3).sum()), float(np.mean(z > 3)), int((z > 5).sum())))
    rng = np.random.default_rng(7)
    nulls = []
    for k in range(5):
        rn = intrinsic_residuals(stars, shuffle=rng)
        zn = np.array([r["z"] for r in rn.values()])
        nulls.append((float(np.mean(zn > 3)), float(np.mean(zn > 5)), float(np.mean([r["z"] > 3 and r["p_res"] >= intrinsic_PMIN for r in rn.values()]))))
    print("measurements reassigned to random neighbours, 5 times: fraction above 3 %.4f +- %.4f, above 5 %.4f, above 3 with residual %.2f per cent or more %.4f" % (
        np.mean([n[0] for n in nulls]), np.std([n[0] for n in nulls]), np.mean([n[1] for n in nulls]), 100 * intrinsic_PMIN, np.mean([n[2] for n in nulls])))
    # residual above five times the field scatter and 0.1 per cent
    cands = []
    for sid, r in res.items():
        s = by[sid]
        floor = max(intrinsic_PMIN, 5 * math.hypot(r["sq"], r["su"]))
        if r["z"] >= intrinsic_Z and r["p_res"] >= floor and (r["dang"] >= intrinsic_ANGLE or r["p_nb"] < floor) and s["d"] <= intrinsic_DMAX and s["dwarf"]:
            cands.append(dict(s, explained=intrinsic_EXPLAINED.get(sid), open=intrinsic_OPEN.get(sid), **r))
    cands.sort(key=lambda c: -c["z"])
    print("%d candidates: dwarfs within %.0f pc at z %.0f or more with a residual above %.2f per cent and five times the field's scatter, discordant by %.0f deg or in a field below that" % (
        len(cands), intrinsic_DMAX, intrinsic_Z, 100 * intrinsic_PMIN, intrinsic_ANGLE))
    for c in cands[:80]:
        print("  Gaia %s %-18s %.0f pc G %.1f BP-RP %.2f M_G %.2f RUWE %s cat %s: q %+.3f u %+.3f %% (+- %.3f, %.3f, %d meas.); %d neighbours%s q %+.3f u %+.3f %% scatter %.3f %.3f; residual %.3f%% at %.0f deg vs field %.3f%% at %.0f deg (%.0f deg apart), z %.1f" % (
            c["gaia"], c["name"][:18], c["d"], c["G"] or 0, c["bprp"], c["MG"], "%.2f" % c["ruwe"] if c["ruwe"] else "-", ",".join(str(x) for x in c["refs"]), 100 * c["q"], 100 * c["u"], 100 * c["eq"], 100 * c["eu"],
            c["n_meas"] if "n_meas" in c else c["n"], c["n"], " of its own catalogues" if c["within"] else " of other catalogues", 100 * (c["q"] - c["dq"]), 100 * (c["u"] - c["du"]), 100 * c["sq"], 100 * c["su"],
            100 * c["p_res"], c["ang_res"], 100 * c["p_nb"], c["ang_nb"], c["dang"], c["z"]))
    tested = [dict(gaia=sid, name=by[sid]["name"], d=by[sid]["d"], dwarf=by[sid]["dwarf"], bprp=by[sid]["bprp"], MG=by[sid]["MG"], ruwe=by[sid]["ruwe"], **r) for sid, r in res.items() if r["z"] >= 3]
    json.dump(dict(n_stars=len(stars), n_tested=len(res), frac_z3=float(np.mean(z > 3)), frac_z5=float(np.mean(z > 5)), null=nulls, candidates=cands, above_3=tested,
                   params=dict(radius_deg=intrinsic_RADIUS_DEG, dist_frac=intrinsic_DIST_FRAC, nmin=intrinsic_NMIN, pmin=intrinsic_PMIN, z=intrinsic_Z, angle=intrinsic_ANGLE, dmax=intrinsic_DMAX)),
              open(core.results(intrinsic_OUT), "w"), indent=1)


def intrinsic_record():
    d = json.load(open(core.results(intrinsic_OUT)))
    for c in d["candidates"]:
        c["explained"], c["open"] = intrinsic_EXPLAINED.get(c["gaia"]), intrinsic_OPEN.get(c["gaia"])
    core.record_search(
        "intrinsic_polarisation", "Intrinsic polarisation of quiet dwarfs", archive="Panopoulou et al. (2025) compilation of optical polarisation (VizieR J/ApJS/276/15); Gaia DR3",
        release="compilation as on VizieR", selection="stars with Stokes q and u, Gaia identifiers and distances, unflagged, with five or more neighbours (their own catalogues' where there are five)",
        code=os.path.abspath(__file__), entry="intrinsic", n_targets=d["n_tested"], params=d["params"],
        null_summary="measurements reassigned to random neighbours: fraction above z 3 %.4f, above 5 %.4f, above 3 with residual %.2f per cent or more %.4f" % (
            np.mean([n[0] for n in d["null"]]), np.mean([n[1] for n in d["null"]]), 100 * d["params"]["pmin"], np.mean([n[2] for n in d["null"]])),
        sensitivity="a residual of 0.1 per cent at 5 sigma, an asymmetric intercepted fraction of a few 1e-3 for 30-50 per cent polarised reflection",
        ledger="Intrinsic polarisation of quiet dwarfs",
        result="%d stars tested: residual above 3 sigma in %.4f (null %.4f), above 5 in %.4f (null %.4f); %d dwarf candidates within %.0f pc%s" % (
            d["n_tested"], d["frac_z3"], np.mean([n[0] for n in d["null"]]), d["frac_z5"], np.mean([n[1] for n in d["null"]]), len(d["candidates"]), d["params"]["dmax"],
            (": " + "; ".join("%s %.2f per cent (%s)" % (c["name"] or ("Gaia DR3 %d" % c["gaia"]), 100 * c["p_res"], c.get("explained") or c.get("open") or "unexplained") for c in d["candidates"])) if d["candidates"] else ""))
    core.clear_targets("intrinsic_polarisation")
    core.record_targets("intrinsic_polarisation", [dict(target="gaia_dr3:%d" % c["gaia"], stat=c["z"], unit="residual Stokes vector over its error",
                                                            vet="residual %.3f per cent at %.0f deg against the field's %.3f at %.0f deg (%s); RUWE %s%s" % (
                                                                100 * c["p_res"], c["ang_res"], 100 * c["p_nb"], c["ang_nb"], "own catalogues" if c["within"] else "other catalogues",
                                                                "%.2f" % c["ruwe"] if c["ruwe"] else "-", ("; " + (c.get("explained") or c.get("open"))) if (c.get("explained") or c.get("open")) else ""),
                                                            note="%s, %.0f pc, BP-RP %.2f, M_G %.2f, %d neighbours, catalogues %s" % (c["name"], c["d"], c["bprp"], c["MG"], c["n"], ",".join(str(x) for x in c["refs"]))) for c in d["candidates"]])
    print("recorded")


def intrinsic_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": intrinsic_fetch, "analyse": intrinsic_analyse, "record": intrinsic_record}[mode]()


poles_DOC = r"""Orbital poles of Gaia DR3 astrometric orbits near the Galactic poles: substellar against latitude-reweighted stellar companions, bootstrapped.

    python -I seti.py neighbourhood poles fetch
    python -I seti.py neighbourhood poles analyse
    python -I seti.py neighbourhood poles record"""

poles_ROOT = os.environ.get("POLES_ROOT", core.data("nss_poles"))
poles_OUT = "orbital_poles.json"
poles_CONE, poles_SUBSTELLAR, poles_STELLAR = 15.0, 0.08, 0.2  # deg; Msun bounds for substellar and stellar companions
poles_NGP = (192.859508, 27.128336)  # north Galactic pole, ICRS


def poles_fetch():
    os.makedirs(poles_ROOT, exist_ok=True)
    p = os.path.join(poles_ROOT, "orbits.csv")
    if os.path.exists(p):
        print("have", p)
        return
    # orbits and masses fetched separately and joined here
    rows = core.tap("gaia", "SELECT source_id, ra, dec, parallax, period, eccentricity, a_thiele_innes, b_thiele_innes, f_thiele_innes, g_thiele_innes, "
                           "a_thiele_innes_error, b_thiele_innes_error, f_thiele_innes_error, g_thiele_innes_error, nss_solution_type, goodness_of_fit, "
                           "significance FROM gaiadr3.nss_two_body_orbit WHERE nss_solution_type IN ('Orbital', 'OrbitalAlternative', "
                           "'OrbitalAlternativeValidated', 'OrbitalTargetedSearch', 'OrbitalTargetedSearchValidated', 'AstroSpectroSB1') AND a_thiele_innes IS NOT NULL",
                   maxrec=400000, timeout=1800)
    masses = {m["source_id"]: m for m in core.tap("gaia", "SELECT source_id, m1, m2, m2_lower, m2_upper, flag FROM gaiadr3.binary_masses", maxrec=400000, timeout=1800)}
    for r in rows:
        m = masses.get(r["source_id"], {})
        for k in ("m1", "m2", "m2_lower", "m2_upper", "flag"):
            r[k] = m.get(k, "")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%d orbits, %d with companion masses" % (len(rows), sum(1 for r in rows if r["m2_upper"])))


def poles_elements(A, B, F, G):
    """Inclination (deg) and two candidate nodes (deg) from Thiele-Innes elements."""
    k1 = math.hypot(A + G, B - F)
    k2 = math.hypot(A - G, B + F)
    if k1 + k2 <= 0:
        return None
    cosi = (k1 - k2) / (k1 + k2)
    wpo = math.atan2(B - F, A + G)
    wmo = math.atan2(-(B + F), A - G)
    node = 0.5 * (wpo - wmo)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosi)))), math.degrees(node) % 360.0, (math.degrees(node) + 180.0) % 360.0, 0.5 * (k1 + k2)


def poles_galactic_latitude(ra, dec, inc, node):
    """Galactic latitude (deg) of the orbital pole for a node and inclination."""
    i, om = math.radians(inc), math.radians(node)
    pe, pn, pk = math.sin(i) * math.cos(om), -math.sin(i) * math.sin(om), math.cos(i)
    a, d = math.radians(ra), math.radians(dec)
    e = np.array([-math.sin(a), math.cos(a), 0.0])
    n = np.array([-math.sin(d) * math.cos(a), -math.sin(d) * math.sin(a), math.cos(d)])
    k = np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)])
    pole = pe * e + pn * n + pk * k
    ag, dg = math.radians(poles_NGP[0]), math.radians(poles_NGP[1])
    z = np.array([math.cos(dg) * math.cos(ag), math.cos(dg) * math.sin(ag), math.sin(dg)])
    return math.degrees(math.asin(max(-1.0, min(1.0, float(pole @ z)))))


def poles_analyse():
    rows = list(csv.DictReader(open(os.path.join(poles_ROOT, "orbits.csv"), encoding="utf-8")))
    out = []
    for r in rows:
        A, B, F, G = (num(r[k]) for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
        if None in (A, B, F, G):
            continue
        el = poles_elements(A, B, F, G)
        if not el:
            continue
        inc, n1, n2, a0 = el
        ra, dec = float(r["ra"]), float(r["dec"])
        b1, b2 = poles_galactic_latitude(ra, dec, inc, n1), poles_galactic_latitude(ra, dec, inc, n2)
        m2u, m2l = num(r.get("m2_upper")), num(r.get("m2_lower"))
        # host Galactic latitude for reweighting
        ag, dg = math.radians(poles_NGP[0]), math.radians(poles_NGP[1])
        a, d = math.radians(ra), math.radians(dec)
        bh = math.degrees(math.asin(math.sin(d) * math.sin(dg) + math.cos(d) * math.cos(dg) * math.cos(a - ag)))
        out.append(dict(source_id=r["source_id"], ra=ra, dec=dec, b_host=bh, inc=inc, a0=a0, period=num(r["period"]), sig=num(r["significance"]),
                        type=r["nss_solution_type"], m2_upper=m2u, m2_lower=m2l, b_pole=max(abs(b1), abs(b2)),
                        substellar=m2l is not None and m2l < poles_SUBSTELLAR, strict=m2l is not None and m2l < poles_SUBSTELLAR and m2u is not None and m2u < poles_STELLAR,
                        aligned=max(abs(b1), abs(b2)) >= 90.0 - poles_CONE))
    sub = [x for x in out if x["substellar"]]
    strict = [x for x in out if x["strict"]]
    stel = [x for x in out if x["m2_lower"] is not None and x["m2_lower"] >= poles_STELLAR]
    # stellar set reweighted in 10 deg latitude bins
    bins = np.arange(-90, 91, 10)
    hs = np.histogram([x["b_host"] for x in sub], bins)[0].astype(float)
    hl = np.histogram([x["b_host"] for x in stel], bins)[0].astype(float)
    wt = np.where(hl > 0, hs / np.maximum(hl, 1), 0.0)
    wst = np.array([wt[min(int((x["b_host"] + 90) // 10), 17)] for x in stel])
    al_st = np.array([x["aligned"] for x in stel], float)
    f_sub = np.mean([x["aligned"] for x in sub]) if sub else float("nan")
    f_st = float(np.sum(wst * al_st) / wst.sum()) if wst.sum() > 0 else float("nan")
    rng = np.random.default_rng(2)
    boots, boots_strict = [], []
    for _ in range(2000):
        idx = rng.choice(len(stel), len(sub), replace=True, p=wst / wst.sum())
        boots.append(float(np.mean(al_st[idx])))
        idx = rng.choice(len(stel), max(len(strict), 1), replace=True, p=wst / wst.sum())
        boots_strict.append(float(np.mean(al_st[idx])))
    boots, boots_strict = np.array(boots), np.array(boots_strict)
    iso = 1 - math.sin(math.radians(90 - poles_CONE))
    f_strict = float(np.mean([x["aligned"] for x in strict])) if strict else float("nan")
    print("%d astrometric orbits with elements: %d companions that may be substellar (m2 lower bound below %.2f Msun, %d of them with the upper bound below %.1f), %d stellar "
          "(lower bound above %.1f); poles within %.0f deg of a Galactic pole on either node: possibly substellar %.4f (%d of %d), the strict set %.4f (%d of %d), stellar %.4f "
          "reweighted to the substellar hosts' latitudes (bootstrap %.4f +- %.4f, p %.3f; at the strict set's size %.4f +- %.4f, p %.3f), isotropic %.4f" % (
              len(out), len(sub), poles_SUBSTELLAR, len(strict), poles_STELLAR, len(stel), poles_STELLAR, poles_CONE, f_sub, sum(1 for x in sub if x["aligned"]), len(sub),
              f_strict, sum(1 for x in strict if x["aligned"]), len(strict), f_st, boots.mean(), boots.std(), float(np.mean(boots >= f_sub)) if sub else float("nan"),
              boots_strict.mean(), boots_strict.std(), float(np.mean(boots_strict >= f_strict)) if strict else float("nan"), iso))
    # face-on selection: inclinations against sin i
    inc = np.array([x["inc"] for x in out])
    print("  inclinations: fraction within 30 deg of face-on %.3f (isotropic %.3f), within 30 deg of edge-on %.3f (%.3f)" % (
        np.mean((inc < 30) | (inc > 150)), 1 - math.cos(math.radians(30)), np.mean(np.abs(inc - 90) < 30), math.cos(math.radians(60))))
    json.dump(dict(n=len(out), n_sub=len(sub), n_strict=len(strict), f_strict=f_strict, n_aligned_strict=sum(1 for x in strict if x["aligned"]), n_stellar=len(stel), f_sub=f_sub,
                   f_stellar_weighted=f_st, boot_mean=float(boots.mean()), boot_sd=float(boots.std()), p_strict=float(np.mean(boots_strict >= f_strict)) if strict else None,
                   boot_strict_sd=float(boots_strict.std()),
                   p=float(np.mean(boots >= f_sub)) if sub else None, isotropic=iso, params=dict(cone=poles_CONE, substellar=poles_SUBSTELLAR, stellar=poles_STELLAR),
                   substellar=[x for x in sub], aligned_substellar=[x for x in sub if x["aligned"]]), open(core.results(poles_OUT), "w"), indent=0)


def poles_record():
    d = json.load(open(core.results(poles_OUT)))
    core.record_search(
        "orbital_poles_galactic", "Orbital poles against the Galactic plane", archive="Gaia DR3 nss_two_body_orbit and binary_masses", release="Gaia DR3",
        selection="astrometric orbits (Orbital, OrbitalAlternative, OrbitalTargetedSearch, AstroSpectroSB1) with Thiele-Innes elements: %d, %d with companions that may be substellar (%d strictly)" % (
            d["n"], d["n_sub"], d["n_strict"]),
        code=os.path.abspath(__file__), entry="poles", n_targets=d["n_sub"], params=d["params"],
        null_summary="the %d stellar-mass companions' aligned fraction reweighted to the substellar hosts' Galactic latitudes: %.4f (bootstrap %.4f +- %.4f); isotropic %.4f" % (
            d["n_stellar"], d["f_stellar_weighted"], d["boot_mean"], d["boot_sd"], d["isotropic"]),
        sensitivity="an excess of aligned poles of a few per cent among the substellar companions",
        ledger="Orbital poles against the Galactic plane",
        result="companions that may be substellar with a candidate pole within %.0f deg of a Galactic pole: %.4f against the stellar companions' %.4f (p %.2f); the strict set %.4f (%d of %d, p %.2f at its size)" % (
            d["params"]["cone"], d["f_sub"], d["f_stellar_weighted"], d["p"] if d["p"] is not None else float("nan"), d["f_strict"], d["n_aligned_strict"], d["n_strict"],
            d["p_strict"] if d["p_strict"] is not None else float("nan")))
    core.clear_targets("orbital_poles_galactic")
    core.record_targets("orbital_poles_galactic", [dict(target="gaia_dr3:%s" % x["source_id"], stat=x["b_pole"], unit="Galactic latitude of the nearer candidate pole, deg",
                                                             vet="aligned within %.0f deg" % d["params"]["cone"] if x["aligned"] else None,
                                                             note="%s, P %.1f d, i %.0f deg, m2 %.3f-%s Msun" % (x["type"], x["period"] or 0, x["inc"], x["m2_lower"] or 0,
                                                                                                              "%.3f" % x["m2_upper"] if x["m2_upper"] is not None else "?")) for x in d["substellar"]])
    print("recorded %d substellar orbits" % d["n_sub"])


def poles_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": poles_fetch, "analyse": poles_analyse, "record": poles_record}[mode]()


hotsmall_DOC = r"""Hot bodies too small to be stars: sources within 100 pc at 7000 K or more with black-body radii below 0.003 R_sun, against the noise-propagated white-dwarf sequence.

    python -I seti.py neighbourhood hotsmall fetch
    python -I seti.py neighbourhood hotsmall analyse
    python -I seti.py neighbourhood hotsmall record"""

hotsmall_ROOT = os.environ.get("HOTSMALL_ROOT", core.data("hotsmall"))
hotsmall_OUT = "hot_small_bodies.json"
hotsmall_TEFF, hotsmall_FLOOR, hotsmall_WD = 7000.0, 0.003, 0.008
hotsmall_LAM_G, hotsmall_F0_G = 0.6400, 2.50e-9  # G pivot (um) and Vega flux density at G 0
hotsmall_LAM_BP, hotsmall_LAM_RP = 0.5110, 0.7770
hotsmall_SOURCE_COLS = ("source_id, ra, dec, parallax, parallax_over_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, teff_gspphot, "
                        "ruwe, ipd_frac_multi_peak, visibility_periods_used, astrometric_excess_noise, phot_bp_rp_excess_factor, non_single_star, "
                        "phot_variable_flag, distance_gspphot, logg_gspphot, ag_gspphot")
hotsmall_AP_COLS = ("radius_gspphot", "lum_flame", "radius_flame", "mass_flame", "classprob_dsc_combmod_whitedwarf", "classprob_dsc_combmod_star",
                    "classprob_dsc_combmod_binarystar", "spectraltype_esphs")


def hotsmall_fetch():
    """Nearby-star sources within 100 pc bluer than BP-RP 0.6 at 10-sigma parallax with parameters."""
    os.makedirs(hotsmall_ROOT, exist_ok=True)
    near = catalogues.gcns("source_id", "parallax > 10 AND phot_bp_mean_mag - phot_rp_mean_mag < 0.6", timeout=1800)
    print("%d nearby sources bluer than BP-RP 0.6" % len(near), flush=True)
    src = core.tap_by_ids("gaia", [r["source_id"] for r in near], "SELECT %s FROM gaiadr3.gaia_source WHERE source_id IN (%%s)" % hotsmall_SOURCE_COLS)
    rows = [r for r in src.values() if (num(r["parallax_over_error"]) or 0) > 10 and (num(r["parallax"]) or 0) > 10]
    print("%d at parallax 10 sigma" % len(rows), flush=True)
    extra = core.tap_by_ids("gaia", [r["source_id"] for r in rows],
                           "SELECT source_id, %s FROM gaiadr3.astrophysical_parameters WHERE source_id IN (%%s)" % ", ".join(hotsmall_AP_COLS))
    for r in rows:
        p = extra.get(str(r["source_id"]), {})
        for k in hotsmall_AP_COLS:
            r[k] = p.get(k)
    json.dump(rows, open(os.path.join(hotsmall_ROOT, "gaia_hot_100pc.json"), "w"))


def hotsmall_planck(lam_um, T):
    """B_lambda (erg/cm2/s/A/sr) at lam_um and T."""
    h, c, k = 6.626e-27, 2.998e10, 1.381e-16
    lam = lam_um * 1e-4
    return 2 * h * c ** 2 / lam ** 5 / math.expm1(h * c / (lam * k * T)) * 1e-8


def hotsmall_radius(G, plx_mas, T):
    """Black-body radius (R_sun) at T giving G at plx_mas."""
    f = hotsmall_F0_G * 10 ** (-0.4 * G)
    d_cm = 3.0857e18 * (1000.0 / plx_mas)
    return d_cm * math.sqrt(f / (math.pi * hotsmall_planck(hotsmall_LAM_G, T))) / 6.957e10


def hotsmall_colour_temperature(bprp):
    """Black-body temperature with BP-RP colour bprp; None beyond 100000 K."""
    def colour(T):
        v = hotsmall_planck(hotsmall_LAM_BP, 9602.0) / hotsmall_planck(hotsmall_LAM_RP, 9602.0)  # Vega as a 9602 K black body
        return -2.5 * math.log10(hotsmall_planck(hotsmall_LAM_BP, T) / hotsmall_planck(hotsmall_LAM_RP, T) / v)
    lo, hi = 2500.0, 100000.0
    if bprp <= colour(hi):
        return None
    if bprp >= colour(lo):
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if colour(mid) > bprp:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


hotsmall_WD_CATS = {"J/MNRAS/508/3877": "Gentile Fusillo et al. (2021)", "J/MNRAS/482/4570": "Gentile Fusillo et al. (2019)", "J/ApJ/934/36": "Bergeron et al. (2022), IR-faint",
                    "J/ApJ/898/84": "Kilic et al. (2020)", "J/MNRAS/486/2169": "Jimenez-Esteban et al. (2018)"}


def hotsmall_vet(x):
    """Catalogues already holding a candidate (Gaia WD, IR-faint WDs, GALEX); verdict in x['vet']."""
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    x["wd_catalogues"], x["galex_nuv"] = [], None
    try:
        res = Vizier(row_limit=3, timeout=120).query_region(SkyCoord(x["ra"], x["dec"], unit="deg"), radius=3 * u.arcsec)
    except Exception:
        x["vet"] = "catalogue query failed"
        return
    for k in res.keys():
        for cat, name in hotsmall_WD_CATS.items():
            if k.startswith(cat) and name not in x["wd_catalogues"]:
                x["wd_catalogues"].append(name)
        if k.startswith("II/335") or k.startswith("II/312"):
            t = res[k]
            col = [c for c in t.colnames if c.lower() in ("nuvmag", "nuv")]
            if col and str(t[col[0]][0]) not in ("--", ""):
                x["galex_nuv"] = float(t[col[0]][0])
    if any("IR-faint" in n for n in x["wd_catalogues"]):
        x["vet"] = "an IR-faint white dwarf (Bergeron et al. 2022): collision-induced absorption, not a hot body"
    elif x["wd_catalogues"]:
        x["vet"] = "a catalogued white dwarf (%s)%s" % (x["wd_catalogues"][0], ", NUV %.1f" % x["galex_nuv"] if x["galex_nuv"] is not None else " with no ultraviolet detection")
    elif x["galex_nuv"] is not None:
        x["vet"] = "uncatalogued, NUV %.1f" % x["galex_nuv"]
    else:
        x["vet"] = "uncatalogued, no ultraviolet detection"


def hotsmall_analyse():
    rows = json.load(open(os.path.join(hotsmall_ROOT, "gaia_hot_100pc.json")))
    out = []
    # colour temperature is primary; GSP-Phot mis-fits white dwarfs
    for r in rows:
        plx, G, bprp, teff = num(r["parallax"]), num(r["phot_g_mean_mag"]), num(r["bp_rp"]), num(r["teff_gspphot"])
        if not plx or G is None or bprp is None:
            continue
        t_col = hotsmall_colour_temperature(bprp)
        if not t_col or t_col < hotsmall_TEFF:
            continue
        R_col = hotsmall_radius(G, plx, t_col)
        R_gsp = hotsmall_radius(G, plx, teff) if teff else None
        clean = (num(r["ruwe"]) or 9) < 1.4 and (num(r["ipd_frac_multi_peak"]) or 0) < 2 and (num(r["visibility_periods_used"]) or 0) >= 10 and (num(r["astrometric_excess_noise"]) or 0) < 1.0
        ex = num(r["phot_bp_rp_excess_factor"])
        single = ex is not None and ex < 1.3 + 0.06 * bprp ** 2
        out.append(dict(source_id=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"]), plx=plx, plx_over_error=num(r["parallax_over_error"]), G=G, bprp=bprp, teff=teff, t_colour=t_col,
                        R_gspphot=R_gsp, R_colour=R_col, radius_gspphot=num(r["radius_gspphot"]), lum_flame=num(r["lum_flame"]), p_wd=num(r["classprob_dsc_combmod_whitedwarf"]),
                        clean=clean, single=single, ruwe=num(r["ruwe"]), excess_factor=ex,
                        candidate=R_col < hotsmall_FLOOR and (R_gsp is None or R_gsp < hotsmall_FLOOR) and clean and single))
    R = np.array([x["R_colour"] for x in out])
    wd = [x for x in out if x["clean"] and x["single"] and hotsmall_WD * 0.7 <= x["R_colour"] <= 0.03]
    # null: noise-propagated white-dwarf radii, 200 times
    rng = np.random.default_rng(8)
    below = 0
    for _ in range(200):
        for x in wd:
            plx = x["plx"] * (1 + rng.normal(0, 1 / x["plx_over_error"]))
            if plx > 0 and hotsmall_radius(x["G"] + rng.normal(0, 0.003), plx, x["t_colour"]) < hotsmall_FLOOR:
                below += 1
    cand = [x for x in out if x["candidate"]]
    for x in cand:
        hotsmall_vet(x)
    print("%d Gaia sources within 100 pc at parallax 10 sigma with colour temperatures above %.0f K; black-body radii at the colour temperature: median %.3f Rsun, %d below %.3f; "
          "%d of those also below it at the gspphot temperature where one exists, with clean astrometry and a single-source BP/RP excess; the white-dwarf sequence's %d members (%.3f-0.03 Rsun) pushed through their errors give %.2f below the floor" % (
              len(out), hotsmall_TEFF, np.median(R), int(np.sum(R < hotsmall_FLOOR)), hotsmall_FLOOR, len(cand), len(wd), hotsmall_WD * 0.7, below / 200.0))
    for x in sorted(cand, key=lambda x: x["R_colour"])[:30]:
        print("  Gaia DR3 %s (%.5f %+.5f): G %.2f BP-RP %+.2f plx %.1f mas; colour %.0f K (R %.4f Rsun), Teff gspphot %s (R %s); gspphot radius %s, P(WD) %s, RUWE %.2f, excess factor %.2f; %s" % (
            x["source_id"], x["ra"], x["dec"], x["G"], x["bprp"], x["plx"], x["t_colour"], x["R_colour"], "%.0f K" % x["teff"] if x["teff"] else "-", "%.4f" % x["R_gspphot"] if x["R_gspphot"] else "-",
            "%.4f" % x["radius_gspphot"] if x["radius_gspphot"] else "-", "%.2f" % x["p_wd"] if x["p_wd"] is not None else "-", x["ruwe"] or 0, x["excess_factor"] or 0, x.get("vet", "")))
    json.dump(dict(n=len(out), n_below_colour=int(np.sum(R < hotsmall_FLOOR)), n_cand=len(cand), n_wd=len(wd), null_below=below / 200.0, cands=cand,
                   params=dict(teff=hotsmall_TEFF, floor=hotsmall_FLOOR)), open(core.results(hotsmall_OUT), "w"), indent=0)


def hotsmall_record():
    d = json.load(open(core.results(hotsmall_OUT)))
    core.record_search(
        "hot_small_bodies", "Hot bodies too small to be stars", archive="Gaia DR3 gaia_source and astrophysical_parameters", release="Gaia DR3",
        selection="sources within 100 pc with parallax_over_error above 10 and black-body colour temperatures above %.0f K (%d)" % (d["params"]["teff"], d["n"]), code=os.path.abspath(__file__), entry="hotsmall",
        n_targets=d["n"], params=d["params"], null_summary="the %d white dwarfs of the sample pushed through their parallax and flux errors 200 times: %.2f below the floor" % (d["n_wd"], d["null_below"]),
        sensitivity="a black-body radius below %.3f Rsun at the colour temperature and at the gspphot temperature where one exists, within 100 pc" % d["params"]["floor"], ledger="Hot bodies too small to be stars",
        result="%d below the floor at the colour temperature, %d also at the gspphot temperature with clean astrometry and single-source photometry: %d catalogued white dwarfs (%d of them IR-faint ones of Bergeron et al. 2022), %d uncatalogued%s" % (
            d["n_below_colour"], d["n_cand"], sum(1 for x in d["cands"] if x.get("wd_catalogues")), sum(1 for x in d["cands"] if "IR-faint" in x.get("vet", "")),
            sum(1 for x in d["cands"] if not x.get("wd_catalogues")),
            (": " + "; ".join("Gaia DR3 %s (%.4f Rsun at %.0f K, %s)" % (x["source_id"], x["R_colour"], x["t_colour"], x.get("vet", "")) for x in d["cands"] if not x.get("wd_catalogues"))) if any(not x.get("wd_catalogues") for x in d["cands"]) else ""))
    core.clear_targets("hot_small_bodies")
    core.record_targets("hot_small_bodies", [dict(target="gaia_dr3:%s" % x["source_id"], stat=x["R_colour"], unit="black-body radius at the colour temperature, solar radii", vet=x.get("vet"),
                                                       note="G %.2f, BP-RP %+.2f, plx %.1f mas, colour %.0f K, Teff gspphot %s" % (x["G"], x["bprp"], x["plx"], x["t_colour"], "%.0f K" % x["teff"] if x["teff"] else "-")) for x in d["cands"]])
    print("recorded %d candidates" % d["n_cand"])


def hotsmall_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": hotsmall_fetch, "analyse": hotsmall_analyse, "record": hotsmall_record}[mode]()


# --------------------------------------------------------------------------------------------------------------
# the nearby stars' velocities, classified spectra, past and catalogues


def nearby_rv():
    """Gaia DR3 stars within 100 pc with radial velocities and parallaxes at 10 sigma, cached under DATA: (rows, columns). An asynchronous query, since a synchronous one on gaia_source stops at the archive's time cap with part of the rows."""
    p = core.data("gaia_rv_100pc.csv")
    if not os.path.exists(p):
        rows = core.tap_async("gaia", "SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, "
                                      "radial_velocity, radial_velocity_error, phot_g_mean_mag, bp_rp, ruwe FROM gaiadr3.gaia_source "
                                      "WHERE parallax > 10 AND parallax_over_error > 10 AND radial_velocity IS NOT NULL", maxrec=500000)
        with open(p + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(p + ".tmp", p)
    R = list(csv.DictReader(open(p)))
    keys = ("ra", "dec", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "radial_velocity",
            "radial_velocity_error", "phot_g_mean_mag", "bp_rp", "ruwe")
    return R, {k: np.array([num(r[k], np.nan) for r in R], float) for k in keys}


def nearby_velocity(c):
    """Velocity relative to the Sun (km/s) along RA, Dec and the line of sight, with each component's error."""
    k = 4.74047 / c["parallax"]
    ea = k * np.hypot(c["pmra_error"], c["pmra"] * c["parallax_error"] / c["parallax"])
    ed = k * np.hypot(c["pmdec_error"], c["pmdec"] * c["parallax_error"] / c["parallax"])
    return k * c["pmra"], k * c["pmdec"], c["radial_velocity"], ea, ed, c["radial_velocity_error"]


def nearby_astrometry():
    """Gaia DR3 stars within 100 pc with parallaxes at 10 sigma, with or without radial velocities, cached under DATA: (rows, columns), an asynchronous query as in `nearby_rv`."""
    p = core.data("gaia_100pc_astrometry.csv")
    if not os.path.exists(p):
        rows = core.tap_async("gaia", "SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, "
                                      "radial_velocity, radial_velocity_error, phot_g_mean_mag, bp_rp, ruwe FROM gaiadr3.gaia_source "
                                      "WHERE parallax > 10 AND parallax_over_error > 10", maxrec=1000000)
        with open(p + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(p + ".tmp", p)
    R = list(csv.DictReader(open(p)))
    keys = ("ra", "dec", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "radial_velocity",
            "radial_velocity_error", "phot_g_mean_mag", "bp_rp", "ruwe")
    return R, {k: np.array([num(r[k], np.nan) for r in R], float) for k in keys}


def nearby_axes(c):
    """Each star's unit vectors toward it and along increasing RA and Dec (ICRS Cartesian), and its position (pc)."""
    a, d = np.radians(c["ra"]), np.radians(c["dec"])
    rhat = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], 1)
    ahat = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], 1)
    dhat = np.stack([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)], 1)
    return rhat, ahat, dhat, rhat * (1000.0 / c["parallax"])[:, None]


rest_DOC = r"""Stars at rest with the Sun: Gaia DR3 stars within 100 pc with radial velocities and parallaxes at 10 sigma whose velocity relative to the Sun is below 1 km/s with its error below 0.5 km/s, a companion keeping pace with the Solar System, against the count a velocity density flat near zero gives from the like-measured stars below 5 km/s.

`null` asks how special the Sun's velocity is: the same count in 1 km/s balls centred elsewhere in velocity space where the density within 5 km/s matches the Sun's (RUWE below 1.4 throughout), stars and systems both, a system being stars linked within 10 pc and 1 km/s; the sideways speeds of every star within 100 pc, radial velocity or not, below 0.25, 0.5 and 1 km/s against the flat two-dimensional density; and further members of each system of the Sun's ball, stars within 15 pc of it whose proper motions match the motion the system's velocity predicts at their positions to 1 km/s, against the count within 5 km/s scaled by area.

    python -I seti.py neighbourhood rest [null|record]"""

# a system: stars within this distance (pc) and velocity difference (km/s) of another member
rest_LINK = (10.0, 1.0)
rest_GROUP_PC = 15.0
# the coronae and associations within 200 pc of Moranta et al. (2022), VizieR
rest_CRIUS = "J/ApJ/939/94"


def rest_systems(P, V):
    """Systems among stars with positions P (pc) and velocities V (km/s): friends of friends within rest_LINK."""
    n = len(P)
    lab = list(range(n))

    def root(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(P[i] - P[j]) < rest_LINK[0] and np.linalg.norm(V[i] - V[j]) < rest_LINK[1]:
                lab[root(i)] = root(j)
    out = {}
    for i in range(n):
        out.setdefault(root(i), []).append(i)
    return list(out.values())


def rest_null():
    from scipy.spatial import cKDTree
    from scipy.stats import poisson
    R, c = nearby_rv()
    va, vd, vr, ea, ed, er = nearby_velocity(c)
    e = np.sqrt(ea ** 2 + ed ** 2 + er ** 2)
    good = np.isfinite(vr) & (e < 0.5) & (c["ruwe"] < 1.4)
    rhat, ahat, dhat, pos = nearby_axes(c)
    vel = rhat * vr[:, None] + ahat * va[:, None] + dhat * vd[:, None]
    P, V = pos[good], vel[good]
    ids = np.array([r["source_id"] for r in R])[good]
    tree = cKDTree(V)
    n5 = len(tree.query_ball_point(np.zeros(3), 5.0))
    inner = sorted(tree.query_ball_point(np.zeros(3), 1.0), key=lambda i: np.linalg.norm(V[i]))
    mu = n5 / 125.0
    p_sun = float(poisson.sf(len(inner) - 1, mu))
    sys_sun = rest_systems(P[inner], V[inner])
    print("%d stars with RUWE below 1.4 and velocities to 0.5 km/s; at the Sun's velocity %d within 5 km/s, %d within 1 km/s against %.2f "
          "(Poisson p %.4f) in %d systems" % (int(good.sum()), n5, len(inner), mu, p_sun, len(sys_sun)), flush=True)
    # balls centred elsewhere: star velocities moved by 5 km/s in each axis, kept beyond 10 km/s of the Sun's and at its density
    rng = np.random.default_rng(17)
    cand = V[rng.integers(0, len(V), 400000)] + rng.normal(0.0, 5.0, (400000, 3))
    cand = cand[np.linalg.norm(cand, axis=1) > 10.0]
    m5 = tree.query_ball_point(cand, 5.0, return_length=True)
    keep = np.abs(m5 - n5) <= 0.2 * n5
    cand, m5 = cand[keep], m5[keep]
    m1 = tree.query_ball_point(cand, 1.0, return_length=True)
    p = poisson.sf(m1 - 1, m5 / 125.0)
    n_sys = np.zeros(len(cand), int)
    for k in np.nonzero(m1 >= len(sys_sun))[0]:
        idx = tree.query_ball_point(cand[k], 1.0)
        n_sys[k] = len(rest_systems(P[idx], V[idx]))
    f_star = float(np.mean(p <= p_sun))
    f_sys = float(np.mean(n_sys >= len(sys_sun)))
    print("%d balls elsewhere at the Sun's density (%d-%d within 5 km/s): mean %.2f within 1 km/s; Poisson p at or below the Sun's in "
          "%.4f of them, %d or more systems in %.4f" % (len(cand), int(m5.min()), int(m5.max()), float(m1.mean()), f_star,
                                                       len(sys_sun), f_sys), flush=True)
    # sideways speeds of every star within 100 pc
    R2, c2 = nearby_astrometry()
    va2, vd2, vr2, ea2, ed2, er2 = nearby_velocity(c2)
    vt, et = np.hypot(va2, vd2), np.hypot(ea2, ed2)
    good2 = np.isfinite(vt) & (et < 0.1) & (c2["ruwe"] < 1.4)
    t5 = int(np.sum(good2 & (vt < 5)))
    side = [dict(below=x, n=int(np.sum(good2 & (vt < x))), expect=t5 * (x / 5.0) ** 2,
                 p=float(poisson.sf(int(np.sum(good2 & (vt < x))) - 1, t5 * (x / 5.0) ** 2))) for x in (0.25, 0.5, 1.0)]
    print("%d stars within 100 pc, %d with sideways speeds to 0.1 km/s and RUWE below 1.4, %d below 5 km/s: %s" % (
        len(R2), int(good2.sum()), t5, "; ".join("below %.2f km/s %d against %.1f (p %.3f)" % (s["below"], s["n"], s["expect"], s["p"]) for s in side)))
    # further members of each multi-star system of the Sun's ball
    rhat2, ahat2, dhat2, pos2 = nearby_axes(c2)
    groups = []
    for s in sys_sun:
        if len(s) < 2:
            continue
        Pg, Vg = P[inner][s].mean(0), V[inner][s].mean(0)
        own = np.isin(np.array([r["source_id"] for r in R2]), [str(ids[inner][j]) for j in s])
        near = (np.linalg.norm(pos2 - Pg, axis=1) < rest_GROUP_PC) & ~own
        pa, pd = ahat2 @ Vg, dhat2 @ Vg
        dt = np.hypot(va2 - pa, vd2 - pd)
        ok = near & np.isfinite(dt) & (et < 0.25) & (c2["ruwe"] < 1.4)
        members = [i for i in np.nonzero(ok & (dt < 1.0))[0]]
        expect = float(np.sum(ok & (dt < 5.0))) / 25.0
        rows = []
        for i in sorted(members, key=lambda i: dt[i]):
            rvp = float(rhat2[i] @ Vg)
            rows.append(dict(source_id=R2[i]["source_id"], dist_pc=float(np.linalg.norm(pos2[i] - Pg)), dvt=float(dt[i]),
                             g=float(c2["phot_g_mean_mag"][i]), rv=None if not np.isfinite(vr2[i]) else float(vr2[i]),
                             rv_err=None if not np.isfinite(er2[i]) else float(er2[i]), rv_predicted=rvp))
        groups.append(dict(stars=[str(ids[inner][j]) for j in s], centre_pc=Pg.tolist(), velocity=Vg.tolist(), n_near=int(np.sum(ok)),
                           members=rows, expect=expect, p=float(poisson.sf(len(rows) - 1, expect)) if rows else 1.0))
        print("system %s: %d other stars within %.0f pc with sideways speeds to 0.25 km/s; %d match its motion to 1 km/s against %.2f by "
              "area (Poisson p %.3f)" % (", ".join(groups[-1]["stars"]), int(np.sum(ok)), rest_GROUP_PC, len(rows), expect, groups[-1]["p"]))
        for r in rows:
            print("  Gaia DR3 %s: %.1f pc from the centre, %.2f km/s off its motion, G %.1f, RV %s (the system's motion predicts %+.2f)" % (
                r["source_id"], r["dist_pc"], r["dvt"], r["g"], "%+.2f +- %.2f" % (r["rv"], r["rv_err"]) if r["rv"] is not None else "-",
                r["rv_predicted"]))
    # catalogued associations of the slow stars (Moranta et al. 2022) with their mean velocities relative to the Sun
    slow = [str(ids[i]) for i in inner]
    crius = {}
    for r in core.tap("vizier", 'SELECT Crius, Cl, Gaia FROM "%s/table1" WHERE Gaia IN (%s)' % (rest_CRIUS, ",".join(slow))):
        g = core.tap("vizier", 'SELECT Crius, Nstar, Dist, U, V, W, "U-mad", "V-mad", "W-mad" FROM "%s/%s" WHERE Crius = %s'
                     % (rest_CRIUS, "table2" if r["Cl"] == "P" else "table3", r["Crius"]))
        if g:
            g = g[0]
            uvw = [float(g[k]) for k in ("U", "V", "W")]
            crius[str(r["Gaia"])] = dict(crius=int(r["Crius"]), n=int(g["Nstar"]), dist_pc=float(g["Dist"]), uvw=uvw,
                                         speed=float(np.linalg.norm(uvw)), mad=[float(g[k]) for k in ("U-mad", "V-mad", "W-mad")])
    for s, g in crius.items():
        print("Gaia DR3 %s: a member of Crius %d (Moranta et al. 2022), %d stars at %.0f pc moving at (%+.1f, %+.1f, %+.1f) km/s relative "
              "to the Sun, %.2f km/s, spread %s" % (s, g["crius"], g["n"], g["dist_pc"], *g["uvw"], g["speed"], g["mad"]))
    out = dict(n_good=int(good.sum()), n5=n5, k=len(inner), mu=mu, p_sun=p_sun,
               stars=[dict(source_id=str(ids[i]), v=float(np.linalg.norm(V[i])), pos=P[i].tolist(), vel=V[i].tolist()) for i in inner],
               systems=[[str(ids[inner][j]) for j in s] for s in sys_sun], balls=len(cand), mean_k=float(m1.mean()),
               f_star=f_star, f_sys=f_sys, sideways=side, sideways_n=int(good2.sum()), groups=groups, crius=crius)
    json.dump(out, open(core.results("rest_null.json"), "w"), indent=1)


def rest_main(record=False):
    from scipy.stats import poisson
    R, c = nearby_rv()
    va, vd, vr, ea, ed, er = nearby_velocity(c)
    v = np.sqrt(va ** 2 + vd ** 2 + vr ** 2)
    e = np.sqrt(ea ** 2 + ed ** 2 + er ** 2)
    good = np.isfinite(v) & (e < 0.5)
    n5 = int(np.sum(good & (v < 5)))
    expect = n5 / 125.0
    slow = sorted(np.nonzero(good & (v < 1))[0], key=lambda i: v[i])
    p = float(poisson.sf(len(slow) - 1, expect)) if slow else 1.0
    print("%d stars, %d with the velocity's error below 0.5 km/s; %d of those below 5 km/s, so %.2f expected below 1 km/s; "
          "found %d (Poisson p %.3f)" % (len(R), int(good.sum()), n5, expect, len(slow), p))
    for i in slow:
        print("  Gaia DR3 %s: %.2f +- %.2f km/s at %.1f pc, G %.1f, RUWE %.2f" % (R[i]["source_id"], v[i], e[i], 1000 / c["parallax"][i],
                                                                            c["phot_g_mean_mag"][i], c["ruwe"][i]))
    out = [dict(source_id=R[i]["source_id"], v=float(v[i]), e=float(e[i]), d=float(1000 / c["parallax"][i])) for i in slow]
    json.dump(dict(n=len(R), n_good=int(good.sum()), n5=n5, expect=expect, p=p, slow=out), open(core.results("rest_stars.json"), "w"), indent=1)
    if record:
        nl = json.load(open(core.results("rest_null.json"))) if os.path.exists(core.results("rest_null.json")) else None
        system_of = {s: grp for grp in (nl["systems"] if nl else []) for s in grp}
        crius = nl.get("crius", {}) if nl else {}
        # stars of one moving group are not independent: the systems decide
        chance = p > 0.01 or (nl is not None and nl["f_sys"] > 0.01)

        def vet(o):
            codes = (["moving-group"] if o["source_id"] in crius else []) + (["chance-rate"] if chance else [])
            return "; ".join(codes) or None

        def note(o):
            s = "+- %.2f km/s at %.1f pc" % (o["e"], o["d"])
            if len(system_of.get(o["source_id"], [])) > 1:
                s += "; one system with %s" % ", ".join("Gaia DR3 %s" % x for x in system_of[o["source_id"]] if x != o["source_id"])
            g = crius.get(o["source_id"])
            if g:
                s += "; a member of Crius %d (Moranta et al. 2022), %d stars at %.0f pc moving at %.1f km/s relative to the Sun" % (
                    g["crius"], g["n"], g["dist_pc"], g["speed"])
            return s
        result = "%d below 1 km/s against %.2f expected (Poisson p %.3f)" % (len(slow), expect, p)
        if nl:
            result += ("; with RUWE below 1.4, %d against %.2f (p %.4f) in %d systems, matched or exceeded by %.3f of %d balls of the same "
                       "density elsewhere in velocity space (%.3f for the systems); sideways speeds of %d stars without radial velocities "
                       "needed: %s" % (nl["k"], nl["mu"], nl["p_sun"], len(nl["systems"]), nl["f_star"], nl["balls"], nl["f_sys"],
                                       nl["sideways_n"], ", ".join("%d below %.2f km/s against %.1f" % (s["n"], s["below"], s["expect"])
                                                                   for s in nl["sideways"])))
            for g in nl["groups"]:
                result += "; %d other stars within %.0f pc share the motion of the system %s to 1 km/s against %.2f by area" % (
                    len(g["members"]), rest_GROUP_PC, " and ".join(g["stars"]), g["expect"])
            for s, g in sorted(crius.items()):
                result += "; Gaia DR3 %s is a member of Crius %d (Moranta et al. 2022), %d stars at %.0f pc moving at (%+.1f, %+.1f, %+.1f) km/s" % (
                    s, g["crius"], g["n"], g["dist_pc"], *g["uvw"])
        core.record_search(
            "gaia_rest_stars", "Stars at rest with the Sun", archive="Gaia DR3 gaia_source", release="DR3",
            selection="parallax above 10 mas at 10 sigma with a radial velocity: %d stars, %d with the velocity's error below 0.5 km/s"
                      % (len(R), int(good.sum())),
            code=os.path.abspath(__file__), entry="rest", n_targets=int(good.sum()), n_unit="stars",
            params=dict(v_kms=1.0, err_kms=0.5, flat_from_kms=5.0, ruwe_null=1.4, link_pc=rest_LINK[0], link_kms=rest_LINK[1],
                        group_pc=rest_GROUP_PC),
            null_summary="a velocity density flat near zero: %d stars below 5 km/s give %.2f below 1 km/s%s" % (
                n5, expect, "; balls of the same density elsewhere in velocity space" if nl else ""),
            sensitivity="a star within 100 pc keeping pace with the Sun to 1 km/s, with radial velocity and astrometry good to 0.5 km/s",
            ledger="Stars at rest with the Sun", result=result)
        core.clear_targets("gaia_rest_stars")
        core.record_targets("gaia_rest_stars", [dict(
            target="gaia_dr3:%s" % o["source_id"], stat=o["v"], unit="speed relative to the Sun, km/s", vet=vet(o), note=note(o))
            for o in out])
        print("recorded")


def rest_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if mode == "null":
        rest_null()
    else:
        rest_main(record=mode == "record")


headon_DOC = r"""Stars on a straight line to the Sun: Gaia DR3 stars within 100 pc moving sideways at under 1 km/s (error below 0.5 km/s) at a radial speed above 20 km/s, an object aimed at the Solar System, each with its least distance to the Sun and the time of it on a straight line, against the count a sideways-velocity density flat near zero gives; the published encounter lists (Bailer-Jones et al. 2022) cover the same stars.

    python -I seti.py neighbourhood headon [record] [--vt-kms 1] [--err-kms 0.5] [--vr-kms 20] [--flat-from-kms 5]"""


def headon_main(record=False, vt_kms=1.0, err_kms=0.5, vr_kms=20.0, flat_from_kms=5.0):
    R, c = nearby_rv()
    va, vd, vr, ea, ed, er = nearby_velocity(c)
    vt = np.hypot(va, vd)
    et = np.hypot(ea, ed)
    v = np.hypot(vt, vr)
    d = 1000 / c["parallax"]
    fast = np.isfinite(vt) & (et < err_kms) & (np.abs(vr) > vr_kms)
    expect = float(np.sum(fast & (vt < flat_from_kms))) / (flat_from_kms / vt_kms) ** 2
    sel = sorted(np.nonzero(fast & (vt < vt_kms))[0], key=lambda i: vt[i] / v[i] * d[i])
    out = []
    for i in sel:
        miss_au = float(vt[i] / v[i] * d[i] * 206264.8)
        t_myr = float(-d[i] * vr[i] / v[i] ** 2 * 0.97779)
        out.append(dict(source_id=R[i]["source_id"], miss_au=miss_au, t_myr=t_myr, d=float(d[i]), rv=float(vr[i]), vt=float(vt[i])))
    print("%d stars with radial speed above 20 km/s and sideways speed measured to 0.5 km/s; below 1 km/s sideways %d against "
          "%.1f from a flat density" % (int(fast.sum()), len(out), expect))
    for o in out[:10]:
        print("  Gaia DR3 %s: %.0f pc, %+.0f km/s, sideways %.2f km/s; least distance %.0f au %s %.2f Myr" % (
            o["source_id"], o["d"], o["rv"], o["vt"], o["miss_au"], "in" if o["t_myr"] > 0 else "ago,", abs(o["t_myr"])))
    json.dump(dict(n=int(fast.sum()), expect=expect, stars=out), open(core.results("headon_stars.json"), "w"), indent=1)
    if record:
        core.record_search(
            "gaia_headon_stars", "Stars on a straight line to the Sun", archive="Gaia DR3 gaia_source", release="DR3",
            selection="parallax above 10 mas at 10 sigma with a radial velocity above 20 km/s in either sense, sideways speed measured "
                      "to 0.5 km/s: %d stars" % int(fast.sum()),
            code=os.path.abspath(__file__), entry="headon", n_targets=int(fast.sum()), n_unit="stars",
            params=dict(vt_kms=vt_kms, err_kms=err_kms, vr_kms=vr_kms, flat_from_kms=flat_from_kms),
            null_summary="a sideways-velocity density flat near zero: %.1f stars below 1 km/s" % expect,
            sensitivity="an object within 100 pc on a line through the Sun to 1 km/s in 20",
            ledger="Stars on a straight line to the Sun",
            result="%d stars against %.1f expected; least distances from %.0f au" % (len(out), expect, out[0]["miss_au"] if out else 0))
        core.clear_targets("gaia_headon_stars")
        core.record_targets("gaia_headon_stars", [dict(
            target="gaia_dr3:%s" % o["source_id"], stat=o["miss_au"], unit="least distance to the Sun on a straight line, au",
            vet="chance-rate" if len(out) <= expect + 3 * math.sqrt(max(expect, 1)) else None,
            note="%.0f pc, %+.0f km/s, %s %.2f Myr" % (o["d"], o["rv"], "in" if o["t_myr"] > 0 else "ago", abs(o["t_myr"])))
            for o in out])
        print("recorded")


qsolike_DOC = r"""Non-stellar spectra at stellar distances: sources of the Gaia Catalogue of Nearby Stars beyond parallax 10 mas that Gaia DR3's discrete source classifier calls quasar or galaxy at 0.9, light that is not a star's at a star's distance, each read in SIMBAD and placed on the colour-magnitude diagram, where white dwarfs, white-dwarf binaries and cataclysmic variables explain it.

    python -I seti.py neighbourhood qsolike [record] [--prob 0.9] [--plx-mas 10]"""


def qsolike_main(record=False, prob=0.9, plx_mas=10.0):
    rows = core.tap("gaia", "SELECT g.source_id, g.parallax, g.parallax_error, g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, "
                            "ap.classprob_dsc_combmod_quasar AS q, ap.classprob_dsc_combmod_galaxy AS gal "
                            "FROM external.gaiaedr3_gcns_main_1 AS g JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id "
                            "WHERE g.parallax > %g AND (ap.classprob_dsc_combmod_quasar > %g OR ap.classprob_dsc_combmod_galaxy > %g)" % (plx_mas, prob, prob),
                    timeout=600)
    ids = ["Gaia DR3 %s" % r["source_id"] for r in rows]
    types = {}
    for k in range(0, len(ids), 200):
        q = ("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)"
             % ",".join("'%s'" % x for x in ids[k:k + 200]))
        for s in core.simbad(q):
            types[str(s["id"])] = (str(s["main_id"]), str(s["otype"]).strip(), s["sp_type"])
    out = []
    for r in rows:
        plx, G = float(r["parallax"]), num(r["phot_g_mean_mag"])
        bprp = (num(r["phot_bp_mean_mag"]) or np.nan) - (num(r["phot_rp_mean_mag"]) or np.nan)
        MG = G + 5 * math.log10(plx / 100) if G is not None else np.nan
        name, otype, sp = types.get("Gaia DR3 %s" % r["source_id"], (None, None, None))
        vet = []
        if otype and ("WD" in otype or (sp and str(sp).startswith("D"))):
            vet.append("white-dwarf")
        elif np.isfinite(MG) and np.isfinite(bprp) and MG > 10 + 3 * bprp:
            vet.append("white-dwarf")
        if otype in ("**", "SB*", "EB*"):
            vet.append("binary")
        # a component of a resolved multiple: its BP and RP windows hold the companion's light
        if name and re.search(r"\d[A-C]$|\s[A-C]$", name.strip()):
            vet.append("blend")
        if otype and "CV" in otype:
            vet.append("natural-type")
        if num(r["parallax_error"]) and plx / float(r["parallax_error"]) < 5:
            vet.append("astrometry")
        if G is None:
            vet.append("artefact")
        out.append(dict(source_id=r["source_id"], plx=plx, G=G, bprp=None if not np.isfinite(bprp) else float(bprp),
                        MG=None if not np.isfinite(MG) else float(MG), q=num(r["q"]), gal=num(r["gal"]), simbad=name, otype=otype,
                        sp=sp, vet=vet))
    # the rest: a catalogued neighbour within 10 arcsec puts its light in the BP and RP windows
    for o in out:
        if o["vet"]:
            continue
        g = core.tap("gaia", "SELECT ra, dec FROM gaiadr3.gaia_source WHERE source_id = %s" % o["source_id"])[0]
        near = core.simbad("SELECT main_id, otype, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', %s, %s)) * 3600 AS sep FROM basic "
                           "WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %s, %s, %f)) = 1" % (g["ra"], g["dec"], g["ra"], g["dec"], 10 / 3600))
        if near:
            o["vet"].append("blend")
            o["neighbours"] = ["%s (%s) %.1f arcsec" % (" ".join(str(x["main_id"]).split()), str(x["otype"]).strip(), float(x["sep"])) for x in near]
    left = [o for o in out if not o["vet"]]
    print("%d nearby sources the classifier calls quasar or galaxy at 0.9; %d explained (white dwarfs %d, binaries %d, components of "
          "multiples %d, CVs %d, astrometry %d, no photometry %d); left %d" % (
              len(out), len(out) - len(left), sum("white-dwarf" in o["vet"] for o in out), sum("binary" in o["vet"] for o in out),
              sum("blend" in o["vet"] for o in out), sum("natural-type" in o["vet"] for o in out),
              sum("astrometry" in o["vet"] for o in out), sum("artefact" in o["vet"] for o in out), len(left)))
    for o in sorted(left, key=lambda o: -o["plx"])[:15]:
        print("  Gaia DR3 %s: %.1f mas, G %.1f, BP-RP %s, M_G %s, quasar %.2f, galaxy %.2f, SIMBAD %s %s" % (
            o["source_id"], o["plx"], o["G"] or 0, o["bprp"], o["MG"], o["q"] or 0, o["gal"] or 0, o["simbad"], o["otype"]))
    json.dump(out, open(core.results("qsolike_nearby.json"), "w"), indent=1, default=str)
    if record:
        core.record_search(
            "gaia_qsolike_nearby", "Non-stellar spectra at stellar distances", archive="Gaia DR3 astrophysical_parameters with the "
            "Gaia Catalogue of Nearby Stars; SIMBAD", release="DR3",
            selection="Gaia Catalogue of Nearby Stars beyond parallax 10 mas, discrete-source-classifier quasar or galaxy probability above 0.9",
            code=os.path.abspath(__file__), entry="qsolike", n_targets=len(out), n_unit="Gaia sources",
            params=dict(classprob=prob, plx_mas=plx_mas, wd_locus="M_G > 10 + 3 (BP-RP)"),
            null_summary="the classifier's known confusions: white dwarfs, white-dwarf binaries and cataclysmic variables",
            sensitivity="a source within 100 pc whose BP/RP spectrum the classifier reads as a quasar's or a galaxy's",
            ledger="Non-stellar spectra at stellar distances",
            result="%d sources, %d left after white dwarfs, binaries, CVs and astrometry" % (len(out), len(left)))
        core.clear_targets("gaia_qsolike_nearby")
        core.record_targets("gaia_qsolike_nearby", [dict(
            target="gaia_dr3:%s" % o["source_id"], stat=max(o["q"] or 0, o["gal"] or 0), unit="classifier's quasar or galaxy probability",
            vet="; ".join(o["vet"]) or None, note="M_G %s, BP-RP %s, SIMBAD %s %s%s" % (
                "%.1f" % o["MG"] if o["MG"] is not None else "-", "%.2f" % o["bprp"] if o["bprp"] is not None else "-", o["simbad"], o["otype"],
                ", within 10 arcsec " + ", ".join(o["neighbours"]) if o.get("neighbours") else ""))
            for o in out])
        print("recorded")


no2mass_DOC = r"""Light that appeared since 2MASS: sources of the Gaia Catalogue of Nearby Stars within 20 pc at G 5-13 without a 2MASS best neighbour in Gaia DR3, a source switched on since 1999, each sought again in the 2MASS point-source catalogue around its position carried back to 2000 (CDS XMatch, II/246) within 5 arcsec plus 2.5 years of its proper motion, since 2MASS observed from 1997 to 2001; a 2MASS source found is the cross-match's miss.

    python -I seti.py neighbourhood no2mass [record] [--plx-mas 50] [--g-range 5 13] [--radius-arcsec 5] [--years 2.5] [--epoch 2000]"""


def no2mass_main(record=False, plx_mas=50.0, g_range=[5.0, 13.0], radius_arcsec=5.0, years=2.5, epoch=2000.0):
    rows = core.tap("gaia", "SELECT g.source_id, g.ra, g.dec, g.parallax, g.phot_g_mean_mag, g.pmra, g.pmdec FROM external.gaiaedr3_gcns_main_1 AS g "
                            "LEFT OUTER JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS t ON g.source_id = t.source_id "
                            "WHERE g.parallax > %g AND g.phot_g_mean_mag BETWEEN %g AND %g AND t.source_id IS NULL" % (plx_mas, g_range[0], g_range[1]), timeout=600)
    up, reach = [], {}
    for r in rows:
        ra, de = core.star_at(float(r["ra"]), float(r["dec"]), num(r["pmra"], 0.0), num(r["pmdec"], 0.0), 2016.0, epoch)
        up.append(dict(source_id=r["source_id"], ra=ra, dec=de))
        reach[r["source_id"]] = radius_arcsec + years * math.hypot(num(r["pmra"], 0.0), num(r["pmdec"], 0.0)) / 1000.0
    found = {}
    for m in core.xmatch(up, "II/246/out", max(reach.values()) if reach else 5.0) if up else []:
        d = float(m["angDist"])
        if d <= reach[m["source_id"]] and (m["source_id"] not in found or d < found[m["source_id"]][0]):
            found[m["source_id"]] = (d, m.get("2MASS"), num(m.get("Kmag")))
    out = []
    for r in rows:
        pm = math.hypot(num(r["pmra"], 0.0), num(r["pmdec"], 0.0))
        f = found.get(r["source_id"])
        out.append(dict(source_id=r["source_id"], G=num(r["phot_g_mean_mag"]), plx=float(r["parallax"]), pm=pm,
                        tmass=f[1] if f else None, sep=f[0] if f else None, K=f[2] if f else None))
    # the rest: SIMBAD's name for each; a secondary of a multiple sits in its primary's glare in the 2MASS images
    for o in out:
        if o["tmass"] is None:
            s = core.simbad("SELECT b.main_id, b.otype FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id = 'Gaia DR3 %s'" % o["source_id"])
            o["simbad"] = " ".join(str(s[0]["main_id"]).split()) if s else None
            o["secondary"] = bool(o["simbad"] and re.search(r"\s[B-D]$", o["simbad"]))
    left = [o for o in out if o["tmass"] is None]
    print("%d nearby stars at G 5-13 without a 2MASS best neighbour (median proper motion %.0f mas/yr); a 2MASS source within 5 arcsec "
          "of the position in 2000 for %d; none for %d" % (len(out), float(np.median([o["pm"] for o in out])) if out else 0,
                                                          len(out) - len(left), len(left)))
    for o in left:
        print("  Gaia DR3 %s = %s: G %.1f, %.1f mas, %.0f mas/yr%s" % (o["source_id"], o["simbad"], o["G"] or 0, o["plx"], o["pm"],
                                                                     ", the secondary of a multiple" if o["secondary"] else ""))
    json.dump(out, open(core.results("no2mass_nearby.json"), "w"), indent=1)
    if record:
        core.record_search(
            "gaia_no2mass_nearby", "Light that appeared since 2MASS", archive="Gaia DR3 with the Gaia Catalogue of Nearby Stars and its 2MASS "
            "best neighbours; 2MASS point sources (VizieR II/246) by CDS XMatch", release="DR3",
            selection="Gaia Catalogue of Nearby Stars beyond parallax 50 mas at G 5-13 without a 2MASS best neighbour",
            code=os.path.abspath(__file__), entry="no2mass", n_targets=len(out), n_unit="stars",
            params=dict(plx_mas=plx_mas, G=g_range, radius_arcsec="%g + %g yr of proper motion" % (radius_arcsec, years), epoch=epoch),
            null_summary="the cross-match's misses: a 2MASS source at the position carried back to 2000",
            sensitivity="a source within 20 pc at G 13 or brighter, which 2MASS reaches at Ks 14 at its 1999 epoch",
            ledger="Light that appeared since 2MASS",
            result="%d stars, %d of them in 2MASS after all; none found: %d, %d of them secondaries of multiples (%s)" % (
                len(out), len(out) - len(left), len(left), sum(o["secondary"] for o in left), ", ".join(o["simbad"] or o["source_id"] for o in left)))
        core.clear_targets("gaia_no2mass_nearby")
        core.record_targets("gaia_no2mass_nearby", [dict(
            target="gaia_dr3:%s" % o["source_id"], stat=o["sep"], unit="arcsec to the nearest 2MASS source at the position in 2000",
            vet="artefact" if o["tmass"] else "blend" if o.get("secondary") else None,
            note="2MASS %s at Ks %s; G %.1f, %.0f mas/yr" % (o["tmass"], o["K"], o["G"] or 0, o["pm"])
            if o["tmass"] else "%s, no 2MASS source within reach; G %.1f, %.0f mas/yr" % (o.get("simbad"), o["G"] or 0, o["pm"])) for o in out])
        print("recorded")


unknown_DOC = r"""Objects of unknown nature within 10 pc: SIMBAD entries with a parallax above 100 mas typed as of unknown nature ('?') or as not an object ('err'), each read for what it is.

    python -I seti.py neighbourhood unknown [record] [--plx-mas 100]"""


def unknown_main(record=False, plx_mas=100.0):
    rows = core.simbad("SELECT main_id, otype, plx_value, sp_type, ra, dec, nbref FROM basic WHERE plx_value > %g AND otype IN ('?', 'err')" % plx_mas)
    print("%d SIMBAD entries within 10 pc of unknown nature or not an object" % len(rows))
    for r in rows:
        print("  %s: %s, %.1f mas, %s, %s references" % (r["main_id"], r["otype"], r["plx_value"], r["sp_type"], r["nbref"]))
    json.dump(rows, open(core.results("unknown_nearby.json"), "w"), indent=1, default=str)
    if record:
        core.record_search(
            "simbad_unknown_nearby", "Objects of unknown nature within 10 pc", archive="SIMBAD",
            selection="parallax above 100 mas and object type '?' or 'err'", code=os.path.abspath(__file__), entry="unknown",
            n_targets=len(rows), n_unit="SIMBAD entries", null_summary="the entry's own record and references",
            sensitivity="any SIMBAD entry within 10 pc left without a nature", ledger="Objects of unknown nature within 10 pc",
            result="%d entries: %s" % (len(rows), "; ".join("%s (%s)" % (r["main_id"], r["otype"]) for r in rows)))
        core.clear_targets("simbad_unknown_nearby")
        core.record_targets("simbad_unknown_nearby", [dict(
            target="simbad:%s" % " ".join(str(r["main_id"]).split()), stat=r["plx_value"], unit="parallax, mas",
            vet="catalogued-false-positive" if r["otype"] == "err" else None, note="SIMBAD type %s" % r["otype"]) for r in rows])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------
# light no class claims, rest in the universe's frame, thrust, copies, and where our own flags fall


def clean_photometry(g, bp_rp, excess):
    """Gaia's corrected BP/RP excess (Riello et al. 2021) within 3 sigma of a single source's."""
    c = core.cstar(excess, bp_rp)
    return c is not None and abs(c) < 3 * core.cstar_sigma(g)


def simbad_names(ids):
    """{Gaia DR3 source_id: (main_id, otype)} from SIMBAD."""
    out = {}
    for k in range(0, len(ids), 200):
        q = ("SELECT i.id, b.main_id, b.otype FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)"
             % ",".join("'Gaia DR3 %s'" % x for x in ids[k:k + 200]))
        for r in core.simbad(q):
            out[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip())
    return out


oacells_DOC = r"""Light Gaia's outlier map could not place: sources within 100 pc (parallax above 10 mas at 10 sigma) at G below 17 with RUWE below 1.4 and a single-source BP/RP excess whose Outlier Analysis neuron (astrophysical_parameters.neuron_oa_id) carries no class label in oa_neuron_information, against the share of all Outlier Analysis sources that sit in unlabelled neurons.

    python -I seti.py neighbourhood oacells [record] [--plx-mas 10] [--sigma 10] [--g-max 17] [--ruwe-max 1.4]"""


def oacells_main(record=False, plx_mas=10.0, sigma=10.0, g_max=17.0, ruwe_max=1.4):
    neurons = core.tap("gaia", "SELECT neuron_id, class_label, quality_category, hits FROM gaiadr3.oa_neuron_information")
    label = {r["neuron_id"]: (r["class_label"] or "").strip() for r in neurons}
    unlabelled = {n for n, c in label.items() if c.upper() in ("", "UNDEFINED", "UNKNOWN", "NOT_DEFINED", "NONE", "NULL")}
    hits_all = sum(int(r["hits"] or 0) for r in neurons)
    hits_un = sum(int(r["hits"] or 0) for r in neurons if r["neuron_id"] in unlabelled)
    rows = core.tap_async("gaia", "SELECT g.source_id, g.parallax, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.phot_bp_rp_excess_factor, "
                                  "a.neuron_oa_id, a.neuron_oa_dist_percentile_rank FROM gaiadr3.gaia_source AS g "
                                  "JOIN gaiadr3.astrophysical_parameters AS a ON g.source_id = a.source_id WHERE g.parallax > %g AND "
                                  "g.parallax_over_error > %g AND g.phot_g_mean_mag < %g AND g.ruwe < %g AND a.neuron_oa_id IS NOT NULL" % (plx_mas, sigma, g_max, ruwe_max),
                          maxrec=500000)
    clean = [r for r in rows if num(r["bp_rp"]) is not None and clean_photometry(float(r["phot_g_mean_mag"]), float(r["bp_rp"]),
                                                                                  float(r["phot_bp_rp_excess_factor"]))]
    hits = sorted([r for r in clean if r["neuron_oa_id"] in unlabelled], key=lambda r: -(num(r["neuron_oa_dist_percentile_rank"]) or 0))
    share_all = hits_un / max(hits_all, 1)
    expect = share_all * len(clean)
    names = simbad_names([r["source_id"] for r in hits[:200]])
    labels = {}
    for r in clean:
        c = label.get(r["neuron_oa_id"], "?") or "(none)"
        labels[c] = labels.get(c, 0) + 1
    print("%d neurons, %d of them unlabelled holding %.1f per cent of all Outlier Analysis sources; %d clean nearby sources in the map, "
          "classes %s; in unlabelled neurons %d against %.1f at the map's share" % (len(neurons), len(unlabelled), 100 * share_all, len(clean),
                                                                                     labels, len(hits), expect))
    for r in hits[:20]:
        n = names.get(r["source_id"], ("-", "-"))
        print("  Gaia DR3 %s: %.1f pc, G %.1f, BP-RP %.2f, neuron %s at distance rank %s; SIMBAD %s (%s)" % (
            r["source_id"], 1000 / float(r["parallax"]), float(r["phot_g_mean_mag"]), float(r["bp_rp"]), r["neuron_oa_id"],
            r["neuron_oa_dist_percentile_rank"], *n))
    json.dump(dict(neurons=len(neurons), unlabelled=len(unlabelled), share=share_all, n=len(clean), labels=labels, hits=hits), open(core.results("oa_cells.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_oa_unlabelled", "Light Gaia's outlier map could not place", "Gaia DR3 astrophysical_parameters and oa_neuron_information",
            "sources within 100 pc at G below 17, RUWE below 1.4, single-source BP/RP excess, in the Outlier Analysis map: %d" % len(clean),
            os.path.abspath(__file__), "oacells", len(clean), "Gaia sources", dict(plx_mas=plx_mas, sigma=sigma, g=g_max, ruwe=ruwe_max),
            "unlabelled neurons hold %.1f per cent of all Outlier Analysis sources: %.1f expected" % (100 * share_all, expect),
            "a clean nearby source in a neuron Gaia's Outlier Analysis left without a class",
            "%d in unlabelled neurons against %.1f" % (len(hits), expect),
            [dict(target="gaia_dr3:%s" % r["source_id"], stat=num(r["neuron_oa_dist_percentile_rank"]), unit="distance rank within the neuron",
                  vet="chance-rate" if len(hits) <= expect + 3 * math.sqrt(max(expect, 1)) else None,
                  note="neuron %s; SIMBAD %s (%s)" % (r["neuron_oa_id"], *names.get(r["source_id"], ("-", "-")))) for r in hits])
        print("recorded")


noclass_DOC = r"""Bright nearby light no class claims: sources within 100 pc (parallax above 10 mas at 10 sigma) at G below 15 with RUWE below 1.4 and a single-source BP/RP excess whose five combined class probabilities from Gaia DR3's discrete source classifier (star, white dwarf, binary star, quasar, galaxy) all fall below 0.5, each read in SIMBAD.

    python -I seti.py neighbourhood noclass [record] [--plx-mas 10] [--sigma 10] [--g-max 15] [--ruwe-max 1.4] [--prob 0.5]"""


def noclass_main(record=False, plx_mas=10.0, sigma=10.0, g_max=15.0, ruwe_max=1.4, prob=0.5):
    cls = ("star", "whitedwarf", "binarystar", "quasar", "galaxy")
    rows = core.tap_async("gaia", "SELECT g.source_id, g.parallax, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.phot_bp_rp_excess_factor, %s "
                                  "FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a ON g.source_id = a.source_id "
                                  "WHERE g.parallax > %g AND g.parallax_over_error > %g AND g.phot_g_mean_mag < %g AND g.ruwe < %g AND %s" % (
                                      ", ".join("a.classprob_dsc_combmod_%s" % c for c in cls), plx_mas, sigma, g_max, ruwe_max,
                                      " AND ".join("a.classprob_dsc_combmod_%s < %g" % (c, prob) for c in cls)), maxrec=200000)
    total = core.tap("gaia", "SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE parallax > %g AND parallax_over_error > %g AND "
                             "phot_g_mean_mag < %g AND ruwe < %g" % (plx_mas, sigma, g_max, ruwe_max))[0]["n"]
    hits = [r for r in rows if num(r["bp_rp"]) is not None and clean_photometry(float(r["phot_g_mean_mag"]), float(r["bp_rp"]),
                                                                                 float(r["phot_bp_rp_excess_factor"]))]
    names = simbad_names([r["source_id"] for r in hits])
    print("%s sources within 100 pc at G below 15 with RUWE below 1.4; every class below 0.5: %d, %d of them with single-source photometry" % (
        total, len(rows), len(hits)))
    for r in hits:
        print("  Gaia DR3 %s: %.1f pc, G %.1f, BP-RP %.2f, probabilities %s; SIMBAD %s (%s)" % (
            r["source_id"], 1000 / float(r["parallax"]), float(r["phot_g_mean_mag"]), float(r["bp_rp"]),
            " ".join("%s %.2f" % (c, float(r["classprob_dsc_combmod_" + c] or 0)) for c in cls), *names.get(r["source_id"], ("-", "-"))))
    json.dump(dict(total=total, rows=rows, hits=hits), open(core.results("noclass_nearby.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_noclass_nearby", "Bright nearby light no class claims", "Gaia DR3 astrophysical_parameters (discrete source classifier)",
            "sources within 100 pc at G below 15 and RUWE below 1.4: %s" % total, os.path.abspath(__file__), "noclass", int(total), "Gaia sources",
            dict(plx_mas=plx_mas, sigma=sigma, g=g_max, ruwe=ruwe_max, prob=prob), "every class below %g among the clean bright sources" % prob,
            "a bright nearby source with single-source photometry that the classifier gives to no class",
            "%d with every class below 0.5, %d of them with single-source photometry" % (len(rows), len(hits)),
            [dict(target="gaia_dr3:%s" % r["source_id"], stat=max(float(r["classprob_dsc_combmod_" + c] or 0) for c in cls),
                  unit="largest class probability", note="SIMBAD %s (%s)" % names.get(r["source_id"], ("-", "-"))) for r in hits])
        print("recorded")


cmbrest_DOC = r"""Something at rest in the universe's frame: the Sun moves at 369.82 km/s toward Galactic (264.021, +48.253) deg relative to the cosmic microwave background (Planck 2018), so a body at rest there moves at that speed toward (84.021, -48.253) as we see it; Gaia DR3 stars with radial velocities, parallaxes at 5 sigma and heliocentric speeds of 300-440 km/s (one asynchronous query), their velocities within 10 km/s of that vector at velocity errors below 10 km/s, against the counts within 10 km/s of 1000 vectors of the same speed in random directions.

    python -I seti.py neighbourhood cmbrest [record] [--tol-kms 10] [--err-kms 10] [--speed-range-kms 300 440] [--draws 1000]"""

CMB_SUN = (369.82, 264.021, 48.253)


def galactic_velocity(rows):
    """Heliocentric velocities (km/s) in Galactic Cartesian axes and their errors, for Gaia rows with radial velocities."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra, de = np.array([float(r["ra"]) for r in rows]), np.array([float(r["dec"]) for r in rows])
    plx = np.array([float(r["parallax"]) for r in rows])
    c = SkyCoord(ra=ra * u.deg, dec=de * u.deg, distance=(1000 / plx) * u.pc, pm_ra_cosdec=np.array([float(r["pmra"]) for r in rows]) * u.mas / u.yr,
                 pm_dec=np.array([float(r["pmdec"]) for r in rows]) * u.mas / u.yr,
                 radial_velocity=np.array([float(r["radial_velocity"]) for r in rows]) * u.km / u.s, frame="icrs")
    v = c.galactic.velocity
    V = np.stack([v.d_x.to(u.km / u.s).value, v.d_y.to(u.km / u.s).value, v.d_z.to(u.km / u.s).value], 1)
    k = 4.74047 / plx
    vt = k * np.hypot([float(r["pmra"]) for r in rows], [float(r["pmdec"]) for r in rows])
    e = np.sqrt(np.array([float(r["radial_velocity_error"] or 0) for r in rows]) ** 2 + (vt * np.array([float(r["parallax_error"]) for r in rows]) / plx) ** 2
                + (k * np.hypot([float(r["pmra_error"] or 0) for r in rows], [float(r["pmdec_error"] or 0) for r in rows])) ** 2)
    return V, e


def cmbrest_main(record=False, tol_kms=10.0, err_kms=10.0, speed_range_kms=[300.0, 440.0], draws=1000):
    rows = core.tap_async("gaia", "SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, "
                                  "radial_velocity_error, phot_g_mean_mag, ruwe FROM gaiadr3.gaia_source WHERE radial_velocity IS NOT NULL AND "
                                  "parallax_over_error > 5 AND radial_velocity * radial_velocity + 22.4721 * (pmra * pmra + pmdec * pmdec) / "
                                  "(parallax * parallax) BETWEEN %g AND %g" % (speed_range_kms[0] ** 2, speed_range_kms[1] ** 2), maxrec=2000000)
    V, e = galactic_velocity(rows)
    ok = e < err_kms
    v0, l0, b0 = CMB_SUN[0], math.radians(CMB_SUN[1] + 180.0), -math.radians(CMB_SUN[2])
    target = v0 * np.array([math.cos(b0) * math.cos(l0), math.cos(b0) * math.sin(l0), math.sin(b0)])
    from scipy.spatial import cKDTree
    tree = cKDTree(V[ok])
    found = tree.query_ball_point(target, tol_kms)
    rng = np.random.default_rng(11)
    d = rng.normal(size=(draws, 3))
    d = v0 * d / np.linalg.norm(d, axis=1)[:, None]
    null = np.array(tree.query_ball_point(d, tol_kms, return_length=True))
    p = float(np.mean(null >= len(found)))
    idx = np.nonzero(ok)[0][found]
    print("%d stars at heliocentric speeds of 300-440 km/s, %d with velocity errors below 10 km/s; within 10 km/s of rest in the microwave "
          "background's frame (U, V, W) = (%.1f, %.1f, %.1f): %d; around 1000 random vectors of the same speed: mean %.2f, maximum %d, as many "
          "or more in %.3f" % (len(rows), int(ok.sum()), *target, len(found), float(null.mean()), int(null.max()), p))
    for i in idx:
        print("  Gaia DR3 %s: %.0f pc, G %.1f, velocity %s, %.1f km/s from the frame" % (
            rows[i]["source_id"], 1000 / float(rows[i]["parallax"]), float(rows[i]["phot_g_mean_mag"]), np.round(V[i], 1).tolist(),
            float(np.linalg.norm(V[i] - target))))
    json.dump(dict(n=len(rows), n_ok=int(ok.sum()), target=target.tolist(), found=[rows[i]["source_id"] for i in idx], null_mean=float(null.mean()),
                   null_max=int(null.max()), p=p), open(core.results("cmb_rest.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_cmb_rest", "Something at rest in the universe's frame", "Gaia DR3 gaia_source", "stars with radial velocities and parallaxes "
            "at 5 sigma at heliocentric speeds of 300-440 km/s: %d, %d with velocity errors below 10 km/s" % (len(rows), int(ok.sum())),
            os.path.abspath(__file__), "cmbrest", int(ok.sum()), "stars", dict(v_kms=CMB_SUN[0], l=CMB_SUN[1], b=CMB_SUN[2], tol_kms=tol_kms, err_kms=err_kms,
                                                                                  speed_range_kms=speed_range_kms, draws=draws),
            "%d vectors of the same speed in random directions: mean %.2f, maximum %d" % (draws, float(null.mean()), int(null.max())),
            "a star at rest in the cosmic microwave background's frame to 10 km/s within Gaia's radial-velocity reach",
            "%d within 10 km/s against %.2f around random directions (as many or more in %.3f)" % (len(found), float(null.mean()), p),
            [dict(target="gaia_dr3:%s" % rows[i]["source_id"], stat=float(np.linalg.norm(V[i] - target)), unit="km/s from rest in the frame",
                  vet="chance-rate" if p > 0.01 else None) for i in idx])
        print("recorded")


thrust_DOC = r"""Stars accelerating along their own motion: Gaia DR3's 7- and 9-parameter acceleration solutions (nss_acceleration_astro) with the acceleration at 5 sigma, the perspective acceleration of the radial velocity (-2 mu v_r parallax) removed where the cached sample within 100 pc gives the radial velocity, the angle between each acceleration and the star's proper motion, which an orbiting companion leaves uniform and thrust would gather near 0 deg; the share within 8 deg against the 4.4 per cent of a uniform angle.

    python -I seti.py neighbourhood thrust [record] [--sigma 5] [--window-deg 8]"""


def thrust_main(record=False, sigma=5.0, window_deg=8.0):
    from scipy.stats import binomtest
    rows = core.tap_async("gaia", "SELECT source_id, nss_solution_type, accel_ra, accel_ra_error, accel_dec, accel_dec_error, pmra, pmdec, parallax "
                                  "FROM gaiadr3.nss_acceleration_astro", maxrec=500000)
    # the perspective term matters only for near, fast stars: their radial velocities come from the cached sample within 100 pc
    R, c = nearby_rv()
    rvs = {r["source_id"]: v for r, v in zip(R, c["radial_velocity"]) if np.isfinite(v)}
    keep, ang = [], []
    for r in rows:
        ar, ad, er, ed = (num(r[k]) for k in ("accel_ra", "accel_dec", "accel_ra_error", "accel_dec_error"))
        pa, pd, plx, rv = num(r["pmra"]), num(r["pmdec"]), num(r["parallax"]), rvs.get(r["source_id"])
        if None in (ar, ad, er, ed, pa, pd, plx) or not er or not ed or math.hypot(pa, pd) < 1:
            continue
        if rv is not None and plx > 0:
            k = -2.0 * rv * plx * 1.0227e-9
            ar, ad = ar - k * pa, ad - k * pd
        if math.hypot(ar / er, ad / ed) < sigma:
            continue
        cosang = (ar * pa + ad * pd) / (math.hypot(ar, ad) * math.hypot(pa, pd))
        keep.append(r)
        ang.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    ang = np.array(ang)
    fwd, back = int(np.sum(ang < window_deg)), int(np.sum(ang > 180 - window_deg))
    p_f = binomtest(fwd, len(ang), window_deg / 180.0, alternative="greater").pvalue if len(ang) else 1.0
    hist = np.histogram(ang, bins=9, range=(0, 180))[0]
    print("%d acceleration solutions, %d at 5 sigma after the perspective term; within 8 deg of the motion %d (%.2f per cent), within 8 deg "
          "against it %d, against 4.44 per cent uniform (p of the forward excess %.3g); counts in 20 deg bins %s" % (
              len(rows), len(ang), fwd, 100 * fwd / max(len(ang), 1), back, p_f, hist.tolist()))
    json.dump(dict(n=len(rows), n_sig=len(ang), forward=fwd, backward=back, p=p_f, hist=hist.tolist()), open(core.results("thrust.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_thrust_alignment", "Stars accelerating along their own motion", "Gaia DR3 nss_acceleration_astro", "acceleration solutions "
            "at 5 sigma after the perspective term: %d" % len(ang), os.path.abspath(__file__), "thrust", len(ang), "stars",
            dict(sigma=sigma, window_deg=window_deg), "a uniform angle between acceleration and motion: %.2f per cent within %g deg" % (100 * window_deg / 180.0, window_deg),
            "an excess of accelerations within 8 deg of the stars' own motion", "%d forward (%.2f per cent) and %d backward against 4.44 per cent "
            "each (forward p %.3g); counts in 20 deg bins %s" % (fwd, 100 * fwd / max(len(ang), 1), back, p_f, hist.tolist()), [])
        print("recorded")


copies_DOC = r"""Copy stars on opposite sides of the sky: Gaia DR3 stars within 100 pc with radial velocities and velocities measured to 0.5 km/s, pairs more than 90 deg apart agreeing in G and BP-RP to 0.002 mag, in parallax to 0.2 per cent and in all three velocity components to 0.5 km/s, against the same count with the velocities shuffled among the stars twenty times.

    python -I seti.py neighbourhood copies [record] [--dmag 0.002] [--dplx 0.002] [--dv-kms 0.5] [--err-kms 0.5] [--shuffles 20]"""


def copies_main(record=False, dmag=0.002, dplx=0.002, dv_kms=0.5, err_kms=0.5, shuffles=20):
    from scipy.spatial import cKDTree
    R, c = nearby_rv()
    va, vd, vr, ea, ed, er = nearby_velocity(c)
    e = np.sqrt(ea ** 2 + ed ** 2 + er ** 2)
    ok = np.isfinite(vr) & (e < err_kms) & np.isfinite(c["bp_rp"]) & np.isfinite(c["phot_g_mean_mag"])
    rhat, ahat, dhat, pos = nearby_axes(c)
    vel = rhat * vr[:, None] + ahat * va[:, None] + dhat * vd[:, None]
    idx = np.nonzero(ok)[0]

    def pairs(V):
        X = np.column_stack([c["phot_g_mean_mag"][idx] / dmag, c["bp_rp"][idx] / dmag, np.log(c["parallax"][idx]) / dplx, V / dv_kms])
        out = []
        for i, j in cKDTree(X).query_pairs(1.0, p=np.inf):
            if float(rhat[idx[i]] @ rhat[idx[j]]) < 0:
                out.append((int(idx[i]), int(idx[j])))
        return out
    found = pairs(vel[idx])
    rng = np.random.default_rng(5)
    null = [len(pairs(vel[idx][rng.permutation(len(idx))])) for _ in range(shuffles)]
    print("%d stars with velocities to 0.5 km/s; copy pairs more than 90 deg apart %d; with velocities shuffled %s" % (len(idx), len(found), null))
    for i, j in found:
        print("  Gaia DR3 %s and %s" % (R[i]["source_id"], R[j]["source_id"]))
    json.dump(dict(n=len(idx), pairs=[(R[i]["source_id"], R[j]["source_id"]) for i, j in found], null=null), open(core.results("copy_stars.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_copy_stars", "Copy stars on opposite sides of the sky", "Gaia DR3 gaia_source", "stars within 100 pc with velocities to 0.5 km/s: %d"
            % len(idx), os.path.abspath(__file__), "copies", len(idx), "stars", dict(dmag=dmag, dcolour=dmag, dplx=dplx, dv_kms=dv_kms, err_kms=err_kms, sep_deg=90, shuffles=shuffles),
            "velocities shuffled among the stars: %s" % null, "two stars more than 90 deg apart alike in brightness, colour, distance and velocity",
            "%d pairs against %.1f with velocities shuffled" % (len(found), float(np.mean(null))),
            [dict(target="gaia_dr3:%s" % R[i]["source_id"], note="copy of Gaia DR3 %s" % R[j]["source_id"]) for i, j in found])
        print("recorded")


def flagzones_positions():
    """Our store's Gaia DR3 targets with their ecliptic latitude and Galactic coordinates, and whether a run vetted them; cached."""
    p = core.results("store_positions.json")
    db = sqlite3.connect(core.COVERAGE)
    tg = {}
    for t, v in db.execute("SELECT target, vet FROM coverage WHERE target LIKE 'gaia_dr3:%'"):
        tg[t] = tg.get(t, False) or bool(v)
    ids = [t.split(":")[1] for t in tg]
    pos = json.load(open(p)) if os.path.exists(p) else {}
    todo = [i for i in ids if i not in pos]
    if todo:
        got = core.tap_by_ids("gaia", todo, "SELECT source_id, l, b, ecl_lat FROM gaiadr3.gaia_source WHERE source_id IN (%s)")
        for i, r in got.items():
            pos[i] = (float(r["l"]), float(r["b"]), float(r["ecl_lat"]))
        json.dump(pos, open(p, "w"))
    return [(i, pos[i], tg["gaia_dr3:" + i]) for i in ids if i in pos]


def flagzones_runs():
    """Each run's Gaia DR3 targets of the coverage store as (star id, position, vetted) triples, keyed by run, for nulls that keep every run's own sky coverage."""
    pos = {i: q for i, q, _ in flagzones_positions()}
    db = sqlite3.connect(core.COVERAGE)
    runs = {}
    for sid, t, v in db.execute("SELECT search_id, target, vet FROM coverage WHERE target LIKE 'gaia_dr3:%'"):
        i = t.split(":")[1]
        if i in pos:
            runs.setdefault(sid, []).append((i, pos[i], bool(v)))
    return runs


flagetz_DOC = r"""Our explained flags in Earth's transit zone: every Gaia DR3 star of the coverage store, the share of those some run vetted within 0.27 deg of the ecliptic, where they would see Earth transit the Sun, against the share of all the store's stars there.

    python -I seti.py neighbourhood flagetz [record] [--ecl-lat-deg 0.27]"""


def flagetz_main(record=False, ecl_lat_deg=0.27):
    from scipy.stats import binomtest
    rows = flagzones_positions()
    zone = [r for r in rows if abs(r[1][2]) < ecl_lat_deg]
    flagged = [r for r in rows if r[2]]
    fz = [r for r in flagged if abs(r[1][2]) < ecl_lat_deg]
    share = len(zone) / max(len(rows), 1)
    p = binomtest(len(fz), len(flagged), share, alternative="greater").pvalue if flagged else 1.0
    print("%d store stars, %d in the transit zone (%.3f per cent); %d vetted, %d of them in the zone against %.1f (p %.3g)" % (
        len(rows), len(zone), 100 * share, len(flagged), len(fz), share * len(flagged), p))
    if record:
        core.record_run(
            "store_flags_etz", "Our explained flags in Earth's transit zone", "the coverage store with Gaia DR3 positions",
            "Gaia DR3 targets of every run: %d" % len(rows), os.path.abspath(__file__), "flagetz", len(rows), "stars", dict(ecl_lat_deg=ecl_lat_deg),
            "the share of all store stars within %g deg of the ecliptic: %.3f per cent" % (ecl_lat_deg, 100 * share),
            "an excess of vetted targets in the zone from which Earth is seen to transit", "%d of %d vetted stars in the zone against %.1f "
            "(p %.3g)" % (len(fz), len(flagged), share * len(flagged), p), [])
        print("recorded")


flagdirs_DOC = r"""Our flags toward the obvious meeting-point directions: every Gaia DR3 star of the coverage store, the share of those some run vetted within 30 deg of the Galactic centre, of the anticentre and of either Galactic pole, against the share of all the store's stars in each.

The share of all store stars toward a direction mixes runs that vet every target with runs that vet none, so the count is also taken run by run, each run's vetted count times its own share of targets toward the direction, with a normal tail on the hypergeometric variance. With `--apex` the directions are the solar apex and antapex instead (l 56.2, b +22.8 deg and its opposite), the way the Sun travels through its neighbours, where a civilisation that has noticed us coming or going would mark the sky, recorded as its own run.

    python -I seti.py neighbourhood flagdirs [record] [--radius-deg 30] [--pole-lat-deg 60] [--apex]"""


def flagdirs_main(record=False, radius_deg=30.0, pole_lat_deg=60.0, apex=False):
    from scipy.stats import binomtest
    rows = flagzones_positions()
    flagged = [r for r in rows if r[2]]

    def ang(l1, b1, l2, b2):
        return math.degrees(math.acos(max(-1.0, min(1.0, math.sin(math.radians(b1)) * math.sin(math.radians(b2)) + math.cos(math.radians(b1)) *
                                                     math.cos(math.radians(b2)) * math.cos(math.radians(l1 - l2))))))
    zones = {"centre": lambda l, b: ang(l, b, 0, 0) < radius_deg, "anticentre": lambda l, b: ang(l, b, 180, 0) < radius_deg, "poles": lambda l, b: abs(b) > pole_lat_deg}
    if apex:
        zones = {"apex": lambda l, b: ang(l, b, 56.2, 22.8) < radius_deg, "antapex": lambda l, b: ang(l, b, 236.2, -22.8) < radius_deg}
    from scipy.stats import norm
    runs = flagzones_runs()
    out = {}
    for k, f in zones.items():
        share = sum(f(r[1][0], r[1][1]) for r in rows) / max(len(rows), 1)
        n = sum(f(r[1][0], r[1][1]) for r in flagged)
        p = binomtest(n, len(flagged), share, alternative="greater").pvalue if flagged else 1.0
        mean, var, n_rt = 0.0, 0.0, 0
        for rr in runs.values():
            k_i = sum(r[2] for r in rr)
            if not k_i:
                continue
            inzone = [f(r[1][0], r[1][1]) for r in rr]
            s_i = sum(inzone) / len(rr)
            mean += k_i * s_i
            var += k_i * s_i * (1 - s_i) * (len(rr) - k_i) / max(len(rr) - 1, 1)
            n_rt += sum(1 for r, z in zip(rr, inzone) if r[2] and z)
        out[k] = dict(share=share, n=n, expect=share * len(flagged), p=p, run_n=n_rt, run_expect=mean, run_p=float(norm.sf((n_rt - mean) / math.sqrt(var))) if var else 1.0)
    print("%d store stars, %d vetted: %s" % (len(rows), len(flagged), "; ".join("%s %d against %.1f from the whole store's share (p %.3g), %d run targets against %.1f when each "
                                                                              "run's own coverage sets its share (p %.3g)" % (k, v["n"], v["expect"], v["p"], v["run_n"], v["run_expect"], v["run_p"])
                                                                              for k, v in out.items())))
    if record:
        core.record_run(
            "store_flags_apex" if apex else "store_flags_directions", "Our flags toward the solar apex" if apex else "Our flags toward the obvious meeting-point directions",
            "the coverage store with Gaia DR3 positions",
            "Gaia DR3 targets of every run: %d" % len(rows), os.path.abspath(__file__), "flagdirs", len(rows), "stars", dict(radius_deg=radius_deg, pole_lat_deg=pole_lat_deg, apex=apex),
            "the share of all store stars toward each direction, and each run's own share of its targets toward it", "an excess of vetted targets toward the solar apex or antapex" if apex else
            "an excess of vetted targets toward the Galactic centre, anticentre or poles",
            "; ".join("%s %d of %d against %.1f from the whole store (p %.3g) and %d run targets against %.1f run by run (p %.3g)" % (
                k, v["n"], len(flagged), v["expect"], v["p"], v["run_n"], v["run_expect"], v["run_p"]) for k, v in out.items()), [])
        print("recorded")


mysteries_DOC = r"""Well-studied neighbours of unknown nature: SIMBAD entries with parallaxes above 10 mas typed of unknown nature ('?') or as not an object ('err') that carry 20 or more references, objects much studied and still unclassified.

    python -I seti.py neighbourhood mysteries [record] [--plx-mas 10] [--min-refs 20]"""


def mysteries_main(record=False, plx_mas=10.0, min_refs=20):
    rows = core.simbad("SELECT main_id, otype, plx_value, sp_type, nbref, ra, dec FROM basic WHERE plx_value > %g AND otype IN ('?', 'err') AND nbref >= %d"
                       % (plx_mas, min_refs))
    print("%d SIMBAD entries within 100 pc of unknown nature with 20 or more references: %s" % (
        len(rows), [(" ".join(str(r["main_id"]).split()), r["otype"], r["plx_value"], r["nbref"]) for r in rows]))
    if record:
        core.record_run("simbad_studied_unknowns", "Well-studied neighbours of unknown nature", "SIMBAD", "entries within 100 pc typed '?' or 'err' "
                        "with 20 or more references: %d" % len(rows), os.path.abspath(__file__), "mysteries", len(rows), "SIMBAD entries", dict(plx_mas=plx_mas, nbref=min_refs),
                        "the entries' own records", "a much-studied neighbour still without a class", "%d: %s" % (len(rows), "; ".join(
                            " ".join(str(r["main_id"]).split()) for r in rows)),
                        [dict(target="simbad:%s" % " ".join(str(r["main_id"]).split()), stat=r["nbref"], unit="references",
                              vet="catalogued-false-positive" if r["otype"] == "err" else None) for r in rows])
        print("recorded")


unbound_DOC = r"""Stars leaving the Galaxy from next door: the cached Gaia DR3 stars within 100 pc with radial velocities and velocity errors below 5 km/s, their speed about the Galactic centre with the Sun at (11.1, 245.0, 7.3) km/s, flagged above 550 km/s, the local escape speed (Monari et al. 2018 give 580 +- 63).

    python -I seti.py neighbourhood unbound [record] [--v-esc-kms 550] [--err-kms 5]"""


def unbound_main(record=False, v_esc_kms=550.0, err_kms=5.0):
    R, c = nearby_rv()
    va, vd, vr, ea, ed, er = nearby_velocity(c)
    e = np.sqrt(ea ** 2 + ed ** 2 + er ** 2)
    rhat, ahat, dhat, pos = nearby_axes(c)
    vel = rhat * vr[:, None] + ahat * va[:, None] + dhat * vd[:, None]
    # ICRS to Galactic axes (Hipparcos matrix), then the Sun's motion about the centre
    A = np.array([[-0.0548755604, -0.8734370902, -0.4838350155], [0.4941094279, -0.4448296300, 0.7469822445], [-0.8676661490, -0.1980763734, 0.4559837762]])
    g = vel @ A.T + np.array([11.1, 245.0, 7.3])
    s = np.linalg.norm(g, axis=1)
    ok = np.isfinite(s) & (e < err_kms)
    hits = sorted([i for i in np.nonzero(ok & (s > v_esc_kms))[0]], key=lambda i: -s[i])
    print("%d stars with velocities to 5 km/s; Galactocentric speed median %.0f, 99.99th percentile %.0f, maximum %.0f km/s; above 550: %d %s" % (
        int(ok.sum()), float(np.median(s[ok])), float(np.percentile(s[ok], 99.99)), float(np.max(s[ok])), len(hits),
        [(R[i]["source_id"], round(float(s[i]))) for i in hits[:10]]))
    if record:
        core.record_run("gaia_unbound_nearby", "Stars leaving the Galaxy from next door", "Gaia DR3 gaia_source (cached nearby sample)",
                        "stars within 100 pc with radial velocities and velocity errors below 5 km/s: %d" % int(ok.sum()), os.path.abspath(__file__),
                        "unbound", int(ok.sum()), "stars", dict(v_esc_kms=v_esc_kms, err_kms=err_kms, sun_uvw=[11.1, 245.0, 7.3]), "the local escape speed",
                        "a star within 100 pc moving faster than the Galaxy can hold", "%d above 550 km/s; the fastest %.0f km/s" % (len(hits), float(np.max(s[ok]))),
                        [dict(target="gaia_dr3:%s" % R[i]["source_id"], stat=float(s[i]), unit="Galactocentric speed, km/s") for i in hits])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------
# the nearest, the fastest and the stillest: limits of the nearby stars' astrometry

proxima_DOC = r"""Something closer than Proxima Centauri: Gaia DR3 sources with parallaxes above 700 mas, flagged where the parallax exceeds Proxima's 768.07 mas at 5 sigma, each read for its solution (parameters solved, parallax over error, RUWE) and in SIMBAD.

    python -I seti.py neighbourhood proxima [record] [--plx-min-mas 700] [--sigma 5]"""

PROXIMA_MAS = 768.07


def proxima_main(record=False, plx_min_mas=700.0, sigma=5.0):
    rows = core.tap("gaia", "SELECT source_id, ra, dec, parallax, parallax_error, parallax_over_error, phot_g_mean_mag, bp_rp, ruwe, pmra, pmdec, "
                            "astrometric_params_solved FROM gaiadr3.gaia_source WHERE parallax > %g" % plx_min_mas)
    names = simbad_names([r["source_id"] for r in rows])
    closer = [r for r in rows if float(r["parallax"]) - sigma * float(r["parallax_error"]) > PROXIMA_MAS]
    print("%d Gaia DR3 sources with parallax above 700 mas; above Proxima's %.2f mas at 5 sigma: %d" % (len(rows), PROXIMA_MAS, len(closer)))
    for r in sorted(rows, key=lambda r: -float(r["parallax"])):
        print("  Gaia DR3 %s: parallax %.1f +- %.1f mas (over error %.0f), G %.1f, RUWE %s, %s parameters; SIMBAD %s (%s)" % (
            r["source_id"], float(r["parallax"]), float(r["parallax_error"]), float(r["parallax_over_error"]), float(r["phot_g_mean_mag"]),
            r["ruwe"], r["astrometric_params_solved"], *names.get(r["source_id"], ("-", "-"))))
    if record:
        core.record_run(
            "gaia_closer_than_proxima", "Something closer than Proxima Centauri", "Gaia DR3 gaia_source; SIMBAD", "sources with parallax above 700 mas: %d" % len(rows),
            os.path.abspath(__file__), "proxima", len(rows), "Gaia sources", dict(parallax_mas=PROXIMA_MAS, sigma=sigma, plx_min_mas=plx_min_mas), "Proxima Centauri's parallax",
            "a source nearer than Proxima Centauri", "%d above %.2f mas at 5 sigma" % (len(closer), PROXIMA_MAS),
            [dict(target="gaia_dr3:%s" % r["source_id"], stat=float(r["parallax"]), unit="parallax, mas",
                  vet="astrometry" if float(r["parallax_over_error"]) < 10 or (num(r["ruwe"]) or 0) > 1.4 else None,
                  note="SIMBAD %s (%s)" % names.get(r["source_id"], ("-", "-"))) for r in closer])
        print("recorded")


barnard_DOC = r"""Something faster across the sky than Barnard's Star: Gaia DR3 sources with total proper motions above 9000 mas/yr, flagged where the motion exceeds Barnard's Star's 10393 mas/yr at 5 sigma, each read for its solution and in SIMBAD.

    python -I seti.py neighbourhood barnard [record] [--pm-min-masyr 9000] [--sigma 5]"""

BARNARD_MASYR = 10393.35
BARNARD_ID = "4472832130942575872"


def barnard_main(record=False, pm_min_masyr=9000.0, sigma=5.0):
    ids = [r["source_id"] for r in core.tap_async("gaia", "SELECT source_id FROM gaiadr3.gaia_source WHERE pm > %g" % pm_min_masyr, maxrec=10000)]
    rows = list(core.tap_by_ids("gaia", ids, "SELECT source_id, pm, pmra, pmra_error, pmdec, pmdec_error, parallax, parallax_over_error, "
                                             "phot_g_mean_mag, ruwe FROM gaiadr3.gaia_source WHERE source_id IN (%s)").values()) if ids else []
    names = simbad_names([r["source_id"] for r in rows])
    faster = [r for r in rows if r["source_id"] != BARNARD_ID and float(r["pm"]) - sigma * math.hypot(float(r["pmra_error"]), float(r["pmdec_error"])) > BARNARD_MASYR]
    print("%d Gaia DR3 sources with proper motion above 9000 mas/yr; above Barnard's Star's %.0f at 5 sigma: %d" % (len(rows), BARNARD_MASYR, len(faster)))
    for r in sorted(rows, key=lambda r: -float(r["pm"])):
        print("  Gaia DR3 %s: %.0f mas/yr, parallax %s, G %.1f, RUWE %s; SIMBAD %s (%s)" % (r["source_id"], float(r["pm"]), r["parallax"],
                                                                                    float(r["phot_g_mean_mag"]), r["ruwe"], *names.get(r["source_id"], ("-", "-"))))
    if record:
        core.record_run(
            "gaia_faster_than_barnard", "Something faster across the sky than Barnard's Star", "Gaia DR3 gaia_source; SIMBAD",
            "sources with proper motion above 9000 mas/yr: %d" % len(rows), os.path.abspath(__file__), "barnard", len(rows), "Gaia sources",
            dict(pm_masyr=BARNARD_MASYR, sigma=sigma, pm_min_masyr=pm_min_masyr), "Barnard's Star's proper motion", "a source crossing the sky faster than Barnard's Star",
            "%d above %.0f mas/yr at 5 sigma" % (len(faster), BARNARD_MASYR),
            [dict(target="gaia_dr3:%s" % r["source_id"], stat=float(r["pm"]), unit="proper motion, mas/yr",
                  vet="astrometry" if (num(r["parallax_over_error"]) or 0) < 10 or (num(r["ruwe"]) or 0) > 1.4 else None,
                  note="SIMBAD %s (%s)" % names.get(r["source_id"], ("-", "-"))) for r in faster])
        print("recorded")


still_DOC = r"""Stars standing still in the Galaxy: the cached Gaia DR3 stars within 100 pc with velocity errors below 5 km/s, their speed about the Galactic centre with the Sun at (11.1, 245.0, 7.3) km/s, flagged below 30 km/s, where the disc circles at 245 km/s and halo stars pass through at hundreds; each read in SIMBAD.

    python -I seti.py neighbourhood still [record] [--v-max-kms 30] [--err-kms 5]"""


def still_main(record=False, v_max_kms=30.0, err_kms=5.0):
    R, c = nearby_rv()
    va, vd, vr, ea, ed, er = nearby_velocity(c)
    e = np.sqrt(ea ** 2 + ed ** 2 + er ** 2)
    rhat, ahat, dhat, pos = nearby_axes(c)
    vel = rhat * vr[:, None] + ahat * va[:, None] + dhat * vd[:, None]
    A = np.array([[-0.0548755604, -0.8734370902, -0.4838350155], [0.4941094279, -0.4448296300, 0.7469822445], [-0.8676661490, -0.1980763734, 0.4559837762]])
    g = vel @ A.T + np.array([11.1, 245.0, 7.3])
    s = np.linalg.norm(g, axis=1)
    ok = np.isfinite(s) & (e < err_kms)
    slow = sorted(np.nonzero(ok & (s < v_max_kms))[0], key=lambda i: s[i])
    # a smooth distribution of velocities fills a sphere of speeds in proportion to its volume
    n60 = int((ok & (s < 2 * v_max_kms)).sum())
    expect = n60 / 8.0
    order = np.argsort(np.where(ok, s, np.inf))[:5]
    names = simbad_names([R[i]["source_id"] for i in order])
    mh = {r["source_id"]: num(r["mh_gspphot"]) for r in core.tap("gaia", "SELECT source_id, mh_gspphot FROM gaiadr3.astrophysical_parameters "
                                                                           "WHERE source_id IN (%s)" % ",".join(R[i]["source_id"] for i in order))}
    print("%d stars with velocities to 5 km/s; Galactocentric speed below 30 km/s: %d against %.1f from the %d below 60; the slowest %s" % (
        int(ok.sum()), len(slow), expect, n60, ["Gaia DR3 %s %.0f km/s (%s, [M/H] %s)" % (R[i]["source_id"], s[i], names.get(R[i]["source_id"], ("-",))[0],
                                                                                       mh.get(R[i]["source_id"])) for i in order]))
    if record:
        core.record_run("gaia_still_in_galaxy", "Stars standing still in the Galaxy", "Gaia DR3 gaia_source (cached nearby sample)",
                        "stars within 100 pc with velocity errors below 5 km/s: %d" % int(ok.sum()), os.path.abspath(__file__), "still", int(ok.sum()),
                        "stars", dict(v_max_kms=v_max_kms, err_kms=err_kms, sun_uvw=[11.1, 245.0, 7.3]), "the stars below %g km/s scaled by the volume of speeds: %.1f" % (2 * v_max_kms, expect),
                        "a star within 100 pc at rest about the Galactic centre", "%d below 30 km/s against %.1f; the slowest %.0f km/s" % (
                            len(slow), expect, float(s[order[0]])),
                        [dict(target="gaia_dr3:%s" % R[i]["source_id"], stat=float(s[i]), unit="Galactocentric speed, km/s",
                              vet="chance-rate" if len(slow) <= expect + 3 * math.sqrt(max(expect, 1)) else None,
                              note="%s, [M/H] %s" % (names.get(R[i]["source_id"], ("-",))[0], mh.get(R[i]["source_id"]))) for i in slow])
        print("recorded")


fastrv_DOC = r"""Radial velocities the Galaxy cannot hold: Gaia DR3 radial velocities beyond +-700 km/s (the local escape speed plus the Sun's own motion) measured to 5 km/s over ten or more transits, within the RVS pipeline's +-1000 km/s search; each read for the share of its transits Gaia deblended from a neighbour's spectrum, its expected S/N and template, and in SIMBAD.

    python -I seti.py neighbourhood fastrv [record] [--rv-kms 700] [--error-kms 5] [--min-transits 10] [--deblended-share 0.3]"""


def fastrv_main(record=False, rv_kms=700.0, error_kms=5.0, min_transits=10, deblended_share=0.3):
    ids = [r["source_id"] for r in core.tap_async("gaia", "SELECT source_id FROM gaiadr3.gaia_source WHERE ABS(radial_velocity) > %g AND "
                                                          "radial_velocity_error < %g AND rv_nb_transits >= %d" % (rv_kms, error_kms, min_transits), maxrec=100000)]
    rows = list(core.tap_by_ids("gaia", ids, "SELECT source_id, radial_velocity, radial_velocity_error, rv_nb_transits, rv_nb_deblended_transits, "
                                             "rv_expected_sig_to_noise, rv_template_teff, rv_method_used, parallax, parallax_over_error, phot_g_mean_mag, bp_rp, ruwe "
                                             "FROM gaiadr3.gaia_source WHERE source_id IN (%s)").values()) if ids else []

    def blended(r):
        # transits whose spectra Gaia deblended carry a neighbour's light
        return (num(r["rv_nb_deblended_transits"]) or 0) > deblended_share * float(r["rv_nb_transits"])
    names = simbad_names([r["source_id"] for r in rows])
    print("%d Gaia DR3 radial velocities beyond +-700 km/s to 5 km/s over ten or more transits" % len(rows))
    for r in sorted(rows, key=lambda r: -abs(float(r["radial_velocity"]))):
        print("  Gaia DR3 %s: %+.1f +- %.1f km/s, %s transits (%s deblended), expected S/N %s, template %s K, G %.1f, BP-RP %s, parallax %s; SIMBAD %s (%s)" % (
            r["source_id"], float(r["radial_velocity"]), float(r["radial_velocity_error"]), r["rv_nb_transits"], r["rv_nb_deblended_transits"],
            r["rv_expected_sig_to_noise"], r["rv_template_teff"], float(r["phot_g_mean_mag"]), r["bp_rp"], r["parallax"], *names.get(r["source_id"], ("-", "-"))))
    if record:
        core.record_run("gaia_rv_beyond_escape", "Radial velocities the Galaxy cannot hold", "Gaia DR3 gaia_source; SIMBAD",
                        "radial velocities beyond +-700 km/s to 5 km/s over ten or more transits: %d" % len(rows), os.path.abspath(__file__), "fastrv",
                        len(rows), "stars", dict(rv_kms=rv_kms, error_kms=error_kms, transits=min_transits, deblended_share=deblended_share), "the local escape speed plus the Sun's motion",
                        "a star whose line-of-sight speed alone escapes the Galaxy", "%d, %d of them with more than 0.3 of their transits deblended" % (
                            len(rows), sum(blended(r) for r in rows)),
                        [dict(target="gaia_dr3:%s" % r["source_id"], stat=float(r["radial_velocity"]), unit="radial velocity, km/s", vet="blend" if blended(r) else None,
                              note="%s of %s transits deblended, expected S/N %s, template %s K; SIMBAD %s (%s)" % (
                                  r["rv_nb_deblended_transits"], r["rv_nb_transits"], r["rv_expected_sig_to_noise"], r["rv_template_teff"],
                                  *names.get(r["source_id"], ("-", "-")))) for r in rows])
        print("recorded")


lattice_DOC = r"""A lattice among the nearby stars: the cached Gaia DR3 stars within 50 pc brighter than absolute G 10 (complete there), systems closer than 0.1 pc merged, the mean distance to the nearest neighbour of those within 45 pc over the Poisson expectation 0.554 n^-1/3 (the Clark-Evans ratio: 1 at random, above 1 for a regular spacing, below for clustering), against 1000 uniform volumes of the same number.

    python -I seti.py neighbourhood lattice [record] [--radius-pc 50] [--inner-pc 45] [--abs-g 10] [--merge-pc 0.1] [--sims 1000]"""


def lattice_main(record=False, radius_pc=50.0, inner_pc=45.0, abs_g=10.0, merge_pc=0.1, sims=1000):
    from scipy.spatial import cKDTree
    R, c = nearby_astrometry()
    plx = np.asarray(c["parallax"], float)
    g = np.asarray(c["phot_g_mean_mag"], float)
    ok = (plx > 1000.0 / radius_pc) & np.isfinite(g) & (g + 5 * np.log10(plx / 100.0) < abs_g)
    rhat, ahat, dhat, pos = nearby_axes(c)
    P = pos[ok]
    # systems: stars within 0.1 pc of one another counted once
    tree = cKDTree(P)
    keep = np.ones(len(P), bool)
    for i, j in sorted(tree.query_pairs(merge_pc)):
        if keep[i] and keep[j]:
            keep[j] = False
    P = P[keep]

    def ratio(X):
        d, _ = cKDTree(X).query(X, k=2)
        inner = np.linalg.norm(X, axis=1) < inner_pc
        n = len(X) / (4 / 3.0 * math.pi * radius_pc ** 3)
        return float(np.mean(d[inner, 1]) / (0.554 * n ** (-1 / 3.0)))
    obs = ratio(P)
    rng = np.random.default_rng(17)
    sims = []
    for _ in range(sims):
        u = rng.normal(size=(len(P), 3))
        u = u / np.linalg.norm(u, axis=1)[:, None] * (radius_pc * rng.random(len(P)) ** (1 / 3.0))[:, None]
        sims.append(ratio(u))
    sims = np.array(sims)
    p_reg = float(np.mean(sims >= obs))
    print("%d systems within 50 pc brighter than absolute G 10; Clark-Evans ratio %.3f against %.3f +- %.3f for uniform volumes (as high or higher in %.3f)" % (
        len(P), obs, float(sims.mean()), float(sims.std()), p_reg))
    if record:
        core.record_run("gaia_nearby_lattice", "A lattice among the nearby stars", "Gaia DR3 gaia_source (cached nearby sample)",
                        "systems within 50 pc brighter than absolute G 10: %d" % len(P), os.path.abspath(__file__), "lattice", len(P), "star systems",
                        dict(radius_pc=radius_pc, inner_pc=inner_pc, abs_g=abs_g, merge_pc=merge_pc, sims=len(sims)), "%d uniform volumes: %.3f +- %.3f" % (len(sims), float(sims.mean()), float(sims.std())),
                        "nearby star systems spaced more regularly than at random", "Clark-Evans ratio %.3f, as high or higher in %.3f of uniform volumes" % (obs, p_reg), [])
        print("recorded")


uncatalogued_DOC = r"""Bright stars no older catalogue holds: Gaia DR3 sources brighter than G 6 without a Hipparcos-2 or a Tycho-2 best neighbour; each read in SIMBAD for a name of its own, and those unnamed matched to Hipparcos (I/239) within 3 arcmin by CDS XMatch, a star within 5 arcsec being the same star; flagged where neither holds it.

    python -I seti.py neighbourhood uncatalogued [record] [--g-max 6] [--hip-arcsec 5] [--search-arcsec 180]"""


def uncatalogued_main(record=False, g_max=6.0, hip_arcsec=5.0, search_arcsec=180.0):
    rows = core.tap_async("gaia", "SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp, g.parallax, g.ruwe, g.astrometric_params_solved, "
                                  "g.duplicated_source FROM gaiadr3.gaia_source AS g LEFT OUTER JOIN gaiadr3.hipparcos2_best_neighbour AS h "
                                  "ON g.source_id = h.source_id LEFT OUTER JOIN gaiadr3.tycho2tdsc_merge_best_neighbour AS t ON g.source_id = t.source_id "
                                  "WHERE g.phot_g_mean_mag < %g AND h.source_id IS NULL AND t.source_id IS NULL" % g_max, maxrec=100000)
    names = simbad_names([r["source_id"] for r in rows])
    # the nearest Hipparcos star within 3 arcmin, whose glare can make a spurious source, by one cross-match of the unnamed
    near = {}
    for m in core.xmatch([dict(source_id="g%s" % r["source_id"], ra=float(r["ra"]), dec=float(r["dec"])) for r in rows if not names.get(r["source_id"])],
                         "I/239/hip_main", search_arcsec):
        k = m["source_id"][1:]
        if k not in near or float(m["angDist"]) < float(near[k]["sep"]):
            near[k] = dict(hip=m["HIP"], hp_mag=m.get("Hpmag"), sep=float(m["angDist"]))
    out = [dict(r, simbad=names.get(r["source_id"]), near=near.get(r["source_id"])) for r in rows]
    named = [o for o in out if o["simbad"]]
    print("%d Gaia DR3 sources brighter than G 6 without a Hipparcos-2 or Tycho-2 best neighbour; %d with a SIMBAD entry of their own" % (len(out), len(named)))
    for o in sorted(out, key=lambda o: (bool(o["simbad"]), float(o["phot_g_mean_mag"])))[:len(out) - len(named) + 5]:
        print("  Gaia DR3 %s: G %.2f, BP-RP %s, %s parameters, RUWE %s, duplicated %s; SIMBAD %s; nearest Hipparcos star %s" % (
            o["source_id"], float(o["phot_g_mean_mag"]), o["bp_rp"], o["astrometric_params_solved"], o["ruwe"], o["duplicated_source"],
            "%s (%s)" % o["simbad"] if o["simbad"] else "-", "HIP %s (Hp %s) at %.0f arcsec" % (o["near"]["hip"], o["near"]["hp_mag"], float(o["near"]["sep"]))
            if o["near"] else "none within 3 arcmin"))
    on_hip = [o for o in out if not o["simbad"] and o["near"] and o["near"]["sep"] < hip_arcsec]
    left = [o for o in out if not o["simbad"] and not (o["near"] and o["near"]["sep"] < hip_arcsec)]
    print("unnamed within 5 arcsec of a Hipparcos star: %d; neither named nor on a Hipparcos star: %d" % (len(on_hip), len(left)))
    if record:
        core.record_run("gaia_bright_uncatalogued", "Bright stars no older catalogue holds", "Gaia DR3 gaia_source, hipparcos2 and tycho2tdsc_merge best "
                        "neighbours; SIMBAD; Hipparcos (I/239) by CDS XMatch", "sources brighter than G 6 without a Hipparcos-2 or Tycho-2 best neighbour: %d"
                        % len(out), os.path.abspath(__file__), "uncatalogued", len(out), "Gaia sources", dict(g=g_max, hip_arcsec=hip_arcsec, search_arcsec=search_arcsec),
                        "SIMBAD's names and the nearest Hipparcos star", "a bright star no catalogue before Gaia held",
                        "%d named in SIMBAD, %d within 5 arcsec of a Hipparcos star, %d neither" % (len(named), len(on_hip), len(left)),
                        [dict(target="gaia_dr3:%s" % o["source_id"], stat=float(o["phot_g_mean_mag"]), unit="G mag",
                              note="nearest Hipparcos star %s" % ("HIP %s at %.0f arcsec" % (o["near"]["hip"], o["near"]["sep"]) if o["near"] else "none within 3 arcmin"))
                         for o in left])
        print("recorded")


galplx_DOC = r"""Galaxies with parallaxes: Gaia DR3 sources the discrete source classifier calls galaxies (combined probability above 0.9) with a parallax above 10 mas at 10 sigma, which nothing beyond the Galaxy can have; each read for its solution (RUWE, a second image peak, image asymmetry, excess noise), for BP/RP light that is not a single source's (the classifier reads the BP and RP spectra) and in SIMBAD.

    python -I seti.py neighbourhood galplx [record] [--prob 0.9] [--plx-mas 10] [--sigma 10] [--ruwe-max 1.4] [--multi-peak 2] [--harmonic 0.1] [--excess-noise-sig 2]"""


def galplx_main(record=False, prob=0.9, plx_mas=10.0, sigma=10.0, ruwe_max=1.4, multi_peak=2.0, harmonic=0.1, excess_noise_sig=2.0):
    rows = core.tap_async("gaia", "SELECT g.source_id, g.l, g.b, g.parallax, g.parallax_over_error, g.pm, g.phot_g_mean_mag, g.bp_rp, g.ruwe, "
                                  "g.ipd_frac_multi_peak, g.ipd_gof_harmonic_amplitude, g.astrometric_excess_noise_sig, g.phot_bp_rp_excess_factor, "
                                  "a.classprob_dsc_combmod_galaxy "
                                  "FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a ON g.source_id = a.source_id WHERE g.parallax > %g "
                                  "AND g.parallax_over_error > %g AND a.classprob_dsc_combmod_galaxy > %g" % (plx_mas, sigma, prob), maxrec=100000)

    def doubtful(r):
        # an astrometric solution Gaia's own image and fit statistics doubt: a poor fit, a second peak, an asymmetric image or excess noise
        return ((num(r["ruwe"]) or 0) > ruwe_max or (num(r["ipd_frac_multi_peak"]) or 0) > multi_peak or (num(r["ipd_gof_harmonic_amplitude"]) or 0) > harmonic
                or (num(r["astrometric_excess_noise_sig"]) or 0) > excess_noise_sig)
    def blended(r):
        # the classifier reads the BP and RP spectra, which a neighbour's light fills when the flux excess is not a single source's
        g, br, c = num(r["phot_g_mean_mag"]), num(r["bp_rp"]), num(r["phot_bp_rp_excess_factor"])
        return g is None or br is None or c is None or not clean_photometry(g, br, c)

    def vet(r):
        return "astrometry" if doubtful(r) else "blend" if blended(r) else None
    good = [r for r in rows if not doubtful(r)]
    names = simbad_names([r["source_id"] for r in good])
    g = np.array([num(r["phot_g_mean_mag"]) or np.nan for r in rows])
    b = np.array([abs(float(r["b"])) for r in rows])
    print("%d Gaia DR3 sources classed galaxies above 0.9 with parallaxes above 10 mas at 10 sigma: median G %.1f, %d within 15 deg of the plane; "
          "%d with a solution Gaia's statistics doubt" % (len(rows), float(np.nanmedian(g)), int((b < 15).sum()), len(rows) - len(good)))
    for r in good:
        print("  Gaia DR3 %s: l %.1f b %.1f, parallax %.1f (over error %.0f), pm %.0f mas/yr, G %s, BP-RP %s, RUWE %s; SIMBAD %s (%s)" % (
            r["source_id"], float(r["l"]), float(r["b"]), float(r["parallax"]), float(r["parallax_over_error"]), num(r["pm"]) or 0, r["phot_g_mean_mag"],
            r["bp_rp"], r["ruwe"], *names.get(r["source_id"], ("-", "-"))))
    if record:
        core.record_run("gaia_galaxies_with_parallax", "Galaxies with parallaxes", "Gaia DR3 gaia_source and astrophysical_parameters; SIMBAD",
                        "sources classed galaxies above 0.9 with parallaxes above 10 mas at 10 sigma: %d" % len(rows), os.path.abspath(__file__), "galplx",
                        len(rows), "Gaia sources", dict(prob=prob, plx_mas=plx_mas, sigma=sigma, ruwe=ruwe_max, multi_peak=multi_peak, harmonic=harmonic, excess_noise_sig=excess_noise_sig),
                        "a parallax that nothing extragalactic has", "a source Gaia calls a galaxy within 100 pc",
                        "%d, %d with a solution Gaia's statistics doubt and %d of the rest with BP/RP light from a neighbour; median G %.1f, %d within 15 deg "
                        "of the Galactic plane" % (len(rows), len(rows) - len(good), sum(blended(r) for r in good), float(np.nanmedian(g)), int((b < 15).sum())),
                        [dict(target="gaia_dr3:%s" % r["source_id"], stat=float(r["parallax"]), unit="parallax, mas", vet=vet(r),
                              note="G %s, l %.1f b %.1f, RUWE %s, galaxy probability %.2f" % (r["phot_g_mean_mag"], float(r["l"]), float(r["b"]), r["ruwe"],
                                                                                              float(r["classprob_dsc_combmod_galaxy"]))) for r in rows])
        print("recorded")


qsomove_DOC = r"""Quasars that move: Gaia DR3 quasar candidates both classifiers call quasars (classlabel_dsc_joint) with a QSOC redshift above 1, whose proper motion exceeds 10 mas/yr at 10 sigma, which no source at cosmological distance can show; each read for its solution, for BP/RP light that is not a single source's, and in SIMBAD.

    python -I seti.py neighbourhood qsomove [record] [--pm-masyr 10] [--sigma 10] [--z-min 1] [--ruwe-max 1.4] [--multi-peak 2] [--harmonic 0.1] [--excess-noise-sig 2]"""


def qsomove_main(record=False, pm_masyr=10.0, sigma=10.0, z_min=1.0, ruwe_max=1.4, multi_peak=2.0, harmonic=0.1, excess_noise_sig=2.0):
    lean = core.gaia_by_range("SELECT q.source_id, q.redshift_qsoc, q.flags_qsoc, g.pm, g.pmra_error, g.pmdec_error FROM gaiadr3.qso_candidates AS q "
                              "JOIN gaiadr3.gaia_source AS g ON q.source_id = g.source_id WHERE q.source_id BETWEEN {lo} AND {hi} AND "
                              "q.classlabel_dsc_joint = 'quasar' AND q.redshift_qsoc > %g AND g.pm > %g" % (z_min, pm_masyr), maxrec=200000)
    lean = {r["source_id"]: r for r in lean if float(r["pm"]) > sigma * math.hypot(float(r["pmra_error"]), float(r["pmdec_error"]))}
    full = core.tap_by_ids("gaia", list(lean), "SELECT source_id, l, b, parallax, parallax_over_error, phot_g_mean_mag, bp_rp, ruwe, ipd_frac_multi_peak, "
                                               "ipd_gof_harmonic_amplitude, astrometric_excess_noise_sig, phot_bp_rp_excess_factor FROM gaiadr3.gaia_source "
                                               "WHERE source_id IN (%s)") if lean else {}
    rows = [dict(full[k], **lean[k]) for k in lean if k in full]

    def doubtful(r):
        return ((num(r["ruwe"]) or 0) > ruwe_max or (num(r["ipd_frac_multi_peak"]) or 0) > multi_peak or (num(r["ipd_gof_harmonic_amplitude"]) or 0) > harmonic
                or (num(r["astrometric_excess_noise_sig"]) or 0) > excess_noise_sig)

    def blended(r):
        g, br, c = num(r["phot_g_mean_mag"]), num(r["bp_rp"]), num(r["phot_bp_rp_excess_factor"])
        return g is None or br is None or c is None or not clean_photometry(g, br, c)
    good = [r for r in rows if not doubtful(r) and not blended(r)]
    names = simbad_names([r["source_id"] for r in good])
    print("%d joint-class quasars at QSOC redshift above 1 moving above 10 mas/yr at 10 sigma; %d with doubtful astrometry, %d more with a neighbour's "
          "BP/RP light; clean %d" % (len(rows), sum(doubtful(r) for r in rows), sum(blended(r) and not doubtful(r) for r in rows), len(good)))
    for r in good[:30]:
        print("  Gaia DR3 %s: pm %.0f mas/yr, parallax %.1f (over error %.0f), G %s, BP-RP %s, z_QSOC %s (flags %s); SIMBAD %s (%s)" % (
            r["source_id"], float(r["pm"]), float(r["parallax"]), float(r["parallax_over_error"]), r["phot_g_mean_mag"], r["bp_rp"], r["redshift_qsoc"],
            r["flags_qsoc"], *names.get(r["source_id"], ("-", "-"))))
    if record:
        core.record_run("gaia_moving_quasars", "Quasars that move", "Gaia DR3 qso_candidates and gaia_source; SIMBAD",
                        "joint-class quasars at QSOC redshift above 1 moving above 10 mas/yr at 10 sigma: %d" % len(rows), os.path.abspath(__file__), "qsomove",
                        len(rows), "Gaia sources", dict(pm_masyr=pm_masyr, sigma=sigma, z=z_min, ruwe=ruwe_max, multi_peak=multi_peak, harmonic=harmonic, excess_noise_sig=excess_noise_sig),
                        "a proper motion nothing at cosmological distance shows",
                        "a quasar that moves across the sky", "%d, %d with doubtful astrometry or a neighbour's light; %d clean" % (
                            len(rows), len(rows) - len(good), len(good)),
                        [dict(target="gaia_dr3:%s" % r["source_id"], stat=float(r["pm"]), unit="proper motion, mas/yr",
                              vet="astrometry" if doubtful(r) else "blend" if blended(r) else None,
                              note="z_QSOC %s, flags %s, parallax %.1f mas; SIMBAD %s (%s)" % (r["redshift_qsoc"], r["flags_qsoc"], float(r["parallax"]),
                                                                                              *names.get(r["source_id"], ("-", "-")))) for r in rows])
        print("recorded")


ETZ_HALF_DEG = 0.264        # Earth is seen to cross the Sun from within this ecliptic latitude


def etz_stars():
    """Gaia Catalogue of Nearby Stars objects (probability above 0.9) within ETZ_HALF_DEG of the ecliptic, which see Earth cross the Sun, and those 1-5 deg from it, each with its position, proper motion and parallax at 2016.0 and its ecliptic latitude; kept in the results directory as gcns_etz.json."""
    path = core.results("gcns_etz.json")
    if os.path.exists(path):
        return json.load(open(path))
    eps = math.radians(23.4392911)
    rows = catalogues.gcns("source_id, ra, dec, pmra, pmdec, parallax", ranges=12)
    out = dict(zone=[], near=[])
    for r in rows:
        a, d = math.radians(float(r["ra"])), math.radians(float(r["dec"]))
        beta = abs(math.degrees(math.asin(math.sin(d) * math.cos(eps) - math.cos(d) * math.sin(eps) * math.sin(a))))
        if beta < ETZ_HALF_DEG or 1 <= beta < 5:
            out["zone" if beta < ETZ_HALF_DEG else "near"].append(dict(id=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"]), pmra=num(r["pmra"]) or 0.0,
                                                                        pmdec=num(r["pmdec"]) or 0.0, plx=float(r["parallax"]), beta=round(beta, 4)))
    json.dump(out, open(path, "w"))
    return out


def etz_share(cat, radius, epoch, keep=None):
    """Stars of the transit zone and of the band 1-5 deg beside it (KEEP selects by position) with a source of the CDS catalogue CAT within RADIUS arcsec at EPOCH: the zone's matches (nearest per star), the band's matched count and size, the count the band's share predicts for the zone, its Poisson p, and the zone's matches moved 2 arcmin in eight directions."""
    from scipy.stats import poisson
    from setitools.catalogues import shifted_matches
    st = etz_stars()

    def place(s):
        ra, de = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, epoch)
        return dict(name="g%s" % s["id"], ra=ra % 360, dec=de)
    zone = [place(s) for s in st["zone"] if keep is None or keep(s)]
    near = [place(s) for s in st["near"] if keep is None or keep(s)]
    hits, null = shifted_matches(zone, cat, radius)
    got = {}
    for m in hits:
        if m["source_id"] not in got or float(m["angDist"]) < float(got[m["source_id"]]["angDist"]):
            got[m["source_id"]] = m
    far = set()
    for k in range(0, len(near), 5000):
        far |= {m["source_id"] for m in core.xmatch([dict(source_id=p["name"], ra=p["ra"], dec=p["dec"]) for p in near[k:k + 5000]], cat, radius)}
    exp = len(far) / max(len(near), 1) * len(zone)
    return dict(zone=len(zone), near=len(near), got=got, far=len(far), exp=exp, p=float(poisson.sf(len(got) - 1, exp)) if got else 1.0, null=null)


def etz_vets(got):
    """SIMBAD's name, type and spectral type for each matched zone star, and an active-star vet where the type says so."""
    sim = {}
    if got:
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % n[1:] for n in got)):
            sim["g" + str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    active = ("BY*", "RS*", "Er*", "Fl*", "TT*", "Y*O", "Y*?", "EB*", "SB*", "**", "CV*", "WD*", "Ro*", "El*", "Em*", "X", "Ae*", "Be*", "a2*")
    return {n: (sim.get(n), "active-star" if sim.get(n) and sim[n][1] in active else None) for n in got}


encounters_DOC = r"""Stars that passed within a parsec of the Sun: Bailer-Jones (2022) encounter stars from Gaia EDR3 with a median perihelion distance below 1 pc (J/ApJ/935/L9), past passages and future ones apart; each read in SIMBAD, matched to eRASS1 within 15 arcsec (2020.0), VLASS epoch-1 components within 3 arcsec (2019.0), GALEX GUVcat AIS within 5 arcsec (2007.0) and AllWISE within 3 arcsec for W1-W3 redder than 0.3 mag, each against the stars moved 2 arcmin in eight directions, and looked up among Breakthrough Listen's open-data targets by HIP and GJ name; a civilisation at one of these stars had the Sun within a parsec. X-rays or ultraviolet at a cool dwarf are its corona and chromosphere, and a W1-W3 under 0.7 at a red star its photosphere.

    python -I seti.py neighbourhood encounters [record] [--w13-min 0.3] [--w3-err-max 0.2]"""


def encounters_main(record=False, w13_min=0.3, w3_err_max=0.2):
    from setitools.catalogues import shifted_matches
    rows = core.tap("vizier", 'SELECT * FROM "J/ApJ/935/L9/table12"')
    st = []
    for r in rows:
        st.append(dict(id=r["GaiaDR3"], t_kyr=num(r["tphmed"]), d_pc=num(r["dphmed"]), d95=num(r["dph95"]), v=num(r["vphmed"]), plx=num(r["Plx"]), g=num(r["Gmag"]),
                       bprp=num(r["BP-RP"])))
    pm = {r["source_id"]: r for r in core.tap("gaia", "SELECT source_id, ra, dec, pmra, pmdec FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(s["id"] for s in st))}
    for s in st:
        s["ra"], s["dec"] = float(pm[s["id"]]["ra"]), float(pm[s["id"]]["dec"])
        s["pmra"], s["pmdec"] = num(pm[s["id"]]["pmra"]) or 0.0, num(pm[s["id"]]["pmdec"]) or 0.0
    names = {}
    for x in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type, b.oid FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
            "'Gaia DR3 %s'" % s["id"] for s in st)):
        names[str(x["id"]).split()[-1]] = dict(main=" ".join(str(x["main_id"]).split()), otype=str(x["otype"]).strip(), sp=(x["sp_type"] or "").strip(), oid=str(x["oid"]), ids=[])
    for x in core.simbad("SELECT oidref, id FROM ident WHERE oidref IN (%s) AND (id LIKE 'HIP %%' OR id LIKE 'GJ %%' OR id LIKE 'NAME %%')" % ",".join(
            n["oid"] for n in names.values())):
        for n in names.values():
            if n["oid"] == str(x["oidref"]):
                n["ids"].append(str(x["id"]))

    def at(epoch):
        return [dict(name=s["id"], **dict(zip(("ra", "dec"), core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, epoch)))) for s in st]
    found = {}
    nulls = {}
    for key, cat, radius, epoch, keyfn in (("eRASS1", "J/A+A/682/A34/erass1-m", 15.0, 2020.0, None), ("VLASS", "J/ApJS/255/30/comp", 3.0, 2019.0, None),
                                           ("GALEX", "II/335/galex_ais", 5.0, 2007.0, None),
                                           ("warm", "II/328/allwise", 3.0, 2010.5, lambda m: num(m.get("W3mag")) is not None and num(m.get("e_W3mag")) is not None
                                            and num(m["e_W3mag"]) < w3_err_max and num(m["W1mag"]) - num(m["W3mag"]) > w13_min)):
        hits, null = shifted_matches(at(epoch), cat, radius, keyfn=keyfn)
        nulls[key] = null
        for m in hits:
            label = ("W1-W3 %.2f" % (num(m["W1mag"]) - num(m["W3mag"])) if key == "warm" else
                     m.get("IAUName") or m.get("CompName") or m.get("Name") or m.get("objid") or "%.1f arcsec" % float(m["angDist"]))
            found.setdefault(m["source_id"], {})[key] = label
    bl = {}
    for s in st:
        for ident in names.get(s["id"], {}).get("ids", []):
            tag = ident.replace("HIP ", "HIP").replace("GJ ", "GJ").replace("NAME ", "").replace(" ", "")
            try:
                files = core.bl_query_files(tag)
            except Exception:
                files = []
            if files:
                bl[s["id"]] = (tag, len(files))
                break
    past = [s for s in st if s["t_kyr"] is not None and s["t_kyr"] < 0]
    print("%d stars within 1 pc at perihelion, %d past and %d to come; matches %s against moved %s; Breakthrough Listen has observed %d" % (
        len(st), len(past), len(st) - len(past), {k: sum(1 for f in found.values() if k in f) for k in nulls}, nulls, len(bl)))
    for s in sorted(st, key=lambda s: s["d_pc"]):
        n = names.get(s["id"], {})
        print("  Gaia DR3 %s %s (%s, %s): %.3f pc %+.0f kyr at %.0f km/s, now %.1f pc, G %.1f; %s; BL %s" % (
            s["id"], n.get("main", "-"), n.get("otype", "-"), n.get("sp", "-"), s["d_pc"], s["t_kyr"], s["v"], 1000 / s["plx"], s["g"], found.get(s["id"], {}), bl.get(s["id"])))
    if record:
        act = ("BY*", "RS*", "Er*", "Fl*", "TT*", "Y*O", "EB*", "**", "SB*", "CV*", "WD*", "Ro*", "Em*", "X", "LM*", "BD*", "LP*", "Mi*", "RG*", "s*r", "sr*")

        def vet(s):
            f = found.get(s["id"])
            if not f:
                return "not-measured"
            if names.get(s["id"], {}).get("otype") in act:
                return "natural-type"
            cool = s["bprp"] is not None and s["bprp"] > 0.9
            if ("eRASS1" in f or "GALEX" in f) and cool:
                return "active-star"
            if set(f) == {"warm"} and cool and float(f["warm"].split()[-1]) < 0.7:
                return "photosphere"
            return None
        core.record_run("close_encounter_stars", "Stars that passed within a parsec of the Sun", "Bailer-Jones (2022) Gaia EDR3 encounters (J/ApJ/935/L9); eRASS1, VLASS, GALEX "
                        "and AllWISE by CDS XMatch; SIMBAD; Breakthrough Listen open data", "stars with a median perihelion below 1 pc: %d" % len(st),
                        os.path.abspath(__file__), "encounters", len(st), "stars", dict(perihelion_pc=1, radius_arcsec=dict(erass1=15, vlass=3, galex=5, allwise=3), w13_min=w13_min,
                                                                                          w3_err_max=w3_err_max),
                        "the stars moved 2 arcmin: %s" % nulls, "emission from a star that had the Sun within a parsec",
                        "%s against means %s moved; %d observed by Breakthrough Listen" % ({k: sum(1 for f in found.values() if k in f) for k in nulls},
                                                                                            {k: round(float(np.mean(v)), 1) for k, v in nulls.items()}, len(bl)),
                        [dict(target="gaia_dr3:%s" % s["id"], stat=s["d_pc"], unit="perihelion, pc", vet=vet(s),
                              note="%s (%s); %+.0f kyr at %.0f km/s; %s; BL %s" % (names.get(s["id"], {}).get("main"), names.get(s["id"], {}).get("otype"), s["t_kyr"], s["v"],
                                                                                   found.get(s["id"], {}), bl.get(s["id"]))) for s in st])
        print("recorded")


nearlens_DOC = r"""Microlensing of the neighbours: Gaia DR3's 363 microlensing events (vari_microlensing) whose lensed source has a parallax above 1 mas at 3 sigma; a lens in front of a star within a kiloparsec is nearer still, and a timescale of weeks to months then needs a lens of stellar mass or more with no star seen, a dark massive body in the solar neighbourhood. Each read for Gaia's own doubt of the parallax (RUWE above 1.4, or under 5 sigma at a source fainter than G 19), for a SIMBAD type that varies on its own, for Gaia's variability class, and for the blended fit's source fraction, under which half the light and the parallax belong to another star; a clean event's lens may sit just in front of its source at any small mass, which these fits cannot exclude.

    python -I seti.py neighbourhood nearlens [record] [--plx-mas 1] [--sigma 3] [--ruwe-max 1.4] [--faint-g 19] [--faint-sigma 5] [--source-fraction 0.5] [--relative-kms 50]"""


def nearlens_main(record=False, plx_mas=1.0, sigma=3.0, ruwe_max=1.4, faint_g=19.0, faint_sigma=5.0, source_fraction=0.5, relative_kms=50.0):
    rows = core.tap("gaia", "SELECT m.source_id, m.paczynski0_te, m.paczynski0_u0, m.paczynski0_g0, m.paczynski1_fs_g, m.paczynski1_fs_g_error, g.parallax, "
                            "g.parallax_over_error, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.ra, g.dec, g.l, g.b, c.best_class_name, c.best_class_score FROM "
                            "gaiadr3.vari_microlensing AS m JOIN gaiadr3.gaia_source AS g ON m.source_id = g.source_id LEFT OUTER JOIN gaiadr3.vari_classifier_result AS c "
                            "ON m.source_id = c.source_id WHERE g.parallax > %g AND g.parallax_over_error > %g" % (plx_mas, sigma), timeout=300)
    total = int(core.tap("gaia", "SELECT COUNT(*) AS n FROM gaiadr3.vari_microlensing")[0]["n"])
    sim = {}
    for x in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
            "'Gaia DR3 %s'" % r["source_id"] for r in rows)):
        sim[str(x["id"]).split()[-1]] = (" ".join(str(x["main_id"]).split()), str(x["otype"]).strip(), (x["sp_type"] or "").strip())
    varies = ("V*", "LP*", "Mi*", "sr*", "RR*", "Ce*", "EB*", "CV*", "No*", "Y*O", "TT*", "Or*", "BY*", "RS*", "Er*", "Fl*", "Sy*", "AB*", "RG*", "s*r", "pA*", "Ro*", "Pu*")
    out = []
    for r in rows:
        s = sim.get(r["source_id"])
        plx, te = float(r["parallax"]), float(r["paczynski0_te"])
        # the lens mass at half the source's distance for a relative motion of 50 km/s: tE = theta_E / mu, theta_E = 2.85 mas sqrt(M pi_rel)
        mu = relative_kms / 4.74 * 2 * plx
        mass = (te / 365.25 * mu / 2.85) ** 2 / plx
        fs = num(r["paczynski1_fs_g"])
        # a parallax under 5 sigma at a faint source is the faint end's tail of spurious solutions; a source carrying under half the light lends its parallax to the blend;
        # a clean event's lens may sit just in front of its source at any small mass, which these columns cannot exclude
        vet = ("astrometry" if float(r["ruwe"]) > ruwe_max or (float(r["parallax_over_error"]) < faint_sigma and float(r["phot_g_mean_mag"]) > faint_g) else
               "natural-type" if s and s[1] in varies else "variable-star" if r["best_class_name"] and r["best_class_name"] != "MICROLENSING" else
               "blend" if fs is not None and fs < source_fraction else "not-measured")
        out.append(dict(id=r["source_id"], plx=round(plx, 2), poe=round(float(r["parallax_over_error"]), 1), te=round(te, 1), u0=round(float(r["paczynski0_u0"]), 2),
                        g=round(float(r["phot_g_mean_mag"]), 1), ruwe=round(float(r["ruwe"]), 2), b=round(float(r["b"]), 1), cls=r["best_class_name"], simbad=s,
                        fs=fs, mass=round(mass, 1), vet=vet))
    left = [o for o in out if o["vet"] == "not-measured"]
    print("%d Gaia DR3 microlensing events, %d on sources with parallax above 1 mas at 3 sigma; %d with doubtful astrometry, %d of a variable type, %d whose source is "
          "under half the light; %d clean events whose lens the fits cannot place %s" % (total, len(out), sum(1 for o in out if o["vet"] == "astrometry"),
                                                                                      sum(1 for o in out if o["vet"] in ("natural-type", "variable-star")),
                                                                                      sum(1 for o in out if o["vet"] == "blend"), len(left), [o["id"] for o in left]))
    for o in sorted(out, key=lambda o: -o["plx"]):
        print("  Gaia DR3 %s: parallax %.2f mas (%.1f sigma), tE %.1f d, u0 %.2f, G %.1f, RUWE %.2f, b %.1f, source fraction %s; class %s; SIMBAD %s; lens at half the "
              "distance %.1f Msun; %s" % (o["id"], o["plx"], o["poe"], o["te"], o["u0"], o["g"], o["ruwe"], o["b"], o["fs"], o["cls"], o["simbad"], o["mass"], o["vet"]))
    if record:
        core.record_run("gaia_microlensing_near", "Microlensing of the neighbours", "Gaia DR3 vari_microlensing, gaia_source and vari_classifier_result; SIMBAD",
                        "microlensing events on sources with parallax above 1 mas at 3 sigma: %d of %d" % (len(out), total), os.path.abspath(__file__), "nearlens", len(out),
                        "events", dict(parallax_mas=plx_mas, sigma=sigma, ruwe=ruwe_max, faint_g=faint_g, faint_sigma=faint_sigma, source_fraction=source_fraction,
                                       relative_kms=relative_kms), "none: each event read for its astrometry, type, class and blending",
                        "a lens of stellar mass or more within a kiloparsec with no star seen", "%d events; %d clean, whose lenses the fits leave anywhere in front of the "
                        "source at any small mass %s" % (len(out), len(left), [o["id"] for o in left]),
                        [dict(target="gaia_dr3:%s" % o["id"], stat=o["mass"], unit="lens mass at half the source distance, Msun", vet=o["vet"],
                              note="parallax %.2f mas at %.1f sigma, tE %.1f d, RUWE %.2f; class %s; SIMBAD %s" % (o["plx"], o["poe"], o["te"], o["ruwe"], o["cls"], o["simbad"]))
                         for o in out])
        print("recorded")


hvsorigin_DOC = r"""Unbound stars from nowhere: the 52 hypervelocity-star candidates of Li et al. (2023; J/AJ/166/12) with Gaia proper motions, velocities and distances, each placed in Galactocentric coordinates and traced straight back along its velocity for 300 Myr: a star thrown by the central black hole passes within 1 kpc of the Centre, one from the Large Magellanic Cloud within 5 kpc of the Cloud, and a disc runaway crossed the disc within 25 kpc of the Centre or sits within 1.5 kpc of it still; a star whose path back does none of these was launched some other way. A velocity error above 150 km/s leaves the path unmeasured, and a star more likely bound than not in the catalogue's own Milky Way potential is a halo star whose speed its distance error inflates.

    python -I seti.py neighbourhood hvsorigin [record] [--centre-kpc 1] [--cloud-kpc 5] [--disc-kpc 25] [--disc-z-kpc 1.5] [--within-myr 300] [--velocity-error-max 150]"""


def hvsorigin_main(record=False, centre_kpc=1.0, cloud_kpc=5.0, disc_kpc=25.0, disc_z_kpc=1.5, within_myr=300.0, velocity_error_max=150.0):
    from astropy.coordinates import SkyCoord, Galactocentric
    import astropy.units as u
    rows = core.tap("vizier", 'SELECT Name, Gaia, RAJ2000, DEJ2000, pmRA, e_pmRA, pmDE, e_pmDE, Vlos, e_Vlos, Dist, e_Dist, VGSR, e_VGSR, "Pub-MW" FROM "J/AJ/166/12/table2"')
    frame = Galactocentric()
    lmc = SkyCoord(80.894 * u.deg, -69.756 * u.deg, distance=49.6 * u.kpc).transform_to(frame)
    lmc_xyz = np.array([lmc.x.value, lmc.y.value, lmc.z.value])
    out = []
    for r in rows:
        c = SkyCoord(float(r["RAJ2000"]) * u.deg, float(r["DEJ2000"]) * u.deg, distance=float(r["Dist"]) * u.kpc, pm_ra_cosdec=float(r["pmRA"]) * u.mas / u.yr,
                     pm_dec=float(r["pmDE"]) * u.mas / u.yr, radial_velocity=float(r["Vlos"]) * u.km / u.s).transform_to(frame)
        x = np.array([c.x.value, c.y.value, c.z.value])
        v = np.array([c.v_x.value, c.v_y.value, c.v_z.value])
        speed = float(np.linalg.norm(v))
        a_gc = float(np.degrees(np.arccos(np.clip(np.dot(x, v) / (np.linalg.norm(x) * speed), -1, 1))))
        d = x - lmc_xyz
        a_lmc = float(np.degrees(np.arccos(np.clip(np.dot(d, v) / (np.linalg.norm(d) * speed), -1, 1))))
        # the straight line back along the velocity (kpc, Myr): the closest approach to the Centre and to the Cloud, and where it last crossed the disc
        kpc_per_myr = speed * 1.0227e-3
        unit = v / speed
        t_gc = max(float(np.dot(x, unit)) / kpc_per_myr, 0.0)
        d_gc = float(np.linalg.norm(x - unit * t_gc * kpc_per_myr))
        t_lmc = max(float(np.dot(d, unit)) / kpc_per_myr, 0.0)
        d_lmc = float(np.linalg.norm(d - unit * t_lmc * kpc_per_myr))
        t_disc = float(x[2] / v[2]) * 977.8 if v[2] != 0 else float("inf")
        r_disc = float(np.hypot(*(x[:2] - v[:2] * t_disc / 977.8))) if 0 < t_disc < within_myr else None
        err = float(r["e_VGSR"])
        # a star still inside the disc is a disc star whatever its speed; one above it is a disc runaway when its path back crosses the disc within 25 kpc
        p_unbound = num(r["Pub-MW"]) or 0.0
        origin = ("not-measured" if err > velocity_error_max else "Galactic Centre" if d_gc < centre_kpc and t_gc < within_myr else
                  "Large Magellanic Cloud" if d_lmc < cloud_kpc and t_lmc < within_myr else
                  "disc" if abs(x[2]) < disc_z_kpc or (r_disc is not None and r_disc < disc_kpc) else "halo" if p_unbound < 0.5 else None)
        out.append(dict(name=r["Name"].strip(), gaia=r["Gaia"], dist_kpc=float(r["Dist"]), vgsr=float(r["VGSR"]), e_vgsr=err, p_unbound=num(r["Pub-MW"]), speed=round(speed),
                        a_gc=round(a_gc), a_lmc=round(a_lmc), d_gc=round(d_gc, 1), d_lmc=round(d_lmc, 1), t_disc=round(t_disc) if r_disc is not None else None,
                        r_disc=round(r_disc, 1) if r_disc is not None else None, z=round(float(x[2]), 1), origin=origin))
    left = [o for o in out if o["origin"] is None]
    from collections import Counter
    print("%d hypervelocity candidates: origins %s; from nowhere %d %s" % (len(out), dict(Counter(o["origin"] for o in out)), len(left),
                                                                          [(o["name"], o["speed"], o["d_gc"], o["d_lmc"], o["z"]) for o in left]))
    for o in sorted(out, key=lambda o: -o["speed"]):
        print("  %s: %d +- %.0f km/s at %.1f kpc (unbound with probability %s); passed %.1f kpc from the Centre and %.1f from the Cloud, crossed the disc %s Myr ago at R %s kpc, "
              "z now %.1f kpc; %s" % (o["name"], o["speed"], o["e_vgsr"], o["dist_kpc"], o["p_unbound"], o["d_gc"], o["d_lmc"], o["t_disc"], o["r_disc"], o["z"], o["origin"]))
    if record:
        core.record_run("hvs_origins", "Unbound stars from nowhere", "Li et al. (2023) hypervelocity-star candidates (J/AJ/166/12) with Gaia astrometry",
                        "hypervelocity candidates: %d" % len(out), os.path.abspath(__file__), "hvsorigin", len(out), "stars",
                        dict(centre_kpc=centre_kpc, cloud_kpc=cloud_kpc, disc_kpc=disc_kpc, disc_z_kpc=disc_z_kpc, within_myr=within_myr, velocity_error_max=velocity_error_max),
                        "none: each star's straight path back against the three natural launch sites",
                        "an unbound star whose path back passes no Centre, Cloud or disc", "%s; %d from nowhere" % (dict(Counter(o["origin"] for o in out)), len(left)),
                        [dict(target="gaia_dr3:%s" % o["gaia"], stat=o["speed"], unit="Galactocentric speed, km/s",
                              vet="not-measured" if o["origin"] == "not-measured" else "natural-source" if o["origin"] else None, p=o["p_unbound"],
                              note="%s; %.1f kpc from the Centre and %.1f from the Cloud on the way back, disc crossing %s Myr ago at R %s kpc; velocity error %.0f km/s; unbound "
                                   "with probability %s" % (o["name"], o["d_gc"], o["d_lmc"], o["t_disc"], o["r_disc"], o["e_vgsr"], o["p_unbound"])) for o in out])
        print("recorded")


siblings_DOC = r"""The solar siblings: HD 162826 and HD 186302, the two stars whose composition and age place their birth in the Sun's own cluster, read in eRASS1 (15 arcsec, 2020.0), VLASS (3 arcsec, 2019.0), GALEX (5 arcsec, 2007.0) and AllWISE (3 arcsec, W1-W3) by CDS XMatch, and looked up among Breakthrough Listen's targets; a civilisation there is the Sun's contemporary.

    python -I seti.py neighbourhood siblings [record] [--stars "HD 162826" "HD 186302"]"""

SIBLINGS = ("HD 162826", "HD 186302")


def siblings_main(record=False, stars=list(SIBLINGS)):
    st = []
    for x in core.simbad("SELECT b.main_id, b.ra, b.dec, b.pmra, b.pmdec, b.plx_value, b.sp_type, b.oid FROM basic AS b WHERE b.main_id IN (%s)" % ",".join("'%s'" % n for n in stars)):
        ids = [str(i["id"]) for i in core.simbad("SELECT id FROM ident WHERE oidref = %s AND (id LIKE 'HIP %%' OR id LIKE 'GJ %%' OR id LIKE 'Gaia DR3 %%')" % x["oid"])]
        st.append(dict(name=" ".join(str(x["main_id"]).split()), ra=float(x["ra"]), dec=float(x["dec"]), pmra=num(x["pmra"]) or 0.0, pmdec=num(x["pmdec"]) or 0.0,
                       plx=num(x["plx_value"]), sp=x["sp_type"], ids=ids))
    found = {s["name"]: {} for s in st}
    for key, cat, radius, epoch in (("eRASS1", "J/A+A/682/A34/erass1-m", 15.0, 2020.0), ("VLASS", "J/ApJS/255/30/comp", 3.0, 2019.0), ("GALEX", "II/335/galex_ais", 5.0, 2007.0),
                                    ("AllWISE", "II/328/allwise", 3.0, 2010.5)):
        up = []
        for s in st:
            ra, de = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2000.0, epoch)
            up.append(dict(source_id=s["name"].replace(" ", "_"), ra=ra, dec=de))
        for m in core.xmatch(up, cat, radius):
            name = m["source_id"].replace("_", " ")
            found[name][key] = (m.get("IAUName") or m.get("CompName") or ("NUV %s FUV %s" % (m.get("NUVmag"), m.get("FUVmag"))) if key != "AllWISE" else
                                "W1-W3 %.2f" % (num(m["W1mag"]) - num(m["W3mag"])) if num(m.get("W3mag")) is not None else "W3 unmeasured")
    bl = {}
    for s in st:
        for ident in s["ids"]:
            tag = ident.replace("HIP ", "HIP").replace("GJ ", "GJ").replace(" ", "")
            try:
                files = core.bl_query_files(tag)
            except Exception:
                files = []
            if files:
                bl[s["name"]] = (tag, len(files))
                break
    for s in st:
        print("%s (%s, %.1f pc): %s; Breakthrough Listen %s" % (s["name"], s["sp"], 1000 / s["plx"], found[s["name"]], bl.get(s["name"])))
    if record:
        core.record_run("solar_siblings", "The solar siblings", "SIMBAD; eRASS1, VLASS, GALEX and AllWISE by CDS XMatch; Breakthrough Listen open data",
                        " and ".join(stars), os.path.abspath(__file__), "siblings", len(st), "stars", dict(stars=list(stars), radius_arcsec=dict(erass1=15, vlass=3, galex=5, allwise=3)),
                        "none: two stars read in each archive", "emission from a star born beside the Sun", "; ".join("%s %s, BL %s" % (s["name"], found[s["name"]], bl.get(s["name"])) for s in st),
                        [dict(target="simbad:%s" % s["name"], vet="not-measured" if not found[s["name"]] and not bl.get(s["name"]) else "photosphere",
                              note="%s; BL %s" % (found[s["name"]], bl.get(s["name"]))) for s in st])
        print("recorded")


comovers_DOC = r"""The Sun's co-movers against the Sun: the stars the store holds within 1 km/s of the Sun's velocity (`gaia_rest_stars`), each read in Gaia DR3's astrophysical parameters for its metallicity (GSP-Spec, else GSP-Phot), alpha enhancement and FLAME age against the Sun's own (metallicity 0.00, alpha 0.00, 4.57 Gyr); a star born in the Sun's cluster shares its composition and age to 0.1 dex and a gigayear, so a co-mover that matches is a sibling and one that does not is either a chance alignment in velocity or something holding station with the Sun. Across a store of several hundred tests the lowest p of 0.004 is itself the expected extreme.

    python -I seti.py neighbourhood comovers [record] [--metallicity-dex 0.1] [--age-gyr 1.5] [--teff-min 4800]"""


def comovers_main(record=False, metallicity_dex=0.1, age_gyr=1.5, teff_min=4800.0):
    import sqlite3
    db = sqlite3.connect(core.COVERAGE)
    rows = db.execute("SELECT target, stat, note FROM coverage WHERE search_id = 'gaia_rest_stars'").fetchall()
    ids = [t.split(":")[-1] for t, _, _ in rows]
    ap = {r["source_id"]: r for r in core.tap("gaia", "SELECT source_id, teff_gspphot, mh_gspphot, mh_gspspec, alphafe_gspspec, age_flame, age_flame_lower, age_flame_upper, "
                                                      "mass_flame FROM gaiadr3.astrophysical_parameters WHERE source_id IN (%s)" % ",".join(ids))}
    sim = {}
    for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join("'Gaia DR3 %s'" % i for i in ids)):
        sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    out = []
    for (t, speed, note), i in zip(rows, ids):
        a = ap.get(i, {})
        mh = num(a.get("mh_gspspec")) if num(a.get("mh_gspspec")) is not None else num(a.get("mh_gspphot"))
        age = num(a.get("age_flame"))
        teff = num(a.get("teff_gspphot"))
        # the comparison holds for stars warm enough for Gaia's spectroscopic parameters; a cool dwarf's metallicity from Gaia is not to be trusted
        measured = teff is not None and teff > teff_min and mh is not None
        sibling = measured and abs(mh) < metallicity_dex and (age is None or abs(age - 4.57) < age_gyr)
        out.append(dict(id=i, speed=round(float(speed), 2), simbad=sim.get(i), teff=teff, mh=mh, alpha=num(a.get("alphafe_gspspec")), age=age,
                        age_lo=num(a.get("age_flame_lower")), age_hi=num(a.get("age_flame_upper")), measured=measured, sibling=sibling,
                        vet="moving-group" if "moving-group" in (note or "") else "not-measured" if not measured else "chance-rate", note=note))
    print("%d stars within 1 km/s of the Sun; %d warm enough to compare; siblings by composition and age: %d" % (len(out), sum(1 for o in out if o["measured"]),
                                                                                                                sum(1 for o in out if o["sibling"])))
    for o in out:
        print("  Gaia DR3 %s %s: %.2f km/s; %s K, [M/H] %s, alpha %s, age %s (%s-%s) Gyr; %s" % (o["id"], o["simbad"], o["speed"], o["teff"], o["mh"], o["alpha"], o["age"],
                                                                                                 o["age_lo"], o["age_hi"], "sibling" if o["sibling"] else o["vet"]))
    if record:
        core.record_run("sun_comovers_composition", "The Sun's co-movers against the Sun", "the store's gaia_rest_stars targets; Gaia DR3 astrophysical_parameters; SIMBAD",
                        "stars within 1 km/s of the Sun's velocity: %d" % len(out), os.path.abspath(__file__), "comovers", len(out), "stars",
                        dict(metallicity_dex=metallicity_dex, age_gyr=age_gyr, teff_min=teff_min), "the Sun's own composition and age",
                        "a co-mover that shares the Sun's composition and age, a sibling, or one that does not and still keeps its velocity",
                        "%d of %d comparable; %d siblings; the warm co-movers are 0.24-0.34 dex metal-poor and 1-2 Gyr younger than the Sun" % (
                            sum(1 for o in out if o["measured"]), len(out), sum(1 for o in out if o["sibling"])),
                        [dict(target="gaia_dr3:%s" % o["id"], stat=o["mh"], unit="metallicity, dex", vet=o["vet"], note="%s; %s K, alpha %s, age %s Gyr; %.2f km/s from the Sun; %s" % (
                            o["simbad"], o["teff"], o["alpha"], o["age"], o["speed"], o["note"])) for o in out])
        print("recorded")


trojanpoints_DOC = r"""The triangular points of nearby binaries: the Sixth Catalog of Orbits of Visual Binary Stars (orb6orbits.txt), orbits of grade 4 or better whose primary Gaia DR3 puts within 100 pc, each solved for 2016.0; a system is kept when Gaia holds the secondary within 0.3 arcsec or 15 per cent of the separation of where the orbit puts it, which checks the orbit, and the separation lies between 1 and 20 arcsec, so the points are resolved from the stars and the tolerance stays within 3 arcsec. L4 and L5 lie at the secondary's distance from the primary, 60 deg ahead of and behind it in the orbital plane, and are unstable for any stellar mass ratio, so nothing natural stays there; Gaia DR3 sources at the two points, other than the components, against the same count at ten other angles of the orbit.

    python -I seti.py neighbourhood trojanpoints [record] [--epoch 2016] [--grade-max 4] [--plx-min 10] [--sep-arcsec 1 20] [--point-deg 60]"""

ORB6_URL = "https://www.astro.gsu.edu/wds/orb6/orb6orbits.txt"
ORB6_LABELS = ["RA,Dec (J2000)....", "WDS.......", "DD............", "ADS..", "HD....", "HIP...", "V1.11*", "V2.22*", "PPPP.PPPPPP*", "eee.eeeeee", "AAA.AAAAA*",
               "ee.eeeee", "III.IIII", "eee.eeee", "NNN.NNNN*", "eee.eeee", "TTTTT.TTTTTT*", "eee.eeeeee", "E.EEEEEE", "e.eeeeee", "OOO.OOOO", "eee.eeee", "EQNX", "LAST",
               "G", "N", "REF.....", "PNGFILE"]


def orb6_orbits():
    """Every ORB6 orbit: position (deg), WDS name, HIP, grade and the elements in years, arcsec and degrees (T as a Julian year)."""
    path = core.results("orb6orbits.txt")
    if not os.path.exists(path):
        open(path, "wb").write(core.get(ORB6_URL, timeout=300))
    lines = open(path, encoding="latin-1").read().split("\n")
    head = lines[5]
    pos, k = [], 0
    for lab in ORB6_LABELS:
        k = head.index(lab, k)
        pos.append(k)
        k += len(lab)
    pos.append(len(head) + 60)

    def val(s):
        m = re.match(r"\s*(\d*\.?\d+)\s*([a-zA-Z]?)", s)
        return (float(m.group(1)), m.group(2)) if m else (None, "")
    out = []
    for line in lines[7:]:
        if len(line) < 200:
            continue
        f = [line[pos[i]:pos[i + 1]] for i in range(len(ORB6_LABELS))]
        try:
            ra = 15 * (int(f[0][0:2]) + int(f[0][2:4]) / 60 + float(f[0][4:9]) / 3600)
            de = (int(f[0][10:12]) + int(f[0][12:14]) / 60 + float(f[0][14:18]) / 3600) * (-1 if f[0][9] == "-" else 1)
        except ValueError:
            continue
        P, Pu = val(f[8])
        a, au = val(f[10])
        i, _ = val(f[12])
        node, _ = val(f[14])
        T, Tu = val(f[16])
        e, _ = val(f[18])
        om, _ = val(f[20])
        grade = f[24].strip()
        if None in (P, a, i, node, T, e, om) or not grade.isdigit():
            continue
        P = P * {"y": 1.0, "d": 1 / 365.25, "c": 100.0}.get(Pu, 1.0)
        a = a * {"a": 1.0, "m": 1e-3}.get(au, 1.0)
        T = T if Tu == "y" else (T - 51544.5) / 365.25 + 2000.0 if Tu == "d" else T
        out.append(dict(wds=f[1].strip(), name=f[2].strip(), hip=f[5].strip(), ra=ra, dec=de, P=P, a=a, i=i, node=node, T=T, e=e, om=om, grade=int(grade)))
    return out


def orb6_place(o, epoch, extra_deg=0.0):
    """Sky offset (east, north; arcsec) of the point on the orbit at the true anomaly of EPOCH plus EXTRA_DEG, at the orbit's radius there."""
    M = 2 * math.pi * ((epoch - o["T"]) / o["P"] % 1.0)
    E = M
    for _ in range(60):
        E = E - (E - o["e"] * math.sin(E) - M) / (1 - o["e"] * math.cos(E))
    nu = 2 * math.atan2(math.sqrt(1 + o["e"]) * math.sin(E / 2), math.sqrt(1 - o["e"]) * math.cos(E / 2))
    r = o["a"] * (1 - o["e"] * math.cos(E))
    u = nu + math.radians(extra_deg) + math.radians(o["om"])
    inc, node = math.radians(o["i"]), math.radians(o["node"])
    rho = r * math.sqrt(math.cos(u) ** 2 + math.sin(u) ** 2 * math.cos(inc) ** 2)
    theta = node + math.atan2(math.sin(u) * math.cos(inc), math.cos(u))
    return rho * math.sin(theta), rho * math.cos(theta)


def trojanpoints_main(record=False, epoch=2016.0, grade_max=4, plx_min=10.0, sep_arcsec=[1.0, 20.0], point_deg=60.0):
    orbits = [o for o in orb6_orbits() if o["grade"] <= grade_max]
    up = [dict(source_id="o%d" % k, ra=o["ra"], dec=o["dec"]) for k, o in enumerate(orbits)]
    near = {}
    for k in range(0, len(up), 5000):
        for m in core.xmatch(up[k:k + 5000], "I/355/gaiadr3", 5.0):
            j = int(m["source_id"][1:])
            if num(m.get("Plx")) is not None and (j not in near or (num(m.get("Gmag")) or 99) < (num(near[j].get("Gmag")) or 99)):
                near[j] = m
    kept, hits, control = [], [], []
    for j, m in near.items():
        o = orbits[j]
        if num(m["Plx"]) < plx_min:
            continue
        ex, ny = orb6_place(o, epoch)
        rho = math.hypot(ex, ny)
        if not sep_arcsec[0] <= rho <= sep_arcsec[1]:
            continue
        ra0, de0 = float(m["RAJ2000"]), float(m["DEJ2000"])
        rad = (3 * rho + 2) / 3600
        src = core.tap("gaia", "SELECT source_id, ra, dec, phot_g_mean_mag, parallax FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.7f, %.7f, %.7f))"
                       % (ra0, de0, rad))
        pts = [(float(s["ra"]), float(s["dec"]), s["source_id"], num(s["phot_g_mean_mag"])) for s in src if s["source_id"] != m["Source"]]
        tol = max(0.3, 0.15 * rho)

        def at(dx, dy):
            return [sid for ra, de, sid, g in pts if math.hypot((ra - ra0) * math.cos(math.radians(de0)) * 3600 - dx, (de - de0) * 3600 - dy) < tol]
        secondary = at(ex, ny)
        if len(secondary) != 1:
            continue
        kept.append(dict(wds=o["wds"], name=o["name"], plx=num(m["Plx"]), rho=round(rho, 2), P=o["P"], grade=o["grade"], primary=m["Source"], secondary=secondary[0]))
        for lab, ang in (("L4", point_deg), ("L5", -point_deg)):
            found = [s for s in at(*orb6_place(o, epoch, ang)) if s != secondary[0]]
            if found:
                hits.append((o["wds"], o["name"], lab, found))
        for ang in (90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0, 30.0, -30.0, 210.0):
            control.append(len([s for s in at(*orb6_place(o, epoch, ang)) if s != secondary[0]]))
    exp = 2 * len(kept) * float(np.mean(control)) if control else 0.0
    print("%d ORB6 orbits of grade 4 or better; %d within 100 pc with the secondary where the orbit puts it and more than 1 arcsec away; Gaia sources at L4 or L5: %d %s "
          "against %.2f from ten other angles of the same orbits" % (len(orbits), len(kept), len(hits), hits, exp))
    if record:
        from scipy.stats import poisson
        core.record_run("binary_triangular_points", "The triangular points of nearby binaries", "Sixth Catalog of Orbits of Visual Binary Stars; Gaia DR3 (I/355 by CDS XMatch, "
                        "gaia_source cones)", "orbits within 100 pc checked against Gaia's secondary: %d" % len(kept), os.path.abspath(__file__), "trojanpoints", len(kept),
                        "binaries", dict(epoch=epoch, grade_max=grade_max, plx_min=plx_min, sep_arcsec=sep_arcsec, point_deg=point_deg, tolerance="max(0.3 arcsec, 0.15 rho)"),
                        "the same count at ten other angles of each orbit: %.2f" % exp, "a Gaia source at L4 or L5 of a stellar pair, where nothing stays unheld",
                        "%d at the points against %.2f (p %.3f)" % (len(hits), exp, float(poisson.sf(len(hits) - 1, exp)) if hits else 1.0),
                        [dict(target="wds:%s" % k["wds"], stat=k["rho"], unit="separation, arcsec", vet="control" if not any(h[0] == k["wds"] for h in hits) else None,
                              note="%s; parallax %.1f mas, P %.1f yr, grade %d; Gaia DR3 %s + %s%s" % (k["name"], k["plx"], k["P"], k["grade"], k["primary"], k["secondary"],
                                                                                                   "".join("; %s %s" % (h[2], h[3]) for h in hits if h[0] == k["wds"]))) for k in kept])
        print("recorded")


watchers_DOC = r"""Watchers of Jupiter's transits: Gaia Catalogue of Nearby Stars objects within the band of sky from which Jupiter is seen to cross the Sun, half a Sun's angular radius at Jupiter's distance (0.051 deg) either side of Jupiter's orbital plane, carried to 2007.0 and matched to GALEX GUVcat AIS within 5 arcsec by CDS XMatch, against the share of stars 1-5 deg from the plane and against the band moved 2 arcmin in eight directions; Jupiter is the planet an outside observer finds first, so a watcher sits where its transits show. Any planet's zone may be asked for, each from its orbital plane and distance.

    python -I seti.py neighbourhood watchers [record] [--planet jupiter] [--catalogue II/335/galex_ais] [--radius-arcsec 5] [--epoch 2007]"""

# semi-major axis (au), inclination and ascending node (deg, J2000 ecliptic) of each planet's orbit
PLANET_ORBITS = {"mercury": (0.38710, 7.005, 48.331), "venus": (0.72333, 3.395, 76.680), "earth": (1.0, 0.0, 0.0), "mars": (1.52368, 1.850, 49.558),
                 "jupiter": (5.2026, 1.303, 100.464), "saturn": (9.5549, 2.489, 113.665), "uranus": (19.2184, 0.773, 74.006), "neptune": (30.110, 1.770, 131.784)}


def planet_zone_stars(planet, control=(1.0, 5.0)):
    """Gaia Catalogue of Nearby Stars objects within PLANET's transit zone (a Sun's angular radius at its distance, either side of its orbital plane) and the band CONTROL deg from the plane, with positions, motions and parallaxes at 2016.0; kept in the results directory."""
    a, inc, node = PLANET_ORBITS[planet]
    half = math.degrees(math.asin(0.0046505 / a))
    path = core.results("gcns_%s_zone.json" % planet)
    if os.path.exists(path):
        return json.load(open(path))
    eps = math.radians(23.4392911)
    i, o = math.radians(inc), math.radians(node)
    n = (math.sin(i) * math.sin(o), -math.sin(i) * math.cos(o), math.cos(i))
    out = dict(zone=[], near=[], half_deg=half)
    for r in catalogues.gcns("source_id, ra, dec, pmra, pmdec, parallax", ranges=12):
        ra, de = math.radians(float(r["ra"])), math.radians(float(r["dec"]))
        x, y, z = math.cos(de) * math.cos(ra), math.cos(de) * math.sin(ra), math.sin(de)
        ye, ze = y * math.cos(eps) + z * math.sin(eps), -y * math.sin(eps) + z * math.cos(eps)
        beta = abs(math.degrees(math.asin(max(-1.0, min(1.0, x * n[0] + ye * n[1] + ze * n[2])))))
        if beta < half or control[0] <= beta < control[1]:
            out["zone" if beta < half else "near"].append(dict(id=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"]), pmra=num(r["pmra"]) or 0.0,
                                                                pmdec=num(r["pmdec"]) or 0.0, plx=float(r["parallax"]), beta=round(beta, 4)))
    json.dump(out, open(path, "w"))
    return out


def watchers_main(record=False, planet="jupiter", catalogue="II/335/galex_ais", radius_arcsec=5.0, epoch=2007.0):
    from scipy.stats import poisson
    from setitools.catalogues import shifted_matches
    planet = planet.lower()
    st = planet_zone_stars(planet)

    def place(s):
        ra, de = core.star_at(s["ra"], s["dec"], s["pmra"], s["pmdec"], 2016.0, epoch)
        return dict(name="g%s" % s["id"], ra=ra % 360, dec=de)
    zone = [place(s) for s in st["zone"]]
    near = [place(s) for s in st["near"]]
    hits, null = shifted_matches(zone, catalogue, radius_arcsec)
    got = {}
    for m in hits:
        if m["source_id"] not in got or float(m["angDist"]) < float(got[m["source_id"]]["angDist"]):
            got[m["source_id"]] = m
    far = set()
    for k in range(0, len(near), 5000):
        far |= {m["source_id"] for m in core.xmatch([dict(source_id=p["name"], ra=p["ra"], dec=p["dec"]) for p in near[k:k + 5000]], catalogue, radius_arcsec)}
    exp = len(far) / max(len(near), 1) * len(zone)
    p = float(poisson.sf(len(got) - 1, exp)) if got else 1.0
    vets = etz_vets(got)
    mean = float(np.mean(null)) if len(null) else 0.0
    print("%s's transit zone is %.3f deg either side of its orbital plane; %d zone stars and %d stars 1-5 deg away; %s within %g arcsec: zone %d, away %d (%.2f expected in "
          "the zone, p %.3g); zone moved 2 arcmin %s (mean %.1f); zone matches by SIMBAD type %s" % (
              planet.capitalize(), st["half_deg"], len(zone), len(near), catalogue, radius_arcsec, len(got), len(far), exp, p, null, mean,
              sorted({(v[0] or (None, "none"))[1] for v in vets.values()})))
    if record:
        slug = catalogue.split("/")[-1].replace("_ais", "")
        core.record_run("%s_transit_zone_%s" % (planet, slug), "Watchers of %s's transits" % planet.capitalize(),
                        "Gaia Catalogue of Nearby Stars (external.gaiaedr3_gcns_main_1); %s by CDS XMatch; SIMBAD" % catalogue,
                        "stars within 100 pc and %.3f deg of %s's orbital plane: %d" % (st["half_deg"], planet.capitalize(), len(zone)), os.path.abspath(__file__), "watchers",
                        len(zone), "stars", dict(planet=planet, zone_deg=round(st["half_deg"], 4), control_deg=[1, 5], radius_arcsec=radius_arcsec, epoch=epoch, catalogue=catalogue),
                        "the share 1-5 deg from the plane (%d of %d); the zone moved 2 arcmin: %s" % (len(far), len(near), null),
                        "a source of %s at a star that sees %s transit, beyond the rate of stars beside the zone" % (catalogue, planet.capitalize()),
                        "%d against %.2f from the share beside the zone (p %.3g) and a mean of %.1f moved" % (len(got), exp, p, mean),
                        [dict(target="gaia_dr3:%s" % n[1:], vet=vets[n][1] or ("chance-rate" if p > 0.01 else None), note="SIMBAD %s; %s at %.1f arcsec" % (
                            vets[n][0], ", ".join("%s %s" % (k, m.get(k)) for k in ("NUVmag", "FUVmag") if m.get(k) is not None) or "match", float(m["angDist"])))
                         for n, m in sorted(got.items())])
        print("recorded")


roundbinaries_DOC = r"""Binaries on perfect circles: spectroscopic binaries of the Ninth Catalogue of Spectroscopic Binary Orbits (VizieR B/sb9) with periods beyond 100 days, where tides cannot have rounded the orbit, whose eccentricity is measured below 0.002 at its upper error, an orbit rounder than formation leaves, against the eccentricities of the rest; an eccentricity fixed at zero by the orbit's author carries a flag or no error and is set aside, and an evolved component rounds its orbit while it is a giant, so each match is read for its spectral types and its orbit's grade.

    python -I seti.py neighbourhood roundbinaries [record] [--p-min-d 100] [--tol 0.002] [--grade-min 3]"""


def roundbinaries_main(record=False, p_min_d=100.0, tol=0.002, grade_min=3.0):
    orbits = core.tap("vizier", 'SELECT Seq, Per, e, e_e, f_e, K1, K2, Grade, Ref FROM "B/sb9/orbits" WHERE Per > %g AND e IS NOT NULL' % p_min_d)
    main = {str(r["Seq"]): r for r in core.tap("vizier", 'SELECT Seq, Name, Sp1, Sp2, mag1, RAJ2000, DEJ2000 FROM "B/sb9/main"')}
    meas, out = [], []
    for r in orbits:
        e, ee, flag = num(r["e"]), num(r["e_e"]), str(r["f_e"] or "").strip()
        if e is None or not ee or flag:
            continue
        meas.append(e + ee)
        if e + ee < tol:
            m = main.get(str(r["Seq"]), {})
            sp = " ".join(x for x in (str(m.get("Sp1") or "").strip(), str(m.get("Sp2") or "").strip()) if x)
            evolved = any(x in sp for x in ("III", "II", "Ib", "Ia", "I ", "K0I", "G8I")) or (sp and sp[-1] == "I")
            grade = num(r["Grade"])
            out.append(dict(seq=str(r["Seq"]), name=str(m.get("Name") or "").strip(), P=num(r["Per"]), e=e, ee=ee, k1=num(r["K1"]), k2=num(r["K2"]), grade=grade, sp=sp,
                            ref=str(r["Ref"] or "").strip(),
                            vet="evolved" if evolved else "unconstrained-geometry" if grade is not None and grade < grade_min else None))
    es = sorted(meas)
    k = max(1, int(0.01 * len(es)))
    left = [o for o in out if o["vet"] is None]
    print("%d orbits beyond %g d with a measured eccentricity (flagged or errorless ones set aside); upper eccentricity below %g: %d, %d with an evolved component, %d of low "
          "grade; unexplained %d; the lowest percentile of upper eccentricities sits at %.4f" % (len(meas), p_min_d, tol, len(out), sum(o["vet"] == "evolved" for o in out),
                                                                                                sum(o["vet"] == "unconstrained-geometry" for o in out), len(left), es[k - 1] if es else 0))
    for o in sorted(out, key=lambda o: o["e"] + o["ee"]):
        print("  %s (SB9 %s): P %.2f d, e %.4f +- %.4f, K1 %s K2 %s km/s, grade %s, %s, %s; %s" % (o["name"], o["seq"], o["P"], o["e"], o["ee"], o["k1"], o["k2"], o["grade"],
                                                                                                 o["sp"] or "-", o["ref"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(meas), hits=out), open(core.results("binaries_perfect_circles.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sb9_perfect_circles", "Binaries on perfect circles", "Ninth Catalogue of Spectroscopic Binary Orbits (VizieR B/sb9)",
            "orbits beyond %g d with a measured eccentricity: %d" % (p_min_d, len(meas)), os.path.abspath(__file__), "roundbinaries", len(meas), "orbits",
            dict(p_min_d=p_min_d, tol=tol, grade_min=grade_min), "the measured upper eccentricities themselves: the lowest percentile at %.4f" % (es[k - 1] if es else 0),
            "an orbit beyond the tidal reach measured rounder than %g at its upper error" % tol,
            "%d below %g: %d with an evolved component, %d of low grade; %d unexplained" % (len(out), tol, sum(o["vet"] == "evolved" for o in out),
                                                                                           sum(o["vet"] == "unconstrained-geometry" for o in out), len(left)),
            [dict(target="sb9:%s" % o["seq"], stat=o["e"] + o["ee"], unit="upper eccentricity", vet=o["vet"],
                  note="%s, P %.2f d, e %.4f +- %.4f, grade %s, %s, %s" % (o["name"], o["P"], o["e"], o["ee"], o["grade"], o["sp"] or "-", o["ref"])) for o in out])
        print("recorded")


escapees_DOC = r"""Stars leaving the nearest clusters too fast: members of the open clusters within 200 pc in Hunt and Reffert (2023) (VizieR J/A+A/673/A114) with a Gaia radial velocity, each star's space velocity against its cluster's mean, those moving faster than 10 km/s relative to their cluster, well beyond the escape speed of a few hundred solar masses, as a star pushed out would; a close binary's orbital motion shows the same in one epoch's velocity, so each is read for its RUWE, its velocity error and its SIMBAD type; only clusters with twenty or more velocities whose members cohere to a median of 3 km/s about the mean are taken, since a loose group's mean is no reference; a membership probability below 0.8 leaves a star as likely a field star, and Gaia's velocities of hot stars carry errors of tens of km/s.

    python -I seti.py neighbourhood escapees [record] [--plx-min 5] [--v-min-kms 10] [--prob-min 0.5] [--rv-err-max 2] [--min-members 20] [--coherent-kms 3] [--prob-firm 0.8]"""


def escapees_main(record=False, plx_min=5.0, v_min_kms=10.0, prob_min=0.5, rv_err_max=2.0, min_members=20, coherent_kms=3.0, prob_firm=0.8):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    clusters = {str(r["Name"]).strip(): r for r in core.tap("vizier", 'SELECT Name, RA_ICRS, DE_ICRS, Plx, pmRA, pmDE, RV, s_RV, n_RV, rt FROM "J/A+A/673/A114/clusters" '
                                                                       'WHERE Plx > %g AND RV IS NOT NULL' % plx_min)}
    members = core.tap("vizier", 'SELECT Name, GaiaDR3, Prob, inrt, RA_ICRS, DE_ICRS, pmRA, pmDE, Plx, RV, e_RV, RUWE, Gmag, "BP-RP" FROM "J/A+A/673/A114/members" '
                                 'WHERE Plx > %g AND RV IS NOT NULL AND Prob > %g' % (plx_min, prob_min), timeout=900)
    members = [m for m in members if str(m["Name"]).strip() in clusters and (num(m["e_RV"]) or 99) < rv_err_max]

    def uvw(ra, dec, plx, pmra, pmdec, rv):
        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=(1000.0 / plx) * u.pc, pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                     radial_velocity=rv * u.km / u.s).galactic
        c.representation_type = "cartesian"
        c.differential_type = "cartesian"
        return np.array([c.velocity.d_x.to(u.km / u.s).value, c.velocity.d_y.to(u.km / u.s).value, c.velocity.d_z.to(u.km / u.s).value])
    cl_v = {n: uvw(float(r["RA_ICRS"]), float(r["DE_ICRS"]), float(r["Plx"]), float(r["pmRA"]), float(r["pmDE"]), float(r["RV"])) for n, r in clusters.items()}
    per, dvs = {}, {}
    for m in members:
        n = str(m["Name"]).strip()
        v = uvw(float(m["RA_ICRS"]), float(m["DE_ICRS"]), float(m["Plx"]), float(m["pmRA"]), float(m["pmDE"]), float(m["RV"]))
        dvs[str(m["GaiaDR3"])] = float(np.linalg.norm(v - cl_v[n]))
        per.setdefault(n, []).append(dvs[str(m["GaiaDR3"])])
    # a cluster counts when its members' speeds about its mean cohere: at least MIN_MEMBERS with velocities and a median below COHERENT_KMS
    coherent = {n for n, v in per.items() if len(v) >= min_members and float(np.median(v)) < coherent_kms}
    members = [m for m in members if str(m["Name"]).strip() in coherent]
    out = []
    for m in members:
        n = str(m["Name"]).strip()
        dv = dvs[str(m["GaiaDR3"])]
        if dv > v_min_kms:
            out.append(dict(cluster=n, id=str(m["GaiaDR3"]), dv=dv, rv=num(m["RV"]), e_rv=num(m["e_RV"]), ruwe=num(m["RUWE"]), prob=num(m["Prob"]), inrt=str(m["inrt"]).strip(),
                            g=num(m["Gmag"]), c=num(m["BP-RP"])))
    sim = {}
    for k in range(0, len(out), 200):
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % o["id"] for o in out[k:k + 200])):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
    for o in out:
        o["simbad"] = sim.get(o["id"])
        typ = (o["simbad"] or (None, "", ""))[1]
        # a membership probability below 0.8 leaves the star as likely a field star; Gaia's velocities of hot stars (BP-RP below 0.3) carry errors of tens of km/s
        o["vet"] = ("binary" if (o["ruwe"] or 0) > 1.4 or typ in ("**", "SB*", "EB*") else "outside-selection" if (o["prob"] or 0) < prob_firm else
                    "artefact" if o["c"] is not None and o["c"] < 0.3 else None)
    left = [o for o in out if o["vet"] is None]
    disp = {n: round(float(np.median(per[n])), 2) for n in sorted(coherent)}
    print("%d members of the %d coherent clusters within %.0f pc (of %d with mean velocities) with radial velocities to %g km/s; median speeds relative to each cluster %s; "
          "faster than %g km/s: %d, %d binaries by RUWE or type, %d of membership probability below %g, %d hot stars; unexplained %d" % (
              len(members), len(coherent), 1000 / plx_min, len(clusters), rv_err_max, disp, v_min_kms, len(out), sum(o["vet"] == "binary" for o in out),
              sum(o["vet"] == "outside-selection" for o in out), prob_firm, sum(o["vet"] == "artefact" for o in out), len(left)))
    for o in sorted(out, key=lambda o: -o["dv"]):
        print("  %s: Gaia DR3 %s at %.1f km/s from the cluster, RV %s +- %s, RUWE %s, probability %s, inside tidal radius %s, G %s, BP-RP %s; SIMBAD %s; %s" % (
            o["cluster"], o["id"], o["dv"], o["rv"], o["e_rv"], o["ruwe"], o["prob"], o["inrt"], o["g"], o["c"], o["simbad"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(members), clusters=disp, hits=out), open(core.results("cluster_escapees.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_cluster_escapees", "Stars leaving the nearest clusters too fast", "Hunt and Reffert (2023) open-cluster members with Gaia radial velocities (VizieR J/A+A/673/A114); "
            "SIMBAD", "members of the %d coherent clusters within %.0f pc with radial velocities to %g km/s: %d" % (len(coherent), 1000 / plx_min, rv_err_max, len(members)),
            os.path.abspath(__file__), "escapees", len(members), "stars", dict(plx_min=plx_min, v_min_kms=v_min_kms, prob_min=prob_min, rv_err_max=rv_err_max,
                                                                                 min_members=min_members, coherent_kms=coherent_kms, prob_firm=prob_firm),
            "the clusters' own median speeds relative to their means: %s" % disp, "a member moving faster than %g km/s relative to its cluster" % v_min_kms,
            "%d: %d binaries, %d of membership probability below %g, %d hot stars; %d unexplained" % (
                len(out), sum(o["vet"] == "binary" for o in out), sum(o["vet"] == "outside-selection" for o in out), prob_firm, sum(o["vet"] == "artefact" for o in out), len(left)),
            [dict(target="gaia_dr3:%s" % o["id"], stat=o["dv"], unit="km/s relative to the cluster", vet=o["vet"],
                  note="%s, RV %s +- %s, RUWE %s, probability %s, inside tidal radius %s; SIMBAD %s" % (o["cluster"], o["rv"], o["e_rv"], o["ruwe"], o["prob"], o["inrt"], o["simbad"]))
             for o in out])
        print("recorded")


darkcircles_DOC = r"""Dark companions on perfect circles: Gaia DR3 astrometric two-body orbits (nss_two_body_orbit, the Orbital solutions) within 200 pc with periods beyond 200 days, where tides cannot round an orbit, whose eccentricity is measured below 0.005 at its upper error, an unseen companion on an orbit rounder than capture or formation leaves; against the solutions' own eccentricities, each read for its goodness of fit, its significance and its SIMBAD type.

    python -I seti.py neighbourhood darkcircles [record] [--plx-min 5] [--p-min-d 200] [--tol 0.005] [--sig-min 20]"""


def darkcircles_main(record=False, plx_min=5.0, p_min_d=200.0, tol=0.005, sig_min=20.0):
    rows = core.tap_async("gaia", "SELECT n.source_id, n.period, n.eccentricity, n.eccentricity_error, n.significance, n.goodness_of_fit, n.nss_solution_type, g.parallax, "
                                  "g.phot_g_mean_mag, g.bp_rp FROM gaiadr3.nss_two_body_orbit AS n JOIN gaiadr3.gaia_source AS g ON n.source_id = g.source_id WHERE "
                                  "n.nss_solution_type IN ('Orbital', 'OrbitalTargetedSearch', 'OrbitalTargetedSearchValidated') AND g.parallax > %g AND n.period > %g "
                                  "AND n.eccentricity IS NOT NULL AND n.eccentricity_error IS NOT NULL" % (plx_min, p_min_d), maxrec=100000)
    st = [dict(id=str(r["source_id"]), P=num(r["period"]), e=num(r["eccentricity"]), ee=num(r["eccentricity_error"]), sig=num(r["significance"]), gof=num(r["goodness_of_fit"]),
               kind=str(r["nss_solution_type"]), plx=num(r["parallax"]), g=num(r["phot_g_mean_mag"]), c=num(r["bp_rp"])) for r in rows]
    st = [s for s in st if s["e"] is not None and s["ee"]]
    hits = [s for s in st if s["e"] + s["ee"] < tol]
    sim = {}
    for k in range(0, len(hits), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in hits[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in hits:
        s["simbad"] = sim.get(s["id"])
        s["vet"] = "unconstrained-geometry" if (s["sig"] or 0) < sig_min else None
    ups = sorted(s["e"] + s["ee"] for s in st)
    k = max(1, int(0.01 * len(ups)))
    left = [s for s in hits if s["vet"] is None]
    print("%d astrometric orbits within %.0f pc with periods beyond %g d and a measured eccentricity; upper eccentricity below %g: %d, %d with significance below %g; unexplained %d; "
          "the lowest percentile of upper eccentricities sits at %.4f" % (len(st), 1000 / plx_min, p_min_d, tol, len(hits), sum(s["vet"] == "unconstrained-geometry" for s in hits),
                                                                        sig_min, len(left), ups[k - 1] if ups else 0))
    for s in sorted(hits, key=lambda s: s["e"] + s["ee"]):
        print("  Gaia DR3 %s: P %.1f d, e %.4f +- %.4f, significance %.1f, fit %.1f, %s, G %.2f, BP-RP %s, %.0f pc; SIMBAD %s; %s" % (
            s["id"], s["P"], s["e"], s["ee"], s["sig"] or 0, s["gof"] or 0, s["kind"], s["g"], s["c"], 1000 / s["plx"], s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), hits=hits), open(core.results("dark_companions_circles.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_nss_perfect_circles", "Dark companions on perfect circles", "Gaia DR3 nss_two_body_orbit and gaia_source; SIMBAD",
            "astrometric orbits within %.0f pc with periods beyond %g d and a measured eccentricity: %d" % (1000 / plx_min, p_min_d, len(st)), os.path.abspath(__file__),
            "darkcircles", len(st), "orbits", dict(plx_min=plx_min, p_min_d=p_min_d, tol=tol, sig_min=sig_min),
            "the solutions' own upper eccentricities: the lowest percentile at %.4f" % (ups[k - 1] if ups else 0),
            "an astrometric orbit beyond the tidal reach measured rounder than %g at its upper error" % tol,
            "%d below %g, %d of low significance; %d unexplained" % (len(hits), tol, sum(s["vet"] == "unconstrained-geometry" for s in hits), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=s["e"] + s["ee"], unit="upper eccentricity", vet=s["vet"],
                  note="P %.1f d, e %.4f +- %.4f, significance %.1f, %s; SIMBAD %s" % (s["P"], s["e"], s["ee"], s["sig"] or 0, s["kind"], s["simbad"])) for s in hits])
        print("recorded")


retroclusters_DOC = r"""Open clusters orbiting backwards: the Hunt and Reffert (2023) clusters (VizieR J/A+A/673/A114) with a parallax at 10 sigma, a radial velocity from three or more members to 10 km/s, Galactic angular momenta from their mean motions and distances; a cluster whose angular momentum about the Galactic axis opposes the Sun's is on a retrograde orbit, which no cluster born of the disc's gas can hold and a fleet could; globular clusters and moving groups set aside by the catalogue's type or the literature's, each read for its velocity's provenance.

    python -I seti.py neighbourhood retroclusters [record] [--plx-sigma 10] [--rv-members 3] [--rv-err 10] [--vz-max 150]"""


# clusters the catalogue types as open that the literature holds to be globular
RETROCLUSTERS_GLOBULAR = {"Pismis_26": "Tonantzintla 2"}


def retroclusters_main(record=False, plx_sigma=10.0, rv_members=3, rv_err=10.0, vz_max=150.0):
    from astropy.coordinates import SkyCoord, Galactocentric
    import astropy.units as u
    rows = core.tap("vizier", 'SELECT Name, Type, N, RA_ICRS, DE_ICRS, Plx, e_Plx, dist50, pmRA, e_pmRA, pmDE, e_pmDE, RV, e_RV, n_RV, logAge50 FROM "J/A+A/673/A114/clusters" '
                              'WHERE RV IS NOT NULL AND n_RV >= %d AND e_RV < %g AND Plx > %g * e_Plx' % (rv_members, rv_err, plx_sigma))
    cl = [dict(name=str(r["Name"]).strip(), kind=str(r["Type"]).strip(), n=int(num(r["N"]) or 0), ra=float(r["RA_ICRS"]), dec=float(r["DE_ICRS"]), plx=num(r["Plx"]), d=num(r["dist50"]),
               pmra=num(r["pmRA"]), pmdec=num(r["pmDE"]), rv=num(r["RV"]), erv=num(r["e_RV"]), nrv=int(num(r["n_RV"]) or 0), logage=num(r["logAge50"])) for r in rows]
    cl = [c for c in cl if None not in (c["d"], c["pmra"], c["pmdec"], c["rv"])]
    c = SkyCoord(ra=[x["ra"] for x in cl] * u.deg, dec=[x["dec"] for x in cl] * u.deg, distance=[x["d"] for x in cl] * u.pc,
                 pm_ra_cosdec=[x["pmra"] for x in cl] * u.mas / u.yr, pm_dec=[x["pmdec"] for x in cl] * u.mas / u.yr, radial_velocity=[x["rv"] for x in cl] * u.km / u.s)
    g = c.transform_to(Galactocentric())
    sun = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1e-6 * u.pc, pm_ra_cosdec=0 * u.mas / u.yr, pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s).transform_to(Galactocentric())
    lz_sun = float((sun.x * sun.v_y - sun.y * sun.v_x).to(u.kpc * u.km / u.s).value)
    x, y = g.x.to(u.kpc).value, g.y.to(u.kpc).value
    vx, vy, vz = g.v_x.to(u.km / u.s).value, g.v_y.to(u.km / u.s).value, g.v_z.to(u.km / u.s).value
    r = np.hypot(x, y)
    for k, o in enumerate(cl):
        o["lz"] = float(x[k] * vy[k] - y[k] * vx[k])
        o["vphi"] = o["lz"] / float(r[k]) * (1 if lz_sun > 0 else -1)
        o["vz"] = float(vz[k])
        o["vet"] = "natural-type" if o["kind"] in ("g", "m") or o["name"] in RETROCLUSTERS_GLOBULAR else None
    retro = [o for o in cl if o["vphi"] < 0]
    fast = [o for o in cl if abs(o["vz"]) > vz_max and o["vphi"] >= 0]
    left = [o for o in retro if o["vet"] is None]
    v = np.array([o["vphi"] for o in cl])
    print("%d clusters with a parallax at %g sigma and a radial velocity from %d or more members; rotation speeds %.0f median, %.0f lowest; retrograde: %d (%d globular or moving "
          "groups); vertical speed above %g km/s: %d; unexplained %d" % (len(cl), plx_sigma, rv_members, np.median(v), v.min(), len(retro), sum(o["vet"] is not None for o in retro),
                                                                        vz_max, len(fast), len(left)))
    for o in sorted(retro + fast, key=lambda o: o["vphi"]):
        print("  %s (%s, %d members, %d with velocities): V_phi %+.0f, V_z %+.0f km/s, RV %.1f +- %.1f, %.0f pc, log age %s; %s" % (
            o["name"], o["kind"], o["n"], o["nrv"], o["vphi"], o["vz"], o["rv"], o["erv"], o["d"], o["logage"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(cl), retro=retro, fast=fast), open(core.results("clusters_retrograde.json"), "w"), indent=1)
    if record:
        core.record_run(
            "clusters_retrograde", "Open clusters orbiting backwards", "Hunt and Reffert (2023) cluster catalogue (VizieR J/A+A/673/A114)",
            "clusters with a parallax at %g sigma and a radial velocity from %d or more members to %g km/s: %d" % (plx_sigma, rv_members, rv_err, len(cl)), os.path.abspath(__file__),
            "retroclusters", len(cl), "clusters", dict(plx_sigma=plx_sigma, rv_members=rv_members, rv_err=rv_err, vz_max=vz_max),
            "the clusters' rotation speeds: median %.0f km/s, lowest %.0f" % (np.median(v), v.min()), "a cluster whose Galactic angular momentum opposes the Sun's",
            "%d retrograde, %d of them globular or moving groups; %d with a vertical speed above %g km/s; %d unexplained" % (
                len(retro), sum(o["vet"] is not None for o in retro), len(fast), vz_max, len(left)),
            [dict(target="cluster:%s" % o["name"], stat=round(o["vphi"], 1), unit="rotation speed, km/s", vet=o["vet"],
                  note="%s, %d members, %d with velocities, V_z %+.0f km/s, RV %.1f +- %.1f, %.0f pc%s" % (
                      o["kind"], o["n"], o["nrv"], o["vz"], o["rv"], o["erv"], o["d"], "; " + RETROCLUSTERS_GLOBULAR[o["name"]] if o["name"] in RETROCLUSTERS_GLOBULAR else ""))
             for o in retro + fast])
        print("recorded")


jumped_DOC = r"""Stars that jumped between 2MASS and Gaia: Gaia DR3 stars within 25 pc with a 2MASS counterpart (tmass_psc_xsc_best_neighbour), the Gaia position carried back to the night 2MASS scanned it with the proper motion and the parallactic shift of that night, against the 2MASS position; a star more than 1 arcsec from where its course puts it changed course between 1998 and 2016, as nothing on a free orbit does and a driven thing may; each read for its 2MASS quality, its neighbours in the 2MASS beam, its Gaia image flags and its brightness, since a saturated or blended 2MASS image sets a false position.

    python -I seti.py neighbourhood jumped [record] [--plx-min 40] [--offset-arcsec 1] [--ks-sat 4]"""


def jumped_main(record=False, plx_min=40.0, offset_arcsec=1.0, ks_sat=4.0):
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    rows = core.tap_async("gaia", "SELECT g.source_id, g.ra, g.dec, g.parallax, g.pmra, g.pmdec, g.phot_g_mean_mag, g.ruwe, g.ipd_frac_multi_peak, g.duplicated_source, t.ra AS tra, "
                                  "t.dec AS tdec, t.j_date, t.ph_qual, t.ks_m, t.err_maj, n.angular_distance, n.number_of_neighbours, n.number_of_mates FROM gaiadr3.gaia_source AS g "
                                  "JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS n ON g.source_id = n.source_id JOIN gaiadr1.tmass_original_valid AS t ON "
                                  "n.original_ext_source_id = t.designation WHERE g.parallax > %g AND g.pmra IS NOT NULL" % plx_min, maxrec=100000)
    st = [dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), plx=float(r["parallax"]), pmra=float(r["pmra"]), pmdec=float(r["pmdec"]), g=num(r["phot_g_mean_mag"]),
               ruwe=num(r["ruwe"]), multi=num(r["ipd_frac_multi_peak"]), dup=num(r["duplicated_source"]), tra=float(r["tra"]), tdec=float(r["tdec"]), jd=float(r["j_date"]),
               qual=str(r["ph_qual"] or "").strip(), ks=num(r["ks_m"]), terr=num(r["err_maj"]), nn=num(r["number_of_neighbours"]), mates=num(r["number_of_mates"])) for r in rows]
    t = Time([s["jd"] for s in st], format="jd", scale="tt")
    earth = get_body_barycentric("earth", t)
    ex, ey, ez = (earth.x.to("au").value, earth.y.to("au").value, earth.z.to("au").value)
    out = []
    for k, s in enumerate(st):
        dt = (s["jd"] - 2457389.0) / 365.25
        ra0, de0 = math.radians(s["ra"]), math.radians(s["dec"])
        ra1 = ra0 + math.radians(s["pmra"] * dt / 3.6e6) / math.cos(de0)
        de1 = de0 + math.radians(s["pmdec"] * dt / 3.6e6)
        u = np.array([math.cos(de1) * math.cos(ra1), math.cos(de1) * math.sin(ra1), math.sin(de1)])
        east = np.array([-math.sin(ra1), math.cos(ra1), 0.0])
        north = np.array([-math.sin(de1) * math.cos(ra1), -math.sin(de1) * math.sin(ra1), math.cos(de1)])
        R = np.array([ex[k], ey[k], ez[k]])
        shift = -(s["plx"] / 1000.0) * (R - np.dot(R, u) * u)
        dra = math.degrees(ra1 - ra0) * 3600 * math.cos(de1) + float(np.dot(shift, east))
        dde = math.degrees(de1 - de0) * 3600 + float(np.dot(shift, north))
        ora = (s["tra"] - s["ra"]) * 3600 * math.cos(de0)
        ode = (s["tdec"] - s["dec"]) * 3600
        s["off"] = math.hypot(ora - dra, ode - dde)
        s["moved"] = math.hypot(dra, dde)
        s["years"] = dt
        if s["off"] > offset_arcsec:
            s["vet"] = ("artefact" if (s["ks"] is not None and s["ks"] < ks_sat) or any(q not in "ABC" for q in s["qual"]) or (s["multi"] or 0) > 2 or (s["ruwe"] or 0) > 1.4
                        else "blend" if (s["nn"] or 0) > 1 or (s["mates"] or 0) > 0 or (s["dup"] or 0) else None)
            out.append(s)
    left = [s for s in out if s["vet"] is None]
    offs = np.array([s["off"] for s in st])
    print("%d stars within %.0f pc with a 2MASS counterpart; offsets from the carried position: median %.2f arcsec, 99th percentile %.2f; above %g arcsec: %d (%d saturated, poor "
          "or flagged, %d with neighbours or mates in 2MASS); unexplained %d" % (len(st), 1000 / plx_min, np.median(offs), np.percentile(offs, 99), offset_arcsec, len(out),
                                                                               sum(s["vet"] == "artefact" for s in out), sum(s["vet"] == "blend" for s in out), len(left)))
    for s in sorted(out, key=lambda s: -s["off"]):
        print("  Gaia DR3 %s: %.2f arcsec off after %.1f arcsec of motion over %.1f yr, G %.2f, Ks %s (%s), RUWE %s, multi %s, 2MASS neighbours %s, mates %s; %s" % (
            s["id"], s["off"], s["moved"], -s["years"], s["g"], s["ks"], s["qual"], s["ruwe"], s["multi"], s["nn"], s["mates"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), hits=out), open(core.results("stars_jumped_2mass.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_2mass_jumped", "Stars that jumped between 2MASS and Gaia", "Gaia DR3 gaia_source and tmass_psc_xsc_best_neighbour; 2MASS PSC (gaiadr1.tmass_original_valid)",
            "stars within %.0f pc with a 2MASS counterpart: %d" % (1000 / plx_min, len(st)), os.path.abspath(__file__), "jumped", len(st), "stars",
            dict(plx_min=plx_min, offset_arcsec=offset_arcsec, ks_sat=ks_sat), "the offsets' own distribution: median %.2f arcsec, 99th percentile %.2f" % (np.median(offs), np.percentile(offs, 99)),
            "a 2MASS position more than %g arcsec from the Gaia course carried back to the 2MASS night" % offset_arcsec,
            "%d: %d saturated, poor or flagged, %d blended in 2MASS; %d unexplained" % (len(out), sum(s["vet"] == "artefact" for s in out), sum(s["vet"] == "blend" for s in out), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["off"], 2), unit="arcsec from the carried position", vet=s["vet"],
                  note="moved %.1f arcsec over %.1f yr, G %.2f, Ks %s (%s), RUWE %s, 2MASS neighbours %s, mates %s" % (
                      s["moved"], -s["years"], s["g"], s["ks"], s["qual"], s["ruwe"], s["nn"], s["mates"])) for s in out])
        print("recorded")


dscgalaxies_DOC = r"""Neighbours Gaia's classifier calls galaxies: Gaia DR3 sources within 100 pc at parallax over error above 10 whose discrete source classifier (astrophysical_parameters, the combined model) gives a galaxy or quasar probability above 0.5, light at a stellar distance that looks like no star to the classifier's photometry, colour and astrometry; each read for its image flags, BP/RP excess and RUWE, since a blend gives the classifier a composite, for its colour, since a source without one or bluer than any star or saturated gives it nothing to read, and in SIMBAD.

    python -I seti.py neighbourhood dscgalaxies [record] [--plx-min 10] [--plx-sigma 10] [--prob-min 0.5]"""


def dscgalaxies_main(record=False, plx_min=10.0, plx_sigma=10.0, prob_min=0.5):
    rows = core.tap_async("gaia", "SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error, g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.ipd_frac_multi_peak, g.duplicated_source, "
                                  "g.phot_bp_rp_excess_factor, a.classprob_dsc_combmod_quasar AS pq, a.classprob_dsc_combmod_galaxy AS pg, a.classprob_dsc_combmod_star AS ps, "
                                  "a.classprob_dsc_combmod_whitedwarf AS pw FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a ON g.source_id = a.source_id "
                                  "WHERE g.parallax > %g AND g.parallax_over_error > %g AND (a.classprob_dsc_combmod_quasar > %g OR a.classprob_dsc_combmod_galaxy > %g)" % (
                                      plx_min, plx_sigma, prob_min, prob_min), maxrec=100000)
    n_all = int(core.tap("gaia", "SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE parallax > %g AND parallax_over_error > %g" % (plx_min, plx_sigma))[0]["n"])
    st = [dict(id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"]), plx=num(r["parallax"]), g=num(r["phot_g_mean_mag"]), c=num(r["bp_rp"]), ruwe=num(r["ruwe"]),
               multi=num(r["ipd_frac_multi_peak"]), dup=num(r["duplicated_source"]), excess=num(r["phot_bp_rp_excess_factor"]), pq=num(r["pq"]), pg=num(r["pg"]), ps=num(r["ps"]), pw=num(r["pw"]))
          for r in rows]
    sim = {}
    for k in range(0, len(st), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % s["id"] for s in st[k:k + 200])):
            sim[str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in st:
        s["simbad"] = sim.get(s["id"])
        typ = (s["simbad"] or (None, "", ""))[1]
        c = s["c"] if s["c"] is not None else 0.0
        s["vet"] = ("artefact" if s["c"] is None or s["c"] < -0.8 or (s["g"] or 99) < 6 else
                    "blend" if (s["multi"] or 0) > 2 or (s["dup"] or 0) or (s["excess"] or 0) > 1.3 + 0.06 * c * c else "astrometry" if (s["ruwe"] or 0) > 1.4 else
                    "natural-type" if typ not in ("", "*", "PM*", "HV*", "low-mass*") else None)
    left = [s for s in st if s["vet"] is None]
    print("%d sources within %.0f pc at parallax over error above %g; called galaxy or quasar at probability above %g: %d (%d without a colour, bluer than any star or saturated, "
          "%d blends, %d of poor astrometry, %d of known types); unexplained %d" % (n_all, 1000 / plx_min, plx_sigma, prob_min, len(st), sum(s["vet"] == "artefact" for s in st),
                                                                                 sum(s["vet"] == "blend" for s in st), sum(s["vet"] == "astrometry" for s in st),
                                                                                 sum(s["vet"] == "natural-type" for s in st), len(left)))
    for s in sorted(st, key=lambda s: -max(s["pq"] or 0, s["pg"] or 0)):
        print("  Gaia DR3 %s: quasar %.2f galaxy %.2f star %.2f WD %.2f, G %s, BP-RP %s, RUWE %s, multi %s, excess %s, %.0f pc; SIMBAD %s; %s" % (
            s["id"], s["pq"] or 0, s["pg"] or 0, s["ps"] or 0, s["pw"] or 0, s["g"], s["c"], s["ruwe"], s["multi"], s["excess"], 1000 / s["plx"], s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=n_all, hits=st), open(core.results("dsc_galaxies_nearby.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_dsc_galaxies_nearby", "Neighbours Gaia's classifier calls galaxies", "Gaia DR3 gaia_source and astrophysical_parameters; SIMBAD",
            "sources within %.0f pc at parallax over error above %g: %d" % (1000 / plx_min, plx_sigma, n_all), os.path.abspath(__file__), "dscgalaxies", n_all, "sources",
            dict(plx_min=plx_min, plx_sigma=plx_sigma, prob_min=prob_min), "Gaia's image flags, RUWE and BP/RP excess; SIMBAD's types",
            "a source at a stellar distance the classifier calls a galaxy or quasar", "%d: %d without a colour, bluer than any star or saturated, %d blends, %d of poor astrometry, "
            "%d of known types; %d unexplained" % (len(st), sum(s["vet"] == "artefact" for s in st), sum(s["vet"] == "blend" for s in st), sum(s["vet"] == "astrometry" for s in st),
                                                  sum(s["vet"] == "natural-type" for s in st), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(max(s["pq"] or 0, s["pg"] or 0), 3), unit="galaxy or quasar probability", vet=s["vet"],
                  note="G %s, BP-RP %s, RUWE %s, multi %s, excess %s, %.0f pc; SIMBAD %s" % (s["g"], s["c"], s["ruwe"], s["multi"], s["excess"], 1000 / s["plx"], s["simbad"]))
             for s in st])
        print("recorded")


receded_DOC = r"""Stars whose distance changed since Hipparcos: the Hipparcos new reduction (van Leeuwen 2007) matched to Gaia DR3, stars whose two parallaxes differ by more than five times their combined error, scaled by the sample's own scatter, and 2 mas, a star closer or farther in 2016 than in 1991 by more than any motion carries it, since a star at 100 pc moving at 100 km/s changes its parallax by a thousandth in that time; each read for RUWE and Gaia's non-single-star flag, since a binary's orbit drags both solutions, for the Hipparcos solution's quality, for brightness, for a Gaia match far fainter than the Hipparcos star, which is a crossmatch gone astray, in SIMBAD, and the remainder against the Gaussian tail of the scaled errors.

    python -I seti.py neighbourhood receded [record] [--sigma 5] [--min-mas 2] [--g-sat 6] [--f2-max 3] [--match-mag 2]"""


def receded_main(record=False, sigma=5.0, min_mas=2.0, g_sat=6.0, f2_max=3.0, match_mag=2.0):
    rows = core.tap_async("gaia", "SELECT h.hip, h.plx, h.e_plx, h.f2, h.hp_mag, h.s_hp, g.source_id, g.parallax, g.parallax_error, g.phot_g_mean_mag, g.ruwe, g.non_single_star, "
                                  "g.ipd_frac_multi_peak, g.astrometric_excess_noise FROM public.hipparcos_newreduction AS h JOIN gaiadr3.hipparcos2_best_neighbour AS n ON "
                                  "h.hip = n.original_ext_source_id JOIN gaiadr3.gaia_source AS g ON n.source_id = g.source_id WHERE h.plx IS NOT NULL AND h.e_plx > 0 AND g.parallax IS NOT NULL",
                          maxrec=200000)
    st = []
    for r in rows:
        hp, ehp, gp, egp = num(r["plx"]), num(r["e_plx"]), num(r["parallax"]), num(r["parallax_error"])
        if None in (hp, ehp, gp, egp):
            continue
        err = math.hypot(ehp, egp)
        st.append(dict(hip=int(r["hip"]), id=str(r["source_id"]), hplx=hp, ehplx=ehp, gplx=gp, egplx=egp, d=hp - gp, z=(hp - gp) / err, f2=num(r["f2"]), g=num(r["phot_g_mean_mag"]), hp=num(r["hp_mag"]),
                       ruwe=num(r["ruwe"]), nss=num(r["non_single_star"]), multi=num(r["ipd_frac_multi_peak"]), aen=num(r["astrometric_excess_noise"]), shp=num(r["s_hp"])))
    from scipy.stats import norm, poisson
    zs0 = np.array([s["z"] for s in st])
    scale = 1.4826 * float(np.median(np.abs(zs0 - np.median(zs0))))
    for s in st:
        s["z"] = s["z"] / scale
    off = [s for s in st if abs(s["z"]) > sigma and abs(s["d"]) > min_mas]
    tail = 2 * norm.sf(sigma) * len(st)
    sim = {}
    for k in range(0, len(off), 200):
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'HIP %d'" % s["hip"] for s in off[k:k + 200])):
            sim[int(str(r["id"]).split()[-1])] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    for s in off:
        s["simbad"] = sim.get(s["hip"])
        typ = (s["simbad"] or (None, "", ""))[1]
        s["vet"] = ("misidentified" if s["g"] is not None and s["hp"] is not None and s["g"] > s["hp"] + match_mag else
                    "binary" if (s["ruwe"] or 0) > 1.4 or (s["nss"] or 0) > 0 or (s["multi"] or 0) > 2 or typ in ("**", "SB*", "EB*", "El*") else
                    "artefact" if (s["g"] or 99) < g_sat or abs(s["f2"] or 0) > f2_max or (s["aen"] or 0) > 1 else None)
    left = [s for s in off if s["vet"] is None]
    p = float(poisson.sf(len(left) - 1, tail)) if left else 1.0
    for s in left:
        if p > 0.01:
            s["vet"] = "chance-rate"
    left = [s for s in off if s["vet"] is None]
    print("%d Hipparcos stars with both parallaxes; the difference's scatter in combined sigmas %.2f, by which the sigmas are scaled; apart by %g scaled sigma and %g mas: %d (%d matched to a "
          "Gaia source %g mag fainter than the Hipparcos star, %d binaries by RUWE, Gaia's flag, image peaks or type, %d saturated or with poor solutions); the rest against the Gaussian "
          "tail's %.1f (Poisson p %.3g); unexplained %d" % (len(st), scale, sigma, min_mas, len(off), sum(s["vet"] == "misidentified" for s in off), match_mag,
                                                             sum(s["vet"] == "binary" for s in off), sum(s["vet"] == "artefact" for s in off), tail, p, len(left)))
    for s in sorted(off, key=lambda s: -abs(s["z"])):
        if s["vet"] is None or len(off) <= 30:
            print("  HIP %d: Hipparcos %.2f +- %.2f, Gaia %.3f +- %.3f mas (%+.2f mas, %.1f sigma), f2 %s, G %s, RUWE %s, excess noise %s; SIMBAD %s; %s" % (
                s["hip"], s["hplx"], s["ehplx"], s["gplx"], s["egplx"], s["d"], s["z"], s["f2"], s["g"], s["ruwe"], s["aen"], s["simbad"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), off=off), open(core.results("parallax_change_hipparcos.json"), "w"), indent=1)
    if record:
        core.record_run(
            "hipparcos_gaia_parallax_change", "Stars whose distance changed since Hipparcos", "Hipparcos new reduction (public.hipparcos_newreduction) and Gaia DR3 by hipparcos2_best_neighbour; SIMBAD",
            "Hipparcos stars with both parallaxes: %d" % len(st), os.path.abspath(__file__), "receded", len(st), "stars",
            dict(sigma=sigma, min_mas=min_mas, g_sat=g_sat, f2_max=f2_max, match_mag=match_mag, scale=round(scale, 3)),
            "the combined errors scaled by the sample's scatter of %.2f; the Gaussian tail's %.1f; RUWE, Gaia's non-single-star flag, image peaks, Hipparcos's goodness of fit, Gaia's "
            "excess noise, the match's brightness, SIMBAD types" % (scale, tail),
            "two parallaxes apart by more than %g scaled sigma and %g mas" % (sigma, min_mas), "%d: %d misidentified, %d binaries, %d saturated or with poor solutions, the rest against "
            "the Gaussian tail (Poisson p %.3g); %d unexplained" % (len(off), sum(s["vet"] == "misidentified" for s in off), sum(s["vet"] == "binary" for s in off),
                                                                   sum(s["vet"] == "artefact" for s in off), p, len(left)),
            [dict(target="hip:%d" % s["hip"], stat=round(s["d"], 2), unit="mas, Hipparcos minus Gaia", vet=s["vet"],
                  note="%.1f sigma, Hipparcos %.2f +- %.2f, Gaia %.3f +- %.3f, f2 %s, G %s, RUWE %s; SIMBAD %s" % (
                      s["z"], s["hplx"], s["ehplx"], s["gplx"], s["egplx"], s["f2"], s["g"], s["ruwe"], s["simbad"])) for s in off])
        print("recorded")


residueclustering_DOC = r"""Sky clustering of our residues: every Gaia DR3 star of the coverage store with its position, the vetted ones among them, the count of vetted pairs within 1, 3 and 10 deg as a share of all possible pairs, since one star vetted by several runs counts once, against the shares that 50 draws give in which each run marks as many of its own targets at random, so that every test's own sky coverage sets the expectation; a civilisation spread over a few parsecs would mark neighbouring stars in different tests, and a pile of residues at one spot would show as pairs beyond the draws.

    python -I seti.py neighbourhood residueclustering [record] [--draws 50]"""


def residueclustering_main(record=False, draws=50):
    from scipy.spatial import cKDTree
    runs = flagzones_runs()
    pos = {r[0]: r[1] for rr in runs.values() for r in rr}
    ids = sorted(pos)
    index = {i: k for k, i in enumerate(ids)}
    lb = np.radians(np.array([[pos[i][0], pos[i][1]] for i in ids]))
    xyz = np.stack([np.cos(lb[:, 1]) * np.cos(lb[:, 0]), np.cos(lb[:, 1]) * np.sin(lb[:, 0]), np.sin(lb[:, 1])], 1)
    scales = (1.0, 3.0, 10.0)

    def pairs(idx):
        tree = cKDTree(xyz[idx])
        return [((tree.count_neighbors(tree, 2 * math.sin(math.radians(d) / 2)) - len(idx)) // 2) / (len(idx) * (len(idx) - 1) / 2) for d in scales]
    real_set = sorted({index[r[0]] for rr in runs.values() for r in rr if r[2]})
    real = pairs(np.array(real_set))
    per_run = [(np.array([index[r[0]] for r in rr]), sum(r[2] for r in rr)) for rr in runs.values()]
    per_run = [(a, k) for a, k in per_run if k]
    rng = np.random.default_rng(5)
    null = []
    for _ in range(draws):
        chosen = set()
        for a, k in per_run:
            chosen.update(rng.choice(a, size=k, replace=False).tolist())
        null.append(pairs(np.array(sorted(chosen))))
    null = np.array(null)
    out = {}
    for k, d in enumerate(scales):
        mean, sd = float(null[:, k].mean()), float(null[:, k].std(ddof=1))
        p = float((null[:, k] >= real[k]).mean())
        out["%g deg" % d] = dict(pairs=1e4 * real[k], expected=1e4 * mean, sd=1e4 * sd, p=p)
    print("%d store stars in %d runs, %d vetted; vetted pairs: %s" % (len(ids), len(runs), len(real_set), "; ".join("within %s %.3f per 10,000 pairs against %.3f +- %.3f (p %.2f)" % (
        k, v["pairs"], v["expected"], v["sd"], v["p"]) for k, v in out.items())))
    if record:
        core.record_run(
            "store_residue_clustering", "Sky clustering of our residues", "the coverage store with Gaia DR3 positions", "Gaia DR3 targets of every run: %d, %d of them vetted" % (len(ids), len(real_set)),
            os.path.abspath(__file__), "residueclustering", len(ids), "stars", dict(draws=draws, scales_deg=list(scales)),
            "%d draws in which each run marks as many of its own targets at random: %s" % (draws, {k: round(v["expected"]) for k, v in out.items()}), "vetted targets paired on the sky beyond the draws",
            "; ".join("within %s %.3f per 10,000 pairs against %.3f +- %.3f (p %.2f)" % (k, v["pairs"], v["expected"], v["sd"], v["p"]) for k, v in out.items()), [])
        print("recorded")


cmbaxis_DOC = r"""The axis of the universe: SIMBAD's objects of unknown nature (type '?') and its unclassified radio, X-ray and gamma-ray sources within 2 deg of the CMB dipole's apex (l 264.02, b +48.25 deg) and antapex, the one direction every observer in the universe can agree on, against 36 patches of the same size at the same Galactic latitude spread in longitude; a beacon placed for anyone who measures the dipole would sit there.

    python -I seti.py neighbourhood cmbaxis [record] [--radius-deg 2] [--controls 36]"""


def cmbaxis_main(record=False, radius_deg=2.0, controls=36):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    types = ("?", "Rad", "X", "gam")

    def counts(l, b):
        c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
        rows = core.simbad("SELECT otype, COUNT(*) AS n FROM basic WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.5f, %.5f, %g)) AND otype IN (%s) GROUP BY otype" % (
            c.ra.deg, c.dec.deg, radius_deg, ",".join("'%s'" % t for t in types)))
        d = {str(r["otype"]).strip(): int(r["n"]) for r in rows}
        return [d.get(t, 0) for t in types]
    out = {}
    for name, (l0, b0) in (("apex", (264.02, 48.25)), ("antapex", (84.02, -48.25))):
        real = counts(l0, b0)
        ctrl = np.array([counts((l0 + 360.0 * (k + 0.5) / controls) % 360, b0) for k in range(controls)])
        ctrl = ctrl[[abs(((l0 + 360.0 * (k + 0.5) / controls) % 360) - l0) > 3 * radius_deg for k in range(controls)]]
        rank = [float((ctrl[:, j] >= real[j]).mean()) for j in range(len(types))]
        out[name] = dict(counts=dict(zip(types, real)), control_median=dict(zip(types, np.median(ctrl, 0).tolist())), control_max=dict(zip(types, ctrl.max(0).tolist())),
                         share_at_or_above=dict(zip(types, rank)))
        print("%s: %s; controls median %s, max %s; share of controls at or above %s" % (name, out[name]["counts"], out[name]["control_median"], out[name]["control_max"], out[name]["share_at_or_above"]))
    if record:
        core.record_run(
            "simbad_cmb_dipole_axis", "The axis of the universe", "SIMBAD", "the circles of %g deg about the CMB dipole's apex and antapex and %d controls at each one's latitude" % (radius_deg, controls),
            os.path.abspath(__file__), "cmbaxis", 2, "directions", dict(radius_deg=radius_deg, controls=controls), "patches of the same size at the same Galactic latitude",
            "more objects of unknown nature or unclassified high-energy and radio sources at the dipole's axis than at any control",
            "; ".join("%s %s against control medians %s" % (k, v["counts"], v["control_median"]) for k, v in out.items()), [])
        print("recorded")


herdedages_DOC = r"""Herded stars, GALAH and ages: the GALAH DR4 dwarfs within 100 pc with a GALAH radial velocity, Gaia DR3 astrometry and an isochrone age, pairs within 10 pc moving together within 1 km/s, and those both hotter than 5600 K, where isochrone ages carry information, whose ages differ by more than 5 Gyr; stars born together share an age, so co-moving stars of very different ages would be a group gathered and kept together; the co-moving pairs are counted against velocity shuffles among the stars, which give the chance pairs of the field and the chance pairs of unlike ages.

    python -I seti.py neighbourhood herdedages [record] [--link-pc 10] [--link-kms 1] [--verr-kms 0.5] [--age-gyr 5] [--teff-min 5600] [--shuffles 20]"""


def herdedages_main(record=False, link_pc=10.0, link_kms=1.0, verr_kms=0.5, age_gyr=5.0, teff_min=5600.0, shuffles=20):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from scipy.spatial import cKDTree
    rows = core.tap("datacentral", "SELECT gaiadr3_source_id, sobject_id, teff, logg, fe_h, age, snr_px_ccd3, rv_comp_1, e_rv_comp_1, rv_comp_2, ruwe FROM galah_dr4.mainstartable "
                                   "WHERE parallax > 10 AND logg > 3.8 AND age IS NOT NULL AND rv_comp_1 IS NOT NULL AND snr_px_ccd3 > 30")
    best = {}
    for r in rows:
        g = str(r["gaiadr3_source_id"])
        if g and g != "None" and (g not in best or (core.num(r["snr_px_ccd3"]) or 0) > (core.num(best[g]["snr_px_ccd3"]) or 0)):
            best[g] = r
    gaia = core.tap_by_ids("gaia", list(best), "SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, pmra_error, pmdec_error FROM gaiadr3.gaia_source WHERE source_id IN (%s)")
    ids = [g for g in best if g in gaia]
    G = [gaia[g] for g in ids]
    B = [best[g] for g in ids]

    def col(rows, k):
        return np.array([core.num(r[k]) if core.num(r[k]) is not None else np.nan for r in rows], float)
    ra, de, plx, eplx, pmra, pmde = (col(G, k) for k in ("ra", "dec", "parallax", "parallax_error", "pmra", "pmdec"))
    rv, erv, age, teff, rv2 = col(B, "rv_comp_1"), col(B, "e_rv_comp_1"), col(B, "age"), col(B, "teff"), col(B, "rv_comp_2")
    c = SkyCoord(ra=ra * u.deg, dec=de * u.deg, distance=(1000 / plx) * u.pc, pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmde * u.mas / u.yr, radial_velocity=rv * u.km / u.s).galactic
    c.representation_type = "cartesian"
    c.differential_type = "cartesian"
    X = np.vstack([c.u.to(u.pc).value, c.v.to(u.pc).value, c.w.to(u.pc).value]).T
    V = np.vstack([c.velocity.d_x.to(u.km / u.s).value, c.velocity.d_y.to(u.km / u.s).value, c.velocity.d_z.to(u.km / u.s).value]).T
    vt = 4.74047 * np.hypot(pmra, pmde) / plx
    sv = np.sqrt((vt * eplx / plx) ** 2 + np.nan_to_num(erv) ** 2)
    ok = np.nonzero(np.isfinite(X).all(1) & np.isfinite(V).all(1) & (sv < verr_kms) & np.isnan(rv2))[0]
    close = [(ok[a], ok[b]) for a, b in cKDTree(X[ok]).query_pairs(link_pc)]

    def comoving(Vs):
        return [(a, b) for a, b in close if np.linalg.norm(Vs[a] - Vs[b]) < link_kms]
    def unlike(pp):
        return [(a, b) for a, b in pp if teff[a] > teff_min and teff[b] > teff_min and abs(age[a] - age[b]) > age_gyr]
    pairs = comoving(V)
    odd = unlike(pairs)
    hot = [(a, b) for a, b in pairs if teff[a] > teff_min and teff[b] > teff_min]
    gaps = sorted(abs(age[a] - age[b]) for a, b in hot)
    rng = np.random.default_rng(17)
    null, null_odd = [], []
    for _ in range(shuffles):
        Vs = V.copy()
        Vs[ok] = V[rng.permutation(ok)]
        pp = comoving(Vs)
        null.append(len(pp))
        null_odd.append(len(unlike(pp)))
    print("%d GALAH dwarfs within 100 pc with ages, velocities and astrometry, %d with velocity errors under %g km/s and one velocity component; co-moving pairs within %g pc and %g km/s: %d "
          "against %.1f from %d velocity shuffles; %d pairs both above %g K, their age gaps at the median %.1f and the largest %.1f Gyr; ages more than %g Gyr apart: %d against %.1f" % (
              len(ids), len(ok), verr_kms, link_pc, link_kms, len(pairs), float(np.mean(null)), shuffles, len(hot), teff_min, float(np.median(gaps)) if gaps else 0, max(gaps) if gaps else 0,
              age_gyr, len(odd), float(np.mean(null_odd))))
    for a, b in odd:
        print("  Gaia DR3 %s (%.1f Gyr, %.0f K) and %s (%.1f Gyr, %.0f K): %.1f pc and %.2f km/s apart" % (
            ids[a], age[a], teff[a], ids[b], age[b], teff[b], np.linalg.norm(X[a] - X[b]), np.linalg.norm(V[a] - V[b])))
    if record:
        core.record_run(
            "galah_herded_ages", "Herded stars, GALAH and ages", "GALAH DR4 (Data Central galah_dr4.mainstartable); Gaia DR3", "dwarfs within 100 pc with ages and 3D velocities to %g km/s: %d" % (
                verr_kms, len(ok)), os.path.abspath(__file__), "herdedages", len(ok), "stars", dict(link_pc=link_pc, link_kms=link_kms, verr_kms=verr_kms, age_gyr=age_gyr, teff_min=teff_min, shuffles=shuffles),
            "%d velocity shuffles: %.1f co-moving pairs, %.1f of unlike ages" % (shuffles, float(np.mean(null)), float(np.mean(null_odd))),
            "co-moving stars above %g K whose ages differ by more than %g Gyr" % (teff_min, age_gyr),
            "%d co-moving pairs against %.1f by chance, %d of them both above %g K with the largest age gap %.1f Gyr; %d of unlike ages against %.1f" % (
                len(pairs), float(np.mean(null)), len(hot), teff_min, max(gaps) if gaps else 0, len(odd), float(np.mean(null_odd))),
            [dict(target="gaia_dr3:%s" % ids[x], stat=round(abs(age[a] - age[b]), 1), unit="Gyr between the pair's ages", vet=None,
                  note="paired with Gaia DR3 %s" % ids[b if x == a else a]) for a, b in odd for x in (a, b)])
        print("recorded")


hzbands_DOC = r"""The temperate rocky-planet hosts in every band: the NASA Exoplanet Archive hosts of planets below 1.8 Earth radii (or 10 Earth masses without a radius) receiving 0.2-2 times Earth's insolation, within 100 pc, each carried to the mean epoch of eRASS1 (X-rays), GALEX AIS (ultraviolet), VLASS (3 GHz) and RACS-low (888 MHz) and matched inside each survey's footprint within a radius widened by its motion over the survey's span, their detected fraction in each band against the other planet hosts within 100 pc weighted to the same temperatures and distances; a world whose inhabitants light up its star in one band would raise that band's detections above the hosts that hold no such world.

    python -I seti.py neighbourhood hzbands [record] [--dist-pc 100] [--insol-lo 0.2] [--insol-hi 2] [--rade-max 1.8]"""

HZBANDS = (("eRASS1 X-rays", "J/A+A/682/A34/erass1-m", 15.0, 2020.1, 0.5, lambda l, b, d: l > 180),
           ("GALEX AIS ultraviolet", "II/335/galex_ais", 5.0, 2007.5, 4.5, lambda l, b, d: True),
           ("VLASS 3 GHz", "J/ApJS/255/30/comp", 3.0, 2018.5, 1.0, lambda l, b, d: d > -40),
           ("RACS-low 888 MHz", "J/other/PASA/38.58/galcut", 10.0, 2019.5, 0.5, lambda l, b, d: d < 41 and abs(b) > 5))


def hzbands_main(record=False, dist_pc=100.0, insol_lo=0.2, insol_hi=2.0, rade_max=1.8):
    from scipy.stats import binom
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    rows = core.exoplanets("SELECT hostname, pl_name, ra, dec, sy_pmra, sy_pmdec, sy_dist, st_teff, pl_rade, pl_bmasse, pl_insol FROM pscomppars WHERE sy_dist < %g" % dist_pc)
    hosts = {}
    for r in rows:
        h = hosts.setdefault(r["hostname"], dict(name=r["hostname"], ra=core.num(r["ra"]), dec=core.num(r["dec"]), pmra=core.num(r["sy_pmra"]) or 0.0, pmdec=core.num(r["sy_pmdec"]) or 0.0,
                                                  d=core.num(r["sy_dist"]), teff=core.num(r["st_teff"]), hz=False))
        rade, mass, ins = core.num(r["pl_rade"]), core.num(r["pl_bmasse"]), core.num(r["pl_insol"])
        rocky = (rade is not None and rade <= rade_max) or (rade is None and mass is not None and mass <= 10)
        if rocky and ins is not None and insol_lo <= ins <= insol_hi:
            h["hz"] = True
    hosts = [h for h in hosts.values() if h["ra"] is not None and h["teff"] is not None and h["d"] is not None]
    gal = SkyCoord([h["ra"] for h in hosts] * u.deg, [h["dec"] for h in hosts] * u.deg).galactic
    for h, g in zip(hosts, gal):
        h["l"], h["b"] = float(g.l.deg), float(g.b.deg)
    tb, db = np.array([3000, 3500, 4000, 5000, 6000, 1e5]), np.array([0, 10, 20, 50, 1e5])

    def cell(h):
        return int(np.digitize(h["teff"], tb)), int(np.digitize(h["d"], db))
    out = {}
    for band, cat, rad, epoch, span, inside in HZBANDS:
        sel = [h for h in hosts if inside(h["l"], h["b"], h["dec"])]
        up = [dict(source_id=k, ra=h["ra"] + (epoch - 2016.0) * h["pmra"] / 3.6e6 / math.cos(math.radians(h["dec"])), dec=h["dec"] + (epoch - 2016.0) * h["pmdec"] / 3.6e6)
              for k, h in enumerate(sel)]
        allow = [rad + span * math.hypot(h["pmra"], h["pmdec"]) / 1000 for h in sel]
        hit = {int(m["source_id"]) for m in core.xmatch(up, cat, min(max(allow), 60.0)) if float(m["angDist"]) <= allow[int(m["source_id"])]} if up else set()
        for k, h in enumerate(sel):
            h["det"] = k in hit
        hz = [h for h in sel if h["hz"]]
        ctrl = [h for h in sel if not h["hz"]]
        rate = {}
        for h in ctrl:
            rate.setdefault(cell(h), []).append(h["det"])
        allrate = float(np.mean([h["det"] for h in ctrl])) if ctrl else 0.0
        exp = sum(float(np.mean(rate[cell(h)])) if cell(h) in rate and len(rate[cell(h)]) >= 5 else allrate for h in hz)
        n = sum(h["det"] for h in hz)
        p = float(binom.sf(n - 1, len(hz), exp / len(hz))) if hz and n else 1.0
        p_low = float(binom.cdf(n, len(hz), exp / len(hz))) if hz else 1.0
        out[band] = dict(n_hz=len(hz), det=n, expected=exp, p=p, p_low=p_low, n_ctrl=len(ctrl), ctrl_det=sum(h["det"] for h in ctrl), names=[h["name"] for h in hz if h["det"]])
        print("%s: %d temperate rocky-planet hosts in the footprint, %d detected against %.1f from %d other hosts (%d detected) at the same temperatures and distances (p %.3g, lower tail %.3g); %s" % (
            band, len(hz), n, exp, len(ctrl), out[band]["ctrl_det"], p, p_low, out[band]["names"]))
    if record:
        core.record_run(
            "exoplanet_hz_hosts_bands", "The temperate rocky-planet hosts in every band", "NASA Exoplanet Archive composite table; eRASS1, GALEX AIS, VLASS and RACS-low by CDS XMatch",
            "hosts within %g pc of planets below %g Earth radii at %g-%g times Earth's insolation: %d" % (dist_pc, rade_max, insol_lo, insol_hi, sum(h["hz"] for h in hosts)), os.path.abspath(__file__),
            "hzbands", len(hosts), "hosts", dict(dist_pc=dist_pc, insol=[insol_lo, insol_hi], rade_max=rade_max),
            "the other planet hosts within %g pc, weighted to the same temperature and distance cells" % dist_pc, "a band in which the temperate rocky-planet hosts are detected beyond their peers",
            "; ".join("%s %d of %d against %.1f (p %.3g, lower tail %.3g)" % (k, v["det"], v["n_hz"], v["expected"], v["p"], v["p_low"]) for k, v in out.items()), [])
        print("recorded")


missingbright_DOC = r"""Clusters missing their brightest stars: the open clusters of Hunt and Reffert (2023, VizieR J/A+A/673/A114) between 100 and 1000 pc, younger than 400 Myr, with A_V below 1.5 and 300 or more members, their members of probability above 0.7 counted at absolute G 0.5-2.5, late-B to F stars that every such cluster still holds on the main sequence, and at absolute G 4-6.5; each cluster's bright count against the faint count times the ratio the clusters of its age give, fitted in log age across all of them, and the deficits' Poisson tails corrected for the number of clusters; a civilisation that lifts or harvests its massive stars would leave a cluster short of them.

    python -I seti.py neighbourhood missingbright [record] [--prob-min 0.7] [--min-members 300] [--max-logage 8.6]"""


def missingbright_main(record=False, prob_min=0.7, min_members=300, max_logage=8.6):
    from scipy.stats import poisson
    cl = core.tap("vizier", 'SELECT Name, N, dist50, logAge50, AV50, MOD50 FROM "J/A+A/673/A114/clusters" WHERE Type = \'o\' AND N >= %d AND dist50 BETWEEN 100 AND 1000 AND logAge50 < %g '
                            'AND AV50 < 1.5' % (min_members, max_logage))
    info = {str(r["Name"]).strip(): r for r in cl}
    names = sorted(info)
    counts = {n: [0, 0] for n in names}
    for k in range(0, len(names), 60):
        q = 'SELECT Name, Gmag FROM "J/A+A/673/A114/members" WHERE Prob > %g AND Name IN (%s)' % (prob_min, ",".join("'%s'" % n.replace("'", "''") for n in names[k:k + 60]))
        for r in core.tap("vizier", q, timeout=600):
            n = str(r["Name"]).strip()
            g = core.num(r["Gmag"])
            if g is None or n not in info:
                continue
            mg = g - float(info[n]["MOD50"]) - 0.83 * float(info[n]["AV50"])
            if 0.5 <= mg < 2.5:
                counts[n][0] += 1
            elif 4.0 <= mg < 6.5:
                counts[n][1] += 1
    rows = [(n, counts[n][0], counts[n][1], float(info[n]["logAge50"]), float(info[n]["dist50"])) for n in names if counts[n][1] >= 20]
    la = np.array([r[3] for r in rows])
    lr = np.log(np.array([(r[1] + 0.5) / r[2] for r in rows]))
    coef = np.polyfit(la, lr, 1)
    out = []
    for n, b, f, a, d in rows:
        exp = f * math.exp(np.polyval(coef, a))
        p = float(poisson.cdf(b, exp))
        out.append(dict(name=n, bright=b, faint=f, logage=a, dist=d, expected=exp, p=p, p_corr=min(1.0, p * len(rows)), vet=None))
    short = [o for o in out if o["p_corr"] < 0.01]
    print("%d open clusters with 20 or more faint members; bright-to-faint ratio %.3f at log age 7.5 and %.3f at 8.5; clusters short of bright stars beyond the %d trials: %d" % (
        len(out), math.exp(np.polyval(coef, 7.5)), math.exp(np.polyval(coef, 8.5)), len(out), len(short)))
    for o in sorted(out, key=lambda o: o["p"])[:10]:
        print("  %s (%.0f pc, log age %.2f): %d bright against %.1f from %d faint (p %.2g, corrected %.2g)" % (o["name"], o["dist"], o["logage"], o["bright"], o["expected"], o["faint"], o["p"], o["p_corr"]))
    json.dump(out, open(core.results("clusters_missing_bright.json"), "w"), indent=1)
    if record:
        core.record_run(
            "hunt_reffert_missing_bright", "Clusters missing their brightest stars", "Hunt and Reffert (2023) clusters and members (VizieR J/A+A/673/A114)",
            "open clusters at 100-1000 pc younger than log age %g with %d or more members and 20 faint members: %d" % (max_logage, min_members, len(out)), os.path.abspath(__file__), "missingbright",
            len(out), "clusters", dict(prob_min=prob_min, min_members=min_members, max_logage=max_logage), "the bright-to-faint ratio fitted in log age across the clusters, Poisson, corrected for %d trials" % len(out),
            "a cluster short of its late-B to F stars beyond chance", "%d short beyond the trials; the deepest %s" % (len(short), "; ".join("%s %d against %.1f (p %.2g)" % (
                o["name"], o["bright"], o["expected"], o["p"]) for o in sorted(out, key=lambda o: o["p"])[:3])),
            [dict(target="cluster:%s" % o["name"], stat=o["bright"], unit="members at absolute G 0.5-2.5", p=o["p"], vet=o["vet"], note="against %.1f from %d faint members, log age %.2f, %.0f pc" % (
                o["expected"], o["faint"], o["logage"], o["dist"])) for o in short])
        print("recorded")


nearoutliers_DOC = r"""A model-free outlier search over the nearest stars: the thousand Gaia DR3 sources with the largest parallaxes measured at 10 sigma, their Gaia, 2MASS, AllWISE, GALEX and eRASS1 measurements, each archive's colours or flux ratio set against the stars of the same BP-RP by a running median and robust spread, the ultracool dwarfs fainter than absolute G 15, whose BP flux is noise, against each other by G-RP, a star counted odd in an archive when one of that archive's quantities stands 5 robust widths off, and the stars odd in three or more independent archives read for what makes them so: a white dwarf below the main sequence, a companion in Gaia's astrometry, or youth and activity in X-rays and the ultraviolet; a star unlike its kind in several independent instruments at once would remain.

    python -I seti.py neighbourhood nearoutliers [record] [--n 1000] [--widths 5] [--min-archives 3]"""


def nearoutliers_main(record=False, n=1000, widths=5.0, min_archives=3):
    g = core.tap("gaia", "SELECT TOP %d source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp, g_rp, ruwe, phot_bp_rp_excess_factor FROM gaiadr3.gaia_source "
                         "WHERE parallax_over_error > 10 AND bp_rp IS NOT NULL ORDER BY parallax DESC" % n)
    st = {r["source_id"]: dict(id=r["source_id"], ra=float(r["ra"]), dec=float(r["dec"]), plx=float(r["parallax"]), pmra=float(r["pmra"] or 0), pmdec=float(r["pmdec"] or 0),
                               g=float(r["phot_g_mean_mag"]), c=float(r["bp_rp"]), grp=core.num(r["g_rp"]), ruwe=core.num(r["ruwe"]), cs=core.cstar(r["phot_bp_rp_excess_factor"], r["bp_rp"])
                               if core.num(r["phot_bp_rp_excess_factor"]) is not None else None) for r in g}
    ids = list(st)
    q2 = ("SELECT x.source_id, t.j_m, t.h_m, t.ks_m FROM gaiadr3.tmass_psc_xsc_best_neighbour AS x JOIN gaiadr3.tmass_psc_xsc_join AS j ON j.clean_tmass_psc_xsc_oid = x.clean_tmass_psc_xsc_oid "
          "JOIN gaiadr1.tmass_original_valid AS t ON t.designation = j.original_psc_source_id WHERE x.source_id IN (%s)")
    qw = ("SELECT x.source_id, w.w1mpro, w.w2mpro, w.w3mpro, w.w4mpro, w.ph_qual FROM gaiadr3.allwise_best_neighbour AS x JOIN gaiadr1.allwise_original_valid AS w ON x.allwise_oid = w.allwise_oid "
          "WHERE x.source_id IN (%s)")
    for q in (q2, qw):
        for i in range(0, len(ids), 400):
            for r in core.tap("gaia", q % ",".join(ids[i:i + 400])):
                st[r["source_id"]].update({k: core.num(v) if k != "ph_qual" else v for k, v in r.items() if k != "source_id"})

    def carried(epoch):
        return [dict(source_id=s["id"], ra=s["ra"] + (epoch - 2016.0) * s["pmra"] / 3.6e6 / math.cos(math.radians(s["dec"])), dec=s["dec"] + (epoch - 2016.0) * s["pmdec"] / 3.6e6)
                for s in st.values()]
    for m in core.xmatch(carried(2007.5), "II/335/galex_ais", 20.0):
        s = st[m["source_id"]]
        if float(m["angDist"]) < 5 + 4.5 * math.hypot(s["pmra"], s["pmdec"]) / 1000 and core.num(m.get("NUVmag")) is not None:
            s["nuv"] = min(s.get("nuv", 99), core.num(m["NUVmag"]))
    north = [x for x in carried(2020.1)]
    for m in core.xmatch(north, "J/A+A/682/A34/erass1-m", 15.0):
        f = core.num(m.get("ML_FLUX_1") or m.get("Flux") or m.get("FluxP"))
        if f:
            st[m["source_id"]]["fx"] = max(st[m["source_id"]].get("fx", 0), f)
    for s in st.values():
        s["mg"] = s["g"] + 5 * math.log10(s["plx"] / 100)
        s["q"] = {}
        s["q"]["gaia"] = dict(mg=s["mg"], ruwe=math.log10(s["ruwe"]) if s["ruwe"] else None, cstar=s["cs"])
        s["q"]["2mass"] = dict(jh=(s.get("j_m") - s.get("h_m")) if s.get("j_m") and s.get("h_m") else None, gk=(s["g"] - s["ks_m"]) if s.get("ks_m") else None)
        s["q"]["allwise"] = dict(w12=(s["w1mpro"] - s["w2mpro"]) if s.get("w1mpro") and s.get("w2mpro") else None, w13=(s["w1mpro"] - s["w3mpro"]) if s.get("w1mpro") and s.get("w3mpro")
                                 and str(s.get("ph_qual") or "xxx")[2:3] in ("A", "B") else None)
        s["q"]["galex"] = dict(nuvg=(s["nuv"] - s["g"]) if s.get("nuv") else None)
        s["q"]["erass1"] = dict(fxg=(math.log10(s["fx"]) + 0.4 * s["g"]) if s.get("fx") else None)
    stars = list(st.values())
    for s in stars:
        # an ultracool dwarf's BP flux is noise, so the faint ones are set against each other by G-RP
        s["cool"] = s["mg"] > 15 and s["grp"] is not None
        s["key"] = s["grp"] if s["cool"] else s["c"]
    odd = {s["id"]: [] for s in stars}
    for arch, key, cool in [(a, k, c) for a in ("gaia", "2mass", "allwise", "galex", "erass1") for k in stars[0]["q"][a] for c in (False, True)]:
        if cool and key == "cstar":
            continue
        order = sorted([s for s in stars if s["cool"] == cool], key=lambda s: s["key"])
        vals = [(s["key"], s["q"][arch][key], s["id"]) for s in order if s["q"][arch][key] is not None]
        if len(vals) < 30:
            continue
        ys = np.array([v[1] for v in vals])
        for k, (c, y, sid) in enumerate(vals):
            lo, hi = max(0, k - 25), min(len(vals), k + 26)
            win = np.delete(ys[lo:hi], k - lo)
            med = float(np.median(win))
            mad = 1.4826 * float(np.median(np.abs(win - med))) or 1e-3
            if abs(y - med) > widths * mad:
                odd[sid].append("%s %s %+.1f" % (arch, key, (y - med) / mad))
    flagged = []
    for s in stars:
        archs = sorted({x.split()[0] for x in odd[s["id"]]})
        if len(archs) >= min_archives:
            why = []
            if s["mg"] > 10 + 2.5 * s["c"] or s["c"] < 0.5 and s["mg"] > 9:
                why.append("below the main sequence")
            if (s["ruwe"] or 0) > 1.4:
                why.append("companion")
            if s.get("fx") or s.get("nuv"):
                why.append("active")
            s["archs"], s["why"] = archs, why
            s["vet"] = ("white-dwarf" if "below the main sequence" in why and s["c"] < 1.0 else "binary" if "companion" in why else "active-star" if "active" in why else None)
            flagged.append(s)
    left = [s for s in flagged if s["vet"] is None]
    print("%d nearest stars (parallax %.1f mas and beyond); odd in one archive %d, in %d or more %d: white dwarfs %d, binaries %d, active %d; unexplained %d" % (
        len(stars), min(s["plx"] for s in stars), sum(1 for s in stars if odd[s["id"]]), min_archives, len(flagged), sum(s["vet"] == "white-dwarf" for s in flagged),
        sum(s["vet"] == "binary" for s in flagged), sum(s["vet"] == "active-star" for s in flagged), len(left)))
    for s in flagged:
        print("  Gaia DR3 %s: %.1f pc, BP-RP %.2f, M_G %.2f; %s; %s" % (s["id"], 1000 / s["plx"], s["c"], s["mg"], "; ".join(odd[s["id"]]), s["vet"] or "unexplained"))
    if record:
        core.record_run(
            "nearest_stars_multi_archive_outliers", "A model-free outlier search over the nearest stars", "Gaia DR3; 2MASS and AllWISE by Gaia's best neighbours; GALEX AIS and eRASS1 by CDS XMatch",
            "the %d Gaia DR3 sources with the largest parallaxes at 10 sigma" % len(stars), os.path.abspath(__file__), "nearoutliers", len(stars), "stars",
            dict(n=n, widths=widths, min_archives=min_archives), "each quantity against the running median and robust spread of the 50 stars nearest in BP-RP",
            "a star odd in %d or more independent archives with no white dwarf, companion or activity behind it" % min_archives,
            "%d odd in %d or more archives: %d white dwarfs, %d binaries, %d active; %d unexplained" % (len(flagged), min_archives, sum(s["vet"] == "white-dwarf" for s in flagged),
                                                                                                    sum(s["vet"] == "binary" for s in flagged), sum(s["vet"] == "active-star" for s in flagged), len(left)),
            [dict(target="gaia_dr3:%s" % s["id"], stat=len(s["archs"]), unit="archives in which the star is odd", vet=s["vet"], note="; ".join(odd[s["id"]])) for s in flagged])
        print("recorded")


timecluster_DOC = r"""Time clustering of our residues: the coverage-store rows that record a candidate event with its date in the note (an MJD, a TESS BTJD or a calendar date) and no natural explanation, a candidate being a row with a p-value below 0.01 or a cosmic-ray multiplet (a row in days between members), since most dated rows record the epochs searched rather than anything found, reduced to the distinct days of each run, and the days on which unexplained events of two runs on different instruments (by the run's leading name) fall within a day of each other, against the count when each run's days are shifted together by a random offset of up to a year, 500 times; something acting at one moment would mark the records of independent instruments on the same day.

    python -I seti.py neighbourhood timecluster [record] [--window-d 1] [--shifts 500]"""


def timecluster_main(record=False, window_d=1.0, shifts=500):
    import datetime as _dt
    pat = re.compile(r"MJD\s*([0-9]{5}(?:\.[0-9]+)?)|BTJD\s*([0-9]{4}(?:\.[0-9]+)?)|\b((?:19|20)[0-9]{2}-[01][0-9]-[0-3][0-9])")
    db = sqlite3.connect(core.COVERAGE)
    days = {}
    for sid, note, pv, unit in db.execute("SELECT search_id, note, p, unit FROM coverage WHERE vet IS NULL AND note IS NOT NULL"):
        if not ((pv is not None and pv < 0.01) or unit == "days between members"):
            continue
        m = pat.search(note)
        if not m:
            continue
        if m.group(1):
            d = float(m.group(1))
        elif m.group(2):
            d = float(m.group(2)) + 56999.5
        else:
            d = (_dt.date.fromisoformat(m.group(3)) - _dt.date(1858, 11, 17)).days
        days.setdefault(sid, set()).add(math.floor(d))
    runs = sorted(days)
    arr = {r: np.array(sorted(days[r]), float) for r in runs}

    def coincide(shift):
        n, pairs = 0, []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                if runs[i].split("_")[0] == runs[j].split("_")[0]:
                    continue
                a, b = arr[runs[i]] + shift[i], arr[runs[j]] + shift[j]
                k = np.searchsorted(b, a - window_d)
                for x, kk in zip(a, k):
                    while kk < len(b) and b[kk] <= x + window_d:
                        n += 1
                        if not shift.any():
                            pairs.append((runs[i], runs[j], int(x)))
                        kk += 1
        return n, pairs
    real, pairs = coincide(np.zeros(len(runs)))
    rng = np.random.default_rng(29)
    null = np.array([coincide(np.round(rng.uniform(-365, 365, len(runs))))[0] for _ in range(shifts)])
    p = float((null >= real).mean())
    print("%d runs with dated candidate events, %d distinct days; days of two runs on different instruments within %g d: %d against %.1f +- %.1f from %d shifts (p %.3g)" % (
        len(runs), sum(len(v) for v in days.values()), window_d, real, null.mean(), null.std(), shifts, p))
    by = {}
    for a, b, d in pairs:
        by.setdefault((a, b), []).append(d)
    for (a, b), ds in sorted(by.items(), key=lambda kv: -len(kv[1]))[:10]:
        print("  %s and %s: %d days, e.g. MJD %s" % (a, b, len(ds), ds[:3]))
    if record:
        core.record_run(
            "store_residue_time_clustering", "Time clustering of our residues", "the coverage store's unexplained candidate events with dates",
            "runs with dated candidate events (p below 0.01 or cosmic-ray multiplets): %d, %d distinct days" % (len(runs), sum(len(v) for v in days.values())), os.path.abspath(__file__), "timecluster", len(runs), "runs",
            dict(window_d=window_d, shifts=shifts), "%d shifts of each run's days by up to a year: %.1f +- %.1f" % (shifts, null.mean(), null.std()),
            "unexplained events of independent runs on the same day beyond the shifted runs", "%d coinciding days against %.1f (p %.3g)" % (real, null.mean(), p), [])
        print("recorded")


APPROACHES = {
    "timecluster": (lambda: core.run_cli(timecluster_main), timecluster_DOC),
    "nearoutliers": (lambda: core.run_cli(nearoutliers_main), nearoutliers_DOC),
    "missingbright": (lambda: core.run_cli(missingbright_main), missingbright_DOC),
    "hzbands": (lambda: core.run_cli(hzbands_main), hzbands_DOC),
    "herdedages": (lambda: core.run_cli(herdedages_main), herdedages_DOC),
    "cmbaxis": (lambda: core.run_cli(cmbaxis_main), cmbaxis_DOC),
    "residueclustering": (lambda: core.run_cli(residueclustering_main), residueclustering_DOC),
    "receded": (lambda: core.run_cli(receded_main), receded_DOC),
    "dscgalaxies": (lambda: core.run_cli(dscgalaxies_main), dscgalaxies_DOC),
    "jumped": (lambda: core.run_cli(jumped_main), jumped_DOC),
    "retroclusters": (lambda: core.run_cli(retroclusters_main), retroclusters_DOC),
    "darkcircles": (lambda: core.run_cli(darkcircles_main), darkcircles_DOC),
    "escapees": (lambda: core.run_cli(escapees_main), escapees_DOC),
    "roundbinaries": (lambda: core.run_cli(roundbinaries_main), roundbinaries_DOC),
    "watchers": (lambda: core.run_cli(watchers_main), watchers_DOC),
    "trojanpoints": (lambda: core.run_cli(trojanpoints_main), trojanpoints_DOC),
    "comovers": (lambda: core.run_cli(comovers_main), comovers_DOC),
    "siblings": (lambda: core.run_cli(siblings_main), siblings_DOC),
    "hvsorigin": (lambda: core.run_cli(hvsorigin_main), hvsorigin_DOC),
    "nearlens": (lambda: core.run_cli(nearlens_main), nearlens_DOC),
    "encounters": (lambda: core.run_cli(encounters_main), encounters_DOC),
    "qsomove": (lambda: core.run_cli(qsomove_main), qsomove_DOC),
    "galplx": (lambda: core.run_cli(galplx_main), galplx_DOC),
    "uncatalogued": (lambda: core.run_cli(uncatalogued_main), uncatalogued_DOC),
    "lattice": (lambda: core.run_cli(lattice_main), lattice_DOC),
    "fastrv": (lambda: core.run_cli(fastrv_main), fastrv_DOC),
    "still": (lambda: core.run_cli(still_main), still_DOC),
    "barnard": (lambda: core.run_cli(barnard_main), barnard_DOC),
    "proxima": (lambda: core.run_cli(proxima_main), proxima_DOC),
    "mysteries": (lambda: core.run_cli(mysteries_main), mysteries_DOC),
    "unbound": (lambda: core.run_cli(unbound_main), unbound_DOC),
    "oacells": (lambda: core.run_cli(oacells_main), oacells_DOC),
    "noclass": (lambda: core.run_cli(noclass_main), noclass_DOC),
    "cmbrest": (lambda: core.run_cli(cmbrest_main), cmbrest_DOC),
    "thrust": (lambda: core.run_cli(thrust_main), thrust_DOC),
    "copies": (lambda: core.run_cli(copies_main), copies_DOC),
    "flagetz": (lambda: core.run_cli(flagetz_main), flagetz_DOC),
    "flagdirs": (lambda: core.run_cli(flagdirs_main), flagdirs_DOC),
    "rest": (rest_cli, rest_DOC),
    "headon": (lambda: core.run_cli(headon_main), headon_DOC),
    "qsolike": (lambda: core.run_cli(qsolike_main), qsolike_DOC),
    "no2mass": (lambda: core.run_cli(no2mass_main), no2mass_DOC),
    "unknown": (lambda: core.run_cli(unknown_main), unknown_DOC),
    "hotsmall": (hotsmall_cli, hotsmall_DOC),
    "poles": (poles_cli, poles_DOC),
    "intrinsic": (intrinsic_cli, intrinsic_DOC),
    "herded": (lambda: core.run_cli(herded_main), herded_DOC),
    "imaging": (imaging_cli, imaging_DOC),
    "phase_space": (phase_space_cli, phase_space_DOC),
    "etz": (lambda: core.run_cli(etz_main), etz_DOC),
    "polarised": (polarised_cli, polarised_DOC),
    "beams": (beams_cli, beams_DOC),
    "joint": (joint_cli, joint_DOC),
}


