"""Artificial objects among the small bodies of the Solar System."""
import argparse
import glob
import json
import math
import os
import re
import sys
import urllib.parse
import zipfile

import numpy as np

from setitools import core
from setitools.core import num, quiet_warnings


# --------------------------------------------------------------------------------------------------------------

nga_DOC = r"""Hollow or thin small bodies: area-to-mass ratios from JPL's radial non-gravitational term, densities below 500 kg/m^3 at the most favourable natural assumptions, transverse or normal terms marking activity.

    python -I seti.py solar_system nga fetch          (every SBDB object with a radial term)
    python -I seti.py solar_system nga analyse [--rho 500]
    python -I seti.py solar_system nga record"""

nga_ROOT = os.environ.get("NGA_ROOT", core.data("nga"))
nga_HERE = core.RESULTS
nga_API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
nga_FIELDS = ("full_name,pdes,kind,class,H,diameter,albedo,A1,A1_sigma,A2,A2_sigma,A3,A3_sigma,DT,e,q,a,i,"
          "n_obs_used,data_arc,condition_code,first_obs,last_obs")
nga_AU_D2 = 1.495978707e11 / 86400.0 ** 2  # m/s^2 per au/d^2
nga_S_OVER_C = 1361.0 / 2.99792458e8  # N/m^2 at 1 AU
nga_CR_MAX = 1.5
nga_ALBEDO_MAX = 0.5


def nga_fetch():
    os.makedirs(nga_ROOT, exist_ok=True)
    u = nga_API + "?" + urllib.parse.urlencode({"fields": nga_FIELDS, "sb-cdata": '{"AND":["A1|DF"]}', "full-prec": "true"})
    d = json.loads(core.get(u, timeout=300))
    json.dump(d, open(os.path.join(nga_ROOT, "sbdb_A1.json"), "w"))
    print("%d objects with a radial non-gravitational term" % d["count"])


def nga_rows():
    d = json.load(open(os.path.join(nga_ROOT, "sbdb_A1.json")))
    return [dict(zip(d["fields"], r)) for r in d["data"]]


def nga_assess(r, rho_limit):
    """Area-to-mass ratio, largest natural density it allows and flags, or None."""
    a1, s1 = num(r["A1"]), num(r["A1_sigma"])
    if a1 is None or not s1 or a1 <= 0 or a1 / s1 < 3:
        return None
    amr = a1 * nga_AU_D2 / (nga_CR_MAX * nga_S_OVER_C)  # smallest A/m the acceleration allows
    if num(r["diameter"]):
        dmin = num(r["diameter"]) * 1000.0
        dsrc = "measured"
    elif num(r["H"]) is not None:
        dmin = 1329.0e3 / math.sqrt(nga_ALBEDO_MAX) * 10 ** (-num(r["H"]) / 5.0)
        dsrc = "H at albedo 0.5"
    else:
        return None
    rho_max = 1.5 / (amr * dmin)
    a2, s2 = num(r["A2"]), num(r["A2_sigma"])
    a3, s3 = num(r["A3"]), num(r["A3_sigma"])
    snr2 = abs(a2) / s2 if a2 is not None and s2 else 0.0
    snr3 = abs(a3) / s3 if a3 is not None and s3 else 0.0
    # activity: a 3-sigma normal term, or a 3-sigma transverse term above a tenth of the radial, which neither radiation pressure nor Yarkovsky gives
    active = bool(snr3 >= 3 or (snr2 >= 3 and abs(a2) > 0.1 * a1))
    a, e, inc = num(r["a"]), num(r["e"]), num(r["i"])
    t_earth = (1.0 / a + 2.0 * math.cos(math.radians(inc)) * math.sqrt(a * (1 - e * e))
               if a and a > 0 and e is not None and e < 1 and inc is not None else None)
    return dict(name=r["full_name"].strip(), pdes=r["pdes"], kind=r["kind"], cls=r["class"], H=num(r["H"]),
                d_min_m=dmin, d_src=dsrc, A1=a1, A1_snr=a1 / s1, amr=amr, rho_max=rho_max,
                A2=a2, A2_snr=snr2, A3=a3, A3_snr=snr3, active=active, light=rho_max < rho_limit,
                t_earth=t_earth, q=num(r["q"]), arc=r["data_arc"], n_obs=r["n_obs_used"])


def nga_analyse(rho_limit):
    out = [x for x in (nga_assess(r, rho_limit) for r in nga_rows()) if x]
    ast = [x for x in out if x["kind"].startswith("a")]
    print("%d objects with a radial term; %d with A1 > 0 at 3 sigma or more (%d asteroids, %d comets)"
          % (len(nga_rows()), len(out), len(ast), len(out) - len(ast)))
    light = sorted([x for x in ast if x["light"]], key=lambda x: x["rho_max"])
    print("asteroids whose radial acceleration needs a density below %d kg/m^3 at C_R %.1f and albedo %.1f: %d"
          % (rho_limit, nga_CR_MAX, nga_ALBEDO_MAX, len(light)))
    for x in light:
        print("  %-32s %-4s H %5.2f D>=%6.1f m (%s) A1 %.2e (%.1f sigma) A/m>=%.2e m2/kg rho<=%7.1f kg/m3; A2 %.1f "
              "sigma, A3 %.1f sigma: %s; Earth Tisserand %s, arc %s d, %s obs"
              % (x["name"][:32], x["cls"], x["H"] or 0, x["d_min_m"], x["d_src"], x["A1"], x["A1_snr"], x["amr"],
                 x["rho_max"], x["A2_snr"], x["A3_snr"], "non-radial terms (activity)" if x["active"] else "radial",
                 "%.3f" % x["t_earth"] if x["t_earth"] else "-", x["arc"], x["n_obs"]))
    json.dump(out, open(os.path.join(nga_HERE, "nga_amr.json"), "w"), indent=1)
    return out


def nga_record():
    out = json.load(open(os.path.join(nga_HERE, "nga_amr.json")))
    every = nga_rows()
    by = {x["pdes"]: x for x in out}
    rws = []
    for r in every:
        x = by.get(r["pdes"])
        vet = None
        if x and x["light"]:
            vet = "outgassing" if x["active"] else None
        rws.append(dict(target="mpc:" + r["pdes"], stat=x["amr"] if x else None, upper_limit=x["rho_max"] if x else None,
                        unit="kg/m^3, largest density allowed (C_R 1.5, albedo 0.5)" if x else None, vet=vet,
                        note=r["full_name"].strip()))
    light = [x for x in out if x["light"] and x["kind"].startswith("a")]
    core.record_search("nga_amr", "Area-to-mass ratio from non-gravitational acceleration",
                           archive="JPL Small-Body Database query API", release="current solutions",
                           selection="every object with a radial non-gravitational term A1", code=os.path.abspath(__file__), entry="nga",
                           params=dict(cr=nga_CR_MAX, albedo=nga_ALBEDO_MAX, a1_snr=3), n_targets=len(rws),
                           result="%d asteroids lighter than a rubble pile at the most favourable assumptions, %d of "
                                  "them with transverse or normal terms of activity" % (len(light), sum(x["active"] for x in light)),
                           ledger="Area-to-mass ratio from non-gravitational acceleration")
    core.record_targets("nga_amr", rws)
    print("recorded %d objects" % len(rws))


def nga_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "record"))
    p.add_argument("--rho", type=float, default=500.0)
    a = p.parse_args()
    {"fetch": nga_fetch, "analyse": lambda: nga_analyse(a.rho), "record": nga_record}[a.mode]()


def nga_cli():
    nga_main()


# --------------------------------------------------------------------------------------------------------------

surfaces_DOC = r"""Metal and solar-cell surfaces in Gaia DR3 asteroid reflectance spectra: chi-square drop above 25 when a template joins a four-component asteroid model, checked against taxonomy and albedo, with injection recovery.

    python -I seti.py solar_system surfaces
    python -I seti.py solar_system surfaces record"""

surfaces_ROOT = os.environ.get("SSO_ROOT", core.data("gaia_sso"))
surfaces_HERE = core.RESULTS
surfaces_BANDS = np.array([374.0, 418.0, 462.0, 506.0, 550.0, 594.0, 638.0, 682.0, 726.0, 770.0, 814.0, 858.0, 902.0, 946.0,
                  990.0, 1034.0])
surfaces_USE = slice(1, 15)
surfaces_HALF = 22.0


def surfaces_nk(name):
    txt = open(os.path.join(surfaces_ROOT, "nk", name)).read()
    block = txt.split("data: |", 1)[1]
    rows = []
    for line in block.splitlines():
        p = line.split()
        if len(p) != 3:
            if rows and line.strip() and not re.match(r"^[\d.eE+-]", line.strip()):
                break
            continue
        try:
            rows.append([float(x) for x in p])
        except ValueError:
            break
    a = np.array(rows)
    return a[:, 0] * 1000.0, a[:, 1], a[:, 2]


def surfaces_band_mean(lam, refl):
    out = []
    for c in surfaces_BANDS:
        x = np.linspace(c - surfaces_HALF, c + surfaces_HALF, 23)
        out.append(float(np.mean(np.interp(x, lam, refl))))
    return np.array(out)


def surfaces_metal(name):
    lam, n, k = surfaces_nk(name)
    r = ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)
    return surfaces_band_mean(lam, r)


def surfaces_solar_cell(d_nm=75.0, n_ar=2.0):
    """Normal-incidence reflectance of silicon under a non-absorbing antireflection layer."""
    lam, n, k = surfaces_nk("Si_Green.yml")
    ns = n + 1j * k
    r01 = (1 - n_ar) / (1 + n_ar)
    r12 = (n_ar - ns) / (n_ar + ns)
    delta = 2 * math.pi * n_ar * d_nm / lam
    r = (r01 + r12 * np.exp(2j * delta)) / (1 + r01 * r12 * np.exp(2j * delta))
    return surfaces_band_mean(lam, np.abs(r) ** 2)


def surfaces_load():
    spec = json.load(open(os.path.join(surfaces_ROOT, "reflectance.json")))
    keep = {}
    for k, d in spec.items():
        r, e, f = np.array(d["r"], float), np.array(d["e"], float), np.array(d["f"], int)
        if len(r) != 16:
            continue
        ok = (f == 0) & np.isfinite(r) & np.isfinite(e) & (e > 0)
        ok[0] = ok[15] = False
        if ok[surfaces_USE].sum() < 12 or not ok[2:8].all():
            continue
        keep[k] = dict(name=d["name"], nsp=d["nsp"], r=r, e=e, ok=ok)
    return keep


def surfaces_wfit(A, y, w):
    """Weighted least squares: coefficients and chi-square."""
    Aw = A * w[:, None]
    c, *_ = np.linalg.lstsq(Aw, y * w, rcond=None)
    res = (y - A @ c) * w
    return c, float(res @ res)


surfaces_DCHI = 25.0


def surfaces_nested(y, e, ok, mu, pcs, t):
    """Asteroid-model chi-square, its drop with a non-negative template added, and the template's 550 nm share."""
    w = 1.0 / np.sqrt(e ** 2 + (0.01 * y) ** 2)
    A0 = np.c_[mu[ok], pcs.T[ok]]
    _, chi0 = surfaces_wfit(A0, y, w)
    tt = (t / t[4])[surfaces_USE][ok]
    c, chi1 = surfaces_wfit(np.c_[A0, tt], y, w)
    if c[-1] <= 0:
        return chi0, 0.0, 0.0
    j = int(np.argmin(np.abs(surfaces_BANDS[surfaces_USE][ok] - 550)))
    return chi0, chi0 - chi1, float(c[-1] * tt[j] / max(y[j], 1e-6))


def surfaces_edge_ok(r, e, name):
    """True when the template's absorption edge spans the bands either side, not a one-band notch."""
    if name == "gold":
        return bool(r[2] < r[3] < r[4] < r[5] and r[1] <= r[2] + 2 * math.hypot(e[1], e[2]))
    if name == "copper":
        return bool(r[4] < r[5] < r[6] and r[3] <= r[4] + 2 * math.hypot(e[3], e[4]))
    return True


def surfaces_inject(R, E, OK, mu, pcs, T, n=3000):
    """Recovery of each surface covering a fraction f of the 550 nm light in real spectra."""
    rng = np.random.default_rng(3)
    idx = rng.choice(len(R), n, replace=False)
    for name in T:
        t = T[name] / T[name][4]
        line = []
        for f in (0.05, 0.1, 0.2, 0.3, 0.5):
            hit = 0
            for i in idx:
                ok = OK[i][surfaces_USE]
                rr = (1 - f) * R[i] + f * t
                _, d, _ = surfaces_nested(rr[surfaces_USE][ok], E[i][surfaces_USE][ok], ok, mu, pcs, T[name])
                hit += d > surfaces_DCHI and surfaces_edge_ok(rr, E[i], name)
            line.append("%.0f%% %.2f" % (100 * f, hit / float(n)))
        print("  %s recovered at a fraction of the 550 nm light of %s" % (name, ", ".join(line)), flush=True)


def surfaces_taxonomy(n):
    """SsODNet taxonomy and albedo of a numbered asteroid."""
    b = core.get("https://ssp.imcce.fr/webservices/ssodnet/api/ssocard/%d" % n, timeout=60, retries=2, soft=True)
    try:
        ph = json.loads(b)["parameters"]["physical"]
        c = ph.get("taxonomy", {}).get("class")
        return (c.get("value") if isinstance(c, dict) else c), ph.get("albedo", {}).get("value")
    except Exception:
        return None, None


def surfaces_main():
    S = surfaces_load()
    keys = list(S)
    print("%d of 60518 spectra with 12 or more good bands between 418 and 990 nm" % len(keys), flush=True)
    R = np.array([S[k]["r"] for k in keys])
    E = np.array([S[k]["e"] for k in keys])
    OK = np.array([S[k]["ok"] for k in keys])
    snr = np.array([np.median(R[i, OK[i]] / E[i, OK[i]]) for i in range(len(keys))])
    best = np.argsort(-snr)[:5000]
    sub = R[best][:, surfaces_USE]
    sub = sub[np.all(OK[best][:, surfaces_USE], axis=1)]
    mu = sub.mean(0)
    u, s, vt = np.linalg.svd(sub - mu, full_matrices=False)
    pcs = vt[:4]
    print("asteroid model: mean and 4 components from %d spectra explain %.4f of the variance"
          % (len(sub), (s[:4] ** 2).sum() / (s ** 2).sum()), flush=True)
    T = {"gold": surfaces_metal("Au_Johnson.yml"), "copper": surfaces_metal("Cu_Johnson.yml"), "silver": surfaces_metal("Ag_Johnson.yml"),
         "aluminium": surfaces_metal("Al_Rakic.yml"), "solar cell": surfaces_solar_cell()}
    for name, t in T.items():
        print("  %-10s normalised at 550 nm: %s" % (name, " ".join("%.2f" % x for x in (t / t[4])[surfaces_USE])))
    surfaces_inject(R, E, OK, mu, pcs, T)
    out = []
    for i, k in enumerate(keys):
        ok = OK[i][surfaces_USE]
        y, e = R[i][surfaces_USE][ok], E[i][surfaces_USE][ok]
        row = dict(mpc=int(k), name=S[k]["name"], nsp=S[k]["nsp"], snr=float(snr[i]), nb=int(ok.sum()))
        for name, t in T.items():
            chi0, d, fr = surfaces_nested(y, e, ok, mu, pcs, t)
            row["chi_ast"] = chi0 / max(ok.sum() - 5, 1)
            row["d_" + name], row["frac_" + name] = d, fr
            row["edge_" + name] = surfaces_edge_ok(R[i], E[i], name)
        out.append(row)
    json.dump(out, open(os.path.join(surfaces_HERE, "sso_metals.json"), "w"))
    ca = np.array([r["chi_ast"] for r in out])
    print("asteroid model reduced chi-square: median %.2f, 99th percentile %.2f, 99.9th %.2f"
          % (np.median(ca), np.percentile(ca, 99), np.percentile(ca, 99.9)))
    for name in T:
        d = np.array([r["d_" + name] for r in out])
        hits = [r for r in out if r["d_" + name] > surfaces_DCHI and r["frac_" + name] > 0.05]
        kept = [r for r in hits if r["edge_" + name]]
        print("%s: chi-square drop above %.0f in %d spectra (99.9th percentile of the drop %.1f), %d with the edge's "
              "shape" % (name, surfaces_DCHI, len(hits), np.percentile(d, 99.9), len(kept)), flush=True)
        for r in sorted(kept, key=lambda r: -r["d_" + name])[:10]:
            tx, al = surfaces_taxonomy(r["mpc"])
            r["taxonomy"], r["albedo"] = tx, al
            print("    (%d) %s: %d epoch spectra, S/N %.0f; drop %.1f, %.0f per cent of the 550 nm light; asteroid "
                  "model %.1f; SsODNet taxonomy %s, albedo %s" % (r["mpc"], r["name"], r["nsp"], r["snr"],
                                                                  r["d_" + name], 100 * r["frac_" + name],
                                                                  r["chi_ast"], tx, al), flush=True)
    json.dump(out, open(os.path.join(surfaces_HERE, "sso_metals.json"), "w"))


def surfaces_record():
    """Record one row per object: model fit and solar-cell and gold drops."""
    out = json.load(open(os.path.join(surfaces_HERE, "sso_metals.json")))
    core.record_search(
        "gaia_sso_surfaces", "Artificial surfaces among the asteroids, Gaia DR3 reflectance spectra",
        archive="CDS I/359/ssor (gaiadr3.sso_reflectance_spectrum); refractiveindex.info optical constants; SsODNet",
        release="Gaia DR3", selection="%d of 60518 Solar System objects with 12 or more good bands between 418 and "
                                      "990 nm" % len(out),
        code="seti.py solar_system surfaces", params=dict(bands="418-990 nm, flag 0", model="mean and 4 principal components",
                                          dchi=surfaces_DCHI, templates=["gold", "copper", "silver", "aluminium",
                                                                "silicon solar cell with 75 nm nitride"]),
        n_targets=len(out),
        null_summary="no solar-cell surface; chi-square drops above %.0f for metal templates only on V, A and S types "
                     "that the four-component model fits badly" % surfaces_DCHI,
        sensitivity="solar cell recovered in 41, 88 and 100 per cent of 3000 injections at 5, 10 and 20 per cent of "
                    "the 550 nm light; gold 19 and copper 9 per cent at 50; silver and aluminium never",
        result="no artificial surface", ledger="Artificial surfaces among the asteroids")
    core.clear_targets("gaia_sso_surfaces")
    core.record_targets("gaia_sso_surfaces", [
        dict(target="mpc:%d" % r["mpc"], stat=float(r["chi_ast"]), unit="reduced chi-square of the asteroid model",
             note="solar cell drop %.1f, gold drop %.1f" % (r["d_solar cell"], r["d_gold"])) for r in out])


def surfaces_cli():
    quiet_warnings()
    if sys.argv[1:] == ["record"]:
        surfaces_record()
    else:
        surfaces_main()


# --------------------------------------------------------------------------------------------------------------

radar_albedo_DOC = r"""Metallic radar reflectors: asteroids with OC radar albedo above 0.3 in the PDS radar compilation, listed with SC/OC and taxonomy.

    python -I seti.py solar_system radar_albedo"""

radar_albedo_ROOT = os.environ.get("RADAR_ROOT", core.data("radar"))
radar_albedo_URL = "https://sbnarchive.psi.edu/pds4/non_mission/compil.ast.radar-properties.zip"
radar_albedo_SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=%s&phys-par=1"


def radar_albedo_val(s):
    try:
        v = float(s.strip())
        return None if v >= 9.98 else v
    except ValueError:
        return None


def radar_albedo_main():
    path = os.path.join(radar_albedo_ROOT, "radar_pds4.zip")
    if not os.path.exists(path):
        os.makedirs(radar_albedo_ROOT, exist_ok=True)
        open(path, "wb").write(core.get(radar_albedo_URL, timeout=120, retries=3))
    t = zipfile.ZipFile(path).read("compil.ast.radar-properties/data/radar.tab").decode().splitlines()
    # fixed columns: number 1-6, name 8-24, designation 26-35, SC/OC 59-62, albedo 69-73
    rows = [dict(num=ln[0:6].strip(), name=ln[7:24].strip(), prov=ln[25:35].strip(), sigma=radar_albedo_val(ln[40:48]),
                 muc=radar_albedo_val(ln[58:62]), alb=radar_albedo_val(ln[68:73]), albe=radar_albedo_val(ln[74:79]), mhz=ln[82:86].strip(), year=ln[87:91].strip())
            for ln in t if ln.strip()]
    objs = {}
    for r in rows:
        k = r["num"] if r["num"] not in ("0", "-", "") else r["prov"]
        objs.setdefault(k, []).append(r)
    print("%d radar measurements of %d objects; %d with a radar albedo" % (len(rows), len(objs),
                                                                        sum(1 for r in rows if r["alb"] is not None)))
    high = []
    for k, rs in objs.items():
        a = [r for r in rs if r["alb"] is not None]
        if a and max(r["alb"] for r in a) > 0.3:
            best = max(a, key=lambda r: r["alb"])
            b = core.get(radar_albedo_SBDB % k.replace(" ", "%20"), timeout=60, retries=2, soft=True)
            cls = {}
            if b:
                for p in json.loads(b).get("phys_par", []):
                    if p["name"] in ("spec_T", "spec_B"):
                        cls[p["name"]] = p["value"]
            high.append(dict(obj=k, name=best["name"] or best["prov"], alb=best["alb"], albe=best["albe"], muc=best["muc"],
                             tholen=cls.get("spec_T"), smass=cls.get("spec_B")))
    print("radar albedo above 0.3:")
    for h in sorted(high, key=lambda h: -h["alb"]):
        print("  %-8s %-12s albedo %.2f +- %s, SC/OC %s, Tholen %s, SMASS %s"
              % (h["obj"], h["name"], h["alb"], h["albe"], h["muc"], h["tholen"], h["smass"]))
    spec = [h for h in high if h["alb"] > 0.8 and (h["muc"] is not None and h["muc"] < 0.1)]
    print("specular metallic reflectors (albedo above 0.8, SC/OC below 0.1): %d" % len(spec))
    json.dump(dict(objects=len(objs), high=high), open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    "radar_albedo.json"), "w"), indent=1)
    out = []
    for k, rs in objs.items():
        a = [r["alb"] for r in rs if r["alb"] is not None]
        h = next((x for x in high if x["obj"] == k), None)
        out.append(dict(target="mpc:" + k, stat=max(a) if a else None, unit="OC radar albedo",
                        vet=("metal-rich or enstatite class (Tholen %s, SMASS %s)" % (h["tholen"], h["smass"])) if h else None,
                        note="%d detections" % len(rs)))
    core.record_search("asteroid_radar_albedo", "Asteroid radar", archive="PDS SBN compil.ast.radar-properties V1.0",
                           release="2020-03-02", selection="every published asteroid radar detection",
                           code=os.path.abspath(__file__), entry="radar_albedo", n_targets=len(out),
                           result="the nine albedos above 0.3 are M or X class bodies, the radar-metallic 1986 DA, E-type "
                                  "Nereus and 1998 WT24, and Sq-type Bacchus at 0.33 +- 0.25; no specular reflector",
                           ledger="Asteroid radar")
    core.record_targets("asteroid_radar_albedo", out)
    print("recorded %d objects" % len(out))


def radar_albedo_cli():
    radar_albedo_main()


# --------------------------------------------------------------------------------------------------------------

hot_DOC = r"""Internally heated asteroids: NEOWISE NEATM beaming parameter with eta plus 2 sigma below 0.6 at two or more free-eta epochs.

    python -I seti.py solar_system hot [record] [--eta-floor 0.6] [--sigma 2]"""


def hot_rows():
    p = core.data("neowisesbpropv2.json")
    if os.path.exists(p):
        return json.load(open(p))
    import csv
    import io
    u = "https://irsa.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(dict(QUERY="SELECT * FROM neowisesbpropv2", FORMAT="csv"))
    rows = list(csv.DictReader(io.StringIO(core.get(u, timeout=900).decode())))
    json.dump(rows, open(p, "w"))
    return rows


def hot_main(record=False, eta_floor=0.6, sigma=2.0):
    rows = hot_rows()
    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")
    fitted = [r for r in rows if "B" in (r["fit_code"] or "")]
    eta = np.array([f(r["beaming_param"]) for r in fitted])
    err = np.array([f(r["beaming_param_err"]) for r in fitted])
    low = [(r, e, s) for r, e, s in zip(fitted, eta, err) if np.isfinite(e) and np.isfinite(s) and e + sigma * s < eta_floor]
    by = {}
    for r in fitted:
        by.setdefault(r["asteroid_number"] or r["prov_desig"] or r["comet_desig"] or r["mpc_packed_name"], []).append(r)
    persistent = [k for k, lst in by.items() if len(lst) >= 2 and all(
        np.isfinite(f(r["beaming_param"])) and np.isfinite(f(r["beaming_param_err"]))
        and f(r["beaming_param"]) + sigma * f(r["beaming_param_err"]) < eta_floor for r in lst)]
    p1, p5, p50 = np.nanpercentile(eta, [1, 5, 50])
    print("%d fits, %d with eta free (%d objects); eta 1st, 5th and 50th percentiles %.2f, %.2f, %.2f; %d fits with eta + %g "
          "sigma below %g; %d objects below it at every fitted epoch" % (len(rows), len(fitted), len(by), p1, p5, p50, len(low), sigma, eta_floor,
                                                                      len(persistent)))
    for r, e, s in sorted(low, key=lambda x: x[1]):
        print("  %-10s %s %s: eta %.2f +- %.2f, D %.2f km, %s, W3/W4 points %s/%s, %d fitted epochs"
              % (r["asteroid_number"] or r["prov_desig"], r["type"], r["mean_jd"], e, s, f(r["diameter"]), r["fit_code"],
                 r["n_w3"], r["n_w4"], len(by[r["asteroid_number"] or r["prov_desig"] or r["comet_desig"] or r["mpc_packed_name"]])))
    if not record:
        return
    core.record_search(
        "neowise_hot_bodies", "Bodies hotter than sunlight allows", archive="IRSA neowisesbpropv2 (NEOWISE diameters and albedos v2)",
        selection="every NEOWISE thermal fit with the beaming parameter free", code=os.path.abspath(__file__), entry="hot",
        n_targets=len(by), params=dict(eta_floor=eta_floor, sigma=sigma),
        null_summary="the population's own beaming distribution: 1st percentile %.2f" % p1,
        sensitivity="internal power raising the dayside beyond the eta = %g limit of a rough sunlit surface" % eta_floor,
        ledger="Bodies hotter than sunlight allows",
        result="%d of %d fits fall below eta %g at %g sigma, each at a single epoch, the lowest pinned at the fit's floor of 0.3; "
               "%d objects stay below at every fitted epoch" % (len(low), len(fitted), eta_floor, sigma, len(persistent)))
    out, seen = [], set()
    for r, e, s in low:
        k = r["asteroid_number"] or r["prov_desig"] or r["comet_desig"] or r["mpc_packed_name"]
        if k in seen:
            continue
        seen.add(k)
        others = [f(x["beaming_param"]) for x in by[k] if x is not r]
        out.append(dict(target="mpc:%s" % k, stat=e, unit="lowest fitted beaming parameter",
                        vet=("pinned at the fit's floor" if s == 0 else
                             "other epochs at eta %s" % ", ".join("%.2f" % v for v in others) if others else "one fitted epoch"),
                        note="%s, D %.2f km, epoch JD %s" % (r["type"], f(r["diameter"]), r["mean_jd"])))
    core.clear_targets("neowise_hot_bodies")
    core.record_targets("neowise_hot_bodies", out)
    print("recorded")


orbits_DOC = r"""Near-Earth objects without a delivery route (Granvik et al. 2018): retrograde at low eccentricity, near-circular inside 0.75 AU above 25 deg, or above 60 deg below e 0.25.

    python -I seti.py solar_system orbits [record] [--e-low 0.3]"""


def orbits_query(cdata):
    u = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?" + urllib.parse.urlencode(
        {"fields": "full_name,a,e,i,q,n_obs_used,data_arc,condition_code,H,kind", "sb-group": "neo", "sb-cdata": json.dumps(cdata)})
    return json.loads(core.get(u, timeout=300))


def orbits_main(record=False, e_low=0.3):
    cells = (("retrograde", {"AND": ["i|GT|90"]}), ("inside Venus, near-circular, above 25 deg", {"AND": ["a|LT|0.75", "e|LT|0.25", "i|GT|25"]}),
             ("above 60 deg, e below 0.25", {"AND": ["i|GT|60", "e|LT|0.25"]}))
    found = {}
    for name, cd in cells:
        r = orbits_query(cd)
        found[name] = r.get("data", [])
        es = [float(x[2]) for x in found[name]]
        print("%s: %d objects, eccentricity %s" % (name, len(found[name]), ("%.2f-%.2f" % (min(es), max(es))) if es else "-"))
        for x in found[name]:
            print("  %s a %s e %s i %s arc %s d cond %s" % (x[0].strip(), x[1], x[2], x[3], x[6], x[7]))
    retro_low = [x for x in found["retrograde"] if float(x[2]) < e_low]
    print("retrograde with e below %g: %d" % (e_low, len(retro_low)))
    if not record:
        return
    core.record_search(
        "neo_unreachable_orbits", "NEOs on unreachable orbits", archive="JPL Small-Body Database",
        selection="near-Earth objects in the zero-density cells of the delivery model", code=os.path.abspath(__file__), entry="orbits",
        n_targets=sum(len(v) for v in found.values()), params=dict(e_low=e_low), null_summary="the Granvik et al. (2018) source model's zero-density cells",
        ledger="NEOs on unreachable orbits",
        result="%d retrograde NEOs, the lowest eccentricity %.2f (Halley-type comets and comet-like asteroids); %d retrograde below e %g; "
               "%d near-circular inside 0.75 AU above 25 deg, within the Atira population; %d above 60 deg with e below 0.25, "
               "reachable through Kozai cycles" % (len(found["retrograde"]), min([float(x[2]) for x in found["retrograde"]] or [float("nan")]), len(retro_low), e_low,
                                                   len(found["inside Venus, near-circular, above 25 deg"]), len(found["above 60 deg, e below 0.25"])))
    core.clear_targets("neo_unreachable_orbits")
    core.record_targets("neo_unreachable_orbits", [dict(target="sbdb:%s" % x[0].strip(), stat=float(x[2]), unit="eccentricity",
                                                            note="%s; a %s, i %s, arc %s d" % (name, x[1], x[3], x[6]))
                                                       for name, v in found.items() for x in v])
    print("recorded")


yarkovsky_DOC = r"""Yarkovsky drifts of the wrong sign for every DAMIT pole, or beyond the efficiency sunlight allows at rho 500 kg/m^3.

    python -I seti.py solar_system yarkovsky fetch      (SBDB objects with a transverse term; DAMIT's asteroid and model tables)
    python -I seti.py solar_system yarkovsky analyse
    python -I seti.py solar_system yarkovsky record"""

yarkovsky_ROOT = os.environ.get("YARKOVSKY_ROOT", core.data("yarkovsky"))
yarkovsky_FIELDS = ("full_name,pdes,kind,class,H,diameter,albedo,rot_per,A1,A1_sigma,A2,A2_sigma,A3,A3_sigma,DT,e,q,a,i,om,w,"
                    "n_obs_used,data_arc,condition_code")
yarkovsky_DAMIT = "https://astro.troja.mff.cuni.cz/projects/damit/exports/table/"
yarkovsky_RHO = 500.0
yarkovsky_COS = 0.3
# Seligman et al. (2024) dark comets, active asteroid 362P and comet 1998 SH2 (Farnocchia et al. 2026)
yarkovsky_KNOWN = {"2003 RM": "dark comet", "2001 ME1": "dark comet", "2005 UY6": "dark comet", "1998 FR11": "dark comet",
                   "2012 UR158": "dark comet", "1998 KY26": "dark comet", "2005 VL1": "dark comet", "2016 NJ33": "dark comet",
                   "2010 VL65": "dark comet", "2010 RF12": "dark comet", "2006 RH120": "dark comet", "2016 GW221": "dark comet",
                   "2013 XY20": "dark comet", "2013 BA74": "dark comet", "2008 GO98": "active comet 362P",
                   "1998 SH2": "active comet, a weak tail imaged in 2025 (Farnocchia et al. 2026)"}
# readings of the darkcomet refits, by primary designation
yarkovsky_REFIT = {"883607": "dark-comet candidate: refitted in darkcomet_2016_ta56",
                   "875163": "active comet, a weak tail imaged in 2025 (Farnocchia et al. 2026), refitted in darkcomet_1998_sh2",
                   "428209": "no drift needed: the gravity-only orbit fits all four apparitions as well (darkcomet_2006_vc)"}


def yarkovsky_fetch():
    os.makedirs(yarkovsky_ROOT, exist_ok=True)
    u = nga_API + "?" + urllib.parse.urlencode({"fields": yarkovsky_FIELDS, "sb-kind": "a", "sb-cdata": '{"AND":["A2|DF"]}',
                                                "full-prec": "true"})
    d = json.loads(core.get(u, timeout=300))
    json.dump(d, open(os.path.join(yarkovsky_ROOT, "sbdb_A2.json"), "w"))
    print("%d asteroids with a transverse non-gravitational term" % d["count"])
    for t in ("asteroids", "asteroid_models"):
        b = core.get(yarkovsky_DAMIT + t, timeout=600)
        open(os.path.join(yarkovsky_ROOT, "damit_%s.csv" % t), "wb").write(b if isinstance(b, bytes) else b.encode("utf-8"))
        print("DAMIT %s: %d bytes" % (t, len(b)))


def yarkovsky_poles():
    """DAMIT poles (lambda, beta deg; period h) keyed by number and designation."""
    import csv
    ids = {}
    for r in csv.DictReader(open(os.path.join(yarkovsky_ROOT, "damit_asteroids.csv"), encoding="utf-8-sig")):
        keys = [k for k in (r["number"], (r["designation"] or "").strip()) if k]
        ids[r["id"]] = keys
    poles = {}
    for r in csv.DictReader(open(os.path.join(yarkovsky_ROOT, "damit_asteroid_models.csv"), encoding="utf-8-sig")):
        lam, bet = num(r["lambda"]), num(r["beta"])
        if lam is None or bet is None:
            continue
        for k in ids.get(r["asteroid_id"], []):
            poles.setdefault(k, []).append(dict(model=r["id"], lam=lam, beta=bet, period=num(r["period"])))
    return poles


def yarkovsky_cos_obliquity(lam, beta, inc, node):
    lam, beta, inc, node = map(math.radians, (lam, beta, inc, node))
    n = (math.sin(inc) * math.sin(node), -math.sin(inc) * math.cos(node), math.cos(inc))
    s = (math.cos(beta) * math.cos(lam), math.cos(beta) * math.sin(lam), math.sin(beta))
    return sum(x * y for x, y in zip(n, s))


def yarkovsky_analyse():
    d = json.load(open(os.path.join(yarkovsky_ROOT, "sbdb_A2.json")))
    rows = [dict(zip(d["fields"], r)) for r in d["data"]]
    poles = yarkovsky_poles()
    out = []
    for r in rows:
        a2, s2 = num(r["A2"]), num(r["A2_sigma"])
        if a2 is None or not s2:
            continue
        if num(r["diameter"]):
            dmin, dsrc = num(r["diameter"]) * 1000.0, "measured"
        elif num(r["H"]) is not None:
            dmin, dsrc = 1329.0e3 / math.sqrt(nga_ALBEDO_MAX) * 10 ** (-num(r["H"]) / 5.0), "H at albedo 0.5"
        else:
            dmin, dsrc = None, None
        amax = nga_S_OVER_C * 1.5 / (dmin * yarkovsky_RHO) if dmin else None
        xi = abs(a2) * nga_AU_D2 / amax if amax else None
        # typical NEA: 2000 kg/m^3, albedo 0.14 unless D is measured
        dtyp = (num(r["diameter"]) * 1000.0 if num(r["diameter"]) else
                1329.0e3 / math.sqrt(0.14) * 10 ** (-num(r["H"]) / 5.0) if num(r["H"]) is not None else None)
        xi_typ = abs(a2) * nga_AU_D2 / (nga_S_OVER_C * 1.5 / (dtyp * 2000.0)) if dtyp else None
        a, e, inc = num(r["a"]), num(r["e"]), num(r["i"])
        tj = (5.2026 / a + 2 * math.cos(math.radians(inc)) * math.sqrt(a / 5.2026 * (1 - e * e))
              if a and a > 0 and e is not None and e < 1 and inc is not None else None)
        known = next((v for k, v in yarkovsky_KNOWN.items() if "(%s)" % k in r["full_name"] or r["pdes"] == k), None)
        ps = poles.get(r["pdes"], [])
        cos = [yarkovsky_cos_obliquity(p["lam"], p["beta"], num(r["i"]), num(r["om"])) for p in ps]
        snr = a2 / s2
        verdict = None
        if cos and abs(snr) >= 3:
            if snr > 0 and max(cos) < -yarkovsky_COS:
                verdict = "outward on a retrograde spin"
            elif snr < 0 and min(cos) > yarkovsky_COS:
                verdict = "inward on a prograde spin"
            elif snr > 0 and min(cos) > 0 or snr < 0 and max(cos) < 0:
                verdict = "agrees"
            elif snr < 0 and all(abs(c) <= yarkovsky_COS for c in cos):
                verdict = "agrees (seasonal)"
            else:
                verdict = "pole solutions disagree"
        a1, s1 = num(r["A1"]), num(r["A1_sigma"])
        a3, s3 = num(r["A3"]), num(r["A3_sigma"])
        out.append(dict(name=r["full_name"].strip(), pdes=r["pdes"], cls=r["class"], H=num(r["H"]), d_min_m=dmin, d_src=dsrc,
                        A2=a2, A2_snr=snr, xi=xi, xi_typ=xi_typ, t_j=tj, q=num(r["q"]), known=known, cos_obl=cos,
                        periods=[p["period"] for p in ps], rot_per=num(r["rot_per"]),
                        A1_snr=a1 / s1 if a1 is not None and s1 else None, A3_snr=a3 / s3 if a3 is not None and s3 else None,
                        verdict=verdict, arc=r["data_arc"], n_obs=r["n_obs_used"], cond=r["condition_code"]))
    sig = [x for x in out if abs(x["A2_snr"]) >= 3]
    tested = [x for x in sig if x["verdict"]]
    print("%d asteroids with A2; %d at 3 sigma or more; %d of these with DAMIT poles" % (len(out), len(sig), len(tested)))
    for v in ("agrees", "agrees (seasonal)", "pole solutions disagree", "inward on a prograde spin", "outward on a retrograde spin"):
        xs = [x for x in tested if x["verdict"] == v]
        print("  %s: %d" % (v, len(xs)))
        if v not in ("agrees", "agrees (seasonal)"):
            for x in xs:
                print("    %-30s %-4s A2 %+.2e (%+.1f sigma) cos obliquity %s; P %s h; xi %.3f; arc %s d"
                      % (x["name"][:30], x["cls"], x["A2"], x["A2_snr"], ", ".join("%+.2f" % c for c in x["cos_obl"]),
                         ", ".join("%.2f" % p for p in x["periods"] if p), x["xi"] or 0, x["arc"]))
    big = sorted([x for x in sig if x["xi"] and x["xi"] > 1], key=lambda x: -x["xi"])
    print("transverse drifts beyond all absorbed sunlight (xi > 1 at rho %d, %s): %d" % (yarkovsky_RHO, "albedo 0.5 or measured D",
                                                                                         len(big)))
    for x in big:
        print("  %-30s %-4s H %5.2f D>=%7.1f m (%s) A2 %+.2e (%+.1f sigma) xi %.2f; A1 %s sigma, A3 %s sigma; T_J %s, q %.2f; "
              "arc %s d, %s obs; %s"
              % (x["name"][:30], x["cls"], x["H"] or 0, x["d_min_m"], x["d_src"], x["A2"], x["A2_snr"], x["xi"],
                 "%.1f" % x["A1_snr"] if x["A1_snr"] is not None else "-", "%.1f" % x["A3_snr"] if x["A3_snr"] is not None else "-",
                 "%.2f" % x["t_j"] if x["t_j"] else "-", x["q"], x["arc"], x["n_obs"], x["known"] or "unpublished"))
    typ = sorted([x for x in sig if x["xi_typ"] and x["xi_typ"] > 1], key=lambda x: -x["xi_typ"])
    print("beyond all absorbed sunlight at 2000 kg/m^3 and albedo 0.14 unless D is measured: %d, %d of them published"
          % (len(typ), sum(1 for x in typ if x["known"])))
    for x in typ:
        print("  %-30s %-4s xi %8.2f A2 %+.2e (%+.1f sigma) T_J %s q %.2f arc %s d; %s"
              % (x["name"][:30], x["cls"], x["xi_typ"], x["A2"], x["A2_snr"], "%.2f" % x["t_j"] if x["t_j"] else "-", x["q"],
                 x["arc"], x["known"] or "unpublished"))
    json.dump(out, open(os.path.join(core.RESULTS, "yarkovsky_sign.json"), "w"), indent=1)
    return out


def yarkovsky_record():
    out = json.load(open(os.path.join(core.RESULTS, "yarkovsky_sign.json")))
    for x in out:
        # published status from the current list, not the analysis's
        x["known"] = next((v for k, v in yarkovsky_KNOWN.items() if "(%s)" % k in x["name"] or x["pdes"] == k), None)
    sig = [x for x in out if abs(x["A2_snr"]) >= 3]
    tested = [x for x in sig if x["verdict"]]
    count = {v: sum(1 for x in tested if x["verdict"] == v) for v in
             ("agrees", "agrees (seasonal)", "pole solutions disagree", "inward on a prograde spin", "outward on a retrograde spin")}
    big = [x for x in sig if x["xi"] and x["xi"] > 1]
    typ = [x for x in sig if x["xi_typ"] and x["xi_typ"] > 1]

    def vet(x):
        if x["pdes"] in yarkovsky_REFIT:
            return yarkovsky_REFIT[x["pdes"]]
        if x["known"]:
            return "published %s" % x["known"] if x["known"] == "dark comet" else x["known"]
        if abs(x["A2_snr"]) >= 3 and x["xi_typ"] and x["xi_typ"] > 1:
            if x["t_j"] and x["t_j"] < 3.05:
                return "unpublished dark-comet candidate: Jupiter-family orbit (T_J %.2f, q %.2f)" % (x["t_j"], x["q"])
            return "open: transverse drift beyond sunlight on an asteroidal orbit (T_J %.2f), A2 at %.1f sigma" % (x["t_j"], x["A2_snr"])
        return x["verdict"]

    def names(xs):
        return ", ".join(re.sub(r"^\d+ ", "", x["name"]).strip("()") for x in xs)
    core.record_search(
        "yarkovsky_sign", "Yarkovsky sign test", archive="JPL Small-Body Database query API; DAMIT",
        release="current solutions; DAMIT table export", selection="every asteroid with a transverse non-gravitational term A2",
        code=os.path.abspath(__file__), entry="yarkovsky", n_targets=len(out),
        params=dict(a2_snr=3, cos_obliquity=yarkovsky_COS, rho=yarkovsky_RHO, albedo=nga_ALBEDO_MAX),
        null_summary="the thermal model's signs: outward needs a prograde spin, inward a retrograde or near-90 deg one",
        sensitivity="A2 at 3 sigma against every DAMIT pole solution; xi against all absorbed sunlight at rho 500, albedo 0.5",
        ledger="Yarkovsky sign test",
        result="%d drifts at 3 sigma with DAMIT poles: %d agree, %d seasonal, %d with pole solutions that disagree, %d inward on "
               "a prograde spin, %d outward on a retrograde spin; beyond all absorbed sunlight at 500 kg/m^3 and albedo 0.5: %d "
               "(published dark comets or active: %s; unpublished: %s); at 2000 kg/m^3 and albedo 0.14: %d, unpublished %s"
               % (len(tested), count["agrees"], count["agrees (seasonal)"], count["pole solutions disagree"],
                  count["inward on a prograde spin"], count["outward on a retrograde spin"], len(big),
                  names([x for x in big if x["known"]]), names([x for x in big if not x["known"]]) or "none", len(typ),
                  names([x for x in typ if not x["known"]]) or "none"))
    core.clear_targets("yarkovsky_sign")
    core.record_targets("yarkovsky_sign", [dict(
        target="mpc:" + x["pdes"], stat=x["xi_typ"], unit="transverse drift / all absorbed sunlight (2000 kg/m^3, albedo 0.14)",
        upper_limit=None, vet=vet(x),
        note="%s; A2 %+.2e at %+.1f sigma; xi %s at 500 kg/m^3, albedo 0.5; cos obliquity %s"
             % (x["name"], x["A2"], x["A2_snr"], "%.3f" % x["xi"] if x["xi"] else "-",
                ", ".join("%+.2f" % c for c in x["cos_obl"]) or "no pole"))
        for x in out])
    print("recorded %d asteroids" % len(out))


def yarkovsky_cli():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "record"))
    a = p.parse_args()
    {"fetch": yarkovsky_fetch, "analyse": yarkovsky_analyse, "record": yarkovsky_record}[a.mode]()


meteors_DOC = r"""Coherent hyperbolic meteor streams in GMN orbits: sporadics sharing an incoming asymptote within 1 deg and 3 km/s, against error-process and speed-shuffled nulls.

    python -I seti.py solar_system meteors fetch
    python -I seti.py solar_system meteors analyse
    python -I seti.py solar_system meteors record
    python -I seti.py solar_system meteors locus          (the group's predicted radiants by date against the GMN table)
    python -I seti.py solar_system meteors locus_record"""

meteors_ROOT = os.environ.get("METEORS_ROOT", core.data("gmn"))
meteors_URL = "https://globalmeteornetwork.org/data/traj_summary_data/traj_summary_yearly_%d.txt"
meteors_YEARS = range(2018, 2027)
meteors_ISM = (259.0, 8.0, 26.0)


def meteors_fetch():
    os.makedirs(meteors_ROOT, exist_ok=True)
    for y in meteors_YEARS:
        p = os.path.join(meteors_ROOT, "traj_summary_yearly_%d.txt" % y)
        b = core.get(meteors_URL % y, timeout=900)
        open(p, "wb").write(b if isinstance(b, bytes) else b.encode("utf-8"))
        print("%d: %d MB" % (y, len(b) // 1000000), flush=True)


def meteors_load():
    """Sporadic meteors hyperbolic at 3 sigma with radiant, speed, v_inf and asymptote."""
    rows = []
    n_all = n_spor = 0
    for y in meteors_YEARS:
        p = os.path.join(meteors_ROOT, "traj_summary_yearly_%d.txt" % y)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            if line.startswith("#") or ";" not in line:
                continue
            c = [x.strip() for x in line.split(";")]
            if len(c) < 85:
                continue
            n_all += 1
            if c[3] != "-1":
                continue
            n_spor += 1
            e, se, a, qc = num(c[25]), num(c[26]), num(c[23]), num(c[80])
            if e is None or se is None or a is None or qc is None or e <= 1 or a >= 0 or (e - 1) < 3 * se or qc < 15:
                continue
            q, inc, w, node = num(c[37]), num(c[27]), num(c[29]), num(c[31])
            f = -math.acos(-1.0 / e)
            u = math.radians(w) + f
            om, i = math.radians(node), math.radians(inc)
            x = math.cos(om) * math.cos(u) - math.sin(om) * math.sin(u) * math.cos(i)
            yy = math.sin(om) * math.cos(u) + math.cos(om) * math.sin(u) * math.cos(i)
            z = math.sin(u) * math.sin(i)
            rows.append(dict(id=c[0], year=int(c[2][:4]), sol=num(c[5]), ra=num(c[7]), dec=num(c[9]), vg=num(c[15]),
                             e=e, se=se, q=q, vinf=29.7847 / math.sqrt(-a), x=x, y=yy, z=z, qc=qc, nst=num(c[84])))
    return rows, n_all, n_spor


def meteors_showers():
    """IAU MDC showers, established first: code, solar longitude, radiant, daily motion, Vg."""
    out = []
    for name, status in (("streamestablisheddata2026.txt", "established"), ("streamworkingdata2026.txt", "working")):
        p = os.path.join(meteors_ROOT, name)
        if not os.path.exists(p):
            b = core.get("https://www.ta3.sk/IAUC22DB/MDC2022/Etc/" + name, timeout=300)
            open(p, "wb").write(b if isinstance(b, bytes) else b.encode("latin-1"))
        for line in open(p, encoding="latin-1"):
            if not line.startswith('"'):
                continue
            c = [x.strip().strip('"').strip() for x in line.split("|")]
            if len(c) < 16:
                continue
            los, ra, de, vg = num(c[10]), num(c[11]), num(c[12]), num(c[15])
            if None in (los, ra, de, vg):
                continue
            out.append(dict(code=c[3], status=status, los=los, ra=ra, de=de, dra=num(c[13]) or 0.0, dde=num(c[14]) or 0.0, vg=vg))
    return out


def meteors_match(r, showers):
    """Shower within 8 deg of the drift-corrected radiant, 15 deg of solar longitude and 20 per cent in Vg, established first; or None."""
    best = None
    for s in showers:
        dl = (r["sol"] - s["los"] + 180) % 360 - 180
        if abs(dl) > 15 or abs(r["vg"] - s["vg"]) > 0.2 * s["vg"]:
            continue
        ra, de = math.radians(s["ra"] + s["dra"] * dl), math.radians(s["de"] + s["dde"] * dl)
        ra1, de1 = math.radians(r["ra"]), math.radians(r["dec"])
        d = math.degrees(math.acos(max(-1.0, min(1.0, math.sin(de) * math.sin(de1) + math.cos(de) * math.cos(de1) * math.cos(ra - ra1)))))
        if d < 8 and (best is None or (s["status"], d) < (best[0]["status"], best[1])):
            best = (s, d)
    return "%s (%s)" % (best[0]["code"], best[0]["status"]) if best else None


def meteors_pvalues(xyz, v):
    from scipy.spatial import cKDTree
    from scipy.stats import poisson
    t = cKDTree(xyz)
    c1 = 2 * math.sin(math.radians(0.5))
    ratio = (1 - math.cos(math.radians(1.0))) / (1 - math.cos(math.radians(10.0)))
    n1 = np.zeros(len(v), int)
    n10 = np.zeros(len(v), int)
    for k, js in enumerate(t.query_ball_point(xyz, 2 * math.sin(math.radians(5.0)))):
        js = np.array(js)
        js = js[(js != k) & (np.abs(v[js] - v[k]) < 3.0)]
        n10[k] = len(js)
        if len(js):
            n1[k] = int(np.sum(np.linalg.norm(xyz[js] - xyz[k], axis=1) < c1))
    lam = np.maximum(n10 * ratio, 1e-3)
    return n1, n10, poisson.sf(n1 - 1, lam)


def meteors_groups(xyz, v, sel):
    from scipy.spatial import cKDTree
    idx = np.where(sel)[0]
    if not len(idx):
        return []
    t = cKDTree(xyz[idx])
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in t.query_pairs(2 * math.sin(math.radians(0.5))):
        if abs(v[idx[a]] - v[idx[b]]) < 3.0:
            parent[find(a)] = find(b)
    groups = {}
    for k in range(len(idx)):
        groups.setdefault(find(k), []).append(int(idx[k]))
    return sorted(groups.values(), key=len, reverse=True)


def meteors_search(rows, label, seed=1):
    xyz = np.array([[r["x"], r["y"], r["z"]] for r in rows])
    v = np.array([r["vinf"] for r in rows])
    n1, n10, p = meteors_pvalues(xyz, v)
    rng = np.random.default_rng(seed)
    nulls = [int(np.sum(meteors_pvalues(xyz, rng.permutation(v))[2] < 1e-4)) for k in range(5)]
    print("%s: %d meteors; with a 1 deg, 3 km/s neighbourhood below p 1e-4: %d; speeds shuffled: %s"
          % (label, len(rows), int(np.sum(p < 1e-4)), nulls), flush=True)
    ism = np.array([math.cos(math.radians(meteors_ISM[1])) * math.cos(math.radians(meteors_ISM[0])),
                    math.cos(math.radians(meteors_ISM[1])) * math.sin(math.radians(meteors_ISM[0])), math.sin(math.radians(meteors_ISM[1]))])
    out = []
    for g in meteors_groups(xyz, v, p < 1e-4):
        m = xyz[g].mean(axis=0)
        m /= np.linalg.norm(m)
        lam, bet = math.degrees(math.atan2(m[1], m[0])) % 360, math.degrees(math.asin(m[2]))
        sh = [rows[k]["shower"] for k in g]
        top = max(set(sh), key=sh.count)
        out.append(dict(n=len(g), lam=lam, beta=bet, vinf=float(np.mean(v[g])), vinf_sd=float(np.std(v[g])),
                        years=sorted({rows[k]["year"] for k in g}), p_min=float(np.min(p[g])),
                        ism_sep=math.degrees(math.acos(max(-1.0, min(1.0, float(m @ ism))))),
                        shower=top if top and sh.count(top) * 2 > len(g) else None, showers=sh,
                        sol=[rows[k]["sol"] for k in g], radiant=[(rows[k]["ra"], rows[k]["dec"], rows[k]["vg"]) for k in g],
                        ids=[rows[k]["id"] for k in g], e=[rows[k]["e"] for k in g]))
    print("  %d groups; with three or more members from two or more years: %d"
          % (len(out), sum(1 for o in out if o["n"] >= 3 and len(o["years"]) >= 2)))
    for o in out[:30]:
        print("  n %3d asymptote %6.1f %+5.1f v_inf %5.1f+-%4.1f km/s p_min %.1e years %s; %.0f deg from the ISM inflow; "
              "shower %s; sol %s; geo %s"
              % (o["n"], o["lam"], o["beta"], o["vinf"], o["vinf_sd"], o["p_min"], ",".join(str(y) for y in o["years"]),
                 o["ism_sep"], o["shower"] or "none (%s)" % ", ".join(sorted({s or "-" for s in o["showers"]})),
                 ",".join("%.0f" % s for s in o["sol"][:6]), " ".join("(%.0f %+.0f %.0f)" % r for r in o["radiant"][:3])))
    return dict(n=len(rows), n_sig=int(np.sum(p < 1e-4)), nulls=nulls, groups=out)


meteors_MU = 0.01720209895 ** 2
meteors_KMS = 1.495978707e8 / 86400.0


def meteors_earth(jd):
    """Earth's heliocentric ecliptic position (au), low-precision solar theory."""
    T = (jd - 2451545.0) / 36525
    L = math.radians((280.46646 + 36000.76983 * T) % 360)
    M = math.radians((357.52911 + 35999.05029 * T) % 360)
    C = math.radians((1.914602 - 0.004817 * T) * math.sin(M) + 0.019993 * math.sin(2 * M))
    R = 1.000001018 * (1 - 0.016708634 ** 2) / (1 + 0.016708634 * math.cos(M + C))
    return np.array([math.cos(L + C + math.pi), math.sin(L + C + math.pi), 0.0]) * R


def meteors_asymptote(r, v):
    """Eccentricity and incoming asymptote unit vector of a heliocentric state (au, au/d); (e, None) if bound."""
    h = np.cross(r, v)
    ev = np.cross(v, h) / meteors_MU - r / np.linalg.norm(r)
    E = float(np.linalg.norm(ev))
    if E <= 1:
        return E, None
    P = ev / E
    Q = np.cross(h / np.linalg.norm(h), P)
    fi = math.acos(-1 / E)
    return E, math.cos(fi) * P - math.sin(fi) * Q


def meteors_null(rows_hyp, k_scale, seed, emin=0.7, bias=None):
    """Error-process null: elliptic sporadics with speeds perturbed by k_scale sigma, or scaled by a shared bias, kept when hyperbolic at 3 sigma."""
    rng = np.random.default_rng(seed)
    out = []
    for y in meteors_YEARS:
        p = os.path.join(meteors_ROOT, "traj_summary_yearly_%d.txt" % y)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            if line.startswith("#") or ";" not in line:
                continue
            c = line.split(";")
            if len(c) < 85 or c[3].strip() != "-1":
                continue
            e, se, qc = num(c[25]), num(c[26]), num(c[80])
            if e is None or se is None or qc is None or not (emin <= e < 1) or qc < 15:
                continue
            vh, svh, lamh, beth = num(c[21]), num(c[22]), num(c[17]), num(c[19])
            if None in (vh, svh, lamh, beth):
                continue
            dv = rng.normal(0.0, k_scale * svh) if bias is None else vh * rng.normal(*bias)
            if vh + dv < 42.0:
                continue
            lr, br = math.radians(lamh), math.radians(beth)
            u = np.array([math.cos(br) * math.cos(lr), math.cos(br) * math.sin(lr), math.sin(br)])
            E, x = meteors_asymptote(meteors_earth(float(c[1])), -(vh + dv) / meteors_KMS * u)
            if x is None or E - 1 < 3 * se:
                continue
            vinf = math.sqrt(max(0.0, (vh + dv) ** 2 - 42.1219 ** 2))
            out.append(dict(id=c[0].strip(), year=y, sol=num(c[5]), ra=num(c[7]), dec=num(c[9]), vg=num(c[15]), e=E, se=se,
                            q=num(c[37]), vinf=vinf, x=x[0], y=x[1], z=x[2], qc=qc, nst=num(c[84]), shower=None))
    return out


def meteors_analyse():
    rows, n_all, n_spor = meteors_load()
    print("%d trajectories, %d sporadic, %d hyperbolic at 3 sigma with a convergence angle above 15 deg"
          % (n_all, n_spor, len(rows)), flush=True)
    showers = meteors_showers()
    by = {}
    for s in showers:
        by.setdefault(int(s["los"]) % 360, []).append(s)
    for r in rows:
        r["shower"] = meteors_match(r, [s for d in range(-16, 17) for s in by.get((int(r["sol"]) + d) % 360, [])])
    print("%d of them within a shower's radiant, date and speed" % sum(1 for r in rows if r["shower"]), flush=True)
    every = meteors_search(rows, "every hyperbolic sporadic")
    rest = meteors_search([r for r in rows if not r["shower"]], "outside the showers")
    # error-process null: speed scale matching the observed hyperbolic count, three realisations
    sims = []
    for k in (1.0, 1.5, 2.0, 3.0):
        n = len(meteors_null(rows, k, 0))
        print("speed errors at %.1f sigma: %d elliptic sporadics become hyperbolic at 3 sigma" % (k, n), flush=True)
        sims.append((abs(n - len(rows)), k))
    k = min(sims)[1]
    nulls = []
    for seed in (1, 2, 3):
        sim = meteors_null(rows, k, seed)
        res = meteors_search(sim, "error-process realisation %d (speed errors at %.1f sigma)" % (seed, k))
        nulls.append(dict(n=res["n"], n_sig=res["n_sig"], largest=max([o["n"] for o in res["groups"]] or [0]),
                          multi=sum(1 for o in res["groups"] if o["n"] >= 3 and len(o["years"]) >= 2),
                          groups=[dict(n=o["n"], lam=o["lam"], beta=o["beta"], vinf=o["vinf"], years=o["years"])
                                  for o in res["groups"][:10]]))
    # shared-bias null: mean speed overestimate matching the observed hyperbolic count
    bsims = []
    for b in (0.015, 0.02, 0.025, 0.03):
        n = len(meteors_null(rows, 0, 0, bias=(b, 0.01)))
        print("speeds overestimated by %.1f +- 1.0 per cent: %d elliptic sporadics become hyperbolic at 3 sigma" % (100 * b, n),
              flush=True)
        bsims.append((abs(n - len(rows)), b))
    b = min(bsims)[1]
    bias_nulls = []
    for seed in (1, 2, 3):
        sim = meteors_null(rows, 0, seed, bias=(b, 0.01))
        res = meteors_search(sim, "shared overestimate realisation %d (%.1f +- 1.0 per cent)" % (seed, 100 * b))
        bias_nulls.append(dict(n=res["n"], n_sig=res["n_sig"], largest=max([o["n"] for o in res["groups"]] or [0]),
                               multi=sum(1 for o in res["groups"] if o["n"] >= 3 and len(o["years"]) >= 2),
                               groups=[dict(n=o["n"], lam=o["lam"], beta=o["beta"], vinf=o["vinf"], years=o["years"])
                                       for o in res["groups"][:10]]))
    json.dump(dict(n_all=n_all, n_spor=n_spor, n_hyp=len(rows), n_shower=sum(1 for r in rows if r["shower"]), every=every,
                   rest=rest, k_scale=k, error_nulls=nulls, bias=b, bias_nulls=bias_nulls),
              open(os.path.join(core.RESULTS, "meteor_streams.json"), "w"), indent=1)


def meteors_record():
    d = json.load(open(os.path.join(core.RESULTS, "meteor_streams.json")))
    ev, rest = d["every"], d["rest"]
    multi = [o for o in rest["groups"] if o["n"] >= 3 and len(o["years"]) >= 2]
    bn = d.get("bias_nulls", [])
    shared = ("; a speed overestimate of %.1f +- 1.0 per cent shared by every elliptic sporadic gives %s meteors below "
              "p 1e-4 among %s, largest groups %s, and %s multi-year groups"
              % (100 * d["bias"], "/".join(str(x["n_sig"]) for x in bn), "/".join(str(x["n"]) for x in bn),
                 "/".join(str(x["largest"]) for x in bn), "/".join(str(x["multi"]) for x in bn))) if bn else ""
    largest_null = max([x["largest"] for x in bn] or [0])
    sp = core.results("sonotaco_streams.json")
    if os.path.exists(sp):
        so = json.load(open(sp))
        shared += ("; in SonotaCo SNMv3 (%d orbits, %d hyperbolic sporadics) the largest group's asymptote holds %d within 1 deg "
                   "and 3 km/s against %.2f from the local density, and the search finds %d groups"
                   % (so["n_all"], so["n_hyp"], so["at_gmn"]["n1"], so["at_gmn"]["exp1"], len(so["search"]["groups"])))
    gp = core.results("meteor_grid_null.json")
    if os.path.exists(gp):
        g = json.load(open(gp))
        shared += ("; thinned to the observed %d at biases of 1.0-2.0 per cent, 15 realisations give %d-%d meteors below "
                   "p 1e-4 and largest groups of %d-%d" % (g[0]["n"], min(x["n_sig"] for x in g), max(x["n_sig"] for x in g),
                                                           min(x["largest"] for x in g), max(x["largest"] for x in g)))
    core.record_search(
        "gmn_hyperbolic_streams", "Coherent hyperbolic meteor streams", archive="Global Meteor Network trajectory summaries",
        release="yearly files 2018-2026", selection="%d sporadic meteors hyperbolic at 3 sigma with a convergence angle above "
                                                    "15 deg, of %d trajectories" % (d["n_hyp"], d["n_all"]),
        code=os.path.abspath(__file__), entry="meteors", n_targets=d["n_hyp"],
        params=dict(e_sigma=3, qc_deg=15, radius_deg=1, dv_kms=3, background_deg=10, p=1e-4, shower_deg=8, shower_sol=15,
                    shower_vg=0.2),
        null_summary="meteors below p 1e-4: %d against %s with speeds shuffled; outside the IAU showers %d against %s; "
                     "independent speed errors at %.1f sigma give %s%s"
                     % (ev["n_sig"], ev["nulls"], rest["n_sig"], rest["nulls"], d.get("k_scale", 0),
                        "/".join(str(x["n_sig"]) for x in d.get("error_nulls", [])), shared),
        sensitivity="three meteors on one asymptote within 1 deg and 3 km/s above the local density of hyperbolic errors",
        ledger="Coherent dust streams",
        result="%d groups among all hyperbolic sporadics, %d of them carried by IAU shower members with overestimated speeds "
               "(%s), the largest (%d members, %d years) on one asymptote at ecliptic %.1f %+.1f and v_inf %.0f km/s, matched by "
               "one unthinned shared-overestimate realisation and beyond the 15 thinned ones; outside the showers %d group%s, %d "
               "with three or more members from two or more years"
               % (len(ev["groups"]), sum(1 for o in ev["groups"] if o["shower"]),
                  ", ".join(sorted({o["shower"].split()[0] for o in ev["groups"] if o["shower"]})), ev["groups"][0]["n"],
                  len(ev["groups"][0]["years"]), ev["groups"][0]["lam"], ev["groups"][0]["beta"], ev["groups"][0]["vinf"],
                  len(rest["groups"]), "" if len(rest["groups"]) == 1 else "s", len(multi)))
    core.clear_targets("gmn_hyperbolic_streams")
    core.record_targets("gmn_hyperbolic_streams", [dict(
        target="sky:ecl %.1f %+.1f v_inf %.0f" % (o["lam"], o["beta"], o["vinf"]), stat=float(o["n"]), p=o["p_min"],
        unit="members within 1 deg and 3 km/s",
        vet=("%s shower members with overestimated speeds" % o["shower"].split()[0] if o["shower"] else
             "interstellar inflow" if o["ism_sep"] < 15 else
             "shared speed overestimate: the null makes groups of up to %d" % largest_null if o["n"] <= largest_null else None),
        note="years %s; %.0f deg from the interstellar inflow" % (",".join(str(y) for y in o["years"]), o["ism_sep"]))
        for o in ev["groups"] + [o for o in rest["groups"] if not o["shower"]]])
    print("recorded")


meteors_ASYM = (134.5, -1.6, 12.0)  # group asymptote: ecliptic lon, lat (deg), v_inf (km/s)
meteors_LOCUS_TOL = (3.0, 3.0)  # locus tolerance: radiant deg, geocentric km/s
meteors_LOCUS_CONTROLS = tuple(30.0 * k for k in range(1, 12))


def meteors_earth_vel(jd):
    return (meteors_earth(jd + 0.01) - meteors_earth(jd - 0.01)) / 0.02


def meteors_arrivals(jd, lon, lat, vinf_kms):
    """Heliocentric velocities (au/d) at Earth for the given asymptote and v_inf: the trajectories passing either side of the Sun."""
    from scipy.optimize import minimize
    r = meteors_earth(jd)
    u = np.array([math.cos(math.radians(lat)) * math.cos(math.radians(lon)), math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
                  math.sin(math.radians(lat))])
    vmag = math.sqrt((vinf_kms / meteors_KMS) ** 2 + 2 * meteors_MU / float(np.linalg.norm(r)))

    def vec(p):
        return vmag * np.array([math.cos(p[1]) * math.cos(p[0]), math.cos(p[1]) * math.sin(p[0]), math.sin(p[1])])

    def miss(p):
        _, a = meteors_asymptote(r, vec(p))
        return 10.0 if a is None else float(np.degrees(np.arccos(np.clip(np.dot(a, u), -1, 1))))
    grid = sorted((miss((th, ph)), th, ph) for th in np.radians(np.arange(0, 360, 4.0)) for ph in np.radians(np.arange(-88, 89, 4.0)))
    sols = []
    for _, th, ph in grid[:16]:
        res = minimize(miss, [th, ph], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-5, maxiter=400))
        if res.fun < 0.1:
            v = vec(res.x)
            if not any(np.linalg.norm(v - s) * meteors_KMS < 0.5 for s in sols):
                sols.append(v)
    return r, sols


def meteors_locus_table(lon, lat, vinf, sols=(60.0, 240.0, 0.5)):
    """Geocentric radiants (ecliptic, deg) and speeds (km/s) of the stream's arrivals per solar longitude."""
    jd0 = 2460500.5  # 2024 July 9
    out = {}
    for ls in np.arange(*sols):
        # date of solar longitude ls
        jd = jd0 + (ls - 107.0) * 365.25 / 360.0
        for _ in range(3):
            re = meteors_earth(jd)
            lsun = math.degrees(math.atan2(-re[1], -re[0])) % 360
            jd += ((ls - lsun + 180) % 360 - 180) * 365.25 / 360.0
        r, vs = meteors_arrivals(jd, lon, lat, vinf)
        ve = meteors_earth_vel(jd)
        pts = []
        for v in vs:
            g = (v - ve) * meteors_KMS
            d = -g / np.linalg.norm(g)
            pts.append((math.degrees(math.atan2(d[1], d[0])) % 360, math.degrees(math.asin(d[2])), float(np.linalg.norm(g))))
        out[round(float(ls), 1)] = pts
    return out


def meteors_locus():
    """Focal-line test: the group's predicted radiants and speeds by date against the GMN table."""
    tol_a, tol_v = meteors_LOCUS_TOL
    lon0, lat0, vinf = meteors_ASYM
    loci = {0.0: meteors_locus_table(lon0, lat0, vinf)}
    for dl in meteors_LOCUS_CONTROLS:
        loci[dl] = meteors_locus_table((lon0 + dl) % 360, lat0, vinf)
    print("loci computed for the stream and %d control asymptotes" % len(meteors_LOCUS_CONTROLS), flush=True)
    for ls in (165.0, 190.0, 130.0, 134.0, 134.5, 135.0, 136.0, 138.0):
        print("  sol lon %.1f: %s" % (ls, ", ".join("(%.0f, %+.0f, %.1f km/s)" % p for p in loci[0.0].get(ls, []))))
    tally = {dl: {} for dl in loci}
    n_read = 0
    for y in range(2022, 2027):
        p = os.path.join(meteors_ROOT, "traj_summary_yearly_%d.txt" % y)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            if line.startswith("#") or ";" not in line:
                continue
            c = line.split(";")
            if len(c) < 85:
                continue
            ls, lam, bet, vg = num(c[5]), num(c[11]), num(c[13]), num(c[15])
            if None in (ls, lam, bet, vg) or not 60.0 <= ls < 240.0:
                continue
            n_read += 1
            key = round(math.floor(ls * 2) / 2, 1)
            code = c[4].strip()
            cb = math.cos(math.radians(bet))
            for dl, table in loci.items():
                for plam, pbet, pv in table.get(key, []):
                    sep = math.degrees(math.acos(max(-1.0, min(1.0, math.sin(math.radians(bet)) * math.sin(math.radians(pbet))
                                                                 + cb * math.cos(math.radians(pbet)) * math.cos(math.radians(lam - plam))))))
                    if sep <= tol_a and abs(vg - pv) <= tol_v:
                        t = tally[dl].setdefault("rows", [])
                        t.append(dict(year=y, sol=ls, lam=lam, bet=bet, vg=vg, code=code, hyper=(num(c[25]) or 0) > 1, id=c[0].strip()))
                        break
    print("%d meteors read between solar longitudes 60 and 240" % n_read)

    def summ(rows, lo, hi):
        w = [r for r in rows if lo <= r["sol"] < hi]
        return dict(n=len(w), south=sum(1 for r in w if r["bet"] < 0), north=sum(1 for r in w if r["bet"] >= 0),
                    per=sum(1 for r in w if r["code"] == "PER"), hyper=sum(1 for r in w if r["hyper"]),
                    far=sum(1 for r in w if abs(r["bet"]) > 10))
    windows = (("Aug 3-11", 130.3, 138.5), ("Jun-Jul", 60.0, 128.0), ("Aug 12-Oct", 139.0, 200.0))
    out = {}
    for dl in loci:
        rows = tally[dl].get("rows", [])
        out[str(dl)] = {name: summ(rows, lo, hi) for name, lo, hi in windows}
        out[str(dl)]["rows"] = rows if dl == 0.0 else len(rows)
    for name, lo, hi in windows:
        s = out["0.0"][name]
        ctl = [out[str(dl)][name] for dl in meteors_LOCUS_CONTROLS]
        print("%s: stream loci hold %d meteors (%d south, %d north, %d beyond 10 deg from the ecliptic, %d Perseid-tagged, %d hyperbolic); "
              "controls %s (mean %.1f, south %.1f)"
              % (name, s["n"], s["south"], s["north"], s["far"], s["per"], s["hyper"], sorted(c["n"] for c in ctl),
                 np.mean([c["n"] for c in ctl]), np.mean([c["south"] for c in ctl])))
    for r in sorted(out["0.0"]["rows"], key=lambda r: r["sol"]):
        if 130.3 <= r["sol"] < 138.5:
            print("  %d sol %.2f: radiant (%.1f, %+.1f) %.1f km/s %s%s %s" % (r["year"], r["sol"], r["lam"], r["bet"], r["vg"], r["code"] or "sporadic",
                                                                     " hyperbolic" if r["hyper"] else "", r["id"]))
    json.dump(out, open(os.path.join(core.RESULTS, "gmn_focal_line.json"), "w"), indent=1)
    return out


def meteors_locus_record():
    out = json.load(open(os.path.join(core.RESULTS, "gmn_focal_line.json")))
    s = out["0.0"]["Aug 3-11"]
    ctl = [out[str(dl)]["Aug 3-11"] for dl in meteors_LOCUS_CONTROLS]
    season = out["0.0"]["Jun-Jul"]["n"] + s["n"] + out["0.0"]["Aug 12-Oct"]["n"]
    cseason = [out[str(dl)]["Jun-Jul"]["n"] + out[str(dl)]["Aug 3-11"]["n"] + out[str(dl)]["Aug 12-Oct"]["n"] for dl in meteors_LOCUS_CONTROLS]
    core.record_search(
        "gmn_focal_line", "Coherent hyperbolic meteor streams", archive="Global Meteor Network trajectory summaries",
        release="2022-2026", selection="every GMN meteor between solar longitudes 60 and 240 against the arrival loci of the group's asymptote",
        code=os.path.abspath(__file__), entry="meteors locus", n_targets=1,
        params=dict(asymptote=list(meteors_ASYM), tolerance=list(meteors_LOCUS_TOL), controls=list(meteors_LOCUS_CONTROLS)),
        null_summary="the same loci for asymptotes 30 to 330 deg away in longitude: Aug 3-11 %s, whole season %s"
                     % (sorted(c["n"] for c in ctl), sorted(cseason)),
        sensitivity="a stream of 16 members over four years would place about 40 per cent of them on the Aug 3-11 loci, half of them south of the ecliptic",
        ledger="Coherent hyperbolic meteor streams",
        result="Aug 3-11 loci: %d meteors (%d south, %d north, %d beyond 10 deg from the ecliptic, %d Perseid-tagged, %d hyperbolic) against a "
               "control mean of %.1f (%.1f south); whole season %d against a control mean of %.1f"
               % (s["n"], s["south"], s["north"], s["far"], s["per"], s["hyper"], np.mean([c["n"] for c in ctl]), np.mean([c["south"] for c in ctl]),
                  season, np.mean(cseason)))
    core.clear_targets("gmn_focal_line")
    core.record_targets("gmn_focal_line", [dict(target="gmn:asymptote 134.5 -1.6 12 km/s", stat=s["n"], unit="meteors on the Aug 3-11 loci",
                                                    note="controls %s" % sorted(c["n"] for c in ctl))])
    print("recorded")


def meteors_cli():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "record", "locus", "locus_record"))
    a = p.parse_args()
    {"fetch": meteors_fetch, "analyse": meteors_analyse, "record": meteors_record, "locus": meteors_locus,
     "locus_record": meteors_locus_record}[a.mode]()


spin_DOC = r"""Centrifugal habitats: LCDB asteroids needing 100 kPa tensile strength at kilometre size, or spinning under 10 minutes, at 2000 kg/m^3.

    python -I seti.py solar_system spin [record] [--rho 2000] [--d-min-km 0.3] [--kpa-min 1] [--km-d 1] [--kpa 100] [--minutes 10]"""


def spin_main(record=False, rho=2000.0, d_min_km=0.3, kpa_min=1.0, km_d=1.0, kpa=100.0, minutes=10.0):
    import math as _m
    p = core.data("lcdb", "F-D_BASIC.TXT")
    lines = open(p, encoding="latin-1").read().splitlines()
    hi = [i for i, ln in enumerate(lines) if ln.startswith("Number")][0]
    hdr = lines[hi]
    fam, nm, cl = hdr.index("Fam"), hdr.index("Name"), hdr.index("Class")
    G = 6.674e-11
    wc2 = 4 * _m.pi * G * rho / 3
    out = []
    for ln in lines[hi + 2:]:
        if not ln.strip():
            continue
        tok = ln[fam:].split()
        try:
            H, _, D, P, _ = [float(x.strip("<>")) for x in tok[1:6]]
        except (ValueError, IndexError):
            continue
        U, amps = None, []
        for x in tok[6:]:
            if re.fullmatch(r"[0-3][+-]?", x):
                U = x
                break
            v = num(x)
            if v is not None:
                amps.append(v)
        w = 2 * _m.pi / (P * 3600)
        R = D * 500
        out.append(dict(num=ln[:nm].strip(), name=ln[nm:cl].strip(), cls=ln[cl:fam].strip(), H=H, D=D, P=P, U=U,
                        amp=max(amps) if amps else None, kpa=rho * R * R / 2 * max(0.0, w * w - wc2) / 1000))
    need = sorted([o for o in out if o["D"] >= d_min_km and o["kpa"] > kpa_min], key=lambda o: -o["kpa"])
    km_fast = [o for o in out if o["D"] >= km_d and o["P"] * 60 < minutes]
    km_strong = [o for o in out if o["D"] >= km_d and o["kpa"] >= kpa]
    print("%d asteroids with periods and diameters; spin limit %.2f h at %g kg/m^3; %d of %g km or more need over %g kPa; bodies of %g km "
          "or more under %g minutes %d, needing %g kPa %d" % (len(out), 2 * _m.pi / _m.sqrt(wc2) / 3600, rho, len(need), d_min_km, kpa_min, km_d,
                                                            minutes, len(km_fast), kpa, len(km_strong)))
    for o in need:
        print("  %-8s %-14s H %5.2f D %6.3f km P %7.2f min U %-3s amplitude %-5s needs %6.1f kPa"
              % (o["num"], o["name"], o["H"], o["D"], o["P"] * 60, o["U"], o["amp"], o["kpa"]))
    json.dump(out, open(core.results("spin_strength.json"), "w"))
    if record:
        core.record_search(
            "lcdb_spin_strength", "Centrifugal habitats", archive="Asteroid Lightcurve Database (LCDB) frequency-diameter table",
            selection="%d asteroids with periods and diameters" % len(out), code=os.path.abspath(__file__), entry="spin",
            n_targets=len(out), params=dict(rho=rho, d_km=[d_min_km, km_d], kpa_min=kpa_min, kpa=kpa, minutes=minutes),
            null_summary="the spin barrier of the catalogue itself: %d bodies of %g km or more need over %g kPa" % (len(need), d_min_km, kpa_min),
            sensitivity="a body of %g km spinning in %g minutes or needing %g kPa, at the LCDB's period quality" % (km_d, minutes, kpa),
            ledger="Centrifugal habitats",
            result="%d bodies of %g km or more under %g minutes and %d needing %g kPa; the most demanding are %s"
                   % (len(km_fast), km_d, minutes, len(km_strong), kpa,
                      "; ".join("%s (%.2f km from %s, %.2f min, U %s) at %.0f kPa" % (o["name"], o["D"], "H", o["P"] * 60, o["U"], o["kpa"]) for o in need[:2])))
        core.clear_targets("lcdb_spin_strength")
        core.record_targets("lcdb_spin_strength", [dict(
            target="mpc:%s" % (o["num"] or o["name"]), stat=o["kpa"], unit="kPa needed at %g kg/m^3" % rho,
            note="%s, D %.3f km, P %.3f h, U %s" % (o["name"], o["D"], o["P"], o["U"])) for o in need])
        print("recorded")


phase_DOC = r"""Self-luminous asteroids: ATLAS H, G1, G2 phase curves at the flattest slope allowed (0.0179 mag/deg over 5-30 deg), judged by fit residual and albedo.

    python -I seti.py solar_system phase [record] [--ph-min-deg 7] [--ph-max-deg 25] [--n-min 100] [--slope-deg 5 30] [--edge-g1 0.05] [--edge-g2 0.9] [--bright 0.6] [--albedo 0.8]"""


def phase_main(record=False, ph_min_deg=7.0, ph_max_deg=25.0, n_min=100, slope_deg=(5.0, 30.0), edge_g1=0.05, edge_g2=0.9, bright=0.6, albedo=0.8):
    from astropy.table import Table
    from sbpy.photometry import HG1G2
    p = core.data("atlas_phase", "atlas_phase.fits")
    if not os.path.exists(p):
        from astroquery.vizier import Vizier
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Vizier(row_limit=-1, timeout=1800, columns=["Number", "Name", "Band", "Class", "N", "phmin", "phmax", "H", "G1", "G2", "rms",
                                                     "B_G1", "b_G1", "B_G2", "b_G2", "G12", "albedo", "e_albedo"]
               ).get_catalogs("VII/288/phase")[0].write(p, overwrite=True)
    t = Table.read(p)

    def a(c):
        return np.array(np.ma.filled(t[c].astype(float), np.nan))
    G1, G2, BG1, bG2 = a("G1"), a("G2"), a("B_G1"), a("b_G2")
    N, phmin, phmax, alb, rms = a("N"), a("phmin"), a("phmax"), a("albedo"), a("rms")
    idx = np.nonzero(np.isfinite(G1) & np.isfinite(G2) & (phmin < ph_min_deg) & (phmax > ph_max_deg) & (N >= n_min))[0]
    al = np.radians(list(slope_deg))
    s = np.array([float(np.diff(HG1G2(H=0, G1=G1[i], G2=G2[i])(al))[0] / (slope_deg[1] - slope_deg[0])) for i in idx])
    edge = (BG1[idx] < edge_g1) & (bG2[idx] > edge_g2)
    e_alb = alb[idx][edge]
    hi = alb[idx] > bright
    glow = [int(i) for i, e in zip(idx, edge) if e and np.isfinite(alb[i]) and alb[i] > albedo]
    print("%d curves kept of %d; slope %g-%g deg median %.4f mag/deg, flattest %.4f; %d pinned at the flat edge (residual median "
          "%.2f mag), %d of them with a literature albedo (%.2f-%.2f), %d above %g; curves with albedo above %g have median slope %.4f"
          % (len(idx), len(t), slope_deg[0], slope_deg[1], np.median(s), s.min(), edge.sum(), np.median(rms[idx][edge]), np.isfinite(e_alb).sum(),
             np.nanmin(e_alb) if np.isfinite(e_alb).any() else np.nan, np.nanmax(e_alb) if np.isfinite(e_alb).any() else np.nan,
             len(glow), albedo, bright, np.median(s[hi])))
    if record:
        core.record_search(
            "atlas_self_luminous", "Self-luminous asteroids", archive="ATLAS phase curves (Mahlke et al. 2021, VizieR VII/288)",
            selection="%d curves with phases below %g and above %g deg and %d or more points" % (len(idx), ph_min_deg, ph_max_deg, n_min),
            code=os.path.abspath(__file__), entry="phase", n_targets=len(idx),
            params=dict(phases_deg=list(slope_deg), ph_min_deg=ph_min_deg, ph_max_deg=ph_max_deg, n_min=n_min, edge_g1=edge_g1, edge_g2=edge_g2, bright=bright, albedo=albedo),
            null_summary="the population's slopes: median %.4f mag/deg; curves with albedo above %g, median %.4f"
                         % (np.median(s), bright, np.median(s[hi])),
            sensitivity="a curve at the model's flattest slope (%.4f mag/deg) with a literature albedo above %g" % (s.min(), albedo),
            ledger="Self-luminous asteroids",
            result="%d curves pinned at the flattest slope the H, G1, G2 model allows, with median residuals of %.2f mag, the mark of "
                   "rotation swamping the phase signal; %d carry albedos (%.2f-%.2f), %d above %g"
                   % (edge.sum(), np.median(rms[idx][edge]), np.isfinite(e_alb).sum(),
                      np.nanmin(e_alb) if np.isfinite(e_alb).any() else 0, np.nanmax(e_alb) if np.isfinite(e_alb).any() else 0, len(glow), albedo))
        core.clear_targets("atlas_self_luminous")
        core.record_targets("atlas_self_luminous", [dict(
            target="mpc:%s" % (str(t["Number"][i]) if not np.ma.is_masked(t["Number"][i]) else str(t["Name"][i])), stat=float(sv),
            unit="mag/deg over %g-%g deg" % tuple(slope_deg), vet="pinned at the model's flat edge" if e else None,
            note="%s band %s, G1 %.3f, G2 %.3f, rms %.2f, albedo %s" % (t["Name"][i], t["Band"][i], G1[i], G2[i], rms[i],
                                                                     "%.2f" % alb[i] if np.isfinite(alb[i]) else "-"))
            for i, sv, e in zip(idx, s, edge)])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------

darkcomet_DOC = r"""One dark-comet candidate: NEOWISE NEATM size and albedo, and ASSIST astrometric refits with and without a transverse acceleration, each apparition left out in turn.

    python -I seti.py solar_system darkcomet fetch DESIG
    python -I seti.py solar_system darkcomet thermal DESIG
    python -I seti.py solar_system darkcomet refit DESIG [--ephem DIR]      (needs assist and the two ephemeris files)
    python -I seti.py solar_system darkcomet record DESIG
    python -I seti.py solar_system darkcomet images DESIG
    python -I seti.py solar_system darkcomet precovery DESIG --night YYYY-MM-DD [--site 568]
    python -I seti.py solar_system darkcomet robust DESIG [--boot 200] [--workers 8]   (night jackknife and bootstrap, weighting variants, prediction ellipses)
    python -I seti.py solar_system darkcomet frames DESIG --night YYYY-MM-DD [--site 568]   (both orbits at each re-measured frame)
    python -I seti.py solar_system darkcomet field DESIG --night YYYY-MM-DD [--site 568]    (known asteroids around the night's track, SkyBoT)
    python -I seti.py solar_system darkcomet stack DESIG
    python -I seti.py solar_system darkcomet variants DESIG      (the NEATM fit under other assumptions; every NEOWISE detection as a table)
    python -I seti.py solar_system darkcomet blockcov DESIG      (A2 under a night-level covariance from restricted maximum likelihood)
    python -I seti.py solar_system darkcomet locus DESIG --night YYYY-MM-DD   (the positions the other apparitions allow, A2 held at each value)
    python -I seti.py solar_system darkcomet forecast DESIG      (daily ephemeris of both orbits at the next apparition, and the A2 it would measure)
    python -I seti.py solar_system darkcomet census DESIG --night YYYY-MM-DD [--workers 6]   (moving sources in the whole CFHT frames, injected
                                                                 copies of the object, and the locus searched; needs the frames in cfht_<night>/full)
    python -I seti.py solar_system darkcomet normal DESIG [--ephem DIR]   (gravity, A2, A2 and A3, all three terms, with and without JPL's radar delays)
    python -I seti.py solar_system darkcomet normal-record DESIG"""

darkcomet_ROOT = os.environ.get("DARKCOMET_ROOT", core.data("darkcomet"))
darkcomet_BANDS = {"w1": (3.3526e-6, 309.540), "w2": (4.6028e-6, 171.787)}  # isophotal wavelength (m), zero point (Jy)
darkcomet_EPS, darkcomet_G, darkcomet_PIR = 0.9, 0.15, 1.6
darkcomet_FLOOR = 0.10
darkcomet_AU = 1.495978707e11


def darkcomet_dir(desig):
    d = os.path.join(darkcomet_ROOT, desig.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)
    return d


def darkcomet_obs80(desig):
    """MPC 80-column observations of one object as text."""
    import urllib.request
    req = urllib.request.Request("https://data.minorplanetcenter.net/api/get-obs", method="GET",
                                 data=json.dumps({"desigs": [desig], "output_format": ["OBS80"]}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))[0]["OBS80"]


def darkcomet_parse(text):
    """Optical observations from 80-column records: UTC JD, RA, Dec (deg), site, satellite geocentric position (km)."""
    from astropy.time import Time
    rows, last = [], None
    for line in text.splitlines():
        if len(line) < 80:
            continue
        if line[14] in "sv" and last is not None:
            if line[14] == "s" and line[32] in "12":
                scale = 1.0 if line[32] == "1" else darkcomet_AU / 1e3
                last["sat_km"] = [float(line[c:c + 11].replace(" ", "")) * scale for c in (34, 46, 58)]
            continue
        if line[14] in "rRxX":
            continue
        y, mo, dd = line[15:19], line[20:22], line[23:32]
        day = float(dd)
        jd = Time("%s-%s-%02dT00:00:00" % (y, mo, int(day)), scale="utc").jd + (day - int(day))
        h, m, s = float(line[32:34]), float(line[35:37]), float(line[38:44])
        sg = -1.0 if line[44] == "-" else 1.0
        dg, dm, ds = float(line[45:47]), float(line[48:50]), float(line[51:56])
        last = dict(jd_utc=jd, ra=15.0 * (h + m / 60 + s / 3600), dec=sg * (dg + dm / 60 + ds / 3600), site=line[77:80],
                    mag=line[65:70].strip(), band=line[70].strip())
        rows.append(last)
    return rows


def darkcomet_horizons(desig, jds, center):
    """Heliocentric and centre distances (au) and phase angle (deg) at each UTC JD."""
    out = []
    for i in range(0, len(jds), 50):
        q = {"format": "json", "COMMAND": "'DES=%s;'" % desig, "CENTER": "'%s'" % center, "EPHEM_TYPE": "'OBSERVER'",
             "QUANTITIES": "'19,20,24'", "TLIST": "'%s'" % " ".join("%.6f" % j for j in jds[i:i + 50]),
             "TLIST_TYPE": "'JD'", "TIME_TYPE": "'UT'", "CSV_FORMAT": "'YES'", "OBJ_DATA": "'NO'"}
        txt = json.loads(core.get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), timeout=120))["result"]
        body = txt.split("$$SOE")[1].split("$$EOE")[0]
        for line in body.strip().splitlines():
            c = [x.strip() for x in line.split(",")]
            nums = [num(x) for x in c[3:] if x]
            out.append(dict(r=nums[0], delta=nums[2], alpha=nums[4]))
    return out


def darkcomet_fetch(desig):
    d = darkcomet_dir(desig)
    text = darkcomet_obs80(desig)
    open(os.path.join(d, "obs80.txt"), "w").write(text)
    sb = json.loads(core.get("https://ssd-api.jpl.nasa.gov/sbdb.api?" + urllib.parse.urlencode(
        {"sstr": desig, "phys-par": 1, "cov": "mat", "full-prec": 1}), timeout=120))
    json.dump(sb, open(os.path.join(d, "sbdb.json"), "w"), indent=1)
    rows = darkcomet_parse(text)
    wise = [r for r in rows if r["site"] == "C51"]

    def one(r):
        mjd = r["jd_utc"] - 2400000.5
        q = ("SELECT ra, dec, mjd, w1mpro, w1sigmpro, w1snr, w2mpro, w2sigmpro, w2snr, qual_frame, cc_flags, ph_qual, nb "
             "FROM neowiser_p1bs_psd WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, 0.003)) = 1 AND "
             "mjd BETWEEN %.6f AND %.6f" % (r["ra"], r["dec"], mjd - 5e-4, mjd + 5e-4))
        found = core.tap("irsa", q, timeout=900)
        return dict(mpc=r, neowise=found[0] if found else None)
    det = core.pmap(one, wise, threads=6)
    geo = darkcomet_horizons(desig, [x["mpc"]["jd_utc"] for x in det], "500@-163")
    for x, g in zip(det, geo):
        x.update(g)
    json.dump(det, open(os.path.join(d, "neowise.json"), "w"), indent=1)
    o = sb["orbit"]
    print("%s: %d observations from %s to %s; %d WISE positions, %d with a NEOWISE detection; A2 %s"
          % (sb["object"]["fullname"], len(rows), o["first_obs"], o["last_obs"], len(wise),
             sum(1 for x in det if x["neowise"]), [(p["name"], p["value"], p["sigma"]) for p in o.get("model_pars", [])]))


def darkcomet_planck(lam, T):
    """Planck's B_lambda (W m^-2 m^-1 sr^-1)."""
    x = 1.438776877e-2 / (lam * np.maximum(T, 1e-3))
    return np.where(T > 0, 1.191042972e-16 / lam ** 5 / np.expm1(np.minimum(x, 700.0)), 0.0)


def darkcomet_phase(alpha):
    t = math.tan(math.radians(alpha) / 2)
    G = darkcomet_G
    return (1 - G) * math.exp(-3.33 * t ** 0.63) + G * math.exp(-1.87 * t ** 1.22)


def darkcomet_flux(band, D_km, eta, pv, r, delta, alpha, n=60):
    """NEATM thermal plus reflected flux density (Jy) in one WISE band."""
    lam, _ = darkcomet_BANDS[band]
    A = (0.290 + 0.684 * darkcomet_G) * pv
    tss = ((1 - A) * 1361.0 / r ** 2 / (eta * darkcomet_EPS * 5.670374419e-8)) ** 0.25
    lat = (np.arange(n) + 0.5) / n * np.pi - np.pi / 2
    lon = (np.arange(n) + 0.5) / n * np.pi - np.pi / 2
    P, L = np.meshgrid(lat, lon, indexing="ij")
    a = math.radians(alpha)
    mu_obs = np.cos(P) * np.cos(L - a)
    T = tss * (np.cos(P) * np.cos(L)) ** 0.25
    w = np.where(mu_obs > 0, mu_obs, 0.0) * np.cos(P) * (np.pi / n) ** 2
    R, dist = D_km * 500.0, delta * darkcomet_AU
    f_th = darkcomet_EPS * R ** 2 / dist ** 2 * np.sum(darkcomet_planck(lam, T) * w)
    sun = np.pi * darkcomet_planck(lam, 5778.0) * (6.957e8 / darkcomet_AU) ** 2
    f_ref = sun / r ** 2 * darkcomet_PIR * pv * R ** 2 / dist ** 2 * darkcomet_phase(alpha)
    return (f_th + f_ref) * lam ** 2 / 2.99792458e8 * 1e26


def darkcomet_thermal(desig, quiet=False, save=True):
    """NEATM fit to W2 at fixed beaming 1.4 +- 0.5, albedo from H; duplicates merged, S/N below 4 and per-apparition outliers set aside, W1 compared."""
    from scipy.optimize import least_squares
    d = darkcomet_dir(desig)
    det, seen = [], set()
    for x in json.load(open(os.path.join(d, "neowise.json"))):
        if x["neowise"] and round(num(x["neowise"]["mjd"]), 6) not in seen:
            seen.add(round(num(x["neowise"]["mjd"]), 6))
            det.append(x)
    H = num(next(p["value"] for p in json.load(open(os.path.join(d, "sbdb.json")))["phys_par"] if p["name"] == "H"))
    lab = darkcomet_apparitions(np.array([x["mpc"]["jd_utc"] for x in det]))
    mags = {b: np.array([num(x["neowise"][b + "mpro"]) if (num(x["neowise"][b + "snr"]) or 0) >= 4 else np.nan for x in det])
            for b in darkcomet_BANDS}
    clip = np.zeros(len(det), bool)
    for k in set(lab):
        m = (lab == k) & np.isfinite(mags["w2"])
        med = np.median(mags["w2"][m])
        mad = 1.4826 * np.median(np.abs(mags["w2"][m] - med))
        clip |= m & (np.abs(mags["w2"] - med) > 3 * max(mad, 0.05))
    obs = []
    for i, x in enumerate(det):
        if np.isfinite(mags["w2"][i]) and not clip[i]:
            s = num(x["neowise"]["w2sigmpro"]) or 0.3
            f = darkcomet_BANDS["w2"][1] * 10 ** (-0.4 * mags["w2"][i])
            obs.append((f, f * math.hypot(0.4 * math.log(10) * s, darkcomet_FLOOR), x["r"], x["delta"], x["alpha"],
                        x["mpc"]["jd_utc"], i))

    def pv_of(D):
        return (1329.0 / D) ** 2 * 10 ** (-0.4 * H)

    def fit_eta(eta):
        f1 = least_squares(lambda p: [(darkcomet_flux("w2", math.exp(p[0]), eta, pv_of(math.exp(p[0])), r, dl, al) - f) / e
                                      for f, e, r, dl, al, _, _ in obs], [math.log(1.0)],
                           bounds=([math.log(0.05)], [math.log(30)]))
        D = math.exp(f1.x[0])
        sD = D * math.sqrt(1.0 / float(f1.jac.T @ f1.jac)) * math.sqrt(max(1.0, 2 * f1.cost / max(len(obs) - 1, 1)))
        return dict(eta=eta, D_km=D, D_err=sD, pv=pv_of(D), chi2=2 * f1.cost)

    scan = [fit_eta(e) for e in (0.9, 1.0, 1.2, 1.4, 1.6, 1.9)]
    best = next(s for s in scan if s["eta"] == 1.4)
    lo, hi = next(s for s in scan if s["eta"] == 0.9), next(s for s in scan if s["eta"] == 1.9)
    D = best["D_km"]
    w1 = [(i, mags["w1"][i], -2.5 * math.log10(darkcomet_flux("w1", D, 1.4, pv_of(D), x["r"], x["delta"], x["alpha"])
                                               / darkcomet_BANDS["w1"][1])) for i, x in enumerate(det) if np.isfinite(mags["w1"][i])]
    out = dict(H=H, n_epochs=len(det), n_w2=len(obs), clipped=[det[i]["mpc"]["jd_utc"] for i in np.nonzero(clip)[0]],
               D_km=D, D_err=best["D_err"], eta=1.4, pv=best["pv"], D_range=[lo["D_km"], hi["D_km"]],
               pv_range=[hi["pv"], lo["pv"]], chi2=best["chi2"], scan=scan,
               r=[min(o[2] for o in obs), max(o[2] for o in obs)], alpha=[min(o[4] for o in obs), max(o[4] for o in obs)],
               w1=[dict(jd=det[i]["mpc"]["jd_utc"], mag=float(m), model=float(mm)) for i, m, mm in w1],
               fluxes=[dict(jy=f, err=e, r=r, delta=dl, alpha=al, jd=jd,
                            model=darkcomet_flux("w2", D, 1.4, pv_of(D), r, dl, al)) for f, e, r, dl, al, jd, _ in obs])
    if save:
        json.dump(out, open(os.path.join(d, "thermal.json"), "w"), indent=1)
    if not quiet:
        print("%s: %d NEOWISE detections, %d W2 fluxes at S/N 4 or more (%d set aside) at r %.2f-%.2f au, phase %.0f-%.0f deg"
              % (desig, len(det), len(obs), int(clip.sum()), out["r"][0], out["r"][1], out["alpha"][0], out["alpha"][1]))
        print("NEATM at eta 1.4: D %.2f +- %.2f km, pV %.3f (H %.2f), chi2 %.1f for %d fluxes; eta 0.9-1.9: D %.2f-%.2f km, "
              "pV %.3f-%.3f" % (D, best["D_err"], best["pv"], H, best["chi2"], len(obs), lo["D_km"], hi["D_km"], hi["pv"],
                                lo["pv"]))
        for s in scan:
            print("  eta %.1f: D %.2f km, pV %.3f, chi2 %.1f" % (s["eta"], s["D_km"], s["pv"], s["chi2"]))
        print("W1 detected at S/N 4 or more: %d, observed minus model %s" % (len(w1), ", ".join("%+.1f" % (m - mm) for _, m, mm in w1)))
    return out


def darkcomet_thermal_variants(desig):
    """The NEATM fit again with each assumption changed in turn (phase slope G, infrared-to-visible albedo ratio, emissivity, flux floor), and every NEOWISE detection with its photometry, flags and use, into thermal_variants.json and neowise_detections.csv."""
    import csv
    global darkcomet_G, darkcomet_PIR, darkcomet_EPS, darkcomet_FLOOR
    d = darkcomet_dir(desig)
    base = (darkcomet_G, darkcomet_PIR, darkcomet_EPS, darkcomet_FLOOR)
    ref = darkcomet_thermal(desig, quiet=True, save=False)
    rows = []
    for name, g_, pir, eps, fl in (("G 0.05", 0.05, base[1], base[2], base[3]), ("G 0.25", 0.25, base[1], base[2], base[3]),
                                   ("p_IR/p_V 1.0", base[0], 1.0, base[2], base[3]), ("p_IR/p_V 2.0", base[0], 2.0, base[2], base[3]),
                                   ("emissivity 0.95", base[0], base[1], 0.95, base[3]), ("floor 0.05", base[0], base[1], base[2], 0.05),
                                   ("floor 0.20", base[0], base[1], base[2], 0.20)):
        darkcomet_G, darkcomet_PIR, darkcomet_EPS, darkcomet_FLOOR = g_, pir, eps, fl
        try:
            t = darkcomet_thermal(desig, quiet=True, save=False)
        finally:
            darkcomet_G, darkcomet_PIR, darkcomet_EPS, darkcomet_FLOOR = base
        rows.append(dict(variant=name, D_km=t["D_km"], pv=t["pv"], chi2=t["chi2"], D_range=t["D_range"], change=t["D_km"] / ref["D_km"] - 1))
        print("%-16s D %.2f km (%+.1f per cent), pV %.3f, chi2 %.1f; eta 0.9-1.9: %.2f-%.2f km" % (
            name, t["D_km"], 100 * rows[-1]["change"], t["pv"], t["chi2"], t["D_range"][0], t["D_range"][1]))
    json.dump(dict(reference=dict(D_km=ref["D_km"], pv=ref["pv"], chi2=ref["chi2"]), variants=rows), open(os.path.join(d, "thermal_variants.json"), "w"), indent=1)
    used = {round(f["jd"], 6) for f in ref["fluxes"]}
    clipped = {round(j, 6) for j in ref["clipped"]}
    w1model = {round(w["jd"], 6): w["model"] for w in ref["w1"]}
    seen = set()
    with open(os.path.join(d, "neowise_detections.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mjd", "ra_deg", "dec_deg", "r_au", "delta_au", "phase_deg", "w1mpro", "w1sigmpro", "w1snr", "w2mpro", "w2sigmpro", "w2snr",
                    "qual_frame", "cc_flags", "ph_qual", "nb", "w2_in_fit", "w2_note", "w1_model_mag_eta1.4"])
        for x in json.load(open(os.path.join(d, "neowise.json"))):
            n_ = x["neowise"]
            if not n_ or round(num(n_["mjd"]), 6) in seen:
                continue
            seen.add(round(num(n_["mjd"]), 6))
            jd = round(x["mpc"]["jd_utc"], 6)
            snr2 = num(n_["w2snr"]) or 0
            note = "used" if jd in used else ("set aside: differs from its apparition's median by more than 3 robust sd" if jd in clipped
                                              else "S/N below 4" if snr2 < 4 else "")
            w.writerow(["%.6f" % num(n_["mjd"]), n_["ra"], n_["dec"], "%.4f" % x["r"], "%.4f" % x["delta"], "%.1f" % x["alpha"], n_["w1mpro"],
                        n_["w1sigmpro"], n_["w1snr"], n_["w2mpro"], n_["w2sigmpro"], n_["w2snr"], n_["qual_frame"], n_["cc_flags"], n_["ph_qual"],
                        n_["nb"], "yes" if jd in used else "no", note, "%.2f" % w1model[jd] if jd in w1model else ""])
    print("%d detections written to neowise_detections.csv" % len(seen))
    return rows


darkcomet_SIGMA = {"C51": 1.0}  # arcsec until the per-site rms replaces it; 0.5 elsewhere
darkcomet_CHI2_OUT = 8.0  # two-coordinate chi-square outlier threshold


def darkcomet_sites(d):
    """MPC parallax constants: site -> (east longitude deg, rho cos phi', rho sin phi')."""
    p = os.path.join(d, "ObsCodes.html")
    if not os.path.exists(p):
        open(p, "wb").write(core.get("https://minorplanetcenter.net/iau/lists/ObsCodes.html", timeout=120))
    out = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        c = line[:3]
        lon, rc, rs = num(line[4:13]), num(line[13:21]), num(line[21:30])
        if len(c) == 3 and lon is not None and rc is not None and rs is not None:
            out[c] = (lon, rc, rs)
    return out


def darkcomet_observers(rows, sites, eph):
    """Barycentric ICRF observer positions (au): Earth plus site through GCRS, or the satellite's reported position."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    t = Time([r["jd_utc"] for r in rows], format="jd", scale="utc")
    tdb = t.tdb.jd
    out = np.zeros((len(rows), 3))
    for i, r in enumerate(rows):
        e = eph.get_particle("earth", tdb[i] - eph.jd_ref)
        out[i] = (e.x, e.y, e.z)
    re_m = 6378137.0
    for site in {r["site"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["site"] == site]
        if "sat_km" in rows[idx[0]]:
            for i in idx:
                out[i] += np.array(rows[i]["sat_km"]) * 1e3 / darkcomet_AU
            continue
        lon, rc, rs = sites[site]
        loc = EarthLocation.from_geocentric(re_m * rc * math.cos(math.radians(lon)), re_m * rc * math.sin(math.radians(lon)),
                                            re_m * rs, unit=u.m)
        pos, _ = loc.get_gcrs_posvel(t[idx])
        out[idx] += np.vstack([pos.x.to_value(u.m), pos.y.to_value(u.m), pos.z.to_value(u.m)]).T / darkcomet_AU
    return tdb, out


def darkcomet_propagate(eph, t0, state, a2, times, a1=0.0, a3=0.0):
    """Barycentric positions at TDB JDs from a state at t0 in ASSIST, with transverse a2, radial a1 and normal a3 (au/d^2 at 1 au, r^-2)."""
    import assist
    import rebound
    out = np.zeros((len(times), 3))
    times = np.asarray(times)
    for sel in (np.nonzero(times < t0)[0][::-1], np.nonzero(times >= t0)[0]):
        if not len(sel):
            continue
        sel = sel[np.argsort(times[sel])[::-1]] if times[sel[0]] < t0 else sel[np.argsort(times[sel])]
        sim = rebound.Simulation()
        sim.add(x=state[0], y=state[1], z=state[2], vx=state[3], vy=state[4], vz=state[5])
        sim.t = t0 - eph.jd_ref
        sim.dt = -1.0 if times[sel[0]] < t0 else 1.0
        ex = assist.Extras(sim, eph)
        ex.particle_params = np.array([a1, a2, a3])
        ex.alpha, ex.nk, ex.nm, ex.nn, ex.r0 = 1.0, 0.0, 2.0, 1.0, 1.0
        for i in sel:
            sim.integrate(times[i] - eph.jd_ref)
            p = sim.particles[0]
            out[i] = (p.x, p.y, p.z)
        del ex
    return out


def darkcomet_radec(vec):
    ra = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(vec[:, 2] / np.linalg.norm(vec, axis=1)))
    return ra, dec


def darkcomet_model(eph, t0, q, t_emit, obs_pos, ra, dec):
    """Residuals (arcsec), RA times cos Dec for every position and then Dec, from the state and any A2 and A1 in q."""
    n = len(q)
    pos = darkcomet_propagate(eph, t0, q[:6], q[6] if n >= 7 else 0.0, t_emit, a1=q[7] if n == 8 else 0.0) - obs_pos
    mra, mdec = darkcomet_radec(pos)
    dra = ((ra - mra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec)) * 3600.0
    return np.concatenate([dra, (dec - mdec) * 3600.0])


def darkcomet_jacobian(eph, t0, p, t_emit, obs_pos, ra, dec):
    """Residuals at p and their forward-difference derivatives with respect to each parameter."""
    h = np.array([1e-9] * 3 + [1e-11] * 3 + [1e-14, 1e-14])[:len(p)]
    r0 = darkcomet_model(eph, t0, p, t_emit, obs_pos, ra, dec)
    J = np.zeros((len(r0), len(p)))
    for k in range(len(p)):
        q = np.array(p, float)
        q[k] += h[k]
        J[:, k] = (darkcomet_model(eph, t0, q, t_emit, obs_pos, ra, dec) - r0) / h[k]
    return r0, J


def darkcomet_fit(eph, t0, p0, with_a2, t_emit, obs_pos, ra, dec, sigma, keep, iters=12, with_a1=False):
    """Gauss-Newton on the state (with A2, optionally A1) by forward differences, outliers re-judged each iteration: parameters, covariance, residuals (arcsec), kept mask."""
    n = 6 + int(with_a2) + int(with_a1 and with_a2)
    p = np.array(list(p0[:7]) + [0.0], float)[:n]

    def model(q):
        return darkcomet_model(eph, t0, q, t_emit, obs_pos, ra, dec)

    keep = keep.copy()
    cov = None
    for _ in range(iters):
        r0, J = darkcomet_jacobian(eph, t0, p, t_emit, obs_pos, ra, dec)
        w = np.concatenate([1.0 / sigma, 1.0 / sigma])
        kk = np.concatenate([keep, keep])
        A, b = J[kk] * w[kk, None], r0[kk] * w[kk]
        # columns scaled to unit norm; partials span orders of magnitude
        s = np.linalg.norm(A, axis=0)
        cov = np.linalg.inv((A / s).T @ (A / s)) / np.outer(s, s)
        dp = -cov @ (A.T @ b)
        p = p + dp
        r1 = model(p)
        chi = (r1[:len(ra)] ** 2 + r1[len(ra):] ** 2) / sigma ** 2
        new_keep = chi < darkcomet_CHI2_OUT
        if np.all(np.abs(dp) < 1e-3 * np.sqrt(np.diag(cov))) and np.array_equal(new_keep, keep):
            break
        keep = new_keep
    res = model(p)
    return p, cov, res[:len(ra)], res[len(ra):], keep


def darkcomet_apparitions(jd, gap=150.0):
    order = np.argsort(jd)
    lab = np.zeros(len(jd), int)
    k = 0
    for a, b in zip(order[:-1], order[1:]):
        if jd[b] - jd[a] > gap:
            k += 1
        lab[b] = k
    return lab


def darkcomet_night_factor(rows, scheme):
    """Same-night uncertainty inflation: 'veres' sqrt(N/4) above four per station-night (Veres et al. 2017), 'night' sqrt(N) per night."""
    f = np.ones(len(rows))
    if scheme == "none":
        return f
    groups = {}
    for i, r in enumerate(rows):
        key = int(math.floor(r["jd_utc"] + 0.5)) if scheme == "night" else (r["site"], int(math.floor(r["jd_utc"] + 0.5)))
        groups.setdefault(key, []).append(i)
    for idx in groups.values():
        n = len(idx)
        if scheme == "veres" and n > 4:
            f[idx] = math.sqrt(n / 4.0)
        elif scheme == "night" and n > 1:
            f[idx] = math.sqrt(n)
    return f


def darkcomet_earth_velocity(eph, tdb):
    """Earth's barycentric velocity (au/d) at each TDB date by one-minute difference."""
    out = np.zeros((len(tdb), 3))
    for i, t in enumerate(tdb):
        a = eph.get_particle("earth", t - eph.jd_ref)
        b = eph.get_particle("earth", t + 1.0 / 1440 - eph.jd_ref)
        out[i] = (np.array([b.x, b.y, b.z]) - np.array([a.x, a.y, a.z])) * 1440.0
    return out


def darkcomet_along_cross(eph, t0, p, t_emit, obs_pos, ra, dec, a2=0.0, a1=0.0, v_obs=None):
    """Offsets (arcsec) from an orbit along and across its one-minute apparent motion seen by the moving observer, Earth by default."""
    if v_obs is None:
        v_obs = darkcomet_earth_velocity(eph, t_emit)
    r1 = darkcomet_radec(darkcomet_propagate(eph, t0, p[:6], a2, t_emit, a1=a1) - obs_pos)
    r2 = darkcomet_radec(darkcomet_propagate(eph, t0, p[:6], a2, t_emit + 1.0 / 1440, a1=a1) - (obs_pos + v_obs / 1440.0))
    cosd = np.cos(np.radians(dec))
    vx, vy = ((r2[0] - r1[0] + 180) % 360 - 180) * cosd * 3600, (r2[1] - r1[1]) * 3600
    rx, ry = ((ra - r1[0] + 180) % 360 - 180) * cosd * 3600, (dec - r1[1]) * 3600
    sp = np.maximum(np.hypot(vx, vy), 1e-9)
    return (rx * vx + ry * vy) / sp, (-rx * vy + ry * vx) / sp


def darkcomet_ephem(ephem_dir=None):
    import assist
    ephem_dir = ephem_dir or core.data("assist")
    return assist.Ephem(os.path.join(ephem_dir, "linux_p1550p2650.440"), os.path.join(ephem_dir, "sb441-n16.bsp"))


def darkcomet_setup(desig, ephem_dir=None):
    """Positions, observers, JPL reference state, emission times and per-site weights from a first fit with A2."""
    d = darkcomet_dir(desig)
    eph = darkcomet_ephem(ephem_dir)
    rows = darkcomet_parse(open(os.path.join(d, "obs80.txt")).read())
    sites = darkcomet_sites(d)
    rows = [r for r in rows if r["site"] in sites or "sat_km" in r]
    tdb, obs_pos = darkcomet_observers(rows, sites, eph)
    ra = np.array([r["ra"] for r in rows])
    dec = np.array([r["dec"] for r in rows])
    sb = json.load(open(os.path.join(d, "sbdb.json")))
    a2_jpl = num(next(p["value"] for p in sb["orbit"]["model_pars"] if p["name"] == "A2"))
    # reference state: JPL solution at an epoch in the best-covered apparition
    lab = darkcomet_apparitions(tdb)
    main = np.bincount(lab).argmax()
    t0 = float(np.round(np.median(tdb[lab == main])) + 0.5)
    q = {"format": "json", "COMMAND": "'DES=%s;'" % desig, "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@0'",
         "REF_PLANE": "'FRAME'", "VEC_TABLE": "'2'", "OUT_UNITS": "'AU-D'", "TLIST": "'%.1f'" % t0, "TIME_TYPE": "'TDB'",
         "CSV_FORMAT": "'YES'", "VEC_CORR": "'NONE'", "OBJ_DATA": "'NO'"}
    txt = json.loads(core.get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), timeout=120))["result"]
    vals = [num(x) for x in txt.split("$$SOE")[1].split("$$EOE")[0].strip().split(",")[2:8]]
    p0 = np.array(vals + [a2_jpl])
    # light time from the reference orbit, fixed through the fits
    geo = darkcomet_propagate(eph, t0, p0[:6], a2_jpl, tdb) - obs_pos
    t_emit = tdb - np.linalg.norm(geo, axis=1) * darkcomet_AU / 2.99792458e8 / 86400.0
    default = np.array([darkcomet_SIGMA.get(r["site"], 0.5) for r in rows])
    sigma = default.copy()
    # first pass sets per-site weights: rms over five or more kept observations, floor 0.15
    p1, cov, dra, ddec, keep = darkcomet_fit(eph, t0, p0, True, t_emit, obs_pos, ra, dec, sigma, np.ones(len(rows), bool))
    site_rms = {}
    for s in {r["site"] for r in rows}:
        m = np.array([r["site"] == s for r in rows]) & keep
        if m.sum() >= 5:
            site_rms[s] = float(np.sqrt(np.mean((dra[m] ** 2 + ddec[m] ** 2) / 2)))
            sigma[np.array([r["site"] == s for r in rows])] = max(site_rms[s], 0.15)
    return dict(d=d, eph=eph, rows=rows, sites=sites, tdb=tdb, obs_pos=obs_pos, ra=ra, dec=dec, sb=sb, a2_jpl=a2_jpl, lab=lab,
                t0=t0, p0=p0, p1=p1, t_emit=t_emit, default=default, sigma=sigma, site_rms=site_rms)


def darkcomet_refit(desig, ephem_dir=None):
    S = darkcomet_setup(desig, ephem_dir)
    d, eph, rows, tdb, obs_pos, ra, dec, sb, a2_jpl, lab, t0, p0, t_emit, sigma, site_rms = (S[k] for k in (
        "d", "eph", "rows", "tdb", "obs_pos", "ra", "dec", "sb", "a2_jpl", "lab", "t0", "p0", "t_emit", "sigma", "site_rms"))
    fits = {}
    everything = np.ones(len(rows), bool)
    # three same-night weightings, and with A1 free
    jobs = [("with A2", True, everything, "none", False), ("with A2, Veres nights", True, everything, "veres", False),
            ("with A2, one position a night", True, everything, "night", False),
            ("with A1 and A2", True, everything, "none", True), ("gravity only", False, everything, "none", False)]
    jobs += [("without %d-%02d" % (int(_y), int(_m)), True, lab != k, "none", False) for k, (_y, _m) in
             ((k, darkcomet_ym(tdb[lab == k].min())) for k in range(lab.max() + 1))]
    for name, with_a2, use, scheme, with_a1 in jobs:
        sel = np.nonzero(use)[0]
        sg = sigma * darkcomet_night_factor(rows, scheme)
        p, cov, dra, ddec, kp = darkcomet_fit(eph, t0, p0, with_a2, t_emit[sel], obs_pos[sel], ra[sel], dec[sel], sg[sel],
                                             np.ones(len(sel), bool), with_a1=with_a1)
        npar = 6 + int(with_a2) + int(with_a1)
        chi2 = float(np.sum(((dra ** 2 + ddec ** 2) / sg[sel] ** 2)[kp]))
        fits[name] = dict(n_obs=int(len(sel)), n_kept=int(kp.sum()), chi2=chi2, dof=int(2 * kp.sum() - npar), weighting=scheme,
                          rms=float(np.sqrt(np.mean((dra ** 2 + ddec ** 2)[kp] / 2))),
                          A2=float(p[6]) if with_a2 else None, A2_sigma=float(math.sqrt(cov[6, 6])) if with_a2 else None,
                          A1=float(p[7]) if with_a1 else None, A1_sigma=float(math.sqrt(cov[7, 7])) if with_a1 else None,
                          rejected=[dict(site=rows[i]["site"], jd=rows[i]["jd_utc"]) for i, k in zip(sel, kp) if not k],
                          resid_max=float(np.max(np.hypot(dra, ddec)[kp])))
        f = fits[name]
        if name in ("with A2", "with A1 and A2", "gravity only"):
            along, cross = darkcomet_along_cross(eph, t0, p, t_emit, obs_pos, ra, dec, a2=p[6] if with_a2 else 0.0,
                                                 a1=p[7] if with_a1 else 0.0)
            f["resid"] = [dict(jd=float(tdb[i]), site=rows[i]["site"], along=float(along[i]), cross=float(cross[i]),
                               kept=bool(kp[i])) for i in range(len(rows))]
        print("%-32s %3d obs, %3d kept, chi2 %7.1f / %3d, rms %.2f arcsec%s%s"
              % (name, f["n_obs"], f["n_kept"], f["chi2"], f["dof"], f["rms"],
                 ", A2 %+.3e +- %.2e (%.1f sigma)" % (f["A2"], f["A2_sigma"], f["A2"] / f["A2_sigma"]) if with_a2 else "",
                 ", A1 %+.3e +- %.2e (%.1f sigma)" % (f["A1"], f["A1_sigma"], f["A1"] / f["A1_sigma"]) if with_a1 else ""))
        if name == "gravity only":
            # gravity-only residuals by apparition; a transverse force grows along track
            for k in range(lab.max() + 1):
                m = (lab[sel] == k) & kp
                if m.any():
                    print("    %d-%02d: %2d obs, mean residual RA %+.2f Dec %+.2f arcsec" % (*darkcomet_ym(tdb[lab == k].min()),
                                                                                       m.sum(), dra[m].mean(), ddec[m].mean()))
                    f.setdefault("by_apparition", []).append(dict(ym="%d-%02d" % darkcomet_ym(tdb[lab == k].min()), n=int(m.sum()),
                                                                  dra=float(dra[m].mean()), ddec=float(ddec[m].mean())))
    out = dict(t0_tdb=t0, a2_jpl=a2_jpl, site_rms=site_rms, fits=fits,
               apparitions=["%d-%02d" % darkcomet_ym(tdb[lab == k].min()) for k in range(lab.max() + 1)])
    # each apparition against the gravity-only orbit of the others: along- and cross-track offsets and equivalent clock error
    for k in range(lab.max() + 1):
        m = lab == k
        pg, cg = darkcomet_fit(eph, t0, p0, False, t_emit[~m], obs_pos[~m], ra[~m], dec[~m], sigma[~m], np.ones((~m).sum(), bool))[:2]
        r1 = darkcomet_radec(darkcomet_propagate(eph, t0, pg[:6], 0.0, t_emit[m]) - obs_pos[m])
        cosd = np.cos(np.radians(dec[m]))
        # prediction uncertainty from the fit covariance by forward differences
        J = np.zeros((2, 6))
        for q in range(6):
            pq = pg[:6].copy()
            hq = 1e-9 if q < 3 else 1e-11
            pq[q] += hq
            rq = darkcomet_radec(darkcomet_propagate(eph, t0, pq, 0.0, t_emit[m]) - obs_pos[m])
            J[0, q] = np.mean(((rq[0] - r1[0] + 180) % 360 - 180) * cosd * 3600) / hq
            J[1, q] = np.mean((rq[1] - r1[1]) * 3600) / hq
        cpred = J @ cg[:6, :6] @ J.T
        # apparent motion: line through the night's positions when three or more, else the model's one-minute motion
        tm = (tdb[m] - tdb[m].min()) * 1440.0
        if m.sum() >= 3 and np.ptp(tm) > 5.0:
            vx = float(np.polyfit(tm, ((ra[m] - ra[m][0] + 180) % 360 - 180) * cosd * 3600, 1)[0])
            vy = float(np.polyfit(tm, (dec[m] - dec[m][0]) * 3600, 1)[0])
            vx, vy = np.full(m.sum(), vx), np.full(m.sum(), vy)
        else:
            ve = darkcomet_earth_velocity(eph, t_emit[m])
            r2 = darkcomet_radec(darkcomet_propagate(eph, t0, pg[:6], 0.0, t_emit[m] + 1.0 / 1440) - (obs_pos[m] + ve / 1440.0))
            vx, vy = ((r2[0] - r1[0] + 180) % 360 - 180) * cosd * 3600, (r2[1] - r1[1]) * 3600
        rx, ry = ((ra[m] - r1[0] + 180) % 360 - 180) * cosd * 3600, (dec[m] - r1[1]) * 3600
        sp = np.hypot(vx, vy)
        along, cross = (rx * vx + ry * vy) / sp, (-rx * vy + ry * vx) / sp
        ux, uy = float(vx[0] / sp[0]), float(vy[0] / sp[0])
        pred_along = float(math.sqrt(max(np.array([ux, uy]) @ cpred @ np.array([ux, uy]), 0.0)))
        pred_cross = float(math.sqrt(max(np.array([-uy, ux]) @ cpred @ np.array([-uy, ux]), 0.0)))
        # every observation against this orbit, for the residual plot
        al_all, cr_all = darkcomet_along_cross(eph, t0, pg, t_emit, obs_pos, ra, dec)
        out.setdefault("left_out", {})[out["apparitions"][k]] = dict(
            n=int(m.sum()), along=float(along.mean()), along_sd=float(along.std()), cross=float(cross.mean()),
            cross_sd=float(cross.std()), rate=float(sp.mean()), clock_s=float(along.mean() / sp.mean() * 60.0),
            pred_along=pred_along, pred_cross=pred_cross,
            resid=[dict(jd=float(tdb[i]), site=rows[i]["site"], along=float(al_all[i]), cross=float(cr_all[i]), left_out=bool(m[i]))
                   for i in range(len(rows))])
        x = out["left_out"][out["apparitions"][k]]
        print("  %s against the gravity-only orbit of the rest: along-track %+.2f +- %.2f, cross-track %+.2f +- %.2f arcsec, "
              "the prediction itself good to %.2f and %.2f; %.2f arcsec/min, a clock error of %+.0f s for the along-track part"
              % (out["apparitions"][k], x["along"], x["along_sd"], x["cross"], x["cross_sd"], pred_along, pred_cross, x["rate"],
                 x["clock_s"]))
    # A2 orbit against the gravity-only orbit fitted without the decisive apparition, seen from Earth
    loo = {a: f for a, f in fits.items() if a.startswith("without")}
    key = min(loo, key=lambda a: abs(loo[a]["A2"] / loo[a]["A2_sigma"]))
    k = out["apparitions"].index(key[8:])
    m = lab != k
    pa = darkcomet_fit(eph, t0, p0, True, t_emit, obs_pos, ra, dec, sigma, np.ones(len(rows), bool))[0]
    pg = darkcomet_fit(eph, t0, p0, False, t_emit[m], obs_pos[m], ra[m], dec[m], sigma[m], np.ones(m.sum(), bool))[0]
    ts = np.arange(np.ceil(tdb.max()) + 0.5, tdb.max() + 8 * 365.25, 10.0)
    earth = np.array([[e.x, e.y, e.z] for e in (eph.get_particle("earth", t - eph.jd_ref) for t in ts)])
    sun = np.array([[s.x, s.y, s.z] for s in (eph.get_particle("sun", t - eph.jd_ref) for t in ts)])
    A, G = (darkcomet_propagate(eph, t0, pa[:6], pa[6], ts), darkcomet_propagate(eph, t0, pg[:6], 0.0, ts))
    (ra1, de1), (ra2, de2) = darkcomet_radec(A - earth), darkcomet_radec(G - earth)
    sep = np.hypot(((ra1 - ra2 + 180) % 360 - 180) * np.cos(np.radians(de1)), de1 - de2) * 3600
    r, dl = np.linalg.norm(A - sun, axis=1), np.linalg.norm(A - earth, axis=1)
    es = earth - sun
    elong = np.degrees(np.arccos(np.sum(-es * (A - earth), axis=1) / np.linalg.norm(es, axis=1) / dl))
    ph = np.degrees(np.arccos(np.clip(np.sum((A - sun) * (A - earth), axis=1) / r / dl, -1, 1)))
    H = num(next(p["value"] for p in sb["phys_par"] if p["name"] == "H"))
    V = H + 5 * np.log10(r * dl) - 2.5 * np.log10([darkcomet_phase(a) for a in ph])
    from astropy.time import Time
    out["prediction"] = dict(without=key[8:], rows=[
        dict(date=Time(t, format="jd", scale="tdb").iso[:10], sep=float(s_), V=float(v), elong=float(e), r=float(rr))
        for t, s_, v, e, rr in zip(ts, sep, V, elong, r) if e > 60 and v < 22.5])
    for x in out["prediction"]["rows"]:
        print("  %s: the two orbits %.1f arcsec apart, V %.1f, elongation %.0f deg" % (x["date"], x["sep"], x["V"], x["elong"]))
    json.dump(out, open(os.path.join(d, "refit.json"), "w"), indent=1)
    return out


# radar stations of the JPL radar astrometry: geodetic longitude, latitude (deg) and height (m)
darkcomet_RADAR_SITES = {"-1": (-66.75278, 18.34417, 497.0), "-13": (-116.79445, 35.24773, 1071.0), "-14": (-116.88954, 35.42590, 1002.0)}
darkcomet_NG = ("A1", "A2", "A3")
darkcomet_NG_SETS = ([], [1], [1, 2], [0, 1, 2])


def darkcomet_radar(desig):
    """JPL's centre-of-mass round-trip delays of a body from stations with known coordinates: [(UTC receive time, delay s, sigma s, receiver, transmitter)]."""
    d = darkcomet_dir(desig)
    path = os.path.join(d, "radar.json")
    if not os.path.exists(path):
        open(path, "wb").write(core.get("https://ssd-api.jpl.nasa.gov/sb_radar.api?" + urllib.parse.urlencode({"des": desig}), timeout=60))
    js = json.load(open(path))
    # a body without radar comes back as the JSON text quoted once more
    js = json.loads(js) if isinstance(js, str) else js
    rows = [dict(zip(js["fields"], r)) for r in js.get("data", [])]
    return [(r["epoch"], float(r["value"]) * 1e-6, float(r["sigma"]) * 1e-6, r["rcvr"], r["xmit"]) for r in rows
            if r["units"] == "us" and r["bp"] == "C" and r["rcvr"] in darkcomet_RADAR_SITES and r["xmit"] in darkcomet_RADAR_SITES]


def darkcomet_delay(eph, t0, q, ng, epoch, rcvr, xmit):
    """Computed round-trip delay (s) received at a UTC epoch: light time iterated on each leg, with the Sun's Shapiro delay on both."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    c = 2.99792458e8

    def station(code, t):
        lon, lat, h = darkcomet_RADAR_SITES[code]
        e = eph.get_particle("earth", t.tdb.jd - eph.jd_ref)
        g, _ = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h * u.m).get_gcrs_posvel(t)
        return np.array([e.x, e.y, e.z]) + np.array([g.x.to_value(u.m), g.y.to_value(u.m), g.z.to_value(u.m)]) / darkcomet_AU

    def body(tb):
        return darkcomet_propagate(eph, t0, q, ng[1], np.array([tb]), a1=ng[0], a3=ng[2])[0]

    t_rx = Time(epoch.replace(" ", "T"), scale="utc")
    r_rx = station(rcvr, t_rx)
    t_b, down = t_rx.tdb.jd, 0.0
    for _ in range(4):
        r_b = body(t_b)
        down = np.linalg.norm(r_b - r_rx) * darkcomet_AU / c
        t_b = t_rx.tdb.jd - down / 86400.0
    up, t_tx = 0.0, t_rx
    for _ in range(4):
        r_tx = station(xmit, t_tx)
        up = np.linalg.norm(r_b - r_tx) * darkcomet_AU / c
        t_tx = t_rx - (down + up) * u.s
    s = eph.get_particle("sun", t_b - eph.jd_ref)
    sun = np.array([s.x, s.y, s.z])

    def shapiro(a, b):
        ra_, rb_, rho = (np.linalg.norm(a - sun) * darkcomet_AU, np.linalg.norm(b - sun) * darkcomet_AU, np.linalg.norm(a - b) * darkcomet_AU)
        return 2 * 1.32712440018e20 / c ** 3 * math.log((ra_ + rb_ + rho) / (ra_ + rb_ - rho))
    return up + down + shapiro(r_tx, r_b) + shapiro(r_b, r_rx)


def darkcomet_ngfit(eph, t0, p0, free, t_emit, obs_pos, ra, dec, sigma, radar=(), iters=12):
    """Gauss-Newton on the state and the free terms among A1, A2, A3 (indices into darkcomet_NG), with any radar delays as further residuals."""
    p = np.array(list(p0[:6]) + [0.0] * len(free), float)
    h = np.array([1e-9] * 3 + [1e-11] * 3 + [1e-14] * len(free))
    keep = np.ones(len(ra), bool)

    def terms(q):
        ng = [0.0, 0.0, 0.0]
        for k, j in enumerate(free):
            ng[j] = q[6 + k]
        return ng

    def resid(q):
        ng = terms(q)
        pos = darkcomet_propagate(eph, t0, q[:6], ng[1], t_emit, a1=ng[0], a3=ng[2]) - obs_pos
        mra, mdec = darkcomet_radec(pos)
        dra = ((ra - mra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec)) * 3600.0
        out = [dra / sigma, (dec - mdec) * 3600.0 / sigma]
        out += [np.array([(tau - darkcomet_delay(eph, t0, q[:6], ng, ep, rx, tx)) / sg for ep, tau, sg, rx, tx in radar])]
        return np.concatenate(out)

    cov = None
    n = len(ra)
    for _ in range(iters):
        r0 = resid(p)
        J = np.zeros((len(r0), len(p)))
        for k in range(len(p)):
            q = p.copy()
            q[k] += h[k]
            J[:, k] = (resid(q) - r0) / h[k]
        kk = np.concatenate([keep, keep, np.ones(len(radar), bool)])
        A, b = J[kk], r0[kk]
        s = np.linalg.norm(A, axis=0)
        cov = np.linalg.inv((A / s).T @ (A / s)) / np.outer(s, s)
        dp = -cov @ (A.T @ b)
        p = p + dp
        r1 = resid(p)
        new = r1[:n] ** 2 + r1[n:2 * n] ** 2 < darkcomet_CHI2_OUT
        if np.all(np.abs(dp) < 1e-3 * np.sqrt(np.diag(cov))) and np.array_equal(new, keep):
            break
        keep = new
    r = resid(p)
    chi = r[:n] ** 2 + r[n:2 * n] ** 2
    return p, cov, chi, keep, r[2 * n:]


def darkcomet_normal(desig, ephem_dir=None):
    """Refits with gravity alone, A2, A2 and A3, and A1, A2 and A3, with and without JPL's radar delays, under site weights and with each night as one position; A3 with each apparition left out."""
    S = darkcomet_setup(desig, ephem_dir)
    d, eph, rows, tdb, obs_pos, ra, dec, lab, t0, p0, t_emit, sigma = (S[k] for k in (
        "d", "eph", "rows", "tdb", "obs_pos", "ra", "dec", "lab", "t0", "p0", "t_emit", "sigma"))
    radar = darkcomet_radar(desig)
    out = dict(n_obs=len(rows), radar=[dict(epoch=ep, delay_s=tau, sigma_s=sg, rcvr=rx, xmit=tx) for ep, tau, sg, rx, tx in radar], fits={}, left_out={},
               apparitions=["%d-%02d" % darkcomet_ym(tdb[lab == k].min()) for k in range(lab.max() + 1)])
    for scheme in ("none", "night"):
        sg = sigma * darkcomet_night_factor(rows, scheme)
        fits = {}
        for use in ((False, True) if radar else (False,)):
            for free in darkcomet_NG_SETS:
                p, cov, chi, keep, rr = darkcomet_ngfit(eph, t0, p0, free, t_emit, obs_pos, ra, dec, sg, radar if use else ())
                name = "%s|%s|%s" % (scheme, "radar" if use else "optical", " ".join(darkcomet_NG[j] for j in free) or "gravity only")
                ng = [0.0, 0.0, 0.0]
                for k, j in enumerate(free):
                    ng[j] = p[6 + k]
                res = [float((tau - darkcomet_delay(eph, t0, p[:6], ng, ep, rx, tx)) * 1e6) for ep, tau, s_, rx, tx in radar]
                fits[name] = dict(chi=chi.tolist(), keep=keep.tolist(), npar=6 + len(free), radar_residual_us=res,
                                  terms={darkcomet_NG[j]: [float(p[6 + k]), float(math.sqrt(cov[6 + k, 6 + k]))] for k, j in enumerate(free)})
        # each model judged on the observations every model of this weighting keeps
        common = np.logical_and.reduce([np.array(f["keep"]) for f in fits.values()])
        for name, f in fits.items():
            f["chi2_common"] = float(np.sum(np.array(f["chi"])[common]))
            f["n_common"] = int(common.sum())
            f["bic"] = f["chi2_common"] + f["npar"] * math.log(2 * common.sum())
            f["kept"] = int(np.sum(f.pop("keep")))
            f.pop("chi")
            print("%-36s chi2 %7.1f on %d common, kept %d, BIC %7.1f; delay residuals %s us; %s" % (
                name, f["chi2_common"], f["n_common"], f["kept"], f["bic"], ", ".join("%+.2f" % x for x in f["radar_residual_us"]) or "none",
                ", ".join("%s %+.3e +- %.1e (%.1f sigma)" % (k, v[0], v[1], v[0] / v[1]) for k, v in f["terms"].items())), flush=True)
        out["fits"].update(fits)
    for k in range(lab.max() + 1):
        m = lab != k
        try:
            p, cov = darkcomet_ngfit(eph, t0, p0, [1, 2], t_emit[m], obs_pos[m], ra[m], dec[m], sigma[m], radar)[:2]
        except np.linalg.LinAlgError:
            # the remaining apparitions do not determine the orbit with both terms
            print("  without %s (%d obs): A2 and A3 undetermined by the rest" % (out["apparitions"][k], (~m).sum()), flush=True)
            continue
        out["left_out"][out["apparitions"][k]] = dict(n=int((~m).sum()), A3=float(p[7]), A3_sigma=float(math.sqrt(cov[7, 7])))
        x = out["left_out"][out["apparitions"][k]]
        print("  without %s (%d obs): A3 %+.3e +- %.1e (%.1f sigma)" % (out["apparitions"][k], x["n"], x["A3"], x["A3_sigma"], x["A3"] / x["A3_sigma"]), flush=True)
    json.dump(out, open(os.path.join(d, "normal.json"), "w"), indent=1)
    return out


def darkcomet_normal_record(desig):
    """Record the normal-term refits: A3 with A2 and with all three terms, radar included, under both weightings, and the BIC margins."""
    d = darkcomet_dir(desig)
    nf = json.load(open(os.path.join(d, "normal.json")))
    f = nf["fits"]
    src = "radar" if nf["radar"] else "optical"

    def a3(key):
        t = f[key]["terms"]["A3"]
        return "%+.2e +- %.1e (%.1f sigma)" % (t[0], t[1], t[0] / t[1])

    def margin(scheme):
        b = {k.split("|")[2]: v["bic"] for k, v in f.items() if k.startswith("%s|%s|" % (scheme, src))}
        best_without = min(v for k, v in b.items() if "A3" not in k)
        return best_without - min(b["A2 A3"], b["A1 A2 A3"])

    loo = nf["left_out"]
    n3 = min(abs(v["A3"] / v["A3_sigma"]) for v in loo.values())
    lead = min(("none", "night"), key=lambda s: abs(f["%s|%s|A1 A2 A3" % (s, src)]["terms"]["A3"][0] / f["%s|%s|A1 A2 A3" % (s, src)]["terms"]["A3"][1]))
    lt = f["%s|%s|A1 A2 A3" % (lead, src)]["terms"]["A3"]
    run = "darkcomet_normal_" + desig.replace(" ", "_").lower()
    radar_txt = ("; %d radar delay%s fitted to %s us" % (len(nf["radar"]), "" if len(nf["radar"]) == 1 else "s",
                                                         ", ".join("%+.1f" % x for x in f["none|radar|A1 A2 A3"]["radar_residual_us"]))
                 if nf["radar"] else "")
    core.record_search(
        run, "Normal acceleration of a dark comet", archive="MPC observations; JPL SBDB, Horizons and radar astrometry",
        release="%d observations, %d radar delays" % (nf["n_obs"], len(nf["radar"])),
        selection=desig, code=os.path.abspath(__file__), entry="darkcomet", n_targets=1, n_unit="asteroids",
        params=dict(sigma_default=0.5, sigma_site=darkcomet_SIGMA, chi2_out=darkcomet_CHI2_OUT, g_r="r^-2", models=[" ".join(darkcomet_NG[j] for j in s) or "gravity" for s in darkcomet_NG_SETS]),
        null_summary={k: dict(chi2=v["chi2_common"], n=v["n_common"], bic=v["bic"]) for k, v in f.items()},
        sensitivity="A3 under site weights and with each night as one position, with and without the radar delays, and with each apparition left out",
        ledger="Bodies pushed out of their orbital planes",
        result="A3 with A1 and A2 free %s with site weights and %s with each night as one position; the BIC margin of a normal term is %+.0f and %+.0f, positive favouring it; "
               "with one apparition left out at a time A3 stays at %.1f sigma or more%s"
               % (a3("none|%s|A1 A2 A3" % src), a3("night|%s|A1 A2 A3" % src), margin("none"), margin("night"), n3, radar_txt))
    core.clear_targets(run)
    core.record_targets(run, [dict(target="mpc:" + desig, stat=lt[0] / lt[1], unit="A3 / sigma with A1 and A2 free, the less significant weighting",
                                   vet="dark-comet", note="A3 %+.2e au/d^2; Seligman et al. (2024) dark comet" % lt[0])])
    print("recorded %s" % run)


darkcomet_POOL = {}


def darkcomet_pool_init(state, ephem_dir):
    darkcomet_POOL.update(state)
    darkcomet_POOL["eph"] = darkcomet_ephem(ephem_dir)


def darkcomet_pool_fit(job):
    """One refit for darkcomet_robust: tag, rows, their sigmas, with A2, clipping threshold, starting parameters."""
    global darkcomet_CHI2_OUT
    tag, sel, sg, with_a2, clip, start = job
    darkcomet_CHI2_OUT = clip
    P = darkcomet_POOL
    p, cov, dra, ddec, kp = darkcomet_fit(P["eph"], P["t0"], start, with_a2, P["t_emit"][sel], P["obs_pos"][sel], P["ra"][sel],
                                          P["dec"][sel], sg, np.ones(len(sel), bool))
    return dict(tag=tag, A2=float(p[6]) if with_a2 else None, A2_sigma=float(math.sqrt(cov[6, 6])) if with_a2 else None,
                n=int(len(sel)), kept=int(kp.sum()), chi2=float(np.sum(((dra ** 2 + ddec ** 2) / sg ** 2)[kp])),
                dof=int(2 * kp.sum() - 6 - int(with_a2)), p=[float(x) for x in p], cov=cov.tolist())


def darkcomet_robust(desig, boot=200, workers=8, ephem_dir=None):
    """A2 left out one night at a time, nights resampled with and without the decisive apparition held, clipping and weighting variants, and both orbits' prediction ellipses at refit's prediction dates, into robust.json."""
    from multiprocessing import Pool
    from astropy.time import Time
    S = darkcomet_setup(desig, ephem_dir)
    rows, sigma, default, site_rms, lab, t0, p1 = (S[k] for k in ("rows", "sigma", "default", "site_rms", "lab", "t0", "p1"))
    rf = json.load(open(os.path.join(S["d"], "refit.json")))
    held_app = rf["apparitions"].index(rf["prediction"]["without"])
    site_of = np.array([r["site"] for r in rows])

    def weights(floor=0.15, dflt=None):
        sg = default.copy() if dflt is None else np.array([darkcomet_SIGMA.get(r["site"], dflt) for r in rows])
        for s, v in site_rms.items():
            sg[site_of == s] = max(v, floor)
        return sg

    night = np.floor(np.array([r["jd_utc"] for r in rows]) + 0.5).astype(int)
    nights = sorted(int(k) for k in set(night))
    held = sorted(int(k) for k in set(night[lab == held_app]))
    n_in = {k: int((night == k).sum()) for k in nights}
    nf = np.array([math.sqrt(n_in[k]) for k in night])
    idx = np.arange(len(rows))
    rest = idx[lab != held_app]
    jobs = [("base none", idx, sigma, True, 8.0, p1), ("base night", idx, sigma * nf, True, 8.0, p1),
            ("gravity rest none", rest, sigma[rest], False, 8.0, p1), ("gravity rest night", rest, (sigma * nf)[rest], False, 8.0, p1)]
    for k in nights:
        sel = idx[night != k]
        jobs += [("jack none %d" % k, sel, sigma[sel], True, 8.0, p1), ("jack night %d" % k, sel, (sigma * nf)[sel], True, 8.0, p1)]
    rng = np.random.default_rng(20261017)
    others = [k for k in nights if k not in held]
    for b in range(boot):
        draw = rng.choice(nights, size=len(nights), replace=True)
        sel = np.concatenate([idx[night == k] for k in draw])
        jobs.append(("boot free %d" % b, sel, (sigma * nf)[sel], True, 8.0, p1))
        draw = held + list(rng.choice(others, size=len(others), replace=True))
        sel = np.concatenate([idx[night == k] for k in draw])
        jobs.append(("boot held %d" % b, sel, (sigma * nf)[sel], True, 8.0, p1))
    variants = [("clip %g" % c, sigma, c) for c in (6.0, 12.0, 1e9)] + [("floor %.2f" % f, weights(floor=f), 8.0) for f in (0.10, 0.25)]
    variants += [("default %.1f" % v, weights(dflt=v), 8.0) for v in (0.3, 0.8)]
    for name, sg, clip in variants:
        jobs += [(name + " none", idx, sg, True, clip, p1), (name + " night", idx, sg * nf, True, clip, p1)]
    state = dict(t0=t0, t_emit=S["t_emit"], obs_pos=S["obs_pos"], ra=S["ra"], dec=S["dec"])
    print("%d positions, %d nights (%d held); %d fits on %d workers" % (len(rows), len(nights), len(held), len(jobs), workers), flush=True)
    with Pool(workers, initializer=darkcomet_pool_init, initargs=(state, ephem_dir)) as pool:
        res = {r["tag"]: r for r in pool.imap_unordered(darkcomet_pool_fit, jobs, chunksize=1)}
    out = dict(desig=desig, n_positions=len(rows), n_nights=len(nights), held=held, boot=boot,
               fits={k: {kk: vv for kk, vv in v.items() if kk not in ("p", "cov")} for k, v in res.items()})
    for scheme in ("none", "night"):
        v = np.array([res["jack %s %d" % (scheme, k)]["A2"] for k in nights if k not in held])
        n = len(v)
        se = float(math.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))
        base = res["base " + scheme]
        out["jackknife_" + scheme] = dict(A2=base["A2"], formal=base["A2_sigma"], se_rest=se, range_rest=[float(v.min()), float(v.max())])
        print("jackknife %-5s: A2 %+.3e formal %.2e, over the nights outside the held apparition %.2e (%.1f sigma), range %+.3e to %+.3e"
              % (scheme, base["A2"], base["A2_sigma"], se, abs(base["A2"]) / se, v.min(), v.max()))
    for kind in ("free", "held"):
        v = np.array([res["boot %s %d" % (kind, b)]["A2"] for b in range(boot)])
        s = np.array([res["boot %s %d" % (kind, b)]["A2_sigma"] for b in range(boot)])
        pct = [float(x) for x in np.percentile(v, [2.5, 16, 50, 84, 97.5])]
        sign = np.sign(v) == np.sign(res["base night"]["A2"])
        same = float(np.mean(sign))
        out["bootstrap_" + kind] = dict(percentiles=pct, frac_same_sign=same, frac_3sigma_same_sign=float(np.mean((np.abs(v / s) > 3) & sign)))
        print("bootstrap %s: 2.5/16/50/84/97.5%% %s; same sign %.0f%%, beyond 3 sigma with it %.0f%%"
              % (kind, " ".join("%+.2e" % x for x in pct), 100 * same, 100 * out["bootstrap_" + kind]["frac_3sigma_same_sign"]))
    for name in sorted({t.rsplit(" ", 1)[0] for t in res if t.startswith(("clip", "floor", "default"))}):
        for scheme in ("none", "night"):
            r = res["%s %s" % (name, scheme)]
            print("variant %-14s %-5s A2 %+.3e +- %.2e (%.1f sigma), %d kept, chi2/dof %.2f"
                  % (name, scheme, r["A2"], r["A2_sigma"], abs(r["A2"] / r["A2_sigma"]), r["kept"], r["chi2"] / r["dof"]))
    # both orbits seen from the geocentre at refit's prediction dates, with 1-sigma ellipses from their covariances
    eph = S["eph"]
    dates = [x["date"] for x in rf["prediction"]["rows"]]
    ts = Time(dates, scale="tdb").jd
    earth = np.array([[e.x, e.y, e.z] for e in (eph.get_particle("earth", t - eph.jd_ref) for t in ts)])
    hstep = np.array([1e-9] * 3 + [1e-11] * 3 + [1e-14])

    def sky(p):
        r_ = darkcomet_radec(darkcomet_propagate(eph, t0, np.array(p[:6]), p[6] if len(p) > 6 else 0.0, ts) - earth)
        return np.array(r_[0]), np.array(r_[1])

    def ellipses(p, cov):
        p = np.array(p)
        ra0, de0 = sky(p)
        cosd = np.cos(np.radians(de0))
        J = np.zeros((len(ts), 2, len(p)))
        for k in range(len(p)):
            q_ = p.copy()
            q_[k] += hstep[k]
            r1, d1 = sky(q_)
            J[:, 0, k] = ((r1 - ra0 + 180) % 360 - 180) * cosd * 3600 / hstep[k]
            J[:, 1, k] = (d1 - de0) * 3600 / hstep[k]
        return ra0, de0, np.array([J[i] @ np.array(cov) @ J[i].T for i in range(len(ts))])

    out["prediction"] = {}
    for scheme in ("none", "night"):
        a, g = res["base " + scheme], res["gravity rest " + scheme]
        raA, deA, CA = ellipses(a["p"], a["cov"])
        raG, deG, CG = ellipses(g["p"], g["cov"])
        rows_out = []
        for i, dt in enumerate(dates):
            dE = ((raA[i] - raG[i] + 180) % 360 - 180) * math.cos(math.radians(deA[i])) * 3600
            dN = (deA[i] - deG[i]) * 3600
            sep = math.hypot(dE, dN)
            u = np.array([dE, dN]) / max(sep, 1e-9)
            sA, sG = math.sqrt(u @ CA[i] @ u), math.sqrt(u @ CG[i] @ u)
            across = [math.sqrt(max(np.linalg.eigvalsh(C)[0], 0.0)) for C in (CA[i], CG[i])]
            rows_out.append(dict(date=dt, sep=sep, sigma_accel=sA, sigma_grav=sG, across_accel=across[0], across_grav=across[1],
                                 z_0p2=sep / math.sqrt(sA ** 2 + sG ** 2 + 0.2 ** 2)))
        out["prediction"][scheme] = rows_out
        for x in rows_out:
            if x["sep"] > 1.0:
                print("%s %-5s: %.2f arcsec apart; 1 sigma along it %.2f and %.2f, across %.2f and %.2f; a 0.2 arcsec position %.1f sigma"
                      % (x["date"], scheme, x["sep"], x["sigma_accel"], x["sigma_grav"], x["across_accel"], x["across_grav"], x["z_0p2"]))
    json.dump(out, open(os.path.join(S["d"], "robust.json"), "w"), indent=1)
    return out


def darkcomet_frames(desig, night, site="568", ephem_dir=None):
    """Each re-measured frame of one night against the orbit with A2 and the gravity-only orbit fitted without the decisive apparition, into frames_<night>.json."""
    S = darkcomet_setup(desig, ephem_dir)
    eph, rows, sigma, lab, t0, p0 = (S[k] for k in ("eph", "rows", "sigma", "lab", "t0", "p0"))
    rf = json.load(open(os.path.join(S["d"], "refit.json")))
    rest = lab != rf["apparitions"].index(rf["prediction"]["without"])
    pa = darkcomet_fit(eph, t0, p0, True, S["t_emit"], S["obs_pos"], S["ra"], S["dec"], sigma, np.ones(len(rows), bool))[0]
    pg = darkcomet_fit(eph, t0, p0, False, S["t_emit"][rest], S["obs_pos"][rest], S["ra"][rest], S["dec"][rest], sigma[rest],
                       np.ones(rest.sum(), bool))[0]
    tag = night.replace("-", "")
    found = [x for x in json.load(open(os.path.join(S["d"], "precovery_%s.json" % tag))) if x["found"]]
    ftdb, fpos = darkcomet_observers([dict(site=site, jd_utc=x["mjd_mid"] + 2400000.5) for x in found], S["sites"], eph)
    femit = ftdb - np.linalg.norm(darkcomet_propagate(eph, t0, pa[:6], pa[6], ftdb) - fpos, axis=1) * darkcomet_AU / 2.99792458e8 / 86400.0
    A = darkcomet_radec(darkcomet_propagate(eph, t0, pa[:6], pa[6], femit) - fpos)
    G = darkcomet_radec(darkcomet_propagate(eph, t0, pg[:6], 0.0, femit) - fpos)
    out = []
    for i, x in enumerate(found):
        cosd = math.cos(math.radians(x["dec"]))
        rec = dict(image=x["image"], ext=x["ext"], filter=x["filter"], mjd_mid=x["mjd_mid"], exptime=x["exptime"], snr=x["matched_peak_sigma"],
                   trail_arcsec=x["trail_px"] * x["scale"], ra=x["ra"], dec=x["dec"], wcs_rms=x["wcs_rms_arcsec"])
        for key, (r_, d_) in (("accel", (A[0][i], A[1][i])), ("grav", (G[0][i], G[1][i]))):
            rec[key] = dict(dE=float(((x["ra"] - r_ + 180) % 360 - 180) * cosd * 3600), dN=float((x["dec"] - d_) * 3600))
        out.append(rec)
        print("%s %s: from the orbit with A2 E %+.2f N %+.2f, from gravity only E %+.2f N %+.2f arcsec; trail %.2f arcsec, S/N %.1f"
              % (x["image"], x["filter"][:1], rec["accel"]["dE"], rec["accel"]["dN"], rec["grav"]["dE"], rec["grav"]["dN"], rec["trail_arcsec"], rec["snr"]))
    json.dump(out, open(os.path.join(S["d"], "frames_%s.json" % tag), "w"), indent=1)
    return out


def darkcomet_blockcov(desig, ephem_dir=None):
    """A2 with the positions of a night sharing an error: covariance s^2 sigma_site^2 plus tau^2 between any two positions of one night, in each coordinate, (s, tau) by restricted maximum likelihood over the positions the site-weighted fit keeps; A2 at the estimate and at the upper 95 per cent bound of tau, with the decisive apparition left out, and against an error shared by the decisive night alone, into blockcov.json."""
    S = darkcomet_setup(desig, ephem_dir)
    eph, rows, t0, p1, t_emit, obs_pos, ra, dec, sigma, lab = (S[k] for k in (
        "eph", "rows", "t0", "p1", "t_emit", "obs_pos", "ra", "dec", "sigma", "lab"))
    rf = json.load(open(os.path.join(S["d"], "refit.json")))
    held = lab == rf["apparitions"].index(rf["prediction"]["without"])
    p, cov0, _, _, keep = darkcomet_fit(eph, t0, p1, True, t_emit, obs_pos, ra, dec, sigma, np.ones(len(rows), bool))
    night = np.floor(np.array([r["jd_utc"] for r in rows]) + 0.5).astype(int)
    sel = np.nonzero(keep)[0]

    def cov_of(idx, s, tau, extra=0.0):
        same = (night[idx][:, None] == night[idx][None, :]).astype(float)
        C = s ** 2 * np.diag(sigma[idx] ** 2) + tau ** 2 * same
        if extra:
            h_ = held[idx].astype(float)
            C = C + extra ** 2 * np.outer(h_, h_)
        Z = np.zeros_like(C)
        return np.block([[C, Z], [Z, C]])

    def whiten(C, J, r):
        L = np.linalg.cholesky(C)
        A, b = np.linalg.solve(L, J), np.linalg.solve(L, r)
        sc = np.linalg.norm(A, axis=0)
        return L, A, b, sc

    def reml(C, J, r):
        L, A, b, sc = whiten(C, J, r)
        Q, R = np.linalg.qr(A / sc)
        e = b - Q @ (Q.T @ b)
        return -0.5 * (2 * np.sum(np.log(np.diag(L))) + 2 * np.sum(np.log(np.abs(np.diag(R)))) + 2 * np.sum(np.log(sc)) + e @ e)

    def solve(idx, q, s, tau, extra=0.0, iters=4):
        q = np.array(q, float)
        for _ in range(iters):
            r, J = darkcomet_jacobian(eph, t0, q, t_emit[idx], obs_pos[idx], ra[idx], dec[idx])
            _, A, b, sc = whiten(cov_of(idx, s, tau, extra), J, r)
            cov = np.linalg.inv((A / sc).T @ (A / sc)) / np.outer(sc, sc)
            dq = -cov @ (A.T @ b)
            q = q + dq
            if np.all(np.abs(dq) < 1e-3 * np.sqrt(np.diag(cov))):
                break
        return q, cov

    r, J = darkcomet_jacobian(eph, t0, p, t_emit[sel], obs_pos[sel], ra[sel], dec[sel])
    ss, tt = np.linspace(0.2, 2.5, 116), np.linspace(0.0, 1.0, 101)
    grid = np.array([[reml(cov_of(sel, s_, t_), J, r) for t_ in tt] for s_ in ss])
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    s_hat, tau_hat = float(ss[i]), float(tt[j])
    prof = grid.max(axis=0)
    inside = tt[2 * (prof.max() - prof) < 3.84]
    tau_lo, tau_hi = float(inside.min()), float(inside.max())
    s_at_hi = float(ss[np.argmax(grid[:, list(tt).index(inside.max())])])
    out = dict(desig=desig, n_kept=int(len(sel)), n_nights=int(len(set(night[sel]))), s_hat=s_hat, tau_hat=tau_hat, tau_95=[tau_lo, tau_hi],
               site_fit=dict(A2=float(p[6]), A2_sigma=float(math.sqrt(cov0[6, 6]))),
               profile_tau=[dict(tau=float(t_), loglik=float(v)) for t_, v in zip(tt, prof)])
    q, c = solve(sel, p, s_hat, tau_hat)
    out["at_estimate"] = dict(A2=float(q[6]), A2_sigma=float(math.sqrt(c[6, 6])))
    q2, c2 = solve(sel, p, s_at_hi, tau_hi)
    out["at_upper_tau"] = dict(s=s_at_hi, A2=float(q2[6]), A2_sigma=float(math.sqrt(c2[6, 6])))
    rest = np.array([k for k in sel if not held[k]])
    q3, c3 = solve(rest, p, s_hat, tau_hat)
    out["without_decisive"] = dict(A2=float(q3[6]), A2_sigma=float(math.sqrt(c3[6, 6])))
    print("%d kept positions on %d nights; restricted maximum likelihood s %.2f, tau %.3f arcsec (95 per cent %.3f-%.3f)"
          % (len(sel), out["n_nights"], s_hat, tau_hat, tau_lo, tau_hi))
    for k, lab_ in (("site_fit", "site weights alone"), ("at_estimate", "block covariance"), ("at_upper_tau", "tau at its upper bound"),
                    ("without_decisive", "block covariance, decisive apparition left out")):
        x = out[k]
        print("  %-48s A2 %+.3e +- %.2e (%.1f sigma)" % (lab_, x["A2"], x["A2_sigma"], abs(x["A2"]) / x["A2_sigma"]))
    scan = []
    for eps in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        qe, ce = solve(sel, q, s_hat, tau_hat, extra=eps)
        scan.append(dict(extra=eps, A2=float(qe[6]), A2_sigma=float(math.sqrt(ce[6, 6])), z=float(abs(qe[6]) / math.sqrt(ce[6, 6]))))
        print("  error shared by the decisive night %.2f arcsec: A2 %+.3e +- %.2e (%.1f sigma)" % (eps, qe[6], math.sqrt(ce[6, 6]), scan[-1]["z"]))
    out["decisive_shared_error"] = scan
    for zc in (5.0, 3.0):
        for a, b in zip(scan[:-1], scan[1:]):
            if a["z"] >= zc > b["z"]:
                out["shared_error_for_%g_sigma" % zc] = float(a["extra"] + (a["z"] - zc) / (a["z"] - b["z"]) * (b["extra"] - a["extra"]))
                print("  A2 falls to %g sigma at a shared error of %.2f arcsec" % (zc, out["shared_error_for_%g_sigma" % zc]))
                break
    json.dump(out, open(os.path.join(S["d"], "blockcov.json"), "w"), indent=1)
    return out


def darkcomet_locus(desig, night, site="568", ephem_dir=None):
    """Where the other apparitions put the object in one night's frames: the orbit fitted without the decisive apparition with A2 held at each value of a grid (the state at its conditional mean given A2 under that fit's covariance), the gravity-only orbit fitted to the same positions, and the orbit with A2 fitted to all of them; with A1 also free, the covariance of the predicted position; into locus_<night>.json."""
    S = darkcomet_setup(desig, ephem_dir)
    eph, rows, sigma, lab, t0, p1 = (S[k] for k in ("eph", "rows", "sigma", "lab", "t0", "p1"))
    rf = json.load(open(os.path.join(S["d"], "refit.json")))
    rest = lab != rf["apparitions"].index(rf["prediction"]["without"])
    args = (S["t_emit"][rest], S["obs_pos"][rest], S["ra"][rest], S["dec"][rest], sigma[rest], np.ones(rest.sum(), bool))
    pr, cr = darkcomet_fit(eph, t0, p1, True, *args)[:2]
    pr2, cr2 = darkcomet_fit(eph, t0, p1, True, *args, with_a1=True)[:2]
    pg = darkcomet_fit(eph, t0, p1, False, *args)[0]
    pa = darkcomet_fit(eph, t0, p1, True, S["t_emit"], S["obs_pos"], S["ra"], S["dec"], sigma, np.ones(len(rows), bool))[0]
    tag = night.replace("-", "")
    frames = json.load(open(os.path.join(S["d"], "precovery_%s.json" % tag)))
    ftdb, fpos = darkcomet_observers([dict(site=site, jd_utc=x["mjd_mid"] + 2400000.5) for x in frames], S["sites"], eph)
    femit = ftdb - np.linalg.norm(darkcomet_propagate(eph, t0, pa[:6], pa[6], ftdb) - fpos, axis=1) * darkcomet_AU / 2.99792458e8 / 86400.0
    vel = darkcomet_earth_velocity(eph, femit)

    def sky(q, dt=0.0):
        r_ = darkcomet_radec(darkcomet_propagate(eph, t0, q[:6], q[6] if len(q) > 6 else 0.0, femit + dt, a1=q[7] if len(q) > 7 else 0.0)
                            - (fpos + vel * dt))
        return np.array(r_[0]), np.array(r_[1])
    grid = np.round(np.arange(-16.0, 12.0001, 0.25), 2) * 1e-12
    cond = [pr + cr[:, 6] / cr[6, 6] * (v - pr[6]) for v in grid]
    loc = [sky(q) for q in cond]
    A, G = sky(pa), sky(pg)
    A1 = sky(pa, 1.0 / 1440)
    # predicted position of the mean frame time with A1 and A2 free: its covariance from that fit
    m = int(np.argmin(np.abs(femit - femit.mean())))
    hstep = np.array([1e-9] * 3 + [1e-11] * 3 + [1e-14, 1e-14])
    r0, d0 = sky(pr2)
    Jp = np.zeros((2, 8))
    for k in range(8):
        q = pr2.copy()
        q[k] += hstep[k]
        r1, d1 = sky(q)
        Jp[0, k] = ((r1[m] - r0[m] + 180) % 360 - 180) * math.cos(math.radians(d0[m])) * 3600 / hstep[k]
        Jp[1, k] = (d1[m] - d0[m]) * 3600 / hstep[k]
    Cp = Jp @ cr2 @ Jp.T
    ev, evec = np.linalg.eigh(Cp)
    out = dict(desig=desig, night=night, site=site, grid=[float(v) for v in grid],
               rest_A2=dict(A2=float(pr[6]), A2_sigma=float(math.sqrt(cr[6, 6]))),
               rest_A1A2=dict(A1=float(pr2[7]), A1_sigma=float(math.sqrt(cr2[7, 7])), A2=float(pr2[6]), A2_sigma=float(math.sqrt(cr2[6, 6]))),
               a1a2_ellipse=dict(frame=frames[m]["image"], major=float(math.sqrt(max(ev[1], 0))), minor=float(math.sqrt(max(ev[0], 0))),
                                 pa=float(math.degrees(math.atan2(evec[0, 1], evec[1, 1])) % 180),
                                 centre_from_grav=[float(((r0[m] - G[0][m] + 180) % 360 - 180) * math.cos(math.radians(G[1][m])) * 3600),
                                                   float((d0[m] - G[1][m]) * 3600)]),
               frames=[])
    for i, x in enumerate(frames):
        cosd = math.cos(math.radians(A[1][i]))
        out["frames"].append(dict(image=x["image"], ext=x["ext"], mjd_mid=x["mjd_mid"], exptime=x["exptime"], filter=x["filter"], found=x["found"],
                                  accel=[float(A[0][i]), float(A[1][i])], grav=[float(G[0][i]), float(G[1][i])],
                                  motion=[float(((A1[0][i] - A[0][i] + 180) % 360 - 180) * cosd * 3600), float((A1[1][i] - A[1][i]) * 3600)],
                                  locus=[[float(l_[0][i]), float(l_[1][i])] for l_ in loc]))
    f0 = out["frames"][m]
    z = grid.tolist().index(0.0)
    off = [((f0["locus"][z][0] - f0["grav"][0] + 180) % 360 - 180) * math.cos(math.radians(f0["grav"][1])) * 3600, (f0["locus"][z][1] - f0["grav"][1]) * 3600]
    out["locus_at_zero_from_grav"] = off
    step = [((f0["locus"][z + 4][0] - f0["locus"][z][0] + 180) % 360 - 180) * math.cos(math.radians(f0["grav"][1])) * 3600, (f0["locus"][z + 4][1] - f0["locus"][z][1]) * 3600]
    out["arcsec_per_1e-12"] = [float(step[0]), float(step[1])]
    print("without %s: A2 %+.3e +- %.2e; with A1 too, A1 %+.3e +- %.2e, A2 %+.3e +- %.2e" % (
        rf["prediction"]["without"], pr[6], math.sqrt(cr[6, 6]), pr2[7], math.sqrt(cr2[7, 7]), pr2[6], math.sqrt(cr2[6, 6])))
    print("locus at A2 = 0 lies E %+.2f N %+.2f arcsec from the gravity-only orbit; each 1e-12 of A2 moves it E %+.2f N %+.2f arcsec" % (*off, *step))
    print("with A1 free the predicted position at %s has a 1-sigma ellipse %.2f x %.2f arcsec at PA %.0f deg, centred E %+.2f N %+.2f from the gravity-only orbit"
          % (out["a1a2_ellipse"]["frame"], out["a1a2_ellipse"]["major"], out["a1a2_ellipse"]["minor"], out["a1a2_ellipse"]["pa"], *out["a1a2_ellipse"]["centre_from_grav"]))
    json.dump(out, open(os.path.join(S["d"], "locus_%s.json" % tag), "w"), indent=1)
    return out


def darkcomet_forecast(desig, ephem_dir=None, start="2030-06-01", days=183, sig_new=0.2):
    """Geocentric daily ephemeris from start of the orbit with A2 and of the gravity-only orbit fitted without the decisive apparition, one position a night, each with its 1-sigma ellipse; and the A2 uncertainty from the other apparitions plus one position a night accurate to sig_new at that apparition, into forecast.json and ephemeris_<year>.csv."""
    import csv
    from astropy.time import Time
    S = darkcomet_setup(desig, ephem_dir)
    eph, rows, sigma, lab, t0, p1, sb = (S[k] for k in ("eph", "rows", "sigma", "lab", "t0", "p1", "sb"))
    rf = json.load(open(os.path.join(S["d"], "refit.json")))
    rest = lab != rf["apparitions"].index(rf["prediction"]["without"])
    sg = sigma * darkcomet_night_factor(rows, "night")
    everything = np.ones(len(rows), bool)
    pa, ca, _, _, ka = darkcomet_fit(eph, t0, p1, True, S["t_emit"], S["obs_pos"], S["ra"], S["dec"], sg, everything)
    sel = np.nonzero(rest)[0]
    pg, cg = darkcomet_fit(eph, t0, p1, False, S["t_emit"][sel], S["obs_pos"][sel], S["ra"][sel], S["dec"][sel], sg[sel], np.ones(len(sel), bool))[:2]
    pr, _, _, _, kr = darkcomet_fit(eph, t0, p1, True, S["t_emit"][sel], S["obs_pos"][sel], S["ra"][sel], S["dec"][sel], sg[sel], np.ones(len(sel), bool))
    t_utc = Time(start, scale="utc").jd + np.arange(days)
    ts = Time(t_utc, format="jd", scale="utc").tdb.jd
    earth = np.array([[e.x, e.y, e.z] for e in (eph.get_particle("earth", t - eph.jd_ref) for t in ts)])
    sun = np.array([[s.x, s.y, s.z] for s in (eph.get_particle("sun", t - eph.jd_ref) for t in ts)])
    hstep = np.array([1e-9] * 3 + [1e-11] * 3 + [1e-14])
    H = num(next(x["value"] for x in sb["phys_par"] if x["name"] == "H"))

    def apparent(q):
        a2 = q[6] if len(q) > 6 else 0.0
        pos = darkcomet_propagate(eph, t0, q[:6], a2, ts)
        temit = ts - np.linalg.norm(pos - earth, axis=1) * darkcomet_AU / 2.99792458e8 / 86400.0
        pos = darkcomet_propagate(eph, t0, q[:6], a2, temit)
        r_, d_ = darkcomet_radec(pos - earth)
        return np.array(r_), np.array(d_), pos

    def with_ellipse(q, cov):
        ra0, de0, pos = apparent(q)
        cosd = np.cos(np.radians(de0))
        J = np.zeros((len(ts), 2, len(q)))
        for k in range(len(q)):
            qq = np.array(q, float)
            qq[k] += hstep[k]
            r1, d1, _ = apparent(qq)
            J[:, 0, k] = ((r1 - ra0 + 180) % 360 - 180) * cosd * 3600 / hstep[k]
            J[:, 1, k] = (d1 - de0) * 3600 / hstep[k]
        C = np.array([J[i] @ cov @ J[i].T for i in range(len(ts))])
        return ra0, de0, pos, C, J

    raA, deA, posA, CA, JA = with_ellipse(pa, ca)
    raG, deG, _, CG, _ = with_ellipse(pg, cg)
    r, dl = np.linalg.norm(posA - sun, axis=1), np.linalg.norm(posA - earth, axis=1)
    es = earth - sun
    elong = np.degrees(np.arccos(np.sum(-es * (posA - earth), axis=1) / np.linalg.norm(es, axis=1) / dl))
    ph = np.degrees(np.arccos(np.clip(np.sum((posA - sun) * (posA - earth), axis=1) / r / dl, -1, 1)))
    V = H + 5 * np.log10(r * dl) - 2.5 * np.log10([darkcomet_phase(a) for a in ph])

    def ell(C):
        ev, evec = np.linalg.eigh(C)
        return float(math.sqrt(max(ev[1], 0))), float(math.sqrt(max(ev[0], 0))), float(math.degrees(math.atan2(evec[0, 1], evec[1, 1])) % 180)
    table = []
    for i in range(len(ts)):
        dE = ((raA[i] - raG[i] + 180) % 360 - 180) * math.cos(math.radians(deA[i])) * 3600
        dN = (deA[i] - deG[i]) * 3600
        a_, b_ = ell(CA[i]), ell(CG[i])
        table.append(dict(date=Time(t_utc[i], format="jd").iso[:10], ra_accel=float(raA[i]), dec_accel=float(deA[i]), ra_grav=float(raG[i]),
                          dec_grav=float(deG[i]), sep=float(math.hypot(dE, dN)), sep_pa=float(math.degrees(math.atan2(dE, dN)) % 360),
                          accel_major=a_[0], accel_minor=a_[1], accel_pa=a_[2], grav_major=b_[0], grav_minor=b_[1], grav_pa=b_[2],
                          V=float(V[i]), r=float(r[i]), delta=float(dl[i]), elong=float(elong[i]), phase=float(ph[i])))
    year = start[:4]
    with open(os.path.join(S["d"], "ephemeris_%s.csv" % year), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date_0h_utc", "ra_accel_deg", "dec_accel_deg", "ra_grav_deg", "dec_grav_deg", "separation_arcsec", "separation_pa_deg",
                    "accel_sigma_major_arcsec", "accel_sigma_minor_arcsec", "accel_ellipse_pa_deg", "grav_sigma_major_arcsec",
                    "grav_sigma_minor_arcsec", "grav_ellipse_pa_deg", "V", "r_au", "delta_au", "elongation_deg", "phase_deg"])
        for x in table:
            w.writerow([x["date"], "%.6f" % x["ra_accel"], "%.6f" % x["dec_accel"], "%.6f" % x["ra_grav"], "%.6f" % x["dec_grav"],
                        "%.2f" % x["sep"], "%.1f" % x["sep_pa"], "%.3f" % x["accel_major"], "%.3f" % x["accel_minor"], "%.1f" % x["accel_pa"],
                        "%.3f" % x["grav_major"], "%.3f" % x["grav_minor"], "%.1f" % x["grav_pa"], "%.2f" % x["V"], "%.4f" % x["r"],
                        "%.4f" % x["delta"], "%.1f" % x["elong"], "%.1f" % x["phase"]])
    # A2 from the other apparitions alone, then with positions at this apparition: information added row by row
    r_, Jr = darkcomet_jacobian(eph, t0, pr, S["t_emit"][sel], S["obs_pos"][sel], S["ra"][sel], S["dec"][sel])
    w_ = np.concatenate([1.0 / sg[sel], 1.0 / sg[sel]]) * np.concatenate([kr, kr])
    I_rest = (Jr * w_[:, None] ** 2).T @ Jr
    r_, Ja = darkcomet_jacobian(eph, t0, pa, S["t_emit"], S["obs_pos"], S["ra"], S["dec"])
    wa = np.concatenate([1.0 / sg, 1.0 / sg]) * np.concatenate([ka, ka])
    I_all = (Ja * wa[:, None] ** 2).T @ Ja
    ok = [i for i, x in enumerate(table) if x["elong"] > 60 and x["V"] < 22.5]
    first = ok[0] if ok else 0
    scen = {"one night": [first], "three nights 30 days apart": [i for i in (first, first + 30, first + 60) if i < len(ts)],
            "six nights 15 days apart": [i for i in range(first, min(first + 76, len(ts)), 15)]}
    out = dict(desig=desig, weighting="one position a night", A2=float(pa[6]), A2_sigma=float(math.sqrt(ca[6, 6])), sig_new=sig_new,
               visible=[table[i]["date"] for i in (ok[0], ok[-1])] if ok else None, rows=table, scenarios={})
    base_rest = math.sqrt(np.linalg.inv(I_rest)[6, 6])
    base_all = math.sqrt(np.linalg.inv(I_all)[6, 6])
    print("one position a night: A2 %+.3e +- %.2e; without %s +- %.2e" % (pa[6], math.sqrt(ca[6, 6]), rf["prediction"]["without"], base_rest))
    for name, idx in scen.items():
        Jn = np.vstack([JA[i] for i in idx]) / sig_new
        s_rest = math.sqrt(np.linalg.inv(I_rest + Jn.T @ Jn)[6, 6])
        s_all = math.sqrt(np.linalg.inv(I_all + Jn.T @ Jn)[6, 6])
        out["scenarios"][name] = dict(dates=[table[i]["date"] for i in idx], A2_sigma_without=s_rest, z_without=abs(pa[6]) / s_rest,
                                      A2_sigma_with=s_all, z_with=abs(pa[6]) / s_all)
        print("  %s (%s), %.2f arcsec each: A2 +- %.2e without %s (%.1f sigma at the fitted value), +- %.2e with it"
              % (name, ", ".join(table[i]["date"] for i in idx), sig_new, s_rest, rf["prediction"]["without"], abs(pa[6]) / s_rest, s_all))
    out["A2_sigma_without_now"], out["A2_sigma_with_now"] = base_rest, base_all
    json.dump(out, open(os.path.join(S["d"], "forecast.json"), "w"), indent=1)
    return out


darkcomet_SKYBOT = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"


def darkcomet_field(desig, night, site="568", radius=0.5):
    """Known asteroids within radius degrees of one night's re-measured track (SkyBoT): density, motions, and the expected number of unrelated ones near the predicted position moving like the object, into field_<night>.json."""
    d = darkcomet_dir(desig)
    tag = night.replace("-", "")
    found = [x for x in json.load(open(os.path.join(d, "precovery_%s.json" % tag))) if x["found"] and math.hypot(*x["obs_minus_pred"]) < 1.0]
    t = np.array([x["mjd_mid"] for x in found])
    ra, dec = np.array([x["ra"] for x in found]), np.array([x["dec"] for x in found])
    cosd = math.cos(math.radians(dec.mean()))
    tm = (t - t.mean()) * 24.0
    ve, vn = np.polyfit(tm, (ra - ra.mean()) * cosd * 3600, 1)[0], np.polyfit(tm, (dec - dec.mean()) * 3600, 1)[0]
    rate, pa = math.hypot(ve, vn), math.degrees(math.atan2(ve, vn)) % 360
    # our detections and the MPC reports of the night against that straight track, across the motion
    ux, uy = ve / rate, vn / rate
    e0, n0 = np.polyfit(tm, (ra - ra.mean()) * cosd * 3600, 1)[1], np.polyfit(tm, (dec - dec.mean()) * 3600, 1)[1]

    def across(jd, r_, d_):
        h = (jd - 2400000.5 - t.mean()) * 24.0
        de_, dn_ = (r_ - ra.mean()) * cosd * 3600 - (e0 + ve * h), (d_ - dec.mean()) * 3600 - (n0 + vn * h)
        return -de_ * uy + dn_ * ux
    ours = np.array([across(x["mjd_mid"] + 2400000.5, x["ra"], x["dec"]) for x in found])
    mpc = [r for r in darkcomet_parse(open(os.path.join(d, "obs80.txt")).read()) if abs(r["jd_utc"] - 2400000.5 - t.mean()) < 0.5]
    rep = np.array([across(r["jd_utc"], r["ra"], r["dec"]) for r in mpc])
    q = {"-ep": "%.5f" % (t.mean() + 2400000.5), "-ra": "%.5f" % ra.mean(), "-dec": "%.5f" % dec.mean(), "-rd": "%g" % radius,
         "-mime": "json", "-output": "all", "-loc": site, "-filter": "0", "-objFilter": "100", "-refsys": "EQJ2000", "-from": "setitools"}
    rows = json.loads(core.get(darkcomet_SKYBOT + "?" + urllib.parse.urlencode(q), timeout=180))
    objs = []
    for r in rows:
        h = [float(v) for v in r["RA (hms)"].split()]
        g = r["DEC (dms)"].split()
        sg = -1.0 if g[0].startswith("-") else 1.0
        o_ra = 15.0 * (h[0] + h[1] / 60 + h[2] / 3600)
        o_dec = sg * (abs(float(g[0])) + float(g[1]) / 60 + float(g[2]) / 3600)
        dra, ddec = float(r["dRA (arcsec/h)"]), float(r["dDEC (arcsec/h)"])
        objs.append(dict(name=r["Name"], cls=r["Class"], V=float(r["VMag (mag)"]), rate=math.hypot(dra, ddec),
                         pa=math.degrees(math.atan2(dra, ddec)) % 360,
                         sep=math.hypot((o_ra - ra.mean()) * cosd, o_dec - dec.mean()) * 3600))
    name = desig.replace(" ", "")
    others = [o for o in objs if name not in o["name"].replace(" ", "")]
    dens = len(others) / (math.pi * radius ** 2)
    r_ = np.array([o["rate"] for o in others])
    p_ = (np.array([o["pa"] for o in others]) - pa + 180) % 360 - 180
    # motions near the object's from a normal fit to the known ones: within 0.5 arcsec/h and 1 degree
    f = (math.exp(-0.5 * ((rate - r_.mean()) / r_.std(ddof=1)) ** 2) / (r_.std(ddof=1) * math.sqrt(2 * math.pi)) * 1.0
         * math.exp(-0.5 * (p_.mean() / p_.std(ddof=1)) ** 2) / (p_.std(ddof=1) * math.sqrt(2 * math.pi)) * 2.0)
    out = dict(track=dict(mjd=float(t.mean()), ra=float(ra.mean()), dec=float(dec.mean()), rate=rate, pa=pa, n=len(found),
                          ours_across_rms=float(np.sqrt(np.mean(ours ** 2)))),
               reports=dict(n=len(rep), across_mean=float(rep.mean()), across_rms=float(np.sqrt(np.mean(rep ** 2))),
                            by_site={s: float(np.mean([a for a, r in zip(rep, mpc) if r["site"] == s])) for s in sorted({r["site"] for r in mpc})}),
               n_known=len(others), density_per_deg2=dens, nearest_arcsec=min(o["sep"] for o in others),
               known_rate_mean=float(r_.mean()), known_rate_sd=float(r_.std(ddof=1)), known_pa_mean=float((p_.mean() + pa) % 360),
               known_pa_sd=float(p_.std(ddof=1)),
               n_within_5_10=int(np.sum((np.abs(r_ - rate) < 5) & (np.abs(p_) < 10))),
               n_within_10_20=int(np.sum((np.abs(r_ - rate) < 10) & (np.abs(p_) < 20))), motion_fraction=f,
               expected_10x_30arcsec=10 * dens * math.pi * (30 / 3600) ** 2 * f, objects=objs)
    print("our %d detections scatter %.2f arcsec across the track; the %d MPC reports of the night lie %.2f rms (mean %+.2f) across it"
          % (len(found), out["track"]["ours_across_rms"], len(rep), out["reports"]["across_rms"], out["reports"]["across_mean"]))
    print("track %.1f arcsec/h toward PA %.0f deg; %d known asteroids besides %s within %g deg (%.1f per square degree), nearest %.0f arcsec; "
          "their motions %.1f +- %.1f arcsec/h toward %.0f +- %.0f deg; %d within 5 arcsec/h and 10 deg, %d within 10 and 20; "
          "at ten times their density, %.1e unrelated within 30 arcsec moving within 0.5 arcsec/h and 1 deg"
          % (rate, pa, len(others), desig, radius, dens, out["nearest_arcsec"], out["known_rate_mean"], out["known_rate_sd"],
             out["known_pa_mean"], out["known_pa_sd"], out["n_within_5_10"], out["n_within_10_20"], out["expected_10x_30arcsec"]))
    json.dump(out, open(os.path.join(d, "field_%s.json" % tag), "w"), indent=1)
    return out


darkcomet_CENSUS = {}


def darkcomet_poly(c, x, y):
    """The detector-to-WCS-pixel map: affine with three coefficients, quadratic in pixels over 1000 with six."""
    v = c[0] * x + c[1] * y + c[2]
    if len(c) > 3:
        u, w_ = x / 1000.0, y / 1000.0
        v = v + c[3] * u * u + c[4] * u * w_ + c[5] * w_ * w_
    return v


def darkcomet_register(img, raw, w, g_ra, g_dec, thresh):
    """Map from detector pixels to the header WCS's pixels from bright peaks of img, centred on raw, matched to Gaia: quadratic with twenty or more matches, else affine; x and y coefficients, rms (px), and the matched peaks' positions with their Gaia indices; None with fewer than six."""
    from scipy.ndimage import maximum_filter
    pk = np.argwhere((img == maximum_filter(img, size=9)) & (img > thresh))
    pk = pk[(pk[:, 0] > 3) & (pk[:, 0] < img.shape[0] - 4) & (pk[:, 1] > 3) & (pk[:, 1] < img.shape[1] - 4)]
    if len(pk) < 6:
        return None
    sx, sy = np.zeros(len(pk)), np.zeros(len(pk))
    for n_, (y, x) in enumerate(pk):
        b = np.clip(raw[y - 2:y + 3, x - 2:x + 3], 0, None)
        s = b.sum()
        sx[n_] = x + ((b.sum(axis=0) @ np.arange(-2, 3)) / s if s > 0 else 0.0)
        sy[n_] = y + ((b.sum(axis=1) @ np.arange(-2, 3)) / s if s > 0 else 0.0)
    gx, gy = w.all_world2pix(g_ra, g_dec, 0)
    gi = np.nonzero((gx > 5) & (gx < img.shape[1] - 5) & (gy > 5) & (gy < img.shape[0] - 5))[0]
    dxs, dys = [], []
    for i in gi:
        dd = np.hypot(sx - gx[i], sy - gy[i])
        for j in np.nonzero(dd < 40)[0]:
            dxs.append(sx[j] - gx[i])
            dys.append(sy[j] - gy[i])
    if len(dxs) < 6:
        return None
    H2, xe, ye = np.histogram2d(dxs, dys, bins=80, range=[[-40, 40], [-40, 40]])
    k = np.unravel_index(np.argmax(H2), H2.shape)
    sh = (0.5 * (xe[k[0]] + xe[k[0] + 1]), 0.5 * (ye[k[1]] + ye[k[1] + 1]))
    pairs = []
    for i in gi:
        dd = np.hypot(sx - gx[i] - sh[0], sy - gy[i] - sh[1])
        j = int(np.argmin(dd))
        if dd[j] < 3.0:
            pairs.append((sx[j], sy[j], gx[i], gy[i], i))
    pairs = np.array(pairs)
    if len(pairs) < 6:
        return None

    def design(p_):
        cols = [p_[:, 0], p_[:, 1], np.ones(len(p_))]
        if len(p_) >= 20:
            u, w_ = p_[:, 0] / 1000.0, p_[:, 1] / 1000.0
            cols += [u * u, u * w_, w_ * w_]
        return np.column_stack(cols)
    for _ in range(3):
        A = design(pairs)
        cx = np.linalg.lstsq(A, pairs[:, 2], rcond=None)[0]
        cy = np.linalg.lstsq(A, pairs[:, 3], rcond=None)[0]
        res = np.hypot(A @ cx - pairs[:, 2], A @ cy - pairs[:, 3])
        keep = res < max(3 * np.median(res), 0.3)
        pairs = pairs[keep]
    A = design(pairs)
    cx = np.linalg.lstsq(A, pairs[:, 2], rcond=None)[0]
    cy = np.linalg.lstsq(A, pairs[:, 3], rcond=None)[0]
    res = np.hypot(A @ cx - pairs[:, 2], A @ cy - pairs[:, 3])
    return cx, cy, float(np.sqrt(np.mean(res ** 2))), pairs


def darkcomet_sky(w, cx, cy, x, y):
    """RA and Dec (deg) of detector pixels through the registration map and the header WCS."""
    r_, d_ = w.all_pix2world(darkcomet_poly(cx, x, y), darkcomet_poly(cy, x, y), 0)
    return np.asarray(r_, float), np.asarray(d_, float)


def darkcomet_pix(w, cx, cy, ra, dec):
    """Detector pixels of RA and Dec (deg), inverting darkcomet_sky by Newton steps from its linear part."""
    gx, gy = (np.asarray(v, float) for v in w.all_world2pix(ra, dec, 0))
    M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
    x, y = np.linalg.solve(M, np.vstack([gx - cx[2], gy - cy[2]]))
    if len(cx) > 3:
        for _ in range(4):
            fx, fy = darkcomet_poly(cx, x, y) - gx, darkcomet_poly(cy, x, y) - gy
            u, v = x / 1000.0, y / 1000.0
            a11 = cx[0] + (2 * cx[3] * u + cx[4] * v) / 1000.0
            a12 = cx[1] + (cx[4] * u + 2 * cx[5] * v) / 1000.0
            a21 = cy[0] + (2 * cy[3] * u + cy[4] * v) / 1000.0
            a22 = cy[1] + (cy[4] * u + 2 * cy[5] * v) / 1000.0
            det = a11 * a22 - a12 * a21
            x, y = x - (a22 * fx - a12 * fy) / det, y - (-a21 * fx + a11 * fy) / det
    return x, y


def darkcomet_trail(dx, dy, sigma):
    """A trail of length (dx, dy) pixels blurred by a Gaussian of sigma pixels, normalised to unit sum."""
    from scipy.ndimage import gaussian_filter
    L = math.hypot(dx, dy)
    half = int(math.ceil(L / 2 + 4 * sigma)) + 1
    K = np.zeros((2 * half + 1, 2 * half + 1))
    for s_ in np.linspace(-0.5, 0.5, max(int(L * 4), 2)):
        x_, y_ = half + s_ * dx, half + s_ * dy
        i0, j0 = int(math.floor(y_)), int(math.floor(x_))
        fy, fx = y_ - i0, x_ - j0
        K[i0, j0] += (1 - fy) * (1 - fx)
        K[i0, j0 + 1] += (1 - fy) * fx
        K[i0 + 1, j0] += fy * (1 - fx)
        K[i0 + 1, j0 + 1] += fy * fx
    K = gaussian_filter(K, sigma)
    return K / K.sum()


def darkcomet_census_ccd(job):
    """One CCD of one frame for darkcomet_census: background, Gaia registration, seeing and zero point, the injected tracks added, sources found with a point-source filter and with a filter trailed by the object's motion, and that filter's S/N at the requested positions."""
    quiet_warnings()
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates, maximum_filter, median_filter, zoom
    from scipy.signal import fftconvolve
    fi, ext, do_inject = job
    St = darkcomet_CENSUS
    fr = St["frames"][fi]
    with fits.open(fr["path"]) as h:
        hd = h[ext].header
        img = np.array(h[ext].data, dtype=np.float32)
    w = WCS(hd)
    ny, nx = img.shape
    bad = ~np.isfinite(img) | (img > 0.9 * float(hd.get("SATURATE", 65000)))
    img = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
    B = np.array([[np.median(img[y:y + 64, x:x + 64]) for x in range(0, nx, 64)] for y in range(0, ny, 64)], dtype=np.float32)
    sub = img - zoom(B, (ny / B.shape[0], nx / B.shape[1]), order=1)[:ny, :nx]
    bad = binary_dilation(bad, iterations=4)
    edge = np.zeros_like(bad)
    edge[:12], edge[-12:], edge[:, :12], edge[:, -12:] = True, True, True, True
    good = ~(bad | edge)
    sig = 1.4826 * float(np.median(np.abs(sub[good][::53])))
    out = dict(fi=fi, ext=ext, ok=False)
    fA = gaussian_filter(sub, 2.0)
    sA = 1.4826 * float(np.median(np.abs(fA[good][::53])))
    reg = darkcomet_register(np.where(good, fA, 0.0), np.where(good, sub, 0.0), w, St["g_ra"], St["g_dec"], 25 * sA)
    if reg is None:
        return out
    cx, cy, rms, pairs = reg
    scale = math.sqrt(abs(np.linalg.det(np.array([[cx[0], cx[1]], [cy[0], cy[1]]]) @ w.pixel_scale_matrix[:2, :2]))) * 3600
    # seeing and zero point from Gaia stars: second moments and Gaussian-profile fluxes
    sizes, zps = [], []
    for x_, y_, _, _, gi in pairs:
        i0, j0 = int(round(y_)), int(round(x_))
        if i0 < 8 or j0 < 8 or i0 > ny - 9 or j0 > nx - 9 or not good[i0, j0]:
            continue
        st = sub[i0 - 7:i0 + 8, j0 - 7:j0 + 8].astype(float)
        if st.max() < 20 * sig or st.max() > 0.5 * float(hd.get("SATURATE", 65000)):
            continue
        yy, xx = np.indices(st.shape) - 7.0
        wt = np.clip(st, 0, None)
        mx, my = (wt * xx).sum() / wt.sum(), (wt * yy).sum() / wt.sum()
        sizes.append(math.sqrt(max(((wt * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / wt.sum()) / 2, 0.25)))
        zps.append((float(St["g_mag"][int(gi)]), st))
    if len(sizes) < 3:
        return out
    # second moments within a 15-pixel box underestimate the Gaussian sigma; corrected by the same box on a Gaussian
    s_box = float(np.median(sizes))
    grid_ = np.indices((15, 15)) - 7.0
    for s_try in np.arange(0.8, 6.0, 0.02):
        g_ = np.exp(-0.5 * (grid_[0] ** 2 + grid_[1] ** 2) / s_try ** 2)
        if math.sqrt((g_ * (grid_[0] ** 2 + grid_[1] ** 2)).sum() / g_.sum() / 2) >= s_box:
            break
    psf = float(s_try)
    P = np.exp(-0.5 * (grid_[0] ** 2 + grid_[1] ** 2) / psf ** 2)
    P /= P.sum()
    zp = [g + 2.5 * math.log10(max((P * st).sum() / (P * P).sum(), 1e-9)) for g, st in zps if (P * st).sum() > 0 and 15 < g < 20.5]
    if len(zp) < 3:
        return out
    zp = float(np.median(zp))
    t_start, t_end = fr["t_mid"] - fr["exptime"] / 7200.0, fr["t_mid"] + fr["exptime"] / 7200.0

    def at(ra0, dec0, ve, vn, t):
        return ra0 + ve * t / 3600.0 / math.cos(math.radians(dec0)), dec0 + vn * t / 3600.0
    injected = []
    if do_inject:
        rng = np.random.default_rng(St["seed"] + 1000 * fi + ext)
        for k, (ra0, dec0, ve, vn, dm) in enumerate(St["inject"]):
            ra1, de1 = at(ra0, dec0, ve, vn, t_start)
            ra2, de2 = at(ra0, dec0, ve, vn, t_end)
            x1, y1 = darkcomet_pix(w, cx, cy, np.array([ra1, ra2]), np.array([de1, de2]))
            if not (20 < x1.min() and x1.max() < nx - 20 and 20 < y1.min() and y1.max() < ny - 20):
                continue
            F = 10 ** (0.4 * (zp - St["mag"][fr["band"]] - dm))
            K = darkcomet_trail(x1[1] - x1[0], y1[1] - y1[0], psf) * F
            hk = K.shape[0] // 2
            xc, yc = int(round(0.5 * (x1[0] + x1[1]))), int(round(0.5 * (y1[0] + y1[1])))
            if hk < yc < ny - hk - 1 and hk < xc < nx - hk - 1:
                gain = float(hd.get("GAIN", 1.6))
                sub[yc - hk:yc + hk + 1, xc - hk:xc + hk + 1] += (K + rng.normal(0, 1, K.shape) * np.sqrt(np.clip(K, 0, None) / gain)).astype(np.float32)
                injected.append(k)
    # the object's motion over this exposure in this CCD's pixels
    ra_c, de_c = darkcomet_sky(w, cx, cy, np.array([nx / 2.0]), np.array([ny / 2.0]))
    ve, vn = St["motion"]
    q1 = at(ra_c[0], de_c[0], ve, vn, -fr["exptime"] / 7200.0)
    q2 = at(ra_c[0], de_c[0], ve, vn, fr["exptime"] / 7200.0)
    mx_, my_ = darkcomet_pix(w, cx, cy, np.array([q1[0], q2[0]]), np.array([q1[1], q2[1]]))
    KT = darkcomet_trail(mx_[1] - mx_[0], my_[1] - my_[0], psf)
    norm = math.sqrt((KT ** 2).sum())
    # cosmic rays: pixels far above the median of their 5x5 neighbourhood, replaced by it
    med5 = median_filter(sub, size=5)
    cr = binary_dilation((sub > 4 * sig) & (med5 < 0.2 * sub), iterations=1)
    sub = np.where(cr, med5, sub)
    fA = gaussian_filter(sub, 2.0)
    zA = fA / sA
    fB = fftconvolve(sub, (KT / norm)[::-1, ::-1], mode="same").astype(np.float32)
    sB = 1.4826 * float(np.median(np.abs(fB[good][::53])))
    zB = fB / sB

    def peaks(z, size, flux):
        """Every local maximum above the catalogue floor: RA, Dec, S/N and magnitude arrays."""
        pk = np.argwhere((z == maximum_filter(z, size=size)) & (z > St["floor"]) & good)
        y0, x0 = pk[:, 0], pk[:, 1]

        def refine(a, b, c):
            dd = a - 2 * b + c
            return np.where(dd != 0, 0.5 * (a - c) / np.where(dd != 0, dd, 1), 0.0)
        xs = x0 + np.clip(refine(z[y0, x0 - 1], z[y0, x0], z[y0, x0 + 1]), -1, 1)
        ys = y0 + np.clip(refine(z[y0 - 1, x0], z[y0, x0], z[y0 + 1, x0]), -1, 1)
        r_, d_ = darkcomet_sky(w, cx, cy, xs, ys)
        f_ = flux(y0, x0)
        return dict(ra=r_, dec=d_, snr=z[y0, x0].astype(np.float32),
                    mag=np.where(f_ > 0, zp - 2.5 * np.log10(np.clip(f_, 1e-9, None)), np.nan).astype(np.float32))
    # fluxes: a point source of flux F peaks at F / (2 pi (psf^2 + 4)) in the 2-pixel Gaussian filter, and a trail matching the kernel
    # gives F times the kernel's norm in the trailed filter
    nA = 2 * math.pi * (psf ** 2 + 4.0)
    corners = darkcomet_sky(w, cx, cy, np.array([12.0, nx - 13.0, nx - 13.0, 12.0]), np.array([12.0, 12.0, ny - 13.0, ny - 13.0]))
    out.update(ok=True, zp=zp, psf=psf, fwhm_arcsec=2.3548 * psf * scale, scale=scale, wcs_rms_arcsec=rms * scale, n_gaia=int(len(pairs)),
               sigma=sig, n_cosmic=int(cr.sum()), injected=injected, corners=[corners[0].tolist(), corners[1].tolist()],
               A=peaks(zA, 7, lambda y0, x0: fA[y0, x0] * nA), B=peaks(zB, 9, lambda y0, x0: fB[y0, x0] / norm))
    # trailed-filter S/N at requested positions on this CCD
    pts = St["points"].get(fi)
    if pts is not None:
        px, py = darkcomet_pix(w, cx, cy, pts[:, 0], pts[:, 1])
        inside = np.nonzero((px > 14) & (px < nx - 15) & (py > 14) & (py < ny - 15))[0]
        if len(inside):
            okp = good[np.round(py[inside]).astype(int), np.round(px[inside]).astype(int)]
            inside = inside[okp]
            out["samples"] = dict(idx=inside.tolist(), z=map_coordinates(zB, [py[inside], px[inside]], order=1).astype(float).tolist())
    return out


def darkcomet_link(t, x, y, f, tol=0.6, min_n=6, vmin=4.0, vmax=80.0, dt_min=0.3, dt_max=0.7):
    """Straight tracks through detections at times t (h) and positions x, y (arcsec) in frames f: each pair of detections dt_min to dt_max apart with a speed between vmin and vmax arcsec/h seeds a track kept when min_n frames have a detection within tol of it, refitted once; member indices of each distinct track."""
    from scipy.spatial import cKDTree
    fr = sorted(set(f.tolist()))
    tf = {k: float(t[f == k][0]) for k in fr}
    idx = {k: np.nonzero(f == k)[0] for k in fr}
    tree = {k: cKDTree(np.column_stack([x[idx[k]], y[idx[k]]])) for k in fr if len(idx[k])}
    found, seen = [], set()

    def gather(x0, y0, vx, vy, t0_, need=0):
        """Detections within tol of each straight track, frame by frame, for arrays of tracks: member index lists of the tracks with at least need."""
        hits = []
        for k in tree:
            d, m = tree[k].query(np.column_stack([x0 + vx * (tf[k] - t0_), y0 + vy * (tf[k] - t0_)]), distance_upper_bound=tol)
            ok_ = np.isfinite(d)
            hits.append((ok_, np.where(ok_, idx[k][np.minimum(m, len(idx[k]) - 1)], -1)))
        count = np.sum([h_[0] for h_ in hits], axis=0)
        return {int(s_): [int(h_[1][s_]) for h_ in hits if h_[0][s_]] for s_ in np.nonzero(count >= need)[0]}
    for a in fr:
        for b in fr:
            dt = tf[b] - tf[a]
            if not dt_min <= dt <= dt_max or a not in tree or b not in tree:
                continue
            near = tree[b].query_ball_point(np.column_stack([x[idx[a]], y[idx[a]]]), vmax * dt)
            ii = np.array([i for i, js in zip(idx[a], near) for _ in js], int)
            jj = np.array([idx[b][j] for js in near for j in js], int)
            if not len(ii):
                continue
            vx, vy = (x[jj] - x[ii]) / dt, (y[jj] - y[ii]) / dt
            fast = np.hypot(vx, vy) >= vmin
            ii, vx, vy = ii[fast], vx[fast], vy[fast]
            for mem in gather(x[ii], y[ii], vx, vy, tf[a], need=min_n).values():
                m_ = np.array(mem)
                px_, py_ = np.polyfit(t[m_], x[m_], 1), np.polyfit(t[m_], y[m_], 1)
                mem = gather(np.array([px_[1]]), np.array([py_[1]]), np.array([px_[0]]), np.array([py_[0]]), 0.0).get(0, [])
                if len(mem) < min_n:
                    continue
                key = tuple(sorted(mem))
                if key not in seen:
                    seen.add(key)
                    found.append(key)
    # a track whose members are mostly another's is the same object
    found.sort(key=len, reverse=True)
    out = []
    for key in found:
        s_ = set(key)
        if all(len(s_ & set(o)) < 0.5 * len(s_) for o in out):
            out.append(key)
    return [list(o) for o in out]


def darkcomet_census(desig, night, workers=6, inject=300, seed=20070914, thresh=5.0, floor=3.0, only=None, reuse=False):
    """Every moving source in one night's CFHT MegaCam frames of at least 300 s: whole frames registered CCD by CCD to Gaia DR3, sources found with a point-source filter and with a filter trailed by the object's motion, those recurring at one position removed, the rest linked into straight tracks of six or more frames; copies of the object injected along random tracks near its motion measure the recovery; the trailed filter's S/N along the positions locus_<night>.json allows, at the gravity-only position and along shifted control loci; into census_<night>.json."""
    quiet_warnings()
    from multiprocessing import Pool
    from astropy.io import fits
    from astropy.time import Time
    d = darkcomet_dir(desig)
    tag = night.replace("-", "")
    full = os.path.join(d, "cfht_" + tag, "full")
    rows = [r for r in json.load(open(os.path.join(d, "images.json"))) if r["Telescope/Instrument"] == "CFHT/MegaCam"
            and Time(float(r["MJD"]), format="mjd").iso[:10] == night and float(r["Exptime"]) >= 300]
    names = sorted({r["Image"] for r in rows})
    frames = []
    for name in names:
        path = os.path.join(full, name + ".fits.fz")
        if not os.path.exists(path):
            print("%s: missing, skipped" % path)
            continue
        h0 = fits.getheader(path, 0)
        t_mid = Time("%sT%s" % (h0["DATE-OBS"], h0["UTC-OBS"]), scale="utc").mjd + h0["EXPTIME"] / 2.0 / 86400.0
        frames.append(dict(image=name, path=path, mjd_mid=float(t_mid), exptime=float(h0["EXPTIME"]), band=h0["FILTER"][0].lower(),
                           n_ext=len(fits.open(path))))
    t_ref = float(np.mean([f_["mjd_mid"] for f_ in frames]))
    for f_ in frames:
        f_["t_mid"] = (f_["mjd_mid"] - t_ref) * 24.0
    found = [x for x in json.load(open(os.path.join(d, "precovery_%s.json" % tag))) if x["found"]]
    tt = np.array([(x["mjd_mid"] - t_ref) * 24.0 for x in found])
    ra_o, de_o = np.array([x["ra"] for x in found]), np.array([x["dec"] for x in found])
    ra0, de0 = float(np.mean(ra_o)), float(np.mean(de_o))
    cos0 = math.cos(math.radians(de0))
    ve = float(np.polyfit(tt, (ra_o - ra0) * cos0 * 3600, 1)[0])
    vn = float(np.polyfit(tt, (de_o - de0) * 3600, 1)[0])
    gaia = core.datalab("SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, %.5f, %.5f, 0.85) = 't' "
                        "AND phot_g_mean_mag < 21" % (ra0, de0))
    dt_y = Time(t_ref, format="mjd").jyear - 2016.0
    # stars without a proper motion keep their catalogue position
    pm = {k: np.nan_to_num(np.array([num(g[k]) if num(g[k]) is not None else 0.0 for g in gaia], float)) for k in ("pmra", "pmdec")}
    g_ra = np.array([float(g["ra"]) for g in gaia]) + pm["pmra"] / 3.6e6 * dt_y / np.cos(np.radians([float(g["dec"]) for g in gaia]))
    g_dec = np.array([float(g["dec"]) for g in gaia]) + pm["pmdec"] / 3.6e6 * dt_y
    g_mag = np.array([float(g["phot_g_mean_mag"]) for g in gaia])
    loc = json.load(open(os.path.join(d, "locus_%s.json" % tag)))
    lf = {x["image"]: x for x in loc["frames"] if x["found"]}
    # positions sampled with the trailed filter: the locus, the two orbits, the detection, a map around the gravity-only orbit, control loci
    grid = np.array(loc["grid"])
    off_e, off_n = np.meshgrid(np.arange(-24.0, 24.01, 0.5), np.arange(-24.0, 24.01, 0.5))
    shifts = [(e, n) for e in np.arange(-150.0, 150.1, 10.0) for n in np.arange(-150.0, 150.1, 6.0) if math.hypot(e, n) > 30.0]
    points, layout = {}, None
    for fi, f_ in enumerate(frames):
        x = lf.get(f_["image"])
        if x is None:
            continue
        L = np.array(x["locus"])
        g_ = np.array(x["grav"])
        det = next(y for y in found if y["image"] == f_["image"])
        cosd = math.cos(math.radians(g_[1]))
        mp = np.column_stack([g_[0] + off_e.ravel() / 3600 / cosd, g_[1] + off_n.ravel() / 3600])
        ctl = np.vstack([np.column_stack([L[:, 0] + e / 3600 / cosd, L[:, 1] + n / 3600]) for e, n in shifts])
        points[fi] = np.vstack([L, [g_], [x["accel"]], [[det["ra"], det["dec"]]], mp, ctl])
        layout = dict(locus=len(L), map=len(mp), map_shape=list(off_e.shape), shifts=len(shifts))
    # the object's brightness in each band from its own frames, before anything is injected
    rng = np.random.default_rng(seed)
    state = dict(frames=frames, g_ra=g_ra, g_dec=g_dec, g_mag=g_mag, inject=[], mag={}, seed=seed, motion=(ve, vn), floor=floor, points={})
    darkcomet_census_init(state)
    own = []
    for fi, f_ in enumerate(frames):
        y = next((y for y in found if y["image"] == f_["image"]), None)
        if y is not None:
            own.append((fi, y))
    jobs = [(fi, int(y["ext"]), False) for fi, y in own]
    with Pool(min(workers, len(jobs)), initializer=darkcomet_census_init, initargs=(state,)) as pool:
        res0 = pool.map(darkcomet_census_ccd, jobs)
    obj_mag = {}
    for (fi, y), r in zip(own, res0):
        if not r["ok"]:
            continue
        ra_, de_ = r["B"]["ra"], r["B"]["dec"]
        if not len(ra_):
            continue
        sep = np.hypot((ra_ - y["ra"]) * math.cos(math.radians(y["dec"])), de_ - y["dec"]) * 3600
        j = int(np.argmin(sep))
        if sep[j] < 1.0 and np.isfinite(r["B"]["mag"][j]):
            obj_mag.setdefault(frames[fi]["band"], []).append(dict(image=frames[fi]["image"], mag=float(r["B"]["mag"][j]), snr=float(r["B"]["snr"][j]),
                                                                 sep=float(sep[j]), zp=r["zp"], fwhm=r["fwhm_arcsec"], wcs_rms=r["wcs_rms_arcsec"]))
    mag = {b: float(np.median([v["mag"] for v in vs])) for b, vs in obj_mag.items()}
    # injected tracks: anywhere on the reference frame's CCDs at least 300 arcsec from the object, motion within 10 arcsec/h and 20 deg of its own
    ref = frames[len(frames) // 2]
    rate0, pa0 = math.hypot(ve, vn), math.degrees(math.atan2(ve, vn))
    inj = []
    with fits.open(ref["path"]) as h:
        exts = [k for k in range(1, len(h)) if h[k].header.get("NAXIS1") and (only is None or k in only)]
        from astropy.wcs import WCS
        wcs_ref = {k: WCS(h[k].header) for k in exts}
    while len(inj) < inject:
        k = int(rng.choice(exts))
        ra_, de_ = wcs_ref[k].all_pix2world(rng.uniform(60, 1990), rng.uniform(60, 4550), 0)
        if math.hypot((float(ra_) - ra0) * cos0, float(de_) - de0) * 3600 < 300:
            continue
        rate, pa = rate0 + rng.uniform(-10, 10), pa0 + rng.uniform(-20, 20)
        inj.append((float(ra_), float(de_), rate * math.sin(math.radians(pa)), rate * math.cos(math.radians(pa)), float(rng.uniform(-0.5, 2.0))))
    state.update(inject=inj, mag=mag, points=points)
    jobs = [(fi, k, True) for fi, f_ in enumerate(frames) for k in range(1, f_["n_ext"]) if only is None or k in only]
    print("%d frames, %d CCD images, %d Gaia stars; the object at %s; %d injected tracks; %d workers" % (
        len(frames), len(jobs), len(g_ra), ", ".join("%s %.2f" % (b, m) for b, m in sorted(mag.items())), len(inj), workers), flush=True)
    # the CCD results, kept so that the analysis below can be repeated without measuring the frames again
    import pickle
    cache = os.path.join(full, "census_ccd_%d_%d_%s.pkl" % (inject, seed, "all" if only is None else "-".join(str(k) for k in sorted(only))))
    if reuse and os.path.exists(cache):
        res = pickle.load(open(cache, "rb"))
    else:
        with Pool(workers, initializer=darkcomet_census_init, initargs=(state,)) as pool:
            res = list(pool.imap_unordered(darkcomet_census_ccd, jobs, chunksize=2))
        pickle.dump(res, open(cache, "wb"))
    ok = [r for r in res if r["ok"]]
    print("%d of %d CCD images registered; WCS rms median %.3f arcsec, FWHM %.2f arcsec" % (
        len(ok), len(res), float(np.median([r["wcs_rms_arcsec"] for r in ok])), float(np.median([r["fwhm_arcsec"] for r in ok]))), flush=True)
    summary = dict(desig=desig, night=night, t_ref_mjd=t_ref, frames=[{k: v for k, v in f_.items() if k != "path"} for f_ in frames],
                   n_ccd=len(res), n_registered=len(ok), wcs_rms_median=float(np.median([r["wcs_rms_arcsec"] for r in ok])),
                   fwhm_median=float(np.median([r["fwhm_arcsec"] for r in ok])), object_motion=dict(rate=rate0, pa=pa0 % 360),
                   object_mag=mag, object_frames=obj_mag, threshold=thresh)
    # sources: static when three other frames have anything above the floor within 1.2 arcsec; tracks through the rest, for each filter
    from scipy.spatial import cKDTree
    cat, statics = {}, []
    for key in ("A", "B"):
        T = np.concatenate([np.full(len(r[key]["ra"]), frames[r["fi"]]["t_mid"]) for r in ok])
        X = np.concatenate([(r[key]["ra"] - ra0) * cos0 * 3600 for r in ok])
        Y = np.concatenate([(r[key]["dec"] - de0) * 3600 for r in ok])
        F = np.concatenate([np.full(len(r[key]["ra"]), r["fi"]) for r in ok]).astype(int)
        S = np.concatenate([r[key]["snr"] for r in ok]).astype(float)
        M = np.concatenate([r[key]["mag"] for r in ok]).astype(float)
        cand = np.nonzero(S >= thresh)[0]
        hits = np.zeros(len(cand), int)
        for k in sorted(set(F.tolist())):
            sk = np.nonzero(F == k)[0]
            d_, _ = cKDTree(np.column_stack([X[sk], Y[sk]])).query(np.column_stack([X[cand], Y[cand]]), distance_upper_bound=1.2)
            hits += np.isfinite(d_) & (F[cand] != k)
        static = hits >= 3
        bright = cand[static & (S[cand] > 50)]
        near_star = np.zeros(len(cand), bool)
        if len(bright):
            d_, _ = cKDTree(np.column_stack([X[bright], Y[bright]])).query(np.column_stack([X[cand], Y[cand]]), distance_upper_bound=4.0)
            near_star = np.isfinite(d_) & ~static
        cat[key] = dict(X=X, Y=Y, F=F, S=S, M=M, trans=cand[~static & ~near_star])
        statics.append(np.column_stack([X[cand[static]], Y[cand[static]], S[cand[static]]]))
        summary["filter_" + key] = dict(n_catalogued=int(len(X)), n_above=int(len(cand)), n_static=int(static.sum()), n_transient=int(len(cat[key]["trans"])))
    allX = np.concatenate([cat[k]["X"] for k in cat])
    allY = np.concatenate([cat[k]["Y"] for k in cat])
    allF = np.concatenate([cat[k]["F"] for k in cat])
    ftree = {k: cKDTree(np.column_stack([allX[allF == k], allY[allF == k]])) for k in sorted(set(allF.tolist()))}
    # each frame's CCDs as quadrilaterals in arcsec from the object's mean position, inside the masked border
    quads = {}
    for r in ok:
        cr_, cd_ = np.array(r["corners"][0]), np.array(r["corners"][1])
        quads.setdefault(r["fi"], []).append(np.column_stack([(cr_ - ra0) * cos0 * 3600, (cd_ - de0) * 3600]))

    def covered(fi, px, py):
        for q_ in quads.get(fi, []):
            c_ = [(q_[(i + 1) % 4, 0] - q_[i, 0]) * (py - q_[i, 1]) - (q_[(i + 1) % 4, 1] - q_[i, 1]) * (px - q_[i, 0]) for i in range(4)]
            if all(v >= 0 for v in c_) or all(v <= 0 for v in c_):
                return True
        return False

    def support(x0, y0, ve_, vn_, tmap):
        """Frames whose CCDs contain a straight track and those of them with a catalogued source within 0.6 arcsec of it."""
        cov, sup = 0, 0
        for k, tr_ in ftree.items():
            p_ = [x0 + ve_ * tmap[k], y0 + vn_ * tmap[k]]
            if covered(k, *p_):
                cov += 1
                sup += int(np.isfinite(tr_.query(p_, distance_upper_bound=0.6)[0]))
        return cov, sup

    def tracks_of(key, tmap):
        """Validated tracks of one filter with frame times tmap: rms within 0.3 arcsec and a source in 60 per cent of the frames covering them."""
        c = cat[key]
        tr_idx = c["trans"]
        tt = np.array([tmap[k] for k in c["F"]])
        out_, rejected = [], 0
        for mem in darkcomet_link(tt[tr_idx], c["X"][tr_idx], c["Y"][tr_idx], c["F"][tr_idx]):
            m_ = tr_idx[np.array(mem)]
            px_, py_ = np.polyfit(tt[m_], c["X"][m_], 1), np.polyfit(tt[m_], c["Y"][m_], 1)
            rms_ = float(np.sqrt(np.mean(np.hypot(c["X"][m_] - np.polyval(px_, tt[m_]), c["Y"][m_] - np.polyval(py_, tt[m_])) ** 2)))
            cov, sup = support(px_[1], py_[1], px_[0], py_[0], tmap)
            if rms_ > 0.3 or sup < 0.6 * cov:
                rejected += 1
                continue
            out_.append(dict(x=float(px_[1]), y=float(py_[1]), ve=float(px_[0]), vn=float(py_[0]), n=int(len(m_)), rms=rms_, coverage=cov, support=sup,
                             snr=float(np.median(c["S"][m_])), mag=float(np.nanmedian(c["M"][m_])) if np.any(np.isfinite(c["M"][m_])) else None,
                             frames=sorted({int(v) for v in c["F"][m_]}), filter=key))
        return out_, rejected
    tmap = {fi: frames[fi]["t_mid"] for fi in range(len(frames))}
    # frame times shuffled: no real source moves in a straight line through them, so what links then is chance
    perm = np.random.default_rng(seed + 7).permutation(len(frames))
    tscr = {fi: frames[int(perm[fi])]["t_mid"] for fi in range(len(frames))}
    merged, chance = [], {}
    for key in ("B", "A"):
        trs, rej = tracks_of(key, tmap)
        scr, _ = tracks_of(key, tscr)
        chance[key] = len(scr)
        summary["filter_" + key].update(n_tracks=len(trs), n_rejected=rej, n_shuffled_times=len(scr))
        print("filter %s: %d sources above %.0f sigma, %d above %.0f, %d of them transient; %d tracks kept, %d rejected; %d with shuffled frame times"
              % (key, summary["filter_" + key]["n_catalogued"], floor, summary["filter_" + key]["n_above"], thresh, summary["filter_" + key]["n_transient"],
                 len(trs), rej, len(scr)), flush=True)
        for tr in trs:
            if all(math.hypot(tr["x"] - o["x"], tr["y"] - o["y"]) > 2.0 or math.hypot(tr["ve"] - o["ve"], tr["vn"] - o["vn"]) > 3.0 for o in merged):
                merged.append(tr)
    t_obj = (np.array([x["mjd_mid"] for x in found]) - t_ref) * 24.0
    x_obj, y_obj = (ra_o - ra0) * cos0 * 3600, (de_o - de0) * 3600
    for tr in merged:
        tr["rate"], tr["pa"] = math.hypot(tr["ve"], tr["vn"]), math.degrees(math.atan2(tr["ve"], tr["vn"])) % 360
        tr["is_object"] = bool(np.median(np.hypot(tr["x"] + tr["ve"] * t_obj - x_obj, tr["y"] + tr["vn"] * t_obj - y_obj)) < 1.5)
        tr["injected"] = None
        for k, (ra_, de_, ve_, vn_, dm) in enumerate(inj):
            if math.hypot(tr["x"] - (ra_ - ra0) * cos0 * 3600, tr["y"] - (de_ - de0) * 3600) < 2.0 and math.hypot(tr["ve"] - ve_, tr["vn"] - vn_) < 3.0:
                tr["injected"] = k
                break
    real = [tr for tr in merged if tr["injected"] is None and not tr["is_object"]]
    rec = {tr["injected"] for tr in merged if tr["injected"] is not None}
    # an injected track counts only where the frames covered it: six frames with twenty catalogued sources within 20 arcsec of its path
    placed = [k for k, (ra_, de_, ve_, vn_, dm) in enumerate(inj) if support((ra_ - ra0) * cos0 * 3600, (de_ - de0) * 3600, ve_, vn_, tmap)[0] >= 6]
    dms = np.array([x[4] for x in inj])
    bins = [(-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    summary["injection"] = dict(n=len(inj), n_covered=len(placed), recovered=len(rec & set(placed)),
                                by_dm=[dict(dm=list(b), n=int(sum(1 for k in placed if b[0] <= dms[k] < b[1])),
                                            recovered=int(sum(1 for k in rec & set(placed) if b[0] <= dms[k] < b[1]))) for b in bins])
    summary["object_recovered"] = [dict(n=tr["n"], filter=tr["filter"], rate=tr["rate"], pa=tr["pa"], coverage=tr["coverage"], support=tr["support"])
                                   for tr in merged if tr["is_object"]]
    summary["chance_tracks"] = chance
    # known asteroids in the field from SkyBoT at the reference time
    q = {"-ep": "%.5f" % (t_ref + 2400000.5), "-ra": "%.5f" % ra0, "-dec": "%.5f" % de0, "-rd": "0.85", "-mime": "json", "-output": "all", "-loc": "568",
         "-filter": "0", "-objFilter": "100", "-refsys": "EQJ2000", "-from": "setitools"}
    known = []
    for r in json.loads(core.get(darkcomet_SKYBOT + "?" + urllib.parse.urlencode(q), timeout=300)):
        h_ = [float(v) for v in r["RA (hms)"].split()]
        g_ = r["DEC (dms)"].split()
        sg = -1.0 if g_[0].startswith("-") else 1.0
        known.append(dict(name=r["Name"], V=float(r["VMag (mag)"]), x=(15.0 * (h_[0] + h_[1] / 60 + h_[2] / 3600) - ra0) * cos0 * 3600,
                          y=(sg * (abs(float(g_[0])) + float(g_[1]) / 60 + float(g_[2]) / 3600) - de0) * 3600,
                          ve=float(r["dRA (arcsec/h)"]), vn=float(r["dDEC (arcsec/h)"]), err=float(r.get("Err (arcsec)") or 0)))
    for tr in real:
        best = min(known, key=lambda o: math.hypot(tr["x"] - o["x"], tr["y"] - o["y"])) if known else None
        if best and math.hypot(tr["x"] - best["x"], tr["y"] - best["y"]) < max(15.0, 3 * best["err"]) and math.hypot(tr["ve"] - best["ve"], tr["vn"] - best["vn"]) < 5.0:
            tr["known"] = dict(name=best["name"], V=best["V"])
    # known asteroids the frames covered, and how many of them the census found
    known_cov = [o for o in known if o["err"] < 10 and support(o["x"], o["y"], o["ve"], o["vn"], tmap)[0] >= 6]
    # the area of the reference frame's registered CCDs, in square degrees
    fref = frames.index(ref)
    reg_ref = {r["ext"] for r in ok if r["fi"] == fref}
    summary["registered"] = sorted([int(r["fi"]), int(r["ext"])] for r in ok)
    area = 0.0
    with fits.open(ref["path"]) as h:
        for k in exts:
            if k in reg_ref:
                area += h[k].header["NAXIS1"] * h[k].header["NAXIS2"] * abs(np.linalg.det(wcs_ref[k].pixel_scale_matrix[:2, :2]))
    rates, pas = np.array([tr["rate"] for tr in real]), (np.array([tr["pa"] for tr in real]) - pa0 + 180) % 360 - 180
    frac = None
    if len(real) >= 5:
        frac = (math.exp(-0.5 * ((rate0 - rates.mean()) / rates.std(ddof=1)) ** 2) / (rates.std(ddof=1) * math.sqrt(2 * math.pi)) * 1.0
                * math.exp(-0.5 * (pas.mean() / pas.std(ddof=1)) ** 2) / (pas.std(ddof=1) * math.sqrt(2 * math.pi)) * 2.0)
    dens = len(real) / area
    summary["movers"] = dict(area_deg2=area, n=len(real), n_known=sum(1 for tr in real if tr.get("known")), n_known_skybot=len(known),
                             n_known_covered=len(known_cov), known_covered=[dict(name=o["name"], V=o["V"]) for o in known_cov],
                             density_per_deg2=dens, rate_mean=float(rates.mean()) if len(real) else None, rate_sd=float(rates.std(ddof=1)) if len(real) > 1 else None,
                             pa_offset_mean=float(pas.mean()) if len(real) else None, pa_offset_sd=float(pas.std(ddof=1)) if len(real) > 1 else None,
                             within_5_10=int(np.sum((np.abs(rates - rate0) < 5) & (np.abs(pas) < 10))), within_10_20=int(np.sum((np.abs(rates - rate0) < 10) & (np.abs(pas) < 20))),
                             motion_fraction=frac, expected_30arcsec=(dens * math.pi * (30 / 3600) ** 2 * frac) if frac is not None else None,
                             control_discs=int(area / (math.pi * (30 / 3600) ** 2)), tracks=real)
    print("%d real movers over %.2f deg^2 (%.0f per deg^2), %d of them known to SkyBoT, which lists %d in the field, %d on the frames; %d within 5 arcsec/h and 10 deg "
          "of the object's motion, %d within 10 and 20" % (len(real), area, dens, summary["movers"]["n_known"], len(known), len(known_cov),
                                                            summary["movers"]["within_5_10"], summary["movers"]["within_10_20"]))
    if frac is not None:
        print("their motions %.1f +- %.1f arcsec/h, %+.0f +- %.0f deg from the object's; expected unrelated movers within 30 arcsec moving within 0.5 arcsec/h and 1 deg: %.1e"
              % (rates.mean(), rates.std(ddof=1), pas.mean(), pas.std(ddof=1), summary["movers"]["expected_30arcsec"]))
    print("the object recovered: %s; injected copies on the frames recovered: %s" % (summary["object_recovered"], ", ".join(
        "dm %+.1f..%+.1f %d/%d" % (b["dm"][0], b["dm"][1], b["recovered"], b["n"]) for b in summary["injection"]["by_dm"])))
    json.dump(summary, open(os.path.join(d, "census_%s.json" % tag), "w"), indent=1)
    # the trailed filter along the locus, at the orbits, over a map around the gravity-only orbit and along control loci: in each frame the S/N
    # over the median of the detection's in that band, a source's flux in units of the detection's, and its median over the frames, which a
    # static source crossing one or two frames does not move
    if layout:
        # samples within 10 arcsec of a CCD's masked border left out, where the background model and the trailed filter are least reliable
        Z = {}
        for r in ok:
            s_ = r.get("samples")
            if s_:
                z = Z.setdefault(r["fi"], np.full(len(points[r["fi"]]), np.nan))
                idx_ = np.array(s_["idx"], int)
                P_ = np.column_stack([(points[r["fi"]][idx_, 0] - ra0) * cos0 * 3600, (points[r["fi"]][idx_, 1] - de0) * 3600])
                q_ = np.column_stack([(np.array(r["corners"][0]) - ra0) * cos0 * 3600, (np.array(r["corners"][1]) - de0) * 3600])
                dmin = np.full(len(P_), np.inf)
                for i in range(4):
                    a_, b_ = q_[i], q_[(i + 1) % 4]
                    ab = b_ - a_
                    tt = np.clip(((P_ - a_) @ ab) / (ab @ ab), 0, 1)
                    dmin = np.minimum(dmin, np.hypot(*(P_ - (a_ + tt[:, None] * ab)).T))
                keep_ = dmin > 10.0
                z[idx_[keep_]] = np.array(s_["z"])[keep_]
        # positions within 4 + 3 (18 - G) arcsec of a Gaia star brighter than G = 18, or within 2.5 arcsec of a static source above S/N 20
        # (which covers bleed trails and bright galaxies), left out on the locus and the controls alike
        gb = (g_mag < 18) & np.isfinite(g_ra) & np.isfinite(g_dec)
        gtree = cKDTree(np.column_stack([(g_ra[gb] - ra0) * cos0 * 3600, (g_dec[gb] - de0) * 3600]))
        st = np.vstack(statics)
        btree = cKDTree(st[st[:, 2] > 20, :2])
        for fi, z in Z.items():
            P_ = np.column_stack([(points[fi][:, 0] - ra0) * cos0 * 3600, (points[fi][:, 1] - de0) * 3600])
            d_, j_ = gtree.query(P_, distance_upper_bound=40.0)
            hit = np.isfinite(d_)
            z[np.nonzero(hit)[0][d_[hit] < 4.0 + 3.0 * (18.0 - g_mag[gb][j_[hit]])]] = np.nan
            d_, _ = btree.query(P_, distance_upper_bound=2.5)
            z[np.isfinite(d_)] = np.nan
        nl = layout["locus"]
        fis = sorted(Z)
        band_w = {}
        for fi in fis:
            band_w.setdefault(frames[fi]["band"], []).append(Z[fi][nl + 2])
        band_w = {b: float(np.nanmedian(v)) for b, v in band_w.items()}
        R = np.array([Z[fi] / band_w[frames[fi]["band"]] for fi in fis])

        def med(rows_):
            """Median flux ratio at each position over the frames that measured it, and their number."""
            rr = R[:, rows_]
            n_ = np.sum(np.isfinite(rr), axis=0)
            return np.where(n_ >= 8, np.nanmedian(np.where(n_ >= 8, rr, 0.0), axis=0), np.nan), n_
        fl, nl_f = med(np.arange(nl))
        fg, ng = med(np.array([nl]))
        fa, na = med(np.array([nl + 1]))
        fd, nd = med(np.array([nl + 2]))
        fm, nm = med(np.arange(nl + 3, nl + 3 + layout["map"]))
        oe, on = off_e.ravel(), off_n.ravel()
        de_off = float(np.mean([((y["ra"] - lf[frames[fi]["image"]]["grav"][0] + 180) % 360 - 180) * math.cos(math.radians(y["dec"])) * 3600
                                for fi in fis for y in found if y["image"] == frames[fi]["image"]]))
        dn_off = float(np.mean([(y["dec"] - lf[frames[fi]["image"]]["grav"][1]) * 3600 for fi in fis for y in found if y["image"] == frames[fi]["image"]]))
        away = (np.hypot(oe - de_off, on - dn_off) > 3.0) & np.isfinite(fm)
        j_away = int(np.nanargmax(np.where(away, fm, np.nan)))
        i_pk = int(np.nanargmax(fl))
        off_pk = np.abs(grid - grid[i_pk]) > 1.5e-12
        j_lo = int(np.nanargmax(np.where(off_pk, fl, np.nan)))
        best_null, at_zero, high = [], [], []
        i0 = list(grid).index(0.0)
        f0 = fis[len(fis) // 2]
        for s_i in range(layout["shifts"]):
            base = nl + 3 + layout["map"] + s_i * nl
            fc, nc = med(np.arange(base, base + nl))
            if np.sum(np.isfinite(fc)) > 0.8 * nl:
                best_null.append(float(np.nanmax(fc)))
                if np.isfinite(fc[i0]):
                    at_zero.append(float(fc[i0]))
                if best_null[-1] >= 0.5:
                    # where the high value lies in the middle frame, and the brightest Gaia star within 120 arcsec of it
                    j = base + int(np.nanargmax(fc))
                    px_, py_ = (points[f0][j, 0] - ra0) * cos0 * 3600, (points[f0][j, 1] - de0) * 3600
                    near_g = np.hypot((g_ra - ra0) * cos0 * 3600 - px_, (g_dec - de0) * 3600 - py_) < 120
                    high.append(dict(flux=best_null[-1], x=float(px_), y=float(py_),
                                     brightest_g_within_120=float(np.nanmin(g_mag[near_g])) if near_g.any() else None))
        best_null, at_zero = np.array(best_null), np.array(at_zero)
        sd0 = float(1.4826 * np.median(np.abs(at_zero - np.median(at_zero)))) if len(at_zero) else None
        summary["locus"] = dict(frames=[frames[fi]["image"] for fi in fis], band_snr=band_w, grid=[float(v) for v in grid],
                                flux=[None if not np.isfinite(v) else float(v) for v in fl], n_frames=[int(v) for v in nl_f],
                                peak_A2=float(grid[i_pk]), peak_flux=float(fl[i_pk]), peak_frames=int(nl_f[i_pk]),
                                elsewhere_max=dict(A2=float(grid[j_lo]), flux=float(fl[j_lo]), frames=int(nl_f[j_lo])),
                                flux_grav=float(fg[0]), n_grav=int(ng[0]), flux_accel=float(fa[0]), flux_detection=float(fd[0]), n_detection=int(nd[0]),
                                null_sd=sd0, dm_limit_3sigma=float(-2.5 * math.log10(max(float(fg[0]) + 3 * sd0, 1e-3))) if sd0 else None,
                                map_max_away=dict(flux=float(fm[j_away]), offset=[float(oe[j_away]), float(on[j_away])], frames=int(nm[j_away])),
                                detection_offset_from_grav=[de_off, dn_off],
                                null_max=dict(n=len(best_null), percentiles=[float(v) for v in np.percentile(best_null, [50, 90, 99, 100])] if len(best_null) else None,
                                              beyond_detection=int(np.sum(best_null >= 0.5 * float(fl[i_pk])))),
                                null_at_zero=dict(n=len(at_zero), median=float(np.median(at_zero)) if len(at_zero) else None, robust_sd=sd0),
                                null_high=sorted(high, key=lambda h: -h["flux"]),
                                map=dict(step=0.5, half=24.0, flux=[None if not np.isfinite(v) else round(float(v), 3) for v in fm], n_frames=[int(v) for v in nm]))
        x_ = summary["locus"]
        print("along the locus (%d frames; the detection's median S/N %s): flux %.2f of the detection's at A2 %+.2e (%d frames); elsewhere on the locus at most %.3f "
              "(A2 %+.2e); at the gravity-only orbit %.3f (%d frames); the map's highest away from the detection %.3f at E %+.1f N %+.1f arcsec"
              % (len(fis), ", ".join("%s %.1f" % kv for kv in sorted(band_w.items())), x_["peak_flux"], x_["peak_A2"], x_["peak_frames"],
                 x_["elsewhere_max"]["flux"], x_["elsewhere_max"]["A2"], x_["flux_grav"], x_["n_grav"], x_["map_max_away"]["flux"], *x_["map_max_away"]["offset"]))
        if len(best_null):
            print("control loci: %d; median flux at A2 = 0 %.3f, robust sd %.3f, so the gravity-only position is at least %.1f mag fainter than the detection (3 sigma); "
                  "highest flux along each: median %.3f, 99th percentile %.3f, maximum %.3f; %d reach half the detection's"
                  % (len(best_null), x_["null_at_zero"]["median"], sd0, x_["dm_limit_3sigma"], x_["null_max"]["percentiles"][0], x_["null_max"]["percentiles"][2],
                     x_["null_max"]["percentiles"][3], x_["null_max"]["beyond_detection"]))
    json.dump(summary, open(os.path.join(d, "census_%s.json" % tag), "w"), indent=1)
    return summary


def darkcomet_census_init(state):
    darkcomet_CENSUS.update(state)


def darkcomet_ym(jd):
    from astropy.time import Time
    s = Time(jd, format="jd").datetime
    return s.year, s.month


def darkcomet_record(desig):
    """Record the thermal and orbit fits, A2 set against all absorbed sunlight at the fitted diameter."""
    d = darkcomet_dir(desig)
    rf = json.load(open(os.path.join(d, "refit.json")))
    f = rf["fits"]
    wa, g = f["with A2"], f["gravity only"]
    loo = {k[8:]: v for k, v in f.items() if k.startswith("without")}
    if os.path.exists(os.path.join(d, "thermal.json")):
        th = json.load(open(os.path.join(d, "thermal.json")))
        size = "NEATM D %.2f +- %.2f km, eta %.2f, pV %.3f" % (th["D_km"], th["D_err"], th["eta"], th["pv"])
    else:
        # no NEOWISE detection: diameter from H at S-type and dark albedos
        sb = json.load(open(os.path.join(d, "sbdb.json")))
        H = float(next(p["value"] for p in sb["phys_par"] if p["name"] == "H"))
        th = dict(D_km=1329.0 / math.sqrt(0.14) * 10 ** (-0.2 * H), pv=0.14)
        size = "no NEOWISE detection; D %.2f km from H %.2f at pV 0.14 (%.2f km at pV 0.03)" % (th["D_km"], H, 1329.0 / math.sqrt(0.03) * 10 ** (-0.2 * H))
    D = th["D_km"] * 1000.0
    xi = {rho: abs(wa["A2"]) * nga_AU_D2 / (nga_S_OVER_C * nga_CR_MAX / (D * rho)) for rho in (500.0, 1000.0, 2000.0)}
    # lead with the weighting that gives the drift least significance
    weights = {k: v for k, v in f.items() if k in ("with A2", "with A2, Veres nights", "with A2, one position a night")}
    lead_k = min(weights, key=lambda k: abs(weights[k]["A2"] / weights[k]["A2_sigma"]))
    lead = weights[lead_k]
    run = "darkcomet_" + desig.replace(" ", "_").lower()
    core.record_search(
        run, "Dark comet candidate", archive="MPC observations; JPL SBDB and Horizons; NEOWISE single-exposure sources (IRSA)",
        release="JPL solution with A2 %.3e; %d observations" % (rf["a2_jpl"], wa["n_obs"]), selection=desig,
        code=os.path.abspath(__file__), entry="darkcomet", n_targets=1,
        params=dict(sigma_default=0.5, sigma_site=darkcomet_SIGMA, chi2_out=darkcomet_CHI2_OUT, g_r="r^-2",
                    neatm=dict(eps=darkcomet_EPS, G=darkcomet_G, p_ir_over_pv=darkcomet_PIR, floor=darkcomet_FLOOR)),
        null_summary=dict(chi2_gravity=g["chi2"], dof_gravity=g["dof"], chi2_a2=wa["chi2"], dof_a2=wa["dof"]),
        sensitivity="A2 from the refit and from each refit with one apparition left out; NEATM diameter from W1 and W2",
        ledger="Yarkovsky sign test",
        result="A2 %+.2e +- %.1e au/d^2 (%.1f sigma) %s, the figure to quote (JPL %+.2e)%s, chi2 %.0f with A2 against "
               "%.0f without; left out in turn: %s; %s; the drift is %.1f, %.1f and %.1f times all absorbed sunlight at "
               "500, 1000 and 2000 kg/m^3"
               % (lead["A2"], lead["A2_sigma"], lead["A2"] / lead["A2_sigma"],
                  "with site weights" if lead_k == "with A2" else lead_k.replace("with A2, ", "with "), rf["a2_jpl"],
                  "".join("; %s: A2 %+.2e +- %.1e (%.1f sigma)%s" % (
                      "site weights" if k == "with A2" else k.replace("with A2, ", ""), v["A2"], v["A2_sigma"],
                      v["A2"] / v["A2_sigma"],
                      ", A1 %+.2e +- %.1e" % (v["A1"], v["A1_sigma"]) if v.get("A1") is not None else "")
                      for k, v in f.items() if k.startswith("with A") and k != lead_k),
                  wa["chi2"], g["chi2"],
                  "; ".join("%s %+.2e (%.1f sigma)" % (k, v["A2"], v["A2"] / v["A2_sigma"]) for k, v in loo.items()),
                  size, xi[500.0], xi[1000.0], xi[2000.0]))
    # fits keep different observations; compared per degree of freedom
    red_a2, red_g = wa["chi2"] / wa["dof"], g["chi2"] / g["dof"]
    known = yarkovsky_KNOWN.get(desig)
    if known:
        vet = "%s, its transverse drift beyond sunlight" % known if xi[1000.0] > 1.0 else known
    elif red_a2 >= red_g - 0.02:
        vet = "gravity alone fits as well (chi2/dof %.2f against %.2f with A2): no drift needed" % (red_g, red_a2)
    elif xi[1000.0] > 1.0:
        vet = "transverse drift beyond sunlight: dark-comet candidate"
    else:
        vet = "transverse drift within sunlight: Yarkovsky"
    core.clear_targets(run)
    core.record_targets(run, [dict(
        target="mpc:" + desig, stat=lead["A2"] / lead["A2_sigma"], unit="A2 / sigma in the refit, the figure to quote", vet=vet,
        note="D %.2f km, pV %.3f; xi %.1f at 1000 kg/m^3; chi2/dof %.2f with A2, %.2f gravity only" % (th["D_km"], th["pv"], xi[1000.0], red_a2, red_g))])
    print("recorded %s" % run)


darkcomet_SSOIS = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/ssos/ssosclf.pl"
darkcomet_CADC_DATA = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/"
darkcomet_IBE = "https://irsa.ipac.caltech.edu/ibe/data/wise/neowiser/p1bm_frm/"


def darkcomet_images(desig, epoch1="2000 01 01", epoch2=None):
    """Archived images covering the ephemeris (CADC SSOIS) with Horizons V per epoch into images.json; lists archives and months brighter than V 24.6."""
    from datetime import date
    d = darkcomet_dir(desig)
    epoch2 = epoch2 or date.today().strftime("%Y %m %d")
    u = darkcomet_SSOIS + "?" + urllib.parse.urlencode({"lang": "en", "object": desig, "search": "bynameCADC", "epoch1": epoch1, "epoch2": epoch2,
                                                        "eellipse": "", "eunits": "arcseconds", "extres": "yes", "xyres": "yes", "format": "tsv"})
    import csv
    import io
    t = core.get(u, timeout=600).decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(t), delimiter="\t"))
    mjds = sorted({round(float(r["MJD"]), 5) for r in rows})
    eph = {}
    for i in range(0, len(mjds), 25):
        chunk = mjds[i:i + 25]
        q = {"format": "json", "COMMAND": "'DES=%s;'" % desig, "CENTER": "'500@399'", "EPHEM_TYPE": "'OBSERVER'", "QUANTITIES": "'1,9,19,20'",
             "TLIST": "'%s'" % " ".join("%.5f" % (m + 2400000.5) for m in chunk), "TLIST_TYPE": "'JD'", "TIME_TYPE": "'UT'",
             "CSV_FORMAT": "'YES'", "OBJ_DATA": "'NO'", "ANG_FORMAT": "'DEG'"}
        txt = json.loads(core.get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), timeout=180))["result"]
        if "$$SOE" not in txt:
            continue
        lines = txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
        for m, line in zip(chunk, lines):  # Horizons keeps the ascending input order
            c = [x.strip() for x in line.split(",")]
            eph[m] = dict(ra=num(c[3]), dec=num(c[4]), V=num(c[5]), r=num(c[7]), delta=num(c[9]))
    for r in rows:
        r.update(eph.get(round(float(r["MJD"]), 5), {}))
    json.dump(rows, open(os.path.join(d, "images.json"), "w"), indent=0)
    from astropy.time import Time
    by = {}
    for r in rows:
        by.setdefault((r["Telescope/Instrument"], Time(float(r["MJD"]), format="mjd").iso[:7]), []).append(r)
    print("%d images of %s in %d archives" % (len(rows), desig, len({r["Telescope/Instrument"] for r in rows})))
    for k, v in sorted(by.items(), key=lambda kv: kv[0][1]):
        vs = [r["V"] for r in v if r.get("V") is not None]
        if vs and min(vs) < 24.6:
            print("  %-24s %s: %3d images, V %.1f-%.1f, exposures %s s, %s" % (k[0], k[1], len(v), min(vs), max(vs),
                                                                            ",".join(sorted({r["Exptime"] for r in v})[:4]), ",".join(sorted({r["Filter"][:6] for r in v})[:4])))
    return rows


def darkcomet_precovery(desig, night, site="568", half_y=700):
    """Re-measure the object in one night's CFHT MegaCam images: Gaia-registered CCD cutouts, predicted trail matched within 2.5 arcsec, offsets from prediction and MPC reports, into precovery_<night>.json."""
    quiet_warnings()
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS
    from photutils.detection import DAOStarFinder
    from scipy.ndimage import gaussian_filter
    from scipy.signal import fftconvolve
    d = darkcomet_dir(desig)
    rows = [r for r in json.load(open(os.path.join(d, "images.json"))) if r["Telescope/Instrument"] == "CFHT/MegaCam"
            and Time(float(r["MJD"]), format="mjd").iso[:10] == night]
    if not rows:
        print("no CFHT images of %s on %s in images.json" % (desig, night))
        return []
    cdir = os.path.join(d, "cfht_" + night.replace("-", ""))
    os.makedirs(cdir, exist_ok=True)
    ra0, dec0 = float(rows[0]["Object_RA"]), float(rows[0]["Object_Dec"])
    gaia = core.datalab("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag FROM gaia_dr3.gaia_source WHERE "
                            "q3c_radial_query(ra, dec, %.5f, %.5f, 0.15) = 't' AND phot_g_mean_mag < 21.5" % (ra0, dec0))
    mpc = [r for r in darkcomet_parse(open(os.path.join(d, "obs80.txt")).read())
           if Time(r["jd_utc"], format="jd").iso[:10] == night]

    def horizons(jd):
        q = {"format": "json", "COMMAND": "'DES=%s;'" % desig, "CENTER": "'%s@399'" % site, "EPHEM_TYPE": "'OBSERVER'", "QUANTITIES": "'1,9'",
             "TLIST": "'%.6f'" % jd, "TLIST_TYPE": "'JD'", "TIME_TYPE": "'UT'", "CSV_FORMAT": "'YES'", "OBJ_DATA": "'NO'", "ANG_FORMAT": "'DEG'"}
        txt = json.loads(core.get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), timeout=180))["result"]
        c = [x.strip() for x in txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()[0].split(",")]
        return float(c[3]), float(c[4]), c[5]

    def sep(ra1, dec1, ra2, dec2):
        return ((ra1 - ra2 + 180) % 360 - 180) * math.cos(math.radians(dec2)) * 3600, (dec1 - dec2) * 3600
    out = []
    for r in sorted(rows, key=lambda r: float(r["MJD"])):
        y, ext = int(float(r["Y"])), int(r["Ext"])
        path = os.path.join(cdir, "%s_%d.fits" % (r["Image"], ext))
        if not os.path.exists(path):
            cut = "[%d][1:2048,%d:%d]" % (ext, max(1, y - half_y), min(4612, y + half_y))
            b = core.get(darkcomet_CADC_DATA + "%s.fits.fz?cutout=%s" % (r["Image"], urllib.parse.quote(cut)), timeout=900, retries=3, soft=True)
            if not b:
                print("%s: no cutout" % r["Image"])
                continue
            open(path, "wb").write(b)
        h = fits.open(path)
        hd, img = h[-1].header, h[-1].data.astype(float)
        t_mid = Time("%sT%s" % (hd["DATE-OBS"], hd["UTC-OBS"]), scale="utc") + (hd["EXPTIME"] / 2.0 / 86400.0)
        w = WCS(hd)
        bkg = np.median(img)
        sig = 1.4826 * np.median(np.abs(img - bkg))
        stars = DAOStarFinder(fwhm=4.0, threshold=6 * sig, exclude_border=True)(img - bkg)
        if stars is None or len(stars) < 6:
            continue
        sx, sy = np.array(stars["xcentroid"]), np.array(stars["ycentroid"])
        dt = t_mid.jyear - 2016.0
        g_ra = np.array([float(g["ra"]) + float(g["pmra"] or 0) / 3.6e6 * dt / math.cos(math.radians(float(g["dec"]))) for g in gaia])
        g_dec = np.array([float(g["dec"]) + float(g["pmdec"] or 0) / 3.6e6 * dt for g in gaia])
        gx, gy = w.all_world2pix(g_ra, g_dec, 0)
        gi = np.nonzero((gx > 5) & (gx < img.shape[1] - 5) & (gy > 5) & (gy < img.shape[0] - 5))[0]
        dxs, dys = [], []
        for i in gi:
            dd = np.hypot(sx - gx[i], sy - gy[i])
            for j in np.nonzero(dd < 40)[0]:
                dxs.append(sx[j] - gx[i])
                dys.append(sy[j] - gy[i])
        if len(dxs) < 6:
            print("%s: %d Gaia stars in the cutout, too few" % (r["Image"], len(gi)))
            continue
        H2, xe, ye = np.histogram2d(dxs, dys, bins=80, range=[[-40, 40], [-40, 40]])
        k = np.unravel_index(np.argmax(H2), H2.shape)
        sh = (0.5 * (xe[k[0]] + xe[k[0] + 1]), 0.5 * (ye[k[1]] + ye[k[1] + 1]))
        pairs = []
        for i in gi:
            dd = np.hypot(sx - gx[i] - sh[0], sy - gy[i] - sh[1])
            j = int(np.argmin(dd))
            if dd[j] < 3.0:
                pairs.append((sx[j], sy[j], gx[i], gy[i]))
        pairs = np.array(pairs)
        if len(pairs) < 6:
            print("%s: only %d Gaia matches" % (r["Image"], len(pairs)))
            continue
        A = np.column_stack([pairs[:, 0], pairs[:, 1], np.ones(len(pairs))])
        for _ in range(3):
            cx = np.linalg.lstsq(A, pairs[:, 2], rcond=None)[0]
            cy = np.linalg.lstsq(A, pairs[:, 3], rcond=None)[0]
            res = np.hypot(A @ cx - pairs[:, 2], A @ cy - pairs[:, 3])
            keep = res < max(3 * np.median(res), 0.3)
            A, pairs = A[keep], pairs[keep]
        res = np.hypot(A @ cx - pairs[:, 2], A @ cy - pairs[:, 3])
        scale = abs(hd["CD2_2"]) * 3600.0
        M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
        pra, pdec, pV = horizons(t_mid.jd)
        px, py = w.all_world2pix(pra, pdec, 0)
        hx, hy = np.linalg.solve(M, np.array([px - cx[2], py - cy[2]]))
        ra1, dec1, _ = horizons(t_mid.jd + 1.0 / 1440)
        px1, py1 = w.all_world2pix(ra1, dec1, 0)
        hx1, hy1 = np.linalg.solve(M, np.array([px1 - cx[2], py1 - cy[2]]))
        vx, vy = (hx1 - hx) * hd["EXPTIME"] / 60.0, (hy1 - hy) * hd["EXPTIME"] / 60.0
        L = math.hypot(vx, vy)
        K = np.zeros((41, 41))
        for s_ in np.linspace(-0.5, 0.5, max(int(L * 2), 2)):
            K[int(round(20 + s_ * vy)), int(round(20 + s_ * vx))] += 1.0
        K = gaussian_filter(K, 1.7)
        K /= np.sqrt(np.sum(K ** 2))
        mf = fftconvolve(img - bkg, K[::-1, ::-1], mode="same")
        msig = 1.4826 * np.median(np.abs(mf - np.median(mf)))
        R = int(2.5 / scale)
        x0, y0 = int(round(hx)), int(round(hy))
        rec = dict(image=r["Image"], ext=ext, mjd_mid=float(t_mid.mjd), exptime=float(hd["EXPTIME"]), filter=hd["FILTER"], n_gaia=int(len(pairs)),
                   shift_px=[float(sh[0]), float(sh[1])], wcs_rms_arcsec=float(np.sqrt(np.mean(res ** 2))) * scale, pred_ra=pra, pred_dec=pdec,
                   pred_V=pV, trail_px=L, scale=scale, found=False)
        if R < x0 < img.shape[1] - R and R < y0 < img.shape[0] - R:
            box = mf[y0 - R:y0 + R + 1, x0 - R:x0 + R + 1]
            iy, ix = np.unravel_index(int(np.nanargmax(box)), box.shape)
            rec["matched_peak_sigma"] = float(box[iy, ix] / msig)
            if box[iy, ix] > 3.0 * msig:
                yy0, xx0 = y0 - R + iy, x0 - R + ix

                def refine(a, b, c):
                    dd = a - 2 * b + c
                    return 0.5 * (a - c) / dd if dd != 0 else 0.0
                ox = xx0 + refine(mf[yy0, xx0 - 1], mf[yy0, xx0], mf[yy0, xx0 + 1])
                oy = yy0 + refine(mf[yy0 - 1, xx0], mf[yy0, xx0], mf[yy0 + 1, xx0])
                gxo, gyo = cx[0] * ox + cx[1] * oy + cx[2], cy[0] * ox + cy[1] * oy + cy[2]
                ora, odec = (float(v) for v in w.all_pix2world(gxo, gyo, 0))
                rec.update(found=True, ra=ora, dec=odec, pix=[float(ox), float(oy)], obs_minus_pred=list(sep(ora, odec, pra, pdec)))
                near = min(mpc, key=lambda m: abs(m["jd_utc"] - t_mid.jd)) if mpc else None
                if near and abs(near["jd_utc"] - t_mid.jd) * 86400 < 120:
                    rec["obs_minus_mpc"] = dict(site=near["site"], dt_s=(t_mid.jd - near["jd_utc"]) * 86400, offset=list(sep(ora, odec, near["ra"], near["dec"])))
        out.append(rec)
        print("%s %s %4.0fs %s: %2d Gaia, WCS rms %.3f\"; %s" % (
            r["Image"], t_mid.iso[11:19], hd["EXPTIME"], hd["FILTER"][:1], rec["n_gaia"], rec["wcs_rms_arcsec"],
            ("trail found at %.1f sigma, obs-pred E %+.2f N %+.2f\"%s" % (rec["matched_peak_sigma"], *rec["obs_minus_pred"],
             (", obs-MPC(%s, %+.0f s) E %+.2f N %+.2f\"" % (rec["obs_minus_mpc"]["site"], rec["obs_minus_mpc"]["dt_s"], *rec["obs_minus_mpc"]["offset"])) if rec.get("obs_minus_mpc") else ""))
             if rec["found"] else "not found (matched peak %.1f sigma)" % rec.get("matched_peak_sigma", 0)), flush=True)
    good = [x for x in out if x["found"] and math.hypot(*x["obs_minus_pred"]) < 1.0]
    if len(good) >= 3:
        e = np.array([x["obs_minus_pred"][0] for x in good])
        n = np.array([x["obs_minus_pred"][1] for x in good])
        t = np.array([x["mjd_mid"] for x in good])
        ra = np.array([x["ra"] for x in good])
        dec = np.array([x["dec"] for x in good])
        cosd = math.cos(math.radians(dec.mean()))
        tm = (t - t.min()) * 24.0
        px_ = np.polyfit(tm, (ra - ra[0]) * cosd * 3600, 1)
        py_ = np.polyfit(tm, (dec - dec[0]) * 3600, 1)
        print("%d positions within 1 arcsec of the prediction: median offset E %+.2f N %+.2f\" (scatter %.2f, %.2f); motion %.1f arcsec/h toward PA %.0f deg"
              % (len(good), np.median(e), np.median(n), 1.4826 * np.median(np.abs(e - np.median(e))), 1.4826 * np.median(np.abs(n - np.median(n))),
                 math.hypot(px_[0], py_[0]), math.degrees(math.atan2(px_[0], py_[0])) % 360))
    # MPC report times against exposure start and end
    for m in mpc:
        best = min(out, key=lambda x: min(abs((m["jd_utc"] - x["mjd_mid"] - 2400000.5) * 86400 + x["exptime"] / 2), abs((m["jd_utc"] - x["mjd_mid"] - 2400000.5) * 86400 - x["exptime"] / 2)))
        s0 = (m["jd_utc"] - best["mjd_mid"] - 2400000.5) * 86400 + best["exptime"] / 2
        print("  %s report at %s: exposure %s start %+.0f s, mid %+.0f s, end %+.0f s" % (m["site"], Time(m["jd_utc"], format="jd").iso[11:19], best["image"], s0, s0 - best["exptime"] / 2, s0 - best["exptime"]))
    json.dump(out, open(os.path.join(d, "precovery_%s.json" % night.replace("-", "")), "w"), indent=1)
    return out


def darkcomet_stack(desig, vmax=21.6, half=20):
    """Shift-and-stack NEOWISE single exposures on the predicted position (V below vmax, clean frames) with a field-star PSF, radial profiles and aperture-corrected 8.25 arcsec magnitudes, into stack.json."""
    quiet_warnings()
    import csv
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import maximum_filter, shift as ndshift
    d = darkcomet_dir(desig)
    fdir = os.path.join(d, "neowise_frames")
    os.makedirs(fdir, exist_ok=True)
    frames_csv = os.path.join(d, "most_frames.csv")
    if os.path.exists(frames_csv):
        for r in csv.DictReader(open(frames_csv)):
            if num(r["vmag"]) is None or num(r["vmag"]) >= vmax:
                continue
            scan, frame, band = r["scan_id"], int(r["frame_num"]), r["band"]
            out = os.path.join(fdir, "%s%03d-w%s.fits" % (scan, frame, band))
            if not os.path.exists(out):
                u = darkcomet_IBE + "%s/%s/%03d/%s%03d-w%s-int-1b.fits?center=%s,%s&size=150arcsec" % (scan[-2:], scan, frame, scan, frame, band, r["ra_obj"], r["dec_obj"])
                b = core.get(u, timeout=300, retries=3, soft=True, pause=10)
                if b:
                    open(out, "wb").write(b)

    def horizons(jds):
        q = {"format": "json", "COMMAND": "'DES=%s;'" % desig, "CENTER": "'500@-163'", "EPHEM_TYPE": "'OBSERVER'", "QUANTITIES": "'1,9'",
             "TLIST": "'%s'" % " ".join("%.6f" % j for j in jds), "TLIST_TYPE": "'JD'", "TIME_TYPE": "'UT'", "CSV_FORMAT": "'YES'",
             "OBJ_DATA": "'NO'", "ANG_FORMAT": "'DEG'"}
        txt = json.loads(core.get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), timeout=180))["result"]
        return [(float(c[3]), float(c[4]), c[5]) for c in ([x.strip() for x in line.split(",")]
                                                           for line in txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines())]

    def window(img, x, y):
        big = np.zeros((img.shape[0] + 2 * half, img.shape[1] + 2 * half))
        big[half:-half, half:-half] = np.where(np.isfinite(img), img, 0.0)
        xi, yi = int(math.floor(x)), int(math.floor(y))
        sub = big[yi:yi + 2 * half + 1, xi:xi + 2 * half + 1]
        return ndshift(sub, (yi - y, xi - x), order=1, mode="nearest") if sub.shape == (2 * half + 1, 2 * half + 1) else None
    yy, xx = np.indices((2 * half + 1, 2 * half + 1))
    rr = np.hypot(xx - half, yy - half)

    def profile(S):
        return [(a, b, float(np.mean(S[(rr >= a) & (rr < b)]))) for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10))]
    res = {}
    for band in ("w1", "w2"):
        files = sorted(glob.glob(os.path.join(fdir, "*-%s.fits" % band)))
        frames = sorted(((f, fits.open(f)[0].header, fits.open(f)[0].data.astype(float)) for f in files), key=lambda t: t[1]["MJD_OBS"])
        if not frames:
            continue
        eph = horizons([hd["MJD_OBS"] + 2400000.5 for _, hd, _ in frames])
        obj, dn, stars, log = [], [], [], []
        for (f, hd, img), (ra, dec, V) in zip(frames, eph):
            w = WCS(hd)
            px, py = (float(v) for v in w.all_world2pix(ra, dec, 0))
            bkg = np.nanmedian(img)
            sig = 1.4826 * np.nanmedian(np.abs(img - bkg))
            c = window(img - bkg, px, py)
            if c is None:
                continue
            stamp = c[half - 6:half + 7, half - 6:half + 7] / sig
            if not np.isfinite(img[int(round(py)), int(round(px))]) or np.max(np.abs(stamp)) > 40 or c[half, half] / sig < -3:
                log.append(dict(file=os.path.basename(f), mjd=hd["MJD_OBS"], skipped=True))
                continue
            obj.append(c / sig)
            dn.append((c, hd["MAGZP"], hd["MJD_OBS"]))
            peaks = np.argwhere((img == maximum_filter(np.nan_to_num(img, nan=-1e9), size=5)) & (np.nan_to_num(img - bkg, nan=0) > 8 * sig))
            for (y1, x1) in peaks:
                if abs(x1 - px) < 6 and abs(y1 - py) < 6 or x1 < 3 or y1 < 3 or x1 > img.shape[1] - 4 or y1 > img.shape[0] - 4:
                    continue
                sub = np.nan_to_num(img[y1 - 1:y1 + 2, x1 - 1:x1 + 2] - bkg, nan=0)
                if sub.sum() <= 0:
                    continue
                cxs = x1 + (sub.sum(axis=0) @ np.array([-1, 0, 1])) / sub.sum()
                cys = y1 + (sub.sum(axis=1) @ np.array([-1, 0, 1])) / sub.sum()
                s = window(img - bkg, float(cxs), float(cys))
                if s is not None and np.isfinite(s).all() and s[half, half] > 0:
                    stars.append(s / s[half, half])
            log.append(dict(file=os.path.basename(f), mjd=hd["MJD_OBS"], V=V, centre_sigma=float(c[half, half] / sig)))
        if not obj or not stars:
            continue
        OB = np.mean(obj, axis=0) * math.sqrt(len(obj))
        far = OB[rr > 12]
        noise = float(1.4826 * np.median(np.abs(far - np.median(far))))
        offs = []
        for c in obj:
            cr = np.clip(c[half - 2:half + 3, half - 2:half + 3], 0, None)
            if cr.sum() > 0:
                jy, jx = np.indices(cr.shape)
                offs.append(((cr * (jx - 2)).sum() / cr.sum(), (cr * (jy - 2)).sum() / cr.sum()))
        offs = np.array(offs)
        jit = (float(np.std(offs[:, 0])), float(np.std(offs[:, 1]))) if len(offs) else (0.0, 0.0)
        rng = np.random.default_rng(1)
        S = np.median([ndshift(st, (rng.normal(0, jit[1]), rng.normal(0, jit[0])), order=1, mode="nearest") for st in stars], axis=0)
        po, ps = profile(OB / OB[half, half]), profile(S / S[half, half])
        apcorr = -2.5 * math.log10(float(S[rr <= 3.0].sum() / S[rr <= 8.0].sum()))
        mags = {}
        for epoch in sorted({str(int(m // 100)) for _, _, m in dn}):
            sel = [(c, zp) for c, zp, m in dn if str(int(m // 100)) == epoch]
            M = np.mean([c for c, _ in sel], axis=0)
            ap = float(M[rr <= 3.0].sum())
            ann = M[(rr > 6) & (rr < 12)]
            err = 1.4826 * np.median(np.abs(ann - np.median(ann))) * math.sqrt(float((rr <= 3.0).sum()))
            mags[epoch] = dict(n=len(sel), mag=(float(np.mean([zp for _, zp in sel])) - 2.5 * math.log10(ap) - apcorr) if ap > 0 else None, snr=ap / err if err > 0 else None)
        res[band] = dict(n_frames=len(obj), n_stars=len(stars), centre_over_noise=float(OB[half, half] / noise), jitter_px=jit, profile_object=po,
                         profile_stars=ps, aperture_correction=apcorr, magnitudes=mags, frames=log)
        print("%s: %d frames, %d stars; stack centre %.1f sigma; centroid scatter %.2f x %.2f px; aperture correction %.2f mag" % (
            band.upper(), len(obj), len(stars), OB[half, half] / noise, jit[0], jit[1], apcorr))
        print("   profile object / stars (normalised to the centre): " + ", ".join("%d-%d px %.3f/%.3f" % (a, b, vo, vs) for (a, b, vo), (_, _, vs) in zip(po, ps)))
        for ep, m in mags.items():
            print("   MJD %s00s: %d frames, %s = %s (S/N %s)" % (ep, m["n"], band.upper(), "%.2f" % m["mag"] if m["mag"] else "-", "%.1f" % m["snr"] if m["snr"] else "-"))
    json.dump(res, open(os.path.join(d, "stack.json"), "w"), indent=1)
    return res


def darkcomet_cli():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "thermal", "refit", "record", "images", "precovery", "robust", "frames", "field", "stack",
                                    "blockcov", "locus", "forecast", "census", "variants", "normal", "normal-record"))
    p.add_argument("desig")
    p.add_argument("--ephem", default=None)
    p.add_argument("--night", default=None, help="UTC date of the CFHT images to re-measure (precovery, frames, field)")
    p.add_argument("--site", default="568", help="MPC site code for the ephemeris of the precovery images")
    p.add_argument("--boot", type=int, default=200, help="bootstrap resamples (robust)")
    p.add_argument("--workers", type=int, default=8, help="parallel refits (robust)")
    a = p.parse_args()
    if a.mode == "fetch":
        darkcomet_fetch(a.desig)
    elif a.mode == "thermal":
        darkcomet_thermal(a.desig)
    elif a.mode == "refit":
        darkcomet_refit(a.desig, a.ephem)
    elif a.mode == "images":
        darkcomet_images(a.desig)
    elif a.mode == "precovery":
        darkcomet_precovery(a.desig, a.night, a.site)
    elif a.mode == "robust":
        darkcomet_robust(a.desig, a.boot, a.workers, a.ephem)
    elif a.mode == "frames":
        darkcomet_frames(a.desig, a.night, a.site, a.ephem)
    elif a.mode == "field":
        darkcomet_field(a.desig, a.night, a.site)
    elif a.mode == "stack":
        darkcomet_stack(a.desig)
    elif a.mode == "blockcov":
        darkcomet_blockcov(a.desig, a.ephem)
    elif a.mode == "locus":
        darkcomet_locus(a.desig, a.night, a.site, a.ephem)
    elif a.mode == "forecast":
        darkcomet_forecast(a.desig, a.ephem)
    elif a.mode == "census":
        darkcomet_census(a.desig, a.night, a.workers)
    elif a.mode == "variants":
        darkcomet_thermal_variants(a.desig)
    elif a.mode == "normal":
        darkcomet_normal(a.desig, a.ephem)
    elif a.mode == "normal-record":
        darkcomet_normal_record(a.desig)
    else:
        darkcomet_record(a.desig)


replicated_DOC = r"""Replicated bodies: LCDB asteroids matching in period (2e-4), H (0.1 mag) and amplitude (0.05 mag) on proper orbits over 300 m/s apart, groups of three against shuffled periods.

    python -I seti.py solar_system replicated analyse
    python -I seti.py solar_system replicated record"""

replicated_ROOT = os.environ.get("REPLICATED_ROOT", core.data("replicated"))
replicated_OUT = "replicated_bodies.json"
replicated_DP, replicated_DH, replicated_DA, replicated_DV = 2e-4, 0.1, 0.05, 0.3
replicated_LCDB = "https://sbnarchive.psi.edu/pds4/non_mission/ast-lightcurve-database_V4_0/data/lc_summary.csv"
replicated_PROPER = "https://sbnarchive.psi.edu/pds4/non_mission/ast.nesvorny.families_V2_0/data/proper_catalog24.tab"


def replicated_load():
    """LCDB rows (number, name, H, period, amplitude, U, family) joined with proper elements by number."""
    import csv
    os.makedirs(replicated_ROOT, exist_ok=True)
    for name, url in (("lc_summary.csv", replicated_LCDB), ("proper_catalog24.tab", replicated_PROPER)):
        p = os.path.join(replicated_ROOT, name)
        if not os.path.exists(p):
            open(p, "wb").write(core.get(url, timeout=1800, retries=2))
    rows = list(csv.reader(open(os.path.join(replicated_ROOT, "lc_summary.csv"), encoding="utf-8", errors="replace")))
    head = next(i for i, r in enumerate(rows) if r and r[0] == "# Column definitions") + 1
    cols = rows[head]
    start = next(i for i, r in enumerate(rows) if r and r[0] == "# Data") + 1
    ix = {c: cols.index(c) for c in ("Number", "Name", "Desig", "Family", "H", "Period", "AmpMin", "AmpMax", "U", "Diam", "IsBinary", "Notes")}
    proper = {}
    for line in open(os.path.join(replicated_ROOT, "proper_catalog24.tab")):
        f = line.split()
        if len(f) >= 12 and f[11].isdigit():
            proper[int(f[11])] = (float(f[0]), float(f[2]), float(f[4]))
    out = []
    for r in rows[start:]:
        if len(r) < len(cols) or not r[ix["Number"]].strip().lstrip("-").isdigit():
            continue
        n = int(r[ix["Number"]])
        H, P, amp = num(r[ix["H"]]), num(r[ix["Period"]]), num(r[ix["AmpMax"]])
        if n <= 0 or n not in proper or H is None or H < -9 or P is None or P <= 0 or amp is None or amp < 0:
            continue
        out.append(dict(number=n, name=r[ix["Name"]] if r[ix["Name"]] != "-" else r[ix["Desig"]], H=H, P=P, amp=amp,
                        U=r[ix["U"]].strip(), family=r[ix["Family"]].strip(), diam=num(r[ix["Diam"]]), binary=r[ix["IsBinary"]].strip(),
                        a=proper[n][0], e=proper[n][1], sini=proper[n][2]))
    return out


def replicated_metric(x, y):
    """Zappala et al. (1990) standard metric between proper-element sets, km/s."""
    a = 0.5 * (x["a"] + y["a"])
    na = 29.78 / math.sqrt(a)
    return na * math.sqrt(1.25 * ((x["a"] - y["a"]) / a) ** 2 + 2 * (x["e"] - y["e"]) ** 2 + 2 * (x["sini"] - y["sini"]) ** 2)


def replicated_groups(objs, periods):
    """Groups of three or more pairwise-matching, pairwise-unrelated objects for the given periods; objs sorted by period."""
    order = np.argsort(periods)
    P = periods[order]
    H = np.array([o["H"] for o in objs])[order]
    A = np.array([o["amp"] for o in objs])[order]
    n = len(P)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = 0
    j0 = 0
    for i in range(n):
        while P[i] - P[j0] > replicated_DP * P[i]:
            j0 += 1
        for j in range(j0, i):
            if abs(H[i] - H[j]) <= replicated_DH and abs(A[i] - A[j]) <= replicated_DA:
                x, y = objs[order[i]], objs[order[j]]
                if replicated_metric(x, y) > replicated_DV:
                    parent[find(i)] = find(j)
                    edges += 1
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(order[i])
    return [g for g in groups.values() if len(g) >= 3], edges


def replicated_analyse():
    objs = [o for o in replicated_load() if o["U"] in ("3", "3-")]
    print("%d numbered asteroids with LCDB quality 3 or 3- periods, H, amplitudes and proper elements" % len(objs))
    periods = np.array([o["P"] for o in objs])
    groups, edges = replicated_groups(objs, periods)
    print("%d matching unrelated pairs; %d groups of three or more" % (edges, len(groups)))
    rng = np.random.default_rng(3)
    H = np.array([o["H"] for o in objs])
    bins = np.floor(H / 0.5)
    null, null_edges = [], []
    for _ in range(200):
        sh = periods.copy()
        for b in np.unique(bins):
            m = np.nonzero(bins == b)[0]
            sh[m] = periods[rng.permutation(m)]
        g, e = replicated_groups(objs, sh)
        null.append(len(g))
        null_edges.append(e)
    print("periods shuffled within half-magnitude bins of H, 200 times: pairs %.1f +- %.1f, groups of three or more %.2f +- %.2f, max %d" % (
        np.mean(null_edges), np.std(null_edges), np.mean(null), np.std(null), max(null)))
    out = []
    for g in sorted(groups, key=len, reverse=True):
        mem = [objs[i] for i in g]
        print("  group of %d: " % len(mem) + "; ".join("(%d) %s P %.4f h H %.2f amp %.2f a %.3f e %.3f sini %.3f fam %s%s" % (
            m["number"], m["name"], m["P"], m["H"], m["amp"], m["a"], m["e"], m["sini"], m["family"], " bin" if m["binary"] == "Y" else "") for m in mem))
        out.append(mem)
    json.dump(dict(n=len(objs), pairs=edges, groups=out, null_groups_mean=float(np.mean(null)), null_groups_sd=float(np.std(null)), null_groups_max=int(max(null)),
                   null_pairs_mean=float(np.mean(null_edges)), null_pairs_sd=float(np.std(null_edges)),
                   dp=replicated_DP, dh=replicated_DH, da=replicated_DA, dv_kms=replicated_DV), open(core.results(replicated_OUT), "w"), indent=1)


def replicated_record():
    d = json.load(open(core.results(replicated_OUT)))
    core.record_search(
        "replicated_bodies", "Replicated bodies on unrelated orbits",
        archive="LCDB summary table V4.0 (PDS); Nesvorny family bundle V2.0 proper elements 2024 (PDS)", release="LCDB 2021-09; proper elements 2024",
        selection="numbered asteroids with LCDB quality 3 or 3- periods and proper elements", code=os.path.abspath(__file__), entry="replicated",
        n_targets=d["n"], params=dict(dP_rel=d["dp"], dH=d["dh"], dAmp=d["da"], d_min_kms=d["dv_kms"]),
        null_summary="periods shuffled within half-magnitude bins of H: pairs %.1f +- %.1f, groups %.2f +- %.2f (max %d)" % (
            d["null_pairs_mean"], d["null_pairs_sd"], d["null_groups_mean"], d["null_groups_sd"], d["null_groups_max"]),
        sensitivity="a series of three or more copies with periods to 2e-4, H to 0.1 and amplitude to 0.05 among the quality-3 objects",
        ledger="Replicated bodies on unrelated orbits",
        result="%d objects: %d matching unrelated pairs against %.1f +- %.1f shuffled; %d groups of three or more against %.2f +- %.2f" % (
            d["n"], d["pairs"], d["null_pairs_mean"], d["null_pairs_sd"], len(d["groups"]), d["null_groups_mean"], d["null_groups_sd"]))
    core.clear_targets("replicated_bodies")
    core.record_targets("replicated_bodies", [dict(target="mpc:%d" % m["number"], stat=len(g), unit="members of the matching group",
                                                        vet="period, H and amplitude shared with %d unrelated bodies" % (len(g) - 1),
                                                        note="P %.4f h, H %.2f, amp %.2f, a %.3f e %.3f sini %.3f" % (m["P"], m["H"], m["amp"], m["a"], m["e"], m["sini"]))
                                                   for g in d["groups"] for m in g])
    print("recorded")


def replicated_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": replicated_analyse, "record": replicated_record}[mode]()


limbs_DOC = r"""Straight limbs in asteroid occultations: longest multi-site run of chord ends fitting a line within timing errors, over the diameter, against the fitted ellipse's own runs.

    python -I seti.py solar_system limbs analyse
    python -I seti.py solar_system limbs record"""

limbs_ROOT = os.environ.get("LIMBS_ROOT", core.data("occultations"))
limbs_BASE = "https://sbnarchive.psi.edu/pds4/non_mission/smallbodiesoccultations_V4_0/data/"
limbs_OUT = "occultation_limbs.json"
limbs_RE_KM = 6378.137
limbs_MIN_CHORDS, limbs_RUN = 5, 4


def limbs_load():
    """Events of fit quality 3-4 with five or more chords: limb points in the asteroid frame (km) with along-motion timing errors."""
    import csv
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    os.makedirs(limbs_ROOT, exist_ok=True)
    for name in ("Asteroid_2024Mar.psv", "AsteroidTimes_2024Mar.psv"):
        p = os.path.join(limbs_ROOT, name)
        if not os.path.exists(p):
            open(p, "wb").write(core.get(limbs_BASE + name, timeout=900, retries=2))
    ev = {}
    for r in csv.DictReader(open(os.path.join(limbs_ROOT, "Asteroid_2024Mar.psv"), encoding="utf-8", errors="replace"), delimiter="|"):
        try:
            if int(r["FIT_QUAL_CODE"]) < 3 or int(r["CHORD_NUM_MAIN"] or 0) < limbs_MIN_CHORDS:
                continue
            ev[r["SEQ_NUM"]] = dict(seq=r["SEQ_NUM"], date=r["OBS_DATE"], t0=float(r["TIME_ZERO"]), ra=float(r["STAR_RA_APPT"]) * 15.0, dec=float(r["STAR_DEC_APPT"]),
                                    vx=[float(r[k] or 0) for k in ("AST_EAST_MOTION", "AST_EAST_MOTION_2ND", "AST_EAST_MOTION_3RD")],
                                    vy=[float(r[k] or 0) for k in ("AST_NORTH_MOTION", "AST_NORTH_MOTION_2ND", "AST_NORTH_MOTION_3RD")],
                                    desig=r["AST_DESIG"], name=r["AST_NAME"], dia=num(r["AST_DIA_NOMINAL"]), major=num(r["MAJOR_AXIS"]), minor=num(r["MINOR_AXIS"]),
                                    pa=num(r["MAJOR_AXIS_PA"]), qual=int(r["FIT_QUAL_CODE"]), n_chords=int(r["CHORD_NUM_MAIN"]), chords=[],
                                    fx=float(r["FPLANE_X_MAIN"]), fy=float(r["FPLANE_Y_MAIN"]), fhour=float(r["EVENT_HOUR"]))
        except (ValueError, KeyError):
            continue
    for r in csv.DictReader(open(os.path.join(limbs_ROOT, "AsteroidTimes_2024Mar.psv"), encoding="utf-8", errors="replace"), delimiter="|"):
        e = ev.get(r["SEQ_NUM"])
        if e is None or r["OBS_NAME1"] == "PREDICTION" or r["D_EVENTCODE"] != "D" or r["R_EVENTCODE"] != "R" or r["D_WEIGHT"] == "0" or r["R_WEIGHT"] == "0":
            continue
        try:
            lon = abs(float(r["OBS_LONGITUDE_DEG"])) + float(r["OBS_LONGITUDE_MIN"] or 0) / 60 + float(r["OBS_LONGITUDE_SEC"] or 0) / 3600
            lat = (abs(float(r["OBS_LATITUDE_DEG"])) + float(r["OBS_LATITUDE_MIN"] or 0) / 60 + float(r["OBS_LATITUDE_SEC"] or 0) / 3600) * (1 if r["OBS_LATITUDE_NS"] == "N" else -1)
            alt = float(r["OBS_ALTITUDE"] or 0)
            td = float(r["D_HR"]) + float(r["D_MIN"]) / 60 + float(r["D_SEC"]) / 3600
            tr = float(r["R_HR"]) + float(r["R_MIN"]) / 60 + float(r["R_SEC"]) / 3600
        except ValueError:
            continue
        default = 0.5 if r["OBS_METHOD"] in ("a", "b", "c") and r["OBS_TIMESRC"] in ("a", "d", "f") else (1.5 if r["OBS_METHOD"] in ("a", "b", "c") else 1.0)
        ed, er = num(r["D_UNCERT"]) or default, num(r["R_UNCERT"]) or default
        # one chord per site
        key = (round(lon, 2), round(lat, 2))
        if any(c["key"] == key for c in e["chords"]):
            continue
        e["chords"].append(dict(site=(lon, lat, alt), key=key, td=td, tr=tr, ed=ed, er=er, method=r["OBS_METHOD"], name=r["OBS_NAME1"]))
    out = []
    for e in ev.values():
        if len(e["chords"]) < limbs_MIN_CHORDS:
            continue
        a, d = math.radians(e["ra"]), math.radians(e["dec"])
        day = Time(e["date"] + "T00:00:00", scale="utc")
        dia = e["dia"] or 0.5 * ((e["major"] or 0) + (e["minor"] or 0))
        if not dia:
            continue

        def shadow(t):
            dt = t - e["t0"]
            return (e["vx"][0] * dt + e["vx"][1] * dt ** 2 + e["vx"][2] * dt ** 3) * limbs_RE_KM, (e["vy"][0] * dt + e["vy"][1] * dt ** 2 + e["vy"][2] * dt ** 3) * limbs_RE_KM
        # fitted asteroid position carried to the time origin
        sx0, sy0 = shadow(e["fhour"])
        cx0, cy0 = e["fx"] - sx0, e["fy"] - sy0
        v = math.hypot(e["vx"][0], e["vy"][0]) * limbs_RE_KM / 3600.0  # km/s along the motion
        ux, uy = e["vx"][0] / math.hypot(e["vx"][0], e["vy"][0]), e["vy"][0] / math.hypot(e["vx"][0], e["vy"][0])
        pts, dropped = [], 0
        for c in e["chords"]:
            best = None
            for sign in (1, -1):
                loc = EarthLocation.from_geodetic(sign * c["site"][0], c["site"][1], c["site"][2])
                gx, gy, gz = (float(w) / (limbs_RE_KM * 1000.0) for w in (loc.x.value, loc.y.value, loc.z.value))
                rho_c, rho_s = math.hypot(gx, gy), gz
                lam0 = math.atan2(gy, gx)
                cand = []
                for t, et, kind in ((c["td"], c["ed"], "D"), (c["tr"], c["er"], "R")):
                    tt = day + t / 24.0
                    theta = tt.sidereal_time("apparent", "greenwich").rad + lam0
                    h = theta - a
                    xi = rho_c * math.sin(h)
                    eta = rho_s * math.cos(d) - rho_c * math.sin(d) * math.cos(h)
                    sx, sy = shadow(t)
                    cand.append(dict(x=xi * limbs_RE_KM - sx - cx0, y=eta * limbs_RE_KM - sy - cy0, err=et * v, kind=kind, ux=ux, uy=uy, site=c["name"]))
                off = math.hypot(0.5 * (cand[0]["x"] + cand[1]["x"]), 0.5 * (cand[0]["y"] + cand[1]["y"]))
                if best is None or off < best[0]:
                    best = (off, cand)
            if best[0] <= 2.0 * dia:
                pts.extend(best[1])
            else:
                dropped += 1
        if len(pts) < 2 * limbs_MIN_CHORDS:
            continue
        e["points"], e["dropped"], e["dia"] = pts, dropped, dia
        out.append(e)
    return out


def limbs_statistic(pts, rng=None, ellipse=None):
    """Longest run of angle-ordered limb points from distinct sites fitting a line at reduced chi-square below 2, over the mean diameter, with the fitted ellipse; with rng, points of that ellipse instead."""
    x = np.array([p["x"] for p in pts])
    y = np.array([p["y"] for p in pts])
    err = np.array([p["err"] for p in pts])
    ux = np.array([p["ux"] for p in pts])
    uy = np.array([p["uy"] for p in pts])
    site = [p["site"] for p in pts]
    if ellipse is None:
        # algebraic least-squares ellipse started at the points' mean
        cx, cy = x.mean(), y.mean()
        A = np.column_stack([(x - cx) ** 2, (x - cx) * (y - cy), (y - cy) ** 2, x - cx, y - cy])
        try:
            coef = np.linalg.lstsq(A, np.ones(len(x)), rcond=None)[0]
            a_, b_, c_, d_, e_ = coef
            M = np.array([[2 * a_, b_], [b_, 2 * c_]])
            x0, y0 = np.linalg.solve(M, -np.array([d_, e_]))
            cx, cy = cx + x0, cy + y0
            k = 1 + a_ * x0 ** 2 + b_ * x0 * y0 + c_ * y0 ** 2 + d_ * x0 + e_ * y0
            w, V = np.linalg.eigh(np.array([[a_, b_ / 2], [b_ / 2, c_]]) / k)
            axes = 2 / np.sqrt(np.maximum(w, 1e-12))
            ellipse = dict(cx=float(cx), cy=float(cy), major=float(axes.max()), minor=float(axes.min()), ok=bool(np.all(w > 0)),
                           theta=float(math.atan2(V[1, int(np.argmin(w))], V[0, int(np.argmin(w))])))
        except np.linalg.LinAlgError:
            ellipse = dict(cx=float(cx), cy=float(cy), major=float("nan"), minor=float("nan"), ok=False, theta=0.0)
    if rng is not None:
        # ellipse points at the same angles, perturbed along the motion by their errors
        ang = np.arctan2(y - ellipse["cy"], x - ellipse["cx"])
        sa, sb, th = 0.5 * ellipse["major"], 0.5 * ellipse["minor"], ellipse.get("theta", 0.0)
        rr = sa * sb / np.sqrt((sb * np.cos(ang - th)) ** 2 + (sa * np.sin(ang - th)) ** 2)
        x = ellipse["cx"] + rr * np.cos(ang) + rng.normal(0, 1, len(x)) * err * ux
        y = ellipse["cy"] + rr * np.sin(ang) + rng.normal(0, 1, len(y)) * err * uy
    ang = np.arctan2(y - ellipse["cy"], x - ellipse["cx"])
    order = np.argsort(ang)
    n = len(order)
    best = (0.0, None)
    for k in range(limbs_RUN, n + 1):
        for s in range(n):
            idx = [order[(s + i) % n] for i in range(k)]
            if len({site[i] for i in idx}) < k:
                continue
            px, py, pe = x[idx], y[idx], err[idx]
            # principal-axis line; perpendicular residuals with projected errors
            mx, my = px.mean(), py.mean()
            cov = np.cov(np.vstack([px - mx, py - my]))
            w, V = np.linalg.eigh(cov)
            nx, ny = V[:, 0]  # best-line normal
            res = (px - mx) * nx + (py - my) * ny
            sig = np.abs(ux[idx] * nx + uy[idx] * ny) * pe + 0.05 * pe
            chi2 = float(np.sum((res / sig) ** 2) / max(k - 2, 1))
            if chi2 < 2.0:
                length = float(math.hypot(px.max() - px.min(), py.max() - py.min()))
                if length > best[0]:
                    best = (length, idx)
    return best, ellipse


def limbs_analyse():
    events = limbs_load()
    print("%d events of fit quality 3 or 4 with %d or more positive chords" % (len(events), limbs_MIN_CHORDS))
    rng = np.random.default_rng(11)
    rows = []
    bad = 0
    for e in events:
        (length, idx), ell = limbs_statistic(e["points"])
        dia = e["dia"]
        if not ell["ok"] or not (0.5 * dia <= ell["major"] <= 2.0 * dia) or length > 1.05 * dia:
            bad += 1
            continue
        frac = length / dia
        null = []
        for _ in range(100):
            (ln, _i), _ = limbs_statistic(e["points"], rng=rng, ellipse=ell)
            null.append(ln / dia)
        null = np.array(null)
        rows.append(dict(seq=e["seq"], desig=e["desig"], name=e["name"], date=e["date"], n_chords=len(e["points"]) // 2, dropped=e["dropped"], n_points=len(e["points"]), dia=dia,
                         major=ell["major"], minor=ell["minor"], straight_km=length, straight_frac=frac, null_median=float(np.median(null)), null_99=float(np.percentile(null, 99)),
                         p=float(np.mean(null >= frac)), run_points=int(len(idx)) if idx else 0, run_sites=sorted({e["points"][i]["site"] for i in idx}) if idx else [], qual=e["qual"]))
    rows.sort(key=lambda r: -(r["straight_frac"] - r["null_99"]))
    fr = np.array([r["straight_frac"] for r in rows])
    ex = [r for r in rows if r["p"] < 0.01]
    print("%d events measured (%d with chords that disagree or an ellipse off the nominal size); straight run over the diameter: median %.2f, 90th percentile %.2f; "
          "beyond the ellipse null at p < 0.01: %d" % (len(rows), bad, np.median(fr), np.percentile(fr, 90), len(ex)))
    for r in rows[:30]:
        print("  (%s) %s %s: %d chords (%d dropped), D %.0f km (ellipse %.0f x %.0f); straight run %.0f km = %.2f D over %d points of %d sites; ellipse null median %.2f, 99%% %.2f, p %.2f" % (
            r["desig"], r["name"], r["date"], r["n_chords"], r["dropped"], r["dia"], r["major"], r["minor"], r["straight_km"], r["straight_frac"], r["run_points"], len(r["run_sites"]),
            r["null_median"], r["null_99"], r["p"]))
    json.dump(dict(n_events=len(rows), n_bad=bad, rows=rows, min_chords=limbs_MIN_CHORDS, run=limbs_RUN), open(core.results(limbs_OUT), "w"), indent=1)


def limbs_record():
    d = json.load(open(core.results(limbs_OUT)))
    rows = d["rows"]
    ex = [r for r in rows if r["p"] < 0.01]
    core.record_search(
        "occultation_limbs", "Straight limbs in occultation profiles", archive="PDS small-bodies occultations V4.0 (Herald et al.)", release="2024 March tables",
        selection="events of fit quality 3 or 4 with five or more positive chords within two diameters of the fitted position", code=os.path.abspath(__file__), entry="limbs", n_targets=len(rows),
        params=dict(min_chords=d["min_chords"], run=d["run"], chi2_max=2.0, ellipse_axis_range=[0.5, 2.0], run_max=1.05),
        null_summary="each event's points replaced by the fitted ellipse's at the same angles with the same timing errors, 100 times; %d events whose chords disagree or whose ellipse is off the nominal size set aside" % d.get("n_bad", 0),
        sensitivity="a straight run of four or more limb points over a stated fraction of the diameter, within the chords' timing errors",
        ledger="Straight limbs in occultation profiles",
        result="%d events: straight run over the diameter median %.2f, 90th percentile %.2f; %d beyond the ellipse null at p below 0.01, the longest %s" % (
            len(rows), float(np.median([r["straight_frac"] for r in rows])), float(np.percentile([r["straight_frac"] for r in rows], 90)), len(ex),
            "; ".join("(%s) %s %s %.2f D over %d points of %d sites" % (r["desig"], r["name"], r["date"], r["straight_frac"], r["run_points"], len(r["run_sites"])) for r in rows[:6])))
    core.clear_targets("occultation_limbs")
    core.record_targets("occultation_limbs", [dict(target="mpc:%s" % r["desig"], stat=r["straight_frac"], p=r["p"], unit="longest straight run of limb points over the diameter",
                                                       vet="%d chords, %d points; ellipse null median %.2f, 99 per cent %.2f%s" % (r["n_chords"], r["n_points"], r["null_median"], r["null_99"],
                                                                                                                        "; run of %d points from %d sites" % (r["run_points"], len(r["run_sites"]))),
                                                       note="%s %s, D %.0f km, run %.0f km over %d points" % (r["name"], r["date"], r["dia"], r["straight_km"], r["run_points"])) for r in rows])
    print("recorded %d events" % len(rows))


def limbs_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": limbs_analyse, "record": limbs_record}[mode]()


plates_DOC = r"""Plate-like bodies: LCDB amplitudes reduced to zero phase above 1.6 mag at quality 2 or better below 15 deg phase, quality 1 as the spurious rate.

    python -I seti.py solar_system plates analyse
    python -I seti.py solar_system plates record"""

plates_OUT = "plate_bodies.json"
plates_SLOPE, plates_PHASE_MAX, plates_A0 = 0.03, 15.0, 1.6
plates_DETAILS = "https://sbnarchive.psi.edu/pds4/non_mission/ast-lightcurve-database_V4_0/data/lc_details.csv"


def plates_analyse():
    import csv
    p = os.path.join(replicated_ROOT, "lc_details.csv")
    if not os.path.exists(p):
        open(p, "wb").write(core.get(plates_DETAILS, timeout=1800, retries=2))
    rows = list(csv.reader(open(p, encoding="utf-8", errors="replace")))
    cols = rows[next(i for i, r in enumerate(rows) if r and r[0] == "# Column definitions") + 1]
    start = next(i for i, r in enumerate(rows) if r and r[0] == "# Data") + 1
    ix = {c: cols.index(c) for c in ("Number", "Name", "Phase", "AmpMin", "AmpMax", "AmpFlag", "U", "Period", "Diam", "H", "Class", "IsBinary", "Notes", "ShortRef", "DateObs", "SumAmp")}
    recs = []
    for r in rows[start:]:
        if len(r) < len(cols):
            continue
        amp, ph = num(r[ix["AmpMax"]]), num(r[ix["Phase"]])
        if amp is None or amp < 0 or ph is None or ph < 0:
            continue
        u = r[ix["U"]].strip()
        recs.append(dict(number=r[ix["Number"]].strip(), name=r[ix["Name"]].strip(), amp=amp, phase=ph, a0=amp / (1 + plates_SLOPE * ph), U=u,
                         period=num(r[ix["Period"]]), diam=num(r[ix["Diam"]]), H=num(r[ix["H"]]), cls=r[ix["Class"]].strip(), binary=r[ix["IsBinary"]].strip(),
                         notes=r[ix["Notes"]].strip(), ref=r[ix["ShortRef"]].strip(), date=r[ix["DateObs"]].strip(), sum_amp=num(r[ix["SumAmp"]])))
    low = [x for x in recs if x["phase"] <= plates_PHASE_MAX]
    good = [x for x in low if x["U"] in ("2", "2+", "3-", "3") and x["binary"] != "Y" and "T" not in x["notes"]]
    poor = [x for x in low if x["U"] in ("1", "1+", "1-")]
    a0g, a0p = np.array([x["a0"] for x in good]), np.array([x["a0"] for x in poor])
    print("%d detail records with an amplitude and a phase angle, %d at phase angles up to %.0f deg: quality 2 and 3 %d (A_0 above %.1f mag: %d, %.4f; above 2.0: %d; above 2.5: %d), "
          "quality 1 %d (above %.1f: %d, %.4f)" % (len(recs), len(low), plates_PHASE_MAX, len(good), plates_A0, int((a0g > plates_A0).sum()), float(np.mean(a0g > plates_A0)),
                                                   int((a0g > 2.0).sum()), int((a0g > 2.5).sum()), len(poor), plates_A0, int((a0p > plates_A0).sum()), float(np.mean(a0p > plates_A0)) if len(poor) else 0))
    tail = sorted((x for x in good if x["a0"] > plates_A0), key=lambda x: -x["a0"])
    objs = {}
    for x in tail:
        objs.setdefault((x["number"], x["name"]), []).append(x)
    print("  %d objects in the tail:" % len(objs))
    for (n, name), xs in sorted(objs.items(), key=lambda kv: -max(x["a0"] for x in kv[1])):
        x = max(xs, key=lambda y: y["a0"])
        print("  %s %s: A_0 %.2f mag (A %.2f at phase %.1f deg, U %s, %s %s), %d records in the tail; P %s h, D %s km, H %s, class %s, binary %s, notes %s" % (
            n, name, x["a0"], x["amp"], x["phase"], x["U"], x["ref"], x["date"], len(xs), "%.3f" % x["period"] if x["period"] else "-", "%.1f" % x["diam"] if x["diam"] and x["diam"] > 0 else "-",
            "%.1f" % x["H"] if x["H"] and x["H"] > -9 else "-", x["cls"] or "-", x["binary"], x["notes"] or "-"))
    json.dump(dict(n_records=len(recs), n_low_phase=len(low), n_good=len(good), n_poor=len(poor), frac_good=float(np.mean(a0g > plates_A0)), frac_poor=float(np.mean(a0p > plates_A0)) if len(poor) else None,
                   tail=tail, phase_max=plates_PHASE_MAX, a0_min=plates_A0, slope=plates_SLOPE), open(core.results(plates_OUT), "w"), indent=1)


def plates_record():
    d = json.load(open(core.results(plates_OUT)))
    objs = {}
    for x in d["tail"]:
        objs.setdefault(x["number"] + " " + x["name"], []).append(x)
    core.record_search(
        "plate_bodies", "Plate-like bodies", archive="LCDB details table V4.0 (PDS, Warner et al.)", release="LCDB 2021-09",
        selection="detail records with an amplitude and a phase angle at phase angles up to %.0f deg" % d["phase_max"], code=os.path.abspath(__file__), entry="plates",
        n_targets=d["n_good"], params=dict(slope=d["slope"], phase_max=d["phase_max"], a0_min=d["a0_min"]),
        null_summary="the quality-1 records' tail above %.1f mag: %s" % (d["a0_min"], "%.4f" % d["frac_poor"] if d["frac_poor"] is not None else "-"),
        sensitivity="an amplitude at zero phase above %.1f mag, an axial ratio above 4.4 for an ellipsoid" % d["a0_min"], ledger="Plate-like bodies",
        result="%d quality-2 and 3 records at low phase, %d above %.1f mag (%.4f) in %d objects against %.4f of the quality-1 records: %s" % (
            d["n_good"], len(d["tail"]), d["a0_min"], d["frac_good"], len(objs), d["frac_poor"] or 0,
            "; ".join("%s A_0 %.2f" % (k, max(x["a0"] for x in v)) for k, v in sorted(objs.items(), key=lambda kv: -max(x["a0"] for x in kv[1]))[:12])))
    core.clear_targets("plate_bodies")
    core.record_targets("plate_bodies", [dict(target="mpc:%s" % (k.split(" ")[0] if k.split(" ")[0] not in ("", "-9") else k), stat=max(x["a0"] for x in v), unit="peak-to-peak amplitude at zero phase, mag",
                                                 vet="%d records; class %s; %s" % (len(v), v[0]["cls"] or "-", "binary" if v[0]["binary"] == "Y" else v[0]["notes"] or "elongated body"),
                                                 note="A %.2f at phase %.1f deg, quality %s, %s" % (v[0]["amp"], v[0]["phase"], v[0]["U"], v[0]["ref"])) for k, v in objs.items()])
    print("recorded %d objects" % len(objs))


def plates_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": plates_analyse, "record": plates_record}[mode]()


# --------------------------------------------------------------------------------------------------------------

isoarrival_DOC = r"""Interstellar arrivals along the ecliptic: the chance of an orbit within 5 deg of the ecliptic, and of 3I/ATLAS's passes by Venus, Mars and Jupiter, for interstellar objects carrying the velocities of the Gaia DR3 stars within 100 pc (arrivals in proportion to speed and to the focused cross-section for perihelia within 2 AU), against isotropic arrivals.

    python -I seti.py solar_system isoarrival fetch                 (1I, 2I and 3I from JPL SBDB, 3I's planetary passes on astropy's ephemeris; Gaia DR3 velocities within 100 pc)
    python -I seti.py solar_system isoarrival run [--n 20000000] [--passes 200000] [--procs 12]
    python -I seti.py solar_system isoarrival record"""

isoarrival_ROOT = os.environ.get("ISOARRIVAL_ROOT", core.data("isoarrival"))
isoarrival_GM = 2.959122082855911e-4  # AU^3 day^-2
isoarrival_KMS = 86400 / 1.495978707e8  # AU/day in a km/s
isoarrival_QMAX = 2.0  # AU, the perihelia a survey finds
isoarrival_ALIGN = 5.0  # deg from the ecliptic plane
isoarrival_SPAN, isoarrival_DT = 1100.0, 0.5  # days either side of perihelion, step
# JPL approximate elements (Standish 1992), J2000 ecliptic: a (AU), e, i, longitude of perihelion, node (deg); the phase is drawn
isoarrival_PLANETS = {"Venus": (0.72333566, 0.00677672, 3.39467605, 131.60246718, 76.67984255),
                      "Mars": (1.52371034, 0.09339410, 1.84969142, -23.94362959, 49.55953891),
                      "Jupiter": (5.20288700, 0.04838624, 1.30439695, 14.72847983, 100.47390909)}


def isoarrival_asymptote(o):
    """Velocity at infinity (AU/day) and impact vector (AU) of a hyperbolic orbit from its elements (q, e, i, om, w)."""
    q, e = o["q"], o["e"]
    i, O, w = (math.radians(o[k]) for k in ("i", "om", "w"))
    P = np.array([math.cos(O) * math.cos(w) - math.sin(O) * math.sin(w) * math.cos(i),
                  math.sin(O) * math.cos(w) + math.cos(O) * math.sin(w) * math.cos(i), math.sin(w) * math.sin(i)])
    Q = np.array([-math.cos(O) * math.sin(w) - math.sin(O) * math.cos(w) * math.cos(i),
                  -math.sin(O) * math.sin(w) + math.cos(O) * math.cos(w) * math.cos(i), math.cos(w) * math.sin(i)])
    v = math.sqrt(isoarrival_GM * (e - 1) / q)
    nu = math.acos(-1 / e)
    vh = -(math.cos(nu) * P - math.sin(nu) * Q)
    return v * vh, (e * P - vh) * isoarrival_GM / v ** 2


def isoarrival_own_passes(o):
    """Least distances (AU) of an orbit, from its elements, to Venus, Mars and Jupiter on astropy's ephemeris, J2000 ecliptic."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    vv, bv = isoarrival_asymptote(o)
    t = np.arange(-isoarrival_SPAN, isoarrival_SPAN + 0.25, 0.25)
    r = isoarrival_track(vv[None, :], bv[None, :], t)[0]
    tt = Time(o["tp"] + t, format="jd", scale="tdb")
    eps = math.radians(23.4392911)
    sun = get_body_barycentric("sun", tt).xyz.to("AU").value
    out = {}
    for name in ("Venus", "Mars", "Jupiter"):
        x, y, z = get_body_barycentric(name.lower(), tt).xyz.to("AU").value - sun
        pe = np.stack([x, y * math.cos(eps) + z * math.sin(eps), -y * math.sin(eps) + z * math.cos(eps)], axis=1)
        out[name] = float(np.min(np.linalg.norm(r - pe, axis=1)))
    return out


def isoarrival_fetch():
    """1I, 2I and 3I's full-precision elements from JPL SBDB, 3I's least distances to Venus, Mars and Jupiter, and the heliocentric ecliptic velocities (km/s) of the Gaia DR3 stars within 100 pc with radial velocities."""
    os.makedirs(isoarrival_ROOT, exist_ok=True)
    objs = {}
    for des in ("1I", "2I", "3I"):
        d = json.loads(core.get("https://ssd-api.jpl.nasa.gov/sbdb.api?" + urllib.parse.urlencode({"sstr": des, "full-prec": "1"}),
                                timeout=120))
        objs[des] = dict(name=d["object"]["fullname"], **{e["name"]: num(e["value"]) for e in d["orbit"]["elements"]})
    objs["3I"]["passes"] = isoarrival_own_passes(objs["3I"])
    json.dump(objs, open(os.path.join(isoarrival_ROOT, "objects.json"), "w"), indent=1)
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
    rows = core.tap("gaia", "SELECT ra, dec, parallax, pmra, pmdec, radial_velocity FROM gaiadr3.gaia_source WHERE parallax > 10 "
                            "AND parallax_over_error > 10 AND radial_velocity IS NOT NULL", maxrec=500000)
    col = {k: np.array([float(r[k]) for r in rows]) for k in ("ra", "dec", "parallax", "pmra", "pmdec", "radial_velocity")}
    c = SkyCoord(ra=col["ra"] * u.deg, dec=col["dec"] * u.deg, distance=(1000 / col["parallax"]) * u.pc,
                 pm_ra_cosdec=col["pmra"] * u.mas / u.yr, pm_dec=col["pmdec"] * u.mas / u.yr,
                 radial_velocity=col["radial_velocity"] * u.km / u.s, frame="icrs")
    v = c.transform_to(BarycentricMeanEcliptic()).velocity.d_xyz.to(u.km / u.s).value.T
    np.savez_compressed(os.path.join(isoarrival_ROOT, "velocities.npz"), v=v.astype(np.float32))
    print("%d stars within 100 pc with radial velocities; 3I's passes (AU) %s" % (len(v), {k: round(x, 3) for k, x in objs["3I"]["passes"].items()}))


def isoarrival_draw(v, n, rng):
    """n arrivals: star velocities drawn in proportion to speed times the focused cross-section for perihelia within QMAX, impact points uniform over that cross-section; velocity (AU/day), impact vector (AU), perihelion, inclination (deg)."""
    s = np.linalg.norm(v, axis=1) * isoarrival_KMS
    bmax2 = isoarrival_QMAX ** 2 + 2 * isoarrival_GM * isoarrival_QMAX / s ** 2
    w = s * bmax2
    k = rng.choice(len(v), n, p=w / w.sum())
    vv, sp = v[k] * isoarrival_KMS, s[k]
    vh = vv / sp[:, None]
    e1 = np.cross(vh, np.where(np.abs(vh[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]))
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(vh, e1)
    b, phi = np.sqrt(bmax2[k] * rng.uniform(0, 1, n)), rng.uniform(0, 2 * math.pi, n)
    bv = b[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    q = (np.sqrt(isoarrival_GM ** 2 + b ** 2 * sp ** 4) - isoarrival_GM) / sp ** 2
    nrm = np.cross(bv, vh)
    inc = np.degrees(np.arccos(np.clip(nrm[:, 2] / np.linalg.norm(nrm, axis=1), -1, 1)))
    return vv, bv, q, inc


def isoarrival_track(vv, bv, t):
    """Heliocentric positions (AU) at times t (days from perihelion) of hyperbolic orbits given by velocity at infinity and impact vector: the eccentricity vector is v^2 b / GM + v-hat."""
    GM = isoarrival_GM
    sp = np.linalg.norm(vv, axis=1)
    vh, b = vv / sp[:, None], np.linalg.norm(bv, axis=1)
    ev = (sp ** 2)[:, None] * bv / GM + vh
    e = np.linalg.norm(ev, axis=1)
    P = ev / e[:, None]
    nrm = np.cross(bv, vh)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    Q = np.cross(nrm, P)
    a = GM / sp ** 2
    M = np.sqrt(GM / a ** 3)[:, None] * t[None, :]
    H = np.arcsinh(M / e[:, None])
    for _ in range(30):
        H -= (e[:, None] * np.sinh(H) - H - M) / (e[:, None] * np.cosh(H) - 1)
    x, y = a[:, None] * (e[:, None] - np.cosh(H)), a[:, None] * np.sqrt(e ** 2 - 1)[:, None] * np.sinh(H)
    return x[..., None] * P[:, None, :] + y[..., None] * Q[:, None, :]


def isoarrival_planet(name, t, M0):
    """Positions (AU) of a planet on its mean ellipse at times t (days) from mean anomalies M0 (rad) at t = 0."""
    a, e, i, varpi, node = isoarrival_PLANETS[name]
    i, w, O = math.radians(i), math.radians(varpi - node), math.radians(node)
    M = M0[:, None] + math.sqrt(isoarrival_GM / a ** 3) * t[None, :]
    E = M.copy()
    for _ in range(8):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    x, y = a * (np.cos(E) - e), a * math.sqrt(1 - e * e) * np.sin(E)
    cw, sw, co, so, ci, si = math.cos(w), math.sin(w), math.cos(O), math.sin(O), math.cos(i), math.sin(i)
    return np.stack([(co * cw - so * sw * ci) * x + (-co * sw - so * cw * ci) * y, (so * cw + co * sw * ci) * x + (-so * sw + co * cw * ci) * y,
                     (sw * si) * x + (cw * si) * y], axis=-1)


def isoarrival_passes(job):
    """Least distances (AU) to Venus, Mars and Jupiter of one batch of arrivals with drawn planetary phases."""
    vv, bv, seed = job
    rng = np.random.default_rng(seed)
    t = np.arange(-isoarrival_SPAN, isoarrival_SPAN + isoarrival_DT, isoarrival_DT)
    r = isoarrival_track(vv, bv, t)
    out = []
    for name in ("Venus", "Mars", "Jupiter"):
        d = np.linalg.norm(r - isoarrival_planet(name, t, rng.uniform(0, 2 * math.pi, len(vv))), axis=-1)
        k = np.clip(np.argmin(d, axis=1), 1, len(t) - 2)
        y0, y1, y2 = (d[np.arange(len(vv)), k + j] for j in (-1, 0, 1))
        den = y0 - 2 * y1 + y2
        out.append(np.where(den > 0, y1 - (y0 - y2) ** 2 / (8 * np.where(den > 0, den, 1)), y1))  # parabola through the least three
    return np.stack(out, axis=1)


def isoarrival_run(n=20000000, passes=200000, procs=12):
    """Inclinations of n arrivals; planetary passes of `passes` of them and of as many drawn within ALIGN of the ecliptic."""
    from multiprocessing import Pool
    v = np.load(os.path.join(isoarrival_ROOT, "velocities.npz"))["v"].astype(float)
    rng = np.random.default_rng(20251029)
    inc, q, sp, ali = [], [], [], []
    for _ in range(n // 1000000):
        vv, bv, qq, ii = isoarrival_draw(v, 1000000, rng)
        inc.append(ii)
        q.append(qq)
        sp.append(np.linalg.norm(vv, axis=1) / isoarrival_KMS)
        m = (ii <= isoarrival_ALIGN) | (ii >= 180 - isoarrival_ALIGN)
        ali.append((vv[m], bv[m], ii[m]))
    inc, q, sp = np.concatenate(inc), np.concatenate(q), np.concatenate(sp)
    av, ab, ai = (np.concatenate([x[j] for x in ali]) for j in range(3))
    vv, bv, _, ii = isoarrival_draw(v, passes, rng)
    sets = dict(all=(vv, bv, ii), aligned=(av[:passes], ab[:passes], ai[:passes]))
    res = {}
    with Pool(procs) as pool:
        for key, (a1, a2, a3) in sets.items():
            jobs = [(a1[j:j + 500], a2[j:j + 500], 7919 * j + len(key)) for j in range(0, len(a1), 500)]
            res[key] = (np.concatenate(pool.map(isoarrival_passes, jobs)), a3)
    np.savez_compressed(os.path.join(isoarrival_ROOT, "arrivals.npz"), inc=inc.astype(np.float32), q=q.astype(np.float32),
                        speed=sp.astype(np.float32), d_all=res["all"][0], i_all=res["all"][1], d_aligned=res["aligned"][0],
                        i_aligned=res["aligned"][1])
    print("%d arrivals, %d within %.0f deg of the ecliptic; passes for %d and %d" % (len(inc), len(ai), isoarrival_ALIGN,
                                                                                    len(res["all"][0]), len(res["aligned"][0])))


def isoarrival_record():
    objs = json.load(open(os.path.join(isoarrival_ROOT, "objects.json")))
    z = np.load(os.path.join(isoarrival_ROOT, "arrivals.npz"))
    inc, sp = z["inc"], z["speed"]
    d3 = np.array([objs["3I"]["passes"][k] for k in ("Venus", "Mars", "Jupiter")])

    def near(d):
        return np.all(d <= d3[None, :], axis=1)
    p_align = np.mean((inc <= isoarrival_ALIGN) | (inc >= 180 - isoarrival_ALIGN))
    p_retro = np.mean(inc >= 180 - isoarrival_ALIGN)
    iso = 1 - math.cos(math.radians(isoarrival_ALIGN))
    fast = sp >= 50
    rows = []
    for des, o in objs.items():
        off = min(o["i"], 180 - o["i"])
        rows.append(dict(target="iso:%s" % des, stat=float(np.mean(np.minimum(inc, 180 - inc) <= off)),
                         unit="fraction of arrivals as close to the ecliptic plane", note="%s: i %.2f deg, q %.3f AU, e %.3f; isotropic %.4f"
                         % (o["name"], o["i"], o["q"], o["e"], 1 - math.cos(math.radians(off)))))
    p_all, p_ali = float(np.mean(near(z["d_all"]))), float(np.mean(near(z["d_aligned"])))
    p_any = float(np.mean(z["d_aligned"].min(axis=1) <= 0.2))
    print("within %.0f deg of the ecliptic: %.4f of arrivals (retrograde %.4f; %.4f at 50 km/s or more), isotropic %.4f"
          % (isoarrival_ALIGN, p_align, p_retro, np.mean(((inc <= isoarrival_ALIGN) | (inc >= 180 - isoarrival_ALIGN))[fast]), iso))
    print("passes at least as close as 3I's (Venus, Mars, Jupiter %s AU): %.2e of arrivals, %.2e of those within %.0f deg"
          % ([round(float(x), 3) for x in d3], p_all, p_ali, isoarrival_ALIGN))
    for j, name in enumerate(("Venus", "Mars", "Jupiter")):
        print("  %s alone: %.4f of arrivals, %.4f of those within %.0f deg" % (name, np.mean(z["d_all"][:, j] <= d3[j]),
                                                                           np.mean(z["d_aligned"][:, j] <= d3[j]), isoarrival_ALIGN))
    for r in rows:
        print("  %s: %.4f (%s)" % (r["target"], r["stat"], r["note"]))
    core.record_search(
        "iso_ecliptic_arrivals", "Interstellar arrivals along the ecliptic", archive="JPL SBDB and CAD; Gaia DR3",
        selection="1I, 2I and 3I; %d arrivals from the velocities of the Gaia DR3 stars within 100 pc" % len(inc),
        code=os.path.abspath(__file__), entry="isoarrival", n_targets=len(rows), n_unit="interstellar objects",
        params=dict(qmax_au=isoarrival_QMAX, align_deg=isoarrival_ALIGN, span_days=isoarrival_SPAN, step_days=isoarrival_DT),
        null_summary="isotropic arrivals give %.4f within %.0f deg of the ecliptic" % (iso, isoarrival_ALIGN),
        sensitivity="arrivals weighted by speed and by the focused cross-section for perihelia within %.0f AU; planets on their mean "
                    "ellipses with drawn phases" % isoarrival_QMAX,
        ledger="Interstellar arrivals along the ecliptic",
        result="%.4f of arrivals lie within %.0f deg of the ecliptic (isotropic %.4f, retrograde alone %.4f); passes at least as close as "
               "3I/ATLAS's by Venus, Mars and Jupiter %.2e of all arrivals and %.2e of those along the ecliptic, %.1e for both together; "
               "along the ecliptic and within 0.2 AU of one of the three planets %.1e" % (p_align, isoarrival_ALIGN, iso, p_retro, p_all, p_ali,
                                                                                        p_align * p_ali, p_align * p_any))
    core.clear_targets("iso_ecliptic_arrivals")
    core.record_targets("iso_ecliptic_arrivals", rows)


def isoarrival_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "run", "record"))
    p.add_argument("--n", type=int, default=20000000)
    p.add_argument("--passes", type=int, default=200000)
    p.add_argument("--procs", type=int, default=12)
    a = p.parse_args()
    if a.mode == "fetch":
        isoarrival_fetch()
    elif a.mode == "run":
        isoarrival_run(a.n, a.passes, a.procs)
    else:
        isoarrival_record()


# --------------------------------------------------------------------------------------------------------------

iso3i_DOC = r"""3I/ATLAS (C/2025 N1) from every MPC position: ASSIST fits on gravity alone and with outgassing accelerations (radial,
transverse and normal, scaled by r^-2 or by the water-ice g(r)), with outgassing that changes at perihelion, and with an impulse
at perihelion or at the closest approaches to Mars and Jupiter; observing nights resampled for the impulses' spread.

    python -I seti.py solar_system iso3i fetch                              (ROOT/obs80.txt, sbdb.json, ObsCodes.html)
    python -I seti.py solar_system iso3i fit [--workers 8]                  (ROOT/fits.json)
    python -I seti.py solar_system iso3i boot [--boot 100] [--workers 8]    (ROOT/boot.json)
    python -I seti.py solar_system iso3i compare [--workers 8]              (ROOT/compare.json: the models on one set of positions)
    python -I seti.py solar_system iso3i scan [--workers 8]                 (ROOT/scan.json: an impulse every 5 days on the binned outgassing)
    python -I seti.py solar_system iso3i boot --bins [--boot 100]          (ROOT/boot_bins.json: nights resampled for the binned impulse fits)
    python -I seti.py solar_system iso3i sites [--workers 8]                (ROOT/sites.json: the binned impulse fits without each busiest site)
    python -I seti.py solar_system iso3i record                             (compare, scan, boot_bins and sites into the coverage store)"""

iso3i_DESIG = "C/2025 N1"
iso3i_ROOT = os.environ.get("ISO3I_ROOT", core.data("iso3i"))
# g(r) = alpha (r/r0)^-nm (1 + (r/r0)^nn)^-nk: the inverse square, and Marsden et al. (1973) for water ice
iso3i_G = {"r2": (1.0, 0.0, 2.0, 0.0, 1.0), "water": (0.1112620426, 4.6142, 2.15, 5.093, 2.808)}
iso3i_MS = darkcomet_AU / 86400.0  # m/s per au/d
iso3i_CLIP = 8.0  # two-coordinate chi-square outlier threshold
iso3i_SPECS = (("gravity", dict(g="r2", ng=False)),
               ("outgassing r^-2", dict(g="r2", ng=True)),
               ("outgassing water", dict(g="water", ng=True)),
               ("outgassing r^-2 changing at perihelion", dict(g="r2", ng=True, split=True)),
               ("outgassing water changing at perihelion", dict(g="water", ng=True, split=True)),
               ("outgassing r^-2, impulse at Jupiter", dict(g="r2", ng=True, kicks=["jupiter"])),
               ("outgassing water, impulse at Jupiter", dict(g="water", ng=True, kicks=["jupiter"])),
               ("outgassing r^-2, impulse at Mars", dict(g="r2", ng=True, kicks=["mars"])),
               ("outgassing r^-2, impulse at perihelion", dict(g="r2", ng=True, kicks=["perihelion"])),
               ("outgassing r^-2 changing at perihelion, impulse at perihelion", dict(g="r2", ng=True, split=True, kicks=["perihelion"])),
               ("outgassing r^-2 changing at perihelion, impulse at Jupiter", dict(g="r2", ng=True, split=True, kicks=["jupiter"])))
iso3i_BOOT = ("outgassing r^-2, impulse at Jupiter", "outgassing water, impulse at Jupiter", "outgassing r^-2, impulse at Mars",
              "outgassing r^-2 changing at perihelion, impulse at perihelion")
iso3i_BOOT_BINS = ("outgassing r^-2 in seven bins, impulse at perihelion", "outgassing r^-2 in seven bins, impulse at Jupiter")
# seven bins of constant outgassing: boundaries 2025 August 1 and September 15, perihelion, 2025 November 30, 2026 January 15
# and February 28 (TDB JD)
iso3i_BINS = (2460888.5, 2460933.5, "perihelion", 2461009.5, 2461055.5, 2461099.5)
iso3i_COMPARE = (("gravity", dict(g="r2", ng=False)),
                 ("outgassing r^-2", dict(g="r2", ng=True)),
                 ("outgassing water", dict(g="water", ng=True)),
                 ("outgassing r^-2 changing at perihelion", dict(g="r2", ng=True, split=True)),
                 ("outgassing r^-2 in seven bins", dict(g="r2", ng=True, bins=iso3i_BINS)),
                 ("outgassing r^-2, impulse at Mars", dict(g="r2", ng=True, kicks=["mars"])),
                 ("outgassing r^-2, impulse at perihelion", dict(g="r2", ng=True, kicks=["perihelion"])),
                 ("outgassing r^-2 in seven bins, impulse at perihelion", dict(g="r2", ng=True, bins=iso3i_BINS, kicks=["perihelion"])),
                 ("outgassing r^-2, impulse at Jupiter", dict(g="r2", ng=True, kicks=["jupiter"])),
                 ("outgassing r^-2 in seven bins, impulse at Jupiter", dict(g="r2", ng=True, bins=iso3i_BINS, kicks=["jupiter"])))
iso3i_POOL = {}


def iso3i_fetch():
    os.makedirs(iso3i_ROOT, exist_ok=True)
    text = darkcomet_obs80(iso3i_DESIG)
    open(os.path.join(iso3i_ROOT, "obs80.txt"), "w").write(text)
    sb = json.loads(core.get("https://ssd-api.jpl.nasa.gov/sbdb.api?" + urllib.parse.urlencode(
        {"sstr": iso3i_DESIG, "phys-par": 1, "cov": "mat", "full-prec": 1}), timeout=120))
    json.dump(sb, open(os.path.join(iso3i_ROOT, "sbdb.json"), "w"), indent=1)
    darkcomet_sites(iso3i_ROOT)
    rows = darkcomet_parse(text)
    o = sb["orbit"]
    print("%s: %d positions from %d sites; JPL solution %s on %s positions from %s to %s, rms %s arcsec; %s"
          % (sb["object"]["fullname"], len(rows), len({r["site"] for r in rows}), o["orbit_id"], o["n_obs_used"], o["first_obs"],
             o["last_obs"], o["rms"], [(p["name"], p["value"], p["sigma"]) for p in o.get("model_pars", [])]))


def iso3i_propagate(eph, t0, state, ng, times, g="r2", events=()):
    """Barycentric positions at TDB JDs from a state at t0 in ASSIST with outgassing accelerations ng (A1, A2, A3 in au/d^2 where g = 1) under g. Each event (t, kind, value) acts as the integration crosses t: 'kick' adds value (au/d) to the velocity forward in time, 'ng' makes value the outgassing accelerations on the far side of t from t0."""
    import assist
    import rebound
    out = np.zeros((len(times), 3))
    times = np.asarray(times)
    alpha, nk, nm, nn, r0 = iso3i_G[g]
    for forward in (False, True):
        sel = np.nonzero(times >= t0 if forward else times < t0)[0]
        if not len(sel):
            continue
        sel = sel[np.argsort(times[sel])] if forward else sel[np.argsort(times[sel])[::-1]]
        evs = sorted([e for e in events if (e[0] >= t0) == forward], key=lambda e: e[0] if forward else -e[0])
        sim = rebound.Simulation()
        sim.add(x=state[0], y=state[1], z=state[2], vx=state[3], vy=state[4], vz=state[5])
        sim.t = t0 - eph.jd_ref
        sim.dt = 1.0 if forward else -1.0
        ex = assist.Extras(sim, eph)
        ex.alpha, ex.nk, ex.nm, ex.nn, ex.r0 = alpha, nk, nm, nn, r0
        params = [np.array(ng, float)]
        ex.particle_params = params[-1]
        k = 0
        for i in sel:
            while k < len(evs) and (evs[k][0] <= times[i] if forward else evs[k][0] > times[i]):
                te, kind, val = evs[k]
                sim.integrate(te - eph.jd_ref)
                if kind == "kick":
                    s = 1.0 if forward else -1.0
                    p = sim.particles[0]
                    p.vx, p.vy, p.vz = p.vx + s * val[0], p.vy + s * val[1], p.vz + s * val[2]
                else:
                    params.append(np.array(val, float))
                    ex.particle_params = params[-1]
                k += 1
            sim.integrate(times[i] - eph.jd_ref)
            p = sim.particles[0]
            out[i] = (p.x, p.y, p.z)
        del ex
    return out


def iso3i_bounds(spec, ev):
    """TDB JDs where the outgassing accelerations change: perihelion for split, else the bins' boundaries (JDs or event names)."""
    if spec.get("split"):
        return [ev["perihelion"]]
    return [ev[b] if isinstance(b, str) else float(b) for b in spec.get("bins", ())]


def iso3i_npar(spec):
    nb = 1 if spec.get("split") else len(spec.get("bins", ()))
    return 6 + 3 * int(spec["ng"]) * (1 + nb) + 3 * len(spec.get("kicks", ()))


def iso3i_unpack(spec, p, ev):
    """State, outgassing at t0 and the events of a parameter vector under spec: outgassing beyond each change on the far side of t0, then the impulses."""
    k, ng, events = 6, np.zeros(3), []
    if spec["ng"]:
        ng = np.array(p[6:9])
        k = 9
        for t in iso3i_bounds(spec, ev):
            events.append((t, "ng", np.array(p[k:k + 3])))
            k += 3
    for name in spec.get("kicks", ()):
        events.append((ev[name], "kick", np.array(p[k:k + 3])))
        k += 3
    return np.array(p[:6]), ng, events


def iso3i_model(P, spec, p, sel):
    """Residuals (arcsec) of the positions sel, RA times cos Dec and then Dec."""
    s, ng, events = iso3i_unpack(spec, p, P["ev"])
    pos = iso3i_propagate(P["eph"], P["t0"], s, ng, P["t_emit"][sel], spec["g"], events) - P["obs_pos"][sel]
    mra, mdec = darkcomet_radec(pos)
    ra, dec = P["ra"][sel], P["dec"][sel]
    return np.concatenate([((ra - mra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec)) * 3600.0, (dec - mdec) * 3600.0])


def iso3i_fit(P, spec, sel, sigma, start, clip=iso3i_CLIP, iters=15, keep0=None):
    """Gauss-Newton by forward differences: parameters, covariance, residuals, kept mask. Outliers are re-judged each iteration, or with keep0 held to that mask throughout."""
    n = iso3i_npar(spec)
    p = np.zeros(n)
    p[:min(n, len(start))] = start[:min(n, len(start))]
    nk = 3 * len(spec.get("kicks", ()))
    h = np.array([1e-9] * 3 + [1e-11] * 3 + [1e-14] * (n - 6 - nk) + [1e-9] * nk)
    keep = np.ones(len(sel), bool) if keep0 is None else keep0.copy()
    w = np.concatenate([1.0 / sigma, 1.0 / sigma])
    lam = 1e-4  # Levenberg-Marquardt damping: a step is taken only if it lowers the chi-square of the kept positions
    for it in range(iters):
        r0 = iso3i_model(P, spec, p, sel)
        J = np.zeros((len(r0), n))
        for k in range(n):
            q = p.copy()
            q[k] += h[k]
            J[:, k] = (iso3i_model(P, spec, q, sel) - r0) / h[k]
        kk = np.concatenate([keep, keep])
        A, b = J[kk] * w[kk, None], r0[kk] * w[kk]
        sc = np.linalg.norm(A, axis=0)
        sc[sc == 0] = 1.0
        N, g = (A / sc).T @ (A / sc), (A / sc).T @ b
        chi0 = float(b @ b)
        while True:
            dp = -np.linalg.solve(N + lam * np.diag(np.diag(N)), g) / sc
            r1 = iso3i_model(P, spec, p + dp, sel)
            chi1 = float(np.sum(((np.concatenate([r1[:len(sel)], r1[len(sel):]]) * w)[kk]) ** 2))
            if chi1 <= chi0 or lam > 1e6:
                break
            lam *= 10.0
        if chi1 <= chi0:
            p, lam = p + dp, max(lam / 10.0, 1e-7)
        cov = np.linalg.inv(N) / np.outer(sc, sc)
        new = (r1[:len(sel)] ** 2 + r1[len(sel):] ** 2) / sigma ** 2 < clip if keep0 is None else keep
        if chi1 > chi0 or (np.all(np.abs(dp) < 1e-3 * np.sqrt(np.diag(cov))) and np.array_equal(new, keep)):
            break
        keep = new
    res = iso3i_model(P, spec, p, sel)
    return p, cov, res[:len(sel)], res[len(sel):], keep, it + 1


def iso3i_events(eph, t0, state, ng):
    """TDB JDs and distances (au) of perihelion and of the closest approaches to Mars and Jupiter on an orbit."""
    out = {}
    for name, body, lo, hi in (("perihelion", "sun", 2460960.5, 2460995.5), ("mars", "mars", 2460935.5, 2460960.5),
                               ("jupiter", "jupiter", 2461100.5, 2461130.5)):
        for step in (0.25, 0.005):
            ts = np.arange(lo, hi, step)
            x = iso3i_propagate(eph, t0, state, ng, ts)
            b = np.array([[q.x, q.y, q.z] for q in (eph.get_particle(body, t - eph.jd_ref) for t in ts)])
            dist = np.linalg.norm(x - b, axis=1)
            k = int(np.argmin(dist))
            lo, hi = ts[max(k - 2, 0)], ts[min(k + 2, len(ts) - 1)]
        out[name], out[name + "_au"] = float(ts[k]), float(dist[k])
    return out


def iso3i_setup(ephem_dir=None):
    """Positions, observers, emission times, the JPL reference state at a mid-arc epoch, the events, and weights: per-site rms from a first fit with r^-2 outgassing (floor 0.15 arcsec), inflated by sqrt(N/4) above four per station-night."""
    d = iso3i_ROOT
    eph = darkcomet_ephem(ephem_dir)
    rows = darkcomet_parse(open(os.path.join(d, "obs80.txt")).read())
    sites = darkcomet_sites(d)
    rows = [r for r in rows if r["site"] in sites or "sat_km" in r]
    tdb, obs_pos = darkcomet_observers(rows, sites, eph)
    ra, dec = np.array([r["ra"] for r in rows]), np.array([r["dec"] for r in rows])
    sb = json.load(open(os.path.join(d, "sbdb.json")))
    mp = {p["name"]: num(p["value"]) for p in sb["orbit"].get("model_pars", [])}
    ng_jpl = np.array([mp.get("A1") or 0.0, mp.get("A2") or 0.0, mp.get("A3") or 0.0])
    t0 = float(np.round(np.median(tdb)) + 0.5)
    path = os.path.join(d, "horizons_%.1f.json" % t0)
    if not os.path.exists(path):
        q = {"format": "json", "COMMAND": "'DES=%s;'" % iso3i_DESIG, "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@0'",
             "REF_PLANE": "'FRAME'", "VEC_TABLE": "'2'", "OUT_UNITS": "'AU-D'", "TLIST": "'%.1f'" % t0, "TIME_TYPE": "'TDB'",
             "CSV_FORMAT": "'YES'", "VEC_CORR": "'NONE'", "OBJ_DATA": "'NO'"}
        json.dump(json.loads(core.get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), timeout=120)),
                  open(path, "w"))
    txt = json.load(open(path))["result"]
    p0 = np.array([num(x) for x in txt.split("$$SOE")[1].split("$$EOE")[0].strip().split(",")[2:8]] + list(ng_jpl))
    geo = iso3i_propagate(eph, t0, p0[:6], ng_jpl, tdb) - obs_pos
    t_emit = tdb - np.linalg.norm(geo, axis=1) * darkcomet_AU / 2.99792458e8 / 86400.0
    ev = iso3i_events(eph, t0, p0[:6], ng_jpl)
    P = dict(eph=eph, t0=t0, t_emit=t_emit, obs_pos=obs_pos, ra=ra, dec=dec, ev=ev)
    idx = np.arange(len(rows))
    nf = darkcomet_night_factor(rows, "veres")
    sigma = np.array([darkcomet_SIGMA.get(r["site"], 0.5) for r in rows]) * nf
    p1, _, dra, ddec, keep, _ = iso3i_fit(P, dict(g="r2", ng=True), idx, sigma, p0)
    site = np.array([r["site"] for r in rows])
    base = np.array([darkcomet_SIGMA.get(r["site"], 0.5) for r in rows])
    site_rms = {}
    for s in set(site):
        m = (site == s) & keep
        if m.sum() >= 5:
            site_rms[s] = float(np.sqrt(np.mean((dra[m] ** 2 + ddec[m] ** 2) / 2 / nf[m] ** 2)))
            base[site == s] = max(site_rms[s], 0.15)
    return dict(P=P, rows=rows, tdb=tdb, sb=sb, ng_jpl=ng_jpl, p0=p0, p1=p1, sigma=base * nf, nf=nf, site_rms=site_rms)


def iso3i_pool_init(state, ephem_dir):
    iso3i_POOL.update(state)
    iso3i_POOL["eph"] = darkcomet_ephem(ephem_dir)


def iso3i_pool_fit(job):
    tag, spec, sel, sg, start = job[:5]
    try:
        p, cov, dra, ddec, kp, its = iso3i_fit(iso3i_POOL, spec, sel, sg, start, keep0=job[5] if len(job) > 5 else None)
    except np.linalg.LinAlgError as e:
        return dict(tag=tag, error=str(e))
    return dict(tag=tag, n=int(len(sel)), kept=int(kp.sum()), chi2=float(np.sum(((dra ** 2 + ddec ** 2) / sg ** 2)[kp])),
                dof=int(2 * kp.sum() - len(p)), rms=float(np.sqrt(np.mean((dra ** 2 + ddec ** 2)[kp] / 2))), iters=its,
                p=[float(x) for x in p], cov=cov.tolist())


def iso3i_rtn(P, spec, p, t):
    """Rows: unit radial, transverse and normal vectors of the fitted orbit about the Sun at TDB JD t."""
    s, ng, events = iso3i_unpack(spec, p, P["ev"])
    x = iso3i_propagate(P["eph"], P["t0"], s, ng, [t - 0.01, t + 0.01], spec["g"], [e for e in events if e[1] == "ng"])
    sun = P["eph"].get_particle("sun", t - P["eph"].jd_ref)
    r = (x[0] + x[1]) / 2 - np.array([sun.x, sun.y, sun.z])
    v = x[1] - x[0]
    R = r / np.linalg.norm(r)
    N = np.cross(r, v)
    N /= np.linalg.norm(N)
    return np.array([R, np.cross(N, R), N])


def iso3i_describe(P, name, spec, r):
    """A fit's outgassing accelerations and impulses with their errors, the impulses in m/s about the Sun's radial, transverse and normal directions, and each impulse's chi-square against none (3 degrees of freedom)."""
    from scipy import stats
    p, cov = np.array(r["p"]), np.array(r["cov"])
    out = dict(name=name, spec=spec, n=r["n"], kept=r["kept"], chi2=r["chi2"], dof=r["dof"], rms=r["rms"], iters=r["iters"])
    if spec["ng"]:
        out["A"] = [[float(p[6 + j]), float(math.sqrt(cov[6 + j, 6 + j]))] for j in range(3)]
        for i, t in enumerate(iso3i_bounds(spec, P["ev"])):
            out.setdefault("A_beyond", []).append(dict(t=t, A=[[float(p[9 + 3 * i + j]), float(math.sqrt(cov[9 + 3 * i + j, 9 + 3 * i + j]))]
                                                             for j in range(3)]))
    k = iso3i_npar(dict(spec, kicks=[])) if spec.get("kicks") else None
    for e in spec.get("kicks", ()):
        dv, C = p[k:k + 3] * iso3i_MS, cov[k:k + 3, k:k + 3] * iso3i_MS ** 2
        M = iso3i_rtn(P, spec, p, P["ev"][e])
        dv_rtn, C_rtn = M @ dv, M @ C @ M.T
        chi = float(dv @ np.linalg.solve(C, dv))
        out.setdefault("impulses", {})[e] = dict(
            t=P["ev"][e], dv_ms=float(np.linalg.norm(dv)), rtn_ms=[float(x) for x in dv_rtn],
            rtn_sigma_ms=[float(math.sqrt(C_rtn[j, j])) for j in range(3)], chi2=chi, p=float(stats.chi2.sf(chi, 3)),
            limit_ms=float(np.linalg.norm(dv) + 3 * math.sqrt(np.max(np.linalg.eigvalsh(C)))))
        k += 3
    return out


def iso3i_run(workers=8, ephem_dir=None):
    from multiprocessing import Pool
    S = iso3i_setup(ephem_dir)
    P, rows, sigma, p1 = S["P"], S["rows"], S["sigma"], S["p1"]
    ev = P["ev"]
    print("%d positions from %d sites, %s to %s TDB JD; perihelion %.3f (%.3f au), Mars %.3f (%.3f au), Jupiter %.3f (%.3f au); "
          "%d positions after Jupiter" % (len(rows), len({r["site"] for r in rows}), S["tdb"].min(), S["tdb"].max(), ev["perihelion"],
                                         ev["perihelion_au"], ev["mars"], ev["mars_au"], ev["jupiter"], ev["jupiter_au"],
                                         int((S["tdb"] > ev["jupiter"]).sum())), flush=True)
    idx = np.arange(len(rows))
    jobs = [(name, spec, idx, sigma, p1) for name, spec in iso3i_SPECS]
    state = {k: P[k] for k in ("t0", "t_emit", "obs_pos", "ra", "dec", "ev")}
    with Pool(workers, initializer=iso3i_pool_init, initargs=(state, ephem_dir)) as pool:
        res = {r["tag"]: r for r in pool.imap_unordered(iso3i_pool_fit, jobs, chunksize=1)}
    out = dict(desig=iso3i_DESIG, t0=P["t0"], events=ev, n=len(rows), ng_jpl=S["ng_jpl"].tolist(), site_rms=S["site_rms"], fits=[])
    for name, spec in iso3i_SPECS:
        f = iso3i_describe(P, name, spec, res[name])
        out["fits"].append(dict(f, p=res[name]["p"], cov=res[name]["cov"]))
        print("%-62s chi2 %9.1f / %5d, rms %.3f, %d kept%s%s" % (
            name, f["chi2"], f["dof"], f["rms"], f["kept"],
            "; A1 A2 A3 %s" % " ".join("%+.3e(%.1e)" % tuple(a) for a in f["A"]) if spec["ng"] else "",
            "".join("; %s: %.3f m/s, RTN %s +- %s, chi2 %.1f, p %.2g" % (e, x["dv_ms"], " ".join("%+.3f" % v for v in x["rtn_ms"]),
                                                                         " ".join("%.3f" % v for v in x["rtn_sigma_ms"]), x["chi2"], x["p"])
                    for e, x in f.get("impulses", {}).items())), flush=True)
    json.dump(out, open(os.path.join(iso3i_ROOT, "fits.json"), "w"), indent=1)


def iso3i_boot(boot=100, workers=8, ephem_dir=None, bins=False):
    """Observing nights resampled with replacement for the impulse fits of iso3i_BOOT, or with bins those of iso3i_BOOT_BINS on binned outgassing: each impulse's spread, into boot.json or boot_bins.json."""
    from multiprocessing import Pool
    S = iso3i_setup(ephem_dir)
    P, rows, sigma = S["P"], S["rows"], S["sigma"]
    names = iso3i_BOOT_BINS if bins else iso3i_BOOT
    fits = {f["name"]: f for f in json.load(open(os.path.join(iso3i_ROOT, "compare.json" if bins else "fits.json")))["fits"]}
    night = np.floor(np.array([r["jd_utc"] for r in rows]) + 0.5).astype(int)
    nights = sorted(set(night.tolist()))
    idx = np.arange(len(rows))
    rng = np.random.default_rng(20261018)
    specs = dict(iso3i_SPECS + iso3i_COMPARE)
    jobs = []
    for b in range(boot):
        draw = rng.choice(nights, size=len(nights), replace=True)
        sel = np.concatenate([idx[night == k] for k in draw])
        jobs += [("%s|%d" % (name, b), specs[name], sel, sigma[sel], np.array(fits[name]["p"])) for name in names]
    state = {k: P[k] for k in ("t0", "t_emit", "obs_pos", "ra", "dec", "ev")}
    print("%d positions, %d nights; %d fits on %d workers" % (len(rows), len(nights), len(jobs), workers), flush=True)
    with Pool(workers, initializer=iso3i_pool_init, initargs=(state, ephem_dir)) as pool:
        res = list(pool.imap_unordered(iso3i_pool_fit, jobs, chunksize=1))
    out = dict(boot=boot, nights=len(nights), failed=sum(1 for r in res if "error" in r), fits={})
    for name in names:
        spec = specs[name]
        k = iso3i_npar(dict(spec, kicks=[]))
        dv = np.array([np.array(r["p"][k:k + 3]) * iso3i_MS for r in res if r["tag"].split("|")[0] == name and "error" not in r])
        M = iso3i_rtn(P, spec, np.array(fits[name]["p"]), P["ev"][spec["kicks"][0]])
        rtn = dv @ M.T
        out["fits"][name] = dict(n=len(dv), rtn_median=np.median(rtn, axis=0).tolist(), rtn_sd=rtn.std(axis=0).tolist(),
                                 norm_pct=np.percentile(np.linalg.norm(dv, axis=1), [50, 84, 97.5, 99.85]).tolist())
        x = out["fits"][name]
        print("%-62s %d resamples: RTN median %s, sd %s m/s; |dv| 50/84/97.5/99.85%% %s m/s" % (
            name, x["n"], " ".join("%+.3f" % v for v in x["rtn_median"]), " ".join("%.3f" % v for v in x["rtn_sd"]),
            " ".join("%.3f" % v for v in x["norm_pct"])), flush=True)
    json.dump(out, open(os.path.join(iso3i_ROOT, "boot_bins.json" if bins else "boot.json"), "w"), indent=1)


def iso3i_sites(workers=8, n_sites=30, ephem_dir=None):
    """The impulse fits of iso3i_BOOT_BINS with each of the n_sites observatories holding most kept positions left out in turn, on the positions the r^-2 outgassing fit keeps; sites.json."""
    from collections import Counter
    from multiprocessing import Pool
    S = iso3i_setup(ephem_dir)
    P, rows, sigma = S["P"], S["rows"], S["sigma"]
    idx = np.arange(len(rows))
    _, _, _, _, keep, _ = iso3i_fit(P, dict(g="r2", ng=True), idx, sigma, S["p1"])
    fits = {f["name"]: f for f in json.load(open(os.path.join(iso3i_ROOT, "compare.json")))["fits"]}
    specs = dict(iso3i_COMPARE)
    site = np.array([r["site"] for r in rows])
    top = [s for s, _ in Counter(site[keep].tolist()).most_common(n_sites)]
    jobs = []
    for s in top:
        m = site != s
        jobs += [("%s|%s" % (name, s), specs[name], idx[m], sigma[m], np.array(fits[name]["p"]), keep[m]) for name in iso3i_BOOT_BINS]
    state = {k: P[k] for k in ("t0", "t_emit", "obs_pos", "ra", "dec", "ev")}
    print("%d positions, %d kept; %d sites left out in turn; %d fits on %d workers" % (len(rows), int(keep.sum()), len(top), len(jobs),
                                                                                       workers), flush=True)
    with Pool(workers, initializer=iso3i_pool_init, initargs=(state, ephem_dir)) as pool:
        res = {r["tag"]: r for r in pool.imap_unordered(iso3i_pool_fit, jobs, chunksize=1)}
    out = dict(sites=top, kept_by_site={s: int((keep & (site == s)).sum()) for s in top}, fits={})
    for name in iso3i_BOOT_BINS:
        for s in top:
            r = res["%s|%s" % (name, s)]
            if "error" in r:
                print("%-56s without %s: %s" % (name, s, r["error"]), flush=True)
                continue
            f = iso3i_describe(P, name, specs[name], r)
            x = list(f["impulses"].values())[0]
            out["fits"].setdefault(name, {})[s] = dict(rtn=x["rtn_ms"], rtn_sigma=x["rtn_sigma_ms"], chi2=x["chi2"], kept=f["kept"])
            print("%-56s without %s (%4d kept there): RTN %s +- %s m/s, chi2 %.1f" % (
                name, s, out["kept_by_site"][s], " ".join("%+.2f" % v for v in x["rtn_ms"]), " ".join("%.2f" % v for v in x["rtn_sigma_ms"]),
                x["chi2"]), flush=True)
    json.dump(out, open(os.path.join(iso3i_ROOT, "sites.json"), "w"), indent=1)


def iso3i_compare(workers=8, ephem_dir=None):
    """Every model of iso3i_COMPARE on the positions the r^-2 outgassing fit keeps, so that their chi-squares compare, and the impulse at Jupiter without each of the two observatories holding most positions after it, and without both; compare.json."""
    from collections import Counter
    from multiprocessing import Pool
    S = iso3i_setup(ephem_dir)
    P, rows, sigma = S["P"], S["rows"], S["sigma"]
    idx = np.arange(len(rows))
    p, _, _, _, keep, _ = iso3i_fit(P, dict(g="r2", ng=True), idx, sigma, S["p1"])
    site = np.array([r["site"] for r in rows])
    top = [s for s, _ in Counter(site[S["tdb"] > P["ev"]["jupiter"]].tolist()).most_common(2)]
    jobs = [(name, spec, idx, sigma, p, keep) for name, spec in iso3i_COMPARE]
    jup = dict(g="r2", ng=True, kicks=["jupiter"])
    for drop in [[s] for s in top] + [top]:
        m = ~np.isin(site, drop)
        jobs.append(("outgassing r^-2, impulse at Jupiter, without %s" % " and ".join(drop), jup, idx[m], sigma[m], p, keep[m]))
    state = {k: P[k] for k in ("t0", "t_emit", "obs_pos", "ra", "dec", "ev")}
    print("%d positions, %d kept by the r^-2 outgassing fit; %d fits on %d workers" % (len(rows), int(keep.sum()), len(jobs), workers),
          flush=True)
    with Pool(workers, initializer=iso3i_pool_init, initargs=(state, ephem_dir)) as pool:
        res = {r["tag"]: r for r in pool.imap_unordered(iso3i_pool_fit, jobs, chunksize=1)}
    ref = res["outgassing r^-2"]["chi2"]
    out = dict(kept=int(keep.sum()), n=len(rows), dropped=top, fits=[])
    for tag, spec, *_ in jobs:
        f = iso3i_describe(P, tag, spec, res[tag])
        f["dchi2"] = None if "without" in tag else f["chi2"] - ref
        out["fits"].append(dict(f, p=res[tag]["p"], cov=res[tag]["cov"]))
        print("%-58s %3d parameters, chi2 %9.1f%s%s" % (
            tag, len(res[tag]["p"]), f["chi2"], "" if f["dchi2"] is None else " (%+.1f)" % f["dchi2"],
            "".join("; %s: %.2f m/s, RTN %s +- %s, chi2 %.1f" % (e, x["dv_ms"], " ".join("%+.2f" % v for v in x["rtn_ms"]),
                                                                " ".join("%.2f" % v for v in x["rtn_sigma_ms"]), x["chi2"])
                    for e, x in f.get("impulses", {}).items())), flush=True)
        for b in f.get("A_beyond", []):
            print("    beyond %.1f: A1 A2 A3 %s" % (b["t"], " ".join("%+.2e(%.1e)" % tuple(a) for a in b["A"])))
    json.dump(out, open(os.path.join(iso3i_ROOT, "compare.json"), "w"), indent=1)


def iso3i_scan(workers=8, step=5.0, ephem_dir=None):
    """An impulse every step days across the observed arc on the r^-2 outgassing of seven bins, on the positions that model keeps: each impulse's radial, transverse and normal parts with their errors and the chi-square it gains; scan.json."""
    from multiprocessing import Pool
    S = iso3i_setup(ephem_dir)
    P, rows, sigma = S["P"], S["rows"], S["sigma"]
    idx = np.arange(len(rows))
    base = dict(g="r2", ng=True, bins=iso3i_BINS)
    p, _, _, _, keep, _ = iso3i_fit(P, base, idx, sigma, S["p1"])
    times = np.arange(np.ceil(S["tdb"].min()) + 10.5, S["tdb"].max() - 5.0, step)
    kept = S["tdb"][keep]
    far = np.array([((kept > t) if t >= P["t0"] else (kept < t)).sum() for t in times])
    print("impulse times with fewer than 20 kept positions beyond them, left out: %s" % [float(t) for t in times[far < 20]], flush=True)
    times = times[far >= 20]
    for t in times:
        P["ev"]["t%.1f" % t] = float(t)
    jobs = [("base", base, idx, sigma, p, keep)] + [("t%.1f" % t, dict(base, kicks=["t%.1f" % t]), idx, sigma, p, keep) for t in times]
    state = {k: P[k] for k in ("t0", "t_emit", "obs_pos", "ra", "dec", "ev")}
    print("%d positions, %d kept; impulses at %d times from %.1f to %.1f; %d fits on %d workers"
          % (len(rows), int(keep.sum()), len(times), times[0], times[-1], len(jobs), workers), flush=True)
    with Pool(workers, initializer=iso3i_pool_init, initargs=(state, ephem_dir)) as pool:
        res = {r["tag"]: r for r in pool.imap_unordered(iso3i_pool_fit, jobs, chunksize=1)}
    ref = res["base"]["chi2"]
    out = dict(kept=int(keep.sum()), base_chi2=ref, scan=[])
    for t in times:
        tag = "t%.1f" % t
        if "error" in res[tag]:
            print("%.1f: %s" % (t, res[tag]["error"]), flush=True)
            continue
        f = iso3i_describe(P, tag, dict(base, kicks=[tag]), res[tag])
        x = f["impulses"][tag]
        out["scan"].append(dict(t=float(t), dchi2=f["chi2"] - ref, rtn=x["rtn_ms"], rtn_sigma=x["rtn_sigma_ms"], chi2=x["chi2"],
                                limit_ms=x["limit_ms"], n_before=int((S["tdb"] < t).sum()), n_after=int((S["tdb"] > t).sum())))
        y = out["scan"][-1]
        print("%.1f: dchi2 %+8.1f; RTN %s +- %s m/s, chi2 %6.1f, 3-sigma |dv| < %.2f m/s; %d positions after"
              % (t, y["dchi2"], " ".join("%+.2f" % v for v in y["rtn"]), " ".join("%.2f" % v for v in y["rtn_sigma"]), y["chi2"], y["limit_ms"],
                 y["n_after"]), flush=True)
    json.dump(out, open(os.path.join(iso3i_ROOT, "scan.json"), "w"), indent=1)


def iso3i_record():
    """The model comparison, the impulse scan, the night resampling and the site jackknife, read from their files into the coverage store: the impulses at perihelion and at the Jupiter passage with their spreads, and the scan's limit on a velocity change at any time of the arc."""
    load = lambda name: json.load(open(os.path.join(iso3i_ROOT, name)))
    fits, scan, boot, sites, first = load("compare.json"), load("scan.json"), load("boot_bins.json"), load("sites.json"), load("fits.json")
    f = {x["name"]: x for x in fits["fits"]}
    chi = lambda name: f[name]["chi2"]
    per = f["outgassing r^-2 in seven bins, impulse at perihelion"]["impulses"]["perihelion"]
    per0 = f["outgassing r^-2, impulse at perihelion"]["impulses"]["perihelion"]
    mars = f["outgassing r^-2, impulse at Mars"]["impulses"]["mars"]
    jup = f["outgassing r^-2 in seven bins, impulse at Jupiter"]["impulses"]["jupiter"]
    jup0 = f["outgassing r^-2, impulse at Jupiter"]["impulses"]["jupiter"]
    without = {x["name"].split("without ")[1]: x["impulses"]["jupiter"] for x in fits["fits"] if "without" in x["name"]}
    bper = boot["fits"]["outgassing r^-2 in seven bins, impulse at perihelion"]
    bjup = boot["fits"]["outgassing r^-2 in seven bins, impulse at Jupiter"]
    sper = sites["fits"]["outgassing r^-2 in seven bins, impulse at perihelion"]
    sjup = sites["fits"]["outgassing r^-2 in seven bins, impulse at Jupiter"]
    span = lambda d, k: (min(v["rtn"][k] for v in d.values()), max(v["rtn"][k] for v in d.values()),
                         min(v["rtn_sigma"][k] for v in d.values()), max(v["rtn_sigma"][k] for v in d.values()))
    ev_j = first["events"]["jupiter"]
    sc = scan["scan"]
    early = [x["limit_ms"] for x in sc if x["t"] < ev_j - 30]
    late = [x["limit_ms"] for x in sc if x["t"] >= ev_j - 30]
    best = sorted(sc, key=lambda x: x["dchi2"])[:4]
    n_after = int((np.array([x["t"] for x in sc]) >= ev_j).sum()) and sc[-1]["n_after"]
    result = ("on %d positions gravity alone leaves chi-square %.0f and r^-2 outgassing %.0f; outgassing changing at perihelion %.0f less and "
              "seven bins %.0f less; an impulse on constant outgassing: %.1f m/s at Mars (%.0f less), %.1f m/s at perihelion (%.0f less), "
              "%.0f +- %.0f m/s radial at Jupiter; on the binned outgassing: perihelion %+.1f +- %.1f m/s radial (%.0f less, 3 sigma below "
              "%.1f m/s), %+.1f +- %.1f over %d resampled night sets, %+.1f to %+.1f +- %.1f-%.1f with each of %d sites left out; Jupiter "
              "%+.0f +- %.0f m/s radial, %+.0f +- %.0f over resampled nights, %s; an impulse every %d days at %d times on the binned "
              "outgassing worsens no fit, 3 sigma |dv| below %.1f-%.1f m/s up to 30 days before the Jupiter passage and %.0f-%.0f m/s "
              "after, the largest gains %s" % (
                  fits["kept"], chi("gravity"), chi("outgassing r^-2"), chi("outgassing r^-2") - chi("outgassing r^-2 changing at perihelion"),
                  chi("outgassing r^-2") - chi("outgassing r^-2 in seven bins"), mars["dv_ms"],
                  chi("outgassing r^-2") - chi("outgassing r^-2, impulse at Mars"), per0["dv_ms"],
                  chi("outgassing r^-2") - chi("outgassing r^-2, impulse at perihelion"), jup0["rtn_ms"][0], jup0["rtn_sigma_ms"][0],
                  per["rtn_ms"][0], per["rtn_sigma_ms"][0],
                  chi("outgassing r^-2 in seven bins") - chi("outgassing r^-2 in seven bins, impulse at perihelion"), per["limit_ms"],
                  bper["rtn_median"][0], bper["rtn_sd"][0], bper["n"], span(sper, 0)[0], span(sper, 0)[1], span(sper, 0)[2], span(sper, 0)[3],
                  len(sper), jup["rtn_ms"][0], jup["rtn_sigma_ms"][0], bjup["rtn_median"][0], bjup["rtn_sd"][0],
                  ", ".join("%+.0f +- %.0f without %s" % (v["rtn_ms"][0], v["rtn_sigma_ms"][0], k) for k, v in without.items()),
                  round(sc[1]["t"] - sc[0]["t"]), len(sc), min(early), max(early), min(late), max(late),
                  "; ".join("%.0f at JD %.1f (%+.1f +- %.1f radial, %+.1f +- %.1f transverse m/s)" % (
                      -x["dchi2"], x["t"], x["rtn"][0], x["rtn_sigma"][0], x["rtn"][1], x["rtn_sigma"][1]) for x in best)))
    print(result)
    core.record_run("iso3i_impulses", "Impulses on 3I/ATLAS's orbit",
                    "Minor Planet Center observations of %s; JPL Small-Body Database; ASSIST with DE440 and sb441-n16" % iso3i_DESIG,
                    "every MPC position of 3I/ATLAS: %d from %d sites, %d after the Jupiter passage" % (first["n"], len(first["site_rms"]), n_after),
                    os.path.abspath(__file__), "iso3i", 1, "interstellar objects",
                    dict(clip=iso3i_CLIP, bins=[b for b in iso3i_BINS], boot=boot["boot"], sites=len(sites["sites"]), scan_step_days=round(sc[1]["t"] - sc[0]["t"])),
                    "the fits' covariances on one set of kept positions, %d resamplings of the %d observing nights, and each of the %d busiest sites "
                    "left out in turn" % (boot["boot"], boot["nights"], len(sites["sites"])),
                    "a velocity change beyond the fit's 3 sigma limit at any time of the arc, %.1f-%.0f m/s" % (min(early), max(late)), result,
                    [dict(target="mpc:%s" % iso3i_DESIG, stat=per["rtn_ms"][0], unit="impulse at perihelion on binned outgassing, radial m/s", vet="activity",
                          note="%+.1f +- %.1f m/s radial at perihelion on seven bins of outgassing, which the water production SOHO/SWAN measured "
                               "after perihelion supplies; Jupiter passage %+.0f +- %.0f m/s radial on %d positions after it" % (
                                   per["rtn_ms"][0], per["rtn_sigma_ms"][0], jup["rtn_ms"][0], jup["rtn_sigma_ms"][0], n_after))])
    print("recorded")


def iso3i_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "fit", "boot", "compare", "scan", "sites", "record"))
    p.add_argument("--bins", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--boot", type=int, default=100)
    p.add_argument("--ephem", default=None)
    a = p.parse_args()
    if a.mode == "fetch":
        iso3i_fetch()
    elif a.mode == "record":
        iso3i_record()
    elif a.mode == "fit":
        iso3i_run(a.workers, a.ephem)
    elif a.mode == "compare":
        iso3i_compare(a.workers, a.ephem)
    elif a.mode == "scan":
        iso3i_scan(a.workers, ephem_dir=a.ephem)
    elif a.mode == "sites":
        iso3i_sites(a.workers, ephem_dir=a.ephem)
    else:
        iso3i_boot(a.boot, a.workers, a.ephem, a.bins)


# --------------------------------------------------------------------------------------------------------------
# circular: asteroid orbits made rather than captured

circular_DOC = r"""Asteroids on circles in the ecliptic: numbered asteroids of JPL's Small-Body Database with eccentricity below 0.001 and inclination below 0.2 deg, an orbit made rather than captured, against the product of the two tails over the numbered population.

    python -I seti.py solar_system circular [record] [--e-max 0.001] [--i-max-deg 0.2]"""


def circular_count(cdata, fields="full_name"):
    q = {"fields": fields, "sb-kind": "a", "sb-ns": "n"}
    if cdata:
        q["sb-cdata"] = json.dumps({"AND": cdata})
    d = json.loads(core.get("https://ssd-api.jpl.nasa.gov/sbdb_query.api?" + urllib.parse.urlencode(q), timeout=300))
    return int(d.get("count", 0)), d.get("data") or []


def circular_main(record=False, e_max=0.001, i_max_deg=0.2):
    n, _ = circular_count(None)
    ne, _ = circular_count(["e|LT|%g" % e_max])
    ni, _ = circular_count(["i|LT|%g" % i_max_deg])
    nb, rows = circular_count(["e|LT|%g" % e_max, "i|LT|%g" % i_max_deg], "full_name,a,e,i,H,condition_code")
    expect = ne * ni / n
    print("%d numbered asteroids; e below %g: %d, i below %g deg: %d, so %.2f expected with both if independent; found %d: %s"
          % (n, e_max, ne, i_max_deg, ni, expect, nb, rows[:10]))
    json.dump(dict(n=n, n_e=ne, n_i=ni, expect=expect, found=rows), open(core.results("circular_asteroids.json"), "w"), indent=1)
    if record:
        core.record_search(
            "sbdb_circular_asteroids", "Asteroids on circles in the ecliptic", archive="JPL Small-Body Database query API",
            selection="numbered asteroids: %d" % n, code=os.path.abspath(__file__), entry="circular", n_targets=n, n_unit="asteroids",
            params=dict(e=e_max, i_deg=i_max_deg), null_summary="the two tails taken as independent: %d and %d of %d give %.2f" % (ne, ni, n, expect),
            sensitivity="a numbered asteroid on an orbit circular to %g within %g deg of the ecliptic" % (e_max, i_max_deg),
            ledger="Asteroids on circles in the ecliptic", result="%d against %.2f expected" % (nb, expect))
        core.clear_targets("sbdb_circular_asteroids")
        core.record_targets("sbdb_circular_asteroids", [dict(
            target="mpc:%s" % str(r[0]).strip(), stat=num(r[2]), unit="eccentricity", note="i %s deg, a %s au, H %s" % (r[3], r[1], r[4]))
            for r in rows])
        print("recorded")


def sbdb_rows(cdata, fields, kind="a"):
    """Asteroids (or comets) of JPL's Small-Body Database query API meeting every constraint, numbered or not: rows as dicts."""
    q = {"fields": fields, "sb-kind": kind, "sb-cdata": json.dumps({"AND": cdata}), "full-prec": "true"}
    d = json.loads(core.get("https://ssd-api.jpl.nasa.gov/sbdb_query.api?" + urllib.parse.urlencode(q), timeout=300))
    return [dict(zip(d["fields"], r)) for r in d.get("data") or []]


mirror_DOC = r"""Asteroids brighter than a mirror: Small-Body Database asteroids whose geometric albedo exceeds 0.8, more light than any rock or ice of the asteroid belt returns and the mark of a polished or white made surface, each read against its albedo in the AKARI asteroid catalogue (Usui et al. 2011, VizieR J/PASJ/63/1117), its WISE albedo and error (Masiero et al. 2011, J/ApJ/741/68) and its taxonomic class from the Small-Body Database and the ATLAS phase-curve catalogue (Mahlke et al. 2021, VII/288), since enstatite E-types, the brightest natural surfaces, reach 0.6, basaltic V-types are bright as well, a dark class at a mirror's albedo marks a wrong albedo, and an albedo pinned at 1.000 is a fit at its bound.

    python -I seti.py solar_system mirror [record] [--albedo 0.8] [--natural 0.6] [--sigma 3]"""

# the brightest natural asteroid surfaces, enstatite E-types
MIRROR_NATURAL = 0.6


def mpc_pack(n):
    """An asteroid number in the MPC's packed form: five digits, or a letter for the ten-thousands from 100000."""
    n = int(n)
    if n < 100000:
        return "%05d" % n
    k, r = divmod(n, 10000)
    return (chr(ord("A") + k - 10) if k < 36 else chr(ord("a") + k - 36)) + "%04d" % r


def mirror_main(record=False, albedo=0.8, natural=MIRROR_NATURAL, sigma=3.0):
    rows = sbdb_rows(["albedo|GT|%g" % albedo], "full_name,pdes,H,albedo,diameter,diameter_sigma,class,spec_T,spec_B,n_obs_used")
    nums = [r["pdes"] for r in rows if str(r["pdes"]).isdigit()]
    akari, wise, atlas_class = {}, {}, {}
    if nums:
        for a in core.tap("vizier", 'SELECT Num, Alb, e_Alb, Diam FROM "J/PASJ/63/1117/acua_v1" WHERE Num IN (%s)' % ",".join(nums)):
            akari[str(a["Num"]).strip()] = (num(a["Alb"]), num(a["e_Alb"]), num(a["Diam"]))
        packed = {mpc_pack(n): n for n in nums}
        for a in core.tap("vizier", 'SELECT MPC, pV, e_pV, Diam FROM "J/ApJ/741/68/table1" WHERE MPC IN (%s)' % ",".join("'%s'" % p for p in packed)):
            p = str(a["MPC"]).strip()
            if p in packed and num(a["pV"]) is not None:
                wise.setdefault(packed[p], []).append((num(a["pV"]), num(a["e_pV"]) or 0.0))
        # taxonomy from the literature collected with the ATLAS phase curves (Mahlke et al. 2021)
        for a in core.tap("vizier", 'SELECT Number, Class FROM "VII/288/phase" WHERE Number IN (%s)' % ",".join(nums)):
            c = str(a["Class"] or "").strip()
            if c and c != "--":
                atlas_class[str(a["Number"]).strip()] = c
    out = []
    for r in rows:
        alb = float(r["albedo"])
        ak = akari.get(str(r["pdes"]))
        # the WISE fit's lowest albedo at SIGMA over every fit of the body
        wl = min((p - sigma * e for p, e in wise.get(str(r["pdes"]), [])), default=None)
        cls = " ".join(x for x in (r["spec_T"], r["spec_B"], atlas_class.get(str(r["pdes"]))) if x)
        vet = []
        # a dark class reported at a mirror's albedo: the albedo, not the surface, is wrong
        dark = any(c[:1] in "CBDPFGT" for c in cls.split())
        if (ak and ak[0] is not None and ak[0] + sigma * (ak[1] or 0) < albedo) or (wl is not None and wl < natural) or dark:
            vet.append("artefact")
        if alb >= 0.999:
            vet.append("fit-boundary")
        if any(c[:1] in "EVA" or c in ("Xe", "Xc") for c in cls.split()):
            vet.append("asteroid-class")
        out.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], H=num(r["H"]), albedo=alb, D=num(r["diameter"]), cls=cls,
                        akari=ak, wise=wise.get(str(r["pdes"])), wise_low=wl, vet=vet))
    left = [o for o in out if not o["vet"]]
    print("%d asteroids above albedo %g in the Small-Body Database; %d with an AKARI albedo, %d with WISE fits and errors; below the brightest "
          "natural surfaces at %g sigma in either: %d; pinned at 1.000: %d; E or Xe class: %d; unexplained: %d" % (
              len(out), albedo, sum(o["akari"] is not None for o in out), sum(o["wise"] is not None for o in out), sigma, sum("artefact" in o["vet"] for o in out),
              sum("fit-boundary" in o["vet"] for o in out), sum("asteroid-class" in o["vet"] for o in out), len(left)))
    for o in sorted(left, key=lambda o: -o["albedo"]):
        print("  %s: albedo %.3f, D %s km, H %s, class %s, WISE %s" % (o["name"], o["albedo"], o["D"], o["H"], o["cls"] or "-", o["wise"]))
    json.dump(out, open(core.results("mirror_asteroids.json"), "w"), indent=1)
    if record:
        core.record_search(
            "sbdb_mirror_asteroids", "Asteroids brighter than a mirror", archive="JPL Small-Body Database query API; AKARI asteroid catalogue "
            "(VizieR J/PASJ/63/1117)", selection="asteroids with a geometric albedo above %g: %d" % (albedo, len(out)), code=os.path.abspath(__file__),
            entry="mirror", n_targets=len(out), n_unit="asteroids", params=dict(albedo=albedo, natural=natural, sigma=sigma),
            null_summary="each body's AKARI albedo, its WISE albedo less %g times its error against the %.1f of the brightest natural "
                         "surfaces, its taxonomic class and the albedo fit's bound" % (sigma, natural),
            sensitivity="an asteroid whose albedo above %g no second survey, error, natural class or fit bound explains" % albedo, ledger="Asteroids brighter than a mirror",
            result="%d above %g: %d within the brightest natural surfaces in AKARI or WISE at %g sigma, %d pinned at 1.000, %d E or Xe class; "
                   "%d unexplained" % (len(out), albedo, sum("artefact" in o["vet"] for o in out), sigma, sum("fit-boundary" in o["vet"] for o in out),
                                       sum("asteroid-class" in o["vet"] for o in out), len(left)))
        core.clear_targets("sbdb_mirror_asteroids")
        core.record_targets("sbdb_mirror_asteroids", [dict(
            target="mpc:%s" % o["pdes"], stat=o["albedo"], unit="geometric albedo", vet="; ".join(o["vet"]) or None,
            note="%s, D %s km, class %s, AKARI %s" % (o["name"], o["D"], o["cls"] or "-", "%.2f +- %.2f" % o["akari"][:2] if o["akari"] and o["akari"][0] is not None else "-"))
            for o in out])
        print("recorded")


retrograde_DOC = r"""Retrograde orbits inside Jupiter's: Small-Body Database asteroids on bound orbits with perihelia beyond 1.3 au, semi-major axes below 4.5 au and inclinations above 90 deg, bodies circling against the planets' flow where collisions and the planets would have removed them, each read for its observational arc and orbit quality and its Tisserand parameter, since a comet-like orbit marks a capture.

    python -I seti.py solar_system retrograde [record] [--i-min-deg 90] [--a-max-au 4.5] [--q-min-au 1.3] [--condition-max 6] [--arc-min-days 30] [--tj-min 2]"""


def retrograde_main(record=False, i_min_deg=90.0, a_max_au=4.5, q_min_au=1.3, condition_max=6, arc_min_days=30.0, tj_min=2.0):
    rows = sbdb_rows(["i|GT|%g" % i_min_deg, "a|LT|%g" % a_max_au, "a|GT|0", "e|LT|1", "q|GT|%g" % q_min_au],
                     "full_name,pdes,a,e,i,q,H,condition_code,data_arc,n_obs_used,t_jup,class")
    out = []
    for r in rows:
        cc = num(r["condition_code"])
        vet = []
        if cc is not None and cc >= condition_max or (num(r["data_arc"]) or 0) < arc_min_days:
            vet.append("unconstrained-geometry")
        if (num(r["t_jup"]) or 3) < tj_min:
            vet.append("natural-type")
        out.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], a=num(r["a"]), e=num(r["e"]), i=num(r["i"]), q=num(r["q"]),
                        H=num(r["H"]), cc=cc, arc=num(r["data_arc"]), n=num(r["n_obs_used"]), tj=num(r["t_jup"]), vet=vet))
    print("%d asteroids with i above %g deg, a below %g au and q beyond %g au" % (len(out), i_min_deg, a_max_au, q_min_au))
    for o in out:
        print("  %s: a %.3f, e %.3f, i %.1f, q %.2f, H %s, condition %s, arc %s d, %s observations, T_J %s; %s" % (
            o["name"], o["a"], o["e"], o["i"], o["q"], o["H"], o["cc"], o["arc"], o["n"], o["tj"], ", ".join(o["vet"]) or "unexplained"))
    json.dump(out, open(core.results("retrograde_asteroids.json"), "w"), indent=1)
    if record:
        left = [o for o in out if not o["vet"]]
        core.record_search(
            "sbdb_retrograde_inner", "Retrograde orbits inside Jupiter's", archive="JPL Small-Body Database query API",
            selection="asteroids with i above %g deg, a below %g au and q beyond %g au: %d" % (i_min_deg, a_max_au, q_min_au, len(out)), code=os.path.abspath(__file__),
            entry="retrograde", n_targets=len(out), n_unit="asteroids", params=dict(i_deg=i_min_deg, a_au=a_max_au, q_au=q_min_au, condition_max=condition_max,
                                                                                    arc_min_days=arc_min_days, tj_min=tj_min),
            null_summary="each orbit's arc, quality and Tisserand parameter", sensitivity="any catalogued retrograde body between Mars-crossing and Jupiter",
            ledger="Retrograde orbits inside Jupiter's",
            result="%d: %s" % (len(out), "; ".join("%s (a %.2f, i %.0f, %s)" % (o["name"], o["a"], o["i"], ", ".join(o["vet"]) or "unexplained") for o in out)))
        core.clear_targets("sbdb_retrograde_inner")
        core.record_targets("sbdb_retrograde_inner", [dict(
            target="mpc:%s" % o["pdes"], stat=o["i"], unit="inclination, deg", vet="; ".join(o["vet"]) or None,
            note="a %.3f au, e %.3f, q %.2f, H %s, condition %s, arc %s d, T_J %s" % (o["a"], o["e"], o["q"], o["H"], o["cc"], o["arc"], o["tj"]))
            for o in out])
        print("recorded; %d unexplained" % len(left))


earthorbit_DOC = r"""Objects on Earth's own orbit: Small-Body Database asteroids with semi-major axes of 0.95-1.05 au, eccentricities below 0.05 and inclinations below 3 deg, something parked on our path round the Sun, each read for a non-gravitational acceleration (a light body under radiation pressure, as a spent rocket stage is) and for its size from H, against the near-Earth population that slow encounters make easy to find.

    python -I seti.py solar_system earthorbit [record] [--a-min 0.95] [--a-max 1.05] [--e-max 0.05] [--i-max-deg 3] [--sigma 3] [--stage-m2-kg 5e-3] [--rubble-kg-m3 500]"""


EARTHORBIT_STAGE = 5e-3  # m^2/kg, the area-to-mass ratio of a spent rocket stage and above
EARTHORBIT_RUBBLE = 500.0  # kg/m^3


def earthorbit_main(record=False, a_min=0.95, a_max=1.05, e_max=0.05, i_max_deg=3.0, sigma=3.0, stage_m2_kg=EARTHORBIT_STAGE, rubble_kg_m3=EARTHORBIT_RUBBLE):
    rows = sbdb_rows(["a|GT|%g" % a_min, "a|LT|%g" % a_max, "e|LT|%g" % e_max, "i|LT|%g" % i_max_deg],
                     "full_name,pdes,a,e,i,H,A1,A1_sigma,A2,A2_sigma,DT,first_obs,last_obs,data_arc,n_obs_used,condition_code")
    out = []
    for r in rows:
        H, a1, s1 = num(r["H"]), num(r["A1"]), num(r["A1_sigma"])
        o = dict(name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], a=num(r["a"]), e=num(r["e"]), i=num(r["i"]), H=H, A1=a1, A1_sigma=s1,
                 A2=num(r["A2"]), first=r["first_obs"], last=r["last_obs"], arc=num(r["data_arc"]), cc=num(r["condition_code"]),
                 D_m=1329e3 / np.sqrt(0.14) * 10 ** (-H / 5) if H is not None else None, amr=None, rho_max=None)
        if a1 is not None and a1 > 0 and s1 and a1 > sigma * s1 and H is not None:
            # radiation pressure at 1 au with a reflection coefficient of 1.5; the densest body allowed is the smallest, at albedo 0.5
            o["amr"] = a1 * nga_AU_D2 / (nga_S_OVER_C * 1.5)
            o["rho_max"] = 3.0 / (2.0 * 1329e3 / np.sqrt(0.5) * 10 ** (-H / 5) * o["amr"])
        o["vet"] = ("not-measured" if o["amr"] is None else
                    "natural-density" if o["amr"] < stage_m2_kg and o["rho_max"] >= rubble_kg_m3 else None)
        out.append(o)
    ng = [o for o in out if o["amr"] is not None]
    print("%d asteroids on Earth-like orbits (a %g-%g au, e below %g, i below %g deg); a radiation-pressure term at %g sigma for %d" % (
        len(out), a_min, a_max, e_max, i_max_deg, sigma, len(ng)))
    for o in sorted(out, key=lambda o: o["H"] or 0):
        print("  %s: a %.4f, e %.4f, i %.2f, H %s (%s m at albedo 0.14), A1 %s +- %s, A2 %s, %s to %s, condition %s%s" % (
            o["name"], o["a"], o["e"], o["i"], o["H"], "%.0f" % o["D_m"] if o["D_m"] else "-", o["A1"], o["A1_sigma"], o["A2"], o["first"], o["last"],
            o["cc"], "; area to mass %.1e m2/kg, densest allowed %.0f kg/m3" % (o["amr"], o["rho_max"]) if o["amr"] else ""))
    json.dump(out, open(core.results("earth_orbit_objects.json"), "w"), indent=1)
    if record:
        core.record_search(
            "sbdb_earth_orbit", "Objects on Earth's own orbit", archive="JPL Small-Body Database query API",
            selection="asteroids with a %g-%g au, e below %g, i below %g deg: %d" % (a_min, a_max, e_max, i_max_deg, len(out)), code=os.path.abspath(__file__),
            entry="earthorbit", n_targets=len(out), n_unit="asteroids", params=dict(a_au=[a_min, a_max], e=e_max, i_deg=i_max_deg, sigma=sigma,
                                                                                    stage_m2_kg=stage_m2_kg, rubble_kg_m3=rubble_kg_m3),
            null_summary="the near-Earth population, which slow encounters make easy to find on these orbits",
            sensitivity="a body on Earth's orbit within a %g-%g au, e %g and i %g deg that the survey catalogue holds, and a spent stage's "
                        "area-to-mass ratio of %.0e m2/kg where radiation pressure is measured" % (a_min, a_max, e_max, i_max_deg, stage_m2_kg),
            ledger="Objects on Earth's own orbit",
            result="%d bodies; radiation pressure measured for %d (%s), %d at a rocket stage's area-to-mass, %d of them as dense as rubble at the "
                   "most favourable albedo" % (len(out), len(ng), ", ".join(o["name"] for o in ng) or "none", sum(o["amr"] >= stage_m2_kg for o in ng),
                                               sum(o["vet"] == "natural-density" for o in ng)))
        core.clear_targets("sbdb_earth_orbit")
        core.record_targets("sbdb_earth_orbit", [dict(
            target="mpc:%s" % o["pdes"], stat=o["amr"], unit="area-to-mass ratio, m2/kg", vet=o["vet"],
            note="a %.4f au, e %.4f, i %.2f deg, H %s, A1 %s +- %s, A2 %s, %s to %s%s" % (
                o["a"], o["e"], o["i"], o["H"], o["A1"], o["A1_sigma"], o["A2"], o["first"], o["last"],
                ", densest allowed %.0f kg/m3" % o["rho_max"] if o["rho_max"] else "")) for o in out])
        print("recorded")


yearcomets_DOC = r"""Comets on Earth's years: periodic comets of JPL's Small-Body Database with periods below 200 years whose period sits within 0.002 years (0.7 days) of a whole number of years, an orbit that returns to Earth's longitude on a fixed calendar, against the same window at half-years, each read for its orbit quality (condition code, data arc and the period's own uncertainty); a comet's period also walks under non-gravitational forces, so a match holds only to the orbit's precision.

    python -I seti.py solar_system yearcomets [record] [--tol-y 0.002] [--p-max-y 200] [--condition-max 6]"""


def yearcomets_main(record=False, tol_y=0.002, p_max_y=200.0, condition_max=6):
    from scipy.stats import poisson
    rows = sbdb_rows(["e|LT|1", "per_y|LT|%g" % p_max_y, "per_y|GT|0"],
                     "full_name,pdes,e,q,i,per_y,per,sigma_per,data_arc,n_obs_used,condition_code,class,A1,A2", kind="c")
    out, half = [], []
    for r in rows:
        P = num(r["per_y"])
        if P is None or P < 0.5:
            continue
        sp = num(r["sigma_per"]) / 365.25 if num(r["sigma_per"]) is not None else None
        n = round(P)
        frac = abs(P - n)
        cc = num(r["condition_code"])
        o = dict(name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], P=P, n=int(n), frac=frac, sigma_y=sp, e=num(r["e"]), q=num(r["q"]), i=num(r["i"]),
                 cc=cc, arc=num(r["data_arc"]), nobs=num(r["n_obs_used"]), cls=r["class"], nongrav=bool(num(r["A1"]) or num(r["A2"])))
        if n >= 1 and frac < tol_y:
            # a lost comet carries a rounded period with no uncertainty or condition code
            o["vet"] = ("not-measured" if cc is None or sp is None else "unconstrained-geometry" if cc >= condition_max or sp > tol_y else None)
            out.append(o)
        elif abs(P - (math.floor(P) + 0.5)) < tol_y:
            half.append(o)
    exp = len(rows) * 2 * tol_y
    p = float(poisson.sf(len(out) - 1, exp)) if out else 1.0
    firm = [o for o in out if o["vet"] is None]
    for o in firm:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d periodic comets with periods below %g years; within %g y of a whole number of years: %d (%.2f expected from uniform fractions, Poisson p %.3g); "
          "within the same window of a half-year: %d; with firm orbits: %d" % (len(rows), p_max_y, tol_y, len(out), exp, p, len(half), len(firm)))
    for o in sorted(out, key=lambda o: o["frac"]):
        print("  %s: P %.4f y (%d, off by %.4f y, sigma %s), q %.2f au, e %.3f, i %.1f, arc %s d, %s obs, condition %s%s; %s" % (
            o["name"], o["P"], o["n"], o["frac"], "%.4f" % o["sigma_y"] if o["sigma_y"] is not None else "-", o["q"], o["e"], o["i"], o["arc"], o["nobs"], o["cc"],
            ", non-gravitational terms" if o["nongrav"] else "", o["vet"] or "unexplained"))
    json.dump(dict(hits=out, half=half, n=len(rows)), open(core.results("comets_earth_years.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_comet_years", "Comets on Earth's years", "JPL Small-Body Database query API", "periodic comets with periods below %g years: %d" % (p_max_y, len(rows)),
            os.path.abspath(__file__), "yearcomets", len(rows), "comets", dict(tol_y=tol_y, p_max_y=p_max_y, condition_max=condition_max),
            "uniform fractional periods give %.2f in the window; the half-year window holds %d" % (exp, len(half)),
            "a comet whose period sits within %g years of a whole number of years at its orbit's precision" % tol_y,
            "%d at whole years against %.2f expected and %d at half-years (Poisson p %.3g); %d with firm orbits" % (len(out), exp, len(half), p, len(firm)),
            [dict(target="sbdb:%s" % o["pdes"], stat=o["frac"], unit="years from a whole number of years", vet=o["vet"],
                  note="%s, P %.4f y (%d), sigma %s y, q %.2f au, e %.3f, arc %s d, condition %s" % (
                      o["name"], o["P"], o["n"], "%.4f" % o["sigma_y"] if o["sigma_y"] is not None else "-", o["q"], o["e"], o["arc"], o["cc"])) for o in out])
        print("recorded")


tangent_DOC = r"""Orbits tangent to Earth's: asteroids and comets of JPL's Small-Body Database whose perihelion or aphelion distance sits within 0.0005 au of 1.0000 au, an orbit that touches Earth's mean distance from the Sun, as a craft parked to meet Earth would have, against the same windows at 0.995 and 1.005 au and the band's own density from 0.99 to 1.01 au, each read for its orbit quality (condition code, data arc and the distance's own uncertainty); a near-Earth population crosses 1 au freely and no dynamics pins an orbit to the number itself.

    python -I seti.py solar_system tangent [record] [--tol-au 0.0005] [--band-au 0.01] [--condition-max 6]"""


def tangent_main(record=False, tol_au=0.0005, band_au=0.01, condition_max=6):
    from scipy.stats import poisson
    out, counts = [], {}
    for key, sig, label in (("q", "sigma_q", "perihelion"), ("ad", "sigma_ad", "aphelion")):
        rows = []
        for kind in ("a", "c"):
            rows += [dict(r, kind=kind) for r in sbdb_rows(["%s|GT|%g" % (key, 1 - band_au), "%s|LT|%g" % (key, 1 + band_au), "e|LT|1"],
                                                            "full_name,pdes,q,ad,e,a,i,%s,condition_code,data_arc,n_obs_used,H" % sig, kind=kind)]
        vals = [(r, num(r[key])) for r in rows if num(r[key]) is not None]
        hits = [(r, v) for r, v in vals if abs(v - 1.0) < tol_au]
        ctl = [v for r, v in vals if abs(v - (1 - band_au / 2)) < tol_au or abs(v - (1 + band_au / 2)) < tol_au]
        exp = len(vals) * 2 * tol_au / (2 * band_au)
        p = float(poisson.sf(len(hits) - 1, exp)) if hits else 1.0
        counts[label] = dict(n=len(vals), hits=len(hits), control=len(ctl), expected=exp, p=p)
        for r, v in hits:
            cc, s = num(r["condition_code"]), num(r[sig])
            out.append(dict(which=label, name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], kind=r["kind"], value=v, off=v - 1.0, sigma=s, q=num(r["q"]),
                            ad=num(r["ad"]), e=num(r["e"]), a=num(r["a"]), i=num(r["i"]), H=num(r["H"]), cc=cc, arc=num(r["data_arc"]), nobs=num(r["n_obs_used"]),
                            vet=("not-measured" if cc is None or s is None else "unconstrained-geometry" if cc >= condition_max or s > tol_au else
                                 "chance-rate" if p > 0.01 else None)))
        print("%s within %g au of 1 au: %d of %d bodies in the %g au band (%.2f expected from the band's density, Poisson p %.3g); the two control windows hold %d" % (
            label, tol_au, len(hits), len(vals), band_au, exp, p, len(ctl)))
    for o in sorted(out, key=lambda o: abs(o["off"])):
        print("  %s %s: %s %.5f au (off by %+.5f, sigma %s), a %.3f, e %.3f, i %.1f, H %s, arc %s d, %s obs, condition %s; %s" % (
            o["which"], o["name"], o["which"], o["value"], o["off"], "%.1e" % o["sigma"] if o["sigma"] is not None else "-", o["a"], o["e"], o["i"], o["H"], o["arc"],
            o["nobs"], o["cc"], o["vet"] or "unexplained"))
    json.dump(dict(counts=counts, hits=out), open(core.results("tangent_earth.json"), "w"), indent=1)
    if record:
        c = counts
        core.record_run(
            "sbdb_tangent_earth", "Orbits tangent to Earth's", "JPL Small-Body Database query API",
            "asteroids and comets with perihelia or aphelia within %g au of 1 au: %d and %d" % (band_au, c["perihelion"]["n"], c["aphelion"]["n"]),
            os.path.abspath(__file__), "tangent", c["perihelion"]["n"] + c["aphelion"]["n"], "orbits", dict(tol_au=tol_au, band_au=band_au, condition_max=condition_max),
            "the band's own density gives %.2f and %.2f in the windows; the control windows at %g and %g au hold %d and %d" % (
                c["perihelion"]["expected"], c["aphelion"]["expected"], 1 - band_au / 2, 1 + band_au / 2, c["perihelion"]["control"], c["aphelion"]["control"]),
            "a perihelion or aphelion within %g au of 1 au at the orbit's precision" % tol_au,
            "perihelia %d against %.2f expected (p %.3g), aphelia %d against %.2f (p %.3g)" % (
                c["perihelion"]["hits"], c["perihelion"]["expected"], c["perihelion"]["p"], c["aphelion"]["hits"], c["aphelion"]["expected"], c["aphelion"]["p"]),
            [dict(target="sbdb:%s" % o["pdes"], stat=o["off"], unit="%s distance less 1 au" % o["which"], vet=o["vet"],
                  note="%s, a %.3f au, e %.3f, i %.1f deg, H %s, sigma %s, arc %s d, condition %s" % (
                      o["name"], o["a"], o["e"], o["i"], o["H"], o["sigma"], o["arc"], o["cc"])) for o in out])
        print("recorded")


hourspins_DOC = r"""Asteroid spins on the clock: rotation periods of the Asteroid Lightcurve Database's frequency-diameter table with secure quality (U of 3) and at least three decimals, those within 0.02 per cent of a whole number of hours between 2 and 48, a spin set to Earth's hours, against the same windows at half-hours and the count uniform fractions give; a period near 24 hours is also where diurnal aliasing leaves wrong values, so the day and its harmonics are counted apart.

    python -I seti.py solar_system hourspins [record] [--tol-rel 0.0002] [--p-min-h 2] [--p-max-h 48]"""


def hourspins_main(record=False, tol_rel=0.0002, p_min_h=2.0, p_max_h=48.0):
    from scipy.stats import poisson
    p = core.data("lcdb", "F-D_BASIC.TXT")
    lines = open(p, encoding="latin-1").read().splitlines()
    hi = [i for i, ln in enumerate(lines) if ln.startswith("Number")][0]
    hdr = lines[hi]
    fam, nm, cl = hdr.index("Fam"), hdr.index("Name"), hdr.index("Class")
    out, half, n_sec, sum_p = [], [], 0, 0.0
    for ln in lines[hi + 2:]:
        if not ln.strip():
            continue
        tok = ln[fam:].split()
        try:
            H, D, P = float(tok[1].strip("<>")), float(tok[3].strip("<>")), float(tok[4].strip("<>"))
        except (ValueError, IndexError):
            continue
        U = next((x for x in tok[6:] if re.fullmatch(r"[0-3][+-]?", x)), None)
        dec = len(tok[4].split(".")[1]) if "." in tok[4] else 0
        if U is None or not U.startswith("3") or dec < 3 or not p_min_h <= P <= p_max_h:
            continue
        n_sec += 1
        sum_p += P
        n = round(P)
        o = dict(num=ln[:nm].strip(), name=ln[nm:cl].strip(), H=H, D=D, P=P, n=int(n), off=P - n, U=U, dec=dec, day=n in (6, 8, 12, 24, 48))
        if abs(P - n) < tol_rel * P:
            out.append(o)
        elif abs(P - (math.floor(P) + 0.5)) < tol_rel * P:
            half.append(o)
    # uniform fractions: each period's window covers 2 tol_rel P of the unit interval
    exp = 2 * tol_rel * sum_p
    pday = [o for o in out if o["day"]]
    rest = [o for o in out if not o["day"]]
    pp = float(poisson.sf(len(out) - 1, exp)) if out else 1.0
    for o in out:
        o["vet"] = "sky-sidereal" if o["day"] else ("chance-rate" if pp > 0.01 else None)
    print("%d secure periods with three decimals between %g and %g h; within %g of a whole hour: %d (%.1f expected from uniform fractions, Poisson p %.3g), %d of them at "
          "the day's harmonics 6, 8, 12, 24 and 48 h; at half-hours: %d" % (n_sec, p_min_h, p_max_h, tol_rel, len(out), exp, pp, len(pday), len(half)))
    for o in sorted(out, key=lambda o: abs(o["off"]) / o["P"]):
        print("  %-7s %-16s P %.4f h (%d, off by %+.4f), D %.2f km, H %.2f, U %s; %s" % (o["num"], o["name"], o["P"], o["n"], o["off"], o["D"], o["H"], o["U"], o["vet"]))
    json.dump(dict(n=n_sec, hits=out, half=half), open(core.results("asteroid_hour_spins.json"), "w"), indent=1)
    if record:
        core.record_run(
            "lcdb_hour_spins", "Asteroid spins on the clock", "Asteroid Lightcurve Database (LCDB) frequency-diameter table",
            "secure periods (U 3) with three decimals between %g and %g h: %d" % (p_min_h, p_max_h, n_sec), os.path.abspath(__file__), "hourspins", n_sec, "asteroids",
            dict(tol_rel=tol_rel, p_h=[p_min_h, p_max_h]), "uniform fractions give %.1f in the windows; the half-hour windows hold %d" % (exp, len(half)),
            "a secure period within %g of a whole number of hours" % tol_rel,
            "%d at whole hours against %.1f expected and %d at half-hours (Poisson p %.3g); %d sit at the day's harmonics" % (len(out), exp, len(half), pp, len(pday)),
            [dict(target="mpc:%s" % (o["num"] or o["name"]), stat=o["off"], unit="hours from a whole hour", vet=o["vet"],
                  note="%s, P %.4f h, D %.2f km, U %s" % (o["name"], o["P"], o["D"], o["U"])) for o in out])
        print("recorded")


lagrange_DOC = r"""Lurkers at Earth's Lagrange points: asteroids of JPL's Small-Body Database with semi-major axes within 0.01 au of Earth's, eccentricities below 0.3 and inclinations below 30 deg, placed by their mean longitude at epoch relative to Earth's; a body kept at L4 or L5 sits 60 deg ahead of or behind Earth, where a watcher of Earth would park, and the count within 20 deg of each point is set against the other longitudes of the same ring. Earth's own Trojans and horseshoe companions sit there by nature, so each match is read for its name, arc and condition code.

    python -I seti.py solar_system lagrange [record] [--da-au 0.01] [--e-max 0.3] [--i-max-deg 30] [--half-width-deg 20]"""


def lagrange_main(record=False, da_au=0.01, e_max=0.3, i_max_deg=30.0, half_width_deg=20.0):
    from scipy.stats import poisson
    rows = sbdb_rows(["a|GT|%g" % (1 - da_au), "a|LT|%g" % (1 + da_au), "e|LT|%g" % e_max, "i|LT|%g" % i_max_deg],
                     "full_name,pdes,a,e,i,om,w,ma,epoch,H,data_arc,n_obs_used,condition_code")
    out = []
    for r in rows:
        om, w, ma, ep = num(r["om"]), num(r["w"]), num(r["ma"]), num(r["epoch"])
        if None in (om, w, ma, ep):
            continue
        lam = (om + w + ma) % 360.0
        # Earth's mean longitude at the epoch (Simon et al. 1994)
        earth = (100.46435 + 0.985609101 * (ep - 2451545.0)) % 360.0
        d = (lam - earth + 180.0) % 360.0 - 180.0
        out.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], a=num(r["a"]), e=num(r["e"]), i=num(r["i"]), H=num(r["H"]), arc=num(r["data_arc"]),
                        nobs=num(r["n_obs_used"]), cc=num(r["condition_code"]), epoch=ep, dlon=d))
    l4 = [o for o in out if abs(o["dlon"] - 60) < half_width_deg]
    l5 = [o for o in out if abs(o["dlon"] + 60) < half_width_deg]
    rest = [o for o in out if abs(o["dlon"] - 60) >= half_width_deg and abs(o["dlon"] + 60) >= half_width_deg]
    exp = len(out) * (2 * half_width_deg) / 360.0
    hist = {lo: sum(1 for o in out if lo <= o["dlon"] < lo + 30) for lo in range(-180, 180, 30)}
    p4, p5 = (float(poisson.sf(len(x) - 1, exp)) if x else 1.0 for x in (l4, l5))
    for o in l4 + l5:
        o["point"] = "L4" if o in l4 else "L5"
        o["vet"] = ("unconstrained-geometry" if (o["cc"] or 0) >= 6 or (o["arc"] or 0) < 30 else "chance-rate" if min(p4, p5) > 0.01 else None)
    print("%d asteroids within %g au of Earth's semi-major axis (e below %g, i below %g deg); within %g deg of L4: %d, of L5: %d, against %.1f expected per point from the ring "
          "(Poisson p %.3g and %.3g); longitudes relative to Earth by 30 deg bins: %s" % (len(out), da_au, e_max, i_max_deg, half_width_deg, len(l4), len(l5), exp, p4, p5, hist))
    for o in sorted(l4 + l5, key=lambda o: abs(abs(o["dlon"]) - 60)):
        print("  %s %s: %+.1f deg from Earth, a %.4f, e %.3f, i %.1f, H %s, arc %s d, condition %s; %s" % (
            o["point"], o["name"], o["dlon"], o["a"], o["e"], o["i"], o["H"], o["arc"], o["cc"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(out), hist=hist, l4=l4, l5=l5), open(core.results("earth_lagrange_points.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_earth_lagrange", "Lurkers at Earth's Lagrange points", "JPL Small-Body Database query API",
            "asteroids within %g au of Earth's semi-major axis with e below %g and i below %g deg: %d" % (da_au, e_max, i_max_deg, len(out)), os.path.abspath(__file__),
            "lagrange", len(out), "asteroids", dict(da_au=da_au, e_max=e_max, i_max_deg=i_max_deg, half_width_deg=half_width_deg),
            "the ring's other longitudes: %.1f expected per point; the 30 deg bins hold %s" % (exp, list(hist.values())),
            "a body within %g deg of L4 or L5 on an orbit the catalogue holds" % half_width_deg,
            "%d at L4 and %d at L5 against %.1f expected (p %.3g and %.3g)" % (len(l4), len(l5), exp, p4, p5),
            [dict(target="sbdb:%s" % o["pdes"], stat=o["dlon"], unit="deg of mean longitude ahead of Earth", vet=o["vet"],
                  note="%s at %s, a %.4f au, e %.3f, i %.1f deg, H %s, arc %s d, condition %s" % (o["name"], o["point"], o["a"], o["e"], o["i"], o["H"], o["arc"], o["cc"]))
             for o in l4 + l5])
        print("recorded")


sunward_DOC = r"""Comets pushed toward the Sun: comets and asteroids of JPL's Small-Body Database whose radial non-gravitational term A1 is negative at three sigma, an acceleration toward the Sun that outgassing, which pushes a nucleus away from the Sun, cannot give and a thrust could; each read for its arc, condition code, the transverse term A2 and whether the solution carries a non-standard outgassing law, since a short arc, a correlated A1 and A2 pair and a fragmenting nucleus each leave a wrong sign.

    python -I seti.py solar_system sunward [record] [--sigma 3]"""


def sunward_main(record=False, sigma=3.0):
    out, n = [], {}
    for kind in ("c", "a"):
        rows = sbdb_rows(["A1|LT|0"], "full_name,pdes,A1,A1_sigma,A2,A2_sigma,A3,A3_sigma,DT,e,q,a,i,data_arc,n_obs_used,condition_code,class", kind=kind)
        n[kind] = len(rows)
        for r in rows:
            a1, s1 = num(r["A1"]), num(r["A1_sigma"])
            if a1 is None or not s1:
                continue
            z = -a1 / s1
            if z < sigma:
                continue
            cc = num(r["condition_code"])
            a2, s2 = num(r["A2"]), num(r["A2_sigma"])
            out.append(dict(kind="comet" if kind == "c" else "asteroid", name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], A1=a1, A1_sigma=s1, z=z, A2=a2, A2_sigma=s2,
                            A3=num(r["A3"]), DT=num(r["DT"]), e=num(r["e"]), q=num(r["q"]), a=num(r["a"]), i=num(r["i"]), arc=num(r["data_arc"]), nobs=num(r["n_obs_used"]),
                            cc=cc, cls=r["class"],
                            vet=("unconstrained-geometry" if (cc is not None and cc >= 6) or (num(r["data_arc"]) or 0) < 100 else
                                 "fit-degeneracy" if a2 is not None and s2 and abs(a2 / s2) >= sigma and abs(a2) > abs(a1) else
                                 "fit-degeneracy" if num(r["DT"]) is not None else None)))
    left = [o for o in out if o["vet"] is None]
    print("bodies with a negative A1 in the Small-Body Database: %d comets and %d asteroids; negative at %g sigma: %d, of them %d on short or poor arcs, %d with a larger "
          "transverse term or a shifted outgassing law; unexplained %d" % (n["c"], n["a"], sigma, len(out), sum(o["vet"] == "unconstrained-geometry" for o in out),
                                                                           sum(o["vet"] == "fit-degeneracy" for o in out), len(left)))
    for o in sorted(out, key=lambda o: -o["z"]):
        print("  %s %s: A1 %.3e +- %.1e (%.1f sigma), A2 %s +- %s, A3 %s, DT %s; q %.2f au, e %.3f, arc %s d, %s obs, condition %s; %s" % (
            o["kind"], o["name"], o["A1"], o["A1_sigma"], o["z"], o["A2"], o["A2_sigma"], o["A3"], o["DT"], o["q"], o["e"], o["arc"], o["nobs"], o["cc"], o["vet"] or "unexplained"))
    json.dump(out, open(core.results("sunward_pushed.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_sunward_a1", "Comets pushed toward the Sun", "JPL Small-Body Database query API",
            "comets and asteroids with a negative radial non-gravitational term: %d and %d" % (n["c"], n["a"]), os.path.abspath(__file__), "sunward", n["c"] + n["a"],
            "bodies", dict(sigma=sigma), "each solution's arc, condition code, transverse term and outgassing law",
            "a radial non-gravitational acceleration toward the Sun at %g sigma" % sigma,
            "%d at %g sigma: %d on short or poor arcs, %d with a larger transverse term or a shifted law; %d unexplained" % (
                len(out), sigma, sum(o["vet"] == "unconstrained-geometry" for o in out), sum(o["vet"] == "fit-degeneracy" for o in out), len(left)),
            [dict(target="sbdb:%s" % o["pdes"], stat=o["z"], unit="sigma of A1 below zero", vet=o["vet"],
                  note="%s %s, A1 %.3e +- %.1e, A2 %s +- %s, DT %s, q %.2f au, arc %s d, condition %s" % (
                      o["kind"], o["name"], o["A1"], o["A1_sigma"], o["A2"], o["A2_sigma"], o["DT"], o["q"], o["arc"], o["cc"])) for o in out])
        print("recorded")


polar_DOC = r"""Asteroids on polar orbits: asteroids and comets of JPL's Small-Body Database on bound orbits whose inclination lies within 0.5 deg of 90 deg, an orbit standing square to the planets' plane, where no resonance or planet places a body and a made orbit surveying the system from above would sit, against the bands 1-3 deg either side and the count a flat inclination density gives, each read for its arc and condition code.

    python -I seti.py solar_system polar [record] [--half-width-deg 0.5] [--band-deg 3] [--condition-max 6]"""


def polar_main(record=False, half_width_deg=0.5, band_deg=3.0, condition_max=6):
    from scipy.stats import poisson
    rows = []
    for kind in ("a", "c"):
        rows += [dict(r, kind=kind) for r in sbdb_rows(["i|GT|%g" % (90 - band_deg), "i|LT|%g" % (90 + band_deg), "e|LT|1"],
                                                        "full_name,pdes,a,e,i,q,H,data_arc,n_obs_used,condition_code,class", kind=kind)]
    hits = [r for r in rows if abs(num(r["i"]) - 90) < half_width_deg]
    ctl = [r for r in rows if abs(num(r["i"]) - 90) >= half_width_deg]
    exp = len(rows) * (2 * half_width_deg) / (2 * band_deg)
    p = float(poisson.sf(len(hits) - 1, exp)) if hits else 1.0
    out = []
    for r in hits:
        cc = num(r["condition_code"])
        out.append(dict(kind="comet" if r["kind"] == "c" else "asteroid", name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], a=num(r["a"]), e=num(r["e"]), i=num(r["i"]),
                        q=num(r["q"]), H=num(r["H"]), arc=num(r["data_arc"]), nobs=num(r["n_obs_used"]), cc=cc, cls=r["class"],
                        vet=("unconstrained-geometry" if (cc is not None and cc >= condition_max) or (num(r["data_arc"]) or 0) < 30 else "chance-rate" if p > 0.01 else None)))
    print("%d bound orbits within %g deg of 90 deg inclination; within %g deg: %d (%.1f expected from a flat density, Poisson p %.3g); the bands either side hold %d" % (
        len(rows), band_deg, half_width_deg, len(hits), exp, p, len(ctl)))
    for o in sorted(out, key=lambda o: abs(o["i"] - 90)):
        print("  %s %s: i %.3f, a %.2f au, e %.3f, q %.2f au, H %s, arc %s d, condition %s, %s; %s" % (
            o["kind"], o["name"], o["i"], o["a"], o["e"], o["q"], o["H"], o["arc"], o["cc"], o["cls"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(rows), hits=out), open(core.results("polar_orbits.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_polar_orbits", "Asteroids on polar orbits", "JPL Small-Body Database query API", "bound orbits within %g deg of 90 deg inclination: %d" % (band_deg, len(rows)),
            os.path.abspath(__file__), "polar", len(rows), "orbits", dict(half_width_deg=half_width_deg, band_deg=band_deg, condition_max=condition_max),
            "a flat inclination density across the band gives %.1f; the bands either side hold %d" % (exp, len(ctl)),
            "an orbit within %g deg of polar that the catalogue holds" % half_width_deg, "%d against %.1f expected (Poisson p %.3g)" % (len(hits), exp, p),
            [dict(target="sbdb:%s" % o["pdes"], stat=o["i"] - 90, unit="deg from a polar orbit", vet=o["vet"],
                  note="%s %s, a %.2f au, e %.3f, q %.2f au, H %s, arc %s d, condition %s, %s" % (o["kind"], o["name"], o["a"], o["e"], o["q"], o["H"], o["arc"], o["cc"], o["cls"]))
             for o in out])
        print("recorded")


giantcircles_DOC = r"""Circles between the giant planets: asteroids and comets of JPL's Small-Body Database with semi-major axes of 5.5-30 au and eccentricities below 0.02, orbits circular to two per cent in the region the giant planets stir, where a Centaur's orbit is thrown about on a scale of a million years and a circle lasts no longer, so a body kept on one is held there; against the Centaurs' own eccentricities, each read for its arc and condition code and for a resonance with a planet that could hold it.

    python -I seti.py solar_system giantcircles [record] [--a-min-au 5.5] [--a-max-au 30] [--e-max 0.02] [--condition-max 6]"""


def giantcircles_main(record=False, a_min_au=5.5, a_max_au=30.0, e_max=0.02, condition_max=6):
    rows = []
    for kind in ("a", "c"):
        rows += [dict(r, kind=kind) for r in sbdb_rows(["a|GT|%g" % a_min_au, "a|LT|%g" % a_max_au, "e|LT|%g" % e_max, "e|GE|0", "q|GT|5.2"],
                                                        "full_name,pdes,a,e,i,q,H,data_arc,n_obs_used,condition_code,class", kind=kind)]
    all_rows = []
    for kind in ("a",):
        all_rows += sbdb_rows(["a|GT|%g" % a_min_au, "a|LT|%g" % a_max_au, "e|LT|0.3", "q|GT|5.2"], "pdes,e,class", kind=kind)
    es = np.array([num(r["e"]) for r in all_rows if num(r["e"]) is not None])
    planets = {"Jupiter": 5.2026, "Saturn": 9.5549, "Uranus": 19.2184, "Neptune": 30.110}
    out = []
    for r in rows:
        a, cc = num(r["a"]), num(r["condition_code"])
        # a semi-major axis within 2 per cent of a planet's, or at a 3:2, 2:1 or 1:2 ratio, is a resonance that can hold a circle
        res = [(pn, ratio) for pn, pa in planets.items() for ratio in (1.0, (3 / 2) ** (2 / 3), 2 ** (2 / 3), 2 ** (-2 / 3)) if abs(a / (pa * ratio) - 1) < 0.02]
        out.append(dict(kind="comet" if r["kind"] == "c" else "asteroid", name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], a=a, e=num(r["e"]), i=num(r["i"]),
                        q=num(r["q"]), H=num(r["H"]), arc=num(r["data_arc"]), nobs=num(r["n_obs_used"]), cc=cc, cls=r["class"], resonance=res,
                        vet=("unconstrained-geometry" if (cc is not None and cc >= condition_max) or (num(r["data_arc"]) or 0) < 365 else "near-resonance" if res else None)))
    left = [o for o in out if o["vet"] is None]
    print("%d bodies with a %g-%g au and e below 0.3 beyond Jupiter; eccentricities median %.3f, 5th percentile %.3f; below %g: %d, %d on short or poor arcs, %d near a "
          "planet's resonance; unexplained %d" % (len(es), a_min_au, a_max_au, float(np.median(es)) if len(es) else 0, float(np.percentile(es, 5)) if len(es) else 0, e_max,
                                                   len(out), sum(o["vet"] == "unconstrained-geometry" for o in out), sum(o["vet"] == "near-resonance" for o in out), len(left)))
    for o in sorted(out, key=lambda o: o["e"]):
        print("  %s %s: a %.3f au, e %.4f, i %.1f, q %.2f, H %s, arc %s d, %s obs, condition %s, %s, resonance %s; %s" % (
            o["kind"], o["name"], o["a"], o["e"], o["i"], o["q"], o["H"], o["arc"], o["nobs"], o["cc"], o["cls"], o["resonance"] or "-", o["vet"] or "unexplained"))
    json.dump(dict(n=len(es), hits=out), open(core.results("giant_region_circles.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_giant_region_circles", "Circles between the giant planets", "JPL Small-Body Database query API",
            "bodies with a %g-%g au, e below 0.3 and q beyond Jupiter: %d" % (a_min_au, a_max_au, len(es)), os.path.abspath(__file__), "giantcircles", len(es), "bodies",
            dict(a_au=[a_min_au, a_max_au], e_max=e_max, condition_max=condition_max), "the region's own eccentricities: median %.3f, 5th percentile %.3f" % (
                float(np.median(es)) if len(es) else 0, float(np.percentile(es, 5)) if len(es) else 0),
            "an orbit circular to %g between the giant planets on a year's arc" % e_max,
            "%d: %d on short or poor arcs, %d near a planet's resonance; %d unexplained" % (len(out), sum(o["vet"] == "unconstrained-geometry" for o in out),
                                                                                           sum(o["vet"] == "near-resonance" for o in out), len(left)),
            [dict(target="sbdb:%s" % o["pdes"], stat=o["e"], unit="eccentricity", vet=o["vet"],
                  note="%s %s, a %.3f au, i %.1f deg, q %.2f au, H %s, arc %s d, condition %s, %s, resonance %s" % (
                      o["kind"], o["name"], o["a"], o["i"], o["q"], o["H"], o["arc"], o["cc"], o["cls"], o["resonance"] or "-")) for o in out])
        print("recorded")


ecliptic_DOC = r"""Asteroids in the ecliptic to the hundredth of a degree: asteroids and comets of JPL's Small-Body Database with inclinations below 0.02 deg, orbits lying in Earth's own orbital plane to a part in ten thousand of a radian, a plane that nothing in the belt's dynamics singles out (the invariable plane lies 1.6 deg away) and a made orbit might; against the count the belt's own inclination density gives below 2 deg, where the density of orbits falls as the inclination itself, each read for its arc and condition code.

    python -I seti.py solar_system ecliptic [record] [--i-max-deg 0.02] [--band-deg 2] [--condition-max 6]"""


def ecliptic_main(record=False, i_max_deg=0.02, band_deg=2.0, condition_max=6):
    from scipy.stats import poisson
    rows = []
    for kind in ("a", "c"):
        rows += [dict(r, kind=kind) for r in sbdb_rows(["i|LT|%g" % band_deg, "e|LT|1"], "full_name,pdes,a,e,i,q,H,data_arc,n_obs_used,condition_code,class", kind=kind)]
    incs = np.array([num(r["i"]) for r in rows if num(r["i"]) is not None])
    hits = [r for r in rows if num(r["i"]) is not None and num(r["i"]) < i_max_deg]
    # a density rising as the inclination from zero puts the fraction (i_max / band)^2 of the band inside i_max
    exp = len(incs) * (i_max_deg / band_deg) ** 2
    # the band's own density, read from 0.02-0.2 deg, scaled to the window
    inner = int(((incs >= i_max_deg) & (incs < 10 * i_max_deg)).sum())
    exp2 = inner * (i_max_deg ** 2) / ((10 * i_max_deg) ** 2 - i_max_deg ** 2)
    p = float(poisson.sf(len(hits) - 1, exp2)) if hits else 1.0
    out = []
    for r in hits:
        cc = num(r["condition_code"])
        out.append(dict(kind="comet" if r["kind"] == "c" else "asteroid", name=" ".join(str(r["full_name"]).split()), pdes=r["pdes"], a=num(r["a"]), e=num(r["e"]), i=num(r["i"]),
                        q=num(r["q"]), H=num(r["H"]), arc=num(r["data_arc"]), nobs=num(r["n_obs_used"]), cc=cc, cls=r["class"],
                        vet=("unconstrained-geometry" if (cc is not None and cc >= condition_max) or (num(r["data_arc"]) or 0) < 30 else "chance-rate" if p > 0.01 else None)))
    print("%d bound orbits within %g deg of the ecliptic; within %g deg: %d (%.1f expected from a density rising as the inclination, %.1f from the band's own density at "
          "%g-%g deg; Poisson p %.3g)" % (len(incs), band_deg, i_max_deg, len(hits), exp, exp2, i_max_deg, 10 * i_max_deg, p))
    for o in sorted(out, key=lambda o: o["i"]):
        print("  %s %s: i %.4f deg, a %.3f au, e %.3f, H %s, arc %s d, %s obs, condition %s, %s; %s" % (
            o["kind"], o["name"], o["i"], o["a"], o["e"], o["H"], o["arc"], o["nobs"], o["cc"], o["cls"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(incs), hits=out), open(core.results("ecliptic_orbits.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_in_the_ecliptic", "Asteroids in the ecliptic to the hundredth of a degree", "JPL Small-Body Database query API",
            "bound orbits within %g deg of the ecliptic: %d" % (band_deg, len(incs)), os.path.abspath(__file__), "ecliptic", len(incs), "orbits",
            dict(i_max_deg=i_max_deg, band_deg=band_deg, condition_max=condition_max),
            "the band's own density at %g-%g deg gives %.1f in the window; a density rising as the inclination gives %.1f" % (i_max_deg, 10 * i_max_deg, exp2, exp),
            "an orbit within %g deg of the ecliptic that the catalogue holds" % i_max_deg, "%d against %.1f expected (Poisson p %.3g)" % (len(hits), exp2, p),
            [dict(target="sbdb:%s" % o["pdes"], stat=o["i"], unit="inclination, deg", vet=o["vet"],
                  note="%s %s, a %.3f au, e %.3f, H %s, arc %s d, condition %s, %s" % (o["kind"], o["name"], o["a"], o["e"], o["H"], o["arc"], o["cc"], o["cls"])) for o in out])
        print("recorded")


pushed_DOC = r"""Asteroids pushed harder than sunlight allows: Small-Body Database asteroids with a transverse non-gravitational term A2 at 3 sigma, scaled by their diameter (measured, or from H at albedo 0.14) against the Yarkovsky drift of Bennu, the lowest-density body with a measured drift and so near the ceiling sunlight sets at any size; a body whose lower bound exceeds twice Bennu's scaled term is being pushed by something other than re-radiated sunlight, which is outgassing from a dark comet or a thrust; each read against the published dark comets, the asteroids imaged in activity, the bodies of our own launching, and the arc its term rests on.

    python -I seti.py solar_system pushed [record] [--sigma 3] [--margin 2] [--albedo 0.14] [--arc-min-d 1000]"""

# Bennu's transverse term and diameter (Farnocchia et al. 2021), the calibration of the ceiling
PUSHED_BENNU_A2 = 4.618e-14
PUSHED_BENNU_D_KM = 0.49
# the dark comets (Seligman et al. 2023, 2024) and the bodies of our own launching in the database
PUSHED_DARK_COMETS = {"1998 KY26", "2003 RM", "2005 VL1", "2006 RH120", "2010 RF12", "2010 VL65", "2016 NJ33", "1998 FR11", "1998 SH2", "2001 ME1", "2005 UY6", "2012 UR158",
                      "2016 TA56"}
PUSHED_LAUNCHED = {"2020 SO"}
# asteroids whose activity has been imaged
PUSHED_ACTIVE = {"2008 GO98"}


def pushed_main(record=False, sigma=3.0, margin=2.0, albedo=0.14, arc_min_d=1000.0):
    rows = sbdb_rows(["A2|DF"], "full_name,pdes,A1,A1_sigma,A2,A2_sigma,A3,A3_sigma,H,diameter,albedo,e,a,q,i,data_arc,n_obs_used,condition_code,class", kind="a")
    out, n = [], 0
    for r in rows:
        a2, s2 = num(r["A2"]), num(r["A2_sigma"])
        if a2 is None or not s2:
            continue
        n += 1
        if abs(a2) / s2 < sigma:
            continue
        d = num(r["diameter"])
        h = num(r["H"])
        if d is None and h is not None:
            d = 1329.0 * 10 ** (-h / 5.0) / math.sqrt(num(r["albedo"]) or albedo)
        if not d:
            continue
        ceiling = PUSHED_BENNU_A2 * PUSHED_BENNU_D_KM / d
        low = (abs(a2) - sigma * s2) / ceiling
        if low < margin:
            continue
        pdes = str(r["pdes"]).strip()
        name = " ".join(str(r["full_name"]).split())
        prov = re.search(r"\(([^)]+)\)", name)
        prov = prov.group(1) if prov else name
        arc = num(r["data_arc"]) or 0
        a1, s1 = num(r["A1"]), num(r["A1_sigma"])
        out.append(dict(name=name, pdes=pdes, A2=a2, A2_sigma=s2, A1=a1, A1_sigma=s1, d_km=d, h=h, ratio=abs(a2) / ceiling, low=low, arc_d=arc, n_obs=num(r["n_obs_used"]),
                        cls=r["class"], a=num(r["a"]), e=num(r["e"]), q=num(r["q"]),
                        vet=("dark-comet" if prov in PUSHED_DARK_COMETS else "activity" if prov in PUSHED_ACTIVE else "our-launch" if prov in PUSHED_LAUNCHED else
                             "fit-degeneracy" if arc < arc_min_d else None)))
    left = [o for o in out if o["vet"] is None]
    print("%d asteroids with a transverse term; at %g sigma and more than %g times Bennu's term scaled to their size: %d (%d dark comets, %d active, %d of our launching, %d on arcs "
          "under %g d); unexplained %d" % (n, sigma, margin, len(out), sum(o["vet"] == "dark-comet" for o in out), sum(o["vet"] == "activity" for o in out),
                                           sum(o["vet"] == "our-launch" for o in out), sum(o["vet"] == "fit-degeneracy" for o in out), arc_min_d, len(left)))
    for o in sorted(out, key=lambda o: -o["low"]):
        print("  %s: A2 %.3e +- %.1e au/d^2 (A1 %s +- %s), D %.4f km, %.0f times Bennu's scaled term (lower bound %.0f), %s, arc %.0f d, %s obs, a %.3f e %.3f; %s" % (
            o["name"], o["A2"], o["A2_sigma"], o["A1"], o["A1_sigma"], o["d_km"], o["ratio"], o["low"], o["cls"], o["arc_d"], o["n_obs"], o["a"], o["e"], o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out), open(core.results("asteroids_pushed.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_pushed_asteroids", "Asteroids pushed harder than sunlight allows", "JPL Small-Body Database (sbdb_query.api)", "asteroids with a transverse non-gravitational term: %d" % n,
            os.path.abspath(__file__), "pushed", n, "asteroids", dict(sigma=sigma, margin=margin, albedo=albedo, arc_min_d=arc_min_d, bennu_a2=PUSHED_BENNU_A2, bennu_d_km=PUSHED_BENNU_D_KM),
            "Bennu's term scaled by diameter as the ceiling of re-radiated sunlight, with a margin of %g" % margin,
            "a transverse term at %g sigma whose lower bound exceeds the Yarkovsky ceiling for the body's size" % sigma,
            "%d: %d dark comets, %d active, %d of our own launching, %d on arcs under %g d; %d unexplained" % (
                len(out), sum(o["vet"] == "dark-comet" for o in out), sum(o["vet"] == "activity" for o in out), sum(o["vet"] == "our-launch" for o in out),
                sum(o["vet"] == "fit-degeneracy" for o in out), arc_min_d, len(left)),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["ratio"], 1), unit="times Bennu's term scaled to the diameter", vet=o["vet"],
                  note="A2 %.3e +- %.1e au/d^2, D %.4f km, %s, arc %.0f d, %s obs" % (o["A2"], o["A2_sigma"], o["d_km"], o["cls"], o["arc_d"], o["n_obs"])) for o in out])
        print("recorded")


reflspike_DOC = r"""Emission lines in asteroid reflectance: Gaia DR3's mean reflectance spectra of asteroids (sso_reflectance_spectrum, sixteen bands from 374 to 1034 nm normalised at 550 nm), bands whose reflectance stands more than 0.3 and five combined errors above the mean of their two neighbours, a line of light the surface adds to the sunlight it returns, which no mineral does and a laser or lamp would; the interior bands alone, since the two at each end carry the instrument's own fall-off and are often flagged, with each spike read for the flags on its band and the number of spectra behind it.

    python -I seti.py solar_system reflspike [record] [--jump 0.3] [--sigma 5] [--edge-bands 2] [--floor 1.5]"""


def reflspike_main(record=False, jump=0.3, sigma=5.0, edge_bands=2, floor=1.5):
    hot = core.tap_async("gaia", "SELECT DISTINCT number_mp FROM gaiadr3.sso_reflectance_spectrum WHERE reflectance_spectrum > %g AND reflectance_spectrum_flag = 0" % floor, maxrec=100000)
    ids = sorted({int(r["number_mp"]) for r in hot})
    spectra = {}
    for k in range(0, len(ids), 500):
        for r in core.tap_async("gaia", "SELECT number_mp, denomination, num_of_spectra, wavelength, reflectance_spectrum, reflectance_spectrum_err, reflectance_spectrum_flag FROM "
                                        "gaiadr3.sso_reflectance_spectrum WHERE number_mp IN (%s)" % ",".join(str(i) for i in ids[k:k + 500]), maxrec=100000):
            spectra.setdefault(int(r["number_mp"]), []).append(r)
    n_all = core.tap("gaia", "SELECT COUNT(DISTINCT number_mp) AS n FROM gaiadr3.sso_reflectance_spectrum")[0]["n"]
    out = []
    for mp, rows in spectra.items():
        rows.sort(key=lambda r: float(r["wavelength"]))
        w = [float(r["wavelength"]) for r in rows]
        f = [num(r["reflectance_spectrum"]) for r in rows]
        e = [num(r["reflectance_spectrum_err"]) for r in rows]
        fl = [int(num(r["reflectance_spectrum_flag"]) or 0) for r in rows]
        for i in range(edge_bands, len(rows) - edge_bands):
            if None in (f[i], f[i - 1], f[i + 1], e[i], e[i - 1], e[i + 1]):
                continue
            base = 0.5 * (f[i - 1] + f[i + 1])
            err = math.sqrt(e[i] ** 2 + 0.25 * (e[i - 1] ** 2 + e[i + 1] ** 2))
            if f[i] - base > jump and (f[i] - base) / err > sigma:
                out.append(dict(mp=mp, name=" ".join(str(rows[0]["denomination"]).split()), nm=w[i], refl=f[i], base=base, err=err, z=(f[i] - base) / err, flags=(fl[i - 1], fl[i], fl[i + 1]),
                                n_spec=int(num(rows[0]["num_of_spectra"]) or 0), spectrum=[round(x, 3) if x is not None else None for x in f],
                                vet="artefact" if fl[i] or fl[i - 1] or fl[i + 1] else "single-record" if int(num(rows[0]["num_of_spectra"]) or 0) < 2 else None))
    left = [o for o in out if o["vet"] is None]
    print("%s asteroids with reflectance spectra, %d with an unflagged interior band above %g; bands %g and %g sigma above their neighbours: %d (%d on flagged bands, %d from a single "
          "spectrum); unexplained %d" % (n_all, len(spectra), floor, jump, sigma, len(out), sum(o["vet"] == "artefact" for o in out), sum(o["vet"] == "single-record" for o in out),
                                         len(left)))
    for o in sorted(out, key=lambda o: -o["z"]):
        print("  (%d) %s: %.0f nm at %.2f against %.2f beside it (%.1f sigma), flags %s, %d spectra: %s; %s" % (
            o["mp"], o["name"], o["nm"], o["refl"], o["base"], o["z"], o["flags"], o["n_spec"], o["spectrum"], o["vet"] or "unexplained"))
    json.dump(dict(n=n_all, hot=len(spectra), hits=out), open(core.results("asteroid_reflectance_spikes.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_sso_reflectance_spikes", "Emission lines in asteroid reflectance", "Gaia DR3 sso_reflectance_spectrum",
            "asteroids with reflectance spectra: %s, %d with an unflagged interior band above %g" % (n_all, len(spectra), floor), os.path.abspath(__file__), "reflspike", int(n_all),
            "asteroids", dict(jump=jump, sigma=sigma, edge_bands=edge_bands, floor=floor), "the bands' own flags and the count of spectra behind each mean",
            "an interior band more than %g and %g sigma above the mean of its neighbours" % (jump, sigma),
            "%d: %d on flagged bands, %d from a single spectrum; %d unexplained" % (len(out), sum(o["vet"] == "artefact" for o in out), sum(o["vet"] == "single-record" for o in out),
                                                                                  len(left)),
            [dict(target="asteroid:%d" % o["mp"], stat=round(o["z"], 1), unit="sigma above the neighbouring bands", vet=o["vet"],
                  note="%s, %.0f nm at %.2f against %.2f, flags %s, %d spectra" % (o["name"], o["nm"], o["refl"], o["base"], o["flags"], o["n_spec"])) for o in out])
        print("recorded")


hyperbolic_DOC = r"""Hyperbolic excess beyond planetary perturbation: Small-Body Database asteroids and comets with eccentricity above 1, their speed at infinity from the eccentricity's 3 sigma lower bound and the perihelion distance, those arriving from beyond the Sun's hold at more than 4 km/s, above the 3.9 km/s that C/1980 E1's Jupiter encounter gave it, the most any recorded planetary encounter has imparted, which the interstellar objects and a visitor on its own course show; each read against the published interstellar objects and the recorded planetary encounters, with the fragments of split comets and short arcs named.

    python -I seti.py solar_system hyperbolic [record] [--vinf-min-kms 4] [--sigma 3] [--arc-min-d 60]"""

# the published interstellar objects and the comets whose hyperbolic orbits a recorded planetary encounter made
HYPERBOLIC_INTERSTELLAR = {"A/2017 U1", "C/2019 Q4", "C/2025 N1", "1I/", "2I/", "3I/"}
HYPERBOLIC_ENCOUNTERS = {"C/1980 E1": "Jupiter, 1980", "C/2024 L5": "Saturn, 2022"}
HYPERBOLIC_GM = 2.9591220828e-4


def hyperbolic_main(record=False, vinf_min_kms=4.0, sigma=3.0, arc_min_d=60.0):
    out, n = [], {}
    for kind in ("a", "c"):
        rows = sbdb_rows(["e|GT|1"], "full_name,pdes,e,sigma_e,q,sigma_q,i,data_arc,n_obs_used,condition_code,class", kind=kind)
        n[kind] = len(rows)
        for r in rows:
            e, se, q = num(r["e"]), num(r["sigma_e"]), num(r["q"])
            if e is None or q is None or q <= 0:
                continue
            se = se or 0.0
            e_low = e - sigma * se
            vinf = math.sqrt(HYPERBOLIC_GM * (e - 1) / q) * 1731.456837 if e > 1 else 0.0
            v_low = math.sqrt(HYPERBOLIC_GM * (e_low - 1) / q) * 1731.456837 if e_low > 1 else 0.0
            if v_low < vinf_min_kms:
                continue
            name = " ".join(str(r["full_name"]).split())
            key = next((k for k in HYPERBOLIC_INTERSTELLAR if k in name), None)
            enc = next((v for k, v in HYPERBOLIC_ENCOUNTERS.items() if k in name), None)
            frag = re.search(r"-[A-Z]", name.split(" (")[0]) is not None
            arc = num(r["data_arc"]) or 0
            out.append(dict(kind="asteroid" if kind == "a" else "comet", name=name, pdes=str(r["pdes"]).strip(), e=e, sigma_e=se, q=q, i=num(r["i"]), vinf=vinf, v_low=v_low,
                            arc=arc, n_obs=num(r["n_obs_used"]), cls=r["class"], encounter=enc,
                            vet=("natural-source" if key else "planetary-encounter" if enc else "activity" if frag else "fit-degeneracy" if not se or arc < arc_min_d else None)))
    left = [o for o in out if o["vet"] is None]
    print("%d asteroids and %d comets with eccentricity above 1; speed at infinity above %g km/s at the eccentricity's %g sigma lower bound: %d (%d published interstellar objects, "
          "%d from recorded planetary encounters, %d fragments of split comets, %d with no eccentricity error or an arc under %g d); unexplained %d" % (
              n["a"], n["c"], vinf_min_kms, sigma, len(out), sum(o["vet"] == "natural-source" for o in out), sum(o["vet"] == "planetary-encounter" for o in out),
              sum(o["vet"] == "activity" for o in out), sum(o["vet"] == "fit-degeneracy" for o in out), arc_min_d, len(left)))
    for o in sorted(out, key=lambda o: -o["v_low"]):
        print("  %s (%s, %s): e %.5f +- %.5f, q %.3f au, i %.1f, v_inf %.2f km/s (lower bound %.2f), arc %s d, %s obs%s; %s" % (
            o["name"], o["kind"], o["cls"], o["e"], o["sigma_e"], o["q"], o["i"] or 0, o["vinf"], o["v_low"], o["arc"], o["n_obs"],
            ", encounter " + o["encounter"] if o["encounter"] else "", o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out), open(core.results("hyperbolic_excess.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_hyperbolic_excess", "Hyperbolic excess beyond planetary perturbation", "JPL Small-Body Database (sbdb_query.api)",
            "asteroids with e above 1: %d; comets: %d" % (n["a"], n["c"]), os.path.abspath(__file__), "hyperbolic", n["a"] + n["c"], "bodies",
            dict(vinf_min_kms=vinf_min_kms, sigma=sigma, arc_min_d=arc_min_d), "the eccentricities' own errors; the published interstellar objects and planetary encounters",
            "a speed at infinity above %g km/s at the eccentricity's %g sigma lower bound" % (vinf_min_kms, sigma),
            "%d: %d interstellar objects, %d from recorded encounters, %d fragments, %d without an eccentricity error or on a short arc; %d unexplained" % (
                len(out), sum(o["vet"] == "natural-source" for o in out), sum(o["vet"] == "planetary-encounter" for o in out), sum(o["vet"] == "activity" for o in out),
                sum(o["vet"] == "fit-degeneracy" for o in out), len(left)),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["v_low"], 2), unit="km/s at infinity, lower bound", vet=o["vet"],
                  note="%s, e %.5f +- %.5f, q %.3f au, v_inf %.2f km/s, arc %s d%s" % (o["kind"], o["e"], o["sigma_e"], o["q"], o["vinf"], o["arc"],
                                                                                     ", encounter " + o["encounter"] if o["encounter"] else "")) for o in out])
        print("recorded")


blackest_DOC = r"""Asteroids blacker than any surface: Small-Body Database asteroids whose geometric albedo lies below 0.02, darker than the darkest comet nuclei and primitive asteroids, the mark of a surface made to swallow light, read against AKARI's albedos (J/PASJ/63/1117) with their errors; a radiometric albedo rests on the diameter and the absolute magnitude together, so an albedo far below the floor is an absolute magnitude too faint or a diameter too large.

    python -I seti.py solar_system blackest [record] [--albedo 0.02] [--sigma 3]"""


def blackest_main(record=False, albedo=0.02, sigma=3.0):
    rows = sbdb_rows(["albedo|LT|%g" % albedo], "full_name,pdes,H,albedo,diameter,diameter_sigma,class,spec_T,spec_B,n_obs_used,data_arc", kind="a")
    nums = [str(r["pdes"]).strip() for r in rows if str(r["pdes"]).strip().isdigit()]
    akari = {}
    for k in range(0, len(nums), 500):
        for a in core.tap("vizier", 'SELECT Num, Alb, e_Alb, Diam FROM "J/PASJ/63/1117/acua_v1" WHERE Num IN (%s)' % ",".join(nums[k:k + 500])):
            akari[str(a["Num"]).strip()] = (num(a["Alb"]), num(a["e_Alb"]), num(a["Diam"]))
    out = []
    for r in rows:
        pdes = str(r["pdes"]).strip()
        ak = akari.get(pdes)
        firm = ak is not None and ak[0] is not None and ak[1] is not None and ak[0] + sigma * ak[1] < albedo
        out.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=pdes, h=num(r["H"]), albedo=num(r["albedo"]), d=num(r["diameter"]), ds=num(r["diameter_sigma"]),
                        cls=r["class"], spec=(str(r["spec_T"] or "").strip() or str(r["spec_B"] or "").strip()), akari=ak, firm=firm,
                        vet=None if firm else "fit-degeneracy"))
    firm = [o for o in out if o["firm"]]
    print("%d asteroids with an albedo below %g in the database, %d of them with an AKARI albedo, %d below %g at AKARI's %g sigma upper bound; unexplained %d" % (
        len(out), albedo, sum(o["akari"] is not None for o in out), len(firm), albedo, sigma, len(firm)))
    for o in sorted(out, key=lambda o: o["albedo"])[:25]:
        print("  %s: albedo %.4f, D %s +- %s km, H %s, %s%s, AKARI %s; %s" % (o["name"], o["albedo"], o["d"], o["ds"], o["h"], o["cls"], " " + o["spec"] if o["spec"] else "", o["akari"],
                                                                          o["vet"] or "unexplained"))
    for o in firm:
        print("  AKARI-firm %s: albedo %.4f, AKARI %.4f +- %.4f, D %s km; unexplained" % (o["name"], o["albedo"], o["akari"][0], o["akari"][1], o["akari"][2]))
    json.dump(out, open(core.results("asteroids_blackest.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_blackest_asteroids", "Asteroids blacker than any surface", "JPL Small-Body Database (sbdb_query.api); AKARI asteroid catalogue (VizieR J/PASJ/63/1117)",
            "asteroids with an albedo below %g: %d" % (albedo, len(out)), os.path.abspath(__file__), "blackest", len(out), "asteroids", dict(albedo=albedo, sigma=sigma),
            "AKARI's albedos with their errors; the radiometric albedo's dependence on H and the diameter", "an albedo below %g at %g sigma in an independent measurement" % (albedo, sigma),
            "%d below %g in the database, %d with AKARI albedos, %d of them below %g at %g sigma; %d unexplained" % (len(out), albedo, sum(o["akari"] is not None for o in out), len(firm),
                                                                                                                albedo, sigma, len(firm)),
            [dict(target="sbdb:%s" % o["pdes"], stat=o["albedo"], unit="geometric albedo", vet=o["vet"],
                  note="D %s +- %s km, H %s, %s%s, AKARI %s" % (o["d"], o["ds"], o["h"], o["cls"], " " + o["spec"] if o["spec"] else "", o["akari"])) for o in out])
        print("recorded")


copycomets_DOC = r"""Copy comets: Small-Body Database comets, pairs whose orbits agree under the Southworth-Hawkins criterion to 0.05, two bodies on one orbit, against the pairs' sharing of a parent designation (the fragments of a split comet), the sungrazing families whose perihelia lie inside 0.1 au and the published pairs and families of one parent, with the rest read for their arcs; a copy on an orbit no family claims is a second body placed on a known comet's path.

    python -I seti.py solar_system copycomets [record] [--d-max 0.05] [--family-q 0.1] [--arc-min-d 30]"""


def copycomets_sh(q1, e1, i1, o1, w1, q2, e2, i2, o2, w2):
    """Southworth-Hawkins D between two orbits (perihelion distance in au, angles in radians)."""
    I = 2 * math.asin(min(1.0, math.sqrt(math.sin((i1 - i2) / 2) ** 2 + math.sin(i1) * math.sin(i2) * math.sin((o1 - o2) / 2) ** 2)))
    sign = 1 if abs(o1 - o2) <= math.pi else -1
    pi1, pi2 = w1 + o1, w2 + o2
    th = 2 * math.asin(min(1.0, math.cos((i1 + i2) / 2) * abs(math.sin((o1 - o2) / 2)) / max(math.cos(I / 2), 1e-9)))
    P = (pi1 - pi2) + sign * th
    return math.sqrt((e1 - e2) ** 2 + (q1 - q2) ** 2 + (2 * math.sin(I / 2)) ** 2 + (((e1 + e2) / 2) * 2 * math.sin(P / 2)) ** 2)


# the published pairs and families of comets split from one parent, by designation
COPYCOMETS_FAMILIES = [{"C/1988 F1", "C/1988 J1"}, {"C/1988 A1", "C/1996 Q1", "C/2015 F3", "C/2019 Y1", "C/2023 V5"}, {"C/1844 Y1", "C/2019 Y4"}, {"C/2002 A1", "C/2002 A2"},
                       {"C/2002 Q2", "C/2002 Q3"}, {"169P", "P/2003 T12"}, {"C/1846 O1", "C/1973 D1"}, {"332P", "412P"}, {"C/2015 ER61", "C/2015 E61"}]


def copycomets_family(a, b):
    """Whether two parent designations fall in one published comet family."""
    return any(a in f and b in f for f in COPYCOMETS_FAMILIES)


def copycomets_parent(name):
    """A comet's designation with any fragment letter removed."""
    base = name.split(" (")[0].strip()
    base = re.sub(r"-[A-Z]+\d*$", "", base)
    return re.sub(r"^(\d+P)(/.*)?$", r"\1", base)


def copycomets_main(record=False, d_max=0.05, family_q=0.1, arc_min_d=30.0):
    rows = sbdb_rows(["q|DF", "e|DF", "i|DF", "om|DF", "w|DF"], "full_name,pdes,prefix,q,e,i,om,w,per_y,data_arc,n_obs_used", kind="c")
    cs = []
    for r in rows:
        q, e, i, o, w = (num(r[k]) for k in ("q", "e", "i", "om", "w"))
        if None in (q, e, i, o, w):
            continue
        cs.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), q=q, e=min(e, 1.0), i=math.radians(i), om=math.radians(o), w=math.radians(w),
                       arc=num(r["data_arc"]) or 0, n_obs=num(r["n_obs_used"]), per=num(r["per_y"])))
    q = np.array([c["q"] for c in cs])
    e = np.array([c["e"] for c in cs])
    inc = np.array([c["i"] for c in cs])
    near = (np.abs(q[:, None] - q[None, :]) < d_max) & (np.abs(e[:, None] - e[None, :]) < d_max) & (np.abs(inc[:, None] - inc[None, :]) < 2 * d_max)
    pairs = []
    for a, b in np.argwhere(np.triu(near, 1)):
        x, y = cs[a], cs[b]
        d = copycomets_sh(x["q"], x["e"], x["i"], x["om"], x["w"], y["q"], y["e"], y["i"], y["om"], y["w"])
        if d < d_max:
            pa, pb = copycomets_parent(x["name"]), copycomets_parent(y["name"])
            vet = ("activity" if pa == pb else "comet-family" if min(x["q"], y["q"]) < family_q or copycomets_family(pa, pb) else
                   "fit-degeneracy" if min(x["arc"], y["arc"]) < arc_min_d else None)
            pairs.append(dict(a=x["name"], b=y["name"], d=d, q=(x["q"], y["q"]), e=(x["e"], y["e"]), i=(math.degrees(x["i"]), math.degrees(y["i"])), arcs=(x["arc"], y["arc"]), vet=vet))
    left = [p for p in pairs if p["vet"] is None]
    print("%d comets with orbits; pairs with D below %g: %d (%d fragments of one comet, %d in the sungrazing families inside %g au or the published families, %d resting on an arc "
          "under %g d); unexplained %d" % (len(cs), d_max, len(pairs), sum(p["vet"] == "activity" for p in pairs), sum(p["vet"] == "comet-family" for p in pairs), family_q,
                                           sum(p["vet"] == "fit-degeneracy" for p in pairs), arc_min_d, len(left)))
    for p in sorted(pairs, key=lambda p: (p["vet"] is not None, p["d"])):
        if p["vet"] is None or (p["vet"] == "comet-family" and min(p["q"]) >= family_q) or p["vet"] == "fit-degeneracy":
            print("  %s and %s: D %.3f, q %.3f/%.3f, e %.3f/%.3f, i %.1f/%.1f, arcs %.0f/%.0f d; %s" % (
                p["a"], p["b"], p["d"], p["q"][0], p["q"][1], p["e"][0], p["e"][1], p["i"][0], p["i"][1], p["arcs"][0], p["arcs"][1], p["vet"] or "unexplained"))
    json.dump(dict(n=len(cs), pairs=pairs), open(core.results("comet_copies.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_copy_comets", "Copy comets", "JPL Small-Body Database (sbdb_query.api)", "comets with orbital elements: %d" % len(cs), os.path.abspath(__file__), "copycomets",
            len(cs), "comets", dict(d_max=d_max, family_q=family_q, arc_min_d=arc_min_d), "the pairs' parent designations and the sungrazing families",
            "two comets on one orbit under the Southworth-Hawkins criterion at %g" % d_max,
            "%d pairs: %d fragments, %d in sungrazing families, %d on short arcs; %d unexplained" % (
                len(pairs), sum(p["vet"] == "activity" for p in pairs), sum(p["vet"] == "comet-family" for p in pairs), sum(p["vet"] == "fit-degeneracy" for p in pairs), len(left)),
            [dict(target="comet:%s" % p["a"], stat=round(p["d"], 4), unit="Southworth-Hawkins D to its copy", vet=p["vet"],
                  note="with %s; q %.3f/%.3f, e %.3f/%.3f, i %.1f/%.1f, arcs %.0f/%.0f d" % (p["b"], p["q"][0], p["q"][1], p["e"][0], p["e"][1], p["i"][0], p["i"][1], p["arcs"][0],
                                                                                        p["arcs"][1])) for p in pairs if p["vet"] != "comet-family" or min(p["q"]) >= family_q])
        print("recorded")


jupl12_DOC = r"""Bodies held at Jupiter's L1 and L2 points: asteroids and comets of JPL's Small-Body Database on bound orbits reaching from inside 5.9 au to beyond 4.6 au, each placed at its element epoch by its own elements against Jupiter's place at that epoch (the ERFA ephemeris astropy carries), those within R_AU of the L1 or L2 point, 0.0683 of Jupiter's distance from Jupiter toward and away from the Sun, where nothing stays without station-keeping and a craft watching Jupiter would sit, counted against the same volumes at the points' distances 120 deg round from Jupiter either way, clear of the Trojan clouds and the Hildas' aphelia; each body found there followed across its observed arc in JPL Horizons every STEP_DAYS, a body within R_AU of its point for HELD of an arc of a year or more held there, and one that passes read for an entry into Jupiter's Hill sphere on the arc, the way a comet is captured by Jupiter for a while and leaves, by L1 or L2; an orbit on an arc under 30 days or of condition code 6 or more is placed and left unfollowed.

    python -I seti.py solar_system jupl12 [record] [--r-au 0.15] [--step-days 30] [--held 0.5]"""

JUPL12_H = 0.0683  # (mu / 3)^(1/3) for Jupiter: the L1 and L2 points' distance from Jupiter, and its Hill radius, as a fraction of its distance from the Sun


def jupl12_place(o):
    """A bound orbit's heliocentric J2000 ecliptic position (au) at its element epoch from its elements."""
    a, e, M = o["a"], o["e"], math.radians(o["ma"] % 360.0)
    E = M if e < 0.8 else math.pi
    for _ in range(60):
        dE = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-13:
            break
    xp, yp = a * (math.cos(E) - e), a * math.sqrt(1 - e * e) * math.sin(E)
    i, O, w = (math.radians(o[k]) for k in ("i", "om", "w"))
    P = (math.cos(O) * math.cos(w) - math.sin(O) * math.sin(w) * math.cos(i), math.sin(O) * math.cos(w) + math.cos(O) * math.sin(w) * math.cos(i), math.sin(w) * math.sin(i))
    Q = (-math.cos(O) * math.sin(w) - math.sin(O) * math.cos(w) * math.cos(i), -math.sin(O) * math.sin(w) + math.cos(O) * math.cos(w) * math.cos(i),
         math.cos(w) * math.sin(i))
    return np.array([xp * P[k] + yp * Q[k] for k in range(3)])


def jupl12_jupiter(jds):
    """Jupiter's heliocentric J2000 ecliptic positions (au) at Julian dates JDS (TDB) from the ERFA ephemeris astropy carries."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(np.atleast_1d(jds), format="jd", scale="tdb")
    x, y, z = get_body_barycentric("jupiter", t).xyz.to("AU").value - get_body_barycentric("sun", t).xyz.to("AU").value
    eps = math.radians(23.4392911)
    return np.stack([x, y * math.cos(eps) + z * math.sin(eps), -y * math.sin(eps) + z * math.cos(eps)], axis=1)


def jupl12_points(rj, deg=0.0):
    """The L1 and L2 points of Jupiter at heliocentric position RJ, turned DEG about the ecliptic pole."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    r = np.array([c * rj[0] - s * rj[1], s * rj[0] + c * rj[1], rj[2]])
    return r * (1 - JUPL12_H), r * (1 + JUPL12_H)


def jupl12_horizons(command, start, stop, step_days):
    """Heliocentric J2000 ecliptic positions (au) of a Horizons body between two Julian dates every STEP_DAYS, as (Julian date, position) pairs."""
    q = {"format": "json", "COMMAND": "'%s'" % command, "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@10'", "REF_PLANE": "'ECLIPTIC'", "START_TIME": "'JD%.5f'" % start,
         "STOP_TIME": "'JD%.5f'" % stop, "STEP_SIZE": "'%d d'" % step_days, "VEC_TABLE": "'1'", "CSV_FORMAT": "'YES'", "OUT_UNITS": "'AU-D'", "OBJ_DATA": "'NO'"}
    txt = json.loads(core.get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), timeout=180))["result"]
    if "$$SOE" not in txt:
        return []
    out = []
    for line in txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        c = [x.strip() for x in line.split(",")]
        out.append((round(float(c[0]), 4), np.array([float(c[2]), float(c[3]), float(c[4])])))
    return out


def jupl12_main(record=False, r_au=0.15, step_days=30.0, held=0.5):
    from astropy.time import Time
    from scipy.stats import poisson
    fields = "full_name,pdes,spkid,a,e,i,om,w,ma,tp,epoch,H,data_arc,n_obs_used,condition_code,class,first_obs,last_obs"
    rows = []
    for kind in ("a", "c"):
        rows += [dict(r, kind=kind) for r in sbdb_rows(["q|LT|5.9", "ad|GT|4.6"], fields, kind=kind)]
    bodies, unplaced = [], 0
    for r in rows:
        o = {k: num(r[k]) for k in ("a", "e", "i", "om", "w", "ma", "tp", "epoch", "H", "data_arc", "n_obs_used", "condition_code")}
        if None in (o["a"], o["e"], o["i"], o["om"], o["w"], o["epoch"]) or not 0 <= o["e"] < 1:
            continue
        if o["ma"] is None and o["tp"] is not None:
            o["ma"] = 0.9856076686 / o["a"] ** 1.5 * (o["epoch"] - o["tp"])
        if o["ma"] is None:
            continue
        o["poor"] = (o["data_arc"] or 0) < 30 or (o["condition_code"] or 0) >= 6
        unplaced += o["poor"]
        o.update(name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), spkid=str(r["spkid"]).strip(), cls=str(r["class"]).strip(), kind=r["kind"],
                 first=str(r["first_obs"] or "").strip(), last=str(r["last_obs"] or "").strip())
        bodies.append(o)
    epochs = sorted({o["epoch"] for o in bodies})
    jup = dict(zip(epochs, jupl12_jupiter(epochs)))
    near, ctrl, phase = [], {120: 0, 240: 0}, []
    for o in bodies:
        r, rj = jupl12_place(o), jup[o["epoch"]]
        phase.append((math.degrees(math.atan2(r[1], r[0]) - math.atan2(rj[1], rj[0])) + 180.0) % 360.0 - 180.0)
        l1, l2 = jupl12_points(rj)
        d1, d2 = float(np.linalg.norm(r - l1)), float(np.linalg.norm(r - l2))
        for deg in ctrl:
            c1, c2 = jupl12_points(rj, deg)
            ctrl[deg] += int(np.linalg.norm(r - c1) < r_au) + int(np.linalg.norm(r - c2) < r_au)
        if min(d1, d2) < r_au:
            o.update(point="L1" if d1 < d2 else "L2", d=min(d1, d2), dj=float(np.linalg.norm(r - rj)) / (JUPL12_H * float(np.linalg.norm(rj))))
            near.append(o)
    exp = sum(ctrl.values()) / float(len(ctrl))
    # the Trojan clouds at 60 deg either side of Jupiter, as the placement's own check
    ph = np.array(phase)
    hist = [int(np.sum((ph >= lo) & (ph < lo + 30))) for lo in range(-180, 180, 30)]
    p = float(poisson.sf(len(near) - 1, exp)) if near else 1.0
    for o in near:
        if o["poor"]:
            o.update(steps=0, share=None, hill_min=None, vet="unconstrained-geometry")
            continue
        t0, t1 = Time(o["first"], scale="utc").jd, Time(o["last"], scale="utc").jd
        t1 = max(t1, t0 + step_days)
        track = jupl12_horizons(o["spkid"], t0, t1, step_days)
        jv = dict(jupl12_horizons("599", t0, t1, step_days))
        dpt, dj = [], []
        for jd, rb in track:
            if jd not in jv:
                continue
            l1, l2 = jupl12_points(jv[jd])
            dpt.append(float(np.linalg.norm(rb - (l1 if o["point"] == "L1" else l2))))
            dj.append(float(np.linalg.norm(rb - jv[jd])) / (JUPL12_H * float(np.linalg.norm(jv[jd]))))
        o["steps"] = len(dpt)
        o["share"] = sum(d < r_au for d in dpt) / float(len(dpt)) if dpt else None
        o["hill_min"] = min(dj) if dj else None
        o["vet"] = ("unconstrained-geometry" if not dpt or t1 - t0 < 365.25 else
                    None if o["share"] >= held else
                    "planetary-encounter" if o["hill_min"] < 1.0 else "chance-rate")
    left = [o for o in near if o["vet"] is None]
    count = lambda k: sum(o["vet"] == k for o in near)
    print("Small-Body Database bound orbits reaching 4.6-5.9 au: %d placed (%d of them on arcs under 30 days or of condition code 6 or more); heliocentric longitude from Jupiter in 30 deg bins from -180: %s; within %g au of Jupiter's L1 or L2 point at their epoch: %d "
          "(L1 %d, L2 %d) against %.1f in the same volumes 120 deg either side (%s; Poisson p %.3g); followed in Horizons: held for %g of a year's arc or more %d, entering "
          "Jupiter's Hill sphere %d, passing %d, arcs under a year or poor orbits %d" % (
              len(bodies), unplaced, hist, r_au, len(near), sum(o["point"] == "L1" for o in near), sum(o["point"] == "L2" for o in near), exp, ctrl, p, held, len(left),
              count("planetary-encounter"), count("chance-rate"), count("unconstrained-geometry")))
    for o in sorted(near, key=lambda o: (o["vet"] is not None, o["d"])):
        print("  %s (%s): %.3f au from %s, %.2f Hill radii from Jupiter at epoch; a %.3f, e %.3f, i %.1f, arc %s to %s; %s steps, %s of them within %g au, least %s Hill radii; %s" % (
            o["name"], o["cls"], o["d"], o["point"], o["dj"], o["a"], o["e"], o["i"], o["first"], o["last"], o["steps"],
            "%.2f" % o["share"] if o["share"] is not None else "-", r_au, "%.2f" % o["hill_min"] if o["hill_min"] is not None else "-", o["vet"] or "unexplained"))
    json.dump(dict(n=len(bodies), ctrl=ctrl, hist=hist, near=near), open(core.results("jupiter_l1_l2.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_jupiter_l1_l2", "Bodies held at Jupiter's L1 and L2 points", "JPL Small-Body Database (sbdb_query.api); JPL Horizons",
            "bound orbits reaching 4.6-5.9 au: %d" % len(bodies), os.path.abspath(__file__), "jupl12", len(bodies), "bodies",
            dict(r_au=r_au, step_days=step_days, held=held), "the same volumes 120 deg either side of Jupiter: %.1f bodies; longitude from Jupiter in 30 deg bins: %s" % (exp, hist),
            "a body within %g au of L1 or L2 for %g of an observed arc of a year or more" % (r_au, held),
            "%d within %g au of L1 or L2 at epoch against %.1f (Poisson p %.3g): %d held, %d entering Jupiter's Hill sphere, %d passing, %d on arcs under a year or poor orbits" % (
                len(near), r_au, exp, p, len(left), count("planetary-encounter"), count("chance-rate"), count("unconstrained-geometry")),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["share"], 3) if o["share"] is not None else None, unit="share of the observed arc within %g au of %s" % (r_au, o["point"]),
                  vet=o["vet"], note="%s (%s), %.3f au from %s at epoch; a %.3f, e %.3f, i %.1f; arc %s to %s; least %s Hill radii from Jupiter" % (
                      o["name"], o["cls"], o["d"], o["point"], o["a"], o["e"], o["i"], o["first"], o["last"],
                      "%.2f" % o["hill_min"] if o["hill_min"] is not None else "-")) for o in near])
        print("recorded")


jupl3_DOC = r"""Lurkers at Jupiter's far Lagrange point: Small-Body Database bodies with semi-major axes within 0.35 au of Jupiter's and eccentricities below 0.3, their mean longitudes at their element epochs against Jupiter's, the count within 20 deg of the L3 point opposite Jupiter, where nothing natural stays and a watcher of the planet could be parked, against the Trojan swarms at L4 and L5 and the passing bodies between; each body near L3 read for its class, since a Trojan of wide libration reaches far from its point, and for its arc.

    python -I seti.py solar_system jupl3 [record] [--half-width-deg 20] [--a-tol 0.35] [--e-max 0.3]"""

JUPL3_A = 5.2026


def jupl3_jupiter_longitude(jd):
    """Jupiter's mean heliocentric longitude in degrees at Julian date JD from its mean elements."""
    return (34.39644051 + 3034.74612775 * (jd - 2451545.0) / 36525.0) % 360.0


def jupl3_main(record=False, half_width_deg=20.0, a_tol=0.35, e_max=0.3):
    rows = sbdb_rows(["a|RG|%g|%g" % (JUPL3_A - a_tol, JUPL3_A + a_tol), "e|LT|%g" % e_max], "full_name,pdes,a,e,i,om,w,ma,epoch,H,data_arc,n_obs_used,class", kind="a")
    bodies = []
    for r in rows:
        om, w, ma, ep = num(r["om"]), num(r["w"]), num(r["ma"]), num(r["epoch"])
        if None in (om, w, ma, ep):
            continue
        lam = (om + w + ma) % 360.0
        phase = (lam - jupl3_jupiter_longitude(ep)) % 360.0
        bodies.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), a=num(r["a"]), e=num(r["e"]), i=num(r["i"]), phase=phase, H=num(r["H"]),
                           arc=num(r["data_arc"]) or 0, n_obs=num(r["n_obs_used"]), cls=str(r["class"]).strip()))
    ph = np.array([b["phase"] for b in bodies])
    bins = {c: int(np.sum(np.abs((ph - c + 180) % 360 - 180) < half_width_deg)) for c in (0, 60, 120, 180, 240, 300)}
    l3 = [b for b in bodies if abs((b["phase"] - 180 + 180) % 360 - 180) < half_width_deg]
    for b in l3:
        b["vet"] = "fit-degeneracy" if b["arc"] < 365 else "co-orbital" if b["cls"] == "TJN" else None
    left = [b for b in l3 if b["vet"] is None]
    print("%d bodies within %g au of Jupiter's semi-major axis with e below %g; within %g deg of the phases 0, 60, 120, 180, 240 and 300 deg from Jupiter: %s; at L3: %d (%d on arcs "
          "under a year, %d Trojans librating widely); unexplained %d" % (len(bodies), a_tol, e_max, half_width_deg, bins, len(l3), sum(b["vet"] == "fit-degeneracy" for b in l3),
                                                                        sum(b["vet"] == "co-orbital" for b in l3), len(left)))
    for b in sorted(l3, key=lambda b: abs(b["phase"] - 180)):
        print("  %s (%s): %.1f deg from Jupiter, a %.3f, e %.3f, i %.1f, H %s, arc %.0f d, %s obs; %s" % (
            b["name"], b["cls"], b["phase"], b["a"], b["e"], b["i"], b["H"], b["arc"], b["n_obs"], b["vet"] or "unexplained"))
    json.dump(dict(n=len(bodies), bins=bins, l3=l3), open(core.results("jupiter_l3.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_jupiter_l3", "Lurkers at Jupiter's far Lagrange point", "JPL Small-Body Database (sbdb_query.api)",
            "bodies within %g au of Jupiter's semi-major axis with e below %g: %d" % (a_tol, e_max, len(bodies)), os.path.abspath(__file__), "jupl3", len(bodies), "bodies",
            dict(half_width_deg=half_width_deg, a_tol=a_tol, e_max=e_max), "the counts within %g deg of the phases 0, 60, 120, 240 and 300 deg: %s" % (half_width_deg, bins),
            "a body within %g deg of the point opposite Jupiter on its orbit" % half_width_deg, "%d at L3 against %d and %d at L4 and L5; %d on short arcs, %d Trojans librating widely; %d unexplained" % (
                len(l3), bins[60], bins[300], sum(b["vet"] == "fit-degeneracy" for b in l3), sum(b["vet"] == "co-orbital" for b in l3), len(left)),
            [dict(target="sbdb:%s" % b["pdes"], stat=round(b["phase"], 1), unit="deg from Jupiter along the orbit", vet=b["vet"],
                  note="%s, a %.3f, e %.3f, i %.1f, H %s, arc %.0f d" % (b["cls"], b["a"], b["e"], b["i"], b["H"], b["arc"])) for b in l3])
        print("recorded")


aimed_DOC = r"""Bodies aimed at Earth: Small-Body Database near-Earth asteroids with a minimum orbit intersection distance with Earth below 0.0001 au, 15,000 km, orbits that pass through the Earth itself or its near space, against the count the distribution of small intersection distances gives, which is uniform near zero so that the bin from 0.0001 to 0.001 au holds nine times the first; a body steered at us would crowd the first bin; bodies on arcs under a year are set aside, since a short arc leaves the intersection distance unfixed.

    python -I seti.py solar_system aimed [record] [--moid-au 0.0001] [--arc-min-d 365]"""


def aimed_main(record=False, moid_au=0.0001, arc_min_d=365.0):
    from scipy.stats import poisson
    rows = sbdb_rows(["moid|LT|%g" % (10 * moid_au), "neo|EQ|Y"], "full_name,pdes,moid,moid_ld,a,e,i,q,H,data_arc,n_obs_used,condition_code,pha", kind="a")
    bodies, short = [], 0
    for r in rows:
        m = num(r["moid"])
        if m is None:
            continue
        if (num(r["data_arc"]) or 0) < arc_min_d:
            short += 1
            continue
        bodies.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), moid=m, a=num(r["a"]), e=num(r["e"]), i=num(r["i"]), H=num(r["H"]),
                           arc=num(r["data_arc"]) or 0, n_obs=num(r["n_obs_used"]), cc=num(r["condition_code"])))
    first = [b for b in bodies if b["moid"] < moid_au]
    rest = len(bodies) - len(first)
    exp = rest / 9.0
    p = float(poisson.sf(len(first) - 1, exp)) if first else 1.0
    for b in first:
        b["vet"] = "chance-rate" if p > 0.01 else None
    left = [b for b in first if b["vet"] is None]
    print("%d near-Earth asteroids on arcs of %g d or more with intersection distances below %g au (%d on shorter arcs set aside); below %g au: %d against %.1f from the uniform run "
          "of the next nine bins (Poisson p %.3g); unexplained %d" % (len(bodies), arc_min_d, 10 * moid_au, short, moid_au, len(first), exp, p, len(left)))
    for b in sorted(first, key=lambda b: b["moid"]):
        print("  %s: MOID %.6f au, a %.3f, e %.3f, i %.1f, H %s, arc %.0f d, %s obs, condition %s; %s" % (
            b["name"], b["moid"], b["a"], b["e"], b["i"], b["H"], b["arc"], b["n_obs"], b["cc"], b["vet"] or "unexplained"))
    json.dump(dict(n=len(bodies), first=first), open(core.results("bodies_aimed_at_earth.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_aimed_at_earth", "Bodies aimed at Earth", "JPL Small-Body Database (sbdb_query.api)", "near-Earth asteroids on arcs of %g d or more with intersection distances below %g au: %d" % (arc_min_d, 10 * moid_au, len(bodies)),
            os.path.abspath(__file__), "aimed", len(bodies), "asteroids", dict(moid_au=moid_au, arc_min_d=arc_min_d),
            "the uniform run of intersection distances near zero: %.1f expected in the first bin from the next nine" % exp,
            "a crowding of intersection distances below %g au" % moid_au, "%d against %.1f (Poisson p %.3g); %d unexplained" % (len(first), exp, p, len(left)),
            [dict(target="sbdb:%s" % b["pdes"], stat=b["moid"], unit="au, minimum orbit intersection distance with Earth", vet=b["vet"],
                  note="a %.3f, e %.3f, i %.1f, H %s, arc %.0f d, condition %s" % (b["a"], b["e"], b["i"], b["H"], b["arc"], b["cc"])) for b in first])
        print("recorded")


kirkwood_DOC = r"""Bodies parked in the Kirkwood gaps: Small-Body Database numbered asteroids by semi-major axis around the 3:1, 5:2 and 2:1 resonances with Jupiter at 2.502, 2.824 and 3.277 au, where the gaps are swept clear within a million years, those within 0.002 au of a resonance centre with eccentricity below 0.1 on arcs of twenty years or more, bodies holding a place nothing holds, against the density of the bins 0.01-0.03 au either side; a numbered body with a long arc in the centre of a gap is passing through on its way out, which the chaotic zone allows for a time, so each is read for its eccentricity, the lower the stranger.

    python -I seti.py solar_system kirkwood [record] [--half-au 0.002] [--e-max 0.1] [--arc-min-d 7305]"""

KIRKWOOD_GAPS = {"3:1": 2.5020, "5:2": 2.8237, "2:1": 3.2770}


def kirkwood_main(record=False, half_au=0.002, e_max=0.1, arc_min_d=7305.0):
    from scipy.stats import poisson
    out, counts = [], {}
    for name, a0 in KIRKWOOD_GAPS.items():
        rows = sbdb_rows(["a|RG|%g|%g" % (a0 - 0.03, a0 + 0.03), "e|LT|%g" % e_max, "data_arc|GT|%g" % arc_min_d, "pdes|DF"], "full_name,pdes,a,e,i,data_arc,n_obs_used,H", kind="a")
        rows = [r for r in rows if str(r["pdes"]).strip().isdigit()]
        a = np.array([float(r["a"]) for r in rows])
        centre = int(np.sum(np.abs(a - a0) < half_au))
        beside = int(np.sum((np.abs(a - a0) >= 0.01) & (np.abs(a - a0) < 0.03)))
        exp = beside * (2 * half_au) / 0.04
        p = float(poisson.sf(centre - 1, exp)) if centre else 1.0
        counts[name] = dict(a0=a0, centre=centre, beside=beside, expected=exp, p=p)
        for r in rows:
            if abs(float(r["a"]) - a0) < half_au:
                out.append(dict(gap=name, name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), a=float(r["a"]), e=num(r["e"]), i=num(r["i"]), arc=num(r["data_arc"]),
                                n_obs=num(r["n_obs_used"]), H=num(r["H"]), vet="chance-rate" if p > 0.01 else None))
    left = [o for o in out if o["vet"] is None]
    print("; ".join("%s at %.4f au: %d within %g au against %.1f from the %d in the bins beside (Poisson p %.3g)" % (k, v["a0"], v["centre"], half_au, v["expected"], v["beside"], v["p"])
                    for k, v in counts.items()) + "; unexplained %d" % len(left))
    for o in sorted(out, key=lambda o: o["e"]):
        print("  %s in the %s gap: a %.5f, e %.4f, i %.1f, arc %.0f d, %s obs, H %s; %s" % (o["name"], o["gap"], o["a"], o["e"], o["i"], o["arc"], o["n_obs"], o["H"], o["vet"] or "unexplained"))
    json.dump(dict(counts=counts, bodies=out), open(core.results("kirkwood_parked.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_kirkwood_parked", "Bodies parked in the Kirkwood gaps", "JPL Small-Body Database (sbdb_query.api)",
            "numbered asteroids within 0.03 au of the 3:1, 5:2 and 2:1 resonances with e below %g on arcs of %g d or more: %d" % (e_max, arc_min_d, sum(v["centre"] + v["beside"] for v in counts.values())),
            os.path.abspath(__file__), "kirkwood", sum(v["centre"] + v["beside"] for v in counts.values()), "asteroids", dict(half_au=half_au, e_max=e_max, arc_min_d=arc_min_d),
            "the density 0.01-0.03 au either side of each resonance: %s" % {k: v["beside"] for k, v in counts.items()},
            "bodies within %g au of a resonance centre beyond what the gap's edges give" % half_au,
            "; ".join("%s: %d against %.1f (p %.3g)" % (k, v["centre"], v["expected"], v["p"]) for k, v in counts.items()) + "; %d unexplained" % len(left),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["a"], 5), unit="au, semi-major axis", vet=o["vet"],
                  note="%s gap, e %.4f, i %.1f, arc %.0f d, H %s" % (o["gap"], o["e"], o["i"], o["arc"], o["H"])) for o in out])
        print("recorded")


tnocircles_DOC = r"""Circles beyond Neptune: Small-Body Database trans-Neptunian objects with semi-major axes beyond 40 au and eccentricities with a formal uncertainty, those whose eccentricity at its 3 sigma upper bound lies below 0.005, an orbit rounder than the cold classical belt's own, which was born with eccentricities of a few hundredths, against the count the Rayleigh rise of the belt's eccentricities near zero gives; each read for its arc and number of observations, since an orbit fitted over a short arc takes a circle by assumption.

    python -I seti.py solar_system tnocircles [record] [--a-min 40] [--e-max 0.005] [--sigma 3] [--arc-min-d 3652]"""


def tnocircles_main(record=False, a_min=40.0, e_max=0.005, sigma=3.0, arc_min_d=3652.0):
    rows = sbdb_rows(["a|GT|%g" % a_min, "sigma_e|DF"], "full_name,pdes,a,e,sigma_e,i,q,data_arc,n_obs_used,condition_code,class", kind="a")
    st = []
    for r in rows:
        e, se = num(r["e"]), num(r["sigma_e"])
        if e is None or se is None:
            continue
        st.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), a=num(r["a"]), e=e, se=se, up=e + sigma * se, i=num(r["i"]), q=num(r["q"]),
                       arc=num(r["data_arc"]) or 0, n_obs=num(r["n_obs_used"]), cc=num(r["condition_code"]), cls=r["class"]))
    from scipy.stats import poisson
    hits = [s for s in st if s["up"] < e_max]
    # a Rayleigh distribution of eccentricities rises as the square near zero, so the count below 10 e_max scaled by a hundredth is the natural expectation
    exp = sum(1 for s in st if s["e"] < 10 * e_max) / 100.0
    p = float(poisson.sf(len(hits) - 1, exp)) if hits else 1.0
    for s in hits:
        s["vet"] = "fit-degeneracy" if s["arc"] < arc_min_d else "chance-rate" if p > 0.01 else None
    left = [s for s in hits if s["vet"] is None]
    ups = sorted(s["up"] for s in st)
    k = max(1, int(0.01 * len(ups)))
    print("%d trans-Neptunian objects beyond %g au with an eccentricity uncertainty; upper eccentricity below %g at %g sigma: %d against %.1f from the Rayleigh rise of the %d below %g "
          "(Poisson p %.3g); %d on arcs under %g d; unexplained %d; the lowest percentile of upper bounds sits at %.4f" % (
              len(st), a_min, e_max, sigma, len(hits), exp, sum(1 for s in st if s["e"] < 10 * e_max), 10 * e_max, p, sum(s["vet"] == "fit-degeneracy" for s in hits), arc_min_d,
              len(left), ups[k - 1] if ups else 0))
    for s in sorted(hits, key=lambda s: s["up"]):
        print("  %s (%s): a %.2f au, e %.5f +- %.5f, i %.1f, arc %.0f d, %s obs, condition %s; %s" % (
            s["name"], s["cls"], s["a"], s["e"], s["se"], s["i"], s["arc"], s["n_obs"], s["cc"], s["vet"] or "unexplained"))
    json.dump(dict(n=len(st), hits=hits), open(core.results("tno_circles.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_tno_circles", "Circles beyond Neptune", "JPL Small-Body Database (sbdb_query.api)", "trans-Neptunian objects beyond %g au with an eccentricity uncertainty: %d" % (a_min, len(st)),
            os.path.abspath(__file__), "tnocircles", len(st), "objects", dict(a_min=a_min, e_max=e_max, sigma=sigma, arc_min_d=arc_min_d),
            "the Rayleigh rise of the eccentricities near zero: %.1f expected from the %d below %g; the lowest percentile of upper bounds at %.4f" % (
                exp, sum(1 for s in st if s["e"] < 10 * e_max), 10 * e_max, ups[k - 1] if ups else 0),
            "an orbit beyond Neptune rounder than %g at its %g sigma upper bound" % (e_max, sigma),
            "%d against %.1f (Poisson p %.3g), %d on short arcs; %d unexplained" % (len(hits), exp, p, sum(s["vet"] == "fit-degeneracy" for s in hits), len(left)),
            [dict(target="sbdb:%s" % s["pdes"], stat=round(s["up"], 5), unit="upper eccentricity", vet=s["vet"],
                  note="a %.2f au, e %.5f +- %.5f, i %.1f, arc %.0f d, %s obs" % (s["a"], s["e"], s["se"], s["i"], s["arc"], s["n_obs"])) for s in hits])
        print("recorded")


sunfocus_DOC = r"""Bodies at the Sun's focus: Small-Body Database bodies with perihelia beyond 100 au, where the Sun's gravitational lens brings the light of other stars to a focus from 550 au outward and a receiver would be parked, against the inner Oort cloud's known members, whose perihelia reach 80 au; a body that never comes closer than 100 au is seen only if large, so each is read for its size and its arc.

    python -I seti.py solar_system sunfocus [record] [--q-min-au 100] [--list-q-au 60]"""


def sunfocus_main(record=False, q_min_au=100.0, list_q_au=60.0):
    rows = sbdb_rows(["q|GT|%g" % list_q_au], "full_name,pdes,a,e,q,ad,i,H,data_arc,n_obs_used,condition_code,class", kind="a")
    rows += sbdb_rows(["q|GT|%g" % list_q_au], "full_name,pdes,a,e,q,ad,i,H,data_arc,n_obs_used,condition_code,class", kind="c")
    bodies = sorted([dict(name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), a=num(r["a"]), e=num(r["e"]), q=num(r["q"]), ad=num(r["ad"]), i=num(r["i"]), H=num(r["H"]),
                          arc=num(r["data_arc"]), n_obs=num(r["n_obs_used"]), cc=num(r["condition_code"]), cls=r["class"]) for r in rows if num(r["q"]) is not None],
                    key=lambda b: -b["q"])
    far = [b for b in bodies if b["q"] > q_min_au]
    for b in far:
        b["vet"] = "fit-degeneracy" if (b["arc"] or 0) < 3652 else None
    left = [b for b in far if b["vet"] is None]
    print("%d bodies with perihelia beyond %g au; beyond %g au: %d; unexplained %d" % (len(bodies), list_q_au, q_min_au, len(far), len(left)))
    for b in bodies[:12]:
        print("  %s (%s): q %.1f au, a %s, e %s, i %.1f, H %s, arc %s d, %s obs; %s" % (b["name"], b["cls"], b["q"], b["a"], b["e"], b["i"] or 0, b["H"], b["arc"], b["n_obs"],
                                                                                        b.get("vet") or ("unexplained" if b["q"] > q_min_au else "inside %g au" % q_min_au)))
    json.dump(dict(n=len(bodies), far=far, nearest=bodies[:12]), open(core.results("sun_focus_bodies.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_sun_focus", "Bodies at the Sun's focus", "JPL Small-Body Database (sbdb_query.api)", "bodies with perihelia beyond %g au: %d" % (list_q_au, len(bodies)),
            os.path.abspath(__file__), "sunfocus", len(bodies), "bodies", dict(q_min_au=q_min_au, list_q_au=list_q_au),
            "the inner Oort cloud's known members: the largest perihelion %s at %.1f au" % (bodies[0]["name"] if bodies else None, bodies[0]["q"] if bodies else 0),
            "a body whose perihelion lies beyond %g au" % q_min_au, "%d; %d unexplained" % (len(far), len(left)),
            [dict(target="sbdb:%s" % b["pdes"], stat=round(b["q"], 2), unit="au, perihelion", vet=b.get("vet"), note="a %s, e %s, i %.1f, H %s, arc %s d" % (b["a"], b["e"], b["i"] or 0, b["H"], b["arc"]))
             for b in far])
        print("recorded")


retroco_DOC = r"""Retrograde co-orbitals of the giant planets: Small-Body Database bodies with semi-major axes within 5 per cent of Jupiter's, Saturn's, Uranus's or Neptune's, inclinations above 90 deg and eccentricities below 0.5, since a comet crossing the planets shares no orbit, a body sharing a planet's orbit while circling the Sun the wrong way, which the 1:1 resonance holds for a time in the published case of (514107) Ka'epaoka'awela at Jupiter, captured from the Oort cloud or beyond, with 2007 VW266 published in the neighbouring 13/-14 resonance, and which a thing steered into place would show; each read for its arc, against the published cases and against the toolkit's own integration of the orbit and fifty clones of its covariance with the four giant planets for 100 kyr each way, where a body the resonance holds never meets the planet.

With `--inner true` the planets are Mercury, Venus, Earth and Mars, recorded as their own run.

    python -I seti.py solar_system retroco [record] [--a-tol 0.05] [--i-min 90] [--e-max 0.5] [--arc-min-d 365] [--inner]"""

RETROCO_PLANETS = {"Jupiter": 5.2026, "Saturn": 9.5549, "Uranus": 19.2184, "Neptune": 30.1104}
RETROCO_INNER = {"Mercury": 0.3871, "Venus": 0.7233, "Earth": 1.0000, "Mars": 1.5237}
# bodies the toolkit's own integration (four giant planets, 20 kyr each way from the nominal orbit) holds in the retrograde co-orbital resonance
RETROCO_INTEGRATED = {"2016 YB13": "held at Jupiter's semi-major axis without an approach inside 0.5 au for 7 kyr each way from the nominal orbit and, over fifty clones of its "
                                   "covariance, for a median of 3 kyr forward and 13 kyr backward, a transient member of the retrograde 1:1 resonance, where the control "
                                   "(514107) Ka'epaoka'awela holds for the whole 100 kyr each way"}
RETROCO_KNOWN = {"514107": "Ka'epaoka'awela, the published retrograde co-orbital of Jupiter (Wiegert et al. 2017)",
                 "2007 VW266": "a published retrograde object near Jupiter's orbit in the 13/-14 resonance (Connors and Wiegert 2018)"}


def retroco_main(record=False, a_tol=0.05, i_min=90.0, e_max=0.5, arc_min_d=365.0, inner=False):
    out, n = [], {}
    which = "an inner" if inner else "a giant"
    for planet, a0 in (RETROCO_INNER if inner else RETROCO_PLANETS).items():
        rows = sbdb_rows(["a|RG|%g|%g" % (a0 * (1 - a_tol), a0 * (1 + a_tol)), "i|GT|%g" % i_min, "e|LT|%g" % e_max], "full_name,pdes,a,e,i,q,data_arc,n_obs_used,class", kind="a")
        rows += sbdb_rows(["a|RG|%g|%g" % (a0 * (1 - a_tol), a0 * (1 + a_tol)), "i|GT|%g" % i_min, "e|LT|%g" % e_max], "full_name,pdes,a,e,i,q,data_arc,n_obs_used,class", kind="c")
        n[planet] = len(rows)
        for r in rows:
            pdes = str(r["pdes"]).strip()
            arc = num(r["data_arc"]) or 0
            out.append(dict(planet=planet, name=" ".join(str(r["full_name"]).split()), pdes=pdes, a=num(r["a"]), e=num(r["e"]), i=num(r["i"]), q=num(r["q"]), arc=arc, n_obs=num(r["n_obs_used"]),
                            cls=str(r["class"]).strip(), vet=("natural-source" if pdes in RETROCO_KNOWN else "co-orbital" if pdes in RETROCO_INTEGRATED else
                                                              "fit-degeneracy" if arc < arc_min_d else None),
                            note=RETROCO_KNOWN.get(pdes) or RETROCO_INTEGRATED.get(pdes)))
    left = [o for o in out if o["vet"] is None]
    print("bodies within %g of %s planet's semi-major axis on retrograde orbits with e below %g: %s (%d published, %d held in the resonance by the toolkit's integration, %d on "
          "arcs under %g d); unexplained %d" % (a_tol, which, e_max, n, sum(o["vet"] == "natural-source" for o in out), sum(o["vet"] == "co-orbital" for o in out),
                                                 sum(o["vet"] == "fit-degeneracy" for o in out), arc_min_d, len(left)))
    for o in sorted(out, key=lambda o: (o["planet"], -o["arc"])):
        print("  %s at %s (%s): a %.3f, e %.3f, i %.1f, q %.2f, arc %.0f d, %s obs%s; %s" % (
            o["name"], o["planet"], o["cls"], o["a"], o["e"], o["i"], o["q"], o["arc"], o["n_obs"], ", " + o["note"] if o["note"] else "", o["vet"] or "unexplained"))
    json.dump(dict(n=n, bodies=out), open(core.results("retrograde_coorbitals_inner.json" if inner else "retrograde_coorbitals.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_retrograde_coorbitals_inner" if inner else "sbdb_retrograde_coorbitals", "Retrograde co-orbitals of the %s planets" % ("inner" if inner else "giant"),
            "JPL Small-Body Database (sbdb_query.api)",
            "retrograde bodies within %g of %s planet's semi-major axis with e below %g: %s" % (a_tol, which, e_max, n), os.path.abspath(__file__), "retroco", len(out), "bodies",
            dict(a_tol=a_tol, i_min=i_min, e_max=e_max, arc_min_d=arc_min_d, inner=inner), "the published retrograde resonators; the arcs", "a retrograde body sharing %s planet's orbit" % which,
            "%d: %d published, %d held in the resonance by the toolkit's integration, %d on short arcs; %d unexplained" % (
                len(out), sum(o["vet"] == "natural-source" for o in out), sum(o["vet"] == "co-orbital" for o in out), sum(o["vet"] == "fit-degeneracy" for o in out), len(left)),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["i"], 1), unit="deg, inclination", vet=o["vet"],
                  note="%s, a %.3f, e %.3f, q %.2f, arc %.0f d%s" % (o["planet"], o["a"], o["e"], o["q"], o["arc"], ", " + o["note"] if o["note"] else "")) for o in out])
        print("recorded")


outofplane_DOC = r"""Bodies pushed out of their orbital planes: Small-Body Database comets and asteroids with a normal non-gravitational term A3 measured at 3 sigma, those whose normal term exceeds half their radial term in size, a push out of the plane as large as the push along the Sun's line, which a jet from a pole does and a thruster aimed across the orbit would; each read for its arc and against the published dark comets, and the ratio against the distribution of every body with both terms.

    python -I seti.py solar_system outofplane [record] [--sigma 3] [--ratio-min 0.5] [--arc-min-d 1000]"""


def outofplane_main(record=False, sigma=3.0, ratio_min=0.5, arc_min_d=1000.0):
    out, n, ratios = [], {}, []
    for kind in ("c", "a"):
        rows = sbdb_rows(["A3|DF", "A1|DF"], "full_name,pdes,A1,A1_sigma,A2,A2_sigma,A3,A3_sigma,q,e,i,data_arc,n_obs_used,class", kind=kind)
        n[kind] = len(rows)
        for r in rows:
            a1, a3, s3 = num(r["A1"]), num(r["A3"]), num(r["A3_sigma"])
            if a1 is None or a3 is None or not s3 or a1 == 0:
                continue
            ratios.append(abs(a3 / a1))
            if abs(a3) / s3 < sigma or abs(a3) < ratio_min * abs(a1):
                continue
            name = " ".join(str(r["full_name"]).split())
            prov = re.search(r"\(([^)]+)\)", name)
            prov = prov.group(1) if prov else name
            arc = num(r["data_arc"]) or 0
            out.append(dict(kind="comet" if kind == "c" else "asteroid", name=name, pdes=str(r["pdes"]).strip(), A1=a1, A3=a3, A3_sigma=s3, ratio=abs(a3 / a1), q=num(r["q"]), e=num(r["e"]),
                            arc=arc, n_obs=num(r["n_obs_used"]), cls=r["class"], vet="dark-comet" if prov in PUSHED_DARK_COMETS else "fit-degeneracy" if arc < arc_min_d else
                            "activity" if kind == "c" else None))
    left = [o for o in out if o["vet"] is None]
    print("%d comets and %d asteroids with normal and radial terms; the normal over radial ratio's median %.2f and 90th percentile %.2f; normal term at %g sigma and above %g of the radial: "
          "%d (%d comets, whose jets push off their poles, %d dark comets, %d on arcs under %g d); unexplained %d" % (
              n["c"], n["a"], float(np.median(ratios)) if ratios else 0, float(np.percentile(ratios, 90)) if ratios else 0, sigma, ratio_min, len(out),
              sum(o["vet"] == "activity" for o in out), sum(o["vet"] == "dark-comet" for o in out), sum(o["vet"] == "fit-degeneracy" for o in out), arc_min_d, len(left)))
    for o in sorted(out, key=lambda o: -o["ratio"]):
        if o["vet"] != "activity" or len(out) <= 30:
            print("  %s (%s, %s): A3 %.3e +- %.1e, A1 %.3e, ratio %.2f, q %.3f, e %.3f, arc %.0f d, %s obs; %s" % (
                o["name"], o["kind"], o["cls"], o["A3"], o["A3_sigma"], o["A1"], o["ratio"], o["q"], o["e"], o["arc"], o["n_obs"], o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out), open(core.results("pushed_out_of_plane.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_out_of_plane", "Bodies pushed out of their orbital planes", "JPL Small-Body Database (sbdb_query.api)", "comets with normal and radial terms: %d; asteroids: %d" % (n["c"], n["a"]),
            os.path.abspath(__file__), "outofplane", n["c"] + n["a"], "bodies", dict(sigma=sigma, ratio_min=ratio_min, arc_min_d=arc_min_d),
            "the ratio's distribution over every body with both terms: median %.2f, 90th percentile %.2f" % (float(np.median(ratios)) if ratios else 0, float(np.percentile(ratios, 90)) if ratios else 0),
            "a normal term at %g sigma above %g of the radial term" % (sigma, ratio_min), "%d: %d comets, %d dark comets, %d on short arcs; %d unexplained" % (
                len(out), sum(o["vet"] == "activity" for o in out), sum(o["vet"] == "dark-comet" for o in out), sum(o["vet"] == "fit-degeneracy" for o in out), len(left)),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["ratio"], 3), unit="normal over radial non-gravitational term", vet=o["vet"],
                  note="%s, A3 %.3e +- %.1e, A1 %.3e, q %.3f, e %.3f, arc %.0f d" % (o["kind"], o["A3"], o["A3_sigma"], o["A1"], o["q"], o["e"], o["arc"])) for o in out])
        print("recorded")


bluesurfaces_DOC = r"""Surfaces bluer than any asteroid class: Gaia DR3's mean reflectance spectra of asteroids (sso_reflectance_spectrum), the slope of each spectrum's unflagged bands from 450 to 850 nm by least squares, those falling more steeply than -15 per cent per 100 nm at 3 sigma, bluer than the B types, the bluest surfaces the belt makes, where a coating made for its colour would sit; each read for the number of spectra behind the mean and against the bluest known family, Pallas's, whose members reach -5 per cent per 100 nm.

    python -I seti.py solar_system bluesurfaces [record] [--slope-max -15] [--sigma 3] [--min-bands 6]"""


def bluesurfaces_main(record=False, slope_max=-15.0, sigma=3.0, min_bands=6):
    rows = core.tap_async("gaia", "SELECT number_mp, denomination, num_of_spectra, wavelength, reflectance_spectrum, reflectance_spectrum_err FROM gaiadr3.sso_reflectance_spectrum WHERE "
                                  "reflectance_spectrum_flag = 0 AND wavelength BETWEEN 450 AND 850 AND reflectance_spectrum IS NOT NULL AND reflectance_spectrum_err > 0", maxrec=1000000)
    by = {}
    for r in rows:
        by.setdefault(int(r["number_mp"]), []).append((float(r["wavelength"]), float(r["reflectance_spectrum"]), float(r["reflectance_spectrum_err"]), str(r["denomination"]).strip(),
                                                        int(num(r["num_of_spectra"]) or 0)))
    out, slopes = [], []
    for mp, pts in by.items():
        if len(pts) < min_bands:
            continue
        w = np.array([p[0] for p in pts]) / 100.0
        f = np.array([p[1] for p in pts])
        e = np.array([p[2] for p in pts])
        A = np.vstack([np.ones_like(w), w - 5.5]).T / e[:, None]
        coef, res, rank, sv = np.linalg.lstsq(A, f / e, rcond=None)
        cov = np.linalg.inv(A.T @ A)
        slope, es = 100.0 * coef[1], 100.0 * math.sqrt(cov[1, 1])
        slopes.append(slope)
        if slope + sigma * es < slope_max:
            out.append(dict(mp=mp, name=pts[0][3], n_spec=pts[0][4], slope=slope, err=es, bands=len(pts), spectrum=[round(float(x), 3) for x in f],
                            vet="single-record" if pts[0][4] < 2 else None))
    left = [o for o in out if o["vet"] is None]
    pct = float(np.percentile(slopes, 0.1)) if slopes else 0.0
    print("%d asteroids with %d or more unflagged bands from 450 to 850 nm; slopes' 0.1st percentile %.1f per cent per 100 nm; below %g at %g sigma: %d (%d from a single spectrum); "
          "unexplained %d" % (len(slopes), min_bands, pct, slope_max, sigma, len(out), sum(o["vet"] == "single-record" for o in out), len(left)))
    for o in sorted(out, key=lambda o: o["slope"]):
        print("  (%d) %s: slope %.1f +- %.1f per cent per 100 nm over %d bands, %d spectra: %s; %s" % (o["mp"], o["name"], o["slope"], o["err"], o["bands"], o["n_spec"], o["spectrum"],
                                                                                                   o["vet"] or "unexplained"))
    json.dump(dict(n=len(slopes), hits=out, percentile=pct), open(core.results("asteroid_blue_surfaces.json"), "w"), indent=1)
    if record:
        core.record_run(
            "gaia_sso_blue_surfaces", "Surfaces bluer than any asteroid class", "Gaia DR3 sso_reflectance_spectrum", "asteroids with %d or more unflagged bands from 450 to 850 nm: %d" % (min_bands, len(slopes)),
            os.path.abspath(__file__), "bluesurfaces", len(slopes), "asteroids", dict(slope_max=slope_max, sigma=sigma, min_bands=min_bands),
            "the slopes' own distribution: 0.1st percentile %.1f per cent per 100 nm; the number of spectra behind each mean" % pct,
            "a reflectance slope below %g per cent per 100 nm at %g sigma" % (slope_max, sigma), "%d, %d from a single spectrum; %d unexplained" % (
                len(out), sum(o["vet"] == "single-record" for o in out), len(left)),
            [dict(target="asteroid:%d" % o["mp"], stat=round(o["slope"], 1), unit="per cent per 100 nm", vet=o["vet"], note="%s, +- %.1f, %d bands, %d spectra" % (o["name"], o["err"], o["bands"], o["n_spec"]))
             for o in out])
        print("recorded")


lightbodies_DOC = r"""Launch provenance of light bodies: Small-Body Database bodies with a radial non-gravitational term A1 measured at 3 sigma, the area-to-mass ratio that solar radiation pressure alone would need to give that term, bodies above 0.1 m^2/kg, where a rock would have to be lighter than foam and a spent rocket stage or a solar sail sits; each read against the bodies of our own launching the database holds, the published dark comets and the interstellar objects, whose outgassing supplies the push instead, and its arc.

    python -I seti.py solar_system lightbodies [record] [--sigma 3] [--am-min 0.1] [--arc-min-d 30]"""

# solar radiation pressure at 1 au on a unit area-to-mass ratio, in au/d^2 per m^2/kg
LIGHTBODIES_PRESSURE = 4.56e-6 * 86400.0 ** 2 / 1.495978707e11


def lightbodies_main(record=False, sigma=3.0, am_min=0.1, arc_min_d=30.0):
    out, n = [], {}
    for kind in ("a", "c"):
        rows = sbdb_rows(["A1|DF"], "full_name,pdes,A1,A1_sigma,A2,A2_sigma,H,diameter,a,e,q,i,data_arc,n_obs_used,condition_code,class", kind=kind)
        n[kind] = len(rows)
        for r in rows:
            a1, s1 = num(r["A1"]), num(r["A1_sigma"])
            if a1 is None or not s1 or a1 / s1 < sigma:
                continue
            am = a1 / LIGHTBODIES_PRESSURE
            if am < am_min:
                continue
            name = " ".join(str(r["full_name"]).split())
            prov = re.search(r"\(([^)]+)\)", name)
            prov = prov.group(1) if prov else name
            arc = num(r["data_arc"]) or 0
            out.append(dict(kind="asteroid" if kind == "a" else "comet", name=name, pdes=str(r["pdes"]).strip(), A1=a1, A1_sigma=s1, am=am, H=num(r["H"]), d=num(r["diameter"]),
                            a=num(r["a"]), e=num(r["e"]), q=num(r["q"]), arc=arc, n_obs=num(r["n_obs_used"]), cls=r["class"],
                            vet=("our-launch" if prov in PUSHED_LAUNCHED else "dark-comet" if prov in PUSHED_DARK_COMETS else "activity" if kind == "c" else
                                 "natural-source" if any(k in name for k in HYPERBOLIC_INTERSTELLAR) else "fit-degeneracy" if arc < arc_min_d else None)))
    left = [o for o in out if o["vet"] is None]
    print("%d asteroids and %d comets with a radial term; at %g sigma and an area-to-mass ratio above %g m^2/kg: %d (%d of our launching, %d dark comets, %d comets, %d on arcs under %g d); "
          "%d interstellar; unexplained %d" % (n["a"], n["c"], sigma, am_min, len(out), sum(o["vet"] == "our-launch" for o in out), sum(o["vet"] == "dark-comet" for o in out),
                             sum(o["vet"] == "activity" for o in out), sum(o["vet"] == "fit-degeneracy" for o in out), arc_min_d, sum(o["vet"] == "natural-source" for o in out), len(left)))
    for o in sorted(out, key=lambda o: -o["am"]):
        print("  %s (%s, %s): A1 %.2e +- %.1e au/d^2, %.2f m^2/kg, H %s, a %.3f e %.3f, arc %.0f d, %s obs; %s" % (
            o["name"], o["kind"], o["cls"], o["A1"], o["A1_sigma"], o["am"], o["H"], o["a"] or 0, o["e"] or 0, o["arc"], o["n_obs"], o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out), open(core.results("light_bodies_provenance.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_light_bodies", "Launch provenance of light bodies", "JPL Small-Body Database (sbdb_query.api)", "asteroids with a radial non-gravitational term: %d; comets: %d" % (n["a"], n["c"]),
            os.path.abspath(__file__), "lightbodies", n["a"] + n["c"], "bodies", dict(sigma=sigma, am_min=am_min, arc_min_d=arc_min_d, pressure=LIGHTBODIES_PRESSURE),
            "the bodies of our own launching in the database; the published dark comets; comets' outgassing; the arcs",
            "an area-to-mass ratio above %g m^2/kg from the radial term at %g sigma" % (am_min, sigma), "%d: %d of our launching, %d dark comets, %d comets, %d interstellar, %d on short arcs; %d unexplained" % (
                len(out), sum(o["vet"] == "our-launch" for o in out), sum(o["vet"] == "dark-comet" for o in out), sum(o["vet"] == "activity" for o in out),
                sum(o["vet"] == "natural-source" for o in out), sum(o["vet"] == "fit-degeneracy" for o in out), len(left)),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["am"], 3), unit="m^2/kg from the radial term", vet=o["vet"],
                  note="%s, A1 %.2e +- %.1e, H %s, a %.3f e %.3f, arc %.0f d" % (o["kind"], o["A1"], o["A1_sigma"], o["H"], o["a"] or 0, o["e"] or 0, o["arc"])) for o in out])
        print("recorded")


spinsurvey_DOC = r"""Centrifugal habitats, survey periods: the rotation periods and diameters of the faint main-belt asteroids of Lo et al. (2020, VizieR J/AJ/159/25), its secure periods and its super-fast rotators, each body's tensile strength needed to hold together at 2000 kg/m^3 beyond the gravity spin limit, against the LCDB test's limits; a hollowed body spun for gravity would spin faster than rubble can at its size.

    python -I seti.py solar_system spinsurvey [record] [--rho 2000] [--d-min-km 0.3] [--kpa-min 1] [--km-d 1] [--kpa 100] [--minutes 10]"""


def spinsurvey_main(record=False, rho=2000.0, d_min_km=0.3, kpa_min=1.0, km_d=1.0, kpa=100.0, minutes=10.0):
    G = 6.674e-11
    wc2 = 4 * math.pi * G * rho / 3
    out = {}
    for t in core.vizier("J/AJ/159/25"):
        name = t.meta.get("name", "").split("/")[-1]
        pc = "Prot" if "Prot" in t.colnames else ("Per" if "Per" in t.colnames else None)
        if pc is None or "Diam" not in t.colnames:
            continue
        for r in t:
            P, D = num(r[pc]), num(r["Diam"])
            if not P or not D:
                continue
            w = 2 * math.pi / (P * 3600)
            R = D * 500
            o = dict(id=str(r["ID"]).strip(), table=name, D=D, P=P, kpa=rho * R * R / 2 * max(0.0, w * w - wc2) / 1000, vet=None)
            if o["id"] not in out or name != "table2":
                out[o["id"]] = o
    out = list(out.values())
    need = sorted([o for o in out if o["D"] >= d_min_km and o["kpa"] > kpa_min], key=lambda o: -o["kpa"])
    km_fast = [o for o in out if o["D"] >= km_d and o["P"] * 60 < minutes]
    km_strong = [o for o in out if o["D"] >= km_d and o["kpa"] >= kpa]
    for o in need:
        o["vet"] = "cohesive" if o["kpa"] < kpa else None
    print("%d asteroids with periods and diameters (periods %.3f-%.1f h, diameters %.2f-%.1f km); spin limit %.2f h; %d of %g km or more need over %g kPa; of %g km or more under %g minutes %d, "
          "needing %g kPa %d" % (len(out), min(o["P"] for o in out), max(o["P"] for o in out), min(o["D"] for o in out), max(o["D"] for o in out), 2 * math.pi / math.sqrt(wc2) / 3600,
                                 len(need), d_min_km, kpa_min, km_d, minutes, len(km_fast), kpa, len(km_strong)))
    for o in need:
        print("  %-12s (%s) D %.3f km, P %.2f min, needs %.1f kPa; %s" % (o["id"], o["table"], o["D"], o["P"] * 60, o["kpa"], o["vet"] or "unexplained"))
    if record:
        core.record_run(
            "lo2020_spin_strength", "Centrifugal habitats, survey periods", "Lo et al. (2020) faint main-belt rotation periods (VizieR J/AJ/159/25)",
            "asteroids with periods and diameters: %d" % len(out), os.path.abspath(__file__), "spinsurvey", len(out), "asteroids",
            dict(rho=rho, d_min_km=d_min_km, kpa_min=kpa_min, km_d=km_d, kpa=kpa, minutes=minutes), "the strength a monolith holds, about %g kPa" % kpa,
            "a body of %g km spinning in %g minutes or needing %g kPa" % (km_d, minutes, kpa),
            "%d bodies of %g km or more need over %g kPa; %d of %g km or more under %g minutes and %d needing %g kPa" % (len(need), d_min_km, kpa_min, len(km_fast), km_d, minutes, len(km_strong), kpa),
            [dict(target="mpc:%s" % o["id"], stat=round(o["kpa"], 1), unit="kPa needed at %g kg/m^3" % rho, vet=o["vet"], note="%s, D %.3f km, P %.2f min" % (o["table"], o["D"], o["P"] * 60))
             for o in need])
        print("recorded")


fireballs_DOC = r"""Fireballs from outside the Solar System: every fireball of the CNEOS database (the JPL fireball API) with a location and a pre-entry velocity vector, the vector taken in Earth-fixed axes into barycentric axes, Earth's gravity removed from its speed, Earth's orbital velocity added, and the heliocentric speed set against the Sun's escape speed at Earth's distance; an interstellar body or probe striking Earth would arrive faster than that speed allows for anything bound, and for each fast one the fraction by which its speed would have to be overestimated for the orbit to be bound; the entries whose listed speed and vector disagree by more than 0.5 km/s are counted apart. Each fast one is redrawn with the errors that CNEOS shows against ground-based networks (Peña-Asensio et al. 2025: about 6 km/s in speed and 20 deg in radiant before 2018, 0.55 km/s and 1.1 deg from 2018), and one bound in more than 1 per cent of the draws is within the sensors' errors; so is one that an independent published velocity (fireballs_STEREO: two satellites' lightning imagers triangulating the trail) brings below the escape speed.

    python -I seti.py solar_system fireballs [record] [--margin-kms 0] [--mismatch-kms 0.5] [--draws 20000] [--p-bound 0.01]"""

# independent pre-entry velocities (Earth-fixed km/s) with their speed errors, by CNEOS date: Silber et al. (2026, RNAAS 10, 183) triangulate
# 2026-04-01 from GOES-East's Geostationary Lightning Mapper and MTG-I1's Lightning Imager, 83 deg apart, in two models
fireballs_STEREO = {"2026-04-01 02:13:14": (((3.5, -28.3, 49.7), 2.0), ((3.5, -27.7, 49.4), 3.0))}


def fireballs_main(record=False, margin_kms=0.0, mismatch_kms=0.5, draws=20000, p_bound=0.01):
    from astropy.coordinates import ITRS, GCRS, CartesianRepresentation, CartesianDifferential, EarthLocation, get_body_barycentric_posvel
    from astropy.time import Time
    import astropy.units as u
    d = json.loads(core.get("https://ssd-api.jpl.nasa.gov/fireball.api?req-loc=true&req-vel=true&vel-comp=true", timeout=120))
    f = d["fields"]
    out = []
    rng0 = np.random.default_rng(29)
    for row in d["data"]:
        r = dict(zip(f, row))
        if not all(r.get(k) for k in ("vx", "vy", "vz", "lat", "lon", "alt", "vel")):
            continue
        t = Time(r["date"].replace(" ", "T"), scale="utc")
        lat = float(r["lat"]) * (1 if r["lat-dir"] == "N" else -1)
        lon = float(r["lon"]) * (1 if r["lon-dir"] == "E" else -1)
        loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, float(r["alt"]) * u.km)
        v = np.array([float(r["vx"]), float(r["vy"]), float(r["vz"])])
        it = ITRS(CartesianRepresentation(loc.x, loc.y, loc.z, differentials=CartesianDifferential(*(v * u.km / u.s))), obstime=t)
        gc = it.transform_to(GCRS(obstime=t))
        vg = gc.velocity.d_xyz.to(u.km / u.s).value
        rr = float(np.linalg.norm(gc.cartesian.xyz.to(u.km).value))
        vesc_e = math.sqrt(2 * 398600.4418 / rr)
        pe, ve = get_body_barycentric_posvel("earth", t)
        ps, vs = get_body_barycentric_posvel("sun", t)
        vearth = (ve - vs).xyz.to(u.km / u.s).value
        unit = vg / np.linalg.norm(vg)

        def helio(speed):
            return float(np.linalg.norm(unit * math.sqrt(max(speed ** 2 - vesc_e ** 2, 0.0)) + vearth))
        vh = helio(float(np.linalg.norm(vg)))
        dsun = float(np.linalg.norm((pe - ps).xyz.to(u.km).value))
        vesc_s = math.sqrt(2 * 1.32712440018e11 / dsun)
        lo_s, hi_s = vesc_e, float(np.linalg.norm(vg))
        if vh > vesc_s:
            for _ in range(60):
                mid = 0.5 * (lo_s + hi_s)
                lo_s, hi_s = (mid, hi_s) if helio(mid) < vesc_s else (lo_s, mid)
        cut = 1 - lo_s / float(np.linalg.norm(vg)) if vh > vesc_s else 0.0
        # an independent velocity's heliocentric speed and the share of draws within its speed error that stay hyperbolic
        stereo = []
        for vs_, sv_ in fireballs_STEREO.get(r["date"], ()):
            it2 = ITRS(CartesianRepresentation(loc.x, loc.y, loc.z, differentials=CartesianDifferential(*(np.array(vs_) * u.km / u.s))), obstime=t)
            vg2 = it2.transform_to(GCRS(obstime=t)).velocity.d_xyz.to(u.km / u.s).value
            u2, s2 = vg2 / np.linalg.norm(vg2), float(np.linalg.norm(vg2))
            sp2 = s2 + sv_ * rng0.standard_normal(draws)
            vh2 = np.linalg.norm(u2[None, :] * np.sqrt(np.clip(sp2 ** 2 - vesc_e ** 2, 0, None))[:, None] + vearth[None, :], axis=1)
            stereo.append((float(np.linalg.norm(u2 * math.sqrt(max(s2 ** 2 - vesc_e ** 2, 0.0)) + vearth)), float(np.mean(vh2 > vesc_s))))
        out.append(dict(date=r["date"], energy_kt=core.num(r["impact-e"]), vel=float(r["vel"]), vmag=float(np.linalg.norm(v)), v_helio=vh, v_esc=vesc_s, cut=cut,
                        _u=unit, _s=float(np.linalg.norm(vg)), _ve=vesc_e, _vE=vearth, stereo=stereo,
                        lat=lat, lon=lon, alt=float(r["alt"])))
    for o in out:
        o["consistent"] = abs(o["vel"] - o["vmag"]) <= mismatch_kms
        o["excess"] = o["v_helio"] - o["v_esc"]
    hyp = [o for o in out if o["excess"] > margin_kms]
    rng = np.random.default_rng(23)
    for o in hyp:
        late = o["date"] >= "2018-01-01"
        sv, sd = (0.55, math.radians(1.1)) if late else (6.0, math.radians(20.0))
        u0 = o["_u"]
        e1 = np.cross(u0, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u0, e1)
        n = rng.standard_normal((draws, 2)) * sd
        uu = u0[None, :] + n[:, :1] * e1[None, :] + n[:, 1:] * e2[None, :]
        uu /= np.linalg.norm(uu, axis=1)[:, None]
        sp = o["_s"] + sv * rng.standard_normal(draws)
        vinf = np.sqrt(np.clip(sp ** 2 - o["_ve"] ** 2, 0, None))
        vh_d = np.linalg.norm(uu * vinf[:, None] + o["_vE"][None, :], axis=1)
        o["p_bound"] = float(np.mean(vh_d < o["v_esc"]))
        o["era"] = "from 2018" if late else "before 2018"
        o["vet"] = "sensor-error" if o["p_bound"] > p_bound or (o["stereo"] and min(x[0] for x in o["stereo"]) < o["v_esc"]) else None
    for o in out:
        for k in ("_u", "_s", "_ve", "_vE"):
            o.pop(k, None)
    hyp_ok = [o for o in hyp if o["consistent"]]
    left = [o for o in hyp if o["vet"] is None]
    print("%d fireballs with location and velocity vector (%d with listed speed and vector agreeing to %g km/s); heliocentric speeds above the Sun's escape speed: %d, %d of them consistent; "
          "bound in more than %g of the draws within CNEOS's errors: %d; beyond them: %d" % (len(out), sum(o["consistent"] for o in out), mismatch_kms, len(hyp), len(hyp_ok), p_bound,
                                                                                         len(hyp) - len(left), len(left)))
    for o in sorted(hyp, key=lambda o: -o["excess"]):
        print("  %s: %.1f km/s listed (vector %.1f), heliocentric %.1f against %.1f km/s (+%.1f), bound if the speed is %.1f per cent too high, bound in %.3f of the draws (%s)%s; impact energy %s kt, %.1f %.1f; %s" % (
            o["date"], o["vel"], o["vmag"], o["v_helio"], o["v_esc"], o["excess"], 100 * o["cut"], o["p_bound"], o["era"],
            "; triangulated %s" % ", ".join("%.1f km/s, hyperbolic in %.2f" % x for x in o["stereo"]) if o["stereo"] else "", o["energy_kt"], o["lat"], o["lon"], o["vet"] or "unexplained"))
    json.dump(out, open(core.results("cneos_fireballs.json"), "w"), indent=1, default=float)
    if record:
        core.record_run(
            "cneos_hyperbolic_fireballs", "Fireballs from outside the Solar System", "CNEOS fireball database (JPL fireball API)", "fireballs with location and pre-entry velocity vector: %d" % len(out),
            os.path.abspath(__file__), "fireballs", len(out), "fireballs", dict(margin_kms=margin_kms, mismatch_kms=mismatch_kms),
            "the Sun's escape speed at Earth's distance on each date; CNEOS's errors against ground-based networks (Peña-Asensio et al. 2025) by Monte Carlo",
            "a fireball arriving faster than anything bound to the Sun beyond the sensors' errors",
            "%d above the escape speed, %d of them bound in more than %g of the draws within CNEOS's errors; %d beyond them" % (len(hyp), len(hyp) - len(left), p_bound, len(left)),
            [dict(target="fireball:%s" % o["date"].replace(" ", "T"), stat=round(o["excess"], 2), unit="km/s above the Sun's escape speed", vet=o["vet"],
                  note="listed %.1f km/s, heliocentric %.1f, bound if the speed is %.1f per cent too high, bound in %.3f of the draws (%s)%s" % (
                      o["vel"], o["v_helio"], 100 * o["cut"], o["p_bound"], o["era"],
                      "; two lightning imagers' triangulated velocity (Silber et al. 2026) gives %s" % " and ".join("%.1f km/s heliocentric, hyperbolic in %.2f of draws" % x for x in o["stereo"])
                      if o["stereo"] else "")) for o in hyp])
        print("recorded")


deliveries_DOC = r"""Deliveries from stellar encounters: the three interstellar objects ('Oumuamua, Borisov and 3I/ATLAS) with their incoming velocity at infinity from the Small-Body Database's elements, set against every Gaia DR3 star within 100 pc with a radial velocity, both carried back on straight lines for 10 Myr, and the stars that passed within 1 pc of the object's path at under 10 km/s relative speed, where something released or dropped by the star would have left it; the count is set against 200 draws of the same speeds in random directions.

    python -I seti.py solar_system deliveries [record] [--dmin-pc 1] [--dv-kms 10] [--tmax-myr 10] [--draws 200]"""


def deliveries_main(record=False, dmin_pc=1.0, dv_kms=10.0, tmax_myr=10.0, draws=200):
    from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic, BarycentricMeanEcliptic, ICRS
    import astropy.units as u
    KMS = 1.0227121650537077
    isos = {}
    for name in ("1I", "2I", "3I"):
        d = json.loads(core.get("https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=%s" % name, timeout=60))
        el = {e["name"]: float(e["value"]) for e in d["orbit"]["elements"] if e.get("value") not in (None, "")}
        e, q, inc, om, w = el["e"], el["q"], math.radians(el["i"]), math.radians(el["om"]), math.radians(el["w"])
        a = q / (1 - e)
        vinf = math.sqrt(1.32712440018e11 / (abs(a) * 1.495978707e8))
        nu = -math.acos(-1 / e)
        # the incoming asymptote's direction in the orbital plane, rotated to ecliptic axes; the body moves away from it
        pq = np.array([math.cos(nu), math.sin(nu), 0.0])
        rz = lambda t: np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
        rx = lambda t: np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
        rhat = rz(om) @ rx(inc) @ rz(w) @ pq
        c = SkyCoord(x=-rhat[0], y=-rhat[1], z=-rhat[2], frame=BarycentricMeanEcliptic(), representation_type="cartesian").transform_to(ICRS())
        isos[d["object"]["fullname"]] = vinf * c.cartesian.xyz.value / np.linalg.norm(c.cartesian.xyz.value)
    rows = core.tap("gaia", "SELECT ra, dec, parallax, pmra, pmdec, radial_velocity, source_id FROM gaiadr3.gaia_source WHERE parallax > 10 AND parallax_over_error > 10 "
                            "AND radial_velocity IS NOT NULL")
    ra, de = np.radians([float(r["ra"]) for r in rows]), np.radians([float(r["dec"]) for r in rows])
    dist = 1000 / np.array([float(r["parallax"]) for r in rows])
    pmra, pmde, rv = (np.array([float(r[k]) for r in rows]) for k in ("pmra", "pmdec", "radial_velocity"))
    ehat = np.stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)], 1)
    ea = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], 1)
    ed = np.stack([-np.sin(de) * np.cos(ra), -np.sin(de) * np.sin(ra), np.cos(de)], 1)
    X = ehat * dist[:, None]
    V = ehat * rv[:, None] + ea * (4.74047 * pmra / 1000 * dist)[:, None] + ed * (4.74047 * pmde / 1000 * dist)[:, None]

    def passes(v):
        dV = (V - v[None, :]) * KMS
        t = -np.einsum("ij,ij->i", X, dV) / np.einsum("ij,ij->i", dV, dV)
        ok = (t < 0) & (t > -tmax_myr)
        dmin = np.linalg.norm(X + dV * t[:, None], axis=1)
        return np.nonzero(ok & (dmin < dmin_pc) & (np.linalg.norm(dV / KMS, axis=1) < dv_kms))[0], t, dmin
    rng = np.random.default_rng(31)
    out = {}
    for name, v in isos.items():
        hit, t, dmin = passes(v)
        null = []
        for _ in range(draws):
            u_ = rng.standard_normal(3)
            null.append(len(passes(np.linalg.norm(v) * u_ / np.linalg.norm(u_))[0]))
        out[name] = dict(vinf=float(np.linalg.norm(v)), n=len(hit), expected=float(np.mean(null)), p=float(np.mean(np.array(null) >= len(hit))) if len(hit) else 1.0,
                         stars=[dict(id=rows[k]["source_id"], t_myr=float(t[k]), dmin_pc=float(dmin[k]), dv=float(np.linalg.norm(V[k] - v))) for k in hit])
        print("%s: v_inf %.1f km/s; stars within %g pc of its path at under %g km/s in the last %g Myr: %d against %.2f in random directions (p %.2f); %s" % (
            name, out[name]["vinf"], dmin_pc, dv_kms, tmax_myr, len(hit), out[name]["expected"], out[name]["p"],
            ["Gaia DR3 %s at %.2f Myr, %.2f pc, %.1f km/s" % (s["id"], s["t_myr"], s["dmin_pc"], s["dv"]) for s in out[name]["stars"]][:5]))
    if record:
        core.record_run(
            "interstellar_object_home_stars", "Deliveries from stellar encounters", "JPL Small-Body Database (1I, 2I, 3I); Gaia DR3 stars within 100 pc with radial velocities",
            "the three interstellar objects against %d Gaia DR3 stars" % len(rows), os.path.abspath(__file__), "deliveries", len(isos), "objects",
            dict(dmin_pc=dmin_pc, dv_kms=dv_kms, tmax_myr=tmax_myr, draws=draws), "%d draws of each object's speed in random directions" % draws,
            "a star whose path met an interstellar object's within %g pc at under %g km/s" % (dmin_pc, dv_kms),
            "; ".join("%s %d against %.2f" % (k.strip(), v["n"], v["expected"]) for k, v in out.items()),
            [dict(target="gaia_dr3:%s" % s["id"], stat=round(s["dmin_pc"], 3), unit="pc from the object's path", vet="chance-rate" if v["p"] > 0.01 else None,
                  note="%s, %.2f Myr ago at %.1f km/s" % (k.strip(), -s["t_myr"], s["dv"])) for k, v in out.items() for s in v["stars"]])
        print("recorded")


GLM_ELLIPSOID = (6394.140, 6362.755)


def glm_sat(lon):
    """A geostationary satellite's Earth-fixed position, km."""
    return np.array([42164.16 * math.cos(math.radians(lon)), 42164.16 * math.sin(math.radians(lon)), 0.0])


def glm_ell_point(lat, lon):
    """The Earth-fixed point on GLM's lightning ellipsoid at a geodetic latitude and longitude, km."""
    a, b = GLM_ELLIPSOID
    e2 = 1 - b * b / (a * a)
    p, l = math.radians(lat), math.radians(lon)
    n = a / math.sqrt(1 - e2 * math.sin(p) ** 2)
    return np.array([n * math.cos(p) * math.cos(l), n * math.cos(p) * math.sin(l), n * (1 - e2) * math.sin(p)])


def glm_height(x):
    """Height above the WGS84 ellipsoid of an Earth-fixed point, km."""
    a, e2 = 6378.137, 0.0066943799901
    r = math.hypot(x[0], x[1])
    lat = math.atan2(x[2], r * (1 - e2))
    h = 0.0
    for _ in range(6):
        n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        h = r / math.cos(lat) - n
        lat = math.atan2(x[2], r * (1 - e2 * n / (n + h)))
    return h


def glm_curve(path):
    """A GLM bolide light curve from the bolide database's CSV: times (s), latitudes, longitudes, energies and the satellite's longitude; rows without times are the 2 ms frames."""
    import datetime as _dt
    rows, sat_lon = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("#Satellite"):
            sat_lon = float(line.split(",")[1].split("/")[1])
        parts = line.strip().split(",")
        if len(parts) == 4 and parts[1].lstrip("-").replace(".", "").isdigit():
            rows.append(parts)
    if rows and rows[0][0].isdigit():
        t = np.array([float(r[0]) / 1000.0 for r in rows])
    else:
        t0 = _dt.datetime.strptime(rows[0][0].split(" (")[0], "%a %b %d %Y %H:%M:%S GMT%z").timestamp()
        t = t0 + 0.002 * np.arange(len(rows))
    return dict(t=t, lon=np.array([float(r[1]) for r in rows]), lat=np.array([float(r[2]) for r in rows]), e=np.array([float(r[3]) for r in rows]), sat=sat_lon)


def glm_stereo(path_a, path_b, frac=0.1):
    """A fireball's Earth-fixed velocity from two GLM views: rays from each satellite through its frame's position on the lightning ellipsoid, triangulated at common times and fitted with a line, weighted by light."""
    a, b = glm_curve(path_a), glm_curve(path_b)
    ta = {round(x, 3): k for k, x in enumerate(a["t"])}
    sa, sb = glm_sat(a["sat"]), glm_sat(b["sat"])
    pts, ts, ws, miss = [], [], [], []
    for kb, t in enumerate(b["t"]):
        ka = ta.get(round(t, 3))
        if ka is None or a["e"][ka] < frac * a["e"].max() or b["e"][kb] < frac * b["e"].max():
            continue
        p1, p2 = glm_ell_point(a["lat"][ka], a["lon"][ka]), glm_ell_point(b["lat"][kb], b["lon"][kb])
        d1, d2 = (p1 - sa) / np.linalg.norm(p1 - sa), (p2 - sb) / np.linalg.norm(p2 - sb)
        w0 = sa - sb
        aa, bb, cc, dd, ee = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w0, d2 @ w0
        den = aa * cc - bb * bb
        u, v = (bb * ee - cc * dd) / den, (aa * ee - bb * dd) / den
        q1, q2 = sa + u * d1, sb + v * d2
        pts.append((q1 + q2) / 2)
        miss.append(float(np.linalg.norm(q1 - q2)))
        ts.append(t)
        ws.append(math.sqrt(a["e"][ka] * b["e"][kb]))
    if len(ts) < 20:
        return None
    pts, ts, ws = np.array(pts), np.array(ts), np.array(ws)
    x = np.stack([np.ones_like(ts), ts - ts.mean()], 1)
    w = np.sqrt(ws / ws.max())
    coef = np.linalg.lstsq(x * w[:, None], pts * w[:, None], rcond=None)[0]
    hs = [glm_height(q) for q in pts]
    return dict(v=coef[1], n=len(ts), span=float(ts.max() - ts.min()), miss=float(np.median(miss)), h=(float(max(hs)), float(min(hs))))


fireballglm_DOC = r"""Fireballs seen twice: every CNEOS fireball since the first GOES lightning mapper began (2017 July 23) inside the mappers' view (longitude 180 W to 5 W, latitude within 54 deg), matched to the NASA GLM bolide database within 60 s and 300 km, the match rate measured on all of them and the fireballs faster than the Sun's escape speed read against it, with each match's GLM duration, beginning altitude and detecting satellites; an independent sighting fixes a fireball's time and place. Where two satellites saw it, the two views are triangulated frame by frame into a trail whose fitted velocity is set against the CNEOS vector on every such fireball, which measures how well the mappers' light centroids follow a fireball's motion before the fast ones are read by it.

    python -I seti.py solar_system fireballglm [record] [--dt-s 60] [--km 300]"""


def fireballglm_main(record=False, dt_s=60.0, km=300.0):
    import datetime as _dt
    glm = json.loads(core.get("https://neo-bolide.ndc.nasa.gov/service/event/public", timeout=300))["data"]
    gt = np.array([g["datetime"] / 1000.0 for g in glm])
    order = np.argsort(gt)
    gt = gt[order]
    glm = [glm[k] for k in order]
    fb = json.load(open(core.results("cneos_fireballs.json")))
    start = _dt.datetime(2017, 7, 23, tzinfo=_dt.timezone.utc).timestamp()
    seen, out = [], []
    for f in fb:
        t = _dt.datetime.fromisoformat(f["date"].replace(" ", "T")).replace(tzinfo=_dt.timezone.utc).timestamp()
        if t < start or not (-180 <= f["lon"] <= -5) or abs(f["lat"]) > 54:
            continue
        lo, hi = np.searchsorted(gt, t - dt_s), np.searchsorted(gt, t + dt_s)
        best = None
        for k in range(lo, hi):
            g = glm[k]
            d = 6371.0 * math.radians(core.sep_arcsec(f["lon"], f["lat"], g["longitude"], g["latitude"]) / 3600.0)
            if d < km and (best is None or d < best[1]):
                best = (g, d, gt[k] - t)
        o = dict(date=f["date"], lat=f["lat"], lon=f["lon"], vel=f["vel"], v_helio=f["v_helio"], excess=f["excess"], matched=best is not None)
        if best:
            g = best[0]
            o.update(glm_id=g["_id"], km=best[1], dt=best[2], by=g.get("detectedBy"), duration=g.get("duration"), h0=g.get("beginningAltitude"), conf=g.get("confidenceRating"))
        out.append(o)
    rate = sum(o["matched"] for o in out) / max(len(out), 1)
    fast = [o for o in out if o["excess"] > 0]
    byid = {g["_id"]: g for g in glm}
    api = json.loads(core.get("https://ssd-api.jpl.nasa.gov/fireball.api?req-loc=true&req-vel=true&vel-comp=true", timeout=120))
    vec = {r[0]: np.array([float(r[9]), float(r[10]), float(r[11])]) for r in api["data"] if r[9]}
    os.makedirs(core.data("glm"), exist_ok=True)
    for o in out:
        if not o["matched"]:
            continue
        paths = {}
        for c in byid[o["glm_id"]].get("csv") or []:
            for sname in ("G16", "G17", "G18", "G19"):
                if "_%s_" % sname in c and sname not in paths:
                    paths[sname] = c
        if len(paths) < 2:
            continue
        local = []
        for sname in sorted(paths)[:2]:
            lp = core.data("glm", os.path.basename(paths[sname]))
            if not os.path.exists(lp):
                open(lp, "wb").write(core.get("https://neo-bolide.ndc.nasa.gov" + paths[sname], timeout=120))
            local.append(lp)
        st = glm_stereo(*local)
        if st is None or o["date"] not in vec:
            continue
        vc = vec[o["date"]]
        o["stereo"] = dict(speed=float(np.linalg.norm(st["v"])), ratio=float(np.linalg.norm(st["v"]) / np.linalg.norm(vc)), angle=math.degrees(math.acos(max(-1.0, min(1.0, float(
            vc @ st["v"] / np.linalg.norm(vc) / np.linalg.norm(st["v"])))))), n=st["n"], span=st["span"], miss=st["miss"], h=st["h"])
    ster = [o for o in out if o.get("stereo")]
    ratios = np.array([o["stereo"]["ratio"] for o in ster if o["excess"] <= 0])
    print("%d CNEOS fireballs inside the mappers' view since 2017 July 23; %d matched to a GLM bolide within %g s and %g km (%.2f); %d seen in stereo, whose GLM trail speeds run at %.2f of "
          "the CNEOS speeds at the median and %.2f-%.2f between the 16th and 84th percentiles" % (len(out), sum(o["matched"] for o in out), dt_s, km, rate, len(ster), float(np.median(ratios)),
                                                                                                 float(np.percentile(ratios, 16)), float(np.percentile(ratios, 84))))
    for o in ster:
        print("  stereo %s: CNEOS %.1f km/s, GLM %.1f km/s between %.0f and %.0f km over %.2f s, ratio %.2f, %.0f deg apart" % (
            o["date"], o["vel"], o["stereo"]["speed"], o["stereo"]["h"][0], o["stereo"]["h"][1], o["stereo"]["span"], o["stereo"]["ratio"], o["stereo"]["angle"]))
    for o in sorted(out, key=lambda o: -o["excess"])[:6]:
        print("  %s (%.1f, %.1f) %.1f km/s, heliocentric %.1f (+%.1f): %s" % (o["date"], o["lat"], o["lon"], o["vel"], o["v_helio"], o["excess"],
              "GLM %s, %.0f km and %+.1f s away, duration %s s, beginning height %s km, %s confidence" % (o["by"], o["km"], o["dt"], o["duration"], o["h0"], o["conf"]) if o["matched"] else "no GLM bolide"))
    json.dump(out, open(core.results("cneos_glm_matches.json"), "w"), indent=1)
    if record:
        core.record_run(
            "cneos_glm_bolides", "Fireballs seen twice", "CNEOS fireball database (JPL fireball API); NASA GLM bolide database (neo-bolide.ndc.nasa.gov)",
            "CNEOS fireballs since 2017 July 23 inside the GOES lightning mappers' view: %d" % len(out), os.path.abspath(__file__), "fireballglm", len(out), "fireballs", dict(dt_s=dt_s, km=km),
            "the match rate of all the fireballs in view: %.2f; GLM stereo trail speeds over CNEOS speeds on %d slower fireballs: median %.2f, 16-84 per cent %.2f-%.2f" % (
                rate, len(ratios), float(np.median(ratios)), float(np.percentile(ratios, 16)), float(np.percentile(ratios, 84))),
            "an independent sighting of a fireball faster than the Sun's escape speed, and its trail speed",
            "%d of %d matched; %s" % (sum(o["matched"] for o in out), len(out), "; ".join("%s %s%s" % (o["date"], "matched" if o["matched"] else "unmatched", (
                ", stereo trail speed %.1f km/s, %.2f of the CNEOS speed" % (o["stereo"]["speed"], o["stereo"]["ratio"])) if o.get("stereo") else "") for o in fast)),
            [dict(target="fireball:%s" % o["date"].replace(" ", "T"), stat=round(o["excess"], 2), unit="km/s above the Sun's escape speed", vet=None,
                  note=("GLM %s, %.0f km, %+.1f s, duration %s s, beginning height %s km" % (o["by"], o["km"], o["dt"], o["duration"], o["h0"])) if o["matched"] else "no GLM bolide")
             for o in fast])
        print("recorded")


fireballgmn_DOC = r"""Fireballs the meteor cameras caught: every CNEOS fireball since 2018 against the Global Meteor Network's trajectories (the yearly summaries, 2018-2026), a camera trajectory beginning within 10 s and 300 km of the fireball's peak, its initial speed set against the CNEOS speed on every match, and the fireballs faster than the Sun's escape speed read against that calibration; a ground network's speed is measured from triangulated video and settles a satellite speed it disagrees with.

    python -I seti.py solar_system fireballgmn [record] [--dt-s 10] [--km 300]"""


def fireballgmn_main(record=False, dt_s=10.0, km=300.0):
    from astropy.time import Time
    fb = json.load(open(core.results("cneos_fireballs.json")))
    jd = np.array([Time(f["date"].replace(" ", "T"), scale="utc").jd for f in fb])
    order = np.argsort(jd)
    jd_s = jd[order]
    win = dt_s / 86400.0
    hits = {}
    for y in meteors_YEARS:
        p = os.path.join(meteors_ROOT, "traj_summary_yearly_%d.txt" % y)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            if line.startswith("#") or ";" not in line:
                continue
            c = line.split(";")
            if len(c) < 86:
                continue
            j = num(c[1])
            if j is None:
                continue
            k = np.searchsorted(jd_s, j - win)
            while k < len(jd_s) and jd_s[k] <= j + win:
                f = fb[order[k]]
                lat, lon = num(c[63]), num(c[65])
                if lat is not None and lon is not None:
                    d = 6371.0 * math.radians(core.sep_arcsec(f["lon"], f["lat"], lon, lat) / 3600.0)
                    if d < km:
                        hits.setdefault(int(order[k]), []).append(dict(id=c[0].strip(), dt=(j - jd[order[k]]) * 86400, km=d, vinit=num(c[59]), evinit=num(c[60]), vgeo=num(c[15]),
                                                                       e=num(c[25]), se=num(c[26]), hbeg=num(c[67]), hend=num(c[73]), nst=num(c[84]), qc=num(c[80])))
                k += 1
    out = []
    for k, f in enumerate(fb):
        if k in hits:
            h = min(hits[k], key=lambda h: abs(h["dt"]))
            out.append(dict(date=f["date"], vel=f["vel"], excess=f["excess"], gmn=h, ratio=(h["vinit"] / f["vel"]) if h["vinit"] else None))
    ratios = np.array([o["ratio"] for o in out if o["ratio"] and o["excess"] <= 0])
    fast = [f for f in fb if f["excess"] > 0]
    print("%d CNEOS fireballs; %d with a GMN trajectory within %g s and %g km; GMN initial speed over CNEOS speed on the bound ones: median %s, 16-84 per cent %s" % (
        len(fb), len(out), dt_s, km, "%.3f" % np.median(ratios) if len(ratios) else "-", "%.3f-%.3f" % (np.percentile(ratios, 16), np.percentile(ratios, 84)) if len(ratios) > 2 else "-"))
    for o in out:
        print("  %s: CNEOS %.1f km/s; GMN %s %+.1f s and %.0f km away, initial %.2f +- %s km/s, e %s +- %s, %s to %s km, %s stations, convergence %s deg; ratio %.3f%s" % (
            o["date"], o["vel"], o["gmn"]["id"], o["gmn"]["dt"], o["gmn"]["km"], o["gmn"]["vinit"] or 0, o["gmn"]["evinit"], o["gmn"]["e"], o["gmn"]["se"], o["gmn"]["hbeg"], o["gmn"]["hend"],
            o["gmn"]["nst"], o["gmn"]["qc"], o["ratio"] or 0, "; faster than escape by CNEOS" if o["excess"] > 0 else ""))
    caught = {o["date"] for o in out}
    print("fast fireballs caught by GMN: %s" % [(f["date"], f["date"] in caught) for f in fast])
    json.dump(out, open(core.results("cneos_gmn_matches.json"), "w"), indent=1)
    if record:
        core.record_run(
            "cneos_gmn_trajectories", "Fireballs the meteor cameras caught", "CNEOS fireball database; Global Meteor Network trajectory summaries 2018-2026",
            "CNEOS fireballs with velocity vectors: %d" % len(fb), os.path.abspath(__file__), "fireballgmn", len(fb), "fireballs", dict(dt_s=dt_s, km=km),
            "GMN initial speed over CNEOS speed on %d bound fireballs: median %s" % (len(ratios), "%.3f" % np.median(ratios) if len(ratios) else "-"),
            "a camera trajectory of a fireball faster than the Sun's escape speed",
            "%d fireballs caught by GMN; of the %d fast ones, %s" % (len(out), len(fast), "; ".join("%s %s" % (f["date"], "caught" if f["date"] in caught else "not caught") for f in fast)),
            [dict(target="fireball:%s" % o["date"].replace(" ", "T"), stat=round(o["ratio"], 3) if o["ratio"] else None, unit="GMN initial speed over the CNEOS speed",
                  vet=None, note="GMN %s, %.1f km/s" % (o["gmn"]["id"], o["gmn"]["vinit"] or 0)) for o in out if o["excess"] > 0])
        print("recorded")


vanished_DOC = r"""Vanished numbered asteroids: Small-Body Database numbered asteroids brighter than absolute magnitude 17 with no observation since a cut-off year, each one's present distance from the Sun found by propagating its catalogued orbit to today with Kepler's equation and its present magnitude at opposition, V = H + 5 log10(r (r - 1)), set against the V 21.5 that the wide surveys reach, and every body that estimate leaves within reach looked up in JPL Horizons for its apparent magnitude and declination tonight; a numbered body bright enough to be seen that no survey has seen would have left its orbit, and one now fainter than the surveys reach is out of their sight by its orbit alone.

    python -I seti.py solar_system vanished [record] [--h-max 17] [--last-before 2017-01-01] [--v-limit 21.5]"""


# bodies whose ephemerides come from the planetary solution, so that their small-body orbits are not refit with new observations
VANISHED_PLANETARY = {"134340": "Pluto, whose ephemeris comes from the planetary solution, so its small-body orbit is not refit"}


def horizons_now(target):
    """Apparent V magnitude and declination of a small body tonight from JPL Horizons, or None."""
    import urllib.parse
    from astropy.time import Time
    day = Time.now().iso[:10]
    q = dict(format="json", COMMAND="'%s;'" % target, EPHEM_TYPE="OBSERVER", CENTER="'500@399'", START_TIME="'%s'" % day, STOP_TIME="'%s 01:00'" % day, STEP_SIZE="'1h'",
             QUANTITIES="'1,9'", ANG_FORMAT="'DEG'")
    try:
        r = json.loads(core.get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), timeout=60))["result"]
        row = r[r.index("$$SOE") + 5:r.index("$$EOE")].strip().splitlines()[0].split()
        return float(row[4]), float(row[3])
    except Exception:
        return None


def vanished_main(record=False, h_max=17.0, last_before="2017-01-01", v_limit=21.5):
    from astropy.time import Time
    rows = sbdb_rows(["H|LT|%g" % h_max, "last_obs|LT|%s" % last_before], "full_name,pdes,class,H,a,e,q,ma,epoch,n,first_obs,last_obs", kind="a")
    rows = [r for r in rows if str(r["pdes"]).strip().isdigit()]
    now = Time.now().jd
    out = []
    for r in rows:
        a, e, ma, ep, n, h = (num(r[k]) for k in ("a", "e", "ma", "epoch", "n", "H"))
        if None in (a, e, ma, ep, n, h) or e >= 1:
            continue
        m = math.radians((ma + n * (now - ep)) % 360.0)
        ecc = m
        for _ in range(50):
            ecc = ecc - (ecc - e * math.sin(ecc) - m) / (1 - e * math.cos(ecc))
        rr = a * (1 - e * math.cos(ecc))
        v = h + 5 * math.log10(rr * max(rr - 1, 0.05))
        out.append(dict(name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), cls=str(r["class"]).strip(), H=h, a=a, e=e, q=num(r["q"]), r_now=rr, v_now=v,
                        last=r["last_obs"], vet="out-of-reach" if v > v_limit else None))
    for o in out:
        if o["vet"] is None:
            if o["pdes"] in VANISHED_PLANETARY:
                o["vet"], o["why"] = "artefact", VANISHED_PLANETARY[o["pdes"]]
                continue
            hz = horizons_now(o["pdes"])
            if hz:
                o["v_horizons"], o["dec_now"] = hz
                if hz[0] > v_limit:
                    o["vet"] = "out-of-reach"
    seen = [o for o in out if o["vet"] is None]
    classes = {}
    for o in out:
        classes[o["cls"]] = classes.get(o["cls"], 0) + 1
    print("%d numbered asteroids with H below %g unobserved since %s, by class %s; beyond V %g tonight or the catalogue's own artefact: %d; within the surveys' reach: %d" % (
        len(out), h_max, last_before, classes, v_limit, len(out) - len(seen), len(seen)))
    for o in sorted(out, key=lambda o: o["v_now"])[:12]:
        print("  %s (%s): H %.1f, a %.2f au, e %.2f, now %.1f au from the Sun, V %.1f at opposition%s, last seen %s; %s" % (
            o["name"], o["cls"], o["H"], o["a"], o["e"], o["r_now"], o["v_now"], ", V %.1f at declination %+.0f tonight" % (o["v_horizons"], o["dec_now"]) if o.get("v_horizons") else "",
            o["last"], o["vet"] or "within reach"))
    json.dump(out, open(core.results("vanished_asteroids.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_vanished_numbered", "Vanished numbered asteroids", "JPL Small-Body Database (sbdb_query.api)",
            "numbered asteroids with H below %g and no observation since %s: %d" % (h_max, last_before, len(out)), os.path.abspath(__file__), "vanished", len(out), "asteroids",
            dict(h_max=h_max, last_before=last_before, v_limit=v_limit), "each orbit propagated to today; the wide surveys' reach of V %g" % v_limit,
            "a numbered asteroid within the surveys' reach that no survey has seen", "%d by class %s; %d now beyond V %g or the catalogue's own artefact, %d within reach" % (
                len(out), classes, len(out) - len(seen), v_limit, len(seen)),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["v_now"], 1), unit="V at opposition now", vet=o["vet"], note="%s, %s, %.1f au from the Sun, last seen %s%s" % (
                o["name"], o["cls"], o["r_now"], o["last"], "; " + o["why"] if o.get("why") else "; V %.1f tonight" % o["v_horizons"] if o.get("v_horizons") else "")) for o in out])
        print("recorded")


lurkers_DOC = r"""Lurkers at the other planets' Lagrange points: asteroids and comets of JPL's Small-Body Database within 2 per cent of the semi-major axis of Venus, Mars, Saturn, Uranus or Neptune, or within one and a half times the half-width of the planet's 1:1 resonance in semi-major axis, a sqrt(8 mu / 3), where that is wider (Saturn's), with eccentricities below 0.3 and inclinations below 40 deg, placed by their mean longitude at the element epoch against the planet's (Standish's mean elements for 1800-2050), the count within 20 deg of the L4 and L5 points, 60 deg ahead of and behind the planet, set against the ring's other longitudes; a watcher kept at a planet's stable points would sit there, where the 1:1 resonance holds natural Trojans as well, so a match on an arc of 30 days or more inside the resonance's half-width is a co-orbital of the kind the planet's published Trojans are, and one outside it a body passing through on its own orbit.

    python -I seti.py solar_system lurkers [record] [--frac 0.02] [--e-max 0.3] [--i-max-deg 40] [--half-width-deg 20]"""

# mean longitude at J2000 (deg), its rate (deg per Julian century), the semi-major axis (au) and the planet's mass over the Sun's:
# Standish's approximate elements, 1800-2050
LURKERS_PLANETS = {"Venus": (181.97909950, 58517.81538729, 0.72333566, 2.4478e-6), "Mars": (-4.55343205, 19140.30268499, 1.52371034, 3.2271e-7),
                   "Saturn": (49.95424423, 1222.49362201, 9.53667594, 2.8589e-4), "Uranus": (313.23810451, 428.48202785, 19.18916464, 4.3662e-5),
                   "Neptune": (-55.12002969, 218.45945325, 30.06992276, 5.1514e-5)}
# the published Trojan populations the co-orbitals belong to
LURKERS_TROJANS = {"Venus": "temporary Trojans (2013 ND15, de la Fuente Marcos and de la Fuente Marcos 2014)",
                   "Mars": "the Mars Trojans and co-orbitals, Eureka's family at L5 among them (Christou 2013; Borisov et al. 2017)",
                   "Saturn": "temporary Trojans (2019 UO14, Hui et al. 2024)",
                   "Uranus": "temporary Trojans (2011 QF99, Alexandersen et al. 2013; 2014 YX49, de la Fuente Marcos and de la Fuente Marcos 2017)",
                   "Neptune": "the Neptune Trojans (Sheppard and Trujillo 2006; Lin et al. 2021)"}


def lurkers_main(record=False, frac=0.02, e_max=0.3, i_max_deg=40.0, half_width_deg=20.0):
    from scipy.stats import poisson
    found, summary = [], {}
    for planet, (l0, rate, ap, mu) in LURKERS_PLANETS.items():
        width = ap * math.sqrt(8 * mu / 3)
        span = max(frac * ap, 1.5 * width)
        rows = []
        for kind in ("a", "c"):
            rows += sbdb_rows(["a|RG|%g|%g" % (ap - span, ap + span), "e|LT|%g" % e_max, "i|LT|%g" % i_max_deg],
                              "full_name,pdes,a,e,i,om,w,ma,epoch,H,data_arc,n_obs_used,condition_code,class", kind=kind)
        ring = []
        for r in rows:
            om, w, ma, ep = num(r["om"]), num(r["w"]), num(r["ma"]), num(r["epoch"])
            if None in (om, w, ma, ep):
                continue
            lam = (om + w + ma) % 360.0
            lp = (l0 + rate * (ep - 2451545.0) / 36525.0) % 360.0
            ring.append(dict(planet=planet, name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), a=num(r["a"]), e=num(r["e"]), i=num(r["i"]),
                             H=num(r["H"]), arc=num(r["data_arc"]) or 0, nobs=num(r["n_obs_used"]), cc=num(r["condition_code"]), cls=str(r["class"]).strip(),
                             dlon=(lam - lp + 180.0) % 360.0 - 180.0))
        l4 = [o for o in ring if abs(o["dlon"] - 60) < half_width_deg]
        l5 = [o for o in ring if abs(o["dlon"] + 60) < half_width_deg]
        exp = len(ring) * 2 * half_width_deg / 360.0
        p4, p5 = (float(poisson.sf(len(x) - 1, exp)) if x else 1.0 for x in (l4, l5))
        hist = [sum(1 for o in ring if lo <= o["dlon"] < lo + 30) for lo in range(-180, 180, 30)]
        for o in l4 + l5:
            o["point"] = "L4" if o in l4 else "L5"
            o["inside"] = abs(o["a"] - ap) < width
            o["vet"] = ("unconstrained-geometry" if (o["cc"] is not None and o["cc"] >= 6) or o["arc"] < 30 else "co-orbital" if o["inside"] else "chance-rate")
            o["ref"] = LURKERS_TROJANS[planet] if o["vet"] == "co-orbital" else ""
        co = [o for o in l4 + l5 if o["vet"] == "co-orbital"]
        passing = [o for o in l4 + l5 if o["vet"] == "chance-rate"]
        exp_pass = sum(1 for o in ring if abs(o["a"] - ap) >= width) * 2 * half_width_deg / 360.0
        summary[planet] = dict(n=len(ring), span=span, width=width, l4=len(l4), l5=len(l5), exp=exp, p4=p4, p5=p5, hist=hist, co=len(co), passing=len(passing), exp_pass=exp_pass)
        found += l4 + l5
        print("%s: %d bodies within %.4f au of its semi-major axis (the 1:1 resonance's half-width %.4f au); within %g deg of L4: %d, of L5: %d, against %.1f expected per "
              "point from the ring (Poisson p %.3g and %.3g); longitudes from the planet in 30 deg bins from -180: %s; on arcs of 30 d or more inside the resonance: %d, "
              "outside it passing through: %d against %.1f per point" % (planet, len(ring), span, width, half_width_deg, len(l4), len(l5), exp, p4, p5, hist, len(co),
                                                                          len(passing), exp_pass))
        for o in sorted(l4 + l5, key=lambda o: abs(abs(o["dlon"]) - 60)):
            if o["vet"] != "unconstrained-geometry":
                print("  %s %s (%s): %+.1f deg from %s, a %.4f (%+.4f), e %.3f, i %.1f, H %s, arc %.0f d, condition %s; %s" % (
                    o["point"], o["name"], o["cls"], o["dlon"], planet, o["a"], o["a"] - ap, o["e"], o["i"], o["H"], o["arc"], o["cc"], o["vet"] or "unexplained"))
    left = [o for o in found if o["vet"] is None]
    print("unexplained at any of the five planets' L4 and L5 points: %d" % len(left))
    json.dump(dict(summary=summary, found=found), open(core.results("planet_lagrange_points.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_planet_lagrange", "Lurkers at the other planets' Lagrange points", "JPL Small-Body Database (sbdb_query.api)",
            "bodies within 2 per cent of the semi-major axis of Venus, Mars, Saturn, Uranus and Neptune, or 1.5 times the 1:1 resonance's half-width, with e below %g and "
            "i below %g deg: %s" % (e_max, i_max_deg, ", ".join("%s %d" % (k, v["n"]) for k, v in summary.items())), os.path.abspath(__file__), "lurkers",
            sum(v["n"] for v in summary.values()), "bodies", dict(frac=frac, e_max=e_max, i_max_deg=i_max_deg, half_width_deg=half_width_deg),
            "each ring's other longitudes: %s expected per point" % ", ".join("%s %.1f" % (k, v["exp"]) for k, v in summary.items()),
            "a body within %g deg of a planet's L4 or L5 point on an orbit the catalogue holds" % half_width_deg,
            "; ".join("%s %d at L4 and %d at L5 against %.1f, %d of them co-orbital" % (k, v["l4"], v["l5"], v["exp"], v["co"]) for k, v in summary.items())
            + "; %d unexplained" % len(left),
            [dict(target="sbdb:%s" % o["pdes"], stat=round(o["dlon"], 1), unit="deg of mean longitude from %s" % o["planet"], vet=o["vet"],
                  note="%s at %s's %s, a %.4f au, e %.3f, i %.1f deg, H %s, arc %.0f d, condition %s%s" % (
                      o["name"], o["planet"], o["point"], o["a"], o["e"], o["i"], o["H"], o["arc"], o["cc"], "; " + o["ref"] if o["ref"] else "")) for o in found])
        print("recorded")



hohmann_DOC = r"""Bodies parked on Hohmann transfer orbits: asteroids and comets of JPL's Small-Body Database whose perihelion lies within 0.005 au of one planet's semi-major axis and whose aphelion lies within 0.005 au of the next planet's, Earth to Mars and Venus to Earth, on orbits inclined less than 2 deg, the least-energy route between the two that a craft plying between them would keep, against the counts of the 0.01 by 0.01 au cells of perihelion and aphelion around it, each match read for its arc, condition code and class.

    python -I seti.py solar_system hohmann [record] [--tol-au 0.005] [--i-max-deg 2] [--cells 4]"""

# perihelion and aphelion of the transfer: the inner and outer planet's semi-major axes in au
HOHMANN_PAIRS = {"Earth to Mars": (1.00000261, 1.52371034), "Venus to Earth": (0.72333566, 1.00000261)}


def hohmann_main(record=False, tol_au=0.005, i_max_deg=2.0, cells=4):
    from scipy.stats import poisson
    w = 2 * tol_au
    found, summary = [], {}
    for pair, (q0, Q0) in HOHMANN_PAIRS.items():
        span = (cells + 0.5) * w
        rows = []
        for kind in ("a", "c"):
            rows += sbdb_rows(["q|RG|%g|%g" % (q0 - span, q0 + span), "a|RG|%g|%g" % ((q0 + Q0) / 2 - span, (q0 + Q0) / 2 + span), "i|LT|%g" % i_max_deg],
                              "full_name,pdes,a,e,i,q,H,data_arc,n_obs_used,condition_code,class", kind=kind)
        grid = np.zeros((2 * cells + 1, 2 * cells + 1), int)
        hits = []
        for r in rows:
            a, q = num(r["a"]), num(r["q"])
            if a is None or q is None:
                continue
            Q = 2 * a - q
            iq, iQ = int(math.floor((q - q0 + tol_au) / w)), int(math.floor((Q - Q0 + tol_au) / w))
            if abs(iq) <= cells and abs(iQ) <= cells:
                grid[iq + cells, iQ + cells] += 1
                if iq == 0 and iQ == 0:
                    cc = num(r["condition_code"])
                    hits.append(dict(pair=pair, name=" ".join(str(r["full_name"]).split()), pdes=str(r["pdes"]).strip(), q=q, Q=Q, e=num(r["e"]), i=num(r["i"]),
                                     H=num(r["H"]), arc=num(r["data_arc"]) or 0, cc=cc, cls=str(r["class"]).strip()))
        centre = int(grid[cells, cells])
        others = np.delete(grid.ravel(), grid.size // 2)
        exp = float(np.mean(others))
        p = float(poisson.sf(centre - 1, exp)) if centre else 1.0
        for h in hits:
            h["vet"] = ("unconstrained-geometry" if (h["cc"] is not None and h["cc"] >= 6) or h["arc"] < 30 else "chance-rate" if p > 0.01 else None)
        summary[pair] = dict(n=int(grid.sum()), centre=centre, exp=exp, p=p, cells=[int(x) for x in others])
        found += hits
        print("%s: %d bodies inclined below %g deg in the %dx%d cells of %.2f au around perihelion %.4f and aphelion %.4f au; the transfer's cell holds %d against %.2f "
              "in the mean cell (Poisson p %.3g; cells range %d-%d)" % (pair, int(grid.sum()), i_max_deg, 2 * cells + 1, 2 * cells + 1, w, q0, Q0, centre, exp, p,
                                                                         min(others), max(others)))
        for h in sorted(hits, key=lambda h: -h["arc"]):
            print("  %s (%s): q %.4f, Q %.4f, e %.3f, i %.2f, H %s, arc %.0f d, condition %s; %s" % (
                h["name"], h["cls"], h["q"], h["Q"], h["e"], h["i"], h["H"], h["arc"], h["cc"], h["vet"] or "unexplained"))
    left = [h for h in found if h["vet"] is None]
    json.dump(dict(summary=summary, found=found), open(core.results("hohmann_orbits.json"), "w"), indent=1)
    if record:
        core.record_run(
            "sbdb_hohmann_orbits", "Bodies parked on Hohmann transfer orbits", "JPL Small-Body Database (sbdb_query.api)",
            "bodies inclined below %g deg in the cells around the Earth-Mars and Venus-Earth transfers: %s" % (
                i_max_deg, ", ".join("%s %d" % (k, v["n"]) for k, v in summary.items())), os.path.abspath(__file__), "hohmann", sum(v["n"] for v in summary.values()), "bodies",
            dict(tol_au=tol_au, i_max_deg=i_max_deg, cells=cells), "the mean of the surrounding %.2f au cells: %s" % (w, ", ".join("%s %.2f" % (k, v["exp"]) for k, v in summary.items())),
            "a body whose perihelion and aphelion lie within %g au of two planets' semi-major axes on an orbit inclined below %g deg" % (tol_au, i_max_deg),
            "; ".join("%s %d against %.2f (p %.3g)" % (k, v["centre"], v["exp"], v["p"]) for k, v in summary.items()) + "; %d unexplained" % len(left),
            [dict(target="sbdb:%s" % h["pdes"], stat=round(h["Q"] - h["q"], 4), unit="au from perihelion to aphelion", vet=h["vet"],
                  note="%s on the %s transfer, q %.4f, Q %.4f, i %.2f deg, H %s, arc %.0f d, condition %s" % (h["name"], h["pair"], h["q"], h["Q"], h["i"], h["H"], h["arc"], h["cc"]))
             for h in found])
        print("recorded")


firestrength_DOC = r"""Fireballs stronger than iron: every fireball of the CNEOS database (the JPL fireball API) with the height and speed of its peak brightness, its ram pressure there, the density of the 1976 US Standard Atmosphere at that height times the square of the speed, the load its strongest surviving fragments bore, which bounds the strength of the body that broke up by then from above; a pressure above 50 MPa, the strength Siraj and Loeb (2022) took for iron meteorites, is redrawn 20,000 times within the speed errors CNEOS shows against ground-based networks (Peña-Asensio et al. 2025: about 6 km/s before 2018, 0.55 km/s from 2018), one that stays above it in fewer than half the draws lies within the sensors' errors, and one that stays above it lies within the compressive strengths of meteorite samples in the laboratory while it stays below 411 MPa in most draws, Canyon Diablo's, the strongest Pohl and Britt (2020) compile (L chondrites reach 450 MPa in Popova et al. 2011); each is read for the height error that would bring it below 50 MPa.

    python -I seti.py solar_system firestrength [record] [--mpa 50] [--lab-mpa 411] [--draws 20000] [--p-above 0.5]"""

# the 1976 US Standard Atmosphere below 84.852 km geopotential: base height (km), base temperature (K), lapse rate (K/km) and base pressure (Pa)
US1976 = [(0.0, 288.15, -6.5, 101325.0), (11.0, 216.65, 0.0, 22632.06), (20.0, 216.65, 1.0, 5474.889), (32.0, 228.65, 2.8, 868.0187),
          (47.0, 270.65, 0.0, 110.9063), (51.0, 270.65, -2.8, 66.93887), (71.0, 214.65, -2.0, 3.956420)]


def us1976_density(h_km):
    """Air density in kg/m^3 at geometric height H_KM from the 1976 US Standard Atmosphere, extended above 86 km with the top layer's scale height."""
    hg = 6356.766 * h_km / (6356.766 + h_km)
    gmr = 34.1632
    base = [b for b in US1976 if b[0] <= hg][-1] if hg >= 0 else US1976[0]
    h0, t0, lapse, p0 = base
    t = t0 + lapse * (hg - h0)
    p = p0 * (t / t0) ** (-gmr / lapse) if lapse else p0 * math.exp(-gmr * (hg - h0) / t0)
    return p / (287.053 * t)


def firestrength_main(record=False, mpa=50.0, lab_mpa=411.0, draws=20000, p_above=0.5):
    d = json.loads(core.get("https://ssd-api.jpl.nasa.gov/fireball.api?req-alt=true&req-vel=true", timeout=120))
    f = d["fields"]
    rng = np.random.default_rng(31)
    out = []
    for row in d["data"]:
        r = dict(zip(f, row))
        h, v = num(r.get("alt")), num(r.get("vel"))
        if h is None or v is None:
            continue
        late = r["date"] >= "2018-01-01"
        sv = 0.55 if late else 6.0
        pram = us1976_density(h) * (v * 1e3) ** 2 / 1e6
        o = dict(date=r["date"], alt=h, vel=v, energy_kt=num(r.get("impact-e")), mpa=pram, era="from 2018" if late else "before 2018", sv=sv)
        if pram > mpa:
            sp = np.clip(v + sv * rng.standard_normal(draws), 0, None)
            pd = us1976_density(h) * (sp * 1e3) ** 2 / 1e6
            o["p_above"] = float(np.mean(pd > mpa))
            o["p_lab"] = float(np.mean(pd > lab_mpa))
            lo, hi = h, h + 40.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if us1976_density(mid) * (v * 1e3) ** 2 / 1e6 > mpa else (lo, mid)
            o["dh_km"] = hi - h
            o["vet"] = "sensor-error" if o["p_above"] < p_above else "meteoritic-strength" if o["p_lab"] < p_above else None
        out.append(o)
    strong = [o for o in out if o["mpa"] > mpa]
    left = [o for o in strong if o["vet"] is None]
    pr = np.array([o["mpa"] for o in out])
    print("%d fireballs with the height and speed of peak brightness; ram pressure there: median %.2f MPa, 90th percentile %.1f, 99th %.1f; above %g MPa: %d, of them above it in "
          "fewer than %g of the draws within CNEOS's speed errors: %d, below the %g MPa of meteorite samples in most draws: %d; beyond both: %d" % (
              len(out), float(np.median(pr)), float(np.percentile(pr, 90)), float(np.percentile(pr, 99)), mpa, len(strong), p_above,
              sum(o["vet"] == "sensor-error" for o in strong), lab_mpa, sum(o["vet"] == "meteoritic-strength" for o in strong), len(left)))
    for o in sorted(out, key=lambda o: -o["mpa"])[:15]:
        print("  %s: %.1f km at %.1f km/s, %.1f MPa%s, %s kt (%s); %s" % (
            o["date"], o["alt"], o["vel"], o["mpa"], ", above %g MPa in %.3f of the draws and above %g in %.3f, below %g MPa %.1f km higher" % (
                mpa, o["p_above"], lab_mpa, o["p_lab"], mpa, o["dh_km"]) if "p_above" in o else "",
            o["energy_kt"], o["era"], o.get("vet") or ("unexplained" if o["mpa"] > mpa else "below %g MPa" % mpa)))
    json.dump(out, open(core.results("fireball_strength.json"), "w"), indent=1)
    if record:
        core.record_run(
            "cneos_fireball_strength", "Fireballs stronger than iron", "CNEOS fireball database (JPL fireball API)",
            "fireballs with the height and speed of peak brightness: %d" % len(out), os.path.abspath(__file__), "firestrength", len(out), "fireballs",
            dict(mpa=mpa, lab_mpa=lab_mpa, draws=draws, p_above=p_above), "each strong one redrawn within CNEOS's speed errors (Peña-Asensio et al. 2025)",
            "a ram pressure at peak brightness above %g MPa, and above the %g MPa of meteorite samples, that the sensors' speed errors keep there" % (mpa, lab_mpa),
            "median %.2f MPa; %d above %g MPa: %d within the speed errors, %d within the strength of meteorite samples; %d beyond" % (
                float(np.median(pr)), len(strong), mpa, sum(o["vet"] == "sensor-error" for o in strong), sum(o["vet"] == "meteoritic-strength" for o in strong), len(left)),
            [dict(target="fireball:%s" % o["date"].replace(" ", "T"), stat=round(o["mpa"], 1), unit="MPa at peak brightness", vet=o["vet"],
                  note="%.1f km at %.1f km/s (%s), above %g MPa in %.3f of the draws and above %g in %.3f; %.1f km higher brings it below %g" % (
                      o["alt"], o["vel"], o["era"], mpa, o["p_above"], lab_mpa, o["p_lab"], o["dh_km"], mpa)) for o in strong])
        print("recorded")


circularmoons_DOC = r"""Moons parked on distant circular orbits: the satellites of JPL's mean-element table (ssd.jpl.nasa.gov/sats/elem) orbiting a giant planet beyond its outermost large moon (Callisto, Iapetus, Oberon, Triton), the irregular moons the planets captured, whose orbits come out eccentric and steeply inclined or retrograde, those with eccentricity below 0.05 on prograde orbits within 10 deg of their reference plane, a station kept far from the planet where capture leaves nothing round and flat, against the count the distant moons' eccentricities and inclinations give taken apart.

    python -I seti.py solar_system circularmoons [record] [--e-max 0.05] [--i-max-deg 10]"""

# the outermost large regular (or, at Neptune, captured) moon of each giant planet, whose semi-major axis in the table bounds the distant moons
CIRCULARMOONS_EDGE = {"Jupiter": "Callisto", "Saturn": "Iapetus", "Uranus": "Oberon", "Neptune": "Triton"}


def circularmoons_main(record=False, e_max=0.05, i_max_deg=10.0):
    import html
    t = core.get("https://ssd.jpl.nasa.gov/sats/elem/", timeout=120).decode("utf-8", "replace")
    body = t[t.index('id="sat_elem"'):]
    body = body[:body.index("</table>")]
    moons = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        c = [html.unescape(re.sub(r"<[^>]+>", " ", x)).split() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        c = [" ".join(x) for x in c]
        if len(c) < 13 or c[1] not in CIRCULARMOONS_EDGE:
            continue
        a, e, inc = num(c[7].rstrip(".")), num(c[8]), num(c[11])
        if None in (a, e, inc):
            continue
        moons.append(dict(planet=c[1], name=c[2], code=c[3], frame=c[5], a_km=a, e=e, i=inc, P_d=num(c[13])))
    edge = {p: max(m["a_km"] for m in moons if m["planet"] == p and m["name"] == name) for p, name in CIRCULARMOONS_EDGE.items()}
    distant = [m for m in moons if m["a_km"] > edge[m["planet"]]]
    circ = [m for m in distant if m["e"] < e_max]
    flat = [m for m in distant if m["i"] < i_max_deg]
    hits = [m for m in distant if m["e"] < e_max and m["i"] < i_max_deg]
    exp = len(circ) * len(flat) / len(distant) if distant else 0.0
    for m in hits:
        m["vet"] = None
    print("%d moons of the giant planets in the table, %d beyond each planet's outermost large moon (%s); eccentricity below %g: %d, prograde within %g deg of the reference "
          "plane: %d, both: %d against %.2f with the two taken apart" % (len(moons), len(distant), ", ".join("%s %d" % (p, sum(m["planet"] == p for m in distant))
                                                                                                           for p in CIRCULARMOONS_EDGE), e_max, len(circ), i_max_deg, len(flat),
                                                                         len(hits), exp))
    for m in sorted(distant, key=lambda m: m["e"])[:10]:
        print("  %s %s: a %.0f km (%.1f of %s's), e %.4f, i %.1f deg (%s); %s" % (
            m["planet"], m["name"], m["a_km"], m["a_km"] / edge[m["planet"]], CIRCULARMOONS_EDGE[m["planet"]], m["e"], m["i"], m["frame"],
            "circular and flat" if m in hits else "circular" if m["e"] < e_max else ""))
    for m in sorted(distant, key=lambda m: m["i"])[:6]:
        print("  lowest inclination: %s %s, i %.1f deg, e %.3f" % (m["planet"], m["name"], m["i"], m["e"]))
    json.dump(dict(moons=moons, hits=hits), open(core.results("circular_distant_moons.json"), "w"), indent=1)
    if record:
        core.record_run(
            "jpl_circular_distant_moons", "Moons parked on distant circular orbits", "JPL Solar System Dynamics planetary satellite mean elements",
            "moons of the giant planets beyond each one's outermost large moon: %d" % len(distant), os.path.abspath(__file__), "circularmoons", len(distant), "moons",
            dict(e_max=e_max, i_max_deg=i_max_deg), "the distant moons' eccentricities and inclinations taken apart: %.2f expected" % exp,
            "a distant moon on an orbit of eccentricity below %g within %g deg of its reference plane" % (e_max, i_max_deg),
            "%d against %.2f expected; eccentricity below %g alone: %d, inclination below %g deg alone: %d" % (len(hits), exp, e_max, len(circ), i_max_deg, len(flat)),
            [dict(target="moon:%s" % m["code"], stat=m["e"], unit="eccentricity", vet=m["vet"], note="%s of %s, a %.0f km, i %.1f deg (%s)" % (
                m["name"], m["planet"], m["a_km"], m["i"], m["frame"])) for m in hits])
        print("recorded")


APPROACHES = {
    "jupl12": (lambda: core.run_cli(jupl12_main), jupl12_DOC),
    "circularmoons": (lambda: core.run_cli(circularmoons_main), circularmoons_DOC),
    "firestrength": (lambda: core.run_cli(firestrength_main), firestrength_DOC),
    "hohmann": (lambda: core.run_cli(hohmann_main), hohmann_DOC),
    "lurkers": (lambda: core.run_cli(lurkers_main), lurkers_DOC),
    "vanished": (lambda: core.run_cli(vanished_main), vanished_DOC),
    "fireballgmn": (lambda: core.run_cli(fireballgmn_main), fireballgmn_DOC),
    "fireballglm": (lambda: core.run_cli(fireballglm_main), fireballglm_DOC),
    "deliveries": (lambda: core.run_cli(deliveries_main), deliveries_DOC),
    "fireballs": (lambda: core.run_cli(fireballs_main), fireballs_DOC),
    "spinsurvey": (lambda: core.run_cli(spinsurvey_main), spinsurvey_DOC),
    "lightbodies": (lambda: core.run_cli(lightbodies_main), lightbodies_DOC),
    "bluesurfaces": (lambda: core.run_cli(bluesurfaces_main), bluesurfaces_DOC),
    "outofplane": (lambda: core.run_cli(outofplane_main), outofplane_DOC),
    "retroco": (lambda: core.run_cli(retroco_main), retroco_DOC),
    "sunfocus": (lambda: core.run_cli(sunfocus_main), sunfocus_DOC),
    "tnocircles": (lambda: core.run_cli(tnocircles_main), tnocircles_DOC),
    "kirkwood": (lambda: core.run_cli(kirkwood_main), kirkwood_DOC),
    "aimed": (lambda: core.run_cli(aimed_main), aimed_DOC),
    "jupl3": (lambda: core.run_cli(jupl3_main), jupl3_DOC),
    "copycomets": (lambda: core.run_cli(copycomets_main), copycomets_DOC),
    "blackest": (lambda: core.run_cli(blackest_main), blackest_DOC),
    "hyperbolic": (lambda: core.run_cli(hyperbolic_main), hyperbolic_DOC),
    "reflspike": (lambda: core.run_cli(reflspike_main), reflspike_DOC),
    "pushed": (lambda: core.run_cli(pushed_main), pushed_DOC),
    "ecliptic": (lambda: core.run_cli(ecliptic_main), ecliptic_DOC),
    "giantcircles": (lambda: core.run_cli(giantcircles_main), giantcircles_DOC),
    "polar": (lambda: core.run_cli(polar_main), polar_DOC),
    "sunward": (lambda: core.run_cli(sunward_main), sunward_DOC),
    "lagrange": (lambda: core.run_cli(lagrange_main), lagrange_DOC),
    "hourspins": (lambda: core.run_cli(hourspins_main), hourspins_DOC),
    "tangent": (lambda: core.run_cli(tangent_main), tangent_DOC),
    "yearcomets": (lambda: core.run_cli(yearcomets_main), yearcomets_DOC),
    "circular": (lambda: core.run_cli(circular_main), circular_DOC),
    "mirror": (lambda: core.run_cli(mirror_main), mirror_DOC),
    "retrograde": (lambda: core.run_cli(retrograde_main), retrograde_DOC),
    "earthorbit": (lambda: core.run_cli(earthorbit_main), earthorbit_DOC),
    "iso3i": (iso3i_cli, iso3i_DOC),
    "isoarrival": (isoarrival_cli, isoarrival_DOC),
    "darkcomet": (darkcomet_cli, darkcomet_DOC),
    "replicated": (replicated_cli, replicated_DOC),
    "limbs": (limbs_cli, limbs_DOC),
    "plates": (plates_cli, plates_DOC),
    "phase": (lambda: core.run_cli(phase_main), phase_DOC),
    "spin": (lambda: core.run_cli(spin_main), spin_DOC),
    "meteors": (meteors_cli, meteors_DOC),
    "yarkovsky": (yarkovsky_cli, yarkovsky_DOC),
    "orbits": (lambda: core.run_cli(orbits_main), orbits_DOC),
    "hot": (lambda: core.run_cli(hot_main), hot_DOC),
    "nga": (nga_cli, nga_DOC),
    "surfaces": (surfaces_cli, surfaces_DOC),
    "radar_albedo": (radar_albedo_cli, radar_albedo_DOC),
}


