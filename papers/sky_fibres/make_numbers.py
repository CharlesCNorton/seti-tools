"""Numbers and galaxy table (table_galaxies.tex) of the sky-fibre paper from the line, counterpart, galaxy, injection and exposure files in data/."""
import collections
import csv
import gzip
import json
import math
import os
import re
import sys

import numpy as np
from astropy.coordinates import SkyCoord

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.argv = sys.argv[:1]
from setitools import core, lasers  # noqa: E402

DATA = os.path.join(HERE, "data")
LINE = {"[OII]3727": "[O\\,\\textsc{ii}]", "Hbeta": "H$\\beta$", "[OIII]4959": "[O\\,\\textsc{iii}]\\,4959",
        "[OIII]5007": "[O\\,\\textsc{iii}]\\,5007", "Halpha": "H$\\alpha$", "[NII]6583": "[N\\,\\textsc{ii}]",
        "[SII]6716": "[S\\,\\textsc{ii}]\\,6716", "[SII]6731": "[S\\,\\textsc{ii}]\\,6731", "Lyalpha": "Ly$\\alpha$"}
TYPE = {"REX": "round exp.", "PSF": "point", "EXP": "exp.", "SER": "S\\'ersic", "DEV": "de Vauc."}


def load():
    rows = []
    with gzip.open(os.path.join(DATA, "skyfibre_lines.csv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(plate=r["plate"], mjd=r["mjd"], file="spPlate-%s-%s.fits" % (r["plate"], r["mjd"]),
                             fiber=int(r["fiber"]), ra=float(r["ra"]), dec=float(r["dec"]), sign=int(r["sign"]),
                             lam=float(r["lam"]), snr=float(r["snr"]), width=float(r["width"]), flux=float(r["flux"]),
                             n_shared=int(r["n_shared"])))
    return rows


def classify(group):
    """Other-plate test (same fibre within three pixels at another sky position) and redshift test (two or more lines at one redshift, one strong) within one sign-and-rule group."""
    by_fibre, by_spec = collections.defaultdict(list), collections.defaultdict(list)
    for p in group:
        by_fibre[p["fiber"]].append(p)
        by_spec[(p["file"], p["fiber"])].append((p["lam"], p["flux"]))
    for p in group:
        p["fixed"] = sorted({q["plate"] for q in by_fibre[p["fiber"]] if q["plate"] != p["plate"]
                             and abs(math.log10(q["lam"] / p["lam"])) < 3e-4 and lasers.skyfibre_sep(p, q) > 1.5})
        p["galaxy"] = lasers.skyfibre_galaxy(by_spec[(p["file"], p["fiber"])])
        p["galaxy_loose"] = lasers.skyfibre_lines(sorted(x for x, _ in by_spec[(p["file"], p["fiber"])]))
    return group


def rest_of(group, thr=8):
    """Lines at thr sigma that the other-plate, redshift and two-sign tests leave."""
    return [p for p in group if p["snr"] >= thr and not p["fixed"] and not p["galaxy"] and not p["both"]]


def shuffle_rate(group, n=200, seed=1):
    """Galaxy identifications with line wavelengths shuffled among multi-line fibres, counts kept: the redshift test's chance rate."""
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for p in group:
        by[(p["file"], p["fiber"])].append((p["lam"], p["flux"]))
    multi = [k for k, v in by.items() if len(v) >= 2]
    lines = [x for k in multi for x in by[k]]
    sizes = [len(by[k]) for k in multi]
    out = []
    for _ in range(n):
        perm = [lines[i] for i in rng.permutation(len(lines))]
        i, hits = 0, 0
        for s in sizes:
            hits += lasers.skyfibre_galaxy(perm[i:i + s]) is not None
            i += s
        out.append(hits)
    return len(multi), float(np.mean(out)), float(np.std(out))


def main():
    rows = load()
    tot = json.load(open(os.path.join(DATA, "skyfibre_totals.json")))
    cp = json.load(open(os.path.join(DATA, "skyfibre_counterparts.json")))
    inj = json.load(open(os.path.join(DATA, "skyfibre_inject.json")))
    print("plates %d, files %d, sky spectra %d; all peaks at 6 sigma: %d emission-like, %d absorption-like"
          % (tot["n_plates"], tot["n_files"], tot["n_sky"], tot["pos_all"], tot["neg_all"]))
    area = tot["n_sky"] * math.pi * 1.0 ** 2 / 3600.0 ** 2
    print("sky area of the fibres: %.3f deg2" % area)
    groups = {(rule, sign): classify([dict(p) for p in rows if p["sign"] == sign and p["n_shared"] < rule])
              for rule in (1, 2) for sign in (1, -1)}
    for rule in (1, 2):
        lasers.skyfibre_both_signs(groups[(rule, 1)], groups[(rule, -1)])
    for (rule, sign), g in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        for thr in (6, 8):
            t = [p for p in g if p["snr"] >= thr]
            print("rule %d %s >= %d sigma: clean %d, fixed to a fibre %d, at a galaxy redshift %d, both signs %d, rest %d"
                  % (rule, "peaks" if sign > 0 else "dips ", thr, len(t), sum(1 for p in t if p["fixed"]),
                     sum(1 for p in t if p["galaxy"] and not p["fixed"]),
                     sum(1 for p in t if p["both"] and not p["galaxy"] and not p["fixed"]), len(rest_of(t, thr))))
    # counterpart test on the rest at 8 sigma
    print("counterparts within 1.5 arcsec (at the fibre / 30 arcsec north), lines at 8 sigma that every test leaves:")
    cpt = {}
    for rule in (1, 2):
        for sign in (1, -1):
            rest = rest_of(groups[(rule, sign)])
            keys = ["%.5f %+.5f" % (p["ra"], p["dec"]) for p in rest]
            have = [k for k in keys if k in cp]
            on = sum(1 for k in have if cp[k]["on"])
            off = sum(1 for k in have if cp[k]["off"])
            cpt[(rule, sign)] = (len(rest), len(have), on, off)
            print("  rule %d %s: %d lines, %d with the catalogue answering, %d with a source against %d at the control"
                  % (rule, "peaks" if sign > 0 else "dips ", len(rest), len(have), on, off))
    # peak excess over dips net of each sign's counterpart excess, and its bound
    print("excess of peaks over dips at 8 sigma, each net of its counterparts' excess over the control:")
    for rule in (1, 2):
        n_p, h_p, on_p, off_p = cpt[(rule, 1)]
        n_d, h_d, on_d, off_d = cpt[(rule, -1)]
        # counterpart excess scaled to every line of the sign, some lying outside the catalogue
        g_p = (on_p - off_p) * n_p / max(h_p, 1)
        g_d = (on_d - off_d) * n_d / max(h_d, 1)
        ex = (n_p - g_p) - (n_d - g_d)
        var = n_p + n_d + (on_p + off_p) * (n_p / max(h_p, 1)) ** 2 + (on_d + off_d) * (n_d / max(h_d, 1)) ** 2
        up = max(ex, 0.0) + 1.645 * math.sqrt(var)
        rec = {1: (0.301, 0.383), 2: (0.560, 0.741)}[rule]
        # recovery at 8/6 of threshold, between injections at 1.0 and 1.5
        r8 = rec[0] + (rec[1] - rec[0]) * (8.0 / 6.0 - 1.0) / 0.5
        print("  rule %d: peaks %d - %.0f = %.0f against dips %d - %.0f = %.0f: excess %+.0f +- %.0f; 95 per cent "
              "upper limit %.0f lines; recovery %.2f; fewer than %.0f per deg2"
              % (rule, n_p, g_p, n_p - g_p, n_d, g_d, n_d - g_d, ex, math.sqrt(var), up, r8, up / (area * r8)))
    exposures(groups, cp, area, rows)
    recurring(rows, groups, cp)
    # widths of the rest: narrow peaks against narrow dips
    for rule in (1, 2):
        pw = np.array([p["width"] for p in rest_of(groups[(rule, 1)])])
        dw = np.array([p["width"] for p in rest_of(groups[(rule, -1)])])
        print("  rule %d: widths below 0.9 px: %d peaks, %d dips; 1.1-1.7 px: %d peaks, %d dips"
              % (rule, (pw < 0.9).sum(), (dw < 0.9).sum(), ((pw >= 1.1) & (pw < 1.7)).sum(), ((dw >= 1.1) & (dw < 1.7)).sum()))
    # redshift test's chance rate, by shuffling and on dips
    for rule in (1, 2):
        nm, mu, sd = shuffle_rate([p for p in groups[(rule, 1)] if not p["fixed"]])
        dipfib = {(p["file"], p["fiber"]) for p in groups[(rule, -1)] if p["galaxy"] and not p["fixed"]}
        pkfib = {(p["file"], p["fiber"]) for p in groups[(rule, 1)] if p["galaxy"] and not p["fixed"]}
        print("rule %d: %d peak fibres at a galaxy redshift; shuffled among the %d fibres with two or more peaks: "
              "%.1f +- %.1f; dip fibres at a galaxy redshift: %d" % (rule, len(pkfib), nm, mu, sd, len(dipfib)))
    # identifications set aside by the ratio requirements, each sign
    for sign in (1, -1):
        g = groups[(2, sign)]
        loose = {(p["file"], p["fiber"]): p["galaxy_loose"] for p in g if p["galaxy_loose"] and not p["fixed"]}
        strict = {(p["file"], p["fiber"]) for p in g if p["galaxy"] and not p["fixed"]}
        aside = collections.Counter()
        for k, (z, lines) in loose.items():
            if k in strict:
                continue
            if "Halpha" in lines and not any(n in lines for n in ("[OII]3727", "Hbeta", "[OIII]5007")):
                aside["H alpha with [N II] or [S II] alone"] += 1
            elif "[OIII]4959" in lines and "[OIII]5007" not in lines:
                aside["[O III] 4959 without 5007"] += 1
            else:
                aside["[O III] at a flux ratio outside 1.5-5"] += 1
        print("two-fibre %s: set aside by the ratio requirements %s" % ("peaks" if sign > 0 else "dips", dict(aside)))
        if sign > 0:
            # set-aside fibres whose lines repeat at the same observed wavelengths on other plates
            sets = {k: sorted(p["lam"] for p in g if (p["file"], p["fiber"]) == k) for k in loose if k not in strict}
            keys = sorted(sets)
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    if a[0] != b[0]:
                        same = [x for x in sets[a] if any(abs(x - y) < 2.0 for y in sets[b])]
                        if len(same) >= 2:
                            print("    repeated: %s fibre %d and %s fibre %d at %s A" % (
                                a[0], a[1], b[0], b[1], ", ".join("%.0f" % x for x in same)))
    # strongest remaining lines and their counterparts
    cpd = json.load(open(os.path.join(DATA, "skyfibre_counterparts.json")))
    rest = sorted(rest_of(groups[(2, 1)]), key=lambda p: -p["snr"])
    for p in rest[:6]:
        c = cpd.get("%.5f %+.5f" % (p["ra"], p["dec"]))
        print("  strongest rest %s fibre %d %.1f A S/N %.1f width %.2f px flux %.1e; LS %s" % (
            p["file"], p["fiber"], p["lam"], p["snr"], p["width"], p["flux"] * 1e-17,
            "not covered" if c is None else ", ".join("%s r %s" % (x["type"], x["mag_r"]) for x in c["on"]) or "none"))
    # lines fixed to a fibre: where they cluster
    fx = [p for p in groups[(2, 1)] if p["fixed"]]
    grp = collections.Counter((p["fiber"], int(round(p["lam"] / 20) * 20)) for p in fx)
    print("fixed peaks (two-fibre rule): %d, %d at 8 sigma, in %d fibres; largest groups %s"
          % (len(fx), sum(1 for p in fx if p["snr"] >= 8), len({p["fiber"] for p in fx}), grp.most_common(6)))
    # sensitivity: 6-sigma line flux from each clean peak's flux and S/N
    pk = [p for p in groups[(2, 1)]]
    thr = np.array([6 * p["flux"] / p["snr"] for p in pk]) * 1e-17
    lam = np.array([p["lam"] for p in pk])
    print("6 sigma line flux: median %.2e, 10-90 per cent %.2e-%.2e; at 8 sigma %.2e"
          % (np.median(thr), *np.percentile(thr, [10, 90]), np.median(thr) * 8 / 6))
    for a, b in ((3600, 5000), (5000, 7000), (7000, 9000), (9000, 10400)):
        m = (lam >= a) & (lam < b)
        print("  %d-%d A: median %.1e (n %d)" % (a, b, np.median(thr[m]), m.sum()))
    pc = 3.0857e18
    lam_med = float(np.median(lam))
    for f, label in ((np.median(thr), "6 sigma"), (np.median(thr) * 8 / 6, "8 sigma")):
        for dpc in (10, 100, 1000):
            dcm = dpc * pc
            # diffraction-limited beam of solid angle (lambda/D)^2, D = 10 m
            print("  %s, %5d pc: isotropic %.1e W; through 10 m at %.0f A %.1e W"
                  % (label, dpc, 4 * math.pi * dcm ** 2 * f * 1e-7, lam_med, f * dcm ** 2 * (lam_med * 1e-8 / 1000.0) ** 2 * 1e-7))
    print("injections: " + ", ".join("%s %.2f" % (k, v["recovered"]) for k, v in sorted(inj.items()) if k.startswith("plain")))
    print("  other clean peaks per fibre in the injected plates: %s" % ", ".join(
        "%s %.3f" % (k, v["false"] / v["fibres"]) for k, v in sorted(inj.items()) if k.startswith("plain") and k.endswith("_1.0")))
    galaxies(groups[(2, 1)])


def exposures(groups, cp, area, rows):
    """Frame-by-frame verdicts of the 8-sigma two-fibre lines left by the other-plate test (data/skyfibre_exposures.json): per sign, galaxy lines as the check, persistent peaks against dips."""
    p = os.path.join(DATA, "skyfibre_exposures.json")
    if not os.path.exists(p):
        return
    ex = json.load(open(p))

    def verdict(q, sign):
        return ex.get("%s %d %.3f %+d" % (q["file"], q["fiber"], q["lam"], sign), {}).get("verdict", "not read")
    print("lines read frame by frame: %d" % len(ex))
    rest, held = {}, {}
    for sign in (1, -1):
        g = groups[(2, sign)]
        gal = [q for q in g if q["snr"] >= 8 and not q["fixed"] and q["galaxy"]]
        for label, sel in (("at a galaxy's redshift", gal), ("the rest", rest_of(g))):
            v = collections.Counter(verdict(q, sign) for q in sel)
            print("  %s, %s: %d lines; %s" % ("peaks" if sign > 0 else "dips", label, len(sel),
                                             ", ".join("%s %d" % kv for kv in v.most_common())))
            if sel is gal:
                held[sign] = v["persistent"] / max(sum(n for k, n in v.items() if k != "not read"), 1)
        rest[sign] = [q for q in rest_of(g) if verdict(q, sign) == "persistent"]
    # persistent lines of each sign against their counterparts
    out = {}
    for sign in (1, -1):
        keys = ["%.5f %+.5f" % (q["ra"], q["dec"]) for q in rest[sign]]
        have = [k for k in keys if k in cp]
        on, off = sum(1 for k in have if cp[k]["on"]), sum(1 for k in have if cp[k]["off"])
        out[sign] = (len(rest[sign]), len(have), on, off)
        print("  persistent %s: %d, %d covered by the catalogue, %d with a source at the fibre against %d at the control"
              % ("peaks" if sign > 0 else "dips", len(rest[sign]), len(have), on, off))
    n_p, h_p, on_p, off_p = out[1]
    n_d, h_d, on_d, off_d = out[-1]
    g_p = (on_p - off_p) * n_p / max(h_p, 1)
    g_d = (on_d - off_d) * n_d / max(h_d, 1)
    exc = (n_p - g_p) - (n_d - g_d)
    var = n_p + n_d + (on_p + off_p) * (n_p / max(h_p, 1)) ** 2 + (on_d + off_d) * (n_d / max(h_d, 1)) ** 2
    up = max(exc, 0.0) + 1.645 * math.sqrt(var)
    # a steady line persists as the galaxies' lines do, at the injections' recovery for 8 sigma
    r8 = (0.560 + (0.741 - 0.560) * (8.0 / 6.0 - 1.0) / 0.5) * held[1]
    print("  persistent peaks net of counterparts %.0f against dips %.0f: excess %+.0f +- %.0f, 95 per cent upper limit %.0f lines; "
          "galaxy lines persistent %.3f (dips' %.3f), recovery %.2f; fewer than %.0f steady transmitters per deg2"
          % (n_p - g_p, n_d - g_d, exc, math.sqrt(var), up, held[1], held[-1], r8, up / (area * r8)))
    # strongest persistent peaks without a counterpart
    for q in sorted(rest[1], key=lambda q: -q["snr"])[:8]:
        c = cp.get("%.5f %+.5f" % (q["ra"], q["dec"]))
        v = ex["%s %d %.3f %+d" % (q["file"], q["fiber"], q["lam"], 1)]
        gal = SkyCoord(q["ra"], q["dec"], unit="deg").galactic
        print("    %s fibre %d %.1f A S/N %.1f: frames %s; LS %s; l %.2f b %+.2f; lines in the spectrum %s" % (
            q["file"], q["fiber"], q["lam"], q["snr"], " ".join("%s:%.1f" % (f[0], f[2]) for f in v["frames"]),
            "not covered" if c is None else ", ".join("%s r %s" % (x["type"], x["mag_r"]) for x in c["on"]) or "none",
            gal.l.deg, gal.b.deg, " ".join("%.1f(%+.0f,%d)" % (x["lam"], x["sign"] * x["snr"], x["n_shared"]) for x in sorted(
                (x for x in rows if x["file"] == q["file"] and x["fiber"] == q["fiber"]), key=lambda x: x["lam"]))))


def positions(lines, r=1.5):
    """Lines grouped by fibre position, friends of friends within r arcsec."""
    cell = collections.defaultdict(list)
    for i, p in enumerate(lines):
        cell[(int(p["dec"] * 3600 // r), int(p["ra"] * math.cos(math.radians(p["dec"])) * 3600 // r))].append(i)
    root = list(range(len(lines)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for (a, b), idx in cell.items():
        near = [j for da in (-1, 0, 1) for db in (-1, 0, 1) for j in cell.get((a + da, b + db), ())]
        for i in idx:
            for j in near:
                if i < j and lasers.skyfibre_sep(lines[i], lines[j]) <= r:
                    root[find(i)] = find(j)
    out = collections.defaultdict(list)
    for i, p in enumerate(lines):
        out[find(i)].append(p)
    return list(out.values())


def recurring(rows, groups, cp):
    """Positions where one clean line recurs on two or more plate-nights, peaks against dips, and the stack of the most frequent."""
    def nights(v, lam):
        return len({q["file"] for q in v if abs(math.log10(q["lam"] / lam)) < 3e-4})
    top = None
    for sign in (1, -1):
        multi = [v for v in positions([p for p in groups[(2, sign)] if not p["fixed"]]) if len({p["file"] for p in v}) >= 2]
        rec = []
        for v in multi:
            best = max(v, key=lambda p: nights(v, p["lam"]))
            if nights(v, best["lam"]) >= 2:
                rec.append((nights(v, best["lam"]), len({p["file"] for p in v}), best, v))
        # a line recurring at one position whose wavelength recurs at another is the sky's
        shared = [x for x in rec if any(y is not x and abs(math.log10(y[2]["lam"] / x[2]["lam"])) < 3e-4 for y in rec)]
        n8 = sum(1 for n, nn, best, v in rec for p in rest_of(v) if abs(math.log10(p["lam"] / best["lam"])) < 3e-4)
        print("%s: positions with clean lines on 2+ plate-nights %d; one line recurring %d (%d of the rest at 8 sigma), "
              "at a wavelength recurring at another position %d"
              % ("peaks" if sign > 0 else "dips", len(multi), len(rec), n8, len(shared)))
        for n, nn, best, v in sorted(rec, key=lambda x: (-x[0], -x[2]["snr"]))[:8]:
            print("  %2d of %2d plate-nights (%d of the rest at 8 sigma): %.2f A at RA %.5f Dec %+.5f on plates %s; other lines %s" % (
                n, nn, sum(1 for p in rest_of(v) if abs(math.log10(p["lam"] / best["lam"])) < 3e-4), best["lam"], best["ra"],
                best["dec"], ",".join(sorted({p["plate"] for p in v})),
                sorted({round(p["lam"], 1) for p in v if abs(math.log10(p["lam"] / best["lam"])) >= 3e-4})[:6]))
        if sign > 0:
            top = max(rec, key=lambda x: x[0])
            others = [p for n, nn, best, v in sorted(rec, key=lambda x: -x[0])[2:] for p in rest_of(v)
                      if abs(math.log10(p["lam"] / best["lam"])) < 3e-4]
            have = [cp["%.5f %+.5f" % (p["ra"], p["dec"])] for p in others if "%.5f %+.5f" % (p["ra"], p["dec"]) in cp]
            print("  the other %d positions: %d recurring lines of the rest at 8 sigma, %d covered, %d with a source at the fibre "
                  "against %d at the control" % (len(rec) - 2, len(others), len(have), sum(1 for x in have if x["on"]),
                                                 sum(1 for x in have if x["off"])))
    n, nn, best, v = top
    plates = sorted({p["plate"] for p in rows if lasers.skyfibre_sep(p, best) <= 1.5})
    stack_report(stack_position(best["ra"], best["dec"], plates), best["lam"])


def stack_position(ra, dec, plates, r=1.5):
    """Inverse-variance stack of every spPlate of the plates with a sky fibre within r arcsec of (ra, dec), cached in data/skyfibre_recurring.json."""
    path = os.path.join(DATA, "skyfibre_recurring.json")
    if os.path.exists(path):
        return json.load(open(path))
    files = []
    for plate in plates:
        listing = lasers.skyfibre_get(lasers.skyfibre_BASE + "%s/" % plate).decode("utf-8", "replace")
        files += ["%s/%s" % (plate, x) for x in sorted(set(re.findall(r'href="(spPlate-%s-\d+\.fits)"' % plate, listing)))]

    def one(name):
        url = lasers.skyfibre_BASE + name
        try:
            hd = lasers.skyfibre_hdus(url, 7)
            pm = lasers.skyfibre_plugmap(url, hd)
            sep = np.hypot((np.asarray(pm["RA"], float) - ra) * math.cos(math.radians(dec)), np.asarray(pm["DEC"], float) - dec) * 3600
            i = int(np.argmin(sep))
            if sep[i] > r or str(pm["OBJTYPE"][i]).strip().upper() != "SKY":
                return None
            h0 = hd[0][0]
            fl, iv = lasers.skyfibre_rows(url, [(hd[0][1], [i]), (hd[1][1], [i])], h0["NAXIS1"] * 4, ">f4")
            return dict(file=name.split("/")[1], fiber=int(pm["FIBERID"][i]), start=int(round(h0["COEFF0"] / 1e-4)),
                        flux=fl[i].astype(float), ivar=iv[i].astype(float))
        except Exception as e:
            print("  %s: %s" % (name, repr(e)[:100]))
            return None
    got = [g for g in core.pmap(one, files, threads=4) if g]
    lo = min(g["start"] for g in got)
    F = np.zeros(max(g["start"] + len(g["flux"]) for g in got) - lo)
    W = np.zeros_like(F)
    for g in got:
        ok = g["ivar"] > 0
        sl = slice(g["start"] - lo, g["start"] - lo + len(g["flux"]))
        F[sl] += np.where(ok, g["flux"] * g["ivar"], 0.0)
        W[sl] += np.where(ok, g["ivar"], 0.0)
    out = dict(ra=ra, dec=dec, n_files=len(files), spectra=[[g["file"], g["fiber"]] for g in got], start=lo,
               flux=[round(float(x), 5) for x in np.where(W > 0, F / np.maximum(W, 1e-30), 0.0)], ivar=[round(float(x), 4) for x in W])
    json.dump(out, open(path, "w"))
    return out


# vacuum rest wavelengths: each reading of the stacked line and the lines it implies inside the spectrum
READINGS = {"Ly alpha": (1215.67, (("N V", 1240.81), ("C IV", 1549.48), ("He II", 1640.42), ("C III]", 1908.73))),
            "C IV": (1549.48, (("Ly alpha", 1215.67), ("He II", 1640.42), ("C III]", 1908.73))),
            "Mg II": (2799.94, (("[O II]", 3728.49), ("H beta", 4862.68), ("[O III]", 5008.24))),
            "[O II]": (3728.49, (("H beta", 4862.68), ("[O III]", 5008.24), ("H alpha", 6564.61)))}


def stack_report(st, lam0):
    """The stacked line's centroid, width, flux and red share, and each reading's implied lines measured in the stack."""
    lam = 10 ** ((st["start"] + np.arange(len(st["flux"]))) * 1e-4)
    f, w = np.array(st["flux"]), np.array(st["ivar"])
    good = w > 0
    s = f * np.sqrt(w)
    scale = 1.4826 * np.median(np.abs(s[good] - np.median(s[good])))
    j = int(np.argmin(np.abs(lam - lam0)))
    j += int(np.argmax(f[j - 3:j + 4])) - 3

    def cont(i):
        sd = np.r_[i - 40:i - 8, i + 9:i + 41]
        return float(np.median(f[sd][good[sd]]))
    c = cont(j)
    k = np.arange(j - 7, j + 8)
    d, dl = f[k] - c, lam[k] * math.log(10) * 1e-4
    flux, eflux = float(np.sum(d * dl)), scale * float(np.sqrt(np.sum(dl ** 2 / w[k])))
    half = d[7] / 2
    a = next(i for i in range(7, -1, -1) if d[i] < half)
    b = next(i for i in range(7, 15) if d[i] < half)
    x0 = lam[k][a] + (half - d[a]) / (d[a + 1] - d[a]) * (lam[k][a + 1] - lam[k][a])
    x1 = lam[k][b - 1] + (d[b - 1] - half) / (d[b - 1] - d[b]) * (lam[k][b] - lam[k][b - 1])
    cen = float(np.sum(lam[k] * d) / np.sum(d))
    red, blue = float(np.sum((d * dl)[8:])), float(np.sum((d * dl)[:7]))
    eside = scale * float(np.sqrt(np.sum(dl[8:] ** 2 / w[k][8:])))
    print("stack of %d spectra (%d spPlate files of the plates): %.2f A, centroid %.2f, peak %.2f, FWHM %.2f A (%.0f km/s), "
          "flux %.2e +- %.1e erg/s/cm2 (S/N %.0f), red of the peak %.2e against blue %.2e (+- %.1e each)"
          % (len(st["spectra"]), st["n_files"], lam[j], cen, f[j], x1 - x0, (x1 - x0) / cen * 2.998e5, flux * 1e-17, eflux * 1e-17,
             flux / eflux, red * 1e-17, blue * 1e-17, eside * 1e-17))
    sig = (x1 - x0) / 2.3548
    for name, (rest, implied) in READINGS.items():
        z = cen / rest - 1
        out = []
        for line, lr in implied:
            lc = lr * (1 + z)
            i = int(np.argmin(np.abs(lam - lc)))
            if lc < lam[0] + 50 or lc > lam[-1] - 50 or i < 45 or i > len(lam) - 46:
                out.append("%s outside" % line)
                continue
            kk = np.arange(i - 6, i + 7)
            g = np.exp(-0.5 * ((lam[kk] - lc) / sig) ** 2)
            amp = float(np.sum(g * (f[kk] - cont(i)) * w[kk]) / np.sum(g ** 2 * w[kk]))
            eamp = scale / math.sqrt(float(np.sum(g ** 2 * w[kk])))
            out.append("%s %.0f A %+.2f +- %.2f of the line" % (line, lc, amp * sig * math.sqrt(2 * math.pi) / flux,
                                                                 eamp * sig * math.sqrt(2 * math.pi) / flux))
        print("  as %s at z %.4f: %s" % (name, z, "; ".join(out)))


def galaxies(peaks):
    """Galaxy table: two-fibre-rule fibres whose clean peaks pass skyfibre_galaxy, with Legacy Surveys and DESI matches."""
    g = json.load(open(os.path.join(DATA, "skyfibre_galaxies.json")))
    keep = {(p["file"], p["fiber"]): p["galaxy"] for p in peaks if p["galaxy"] and not p["fixed"]}
    loose = {(p["file"], p["fiber"]): p["galaxy_loose"] for p in peaks if p["galaxy_loose"] and not p["fixed"]}
    cpd = json.load(open(os.path.join(DATA, "skyfibre_counterparts.json")))
    gl = []
    for x in g["galaxies"]:
        k = ("spPlate-%d-%d.fits" % (x["plate"], x["mjd"]), x["fiber"])
        if k in keep:
            z, lines = keep[k]
            # Legacy Surveys sources from the viewer catalogue (DR9 north, DR10 south)
            c = cpd.get("%.5f %+.5f" % (x["ra"], x["dec"]))
            ls = sorted(c["on"], key=lambda s: s["sep"]) if c else None
            gl.append(dict(x, z=z, lines=lines, ls=ls or [], covered=c is not None))
    gl.sort(key=lambda x: x["z"])
    dropped = collections.Counter("+".join(v[1]) for k, v in loose.items() if k not in keep)
    print("redshift test: %d fibres by skyfibre_lines, %d by skyfibre_galaxy (in the table file %d); set aside %s"
          % (len(loose), len(keep), len(gl), dropped.most_common()))
    gal = [x for x in gl if x["z"] > 0.01]
    zs = np.array([x["z"] for x in gal])
    print("galaxies %d (z > 0.01), Galactic %d; z %.3f-%.3f median %.3f; quartiles %s"
          % (len(gal), len(gl) - len(gal), zs.min(), zs.max(), np.median(zs), np.percentile(zs, [25, 75]).round(3)))
    print("  lines:", collections.Counter(n for x in gal for n in x["lines"]).most_common())
    o3 = [x["oiii_flux"] for x in gal if x["oiii_flux"]]
    print("  [OIII]5007 fluxes: n %d median %.1e range %.1e-%.1e" % (len(o3), np.median(o3), min(o3), max(o3)))
    withls = [x for x in gal if x["ls"]]
    r = np.array([float(x["ls"][0]["mag_r"]) for x in withls if x["ls"][0]["mag_r"] not in (None, "", "nan", "inf")])
    print("  LS counterparts %d of %d, %d outside the catalogue; types %s; r median %.2f range %.2f-%.2f" % (
        len(withls), len(gal), sum(1 for x in gal if not x["covered"]),
        collections.Counter(x["ls"][0]["type"] for x in withls).most_common(), np.median(r), r.min(), r.max()))
    print("  DESI redshifts within 1.5 arcsec: %d" % sum(1 for x in gal if x["desi"]))
    red = [x for x in gal if all(lasers.skyfibre_LINES[n] * (1 + x["z"]) > 8300 for n in x["lines"])]
    print("  galaxies whose lines all lie above 8300 A: %d (%s)" % (len(red), ", ".join("+".join(x["lines"]) for x in red)))
    o3pairs = [x for x in gal if set(x["lines"]) == {"[OIII]4959", "[OIII]5007"}]
    ratios = []
    for x in o3pairs:
        ps = [p for p in peaks if p["file"] == "spPlate-%d-%d.fits" % (x["plate"], x["mjd"]) and p["fiber"] == x["fiber"]]
        f = {n: max((p["flux"] for p in ps if abs(p["lam"] / (lasers.skyfibre_LINES[n] * (1 + x["z"])) - 1) < 4e-4),
                    default=None) for n in ("[OIII]4959", "[OIII]5007")}
        if f["[OIII]4959"] and f["[OIII]5007"]:
            ratios.append(f["[OIII]5007"] / f["[OIII]4959"])
    print("  found by the [O III] doublet alone: %d, flux ratios 5007/4959 %s" % (
        len(o3pairs), ", ".join("%.1f" % r for r in sorted(ratios))))
    twice = [(a, b) for i, a in enumerate(gl) for b in gl[i + 1:]
             if abs(a["dec"] - b["dec"]) < 2 / 3600 and abs(a["ra"] - b["ra"]) * math.cos(math.radians(a["dec"])) < 2 / 3600]
    for a, b in twice:
        print("  seen twice: %d-%d-%d and %d-%d-%d, z %.4f and %.4f" % (a["plate"], a["mjd"], a["fiber"], b["plate"],
                                                                         b["mjd"], b["fiber"], a["z"], b["z"]))
    rows = []
    for x in gl:
        ls = x["ls"][0] if x["ls"] else None
        if ls:
            r = float(ls["mag_r"]) if ls["mag_r"] not in (None, "") else float("nan")
            src = "%s, $r=%.1f$" % (TYPE.get(ls["type"], ls["type"]), r) if np.isfinite(r) else TYPE.get(ls["type"], ls["type"])
        else:
            src = "--" if x["covered"] else "outside"
        rows.append("%d-%d-%d & %.5f & %+.5f & %.4f & %s & %.1f & %s\\\\" % (
            x["plate"], x["mjd"], x["fiber"], x["ra"], x["dec"], x["z"], ", ".join(LINE[n] for n in x["lines"]), x["snr"],
            src))
    open(os.path.join(HERE, "table_galaxies.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(rows) + "\n")
    print("table rows", len(rows))


if __name__ == "__main__":
    from setitools.core import quiet_warnings
    quiet_warnings()
    main()
