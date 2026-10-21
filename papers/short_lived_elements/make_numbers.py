"""Numbers, tables and figure of the short-lived elements paper from the composition elements results and Korg curves of growth."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.argv = sys.argv[:1]
from setitools import core  # noqa: E402
from setitools import composition as cp  # noqa: E402

INST = ("HARPS", "UVES", "FEROS", "ESPRESSO")
ELEM = ("Tc", "Pm", "Pa", "Pu", "Am", "Cm")
NAME = {"Tc": "technetium", "Pm": "promethium", "Pa": "protactinium", "Pu": "plutonium", "Am": "americium",
        "Cm": "curium"}
ION = {"Tc": r"\ion{Tc}{i}", "Pm": r"\ion{Pm}{ii}", "Pa": r"\ion{Pa}{ii}", "Pu": r"\ion{Pu}{ii}",
       "Am": r"\ion{Am}{ii}", "Cm": r"\ion{Cm}{ii}"}
# Pecaut and Mamajek (2013, v2022.04.16): dwarf BP-RP and Teff, F3 V to K2 V
MAMAJEK = ((0.518, 6750), (0.546, 6670), (0.587, 6550), (0.640, 6350), (0.670, 6280), (0.694, 6180), (0.719, 6050),
           (0.767, 5990), (0.784, 5930), (0.803, 5860), (0.823, 5770), (0.832, 5720), (0.841, 5680), (0.850, 5660),
           (0.869, 5600), (0.880, 5550), (0.900, 5480), (0.950, 5380), (0.983, 5270), (1.010, 5170), (1.100, 5100))
# Bahcall, Serenelli and Basu (2006): solar convective-zone mass (GS98) and hydrogen fraction
M_CZ_SUN = 0.02403
X_SURF = 0.74
M_SUN = 1.98892e30
M_H = 1.6735575e-27
U = 1.66053907e-27
# JEFF-3.1 cumulative 235U thermal fission yields; ENSDF half-lives
YIELD = {"Tc": 0.06132, "Pm": 0.02232}
HALF_LIFE_YR = {"Tc": 2.111e5, "Pm": 2.6234}
MASS_NUMBER = {"Tc": 99, "Pm": 147}
E_FISSION_J = 200e6 * 1.602176634e-19
YEAR = 3.15576e7
L_SUN = 3.828e26


def load(name):
    """Result file from data/, else the results directory."""
    p = os.path.join(HERE, "data", name)
    return json.load(open(p if os.path.exists(p) else core.results(name)))


def teff(bp_rp):
    b, t = zip(*MAMAJEK)
    return float(np.interp(bp_rp, b, t))


def curves():
    """{(element, line): (teff, abundance grid, log10 EW [teff, A])} at [M/H] 0, with -0.3 and +0.3 rows at 5800 K."""
    rows, grid, mh_rows = {}, None, {}
    for line in open(os.path.join(HERE, "curves_of_growth.txt")):
        if line.startswith("#"):
            grid = np.array([float(x.split("_A")[1]) for x in line.split()[8:]])
            continue
        p = line.split()
        t, mh, el, lam = float(p[0]), float(p[2]), p[3], float(p[4])
        ew = np.array([float(x) for x in p[7:]])
        if mh == 0.0:
            rows.setdefault((el, lam), []).append((t, ew))
        else:
            mh_rows[(el, lam, mh)] = ew
    out = {}
    for k, v in rows.items():
        v.sort()
        out[k] = (np.array([t for t, _ in v]), grid, np.log10(np.array([e for _, e in v])))
    return out, mh_rows


def abundance(cog, el, lam, t, ew_ma):
    """A(X) giving equivalent width ew_ma (mA) at temperature t from the interpolated curve of growth, the weak-line law below the grid."""
    ts, grid, lew = cog[(el, lam)]
    row = np.array([np.interp(t, ts, lew[:, j]) for j in range(len(grid))])
    y = np.log10(ew_ma)
    if y <= row[0]:
        return float(grid[0] + (y - row[0]))
    return float(np.interp(y, row, grid))


def strongest(lines):
    return max(lines, key=lambda x: x[1])[0]


def main():
    census = load("elements_census.json")
    harps = load("harps_sample.json")
    meta = {}
    print("== samples")
    print("census: %d Gaia DR3 stars within 50 pc in the colour, magnitude, RUWE and declination cuts" % len(census))
    print("HARPS sample: %d stars within 30 pc with >= 5 HARPS spectra, a radial velocity and < 200 SIMBAD references "
          "(references %d-%d)" % (len(harps), min(r["refs"] for r in harps), max(r["refs"] for r in harps)))
    for inst in INST:
        n2 = (len(harps) if inst == "HARPS" else
              sum(1 for r in census if r["eso"] and r["eso"].get(inst, 0) >= 2 and r.get("radial_velocity")))
        d = os.path.join(cp.elements_ROOT, inst)
        stars = [s for s in os.listdir(d) if os.path.exists(os.path.join(d, s, "star.json"))]
        nspec = sum(len([f for f in os.listdir(os.path.join(d, s)) if f.endswith(".fits")]) for s in stars)
        st = np.load(os.path.join(cp.elements_ROOT, "stacks_%s.npz" % inst))
        tc = load("element_Tc_%s.json" % inst)["results"]
        dv = np.array(st["v"]) - np.array(st["rv"])
        for s in stars:
            meta[s] = json.load(open(os.path.join(d, s, "star.json")))
        snr = []
        for T in st["stacks"]:
            seg = T[(cp.elements_GRID > 4150) & (cp.elements_GRID < 4550)].astype(float)
            snr.append(np.nanmedian(seg) / (1.4826 * np.nanmedian(np.abs(np.diff(seg))) / np.sqrt(2)))
        print("%-8s eligible %d; fetched %d stars, %d spectra; usable stacks %d; measured %d; median spectra a stack "
              "%.0f; median stack S/N a 0.01 A sample at 4150-4550 A %.0f; line velocity minus Gaia median %+.2f km/s, "
              "robust sd %.2f, more than 3 km/s off %d"
              % (inst, n2, len(stars), nspec, len(st["names"]), len(tc), np.median(st["nspec"]), np.median(snr),
                 np.median(dv), 1.4826 * np.median(np.abs(dv - np.median(dv))),
                 int(np.sum(np.abs(dv - np.median(dv)) > 3))))
    every = set()
    for inst in INST:
        for el in ELEM:
            every |= {r["star"] for r in load("element_%s_%s.json" % (el, inst))["results"]}
    dist = np.array([1000.0 / meta[s]["plx"] for s in every])
    col = np.array([meta[s]["bp_rp"] for s in every])
    print("distinct stars measured %d; distance %.1f-%.1f pc (median %.1f); BP-RP %.2f-%.2f"
          % (len(every), dist.min(), dist.max(), np.median(dist), col.min(), col.max()))

    print("== results")
    ntest, allp = 0, []
    for el in ELEM:
        for inst in INST:
            d = load("element_%s_%s.json" % (el, inst))
            res = d["results"]
            p = np.array([r["p"] for r in res])
            lim = np.array([r["ew_limit_ma"] for r in res])
            low = min(res, key=lambda r: r["p"])
            ntest += len(res)
            allp += [(r["p"], el, inst, r["star"]) for r in res]
            print("%s %-8s n %3d  p<0.01 %d (exp %.1f)  lowest p %.4f (%s)  limit median %.2f mA (10-90%% %.2f-%.2f)"
                  % (el, inst, len(res), int((p < 0.01).sum()), 0.01 * len(res), low["p"], low["star"],
                     np.median(lim), *np.percentile(lim, [10, 90])))
    for el in ELEM:
        for inst in INST:
            d = load("element_%s_%s.json" % (el, inst))
            out = [lam for (lam, _), k in zip(d["lines"], d["stable"]) if not k]
            if out:
                print("lines left out: %s %s %s" % (el, inst, out))
    for inst in INST:
        rows = load("element_check_%s.json" % inst)
        p = np.array([r["p"] for r in rows])
        multi = np.array([r["p"] for r in rows if r["element"] != "Cm"])
        print("injection check %-8s %d injections into %d stars: p median %.4f (multi-line %.4f), 10-90%% %.4f-%.4f"
              % (inst, len(rows), len({r["star"] for r in rows}), np.median(p), np.median(multi),
                 *np.percentile(multi, [10, 90])))
    allp.sort()
    print("tests %d; lowest p %.5f (%s %s %s), expected number below it %.2f; below 0.01: %d (exp %.1f); below 0.001: "
          "%d (exp %.1f)" % (ntest, allp[0][0], allp[0][1], allp[0][2], allp[0][3], allp[0][0] * ntest,
                             sum(1 for x in allp if x[0] < 0.01), 0.01 * ntest,
                             sum(1 for x in allp if x[0] < 0.001), 0.001 * ntest))
    for x in allp[:6]:
        print("   ", x)

    print("== abundance limits")
    cog, mh_rows = curves()
    for el in ("Tc", "Pm"):
        for inst in INST:
            d = load("element_%s_%s.json" % (el, inst))
            lam = strongest(d["lines"])
            a = []
            for r in d["results"]:
                t = teff(meta[r["star"]]["bp_rp"])
                a.append((abundance(cog, el, lam, t, max(r["ew_limit_ma"], 1e-3)), t, r["star"]))
            a.sort()
            av = np.array([x[0] for x in a])
            print("%s %-8s line %.2f  A limit median %.2f, best %.2f (%s, %.0f K), 10-90%% %.2f-%.2f"
                  % (el, inst, lam, np.median(av), a[0][0], a[0][2], a[0][1], *np.percentile(av, [10, 90])))
        for lam in sorted({k[1] for k in cog if k[0] == el}):
            ts, grid, lew = cog[(el, lam)]
            for t in (5200, 5800, 6400):
                print("   %s %.2f at %d K: A for 1 mA %.2f" % (el, lam, t, abundance(cog, el, lam, t, 1.0)))
        for mh in (-0.3, 0.3):
            lam = 4297.06 if el == "Tc" else 4186.03
            ts, grid, lew = cog[(el, lam)]
            j = int(np.argmin(np.abs(ts - 5800)))
            ratio = mh_rows[(el, lam, mh)][2] / 10 ** lew[j, 2]
            print("   %s %.2f at 5800 K, [M/H] %+.1f: EW at A=0 changes by %.2f (%.2f dex)" % (el, lam, mh, ratio,
                                                                                         np.log10(ratio)))

    print("== masses and fission power for a solar-type star")
    n_h = X_SURF * M_CZ_SUN * M_SUN / M_H
    for el in ("Tc", "Pm"):
        for inst in INST:
            d = load("element_%s_%s.json" % (el, inst))
            lam = strongest(d["lines"])
            sun = [r for r in d["results"] if 5600 <= teff(meta[r["star"]]["bp_rp"]) <= 6000]
            if not sun:
                continue
            a = np.median([abundance(cog, el, lam, teff(meta[r["star"]]["bp_rp"]), r["ew_limit_ma"]) for r in sun])
            n = 10 ** (a - 12.0) * n_h
            mass = n * MASS_NUMBER[el] * U
            rate = n * np.log(2) / (HALF_LIFE_YR[el] * YEAR)
            power = rate / YIELD[el] * E_FISSION_J
            fissions = n / YIELD[el]
            print("%s %-8s %d stars at 5600-6000 K: median A limit %.2f; %.2g atoms, %.2g kg in a solar convective "
                  "zone; supply %.2g kg/yr; fission power %.2g W (%.2g L_sun); the product of %.2g fissions, %.2g kg of U-235, "
                  "%.0f times the Earth's uranium (8e16 kg, McDonough and Sun 1995)"
                  % (el, inst, len(sun), a, n, mass, mass * np.log(2) / HALF_LIFE_YR[el], power, power / L_SUN,
                     fissions, fissions * 235 * U, fissions * 235 * U / 8e16))


def tables():
    """LaTeX rows of Tables 1-3."""
    print("== table 1 rows")
    census = load("elements_census.json")
    harps = load("harps_sample.json")
    for inst in INST:
        n2 = (len(harps) if inst == "HARPS" else
              sum(1 for r in census if r["eso"] and r["eso"].get(inst, 0) >= 2 and r.get("radial_velocity")))
        d = os.path.join(cp.elements_ROOT, inst)
        stars = [s for s in os.listdir(d) if os.path.exists(os.path.join(d, s, "star.json"))]
        nspec = sum(len([f for f in os.listdir(os.path.join(d, s)) if f.endswith(".fits")]) for s in stars)
        st = np.load(os.path.join(cp.elements_ROOT, "stacks_%s.npz" % inst))
        snr, hi = [], []
        for T in st["stacks"]:
            seg = T[(cp.elements_GRID > 4150) & (cp.elements_GRID < 4550)].astype(float)
            snr.append(np.nanmedian(seg) / (1.4826 * np.nanmedian(np.abs(np.diff(seg))) / np.sqrt(2)))
            hi.append(cp.elements_GRID[np.isfinite(T)].max())
        n = len(load("element_Tc_%s.json" % inst)["results"])
        print(r"%s & $%s$ & $4100$--$%.0f$ & $%d$ & $%d$ & $%d$ & $%d$ & $%.0f$ \\"
              % (inst, "{:,}".format(int(cp.elements_RES[inst])).replace(",", "{,}"), np.median(hi), n2, len(stars),
                 nspec, n, np.median(snr)))
    print("== table 2 rows")
    src = {"Tc": "Palmeri et al.~\\cite{Palmeri2005}", "Pm": "Fivet et al.~\\cite{Fivet2007}"}
    for el in ELEM:
        lines = load("element_%s_HARPS.json" % el)["lines"]
        print(r"%s & %s & %s \\" % (ION[el], ", ".join("%.2f (%.2f)" % (lam, w) for lam, w in lines),
                                    src.get(el, "NIST arc intensity")))
    print("== table 3 rows")
    for el in ELEM:
        cells = []
        for inst in INST:
            res = load("element_%s_%s.json" % (el, inst))["results"]
            p = min(r["p"] for r in res)
            lim = np.median([r["ew_limit_ma"] for r in res])
            ps = ("%.3f" % p) if p >= 0.001 else ("%.4f" % p)
            cells.append(r"$%d$ & $%s$ & $%.2f$" % (len(res), ps, lim))
        print(r"%s & %s \\" % (ION[el], " & ".join(cells)))


STYLE = {"HARPS": ("black", "o"), "UVES": ("black", "square*"), "FEROS": ("black!55", "triangle*"),
         "ESPRESSO": ("black!75", "diamond*")}


def figure():
    """pgfplots code of Figure 1: abundance limits against temperature, Tc and Pm."""
    cog, _ = curves()
    census = {}
    for inst in INST:
        d = os.path.join(cp.elements_ROOT, inst)
        for s in os.listdir(d):
            if os.path.exists(os.path.join(d, s, "star.json")):
                census[s] = json.load(open(os.path.join(d, s, "star.json")))
    print("== figure")
    for el, panel in (("Tc", "a"), ("Pm", "b")):
        print("%% panel (%s), %s" % (panel, el))
        for inst in INST:
            d = load("element_%s_%s.json" % (el, inst))
            lam = strongest(d["lines"])
            pts = sorted((teff(census[r["star"]]["bp_rp"]),
                          abundance(cog, el, lam, teff(census[r["star"]]["bp_rp"]), r["ew_limit_ma"]))
                         for r in d["results"])
            col, mk = STYLE[inst]
            print(r"\addplot[only marks,mark=%s,mark size=1.0pt,%s,fill opacity=0.8] coordinates {%s};"
                  % (mk, col, " ".join("(%.0f,%.2f)" % p for p in pts)))
            print(r"\addlegendentry{%s}" % inst)
        lam = 4297.06 if el == "Tc" else 4186.03
        ts = np.arange(5100, 6501, 100)
        print(r"\addplot[densely dotted,black!45,forget plot] coordinates {%s};"
              % " ".join("(%d,%.2f)" % (t, abundance(cog, el, lam, t, 1.0)) for t in ts))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
    tables()
    figure()
