"""Numbers, Tables 4 and 5 and Figure 1 of the ultraviolet laser paper from the ultraviolet results."""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.argv = sys.argv[:1]
from setitools import core  # noqa: E402
from setitools import ultraviolet as uv  # noqa: E402

RES = core.RESULTS
SETS = ("starcat", "stis", "cos", "iue", "fuse")
INST = {"starcat": "STIS", "stis": "STIS, later", "cos": "COS", "iue": "IUE", "fuse": "FUSE"}
NAMES = {"Proxima Centauri": "Proxima Centauri", "alf Cen A": r"$\alpha$~Cen~A", "alf Cen B": r"$\alpha$~Cen~B",
         "Barnard's star": "Barnard's star", "alf CMa": "Sirius~A", "alf CMa B": "Sirius~B",
         "eps Eri": r"$\epsilon$~Eri", "HD 217987": "GJ~887", "61 Cyg A": "61~Cyg~A", "alf CMi": "Procyon",
         "HD 173739": "GJ~725~A", "eps Ind": r"$\epsilon$~Ind", "tau Cet": r"$\tau$~Cet", "L 372-58": "GJ~1061",
         "HD 33793": "Kapteyn's star", "BD-12 4523": "GJ~628", "Wolf 28": "van Maanen's star", "CD-46 11540": "GJ~674",
         "LAWD 37": "LAWD~37", "BD-15 6290": "GJ~876", "DENIS J025503.3-470049": r"DENIS~J0255$-$4700",
         "AD Leo": "AD~Leo", "HD 204961": "GJ~832", "alf Boo": "Arcturus", "R Leo": "R~Leo", "EG And": "EG~And"}
TYPES = {"alf Boo": r"K1.5\,III", "R Leo": "M7--9e, Mira", "EG And": r"M2.4\,III, symbiotic", "eps Eri": r"K2\,V"}
STYLE = {"starcat": ("black", "o", "1.5pt", "solid"), "stis": ("black!65", "x", "1.8pt", "dashdotted"),
         "cos": ("black", "square*", "1.3pt", "solid"), "iue": ("black!55", "triangle*", "1.6pt", "densely dashed"),
         "fuse": ("black!75", "diamond*", "1.6pt", "densely dotted")}


def load(name):
    """Result file from data/, else the results directory."""
    p = os.path.join(HERE, "data", name)
    return json.load(open(p if os.path.exists(p) else os.path.join(RES, name)))


def short(n):
    return " ".join(n.replace("NAME ", "").replace("V* ", "").replace("* ", "").split())


def sci(x, digits=1):
    """LaTeX scientific notation."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return r"%.*f\times10^{%d}" % (digits, m, e)


def searches():
    sc, cos, fu = load("uv_laser.json"), load("cos_laser.json"), load("fuse_laser.json")
    left = uv.limits_left("starcat", sc)
    print("STIS: %d exposures of %d objects; %d peaks; %d persistent; %d pass the first test; %d left in %d stars" % (
        sc["exposures"], sc["objects"], sc["peaks"], len(sc["persistent"]),
        sum(1 for p in sc["persistent"] if "forced_rest" in p), len(left), len({p["obj"] for p in left})))
    st = load("stis_laser.json")
    sl = uv.limits_left("stis", st)
    print("STIS x1d: %d exposures of %d objects; %d peaks; %d persistent in %d stars; %d pass the first test; %d left in %d "
          "stars, %d with a NIST line within 15 km/s" % (
              st["exposures"], st["objects"], st["peaks"], len(st["persistent"]), len({p["star"] for p in st["persistent"]}),
              sum(1 for p in st["persistent"] if p["stars_rest"] < 2 and p["stars_obs"] < 2), len(sl),
              len({p["star"] for p in sl}), sum(1 for p in st["identified"] if p["id"])))
    first = [p for p in cos["persistent"] if p["stars_rest"] < 2 and p["stars_obs"] < 2]
    print("COS: %d spectra of %d stars; %d peaks; %d persistent in %d stars; %d pass the first test; %d left" % (
        cos["spectra"], cos["stars"], cos["peaks"], len(cos["persistent"]),
        len({p["star"] for p in cos["persistent"]}), len(first), len(uv.limits_left("cos", cos))))
    for tag in ("6sig", "45sig"):
        d = load("iue_laser_%s.json" % tag)
        pp = [p for p in d["persistent"] if "aperture_fixed" in p]
        print("IUE %s: %d images; %d peaks; %d persistent; %d pass the first test (%d fixed to an aperture); %d left" % (
            tag, d["images"], d["peaks"], len(d["persistent"]), len(pp), sum(p["aperture_fixed"] for p in pp),
            len(uv.limits_left("iue", d))))
    print("FUSE: %d observations of %d stars; %d single-channel peaks; %d confirmed" % (
        fu["files"], fu["objects"], len(fu["found"]), sum(1 for x in fu["found"] if x["confirmed"])))
    for x in fu["found"]:
        print("   %-22s %s %.3f A, z %.1f" % (x["star"], x["channel"], x["lam"], x["z"]))


def sensitivity():
    tab = load("uv_limits_table.json")
    for key in SETS:
        d = load("uv_limits_%s.json" % key)
        rows = [r for r in tab["rows"] if r["set"] == key]
        near = [r for r in rows if r["plx"] and r["plx"] > 100]
        print("%s: %d exposures injected; plateau %.2f, m90 %.2f; %d stars, median %.1e erg cm-2 s-1; %d within 10 pc, "
              "median %.1e W isotropic, %.2g W through 10 m" % (
                  INST[key], d["n_injected"], d["plateau"], d["m90"], len(rows), np.median([r["flux"] for r in rows]),
                  len(near), np.median([r["L_iso_W"] for r in near]), np.median([r["P_10m_W"] for r in near])))


def table_left():
    sc, cos = load("uv_laser.json"), load("cos_laser.json")
    for p in sorted(uv.limits_left("starcat", sc), key=lambda p: (short(p["main_id"]) != "alf Boo", p["main_id"], p["rest"])):
        s = short(p["main_id"])
        print(r"%s & %s & STIS %s & $%.3f$ & $%d/%d$ & $%.1f$\\" % (NAMES[s], TYPES[s], p["mode"], p["rest"], p["n"],
                                                                   p["of"], p["z"]))
    for p in sorted(uv.limits_left("cos", cos), key=lambda p: p["rest"]):
        s = short(p["star"])
        print(r"%s & %s & COS %s & $%.3f$ & $%d$ & $p=%s$\\" % (NAMES[s], TYPES[s], p["grating"], p["rest"], p["n"],
                                                               sci(p["p"], 0).replace(r"1\times", "")))


def table_nearest():
    best = {}
    for r in load("uv_limits_table.json")["rows"]:
        if r["plx"] and r["plx"] >= 200:
            n = short(r["sp"].split(";")[0] if r["set"] == "starcat" and r.get("sp") else r["star"])
            if n not in best or r["L_iso_W"] < best[n]["L_iso_W"]:
                best[n] = dict(r, name=n)
    for r in sorted(best.values(), key=lambda r: -r["plx"]):
        print(r"%s & $%.2f$ & %s & $%.0f$ & $%s$ & $%s$ & $%s$\\" % (
            NAMES.get(r["name"], r["name"]), 1000.0 / r["plx"], INST[r["set"]], r["lam"], sci(r["flux"]),
            sci(r["L_iso_W"]), sci(r["P_10m_W"])))
    P = np.array([r["P_10m_W"] for r in best.values()])
    print("%% %d stars within 5 pc; median %.2g W; %d below 10 W, %.2g to %.2g W" % (
        len(P), np.median(P), (P < 10).sum(), P[P < 10].min(), P[P < 10].max()))


def figure():
    print(r"\begin{tikzpicture}")
    print(r"\begin{semilogxaxis}[name=rec,width=0.47\textwidth,height=0.4\textwidth,xmin=0.9,xmax=9,ymin=0,ymax=1.02,")
    print(r"  xtick={1,1.5,2,3,5,8},xticklabels={$1$,$1.5$,$2$,$3$,$5$,$8$},minor xtick={},")
    print(r"  xlabel={injected flux / $F_{\mathrm{nom}}$},ylabel={fraction recovered},")
    print(r"  tick label style={font=\footnotesize},label style={font=\small},title={(a)},title style={font=\footnotesize},")
    print(r"  legend style={font=\scriptsize,at={(0.97,0.03)},anchor=south east,draw=none,fill=none},legend cell align=left]")
    for key in SETS:
        d = load("uv_limits_%s.json" % key)
        col, mk, ms, line = STYLE[key]
        print(r"\addplot[%s,thick,%s,mark=%s,mark size=%s,mark options={solid}] coordinates {%s};" % (
            col, line, mk, ms, " ".join("(%g,%.3f)" % (m, x) for m, x in zip(d["mult"], d["recovery"]))))
        print(r"\addlegendentry{%s ($%d$)}" % (INST[key], d["n_injected"]))
    print(r"\end{semilogxaxis}")
    print(r"\begin{loglogaxis}[at={(rec.east)},anchor=west,xshift=1.9cm,width=0.47\textwidth,height=0.4\textwidth,")
    print(r"  xmin=1,xmax=1000,ymin=1e-17,ymax=1e-9,xlabel={distance (pc)},ylabel={line-flux limit (\flux)},")
    print(r"  tick label style={font=\footnotesize},label style={font=\small},title={(b)},title style={font=\footnotesize},")
    print(r"  legend style={font=\scriptsize,at={(0.03,0.97)},anchor=north west,draw=none,fill=white,fill opacity=0.8,")
    print(r"  text opacity=1},legend cell align=left]")
    tab = load("uv_limits_table.json")
    for key in SETS:
        col, mk, _, _ = STYLE[key]
        pts = sorted((1000.0 / r["plx"], r["flux"]) for r in tab["rows"] if r["set"] == key and r["plx"] and r["plx"] > 0)
        print(r"\addplot[only marks,mark=%s,mark size=1.1pt,%s%s] coordinates {%s};" % (
            mk, col, "" if key == "starcat" else ",fill opacity=0.8", " ".join("(%.3g,%.3g)" % p for p in pts)))
        print(r"\addlegendentry{%s}" % INST[key])
    for L in (16, 18, 20, 22):
        d_end = min(1000.0, math.sqrt(10 ** (L + 7) / (4 * math.pi * 1e-17)) / 3.0857e18)
        print(r"\addplot[densely dotted,black!45,domain=1:%.4g,samples=2,forget plot] {1e%d/(4*pi*(x*3.0857e18)^2)} "
              r"node[pos=0.78,sloped,above,font=\tiny,text=black!60,inner sep=1pt] {$10^{%d}$\,W};" % (d_end, L + 7, L))
    print(r"\end{loglogaxis}")
    print(r"\end{tikzpicture}")


def arcturus():
    """Arcturus features passing both tests: peak against the brightest flux within 10 A per E230H exposure, and iron-group NIST lines within 15 km/s with chance-match odds."""
    import glob
    import re
    C = 299792.458
    IONS = ("Fe I", "Fe II", "Cr I", "Cr II", "Ti I", "Ti II", "Mn I", "Mn II", "V I", "V II", "Ni I", "Ni II", "Co I", "Co II",
            "Si I", "Mg I", "Ca I", "Ca II", "Al I", "Sc II", "Y II", "Zr II", "Cu I", "Zn I")
    sc = load("uv_laser.json")
    feats = [p for p in sc["persistent"] if p.get("main_id") == "* alf Boo" and p["stars_rest"] < 2 and p["stars_obs"] < 2]
    spectra = []
    for f in sorted(glob.glob(os.path.join(uv.starcat_ROOT, "fits", "HD124897", "*.fits"))):
        x = uv.starcat_load(f)
        if x and x[3]["mode"] == "E230H":
            spectra.append(x)
    lines = []
    for ion in IONS:
        lines += core.nist_lines(ion, 2600, 3100, cache_dir=os.path.join(os.path.dirname(uv.starcat_ROOT), "nist"))

    def inten(s):
        m = re.match(r"\s*(\d+)", s or "")
        return int(m.group(1)) if m else 0
    strong = [x for x in lines if inten(x["intens"]) >= 100]
    dens = len(strong) / 500.0
    print("%d features; %d E230H exposures; %d NIST lines of %d ions over 2600-3100 A, %d with intensity 100 or more: a random "
          "wavelength lies within 0.5 and 5 km/s of a strong one with probability %.2f and %.2f"
          % (len(feats), len(spectra), len(lines), len(IONS), len(strong), min(1, dens * 2 * 0.5 / C * 2850), min(1, dens * 2 * 5 / C * 2850)))
    for p in sorted(feats, key=lambda p: p["rest"]):
        ratios = []
        for w, f, e, meta in spectra:
            i = int(np.argmin(np.abs(w - p["lam"])))
            if abs(w[i] - p["lam"]) > 0.05 or w.min() + 10 > p["lam"] or w.max() - 10 < p["lam"]:
                continue
            near = (np.abs(w - p["lam"]) < 10.0) & (np.abs(w - p["lam"]) > 0.15)
            ratios.append(float(np.max(f[np.abs(w - p["lam"]) <= 0.06]) / np.max(f[near])))
        ids = sorted((x for x in strong if abs(x["lam"] - p["rest"]) / p["rest"] * C < 15), key=lambda x: abs(x["lam"] - p["rest"]))
        print("%.3f A (z %.1f, %d/%d): peak over the brightest flux within 10 A %s; strong NIST lines within 15 km/s: %s" % (
            p["rest"], p["z"], p["n"], p["of"], ", ".join("%.2f" % r for r in ratios),
            "; ".join("%s %.3f (%+.1f km/s)" % (x["ion"], x["lam"], (x["lam"] - p["rest"]) / p["rest"] * C) for x in ids[:4]) or "none"))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for title, fn in (("searches", searches), ("sensitivity", sensitivity), ("Table 4 rows", table_left),
                      ("Table 5 rows", table_nearest), ("Figure 1", figure), ("Arcturus", arcturus)):
        print("%% == %s" % title)
        fn()
