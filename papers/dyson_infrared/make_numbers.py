"""Coverage-store records and result-file numbers of the infrared waste-heat paper."""
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from setitools import core  # noqa: E402

RUNS = ("dark_companions_wise", "pma_dark_companions", "hz_stack_wise", "enclosed_stars", "cold_shells_akari",
        "neowise_decade_trends", "vasco_ir_red", "ir_excess_bl", "spherex_hephaistos", "spherex_contardo53")


def results(name):
    """Result file from data/, else the results directory."""
    p = os.path.join(HERE, "data", name)
    return p if os.path.exists(p) else core.results(name)


def sensitivities():
    """Reach of the enclosed-star, AKARI and vanished-star tests: WISE and AKARI black-body fluxes."""
    import math
    h, c, k, sigma = 6.62607e-34, 2.99792e8, 1.380649e-23, 5.670374e-8
    Lsun, pc = 3.828e26, 3.0857e16
    zp = {"W2": (4.6028e-6, 171.787), "W3": (11.5608e-6, 31.674), "W4": (22.0883e-6, 8.363)}  # Jarrett et al. (2011)

    def fnu(L, T, d, lam):
        nu = c / lam
        return L / (4 * math.pi * d ** 2) * math.pi * (2 * h * nu ** 3 / c ** 2 / math.expm1(h * nu / (k * T))) / (sigma * T ** 4) * 1e26

    def mag(L, T, d, band):
        return -2.5 * math.log10(fnu(L, T, d, zp[band][0]) / zp[band][1])
    print("== sensitivities")
    for T in (200, 300, 500):
        for L in (0.001, 0.01, 0.1):
            m = mag(L * Lsun, T, 100 * pc, "W3")
            print("enclosed star: %d K shell around %.3f Lsun: W3 %.2f at 100 pc, W3 < 8 to %.0f pc" % (T, L, m, 100 * 10 ** (-0.2 * (m - 8))))
    for T in (30, 50, 100):
        for L, d in ((1.0, 50), (1.0, 20), (0.01, 10)):
            print("AKARI 90 um, 0.55 Jy: %d K shell around %.2f Lsun at %d pc detectable above f = %.1e" % (T, L, d, 0.55 / fnu(L * Lsun, T, d * pc, 90e-6)))
    Fbol = 2.7e-8 * 10 ** (-0.4 * 19.5)  # a solar-type star of V 19.5 (R 19)
    for T in (200, 300, 600):
        print("vanished star of R 19 enclosed at %d K: %s" % (T, ", ".join("%s %.1f" % (b, -2.5 * math.log10(
            Fbol * math.pi * (2 * h * (c / lam) ** 3 / c ** 2 / math.expm1(h * (c / lam) / (k * T))) / (sigma * T ** 4) * 1e26 / z))
            for b, (lam, z) in zp.items())))


def main():
    sensitivities()
    con = sqlite3.connect(core.COVERAGE)
    cols = [r[1] for r in con.execute("pragma table_info(searches)")]
    for run in RUNS:
        r = dict(zip(cols, con.execute("select * from searches where id = ?", (run,)).fetchone()))
        print("== %s (%s, %s targets)" % (run, r["code"], r["n_targets"]))
        for k in ("selection", "null_summary", "sensitivity", "result"):
            if r[k]:
                print("   %s: %s" % (k, r[k]))
    hz = json.load(open(results("hz_stack.json")))
    for band in ("W3", "W4"):
        x = hz[band]
        print("habitable-zone stack %s: hosts %+.4f +- %.4f (%d), other hosts %+.4f +- %.4f (%d), controls as hosts "
              "%+.4f +- %.4f (%d); covering fraction bound %.2g"
              % (band, *x["hz"], *x["other"], *x["null"], x["covering_fraction"]))
    pma = json.load(open(results("pma_dark.json")))
    print("proper-motion companions: %d stars" % len(pma))
    import numpy as np
    for band in ("W3", "W4"):
        d = np.array([o[band] - o[band + "_allsky"] for o in pma if o.get(band) is not None and o.get(band + "_allsky") is not None])
        print("  %s AllWISE minus All-Sky over %d stars: median %+.3f, robust scatter %.3f, 5-95 per cent %s"
              % (band, len(d), np.median(d), 1.4826 * np.median(np.abs(d - np.median(d))), np.percentile(d, [5, 95]).round(3)))
    for o in pma:
        if "reading" in o:
            print("  %s: Ks-W3 %+.3f (%.1f sigma), Ks-W4 %+.3f (%.1f sigma); All-Sky %+.3f and %+.3f; W3 %.3f against %s, W4 %.3f against %s; %s" % (
                o["name"], o["W3_excess_mag"], o["W3_excess_sig"], o["W4_excess_mag"], o["W4_excess_sig"],
                o.get("W3_excess_mag_allsky") or 0, o.get("W4_excess_mag_allsky") or 0, o["W3"], o["W3_allsky"], o["W4"], o["W4_allsky"],
                o["reading"]))
    dark = json.load(open(results("dark_companion_ds.json")))
    print("dark companions: %d" % len(dark))


if __name__ == "__main__":
    main()
