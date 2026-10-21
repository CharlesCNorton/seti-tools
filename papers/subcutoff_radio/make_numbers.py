"""Numbers of the sub-cutoff radio paper: allocated band fractions, Wind and STEREO channel scatter, flux thresholds and reach, and Wind line classes."""
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.argv = sys.argv[:1]
from setitools import core  # noqa: E402
from setitools import radio_sky as rs  # noqa: E402

# Cane (1979) polar galactic background (W m^-2 Hz^-1 sr^-1) with free-free turnover, nu in MHz
def cane(nu_mhz):
    tau = 5.0 * nu_mhz ** -2.1
    return 2.48e-20 * nu_mhz ** -0.52 * (1 - math.exp(-tau)) / tau + 1.06e-20 * nu_mhz ** -0.80 * math.exp(-tau)


def union(ivs):
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(ivs, lo, hi):
    return sum(max(0, min(b, hi) - max(a, lo)) for a, b in union(ivs))


def bands():
    print("== bands")
    sw = [(a - 25, b + 25) for a, b in rs.wind_SW_BANDS]
    mob = [(a - 25, b + 25) for a, b in rs.wind_MARITIME + rs.wind_AERONAUTICAL]
    ama = list(rs.wind_AMATEUR)
    oob = [(a - rs.wind_OUT_OF_BAND, b + rs.wind_OUT_OF_BAND) for a, b in rs.wind_SW_BANDS]
    nat = [(0, 100), (100, 700)]
    for name, lo, hi in (("RAD1", 20, 1040), ("RAD2", 1075, 13825), ("STEREO HFR", 125, 16025)):
        print("%s %d-%d kHz: service bands with 25 kHz margins %.1f%%; with out-of-band broadcasting %.1f%%; plasma line and "
              "auroral ranges %.1f%%; all known %.1f%%" % (
                  name, lo, hi, 100 * covered(sw + mob + ama, lo, hi) / (hi - lo), 100 * covered(sw + mob + ama + oob, lo, hi) / (hi - lo),
                  100 * covered(nat, lo, hi) / (hi - lo), 100 * covered(sw + mob + ama + oob + nat, lo, hi) / (hi - lo)))


def neighbour_scatter(p):
    """Median over channels of the robust deviation from the eight-neighbour median, relative to it, over one day's spectra."""
    n, m = p.shape
    rel = []
    for j in range(4, m - 4):
        nb = np.concatenate([p[:, j - 4:j], p[:, j + 1:j + 5]], axis=1)
        med = np.nanmedian(nb, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(nb - med[:, None]), axis=1)
        rel.append(np.nanmedian(mad / med))
    return np.array(rel)


def wind_scatter(years=(1999, 2020), per_year=6):
    """Relative channel scatter of RAD2 in each year's cached days, and of RAD1 at its measured channels."""
    import cdflib
    print("== Wind channel scatter")
    for y in years:
        files = sorted(glob.glob(os.path.join(rs.wind_ROOT, str(y), "*.cdf")))
        if not files:
            print("%d: no cached days" % y)
            continue
        files = [files[i] for i in np.linspace(0, len(files) - 1, min(per_year, len(files))).astype(int)]
        r2, r1, nmeas = [], [], []
        for f in files:
            c = cdflib.CDF(f)
            x = c.varget("E_VOLTAGE_RAD2").astype(float)
            x[x <= 0] = np.nan
            r2.append(np.nanmedian(neighbour_scatter(x ** 2)))
            fr = np.atleast_2d(c.varget("Frequency_RAD1"))[0]
            vm = np.atleast_2d(c.varget("Minimum_voltage_RAD1"))[0]
            meas = np.nonzero(vm > 0)[0]
            nmeas.append(len(meas))
            x = c.varget("E_VOLTAGE_RAD1").astype(float)
            x[x <= 0] = np.nan
            p = x[:, meas] ** 2
            rel = neighbour_scatter(p) if p.shape[1] > 8 else np.array([np.nan])
            hi = fr[meas][4:-4] >= 300 if p.shape[1] > 8 else np.array([])
            r1.append((np.nanmedian(rel[~hi]) if (~hi).any() else np.nan, np.nanmedian(rel[hi]) if hi.any() else np.nan))
        print("%d (%d days): RAD2 relative scatter median %.3f; RAD1 measured channels %d-%d, scatter below 300 kHz %.2f, above %.2f"
              % (y, len(files), np.nanmedian(r2), min(nmeas), max(nmeas), np.nanmedian([a for a, _ in r1]), np.nanmedian([b for _, b in r1])))
    r = 0.024
    for nu in (2.0, 5.0, 10.0):
        B = cane(nu)
        S = 8 * math.pi / 3 * B
        F = 5 * r * S * 20e3
        print("RAD2 at %.0f MHz: Cane background %.1e W/m2/Hz/sr, dipole-equivalent flux density %.1e W/m2/Hz, 5 sigma line %.1e W/m2: "
              "isotropic %.1e W at 1 au, %.1e W at 1 pc, %.1e W at 10 pc" % (
                  nu, B, S, F, 4 * math.pi * (1.495978707e11) ** 2 * F, 4 * math.pi * (3.0857e16) ** 2 * F, 4 * math.pi * (3.0857e17) ** 2 * F))


def stereo_scatter():
    """STEREO relative channel scatter in raw power and background-subtracted flux, with the background in flux units."""
    import cdflib
    files = sorted(glob.glob(os.path.join(core.data("stereo"), "*.cdf")))[:10]
    if not files:
        print("== STEREO: no cached days")
        return
    print("== STEREO channel scatter (%d days)" % len(files))
    raw_r, flux_r, bg_flux, level = [], [], [], []
    for f in files:
        c = cdflib.CDF(f)
        fr = np.atleast_2d(c.varget("FREQUENCY"))[0].astype(float)
        pf = c.varget("PSD_FLUX").astype(float)
        pf[~(pf > 0)] = np.nan
        v2 = c.varget("PSD_V2").astype(float)[:, :, 0, 0]
        v2[~(v2 > 0)] = np.nan
        bg = c.varget("BACKGROUND").astype(float)
        bgc = bg[0, :, 0, 0] if bg.ndim == 4 else bg[:, 0, 0]
        hi = fr > 2e6
        flux_r.append(np.nanmedian(neighbour_scatter(pf)[hi[4:-4]]))
        raw_r.append(np.nanmedian(neighbour_scatter(v2)[hi[4:-4]]))
        with np.errstate(all="ignore"):
            cal = np.nanmedian(pf / (v2 - bgc[None, :]), axis=0)
        bg_flux.append(np.nanmedian((bgc * cal)[hi]))
        level.append(np.nanmedian(pf[:, hi]))
    print("above 2 MHz: raw power scatter %.3f of the level, background-subtracted flux scatter %.3f of a median level of %.1e W/m2/Hz; "
          "background in flux units %.1e W/m2/Hz; 5 sigma line in a 50 kHz channel %.1e W/m2 (raw) or %.1e (subtracted)"
          % (np.median(raw_r), np.median(flux_r), np.median(level), np.median(bg_flux),
             5 * np.median(raw_r) * np.median(bg_flux) * 5e4, 5 * np.median(flux_r) * np.median(level) * 5e4))


def results_glob(pattern):
    """Result files from data/, else the results directory."""
    here = sorted(glob.glob(os.path.join(HERE, "data", pattern)))
    return here or sorted(glob.glob(core.results(pattern)))


def classes():
    print("== Wind line classes")
    for p in results_glob("wind_*.json"):
        d = json.load(open(p))
        if "RAD1" not in d:
            continue
        for band in ("RAD1", "RAD2"):
            ex = d[band]["explained"]
            cls = {}
            for v in ex.values():
                k = re.sub(r"\(.*\)", "", v).strip()
                cls[k] = cls.get(k, 0) + 1
            print("%s %s: %d persistent, %d left; %s" % (os.path.basename(p)[5:9], band, d[band]["persistent"], len(d[band]["left"]),
                                                       ", ".join("%s %d" % kv for kv in sorted(cls.items(), key=lambda kv: -kv[1]))))
    for p in results_glob("wind_rad1_*.json"):
        d = json.load(open(p))
        pers = [c for c in d["channels"] if c["occupancy"] > 0.01 and c["days"] >= 30]
        print("RAD1 measured channels %s: %d channels, %d persistent lines (%s), %d above 700 kHz; scatter below 300 kHz %.2f, above %.2f" % (
            d["year"], len(d["channels"]), len(pers), ", ".join(sorted({c["known"] or "unexplained" for c in pers})),
            sum(1 for c in pers if c["khz"] > 700), np.median([c["rel_dev"] for c in d["channels"] if c["khz"] < 300]),
            np.median([c["rel_dev"] for c in d["channels"] if c["khz"] >= 300])))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    bands()
    classes()
    wind_scatter()
    stereo_scatter()
