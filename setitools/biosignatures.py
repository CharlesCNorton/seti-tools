"""Biosignatures: gases of life and their abiotic mimics in the public spectra of temperate worlds."""
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from setitools import core

ARCHIVE = "https://exoplanetarchive.ipac.caltech.edu"


def exo_workspace():
    """A workspace path the archive's spectra page creates, under which its spectrum files are served."""
    t = core.get(ARCHIVE + "/cgi-bin/atmospheres/nph-firefly?atmospheres", timeout=120).decode("utf-8", "replace")
    m = re.search(r"FF_InitPage \('[^']*', '(/workspace/TMP_[A-Za-z0-9_]+)'", t)
    if not m:
        raise IOError("the archive's spectra page gave no workspace")
    return m.group(1)


def exo_spectrum(spec_path, workspace=None):
    """An archive spectrum file as arrays of wavelength (um), half bin width (um), transit or eclipse depth and its error (ppm), limits left out, downloaded once into the cache."""
    local = core.data("exospectra", spec_path.replace("/", "_"))
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        ws = workspace or exo_workspace()
        open(local, "wb").write(core.get(ARCHIVE + ws + "/atmospheres/tab1/data/" + spec_path, timeout=120))
    cols, rows = None, []
    for line in open(local, encoding="utf-8", errors="replace"):
        if line.startswith("|") and cols is None:
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            continue
        if line.startswith(("\\", "|")) or not line.strip():
            continue
        rows.append(line.split())
    k = {c: i for i, c in enumerate(cols)}

    def col(name):
        return np.array([float(r[k[name]]) if r[k[name]] not in ("null", "") else np.nan for r in rows])
    w, bw = col("CENTRALWAVELNG"), col("BANDWIDTH")
    # transit files give PL_TRANDEP and eclipse files ESPECLIPDEP, both in per cent, a nonzero LIM marking a limit
    dep = "PL_TRANDEP" if "PL_TRANDEP" in k else "ESPECLIPDEP"
    d = col(dep) * 1e4
    e = 0.5 * (np.abs(col(dep + "ERR1")) + np.abs(col(dep + "ERR2"))) * 1e4
    lim = col(dep + "LIM") if dep + "LIM" in k else np.zeros(len(d))
    ok = np.isfinite(w) & np.isfinite(d) & np.isfinite(e) & (e > 0) & ~(np.nan_to_num(lim) != 0)
    w, bw, d, e = w[ok], bw[ok], d[ok], e[ok]
    if len(w) > 1 and not np.all(np.isfinite(bw)):
        # a file without bin widths takes them from the spacing of its wavelengths
        bw = np.where(np.isfinite(bw), bw, np.abs(np.gradient(w)))
    return w, bw / 2.0, d, e


def temperate_spectra(rade_max=3.5, teq_max=450.0, facility="James Webb"):
    """The archive's transmission spectra of small temperate planets from one facility, one reference per planet and instrument, each visit or detector once, with each planet's parameters."""
    pl = core.exoplanets("SELECT pl_name, pl_rade, pl_bmasse, pl_eqt, pl_insol, st_rad, st_teff FROM pscomppars WHERE pl_rade < %g AND (pl_eqt < %g OR pl_insol < 3)" % (rade_max, teq_max))
    planets = {p["pl_name"]: p for p in pl}
    sp = core.exoplanets("SELECT pl_name, spec_type, authors, num_datapoints, instrument, minwavelng, maxwavelng, note, bibcode, spec_path FROM spectra WHERE spec_type = 'Transmission' "
                         "AND facility LIKE '%%%s%%'" % facility)
    sp = [s for s in sp if s["pl_name"] in planets]
    chosen = {}
    for s in sp:
        key = (s["pl_name"], s["instrument"])
        chosen.setdefault(key, {}).setdefault(s["authors"], []).append(s)
    out = {}
    for (name, inst), refs in chosen.items():
        best = max(refs.values(), key=lambda v: max(int(x["num_datapoints"] or 0) for x in v))
        out.setdefault(name, []).extend(distinct_spectra(best))
    return planets, out


def distinct_spectra(group):
    """One spectrum per visit and detector from one reference's spectra of one instrument: native binning over low, a combined spectrum over its visits, and the plainest of alternative reductions."""
    notes = [(x.get("note") or "").lower() for x in group]
    if any("native" in n for n in notes):
        group = [x for x, n in zip(group, notes) if "low resolution" not in n]
    comb = [x for x in group if re.search(r"combined|joint", (x.get("note") or "").lower())]
    if comb:
        group = comb
    keep = {}
    for x in group:
        n = (x.get("note") or "").lower()
        visit = re.search(r"visit\s*(\d+)", n)
        det = re.search(r"nrs\s*[12]|g235h|g395h|g140h|g395m", n)
        key = (visit.group(1) if visit and not comb else None, det.group(0).replace(" ", "") if det else None)
        if key not in keep or (len(n), -int(x["num_datapoints"] or 0)) < (len((keep[key].get("note") or "")), -int(keep[key]["num_datapoints"] or 0)):
            keep[key] = x
    return list(keep.values())


def band_template(w, hw, bands):
    """A gas's band template averaged over each bin: Gaussian bands of the given centres, widths and strengths."""
    from scipy.special import erf
    t = np.zeros(len(w))
    for c, fwhm, a in bands:
        s = fwhm / 2.355
        lo, hi = (w - hw - c) / (np.sqrt(2) * s), (w + hw - c) / (np.sqrt(2) * s)
        t += a * 0.5 * (erf(hi) - erf(lo)) * s * np.sqrt(2 * np.pi) / np.maximum(2 * hw, 1e-6)
    return t


def scale_height_ppm(p, mu=2.3):
    """The transit depth one atmospheric scale height adds for a planet, in ppm, at mean molecular weight mu."""
    rp, mp, teq, rs = (core.num(p.get(k)) for k in ("pl_rade", "pl_bmasse", "pl_eqt", "st_rad"))
    if None in (rp, mp, teq, rs) or mp <= 0:
        return None
    g = 6.674e-11 * mp * 5.972e24 / (rp * 6.371e6) ** 2
    h = 1.380649e-23 * teq / (mu * 1.6605e-27 * g)
    return 2 * rp * 6.371e6 * h / (rs * 6.957e8) ** 2 * 1e6


disequilibrium_DOC = r"""Methane and carbon dioxide without carbon monoxide: every JWST transmission spectrum the NASA Exoplanet Archive holds for planets below 3.5 Earth radii that are temperate (equilibrium temperature below 450 K or insolation below three Earths'), one reference per planet and instrument, fitted for each planet at once with an offset per spectrum and templates of H2O, CH4, CO2, CO, NH3, dimethyl sulfide, N2O, SO2, OCS and O3 built from their cross-sections at the planet's temperature; each gas counted by how much the fit worsens without it, so that overlapping bands cannot share one feature; the pair CH4 and CO2 at 3 sigma with CO below 2 sigma is the chemical disequilibrium that methanogenic life keeps and that hydrogen-rich mini-Neptunes and magma oceans can also make, and dimethyl sulfide at 3 sigma is the gas claimed for K2-18 b; band amplitudes are given in ppm and in scale heights of a hydrogen atmosphere.

    python -I seti.py biosignatures disequilibrium [record] [--rade-max 3.5] [--teq-max 450] [--sigma 3]"""


def disequilibrium_main(record=False, rade_max=3.5, teq_max=450.0, sigma=3.0):
    planets, spectra = temperate_spectra(rade_max, teq_max)
    ws = exo_workspace()
    out = []
    for name in sorted(spectra):
        parts = []
        for s in spectra[name]:
            try:
                parts.append((s, exo_spectrum(s["spec_path"], ws)))
            except Exception as e:
                print("  %s: %s unreadable (%s)" % (name, s["spec_path"], str(e)[:60]))
        parts = [(s, x) for s, x in parts if len(x[0]) >= 5]
        if not parts:
            continue
        w = np.concatenate([x[0] for _, x in parts])
        hw = np.concatenate([x[1] for _, x in parts])
        d = np.concatenate([x[2] for _, x in parts])
        e = np.concatenate([x[3] for _, x in parts])
        cols, names = [], []
        for k, (_, x) in enumerate(parts):
            c = np.zeros(len(w))
            start = sum(len(y[0]) for _, y in parts[:k])
            c[start:start + len(x[0])] = 1.0
            cols.append(c)
            names.append("offset %d" % k)
        gases = []
        for gas in LINE_GASES:
            t = gas_template(gas, w, hw, core.num(planets[name]["pl_eqt"]) or XS_T)
            if np.ptp(t) > 0.05:
                cols.append(t)
                names.append(gas)
                gases.append(gas)
        A = np.array(cols).T
        W = 1.0 / e ** 2

        def fit(M):
            cov = np.linalg.pinv(M.T @ (M * W[:, None]))
            par = cov @ (M.T @ (W * d))
            return par, cov, float(np.sum(W * (d - M @ par) ** 2))
        par, cov, chi = fit(A)
        chi_flat = fit(np.array(cols[:len(parts)]).T)[2]
        amp = {g: (float(par[names.index(g)]), float(math.sqrt(max(cov[names.index(g), names.index(g)], 0)))) for g in gases}
        sig = {}
        for g in gases:
            j = names.index(g)
            dchi = fit(np.delete(A, j, axis=1))[2] - chi
            # a gas counts by how much the fit worsens without it, signed by its band's sense
            sig[g] = math.copysign(math.sqrt(max(dchi, 0.0)), amp[g][0])
        hppm = scale_height_ppm(planets[name])
        diseq = sig.get("CH4", 0) >= sigma and sig.get("CO2", 0) >= sigma and sig.get("CO", 0) < 2.0
        p = planets[name]
        o = dict(planet=name, rade=core.num(p["pl_rade"]), teq=core.num(p["pl_eqt"]), st_teff=core.num(p["st_teff"]), n=len(w), spectra=[(s["instrument"], s["authors"], s.get("note")) for s, _ in parts],
                 chi2=chi, chi2_flat=chi_flat, dof=len(w) - len(names), amp=amp, sig=sig, h_ppm=hppm, diseq=diseq, dms=sig.get("DMS", 0) >= sigma)
        o["vet"] = None
        if diseq or o["dms"]:
            # hydrogen-rich mini-Neptunes and magma oceans keep CH4 and CO2 without CO by abiotic chemistry; cool M-dwarf hosts add spots and faculae to the spectrum
            o["vet"] = "abiotic-chemistry" if (o["rade"] or 0) > 1.6 else ("stellar-contamination" if (o["st_teff"] or 9999) < 3500 else None)
        out.append(o)
    print("%d temperate small planets with JWST transmission spectra; CH4 and CO2 at %g sigma without CO: %d; DMS at %g sigma: %d" % (
        len(out), sigma, sum(o["diseq"] for o in out), sigma, sum(o["dms"] for o in out)))
    for o in out:
        print("  %-13s %.2f R_E, %s K, %d points (%s): chi2 %.0f against %.0f flat for %d dof; %s; one scale height %s ppm%s" % (
            o["planet"], o["rade"] or 0, o["teq"], o["n"], "; ".join("%s %s" % (i.split("(")[-1].rstrip(")")[:12], a) for i, a, _ in o["spectra"]), o["chi2"], o["chi2_flat"], o["dof"],
            ", ".join("%s %+.1f" % (g, s) for g, s in sorted(o["sig"].items(), key=lambda kv: -kv[1])), "%.0f" % o["h_ppm"] if o["h_ppm"] else "-",
            "; DISEQUILIBRIUM, %s" % (o["vet"] or "unexplained") if o["diseq"] else ("; DMS, %s" % (o["vet"] or "unexplained") if o["dms"] else "")))
    json.dump(out, open(core.results("biosig_disequilibrium.json"), "w"), indent=1, default=float)
    if record:
        core.record_run(
            "jwst_temperate_disequilibrium", "Methane and carbon dioxide without carbon monoxide", "NASA Exoplanet Archive atmospheric spectroscopy table (JWST transmission spectra); planetary parameters from the composite table",
            "planets below %g Earth radii with equilibrium temperature below %g K or insolation below three Earths', with JWST transmission spectra: %d" % (rade_max, teq_max, len(out)),
            os.path.abspath(__file__), "disequilibrium", len(out), "planets", dict(rade_max=rade_max, teq_max=teq_max, sigma=sigma),
            "a flat spectrum with an offset per spectrum; the band templates' formal errors",
            "CH4 and CO2 at %g sigma without CO at 2 sigma, or dimethyl sulfide at %g sigma" % (sigma, sigma),
            "%d with the disequilibrium pair, %d with dimethyl sulfide" % (sum(o["diseq"] for o in out), sum(o["dms"] for o in out)),
            [dict(target="planet:%s" % o["planet"], stat=round(min(o["sig"].get("CH4", 0), o["sig"].get("CO2", 0)), 2), unit="sigma of the weaker of CH4 and CO2", vet=o["vet"],
                  note="%s; CO %+.1f sigma, DMS %+.1f sigma, chi2 %.0f against %.0f flat" % (", ".join("%s %+.1f" % (g, s) for g, s in o["sig"].items()), o["sig"].get("CO", 0), o["sig"].get("DMS", 0),
                                                                                         o["chi2"], o["chi2_flat"])) for o in out])
        print("recorded")


# a planet's name as papers and file names write it: survey or catalogue, number, an optional component and a planet letter
PLANET_RE = re.compile(r"\b(WASP|HAT-?P|HATS|KELT|XO|TrES|Qatar|CoRoT|Kepler|K2|TOI|GJ|LHS|LTT|LP|HD|HIP|TRAPPIST|L|WD|TIC|NGTS|KOI|NGC|HR|AU Mic|"
                       r"V1298 Tau|Ross|Wolf|Teegarden's Star|Barnard's Star|Gliese)[-_ ]?([0-9][0-9.\-+]*)[ _-]?(A|B)?[ _-]?([b-i])\b", re.I)


NAMED_RE = re.compile(r"\b(55 Cnc|AU Mic|pi Men|tau Boo|51 Peg|HR 8799|beta Pic|51 Eri|kappa And|HD 189733|HD 209458|HD 149026|HD 80606|HD 97658|HD 3167|HD 106315|HD 219134|HD 260655|HD 15337|HD 73583)[ _-]?([b-i])\b", re.I)


def planet_names(text):
    """Planet names written in a text, normalised as the catalogues write them (WASP-39 b, TRAPPIST-1 e, L 98-59 d, GJ 1214 b)."""
    canon = {n.lower(): n for n in NAMED_RE.pattern.split("(")[1].split(")")[0].split("|")}
    out = ["%s %s" % (canon.get(m.group(1).lower(), m.group(1)), m.group(2).lower()) for m in NAMED_RE.finditer(text or "")]
    for m in PLANET_RE.finditer(text or ""):
        survey, num, comp, letter = m.group(1), m.group(2).rstrip(".-"), m.group(3) or "", m.group(4).lower()
        su = {"hatp": "HAT-P", "hat-p": "HAT-P", "gliese": "GJ"}.get(survey.lower(), survey)
        su = {"wasp": "WASP", "hats": "HATS", "kelt": "KELT", "xo": "XO", "tres": "TrES", "qatar": "Qatar", "corot": "CoRoT", "kepler": "Kepler",
              "k2": "K2", "toi": "TOI", "gj": "GJ", "lhs": "LHS", "ltt": "LTT", "lp": "LP", "hd": "HD", "hip": "HIP", "trappist": "TRAPPIST",
              "l": "L", "wd": "WD", "tic": "TIC", "ngts": "NGTS", "koi": "KOI", "hr": "HR"}.get(su.lower(), su)
        sep = " " if su in ("GJ", "LHS", "LTT", "LP", "HD", "HIP", "L", "WD", "TIC", "HR", "Ross", "Wolf", "BD", "G", "55 Cnc") else "-"
        out.append("%s%s%s%s %s" % (su, sep, num, " " + comp if comp else "", letter))
    return list(dict.fromkeys(out))


class _Columns:
    """A table as named columns, enough for the column reader: colnames and indexing by name."""

    def __init__(self, names, rows, note=""):
        self.colnames, self._a, self.note = list(names), np.asarray(rows, dtype=float), note

    def __getitem__(self, name):
        return self._a[:, self.colnames.index(name)]


def _text_table(path):
    """A plain text table read line by line: comment lines skipped, the last of them or a non-numeric first line as the header, comma, tab or space delimited; None when fewer than three numeric columns or four rows."""
    header, data = None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        t = line.strip()
        if not t:
            continue
        if t[0] in "#%\\|;!" or t.startswith("//"):
            h = t.lstrip("#%\\|;!/ ").strip()
            if re.search(r"[A-Za-z]", h):
                header = h
            continue
        data.append(t)
    if len(data) < 5:
        return None
    delim = "," if data[0].count(",") >= 2 else ("\t" if "\t" in data[0] else None)

    def split(t):
        out = [x.strip() for x in (t.split(delim) if delim else t.split())]
        while out and out[-1] == "":
            out.pop()
        return out

    def number(x):
        try:
            return float(x)
        except ValueError:
            return None if x not in ("", "nan", "NaN", "null", "--", "NA") else float("nan")
    first = split(data[0])
    if any(number(x) is None for x in first):
        names, rows = first, data[1:]
    else:
        names, rows = (split(header) if header else None), data
    out, ncol = [], None
    for t in rows:
        vals = [number(x) for x in split(t)]
        if any(v is None for v in vals):
            continue
        ncol = ncol or len(vals)
        if len(vals) == ncol:
            out.append(vals)
    if len(out) < 4 or not ncol or ncol < 3:
        return None
    if not names or len(names) != ncol:
        names = ["col%d" % (i + 1) for i in range(ncol)]
    return _Columns(names, out, note=header or "")


def table_spectrum(path, hint=""):
    """A published spectrum file read as wavelength (um), half bin width (um), depth and its error (ppm), with its kind (transmission or emission) guessed from its columns and the hint text; None when no column set fits."""
    from astropy.io import ascii as aascii
    from astropy.io import fits
    from astropy.table import Table
    tables = []
    low = path.lower()
    try:
        if low.endswith((".fits", ".fit", ".fits.gz")):
            with fits.open(path) as h:
                for e in h[1:]:
                    if hasattr(e, "columns"):
                        tables.append(Table(e.data))
        elif low.endswith(".ecsv"):
            tables.append(Table.read(path, format="ascii.ecsv"))
        elif low.endswith((".npy",)):
            a = np.load(path, allow_pickle=False)
            if a.ndim == 2 and min(a.shape) >= 3:
                tables.append(Table(a if a.shape[1] < a.shape[0] else a.T))
        else:
            head = open(path, encoding="utf-8", errors="replace").read(4000)
            t = aascii.read(path, format="mrt") if "Byte-by-byte Description" in head else _text_table(path)
            if t is None and os.path.getsize(path) < 2e6:
                t = aascii.read(path, guess=True, fast_reader=False)
            if t is not None:
                tables.append(t)
    except Exception:
        return None
    head = ""
    if not low.endswith((".fits", ".fit", ".fits.gz", ".npy")):
        head = open(path, encoding="utf-8", errors="replace").read(2000)
    for t in tables:
        r = _columns_spectrum(t, hint + " " + os.path.basename(path) + " " + getattr(t, "note", "") + " " + head)
        if r is not None:
            return r
    return None


def _unnamed_spectrum(cols, note):
    """A table without column names read by the layouts papers use: wavelength first, then bin edges, a width or a half width, then depth and error, told apart by their sizes."""
    c = [np.asarray(x, dtype=float) for x in cols]
    w = c[0]
    f = np.isfinite(w)
    # wavelengths that mostly rise, as a table of two detectors or orders in turn does, are put in order with their rows
    if f.sum() < 5 or np.mean(np.diff(w[f]) > 0) < 0.8 or not (0.2 <= np.nanmedian(w) <= 30):
        return None
    order = np.argsort(w)
    c = [x[order] for x in c]
    w = c[0]
    f = np.isfinite(w)
    hw = None
    rest = c[1:]
    if len(c) >= 4 and np.all(c[1][f] > w[f]) and np.nanmedian((c[1] - w) / w) < 0.2:
        hw, w, rest = (c[1] - w) / 2, (c[1] + w) / 2, c[2:]
    elif len(c) >= 4 and np.nanmedian(c[1] / w) < 0.2:
        step = np.nanmedian(np.diff(w))
        half = re.search(r"half", note.lower()) or abs(np.nanmedian(c[1]) - step / 2) < abs(np.nanmedian(c[1]) - step)
        hw, rest = (c[1] if half else c[1] / 2), c[2:]
    if len(rest) < 2:
        return None
    d, e = rest[0], np.abs(rest[1])
    if len(rest) >= 3 and np.nanmedian(np.abs(rest[2])) < np.nanmedian(np.abs(d)) and np.nanmedian(e) < np.nanmedian(np.abs(d)):
        # an upper and a lower error: their mean
        e = (np.abs(rest[1]) + np.abs(rest[2])) / 2
    return w, (hw if hw is not None else np.abs(np.gradient(w)) / 2), d, e


def _columns_spectrum(t, hint):
    """The wavelength, depth, error and bin-width columns of a table by name and by value, converted to um and ppm."""
    names = list(t.colnames)
    num_cols = {}
    for n in names:
        try:
            v = np.asarray(t[n], dtype=float)
        except (TypeError, ValueError):
            continue
        if v.ndim == 1 and np.isfinite(v).sum() >= 4:
            num_cols[n] = v
    if len(num_cols) < 3:
        return None
    low = {n: n.lower().replace(" ", "") for n in num_cols}
    if all(re.match(r"^col\d+$", n) for n in num_cols):
        u = _unnamed_spectrum([num_cols[n] for n in num_cols], hint)
        if u is None:
            return None
        w, hw, d, e = u
        md = np.nanmedian(np.abs(d))
        if md < 0.2:
            d, e = d * 1e6, e * 1e6
        elif md < 20 and not re.search(r"ppm", hint.lower()):
            d, e = d * 1e4, e * 1e4
        ok = np.isfinite(w) & np.isfinite(hw) & np.isfinite(d) & np.isfinite(e) & (e > 0) & (w > 0.2) & (w < 30)
        if ok.sum() < 5:
            return None
        text = hint.lower()
        kind = "emission" if re.search(r"emission|eclipse|dayside|fp/?fs|fpfs|secondary|occultation", text) and not re.search(r"transmission|transit.?depth|rp/?rs", text) else "transmission"
        return dict(w=w[ok], hw=hw[ok], d=d[ok], e=e[ok], kind=kind, columns=("col1", "layout", "", ""))

    def pick(patterns, exclude=()):
        for pat in patterns:
            for n in num_cols:
                if re.search(pat, low[n]) and not any(re.search(x, low[n]) for x in exclude):
                    return n
        return None
    wl = pick([r"^wave|^wl|^lam|wavelength|^wav|micron|^w$|^x$"], exclude=[r"err|width|bin|low|up|min|max|edge"])
    if wl is None and re.search(r"spec|transmission|emission|depth|eclipse|transit|fpfs|tspec|dayside", hint.lower()):
        # the first column whose values look like wavelengths in um, nm or Angstrom and rise
        for n, v in num_cols.items():
            f = v[np.isfinite(v)]
            if len(f) and np.all(np.diff(f) >= 0) and (0.2 <= np.median(f) <= 30 or 200 <= np.median(f) <= 30000):
                wl = n
                break
    if wl is None:
        return None
    w = num_cols[wl]
    med = np.nanmedian(w)
    scale = 1.0 if med < 40 else (1e-3 if med < 40000 else 1e-4)
    rprs = pick([r"rp/?rs|rprs|radius.?ratio|^k$|r_p/r_s|rp_rs"], exclude=[r"err|unc|sig|low|up"])
    dep = pick([r"depth|trandep|dppm|^td$|transit|^d$|fp/?fs|fpfs|eclipse|ppm|occult|flux.?ratio|^y$|signal|spec"], exclude=[r"err|unc|sig|low|up|wave|wl|lam|width|bin"])
    col = rprs or dep
    if col is None:
        # a column that holds the signal: named for flux, spectrum or ratio first, then any that is not an error, a width or a wavelength
        rest = [n for n in num_cols if n != wl and not re.search(r"err|unc|sig|width|bin|wave|wl|lam|^e_|_e$|model|fit", low[n])]
        named = [n for n in rest if re.search(r"flux|spec|ratio|signal|abs|excess", low[n])]
        if not (named or rest):
            return None
        col = (named or rest)[0]
    err = None
    for n in num_cols:
        if n in (wl, col):
            continue
        if re.search(r"err|unc|sig|^e_|_e$|^d" + re.escape(low[col]) + r"|\+|-", low[n]) and not re.search(r"wave|wl|lam|width|bin", low[n]):
            err = n if err is None else err
    if err is None:
        cands = [n for n in num_cols if n not in (wl, col)]
        for n in cands:
            v = num_cols[n]
            if np.nanmedian(np.abs(v)) < np.nanmedian(np.abs(num_cols[col])) and np.all(np.asarray(v)[np.isfinite(v)] >= 0):
                err = n
                break
    if err is None:
        return None
    hw_col = pick([r"half|bin.?width|width|^dw|dlam|wave.?err|wl.?err|^bw"], exclude=[r"depth|flux"])
    if re.search(r"flux|fnu|flam|mjy|ujy|jy|erg|w/m", low[col]) and not re.search(r"ratio|depth|fp|rp|transit|eclipse|occult|contrast", low[col]):
        return None
    d, e = num_cols[col].copy(), np.abs(num_cols[err].copy())
    if col == rprs:
        e, d = 2 * np.abs(d) * e, d ** 2
    md = np.nanmedian(np.abs(d))
    if md < 0.2:
        d, e = d * 1e6, e * 1e6
    elif md < 20 and not re.search(r"ppm", low[col]):
        d, e = d * 1e4, e * 1e4
    hw = np.abs(num_cols[hw_col]) * scale if hw_col else np.full(len(w), np.nan)
    if hw_col and re.search(r"width|^bw|^dw|bin", low[hw_col]) and not re.search(r"half", low[hw_col]):
        hw = hw / 2.0
    w = w * scale
    ok = np.isfinite(w) & np.isfinite(d) & np.isfinite(e) & (e > 0) & (w > 0.2) & (w < 30)
    if ok.sum() < 4:
        return None
    w, hw, d, e = w[ok], hw[ok], d[ok], e[ok]
    if not np.all(np.isfinite(hw)):
        hw = np.where(np.isfinite(hw), hw, np.abs(np.gradient(w)) / 2.0 if len(w) > 1 else 0.01)
    order = np.argsort(w)
    text = (hint + " " + " ".join(names)).lower()
    kind = "emission" if re.search(r"emission|eclipse|dayside|fp/?fs|fpfs|secondary|occultation", text) and not re.search(r"transmission|transit.?depth|rp/?rs", text) else "transmission"
    return dict(w=w[order], hw=hw[order], d=d[order], e=e[order], kind=kind, columns=(wl, col, err, hw_col))


ZENODO_API = "https://zenodo.org/api/records"
ZENODO_QUERIES = (
    'JWST "transmission spectrum"', 'JWST "transmission spectra"', 'JWST "emission spectrum"', 'JWST "eclipse spectrum"', 'JWST "phase curve"',
    'JWST exoplanet NIRSpec', 'JWST exoplanet NIRISS', 'JWST exoplanet MIRI', 'JWST exoplanet NIRCam', 'JWST exoplanet atmosphere spectrum',
    'HST WFC3 "transmission spectrum"', 'Hubble "transmission spectrum" exoplanet', 'STIS "transmission spectrum" exoplanet',
    '"transmission spectrum" exoplanet data', '"secondary eclipse" spectrum JWST', 'exoplanet "spectroscopic light curves"', 'Eureka! JWST exoplanet',
    'ExoTiC JWST', '"JWST-TST"', 'JWST ERS transiting exoplanet', 'JWST "sub-Neptune" spectrum', 'JWST super-Earth spectrum', 'JWST rocky planet eclipse',
    'JWST hot Jupiter spectrum', 'HST WFC3 population study transmission spectra', 'population exoplanet atmospheres transmission spectra',
    'homogeneous survey transmission spectra', 'panchromatic transmission spectrum', 'PanCET', 'BOWIE-ALIGN', 'Hidden GEMS JWST', 'COMPASS JWST',
    'Hot Rocks Survey', 'JWST MIRI LRS transmission', 'JWST NIRSpec G395H transmission', 'emission spectrum hot Jupiter HST',
    'thermal emission spectrum rocky exoplanet', 'SPACE program transmission spectrum', 'transmission spectrum sub-Neptune',
    'transmission spectrum super-Earth', 'transmission spectrum mini-Neptune', 'dayside spectrum exoplanet', 'eclipse depths exoplanet spectrum',
    'Spitzer secondary eclipse exoplanet', 'exoplanet atmosphere retrieval data products',
)
# a record that names no planet in its title is still taken when it speaks of planetary spectra, its planets named inside its files
SPECTRA_RECORD = re.compile(r"transmission spectr|emission spectr|eclipse spectr|dayside spectr|exoplanet atmospher|planetary atmospher|transit spectr", re.I)
# records of models, retrievals and opacities, whose large archives hold no observed spectrum
MODEL_RECORD = re.compile(r"\bmodels?\b|cross.?sections|opacit|grid|synthetic|simulat|\bGCM\b|photochem|retriev|uncalibrated|database", re.I)
SPECTRUM_EXT = (".txt", ".dat", ".csv", ".ecsv", ".tsv", ".tbl", ".fits", ".fit", ".npy", ".zip", ".tar", ".gz", ".tgz")
atlas_ROOT = os.environ.get("ATLAS_ROOT", core.data("spectra_atlas"))


def instrument_of(text):
    """The instrument and mode a file name or a title names, as the archive writes them."""
    t = (text or "").lower()
    for pat, name in ((r"prism", "NIRSpec PRISM"), (r"g395h", "NIRSpec G395H"), (r"g395m", "NIRSpec G395M"), (r"g235h", "NIRSpec G235H"),
                      (r"g140h", "NIRSpec G140H"), (r"nirspec", "NIRSpec"), (r"soss", "NIRISS SOSS"), (r"niriss", "NIRISS"), (r"miri.?lrs|\blrs\b", "MIRI LRS"),
                      (r"miri", "MIRI"), (r"f322w2", "NIRCam F322W2"), (r"f444w", "NIRCam F444W"), (r"nircam", "NIRCam"), (r"g141", "WFC3 G141"),
                      (r"g102", "WFC3 G102"), (r"wfc3", "WFC3"), (r"stis|g430l|g750l", "STIS"), (r"spitzer|irac", "Spitzer")):
        if re.search(pat, t):
            return name
    return None


def zenodo_search(queries=ZENODO_QUERIES, pages=4, threads=4):
    """Zenodo records the queries find, keyed by record id, with title, description, DOI and files: each query's best matches, several queries at once."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    def one(q):
        seen = {}
        for page in range(1, pages + 1):
            url = ZENODO_API + "?" + urllib.parse.urlencode(dict(q=q, size=25, page=page, sort="bestmatch"))
            d = None
            for attempt in range(4):
                try:
                    d = json.loads(core.get(url, timeout=60, retries=1))
                    break
                except Exception as e:
                    # Zenodo answers 429 to a guest asking too often; the query waits and asks again
                    if "429" not in str(e):
                        break
                    _time.sleep(15 * (attempt + 1))
            if d is None:
                break
            hits = d["hits"]["hits"]
            for h in hits:
                m = h["metadata"]
                seen[str(h["id"])] = dict(id=h["id"], q=q, title=m.get("title", ""), date=m.get("publication_date"), doi=m.get("doi"),
                                          desc=re.sub(r"<[^>]+>", " ", m.get("description") or "")[:2000],
                                          files=[dict(key=f["key"], size=f["size"], url=f["links"]["self"]) for f in h.get("files", [])])
            if len(hits) < 25:
                break
        return seen
    seen = {}
    with ThreadPoolExecutor(threads) as ex:
        for q, got in zip(queries, ex.map(one, queries)):
            for k, v in got.items():
                seen.setdefault(k, v)
            print("  %-45s %3d records, %d in all" % (q[:45], len(got), len(seen)), flush=True)
    return seen


def safe_member(root, name):
    """The path under root for an archive member's name, each part stripped of the characters Windows refuses, or None for a name that would leave root."""
    parts = [re.sub(r'[<>:"|?*\x00-\x1f]', "_", p).rstrip(" .") for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or ".." in parts or "__MACOSX" in parts or not all(parts):
        return None
    return os.path.join(root, *parts)


def zenodo_fetch(records, max_mb=300, threads=4):
    """Each record that names a planet: its files that may hold a spectrum, downloaded once into ROOT/zenodo/<id>/ on a few threads and archives unpacked beside them."""
    import tarfile
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    tasks = []
    for rid, r in records.items():
        text = " ".join([r["title"], r["desc"]] + [f["key"] for f in r["files"]])
        about = r["title"] + " " + r["desc"]
        # optics and materials records speak of transmission spectra too; one that names no planet must speak of planets
        if not planet_names(text) and not (SPECTRA_RECORD.search(about) and re.search(r"planet", about, re.I)):
            continue
        for f in r["files"]:
            # an archive too large to take whole is read by its members only when neither it nor its record is of models, retrievals or opacities
            big = f["size"] > max_mb * 1e6
            if not f["key"].lower().endswith(SPECTRUM_EXT) or NOT_PLANET.search(f["key"]) or (big and (not f["key"].lower().endswith(".zip") or MODEL_RECORD.search(r["title"]))):
                continue
            tasks.append((rid, f))

    def members(rid, f, max_member_mb=20):
        """A large zip's small tables read by byte range, without the rest of the archive."""
        out = os.path.join(atlas_ROOT, "zenodo", rid, os.path.basename(f["key"]) + "_x")
        if os.path.exists(out):
            return None
        try:
            z = core.open_zip(f["url"])
            for m in z.infolist():
                name = m.filename
                if m.is_dir() or ".." in name or name.startswith("/") or "__MACOSX" in name or not name.lower().endswith(TABLE_EXT) or m.file_size > max_member_mb * 1e6:
                    continue
                dst = os.path.join(out, *name.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                open(dst, "wb").write(z.read(m))
        except Exception as e:
            if os.path.isdir(out):
                import shutil
                shutil.rmtree(out)
            return "%s %s: members unread (%s)" % (rid, f["key"], str(e)[:60])
        return None

    def one(task):
        rid, f = task
        if f["size"] > max_mb * 1e6:
            return members(rid, f)
        d = os.path.join(atlas_ROOT, "zenodo", rid)
        local = os.path.join(d, os.path.basename(f["key"]))
        if os.path.exists(local) and os.path.getsize(local) != f["size"]:
            # a file cut short by an interrupted download is fetched again, its unpacked copy with it
            os.remove(local)
            if os.path.isdir(local + "_x"):
                import shutil
                shutil.rmtree(local + "_x")
        if not os.path.exists(local):
            os.makedirs(d, exist_ok=True)
            try:
                data = core.get(f["url"], timeout=900, retries=2)
            except Exception as e:
                return "%s %s: %s" % (rid, f["key"], str(e)[:80])
            open(local + ".part", "wb").write(data)
            os.replace(local + ".part", local)
        low, out = local.lower(), local + "_x"
        if os.path.exists(out) or not low.endswith((".zip", ".tar", ".tgz", ".tar.gz")):
            return None
        # an archive's tables are written to a side directory renamed into place once all are out, so that one cut short is unpacked again
        import shutil
        tmp = out + ".part"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            if low.endswith(".zip"):
                with zipfile.ZipFile(local) as z:
                    for m in z.infolist():
                        dst = None if m.is_dir() or not m.filename.lower().endswith(TABLE_EXT) else safe_member(tmp, m.filename)
                        if dst:
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            open(dst, "wb").write(z.read(m))
            else:
                with tarfile.open(local) as t:
                    for m in t.getmembers():
                        dst = safe_member(tmp, m.name) if m.isfile() and m.name.lower().endswith(TABLE_EXT) else None
                        if dst:
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            open(dst, "wb").write(t.extractfile(m).read())
            os.makedirs(tmp, exist_ok=True)
            os.replace(tmp, out)
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            return "%s %s: unpacked badly (%s)" % (rid, f["key"], str(e)[:60])
        return None
    import time as _time
    left = []
    with ThreadPoolExecutor(threads) as ex:
        for task, msg in zip(tasks, ex.map(one, tasks)):
            if msg and "429" in msg:
                left.append(task)
            elif msg:
                print("  " + msg, flush=True)
    # Zenodo answers too many requests with 429; those files are taken again one at a time, a minute apart after each refusal
    for attempt in range(6):
        if not left:
            break
        print("  %d files refused for too many requests; again one at a time (pass %d)" % (len(left), attempt + 1), flush=True)
        again = []
        for task in left:
            msg = one(task)
            if msg and "429" in msg:
                again.append(task)
                _time.sleep(60)
            elif msg:
                print("  " + msg, flush=True)
            else:
                _time.sleep(2)
        left = again
    for rid, f in left:
        print("  %s %s: still refused" % (rid, f["key"]), flush=True)
    print("%d files of %d records under %s" % (len(tasks), len({t[0] for t in tasks}), os.path.join(atlas_ROOT, "zenodo")))


EXOMAST_API = "https://exo.mast.stsci.edu/api/v0.1/spectra/"


def exomast_fetch(names, threads=6):
    """The spectra MAST's Exo.MAST serves for each named planet, downloaded once into ROOT/exomast/<planet>/ with an index of each file's planet."""
    from concurrent.futures import ThreadPoolExecutor
    root = os.path.join(atlas_ROOT, "exomast")
    os.makedirs(root, exist_ok=True)
    ipath = os.path.join(root, "index.json")
    index = json.load(open(ipath)) if os.path.exists(ipath) else {}

    def one(name):
        q = urllib.parse.quote(name)
        b = core.get(EXOMAST_API + q + "/filelist", timeout=60, retries=1, soft=True)
        try:
            files = json.loads(b).get("filenames") or [] if b else []
        except ValueError:
            files = []
        got = []
        for f in files:
            rel = os.path.join(planet_key(name), os.path.basename(f))
            out = os.path.join(root, rel)
            if not os.path.exists(out):
                body = core.get(EXOMAST_API + q + "/file/" + urllib.parse.quote(f), timeout=120, retries=3, soft=True)
                if body is None:
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                open(out, "wb").write(body)
            got.append((rel.replace(os.sep, "/"), name))
        return got
    with ThreadPoolExecutor(threads) as ex:
        for got in ex.map(one, names):
            index.update(dict(got))
    json.dump(index, open(ipath, "w"), indent=1)
    print("%d Exo.MAST spectra of %d planets under %s" % (len(index), len(set(index.values())), root))


def exomast_spectra():
    """Every spectrum downloaded from Exo.MAST read as one, its planet the one it was asked for."""
    root = os.path.join(atlas_ROOT, "exomast")
    ipath = os.path.join(root, "index.json")
    if not os.path.exists(ipath):
        return []
    known = known_planets()
    out = []
    for rel, name in sorted(json.load(open(ipath)).items()):
        path = os.path.join(root, rel)
        x = _parse_one((path, "exo.mast " + rel))
        if not usable_spectrum(x):
            continue
        head = open(path, encoding="utf-8", errors="replace").read(1500)
        out.append(dict(planet=canonical_planet(name, known), kind=x["kind"], instrument=instrument_of(rel + " " + head), source="exomast:%s" % rel,
                        doi=None, file=rel, n=len(x["w"]), wmin=float(x["w"][0]), wmax=float(x["w"][-1]), columns=x["columns"],
                        w=x["w"], hw=x["hw"], d=x["d"], e=x["e"]))
    return out


def dedupe_spectra(sp):
    """Spectra kept once where two sources hold the same numbers for the same planet."""
    seen, out = set(), []
    for x in sp:
        key = (x["planet"], x["n"], round(float(np.sum(x["w"])), 3), round(float(np.sum(x["d"])) / max(x["n"], 1), 0))
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


# descriptions of VizieR tables that may hold a planet's spectrum, and of those that do not
VIZIER_SPECTRA = ("ransmission spectr", "mission spectr", "ransit spectr", "clipse spectr", "ransit depth", "clipse depth", "adius ratio", "Rp/R",
                  "planet-to-star", "pectroscopic channel", "avelength-dependent", "avelength bin", "pectral bin", "ayside spectr", "ransit radi",
                  "pectroscopic transit", "pectroscopic light curve fit")
VIZIER_NOT = re.compile(r"light curves?(?! fit)|limb.?dark|LD coeff|radial velocit|filter|\bmodel|torque|standard star|quasar|galax|candidates|metric|bias|ephemer|"
                        r"mid-?transit time|Venus|Titan|H\{alpha|Ca ?II|Na ?I\b|Fe ?II|Mg ?I\b|He ?I\b", re.I)
WAVE_UNITS = {"nm": 1e-3, "Angstrom": 1e-4, "0.1nm": 1e-4, "um": 1.0, "mum": 1.0, "micron": 1.0, "cm": 1e4, "m": 1e6}


def vizier_path(table):
    """The file under ROOT/vizier/ that keeps a VizieR table's rows."""
    return os.path.join(atlas_ROOT, "vizier", re.sub(r"[^0-9A-Za-z_.-]", "_", table) + ".csv")


VIZIER_ASU = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


def vizier_meta(table):
    """A VizieR table's catalogue title and its columns' descriptions and UCDs, from the ASU service's metadata."""
    text = core.get(VIZIER_ASU + "?" + urllib.parse.urlencode({"-meta": "", "-source": table, "-out.max": 1}), timeout=120).decode("utf-8", "replace")
    title, cols, lines = "", {}, text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("#Title:") and not title and i + 1 < len(lines):
            title = lines[i + 1].strip("# \t")
        elif line.startswith("#Column"):
            f = line.split("\t")
            ucd = re.search(r"\[ucd=([^\]]*)\]", line)
            if len(f) > 3:
                cols[f[1].strip()] = dict(desc=re.sub(r"^\[[^\]]*\]\s*", "", f[3].strip()), ucd=ucd.group(1) if ucd else "")
    return title, cols


def vizier_fetch(threads=4):
    """Every VizieR table whose description speaks of a planet's spectrum, its transit or eclipse depths or its radius ratios by wavelength: its rows kept as CSV under ROOT/vizier/, beside an index of its catalogue's title, its columns' units, descriptions and UCDs, and its catalogue's other tables."""
    import csv
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(os.path.join(atlas_ROOT, "vizier"), exist_ok=True)
    rows = core.tap("vizier", "SELECT table_name, description, nrows FROM TAP_SCHEMA.tables WHERE (%s) AND nrows >= 5 AND nrows <= 20000" % " OR ".join(
        "description LIKE '%%%s%%'" % w for w in VIZIER_SPECTRA))
    cands = {r["table_name"].strip("'"): r["description"] for r in rows if not VIZIER_NOT.search(r["description"])}
    names = sorted(cands)
    cats = sorted({t.rsplit("/", 1)[0] for t in names})
    sib, units = {}, {}
    # VizieR keeps its table names inside single quotes, which its ADQL reaches with LIKE
    for i in range(0, len(cats), 30):
        for r in core.tap("vizier", "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE " + " OR ".join("table_name LIKE '%%%s/%%'" % c for c in cats[i:i + 30])):
            sib.setdefault(r["table_name"].strip("'").rsplit("/", 1)[0], []).append(r["description"])
    for i in range(0, len(names), 30):
        for r in core.tap("vizier", "SELECT table_name, column_name, unit FROM TAP_SCHEMA.columns WHERE " + " OR ".join("table_name LIKE '%%%s%%'" % t for t in names[i:i + 30])):
            units.setdefault(r["table_name"].strip("'"), {})[r["column_name"].strip("'\"")] = r["unit"]

    def one(t):
        path = vizier_path(t)
        meta = path[:-4] + ".meta.json"
        try:
            if not os.path.exists(path):
                data = core.tap("vizier", 'SELECT * FROM "%s"' % t)
                with open(path + ".part", "w", newline="", encoding="utf-8") as f:
                    if data:
                        w = csv.DictWriter(f, fieldnames=list(data[0]))
                        w.writeheader()
                        w.writerows(data)
                os.replace(path + ".part", path)
            if not os.path.exists(meta):
                title, cols = vizier_meta(t)
                json.dump(dict(title=title, columns=cols), open(meta, "w"), indent=1)
        except Exception as e:
            return "%s: %s" % (t, str(e)[:80])
        return None
    with ThreadPoolExecutor(threads) as ex:
        for msg in ex.map(one, names):
            if msg:
                print("  " + msg, flush=True)
    index = {}
    for t in names:
        mp = vizier_path(t)[:-4] + ".meta.json"
        m = json.load(open(mp)) if os.path.exists(mp) else dict(title="", columns={})
        index[t] = dict(description=cands[t], title=m["title"], catalogue=sib.get(t.rsplit("/", 1)[0], []), units=units.get(t, {}), columns=m["columns"])
    json.dump(index, open(os.path.join(atlas_ROOT, "vizier", "index.json"), "w"), indent=1)
    print("%d VizieR tables of %d catalogues under %s" % (len(names), len(cats), os.path.join(atlas_ROOT, "vizier")))


def _vizier_um(v, unit):
    """Wavelengths or widths in a VizieR column's unit put in um; without a unit, by their size."""
    if unit in WAVE_UNITS:
        return v * WAVE_UNITS[unit]
    med = np.nanmedian(np.abs(v[np.isfinite(v)])) if np.isfinite(v).any() else 1.0
    return v if med < 40 else v * 1e-3


def vizier_spectra():
    """Every kept VizieR table read as one spectrum or several: the wavelength found by its UCD and put in um by its unit, bin edges made a centre and half width, bounds made errors, rows a flag excludes dropped, a table of several planets, instruments or visits split by its text column; each planet named by its group, the table, the catalogue's title or its other tables."""
    import csv
    ipath = os.path.join(atlas_ROOT, "vizier", "index.json")
    if not os.path.exists(ipath):
        return []
    known = known_planets()
    out = []
    for t, meta in sorted(json.load(open(ipath)).items()):
        path = vizier_path(t)
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        cols, units = meta.get("columns", {}), meta.get("units", {})

        def ucd(c):
            return cols.get(c, {}).get("ucd", "")

        def desc(c):
            return cols.get(c, {}).get("desc", "").lower()
        # rows a flag marks as left out of the paper's fits are left out here
        for c in list(rows[0]) if rows else []:
            if ucd(c).startswith("meta.code") and "exclud" in desc(c):
                rows = [r for r in rows if not (r[c] or "").strip()]
        if len(rows) < 5:
            continue
        num, text = {}, {}
        for c in rows[0]:
            vals = [(r[c] or "").strip() for r in rows]
            try:
                v = np.array([float(x) if x else np.nan for x in vals])
            except ValueError:
                if not ucd(c).startswith("meta.code"):
                    text[c] = vals
                continue
            if c == "recno" or ucd(c).startswith(("phot.limbDark", "time", "meta.number", "meta.id")):
                continue
            num[c] = v
        # the wavelength columns, told apart by unit, UCD, description and name: a centre, the bin's edges, its width
        lengths = [c for c in num if (units.get(c) or "").strip() in WAVE_UNITS or "em.wl" in ucd(c) or "spect.binSize" in ucd(c)]
        # a column's name settles which edge it is before its description does, since descriptions are sometimes swapped
        edge_lo = [c for c in lengths if re.search(r"(?i)min|^lo|start", c)] or [c for c in lengths if "stat.min" in ucd(c) or re.search(r"lower|minimum|start", desc(c))]
        edge_hi = [c for c in lengths if re.search(r"(?i)max|^hi|end", c)] or [c for c in lengths if "stat.max" in ucd(c) or re.search(r"upper|maximum|end", desc(c))]
        widths = [c for c in lengths if c not in edge_lo + edge_hi and re.search(r"width|bandwidth", desc(c) + " " + ucd(c).replace("instr.bandwidth", "width")) and not re.search(r"cent", desc(c))]
        centres = [c for c in lengths if c not in edge_lo + edge_hi + widths]
        half = None
        if widths:
            width = _vizier_um(num[widths[0]], units.get(widths[0], ""))
            half = width if "half" in desc(widths[0]) else width / 2
        if centres:
            w = _vizier_um(num[centres[0]], units.get(centres[0], ""))
            if half is None and edge_lo and edge_hi:
                half = np.abs(_vizier_um(num[edge_hi[0]], units.get(edge_hi[0], "")) - _vizier_um(num[edge_lo[0]], units.get(edge_lo[0], ""))) / 2
        elif edge_lo and edge_hi:
            a, b = _vizier_um(num[edge_lo[0]], units.get(edge_lo[0], "")), _vizier_um(num[edge_hi[0]], units.get(edge_hi[0], ""))
            w, half = (a + b) / 2, np.abs(b - a) / 2
        else:
            continue
        for c in lengths:
            num.pop(c)
        # a measurement given with its lower and upper bounds takes half their spread as its error
        for c in list(num):
            for a, b in ((c + "min", c + "max"), (c + "low", c + "up")):
                if a in num and b in num:
                    num["e_" + c] = np.abs(num.pop(b) - num.pop(a)) / 2
        eclipse = re.search(r"eclipse|emission|dayside|occultation", (meta["description"] + " " + meta.get("title", "")).lower())
        named = {}
        for c, v in num.items():
            n = re.sub(r"(?i)^([e]_)?(rp/r\*|rp/rs|r_?p/r_?\*|rp/r_?s)$", lambda m: (m.group(1) or "") + "RpRs", c)
            if eclipse and re.match(r"(?i)^(e_)?flux$", n):
                n = re.sub(r"(?i)flux", "FpFs", n)
            named[n] = v
        named["wave"] = w
        if half is not None:
            named["halfwidth"] = half
        key = None
        for c, vals in text.items():
            k = len(set(vals))
            if 2 <= k <= 30 and k < len(vals) / 2 and (key is None or planet_names(" ".join(set(vals)))):
                key = c
        groups = sorted(set(text[key])) if key else [""]
        about = " ".join([meta["description"], meta.get("title", "")])
        cat = planet_names(" ".join(meta["catalogue"]))
        own = planet_names(meta["description"]) or planet_names(meta.get("title", "")) or (cat if len(cat) == 1 else [])
        for g in groups:
            m = np.array([v == g for v in text[key]]) if key else np.ones(len(rows), bool)
            pl = planet_names(g)
            if not pl and re.match(r"^[b-i]$", g):
                # a column of planet letters beside a system the table or catalogue names
                pl = [k for k in known.values() if k.endswith(" " + g) and re.search(r"(?<![0-9A-Za-z])" + re.escape(k[:-2]) + r"(?![0-9])", about, re.I)][:1]
            pl = list(dict.fromkeys(canonical_planet(x, known) for x in (pl or own)))
            if len(pl) != 1 or pl[0] not in known.values() or m.sum() < 5:
                continue
            hint = " ".join([about, g] + [desc(c) for c in num])
            order = list(named)
            x = _columns_spectrum(_Columns(order, np.array([named[c][m] for c in order]).T, note=meta["description"]), hint)
            if x is None:
                continue
            x = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in x.items()}
            if not usable_spectrum(x):
                continue
            # a group that names its own kind, as a table of a planet's transit and eclipse spectra does, keeps it
            if re.search(r"emission|eclipse|dayside|occultation", g, re.I):
                x["kind"] = "emission"
            elif re.search(r"transmission|transit", g, re.I):
                x["kind"] = "transmission"
            out.append(dict(planet=pl[0], kind=x["kind"], instrument=instrument_of(hint), source="vizier:%s" % t, doi=None, file=t + (":" + g if g else ""),
                            n=len(x["w"]), wmin=float(x["w"][0]), wmax=float(x["w"][-1]), columns=x["columns"], w=x["w"], hw=x["hw"], d=x["d"], e=x["e"]))
    return out


EU_TAP = "http://voparis-tap-planeto.obspm.fr/tap/sync"


def eu_planets(names):
    """Planet and host parameters by planet name from the Encyclopaedia of Exoplanetary Systems' TAP service (radii in Jupiter's, axes in au, distances in pc)."""
    import csv
    import io
    q = ("SELECT target_name, radius, radius_error_min, radius_error_max, mass, semi_major_axis, temp_calculated, star_teff, star_radius, star_distance "
         "FROM exoplanet.epn_core WHERE target_name IN (%s)" % ",".join("'%s'" % n.replace("'", "''") for n in names))
    url = EU_TAP + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
    return {r["target_name"]: r for r in csv.DictReader(io.StringIO(core.get(url, timeout=120).decode("utf-8", "replace")))}


TEPCAT_URL = "https://www.astro.keele.ac.uk/jkt/tepcat/allplanets-ascii.txt"


def tepcat_name(system):
    """A TEPCat system name as the catalogues write the planet, zero padding dropped: LHS_1140b to LHS 1140 b, WASP-039 to WASP-39 b, LTT_1445Ab to LTT 1445 A b."""
    s = re.sub(r"(?<![0-9.])0+(?=[0-9])", "", system.replace("_", " ").strip())
    m = re.match(r"^(.*?[0-9])\s?([AB])?\s?([b-i])$", s) or re.match(r"^(.*\s)([AB])?\s?([b-i])$", s)
    if m:
        return "%s%s %s" % (m.group(1).rstrip(), " " + m.group(2) if m.group(2) else "", m.group(3))
    return s + " b"


def planet_key(name):
    """A planet name's letters and digits alone, the coordinate suffix of a designation like WD 1856+534 dropped, for matching spellings."""
    return re.sub(r"[^0-9a-z]", "", re.sub(r"(?<=[0-9]{4})[+-][0-9]{2,}", "", name.lower()))


def tepcat():
    """TEPCat's homogeneous parameters of the transiting planets (Southworth 2011), keyed by planet name: Teff (K), R* (Rsun), a (au), Rp (Rjup) and Teq (K) with their errors."""
    import time as _time
    local = core.data("tepcat", "allplanets-ascii.txt")
    if not os.path.exists(local) or _time.time() - os.path.getmtime(local) > 7 * 86400:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        body = core.get(TEPCAT_URL, timeout=120)
        open(local, "wb").write(body)
    out = {}
    for line in open(local, encoding="utf-8", errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 41:
            continue

        def v(i):
            x = core.num(f[i])
            return None if x is None or x == -1 else x
        out[tepcat_name(f[0])] = dict(system=f[0], teff=v(1), teff_err=v(2), rs=v(10), rs_err=v(11), a_au=v(23), a_err=v(24), rp_rjup=v(29),
                                      rp_err=v(30), mp_mjup=v(26), teq=v(38), porb=v(19))
    return out


ROCKY_PAGE = "https://archive.stsci.edu/hlsp/rocky-worlds"
rocky_ROOT = os.environ.get("ROCKY_ROOT", core.data("rocky_worlds"))
rocky_LAMBDA_UM = 15.0  # MIRI F1500W


def rockyworlds_fetch():
    """Every JWST eclipse catalogue the Rocky Worlds DDT page lists, into ROOT."""
    os.makedirs(rocky_ROOT, exist_ok=True)
    page = core.get(ROCKY_PAGE, timeout=120).decode("utf-8", "replace")
    links = sorted(set(re.findall(r'href="(https://archive\.stsci\.edu/hlsps/rocky-worlds/JWST/[^"]+_eclipse-cat\.fits)"', page)))
    for u in links:
        local = os.path.join(rocky_ROOT, os.path.basename(u))
        if not os.path.exists(local):
            open(local, "wb").write(core.get(u, timeout=300))
    print("%d eclipse catalogues in %s" % (len(links), rocky_ROOT))


def planck_inverse(b, lam_um):
    """Brightness temperature (K) of a specific intensity b (W m^-2 Hz^-1 sr^-1) at a wavelength in um."""
    nu = 2.998e8 / (lam_um * 1e-6)
    with np.errstate(all="ignore"):
        return 6.626e-34 * nu / (1.381e-23 * np.log1p(2 * 6.626e-34 * nu ** 3 / (2.998e8 ** 2 * b)))


rockyworlds_DOC = r"""Atmospheres of the Rocky Worlds planets: every JWST MIRI 15 um eclipse the Rocky Worlds DDT has released, each planet's depth (the team's joint fit where one exists, else the weighted mean of its eclipses) turned into a dayside brightness temperature through the star's own measured 15 um flux, its radius and distance and the planet's radius ratio, against the hottest a bare dark rock without redistribution can be, T_eff (R*/a)^1/2 (2/3)^1/4; a ratio below 0.8 at 3 sigma is heat carried away by an atmosphere or reflected by a bright surface, one above 1 at 3 sigma more heat than starlight supplies; the errors drawn from the depth's and the radius's.

    python -I seti.py biosignatures rockyworlds fetch
    python -I seti.py biosignatures rockyworlds [record] [--ratio-atm 0.8] [--sigma 3] [--draws 20000]"""


def rockyworlds_main(record=False, ratio_atm=0.8, sigma=3.0, draws=20000):
    import glob
    from astropy.io import fits
    by = {}
    for path in sorted(glob.glob(os.path.join(rocky_ROOT, "*_eclipse-cat.fits"))):
        with fits.open(path) as h:
            name = h[0].header.get("HLSPTARG")
            m = h["MEASUREMENTS"].data
            by.setdefault(name, []).append(dict(file=os.path.basename(path), joint="checkpoint" in path, depth=float(m["ECLIPSEDEPTH"][0]),
                                                err=float(m["ECLIPSEDEPTHERROR"][0]), absflux=float(np.mean(m["ABSFLUX"])), n=len(m)))
    tep = tepcat()
    dist = {k: core.num(v["star_distance"]) for k, v in eu_planets(sorted(by)).items()}
    rng = np.random.default_rng(1)
    out = []
    for name, cats in sorted(by.items()):
        joint = [c for c in cats if c["joint"]]
        if joint:
            c = sorted(joint, key=lambda c: c["file"])[-1]
            depth, err, n_ecl = c["depth"], c["err"], c["n"]
        else:
            w = np.array([1 / c["err"] ** 2 for c in cats])
            depth = float(np.sum(w * np.array([c["depth"] for c in cats])) / np.sum(w))
            err, n_ecl = float(1 / math.sqrt(np.sum(w))), len(cats)
        f_star = float(np.mean([c["absflux"] for c in cats]))
        p = tep.get(name)
        if not p or not dist.get(name) or None in (p["rp_rjup"], p["rs"], p["a_au"], p["teff"]):
            print("  %s: no parameters" % name)
            continue
        rp, rp_err = p["rp_rjup"] * 7.1492e7, (p["rp_err"] or 0) * 7.1492e7
        rs, a_au, teff, d_m = p["rs"] * 6.957e8, p["a_au"], p["teff"], dist[name] * 3.0857e16
        b_star = f_star * 1e-29 / (math.pi * (rs / d_m) ** 2)
        t_max = teff * math.sqrt(rs / (a_au * 1.496e11)) * (2 / 3) ** 0.25
        dd = rng.normal(depth, err, draws) * 1e-6
        kk = rng.normal(rp, rp_err, draws) / rs
        tb = planck_inverse(np.where(dd > 0, dd, np.nan) * b_star / kk ** 2, rocky_LAMBDA_UM)
        ratio = np.nan_to_num(tb / t_max, nan=0.0)
        t_day = float(planck_inverse(depth * 1e-6 * b_star / (rp / rs) ** 2, rocky_LAMBDA_UM)) if depth > 0 else 0.0
        lo, med, hi = np.percentile(ratio, [16, 50, 84])
        p_atm = float(np.mean(ratio >= ratio_atm))
        p_hot = float(np.mean(ratio <= 1.0))
        z_atm = (ratio_atm - med) / max(med - lo, 1e-9) if med < ratio_atm else -(med - ratio_atm) / max(med - lo, 1e-9)
        z_hot = (med - 1.0) / max(med - lo, 1e-9)
        verdict = ("heat carried off or reflected" if z_atm >= sigma else "more heat than starlight supplies" if z_hot >= sigma else "a bare rock's dayside" if lo >= ratio_atm else "undecided")
        out.append(dict(planet=name, depth=depth, err=err, n_eclipses=n_ecl, absflux_mjy=f_star, t_day=t_day, t_max=t_max, ratio=float(med), ratio_lo=float(lo),
                        ratio_hi=float(hi), p_ratio_above_atm=p_atm, p_ratio_below_1=p_hot, z_atm=z_atm, z_hot=z_hot, teq=p["teq"], verdict=verdict))
    print("%d Rocky Worlds planets with JWST 15 um eclipses" % len(out))
    for o in out:
        print("  %-10s %d eclipse%s, depth %.0f +- %.0f ppm, star %.2f mJy: dayside %.0f K against a bare rock's %.0f K, ratio %.2f (%.2f-%.2f); %s" % (
            o["planet"], o["n_eclipses"], "" if o["n_eclipses"] == 1 else "s", o["depth"], o["err"], o["absflux_mjy"], o["t_day"], o["t_max"], o["ratio"], o["ratio_lo"], o["ratio_hi"], o["verdict"]))
    json.dump(out, open(core.results("biosig_rockyworlds.json"), "w"), indent=1)
    if record:
        core.record_run(
            "rocky_worlds_15um", "Atmospheres of the Rocky Worlds planets", "MAST high-level science products of the JWST Rocky Worlds DDT (MIRI F1500W eclipse catalogues); planetary and stellar parameters from TEPCat, distances from the Encyclopaedia of Exoplanetary Systems",
            "every Rocky Worlds planet with a released JWST eclipse: %s" % ", ".join(o["planet"] for o in out), os.path.abspath(__file__), "rockyworlds", len(out), "planets",
            dict(ratio_atm=ratio_atm, sigma=sigma, draws=draws, wavelength_um=rocky_LAMBDA_UM),
            "the bare dark rock's dayside without redistribution, T_eff (R*/a)^1/2 (2/3)^1/4", "the depth's and the planet radius's errors drawn %d times" % draws,
            "; ".join("%s %.2f of a bare rock's dayside, %s" % (o["planet"], o["ratio"], o["verdict"]) for o in out),
            [dict(target="planet:%s" % o["planet"], stat=round(o["ratio"], 3), unit="dayside brightness temperature over a bare dark rock's",
                  note="%d eclipse%s, depth %.0f +- %.0f ppm, dayside %.0f K against %.0f K (16-84 per cent %.2f-%.2f); %s" % (
                      o["n_eclipses"], "" if o["n_eclipses"] == 1 else "s", o["depth"], o["err"], o["t_day"], o["t_max"], o["ratio_lo"], o["ratio_hi"], o["verdict"])) for o in out])
        print("recorded")


# the screen of every planet adds the optical alkali lines, which hot giants' spectra carry
ATOMIC_BANDS = dict(Na=((0.589, 0.02, 1.0),), K=((0.767, 0.02, 1.0),))
HITRAN_API = "https://hitran.org/lbl/api"
# N2O's main isotopologue by HITRAN's global isotopologue id; Exo-Transmit's tables lack it
HITRAN_ISO = dict(N2O=21)
EXO_TRANSMIT = "https://raw.githubusercontent.com/elizakempton/Exo_Transmit/master/Opac/opac%s.dat"
NIST_DMS = "https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C75183&Index=0&Type=IR"
LINE_GASES = ("H2O", "CH4", "CO2", "CO", "NH3", "DMS", "N2O", "SO2", "OCS", "O3")
XS_R, XS_UM = 1000.0, (0.3, 30.0)
# the cross-sections of a temperate atmosphere's middle, 300 K and 0.1 bar
XS_T, XS_PA = 300.0, 1e4
# a gas stops shaping the spectrum where its cross-section falls below this fraction of its strongest, about where Earth's methane meets the rest of the opacity
XS_FLOOR = 1e-4


def xs_grid():
    """The edges of the logarithmic wavelength grid (um) on which cross-sections are kept, at R 1,000 from 0.3 to 30 um."""
    return np.exp(np.arange(np.log(XS_UM[0]), np.log(XS_UM[1]), 1 / XS_R))


def xs_centres():
    """The centres of the cross-section grid's bins (um)."""
    e = xs_grid()
    return np.sqrt(e[1:] * e[:-1])


def exo_transmit_xs(gas):
    """A gas's cross-sections on the grid at every temperature of Exo-Transmit's opacity table (Kempton et al. 2017), at 0.1 bar, in the table's m2: the temperatures (K) and one row per temperature."""
    path = core.data("opacities", "%s_R%d_T.npz" % (gas, XS_R))
    if os.path.exists(path):
        z = np.load(path)
        return z["temps"], z["xs"]
    local = core.data("opacities", "opac%s.dat" % gas)
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        body = core.get(EXO_TRANSMIT % gas, timeout=600)
        open(local, "wb").write(body)
    lines = [line for line in open(local) if line.strip()]
    temps, pres = np.array(lines[0].split(), dtype=float), np.array(lines[1].split(), dtype=float)
    ip = int(np.argmin(np.abs(np.log10(pres) - np.log10(XS_PA))))
    # each wavelength (m) is followed by one row per pressure: its log10 and the cross-section at each temperature
    lam = np.array([float(lines[i]) for i in range(2, len(lines), 1 + len(pres))]) * 1e6
    xs = np.array([lines[i + 1 + ip].split()[1:] for i in range(2, len(lines), 1 + len(pres))], dtype=float)
    c = np.log(xs_centres())
    out = np.exp(np.array([np.interp(c, np.log(lam), np.log(np.maximum(xs[:, j], 1e-60))) for j in range(len(temps))]))
    np.savez(path, temps=temps, xs=out)
    return temps, out


def hitran_xs(gas, threads=8, step=250.0):
    """A gas's mean cross-section (cm2) in each bin of the grid from HITRAN's lines of its main isotopologue at 296 K, each wavenumber range read page by page until a page comes back empty, since the service cuts long answers short."""
    import http.client
    from concurrent.futures import ThreadPoolExecutor
    path = core.data("opacities", "%s_R%d.npy" % (gas, XS_R))
    if os.path.exists(path):
        return np.load(path)
    edges = xs_grid()

    def page(lo, hi):
        url = HITRAN_API + "?" + urllib.parse.urlencode(dict(iso_ids_list=HITRAN_ISO[gas], numin="%.6f" % lo, numax="%.6f" % hi, request_params="nu,sw"))
        for attempt in range(4):
            try:
                body = urllib.request.urlopen(urllib.request.Request(url, headers=core.UA), timeout=300).read()
            except http.client.IncompleteRead as e:
                body = e.partial
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return np.zeros((0, 2))
                time.sleep(5 * (attempt + 1))
                continue
            except OSError:
                time.sleep(5 * (attempt + 1))
                continue
            rows = [r.split(",") for r in body.decode("ascii", "replace").split() if r.count(",") == 1]
            # the last row of an answer cut short may be cut too
            return np.array(rows[:-1] if len(rows) > 1 else rows, dtype=float).reshape(-1, 2)
        raise IOError("HITRAN did not answer for %s %g-%g" % (gas, lo, hi))

    def one(lo):
        hi, parts = lo + step, []
        while True:
            a = page(lo, hi)
            if not len(a):
                return np.concatenate(parts) if parts else a
            parts.append(a)
            lo = a[-1, 0] + 1e-6
    with ThreadPoolExecutor(threads) as ex:
        a = np.concatenate(list(ex.map(one, np.arange(1e4 / edges[-1], 1e4 / edges[0], step))))
    a = np.unique(a, axis=0)
    k = np.searchsorted(edges, 1e4 / a[:, 0]) - 1
    ok = (k >= 0) & (k < len(edges) - 1)
    xs = np.bincount(k[ok], weights=a[ok, 1], minlength=len(edges) - 1) / (1e4 / edges[:-1] - 1e4 / edges[1:])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, xs)
    return xs


def nist_dms_xs():
    """Dimethyl sulfide's absorbance on the grid, from the NIST WebBook's infrared spectrum (Coblentz 5969, liquid, 2.5-16 um), its baseline removed; zero outside it."""
    local = core.data("opacities", "DMS_nist.jdx")
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        body = core.get(NIST_DMS, timeout=120)
        open(local, "wb").write(body)
    nu, tr, step = [], [], 1.0
    for line in open(local, encoding="utf-8", errors="replace"):
        if line.startswith("##DELTAX="):
            step = float(line.split("=")[1])
        elif line[:1].isdigit():
            v = [float(x) for x in line.split()]
            nu += [v[0] + step * i for i in range(len(v) - 1)]
            tr += v[1:]
    nu, ab = np.array(nu), -np.log10(np.clip(tr, 1e-4, 1))
    ab = np.clip(ab - np.percentile(ab, 5), 0, None)
    c = xs_centres()
    o = np.argsort(1e4 / nu)
    lam = (1e4 / nu)[o]
    return np.where((c >= lam[0]) & (c <= lam[-1]), np.interp(c, lam, ab[o]), 0.0)


_TEMPLATES = {}


def gas_template(gas, w, hw, temp=XS_T):
    """A gas's transmission template averaged over each bin, from its cross-sections at the tabulated temperature nearest the atmosphere's: the logarithm of the cross-section above a floor of its strongest, which is how an atmosphere's effective height follows opacity (Lecavelier des Etangs et al. 2008), scaled to one at its peak."""
    if gas in HITRAN_ISO or gas == "DMS":
        key = (gas, None)
    else:
        temps, table = exo_transmit_xs(gas)
        j = int(np.argmin(np.abs(temps - temp)))
        key = (gas, j)
    if key not in _TEMPLATES:
        xs = nist_dms_xs() if gas == "DMS" else hitran_xs(gas) if gas in HITRAN_ISO else table[key[1]]
        floor = XS_FLOOR * xs.max()
        t = np.log(np.maximum(xs, floor) / floor) / np.log(1 / XS_FLOOR)
        _TEMPLATES[key] = np.concatenate([[0.0], np.cumsum(t)])
    c = _TEMPLATES[key]
    edges = xs_grid()
    hw = np.where(hw > 0, hw, np.abs(np.gradient(w)) / 2 if len(w) > 1 else 0.01)
    # a bin narrower than the grid's takes the grid bin it falls in
    lo = np.clip(np.searchsorted(edges, w - hw) - 1, 0, len(c) - 2)
    hi = np.clip(np.maximum(np.searchsorted(edges, w + hw) - 1, lo + 1), 1, len(c) - 1)
    inside = (w > edges[0]) & (w < edges[-1])
    return np.where(inside, (c[hi] - c[lo]) / (hi - lo), 0.0)


TABLE_EXT = (".txt", ".dat", ".csv", ".ecsv", ".tsv", ".tbl", ".fits", ".fit", ".npy")


def bin_spectrum(w, hw, d, e, resolution=100.0):
    """A spectrum averaged with inverse-variance weights into bins of constant resolution, each with its half width; a spectrum already that coarse is returned as it is."""
    ok = np.isfinite(w) & (w > 0) & np.isfinite(d) & np.isfinite(e) & (e > 0)
    w, hw, d, e = w[ok], np.where(np.isfinite(hw[ok]), hw[ok], 0.0), d[ok], e[ok]
    if len(w) < 2 or np.median(w / np.maximum(2 * hw, 1e-9)) <= resolution * 1.5:
        return w, hw, d, e
    edges = np.exp(np.arange(np.log(w.min()), np.log(w.max()) + 1 / resolution, 1 / resolution))
    k = np.digitize(w, edges)
    ow, ohw, od, oe = [], [], [], []
    for j in np.unique(k):
        m = k == j
        wt = 1 / e[m] ** 2
        ow.append(float(np.sum(wt * w[m]) / np.sum(wt)))
        od.append(float(np.sum(wt * d[m]) / np.sum(wt)))
        oe.append(float(1 / math.sqrt(np.sum(wt))))
        ohw.append(float((w[m].max() + hw[m][np.argmax(w[m])] - w[m].min() + hw[m][np.argmin(w[m])]) / 2))
    return np.array(ow), np.array(ohw), np.array(od), np.array(oe)


def screen_spectrum(w, hw, d, e, gases=LINE_GASES + tuple(ATOMIC_BANDS), temp=XS_T):
    """One spectrum, binned to R 100, fitted with an offset, a scattering slope where it reaches below 1 um, and every gas whose bands it covers at the atmosphere's temperature; each gas's signed significance from how much the fit worsens without it, its errors scaled up to the fit's own scatter."""
    w, hw, d, e = bin_spectrum(w, hw, d, e)
    cols, names = [np.ones(len(w))], ["offset"]
    if np.min(w) < 1.0:
        cols.append(np.log(w))
        names.append("slope")
    base = len(cols)
    for gas in gases:
        t = gas_template(gas, w, hw, temp) if gas in LINE_GASES else band_template(w, hw, ATOMIC_BANDS[gas])
        if np.ptp(t) > 0.05:
            cols.append(t)
            names.append(gas)
    A = np.array(cols).T
    W = 1.0 / e ** 2

    def fit(M):
        cov = np.linalg.pinv(M.T @ (M * W[:, None]))
        par = cov @ (M.T @ (W * d))
        return par, cov, float(np.sum(W * (d - M @ par) ** 2))
    par, cov, chi = fit(A)
    dof = len(w) - len(names)
    scale = max(1.0, chi / dof) if dof > 0 else 1.0
    sig = {}
    for j in range(base, len(names)):
        dchi = (fit(np.delete(A, j, axis=1))[2] - chi) / scale
        sig[names[j]] = math.copysign(math.sqrt(max(dchi, 0.0)), par[j])
    return dict(sig=sig, chi2=chi, chi2_flat=fit(A[:, :base])[2], dof=dof, n_binned=len(w), error_scale=math.sqrt(scale))


# files that hold the sky's, a star's or a model's spectrum rather than a planet's
NOT_PLANET = re.compile(r"sky|telluric|stellar|star_spec|host|phoenix|bt.?settl|model|retriev|posterior|sample|chain|forward|synthetic|simulat|opacity|xsec|cross.?sec|template|grid|mass.?metal|mh_curve", re.I)


def canonical_planet(name, known):
    """A planet name matched to a catalogue's spelling by its letters and digits alone (LP 79118 c to LP 791-18 c), its numbers' zero padding dropped (WASP-039 b), or its component letter dropped (WASP-77 A b to WASP-77 b); otherwise itself."""
    unpadded = re.sub(r"(?<![0-9.])0+(?=[0-9])", "", name)
    for n in (name, unpadded, re.sub(r"\s[AB]\s(?=[b-i]$)", " ", unpadded)):
        for k in (re.sub(r"[^0-9a-z]", "", n.lower()), planet_key(n)):
            if k in known:
                return known[k]
    return name


def known_planets():
    """TEPCat's planets keyed by their names' letters and digits, with and without a designation's coordinate suffix."""
    known = {}
    for k in tepcat():
        known[re.sub(r"[^0-9a-z]", "", k.lower())] = k
        known.setdefault(planet_key(k), k)
    return known


SPEC_NAME = re.compile(r"spec|transmission|emission|depth|eclipse|transit|fpfs|tspec|dayside|planet|trans_|emiss|rprs|fp_fs", re.I)


def _parse_one(task):
    """One file read as a spectrum, its arrays as lists, or None."""
    path, hint = task
    try:
        x = table_spectrum(path, hint)
    except Exception:
        return None
    if x is None:
        return None
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in x.items()}


def usable_spectrum(x):
    """True for a reading with five to 20,000 points over at least 0.2 um, depths between 1 and 200,000 ppm and errors below ten times them."""
    # a spectrum narrower than 0.2 um is a single line's, read by other means than band templates
    if not x or len(x["w"]) < 5 or len(x["w"]) > 20000 or x["w"][-1] - x["w"][0] < 0.2:
        return False
    md, me = np.median(np.abs(x["d"])), np.median(x["e"])
    return bool(1 <= md <= 2e5 and 0 < me <= 10 * max(md, 1))


def atlas_parse(records, procs=8):
    """Every downloaded Zenodo file that may be a planet's spectrum read as one where its columns allow, on several processes with a cache of each file's reading; a record of many files is read only where its names speak of spectra; identical spectra kept once."""
    from concurrent.futures import ProcessPoolExecutor
    out, seen, dropped = [], set(), set()
    known = known_planets()
    root = os.path.join(atlas_ROOT, "zenodo")
    cache_path = os.path.join(atlas_ROOT, "parse_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    jobs = []
    for rid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        r = records.get(rid, {})
        rec_planets = planet_names(" ".join([r.get("title", ""), r.get("desc", "")]))
        files = []
        for d, _, fs in os.walk(os.path.join(root, rid)):
            for f in fs:
                path = os.path.join(d, f)
                rel = os.path.relpath(path, os.path.join(root, rid))
                if f.lower().endswith(TABLE_EXT) and os.path.getsize(path) <= 50e6 and not NOT_PLANET.search(rel):
                    files.append((path, rel))
        if len(files) > 20:
            files = [(p, rel) for p, rel in files if SPEC_NAME.search(rel)]
        for path, rel in files:
            pl = planet_names(rel.replace(os.sep, " ")) or (rec_planets if len(rec_planets) == 1 else [])
            pl = [canonical_planet(x, known) for x in pl]
            if len(pl) != 1:
                continue
            if pl[0] not in known.values():
                # transit and eclipse spectra belong to transiting planets, which TEPCat lists; a name it lacks is a companion imaged directly or a misreading
                dropped.add(pl[0])
                continue
            key = "%s|%s|%d|%d" % (rid, rel, os.path.getsize(path), int(os.path.getmtime(path)))
            jobs.append((key, path, rel, rid, r, pl[0]))
    todo = [j for j in jobs if j[0] not in cache]
    if todo:
        with ProcessPoolExecutor(procs) as ex:
            for j, x in zip(todo, ex.map(_parse_one, [(j[1], j[4].get("title", "") + " " + j[2]) for j in todo], chunksize=8)):
                cache[j[0]] = x
        json.dump(cache, open(cache_path, "w"))
    for key, path, rel, rid, r, planet in jobs:
        x = cache.get(key)
        if not usable_spectrum(x):
            continue
        dedupe = (planet, len(x["w"]), round(float(np.sum(x["w"])), 4), round(float(np.sum(x["d"])), 1))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        out.append(dict(planet=planet, kind=x["kind"], instrument=instrument_of(rel + " " + r.get("title", "")), source="zenodo:%s" % rid,
                        doi=r.get("doi"), file=rel, n=len(x["w"]), wmin=float(x["w"][0]), wmax=float(x["w"][-1]), columns=x["columns"],
                        w=x["w"], hw=x["hw"], d=x["d"], e=x["e"]))
    print("  %d files considered, %d read now and %d from the cache" % (len(jobs), len(todo), len(jobs) - len(todo)))
    if dropped:
        print("  names TEPCat does not list as transiting planets, set aside: %s" % ", ".join(sorted(dropped)))
    return out


atlas_DOC = r"""Gases of every planet with a published spectrum: every transmission and emission spectrum the NASA Exoplanet Archive holds, every one MAST's Exo.MAST serves, every one a VizieR table holds and every one published on Zenodo that a table reader can take (wavelength, depth and error columns by name and value, radius ratios squared, units to um and ppm), each binned to R 100 and fitted alone with an offset, a scattering slope below 1 um and templates of H2O, CH4, CO2, CO, NH3, SO2, OCS and O3 from Exo-Transmit's cross-sections at the tabulated temperature nearest the planet's, N2O from HITRAN's lines and dimethyl sulfide from NIST's infrared spectrum, each the logarithm of its cross-section above a floor of 1e-4 of its peak, with Na and K as bands; each gas counted by how much the fit worsens without it, its errors scaled to the fit's own scatter, and by its median over a planet's spectra; in transmission, CH4 and CO2 at 3 sigma with CO below 2 sigma, dimethyl sulfide at 3 sigma or N2O at 3 sigma flag a planet, which a hot, giant or hydrogen-enveloped planet's own chemistry, methane beside a DMS reading and a cool host's spots then explain where they apply; Earth's transit spectrum from ACE-FTS (Macdonald & Cowan 2019) is the control, read alone and laid on each temperate small planet's spectra with their errors; TEPCat gives each planet's size, temperature and host.

    python -I seti.py biosignatures atlas search|fetch|exomast|vizier|parse
    python -I seti.py biosignatures atlas [record] [--sigma 3] [--hot-k 450] [--giant-rjup 0.32] [--envelope-rjup 0.143] [--cool-host-k 3500]"""


def atlas_cli():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    scan = os.path.join(atlas_ROOT, "zenodo_records.json")
    if mode == "search":
        os.makedirs(atlas_ROOT, exist_ok=True)
        recs = json.load(open(scan)) if os.path.exists(scan) else {}
        qpath = os.path.join(atlas_ROOT, "zenodo_queries.json")
        done = set(json.load(open(qpath))) if os.path.exists(qpath) and recs else set()
        todo = [q for q in ZENODO_QUERIES if q not in done]
        recs.update(zenodo_search(todo))
        json.dump(recs, open(scan, "w"), indent=1)
        json.dump(sorted(done | set(todo)), open(qpath, "w"), indent=1)
        print("%d Zenodo records, %d naming a planet" % (len(recs), sum(1 for r in recs.values() if planet_names(r["title"] + " " + r["desc"]))))
        return
    if mode == "fetch":
        zenodo_fetch(json.load(open(scan)))
        return
    if mode == "exomast":
        exomast_fetch(sorted(tepcat()))
        return
    if mode == "vizier":
        vizier_fetch()
        return
    if mode == "parse":
        sp = atlas_parse(json.load(open(scan)))
        ex = exomast_spectra()
        vz = vizier_spectra()
        ar = archive_atlas_spectra()
        print("  %d Zenodo, %d Exo.MAST, %d VizieR and %d archive spectra" % (len(sp), len(ex), len(vz), len(ar)))
        sp = dedupe_spectra(sp + ex + vz + ar)
        json.dump(sp, open(os.path.join(atlas_ROOT, "atlas.json"), "w"))
        pl = {}
        for x in sp:
            pl.setdefault(x["planet"], set()).add(x["kind"])
        print("%d spectra of %d planets (%d with transmission, %d with emission)" % (
            len(sp), len(pl), sum("transmission" in v for v in pl.values()), sum("emission" in v for v in pl.values())))
        return
    core.run_cli(atlas_main)


def archive_atlas_spectra():
    """Every transit and eclipse spectrum the NASA Exoplanet Archive's spectroscopy table lists, read through its workspace on a few threads, or none while the archive does not answer."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        rows = core.exoplanets("SELECT pl_name, spec_type, instrument, facility, authors, note, num_datapoints, bibcode, spec_path FROM spectra")
        ws = exo_workspace()
    except Exception as e:
        print("  the NASA Exoplanet Archive did not answer (%s); its spectra wait for the next parse" % str(e)[:60])
        return []
    known = known_planets()

    def fetch(r):
        """One spectrum read, through the cache, or None."""
        try:
            return exo_spectrum(r["spec_path"], ws)
        except Exception:
            return None
    rows = [r for r in rows if r["spec_type"] in ("Transmission", "Eclipse")]
    with ThreadPoolExecutor(8) as ex:
        read = list(ex.map(fetch, rows))
    out = []
    for r, x in zip(rows, read):
        kind = {"Transmission": "transmission", "Eclipse": "emission"}[r["spec_type"]]
        if x is None:
            continue
        o = np.argsort(x[0])
        w, hw, d, e = (a[o] for a in x)
        y = dict(planet=canonical_planet(r["pl_name"], known), kind=kind, instrument=r["instrument"], source="archive:%s" % r["bibcode"], doi=None, file=r["spec_path"],
                 n=len(w), wmin=float(w[0]) if len(w) else 0.0, wmax=float(w[-1]) if len(w) else 0.0, columns=None, w=w.tolist(), hw=hw.tolist(), d=d.tolist(), e=e.tolist())
        if usable_spectrum(y):
            out.append(y)
    return out


EARTH_SPECTRUM = "https://zenodo.org/api/records/8280710/files/Spectrum.txt/content"
R_EARTH_KM, R_SUN_KM = 6371.0, 695700.0


def earth_transit():
    """Earth's transit spectrum built from ACE-FTS solar occultations (Macdonald & Cowan 2019): wavelengths (um, rising) and the atmosphere's effective height (km)."""
    npz = core.data("earth_control", "earth_transit.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return z["w"], z["h"]
    local = core.data("earth_control", "Spectrum.txt")
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        body = core.get(EARTH_SPECTRUM, timeout=600)
        open(local, "wb").write(body)
    rows = [line for line in open(local) if line.strip()]
    w, h = np.array(rows[0].split(), dtype=float), np.array(rows[1].split(), dtype=float)
    o = np.argsort(w)
    np.savez(npz, w=w[o], h=h[o])
    return w[o], h[o]


def earth_depth_on(centres, halves, r_star_sun):
    """The depth (ppm) Earth's atmosphere adds to its transit of a star of the given radius, averaged over each bin; NaN where a bin reaches past the spectrum's 2.2-14.3 um."""
    w, h = earth_transit()
    d = ((R_EARTH_KM + h) ** 2 - R_EARTH_KM ** 2) / (r_star_sun * R_SUN_KM) ** 2 * 1e6
    c = np.concatenate([[0.0], np.cumsum(d)])
    halves = np.where(halves > 0, halves, np.abs(np.gradient(centres)) / 2 if len(centres) > 1 else 0.01)
    lo, hi = np.searchsorted(w, centres - halves), np.searchsorted(w, centres + halves)
    out = np.full(len(centres), np.nan)
    ok = (hi > lo) & (centres - halves >= w[0]) & (centres + halves <= w[-1])
    out[ok] = (c[hi[ok]] - c[lo[ok]]) / (hi[ok] - lo[ok])
    return out


def earth_control():
    """Earth's own transit spectrum through the screen at R 100: each gas's significance against the templates' misfit, the noise when the errors are set far below the signal."""
    w, _ = earth_transit()
    edges = np.exp(np.arange(np.log(w[0]), np.log(w[-1]), 0.01))
    centres, halves = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
    d = earth_depth_on(centres, halves, 1.0)
    ok = np.isfinite(d)
    return screen_spectrum(centres[ok], halves[ok], d[ok], np.full(int(ok.sum()), 1e-6 * float(np.ptp(d[ok]))), temp=288.0)


def earth_twin(spectra, r_star_sun, temp=288.0):
    """An Earth twin in front of a planet's host: Earth's transit spectrum laid on the bins of each of the planet's spectra with their errors, as their own fits scaled them, and screened as they were; each gas's median significance and the spectra used."""
    sigs = []
    for x in spectra:
        w, hw = np.asarray(x["w"], dtype=float), np.asarray(x["hw"], dtype=float)
        e = np.asarray(x["e"], dtype=float) * x.get("error_scale", 1.0)
        d = earth_depth_on(w, hw, r_star_sun)
        ok = np.isfinite(d)
        if ok.sum() >= 5:
            sigs.append(screen_spectrum(w[ok], hw[ok], d[ok], e[ok], temp=temp)["sig"])
    gases = sorted({g for z in sigs for g in z})
    return {g: float(np.median([z[g] for z in sigs if g in z])) for g in gases}, len(sigs)


def flags_of(tr, sigma):
    """The biosignature flags a planet's median transmission significances raise: CH4 and CO2 without CO, dimethyl sulfide, N2O."""
    flags = []
    if tr.get("CH4", 0) >= sigma and tr.get("CO2", 0) >= sigma and tr.get("CO", 0) < 2:
        flags.append("CH4 %.1f and CO2 %.1f sigma without CO" % (tr["CH4"], tr["CO2"]))
    if tr.get("DMS", 0) >= sigma:
        flags.append("dimethyl sulfide %.1f sigma" % tr["DMS"])
    if tr.get("N2O", 0) >= sigma:
        flags.append("N2O %.1f sigma" % tr["N2O"])
    return flags


def atlas_main(record=False, sigma=3.0, hot_k=450.0, giant_rjup=0.32, envelope_rjup=0.143, cool_host_k=3500.0):
    sp = json.load(open(os.path.join(atlas_ROOT, "atlas.json")))
    tep = tepcat()
    earth = earth_control()
    planets = {}
    for x in sp:
        x.update(screen_spectrum(np.array(x["w"]), np.array(x["hw"]), np.array(x["d"]), np.array(x["e"]), temp=tep.get(x["planet"], {}).get("teq") or XS_T))
        planets.setdefault(x["planet"], []).append(x)
    out = []
    for name, xs in sorted(planets.items()):
        t = tep.get(name, {})
        summary = {}
        for kind in ("transmission", "emission"):
            ks = [x for x in xs if x["kind"] == kind]
            gases = sorted({g for x in ks for g in x["sig"]})
            # each gas's median significance over the spectra whose range covers its bands, so that one reduction among many cannot carry it
            summary[kind] = {g: dict(median=float(np.median([x["sig"][g] for x in ks if g in x["sig"]])), n=sum(1 for x in ks if g in x["sig"]),
                                     n_sig=sum(1 for x in ks if x["sig"].get(g, 0) >= sigma)) for g in gases}
        tr = {g: v["median"] for g, v in summary["transmission"].items()}
        flags = flags_of(tr, sigma)
        twin, n_twin, need = {}, 0, None
        ks = [x for x in xs if x["kind"] == "transmission"]
        if ks and t.get("rs") and (t.get("teq") or 9999) <= hot_k and (t.get("rp_rjup") or 1) <= envelope_rjup:
            # a temperate small planet: what its data would show if it were Earth
            twin, n_twin = earth_twin(ks, t["rs"])
            # a gas whose twin reading is below a hundredth of a sigma is out of reach of any number of such spectra
            if twin.get("CH4", 0) >= 0.01 and twin.get("CO2", 0) >= 0.01:
                need = max((sigma / twin["CH4"]) ** 2, (sigma / twin["CO2"]) ** 2)
        vet, why = None, None
        if flags:
            if (t.get("teq") or 0) > hot_k or (t.get("rp_rjup") or 0) > envelope_rjup:
                kind = "hot" if (t.get("teq") or 0) > hot_k else "giant" if (t.get("rp_rjup") or 0) > giant_rjup else "hydrogen-enveloped"
                vet, why = "abiotic-chemistry", "a %s planet (%s K, %.2f Jupiter radii), whose hydrogen atmosphere makes these gases without life, its CH4 band at 3.3 um beside the DMS template's" % (
                    kind, t.get("teq"), t.get("rp_rjup") or 0)
            elif all(f.startswith("dimethyl") for f in flags) and tr.get("CH4", 0) >= 2:
                vet, why = "methane-overlap", "methane at %.1f sigma, whose 3.3 um band the DMS template overlaps; Earth's own transit spectrum, without detectable DMS, reads DMS at %.1f sigma beside its methane" % (
                    tr["CH4"], earth["sig"].get("DMS", 0))
            elif (t.get("teff") or 9999) < cool_host_k:
                vet, why = "stellar-contamination", "a %s K host whose spots and faculae imprint bands on the transit spectrum" % t.get("teff")
        detected = {k: sorted(g for g, v in summary[k].items() if v["median"] >= sigma) for k in summary}
        out.append(dict(planet=name, n_spectra=len(xs), kinds=sorted({x["kind"] for x in xs}), instruments=sorted({x["instrument"] or "?" for x in xs}),
                        sources=sorted({x["source"] for x in xs}), teq=t.get("teq"), rp_rjup=t.get("rp_rjup"), host_teff=t.get("teff"), summary=summary,
                        detected=detected, flags=flags, vet=vet, why=why, earth_twin=twin, n_twin=n_twin, need=need))
    n_flag = sum(1 for o in out if o["flags"])
    eflags = flags_of(earth["sig"], sigma)
    control = "Earth's own transit spectrum (ACE-FTS, Macdonald & Cowan 2019) at R 100 reads %s and %s" % (
        ", ".join("%s %.1f" % (g, v) for g, v in sorted(earth["sig"].items(), key=lambda kv: -kv[1]) if v >= 2) or "nothing at 2 sigma",
        "raises " + "; ".join(eflags) if eflags else "raises no flag")
    print(control)
    print("%d spectra of %d planets; flagged %d, unexplained %d" % (len(sp), len(out), n_flag, sum(1 for o in out if o["flags"] and not o["vet"])))
    for o in out:
        print("  %-16s %5s K %5.2f R_J host %5s K %3d spectra  transmission: %-32s emission: %-24s%s" % (
            o["planet"], o["teq"], o["rp_rjup"] or 0, o["host_teff"], o["n_spectra"], ",".join(o["detected"].get("transmission", [])) or "-",
            ",".join(o["detected"].get("emission", [])) or "-", ("  FLAG " + "; ".join(o["flags"]) + " -> " + (o["why"] or "unexplained")) if o["flags"] else ""))
    twins = [o for o in out if o["n_twin"]]
    for o in sorted(twins, key=lambda o: o["need"] or 1e9):
        print("  an Earth twin in place of %-14s CO2 %4.1f, CH4 %4.1f, N2O %4.1f sigma in the median of %d spectra; %s" % (
            o["planet"], o["earth_twin"].get("CO2", 0), o["earth_twin"].get("CH4", 0), o["earth_twin"].get("N2O", 0), o["n_twin"],
            "%.0f spectra like the median one for CH4 and CO2 at %g sigma" % (o["need"], sigma) if o["need"] else "CH4 or CO2 out of reach"))
    json.dump(dict(spectra=[{k: v for k, v in x.items() if k not in ("w", "hw", "d", "e")} for x in sp], planets=out),
              open(core.results("biosig_atlas.json"), "w"), indent=1, default=float)
    if record:
        core.record_run(
            "spectra_atlas_gases", "Gases of every planet with a published spectrum",
            "NASA Exoplanet Archive atmospheric spectroscopy table, MAST's Exo.MAST spectra, VizieR's published tables and Zenodo's published spectra; planet parameters from TEPCat",
            "every planet with a transmission or emission spectrum the archive lists, Exo.MAST serves, a VizieR table holds or a Zenodo record holds in a readable table: %d spectra of %d planets" % (len(sp), len(out)),
            os.path.abspath(__file__), "atlas", len(out), "planets", dict(sigma=sigma, hot_k=hot_k, giant_rjup=giant_rjup, envelope_rjup=envelope_rjup, cool_host_k=cool_host_k),
            "each spectrum's flat fit with its offset and slope; the band templates' formal errors; each gas's median over the spectra that cover it; %s; "
            "an Earth twin laid on each temperate small planet's spectra with their errors, %s" % (control, "; ".join(
                "%s CO2 %.1f, CH4 %.1f sigma (%s)" % (o["planet"], o["earth_twin"].get("CO2", 0), o["earth_twin"].get("CH4", 0),
                                                       "%.0f spectra like its median one for both at %g sigma" % (o["need"], sigma) if o["need"] else "out of reach")
                for o in sorted(twins, key=lambda o: o["need"] or 1e9)) or "none such"),
            "CH4 and CO2 at %g sigma without CO at 2 sigma, dimethyl sulfide or N2O at %g sigma, in transmission, as medians over a planet's spectra" % (sigma, sigma),
            "%d planets: %d flagged, %d of them unexplained" % (len(out), n_flag, sum(1 for o in out if o["flags"] and not o["vet"])),
            [dict(target="planet:%s" % o["planet"], stat=float(len(o["detected"].get("transmission", [])) + len(o["detected"].get("emission", []))),
                  unit="gases at %g sigma in the median of its spectra" % sigma, vet=o["vet"],
                  note="%d spectra (%s); transmission %s; emission %s%s%s" % (
                      o["n_spectra"], ", ".join(o["instruments"])[:100], ", ".join(o["detected"].get("transmission", [])) or "none",
                      ", ".join(o["detected"].get("emission", [])) or "none", "; " + "; ".join(o["flags"]) + ("; " + o["why"] if o["why"] else "") if o["flags"] else "",
                      "; an Earth twin here: CO2 %.1f, CH4 %.1f sigma" % (o["earth_twin"].get("CO2", 0), o["earth_twin"].get("CH4", 0)) if o["n_twin"] else ""))
             for o in out])
        print("recorded")


def rockyworlds_cli():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        rockyworlds_fetch()
        return
    core.run_cli(rockyworlds_main)


APPROACHES = {
    "disequilibrium": (lambda: core.run_cli(disequilibrium_main), disequilibrium_DOC),
    "rockyworlds": (rockyworlds_cli, rockyworlds_DOC),
    "atlas": (atlas_cli, atlas_DOC),
}
