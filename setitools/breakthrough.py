"""Narrowband searches of Breakthrough Listen spectrograms."""
import argparse
import glob
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from setitools import core, science
from setitools.core import quiet_warnings


# --------------------------------------------------------------------------------------------------------------

frames_DOC = r"""H I and OH lines tracked in the CMB, barycentre, star and planet frames through BL cadences; stacked ON sums against shifted-track nulls.

    python -I seti.py breakthrough frames --session AGBT16A_999_240 --target Hip3765 [--workers 6]
    python -I seti.py breakthrough frames summary [--p 1e-3]
    python -I seti.py breakthrough frames waterfall SESSION FREQ_MHZ SCANFILE [SCANFILE ...] [--half 3000] [--out hit.png]"""

frames_HERE = core.RESULTS
frames_PIPE = "https://bldata.berkeley.edu/pipeline/"
frames_LINES = {"HI": 1420.405751768, "OH1665": 1665.401803, "OH1667": 1667.358996}
frames_EXTRA = {"piHI": 1420.405751768 * math.pi, "He3": 8665.65, "2piHI": 2.0 * math.pi * 1420.405751768,
         "OH1612": 1612.231, "OH1720": 1720.530}
frames_C = science.C
frames_LB = 1.550519768e-8
frames_COARSE = science.COARSE
frames_NNULL = 2000
frames_NULL_VALUES = 40000000
frames_EDGE = int(0.03 * science.COARSE)


def frames_null_reach(half):
    """Farthest null-window centre offset for a window of +-half channels."""
    return 4 * half + 22000
frames_GBT = (-79.839835, 38.433121, 807.43)


def frames_star_params(hip):
    q = ("SELECT basic.ra, basic.dec, basic.pmra, basic.pmdec, basic.plx_value, basic.rvz_radvel, basic.rvz_err, "
         "basic.rvz_type, basic.main_id FROM basic JOIN ident ON ident.oidref=basic.oid WHERE ident.id='%s'" % hip)
    r = core.tap("simbad", q, fmt="json", timeout=90)["data"][0]
    ra, dec, pmra, pmdec, plx, rv, rve, rvt, name = r
    return dict(ra=ra, dec=dec, pmra=pmra or 0.0, pmdec=pmdec or 0.0, plx=plx or np.nan,
                rv=rv if rv is not None and rvt in ("v", None) else np.nan,
                rv_err=rve if rve is not None else np.nan, name=name)


def frames_tracks(st, mjd0, tsamp, nt, lines=None):
    """{line: {frame: topocentric MHz per spectrum}} and window half-widths (km/s)."""
    lines = lines or frames_LINES
    from astropy.coordinates import EarthLocation, get_body_barycentric_posvel
    from astropy.time import Time
    import astropy.units as u
    t = Time(mjd0 + (np.arange(nt) + 0.5) * tsamp / 86400.0, format="mjd", scale="utc")
    yrs = (t.jd.mean() - 2451545.0) / 365.25
    ra = st["ra"] + st["pmra"] / 3.6e6 * yrs / np.cos(np.radians(st["dec"]))
    dec = st["dec"] + st["pmdec"] / 3.6e6 * yrs
    n = science.unit_vector(ra, dec)
    _, vel = get_body_barycentric_posvel("earth", t, ephemeris="builtin")
    loc = EarthLocation.from_geodetic(lon=frames_GBT[0] * u.deg, lat=frames_GBT[1] * u.deg, height=frames_GBT[2] * u.m)
    _, gv = loc.get_gcrs_posvel(t)
    v = vel.xyz.to(u.km / u.s).value + gv.xyz.to(u.km / u.s).value
    obs = (1.0 + n @ v / frames_C) * (1.0 + frames_LB)
    bs = science.cmb_beta()
    gs = 1.0 / math.sqrt(1.0 - bs @ bs)
    out, halfs = {}, {"cmb": 0.45, "ssb": 0.021}
    if np.isfinite(st["rv"]):
        vt = 4.74047 * math.hypot(st["pmra"], st["pmdec"]) / 1000.0 * (1000.0 / st["plx"]) if st["plx"] > 0 else 0.0
        g = 1.0 / math.sqrt(1.0 - (st["rv"] / frames_C) ** 2 - (vt / frames_C) ** 2)
        err = st["rv_err"] if np.isfinite(st["rv_err"]) else 1.0
        halfs["star"] = 3.0 * err + 0.7
    pl = st.get("planet")
    if pl and "star" in halfs:
        from astropy.coordinates import SkyCoord
        sc = SkyCoord(ra * u.deg, dec * u.deg)
        bjd = (t.tdb + t.light_travel_time(sc, kind="barycentric", location=loc)).jd
        phase = ((bjd - pl["T0"]) / pl["P"]) % 1.0
        vorb = (2.0 * math.pi * 6.674e-11 * pl["mass"] * 1.98892e30 / (pl["P"] * 86400.0)) ** (1.0 / 3.0) / 1000.0
        vp = vorb * np.sin(2.0 * math.pi * phase)
        # one reference epoch per cadence, so every scan shares the window
        cycles = abs((pl.get("t_ref", bjd.mean()) - pl["T0"]) / pl["P"])
        dph = (pl["eT0"] + cycles * pl["eP"]) / pl["P"]
        # bounds valid at every orbital phase
        ev_mass = vorb * pl["e_mass"] / (3.0 * pl["mass"])
        ev_phase = vorb * 2.0 * math.pi * dph
        err_p = math.sqrt(err ** 2 + ev_mass ** 2 + ev_phase ** 2)
        halfs["planet"] = 3.0 * err_p + 1.0
        gp = 1.0 / np.sqrt(1.0 - ((st["rv"] + vp) / frames_C) ** 2 - (vt / frames_C) ** 2)
        st["planet_state"] = dict(phase=[round(float(x), 4) for x in phase[[0, -1]]], vorb=round(vorb, 2),
                                  vp=[round(float(x), 3) for x in vp[[0, -1]]], err_kms=round(err_p, 3))
    for line, f0 in lines.items():
        fr = {"cmb": f0 * gs * (1.0 + n @ bs), "ssb": f0}
        if "star" in halfs:
            fr["star"] = f0 / (g * (1.0 + st["rv"] / frames_C))
        if "planet" in halfs:
            fr["planet"] = f0 / (gp * (1.0 + (st["rv"] + vp) / frames_C))
        out[line] = {k: f * obs for k, f in fr.items()}
    return out, halfs


def frames_scan_job(job):
    """Track sums in one scan, with the random-offset null for ON scans."""
    url, st, deltas, is_on, lines = job
    try:
        meta = science.header(url)
        fch1, foff, tsamp, mjd0 = meta["fch1"], meta["foff"], meta["tsamp"], meta["tstart"]
        nt, nch = meta["nt"], meta["nch"]
        src = tuple(meta["src"])
        trk, halfs = frames_tracks(st, mjd0, tsamp, nt, lines)
        need = {}
        for line, frames_ in trk.items():
            for frame, path in frames_.items():
                half = int(math.ceil(halfs[frame] / frames_C * lines[line] * 1e6 / (abs(foff) * 1e6)))
                apos = (path - fch1) / foff
                lo, hi = int(math.floor(apos.min())) - half - 2, int(math.ceil(apos.max())) + half + 2
                if lo < 0 or hi >= nch:
                    continue
                # ON scans also load the null's reach
                ext = frames_null_reach(half) + half + 2 if is_on else 0
                lo, hi = max(lo - ext, 0), min(hi + ext, nch - 1)
                need.setdefault((lo // frames_COARSE, hi // frames_COARSE), []).append((line, frame, half))
        science.header(url, sorted({k for blk in need for k in range(blk[0], blk[1] + 1)}))
        out = dict(url=url, mjd=mjd0, src=src, res={}, planet=st.get("planet_state"))
        flat_cache = {}
        for (k0, k1), items in need.items():
            parts = []
            for k in range(k0, k1 + 1):
                if k not in flat_cache:
                    x, _ = science.coarse_channel(url, k)
                    flat_cache[k] = science.flatten(x)
                parts.append(flat_cache[k])
            z = parts[0] if len(parts) == 1 else np.hstack(parts)
            # filterbank roll-off inflates noise at coarse-channel edges; edges zeroed for window and null
            z = z.copy()
            for i in range(len(parts)):
                z[:, i * frames_COARSE:i * frames_COARSE + frames_EDGE] = 0.0
                z[:, (i + 1) * frames_COARSE - frames_EDGE:(i + 1) * frames_COARSE] = 0.0
            width = z.shape[1]
            fstart = fch1 + foff * k0 * frames_COARSE
            inj = os.environ.get("FRAMESEARCH_INJECT")
            if inj and is_on:
                il, ifr, ihz, iamp = inj.split(",")
                if any((il, ifr) == (ln, fr) for ln, fr, _ in items):
                    ipos = np.round((trk[il][ifr] - fstart) / foff + float(ihz) / (foff * 1e6)).astype(int)
                    z[np.arange(nt), ipos] += float(iamp)
            for line, frame, half in items:
                path = trk[line][frame]
                pos = (path - fstart) / foff
                base = np.round(pos).astype(int)
                offs = np.arange(-half, half + 1)
                if base.min() - half < 0 or base.max() + half >= width:
                    continue
                s_ = science.track_sum(z, base, offs)
                flat = science.track_sum(z, np.full(nt, int(round(pos.mean()))), offs)
                rec = dict(s=s_.astype(np.float32), flat=flat.astype(np.float32), half=half,
                           fmid=float(path.mean()), drift_hz_s=float((path[-1] - path[0]) * 1e6 / ((nt - 1) * tsamp)),
                           df_hz=foff * 1e6)
                if is_on:
                    # wide windows keep fewer null values (at most about 80 MB per scan and frame); the tail rests on the Gumbel fit
                    n_use = min(len(deltas), max(50, int(frames_NULL_VALUES / (2 * half + 1))))
                    nul = np.full((n_use, 2 * half + 1), np.nan, dtype=np.float16)
                    for i, u in enumerate(deltas[:n_use]):
                        # null windows lie 2 half + 2000 to 4 half + 22000 channels away
                        dl = int(np.sign(u) * (2 * half + 2000 + abs(u) * (2 * half + 20000)))
                        b = base + dl
                        if b.min() - half < 0 or b.max() + half >= width:
                            continue
                        # DC bins zeroed alike for window and null
                        nul[i] = science.track_sum(z, b, offs)
                    rec["null"] = nul
                out["res"][(line, frame)] = rec
        return out
    except Exception as e:
        return dict(url=url, error=str(e)[:200])


def frames_search(cad, hip_name, st, workers, roles=None, lines=None):
    lines = lines or frames_LINES
    rng = np.random.default_rng(12345)
    deltas = rng.uniform(-1.0, 1.0, size=frames_NNULL)
    roles = roles or [i % 2 == 0 for i in range(len(cad))]
    jobs = [(u, st, deltas, roles[i], lines) for i, (_, u) in enumerate(cad)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(frames_scan_job, jobs))
    errs = [r for r in res if "error" in r]
    if errs:
        return dict(error=errs[0]["error"])
    on = [r for r, is_on in zip(res, roles) if is_on]
    off = [r for r, is_on in zip(res, roles) if not is_on]
    norm = math.sqrt(len(on))
    report = []
    for line in lines:
        for frame in ("cmb", "ssb", "star", "planet"):
            key = (line, frame)
            if not all(key in r["res"] for r in on):
                continue
            ss = np.array([r["res"][key]["s"] for r in on])
            stack = ss.sum(0) / norm
            j = int(np.argmax(stack))
            nulls = np.array([r["res"][key]["null"].astype(np.float32) for r in on])
            good = np.all(np.isfinite(nulls).all(axis=2), axis=0)
            nullmax = (nulls[:, good, :].sum(0) / norm).max(axis=1)
            p_emp = (1 + int(np.sum(nullmax >= stack[j]))) / (1 + len(nullmax))
            p_gum = science.gumbel_p(nullmax, float(stack[j]))
            rec0 = on[0]["res"][key]
            f_hit = rec0["fmid"] + (j - rec0["half"]) * rec0["df_hz"] / 1e6
            offv = []
            for r in off:
                if key not in r["res"]:
                    continue
                ro = r["res"][key]
                k = int(round((f_hit - ro["fmid"]) * 1e6 / ro["df_hz"])) + ro["half"]
                lo, hi = max(k - 40, 0), min(k + 41, len(ro["s"]))
                offv.append(float(max(ro["s"][lo:hi].max(), ro["flat"][lo:hi].max())) if hi > lo else float("nan"))
            flat = [float(r["res"][key]["flat"][max(j - 1, 0):j + 2].max()) for r in on]
            report.append(dict(target=hip_name, line=line, frame=frame, stack_max=round(float(stack[j]), 2),
                               offset_hz=round((j - rec0["half"]) * rec0["df_hz"], 1),
                               per_scan=[round(float(x), 2) for x in ss[:, j]], off=[round(x, 2) for x in offv],
                               zero_drift_on=[round(x, 2) for x in flat], p_emp=p_emp, p_gumbel=p_gum,
                               n_null=int(len(nullmax)), window_hz=round(rec0["half"] * abs(rec0["df_hz"]), 1),
                               drift_hz_s=round(rec0["drift_hz_s"], 4), fmid_mhz=round(rec0["fmid"], 6)))
    return dict(report=report, scans=[(r["url"].split("/")[-1], r["mjd"], r["src"], r.get("planet")) for r in res])


def frames_main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--base", help="full directory URL of the scans, in place of the session's holding area")
    p.add_argument("--simbad", help="SIMBAD identifier of the star, in place of HIP + the target's number")
    p.add_argument("--planet", help="TIC id whose transiting planet (planet_targets.json) adds a planet-frame track")
    p.add_argument("--lines", default="HI,OH1665,OH1667",
                   help="comma-separated line names from LINES and EXTRA (piHI, He3, 2piHI, OH1612, OH1720)")
    p.add_argument("--single", action="store_true",
                   help="the target has one scan (an OFF position of another cadence); the scans before and "
                        "after it, pointed elsewhere, serve as its OFF scans")
    a = p.parse_args()
    session = a.base or (frames_PIPE + a.session + "/holding/")
    roles = None
    if a.single:
        scans = core.bl_scans(session)
        idx = [k for k, x in enumerate(scans) if x[2].lower() == a.target.lower()]
        if not idx or idx[0] == 0 or idx[0] == len(scans) - 1:
            print("%s %s: no single scan with neighbours" % (a.session, a.target), flush=True)
            return
        k = idx[0]
        cads = [[(scans[i][2], scans[i][3]) for i in (k, k - 1, k + 1)]]
        roles = [True, False, False]
    else:
        cads = [c for c in core.bl_cadences(session) if c[0][0].lower() == a.target.lower()]
    if not cads:
        print("%s %s: no full cadence" % (a.session, a.target), flush=True)
        return
    hip = a.simbad or ("HIP " + a.target[3:])
    pt = ({str(r["tid"]): r for r in json.load(open(os.path.join(frames_HERE, "planet_targets.json")))}[a.planet]
          if a.planet else None)
    try:
        st = frames_star_params(hip)
    except Exception:
        if pt is None:
            raise
        st = dict(ra=float(pt["ra"]), dec=float(pt["dec"]), pmra=0.0, pmdec=0.0, plx=np.nan, rv=np.nan,
                  rv_err=np.nan, name="TOI-" + pt["toi"])
    if a.planet:
        if pt.get("gaia_rv"):
            st["rv"], st["rv_err"] = float(pt["gaia_rv"]), float(pt["gaia_rv_err"] or 1.0)
        st["planet"] = dict(P=float(pt["pl_orbper"]), eP=float(pt["pl_orbpererr1"] or 1e-4),
                            T0=float(pt["pl_tranmid"]), eT0=float(pt["pl_tranmiderr1"] or 1e-3),
                            mass=float(pt["mass"]), e_mass=float(pt["e_mass"]), toi=pt["toi"])
    if a.planet:
        import re
        m = re.search(r"guppi_(\d{5})_(\d+)_", cads[0][0][1])
        if m:
            st["planet"]["t_ref"] = int(m.group(1)) + int(m.group(2)) / 86400.0 + 2400000.5
    t0 = time.time()
    lines = {k: {**frames_LINES, **frames_EXTRA}[k] for k in a.lines.split(",")}
    out = frames_search(cads[0], a.target, st, a.workers, roles, lines)
    out.update(session=a.session, star=st, single=bool(a.single))
    with open(os.path.join(frames_HERE, "framesearch.jsonl"), "a") as fh:
        fh.write(json.dumps(out, default=float) + "\n")
    if "error" in out:
        print("%s %s: error %s" % (a.session, a.target, out["error"]), flush=True)
        return
    for r in out["report"]:
        flag = ""
        offs = [x for x in r["off"] if np.isfinite(x)]
        if r["p_gumbel"] < 1e-4 and offs and max(offs) < 3.0 and max(r["zero_drift_on"]) < r["stack_max"]:
            flag = "CANDIDATE "
        print("%s%s %s %-6s %-4s max %.2f at %+.1f Hz (window +-%.0f Hz, drift %+.4f Hz/s) scans %s off %s "
              "flat %s p_emp %.4f p_gumbel %.2e"
              % (flag, a.session, a.target, r["line"], r["frame"], r["stack_max"], r["offset_hz"], r["window_hz"],
                 r["drift_hz_s"], r["per_scan"], r["off"], r["zero_drift_on"], r["p_emp"], r["p_gumbel"]),
              flush=True)
    print("%s %s %s: %.0fs" % (a.session, a.target, st["name"], time.time() - t0), flush=True)


def frames_summary_main(argv):
    p = argparse.ArgumentParser(prog="seti.py breakthrough frames summary")
    p.add_argument("--p", type=float, default=1e-3)
    a = p.parse_args(argv)
    rows = [json.loads(line) for line in open(os.path.join(frames_HERE, "framesearch.jsonl"))]
    tests, errors = [], []
    for o in rows:
        if "error" in o:
            errors.append((o.get("session"), o["star"]["name"] if "star" in o else "?", o["error"][:100]))
            continue
        for r in o["report"]:
            tests.append(dict(r, session=o["session"], star=o["star"]["name"]))
    pe = np.array([t["p_emp"] for t in tests])
    print("%d cadences, %d tests, %d errors" % (len(rows) - len(errors), len(tests), len(errors)))
    for lim in (0.5, 0.1, 0.01):
        print("  fraction with empirical p < %g: %.3f" % (lim, float(np.mean(pe < lim)) if len(pe) else 0.0))
    for t in sorted(tests, key=lambda t: t["p_gumbel"]):
        if t["p_gumbel"] > a.p:
            break
        print("%s %s %s %s max %.2f at %+.1f Hz scans %s off %s flat %s p_emp %.4f p_gumbel %.2e"
              % (t["session"], t["star"], t["line"], t["frame"], t["stack_max"], t["offset_hz"], t["per_scan"],
                 t["off"], t["zero_drift_on"], t["p_emp"], t["p_gumbel"]))
    for e in errors:
        print("error", *e)


def frames_waterfall_main(argv):
    p = argparse.ArgumentParser(prog="seti.py breakthrough frames waterfall")
    p.add_argument("session")
    p.add_argument("freq", type=float)
    p.add_argument("scans", nargs="+")
    p.add_argument("--half", type=float, default=3000.0)
    p.add_argument("--maxd", type=float, default=1.0)
    p.add_argument("--out", default=os.path.join(frames_HERE, "hit.png"))
    a = p.parse_args(argv)
    panels = []
    for name in a.scans:
        url = frames_PIPE + a.session + "/holding/" + name
        meta = science.header(url)
        k = int((a.freq - meta["fch1"]) / meta["foff"]) // frames_COARSE
        x, f0 = science.coarse_channel(url, k)
        z = science.flatten(x)
        df = meta["foff"]
        c = int(round((a.freq - f0) / df))
        h = int(a.half / (abs(df) * 1e6))
        lo, hi = max(c - h, 0), min(c + h + 1, z.shape[1])
        w = z[:, lo:hi]
        nt = w.shape[0]
        maxd = int(math.ceil(a.maxd * meta["tsamp"] * (nt - 1) / (abs(df) * 1e6)))
        best = (-np.inf, 0, 0)
        for dr in range(-maxd, maxd + 1):
            s = np.zeros(w.shape[1])
            for t in range(nt):
                s += np.roll(w[t], -int(round(dr * t / (nt - 1))))
            s /= math.sqrt(nt)
            j = int(np.argmax(s[maxd:-maxd or None])) + maxd
            if s[j] > best[0]:
                best = (float(s[j]), dr, j)
        flat = w.sum(0) / math.sqrt(nt)
        jf = int(np.argmax(flat))
        fbest = f0 + df * (lo + best[2])
        print("%s  MJD %.5f  best track %.1f sigma at %.6f MHz, drift %+.4f Hz/s;  zero-drift peak %.1f sigma at "
              "%.6f MHz" % (name.split("guppi_")[1][:40], meta["tstart"], best[0], fbest,
                           best[1] * abs(df) * 1e6 / (meta["tsamp"] * (nt - 1)) * (-1 if df < 0 else 1),
                           float(flat[jf]), f0 + df * (lo + jf)), flush=True)
        panels.append((name, w, f0 + df * lo, df))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(panels), 1, figsize=(10, 2.2 * len(panels)), sharex=True)
    for ax, (name, w, fl, df) in zip(np.atleast_1d(axs), panels):
        ext = [(fl - a.freq) * 1e6, (fl + df * w.shape[1] - a.freq) * 1e6, w.shape[0], 0]
        ax.imshow(w, aspect="auto", extent=ext, cmap="viridis", vmin=-1, vmax=6, interpolation="nearest")
        ax.set_ylabel(name.split("_")[-2][:12], fontsize=7)
    np.atleast_1d(axs)[-1].set_xlabel("Hz from %.6f MHz" % a.freq)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)


def frames_cli():
    modes = {"summary": frames_summary_main, "waterfall": frames_waterfall_main}
    if len(sys.argv) > 1 and sys.argv[1] in modes:
        modes[sys.argv[1]](sys.argv[2:])
    else:
        frames_main()


# --------------------------------------------------------------------------------------------------------------

curved_DOC = r"""Curved-drift narrowband search on the GPU: joined half-scan tracks, ON recurrence, OFF veto, occupied allocations excluded, and same-session sidelobe sweep.

    python -I seti.py breakthrough curved search [--limit N]      (GPU; reads setitools's cache, idx and blob)
    python -I seti.py breakthrough curved cadences FRAMESEARCH.jsonl
    python -I seti.py breakthrough curved session FRAMESEARCH.jsonl
    python -I seti.py breakthrough curved record CADENCES.jsonl --run RUN_ID --release SESSIONS [--ledger ROW]"""

curved_CACHE = core.CACHE
curved_OUT = os.environ.get("CURVED_OUT", os.path.join(curved_CACHE, "curved"))
curved_COARSE = 1048576
curved_STEP = 1.0 / 7.0
curved_DMAX = 26.0
curved_DELTA = 5.0
curved_HALF_SNR = 5.0
curved_BLOCK = 131072
curved_MARGIN = 1024


def curved_load(blobfile):
    """Spectrogram and metadata of one cached coarse channel."""
    name, k = os.path.basename(blobfile).rsplit(".k", 1)
    k = int(k[:-4])
    got = science.cached_channel(name, k)
    if got is None:
        return None
    x, meta = got
    return x, dict(name=name, k=k, f_first=meta["fch1"] + meta["foff"] * k * curved_COARSE, foff=meta["foff"],
                   tsamp=meta["tsamp"], tstart=meta["tstart"], nt=meta["nt"])


def curved_flatten(x):
    """flatten, clipped at +-50."""
    return np.clip(science.flatten(x), -50, 50)


def curved_search_scan(z, dev):
    """Per channel: best curved statistic, its drifts, and best straight statistic over 16 spectra."""
    import torch
    nt = z.shape[0]
    h = nt // 2
    drifts = np.arange(-curved_DMAX, curved_DMAX + 1e-9, curved_STEP)
    nd = len(drifts)
    tt = np.arange(h)
    shifts = torch.as_tensor(np.round(drifts[:, None] * tt[None, :]).astype(np.int64), device=dev)
    s8 = torch.as_tensor(np.round(drifts * h).astype(np.int64), device=dev)
    win = int(round(curved_DELTA / curved_STEP))
    C = z.shape[1]
    best_c = np.full(C, -np.inf, np.float32)
    best_l = np.full(C, -np.inf, np.float32)
    best_d1 = np.zeros(C, np.float32)
    best_d2 = np.zeros(C, np.float32)
    for b0 in range(0, C, curved_BLOCK):
        lo, hi = max(b0 - curved_MARGIN, 0), min(b0 + curved_BLOCK + curved_MARGIN, C)
        zb = torch.as_tensor(z[:, lo:hi], device=dev)
        B = hi - lo
        base = torch.arange(B, device=dev)
        S = []
        for half in (0, 1):
            acc = torch.zeros((nd, B), device=dev)
            for t in range(h):
                idx = (base[None, :] + shifts[:, t:t + 1]).clamp(0, B - 1)
                acc += zb[half * h + t][idx]
            S.append(acc)
        S1, S2 = S
        bc = torch.full((B,), -1e9, device=dev)
        bl = torch.full((B,), -1e9, device=dev)
        bd1 = torch.zeros(B, device=dev)
        bd2 = torch.zeros(B, device=dev)
        half_thr = curved_HALF_SNR * math.sqrt(h)
        for j1 in range(nd):
            c2 = (base + s8[j1]).clamp(0, B - 1)
            ja, jb = max(j1 - win, 0), min(j1 + win, nd - 1)
            s2 = S2[ja:jb + 1][:, c2]
            tot = S1[j1][None, :] + s2
            far = torch.as_tensor(np.abs(np.arange(ja, jb + 1) - j1) * curved_STEP >= 1.0, device=dev)
            # signal required in both halves
            both = (S1[j1][None, :] >= half_thr) & (s2 >= half_thr)
            curved = torch.where(far[:, None] & both, tot, torch.full_like(tot, -1e9))
            m, arg = curved.max(0)
            upd = m > bc
            bc = torch.where(upd, m, bc)
            bd1 = torch.where(upd, torch.full_like(bd1, float(drifts[j1])), bd1)
            bd2 = torch.where(upd, torch.as_tensor(drifts[ja:jb + 1], device=dev, dtype=torch.float32)[arg], bd2)
            bl = torch.maximum(bl, tot[j1 - ja])
        keep = slice(b0 - lo, b0 - lo + min(curved_BLOCK, C - b0))
        norm = math.sqrt(nt)
        best_c[b0:b0 + min(curved_BLOCK, C - b0)] = (bc[keep] / norm).cpu().numpy()
        best_l[b0:b0 + min(curved_BLOCK, C - b0)] = (bl[keep] / norm).cpu().numpy()
        best_d1[b0:b0 + min(curved_BLOCK, C - b0)] = bd1[keep].cpu().numpy()
        best_d2[b0:b0 + min(curved_BLOCK, C - b0)] = bd2[keep].cpu().numpy()
    return best_c, best_l, best_d1, best_d2


def curved_hits_of(best_c, best_l, d1, d2, meta, thr=10.0, gain=2.0, sep=20):
    cand = np.nonzero((best_c >= thr) & (best_c - best_l >= gain))[0]
    out = []
    for i in cand[np.argsort(-best_c[cand])]:
        if all(abs(i - j["chan"]) >= sep for j in out):
            out.append(dict(chan=int(i), mhz=float(meta["f_first"] + meta["foff"] * i), snr_curved=float(best_c[i]),
                            snr_straight=float(best_l[i]), d1_hz_s=float(d1[i] * meta["foff"] * 1e6 / meta["tsamp"]),
                            d2_hz_s=float(d2[i] * meta["foff"] * 1e6 / meta["tsamp"])))
        if len(out) >= 300:
            break
    return out


def curved_search(limit):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(curved_OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(curved_CACHE, "blob", "*.bin")))[:limit or None]
    import time
    for f in files:
        out = os.path.join(curved_OUT, os.path.basename(f) + ".json")
        if os.path.exists(out):
            continue
        t0 = time.time()
        r = curved_load(f)
        if r is None:
            continue
        x, meta = r
        z = curved_flatten(x)
        bc, bl, d1, d2 = curved_search_scan(z, dev)
        hits = curved_hits_of(bc, bl, d1, d2, meta)
        straight_hits = int(np.sum(bl >= 10.0))
        # strong straight-track channels kept for the OFF veto
        strong = np.nonzero(bl >= 6.0)[0]
        strong = strong[np.argsort(-bl[strong])][:20000]
        json.dump(dict(meta=meta, hits=hits, n_straight_ge10=straight_hits,
                       strong_mhz=[round(float(meta["f_first"] + meta["foff"] * i), 7) for i in strong],
                       curved_hist=np.histogram(bc[np.isfinite(bc)], bins=[0, 6, 8, 10, 12, 15, 20, 1e9])[0].tolist()),
                  open(out, "w"))
        print("%s k%d: %d curved hits, %d straight samples >= 10, %.1f s" % (meta["name"][:60], meta["k"], len(hits),
                                                                            straight_hits, time.time() - t0), flush=True)


def curved_cadences(fs_path):
    """ON-only curved hits per cadence and the OFF-only count as null."""
    res = {}
    for p in glob.glob(os.path.join(curved_OUT, "*.bin.json")):
        d = json.load(open(p))
        res.setdefault(d["meta"]["name"], {})[d["meta"]["k"]] = d
    on_only, off_only, cads, single_trials = [], [], 0, [0, 0]
    for line in open(fs_path):
        o = json.loads(line)
        if "scans" not in o:
            continue
        names = [s[0] for s in o["scans"]]
        roles = [True, False, False] if o.get("single") else [i % 2 == 0 for i in range(len(names))]
        ks = set()
        for n in names:
            ks |= set(res.get(n, {}).keys())
        for k in ks:
            scans = [(n, r, res.get(n, {}).get(k)) for n, r in zip(names, roles)]
            if any(s[2] is None for s in scans):
                continue
            cads += 1
            if o.get("single"):
                # single-ON cadence: each scan against the other two, ON for the search, OFF for the null
                for i, (n, r, d) in enumerate(scans):
                    single_trials[0 if r else 1] += 1
                    other = np.array(sorted([g["mhz"] for j, (n2, r2, d2) in enumerate(scans) if j != i
                                             for g in d2["hits"]] +
                                            [m for j, (n2, r2, d2) in enumerate(scans) if j != i
                                             for m in d2.get("strong_mhz", [])]))
                    for h in d["hits"]:
                        jj = np.searchsorted(other, h["mhz"])
                        near = other[max(jj - 1, 0):jj + 1]
                        if not len(near) or np.min(np.abs(near - h["mhz"])) * 1e6 > 2000:
                            (on_only if r else off_only).append(dict(h, target=o.get("star", {}).get("name"),
                                                                     session=o.get("session"), k=k, single=True, scan=n))
                continue
            for role_on in (True, False):
                # a source recurs in another scan of the same role within 5 kHz
                mine = []
                same_role = [(n, d) for n, r, d in scans if r == role_on]
                for n, d in same_role:
                    for h in d["hits"]:
                        if any(n2 != n and any(abs(h["mhz"] - g["mhz"]) * 1e6 < 5000 for g in d2["hits"])
                               for n2, d2 in same_role):
                            mine.append(dict(h, scan=n))
                other = np.array(sorted([g["mhz"] for n, r, d in scans if r != role_on for g in d["hits"]] +
                                        [m for n, r, d in scans if r != role_on for m in d.get("strong_mhz", [])]))
                for h in mine:
                    j = np.searchsorted(other, h["mhz"])
                    near = other[max(j - 1, 0):j + 1]
                    if not len(near) or np.min(np.abs(near - h["mhz"])) * 1e6 > 2000:
                        (on_only if role_on else off_only).append(dict(h, target=o.get("star", {}).get("name"),
                                                                       session=o.get("session"), k=k))
    for h in on_only + off_only:
        h["occupied"] = science.occupied(h["mhz"])
    print("%d cadence-channels searched; curved hits found only in ON scans %d, only in OFF scans (null) %d"
          % (cads, len(on_only), len(off_only)))
    for name, hs in (("ON", on_only), ("OFF", off_only)):
        by = {}
        for h in hs:
            by[h["occupied"] or "no occupied allocation"] = by.get(h["occupied"] or "no occupied allocation", 0) + 1
        print("  %s-only hits by allocation: %s" % (name, "; ".join("%s %d" % kv for kv in sorted(by.items()))))
    if sum(single_trials):
        print("  of which single-ON cadences: %d ON scan-channels with %d lone hits, %d OFF scan-channels with %d"
              % (single_trials[0], sum(1 for h in on_only if h.get("single")), single_trials[1],
                 sum(1 for h in off_only if h.get("single"))))
    for h in sorted(on_only, key=lambda h: -h["snr_curved"])[:20]:
        print("  %s %s k%d %.6f MHz curved %.1f straight %.1f drifts %+.3f -> %+.3f Hz/s"
              % (h["session"], h["target"], h["k"], h["mhz"], h["snr_curved"], h["snr_straight"], h["d1_hz_s"],
                 h["d2_hz_s"]))
    json.dump(dict(on_only=on_only, off_only=off_only, cadences=cads, single_trials=single_trials),
              open(os.path.join(curved_OUT, "cadences.json"), "w"))


curved_SESSION_S = 3600.0  # seconds either side swept for the same carrier
curved_SESSION_HZ = 1500.0  # Hz from the drift-extrapolated frequency


def curved_session(fs_path, session_dir_of=None):
    """Sweep each ON-only hit's carrier through every scan of its session within an hour; recurrences mark sidelobe transmitters."""
    path = os.path.join(curved_OUT, "cadences.json")
    d = json.load(open(path))
    cads = [json.loads(line) for line in open(fs_path) if line.strip()]

    def dir_of(scan):
        """Session directory of a scan from the open-data API."""
        tgt = scan.split("_guppi_")[1].split("_", 2)[2].rsplit("_", 1)[0]
        for r in core.bl_query_files(tgt):
            if scan in r.get("url", ""):
                return r["url"].rsplit("/", 1)[0] + "/"
        return None
    session_dir_of = session_dir_of or dir_of
    drifts = np.arange(-8, 8.01, 0.25)
    for h in d["on_only"]:
        if h.get("occupied") or science.occupied(h["mhz"]):
            continue
        o = next((o for o in cads if o.get("session") == h.get("session") and o["star"]["name"] == h["target"]), None)
        if o is None:
            continue
        roles = [True, False, False] if o.get("single") else [i % 2 == 0 for i in range(len(o["scans"]))]
        on_scans = [s[0] for s, r in zip(o["scans"], roles) if r]
        sess_dir = o.get("dir") or session_dir_of(on_scans[0])
        if not sess_dir:
            print("  %s %.6f MHz: no session directory" % (h["target"], h["mhz"]))
            continue
        scans = core.bl_scans(sess_dir)
        on_url = next(s for s in scans if any(n in s[3] for n in ([h["scan"]] if h.get("scan") else on_scans)))
        t_hit = on_url[1]
        rate = 0.5 * (h["d1_hz_s"] + h["d2_hz_s"])
        # companion lines within 500 kHz switching on with the hit at its drift
        x, f0 = science.coarse_channel(on_url[3], h["k"])
        z = np.clip(science.flatten(x), -50, 50)
        hdr = science.header(on_url[3], (h["k"],))
        foff, tsamp, nt = hdr["foff"], hdr["tsamp"], z.shape[0]
        members = [(h["mhz"], rate)]
        c0 = int(round((h["mhz"] - f0) / foff))
        lo, hi = max(c0 - 180000, 0), min(c0 + 180000, z.shape[1])
        d1_ch = h["d1_hz_s"] * tsamp / (foff * 1e6)

        def track(t0, dr, c, width):
            """Straight-track sums over spectra t0..t0+5 at drift dr for channels c+-width, and the first channel index."""
            a, b = max(c - width, 0), min(c + width + 1, z.shape[1])
            return sum(np.roll(z[t0 + n, a:b], -int(round(dr * n))) for n in range(6)) / math.sqrt(6), a

        # brightest straight track near the hit in its six spectra
        best_h = (-np.inf, 0, d1_ch)
        for t0 in range(0, nt - 5):
            for dr in drifts:
                acc, _ = track(t0, dr, int(round(c0 + d1_ch * t0)), 40)
                if float(acc.max()) > best_h[0]:
                    best_h = (float(acc.max()), t0, float(dr))
        _, t_h, dr_h = best_h
        # lines within 500 kHz bright in those spectra at nearby drifts
        acc_all = np.full(hi - lo, -np.inf, np.float32)
        for dr in drifts[np.abs(drifts - dr_h) <= 1.5]:
            acc_all = np.maximum(acc_all, sum(np.roll(z[t_h + n, lo:hi], -int(round(dr * n))) for n in range(6)) / math.sqrt(6))
        idx = np.where(acc_all >= 9.0)[0]
        groups = []
        for i in idx:
            if not groups or i - groups[-1][-1] > 20:
                groups.append([i])
            else:
                groups[-1].append(i)
        c_hit = int(round(c0 + d1_ch * t_h))
        for g in groups:
            j = lo + g[int(np.argmax(acc_all[g]))]
            if abs(j - c_hit) * abs(foff) * 1e6 <= 2000:
                continue
            # a member shares the hit's best drift over the full range
            snr, dr_m, j_m = -np.inf, 0.0, j
            for dr in drifts:
                acc, a = track(t_h, dr, j, 30)
                k = int(np.argmax(acc))
                if float(acc[k]) > snr:
                    snr, dr_m, j_m = float(acc[k]), float(dr), a + k
            if abs(dr_m - dr_h) > 1.5:
                continue
            # a member is under half its S/N outside the hit's spectra
            rest = [t for t in range(nt) if not t_h <= t < t_h + 6]
            if rest:
                vals = []
                for dc in range(-3, 4):
                    s_ = 0.0
                    for t in rest:
                        c = j_m + dc + int(round(dr_m * (t - t_h)))
                        if 0 <= c < z.shape[1]:
                            s_ += float(z[t, c])
                    vals.append(s_ / math.sqrt(len(rest)))
                if max(vals) >= 0.5 * snr:
                    continue
            r_m = dr_m * foff * 1e6 / tsamp
            f_t0 = f0 + foff * j_m - r_m * t_h * tsamp * 1e-6  # member frequency at spectrum 0
            members.append((f_t0, r_m))
        h["companions"] = [round(m[0], 6) for m in members[1:]]
        seen = []
        for mjd, sec, tgt, url in scans:
            if abs(sec - t_hit) > curved_SESSION_S or any(n in url for n in on_scans):
                continue
            try:
                x, f0 = science.coarse_channel(url, h["k"])
            except Exception as e:
                print("  %s: fetch failed %s" % (tgt, str(e)[:60]))
                continue
            z = np.clip(science.flatten(x), -50, 50)
            hdr = science.header(url, (h["k"],))
            foff, tsamp, nt = hdr["foff"], hdr["tsamp"], z.shape[0]
            for f_m, r_m in members:
                best = (0.0, None, None)
                for t0 in range(0, nt - 5):
                    f_pred = f_m + r_m * ((sec + t0 * tsamp) - t_hit) * 1e-6
                    c = int(round((f_pred - f0) / foff))
                    lo2, hi2 = c - 3000, c + 3000
                    if lo2 < 0 or hi2 >= z.shape[1]:
                        continue
                    seg = z[t0:t0 + 6, lo2:hi2]
                    for dr in drifts:
                        acc = sum(np.roll(seg[n], -int(round(dr * n))) for n in range(6)) / math.sqrt(6)
                        j = int(np.argmax(acc))
                        if acc[j] > best[0]:
                            best = (float(acc[j]), f0 + foff * (lo2 + j), f_pred)
                if best[1] is not None and best[0] >= 7.0 and abs(best[1] - best[2]) * 1e6 < curved_SESSION_HZ:
                    seen.append(dict(target=tgt, sec=sec, snr=round(best[0], 1), member=round(f_m, 6), mhz=best[1], off_hz=round((best[1] - best[2]) * 1e6)))
        h["session_recur"] = seen
        print("  %s %.6f MHz with %d companions (%s): on its drift (%+.3f Hz/s) in %d other pointings of the session: %s" % (
            h["target"], h["mhz"], len(members) - 1, ", ".join("%.6f" % m[0] for m in members[1:]), rate, len(seen),
            "; ".join("%s at %d s, the %.6f MHz member at S/N %.1f, %+d Hz from the extrapolation" % (s["target"], s["sec"], s["member"], s["snr"], s["off_hz"]) for s in seen)))
    json.dump(d, open(path, "w"))


def curved_record(fs_path, run_id, release, ledger="Reply windows, further archives"):
    """Record a cadence judgment: one row per target with its ON-only hits outside occupied allocations."""
    d = json.load(open(os.path.join(curved_OUT, "cadences.json")))
    cads = [json.loads(line) for line in open(fs_path) if line.strip()]
    names = sorted({o["star"]["name"] for o in cads})
    on_scans = set()
    for o in cads:
        roles = [True, False, False] if o.get("single") else [i % 2 == 0 for i in range(len(o["scans"]))]
        on_scans |= {s[0] for s, r in zip(o["scans"], roles) if r}
    searched = sum(1 for p in glob.glob(os.path.join(curved_OUT, "*.bin.json")) if os.path.basename(p).split(".k")[0] in on_scans)
    left = [h for h in d["on_only"] if not h.get("occupied") and not h.get("session_recur")]
    swept = [h for h in d["on_only"] if not h.get("occupied") and h.get("session_recur")]
    core.record_search(
        run_id, "Curved drift tracks, radio", archive="Breakthrough Listen GBT high-resolution spectrograms (open data)",
        release=release, selection="%s: every coarse channel of every ON scan (%d scan-channels); OFF scans where an ON "
                                   "hit recurs, and every channel holding a hit in a single-ON cadence"
                                   % (", ".join(names), searched),
        code=os.path.abspath(__file__), entry="curved",
        params=dict(threshold=10.0, gain=2.0, half_snr=curved_HALF_SNR, max_drift_ch=curved_DMAX,
                                            max_bend_ch=curved_DELTA, recur_hz=5000, veto_hz=2000, veto_straight_snr=6.0),
        n_targets=len(names),
        null_summary="%d cadence-channels judged: %d ON-only hits against %d OFF-only; single-ON cadences %d ON "
                     "scan-channels against %d OFF; ON-only hits outside the occupied allocations %d, %d of them recurring on their drift in other targets' pointings of the session"
                     % (d["cadences"], len(d["on_only"]), len(d["off_only"]), d.get("single_trials", [0, 0])[0],
                        d.get("single_trials", [0, 0])[1], len(left) + len(swept), len(swept)),
        sensitivity="a track bending by up to 5 channels per spectrum at S/N 10 over 16 spectra, each half at S/N 5",
        result=("no curved hit found only in ON scans" if not d["on_only"] else
                "every ON-only hit lies in an occupied allocation" if not swept else
                "every ON-only hit lies in an occupied allocation or recurs in other targets' pointings on its drift") if not left else
        "%d ON-only hits outside the occupied allocations" % len(left), ledger=ledger)
    core.clear_targets(run_id)
    core.record_targets(run_id, [
        dict(target="simbad:%s" % n, stat=float(sum(1 for h in left if h["target"] == n)),
             unit="ON-only curved hits outside occupied allocations",
             vet=("; ".join("%.6f MHz recurs on its drift in %s" % (h["mhz"], ", ".join("%s (%d s, S/N %.1f)" % (r["target"], r["sec"], r["snr"]) for r in h["session_recur"]))
                            for h in swept if h["target"] == n) or None),
             note="%d ON-only hits in occupied allocations" % sum(1 for h in d["on_only"] if h["target"] == n
                                                                    and h.get("occupied"))) for n in names])
    print("recorded")


def curved_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("search", "cadences", "session", "record"))
    p.add_argument("fs", nargs="?")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--run")
    p.add_argument("--release")
    p.add_argument("--ledger", default="Reply windows, further archives")
    a = p.parse_args()
    if a.mode == "search":
        curved_search(a.limit)
    elif a.mode == "cadences":
        curved_cadences(a.fs)
    elif a.mode == "session":
        curved_session(a.fs)
    else:
        curved_record(a.fs, a.run, a.release, a.ledger)


def curved_cli():
    quiet_warnings()
    curved_main()


# --------------------------------------------------------------------------------------------------------------

curved_batch_DOC = r"""Curved-drift search of BL coarse channels fetched by byte range from Berkeley, hits kept and chunks deleted.

    python -I seti.py breakthrough curved_batch TASKS.json [--threads 8]
    python -I seti.py breakthrough curved_batch split TASKS.json N     (the remaining tasks in N parts by file, curved_part<i>.json)"""



def curved_batch_main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.environ.get("CURVED_OUT", core.data("curved"))
    os.makedirs(out_dir, exist_ok=True)
    tasks = json.load(open(a.tasks))
    by = {}
    for t in tasks:
        by.setdefault(t["url"], []).append(int(t["k"]))
    print("%d coarse channels in %d files on %s" % (len(tasks), len(by), dev), flush=True)
    for url, ks in by.items():
        name = url.rstrip("/").split("/")[-1]
        todo = [k for k in ks if not os.path.exists(os.path.join(out_dir, "%s.k%d.bin.json" % (name, k)))]
        if not todo:
            continue
        try:
            science.header(url, todo)
        except Exception as e:
            print("%s: header failed (%s)" % (name, str(e)[:80]), flush=True)
            continue
        for k in todo:
            t0 = time.time()
            try:
                x, f_first = science.coarse_channel(url, k, threads=a.threads)
            except Exception as e:
                print("%s k%d: fetch failed (%s)" % (name, k, str(e)[:80]), flush=True)
                continue
            meta_all = json.load(open(os.path.join(core.CACHE, "idx", name + ".json")))
            meta = dict(name=name, k=k, f_first=f_first, foff=meta_all["foff"], tsamp=meta_all["tsamp"],
                        tstart=meta_all["tstart"], nt=meta_all["nt"])
            z = curved_flatten(x)
            bc, bl, d1, d2 = curved_search_scan(z, dev)
            hits = curved_hits_of(bc, bl, d1, d2, meta)
            strong = np.nonzero(bl >= 6.0)[0]
            strong = strong[np.argsort(-bl[strong])][:20000]
            json.dump(dict(meta=meta, hits=hits, n_straight_ge10=int(np.sum(bl >= 10.0)),
                           strong_mhz=[round(float(f_first + meta["foff"] * i), 7) for i in strong],
                           curved_hist=np.histogram(bc[np.isfinite(bc)], bins=[0, 6, 8, 10, 12, 15, 20, 1e9])[0].tolist()),
                      open(os.path.join(out_dir, "%s.k%d.bin.json" % (name, k)), "w"))
            blob = os.path.join(core.CACHE, "blob", "%s.k%d.bin" % (name, k))
            if os.path.exists(blob):
                os.remove(blob)
            print("%s k%d: %d curved hits, %.1f s" % (name[:60], k, len(hits), time.time() - t0), flush=True)


def curved_batch_split(tasks, n):
    """Unsearched tasks split by file into n parts."""
    out_dir = os.environ.get("CURVED_OUT", core.data("curved"))
    t = json.load(open(tasks))
    done = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
    rest = [x for x in t if "%s.k%d.bin.json" % (x["url"].rstrip("/").split("/")[-1], x["k"]) not in done]
    files = sorted({x["url"] for x in rest})
    for i in range(n):
        mine = set(files[i::n])
        json.dump([x for x in rest if x["url"] in mine], open("curved_part%d.json" % i, "w"))
    print("%d done, %d left in %d parts" % (len(done), len(rest), n))


def curved_batch_cli():
    quiet_warnings()
    if len(sys.argv) > 1 and sys.argv[1] == "split":
        curved_batch_split(sys.argv[2], int(sys.argv[3]))
    else:
        curved_batch_main()


# --------------------------------------------------------------------------------------------------------------

offs_DOC = r"""OFF-scan task list for channels whose ON curved hits recur or stand in single-ON sessions.

    python -I seti.py breakthrough offs CURVED_OUT PLAN.json OUT.json      (PLAN: {"on": [URL ...], "offs": [URL ...]})"""


def offs_name(url):
    return url.rstrip("/").split("/")[-1]


def offs_session(name):
    m = re.search(r"guppi_(\d+)_", name)
    return m.group(1) if m else ""


def offs_main(out_dir, plan_path, dest):
    res = {}
    for p in glob.glob(os.path.join(out_dir, "*.bin.json")):
        d = json.load(open(p))
        res.setdefault(d["meta"]["name"], {})[d["meta"]["k"]] = d
    plan = json.load(open(plan_path))
    ons = {}
    for u in plan["on"]:
        ons.setdefault(offs_session(offs_name(u)), []).append(offs_name(u))
    need = set()
    for sess, names in ons.items():
        names = [n for n in names if n in res]
        ks = set().union(*(set(res[n]) for n in names)) if names else set()
        for k in ks:
            hits = [(n, h["mhz"]) for n in names if k in res[n] for h in res[n][k]["hits"]]
            if len(names) == 1:
                if hits:
                    need.add((sess, k))
                continue
            if any(n2 != n1 and abs(f1 - f2) * 1e6 < 5000 for n1, f1 in hits for n2, f2 in hits):
                need.add((sess, k))
    out = [dict(url=u, k=k) for u in plan["offs"] for sess, k in sorted(need) if offs_session(offs_name(u)) == sess]
    json.dump(out, open(dest, "w"))
    print("%d ON coarse channels searched in %d sessions; %d need their OFF scans (%s); %d tasks"
          % (sum(len(res[n]) for v in ons.values() for n in v if n in res), len(ons), len(need),
             ", ".join("%s: %d" % (s, sum(1 for x in need if x[0] == s)) for s in sorted(ons)), len(out)))


def offs_cli():
    offs_main(*sys.argv[1:4])


# --------------------------------------------------------------------------------------------------------------

followup_DOC = r"""Follow-up of first-ON curved searches: cadence resolution, remaining ON scans, then OFF scans where hits recur.

    python -I seti.py breakthrough followup cadences PLAN.json CADENCES.jsonl
    python -I seti.py breakthrough followup ons CADENCES.jsonl TASKS.json
    python -I seti.py breakthrough followup offs CADENCES.jsonl TASKS.json"""


def followup_results():
    res = {}
    for p in glob.glob(os.path.join(curved_OUT, "*.bin.json")):
        d = json.load(open(p))
        res.setdefault(d["meta"]["name"], {})[d["meta"]["k"]] = [h for h in d["hits"] if not science.occupied(h["mhz"])]
    return res


def followup_cadences(plan_path, dest):
    plan = json.load(open(plan_path))
    n = 0
    with open(dest, "w") as fh:
        for star, bands in plan.items():
            for band, d in bands.items():
                first = d["on"][0]
                found = [c for c in core.bl_cadences(first.rsplit("/", 1)[0] + "/") if c[0][1] == first]
                if not found:
                    print("%s %s: no ABACAD cadence begins with %s" % (star, band, offs_name(first)))
                    continue
                c = found[0]
                fh.write(json.dumps(dict(star=dict(name=star), band=band, session=offs_session(offs_name(first)),
                                         scans=[[offs_name(u), t] for t, u in c], urls=[u for t, u in c])) + "\n")
                n += 1
    print("%d cadences written to %s" % (n, dest))


def followup_tasks(cad_path, dest, stage):
    res = followup_results()
    tasks, chans = [], 0
    for line in open(cad_path):
        c = json.loads(line)
        names, urls = [s[0] for s in c["scans"]], c["urls"]
        if stage == "ons":
            ks = sorted(k for k, hs in res.get(names[0], {}).items() if hs)
            todo = [urls[2], urls[4]]
        else:
            ks = []
            for k in sorted(res.get(names[0], {})):
                found = [(n, h["mhz"]) for n in names[0::2] for h in res.get(n, {}).get(k, [])]
                if any(n1 != n2 and abs(f1 - f2) * 1e6 < 5000 for n1, f1 in found for n2, f2 in found):
                    ks.append(k)
            todo = [urls[1], urls[3], urls[5]]
        chans += len(ks)
        tasks += [dict(url=u, k=k) for u in todo for k in ks]
    json.dump(tasks, open(dest, "w"))
    print("%s: %d cadence-channels need them; %d tasks written to %s" % (stage, chans, len(tasks), dest))


def followup_cli():
    if len(sys.argv) < 4 or sys.argv[1] not in ("cadences", "ons", "offs"):
        print(followup_DOC)
        sys.exit(2)
    if sys.argv[1] == "cadences":
        followup_cadences(sys.argv[2], sys.argv[3])
    else:
        followup_tasks(sys.argv[2], sys.argv[3], sys.argv[1])


# --------------------------------------------------------------------------------------------------------------

combs_DOC = r"""Frequency combs: Fourier power of averaged spectra at each spacing; all ON scans above threshold, no OFF above a third; roles swapped as null.

    python -I seti.py breakthrough combs search [--thr 20]
    python -I seti.py breakthrough combs record"""

combs_HERE = core.RESULTS
combs_NAME = re.compile(r"guppi_(\d+)_(\d+)_(.+)_(\d{4})\.gpuspec\.0000\.h5$")


def combs_cadences():
    """ABACAD cadences among cached scans."""
    names = sorted({os.path.basename(p).split(".k")[0] for p in glob.glob(os.path.join(core.CACHE, "blob", "*.bin"))})
    scans = []
    for n in names:
        m = combs_NAME.search(n)
        if m:
            scans.append(dict(name=n, mjd=int(m.group(1)), sec=int(m.group(2)), target=m.group(3), scan=int(m.group(4))))
    scans.sort(key=lambda s: (s["mjd"], s["sec"]))
    out = []
    for i in range(len(scans) - 5):
        six = scans[i:i + 6]
        if any(six[j + 1]["scan"] != six[j]["scan"] + 1 or six[j + 1]["mjd"] != six[j]["mjd"] for j in range(5)):
            continue
        t = [s["target"] for s in six]
        if t[0] == t[2] == t[4] and len({t[1], t[3], t[5]} - {t[0]}) == 3:
            out.append(six)
    return out


def combs_comb_power(spectra, dev):
    """Normalised Fourier power of each spectrum against spacing index."""
    import torch
    x = torch.as_tensor(np.asarray(spectra, np.float32), device=dev)
    x = x - x.mean(dim=1, keepdim=True)
    p = torch.fft.rfft(x, dim=1).abs() ** 2
    p = p[:, 1:].cpu().numpy()
    q = np.arange(1, p.shape[1] + 1)
    lq = np.log10(q)
    edges = np.arange(0, lq[-1] + 0.02, 0.02)
    k = np.digitize(lq, edges)
    out = np.empty_like(p)
    for b in np.unique(k):
        sel = k == b
        med = np.median(p[:, sel], axis=1, keepdims=True)
        out[:, sel] = p[:, sel] / np.maximum(med, 1e-30) * math.log(2)
    return out


def combs_search(thr):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cads = combs_cadences()
    print("%d cadences in the cache" % len(cads), flush=True)
    rows, null_hits, tests = [], 0, 0
    for six in cads:
        ks = None
        for s in six:
            got = {int(re.search(r"\.k(\d+)\.bin$", p).group(1)) for p in glob.glob(os.path.join(
                core.CACHE, "blob", s["name"] + ".k*.bin"))}
            ks = got if ks is None else ks & got
        for k in sorted(ks or ()):
            specs = []
            for s in six:
                got = science.cached_channel(s["name"], k)
                if got is None:
                    break
                z = science.flatten(got[0])
                specs.append(z.mean(axis=0))
            if len(specs) < 6:
                continue
            foff = abs(float(got[1]["foff"])) * 1e6
            p = combs_comb_power(specs, dev)
            on, off = p[[0, 2, 4]], p[[1, 3, 5]]
            tests += p.shape[1]
            hit = np.nonzero((on.min(axis=0) > thr) & (off.max(axis=0) < thr / 3))[0]
            # null: the second OFF target as ON, the ON scans as OFF
            alt_on, alt_off = p[[1, 3, 5]], p[[0, 2, 4]]
            null_hits += int(np.sum((alt_on.min(axis=0) > thr) & (alt_off.max(axis=0) < thr / 3)))
            n = len(specs[0])
            for j in hit:
                q = j + 1
                rows.append(dict(target=six[0]["target"], mjd=six[0]["mjd"], k=k, q=int(q),
                                 spacing_hz=float(n / q * foff), on=[float(x) for x in on[:, j]],
                                 off=[float(x) for x in off[:, j]], scans=[s["name"] for s in six]))
            print("%s k%d: %d spacings on the target alone" % (six[0]["target"], k, len(hit)), flush=True)
    json.dump(dict(tests=tests, null_hits=null_hits, hits=rows, thr=thr), open(os.path.join(combs_HERE, "radio_combs.json"), "w"),
              indent=1)
    print("%d spacing tests; %d on the target alone against %d with the roles exchanged" % (tests, len(rows), null_hits))
    for r in sorted(rows, key=lambda r: -min(r["on"]))[:20]:
        print("  %s MJD %d k%d spacing %.2f Hz: ON %s OFF %s" % (r["target"], r["mjd"], r["k"], r["spacing_hz"],
                                                               [round(x, 1) for x in r["on"]],
                                                               [round(x, 1) for x in r["off"]]))


def combs_record():
    """Record the run: one row per cadence and coarse channel."""
    r = json.load(open(os.path.join(combs_HERE, "radio_combs.json")))
    cads = combs_cadences()
    core.record_search(
        "bl_frequency_combs", "Frequency combs on the target alone, Breakthrough Listen",
        archive="Breakthrough Listen GBT L band (open data cache)", release="AGBT16A-17B sessions",
        selection="the cached H I and OH coarse channels of %d ABACAD cadences" % len(cads), code="seti.py breakthrough combs",
        params=dict(threshold=r["thr"], off_ceiling=r["thr"] / 3, normalisation="median per 1/50 dex of spacing"),
        n_targets=len(cads),
        null_summary="%d spacing tests: %d on the target alone against %d with an OFF target standing in"
                     % (r["tests"], len(r["hits"]), r["null_hits"]),
        sensitivity="teeth summing coherently to 4.5 sigma times the root of the channel count in every ON scan, "
                    "e.g. 1e5 teeth at 0.05 sigma each in the scan-averaged spectrum",
        result="no comb on a target alone", ledger="Other radio morphologies")
    core.clear_targets("bl_frequency_combs")
    core.record_targets("bl_frequency_combs", [
        dict(target="bl:%s" % six[0]["name"], note="cadence on %s, MJD %d" % (six[0]["target"], six[0]["mjd"]))
        for six in cads])
    print("recorded")


def combs_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("search", "record"))
    p.add_argument("--thr", type=float, default=20.0)
    a = p.parse_args()
    combs_search(a.thr) if a.mode == "search" else combs_record()


def combs_cli():
    quiet_warnings()
    combs_main()


# --------------------------------------------------------------------------------------------------------------

hops_DOC = r"""Frequency hopping: isolated single-spectrum bright pixels counted per cadence, ON above OFF with reused frequencies; roles swapped as null.

    python -I seti.py breakthrough hops search
    python -I seti.py breakthrough hops record"""

hops_HERE = core.RESULTS
hops_Z_HOP, hops_Z_NEAR, hops_REACH, hops_Z_VETO = 8.0, 5.0, 30, 7.0


def hops_hops(z):
    """(time step, channel) of isolated bright pixels, excluding each scan's first and last spectra."""
    from scipy.ndimage import maximum_filter1d
    nt = z.shape[0]
    near = maximum_filter1d(z, size=2 * hops_REACH + 1, axis=1)
    out = []
    t, c = np.nonzero(z[1:nt - 1] > hops_Z_HOP)
    for ti, ci in zip(t + 1, c):
        if near[ti - 1, ci] < hops_Z_NEAR and near[ti + 1, ci] < hops_Z_NEAR:
            out.append((int(ti), int(ci)))
    return out


def hops_busy(z, c):
    """Brightest pixel within REACH channels of c in any spectrum."""
    return float(z[:, max(c - hops_REACH, 0):c + hops_REACH + 1].max())


def hops_search():
    cads = combs_cadences()
    print("%d cadences in the cache" % len(cads), flush=True)
    rows, n_cc = [], 0
    stats = dict(on_above=0, off_above=0, on_reuse=0, off_reuse=0)
    for six in cads:
        ks = None
        for s in six:
            got = {int(re.search(r"\.k(\d+)\.bin$", p).group(1)) for p in glob.glob(os.path.join(
                core.CACHE, "blob", s["name"] + ".k*.bin"))}
            ks = got if ks is None else ks & got
        for k in sorted(ks or ()):
            hs, zs = [], []
            for s in six:
                got = science.cached_channel(s["name"], k)
                if got is None:
                    break
                z = science.flatten(got[0])
                zs.append(z)
                hs.append(hops_hops(z))
            if len(hs) < 6:
                continue
            n_cc += 1
            counts = [len(h) for h in hs]
            for role, on_i, off_i in (("on", (0, 2, 4), (1, 3, 5)), ("off", (1, 3, 5), (0, 2, 4))):
                if min(counts[i] for i in on_i) > max(counts[i] for i in off_i):
                    stats[role + "_above"] += 1
                # hop frequency reused within 2 channels and absent from the other role
                other = np.array(sorted(c for i in off_i for _, c in hs[i]))
                reuse = []
                for a in on_i:
                    for _, c in hs[a]:
                        if any(any(abs(c - c2) <= 2 for _, c2 in hs[b]) for b in on_i if b != a):
                            j = np.searchsorted(other, c)
                            near = other[max(j - 1, 0):j + 1]
                            # the other role must be quiet there
                            quiet = max(hops_busy(zs[i], c) for i in off_i) < hops_Z_VETO
                            if (not len(near) or np.min(np.abs(near - c)) > 2) and quiet:
                                reuse.append(c)
                if reuse:
                    stats[role + "_reuse"] += 1
                    if role == "on":
                        foff = abs(float(got[1]["foff"]))
                        rows.append(dict(target=six[0]["target"], mjd=six[0]["mjd"], k=k, counts=counts,
                                         channels=sorted(set(reuse)),
                                         mhz=[float(got[1]["fch1"]) + float(got[1]["foff"]) * (k * science.COARSE + c)
                                              for c in sorted(set(reuse))], foff_hz=foff * 1e6))
            print("%s k%d: isolated hops per scan %s" % (six[0]["target"], k, counts), flush=True)
    json.dump(dict(cadence_channels=n_cc, stats=stats, reuse=rows), open(os.path.join(hops_HERE, "radio_hops.json"), "w"),
              indent=1)
    print("%d cadence-channels: every ON scan above every OFF scan in %d (roles exchanged %d); a hop frequency reused in "
          "two ON scans and absent from the OFF scans in %d (roles exchanged %d)"
          % (n_cc, stats["on_above"], stats["off_above"], stats["on_reuse"], stats["off_reuse"]))
    for r in rows[:20]:
        print("  %s MJD %d k%d: hops %s, reused at %s MHz" % (r["target"], r["mjd"], r["k"], r["counts"],
                                                           ", ".join("%.6f" % m for m in r["mhz"][:5])))


def hops_record():
    """Record the run: one row per cadence."""
    r = json.load(open(os.path.join(hops_HERE, "radio_hops.json")))
    cads = combs_cadences()
    s = r["stats"]
    core.record_search(
        "bl_frequency_hops", "Frequency hopping on the target alone, Breakthrough Listen",
        archive="Breakthrough Listen GBT L band (open data cache)", release="AGBT16A-17B sessions",
        selection="the cached H I and OH coarse channels of %d ABACAD cadences" % len(cads), code="seti.py breakthrough hops",
        params=dict(z_hop=hops_Z_HOP, z_near=hops_Z_NEAR, reach_channels=hops_REACH, reuse_channels=2),
        n_targets=len(cads),
        null_summary="%d cadence-channels: every ON scan above every OFF in %d (roles exchanged %d); a hop frequency "
                     "reused in two ON scans and absent from the OFF scans in %d (roles exchanged %d)"
                     % (r["cadence_channels"], s["on_above"], s["off_above"], s["on_reuse"], s["off_reuse"]),
        sensitivity="single-spectrum bursts at 8 robust deviations in 2.79 Hz channels",
        result="no hopping on a target alone beyond the role-exchanged count" if s["on_reuse"] <= s["off_reuse"] and
        s["on_above"] <= s["off_above"] else "see the null summary", ledger="Other radio morphologies")
    core.clear_targets("bl_frequency_hops")
    core.record_targets("bl_frequency_hops", [
        dict(target="bl:%s" % six[0]["name"], note="cadence on %s, MJD %d" % (six[0]["target"], six[0]["mjd"]))
        for six in cads])
    print("recorded")


def hops_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("search", "record"))
    a = p.parse_args()
    hops_search() if a.mode == "search" else hops_record()


def hops_cli():
    quiet_warnings()
    hops_main()


# --------------------------------------------------------------------------------------------------------------

pulses_DOC = r"""Broadband pulses at DM -30 to +3 in 349 us BL products (ON against OFF), and zero-DM folds at periods keyed to J0437-4715, Crab, Vela and B1937+21 as seen from each star.

    python -I seti.py breakthrough pulses tasks CADENCES.jsonl TASKS.json
    python -I seti.py breakthrough pulses search TASKS.json [--workers 2]
    python -I seti.py breakthrough pulses summary
    python -I seti.py breakthrough pulses record [--control DIR]      (OFF scans folded the same way as the null)"""

pulses_OUT = os.environ.get("PULSES_OUT", core.data("bl_pulses"))
pulses_SNR = 7.0
pulses_NSUB = 64
pulses_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
pulses_KDM = 4.148808e-3  # s GHz^2 per pc cm^-3
# name, RA, Dec (deg), F0 (Hz), F1, F2, PEPOCH (MJD), distance (pc), proper motion (mas/yr)
pulses_PSRS = (("J0437-4715", 69.3168725, -47.2527840, 173.6879456649439, -1.728367e-15, -6.1e-29, 55486.0, 157.0, 121.4, -71.5),
               ("Crab", 83.6330565, 22.0144980, 29.9469230, -3.77535e-10, 1.1147e-20, 48442.5, 2000.0, -11.5, 2.3),
               ("Vela", 128.8358812, -45.1763542, 11.1946499395, -1.5666e-11, 1.028e-21, 51559.319, 280.0, -49.7, 29.9),
               ("B1937+21", 294.9106716, 21.5830943, 641.928222127829, -4.330987e-14, 0.0, 55656.0, 3500.0, 0.07, -0.42))
pulses_C = 299792.458


def pulses_unit(ra, dec):
    a, d = math.radians(ra), math.radians(dec)
    return np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)])


def pulses_vel(ra, dec, pmra, pmdec, rv, dist_pc):
    """Barycentric space velocity (km/s) from astrometry, radial velocity and distance."""
    u = pulses_unit(ra, dec)
    a, d = math.radians(ra), math.radians(dec)
    east = np.array([-math.sin(a), math.cos(a), 0.0])
    north = np.array([-math.sin(d) * math.cos(a), -math.sin(d) * math.sin(a), math.cos(d)])
    k = 4.74047 * dist_pc / 1000.0
    return rv * u + k * (pmra * east + pmdec * north)


def pulses_periods(star, mjd, berv):
    """Topocentric period of a sender at the star keyed on each pulsar, and the pulsar's own."""
    out = {}
    uS = pulses_unit(star["ra"], star["dec"])
    vS = pulses_vel(star["ra"], star["dec"], star["pmra"], star["pmdec"], star["rv"], star["d"])
    for name, ra, dec, f0, f1, f2, pep, dpsr, pmra, pmdec in pulses_PSRS:
        uP = pulses_unit(ra, dec)
        cos_t = float(np.dot(uS, uP))
        lag_days = star["d"] * 3.0857e13 / pulses_C / 86400.0 * (1.0 - cos_t)
        def f_at(m):
            dt = (m - pep) * 86400.0
            return f0 + f1 * dt + 0.5 * f2 * dt * dt
        p_bary_now = 1.0 / f_at(mjd)
        p_bary_then = 1.0 / f_at(mjd - lag_days)
        # star-to-pulsar direction and the pulsar's transverse velocity
        xS = uS * star["d"]
        xP = uP * dpsr
        uSP = (xP - xS) / np.linalg.norm(xP - xS)
        vP = pulses_vel(ra, dec, pmra, pmdec, 0.0, dpsr)
        p_sender = p_bary_then * (1.0 + (float(np.dot(vP, uSP - uP)) - float(np.dot(vS, uSP))) / pulses_C)
        p_bary_beacon = p_sender * (1.0 + float(np.dot(vS, uS)) / pulses_C)
        out[name] = dict(beacon=p_bary_beacon * (1.0 - berv / pulses_C), psr_topo=p_bary_now,
                         lag_years=lag_days / 365.25, cos_theta=cos_t)
    return out


def pulses_load(path, nsub=pulses_NSUB):
    """Normalised subband time series of a 349 us product with outlier channels removed: series, subband centres (GHz), header."""
    import h5py
    import hdf5plugin  # noqa: F401
    f = h5py.File(path, "r")
    d = f["data"]
    hdr = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in d.attrs.items() if k != "DIMENSION_LABELS"}
    nt, nch = d.shape[0], d.shape[2]
    freqs = (hdr["fch1"] + hdr["foff"] * np.arange(nch)) / 1000.0
    # channel statistics from 16 blocks
    samp = np.concatenate([d[s:s + 2048, 0, :] for s in np.linspace(0, nt - 2048, 16).astype(int)]).astype(np.float32)
    med = np.median(samp, axis=0)
    mad = 1.4826 * np.median(np.abs(samp - med), axis=0)
    ok = mad > 0
    lm = np.log(np.where(ok, mad, 1.0))
    centre = np.median(lm[ok])
    spread = 1.4826 * np.median(np.abs(lm[ok] - centre))
    ok &= np.abs(lm - centre) < 4 * max(spread, 0.05)
    kurt = np.mean(((samp - med) / np.where(ok, mad, 1.0)) ** 4, axis=0)
    ok &= kurt < np.median(kurt[ok]) + 6 * 1.4826 * np.median(np.abs(kurt[ok] - np.median(kurt[ok])))
    edges = np.linspace(0, nch, nsub + 1).astype(int)
    S = np.zeros((nsub, nt), np.float32)
    fsub = np.array([freqs[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    nok = np.array([ok[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    for s in range(0, nt, 32768):
        blk = d[s:s + 32768, 0, :].astype(np.float32)
        z = (blk - med) / np.where(ok, mad, 1.0)
        z[:, ~ok] = 0.0
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if nok[i]:
                S[i, s:s + len(blk)] = z[:, a:b].sum(axis=1) / math.sqrt(nok[i])
    f.close()
    return S, fsub, nok, hdr, float(ok.mean())


def pulses_search_scan(S, fsub, nok, tsamp):
    """Single pulses over DM -30 to +3: strongest per 0.5 s and counts above threshold per DM."""
    from scipy.ndimage import uniform_filter1d
    use = nok > 0
    ftop = fsub[use].max()
    kdm = pulses_KDM * (fsub ** -2 - ftop ** -2) / tsamp  # samples per unit DM per subband
    step = 1.0 / max(np.max(np.abs(kdm[use])), 1e-9)
    dms = np.arange(-30.0, 3.0 + 1e-9, step)
    nt = S.shape[1]
    w_all = math.sqrt(float(use.sum()))
    best = {}
    counts = []
    iu = np.nonzero(use)[0]
    Ssh = np.empty((len(iu), nt), np.float32)
    for dm in dms:
        sh = np.round(dm * kdm).astype(int)
        for r, i in enumerate(iu):
            Ssh[r] = np.roll(S[i], -sh[i])
        ts = Ssh.sum(axis=0) / w_all
        base = uniform_filter1d(ts, 2861, mode="nearest")  # about 1 s
        ts = ts - base
        sd = 1.4826 * np.median(np.abs(ts[::7] - np.median(ts[::7])))
        n_above = 0
        for w in pulses_WIDTHS:
            sm = uniform_filter1d(ts, w, mode="nearest") * math.sqrt(w) / max(sd, 1e-6)
            above = sm >= pulses_SNR
            n_above += int(above.sum())
            # local maxima above threshold, one per 0.5 s
            idx = np.nonzero(above & (sm >= np.roll(sm, 1)) & (sm > np.roll(sm, -1)))[0]
            for j in idx:
                key = int(j * tsamp / 0.5)
                if key in best and sm[j] <= best[key]["snr"]:
                    continue
                a, b = max(j - w // 2, 0), min(j + w - w // 2, nt)
                sub = Ssh[:, a:b].sum(axis=1) / math.sqrt(max(b - a, 1))
                best[key] = dict(t=float(j * tsamp), dm=float(dm), width=int(w), snr=float(sm[j]),
                                 frac_sub=float(np.mean(sub > 1.0)))
        counts.append(n_above)
    cands = sorted(best.values(), key=lambda c: -c["snr"])[:200]
    # DM sensitivity: S/N at DM 0 against DMs smearing it over four widths; steps score flat
    kmax = max(np.max(np.abs(kdm[use])), 1e-9)
    for c in cands:
        j, w = int(round(c["t"] / tsamp)), c["width"]
        a, b = max(j - 3000, 0), min(j + 3000, nt)
        def snr_at(dm):
            sh = np.round(dm * kdm).astype(int)
            seg = np.zeros(b - a, np.float32)
            for i in iu:
                seg += np.roll(S[i], -sh[i])[a:b]
            seg /= w_all
            seg -= np.median(seg)
            sd = 1.4826 * np.median(np.abs(seg))
            sm = uniform_filter1d(seg, w, mode="nearest") * math.sqrt(w) / max(sd, 1e-6)
            k = j - a
            return float(sm[max(k - w, 0):k + w + 1].max())
        delta = 4.0 * w / kmax
        c["snr_dm0"] = snr_at(0.0)
        c["peaked"] = c["snr"] / max(snr_at(c["dm"] - delta), snr_at(c["dm"] + delta), 1e-6)
    return dms, np.array(counts), cands


def pulses_fold(ts, tsamp, period, nbins):
    ph = (np.arange(len(ts)) * tsamp / period) % 1.0
    idx = np.minimum((ph * nbins).astype(int), nbins - 1)
    s = np.bincount(idx, weights=ts, minlength=nbins)
    n = np.bincount(idx, minlength=nbins)
    prof = s / np.maximum(n, 1)
    sd = np.std(ts)
    return float(np.sum((prof - prof.mean()) ** 2 / (sd ** 2 / np.maximum(n, 1)))), prof


def pulses_scan(task):
    """Download, search and delete one scan's 349 us product; result in OUT/<name>.json."""
    name = task["url"].rstrip("/").split("/")[-1]
    out_p = os.path.join(pulses_OUT, name + ".json")
    if os.path.exists(out_p):
        return 1
    tmp = os.path.join(pulses_OUT, "tmp", name)
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    try:
        core.download(task["url"], tmp)
        S, fsub, nok, hdr, frac_ok = pulses_load(tmp)
    except Exception as e:
        print("%s: %s" % (name[:60], str(e)[:100]), flush=True)
        return 0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    tsamp = float(hdr["tsamp"])
    dms, counts, cands = pulses_search_scan(S, fsub, nok, tsamp)
    use = nok > 0
    ts0 = S[use].sum(axis=0) / math.sqrt(use.sum())
    mjd = float(hdr["tstart"]) + 0.5 * len(ts0) * tsamp / 86400.0
    folds = {}
    star = task.get("star")
    if star:
        berv = science.barycentric_correction(star["ra"], star["dec"], mjd, "GBT")
        for psr, pr in pulses_periods(star, mjd, berv).items():
            nb = int(min(64, max(4, pr["beacon"] / tsamp)))
            span = len(ts0) * tsamp
            step = pr["beacon"] / (span / pr["beacon"]) / (4 * nb)  # a quarter bin of drift over the scan
            trials = [pr["beacon"] + k * step for k in range(-3, 4)]
            obs = max(pulses_fold(ts0, tsamp, p, nb)[0] for p in trials)
            null = [max(pulses_fold(ts0, tsamp, pr["beacon"] * (1 + off), nb)[0] for _ in [0])
                    for off in np.linspace(2e-4, 2e-3, 60) * np.where(np.arange(60) % 2, 1, -1)]
            leak = pulses_fold(ts0, tsamp, pr["psr_topo"] * (1 - berv / pulses_C), nb)[0]
            folds[psr] = dict(period=pr["beacon"], lag_years=pr["lag_years"], chi2=obs, nbins=nb,
                              p=float(np.mean(np.array(null) >= obs)), null_median=float(np.median(null)), leak_chi2=leak)
    res = dict(name=name, target=task.get("target"), on=bool(task.get("on")), session=task.get("session"), mjd=mjd,
               tsamp=tsamp, frac_channels=frac_ok, dms=[float(dms[0]), float(dms[-1]), len(dms)],
               counts=counts.tolist(), cands=cands, folds=folds)
    os.makedirs(pulses_OUT, exist_ok=True)
    json.dump(res, open(out_p, "w"))
    top = cands[0] if cands else None
    print("%s: %s; best pulse %s; folds %s" % (name[:50], "ON" if task.get("on") else "OFF",
                                               ("DM %.2f SNR %.1f w %d frac %.2f" % (top["dm"], top["snr"], top["width"], top["frac_sub"])) if top else "none",
                                               ", ".join("%s p %.2f" % (k, v["p"]) for k, v in folds.items())), flush=True)
    return 1


def pulses_tasks(cad_path, dest):
    """Every scan of the cadences with the target's astrometry for the folds."""
    from astroquery.simbad import Simbad
    s = Simbad()
    s.add_votable_fields("pmra", "pmdec", "plx_value", "rvz_radvel")
    stars, tasks = {}, []
    for line in open(cad_path):
        c = json.loads(line)
        name = c["star"]["name"]
        if name not in stars:
            q = re.sub(r"^GJ(\d)", r"GJ \1", re.sub(r"^HIP(\d)", r"HIP \1", name))
            try:
                r = s.query_object(q)
                stars[name] = dict(ra=float(r["ra"][0]), dec=float(r["dec"][0]), pmra=float(r["pmra"][0] or 0.0),
                                   pmdec=float(r["pmdec"][0] or 0.0), d=1000.0 / float(r["plx_value"][0]),
                                   rv=float(r["rvz_radvel"][0]) if r["rvz_radvel"][0] is not np.ma.masked else 0.0)
            except Exception as e:
                print("%s: not resolved (%s)" % (name, str(e)[:60]))
                stars[name] = None
        for k, ((scan, tgt), url) in enumerate(zip(c["scans"], c["urls"])):
            u = url.replace(".gpuspec.0000.h5", ".gpuspec.8.0001.h5")
            # OFF scans carry the star for the far-field control folds
            tasks.append(dict(url=u, target=tgt, on=k % 2 == 0, session=c["session"], star=stars[name]))
    json.dump(tasks, open(dest, "w"))
    print("%d scans in %d cadences of %d stars" % (len(tasks), len(tasks) // 6, len(stars)))


def pulses_search_main(tasks_path, workers):
    tasks = json.load(open(tasks_path))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        n = sum(ex.map(pulses_scan, tasks))
    print("%d of %d scans searched" % (n, len(tasks)))


def pulses_summary():
    R = [json.load(open(p)) for p in glob.glob(os.path.join(pulses_OUT, "*.h5.json"))]
    on = [r for r in R if r["on"]]
    off = [r for r in R if not r["on"]]
    print("%d scans (%d ON, %d OFF)" % (len(R), len(on), len(off)))
    for lab, sel in (("DM < -1", lambda c: c["dm"] < -1), ("-1 <= DM <= 1", lambda c: -1 <= c["dm"] <= 1), ("DM > 1", lambda c: c["dm"] > 1)):
        for thr in (7, 9, 12):
            def keep(c, thr=thr):
                return sel(c) and c["snr"] >= thr and c["frac_sub"] >= 0.5 and c.get("peaked", 2.0) >= 1.5
            n_on = sum(1 for r in on for c in r["cands"] if keep(c))
            n_off = sum(1 for r in off for c in r["cands"] if keep(c))
            print("  %-14s SNR >= %2d, broadband and DM-peaked: ON %4d (%.2f per scan)  OFF %4d (%.2f per scan)"
                  % (lab, thr, n_on, n_on / max(len(on), 1), n_off, n_off / max(len(off), 1)))
    fl = [(r["name"], k, v) for r in on for k, v in r["folds"].items()]
    fl.sort(key=lambda x: x[2]["p"])
    fo = [(r["name"], k, v) for r in off for k, v in r.get("folds", {}).items()]
    print("pulsar-locked folds: ON %d, p below 0.05: %d (%.1f expected); OFF %d, p below 0.05: %d"
          % (len(fl), sum(1 for x in fl if x[2]["p"] < 0.05), 0.05 * len(fl), len(fo), sum(1 for x in fo if x[2]["p"] < 0.05)))
    for n, k, v in fl[:10]:
        print("  %s %s: chi2 %.1f (null median %.1f) p %.3f, %d bins, lag %.1f yr; pulsar's own period chi2 %.1f"
              % (n[:50], k, v["chi2"], v["null_median"], v["p"], v["nbins"], v["lag_years"], v["leak_chi2"]))


def pulses_record(control=None):
    """Record pulses by DM class (ON against OFF) and pulsar-locked folds against chance; one row per ON target."""
    R = [json.load(open(p)) for p in glob.glob(os.path.join(pulses_OUT, "*.h5.json"))]
    on = [r for r in R if r["on"]]
    off = [r for r in R if not r["on"]]

    def keep(c):
        return c["snr"] >= 7 and c["frac_sub"] >= 0.5 and c.get("peaked", 2.0) >= 1.5

    def rate(rs, lo, hi):
        return sum(1 for r in rs for c in r["cands"] if keep(c) and lo <= c["dm"] < hi) / max(len(rs), 1)
    classes = (("negative", -99.0, -1.0), ("zero", -1.0, 1.0), ("positive", 1.0, 99.0))
    rates = {k: (rate(on, lo, hi), rate(off, lo, hi)) for k, lo, hi in classes}
    fl = [v for r in on for v in r["folds"].values()]
    low = sum(1 for v in fl if v["p"] < 0.05)
    ctl = ""
    if control and os.path.isdir(control):
        C = [json.load(open(p)) for p in glob.glob(os.path.join(control, "*.h5.json"))]
        cf = [v for r in C if not r["on"] for v in r.get("folds", {}).values()]
        ctl = "; the OFF scans of a control cadence folded the same way give %d of %d below p 0.05" % (
            sum(1 for v in cf if v["p"] < 0.05), len(cf))
    targets = sorted({r["target"] for r in on})
    core.record_search(
        "bl_pulses", "Negatively dispersed and pulsar-locked pulses",
        archive="Breakthrough Listen GBT 349 us products (gpuspec.8.0001.h5, open data)",
        release="%d scans (%d ON, %d OFF) of %d targets" % (len(R), len(on), len(off), len(targets)),
        selection="ABACAD cadences of the nearest stars with high-time-resolution products in the open data",
        code=os.path.abspath(__file__), entry="pulses", n_targets=len(targets),
        params=dict(snr=pulses_SNR, nsub=pulses_NSUB, widths=list(pulses_WIDTHS), dm=[-30, 3], frac_sub=0.5, peaked=1.5),
        null_summary="OFF scans of the same cadences: broadband DM-peaked pulses per scan at DM < -1: %.1f, -1 to 1: %.1f, "
                     "> 1: %.1f%s" % (rates["negative"][1], rates["zero"][1], rates["positive"][1], ctl),
        sensitivity="a 7 sigma pulse of 0.35 to 22 ms in at least half of 64 subbands, DM -30 to +3 pc cm^-3",
        ledger="Pulses at zero and negative dispersion measure",
        result="ON scans: broadband DM-peaked pulses per scan at DM < -1: %.1f, -1 to 1: %.1f, > 1: %.1f, against %.1f, %.1f "
               "and %.1f in the OFF scans; pulsar-locked folds of the ON scans: %d of %d below p 0.05 against %.1f by chance%s"
               % (rates["negative"][0], rates["zero"][0], rates["positive"][0], rates["negative"][1], rates["zero"][1],
                  rates["positive"][1], low, len(fl), 0.05 * len(fl), ctl))
    rows = []
    for t in targets:
        rs = [r for r in on if r["target"] == t]
        neg = [c for r in rs for c in r["cands"] if keep(c) and c["dm"] < -1]
        pmin = min(((v["p"], k) for r in rs for k, v in r["folds"].items()), default=None)
        rows.append(dict(target="bl:" + t, stat=max((c["snr"] for c in neg), default=None), p=pmin[0] if pmin else None,
                         unit="S/N of the strongest broadband pulse at DM < -1",
                         vet="OFF scans carry such pulses at %.1f per scan" % rates["negative"][1],
                         note="%d ON scans; %d negative-DM pulses; lowest fold p %s (%s)"
                              % (len(rs), len(neg), "%.3f" % pmin[0] if pmin else "-", pmin[1] if pmin else "-")))
    core.clear_targets("bl_pulses")
    core.record_targets("bl_pulses", rows)
    print("recorded %d scans, %d targets" % (len(R), len(targets)))


def pulses_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("tasks", "search", "summary", "record"))
    p.add_argument("args", nargs="*")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--control", default=None, help="a directory of OFF scans folded the same way, the folds' null")
    a = p.parse_args()
    if a.mode == "tasks":
        pulses_tasks(*a.args)
    elif a.mode == "search":
        pulses_search_main(a.args[0], a.workers)
    elif a.mode == "record":
        pulses_record(a.control)
    else:
        pulses_summary()


# --------------------------------------------------------------------------------------------------------------

highdrift_DOC = r"""High-drift narrowband search of 2.86 kHz BL products on the GPU (+-300 Hz/s, with curvature); hits in all ON scans, OFF below 5 sigma.

    python -I seti.py breakthrough highdrift sessions               (every session's mid-resolution scans, ROOT/sessions.json)
    python -I seti.py breakthrough highdrift plan                   (ABACAD cadences, ROOT/plan.json; new cadences after the existing ones)
    python -I seti.py breakthrough highdrift search [--part k/n] [--streams 4] [--limit N] [--redo]   (--redo: results before the fitted bandpass, and failed fetches)
    python -I seti.py breakthrough highdrift inject [--scans 1000] [--per 40]   (completeness on synthetic scans, ROOT/injections.jsonl)
    python -I seti.py breakthrough highdrift record"""

highdrift_ROOT = os.environ.get("HIGHDRIFT_ROOT", core.data("bl_midres"))
highdrift_PIPE = "https://bldata.berkeley.edu/pipeline/"
highdrift_RX = re.compile(r'href="(spliced_blc[0-9]+_guppi_(\d+)_(\d+)_([^"]+?)_(\d{4})\.gpuspec\.0002\.h5)"')
highdrift_DRIFT = 300.0  # Hz/s
highdrift_CURV = 0.3  # Hz/s^2; an Earth-like rotating planet gives 0.16 Hz/s^2 at L band
highdrift_CURV_STEP = 0.03   # Hz/s^2
highdrift_SNR = 10.0
highdrift_OFF_SNR = 5.0
highdrift_BLOCK = 4096
highdrift_COARSE = 1024  # channels per coarse channel
highdrift_TOL = 3  # channel tolerance across the cadence
highdrift_SEFD = 10.0  # Jy, GBT at L and S band
highdrift_NORM = 2  # bandpass from each coarse channel's level and the folded coarse-channel response, noise per position; results without it used a smoothed level alone


def highdrift_sessions():
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(highdrift_ROOT, exist_ok=True)
    page = core.get(highdrift_PIPE, timeout=120).decode(errors="replace")
    sessions = sorted(set(re.findall(r'href="(AGBT[0-9]{2}[AB]_[0-9]+_[0-9]+)/?"', page)))

    def one(s):
        d = core.get(highdrift_PIPE + s + "/holding/", timeout=120, soft=True)
        if d is None:
            return s, None
        return s, sorted({(int(m[1]), int(m[2]), m[3], m[0]) for m in highdrift_RX.findall(d.decode(errors="replace"))})
    out = {}
    with ThreadPoolExecutor(6) as ex:
        for s, files in ex.map(one, sessions):
            out[s] = files
    json.dump(out, open(os.path.join(highdrift_ROOT, "sessions.json"), "w"))
    print("%d sessions, %d with mid-resolution scans, %d scans, %d failed listings"
          % (len(out), sum(1 for v in out.values() if v), sum(len(v) for v in out.values() if v),
             sum(1 for v in out.values() if v is None)))


def highdrift_plan():
    """ABACAD cadences: six scans within 0.1 d, target at 0, 2 and 4. An existing plan keeps its order and new cadences follow it, so each part's share of the earlier plan is unchanged."""
    d = json.load(open(os.path.join(highdrift_ROOT, "sessions.json")))
    path = os.path.join(highdrift_ROOT, "plan.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    plan = []
    for sess, files in sorted(d.items()):
        s = sorted(files or [])
        i = 0
        while i <= len(s) - 6:
            w = s[i:i + 6]
            a = w[0][2]
            t = [f[0] + f[1] / 86400.0 for f in w]
            if (w[2][2] == a and w[4][2] == a and len({w[1][2], w[3][2], w[5][2], a}) == 4 and t[5] - t[0] < 0.1
                    and not a.lower().startswith("diag")):
                plan.append(dict(session=sess, target=a, scans=[dict(mjd=t[j], target=f[2], file=f[3], on=(j % 2 == 0))
                                                                 for j, f in enumerate(w)]))
                i += 6
            else:
                i += 1
    seen = {(c["session"], c["scans"][0]["file"]) for c in old}
    new = [c for c in plan if (c["session"], c["scans"][0]["file"]) not in seen]
    plan = old + new
    json.dump(plan, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    print("%d cadences of %d targets in %d sessions, %d of them new" % (len(plan), len({c["target"] for c in plan}),
                                                                        len({c["session"] for c in plan}), len(new)))


def highdrift_fetch(url, path, streams=4):
    """Download one file by parallel byte ranges."""
    import urllib.request
    if os.path.exists(path):
        return path
    req = urllib.request.Request(url, method="HEAD", headers=core.UA)
    size = int(urllib.request.urlopen(req, timeout=120).headers["Content-Length"])
    edges = [size * i // streams for i in range(streams + 1)]
    with open(path + ".tmp", "wb") as fh:
        fh.truncate(size)

    def part(i):
        b = core.get(url, byte_range=(edges[i], edges[i + 1] - 1), timeout=1800)
        with open(path + ".tmp", "r+b") as fh:
            fh.seek(edges[i])
            fh.write(b)
        return len(b)
    got = core.pmap(part, range(streams), threads=streams)
    if sum(got) != size:
        raise IOError("%s: %d of %d bytes" % (url, sum(got), size))
    os.replace(path + ".tmp", path)
    return path


def highdrift_read(path):
    import h5py
    import hdf5plugin  # noqa: F401  (registers the bitshuffle filter the files use)
    with h5py.File(path, "r") as f:
        d = f["data"]
        hdr = {k: d.attrs[k] for k in ("fch1", "foff", "tsamp", "tstart", "source_name", "src_raj", "src_dej")}
        x = d[:, 0, :]
    hdr = {k: (str(v) if k == "source_name" else float(v)) for k, v in hdr.items()}
    return np.asarray(x, np.float32), hdr


def highdrift_scan(x, hdr, dev):
    """Hits of one scan and its best sum over all drifts per channel (noise units)."""
    import torch
    nt, nch = x.shape
    tsamp, foff = hdr["tsamp"], hdr["foff"] * 1e6
    C = highdrift_COARSE
    nc = max(nch // C, 1)
    m = np.median(x, axis=0)
    mc = m[:nc * C].reshape(nc, C)
    # each coarse channel's level, a line through the medians of its two middle quarters (coarse channels differ in gain), and
    # the filterbank's response across a coarse channel, the same in every one, folded from the spectrum over that level; a
    # smooth level alone leaves every coarse channel's edges at half the bandpass
    q, uu = C // 4, (np.arange(C) + 0.5) / C
    resp = np.ones(C)
    for _ in range(2):
        r = mc / resp[None, :]
        lo, hi = np.median(r[:, q:2 * q], axis=1), np.median(r[:, 2 * q:3 * q], axis=1)
        level = np.maximum(lo[:, None] + (hi - lo)[:, None] * (uu[None, :] - 0.375) / 0.25, 1e-6 * np.median(m))
        resp = np.maximum(np.median(mc / level, axis=0), 1e-3)
    band = (level * resp[None, :]).ravel()
    band = np.concatenate([band, np.resize(band[-C:], nch - nc * C)])
    z = x / band[None, :] - 1.0
    dc = np.concatenate([np.clip(np.arange(C // 2, nch, C) + off, 0, nch - 1) for off in (-1, 0, 1)])
    z[:, dc] = 0.0
    B = highdrift_BLOCK
    nb = max(nch // B, 1)
    med = np.median(z[:, :nb * B].reshape(nt, nb, B), axis=(0, 2))
    med = np.concatenate([np.repeat(med, B), np.full(nch - nb * B, med[-1])])
    # the noise of each block, shaped across the coarse channel by the folded noise of single channels over time
    s_ch = 1.4826 * np.median(np.abs(z - np.median(z, axis=0)[None, :]), axis=0)
    lvl = np.median(s_ch[:nb * B].reshape(nb, B), axis=1)
    lvl = np.maximum(np.concatenate([np.repeat(lvl, B), np.full(nch - nb * B, lvl[-1])]), 1e-9)
    prof = np.maximum(np.median((s_ch / lvl)[:nc * C].reshape(nc, C), axis=0), 1e-3)
    sig = np.maximum(lvl * np.resize(prof, nch), 1e-6)
    z = ((z - med[None, :]) / sig[None, :]).astype(np.float32)
    z[:, dc] = 0.0
    step = abs(foff) / (nt * tsamp) / 2.0
    drifts = step * np.arange(-int(highdrift_DRIFT / step), int(highdrift_DRIFT / step) + 1)
    curvs = highdrift_CURV_STEP * np.arange(-int(highdrift_CURV / highdrift_CURV_STEP), int(highdrift_CURV / highdrift_CURV_STEP) + 1)
    t = np.arange(nt) * tsamp
    zt = torch.as_tensor(z, device=dev)
    base = torch.arange(nch, device=dev)
    best = torch.full((nch,), -1e9, device=dev)
    bestd = torch.zeros(nch, dtype=torch.long, device=dev)
    bestc = torch.zeros(nch, dtype=torch.long, device=dev)
    zero = None
    for i, d in enumerate(drifts):
        for ci, a in enumerate(curvs):
            # track channel per sample from midpoint drift and curvature
            sh = torch.as_tensor(np.round((d * (t - t[-1] / 2) + 0.5 * a * (t - t[-1] / 2) ** 2) / foff).astype(np.int64), device=dev)
            idx = (base[None, :] + sh[:, None]).clamp_(0, nch - 1)
            acc = zt.gather(1, idx).sum(0) / math.sqrt(nt)
            if abs(d) < step / 2 and abs(a) < highdrift_CURV_STEP / 2:
                zero = acc.clone()
            better = acc > best
            best = torch.where(better, acc, best)
            bestd = torch.where(better, torch.full_like(bestd, i), bestd)
            bestc = torch.where(better, torch.full_like(bestc, ci), bestc)
    snr, di, cvi, s0 = best.cpu().numpy(), bestd.cpu().numpy(), bestc.cpu().numpy(), zero.cpu().numpy()
    edge = int(np.ceil(highdrift_DRIFT * nt * tsamp / abs(foff))) + 2
    snr[:edge] = -1e9
    snr[nch - edge:] = -1e9
    # a hit near a brighter zero-drift line is that line's sidelobe
    hits = []
    order = np.argsort(-snr)
    taken = np.zeros(nch, bool)
    span_per = nt * tsamp / abs(foff)
    for c in order:
        if snr[c] < highdrift_SNR:
            break
        if taken[c]:
            continue
        d, a = float(drifts[di[c]]), float(curvs[cvi[c]])
        span = int((abs(d) + abs(a) * nt * tsamp / 2) * span_per / 2) + 1
        peak0 = float(s0[max(c - span, 0):c + span + 1].max())
        if (abs(d) >= step or abs(a) >= highdrift_CURV_STEP) and peak0 >= max(highdrift_SNR, snr[c] / 3.0) and s0[c] < peak0 / 2.0:
            taken[c] = True
            continue
        taken[max(c - highdrift_TOL, 0):c + highdrift_TOL + 1] = True
        # channel, frequency and drift at the scan midpoint
        hits.append(dict(chan=int(c), freq_mhz=float(hdr["fch1"] + hdr["foff"] * c), drift=d, curv=a, snr=float(snr[c]), snr0=float(s0[c])))
    return hits, np.maximum(snr, s0), step


def highdrift_fast(drift, snr, snr0):
    """Hit beyond +-4 Hz/s whose zero-drift sum is under half its best."""
    return abs(drift) > 4.0 and snr0 < 0.5 * snr


def highdrift_events(scans, step):
    """Hits in all three ON scans along one drift, OFF scans below threshold at the predicted channels."""
    on = [s for s in scans if s["on"]]
    off = [s for s in scans if not s["on"]]
    foff = on[0]["hdr"]["foff"] * 1e6
    events, pairs = [], 0

    def at(s, h, s_from):
        """Channel and drift in scan s of hit h from scan s_from."""
        dt = (s["mjd"] - s_from["mjd"]) * 86400.0
        a = h.get("curv", 0.0)
        return h["chan"] + (h["drift"] * dt + 0.5 * a * dt * dt) / foff, h["drift"] + a * dt

    for h in on[0]["hits"]:
        chain = [h]
        for s in on[1:]:
            c, d_at = at(s, h, on[0])
            m = [g for g in s["hits"] if abs(g["chan"] - c) <= highdrift_TOL and abs(g["drift"] - d_at) <= 2.5 * step]
            if m:
                chain.append(max(m, key=lambda g: g["snr"]))
        if len(chain) == 2:
            pairs += 1
        if len(chain) < 3:
            continue
        clean = True
        offs = []
        for s in off:
            c = int(round(at(s, h, on[0])[0]))
            lo, hi = max(c - highdrift_TOL, 0), min(c + highdrift_TOL + 1, len(s["best"]))
            v = float(s["best"][lo:hi].max()) if hi > lo else 0.0
            offs.append(v)
            if v >= highdrift_OFF_SNR:
                clean = False
        if clean:
            events.append(dict(freq_mhz=h["freq_mhz"], drift=h["drift"], curv=h.get("curv", 0.0), snr=[g["snr"] for g in chain],
                               snr0=[g["snr0"] for g in chain], off_max=offs))
    return events, pairs


def highdrift_search(part=None, streams=4, limit=0, redo=False):
    """Search the part's cadences without a result; with redo, those whose result predates highdrift_NORM or records a failed fetch, the earlier result kept under ROOT/results_norm1."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan = json.load(open(os.path.join(highdrift_ROOT, "plan.json")))
    k, n = (int(x) for x in part.split("/")) if part else (0, 1)
    todo = plan[k::n]
    if limit:
        todo = todo[:limit]
    tmp = os.path.join(highdrift_ROOT, "tmp_%dof%d" % (k, n) if part else "tmp")  # one scratch directory per part
    res = os.path.join(highdrift_ROOT, "results")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(res, exist_ok=True)

    def name(c):
        return "%s_%s_%d.json" % (c["session"], re.sub(r"[^A-Za-z0-9+-]", "_", c["target"]), int(c["scans"][0]["mjd"] * 86400) % 86400)

    def fetch(c):
        paths = []
        for s in c["scans"]:
            paths.append(highdrift_fetch(highdrift_PIPE + c["session"] + "/holding/" + s["file"], os.path.join(tmp, s["file"]), streams))
        return paths

    def current(c):
        p = os.path.join(res, name(c))
        if not os.path.exists(p):
            return False
        if not redo:
            return True
        try:
            return json.load(open(p)).get("norm") == highdrift_NORM
        except ValueError:
            return False

    def keep(c):
        p = os.path.join(res, name(c))
        if os.path.exists(p):
            os.makedirs(os.path.join(highdrift_ROOT, "results_norm1"), exist_ok=True)
            os.replace(p, os.path.join(highdrift_ROOT, "results_norm1", name(c)))

    todo = [c for c in todo if not current(c)]
    print("%d cadences to search on %s%s" % (len(todo), dev, " again" if redo else ""), flush=True)
    ex = ThreadPoolExecutor(1)
    fut = ex.submit(fetch, todo[0]) if todo else None
    for i, c in enumerate(todo):
        t0 = time.time()
        try:
            paths = fut.result()
        except Exception as e:
            keep(c)
            json.dump(dict(session=c["session"], target=c["target"], error=str(e)[:200]), open(os.path.join(res, name(c)), "w"))
            print("%s %s: fetch failed, %s" % (c["session"], c["target"], str(e)[:80]), flush=True)
            fut = ex.submit(fetch, todo[i + 1]) if i + 1 < len(todo) else None
            continue
        fut = ex.submit(fetch, todo[i + 1]) if i + 1 < len(todo) else None
        t1 = time.time()
        scans, step = [], None
        for s, p in zip(c["scans"], paths):
            x, hdr = highdrift_read(p)
            os.remove(p)
            hits, best, step = highdrift_scan(x, hdr, dev)
            scans.append(dict(s, hdr=hdr, nt=int(x.shape[0]), nch=int(x.shape[1]), hits=hits, best=best))
            del x
        events, pairs = highdrift_events(scans, step)
        nt, tsamp = scans[0]["nt"], scans[0]["hdr"]["tsamp"]
        dnu = abs(scans[0]["hdr"]["foff"]) * 1e6
        flux = highdrift_SNR * highdrift_SEFD / math.sqrt(dnu * nt * tsamp)
        out = dict(session=c["session"], target=c["target"], norm=highdrift_NORM,
                   band=("L" if scans[0]["hdr"]["fch1"] < 2000 else "S" if scans[0]["hdr"]["fch1"] < 3500 else "other"),
                   fch1=scans[0]["hdr"]["fch1"], nt=nt, nch=scans[0]["nch"], tsamp=tsamp, dnu_hz=dnu, step_hz_s=step,
                   flux_limit_jy=flux, events=events, pairs=pairs,
                   scans=[dict(file=s["file"], on=s["on"], mjd=s["mjd"], target=s["target"], source=s["hdr"]["source_name"],
                               n_hits=len(s["hits"]), n_hits_fast=sum(1 for h in s["hits"] if highdrift_fast(h["drift"], h["snr"], h["snr0"])),
                               n_fast_iridium=sum(1 for h in s["hits"] if highdrift_fast(h["drift"], h["snr"], h["snr0"])
                                                  and 1616.0 <= h["freq_mhz"] <= 1626.5),
                               hits=sorted(s["hits"], key=lambda h: -h["snr"])[:200]) for s in scans])
        keep(c)
        json.dump(out, open(os.path.join(res, name(c)), "w"))
        print("%s %s: hits per ON scan %s, OFF %s; %d pairs, %d events; fetch %.0f s, search %.0f s"
              % (c["session"], c["target"], [s["n_hits"] for s in out["scans"] if s["on"]],
                 [s["n_hits"] for s in out["scans"] if not s["on"]], pairs, len(events), t1 - t0, time.time() - t1), flush=True)
    ex.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


def highdrift_inject(scans=1000, per=40, nch=315392, nt=277, tsamp=1.058013):
    """Completeness of the per-scan search on synthetic scans of the L-band products' shape: Gaussian noise on a bandpass dipping at every coarse-channel edge, per lines a scan at drifts to +-highdrift_DRIFT, curvatures to +-highdrift_CURV and integrated S/N 5-30, each sample's power split between the two channels it falls between; one line a scan in ROOT/injections.jsonl (resumable)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(highdrift_ROOT, "injections.jsonl")
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    foff = -2.861023e-3  # MHz
    hdr = dict(fch1=1900.0, foff=foff, tsamp=tsamp, tstart=0.0, source_name="injection", src_raj=0.0, src_dej=0.0)
    u = (np.arange(nch) % highdrift_COARSE) / highdrift_COARSE
    band = ((1 + 0.2 * np.sin(2 * np.pi * np.arange(nch) / nch)) * (1 - 0.5 * np.abs(2 * u - 1) ** 8)).astype(np.float32)
    tm = np.arange(nt) * tsamp
    tm -= tm[-1] / 2
    edge = int(np.ceil(highdrift_DRIFT * nt * tsamp / (abs(foff) * 1e6))) + 2
    rows = np.arange(nt)
    print("%d synthetic scans to search on %s, %d done" % (scans - done, dev, done), flush=True)
    t0 = time.time()
    for k in range(done, scans):
        rng = np.random.default_rng(1000003 * (k + 1))
        c = []
        while len(c) < per:  # midpoint channels 300 apart, clear of the band edges
            x0 = int(rng.integers(edge + 100, nch - edge - 100))
            if all(abs(x0 - y) >= 300 for y in c):
                c.append(x0)
        d = rng.uniform(-highdrift_DRIFT, highdrift_DRIFT, per)
        a = rng.uniform(-highdrift_CURV, highdrift_CURV, per)
        s = rng.uniform(5.0, 30.0, per)
        x = rng.standard_normal((nt, nch), dtype=np.float32)
        for ci, di, ai, si in zip(c, d, a, s):
            p = ci + (di * tm + 0.5 * ai * tm ** 2) / (foff * 1e6)
            lo = np.floor(p).astype(np.int64)
            w = (p - lo).astype(np.float32)
            x[rows, lo] += si / math.sqrt(nt) * (1 - w)
            x[rows, lo + 1] += si / math.sqrt(nt) * w
        x = band[None, :] * (1 + 0.01 * x)
        hits, _, step = highdrift_scan(x, hdr, dev)
        out = []
        for ci, di in zip(c, d):
            m = [h for h in hits if abs(h["chan"] - ci) <= highdrift_TOL and abs(h["drift"] - di) <= 2.5 * step]
            out.append(round(max(h["snr"] for h in m), 2) if m else None)
        with open(path, "a") as fh:
            fh.write(json.dumps(dict(scan=k, norm=highdrift_NORM, chan=c, drift=[round(float(v), 3) for v in d],
                                     curv=[round(float(v), 4) for v in a], snr_in=[round(float(v), 2) for v in s], snr_out=out,
                                     hits=len(hits)), separators=(",", ":")) + "\n")
        if (k + 1) % 10 == 0:
            print("%d of %d, %.0f s a scan" % (k + 1, scans, (time.time() - t0) / (k + 1 - done)), flush=True)


def highdrift_distances(targets):
    """Parallaxes (mas) of targets by SIMBAD identifier."""
    ids = {t: re.sub(r"^(HIP|Hip|GJ|Gj|GL|LHS|BD)(?=[0-9+-])", lambda m: m.group(1).upper() + " ", t) for t in targets}
    out = {}
    keys = sorted(set(ids.values()))
    for k in range(0, len(keys), 50):
        q = ("SELECT i.id, b.plx_value FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)"
             % ",".join("'%s'" % x.replace("'", "''") for x in keys[k:k + 50]))
        for r in core.simbad(q):
            out[" ".join(str(r["id"]).split())] = r.get("plx_value")
    return {t: out.get(v) for t, v in ids.items()}


def highdrift_record():
    R = [json.load(open(p)) for p in glob.glob(os.path.join(highdrift_ROOT, "results", "*.json"))]
    ok = [r for r in R if "events" in r]
    targets = sorted({r["target"] for r in ok})
    on_fast = [s["n_hits_fast"] for r in ok for s in r["scans"] if s["on"]]
    off_fast = [s["n_hits_fast"] for r in ok for s in r["scans"] if not s["on"]]
    irid = [s.get("n_fast_iridium") for r in ok for s in r["scans"] if s.get("n_fast_iridium") is not None]
    irid_frac = sum(irid) / max(sum(s["n_hits_fast"] for r in ok for s in r["scans"] if s.get("n_fast_iridium") is not None), 1)
    on_all = [s["n_hits"] for r in ok for s in r["scans"] if s["on"]]
    off_all = [s["n_hits"] for r in ok for s in r["scans"] if not s["on"]]
    ev = [(r, e) for r in ok for e in r["events"]]
    ev_fast = [(r, e) for r, e in ev if all(highdrift_fast(e["drift"], s, s0) for s, s0 in zip(e["snr"], e["snr0"]))]
    plx = highdrift_distances(targets)
    rows = []
    for t in targets:
        rs = [r for r in ok if r["target"] == t]
        flux = min(r["flux_limit_jy"] for r in rs)
        p = plx.get(t)
        eirp = 4 * math.pi * (3.0857e16 * 1000.0 / float(p)) ** 2 * flux * 1e-26 * rs[0]["dnu_hz"] if p else None
        n_ev = sum(len(r["events"]) for r in rs)
        n_fast = sum(1 for r in rs for e in r["events"] if all(highdrift_fast(e["drift"], s, s0) for s, s0 in zip(e["snr"], e["snr0"])))
        rows.append(dict(target="bl:" + t, stat=n_ev, upper_limit=eirp, unit="EIRP (W) of an unresolved line at 10 sigma",
                         vet=None if n_fast else ("events only within +-4 Hz/s" if n_ev else None),
                         note="%d cadences (%s band); %.3f Jy over %.0f Hz; events at |drift| > 4 Hz/s: %d"
                              % (len(rs), rs[0]["band"], flux, rs[0]["dnu_hz"], n_fast)))
    core.record_search(
        "bl_highdrift", "Narrowband signals at high drift rates",
        archive="Breakthrough Listen GBT mid-resolution products (gpuspec.0002.h5, open data)",
        release="%d cadences of %d targets in %d sessions (%d failed)" % (len(ok), len(targets), len({r["session"] for r in ok}), len(R) - len(ok)),
        selection="every ABACAD cadence with mid-resolution products in the pipeline directory",
        code=os.path.abspath(__file__), entry="highdrift", n_targets=len(targets),
        params=dict(drift=highdrift_DRIFT, step="half a channel per scan", snr=highdrift_SNR, off_snr=highdrift_OFF_SNR,
                    block=highdrift_BLOCK, tol=highdrift_TOL, sefd=highdrift_SEFD),
        null_summary="OFF scans: %.1f drifting hits per scan beyond +-4 Hz/s (%.1f hits at any drift) against %.1f (%.1f) in the ON "
                     "scans; %.0f per cent of the drifting hits lie in the Iridium downlink band, 1616-1626.5 MHz"
                     % (np.mean(off_fast) if off_fast else 0, np.mean(off_all) if off_all else 0,
                        np.mean(on_fast) if on_fast else 0, np.mean(on_all) if on_all else 0, 100 * irid_frac),
        sensitivity="an unresolved line at 10 sigma over 2.86 kHz and 293 s: %.3f Jy at SEFD %.0f Jy; drifts to +-%.0f Hz/s"
                    % (np.median([r["flux_limit_jy"] for r in ok]) if ok else 0, highdrift_SEFD, highdrift_DRIFT),
        ledger="Narrowband signals at high drift rates",
        result="%d events in all three ON scans and absent from the OFF scans, %d of them beyond +-4 Hz/s, in %d cadences of %d targets"
               % (len(ev), len(ev_fast), len(ok), len(targets)))
    core.clear_targets("bl_highdrift")
    core.record_targets("bl_highdrift", rows)
    print("recorded %d cadences, %d targets, %d events (%d beyond 4 Hz/s)" % (len(ok), len(targets), len(ev), len(ev_fast)))
    for r, e in sorted(ev_fast, key=lambda x: -max(x[1]["snr"]))[:20]:
        print("  %s %s: %.6f MHz, %+.1f Hz/s, S/N %s, zero-drift %s, OFF max %s"
              % (r["session"], r["target"], e["freq_mhz"], e["drift"], ["%.0f" % v for v in e["snr"]],
                 ["%.0f" % v for v in e["snr0"]], ["%.1f" % v for v in e["off_max"]]))


def highdrift_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("sessions", "plan", "search", "inject", "record"))
    p.add_argument("--part", default=None)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--scans", type=int, default=1000)
    p.add_argument("--per", type=int, default=40)
    p.add_argument("--redo", action="store_true")
    a = p.parse_args()
    if a.mode == "sessions":
        highdrift_sessions()
    elif a.mode == "plan":
        highdrift_plan()
    elif a.mode == "search":
        highdrift_search(a.part, a.streams, a.limit, a.redo)
    elif a.mode == "inject":
        highdrift_inject(a.scans, a.per)
    else:
        highdrift_record()


# --------------------------------------------------------------------------------------------------------------

soi_DOC = r"""Published signals of interest (Ma et al. 2023, Table 1) traced through their whole session in the 2.84 Hz product: each
signal's line through the per-spectrum peaks of its ON scans, its power on that line in every scan against parallel tracks, and
the narrowband tracks near it in every scan, each spectrum summed over the channels a track's drift crosses within it.

    python -I seti.py breakthrough soi trace [--tag MLc8] [--procs 4]   (ROOT/<tag>.jsonl: the line, then one line a scan)
    python -I seti.py breakthrough soi table                            (ROOT/table.json)
    python -I seti.py breakthrough soi waterfall --tag MLc8             (ROOT/<tag>_waterfall.npz: every scan over the band the line crosses)
    python -I seti.py breakthrough soi record"""

soi_ROOT = os.environ.get("SOI_ROOT", core.data("bl_soi"))
# Ma et al. (2023), Table 1: tag, star, session, frequency (MHz) and drift rate (Hz/s) as tabulated
soi_LIST = (("MLc1", "HIP13402", "AGBT16A_999_212", 1188.539231, 1.11), ("MLc2", "HIP118212", "AGBT16B_999_105", 1347.862244, -0.44),
            ("MLc3", "HIP62207", "AGBT16A_999_213", 1351.625410, -0.05), ("MLc4", "HIP54677", "AGBT16A_999_182", 1372.987594, -0.11),
            ("MLc5", "HIP54677", "AGBT16A_999_182", 1376.988694, -0.11), ("MLc6", "HIP56802", "AGBT16A_999_188", 1435.940307, -0.18),
            ("MLc7", "HIP13402", "AGBT16A_999_215", 1487.482046, 0.10), ("MLc8", "HIP62207", "AGBT16A_999_213", 1724.972561, -0.126))
soi_HALF = 25000.0  # Hz searched either side of the tabulated frequency
soi_OWN = 6000.0  # Hz from the tabulated frequency within which the signal is the strongest track of each ON scan
soi_DRIFTS = np.arange(-2.0, 2.0001, 0.002)  # Hz/s
soi_SNR = 8.0
soi_NEAR = (500.0, 0.05)  # a track of another pointing: Hz from the line carried to its time, Hz/s from the line's drift
soi_OFF_Z = 3.0  # combined significance of the three OFF scans' power on the line


def soi_listing(session):
    """Every mid-resolution scan of a session in time order, each marked ON, OFF or neither by its ABACAD cadence."""
    path = os.path.join(soi_ROOT, "listing_%s.json" % session)
    if not os.path.exists(path):
        os.makedirs(soi_ROOT, exist_ok=True)
        d = core.get(highdrift_PIPE + session + "/holding/", timeout=120).decode(errors="replace")
        json.dump(sorted({(int(m[1]), int(m[2]), m[3], m[0]) for m in highdrift_RX.findall(d)}), open(path, "w"))
    s = [tuple(x) for x in json.load(open(path))]
    scans = [dict(mjd=f[0] + f[1] / 86400.0, target=f[2], file=f[3], on=None, cadence=None) for f in s]
    i = 0
    while i <= len(s) - 6:
        w = s[i:i + 6]
        a = w[0][2]
        t = [f[0] + f[1] / 86400.0 for f in w]
        if w[2][2] == a and w[4][2] == a and len({w[1][2], w[3][2], w[5][2], a}) == 4 and t[5] - t[0] < 0.1:
            for j in range(6):
                scans[i + j].update(on=(j % 2 == 0), cadence=a)
            i += 6
        else:
            i += 1
    return scans


def soi_window(session, scan, f0, smooth):
    """Unit-noise spectrogram of a scan within soi_HALF of f0 and its channels in Hz from f0: over each channel's own level in the scan, or with smooth over 256-channel block medians of the scan's median spectrum, which keeps a steady tone."""
    url = highdrift_PIPE + session + "/holding/" + scan["file"].replace(".gpuspec.0002.h5", ".gpuspec.0000.h5")
    meta = science.header(url)
    k = int(((f0 - meta["fch1"]) / meta["foff"]) // science.COARSE)
    x, ffirst = science.coarse_channel(url, k)
    c = int(round((f0 - ffirst) / meta["foff"]))
    h = int(soi_HALF / abs(meta["foff"] * 1e6))
    lo = max(c - h, 0)
    w = x[:, lo:c + h + 1].astype(float)
    if smooth:
        m = np.median(w, axis=0)
        nb = len(m) // 256
        z = w / np.interp(np.arange(len(m)), np.arange(nb) * 256 + 128, [np.median(m[j * 256:(j + 1) * 256]) for j in range(nb)])[None, :] - 1
    else:
        z = w / np.median(w, axis=0, keepdims=True) - 1
    z /= 1.4826 * np.median(np.abs(z - np.median(z)))
    return dict(z=z, hz=(ffirst + meta["foff"] * np.arange(lo, lo + z.shape[1]) - f0) * 1e6, foff=meta["foff"] * 1e6,
                t=meta["tstart"] + (np.arange(meta["nt"]) + 0.5) * meta["tsamp"] / 86400, tsamp=meta["tsamp"])


def soi_width(d, w):
    """Channels a line drifting at d (Hz/s) crosses within one spectrum."""
    return max(1, int(round(abs(d) * w["tsamp"] / abs(w["foff"]))))


def soi_boxcar(z, n):
    """Each channel summed with its neighbours over n channels centred on it, over sqrt(n), so the noise stays unit."""
    if n == 1:
        return z
    c = np.cumsum(np.pad(z, ((0, 0), (1, 0))), axis=1)
    out = np.zeros_like(z)
    out[:, (n - 1) // 2:(n - 1) // 2 + z.shape[1] - n + 1] = (c[:, n:] - c[:, :-n]) / math.sqrt(n)
    return out


def soi_sweep(w):
    """Best straight-track sum over the drifts by mid-scan channel and the drift giving it, each drift's spectra summed over the channels it crosses."""
    z, foff, tsamp = w["z"], w["foff"], w["tsamp"]
    nt, nch = z.shape
    tt = (np.arange(nt) - (nt - 1) / 2) * tsamp
    best, bd, zs = np.full(nch, -1e9), np.zeros(nch), {}
    for d in soi_DRIFTS:
        n = soi_width(d, w)
        if n not in zs:
            zs[n] = soi_boxcar(z, n)
        sh = np.round(d * tt / foff).astype(int)
        lo, hi = max(0, -sh.min()), nch - max(0, sh.max())
        acc = sum(zs[n][i, lo + sh[i]:hi + sh[i]] for i in range(nt)) / math.sqrt(nt)
        seg = best[lo:hi]
        up = acc > seg
        seg[up] = acc[up]
        bd[lo:hi][up] = d
    return best, bd


def soi_tracks(w, sep=20):
    """Local maxima of the sweep above soi_SNR, at most 40: (S/N, Hz from f0 at mid-scan, drift)."""
    best, bd = soi_sweep(w)
    out, taken = [], np.zeros(len(best), bool)
    for j in np.argsort(-best):
        if best[j] < soi_SNR or len(out) >= 40:
            break
        if taken[max(0, j - sep):j + sep + 1].any():
            continue
        taken[j] = True
        out.append((round(float(best[j]), 1), round(float(w["hz"][j]), 1), round(float(bd[j]), 3)))
    return out


def soi_line(session, star, f0, scans):
    """The signal's topocentric line through the per-spectrum peaks of the strongest track within soi_OWN of f0 in each of its three ON scans: reference MJD, Hz from f0 there and drift (Hz/s), with those tracks' S/N."""
    T, F, H, snr, first = [], [], [], [], None
    for s in [s for s in scans if s["cadence"] == star and s["on"]][:3]:
        w = soi_window(session, s, f0, False)
        best, bd = soi_sweep(w)
        best[np.abs(w["hz"]) > soi_OWN] = -1e9
        c = int(np.argmax(best))
        d = float(bd[c])
        snr.append(round(float(best[c]), 1))
        if first is None or best[c] > first[0]:
            first = (float(best[c]), float(w["t"].mean()), float(w["hz"][c]), d)
        z = soi_boxcar(w["z"], soi_width(d, w))
        nt, nch = z.shape
        tt = (np.arange(nt) - (nt - 1) / 2) * w["tsamp"]
        for i in range(nt):
            ci = int(round(c + d * tt[i] / w["foff"]))
            seg = z[i, ci - 4:ci + 5]
            j = int(np.argmax(seg))
            dj = 0.0
            if 0 < j < 8:
                den = seg[j - 1] - 2 * seg[j] + seg[j + 1]
                dj = 0.5 * (seg[j - 1] - seg[j + 1]) / den if den != 0 else 0.0
            T.append(w["t"][i])
            F.append(float(np.interp(ci - 4 + j + dj, np.arange(nch), w["hz"])))
            H.append(float(seg[j]))
    T, F, H = np.array(T), np.array(F), np.array(H)
    g = H >= 3
    if g.sum() < 8:  # too faint for peaks: the strongest ON track itself
        return dict(t0=first[1], hz=first[2], drift=first[3], on_snr=snr)
    t0 = float(T[g].mean())
    A = np.vstack([np.ones(g.sum()), (T[g] - t0) * 86400]).T
    wt = np.minimum(H[g], 30) ** 2
    beta = np.linalg.solve(A.T @ (A * wt[:, None]), A.T @ (wt * F[g]))
    return dict(t0=t0, hz=float(beta[0]), drift=float(beta[1]), on_snr=snr)


def soi_scan(job):
    """One scan of the session: its tracks above soi_SNR, and the power on the signal's line (each spectrum summed over the channels the line crosses, three such sums about it) against parallel tracks 20 to 2000 channels away."""
    session, f0, s, line = job
    try:
        w = soi_window(session, s, f0, True)
    except Exception as e:
        return dict(file=s["file"], target=s["target"], cadence=s["cadence"], on=s["on"], error=str(e)[:200])
    z = soi_boxcar(w["z"], soi_width(line["drift"], w))
    nt, nch = z.shape
    ci = np.round(np.interp(line["hz"] + line["drift"] * (w["t"] - line["t0"]) * 86400, w["hz"][::-1], np.arange(nch)[::-1])).astype(int)
    pw = []
    for o in [0] + [o for o in range(-2000, 2001, 20) if abs(o) >= 20]:
        idx = ci + o
        pw.append(np.nan if idx.min() < 1 or idx.max() > nch - 2 else float(np.mean([z[i, idx[i] - 1:idx[i] + 2].sum() for i in range(nt)])))
    pw = np.array(pw)
    null = pw[1:][np.isfinite(pw[1:])]
    return dict(file=s["file"], target=s["target"], cadence=s["cadence"], on=s["on"], mjd=round(float(w["t"].mean()), 5),
                on_line=round(float(pw[0]), 2), null_mean=round(float(null.mean()), 2), null_std=round(float(null.std()), 2),
                null_n=int(len(null)), as_high=int((null >= pw[0]).sum()), tracks=soi_tracks(w))


def soi_trace(tag=None, procs=4):
    """Each signal of interest, or the one tagged, through every scan of its session: ROOT/<tag>.jsonl, the line, then a scan a line."""
    from multiprocessing import Pool
    os.makedirs(soi_ROOT, exist_ok=True)
    for name, star, session, f0, _ in soi_LIST:
        if tag and name != tag:
            continue
        scans = soi_listing(session)
        line = soi_line(session, star, f0, scans)
        print("%s, %s %s: %.6f MHz at MJD %.5f, drift %+.4f Hz/s, ON tracks S/N %s; %d scans"
              % (name, session, star, f0 + line["hz"] / 1e6, line["t0"], line["drift"], line["on_snr"], len(scans)), flush=True)
        with Pool(procs) as p, open(os.path.join(soi_ROOT, "%s.jsonl" % name), "w") as fh:
            fh.write(json.dumps(dict(line, tag=name, star=star, session=session, f0=f0)) + "\n")
            for r in p.imap(soi_scan, [(session, f0, s, line) for s in scans]):
                fh.write(json.dumps(r) + "\n")


def soi_off_z(off):
    """Combined significance of the OFF scans' power on the line against their parallel tracks."""
    z = [(o["on_line"] - o["null_mean"]) / o["null_std"] for o in off if o["null_std"] > 0]
    return sum(z) / math.sqrt(len(z)) if z else 0.0


def soi_table():
    """Each signal: its line, ON tracks and power on the line, its own OFF scans' power on the line, and the pointings away from its star holding a track within soi_NEAR of the line above soi_SNR; ROOT/table.json."""
    out = []
    for name, star, session, f0, d_tab in soi_LIST:
        rows = [json.loads(l) for l in open(os.path.join(soi_ROOT, "%s.jsonl" % name))]
        line, rows = rows[0], rows[1:]
        ok = [r for r in rows if "error" not in r]
        own = [r for r in ok if r["cadence"] == star and abs(r["mjd"] - line["t0"]) < 0.05]
        off = [dict((k, r[k]) for k in ("file", "on_line", "null_mean", "null_std", "as_high", "null_n")) for r in own if r["on"] is False]
        other = []
        for r in ok:
            if r["target"] == star:
                continue
            pred = line["hz"] + line["drift"] * (r["mjd"] - line["t0"]) * 86400
            near = [q for q in r["tracks"] if abs(q[1] - pred) <= soi_NEAR[0] and abs(q[2] - line["drift"]) <= soi_NEAR[1]]
            if near:
                other.append(dict(file=r["file"], target=r["target"], on=r["on"], hours=round((r["mjd"] - line["t0"]) * 24, 2),
                                  snr=max(q[0] for q in near)))
        out.append(dict(tag=name, star=star, session=session, f0=f0, drift_tabulated=d_tab, mhz=round(f0 + line["hz"] / 1e6, 6),
                        drift=round(line["drift"], 4), t0=line["t0"], on_snr=line["on_snr"], on_line=[r["on_line"] for r in own if r["on"]],
                        off=off, off_z=round(soi_off_z(off), 1), other=other, scans=len(rows), failed=len(rows) - len(ok)))
        o = out[-1]
        print("%s %s: %.6f MHz, %+.3f Hz/s (tabulated %+.3f), ON S/N %s, on the line ON %s OFF %s (%.1f sigma); %d scans away from "
              "the star hold it, S/N to %s" % (name, star, o["mhz"], o["drift"], d_tab, o["on_snr"], o["on_line"], [x["on_line"] for x in off],
                                               o["off_z"], len(other), max([x["snr"] for x in other], default="-")))
    json.dump(out, open(os.path.join(soi_ROOT, "table.json"), "w"), indent=1)


def soi_waterfall(tag):
    """Every scan of the signal's session over the band its line crosses in the session and 700 Hz either side, on the smooth level: ROOT/<tag>_waterfall.npz with each scan's spectra (float16, NaN-padded), their MJDs, Hz from f0, ON, OFF or neither (1, 0, -1), and each scan's target and cadence."""
    line = json.loads(open(os.path.join(soi_ROOT, "%s.jsonl" % tag)).readline())
    name, star, session, f0, _ = [x for x in soi_LIST if x[0] == tag][0]
    scans = soi_listing(session)
    ends = [line["hz"] + line["drift"] * (t - line["t0"]) * 86400 for t in (scans[0]["mjd"], scans[-1]["mjd"] + 300 / 86400)]
    lo, hi = min(ends) - 700, max(ends) + 700
    W = [soi_window(session, s, f0, True) for s in scans]
    m = (W[0]["hz"] >= lo) & (W[0]["hz"] <= hi)
    nt = max(w["z"].shape[0] for w in W)
    Z = np.full((len(W), nt, int(m.sum())), np.nan, np.float16)
    T = np.full((len(W), nt), np.nan)
    for i, w in enumerate(W):
        Z[i, :w["z"].shape[0]] = w["z"][:, m]
        T[i, :len(w["t"])] = w["t"]
    np.savez_compressed(os.path.join(soi_ROOT, "%s_waterfall.npz" % tag), z=Z, t=T, hz=W[0]["hz"][m],
                        on=np.array([{True: 1, False: 0, None: -1}[s["on"]] for s in scans]),
                        target=np.array([s["target"] for s in scans]), cadence=np.array([s["cadence"] or "" for s in scans]))
    print("%s: %d scans, %.0f to %.0f Hz from %.6f MHz" % (tag, len(W), lo, hi, f0))


def soi_record():
    T = json.load(open(os.path.join(soi_ROOT, "table.json")))
    rows = []
    for star in sorted({t["star"] for t in T}):
        notes, vets = [], []
        for t in [t for t in T if t["star"] == star]:
            v = "sidelobe-transmitter" if t["other"] else "off-source" if t["off_z"] >= soi_OFF_Z else None
            vets.append(v)
            notes.append("%s %.6f MHz %+.3f Hz/s, ON S/N %s: %s" % (
                t["tag"], t["mhz"], t["drift"], "/".join("%.0f" % x for x in t["on_snr"]),
                "in %d scans away from the star (S/N to %.0f) and its OFF scans at %.1f sigma" % (len(t["other"]), max(x["snr"] for x in t["other"]), t["off_z"])
                if t["other"] else "in its OFF scans at %.1f sigma" % t["off_z"] if v else "no track above S/N %.0f in its ON scans" % soi_SNR
                if max(t["on_snr"]) < soi_SNR else "absent from its OFF scans (%.1f sigma) and other pointings" % t["off_z"]))
        rows.append(dict(target="bl:" + star, stat=len(notes), unit="signals of interest of Ma et al. (2023)",
                         vet=vets[0] if all(vets) else None, note="; ".join(notes)))
    explained = sum(1 for t in T if t["other"] or t["off_z"] >= soi_OFF_Z)
    core.record_search(
        "bl_soi", "Signals of interest traced through their sessions",
        archive="Breakthrough Listen GBT high-resolution products (gpuspec.0000.h5, open data)",
        release="%d signals of interest in %d sessions, %d scans" % (len(T), len({t["session"] for t in T}), sum(t["scans"] for t in T)),
        selection="the eight signals of interest of Ma et al. (2023), Table 1",
        code=os.path.abspath(__file__), entry="soi", n_targets=len(rows), n_unit="stars",
        params=dict(half_hz=soi_HALF, own_hz=soi_OWN, drifts="%+.0f to %+.0f Hz/s by 0.002" % (soi_DRIFTS[0], soi_DRIFTS[-1]), snr=soi_SNR,
                    near_hz=soi_NEAR[0], near_drift=soi_NEAR[1], off_z=soi_OFF_Z),
        null_summary="each scan's power on the signal's line against parallel tracks 20 to 2000 channels away in the same scan",
        sensitivity="tracks of S/N %.0f over a 5-minute scan in 2.84 Hz channels, each spectrum summed over the channels the drift crosses" % soi_SNR,
        ledger="Signals of interest traced through their sessions",
        result="%d of %d signals are in pointings away from their star or in their own OFF scans on their line" % (explained, len(T)))
    core.clear_targets("bl_soi")
    core.record_targets("bl_soi", rows)
    print("recorded %d signals of %d stars, %d explained" % (len(T), len(rows), explained))


def soi_cli():
    quiet_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("trace", "table", "waterfall", "record"))
    p.add_argument("--tag", default=None)
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    if a.mode == "trace":
        soi_trace(a.tag, a.procs)
    elif a.mode == "table":
        soi_table()
    elif a.mode == "waterfall":
        soi_waterfall(a.tag or "MLc8")
    else:
        soi_record()


shgb_DOC = r"""The 2004 SETI@home candidate SHGb02+14a: a tone near 1420 MHz between Pisces and Aries beside G 73-10 and TZ Arietis (GJ 83.1, 4.5 pc), drifting 8-37 Hz/s. Breakthrough Listen's GBT cadences on GJ 83.1 (2016-06-18 at L band, which holds the line; 2016-09-03 and 2017-08-21 at S band, which does not): the coarse channel holding 1420.4 MHz in every ON and OFF scan, within 0.5 MHz of the line, each spectrogram divided by its channels' medians over time, dedrifted at every rate from -40 to +40 Hz/s in steps of 0.25 Hz/s and summed over its spectra, and peaks above 10 times the robust noise kept; a tone on the ON scans of a cadence that no OFF scan of the cadence shows.

    python -I seti.py breakthrough shgb [record] [--target gj83.1] [--line-mhz 1420.405751] [--window-mhz 0.5] [--rate-max 40] [--rate-step 0.25] [--snr 10] [--drift-min 8] [--drift-max 37]"""

SHGB_DIRS = ("https://bldata.berkeley.edu/pipeline/AGBT16A_999_224/holding/", "https://bldata.berkeley.edu/pipeline/AGBT16B_999_19/holding/",
             "https://bldata.berkeley.edu/pipeline/AGBT17B_999_12/holding/")
SHGB_FILE = re.compile(r"href=\"([^\"]*guppi_(\d+)_(\d+)_([^_\"/]+?)_(\d{4})\.gpuspec\.0000\.h5)\"")
SHGB_HI = 1420.405751


def shgb_scans(directory_url):
    """High-resolution scans of a BL directory in time order: (scan number, target, url)."""
    page = core.get(directory_url, timeout=60).decode(errors="replace")
    return sorted({(int(m[4]), m[3], directory_url + m[0]) for m in SHGB_FILE.findall(page)})


def shgb_peaks(x, dt_s, foff_hz, rates, snr=10.0):
    """Peaks of the dedrifted sums of a normalised spectrogram at each rate (Hz/s): (rate, channel, S/N) above SNR times the robust noise."""
    nt, nc = x.shape
    out = []
    for r in rates:
        s = np.zeros(nc)
        for t in range(nt):
            s += np.roll(x[t], -int(round(r * t * dt_s / foff_hz)))
        med = np.median(s)
        z = (s - med) / (1.4826 * np.median(np.abs(s - med)) + 1e-12)
        for c in np.nonzero(z > snr)[0]:
            out.append((float(r), int(c), float(z[c])))
    return out


def shgb_main(record=False, target="gj83.1", line_mhz=SHGB_HI, window_mhz=0.5, rate_max=40.0, rate_step=0.25, snr=10.0, drift_min=8.0, drift_max=37.0):
    rates = np.arange(-rate_max, rate_max + rate_step / 2, rate_step)
    on_name = target.lower()
    cadences = []
    for d in SHGB_DIRS:
        scans = shgb_scans(d)
        on = [s for s in scans if s[1].lower() == on_name]
        if not on:
            continue
        lo, hi = on[0][0], on[-1][0]
        cadences.append([s for s in scans if lo <= s[0] <= hi + 1])
    summary = []
    for cad in cadences:
        peaks = {}
        for n, target, url in cad:
            try:
                meta = science.header(url)
                foff_hz = meta["foff"] * 1e6
                k = int((line_mhz - meta["fch1"]) / (meta["foff"] * science.COARSE))
                if not 0 <= k < meta["nch"] // science.COARSE:
                    # a band that does not hold the line: the S-band cadences of 2016-09 and 2017
                    print("scan %04d %s: %.0f-%.0f MHz, no coverage of the line" % (n, target, meta["fch1"] + meta["foff"] * meta["nch"], meta["fch1"]), flush=True)
                    peaks[(n, target)] = None
                    continue
                x, f0 = science.coarse_channel(url, k)
            except Exception as exc:
                print("scan %04d %s: not read (%s)" % (n, target, repr(exc)[:80]), flush=True)
                peaks[(n, target)] = None
                continue
            f = f0 + meta["foff"] * np.arange(x.shape[1])
            win = np.abs(f - line_mhz) < window_mhz
            x = x[:, win].astype(np.float64)
            x = x / np.median(x, axis=0) - 1.0
            fw = f[win]
            p = shgb_peaks(x, meta["tsamp"], foff_hz, rates, snr)
            peaks[(n, target)] = [(r, float(fw[c]), z) for r, c, z in p]
            print("scan %04d %s: %d peaks above %g sigma; strongest %s" % (n, target, len(p), snr, sorted(peaks[(n, target)], key=lambda t: -t[2])[:3]), flush=True)
        off = [pk for (n, t), v in peaks.items() if t.lower() != on_name and v for pk in v]
        left = []
        for (n, t), v in peaks.items():
            if t.lower() != on_name or v is None:
                continue
            for r, fq, z in v:
                if not any(abs(fq - fo) < 0.002 and abs(r - ro) < 1.0 for ro, fo, zo in off):
                    left.append((n, r, fq, z))
        summary.append(dict(dir=cad[0][2].rsplit("/", 2)[0], scans=[(n, t) for n, t, _ in cad], on_only=left, unread=[(n, t) for (n, t), v in peaks.items() if v is None],
                            n_on=sum(len(v) for (n, t), v in peaks.items() if t.lower() == on_name and v), n_off=len(off)))
        print("cadence %s: %d ON peaks, %d OFF peaks, %d on the target alone %s" % (summary[-1]["dir"], summary[-1]["n_on"], len(off), len(left), left[:10]), flush=True)
    cand = [(s["dir"], pk) for s in summary for pk in s["on_only"] if drift_min <= abs(pk[1]) <= drift_max]
    print("%d cadences; peaks on %s alone: %d, of them in the candidate's drift range %g-%g Hz/s: %d %s" % (
        len(summary), on_name, sum(len(s["on_only"]) for s in summary), drift_min, drift_max, len(cand), cand))
    if record:
        core.record_run("shgb02_14a", "The 2004 SETI@home candidate SHGb02+14a", "Breakthrough Listen open data, GBT L band, GJ 83.1 cadences of 2016-2017",
                        "%d cadences, %d scans" % (len(summary), sum(len(s["scans"]) for s in summary)), os.path.abspath(__file__), "shgb", len(summary), "cadences",
                        dict(line_mhz=line_mhz, window_mhz=window_mhz, rates_hz_s=[-rate_max, rate_max], step=rate_step, snr=snr, drift_hz_s=[drift_min, drift_max]),
                        "the OFF scans of each cadence, dedrifted the same way",
                        "a tone within %g MHz of %.1f MHz drifting %g-%g Hz/s on TZ Arietis and not off it" % (window_mhz, line_mhz, drift_min, drift_max),
                        "%d peaks on the target alone, %d in the candidate's drift range" % (sum(len(s["on_only"]) for s in summary), len(cand)),
                        [dict(target="bl:GJ83.1:%s" % s["dir"].rsplit("/", 1)[-1], stat=len(s["on_only"]), unit="peaks on the target alone",
                              vet=None if any(drift_min <= abs(pk[1]) <= drift_max for pk in s["on_only"]) else "not-measured" if not s["n_on"] and not s["n_off"] else "off-source",
                              note="scans %s; unread %s; ON-only peaks %s" % (s["scans"], s["unread"], s["on_only"][:10])) for s in summary])
        print("recorded")


APPROACHES = {
    "shgb": (lambda: core.run_cli(shgb_main), shgb_DOC),
    "highdrift": (highdrift_cli, highdrift_DOC),
    "soi": (soi_cli, soi_DOC),
    "pulses": (pulses_cli, pulses_DOC),
    "frames": (frames_cli, frames_DOC),
    "curved": (curved_cli, curved_DOC),
    "curved_batch": (curved_batch_cli, curved_batch_DOC),
    "offs": (offs_cli, offs_DOC),
    "followup": (followup_cli, followup_DOC),
    "combs": (combs_cli, combs_DOC),
    "hops": (hops_cli, hops_DOC),
}


