"""Radio surveys, spacecraft and ground receivers, pulsars and radar beams."""
import argparse
import glob
import gzip
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

from setitools import catalogues, core, science
from setitools.core import num, quiet_warnings, sep_arcsec


# --------------------------------------------------------------------------------------------------------------

hi4pi_DOC = r"""Single-channel 21 cm spikes at nearby stars' rest velocities in HI4PI, ring-differenced against spectral and sky controls; repeated in the CMB and GSR frames.

    python -I seti.py radio_sky hi4pi fetch [--threads 4]
    python -I seti.py radio_sky hi4pi search
    python -I seti.py radio_sky hi4pi frames
    python -I seti.py radio_sky hi4pi record"""

hi4pi_ROOT = os.environ.get("HI4PI_ROOT", core.data("hi4pi"))
hi4pi_HERE = core.RESULTS
hi4pi_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/594/A116/HPX/"
hi4pi_NSIDE = 1024
hi4pi_START, hi4pi_ROW, hi4pi_NCH, hi4pi_NROW = 11520, 3772, 933, 65536
hi4pi_SIZE = 247213440
hi4pi_CRPIX, hi4pi_CDELT = 466.921630003, 1.2882149691242  # km/s, LSRK radio velocity, 1-based pixel
hi4pi_BEAM_ARCMIN = 16.2
hi4pi_HALF_NOISE = 40
hi4pi_N_CTRL = 20


def hi4pi_fetch(threads):
    """All 192 tables into ROOT/hpx, full size only."""
    from concurrent.futures import ThreadPoolExecutor
    d = os.path.join(hi4pi_ROOT, "hpx")
    os.makedirs(d, exist_ok=True)

    def one(k):
        name = "HPX_%03d.fits" % k
        return name, core.download(hi4pi_BASE + name, os.path.join(d, name), size=hi4pi_SIZE)

    with ThreadPoolExecutor(threads) as ex:
        for name, ok in ex.map(one, range(1, 193)):
            print(name, "ok" if ok else "failed", flush=True)


class hi4pi_Sky:
    """Row access to the HEALPix tables through memory maps."""

    def __init__(self):
        from astropy_healpix import HEALPix
        from astropy.coordinates import Galactic
        self.hp = HEALPix(nside=hi4pi_NSIDE, order="ring", frame=Galactic())
        self.maps, self.order = {}, {}

    def _file(self, parent_ring):
        if parent_ring not in self.maps:
            from astropy_healpix.core import nested_to_ring, ring_to_nested
            p = os.path.join(hi4pi_ROOT, "hpx", "HPX_%03d.fits" % (parent_ring + 1))
            self.maps[parent_ring] = np.memmap(p, dtype=np.uint8, mode="r", offset=hi4pi_START, shape=(hi4pi_NROW, hi4pi_ROW))
            pn = int(ring_to_nested(parent_ring, 4))
            self.order[parent_ring] = np.sort(nested_to_ring(pn * hi4pi_NROW + np.arange(hi4pi_NROW), hi4pi_NSIDE).astype(np.int64))
        return self.maps[parent_ring], self.order[parent_ring]

    def spectrum(self, ring):
        from astropy_healpix.core import nested_to_ring, ring_to_nested
        parent = int(nested_to_ring(int(ring_to_nested(int(ring), hi4pi_NSIDE)) // hi4pi_NROW, 4))
        m, order = self._file(parent)
        i = int(np.searchsorted(order, ring))
        b = m[i]
        idx = int(np.frombuffer(b[:8].tobytes(), ">i8")[0])
        if idx != ring:
            raise ValueError("row %d of base pixel %d holds pixel %d, not %d" % (i, parent, idx, ring))
        return np.frombuffer(b[40:].tobytes(), ">f4").astype(float)

    def pixel(self, glon, b):
        import astropy.units as u
        return int(self.hp.lonlat_to_healpix(glon * u.deg, b * u.deg))

    def near(self, glon, b, radius_arcmin):
        import astropy.units as u
        return [int(x) for x in self.hp.cone_search_lonlat(glon * u.deg, b * u.deg, radius=radius_arcmin * u.arcmin)]

    def lonlat(self, ring):
        lo, la = self.hp.healpix_to_lonlat(ring)
        return float(lo.deg), float(la.deg)


def hi4pi_channel(v_kms):
    return v_kms / hi4pi_CDELT + hi4pi_CRPIX - 1.0


def hi4pi_residual(s):
    """Channel minus its quadratic prediction from two channels either side; spikes keep their height."""
    d = np.zeros_like(s)
    d[2:-2] = s[2:-2] - (4 * (s[1:-3] + s[3:-1]) - (s[:-4] + s[4:])) / 6.0
    return d


def hi4pi_zmax(r, c0, half):
    """Largest residual in the window over the outside robust scatter: (z, channel), or (nan, -1) at edges."""
    lo, hi = c0 - half, c0 + half + 1
    if lo < hi4pi_HALF_NOISE or hi > hi4pi_NCH - hi4pi_HALF_NOISE:
        return float("nan"), -1
    ring = np.r_[r[lo - hi4pi_HALF_NOISE:lo], r[hi:hi + hi4pi_HALF_NOISE]]
    sd = 1.4826 * np.median(np.abs(ring - np.median(ring)))
    if not sd > 0:
        return float("nan"), -1
    k = int(np.argmax(r[lo:hi]))
    return float(r[lo + k] / sd), lo + k


def hi4pi_beam_fit(sky, glon, b, ch):
    """Point-source fit within 15 arcmin: amplitude, error, plane reduced chi-square, pixel count."""
    pix = sky.near(glon, b, 15.0)
    vals, dx, dy = [], [], []
    cb = math.cos(math.radians(b))
    for p in pix:
        r = hi4pi_residual(sky.spectrum(p))
        vals.append(r[ch])
        pl, pb = sky.lonlat(p)
        dx.append(((pl - glon + 180) % 360 - 180) * cb * 60)
        dy.append((pb - b) * 60)
    vals, dx, dy = np.array(vals), np.array(dx), np.array(dy)
    g = np.exp(-4 * math.log(2) * (dx ** 2 + dy ** 2) / hi4pi_BEAM_ARCMIN ** 2)
    A = np.c_[g, np.ones_like(g), dx, dy]
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    res = vals - A @ coef
    dof = max(len(vals) - 4, 1)
    s2 = float(res @ res / dof)
    cov = s2 * np.linalg.pinv(A.T @ A)
    return float(coef[0]), float(math.sqrt(max(cov[0, 0], 0))), s2, len(vals)


def hi4pi_targets():
    path = os.path.join(hi4pi_ROOT, "targets.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows = core.gaia("SELECT source_id, ra, dec, l, b, parallax, pmra, pmdec, radial_velocity, "
                         "radial_velocity_error, phot_g_mean_mag, bp_rp FROM gaiadr3.gaia_source WHERE parallax > 20 "
                         "AND radial_velocity IS NOT NULL")
    vl = science.helio_to_lsrk([float(r["radial_velocity"]) for r in rows], [float(r["l"]) for r in rows],
                          [float(r["b"]) for r in rows])
    out = [dict(source_id=r["source_id"], l=float(r["l"]), b=float(r["b"]), plx=float(r["parallax"]),
                rv=float(r["radial_velocity"]), e_rv=float(r["radial_velocity_error"] or 1.0), v_lsrk=float(v),
                g=num(r["phot_g_mean_mag"], float("nan")), bp_rp=num(r["bp_rp"], float("nan")))
           for r, v in zip(rows, vl)]
    os.makedirs(hi4pi_ROOT, exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


hi4pi_RING = (20.0, 40.0)  # arcmin; beam at 1.5 per cent by 20, Galactic structure persists across 40
hi4pi_N_RING, hi4pi_N_SKY = 24, 4


def hi4pi_ring_pixels(sky, glon, b, rng):
    """Up to N_RING pixels in the ring around (l, b)."""
    inner = set(sky.near(glon, b, hi4pi_RING[0]))
    pix = [p for p in sky.near(glon, b, hi4pi_RING[1]) if p not in inner]
    if len(pix) > hi4pi_N_RING:
        pix = [int(x) for x in rng.choice(pix, hi4pi_N_RING, replace=False)]
    return pix


def hi4pi_differenced(sky, glon, b, rng):
    """Residual minus the ring median with ring scatter, floored by the quiet-channel scatter."""
    r0 = hi4pi_residual(sky.spectrum(sky.pixel(glon, b)))
    R = np.array([hi4pi_residual(sky.spectrum(p)) for p in hi4pi_ring_pixels(sky, glon, b, rng)])
    med = np.median(R, axis=0)
    sd = 1.4826 * np.median(np.abs(R - med), axis=0)
    diff = r0 - med
    v = (np.arange(hi4pi_NCH) + 1 - hi4pi_CRPIX) * hi4pi_CDELT
    # GASS spectra are blank beyond +-470 km/s; only finite channels set the floor
    quiet = (np.abs(v) > 200) & (np.arange(hi4pi_NCH) > 2) & (np.arange(hi4pi_NCH) < hi4pi_NCH - 3) & np.isfinite(diff)
    floor = 1.4826 * np.median(np.abs(diff[quiet] - np.median(diff[quiet]))) if quiet.sum() > 50 else np.nan
    return diff, np.fmax(sd, floor)


hi4pi_SPIKE = np.array([1 / 6.0, -2 / 3.0, 1.0, -2 / 3.0, 1 / 6.0])


def hi4pi_spike_shape(diff, ch):
    """Correlation of the differenced residual with a one-channel spike's pattern."""
    if ch < 2 or ch > hi4pi_NCH - 3:
        return float("nan")
    x = diff[ch - 2:ch + 3]
    if not np.all(np.isfinite(x)) or np.std(x) == 0:
        return float("nan")
    return float(np.corrcoef(x, hi4pi_SPIKE)[0, 1])


def hi4pi_window_max(diff, sd, c0, half):
    lo, hi = c0 - half, c0 + half + 1
    if lo < 3 or hi > hi4pi_NCH - 3:
        return float("nan"), -1
    z = diff[lo:hi] / sd[lo:hi]
    k = int(np.argmax(z))
    return float(z[k]), lo + k


def hi4pi_search():
    ts = hi4pi_targets()
    sky = hi4pi_Sky()
    rng = np.random.default_rng(7)
    res = []
    t0 = time.time()
    offs = [o for o in range(-200, 201, 10) if abs(o) >= 15]
    for n, t in enumerate(ts):
        c0 = int(round(hi4pi_channel(t["v_lsrk"])))
        half = int(min(max(1, math.ceil(2 * t["e_rv"] / hi4pi_CDELT)), 5))
        diff, sd = hi4pi_differenced(sky, t["l"], t["b"], rng)
        z, ch = hi4pi_window_max(diff, sd, c0, half)
        # the window elsewhere in the star's own spectrum
        ctrl_spec = []
        for o in rng.choice(offs, hi4pi_N_CTRL, replace=False):
            zc, cc = hi4pi_window_max(diff, sd, c0 + o, half)
            ctrl_spec.append([zc, hi4pi_spike_shape(diff, cc) if cc >= 0 else float("nan")])
        # the window at positions 1-2 deg away, each with its ring
        ctrl_sky = []
        for _ in range(hi4pi_N_SKY):
            ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(1.0, 2.0)
            bb = max(min(t["b"] + dist * math.sin(ang), 89.9), -89.9)
            ll = (t["l"] + dist * math.cos(ang) / max(math.cos(math.radians(bb)), 0.05)) % 360
            d2, s2 = hi4pi_differenced(sky, ll, bb, rng)
            zc, cc = hi4pi_window_max(d2, s2, c0, half)
            ctrl_sky.append([zc, hi4pi_spike_shape(d2, cc) if cc >= 0 else float("nan")])
        res.append(dict(t, channel=c0, half=half, z=z, ch=ch, shape=hi4pi_spike_shape(diff, ch) if ch >= 0 else float("nan"),
                        ctrl_spec=ctrl_spec, ctrl_sky=ctrl_sky))
        if n % 1000 == 0:
            print("%d of %d stars, %.0f s" % (n, len(ts), time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(hi4pi_HERE, "hi4pi_spikes.json"), "w"), default=float)
    hi4pi_judge(res, sky)


hi4pi_NARROW = 0.8


def hi4pi_judge(res, sky):
    """Counts above 4-6 sigma for stars and both control sets; narrow stars above 5 sigma with p and beam fit."""
    zs = np.array([r["z"] for r in res])
    sh = np.array([r.get("shape", np.nan) for r in res])
    cs = np.array([x for r in res for x in r["ctrl_spec"] if np.isfinite(x[0])])
    ck = np.array([x for r in res for x in r["ctrl_sky"] if np.isfinite(x[0])])
    ok = np.isfinite(zs)
    print("%d stars within 50 pc with Gaia radial velocities; %d with a finite statistic" % (len(res), ok.sum()))
    for thr in (4, 5, 6):
        print("  window maximum above %d sigma: stars %.4f, spectral controls %.4f, sky controls %.4f; narrow (shape "
              "above %.1f): stars %d (%.5f), spectral %.5f, sky %.5f"
              % (thr, np.mean(zs[ok] > thr), np.mean(cs[:, 0] > thr), np.mean(ck[:, 0] > thr), hi4pi_NARROW,
                 int(np.sum((zs[ok] > thr) & (sh[ok] > hi4pi_NARROW))), np.mean((zs[ok] > thr) & (sh[ok] > hi4pi_NARROW)),
                 np.mean((cs[:, 0] > thr) & (cs[:, 1] > hi4pi_NARROW)), np.mean((ck[:, 0] > thr) & (ck[:, 1] > hi4pi_NARROW))))
    cand = sorted([r for r in res if np.isfinite(r["z"]) and r["z"] > 5 and r.get("shape", 0) > hi4pi_NARROW],
                  key=lambda r: -r["z"])
    narrow_spec = cs[cs[:, 1] > hi4pi_NARROW, 0]
    narrow_sky = ck[ck[:, 1] > hi4pi_NARROW, 0]
    for r in cand[:40]:
        p_spec = (1 + np.sum(narrow_spec >= r["z"])) / (1.0 + len(cs))
        p_sky = (1 + np.sum(narrow_sky >= r["z"])) / (1.0 + len(ck))
        a, ea, chi2, npix = hi4pi_beam_fit(sky, r["l"], r["b"], r["ch"])
        r.update(p_spec=p_spec, p_sky=p_sky, beam_amp=a, beam_err=ea, beam_chi2=chi2, beam_npix=npix)
        print("  Gaia DR3 %s %.1f pc G %.2f l %.2f b %+.2f: %.1f sigma, shape %.2f, channel %d (v_lsrk %+.1f km/s, "
              "star %+.1f +- %.1f), p %.2g against its spectrum and %.2g against the sky; beam fit %.3f +- %.3f K"
              % (r["source_id"], 1000 / r["plx"], r["g"], r["l"], r["b"], r["z"], r["shape"], r["ch"],
                 (r["ch"] + 1 - hi4pi_CRPIX) * hi4pi_CDELT, r["v_lsrk"], r["e_rv"], p_spec, p_sky, a, ea))
    json.dump(cand, open(os.path.join(hi4pi_HERE, "hi4pi_candidates.json"), "w"), default=float, indent=1)


# beacons at rest in the CMB (+-2 channels) and GSR (+-3) frames
hi4pi_FRAMES = {"CMB": ("cmb", 2), "GSR": ("gsr", 3)}
hi4pi_EMISSION_K = 0.3  # K; windows above this are inside line emission


def hi4pi_frames():
    """Every pixel at the CMB- and GSR-frame 21 cm channel against channels 60 and 120 away, emission windows excluded."""
    offs = np.r_[np.arange(-hi4pi_HALF_NOISE, -5), np.arange(6, hi4pi_HALF_NOISE + 1)]
    out, counts = [], {}
    t0 = time.time()
    for parent in range(192):
        m = np.memmap(os.path.join(hi4pi_ROOT, "hpx", "HPX_%03d.fits" % (parent + 1)), dtype=np.uint8, mode="r",
                      offset=hi4pi_START, shape=(hi4pi_NROW, hi4pi_ROW))
        a = np.ascontiguousarray(m)
        ring = a[:, 0:8].copy().view(">i8").ravel().astype(np.int64)
        glon = a[:, 24:32].copy().view(">f8").ravel().astype(float)
        b = a[:, 32:40].copy().view(">f8").ravel().astype(float)
        s = a[:, 40:].copy().view(">f4").astype(np.float32)
        d = np.zeros_like(s)
        d[:, 2:-2] = s[:, 2:-2] - (4 * (s[:, 1:-3] + s[:, 3:-1]) - (s[:, :-4] + s[:, 4:])) / 6.0
        rows = np.arange(hi4pi_NROW)
        for name, (frame, half) in hi4pi_FRAMES.items():
            v = science.helio_to_lsrk(science.frame_rest_velocity(frame, glon, b), glon, b)
            c0 = np.rint(v / hi4pi_CDELT + hi4pi_CRPIX - 1).astype(int)
            for tag, shift in (("beacon", 0), ("control", 60), ("control", -60), ("control", 120), ("control", -120)):
                c = c0 + shift
                ok = (c - hi4pi_HALF_NOISE >= 2) & (c + hi4pi_HALF_NOISE < hi4pi_NCH - 2)
                win = np.clip(c[:, None] + np.arange(-half, half + 1)[None, :], 0, hi4pi_NCH - 1)
                noise = d[rows[:, None], np.clip(c[:, None] + offs[None, :], 0, hi4pi_NCH - 1)]
                med = np.median(noise, axis=1)
                sd = 1.4826 * np.median(np.abs(noise - med[:, None]), axis=1)
                w = d[rows[:, None], win]
                k = np.argmax(w, axis=1)
                # windows within 5 channels of hydrogen emission are excluded and counted untested
                around = s[rows[:, None], np.clip(c[:, None] + np.arange(-5, 6)[None, :], 0, hi4pi_NCH - 1)]
                clear = np.nanmax(np.abs(around), axis=1) < hi4pi_EMISSION_K
                z = np.where(ok & clear & (sd > 0), w[rows, k] / np.where(sd > 0, sd, 1), np.nan)
                key = (name, tag)
                cnt = counts.setdefault(key, [0, 0, 0, 0])
                good = np.isfinite(z)
                cnt[0] += int(good.sum())
                for j, thr in enumerate((4.0, 5.0, 6.0)):
                    cnt[j + 1] += int((z[good] > thr).sum())
                if tag == "beacon":
                    for i in np.where(good & (z > 5.0))[0]:
                        out.append(dict(frame=name, ring=int(ring[i]), l=float(glon[i]), b=float(b[i]), z=float(z[i]),
                                        ch=int(win[i, k[i]]), v_lsrk=float(v[i]), sd=float(sd[i]),
                                        amp=float(w[i, k[i]])))
        del a, s, d, m
        if parent % 16 == 15:
            print("%d of 192 base pixels, %d pixels above 5 sigma, %.0f s" % (parent + 1, len(out),
                                                                                  time.time() - t0), flush=True)
    for (name, tag), c in sorted(counts.items()):
        print("%s %s: %d pixel windows; above 4, 5 and 6 sigma %d, %d, %d (%.2e, %.2e, %.2e)"
              % (name, tag, c[0], c[1], c[2], c[3], c[1] / c[0], c[2] / c[0], c[3] / c[0]))
    json.dump(dict(counts={"%s %s" % k: v for k, v in counts.items()}, peaks=out),
              open(os.path.join(hi4pi_HERE, "hi4pi_frames.json"), "w"), indent=1)
    # a beacon fills the 16.2 arcmin beam
    sky = hi4pi_Sky()
    peaks = sorted(out, key=lambda p: -p["z"])
    for p in peaks[:60]:
        a_, ea, chi2, npix = hi4pi_beam_fit(sky, p["l"], p["b"], p["ch"])
        p.update(beam_amp=a_, beam_err=ea, beam_chi2=chi2, beam_npix=npix)
        print("  %s l %.3f b %+.3f: %.1f sigma (%.3f K) in channel %d (v_lsrk %+.1f); beam fit %.3f +- %.3f K over %d "
              "pixels" % (p["frame"], p["l"], p["b"], p["z"], p["amp"], p["ch"], p["v_lsrk"], a_, ea, npix), flush=True)
    json.dump(dict(counts={"%s %s" % k: v for k, v in counts.items()}, peaks=peaks),
              open(os.path.join(hi4pi_HERE, "hi4pi_frames.json"), "w"), indent=1)


def hi4pi_record():
    """Record the stars' windows and the whole-sky frame counts."""
    res = json.load(open(os.path.join(hi4pi_HERE, "hi4pi_spikes.json")))
    zs = np.array([r["z"] for r in res], float)
    sh = np.array([r.get("shape", np.nan) for r in res], float)
    ck = np.array([x for r in res for x in r["ctrl_sky"] if np.isfinite(x[0])], float)
    ok = np.isfinite(zs)
    star5 = float(np.mean((zs[ok] > 5) & (sh[ok] > hi4pi_NARROW)))
    sky5 = float(np.mean((ck[:, 0] > 5) & (ck[:, 1] > hi4pi_NARROW)))
    core.record_search(
        "hi4pi_star_frames", "Single-channel 21 cm lines at nearby stars' rest frequency, HI4PI",
        archive="HI4PI HEALPix spectra (CDS J/A+A/594/A116/HPX); Gaia DR3 radial velocities", release="HI4PI 2016",
        selection="Gaia DR3 stars within 50 pc with radial velocities", code="seti.py radio_sky hi4pi",
        params=dict(window="LSRK velocity +- 2 errors", ring_arcmin=list(hi4pi_RING), ring_pixels=hi4pi_N_RING, narrow=hi4pi_NARROW,
                    controls_spectral=hi4pi_N_CTRL, controls_sky=hi4pi_N_SKY),
        n_targets=int(ok.sum()),
        null_summary="narrow features above 5 sigma at %.3f per cent of stars against %.3f per cent of positions 1-2 "
                     "deg away at the same velocity" % (100 * star5, 100 * sky5),
        sensitivity="5 sigma of the ring-differenced residual, about 0.3 K or 0.5 Jy in 6.1 kHz",
        result="no excess over the sky controls; the narrow features are cold Galactic hydrogen",
        ledger="Single-channel 21 cm lines, HI4PI")
    core.record_targets("hi4pi_star_frames", [
        dict(target="gaia_dr3:%s" % r["source_id"], stat=None if not np.isfinite(r["z"]) else float(r["z"]),
             note="shape %.2f" % r["shape"] if np.isfinite(r.get("shape", np.nan)) else None) for r in res])
    fr_ = json.load(open(os.path.join(hi4pi_HERE, "hi4pi_frames.json")))
    c = fr_["counts"]
    core.record_search(
        "hi4pi_cmb_gsr_frames", "21 cm beacons at rest in the CMB frame and the Galactic standard of rest, HI4PI",
        archive="HI4PI HEALPix spectra (CDS J/A+A/594/A116/HPX)", release="HI4PI 2016",
        selection="every nside-1024 pixel whose frame channel lies clear of hydrogen emission above %.1f K" % hi4pi_EMISSION_K,
        code="seti.py radio_sky hi4pi", params=dict(frames=hi4pi_FRAMES, emission_k=hi4pi_EMISSION_K, controls="60 and 120 channels away"),
        n_targets=int(c["CMB beacon"][0] + c["GSR beacon"][0]),
        null_summary="above 6 sigma: CMB %.2e against %.2e in control channels, GSR %.2e against %.2e"
                     % (c["CMB beacon"][3] / c["CMB beacon"][0], c["CMB control"][3] / c["CMB control"][0],
                        c["GSR beacon"][3] / c["GSR beacon"][0], c["GSR control"][3] / c["GSR control"][0]),
        sensitivity="6 sigma of the leave-one-out residual, about 0.36 K or 0.56 Jy in 6.1 kHz",
        result="no frame-coherent excess", ledger="Single-channel 21 cm lines, HI4PI")
    print("recorded %d stars and the whole-sky frames" % ok.sum())


def hi4pi_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "search", "judge", "frames", "record"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.mode == "fetch":
        hi4pi_fetch(a.threads)
    elif a.mode == "search":
        hi4pi_search()
    elif a.mode == "frames":
        hi4pi_frames()
    elif a.mode == "record":
        hi4pi_record()
    else:
        hi4pi_judge(json.load(open(os.path.join(hi4pi_HERE, "hi4pi_spikes.json"))), hi4pi_Sky())


def hi4pi_cli():
    quiet_warnings()
    hi4pi_main()


# --------------------------------------------------------------------------------------------------------------

hi4pi_blind_DOC = r"""Unresolved narrowband sources in every HI4PI voxel on the GPU: point-like isolated groups beyond 6 sigma, negative groups as null.

    python -I seti.py radio_sky hi4pi_blind search
    python -I seti.py radio_sky hi4pi_blind vet [--top 60]
    python -I seti.py radio_sky hi4pi_blind record"""

hi4pi_blind_HERE = core.RESULTS
hi4pi_blind_Z = 6.0
hi4pi_blind_EMISSION_K = 0.3


def hi4pi_blind_file_voxels(parent, dev):
    """Voxels beyond +-Z in one base pixel's table: (ring index, l, b, channel, z, K)."""
    import torch
    m = np.memmap(os.path.join(hi4pi_ROOT, "hpx", "HPX_%03d.fits" % (parent + 1)), dtype=np.uint8, mode="r",
                  offset=hi4pi_START, shape=(hi4pi_NROW, hi4pi_ROW))
    a = np.ascontiguousarray(m)
    ring = a[:, 0:8].copy().view(">i8").ravel().astype(np.int64)
    glon = a[:, 24:32].copy().view(">f8").ravel().astype(float)
    b = a[:, 32:40].copy().view(">f8").ravel().astype(float)
    s = torch.from_numpy(a[:, 40:].copy().view(">f4").astype(np.float32)).to(dev)
    del a, m
    finite = torch.isfinite(s)
    s0 = torch.nan_to_num(s, nan=0.0)
    d = torch.zeros_like(s0)
    d[:, 2:-2] = s0[:, 2:-2] - (4 * (s0[:, 1:-3] + s0[:, 3:-1]) - (s0[:, :-4] + s0[:, 4:])) / 6.0
    # voxels near bright emission, blank channels or band edges
    bright = (s0.abs() > hi4pi_blind_EMISSION_K) | ~finite
    near = torch.nn.functional.max_pool1d(bright.float().unsqueeze(1), kernel_size=11, stride=1, padding=5).squeeze(1) > 0
    near[:, :3] = True
    near[:, -3:] = True
    clear = ~near
    # per-pixel robust residual scatter over clear channels
    dm = torch.where(clear, d, torch.full_like(d, float("nan")))
    med = torch.nanmedian(dm, dim=1, keepdim=True).values
    mad = torch.nanmedian((dm - med).abs(), dim=1, keepdim=True).values * 1.4826
    z = torch.where(clear, (d - med) / mad.clamp_min(1e-6), torch.zeros_like(d))
    out = {}
    for sign in (1, -1):
        idx = torch.nonzero(sign * z > hi4pi_blind_Z)
        rows, ch = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        zz = z[idx[:, 0], idx[:, 1]].cpu().numpy()
        kk = d[idx[:, 0], idx[:, 1]].cpu().numpy()
        out[sign] = [(int(ring[r]), float(glon[r]), float(b[r]), int(c), float(v), float(k))
                     for r, c, v, k in zip(rows, ch, zz, kk)]
    n_clear = int(clear.sum().item())
    del s, s0, d, dm, z, near, clear, bright, finite
    return out, n_clear


def hi4pi_blind_group(vox, radius_deg=0.25):
    """Positive voxels grouped by channel (+-1) within radius_deg: peak, size, extent."""
    vox = sorted(vox, key=lambda v: -abs(v[4]))
    used = np.zeros(len(vox), bool)
    arr = np.array([(v[1], v[2], v[3]) for v in vox], float) if vox else np.zeros((0, 3))
    groups = []
    for i, v in enumerate(vox):
        if used[i]:
            continue
        dl = ((arr[:, 0] - v[1] + 180) % 360 - 180) * math.cos(math.radians(v[2]))
        sel = (~used) & (np.abs(arr[:, 2] - v[3]) <= 1) & (np.hypot(dl, arr[:, 1] - v[2]) < radius_deg)
        members = np.nonzero(sel)[0]
        used[members] = True
        groups.append(dict(ring=v[0], l=v[1], b=v[2], ch=v[3], z=v[4], k=v[5], n=int(len(members)),
                           v_lsrk=(v[3] + 1 - hi4pi_CRPIX) * hi4pi_CDELT))
    return groups


def hi4pi_blind_search():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos, neg, clear = [], [], 0
    t0 = time.time()
    for parent in range(192):
        out, n = hi4pi_blind_file_voxels(parent, dev)
        pos += out[1]
        neg += out[-1]
        clear += n
        if parent % 16 == 15:
            print("%d of 192 base pixels: %d clear voxels, %d above +%.0f and %d below -%.0f sigma, %.0f s"
                  % (parent + 1, clear, len(pos), hi4pi_blind_Z, len(neg), hi4pi_blind_Z, time.time() - t0), flush=True)
    gp, gn = hi4pi_blind_group(pos), hi4pi_blind_group(neg)
    json.dump(dict(clear=clear, n_pos=len(pos), n_neg=len(neg), pos=gp, neg=gn),
              open(os.path.join(hi4pi_blind_HERE, "hi4pi_blind.json"), "w"))
    print("%d clear voxels; %d above +%.0f sigma in %d groups, %d below -%.0f sigma in %d groups"
          % (clear, len(pos), hi4pi_blind_Z, len(gp), len(neg), hi4pi_blind_Z, len(gn)))
    for size in (1, 2, 3, 5, 10):
        print("  groups of %d voxels or more: positive %d, negative %d"
              % (size, sum(1 for g in gp if g["n"] >= size), sum(1 for g in gn if g["n"] >= size)))


def hi4pi_blind_isolated(groups, g, radius_deg=5.0):
    """Other same-channel groups within radius_deg of g."""
    n = 0
    for o in groups:
        if o is g or o["ch"] != g["ch"]:
            continue
        dl = ((o["l"] - g["l"] + 180) % 360 - 180) * math.cos(math.radians(g["b"]))
        if math.hypot(dl, o["b"] - g["b"]) < radius_deg:
            n += 1
    return n


def hi4pi_blind_vet(top):
    """Beam-fit fraction, significance and same-channel neighbours of every group; candidates keep 0.7 of their peak at 5 sigma, alone."""
    res = json.load(open(os.path.join(hi4pi_blind_HERE, "hi4pi_blind.json")))
    sky = hi4pi_Sky()
    counts = {}
    for sign, key in ((1, "pos"), (-1, "neg")):
        gs = res[key]
        for g in gs:
            a, ea, chi2, npix = hi4pi_beam_fit(sky, g["l"], g["b"], g["ch"])
            a *= sign
            g.update(beam_amp=a, beam_err=ea, beam_chi2=chi2, beam_frac=a / abs(g["k"]) if g["k"] else 0.0,
                     neighbours=hi4pi_blind_isolated(gs, g))
        ok = [g for g in gs if g["beam_frac"] >= 0.7 and g["beam_amp"] >= 5 * g["beam_err"] and g["neighbours"] == 0]
        counts[key] = len(ok)
        res[key + "_candidates"] = sorted(ok, key=lambda g: -g["z"])
        print("%s groups: %d, point-like and alone in their channel: %d" % (key, len(gs), len(ok)), flush=True)
    for g in res["pos_candidates"][:top]:
        print("  l %8.3f b %+7.3f channel %d (v_lsrk %+7.1f km/s): peak %.1f sigma (%.3f K), %d voxels; beam fit %.3f "
              "+- %.3f K (%.2f of the peak)" % (g["l"], g["b"], g["ch"], g["v_lsrk"], g["z"], g["k"], g["n"],
                                                g["beam_amp"], g["beam_err"], g["beam_frac"]))
    json.dump(res, open(os.path.join(hi4pi_blind_HERE, "hi4pi_blind.json"), "w"))


def hi4pi_blind_record():
    """Record counts of both signs and one row per positive candidate."""
    res = json.load(open(os.path.join(hi4pi_blind_HERE, "hi4pi_blind.json")))
    pc, nc = res.get("pos_candidates", []), res.get("neg_candidates", [])
    core.record_search(
        "hi4pi_blind_voxels", "Unresolved narrowband sources anywhere in HI4PI",
        archive="HI4PI HEALPix spectra (CDS J/A+A/594/A116/HPX)", release="HI4PI 2016",
        selection="every voxel more than 5 channels from hydrogen emission above %.1f K" % hi4pi_blind_EMISSION_K,
        code="seti.py radio_sky hi4pi_blind", params=dict(z=hi4pi_blind_Z, emission_k=hi4pi_blind_EMISSION_K, group_radius_deg=0.25, beam_fraction=0.7,
                                           beam_sigma=5, same_channel_radius_deg=5),
        n_targets=int(res["clear"]),
        null_summary="%d voxels above +%.0f sigma in %d groups and %d below -%.0f in %d; point-like and alone in their "
                     "channel: %d positive against %d negative" % (res["n_pos"], hi4pi_blind_Z, len(res["pos"]), res["n_neg"], hi4pi_blind_Z,
                                                                   len(res["neg"]), len(pc), len(nc)),
        sensitivity="6 sigma of the leave-one-out residual, about 0.25-0.4 K or 0.4-0.6 Jy in 6.1 kHz",
        result="no excess of point-like positive sources over the negative null; the extended "
                                    "groups are faint high-velocity clouds", ledger="Single-channel 21 cm lines, HI4PI")
    core.clear_targets("hi4pi_blind_voxels")
    core.record_targets("hi4pi_blind_voxels", [
        dict(target="hi4pi:%d#ch%d" % (g["ring"], g["ch"]), stat=float(g["z"]), unit="robust deviations",
             note="l %.3f b %+.3f v_lsrk %+.1f km/s, %d voxels, beam fit %.2f of the peak"
                  % (g["l"], g["b"], g["v_lsrk"], g["n"], g["beam_frac"])) for g in pc])
    print("recorded")


def hi4pi_blind_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("search", "vet", "record"))
    p.add_argument("--top", type=int, default=60)
    a = p.parse_args()
    {"search": hi4pi_blind_search, "vet": lambda: hi4pi_blind_vet(a.top), "record": hi4pi_blind_record}[a.mode]()


def hi4pi_blind_cli():
    quiet_warnings()
    hi4pi_blind_main()


# --------------------------------------------------------------------------------------------------------------

vlass_DOC = r"""VLASS 3 GHz emission from inactive solar-type stars (Ayanabha et al. 2024), vetted for X-rays, youth, binarity, activity and background sources.

    python -I seti.py radio_sky vlass"""

vlass_HERE = core.RESULTS
vlass_ACTIVE = ("BY*", "RS*", "Er*", "Fl*", "TT*", "Y*O", "Y*?", "EB*", "SB*", "**", "El*", "WD*", "CV*", "Ae*", "Em*",
          "V*", "Ro*", "PM*?", "RG*", "AB*", "C*", "LP*", "Mi*", "BD*", "Pec*", "a2*", "Pu*", "X")


def vlass_main():
    t = core.vizier("J/AJ/168/288")[0]
    rows = [dict(zip(t.colnames, [r[c] for c in t.colnames])) for r in t]
    pos = [dict(source_id=str(r["GaiaDR3"]), ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"])) for r in rows]
    xr = {}
    for cat, rad, name in (("J/A+A/682/A34/erass1-m", 15.0, "eRASS1"), ("IX/69/xmm4d13s", 10.0, "4XMM"),
                           ("IX/70/csc21mas", 5.0, "CSC"), ("J/A+A/588/A103/cat2rxs", 30.0, "2RXS")):
        try:
            m = core.xmatch(pos, cat, rad)
        except Exception as e:
            print("%s: %s" % (name, str(e)[:80]), flush=True)
            m = []
        for r in m:
            xr.setdefault(r["source_id"], set()).add(name)
        print("%s: %d of %d stars within %.0f arcsec" % (name, len({r["source_id"] for r in m}), len(pos), rad), flush=True)
    g = {r["source_id"]: r for r in core.gaia(
        "SELECT g.source_id, g.ruwe, g.non_single_star, g.phot_variable_flag, g.bp_rp, g.phot_g_mean_mag, g.parallax, "
        "a.activityindex_espcs, a.classprob_dsc_combmod_star FROM gaiadr3.gaia_source AS g LEFT OUTER JOIN "
        "gaiadr3.astrophysical_parameters AS a ON g.source_id = a.source_id WHERE g.source_id IN (%s)"
        % ",".join(p["source_id"] for p in pos))}
    out = []
    for r, p in zip(rows, pos):
        sid = p["source_id"]
        x = g.get(sid, {})
        bprp, gm, plx = num(x.get("bp_rp")), num(x.get("phot_g_mean_mag")), num(x.get("parallax"))
        mg = gm + 5 * math.log10(plx / 100.0) if gm and plx and plx > 0 else None
        typ = str(r.get("Type") or "")
        why = []
        if sid in xr:
            why.append("X-ray source (%s)" % ", ".join(sorted(xr[sid])))
        if any(typ.strip().startswith(a) for a in vlass_ACTIVE) or typ.strip() in vlass_ACTIVE:
            why.append("SIMBAD type %s" % typ)
        if (num(x.get("ruwe")) or 0) > 1.4 or str(x.get("non_single_star")) not in ("0", "", "None"):
            why.append("binary (RUWE %s, NSS %s)" % (x.get("ruwe"), x.get("non_single_star")))
        if bprp is None or not 0.5 <= bprp <= 1.5:
            why.append("colour BP-RP %s outside F-early K" % (None if bprp is None else round(bprp, 2)))
        # main sequence: M_G within 1 mag of 4.3 (BP-RP - 0.8) + 4.9
        if mg is not None and bprp is not None and abs(mg - (4.3 * (bprp - 0.8) + 4.9)) > 1.0:
            why.append("off the main sequence (M_G %.1f)" % mg)
        out.append(dict(gaia=sid, simbad=str(r.get("Simbad") or ""), type=typ, cat=str(r.get("Category") or ""),
                        dist=num(r.get("Dist")), f1=num(r.get("Flux1")), f2=num(r.get("Flux2")), bp_rp=bprp, mg=mg,
                        ruwe=num(x.get("ruwe")), act=num(x.get("activityindex_espcs")), why=why))
    json.dump(out, open(os.path.join(vlass_HERE, "vlass_quiet.json"), "w"), indent=1)
    quiet = [o for o in out if not o["why"]]
    print("%d VLASS radio stars; %d inactive single main-sequence F to early K stars without X-rays" % (len(out), len(quiet)))
    for o in quiet:
        print("  Gaia DR3 %s %-22s %-8s %s: %.0f pc, BP-RP %.2f, M_G %.2f, RUWE %.2f, Ca IRT index %s, 3 GHz %s / %s mJy"
              % (o["gaia"], o["simbad"][:22], o["type"][:8], o["cat"][:12], o["dist"] or 0, o["bp_rp"], o["mg"],
                 o["ruwe"] or 0, o["act"], o["f1"], o["f2"]))


def vlass_cli():
    quiet_warnings()
    vlass_main()


# --------------------------------------------------------------------------------------------------------------

lowfreq_DOC = r"""LoTSS-DR2 and RACS-low sources on inactive F-K dwarfs within 100 pc, against 1 arcmin offset chance matches.

    python -I seti.py radio_sky lowfreq stars|match|judge
    python -I seti.py radio_sky lowfreq record"""

lowfreq_ROOT = os.environ.get("LOWFREQ_ROOT", core.data("lowfreq"))
lowfreq_HERE = core.RESULTS
lowfreq_SURVEYS = {"LoTSS-DR2": ("J/A+A/659/A1/catalog", 3.0), "RACS-low": ("J/other/PASA/38.58/galcut", 5.0)}
lowfreq_XRAY = (("J/A+A/682/A34/erass1-m", 15.0, "eRASS1"), ("IX/69/xmm4d13s", 10.0, "4XMM"), ("IX/70/csc21mas", 5.0, "CSC"),
        ("J/A+A/588/A103/cat2rxs", 30.0, "2RXS"))
lowfreq_ACTIVE = ("BY*", "RS*", "Er*", "Fl*", "TT*", "Y*O", "Y*?", "EB*", "SB*", "**", "El*", "WD*", "CV*", "Ae*", "Em*",
          "V*", "Ro*", "RG*", "AB*", "C*", "LP*", "Mi*", "BD*", "Pec*", "a2*", "Pu*", "X")


def lowfreq_stars():
    os.makedirs(lowfreq_ROOT, exist_ok=True)
    raw = core.tap_async("gaia", "SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp, ruwe FROM "
                                "gaiadr3.gaia_source WHERE parallax > 10 AND parallax_over_error > 10")
    rows = raw if isinstance(raw, list) else []
    json.dump(rows, open(os.path.join(lowfreq_ROOT, "stars.json"), "w"))
    print("%d stars within 100 pc" % len(rows))


def lowfreq_match():
    st = json.load(open(os.path.join(lowfreq_ROOT, "stars.json")))
    pos = [dict(source_id=str(r["source_id"]), ra=float(r["ra"]), dec=float(r["dec"])) for r in st]
    shifted = [dict(p, dec=p["dec"] + 1 / 60.0) for p in pos if p["dec"] < 89]
    out = {}
    for name, (cat, rad) in lowfreq_SURVEYS.items():
        res = {}
        for label, rows in (("stars", pos), ("shifted", shifted)):
            hits = []
            for k in range(0, len(rows), 50000):
                hits += core.xmatch(rows[k:k + 50000], cat, rad)
            res[label] = hits
            print("%s %s: %d matches" % (name, label, len(hits)), flush=True)
        out[name] = res
    json.dump(out, open(os.path.join(lowfreq_ROOT, "matches.json"), "w"), default=str)


def lowfreq_judge():
    st = {str(r["source_id"]): r for r in json.load(open(os.path.join(lowfreq_ROOT, "stars.json")))}
    m = json.load(open(os.path.join(lowfreq_ROOT, "matches.json")))
    ids = sorted({str(h["source_id"]) for s in m.values() for h in s["stars"]})
    pos = [dict(source_id=i, ra=float(st[i]["ra"]), dec=float(st[i]["dec"])) for i in ids]
    xray = set()
    for cat, rad, name in lowfreq_XRAY:
        try:
            xray |= {str(h["source_id"]) for h in core.xmatch(pos, cat, rad)}
        except Exception as e:
            print("%s: %s" % (name, str(e)[:80]), flush=True)
    types = {}
    for k in range(0, len(ids), 300):
        chunk = ids[k:k + 300]
        q = ("SELECT i.id, o.otype FROM ident AS i JOIN otypes AS o ON o.oidref = i.oidref WHERE i.id IN (%s)"
             % ", ".join("'Gaia DR3 %s'" % x for x in chunk))
        for r in core.simbad(q):
            types.setdefault(r["id"].split()[-1], set()).add(r["otype"])
    def dwarf(r):
        bp_rp, g, plx, ruwe = num(r["bp_rp"]), num(r["phot_g_mean_mag"]), num(r["parallax"]), num(r["ruwe"])
        if bp_rp is None or g is None or not plx:
            return False
        absg = g + 5 + 5 * math.log10(plx / 1000.0)
        return 0.4 <= bp_rp <= 1.2 and 2.5 + 3 * bp_rp <= absg + 1.5 and absg <= 3.5 + 3.5 * bp_rp and (ruwe or 0) <= 1.4
    out = {}
    for name, s in m.items():
        quiet = []
        for h in s["stars"]:
            i = str(h["source_id"])
            r = st[i]
            if dwarf(r) and i not in xray and not any(t in lowfreq_ACTIVE for t in types.get(i, ())):
                quiet.append(dict(source_id=i, sep=num(h.get("angDist")), g=num(r["phot_g_mean_mag"]),
                                  bp_rp=num(r["bp_rp"]), plx=num(r["parallax"])))
        # chance matches among stars passing the same cuts
        chance = sum(1 for h in s["shifted"] if dwarf(st[str(h["source_id"])]))
        dwarfs = sum(1 for h in s["stars"] if dwarf(st[str(h["source_id"])]))
        out[name] = dict(matches=len(s["stars"]), shifted=len(s["shifted"]), dwarfs=dwarfs, chance_dwarfs=chance,
                         quiet=quiet)
        print("%s: %d stars matched against %d at positions 1 arcmin away; F to early K dwarfs %d against %d by chance; "
              "%d of them inactive and single" % (name, len(s["stars"]), len(s["shifted"]), dwarfs, chance, len(quiet)))
    json.dump(out, open(os.path.join(lowfreq_HERE, "lowfreq_quiet.json"), "w"), indent=1, default=str)


def lowfreq_record():
    r = json.load(open(os.path.join(lowfreq_HERE, "lowfreq_quiet.json")))
    for name, (cat, rad) in lowfreq_SURVEYS.items():
        x = r[name]
        run = "lowfreq_quiet_%s" % name.split("-")[0].lower()
        core.record_search(
            run, "Low-frequency radio emission from inactive stars", archive="%s (VizieR %s)" % (name, cat),
            release=name, selection="Gaia DR3 stars with parallax above 10 mas at S/N 10, matched within %.0f arcsec"
                                    % rad,
            code="seti.py radio_sky lowfreq", params=dict(radius_arcsec=rad, shift_arcmin=1),
            n_targets=x["matches"],
            null_summary="%d stars matched against %d at shifted positions; F to early K dwarfs %d against %d by chance"
                         % (x["matches"], x["shifted"], x["dwarfs"], x["chance_dwarfs"]),
            sensitivity="the survey's detection limit at the star's position",
            result="%d inactive single F to early K dwarfs, as many as chance alignments give"
                                % len(x["quiet"]),
            ledger="Low-frequency radio emission from inactive stars")
        core.clear_targets(run)
        core.record_targets(run, [dict(target="gaia_dr3:%s" % q["source_id"], stat=q["sep"],
                                           unit="arcsec to the radio source", note="inactive F to early K dwarf")
                                      for q in x["quiet"]])
    print("recorded")


def lowfreq_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("stars", "match", "judge", "record"))
    a = p.parse_args()
    if a.mode == "record":
        lowfreq_record()
    else:
        {"stars": lowfreq_stars, "match": lowfreq_match, "judge": lowfreq_judge}[a.mode]()


def lowfreq_cli():
    quiet_warnings()
    lowfreq_main()


# --------------------------------------------------------------------------------------------------------------

sgl_DOC = r"""Unassociated RACS and VLASS components on the solar-gravitational-lens tracks of stars within 6.7 pc, against control positions.

    python -I seti.py radio_sky sgl targets
    python -I seti.py radio_sky sgl fetch
    python -I seti.py radio_sky sgl vlass
    python -I seti.py radio_sky sgl fit
    python -I seti.py radio_sky sgl record"""

sgl_HERE = core.RESULTS
sgl_ROOT = os.environ.get("SGL_ROOT", core.data("sgl"))
sgl_CATALOGUES = {
    "racs_low": dict(table="AS110.racs_dr1_gaussians_galacticcut_v2021_08_v02", mhz=887.5, time="obs_start_time"),
    "racs_mid": dict(table="AS110.racs_mid_components_v01", mhz=1367.5, time="scan_start_mjd"),
    "racs_high": dict(table="AS110.racs_high_components_v01", mhz=1655.5, time=None),
    "racs_low2": dict(table="AS110.racs_low2_v02", mhz=887.5, time=None),
}
sgl_RADIUS_DEG = 0.2
sgl_CONTROLS = ((3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0))
sgl_D_MIN = 550.0
sgl_TOL_ARCSEC = 3.0
sgl_ASSOC_ARCSEC = 5.0


def sgl_unit(ra, dec):
    return science.unit_vector(ra, dec)


def sgl_radec(v):
    v = np.asarray(v, float)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))


# ----------------------------------------------------------------------------------------------------------- targets


def sgl_targets():
    rows = core.simbad("SELECT main_id, ra, dec, pmra, pmdec, plx_value FROM basic WHERE plx_value > 150 "
                           "AND ra IS NOT NULL ORDER BY plx_value DESC")
    systems = []
    for r in rows:
        ra, dec = float(r["ra"]), float(r["dec"])
        for s in systems:
            if sep_arcsec(ra, dec, s["ra"], s["dec"]) < 3600.0:
                s["members"].append(r["main_id"])
                break
        else:
            systems.append(dict(name=r["main_id"], ra=ra, dec=dec, pmra=float(r["pmra"] or 0), pmdec=float(r["pmdec"] or 0),
                                plx=float(r["plx_value"]), members=[r["main_id"]]))
    os.makedirs(sgl_ROOT, exist_ok=True)
    json.dump(systems, open(os.path.join(sgl_ROOT, "targets.json"), "w"), indent=1)
    print("%d SIMBAD objects within 6.7 pc in %d systems" % (len(rows), len(systems)))


def sgl_star_dir(s, mjd):
    """Unit vector to the star at a date from J2000 position and proper motion."""
    yrs = (mjd - 51544.5) / 365.25
    return sgl_unit(s["ra"] + s["pmra"] / 3.6e6 * yrs / math.cos(math.radians(s["dec"])), s["dec"] + s["pmdec"] / 3.6e6 * yrs)


# ------------------------------------------------------------------------------------------------------------- fetch


def sgl_field_dates(ra, dec):
    """{(field, MHz): [MJD]} of RACS images covering a position."""
    q = ("SELECT t_min, em_min, filename FROM ivoa.obscore WHERE obs_collection = 'The Rapid ASKAP Continuum Survey' "
         "AND dataproduct_type = 'image' AND CONTAINS(POINT('ICRS', %f, %f), s_region) = 1" % (ra, dec))
    out = {}
    for r in core.tap("casda", q):
        m = re.search(r"RACS_\d{4}[+-]\d{2}", r["filename"])
        if not m or not r["t_min"] or not r["em_min"]:
            continue
        mhz = 299.792458 / float(r["em_min"])
        out.setdefault((m.group(0), round(mhz)), set()).add(round(float(r["t_min"]), 4))
    return {k: sorted(v) for k, v in out.items()}


def sgl_cone(cat, ra, dec):
    c = sgl_CATALOGUES[cat]
    cols = "ra, dec, e_ra, e_dec, peak_flux, maj_axis, s_code"
    cols += ", " + c["time"] if c["time"] else ", field_id"
    q = ("SELECT %s FROM %s WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %f, %f, %f))"
         % (cols, c["table"], ra, dec, sgl_RADIUS_DEG))
    return core.tap("casda", q)


def sgl_mjd_of(cat, row, dates):
    c = sgl_CATALOGUES[cat]
    if c["time"] == "scan_start_mjd":
        return float(row["scan_start_mjd"])
    if c["time"] == "obs_start_time":
        v = row["obs_start_time"]
        try:
            return float(v)
        except ValueError:
            from astropy.time import Time
            return float(Time(v.replace(" ", "T")).mjd)
    key = (row["field_id"], round(c["mhz"]))
    cands = dates.get(key, [])
    if not cands:
        return None
    # RACS-low and low2 share 887.5 MHz; low2 is the 2022 field
    if cat == "racs_low2":
        cands = [t for t in cands if t > 59500] or cands
    return cands[0] if len(cands) == 1 else float(np.median(cands))


def sgl_one_cone(job):
    """Dated components of every catalogue within the radius of one position."""
    name, k, ra, dec, dra, ddec = job
    try:
        dates = sgl_field_dates(ra, dec)
    except Exception:
        dates = {}
    comps = []
    for cat in sgl_CATALOGUES:
        try:
            rows = sgl_cone(cat, ra, dec)
        except Exception as e:
            print("%s %s: %s" % (name, cat, str(e)[:80]), flush=True)
            continue
        for r in rows:
            t = sgl_mjd_of(cat, r, dates)
            if t is None:
                continue
            comps.append(dict(cat=cat, ra=float(r["ra"]), dec=float(r["dec"]), e_ra=float(r["e_ra"] or 0),
                              e_dec=float(r["e_dec"] or 0), flux=float(r["peak_flux"] or 0), mjd=t,
                              code=r.get("s_code")))
    return dict(system=name, control=k, centre=[ra, dec], dra=dra, ddec=ddec, comps=comps)


def sgl_fetch(threads=4):
    from concurrent.futures import ThreadPoolExecutor
    systems = json.load(open(os.path.join(sgl_ROOT, "targets.json")))
    jobs = []
    for s in systems:
        anti = sgl_radec(-sgl_star_dir(s, 59400.0))
        for k, (dra, ddec) in enumerate([(0.0, 0.0)] + list(sgl_CONTROLS)):
            ra = (anti[0] + dra / max(math.cos(math.radians(anti[1])), 0.05)) % 360.0
            dec = float(np.clip(anti[1] + ddec, -89.9, 89.9))
            jobs.append((s["name"], k, ra, dec, dra, ddec))
    out = []
    with ThreadPoolExecutor(threads) as ex:
        for r in ex.map(sgl_one_cone, jobs):
            out.append(r)
            if r["control"] == 0:
                print("%s: antipode %.3f %+.3f, %d components" % (r["system"], r["centre"][0], r["centre"][1],
                                                                   len(r["comps"])), flush=True)
    json.dump(out, open(os.path.join(sgl_ROOT, "cones.json"), "w"))


# --------------------------------------------------------------------------------------------------------------- fit


def sgl_earth_helio(mjd):
    """Earth's heliocentric position (AU) at a date."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(mjd, format="mjd", scale="utc")
    e = get_body_barycentric("earth", t).xyz.to("au").value
    s = get_body_barycentric("sun", t).xyz.to("au").value
    return e - s


def sgl_predicted(axis, mjd, D):
    """Position seen from Earth of a point D AU along axis from the Sun."""
    return sgl_radec(D * axis - sgl_earth_helio(mjd))


def sgl_best_distance(axis, c):
    """(D, separation arcsec) of the track point closest to component c."""
    re_ = sgl_earth_helio(c["mjd"])
    grid = np.geomspace(sgl_D_MIN, 1e6, 400)
    seps = []
    for D in grid:
        ra, dec = sgl_radec(D * axis - re_)
        seps.append(sep_arcsec(ra, dec, c["ra"], c["dec"]))
    j = int(np.argmin(seps))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        s1 = sep_arcsec(*sgl_radec(m1 * axis - re_), c["ra"], c["dec"])
        s2 = sep_arcsec(*sgl_radec(m2 * axis - re_), c["ra"], c["dec"])
        if s1 < s2:
            hi = m2
        else:
            lo = m1
    D = 0.5 * (lo + hi)
    return D, sep_arcsec(*sgl_radec(D * axis - re_), c["ra"], c["dec"])


def sgl_fit():
    systems = {s["name"]: s for s in json.load(open(os.path.join(sgl_ROOT, "targets.json")))}
    cones = json.load(open(os.path.join(sgl_ROOT, "cones.json")))
    report = []
    for cn in cones:
        s = systems[cn["system"]]
        comps = cn["comps"]
        on = []
        for i, c in enumerate(comps):
            # a background source recurs in another epoch
            if any(o["cat"] != c["cat"] and sep_arcsec(o["ra"], o["dec"], c["ra"], c["dec"]) < sgl_ASSOC_ARCSEC
                   for o in comps):
                continue
            star = sgl_star_dir(s, c["mjd"])
            axis = -star
            if cn["control"]:
                ra0, dec0 = sgl_radec(axis)
                axis = sgl_unit((ra0 + cn["dra"] / max(math.cos(math.radians(dec0)), 0.05)) % 360.0,
                            float(np.clip(dec0 + cn["ddec"], -89.9, 89.9)))
            D, sep = sgl_best_distance(axis, c)
            tol = max(sgl_TOL_ARCSEC, 3.0 * math.hypot(c["e_ra"], c["e_dec"]))
            if sep < tol:
                on.append(dict(c, D=D, sep=sep, tol=tol, axis=axis.tolist()))
        chains = []
        for i in range(len(on)):
            for j in range(i + 1, len(on)):
                a, b = on[i], on[j]
                if a["cat"] == b["cat"]:
                    continue
                # one distance must place both
                for D in (a["D"], b["D"]):
                    sa = sep_arcsec(*sgl_predicted(np.array(a["axis"]), a["mjd"], D), a["ra"], a["dec"])
                    sb = sep_arcsec(*sgl_predicted(np.array(b["axis"]), b["mjd"], D), b["ra"], b["dec"])
                    if sa < a["tol"] and sb < b["tol"]:
                        chains.append(dict(D=D, members=[a, b]))
                        break
        report.append(dict(system=cn["system"], control=cn["control"], n_comps=len(comps), n_on=len(on),
                           on=on, chains=chains))
    json.dump(report, open(os.path.join(sgl_HERE, "sgl_parallax.json"), "w"), indent=1, default=float)
    real = [r for r in report if r["control"] == 0]
    ctrl = [r for r in report if r["control"] != 0]
    print("%d systems: %d components within %.1f deg of the antipodes, %d on a track and unassociated, %d chains"
          % (len(real), sum(r["n_comps"] for r in real), sgl_RADIUS_DEG, sum(r["n_on"] for r in real),
             sum(len(r["chains"]) for r in real)))
    print("%d control positions: %d components, %d on their tracks and unassociated, %d chains"
          % (len(ctrl), sum(r["n_comps"] for r in ctrl), sum(r["n_on"] for r in ctrl), sum(len(r["chains"]) for r in ctrl)))
    for r in real:
        for ch in r["chains"]:
            print("  CHAIN %s: D %.0f AU, %s" % (r["system"], ch["D"], "; ".join(
                "%s MJD %.2f %.5f %+.5f %.2f mJy sep %.1f\"" % (m["cat"], m["mjd"], m["ra"], m["dec"], m["flux"], m["sep"])
                for m in ch["members"])))


def sgl_record():
    report = json.load(open(os.path.join(sgl_HERE, "sgl_parallax.json")))
    real = [r for r in report if r["control"] == 0]
    ctrl = [r for r in report if r["control"] != 0]
    rows = [dict(target="simbad:" + r["system"], stat=len(r["chains"]), note="%d components, %d on track" % (r["n_comps"], r["n_on"]))
            for r in real]
    core.record_search("sgl_parallax_racs_vlass", "Receivers on the solar gravitational lens",
                           archive="CASDA RACS component catalogues (low, mid, high, low2); CIRADA VLASS Quick Look "
                                   "components, epochs 1 and 2", release="AS110; VLASS1QLv3.1, VLASS2QLv2",
                           selection="SIMBAD systems within 6.7 pc; antipodes and four control positions 3 deg away",
                           code=os.path.abspath(__file__), entry="sgl", params=dict(radius_deg=sgl_RADIUS_DEG, d_min_au=sgl_D_MIN,
                           tol_arcsec=sgl_TOL_ARCSEC, assoc_arcsec=sgl_ASSOC_ARCSEC), n_targets=len(rows),
                           null_summary=dict(control_chains=sum(len(r["chains"]) for r in ctrl), controls=len(ctrl)),
                           result="%d chains at the antipodes against %d at %d control positions"
                                  % (sum(len(r["chains"]) for r in real), sum(len(r["chains"]) for r in ctrl), len(ctrl)),
                           ledger="Receivers on the solar gravitational lens")
    core.record_targets("sgl_parallax_racs_vlass", rows)
    print("recorded %d systems" % len(rows))


sgl_VLASS = {"vlass1": ("CIRADA_VLASS1QLv3.1_table1_components.csv.gz", "CIRADA_VLASS1QLv3_table3_subtile.csv.gz"),
          "vlass2": ("CIRADA_VLASS2QLv2_table1_components.csv.gz", "CIRADA_VLASS2QLv1_table3_subtile.csv.gz")}


def sgl_vlass():
    """Add dated, unflagged VLASS components near every antipode and control to cones.json."""
    import pandas as pd
    from astropy.time import Time
    from scipy.spatial import cKDTree
    cones = json.load(open(os.path.join(sgl_ROOT, "cones.json")))
    cen = np.array([sgl_unit(c["centre"][0], c["centre"][1]) for c in cones])
    tree = cKDTree(cen)
    chord = 2 * math.sin(math.radians(sgl_RADIUS_DEG) / 2)
    vdir = os.path.join(sgl_ROOT, "vlass")
    for c in cones:
        c["comps"] = [x for x in c["comps"] if not x["cat"].startswith("vlass")]
    for cat, (comp_file, sub_file) in sgl_VLASS.items():
        sub = pd.read_csv(os.path.join(vdir, sub_file), low_memory=False)
        dcol = next(k for k in sub.columns if k.upper().replace("-", "_") in ("DATE_OBS", "DATEOBS"))
        # subtile names repeat between half-epochs; tile is part of the key
        when = dict(zip(zip(sub["Tile"].astype(str), sub["Subtile"].astype(str)),
                        Time(list(sub[dcol].astype(str).str.replace(" ", "T")), format="isot", scale="utc").mjd))
        n = 0
        for ch in pd.read_csv(os.path.join(vdir, comp_file), chunksize=500000, low_memory=False,
                              usecols=lambda k: k in ("RA", "DEC", "E_RA", "E_DEC", "Peak_flux", "S_Code", "Subtile", "Tile",
                                                      "Duplicate_flag", "Quality_flag", "Component_name")):
            ch = ch[(ch["Duplicate_flag"].fillna(0) < 2) & (ch["Quality_flag"].fillna(0) == 0)]
            ra, dec = np.radians(ch["RA"].to_numpy(float)), np.radians(ch["DEC"].to_numpy(float))
            v = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)
            hits = tree.query_ball_point(v, chord)
            for i, js in enumerate(hits):
                if not js:
                    continue
                r = ch.iloc[i]
                t = when.get((str(r["Tile"]), str(r["Subtile"])))
                if t is None:
                    continue
                for j in js:
                    cones[j]["comps"].append(dict(cat=cat, ra=float(r["RA"]), dec=float(r["DEC"]),
                                                  e_ra=float(r["E_RA"]) * 3600, e_dec=float(r["E_DEC"]) * 3600,
                                                  flux=float(r["Peak_flux"]), mjd=float(t), code=str(r["S_Code"])))
                    n += 1
        print("%s: %d components within %.1f deg of the antipodes and controls" % (cat, n, sgl_RADIUS_DEG), flush=True)
    json.dump(cones, open(os.path.join(sgl_ROOT, "cones.json"), "w"))


def sgl_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("targets", "fetch", "vlass", "fit", "record"))
    a = p.parse_args()
    {"targets": sgl_targets, "fetch": sgl_fetch, "vlass": sgl_vlass, "fit": sgl_fit, "record": sgl_record}[a.mode]()


def sgl_cli():
    quiet_warnings()
    sgl_main()


# --------------------------------------------------------------------------------------------------------------

pulsars_DOC = r"""AllWISE warm counterparts of ATNF pulsars against offset chance rates, and nearby Gaia stars at bright-pulsar antipodes with BL coverage.

    python -I seti.py radio_sky pulsars ir
    python -I seti.py radio_sky pulsars antipodes"""

pulsars_ROOT = os.environ.get("PULSAR_ROOT", core.data("pulsars"))
pulsars_HERE = core.RESULTS
pulsars_WISE_EPOCH_MJD = 55377.0  # 2010.5
pulsars_SHIFTS = [(60 * math.cos(a), 60 * math.sin(a)) for a in np.arange(8) * math.pi / 4]


def pulsars_sexa(v, hours):
    s = v.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    parts = [float(x) for x in s.lstrip("+-").split(":")]
    while len(parts) < 3:
        parts.append(0.0)
    val = parts[0] + parts[1] / 60 + parts[2] / 3600
    return sign * val * (15.0 if hours else 1.0)


def pulsars_last_unit(v):
    """Size of the last quoted digit of a sexagesimal value."""
    s = v.strip().split(":")[-1]
    return 10.0 ** -(len(s.split(".")[1]) if "." in s else 0)


def pulsars_pulsars():
    out = []
    for x in json.load(open(os.path.join(pulsars_ROOT, "psrcat.json"))):
        if "RAJ" not in x or "DECJ" not in x:
            continue
        try:
            ra, dec = pulsars_sexa(x["RAJ"][0], True), pulsars_sexa(x["DECJ"][0], False)
            era = float(x["RAJ"][1]) * pulsars_last_unit(x["RAJ"][0]) * 15.0 * math.cos(math.radians(dec)) if len(x["RAJ"]) > 1 else 99.0
            edec = float(x["DECJ"][1]) * pulsars_last_unit(x["DECJ"][0]) if len(x["DECJ"]) > 1 else 99.0
        except (ValueError, IndexError):
            continue
        def field(k):
            return float(x[k][0]) if k in x and x[k][0] not in ("*",) else None
        out.append(dict(name=x["PSRJ"][0], ra=ra, dec=dec, err=math.hypot(era, edec), pmra=field("PMRA") or 0.0,
                        pmdec=field("PMDEC") or 0.0, posepoch=field("POSEPOCH"), s1400=field("S1400"),
                        dist=field("DIST") or field("DIST_DM"), p0=field("P0"), assoc=" ".join(x.get("ASSOC", [])),
                        type=" ".join(x.get("TYPE", [])), bincomp=" ".join(x.get("BINCOMP", []))))
    return out


def pulsars_at_epoch(p, mjd):
    if p["posepoch"] is None:
        return p["ra"], p["dec"]
    yrs = (mjd - p["posepoch"]) / 365.25
    return (p["ra"] + p["pmra"] / 3.6e6 * yrs / math.cos(math.radians(p["dec"])), p["dec"] + p["pmdec"] / 3.6e6 * yrs)


def pulsars_warm(x):
    """Warm ring signature: A or B quality at 12 or 22 um without artifacts."""
    q, c = str(x.get("qph") or "UUUU"), str(x.get("ccf") or "0000")
    return any(q[i] in "AB" and c[i] == "0" for i in (2, 3))


def pulsars_no_gaia(xs, radius=2.0):
    """Matches without a Gaia source within radius."""
    if not xs:
        return []
    probe = [dict(source_id=str(i), ra=float(x["RAJ2000"]), dec=float(x["DEJ2000"])) for i, x in enumerate(xs)]
    hit = {r["source_id"] for r in core.xmatch(probe, "I/355/gaiadr3", radius)}
    return [x for i, x in enumerate(xs) if str(i) not in hit]


def pulsars_ring_like(matches, sep=1.0):
    best = {}
    for x in matches:
        if x["source_id"] not in best or float(x["angDist"]) < float(best[x["source_id"]]["angDist"]):
            best[x["source_id"]] = x
    close = [x for x in best.values() if float(x["angDist"]) < sep and pulsars_warm(x)]
    return best, pulsars_no_gaia(close)


def pulsars_ir():
    ps = [p for p in pulsars_pulsars() if p["err"] < 1.0]
    rows = []
    for p in ps:
        ra, dec = pulsars_at_epoch(p, pulsars_WISE_EPOCH_MJD)
        rows.append(dict(source_id=p["name"], ra=ra, dec=dec))
    m = core.xmatch(rows, "II/328/allwise", 3.0)
    null, null_ring = 0, 0
    for dx, dy in pulsars_SHIFTS:
        sh = [dict(source_id=r["source_id"], ra=r["ra"] + dx / 3600.0 / max(math.cos(math.radians(r["dec"])), 0.05),
                   dec=r["dec"] + dy / 3600.0) for r in rows]
        nm = core.xmatch(sh, "II/328/allwise", 3.0)
        nb, nr = pulsars_ring_like(nm)
        null += len(nb)
        null_ring += len(nr)
    _, rings = pulsars_ring_like(m)
    print("ring-like (within 1 arcsec, warm at 12 or 22 um without artifact flags, no Gaia source within 2 arcsec): %d "
          "pulsars against %.2f per shifted copy" % (len(rings), null_ring / len(pulsars_SHIFTS)))
    for x in rings:
        print("  RING-LIKE %s %.2f\" W1 %s W2 %s W3 %s W4 %s qph %s ccf %s" % (x["source_id"], float(x["angDist"]), x.get("W1mag"),
                                                                           x.get("W2mag"), x.get("W3mag"), x.get("W4mag"),
                                                                           x.get("qph"), x.get("ccf")))
    by = {p["name"]: p for p in ps}
    hits = {}
    for x in m:
        if x["source_id"] not in hits or float(x["angDist"]) < float(hits[x["source_id"]]["angDist"]):
            hits[x["source_id"]] = x
    print("%d pulsars with positions good to 1 arcsec; %d with an AllWISE source within 3 arcsec, %.1f expected by "
          "chance (8 shifted copies)" % (len(ps), len(hits), null / len(pulsars_SHIFTS)))
    out = []
    for name, x in sorted(hits.items(), key=lambda kv: float(kv[1]["angDist"])):
        p = by[name]
        w = {k: x.get(k) for k in ("W1mag", "W2mag", "W3mag", "W4mag", "e_W3mag", "e_W4mag", "ccf", "ex", "qph")}
        known = ", ".join(t for t in (p["assoc"], p["type"], ("companion " + p["bincomp"]) if p["bincomp"] else "") if t)
        out.append(dict(pulsar=name, sep=float(x["angDist"]), wise=w, assoc=p["assoc"], type=p["type"], bincomp=p["bincomp"],
                        dist_kpc=p["dist"], p0=p["p0"]))
        print("  %-12s %.2f\" W1 %s W2 %s W3 %s W4 %s qph %s ccf %s; %s" % (name, float(x["angDist"]), w["W1mag"], w["W2mag"],
                                                                         w["W3mag"], w["W4mag"], w["qph"], w["ccf"],
                                                                         known or "no association, companion or class"))
    json.dump(dict(n=len(ps), hits=out, chance=null / len(pulsars_SHIFTS)), open(os.path.join(pulsars_HERE, "pulsar_ir.json"), "w"), indent=1)


def pulsars_antipodes():
    ps = [p for p in pulsars_pulsars() if p["s1400"] and "GC:" not in p["assoc"] and "EXGAL" not in p["assoc"]]
    ps = sorted(ps, key=lambda p: -p["s1400"])[:100]
    bl = set(json.load(open(os.path.join(pulsars_HERE, "bl_targets.json"))))
    bl_norm = {b.replace(" ", "").upper() for b in bl}
    out = []
    for p in ps:
        ra, dec = (p["ra"] + 180.0) % 360.0, -p["dec"]
        q = ("SELECT g.source_id, g.ra, g.dec, g.parallax, g.phot_g_mean_mag, g.bp_rp, h.original_ext_source_id AS hip "
             "FROM gaiadr3.gaia_source AS g LEFT OUTER JOIN gaiadr3.hipparcos2_best_neighbour AS h "
             "ON g.source_id = h.source_id WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', %f, %f, 1.0)) "
             "AND g.parallax > 5 AND g.parallax_over_error > 5" % (ra, dec))
        try:
            stars = core.gaia(q)
        except Exception as e:
            print(p["name"], "Gaia query failed", str(e)[:60])
            continue
        for s in stars:
            hip = ("HIP" + s["hip"]) if s.get("hip") else None
            s["bl"] = bool(hip and hip.upper() in bl_norm)
            s["hip"] = hip
        out.append(dict(pulsar=p["name"], s1400=p["s1400"], p0=p["p0"], antipode=[ra, dec], stars=stars))
        if stars:
            print("%-12s S1400 %6.1f mJy P %.4f s: %d stars within 1 deg of the antipode and 200 pc%s" % (
                p["name"], p["s1400"], p["p0"] or 0, len(stars), "; " + ", ".join(
                    "%s%s" % (s["hip"], " (BL)" if s["bl"] else "") for s in stars if s["hip"]) if any(s["hip"] for s in stars) else ""))
    json.dump(out, open(os.path.join(pulsars_HERE, "pulsar_antipodes.json"), "w"), indent=1)
    n = sum(len(o["stars"]) for o in out)
    print("%d antipodes, %d stars within 1 deg and 200 pc, %d of them Hipparcos stars, %d observed by Breakthrough Listen"
          % (len(out), n, sum(1 for o in out for s in o["stars"] if s["hip"]), sum(1 for o in out for s in o["stars"] if s["bl"])))


def pulsars_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("ir", "antipodes"))
    a = p.parse_args()
    {"ir": pulsars_ir, "antipodes": pulsars_antipodes}[a.mode]()


def pulsars_cli():
    quiet_warnings()
    pulsars_main()


# --------------------------------------------------------------------------------------------------------------

wind_DOC = r"""Persistent narrowband lines below the ionospheric cutoff in Wind/WAVES and STEREO spectra, classified by diurnal modulation, broadcast correlation and spacecraft coincidence.

    python -I seti.py radio_sky wind fetch YEAR [--threads 4]
    python -I seti.py radio_sky wind analyse YEAR
    python -I seti.py radio_sky wind daily YEAR [--freq KHZ ...]
    python -I seti.py radio_sky wind stereo YEAR [--freq KHZ ...] [--day0 1] [--days 10]
    python -I seti.py radio_sky wind stereo_year YEAR [--craft ahead|behind] [--threads 2]
    python -I seti.py radio_sky wind classify YEAR                   (again from the year's saved line occupancies)
    python -I seti.py radio_sky wind rad1 YEAR                       (RAD1 among its measured channels alone)
    python -I seti.py radio_sky wind rad1_record YEAR [YEAR ...]
    python -I seti.py radio_sky wind stereo_classify YEAR [--craft ahead|behind]
    python -I seti.py radio_sky wind stereo_record YEAR [YEAR ...]   (the STEREO years classified, into the coverage store)
    python -I seti.py radio_sky wind record YEAR [YEAR ...]          (the years analysed, into the coverage store)"""

wind_ROOT = os.environ.get("WIND_ROOT", core.data("wind"))
wind_UA = core.UA
# known persistent-line sources (kHz): shortwave bands, AKR, solar-wind plasma line
wind_SW_BANDS = ((2300, 2495), (3200, 3400), (3900, 4000), (4750, 5060), (5900, 6200), (7200, 7450), (9400, 9900),
            (11600, 12100), (13570, 13870))
# ITU maritime and aeronautical HF allocations (kHz)
wind_MARITIME = ((4063, 4438), (6200, 6525), (8195, 8815), (12230, 13200))
wind_AERONAUTICAL = ((2850, 3155), (3400, 3500), (4650, 4750), (5450, 5730), (6525, 6765), (8815, 9040), (10005, 10100),
                (11175, 11400), (13200, 13360))
# amateur allocations below 13.8 MHz
wind_AMATEUR = ((1800, 2000), (3500, 4000), (5351, 5367), (7000, 7300), (10100, 10150))
# broadcasters work up to 200 kHz outside the bands
wind_OUT_OF_BAND = 200
# UT-hour modulation above the mean marks terrestrial sources
wind_DIURNAL = 1.0


# lines made on Wind: 8025 and 1175 kHz
wind_WIND_LOCAL = (8025.0, 1175.0)
# spacecraft anomaly excluded, 2014 Oct 26 to 2015 Jan 30
wind_BAD_PERIODS = ((date(2014, 10, 26), date(2015, 1, 30)),)


def wind_bad_day(name):
    """True for a WI_H1_WAV file in a listed anomaly."""
    try:
        s = name.split("_")[3]
        d = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (IndexError, ValueError):
        return False
    return any(a <= d <= b for a, b in wind_BAD_PERIODS)


def wind_known(f_khz, modulation=None):
    if any(abs(f_khz - x) < 1.0 for x in wind_WIND_LOCAL):
        return "Wind's own line"
    if any(lo - 25 <= f_khz <= hi + 25 for lo, hi in wind_SW_BANDS):
        return "shortwave broadcast band"
    if 100 <= f_khz <= 700:
        return "auroral kilometric radiation range"
    if f_khz < 100:
        return "solar-wind plasma line range"
    if any(lo - 25 <= f_khz <= hi + 25 for lo, hi in wind_MARITIME):
        return "maritime mobile band"
    if any(lo - 25 <= f_khz <= hi + 25 for lo, hi in wind_AERONAUTICAL):
        return "aeronautical mobile band"
    if any(lo <= f_khz <= hi for lo, hi in wind_AMATEUR):
        return "amateur band"
    if any(lo - wind_OUT_OF_BAND <= f_khz <= hi + wind_OUT_OF_BAND for lo, hi in wind_SW_BANDS):
        return "out-of-band broadcasting"
    if modulation is not None and modulation > wind_DIURNAL:
        return "terrestrial (24 h modulation)"
    return None


def wind_fetch_day(day):
    name = "wi_h1_wav_%s_v01.cdf" % day.strftime("%Y%m%d")
    path = os.path.join(wind_ROOT, str(day.year), name)
    if os.path.exists(path):
        return 1
    url = "https://cdaweb.gsfc.nasa.gov/pub/data/wind/waves/wav_h1/%d/%s" % (day.year, name)
    for attempt in range(6):
        try:
            b = urllib.request.urlopen(urllib.request.Request(url, headers=wind_UA), timeout=180).read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path + ".tmp", "wb").write(b)
            os.replace(path + ".tmp", path)
            return 1
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 0
        except Exception:
            pass
        time.sleep(10 * 2 ** attempt)
    return 0


def wind_fetch(year, threads):
    """The year's daily files in two passes; missing days printed."""
    days = [date(year, 1, 1) + timedelta(k) for k in range(366) if (date(year, 1, 1) + timedelta(k)).year == year]
    for _ in range(2):
        with ThreadPoolExecutor(threads) as ex:
            got = list(ex.map(wind_fetch_day, days))
        days = [d for d, g in zip(days, got) if not g]
    n = len([p for p in os.listdir(os.path.join(wind_ROOT, str(year))) if p.endswith(".cdf")]) \
        if os.path.isdir(os.path.join(wind_ROOT, str(year))) else 0
    print("%d: %d daily files; missing %s" % (year, n, ", ".join(d.isoformat() for d in days) or "none"))


def wind_analyse(year):
    import cdflib
    from scipy.ndimage import median_filter
    d = os.path.join(wind_ROOT, str(year))
    occ = {}
    hourly = {}
    days_with = {}
    interp = {}
    nmin = 0
    freqs = {}
    skipped = 0
    for p in sorted(os.listdir(d)):
        if not p.endswith(".cdf"):
            continue
        if wind_bad_day(p):
            skipped += 1
            continue
        try:
            c = cdflib.CDF(os.path.join(d, p))
            ep = cdflib.cdfepoch.to_datetime(c.varget("Epoch"))
        except Exception:
            continue
        hours = np.array([int(str(x)[11:13]) for x in ep])
        for band in ("RAD1", "RAD2"):
            v = np.asarray(c.varget("E_VOLTAGE_" + band), float)
            f = np.asarray(c.varget("Frequency_" + band), float)
            freqs[band] = f
            v = np.where(v > 0, v, np.nan)
            # unmeasured channels carry their neighbours' mean (list mode)
            mid = 0.5 * (v[:, :-2] + v[:, 2:])
            ip = np.zeros(v.shape)
            ip[:, 1:-1] = np.abs(v[:, 1:-1] - mid) < 1e-3 * np.abs(v[:, 1:-1])
            interp.setdefault(band, np.zeros(len(f)))
            interp[band] += np.nansum(ip, axis=0)
            lv = np.log(v)
            base = median_filter(np.nan_to_num(lv, nan=np.nanmedian(lv)), size=(1, 9), mode="nearest")
            r = lv - base
            mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r, axis=1, keepdims=True)), axis=1, keepdims=True)
            hit = r > 5 * np.maximum(mad, 1e-3)
            occ.setdefault(band, np.zeros(len(f)))
            occ[band] += np.nansum(hit, axis=0)
            hourly.setdefault(band, np.zeros((24, len(f))))
            for h in range(24):
                hourly[band][h] += np.nansum(hit[hours == h], axis=0)
            days_with.setdefault(band, np.zeros(len(f)))
            days_with[band] += (np.nansum(hit, axis=0) > 60)
        nmin += len(ep)
    print("%d: %d minutes of spectra, %.1f days (%d days in spacecraft anomalies left out)"
          % (year, nmin, nmin / 1440.0, skipped))
    for band in ("RAD1", "RAD2"):
        np.savez(os.path.join(wind_ROOT, "lines_%s_%d.npz" % (band, year)), freq=freqs[band], occupancy=occ[band] / nmin,
                 hourly=hourly[band], days=days_with[band], interp=interp[band] / nmin, minutes=nmin, skipped=skipped)
    wind_classify(year)


def wind_classify(year):
    """Classify an analysed year's persistent lines against known sources and day-to-day tests."""
    summary, nmin, skipped = {}, None, None
    for band in ("RAD1", "RAD2"):
        z = np.load(os.path.join(wind_ROOT, "lines_%s_%d.npz" % (band, year)))
        f, o, hp_all, days = z["freq"], z["occupancy"], z["hourly"], z["days"]
        if "interp" not in z.files:
            # RAD1 lines need the list-mode fill fractions
            raise SystemExit("%s %d: the saved occupancies lack the list-mode fractions; analyse the year from its "
                             "spectra" % (band, year))
        ifrac = z["interp"]
        if "minutes" in z.files:
            nmin, skipped = int(z["minutes"]), int(z["skipped"])
        days_with = {band: days}
        top = np.argsort(-o)[:15]
        print("%s: channels with the highest line occupancy" % band)
        for j in top:
            hp = hp_all[:, j]
            print("  %8.1f kHz: occupancy %.3f, days with an hour or more %d, 24 h modulation (max-min)/mean %.2f, "
                  "peak UT hour %d" % (f[j], o[j], int(days[j]), (hp.max() - hp.min()) / max(hp.mean(), 1e-9),
                                       int(np.argmax(hp))))
        mod = (hp_all.max(axis=0) - hp_all.min(axis=0)) / np.maximum(hp_all.mean(axis=0), 1e-9)
        pers = [j for j in np.argsort(-o) if o[j] > 0.01 and days[j] >= 30]

        def why_of(j):
            # measured channels beside interpolated ones stand out spuriously
            if j > 0 and j < len(f) - 1 and (min(ifrac[j - 1], ifrac[j + 1]) > 0.5 or ifrac[j] > 0.5):
                return "list-mode channel (neighbours interpolated %.0f and %.0f per cent of minutes)" % (
                    100 * ifrac[j - 1], 100 * ifrac[j + 1])
            return wind_known(f[j], mod[j])
        for j in pers:
            why = why_of(j)
            if why:
                print("  %8.1f kHz: occupancy %.3f, %d days, 24 h modulation %.2f: %s"
                      % (f[j], o[j], int(days_with[band][j]), mod[j], why))
        explained = {float(f[j]): why_of(j) for j in pers if why_of(j)}
        rest = [j for j in pers if not why_of(j)]
        if band == "RAD2" and rest:
            # terrestrial leakage tracks the broadcast lines day to day; receiver lines anticorrelate
            path = os.path.join(wind_ROOT, "daily_%d.json" % year)
            saved = json.load(open(path)) if os.path.exists(path) else {}
            dl = {float(k): v for k, v in saved.items()}
            need = [float(f[j]) for j in rest if float(f[j]) not in dl]
            if need:
                dl.update(wind_daily(year, need, wind_REFS))
                json.dump({"%.1f" % k: v for k, v in dl.items()}, open(path, "w"))
            for j in rest:
                x = dl[float(f[j])]
                if x["p"] < 1e-3 and abs(x["rho_ref"]) > 0.3:
                    why = ("terrestrial leakage" if x["rho_ref"] > 0 else "the receiver's own line") + \
                          " (day-to-day Spearman %.2f with the broadcast lines)" % x["rho_ref"]
                    explained[float(f[j])] = why
                    print("  %8.1f kHz: occupancy %.3f, %d days: %s" % (f[j], o[j], int(days_with[band][j]), why))
            rest = [j for j in rest if float(f[j]) not in explained]
        # a channel diurnal in another year is terrestrial
        diurnal = wind_diurnal_years(band, year, f)
        for j in rest:
            if diurnal.get(j):
                why = "terrestrial (24 h modulation in %s)" % ", ".join(str(y) for y in diurnal[j][:4])
                explained[float(f[j])] = why
                print("  %8.1f kHz: occupancy %.3f, %d days, 24 h modulation %.2f: %s"
                      % (f[j], o[j], int(days_with[band][j]), mod[j], why))
        rest = [j for j in rest if float(f[j]) not in explained]
        print("%s: persistent lines (occupancy above 1 per cent on 30 days or more) outside the known sources: %d"
              % (band, len(rest)))
        for j in rest[:15]:
            print("  %8.1f kHz: occupancy %.3f, %d days, 24 h modulation %.2f"
                  % (f[j], o[j], int(days_with[band][j]), mod[j]))
        summary[band] = dict(persistent=len(pers), explained=explained,
                             left=[dict(khz=float(f[j]), occupancy=float(o[j]), days=int(days_with[band][j]),
                                        modulation=float(mod[j])) for j in rest])
    if nmin is None:
        nmin = 1440 * (date(year + 1, 1, 1) - date(year, 1, 1)).days
    json.dump(dict(year=year, minutes=nmin, skipped_days=skipped, **summary),
              open(core.results("wind_%d.json" % year), "w"), indent=1)


def wind_diurnal_years(band, year, f):
    """{channel: [years]} of channels diurnal at 1 per cent occupancy on 30 days in another year."""
    out = {}
    for p in sorted(os.listdir(wind_ROOT)):
        m = re.match(r"lines_%s_(\d{4})\.npz$" % band, p)
        if not m or int(m.group(1)) == year:
            continue
        z = np.load(os.path.join(wind_ROOT, p))
        if len(z["freq"]) != len(f):
            continue
        h = z["hourly"]
        mod = (h.max(axis=0) - h.min(axis=0)) / np.maximum(h.mean(axis=0), 1e-9)
        for j in np.nonzero((z["occupancy"] > 0.01) & (z["days"] >= 30) & (mod > wind_DIURNAL))[0]:
            out.setdefault(int(j), []).append(int(m.group(1)))
    return out


def wind_record(years):
    """Record each analysed year."""
    for y in years:
        d = json.load(open(core.results("wind_%d.json" % y)))
        left = d["RAD1"]["left"] + d["RAD2"]["left"]
        expl = {**d["RAD1"]["explained"], **d["RAD2"]["explained"]}
        own = sorted(float(k) for k, v in expl.items() if "own line" in v)
        core.record_search(
            "wind_waves_%d" % y, "Persistent lines below the ionospheric cutoff, Wind/WAVES",
            archive="CDAWeb WI_H1_WAV (Wind/WAVES RAD1, RAD2)", release="v01", selection="every minute of %d" % y,
            code=os.path.abspath(__file__), entry="wind", params=dict(line_mad=5, occupancy=0.01, days=30,
                                                                     diurnal=wind_DIURNAL),
            null_summary="%.1f days measured; %d persistent RAD1 and %d RAD2 lines, %d explained"
                         % (d["minutes"] / 1440.0, d["RAD1"]["persistent"], d["RAD2"]["persistent"], len(expl)),
            ledger="Persistent lines below the ionospheric cutoff",
            result=("no persistent line outside the known sources in RAD1 or RAD2" if not left else
                    "%d lines outside the known sources: %s" % (len(left), ", ".join("%.0f kHz" % x["khz"] for x in left)))
                   + ("; the receiver's own lines at %s kHz" % ", ".join("%.0f" % x for x in own) if own else ""))
        print("recorded %d: %d left" % (y, len(left)))


wind_STEREO_ROOT = os.environ.get("STEREO_ROOT", core.data("stereo"))
# 41 m broadcast band references
wind_REFS = (7225.0, 7275.0, 7325.0, 7375.0)


def wind_stereo_year(year, craft, threads):
    """A year of STEREO/WAVES HFR spectra: per-channel line occupancy, days, diurnal modulation and persistent lines outside known sources."""
    import cdflib
    from scipy.ndimage import median_filter
    sc = "a" if craft == "ahead" else "b"
    d = os.path.join(wind_STEREO_ROOT, "tmp_%s_%d" % (sc, year))
    os.makedirs(d, exist_ok=True)
    days = [date(year, 1, 1) + timedelta(k) for k in range(366) if (date(year, 1, 1) + timedelta(k)).year == year]

    def one(day):
        name = "st%s_l3_wav_hfr_%s_v01.cdf" % (sc, day.strftime("%Y%m%d"))
        kept = os.path.join(wind_STEREO_ROOT, name)
        p = kept if os.path.exists(kept) else os.path.join(d, name)
        url = "https://cdaweb.gsfc.nasa.gov/pub/data/stereo/%s/l3/waves/hfr-direction-finding/%d/%s" % (craft, year, name)
        if p != kept:
            for attempt in range(6):
                try:
                    b = urllib.request.urlopen(urllib.request.Request(url, headers=wind_UA), timeout=300).read()
                    open(p, "wb").write(b)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        return None
                except Exception:
                    pass
                time.sleep(10 * 2 ** attempt)
            else:
                return "missing"
        try:
            c = cdflib.CDF(p)
            ep = np.array(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
            v = np.asarray(c.varget("PSD_FLUX"), float)
            f = np.asarray(c.varget("FREQUENCY"), float).ravel() / 1e3
        except Exception:
            return "unreadable"
        finally:
            if p != kept and os.path.exists(p):
                os.remove(p)
        lv = np.log(np.where(v > 0, v, np.nan))
        base = median_filter(np.nan_to_num(lv, nan=np.nanmedian(lv)), size=(1, 9), mode="nearest")
        r = lv - base
        mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r, axis=1, keepdims=True)), axis=1, keepdims=True)
        hit = r > 5 * np.maximum(mad, 1e-3)
        hours = ep.astype("datetime64[h]").astype(np.int64) % 24
        return dict(f=f, n=len(ep), occ=hit.sum(axis=0), hourly=np.array([hit[hours == h].sum(axis=0) for h in range(24)]),
                    hour_plus=hit.sum(axis=0) > len(ep) / 24.0)

    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, days))
    good = [x for x in res if isinstance(x, dict)]
    print("STEREO-%s %d: %d days measured, %d missing or unreadable, %d absent from the archive"
          % (sc.upper(), year, len(good), sum(1 for x in res if isinstance(x, str)), sum(1 for x in res if x is None)))
    if not good:
        return
    f = good[0]["f"]
    good = [x for x in good if len(x["f"]) == len(f)]
    n = sum(x["n"] for x in good)
    o = np.sum([x["occ"] for x in good], axis=0) / n
    hp = np.sum([x["hourly"] for x in good], axis=0)
    days_with = np.sum([x["hour_plus"] for x in good], axis=0)
    np.savez(os.path.join(wind_STEREO_ROOT, "lines_st%s_%d.npz" % (sc, year)), freq=f, occupancy=o, hourly=hp,
             days=days_with, measured=len(good))
    wind_stereo_classify(year, craft)


def wind_stereo_classify(year, craft):
    """Classify a STEREO year's persistent lines against known sources and Wind."""
    sc = "a" if craft == "ahead" else "b"
    z = np.load(os.path.join(wind_STEREO_ROOT, "lines_st%s_%d.npz" % (sc, year)))
    f, o, hp, days_with = z["freq"], z["occupancy"], z["hourly"], z["days"]
    measured = int(z["measured"]) if "measured" in z.files else None
    mod = (hp.max(axis=0) - hp.min(axis=0)) / np.maximum(hp.mean(axis=0), 1e-9)
    pers = [j for j in np.argsort(-o) if o[j] > 0.01 and days_with[j] >= 30]
    wind_occ = wind_occupancy_near(year)
    osc = "b" if sc == "a" else "a"
    op = os.path.join(wind_STEREO_ROOT, "lines_st%s_%d.npz" % (osc, year))
    oz = np.load(op) if os.path.exists(op) else None
    rest, explained = [], {}
    for j in pers:
        why = wind_known(f[j], mod[j])
        # a line STEREO holds five times as often as Wind is made on STEREO
        w = wind_occ(f[j]) if wind_occ else None
        if not why and w is not None and w < 0.2 * o[j]:
            why = "local to STEREO (Wind %.3f within 25 kHz in %d)" % (w, year)
        # likewise the other STEREO spacecraft
        sel = np.abs(oz["freq"] - f[j]) <= 25.0 if oz is not None else None
        wo = float(oz["occupancy"][sel].max()) if sel is not None and sel.any() else None
        if not why and wo is not None and wo < 0.2 * o[j]:
            why = "local to STEREO-%s (STEREO-%s %.3f within 25 kHz in %d)" % (sc.upper(), osc.upper(), wo, year)
        print("  %8.1f kHz: occupancy %.3f, %d days, 24 h modulation %.2f: %s"
              % (f[j], o[j], int(days_with[j]), mod[j], why or "not a known source"))
        if why:
            explained[float(f[j])] = why
        else:
            rest.append(j)
    print("STEREO-%s %d: persistent lines (occupancy above 1 per cent on 30 days or more) outside the known sources: %d"
          % (sc.upper(), year, len(rest)))
    json.dump(dict(year=year, craft=craft, days=measured, persistent=len(pers), explained=explained,
                   left=[dict(khz=float(f[j]), occupancy=float(o[j]), days=int(days_with[j]), modulation=float(mod[j]))
                         for j in rest]), open(core.results("stereo_%s_%d.json" % (sc, year)), "w"), indent=1)


def wind_stereo_record(years):
    """Record each classified STEREO year."""
    for y in years:
        for sc, craft in (("a", "ahead"), ("b", "behind")):
            p = core.results("stereo_%s_%d.json" % (sc, y))
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            left = d["left"]
            core.record_search(
                "stereo_waves_%s_%d" % (sc, y), "Persistent lines below the ionospheric cutoff, STEREO/WAVES",
                archive="CDAWeb STEREO-%s WAVES HFR L3 (PSD_FLUX)" % sc.upper(), release="v01",
                selection="every spectrum of %d" % y, code=os.path.abspath(__file__), entry="wind",
                params=dict(line_mad=5, occupancy=0.01, days=30, craft=craft),
                null_summary="%s days measured; %d persistent lines, %d explained"
                             % (d["days"] if d["days"] is not None else "all", d["persistent"], len(d["explained"])),
                ledger="Persistent lines below the ionospheric cutoff",
                result="no persistent line outside the known sources" if not left else
                       "%d lines outside the known sources: %s" % (len(left), ", ".join("%.0f kHz" % x["khz"] for x in left)))
            print("recorded STEREO-%s %d: %d left" % (sc.upper(), y, len(left)))


def wind_occupancy_near(year, width=25.0):
    """Function giving Wind's largest occupancy within width kHz in a year, or None."""
    zs = []
    for band in ("RAD1", "RAD2"):
        p = os.path.join(wind_ROOT, "lines_%s_%d.npz" % (band, year))
        if os.path.exists(p):
            z = np.load(p)
            zs.append((z["freq"], z["occupancy"]))
    if not zs:
        return None
    f = np.concatenate([a for a, _ in zs])
    o = np.concatenate([b for _, b in zs])

    def at(f_khz):
        sel = np.abs(f - f_khz) <= width
        return float(o[sel].max()) if sel.any() else None
    return at


def wind_line_minutes(epochs, values, freqs, f_khz):
    """Whole minutes with a 5-sigma line at the channel nearest f_khz."""
    from scipy.ndimage import median_filter
    v = np.where(values > 0, values, np.nan)
    lv = np.log(v)
    base = median_filter(np.nan_to_num(lv, nan=np.nanmedian(lv)), size=(1, 9), mode="nearest")
    r = lv - base
    mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r, axis=1, keepdims=True)), axis=1, keepdims=True)
    hit = r > 5 * np.maximum(mad, 1e-3)
    j = int(np.argmin(np.abs(freqs - f_khz)))
    mins = (epochs.astype("datetime64[m]").astype(np.int64))
    return set(mins[hit[:, j]].tolist()), set(mins.tolist()), float(freqs[j])


def wind_stereo(year, f_khz, day0, ndays):
    """Line minutes at f_khz at Wind and STEREO-A over the same days with chance coincidences."""
    import cdflib
    band = "RAD1" if f_khz < 1040 else "RAD2"
    wh, wm, sh, sm = set(), set(), set(), set()
    fw = fs = float("nan")
    for k in range(ndays):
        dd = date(year, 1, 1) + timedelta(day0 - 1 + k)
        wp = os.path.join(wind_ROOT, str(year), "wi_h1_wav_%s_v01.cdf" % dd.strftime("%Y%m%d"))
        sp = os.path.join(wind_STEREO_ROOT, "sta_l3_wav_hfr_%s_v01.cdf" % dd.strftime("%Y%m%d"))
        if not os.path.exists(sp):
            # HFR files under hfr-direction-finding or hfr
            for sub in ("hfr-direction-finding", "hfr"):
                url = "https://cdaweb.gsfc.nasa.gov/pub/data/stereo/ahead/l3/waves/%s/%d/%s" % (sub, year,
                                                                                               os.path.basename(sp))
                try:
                    b = urllib.request.urlopen(urllib.request.Request(url, headers=wind_UA), timeout=180).read()
                except Exception:
                    continue
                os.makedirs(wind_STEREO_ROOT, exist_ok=True)
                open(sp, "wb").write(b)
                break
            if not os.path.exists(sp):
                continue
        if not os.path.exists(wp):
            continue
        c = cdflib.CDF(wp)
        h, m, fw = wind_line_minutes(np.array(cdflib.cdfepoch.to_datetime(c.varget("Epoch"))),
                                np.asarray(c.varget("E_VOLTAGE_" + band), float),
                                np.asarray(c.varget("Frequency_" + band), float), f_khz)
        wh |= h
        wm |= m
        s = cdflib.CDF(sp)
        h, m, fs = wind_line_minutes(np.array(cdflib.cdfepoch.to_datetime(s.varget("Epoch"))),
                                np.asarray(s.varget("STOKES_I"), float),
                                np.asarray(s.varget("FREQUENCY"), float) / 1e3, f_khz)
        sh |= h
        sm |= m
    common = wm & sm
    a, b = wh & common, sh & common
    both = a & b
    exp = len(a) * len(b) / max(len(common), 1)
    print("%.1f kHz (Wind %.1f, STEREO-A %.1f): %d common minutes; line at Wind %d, at STEREO-A %d, both %d against "
          "%.1f by chance" % (f_khz, fw, fs, len(common), len(a), len(b), len(both), exp))
    return dict(f=f_khz, common=len(common), wind=len(a), stereo=len(b), both=len(both), chance=exp)


def wind_daily(year, freqs, refs):
    """Daily line minutes per frequency and coincidences; terrestrial lines track the broadcast bands."""
    import cdflib
    from scipy.stats import spearmanr
    d = os.path.join(wind_ROOT, str(year))
    allf = list(freqs) + list(refs)
    per_day = {f: [] for f in allf}
    mins = {f: set() for f in allf}
    dates = []
    total = 0
    for p in sorted(os.listdir(d)):
        if not p.endswith(".cdf"):
            continue
        try:
            c = cdflib.CDF(os.path.join(d, p))
            ep = np.array(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
            v = np.asarray(c.varget("E_VOLTAGE_RAD2"), float)
            fr = np.asarray(c.varget("Frequency_RAD2"), float)
        except Exception:
            continue
        total += len(ep)
        dates.append(p.split("_")[3])
        for f in allf:
            h, _, _ = wind_line_minutes(ep, v, fr, f)
            per_day[f].append(len(h))
            mins[f] |= h
    ref = np.sum([per_day[f] for f in refs], axis=0)
    out = {}
    for f in freqs:
        rho, p = spearmanr(per_day[f], ref)
        out[f] = dict(days=int(np.sum(np.array(per_day[f]) > 0)), minutes=len(mins[f]), rho_ref=float(rho), p=float(p),
                      per_day=dict(zip(dates, per_day[f])), line_minutes=sorted(mins[f]))
        print("%.1f kHz: line minutes %d on %d days; day-to-day Spearman correlation with the broadcast references %s: "
              "%.2f (p %.1g)" % (f, len(mins[f]), out[f]["days"], "+".join("%.0f" % r for r in refs), rho, p))
        top = sorted(out[f]["per_day"].items(), key=lambda kv: -kv[1])[:20]
        print("  busiest days: " + ", ".join("%s %d" % kv for kv in top))
    import json
    json.dump({"%.1f" % f: o for f, o in out.items()}, open(os.path.join(wind_ROOT, "daily_%d.json" % year), "w"))
    for i, f in enumerate(freqs):
        for g in list(freqs)[i + 1:] + list(refs):
            both = len(mins[f] & mins[g])
            exp = len(mins[f]) * len(mins[g]) / max(total, 1)
            print("  %.1f and %.1f kHz: %d minutes together against %.1f by chance" % (f, g, both, exp))
    return out


def wind_rad1(year):
    """RAD1 lines among measured channels only, against measured neighbours; into wind_rad1_<year>.json."""
    import cdflib
    d = os.path.join(wind_ROOT, str(year))
    occ, days, hourly, rel, nmin, nday = {}, {}, {}, {}, 0, 0
    for p in sorted(os.listdir(d)):
        if not p.endswith(".cdf") or wind_bad_day(p):
            continue
        try:
            c = cdflib.CDF(os.path.join(d, p))
            ep = cdflib.cdfepoch.to_datetime(c.varget("Epoch"))
        except Exception:
            continue
        fr = np.atleast_2d(c.varget("Frequency_RAD1"))[0]
        vm = np.atleast_2d(c.varget("Minimum_voltage_RAD1"))[0]
        meas = np.nonzero(vm > 0)[0]
        if len(meas) < 9:
            continue
        hours = np.array([int(str(x)[11:13]) for x in ep])
        x = np.asarray(c.varget("E_VOLTAGE_RAD1"), float)
        x = np.where(x > 0, x, np.nan)
        pw = x[:, meas] ** 2
        lv = np.log(pw)
        n, m = lv.shape
        for a in range(m):
            cols = [b for b in range(max(0, a - 4), min(m, a + 5)) if b != a]
            nb = lv[:, cols]
            med = np.nanmedian(nb, axis=1)
            resid = lv[:, a] - med
            mad = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
            hit = resid > 5 * max(mad, 1e-3)
            k = int(round(fr[meas[a]]))
            occ[k] = occ.get(k, 0) + int(np.nansum(hit))
            days[k] = days.get(k, 0) + int(np.nansum(hit) > 60)
            hourly.setdefault(k, np.zeros(24))
            for h in range(24):
                hourly[k][h] += np.nansum(hit[hours == h])
            # measured neighbours' scatter as a fraction of their median
            pn = pw[:, cols]
            pm = np.nanmedian(pn, axis=1)
            rel.setdefault(k, []).append(float(np.nanmedian(1.4826 * np.nanmedian(np.abs(pn - pm[:, None]), axis=1) / pm)))
        nmin += n
        nday += 1
    out = []
    for k in sorted(occ):
        h = hourly[k]
        known = "solar wind plasma line" if k < 100 else ("auroral kilometric radiation" if k <= 700 else "")
        out.append(dict(khz=k, occupancy=occ[k] / max(nmin, 1), days=days[k], modulation=float((h.max() - h.min()) / max(h.mean(), 1e-9)),
                        known=known, rel_dev=float(np.median(rel[k]))))
    pers = [c for c in out if c["occupancy"] > 0.01 and c["days"] >= 30]
    print("%d: %d days, %d minutes; %d measured channels, %d persistent lines" % (year, nday, nmin, len(out), len(pers)))
    for c in pers:
        print("  %5d kHz: occupancy %.3f on %d days, 24 h modulation %.2f, scatter against measured neighbours %.2f: %s"
              % (c["khz"], c["occupancy"], c["days"], c["modulation"], c["rel_dev"],
                 c["known"] or ("diurnal (terrestrial)" if c["modulation"] >= wind_DIURNAL else "unexplained")))
    json.dump(dict(year=year, days=nday, minutes=nmin, channels=out), open(core.results("wind_rad1_%d.json" % year), "w"), indent=1)
    return out


def wind_rad1_record(years):
    for y in years:
        d = json.load(open(core.results("wind_rad1_%d.json" % y)))
        ch = d["channels"]
        pers = [c for c in ch if c["occupancy"] > 0.01 and c["days"] >= 30]
        left = [c for c in pers if not c["known"] and c["modulation"] < wind_DIURNAL]
        core.record_search(
            "wind_rad1_%d" % y, "Lines below the ionospheric cutoff", archive="CDAWeb WI_H1_WAV (Wind/WAVES RAD1)", release="v01",
            selection="every minute of %d at the %d channels RAD1 measured directly" % (y, len(ch)), code=os.path.abspath(__file__),
            entry="wind", n_targets=len(ch),
            params=dict(sigma=5.0, neighbours=8, occupancy=0.01, days=30, measured_only=True),
            null_summary="%.1f days; a measured channel against the median of its eight nearest measured neighbours; scatter %.2f "
                         "of the background below 300 kHz and %.2f above" % (
                             d["minutes"] / 1440.0, np.median([c["rel_dev"] for c in ch if c["khz"] < 300]),
                             np.median([c["rel_dev"] for c in ch if c["khz"] >= 300])),
            sensitivity="a line of 5 robust deviations of the measured neighbours' scatter in a 3 kHz channel, %.2f of the background "
                        "above 300 kHz" % (5 * np.median([c["rel_dev"] for c in ch if c["khz"] >= 300])),
            ledger="Lines below the ionospheric cutoff",
            result=("%d persistent lines at the measured channels, all in the plasma-line and auroral ranges below 500 kHz; none above "
                    "700 kHz" % len(pers)) if not left else "%d lines outside the known sources: %s" % (len(left), ", ".join("%d kHz" % c["khz"] for c in left)))
        core.clear_targets("wind_rad1_%d" % y)
        core.record_targets("wind_rad1_%d" % y, [dict(
            target="wind:rad1:%d kHz" % c["khz"], stat=c["occupancy"], unit="occupancy", vet=c["known"] or None,
            note="%d days, 24 h modulation %.2f" % (c["days"], c["modulation"])) for c in pers])
        print("recorded wind_rad1_%d" % y)


def wind_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "classify", "stereo", "daily", "stereo_year",
                                     "stereo_classify", "record", "stereo_record", "rad1", "rad1_record"))
    p.add_argument("--craft", choices=("ahead", "behind"), default="ahead")
    p.add_argument("--refs", type=float, nargs="+", default=list(wind_REFS))
    p.add_argument("year", type=int, nargs="+")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--freq", type=float, nargs="+", default=[8025.0])
    p.add_argument("--day0", type=int, default=1)
    p.add_argument("--days", type=int, default=10)
    a = p.parse_args()
    if a.mode in ("record", "stereo_record", "rad1_record"):
        {"record": wind_record, "stereo_record": wind_stereo_record, "rad1_record": wind_rad1_record}[a.mode](a.year)
        return
    a.year = a.year[0]
    if a.mode == "fetch":
        wind_fetch(a.year, a.threads)
    elif a.mode == "rad1":
        wind_rad1(a.year)
    elif a.mode == "analyse":
        wind_analyse(a.year)
    elif a.mode == "classify":
        wind_classify(a.year)
    elif a.mode == "daily":
        wind_daily(a.year, a.freq, a.refs)
    elif a.mode == "stereo_year":
        wind_stereo_year(a.year, a.craft, a.threads)
    elif a.mode == "stereo_classify":
        wind_stereo_classify(a.year, a.craft)
    else:
        for f in a.freq:
            wind_stereo(a.year, f, a.day0, a.days)


def wind_cli():
    quiet_warnings()
    wind_main()


# --------------------------------------------------------------------------------------------------------------

callisto_DOC = r"""Sidereal test on a year of e-Callisto spectra: channels with sidereal over solar Lomb-Scargle power, excluding broadband, shared, GNSS and FM swings.

    python -I seti.py radio_sky callisto fetch STATION FOCUSCODE YEAR [--threads 6]
    python -I seti.py radio_sky callisto analyse STATION FOCUSCODE YEAR
    python -I seti.py radio_sky callisto report STATION FOCUSCODE YEAR     (the listing again from the saved result)
    python -I seti.py radio_sky callisto compare STATION_A CODE_A STATION_B CODE_B YEAR   (each residual at the other station)"""

callisto_ROOT = os.environ.get("CALLISTO_ROOT", core.data("callisto"))
# GNSS downlinks (MHz) repeat every sidereal day
callisto_GNSS = ((1164.0, 1300.0), (1559.0, 1611.0))
# FM broadcast band
callisto_BROADCAST = ((87.5, 108.0),)
# |r| with the broadband sidereal profile marking the Galaxy's swing
callisto_SKY_R = 0.8
# swings shared within 10 MHz and 2 h belong to the band
callisto_SHARED_MHZ, callisto_SHARED_H, callisto_SHARED_MIN = 10.0, 2.0, 2
# dB per digit of the logarithmic detector
callisto_DB_PER_DIGIT = 2500.0 / 256 / 25.4


def callisto_known(f_mhz, row=None):
    if any(lo <= f_mhz <= hi for lo, hi in callisto_GNSS):
        return "satellite navigation (sidereal by orbit design)"
    if any(lo <= f_mhz <= hi for lo, hi in callisto_BROADCAST):
        return "FM broadcast band"
    if row is not None and row.get("sky_locked"):
        return "the broadband sky's sidereal profile (r %+.2f)" % row["r_sky"]
    if row is not None and row.get("band_shared", 0) >= callisto_SHARED_MIN:
        return "a swing shared by %d channels within %.0f MHz" % (row["band_shared"], callisto_SHARED_MHZ)
    return None


# bright low-frequency sources by RA (h)
callisto_SKY_TRANSITS = {"Galactic centre": 17.76, "Cygnus A": 19.99, "Cassiopeia A": 23.39, "Taurus A": 5.58, "Virgo A": 12.51}
callisto_SKY_PHASE_H = 1.5


def callisto_compare(paths):
    """Each residual of one station at the other: sidereal power over noise, phase, and bright-source transits."""
    st = [json.load(open(p)) for p in paths]
    for s in st:
        callisto_mark_shared(s["channels"])
    for a, b in ((st[0], st[1]), (st[1], st[0])):
        ra = [r for r in a["channels"] if r["ratio"] > 2 and r["sid_over_noise"] > 1 and not callisto_known(r["freq"], r)]
        print("%s %s residuals against %s %s:" % (a["station"], a["code"], b["station"], b["code"]))
        for r in sorted(ra, key=lambda r: -r["ratio"]):
            q = min(b["channels"], key=lambda x: abs(x["freq"] - r["freq"]))
            if abs(q["freq"] - r["freq"]) > 0.5 or "phase_h" not in r or "phase_h" not in q:
                print("  %.3f MHz: no channel within 0.5 MHz at the other station" % r["freq"])
                continue
            dph = abs((r["phase_h"] - q["phase_h"] + 12) % 24 - 12)
            sky = [n for n, h in callisto_SKY_TRANSITS.items() if abs((r["phase_h"] - h + 12) % 24 - 12) <= callisto_SKY_PHASE_H
                   and abs((q["phase_h"] - h + 12) % 24 - 12) <= callisto_SKY_PHASE_H]
            print("  %.3f MHz (ratio %.1f, over noise %.2f, phase %.1f h) vs %.3f MHz (ratio %.1f, over noise %.2f, phase "
                  "%.1f h): phases %.1f h apart%s" % (r["freq"], r["ratio"], r["sid_over_noise"], r["phase_h"], q["freq"],
                                                      q["ratio"], q["sid_over_noise"], q["phase_h"], dph,
                                                      "; both at the transit of " + ", ".join(sky) if sky else ""))


def callisto_mark_shared(rows):
    """Count other sidereal channels within SHARED_MHZ agreeing in phase within SHARED_H (mod 12 h)."""
    sel = [r for r in rows if r["sid_over_noise"] > 1 and r["ratio"] > 2 and "phase_h" in r]
    for r in rows:
        r["band_shared"] = 0
    for r in sel:
        for q in sel:
            if q is r or abs(q["freq"] - r["freq"]) > callisto_SHARED_MHZ:
                continue
            dh = abs(r["phase_h"] % 12 - q["phase_h"] % 12)
            if min(dh, 12 - dh) <= callisto_SHARED_H:
                r["band_shared"] += 1
callisto_BASE = "http://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/"
callisto_UA = core.UA


def callisto_get(url, timeout=120):
    """Bytes of a URL, or None when missing or still failing after three tries."""
    return core.get(url, timeout=timeout, retries=3, soft=True, pause=0)


def callisto_reduce_file(raw):
    """Per-channel median power and frequencies of one e-Callisto file."""
    from astropy.io import fits
    with fits.open(io.BytesIO(gzip.decompress(raw))) as h:
        d = np.asarray(h[0].data, float)
        freqs = np.asarray(h[1].data["FREQUENCY"][0], float) if len(h) > 1 else np.arange(d.shape[0], dtype=float)
        hdr = h[0].header
        t0 = "%sT%s" % (hdr.get("DATE-OBS", ""), hdr.get("TIME-OBS", ""))
    return np.median(d, axis=1), freqs, t0


def callisto_fetch_day(args):
    station, code, day = args
    out = os.path.join(callisto_ROOT, "%s_%s" % (station, code), day.strftime("%Y%m%d") + ".npz")
    if os.path.exists(out):
        return 1
    d = day.strftime("%Y/%m/%d/")
    page = callisto_get(callisto_BASE + d)
    if not page:
        return 0
    names = sorted(set(re.findall(r'href="(%s_\d{8}_(\d{2})(\d{2})\d{2}_%s\.fit\.gz)"' % (re.escape(station), code),
                                  page.decode(errors="replace"))))
    hourly = {}
    for name, hh, mm in names:
        hourly.setdefault(hh, name)
    P, T, F = [], [], None
    for hh, name in sorted(hourly.items()):
        raw = callisto_get(callisto_BASE + d + name)
        if not raw:
            continue
        try:
            p, f, t0 = callisto_reduce_file(raw)
        except Exception:
            continue
        if F is None:
            F = f
        if len(p) != len(F):
            continue
        P.append(p)
        T.append(day.toordinal() + int(hh) / 24.0 + 0.3 / 24.0)
    if not P:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, power=np.array(P, np.float32), t=np.array(T), freq=F)
    return 1


def callisto_fetch(station, code, year, threads):
    days = [date(year, 1, 1) + timedelta(k) for k in range(366) if (date(year, 1, 1) + timedelta(k)).year == year]
    with ThreadPoolExecutor(threads) as ex:
        n = sum(ex.map(callisto_fetch_day, [(station, code, d) for d in days]))
    print("%s %s %d: %d days" % (station, code, year, n))


def callisto_analyse(station, code, year):
    from astropy.timeseries import LombScargle
    d = os.path.join(callisto_ROOT, "%s_%s" % (station, code))
    P, T, F = [], [], None
    for p in sorted(os.listdir(d)):
        if not p.startswith(str(year)):
            continue
        z = np.load(os.path.join(d, p))
        if F is None:
            F = z["freq"]
        if z["power"].shape[1] != len(F):
            continue
        P.append(z["power"])
        T.append(z["t"])
    P, T = np.vstack(P), np.concatenate(T)
    order = np.argsort(F)
    X = P[:, order].astype(float)
    Fs = F[order]
    # repeated frequencies are unused program slots
    uniq, cnt = np.unique(Fs, return_counts=True)
    keep = ~np.isin(Fs, uniq[cnt > 1])
    X, Fs = X[:, keep], Fs[keep]
    # linear power per channel fitted against its neighbours' median; the residual carries the channel's own signal
    L = 10 ** (callisto_DB_PER_DIGIT * X / 10.0)
    Ln = L / np.nanmedian(L, axis=0)
    level = np.nanmedian(X, axis=0)
    S = np.nanmedian(Ln, axis=1)
    okS = np.isfinite(S)
    ex = np.full_like(Ln, np.nan)
    nch = Ln.shape[1]
    for j in range(nch):
        nb = [k for k in range(max(0, j - 4), min(nch, j + 5)) if k != j]
        N = np.nanmedian(Ln[:, nb], axis=1)
        y = Ln[:, j]
        ok = np.isfinite(N) & np.isfinite(y)
        if ok.sum() > 100:
            A = np.vstack([np.ones(ok.sum()), N[ok]]).T
            c, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
            ex[ok, j] = y[ok] - A @ c
    t_days = T - T.min()
    sol, sid = 1.0, 1.0027379093
    rand = np.random.default_rng(1).uniform(0.96, 1.04, 200)
    hour = np.floor((t_days * sid % 1.0) * 24).astype(int)

    def sidereal_profile(y, ok):
        return np.array([np.mean(y[ok & (hour == h)]) if np.any(ok & (hour == h)) else np.nan for h in range(24)])

    prof_sky = sidereal_profile(S, okS)

    def sidereal_hour(y, ok):
        w = 2 * np.pi * sid * t_days[ok]
        A = np.vstack([np.ones(ok.sum()), np.cos(w), np.sin(w)]).T
        c, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        return float((np.degrees(np.arctan2(c[2], c[1])) / 15.0) % 24)
    rows = []
    for j in range(ex.shape[1]):
        y = ex[:, j]
        ok = np.isfinite(y)
        if ok.sum() < 1000 or np.std(y[ok]) == 0:
            continue
        ls = LombScargle(t_days[ok], y[ok])
        p_sol, p_sid = float(ls.power(sol)), float(ls.power(sid))
        p_rand = ls.power(rand)
        pr = sidereal_profile(y, ok)
        good = np.isfinite(pr) & np.isfinite(prof_sky)
        r_sky = float(np.corrcoef(pr[good], prof_sky[good])[0, 1]) if good.sum() > 12 and np.std(pr[good]) > 0 else 0.0
        rows.append(dict(freq=float(Fs[j]), p_sol=p_sol, p_sid=p_sid, ratio=p_sid / max(p_sol, 1e-9),
                         sid_over_noise=p_sid / max(float(np.percentile(p_rand, 95)), 1e-9),
                         level=float(level[j] - np.median(level)), r_sky=r_sky, phase_h=sidereal_hour(y, ok),
                         sky_locked=bool(abs(r_sky) >= callisto_SKY_R)))
    json.dump(dict(station=station, code=code, year=year, hours=int(len(T)), channels=rows),
              open(os.path.join(callisto_ROOT, "sidereal_%s_%s_%d.json" % (station, code, year)), "w"))
    np.savez_compressed(os.path.join(callisto_ROOT, "residual_%s_%s_%d.npz" % (station, code, year)), ex=ex.astype(np.float32),
                        sky=S.astype(np.float32), t=T, freq=Fs)
    print("%s %s %d: %d hours, %d channels" % (station, code, year, len(T), len(rows)))
    callisto_report(rows)


def callisto_report(rows):
    """Channels with sidereal power twice the solar and above the local 95th percentile, known sources set aside."""
    callisto_mark_shared(rows)
    strong = sorted((r for r in rows if r["sid_over_noise"] > 1 and r["ratio"] > 2), key=lambda r: -r["ratio"])
    kn = [r for r in strong if callisto_known(r["freq"], r)]
    rest = [r for r in strong if not callisto_known(r["freq"], r)]
    free = [r for r in rows if not callisto_known(r["freq"], r)]
    over = sum(r["sid_over_noise"] > 1 for r in free)
    print("channels outside known sidereal sources: %d; above the 95th percentile of nearby frequencies: %d "
          "(%.1f by chance); also twice their solar power: %d" % (len(free), over, 0.05 * len(free), len(rest)))
    for r in kn[:10]:
        print("  %.3f MHz: sidereal %.4f, solar %.4f, ratio %.1f, over noise %.2f: %s"
              % (r["freq"], r["p_sid"], r["p_sol"], r["ratio"], r["sid_over_noise"], callisto_known(r["freq"], r)))
    for r in rest[:15]:
        print("  %.3f MHz: sidereal %.4f, solar %.4f, ratio %.1f, over noise %.2f, level %+.1f digits%s, r with the "
              "sky %+.2f" % (r["freq"], r["p_sid"], r["p_sol"], r["ratio"], r["sid_over_noise"], r.get("level", 0.0),
                             ", phase %.1f h" % r["phase_h"] if "phase_h" in r else "", r.get("r_sky", float("nan"))))
    top = sorted(rows, key=lambda r: -r["p_sol"])[:5]
    print("strongest solar-day channels:", ", ".join("%.2f MHz (%.2f)" % (r["freq"], r["p_sol"]) for r in top))
    return rest


def callisto_main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # compare STATION_A CODE_A STATION_B CODE_B YEAR
        sa, ca, sb, cb, yr = sys.argv[2:7]
        callisto_compare([os.path.join(callisto_ROOT, "sidereal_%s_%s_%s.json" % (s, c, yr)) for s, c in ((sa, ca), (sb, cb))])
        return
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fetch", "analyse", "report"))
    p.add_argument("station")
    p.add_argument("code")
    p.add_argument("year", type=int)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    if a.mode == "fetch":
        callisto_fetch(a.station, a.code, a.year, a.threads)
    elif a.mode == "analyse":
        callisto_analyse(a.station, a.code, a.year)
    else:
        d = json.load(open(os.path.join(callisto_ROOT, "sidereal_%s_%s_%d.json" % (a.station, a.code, a.year))))
        print("%s %s %d: %d hours, %d channels" % (a.station, a.code, a.year, d["hours"], len(d["channels"])))
        callisto_report(d["channels"])


def callisto_cli():
    quiet_warnings()
    callisto_main()


# --------------------------------------------------------------------------------------------------------------

radar_DOC = r"""Stars within 100 pc in the beams of Arecibo and Goldstone asteroid radar, with reply years.

    python -I seti.py radio_sky radar tracks
    python -I seti.py radio_sky radar stars
    python -I seti.py radio_sky radar record"""

radar_ROOT = os.environ.get("RADAR_ROOT", core.data("radar"))
radar_HERE = core.RESULTS
radar_API = "https://ssd-api.jpl.nasa.gov/sb_radar.api?fullname=true&observer=true"
radar_HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
radar_BEAM_ARCMIN = {"-1": 1.0, "-14": 1.1, "-63": 1.1, "-43": 1.1, "-13": 2.3, "-35": 2.3, "-36": 2.3}
radar_NAMES = {"-1": "Arecibo", "-14": "Goldstone DSS-14", "-13": "Goldstone DSS-13", "-43": "Canberra DSS-43",
         "-35": "Canberra DSS-35", "-36": "Canberra DSS-36", "-63": "Madrid DSS-63", "-2": "Haystack"}


def radar_horizons(des, site, start, stop, step):
    q = dict(format="json", COMMAND="'DES=%s;CAP;'" % des if not des.startswith(("C/", "P/")) and not des[0].isdigit()
             else "'%s'" % des, EPHEM_TYPE="OBSERVER", CENTER="'%s@399'" % site, START_TIME="'%s'" % start,
             STOP_TIME="'%s'" % stop, STEP_SIZE="'%s'" % step, QUANTITIES="'1'", ANG_FORMAT="DEG", CSV_FORMAT="YES",
             OBJ_DATA="NO", MAKE_EPHEM="YES")
    b = core.get(radar_HORIZONS + "?" + urllib.parse.urlencode(q), timeout=120, retries=3, soft=True, pause=2)
    if not b:
        return []
    r = json.loads(b).get("result", "")
    if "$$SOE" not in r:
        return []
    out = []
    for line in r.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        c = [x.strip() for x in line.split(",")]
        try:
            out.append((c[0], float(c[3]), float(c[4])))
        except (IndexError, ValueError):
            continue
    return out


def radar_tracks():
    os.makedirs(radar_ROOT, exist_ok=True)
    d = json.loads(core.get(radar_API, timeout=120, retries=3))
    f = d["fields"]
    rows = [dict(zip(f, r)) for r in d["data"]]
    groups = {}
    for r in rows:
        day = r["epoch"][:10]
        groups.setdefault((r["des"], r["xmit"], day), []).append(r["epoch"])
    out_path = os.path.join(radar_ROOT, "tracks.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(t["des"], t["xmit"], t["day"]) for t in (json.loads(ln) for ln in open(out_path))}
    print("%d radar measurements; %d target-transmitter-days (%d done)" % (len(rows), len(groups), len(done)), flush=True)
    with open(out_path, "a") as fh:
        for (des, xmit, day), eps in sorted(groups.items()):
            if (des, xmit, day) in done:
                continue
            eps = sorted(eps)
            from datetime import datetime
            t0 = datetime.strptime(eps[0], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(eps[-1], "%Y-%m-%d %H:%M:%S")
            a = (t0 - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
            b = (t1 + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
            eph = radar_horizons(des, xmit, a, b, "5m")
            fh.write(json.dumps(dict(des=des, xmit=xmit, day=day, n=len(eps), eph=eph)) + "\n")
            fh.flush()
            time.sleep(0.5)
    print("tracks written to %s" % out_path)


def radar_stars():
    trs = [json.loads(ln) for ln in open(os.path.join(radar_ROOT, "tracks.jsonl"))]
    trs = [t for t in trs if t["eph"]]
    g = core.gaia("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp FROM gaiadr3.gaia_source "
                      "WHERE parallax > 10 AND parallax_over_error > 5")
    sra = np.radians([float(r["ra"]) for r in g])
    sde = np.radians([float(r["dec"]) for r in g])
    pmra = np.array([float(r["pmra"] or 0) for r in g])
    pmde = np.array([float(r["pmdec"] or 0) for r in g])
    plx = np.array([float(r["parallax"]) for r in g])
    from astropy.time import Time
    hits = []
    for t in trs:
        rad = radar_BEAM_ARCMIN.get(t["xmit"], 2.3) / 60.0
        ra = np.array([e[1] for e in t["eph"]])
        de = np.array([e[2] for e in t["eph"]])
        # densify to quarter-beam steps
        pts_ra, pts_de = [], []
        for i in range(len(ra) - 1):
            # short way round in RA
            dra = (ra[i + 1] - ra[i] + 180.0) % 360.0 - 180.0
            sep = math.hypot(dra * math.cos(math.radians(de[i])), de[i + 1] - de[i])
            n = max(int(sep / (rad / 4)) + 1, 1)
            for k in range(n):
                pts_ra.append((ra[i] + dra * k / n) % 360.0)
                pts_de.append(de[i] + (de[i + 1] - de[i]) * k / n)
        pts_ra.append(ra[-1])
        pts_de.append(de[-1])
        pr, pd = np.radians(pts_ra), np.radians(pts_de)
        yrs = (Time(t["day"]).jyear - 2016.0)
        cra = sra + np.radians(pmra / 3.6e6 * yrs) / np.cos(sde)
        cde = sde + np.radians(pmde / 3.6e6 * yrs)
        # stars near the track's bounding box first
        box = (cde > pd.min() - 0.02) & (cde < pd.max() + 0.02)
        idx = np.where(box)[0]
        if len(idx) == 0:
            continue
        for j in idx:
            cs = np.sin(cde[j]) * np.sin(pd) + np.cos(cde[j]) * np.cos(pd) * np.cos(cra[j] - pr)
            m = float(np.degrees(np.arccos(np.clip(cs.max(), -1, 1))))
            if m < rad:
                d_ly = 3261.56 / plx[j]
                year = Time(t["day"]).jyear
                hits.append(dict(gaia=g[j]["source_id"], dist_ly=d_ly, g=g[j]["phot_g_mean_mag"], bp_rp=g[j]["bp_rp"],
                                 target=t["des"], station=radar_NAMES.get(t["xmit"], t["xmit"]), date=t["day"],
                                 sep_arcmin=m * 60, reply_year=year + 2 * d_ly))
    hits.sort(key=lambda h: h["reply_year"])
    json.dump(hits, open(os.path.join(radar_HERE, "radar_beams.json"), "w"), indent=1)
    print("%d radar target-days with tracks; %d pings of %d Gaia stars within 100 pc" % (
        len(trs), len(hits), len({h["gaia"] for h in hits})))
    for h in hits[:40]:
        print("  Gaia DR3 %s %5.1f ly G %s: %s on %s from %s, %.2f arcmin off the beam axis; reply could arrive from %.1f"
              % (h["gaia"], h["dist_ly"], h["g"], h["target"], h["date"], h["station"], h["sep_arcmin"], h["reply_year"]))


def radar_record():
    """Record one row per ping with star and earliest reply year."""
    hits = json.load(open(os.path.join(radar_HERE, "radar_beams.json")))
    nearest = min(hits, key=lambda h: h["dist_ly"])
    core.record_search(
        "radar_beam_stars", "Stars in the beams of planetary radar (reply windows)",
        archive="JPL small-body radar astrometry API; JPL Horizons; Gaia DR3", release="radar astrometry to 2025-10",
        selection="Gaia DR3 stars within 100 pc (parallax S/N 5) in the half-power beam of every radar target-day",
        code="seti.py radio_sky radar", params=dict(beam_arcmin=radar_BEAM_ARCMIN, step="quarter beam", margin_h=1),
        n_targets=len({h["gaia"] for h in hits}),
        null_summary="no pinged star lies close enough for its reply window to have opened (earliest %.1f)"
                     % min(h["reply_year"] for h in hits),
        sensitivity="every published delay and Doppler measurement, 1968-2025",
        result="%d pings of %d stars; nearest %.1f ly (%s, %s, %s), reply from %.1f"
               % (len(hits), len({h["gaia"] for h in hits}), nearest["dist_ly"], nearest["station"], nearest["target"],
                  nearest["date"], nearest["reply_year"]),
        ledger="Stars in planetary-radar beams")
    core.clear_targets("radar_beam_stars")
    first = {}
    for h in sorted(hits, key=lambda h: h["reply_year"]):
        first.setdefault(h["gaia"], h)
    core.record_targets("radar_beam_stars", [
        dict(target="gaia_dr3:%s" % g, stat=float(h["reply_year"]), unit="earliest reply year",
             note="%s from %s on %s, %.2f arcmin off axis, %.1f ly" % (h["target"], h["station"], h["date"],
                                                                      h["sep_arcmin"], h["dist_ly"]))
        for g, h in first.items()])


def radar_main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("tracks", "stars", "record"))
    a = p.parse_args()
    {"tracks": radar_tracks, "stars": radar_stars, "record": radar_record}[a.mode]()


def radar_cli():
    quiet_warnings()
    radar_main()


# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------

alma_DOC = r"""Narrowband ALMA cube lines at stars within 20 pc: 6-sigma one- or two-channel peaks against annulus noise, negatives as null, Splatalogue identification.

    python -I seti.py radio_sky alma fetch [--max-gb 1.5] [--units MOUS ...]
    python -I seti.py radio_sky alma search
    python -I seti.py radio_sky alma record                (peaks above the null named against Splatalogue)"""

alma_ROOT = os.environ.get("ALMA_ROOT", core.data("alma"))
alma_LIST = core.results("alma_nearby.json")
alma_OUT = core.results("alma_lines.json")
alma_MOLECULES = ("CO", "HCN", "HNC", "HCO+", "DCO+", "DCN", "CS", "SiO", "SO", "CN", "H2O", "H2CO", "N2H+", "C2H", "CCH",
                  "CI", "C-atom", "H&alpha;", "H&beta;", "H&gamma;", "He&alpha;")


def alma_units():
    """{member ous: [Gaia DR3 ids]} from alma_nearby.json."""
    out = {}
    for row in json.load(open(alma_LIST)):
        out.setdefault(row[5], set()).add(str(row[0]))
    return {k: sorted(v) for k, v in out.items()}


def alma_dir(mous):
    return os.path.join(alma_ROOT, "cubes", mous.replace("uid://", "").replace("/", "_"))


def alma_fetch_main(max_gb, only=None):
    import tarfile
    for mous in sorted(alma_units()):
        if only and mous not in only:
            continue
        dest = alma_dir(mous)
        if os.path.exists(os.path.join(dest, ".done")):
            continue
        prod = [f for f in core.alma_datalink(mous) if f["semantics"] == "#this" and str(f["bytes"]).isdigit()]
        size = sum(int(f["bytes"]) for f in prod)
        if not prod or size > max_gb * 1e9:
            print("%s: %.2f GB of products, not fetched" % (mous, size / 1e9), flush=True)
            continue
        os.makedirs(dest, exist_ok=True)
        kept, failed = 0, False
        for f in prod:
            name = os.path.basename(f["url"])
            if not name.endswith((".tar", ".tgz", ".tar.gz")):
                # products served as their own files: cubes only
                if "_sci." in name and ".cube.I.pb" in name:
                    if os.path.exists(os.path.join(dest, name)) or core.download(f["url"], os.path.join(dest, name),
                                                                               size=int(f["bytes"])):
                        kept += 1
                    else:
                        print("%s: %s failed" % (mous, name), flush=True)
                        failed = True
                continue
            tmp = os.path.join(alma_ROOT, "tar", name)
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            if not core.download(f["url"], tmp, size=int(f["bytes"])):
                print("%s: %s failed" % (mous, os.path.basename(f["url"])), flush=True)
                failed = True
                continue
            try:
                with tarfile.open(tmp) as t:
                    for m in t.getmembers():
                        n = os.path.basename(m.name)
                        if m.isfile() and "_sci." in n and ".cube.I.pb" in n:
                            with open(os.path.join(dest, n), "wb") as fh:
                                fh.write(t.extractfile(m).read())
                            kept += 1
            except (tarfile.TarError, EOFError, OSError) as e:
                print("%s: %s unreadable (%s), removed for a later pass" % (mous, os.path.basename(f["url"]),
                                                                            str(e).splitlines()[0][:60]), flush=True)
                failed = True
            os.remove(tmp)
        if not failed:
            open(os.path.join(dest, ".done"), "w").write(str(kept))
        print("%s: %.2f GB of products, %d cube files kept%s" % (mous, size / 1e9, kept,
                                                               ", a product to fetch again" if failed else ""), flush=True)


def alma_spectrum(path, ra, dec):
    """(Hz, Jy/beam, noise) at a position of a pbcor cube, annulus-subtracted; None off the image."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    with fits.open(path) as h:
        hdr = h[0].header
        cube = np.squeeze(h[0].data).astype(np.float32)
    pbp = path.replace(".pbcor.fits", ".pb.fits.gz")
    pb = np.squeeze(fits.getdata(pbp)).astype(np.float32) if os.path.exists(pbp) else np.ones_like(cube)
    if cube.ndim != 3:
        return None
    x, y = WCS(hdr).celestial.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    ix, iy = int(round(float(x))), int(round(float(y)))
    nz, ny, nx = cube.shape
    if not (0 <= ix < nx and 0 <= iy < ny):
        return None
    beam = np.sqrt(hdr["BMAJ"] * hdr["BMIN"]) / abs(hdr["CDELT1"])
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - float(x), yy - float(y))
    ann = (r > 2 * beam) & (r < 4 * beam)
    if ann.sum() < 20:
        return None
    flat = cube[:, ann] * pb[:, ann]
    with np.errstate(all="ignore"):
        med = np.nanmedian(flat, axis=1)
        rms = 1.4826 * np.nanmedian(np.abs(flat - med[:, None]), axis=1)
        noise = rms / pb[:, iy, ix]
        local = cube[:, iy, ix] - np.nanmedian(cube[:, ann], axis=1)
    freq = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    return freq, local.astype(float), noise.astype(float)


def alma_peaks(freq, s, e, thr=6.0, win=25):
    """One- or two-channel runs thr above a win-channel running median, away from window ends: [(index, z, sign)]."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(s) & np.isfinite(e) & (e > 0)
    if ok.sum() < 3 * win:
        return []
    edge = max(3, len(s) // 32)
    ok[:edge] = ok[-edge:] = False
    base = median_filter(np.where(ok, s, np.nanmedian(s[ok])), size=win, mode="nearest")
    z = np.where(ok, (s - base) / np.where(ok, e, 1.0), 0.0)
    out = []
    for sign in (1, -1):
        above = sign * z > 3.0
        k = 0
        while k < len(z):
            if above[k]:
                j = k
                while j < len(z) and above[j]:
                    j += 1
                run = slice(k, j)
                top = k + int(np.argmax(sign * z[run]))
                if j - k <= 2 and sign * z[top] >= thr:
                    out.append((top, float(sign * z[top]), sign))
                k = j
            else:
                k += 1
    return out


def alma_search_main():
    from astropy.io import fits
    units = alma_units()
    ids = sorted({s for v in units.values() for s in v})
    gaia = {}
    for i in range(0, len(ids), 200):
        for r in core.gaia("SELECT source_id, ra, dec, pmra, pmdec, radial_velocity FROM gaiadr3.gaia_source WHERE "
                               "source_id IN (%s)" % ",".join(ids[i:i + 200])):
            gaia[r["source_id"]] = r
    rows, peaks = [], []
    rng = np.random.default_rng(1)
    injected = recovered = 0
    for mous, sids in sorted(units.items()):
        d = alma_dir(mous)
        for path in sorted(glob.glob(os.path.join(d, "*.cube.I.pbcor.fits"))):
            hdr = fits.getheader(path)
            t = (time.mktime(time.strptime(hdr["DATE-OBS"][:19], "%Y-%m-%dT%H:%M:%S")) / 86400.0 + 40587.0 - 57388.5) / 365.25
            for sid in sids:
                g = gaia.get(sid)
                if not g:
                    continue
                dec = float(g["dec"]) + float(g["pmdec"] or 0) / 3.6e6 * t
                ra = float(g["ra"]) + float(g["pmra"] or 0) / 3.6e6 * t / np.cos(np.radians(dec))
                sp = alma_spectrum(path, ra, dec)
                if sp is None:
                    continue
                freq, s, e = sp
                found = alma_peaks(freq, s, e)
                edge = max(3, len(s) // 32)
                for _ in range(5):
                    k = int(rng.integers(edge + 15, len(s) - edge - 15))
                    s2 = s.copy()
                    s2[k] += 8 * e[k]
                    injected += 1
                    recovered += any(abs(j - k) <= 1 and sg > 0 for j, _, sg in alma_peaks(freq, s2, e))
                rows.append(dict(mous=mous, star=sid, cube=os.path.basename(path), nchan=len(freq),
                                 chan_mhz=abs(float(freq[1] - freq[0])) / 1e6, ghz=float(np.median(freq)) / 1e9,
                                 noise_mjy=float(np.nanmedian(e)) * 1e3, rv=g.get("radial_velocity"), ra=ra, dec=dec))
                for k, z, sign in found:
                    peaks.append(dict(mous=mous, star=sid, cube=os.path.basename(path), ghz=float(freq[k]) / 1e9,
                                      z=z, sign=sign, chan_mhz=rows[-1]["chan_mhz"], flux_mjy=float(s[k]) * 1e3,
                                      rv=g.get("radial_velocity"), ra=ra, dec=dec))
    pos = sorted((p for p in peaks if p["sign"] > 0), key=lambda p: -p["z"])
    neg = sorted((p for p in peaks if p["sign"] < 0), key=lambda p: -p["z"])
    top = neg[0]["z"] if neg else 0.0
    above = [p for p in pos if p["z"] > top]
    print("%d cube spectra of %d stars in %d units; 6-sigma peaks %d emission-like, %d absorption-like (strongest %.1f); "
          "%d emission-like above it" % (len(rows), len({r["star"] for r in rows}), len({r["mous"] for r in rows}),
                                         len(pos), len(neg), top, len(above)))
    for p in pos[:30]:
        print("  %s %s %.5f GHz z %.1f, %.1f mJy, %.2f MHz channels%s" % (p["star"], p["cube"].split(".")[-5],
                                                                         p["ghz"], p["z"], p["flux_mjy"], p["chan_mhz"],
                                                                         " ABOVE THE NULL" if p in above else ""))
    print("%d of %d injected 8-sigma single-channel lines recovered" % (recovered, injected))
    json.dump(dict(spectra=rows, peaks=peaks, injected=injected, recovered=recovered), open(alma_OUT, "w"), indent=0)


def alma_lsrk(ra, dec, rv):
    """Star velocity in the cubes' LSRK frame (km/s)."""
    from astropy.coordinates import SkyCoord, LSRK
    import astropy.units as u
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=10 * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
                 pm_dec=0 * u.mas / u.yr, radial_velocity=rv * u.km / u.s, frame="icrs")
    return float(c.transform_to(LSRK()).radial_velocity.to(u.km / u.s).value)


def alma_identify(p, tol_chan=2.0, shift=0.0):
    """Splatalogue species within tol_chan channels of a rest-frame peak; shift measures chance coincidences."""
    import re
    from astroquery.splatalogue import Splatalogue
    import astropy.units as u
    known = p.get("rv") not in (None, "")
    v = alma_lsrk(p["ra"], p["dec"], float(p["rv"]) if known else 0.0)
    rest = p["ghz"] * (1 + v / 299792.458) + shift
    half = tol_chan * p["chan_mhz"] / 1e3 + rest * (3.0 if known else 60.0) / 299792.458
    try:
        t = Splatalogue.query_lines((rest - half) * u.GHz, (rest + half) * u.GHz, only_NRAO_recommended=True)
    except Exception:
        return None
    if t is None or not len(t):
        return []
    found = set()
    for raw in t["name"]:
        parent = re.sub(r"<[^>]+>", "", re.sub(r"<sup>\d+</sup>", "", str(raw))).split()
        if parent and parent[0] in alma_MOLECULES:
            found.add(re.sub(r"<[^>]+>", "", str(raw)).split()[0])
    return sorted(found)


def alma_record_main():
    d = json.load(open(alma_OUT))
    pos = [p for p in d["peaks"] if p["sign"] > 0]
    neg = [p for p in d["peaks"] if p["sign"] < 0]
    top = max([p["z"] for p in neg] or [0.0])
    above = [p for p in pos if p["z"] > top]
    chance = []
    for p in above:
        p["molecules"] = alma_identify(p)
        chance += [bool(alma_identify(p, shift=s)) for s in (-0.05, 0.05)]
    left = [p for p in above if not p.get("molecules")]
    rate = float(np.mean(chance)) if chance else 0.0
    stars = sorted({r["star"] for r in d["spectra"]})
    core.record_search(
        "alma_nearby", "Narrowband millimetre lines, ALMA cubes of nearby stars", archive="ALMA archive pipeline cubes",
        selection="observing units within 15 arcsec of Gaia DR3 stars within 20 pc with products of 1.5 GB or less",
        code=os.path.abspath(__file__), entry="alma", n_targets=len(stars),
        params=dict(threshold=6.0, run_channels=2, median_channels=25),
        null_summary="%d absorption-like peaks, strongest %.1f" % (len(neg), top),
        sensitivity="median channel noise %.2f mJy/beam; %d of %d injected 8-sigma single-channel lines recovered"
                    % (float(np.median([r["noise_mjy"] for r in d["spectra"]])), d["recovered"], d["injected"]),
        ledger="ALMA cubes of nearby stars",
        result="%d emission-like peaks above the null, %d on molecular lines (chance rate %.2f), %d left"
               % (len(above), len(above) - len(left), rate, len(left)))
    core.clear_targets("alma_nearby")
    core.record_targets("alma_nearby", [dict(target="gaia_dr3:%s" % s, stat=sum(1 for p in left if p["star"] == s),
                                                 unit="unexplained narrowband peaks",
                                                 note="%d cubes" % sum(1 for r in d["spectra"] if r["star"] == s))
                                            for s in stars])
    print("%d stars; %d emission-like peaks above the null (%.1f), %d on molecular lines (chance rate %.2f), %d left"
          % (len(stars), len(above), top, len(above) - len(left), rate, len(left)))
    for p in above:
        print("  %s %.5f GHz z %.1f: %s" % (p["star"], p["ghz"], p["z"], ", ".join(p["molecules"] or []) or "no line"))


def alma_cli():
    quiet_warnings()
    p = argparse.ArgumentParser(prog="seti.py radio_sky alma")
    p.add_argument("mode", choices=("fetch", "search", "record"))
    p.add_argument("--max-gb", type=float, default=1.5)
    p.add_argument("--units", nargs="*", help="fetch only these member OUS ids, at any size up to --max-gb")
    a = p.parse_args()
    if a.mode == "fetch":
        alma_fetch_main(a.max_gb, set(a.units or ()))
    elif a.mode == "record":
        alma_record_main()
    else:
        alma_search_main()


# --------------------------------------------------------------------------------------------------------------

lpt_DOC = r"""Gaia stars inside long-period radio transient localisations at consistent distances, against ring controls.

    python -I seti.py radio_sky lpt match
    python -I seti.py radio_sky lpt record"""

lpt_OUT = core.results("lpt_gaia.json")
# name, position, localisation radius (arcsec), period (min), DM distance (kpc), nature
lpt_SOURCES = [
    ("GCRT J1745-3009", ("eq", "17 45 05.0", "-30 09 54"), 10.0, 77.01, None, ""),
    ("GLEAM-X J1627-5235", ("eq", "16 27 59.5", "-52 35 04.3"), 2.0, 18.18, (1.3, 0.5), ""),
    ("GPM J1839-10", ("eq", "18 39 02.00", "-10 31 49.37"), 1.0, 21.97, (5.7, 2.9), "binary, 8.75 h orbit"),
    ("ASKAP J1935+2148", ("eq", "19 35 05.1", "+21 48 41.0"), 3.0, 53.76, (4.3, 1.3), "three emission states"),
    ("CHIME J0630+25", ("eq", "06 30 38.4", "+25 26 24"), 120.0, 7.02, (0.17, 0.2), "spin-down and a glitch"),
    ("ASKAP/DART J1832-0911", ("gal", 22.6406, -0.0839), 3.0, 44.27, (4.5, 1.5), "X-ray outburst at the radio period"),
    ("ILT J1101+5521", ("gal", 150.4551, 55.5199), 3.0, 125.53, (0.50, 0.15), "white dwarf and M dwarf, period = orbit"),
    ("GLEAM-X J0704-37", ("eq", "07 04 13.19", "-37 06 14.3"), 1.0, 174.94, (1.5, 0.5),
     "white dwarf and M dwarf, period = orbit"),
    ("ASKAP J1839-0756", ("eq", "18 39 50.573", "-07 56 39.17"), 1.0, 387.02, (4.0, 1.2), "interpulse at half the period"),
    ("ASKAP J1448-6856", ("gal", 313.1644, -8.4338), 3.0, 93.85, None, "white dwarf and M dwarf"),
    ("CHIME/ILT J1634+44", ("gal", 70.1692, 42.5754), 2.0, 14.02, (2.6, 1.7), "spin-up, fully circular"),
    ("ASKAP J1755-2527", ("eq", "17 55 34.87", "-25 27 49.1"), 3.0, 69.77, None, ""),
    ("ASKAP J1424-6126", ("eq", "14 24 31.2", "-61 26 11"), 3.0, 35.79, None, ""),
    ("ASKAP J1745-5051", ("eq", "17 45 08.9", "-50 51 49"), 3.0, 80.68, None, "accreting white dwarf, X-rays at the orbit"),
    ("ASKAP J1651-4505", ("eq", "16 51 30.3", "-45 05 20"), 3.0, 388.6, None, ""),
    ("ASKAP J1700-4457", ("eq", "17 00 36.6", "-44 57 58"), 3.0, 281.6, None, ""),
]


def lpt_radec(pos):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if pos[0] == "gal":
        c = SkyCoord(l=pos[1] * u.deg, b=pos[2] * u.deg, frame="galactic").icrs
    else:
        c = SkyCoord(pos[1], pos[2], unit=(u.hourangle, u.deg), frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def lpt_at_distance(r, dist):
    """Whether a parallax reaches the DM distance range within 2 sigma; any star without a distance."""
    if not dist:
        return True
    plx, e = num(r["parallax"]), num(r["parallax_error"])
    if plx is None or not e:
        return False
    lo, hi = 1.0 / (dist[0] + 2 * dist[1]), 1.0 / max(dist[0] - 2 * dist[1], 0.01)
    return plx + 2 * e >= lo and plx - 2 * e <= hi


def lpt_match_main():
    out = []
    for name, pos, rad, per, dist, nature in lpt_SOURCES:
        ra, dec = lpt_radec(pos)
        ring_r = max(60.0, 4 * rad)
        rows = core.gaia("SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag "
                             "FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
                             "CIRCLE('ICRS', %.6f, %.6f, %.6f))" % (ra, dec, (ring_r + 2 * rad) / 3600.0))
        stars = []
        for r in rows:
            if not lpt_at_distance(r, dist):
                continue
            d = float(r["dec"]) + num(r["pmdec"], 0.0) / 3.6e6 * 8.0
            a = float(r["ra"]) + num(r["pmra"], 0.0) / 3.6e6 * 8.0 / math.cos(math.radians(d))
            stars.append((a, d, r))
        hits = [r for a, d, r in stars if sep_arcsec(ra, dec, a, d) <= rad]
        ring = []
        for k in range(24):
            th = 2 * math.pi * k / 24
            d0 = dec + ring_r / 3600.0 * math.sin(th)
            a0 = ra + ring_r / 3600.0 * math.cos(th) / math.cos(math.radians(dec))
            ring.append(sum(sep_arcsec(a0, d0, a, d) <= rad for a, d, _ in stars))
        chance = float(np.mean(ring))
        p = 1.0 - math.exp(-chance) * sum(chance ** j / math.factorial(j) for j in range(len(hits))) if hits else 1.0
        out.append(dict(name=name, ra=ra, dec=dec, radius=rad, period_min=per, dm_kpc=dist, nature=nature,
                        n=len(hits), chance=chance, p=p, stars=[dict(source_id=str(r["source_id"]),
                        parallax=num(r["parallax"]), parallax_error=num(r["parallax_error"]),
                        g=num(r["phot_g_mean_mag"]), sep=sep_arcsec(ra, dec, float(r["ra"]), float(r["dec"])))
                        for r in hits]))
        print("%-22s %4.0f\" %d star(s)%s against %.2f by chance (p %.2f)%s  %s" % (
            name, rad, len(hits), " at the DM distance" if dist else "", chance, p,
            "".join(" [plx %.2f+-%.2f G %.1f]" % (s["parallax"] or 0, s["parallax_error"] or 0, s["g"] or 0)
                    for s in out[-1]["stars"][:6]), nature or "no counterpart"), flush=True)
    json.dump(out, open(lpt_OUT, "w"), indent=1)


def lpt_record_main():
    d = json.load(open(lpt_OUT))
    explained = [x for x in d if x["nature"]]
    hosts = [x for x in d if not x["nature"] and x["n"] and x["p"] < 0.05]
    core.record_search(
        "lpt_gaia", "Long-period radio transients as beacons on stars", archive="published LPT positions; Gaia DR3",
        selection="the twelve LPTs of arXiv:2601.10393 and four ASKAP LPTs of 2026", code=os.path.abspath(__file__),
        entry="lpt", n_targets=len(d), params=dict(ring_radii=4, ring_min_arcsec=60, ring_points=24, epoch=2024.0,
                                                   parallax_sigma=2),
        null_summary="the same count at 24 positions four localisation radii out",
        sensitivity="Gaia DR3, G < 21", ledger="Long-period radio transients",
        result="%d of %d LPTs given a natural nature by their papers; %d of the rest with a Gaia star at the "
               "dispersion-measure distance beyond the chance rate" % (len(explained), len(d), len(hosts)))
    core.clear_targets("lpt_gaia")
    core.record_targets("lpt_gaia", [dict(target="lpt:" + x["name"], stat=x["n"], p=x["p"],
                                              unit="Gaia stars within the localisation",
                                              vet=("natural" if x["nature"] else None),
                                              note=(x["nature"] or "no counterpart") + "; %.2f by chance" % x["chance"])
                                         for x in d])
    print("%d LPTs; %d with a natural nature; host candidates: %s" % (len(d), len(explained),
                                                                     ", ".join(x["name"] for x in hosts) or "none"))


def lpt_cli():
    p = argparse.ArgumentParser(prog="seti.py radio_sky lpt")
    p.add_argument("mode", choices=("match", "record"))
    a = p.parse_args()
    lpt_record_main() if a.mode == "record" else lpt_match_main()


# --------------------------------------------------------------------------------------------------------------

wow_DOC = r"""HI4PI H I at -74 km/s in the two Wow! fields (Mendez et al. 2025): brightness, line limit, column density, nearest emission.

    python -I seti.py radio_sky wow [record] [--v-lsr -74] [--dv 5] [--average-arcmin 10] [--search-deg 3] [--nearest-sigma 5]"""

wow_FIELDS = (("19h25m02s", 291.2583), ("19h27m55s", 291.9792))  # RA (deg)
wow_DEC = -26.95
wow_V = -74.0  # km/s, LSR
wow_DV = 5.0  # km/s either side


def wow_main(record=False, v_lsr=wow_V, dv=wow_DV, average_arcmin=10.0, search_deg=3.0, nearest_sigma=5.0):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    sky = hi4pi_Sky()
    v = (np.arange(hi4pi_NCH) + 1 - hi4pi_CRPIX) * hi4pi_CDELT
    win = (v >= v_lsr - dv) & (v <= v_lsr + dv)
    far = (np.abs(v) > 200) & (np.abs(v) < 400)  # emission-free valid channels
    ch = int(round(hi4pi_channel(v_lsr)))
    out = []
    for name, ra in wow_FIELDS:
        c = SkyCoord(ra=ra * u.deg, dec=wow_DEC * u.deg, frame="icrs").galactic
        glon, b = float(c.l.deg), float(c.b.deg)
        pix = sky.near(glon, b, average_arcmin)
        spec = np.nanmean([sky.spectrum(p) for p in pix], axis=0)
        ok = np.isfinite(spec)
        noise = 1.4826 * float(np.nanmedian(np.abs(spec[far & ok] - np.nanmedian(spec[far & ok]))))
        main = int(np.nanargmax(np.where(ok, spec, -np.inf)))
        cb = math.cos(math.radians(b))
        hits, stack, pix_noise = [], [], []
        for p in sky.near(glon, b, search_deg * 60.0):
            s = sky.spectrum(p)
            stack.append(s)
            # each pixel against its own emission-free noise
            f = s[far & np.isfinite(s)]
            pn = 1.4826 * float(np.median(np.abs(f - np.median(f))))
            pix_noise.append(pn)
            m = float(np.nanmax(s[win]))
            if m > nearest_sigma * pn:
                pl, pb = sky.lonlat(p)
                hits.append((math.hypot(((pl - glon + 180) % 360 - 180) * cb, pb - b), m, pl, pb, float(v[win][np.nanargmax(s[win])])))
        near = min(hits) if hits else None
        # mean spectrum of every pixel within the search radius
        st = np.nanmean(stack, axis=0)
        st_noise = 1.4826 * float(np.nanmedian(np.abs(st[far] - np.nanmedian(st[far]))))
        stack_mean = float(np.nanmean(st[win]))
        # brightest emission at the velocity within the field's uncertainty
        box = []
        for p in sky.near(glon, b, 24.0):
            pl, pb = sky.lonlat(p)
            q = SkyCoord(l=pl * u.deg, b=pb * u.deg, frame="galactic").icrs
            if abs(float(q.dec.deg) - wow_DEC) * 60 <= 20.0 and abs((float(q.ra.deg) - ra) * math.cos(math.radians(wow_DEC))) * 60 <= 3.0:
                s = sky.spectrum(p)
                box.append((float(np.nanmax(s[win])), float(v[win][np.nanargmax(s[win])])))
        box_max = max(box) if box else (float("nan"), float("nan"))
        x = dict(field=name, ra=ra, dec=wow_DEC, l=glon, b=b, n_pix=len(pix), noise_k=noise, t_at_v=float(spec[ch]),
                 box_pixels=len(box), box_max_k=box_max[0], box_max_v=box_max[1],
                 peak_window_k=float(np.nanmax(spec[win])), v_peak_window=float(v[win][np.nanargmax(spec[win])]),
                 n_hi_limit=1.823e18 * 3 * noise * hi4pi_CDELT, v_main=float(v[main]), t_main=float(spec[main]),
                 pixel_noise_k=float(np.median(pix_noise)), n_pixels_3deg=len(stack), stack_mean_k=stack_mean, stack_noise_k=st_noise,
                 stack_n_hi=1.823e18 * stack_mean * hi4pi_CDELT * int(win.sum()),
                 n_emitting_3deg=len(hits), nearest=dict(sep_deg=near[0], t_k=near[1], l=near[2], b=near[3], v=near[4]) if near else None)
        out.append(x)
        print("%s -26d57m (l %.2f b %.2f): T at %.0f km/s %.3f K against noise %.3f K; peak within +-%.0f km/s %.3f K at %.1f km/s; "
              "N(HI) of a one-channel line below %.1e cm-2; the field's emission peaks at %.1f km/s with %.1f K; within the field's "
              "+-20 arcmin the brightest pixel at the velocity holds %.3f K at %.1f km/s (%d pixels, single-pixel noise %.3f K); %d of the "
              "%d pixels within %g deg emit at the velocity above %g sigma of their own noise, nearest %s; their mean spectrum holds %.4f K "
              "over the window (noise %.4f K a channel), N(HI) %.1e cm-2"
              % (name, glon, b, v_lsr, x["t_at_v"], noise, dv, x["peak_window_k"], x["v_peak_window"], x["n_hi_limit"], x["v_main"],
                 x["t_main"], x["box_max_k"], x["box_max_v"], len(box), x["pixel_noise_k"], len(hits), len(stack), search_deg, nearest_sigma,
                 ("%.2f deg away, %.2f K at %.1f km/s" % (near[0], near[1], near[4])) if near else "none",
                 stack_mean, st_noise, x["stack_n_hi"]))
    json.dump(out, open(core.results("wow_hi4pi.json"), "w"), indent=1)
    if record:
        core.record_search(
            "wow_hi4pi_cloud", "The Wow! signal's cloud", archive="HI4PI HPX tables (LSRK)", release="HI4PI",
            selection="the two fields of Arecibo Wow! II (2025)", code=os.path.abspath(__file__), entry="wow", n_targets=2,
            params=dict(v_lsr=v_lsr, dv=dv, average_arcmin=average_arcmin, search_deg=search_deg, nearest_sigma=nearest_sigma),
            null_summary="the channels beyond +-250 km/s give the noise; each field's own emission at %s km/s"
                         % " and ".join("%.0f" % x["v_main"] for x in out),
            sensitivity="a one-channel (1.29 km/s) line at 3 sigma: N(HI) %s cm-2" % " and ".join("%.1e" % x["n_hi_limit"] for x in out),
            ledger="The Wow! signal's cloud",
            result="; ".join("%s: %.3f K at %g km/s (noise %.3f K), peak %.3f K within %g km/s, N(HI) below %.1e; within the field's "
                             "+-20 arcmin the brightest pixel at the velocity holds %.3f K against a single-pixel noise of %.3f K; pixels within "
                             "%g deg above %g sigma of their own noise at the velocity: %s; the %g deg mean holds %.4f K, N(HI) %.1e"
                             % (x["field"], x["t_at_v"], v_lsr, x["noise_k"], x["peak_window_k"], dv, x["n_hi_limit"], x["box_max_k"], x["pixel_noise_k"],
                                search_deg, nearest_sigma,
                                ("%d, nearest %.2f deg away (%.2f K)" % (x["n_emitting_3deg"], x["nearest"]["sep_deg"], x["nearest"]["t_k"])) if x["nearest"] else "none",
                                search_deg, x["stack_mean_k"], x["stack_n_hi"])
                             for x in out))
        core.clear_targets("wow_hi4pi_cloud")
        core.record_targets("wow_hi4pi_cloud", [dict(
            target="sky:ra %.4f dec %+.4f" % (x["ra"], x["dec"]), stat=x["t_at_v"], upper_limit=x["n_hi_limit"],
            unit="K at %g km/s; N(HI) cm-2 of a one-channel line at 3 sigma" % v_lsr,
            note="field %s; the field's emission peaks at %.1f km/s (%.1f K); nearest emission at the velocity %s"
                 % (x["field"], x["v_main"], x["t_main"], ("%.2f deg away" % x["nearest"]["sep_deg"]) if x["nearest"] else "none within %g deg" % search_deg))
            for x in out])
        print("recorded")


polceiling_DOC = r"""NVSS sources polarised above the 72 per cent synchrotron ceiling at 3 sigma, vetted by size, pulsars, bright neighbours and SIMBAD, against a Rician null.

    python -I seti.py radio_sky polceiling fetch
    python -I seti.py radio_sky polceiling analyse
    python -I seti.py radio_sky polceiling record"""

polceiling_ROOT = os.environ.get("POLCEILING_ROOT", core.data("polceiling"))
polceiling_OUT = "polarisation_ceiling.json"
polceiling_SMIN, polceiling_PMIN, polceiling_CEILING, polceiling_BRIGHT = 10.0, 6.0, 0.72, 500.0


def polceiling_fetch():
    """NVSS sources above 10 mJy with polarised flux above 6 mJy."""
    from astroquery.vizier import Vizier
    os.makedirs(polceiling_ROOT, exist_ok=True)
    p = os.path.join(polceiling_ROOT, "nvss_polarised.json")
    if os.path.exists(p):
        print("have", p)
        return
    v = Vizier(columns=["NVSS", "RAJ2000", "DEJ2000", "S1.4", "e_S1.4", "MajAxis", "MinAxis", "l_MajAxis", "polFlux", "e_polFlux", "polPA", "resFlux"],
               column_filters={"S1.4": ">%g" % polceiling_SMIN, "polFlux": ">%g" % polceiling_PMIN}, row_limit=-1, timeout=1800)
    t = v.get_catalogs("VIII/65/nvss")[0]
    rows = []
    for r in t:
        rows.append({c: (None if np.ma.is_masked(r[c]) else (r[c].item() if hasattr(r[c], "item") else str(r[c]))) for c in t.colnames})
    json.dump(rows, open(p, "w"), default=str)
    print("%d sources" % len(rows))


def polceiling_deg(v, hours):
    """Degrees from a sexagesimal string or number."""
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).replace(":", " ").split()
    sign = -1.0 if parts[0].strip().startswith("-") else 1.0
    d = abs(float(parts[0])) + (float(parts[1]) if len(parts) > 1 else 0.0) / 60 + (float(parts[2]) if len(parts) > 2 else 0.0) / 3600
    return d * 15.0 if hours else sign * d


def polceiling_analyse():
    rows = json.load(open(os.path.join(polceiling_ROOT, "nvss_polarised.json")))
    src = []
    for r in rows:
        S, eS, P, eP = (num(r.get(k)) for k in ("S1.4", "e_S1.4", "polFlux", "e_polFlux"))
        if not S or not P or not eP:
            continue
        pd = math.sqrt(max(P ** 2 - eP ** 2, 0.0))
        low = (pd - 3 * eP) / (S + 3 * (eS or 0.0))
        src.append(dict(name=r["NVSS"], ra=polceiling_deg(r["RAJ2000"], True), dec=polceiling_deg(r["DEJ2000"], False), S=S, eS=eS, P=P, eP=eP, p=pd / S, p_low=low,
                        maj=num(r.get("MajAxis")), unresolved=str(r.get("l_MajAxis") or "").strip() == "<", pa=num(r.get("polPA")), res=num(r.get("resFlux"))))
    cands = [s for s in src if s["p_low"] > polceiling_CEILING]
    # Rician null: every source at the ceiling with its own errors
    rng = np.random.default_rng(6)
    false = 0
    for s in src:
        P0 = polceiling_CEILING * s["S"]
        Po = np.hypot(P0 + rng.normal(0, s["eP"], 200), rng.normal(0, s["eP"], 200))
        pd = np.sqrt(np.maximum(Po ** 2 - s["eP"] ** 2, 0))
        false += float(np.mean((pd - 3 * s["eP"]) / (s["S"] + 3 * (s["eS"] or 0.0)) > polceiling_CEILING))
    print("%d NVSS sources above %.0f mJy with polarised flux above %.0f mJy: debiased fraction median %.3f, 99th percentile %.3f; %d with the lower bound above %.2f (the errors alone would give %.2f at the ceiling)" % (
        len(src), polceiling_SMIN, polceiling_PMIN, np.median([s["p"] for s in src]), np.percentile([s["p"] for s in src], 99), len(cands), polceiling_CEILING, false))
    if cands:
        pulsars = pulsars_pulsars()
        pra, pdec = np.array([p["ra"] for p in pulsars]), np.array([p["dec"] for p in pulsars])
        for s in cands:
            sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(pdec)) * math.sin(math.radians(s["dec"])) + np.cos(np.radians(pdec)) * math.cos(math.radians(s["dec"])) * np.cos(np.radians(pra - s["ra"])), -1, 1))) * 3600
            i = int(np.argmin(sep))
            s["pulsar"] = dict(name=pulsars[i]["name"], sep=float(sep[i]), s1400=pulsars[i]["s1400"]) if sep[i] < max(30.0, pulsars[i]["err"]) else None
            # polarised flux at or above the total is the field's diffuse emission
            s["beyond_unity"] = s["p"] >= 0.95
            if s["beyond_unity"]:
                s["bright"], s["simbad"] = [], []
                print("  %s (%.5f %+.5f): S %.1f +- %.1f mJy, P %.1f +- %.1f, debiased %.2f, %s: polarised flux at or above the total, the field's diffuse polarised emission" % (
                    s["name"], s["ra"], s["dec"], s["S"], s["eS"] or 0, s["P"], s["eP"], s["p"], "unresolved" if s["unresolved"] else "size %s arcsec" % s["maj"]))
                continue
            # bright neighbours within 10 arcmin
            try:
                near = core.tap("vizier", 'SELECT "NVSS", "S1.4" FROM "VIII/65/nvss" WHERE "S1.4" > %g AND 1 = CONTAINS(POINT(\'ICRS\', "RAJ2000", "DEJ2000"), CIRCLE(\'ICRS\', %.6f, %.6f, %.4f))'
                               % (polceiling_BRIGHT, s["ra"], s["dec"], 10.0 / 60), timeout=120)
                s["bright"] = [(x["NVSS"], num(x["S1.4"])) for x in near if x["NVSS"] != s["name"]]
            except Exception as e:
                s["bright"] = "query failed: %s" % str(e)[:60]
            try:
                sb = core.simbad("SELECT main_id, otype, sp_type FROM basic WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.5f)) = 1" % (s["ra"], s["dec"], 30.0 / 3600))
                s["simbad"] = [(x["main_id"], x["otype"]) for x in sb]
            except Exception as e:
                s["simbad"] = "query failed: %s" % str(e)[:60]
            print("  %s (%.5f %+.5f): S %.1f +- %.1f mJy, P %.1f +- %.1f, debiased %.2f, lower bound %.2f, %s; pulsar %s; bright within 10 arcmin %s; SIMBAD %s" % (
                s["name"], s["ra"], s["dec"], s["S"], s["eS"] or 0, s["P"], s["eP"], s["p"], s["p_low"], "unresolved" if s["unresolved"] else "size %s arcsec" % s["maj"],
                ("%s at %.0f arcsec" % (s["pulsar"]["name"], s["pulsar"]["sep"])) if s["pulsar"] else "none", s["bright"], s["simbad"]))
    json.dump(dict(n=len(src), n_cand=len(cands), false=false, cands=cands, params=dict(smin=polceiling_SMIN, pmin=polceiling_PMIN, ceiling=polceiling_CEILING, bright=polceiling_BRIGHT)),
              open(core.results(polceiling_OUT), "w"), indent=0, default=str)


def polceiling_record():
    d = json.load(open(core.results(polceiling_OUT)))
    cands = d["cands"]
    core.record_search(
        "nvss_polarisation_ceiling", "Linear polarisation above the synchrotron ceiling", archive="NVSS (VIII/65) through VizieR; ATNF pulsar catalogue; SIMBAD", release="NVSS",
        selection="sources above %.0f mJy with polarised flux above %.0f mJy (%d)" % (d["params"]["smin"], d["params"]["pmin"], d["n"]), code=os.path.abspath(__file__), entry="polceiling",
        n_targets=d["n"], params=d["params"], null_summary="every source placed at the ceiling with its own Rician errors: %.2f false positives" % d["false"],
        sensitivity="a debiased polarised fraction whose lower bound at three sigma exceeds %.2f" % d["params"]["ceiling"], ledger="Linear polarisation above the synchrotron ceiling",
        result="%d sources with the lower bound above %.2f: %d with the polarised flux at or above the total, the field's diffuse polarised emission, %d at known pulsars, %d within 10 arcmin of a source above %.0f mJy, %d unresolved%s" % (
            len(cands), d["params"]["ceiling"], sum(1 for s in cands if s.get("beyond_unity")), sum(1 for s in cands if s.get("pulsar")), sum(1 for s in cands if isinstance(s.get("bright"), list) and s["bright"]), d["params"]["bright"],
            sum(1 for s in cands if s["unresolved"]), ("; " + "; ".join("%s %.2f (%s)" % (s["name"], s["p"], ("pulsar " + s["pulsar"]["name"]) if s.get("pulsar") else ("beyond unity" if s.get("beyond_unity") else ("beside %s" % s["bright"][0][0] if isinstance(s.get("bright"), list) and s["bright"] else "unexplained"))) for s in cands[:12])) if cands else ""))
    core.clear_targets("nvss_polarisation_ceiling")
    core.record_targets("nvss_polarisation_ceiling", [dict(target="nvss:%s" % s["name"], stat=s["p_low"], unit="lower bound of the debiased polarised fraction",
                                                                 vet=("pulsar %s" % s["pulsar"]["name"]) if s.get("pulsar") else ("polarised flux at or above the total: the field's diffuse polarised emission" if s.get("beyond_unity") else (("bright source within 10 arcmin") if isinstance(s.get("bright"), list) and s["bright"] else None)),
                                                                 note="S %.1f mJy, P %.1f +- %.1f, debiased %.2f, %s" % (s["S"], s["P"], s["eP"], s["p"], "unresolved" if s["unresolved"] else "resolved")) for s in cands])
    print("recorded %d candidates" % len(cands))


def polceiling_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": polceiling_fetch, "analyse": polceiling_analyse, "record": polceiling_record}[mode]()


gleamx_DOC = r"""Step edges in GLEAM-X sub-band spectra: top-0.1-per-cent chi-square gain of a within-setting step over a curved power law, unshared by neighbours.

    python -I seti.py radio_sky gleamx analyse
    python -I seti.py radio_sky gleamx record"""

gleamx_ROOT = os.environ.get("GLEAMX_ROOT", core.data("gleamx"))
gleamx_OUT = "gleamx_edges.json"
gleamx_BANDS = ("076", "084", "092", "099", "107", "115", "122", "130", "143", "151", "158", "166", "174", "181", "189", "197", "204", "212", "220", "227")
gleamx_SETTINGS = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)  # five 30 MHz settings
gleamx_SNR, gleamx_STEP, gleamx_TAIL = 20.0, 0.30, 0.001


def gleamx_fit(x, y, w, step_at=None):
    """Weighted fit of a + b x + c x^2 (+ d past step_at): chi-square and coefficients."""
    cols = [np.ones_like(x), x, x * x]
    if step_at is not None:
        cols.append((np.arange(len(x)) > step_at).astype(float))
    A = np.column_stack(cols) * np.sqrt(w)[:, None]
    c, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    r = y * np.sqrt(w) - A @ c
    return float(np.sum(r * r)), c


def gleamx_analyse():
    from astropy.table import Table
    from scipy.spatial import cKDTree
    t = Table.read(os.path.join(gleamx_ROOT, "gleamx_dr1.ecsv"), format="ascii.ecsv")
    if "Fint092" not in t.colnames:
        # 92 MHz columns fetched separately and joined
        side = Table.read(os.path.join(gleamx_ROOT, "gleamx_dr1_092.ecsv"), format="ascii.ecsv")
        look = {str(n): (s, e) for n, s, e in zip(side["GLEAM-X"], side["Fpint092"], side["e_Fpint092"])}
        t["Fint092"] = [float(look.get(str(n), (np.nan, np.nan))[0]) if look.get(str(n), (None,))[0] is not None else np.nan for n in t["GLEAM-X"]]
        t["e_Fint092"] = [float(look.get(str(n), (np.nan, np.nan))[1]) if look.get(str(n), (None, None))[1] is not None else np.nan for n in t["GLEAM-X"]]
    nu = np.log10(np.array([float(b) for b in gleamx_BANDS]))
    S = np.column_stack([np.array(t["Fint%s" % b], float) for b in gleamx_BANDS])
    E = np.column_stack([np.array(t["e_Fint%s" % b], float) for b in gleamx_BANDS])
    wide, e_wide = np.array(t["Fintwide"], float), np.array(t["e_Fintwide"], float)
    ra, dec = np.array(t["RAJ2000"], float), np.array(t["DEJ2000"], float)
    names = [str(x) for x in t["GLEAM-X"]]
    good = np.isfinite(wide) & np.isfinite(e_wide) & (wide / np.maximum(e_wide, 1e-9) > gleamx_SNR)
    print("%d sources, %d above %.0f sigma in the wide band" % (len(t), good.sum(), gleamx_SNR), flush=True)
    inner = [k for k in range(19) if gleamx_SETTINGS[k] == gleamx_SETTINGS[k + 1]]
    rows = []
    for i in np.where(good)[0]:
        s, e = S[i], E[i]
        ok = np.isfinite(s) & np.isfinite(e) & (s > 0) & (e > 0)
        if ok.sum() < 16:
            continue
        y = np.log10(s[ok])
        w = (s[ok] / e[ok]) ** 2 / (np.log(10) ** 2)
        x = nu[ok]
        chi0, c0 = gleamx_fit(x, y, w)
        best = (0.0, None, 0.0)
        idx = np.where(ok)[0]
        for k in inner:
            # step after band k in kept-band index
            pos = np.searchsorted(idx, k, side="right") - 1
            if pos < 2 or pos > len(idx) - 4:
                continue
            chi, c = gleamx_fit(x, y, w, step_at=pos)
            if chi0 - chi > best[0]:
                best = (chi0 - chi, k, float(10 ** c[3] - 1))
        resid = y - (c0[0] + c0[1] * x + c0[2] * x * x)
        rows.append(dict(i=int(i), name=names[i], ra=ra[i], dec=dec[i], wide=float(wide[i]), snr=float(wide[i] / e_wide[i]), n=int(ok.sum()), chi_curved=chi0,
                         gain=best[0], boundary=best[1], step=best[2], scatter=float(1.4826 * np.median(np.abs(resid))), alpha=float(c0[1]), curv=float(c0[2])))
    gains = np.array([r["gain"] for r in rows])
    thr = float(np.quantile(gains, 1 - gleamx_TAIL))
    cand = [r for r in rows if r["gain"] >= thr and abs(r["step"]) >= gleamx_STEP and abs(np.log10(1 + r["step"])) >= 3 * r["scatter"]]
    # the same step at the boundary in the twenty nearest sources
    xyz = np.stack([np.cos(np.radians(dec)) * np.cos(np.radians(ra)), np.cos(np.radians(dec)) * np.sin(np.radians(ra)), np.sin(np.radians(dec))], axis=1)
    fitted = np.array([r["i"] for r in rows])
    tree = cKDTree(xyz[fitted])
    all_tree = cKDTree(xyz)
    for r in cand:
        _, nb = tree.query(xyz[r["i"]], k=21)
        steps = []
        for j in nb[1:]:
            ii = fitted[j]
            s, e = S[ii], E[ii]
            ok = np.isfinite(s) & np.isfinite(e) & (s > 0) & (e > 0)
            idx = np.where(ok)[0]
            pos = np.searchsorted(idx, r["boundary"], side="right") - 1
            if ok.sum() < 16 or pos < 2 or pos > len(idx) - 4:
                continue
            y = np.log10(s[ok])
            w = (s[ok] / e[ok]) ** 2 / (np.log(10) ** 2)
            steps.append(10 ** gleamx_fit(nu[ok], y, w, step_at=pos)[1][3] - 1)
        steps = np.array(steps)
        r["neighbour_step_median"] = float(np.median(steps)) if len(steps) else None
        r["neighbour_step_scatter"] = float(1.4826 * np.median(np.abs(steps - np.median(steps)))) if len(steps) else None
        r["shared"] = bool(len(steps) > 5 and abs(r["step"] - np.median(steps)) < 3 * max(r["neighbour_step_scatter"], 0.02))
        r["neighbour_steps"] = [float(x) for x in steps]
        # a second source within two beams splits and merges with the beam
        dd, jj = all_tree.query(xyz[r["i"]], k=2)
        other = int(jj[1])
        r["companion_arcsec"] = float(2 * np.degrees(np.arcsin(dd[1] / 2)) * 3600)
        r["companion_jy"] = float(wide[other]) if np.isfinite(wide[other]) else None
        r["blend"] = bool(r["companion_arcsec"] < 150 and (r["companion_jy"] or 0) > 0.2 * r["wide"])
    left = [r for r in cand if not r["shared"] and not r["blend"]]
    print("%d fitted; gain top %.1f per cent threshold %.1f; %d candidates with steps above %.0f per cent, %d shared with their neighbours at the same boundary, %d blended with a second source within two beams, %d left" % (
        len(rows), 100 * gleamx_TAIL, thr, len(cand), 100 * gleamx_STEP, sum(r["shared"] for r in cand), sum(r["blend"] for r in cand), len(left)))
    for r in sorted(cand, key=lambda r: -r["gain"])[:25]:
        print("  %s (%.4f %+.4f): %.2f Jy at %.0f sigma, step %+.0f per cent after %s MHz (gain %.0f, band scatter %.3f dex, alpha %.2f); neighbours' step there %s +- %s; nearest source %.0f arcsec at %s Jy%s%s" % (
            r["name"], r["ra"], r["dec"], r["wide"], r["snr"], 100 * r["step"], gleamx_BANDS[r["boundary"]], r["gain"], r["scatter"], r["alpha"],
            "%.2f" % r["neighbour_step_median"] if r["neighbour_step_median"] is not None else "-", "%.2f" % r["neighbour_step_scatter"] if r["neighbour_step_scatter"] is not None else "-",
            r["companion_arcsec"], "%.2f" % r["companion_jy"] if r["companion_jy"] is not None else "-", "; shared" if r["shared"] else "", "; blend" if r["blend"] else ""))
    json.dump(dict(n=len(t), n_fitted=len(rows), threshold=thr, n_cand=len(cand), n_left=len(left), cands=cand, params=dict(snr=gleamx_SNR, step=gleamx_STEP, tail=gleamx_TAIL),
                   gain_percentiles={str(p): float(np.percentile(gains, p)) for p in (50, 90, 99, 99.9)}), open(core.results(gleamx_OUT), "w"), indent=0)


def gleamx_record():
    d = json.load(open(core.results(gleamx_OUT)))
    left = [r for r in d["cands"] if not r["shared"] and not r["blend"]]
    core.record_search(
        "gleamx_edges", "Brick-wall spectral edges in GLEAM-X", archive="GLEAM-X DR1 (VizieR VIII/110), twenty sub-bands 72-231 MHz", release="GLEAM-X DR1",
        selection="%d sources above %.0f sigma in the wide band with sixteen or more sub-bands (%d fitted)" % (d["n_fitted"], d["params"]["snr"], d["n_fitted"]), code=os.path.abspath(__file__), entry="gleamx",
        n_targets=d["n_fitted"], params=d["params"], null_summary="the chi-square gain of the best within-setting step over a curved power law across the sample: median %.1f, 99th percentile %.1f, 99.9th %.1f" % (
            d["gain_percentiles"]["50"], d["gain_percentiles"]["99"], d["gain_percentiles"]["99.9"]),
        sensitivity="a step above %.0f per cent of the flux and three times the band scatter within one observing setting" % (100 * d["params"]["step"]), ledger="Brick-wall spectral edges in GLEAM-X",
        result="%d candidates in the top %.1f per cent of gains with steps above %.0f per cent: %d shared with their neighbours at the same boundary, %d blended with a second source within two beams, %d left%s" % (
            d["n_cand"], 100 * d["params"]["tail"], 100 * d["params"]["step"], sum(r["shared"] for r in d["cands"]), sum(r["blend"] for r in d["cands"]), len(left),
            (": " + "; ".join("%s (%+.0f per cent after %s MHz)" % (r["name"], 100 * r["step"], gleamx_BANDS[r["boundary"]]) for r in left[:8])) if left else ""))
    core.clear_targets("gleamx_edges")
    core.record_targets("gleamx_edges", [dict(target="gleamx:%s" % r["name"], stat=r["gain"], unit="chi-square gain of a step over a curved power law",
                                                  vet="shared with neighbours" if r["shared"] else ("a second source %.0f arcsec away at %.2f Jy inside the low-frequency beam" % (r["companion_arcsec"], r["companion_jy"] or 0) if r["blend"] else None),
                                                  note="step %+.0f per cent after %s MHz, %.2f Jy at %.0f sigma" % (100 * r["step"], gleamx_BANDS[r["boundary"]], r["wide"], r["snr"])) for r in d["cands"]])
    print("recorded %d candidates" % d["n_cand"])


def gleamx_cli():
    quiet_warnings()
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"analyse": gleamx_analyse, "record": gleamx_record}[mode]()


# --------------------------------------------------------------------------------------------------------------
# pulsars that hold their period, and pulsars on nearby stars

spinup_DOC = r"""Pulsars that hold their period: isolated pulsars of the ATNF catalogue (VizieR B/psr) outside globular clusters and the Magellanic Clouds whose measured period derivative is zero or negative, a beacon kept at a constant period, each derivative read against its error; those whose derivative is negative only once corrected for the Galactic acceleration and the Shklovskii term at their DM distance (P1I) are counted beside them.

    python -I seti.py radio_sky spinup [record] [--sigma 3]"""


def spinup_main(record=False, sigma=3.0):
    rows = catalogues.pulsars()
    iso = [r for r in rows if catalogues.pulsar_isolated(r) and catalogues.pulsar_in_field(r)]
    measured = [r for r in iso if num(r["P1"]) is not None]
    out, corrected = [], []
    for r in measured:
        p1, e1, p1i = num(r["P1"]), num(r["e_P1"]), num(r["P1I"])
        if p1 <= 0:
            z = p1 / e1 if e1 else None
            out.append(dict(psr=r["PSRJ"], P0=num(r["P0"]), P1=p1, e_P1=e1, P1I=p1i, z=z, dm=num(r["DM"]), dist=num(r["Dist"]),
                            assoc=(r["Assoc"] or "").strip()))
        elif p1i is not None and p1i <= 0:
            corrected.append(r["PSRJ"])
    held = [o for o in out if o["z"] is not None and o["z"] < -sigma]
    print("%d pulsars, %d isolated in the field, %d of them with a period derivative; zero or negative as measured: %d; below zero at 3 "
          "sigma: %d; negative only once corrected at the DM distance: %d (%s)" % (len(rows), len(iso), len(measured), len(out), len(held),
                                                                                  len(corrected), ", ".join(corrected)))
    for o in out:
        print("  %s: P %.4f s, P1 %s +- %s (%s sigma), P1I %s, DM %s, %s kpc %s" % (
            o["psr"], o["P0"] or 0, o["P1"], o["e_P1"], "%.1f" % o["z"] if o["z"] is not None else "-", o["P1I"], o["dm"], o["dist"], o["assoc"]))
    json.dump(out, open(core.results("spinup_pulsars.json"), "w"), indent=1)
    if record:
        core.record_search(
            "atnf_spinup_pulsars", "Pulsars that hold their period", archive="ATNF pulsar catalogue (VizieR B/psr)",
            selection="isolated pulsars outside globular clusters and the Magellanic Clouds with a period derivative: %d" % len(measured),
            code=os.path.abspath(__file__), entry="spinup", n_targets=len(measured), n_unit="pulsars",
            params=dict(sigma=sigma), null_summary="each derivative's own error",
            sensitivity="an isolated field pulsar whose period derivative lies below zero at 3 sigma",
            ledger="Pulsars that hold their period",
            result="%d with a measured derivative zero or negative, %d below zero at 3 sigma; %d more negative only once corrected at "
                   "the DM distance (%s)" % (len(out), len(held), len(corrected), ", ".join(corrected)))
        core.clear_targets("atnf_spinup_pulsars")
        core.record_targets("atnf_spinup_pulsars", [dict(
            target="psr:%s" % o["psr"], stat=o["z"], unit="period derivative over its error",
            vet="not-measured" if o["z"] is None or o["z"] > -sigma else None,
            note="P1 %s +- %s, P1I %s" % (o["P1"], o["e_P1"], o["P1I"])) for o in out])
        print("recorded")


psrstar_DOC = r"""Pulsars on nearby stars: every ATNF pulsar (VizieR B/psr) matched to Gaia DR3 within 2 arcsec by CDS XMatch, the matches within 200 pc at 5 sigma in parallax, a beacon placed at a star, read for the pulsar's own binary companion and for the crowding of globular-cluster fields, against the matches beyond 200 pc as the chance rate of the fields.

    python -I seti.py radio_sky psrstar [record] [--radius-arcsec 2] [--plx-mas 5] [--plx-sigma 5]"""


def psrstar_main(record=False, radius_arcsec=2.0, plx_mas=5.0, plx_sigma=5.0):
    rows = catalogues.pulsars()
    up, by = [], {}
    for r in rows:
        ra, de = num(r["RAJ2000"]), num(r["DEJ2000"])
        if ra is not None and de is not None:
            up.append(dict(source_id=r["PSRJ"], ra=ra, dec=de))
            by[r["PSRJ"]] = r
    m = core.xmatch(up, "I/355/gaiadr3", radius_arcsec)
    near = [x for x in m if (num(x.get("Plx")) or 0) > plx_mas and (num(x.get("Plx")) or 0) > plx_sigma * (num(x.get("e_Plx")) or 1e9)]
    out = []
    for x in near:
        r = by[x["source_id"]]
        vet = []
        if not catalogues.pulsar_in_field(r):
            # a globular cluster lies kiloparsecs away: a star within 200 pc is in front of it
            vet.append("chance-rate")
        elif not catalogues.pulsar_isolated(r):
            vet.append("companion")
            if (r.get("BINCOMP") or "").strip()[:2] in ("He", "CO", "ON"):
                vet.append("white-dwarf")
        out.append(dict(psr=r["PSRJ"], gaia=x.get("Source"), plx=num(x.get("Plx")), G=num(x.get("Gmag")), sep=float(x["angDist"]),
                        binary=(r.get("Binary") or "").strip(), comp=(r.get("BINCOMP") or "").strip(), assoc=(r.get("Assoc") or "").strip(), vet=vet))
    print("%d pulsars, %d Gaia sources within 2 arcsec, %d within 200 pc at 5 sigma" % (len(up), len(m), len(out)))
    for o in out:
        print("  %s on Gaia DR3 %s: %.2f mas, G %.1f, %.2f arcsec; binary %s, companion %s; %s -> %s" % (
            o["psr"], o["gaia"], o["plx"], o["G"] or 0, o["sep"], o["binary"] or "-", o["comp"] or "-", o["assoc"][:50], o["vet"] or "unexplained"))
    json.dump(out, open(core.results("pulsars_on_stars.json"), "w"), indent=1)
    if record:
        core.record_search(
            "atnf_pulsars_on_stars", "Pulsars on nearby stars", archive="ATNF pulsar catalogue (VizieR B/psr); Gaia DR3 (I/355) by CDS XMatch",
            selection="every catalogued pulsar position against Gaia DR3 within 2 arcsec", code=os.path.abspath(__file__), entry="psrstar",
            n_targets=len(up), n_unit="pulsars", params=dict(radius_arcsec=radius_arcsec, plx_mas=plx_mas, plx_sigma=plx_sigma),
            null_summary="%d Gaia sources within 2 arcsec of %d pulsars, nearly all background stars of the Galactic plane" % (len(m), len(up)),
            sensitivity="a star within 200 pc inside 2 arcsec of a pulsar's position", ledger="Pulsars on nearby stars",
            result="%d within 200 pc: %s" % (len(out), "; ".join("%s (%s)" % (o["psr"], ", ".join(o["vet"]) or "unexplained") for o in out)))
        core.clear_targets("atnf_pulsars_on_stars")
        core.record_targets("atnf_pulsars_on_stars", [dict(
            target="psr:%s" % o["psr"], stat=o["sep"], unit="arcsec to the Gaia source", vet="; ".join(o["vet"]) or None,
            note="Gaia DR3 %s at %.2f mas, G %.1f" % (o["gaia"], o["plx"], o["G"] or 0)) for o in out])
        print("recorded")


psrclock_DOC = r"""Pulsars on one clock: pairs of ATNF pulsars (VizieR B/psr) with frequencies measured to 1e-7 and first derivatives, more than 1 deg apart and not in one globular cluster, whose rotation frequencies carried to MJD 56000 agree to 1e-6, a beacon period shared between sites, against the pairs within 1e-4 scaled to the tolerance, since a smooth distribution of frequencies gives pairs in proportion to it; catalogue periods given to a few digits pile onto round values and are left out.

    python -I seti.py radio_sky psrclock [record] [--tolerance 1e-6] [--precision 1e-7] [--epoch-mjd 56000] [--min-sep-deg 1]"""

def psrclock_main(record=False, tolerance=1e-6, precision=1e-7, epoch_mjd=56000.0, min_sep_deg=1.0):
    rows = catalogues.pulsars("PSRJ, RA2000, DE2000, P0, e_P0, F0, e_F0, F1, PEpoch, Assoc")
    good = []
    for r in rows:
        f0, f1, ep = num(r["F0"]), num(r["F1"]), num(r["PEpoch"])
        ef = num(r["e_F0"])
        if f0 is None and num(r["P0"]):
            f0 = 1.0 / num(r["P0"])
            ef = num(r["e_P0"]) / num(r["P0"]) ** 2 if num(r["e_P0"]) else None
        if f0 is None or f0 <= 0 or num(r["RA2000"]) is None or not ef or ef > precision * f0 or f1 is None or ep is None:
            continue
        f = f0 + f1 * (epoch_mjd - ep) * 86400.0
        good.append(dict(psr=r["PSRJ"], f=f, ef=ef, f1=f1, ra=num(r["RA2000"]), dec=num(r["DE2000"]), gc=(r["Assoc"] or "").split("GC:")[1].split(",")[0]
                         if "GC:" in (r["Assoc"] or "") else None))
    good.sort(key=lambda x: x["f"])
    t1, t2, t3 = tolerance, 10 * tolerance, 100 * tolerance
    counts, pairs = {t1: 0, t2: 0, t3: 0}, []
    for i, a in enumerate(good):
        for b in good[i + 1:]:
            d = (b["f"] - a["f"]) / a["f"]
            if d > t3:
                break
            if (a["gc"] and a["gc"] == b["gc"]) or core.sep_arcsec(a["ra"], a["dec"], b["ra"], b["dec"]) < 3600.0 * min_sep_deg:
                continue
            for t in counts:
                counts[t] += d < t
            if d < t2:
                e = np.hypot(a["ef"] or 0, b["ef"] or 0)
                pairs.append(dict(a=a["psr"], b=b["psr"], fa=a["f"], fb=b["f"], frac=d, sigma=(b["f"] - a["f"]) / e if e > 0 else None,
                                  carried=bool(a["f1"] and b["f1"])))
    expect = counts[t3] / 100.0
    print("%d pulsars with a frequency; pairs more than %g deg apart outside one cluster within %g: %d, %g: %d, %g: %d against %.2f "
          "scaled" % (len(good), min_sep_deg, t3, counts[t3], t2, counts[t2], t1, counts[t1], expect))
    for x in sorted(pairs, key=lambda x: x["frac"]):
        print("  %s %.9f Hz and %s %.9f Hz: %.1e apart, %s combined sigma%s" % (
            x["a"], x["fa"], x["b"], x["fb"], x["frac"], "%.0f" % x["sigma"] if x["sigma"] is not None else "-", "" if x["carried"] else ", not both carried"))
    json.dump(dict(n=len(good), counts={str(k): v for k, v in counts.items()}, expect=expect, pairs=pairs), open(core.results("pulsar_clocks.json"), "w"), indent=1)
    if record:
        core.record_search(
            "atnf_shared_clocks", "Pulsars on one clock", archive="ATNF pulsar catalogue (VizieR B/psr)",
            selection="pulsars with frequencies measured to 1e-7 and first derivatives: %d" % len(good), code=os.path.abspath(__file__), entry="psrclock",
            n_targets=len(good), n_unit="pulsars", params=dict(tolerance=tolerance, precision=precision, epoch_mjd=epoch_mjd, min_sep_deg=min_sep_deg),
            null_summary="pairs within %g (%d) scaled to %g: %.2f" % (t3, counts[t3], t1, expect),
            sensitivity="two pulsars more than %g deg apart on one frequency to %g at a common epoch" % (min_sep_deg, t1), ledger="Pulsars on one clock",
            result="%d pairs within %g against %.2f; %d within %g" % (counts[t1], t1, expect, counts[t2], t2))
        core.clear_targets("atnf_shared_clocks")
        core.record_targets("atnf_shared_clocks", [dict(
            target="psr:%s" % x["a"], stat=x["frac"], unit="fractional frequency difference",
            vet="chance-rate" if counts[t1] <= expect + 3 * np.sqrt(max(expect, 1)) else None,
            note="with %s, %s combined sigma" % (x["b"], "%.0f" % x["sigma"] if x["sigma"] is not None else "-")) for x in pairs])
        print("recorded")


frbstars_DOC = r"""Fast radio bursts at nearby stars: the CHIME/FRB Catalog 1 localisations (VizieR J/ApJS/257/59) against the sources of the Gaia Catalogue of Nearby Stars within 10 pc, a burst sent from a neighbour, each star inside a burst's 2-sigma ellipse counted against the same count with every burst moved by 5, 10, 15 and 20 deg in right ascension either way.

    python -I seti.py radio_sky frbstars [record] [--plx-mas 100] [--ellipse-sigma 2] [--shifts-deg 5 10 15 20]"""

def frbstars_inside(frbs, stars, shift=0.0, sigma=2.0):
    """The (burst, star, separation in deg) triples with the star inside the burst's SIGMA-sigma error ellipse, the bursts moved SHIFT deg in right ascension."""
    out = []
    for b in frbs:
        for s in stars:
            da = ((s["ra"] - b["ra"] - shift + 180.0) % 360.0) - 180.0
            dd = s["dec"] - b["dec"]
            if (da / b["era"]) ** 2 + (dd / b["edec"]) ** 2 < sigma ** 2:
                out.append((b["name"], s["source_id"], float(np.hypot(da * np.cos(np.radians(b["dec"])), dd))))
    return out


def frbstars_main(record=False, plx_mas=100.0, ellipse_sigma=2.0, shifts_deg=[5.0, 10.0, 15.0, 20.0]):
    rows = catalogues.frb_catalog1("Name, RAJ2000, e_RAJ2000, DEJ2000, e_DEJ2000, DM, DMeYMW16", one_per_burst=False)
    frbs = {}
    for r in rows:
        if r["Name"] not in frbs and num(r["e_RAJ2000"]) and num(r["e_DEJ2000"]):
            frbs[r["Name"]] = dict(name=r["Name"], ra=num(r["RAJ2000"]), dec=num(r["DEJ2000"]), era=num(r["e_RAJ2000"]), edec=num(r["e_DEJ2000"]),
                                   dm=num(r["DM"]))
    frbs = list(frbs.values())
    stars = catalogues.gcns("source_id, ra, dec, parallax", "parallax > %g" % plx_mas)
    stars = [dict(source_id=s["source_id"], ra=float(s["ra"]), dec=float(s["dec"]), plx=float(s["parallax"])) for s in stars]
    hits = frbstars_inside(frbs, stars, sigma=ellipse_sigma)
    shifts = [s for k in shifts_deg for s in (k, -k)]
    null = [len(frbstars_inside(frbs, stars, s, ellipse_sigma)) for s in shifts]
    area = float(np.sum([np.pi * ellipse_sigma ** 2 * b["era"] * b["edec"] * np.cos(np.radians(b["dec"])) for b in frbs]))
    print("%d bursts (2-sigma ellipses covering %.1f deg2 together), %d sources within 10 pc; inside an ellipse: %d against %.2f with the bursts "
          "moved (%s)" % (len(frbs), area, len(stars), len(hits), float(np.mean(null)), null))
    for h in hits:
        print("  %s holds Gaia DR3 %s, %.2f deg from the burst's position" % h)
    json.dump(dict(n_frb=len(frbs), n_star=len(stars), area_deg2=area, hits=hits, null=null), open(core.results("frb_nearby_stars.json"), "w"), indent=1)
    if record:
        core.record_search(
            "chime_frb_nearby_stars", "Fast radio bursts at nearby stars", archive="CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59); Gaia Catalogue of "
            "Nearby Stars", selection="%d bursts; %d sources within 10 pc" % (len(frbs), len(stars)), code=os.path.abspath(__file__),
            entry="frbstars", n_targets=len(frbs), n_unit="bursts", params=dict(ellipse_sigma=ellipse_sigma, shifts_deg=shifts, plx_mas=plx_mas),
            null_summary="the bursts moved in right ascension: %s" % null,
            sensitivity="a burst of CHIME's first catalogue whose 2-sigma ellipse holds a star within 10 pc", ledger="Fast radio bursts at nearby stars",
            result="%d stars inside an ellipse against %.2f with the bursts moved" % (len(hits), float(np.mean(null))))
        core.clear_targets("chime_frb_nearby_stars")
        core.record_targets("chime_frb_nearby_stars", [dict(
            target="gaia_dr3:%s" % h[1], stat=h[2], unit="deg from the burst's position",
            vet="chance-rate" if len(hits) <= np.mean(null) + 3 * np.sqrt(max(np.mean(null), 1)) else None, note="inside %s" % h[0]) for h in hits])
        print("recorded")


# --------------------------------------------------------------------------------------------------------------
# pulse shapes, dispersions, clocks and copies among pulsars and bursts


def psr_all():
    """Every pulsar with the timing, width, luminosity and orbit columns the catalogue-wide tests read."""
    return catalogues.pulsars("PSRJ, P0, F0, e_F0, F1, PEpoch, W50, W10, DM, RLUM14, Edot, PB, Assoc, Dist, DistDM")


def psr_record(run, approach, n, params, null, sens, result, hits, unit, stat, note):
    entry = {"atnf_square_pulses": "boxcar", "atnf_no_dispersion": "nodm", "atnf_breakup": "breakup", "atnf_overpowered": "overpower",
             "atnf_year_orbits": "yearpsr"}[run]
    core.record_run(run, approach, "ATNF pulsar catalogue (VizieR B/psr)", "catalogued pulsars with the quantity measured: %d" % n,
                    os.path.abspath(__file__), entry, n, "pulsars", params, null, sens, result,
                    [dict(target="psr:%s" % h["PSRJ"], stat=stat(h), unit=unit, note=note(h)) for h in hits])


boxcar_DOC = r"""Pulsars with square pulses: ATNF pulsars with both pulse widths measured whose width at 10 per cent of the peak is within 5 per cent of the width at 50 per cent, where a Gaussian pulse gives a ratio of 1.82 and a square one 1.0.

    python -I seti.py radio_sky boxcar [record] [--ratio-max 1.05]"""


def boxcar_main(record=False, ratio_max=1.05):
    rows = [r for r in psr_all() if num(r["W50"]) and num(r["W10"])]
    ratio = lambda r: num(r["W10"]) / num(r["W50"])  # noqa: E731
    hits = sorted([r for r in rows if ratio(r) < ratio_max], key=ratio)
    print("%d pulsars with both widths; W10/W50 below %g: %d, median ratio %.2f: %s" % (
        len(rows), ratio_max, len(hits), float(np.median([ratio(r) for r in rows])), ["%s %.3f (W50 %s ms)" % (r["PSRJ"], ratio(r), r["W50"]) for r in hits]))
    if record:
        psr_record("atnf_square_pulses", "Pulsars with square pulses", len(rows), dict(ratio=ratio_max), "a Gaussian pulse's 1.82",
                   "a pulse as wide at 10 per cent as at 50 per cent of its peak", "%d below 1.05" % len(hits), hits, "W10 over W50", ratio,
                   lambda r: "W50 %s ms, W10 %s ms" % (r["W50"], r["W10"]))
        print("recorded")


nodm_DOC = r"""Pulsars with no electrons in front: ATNF pulsars with a dispersion measure below 1 pc cm^-3, well under the 2.64 of J0437-4715, the nearest millisecond pulsar at 157 pc.

    python -I seti.py radio_sky nodm [record] [--dm-max 1]"""


def nodm_main(record=False, dm_max=1.0):
    rows = [r for r in psr_all() if num(r["DM"]) is not None]
    hits = sorted([r for r in rows if num(r["DM"]) < dm_max], key=lambda r: num(r["DM"]))
    print("%d pulsars with a dispersion measure; below %g: %d %s; the lowest five %s" % (
        len(rows), dm_max, len(hits), [(r["PSRJ"], r["DM"]) for r in hits], [(r["PSRJ"], r["DM"]) for r in sorted(rows, key=lambda r: num(r["DM"]))[:5]]))
    if record:
        psr_record("atnf_no_dispersion", "Pulsars with no electrons in front", len(rows), dict(dm=dm_max), "J0437-4715's 2.64 at 157 pc",
                   "a pulsar with almost no free electrons on its sightline", "%d below 1 pc cm^-3; the lowest %s" % (
                       len(hits), ", ".join("%s %s" % (r["PSRJ"], r["DM"]) for r in sorted(rows, key=lambda r: num(r["DM"]))[:3])),
                   hits, "dispersion measure, pc cm^-3", lambda r: num(r["DM"]), lambda r: "P %s s" % r["P0"])
        print("recorded")


breakup_DOC = r"""Pulsars spinning faster than a neutron star survives: ATNF periods below 0.7 ms, the break-up limit, against the fastest known at 1.396 ms.

    python -I seti.py radio_sky breakup [record] [--p-ms 0.7]"""


def breakup_main(record=False, p_ms=0.7):
    rows = [r for r in psr_all() if num(r["P0"])]
    hits = [r for r in rows if num(r["P0"]) < p_ms / 1000.0]
    fastest = sorted(rows, key=lambda r: num(r["P0"]))[:3]
    print("%d pulsars with a period; below %g ms: %d; the fastest %s" % (len(rows), p_ms, len(hits), [(r["PSRJ"], r["P0"]) for r in fastest]))
    if record:
        psr_record("atnf_breakup", "Pulsars spinning faster than a neutron star survives", len(rows), dict(p_ms=p_ms), "the break-up limit",
                   "a period below 0.7 ms", "%d; the fastest %s" % (len(hits), ", ".join("%s %s s" % (r["PSRJ"], r["P0"]) for r in fastest)),
                   hits, "period, s", lambda r: num(r["P0"]), lambda r: "")
        print("recorded")


overpower_DOC = r"""Pulsars radiating more than their spin supplies: ATNF 1.4 GHz luminosities (mJy kpc^2) taken isotropic over a 1 GHz band, 1.2e20 W per mJy kpc^2, against the spin-down power, flagged where the radio exceeds it; beaming makes the isotropic figure an upper bound.

    python -I seti.py radio_sky overpower [record] [--band-hz 1e9]"""


def overpower_main(record=False, band_hz=1e9):
    k = 4 * math.pi * 1e-29 * (3.0857e19) ** 2 * band_hz
    rows = [r for r in psr_all() if num(r["RLUM14"]) and num(r["Edot"])]
    for r in rows:
        r["ratio"] = num(r["RLUM14"]) * k / (num(r["Edot"]) * 1e-7)
    hits = sorted([r for r in rows if r["ratio"] > 1], key=lambda r: -r["ratio"])
    print("%d pulsars with radio luminosity and spin-down power; radio above spin-down (isotropic, 1 GHz): %d %s" % (
        len(rows), len(hits), [("%s %.2f, P %s s, distance %s kpc (DM %s)" % (r["PSRJ"], r["ratio"], r["P0"], r["Dist"], r["DistDM"])) for r in hits[:10]]))
    if record:
        psr_record("atnf_overpowered", "Pulsars radiating more than their spin supplies", len(rows), dict(band_hz=band_hz, isotropic=True),
                   "the spin-down power", "an isotropic 1.4 GHz luminosity above the spin-down power", "%d above" % len(hits), hits,
                   "radio over spin-down power", lambda r: r["ratio"], lambda r: "P %s s, distance %s kpc" % (r["P0"], r["Dist"]))
        print("recorded")


gcclock_DOC = r"""Pulsars of one globular cluster on one clock: ATNF pulsars of the same cluster whose rotation frequencies, carried to MJD 56000 with their derivatives and measured to 1e-7, agree to 1e-6, against the pairs within 1e-4 scaled to the tolerance.

    python -I seti.py radio_sky gcclock [record] [--tol 1e-6] [--precision 1e-7] [--epoch 56000]"""


def gcclock_main(record=False, tol=1e-6, precision=1e-7, epoch=56000.0):
    good = []
    for r in psr_all():
        a = r["Assoc"] or ""
        f0, ef, f1, ep = num(r["F0"]), num(r["e_F0"]), num(r["F1"]), num(r["PEpoch"])
        if "GC:" not in a or f0 is None or not ef or ef > precision * f0 or f1 is None or ep is None:
            continue
        good.append(dict(psr=r["PSRJ"], gc=a.split("GC:")[1].split(",")[0].split("[")[0], f=f0 + f1 * (epoch - ep) * 86400.0))
    wide = 100 * tol
    counts, pairs = {tol: 0, wide: 0}, []
    for i, a in enumerate(good):
        for b in good[i + 1:]:
            if a["gc"] != b["gc"]:
                continue
            d = abs(a["f"] - b["f"]) / a["f"]
            for t in counts:
                counts[t] += d < t
            if d < tol:
                pairs.append((a, b, d))
    expect = counts[wide] / 100.0
    print("%d cluster pulsars timed to %g in %d clusters; same-cluster pairs within %g: %d, within %g: %d against %.3f %s" % (
        len(good), precision, len({g["gc"] for g in good}), wide, counts[wide], tol, counts[tol], expect, [(a["psr"], b["psr"], d) for a, b, d in pairs]))
    if record:
        core.record_run(
            "atnf_cluster_clocks", "Pulsars of one globular cluster on one clock", "ATNF pulsar catalogue (VizieR B/psr)",
            "globular-cluster pulsars timed to 1e-7: %d" % len(good), os.path.abspath(__file__), "gcclock", len(good), "pulsars",
            dict(tol=tol, precision=precision, epoch=epoch), "same-cluster pairs within %g scaled: %.3f" % (wide, expect), "two pulsars of one cluster on one frequency to %g" % tol,
            "%d pairs against %.3f" % (counts[tol], expect), [dict(target="psr:%s" % a["psr"], stat=d, note="with %s" % b["psr"]) for a, b, d in pairs])
        print("recorded")


yearpsr_DOC = r"""Binary pulsars on Earth's year: ATNF binary pulsars whose orbital period lies within 0.1 per cent of 365.25 days.

    python -I seti.py radio_sky yearpsr [record] [--tol 0.001]"""


def yearpsr_main(record=False, tol=0.001):
    rows = [r for r in psr_all() if num(r["PB"])]
    hits = [r for r in rows if abs(num(r["PB"]) / 365.25 - 1) < tol]
    near = sorted(rows, key=lambda r: abs(num(r["PB"]) / 365.25 - 1))[:3]
    print("%d binary pulsars; within %g per cent of a year: %d; the nearest %s" % (len(rows), 100 * tol, len(hits), [(r["PSRJ"], r["PB"]) for r in near]))
    if record:
        psr_record("atnf_year_orbits", "Binary pulsars on Earth's year", len(rows), dict(tol=tol), "the density of orbital periods near a year",
                   "a binary pulsar whose orbit lasts one Earth year to 0.1 per cent", "%d; the nearest %s" % (
                       len(hits), ", ".join("%s %s d" % (r["PSRJ"], r["PB"]) for r in near)), hits, "orbital period, d", lambda r: num(r["PB"]),
                   lambda r: "")
        print("recorded")


def frb_rows():
    """CHIME/FRB Catalog 1, one row per burst (its first sub-burst), with the timing, width, fluence, scattering and spectral columns."""
    return catalogues.frb_catalog1("Name, RpName, RAJ2000, DEJ2000, DMfitb, e_DMfitb, MJDinf, Widthfitb, e_Widthfitb, Fluence, e_Fluence, Scat, e_Scat, SpInd, "
                                   "e_SpInd, Nsb")


frbdm_DOC = r"""Burst dispersions on shared values: CHIME/FRB Catalog 1 sources, one burst for each repeater, pairs whose fitted dispersion measures agree to 0.5 pc cm^-3, against 1000 draws from the same distribution smoothed by 20 pc cm^-3.

    python -I seti.py radio_sky frbdm [record] [--tol 0.5] [--smooth 20] [--draws 1000]"""


def frbdm_main(record=False, tol=0.5, smooth=20.0, draws=1000):
    seen, rows = set(), []
    for r in frb_rows():
        src = (r["RpName"] or "").strip()
        src = r["Name"] if not src.strip("-9") else src
        if num(r["DMfitb"]) and src not in seen:
            seen.add(src)
            rows.append(r)
    dm = np.sort(np.array([num(r["DMfitb"]) for r in rows]))

    def npairs(x):
        x = np.sort(x)
        return int(sum(np.searchsorted(x, x[i] + tol) - i - 1 for i in range(len(x))))
    obs = npairs(dm)
    rng = np.random.default_rng(3)
    null = np.array([npairs(rng.choice(dm, len(dm)) + rng.normal(0, smooth, len(dm))) for _ in range(draws)])
    p = float(np.mean(null >= obs))
    print("%d bursts; pairs within 0.5 pc cm^-3: %d against %.1f +- %.1f (as many or more in %.3f)" % (len(rows), obs, null.mean(), null.std(), p))
    if record:
        core.record_run(
            "chime_dm_shells", "Burst dispersions on shared values", "CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59)", "bursts: %d" % len(rows),
            os.path.abspath(__file__), "frbdm", len(rows), "bursts", dict(tol=tol, smooth=smooth, draws=draws), "%d smoothed draws: %.1f +- %.1f" % (draws, null.mean(), null.std()),
            "dispersion measures clustering on shared values", "%d pairs against %.1f +- %.1f (p %.3f)" % (obs, null.mean(), null.std(), p), [])
        print("recorded")


frbsimul_DOC = r"""Bursts arriving together from different directions: CHIME/FRB Catalog 1 bursts more than 10 deg apart whose arrival times at infinite frequency agree to 1 second.

    python -I seti.py radio_sky frbsimul [record] [--dt-s 1] [--sep-deg 10]"""


def frbsimul_main(record=False, dt_s=1.0, sep_deg=10.0):
    rows = sorted([r for r in frb_rows() if num(r["MJDinf"])], key=lambda r: num(r["MJDinf"]))
    hits = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            dt = (num(b["MJDinf"]) - num(a["MJDinf"])) * 86400
            if dt > dt_s:
                break
            if sep_arcsec(num(a["RAJ2000"]), num(a["DEJ2000"]), num(b["RAJ2000"]), num(b["DEJ2000"])) > 3600 * sep_deg:
                hits.append((a["Name"], b["Name"], dt))
    span = (num(rows[-1]["MJDinf"]) - num(rows[0]["MJDinf"])) * 86400
    expect = len(rows) ** 2 / 2 * 2 * dt_s / span
    print("%d bursts over %.0f days; pairs more than 10 deg apart within 1 s: %d against %.4f by chance %s" % (len(rows), span / 86400, len(hits), expect, hits))
    if record:
        core.record_run(
            "chime_simultaneous", "Bursts arriving together from different directions", "CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59)",
            "bursts: %d" % len(rows), os.path.abspath(__file__), "frbsimul", len(rows), "bursts", dict(dt_s=dt_s, sep_deg=sep_deg),
            "uniform arrival times over the catalogue's span: %.4f pairs" % expect, "two bursts from different directions within 1 s",
            "%d against %.4f" % (len(hits), expect), [dict(target="frb:%s" % a, stat=dt, unit="s apart", note="with %s" % b) for a, b, dt in hits])
        print("recorded")


frbcopies_DOC = r"""Copy bursts from different directions: non-repeating CHIME/FRB Catalog 1 bursts more than 10 deg apart whose width, fluence, scattering time and spectral index all agree within their combined errors and to 10 per cent, against the same count with each property shuffled among the bursts.

    python -I seti.py radio_sky frbcopies [record] [--tol-frac 0.1] [--sep-deg 10] [--shuffles 20]"""


def frbcopies_main(record=False, tol_frac=0.1, sep_deg=10.0, shuffles=20):
    keys = (("Widthfitb", "e_Widthfitb"), ("Fluence", "e_Fluence"), ("Scat", "e_Scat"), ("SpInd", "e_SpInd"))
    rows = [r for r in frb_rows() if not (r["RpName"] or "").strip().strip("-9").strip() and all(num(r[k]) is not None and num(r[e]) for k, e in keys)]
    X = np.array([[num(r[k]) for k, e in keys] for r in rows])
    E = np.array([[num(r[e]) for k, e in keys] for r in rows])
    pos = [(num(r["RAJ2000"]), num(r["DEJ2000"])) for r in rows]

    def count(X):
        n = []
        for i in range(len(rows)):
            d = np.abs(X - X[i])
            ok = np.all((d <= np.hypot(E, E[i])) & (d <= tol_frac * np.abs(X[i]) + 1e-12), axis=1)
            for j in np.nonzero(ok)[0]:
                if j > i and sep_arcsec(*pos[i], *pos[j]) > 3600 * sep_deg:
                    n.append((i, int(j)))
        return n
    found = count(X)
    rng = np.random.default_rng(21)
    null = []
    for _ in range(shuffles):
        Y = X.copy()
        for c in range(Y.shape[1]):
            Y[:, c] = Y[rng.permutation(len(Y)), c]
        null.append(len(count(Y)))
    print("%d non-repeating bursts with all four measured; copies %d against %.1f shuffled %s" % (
        len(rows), len(found), float(np.mean(null)), [(rows[i]["Name"], rows[j]["Name"]) for i, j in found][:10]))
    if record:
        core.record_run(
            "chime_copy_bursts", "Copy bursts from different directions", "CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59)",
            "non-repeating bursts with width, fluence, scattering and spectral index: %d" % len(rows), os.path.abspath(__file__), "frbcopies",
            len(rows), "bursts", dict(tol_frac=tol_frac, sep_deg=sep_deg, shuffles=shuffles), "properties shuffled among the bursts: mean %.1f" % float(np.mean(null)),
            "two bursts from different directions alike in width, fluence, scattering and spectral index", "%d pairs against %.1f" % (
                len(found), float(np.mean(null))),
            [dict(target="frb:%s" % rows[i]["Name"], note="copy of %s" % rows[j]["Name"],
                  vet="chance-rate" if len(found) <= np.mean(null) + 3 * math.sqrt(max(np.mean(null), 1)) else None) for i, j in found])
        print("recorded")


psrratio_DOC = r"""Pulsars paired in simple ratios: ATNF pulsars within 1 deg of each other outside globular clusters whose rotation frequencies, measured to 1e-7 and carried to MJD 56000, stand in a ratio p/q (q up to 5, p up to 12) to 1e-6, a pair of beacons on one clock, against the hits among pairs 10-20 deg apart scaled by the number of pairs.

    python -I seti.py radio_sky psrratio [record] [--tol 1e-6] [--sep-deg 1] [--control-deg 10 20] [--q-max 5] [--p-max 12]"""


def psrratio_main(record=False, tol=1e-6, sep_deg=1.0, control_deg=[10.0, 20.0], q_max=5, p_max=12, precision=1e-7, epoch=56000.0):
    good = []
    for r in catalogues.pulsars("PSRJ, RA2000, DE2000, F0, e_F0, F1, PEpoch, Assoc"):
        f0, ef, f1, ep = num(r["F0"]), num(r["e_F0"]), num(r["F1"]), num(r["PEpoch"])
        if f0 and ef and ef < precision * f0 and f1 is not None and ep and "GC:" not in (r["Assoc"] or "") and num(r["RA2000"]) is not None:
            good.append((r["PSRJ"], num(r["RA2000"]), num(r["DE2000"]), f0 + f1 * (epoch - ep) * 86400.0))
    ratios = sorted({p / q for q in range(1, q_max + 1) for p in range(1, p_max + 1) if p != q})

    def hits(lo, hi):
        n, out = 0, []
        for i in range(len(good)):
            for j in range(i + 1, len(good)):
                s = core.sep_arcsec(good[i][1], good[i][2], good[j][1], good[j][2]) / 3600
                if lo <= s < hi:
                    n += 1
                    x = good[j][3] / good[i][3]
                    if any(abs(x / k - 1) < tol for k in ratios):
                        out.append((good[i][0], good[j][0], x))
        return n, out
    n_near, near = hits(0, sep_deg)
    n_far, far = hits(control_deg[0], control_deg[1])
    expect = len(far) * n_near / max(n_far, 1)
    print("%d pulsars timed to 1e-7 outside clusters; %d pairs within 1 deg, %d in simple ratios to 1e-6 %s; at 10-20 deg %d of %d, so %.3f expected" % (
        len(good), n_near, len(near), near, len(far), n_far, expect))
    if record:
        core.record_run("atnf_simple_ratios", "Pulsars paired in simple ratios", "ATNF pulsar catalogue (VizieR B/psr)",
                        "pulsars timed to 1e-7 outside globular clusters: %d" % len(good), os.path.abspath(__file__), "psrratio", n_near, "pulsar pairs",
                        dict(tol=tol, sep_deg=sep_deg, control_deg=control_deg, q_max=q_max, p_max=p_max), "pairs %g-%g deg apart scaled: %.3f" % (control_deg[0], control_deg[1], expect),
                        "two neighbouring pulsars on frequencies in a simple ratio",
                        "%d against %.3f" % (len(near), expect), [dict(target="psr:%s" % a, stat=x, unit="frequency ratio", note="with %s" % b) for a, b, x in near])
        print("recorded")


frbpsr_DOC = r"""Bursts masquerading from pulsars: CHIME/FRB Catalog 1 bursts within 1 deg of an ATNF pulsar whose dispersion measure agrees with the pulsar's to 5 per cent, a burst from a known beacon's place and distance, against the same match with the bursts moved 5 deg in right ascension either way.

    python -I seti.py radio_sky frbpsr [record] [--sep-deg 1] [--dm-tol 0.05] [--shifts-deg 5 10]"""


def frbpsr_main(record=False, sep_deg=1.0, dm_tol=0.05, shifts_deg=[5.0, 10.0]):
    frbs = [r for r in frb_rows() if num(r["DMfitb"])]
    psrs = [r for r in catalogues.pulsars("PSRJ, RA2000, DE2000, DM") if num(r["DM"]) and num(r["RA2000"]) is not None]

    def match(shift):
        out = []
        for b in frbs:
            for p in psrs:
                if abs(num(b["DMfitb"]) / num(p["DM"]) - 1) < dm_tol and core.sep_arcsec((num(b["RAJ2000"]) + shift) % 360, num(b["DEJ2000"]),
                                                                                         num(p["RA2000"]), num(p["DE2000"])) < 3600 * sep_deg:
                    out.append((b["Name"], p["PSRJ"]))
        return out
    found = match(0.0)
    null = [len(match(s)) for k in shifts_deg for s in (k, -k)]
    print("%d bursts, %d pulsars; within 1 deg and 5 per cent in DM: %d %s; moved %s" % (len(frbs), len(psrs), len(found), found, null))
    if record:
        core.record_run("chime_frb_pulsars", "Bursts masquerading from pulsars", "CHIME/FRB Catalog 1; ATNF pulsar catalogue", "bursts: %d" % len(frbs),
                        os.path.abspath(__file__), "frbpsr", len(frbs), "bursts", dict(sep_deg=sep_deg, dm_tol=dm_tol, shifts_deg=shifts_deg), "bursts moved %s deg: %s" % (shifts_deg, null),
                        "a burst at a pulsar's position and dispersion measure", "%d against %.2f" % (len(found), float(np.mean(null))),
                        [dict(target="frb:%s" % b, note="at %s" % p) for b, p in found])
        print("recorded")


copypsr_DOC = r"""Copy pulsars: ATNF pulsars with period, period derivative, dispersion measure and W50 measured; pairs more than 10 deg apart alike to 1e-3 in period and 5 per cent in period derivative, dispersion measure and W50, against the pairs found with the last three shuffled among pulsars of like period twenty times.

    python -I seti.py radio_sky copypsr [record] [--period-tol 1e-3] [--other-tol 0.05] [--sep-deg 10] [--shuffles 20]"""


def copypsr_main(record=False, period_tol=1e-3, other_tol=0.05, sep_deg=10.0, shuffles=20):
    rows = []
    for r in catalogues.pulsars("PSRJ, RA2000, DE2000, P0, P1, DM, W50"):
        v = [num(r[k]) for k in ("RA2000", "DE2000", "P0", "P1", "DM", "W50")]
        if None not in v and v[2] > 0 and v[3] > 0 and v[4] > 0 and v[5] > 0:
            rows.append((r["PSRJ"], *v))
    rows.sort(key=lambda t: t[3])
    P = np.array([t[3] for t in rows])
    X = np.array([[t[4], t[5], t[6]] for t in rows])
    ra, dec = np.array([t[1] for t in rows]), np.array([t[2] for t in rows])

    def pairs(Y):
        out = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if P[j] / P[i] - 1 > period_tol:
                    break
                if np.all(np.abs(Y[j] / Y[i] - 1) < other_tol) and core.sep_arcsec(ra[i], dec[i], ra[j], dec[j]) > 3600 * sep_deg:
                    out.append((i, j))
        return out
    found = pairs(X)
    rng = np.random.default_rng(23)
    bins = np.digitize(np.log10(P), np.linspace(np.log10(P.min()), np.log10(P.max()), 11))
    null = []
    for _ in range(shuffles):
        Y = X.copy()
        for b in np.unique(bins):
            idx = np.nonzero(bins == b)[0]
            for k in range(3):
                Y[idx, k] = Y[rng.permutation(idx), k]
        null.append(len(pairs(Y)))
    print("%d pulsars with period, derivative, DM and W50; copies more than 10 deg apart: %d %s; shuffled %s" % (
        len(rows), len(found), [(rows[i][0], rows[j][0]) for i, j in found], null))
    if record:
        core.record_run("atnf_copy_pulsars", "Copy pulsars", "ATNF pulsar catalogue (VizieR B/psr)",
                        "pulsars with period, period derivative, dispersion measure and W50: %d" % len(rows), os.path.abspath(__file__), "copypsr", len(rows),
                        "pulsars", dict(period=period_tol, others=other_tol, sep_deg=sep_deg, shuffles=shuffles), "period derivative, DM and W50 shuffled among pulsars of like period: mean %.2f"
                        % float(np.mean(null)), "two pulsars far apart alike in period, spin-down, dispersion and pulse width",
                        "%d pairs against %.2f shuffled" % (len(found), float(np.mean(null))),
                        [dict(target="psr:%s" % rows[i][0], note="copy of %s" % rows[j][0],
                              vet="chance-rate" if len(found) <= np.mean(null) + 3 * math.sqrt(max(np.mean(null), 1)) else None) for i, j in found])
        print("recorded")


daypsr_DOC = r"""Binary pulsars on Earth's day: ATNF orbital periods within 0.1 per cent of the solar day (1 d) or the sidereal day (0.99727 d), against the binaries within 10 per cent of a day scaled to the two windows.

    python -I seti.py radio_sky daypsr [record] [--tol 1e-3]"""


def daypsr_main(record=False, tol=1e-3):
    rows = [r for r in psr_all() if num(r["PB"])]
    days = {"solar day": 1.0, "sidereal day": 0.9972696}
    hits = [(r["PSRJ"], num(r["PB"]), k) for r in rows for k, d in days.items() if abs(num(r["PB"]) / d - 1) < tol]
    wide = sum(1 for r in rows if abs(num(r["PB"]) - 1) < 0.1)
    expect = wide * (2 * 2 * tol) / 0.2
    print("%d binary pulsars; within %g per cent of the solar or sidereal day: %d %s against %.2f from the %d within 10 per cent of a day" % (
        len(rows), 100 * tol, len(hits), hits, expect, wide))
    if record:
        core.record_run("atnf_day_orbits", "Binary pulsars on Earth's day", "ATNF pulsar catalogue (VizieR B/psr)", "binary pulsars: %d" % len(rows),
                        os.path.abspath(__file__), "daypsr", len(rows), "pulsars", dict(tol=tol, days=list(days.values())),
                        "binaries within 10 per cent of a day scaled: %.2f" % expect, "a binary pulsar orbiting once per Earth day",
                        "%d against %.2f" % (len(hits), expect),
                        [dict(target="psr:%s" % n, stat=pb, unit="orbital period, d", note=k,
                              vet="chance-rate" if len(hits) <= expect + 3 * math.sqrt(max(expect, 1)) else None) for n, pb, k in hits])
        print("recorded")


bfield_DOC = r"""Pulsars beyond the magnetar limit: ATNF pulsars with a positive period derivative, their dipole field 3.2e19 (P dP/dt)^1/2 G, flagged above 1e16 G, past the strongest magnetar (about 2e15 G); each read for its association, where a globular cluster's acceleration or a binary distorts the derivative.

    python -I seti.py radio_sky bfield [record] [--b-gauss 1e16]"""


def bfield_main(record=False, b_gauss=1e16):
    rows = []
    for r in catalogues.pulsars("PSRJ, P0, P1, Assoc, Binary, Type"):
        p, pd = num(r["P0"]), num(r["P1"])
        if p and pd and pd > 0:
            rows.append((r["PSRJ"], p, pd, 3.2e19 * math.sqrt(p * pd), r["Assoc"], r["Type"]))
    rows.sort(key=lambda t: -t[3])
    hits = [t for t in rows if t[3] > b_gauss]
    print("%d pulsars with a positive period derivative; field above %g G: %d; the strongest %s" % (
        len(rows), b_gauss, len(hits), ["%s %.2e G (P %.3f s, %s, %s)" % (t[0], t[3], t[1], t[4], t[5]) for t in rows[:5]]))
    if record:
        core.record_run("atnf_beyond_magnetars", "Pulsars beyond the magnetar limit", "ATNF pulsar catalogue (VizieR B/psr)",
                        "pulsars with a positive period derivative: %d" % len(rows), os.path.abspath(__file__), "bfield", len(rows), "pulsars",
                        dict(b_gauss=b_gauss), "the strongest magnetar's field, about 2e15 G", "a neutron star with a dipole field no magnetar reaches",
                        "%d above 1e16 G; the strongest %s at %.2e G" % (len(hits), rows[0][0], rows[0][3]),
                        [dict(target="psr:%s" % t[0], stat=t[3], unit="dipole field, G", note="P %.4f s, association %s, type %s" % (t[1], t[4], t[5])) for t in hits])
        print("recorded")


psrconst_DOC = r"""Pulsar pairs whose frequencies spell constants: ATNF pulsars timed to 1e-7 outside globular clusters, frequencies carried to MJD 56000 with their derivatives; pairs anywhere on the sky whose frequency ratio equals pi, e, the golden ratio, sqrt 2 or sqrt 3 to 1e-6, against the pairs within 1e-4 scaled to the tolerance.

    python -I seti.py radio_sky psrconst [record] [--tol 1e-6] [--precision 1e-7] [--epoch 56000]"""


def psrconst_main(record=False, tol=1e-6, precision=1e-7, epoch=56000.0):
    good = []
    for r in catalogues.pulsars("PSRJ, F0, e_F0, F1, PEpoch, Assoc"):
        f0, ef, f1, ep = num(r["F0"]), num(r["e_F0"]), num(r["F1"]), num(r["PEpoch"])
        if f0 and ef and ef < precision * f0 and f1 is not None and ep and "GC:" not in (r["Assoc"] or ""):
            good.append((r["PSRJ"], f0 + f1 * (epoch - ep) * 86400.0))
    good.sort(key=lambda t: t[1])
    f = np.array([t[1] for t in good])
    consts = {"pi": math.pi, "e": math.e, "golden ratio": (1 + math.sqrt(5)) / 2, "sqrt 2": math.sqrt(2), "sqrt 3": math.sqrt(3)}
    wide, hits = 0, []
    for name, c in consts.items():
        lo4, hi4 = np.searchsorted(f, f * c * (1 - 100 * tol)), np.searchsorted(f, f * c * (1 + 100 * tol))
        wide += int((hi4 - lo4).sum())
        lo6, hi6 = np.searchsorted(f, f * c * (1 - tol)), np.searchsorted(f, f * c * (1 + tol))
        for i in np.nonzero(hi6 > lo6)[0]:
            for j in range(lo6[i], hi6[i]):
                hits.append((good[j][0], good[i][0], name, f[j] / f[i] / c - 1))
    expect = wide / 100.0
    print("%d pulsars timed to 1e-7; frequency ratios within 1e-6 of a constant: %d %s against %.2f from %d within 1e-4" % (len(good), len(hits), hits, expect, wide))
    if record:
        core.record_run("atnf_constant_ratios", "Pulsar pairs whose frequencies spell constants", "ATNF pulsar catalogue (VizieR B/psr)",
                        "pulsars timed to 1e-7 outside globular clusters: %d" % len(good), os.path.abspath(__file__), "psrconst", len(good), "pulsars",
                        dict(tol=tol, precision=precision, epoch=epoch, constants=list(consts)), "pairs within %g scaled: %.2f" % (100 * tol, expect),
                        "two pulsars whose frequencies stand in a mathematical constant",
                        "%d against %.2f" % (len(hits), expect),
                        [dict(target="psr:%s" % a, stat=d, unit="fractional offset", note="over %s, %s" % (b, n),
                              vet="chance-rate" if len(hits) <= expect + 3 * math.sqrt(max(expect, 1)) else None) for a, b, n, d in hits])
        print("recorded")


frblocal_DOC = r"""Bursts from inside the Galaxy: CHIME/FRB Catalog 1 bursts whose dispersion measure falls below the Milky Way's along their line of sight in both the NE2001 and YMW16 models (the catalogue's excess columns below zero), a source within the Galaxy; one burst per source, the smallest excesses listed.

    python -I seti.py radio_sky frblocal [record]"""


def frblocal_main(record=False):
    rows, seen = [], set()
    for r in catalogues.frb_catalog1("Name, RpName, GLON, GLAT, DMfitb, e_DMfitb, DMeNE2001, DMeYMW16", one_per_burst=False):
        rp = (r["RpName"] or "").strip()
        src = r["Name"] if not rp or rp.startswith("-") else rp
        if src in seen or num(r["DMeNE2001"]) is None or num(r["DMeYMW16"]) is None:
            continue
        seen.add(src)
        rows.append(r)
    inside = [r for r in rows if num(r["DMeNE2001"]) < 0 and num(r["DMeYMW16"]) < 0]
    low = sorted(rows, key=lambda r: max(num(r["DMeNE2001"]), num(r["DMeYMW16"])))[:5]
    print("%d CHIME/FRB sources; dispersion below the Galaxy's in both models: %d; the smallest excesses %s" % (len(rows), len(inside), [
        "%s DM %s, excess NE2001 %s, YMW16 %s, b %s" % (r["Name"], r["DMfitb"], r["DMeNE2001"], r["DMeYMW16"], r["GLAT"]) for r in low]))
    if record:
        core.record_run("chime_inside_galaxy", "Bursts from inside the Galaxy", "CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59)",
                        "sources, one burst for each repeater: %d" % len(rows), os.path.abspath(__file__), "frblocal", len(rows), "burst sources",
                        dict(models=["NE2001", "YMW16"]), "the Galaxy's dispersion along each line of sight", "a fast radio burst from within the Milky Way",
                        "%d; the smallest excesses %s" % (len(inside), "; ".join("%s (%s, %s pc cm^-3)" % (r["Name"], r["DMeNE2001"], r["DMeYMW16"]) for r in low[:3])),
                        [dict(target="frb:%s" % r["Name"], stat=num(r["DMeYMW16"]), unit="DM excess over YMW16, pc cm^-3") for r in inside])
        print("recorded")


frbetz_DOC = r"""Bursts from Earth's transit zone: CHIME/FRB Catalog 1 sources, one burst per source, within 0.264 deg of the ecliptic, from where Earth is seen to transit the Sun, against the expectation from 2000 random right ascensions at each source's own declination (a transit telescope covers every right ascension of a declination alike).

    python -I seti.py radio_sky frbetz [record] [--zone-deg 0.264] [--draws 2000]"""


def frbetz_main(record=False, zone_deg=0.264, draws=2000):
    seen, ra, de, names = set(), [], [], []
    for r in catalogues.frb_catalog1("Name, RpName, RAJ2000, DEJ2000", one_per_burst=False):
        rp = (r["RpName"] or "").strip()
        src = r["Name"] if not rp or rp.startswith("-") else rp
        if src in seen or num(r["RAJ2000"]) is None:
            continue
        seen.add(src)
        ra.append(num(r["RAJ2000"]))
        de.append(num(r["DEJ2000"]))
        names.append(src)
    ra, de = np.radians(ra), np.radians(de)
    eps = math.radians(23.4392911)

    def beta(a, d):
        return np.degrees(np.arcsin(np.sin(d) * math.cos(eps) - np.cos(d) * math.sin(eps) * np.sin(a)))
    b = beta(ra, de)
    hits = [names[i] for i in np.nonzero(np.abs(b) < zone_deg)[0]]
    rng = np.random.default_rng(29)
    p = np.mean(np.abs(beta(rng.uniform(0, 2 * math.pi, (draws, len(de))), de[None, :])) < zone_deg, axis=0)
    expect = float(p.sum())
    print("%d CHIME/FRB sources; within 0.264 deg of the ecliptic: %d %s against %.2f from random right ascensions" % (len(names), len(hits), hits, expect))
    if record:
        core.record_run("chime_transit_zone", "Bursts from Earth's transit zone", "CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59)",
                        "sources, one burst for each repeater: %d" % len(names), os.path.abspath(__file__), "frbetz", len(names), "burst sources",
                        dict(zone_deg=zone_deg, draws=draws), "%d random right ascensions at each declination: %.2f" % (draws, expect),
                        "fast radio bursts gathered where Earth is seen to transit the Sun", "%d against %.2f" % (len(hits), expect),
                        [dict(target="frb:%s" % h, vet="chance-rate" if len(hits) <= expect + 3 * math.sqrt(max(expect, 1)) else None) for h in hits])
        print("recorded")


frbladder_DOC = r"""Burst dispersions on a ladder: CHIME/FRB Catalog 1 dispersion measures, one burst per source, tested for bunching at whole multiples of a spacing between 5 and 200 pc cm^-3 by the Rayleigh power |sum exp(2 pi i DM / s)|^2 / N; the largest power over spacings against the largest from 500 draws of the same number from the distribution smoothed by 20 pc cm^-3.

    python -I seti.py radio_sky frbladder [record] [--spacing-pc-cm3 5 200] [--smoothing 20] [--draws 500]"""


def frbladder_main(record=False, spacing_pc_cm3=[5.0, 200.0], smoothing=20.0, draws=500):
    seen, dm = set(), []
    for r in catalogues.frb_catalog1("Name, RpName, DMfitb", one_per_burst=False):
        rp = (r["RpName"] or "").strip()
        src = r["Name"] if not rp or rp.startswith("-") else rp
        if src in seen or num(r["DMfitb"]) is None:
            continue
        seen.add(src)
        dm.append(num(r["DMfitb"]))
    dm = np.array(dm)
    s = np.arange(spacing_pc_cm3[0], spacing_pc_cm3[1], 0.05)

    def zmax(x):
        ph = np.exp(2j * np.pi * x[None, :] / s[:, None])
        z = np.abs(ph.sum(axis=1)) ** 2 / len(x)
        k = int(np.argmax(z))
        return float(z[k]), float(s[k])
    obs, at = zmax(dm)
    rng = np.random.default_rng(31)
    null = np.array([zmax(np.abs(rng.choice(dm, len(dm)) + rng.normal(0, smoothing, len(dm))))[0] for _ in range(draws)])
    p = float(np.mean(null >= obs))
    print("%d CHIME/FRB sources; the strongest ladder at %.2f pc cm^-3 with Rayleigh power %.1f; smoothed draws reach %.1f at the median, as high or higher in %.3f" % (
        len(dm), at, obs, float(np.median(null)), p))
    if record:
        core.record_run("chime_dm_ladder", "Burst dispersions on a ladder", "CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59)",
                        "sources, one burst for each repeater: %d" % len(dm), os.path.abspath(__file__), "frbladder", len(dm), "burst sources",
                        dict(spacing_pc_cm3=spacing_pc_cm3, smoothing=smoothing, draws=draws), "the largest power from smoothed draws: median %.1f" % float(np.median(null)),
                        "dispersion measures stepping in whole multiples of one spacing", "largest power %.1f at %.2f pc cm^-3, as high or higher in %.3f of draws" % (
                            obs, at, p), [])
        print("recorded")


coldwdradio_DOC = r"""Radio from cold white dwarfs: Gentile Fusillo et al. (2021) white dwarfs within 100 pc with a white-dwarf probability above 0.75 and hydrogen-model temperatures below 8000 K, carried from 2016.0 to NVSS's 1995.0 by their proper motions, against NVSS (VIII/65) by CDS XMatch within 15 arcsec, against the same match with the white dwarfs moved 2 arcmin in eight directions; each match read for another Gaia star near the white dwarf.

    python -I seti.py radio_sky coldwdradio [record] [--teff-max 8000] [--radius-arcsec 15] [--epoch 1995]"""


def coldwdradio_main(record=False, teff_max=8000.0, radius_arcsec=15.0, epoch=1995.0):
    from setitools.catalogues import shifted_matches
    wds = catalogues.white_dwarfs(teff_max=teff_max)
    pos = []
    for w in wds:
        ra, de = core.star_at(float(w["RA_ICRS"]), float(w["DE_ICRS"]), num(w["pmRA"], 0.0), num(w["pmDE"], 0.0), 2016.0, epoch)
        pos.append(dict(name=w["WDJname"].strip(), ra=ra, dec=de))
    hits, null = shifted_matches(pos, "VIII/65/nvss", radius_arcsec)
    names = sorted({m["source_id"] for m in hits})
    mean = float(np.mean(null))
    print("%d white dwarfs within 100 pc below 8000 K; NVSS sources within 15 arcsec: %d %s; moved 2 arcmin %s (mean %.1f)" % (
        len(pos), len(names), [(m["source_id"], m.get("S1.4"), round(float(m["angDist"]), 1)) for m in hits], null, mean))
    if record:
        core.record_run("nvss_cold_white_dwarfs", "Radio from cold white dwarfs", "Gentile Fusillo et al. (2021) white dwarfs; NVSS (VIII/65) by CDS XMatch",
                        "white dwarfs within 100 pc below 8000 K: %d" % len(pos), os.path.abspath(__file__), "coldwdradio", len(pos), "white dwarfs",
                        dict(radius_arcsec=radius_arcsec, epoch=epoch, teff_k=teff_max), "the white dwarfs moved 2 arcmin: %s" % null,
                        "a radio source on a white dwarf too cool for any", "%d against a mean of %.1f moved" % (len(names), mean),
                        [dict(target="wd:%s" % n, vet="chance-rate" if len(names) <= mean + 3 * math.sqrt(max(mean, 1)) else None) for n in names])
        print("recorded")


msplight_DOC = r"""Infrared at isolated millisecond pulsars: ATNF pulsars with periods below 30 ms, no binary orbit, no globular-cluster association and a timing position good to 1 arcsec, against AllWISE (II/328) by CDS XMatch within 3 arcsec, against the same match with the pulsars moved 2 arcmin in eight directions; an old neutron star with no companion has nothing to warm, so a shell around one shows in the mid-infrared. A match is a companion when a Gaia DR3 star at the infrared position shares the pulsar's proper motion, a background galaxy when the Legacy Survey DR10 resolves a galaxy there, and a neighbour's light when it lies more than three combined errors off the pulsar.

    python -I seti.py radio_sky msplight [record] [--period-s 0.03] [--position-arcsec 1] [--radius-arcsec 3]"""


def msplight_vet(p, m):
    """Explanation of the AllWISE source M matched to pulsar P: a Gaia star sharing its proper motion, a galaxy the Legacy Survey resolves, or a position off the pulsar's."""
    ra, dec = float(m["RAJ2000"]), float(m["DEJ2000"])
    if p["pmra"] is not None:
        for g in core.xmatch([dict(source_id="w", ra=ra, dec=dec)], "I/355/gaiadr3", 1.5):
            if num(g["pmRA"]) is not None and math.hypot(num(g["pmRA"]) - p["pmra"], num(g["pmDE"]) - p["pmdec"]) < (
                    1 + 3 * math.hypot(num(g["e_pmRA"]) or 1, num(g["e_pmDE"]) or 1)):
                return "companion"
    box = 1.0 / 3600
    for r in core.datalab("SELECT type FROM ls_dr10.tractor WHERE ra BETWEEN %.6f AND %.6f AND dec BETWEEN %.6f AND %.6f" % (
            ra - box / math.cos(math.radians(dec)), ra + box / math.cos(math.radians(dec)), dec - box, dec + box)):
        if r["type"] not in ("PSF", "DUP"):
            return "background-galaxy"
    if float(m["angDist"]) > 3 * math.hypot(p["err"], num(m.get("eeMaj")) or 0.5):
        return "blend"
    return None


def msplight_main(record=False, period_s=0.03, position_arcsec=1.0, radius_arcsec=3.0):
    from setitools.catalogues import shifted_matches
    pos = []
    for r in catalogues.pulsars("PSRJ, RAJ2000, DEJ2000, e_RAJ2000, e_DEJ2000, pmRA, pmDE, P0, Binary, PB, Assoc"):
        p, era, ede = num(r["P0"]), num(r["e_RAJ2000"]), num(r["e_DEJ2000"])
        if not (p and p < period_s and catalogues.pulsar_isolated(r) and "GC:" not in (r["Assoc"] or "") and era is not None and ede is not None):
            continue
        ra, dec = num(r["RAJ2000"]), num(r["DEJ2000"])
        err = math.hypot(era * 15 * math.cos(math.radians(dec)), ede)
        if err < position_arcsec:
            pos.append(dict(name=r["PSRJ"].strip(), ra=ra, dec=dec, err=err, pmra=num(r["pmRA"]), pmdec=num(r["pmDE"])))
    hits, null = shifted_matches(pos, "II/328/allwise", radius_arcsec)
    by = {p["name"]: p for p in pos}
    vets = {}
    for m in hits:
        vets.setdefault(m["source_id"], msplight_vet(by[m["source_id"]], m))
    mean = float(np.mean(null))
    left = sorted(n for n, v in vets.items() if v is None)
    print("%d isolated millisecond pulsars outside clusters with timing positions; AllWISE sources within 3 arcsec: %d %s; moved 2 arcmin %s (mean %.1f); unexplained %s" % (
        len(pos), len(vets), [(m["source_id"], m.get("W1mag"), m.get("W2mag"), round(float(m["angDist"]), 1), vets[m["source_id"]]) for m in hits],
        null, mean, left))
    if record:
        core.record_run("allwise_isolated_msps", "Infrared at isolated millisecond pulsars", "ATNF pulsar catalogue (VizieR B/psr); AllWISE (II/328) by CDS XMatch; Gaia DR3; Legacy Survey DR10 (Data Lab)",
                        "isolated millisecond pulsars outside globular clusters with positions good to 1 arcsec: %d" % len(pos), os.path.abspath(__file__), "msplight",
                        len(pos), "pulsars", dict(period_s=period_s, position_arcsec=position_arcsec, radius_arcsec=radius_arcsec), "the pulsars moved 2 arcmin: %s" % null,
                        "infrared from a lone old neutron star", "%d against a mean of %.1f moved; %d unexplained" % (len(vets), mean, len(left)),
                        [dict(target="psr:%s" % n, vet=v) for n, v in sorted(vets.items())])
        print("recorded")


etzradio_DOC = r"""Radio from Earth's transit zone: Gaia Catalogue of Nearby Stars objects (probability above 0.9) north of declination -40 deg, carried from 2016.0 to 2019.0, against VLASS epoch-1 components (J/ApJS/255/30) within 3 arcsec by CDS XMatch; the share with a 3 GHz source among stars within 0.264 deg of the ecliptic, which see Earth cross the Sun, against the share 1-5 deg away, and the zone's matches against the zone moved 2 arcmin in eight directions. A match is an active star or binary by its SIMBAD type, or a background source where the Legacy Survey DR10 resolves a galaxy at the radio position.

    python -I seti.py radio_sky etzradio [record] [--zone-deg 0.264] [--control-deg 1 5] [--radius-arcsec 3] [--epoch 2019] [--dec-min -40]"""


def etzradio_main(record=False, zone_deg=0.264, control_deg=[1.0, 5.0], radius_arcsec=3.0, epoch=2019.0, dec_min=-40.0):
    from setitools.catalogues import shifted_matches
    from scipy.stats import poisson
    eps = math.radians(23.4392911)
    rows = catalogues.gcns("source_id, ra, dec, pmra, pmdec, parallax", ranges=12)
    zone, near = [], []
    for r in rows:
        ra, de = float(r["ra"]), float(r["dec"])
        if de < dec_min:
            continue
        a, d = math.radians(ra), math.radians(de)
        beta = abs(math.degrees(math.asin(math.sin(d) * math.cos(eps) - math.cos(d) * math.sin(eps) * math.sin(a))))
        if beta < zone_deg or control_deg[0] <= beta < control_deg[1]:
            ra19, de19 = core.star_at(ra, de, num(r["pmra"]) or 0.0, num(r["pmdec"]) or 0.0, 2016.0, epoch)
            (zone if beta < zone_deg else near).append(dict(name="g%s" % r["source_id"], ra=ra19 % 360, dec=de19, plx=float(r["parallax"])))
    hits, null = shifted_matches(zone, "J/ApJS/255/30/comp", radius_arcsec)
    far = set()
    for k in range(0, len(near), 5000):
        far |= {m["source_id"] for m in core.xmatch([dict(source_id=p["name"], ra=p["ra"], dec=p["dec"]) for p in near[k:k + 5000]],
                                                    "J/ApJS/255/30/comp", radius_arcsec)}
    got = {}
    for m in hits:
        if m["source_id"] not in got or float(m["angDist"]) < float(got[m["source_id"]]["angDist"]):
            got[m["source_id"]] = m
    share = len(far) / max(len(near), 1)
    exp = share * len(zone)
    p = float(poisson.sf(len(got) - 1, exp)) if got else 1.0
    sim = {}
    if got:
        for r in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % n[1:] for n in got)):
            sim["g" + str(r["id"]).split()[-1]] = (" ".join(str(r["main_id"]).split()), str(r["otype"]).strip(), (r["sp_type"] or "").strip())
    active = ("BY*", "RS*", "Er*", "Fl*", "TT*", "Y*O", "Y*?", "EB*", "SB*", "**", "CV*", "WD*", "Ro*", "El*", "Em*", "X", "Ae*", "Be*", "a2*", "Pu*", "LP*", "Mi*")
    rows_out = []
    for n, m in sorted(got.items()):
        s = sim.get(n)
        vet = "active-star" if s and s[1] in active else None
        if vet is None:
            ra, de = float(m["RAJ2000"]), float(m["DEJ2000"])
            box = 1.0 / 3600
            for g in core.datalab("SELECT type FROM ls_dr10.tractor WHERE ra BETWEEN %.6f AND %.6f AND dec BETWEEN %.6f AND %.6f" % (
                    ra - box / math.cos(math.radians(de)), ra + box / math.cos(math.radians(de)), de - box, de + box)):
                if g["type"] not in ("PSF", "DUP"):
                    vet = "background-galaxy"
        rows_out.append(dict(name=n, simbad=s, flux=m.get("Ftot"), sep=round(float(m["angDist"]), 2), vet=vet))
    mean = float(np.mean(null))
    print("%d zone stars and %d stars 1-5 deg away north of -40; VLASS within 3 arcsec: zone %d %s, away %d (share %.4f, %.2f expected in the zone, p %.3g); "
          "zone moved 2 arcmin %s (mean %.1f)" % (len(zone), len(near), len(got), [(r["name"], r["simbad"], r["flux"], r["sep"], r["vet"]) for r in rows_out],
                                                  len(far), share, exp, p, null, mean))
    if record:
        core.record_run("etz_vlass_stars", "Radio from Earth's transit zone", "Gaia Catalogue of Nearby Stars (external.gaiaedr3_gcns_main_1); VLASS epoch-1 "
                        "components (J/ApJS/255/30) by CDS XMatch; SIMBAD; Legacy Survey DR10 (Data Lab)",
                        "stars within 100 pc and 0.264 deg of the ecliptic north of -40 deg: %d" % len(zone), os.path.abspath(__file__), "etzradio", len(zone),
                        "stars", dict(zone_deg=zone_deg, control_deg=control_deg, radius_arcsec=radius_arcsec, epoch=epoch, dec_min=dec_min),
                        "the share 1-5 deg away (%d of %d); the zone moved 2 arcmin: %s" % (len(far), len(near), null),
                        "3 GHz emission from a star that sees Earth transit, beyond the rate of stars beside the zone",
                        "%d against %.2f from the share beside the zone (p %.3g) and a mean of %.1f moved" % (len(got), exp, p, mean),
                        [dict(target="gaia_dr3:%s" % r["name"][1:], vet=r["vet"] or ("chance-rate" if p > 0.01 else None),
                              note="SIMBAD %s; VLASS %s mJy at %.2f arcsec" % (r["simbad"], r["flux"], r["sep"])) for r in rows_out])
        print("recorded")


glitchsync_DOC = r"""Glitches firing together: the Jodrell Bank glitch catalogue (Basu et al. 2022, gTable.html); glitches whose epochs are known to 2 days or better, pairs in different pulsars whose epochs agree within the sum of their uncertainties, against the count with each pulsar's glitches moved together by a random 100-2000 days either way in 1000 draws. A spin-up is the neutron star's own superfluid letting go, and nothing natural reaches two neutron stars at once.

    python -I seti.py radio_sky glitchsync [record] [--max-err-days 2] [--shift-days 100 2000] [--draws 1000]"""

GLITCH_URL = "https://www.jb.man.ac.uk/pulsar/glitches/gTable.html"


def glitch_rows():
    """Every glitch of the Jodrell Bank table: pulsar, epoch (MJD), its uncertainty in days (None where the table gives none) and dF/F."""
    path = core.results("jbo_glitches.html")
    if not os.path.exists(path):
        open(path, "wb").write(core.get(GLITCH_URL, timeout=300))
    import html as htmllib
    text = open(path, encoding="latin-1").read()
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S):
        c = [htmllib.unescape(re.sub(r"<[^>]+>", "", x)).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(c) >= 7 and c[0].isdigit() and num(c[4]) is not None:
            out.append(dict(psr=c[2], name=c[1], mjd=float(c[4]), err=num(c[5]), size=num(c[6])))
    return out


def glitch_pairs(gl):
    """Pairs of glitches in different pulsars whose epochs agree within the sum of their uncertainties."""
    gl = sorted(gl, key=lambda g: g["mjd"])
    out = []
    for i, a in enumerate(gl):
        for b in gl[i + 1:]:
            if b["mjd"] - a["mjd"] > 4:
                break
            if a["psr"] != b["psr"] and abs(b["mjd"] - a["mjd"]) <= a["err"] + b["err"]:
                out.append((a, b))
    return out


def glitchsync_main(record=False, max_err_days=2.0, shift_days=[100.0, 2000.0], draws=1000):
    every = glitch_rows()
    gl = [g for g in every if g["err"] is not None and g["err"] <= max_err_days]
    pairs = glitch_pairs(gl)
    rng = np.random.default_rng(7)
    by = {}
    for g in gl:
        by.setdefault(g["psr"], []).append(g)
    drawn = []
    for _ in range(draws):
        moved = []
        for psr, gs in by.items():
            d = rng.uniform(shift_days[0], shift_days[1]) * rng.choice((-1, 1))
            moved += [dict(g, mjd=g["mjd"] + d) for g in gs]
        drawn.append(len(glitch_pairs(moved)))
    draws = np.array(drawn)
    p = float(np.mean(draws >= len(pairs)))
    print("%d glitches in the table, %d timed to 2 days or better in %d pulsars; pairs in different pulsars within their uncertainties: %d %s; moved: mean %.2f, "
          "95th percentile %d, as many or more in %.3f" % (len(every), len(gl), len(by), len(pairs), [(a["name"], a["mjd"], a["err"], b["name"], b["mjd"], b["err"])
                                                                                                    for a, b in pairs], draws.mean(), np.percentile(draws, 95), p))
    if record:
        core.record_run("glitch_coincidences", "Glitches firing together", "Jodrell Bank glitch catalogue (Basu et al. 2022)",
                        "glitches timed to 2 days or better: %d in %d pulsars" % (len(gl), len(by)), os.path.abspath(__file__), "glitchsync", len(gl), "glitches",
                        dict(max_err_days=max_err_days, shift_days=shift_days, draws=len(draws)), "each pulsar's glitches moved together: mean %.2f, 95th percentile %d" % (
                            draws.mean(), np.percentile(draws, 95)), "two neutron stars spinning up at the same moment",
                        "%d pairs against a mean of %.2f moved (p %.3f)" % (len(pairs), draws.mean(), p),
                        [dict(target="psr:J%s" % g["psr"], vet="chance-rate" if p > 0.01 else None, note="glitch at MJD %.2f +- %s, with %s at MJD %.2f" % (
                            g["mjd"], g["err"], o["name"], o["mjd"])) for a, b in pairs for g, o in ((a, b), (b, a))])
        print("recorded")


methanolhigh_DOC = r"""Methanol masers away from the Galactic plane: the 6.7 GHz methanol masers of Pestalozzi, Minier and Booth (2005; J/A+A/432/737) more than 2 deg from the plane; this maser arises in the envelopes of forming massive stars and nowhere else, and the massive-star nurseries lie in the plane. Each is read for its IRAS or MSX counterpart and for SIMBAD's young stars, H II regions and clouds within 1 arcmin, then 5.

    python -I seti.py radio_sky methanolhigh [record] [--latitude-deg 2] [--radius-arcmin 1 5]"""

METHANOL_YOUNG = ("Y*O", "Y*?", "TT*", "TT?", "HII", "MoC", "DNe", "RNe", "ISM", "HH", "out", "of?", "pA?", "smm", "mm", "Cl*", "OpC", "cor", "Ae*", "Ae?", "Or*",
                  "FIR", "IR", "EmO", "PoC", "GNe", "SFR", "bub", "Cld", "glb")


def methanolhigh_main(record=False, latitude_deg=2.0, radius_arcmin=[1.0, 5.0]):
    rows = core.tap("vizier", 'SELECT Name, RAJ2000, DEJ2000, GLON, GLAT, IRAS, MSX, OName, PFlux FROM "J/A+A/432/737/catalog"')
    off = [r for r in rows if abs(float(r["GLAT"])) > latitude_deg]
    out = []
    for r in off:
        young, radius = {}, None
        for arcmin in radius_arcmin:
            kinds = core.simbad_types_near(float(r["RAJ2000"]), float(r["DEJ2000"]), arcmin / 60.0)
            young = {k: n for k, n in kinds.items() if k in METHANOL_YOUNG}
            if young:
                radius = arcmin
                break
        counterpart = (r["IRAS"] or "").strip() or (r["MSX"] or "").strip()
        vet = "young-star" if young or counterpart else None
        out.append(dict(name=r["Name"].strip(), b=float(r["GLAT"]), flux=num(r["PFlux"]), other=(r["OName"] or "").strip(), counterpart=counterpart,
                        young=young, radius=radius, vet=vet))
    left = [o for o in out if o["vet"] is None]
    print("%d methanol masers, %d more than 2 deg from the plane; %d at a star-forming region (an IRAS or MSX source, or SIMBAD's young objects within 1 or 5 arcmin); "
          "unexplained %d %s" % (len(rows), len(off), len(off) - len(left), len(left), [(o["name"], o["b"], o["flux"]) for o in left]))
    for o in out:
        print("  %s b %+.2f, %s Jy, %s; counterpart %s; SIMBAD within %s arcmin %s; %s" % (o["name"], o["b"], o["flux"], o["other"] or "-",
                                                                                    o["counterpart"] or "none", o["radius"], o["young"], o["vet"]))
    if record:
        core.record_run("methanol_masers_off_plane", "Methanol masers away from the Galactic plane", "Pestalozzi, Minier and Booth (2005) 6.7 GHz methanol masers "
                        "(J/A+A/432/737); SIMBAD", "masers more than 2 deg from the plane: %d of %d" % (len(off), len(rows)), os.path.abspath(__file__), "methanolhigh",
                        len(off), "masers", dict(latitude_deg=latitude_deg, radius_arcmin=radius_arcmin), "none: each maser read for its star-forming region",
                        "a 6.7 GHz methanol maser with no forming massive star to pump it", "%d at star-forming regions, %d unexplained" % (len(off) - len(left), len(left)),
                        [dict(target="maser:%s" % o["name"], stat=o["b"], unit="Galactic latitude, deg", vet=o["vet"],
                              note="%s; counterpart %s; SIMBAD within %s arcmin %s" % (o["other"] or "-", o["counterpart"] or "none", o["radius"], o["young"])) for o in out])
        print("recorded")


rfcnear_DOC = r"""Compact radio sources that move like neighbours: the Radio Fundamental Catalogue (Petrov and Kovalev 2025, rfc_2026c: 23866 VLBI-compact sources) against Gaia DR3 within 0.5 arcsec by CDS XMatch, keeping sources whose Gaia counterpart has a parallax above 2 mas at 5 sigma; a milliarcsecond-compact radio emitter within 500 pc with nothing stellar under it. Each read in SIMBAD, where radio stars, young stars and binaries explain their own emission; a counterpart more than 0.1 arcsec from the VLBI position is a star in front of a background source, and a galaxy's parallax or a doubtful Gaia solution (RUWE, excess noise, IPD peaks) is spurious.

    python -I seti.py radio_sky rfcnear [record] [--radius-arcsec 0.5] [--parallax-mas 2] [--sigma 5] [--offset-arcsec 0.1]"""

RFC_URL = "https://astrogeo.org/sol/rfc/rfc_2026c/rfc_2026c_cat.txt"


def rfc_sources():
    """Every RFC source: name, common name, position (deg), position errors (mas) and the flux densities at S, C, X, U and K band (Jy, short baselines)."""
    path = core.results("rfc_2026c_cat.txt")
    if not os.path.exists(path):
        open(path, "wb").write(core.get(RFC_URL, timeout=300))
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("RFC "):
            continue
        ra = 15 * (int(line[26:28]) + int(line[29:31]) / 60 + float(line[32:41]) / 3600)
        de = (int(line[43:45]) + int(line[46:48]) / 60 + float(line[49:57]) / 3600) * (-1 if line[42] == "-" else 1)
        flux = {b: num(line[k:k + 6]) for b, k in (("S", 104), ("C", 126), ("X", 148), ("U", 170), ("K", 192))}
        out.append(dict(name=line[0:14].strip(), common=line[16:24].strip(), ra=ra, dec=de, era=num(line[59:65]), ede=num(line[66:72]),
                        flux={b: f for b, f in flux.items() if f is not None and f > -9}))
    return out


def rfcnear_main(record=False, radius_arcsec=0.5, parallax_mas=2.0, sigma=5.0, offset_arcsec=0.1):
    src = rfc_sources()
    up = [dict(source_id="r%d" % i, ra=s["ra"], dec=s["dec"]) for i, s in enumerate(src)]
    near = {}
    for k in range(0, len(up), 5000):
        for m in core.xmatch(up[k:k + 5000], "I/355/gaiadr3", radius_arcsec):
            plx, e = num(m.get("Plx")), num(m.get("e_Plx"))
            if plx is not None and e and plx > parallax_mas and plx > sigma * e:
                i = int(m["source_id"][1:])
                if i not in near or float(m["angDist"]) < float(near[i]["angDist"]):
                    near[i] = m
    sim, astro = {}, {}
    if near:
        for s in core.simbad("SELECT i.id, b.main_id, b.otype, b.sp_type FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE i.id IN (%s)" % ",".join(
                "'Gaia DR3 %s'" % m["Source"] for m in near.values())):
            sim[str(s["id"]).split()[-1]] = (" ".join(str(s["main_id"]).split()), str(s["otype"]).strip(), (s["sp_type"] or "").strip())
        for g in core.tap("gaia", "SELECT source_id, parallax_over_error, ruwe, astrometric_excess_noise_sig, ipd_frac_multi_peak FROM gaiadr3.gaia_source WHERE source_id IN (%s)"
                          % ",".join(m["Source"] for m in near.values())):
            astro[g["source_id"]] = g
    stellar = ("*", "PM*", "HV*", "V*", "Ro*", "BY*", "RS*", "Er*", "Fl*", "TT*", "Y*O", "Y*?", "EB*", "SB*", "**", "Al*", "bL*", "WR*", "Be*", "Ae*", "s*b", "s*r", "s*y", "RG*",
               "AB*", "Mi*", "LP*", "sr*", "Pu*", "dS*", "Ce*", "gD*", "a2*", "Sy*", "CV*", "No*", "XB*", "LXB", "HXB", "Psr", "WD*", "BD*", "LM*", "Em*", "pA*", "pr*", "Or*", "HB*", "Ir*", "HS*", "Pe*", "Y*")
    galaxies = ("QSO", "AGN", "Bla", "BLL", "Bz?", "Q?", "AG?", "G", "Sy1", "Sy2", "rG", "LIN", "SyG", "EmG", "GiG", "BiC", "PaG", "IG")
    out = []
    for i, m in sorted(near.items()):
        s = sim.get(m["Source"])
        a = astro.get(m["Source"], {})
        doubtful = (num(a.get("ruwe")) or 0) > 1.4 or (num(a.get("astrometric_excess_noise_sig")) or 0) > 2 or (num(a.get("ipd_frac_multi_peak")) or 0) > 2
        # a VLBI position and a Gaia position of the same object agree to milliarcseconds; a 0.1 arcsec offset is a star in front of a background source
        vet = ("natural-type" if s and s[1] in stellar else "blend" if float(m["angDist"]) > offset_arcsec else
               "astrometry" if doubtful or (s and s[1] in galaxies) else None)
        out.append(dict(rfc=src[i]["name"], common=src[i]["common"], gaia=m["Source"], plx=num(m["Plx"]), g=num(m["Gmag"]), sep=round(float(m["angDist"]), 2),
                        flux=src[i]["flux"], simbad=s, ruwe=a.get("ruwe"), aen=a.get("astrometric_excess_noise_sig"), vet=vet))
    left = [o for o in out if o["vet"] is None]
    print("%d RFC sources; %d with a Gaia DR3 counterpart within 0.5 arcsec at parallax above 2 mas (5 sigma); %d of a stellar SIMBAD type; unexplained %d" % (
        len(src), len(out), len(out) - len(left), len(left)))
    for o in sorted(out, key=lambda o: -o["plx"]):
        print("  %s (%s): Gaia DR3 %s, parallax %.2f mas, G %.1f, %.2f arcsec, RUWE %s, excess noise %s sigma; %s Jy; SIMBAD %s; %s" % (
            o["rfc"], o["common"] or "-", o["gaia"], o["plx"], o["g"], o["sep"], o["ruwe"], o["aen"], o["flux"], o["simbad"], o["vet"]))
    if record:
        core.record_run("rfc_nearby_compact", "Compact radio sources that move like neighbours", "Radio Fundamental Catalogue rfc_2026c; Gaia DR3 (I/355) by CDS XMatch; SIMBAD",
                        "VLBI-compact sources: %d" % len(src), os.path.abspath(__file__), "rfcnear", len(src), "radio sources",
                        dict(radius_arcsec=radius_arcsec, parallax_mas=parallax_mas, sigma=sigma, offset_arcsec=offset_arcsec),
                        "none: each counterpart read in SIMBAD", "a milliarcsecond-compact radio source within 500 pc that is not a star",
                        "%d within 500 pc, %d of them stars; %d unexplained" % (len(out), len(out) - len(left), len(left)),
                        [dict(target="gaia_dr3:%s" % o["gaia"], stat=o["plx"], unit="parallax mas", vet=o["vet"], note="%s %s at %.2f arcsec; %s Jy; RUWE %s; SIMBAD %s" % (
                            o["rfc"], o["common"], o["sep"], o["flux"], o["ruwe"], o["simbad"])) for o in out])
        print("recorded")


frbrhythm_DOC = r"""Rhythm in the repeaters: the CHIME/FRB first catalogue's repeaters with five or more bursts (Hashimoto et al. 2022 table, J/MNRAS/511/1961), sub-bursts at one arrival time counted once; for each, the share of its burst intervals that are integer multiples of a base period to 2 per cent, over base periods from 10 minutes to 100 days, leaving out the multiples and submultiples of half a sidereal day, which are the transit telescope's own windows, against 2000 sets of the same number of bursts drawn on random days at the real bursts' phases in the sidereal day. A clock keeps its ticks to a fraction of a per cent; a magnetar's activity window, like FRB 20180916B's 16-day cycle, keeps its bursts inside a quarter of the cycle and nowhere near its ticks.

    python -I seti.py radio_sky frbrhythm [record] [--min-bursts 5] [--tol 0.02] [--draws 2000] [--period-range-d 0.00694 100] [--n-periods 4000]"""


def frbrhythm_score(t, periods, tol=0.02):
    """The largest share of a burst list's intervals that are integer multiples of a base period to tol, and that period."""
    t = np.sort(np.asarray(t, float))
    d = t[1:] - t[0]
    best = (0.0, None)
    for P in periods:
        k = d / P
        frac = np.abs(k - np.round(k))
        share = float(np.mean((frac < tol) & (np.round(k) >= 1)))
        if share > best[0]:
            best = (share, float(P))
    return best


def frbrhythm_main(record=False, min_bursts=5, tol=0.02, draws=2000, period_range_d=[10.0 / 1440.0, 100.0], n_periods=4000):
    rows = core.tap("vizier", 'SELECT Name, RpName, MJD400 FROM "J/MNRAS/511/1961/table" WHERE MJD400 IS NOT NULL')
    rep = {}
    for r in rows:
        n = (r["RpName"] or "").strip()
        if n and not n.startswith("-"):
            # the table carries a burst's sub-bursts as rows of their own at one arrival time
            if all(abs(float(r["MJD400"]) - t) > 1e-3 for t in rep.get(n, [])):
                rep.setdefault(n, []).append(float(r["MJD400"]))
    sidereal = 0.99726957
    periods = np.logspace(math.log10(period_range_d[0]), math.log10(period_range_d[1]), n_periods)
    # a transit telescope sees a source once or twice a sidereal day, so its bursts fall at multiples of half that day whatever the source does:
    # ticks at the sidereal day's multiples and submultiples are the telescope's, and the random bursts keep the real bursts' phases in the sidereal day
    periods = periods[np.abs(periods / (sidereal / 2) - np.round(periods / (sidereal / 2))) > 0.01]
    rng = np.random.default_rng(16)
    out = []
    for n, t in sorted(rep.items()):
        if len(t) < min_bursts:
            continue
        t = np.sort(t)
        share, P = frbrhythm_score(t, periods, tol)
        span = t[-1] - t[0]
        phases = np.mod(t, sidereal)
        days = int(span / sidereal) + 1

        def draw():
            return np.sort(t[0] + sidereal * rng.integers(0, days, len(t)) + rng.choice(phases, len(t)) - phases[0])
        null = np.array([frbrhythm_score(draw(), periods, tol)[0] for _ in range(draws)])
        p = float(np.mean(null >= share))
        out.append(dict(name=n, n=len(t), span=round(float(span), 1), share=round(share, 3), period_d=round(P, 4), p=p, null=round(float(np.mean(null)), 3)))
        print("  %s: %d bursts over %.1f d; %.0f per cent of intervals on a %.4f d tick, random bursts in the telescope's windows give %.0f per cent (p %.3f)" % (
            n, len(t), span, 100 * share, P, 100 * np.mean(null), p), flush=True)
    print("%d repeaters with five or more bursts; on a clock beyond chance: %d" % (len(out), sum(1 for o in out if o["p"] < 0.01)))
    if record:
        core.record_run("chime_repeater_rhythm", "Rhythm in the repeaters", "CHIME/FRB first catalogue as tabulated by Hashimoto et al. (2022; J/MNRAS/511/1961)",
                        "repeaters with %d or more bursts: %d" % (min_bursts, len(out)), os.path.abspath(__file__), "frbrhythm", len(out), "repeaters",
                        dict(min_bursts=min_bursts, tolerance=tol, periods_d=period_range_d, n_periods=n_periods, draws=draws, excluded="multiples of half a sidereal day"),
                        "%d random burst sets on random days at the real bursts' sidereal phases" % draws,
                        "a repeater whose intervals are integer multiples of one period to %g" % tol, "; ".join("%s %.0f per cent on %.3f d (p %.3f)" % (
                            o["name"], 100 * o["share"], o["period_d"], o["p"]) for o in out),
                        [dict(target="frb:%s" % o["name"], stat=o["share"], p=o["p"], unit="share of intervals on the best tick", vet="chance-rate" if o["p"] > 0.01 else None,
                              note="%d bursts over %.1f d; tick %.4f d; random %.0f per cent" % (o["n"], o["span"], o["period_d"], 100 * o["null"])) for o in out])
        print("recorded")


wholehz_DOC = r"""Pulsars ticking in whole hertz: ATNF catalogue pulsars whose spin frequency sits within 0.00001 Hz of a whole number of hertz, or whose period sits within 0.00001 ms of a whole number of milliseconds, a clock set to the second, against the same windows at half-integers and the count uniform fractions give; a catalogue period rounded to few digits lands on the grid by construction, so each match is read for the digits its period carries.

    python -I seti.py radio_sky wholehz [record] [--tol 0.00001] [--digits-min 8]"""


def wholehz_main(record=False, tol=0.00001, digits_min=8):
    from scipy.stats import poisson
    rows = catalogues.pulsars("PSRJ, P0, P1, DM, Dist, Binary, Assoc")
    out, half, n, n_prec = [], [], 0, 0
    for r in rows:
        raw = str(r["P0"]).strip()
        P = num(r["P0"])
        if P is None or P <= 0:
            continue
        digits = len(raw.replace("-", "").replace(".", "").lstrip("0"))
        n += 1
        n_prec += digits >= digits_min
        f = 1.0 / P
        for key, val, unit in (("frequency", f, "Hz"), ("period", P * 1000.0, "ms")):
            k = round(val)
            o = dict(psr=r["PSRJ"], key=key, value=val, n=int(k), off=val - k, unit=unit, P=P, raw=raw, digits=digits, dm=num(r["DM"]), binary=str(r["Binary"] or "").strip(),
                     assoc=str(r["Assoc"] or "").strip(), vet=None)
            if k >= 1 and abs(val - k) < tol:
                o["vet"] = "single-record" if digits < digits_min else None
                out.append(o)
            elif abs(val - (math.floor(val) + 0.5)) < tol:
                half.append(o)
    firm = [o for o in out if o["vet"] is None]
    rounded = [o for o in out if o["vet"] == "single-record"]
    half = [o for o in half if o["digits"] >= digits_min]
    # the expectation counts the periods measured finely enough to land in the window by measurement rather than by rounding
    exp = n_prec * 2 * tol * 2
    p = float(poisson.sf(len(firm) - 1, exp)) if firm else 1.0
    for o in firm:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d pulsars with periods, %d of them quoted to %d digits or more; of those, within %g of a whole hertz or whole millisecond: %d (%.3f expected from uniform "
          "fractions, Poisson p %.3g), at half-integers: %d; periods rounded to fewer digits that land on the grid by construction: %d" % (
              n, n_prec, digits_min, tol, len(firm), exp, p, len(half), len(rounded)))
    for o in sorted(out, key=lambda o: abs(o["off"])):
        print("  %s: %s %.6f %s (%d, off by %+.6f), P0 %s s (%d digits), DM %s, %s%s; %s" % (
            o["psr"], o["key"], o["value"], o["unit"], o["n"], o["off"], o["raw"], o["digits"], o["dm"], "binary" if o["binary"] else "isolated", " " + o["assoc"] if o["assoc"] else "",
            o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out, half=half), open(core.results("pulsars_whole_hz.json"), "w"), indent=1)
    if record:
        core.record_run(
            "psrcat_whole_hertz", "Pulsars ticking in whole hertz", "ATNF pulsar catalogue (VizieR B/psr)",
            "pulsars with periods quoted to %d digits or more: %d of %d" % (digits_min, n_prec, n), os.path.abspath(__file__), "wholehz",
            n_prec, "pulsars", dict(tol=tol, digits_min=digits_min), "uniform fractions give %.3f in the two windows; the half-integer windows hold %d" % (exp, len(half)),
            "a spin frequency within %g Hz of a whole hertz or a period within %g ms of a whole millisecond, among periods quoted to %d digits or more" % (tol, tol, digits_min),
            "%d against %.3f expected and %d at half-integers (Poisson p %.3g); %d more periods rounded to fewer digits land on the grid by construction" % (
                len(firm), exp, len(half), p, len(rounded)),
            [dict(target="psr:%s" % o["psr"], stat=o["off"], unit="%s from a whole %s" % (o["unit"], o["unit"]), vet=o["vet"],
                  note="%s %.6f %s, P0 %s s (%d digits), DM %s" % (o["key"], o["value"], o["unit"], o["raw"], o["digits"], o["dm"])) for o in out])
        print("recorded")


hydrogenhz_DOC = r"""Pulsars on the hydrogen line's decimals: ATNF catalogue pulsars whose spin frequency, quoted to eight digits or more, sits within one part in 100,000 of the hydrogen line's frequency scaled by a power of ten (1.420405751 Hz, 14.20405751 Hz, 142.0405751 Hz, 0.1420405751 Hz), the one frequency every radio astronomer knows, against the same windows around the digits shifted by half a decade and the count uniform logarithmic fractions give.

    python -I seti.py radio_sky hydrogenhz [record] [--tol-rel 0.00001] [--digits-min 8]"""

HYDROGEN_HZ = 1420.405751768


def hydrogenhz_main(record=False, tol_rel=0.00001, digits_min=8):
    from scipy.stats import poisson
    rows = catalogues.pulsars("PSRJ, P0, DM, Dist, Binary, Assoc")
    out, ctl, n = [], [], 0
    for r in rows:
        raw = str(r["P0"]).strip()
        P = num(r["P0"])
        if P is None or P <= 0 or len(raw.replace("-", "").replace(".", "").lstrip("0")) < digits_min:
            continue
        n += 1
        f = 1.0 / P
        # the decade in which the line's digits fall nearest this frequency
        k = round(math.log10(f / HYDROGEN_HZ))
        target = HYDROGEN_HZ * 10.0 ** k
        o = dict(psr=r["PSRJ"], f=f, target=target, decade=int(k), rel=f / target - 1, P=P, dm=num(r["DM"]), binary=str(r["Binary"] or "").strip(), vet=None)
        if abs(o["rel"]) < tol_rel:
            out.append(o)
        k2 = round(math.log10(f / HYDROGEN_HZ) - 0.5)
        t2 = HYDROGEN_HZ * 10.0 ** (k2 + 0.5)
        if abs(f / t2 - 1) < tol_rel:
            ctl.append(dict(o, target=t2))
    # a window of width 2 tol_rel in log10 f covers that fraction of each decade, uniform in log frequency
    exp = n * 2 * tol_rel / math.log(10)
    p = float(poisson.sf(len(out) - 1, exp)) if out else 1.0
    for o in out:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d pulsars with frequencies from periods of %d digits or more; within %g of the hydrogen line's digits in any decade: %d (%.4f expected, Poisson p %.3g); "
          "at the half-decade shift: %d" % (n, digits_min, tol_rel, len(out), exp, p, len(ctl)))
    for o in sorted(out, key=lambda o: abs(o["rel"])):
        print("  %s: %.8f Hz against %.8f (decade %d, off by %.2e), P0 %.9f s, DM %s, %s; %s" % (
            o["psr"], o["f"], o["target"], o["decade"], o["rel"], o["P"], o["dm"], "binary" if o["binary"] else "isolated", o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out, control=ctl), open(core.results("pulsars_hydrogen_decimals.json"), "w"), indent=1)
    if record:
        core.record_run(
            "psrcat_hydrogen_decimals", "Pulsars on the hydrogen line's decimals", "ATNF pulsar catalogue (VizieR B/psr)",
            "pulsars with periods quoted to %d digits or more: %d" % (digits_min, n), os.path.abspath(__file__), "hydrogenhz", n, "pulsars",
            dict(tol_rel=tol_rel, digits_min=digits_min, line_hz=HYDROGEN_HZ), "uniform logarithmic fractions give %.4f; the half-decade windows hold %d" % (exp, len(ctl)),
            "a spin frequency within %g of 1420.405751 Hz scaled by a power of ten" % tol_rel,
            "%d against %.4f expected and %d at the half-decade shift (Poisson p %.3g)" % (len(out), exp, len(ctl), p),
            [dict(target="psr:%s" % o["psr"], stat=o["rel"], unit="fractional offset from the line's digits", vet=o["vet"],
                  note="%.8f Hz against %.8f (decade %d), P0 %.9f s, DM %s" % (o["f"], o["target"], o["decade"], o["P"], o["dm"])) for o in out])
        print("recorded")


maserrest_DOC = r"""Masers at rest in the universe's frame: the 6.7 GHz methanol masers of Pestalozzi et al. (2005) (VizieR J/A+A/432/737), each peak velocity taken from the kinematic local standard of rest to the Sun and then to the cosmic microwave background's frame (the Sun at 369.82 km/s toward Galactic 264.0, +48.3 deg), those within 2 km/s of rest there, a beacon tuned to the one frame every observer shares, against the count the velocities' own density around zero gives; a maser's velocity is its cloud's, so a match at rest is a coincidence unless the masers crowd there.

    python -I seti.py radio_sky maserrest [record] [--tol-kms 2] [--window-kms 30]"""


def maserrest_main(record=False, tol_kms=2.0, window_kms=30.0):
    from scipy.stats import poisson
    rows = core.tap("vizier", 'SELECT Name, RAJ2000, DEJ2000, GLON, GLAT, Vlsr, Vmin, Vmax, PFlux FROM "J/A+A/432/737/catalog" WHERE Vlsr IS NOT NULL')

    def unit(l, b):
        l, b = math.radians(l), math.radians(b)
        return (math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b))
    # the standard solar motion defining the kinematic LSR, 20 km/s toward RA 18h Dec +30 (1900), in Galactic coordinates; the CMB dipole from Planck 2018
    std = unit(56.16, 22.75)
    cmb = unit(264.021, 48.253)
    out = []
    for r in rows:
        v, l, b = num(r["Vlsr"]), num(r["GLON"]), num(r["GLAT"])
        if None in (v, l, b):
            continue
        n = unit(l, b)
        v_helio = v - 20.0 * sum(a * c for a, c in zip(n, std))
        v_cmb = v_helio + 369.82 * sum(a * c for a, c in zip(n, cmb))
        out.append(dict(name=str(r["Name"]).strip(), l=l, b=b, v_lsr=v, v_helio=v_helio, v_cmb=v_cmb, flux=num(r["PFlux"]), vmin=num(r["Vmin"]), vmax=num(r["Vmax"])))
    vc = np.array([o["v_cmb"] for o in out])
    hits = [o for o in out if abs(o["v_cmb"]) < tol_kms]
    near = int((np.abs(vc) < window_kms).sum())
    exp = near * tol_kms / window_kms
    p = float(poisson.sf(len(hits) - 1, exp)) if hits else 1.0
    for o in hits:
        o["vet"] = "chance-rate" if p > 0.01 else None
    print("%d methanol masers with velocities; in the microwave background's frame they run from %.0f to %.0f km/s, %d within %g km/s of rest; at rest within %g km/s: %d "
          "(%.2f expected from the density within %g km/s, Poisson p %.3g)" % (len(out), vc.min(), vc.max(), near, window_kms, tol_kms, len(hits), exp, window_kms, p))
    for o in sorted(hits, key=lambda o: abs(o["v_cmb"])):
        print("  %s (l %.2f, b %+.2f): V_LSR %+.1f, heliocentric %+.1f, in the background's frame %+.2f km/s, range %s to %s, %s Jy; %s" % (
            o["name"], o["l"], o["b"], o["v_lsr"], o["v_helio"], o["v_cmb"], o["vmin"], o["vmax"], o["flux"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(out), hits=hits), open(core.results("masers_cmb_rest.json"), "w"), indent=1)
    if record:
        core.record_run(
            "methanol_masers_cmb_rest", "Masers at rest in the universe's frame", "Pestalozzi et al. (2005) 6.7 GHz methanol maser catalogue (VizieR J/A+A/432/737)",
            "masers with peak velocities: %d" % len(out), os.path.abspath(__file__), "maserrest", len(out), "masers", dict(tol_kms=tol_kms, window_kms=window_kms,
                                                                                                                         sun_cmb_kms=[369.82, 264.021, 48.253]),
            "the velocities' own density within %g km/s of rest: %d, giving %.2f in the window" % (window_kms, near, exp),
            "a maser within %g km/s of rest in the microwave background's frame" % tol_kms, "%d against %.2f expected (Poisson p %.3g)" % (len(hits), exp, p),
            [dict(target="maser:%s" % o["name"], stat=o["v_cmb"], unit="km/s in the microwave background's frame", vet=o["vet"],
                  note="V_LSR %+.1f, heliocentric %+.1f km/s, range %s to %s, %s Jy, l %.2f b %+.2f" % (o["v_lsr"], o["v_helio"], o["vmin"], o["vmax"], o["flux"], o["l"], o["b"]))
             for o in hits])
        print("recorded")


psrages_DOC = r"""Slow pulsars older than the universe: ATNF catalogue pulsars with periods above 0.1 s, so not recycled, whose intrinsic period derivative (the Shklovskii term removed) is positive but so small that the characteristic age P / (2 Pdot) exceeds 15 Gyr, a slow clock that has stopped losing time, which a dipole spinning down cannot be; each read for the derivative's error, since an age beyond the universe's from a derivative measured at a few sigma is the error, and for its binary and association.

    python -I seti.py radio_sky psrages [record] [--p-min-s 0.1] [--age-min-gyr 15] [--sigma 3]"""


def psrages_main(record=False, p_min_s=0.1, age_min_gyr=15.0, sigma=3.0):
    rows = catalogues.pulsars("PSRJ, P0, P1, e_P1, P1I, DM, Dist, Binary, Assoc")
    out, n = [], 0
    for r in rows:
        P, pd, epd, pdi = num(r["P0"]), num(r["P1"]), num(r["e_P1"]), num(r["P1I"])
        if P is None or P < p_min_s or pd is None:
            continue
        n += 1
        use = pdi if pdi is not None else pd
        if use <= 0:
            continue
        age = P / (2 * use) / (365.25 * 86400 * 1e9)
        if age > age_min_gyr:
            out.append(dict(psr=r["PSRJ"], P=P, pdot=pd, e_pdot=epd, pdot_int=pdi, age_gyr=age, snr=(pd / epd) if epd else None, dm=num(r["DM"]), dist=num(r["Dist"]),
                            binary=str(r["Binary"] or "").strip(), assoc=str(r["Assoc"] or "").strip(),
                            vet="single-record" if not epd or pd / epd < sigma else "binary" if str(r["Binary"] or "").strip() else None))
    left = [o for o in out if o["vet"] is None]
    print("%d pulsars slower than %g s with a period derivative; characteristic ages beyond %g Gyr: %d, %d with derivatives below %g sigma, %d in binaries; unexplained %d" % (
        n, p_min_s, age_min_gyr, len(out), sum(o["vet"] == "single-record" for o in out), sigma, sum(o["vet"] == "binary" for o in out), len(left)))
    for o in sorted(out, key=lambda o: -o["age_gyr"]):
        print("  %s: P %.4f s, Pdot %.2e +- %s (intrinsic %s), age %.0f Gyr, DM %s, %s kpc%s%s; %s" % (
            o["psr"], o["P"], o["pdot"], o["e_pdot"], o["pdot_int"], o["age_gyr"], o["dm"], o["dist"], ", binary " + o["binary"] if o["binary"] else "",
            ", " + o["assoc"] if o["assoc"] else "", o["vet"] or "unexplained"))
    json.dump(dict(n=n, hits=out), open(core.results("pulsars_older_than_universe.json"), "w"), indent=1)
    if record:
        core.record_run(
            "psrcat_slow_ancient", "Slow pulsars older than the universe", "ATNF pulsar catalogue (VizieR B/psr)", "pulsars slower than %g s with a period derivative: %d" % (p_min_s, n),
            os.path.abspath(__file__), "psrages", n, "pulsars", dict(p_min_s=p_min_s, age_min_gyr=age_min_gyr, sigma=sigma), "the derivatives' errors and the binaries",
            "a slow pulsar whose intrinsic spin-down gives an age beyond %g Gyr" % age_min_gyr,
            "%d: %d with derivatives below %g sigma, %d in binaries; %d unexplained" % (len(out), sum(o["vet"] == "single-record" for o in out), sigma,
                                                                                       sum(o["vet"] == "binary" for o in out), len(left)),
            [dict(target="psr:%s" % o["psr"], stat=o["age_gyr"], unit="characteristic age, Gyr", vet=o["vet"],
                  note="P %.4f s, Pdot %.2e +- %s, intrinsic %s, DM %s, %s" % (o["P"], o["pdot"], o["e_pdot"], o["pdot_int"], o["dm"], o["binary"] or "isolated")) for o in out])
        print("recorded")


coldbdradio_DOC = r"""Radio from cold brown dwarfs: the T and Y dwarfs of the Kirkpatrick et al. (2021) 20 pc census carried to 2019.0 and matched to the VLASS epoch-1 component catalogue (Gordon et al. 2021, VizieR J/ApJS/255/30) within 5 arcsec by CDS XMatch, a 3 GHz source at a body of a few hundred kelvin, whose auroral radio is rare and faint and whose made transmitter would not be, against the positions moved 2 arcmin in eight directions; each match read for its flux and extent.

    python -I seti.py radio_sky coldbdradio [record] [--radius-arcsec 5] [--epoch 2019.0]"""


def coldbdradio_main(record=False, radius_arcsec=5.0, epoch=2019.0):
    from scipy.stats import poisson
    from setitools.catalogues import cold_dwarfs, shifted_matches
    dw = cold_dwarfs(epoch)
    pos = [dict(name=d["name"], ra=d["ra"], dec=d["dec"]) for d in dw]
    hits, null = shifted_matches(pos, "J/ApJS/255/30/comp", radius_arcsec)
    got = {}
    for m in hits:
        if m["source_id"] not in got or float(m["angDist"]) < float(got[m["source_id"]]["angDist"]):
            got[m["source_id"]] = m
    exp = float(np.mean(null)) if len(null) else 0.0
    p = float(poisson.sf(len(got) - 1, exp)) if got and exp > 0 else 1.0
    out = []
    by = {d["name"]: d for d in dw}
    for n, m in got.items():
        d = by[n]
        out.append(dict(name=n, sptype=d["sptype"], teff=d["teff"], plx=d["plx"], sep=float(m["angDist"]), flux_mjy=num(m.get("Ftot")), peak=num(m.get("Fpeak")), maj=num(m.get("Maj")),
                        vet="chance-rate" if p > 0.01 else None))
    print("%d T and Y dwarfs within 20 pc; VLASS components within %g arcsec at %.1f: %d (the positions moved 2 arcmin give %s, mean %.2f; Poisson p %.3g)" % (
        len(dw), radius_arcsec, epoch, len(got), null, exp, p))
    for o in sorted(out, key=lambda o: o["sep"]):
        print("  %s (%s, %s K, %.1f mas): %.1f mJy total, %.1f mJy peak, %s arcsec major axis at %.1f arcsec; %s" % (
            o["name"], o["sptype"], o["teff"], o["plx"], o["flux_mjy"] or 0, o["peak"] or 0, o["maj"], o["sep"], o["vet"] or "unexplained"))
    json.dump(dict(n=len(dw), null=null, hits=out), open(core.results("cold_dwarfs_vlass.json"), "w"), indent=1)
    if record:
        core.record_run(
            "vlass_cold_dwarfs", "Radio from cold brown dwarfs", "Kirkpatrick et al. (2021) 20 pc census; VLASS epoch-1 components (VizieR J/ApJS/255/30) by CDS XMatch",
            "T and Y dwarfs within 20 pc: %d" % len(dw), os.path.abspath(__file__), "coldbdradio", len(dw), "dwarfs", dict(radius_arcsec=radius_arcsec, epoch=epoch),
            "the positions moved 2 arcmin in eight directions: %s" % null, "a VLASS component within %g arcsec of a T or Y dwarf at %.1f" % (radius_arcsec, epoch),
            "%d against a mean of %.2f at the moved positions (Poisson p %.3g)" % (len(got), exp, p),
            [dict(target="dwarf:%s" % " ".join(o["name"].split()), stat=o["flux_mjy"], unit="VLASS total flux, mJy", vet=o["vet"],
                  note="%s, %s K, %.1f arcsec, peak %s mJy, major axis %s arcsec" % (o["sptype"], o["teff"], o["sep"], o["peak"], o["maj"])) for o in out])
        print("recorded")


psretz_DOC = r"""Pulsars in Earth's transit zone: the ATNF catalogue's pulsars by ecliptic latitude, the count within the 0.262 deg half-width of the zone from which Earth is seen to transit the Sun against the mean and scatter of the twenty strips of that width on either side out to 5.5 deg, a beacon placed where a watcher of our transits would place it; the Galactic plane crosses the ecliptic twice and fills the strips beside the zone as it fills the zone, and each pulsar in the zone is listed with its period, distance and binary.

    python -I seti.py radio_sky psretz [record] [--half-width-deg 0.262] [--band-deg 5.5]"""


def psretz_main(record=False, half_width_deg=0.262, band_deg=5.5):
    rows = catalogues.pulsars("PSRJ, RAJ2000, DEJ2000, P0, DM, Dist, Assoc, Binary, PB")
    ps = [dict(name=str(r["PSRJ"]).strip(), ra=num(r["RAJ2000"]), dec=num(r["DEJ2000"]), P=num(r["P0"]), dm=num(r["DM"]), dist=num(r["Dist"]), assoc=str(r["Assoc"] or "").strip(),
               binary=str(r["Binary"] or "").strip(), pb=num(r["PB"])) for r in rows]
    ps = [p for p in ps if p["ra"] is not None and p["dec"] is not None]
    lam, beta = science.ecliptic_lon_lat([p["ra"] for p in ps], [p["dec"] for p in ps])
    for p, l, b in zip(ps, lam, beta):
        p["lam"], p["beta"] = float(l), abs(float(b))
    z = science.zone_excess([p["beta"] for p in ps], half_width_deg, band_deg)
    msp = science.zone_excess([p["beta"] for p in ps if p["P"] and p["P"] < 0.03], half_width_deg, band_deg)
    inzone = [p for p in ps if p["beta"] < half_width_deg]
    for p in inzone:
        p["vet"] = "chance-rate" if z["p"] > 0.01 and msp["p"] > 0.01 else None
    print("%d pulsars; in the zone: %d against %.2f +- %.2f beside it (z %+.2f, Poisson p %.3g); millisecond pulsars: %d against %.2f +- %.2f (z %+.2f, p %.3g)" % (
        len(ps), z["zone"], z["mean"], z["sd"], z["z"], z["p"], msp["zone"], msp["mean"], msp["sd"], msp["z"], msp["p"]))
    for p in sorted(inzone, key=lambda p: p["beta"]):
        print("  %s: ecliptic latitude %+.3f deg, P %s s, DM %s, %s kpc, %s%s; %s" % (p["name"], p["beta"], p["P"], p["dm"], p["dist"], p["assoc"] or "no association",
                                                                                  ", binary " + p["binary"] if p["binary"] else "", p["vet"] or "unexplained"))
    json.dump(dict(n=len(ps), all=z, msp=msp, zone=inzone), open(core.results("pulsars_transit_zone.json"), "w"), indent=1)
    if record:
        core.record_run(
            "atnf_transit_zone", "Pulsars in Earth's transit zone", "ATNF pulsar catalogue (VizieR B/psr)", "pulsars with positions: %d" % len(ps), os.path.abspath(__file__),
            "psretz", len(ps), "pulsars", dict(half_width_deg=half_width_deg, band_deg=band_deg),
            "the twenty strips of the zone's width beside it: %.2f +- %.2f, millisecond pulsars %.2f +- %.2f" % (z["mean"], z["sd"], msp["mean"], msp["sd"]),
            "an excess of pulsars in the band of sky that sees Earth transit the Sun",
            "%d in the zone (z %+.2f, p %.3g), %d millisecond pulsars (z %+.2f, p %.3g)" % (z["zone"], z["z"], z["p"], msp["zone"], msp["z"], msp["p"]),
            [dict(target="pulsar:%s" % p["name"], stat=round(p["beta"], 4), unit="ecliptic latitude, deg", vet=p["vet"],
                  note="P %s s, DM %s, %s kpc, %s%s" % (p["P"], p["dm"], p["dist"], p["assoc"] or "no association", ", binary " + p["binary"] if p["binary"] else ""))
             for p in inzone])
        print("recorded")


psrkick_DOC = r"""Pulsars faster than a kick can throw: ATNF catalogue pulsars with a measured proper motion and a distance, their transverse speeds with the proper motion taken at its lower bound, those above 2000 km/s, beyond the fastest kick a supernova is seen to give and beyond the Galaxy's escape speed, a thing driven rather than thrown; each read for its distance's provenance, since a dispersion-measure distance can be several times wrong, and for its proper motion's error.

    python -I seti.py radio_sky psrkick [record] [--v-max 2000] [--sigma 3]"""


def psrkick_main(record=False, v_max=2000.0, sigma=3.0):
    rows = catalogues.pulsars("PSRJ, pmRA, e_pmRA, pmDE, e_pmDE, Plx, e_Plx, Dist, DistDM, r_DistDM, P0, Binary, Assoc, VTrans", "pmRA IS NOT NULL AND pmDE IS NOT NULL AND Dist IS NOT NULL")
    ps = []
    for r in rows:
        pmra, pmdec, d = num(r["pmRA"]), num(r["pmDE"]), num(r["Dist"])
        if None in (pmra, pmdec, d) or d <= 0:
            continue
        era, ede = num(r["e_pmRA"]) or 0.0, num(r["e_pmDE"]) or 0.0
        pm = math.hypot(pmra, pmdec)
        epm = math.hypot(pmra * era, pmdec * ede) / pm if pm else 0.0
        low = max(pm - sigma * epm, 0.0)
        ps.append(dict(name=str(r["PSRJ"]).strip(), pm=pm, epm=epm, d=d, dm_distance=num(r["DistDM"]) == d, plx=num(r["Plx"]), eplx=num(r["e_Plx"]), v=4.74047 * pm * d,
                       v_low=4.74047 * low * d, P=num(r["P0"]), binary=str(r["Binary"] or "").strip(), assoc=str(r["Assoc"] or "").strip()))
    hits = [p for p in ps if p["v_low"] > v_max]
    for p in hits:
        p["vet"] = "published-solution" if p["dm_distance"] and not p["plx"] else None
    left = [p for p in hits if p["vet"] is None]
    vs = sorted(p["v"] for p in ps)
    print("%d pulsars with a proper motion and a distance; transverse speeds: median %.0f km/s, the fastest %.0f; above %g km/s at the proper motion's %g sigma lower bound: %d "
          "(%d on dispersion-measure distances); unexplained %d" % (len(ps), np.median(vs), vs[-1], v_max, sigma, len(hits), sum(p["vet"] == "published-solution" for p in hits),
                                                                  len(left)))
    for p in sorted(hits, key=lambda p: -p["v_low"]):
        print("  %s: %.0f km/s (lower bound %.0f) from %.1f +- %.1f mas/yr at %.2f kpc (%s), P %s s, %s%s; %s" % (
            p["name"], p["v"], p["v_low"], p["pm"], p["epm"], p["d"], "dispersion measure" if p["dm_distance"] else "independent distance", p["P"], p["assoc"] or "no association",
            ", binary " + p["binary"] if p["binary"] else "", p["vet"] or "unexplained"))
    json.dump(dict(n=len(ps), hits=hits), open(core.results("pulsars_kick_speeds.json"), "w"), indent=1)
    if record:
        core.record_run(
            "atnf_kick_speeds", "Pulsars faster than a kick can throw", "ATNF pulsar catalogue (VizieR B/psr)", "pulsars with a proper motion and a distance: %d" % len(ps),
            os.path.abspath(__file__), "psrkick", len(ps), "pulsars", dict(v_max=v_max, sigma=sigma), "the transverse speeds' distribution: median %.0f km/s" % np.median(vs),
            "a transverse speed above %g km/s at the proper motion's %g sigma lower bound" % (v_max, sigma),
            "%d, %d of them on dispersion-measure distances; %d unexplained" % (len(hits), sum(p["vet"] == "published-solution" for p in hits), len(left)),
            [dict(target="pulsar:%s" % p["name"], stat=round(p["v_low"]), unit="km/s, transverse, lower bound", vet=p["vet"],
                  note="%.0f km/s from %.1f +- %.1f mas/yr at %.2f kpc (%s)" % (p["v"], p["pm"], p["epm"], p["d"], "dispersion measure" if p["dm_distance"] else "independent distance"))
             for p in hits])
        print("recorded")


psrpairs_DOC = r"""Two pulsars at one spot: ATNF catalogue pulsars outside globular clusters and the Magellanic Clouds, pairs within 2 arcmin of one another, two beacons at one place where the Galaxy puts one pulsar per square degree at most, against the pairs found with the declinations mirrored; a pair bound to each other is a binary pulsar system, read from the catalogue's binary fields, and the rest are read for their dispersion measures, which agree for two pulsars at one distance.

    python -I seti.py radio_sky psrpairs [record] [--radius-arcmin 2] [--dm-tol 0.1]"""


def psrpairs_main(record=False, radius_arcmin=2.0, dm_tol=0.1):
    from scipy.stats import poisson
    rows = catalogues.pulsars("PSRJ, RAJ2000, DEJ2000, P0, DM, Dist, Assoc, Binary, PB, BINCOMP")
    ps = [dict(name=str(r["PSRJ"]).strip(), ra=num(r["RAJ2000"]), dec=num(r["DEJ2000"]), P=num(r["P0"]), dm=num(r["DM"]), dist=num(r["Dist"]), assoc=str(r["Assoc"] or "").strip(),
               binary=str(r["Binary"] or "").strip(), pb=num(r["PB"]), comp=str(r["BINCOMP"] or "").strip()) for r in rows]
    ps = [p for p in ps if p["ra"] is not None and p["dec"] is not None and "GC:" not in p["assoc"] and "SMC" not in p["assoc"] and "LMC" not in p["assoc"]]
    ra = np.radians([p["ra"] for p in ps])
    dec = np.radians([p["dec"] for p in ps])

    def pairs(dec_arr):
        xyz = np.stack([np.cos(dec_arr) * np.cos(ra), np.cos(dec_arr) * np.sin(ra), np.sin(dec_arr)], 1)
        m = xyz @ xyz.T > math.cos(math.radians(radius_arcmin / 60.0))
        return [(int(i), int(j)) for i, j in np.argwhere(np.triu(m, 1))]

    real = pairs(dec)
    null = len(pairs(-dec))
    exp = max(null, 0.5)
    out = []
    for i, j in real:
        a, b = ps[i], ps[j]
        sep = core.sep_arcsec(a["ra"], a["dec"], b["ra"], b["dec"])
        dm_same = a["dm"] is not None and b["dm"] is not None and abs(a["dm"] - b["dm"]) <= dm_tol * max(a["dm"], b["dm"])
        bound = a["pb"] is not None and b["pb"] is not None and abs(a["pb"] - b["pb"]) < 1e-3 * max(a["pb"], b["pb"])
        out.append(dict(a=a["name"], b=b["name"], sep_arcsec=sep, P=(a["P"], b["P"]), dm=(a["dm"], b["dm"]), dm_same=dm_same, vet="binary" if bound else "chance-rate" if not dm_same else None))
    p = float(poisson.sf(len(real) - 1, exp)) if real else 1.0
    left = [o for o in out if o["vet"] is None]
    print("%d pulsars outside clusters and the Clouds; pairs within %g arcmin: %d against %d with the declinations mirrored (Poisson p %.3g); %d bound as a double neutron star, "
          "%d at different dispersion measures; unexplained %d" % (len(ps), radius_arcmin, len(real), null, p, sum(o["vet"] == "binary" for o in out),
                                                                 sum(o["vet"] == "chance-rate" for o in out), len(left)))
    for o in sorted(out, key=lambda o: o["sep_arcsec"]):
        print("  %s and %s: %.1f arcsec apart, P %s and %s s, DM %s and %s; %s" % (o["a"], o["b"], o["sep_arcsec"], o["P"][0], o["P"][1], o["dm"][0], o["dm"][1], o["vet"] or "unexplained"))
    json.dump(dict(n=len(ps), null=null, pairs=out), open(core.results("pulsar_pairs_one_spot.json"), "w"), indent=1)
    if record:
        core.record_run(
            "atnf_pulsars_one_spot", "Two pulsars at one spot", "ATNF pulsar catalogue (VizieR B/psr)", "pulsars outside globular clusters and the Magellanic Clouds: %d" % len(ps),
            os.path.abspath(__file__), "psrpairs", len(ps), "pulsars", dict(radius_arcmin=radius_arcmin, dm_tol=dm_tol), "the pairs with the declinations mirrored: %d" % null,
            "two pulsars within %g arcmin" % radius_arcmin, "%d pairs against %d mirrored; %d bound, %d at different dispersion measures; %d unexplained" % (
                len(real), null, sum(o["vet"] == "binary" for o in out), sum(o["vet"] == "chance-rate" for o in out), len(left)),
            [dict(target="pulsar:%s" % o["a"], stat=round(o["sep_arcsec"], 1), unit="arcsec to its partner", vet=o["vet"],
                  note="with %s, P %s and %s s, DM %s and %s" % (o["b"], o["P"][0], o["P"][1], o["dm"][0], o["dm"][1])) for o in out])
        print("recorded")


glitchsched_DOC = r"""Glitches on a schedule: the Jodrell Bank glitch table, pulsars with five or more dated glitches, the scatter of the intervals between successive glitches as a fraction of their mean, against the scatter that intervals drawn at random from an exponential of the same mean give in 4000 draws; a pulsar that glitches on a timetable is keeping time, which the published quasi-periodic glitchers and the young pulsars whose large glitches recur at the crust's critical lag, as Vela's do, keep through their own crust and superfluid; each regular pulsar is read against them and for the size of its glitches.

    python -I seti.py radio_sky glitchsched [record] [--min-glitches 5] [--cv-max 0.3] [--draws 4000]"""

# pulsars whose quasi-periodic glitching is published
GLITCHSCHED_KNOWN = {"0537-6910", "0833-45", "0835-4510", "0205+6449", "1737-30", "1740-3015", "0631+1036", "1758-23", "1801-2304"}
# the glitch size above which a glitch is of Vela's kind, a fractional frequency step of a part in a million
GLITCHSCHED_LARGE = 1e-6


def glitchsched_main(record=False, min_glitches=5, cv_max=0.3, draws=4000):
    rows = glitch_rows()
    by, sizes = {}, {}
    for g in rows:
        by.setdefault(g["psr"], []).append(g["mjd"])
        sizes.setdefault(g["psr"], []).append(g["size"])
    rng = np.random.default_rng(11)
    out, n_tested = [], 0
    for psr, mjds in by.items():
        mjds = sorted(set(mjds))
        if len(mjds) < min_glitches:
            continue
        n_tested += 1
        gaps = np.diff(mjds)
        cv = float(np.std(gaps, ddof=1) / np.mean(gaps))
        sims = rng.exponential(1.0, size=(draws, len(gaps)))
        p = float(np.mean(np.std(sims, axis=1, ddof=1) / np.mean(sims, axis=1) <= cv))
        big = [x for x in sizes[psr] if x is not None]
        large = bool(big) and float(np.median(big)) * 1e-9 >= GLITCHSCHED_LARGE
        o = dict(psr=psr, n=len(mjds), span_d=float(mjds[-1] - mjds[0]), mean_gap_d=float(np.mean(gaps)), cv=cv, p=p, gaps=[round(float(x), 1) for x in gaps], large=large)
        if cv < cv_max:
            key = psr.lstrip("JB")
            o["vet"] = "natural-source" if key in GLITCHSCHED_KNOWN else "vela-like" if large else ("chance-rate" if p * n_tested > 0.05 else None)
            out.append(o)
    left = [o for o in out if o["vet"] is None]
    print("%d pulsars with %d or more dated glitches; interval scatter below %g of the mean: %d (%d published quasi-periodic glitchers, %d glitching large like Vela, %d within "
          "chance over the %d tested); unexplained %d" % (n_tested, min_glitches, cv_max, len(out), sum(o["vet"] == "natural-source" for o in out),
                                                          sum(o["vet"] == "vela-like" for o in out), sum(o["vet"] == "chance-rate" for o in out), n_tested, len(left)))
    for o in sorted(out, key=lambda o: o["cv"]):
        print("  %s: %d glitches over %.0f d, mean gap %.0f d, scatter %.2f of the mean (p %.2g, %.2g over the pulsars tested), gaps %s, %s glitches; %s" % (
            o["psr"], o["n"], o["span_d"], o["mean_gap_d"], o["cv"], o["p"], min(1.0, o["p"] * n_tested), o["gaps"], "large" if o["large"] else "small", o["vet"] or "unexplained"))
    json.dump(dict(tested=n_tested, regular=out), open(core.results("glitch_schedules.json"), "w"), indent=1)
    if record:
        core.record_run(
            "jbo_glitch_schedules", "Glitches on a schedule", "Jodrell Bank glitch table", "pulsars with %d or more dated glitches: %d" % (min_glitches, n_tested),
            os.path.abspath(__file__), "glitchsched", n_tested, "pulsars", dict(min_glitches=min_glitches, cv_max=cv_max, draws=draws),
            "intervals drawn from an exponential of the same mean, %d draws per pulsar" % draws, "glitch intervals scattered by less than %g of their mean" % cv_max,
            "%d regular: %d published quasi-periodic glitchers, %d glitching large like Vela, %d within chance; %d unexplained" % (
                len(out), sum(o["vet"] == "natural-source" for o in out), sum(o["vet"] == "vela-like" for o in out), sum(o["vet"] == "chance-rate" for o in out), len(left)),
            [dict(target="pulsar:%s" % o["psr"], stat=round(o["cv"], 3), unit="interval scatter over the mean", vet=o["vet"],
                  note="%d glitches over %.0f d, mean gap %.0f d, p %.2g, gaps %s" % (o["n"], o["span_d"], o["mean_gap_d"], o["p"], o["gaps"])) for o in out])
        print("recorded")


switchoff_DOC = r"""Radio sources that switched off: NVSS (VizieR VIII/65) sources north of -40 deg and more than 10 deg from the Galactic plane, brighter than 100 mJy at 1.4 GHz and compact, with major axes under 25 arcsec, against the VLASS epoch-1 quick-look components (J/ApJS/255/30) within 20 arcsec by CDS XMatch; a source of that strength falls below VLASS's limit only by fading a hundredfold between 1995 and 2019, which a switched-off transmitter does and a blazar in a deep low state can; each missing source read in SIMBAD, for its NVSS size, since a resolved source VLASS's short baselines dissolve is no fading, for a source above 5 Jy within half a degree, whose sidelobes spoil the quick-look image around it, in RACS-low (J/other/PASA/38.58, 888 MHz, 2019-2020, south of +41 deg), where a source still shining has not faded whatever VLASS lists, and for the catalogue's own holes, a quarter degree holding no component at all, which the component list leaves where its images failed; the rest in the surveys made after VLASS epoch 1 (RACS-mid, RACS-high, RACS-low3, LoTSS DR2) within 30 arcsec, as a lobe or hotspot of a brighter source that RACS-mid or TGSS catalogue as one source within 90 arcsec, and in the VLASS quick-look images themselves at the CADC, where a source above 10 sigma within 30 arcsec in any epoch is one the component list dropped.

    python -I seti.py radio_sky switchoff [record] [--flux-min-mjy 100] [--major-max-arcsec 25] [--radius-arcsec 20] [--dec-min -40] [--hole-deg 0.25] [--b-min 10] [--bright-jy 5] [--bright-deg 0.5] [--racs-dec-max 41] [--later-arcsec 30] [--lobe-arcsec 90] [--image-sigma 10]"""


def switchoff_vlass_images(ra, dec, radius_arcsec=30.0):
    """The brightest pixel within RADIUS_ARCSEC of a position in every VLASS quick-look image the CADC holds of it, in units of the image's noise: [(epoch, sigma, mJy within the radius)]."""
    import io
    from astropy.io import fits
    from astropy.io.votable import parse_single_table
    from astropy.wcs import WCS
    base = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/sia/v2query?"
    out = []
    t = parse_single_table(io.BytesIO(core.get(base + urllib.parse.urlencode({"POS": "CIRCLE %.5f %.5f 0.01" % (ra, dec), "COLLECTION": "VLASS"}), timeout=120))).to_table()
    for r in t:
        try:
            links = parse_single_table(io.BytesIO(core.get(str(r["access_url"]), timeout=120))).to_table()
            url = next((str(x["access_url"]) for x in links if str(x["semantics"]) == "#this" and all(k in str(x["access_url"]) for k in ("pbcor", "tt0", "subim"))), None)
            if not url:
                continue
            h = fits.open(io.BytesIO(core.get(url + "?" + urllib.parse.urlencode({"CIRCLE": "%.6f %.6f %.4f" % (ra, dec, 2 * radius_arcsec / 3600)}), timeout=300)))
            d = np.squeeze(h[0].data).astype(float)
            w = WCS(h[0].header).celestial
            pix = abs(h[0].header.get("CDELT1") or h[0].header.get("CD1_1"))
            beam = 1.1331 * h[0].header["BMAJ"] * h[0].header["BMIN"] / pix ** 2
            x, y = w.world_to_pixel_values(ra, dec)
            yy, xx = np.mgrid[:d.shape[0], :d.shape[1]]
            rr = np.hypot(xx - x, yy - y) * pix * 3600
            ok = np.isfinite(d)
            inner, outer = d[(rr <= radius_arcsec) & ok], d[(rr > radius_arcsec) & ok]
            if not len(inner) or len(outer) < 50:
                continue
            noise = 1.4826 * float(np.median(np.abs(outer - np.median(outer))))
            out.append((str(r["obs_id"]).split(".")[0], float(inner.max() / noise), float(inner.sum() / beam * 1e3)))
        except Exception:
            continue
    return out


def switchoff_main(record=False, flux_min_mjy=100.0, major_max_arcsec=25.0, radius_arcsec=20.0, dec_min=-40.0, hole_deg=0.25, b_min=10.0, bright_jy=5.0, bright_deg=0.5,
                   racs_dec_max=41.0, later_arcsec=30.0, lobe_arcsec=90.0, image_sigma=10.0):
    rows = core.tap("vizier", 'SELECT NVSS, RAJ2000, DEJ2000, "S1.4", MajAxis, MinAxis FROM "VIII/65/nvss" WHERE "S1.4" > %g AND DEJ2000 > %g' % (flux_min_mjy, dec_min), timeout=600)
    sk = next(k for k in rows[0] if k.replace("_", ".").lower() == "s1.4") if rows else "S1.4"
    src = [dict(name=str(r["NVSS"]).strip(), ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"]), s=float(r[sk]), maj=num(r["MajAxis"]), min_=num(r["MinAxis"])) for r in rows]
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    gb = SkyCoord(ra=[s["ra"] for s in src] * u.deg, dec=[s["dec"] for s in src] * u.deg, frame="icrs").galactic.b.deg
    for s, b in zip(src, gb):
        s["b"] = float(b)
    compact = [s for s in src if s["maj"] is not None and s["maj"] < major_max_arcsec and abs(s["b"]) > b_min]
    bright = [s for s in src if s["s"] > 1000 * bright_jy]
    seen = set()
    for k in range(0, len(compact), 20000):
        seen |= {m["source_id"] for m in core.xmatch([dict(source_id=s["name"], ra=s["ra"], dec=s["dec"]) for s in compact[k:k + 20000]], "J/ApJS/255/30/comp", radius_arcsec)}
    gone = [s for s in compact if s["name"] not in seen]
    sim = {}
    for k in range(0, len(gone), 100):
        for m in core.xmatch([dict(source_id=s["name"], ra=s["ra"], dec=s["dec"]) for s in gone[k:k + 100]], "simbad", 10.0):
            if m["source_id"] not in sim or float(m["angDist"]) < sim[m["source_id"]][0]:
                sim[m["source_id"]] = (float(m["angDist"]), " ".join(str(m.get("main_id") or "").split()), str(m.get("main_type") or "").strip())
    racs = {}
    for m in core.xmatch([dict(source_id=s["name"], ra=s["ra"], dec=s["dec"]) for s in gone if s["dec"] < racs_dec_max], "J/other/PASA/38.58/galcut", radius_arcsec):
        f = num(m.get("Ftot")) or num(m.get("Fint")) or num(m.get("Stot"))
        if m["source_id"] not in racs or (f or 0) > (racs[m["source_id"]] or 0):
            racs[m["source_id"]] = f
    for s in gone:
        s["simbad"] = sim.get(s["name"])
        s["racs_mjy"] = racs.get(s["name"])
        typ = (s["simbad"] or (0, "", ""))[2]
        s["vet"] = ("natural-type" if typ and typ not in ("Radio", "Unknown", "?") else "catalogue-dropped" if s["name"] in racs else None)
        s["around"] = None
        if s["vet"] is None:
            near = [b for b in bright if b["name"] != s["name"] and core.sep_arcsec(s["ra"], s["dec"], b["ra"], b["dec"]) < 3600 * bright_deg]
            if near:
                s["vet"] = "bright-neighbour"
                s["bright"] = "%s (%.1f Jy)" % (near[0]["name"], near[0]["s"] / 1000)
        if s["vet"] is None:
            n = core.tap("vizier", 'SELECT COUNT(*) AS n FROM "J/ApJS/255/30/comp" WHERE 1 = CONTAINS(POINT(\'ICRS\', RAJ2000, DEJ2000), CIRCLE(\'ICRS\', %.5f, %.5f, %g))' % (
                s["ra"], s["dec"], hole_deg))
            s["around"] = int(n[0]["n"]) if n else 0
            if s["around"] == 0:
                s["vet"] = "catalogue-dropped"
    # the rest in the surveys made after VLASS epoch 1 (RACS-mid of 2020-21, RACS-high, RACS-low3, LoTSS DR2), as a lobe of a brighter source those surveys and TGSS catalogue
    # as one source within LOBE_ARCSEC (five times the NVSS flux at 1.4 GHz, or twenty-five times at 150 MHz), and in the VLASS images themselves
    later = (("J/other/PASA/41.3/sourcesm", "RACS-mid"), ("J/other/PASA/42.38/sourcesh", "RACS-high"), ("J/other/PASA/43.89/sources", "RACS-low3"), ("J/A+A/659/A1/catalog", "LoTSS DR2"))
    rest = [s for s in gone if s["vet"] is None]
    for cat, label in later:
        for m in core.xmatch([dict(source_id=s["name"], ra=s["ra"], dec=s["dec"]) for s in rest if s["vet"] is None], cat, later_arcsec):
            f = num(m.get("Ftot")) or num(m.get("Stotal")) or num(m.get("SpeakTot")) or num(m.get("Fpk"))
            for s in rest:
                if s["name"] == m["source_id"] and s["vet"] is None:
                    s["vet"], s["later"] = "catalogue-dropped", "%s %s mJy at %.0f arcsec" % (label, "%.0f" % f if f else "?", float(m["angDist"]))
    for cat, label, factor in (("J/other/PASA/41.3/sourcesm", "RACS-mid", 5.0), ("J/A+A/598/A78/table3", "TGSS", 25.0)):
        for m in core.xmatch([dict(source_id=s["name"], ra=s["ra"], dec=s["dec"]) for s in rest if s["vet"] is None], cat, lobe_arcsec):
            f = num(m.get("Ftot")) or num(m.get("Stotal"))
            for s in rest:
                if s["name"] == m["source_id"] and s["vet"] is None and f and f > factor * s["s"] and float(m["angDist"]) > later_arcsec:
                    s["vet"], s["later"] = "radio-hotspot", "%s %.0f mJy at %.0f arcsec" % (label, f, float(m["angDist"]))
    for s in rest:
        if s["vet"] is None:
            seen_img = [x for x in switchoff_vlass_images(s["ra"], s["dec"]) if x[1] > image_sigma]
            if seen_img:
                s["vet"], s["later"] = "catalogue-dropped", "VLASS images %s" % ", ".join("%s %.0f sigma, %.0f mJy" % x for x in seen_img)
    left = [s for s in gone if s["vet"] is None]
    print("%d NVSS sources above %g mJy north of %g deg, %d compact under %g arcsec more than %g deg from the Galactic plane; without a VLASS component within %g arcsec: %d (%d of "
          "known SIMBAD types, %d within %g deg of a source above %g Jy, %d still shining in RACS-low of 2019-2020 or in holes of the catalogue with no component within %g deg, %d of "
          "them in RACS); unexplained %d" % (len(src), flux_min_mjy, dec_min, len(compact), major_max_arcsec, b_min, radius_arcsec, len(gone),
                                           sum(s["vet"] == "natural-type" for s in gone), sum(s["vet"] == "bright-neighbour" for s in gone), bright_deg, bright_jy,
                                           sum(s["vet"] == "catalogue-dropped" for s in gone), hole_deg, sum(1 for s in gone if s["racs_mjy"] is not None), len(left)))
    for s in sorted(gone, key=lambda s: -s["s"]):
        if s["vet"] != "natural-type":
            print("  NVSS %s: %.0f mJy, %.0f x %.0f arcsec at RA %.4f Dec %+.4f, b %+.1f, RACS %s mJy, %s components within %g deg%s; SIMBAD %s%s; %s" % (
                s["name"], s["s"], s["maj"], s["min_"] or 0, s["ra"], s["dec"], s["b"], s["racs_mjy"], s["around"], hole_deg, ", near " + s["bright"] if s.get("bright") else "",
                s["simbad"], "; " + s["later"] if s.get("later") else "", s["vet"] or "unexplained"))
    json.dump(dict(n=len(src), compact=len(compact), gone=gone), open(core.results("radio_switchoffs.json"), "w"), indent=1)
    if record:
        core.record_run(
            "nvss_vlass_switchoffs", "Radio sources that switched off", "NVSS (VizieR VIII/65); VLASS epoch-1 quick-look components (J/ApJS/255/30) by CDS XMatch; SIMBAD",
            "compact NVSS sources above %g mJy north of %g deg and more than %g deg from the Galactic plane: %d" % (flux_min_mjy, dec_min, b_min, len(compact)),
            os.path.abspath(__file__), "switchoff", len(compact), "sources",
            dict(flux_min_mjy=flux_min_mjy, major_max_arcsec=major_max_arcsec, radius_arcsec=radius_arcsec, dec_min=dec_min, hole_deg=hole_deg, b_min=b_min, bright_jy=bright_jy,
                 bright_deg=bright_deg, racs_dec_max=racs_dec_max),
            "VLASS's component limit of about 1 mJy against a hundredfold fade; SIMBAD's types; the artefact zones of the brightest sources; RACS-low of 2019-2020; the catalogue's own holes",
            "a compact 1.4 GHz source of 1995 with no 3 GHz component in 2019",
            "%d missing of %d, %d of known types, %d beside sources above %g Jy, %d still in RACS-low or in catalogue holes; %d unexplained" % (
                len(gone), len(compact), sum(s["vet"] == "natural-type" for s in gone), sum(s["vet"] == "bright-neighbour" for s in gone), bright_jy,
                sum(s["vet"] == "catalogue-dropped" for s in gone), len(left)),
            [dict(target="nvss:%s" % s["name"], stat=s["s"], unit="mJy at 1.4 GHz, absent from VLASS", vet=s["vet"],
                  note="%.0f x %.0f arcsec; b %+.1f; RACS %s mJy; %s components within %g deg%s; SIMBAD %s%s" % (s["maj"], s["min_"] or 0, s["b"], s["racs_mjy"], s["around"], hole_deg,
                                                                                                     ", near " + s["bright"] if s.get("bright") else "", s["simbad"],
                                                                                                     "; " + s["later"] if s.get("later") else "")) for s in gone])
        print("recorded")


maseretz_DOC = r"""Masers in Earth's transit zone: the 6.7 GHz methanol masers of Pestalozzi et al. (2005, VizieR J/A+A/432/737) by ecliptic latitude, the count within the 0.262 deg half-width of the zone from which Earth is seen to transit the Sun against the mean and scatter of the twenty strips of that width on either side out to 5.5 deg, a bright natural line a sender might choose to sit behind; the masers follow the Galactic plane, which crosses the ecliptic twice and fills the strips beside the zone as it fills the zone.

    python -I seti.py radio_sky maseretz [record] [--half-width-deg 0.262] [--band-deg 5.5]"""


def maseretz_main(record=False, half_width_deg=0.262, band_deg=5.5):
    rows = core.tap("vizier", 'SELECT Name, RAJ2000, DEJ2000, Vlsr, PFlux FROM "J/A+A/432/737/catalog" WHERE RAJ2000 IS NOT NULL')
    ms = [dict(name=str(r["Name"]).strip(), ra=float(r["RAJ2000"]), dec=float(r["DEJ2000"]), v=num(r["Vlsr"]), flux=num(r["PFlux"])) for r in rows]
    lam, beta = science.ecliptic_lon_lat([m["ra"] for m in ms], [m["dec"] for m in ms])
    for m, b in zip(ms, beta):
        m["beta"] = abs(float(b))
    z = science.zone_excess([m["beta"] for m in ms], half_width_deg, band_deg)
    inzone = [m for m in ms if m["beta"] < half_width_deg]
    for m in inzone:
        m["vet"] = "chance-rate" if z["p"] > 0.01 else None
    print("%d methanol masers; in the zone: %d against %.2f +- %.2f beside it (z %+.2f, Poisson p %.3g)" % (len(ms), z["zone"], z["mean"], z["sd"], z["z"], z["p"]))
    for m in sorted(inzone, key=lambda m: m["beta"]):
        print("  %s: ecliptic latitude %+.3f deg, Vlsr %s km/s, %s Jy; %s" % (m["name"], m["beta"], m["v"], m["flux"], m["vet"] or "unexplained"))
    json.dump(dict(n=len(ms), zone=z, masers=inzone), open(core.results("masers_transit_zone.json"), "w"), indent=1)
    if record:
        core.record_run(
            "methanol_masers_transit_zone", "Masers in Earth's transit zone", "Pestalozzi et al. (2005) 6.7 GHz methanol masers (VizieR J/A+A/432/737)", "masers with positions: %d" % len(ms),
            os.path.abspath(__file__), "maseretz", len(ms), "masers", dict(half_width_deg=half_width_deg, band_deg=band_deg),
            "the twenty strips of the zone's width beside it: %.2f +- %.2f" % (z["mean"], z["sd"]), "an excess of masers in the band of sky that sees Earth transit the Sun",
            "%d in the zone (z %+.2f, p %.3g)" % (z["zone"], z["z"], z["p"]),
            [dict(target="maser:%s" % m["name"], stat=round(m["beta"], 4), unit="ecliptic latitude, deg", vet=m["vet"], note="Vlsr %s km/s, %s Jy" % (m["v"], m["flux"])) for m in inzone])
        print("recorded")


def atnf_table(fields, condition, errors=True, timeout=120):
    """ATNF pulsar catalogue rows from its own query form: each field as a (value, error) pair with ERRORS, else a float or a string, None where the catalogue has none."""
    import html
    p = dict(table_top="y", JName="JName", startUserDefined="true", sort_attr="jname", sort_order="asc", condition=condition, ephemeris="short",
             style="Long with errors" if errors else "Short without errors", no_value="*", state="query", version="2.6.1")
    for f in fields:
        p[f] = f
    t = core.get("https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?" + urllib.parse.urlencode(p), timeout=timeout).decode("utf-8", "replace")
    pre = re.search(r"<pre>(.*?)</pre>", t, re.S).group(1)
    rows = []
    for line in html.unescape(re.sub(r"<a [^>]*>[^<]*</a>", " @ ", pre)).replace("\xa0", " ").splitlines():
        tok = line.split()
        if not tok or not tok[0].isdigit():
            continue
        r, k = dict(PSRJ=tok[1]), 2
        if errors and k < len(tok) and tok[k] == "@":
            k += 1
        for f in fields:
            if k >= len(tok):
                r[f] = None
                continue
            v = tok[k]
            if not errors:
                r[f] = None if v == "*" else (core.num(v) if core.num(v) is not None else v)
                k += 1
                continue
            if v == "*":
                r[f], k = None, k + 3
                continue
            e = core.num(tok[k + 1]) if k + 1 < len(tok) and tok[k + 1] != "@" else None
            k += 2 if e is not None else 1
            if k < len(tok) and tok[k] in ("@", "*"):
                k += 1
            r[f] = (core.num(v), e)
        rows.append(r)
    return rows


pbdot_DOC = r"""Binary pulsars decaying at the wrong rate: the binary pulsars of the ATNF catalogue outside globular clusters with a measured orbital decay and the companion's mass, the total mass from the periastron advance or from the companion's mass, sin i and the mass function, the decay general relativity predicts from the masses, period and eccentricity, the Shklovskii term from the proper motion and distance and the Galactic term from a flat rotation curve and the vertical field, and the remainder against zero in units of its error, with the masses and a 20 per cent distance error carried by Monte Carlo; an orbit drained or pumped by something other than gravity would leave a remainder, as would mass lost from a spider companion or a wrong distance.

    python -I seti.py radio_sky pbdot [record] [--sigma 3] [--dist-frac 0.2]"""


def pbdot_main(record=False, sigma=3.0, dist_frac=0.2):
    T_SUN, C, KPC, YR = 4.925490947e-6, 2.99792458e8, 3.0856775814913673e19, 3.15576e7
    rng = np.random.default_rng(3)
    rows = atnf_table(["PB", "PBDOT", "ECC", "M2", "OMDOT", "SINI", "A1", "PMTOT"], "exist(PBDOT) && exist(M2)")
    pos = {r["PSRJ"]: r for r in atnf_table(["DIST", "GL", "GB", "BINCOMP", "ASSOC"], "exist(PBDOT) && exist(M2)", errors=False)}
    for r in rows:
        r.update({k: v for k, v in pos.get(r["PSRJ"], {}).items() if k != "PSRJ"})
        pb, e = r["PB"][0] * 86400, (r["ECC"][0] if r.get("ECC") else 0.0)
        if r.get("OMDOT") and r["OMDOT"][0] and r["OMDOT"][0] > 0.05:
            om = r["OMDOT"][0] * math.pi / 180 / YR
            mt = (om * (1 - e ** 2) / 3 * (pb / (2 * math.pi)) ** (5 / 3)) ** 1.5 / T_SUN
            r["MTOT"] = (mt, 1.5 * mt * (r["OMDOT"][1] or 0) / r["OMDOT"][0])
        elif r.get("SINI") and r.get("A1") and r["SINI"][0]:
            f = r["A1"][0] ** 3 * (2 * math.pi / pb) ** 2 / T_SUN
            mt = math.sqrt((r["M2"][0] * r["SINI"][0]) ** 3 / f)
            r["MTOT"] = (mt, 1.5 * mt * math.hypot((r["M2"][1] or 0) / r["M2"][0], (r["SINI"][1] or 0) / r["SINI"][0]))
        if isinstance(r.get("DIST"), float):
            r["DIST"] = (r["DIST"], None)
        for g in ("GL", "GB"):
            if isinstance(r.get(g), float):
                r[g] = (r[g], None)
    out = []
    for r in rows:
        if isinstance(r.get("ASSOC"), str) and "GC" in r["ASSOC"]:
            continue
        if not all(isinstance(r.get(k), tuple) for k in ("PB", "PBDOT", "M2", "MTOT", "DIST")) or not isinstance(r.get("GL"), tuple):
            continue
        n = 4000
        pb = r["PB"][0] * 86400
        e = r["ECC"][0] if isinstance(r.get("ECC"), tuple) else 0.0
        mt = rng.normal(r["MTOT"][0], r["MTOT"][1] or 0, n)
        m2 = rng.normal(r["M2"][0], r["M2"][1] or 0, n)
        m1 = mt - m2
        d = r["DIST"][0] * (1 + dist_frac * rng.standard_normal(n))
        mu = (r["PMTOT"][0] if isinstance(r.get("PMTOT"), tuple) else 0.0) / 1000 / 3600 * math.pi / 180 / YR
        fe = (1 + 73 / 24 * e ** 2 + 37 / 96 * e ** 4) / (1 - e ** 2) ** 3.5
        gr = -192 * math.pi / 5 * T_SUN ** (5 / 3) * (pb / (2 * math.pi)) ** (-5 / 3) * fe * m1 * m2 * np.abs(mt) ** (-1 / 3)
        shk = pb * mu ** 2 * d * KPC / C
        l, b = math.radians(r["GL"][0]), math.radians(r["GB"][0])
        r0, th0 = 8.178, 236e3
        beta = d / r0 * math.cos(b) - math.cos(l)
        gal_r = -(th0 ** 2 / (C * r0 * KPC)) * math.cos(b) * (math.cos(l) + beta / (math.sin(l) ** 2 + beta ** 2))
        z = np.abs(d * math.sin(b))
        kz = (2.27 * z + 3.68 * (1 - np.exp(-4.31 * z))) * 1e-11
        gal_z = -kz * abs(math.sin(b)) / C
        model = gr + shk + pb * (gal_r + gal_z)
        obs, eobs = r["PBDOT"]
        if abs(obs) > 1e-7:
            # timing packages accept the decay in units of 1e-12 when its magnitude is large
            obs, eobs = obs * 1e-12, (eobs or 0) * 1e-12
        res = obs - np.median(model)
        err = math.sqrt((eobs or 0) ** 2 + float(np.std(model)) ** 2)
        out.append(dict(psr=r["PSRJ"], pb_d=r["PB"][0], obs=obs, gr=float(np.median(gr)), kin=float(np.median(shk + pb * (gal_r + gal_z))), res=res, err=err,
                        z=res / err if err else None, comp=r.get("BINCOMP") if isinstance(r.get("BINCOMP"), str) else None, vet=None))
    off = [o for o in out if o["z"] is not None and abs(o["z"]) > sigma]
    for o in off:
        # main-sequence companions in orbits under a day are the redbacks, whose orbital periods wander (Pletsch and Clark 2015)
        if (o["comp"] or "").startswith("MS") and o["pb_d"] < 1:
            o["vet"] = "spider-orbit"
    print("%d field binary pulsars with a measured decay and both masses; remainder beyond %g sigma: %d" % (len(out), sigma, len(off)))
    for o in sorted(out, key=lambda o: -abs(o["z"] or 0)):
        print("  %-12s Pb %.4f d %-5s observed %.3e, relativity %.3e, kinematic %.3e, remainder %.3e +- %.1e (%.1f sigma)" % (
            o["psr"], o["pb_d"], o["comp"] or "", o["obs"], o["gr"], o["kin"], o["res"], o["err"], o["z"] or 0) + ("; " + o["vet"] if o["vet"] else ""))
    json.dump(out, open(core.results("pulsar_orbital_decay.json"), "w"), indent=1)
    if record:
        core.record_run(
            "atnf_orbital_decay_remainder", "Binary pulsars decaying at the wrong rate", "ATNF pulsar catalogue (its query form, version 2.6.1)",
            "field binary pulsars with a measured orbital decay and both masses (the total from the periastron advance or the mass function): %d" % len(out), os.path.abspath(__file__), "pbdot", len(out), "pulsars",
            dict(sigma=sigma, dist_frac=dist_frac), "general relativity from the masses, the Shklovskii term and the Galactic acceleration, with the masses and distance carried by Monte Carlo",
            "a decay that misses relativity and kinematics beyond %g sigma" % sigma, "%d of %d beyond %g sigma, %d of them redbacks; %d unexplained" % (
                len(off), len(out), sigma, sum(o["vet"] == "spider-orbit" for o in off), sum(o["vet"] is None for o in off)),
            [dict(target="psr:%s" % o["psr"], stat=round(o["z"], 2), unit="sigma", vet=o["vet"],
                  note="observed %.3e, relativity %.3e, kinematic %.3e" % (o["obs"], o["gr"], o["kin"])) for o in off])
        print("recorded")


hostless_DOC = r"""Hostless flat-spectrum radio sources: the CRATES flat-spectrum sources (Healey et al. 2007, VizieR J/ApJS/171/61) with spectral index above -0.5, point-like at 8.4 GHz with an interferometric position (each source's main row, not the components of extended ones), Galactic latitude above 30 deg and declination above -30 deg, matched to AllWISE within 3 arcsec and Pan-STARRS1 within 2 arcsec, and those without either to the deeper CatWISE2020 and unWISE; a flat-spectrum source is a compact, self-absorbed emitter, and in nature it sits in an active galaxy that the infrared or the optical shows, so one with neither is a compact radio source with no galaxy around it, which the classes SIMBAD gives within 5 arcsec are read to explain, and a compact component more than 3 arcsec from its NVSS or SUMSS centroid is a knot or hotspot of a larger radio source whose galaxy lies elsewhere; the rest are looked for in the deeper Legacy Surveys DR10 within 2 arcsec, and those still bare in VLASS within 30 arcsec, where nearly every CRATES source appears, so that a bare position VLASS does not see is the catalogue's position rather than the source's.

    python -I seti.py radio_sky hostless [record] [--alpha-min -0.5] [--b-min 30] [--dec-min -30] [--wise-arcsec 3] [--ps1-arcsec 2] [--offset-arcsec 3]"""


def hostless_main(record=False, alpha_min=-0.5, b_min=30.0, dec_min=-30.0, wise_arcsec=3.0, ps1_arcsec=2.0, offset_arcsec=3.0):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    t = [x for x in core.vizier("J/ApJS/171/61") if x.meta.get("name", "").endswith("table5")][0]
    src = []
    for r in t:
        a = num(r["alpha"])
        ra2, de2 = str(r["RAJ2000"]).strip(), str(r["DEJ2000"]).strip()
        if a is None or a <= alpha_min or str(r["r_S8.4"]).strip() not in ("V", "A", "X") or str(r["Morph"]).strip() != "P" or not ra2 or ra2 in ("--", "nan") or not de2:
            continue
        src.append(dict(name=str(r["CRATES"]).strip(), ra2=ra2, de2=de2, alpha=a, s48=num(r["S4.8"]), s84=num(r["S8.4"]), origin=str(r["r_S8.4"]).strip(), nv_ra=str(r["RA2"]).strip(), nv_de=str(r["DE2"]).strip()))
    c = SkyCoord([s["ra2"] for s in src], [s["de2"] for s in src], unit=(u.hourangle, u.deg))
    g = c.galactic
    for s, ci, gi in zip(src, c, g):
        s["ra"], s["dec"], s["b"] = float(ci.ra.deg), float(ci.dec.deg), float(gi.b.deg)
    sel = [s for s in src if abs(s["b"]) > b_min and s["dec"] > dec_min]
    up = [dict(source_id=k, ra=s["ra"], dec=s["dec"]) for k, s in enumerate(sel)]
    wise = {int(m["source_id"]) for m in core.xmatch(up, "II/328/allwise", wise_arcsec)}
    ps1 = {int(m["source_id"]) for m in core.xmatch(up, "II/349/ps1", ps1_arcsec)}
    rest = [dict(source_id=k, ra=sel[k]["ra"], dec=sel[k]["dec"]) for k in range(len(sel)) if k not in wise and k not in ps1]
    deep = set()
    for cat in ("II/365/catwise", "II/363/unwise"):
        if rest:
            deep |= {int(m["source_id"]) for m in core.xmatch(rest, cat, wise_arcsec)}
    bare = [sel[k] for k in range(len(sel)) if k not in wise and k not in ps1 and k not in deep]
    simbad = {}
    if bare:
        for m in core.xmatch([dict(source_id=k, ra=s["ra"], dec=s["dec"]) for k, s in enumerate(bare)], "simbad", 5.0):
            simbad.setdefault(int(m["source_id"]), []).append((str(m.get("main_id", "")).strip(), str(m.get("main_type", m.get("otype", ""))).strip()))
    for k, s in enumerate(bare):
        s["simbad"] = simbad.get(k, [])
        s["offset"] = None
        if s["nv_ra"] and s["nv_ra"] not in ("--", "nan") and s["nv_de"]:
            s["offset"] = float(SkyCoord(s["nv_ra"], s["nv_de"], unit=(u.hourangle, u.deg)).separation(SkyCoord(s["ra"] * u.deg, s["dec"] * u.deg)).arcsec)
        s["vet"] = "published-classification" if any(t in ("QSO", "BLLac", "Blazar", "AGN", "Sy1", "Sy2", "G", "GinCl", "BClG", "LINER", "QSO_Candidate", "Bla", "BLL", "rG", "RadioG",
                                                            "Blazar_Candidate", "AGN_Candidate", "BLLac_Candidate") for _, t in s["simbad"]) else (
            "radio-hotspot" if s["offset"] is not None and s["offset"] > offset_arcsec else "bright-neighbour" if any(t in ("Star", "*", "PM*", "HighPM*", "**", "V*") for _, t in s["simbad"]) else None)
    for s in [x for x in bare if x["vet"] is None]:
        ls = core.datalab("SELECT type, mag_r, mag_z FROM ls_dr10.tractor WHERE q3c_radial_query(ra, dec, %.6f, %.6f, %.6f) = 't'" % (s["ra"], s["dec"], ps1_arcsec / 3600))
        if ls:
            s["vet"] = "deep-counterpart"
            s["legacy"] = [(r["type"], r["mag_r"], r["mag_z"]) for r in ls][:1]
            continue
        vl = core.tap("vizier", 'SELECT RAJ2000, DEJ2000 FROM "J/ApJS/255/30/comp" WHERE 1 = CONTAINS(POINT(\'ICRS\', RAJ2000, DEJ2000), CIRCLE(\'ICRS\', %.6f, %.6f, %.6f))' % (
            s["ra"], s["dec"], 30 / 3600), timeout=120)
        s["vlass"] = min((core.sep_arcsec(s["ra"], s["dec"], float(r["RAJ2000"]), float(r["DEJ2000"])) for r in vl), default=None)
        if s["vlass"] is None:
            s["vet"] = "artefact"
    left = [s for s in bare if s["vet"] is None]
    print("%d CRATES point-like flat-spectrum sources with 8.4 GHz positions; %d at |b| above %g and declination above %g; with AllWISE %d, with Pan-STARRS1 %d, %d more in CatWISE2020 or unWISE, "
          "with none %d (%d typed as active galaxies by SIMBAD within 5 arcsec, %d more than %g arcsec from their NVSS or SUMSS centroid, %d with a Legacy DR10 counterpart, %d that VLASS does not see "
          "within 30 arcsec); unexplained %d, each with VLASS at its position" % (
              len(src), len(sel), b_min, dec_min, len(wise), len(ps1), len(deep), len(bare), sum(s["vet"] == "published-classification" for s in bare), sum(s["vet"] == "radio-hotspot" for s in bare),
              offset_arcsec, sum(s["vet"] == "deep-counterpart" for s in bare), sum(s["vet"] == "artefact" for s in bare), len(left)))
    for s in bare:
        print("  %s: RA %.5f, Dec %.5f, alpha %.2f, S4.8 %s mJy, 8.4 GHz position from %s, %s arcsec from the NVSS centroid; SIMBAD %s; %s" % (
            s["name"], s["ra"], s["dec"], s["alpha"], s["s48"], s["origin"], "%.1f" % s["offset"] if s["offset"] is not None else "?", s["simbad"], s["vet"] or "unexplained"))
    json.dump(bare, open(core.results("crates_hostless.json"), "w"), indent=1)
    if record:
        core.record_run(
            "crates_hostless_flat_spectrum", "Hostless flat-spectrum radio sources", "CRATES (VizieR J/ApJS/171/61); AllWISE (II/328), Pan-STARRS1 (II/349), CatWISE2020 (II/365) and unWISE (II/363) by CDS XMatch; SIMBAD; Legacy Surveys DR10 (Data Lab); VLASS (J/ApJS/255/30)",
            "flat-spectrum sources with 8.4 GHz positions at |b| above %g and declination above %g: %d" % (b_min, dec_min, len(sel)), os.path.abspath(__file__), "hostless", len(sel), "sources",
            dict(alpha_min=alpha_min, b_min=b_min, dec_min=dec_min, wise_arcsec=wise_arcsec, ps1_arcsec=ps1_arcsec, offset_arcsec=offset_arcsec), "the counterpart rate of the same sources: %d of %d in AllWISE" % (len(wise), len(sel)),
            "a compact flat-spectrum radio source with no host in the infrared or the optical", "%d with no counterpart in AllWISE, Pan-STARRS1, CatWISE2020 or unWISE: %d active galaxies in SIMBAD, %d knots or hotspots off their NVSS centroids, %d with a Legacy DR10 counterpart, %d "
            "at positions VLASS does not see; %d unexplained, each confirmed by VLASS and bare to the Legacy depth" % (
                len(bare), sum(s["vet"] == "published-classification" for s in bare), sum(s["vet"] == "radio-hotspot" for s in bare), sum(s["vet"] == "deep-counterpart" for s in bare),
                sum(s["vet"] == "artefact" for s in bare), len(left)),
            [dict(target="crates:%s" % s["name"], stat=s["alpha"], unit="spectral index", vet=s["vet"], note="S4.8 %s mJy; SIMBAD %s" % (s["s48"], s["simbad"][:2])) for s in bare])
        print("recorded")


hostlesssed_DOC = r"""Radio spectra of the hostless flat-spectrum sources: the compact CRATES sources that the hostless test left without an infrared or optical counterpart, each one's flux densities from TGSS at 150 MHz, GLEAM at 200 MHz, RACS-low at 888 MHz, NVSS at 1.4 GHz, VLASS at 3 GHz, CRATES at 4.8 and 8.4 GHz and the VLBI bands of the Radio Fundamental Catalogue, the spectrum classed as peaked, inverted, flat or steep, VLASS set against the NVSS-to-CRATES interpolation for variability, and the VLBI detection rate and the share of each spectral class set against 200 hosted CRATES sources treated the same way; a young or distant compact galaxy peaks at GHz frequencies and shows on VLBI baselines, while a source with no galaxy around it would have to be something else.

    python -I seti.py radio_sky hostlesssed [record] [--controls 200]"""

HOSTLESS_SED = (("TGSS", "J/A+A/598/A78/table3", 15.0, 0.150, "Stotal", 1e-3), ("GLEAM", "VIII/100/gleamegc", 60.0, 0.200, "Fintwide", 1.0),
                ("RACS", "J/other/PASA/38.58/galcut", 10.0, 0.888, "Ftot", 1e-3), ("NVSS", "VIII/65/nvss", 15.0, 1.4, "S1_4", 1e-3),
                ("VLASS", "J/ApJS/255/30/comp", 3.0, 3.0, "Ftot", 1e-3))


def rfc_index():
    """The RFC sources in a sky tree: a function of a position and a radius in arcsec giving the nearest source within it, or None."""
    from scipy.spatial import cKDTree
    src = rfc_sources()
    ra, de = np.radians([x["ra"] for x in src]), np.radians([x["dec"] for x in src])
    tree = cKDTree(np.stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)], 1))

    def near(ra0, de0, radius_arcsec):
        a, d = math.radians(ra0), math.radians(de0)
        dist, k = tree.query([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)], distance_upper_bound=2 * math.sin(math.radians(radius_arcsec / 3600) / 2))
        return src[k] if np.isfinite(dist) else None
    return near


def hostless_sed_class(pts):
    """A radio spectrum's class from (GHz, Jy) points: peaked when it rises below its highest point and falls above it, inverted when it rises to the top, steep below -0.5, flat otherwise."""
    pts = sorted((f, s) for f, s in pts if s and s > 0)
    if len(pts) < 3:
        return "too few points", None
    lf, ls = np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts])
    k = int(np.argmax(ls))
    lo = np.polyfit(lf[:k + 1], ls[:k + 1], 1)[0] if k >= 1 else None
    hi = np.polyfit(lf[k:], ls[k:], 1)[0] if k <= len(pts) - 2 else None
    if lo is not None and hi is not None and lo > 0.3 and hi < -0.3:
        return "peaked", float(10 ** lf[k])
    if hi is None and lo is not None and lo > 0.3:
        return "inverted", float(10 ** lf[k])
    slope = float(np.polyfit(lf, ls, 1)[0])
    return ("steep" if slope < -0.5 else "flat"), float(10 ** lf[k])


def hostlesssed_main(record=False, controls=200):
    bare = [b for b in json.load(open(core.results("crates_hostless.json"))) if b.get("vet") is None]
    t = [x for x in core.vizier("J/ApJS/171/61") if x.meta.get("name", "").endswith("table5")][0]
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    main = [r for r in t if str(r["Morph"]).strip() == "P" and str(r["r_S8.4"]).strip() in ("V", "A", "X")]
    names = {b["name"] for b in bare}
    rng = np.random.default_rng(37)
    pool = [r for r in main if str(r["CRATES"]).strip() not in names]
    ctrl_rows = [pool[k] for k in rng.choice(len(pool), size=min(controls, len(pool)), replace=False)]
    cc = SkyCoord([str(r["RAJ2000"]) for r in ctrl_rows], [str(r["DEJ2000"]) for r in ctrl_rows], unit=(u.hourangle, u.deg))
    ctrl = [dict(name=str(r["CRATES"]).strip(), ra=float(c.ra.deg), dec=float(c.dec.deg), s48=num(r["S4.8"]), s84=num(r["S8.4"])) for r, c in zip(ctrl_rows, cc)]
    rfc = rfc_index()
    for group in (bare, ctrl):
        for s in group:
            s["vlbi"] = rfc(s["ra"], s["dec"], 1.0)
    for group in (bare, ctrl):
        for s in group:
            s["sed"] = {"CRATES 4.8": (4.8, (s.get("s48") or 0) / 1000.0), "CRATES 8.4": (8.4, (s.get("s84") or 0) / 1000.0)}
            if s["vlbi"]:
                for band, f in s["vlbi"]["flux"].items():
                    s["sed"]["RFC " + band] = ({"S": 2.3, "C": 4.8, "X": 8.6, "U": 15.4, "K": 23.8}[band], f)
        up = [dict(source_id=k, ra=s["ra"], dec=s["dec"]) for k, s in enumerate(group)]
        for name, cat, rad, ghz, col, scale in HOSTLESS_SED:
            for m in core.xmatch(up, cat, rad):
                v = num(m.get(col) if m.get(col) not in (None, "") else m.get(col.replace("_", ".")))
                if v is not None and v > 0:
                    s = group[int(m["source_id"])]
                    key = "%s %g" % (name, ghz)
                    s["sed"][key] = (ghz, max(s["sed"].get(key, (ghz, 0))[1], v * scale))
        for s in group:
            s["cls"], s["peak_ghz"] = hostless_sed_class(list(s["sed"].values()))
            nv, cr, vl = s["sed"].get("NVSS 1.4"), s["sed"].get("CRATES 4.8"), s["sed"].get("VLASS 3")
            s["var"] = None
            if nv and cr and vl and nv[1] > 0 and cr[1] > 0:
                alpha = math.log10(cr[1] / nv[1]) / math.log10(4.8 / 1.4)
                s["var"] = vl[1] / (nv[1] * (3.0 / 1.4) ** alpha)
    det_bare = sum(1 for s in bare if s["vlbi"]) / max(len(bare), 1)
    det_ctrl = sum(1 for s in ctrl if s["vlbi"]) / max(len(ctrl), 1)
    classes = {c: sum(1 for s in bare if s["cls"] == c) for c in ("peaked", "inverted", "flat", "steep", "too few points")}
    cclasses = {c: sum(1 for s in ctrl if s["cls"] == c) for c in ("peaked", "inverted", "flat", "steep", "too few points")}
    from scipy.stats import fisher_exact
    p_peak = float(fisher_exact([[classes["peaked"], len(bare) - classes["peaked"]], [cclasses["peaked"], len(ctrl) - cclasses["peaked"]]], alternative="greater")[1])
    p_flat = float(fisher_exact([[classes["flat"], len(bare) - classes["flat"]], [cclasses["flat"], len(ctrl) - cclasses["flat"]]], alternative="less")[1])
    varied = [s for s in bare if s["var"] is not None and (s["var"] > 2 or s["var"] < 0.5)]
    print("%d hostless sources; VLBI-detected %.2f against %.2f of %d hosted CRATES sources; spectra %s against the hosted sources' %s (peaked share Fisher p %.3g, flat share lower at p %.3g); VLASS off the "
          "NVSS-to-CRATES line by more than a factor 2: %d of %d" % (len(bare), det_bare, det_ctrl, len(ctrl), classes, cclasses, p_peak, p_flat, len(varied), sum(1 for s in bare if s["var"] is not None)))
    for s in sorted(bare, key=lambda s: s["name"]):
        pts = ", ".join("%s %.3g" % (k, v[1]) for k, v in sorted(s["sed"].items(), key=lambda kv: kv[1][0]) if v[1] > 0)
        print("  %s: %s, peak %s GHz, VLASS over the interpolation %s, VLBI %s; %s" % (s["name"], s["cls"], s["peak_ghz"], "%.2f" % s["var"] if s["var"] else "-",
                                                                                     s["vlbi"]["name"] if s["vlbi"] else "none", pts))
    json.dump(bare, open(core.results("hostless_seds.json"), "w"), indent=1, default=str)
    if record:
        core.record_run(
            "crates_hostless_seds", "Radio spectra of the hostless flat-spectrum sources", "CRATES; TGSS ADR1, GLEAM, RACS-low, NVSS, VLASS and AT20G by CDS XMatch; the Radio Fundamental Catalogue rfc_2026c",
            "the hostless sources of crates_hostless_flat_spectrum: %d" % len(bare), os.path.abspath(__file__), "hostlesssed", len(bare), "sources", dict(controls=controls),
            "%d hosted CRATES point sources: VLBI-detected %.2f, spectra %s" % (len(ctrl), det_ctrl, cclasses), "spectra and compactness unlike those of compact radio galaxies",
            "VLBI-detected %.2f against %.2f; spectra %s, the peaked share above the hosted sources' at Fisher p %.3g and the flat share below at p %.3g; %d varied by more than a factor 2 "
            "between surveys" % (det_bare, det_ctrl, classes, p_peak, p_flat, len(varied)),
            [dict(target="crates:%s" % s["name"], stat=s["peak_ghz"], unit="GHz, frequency of the highest point", vet=None,
                  note="%s spectrum, VLBI %s, VLASS over the interpolation %s" % (s["cls"], "detected" if s["vlbi"] else "not detected", "%.2f" % s["var"] if s["var"] else "-")) for s in bare])
        print("recorded")


hostlessnir_DOC = r"""Hosts of the hostless radio sources in the near infrared: the compact CRATES sources the hostless test left bare, against VISTA's VHS DR5 and VIKING DR2 and UKIDSS's LAS DR9 within 2 arcsec, coverage judged by any catalogue source within 1 arcmin, and each K magnitude, or the survey's depth where none is found, turned into a redshift by the K-z relation of radio galaxies (Willott et al. 2003: K = 17.37 + 4.53 log z - 0.31 (log z)^2); a radio galaxy's host follows that relation, so a detection dates the source and a deep non-detection puts it far away; and the HSC-SSP DR3 g, r and z images through the Legacy Survey viewer, the signal within an arcsecond against the same aperture at 200 random places in the cutout.

    python -I seti.py radio_sky hostlessnir [record] [--arcsec 2] [--hsc-pix 0.168]"""

HOSTLESS_NIR = (("VHS DR5", "II/367/vhs_dr5", 18.1), ("VIKING DR2", "II/343/viking2", 19.2), ("UKIDSS LAS DR9", "II/319/las9", 18.2))


def kz_redshift(k):
    """The redshift a radio galaxy's host of K magnitude k has on the K-z relation of Willott et al. (2003), by bisection between z 0.01 and 12."""
    lo, hi = -2.0, math.log10(12.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 17.37 + 4.53 * mid - 0.31 * mid * mid < k:
            lo = mid
        else:
            hi = mid
    return 10 ** (0.5 * (lo + hi))


def hostlessnir_main(record=False, arcsec=2.0, hsc_pix=0.168):
    bare = [b for b in json.load(open(core.results("crates_hostless.json"))) if b.get("vet") is None]
    up = [dict(source_id=k, ra=b["ra"], dec=b["dec"]) for k, b in enumerate(bare)]
    for b in bare:
        b["nir"], b["covered"] = {}, []
    for name, cat, depth in HOSTLESS_NIR:
        try:
            near = core.xmatch(up, cat, 60.0)
        except Exception as e:
            print("%s unavailable: %s" % (name, str(e)[:80]))
            continue
        for m in near:
            b = bare[int(m["source_id"])]
            if name not in b["covered"]:
                b["covered"].append(name)
            if float(m["angDist"]) <= arcsec:
                ks = [k for k in m if re.fullmatch(r"K(s)?(ap3|mag|ap\d|pmag|petmag|AperMag3)?", k)]
                vals = [num(m[k]) for k in ks if num(m[k]) is not None and 5 < num(m[k]) < 25]
                if vals:
                    b["nir"][name] = (float(min(vals)), float(m["angDist"]))
    for b in bare:
        if b["nir"]:
            k = min(v[0] for v in b["nir"].values())
            b["k"], b["z"], b["z_kind"] = k, kz_redshift(k), "estimate"
        elif b["covered"]:
            depth = max(d for n, c, d in HOSTLESS_NIR if n in b["covered"])
            b["k"], b["z"], b["z_kind"] = depth, kz_redshift(depth), "lower limit"
        else:
            b["k"], b["z"], b["z_kind"] = None, None, "not covered"
    # HSC-SSP DR3 through the Legacy Survey viewer: the signal within an arcsecond against the same aperture placed at random in the cutout
    import io
    from astropy.io import fits
    rng = np.random.default_rng(5)
    for b in bare:
        b["hsc"] = None
        raw = core.get("https://www.legacysurvey.org/viewer/fits-cutout?" + urllib.parse.urlencode(
            dict(ra="%.6f" % b["ra"], dec="%.6f" % b["dec"], layer="hsc-dr3", pixscale=hsc_pix, bands="grz", size=101)), timeout=120, soft=True)
        if not raw:
            continue
        cube = np.atleast_3d(fits.open(io.BytesIO(raw))[0].data.astype(float))
        cube = cube if cube.shape[0] == 3 else np.moveaxis(cube, -1, 0)
        if not np.any(np.nan_to_num(cube)):
            continue
        yy, xx = np.mgrid[:101, :101]
        snr = []
        for img in cube:
            img = np.nan_to_num(img)
            ap = np.hypot(xx - 50, yy - 50) * hsc_pix <= 1.0
            rand = [img[np.hypot(xx - x, yy - y) * hsc_pix <= 1.0].sum() for x, y in rng.integers(10, 91, size=(200, 2))]
            snr.append(round(float((img[ap].sum() - np.median(rand)) / (1.4826 * np.median(np.abs(rand - np.median(rand))))), 1))
        b["hsc"] = snr
    det = [b for b in bare if b["z_kind"] == "estimate"]
    lim = [b for b in bare if b["z_kind"] == "lower limit"]
    hsc = [b for b in bare if b["hsc"] is not None]
    print("%d hostless sources: %d with a near-infrared counterpart within %g arcsec, %d covered with none, %d outside the surveys; HSC-SSP DR3 covers %d: %s" % (
        len(bare), len(det), arcsec, len(lim), len(bare) - len(det) - len(lim), len(hsc), "; ".join("%s S/N g r z %s" % (b["name"], b["hsc"]) for b in hsc)))
    for b in sorted(bare, key=lambda b: -(b["z"] or 0)):
        print("  %s: %s; K %s, z %s (%s)" % (b["name"], b["nir"] or ("covered by " + ", ".join(b["covered"]) if b["covered"] else "not covered"), "%.2f" % b["k"] if b["k"] else "-",
                                              "%.1f" % b["z"] if b["z"] else "-", b["z_kind"]))
    json.dump(bare, open(core.results("hostless_nir.json"), "w"), indent=1, default=str)
    if record:
        core.record_run(
            "crates_hostless_nir", "Hosts of the hostless radio sources in the near infrared", "VHS DR5, VIKING DR2 and UKIDSS LAS DR9 by CDS XMatch; the K-z relation of Willott et al. (2003)",
            "the hostless sources of crates_hostless_flat_spectrum: %d" % len(bare), os.path.abspath(__file__), "hostlessnir", len(bare), "sources", dict(arcsec=arcsec),
            "each survey's depth (Ks 18.1, 19.2 and K 18.2) on the K-z relation", "a host in the near infrared, or a depth that puts it far away",
            "%d with a counterpart, redshifts %s; %d covered with none, lower limits %s; %d outside the surveys; %d in HSC-SSP DR3, %d of them detected at S/N 5 in g, r or z" % (
                len(det), ", ".join("%.1f" % b["z"] for b in det), len(lim), ", ".join("%.1f" % b["z"] for b in lim), len(bare) - len(det) - len(lim),
                len(hsc), sum(max(b["hsc"]) >= 5 for b in hsc)),
            [dict(target="crates:%s" % b["name"], stat=round(b["z"], 2) if b["z"] else None, unit="redshift on the K-z relation (%s)" % b["z_kind"], vet=None,
                  note="K %s from %s%s" % ("%.2f" % b["k"] if b["k"] else "-", ", ".join(b["nir"]) or ", ".join(b["covered"]) or "no survey",
                                           "; HSC S/N g r z %s" % b["hsc"] if b["hsc"] is not None else "")) for b in bare])
        print("recorded")


APPROACHES = {
    "hostlessnir": (lambda: core.run_cli(hostlessnir_main), hostlessnir_DOC),
    "hostlesssed": (lambda: core.run_cli(hostlesssed_main), hostlesssed_DOC),
    "hostless": (lambda: core.run_cli(hostless_main), hostless_DOC),
    "pbdot": (lambda: core.run_cli(pbdot_main), pbdot_DOC),
    "maseretz": (lambda: core.run_cli(maseretz_main), maseretz_DOC),
    "switchoff": (lambda: core.run_cli(switchoff_main), switchoff_DOC),
    "glitchsched": (lambda: core.run_cli(glitchsched_main), glitchsched_DOC),
    "psrpairs": (lambda: core.run_cli(psrpairs_main), psrpairs_DOC),
    "psrkick": (lambda: core.run_cli(psrkick_main), psrkick_DOC),
    "psretz": (lambda: core.run_cli(psretz_main), psretz_DOC),
    "coldbdradio": (lambda: core.run_cli(coldbdradio_main), coldbdradio_DOC),
    "psrages": (lambda: core.run_cli(psrages_main), psrages_DOC),
    "maserrest": (lambda: core.run_cli(maserrest_main), maserrest_DOC),
    "hydrogenhz": (lambda: core.run_cli(hydrogenhz_main), hydrogenhz_DOC),
    "wholehz": (lambda: core.run_cli(wholehz_main), wholehz_DOC),
    "frbrhythm": (lambda: core.run_cli(frbrhythm_main), frbrhythm_DOC),
    "rfcnear": (lambda: core.run_cli(rfcnear_main), rfcnear_DOC),
    "methanolhigh": (lambda: core.run_cli(methanolhigh_main), methanolhigh_DOC),
    "glitchsync": (lambda: core.run_cli(glitchsync_main), glitchsync_DOC),
    "etzradio": (lambda: core.run_cli(etzradio_main), etzradio_DOC),
    "msplight": (lambda: core.run_cli(msplight_main), msplight_DOC),
    "coldwdradio": (lambda: core.run_cli(coldwdradio_main), coldwdradio_DOC),
    "frbladder": (lambda: core.run_cli(frbladder_main), frbladder_DOC),
    "frbetz": (lambda: core.run_cli(frbetz_main), frbetz_DOC),
    "frblocal": (lambda: core.run_cli(frblocal_main), frblocal_DOC),
    "psrconst": (lambda: core.run_cli(psrconst_main), psrconst_DOC),
    "bfield": (lambda: core.run_cli(bfield_main), bfield_DOC),
    "daypsr": (lambda: core.run_cli(daypsr_main), daypsr_DOC),
    "copypsr": (lambda: core.run_cli(copypsr_main), copypsr_DOC),
    "psrratio": (lambda: core.run_cli(psrratio_main), psrratio_DOC),
    "frbpsr": (lambda: core.run_cli(frbpsr_main), frbpsr_DOC),
    "boxcar": (lambda: core.run_cli(boxcar_main), boxcar_DOC),
    "nodm": (lambda: core.run_cli(nodm_main), nodm_DOC),
    "breakup": (lambda: core.run_cli(breakup_main), breakup_DOC),
    "overpower": (lambda: core.run_cli(overpower_main), overpower_DOC),
    "gcclock": (lambda: core.run_cli(gcclock_main), gcclock_DOC),
    "yearpsr": (lambda: core.run_cli(yearpsr_main), yearpsr_DOC),
    "frbdm": (lambda: core.run_cli(frbdm_main), frbdm_DOC),
    "frbsimul": (lambda: core.run_cli(frbsimul_main), frbsimul_DOC),
    "frbcopies": (lambda: core.run_cli(frbcopies_main), frbcopies_DOC),
    "spinup": (lambda: core.run_cli(spinup_main), spinup_DOC),
    "psrstar": (lambda: core.run_cli(psrstar_main), psrstar_DOC),
    "psrclock": (lambda: core.run_cli(psrclock_main), psrclock_DOC),
    "frbstars": (lambda: core.run_cli(frbstars_main), frbstars_DOC),
    "gleamx": (gleamx_cli, gleamx_DOC),
    "polceiling": (polceiling_cli, polceiling_DOC),
    "wow": (lambda: core.run_cli(wow_main), wow_DOC),
    "hi4pi": (hi4pi_cli, hi4pi_DOC),
    "hi4pi_blind": (hi4pi_blind_cli, hi4pi_blind_DOC),
    "vlass": (vlass_cli, vlass_DOC),
    "lowfreq": (lowfreq_cli, lowfreq_DOC),
    "sgl": (sgl_cli, sgl_DOC),
    "pulsars": (pulsars_cli, pulsars_DOC),
    "wind": (wind_cli, wind_DOC),
    "callisto": (callisto_cli, callisto_DOC),
    "radar": (radar_cli, radar_DOC),
    "alma": (alma_cli, alma_DOC),
    "lpt": (lpt_cli, lpt_DOC),
}


