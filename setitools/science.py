"""Analysis for seti-tools: rest frames, spectral line searches, radio track searches, trends, light-curve statistics, photometry."""
import io
import json
import math
import os

import numpy as np

from . import core
from .core import quiet_warnings


# --------------------------------------------------------------------------------------------------------------------
# frames: Doppler frames, sites and line-frequency predictions; km/s positive receding

C = 299792.458
L_B = 1.550519768e-8  # TT rate against TCB
CMB_DIPOLE = (369.82, 264.021, 48.253)   # km/s toward galactic (l, b), Planck 2018
# solar motion (km/s, Galactic x, y, z) against the GSR (Schoenrich 2010, Reid and Brunthaler 2020) and the radio LSRK
SUN_VS_GSR = (11.1, 248.5, 7.25)
SUN_VS_LSRK = (20.0, 56.157, 22.765)   # km/s toward galactic (l, b)

SITES = {
    "GBT": (-79.839835, 38.433121, 807.4), "Parkes": (148.263526, -32.998370, 415.0),
    "Paranal": (-70.4045, -24.6272, 2635.0), "LaSilla": (-70.7380, -29.2567, 2400.0),
    "SSO": (149.0661, -31.2770, 1164.0), "APO": (-105.8203, 32.7802, 2788.0), "LCO": (-70.6926, -29.0146, 2380.0),
    "ALMA": (-67.7552, -23.0234, 5058.0), "MWA": (116.6708, -26.7033, 377.0), "ASKAP": (116.6371, -26.6969, 361.0),
    "LOFAR": (6.8689, 52.9153, 15.0), "ATA": (-121.4733, 40.8178, 1043.0),
    "KPNO": (-111.5997, 31.9634, 2120.0), "CFHT": (-155.4681, 19.8253, 4204.0),
}


def location(site):
    """EarthLocation of a named site; geocentre for spacecraft."""
    from astropy.coordinates import EarthLocation
    import astropy.units as u
    if site == "geocentre":
        return EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 0 * u.m)
    lon, lat, h = SITES[site]
    return EarthLocation.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=h * u.m)


def unit_vector(ra_deg, dec_deg):
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])


def direction(ra, dec, pmra=0.0, pmdec=0.0, epoch=2016.0, at=2016.0):
    """Unit vector to a star at year at from position and proper motion (mas/yr) at epoch."""
    dt = at - epoch
    return unit_vector(ra + pmra / 3.6e6 * dt / math.cos(math.radians(dec)), dec + pmdec / 3.6e6 * dt)


def observer_velocity(mjd_utc, site):
    """Barycentric velocity (km/s, shape (3, n)) of a site at UTC MJDs (ERFA)."""
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time
    import astropy.units as u
    t = Time(np.atleast_1d(mjd_utc), format="mjd", scale="utc")
    _, vel = get_body_barycentric_posvel("earth", t, ephemeris="builtin")
    _, gv = location(site).get_gcrs_posvel(t)
    return vel.xyz.to(u.km / u.s).value + gv.xyz.to(u.km / u.s).value


def barycentric_correction(ra, dec, mjd_utc, site):
    """Barycentric correction (km/s) for an observatory-frame radial velocity."""
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    return SkyCoord(ra * u.deg, dec * u.deg).radial_velocity_correction(
        kind="barycentric", obstime=Time(mjd_utc, format="mjd"), location=location(site)).to(u.km / u.s).value


def barycentre_seconds(t, mjdref, ra, dec, scale="tt", site="geocentre", n_grid=60):
    """Arrival times (s since mjdref) moved to the barycentre toward (ra, dec), interpolated on a grid; geocentric error is the spacecraft light time."""
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    t = np.asarray(t, float)
    grid = np.linspace(t.min(), t.max(), n_grid)
    tt = Time(mjdref + grid / 86400.0, format="mjd", scale=scale, location=location(site))
    ltt = tt.light_travel_time(SkyCoord(ra * u.deg, dec * u.deg), kind="barycentric").to(u.s).value
    return t + np.interp(t, grid, ltt)


def ecliptic_lon_lat(ra_deg, dec_deg):
    """Barycentric ecliptic longitude and latitude in degrees of ICRS positions."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra=np.asarray(ra_deg, float) * u.deg, dec=np.asarray(dec_deg, float) * u.deg, frame="icrs").barycentrictrueecliptic
    return np.asarray(c.lon.deg), np.asarray(c.lat.deg)


def zone_excess(beta_deg, half_width_deg=0.262, band_deg=5.5):
    """Count within HALF_WIDTH_DEG of the ecliptic against the strips of that width beside it out to BAND_DEG: zone, strips, mean, sd, z and Poisson p."""
    from scipy.stats import poisson
    b = np.abs(np.asarray(beta_deg, float))
    n = int(round(band_deg / half_width_deg)) - 1
    zone = int(np.sum(b < half_width_deg))
    strips = [int(np.sum((b >= half_width_deg * k) & (b < half_width_deg * (k + 1)))) for k in range(1, n + 1)]
    mean = float(np.mean(strips))
    sd = float(np.std(strips, ddof=1)) if n > 1 else 0.0
    p = float(poisson.sf(zone - 1, mean)) if zone > mean else 1.0 - float(poisson.sf(zone, mean))
    return dict(zone=zone, strips=strips, mean=mean, sd=sd, z=(zone - mean) / sd if sd else 0.0, p=p)


def spectrum_spikes(pts, sigma=4.0, frac=0.02):
    """Bins of a binned spectrum PTS (wavelength, width, value, error) whose value stands SIGMA combined errors, SIGMA bin-to-bin scatters and a FRAC fraction away from the mean of their neighbours."""
    pts = sorted(pts)
    res = [pts[i][2] - 0.5 * (pts[i - 1][2] + pts[i + 1][2]) for i in range(1, len(pts) - 1)]
    scale = 1.4826 * float(np.median(np.abs(res))) if res else 0.0
    out = []
    for i in range(1, len(pts) - 1):
        w, bw, d, e = pts[i]
        dev = res[i - 1]
        err = math.sqrt(e ** 2 + 0.25 * (pts[i - 1][3] ** 2 + pts[i + 1][3] ** 2))
        if abs(dev) > sigma * err and abs(dev) > sigma * scale and abs(dev) > frac * abs(d):
            out.append(dict(um=w, bw=bw, depth=d, dev=dev, err=err, z=dev / err, scale=scale))
    return out


def galactic_unit(l_deg, b_deg):
    """Unit vectors (n, 3) toward Galactic (l, b) in degrees."""
    lr, br = np.radians(np.atleast_1d(l_deg)), np.radians(np.atleast_1d(b_deg))
    return np.c_[np.cos(br) * np.cos(lr), np.cos(br) * np.sin(lr), np.sin(br)]


def _sun_vector(frame):
    if frame == "cmb":
        return CMB_DIPOLE[0] * galactic_unit(CMB_DIPOLE[1], CMB_DIPOLE[2])[0]
    if frame == "gsr":
        return np.array(SUN_VS_GSR, float)
    if frame == "lsrk":
        return SUN_VS_LSRK[0] * galactic_unit(SUN_VS_LSRK[1], SUN_VS_LSRK[2])[0]
    raise ValueError(frame)


def frame_rest_velocity(frame, l_deg, b_deg):
    """Heliocentric radial velocity (km/s) of a source at rest in cmb, gsr or lsrk toward (l, b)."""
    return -(galactic_unit(l_deg, b_deg) @ _sun_vector(frame))


def helio_to_lsrk(v_helio, l_deg, b_deg):
    """LSRK radial velocity (km/s) of a barycentric one toward (l, b), as astropy's LSRK."""
    return np.asarray(v_helio, float) + galactic_unit(l_deg, b_deg) @ _sun_vector("lsrk")


def rest_to_observatory(lam_rest, rv_star, bc):
    """Observatory wavelength of a rest line in a star of barycentric velocity rv_star, given correction bc."""
    return lam_rest * (1.0 + rv_star / C) * (1.0 - bc / C)


def bjd_tdb(mjd_utc, ra, dec, site):
    """BJD (TDB) of UTC MJDs at a site toward (ra, dec)."""
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    t = Time(np.atleast_1d(mjd_utc), format="mjd", scale="utc", location=location(site))
    return (t.tdb + t.light_travel_time(SkyCoord(ra * u.deg, dec * u.deg), kind="barycentric")).jd


def cmb_beta():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = SkyCoord(l=CMB_DIPOLE[1] * u.deg, b=CMB_DIPOLE[2] * u.deg, frame="galactic").icrs
    return CMB_DIPOLE[0] / C * unit_vector(g.ra.deg, g.dec.deg)


def planet_velocity(bjd, period_d, t0_bjd, star_mass_msun):
    """Radial velocity (km/s, receding after transit) and orbital speed of a circular transiting planet."""
    vorb = (2 * math.pi * 6.674e-11 * star_mass_msun * 1.98892e30 / (period_d * 86400.0)) ** (1 / 3) / 1000.0
    phase = ((np.asarray(bjd) - t0_bjd) / period_d) % 1.0
    return vorb * np.sin(2 * math.pi * phase), vorb, phase


def line_tracks(f0, ra, dec, mjd_utc, site, star=None, planet=None, frames=("cmb", "ssb", "star", "planet")):
    """Observatory-frame frequency tracks and km/s half-windows of a line emitted at f0 in the cmb, ssb, star or planet frame."""
    mjd = np.atleast_1d(mjd_utc)
    star = star or {}
    year = 2000.0 + (np.mean(mjd) - 51544.5) / 365.25
    n = direction(ra, dec, star.get("pmra", 0.0), star.get("pmdec", 0.0), 2016.0, year)
    obs = (1.0 + n @ observer_velocity(mjd, site) / C) * (1.0 + L_B)
    out, half = {}, {}
    if "cmb" in frames:
        b = cmb_beta()
        out["cmb"] = f0 / math.sqrt(1 - b @ b) * (1.0 + n @ b) * obs
        half["cmb"] = 0.45
    if "ssb" in frames:
        out["ssb"] = f0 * obs
        half["ssb"] = 0.021
    rv = star.get("rv")
    if rv is not None and np.isfinite(rv):
        plx = star.get("plx", 0.0) or 0.0
        vt = 4.74047 * math.hypot(star.get("pmra", 0.0), star.get("pmdec", 0.0)) / plx if plx > 0 else 0.0
        err = star.get("rv_err") or 1.0
        if "star" in frames:
            g = 1.0 / math.sqrt(1.0 - (rv / C) ** 2 - (vt / C) ** 2)
            out["star"] = f0 / (g * (1.0 + rv / C)) * obs
            half["star"] = 3.0 * err + 0.7
        if planet and "planet" in frames:
            bjd = bjd_tdb(mjd, ra, dec, site)
            vp, vorb, _ = planet_velocity(bjd, planet["P"], planet["T0"], planet["mass"])
            cycles = abs((np.mean(bjd) - planet["T0"]) / planet["P"])
            dph = (planet.get("eT0", 1e-3) + cycles * planet.get("eP", 1e-4)) / planet["P"]
            e_p = math.sqrt(err ** 2 + (vorb * planet.get("e_mass", 0.1 * planet["mass"]) / (3 * planet["mass"])) ** 2
                            + (vorb * 2 * math.pi * dph) ** 2)
            g = 1.0 / np.sqrt(1.0 - ((rv + vp) / C) ** 2 - (vt / C) ** 2)
            out["planet"] = f0 / (g * (1.0 + (rv + vp) / C)) * obs
            half["planet"] = 3.0 * e_p + 1.0
    return out, half


# --------------------------------------------------------------------------------------------------------------------
# spectra: narrow-line detection and vetting on one grid; residual = spectrum / model - 1, dips as the null

def vac_to_air(w):
    """Air wavelength (A) of a vacuum wavelength (Morton 2000)."""
    s2 = (1e4 / np.asarray(w, float)) ** 2
    return w / (1.0 + 0.0000834254 + 0.02406147 / (130.0 - s2) + 0.00015998 / (38.9 - s2))


def air_to_vac(w, iterations=3):
    """Vacuum wavelength (A) of an air wavelength."""
    v = np.asarray(w, float).copy()
    for _ in range(iterations):
        v = v + (w - vac_to_air(v))
    return v


# strong unblended FGK lines at laboratory air wavelengths (A): Fe I, Ca I, Mg I, Na I, Ba II
STRONG_LINES = (4045.812, 4063.594, 4071.738, 4202.029, 4226.728, 4250.119, 4260.474, 4271.760, 4325.762, 4383.545,
                4404.750, 4415.122, 4427.310, 4454.779, 4459.118, 4494.563, 4528.614, 4554.029, 4890.755, 4891.492,
                4920.502, 4957.596, 5167.322, 5172.684, 5183.604, 5269.537, 5328.039, 5371.489, 5397.128, 5405.775,
                5429.696, 5446.917, 5588.749, 5615.644, 5889.950, 5895.924, 6122.217, 6137.692, 6141.713, 6162.173,
                6191.558, 6230.723, 6252.555, 6393.601, 6439.075, 6494.980, 6496.897)


def line_velocity(flux, grid, lines=STRONG_LINES, span=200.0, step=0.05, coarse=1.0, min_lines=8):
    """Velocity (km/s) minimising the mean flux at strong-line cores, coarse then fine; NaN with fewer than min_lines lines."""
    grid = np.asarray(grid, float)
    step_a = grid[1] - grid[0]
    use = []
    for lam in lines:
        lo, hi = lam * (1 - span / C) - 1.0, lam * (1 + span / C) + 1.0
        if lo < grid[0] or hi > grid[-1]:
            continue
        j0, j1 = int((lo - grid[0]) / step_a), int((hi - grid[0]) / step_a) + 1
        if np.isfinite(flux[j0:j1]).mean() > 0.98:
            use.append(lam)
    if len(use) < min_lines:
        return float("nan")
    use = np.array(use)
    f = np.where(np.isfinite(flux), flux, np.nanmedian(flux))

    def best(vs):
        s = [np.mean(np.interp(use * (1.0 + v / C), grid, f)) for v in vs]
        return float(vs[int(np.argmin(s))])
    v0 = best(np.arange(-span, span + 1e-9, coarse))
    return best(np.arange(v0 - 2 * coarse, v0 + 2 * coarse + 1e-9, step))


def to_rest(flux, grid, v):
    """Flux moved from a frame receding at v km/s to the rest frame on the same grid; NaN beyond the data."""
    return np.interp(grid, np.asarray(grid, float) / (1.0 + v / C), flux, left=np.nan, right=np.nan)


def velocity_offset(flux, ref, grid, mask=None, span=6.0, step=0.05, coarse=None):
    """Velocity (km/s) of flux relative to ref by cross-correlation over mask within +-span."""
    m = np.isfinite(flux) & np.isfinite(ref)
    if mask is not None:
        m &= mask
    if m.sum() < 100:
        return 0.0
    lam = np.asarray(grid, float)[m]
    a = flux[m] - flux[m].mean()
    r = np.nan_to_num(ref, nan=1.0)

    def best(vs):
        c = []
        for v in vs:
            b = np.interp(lam, grid * (1.0 + v / C), r)
            c.append(np.dot(a, b - b.mean()))
        return float(vs[int(np.argmax(c))])
    if coarse:
        v0 = best(np.arange(-span, span + 1e-9, coarse))
        return best(np.arange(v0 - coarse, v0 + coarse + 1e-9, step))
    return best(np.arange(-span, span + 1e-9, step))


def line_absorption(target, twins, grid, lam0, R, core=0.25, half=8.0, residual=False):
    """Extra absorption at lam0 beyond a twin model: (z, equivalent width in mA), with the window residual when residual is set."""
    from scipy.optimize import nnls
    nan = (float("nan"), float("nan")) + ((None,) if residual else ())
    step = grid[1] - grid[0]
    j0 = int(round((lam0 - grid[0]) / step))
    h = int(half / step)
    if j0 - h < 0 or j0 + h + 1 > len(grid):
        return nan
    sl = slice(j0 - h, j0 + h + 1)
    y = target[sl]
    B = np.array([t[sl] for t in twins])
    x = (grid[sl] - lam0) / half
    ok = np.isfinite(y) & np.all(np.isfinite(B), axis=0)
    fitm = ok & (np.abs(grid[sl] - lam0) > core)
    if fitm.sum() < 0.5 * len(y) or ok[h - int(0.2 / step):h + int(0.2 / step) + 1].mean() < 0.9:
        return nan
    A = np.nan_to_num(np.vstack([B, np.ones_like(x), -np.ones_like(x), x, -x]).T)
    coef, _ = nnls(A[fitm], y[fitm])
    m0 = A @ coef
    g1 = np.gradient(m0)
    g2 = np.gradient(g1)
    A2 = np.hstack([A, np.vstack([g1, -g1, g2, -g2]).T])
    coef, _ = nnls(A2[fitm], y[fitm])
    res = y - A2 @ coef
    noise = 1.4826 * np.median(np.abs(res[fitm] - np.median(res[fitm])))
    sp = lam0 / R / 2.3548 / step
    k = np.arange(-int(4 * sp) - 1, int(4 * sp) + 2)
    g = np.exp(-0.5 * (k / sp) ** 2)
    seg = res[h + k[0]:h + k[-1] + 1]
    if not np.all(np.isfinite(seg)) or noise <= 0:
        return nan
    amp = np.dot(seg, g) / np.dot(g, g)
    out = float(amp / (noise / np.sqrt(np.dot(g, g)))), float(amp * sp * step * np.sqrt(2 * np.pi) * 1000.0)
    return out + (res,) if residual else out


def combined_line_score(z, weights, null, draws=20000, seed=7):
    """Weighted combined line score and its lower-tail probability among null draws."""
    z, w = np.asarray(z, float), np.asarray(weights, float)
    ok = np.isfinite(z)
    if not ok.any() or len(null) < 50:
        return float("nan"), float("nan")
    z, w = z[ok], w[ok]
    zc = float(np.dot(w, z) / np.sqrt(np.dot(w, w)))
    if len(z) == 1:
        # one line: its rank among the null values
        return zc, float((1 + np.sum(np.asarray(null, float) <= zc)) / (1 + len(null)))
    rng = np.random.default_rng(seed)
    sims = rng.choice(np.asarray(null, float), size=(draws, len(z))) @ w / np.sqrt(np.dot(w, w))
    return zc, float((1 + np.sum(sims <= zc)) / (1 + draws))


def combined_line_threshold(weights, null, p, draws=100000, seed=7):
    """Combined score at lower-tail probability p among null draws."""
    w = np.asarray(weights, float)
    if len(w) == 1:
        return float(np.quantile(np.asarray(null, float), p)) if p * len(null) >= 1 else float(np.min(null))
    rng = np.random.default_rng(seed)
    sims = rng.choice(np.asarray(null, float), size=(draws, len(w))) @ w / np.sqrt(np.dot(w, w))
    return float(np.quantile(sims, p))


def envelope(f, block=500, q=90.0, smooth=5):
    """Running upper envelope: block percentiles, interpolated and smoothed."""
    n = len(f)
    cen, val = [], []
    for s in range(0, n, block):
        x = f[s:s + block]
        x = x[np.isfinite(x)]
        if len(x) > block // 4:
            cen.append(s + block // 2)
            val.append(np.percentile(x, q))
    if len(cen) < 2:
        return np.full(n, np.nan)
    env = np.interp(np.arange(n), cen, val)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        env = np.convolve(np.pad(env, smooth // 2, mode="edge"), k, mode="valid")[:n]
    return env


def block_median(x, block=150):
    """Running baseline from block medians."""
    n = len(x)
    cen, val = [], []
    for s in range(0, n, block):
        v = x[s:s + block]
        v = v[np.isfinite(v)]
        if len(v) > block // 4:
            cen.append(s + block // 2)
            val.append(np.median(v))
    if len(cen) < 2:
        return np.ones(n)
    return np.interp(np.arange(n), cen, val)


def block_mad(res, block=500, floor=2e-3):
    """Blockwise robust scatter (1.4826 MAD), interpolated, floored."""
    n = len(res)
    cen, val = [], []
    for s in range(0, n, block):
        x = res[s:s + block]
        x = x[np.isfinite(x)]
        if len(x) > block // 4:
            cen.append(s + block // 2)
            val.append(1.4826 * np.median(np.abs(x - np.median(x))))
    if not cen:
        return np.full(n, np.inf)
    return np.maximum(np.interp(np.arange(n), cen, val), floor)


def lsf_sigma_px(R, lam=None, step=None, dlog10=None):
    """Gaussian sigma (samples) of an unresolved line on a linear or log10 grid."""
    if dlog10 is not None:
        return 1.0 / (R * dlog10 * np.log(10.0)) / 2.3548
    return lam / R / 2.3548 / step


def matched_filter(res, err, sigma_px):
    """Matched-filter significance and amplitude of an unresolved line at every sample."""
    k = np.arange(-int(4 * sigma_px) - 1, int(4 * sigma_px) + 2)
    g = np.exp(-0.5 * (k / sigma_px) ** 2)
    w = np.where(np.isfinite(res) & np.isfinite(err) & (err > 0), 1.0 / np.maximum(err, 1e-12) ** 2, 0.0)
    r0 = np.where(w > 0, res, 0.0)
    num = np.convolve(r0 * w, g, mode="same")
    den = np.convolve(w, g * g, mode="same")
    return num / np.sqrt(np.maximum(den, 1e-30)), num / np.maximum(den, 1e-30)


def matched_filter_linear(res, err, lam, R, step, chunk=10000):
    """Matched filter on a linear grid with line width growing with wavelength."""
    snr = np.zeros(len(res))
    amp = np.zeros(len(res))
    for s in range(0, len(res), chunk):
        sp = lsf_sigma_px(R, lam[min(s + chunk // 2, len(lam) - 1)], step)
        pad = int(5 * sp) + 2
        lo, hi = max(s - pad, 0), min(s + chunk + pad, len(res))
        a, b = matched_filter(res[lo:hi], err[lo:hi], sp)
        snr[s:s + chunk] = a[s - lo:s - lo + min(chunk, len(res) - s)]
        amp[s:s + chunk] = b[s - lo:s - lo + min(chunk, len(res) - s)]
    return snr, amp


def width_ratio(res, j, sigma_px):
    """Best Gaussian width at sample j in units of the instrumental width; 0.6-1.6 is unresolved."""
    half = int(5 * sigma_px) + 2
    lo, hi = max(j - half, 0), min(j + half + 1, len(res))
    x = np.arange(lo, hi) - j
    y = res[lo:hi]
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return float("nan")
    best, bs = np.inf, float("nan")
    for s in np.arange(0.4, 4.01, 0.1) * sigma_px:
        gg = np.exp(-0.5 * (x / s) ** 2)
        a = np.dot(gg[ok], y[ok]) / np.dot(gg[ok], gg[ok])
        chi = np.sum((y[ok] - a * gg[ok]) ** 2)
        if chi < best:
            best, bs = chi, s
    return float(bs / sigma_px)


def find_peaks(score, thr, sep=10, mask=None, limit=2000):
    """Local maxima above thr, strongest first, at least sep samples apart."""
    ok = score > thr
    if mask is not None:
        ok &= mask
    idx = np.nonzero(ok)[0]
    out = []
    for i in idx[np.argsort(-score[idx])]:
        if all(abs(i - j) >= sep for j in out):
            out.append(int(i))
        if len(out) >= limit:
            break
    return out


def profile_is_spiky(flux, j, sigma_px, base=None):
    """True when the pixels around j are spiky (cosmic ray, hot pixel) rather than a smooth line."""
    half = max(3, int(3 * sigma_px))
    seg = np.asarray(flux[max(j - half, 0):j + half + 1], float)
    if base is None:
        base = np.nanmedian(flux[max(j - 40, 0):j + 40])
    x = seg - base
    k = int(np.nanargmax(x))
    if not (0 < k < len(x) - 1) or x[k] <= 0:
        return True
    if x[k - 1] < 0.35 * x[k] and x[k + 1] < 0.35 * x[k]:
        return True
    for i in range(max(k - 2, 1), min(k + 3, len(x) - 1)):
        if i != k and x[i] < 0.3 * x[k] and (x[i - 1] > 0.5 * x[k] or x[i + 1] > 0.5 * x[k]):
            return True
    return False


def groups(peaks, key, tol, member="visit", min_members=2):
    """Clusters of peaks within tol in key, drawn from at least min_members distinct members."""
    pts = sorted(peaks, key=lambda q: q[key])
    out, cur = [], []
    for q in pts:
        if cur and q[key] - cur[-1][key] > tol:
            out.append(cur)
            cur = []
        cur.append(q)
    if cur:
        out.append(cur)
    return [g for g in out if len({q[member] for q in g}) >= min_members]


def strength(group):
    """Strength of a recurring group: its second-strongest member."""
    v = sorted((q["snr"] for q in group), reverse=True)
    return v[1] if len(v) > 1 else 0.0


def nearest_spectra(N, good, k):
    """Indices of the k most similar rows of N by mean-square difference over shared good samples (torch GPU when present)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = torch.as_tensor(good, dtype=torch.float32, device=dev)
    X = torch.as_tensor(np.nan_to_num(N, nan=0.0), dtype=torch.float32, device=dev) * W
    X2 = X * X
    overlap = W @ W.T
    d = (X2 @ W.T + W @ X2.T - 2.0 * X @ X.T) / overlap.clamp(min=1.0)
    d[overlap < 0.5 * W.sum(1, keepdim=True)] = float("inf")
    d.fill_diagonal_(float("inf"))
    return torch.topk(d, k, largest=False).indices.cpu().numpy()


def fill_gaps(row, ok):
    """Spectrum with bad samples linearly interpolated."""
    if ok.all():
        return row
    x = np.arange(len(row))
    if ok.sum() < 2:
        return np.ones_like(row)
    return np.interp(x, x[ok], row[ok])


def twin_model(target, basis, mask, poly_order=3, derivatives=True):
    """Non-negative fit of comparison spectra plus signed polynomial continuum and optional shift and width terms."""
    from scipy.optimize import nnls
    n = len(target)
    x = np.linspace(-1, 1, n)
    poly = [x ** k * s for k in range(poly_order + 1) for s in (1.0, -1.0)]
    A = np.nan_to_num(np.vstack(list(basis) + poly).T)
    coef, _ = nnls(A[mask], target[mask])
    model = A @ coef
    if derivatives:
        g1 = np.gradient(model)
        g2 = np.gradient(g1)
        g1 /= max(np.abs(g1[mask]).max(), 1e-12)
        g2 /= max(np.abs(g2[mask]).max(), 1e-12)
        A = np.hstack([A, np.vstack([g1, -g1, g2, -g2]).T])
        coef, _ = nnls(A[mask], target[mask])
        model = A @ coef
    weights = coef[:len(basis)]
    template = np.asarray(basis).T @ weights / max(weights.sum(), 1e-12)
    return model, template


def drop_dropouts(V, bad, limit=0.6, grow=2):
    """Bad mask extended over visit dropouts below limit of the visit median, grown by grow samples."""
    Vn = np.where(bad, np.nan, V)
    med = np.nanmedian(Vn, axis=0)
    out = bad.copy()
    for k in range(V.shape[0]):
        r = Vn[k] / med
        low = np.nan_to_num(r / block_median(r), nan=1.0) < limit
        if low.any():
            out[k] |= np.convolve(low.astype(float), np.ones(2 * grow + 1), mode="same") > 0
    return out


def visit_recurrence(lam, V, E, bad, vobs, sigma_px, thr=6.0, min_spread=2.0, tol_px=0.5):
    """Lines recurring at one rest wavelength in several visits but not at one observatory wavelength: dict(pos, neg, terrestrial) of groups."""
    nv = V.shape[0]
    bad = drop_dropouts(V, bad)
    Vn = np.where(bad, np.nan, V)
    med = np.nanmedian(Vn, axis=0)
    frac = np.nanmedian(E / np.maximum(np.abs(V), 1e-12), axis=0)
    dlam = np.gradient(lam)
    tol = tol_px * np.median(dlam)
    found = []
    for k in range(nv):
        ratio = Vn[k] / med
        res = ratio / block_median(ratio) - 1.0
        rerr = np.sqrt((E[k] / np.maximum(np.abs(V[k]), 1e-12)) ** 2 + (1.2533 * frac) ** 2 / nv)
        res = np.where(bad[k] | ~np.isfinite(med), np.nan, res)
        snr, _ = matched_filter(res, rerr, sigma_px)
        for sign in (1, -1):
            for j in find_peaks(sign * snr, thr, sep=8, limit=100):
                if not np.isfinite(vobs[k]):
                    continue
                found.append(dict(visit=k, sign=sign, lam=float(lam[j]), snr=float(sign * snr[j]),
                                  width=width_ratio(sign * res, j, sigma_px), vobs=float(vobs[k]),
                                  lam_obs=float(lam[j] * (1.0 + vobs[k] / C))))
    narrow = [q for q in found if 0.6 <= q["width"] <= 1.6]
    out = dict(pos=[], neg=[], terrestrial=[])
    for sign, key in ((1, "pos"), (-1, "neg")):
        sel = [q for q in narrow if q["sign"] == sign]
        topo = groups(sel, "lam_obs", tol)
        topo_l = [float(np.mean([q["lam_obs"] for q in g])) for g in topo]
        if sign > 0:
            out["terrestrial"] = [round(x, 3) for x in topo_l]
        for g in groups(sel, "lam", tol):
            vo = [q["vobs"] for q in g]
            if np.ptp(vo) < min_spread:
                continue
            lo = float(np.mean([q["lam_obs"] for q in g]))
            if any(abs(lo - x) < 2 * tol for x in topo_l):
                continue
            out[key].append(dict(lam=round(float(np.mean([q["lam"] for q in g])), 4),
                                 visits=sorted({q["visit"] for q in g}), snr=[round(q["snr"], 1) for q in g],
                                 width=[round(q["width"], 2) for q in g], strength=round(strength(g), 2),
                                 vobs_spread=round(float(np.ptp(vo)), 2)))
    return out


def null_summary(pos_strengths, neg_strengths, thresholds=(5, 6, 8, 10, 15, 20)):
    """Counts of emission-like and absorption-like detections above each threshold, and the strongest null."""
    pos, neg = np.asarray(pos_strengths, float), np.asarray(neg_strengths, float)
    return dict(null_max=float(neg.max()) if len(neg) else 0.0,
                counts={t: (int((pos > t).sum()), int((neg > t).sum())) for t in thresholds})


# --------------------------------------------------------------------------------------------------------------------
# radio: Breakthrough Listen HDF5 read by byte range with cached indexes and chunks, and track searches

COARSE = 1048576

# GBT L- and S-band allocations occupied by transmitters (47 CFR 2.106); 1400-1427 MHz is passive
OCCUPIED_MHZ = (
    (960.0, 1215.0, "aeronautical radionavigation (DME, TACAN) and GNSS L5/E5"),
    (1215.0, 1300.0, "GNSS L2/E6 and air-route surveillance radar"),
    (1300.0, 1390.0, "air-route surveillance radar"),
    (1435.0, 1525.0, "aeronautical mobile telemetry"),
    (1525.0, 1559.0, "mobile-satellite downlinks (Inmarsat)"),
    (1559.0, 1610.0, "GNSS L1/E1/G1"),
    (1610.0, 1626.5, "mobile-satellite (Iridium, Globalstar uplink)"),
    (1626.5, 1660.5, "mobile-satellite Earth-to-space (Inmarsat aeronautical and maritime terminals)"),
    (1670.0, 1710.0, "meteorological satellites and radiosondes"),
    (1710.0, 1780.0, "AWS mobile uplink"),
    (1850.0, 1915.0, "PCS mobile uplink"),
    (1920.0, 1930.0, "unlicensed PCS (DECT 6.0 cordless telephones)"),
    (1930.0, 1995.0, "PCS mobile downlink"),
    (2025.0, 2110.0, "space operations Earth-to-space, TDRSS forward links (2025.8-2117.9) and broadcast auxiliary"),
    (2110.0, 2200.0, "AWS mobile downlink and mobile-satellite"),
    (2200.0, 2290.0, "space operations and space research space-to-Earth telemetry, TDRSS return links"),
    (2305.0, 2360.0, "WCS and satellite digital audio radio (SDARS)"),
    (2360.0, 2395.0, "aeronautical mobile telemetry"),
    (2400.0, 2500.0, "ISM (Wi-Fi, Bluetooth, microwave ovens) and Globalstar downlink"),
    (2500.0, 2690.0, "broadband radio service (LTE band 41)"),
)


def occupied(mhz):
    """Occupied allocation holding a frequency (MHz), or None."""
    for lo, hi, what in OCCUPIED_MHZ:
        if lo <= mhz < hi:
            return what
    return None


def _cache(*parts):
    p = os.path.join(core.CACHE, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def header(url, coarse_channels=()):
    """File header and per-coarse-channel chunk byte spans, cached under CACHE/idx."""
    name = url.rstrip("/").split("/")[-1]
    path = _cache("idx", name + ".json")
    meta = json.load(open(path)) if os.path.exists(path) else None
    missing = [k for k in coarse_channels if meta is None or str(k) not in meta["spans"]]
    if meta is not None and not missing:
        return meta
    import fsspec
    import h5py
    f = h5py.File(fsspec.open(url, block_size=2 ** 16).open(), "r")
    d = f["data"]
    if meta is None:
        meta = dict(fch1=float(d.attrs["fch1"]), foff=float(d.attrs["foff"]), tsamp=float(d.attrs["tsamp"]),
                    tstart=float(d.attrs["tstart"]), nt=int(d.shape[0]), nch=int(d.shape[2]), cw=int(d.chunks[2]),
                    src=[float(d.attrs.get("src_raj", np.nan)), float(d.attrs.get("src_dej", np.nan))], spans={})
    cw, nt = meta["cw"], meta["nt"]
    for k in missing:
        j0, j1 = (k * COARSE) // cw, ((k + 1) * COARSE - 1) // cw
        meta["spans"][str(k)] = [j0, j1, [[[i.byte_offset, i.size] for i in
                                           (d.id.get_chunk_info_by_coord((t, 0, j * cw)) for j in range(j0, j1 + 1))]
                                          for t in range(nt)]]
    f.close()
    tmp = path + ".%d.tmp" % os.getpid()
    json.dump(meta, open(tmp, "w"))
    os.replace(tmp, path)
    return meta


class RangeFile:
    """A read-only file over HTTP for h5py: each read is served from 64 KB blocks fetched by byte range through core.get, whose timeouts and retries hold against a stalled connection."""

    BLOCK = 1 << 16

    def __init__(self, url):
        import urllib.request
        self.url, self.pos, self.blocks, self.size = url, 0, {}, 0
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                    self.size = int(r.headers.get("Content-Length", 0))
                break
            except Exception:
                if attempt == 2:
                    raise

    def _block(self, i):
        if i not in self.blocks:
            self.blocks[i] = core.get(self.url, byte_range=(i * self.BLOCK, (i + 1) * self.BLOCK - 1), timeout=120)
        return self.blocks[i]

    def read(self, n=-1):
        if n < 0:
            raise IOError("whole-file reads are not served")
        out, pos = [], self.pos
        while n > 0:
            i, off = divmod(pos, self.BLOCK)
            b = self._block(i)[off:off + n]
            if not b:
                break
            out.append(b)
            pos += len(b)
            n -= len(b)
        self.pos = pos
        return b"".join(out)

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = (self.size or 0) + off
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def flush(self):
        pass

    def close(self):
        self.blocks = {}


def chunk_table(url, threads=32, coarse_channels=None):
    """Every chunk's (byte offset, size) of a file's "data" dataset by time step and chunk column, read from the HDF5 version 1 chunk B-tree directly: the dataset's layout message gives the tree's root, and each level's nodes are fetched THREADS at a time by byte range; with COARSE_CHANNELS the walk descends only into the branches holding those channels' chunks and the table holds zeros elsewhere; the metadata (fch1, foff, tsamp, tstart, nt, nch, cw) comes with it."""
    import struct
    import h5py
    with h5py.File(RangeFile(url), "r") as f:
        d = f["data"]
        meta = dict(fch1=float(d.attrs["fch1"]), foff=float(d.attrs["foff"]), tsamp=float(d.attrs["tsamp"]), tstart=float(d.attrs["tstart"]),
                    nt=int(d.shape[0]), nch=int(d.shape[2]), cw=int(d.chunks[2]),
                    src=[float(d.attrs.get("src_raj", np.nan)), float(d.attrs.get("src_dej", np.nan))], spans={})
        # the hard link's value is the dataset's object header address; an object-info call would walk the whole chunk index
        addr, rank = f.id.links.get_info(b"data").u, len(d.shape)
    cw = meta["cw"]
    wanted = None
    if coarse_channels is not None:
        wanted = set()
        for k in coarse_channels:
            wanted |= set(range((k * COARSE) // cw, ((k + 1) * COARSE - 1) // cw + 1))

    def read(off, n):
        return core.get(url, byte_range=(off, off + n - 1), timeout=120)

    # the dataset's object header: version 1 messages until the layout message (type 8, version 3, chunked) names the tree's root
    def layout(addr):
        h = read(addr, 16)
        if h[0] != 1:
            raise IOError("object header version %d" % h[0])
        nmsg, size = struct.unpack("<H", h[2:4])[0], struct.unpack("<I", h[8:12])[0]
        todo, seen = [(addr + 16, size)], 0
        while todo:
            start, length = todo.pop(0)
            b = read(start, length)
            p = 0
            while p + 8 <= len(b) and seen < nmsg:
                mtype, msize = struct.unpack("<HH", b[p:p + 4])
                body = b[p + 8:p + 8 + msize]
                p += 8 + msize
                seen += 1
                if mtype == 0x10:
                    todo.append(struct.unpack("<QQ", body[:16]))
                elif mtype == 0x08 and body[0] == 3 and body[1] == 2:
                    return struct.unpack("<Q", body[3:11])[0]
        raise IOError("no chunked layout message")
    root = layout(addr)
    klen = 8 + 8 * (rank + 1)
    table = {}

    def node(off):
        """A node's level and entries (key coordinates, chunk size, child address), each entry carrying the next key as its upper bound."""
        b = read(off, 24 + 65 * klen + 64 * 8)
        if b[:4] != b"TREE" or b[4] != 1:
            raise IOError("not a chunk tree node at %d" % off)
        level, used = b[5], struct.unpack("<H", b[6:8])[0]
        keys, children = [], []
        p = 24
        for i in range(used + 1):
            size = struct.unpack("<I", b[p:p + 4])[0]
            keys.append((struct.unpack("<%dQ" % (rank + 1), b[p + 8:p + klen]), size))
            p += klen
            if i < used:
                children.append(struct.unpack("<Q", b[p:p + 8])[0])
                p += 8
        return level, [(keys[i][0], keys[i][1], children[i], keys[i + 1][0]) for i in range(used)]

    def needed(lo, hi):
        """Whether a subtree keyed from chunk coordinate lo up to hi holds a wanted chunk column."""
        if wanted is None:
            return True
        (ta, ja), (tb, jb) = (lo[0], lo[2] // cw), (hi[0], hi[2] // cw)
        if ta < tb:
            return True
        return any(ja <= j < jb for j in wanted)
    level, entries = node(root)
    while True:
        if level == 0:
            for offs, size, child, _ in entries:
                if wanted is None or offs[2] // cw in wanted:
                    table[(offs[0], offs[2] // cw)] = (child, size)
            break
        branch = [e for e in entries if needed(e[0], e[3])]
        children = core.pmap(lambda e: node(e[2]), branch, threads)
        level, entries = children[0][0], [x for _, es in children for x in es]
    nj = -(-meta["nch"] // cw)
    chunks = np.zeros((meta["nt"], nj, 2), np.int64)
    for (t, j), (off, size) in table.items():
        chunks[t, j] = (off, size)
    return meta, chunks


def _decompress(blobs, nt, nj, cw):
    import h5py
    import hdf5plugin
    ff = h5py.File("grid.h5", "w", driver="core", backing_store=False)
    ds = ff.create_dataset("x", shape=(nt, 1, nj * cw), dtype="f4", chunks=(1, 1, cw),
                           **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    for i, b in enumerate(blobs):
        t, j = divmod(i, nj)
        ds.id.write_direct_chunk((t, 0, j * cw), b)
    x = ds[:, 0, :]
    ff.close()
    return x


def coarse_channel(url, k, threads=8):
    """Spectrogram of coarse channel k and its first channel's frequency (MHz)."""
    meta = header(url, (k,))
    j0, j1, sp = meta["spans"][str(k)]
    spans = [tuple(x) for row in sp for x in row]
    name = url.rstrip("/").split("/")[-1]
    path = _cache("blob", "%s.k%d.bin" % (name, k))
    blobs = None
    if os.path.exists(path):
        raw = open(path, "rb").read()
        if len(raw) == sum(s for _, s in spans):
            blobs, pos = [], 0
            for _, s in spans:
                blobs.append(raw[pos:pos + s])
                pos += s
    if blobs is None:
        blobs = core.pmap(lambda q: core.get(url, byte_range=(q[0], q[0] + q[1] - 1), timeout=120), spans, threads)
        tmp = path + ".%d.tmp" % os.getpid()
        with open(tmp, "wb") as fh:
            for b in blobs:
                fh.write(b)
        os.replace(tmp, path)
    x = _decompress(blobs, meta["nt"], j1 - j0 + 1, meta["cw"])
    off = k * COARSE - j0 * meta["cw"]
    return x[:, off:off + COARSE], meta["fch1"] + meta["foff"] * k * COARSE


def cached_channel(name, k):
    """Cached spectrogram and header of coarse channel k, or None when incomplete."""
    pidx = os.path.join(core.CACHE, "idx", name + ".json")
    pblob = os.path.join(core.CACHE, "blob", "%s.k%d.bin" % (name, k))
    if not (os.path.exists(pidx) and os.path.exists(pblob)):
        return None
    meta = json.load(open(pidx))
    if str(k) not in meta["spans"]:
        return None
    j0, j1, sp = meta["spans"][str(k)]
    sizes = [s for row in sp for _, s in row]
    raw = open(pblob, "rb").read()
    if len(raw) != sum(sizes):
        return None
    blobs, pos = [], 0
    for s in sizes:
        blobs.append(raw[pos:pos + s])
        pos += s
    x = _decompress(blobs, meta["nt"], j1 - j0 + 1, meta["cw"])
    off = k * COARSE - j0 * meta["cw"]
    return x[:, off:off + COARSE], meta


def flatten(x, dc_halfwidth=2):
    """Bandpass-flattened, robustly standardised spectrogram with the DC bin zeroed."""
    from scipy.ndimage import median_filter
    base = median_filter(x.mean(0), size=1025, mode="nearest")
    r = x / np.maximum(base, 1e-12)[None, :]
    med = np.median(r, axis=1, keepdims=True)
    mad = np.median(np.abs(r - med), axis=1, keepdims=True) * 1.4826 + 1e-12
    z = ((r - med) / mad).astype(np.float32)
    c = x.shape[1] // 2
    z[:, c - dc_halfwidth:c + dc_halfwidth + 1] = 0.0
    return z


def track_sum(z, base, offsets):
    """Unit-noise sum of z along a track at every channel offset."""
    nt = z.shape[0]
    idx = base[:, None] + offsets[None, :]
    return z[np.arange(nt)[:, None], idx].sum(0) / math.sqrt(nt)


def track_search(z, path_mhz, f_first, foff, half_ch, deltas):
    """Track statistic over +-half_ch, its zero-drift counterpart, and the null from shifted tracks."""
    pos = (np.asarray(path_mhz) - f_first) / foff
    base = np.round(pos).astype(int)
    offs = np.arange(-half_ch, half_ch + 1)
    width = z.shape[1]
    if base.min() - half_ch < 0 or base.max() + half_ch >= width:
        return None
    s = track_sum(z, base, offs)
    flat = track_sum(z, np.full(len(base), int(round(pos.mean()))), offs)
    null = []
    for dl in deltas:
        if abs(int(dl)) < 2 * half_ch + 2000:
            continue
        b = base + int(dl)
        if b.min() - half_ch < 0 or b.max() + half_ch >= width:
            continue
        null.append(track_sum(z, b, offs).max())
    return dict(s=s, flat=flat, null=np.array(null))


def gumbel_p(null_max, x):
    """Upper-tail probability of x under a moment-fitted Gumbel law of null maxima."""
    m, sd = float(np.mean(null_max)), float(np.std(null_max))
    beta = sd * math.sqrt(6.0) / math.pi
    if beta <= 0:
        return 1.0 if x <= m else 0.0
    mu = m - 0.5772156649 * beta
    z = -(x - mu) / beta
    # exp(z) overflows far below the null maxima; p is then 1
    if z > 700:
        return 1.0
    return -math.expm1(-math.exp(z))


# --------------------------------------------------------------------------------------------------------------------
# timeseries: robust trends of irregular light curves (GPU when present) and folded-profile shape statistics

def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def theil_sen(x, y):
    """Median of all pairwise slopes."""
    import torch
    dev = _device()
    xt = torch.as_tensor(np.asarray(x, np.float64) - np.mean(x), dtype=torch.float32, device=dev)
    yt = torch.as_tensor(np.asarray(y, np.float64), dtype=torch.float32, device=dev)
    i, j = torch.triu_indices(len(xt), len(xt), 1, device=dev)
    dx = xt[j] - xt[i]
    keep = dx.abs() > 1e-6
    return float(((yt[j] - yt[i])[keep] / dx[keep]).median())


def slope(t, y, per=100.0, nboot=200, seed=1):
    """Theil-Sen slope per per units of t with a bootstrap error."""
    import torch
    dev = _device()
    x = (np.asarray(t, np.float64) - np.mean(t)) / per
    s = theil_sen(x, y)
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(np.asarray(y, np.float64), dtype=torch.float32, device=dev)
    n = len(xt)
    g = torch.Generator(device=dev).manual_seed(seed)
    i, j = torch.triu_indices(n, n, 1, device=dev)
    bs = []
    for _ in range(nboot):
        k = torch.randint(0, n, (n,), generator=g, device=dev)
        dx = xt[k][j] - xt[k][i]
        keep = dx.abs() > 1e-6
        bs.append(float(((yt[k][j] - yt[k][i])[keep] / dx[keep]).median()))
    return s, float(np.std(bs))


def per_group_slopes(t, y, labels, per=100.0, min_n=30, min_span=20.0):
    """Theil-Sen slope within each label with enough points and span."""
    out = {}
    for name in set(labels):
        k = labels == name
        if k.sum() >= min_n and np.ptp(t[k]) >= min_span:
            out[str(name)] = theil_sen((t[k] - t[k].mean()) / per, y[k])
    return out


def series_coherence(diffs, spreads, floor=0.05):
    """(common slope, error, chi-square p) of per-series differential slopes weighted by comparison spread."""
    from scipy.stats import chi2
    d = np.asarray(diffs, float)
    w = 1.0 / np.maximum(np.asarray(spreads, float), floor) ** 2
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan")
    D = float(np.sum(d * w) / np.sum(w))
    return D, float(1.0 / np.sqrt(np.sum(w))), float(chi2.sf(np.sum((d - D) ** 2 * w), len(d) - 1))


def differential_trend(target, comps, per=100.0, min_points=50, min_span=50.0, spread_floor=0.0):
    """Target slope minus the comparisons' median in units of their spread and the bootstrap error, with per-series slopes and sign agreement."""
    t, y, lab = target
    if len(t) < min_points or np.ptp(t) < min_span:
        return None
    s, e = slope(t, y, per)
    cs = [slope(c[0], c[1], per)[0] for c in comps if len(c[0]) >= min_points and np.ptp(c[0]) >= min_span]
    if len(cs) < 3:
        return dict(slope=s, err=e, n=len(t), comps=cs, note="fewer than 3 comparisons")
    med = float(np.median(cs))
    spread = float(1.4826 * np.median(np.abs(np.array(cs) - med)))
    diff = s - med
    series = per_group_slopes(t, y, lab, per)
    agree = (sum(1 for v in series.values() if v * diff > 0) / len(series)) if series else None
    return dict(slope=s, err=e, n=int(len(t)), comps=cs, comp_median=med, comp_spread=spread, diff=diff,
                sigma=diff / max(np.hypot(spread, e), spread_floor, 1e-3), series=series, series_agreeing=agree)


# folded-profile shape statistics of periodic events


def trapezoid(ph, depth, dur, ingress, centre=0.0):
    x = np.abs(ph - centre)
    y = np.zeros_like(ph)
    full = x <= dur / 2 - ingress
    edge = (x > dur / 2 - ingress) & (x < dur / 2)
    y[full] = -depth
    y[edge] = -depth * (dur / 2 - x[edge]) / max(ingress, 1e-9)
    return y


def profile(t, f, P, t0, dur_d, nb=60, adapt=True):
    """Binned folded profile over +-1.5 durations; fewer, fuller bins for few events unless adapt is off."""
    ph = ((t - t0) / P + 0.5) % 1.0 - 0.5
    w = 3.0 * dur_d / P
    sel = np.abs(ph) < w / 2
    if adapt:
        nb = int(min(nb, max(10, sel.sum() // 6)))
    edges = np.linspace(-w / 2, w / 2, nb + 1)
    k = np.digitize(ph[sel], edges) - 1
    prof = np.array([np.median(f[sel][k == i]) - 1.0 if np.sum(k == i) >= 2 else np.nan for i in range(nb)])
    err = np.array([max(1.2533 * np.std(f[sel][k == i]), 1e-5) / np.sqrt(np.sum(k == i)) if np.sum(k == i) >= 2 else np.nan
                    for i in range(nb)])
    return 0.5 * (edges[1:] + edges[:-1]) * P, prof, err


def fit_trapezoid(x, y, e, dur_d):
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
    best = None
    for dur in np.linspace(0.4, 1.6, 25) * dur_d:
        for ing in np.linspace(0.02, 0.5, 13) * dur:
            for c in np.linspace(-0.2, 0.2, 9) * dur_d:
                m = trapezoid(x, 1.0, dur, ing, c)
                a = np.sum(m[ok] * y[ok] / e[ok] ** 2) / max(np.sum(m[ok] ** 2 / e[ok] ** 2), 1e-30)
                chi = np.sum((y[ok] - a * m[ok]) ** 2 / e[ok] ** 2)
                if best is None or chi < best[0]:
                    best = (chi, a, dur, ing, c)
    return best, int(ok.sum())


def shape_stats(t, f, P, t0, dur, cadence_d):
    """Folded-profile statistics: trapezoid fit, S/N, asymmetry, halves, odd-even, secondary depth, ingress fraction; None when sparse."""
    from scipy.ndimage import median_filter
    ph = ((t - t0) / P + 0.5) % 1.0 - 0.5
    intr = np.abs(ph * P) < 1.5 * dur
    base = np.interp(t, t[~intr], median_filter(f[~intr], size=max(int(3 * dur / cadence_d), 5), mode="nearest")) \
        if (~intr).sum() > 100 else np.ones_like(f)
    g = f / base
    x, y, e = profile(t, g, P, t0, dur)
    best, n = fit_trapezoid(x, y, e, dur)
    if best is None or n < 8:
        return None
    chi_flat = float(np.nansum((y / e) ** 2))
    snr = float(np.sqrt(max(chi_flat - best[0], 0)))
    red = best[0] / max(n - 5, 1)
    mid =np.abs(x - best[4]) < best[2] / 2
    asym = float(np.nansum(y[mid & (x > best[4])]) - np.nansum(y[mid & (x < best[4])])) / max(abs(np.nansum(y[mid])), 1e-12)
    dips = 0
    yy = np.where(np.isfinite(y), y, 0.0)
    for i in range(1, len(yy) - 1):
        if mid[i] and yy[i] < yy[i - 1] and yy[i] < yy[i + 1] and yy[i] < -3 * e[i] and \
                min(yy[i - 1], yy[i + 1]) - yy[i] > 2 * e[i]:
            dips += 1
    half = t < np.median(t)
    nbx = len(x)
    _, y1, _ = profile(t[half], g[half], P, t0, dur, nbx, adapt=False)
    _, y2, _ = profile(t[~half], g[~half], P, t0, dur, nbx, adapt=False)
    ep = np.round((t - t0) / P).astype(int)
    _, yo, _ = profile(t[ep % 2 == 1], g[ep % 2 == 1], P, t0, dur, nbx, adapt=False)
    _, ye, _ = profile(t[ep % 2 == 0], g[ep % 2 == 0], P, t0, dur, nbx, adapt=False)

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > min(10, nbx // 2) and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else float("nan")
    # secondary eclipse: deepest binned dip at phase 0.5
    ph2 = np.abs(np.abs(ph) - 0.5) * P < dur / 2
    sec = float(1.0 - np.median(g[ph2])) if ph2.sum() > 10 else float("nan")
    return dict(period=P, dur_h=dur * 24, snr=round(snr, 1),
                red_chi=round(float(red), 2), asym=round(asym, 3), dips=dips, stable=round(corr(y1, y2), 3),
                odd_even=round(corr(yo, ye), 3), secondary=sec, depth=float(best[1]),
                ingress_frac=round(float(best[3] / best[2]), 3))


def flash_events(t, f, cadence_d, quiet=4, window=31, sigma=5.0, max_len=3):
    """Brightenings of at most max_len cadences above sigma, quiet for quiet cadences each side: [(time, peak, cadences)] and the scatter."""
    from scipy.ndimage import median_filter
    t, f = np.asarray(t, float), np.asarray(f, float)
    if len(f) < 4 * window:
        return [], float("nan")
    r = f / median_filter(f, size=window, mode="nearest") - 1.0
    sig = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    out, seen = [], set()
    for i in np.nonzero(r > sigma * sig)[0]:
        if i in seen:
            continue
        j = i
        while j + 1 < len(r) and r[j + 1] > sigma * sig and t[j + 1] - t[j] < 1.5 * cadence_d:
            j += 1
        seen.update(range(i, j + 1))
        if j - i >= max_len or i < quiet + 1 or j > len(r) - quiet - 2:
            continue
        before, after = r[i - quiet - 1:i - 1], r[j + 2:j + quiet + 2]
        if np.all(np.abs(before) < 2 * sig) and np.all(np.abs(after) < 2 * sig) and                 t[i] - t[i - quiet - 1] < (quiet + 2) * cadence_d and t[j + quiet + 1] - t[j] < (quiet + 2) * cadence_d:
            k = i + int(np.argmax(r[i:j + 1]))
            out.append((float(t[k]), float(r[k] / sig), int(j - i + 1)))
    return out, sig


def phase_cluster(times, good_t, period, t0, width_d, rng, ntrial=400):
    """Most event times in one phase window at the period and its false-alarm probability against times drawn from good_t."""
    w = width_d / period

    def best(ts):
        ph = np.sort(((np.asarray(ts) - t0) / period) % 1.0)
        if len(ph) < 2:
            return len(ph)
        ext = np.concatenate([ph, ph + 1.0])
        j = np.searchsorted(ext, ext[:len(ph)] + w, side="right")
        return int(np.max(j - np.arange(len(ph))))
    n0 = best(times)
    null = [best(rng.choice(good_t, size=len(times), replace=False)) for _ in range(ntrial)]
    return n0, (1 + sum(x >= n0 for x in null)) / (ntrial + 1.0)


# --------------------------------------------------------------------------------------------------------------------
# photometry: forced SPHEREx photometry with the spectral-wavelength map, and binned spectral ratios

SPHEREX_PIX_SR = (6.15 / 206264.806) ** 2
SPHEREX_SIGMA_PX = np.sqrt(4.38 / (4.0 * np.pi))
# defect bits; source and halo bits are kept
SPHEREX_BAD = sum(1 << b for b in (0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 15, 17, 19, 22, 23, 24, 26))


def spherex_point(cutout_bytes, ra, dec, radius_px=3.5):
    """Flux density (mJy), error and wavelength (um) of a point source in a SPHEREx cutout: Gaussian PSF plus background on unflagged pixels."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    quiet_warnings()
    with fits.open(io.BytesIO(cutout_bytes)) as h:
        hdr = h["IMAGE"].header
        img = h["IMAGE"].data.astype(float) - h["ZODI"].data.astype(float)
        var = h["VARIANCE"].data.astype(float)
        flg = np.asarray(h["FLAGS"].data).astype(np.int64)
        x, y = WCS(hdr).world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        sw = WCS(hdr, fobj=h, key="W")
        sw.sip = None
        lam = sw.pixel_to_world(x, y)[0].to(u.micrometer).value
    return spherex_fit(img, var, flg, float(x), float(y), lam, radius_px)


def spherex_point_pixels(px, radius_px=3.5, core=1.5):
    """spherex_point on a pixel block; None when a defect flag lies within core pixels of the source."""
    from astropy import log
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u
    quiet_warnings()
    log.setLevel("WARNING")  # the wavelength WCS carries SIP keywords without a -SIP type
    sw = WCS(px["header"], fobj=fits.HDUList([fits.PrimaryHDU(), px["wave"]]), key="W")
    sw.sip = None
    lam = sw.pixel_to_world(px["x0"] + px["x"], px["y0"] + px["y"])[0].to(u.micrometer).value
    img = px["IMAGE"].astype(float) - px["ZODI"].astype(float)
    return spherex_fit(img, px["VARIANCE"].astype(float), px["FLAGS"].astype(np.int64), px["x"], px["y"], lam,
                       radius_px, core)


def spherex_fit(img, var, flg, x, y, lam, radius_px=3.5, core=0.0):
    """Gaussian PSF plus background fit within radius_px: wavelength, flux density and error (mJy); None when the fit is unsupported."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    m = (r2 < radius_px ** 2) & ((flg & SPHEREX_BAD) == 0) & np.isfinite(img) & np.isfinite(var) & (var > 0)
    if m.sum() < 12 or not (2 <= x < nx - 2 and 2 <= y < ny - 2):
        return None
    if core and np.any(((flg & SPHEREX_BAD) != 0) & (r2 <= core ** 2)):
        return None
    psf = np.exp(-0.5 * r2 / SPHEREX_SIGMA_PX ** 2)
    psf /= psf.sum()
    A = np.vstack([psf[m], np.ones(m.sum())]).T
    w = 1.0 / var[m]
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (img[m] * w))
    return dict(wave=float(lam), flux=float(coef[0] * SPHEREX_PIX_SR * 1e9), err=float(np.sqrt(cov[0, 0]) * SPHEREX_PIX_SR * 1e9))


def binned_ratio(target, controls, edges, norm_range=(1.0, 1.6)):
    """Target over mean control spectrum in wavelength bins, each normalised over norm_range: centres, ratio, error."""
    def binned(s):
        w = np.array([p["wave"] for p in s])
        f = np.array([p["flux"] for p in s])
        cen, val, err = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = (w >= lo) & (w < hi)
            if k.sum() >= 3:
                cen.append(0.5 * (lo + hi))
                val.append(np.median(f[k]))
                err.append(1.25 * np.std(f[k]) / np.sqrt(k.sum()))
        cen, val, err = np.array(cen), np.array(val), np.array(err)
        nm = np.median(val[(cen > norm_range[0]) & (cen < norm_range[1])])
        return cen, val / nm, err / nm
    tc, tv, te = binned(target)
    cs = [binned(c) for c in controls]
    cm = np.mean([np.interp(tc, c[0], c[1]) for c in cs], axis=0)
    ce = np.sqrt(np.sum([np.interp(tc, c[0], c[2]) ** 2 for c in cs], axis=0)) / len(cs)
    ratio = tv / cm
    return tc, ratio, ratio * np.hypot(te / tv, ce / cm)
