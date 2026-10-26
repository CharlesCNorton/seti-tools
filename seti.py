"""Command line of seti-tools: search families, the coverage store and the self-test."""
import math
import os
import sys
import tempfile
import time
import traceback

# one math-library thread per process, so the process cap bounds CPU load
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

for _p in (os.environ.get("SETITOOLS", ""), os.path.dirname(os.path.abspath(__file__))):
    if _p and os.path.isdir(os.path.join(_p, "setitools")):
        sys.path.insert(0, _p)
        break

from setitools import science  # noqa: E402

CHECKS = []


def check(name):
    def wrap(fn):
        def run():
            t0 = time.time()
            try:
                msg = fn() or ""
                CHECKS.append((name, True, msg, time.time() - t0))
            except Exception as e:
                CHECKS.append((name, False, "%s: %s" % (type(e).__name__, e), time.time() - t0))
                traceback.print_exc()
        run.__name__ = fn.__name__
        return run
    return wrap


def synthetic_star(n=6000, nv=6, seed=0, line_at=3000, line_amp=0.0, line_visits=(1, 4), topo_at=None, topo_amp=0.0,
                   sigma=1.5):
    """Synthetic visits of one star: absorption features, noise, and optional rest-frame and observatory-frame lines."""
    rng = np.random.default_rng(seed)
    dlog = 6e-6
    lam = 10 ** (np.log10(15200.0) + dlog * np.arange(n))
    base = 1.0 - 0.3 * np.exp(-0.5 * ((np.arange(n)[:, None] - rng.uniform(0, n, 60)[None, :]) / 3.0) ** 2).sum(1)
    vobs = np.linspace(-25.0, 25.0, nv)
    V, E = [], []
    x = np.arange(n)
    for k in range(nv):
        f = base * (1.0 + (line_amp if k in line_visits else 0.0) * np.exp(-0.5 * ((x - line_at) / sigma) ** 2))
        if topo_at is not None:
            # an observatory-frame line moves in the rest frame from visit to visit
            shift = -vobs[k] / science.C / (dlog * math.log(10.0))
            f = f * (1.0 + topo_amp * np.exp(-0.5 * ((x - topo_at - shift) / sigma) ** 2))
        e = np.full(n, 0.004)
        V.append(1000.0 * (f + rng.normal(0, 0.004, n)))
        E.append(1000.0 * e)
    return lam, np.array(V), np.array(E), np.zeros((nv, n), bool), vobs, dlog


@check("spectra: matched filter recovers an unresolved line at its width")
def t_matched():
    rng = np.random.default_rng(1)
    n, sp = 4000, 2.0
    res = rng.normal(0, 0.01, n)
    res += 0.06 * np.exp(-0.5 * ((np.arange(n) - 1700) / sp) ** 2)
    snr, amp = science.matched_filter(res, np.full(n, 0.01), sp)
    j = int(np.argmax(snr))
    w = science.width_ratio(res, j, sp)
    assert abs(j - 1700) <= 1 and snr[j] > 8 and 0.7 < w < 1.4, (j, snr[j], w)
    return "peak %d, S/N %.1f, width ratio %.2f, amplitude %.3f" % (j, snr[j], w, amp[j])


@check("spectra: a single hot pixel is spiky, a resolved line is not")
def t_spiky():
    f = np.ones(200)
    f[100] = 3.0
    assert science.profile_is_spiky(f, 100, 2.0, base=1.0)
    g = 1.0 + np.exp(-0.5 * ((np.arange(200) - 100) / 2.0) ** 2)
    assert not science.profile_is_spiky(g, 100, 2.0, base=1.0)


@check("spectra: LSF width on log and linear grids")
def t_lsf():
    a = science.lsf_sigma_px(22500, dlog10=6e-6)
    b = science.lsf_sigma_px(115000, 5000.0, 0.01)
    assert 1.2 < a < 1.5 and 1.7 < b < 1.9, (a, b)
    return "APOGEE %.2f px, HARPS %.2f px" % (a, b)


@check("spectra: visit recurrence keeps an intermittent rest-frame line and rejects an observatory-frame one")
def t_recurrence():
    sp = science.lsf_sigma_px(22500, dlog10=6e-6)
    lam, V, E, bad, vobs, _ = synthetic_star(line_at=2500, line_amp=0.06, topo_at=4200, topo_amp=0.06, sigma=sp)
    out = science.visit_recurrence(lam, V, E, bad, vobs, sp)
    pos = [g["lam"] for g in out["pos"]]
    assert any(abs(x - lam[2500]) < 0.5 for x in pos), pos
    assert not any(abs(x - lam[4200]) < 3.0 for x in pos), pos
    assert out["terrestrial"], out
    return "rest-frame groups %s, terrestrial %s" % (pos, out["terrestrial"])


@check("spectra: dropouts in several visits do not turn the others into emission")
def t_dropout():
    sp = science.lsf_sigma_px(22500, dlog10=6e-6)
    lam, V, E, bad, vobs, _ = synthetic_star(nv=7, line_amp=0.0, sigma=sp)
    for k, s in ((0, 3000), (4, 3004), (5, 3001)):
        V[k, s:s + 8] *= 0.33
    bad[1:3, 2990:3020] = True
    out = science.visit_recurrence(lam, V, E, bad, vobs, sp)
    assert not any(abs(g["lam"] - lam[3003]) < 3.0 for g in out["pos"]), out["pos"]
    return "emission groups near the dropouts: none"


@check("spectra: twin model cannot absorb narrow emission")
def t_twin():
    rng = np.random.default_rng(2)
    n = 800
    x = np.arange(n)
    feats = rng.uniform(0, n, 30)
    def star(s):
        return 1 - 0.2 * np.exp(-0.5 * ((x[:, None] - feats[None, :] - s) / 3.0) ** 2).sum(1)
    basis = np.array([star(0.0) + rng.normal(0, 0.002, n) for _ in range(5)])
    target = star(0.0) + 0.05 * np.exp(-0.5 * ((x - 400) / 1.5) ** 2) + rng.normal(0, 0.002, n)
    model, _ = science.twin_model(target, basis, np.ones(n, bool))
    res = target / model - 1
    assert res[400] > 0.03, res[400]
    return "residual at the line %.3f" % res[400]


@check("spectra: fixed-line absorption against twins, and the weighted multi-line score")
def t_absorption():
    rng = np.random.default_rng(5)
    grid = np.round(np.arange(4200.0, 4300.0, 0.01), 2)
    feats = rng.uniform(4200, 4300, 150)

    def star(depth_scale):
        f = np.ones(len(grid))
        for c in feats:
            f *= 1 - depth_scale * 0.3 * np.exp(-0.5 * ((grid - c) / 0.05) ** 2)
        return f
    twins = [star(1.0 + 0.02 * k) + rng.normal(0, 0.003, len(grid)) for k in range(6)]
    lines = (4220.0, 4250.0, 4280.0)
    target = star(1.01) + rng.normal(0, 0.003, len(grid))
    sp = 4250.0 / 115000.0 / 2.3548
    for c in lines:
        target *= 1 - 0.03 * np.exp(-0.5 * ((grid - c) / sp) ** 2)
    z = [science.line_absorption(target, twins, grid, c, 115000.0)[0] for c in lines]
    centres = rng.uniform(4210, 4290, 300)
    null = np.array([science.line_absorption(target, twins, grid, c, 115000.0)[0] for c in centres
                     if all(abs(c - x) > 1 for x in lines)])
    zc, p = science.combined_line_score(z, [1.0, 1.0, 1.0], null[np.isfinite(null)])
    assert all(v < -4 for v in z) and p < 0.01, (z, zc, p)
    return "line scores %s, combined %.1f, p %.4f" % ([round(v, 1) for v in z], zc, p)


@check("frames: barycentric correction and Solar System frame track")
def t_frames():
    bc = float(science.barycentric_correction(83.82, -5.39, 60000.0, "Paranal"))
    assert abs(bc) < 31.0, bc
    tr, half = science.line_tracks(1420.405751768, 83.82, -5.39, np.array([60000.0, 60000.01]), "GBT",
                                  star=dict(rv=20.0, rv_err=0.5, plx=50.0, pmra=10.0, pmdec=-5.0))
    df = (tr["ssb"] - 1420.405751768) / 1420.405751768 * science.C
    assert np.all(np.abs(df) < 31.0) and set(tr) == {"cmb", "ssb", "star"}, (df, tr.keys())
    return "Paranal correction %.3f km/s; GBT SSB offset %.3f km/s" % (bc, df[0])


@check("radio: track search recovers an injected drifting tone")
def t_track():
    rng = np.random.default_rng(3)
    nt, nch = 16, 40000
    z = rng.normal(0, 1, (nt, nch)).astype(np.float32)
    path_ch = 20000 + 3.0 * np.arange(nt)
    z[np.arange(nt), path_ch.astype(int)] += 2.5
    foff, f_first = 2.8e-6, 1420.0
    deltas = np.arange(-15000, 15001, 500)
    out = science.track_search(z, f_first + foff * path_ch, f_first, foff, 3, deltas)
    p = science.gumbel_p(out["null"], out["s"].max())
    assert out["s"].max() > 7 and p < 1e-3 and out["flat"].max() < out["s"].max(), (out["s"].max(), p)
    return "track %.1f sigma, zero-drift %.1f, p %.1e" % (out["s"].max(), out["flat"].max(), p)


@check("timeseries: Theil-Sen slope and differential trend")
def t_trend():
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(1890, 1990, 400))
    lab = np.where(t < 1940, "A", "B")

    def curve(s):
        return t, 10.0 + s * (t - 1940) / 100.0 + rng.normal(0, 0.15, len(t)), lab
    r = science.differential_trend(curve(0.30), [curve(0.0) for _ in range(6)])
    assert abs(r["diff"] - 0.30) < 0.06 and r["sigma"] > 3, r
    return "difference %.3f mag/century, %.1f sigma, device %s" % (r["diff"], r["sigma"], science._device())


@check("photometry: binned spectral ratio")
def t_ratio():
    w = np.linspace(0.75, 5.0, 300)
    tgt = [dict(wave=x, flux=1.0 + (0.1 if x > 4 else 0.0), err=0.01) for x in w]
    ctl = [dict(wave=x, flux=1.0, err=0.01) for x in w]
    c, r, e = science.binned_ratio(tgt, [ctl, ctl], np.linspace(0.75, 5.0, 18))
    assert abs(r[c > 4.2].mean() - 1.1) < 0.01 and abs(r[c < 3].mean() - 1.0) < 0.01, r
    return "ratio above 4.2 um %.3f" % r[c > 4.2].mean()


@check("coverage: record a run, read a target's card and coverage, vets stored as codes")
def t_coverage():
    from setitools import core
    path = os.path.join(tempfile.mkdtemp(), "coverage.db")
    db = core.connect(path)
    sel = core.ids_selection(["gaia_dr3:1", "gaia_dr3:2", "gaia_dr3:3"])
    core.record_search("test_run", "Test approach", archive="test", selection=sel, code=__file__, db=db,
                           n_targets=3, n_unit="stars")
    core.record_targets("test_run", [dict(target="gaia_dr3:1", stat=2.5, p=0.3),
                                         dict(target="gaia_dr3:2", vet="cosmic-ray"),
                                         dict(target="gaia_dr3:3", vet="X-ray source (eRASS1); binary (RUWE 2.1, NSS 0)",
                                              note="5 pc")], db=db)
    card = core.card("gaia_dr3:2", db)
    row = db.execute("SELECT vet, note FROM coverage WHERE target = 'gaia_dr3:3'").fetchone()
    assert core.covered("gaia_dr3:1", "Test approach", db) == ["test_run"] and card[0][7] == "cosmic-ray", card
    assert row == ("active-star; binary", "X-ray source (eRASS1); binary (RUWE 2.1, NSS 0); 5 pc"), row
    assert not core.covered("gaia_dr3:4", db=db)
    try:
        core.record_targets("test_run", [dict(target="gaia_dr3:4", vet="an explanation with no code")], db=db)
        raise AssertionError("an explanation without a code was recorded")
    except KeyError:
        pass
    for t in ("gaia_dr3:-1", "star:--"):
        try:
            core.record_targets("test_run", [dict(target=t, stat=1.0)], db=db)
            raise AssertionError("the identifier %s was recorded" % t)
        except ValueError:
            pass
    db.close()
    return ("three targets recorded and read back; a worded vet stored as its codes; an uncoded one and identifiers that "
            "stand for missing ones refused")


def synthetic_plate(nfib=40, npix=4000, seed=0, sigma=1.0):
    """Synthetic sky-fibre plate: Gaussian noise of known variance and a sky model with thirty airglow lines."""
    rng = np.random.default_rng(seed)
    lam = 10 ** (3.5563 + 1e-4 * np.arange(npix))
    cont = 1.0 + 0.3 * np.sin(np.arange(npix) / 700.0)
    sky = np.tile(cont, (nfib, 1))
    glow = rng.choice(np.arange(60, npix - 60), 30, replace=False)
    for j in glow:
        sky[:, j - 1:j + 2] += 8.0
    return dict(lam=lam, sky=np.arange(nfib), flux=rng.normal(0, sigma, (nfib, npix)),
                ivar=np.full((nfib, npix), 1.0 / sigma ** 2), skym=sky), glow


@check("end to end, sky fibres: unresolved lines recovered; shared, airglow, masked and absorption features set aside")
def t_skyfibre():
    from setitools import lasers as L
    k = np.arange(-4, 5)
    w = np.exp(-0.5 * (k / L.skyfibre_LSF_PX) ** 2)
    gain = math.sqrt(float(np.sum(w ** 2)))
    clean = lambda p: p[0] > 0 and not p[5]  # noqa: E731
    # uninjected plate: the false-alarm rate of clean peaks per fibre
    d0, glow = synthetic_plate(seed=11)
    false = sum(1 for p in L.skyfibre_detect(d0, common=False, shared_min=1)[0] if clean(p))
    rng = np.random.default_rng(12)
    rec = {1: [0, 0], 2: [0, 0]}
    for trial in range(3):
        d, glow = synthetic_plate(seed=20 + trial)
        _, on_line, _ = L.skyfibre_detect(d, common=False)
        ok = np.nonzero(~on_line)[0]
        ok = ok[(ok > 10) & (ok < len(d["lam"]) - 10)]
        where = [(i, int(rng.choice(ok))) for i in range(len(d["sky"]))]
        for i, j in where:
            d["flux"][i][j - 4:j + 5] += 10.0 / gain * w        # S/N 10 in the matched filter
        for rule in (1, 2):
            peaks = L.skyfibre_detect(d, common=False, shared_min=rule)[0]
            for i, j in where:
                rec[rule][1] += 1
                rec[rule][0] += any(clean(p) and p[1] == i and abs(p[2] - j) <= 1 for p in peaks)
    # hard negatives: a line in six fibres, on an airglow line, on a masked pixel, and an absorption dip; a one-pixel spike passes the width test
    d, glow = synthetic_plate(seed=30)
    j_sh, j_sp, j_gl, j_m, j_dip = 900, 1700, int(glow[0]), 2600, 3300
    for i in range(6):
        d["flux"][i][j_sh - 4:j_sh + 5] += 10.0 / gain * w
    d["flux"][10][j_sp] += 25.0
    d["flux"][11][j_gl - 4:j_gl + 5] += 10.0 / gain * w
    d["flux"][12][j_m - 4:j_m + 5] += 10.0 / gain * w
    d["ivar"][12][j_m] = 0.0
    d["flux"][13][j_dip - 4:j_dip + 5] -= 10.0 / gain * w
    peaks = L.skyfibre_detect(d, common=False, shared_min=1)[0]
    near = lambda i, j: [p for p in peaks if p[1] == i and abs(p[2] - j) <= 1]  # noqa: E731
    assert all(near(i, j_sh) and all("shared" in p[5] for p in near(i, j_sh)) for i in range(6)), "shared line kept"
    spike = [p for p in near(10, j_sp) if p[0] > 0]
    assert not any(clean(p) for p in near(11, j_gl)), near(11, j_gl)
    assert not any(clean(p) for p in near(12, j_m)), near(12, j_m)
    assert not any(clean(p) for p in near(13, j_dip)) and any(p[0] < 0 for p in near(13, j_dip)), near(13, j_dip)
    # two strong lines at one redshift mark a galaxy; a lone line or [S II] pair does not
    z = 0.3
    gal = L.skyfibre_lines(sorted([6562.80 * (1 + z), 5006.84 * (1 + z)]))
    sii = L.skyfibre_lines(sorted([6716.44 * (1 + z), 6730.82 * (1 + z)]))
    assert gal and abs(gal[0] - z) < 1e-3 and L.skyfibre_lines([7000.0]) is None and sii is None, (gal, sii)
    r1, r2 = rec[1][0] / rec[1][1], rec[2][0] / rec[2][1]
    assert r1 > 0.7 and r2 > 0.9 and false <= 1, (r1, r2, false)
    return ("S/N 10 lines recovered %.2f (one-fibre rule) and %.2f (two-fibre); %d clean peaks in %d noise fibres; "
            "a one-pixel spike %s" % (r1, r2, false, len(d0["sky"]),
                                      "kept at S/N %.0f, width %.2f px" % (spike[0][3], spike[0][4]) if spike and
                                      clean(spike[0]) else "set aside"))


@check("end to end, Kepler glints: periodic flashes recovered and phase-locked, flare-like and random ones not")
def t_glints():
    """Isolated-flash completeness is bounded by the quiet-neighbour rule: 0.954^8 = 0.69 in white noise."""
    from setitools import flashes as F
    rng = np.random.default_rng(40)
    t = np.arange(0.0, 60.0, F.kepler_CAD)
    q = np.zeros(len(t), int)

    def curve():
        return 1.0 + 0.002 * np.sin(2 * np.pi * t / 7.3) + rng.normal(0, 780e-6, len(t))
    sig = 780e-6
    P, t0 = 3.3, 0.71
    f = curve()
    inj = [t0 + n * P for n in range(int((t[-1] - t0) / P) + 1)]
    for ti in inj:
        f[int(round((ti - t[0]) / F.kepler_CAD))] += 10 * sig
    found, good = F.kepler_flashes(t, f, q)
    got = [x for x in found if min(abs(x[0] - ti) for ti in inj) < 2 * F.kepler_CAD]
    n0, fap = F.kepler_phase_cluster([x[0] for x in got], good, P, t0, rng, ntrial=300)
    # the same number of flashes at random times
    g = curve()
    rand = rng.uniform(t[0], t[-1], len(inj))
    for ti in rand:
        g[int(round((ti - t[0]) / F.kepler_CAD))] += 10 * sig
    found_r, good_r = F.kepler_flashes(t, g, q)
    n0_r, fap_r = F.kepler_phase_cluster([x[0] for x in found_r], good_r, P, t0, rng, ntrial=300)
    # a six-cadence flare and a flash beside a dip are rejected; pure noise gives none
    h = curve()
    h[20000:20006] += 10 * sig
    h[40000] += 10 * sig
    h[40002:40005] -= 5 * sig
    found_h, _ = F.kepler_flashes(t, h, q)
    bad = [x for x in found_h if abs(x[0] - t[20003]) < 0.01 or abs(x[0] - t[40000]) < 0.01]
    noise, _ = F.kepler_flashes(t, curve(), q)
    # completeness over isolated flashes in fresh noise, against the quiet-neighbour bound
    hits = n_inj = 0
    for trial in range(4):
        c = curve()
        idx = rng.choice(np.arange(100, len(t) - 100, 40), 200, replace=False)
        c[idx] += 10 * sig
        fl, _ = F.kepler_flashes(t, c, q)
        ft = np.array([x[0] for x in fl])
        hits += sum(1 for i in idx if len(ft) and np.min(np.abs(ft - t[i])) < 2 * F.kepler_CAD)
        n_inj += len(idx)
    comp = hits / n_inj
    assert 0.6 < comp < 0.8 and fap < 0.01 and fap_r > 0.05 and not bad and len(noise) <= 2, (
        comp, fap, fap_r, bad, len(noise))
    return ("%d of %d periodic flashes found, phase FAP %.3f; random flashes FAP %.2f; completeness for 10 sigma "
            "flashes %.2f against 0.69 from the quiet-neighbour rule; %d flashes in pure noise" % (
                len(got), len(inj), fap, fap_r, comp, len(noise)))


@check("end to end, hyperbolic meteor streams: a coherent stream found, a speed-incoherent group and the background not")
def t_meteors():
    from setitools import solar_system as SS
    rng = np.random.default_rng(50)

    def iso(n):
        x = rng.normal(size=(n, 3))
        return x / np.linalg.norm(x, axis=1)[:, None]

    def around(axis, n, deg):
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        x = a[None, :] + np.radians(deg) * rng.normal(size=(n, 3))
        return x / np.linalg.norm(x, axis=1)[:, None]
    nb = 20000
    xyz_b, v_b = iso(nb), np.abs(rng.normal(10.0, 5.0, nb))
    axis = (0.3, -0.8, 0.52)
    for label, v_s in (("coherent", rng.normal(12.0, 0.4, 16)), ("incoherent", rng.uniform(1.0, 40.0, 16))):
        xyz = np.vstack([xyz_b, around(axis, 16, 0.25)])
        v = np.concatenate([v_b, v_s])
        _, _, p = SS.meteors_pvalues(xyz, v)
        hit = int(np.sum(p[nb:] < 1e-4))
        if label == "coherent":
            groups = SS.meteors_groups(xyz, v, p < 1e-4)
            big = max((len([k for k in g if k >= nb]) for g in groups), default=0)
            coherent = (hit, big)
        else:
            incoherent = hit
    _, _, p0 = SS.meteors_pvalues(xyz_b, v_b)
    false = int(np.sum(p0 < 1e-4))
    assert coherent[0] >= 12 and coherent[1] >= 12 and incoherent <= 3 and false <= 10, (coherent, incoherent, false)
    return "coherent stream: %d of 16 members below p 1e-4, %d in one group; speed-incoherent group: %d; background: %d of %d" % (
        coherent[0], coherent[1], incoherent, false, nb)


@check("ultraviolet: a pointing belongs to the star whose proper-motion track passes through it")
def t_owner():
    from setitools import ultraviolet as uv
    star = dict(ra=269.452, dec=4.693, pmra=-801.6, pmdec=10362.4)           # Barnard's star
    comp = dict(star, ra=star["ra"] + 10 / 3600.0)                              # a companion 10 arcsec east
    mjd = 58700.0
    yrs = (mjd - 51544.5) / 365.25
    ra = star["ra"] + star["pmra"] / 3.6e6 * yrs / math.cos(math.radians(star["dec"]))
    dec = star["dec"] + star["pmdec"] / 3.6e6 * yrs
    d_now = float(uv.uv_track_dist(star, [ra], [dec], [mjd])[0])
    d_cat = float(uv.uv_track_dist(star, [star["ra"]], [star["dec"]], [mjd])[0])
    d_comp = float(uv.uv_track_dist(comp, [ra], [dec], [mjd])[0])
    owner = uv.uv_owner({"star": d_now, "companion": d_comp}, 3.0)
    assert d_now < 0.01 and d_cat < 0.01 and 9.5 < d_comp < 10.5 and owner == "star", (d_now, d_cat, d_comp, owner)
    groups = uv.uv_groups([1.0, 1.01, 1.02, 2.0], [0.015] * 4)
    assert groups == [[0, 1, 2], [3]], groups
    return "epoch and catalogue positions %.3f and %.3f arcsec from the track, companion %.1f" % (d_now, d_cat, d_comp)


@check("package: every docstring is one line")
def t_docstrings():
    import ast
    import glob
    root = os.path.dirname(os.path.abspath(__file__))
    files = (sorted(glob.glob(os.path.join(root, "setitools", "*.py"))) + [os.path.abspath(__file__)] +
             sorted(glob.glob(os.path.join(root, "papers", "*", "*.py"))))
    bad, n = [], 0
    for f in files:
        for node in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                d = ast.get_docstring(node, clean=False)
                if d is not None:
                    n += 1
                    if chr(10) in d.strip():
                        bad.append("%s:%d %s" % (os.path.relpath(f, root), node.body[0].lineno, getattr(node, "name", "module")))
    assert not bad, bad[:10]
    return "%d docstrings in %d files" % (n, len(files))


def online():
    from setitools import core

    @check("online: Gaia")
    def a():
        return "%d row" % len(core.gaia("SELECT TOP 1 source_id FROM gaiadr3.gaia_source"))

    @check("online: SIMBAD")
    def b():
        return "%d row" % len(core.simbad("SELECT TOP 1 main_id FROM basic"))

    @check("online: ESO")
    def c():
        return "%d spectra near HD 10700" % len(core.eso_spectra(26.017, -15.937, "HARPS"))

    @check("online: IRSA")
    def d():
        return "%d row" % len(core.irsa("SELECT TOP 1 designation FROM allwise_p3as_psd"))

    @check("online: VizieR TAP")
    def e():
        from setitools import core
        return "%d row" % len(core.tap("vizier", 'SELECT TOP 1 * FROM "I/239/hip_main"'))

    for fn in (a, b, c, d, e):
        fn()


def selftest():
    for fn in (t_matched, t_spiky, t_lsf, t_recurrence, t_dropout, t_twin, t_absorption, t_frames, t_track, t_trend,
               t_ratio, t_coverage, t_owner, t_skyfibre, t_glints, t_meteors, t_docstrings):
        fn()
    if "--online" in sys.argv:
        online()
    print()
    for name, ok, msg, dt in CHECKS:
        print("%-4s %-80s %5.1fs  %s" % ("ok" if ok else "FAIL", name, dt, msg))
    sys.exit(0 if all(r[1] for r in CHECKS) else 1)


def main():
    argv = list(sys.argv)
    if len(argv) > 1 and argv[1] in ("selftest", "store"):
        from setitools.core import quiet_warnings
        quiet_warnings()
        sys.argv = [argv[0]] + argv[2:]
        if argv[1] == "selftest":
            return selftest()
        from setitools import core
        return core.main()
    from setitools import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
