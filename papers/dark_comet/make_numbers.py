"""Numbers, tables and figures of the 2016 TA56 paper from the darkcomet refit, thermal, robustness, covariance, precovery, field, locus, census and forecast results and the Yarkovsky screen; Table S1 as CSV and the precovery cutouts as PDF beside this file."""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.argv = sys.argv[:1]
from setitools import core  # noqa: E402
from setitools import solar_system as ss  # noqa: E402

DESIG = "2016 TA56"
NIGHT = "20070914"


def results(name):
    """Result file from data/, else the results directory."""
    p = os.path.join(HERE, "data", name)
    return p if os.path.exists(p) else core.results(name)


def darkcomet(name):
    """darkcomet result file from data/, else the darkcomet directory."""
    p = os.path.join(HERE, "data", DESIG.replace(" ", "_"), name)
    return json.load(open(p if os.path.exists(p) else os.path.join(ss.darkcomet_dir(DESIG), name)))


def load():
    d = os.path.join(HERE, "data", DESIG.replace(" ", "_"))
    if not os.path.exists(os.path.join(d, "refit.json")):
        d = ss.darkcomet_dir(DESIG)
    return json.load(open(os.path.join(d, "refit.json"))), json.load(open(os.path.join(d, "thermal.json")))


def year(jd):
    return 2000.0 + (jd - 2451545.0) / 365.25


def fits():
    rf, th = load()
    print("== fits")
    for name, f in rf["fits"].items():
        print("%-32s %3d obs %3d kept chi2 %6.1f/%3d rms %.2f%s%s" % (
            name, f["n_obs"], f["n_kept"], f["chi2"], f["dof"], f["rms"],
            " A2 %+.3e +- %.2e (%.1f sigma)" % (f["A2"], f["A2_sigma"], f["A2"] / f["A2_sigma"]) if f["A2"] else "",
            " A1 %+.3e +- %.2e (%.1f sigma)" % (f["A1"], f["A1_sigma"], f["A1"] / f["A1_sigma"]) if f.get("A1") else ""))
    for app, x in rf["left_out"].items():
        print("%s left out: %d positions, along %+.2f +- %.2f, cross %+.2f +- %.2f arcsec (prediction %.2f, %.2f); %.2f arcsec/min"
              % (app, x["n"], x["along"], x["along_sd"], x["cross"], x["cross_sd"], x["pred_along"], x["pred_cross"], x["rate"]))
    print("NEATM: D %.2f +- %.2f km, pV %.3f, eta %.1f, chi2 %.1f for %d fluxes; eta scan %s" % (
        th["D_km"], th["D_err"], th["pv"], th["eta"], th["chi2"], th["n_w2"],
        ", ".join("%.1f: D %.2f pV %.3f chi2 %.1f" % (s["eta"], s["D_km"], s["pv"], s["chi2"]) for s in th["scan"])))


def screen():
    """Published dark comets in the screen and how many the transverse screen flags."""
    d = json.load(open(results("yarkovsky_sign.json")))
    known = [x for x in d if x.get("known") == "dark comet"]
    gen = [x for x in known if x["xi"] and x["xi"] > 1]
    typ = [x for x in known if x["xi_typ"] and x["xi_typ"] > 1]
    print("== screen: %d published dark comets carry a transverse term; %d exceed sunlight at 500 kg/m^3 and albedo 0.5 (%s), "
          "%d at 2000 kg/m^3 and albedo 0.14 (%s); the others: %s" % (
              len(known), len(gen), ", ".join(x["name"] for x in gen), len(typ), ", ".join(x["name"] for x in typ),
              "; ".join("%s xi_typ %.3f, D>=%.0f m, A1 %s sigma, A3 %s sigma" % (
                  x["name"], x["xi_typ"], x["d_min_m"], "%.1f" % x["A1_snr"] if x["A1_snr"] is not None else "-",
                  "%.1f" % x["A3_snr"] if x["A3_snr"] is not None else "-") for x in known if x not in typ)))


def spins():
    """Obliquities of the significant drifts with DAMIT poles, and the efficiencies of all significant drifts at typical assumptions."""
    import math
    d = json.load(open(results("yarkovsky_sign.json")))
    sig = [x for x in d if x["A2_snr"] is not None and abs(x["A2_snr"]) >= 3]
    print("== spins")
    for x in sig:
        if x["cos_obl"]:
            print("%-28s A2 %+.2e (%.1f sigma) obliquities %s, %s" % (x["name"], x["A2"], x["A2_snr"],
                  ", ".join("%.0f" % math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in x["cos_obl"]), x["verdict"]))
    xt = sorted(x["xi_typ"] for x in sig if x["xi_typ"] is not None)
    q = [xt[int(round(p * (len(xt) - 1)))] for p in (0.16, 0.5, 0.84, 0.95)]
    print("%d significant terms: xi at 2000 kg/m^3 and albedo 0.14, 16/50/84/95 per cent %.3f %.3f %.3f %.3f; %d above one" % (
        len(xt), *q, sum(v > 1 for v in xt)))


def covariance():
    """A2 under the night-level covariance, and the error the precovery night would have to share to lose it."""
    b = darkcomet("blockcov.json")
    print("== block covariance: %d positions on %d nights, s %.2f, tau %.3f (%.3f-%.3f)" % (b["n_kept"], b["n_nights"], b["s_hat"], b["tau_hat"], *b["tau_95"]))
    for k in ("site_fit", "at_estimate", "at_upper_tau", "without_decisive"):
        print("  %-18s A2 %+.3e +- %.2e (%.1f sigma)" % (k, b[k]["A2"], b[k]["A2_sigma"], abs(b[k]["A2"]) / b[k]["A2_sigma"]))
    print("  shared error of the precovery night for 5 and 3 sigma: %.2f, %.2f arcsec" % (b["shared_error_for_5_sigma"], b["shared_error_for_3_sigma"]))


def locus():
    """The positions the later apparitions allow in 2007, what the frames show there, and the figure of the flux along them."""
    lo = darkcomet("locus_%s.json" % NIGHT)
    c = darkcomet("census_%s.json" % NIGHT)
    L = c["locus"]
    print("== locus: without 2007 A2 %+.2e +- %.2e; with A1, A1 %+.2e +- %.2e; per 1e-12 of A2 E %+.2f N %+.2f arcsec; A1 ellipse %.2f x %.2f at PA %.0f"
          % (lo["rest_A2"]["A2"], lo["rest_A2"]["A2_sigma"], lo["rest_A1A2"]["A1"], lo["rest_A1A2"]["A1_sigma"], *lo["arcsec_per_1e-12"],
             lo["a1a2_ellipse"]["major"], lo["a1a2_ellipse"]["minor"], lo["a1a2_ellipse"]["pa"]))
    print("  " + json.dumps({k: v for k, v in L.items() if k not in ("grid", "flux", "n_frames", "map", "frames")}))
    print(r"\begin{tikzpicture}")
    print(r"\begin{axis}[width=0.95\textwidth,height=0.36\textwidth,xmin=-16,xmax=12,ymin=-0.2,ymax=1.1,xlabel={$A_2$ ($10^{-12}$\,au\,d$^{-2}$)},"
          r"ylabel={relative flux},tick label style={font=\footnotesize},label style={font=\small}]")
    print(r"\addplot[black,thick] coordinates {%s};" % " ".join("(%.2f,%.3f)" % (g * 1e12, f) for g, f in zip(L["grid"], L["flux"]) if f is not None))
    s3 = 3 * L["null_sd"]
    print(r"\addplot[black!50,densely dotted] coordinates {(-16,%.3f) (12,%.3f)};" % (s3, s3))
    print(r"\addplot[black!50,densely dotted] coordinates {(-16,%.3f) (12,%.3f)};" % (-s3, -s3))
    print(r"\addplot[black!50,dashed] coordinates {(0,-0.2) (0,1.1)};")
    print(r"\end{axis}")
    print(r"\end{tikzpicture}")


def census():
    """Moving sources in the whole 2007 frames, the injected copies recovered, and the chance tracks."""
    c = darkcomet("census_%s.json" % NIGHT)
    m = c["movers"]
    print("== census: %d CCD images, %d registered, WCS rms %.3f, FWHM %.2f; the object r %.2f" % (
        c["n_ccd"], c["n_registered"], c["wcs_rms_median"], c["fwhm_median"], c["object_mag"].get("r", float("nan"))))
    for k in ("filter_A", "filter_B"):
        print("  %s %s" % (k, json.dumps(c[k])))
    print("  object recovered %s; injections %s" % (json.dumps(c["object_recovered"]), json.dumps(c["injection"])))
    print("  movers %d over %.3f deg^2 (%.0f per deg^2), %d known of %d known on the frames; rates %.1f +- %.1f, PA offset %+.0f +- %.0f; "
          "within 5/10: %d, 10/20: %d; expected within 30 arcsec %s" % (
              m["n"], m["area_deg2"], m["density_per_deg2"], m["n_known"], m["n_known_covered"], m["rate_mean"], m["rate_sd"], m["pa_offset_mean"],
              m["pa_offset_sd"], m["within_5_10"], m["within_10_20"], m["expected_30arcsec"]))
    for t in sorted(m["tracks"], key=lambda t: t["rate"]):
        print("    %5.1f arcsec/h PA %3.0f, %2d detections, support %d/%d, r %s%s" % (
            t["rate"], t["pa"], t["n"], t["support"], t["coverage"], "%.1f" % t["mag"] if t["mag"] else "-",
            ", %s" % t["known"]["name"] if t.get("known") else ""))


def forecast():
    """A2 from the 2030 apparition with and without the precovery."""
    f = darkcomet("forecast.json")
    print("== forecast: A2 %+.3e +- %.2e; without the precovery now +- %.2e; visible %s" % (f["A2"], f["A2_sigma"], f["A2_sigma_without_now"], f["visible"]))
    for k, s in f["scenarios"].items():
        print("  %-28s %s: +- %.2e without (%.1f sigma), +- %.2e with" % (k, ",".join(s["dates"]), s["A2_sigma_without"], s["z_without"], s["A2_sigma_with"]))


def thermal_variants():
    t = darkcomet("thermal_variants.json")
    print("== NEATM variants: reference D %.2f km" % t["reference"]["D_km"])
    for v in t["variants"]:
        print("  %-16s D %.2f km (%+.1f per cent)" % (v["variant"], v["D_km"], 100 * v["change"]))


def figure():
    """pgfplots code: cross- and along-track offsets against time for the orbits without 2007, without 2025, and with A2."""
    rf, _ = load()
    panels = [("(a) gravity only, fitted without 2007 September", rf["left_out"]["2007-09"]["resid"]),
              ("(b) gravity only, fitted without 2025 October", rf["left_out"]["2025-10"]["resid"]),
              ("(c) with $A_2$", rf["fits"]["with A2"]["resid"])]
    print("== figure")
    print(r"\begin{tikzpicture}")
    for k, (title, res) in enumerate(panels):
        at = "" if k == 0 else "at={(p%d.south)},anchor=north,yshift=-0.9cm," % (k - 1)
        ymin, ymax = (-12, 12) if k < 2 else (-1.5, 1.5)
        print(r"\begin{axis}[name=p%d,%swidth=0.95\textwidth,height=0.3\textwidth,xmin=2006.5,xmax=2027,ymin=%g,ymax=%g," % (k, at, ymin, ymax))
        print(r"  xlabel={%s},ylabel={offset (arcsec)},title={%s},title style={font=\small},tick label style={font=\footnotesize},"
              % ("year" if k == 2 else "", title))
        print(r"  label style={font=\small},x tick label style={/pgf/number format/1000 sep={}},legend style={font=\scriptsize,"
              r"at={(0.03,0.97)},anchor=north west,draw=none,fill=white,fill opacity=0.8,text opacity=1},legend columns=2]")
        for key, mark, col, lab in (("cross", "*", "black", "across the motion"), ("along", "o", "black!60", "along the motion")):
            kept = [r for r in res if r.get("kept", True)]
            print(r"\addplot[only marks,mark=%s,mark size=1.3pt,%s] coordinates {%s};"
                  % (mark, col, " ".join("(%.3f,%.2f)" % (year(r["jd"]), r[key]) for r in kept)))
            print(r"\addlegendentry{%s}" % lab)
            if k == 2 and any(not r.get("kept", True) for r in res):
                print(r"\addplot[only marks,mark=%s,mark size=1.3pt,%s,opacity=0.35,forget plot] coordinates {%s};"
                      % (mark, col, " ".join("(%.3f,%.2f)" % (year(r["jd"]), max(min(r[key], ymax), ymin)) for r in res if not r.get("kept", True))))
        print(r"\addplot[black!40,densely dotted,forget plot] coordinates {(2006.5,0) (2027,0)};")
        print(r"\end{axis}")
    print(r"\end{tikzpicture}")


def robust():
    """Night jackknife and bootstrap, weighting variants and the prediction table."""
    r = darkcomet("robust.json")
    print("== robustness")
    for k in ("jackknife_none", "jackknife_night", "bootstrap_free", "bootstrap_held"):
        print(k, json.dumps(r[k]))
    for k in sorted(t for t in r["fits"] if t.startswith(("clip", "floor", "default"))):
        f = r["fits"][k]
        print("%-22s A2 %+.3e +- %.2e (%.1f sigma), chi2/dof %.2f" % (k, f["A2"], f["A2_sigma"], abs(f["A2"] / f["A2_sigma"]), f["chi2"] / f["dof"]))
    night = {x["date"]: x for x in r["prediction"]["night"]}
    site = {x["date"]: x for x in r["prediction"]["none"]}
    for dt in ("2030-07-30", "2030-08-19", "2030-09-08", "2030-09-28"):
        n, s = night[dt], site[dt]
        print("%s & %.2f & %.2f & %.2f & %.1f & %.2f & %.2f & %.2f & %.1f \\\\  %% across %.2f %.2f"
              % (dt, n["sep"], n["sigma_accel"], n["sigma_grav"], n["z_0p2"], s["sep"], s["sigma_accel"], s["sigma_grav"], s["z_0p2"],
                 n["across_accel"], n["across_grav"]))


def precovery():
    """The re-measured 2007 frames against both orbits, as table rows, and the known asteroids around the track."""
    from astropy.time import Time
    print("== precovery frames")
    for f in darkcomet("frames_%s.json" % NIGHT):
        print("%s & %s & %s & %.0f & %.1f & %.2f & %.6f & $%+.6f$ & $%+.2f$, $%+.2f$ & $%+.2f$, $%+.2f$ \\\\" % (
            f["image"], Time(f["mjd_mid"], format="mjd").iso[11:19], f["filter"][:1], f["exptime"], f["snr"], f["trail_arcsec"],
            f["ra"], f["dec"], f["accel"]["dE"], f["accel"]["dN"], f["grav"]["dE"], f["grav"]["dN"]))
    fl = darkcomet("field_%s.json" % NIGHT)
    print("field:", json.dumps({k: v for k, v in fl.items() if k != "objects"}))


def table_s1():
    """Table S1: every asteroid of the screen, significant drifts first, with the published status the store records."""
    rows = json.load(open(results("yarkovsky_sign.json")))
    cols = ["designation", "A2_au_d2", "A2_snr", "A1_snr", "A3_snr", "H", "D_used_m", "D_source", "xi_500kgm3_albedo0.5",
            "xi_2000kgm3_albedo0.14", "T_J", "q_au", "orbit_class", "n_obs", "arc_days", "damit_cos_obliquity", "spin_sign", "published_status"]

    def f(x, fmt):
        return "" if x is None else fmt % x
    out = os.path.join(HERE, "data", "table_S1_screen.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for x in sorted(rows, key=lambda r: (abs(r["A2_snr"] or 0) < 3, -(r["xi"] or 0))):
            known = next((v for k, v in ss.yarkovsky_KNOWN.items() if "(%s)" % k in x["name"] or x["pdes"] == k), None)
            if x["pdes"] == "883607":
                known = "dark-comet candidate (this paper)"
            name = x["name"][1:-1] if x["name"].startswith("(") and x["name"].endswith(")") else x["name"]
            w.writerow([name, f(x["A2"], "%.4e"), f(x["A2_snr"], "%.2f"), f(x["A1_snr"], "%.2f"), f(x["A3_snr"], "%.2f"), f(x["H"], "%.2f"),
                        f(x["d_min_m"], "%.0f"), x["d_src"], f(x["xi"], "%.4g"), f(x["xi_typ"], "%.4g"), f(x["t_j"], "%.3f"), f(x["q"], "%.4f"),
                        x["cls"], x["n_obs"], x["arc"], ";".join("%.3f" % c for c in x["cos_obl"]), x["verdict"] or "", known or ""])
    print("== Table S1: %d rows to %s" % (len(rows), out))


def cutouts():
    """Figure: the object in each re-measured frame, north up and east left, with the positions both orbits give."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits as pyfits
    from astropy.time import Time
    from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
    pre = {(x["image"], x["ext"]): x for x in darkcomet("precovery_%s.json" % NIGHT) if x["found"]}
    frames = darkcomet("frames_%s.json" % NIGHT)
    cdir = os.path.join(ss.darkcomet_dir(DESIG), "cfht_%s" % NIGHT)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8})
    half = 8.0
    fig, axes = plt.subplots(4, 4, figsize=(6.6, 7.0))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, f in zip(axes.flat, frames):
        e = pre[(f["image"], f["ext"])]
        h = pyfits.open(os.path.join(cdir, "%s_%d.fits" % (f["image"], f["ext"])))[-1]
        img = h.data.astype(float)
        n = int(round(half / e["scale"]))
        xi, yi = (int(round(v)) for v in e["pix"])
        cut = img[max(yi - n, 0):yi + n + 1, max(xi - n, 0):xi + n + 1]
        # north up, east left: flip x where RA grows with x, flip y where Dec falls with y
        if h.header["CD1_1"] > 0:
            cut = cut[:, ::-1]
        if h.header["CD2_2"] < 0:
            cut = cut[::-1, :]
        ax.imshow(cut, origin="lower", cmap="gray_r", extent=[half, -half, -half, half], interpolation="nearest",
                  norm=ImageNormalize(cut, interval=ZScaleInterval(contrast=0.15), stretch=AsinhStretch(0.3)))
        ax.plot(0, 0, marker="+", ms=7, mew=0.8, color="#1f6fb4")
        ax.add_patch(plt.Circle((-f["accel"]["dE"], -f["accel"]["dN"]), 0.9, fill=False, lw=0.9, color="#c0392b"))
        ax.add_patch(plt.Rectangle((-f["grav"]["dE"] - 0.8, -f["grav"]["dN"] - 0.8), 1.6, 1.6, fill=False, lw=0.9, color="#2e8b57"))
        ax.set_xlim(half, -half)
        ax.set_ylim(-half, half)
        box = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8)
        ax.text(0.04, 0.95, "%s  %s UT" % (f["image"][:6], Time(f["mjd_mid"], format="mjd").iso[11:16]), transform=ax.transAxes,
                va="top", fontsize=6.5, bbox=box)
        ax.text(0.04, 0.05, "%s, %.0f s, S/N %.1f" % (f["filter"][:1], f["exptime"], f["snr"]), transform=ax.transAxes, va="bottom",
                fontsize=6.5, bbox=box)
    for ax in axes.flat[len(frames):]:
        ax.axis("off")
    key = axes.flat[len(frames)]
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.set_aspect("equal")
    key.plot(0.16, 0.88, marker="+", ms=7, mew=0.8, color="#1f6fb4")
    key.text(0.30, 0.88, "our centroid", va="center", fontsize=7)
    key.add_patch(plt.Circle((0.16, 0.74), 0.045, fill=False, lw=0.9, color="#c0392b"))
    key.text(0.30, 0.74, "orbit with $A_2$", va="center", fontsize=7)
    key.add_patch(plt.Rectangle((0.12, 0.56), 0.08, 0.08, fill=False, lw=0.9, color="#2e8b57"))
    key.text(0.30, 0.60, "gravity only,\nfitted without 2007", va="center", fontsize=7)
    # compass, north up and east left, and a 4 arcsec bar: a quarter of a panel's 16 arcsec
    key.annotate("", xy=(0.36, 0.38), xytext=(0.36, 0.16), arrowprops=dict(arrowstyle="->", lw=0.8))
    key.text(0.36, 0.40, "N", ha="center", va="bottom", fontsize=7)
    key.annotate("", xy=(0.14, 0.16), xytext=(0.36, 0.16), arrowprops=dict(arrowstyle="->", lw=0.8))
    key.text(0.12, 0.16, "E", ha="right", va="center", fontsize=7)
    key.plot([0.56, 0.81], [0.16, 0.16], color="k", lw=1.2)
    key.text(0.685, 0.20, "4$''$", ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04, hspace=0.04)
    fig.savefig(os.path.join(HERE, "cfht2007_cutouts.pdf"))
    print("== cutouts: %d frames to cfht2007_cutouts.pdf" % len(frames))


if __name__ == "__main__":
    fits()
    screen()
    spins()
    figure()
    robust()
    covariance()
    precovery()
    locus()
    census()
    forecast()
    thermal_variants()
    table_s1()
    cutouts()
