"""Numbers, Table 1 and Figure 1 of the note on Ma et al. (2023)'s eight signals of interest, from the soi approach's results in data/ (else the approach's results directory): each signal's line and its power in every scan of its session, the table of the eight, and the waterfall of MLc8's session as mlc8_session.pdf beside this file."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.argv = sys.argv[:1]
from setitools import breakthrough as B  # noqa: E402
from setitools import core, science  # noqa: E402


def data(name):
    """A result file from data/, else the soi results directory."""
    p = os.path.join(HERE, "data", name)
    return p if os.path.exists(p) else os.path.join(B.soi_ROOT, name)


def scans(tag):
    rows = [json.loads(l) for l in open(data("%s.jsonl" % tag))]
    return rows[0], rows[1:]


def utc(mjd):
    s = round(mjd % 1 * 86400)
    return "%02d:%02d" % (s // 3600 % 24, s % 3600 // 60)


def date(mjd):
    import datetime
    return (datetime.datetime(1858, 11, 17) + datetime.timedelta(days=float(mjd))).strftime("%Y %B %-d" if os.name != "nt" else "%Y %B %#d")


ERR = dict(MLc1=0.25, MLc2=0.07, MLc3=0.10, MLc4=0.03, MLc5=0.02, MLc6=0.04, MLc7=0.02, MLc8=0.010)  # Ma et al. (2023), Table 1


T = json.load(open(data("table.json")))
print("Table 1")
for t in T:
    away = t["other"]
    print("  %s & HIP~%s & %.6f & $%+.3f$ & $%+.3f$ & %s & %.1f & %s \\\\" % (
        t["tag"], t["star"][3:], t["mhz"], t["drift"], t["drift_tabulated"], ", ".join("%.0f" % x for x in t["on_snr"]), t["off_z"],
        "%d (%.0f)" % (len(away), max(a["snr"] for a in away)) if away else "0"))
print("\nmeasured minus tabulated frequency (kHz): %s" % ", ".join("%s %+.2f" % (t["tag"], (t["mhz"] - t["f0"]) * 1e3) for t in T))
print("drift signs opposite to the tabulated ones: %d of %d" % (sum(1 for t in T if t["drift"] * t["drift_tabulated"] < 0), len(T)))
print("|drift| minus |tabulated| in tabulated errors: %s" % ", ".join(
    "%s %+.1f" % (t["tag"], (abs(t["drift"]) - abs(t["drift_tabulated"])) / ERR[t["tag"]]) for t in T))
print("sessions: %s" % ", ".join("%s %s (%s to %s UTC)" % (t["tag"], date(t["t0"]), utc(min(r["mjd"] for r in scans(t["tag"])[1] if "mjd" in r)),
                                                           utc(max(r["mjd"] for r in scans(t["tag"])[1] if "mjd" in r))) for t in T))
print("own OFF scans on the line, combined: %s sigma" % ", ".join("%s %.1f" % (t["tag"], t["off_z"]) for t in T))
print("scans away from the star holding the signal: %s" % ", ".join("%s %d" % (t["tag"], len(t["other"])) for t in T))
print("scans in the sessions: %d, failed reads %d" % (sum(t["scans"] for t in T), sum(t["failed"] for t in T)))



def stars():
    """The stars' SIMBAD positions, kept in data/stars.json."""
    p = os.path.join(HERE, "data", "stars.json")
    if not os.path.exists(p):
        out = {}
        for s in sorted({t["star"] for t in T}):
            r = core.simbad("SELECT b.ra, b.dec FROM basic AS b JOIN ident AS i ON i.oidref = b.oid WHERE i.id = 'HIP %s'" % s[3:])[0]
            out[s] = [float(r["ra"]), float(r["dec"])]
        json.dump(out, open(p, "w"), indent=1)
    return json.load(open(p))


def skyfixed(star, mhz, mjd):
    """Drift (Hz/s) at the GBT of a transmitter at rest at the star: the observatory's barycentric velocity toward it over 60 s."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astropy.utils import iers
    iers.conf.auto_download = False  # the bundled Earth-orientation tables cover 2016
    src = SkyCoord(*S[star], unit="deg")
    v = [src.radial_velocity_correction(kind="barycentric", obstime=Time(mjd + dt / 86400, format="mjd", scale="utc"),
                                        location=science.location("GBT")).to(u.km / u.s).value for dt in (-30.0, 30.0)]
    return mhz * 1e6 * (v[1] - v[0]) / 60.0 / science.C


S = stars()
print("drift of a transmitter at rest at the star, against the line's: %s" % ", ".join(
    "%s %+.3f (%+.3f)" % (t["tag"], skyfixed(t["star"], t["mhz"], t["t0"]), t["drift"]) for t in T))
print("OFF level against ON on the line: %s" % ", ".join(
    "%s %.3f" % (t["tag"], np.mean([o["on_line"] for o in t["off"]]) / np.mean(t["on_line"])) for t in T))
m1 = [t for t in T if t["tag"] == "MLc1"][0]
line1, rows1 = scans("MLc1")
d1 = [max(q for q in r["tracks"] if abs(q[1] - (line1["hz"] + line1["drift"] * (r["mjd"] - line1["t0"]) * 86400)) <= 3000)[2]
      for r in rows1 if r.get("cadence") == "HIP13402" and r["on"] and r["tracks"] and abs(r["mjd"] - line1["t0"]) < 0.05
      and any(abs(q[1] - (line1["hz"] + line1["drift"] * (r["mjd"] - line1["t0"]) * 86400)) <= 3000 for q in r["tracks"])]
c = science.C * 1e3 / (m1["mhz"] * 1e6)  # m/s^2 per Hz/s
print("MLc1: line-of-sight acceleration of its own at HIP 13402 %.3f m/s^2; ON tracks' drifts %s Hz/s, a change of %.3f m/s^2; "
      "session %s to %s UTC" % ((m1["drift"] - skyfixed("HIP13402", m1["mhz"], m1["t0"])) * c, d1, abs(d1[1] - d1[0]) * c,
                                utc(rows1[0]["mjd"] - 150 / 86400), utc(rows1[-1]["mjd"] + 150 / 86400)))

m4, m5 = [t for t in T if t["tag"] in ("MLc4", "MLc5")]
print("\nMLc4 and MLc5: %.6f MHz apart, drifts %+.4f and %+.4f Hz/s, away scans %s and %s" % (
    m5["mhz"] - m4["mhz"], m4["drift"], m5["drift"], [(a["target"], a["snr"]) for a in m4["other"]], [(a["target"], a["snr"]) for a in m5["other"]]))

for tag in ("MLc8", "MLc1"):
    line, rows = scans(tag)
    t = [x for x in T if x["tag"] == tag][0]
    print("\n%s: %.6f MHz at %s UTC (MJD %.5f), %+.4f Hz/s; ON tracks S/N %s" % (tag, t["mhz"], utc(line["t0"]), line["t0"], line["drift"], t["on_snr"]))
    for r in rows:
        if r.get("cadence") != line["star"] or abs(r["mjd"] - line["t0"]) > 0.05:
            continue
        pred = line["hz"] + line["drift"] * (r["mjd"] - line["t0"]) * 86400
        near = [q for q in r["tracks"] if abs(q[1] - pred) <= 3000]
        print("  %s %-9s %-3s on the line %.2f against %.2f +- %.2f (%d of %d parallel tracks as high); tracks within 3 kHz %s" % (
            utc(r["mjd"]), r["target"], "ON" if r["on"] else "OFF", r["on_line"], r["null_mean"], r["null_std"], r["as_high"], r["null_n"], near[:2]))
    on = [r["on_line"] for r in rows if r.get("cadence") == line["star"] and r["on"] and abs(r["mjd"] - line["t0"]) < 0.05]
    off = [r["on_line"] for r in rows if r.get("cadence") == line["star"] and r["on"] is False and abs(r["mjd"] - line["t0"]) < 0.05]
    print("  OFF level against ON on the line: %.3f" % (np.mean(off) / np.mean(on)))
    if t["other"]:
        a = sorted(t["other"], key=lambda a: a["hours"])
        print("  away from the star: %d scans (%d ON, %d OFF, %d neither) from %s to %s UTC, %d stars' cadences; S/N %s" % (
            len(a), sum(1 for x in a if x["on"]), sum(1 for x in a if x["on"] is False), sum(1 for x in a if x["on"] is None),
            utc(line["t0"] + a[0]["hours"] / 24), utc(line["t0"] + a[-1]["hours"] / 24),
            len({r["cadence"] for r in rows for x in a if r["file"] == x["file"]}), [x["snr"] for x in a]))
        print("  strongest: %s" % [(utc(line["t0"] + x["hours"] / 24), x["target"], "ON" if x["on"] else "OFF", x["snr"])
                                   for x in sorted(a, key=lambda x: -x["snr"])[:4]])
        for x in a:
            r = [r for r in rows if r["file"] == x["file"]][0]
            pred = line["hz"] + line["drift"] * (r["mjd"] - line["t0"]) * 86400
            q = max([q for q in r["tracks"] if abs(q[1] - pred) <= B.soi_NEAR[0] and abs(q[2] - line["drift"]) <= B.soi_NEAR[1]])
            print("    %s %-9s %-3s %.6f MHz %+.3f Hz/s S/N %.1f" % (utc(r["mjd"]), r["target"], "ON" if r["on"] else "OFF",
                                                                 t["f0"] + q[1] / 1e6, q[2], q[0]))

# Figure 1: MLc8's session
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

w = np.load(data("MLc8_waterfall.npz"))
z, tm, hz, on, tgt, cad = w["z"], w["t"], w["hz"], w["on"], w["target"], w["cadence"]
x = (hz + (1724.972561 - 1724.97) * 1e6) / 1e3  # kHz above 1724.97 MHz
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(6.0, 7.6))
dt = np.nanmedian(np.diff(tm, axis=1)) * 24
day0 = np.floor(np.nanmax(tm))  # the session's second day
hours = (tm - day0) * 24  # hours from 0 UTC on that day
for i in range(len(z)):
    k = np.isfinite(tm[i])
    h = hours[i][k]
    ax.imshow(np.clip(z[i][k].astype(float), -1, 8)[:, ::-1] if hz[0] > hz[-1] else np.clip(z[i][k].astype(float), -1, 8),
              extent=[x.min(), x.max(), h[-1] + dt / 2, h[0] - dt / 2], aspect="auto", cmap="gray_r", vmin=-1, vmax=8,
              interpolation="nearest")
    ax.add_patch(plt.Rectangle((x.min() - 0.17, h[0] - dt / 2), 0.11, len(h) * dt,
                               color={1: "tab:orange", 0: "tab:blue", -1: "0.6"}[int(on[i])], clip_on=False))
for c in sorted({str(v) for v in cad if v}):
    hs = [hours[i][0] for i in range(len(z)) if str(cad[i]) == c]
    he = [hours[i][np.isfinite(hours[i])][-1] for i in range(len(z)) if str(cad[i]) == c]
    ax.text(x.max() + 0.06, (min(hs) + max(he)) / 2, c.replace("HIP", "HIP "), va="center", fontsize=8)
    if c == "HIP62207":
        ax.add_patch(plt.Rectangle((x.min(), min(hs) - dt), x.max() - x.min(), max(he) - min(hs) + 2 * dt, fill=False, ec="tab:red", lw=1.2))
ax.set_xlim(x.min() - 0.19, x.max())
ax.set_ylim(np.nanmax(hours) + dt, np.nanmin(hours) - dt)
ax.set_xlabel(r"$\nu - 1724.97$ MHz (kHz)")
ax.set_ylabel("UTC (h) from 0 h on %s" % date(day0))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "mlc8_session.pdf"))
print("\nFigure 1: mlc8_session.pdf, %d scans, %.2f to %.2f kHz above 1724.97 MHz" % (len(z), x.min(), x.max()))
