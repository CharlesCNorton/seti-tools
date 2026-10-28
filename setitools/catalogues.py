"""Readers of the catalogues that several approaches share, each answering rows as dicts, with the epochs of their positions and the moved-position null."""
import math
import os
import re

from setitools import core

# the epoch of each CDS catalogue's positions, to which a star's proper motion carries it before a cross-match
EPOCH = {"I/355/gaiadr3": 2016.0, "II/246/out": 2000.0, "II/328/allwise": 2010.5, "II/335/galex_ais": 2007.0, "J/A+A/682/A34/erass1-m": 2020.0,
         "J/ApJS/255/30/comp": 2019.0, "VIII/65/nvss": 1995.0, "II/125/main": 1983.5}

GCNS = "external.gaiaedr3_gcns_main_1"
WD_CATALOGUE = '"J/MNRAS/508/3877/maincat"'
FRB_CATALOGUE = '"J/ApJS/257/59/table2"'
PSR_CATALOGUE = '"B/psr/psr"'
TY_CATALOGUE = ('"J/ApJS/253/7/plx"', '"J/ApJS/253/7/table11"')


DWARF_SEQUENCE = "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"


def dwarf_sequence():
    """Pecaut and Mamajek's (2013, updated) dwarf sequence: each type with its subclass number (O0 0 to M0 60), temperature, absolute V, B-V, BP-RP and mass, None where the table has none."""
    out = []
    for x in core.get(DWARF_SEQUENCE, timeout=120).decode("latin-1").splitlines():
        t = x.split()
        m = re.match(r"^([OBAFGKMLTY])([0-9.]+)V$", t[0]) if t else None
        if not m or len(t) < 30:
            continue
        out.append(dict(type=t[0], sub=10 * "OBAFGKMLTY".index(m.group(1)) + float(m.group(2)), teff=core.num(t[1]), mv=core.num(t[7]), bv=core.num(t[8]),
                        bprp=core.num(t[11]), mass=core.num(t[-2])))
    return out


GCVS = "http://www.sai.msu.su/gcvs/gcvs/gcvs5/gcvs5.txt"


def gcvs():
    """The General Catalogue of Variable Stars (Samus et al. 2017) from its own site into the data cache: name, J2000 position with its epoch and proper motion (mas/yr), type, magnitude range and band, epoch (JD - 2400000), outburst year and period (days) with the period's own marks, and the mark on the identification of the position (? where the GCVS doubts it)."""
    path = core.data("gcvs5.txt")
    if not os.path.exists(path):
        core.download(GCVS, path)

    def first(s):
        v = re.search(r"\d+(?:\.\d+)?", s)
        return float(v.group(0)) if v else None
    out = []
    for x in open(path, encoding="latin-1"):
        f = x.split("|")
        m = re.match(r"\s*(\d\d)(\d\d)(\d\d(?:\.\d*)?)\s*([+-])(\d\d)(\d\d)(\d\d(?:\.\d*)?)", f[2]) if len(f) >= 20 else None
        if not m:
            continue
        pm = re.findall(r"[+-]\d+\.\d+", f[15])
        lo, per = f[5].strip(), f[10].strip()
        out.append(dict(name=" ".join(f[1].replace("*", " ").split()), ra=15 * (int(m.group(1)) + int(m.group(2)) / 60 + float(m.group(3)) / 3600),
                        dec=(int(m.group(5)) + int(m.group(6)) / 60 + float(m.group(7)) / 3600) * (-1 if m.group(4) == "-" else 1),
                        pos_epoch=core.num(f[16].strip()) or 2000.0, pmra=1000 * float(pm[0]) if len(pm) == 2 else None,
                        pmdec=1000 * float(pm[1]) if len(pm) == 2 else None, type=f[3].strip(), max=first(f[4]),
                        min=None if lo.startswith("(") else first(lo), amp=first(lo) if lo.startswith("(") else None, fainter="<" in lo, band=f[7].strip(),
                        epoch=core.num(f[8].strip().rstrip(":")), year=core.num(f[9].strip()), period=first(per) if per and not per.startswith("(") else None,
                        period_mark=per, spectrum=f[12].strip(), ident=f[17].strip() if len(f) > 17 else ""))
    return out


_VSX_LAST = [0.0]


def vsx_api(params, gap=3.0):
    """A reply of AAVSO's VSX API as parsed JSON, kept in the data cache and fetched no sooner than GAP seconds after the last fetch; a VSX that does not answer raises, and a rerun resumes from the cache."""
    import hashlib
    import json
    import time
    import urllib.parse
    query = urllib.parse.urlencode(params)
    path = core.data("vsx", hashlib.sha1(query.encode()).hexdigest()[:24] + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    time.sleep(max(0.0, gap - (time.time() - _VSX_LAST[0])))
    try:
        raw = core.get("https://vsx.aavso.org/index.php?" + query, timeout=90, pause=30)
    finally:
        _VSX_LAST[0] = time.time()
    js = json.loads(raw.decode("utf-8", "replace") or "{}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(js, f)
    return js


def vsx_object(name):
    """AAVSO's VSX record of a variable by its name: name, J2000 position, type, period and magnitude range, or None."""
    js = vsx_api(dict(view="api.object", ident=name, format="json"))
    o = (js.get("VSXObject") if isinstance(js, dict) else None) or {}
    if not o.get("RA2000"):
        return None
    return dict(name=o.get("Name"), ra=float(o["RA2000"]), dec=float(o["Declination2000"]), type=str(o.get("VariabilityType") or ""), P=core.num(o.get("Period")),
                max=o.get("MaxMag"), min=o.get("MinMag"))


def vsx_near(ra, dec, radius_arcsec):
    """The variables AAVSO's VSX lists within RADIUS_ARCSEC of a position: name, J2000 position, type, period and separation in arcsec, nearest first."""
    js = vsx_api(dict(view="api.list", ra="%.6f" % ra, dec="%.6f" % dec, radius="%.6f" % (radius_arcsec / 3600.0), format="json"))
    items = (js.get("VSXObjects") or {}).get("VSXObject") or [] if isinstance(js, dict) else []
    items = [items] if isinstance(items, dict) else items
    out = [dict(name=i.get("Name"), ra=float(i["RA2000"]), dec=float(i["Declination2000"]), type=str(i.get("VariabilityType") or ""), P=core.num(i.get("Period")),
                sep=core.sep_arcsec(ra, dec, float(i["RA2000"]), float(i["Declination2000"]))) for i in items if i.get("RA2000")]
    return sorted(out, key=lambda i: i["sep"])


def reddening(ra, dec):
    """Schlafly and Finkbeiner's (2011) E(B-V) at a position from IRSA's dust service, or None."""
    raw = core.get("https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr=%.6f+%.6f+equ+j2000" % (ra, dec), timeout=120, soft=True)
    m = re.search(r"<refPixelValueSandF>\s*([-\d.]+)", raw.decode("utf-8", "replace")) if raw else None
    return float(m.group(1)) if m else None


def gaia_match(rows, cols="g.source_id, g.phot_g_mean_mag", radius_arcsec=5.0, batch=5000, epoch=2016.0):
    """Gaia DR3 sources within RADIUS_ARCSEC of each row's position, carried from its pos_epoch to EPOCH by its proper motion where it has one, through the archive's table upload in batches: the matches of each row's index, nearest first, with their separation in arcsec as sep."""
    out = {}
    for k in range(0, len(rows), batch):
        chunk = rows[k:k + batch]
        pos = [carried(r["ra"], r["dec"], r["pmra"], r["pmdec"], epoch - (r.get("pos_epoch") or 2000.0)) if r.get("pmra") is not None and r.get("pmdec") is not None
               else (r["ra"], r["dec"]) for r in chunk]
        q = ("SELECT u.i, DISTANCE(POINT('ICRS', u.ra, u.dec), POINT('ICRS', g.ra, g.dec)) * 3600 AS sep, %s FROM tap_upload.t AS u "
             "JOIN gaiadr3.gaia_source AS g ON 1 = CONTAINS(POINT('ICRS', u.ra, u.dec), CIRCLE('ICRS', g.ra, g.dec, %.8f))" % (cols, radius_arcsec / 3600.0))
        for r in core.tap_async("gaia", q, upload=dict(t=dict(i=list(range(k, k + len(chunk))), ra=[p[0] for p in pos], dec=[p[1] for p in pos])), poll=5,
                                maxrec=20 * len(chunk)):
            out.setdefault(int(r["i"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: float(r["sep"]))
    return out


def carried(ra, dec, pmra, pmdec, years):
    """A position (deg) carried YEARS forward by proper motions in mas/yr (pmra with cos dec)."""
    return ra + pmra / 3.6e6 * years / max(math.cos(math.radians(dec)), 1e-9), dec + pmdec / 3.6e6 * years


def gcns(columns, where="gcns_prob > 0.9", ranges=0, maxrec=400000, timeout=600):
    """Gaia Catalogue of Nearby Stars rows: COLUMNS of the sources meeting WHERE, read in RANGES source_id ranges when the selection is too large for one statement."""
    if ranges:
        return core.gaia_by_range("SELECT %s FROM %s WHERE source_id BETWEEN {lo} AND {hi} AND %s" % (columns, GCNS, where), n=ranges, maxrec=maxrec)
    return core.tap("gaia", "SELECT %s FROM %s WHERE %s" % (columns, GCNS, where), maxrec=maxrec, timeout=timeout)


def white_dwarfs(columns="WDJname, RA_ICRS, DE_ICRS, pmRA, pmDE, Plx, TeffH", plx_min=10.0, pwd=0.75, teff_max=None, where=None, timeout=600):
    """Gentile Fusillo et al. (2021) white dwarfs beyond PLX_MIN mas at white-dwarf probability above PWD, below TEFF_MAX in the hydrogen model, meeting WHERE."""
    q = "SELECT %s FROM %s WHERE Plx > %g AND Pwd > %g" % (columns, WD_CATALOGUE, plx_min, pwd)
    if teff_max is not None:
        q += " AND TeffH < %g" % teff_max
    if where:
        q += " AND " + where
    return core.tap("vizier", q, timeout=timeout)


def white_dwarf_positions(epoch, teff_max=None, plx_min=10.0, pwd=0.75):
    """White dwarfs as cross-match positions (name, ra, dec) carried from 2016.0 to EPOCH, with each row's TeffH and Plx."""
    out = []
    for w in white_dwarfs(plx_min=plx_min, pwd=pwd, teff_max=teff_max):
        ra, de = carried(float(w["RA_ICRS"]), float(w["DE_ICRS"]), core.num(w["pmRA"]) or 0.0, core.num(w["pmDE"]) or 0.0, epoch - 2016.0)
        out.append(dict(name=w["WDJname"].strip(), ra=ra, dec=de, teff=core.num(w["TeffH"]), plx=core.num(w["Plx"])))
    return out


def frb_catalog1(columns="Name, RpName, RAJ2000, DEJ2000, DMfitb, e_DMfitb, MJDinf, MJD400", one_per_burst=True, one_per_source=False, timeout=300):
    """CHIME/FRB Catalog 1 rows: COLUMNS, one row per burst (the table lists sub-bursts as rows of their own) and, if asked, one per source, the first burst of each repeater."""
    rows = core.tap("vizier", "SELECT %s FROM %s" % (columns, FRB_CATALOGUE), timeout=timeout)
    if one_per_burst or one_per_source:
        out = {}
        for r in rows:
            key = (r.get("RpName") or "").strip() if one_per_source and (r.get("RpName") or "").strip() not in ("", "-") else r["Name"]
            out.setdefault(key, r)
        rows = list(out.values())
    return rows


def pulsars(columns="PSRJ, RAJ2000, DEJ2000, P0, P1, e_P1, P1I, DM, Dist, Assoc, Binary, PB, BINCOMP", where=None, timeout=300):
    """ATNF pulsar catalogue rows (VizieR B/psr): COLUMNS of the pulsars meeting WHERE."""
    q = "SELECT %s FROM %s" % (columns, PSR_CATALOGUE)
    if where:
        q += " WHERE " + where
    return core.tap("vizier", q, timeout=timeout)


def pulsar_isolated(r):
    """A pulsar with no binary companion in the catalogue."""
    return not (r.get("Binary") or "").strip() and not (r.get("PB") or "").strip()


def pulsar_in_field(r):
    """A pulsar outside globular clusters and other galaxies by the catalogue's association."""
    a = r.get("Assoc") or ""
    return "GC:" not in a and "EXGAL" not in a


def cold_dwarfs(epoch=2020.0, types=("T", "Y")):
    """The dwarfs of the Kirkpatrick et al. (2021) 20 pc census with Spitzer astrometry whose infrared type begins with one of TYPES: name, type, temperature and position carried to EPOCH by the census's proper motion."""
    sp = {}
    for r in core.tap("vizier", "SELECT FName, SpTIR, Teff FROM %s" % TY_CATALOGUE[1]):
        m = re.search(r"J\d{6}\.\d{2}[+-]\d{6}\.\d", r["FName"])
        if m:
            sp[m.group(0)] = r
    out = []
    for r in core.tap("vizier", "SELECT FName, RA_ICRS, DE_ICRS, MJD, plx, pmRA, pmDE FROM %s" % TY_CATALOGUE[0]):
        m = re.search(r"J\d{6}\.\d{2}[+-]\d{6}\.\d", r["FName"])
        s = sp.get(m.group(0)) if m else None
        if s is None or str(s["SpTIR"]).strip()[:1] not in types or core.num(r["RA_ICRS"]) is None or core.num(r["MJD"]) is None:
            continue
        years = epoch - (2000.0 + (float(r["MJD"]) - 51544.5) / 365.25)
        ra, de = carried(float(r["RA_ICRS"]), float(r["DE_ICRS"]), core.num(r["pmRA"]) or 0.0, core.num(r["pmDE"]) or 0.0, years)
        out.append(dict(name=r["FName"].strip(), ra=ra, dec=de, plx=core.num(r["plx"]), sptype=(s or {}).get("SpTIR"), teff=core.num((s or {}).get("Teff"))))
    return out


def shifted_matches(pos, cat, radius, shift_arcmin=2.0, keyfn=None):
    """CDS XMatch of positions (dicts with name, ra, dec) at their places and moved SHIFT_ARCMIN in eight directions: the matches at their places, and the counts of matched positions when moved, the null of chance alignment."""
    def moved(dra, dde):
        up = []
        for p in pos:
            ra, de = p["ra"] + dra / max(math.cos(math.radians(p["dec"])), 0.05), p["dec"] + dde
            if abs(de) > 90:
                ra, de = ra + 180, math.copysign(180, de) - de
            up.append(dict(source_id=p["name"], ra=ra % 360, dec=de))
        return core.xmatch(up, cat, radius)
    d = shift_arcmin / 60.0
    at = moved(0, 0)
    null = [len({m["source_id"] for m in moved(x, y) if keyfn is None or keyfn(m)})
            for x, y in ((d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, d), (d, -d), (-d, -d))]
    return [m for m in at if keyfn is None or keyfn(m)], null
