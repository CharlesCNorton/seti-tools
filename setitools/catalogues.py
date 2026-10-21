"""Readers of the catalogues that several approaches share, each answering rows as dicts, with the epochs of their positions and the moved-position null."""
import math
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
