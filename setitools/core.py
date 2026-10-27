"""Core of seti-tools: configuration, network access, archive clients, Kepler and TESS light curves, and the coverage store."""
import csv
import hashlib
import http.client
import io
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# --------------------------------------------------------------------------------------------------------------------
# config: paths and resource caps from SETI_* variables, else the config file SETITOOLS_CONFIG names, else repository defaults

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config_path():
    if os.environ.get("SETITOOLS_CONFIG"):
        return os.environ["SETITOOLS_CONFIG"]
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "setitools", "config.json")


_path = _config_path()
LOCAL = json.load(open(_path)) if os.path.exists(_path) else {}


def _setting(key, default, *env):
    for name in env:
        if os.environ.get(name):
            return os.environ[name]
    return LOCAL.get(key, default)


DATA = _setting("data", os.path.join(ROOT, "data"), "SETI_DATA")
WORK = _setting("work", os.path.join(ROOT, "work"), "SETI_WORK")
CACHE = _setting("cache", os.path.join(ROOT, "cache"), "SETI_CACHE")
RESULTS = _setting("results", os.path.join(ROOT, "results"), "SETI_RESULTS", "SETI_HERE")
COVERAGE = _setting("coverage", os.path.join(ROOT, "coverage.db"), "SETI_COVERAGE")

MAX_PROCS = int(LOCAL.get("max_procs", 8))
MAX_CONNECTIONS = int(LOCAL.get("max_connections", 32))


def data(*parts):
    """Path under the bulk-data directory."""
    return os.path.join(DATA, *parts)


def results(*parts):
    """Path under the results directory."""
    return os.path.join(RESULTS, *parts)


# --------------------------------------------------------------------------------------------------------------------
# util: catalogue values, angles, proper motion, Gaia photometry, running statistics, warnings

def quiet_warnings():
    """Silence the expected astropy and numpy warnings (empty, all-NaN, invalid, overflow, divide); show all others."""
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message=r"(Mean of empty slice|All-NaN (slice|axis) encountered|invalid value encountered"
                                    r"|divide by zero encountered|overflow encountered|Degrees of freedom <= 0)")
    try:
        from astropy.utils.exceptions import AstropyWarning
    except ImportError:
        return
    warnings.filterwarnings("ignore", category=AstropyWarning)


def num(x, missing=None):
    """Catalogue value as a finite float, else missing."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return missing
    return v if math.isfinite(v) else missing


def sep_arcsec(ra1, dec1, ra2, dec2):
    """Angular separation in arcsec (haversine)."""
    r1, d1, r2, d2 = (math.radians(float(v)) for v in (ra1, dec1, ra2, dec2))
    s = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(s)))) * 3600.0


def star_at(ra, dec, pmra, pmdec, epoch, at):
    """Position (deg) at Julian year at from epoch and proper motion (mas/yr, pmra with cos dec), linear in time."""
    dt = at - epoch
    return (ra + (pmra or 0.0) / 3.6e6 * dt / max(math.cos(math.radians(dec)), 1e-6),
            dec + (pmdec or 0.0) / 3.6e6 * dt)


def mjd_to_year(mjd):
    """Julian year of an MJD."""
    return 2000.0 + (mjd - 51544.5) / 365.25


def ms_luminosity(mass_msun):
    """Main-sequence luminosity (L_sun): M^4 above 0.43 M_sun, 0.23 M^2.3 below."""
    m = float(mass_msun)
    return m ** 4 if m >= 0.43 else 0.23 * m ** 2.3


def cstar(excess_factor, bp_rp):
    """Corrected BP/RP flux excess C* (Riello et al. 2021, eq. 6); a companion lost by the G fit raises it."""
    c, x = float(excess_factor), float(bp_rp)
    if x < 0.5:
        return c - (1.154360 + 0.033772 * x + 0.032277 * x * x)
    if x < 4.0:
        return c - (1.162004 + 0.011464 * x + 0.049255 * x * x - 0.005879 * x ** 3)
    return c - (1.057572 + 0.140537 * x)


def cstar_sigma(g):
    """Scatter of C* at G magnitude g (Riello et al. 2021, eq. 18)."""
    return 0.0059898 + 8.817481e-12 * float(g) ** 7.618399


def gaia_g_scatter(flux, flux_error, n_obs):
    """Per-observation G scatter (mag) implied by Gaia's mean flux error."""
    return 1.0857 * float(flux_error) * math.sqrt(float(n_obs)) / float(flux)


def robust_sigma(x):
    """1.4826 times the median absolute deviation."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def running(x, y, at, half=100):
    """Median and robust scatter of y over the 2*half rank-nearest points of x at each value of at."""
    o = np.argsort(x)
    xs, ys = np.asarray(x, float)[o], np.asarray(y, float)[o]
    med, sd = [], []
    for v in np.atleast_1d(at):
        j = np.searchsorted(xs, v)
        sl = ys[max(j - half, 0):min(j + half, len(xs))]
        m = float(np.median(sl))
        med.append(m)
        sd.append(float(1.4826 * np.median(np.abs(sl - m))))
    return np.array(med), np.array(sd)


# --------------------------------------------------------------------------------------------------------------------
# net: HTTP and TAP access with retries, byte ranges and bounded parallelism

# user agent naming the software alone
UA = {"User-Agent": "seti-tools Python-urllib/%d.%d" % sys.version_info[:2]}
# soft fetches that failed other than as missing files, as (url, error)
SOFT_FAILURES = []

# TAP sync endpoints; ESO answers only HTTPS GET, SIMBAD wants lower-case parameters
TAP = {
    "gaia": "https://gea.esac.esa.int/tap-server/tap/sync",
    "simbad": "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
    "eso": "https://archive.eso.org/tap_obs/sync",
    "irsa": "https://irsa.ipac.caltech.edu/TAP/sync",
    "alma": "https://almascience.eso.org/tap/sync",
    "casda": "https://casda.csiro.au/casda_vo_tools/tap/sync",
    "datacentral": "https://datacentral.org.au/vo/tap/sync",
    "vizier": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
    "exoplanets": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
    "datalab": "https://datalab.noirlab.edu/tap/sync",
    "xsa": "https://nxsa.esac.esa.int/tap-server/tap/sync",
    "lamost": "https://www.lamost.org/dr10/v2.0/voservice/tap/sync",
}
MAXREC = 10000000


def get(url, headers=None, data=None, timeout=120, retries=4, byte_range=None, soft=False, pause=5):
    """Bytes of a URL with retries and growing pauses; byte_range slices; soft returns None on 404 or exhausted retries and logs the latter."""
    h = dict(UA)
    if headers:
        h.update(headers)
    if byte_range:
        h["Range"] = "bytes=%d-%d" % byte_range
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=h, data=data)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            last = e
            if soft and e.code == 404:
                return None
        except Exception as e:
            last = e
        if attempt < retries - 1:
            time.sleep(pause * (attempt + 1))
    if soft:
        SOFT_FAILURES.append((url, repr(last)[:200]))
        sys.stderr.write("net.get: %s failed %d times (%s); returned None\n" % (url, retries, repr(last)[:200]))
        return None
    raise last


_CONNECTIONS = threading.local()


def _drop_connection(scheme, host):
    c = getattr(_CONNECTIONS, "pool", {}).pop((scheme, host), None)
    if c is not None:
        c.close()


def _kept_get(scheme, host, path, headers, timeout):
    """Response on the thread's kept connection to host, reopened when the server closed it."""
    pool = getattr(_CONNECTIONS, "pool", None)
    if pool is None:
        pool = _CONNECTIONS.pool = {}
    for fresh in (False, True):
        c = None if fresh else pool.get((scheme, host))
        new = c is None
        if new:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            c = pool[(scheme, host)] = cls(host, timeout=timeout)
        try:
            c.request("GET", path, headers=headers)
            return c.getresponse()
        except Exception:
            _drop_connection(scheme, host)
            if new:
                raise


class RangeError(IOError):
    """Range past the end of the file (206 without it, or 416); not retried."""


def get_ranges(url, ranges, timeout=60, retries=4, pause=5, size=False):
    """Several (first, last) slices of one URL in one request, with the file length when size is set; raises unless 206 with every range."""
    u = urllib.parse.urlsplit(url)
    path = u.path + ("?" + u.query if u.query else "")
    h = dict(UA)
    h["Range"] = "bytes=" + ",".join("%d-%d" % r for r in ranges)
    last = None
    for attempt in range(retries):
        try:
            r = _kept_get(u.scheme, u.netloc, path, h, timeout)
            if r.status == 416:
                # IRSA refuses the whole request when any range starts past the end
                _drop_connection(u.scheme, u.netloc)
                raise RangeError("%s answered 416: a range lies past the end of the file" % url)
            if r.status != 206:
                _drop_connection(u.scheme, u.netloc)
                raise IOError("%s answered %d to a range request" % (url, r.status))
            body = r.read()
            parts = []
            ctype = r.getheader("Content-Type", "")
            if ctype.startswith("multipart/byteranges"):
                boundary = ctype.split("boundary=")[1].strip().strip('"').encode()
                for chunk in body.split(b"--" + boundary)[1:]:
                    head, sep, data = chunk.partition(b"\r\n\r\n")
                    m = re.search(rb"Content-Range:\s*bytes (\d+)-(\d+)/(\d+)", head, re.I)
                    if sep and m:
                        a, b, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        parts.append((a, b, total, data[:b - a + 1]))
            else:
                m = re.match(r"bytes (\d+)-(\d+)/(\d+)", r.getheader("Content-Range", ""))
                a, b, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
                parts.append((a, b, total, body[:b - a + 1]))
            out = []
            for first, end in ranges:
                # a range past the end comes back cut at the end
                p = next((p for p in parts if p[0] <= first <= p[1] and (end <= p[1] or p[1] == p[2] - 1)), None)
                if p is None:
                    raise RangeError("%s left out bytes %d-%d" % (url, first, end))
                out.append(p[3][first - p[0]:end - p[0] + 1])
            return (out, parts[0][2]) if size else out
        except RangeError:
            raise
        except Exception as e:
            last = e
            _drop_connection(u.scheme, u.netloc)
        if attempt < retries - 1:
            time.sleep(pause * (attempt + 1))
    raise last


def download(url, path, size=None, headers=None, retries=4, timeout=3600, pause=20, chunk=1 << 20):
    """Stream a URL to path through path.tmp with retries; True when complete (and size bytes long when given)."""
    import os
    if os.path.exists(path) and (size is None or os.path.getsize(path) == size):
        return True
    h = dict(UA)
    h.update(headers or {})
    tmp = path + ".tmp"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r,                     open(tmp, "wb") as fh:
                while True:
                    b = r.read(chunk)
                    if not b:
                        break
                    fh.write(b)
            if size is None or os.path.getsize(tmp) == size:
                os.replace(tmp, path)
                return True
        except Exception:
            pass
        time.sleep(pause * (attempt + 1))
    return False


def tap(service, query, fmt="csv", method=None, timeout=600, maxrec=None):
    """ADQL on a named TAP service or sync URL: rows as dicts (csv) or text; MAXREC set to avoid silent 50000-row cuts."""
    url = TAP.get(service, service)
    if maxrec is None and service != "datalab":
        maxrec = MAXREC
    params = dict(REQUEST="doQuery", LANG="ADQL", FORMAT=fmt, QUERY=query)
    if maxrec:
        params["MAXREC"] = maxrec
    if service == "simbad":
        params = dict(request="doQuery", lang="adql", format="json" if fmt == "json" else fmt, query=query,
                      maxrec=maxrec)
    method = method or ("GET" if service in ("eso", "irsa", "alma", "casda", "datacentral", "exoplanets", "vizier",
                                             "datalab") else "POST")
    enc = urllib.parse.urlencode(params)
    if method == "GET":
        raw = get(url + "?" + enc, timeout=timeout)
    else:
        raw = get(url, data=enc.encode(), timeout=timeout)
    text = raw.decode("utf-8-sig", errors="replace")
    if fmt == "votable":
        if b"QUERY_STATUS\" value=\"ERROR" in raw:
            raise IOError("%s answered with an error: %s" % (service, text[:400]))
    elif fmt in ("csv", "json") and text.lstrip()[:1] == "<":
        # a maintenance or error page must not pass for rows
        raise IOError("%s answered with an HTML or XML page instead of %s: %s" % (service, fmt, text[:200]))
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "json":
        return json.loads(text)
    if fmt == "votable":
        return votable_rows(raw)
    return text


def tap_by_ids(service, ids, query, key="source_id", size=2000, timeout=600, tries=4):
    """Rows of an IN (%s) query by id in batches, keyed by key; a batch failing every retry raises."""
    out = {}
    ids = [str(x) for x in ids]
    for i in range(0, len(ids), size):
        chunk = ids[i:i + size]
        part, last = None, None
        for attempt in range(tries):
            try:
                part = tap(service, query % ",".join(chunk), maxrec=3 * size, timeout=timeout)
                break
            except Exception as e:
                last = e
                print("batch %d attempt %d: %s" % (i, attempt, repr(e)[:120]), flush=True)
                if attempt < tries - 1:
                    time.sleep(20 * (attempt + 1))
        if part is None:
            raise IOError("%s: the batch of ids %d-%d of %d failed %d times: %r"
                          % (service, i, i + len(chunk) - 1, len(ids), tries, last))
        for p in part:
            out[str(p[key])] = p
        print("  %d of %d ids; %d rows" % (min(i + size, len(ids)), len(ids), len(out)), flush=True)
    return out


def votable_rows(raw):
    """Rows of a VOTable as dicts of Python values."""
    from astropy.io.votable import parse_single_table
    t = parse_single_table(io.BytesIO(raw)).to_table()
    return [dict(zip(t.colnames, [x.item() if hasattr(x, "item") else x for x in row])) for row in t]


GAIA_ID_MAX = 6917528997577384320


def gaia_by_range(query, n=24, maxrec=None):
    """Rows of a Gaia query too long for the archive's 60-s cap on an anonymous statement, run as N asynchronous jobs over equal ranges of source_id (which follows position on the sky): QUERY holds {lo} and {hi} for a BETWEEN on the joined source_id."""
    rows = []
    step = GAIA_ID_MAX // n + 1
    for k in range(n):
        rows += tap_async("gaia", query.format(lo=k * step, hi=min((k + 1) * step - 1, GAIA_ID_MAX)), maxrec=maxrec, poll=5)
    return rows


def tap_async(service, query, fmt="csv", maxrec=None, poll=15, limit=14400):
    """ADQL as an asynchronous TAP job (create, start, poll, read): rows as dicts (csv) or text."""
    import re
    url = TAP.get(service, service).rsplit("/sync", 1)[0] + "/async"
    params = dict(REQUEST="doQuery", LANG="ADQL", FORMAT=fmt, QUERY=query, PHASE="RUN")
    if maxrec is None and service != "datalab":
        maxrec = MAXREC
    if maxrec:
        params["MAXREC"] = maxrec
    req = urllib.request.Request(url, data=urllib.parse.urlencode(params).encode(), headers=UA)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    try:
        resp = urllib.request.build_opener(NoRedirect).open(req, timeout=120)
        job = resp.geturl()
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303):
            raise
        job = e.headers["Location"]
    if job.rstrip("/").endswith("/async"):
        raise IOError("%s returned no job location" % service)
    t0 = time.time()
    while True:
        phase = get(job + "/phase", timeout=60).decode().strip()
        if phase == "COMPLETED":
            break
        if phase in ("ERROR", "ABORTED"):
            err = get(job + "/error", timeout=60, soft=True) or get(job, timeout=60)
            # the reason follows the echoed query, after the markup
            msg = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", err.decode("utf-8", "replace"))).strip()
            raise IOError("%s job %s: %s" % (service, phase, msg[-600:]))
        if time.time() - t0 > limit:
            raise IOError("%s job still %s after %d s: %s" % (service, phase, limit, job))
        time.sleep(poll)
    text = get(job + "/results/result", timeout=1800).decode("utf-8-sig", errors="replace")
    if fmt in ("csv", "json") and text.lstrip()[:1] == "<":
        raise IOError("%s answered with an HTML or XML page instead of %s: %s" % (service, fmt, text[:200]))
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "json":
        return json.loads(text)
    return text


def pmap(fn, items, threads=8):
    """fn over items on a bounded thread pool, results in order."""
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(fn, items))


# --------------------------------------------------------------------------------------------------------------------
# remotezip: members of a remote zip archive read through HTTP byte ranges

HEADERS = dict(UA, Accept="*/*")


class RangeFile(io.RawIOBase):
    """Seekable read-only file over a ranged URL, cached in 1 MB blocks."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, method="HEAD", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, k):
        if k not in self.cache:
            lo, hi = k * self.block, min((k + 1) * self.block, self.size) - 1
            req = urllib.request.Request(self.url, headers=dict(HEADERS, Range="bytes=%d-%d" % (lo, hi)))
            with urllib.request.urlopen(req, timeout=300) as r:
                self.cache[k] = r.read()
        return self.cache[k]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            k, o = divmod(self.pos, self.block)
            b = self._block(k)[o:o + n]
            out += b
            self.pos += len(b)
            n -= len(b)
        return bytes(out)

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)


def open_zip(url):
    """zipfile.ZipFile over a remote archive."""
    return zipfile.ZipFile(RangeFile(url))


# --------------------------------------------------------------------------------------------------------------------
# archives: clients for the public archives, returning rows or bytes

# ---------------------------------------------------------------------------------------------- Gaia and SIMBAD


def gaia(query):
    """Rows of ADQL on the Gaia archive."""
    return tap("gaia", query)


def gaia_rvs_spectra(source_ids, outdir, batch=50):
    """Gaia DR3 mean RVS spectra (846-870 nm) written as CSV files in outdir; returns the paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for k in range(0, len(source_ids), batch):
        ids = [str(s) for s in source_ids[k:k + batch]]
        u = "https://gea.esac.esa.int/data-server/data?" + urllib.parse.urlencode(dict(
            RETRIEVAL_TYPE="RVS", RELEASE="Gaia DR3", DATA_STRUCTURE="DATAMODEL_STANDARD", FORMAT="csv",
            ID=",".join("Gaia DR3 " + i for i in ids)))
        b = get(u, timeout=300)
        if b[:2] == b"PK":
            z = zipfile.ZipFile(io.BytesIO(b))
            for n in z.namelist():
                base = os.path.basename(n)
                if base.startswith("RVS_Gaia_DR3_") and base.endswith(".csv"):
                    p = os.path.join(outdir, base)
                    open(p, "wb").write(z.read(n))
                    paths.append(p)
        elif len(ids) == 1 and b.startswith(b"source_id"):
            p = os.path.join(outdir, "RVS_Gaia_DR3_%s.csv" % ids[0])
            open(p, "wb").write(b)
            paths.append(p)
    return paths


def simbad(query):
    """Rows of ADQL on SIMBAD."""
    d = tap("simbad", query, fmt="json")
    names = [m["name"] for m in d["metadata"]]
    return [dict(zip(names, r)) for r in d["data"]]


def simbad_otypes(identifiers, batch=300):
    """Every SIMBAD object type of each identifier, own and listed: {identifier: set}."""
    out = {}
    ids = list(identifiers)
    for k in range(0, len(ids), batch):
        q = ("SELECT i.id, o.otype FROM ident AS i JOIN otypes AS o ON i.oidref = o.oidref WHERE i.id IN (%s)"
             % ",".join("'%s'" % x.replace("'", "''") for x in ids[k:k + batch]))
        for r in simbad(q):
            out.setdefault(r["id"], set()).add(str(r["otype"]).strip())
    return out


def simbad_types_near(ra, dec, radius_deg):
    """Counter of SIMBAD main object types within radius_deg of a position."""
    from collections import Counter
    q = ("SELECT b.otype FROM basic AS b WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', %f, %f, %f)) = 1"
         % (ra, dec, radius_deg))
    return Counter(str(r["otype"]).strip() for r in simbad(q))


# ------------------------------------------------------------------------------------------------ Solar System


def ssocard(number):
    """SsODNet taxonomy class, albedo and diameter (km) of a numbered minor body; None where unknown."""
    b = get("https://ssp.imcce.fr/webservices/ssodnet/api/ssocard/%d" % int(number), timeout=60, retries=2,
                soft=True)
    if not b:
        return None, None, None
    try:
        ph = json.loads(b)["parameters"]["physical"]
    except (ValueError, KeyError, TypeError):
        return None, None, None
    c = ph.get("taxonomy", {}).get("class")

    def val(d):
        return d.get("value") if isinstance(d, dict) else d
    return val(c), val(ph.get("albedo")), val(ph.get("diameter"))


# ----------------------------------------------------------------------------------------------- XMM-Newton


def xsa(query):
    """Rows of ADQL on the XMM-Newton Science Archive."""
    return tap("xsa", query)


# ------------------------------------------------------------------------------------------------------- ESO


def eso_spectra(ra, dec, collection, radius_deg=0.0083, covers=None):
    """ESO Phase 3 1D spectra of one instrument near a position; covers=(lo, hi) in metres keeps spanning spectra."""
    q = ("SELECT dp_id, target_name, t_min, snr, em_min, em_max, em_res_power, s_ra, s_dec FROM ivoa.ObsCore "
         "WHERE obs_collection='%s' AND dataproduct_type='spectrum' AND "
         "CONTAINS(POINT('ICRS',s_ra,s_dec),CIRCLE('ICRS',%f,%f,%f))=1" % (collection, ra, dec, radius_deg))
    if covers:
        q += " AND em_min < %g AND em_max > %g" % covers
    return tap("eso", q)


def archive_spectra(ra, dec, radius_deg=0.01):
    """Every archived spectrum of a position: ESO Phase 3 products, CFHT SPIRou planes through the CADC and Keck NIRSPEC frames through KOA, each with its source, instrument, identifier, time span (MJD), exposure (s) and wavelength span (m)."""
    rows = []
    q = ("SELECT dp_id, instrument_name, t_min, t_max, t_exptime, em_min, em_max, snr FROM ivoa.ObsCore WHERE dataproduct_type = 'spectrum' AND "
         "CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', %.5f, %.5f, %g)) = 1" % (ra, dec, radius_deg))
    for r in tap("eso", q):
        rows.append(dict(src="ESO", inst=r["instrument_name"], id=r["dp_id"], t0=float(r["t_min"]), t1=float(r["t_max"]), exp=num(r["t_exptime"]),
                         lo=num(r["em_min"]), hi=num(r["em_max"]), snr=num(r["snr"])))
    q = ("SELECT p.productID, p.time_bounds_lower, p.time_bounds_upper, p.time_exposure, p.energy_bounds_lower, p.energy_bounds_upper, p.dataRelease "
         "FROM caom2.Observation o JOIN caom2.Plane p ON o.obsID = p.obsID WHERE o.collection = 'CFHT' AND o.instrument_name = 'SPIRou' AND "
         "o.intent = 'science' AND INTERSECTS(CIRCLE('ICRS', %.5f, %.5f, %g), p.position_bounds) = 1" % (ra, dec, radius_deg))
    # one plane per exposure, the telluric-corrected product (t), and public ones alone
    for r in tap(CADC_TAP, q, method="POST", maxrec=20000):
        t0, t1 = num(r["time_bounds_lower"]), num(r["time_bounds_upper"])
        if t0 is not None and t1 is not None and r["productID"].endswith("t") and (r.get("dataRelease") or "")[:10] <= time.strftime("%Y-%m-%d"):
            rows.append(dict(src="CADC", inst="SPIRou", id=r["productID"], t0=t0, t1=t1, exp=num(r["time_exposure"]),
                             lo=num(r["energy_bounds_lower"]), hi=num(r["energy_bounds_upper"]), snr=None, release=r.get("dataRelease")))
    return rows + koa_spectra(ra, dec, radius_deg)


def koa_spectra(ra, dec, radius_deg=0.01):
    """Keck NIRSPEC spectrograph frames of a position through KOA, object frames of the spectrograph camera alone, each with its start (MJD), exposure (s) and wavelength span (m)."""
    q = ("SELECT koaid, mjd, elaptime, waveblue, wavered FROM koa_nirspec WHERE camera = 'spec' AND koaimtyp = 'object' AND "
         "CONTAINS(POINT('icrs', ra, dec), CIRCLE('icrs', %.5f, %.5f, %g)) = 1" % (ra, dec, radius_deg))
    rows = []
    for r in tap(KOA_TAP, q, method="POST", maxrec=20000):
        t0, exp = num(r["mjd"]), num(r["elaptime"]) or 0.0
        if t0 is None:
            continue
        lo, hi = num(r["waveblue"]), num(r["wavered"])
        rows.append(dict(src="KOA", inst="NIRSPEC", id=r["koaid"], t0=t0, t1=t0 + exp / 86400.0, exp=exp, lo=lo * 1e-6 if lo else None,
                         hi=hi * 1e-6 if hi else None, snr=None))
    return rows


CADC_TAP = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"
KOA_TAP = "https://koa.ipac.caltech.edu/TAP/sync"


def eso_download(dp_id, path):
    """Download one public ESO product to path; True when a FITS file arrived."""
    b = get("https://dataportal.eso.org/dataportal_new/file/" + dp_id, timeout=300)
    if b[:6] != b"SIMPLE":
        return False
    tmp = path + ".tmp"
    open(tmp, "wb").write(b)
    os.replace(tmp, path)
    return True


# --------------------------------------------------------------------------------------------------- IRSA


def irsa(query):
    """Rows of ADQL on IRSA."""
    return tap("irsa", query)


ALLWISE_NUM = ("ra", "dec", "w1mpro", "w2mpro", "w3mpro", "w3sigmpro", "w4mpro", "w4sigmpro", "w3snr", "w4snr",
               "k_m_2mass")


def allwise_nearest(ra, dec, radius_arcsec=6.0):
    """Nearest AllWISE source within the radius as a dict, or None."""
    import math
    q = ("SELECT ra, dec, w1mpro, w2mpro, w3mpro, w3sigmpro, w4mpro, w4sigmpro, w3snr, w4snr, k_m_2mass, ph_qual, "
         "cc_flags, ext_flg FROM allwise_p3as_psd WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, "
         "%.6f))=1" % (ra, dec, radius_arcsec / 3600.0))
    rows = tap("irsa", q)
    if not rows:
        return None
    for r in rows:
        for k in ALLWISE_NUM:
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
    cosd = math.cos(math.radians(dec))
    return min(rows, key=lambda r: math.hypot((r["ra"] - ra) * cosd, r["dec"] - dec))


def xmatch(rows, cat, radius_arcsec, id_col="source_id", ra_col="ra", dec_col="dec", timeout=900):
    """CDS XMatch of rows (id, ra, dec in deg) against a VizieR catalogue or SIMBAD within radius_arcsec; rows with angDist; a server error is retried four times, 30 seconds more apart each time."""
    import csv
    import requests
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([id_col, ra_col, dec_col])
    for r in rows:
        w.writerow([r[id_col], "%.7f" % float(r[ra_col]), "%.7f" % float(r[dec_col])])
    for attempt in range(5):
        resp = requests.post("http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync",
                             data=dict(request="xmatch", distMaxArcsec=radius_arcsec, RESPONSEFORMAT="csv",
                                       cat2=cat if cat == "simbad" else "vizier:" + cat, colRA1=ra_col, colDec1=dec_col),
                             files={"cat1": ("cat1.csv", buf.getvalue())}, timeout=timeout)
        if resp.status_code < 500 or attempt == 4:
            break
        time.sleep(30 * (attempt + 1))
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def spherex_images(ra, dec):
    """SPHEREx spectral images covering a position: uri, time, band; the archive is asked four times, ten seconds more apart each time, before an empty answer."""
    q = ("SELECT 'https://irsa.ipac.caltech.edu/' || a.uri AS uri, p.time_bounds_lower AS t, p.energy_bandpassname "
         "AS band FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid "
         "WHERE 1 = CONTAINS(POINT('ICRS', %f, %f), p.poly)" % (ra, dec))
    for attempt in range(4):
        try:
            return tap("irsa", q)
        except Exception:
            time.sleep(10 * (attempt + 1))
    return []


def spherex_cutout(uri, ra, dec, size_deg=0.05):
    """Bytes of a SPHEREx cutout (IMAGE, FLAGS, VARIANCE, ZODI, PSF, WCS-WAVE), about 5 MB."""
    return get(uri + "?center=%f,%fd&size=%g" % (ra, dec, size_deg), timeout=180, retries=3)


SPHEREX_SIDE = 2040
SPHEREX_PLANE = (SPHEREX_SIDE ** 2 * 4 + 2879) // 2880 * 2880        # bytes of one 2040 x 2040 32-bit plane
SPHEREX_TAIL = 2880 * 4                  # the WCS-WAVE table, last in the file, lies within this many final bytes


def _fits_cards(raw):
    return [raw[i:i + 80].decode("ascii", "replace") for i in range(0, len(raw) - 79, 80)]


def _fits_header_length(raw):
    """Length of the header at the start of raw (multiple of 2880), or None without END."""
    end = next((i for i, c in enumerate(_fits_cards(raw)) if c.rstrip() == "END"), None)
    return None if end is None else ((end + 1) * 80 + 2879) // 2880 * 2880


def _fits_keys(cards):
    kv = {}
    for c in cards:
        if c[8:10] == "= ":
            kv.setdefault(c[:8].strip(), c[10:].split("/")[0].strip().strip("'").strip())
    return kv


def _fits_signature(cards):
    """XTENSION, BITPIX, NAXIS, axis lengths and EXTNAME from a header's first cards."""
    kv = _fits_keys(cards)
    n = int(kv.get("NAXIS", 0) or 0)
    sig = (kv.get("XTENSION"), kv.get("BITPIX"), n) + tuple(kv.get("NAXIS%d" % k) for k in range(1, n + 1))
    return sig, kv.get("EXTNAME")


def _fits_data_length(kv):
    """Bytes of an HDU's data unit with heap, padded to 2880."""
    n = 0
    if int(kv.get("NAXIS", 0) or 0):
        n = 1
        for k in range(1, int(kv["NAXIS"]) + 1):
            n *= int(kv["NAXIS%d" % k])
    return (abs(int(kv.get("BITPIX", 8))) // 8 * n + int(kv.get("PCOUNT", 0) or 0) + 2879) // 2880 * 2880


def spherex_layout(uri, off=0):
    """HDU offsets of a SPHEREx image from byte off to WCS-WAVE, walked by byte ranges: {EXTNAME: (offset, header, data, signature)}."""
    out = {}
    while "WCS-WAVE" not in out:
        raw, hlen = b"", None
        while hlen is None:
            chunk = get_ranges(uri, [(off + len(raw), off + len(raw) + 28799)])[0]
            if not chunk:
                raise IOError("%s ends before its WCS-WAVE extension" % uri)
            raw += chunk
            hlen = _fits_header_length(raw)
        cards = _fits_cards(raw[:hlen])
        sig, name = _fits_signature(cards)
        dlen = _fits_data_length(_fits_keys(cards))
        out[name or "PRIMARY"] = (off, hlen, dlen, sig)
        off += hlen + dlen
    return out


def _rice1(blocksize, bytepix, tilesize):
    try:
        from astropy.io.fits.hdu.compressed._codecs import Rice1
    except ImportError:
        from astropy.io.fits._tiled_compression.codecs import Rice1
    return Rice1(blocksize=blocksize, bytepix=bytepix, tilesize=tilesize)


def _spherex_rows(start, x0, y0, n):
    """Byte ranges of an n x n block at (x0, y0) of a 2040-pixel 32-bit plane starting at start."""
    return [(start + ((y0 + j) * SPHEREX_SIDE + x0) * 4, start + ((y0 + j) * SPHEREX_SIDE + x0 + n) * 4 - 1)
            for j in range(n)]


def _spherex_wave(tail):
    """WCS-WAVE table HDU from the file's final bytes, or None."""
    from astropy.io import fits
    for k in range(0, len(tail) - 2879, 2880):
        cards = _fits_cards(tail[k:k + 2880])
        if cards and cards[0].startswith("XTENSION= 'BINTABLE'"):
            hlen = _fits_header_length(tail[k:])
            if hlen and _fits_keys(_fits_cards(tail[k:k + hlen])).get("EXTNAME") == "WCS-WAVE":
                return fits.BinTableHDU.fromstring(tail[k:])
    return None


def spherex_pixels(uri, ra, dec, half=5):
    """IMAGE, FLAGS, VARIANCE and ZODI pixels within half of a position, read by byte ranges (about 100 kB), with header and wave; None near the edge."""
    import warnings
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    lo = 2880 * 6 + SPHEREX_PLANE            # a primary header and an IMAGE header of five blocks or more come first
    (head, span), total = get_ranges(uri, [(0, 2880 * 12 - 1), (lo, lo + 2880 * 24 - 1)], size=True)
    plen = _fits_header_length(head)
    ilen = _fits_header_length(head[plen:]) if plen else None
    if ilen is None:
        lay = spherex_layout(uri)
        plen, ilen = lay["IMAGE"][:2]
        head = get_ranges(uri, [(0, plen + ilen - 1)])[0]
    icards = _fits_cards(head[plen:plen + ilen])
    if _fits_signature(icards) != (("IMAGE", "-32", 2, str(SPHEREX_SIDE), str(SPHEREX_SIDE)), "IMAGE"):
        raise IOError("%s: the first extension is not a %d-pixel IMAGE plane" % (uri, SPHEREX_SIDE))
    hdr = fits.Header.fromstring(head[plen:plen + ilen].decode("ascii", "replace"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = WCS(hdr).world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    x, y = float(x), float(y)
    if not (half <= x < SPHEREX_SIDE - half - 1 and half <= y < SPHEREX_SIDE - half - 1):
        return None
    x0, y0, n = int(round(x)) - half, int(round(y)) - half, 2 * half + 1
    # FLAGS header
    foff = plen + ilen + SPHEREX_PLANE
    fraw = span[foff - lo:] if lo <= foff < lo + len(span) else b""
    flen = _fits_header_length(fraw) if fraw else None
    if flen is None:
        fraw = get_ranges(uri, [(foff, foff + 2880 * 24 - 1)])[0]
        flen = _fits_header_length(fraw)
    fkv = _fits_keys(_fits_cards(fraw[:flen])) if flen else {}
    if fkv.get("EXTNAME") != "FLAGS":
        raise IOError("%s: no FLAGS extension after the IMAGE plane" % uri)
    fdata = foff + flen
    side = str(SPHEREX_SIDE)
    if (fkv.get("XTENSION"), fkv.get("BITPIX"), fkv.get("NAXIS1"), fkv.get("NAXIS2")) == ("IMAGE", "32", side, side):
        rice = None
    elif (fkv.get("XTENSION") == "BINTABLE" and fkv.get("ZIMAGE") == "T" and fkv.get("ZCMPTYPE") == "RICE_1"
          and fkv.get("ZBITPIX") == "32" and fkv.get("ZNAXIS1") == side and fkv.get("ZNAXIS2") == side
          and fkv.get("ZTILE1") == side and fkv.get("ZTILE2") == "1" and fkv.get("NAXIS1") == "8"):
        par = {fkv["ZNAME%d" % k]: int(fkv["ZVAL%d" % k]) for k in range(1, 10) if "ZNAME%d" % k in fkv}
        heap = fdata + int(fkv.get("THEAP") or 8 * SPHEREX_SIDE)
        rice = (heap, _rice1(par.get("BLOCKSIZE", 32), par.get("BYTEPIX", 4), SPHEREX_SIDE))
    else:
        raise IOError("%s: FLAGS is neither a 32-bit plane nor RICE_1 compressed by rows" % uri)
    after = fdata + _fits_data_length(fkv)
    plane = ("IMAGE", "-32", 2, side, side)
    vlen = zlen = 11520  # VARIANCE and ZODI header lengths in every file read so far
    for attempt in range(2):
        voff = after
        zoff = voff + vlen + SPHEREX_PLANE
        ranges = [(voff, voff + 1599), (zoff, zoff + 1599), (total - SPHEREX_TAIL, total - 1)]
        ranges += _spherex_rows(plen + ilen, x0, y0, n)
        ranges += _spherex_rows(voff + vlen, x0, y0, n)
        ranges += _spherex_rows(zoff + zlen, x0, y0, n)
        ranges += _spherex_rows(fdata, x0, y0, n) if rice is None else [(fdata + y0 * 8, fdata + (y0 + n) * 8 - 1)]
        got = get_ranges(uri, ranges)
        if (_fits_signature(_fits_cards(got[0])) == (plane, "VARIANCE")
                and _fits_signature(_fits_cards(got[1])) == (plane, "ZODI")):
            break
        lay = spherex_layout(uri, after)
        if not {"VARIANCE", "ZODI"} <= set(lay) or lay["ZODI"][0] != lay["VARIANCE"][0] + lay["VARIANCE"][1] + \
                SPHEREX_PLANE or lay["VARIANCE"][3] != plane or lay["ZODI"][3] != plane:
            raise IOError("%s: VARIANCE and ZODI do not follow FLAGS as %d-pixel planes" % (uri, SPHEREX_SIDE))
        vlen, zlen = lay["VARIANCE"][1], lay["ZODI"][1]
    else:
        raise IOError("%s does not have the SPHEREx spectral image layout" % uri)
    wave = _spherex_wave(got[2])
    if wave is None:
        lay = spherex_layout(uri, after)
        w = lay["WCS-WAVE"]
        wave = fits.BinTableHDU.fromstring(get_ranges(uri, [(w[0], w[0] + w[1] + w[2] - 1)])[0])
    out = dict(header=hdr, wave=wave, x0=x0, y0=y0, x=x - x0, y=y - y0)
    for i, name in enumerate(("IMAGE", "VARIANCE", "ZODI")):
        out[name] = np.frombuffer(b"".join(got[3 + i * n:3 + (i + 1) * n]), dtype=">f4").reshape(n, n)
    if rice is None:
        out["FLAGS"] = np.frombuffer(b"".join(got[3 + 3 * n:3 + 4 * n]), dtype=">i4").reshape(n, n)
    else:
        desc = np.frombuffer(got[3 + 3 * n], dtype=">i4").reshape(n, 2)
        if (desc[:, 0] <= 0).any():
            raise IOError("%s: an empty FLAGS tile" % uri)
        tiles = get_ranges(uri, [(rice[0] + int(o), rice[0] + int(o) + int(m) - 1) for m, o in desc])
        out["FLAGS"] = np.array([rice[1].decode(np.frombuffer(t, dtype=np.uint8))[x0:x0 + n] for t in tiles],
                                dtype=np.int64)
    return out


# ------------------------------------------------------------------------------------------- GALAH, SDSS, LAMOST


def galah(query):
    """Rows of ADQL on Data Central."""
    return tap("datacentral", query)


def galah_spectrum(sobject_id, camera):
    """GALAH DR4 spectrum of one camera (B, V, R, I) from Data Central's SODA links: normalised, on barycentric wavelengths with the star's own velocity left in (the catalogue's rv_comp_1 takes it to rest; IDX 1 serves the counts); header RV is the barycentric correction."""
    u = ("https://datacentral.org.au/vo/slink/links?ID=%s&DR=galah_dr4&IDX=0&FILT=%s&RESPONSEFORMAT=fits"
         % (sobject_id, camera))
    return get(u, timeout=120)


def sdss_sql(sql, release="dr17"):
    """Rows of SQL on the SDSS SkyServer."""
    u = "https://skyserver.sdss.org/%s/SkyServerWS/SearchTools/SqlSearch?" % release + urllib.parse.urlencode(
        dict(cmd=sql, format="csv"))
    text = get(u, timeout=600).decode("utf-8", errors="replace")
    lines = text.splitlines()
    if lines and lines[0].startswith("#Table"):
        lines = lines[1:]
    import csv
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


AAT_API = "https://archives.datacentral.org.au/api/"


def aat_post(path, body, timeout=600):
    """Bytes of a JSON POST to Data Central's AAT archive API."""
    req = urllib.request.Request(AAT_API + path, data=json.dumps(body).encode(), method="POST",
                                 headers=dict(UA, **{"Content-Type": "application/json", "Accept": "application/json"}))
    return urllib.request.urlopen(req, timeout=timeout).read()


def aat_night(date):
    """Every row of one night's AAT log (YYYY-MM-DD): file, detector, class, object, exposure and header. The first page answers the date and each next one the query's uuid with the last row's aat_id."""
    page = json.loads(aat_post("query/", {"obs_date": date}))[0]
    rows = list(page["results"])
    seen = {r["aat_id"] for r in rows}
    while page.get("next") and len(rows) < page["count"]:
        nxt = page["next"]["aat_id"]
        page = json.loads(aat_post("query/", {"uuid": page["uuid"], "next": nxt, "previous": nxt}))[0]
        new = [r for r in page["results"] if r["aat_id"] not in seen]
        if not new:
            break
        rows += new
        seen |= {r["aat_id"] for r in new}
    return rows


def aat_fetch(rows, out):
    """The rows' raw FITS files from the AAT archive into OUT, fetching those not there yet (the download is a gzipped tar); their paths in the rows' order."""
    import tarfile
    os.makedirs(out, exist_ok=True)
    need = [r for r in rows if not os.path.exists(os.path.join(out, os.path.basename(r["filename"])))]
    if need:
        tf = tarfile.open(fileobj=io.BytesIO(aat_post("download/", {"aat_ids": [int(r["aat_id"]) for r in need]}, timeout=3600)), mode="r:*")
        for m in tf.getmembers():
            if m.isfile():
                open(os.path.join(out, os.path.basename(m.name)), "wb").write(tf.extractfile(m).read())
    return [os.path.join(out, os.path.basename(r["filename"])) for r in rows]


def apogee_apstar_url(telescope, field, apogee_id):
    """URL of an APOGEE DR17 apStar file (visits in the rest frame, vacuum)."""
    return ("https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/%s/%s/apStar-dr17-%s.fits"
            % (telescope, field, apogee_id))


def datalab(query):
    """Rows of ADQL on NOIRLab Astro Data Lab (Postgres: NaN sorts high, booleans compare with 't')."""
    return tap("datalab", query)


def desi_spectra(targetids, include=("targetid", "flux", "ivar", "mask", "model", "wave_sigma"), release="DESI-DR1"):
    """DESI coadded spectra of each target's primary spectrum from SPARCL, with the included fields."""
    from sparcl.client import SparclClient
    c = SparclClient()
    found = c.find(outfields=["sparcl_id", "targetid", "specprimary"],
                   constraints={"targetid": list(targetids), "data_release": [release]}, limit=4 * len(targetids))
    ids = [r["sparcl_id"] for r in found.records if r.get("specprimary")]
    return c.retrieve(uuid_list=ids, include=list(include), dataset_list=[release], limit=len(ids) + 10).records


def lamost_cone(ra, dec, radius_deg):
    """LAMOST DR10 v2.0 low-resolution rows within a radius."""
    u = "https://www.lamost.org/dr10/v2.0/voservice/conesearch?ra=%f&dec=%f&sr=%g&format=csv" % (ra, dec, radius_deg)
    import csv
    return list(csv.DictReader(io.StringIO(get(u).decode("utf-8-sig", errors="replace"))))


# --------------------------------------------------------------------------------------- radio and millimetre


def alma(query):
    """Rows of ADQL on the ALMA archive."""
    return tap("alma", query, maxrec=2000000)


def alma_datalink(member_ous_uid):
    """Files of one ALMA observing unit with sizes."""
    from astropy.io.votable import parse_single_table
    u = "https://almascience.eso.org/datalink/sync?" + urllib.parse.urlencode(dict(ID=member_ous_uid))
    t = parse_single_table(io.BytesIO(get(u, timeout=180))).to_table(use_names_over_ids=True)
    return [dict(url=str(r["access_url"]), bytes=r["content_length"], semantics=str(r["semantics"])) for r in t]


def casda(query):
    """Rows of ADQL on CASDA."""
    return tap("casda", query)


HEASARC_TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"


def heasarc(query, maxrec=200000):
    """Rows of ADQL on HEASARC's Xamin TAP service as dicts; the service answers in VOTable only, over GET."""
    return tap(HEASARC_TAP, query, fmt="votable", method="GET", maxrec=maxrec)


def vizier(catalog):
    """All tables of a VizieR catalogue, all columns, no row limit."""
    from astroquery.vizier import Vizier
    return Vizier(columns=["**"], row_limit=-1).get_catalogs(catalog)


def exoplanets(query):
    """Rows of ADQL on the NASA Exoplanet Archive."""
    return tap("exoplanets", query)


NIST_LINES = ("https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=%s&output_type=0&low_w=%g&upp_w=%g&unit=0"
              "&submit=Retrieve+Data&de=0&plot_out=0&I_scale_type=1&format=3&line_out=0&remove_js=on&en_unit=0&output=0"
              "&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2"
              "&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1"
              "&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on")


def nist_lines(spectrum, lo, hi, cache_dir=None):
    """NIST ASD lines of one spectrum between lo and hi (A): ion, vacuum wavelength, intensity, Aki, lower-level energy."""
    import csv
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "%s_%g_%g.tsv" % (spectrum.replace(" ", "_"), lo, hi))
    if path and os.path.exists(path):
        t = open(path, encoding="utf-8").read()
    else:
        b = get(NIST_LINES % (urllib.parse.quote_plus(spectrum), lo, hi), timeout=300, retries=3, soft=True)
        t = b.decode("utf-8", "replace") if b else ""
        if "Input Error" in t or "<html" in t[:200].lower():
            t = ""
        if path and t:
            open(path, "w", encoding="utf-8").write(t)
    from . import science
    out = []
    for r in csv.DictReader(io.StringIO(t), delimiter="\t"):
        r = {k.strip('"'): (v or "").strip('"') for k, v in r.items() if k}
        w = r.get("obs_wl_vac(A)") or r.get("ritz_wl_vac(A)")
        air = not w
        w = w or r.get("obs_wl_air(A)") or r.get("ritz_wl_air(A)")
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if air and w > 2000.0:
            w = float(science.air_to_vac(w))
        out.append(dict(ion=spectrum, lam=w, intens=r.get("intens", ""), aki=r.get("Aki(s^-1)", ""),
                        ei=r.get("Ei(cm-1)", "")))
    return out


BL_FILE = re.compile(r"href=\"([^\"]*guppi_(\d+)_(\d+)_([A-Za-z0-9+%_-]+?)_(\d{4})\.gpuspec\.0000\.h5)\"")


def bl_query_files(target):
    """Breakthrough Listen open-data files for a target."""
    import json
    d = json.loads(get("http://seti.berkeley.edu/opendata/api/query-files?target=%s" % target))
    return d.get("data", [])


def bl_scans(directory_url):
    """High-resolution scans in a BL directory in time order: (mjd, seconds, target, url)."""
    page = get(directory_url, timeout=60).decode(errors="replace")
    return sorted({(int(m[1]), int(m[2]), m[3], directory_url + m[0]) for m in BL_FILE.findall(page)})


def bl_cadences(directory_url):
    """ABACAD cadences in a BL directory: six (target, url) with the target at 0, 2 and 4."""
    s = bl_scans(directory_url)
    out = []
    for i in range(len(s) - 5):
        w = s[i:i + 6]
        a = w[0][2]
        if w[2][2] == a and w[4][2] == a and all(x[2] != a for x in (w[1], w[3], w[5])):
            out.append([(x[2], x[3]) for x in w])
    return out


# ----------------------------------------------------------------------------------------------------- DASCH


def dasch_lightcurves(ra, dec, session_dir, radius_arcsec=1800, n_comp=10, dmag=1.0, dcolor=0.5, retries=3):
    """DASCH light curves of a star and up to n_comp comparison stars of similar magnitude and colour on the same plates."""
    import contextlib
    import shutil
    import numpy as np
    import daschlab
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    daschlab.REFCAT_RADIUS = radius_arcsec * u.arcsec
    sink = io.StringIO()
    for attempt in range(retries):
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                sess = daschlab.open_session(session_dir)
                sess.select_target(ra_deg=ra, dec_deg=dec)
                sess.select_refcat("apass")
                rc = sess.refcat()
            break
        except Exception:
            if attempt == retries - 1:
                raise
            shutil.rmtree(session_dir, ignore_errors=True)
            time.sleep(10 * (attempt + 1))
    sep = rc["pos"].separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec
    i0 = int(np.argmin(sep))
    if sep[i0] > 20:
        raise ValueError("no reference-catalogue star within 20 arcsec")
    m0, c0 = float(rc["stdmag"][i0]), float(rc["color"][i0])
    comp = [i for i in range(len(rc)) if i != i0 and np.isfinite(rc["stdmag"][i]) and np.isfinite(rc["color"][i])
            and abs(float(rc["stdmag"][i]) - m0) < dmag and abs(float(rc["color"][i]) - c0) < dcolor]
    comp = sorted(comp, key=lambda i: sep[i])[:n_comp]

    def lc(i):
        for attempt in range(retries):
            try:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    x = sess.lightcurve(i)
                    x.apply_standard_rejections()
                mag = np.asarray(x["magcal_magdep"].filled(np.nan) if hasattr(x["magcal_magdep"], "filled")
                                 else x["magcal_magdep"], float)
                ok = np.isfinite(mag) & (np.asarray(x["reject"], int) == 0)
                return (x["time"].jyear[ok], mag[ok], np.asarray(x["series"]).astype(str)[ok])
            except Exception:
                time.sleep(5 * (attempt + 1))
        return None
    return dict(target=lc(i0), comps=[c for c in (lc(i) for i in comp) if c is not None], mag=m0, color=c0)


# --------------------------------------------------------------------------------------------------------------------
# tess: Kepler and TESS light curves from the STScI AWS mirror and MAST HLSPs; sectors from TESScut or tess-point

S3 = "https://stpubdata.s3.amazonaws.com/"
HLSP = "https://archive.stsci.edu/hlsps/"
TESSCUT = "https://mast.stsci.edu/tesscut/api/v0.1/sector?ra=%f&dec=%f&radius=1m"
FLUX_COLUMNS = ("PDCSAP_FLUX", "KSPSAP_FLUX", "DET_FLUX", "SAP_FLUX")


def tic_path(tic):
    """Four-part MAST directory of a TIC id."""
    t16 = "%016d" % int(tic)
    return "%s/%s/%s/%s" % (t16[0:4], t16[4:8], t16[8:12], t16[12:16])


def tic_near(ra, dec, radius_arcsec=3.0):
    """Nearest TIC 8.2 id within the radius, or None."""
    best = None
    for m in xmatch([dict(source_id="s", ra=ra, dec=dec)], "IV/39/tic82", radius_arcsec):
        if best is None or float(m["angDist"]) < best[1]:
            best = (int(m["TIC"]), float(m["angDist"]))
    return best[0] if best else None


def sectors(ra, dec):
    """Sectors covering a position: TESScut, else tess-point for sectors ended two months ago."""
    b = get(TESSCUT % (ra, dec), timeout=60, retries=2, soft=True)
    if b:
        return sorted({int(x["sector"]) for x in json.loads(b)["results"]})
    try:
        from tess_stars2px import tess_stars2px_function_entry
    except ImportError:
        return []
    import datetime
    # sector 1 began 2018 July 25; sectors average 27.4 days
    last = int((datetime.date.today() - datetime.date(2018, 7, 25)).days / 27.4) - 2
    return sorted({int(s) for s in tess_stars2px_function_entry(0, ra, dec)[3] if 0 < int(s) <= last})


def s3_keys(prefix, pattern, strict=False):
    """Keys under an S3 prefix matching the pattern; with STRICT a listing that fails raises IOError instead of reading as empty."""
    lst = get(S3 + "?list-type=2&prefix=" + prefix, timeout=120, retries=3, soft=True, pause=0)
    if lst is None and strict:
        raise IOError("S3 listing failed: " + prefix)
    if not lst:
        return []
    return [k for k in re.findall(r"<Key>([^<]*)</Key>", lst.decode()) if re.search(pattern, k)]


def spoc_keys(tic, sector, fast=False, strict=False):
    """S3 keys of a star's SPOC light curves in one sector, two-minute or twenty-second; STRICT as for s3_keys."""
    prefix = "tess/public/tid/s%04d/%s/" % (sector, tic_path(tic))
    return s3_keys(prefix, r"fast-lc\.fits$" if fast else r"-s_lc\.fits$", strict)


def spoc_bytes(tic, sector, fast=False, strict=False):
    """SPOC light curve of one sector as FITS bytes, or None where the archive holds none; with STRICT a listing or download that fails raises IOError instead."""
    keys = spoc_keys(tic, sector, fast, strict)
    if not keys:
        return None
    b = get(S3 + keys[0], timeout=300, retries=3, soft=True, pause=2)
    if b is None and strict:
        raise IOError("S3 download failed: " + keys[0])
    return b if b and b.startswith(b"SIMPLE") else None


def ffi_urls(tic, sector):
    """[(source, URL)] of a star's FFI light curves in one sector, TESS-SPOC before QLP."""
    path, t16 = tic_path(tic), "%016d" % int(tic)
    return [("tess-spoc", HLSP + "tess-spoc/s%04d/target/%s/hlsp_tess-spoc_tess_phot_%s-s%04d_tess_v1_lc.fits"
             % (sector, path, t16, sector)),
            ("qlp", HLSP + "qlp/s%04d/%s/hlsp_qlp_tess_ffi_s%04d-%s_tess_v01_llc.fits" % (sector, path, sector, t16))]


def ffi_bytes(tic, sector, retries=1):
    """FFI light curve of one sector as (source, bytes), or (None, None)."""
    for source, u in ffi_urls(tic, sector):
        b = get(u, timeout=120, retries=retries, soft=True, pause=1)
        if b and b.startswith(b"SIMPLE"):
            return source, b
    return None, None


def read(fits_bytes, columns=FLUX_COLUMNS, good_only=True, normalise=True):
    """(time, flux, quality) from the first flux column present; flagged cadences dropped when good_only."""
    from astropy.io import fits
    with fits.open(io.BytesIO(fits_bytes)) as h:
        x = h[1].data
        names = x.columns.names
        col = next(c for c in columns if c in names)
        t, f = np.asarray(x["TIME"], float), np.asarray(x[col], float)
        q = np.asarray(x["QUALITY"] if "QUALITY" in names else x["SAP_QUALITY"], int)
    ok = np.isfinite(t) & np.isfinite(f)
    if good_only:
        ok &= q == 0
    t, f, q = t[ok], f[ok], q[ok]
    if normalise and len(f):
        f = f / np.median(f)
    return t, f, q


def curves(ra, dec, ffi=True, spoc=False, min_points=100):
    """TIC id and light curves of the star at a position: [(sector, source, time, flux)], SPOC first when spoc."""
    tic = tic_near(ra, dec)
    if tic is None:
        return None, []
    out = []
    for s in sectors(ra, dec):
        got = None
        if spoc:
            b = spoc_bytes(tic, s)
            if b:
                got = ("spoc", b)
        if got is None and ffi:
            src, b = ffi_bytes(tic, s)
            if b:
                got = (src, b)
        if got is None:
            continue
        t, f, _ = read(got[1])
        if len(t) > min_points:
            out.append((s, got[0], t, f))
    return tic, out


def modulation(lcs, pmin=0.07, pmax=15.0, nfreq=6000):
    """Best Lomb-Scargle period (d), its power and the median 2-98 per cent range over sectors."""
    from astropy.timeseries import LombScargle
    rows = []
    for _, _, t, f in lcs:
        fr = np.linspace(1 / pmax, 1 / pmin, nfreq)
        pw = LombScargle(t, f).power(fr)
        k = int(np.argmax(pw))
        lo, hi = np.percentile(f, [2, 98])
        rows.append((1 / fr[k], float(pw[k]), float(hi - lo)))
    if not rows:
        return None
    best = max(rows, key=lambda r: r[1])
    return dict(period=float(best[0]), power=best[1], range=float(np.median([r[2] for r in rows])), sectors=len(rows))


def kepler_keys(kic, short=True):
    """S3 keys of a Kepler star's light curves, short or long cadence."""
    k9 = "%09d" % int(kic)
    return s3_keys("kepler/public/lightcurves/%s/%s/" % (k9[:4], k9), r"_slc\.fits$" if short else r"_llc\.fits$")


# --------------------------------------------------------------------------------------------------------------------
# vetting: vet codes of the coverage store and the rules mapping explanations to codes

VET = {
    # spectra
    "fibre-fixed": "a line one fibre holds at the same wavelength on other plates or stars: the detector's",
    "neighbouring-fibres": "a feature fibres beside it on the same spectrograph show in other stars",
    "shared-feature": "a feature other stars show at the same rest or observed wavelength",
    "detector-pixel": "a peak narrower than the instrumental profile: detector pixels",
    "off-trace": "charge in the raw frame away from the fibre's trace, where light through the spectrograph does not fall",
    "bad-pixel": "masked, variance-inflated or dead pixels at the peak",
    "detector-edge": "a feature at the end of an arm or order, or at a detector gap",
    "instrument-frame": "a feature whose nights agree in the instrument's frame and not in the star's",
    "not-one-feature": "peaks of the exposures that lie apart by more than half a resolution element",
    "cosmic-ray": "a feature in one exposure or cadence with the others flat",
    "one-camera": "a line that one spectrograph camera shows and the other, covering the same wavelength, lacks",
    "defective-spectrum": "a spectrum holding clean lines of both signs, which no source of light makes",
    "hot-pixel": "a detector site that hosts the strongest events of several targets",
    "detector-wide": "an event that other sources across the detector share",
    "stellar-blend": "absorption that other stars of the spectrograph share at the line, a blend whose strength varies",
    "broad-feature": "a feature wider than an unresolved line",
    "line-core": "a peak inside a Balmer or He I line where the comparison template dips",
    "background-galaxy": "the lines or light of a background galaxy or active nucleus",
    # radio
    "off-source": "the same class of signal in the OFF scans",
    "sidelobe-transmitter": "a carrier that recurs on one drift in other targets' pointings of the session",
    "broadcast-band": "a frequency inside a broadcast band",
    "sky-sidereal": "the Galaxy's sidereal swing, which channels across the band share",
    "gnss": "satellite navigation signals, sidereal through the constellation's orbits",
    "auroral-kilometric-radiation": "auroral kilometric radiation",
    "plasma-line": "the solar wind's plasma line",
    "terrestrial-hf": "terrestrial HF leakage",
    "spacecraft-local": "the spacecraft's own emission",
    "spacecraft-anomaly": "a spacecraft anomaly",
    "instrument-mode": "a receiver mode that measures some channels and interpolates the rest",
    "spacecraft-timing": "a period of the spacecraft's operations",
    "natural-source": "a source its discovery papers give a natural nature",
    # photometry
    "single-epoch": "a dip or excess resting on one epoch, transit or pass, with its neighbours undimmed",
    "single-band": "a dip in one band that the other bands, taken at the same transits, do not share",
    "field-event": "an event other stars of the field show at the same time",
    "binary": "a binary: multiplicity flags, astrometric excess, velocity scatter or blended double lines",
    "eclipsing-binary": "an eclipsing or heartbeat binary",
    "variable-star": "a catalogued or measured variable star or pulsator",
    "rotation": "rotational modulation, a spotted rotator or a period at the rotation period or its multiple",
    "chromatic": "modulation that changes colour, as spots do",
    "burst": "an isolated burst feeding a phase peak, not a periodic train",
    "flare": "a flare on a flare star",
    "not-periodic": "a phase scatter above a clock's",
    "timing-variation": "a published transit-timing variation",
    "data-edge": "a step at the first or last maximum of a data source",
    "detrending": "a shoulder made by the detrending gap and the transit's timing and duration",
    "fit-boundary": "a fit pinned at the edge of its model",
    "fit-degeneracy": "a value inside the pile-up the fits themselves produce",
    "unstable-profile": "a profile unstable between sectors, of low S/N, with a secondary eclipse or too few events",
    "unconstrained-geometry": "a geometry the catalogue leaves unconstrained",
    "single-record": "a value resting on one catalogue record",
    # planetary systems
    "near-resonance": "a period ratio within 1 per cent of a small-integer ratio, where the resonant phases rather than random ones decide stability",
    "estimated-masses": "planet masses estimated from radii rather than measured",
    "loose-eccentricities": "eccentricities measured no better than 0.1",
    "twin-flags": "stable configurations observed again with the same errors flagged as well",
    "catalogue-crossing": "adjacent orbits that already cross at the catalogue's own values",
    "unmeasured-eccentricities": "eccentricities the catalogue gives without errors or as limits",
    "mass-limits": "masses the catalogue gives as upper limits",
    # stars and catalogues
    "active-star": "an active star: X-ray source, chromospheric emission or emission lines",
    "evolved": "an evolved star or host, off the main sequence",
    "young-star": "a young star or star-forming region",
    "moving-group": "a member of a catalogued moving group or association, whose shared velocity places it",
    "natural-type": "a SIMBAD type that explains the signal",
    "white-dwarf": "a white dwarf or white-dwarf companion",
    "dust": "warm or hot dust, a debris disc or an infrared excess of dust",
    "companion": "a known or red companion",
    "blend": "light of a neighbour, a background star or a third body in the aperture or beam",
    "crowding": "a depth or amplitude difference made by a pipeline's correction for neighbours' light, absent from the uncorrected flux",
    "interstellar": "interstellar dust or gas in the beam or along the sightline",
    "artefact": "a catalogue or imaging artefact: ghost, spike, single-exposure detection or a catalogue that reads high",
    "catalogue-dropped": "a source one catalogue's processing dropped and another's holds: a faint neighbour inside a bright star's image, a fast mover the "
                         "cross-match lost, or too few transits in the shorter span",
    "event-boundary": "a start, peak or end set by the catalogue's own detection boundary, so the shape is the catalogue's and not the event's",
    "our-launch": "a body of our own launching, identified in the database by its orbit and its response to sunlight",
    "atmospheric-line": "an absorption line of a known atmospheric species at its wavelength",
    "planetary-encounter": "an orbit changed by a close approach to a planet on record",
    "comet-family": "a member of a published comet family of one parent body",
    "co-orbital": "a body on a horseshoe or wide-librating co-orbital path about a planet, which the 1:1 resonance holds for a time",
    "vela-like": "a young pulsar whose large glitches recur at the crust's critical lag, as Vela's do",
    "perpendicular-peak": "an orbit in the pile-up of measured obliquities near 90 deg (Albrecht et al. 2021), a natural population feature with tidal and secular mechanisms proposed",
    "misidentified": "a spectrum of a different star from the catalogued one, by fibre placement or crossmatch, shown by temperatures that disagree",
    "bright-neighbour": "within the artefact zone of a source bright enough to spoil the image around it",
    "spider-orbit": "a black-widow or redback pulsar whose orbital period wanders as its companion's shape changes (gravitational quadrupole coupling), swamping relativity's decay",
    "arm-tangent": "a line of sight along a spiral arm's tangent, where a whole arm's young pulsars, nebulae and star-forming regions pile up",
    "cohesive": "a spin that needs less tensile strength than solid rock holds, about 100 kPa, which a coherent fragment or a cohesive rubble pile supplies",
    "sensor-error": "a speed or direction within the published errors of the sensor that measured it",
    "radio-hotspot": "a compact radio component away from its source's low-resolution centroid, a knot or hotspot of a larger radio galaxy whose host lies elsewhere",
    "out-of-reach": "a known body now too faint or too far for the surveys that would see it",
    "abiotic-chemistry": "a gas mixture that published abiotic models of the planet make, hydrogen-rich mini-Neptune and magma-ocean chemistry keeping CH4 and CO2 without CO",
    "stellar-contamination": "a spectral feature that spots and faculae on a cool host star imprint on a transit spectrum",
    "methane-overlap": "a dimethyl sulfide reading beside methane, whose 3.3 um band the DMS template overlaps, as Earth's own transit spectrum reads DMS it does not hold",
    "deep-counterpart": "a counterpart in deeper imaging than the surveys that missed it",
    "photosphere": "a colour within the photosphere's range",
    "diffuse-emission": "the field's diffuse emission",
    "astrometry": "an astrometric solution not to be trusted",
    "outside-selection": "a star outside the test's range of colour, temperature or distance",
    "metal-poor": "a metal-poor subdwarf, placed below the main sequence by its composition: a subdwarf spectral type or a J-Ks bluer than the main sequence's",
    "abundance-pattern": "an abundance pattern of natural mixing or mass transfer",
    "published-classification": "classified by the team that published the alert or catalogue",
    "published-solution": "an error in the published solution",
    "catalogued-false-positive": "a false positive or alarm in the catalogue that lists it",
    "group-depth": "a distance its group's other members place elsewhere",
    "dust-lane": "a dust lane over the galaxy's light",
    "energy-budget": "an energy budget the infrared does not carry",
    # small bodies
    "asteroid-class": "a metal-rich or enstatite taxonomic class",
    "activity": "outgassing or cometary activity",
    "yarkovsky": "a drift with the sign and size the Yarkovsky effect gives",
    "dark-comet": "a dark comet, published or refitted",
    "natural-density": "an area-to-mass ratio from radiation pressure that a rock of the body's size gives",
    "no-drift": "a gravity-only orbit that fits as well",
    "meteor-shower": "members of a known shower with overestimated speeds",
    "speed-bias": "groups that a shared speed overestimate makes",
    "tidal-synchronisation": "an orbit inside the range where tides synchronise",
    # population-level tests
    "chance-rate": "a target of a set that the null produces as often as observed",
    "control": "a control target of the test, not a candidate",
    "not-measured": "a target the test could not measure",
}

# (pattern, code); code None marks a descriptive clause kept in the note
RULES = [
    (r"^\d+ chords, \d+ points$|^ellipse null|^run of \d+ points|^class |^[A-]$|^\d+ plates$|^\d+ comparisons$"
     r"|^excess [\d.]+ mag$|^Ks-W\d [+-]?[\d.]+ at [\d.]+ sigma|^Ca II K|^residual [\d.]+ per cent|^HD 31435,"
     r"|^no other polarimetry|^Gaia gives E|^WISE sees the photosphere|^the angle matches|^Planck's"
     r"|^the star's Gaia XP spectrum|^one (new|modern) measurement|^Gaia luminosities sum|^SIMBAD\s+no type$"
     r"|^undecided$|^above the ceiling: to vet$|^no X-ray source$|^flat-bottomed [\d.]+ per cent box eclipse$"
     r"|^[\d.]+ mag$|^a wrong star in \d+|^offset [\d.]+ arcsec$|^alpha [-+\d.]+ to [-+\d.]+$", None),
    (r"fibre-fixed|the same fibre (holds|shows) (this line|it)", "fibre-fixed"),
    (r"fibres within \d+ on the same spectrograph|fibres beside it on the same plate", "neighbouring-fibres"),
    (r"of the arm's end", "detector-edge"),
    (r"agree in the instrument's frame", "instrument-frame"),
    (r"in every exposure of one night and in none of", "single-epoch"),
    (r"shared in the (stellar|instrument's) frame", "shared-feature"),
    (r"narrower than the instrument|detector pixels", "detector-pixel"),
    (r"masked, variance-inflated or dead pixels", "bad-pixel"),
    (r"peaks lie [\d.]+ A apart", "not-one-feature"),
    (r"cosmic.ray", "cosmic-ray"),
    (r"^one camera of the two covering it", "one-camera"),
    (r"^a spectrum with clean lines of both signs", "defective-spectrum"),
    (r"^hot pixel", "hot-pixel"),
    (r"other events detector-wide", "detector-wide"),
    (r"absorption at [\d.]+ A shared by \d+ other stars", "stellar-blend"),
    (r"times as wide as an unresolved line", "broad-feature"),
    (r"in a Balmer or He I line", "line-core"),
    (r"background galaxy|^background_galaxy$|^radio_agn$|background radio galaxy|^natural type (AGN|QSO)$",
     "background-galaxy"),
    (r"^OFF scans carry", "off-source"),
    (r"recurs on its drift in", "sidelobe-transmitter"),
    (r"^FM broadcast band$", "broadcast-band"),
    (r"broadband sky's sidereal profile|^sky_swing$|^a swing shared by \d+ channels", "sky-sidereal"),
    (r"^gnss_sidereal$", "gnss"),
    (r"^auroral kilometric radiation$", "auroral-kilometric-radiation"),
    (r"^solar wind plasma line$", "plasma-line"),
    (r"^terrestrial_hf$", "terrestrial-hf"),
    (r"^wind_local$", "spacecraft-local"),
    (r"^spacecraft anomaly", "spacecraft-anomaly"),
    (r"^RAD\d list mode", "instrument-mode"),
    (r"^spacecraft timing", "spacecraft-timing"),
    (r"^natural$", "natural-source"),
    (r"one dipping epoch|every deep dip is one transit|^one fitted epoch$|^single-pass dip"
     r"|the pass partner [\d.]+ minutes away (undimmed|missing)|one transit carries it", "single-epoch"),
    (r"^G-only dips", "single-band"),
    (r"other catalogue stars dip within half an hour|pair dips run at the catalogue rate|other stars step within",
     "field-event"),
    (r"Hipparcos (variability )?type D|^SIMBAD (type )?(\*\*|SB\*|RS\*)$|^RUWE [\d.]+$|Gaia binary|^binary \("
     r"|^companion \((RUWE|SB\*)|^RV scatter|blended lines of a double star|RUWE [\d.]+: a close binary"
     r"|long-period companion|RSCVN", "binary"),
    (r"eclipsing binary|^eb_catalogued$|^SIMBAD EB\*$|^heartbeat_binary$", "eclipsing-binary"),
    (r"^VSX (?!.* ROT$)|Hipparcos (variability )?type [PUMR]|^SIMBAD (Em|dS|a2|V|Pu|gD|Er)\*$|^stellar.variability"
     r"|^delta_scuti$|events: variability|pulsator|Gaia G scatter|^Gaia variable$|^SIMBAD V\*,", "variable-star"),
    (r"rotation period or a multiple|^VSX .* ROT$|spotted rotator|^SIMBAD (BY|Ro)\*$|^vsini", "rotation"),
    (r"^chromatic$", "chromatic"),
    (r"^a burst", "burst"),
    (r"^a flare on a flare star", "flare"),
    (r"phase scatter [\d.]+ of the period", "not-periodic"),
    (r"Holczer (periodic|polynomial) TTV", "timing-variation"),
    (r"of a data source's first or last maximum", "data-edge"),
    (r"detrending gap", "detrending"),
    (r"pinned at the (model's flat edge|fit's floor)", "fit-boundary"),
    (r"within the fit's own pile-up", "fit-degeneracy"),
    (r"^unstable, low S/N", "unstable-profile"),
    (r"within 1 per cent of a small-integer period ratio", "near-resonance"),
    (r"^masses estimated from radii$", "estimated-masses"),
    (r"^eccentricity errors above 0\.1$", "loose-eccentricities"),
    (r"per cent of its stable twins flagged as well$", "twin-flags"),
    (r"^orbits that cross at the archive's own values$", "catalogue-crossing"),
    (r"^eccentricities (given without errors|that are upper limits)$", "unmeasured-eccentricities"),
    (r"^masses that are upper limits$", "mass-limits"),
    (r"no impact parameter in the TOI table|a grazing geometry is not excluded", "unconstrained-geometry"),
    (r"^1 records$", "single-record"),
    (r"^X-ray source|^active(?! comet)|ultracool dwarf with emission lines|^SIMBAD type X$", "active-star"),
    (r"not main-sequence by Gaia|off the main sequence|^giant host|^evolved \(|mag above the main sequence",
     "evolved"),
    (r"star_forming_region|natural type (YSO|YSO_Candidate|OrionV\*|TTauri\*)$|^young_disk$|pre-main-sequence"
     r"|^SIMBAD type Ae\*$|^SIMBAD TT\*$", "young-star"),
    (r"^natural type (Em\*|CV\*|CV\?)$|^SIMBAD Pe\*$|^planetary nebula ", "natural-type"),
    (r"white dwarf|white-dwarf|WD\*", "white-dwarf"),
    (r"^warm dust$|^hot dust$|^infrared excess|debris disc", "dust"),
    (r"known brown dwarf in orbital motion|^close planet|^red companion or blend$", "companion"),
    (r"\(control\)$|^control$", "control"),
    (r"a neighbour in the windows|blend changing with proper motion|^moves [\d.]+ arcsec in|^WDS companion"
     r"|a companion [\d.]+(-[\d.]+)? arcsec away|arcsec from HD \d+ A|background giant in the aperture|third light or blend"
     r"|summed part of their light|Gaia's G misses a close companion|second source [\d.]+ arcsec away"
     r"|extended AllWISE source|crowded SMC field|shallow: blended|^neighbours' light in TESS's aperture", "blend"),
    (r"a cloud in the beam|a cold clump in the beam|^cloud_alignment$|Local Leo Cold Cloud|^interstellar:"
     r"|within the interstellar bound|aligned with the interstellar field|Lupus \d field|in the Heiles"
     r"|cirrus in the beam", "interstellar"),
    (r"^crowding correction", "crowding"),
    (r"source catalogue reads high|optical ghost|diffraction spike|W\d in \d+ of \d+ single exposures|scattered-light halo"
     r"|latent image|^release-dependent photometry", "artefact"),
    (r"within the M\d photospheric colour range", "photosphere"),
    (r"field's diffuse polarised emission", "diffuse-emission"),
    (r"RUWE above [\d.]+, parallax not trusted|scanning-law precession|^Hipparcos stochastic solution$", "astrometry"),
    (r"colour BP-RP .* outside F-early K|^Teff edge$|^hot$|^beyond [\d.]+ kpc$|^early type \(|^colour term extrapolated",
     "outside-selection"),
    (r"^\[Na/Fe\]", "abundance-pattern"),
    (r"^explained_by_alert_team$", "published-classification"),
    (r"published solution too hot|^TOI depth not in the light curves|^the light curve's duration allows a bound orbit|^not in ZTF at the Gaia period",
     "published-solution"),
    (r"false positive or alarm|KOI false-positive flags|\(TFOP (FP|FA)\)$|on a blue supergiant \(TFOP comment\)$"
     r"|, (FALSE POSITIVE|FP|FA)\)$", "catalogued-false-positive"),
    (r"group modulus|group's other moduli", "group-depth"),
    (r"dust lane", "dust-lane"),
    (r"^energy budget", "energy-budget"),
    (r"metal-rich or enstatite class", "asteroid-class"),
    (r"^outgassing$|^active comet", "activity"),
    (r"^agrees$|within sunlight: Yarkovsky", "yarkovsky"),
    (r"published dark comet|dark comet: refitted|dark-comet candidate", "dark-comet"),
    (r"no drift needed|gravity alone fits as well", "no-drift"),
    (r"shower members with overestimated speeds", "meteor-shower"),
    (r"shared speed overestimate", "speed-bias"),
    (r"where tides synchronise", "tidal-synchronisation"),
    (r"at the chance rate|within the expected rate|^within the (spread|null|re-paired null)$"
     r"|^FAP [\d.]+ against \d+ KOIs$|^aligned within \d+ deg$|^beyond \d+ d, matched|shared with \d+ unrelated bodies",
     "chance-rate"),
    (r"fewer than \d+ comparison light curves|target light curve too short|no differential light curve"
     r"|^transit not recovered at the TOI ephemeris"
     r"|^\d+ target detections$|querycat API request failed|no refcat match", "not-measured"),
]
_RULES = [(re.compile(p, re.I if code is None else 0), code) for p, code in RULES]
# polarisation narratives: HD 31435 open, the others explained
_PER_TARGET = {"gaia_dr3:154235505416809344": [], "gaia_dr3:4687221887455439616": ["blend", "binary"],
               "gaia_dr3:6010997328529547904": ["interstellar"], "gaia_dr3:6286369839598558592": ["interstellar"]}
# runs whose record states a measurement while the ledger gives the explanation
_PER_RUN = {"desi_wd_dust_pollution": ["companion"]}


def clauses(text):
    """Clauses of a vet or note, split at semicolons outside parentheses."""
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == ";" and depth <= 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return [c for c in out if c]


def codes(search_id, vet, note=None, target=None):
    """(codes, note) of a free-text vet: the codes its clauses name, words kept in the note; an unmatched clause raises KeyError."""
    if vet is None:
        return [], note
    parts = [c.strip() for c in vet.split(";")]
    if all(p in VET for p in parts):
        return parts, note
    words = clauses(vet)
    if search_id == "intrinsic_polarisation" and target in _PER_TARGET:
        found = list(_PER_TARGET[target])
    else:
        if vet.strip() == "explained" and note:
            # the explanation follows the feature's position or the dimming's size
            words = clauses(re.sub(r"^[RB] [\d.]+ A: |^[-+\d.]+ mag; ", "", note))
        found = []
        for w in words:
            hit = [code for rx, code in _RULES if rx.search(w)]
            if not hit:
                raise KeyError("no vet code for %r (%s)" % (w, search_id))
            for code in hit:
                if code and code not in found:
                    found.append(code)
        found = found or list(_PER_RUN.get(search_id, []))
    text = vet if vet.strip() != "explained" else None
    return found, "; ".join(x for x in (text, note) if x) or None


# --------------------------------------------------------------------------------------------------------------------
# coverage: the SQLite store of runs (searches) and per-target rows (coverage); columns as in SCHEMA

SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    id TEXT PRIMARY KEY, approach TEXT NOT NULL, archive TEXT, release TEXT, selection TEXT, code TEXT,
    code_sha256 TEXT, params TEXT, n_targets INTEGER, null_summary TEXT, sensitivity TEXT,
    result TEXT, ledger TEXT, finished TEXT, n_unit TEXT, code_file TEXT, outputs TEXT, ran TEXT);
CREATE TABLE IF NOT EXISTS coverage (
    search_id TEXT NOT NULL REFERENCES searches(id), target TEXT NOT NULL, stat REAL, p REAL, upper_limit REAL,
    unit TEXT, vet TEXT, note TEXT, PRIMARY KEY (search_id, target));
CREATE INDEX IF NOT EXISTS coverage_target ON coverage(target);
"""
ADDED = ("n_unit", "code_file", "outputs", "ran")
LEDGER = os.path.join(ROOT, "LEDGER.md")
RESULTS_MD = os.path.join(ROOT, "RESULTS.md")


def connect(path=None):
    db = sqlite3.connect(path or COVERAGE)
    db.executescript(SCHEMA)
    have = [r[1] for r in db.execute("PRAGMA table_info(searches)")]
    for col in ADDED:
        if col not in have:
            db.execute("ALTER TABLE searches ADD COLUMN %s TEXT" % col)
    return db


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ids_selection(targets):
    """Selection string for a run over an explicit target list."""
    return "ids:" + hashlib.sha256("\n".join(sorted(map(str, targets))).encode()).hexdigest()


def _j(x):
    return x if x is None or isinstance(x, str) else json.dumps(x, sort_keys=True)


def record_search(id, approach, archive=None, release=None, selection=None, code=None, params=None, n_targets=None,
                  null_summary=None, sensitivity=None, result=None, ledger=None, finished=None, db=None, entry=None,
                  n_unit=None, outputs=None):
    """Insert or replace one run's row; code is hashed, outputs' SHA-256 and times kept; unit, ledger and outputs persist when not given."""
    own = db is None
    db = db or connect()
    digest = sha256_file(code) if code and os.path.exists(code) else None
    name = os.path.basename(code) + (" " + entry if entry else "") if code else None
    outs, ran = None, None
    if outputs:
        rows = [dict(name=os.path.basename(p), sha256=sha256_file(p),
                     modified=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p))))
                for p in outputs if os.path.exists(p)]
        if rows:
            outs = json.dumps(rows, sort_keys=True)
            days = sorted(r["modified"][:10] for r in rows)
            ran = days[0] if days[0] == days[-1] else "%s/%s" % (days[0], days[-1])
    old = db.execute("SELECT n_unit, ledger, outputs, ran FROM searches WHERE id = ?", (id,)).fetchone()
    if old:
        n_unit = n_unit or old[0]
        ledger = old[1] if old[1] and ledger_row(old[1]) and not ledger_row(ledger or "") else ledger
        outs, ran = (outs, ran) if outs else (old[2], old[3])
    db.execute("INSERT OR REPLACE INTO searches (id, approach, archive, release, selection, code, code_sha256, params, "
               "n_targets, null_summary, sensitivity, result, ledger, finished, n_unit, code_file, outputs, ran) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (id, approach, archive, release, selection, name, digest,
                _j(params), n_targets, _j(null_summary), _j(sensitivity), result, ledger,
                finished or time.strftime("%Y-%m-%d"), n_unit, os.path.basename(code) if digest else None, outs, ran))
    db.commit()
    if own:
        db.close()


# identifier values that stand for a missing one; every target given one would share a single row of a run
MISSING_IDS = ("", "-", "--", "---", "-1", "none", "nan", "null", "n/a")


def missing_id(target):
    """True when a target's identifier, before any # facet, stands for a missing one, or a gaia_dr3 identifier is not a positive source id."""
    prefix, sep, value = str(target).partition(":")
    value = (value if sep else prefix).split("#")[0].strip()
    if value.lower() in MISSING_IDS:
        return True
    return prefix == "gaia_dr3" and not (value.isdigit() and int(value) > 0)


def record_targets(search_id, rows, db=None):
    """Insert per-target rows; worded vets are stored as codes with the words in the note, and a target whose identifier stands for a missing one is refused."""
    rows = list(rows)
    missing = sorted({str(r["target"]) for r in rows if missing_id(r["target"])})
    if missing:
        raise ValueError("%s: identifiers that stand for a missing one: %s" % (search_id, ", ".join(missing)))
    own = db is None
    db = db or connect()
    out = []
    for r in rows:
        vet, note = codes(search_id, r.get("vet"), r.get("note"), r["target"])
        out.append((search_id, r["target"], r.get("stat"), r.get("p"), r.get("upper_limit"), r.get("unit"),
                    "; ".join(vet) or None, note))
    db.executemany("INSERT OR REPLACE INTO coverage VALUES (?,?,?,?,?,?,?,?)", out)
    db.commit()
    if own:
        db.close()


def record_run(id, approach, archive, selection, code, entry, n, unit, params, null, sensitivity, result, rows, release=None):
    """A run, its ledger row and its per-target rows in one call."""
    record_search(id, approach, archive=archive, release=release, selection=selection, code=code, entry=entry, n_targets=n, n_unit=unit,
                  params=params, null_summary=null, sensitivity=sensitivity, ledger=approach, result=result)
    clear_targets(id)
    if rows:
        record_targets(id, rows)


def run_cli(main):
    """The command line of an approach whose modes are its analysis and `record`, which stores what the analysis prints; every other keyword of MAIN is an option, `--name value` (a list as several values), with the keyword's default."""
    import argparse
    import inspect
    quiet_warnings()
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("mode", nargs="?", choices=("run", "record"), default="run")
    for name, par in inspect.signature(main).parameters.items():
        d = par.default
        if name == "record" or d is inspect.Parameter.empty or d is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(d, bool):
            p.add_argument(flag, dest=name, default=d, type=lambda v: v.lower() in ("1", "true", "yes", "on"))
        elif isinstance(d, (list, tuple)):
            p.add_argument(flag, dest=name, default=list(d), nargs="+", type=type(d[0]) if d else str)
        else:
            p.add_argument(flag, dest=name, default=d, type=type(d))
    a = vars(p.parse_args())
    main(record=a.pop("mode") == "record", **a)


def clear_targets(search_id, db=None):
    """Remove a run's per-target rows."""
    own = db is None
    db = db or connect()
    db.execute("DELETE FROM coverage WHERE search_id = ?", (search_id,))
    db.commit()
    if own:
        db.close()


# a run that tests one target several ways (instruments, lines, channels) keys each test as the target's identifier followed by # and the facet
FACETED = "(c.target = ? OR substr(c.target, 1, length(?) + 1) = ? || '#')"


def card(target, db=None):
    """Every run covering a target with its result, its faceted rows among them, each with the row's own target last."""
    db = db or connect()
    q = ("SELECT s.approach, s.archive, s.release, c.stat, c.p, c.upper_limit, c.unit, c.vet, c.note, s.id, c.target "
         "FROM coverage c JOIN searches s ON c.search_id = s.id WHERE " + FACETED + " ORDER BY s.finished, c.target")
    return db.execute(q, (target, target, target)).fetchall()


def covered(target, approach=None, db=None):
    """Run ids covering a target, its faceted rows included, optionally for one approach."""
    db = db or connect()
    if approach:
        q = ("SELECT DISTINCT c.search_id FROM coverage c JOIN searches s ON c.search_id = s.id "
             "WHERE " + FACETED + " AND s.approach = ?")
        return [r[0] for r in db.execute(q, (target, target, target, approach))]
    return [r[0] for r in db.execute("SELECT DISTINCT c.search_id FROM coverage c WHERE " + FACETED, (target, target, target))]


def summary(db=None):
    db = db or connect()
    rows = db.execute("SELECT s.id, s.approach, s.archive, s.n_targets, COUNT(c.target), "
                      "SUM(c.vet IS NOT NULL), s.result FROM searches s LEFT JOIN coverage c ON c.search_id = s.id "
                      "GROUP BY s.id ORDER BY s.finished, s.id").fetchall()
    return rows


def ledger_rows(path=None):
    """Searched table of LEDGER.md as [(approach, data, result)]; empty without the ledger."""
    path = path or LEDGER
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path, encoding="utf-8").read().split("## Queued")[0].splitlines():
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| Approach |"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows.append((c[0], c[1], c[2]))
    return rows


def ledger_row(name, rows=None):
    """Searched row of an approach, or None."""
    return next((r for r in (rows if rows is not None else ledger_rows()) if r[0] == name), None)


def _runs(db):
    return db.execute("SELECT s.id, s.ledger, s.n_targets, s.n_unit, COUNT(c.target), SUM(c.vet IS NOT NULL), "
                      "s.finished FROM searches s LEFT JOIN coverage c ON c.search_id = s.id GROUP BY s.id "
                      "ORDER BY s.finished, s.id").fetchall()


def singular(unit):
    """A sample unit in the singular, for a sample of one: the noun before any parenthesis loses its plural ending."""
    words, _, rest = unit.partition(" (")
    w = words.split(" ")
    irregular = {"spectra": "spectrum", "data": "datum", "criteria": "criterion"}
    last = w[-1]
    if last in irregular:
        w[-1] = irregular[last]
    elif last.endswith("ies"):
        w[-1] = last[:-3] + "y"
    elif last.endswith("s") and not last.endswith("ss"):
        w[-1] = last[:-1]
    return " ".join(w) + (" (" + rest if rest else "")


def table(db=None):
    """RESULTS.md text: each Searched row with its data, runs, samples, per-target rows and result."""
    db = db or connect()
    rows = ledger_rows()
    runs = _runs(db)
    n_rows = db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0]
    n_targets = db.execute("SELECT COUNT(DISTINCT target) FROM coverage").fetchone()[0]
    out = ["# Results", "",
           "Generated by `python -I seti.py store table` from the Searched table of `LEDGER.md` and the "
           "coverage store. Each approach's result is the ledger's, stated after vetting; each run's own output "
           "before vetting is the `result` column of its row in the store.", "",
           "A run is one row of the store's `searches` table. Its sample is the number of things it searched "
           "(`n_targets`) in the unit named (`n_unit`), and samples in different units are never added. Its rows are "
           "its rows in the `coverage` table, one for each target for which it recorded a statistic, a limit or a "
           "vet; many runs record only their flagged targets, so a run's rows are not its sample. A vetted row "
           "carries the codes of `README.md` for the explanations that removed it.", "",
           "%d runs of %d approaches; %d per-target rows over %d distinct target identifiers."
           % (len(runs), len(rows), n_rows, n_targets), ""]
    for name, data, result in rows:
        mine = [r for r in runs if r[1] == name]
        out.append("## %s" % name)
        out.append("")
        out.append("**Data.** %s" % data)
        out.append("")
        if mine:
            out.append("**Runs.** " + "; ".join(
                "`%s`: %s, %d per-target row%s%s" % (
                    sid, ("%s %s" % ("{:,}".format(n), unit if n != 1 else singular(unit))) if n is not None else "sample not recorded (%s)" % unit,
                    k, "" if k == 1 else "s", (", %d vetted" % v) if v else "")
                for sid, _, n, unit, k, v, _ in mine))
        else:
            out.append("**Runs.** none in the store")
        out.append("")
        out.append("**Result.** %s" % result)
        out.append("")
    return "\n".join(out)


def _bad_vets(db):
    return sorted({v for (v,) in db.execute("SELECT DISTINCT vet FROM coverage WHERE vet IS NOT NULL")
                   if any(c.strip() not in VET for c in v.split(";"))})


def _missing_ids(db):
    return sorted("%s %s" % r for r in db.execute("SELECT search_id, target FROM coverage") if missing_id(r[1]))


def _undocumented_codes():
    path = os.path.join(ROOT, "README.md")
    text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    return [c for c in VET if "`%s`" % c not in text]


def verify(db=None):
    """Checks of the store against the ledger and itself as [(check, items, fatal)]."""
    db = db or connect()
    rows = ledger_rows()
    names = [r[0] for r in rows]
    text = open(LEDGER, encoding="utf-8").read() if os.path.exists(LEDGER) else ""
    queued = [line.split("|")[1].strip() for line in text.split("## Queued")[-1].split("\n## ")[0].splitlines()
              if line.startswith("|") and not line.startswith("|---")] if "## Queued" in text else []
    runs = _runs(db)
    linked = {r[1] for r in runs}
    out = [
        ("runs whose ledger row is not in the Searched table", [r[0] for r in runs if r[1] not in names], True),
        ("runs linked to a Queued row", [r[0] for r in runs if r[1] in queued and r[1] not in names], True),
        ("runs without a sample unit", [r[0] for r in runs if not r[3]], True),
        ("Searched rows named twice", sorted({n for n in names if names.count(n) > 1}), True),
        ("per-target rows of no run", [str(r[0]) for r in db.execute(
            "SELECT COUNT(*) FROM coverage c LEFT JOIN searches s ON s.id = c.search_id WHERE s.id IS NULL")
            if r[0]], True),
        ("vets that are not codes of setitools.core", _bad_vets(db), True),
        ("per-target rows whose identifier stands for a missing one", _missing_ids(db), True),
        ("vet codes missing from README.md", _undocumented_codes(), True),
        ("Searched rows with no run in the store", [n for n in names if n not in linked], False),
    ]
    current = open(RESULTS_MD, encoding="utf-8").read() if os.path.exists(RESULTS_MD) else ""
    out.append(("RESULTS.md differs from table()", [] if current == table(db) else ["RESULTS.md"], True))
    return out


STORE_USAGE = """    python -I seti.py store summary
    python -I seti.py store table          (writes RESULTS.md beside the store)
    python -I seti.py store verify         (exits 1 when a check fails)
    python -I seti.py store card gaia_dr3:4472832130942575872
    python -I seti.py store covered kic:8719324
    python -I seti.py store codes"""


def main():
    """Store command line: summary, table, verify, card, covered, codes."""
    if len(sys.argv) < 2 or sys.argv[1] not in ("summary", "card", "covered", "table", "verify", "codes"):
        print(STORE_USAGE)
        return
    if sys.argv[1] == "codes":
        for k, v in VET.items():
            print("%-28s %s" % (k, v))
        return
    db = connect()
    if sys.argv[1] == "verify":
        bad = 0
        for check, items, fatal in verify(db):
            print("%-4s %s: %d%s" % ("FAIL" if items and fatal else "ok" if not items else "note", check, len(items),
                                     (" (" + ", ".join(items[:12]) + (", ..." if len(items) > 12 else "") + ")")
                                     if items else ""))
            bad += bool(items and fatal)
        sys.exit(1 if bad else 0)
    if sys.argv[1] == "table":
        open(RESULTS_MD, "w", encoding="utf-8", newline="\n").write(table(db))
        print("wrote", RESULTS_MD)
        return
    if sys.argv[1] == "summary":
        tot = 0
        for r in summary(db):
            tot += r[4]
            print("%-34s %-30s %-22s %7s targets, %6d rows, %5d vetted  %s"
                  % (r[0], r[1][:30], (r[2] or "")[:22], r[3], r[4], r[5] or 0, (r[6] or "")[:80]))
        n = db.execute("SELECT COUNT(DISTINCT target) FROM coverage").fetchone()[0]
        print("%d rows over %d distinct targets" % (tot, n))
    elif sys.argv[1] == "card":
        for r in card(sys.argv[2], db):
            print(" | ".join("" if x is None else str(x) for x in r))
    else:
        print("\n".join(covered(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, db)))
