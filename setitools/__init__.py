"""seti-tools: technosignature and biosignature searches on public archives; core, science and thirteen family modules of approaches."""

import importlib
import os
import sys

from setitools.core import quiet_warnings

# one math-library thread per process, so the process cap bounds CPU load
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

USAGE = """    python -I seti.py                              the families
    python -I seti.py <family>                     a family's approaches
    python -I seti.py <family> <approach> --help   an approach's method and modes
    python -I seti.py <family> <approach> <mode>   a run
    python -I seti.py store <command>              the coverage store: summary, table, verify, card, covered, codes
    python -I seti.py selftest [--online]          the package's checks on synthetic data
"""

FAMILIES = {
    "lasers": "laser lines in stellar and sky-fibre spectra",
    "ultraviolet": "ultraviolet lines and photon timing",
    "composition": "short-lived elements, fission products and industrial gases",
    "flashes": "glints, flares, levels and clocks in light curves",
    "transits": "transit shapes, timing, belts and geometry",
    "dimming": "dimming, dips and colour in survey photometry",
    "infrared": "waste heat, dark companions and missing flux",
    "breakthrough": "Breakthrough Listen radio tracks, pulses and drifts",
    "radio_sky": "persistent lines, sidereal tests and lensed receivers in radio surveys",
    "highenergy": "X-ray, gamma-ray, neutrino and cosmic-ray tests",
    "solar_system": "asteroids, comets, meteors and occultations",
    "neighbourhood": "the nearby stars' kinematics, polarisation and companions",
    "biosignatures": "gases of life and their abiotic mimics in temperate worlds' spectra",
}


def family(name):
    """The family module by name, imported on demand."""
    if name not in FAMILIES:
        raise KeyError(name)
    return importlib.import_module("setitools." + name)


def run(argv=None):
    """Dispatch the command line: family, approach, then the approach's own arguments."""
    quiet_warnings()
    argv = list(sys.argv if argv is None else argv)
    prog = os.path.basename(argv[0]) if argv else "seti.py"
    if len(argv) < 2 or argv[1] not in FAMILIES:
        print(USAGE)
        for name, what in FAMILIES.items():
            print("    %-14s %s" % (name, what))
        return 0
    mod = family(argv[1])
    approaches = mod.APPROACHES
    if len(argv) < 3 or argv[2] not in approaches:
        # family summary and each registered approach with the first line of its method
        print((mod.__doc__ or "").strip() + "\n")
        for k, (_, doc) in approaches.items():
            print("    %-16s %s" % (k, doc.strip().split("\n")[0]))
        print("\n    python -I seti.py %s <approach> [mode ...]      (each approach's own modes and options; help lists them)"
              % argv[1])
        return 0
    name = argv[2]
    if len(argv) > 3 and argv[3] in ("-h", "--help", "help"):
        print(approaches[name][1])
        return 0
    sys.argv = ["%s %s %s" % (prog, argv[1], name)] + argv[3:]
    approaches[name][0]()
    return 0
