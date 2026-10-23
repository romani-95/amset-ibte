"""Rung 1: bit-for-bit regression of solver: rta against pristine upstream AMSET.

Runs the same calculation with the pristine code (``--pristine`` path, put first on
PYTHONPATH) and with this branch, in separate processes, and compares the transport
results exactly.

Usage:
    python regression_upstream.py --pristine /path/to/amset-main [--factor 10]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CASES = {
    "GaAs": {
        "settings_file": "examples/GaAs/settings.yaml",
        "overrides": {"write_mesh": False, "nworkers": 4},
    },
    "Si": {
        "settings_file": "examples/Si/settings.yaml",
        "overrides": {"write_mesh": False, "nworkers": 4},
    },
}

_RUNNER = """
import json, sys
from pathlib import Path
from monty.serialization import loadfn
import amset
from amset.core.run import Runner
case_dir, settings_file, overrides, outdir = sys.argv[1:5]
settings = loadfn(settings_file)
settings.update(json.loads(overrides))
for key in ("deformation_potential",):
    if settings.get(key) == "deformation.h5":
        settings[key] = str(Path(case_dir) / "deformation.h5")
settings["wavefunction_coefficients"] = str(Path(case_dir) / "wavefunction.h5")
settings["print_log"] = False
runner = Runner.from_vasprun(str(Path(case_dir) / "vasprun.xml.gz"), settings)
runner.run(directory=outdir)
print("amset from", amset.__file__)
"""


def run(pythonpath, case, factor, outdir):
    info = CASES[case]
    overrides = dict(info["overrides"], interpolation_factor=factor)
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    else:
        env.pop("PYTHONPATH", None)
    case_dir = REPO / "examples" / case
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            str(case_dir),
            str(REPO / info["settings_file"]),
            json.dumps(overrides),
            str(outdir),
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=tempfile.gettempdir(),
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-3000:])
    print(out.stdout.strip().splitlines()[-1])
    files = list(Path(outdir).glob("transport_*.json"))
    return json.loads(files[0].read_text())


def _arrays(d, prefix=""):
    if isinstance(d, dict) and "@class" in d and d.get("@module") == "numpy":
        yield prefix, np.array(d["data"])
    elif isinstance(d, dict):
        for k, v in d.items():
            yield from _arrays(v, f"{prefix}/{k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pristine", required=True)
    parser.add_argument("--factor", type=float, default=10)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    args = parser.parse_args()

    report = {}
    ok = True
    for case in args.cases:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ref = run(args.pristine, case, args.factor, a)
            new = run(None, case, args.factor, b)
        ref_arrays = dict(_arrays(ref))
        new_arrays = dict(_arrays(new))
        diffs = {}
        for key, val in ref_arrays.items():
            other = new_arrays[key]
            diffs[key] = {
                "identical": bool(np.array_equal(val, other)),
                "max_abs_diff": float(np.max(np.abs(val - other))) if val.size else 0.0,
            }
            ok &= diffs[key]["identical"]
        report[case] = diffs
        for key, d in diffs.items():
            print(f"{case:5s} {key:45s} identical={d['identical']} "
                  f"max|Δ|={d['max_abs_diff']:.3e}")

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "regression_upstream.json").write_text(json.dumps(report, indent=1))
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
