"""Run one AMSET calculation and print SERTA / MRTA / RTA / IBTE mobilities.

Usage:
    python run_case.py VASPRUN SETTINGS_YAML OUTDIR [key=value ...]

Extra ``key=value`` pairs override the settings (values parsed as YAML), e.g.
``solver=ibte interpolation_factor=10 "scattering_type=[ADP, POP]"``.
A JSON summary is written to OUTDIR/summary.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from monty.serialization import loadfn

from amset.constants import bohr_to_cm
from amset.core.run import Runner
from amset.util import tensor_average


def main():
    vasprun, settings_file, outdir = sys.argv[1:4]
    settings = loadfn(settings_file) if settings_file != "-" else {}
    for arg in sys.argv[4:]:
        key, value = arg.split("=", 1)
        settings[key] = yaml.safe_load(value)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    settings.setdefault("write_log", True)
    settings.setdefault("file_format", "json")

    if settings.get("deformation_potential") == "deformation.h5":
        settings["deformation_potential"] = str(Path(vasprun).parent / "deformation.h5")
    wfc = settings.get("wavefunction_coefficients", "wavefunction.h5")
    if not Path(wfc).is_absolute():
        settings["wavefunction_coefficients"] = str(Path(vasprun).parent / wfc)

    t0 = time.perf_counter()
    runner = Runner.from_vasprun(vasprun, settings)
    amset_data, usage = runner.run(directory=outdir, return_usage_stats=True)
    wall = time.perf_counter() - t0

    summary = {"settings_overrides": sys.argv[4:], "wall_time": wall, "usage": usage}
    summary["mesh"] = [int(x) for x in amset_data.kpoint_mesh]
    rows = []
    for n, t in np.ndindex(amset_data.fermi_levels.shape):
        row = {
            "doping": float(amset_data.doping[n] * (1 / bohr_to_cm) ** 3),
            "temperature": float(amset_data.temperatures[t]),
        }
        for key, mob in amset_data.mobility.items():
            row[key] = float(tensor_average(mob[n, t]))
            row[key + "_tensor"] = np.asarray(mob[n, t]).tolist()
        rows.append(row)
    summary["results"] = rows
    if amset_data.ibte_results is not None:
        summary["ibte"] = amset_data.ibte_results
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)

    for r in rows:
        keys = [k for k in r if not k.endswith("_tensor") and k not in ("doping",)]
        print(f"doping {r['doping']:.3e}: " + ", ".join(f"{k}={r[k]:.3f}" for k in keys))


if __name__ == "__main__":
    main()
