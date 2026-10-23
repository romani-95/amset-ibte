"""Mesh convergence of SERTA, MRTA and IBTE mobilities (spec section 3.8).

Claes et al. (SI Sec. II) found that the IBTE needs denser meshes than the RTAs and
converged on the IBTE result: three consecutive meshes within 5%.

Usage:
    python convergence_scan.py VASPRUN SETTINGS_YAML OUTDIR --factors 10 20 40 \
        [--set key=value ...]

Each factor runs in its own process (``run_case.py``); the table is written to
OUTDIR/convergence.json and OUTDIR/convergence.md.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def converged_flags(values, tol=0.05, window=3):
    """True at position i if values[i-window+1 .. i] are all within tol of each other."""
    flags = []
    for i in range(len(values)):
        if i < window - 1:
            flags.append(False)
            continue
        win = values[i - window + 1 : i + 1]
        flags.append(max(win) / min(win) - 1 <= tol)
    return flags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("vasprun")
    p.add_argument("settings")
    p.add_argument("outdir")
    p.add_argument("--factors", nargs="+", type=float, required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--timeout", type=float, default=3600, help="seconds per run")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for factor in args.factors:
        run_dir = outdir / f"factor_{factor:g}"
        cmd = [
            sys.executable,
            str(HERE / "run_case.py"),
            args.vasprun,
            args.settings,
            str(run_dir),
            "solver=ibte",
            f"interpolation_factor={factor}",
            "print_log=false",
        ] + args.set
        print("running", " ".join(cmd[2:]), flush=True)
        # a run hit by the OOM killer can leave AMSET hanging: run it in its own
        # process group and kill the whole group (workers included) on timeout
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _, stderr = proc.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            print(f"factor {factor} timed out after {args.timeout} s; skipping")
            continue
        if proc.returncode != 0:
            print(stderr[-3000:])
            print(f"factor {factor} failed (exit {proc.returncode}); skipping")
            continue
        summary = json.loads((run_dir / "summary.json").read_text())
        for r in summary["results"]:
            rows.append(
                {
                    "factor": factor,
                    "mesh": "x".join(map(str, summary["mesh"])),
                    "doping": r["doping"],
                    "temperature": r["temperature"],
                    "SERTA": r["SERTA"],
                    "MRTA": r["MRTA"],
                    "RTA": r["overall"],
                    "IBTE": r["IBTE"],
                    "kernel_MB": summary["ibte"]["kernel_mb"],
                    "wall_s": summary["wall_time"],
                }
            )
        (outdir / "convergence.json").write_text(json.dumps(rows, indent=1))

    keys = sorted({(r["doping"], r["temperature"]) for r in rows})
    lines = []
    for doping, temp in keys:
        sub = [r for r in rows if (r["doping"], r["temperature"]) == (doping, temp)]
        flags = converged_flags([r["IBTE"] for r in sub])
        lines.append(f"\n### doping {doping:.2e} cm^-3, {temp:.0f} K\n")
        lines.append(
            "| factor | mesh | SERTA | MRTA | RTA (amset) | IBTE | SERTA/IBTE | "
            "MRTA/IBTE | IBTE conv. (3×5%) | kernel MB | wall s |"
        )
        lines.append("|" + "---|" * 11)
        for r, f in zip(sub, flags):
            lines.append(
                f"| {r['factor']:g} | {r['mesh']} | {r['SERTA']:.1f} | {r['MRTA']:.1f} | "
                f"{r['RTA']:.1f} | {r['IBTE']:.1f} | {r['SERTA'] / r['IBTE']:.3f} | "
                f"{r['MRTA'] / r['IBTE']:.3f} | {'yes' if f else 'no'} | "
                f"{r['kernel_MB']:.0f} | {r['wall_s']:.0f} |"
            )
    text = "\n".join(lines)
    (outdir / "convergence.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
