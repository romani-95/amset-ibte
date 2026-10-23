"""Consistency checks on AMSET-IBTE results (summary.json from run_case.py).

For every run and doping:

1. IBTE ≥ SERTA. SERTA drops the in-scattering. For forward-peaked or isotropic
   scattering (POP, IMP, PIE, ADP) the in-scattering returns momentum, so the
   IBTE must be higher. A lower IBTE means back-scattering dominates or there is a
   problem.
2. The iterative solution converged (``ibte.points[*].converged``).
3. Little transport weight on states without in-scattering (< 1%).
4. Cubic cells: the mobility tensor is isotropic (diagonal spread and off-diagonal
   elements < 1e-6 relative).
5. IBTE/MRTA is reported (no fixed expectation: the MRTA is exact only for
   elastic, isotropic, parabolic cases).

Usage:
    python sanity_checks.py ROOT [ROOT ...] [--cubic]   # ROOT/**/summary.json

--cubic also requires an isotropic tensor (use only for cubic materials).
"""

import json
import sys
from pathlib import Path

import numpy as np


def check(summary_file, cubic=False):
    d = json.loads(Path(summary_file).read_text())
    problems, notes = [], []
    points = (d.get("ibte") or {}).get("points", [])
    points_full = (d.get("ibte") or {}).get("points_full", [])
    for i, r in enumerate(d["results"]):
        if "IBTE" not in r:
            continue  # RTA-only run
        tag = f"{r['doping']:.0e}"
        for key in ("IBTE", "IBTE_full"):
            if key in r and r[key] < r["SERTA"] * (1 - 1e-6):
                problems.append(f"{tag}: {key} {r[key]:.4g} < SERTA {r['SERTA']:.4g}")
        for key, pts in (("IBTE", points), ("IBTE_full", points_full)):
            if i < len(pts) and not pts[i].get("converged", True):
                problems.append(f"{tag}: {key} not converged "
                                f"(residual {pts[i]['residual']:.1e})")
        if i < len(points):
            p = points[i]
            frac = max((p.get("flagged_weight_fraction") or {}).values() or [0])
            if frac > 0.01:
                problems.append(f"{tag}: {100 * frac:.1f}% weight without in-scattering")
        t = np.asarray(r["IBTE_tensor"])
        diag = np.diag(t)
        aniso = np.ptp(diag) / diag.mean()
        offd = np.abs(t - np.diag(diag)).max() / diag.mean()
        if cubic and max(aniso, offd) > 1e-6:
            problems.append(f"{tag}: cubic but anisotropic IBTE tensor "
                            f"(diag spread {aniso:.1e}, off-diag {offd:.1e})")
        full = r.get("IBTE_full", r["IBTE"])
        notes.append((tag, full / r["SERTA"], full / r["MRTA"], aniso, offd, str(summary_file)))
    return problems, notes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cubic = "--cubic" in sys.argv
    files = sorted({f for root in args for f in Path(root).rglob("summary.json")})
    nprob = 0
    ratios_s, ratios_m = [], []
    for f in files:
        try:
            problems, notes = check(f, cubic)
        except (KeyError, ValueError) as e:
            print(f"?? {f}: unreadable ({e})")
            continue
        for p in problems:
            print(f"!! {f}: {p}")
        nprob += len(problems)
        for _, rs, rm, _, _, src in notes:
            ratios_s.append(rs)
            ratios_m.append((rm, src))
    print(f"\n{len(files)} summaries, {len(ratios_s)} IBTE solutions, {nprob} problems")
    if ratios_s:
        print(f"IBTE/SERTA: min {min(ratios_s):.3f}, max {max(ratios_s):.3g}")
        lo, hi = min(ratios_m), max(ratios_m)
        print(f"IBTE/MRTA (full IBTE where available): min {lo[0]:.3f} ({lo[1]}), "
              f"max {hi[0]:.3f} ({hi[1]})")


if __name__ == "__main__":
    main()
