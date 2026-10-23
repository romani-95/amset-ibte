# Helper scripts

Run them with the `amset-ibte` environment active.

| script | what it does |
|---|---|
| `lo_phonons.py` | NAC-corrected LO modes from a VASP Γ-phonon run: DFPT (IBRION 7/8 + LEPSILON) or finite differences (IBRION 5/6 + LCALCEPS). Prints the LO frequencies and coupling shares (`pop_frequency` / `pop_weights`), and writes the `pop_modes_file` with direction-dependent weights. |
| `run_case.py` | One AMSET run with `key=value` overrides of a `settings.yaml`. Writes `summary.json` with SERTA, MRTA, RTA, IBTE and per-mechanism mobilities. |
| `convergence_scan.py` | Mesh-convergence scan (several `interpolation_factor`s) with `run_case.py`. Tabulates SERTA / MRTA / RTA / IBTE and flags 3 consecutive meshes within 5%. |
| `sanity_checks.py` | Checks `summary.json` files: IBTE ≥ SERTA, IBTE converged, transport weight without in-scattering < 1%, and (with `--cubic`) an isotropic tensor. |
| `regression_upstream.py` | Bit-for-bit comparison of `solver: rta` with a pristine AMSET checkout on the GaAs and Si examples. |

## LO modes for `pop_modes_file`

```bash
python scripts/lo_phonons.py path/to/phonon_run --min-weight 0.01 --modes-file modes.json
# optional: --eps-inf "e1 e2 e3"   (or 9 values) to use another ε∞ than the run's own
#           --max-degree 6          for strongly anisotropic cells
```

The folder needs `vasprun.xml(.gz)`, plus `OUTCAR(.gz)` for DFPT runs. Then, in
`settings.yaml`:

```yaml
pop_modes_file: /full/path/to/modes.json
high_frequency_dielectric: ...   # ε∞ printed by lo_phonons.py ("ε∞ used")
static_dielectric: ...           # ε₀ = ε∞ + ε_ion (printed)
```

Use the ε∞ that the modes were computed with, so that the LO frequencies and the
total coupling 1/ε∞ − 1/ε₀ are consistent.

## Mesh convergence

```bash
python scripts/convergence_scan.py vasprun.xml settings.yaml scan_out \
    --factors 5 10 20 --set "doping=[1e16]" nworkers=16
python scripts/sanity_checks.py scan_out
```
