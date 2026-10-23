# Iterative Boltzmann transport equation (IBTE) in AMSET

By default AMSET solves the Boltzmann transport equation in a relaxation-time
approximation: the elastic rates (ADP, PIE, IMP) carry the momentum-relaxation
(MRTA) factor (1 − v·v′/|v|²), and the polar-optical-phonon (POP, inelastic) rates
are plain out-scattering rates (SERTA-like). Claes *et al.*, PRB **106**, 094302
(2022), showed that SERTA always underestimates the exact (IBTE) mobility, often by
a factor of about 2 in polar materials.

`solver: ibte` adds an exact iterative solution of the linearized BTE for the
mobility, next to the existing RTA solution. The default (`solver: rta`) is
unchanged.

## Theory

Notation follows the Supplementary Material of Claes *et al.* (2022).

Out-scattering rate, Eq. (S13). This is what AMSET computes for POP:

    1/τ⁰_nk = 2π Σ_m ∫dq/Ω_BZ |g|² [ (n_q + f⁰_{m,k+q}) δ(ε_nk − ε_{mk+q} + ω)
                                   + (n_q + 1 − f⁰_{m,k+q}) δ(ε_nk − ε_{mk+q} − ω) ]

IBTE, Eq. (S12):

    ∂f_nk/∂E_β = (∂f⁰_nk/∂ε) v_nk,β τ⁰_nk
               + 2π τ⁰_nk Σ_m ∫dq/Ω_BZ |g|² [ (1 + n_q − f⁰_nk) δ(ε_nk − ε_{mk+q} + ω)
                                           + (n_q + f⁰_nk)     δ(ε_nk − ε_{mk+q} − ω) ]
                 × ∂f_{m,k+q}/∂E_β

Mobility, Eq. (S3): μ_αβ = −1/(Ω n) Σ_n ∫dk/Ω_BZ (∂f_nk/∂E_β) v_nk,α.

The final-state surfaces, |g|², overlaps and quadrature weights are the same for
out- and in-scattering. Only the occupation factors change:

|           | from ε′ = ε + ħω | from ε′ = ε − ħω |
|-----------|------------------|------------------|
| out (S13) | n + f⁰(ε+ħω)     | n + 1 − f⁰(ε−ħω) |
| in (S12)  | n + 1 − f⁰(ε)    | n + f⁰(ε)        |

For elastic mechanisms there are no occupation factors and the in-kernel equals the
out-kernel. In the IBTE the elastic kernel is the raw rate, without the MRTA factor.

### Unknown used in AMSET

AMSET iterates on Λ_nk ≡ (∂f_nk/∂E)/(∂f⁰_nk/∂ε), a vector with units of
velocity × time:

    Λ_nk = τ⁰_nk v_nk + τ⁰_nk Σ_k′ K_in(k, k′) [f⁰′(ε′)/f⁰′(ε_nk)] Λ_k′

The Fermi-derivative ratio is 1 for elastic processes and about e^{∓ħω/kT} in the
non-degenerate limit for POP. The mobility then uses AMSET's transport-DOS machinery
with v⊗Λ in place of v⊗v τ. For Λ = τ⁰v (no in-scattering) this is exactly SERTA.

A useful identity: in × f⁰′(ε′)/f⁰′(ε) = out for each POP channel. So every row of
the iteration matrix τ⁰ K_in f⁰′(ε′)/f⁰′(ε) sums to one. This is the
number-conservation zero mode of the collision operator, and the test suite uses it
as a consistency check.

## Implementation

| Step | Where |
|---|---|
| Kernel capture: for each irreducible state, the quadrature points k′ of AMSET's rate integral, with their geometric weight (\|g\|² × overlap × quadrature weight, without prefactor, occupations or MRTA factor) | `scattering/calculate.py::calculate_rate(kernel_mapper=...)`, `_get_rates_and_kernel_row` |
| Off-grid k′ → trilinear corners on the full mesh (identical to `MRTACalculator` interpolation, checked to 1e-10) | `ibte/kernel.py::TrilinearMapper` |
| Sparse storage: one CSR pattern per spin and channel (elastic, absorption, emission), one value array per mechanism; HDF5 I/O | `ibte/kernel.py::KernelChannel`, `IBTEKernel` |
| Vector unfolding Λ(Sk) = S Λ(k), with time reversal as −R, checked against the FFT velocities | `ibte/symmetry.py` |
| Little-group projection of the in-scattering term | `ibte/symmetry.py::get_little_group_projectors` |
| Occupation factors, iteration matrix, linear mixing and GMRES, option-B check | `ibte/solver.py` |
| Mobility from v⊗Λ | `ibte/transport.py` |
| Orchestration and logging | `ibte/run.py`, hooks in `core/run.py` |

The prefactors, occupation factors and Fermi-derivative ratios depend only on the
initial state (row). They are applied as row scalings at solve time, so a single
kernel serves every doping and temperature.

## Settings

```yaml
solver: rta              # rta (default, upstream behaviour) | ibte
rta_mode: amset          # amset (MRTA elastic + SERTA POP) | serta | mrta
ibte_method: iterate     # iterate (linear mixing) | gmres
ibte_niter: 100
ibte_tol: 1.0e-4         # relative (weighted Λ change and mobility change)
ibte_alpha_mix: 0.7
ibte_save_kernel: false  # write ibte_kernel.h5 in the output directory
ibte_load_kernel: null   # reuse a kernel (same band structure, mesh and settings)
```

- `rta_mode` also works with `solver: rta`. `serta` drops the MRTA factor
  everywhere. `mrta` applies it to POP as well; that is not rigorous for inelastic
  scattering, but it is a useful bracket.
- With `solver: ibte` the transport output contains, in addition to the existing
  keys, `mobility["IBTE"]`, `mobility["SERTA"]` (pure SERTA, iteration 0) and
  `mobility["MRTA"]` (MRTA factor on all mechanisms). `mobility["overall"]` stays the
  `rta_mode` result. An `ibte` section stores convergence histories, the fraction of
  transport weight carried by states without in-scattering, kernel sizes and timings.
- A kernel can be reloaded (`ibte_load_kernel`) to re-solve at other temperatures or
  dopings without recomputing rates. Rates are rebuilt from the kernel row sums.
  States outside the energy window of the original run have no rows, and are
  reported as flagged.

### Mode-resolved POP (independent of the solver)

```yaml
pop_frequency: [5.0, 9.6, 20.0]   # LO mode frequencies (THz)
pop_weights:   [0.2, 0.2, 0.6]    # shares of 1/ε∞ − 1/ε₀ (normalized)
```

A list of frequencies replaces the single effective POP mode with one Fröhlich
term per polar LO mode ν (generalized Fröhlich model):

|g_ν(q)|² ∝ (ω_ν/2) · w_ν · 4π(1/ε∞(q̂) − 1/ε₀(q̂)) / q²

Each mode has its own phonon occupation and its own final-state surfaces ε ± ħω_ν.
Per-mechanism outputs are labelled POP1, POP2, …. In the IBTE each mode gets its own
absorption/emission kernel channels. A scalar `pop_frequency` is unchanged
upstream behaviour.

`scripts/lo_phonons.py` produces the inputs from a VASP phonon run at Γ: DFPT
(IBRION = 7/8 with LEPSILON) or finite differences (IBRION = 5/6 with LCALCEPS; Born
charges and ε are then read from `vasprun.xml`). It adds the non-analytic correction
for 100+ directions,
direction-averages the LO frequencies and weights, and merges nearby or weak
modes. It also prints the single-mode alternative. Note that
`amset phonon-frequency` without a NAC-corrected OUTCAR averages **TO**
frequencies, which can strongly underestimate the mobility of polar materials.
Use ε∞ consistent with the phonons and Born charges (e.g. the DFPT one).

**Direction-dependent weights.** In anisotropic crystals, each mode's share of the
coupling depends strongly on the direction of q (one mode can carry most of the
coupling for q ∥ c and little for q ⊥ c). Direction-averaged
weights then give the right mobility trace but too isotropic a tensor. Use:

```yaml
pop_modes_file: modes.json   # overrides pop_frequency / pop_weights
```

`lo_phonons.py DFPT_DIR --modes-file FILE [--eps-inf ...] [--max-degree 4]` writes
this file:

- `frequencies`: the merged LO frequencies, direction-averaged;
- `weights`: the averaged shares;
- `exponents`, `coefficients`: per mode, a fit of the share w_ν(q̂) by even
  monomials q_x^a q_y^b q_z^c (degree 4 by default).

Σ_ν w_ν(q̂) = 1 holds exactly for every q̂, so the total coupling still follows
1/ε∞(q̂) − 1/ε₀(q̂). The fit is computed from the DFPT Born charges and force
constants; it is never fitted to transport results. The script prints the fit
error per mode. Increase `--max-degree` (6) if the error is above a few percent.
The mode frequencies themselves are kept direction-averaged.

### `fix_kpoint_symmetry` (independent of the solver)

Upstream `get_kpoints_tetrahedral` passes the transposed lattice to spglib. This is
harmless when the lattice matrix is symmetric (orthogonal cells, the standard FCC
primitive cell), but for hexagonal or trigonal cells spglib then finds only
inversion. The mesh is then reduced ~6× less than possible: still correct, but much
more time and memory. `fix_kpoint_symmetry: true` passes the lattice correctly. It
is off by default so that upstream results stay bit-for-bit identical. In tests on
hexagonal cells it changed the mobilities by <0.5% and cut memory by about 3×.

### Ionized impurities and free-carrier screening

IMP (always screened, Brooks–Herring), and POP/PIE with `free_carrier_screening:
true`, contain the screening length 1/(q² + β²), where β depends on doping and
temperature. For these mechanisms the kernel stores one entry per (doping,
temperature). All other mechanisms keep a single entry, so the kernel grows only by
the doping-dependent part:

- IMP adds ndops × ntemps entries to the elastic channel;
- screening multiplies each POP channel by ndops × ntemps.

`ibte_save_kernel` stores the entry layout. A doping-dependent kernel can only be
reloaded for the same dopings and temperatures.

**IMP at low carrier density.** IMP is then strongly forward-peaked (screening
wavevector β ≪ mesh spacing Δk):

- **Convergence.** SERTA can be 10⁴× below MRTA, so linear mixing converges far too
  slowly; use `ibte_method: gmres` (with `ibte_niter` of a few hundred restart
  cycles).
- **Resolution.** The kernel cannot resolve q ≲ Δk, and the full IBTE comes out too
  low. In tests the IBTE/MRTA ratio dropped to 0.3–0.7 at β/Δk < 0.1 and was
  reliable from β ≳ Δk.

```yaml
ibte_mrta_mechanisms: [IMP]
```

With this option, IMP enters the IBTE only through its MRTA out-scattering rate (no
in-scattering), while the other mechanisms are solved exactly. The full IBTE is
reported as well (`mobility["IBTE_full"]`). For elastic, nearly isotropic scattering
the MRTA is accurate, so this is the recommended setting at low doping. Once β ≳ Δk,
`IBTE_full` is the better value.

### What to expect, mechanism by mechanism

**SERTA vs IBTE.** SERTA keeps only the out-scattering. The in-scattering term gives
back the part of the momentum that a collision does not destroy. For forward-peaked
or isotropic scattering this part is positive, so μ_IBTE ≥ μ_SERTA. This held for
every solution in our tests. It is not a theorem: scattering dominated by
back-scattering (cos θ < 0) would give μ_IBTE < μ_SERTA. A result with IBTE < SERTA
in a normal semiconductor therefore points to a problem
(`scripts/sanity_checks.py` checks this).

**MRTA vs IBTE.** The MRTA puts the factor (1 − v·v′/|v|²) in the out-scattering.
It is exact only for elastic scattering in an isotropic parabolic band, so
IBTE/MRTA can be on either side of 1:

- POP (inelastic): typically within about ±15%;
- warped, SOC-split valence bands with IMP: up to about +30%.

| mechanism | \|g\|² | in the IBTE |
|---|---|---|
| ADP | ~constant in q (elastic, ~isotropic) | full IBTE; small in-scattering gain |
| POP | ∝ 1/q² (inelastic) | full IBTE; the largest gains (IBTE/SERTA often 1.5–2 in polar materials). Use mode-resolved LO frequencies (`lo_phonons.py`) |
| PIE | ∝ 1/q² (elastic) | full IBTE, like POP; screened if `free_carrier_screening` (tested: detailed balance per doping) |
| IMP | ∝ 1/(q² + β²)² (elastic) | full IBTE only once β ≳ mesh spacing Δk (high doping or dense mesh). Otherwise use `ibte_mrta_mechanisms: [IMP]` |

PIE needs no special treatment. Its 1/q² is the same, integrable divergence as POP,
which the kernel resolves (validated against full ab initio IBTE calculations). IMP's 1/q⁴ at β → 0 is
different: the scattering concentrates at q ≲ β, below the mesh resolution.

**Free-carrier screening** lowers the weight of small-q POP/PIE scattering. That
small-q scattering barely relaxes momentum, and the IBTE already discounts it
through in-scattering. So screening changes the IBTE much less than the RTA (in
tests at 1e20 cm⁻³: +7% vs +19%).

## Limitations

- Mechanisms: ADP, PIE, POP and IMP, with or without free-carrier screening. Basic
  mechanisms (CRT, SRT, MFP) have no kernel and are refused. For phonon-limited
  comparisons use `scattering_type: [ADP, POP]`.
- Mobility only. Conductivity, Seebeck and electronic thermal conductivity are still
  RTA results (`rta_mode`), because the IBTE needs a second driving term
  ∝ (ε − μ)/T for them.
- `separate_mobility` gives RTA-only per-mechanism mobilities. Mechanisms do not add
  under the IBTE.
- AMSET's ADP matrix element uses the deformation potential of the initial state. The
  in-scattering term reuses |g(k, k′)|² with k as the initial state, as in Eq. (S12).
- States outside the Fermi–Dirac cut-offs, or without final states, keep
  Λ = τ⁰v. Their share of the transport weight is logged, with a warning above 1%.
- Spin-polarized systems loop over spins consistently but are untested.

## Convergence advice

The IBTE needs denser meshes than SERTA or MRTA (Claes *et al.*, SI Sec. II).
Converge `interpolation_factor` on the IBTE mobility: three consecutive meshes within
5%. `scripts/convergence_scan.py` runs such a scan and tabulates SERTA, MRTA, RTA
and IBTE together.

The kernel size grows roughly like (number of states in the window) × (number of
mesh points on a constant-energy surface), i.e. like N_k^(5/3) for N_k mesh points.
It is printed per band while the rates are computed. If memory is tight, reduce
`nworkers` (each worker holds its own buffers) or the energy window (`fd_tol`).
