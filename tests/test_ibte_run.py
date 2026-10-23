"""Integration tests of solver: ibte (validation ladder rungs 2, 3, 5 and 6).

Fast tests use small meshes: the numbers are not converged, but every identity
tested here holds at any mesh.
"""

import copy

import numpy as np
import pytest
from pymatgen.electronic_structure.core import Spin

from amset.constants import boltzmann_au
from amset.core.run import Runner
from amset.core.transport import _get_band_idx
from amset.ibte.run import _filled
from amset.ibte.solver import pop_occupation_factors, reconstruct_rates
from amset.ibte.transport import calculate_mobility_from_lambda

from .ibte_helpers import EXAMPLE_DIR, avg, cosine_band_structure, slow, solver_from_amset_data

GAAS_SETTINGS = {
    "doping": [-1e18],
    "temperatures": [300],
    "bandgap": 1.33,
    "interpolation_factor": 8,
    "scattering_type": ["ADP", "POP"],
    "deformation_potential": (1.2, 8.6),
    "elastic_constant": 139.7,
    "high_frequency_dielectric": 10.32,
    "static_dielectric": 12.18,
    "pop_frequency": 8.16,
    "wavefunction_coefficients": str(EXAMPLE_DIR / "GaAs" / "wavefunction.h5"),
    "nworkers": 4,
    "print_log": False,
    "write_log": False,
    "file_format": None,
    "separate_mobility": False,
}

# synthetic cosine-band model with ADP + POP and unity overlap: cheap, and with many
# states inside the transport window (unlike GaAs on a coarse mesh)
MODEL_SETTINGS = {
    "doping": [-1e19],
    "temperatures": [300],
    "interpolation_factor": 4,
    "scattering_type": ["ADP", "POP"],
    "deformation_potential": (5.0, 5.0),
    "elastic_constant": 100,
    "unity_overlap": True,
    "high_frequency_dielectric": 10,
    "static_dielectric": 15,
    "pop_frequency": 7,
    "nworkers": 4,
    "print_log": False,
    "write_log": False,
    "file_format": None,
    "separate_mobility": False,
}


def _run_vasprun(settings, directory="."):
    runner = Runner.from_vasprun(
        str(EXAMPLE_DIR / "GaAs" / "vasprun.xml.gz"), copy.deepcopy(settings)
    )
    return runner.run(directory=directory)


def _run_model(settings, directory="."):
    bs = cosine_band_structure(nk=6, t_c=0.1, t_v=0.1)
    return Runner(bs, 2, copy.deepcopy(settings)).run(directory=directory)


@pytest.fixture(scope="module")
def model_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model")
    runs = {}
    runs["ibte"] = _run_model(
        {**MODEL_SETTINGS, "solver": "ibte", "ibte_save_kernel": True}, tmp
    )
    for mode in ("amset", "serta", "mrta"):
        runs[mode] = _run_model({**MODEL_SETTINGS, "rta_mode": mode})
    runs["kernel_file"] = tmp / "ibte_kernel.h5"
    return runs


@pytest.fixture(scope="module")
def gaas_runs():
    return {
        "ibte": _run_vasprun({**GAAS_SETTINGS, "solver": "ibte"}),
        "amset": _run_vasprun(GAAS_SETTINGS),
    }


def test_gaas_overlap_path(gaas_runs):
    """With wavefunction overlaps: RTA rates identical, IBTE isotropic, > SERTA."""
    ibte = gaas_runs["ibte"].mobility
    np.testing.assert_allclose(
        ibte["overall"], gaas_runs["amset"].mobility["overall"], rtol=1e-12
    )
    tensor = ibte["IBTE"][0, 0]
    diag = np.diag(tensor)
    assert np.ptp(diag) / diag.mean() < 1e-8
    assert np.abs(tensor - np.diag(diag)).max() / diag.mean() < 1e-8
    assert avg(tensor) > avg(ibte["SERTA"][0, 0])


def test_rta_references_identical(model_runs):
    """The rates of the IBTE pass reproduce the three rta_mode runs exactly."""
    ibte = model_runs["ibte"].mobility
    np.testing.assert_allclose(
        ibte["overall"], model_runs["amset"].mobility["overall"], rtol=1e-12
    )
    np.testing.assert_allclose(
        ibte["SERTA"], model_runs["serta"].mobility["overall"], rtol=1e-12
    )
    np.testing.assert_allclose(
        ibte["MRTA"], model_runs["mrta"].mobility["overall"], rtol=1e-12
    )


def test_iteration_zero_is_serta(model_runs):
    """Rung 2: Λ = τ⁰ v reproduces rta_mode: serta."""
    amset_data = model_runs["ibte"]
    solver = solver_from_amset_data(amset_data, MODEL_SETTINGS)
    serta = _filled(amset_data, amset_data.ibte["raw_rates"]["serta"])
    band_idx = _get_band_idx(amset_data.energies, amset_data.vb_idx, amset_data.doping[0])
    lam, _ = solver.solve(0, 0, serta, band_idx=band_idx, inscattering=False)
    mob = calculate_mobility_from_lambda(amset_data, lam, 0, 0, band_idx)
    np.testing.assert_allclose(
        mob, model_runs["serta"].mobility["overall"][0, 0], rtol=1e-10, atol=1e-8
    )


def test_zero_mode_row_sums(model_runs):
    """Rows of τ⁰ K_in f⁰'(ε')/f⁰'(ε) sum to one (number conservation).

    τ⁰ comes from AMSET's own summation, the kernel from the trilinear mapping, and
    the occupation factors from the in-scattering table, so this checks all three.
    """
    amset_data = model_runs["ibte"]
    solver = solver_from_amset_data(amset_data, MODEL_SETTINGS)
    serta = _filled(amset_data, amset_data.ibte["raw_rates"]["serta"])
    spin = Spin.up
    tau = (1 / serta[spin][:, 0, 0].sum(axis=0))[:, amset_data.ir_kpoints_idx]
    active = solver.active_rows(spin)
    # states whose rate of some mechanism was zero (and then interpolated by AMSET)
    # have no consistent kernel row
    kernel = amset_data.ibte["kernel"]
    nb = kernel.nbands[spin]
    elastic = kernel.channels[spin]["elastic"].has_entries().reshape(nb, -1)
    pop = (
        kernel.channels[spin]["absorption"].has_entries()
        | kernel.channels[spin]["emission"].has_entries()
    ).reshape(nb, -1)
    full_rows = active & elastic & pop
    assert full_rows.sum() > 200

    matrix = solver.iteration_matrix(spin, 0, 0, tau, option="A", active=active)
    rowsum = np.asarray(matrix.sum(axis=1)).reshape(tau.shape)
    # 1e-7, not machine precision: AMSET converts ħω_po to Hartree in two slightly
    # different ways (energy shift via hbar in eV·s, n_po via s_to_au); the CODATA
    # constants differ at the 1e-9 level
    np.testing.assert_allclose(rowsum[full_rows], 1.0, rtol=1e-7)
    assert np.all(rowsum[~active] == 0)


def test_reconstructed_rates(model_runs):
    """Rates rebuilt from the stored kernel equal AMSET's summed rates."""
    amset_data = model_runs["ibte"]
    state = amset_data.ibte
    rates, computed = reconstruct_rates(
        state["kernel"], state["mech"], amset_data, amset_data.scattering_labels
    )
    for mode in ("serta", "mrta"):
        for spin, r in rates[mode].items():
            raw = state["raw_rates"][mode][spin][..., amset_data.ir_kpoints_idx]
            # where AMSET interpolated a zero rate the kernel row sum is zero
            mask = np.broadcast_to(computed[spin], r.shape) & (r > 0)
            assert mask.sum() > 200
            np.testing.assert_allclose(r[mask], raw[mask], rtol=1e-10)


def test_option_a_vs_b(model_runs):
    """Rung 5: iterating on Λ (A) and on ∂f/∂E (B) give the same mobility.

    They differ only through the trilinear interpolation of f⁰' Λ versus Λ, so
    agreement is to discretisation accuracy, not machine precision.
    """
    amset_data = model_runs["ibte"]
    solver = solver_from_amset_data(amset_data, {**MODEL_SETTINGS, "ibte_tol": 1e-6})
    serta = _filled(amset_data, amset_data.ibte["raw_rates"]["serta"])
    band_idx = _get_band_idx(amset_data.energies, amset_data.vb_idx, amset_data.doping[0])
    ratio_b = solver.solve_option_b(0, 0, serta, band_idx=band_idx)[Spin.up]
    _, info = solver.solve(0, 0, serta, band_idx=band_idx)
    ratio_a = info["iterations"][0]["history"][-1][1]
    assert ratio_a > 1
    assert abs(ratio_b / ratio_a - 1) < 0.05


def test_iterate_vs_gmres(model_runs):
    amset_data = model_runs["ibte"]
    serta = _filled(amset_data, amset_data.ibte["raw_rates"]["serta"])
    band_idx = _get_band_idx(amset_data.energies, amset_data.vb_idx, amset_data.doping[0])
    mobs = {}
    for method in ("iterate", "gmres"):
        solver = solver_from_amset_data(
            amset_data, {**MODEL_SETTINGS, "ibte_method": method, "ibte_tol": 1e-6}
        )
        lam, info = solver.solve(0, 0, serta, band_idx=band_idx)
        assert info["converged"]
        mobs[method] = calculate_mobility_from_lambda(amset_data, lam, 0, 0, band_idx)
    np.testing.assert_allclose(mobs["iterate"], mobs["gmres"], rtol=1e-4, atol=1e-6)


def test_ibte_polar_enhancement_and_isotropy(model_runs):
    mob = model_runs["ibte"].mobility
    ibte = mob["IBTE"][0, 0]
    serta = mob["SERTA"][0, 0]
    assert avg(ibte) > avg(serta)
    # cubic: isotropic and diagonal
    diag = np.diag(ibte)
    assert np.ptp(diag) / diag.mean() < 1e-8
    assert np.abs(ibte - np.diag(diag)).max() / diag.mean() < 1e-8


def test_detailed_balance_statistical(model_runs):
    """Microscopic reversibility of the stored POP kernel.

    With |g(k,k')|² symmetric, the bare absorption flux out of an energy window W,
    Σ_{k∈W} Γ_abs(k), equals the bare emission flux Σ_{k'∈W+ħω} Γ_em(k'). Swapping
    the absorption/emission channels or the sign of ħω breaks this.
    """
    amset_data = model_runs["ibte"]
    state = amset_data.ibte
    kernel, mech = state["kernel"], state["mech"]
    spin = Spin.up
    n_ir = kernel.n_ir_kpoints
    mult = np.bincount(amset_data.ir_to_full_kpoint_mapping, minlength=n_ir)
    ir_e = amset_data.energies[spin][:, amset_data.ir_kpoints_idx]
    gamma_abs = kernel.channels[spin]["absorption"].rowsum[0].reshape(ir_e.shape)
    gamma_em = kernel.channels[spin]["emission"].rowsum[0].reshape(ir_e.shape)
    omega = mech.omega["absorption"]

    cbm = ir_e[amset_data.vb_idx[spin] + 1 :].min()
    emax = amset_data.fd_cutoffs[1]
    width = 0.08 / 27.2114  # 80 meV smooth weight: sharp bins are dominated by the
    # discrete energies of the few irreducible states they contain

    def phi(e):
        top = emax - omega - 0.002
        taper = np.clip((top - e) / (4 * width), 0, 1)
        return np.where((e >= cbm) & (e <= top), np.exp(-(e - cbm) / width) * taper, 0)

    flux_up = np.sum(mult[None] * gamma_abs * phi(ir_e))
    flux_down = np.sum(mult[None] * gamma_em * phi(ir_e - omega))
    wrong = np.sum(mult[None] * gamma_em * phi(ir_e + omega))
    assert flux_up > 0
    # ~5% residual from band-edge discretisation on this coarse mesh
    assert abs(flux_down / flux_up - 1) < 0.10
    assert wrong / flux_up < 0.5


def test_load_kernel(model_runs):
    """ibte_load_kernel reproduces the run without recomputing rates."""
    loaded = _run_model(
        {
            **MODEL_SETTINGS,
            "solver": "ibte",
            "ibte_load_kernel": str(model_runs["kernel_file"]),
        }
    )
    ref = model_runs["ibte"].mobility
    for key in ("IBTE", "SERTA", "MRTA", "overall"):
        np.testing.assert_allclose(
            loaded.mobility[key], ref[key], rtol=1e-8, atol=1e-8 * avg(ref[key][0, 0])
        )


def test_unsupported_settings():
    # basic mechanisms (constant relaxation time etc.) have no kernel
    with pytest.raises(ValueError, match="CRT"):
        Runner.from_vasprun(
            str(EXAMPLE_DIR / "GaAs" / "vasprun.xml.gz"),
            {
                **GAAS_SETTINGS,
                "solver": "ibte",
                "scattering_type": ["ADP", "CRT"],
                "constant_relaxation_time": 1e-14,
            },
        )
    with pytest.raises(ValueError, match="auto"):
        Runner.from_vasprun(
            str(EXAMPLE_DIR / "GaAs" / "vasprun.xml.gz"),
            {
                **GAAS_SETTINGS,
                "solver": "ibte",
                "scattering_type": "auto",
                "constant_relaxation_time": 1e-14,
            },
        )
    # IMP and free-carrier screening are supported (doping-dependent kernel)
    Runner.from_vasprun(
        str(EXAMPLE_DIR / "GaAs" / "vasprun.xml.gz"),
        {
            **GAAS_SETTINGS,
            "solver": "ibte",
            "scattering_type": ["ADP", "IMP", "POP"],
            "free_carrier_screening": True,
        },
    )


def test_analytic_isotropic_model():
    """Rung 3: constant matrix element ⇒ in-scattering vanishes ⇒ IBTE = SERTA = MRTA.

    Cosine bands on a simple cubic lattice, ADP with an isotropic elastic tensor and
    unity overlap: |g|² is constant on each energy surface and Σ_k' v(k') = 0 by
    inversion symmetry.
    """
    settings = {
        "doping": [-1e19, 1e19],
        "temperatures": [300],
        "interpolation_factor": 4,
        "scattering_type": ["ADP"],
        "deformation_potential": (5.0, 5.0),
        "elastic_constant": 100,
        "unity_overlap": True,
        "solver": "ibte",
        "nworkers": 4,
        "print_log": False,
        "write_log": False,
        "file_format": None,
        "ibte_tol": 1e-8,
    }
    mob = _run_model(settings).mobility
    for n in range(2):
        ibte, serta, mrta = (avg(mob[k][n, 0]) for k in ("IBTE", "SERTA", "MRTA"))
        # not machine precision: AMSET's constant-energy-surface quadrature is not
        # exactly inversion symmetric (e.g. tetrahedron vertices with equal energies
        # can be ordered differently in equivalent tetrahedra). Upstream's own MRTA
        # shows the same ~1e-4 deviation from SERTA in this model; the in-scattering
        # term is ~3e-5 of τ⁰v per state (median).
        assert abs(ibte / serta - 1) < 5e-4
        assert abs(mrta / serta - 1) < 5e-4


@slow
def test_si_elastic_close_to_mrta():
    """Rung 4: Si, ADP only (deformation.h5): IBTE close to MRTA, isotropic."""
    settings = {
        "doping": [-1e16],
        "temperatures": [300],
        "bandgap": 1.14,
        "interpolation_factor": 20,
        "scattering_type": ["ADP"],
        "deformation_potential": str(EXAMPLE_DIR / "Si" / "deformation.h5"),
        "elastic_constant": [
            [144, 53, 53, 0, 0, 0],
            [53, 144, 53, 0, 0, 0],
            [53, 53, 144, 0, 0, 0],
            [0, 0, 0, 75, 0, 0],
            [0, 0, 0, 0, 75, 0],
            [0, 0, 0, 0, 0, 75],
        ],
        "wavefunction_coefficients": str(EXAMPLE_DIR / "Si" / "wavefunction.h5"),
        "solver": "ibte",
        "nworkers": 4,
        "print_log": False,
        "write_log": False,
        "file_format": None,
    }
    runner = Runner.from_vasprun(str(EXAMPLE_DIR / "Si" / "vasprun.xml.gz"), settings)
    mob = runner.run().mobility
    ibte = mob["IBTE"][0, 0]
    assert abs(avg(ibte) / avg(mob["MRTA"][0, 0]) - 1) < 0.05
    diag = np.diag(ibte)
    assert np.ptp(diag) / diag.mean() < 1e-3


@slow
def test_gaas_polar_ratio():
    """Rung 6: n-GaAs, ADP + POP, 300 K: μ_IBTE/μ_SERTA in the 1.4–2.6 range.

    AMSET's single-LO-mode model gives ≈1.8 (83³ mesh);
    published ab initio values are 1.2–1.4. Factor 30 fits in ~4 GB of memory.
    """
    settings = {**GAAS_SETTINGS, "doping": [-1e16], "interpolation_factor": 30}
    mob = _run_vasprun({**settings, "solver": "ibte"}).mobility
    ratio = avg(mob["IBTE"][0, 0]) / avg(mob["SERTA"][0, 0])
    assert 1.4 < ratio < 2.6
