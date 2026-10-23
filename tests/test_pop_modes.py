"""Mode-resolved POP scattering (pop_frequency list + pop_weights)."""

import numpy as np
import pytest
from pymatgen.electronic_structure.core import Spin

from amset.ibte.run import _filled
from amset.util import _validate_pop_modes

from .ibte_helpers import avg, solver_from_amset_data
from .test_ibte_run import MODEL_SETTINGS, _run_model


def test_validate_pop_modes():
    s = {"pop_frequency": 8.0, "pop_weights": None}
    _validate_pop_modes(s)
    assert s["pop_frequency"] == 8.0

    s = {"pop_frequency": [4, 10], "pop_weights": [1, 3]}
    _validate_pop_modes(s)
    assert s["pop_frequency"] == [4.0, 10.0]
    np.testing.assert_allclose(s["pop_weights"], [0.25, 0.75])

    s = {"pop_frequency": [7.0], "pop_weights": [2.0]}
    _validate_pop_modes(s)
    assert s["pop_frequency"] == 7.0 and s["pop_weights"] is None

    for bad in (
        {"pop_frequency": [4, 10], "pop_weights": None},
        {"pop_frequency": [4, 10], "pop_weights": [1]},
        {"pop_frequency": 7.0, "pop_weights": [1.0]},
        {"pop_frequency": [4, -1], "pop_weights": [1, 1]},
    ):
        with pytest.raises(ValueError):
            _validate_pop_modes(bad)


@pytest.fixture(scope="module")
def mode_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("modes")
    base = {**MODEL_SETTINGS, "solver": "ibte", "separate_mobility": True}
    runs = {
        "single": _run_model({**base, "pop_frequency": 7.0}),
        "split": _run_model(
            {**base, "pop_frequency": [7.0, 7.0], "pop_weights": [0.3, 0.7]}
        ),
        "two": _run_model(
            {
                **base,
                "pop_frequency": [3.0, 10.0],
                "pop_weights": [0.4, 0.6],
                "ibte_save_kernel": True,
            },
            tmp,
        ),
    }
    runs["kernel_file"] = tmp / "ibte_kernel.h5"
    return runs


def test_split_mode_equals_single_mode(mode_runs):
    """A mode split into two identical-frequency halves changes nothing."""
    single, split = mode_runs["single"].mobility, mode_runs["split"].mobility
    for key in ("overall", "SERTA", "MRTA", "IBTE", "ADP"):
        np.testing.assert_allclose(
            split[key], single[key], rtol=1e-9, atol=1e-9 * avg(single[key][0, 0])
        )
    assert "POP1" in split and "POP2" in split
    # rates add up: 1/μ_POP = 1/μ_POP1 + 1/μ_POP2 holds for the summed rates
    spin = Spin.up
    r_single = mode_runs["single"].scattering_rates[spin]
    r_split = mode_runs["split"].scattering_rates[spin]
    labels = mode_runs["split"].scattering_labels
    pop = r_split[labels.index("POP1")] + r_split[labels.index("POP2")]
    ref = r_single[mode_runs["single"].scattering_labels.index("POP")]
    np.testing.assert_allclose(pop, ref, rtol=1e-9)


def test_two_modes_zero_mode_and_channels(mode_runs):
    """Each mode has its own ħω and n: the zero-mode row sums still equal 1."""
    amset_data = mode_runs["two"]
    kernel = amset_data.ibte["kernel"]
    spin = Spin.up
    assert set(kernel.channels[spin]) == {
        "elastic", "absorption_0", "emission_0", "absorption_1", "emission_1"
    }
    mech = amset_data.ibte["mech"]
    assert mech.omega["absorption_0"] < mech.omega["absorption_1"]
    assert mech.n_po["absorption_0"][0] > mech.n_po["absorption_1"][0]

    solver = solver_from_amset_data(amset_data, MODEL_SETTINGS)
    serta = _filled(amset_data, amset_data.ibte["raw_rates"]["serta"])
    tau = (1 / serta[spin][:, 0, 0].sum(axis=0))[:, amset_data.ir_kpoints_idx]
    active = solver.active_rows(spin)
    nb = kernel.nbands[spin]
    complete = kernel.channels[spin]["elastic"].has_entries().reshape(nb, -1)
    for mode in (0, 1):
        complete &= (
            kernel.channels[spin][f"absorption_{mode}"].has_entries()
            | kernel.channels[spin][f"emission_{mode}"].has_entries()
        ).reshape(nb, -1)
    rows = active & complete
    assert rows.sum() > 200
    matrix = solver.iteration_matrix(spin, 0, 0, tau, option="A", active=active)
    rowsum = np.asarray(matrix.sum(axis=1)).reshape(tau.shape)
    np.testing.assert_allclose(rowsum[rows], 1.0, rtol=1e-7)

    mob = amset_data.mobility
    assert avg(mob["IBTE"][0, 0]) > avg(mob["SERTA"][0, 0])


def test_two_modes_kernel_load(mode_runs):
    loaded = _run_model(
        {
            **MODEL_SETTINGS,
            "solver": "ibte",
            "pop_frequency": [3.0, 10.0],
            "pop_weights": [0.4, 0.6],
            "ibte_load_kernel": str(mode_runs["kernel_file"]),
        }
    )
    ref = mode_runs["two"].mobility
    for key in ("IBTE", "SERTA", "MRTA", "overall"):
        np.testing.assert_allclose(
            loaded.mobility[key], ref[key], rtol=1e-8, atol=1e-8 * avg(ref[key][0, 0])
        )


def _modes_file(path, coefficients, exponents):
    import json

    path.write_text(json.dumps({
        "frequencies": [3.0, 10.0],
        "weights": [0.4, 0.6],
        "exponents": exponents,
        "coefficients": coefficients,
    }))
    return str(path)


def test_isotropic_expansion_equals_constant_weights(mode_runs, tmp_path):
    """A pop_modes_file with only the constant term reproduces pop_weights."""
    f = _modes_file(tmp_path / "iso.json", [[0.4], [0.6]], [[0, 0, 0]])
    run = _run_model({**MODEL_SETTINGS, "solver": "ibte", "pop_modes_file": f})
    ref = mode_runs["two"].mobility
    for key in ("IBTE", "SERTA", "MRTA", "overall"):
        np.testing.assert_allclose(
            run.mobility[key], ref[key], rtol=1e-9, atol=1e-9 * avg(ref[key][0, 0])
        )


def test_anisotropic_weights(tmp_path):
    """Direction-dependent weights w1 = q_z², w2 = 1 − q_z²: Σ w = 1 along every q̂,
    so the zero-mode row sums stay 1, but scattering becomes anisotropic."""
    f = _modes_file(
        tmp_path / "aniso.json", [[0.0, 1.0], [1.0, -1.0]], [[0, 0, 0], [0, 0, 2]]
    )
    amset_data = _run_model({**MODEL_SETTINGS, "solver": "ibte", "pop_modes_file": f})
    pop = [s for s in amset_data.ibte["mech"].labels if s.startswith("absorption")]
    assert len(pop) == 2

    spin = Spin.up
    solver = solver_from_amset_data(amset_data, MODEL_SETTINGS)
    serta = _filled(amset_data, amset_data.ibte["raw_rates"]["serta"])
    tau = (1 / serta[spin][:, 0, 0].sum(axis=0))[:, amset_data.ir_kpoints_idx]
    active = solver.active_rows(spin)
    kernel = amset_data.ibte["kernel"]
    nb = kernel.nbands[spin]
    complete = active.copy()
    for c, ch in kernel.channels[spin].items():
        if c == "elastic" or c.startswith("absorption"):
            complete &= ch.has_entries().reshape(nb, -1)
    matrix = solver.iteration_matrix(spin, 0, 0, tau, option="A", active=active)
    rowsum = np.asarray(matrix.sum(axis=1)).reshape(tau.shape)
    assert complete.sum() > 100
    np.testing.assert_allclose(rowsum[complete], 1.0, rtol=1e-7)

    # The direction dependence is active: compared with constant weights equal to
    # the direction averages (<q_z²> = 1/3), the result changes. (The tensor itself
    # stays isotropic: AMSET symmetrizes with the cubic point group, and weights
    # computed from DFPT always respect the crystal symmetry anyway.)
    iso = _run_model(
        {**MODEL_SETTINGS, "solver": "ibte", "pop_frequency": [3.0, 10.0],
         "pop_weights": [1 / 3, 2 / 3]}
    )
    for key in ("SERTA", "IBTE"):
        ratio = avg(amset_data.mobility[key][0, 0]) / avg(iso.mobility[key][0, 0])
        assert abs(ratio - 1) > 1e-3
