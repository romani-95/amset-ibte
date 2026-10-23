"""IBTE with doping-dependent kernels: ionized impurities and free-carrier screening.

IMP, and POP/PIE with free-carrier screening, contain the screening length
1/(q² + β²) with β depending on doping and temperature. Their kernel has one entry
per (doping, temperature); all other mechanisms keep a single entry.
"""

import numpy as np
import pytest
from pymatgen.electronic_structure.core import Spin

from amset.ibte.kernel import IBTEKernel, entry_layout
from amset.ibte.run import _filled

from .ibte_helpers import avg, solver_from_amset_data
from .test_ibte_run import MODEL_SETTINGS, _run_model

DOPINGS = [-1e18, -1e19]
IMP_SETTINGS = {
    **MODEL_SETTINGS,
    "solver": "ibte",
    "doping": DOPINGS,
    "scattering_type": ["ADP", "IMP", "POP"],
}


def _zero_mode_rowsums(amset_data, settings, n, t):
    """Row sums of the option-A iteration matrix with Λ = τ⁰·1 (should be 1)."""
    spin = Spin.up
    solver = solver_from_amset_data(amset_data, settings)
    serta = _filled(amset_data, amset_data.ibte["raw_rates"]["serta"])
    tau = (1 / serta[spin][:, n, t].sum(axis=0))[:, amset_data.ir_kpoints_idx]
    active = solver.active_rows(spin)
    kernel = amset_data.ibte["kernel"]
    nb = kernel.nbands[spin]
    complete = active.copy()
    for c, ch in kernel.channels[spin].items():
        if c == "elastic" or c.startswith("absorption"):
            complete &= ch.has_entries().reshape(nb, -1)
    matrix = solver.iteration_matrix(spin, n, t, tau, option="A", active=active)
    rowsum = np.asarray(matrix.sum(axis=1)).reshape(tau.shape)
    return rowsum[complete], complete.sum()


@pytest.fixture(scope="module")
def imp_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("imp")
    runs = {
        "multi": _run_model({**IMP_SETTINGS, "ibte_save_kernel": True}, tmp),
        "single": [_run_model({**IMP_SETTINGS, "doping": [d]}) for d in DOPINGS],
    }
    runs["kernel_file"] = tmp / "ibte_kernel.h5"
    return runs


def test_entry_layout():
    index = entry_layout([1, 4, 1], 2, 2)
    assert index.shape == (3, 2, 2)
    np.testing.assert_array_equal(index[0], 0)
    np.testing.assert_array_equal(index[1], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(index[2], 5)
    with pytest.raises(ValueError):
        entry_layout([3], 2, 2)


def test_imp_kernel_entries(imp_runs):
    """ADP keeps one entry; IMP gets one per doping; POP (unscreened) one."""
    kernel = imp_runs["multi"].ibte["kernel"]
    assert kernel.mechanisms["elastic"] == ["ADP", "IMP"]
    np.testing.assert_array_equal(kernel.entry_index["elastic"][:, :, 0], [[0, 0], [1, 2]])
    assert kernel.channels[Spin.up]["elastic"].nmech == 3
    assert kernel.channels[Spin.up]["absorption"].nmech == 1
    assert kernel.doping_dependent


def test_imp_zero_mode(imp_runs):
    """Detailed balance (Σ_k' K = out-scattering rate) holds for each doping."""
    for n in range(len(DOPINGS)):
        rowsums, nrows = _zero_mode_rowsums(imp_runs["multi"], IMP_SETTINGS, n, 0)
        assert nrows > 100
        np.testing.assert_allclose(rowsums, 1.0, rtol=1e-7)


def test_multi_doping_equals_single_doping_runs(imp_runs):
    """A two-doping run equals two one-doping runs (the energy cut-offs, set by all
    dopings together, differ slightly, so only states of negligible weight change)."""
    multi = imp_runs["multi"].mobility
    for n, single in enumerate(imp_runs["single"]):
        for key in ("IBTE", "SERTA", "MRTA", "overall"):
            np.testing.assert_allclose(
                avg(multi[key][n, 0]), avg(single.mobility[key][0, 0]), rtol=1e-2
            )


def test_imp_kernel_load(imp_runs):
    loaded = _run_model(
        {**IMP_SETTINGS, "ibte_load_kernel": str(imp_runs["kernel_file"])}
    )
    ref = imp_runs["multi"].mobility
    for key in ("IBTE", "SERTA", "MRTA", "overall"):
        for n in range(len(DOPINGS)):
            np.testing.assert_allclose(
                loaded.mobility[key][n], ref[key][n], rtol=1e-8,
                atol=1e-8 * avg(ref[key][n, 0]),
            )
    stored = IBTEKernel.from_file(imp_runs["kernel_file"])
    assert stored.doping_dependent
    with pytest.raises(ValueError):
        _run_model(
            {
                **IMP_SETTINGS,
                "doping": [-1e18, -2e19],
                "ibte_load_kernel": str(imp_runs["kernel_file"]),
            }
        )


def test_imp_ibte_close_to_mrta():
    """For elastic scattering in a (near-)parabolic isotropic band the MRTA is
    close to the exact solution; the IBTE must land near it, above SERTA."""
    run = _run_model({**IMP_SETTINGS, "scattering_type": ["ADP", "IMP"]})
    for n in range(len(DOPINGS)):
        ibte = avg(run.mobility["IBTE"][n, 0])
        mrta = avg(run.mobility["MRTA"][n, 0])
        serta = avg(run.mobility["SERTA"][n, 0])
        assert ibte > serta
        assert abs(ibte / mrta - 1) < 0.05


def test_screened_pop_zero_mode():
    """Free-carrier screening: POP kernel entries per doping; detailed balance."""
    settings = {
        **IMP_SETTINGS,
        "scattering_type": ["ADP", "POP"],
        "free_carrier_screening": True,
    }
    run = _run_model(settings)
    kernel = run.ibte["kernel"]
    assert kernel.channels[Spin.up]["absorption"].nmech == len(DOPINGS)
    assert kernel.channels[Spin.up]["elastic"].nmech == 1
    for n in range(len(DOPINGS)):
        rowsums, nrows = _zero_mode_rowsums(run, settings, n, 0)
        assert nrows > 100
        np.testing.assert_allclose(rowsums, 1.0, rtol=1e-7)
    # screening weakens POP: higher mobility than the unscreened run
    unscreened = _run_model({**settings, "free_carrier_screening": False})
    n = DOPINGS.index(-1e19)
    assert avg(run.mobility["IBTE"][n, 0]) > avg(unscreened.mobility["IBTE"][n, 0])


def test_imp_in_mrta_hybrid():
    """ibte_mrta_mechanisms: [IMP]: IMP enters through its MRTA rate only.

    With IMP alone there is no in-scattering left, so the IBTE equals the MRTA.
    With all mechanisms the full IBTE is kept as IBTE_full.
    """
    run = _run_model(
        {**IMP_SETTINGS, "scattering_type": ["IMP"], "ibte_mrta_mechanisms": ["IMP"]}
    )
    for n in range(len(DOPINGS)):
        np.testing.assert_allclose(
            avg(run.mobility["IBTE"][n, 0]), avg(run.mobility["MRTA"][n, 0]), rtol=1e-6
        )
    assert "IBTE_full" in run.mobility

    run = _run_model({**IMP_SETTINGS, "ibte_mrta_mechanisms": ["IMP"],
                      "ibte_method": "gmres"})
    for n in range(len(DOPINGS)):
        hybrid = avg(run.mobility["IBTE"][n, 0])
        full = avg(run.mobility["IBTE_full"][n, 0])
        assert hybrid > avg(run.mobility["SERTA"][n, 0])
        assert full > avg(run.mobility["SERTA"][n, 0])
        assert abs(hybrid / full - 1) < 0.2


@pytest.mark.parametrize("screening", [False, True])
def test_piezoelectric_ibte(screening):
    """PIE (|g|² ∝ 1/q², like POP but elastic) in the IBTE: detailed balance for each
    doping, IBTE ≥ SERTA, and one kernel entry per doping only with screening."""
    settings = {
        **IMP_SETTINGS,
        "scattering_type": ["ADP", "PIE"],
        # zinc-blende-like e14 (C/m²)
        "piezoelectric_constant": [
            [0, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0.5],
        ],
        "free_carrier_screening": screening,
    }
    run = _run_model(settings)
    kernel = run.ibte["kernel"]
    assert kernel.mechanisms["elastic"] == ["ADP", "PIE"]
    expected = 1 + (len(DOPINGS) if screening else 1)
    assert kernel.channels[Spin.up]["elastic"].nmech == expected
    for n in range(len(DOPINGS)):
        rowsums, nrows = _zero_mode_rowsums(run, settings, n, 0)
        assert nrows > 100
        np.testing.assert_allclose(rowsums, 1.0, rtol=1e-7)
        assert avg(run.mobility["IBTE"][n, 0]) >= avg(run.mobility["SERTA"][n, 0])
