"""Tests of the kernel building blocks: trilinear mapping, symmetry, storage."""

import numpy as np
import pytest
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import Vasprun

from amset.electronic_structure.common import get_band_structure
from amset.electronic_structure.kpoints import kpoints_to_first_bz
from amset.ibte.kernel import IBTEKernel, KernelChannel, TrilinearMapper
from amset.ibte.symmetry import check_velocity_unfolding, get_full_kpoint_rotations
from amset.interpolation.bandstructure import Interpolator

from .ibte_helpers import EXAMPLE_DIR, cosine_band_structure


def _amset_data(bs, nelect, factor):
    interp = Interpolator(bs, nelect, interpolation_factor=factor)
    return interp.get_amset_data(energy_cutoff=1.5, nworkers=1)


@pytest.fixture(scope="module")
def gaas_data():
    vr = Vasprun(EXAMPLE_DIR / "GaAs" / "vasprun.xml.gz", parse_projected_eigen=False)
    bs = get_band_structure(vr)
    return _amset_data(bs, vr.parameters["NELECT"], 3)


@pytest.fixture(scope="module")
def cubic_data():
    return _amset_data(cosine_band_structure(nk=8), 2, 3)


@pytest.mark.parametrize("which", ["gaas_data", "cubic_data"])
def test_trilinear_matches_mrta_interpolator(which, request):
    """Spec 3.2: applying the weights to v reproduces MRTACalculator to ~1e-10."""
    amset_data = request.getfixturevalue(which)
    spin = Spin.up
    rng = np.random.default_rng(0)
    nb = len(amset_data.energies[spin])
    kpts = rng.uniform(-0.5, 0.5, size=(2000, 3))
    # include zone-boundary points, grid points and their neighbourhood
    kpts = np.concatenate(
        [kpts, amset_data.kpoints[:200], amset_data.kpoints[:200] + 1e-7, [[0.5, 0.5, 0.5]]]
    )
    kpts = kpoints_to_first_bz(kpts)
    bands = rng.integers(0, nb, size=len(kpts))

    mapper = TrilinearMapper.from_kpoints(amset_data.kpoints)
    ours = mapper.interpolate(amset_data.velocities[spin], bands, kpts)
    ref = amset_data.mrta_calculator.interpolate(spin, bands, kpts)
    scale = np.abs(ref).max()
    assert np.abs(ours - ref).max() / scale < 1e-10

    idx, w = mapper.weights(kpts)
    np.testing.assert_allclose(w.sum(axis=1), 1, rtol=1e-12)
    assert idx.min() >= 0 and idx.max() < len(amset_data.kpoints)


def test_map_values_conserves_sums(gaas_data):
    mapper = TrilinearMapper.from_kpoints(gaas_data.kpoints)
    rng = np.random.default_rng(1)
    kpts = rng.uniform(-0.5, 0.5, size=(500, 3))
    bands = rng.integers(0, 3, size=500)
    values = rng.uniform(size=(2, 500))
    cols, row_values = mapper.map_values(kpts, bands, values)
    np.testing.assert_allclose(row_values.sum(axis=1), values.sum(axis=1))
    assert len(np.unique(cols)) == len(cols)
    # values interpolated with the row reproduce the pointwise interpolation
    data = rng.normal(size=(3, len(gaas_data.kpoints)))
    pointwise = np.sum(values * mapper.interpolate(data[..., None], bands, kpts)[:, 0], 1)
    flat = data.ravel()
    np.testing.assert_allclose(row_values @ flat[cols], pointwise, rtol=1e-10)


@pytest.mark.parametrize("which", ["gaas_data", "cubic_data"])
def test_velocity_unfolding(which, request):
    """Spec 3.3: S v(k_ir) reproduces the FFT velocities on the full mesh."""
    amset_data = request.getfixturevalue(which)
    rotations = get_full_kpoint_rotations(amset_data, symprec=0.01, time_reversal=True)
    np.testing.assert_allclose(
        np.einsum("kij,kjl->kil", rotations, rotations.transpose(0, 2, 1)),
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1e-10,
    )
    for median, p99, _ in check_velocity_unfolding(amset_data, rotations).values():
        assert median < 1e-10
        assert p99 < 1e-8


def test_kernel_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    nrows, ncols = 6, 20
    rows = []
    for r in (0, 2, 5):
        cols = np.sort(rng.choice(ncols, size=4, replace=False)).astype(np.int32)
        rows.append((r, cols, rng.uniform(size=(2, 4)), rng.uniform(size=2)))
    rows.append((3, np.zeros(0, dtype=np.int32), np.zeros((2, 0)), np.zeros(2)))
    ch = KernelChannel.from_rows(rows, nrows, ncols, 2)
    dense = ch.matrix(np.ones((2, nrows))).toarray()
    for r, cols, vals, _ in rows:
        np.testing.assert_allclose(dense[r, cols], vals.sum(axis=0))
    np.testing.assert_allclose(ch.rowsum.sum(axis=0), dense.sum(axis=1))
    assert list(ch.computed) == [True, False, True, True, False, True]

    kernel = IBTEKernel(
        {Spin.up: {"elastic": ch}},
        {"elastic": ["ADP", "PIE"], "absorption": [], "emission": []},
        {Spin.up: 3},
        2,
        10,
        {"mesh": [1, 2, 5]},
    )
    kernel.to_file(tmp_path / "k.h5")
    loaded = IBTEKernel.from_file(tmp_path / "k.h5")
    lch = loaded.channels[Spin.up]["elastic"]
    np.testing.assert_array_equal(lch.indptr, ch.indptr)
    np.testing.assert_array_equal(lch.indices, ch.indices)
    np.testing.assert_array_equal(lch.data, ch.data)
    np.testing.assert_array_equal(lch.computed, ch.computed)
    assert loaded.mechanisms == kernel.mechanisms
    assert loaded.metadata == kernel.metadata
