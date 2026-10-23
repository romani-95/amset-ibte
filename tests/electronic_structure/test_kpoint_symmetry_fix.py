"""fix_kpoint_symmetry: spglib must receive the lattice vectors as rows."""

import numpy as np
from monty.serialization import loadfn

from amset.electronic_structure.kpoints import get_kpoints_tetrahedral
from amset.electronic_structure.symmetry import get_reciprocal_point_group_operations


def _maps_by_true_symmetry(structure, full, ir_idx, ir_to_full, tol=1e-6):
    rotations, _, _ = get_reciprocal_point_group_operations(structure)
    ir_of_full = full[ir_idx][ir_to_full]
    ok = np.zeros(len(full), dtype=bool)
    for r in rotations:
        diff = ir_of_full @ r.T - full
        ok |= np.all(np.abs(diff - np.round(diff)) < tol, axis=1)
    return ok.all()


def test_hexagonal_mesh_reduction(test_dir):
    structure = loadfn(test_dir / "structures" / "ZnO_186.json.gz")
    mesh = [12, 12, 8]
    old = get_kpoints_tetrahedral(mesh, structure)
    new = get_kpoints_tetrahedral(mesh, structure, fix_kpoint_symmetry=True)
    n_full = np.prod(mesh)
    # upstream (transposed lattice) misses most of the hexagonal symmetry
    assert len(new[0]) < len(old[0]) / 3
    assert len(new[0]) < n_full / 8
    # both mappings are valid (every point maps to its ir point by a true operation)
    for res in (old, new):
        assert _maps_by_true_symmetry(structure, res[2], res[3], res[4])
    # weights sum to the full mesh
    assert new[1].sum() == n_full


def test_default_unchanged_for_symmetric_lattice(test_dir):
    """For a symmetric lattice matrix (FCC primitive cell) the fix changes nothing."""
    structure = loadfn(test_dir / "structures" / "Si_227.json.gz")
    old = get_kpoints_tetrahedral([10, 10, 10], structure)
    new = get_kpoints_tetrahedral([10, 10, 10], structure, fix_kpoint_symmetry=True)
    if np.allclose(structure.lattice.matrix, structure.lattice.matrix.T):
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a, b)
    else:
        assert len(new[0]) <= len(old[0])
