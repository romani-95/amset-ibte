"""Unfolding of vector quantities from irreducible k-points to the full mesh.

AMSET computes scattering rates on the irreducible k-points only and copies them to
the full mesh with ``ir_to_full_kpoint_mapping``. That is correct for scalars but
not for vectors such as Λ = (∂f/∂E)/(∂f⁰/∂ε): for a full-mesh point k = R k_ir,
Λ(k) = S Λ(k_ir), where S is the Cartesian form of the point-group operation R
(time reversal is included as -R, giving Λ(-k) = -Λ(k)). This is the same
transformation as for group velocities.
"""

import logging

import numpy as np

from amset.electronic_structure.common import get_angstrom_structure
from amset.electronic_structure.symmetry import get_reciprocal_point_group_operations

logger = logging.getLogger(__name__)


def cartesian_rotations(reciprocal_lattice: np.ndarray, rotations: np.ndarray):
    """Convert rotations acting on fractional reciprocal coords to Cartesian.

    With rows of ``reciprocal_lattice`` the reciprocal lattice vectors B,
    k_cart = Bᵀ k_frac, so k'_frac = R k_frac implies k'_cart = Bᵀ R B⁻ᵀ k_cart.
    """
    bt = reciprocal_lattice.T
    bt_inv = np.linalg.inv(bt)
    return np.einsum("ij,njk,kl->nil", bt, rotations, bt_inv)


def get_full_kpoint_rotations(amset_data, symprec=0.01, time_reversal=True, tol=1e-4):
    """Get, for every full-mesh k-point, the Cartesian rotation from its ir point.

    Args:
        amset_data: AmsetData object.
        symprec: Symmetry precision in Å (as in the AMSET settings).
        time_reversal: Whether time-reversal symmetry is present (no SOC).
        tol: Tolerance on fractional coordinates when matching k-points.

    Returns:
        Rotations with shape (nkpoints, 3, 3) such that
        ``X(k) = S[k] @ X(k_ir(k))`` for any vector quantity X.
    """
    structure = get_angstrom_structure(amset_data.structure)
    rotations, _, _ = get_reciprocal_point_group_operations(
        structure, symprec=symprec, time_reversal=time_reversal
    )
    kpoints = amset_data.kpoints
    ir_kpoints = kpoints[amset_data.ir_kpoints_idx]
    ir_of_full = ir_kpoints[amset_data.ir_to_full_kpoint_mapping]

    op_index = np.full(len(kpoints), -1, dtype=int)
    for i, r in enumerate(rotations):
        todo = op_index < 0
        if not np.any(todo):
            break
        diff = ir_of_full[todo] @ r.T - kpoints[todo]
        match = np.all(np.abs(diff - np.round(diff)) < tol, axis=1)
        idx = np.where(todo)[0][match]
        op_index[idx] = i

    if np.any(op_index < 0):
        raise RuntimeError(
            f"Could not find symmetry operations for {np.sum(op_index < 0)} k-points"
        )

    # rotations in the atomic-units structure (same Cartesian frame, scaled)
    reciprocal_lattice = amset_data.structure.lattice.reciprocal_lattice.matrix
    cart = cartesian_rotations(reciprocal_lattice, rotations)
    return cart[op_index]


def get_little_group_projectors(amset_data, symprec=0.01, time_reversal=True, tol=1e-4):
    """Projectors onto vectors invariant under the little group of each ir k-point.

    P_k = (1/|G_k|) Σ_{S ∈ G_k} S, with G_k the operations (including time reversal
    as −R) that leave k_ir invariant modulo a reciprocal lattice vector. The exact
    Λ(k_ir) satisfies P_k Λ = Λ. The numerical in-scattering term does not exactly:
    trilinear interpolation on the (non-orthogonal) k-mesh and the fixed tetrahedron
    decomposition are not covariant under every point-group operation, which gives a
    small spurious anisotropy (≈1e-3 for FCC). Projecting removes it.

    Returns:
        Array of shape (n_ir_kpoints, 3, 3).
    """
    structure = get_angstrom_structure(amset_data.structure)
    rotations, _, _ = get_reciprocal_point_group_operations(
        structure, symprec=symprec, time_reversal=time_reversal
    )
    reciprocal_lattice = amset_data.structure.lattice.reciprocal_lattice.matrix
    cart = cartesian_rotations(reciprocal_lattice, rotations)
    ir_kpoints = amset_data.kpoints[amset_data.ir_kpoints_idx]

    projectors = np.zeros((len(ir_kpoints), 3, 3))
    counts = np.zeros(len(ir_kpoints))
    for r, s in zip(rotations, cart):
        diff = ir_kpoints @ r.T - ir_kpoints
        match = np.all(np.abs(diff - np.round(diff)) < tol, axis=1)
        projectors[match] += s
        counts[match] += 1
    return projectors / counts[:, None, None]


def project_vectors(ir_vectors, projectors):
    """Apply little-group projectors to vectors (nbands, n_ir, 3)."""
    return np.einsum("kij,bkj->bki", projectors, ir_vectors)


def unfold_vectors(ir_vectors, full_rotations, ir_to_full):
    """Unfold vectors (nbands, n_ir, 3) to the full mesh (nbands, nkpoints, 3)."""
    return np.einsum("kij,bkj->bki", full_rotations, ir_vectors[:, ir_to_full])


def check_velocity_unfolding(amset_data, full_rotations):
    """Relative error of S v(k_ir) against the FFT velocities on the full mesh.

    Velocities are computed by BoltzTraP2 directly on the full mesh, so this is an
    independent check of the rotations. Degenerate band crossings can give isolated
    outliers, so both the median and the 99th percentile are returned.
    """
    results = {}
    for spin in amset_data.spins:
        v_full = amset_data.velocities[spin]
        v_ir = v_full[:, amset_data.ir_kpoints_idx]
        v_unfold = unfold_vectors(
            v_ir, full_rotations, amset_data.ir_to_full_kpoint_mapping
        )
        scale = np.max(np.linalg.norm(v_full, axis=-1))
        err = np.linalg.norm(v_unfold - v_full, axis=-1).ravel() / scale
        results[spin] = (np.median(err), np.percentile(err, 99), err.max())
    return results
