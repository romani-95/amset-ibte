"""Shared helpers for the IBTE tests."""

import os
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.core import Spin

from amset.ibte.solver import IBTESolver

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples"

slow = pytest.mark.skipif(
    os.environ.get("AMSET_RUN_SLOW", "0") != "1",
    reason="slow test; set AMSET_RUN_SLOW=1 to run",
)


def cosine_band_structure(nk=12, a=4.0, t_c=0.6, t_v=0.8, gap=1.0):
    """Two tight-binding bands on a simple cubic lattice.

    E_c(k) = gap + 2 t_c (3 − Σ_i cos 2πk_i),  E_v(k) = −2 t_v (3 − Σ_i cos 2πk_i)

    Both bands are exactly representable by the lowest BoltzTraP2 star functions, so
    the interpolation is exact; band extrema are at Γ (parabolic near Γ).
    """
    structure = Structure(Lattice.cubic(a), ["Po"], [[0, 0, 0]])
    grid = np.arange(nk) / nk
    grid[grid >= 0.5] -= 1
    kpts = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    s = 3 - np.cos(2 * np.pi * kpts).sum(axis=1)
    bands = np.array([-2 * t_v * s, gap + 2 * t_c * s])
    bs = BandStructure(
        kpts,
        {Spin.up: bands},
        structure.lattice.reciprocal_lattice,
        efermi=gap / 2,
        structure=structure,
    )
    return bs


def solver_from_amset_data(amset_data, settings):
    state = amset_data.ibte
    return IBTESolver(
        amset_data,
        state["kernel"],
        state["mech"],
        state["rotations"],
        method=settings.get("ibte_method", "iterate"),
        niter=settings.get("ibte_niter", 100),
        tol=settings.get("ibte_tol", 1e-4),
        alpha_mix=settings.get("ibte_alpha_mix", 0.7),
        projectors=state["projectors"],
    )


def avg(tensor):
    return np.trace(tensor) / 3
