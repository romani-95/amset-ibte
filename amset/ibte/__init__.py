"""Iterative (exact) solution of the linearized Boltzmann transport equation.

The implementation follows the Supplementary Material of Claes et al., Phys. Rev. B
106, 094302 (2022), Eqs. (S3), (S12) and (S13). See ``docs/ibte.md`` for the theory,
the settings and the limitations.

Modules:
    kernel: capture of the in-scattering kernel, trilinear mapping of the off-grid
        final states onto the full mesh, sparse storage and HDF5 I/O.
    symmetry: unfolding of vector quantities from the irreducible to the full mesh.
    solver: occupation factors, iterative and GMRES solvers.
    transport: mobility from the converged solution using AMSET's transport DOS.
    settings: validation of IBTE related settings.
"""
