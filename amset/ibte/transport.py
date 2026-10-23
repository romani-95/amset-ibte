"""Mobility from the IBTE solution, using AMSET's transport machinery.

In the RTA, AMSET builds the transport DOS from the integrand v ⊗ v τ. With
Λ = (∂f/∂E)/(∂f⁰/∂ε) the exact linear response gives (Claes et al. Eq. S3)::

    μ_αβ = −1/(Ω n) Σ_n ∫dk/Ω_BZ (∂f_nk/∂E_β) v_nk,α
         =  1/(Ω n) Σ_n ∫dk/Ω_BZ (−∂f⁰/∂ε) v_nk,α Λ_nk,β

so the same Fermi integrals are used with v ⊗ Λ in place of v ⊗ v τ. For
Λ = τ⁰ v this reduces exactly to the SERTA result.
"""

import numpy as np
from BoltzTraP2.bandlib import calc_Onsager_coefficients

from amset.constants import bohr_to_cm, e_si
from amset.interpolation.boltztrap import fermiintegrals


def calculate_mobility_from_lambda(amset_data, lambda_full, n, t, band_idx):
    """Mobility tensor (cm²/Vs) at doping n and temperature t.

    Args:
        amset_data: AmsetData.
        lambda_full: ``{spin: (nbands, nkpoints, 3)}`` on the full mesh.
        n, t: Doping and temperature indices.
        band_idx: ``{spin: band indices}`` (as ``core.transport._get_band_idx``).
    """
    tbs = amset_data.tetrahedral_band_structure
    volume = amset_data.structure.volume
    epsilon, dos = tbs.get_density_of_states(
        amset_data.dos.energies, sum_spins=True, use_cached_weights=True
    )

    integrand = {
        s: amset_data.velocities[s][:, :, :, None] * lambda_full[s][:, :, None, :]
        for s in amset_data.spins
    }
    _, vvdos = tbs.get_density_of_states(
        amset_data.dos.energies,
        integrand=integrand,
        sum_spins=True,
        band_idx=band_idx,
        use_cached_weights=True,
    )
    vvdos = vvdos.transpose(1, 2, 0)

    fermi = amset_data.fermi_levels[n, t][None]
    temp = amset_data.temperatures[t][None]
    _, l0, l1, l2, _ = fermiintegrals(
        epsilon, dos, vvdos, mur=fermi, Tr=temp, dosweight=amset_data.dos.dos_weight
    )
    sigma, _, _, _ = calc_Onsager_coefficients(l0, l1, l2, fermi, temp, volume)

    if amset_data.doping[n] < 0:
        carrier_conc = amset_data.electron_conc[n, t]
    else:
        carrier_conc = amset_data.hole_conc[n, t]

    uc = 0.01 / (e_si * carrier_conc * (1 / bohr_to_cm) ** 3)
    return sigma[0, 0] * uc


def mobility_with_rates(amset_data, rates):
    """Overall RTA mobility for a given set of (filled) rates."""
    from amset.core.transport import _calculate_mobility

    original = amset_data.scattering_rates
    try:
        amset_data.scattering_rates = rates
        nscat = len(amset_data.scattering_labels)
        return _calculate_mobility(amset_data, np.arange(nscat), pbar_label=None)
    finally:
        amset_data.scattering_rates = original
