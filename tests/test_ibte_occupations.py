"""Unit tests of the IBTE occupation factors (spec rung 5, detailed balance).

A sign or index swap in the in/out occupation table of Claes et al. Eqs. (S12)-(S13)
fails these tests.
"""

import numpy as np
import pytest

from amset.constants import boltzmann_au, ev_to_hartree
from amset.electronic_structure.fd import dfdde, fd
from amset.ibte.solver import dfde_ratio, pop_occupation_factors

rng = np.random.default_rng(1234)


def _random_case():
    temp = rng.uniform(50, 900)
    kbt = temp * boltzmann_au
    omega = rng.uniform(0.005, 0.08) * ev_to_hartree
    mu = rng.uniform(-0.3, 0.3) * ev_to_hartree
    energies = rng.uniform(-0.5, 0.5, size=200) * ev_to_hartree
    n_po = 1 / (np.exp(omega / kbt) - 1)
    return energies, mu, kbt, omega, n_po


@pytest.mark.parametrize("seed", range(5))
def test_zero_mode_identity(seed):
    """in × f⁰'(ε')/f⁰'(ε) == out for each channel.

    This makes every row of the option-A iteration matrix sum to one (number
    conservation), independently of the matrix elements.
    """
    energies, mu, kbt, omega, n_po = _random_case()
    occ = pop_occupation_factors(energies, mu, kbt, omega, n_po)
    for c in ("absorption", "emission"):
        np.testing.assert_allclose(
            occ["in_" + c] * occ["ratio_" + c], occ["out_" + c], rtol=1e-10
        )


@pytest.mark.parametrize("seed", range(5))
def test_microscopic_detailed_balance(seed):
    """S(k→k') f(1 − f') = S(k'→k) f'(1 − f) with bare phonon factors.

    k at ε, k' at ε + ħω: absorption from k uses n, emission from k' uses n + 1.
    """
    energies, mu, kbt, omega, n_po = _random_case()
    f = fd(energies, mu, kbt)
    fp = fd(energies + omega, mu, kbt)
    lhs = n_po * f * (1 - fp)
    rhs = (n_po + 1) * fp * (1 - f)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_out_factors_match_amset_pop_factor():
    """The out factors equal those in PolarOpticalScattering.factor."""
    energies, mu, kbt, omega, n_po = _random_case()
    occ = pop_occupation_factors(energies, mu, kbt, omega, n_po)
    # absorption: final energy ε + ω, factor n + f(final)
    np.testing.assert_allclose(occ["out_absorption"], n_po + fd(energies + omega, mu, kbt))
    # emission: final energy ε − ω, factor n + 1 − f(final)
    np.testing.assert_allclose(
        occ["out_emission"], n_po + 1 - fd(energies - omega, mu, kbt)
    )


def test_dfde_ratio():
    energies, mu, kbt, omega, _ = _random_case()
    direct = dfdde(energies + omega, mu, kbt) / dfdde(energies, mu, kbt)
    np.testing.assert_allclose(dfde_ratio(energies + omega, energies, mu, kbt), direct)
    # stable far from the Fermi level, where the direct ratio is 0/0
    far = np.array([10.0, -10.0]) * ev_to_hartree
    ratio = dfde_ratio(far + omega, far, 0.0, 300 * boltzmann_au)
    assert np.all(np.isfinite(ratio))
    # non-degenerate limit: ratio = exp(∓ω/kT)
    np.testing.assert_allclose(ratio[0], np.exp(-omega / (300 * boltzmann_au)), rtol=1e-8)
    np.testing.assert_allclose(ratio[1], np.exp(omega / (300 * boltzmann_au)), rtol=1e-8)
