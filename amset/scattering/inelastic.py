import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from pymatgen.electronic_structure.core import Spin

from amset.constants import boltzmann_au, ev_to_hartree, hbar, s_to_au
from amset.core.data import AmsetData
from amset.log import log_list
from amset.scattering.common import calculate_inverse_screening_length_sq

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class AbstractInelasticScattering(ABC):
    name: str
    required_properties: Tuple[str]

    def __init__(self, properties, doping, temperatures, nbands):
        self.properties = properties
        self.doping = doping
        self.temperatures = temperatures
        self.nbands = nbands
        self.spins = list(nbands.keys())

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
        )

    @abstractmethod
    def prefactor(self, spin: Spin, b_idx: int):
        pass

    @abstractmethod
    def factor(self, unit_q, norm_q_sq, emission, f):
        pass

    @classmethod
    def get_properties(cls, materials_properties):
        return {p: materials_properties[p] for p in cls.required_properties}

    @staticmethod
    def get_nbands(amset_data):
        return {s: len(amset_data.energies[s]) for s in amset_data.spins}


class PolarOpticalScattering(AbstractInelasticScattering):
    name = "POP"
    required_properties = (
        "pop_frequency",
        "static_dielectric",
        "high_frequency_dielectric",
        "free_carrier_screening",
    )

    def __init__(
        self,
        properties,
        doping,
        temperatures,
        nbands,
        pop_frequency,
        n_po,
        inverse_screening_length_sq,
        weight=1.0,
        name=None,
        energy_diff=None,
        weight_expansion=None,
    ):
        """
        Args:
            pop_frequency: Angular frequency of the polar mode in atomic units.
            weight: Share of the Fröhlich coupling 4π(1/ε∞ − 1/ε₀) carried by this
                mode (mode-resolved POP, see :meth:`from_amset_data`). 1 for the
                usual single effective mode.
            name: Mechanism label ("POP" for a single mode, "POP1", "POP2", ... for
                mode-resolved POP).
            energy_diff: ħω in Hartree, computed exactly as the energy shift used for
                the final-state surfaces in ScatteringCalculator.
            weight_expansion: Optional ``(exponents, coefficients)`` giving a
                direction-dependent share w(q̂) = Σ_k c_k q_x^a q_y^b q_z^c of the
                coupling (``pop_modes_file``); replaces the constant ``weight``.
        """
        super().__init__(properties, doping, temperatures, nbands)
        self.pop_frequency = pop_frequency
        self.n_po = n_po
        self.inverse_screening_length_sq = inverse_screening_length_sq
        self._prefactor = s_to_au * pop_frequency / 2
        self.weight = weight
        if name is not None:
            self.name = name
        self.energy_diff = energy_diff
        self.weight_expansion = None
        if weight_expansion is not None:
            exponents, coefficients = weight_expansion
            self.weight_expansion = (
                np.asarray(exponents, dtype=int),
                np.asarray(coefficients, dtype=float),
            )

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        """Build the POP scatterer(s).

        With a scalar ``pop_frequency`` a single effective mode is used (upstream
        behaviour). With a list, one scatterer per polar LO mode ν is returned, each
        with its own frequency ω_ν and a share w_ν (``pop_weights``, Σ w_ν = 1) of the
        Fröhlich coupling (generalized Fröhlich model, e.g. Miglio et al., npj
        Comput. Mater. 6, 167 (2020))::

            |g_ν(q)|² ∝ (ω_ν / 2) · w_ν · 4π (1/ε∞(q̂) − 1/ε₀(q̂)) / q²

        Each mode has its own Bose-Einstein occupation and its own final-state
        surfaces ε ± ħω_ν.
        """
        frequencies = materials_properties["pop_frequency"]
        if np.ndim(frequencies) == 0:
            return cls._from_mode(materials_properties, amset_data, frequencies)

        weights = materials_properties.get("pop_weights")
        coefficients = materials_properties.get("pop_weight_coefficients")
        exponents = materials_properties.get("pop_weight_exponents")
        logger.info(f"Mode-resolved POP scattering with {len(frequencies)} modes")
        if coefficients is not None:
            logger.info("Using direction-dependent mode weights (pop_modes_file)")
        return [
            cls._from_mode(
                materials_properties,
                amset_data,
                f,
                weight=w,
                name=f"POP{i + 1}",
                weight_expansion=(
                    None if coefficients is None else (exponents, coefficients[i])
                ),
            )
            for i, (f, w) in enumerate(zip(frequencies, weights))
        ]

    @classmethod
    def _from_mode(
        cls,
        materials_properties,
        amset_data,
        frequency_thz,
        weight=1.0,
        name=None,
        weight_expansion=None,
    ):
        logger.info(f"Initializing {name or 'POP'} scattering")

        # convert from THz to angular frequency in Hz
        pop_frequency = frequency_thz * 1e12 * 2 * np.pi / s_to_au
        energy_diff = frequency_thz * 1e12 * 2 * np.pi * hbar * ev_to_hartree

        if materials_properties["free_carrier_screening"]:
            # use high-frequency diel for screening length
            avg_diel = np.linalg.eigvalsh(
                materials_properties["high_frequency_dielectric"]
            ).mean()
            inverse_screening_length_sq = calculate_inverse_screening_length_sq(
                amset_data, avg_diel
            )
        else:
            inverse_screening_length_sq = np.zeros_like(amset_data.fermi_levels)

        # n_po (phonon concentration) has shape (ntemps, )
        n_po = 1 / (
            np.exp(pop_frequency / (boltzmann_au * amset_data.temperatures)) - 1
        )

        n_po = n_po[None, :]

        info = [
            f"average N_po: {np.mean(n_po):.4f}",
            "ω_po: {:.4g} 2π THz".format(frequency_thz * 2 * np.pi),
            f"ħω: {pop_frequency * hbar * s_to_au:.4f} eV",
        ]
        if name is not None:
            info.append(f"coupling weight: {weight:.4f}")
        log_list(info)

        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
            pop_frequency,
            n_po,
            inverse_screening_length_sq,
            weight=weight,
            name=name,
            energy_diff=energy_diff,
            weight_expansion=weight_expansion,
        )

    def mode_weight(self, unit_q):
        """Share of the Fröhlich coupling carried by this mode along each q̂."""
        if self.weight_expansion is None:
            return self.weight
        exponents, coefficients = self.weight_expansion
        mono = np.prod(unit_q[:, None, :] ** exponents[None, :, :], axis=2)
        return np.clip(mono @ coefficients, 0, None)

    def prefactor(self, spin: Spin, b_idx: int):
        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        return self._prefactor * np.ones((len(self.doping), len(self.temperatures)))

    def factor(self, unit_q: np.ndarray, norm_q_sq: np.ndarray, emission, f):
        # presuming that this is out scattering
        if emission:
            factor = self.n_po + 1 - f
        else:
            factor = self.n_po + f

        high_t = self.properties["high_frequency_dielectric"]
        static_t = self.properties["static_dielectric"]
        high_freq_diel = np.einsum("ij,ij->i", unit_q, np.dot(high_t, unit_q.T).T)
        static_diel = np.einsum("ij,ij->i", unit_q, np.dot(static_t, unit_q.T).T)

        dielectric_term = 4 * np.pi * (1 / high_freq_diel - 1 / static_diel)
        if self.weight_expansion is not None or self.weight != 1.0:
            dielectric_term = dielectric_term * self.mode_weight(unit_q)

        return (
            factor[..., None]
            * dielectric_term[None, None]
            / (norm_q_sq[None, None] + self.inverse_screening_length_sq[..., None])
        )

    def geometric_factor(self, unit_q: np.ndarray, norm_q_sq: np.ndarray):
        """The q-dependent part of |g|² without the occupation factor.

        Used by the iterative BTE solver, which applies the in- and out-scattering
        occupation factors itself (Claes et al. PRB 106, 094302, Eqs. S12-S13).

        Returns:
            Shape (ndops, ntemps, nq): with free-carrier screening the screening
            length depends on doping and temperature; without it all entries are
            equal.
        """
        high_t = self.properties["high_frequency_dielectric"]
        static_t = self.properties["static_dielectric"]
        high_freq_diel = np.einsum("ij,ij->i", unit_q, np.dot(high_t, unit_q.T).T)
        static_diel = np.einsum("ij,ij->i", unit_q, np.dot(static_t, unit_q.T).T)
        dielectric_term = 4 * np.pi * (1 / high_freq_diel - 1 / static_diel)
        if self.weight_expansion is not None or self.weight != 1.0:
            dielectric_term = dielectric_term * self.mode_weight(unit_q)
        return dielectric_term[None, None] / (
            norm_q_sq[None, None] + self.inverse_screening_length_sq[..., None]
        )

        # # factor should have shape (ndops, ntemps, nkpts)
        # factor = 1 / np.tile(k_diff_sq, (len(self.doping),
        #                                  len(self.temperatures), 1))
        # if emission:
        #     if out:
        #         return factor * self.emission_f_out[spin][:, :, b_idx, k_idx]
        #     else:
        #         return factor * self.emission_f_in[spin][:, :, b_idx, k_idx]
        # else:
        #     if out:
        #         return factor * self.absorption_f_out[spin][:, :, b_idx, k_idx]
        #     else:
        #         return factor * self.absorption_f_in[spin][:, :, b_idx, k_idx]
