"""Iterative solution of the linearized BTE (Claes et al., PRB 106, 094302 (2022)).

Unknown (option A of the implementation spec)::

    Λ_nk ≡ (∂f_nk/∂E) / (∂f⁰_nk/∂ε)

Dividing Eq. (S12) by ∂f⁰_nk/∂ε gives::

    Λ_nk = τ⁰_nk v_nk + τ⁰_nk Σ_k' K_in(k, k') [f⁰'(ε') / f⁰'(ε_nk)] Λ_k'      (A)

The in-scattering occupation factors of Eq. (S12) and the out-scattering ones of
Eq. (S13) (used by AMSET for τ⁰) are, with n the Bose-Einstein occupation of the
polar phonon and f evaluated at the energy indicated::

                        from ε' = ε + ħω        from ε' = ε − ħω
    out (S13)           n + f(ε + ħω)           n + 1 − f(ε − ħω)
    in  (S12)           n + 1 − f(ε)            n + f(ε)

For elastic mechanisms there are no occupation factors and the ratio of Fermi
derivatives is 1. With these factors every row of τ⁰ K_in f⁰'(ε')/f⁰'(ε) sums
exactly to one (the number-conservation zero mode of the collision operator), which
is used as a consistency test.

Option B iterates directly on g = ∂f/∂E (Eq. S12 without dividing); it is only
used as an independent check.
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pymatgen.electronic_structure.core import Spin

from amset.constants import boltzmann_au
from amset.electronic_structure.fd import dfdde, fd
from amset.ibte.kernel import IBTEKernel, channel_kind, inelastic_channel
from amset.ibte.symmetry import project_vectors, unfold_vectors

logger = logging.getLogger(__name__)


def _log_cosh(x):
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2 * ax)) - np.log(2)


def dfde_ratio(e_prime, e, mu, kbt):
    """f⁰'(ε') / f⁰'(ε), computed stably (f⁰' ∝ 1 / cosh²((ε − μ) / 2kT))."""
    xa = 0.5 * (e - mu) / kbt
    xb = 0.5 * (e_prime - mu) / kbt
    return np.exp(2 * (_log_cosh(xa) - _log_cosh(xb)))


def pop_occupation_factors(energies, mu, kbt, omega, n_po):
    """Occupation factors of the table in the module docstring.

    Args:
        energies: Initial-state energies ε (Hartree).
        mu: Fermi level (Hartree).
        kbt: k_B T (Hartree).
        omega: ħω_po (Hartree).
        n_po: Bose-Einstein occupation of the polar phonon.

    Returns:
        Dict with keys ``out_absorption``, ``out_emission``, ``in_absorption``,
        ``in_emission``, ``ratio_absorption`` and ``ratio_emission``.
    """
    f = fd(energies, mu, kbt)
    return {
        "out_absorption": n_po + fd(energies + omega, mu, kbt),
        "out_emission": n_po + 1 - fd(energies - omega, mu, kbt),
        "in_absorption": n_po + 1 - f,
        "in_emission": n_po + f,
        "ratio_absorption": dfde_ratio(energies + omega, energies, mu, kbt),
        "ratio_emission": dfde_ratio(energies - omega, energies, mu, kbt),
    }


class MechanismData:
    """Prefactors and POP parameters needed to turn the kernel into rates.

    Channels are "elastic" plus, for each polar mode, an absorption and an
    emission channel (``amset.ibte.kernel.inelastic_channel``); each mode has its
    own ħω and Bose-Einstein occupation.

    Args:
        prefactors: ``{channel: {spin: array (nmech, ndops, ntemps, nbands)}}``,
            including the ``Ω_BZ / 4π²`` conversion factor of ScatteringCalculator.
        omega: ``{channel: ħω in Hartree}`` for the inelastic channels.
        n_po: ``{channel: phonon occupation per temperature (ntemps,)}``.
        labels: ``{channel: [names]}`` in the order of the kernel mechanisms.
        rate_index: ``{channel: [index in amset_data.scattering_labels]}``.
    """

    def __init__(self, prefactors, omega, n_po, labels, rate_index):
        self.prefactors = prefactors
        self.omega = omega
        self.n_po = n_po
        self.labels = labels
        self.rate_index = rate_index

    @classmethod
    def from_scatterers(cls, elastic, inelastic, amset_data, settings, labels):
        from amset.scattering.calculate import _pop_energy_diff

        vol = amset_data.structure.lattice.reciprocal_lattice.volume
        conversion = vol / (4 * np.pi**2)

        def prefactor(mechanisms, spin):
            nb = len(amset_data.energies[spin])
            return conversion * np.array(
                [
                    np.stack([m.prefactor(spin, b) for b in range(nb)], axis=-1)
                    for m in mechanisms
                ]
            )

        prefactors, omega, n_po, mech_labels = {}, {}, {}, {}
        if elastic:
            mech_labels["elastic"] = [m.name for m in elastic]
            prefactors["elastic"] = {s: prefactor(elastic, s) for s in amset_data.spins}
        for mode, scatterer in enumerate(inelastic):
            for kind in ("absorption", "emission"):
                c = inelastic_channel(kind, mode, len(inelastic))
                mech_labels[c] = [scatterer.name]
                prefactors[c] = {
                    s: prefactor([scatterer], s) for s in amset_data.spins
                }
                omega[c] = _pop_energy_diff(scatterer, settings)
                n_po[c] = np.asarray(scatterer.n_po).reshape(-1)

        rate_index = {
            c: [labels.index(name) for name in names]
            for c, names in mech_labels.items()
        }
        return cls(prefactors, omega, n_po, mech_labels, rate_index)


def reconstruct_rates(kernel: IBTEKernel, mech: MechanismData, amset_data, labels):
    """Rebuild SERTA and MRTA rates from a stored kernel.

    Used when the kernel is loaded from file. The out-scattering rate is the row
    sum of the geometric kernel times the prefactor and, for POP, the Eq. (S13)
    occupation factor evaluated at the initial-state energy, as in AMSET.

    Returns:
        ``(rates, computed)`` where rates is ``{"serta": {spin: arr},
        "mrta": {spin: arr}}`` with arrays (nscat, ndops, ntemps, nbands, n_ir) on
        the irreducible k-points, and computed the per-spin (nbands, n_ir) mask of
        rows that were calculated.
    """
    n_ir = kernel.n_ir_kpoints
    ndop, ntemp = amset_data.fermi_levels.shape
    rates = {"serta": {}, "mrta": {}}
    computed = {}
    for spin in amset_data.spins:
        nb = kernel.nbands[spin]
        ir_e = amset_data.energies[spin][:, amset_data.ir_kpoints_idx]
        shape = (len(labels), ndop, ntemp, nb, n_ir)
        r_serta = np.zeros(shape)
        r_mrta = np.zeros(shape)
        for c, ch in kernel.channels[spin].items():
            # one row per kernel entry (doping-dependent mechanisms have several)
            rowsum = ch.rowsum.reshape(ch.nmech, nb, n_ir)
            rowsum_mrta = ch.rowsum_mrta.reshape(ch.nmech, nb, n_ir)
            pref = mech.prefactors[c][spin]  # (nmech, ndop, ntemp, nb)
            for im, s_idx in enumerate(mech.rate_index[c]):
                for n, t in np.ndindex(ndop, ntemp):
                    e = kernel.entries(c, n, t)[im]
                    occ = 1.0
                    if c != "elastic":
                        kbt = amset_data.temperatures[t] * boltzmann_au
                        occ = pop_occupation_factors(
                            ir_e,
                            amset_data.fermi_levels[n, t],
                            kbt,
                            mech.omega[c],
                            mech.n_po[c][t],
                        )["out_" + channel_kind(c)]
                    p = pref[im, n, t][:, None] * occ
                    r_serta[s_idx, n, t] += p * rowsum[e]
                    r_mrta[s_idx, n, t] += p * rowsum_mrta[e]
        rates["serta"][spin] = r_serta
        rates["mrta"][spin] = r_mrta
        computed[spin] = kernel.computed(spin).reshape(nb, n_ir)
    return rates, computed


class IBTESolver:
    """Solve Eq. (A) for each doping and temperature.

    Args:
        amset_data: AmsetData (energies, velocities, mesh, Fermi levels).
        kernel: The in-scattering kernel.
        mech: Prefactors and POP parameters.
        full_rotations: Cartesian rotations for every full-mesh k-point (nk, 3, 3).
        method: "iterate" (linear mixing) or "gmres".
        niter: Maximum number of iterations.
        tol: Relative tolerance (weighted norm of the change in Λ, see
            :meth:`_weighted_norm`, and relative change of the mobility).
        alpha_mix: Linear mixing parameter.
    """

    def __init__(
        self,
        amset_data,
        kernel: IBTEKernel,
        mech: MechanismData,
        full_rotations: np.ndarray,
        method: str = "iterate",
        niter: int = 100,
        tol: float = 1e-4,
        alpha_mix: float = 0.7,
        projectors: Optional[np.ndarray] = None,
        exclude=(),
    ):
        # mechanisms without in-scattering (treated in the MRTA through their
        # out-scattering rate, see ibte_mrta_mechanisms)
        self.exclude = set(exclude)
        self.amset_data = amset_data
        self.kernel = kernel
        self.mech = mech
        self.full_rotations = full_rotations
        self.projectors = projectors
        self.method = method
        self.niter = niter
        self.tol = tol
        self.alpha_mix = alpha_mix
        self.ir_idx = amset_data.ir_kpoints_idx
        self.ir_to_full = amset_data.ir_to_full_kpoint_mapping
        self.n_ir = len(self.ir_idx)
        self.multiplicity = np.bincount(self.ir_to_full, minlength=self.n_ir)

    def active_rows(self, spin):
        """Rows taking part in the in-scattering: computed, with entries, and
        inside the Fermi–Dirac cut-offs (outside them τ⁰ is a filled value)."""
        nb = self.kernel.nbands[spin]
        ir_e = self.amset_data.energies[spin][:, self.ir_idx]
        inside = np.ones_like(ir_e, dtype=bool)
        if self.amset_data.fd_cutoffs is not None:
            emin, emax = self.amset_data.fd_cutoffs
            inside = (ir_e >= emin) & (ir_e <= emax)
        has_row = self.kernel.has_row(spin).reshape(nb, self.n_ir)
        return has_row & inside

    def iteration_matrix(self, spin, n, t, tau, option="A", active=None):
        """Build T with Λ_out = τ⁰ v + T Λ_full (option A) or the option B form.

        Args:
            spin: Spin channel.
            n, t: Doping and temperature indices.
            tau: τ⁰ on the irreducible states (nbands, n_ir).
            option: "A" (includes the Fermi derivative ratio) or "B".
            active: Mask of active rows (nbands, n_ir); inactive rows are zero.
        """
        nb = self.kernel.nbands[spin]
        ir_e = self.amset_data.energies[spin][:, self.ir_idx]
        mu = self.amset_data.fermi_levels[n, t]
        kbt = self.amset_data.temperatures[t] * boltzmann_au
        if active is None:
            active = self.active_rows(spin)
        row_tau = (tau * active).reshape(-1)
        band_of_row = np.repeat(np.arange(nb), self.n_ir)

        matrix = None
        for c, ch in self.kernel.channels[spin].items():
            keep = np.array([name not in self.exclude for name in self.mech.labels[c]])
            if not keep.any():
                continue
            pref = self.mech.prefactors[c][spin][:, n, t]  # (nmech, nb)
            coef = pref[:, band_of_row] * row_tau[None]
            coef[~keep] = 0
            if c != "elastic":
                # each polar mode has its own ħω and phonon occupation
                kind = channel_kind(c)
                occ = pop_occupation_factors(
                    ir_e.reshape(-1), mu, kbt, self.mech.omega[c], self.mech.n_po[c][t]
                )
                coef = coef * occ["in_" + kind][None]
                if option == "A":
                    coef = coef * occ["ratio_" + kind][None]
            m = ch.matrix(coef, self.kernel.entries(c, n, t))
            matrix = m if matrix is None else matrix + m
        if matrix is None:  # every mechanism excluded from the in-scattering
            import scipy.sparse as sp

            any_ch = next(iter(self.kernel.channels[spin].values()))
            return sp.csr_matrix(any_ch.shape)
        return matrix.tocsr()

    def _weights(self, spin, n, t):
        ir_e = self.amset_data.energies[spin][:, self.ir_idx]
        mu = self.amset_data.fermi_levels[n, t]
        kbt = self.amset_data.temperatures[t] * boltzmann_au
        return -dfdde(ir_e, mu, kbt) * self.multiplicity[None, :]

    def _weighted_norm(self, x, w):
        """sqrt(Σ_nk w_nk |x_nk|²) with w = −f⁰'(ε) × k-point multiplicity."""
        return np.sqrt(np.sum(w[..., None] * x**2))

    def unfold(self, ir_vectors):
        return unfold_vectors(ir_vectors, self.full_rotations, self.ir_to_full)

    def inscattering(self, matrix, ir_vectors):
        """T Λ for Λ given on the ir states; result projected on the little group."""
        nb = ir_vectors.shape[0]
        out = (matrix @ self.unfold(ir_vectors).reshape(-1, 3)).reshape(
            nb, self.n_ir, 3
        )
        if self.projectors is not None:
            out = project_vectors(out, self.projectors)
        return out

    def solve(self, n, t, serta_rates, band_idx=None, inscattering=True):
        """Solve for Λ at doping n and temperature t.

        Args:
            n, t: Doping and temperature indices.
            serta_rates: SERTA rates on the full mesh, ``{spin: (nscat, ndop, ntemp,
                nbands, nkpoints)}``, already filled outside the FD cut-offs.
            band_idx: ``{spin: band indices}`` used for the mobility estimate in the
                iteration log (conduction bands for n-type, valence for p-type).
            inscattering: If False, return Λ = τ⁰ v (iteration 0, pure SERTA).

        Returns:
            ``(lambda_full, info)`` with lambda_full ``{spin: (nbands, nk, 3)}``.
        """
        lambda_full = {}
        info = {"iterations": [], "converged": True, "residual": 0.0}
        for spin in self.amset_data.spins:
            tau_full = 1 / np.sum(serta_rates[spin][:, n, t], axis=0)
            v_full = self.amset_data.velocities[spin]
            if not inscattering:
                lambda_full[spin] = tau_full[..., None] * v_full
                continue

            nb = self.kernel.nbands[spin]
            tau = tau_full[:, self.ir_idx]
            v_ir = v_full[:, self.ir_idx]
            lambda0 = tau[..., None] * v_ir
            active = self.active_rows(spin)
            matrix = self.iteration_matrix(spin, n, t, tau, option="A", active=active)
            weights = self._weights(spin, n, t)
            mob_mask = np.zeros(nb, dtype=bool)
            mob_mask[band_idx[spin] if band_idx is not None else slice(None)] = True
            w_mob = weights * mob_mask[:, None]

            def apply(lam):
                return self.inscattering(matrix, lam)

            def mobility_proxy(lam):
                # trace of Σ (−f⁰') v ⊗ Λ, proportional to the average mobility
                return np.sum(w_mob * np.einsum("bki,bki->bk", v_ir, lam))

            mu0 = mobility_proxy(lambda0)
            t0 = time.perf_counter()
            if self.method == "iterate":
                lam, spin_info = self._iterate(
                    apply, lambda0, weights, mobility_proxy, mu0
                )
            elif self.method == "gmres":
                lam, spin_info = self._gmres(apply, lambda0, weights, mobility_proxy, mu0)
            else:
                raise ValueError(f"Unknown ibte_method: {self.method}")
            spin_info["time"] = time.perf_counter() - t0
            spin_info["spin"] = spin.name
            spin_info["n_active_rows"] = int(active.sum())
            spin_info["matrix_nnz"] = int(matrix.nnz)
            info["iterations"].append(spin_info)
            info["converged"] &= spin_info["converged"]
            info["residual"] = max(info["residual"], spin_info["residual"])

            delta = self.unfold(lam - lambda0)
            lambda_full[spin] = tau_full[..., None] * v_full + delta
        return lambda_full, info

    def _iterate(self, apply, lambda0, weights, mobility_proxy, mu0):
        """Linear-mixing fixed-point iteration."""
        lam = lambda0.copy()
        history = []
        converged = False
        resid = np.inf
        mob_old = mu0
        for it in range(1, self.niter + 1):
            lam_out = lambda0 + apply(lam)
            resid = self._weighted_norm(lam_out - lam, weights) / self._weighted_norm(
                lam_out, weights
            )
            mob = mobility_proxy(lam_out)
            dmob = abs(mob - mob_old) / abs(mob)
            history.append((it, mob / mu0, resid))
            logger.info(
                f"    iter {it:3d}: μ/μ_SERTA = {mob / mu0:.6f}, "
                f"residual = {resid:.3e}, Δμ/μ = {dmob:.3e}"
            )
            mob_old = mob
            if resid < self.tol and dmob < self.tol:
                lam = lam_out
                converged = True
                break
            lam = (1 - self.alpha_mix) * lam + self.alpha_mix * lam_out

        if not converged:
            logger.warning(
                f"IBTE did not converge in {self.niter} iterations "
                f"(residual {resid:.2e}). Increase ibte_niter or reduce ibte_alpha_mix."
            )
        return lam, {"history": history, "converged": converged, "residual": resid}

    def _gmres(self, apply, lambda0, weights, mobility_proxy, mu0):
        """Solve (1 − T) Λ = τ⁰ v with GMRES."""
        shape = lambda0.shape
        size = lambda0.size

        def matvec(x):
            lam = x.reshape(shape)
            return (lam - apply(lam)).ravel()

        op = spla.LinearOperator((size, size), matvec=matvec, dtype=np.float64)
        history = []

        def callback(pr_norm):
            history.append((len(history) + 1, np.nan, float(pr_norm)))

        b = lambda0.ravel()
        x, status = spla.gmres(
            op,
            b,
            x0=b.copy(),
            rtol=self.tol * 1e-2,
            atol=0.0,
            restart=min(50, size),
            maxiter=self.niter,
            callback=callback,
            callback_type="pr_norm",
        )
        lam = x.reshape(shape)
        resid_vec = (lambda0 + apply(lam)) - lam
        resid = self._weighted_norm(resid_vec, weights) / self._weighted_norm(
            lam, weights
        )
        mob = mobility_proxy(lam)
        logger.info(
            f"    GMRES: {len(history)} inner iterations, status = {status}, "
            f"μ/μ_SERTA = {mob / mu0:.6f}, residual = {resid:.3e}"
        )
        converged = status == 0 and resid < self.tol
        if not converged:
            logger.warning(f"GMRES did not fully converge (status {status}).")
        return lam, {"history": history, "converged": converged, "residual": resid}

    def solve_option_b(self, n, t, serta_rates, band_idx=None, niter=None):
        """Independent check: iterate on g = ∂f/∂E directly (Eq. S12).

        Returns the mobility proxy ratio μ_IBTE / μ_SERTA for each spin, computed
        as Σ (−v · g) over the irreducible states (weights: k-point multiplicity),
        which equals the option A proxy Σ (−f⁰') v · Λ for the same solution.
        """
        niter = niter or self.niter
        ratios = {}
        for spin in self.amset_data.spins:
            nb = self.kernel.nbands[spin]
            tau = (1 / np.sum(serta_rates[spin][:, n, t], axis=0))[:, self.ir_idx]
            v_ir = self.amset_data.velocities[spin][:, self.ir_idx]
            ir_e = self.amset_data.energies[spin][:, self.ir_idx]
            mu = self.amset_data.fermi_levels[n, t]
            kbt = self.amset_data.temperatures[t] * boltzmann_au
            dfde = dfdde(ir_e, mu, kbt)
            active = self.active_rows(spin)
            matrix = self.iteration_matrix(spin, n, t, tau, option="B", active=active)
            mob_mask = np.zeros(nb, dtype=bool)
            mob_mask[band_idx[spin] if band_idx is not None else slice(None)] = True
            w = self.multiplicity[None, :] * mob_mask[:, None]

            g0 = (dfde * tau)[..., None] * v_ir
            g = g0.copy()

            def proxy(x):
                return -np.sum(w * np.einsum("bki,bki->bk", v_ir, x))

            # converge on the mobility proxy (g spans many orders of magnitude)
            mob_old = proxy(g0)
            for _ in range(niter):
                g_out = g0 + self.inscattering(matrix, g)
                mob = proxy(g_out)
                g = (1 - self.alpha_mix) * g + self.alpha_mix * g_out
                if abs(mob - mob_old) < self.tol * 1e-2 * abs(mob):
                    g = g_out
                    break
                mob_old = mob

            ratios[spin] = proxy(g) / proxy(g0)
        return ratios

    def flagged_weight_fraction(self, spin, n, t, band_idx=None):
        """Fraction of the transport weight (−f⁰' |v|²) carried by inactive states."""
        nb = self.kernel.nbands[spin]
        v_ir = self.amset_data.velocities[spin][:, self.ir_idx]
        w = self._weights(spin, n, t) * np.sum(v_ir**2, axis=-1)
        if band_idx is not None:
            mask = np.zeros(nb, dtype=bool)
            mask[band_idx[spin]] = True
            w = w * mask[:, None]
        active = self.active_rows(spin)
        total = w.sum()
        if total == 0:
            return 0.0, int((~active).sum())
        return float(w[~active].sum() / total), int((~active).sum())
