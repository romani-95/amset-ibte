"""Orchestration of the IBTE steps inside ``amset.core.run.Runner``."""

import copy
import logging
import time
from pathlib import Path

import numpy as np
from tabulate import tabulate

from amset.constants import bohr_to_cm
from amset.core.transport import _get_band_idx
from amset.ibte.kernel import (
    IBTEKernel,
    TrilinearMapper,
    check_kernel_compatible,
    entry_layout,
    get_kernel_metadata,
    inelastic_channel,
)
from amset.ibte.solver import IBTESolver, MechanismData, reconstruct_rates
from amset.ibte.symmetry import (
    check_velocity_unfolding,
    get_full_kpoint_rotations,
    get_little_group_projectors,
)
from amset.ibte.transport import calculate_mobility_from_lambda, mobility_with_rates
from amset.log import log_list
from amset.util import tensor_average

logger = logging.getLogger(__name__)


def _combine_amset_rates(rates, n_elastic):
    """rta_mode "amset": MRTA rates for elastic, SERTA rates for inelastic."""
    out = {}
    for spin in rates["serta"]:
        r = rates["serta"][spin].copy()
        r[:n_elastic] = rates["mrta"][spin][:n_elastic]
        out[spin] = r
    return out


def select_rta_rates(rates, rta_mode, n_elastic):
    if rta_mode == "serta":
        return {s: r.copy() for s, r in rates["serta"].items()}
    if rta_mode == "mrta":
        return {s: r.copy() for s, r in rates["mrta"].items()}
    return _combine_amset_rates(rates, n_elastic)


def kernel_entry_index(elastic, inelastic, ndops, ntemps):
    """Entry layout of every kernel channel (see ``amset.ibte.kernel.entry_layout``).

    IMP, and POP/PIE with free-carrier screening, get one entry per (doping,
    temperature); all other mechanisms a single entry.
    """
    from amset.scattering.calculate import kernel_entries

    nstates = ndops * ntemps
    index = {}
    if elastic:
        counts = [kernel_entries(s, nstates) for s in elastic]
        index["elastic"] = entry_layout(counts, ndops, ntemps)
    for mode, scatterer in enumerate(inelastic):
        layout = entry_layout([kernel_entries(scatterer, nstates)], ndops, ntemps)
        for kind in ("absorption", "emission"):
            index[inelastic_channel(kind, mode, len(inelastic))] = layout
    return index


def compute_or_load_kernel(amset_data, settings, cutoff_pad):
    """Compute (or load) the kernel and SERTA/MRTA rates; set amset_data.ibte."""
    from amset.scattering.calculate import (
        ScatteringCalculator,
        _interpolate_zero_rates,
    )

    t0 = time.perf_counter()
    metadata = get_kernel_metadata(settings, amset_data)
    scatterers = ScatteringCalculator.get_scatterers(
        settings["scattering_type"], settings, amset_data
    )
    from amset.scattering.elastic import AbstractElasticScattering
    from amset.scattering.inelastic import AbstractInelasticScattering

    # same ordering as ScatteringCalculator.scatterer_labels (elastic, inelastic);
    # mode-resolved POP gives one inelastic scatterer per polar mode
    elastic = [s for s in scatterers if isinstance(s, AbstractElasticScattering)]
    inelastic = [s for s in scatterers if isinstance(s, AbstractInelasticScattering)]
    labels = [s.name for s in elastic] + [s.name for s in inelastic]
    mech = MechanismData.from_scatterers(elastic, inelastic, amset_data, settings, labels)
    entry_index = kernel_entry_index(elastic, inelastic, *amset_data.fermi_levels.shape)
    if any(len(np.unique(v)) > 1 for v in entry_index.values()):
        logger.info(
            "IBTE kernel depends on doping/temperature (IMP or free-carrier "
            "screening): one kernel entry per doping and temperature"
        )

    if settings["ibte_load_kernel"]:
        logger.info(f"Loading IBTE kernel from {settings['ibte_load_kernel']}")
        kernel = IBTEKernel.from_file(settings["ibte_load_kernel"])
        check_kernel_compatible(kernel, metadata)
        if kernel.mechanisms != mech.labels:
            raise ValueError(
                f"Kernel mechanisms {kernel.mechanisms} do not match {mech.labels}"
            )
        for c, layout in entry_index.items():
            stored = kernel.entry_index.get(c)
            if stored is None:
                stored = entry_layout([1] * len(kernel.mechanisms[c]), *layout.shape[1:])
            if not np.array_equal(stored, layout):
                raise ValueError(
                    f"Kernel entry layout of channel {c} does not match this run "
                    "(doping-dependent kernels need the same dopings and temperatures)"
                )
        ir_rates, computed = reconstruct_rates(kernel, mech, amset_data, labels)
        ir_to_full = amset_data.ir_to_full_kpoint_mapping
        ir_idx = amset_data.ir_kpoints_idx
        rates = {}
        for m, mode_rates in ir_rates.items():
            full = {s: r[..., ir_to_full] for s, r in mode_rates.items()}
            masks = {
                s: np.broadcast_to(~computed[s][:, ir_to_full], full[s].shape).copy()
                for s in full
            }
            for s in full:
                full[s][masks[s]] = 1e14
            full = _interpolate_zero_rates(full, amset_data.kpoints, masks)
            rates[m] = {s: r[..., ir_idx][..., ir_to_full] for s, r in full.items()}
        ncomputed = sum(int(c.sum()) for c in computed.values())
        logger.info(f"Rates reconstructed from kernel for {ncomputed} states")
    else:
        mapper = TrilinearMapper.from_kpoints(amset_data.kpoints)
        scatter = ScatteringCalculator(
            settings,
            amset_data,
            cutoff_pad,
            scattering_type=settings["scattering_type"],
            progress_bar=settings["print_log"],
            cache_wavefunction=settings["cache_wavefunction"],
            nworkers=settings["nworkers"],
            rta_mode=settings["rta_mode"],
            kernel_mapper=mapper,
        )
        if scatter.scatterer_labels != labels:
            raise RuntimeError("Unexpected scatterer ordering")
        rates, _, rows = scatter.calculate_scattering_rates_ibte()
        # release the cached wavefunction coefficients (shared memory) before the
        # kernel is assembled and solved
        del scatter
        import gc

        gc.collect()
        nbands = {s: len(amset_data.energies[s]) for s in amset_data.spins}
        logger.info("Assembling sparse IBTE kernel")
        kernel = IBTEKernel.from_rows(
            rows,
            mech.labels,
            nbands,
            len(amset_data.ir_kpoints_idx),
            len(amset_data.kpoints),
            metadata,
            entry_index,
        )
        del rows

    logger.info("IBTE kernel:")
    log_list(kernel.summary() + [f"total: {kernel.nbytes / 1e6:.1f} MB"])

    amset_data.set_scattering_rates(
        select_rta_rates(rates, settings["rta_mode"], len(elastic)), labels
    )
    amset_data.ibte = {
        "kernel": kernel,
        "mech": mech,
        "raw_rates": rates,
        "kernel_time": time.perf_counter() - t0,
    }
    return amset_data


def save_kernel(amset_data, directory, prefix=None):
    filename = Path(directory) / (
        f"{prefix}_ibte_kernel.h5" if prefix else "ibte_kernel.h5"
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    amset_data.ibte["kernel"].to_file(filename)
    logger.info(
        f"IBTE kernel written to {filename} ({time.perf_counter() - t0:.1f} s)"
    )


def _filled(amset_data, rates):
    """Copy of rates filled outside the FD cut-offs like the RTA rates."""
    original = amset_data.scattering_rates
    data_logger = logging.getLogger("amset.core.data")
    level = data_logger.level
    try:
        amset_data.scattering_rates = copy.deepcopy(rates)
        data_logger.setLevel(logging.WARNING)
        amset_data.fill_rates_outside_cutoffs()
        return amset_data.scattering_rates
    finally:
        data_logger.setLevel(level)
        amset_data.scattering_rates = original


def run_ibte(amset_data, settings):
    """Solve the IBTE for every doping and temperature; adds mobility keys.

    Adds ``amset_data.mobility["IBTE"]``, ``["SERTA"]`` and ``["MRTA"]`` (MRTA
    factor on all mechanisms), keeping the existing keys unchanged.
    """
    from amset.log import log_banner

    log_banner("IBTE")
    logger.warning(
        "Conductivity, Seebeck and electronic thermal conductivity are RTA results "
        f"(rta_mode: {settings['rta_mode']}); the IBTE gives the mobility only."
    )
    if settings["separate_mobility"]:
        logger.info("The per-mechanism mobilities above are RTA-only.")

    state = amset_data.ibte
    kernel, mech = state["kernel"], state["mech"]

    t0 = time.perf_counter()
    serta = _filled(amset_data, state["raw_rates"]["serta"])
    mrta = _filled(amset_data, state["raw_rates"]["mrta"])
    mob_serta = mobility_with_rates(amset_data, serta)
    mob_mrta = mobility_with_rates(amset_data, mrta)
    t_rta = time.perf_counter() - t0

    if "rotations" not in state:
        state["rotations"] = get_full_kpoint_rotations(
            amset_data, symprec=settings["symprec"], time_reversal=not settings["soc"]
        )
        checks = check_velocity_unfolding(amset_data, state["rotations"])
        logger.info("Symmetry unfolding check, |S v(k_ir) − v(k)| / max|v|:")
        log_list(
            [
                f"{s.name}: median {c[0]:.2e}, 99th percentile {c[1]:.2e}, "
                f"max {c[2]:.2e}"
                for s, c in checks.items()
            ]
        )
        state["unfold_check"] = {s.name: list(c) for s, c in checks.items()}
        state["projectors"] = get_little_group_projectors(
            amset_data, symprec=settings["symprec"], time_reversal=not settings["soc"]
        )

    # ibte_mrta_mechanisms: these mechanisms enter through their MRTA out-scattering
    # rate only (no in-scattering). For strongly forward-peaked scattering (IMP at
    # low carrier density, screening length ≫ mesh spacing) the kernel cannot
    # resolve q → 0 and the MRTA is the more accurate treatment.
    labels = list(amset_data.scattering_labels)
    mrta_mechs = [m for m in (settings.get("ibte_mrta_mechanisms") or []) if m in labels]
    variants = [("IBTE", serta, ())]
    if mrta_mechs:
        hybrid = {}
        for spin, r in serta.items():
            hybrid[spin] = r.copy()
            for name in mrta_mechs:
                i = labels.index(name)
                hybrid[spin][i] = mrta[spin][i]
        logger.info(
            "IBTE with " + ", ".join(mrta_mechs) + " in the MRTA (no in-scattering); "
            "the full IBTE is reported as IBTE_full"
        )
        variants = [("IBTE", hybrid, tuple(mrta_mechs)), ("IBTE_full", serta, ())]

    def make_solver(exclude):
        return IBTESolver(
            amset_data,
            kernel,
            mech,
            state["rotations"],
            method=settings["ibte_method"],
            niter=settings["ibte_niter"],
            tol=settings["ibte_tol"],
            alpha_mix=settings["ibte_alpha_mix"],
            projectors=state["projectors"],
            exclude=exclude,
        )

    shape = amset_data.fermi_levels.shape
    t_solve = 0.0
    t_transport = 0.0
    all_results = {}
    for key, rates, exclude in variants:
        solver = make_solver(exclude)
        mob_ibte, results, ts, tt = _solve_all(
            amset_data, solver, rates, shape, key
        )
        t_solve += ts
        t_transport += tt
        amset_data.mobility[key] = mob_ibte
        all_results[key] = results
    results = all_results["IBTE"]

    amset_data.mobility["SERTA"] = mob_serta
    amset_data.mobility["MRTA"] = mob_mrta

    timings = {
        "kernel": state.get("kernel_time", 0.0),
        "rta_references": t_rta,
        "solve": t_solve,
        "transport": t_transport,
    }
    amset_data.ibte_results = {
        "method": settings["ibte_method"],
        "rta_mode": settings["rta_mode"],
        "mrta_mechanisms": mrta_mechs,
        "kernel_nnz": {
            f"{s.name}_{c}": int(ch.nnz)
            for s, sc in kernel.channels.items()
            for c, ch in sc.items()
        },
        "kernel_mb": kernel.nbytes / 1e6,
        "timings": timings,
        "unfold_check": state.get("unfold_check"),
        "points": results,
    }
    if "IBTE_full" in all_results:
        amset_data.ibte_results["points_full"] = all_results["IBTE_full"]

    _log_ibte_summary(amset_data, results, timings, kernel)


def _solve_all(amset_data, solver, serta, shape, key):
    """Solve for every doping and temperature; returns (mobility, results, times)."""
    mob_ibte = np.zeros(shape + (3, 3))
    results = []
    t_solve = 0.0
    t_transport = 0.0
    for n, t in np.ndindex(shape):
        doping = amset_data.doping[n] * (1 / bohr_to_cm) ** 3
        temp = amset_data.temperatures[t]
        logger.info(f"Solving IBTE for {doping:.2e} cm⁻³, {temp:.1f} K")
        band_idx = _get_band_idx(
            amset_data.energies, amset_data.vb_idx, amset_data.doping[n]
        )
        ts = time.perf_counter()
        lam, info = solver.solve(n, t, serta, band_idx=band_idx)
        t_solve += time.perf_counter() - ts

        ts = time.perf_counter()
        mob_ibte[n, t] = calculate_mobility_from_lambda(
            amset_data, lam, n, t, band_idx
        )
        t_transport += time.perf_counter() - ts

        flagged = {}
        for spin in amset_data.spins:
            frac, nflag = solver.flagged_weight_fraction(spin, n, t, band_idx)
            flagged[spin.name] = (frac, nflag)
            if frac > 0.01:
                logger.warning(
                    f"{100 * frac:.1f}% of the transport weight is carried by states "
                    "without in-scattering (outside the cut-offs or without final "
                    "states). Consider increasing fd_tol precision / the mesh."
                )
        results.append(
            {
                "doping": float(doping),
                "temperature": float(temp),
                "converged": bool(info["converged"]),
                "residual": float(info["residual"]),
                "niterations": [len(i["history"]) for i in info["iterations"]],
                "history": [i["history"] for i in info["iterations"]],
                "flagged_weight_fraction": {k: v[0] for k, v in flagged.items()},
                "n_flagged_states": {k: v[1] for k, v in flagged.items()},
            }
        )
    return mob_ibte, results, t_solve, t_transport


def _log_ibte_summary(amset_data, results, timings, kernel):
    rows = []
    for (n, t), r in zip(np.ndindex(amset_data.fermi_levels.shape), results):
        ibte = tensor_average(amset_data.mobility["IBTE"][n, t])
        serta = tensor_average(amset_data.mobility["SERTA"][n, t])
        mrta = tensor_average(amset_data.mobility["MRTA"][n, t])
        rta = tensor_average(amset_data.mobility["overall"][n, t])
        frac = max(r["flagged_weight_fraction"].values())
        rows.append(
            (
                r["doping"],
                r["temperature"],
                serta,
                mrta,
                rta,
                ibte,
                serta / ibte,
                mrta / ibte,
                max(r["niterations"]) if r["niterations"] else 0,
                100 * frac,
            )
        )
    logger.info("IBTE mobility results (cm²/Vs):")
    table = tabulate(
        rows,
        headers=(
            "conc [cm⁻³]",
            "temp [K]",
            "μ_SERTA",
            "μ_MRTA",
            "μ_RTA",
            "μ_IBTE",
            "SERTA/IBTE",
            "MRTA/IBTE",
            "# iter",
            "flagged [%]",
        ),
        numalign="right",
        stralign="center",
        floatfmt=(".2e", ".1f", ".2f", ".2f", ".2f", ".2f", ".3f", ".3f", "d", ".2f"),
    )
    logger.info(table)
    logger.info("μ_RTA is the 'overall' mobility (rta_mode) also reported above.")
    log_list(
        [f"{k} time: {v:.2f} s" for k, v in timings.items()]
        + [f"kernel size: {kernel.nbytes / 1e6:.1f} MB"]
    )
