"""Validation of the solver / IBTE settings."""

import logging

logger = logging.getLogger(__name__)

SOLVERS = ("rta", "ibte")
RTA_MODES = ("amset", "serta", "mrta")
IBTE_METHODS = ("iterate", "gmres")

# Mechanisms with a kernel. IMP, and POP/PIE with free-carrier screening, depend on
# doping and temperature through the screening length: their kernel has one entry
# per (doping, temperature). Basic mechanisms (CRT, SRT, MFP) have no kernel.
IBTE_SUPPORTED_MECHANISMS = ("ADP", "PIE", "POP", "IMP")


def resolve_scattering_type(settings):
    """Resolve ``scattering_type: auto`` exactly as ScatteringCalculator does."""
    from amset.scattering.calculate import _scattering_mechanisms

    if settings["scattering_type"] != "auto":
        return list(settings["scattering_type"])
    return [
        name
        for name, mechanism in _scattering_mechanisms.items()
        if all(settings.get(x, None) is not None for x in mechanism.required_properties)
    ]


def validate_ibte_settings(settings):
    """Check the solver settings; modifies and returns ``settings``.

    Raises:
        ValueError: For unsupported combinations in IBTE v1.
    """
    if settings["solver"] not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {settings['solver']}")
    if settings["rta_mode"] not in RTA_MODES:
        raise ValueError(
            f"rta_mode must be one of {RTA_MODES}, got {settings['rta_mode']}"
        )

    if settings["solver"] != "ibte":
        if settings.get("ibte_load_kernel") or settings.get("ibte_save_kernel"):
            logger.warning("ibte_save_kernel/ibte_load_kernel ignored with solver: rta")
        return settings

    if settings["ibte_method"] not in IBTE_METHODS:
        raise ValueError(f"ibte_method must be one of {IBTE_METHODS}")
    if not 0 < settings["ibte_alpha_mix"] <= 1:
        raise ValueError("ibte_alpha_mix must be in (0, 1]")
    if settings["ibte_tol"] <= 0 or settings["ibte_niter"] < 1:
        raise ValueError("ibte_tol must be > 0 and ibte_niter >= 1")

    resolved = resolve_scattering_type(settings)
    unsupported = [s for s in resolved if s not in IBTE_SUPPORTED_MECHANISMS]
    if unsupported:
        extra = ""
        if settings["scattering_type"] == "auto":
            extra = (
                " (scattering_type: auto enabled them; set scattering_type explicitly, "
                "e.g. [ADP, POP] for phonon-limited mobilities)"
            )
        raise ValueError(
            "solver: ibte supports only "
            + ", ".join(IBTE_SUPPORTED_MECHANISMS)
            + " in this version; unsupported: "
            + ", ".join(unsupported)
            + extra
        )
    if not resolved:
        raise ValueError("No scattering mechanisms for solver: ibte")
    settings["scattering_type"] = resolved

    mrta_mechs = settings.get("ibte_mrta_mechanisms") or []
    if isinstance(mrta_mechs, str):
        mrta_mechs = [mrta_mechs]
    bad = [m for m in mrta_mechs if m not in IBTE_SUPPORTED_MECHANISMS]
    if bad:
        raise ValueError(f"ibte_mrta_mechanisms: unknown mechanisms {bad}")
    settings["ibte_mrta_mechanisms"] = list(mrta_mechs) or None
    if "IMP" in resolved and "IMP" not in mrta_mechs and settings["ibte_method"] == "iterate":
        logger.warning(
            "IMP in the IBTE: forward-peaked scattering makes linear mixing converge "
            "very slowly; consider ibte_method: gmres, or ibte_mrta_mechanisms: [IMP]"
        )

    if not settings["calculate_mobility"]:
        raise ValueError("solver: ibte requires calculate_mobility: true")

    logger.warning(
        "solver: ibte computes the mobility only. Conductivity, Seebeck and thermal "
        f"conductivity are still computed in the RTA (rta_mode: {settings['rta_mode']})."
    )
    if settings["separate_mobility"]:
        logger.info(
            "Per-mechanism mobilities (separate_mobility) are RTA-only; scattering "
            "mechanisms do not add under the IBTE."
        )
    return settings
