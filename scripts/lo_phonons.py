"""LO phonons from a VASP DFPT run (IBRION = 7/8, LEPSILON) and effective POP frequencies.

`amset phonon-frequency` without a NAC-corrected OUTCAR averages the plain Γ-point
(TO) frequencies. Fröhlich scattering is carried by the LO modes, which in strongly
polar oxides are far above the TO modes (ω_LO/ω_TO ≈ √(ε₀/ε∞)). Here the
non-analytic correction (NAC) is added to the Γ dynamical matrix for each direction q̂
on a sphere::

    D(q̂) = D_an + (4π e²/Ω) (q̂·Z*_κ)(q̂·Z*_κ') / (q̂·ε∞·q̂) / √(M_κ M_κ')

and the resulting modes are used to compute effective frequencies.

Checks printed:
  1. D_an reproduces the vasprun eigenvalues.
  2. The ionic dielectric tensor from the TO modes matches the OUTCAR one.
  3. Generalized Lyddane–Sachs–Teller: Π ω_LO²/Π ω_TO² = (q̂ε₀q̂)/(q̂ε∞q̂).
  4. The AMSET recipe without NAC reproduces the original `pop_frequency`.
  5. Sum rule: Σ_ν C_ν(q̂) = 1/(q̂ε∞q̂) − 1/(q̂ε₀q̂) (mode-resolved Fröhlich weights).

Usage:
    python lo_phonons.py PHONON_DIR [--eps-inf "e1 e2 e3"] [--modes-file modes.json] [--json out.json]

By default ε∞ is the one of the phonon run (OUTCAR, or vasprun.xml for LCALCEPS),
consistent with the force constants
and Born charges.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core.tensors import DEFAULT_QUAD
from pymatgen.io.vasp import Outcar, Vasprun

COULOMB_EV_ANG = 14.399645  # e²/(4πε0) in eV·Å
EV_ANG2_AMU_TO_THZ2 = 15.633302**2  # (eV/Å²/amu) -> THz² (frequency, not angular)


def _xml_dielectric_data(path):
    """Born charges, ε∞ and ε_ion from vasprun.xml(.gz).

    Needed for finite-difference phonons (IBRION = 5/6) with LCALCEPS = .TRUE., where
    the OUTCAR has no LEPSILON blocks. ε∞ is ``epsilon_scf`` (finite-field, includes
    local-field effects), falling back to ``epsilon``.
    """
    import gzip
    import xml.etree.ElementTree as ET

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        root = ET.parse(f).getroot()

    def tensor(name):
        node = root.find(f".//varray[@name='{name}']")
        if node is None:
            return None
        return np.array([[float(x) for x in v.text.split()] for v in node.findall("v")])

    born_node = root.find(".//array[@name='born_charges']")
    born = np.array([
        [[float(x) for x in v.text.split()] for v in s.findall("v")]
        for s in born_node.findall("set")
    ])
    eps_inf = tensor("epsilon_scf")
    if eps_inf is None:
        eps_inf = tensor("epsilon")
    return born, eps_inf, tensor("epsilon_ion")


def load(dfpt_dir):
    dfpt_dir = Path(dfpt_dir)
    vr_path = next(p for p in (dfpt_dir / "vasprun.xml", dfpt_dir / "vasprun.xml.gz")
                   if p.exists())
    vr = Vasprun(str(vr_path), parse_potcar_file=False)
    out = None
    for name in ("OUTCAR", "OUTCAR.gz"):
        if (dfpt_dir / name).exists():
            out = Outcar(str(dfpt_dir / name))
            break
    try:
        out.read_lepsilon()
        out.read_lepsilon_ionic()
        born = np.asarray(out.born)
        eps_inf = np.asarray(out.dielectric_tensor)
        eps_ion = np.asarray(out.dielectric_ionic_tensor)
        if born.size == 0:
            raise ValueError("no LEPSILON output")
    except Exception:
        born, eps_inf, eps_ion = _xml_dielectric_data(vr_path)
    structure = vr.final_structure
    n = len(structure)
    hessian = np.asarray(vr.force_constants).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    major = int(vr.vasp_version.split(".")[0])
    if major < 6:
        hessian = hessian * EV_ANG2_AMU_TO_THZ2
    return {
        "structure": structure,
        "masses": np.array([s.specie.atomic_mass for s in structure]),
        "dynmat": -hessian,  # THz² (VASP stores the negative mass-weighted Hessian)
        "vasprun_eigs": -np.asarray(vr.normalmode_eigenvals),
        "born": born,  # (n, 3 field, 3 displacement)
        "eps_inf": eps_inf,
        "eps_ion": eps_ion,
        "vasprun": vr,
    }


def nac_term(qhat, born, masses, eps_inf, volume):
    """Non-analytic part of the dynamical matrix along qhat, in THz²."""
    qz = np.einsum("a,kab->kb", qhat, born)  # (n, 3): (q̂·Z*_κ)_β
    vec = (qz / np.sqrt(masses)[:, None]).ravel()
    pref = 4 * np.pi * COULOMB_EV_ANG / volume / (qhat @ eps_inf @ qhat)
    return pref * np.outer(vec, vec) * EV_ANG2_AMU_TO_THZ2


def modes(dynmat):
    w2, vecs = np.linalg.eigh(dynmat)
    freqs = np.sqrt(np.abs(w2)) * np.sign(w2)
    return freqs, vecs.T  # vecs[ν] is the normalized mass-weighted eigenvector


def polarity(vec, born, masses):
    """p_ν = Σ_κ Z*_κ · e_κν / √M_κ  (vector over field directions)."""
    e = vec.reshape(-1, 3)
    return np.einsum("kab,kb->a", born, e / np.sqrt(masses)[:, None])


def amset_weight(qhat, vec, freq, born, masses):
    """AMSET's per-mode weight |q̂·Σ Z*·e/√(Mω)| (amset.tools.phonon_frequency)."""
    if freq <= 0.1:
        return 0.0
    return abs(qhat @ polarity(vec, born, masses)) / np.sqrt(freq)


def analyse(data, eps_inf=None):
    s = data["structure"]
    vol = s.volume
    born, masses = data["born"], data["masses"]
    eps_inf = data["eps_inf"] if eps_inf is None else eps_inf
    eps0 = eps_inf + data["eps_ion"]
    report = {"eps_inf": eps_inf.tolist(), "eps_ion": data["eps_ion"].tolist()}

    # 1. analytic part reproduces VASP
    f_to, v_to = modes(data["dynmat"])
    vasp_f = np.sort(np.sqrt(np.abs(data["vasprun_eigs"])) * np.sign(data["vasprun_eigs"]))
    report["check_vasp_freqs_max_abs_THz"] = float(np.max(np.abs(np.sort(f_to) - vasp_f)))

    # 2. ionic dielectric from TO modes
    eps_ion = np.zeros((3, 3))
    for f, v in zip(f_to, v_to):
        if f > 0.5:
            p = polarity(v, born, masses)
            eps_ion += np.outer(p, p) / f**2
    eps_ion *= 4 * np.pi * COULOMB_EV_ANG / vol * EV_ANG2_AMU_TO_THZ2
    report["eps_ion_from_modes"] = eps_ion.tolist()

    # 4. AMSET recipe without NAC (TO modes), as amset phonon-frequency
    points, qweights = DEFAULT_QUAD["points"], DEFAULT_QUAD["weights"]
    w_to = np.array(
        [
            np.average([amset_weight(q, v, f, born, masses) for q in points], weights=qweights)
            for f, v in zip(f_to, v_to)
        ]
    )
    report["omega_eff_amset_TO"] = float(np.sum(w_to * f_to) / np.sum(w_to))

    # with NAC, direction by direction
    sum_w_amset = sum_wf_amset = sum_c = sum_cf = 0.0
    lst_err = []
    sumrule_err = []
    per_mode = []  # (direction-averaged ω_LO, coupling share) per mode index
    mode_f = np.zeros(len(f_to))
    mode_c = np.zeros(len(f_to))
    for q, qw in zip(points, qweights):
        f_lo, v_lo = modes(data["dynmat"] + nac_term(q, born, masses, eps_inf, vol))
        # 3. generalized LST over optical modes (skip the three acoustic ones)
        ratio = np.prod(np.sort(f_lo)[3:] ** 2 / np.sort(f_to)[3:] ** 2)
        lst_err.append(ratio / ((q @ eps0 @ q) / (q @ eps_inf @ q)) - 1)
        # 5. mode-resolved Fröhlich weights C_ν = (4π/Ω)|q̂·p_ν|²/(ω_ν² (q̂ε∞q̂)²)
        c = np.zeros(len(f_lo))
        for i, (f, v) in enumerate(zip(f_lo, v_lo)):
            if f > 0.5:
                c[i] = (q @ polarity(v, born, masses)) ** 2 / f**2
        c *= 4 * np.pi * COULOMB_EV_ANG / vol * EV_ANG2_AMU_TO_THZ2 / (q @ eps_inf @ q) ** 2
        target = 1 / (q @ eps_inf @ q) - 1 / (q @ eps0 @ q)
        sumrule_err.append(c.sum() / target - 1)
        # effective frequencies
        wa = np.array([amset_weight(q, v, f, born, masses) for f, v in zip(f_lo, v_lo)])
        sum_w_amset += qw * wa.sum()
        sum_wf_amset += qw * (wa * f_lo).sum()
        sum_c += qw * c.sum()
        sum_cf += qw * (c * f_lo).sum()
        order = np.argsort(f_lo)
        mode_f += qw * f_lo[order]
        mode_c += qw * c[order]

    report["check_lst_max_rel_err"] = float(np.max(np.abs(lst_err)))
    report["check_sumrule_max_rel_err"] = float(np.max(np.abs(sumrule_err)))
    report["omega_eff_amset_LO"] = float(sum_wf_amset / sum_w_amset)
    report["omega_eff_coupling_LO"] = float(sum_cf / sum_c)
    qsum = np.sum(qweights)
    mode_f /= qsum
    mode_c /= qsum
    keep = mode_c > 1e-3 * mode_c.sum()
    report["lo_modes"] = [
        {"omega_THz": float(f), "weight": float(c / mode_c.sum())}
        for f, c in zip(mode_f[keep], mode_c[keep])
    ]
    return report


def merge_modes(modes, merge_thz=1.0, min_weight=0.02):
    """Reduce the LO mode list for mode-resolved POP in AMSET.

    1. Modes closer than ``merge_thz`` are combined (weights add, frequency is the
       weight-averaged one).
    2. Groups with weight < ``min_weight`` are folded into the nearest remaining
       group in frequency (weights add; that group keeps its frequency).
    Total weight is conserved (the weights sum to 1).
    """
    modes = sorted(modes, key=lambda m: m["omega_THz"])
    groups = []
    for m in modes:
        if groups and m["omega_THz"] - groups[-1]["last"] < merge_thz:
            g = groups[-1]
            g["wf"] += m["weight"] * m["omega_THz"]
            g["w"] += m["weight"]
            g["last"] = m["omega_THz"]
        else:
            groups.append({"wf": m["weight"] * m["omega_THz"], "w": m["weight"],
                           "last": m["omega_THz"]})
    merged = [{"omega_THz": g["wf"] / g["w"], "weight": g["w"]} for g in groups]
    while len(merged) > 1:
        small = [i for i, m in enumerate(merged) if m["weight"] < min_weight]
        if not small:
            break
        i = min(small, key=lambda j: merged[j]["weight"])
        others = [j for j in range(len(merged)) if j != i]
        j = min(others, key=lambda k: abs(merged[k]["omega_THz"] - merged[i]["omega_THz"]))
        merged[j]["weight"] += merged[i]["weight"]
        del merged[i]
    total = sum(m["weight"] for m in merged)
    return [{"omega_THz": m["omega_THz"], "weight": m["weight"] / total} for m in merged]


def monomial_exponents(max_degree=4):
    """Exponents (a, b, c) of the even monomials q_x^a q_y^b q_z^c up to max_degree.

    Restricted to the unit sphere these span the spherical harmonics l = 0, 2, ...,
    max_degree (the set is over-complete because q² = 1; least squares handles it).
    """
    return [
        (a, b, d - a - b)
        for d in range(0, max_degree + 1, 2)
        for a in range(d + 1)
        for b in range(d - a + 1)
    ]


def monomials(q, exponents):
    q = np.atleast_2d(q)
    e = np.asarray(exponents)
    return np.prod(q[:, None, :] ** e[None, :, :], axis=2)


def anisotropic_weights(data, eps_inf, frequencies, max_degree=4):
    """Direction-dependent mode weights w_m(q̂) for mode-resolved POP.

    For every direction q̂ (974-point Lebedev grid) the NAC LO modes are computed and
    the Fröhlich coupling C_ν(q̂) of each is assigned to the nearest frequency in
    ``frequencies``; w_m(q̂) = C_m(q̂) / Σ C(q̂) is the exact share of
    4π(1/ε∞(q̂) − 1/ε₀(q̂)) carried by mode m along q̂. Each w_m is fitted with even
    monomials up to ``max_degree``. Least squares is linear, so Σ_m w_m(q̂) = 1
    exactly for the fits, and the direction average of each fit equals the
    direction-averaged share.
    """
    born, masses, vol = data["born"], data["masses"], data["structure"].volume
    bins = np.asarray(frequencies)
    points = np.asarray(DEFAULT_QUAD["points"])
    qweights = np.asarray(DEFAULT_QUAD["weights"])
    shares = np.zeros((len(points), len(bins)))
    for iq, q in enumerate(points):
        f_lo, v_lo = modes(data["dynmat"] + nac_term(q, born, masses, eps_inf, vol))
        for f, v in zip(f_lo, v_lo):
            if f > 0.5:
                shares[iq, np.argmin(np.abs(bins - f))] += (q @ polarity(v, born, masses)) ** 2 / f**2
        shares[iq] /= shares[iq].sum()

    exponents = monomial_exponents(max_degree)
    basis = monomials(points, exponents)
    sw = np.sqrt(qweights)[:, None]
    coeffs, *_ = np.linalg.lstsq(basis * sw, shares * sw, rcond=None)  # (K, nmodes)
    fitted = basis @ coeffs
    resid = fitted - shares
    return {
        "frequencies": bins.tolist(),
        "weights": (qweights @ shares / qweights.sum()).tolist(),
        "exponents": [list(e) for e in exponents],
        "coefficients": coeffs.T.tolist(),
        "fit_rms": np.sqrt(qweights @ resid**2 / qweights.sum()).tolist(),
        "fit_max_abs": np.abs(resid).max(axis=0).tolist(),
        "fit_min_value": fitted.min(axis=0).tolist(),
        "sum_rule_max_err": float(np.abs(fitted.sum(axis=1) - 1).max()),
        "share_range": [shares.min(axis=0).tolist(), shares.max(axis=0).tolist()],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dfpt_dir")
    p.add_argument("--eps-inf", help="ε∞ to use instead of the DFPT one: 3 diagonal "
                   "values or 9 values of the full tensor (row-major)")
    p.add_argument("--json")
    p.add_argument("--merge-thz", type=float, default=1.0,
                   help="combine LO modes closer than this (THz)")
    p.add_argument("--min-weight", type=float, default=0.02,
                   help="fold modes with a smaller coupling share into a neighbour")
    p.add_argument("--max-degree", type=int, default=4,
                   help="even polynomial degree of the direction-dependent weight fit")
    p.add_argument("--modes-file",
                   help="write the AMSET pop_modes_file (direction-dependent weights)")
    args = p.parse_args()
    data = load(args.dfpt_dir)
    eps_inf = None
    if args.eps_inf:
        vals = [float(x) for x in args.eps_inf.replace(",", " ").split()]
        eps_inf = np.diag(vals) if len(vals) == 3 else np.reshape(vals, (3, 3))
    rep = analyse(data, eps_inf)

    print(f"ε∞ used (diag): {np.diag(rep['eps_inf']).round(3)}")
    print(f"ε_ion (phonon run, diag): {np.diag(rep['eps_ion']).round(3)}")
    print(f"ε₀ = ε∞ + ε_ion (diag): {np.diag(np.asarray(rep['eps_inf']) + np.asarray(rep['eps_ion'])).round(3)}")
    print(f"ε_ion from TO modes (diag): {np.diag(rep['eps_ion_from_modes']).round(3)}")
    print(f"check: |f - f_VASP| max = {rep['check_vasp_freqs_max_abs_THz']:.2e} THz")
    print(f"check: LST max rel. error = {rep['check_lst_max_rel_err']:.2e}")
    print(f"check: Fröhlich sum rule max rel. error = {rep['check_sumrule_max_rel_err']:.2e}")
    print(f"ω_eff AMSET recipe, TO modes (no NAC): {rep['omega_eff_amset_TO']:.2f} THz")
    print(f"ω_eff AMSET recipe, LO modes (NAC)   : {rep['omega_eff_amset_LO']:.2f} THz")
    print(f"ω_eff coupling-weighted, LO modes    : {rep['omega_eff_coupling_LO']:.2f} THz")
    print("LO modes (direction-averaged ω, share of 1/ε∞ − 1/ε₀):")
    for m in rep["lo_modes"]:
        print(f"   {m['omega_THz']:7.2f} THz   {m['weight']:.3f}")
    rep["merged_modes"] = merge_modes(rep["lo_modes"], args.merge_thz, args.min_weight)
    print(f"Merged modes (merge < {args.merge_thz} THz, min weight {args.min_weight}):")
    print("   pop_frequency: [" + ", ".join(
        f"{m['omega_THz']:.2f}" for m in rep["merged_modes"]) + "]")
    print("   pop_weights:   [" + ", ".join(
        f"{m['weight']:.4f}" for m in rep["merged_modes"]) + "]")
    if args.modes_file:
        eps = data["eps_inf"] if eps_inf is None else eps_inf
        aniso = anisotropic_weights(
            data, eps, [m["omega_THz"] for m in rep["merged_modes"]], args.max_degree
        )
        Path(args.modes_file).write_text(json.dumps(aniso, indent=1))
        rep["anisotropic"] = aniso
        print("Direction-dependent weights (pop_modes_file):")
        for i, f in enumerate(aniso["frequencies"]):
            lo_s, hi_s = aniso["share_range"][0][i], aniso["share_range"][1][i]
            print(f"   {f:6.2f} THz  mean {aniso['weights'][i]:.3f}  "
                  f"range {lo_s:.3f}–{hi_s:.3f}  fit rms {aniso['fit_rms'][i]:.3f}  "
                  f"max {aniso['fit_max_abs'][i]:.3f}")
        print(f"   sum rule max error {aniso['sum_rule_max_err']:.1e}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
