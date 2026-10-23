"""In-scattering kernel of the iterative Boltzmann transport equation.

AMSET integrates the scattering rate of a state (n, k) over the constant energy
surface ε' = ε (elastic) or ε' = ε ± ħω (POP) using quadrature points k' that do not
lie on the k-point mesh. In the IBTE the same quadrature points are needed for the
in-scattering term of Claes et al. Eq. (S12)::

    Λ_nk = τ⁰_nk v_nk + τ⁰_nk Σ_k' K_in(k, k') [f⁰'(ε') / f⁰'(ε)] Λ_k'

where Λ = (∂f/∂E) / (∂f⁰/∂ε). Λ at the off-grid k' is obtained by trilinear
interpolation from the full mesh, exactly as ``MRTACalculator`` interpolates the
velocities. Precomputing the eight corner indices and weights turns the
in-scattering term into a sparse matrix-vector product, with rows the irreducible
states (n, k_ir) and columns the full-mesh states (m, k).

The stored matrix elements are "geometric": they contain |g|², the wavefunction
overlap and the quadrature weight, but not the (doping, temperature) dependent
prefactor, the occupation factors or the MRTA factor. These are applied at solve
time (see ``amset.ibte.solver``).
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from pymatgen.electronic_structure.core import Spin

from amset.interpolation.periodic import PeriodicLinearInterpolator

__author__ = "AMSET-IBTE"

logger = logging.getLogger(__name__)

# elastic: ε' = ε; absorption: ε' = ε + ħω; emission: ε' = ε − ħω
KERNEL_CHANNELS = ("elastic", "absorption", "emission")


def inelastic_channel(kind, mode, nmodes):
    """Kernel channel name of polar mode ``mode`` ("absorption" or "emission").

    A single POP mode keeps the plain names (so kernels written before
    mode-resolved POP still load); several modes use "absorption_0", ... .
    """
    return kind if nmodes == 1 else f"{kind}_{mode}"


def channel_kind(channel):
    """"elastic", "absorption" or "emission" for any channel name."""
    return channel.split("_")[0]

_CORNERS = np.array(
    [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=int
)


class TrilinearMapper:
    """Trilinear interpolation weights on the full mesh.

    Reproduces the grid ordering, periodic padding and index clipping used by
    ``PeriodicLinearInterpolator`` (``interpolation.splines.eval_linear`` on a
    ``UCGrid`` with linear extrapolation)::

        u = (x - a) (n - 1) / (b - a);  i = clip(floor(u), 0, n - 2);  λ = u - i

    so that ``Σ_c w_c X[c]`` equals ``PeriodicLinearInterpolator.interpolate`` for
    any data X given on the full mesh. The band coordinate is an integer, so the
    interpolation along the band axis is exact and not needed.

    Args:
        start: Start of the padded grid in each direction (3,).
        stop: End of the padded grid in each direction (3,).
        npad: Number of padded grid points in each direction (3,).
        grid_to_full: Map from padded grid index (i, j, k) to full mesh k-point
            index, shape ``npad``.
        nkpoints: Number of k-points in the full mesh.
    """

    def __init__(self, start, stop, npad, grid_to_full, nkpoints):
        self.start = np.asarray(start, dtype=float)
        self.stop = np.asarray(stop, dtype=float)
        self.npad = np.asarray(npad, dtype=int)
        self.grid_to_full = np.asarray(grid_to_full, dtype=np.int64)
        self.nkpoints = int(nkpoints)
        self.inv_step = (self.npad - 1.0) / (self.stop - self.start)

    @classmethod
    def from_kpoints(cls, kpoints: np.ndarray):
        """Build the mapper from the full-mesh k-points (fractional coordinates)."""
        grid_kpoints, mesh_dim, sort_idx = PeriodicLinearInterpolator._grid_kpoints(
            kpoints
        )
        x = grid_kpoints[:, 0, 0, 0]
        y = grid_kpoints[0, :, 0, 1]
        z = grid_kpoints[0, 0, :, 2]
        start = np.array([x[0], y[0], z[0]])
        stop = np.array([x[-1], y[-1], z[-1]])
        npad = np.array([len(x), len(y), len(z)])

        # padded index p corresponds to unpadded index (p - 1) mod n ("wrap" padding)
        nx, ny, nz = mesh_dim
        ia = (np.arange(nx + 2) - 1) % nx
        ib = (np.arange(ny + 2) - 1) % ny
        ic = (np.arange(nz + 2) - 1) % nz
        flat = ia[:, None, None] * ny * nz + ib[None, :, None] * nz + ic[None, None, :]
        grid_to_full = np.asarray(sort_idx)[flat]
        return cls(start, stop, npad, grid_to_full, len(kpoints))

    def weights(self, kpoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the 8 corner k-point indices and weights for each k-point.

        Args:
            kpoints: Fractional coordinates in the first BZ, shape (npts, 3).

        Returns:
            ``(indices, weights)``, both with shape (npts, 8).
        """
        kpoints = np.atleast_2d(kpoints)
        u = (kpoints - self.start) * self.inv_step
        i = np.floor(u).astype(int)
        i = np.clip(i, 0, self.npad - 2)
        lam = u - i

        corner_idx = i[:, None, :] + _CORNERS[None, :, :]  # (npts, 8, 3)
        indices = self.grid_to_full[
            corner_idx[..., 0], corner_idx[..., 1], corner_idx[..., 2]
        ]
        # weight of a corner is Π_d (λ_d if corner_d == 1 else 1 - λ_d)
        w = np.where(_CORNERS[None, :, :] == 1, lam[:, None, :], 1 - lam[:, None, :])
        return indices, np.prod(w, axis=-1)

    def interpolate(self, data: np.ndarray, bands, kpoints):
        """Interpolate full-mesh data (nbands, nkpoints, ...) at (band, k)."""
        indices, weights = self.weights(kpoints)
        bands = np.asarray(bands)
        values = data[bands[:, None], indices]  # (npts, 8, ...)
        expand = (slice(None), slice(None)) + (None,) * (values.ndim - 2)
        return np.sum(values * weights[expand], axis=1)

    def map_values(self, kpoints, bands, values):
        """Distribute quadrature values onto the full-mesh states.

        Args:
            kpoints: Off-grid final k-points (npts, 3), fractional.
            bands: Final band indices (npts,).
            values: Values per channel at each point (nch, npts).

        Returns:
            ``(cols, row_values)``: unique full-state column indices
            ``band * nkpoints + k`` (int32) and the summed values (nch, ncols).
        """
        values = np.atleast_2d(values)
        if len(bands) == 0:
            return self.empty_row(len(values))

        indices, weights = self.weights(kpoints)
        cols = np.asarray(bands, dtype=np.int64)[:, None] * self.nkpoints + indices
        cols = cols.ravel()
        contrib = (values[:, :, None] * weights[None]).reshape(len(values), -1)

        unique_cols, inverse = np.unique(cols, return_inverse=True)
        row_values = np.zeros((len(values), len(unique_cols)))
        for ich in range(len(values)):
            row_values[ich] = np.bincount(
                inverse, weights=contrib[ich], minlength=len(unique_cols)
            )
        return unique_cols.astype(np.int32), row_values

    @staticmethod
    def empty_row(nchannels):
        return np.zeros(0, dtype=np.int32), np.zeros((nchannels, 0))


class KernelChannel:
    """Sparse geometric kernel for one spin and one channel.

    Stored as a CSR pattern shared by all mechanisms of the channel, with one data
    array per mechanism: ``data`` has shape (nmech, nnz).

    Args:
        indptr: CSR row pointer (nrows + 1,).
        indices: CSR column indices (nnz,).
        data: Values (nmech, nnz).
        shape: (nrows, ncols) = (nbands * n_ir_kpoints, nbands * nkpoints).
        rowsum_mrta: Row sums weighted by the MRTA factor (nmech, nrows); used to
            reconstruct MRTA rates when the kernel is loaded from file.
        computed: Whether the row was computed (inside the energy cut-offs).
    """

    def __init__(self, indptr, indices, data, shape, rowsum_mrta, computed):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        self.shape = tuple(int(x) for x in shape)
        self.rowsum_mrta = np.atleast_2d(rowsum_mrta)
        self.computed = np.asarray(computed, dtype=bool)
        self._row_of_nnz = None

    @property
    def nmech(self):
        return self.data.shape[0]

    @property
    def nnz(self):
        return len(self.indices)

    @property
    def nbytes(self):
        return self.indptr.nbytes + self.indices.nbytes + self.data.nbytes

    @property
    def row_of_nnz(self):
        if self._row_of_nnz is None:
            counts = np.diff(self.indptr)
            self._row_of_nnz = np.repeat(
                np.arange(self.shape[0], dtype=np.int32), counts
            )
        return self._row_of_nnz

    @property
    def rowsum(self):
        """Row sums of the geometric kernel (nmech, nrows).

        Because the trilinear weights of each quadrature point sum to 1, this equals
        the plain quadrature sum, i.e. the out-scattering rate without prefactor and
        occupation factors.
        """
        out = np.zeros((self.nmech, self.shape[0]))
        for m in range(self.nmech):
            out[m] = np.bincount(
                self.row_of_nnz, weights=self.data[m], minlength=self.shape[0]
            )
        return out

    def has_entries(self):
        return np.diff(self.indptr) > 0

    def matrix(self, row_coefficients: np.ndarray, entries=None) -> sp.csr_matrix:
        """Get Σ_m diag(c_m) G_{e(m)} as a CSR matrix.

        Args:
            row_coefficients: Coefficients per mechanism and row (nmech, nrows).
            entries: Data row used for each mechanism (nmech,); defaults to
                ``range(nmech)`` (one entry per mechanism).
        """
        row_coefficients = np.atleast_2d(row_coefficients)
        if entries is None:
            entries = range(len(row_coefficients))
        data = np.zeros(self.nnz)
        rows = self.row_of_nnz
        for m, e in enumerate(entries):
            data += row_coefficients[m][rows] * self.data[e]
        return sp.csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    @classmethod
    def from_rows(cls, rows: List[tuple], nrows: int, ncols: int, nmech: int):
        """Assemble from a list of ``(row_index, cols, values, rowsum_mrta)``."""
        rows = sorted(rows, key=lambda r: r[0])
        counts = np.zeros(nrows, dtype=np.int64)
        rowsum_mrta = np.zeros((nmech, nrows))
        computed = np.zeros(nrows, dtype=bool)
        for r_idx, cols, _, r_mrta in rows:
            counts[r_idx] = len(cols)
            rowsum_mrta[:, r_idx] = r_mrta
            computed[r_idx] = True

        indptr = np.zeros(nrows + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(counts)
        nnz = int(indptr[-1])
        indices = np.zeros(nnz, dtype=np.int32)
        data = np.zeros((nmech, nnz))
        for r_idx, cols, values, _ in rows:
            start = indptr[r_idx]
            indices[start : start + len(cols)] = cols
            data[:, start : start + len(cols)] = values
        return cls(indptr, indices, data, (nrows, ncols), rowsum_mrta, computed)


def entry_layout(counts, ndops, ntemps):
    """Data row of each (mechanism, doping, temperature) in a kernel channel.

    Args:
        counts: Number of entries per mechanism: 1 (independent of doping and
            temperature) or ndops × ntemps (see ``kernel_entries`` in
            ``amset.scattering.calculate``).

    Returns:
        Integer array (nmech, ndops, ntemps).
    """
    index = np.zeros((len(counts), ndops, ntemps), dtype=int)
    start = 0
    for m, count in enumerate(counts):
        if count == 1:
            index[m] = start
        elif count == ndops * ntemps:
            index[m] = start + np.arange(count).reshape(ndops, ntemps)
        else:
            raise ValueError(f"Invalid number of kernel entries: {count}")
        start += count
    return index


class IBTEKernel:
    """The full in-scattering kernel (all spins and channels).

    Args:
        channels: ``{spin: {channel: KernelChannel}}``.
        mechanisms: ``{channel: [mechanism names]}``.
        nbands: ``{spin: nbands}``.
        n_ir_kpoints: Number of irreducible k-points.
        nkpoints: Number of full-mesh k-points.
        metadata: Free-form metadata (mesh, settings hash, version).
        entry_index: ``{channel: array (nmech, ndops, ntemps)}`` giving the data row
            of each mechanism for each doping and temperature (see
            :func:`entry_layout`). ``None`` (or a missing channel) means one entry
            per mechanism, valid for all dopings and temperatures.
    """

    def __init__(
        self,
        channels,
        mechanisms,
        nbands,
        n_ir_kpoints,
        nkpoints,
        metadata,
        entry_index=None,
    ):
        self.channels = channels
        self.mechanisms = mechanisms
        self.nbands = nbands
        self.n_ir_kpoints = int(n_ir_kpoints)
        self.nkpoints = int(nkpoints)
        self.metadata = metadata
        self.entry_index = {
            c: np.asarray(v, dtype=int) for c, v in (entry_index or {}).items()
        }

    @property
    def spins(self):
        return list(self.channels.keys())

    @property
    def doping_dependent(self):
        """Whether any mechanism has separate entries per doping/temperature."""
        return any(len(np.unique(v[m])) > 1 for v in self.entry_index.values()
                   for m in range(len(v)))

    def entries(self, channel, n, t):
        """Data row of each mechanism of ``channel`` at doping n, temperature t."""
        index = self.entry_index.get(channel)
        if index is None:
            return np.arange(len(self.mechanisms[channel]))
        return index[:, n, t]

    @classmethod
    def from_rows(
        cls,
        kernel_rows,
        mechanisms,
        nbands,
        n_ir_kpoints,
        nkpoints,
        metadata,
        entry_index=None,
    ):
        entry_index = entry_index or {}
        channels = {}
        for spin, spin_rows in kernel_rows.items():
            nrows = nbands[spin] * n_ir_kpoints
            ncols = nbands[spin] * nkpoints
            channels[spin] = {}
            for c in mechanisms:
                if len(mechanisms[c]) == 0:
                    continue
                nentries = (
                    int(entry_index[c].max()) + 1
                    if c in entry_index
                    else len(mechanisms[c])
                )
                channels[spin][c] = KernelChannel.from_rows(
                    spin_rows[c], nrows, ncols, nentries
                )
        return cls(
            channels, mechanisms, nbands, n_ir_kpoints, nkpoints, metadata, entry_index
        )

    def summary(self):
        lines = []
        for spin, spin_channels in self.channels.items():
            for c, ch in spin_channels.items():
                lines.append(
                    f"{spin.name} {c}: nnz = {ch.nnz}, {ch.nbytes / 1e6:.1f} MB, "
                    f"{int(ch.has_entries().sum())} non-empty rows"
                )
        return lines

    @property
    def nbytes(self):
        return sum(ch.nbytes for sc in self.channels.values() for ch in sc.values())

    def has_row(self, spin):
        """States (rows) with at least one kernel entry in any channel."""
        out = None
        for ch in self.channels[spin].values():
            out = ch.has_entries() if out is None else out | ch.has_entries()
        return out

    def computed(self, spin):
        out = None
        for ch in self.channels[spin].values():
            out = ch.computed if out is None else out | ch.computed
        return out

    def to_file(self, filename):
        import h5py

        with h5py.File(filename, "w") as f:
            f.attrs["metadata"] = json.dumps(self.metadata)
            f.attrs["mechanisms"] = json.dumps(self.mechanisms)
            f.attrs["n_ir_kpoints"] = self.n_ir_kpoints
            f.attrs["nkpoints"] = self.nkpoints
            if self.entry_index:
                f.attrs["entry_index"] = json.dumps(
                    {c: v.tolist() for c, v in self.entry_index.items()}
                )
            for spin, spin_channels in self.channels.items():
                g = f.create_group(spin.name)
                g.attrs["nbands"] = self.nbands[spin]
                for c, ch in spin_channels.items():
                    gc = g.create_group(c)
                    kw = {"compression": "gzip", "compression_opts": 4}
                    gc.create_dataset("indptr", data=ch.indptr, **kw)
                    gc.create_dataset("indices", data=ch.indices, **kw)
                    gc.create_dataset("data", data=ch.data, **kw)
                    gc.create_dataset("rowsum_mrta", data=ch.rowsum_mrta, **kw)
                    gc.create_dataset("computed", data=ch.computed, **kw)
                    gc.attrs["shape"] = ch.shape

    @classmethod
    def from_file(cls, filename):
        import h5py

        channels = {}
        nbands = {}
        with h5py.File(filename, "r") as f:
            metadata = json.loads(f.attrs["metadata"])
            mechanisms = json.loads(f.attrs["mechanisms"])
            n_ir = int(f.attrs["n_ir_kpoints"])
            nk = int(f.attrs["nkpoints"])
            entry_index = None
            if "entry_index" in f.attrs:
                entry_index = json.loads(f.attrs["entry_index"])
            for spin_name in f.keys():
                spin = Spin.up if spin_name == "up" else Spin.down
                g = f[spin_name]
                nbands[spin] = int(g.attrs["nbands"])
                channels[spin] = {}
                for c in g.keys():
                    gc = g[c]
                    channels[spin][c] = KernelChannel(
                        gc["indptr"][()],
                        gc["indices"][()],
                        gc["data"][()],
                        tuple(gc.attrs["shape"]),
                        gc["rowsum_mrta"][()],
                        gc["computed"][()],
                    )
        return cls(channels, mechanisms, nbands, n_ir, nk, metadata, entry_index)


# settings that change the kernel; used to check a loaded kernel matches the run
_KERNEL_SETTINGS = (
    "scattering_type",
    "interpolation_factor",
    "energy_cutoff",
    "scissor",
    "bandgap",
    "symprec",
    "soc",
    "wavefunction_coefficients",
    "use_projections",
    "unity_overlap",
    "high_frequency_dielectric",
    "static_dielectric",
    "elastic_constant",
    "deformation_potential",
    "piezoelectric_constant",
    "pop_frequency",
    "pop_weights",
    "pop_weight_coefficients",
    "free_carrier_screening",
)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, list)):
        return [_jsonable(i) for i in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def get_kernel_metadata(settings, amset_data):
    """Metadata identifying the band structure and settings a kernel belongs to."""
    from amset import __version__

    relevant = {k: _jsonable(settings.get(k)) for k in _KERNEL_SETTINGS}
    settings_hash = hashlib.sha256(
        json.dumps(relevant, sort_keys=True, default=str).encode()
    ).hexdigest()
    energy_hash = hashlib.sha256(
        b"".join(
            np.round(amset_data.energies[s], 8).tobytes() for s in amset_data.spins
        )
    ).hexdigest()
    return {
        "amset_version": __version__,
        "mesh": [int(x) for x in amset_data.kpoint_mesh],
        "nkpoints": int(len(amset_data.kpoints)),
        "n_ir_kpoints": int(len(amset_data.ir_kpoints_idx)),
        "settings_hash": settings_hash,
        "energy_hash": energy_hash,
        "settings": relevant,
        # needed for doping-dependent kernels (IMP, free-carrier screening)
        "doping": _jsonable(np.asarray(amset_data.doping, dtype=float)),
        "temperatures": _jsonable(np.asarray(amset_data.temperatures, dtype=float)),
    }


def check_kernel_compatible(kernel: IBTEKernel, metadata: Dict, strict: bool = True):
    """Raise if a loaded kernel does not belong to the current run."""
    problems = []
    keys = ["mesh", "nkpoints", "n_ir_kpoints", "energy_hash"]
    if kernel.doping_dependent:
        keys += ["doping", "temperatures"]
    for key in keys:
        if kernel.metadata.get(key) != metadata.get(key):
            problems.append(key)
    if kernel.metadata.get("settings_hash") != metadata.get("settings_hash"):
        problems.append("settings_hash")
    if problems:
        msg = (
            "Loaded IBTE kernel does not match this run (mismatch in: "
            + ", ".join(problems)
            + "). Recompute the kernel."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
