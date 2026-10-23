# Installing AMSET-IBTE (laptop or cluster)

## Conda (recommended)

```bash
git clone <your-repo-url> amset-ibte
cd amset-ibte
conda env create -f environment.yml      # creates "amset-ibte", installs this repo with pip -e
conda activate amset-ibte
```

Takes ~2 minutes. Everything comes from conda-forge, including a pre-built
BoltzTraP2. The package is installed in editable mode, so a `git pull` updates it
without reinstalling.

If your cluster's conda is old, `mamba env create -f environment.yml` is faster.
Load an Anaconda/Miniforge module first if needed (e.g. `module load Miniforge3`).

## Check the installation

```bash
python -c "from amset.core.run import Runner; import BoltzTraP2; print('ok')"
pytest tests/test_ibte_run.py -q          # ~5 min; full suite: pytest tests (~20 min)
```

Note on spglib/BoltzTraP2: the conda-forge BoltzTraP2 binary needs spglib's C
library loaded globally. `import amset` now does this itself, so
`LD_PRELOAD=$CONDA_PREFIX/lib/libsymspg.so` is no longer needed. If you import
BoltzTraP2 directly before amset in your own scripts, `import amset` first.

## Pinned versions and why

| package | version | reason |
|---|---|---|
| numpy | < 2 (1.26) | upstream AMSET uses `np.Inf` / `np.trapz` |
| spglib | 2.6.0 | AMSET uses `from spglib import spglib` (removed in 2.7+) |
| setuptools | < 70 | AMSET imports `pkg_resources` |
| BoltzTraP2 | 25.3.1 (conda-forge) | the pip build needs cmake and a compiler |
| python | 3.11 | tested version |

`requirements.txt` lists the same versions for a pip-only install
(`pip install -r requirements.txt && pip install -e .`). This needs a working C++
toolchain and cmake for BoltzTraP2.

## Running on a cluster

AMSET runs in parallel with Python multiprocessing on one node (`nworkers` in
`settings.yaml`), not MPI. Example SLURM script:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"   # needed in batch jobs
conda activate amset-ibte
export OMP_NUM_THREADS=1        # one thread per worker; avoids oversubscription
cd $SLURM_SUBMIT_DIR            # contains vasprun.xml, settings.yaml, wavefunction.h5, ...
amset run                       # nworkers in settings.yaml = cpus-per-task
```

Set `nworkers` explicitly. `nworkers: -1` may see every core of the node, not only
your allocation. Create the environment on a login node: compute nodes often have
no internet access.

Memory (mostly for the IBTE kernel):

- **Kernel size grows fast with the mesh.** Example, a cubic oxide with SOC (holes,
  8 valence bands in the window): 0.7 GB at 37³, 1.8 GB at 45³, about 7 GB at 63³.
  Requesting ≥ 4× the kernel size is safe.
- **Workers:** each needs ~0.5–1 GB plus the shared wavefunction cache.
- **Doping-dependent mechanisms** (IMP; POP/PIE with `free_carrier_screening:
  true`) store one kernel entry per doping and temperature. A 10-doping screened run
  then has ~10× the POP kernel. If memory is short, run fewer dopings per job.
- **Degenerate doping** (Fermi level inside the band) widens the energy window and
  the kernel a lot (same example at 1e21 cm⁻³: > 5 GB at 37³).

## IBTE settings quick reference

```yaml
solver: ibte                  # rta = upstream AMSET (default, bit-identical)
ibte_method: gmres            # needed when IMP is present; iterate is fine for phonons
ibte_niter: 400
ibte_mrta_mechanisms: [IMP]   # recommended with IMP at low/intermediate doping
pop_modes_file: modes.json    # mode-resolved, NAC-corrected POP (scripts/lo_phonons.py)
fix_kpoint_symmetry: true     # fewer irreducible k-points for hexagonal/FCC cells
```

Outputs, in `transport_*.json` under `mobility`:

- `IBTE`;
- `IBTE_full` (with `ibte_mrta_mechanisms`);
- `SERTA` and `MRTA` (MRTA factor on all mechanisms);
- `overall` (the AMSET default RTA);
- per-mechanism RTA values.

Physics and options: `docs/ibte.md`. Helper scripts: `scripts/` (see
`scripts/README.md`). `scripts/sanity_checks.py <run_dir>` checks finished runs:
IBTE ≥ SERTA, convergence, isotropy for cubic cells (`--cubic`).
