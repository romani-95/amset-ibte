# <img alt="amset" src="https://raw.githubusercontent.com/hackingmaterials/amset/main/docs/src/amset_logo.png" width="250">

> **AMSET-IBTE fork.** This version adds an iterative solution of the Boltzmann
> transport equation (`solver: ibte`, Claes et al. PRB 106, 094302) to AMSET 0.5.1.
> It also adds mode-resolved and direction-dependent polar-optical scattering
> (`pop_frequency` lists / `pop_modes_file`), IMP and free-carrier screening in the
> IBTE, and `ibte_mrta_mechanisms`. With default settings the results are
> bit-identical to upstream AMSET.
>
> - Installation (conda, clusters): [INSTALL.md](INSTALL.md)
> - IBTE settings and physics: [docs/ibte.md](docs/ibte.md)
> - Helper scripts (LO modes from DFPT, convergence scans, result checks):
>   [scripts/](scripts/README.md)
>
> This is a beta version. Please report problems in the Issues tab with your
> `settings.yaml` and `amset.log`.

AMSET is a package for calculating electronic transport properties from
first-principles calculations.

- **Website (including documentation):** https://hackingmaterials.lbl.gov/amset/
- **Help/Support:** https://discuss.matsci.org/c/amset
- **Source:** https://github.com/hackingmaterials/amset

If you find AMSET useful, please consider citing [our paper](https://www.nature.com/articles/s41467-021-22440-5):

```
Ganose, A. M., Park, J., Faghaninia, A., Woods-Robinson, R., Persson, K. A., Jain, A.
Efficient calculation of carrier scattering rates from first principles.
Nat. Commun. 12, 2222 (2021)
```

Interested in contributing? See our [contribution guidelines](https://github.com/hackingmaterials/amset/blob/main/CONTRIBUTING.md)
