# Population Synthesis

A population-synthesis code such as SEVN evolves binaries at a handful of
metallicities. For each binary that ends as a merging compact pair, it reports
the delay from formation to merger. `gwmock_pop.synthesis` turns such a run into
a merger-rate density, then into the mergers an observer sees in a year, then
into a catalogue the [catalogue draw](published_populations.md) reads.

The convolution is cosmoRate's (Santoliquido et al. 2020, 2021): the Madau &
Fragos (2017) star-formation history and mean metallicity, a log-normal
metallicity spread of 0.2 dex, and Planck 2018. Each step is documented in
[the API reference](../api/synthesis/index.md).

## Black hole + neutron star mergers from Iorio et al. 2023

Iorio et al. 2023 published their SEVN runs on
[Zenodo 7794546](https://zenodo.org/records/7794546) (CC BY 4.0). The record is
one 3.8 GB archive, so gwmock-pop does not fetch it.

1. Download it and extract `BHNSm.csv` from
   `data_from_simulations/simulation_F/`, the fiducial model.
2. Pass that file's path to `read_iorio2023`, which checks it against its pinned
   SHA-256 before reading it.

```python
import numpy as np

from gwmock_pop.eos import tidal_deformability
from gwmock_pop.synthesis.catalogue import catalogue_columns, write_catalogue
from gwmock_pop.synthesis.convolution import draw_mergers, merger_rate_density
from gwmock_pop.synthesis.iorio2023 import black_hole_neutron_star_masses, read_iorio2023

run = read_iorio2023("BHNSm.csv", alpha=1.0)  # common-envelope efficiency 1
rate = merger_rate_density(run)
rng = np.random.default_rng(1)
mergers = draw_mergers(run, rate, rng)  # one year
black_hole, neutron_star = black_hole_neutron_star_masses(mergers)
spin = np.zeros_like(black_hole)  # the spins are the caller's choice
columns = catalogue_columns(
    mergers["redshift"],
    black_hole,
    neutron_star,
    spin,
    spin,
    rng=rng,
    lambda_1=np.zeros_like(black_hole),
    lambda_2=tidal_deformability(neutron_star, "PCP_BSK24_BSK24"),
)
write_catalogue("nsbh.h5", columns, {"input": "Iorio et al. 2023, BHNSm.csv, alpha = 1", "seed": 1})
```

These mergers come to a local rate of 25.5 Gpc⁻³ yr⁻¹ and 48,800 mergers a year
out to redshift 15. In the table, the neutron-star masses run from 1.10 to 1.66
solar masses and the black-hole masses from 5.7 to 47.5.

## Checked against a published population

Dupletsa et al. 2026 ([Zenodo 21682194](https://zenodo.org/records/21682194))
publish binary neutron stars built from Iorio et al.'s `NSNSm.csv` at the same
common-envelope efficiency (Loffredo et al. 2025). The record holds 351,205
mergers in a year, with a local rate of 107 Gpc⁻³ yr⁻¹.

From the same file, `gwmock_pop.synthesis` gives a redshift distribution within
0.005 of the record's in cumulative probability, so the shape matches.

The normalisation does not. The defaults give 435,560 mergers a year and 131.8
Gpc⁻³ yr⁻¹ at redshift 0.0075, 1.25 times the record. The defaults are the
corrections Iorio et al. state for their runs: an IMF correction of 0.255 and a
binary fraction of 0.5. The record's count follows from a product of 0.102
instead (for example, a binary fraction of 0.4), which neither paper states.
Pass `CosmicHistory(binary_fraction=0.4)` to reproduce the record's count.
