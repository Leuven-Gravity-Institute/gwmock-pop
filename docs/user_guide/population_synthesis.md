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

## Choices and departures

Every setting below either follows its source, or departs from it for the reason
given.

| Setting                                 | Value                                                                         | Source, or why it departs                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Star formation and mean metallicity     | Madau & Fragos (2017)                                                         | As cosmoRate.                                                                                                                                                                                                                                                                                                                                                                                                  |
| Metallicity spread                      | 0.2 dex, mean shifted down by ln(10) σ²/2                                     | As cosmoRate. The shift keeps the mean metallicity on Madau & Fragos's fit, which is of the mass-weighted mean.                                                                                                                                                                                                                                                                                                |
| Solar metallicity                       | 0.019                                                                         | As cosmoRate. Madau & Fragos use the Anders & Grevesse (1989) scale.                                                                                                                                                                                                                                                                                                                                           |
| Binary fraction                         | 0.4                                                                           | As Iorio et al.'s published merger rates. Their text uses 0.5 for the merger efficiencies; their rate output, named by cosmoRate after the binary fraction, shows 0.4.                                                                                                                                                                                                                                         |
| IMF correction, Iorio et al.'s runs     | 0.25643                                                                       | Their own definition (footnote 13), computed exactly. They quote 0.255 ± 0.001 from a Monte Carlo; with a binary fraction of 0.4, the exact value reproduces their published rates and 0.255 does not.                                                                                                                                                                                                         |
| Run sampling against the IMF correction | Not corrected                                                                 | Iorio et al.'s run gives every primary above 5 M☉ a secondary above 2.2 M☉, where their correction counts only the binaries passing both cuts. Correcting this would raise the rates by 5.7 % (binary neutron stars) and 11.3 % (black hole + neutron star), but only under an assumption nothing external checks: that binaries with lighter secondaries make no mergers. The authors' normalisation is kept. |
| Redshift grid                           | 1001 edges, 0 to 15                                                           | **Departs** from cosmoRate's default of 100, which misplaces mergers around the peak of the rate by up to 1 % in cumulative probability. The result stops changing at about 750 edges.                                                                                                                                                                                                                         |
| Cosmology                               | Flat Planck 2018, matter density 0.3111, for both the lookback time and dt/dz | **Departs** from cosmoRate. It takes lookback times from astropy's Planck 2018, which includes neutrinos and radiation, but dt/dz from a formula that leaves them out. The two do not agree; here they cannot. The rate is 0.25 % below cosmoRate's.                                                                                                                                                           |
| A merger's redshift within its bin      | ∝ dV_c/dz / (1 + z)                                                           | **Departs** from cosmoRate's catalogues, which place every merger at its bin's centre.                                                                                                                                                                                                                                                                                                                         |
| A merger's formation redshift           | Its merger time plus its delay                                                | **Departs** from cosmoRate's catalogues, which give the formation bin's centre. That can fall after the merger when the merger is drawn within its bin.                                                                                                                                                                                                                                                        |

## Black hole + neutron star mergers from Iorio et al. 2023

Iorio et al. 2023 published their SEVN runs on
[Zenodo 7794546](https://zenodo.org/records/7794546) (CC BY 4.0). The record is
one 3.8 GB archive, so gwmock-pop does not fetch it.

1. Download it and extract `BHNSm.csv` from
   `data_from_simulations/simulation_F/`, the fiducial model.
2. Pass that file's path to `read_iorio2023`, which checks it against its pinned
   SHA-256 before reading it.

```python
import dataclasses

import numpy as np

from gwmock_pop.eos import tidal_deformability
from gwmock_pop.synthesis.catalogue import catalogue_columns, write_catalogue
from gwmock_pop.synthesis.convolution import CosmicHistory, draw_mergers, merger_rate_density
from gwmock_pop.synthesis.iorio2023 import black_hole_neutron_star_masses, read_iorio2023

run = read_iorio2023("BHNSm.csv", alpha=1.0)  # common-envelope efficiency 1
history = CosmicHistory()  # the settings in the table above
rate = merger_rate_density(run, history)
rng = np.random.default_rng(1)
mergers = draw_mergers(run, rate, rng, history=history)  # one year
black_hole, neutron_star = black_hole_neutron_star_masses(mergers)
# the tables carry no spins: here, aligned and uniform in (-0.05, 0.05)
spin_1z = rng.uniform(-0.05, 0.05, black_hole.size)
spin_2z = rng.uniform(-0.05, 0.05, black_hole.size)
columns = catalogue_columns(
    mergers["redshift"],
    black_hole,
    neutron_star,
    spin_1z,
    spin_2z,
    rng=rng,
    lambda_1=np.zeros_like(black_hole),
    lambda_2=tidal_deformability(neutron_star, "PCP_BSK24_BSK24"),
)
provenance = {
    "input": "Iorio et al. 2023, simulation_F/BHNSm.csv, alpha = 1",
    "imf_fraction": run.imf_fraction,
    "history": dataclasses.asdict(history),
    "seed": 1,
}
write_catalogue("nsbh.h5", columns, provenance)
```

The provenance carries every setting, so a catalogue records the choices it was
made with.

These mergers come to a local rate of 20.5 Gpc⁻³ yr⁻¹ and 39,200 mergers a year
out to redshift 15. In the table, the neutron-star masses run from 1.10 to 1.66
solar masses and the black-hole masses from 5.7 to 47.5.

## Checked against published results

**Iorio et al.'s own rates.** The cosmoRate output behind their Figs. 21 and 22
is in their paper's repository
([iorio22_plot](https://gitlab.com/iogiul/iorio22_plot), tag `v3`). Below
redshift 8, `gwmock_pop.synthesis` reproduces their binary neutron star and
black hole + neutron star curves at a median ratio of 0.997, and within 1.1 % in
every bin. The 0.3 % is the cosmology departure above.

**Dupletsa et al. 2026**
([Zenodo 21682194](https://zenodo.org/records/21682194)) publish binary neutron
stars built from Iorio et al.'s `NSNSm.csv` at the same common-envelope
efficiency (Loffredo et al. 2025). The record holds 351,205 mergers in a year,
with a local rate of 107 Gpc⁻³ yr⁻¹. From the same file, `gwmock_pop.synthesis`
gives:

- 350,400 mergers a year, 0.23 % fewer;
- 106.0 Gpc⁻³ yr⁻¹ at redshift 0.0075;
- a redshift distribution within 0.005 of the record's in cumulative
  probability.
