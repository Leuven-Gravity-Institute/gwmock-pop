# Published Populations

gwmock-pop pins three published population products by digest, and draws from
any of them with the same seeded draw: a Poisson count per class over the span,
rows sampled without replacement, and merger times uniform in the span.

| Product                                                                                                                                 | Entries                   | Classes                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------- |
| Einstein Telescope design-comparison catalogues (ET document 18321)                                                                     | `COBA_CATALOGUES`         | binary neutron stars, binary black holes                                                              |
| [Dupletsa et al. 2026](https://arxiv.org/abs/2607.27311), [Zenodo 21682194](https://zenodo.org/records/21682194) (CC BY 4.0)            | `DUPLETSA2026_CATALOGUES` | binary neutron stars, Pop I/II binary black holes, Pop III binary black holes, primordial black holes |
| Black hole + neutron star mergers from Iorio et al. 2023's SEVN run, [Zenodo 24127546](https://zenodo.org/records/24127546) (CC BY 4.0) | `IORIO2023_NSBH`          | black hole + neutron star mergers                                                                     |

```python
from gwmock_pop.catalogue import DUPLETSA2026_CATALOGUES, IORIO2023_NSBH, draw_catalogue_population

draw = draw_catalogue_population(
    seed=1,
    span_seconds=30 * 86400,
    catalogues=(*DUPLETSA2026_CATALOGUES, IORIO2023_NSBH),
)
draw.parameters["detector_frame_mass_1"], draw.population_class
```

Dupletsa et al.'s binary neutron stars and the black hole + neutron star mergers
come from the same SEVN run (Iorio et al. 2023, fiducial model, common-envelope
efficiency 1), turned into mergers by the same method, so the two classes share
their normalisation.

## The Dupletsa et al. Zenodo record

The record publishes its four population files in one archive, `injections.zip`.
gwmock-pop downloads the archive once, checks it against its pinned SHA-256, and
extracts each file beside it, checking it against the file's own pinned SHA-256.

The files use two layouts, which gwmock-pop reads into its own names and units
(see [Catalogue Layouts](../api/catalogue/layouts.md)):

- `BNSs.hdf5` and `BBHs_from_POP3.hdf5` are pandas frames in gwfish's names.
  Neither pandas nor h5py is needed to read them.
- `BBHs.h5` and `PBHs.h5` are flat datasets in gwfast's names, with distances in
  Gpc.

What the reading decides:

- **Masses**: the binary-neutron-star file gives the detector-frame chirp mass
  and the mass ratio; the source-frame masses are derived from them and the
  redshift.
- **Spins**: binary neutron stars carry aligned, signed spins. The binary black
  holes of `BBHs.h5` carry Cartesian components, read as stored. The Pop III and
  primordial black holes give spins as angles, converted to components at 2 Hz
  with detector-frame masses, as LALSimulation converts them. 2 Hz is the
  frequency `BBHs.h5`'s own components were computed at; the record does not
  state one for the other files.
- **Rates**: every file is one year of mergers, as the paper's Table 2 counts
  them. The coalescence times inside the files are not used. `PBHs.h5` spreads
  its rows over ten years, but the paper and its source give its 1.92 million
  mergers per year.
- **Tides**: binary neutron stars carry the record's tidal deformabilities, from
  the PCP(BSK24) equation of state. To use another equation of state, recompute
  them from the masses (see
  [Tidal Deformabilities from an Equation of State](equation_of_state.md)).

## Black hole + neutron star mergers

[Zenodo 24127546](https://zenodo.org/records/24127546) holds one year of black
hole + neutron star mergers, 39,027 of them out to redshift 15. They were made
by [Population Synthesis](population_synthesis.md) from the merging black hole +
neutron star binaries of Iorio et al. (2023). The file is in the draw's own
columns and is read as is. Its spins are aligned, uniform in (−0.05, 0.05). The
black holes' tidal deformability is zero, and the neutron stars' is from
PCP(BSK24). The record's README documents every step and the checks run on it,
and the file carries its provenance.
