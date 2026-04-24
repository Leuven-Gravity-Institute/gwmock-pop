# Quick Start

Welcome to **gwmock-pop**. This package simulates populations of
gravitational-wave sources.

## Generate a Population with the CLI

The current minimal CLI workflow uses `GraphSimulator` with an explicit
`n_samples` run configuration.

Create a configuration file like this:

```yaml
run:
    name: demo_population
    mode: fixed_n_samples
    n_samples: 100
    seed: 42
    output:
        directory: outputs
        format: csv
        overwrite: true

parameters:
    mass_1:
        sampler:
            function: gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
            arguments:
                alpha_1: 1.72
                alpha_2: 4.51
                transition: 35.6
                minimum: 5.06
                maximum: 300.0
                mean_1: 9.76
                sigma_1: 0.649
                mean_2: 32.8
                sigma_2: 3.92
                taper_range: 4.32
                lambda_0: 0.361
                lambda_1: 0.586
```

Then run:

```bash
gwmock-pop simulate population.yaml
```

This writes `outputs/demo_population.csv`.

## Programmatic Usage

All simulators and compatible loaders implement the same protocol surface:

- `source_type` is a non-empty string
- `simulate(n_samples, **kwargs)` returns a mapping of 1-D arrays

```python
from gwmock_pop import CBCPriorSimulator

simulator = CBCPriorSimulator(source_type="bbh", seed=42)
population = simulator.simulate(5)
print(population["detector_frame_mass_1"].shape)  # (5,)
```

## Notes

- The CLI MVP currently supports `run.mode: fixed_n_samples`.
- `parameters` is passed directly to `GraphSimulator`.
- `FilePopulationLoader` supports CSV and HDF5 catalogues (including
  group-of-datasets HDF5 layouts).

## Next Steps

- [Request New Features](../CONTRIBUTING.md) - How to request new features or
  improvements.
- [API Reference](../api/index.md) - Programmatic usage documentation.
