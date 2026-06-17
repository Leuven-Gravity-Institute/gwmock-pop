# Quick Start

Welcome to **gwmock-pop**. This package simulates populations of
gravitational-wave sources using JAX and a graph-based configuration model.

## Simulate with the CLI

1. Install the package (see [Installation](installation.md)).
2. Run **`gwmock-pop simulate`** with a **preset name** or a path to a
   **YAML/TOML** file whose top level defines a `parameters` mapping (same shape
   as `GraphSimulator` expects).

Examples:

```bash
# Packaged BBH-style preset (names from `gwmock-pop list`)
gwmock-pop simulate --config gwtc4 --n 1000 --output outputs/population.csv --seed 42

# Full graph from the repository examples
gwmock-pop simulate --config examples/gwtc4/bbh_population.yaml --n 500 --output outputs/population.h5
```

Options:

- **`--config`**: preset identifier or path to `.yaml` / `.yml` / `.toml`.
- **`--n`**: number of samples (events).
- **`--output`**: destination `.csv`, `.h5`, or `.hdf5` (must not already
  exist).
- **`--seed`**: optional integer for reproducibility.

Other useful commands:

- **`gwmock-pop validate --config <file>`** — static validation of the graph
  config (no JAX sampling).
- **`gwmock-pop convert`** — convert between CSV and HDF5 with an optional
  column map.
- **`gwmock-pop inspect`** — summary statistics for a population file.
- **`gwmock-pop list`** — presets and exported simulator classes.

## Programmatic usage

Simulators exposed from `gwmock_pop` implement **`GWPopSimulator`**: a non-empty
`source_type`, stable `parameter_names`, and `simulate(n_samples, **kwargs)`
returning a mapping of **1-D** `jax.Array` columns of length `n_samples`.

```python
from gwmock_pop import CBCSimulator

simulator = CBCSimulator(seed=42)
population = simulator.simulate(5)
print(population["detector_frame_mass_1"].shape)  # (5,)

# Override the distribution of any single parameter via the graph config:
simulator = CBCSimulator(
    seed=42,
    parameters={
        "detector_frame_mass_1": {
            "sampler": {
                "function": "power_law_plus_peak",
                "arguments": {
                    "alpha": 3.5,
                    "minimum": 5.0,
                    "maximum": 100.0,
                    "lambda_peak": 0.1,
                    "peak_mean": 35.0,
                    "peak_sigma": 5.0,
                    "peak_maximum": 100.0,
                },
            }
        }
    },
)
```

To sample **source redshifts** from the Madau-Dickinson rate-weighted
distribution instead of drawing luminosity distance uniformly in comoving
volume, sample `redshift` directly and derive `distance` with
[`redshift_to_luminosity_distance`](../api/transforms/index.md):

```python
simulator = CBCSimulator(
    seed=42,
    parameters={
        "redshift": {
            "sampler": {
                "function": "madau_dickinson_redshift",
                "arguments": {
                    "z_max": 3.0,
                },
            }
        },
        "distance": {
            "transform": {
                "function": "redshift_to_luminosity_distance",
                "arguments": {
                    "redshift": "@redshift",
                },
            }
        },
    },
)
```

The same `sampler` / `transform` blocks work in YAML graph configs. See
[Madau-Dickinson](../api/distributions/madau_dickinson.md) and
[Madau-Dickinson Redshift](../api/samplers/madau_dickinson_redshift.md) for
parameters and defaults.

For graph-based populations:

```python
from pathlib import Path

from gwmock_pop import GraphSimulator

sim = GraphSimulator.from_config_file(Path("examples/gwtc4/bbh_population.yaml"), source_type="bbh", seed=0)
population = sim.simulate(10)
```

Preset-driven construction:

```python
from gwmock_pop import GraphSimulator

sim = GraphSimulator.from_preset("gwtc4", seed=0)
```

## Notes

- Graph **parameter names** in YAML should match the keys your downstream stack
  expects (see examples under `examples/`).
- **`FilePopulationLoader`** and **`read_population_catalogue`** support CSV and
  HDF5 (structured `data` dataset or group-of-datasets layouts).
- **`validate_sample`** can check a simulated batch against the protocol shape
  contract.

## Next steps

- [Contributing](../contributing.md) — how to propose changes.
- [API reference](../api/index.md) — organised index of all reference pages.
- [Troubleshooting](../troubleshooting/index.md) — common issues.
