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
- **`--n`**: number of samples (events). Optional when the configuration sets
  `run.n_samples`.
- **`--output`**: destination `.csv`, `.h5`, or `.hdf5` (must not already exist,
  unless the configuration sets `run.output.overwrite`).
- **`--seed`**: optional integer for reproducibility. Falls back to a configured
  `run.seed`, and failing that a seed is drawn and recorded.

## Provenance

Every catalogue `simulate` writes carries a machine-readable record of the run
behind it, so the catalogue can be reconstructed later without the config file
that produced it:

- the package version, and the commit and dirty flag when running from a
  checkout;
- the **complete resolved** configuration — every sampler argument, including
  the defaults the config file never wrote down;
- the RNG seed **as actually used**, so a run started without `--seed` is still
  repeatable;
- the source type, the row count and the column names in output order;
- the creation time and the code that wrote the file.

HDF5 files carry the record in a `metadata` group beside `data`; CSV files get a
`<name>.csv.provenance.json` sidecar. Prefer HDF5 for anything you publish: a
sidecar is a separate file, and a CSV catalogue that arrives without it arrives
without its provenance.

```python
from gwmock_pop.provenance import read_provenance, replay_catalogue

record = read_provenance("outputs/population.h5")
print(record["run"]["seed"], record["tool"]["version"])

# Redraw the same catalogue from the record alone.
catalogue = replay_catalogue(record)
```

Set `run.output.save_metadata` to `false` to write the samples on their own.

## Configuration file

A configuration file may carry the `run` and `output` blocks alongside its
`parameters` graph, and `simulate` honours them:

```yaml
run:
    name: gwtc4-rerun
    seed: 1234
    n_samples: 5000
    output:
        overwrite: true
        compression: gzip
        save_metadata: true
parameters:
    # ... the parameter graph
```

Command-line options win over the file. `run.seed`, `run.n_samples`,
`output.directory`, `output.format` and `output.compression` apply only when the
file states them, because each of their schema defaults would otherwise override
what `--output` on its own asks for; a declared `output.format` that contradicts
the `--output` suffix is an error rather than a silent choice between them.

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
