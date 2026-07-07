# gwmock-pop

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwmock-pop/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock-pop/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwmock-pop/)
[![codecov](https://codecov.io/gh/leuven-gravity-institute/gwmock-pop/graph/badge.svg?token=Vwf7NYTHCm)](https://codecov.io/gh/leuven-gravity-institute/gwmock-pop)
[![PyPI Version](https://img.shields.io/pypi/v/gwmock-pop)](https://pypi.org/project/gwmock-pop/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwmock-pop)](https://pypi.org/project/gwmock-pop/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/1147941311.svg)](https://doi.org/10.5281/zenodo.18574076)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

`gwmock-pop` is a Python package for simulating populations of
gravitational-wave sources.

## Current package surface

- **Protocols:** `GWPopSimulator` (population simulators),
  `ExternalPopulationLoader` (catalogue loaders).
- **Graph-driven sampling:** `GraphSimulator` from a YAML/TOML `parameters`
  graph (packaged **presets** via `list_presets()` /
  `GraphSimulator.from_preset`).
- **Graph-backed CBC simulators:** `CBCSimulator`, `BBHSimulator`,
  `BNSSimulator`, and `NSBHSimulator` — configurable priors built on the graph
  engine; override any parameter's distribution via the `parameters=` argument.
- **Composition:** `MixtureSimulator`, `PoissonEventSampler` (event-count helper
  used in mixture workflows).
- **Catalogues:** `FilePopulationLoader` and `read_population_catalogue` /
  `write_population_catalogue` for CSV and HDF5 (structured or group-of-datasets
  layouts), including remote URL loading with local caching and CBC
  canonicalization in the loader.
- **Quality checks:** `validate_sample` for arrays returned by simulators.

Public re-exports live in `gwmock_pop.__all__`; full module reference is in the
**[API docs](docs/api/index.md)**.

## Requirements

- Python `>=3.12` (tested on 3.12–3.14)
- Linux, macOS, or Windows

## Floating-point precision

Importing `gwmock_pop` enables 64-bit JAX floats (`jax_enable_x64`). GPS-scale
parameters such as `coa_time` (~1.6 × 10⁹ s) are unusable in JAX's 32-bit
default, where the float32 spacing at that magnitude is 128 s. The flag is
global JAX state, so it also affects arrays your own code creates after the
import. To keep JAX's 32-bit default (e.g. for GPU-throughput studies that do
not sample absolute times), set `GWMOCK_POP_DISABLE_X64=1` in the environment
before importing the package.

## Installation

Install from PyPI:

```bash
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install gwmock-pop
```

Install from source:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-pop.git
cd gwmock-pop
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync --no-dev
```

Developer setup:

```bash
uv sync --group dev
uv run prek install
```

Docs setup:

```bash
uv sync --group docs
uv run zensical serve
```

## Getting started (CLI)

The `gwmock-pop` CLI uses **Typer**. Typical flow: pick a **packaged preset** or
a **graph config file**, set the sample count, and write a CSV or HDF5
catalogue.

```bash
# Packaged preset (see `gwmock-pop list` for names)
gwmock-pop simulate --config gwtc4 --n 1000 --output population.csv --seed 42

# Or a graph YAML/TOML (top-level `parameters:` as in `examples/gwtc4/bbh_population.yaml`)
gwmock-pop simulate --config examples/gwtc4/bbh_population.yaml --n 500 --output out.h5
```

Other commands:

| Command               | Purpose                                                                          |
| --------------------- | -------------------------------------------------------------------------------- |
| `gwmock-pop convert`  | Convert population files between CSV and HDF5; optional `--column-map` JSON/YAML |
| `gwmock-pop validate` | Check a graph config without sampling                                            |
| `gwmock-pop inspect`  | Summary statistics for a population file                                         |
| `gwmock-pop list`     | List presets and public simulator classes                                        |

```bash
gwmock-pop --help
gwmock-pop simulate --help
```

## Getting started (library)

```python
from gwmock_pop import CBCSimulator

sim = CBCSimulator(seed=42)
population = sim.simulate(100)
assert population["detector_frame_mass_1"].shape == (100,)
```

Use `GraphSimulator.from_config_file(...)` or `GraphSimulator.from_preset(...)`
for full graph configs (see `examples/` and `gwmock_pop.simulators.graph`).

Sample from a packaged preset (run `gwmock-pop list` or `list_presets()` for the
available names):

```python
from gwmock_pop import GraphSimulator, list_presets

print(list_presets())                 # e.g. ["gwtc4", ...]
sim = GraphSimulator.from_preset("gwtc4")
catalogue = sim.simulate(1000)
```

`FilePopulationLoader` also accepts `http://`, `https://`, `s3://`, and
`zenodo://<record>/<file>` sources. Remote catalogues are cached under
`${GWMOCK_POP_CACHE_DIR}` or `${XDG_CACHE_HOME:-~/.cache}/gwmock-pop`, and CBC
catalogues are validated and rewritten to canonical gwmock-pop parameter names
before sampling.

## Verification

```bash
gwmock-pop --help
python -c "import gwmock_pop; print(gwmock_pop.__version__)"
```

## Testing

Default test run excludes `integration`-marked tests:

```bash
uv run pytest
```

Run integration tests explicitly:

```bash
uv run pytest -m integration
```

## Documentation

- **Site:**
  [https://leuven-gravity-institute.github.io/gwmock-pop/](https://leuven-gravity-institute.github.io/gwmock-pop/)
- **API index (tables + navigation):** [docs/api/index.md](docs/api/index.md)
- **User guide:**
  [docs/user_guide/installation.md](docs/user_guide/installation.md),
  [docs/user_guide/quick_start.md](docs/user_guide/quick_start.md)

## License

BSD 3-Clause, see [LICENSE](LICENSE).
