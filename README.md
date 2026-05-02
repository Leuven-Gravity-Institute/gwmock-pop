# gwmock-pop

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwmock-pop/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock-pop/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwmock-pop/)
[![codecov](https://codecov.io/gh/leuven-gravity-institute/gwmock-pop/graph/badge.svg?token=Vwf7NYTHCm)](https://codecov.io/gh/leuven-gravity-institute/gwmock-pop)
[![PyPI Version](https://img.shields.io/pypi/v/gwmock-pop)](https://pypi.org/project/gwmock-pop/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwmock-pop)](https://pypi.org/project/gwmock-pop/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DOI](https://zenodo.org/badge/1147941311.svg)](https://doi.org/10.5281/zenodo.18574076)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

`gwmock-pop` is a Python package for simulating populations of
gravitational-wave sources.

## Current Package Surface

- Protocol-first simulator interface via `GWPopSimulator`:
    - `source_type: str` (non-empty routing key)
    - `simulate(n_samples, **kwargs) -> Mapping[str, jax.Array]`
    - each returned parameter array is 1-D with length `n_samples`
- Core simulator implementations:
    - `GraphSimulator` (config-driven dependency graph)
    - `CBCPriorSimulator` (lightweight analytic CBC priors)
- External catalogue loader:
    - `FilePopulationLoader` for CSV/HDF5
    - supports structured HDF5 datasets and group-of-datasets layouts

## Requirements

- Python `>=3.12` (tested on 3.12-3.14)
- Linux, macOS, or Windows

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
uv run pre-commit install
```

Docs setup:

```bash
uv sync --group docs
uv run zensical serve
```

## Quick Start (CLI)

Create `population.yaml`:

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

Run:

```bash
gwmock-pop simulate population.yaml
```

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

Run integration/smoke tests explicitly:

```bash
uv run pytest -m integration
```

## Documentation

- Docs home:
  [https://leuven-gravity-institute.github.io/gwmock-pop/](https://leuven-gravity-institute.github.io/gwmock-pop/)
- API reference: [docs/api/index.md](docs/api/index.md)

## License

BSD 3-Clause, see [LICENSE](LICENSE).
