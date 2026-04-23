# gwmock-pop

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/CI.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/CI.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwmock-pop/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock-pop/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwmock-pop/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwmock-pop/)
[![codecov](https://codecov.io/gh/leuven-gravity-institute/gwmock-pop/graph/badge.svg?token=Vwf7NYTHCm)](https://codecov.io/gh/leuven-gravity-institute/gwmock-pop)
[![PyPI Version](https://img.shields.io/pypi/v/gwmock-pop)](https://pypi.org/project/gwmock-pop/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwmock-pop)](https://pypi.org/project/gwmock-pop/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DOI](https://zenodo.org/badge/1147941311.svg)](https://doi.org/10.5281/zenodo.18574076)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

A Python package for simulating populations of gravitational-wave sources.

## Installation

We recommend using `uv` to manage virtual environments for installing
`gwmock-pop`.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

### Requirements

- Python 3.12 or higher
- Operating System: Linux, macOS, or Windows

**Note:** The package is built and tested against Python 3.12-3.14. When
creating a virtual environment with `uv`, specify the Python version to ensure
compatibility: `uv venv --python 3.12` (replace `3.12` with your preferred
version in the 3.12-3.14 range). This avoids potential issues with unsupported
Python versions.

### Install from PyPI

The recommended way to install `gwmock-pop` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock-pop
```

### Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-pop.git
cd gwmock-pop
# Create a virtual environment (recommended with uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

#### Development Installation

To set up for development:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-pop.git
cd gwmock-pop

# Create a virtual environment (recommended with uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# For development
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

To build the documentation locally:

```bash
# For building the documentation locally
uv sync --group docs

# Start the documentation server
uv run zensical serve
```

### Verify Installation

Check that `gwmock-pop` is installed correctly:

```bash
gwmock-pop --help
```

```bash
python -c "import gwmock-pop; print(gwmock-pop.__version__)"
```

## Documentation

Full documentation to be available at
[https://leuven-gravity-institute.github.io/gwmock-pop](https://leuven-gravity-institute.github.io/gwmock-pop).

## Quick Start with the CLI

The current MVP CLI supports fixed-size population generation through
`GraphSimulator`.

Create a configuration file such as `population.yaml`:

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
            function: gwmock-pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
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

Run the simulator:

```bash
gwmock-pop simulate population.yaml
```

This writes `outputs/demo_population.csv`. For the MVP, the CLI only supports
`run.mode: fixed_n_samples`.

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Release Schedule

Releases follow a fixed schedule: every Tuesday at 00:00 UTC, unless an emergent
bugfix is required. This ensures predictable updates while allowing flexibility
for critical issues. Users can view upcoming changes in the draft release on the
[GitHub Releases page](https://github.com/Leuven-Gravity-Institute/gwmock-pop/releases).

## Testing

Run the test suite:

```bash
pytest
```

## License

This project is licensed under the BSD 3-Clause License - see the
[LICENSE](LICENSE) file for details.

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/Leuven-Gravity-Institute/gwmock-pop/issues/new) or
contact the maintainers.
