# gwsim_pop

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwsim_pop/actions/workflows/CI.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwsim_pop/actions/workflows/CI.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwsim_pop/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwsim_pop/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwsim_pop/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwsim_pop/)
[![codecov](https://codecov.io/gh/leuven-gravity-institute/gwsim_pop/graph/badge.svg?token=Vwf7NYTHCm)](https://codecov.io/gh/leuven-gravity-institute/gwsim_pop)
[![PyPI Version](https://img.shields.io/pypi/v/gwsim_pop)](https://pypi.org/project/gwsim_pop/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwsim_pop)](https://pypi.org/project/gwsim_pop/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DOI](https://zenodo.org/badge/924023559.svg)](https://doi.org/10.5281/zenodo.18017404)

A Python package for simulating populations of gravitational-wave sources.

## Installation

We recommend using `uv` to manage virtual environments for installing
`gwsim_pop`.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

### Requirements

- Python 3.10 or higher
- Operating System: Linux, macOS, or Windows

**Note:** The package is built and tested against Python 3.10-3.12. When
creating a virtual environment with `uv`, specify the Python version to ensure
compatibility: `uv venv --python 3.10` (replace `3.10` with your preferred
version in the 3.10-3.12 range). This avoids potential issues with unsupported
Python versions.

### Install from PyPI

The recommended way to install `gwsim_pop` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwsim_pop
```

#### Optional Dependencies

For development or specific features:

```bash
# Development dependencies (testing, linting, etc.)
uv pip install gwsim_pop[dev]

# Documentation dependencies
uv pip install gwsim_pop[docs]

# All dependencies
uv pip install gwsim_pop[dev,docs]
```

### Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwsim_pop.git
cd gwsim_pop
# Create a virtual environment (recommended with uv)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .
```

#### Development Installation

To set up for development:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwsim_pop.git
cd gwsim_pop

# Create a virtual environment (recommended with uv)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install ".[dev]"

# Install the commitlint dependencies
npm install

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### Verify Installation

Check that `gwsim_pop` is installed correctly:

```bash
gwsim_pop --help
```

```bash
python -c "import gwsim_pop; print(gwsim_pop.__version__)"
```

## Documentation

Full documentation to be available at
[https://leuven-gravity-institute.github.io/gwsim_pop](https://leuven-gravity-institute.github.io/gwsim_pop).

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
[GitHub Releases page](https://github.com/Leuven-Gravity-Institute/gwsim_pop/releases).

## Testing

Run the test suite:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/Leuven-Gravity-Institute/gwsim_pop/issues/new) or
contact the maintainers.
