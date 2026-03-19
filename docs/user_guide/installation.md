# Installation

We recommend using `uv` to manage virtual environments for installing
`gwmock_pop`.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

## Requirements

- Python 3.10 or higher
- Operating System: Linux, macOS, or Windows

<!-- prettier-ignore-start -->
!!!note
    The package is built and tested against Python 3.10-3.12. When creating a virtual environment with `uv`,
    specify the Python version to ensure compatibility:
    `uv venv --python 3.10` (replace `3.10` with your preferred version in the 3.10-3.12 range).
    This avoids potential issues with unsupported Python versions.

<!-- prettier-ignore-end -->

## Install from PyPI

The recommended way to install `gwmock_pop` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock_pop
```

### Optional Dependencies

For development or specific features:

```bash
# Development dependencies (testing, linting, etc.)
uv pip install gwmock_pop[dev]

# Documentation dependencies
uv pip install gwmock_pop[docs]

# All dependencies
uv pip install gwmock_pop[dev,docs]
```

## Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock_pop.git
cd gwmock_pop
# Create a virtual environment (recommended with uv)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .
```

### Development Installation

To set up for development:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock_pop.git
cd gwmock_pop

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

## Verify Installation

Check that `gwmock_pop` is installed correctly:

```bash
gwmock_pop --help
```

```bash
python -c "import gwmock_pop; print(gwmock_pop.__version__)"
```

## Dependencies

### Core Dependencies

- **typer**: CLI framework

## Getting Help

<!-- prettier-ignore-start -->

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing [issues](https://github.com/Leuven-Gravity-Institute/gwmock_pop/issues)
3. Create a new issue with:
    - Your operating system and Python version
    - Full error message
    - Steps to reproduce the problem

<!-- prettier-ignore-end -->
