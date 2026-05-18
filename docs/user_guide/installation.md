# Installation

We recommend using `uv` to manage virtual environments for `gwmock-pop`.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

## Requirements

- Python 3.12 or higher
- Operating System: Linux, macOS, or Windows

<!-- prettier-ignore-start -->
!!!note
    The package is built and tested against Python 3.12-3.14. When creating a virtual environment with `uv`,
    specify the Python version to ensure compatibility:
    `uv venv --python 3.12` (replace `3.12` with your preferred version in the 3.12-3.14 range).
    This avoids potential issues with unsupported Python versions.

<!-- prettier-ignore-end -->

## Install from PyPI

The recommended way to install `gwmock-pop` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock-pop
```

## Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-pop.git
cd gwmock-pop
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --no-dev
```

### Development Installation

To set up for development:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-pop.git
cd gwmock-pop

# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --group dev

# Install prek hooks
uv run prek install
```

## Verify Installation

Check that `gwmock-pop` is installed correctly:

```bash
gwmock-pop --help
```

```bash
python -c "import gwmock_pop; print(gwmock_pop.__version__)"
```

## Dependencies

Core dependencies are defined in `pyproject.toml` (`jax`, `typer`, `h5py`,
`networkx`, `PyYAML`, `ruyaml`, `pydantic`).

## Getting Help

<!-- prettier-ignore-start -->

1. Check the [troubleshooting guide](../troubleshooting/index.md)
2. Search existing [issues](https://github.com/Leuven-Gravity-Institute/gwmock-pop/issues)
3. Create a new issue with:
    - Your operating system and Python version
    - Full error message
    - Steps to reproduce the problem

<!-- prettier-ignore-end -->
