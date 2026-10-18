# Installation

We recommend using `uv` to manage virtual environments for `gwmock-pop`.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

## Requirements

- Python 3.13 or higher
- Operating System: Linux, macOS, or Windows

<!-- prettier-ignore-start -->
!!!note
    The package is built and tested against Python 3.13-3.15. When creating a virtual environment with `uv`,
    specify the Python version to ensure compatibility:
    `uv venv --python 3.13` (replace `3.13` with your preferred version in the 3.13-3.15 range).
    On Python 3.15, h5py publishes no wheel yet, so the `hdf5` extra may not install there. HDF5 catalogues
    still work: without h5py they are written and read by gwmock-data's own HDF5 support.

<!-- prettier-ignore-end -->

## Install from PyPI

The recommended way to install `gwmock-pop` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock-pop
```

HDF5 catalogues (`.h5`, `.hdf5`), including the published catalogues the
`catalogue` commands draw from, work with the core install. The `hdf5` extra
adds `h5py`, which is faster and compresses what it writes:

```bash
uv pip install "gwmock-pop[hdf5]"
```

## Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-pop.git
cd gwmock-pop
# Create a virtual environment (recommended with uv)
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --no-dev
```

### Development Installation

To set up for development:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-pop.git
cd gwmock-pop

# Create a virtual environment (recommended with uv)
uv venv --python 3.13
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

Core dependencies are defined in `pyproject.toml` (`jax`, `typer`, `networkx`,
`PyYAML`, `ruyaml`, `pydantic`, `gwmock-data`). `h5py` is the optional `hdf5`
extra, a faster HDF5 backend.

## Getting Help

<!-- prettier-ignore-start -->

1. Check the [troubleshooting guide](../troubleshooting/index.md)
2. Search existing [issues](https://github.com/Leuven-Gravity-Institute/gwmock-pop/issues)
3. Create a new issue with:
    - Your operating system and Python version
    - Full error message
    - Steps to reproduce the problem

<!-- prettier-ignore-end -->
