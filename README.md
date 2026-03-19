# gwsim-pop – This Package Has Moved

[![Status: Moved](https://img.shields.io/badge/status-moved-critical)](https://pypi.org/project/gwmock-pop/)
[![PyPI - gwmock-pop](https://img.shields.io/badge/pypi-gwmock--pop-blue)](https://pypi.org/project/gwmock-pop/)
[![PyPI Version](https://img.shields.io/pypi/v/gwsim-pop)](https://pypi.org/project/gwsim-pop/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwsim-pop)](https://pypi.org/project/gwsim-pop/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1147941311.svg)](https://doi.org/10.5281/zenodo.18574076)

**gwsim-pop** is now **[gwmock-pop](https://pypi.org/project/gwmock-pop/)**.

## Why?

The renaming is intended to avoid confusion with another Python package,
[GWSim](https://git.ligo.org/benoit.revenu/gwsim), which is designed for
creating mock GW samples for different astrophysical populations and
cosmological models of binary black holes.

## How to Upgrade

```bash
pip uninstall gwsim-pop
pip install gwmock-pop
```

## Migration Note

Installing this version of `gwsim-pop` will automatically install `gwmock-pop`
as a dependency. Your existing imports will continue to work via a wrapper, but
you should update them to import `gwmock_pop` as soon as possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/Leuven-Gravity-Institute/gwmock-pop/issues/new) or
contact the maintainers.
