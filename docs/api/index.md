---
title: API Reference
description: Generated reference pages for committed public modules.
icon: material/api
---

This section documents **gwmock-pop** with
[mkdocstrings](https://mkdocstrings.github.io/) for Python (via Zensical).
Source paths are under `src/gwmock_pop/`.

## Entry points

| Page                  | Scope                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [Package](package.md) | Symbols re-exported from `gwmock_pop` (`GWPopSimulator`, simulators, loaders, `validate_sample`, `list_presets`, …) |
| [Version](version.md) | `__version__`                                                                                                       |

## Command-line interface

| Page                         | Scope                                           |
| ---------------------------- | ----------------------------------------------- |
| [CLI overview](cli/index.md) | `gwmock_pop.cli` package                        |
| [Main](cli/main.md)          | Typer app, logging, `--version`                 |
| [Simulate](cli/simulate.md)  | `--config`, `--n`, `--output`, `--seed`         |
| [Convert](cli/convert.md)    | CSV ↔ HDF5 catalogue conversion, `--column-map` |
| [Validate](cli/validate.md)  | Graph config validation without sampling        |
| [Inspect](cli/inspect.md)    | Quick population summaries                      |
| [List](cli/list.md)          | Packaged presets and public simulator classes   |

## Protocols, loaders, validation

| Page                                            | Scope                                                  |
| ----------------------------------------------- | ------------------------------------------------------ |
| [Protocols overview](protocols/index.md)        | `gwmock_pop.protocols`                                 |
| [GWPopSimulator](protocols/simulator.md)        | Population simulator protocol                          |
| [ExternalPopulationLoader](protocols/loader.md) | File-backed loader protocol                            |
| [Loaders](loaders/index.md)                     | `FilePopulationLoader`, `read_population_catalogue`, … |
| [Population validation](validation.md)          | `validate_sample` and catalogue checks                 |
| [Graph validation](graph/validation.md)         | `validate_graph_config`, `ValidationReport`            |

## Configuration and presets

| Page                                      | Scope                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------- |
| [Configuration overview](config/index.md) | Pydantic run/population/output models                                 |
| [Presets](configs.md)                     | Packaged YAML/TOML presets (`list_presets`, `get_packaged_preset`, …) |

## Simulation stack

| Page                                         | Scope                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------- |
| [Simulators overview](simulators/index.md)   | `GraphSimulator`, `CBCPriorSimulator`, mixture, CBC priors, Poisson helper |
| [Graph simulator](simulators/graph.md)       | Config-driven `GraphSimulator`                                             |
| [Simulator base](simulators/simulator.md)    | `Simulator` ABC                                                            |
| [Binary black hole](simulators/bbh/index.md) | `BBHSimulator`                                                             |

## Graph, transforms, samplers, distributions

| Page                                    | Scope                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [Graph utilities](graph/index.md)       | Build, sampler/transform dependency extraction; see also [config validation](graph/validation.md) |
| [Graph build](graph/build.md)           | `build_dependency_graph`                                                                          |
| [Transforms](transforms/index.md)       | `luminosity_distance_to_redshift`, `multiply`, …                                                  |
| [Samplers](samplers/index.md)           | JAX sampling functions used in graph configs                                                      |
| [Distributions](distributions/index.md) | PDFs / CDFs used by samplers                                                                      |

## Other modules

| Section                             | Overview page                      |
| ----------------------------------- | ---------------------------------- |
| [Cosmology](cosmology/index.md)     | Flat ΛCDM distance helpers         |
| [Conversion](conversion/index.md)   | CBC parameter conversions          |
| [Parameters](parameters/index.md)   | Typed parameter records (e.g. BBH) |
| [RNG](rng/index.md)                 | `RNGManager`                       |
| [Integrators](integrators/index.md) | Quadrature helpers                 |
| [Utilities](utils/index.md)         | YAML, logging, dynamic imports     |

Use the **left navigation** for the full tree (including per-module detail pages
under each overview).
