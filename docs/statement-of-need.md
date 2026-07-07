# Statement of need

Building the source population for a gravitational-wave mock data challenge
usually means bespoke sampling scripts that are hard to reproduce and hard to
reconfigure when priors change. `gwmock-pop` is the _forward_ counterpart to
population-inference tools such as `gwpopulation`: instead of inferring
hyper-parameters from observed events, it draws synthetic catalogues from
configurable priors. Its graph-driven sampler lets users declare arbitrary
parameter-dependency structures in YAML/TOML — validated without executing
arbitrary Python — and ships presets reflecting recent observed populations. It
is the population layer of the `gwmock` mock-data-challenge ecosystem, usable
standalone or through the `gwmock` orchestrator, and scales to the catalogue
sizes (of order 10⁵ sources per year) expected from next-generation detectors
such as the Einstein Telescope.
