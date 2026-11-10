# Tidal Deformabilities from an Equation of State

A neutron star's dimensionless tidal deformability, Λ, is fixed by its mass and
by the equation of state (EOS) of neutron-star matter. Drawing Λ from one named
EOS, instead of from a distribution, gives every neutron star in a population
the same microphysics, which an EOS study needs.

The `eos_tidal_deformability` transform gives each star the Λ of a named EOS at
its **source-frame** mass. It takes the detector-frame mass and the redshift and
converts them itself, `m / (1 + z)`, so a detector-frame mass cannot be passed
where the source-frame one is meant. The EOS and the stellar structure come from
LALSuite's `lalsimulation`, which names some 150 equations of state, among them
`SLY`, `APR4_EPP`, `PCP_BSK24_BSK24` (PCP(BSK24)) and `XMLSLZ_DDLZ1_BSK24`
(DD-LZ1). Every value is `lalsimulation`'s own, at that mass; nothing is
interpolated. It needs the `eos` extra:

```bash
uv pip install "gwmock-pop[eos]"
```

The extra needs LALSuite 8.0.0 or later. LALSuite publishes no wheels for
Windows.

Λ depends a little on the LALSuite release: 8.0.0 rebuilt the neutron-star
structure solution, and its values differ from 7.x by up to 0.2% (SLY) and
1.3×10⁻⁴ for PCP(BSK24) and DD-LZ1. A population is reproduced exactly with the
LALSuite version its run recorded.

## In a population

Replace a preset's deformabilities with the transform, here for binary neutron
stars with PCP(BSK24):

```python
from gwmock_pop import CBCSimulator


def from_eos(mass: str) -> dict:
    return {
        "transform": {
            "function": "eos_tidal_deformability",
            "arguments": {"detector_frame_mass": mass, "redshift": "@redshift", "eos": "PCP_BSK24_BSK24"},
        }
    }


simulator = CBCSimulator(
    source_type="bns",
    m_min=1.1,
    m_max=2.0,
    seed=3,
    parameters={"lambda_1": from_eos("@detector_frame_mass_1"), "lambda_2": from_eos("@detector_frame_mass_2")},
)
population = simulator.simulate(1000)
```

The same block works in a YAML graph config:

```yaml
lambda_1:
    transform:
        function: eos_tidal_deformability
        arguments:
            detector_frame_mass: '@detector_frame_mass_1'
            redshift: '@redshift'
            eos: PCP_BSK24_BSK24
```

For a neutron star–black hole binary, give the neutron star the transform and
keep the black hole's Λ at zero.

## Masses without a neutron star

An EOS has neutron stars only between a minimum and a maximum mass;
`gwmock_pop.eos.neutron_star_mass_range(eos)` gives them. A source-frame mass
outside `[minimum, maximum)`, or one that is not finite, has no Λ to give, so
the transform refuses it. Under `jax.jit` or `jax.vmap` a value cannot raise, so
such a mass is returned as NaN there instead.

## Checked against

The Λ of the binary neutron stars of Dupletsa et al. 2026
([arXiv:2607.27311](https://arxiv.org/abs/2607.27311), population on
[Zenodo](https://doi.org/10.5281/zenodo.21682194)) come from PCP(BSK24) at the
source-frame masses, computed by other software. The transform reproduces them
to within 0.4 %, and the tests keep five of them as a check.
