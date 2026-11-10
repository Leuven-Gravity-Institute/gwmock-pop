"""Tidal deformabilities from a named equation of state."""

from __future__ import annotations

import builtins

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("lalsimulation", reason="the eos extra (lalsuite) is not installed")

from gwmock_pop.eos import neutron_star_mass_range, tidal_deformability
from gwmock_pop.transforms import eos_tidal_deformability

pytestmark = pytest.mark.unit

#: Five binary neutron stars of the population of Dupletsa et al. 2026 (arXiv:2607.27311), whose
#: deformabilities come from the PCP(BSK24) equation of state at the source-frame masses: detector-frame
#: chirp mass, mass ratio, redshift, Lambda_1, Lambda_2. From BNSs.hdf5 in Zenodo record 21682194
#: (doi:10.5281/zenodo.21682194, CC BY 4.0), picked across the range of source-frame masses. Their values
#: were computed by other software than lalsimulation, from the same equation of state, so they anchor the
#: equation-of-state name, the source-frame conversion and the formula for Lambda together.
_DUPLETSA_BNS = [
    (2.442763277577575, 0.9531097551231434, 1.74115, 2564.0643711372895, 3282.3423715477843),
    (3.4080556083559577, 0.9161625190404505, 2.07595, 701.1008942993655, 1148.0392642811794),
    (3.5418839313240973, 0.9963733128138633, 1.95565, 566.7508006746007, 578.9068990262921),
    (7.90422813282411, 0.9200605112159643, 5.62215, 456.2623204540012, 746.3760717371855),
    (4.713447856235451, 0.782682607066374, 2.53005, 127.94117364914284, 617.203447787053),
]


@pytest.mark.parametrize(("chirp_mass", "mass_ratio", "redshift", "lambda_1", "lambda_2"), _DUPLETSA_BNS)
def test_deformabilities_match_a_published_population_drawn_from_the_same_equation_of_state(
    chirp_mass: float, mass_ratio: float, redshift: float, lambda_1: float, lambda_2: float
) -> None:
    """Within 0.5 %: lalsimulation's values sit 0.1 to 0.4 % below the catalogue's, the other software's."""
    mass_1 = chirp_mass * (1 + mass_ratio) ** 0.2 / mass_ratio**0.6
    masses = jnp.array([mass_1, mass_ratio * mass_1])
    values = eos_tidal_deformability(masses, jnp.full(2, redshift), "PCP_BSK24_BSK24")
    np.testing.assert_allclose(np.asarray(values), [lambda_1, lambda_2], rtol=5e-3, atol=0.0)


def test_the_equation_of_state_is_evaluated_at_the_source_frame_mass() -> None:
    """A star seen at redshift z with detector-frame mass m is the star of mass m / (1 + z) at rest."""
    detector_frame = jnp.array([1.6, 2.2, 2.9])
    redshift = jnp.array([0.2, 0.5, 1.2])
    shifted = eos_tidal_deformability(detector_frame, redshift, "SLY")
    at_rest = eos_tidal_deformability(detector_frame / (1 + redshift), jnp.zeros(3), "SLY")
    np.testing.assert_array_equal(np.asarray(shifted), np.asarray(at_rest))


def test_a_heavier_star_is_less_deformable() -> None:
    """Lambda falls steeply with mass over the whole stable branch."""
    low, high = neutron_star_mass_range("APR4_EPP")
    masses = np.linspace(low, high, 200, endpoint=False)
    values = tidal_deformability(masses, "APR4_EPP")
    assert np.all(np.diff(values) < 0)


def test_a_mass_without_a_neutron_star_is_refused() -> None:
    """Above the maximum mass, below the minimum, or not finite: there is no deformability to give."""
    low, high = neutron_star_mass_range("SLY")
    for mass in (high, high + 0.1, low - 0.01, np.nan):
        with pytest.raises(ValueError, match="outside it"):
            eos_tidal_deformability(jnp.array([1.4, mass]), jnp.zeros(2), "SLY")


def test_a_traced_mass_without_a_neutron_star_is_nan() -> None:
    """Under jit a value cannot raise, so it is NaN, and the others are the eager values."""
    _, high = neutron_star_mass_range("SLY")
    masses = jnp.array([1.4, high + 0.1, 1.2])
    traced = jax.jit(lambda m: eos_tidal_deformability(m, jnp.zeros(3), "SLY"))(masses)
    eager = eos_tidal_deformability(masses[jnp.array([0, 2])], jnp.zeros(2), "SLY")
    assert np.isnan(np.asarray(traced)[1])
    np.testing.assert_array_equal(np.asarray(traced)[[0, 2]], np.asarray(eager))


def test_a_vmapped_call_matches_the_eager_one_and_is_nan_without_a_neutron_star() -> None:
    """Under vmap too, whether each call takes one star or a row of them."""
    _, high = neutron_star_mass_range("SLY")
    masses = jnp.array([[1.4, 1.6], [high + 0.1, 1.2]])
    redshift = jnp.zeros_like(masses)
    per_row = jax.vmap(lambda m, z: eos_tidal_deformability(m, z, "SLY"))(masses, redshift)
    per_star = jax.vmap(lambda m, z: eos_tidal_deformability(m, z, "SLY"))(masses.ravel(), redshift.ravel())
    eager = eos_tidal_deformability(jnp.array([1.4, 1.6, 1.2]), jnp.zeros(3), "SLY")
    for values in (np.asarray(per_row).ravel(), np.asarray(per_star)):
        assert np.isnan(values[2])
        np.testing.assert_array_equal(values[[0, 1, 3]], np.asarray(eager))


def test_an_unknown_equation_of_state_is_refused_even_when_traced() -> None:
    """A misspelt name fails when the population is built, not as NaN in every row."""
    with pytest.raises(ValueError, match="no neutron-star equation of state called 'NOT_AN_EOS'"):
        eos_tidal_deformability(jnp.array([1.4]), jnp.zeros(1), "NOT_AN_EOS")
    with pytest.raises(ValueError, match="NOT_AN_EOS"):
        jax.jit(lambda m: eos_tidal_deformability(m, jnp.zeros(1), "NOT_AN_EOS"))(jnp.array([1.4]))


def test_without_lalsimulation_the_message_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core install has no lalsimulation; the error says which extra brings it."""
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name in {"lal", "lalsimulation"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match=r"gwmock-pop\[eos\]"):
        tidal_deformability(np.array([1.4]), "SLY")
