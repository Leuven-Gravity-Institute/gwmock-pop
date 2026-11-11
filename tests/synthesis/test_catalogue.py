"""Catalogue columns from drawn mergers, and the file a catalogue draw reads."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from gwmock_pop.catalogue.population import read_catalogue_columns
from gwmock_pop.constants import PLANCK18_H0_KM_S_MPC, PLANCK18_OMEGA_M
from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.provenance.io import read_provenance
from gwmock_pop.synthesis.catalogue import catalogue_columns, write_catalogue

pytestmark = pytest.mark.unit

_N = 20000


def _inputs(n: int = _N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "redshift": rng.uniform(0.01, 14.0, n),
        "mass_1": rng.uniform(5.0, 40.0, n),
        "mass_2": rng.uniform(1.1, 2.0, n),
        "spin_1z": rng.uniform(-0.5, 0.5, n),
        "spin_2z": rng.uniform(-0.05, 0.05, n),
    }


def test_the_given_quantities_are_kept_and_the_distance_is_planck18() -> None:
    """Masses, spins and redshift are stored as given; dL is astropy's, in Gpc."""
    given = _inputs()
    columns = catalogue_columns(**given, rng=np.random.default_rng(1))
    assert np.array_equal(columns["m1_source"], given["mass_1"])
    assert np.array_equal(columns["m2_source"], given["mass_2"])
    assert np.array_equal(columns["chi1z"], given["spin_1z"])
    assert np.array_equal(columns["chi2z"], given["spin_2z"])
    assert np.array_equal(columns["z"], given["redshift"])
    cosmology = FlatLambdaCDM(H0=PLANCK18_H0_KM_S_MPC, Om0=PLANCK18_OMEGA_M)
    expected = cosmology.luminosity_distance(given["redshift"]).to(u.Gpc).value
    assert np.allclose(columns["dL"], expected, rtol=1e-5, atol=0.0)
    assert "Lambda1" not in columns


def test_orientations_and_sky_positions_are_isotropic() -> None:
    """ra, Phicoal, psi are uniform on their ranges; sin(dec) and cos(iota) are uniform in [-1, 1]."""
    columns = catalogue_columns(**_inputs(), rng=np.random.default_rng(1))
    uniform = {
        "ra": columns["ra"] / (2.0 * np.pi),
        "Phicoal": columns["Phicoal"] / (2.0 * np.pi),
        "psi": columns["psi"] / np.pi,
        "sin_dec": (np.sin(columns["dec"]) + 1.0) / 2.0,
        "cos_iota": (np.cos(columns["iota"]) + 1.0) / 2.0,
    }
    for name, values in uniform.items():
        assert np.all((values >= 0.0) & (values <= 1.0)), name
        counts, _ = np.histogram(values, bins=10, range=(0.0, 1.0))
        assert np.all(np.abs(counts - _N / 10) < 5.0 * np.sqrt(_N / 10)), name
    assert abs(np.mean(columns["dec"])) < 0.05
    assert np.all((columns["dec"] >= -np.pi / 2) & (columns["dec"] <= np.pi / 2))


def test_deformabilities_are_stored_when_given() -> None:
    """Lambda1 and Lambda2 are stored as given."""
    given = _inputs(5)
    columns = catalogue_columns(
        **given, rng=np.random.default_rng(1), lambda_1=np.zeros(5), lambda_2=np.arange(5.0) * 100.0
    )
    assert np.array_equal(columns["Lambda1"], np.zeros(5))
    assert np.array_equal(columns["Lambda2"], np.arange(5.0) * 100.0)


def test_one_deformability_alone_is_refused() -> None:
    """A deformability for one component and not the other is refused."""
    with pytest.raises(PopulationValidationError, match="both tidal deformabilities"):
        catalogue_columns(**_inputs(5), rng=np.random.default_rng(1), lambda_2=np.zeros(5))


@pytest.mark.parametrize("name", ["mass_1", "mass_2", "spin_1z", "spin_2z"])
def test_misaligned_inputs_are_refused(name) -> None:
    """Every per-merger input must have one value per redshift."""
    given = _inputs(5)
    given[name] = given[name][:4]
    with pytest.raises(PopulationValidationError, match="do not match"):
        catalogue_columns(**given, rng=np.random.default_rng(1))


def test_a_redshift_beyond_the_cosmology_table_is_refused() -> None:
    """A merger beyond the history's highest redshift has no tabulated distance, and is refused."""
    given = _inputs(3)
    given["redshift"][1] = 16.0
    with pytest.raises(ValueError, match="redshift"):
        catalogue_columns(**given, rng=np.random.default_rng(1))


def test_the_written_catalogue_is_read_by_the_draw(tmp_path) -> None:
    """The draw's own reader returns every column unchanged, and the provenance is kept."""
    given = _inputs(50)
    columns = catalogue_columns(**given, rng=np.random.default_rng(1), lambda_1=np.zeros(50), lambda_2=np.ones(50))
    path = tmp_path / "catalogue.h5"
    write_catalogue(path, columns, {"source": "test", "seed": 1})
    read = read_catalogue_columns(path, tuple(columns))
    assert all(np.array_equal(read[name], columns[name]) for name in columns)
    assert read_provenance(path) == {"source": "test", "seed": 1}
