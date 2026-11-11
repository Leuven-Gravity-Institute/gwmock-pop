"""Reading Iorio et al.'s merging-binary tables: the pin, the selection, the delay and the split."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.synthesis import iorio2023
from gwmock_pop.synthesis.iorio2023 import (
    IORIO2023_EVOLVED_MASS,
    black_hole_neutron_star_masses,
    imf_fraction,
    read_iorio2023,
)

pytestmark = pytest.mark.unit

_HEADER = "ID,name,BWorldtime,Mass_0,RemnantType_0,Mass_1,RemnantType_1,GWtime,Mzams_0,Mzams_1,Z,alpha"
_ROWS = (
    "1,a,10.0,8.5,6,1.3,5,90.0,30.0,12.0,0.0001,1.0",
    "2,b,20.0,1.4,4,12.0,6,30.0,9.0,10.0,0.02,1.0",
    "3,c,5.0,9.0,6,1.2,5,1.5,25.0,14.0,0.02,0.5",
)


@pytest.fixture
def table(tmp_path, monkeypatch):
    """A small ``BHNSm.csv`` in the published columns, pinned in place of the real one."""
    path = tmp_path / "BHNSm.csv"
    path.write_text("\n".join((_HEADER, *_ROWS)) + "\n")
    monkeypatch.setitem(iorio2023.IORIO2023_TABLES, "BHNSm.csv", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_one_efficiency_is_read_with_its_delays(table) -> None:
    """Only the rows at the requested alpha are read; a delay is BWorldtime plus GWtime, in years."""
    run = read_iorio2023(table, alpha=1.0)
    assert np.array_equal(run.delay_time, [100e6, 50e6])
    assert np.array_equal(run.metallicity, [0.0001, 0.02])
    assert run.evolved_mass == IORIO2023_EVOLVED_MASS
    assert np.array_equal(run.properties["mass_0"], [8.5, 1.4])
    assert np.array_equal(run.properties["remnant_type_1"], [5, 6])
    assert read_iorio2023(table, alpha=0.5).delay_time.tolist() == [6.5e6]
    assert run.imf_fraction == imf_fraction()


def test_a_changed_table_is_refused(table) -> None:
    """A file that does not match its pin is refused before it is read."""
    table.write_text(table.read_text().replace("90.0", "91.0"))
    with pytest.raises(PopulationValidationError, match="not the pinned"):
        read_iorio2023(table)


def test_an_unpinned_table_is_refused(tmp_path) -> None:
    """Only the pinned tables are read."""
    path = tmp_path / "BHBHm.csv"
    path.write_text(_HEADER + "\n")
    with pytest.raises(PopulationValidationError, match="is not one of"):
        read_iorio2023(path)


def test_an_absent_efficiency_is_refused(table) -> None:
    """An alpha the table does not hold is refused, not read as an empty run."""
    with pytest.raises(PopulationValidationError, match="no binary at alpha"):
        read_iorio2023(table, alpha=3.0)


def test_the_pins_are_the_published_files() -> None:
    """The pins are the fiducial model's tables in Zenodo record 7794546."""
    assert iorio2023.IORIO2023_TABLES == {
        "BHNSm.csv": "3ee9edaa86fced8c72833e408c2651665fc53f2d24667553c2f4f70bd6d2c4fa",
        "NSNSm.csv": "16da4c8858a9361ce88ad573da3314db0f5aa40825ce41b4a2822aaecdb65aa2",
    }


def test_the_black_hole_is_found_in_either_column() -> None:
    """The black hole's mass is taken from whichever component SEVN typed as one."""
    mergers = {
        "mass_0": np.array([8.5, 1.4, 1.2]),
        "mass_1": np.array([1.3, 12.0, 20.0]),
        "remnant_type_0": np.array([6, 4, 5]),
        "remnant_type_1": np.array([5, 6, 6]),
    }
    black_hole, neutron_star = black_hole_neutron_star_masses(mergers)
    assert np.array_equal(black_hole, [8.5, 12.0, 20.0])
    assert np.array_equal(neutron_star, [1.3, 1.4, 1.2])


@pytest.mark.parametrize(("type_0", "type_1"), [(6, 6), (5, 5), (6, 3), (1, 6)])
def test_a_merger_that_is_not_one_of_each_is_refused(type_0, type_1) -> None:
    """Two black holes, two neutron stars, or any other remnant are refused."""
    mergers = {
        "mass_0": np.array([8.0]),
        "mass_1": np.array([1.3]),
        "remnant_type_0": np.array([type_0]),
        "remnant_type_1": np.array([type_1]),
    }
    with pytest.raises(PopulationValidationError, match="one black hole and one neutron star"):
        black_hole_neutron_star_masses(mergers)


def _full_population(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Binaries as the archive's estimate_mass_correction.py draws them: Kroupa primaries 0.08-150, q^-0.1."""
    low, mid, high = 0.08, 0.5, 150.0
    first = (mid**-0.3 - low**-0.3) / -0.3
    second = mid ** (-1.3 + 2.3) * (high**-1.3 - mid**-1.3) / -1.3
    below = rng.uniform(size=n) < first / (first + second)
    u = rng.uniform(size=n)
    primary = np.where(
        below,
        (low**-0.3 + u * (mid**-0.3 - low**-0.3)) ** (1 / -0.3),
        (mid**-1.3 + u * (high**-1.3 - mid**-1.3)) ** (1 / -1.3),
    )
    return primary, primary * _sana_q(np.maximum(0.08 / primary, 0.1), rng)


def _sana_q(floor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=floor.size)
    return (floor**0.9 + u * (1.0 - floor**0.9)) ** (1 / 0.9)


def test_the_imf_fraction_is_the_published_cut_fraction() -> None:
    """A Monte Carlo of their estimate_mass_correction.py gives imf_fraction: primary mass passing both cuts.

    They quote 0.255 +- 0.001, from a Monte Carlo; this exact value, not 0.255, reproduces their
    published rates.
    """
    rng = np.random.default_rng(1)
    primary, secondary = _full_population(4_000_000, rng)
    cut = (primary >= 5.0) & (secondary >= 2.2)
    monte_carlo = primary[cut].sum() / primary.sum()
    assert imf_fraction() == pytest.approx(monte_carlo, abs=0.005)  # this sample's noise is about 0.001
    assert imf_fraction() == pytest.approx(0.25643, abs=1e-5)
