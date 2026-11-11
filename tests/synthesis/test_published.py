"""The convolution against the published population built from the same run (opt-in: needs the files).

Set ``GWMOCK_POP_IORIO2023_DIR`` to the folder holding Iorio et al.'s ``NSNSm.csv``
(``simulation_F`` of Zenodo 7794546) and ``GWMOCK_POP_DUPLETSA2026_BNS`` to ``BNSs.hdf5`` from
Zenodo 21682194.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gwmock_pop.catalogue.layouts import read_layout
from gwmock_pop.synthesis.convolution import draw_mergers, expected_mergers, merger_rate_density
from gwmock_pop.synthesis.iorio2023 import read_iorio2023

pytestmark = pytest.mark.integration

_IORIO = os.environ.get("GWMOCK_POP_IORIO2023_DIR")
_DUPLETSA = os.environ.get("GWMOCK_POP_DUPLETSA2026_BNS")


@pytest.mark.skipif(not (_IORIO and _DUPLETSA), reason="needs the published files")
def test_binary_neutron_stars_reproduce_the_published_population() -> None:
    """The redshifts match the record's to 0.007 in cumulative probability; the count is 1.31 times it.

    The count follows from the stated binary fraction and the run's own normalisation, which the
    record does not use; the factor is asserted so that a change to either is seen.
    """
    run = read_iorio2023(Path(str(_IORIO)) / "NSNSm.csv", alpha=1.0)
    rate = merger_rate_density(run)
    published = read_layout(Path(str(_DUPLETSA)), "gwfish_pandas")["z"]
    assert expected_mergers(rate) == pytest.approx(1.311 * published.size, rel=0.01)
    drawn = np.sort(draw_mergers(run, rate, np.random.default_rng(1))["redshift"])
    grid = np.linspace(0.0, 15.0, 3001)
    distance = np.abs(
        np.searchsorted(drawn, grid, side="right") / drawn.size
        - np.searchsorted(np.sort(published), grid, side="right") / published.size
    )
    assert np.max(distance) < 0.007
