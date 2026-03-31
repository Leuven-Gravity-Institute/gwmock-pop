"""Configuration and fixtures for pytest.

Hypothesis test profiles
------------------------
Three named profiles are registered here.  Activate them by setting the
``HYPOTHESIS_PROFILE`` environment variable before running pytest, or call
``hypothesis.settings.load_profile("<name>")`` programmatically.

``default``
    100 examples, all five phases enabled.  Used for regular local development.

``dev``
    10 examples, ``HealthCheck.too_slow`` suppressed, ``Phase.target`` omitted.
    For rapid iteration during exploratory work.

``ci``
    50 examples, ``HealthCheck.too_slow`` suppressed, ``Phase.target`` omitted.
    Used by GitHub Actions; balances coverage against wall-clock time.

The ``default`` profile is loaded automatically so the suite always has sensible
defaults without requiring an environment variable.
"""

from __future__ import annotations

from hypothesis import HealthCheck, Phase, settings

settings.register_profile(
    "default",
    max_examples=100,
    phases=list(Phase),
)

settings.register_profile(
    "dev",
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
    phases=[p for p in Phase if p is not Phase.target],
)

settings.register_profile(
    "ci",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    phases=[p for p in Phase if p is not Phase.target],
)

settings.load_profile("default")
