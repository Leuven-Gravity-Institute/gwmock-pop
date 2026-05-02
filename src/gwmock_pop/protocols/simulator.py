"""Protocol definition for gravitational-wave population simulators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class GWPopSimulator(Protocol):
    """Structural contract between ``gwmock-pop`` and ``gwmock``.

    Any object that exposes ``parameter_names``, ``source_type``, and
    ``simulate`` satisfies this protocol, regardless of its class hierarchy.
    ``@runtime_checkable`` means ``isinstance`` checks work at runtime, so
    ``gwmock`` can guard calls with::

        assert isinstance(sim, GWPopSimulator)

    and receive a clear ``TypeError`` if a misconfigured backend is passed.

    **How gwmock uses this protocol**

    ``gwmock`` drives a population run by calling ``simulate(n_samples)`` once
    and slicing the returned mapping into per-event dicts for ``gwmock-signal``::

        population = sim.simulate(n_samples=1000)
        for i in range(n_samples):
            params_i = {k: v[i] for k, v in population.items()}
            signal_sim.simulate(params_i, ...)

    The ``source_type`` property is used by ``gwmock`` to select the matching
    ``gwmock-signal`` simulator (e.g. a ``CBCSimulator`` for compact-binary keys
    such as ``"bbh"``, ``"bns"``, or ``"nsbh"``).

    **Valid ``source_type`` values**

    +------------------+-------------------------------------------+
    | ``"bbh"``        | Binary black hole coalescence             |
    +------------------+-------------------------------------------+
    | ``"bns"``        | Binary neutron star coalescence           |
    +------------------+-------------------------------------------+
    | ``"nsbh"``       | Neutron star - black hole coalescence     |
    +------------------+-------------------------------------------+
    | ``"stochastic"`` | Stochastic gravitational-wave background  |
    +------------------+-------------------------------------------+
    | ``"supernova"``  | Core-collapse supernova                   |
    +------------------+-------------------------------------------+
    | ``"cosmic_string"`` | Cosmic string burst                    |
    +------------------+-------------------------------------------+

    **Array contract**

    Every value in the mapping returned by ``simulate`` is a 1-D ``jax.Array``
    of shape ``(n_samples,)``.  The keys are exactly ``parameter_names`` in the
    same order.
    """

    @property
    def parameter_names(self) -> Sequence[str]:
        """Stable ordered list of output parameter keys.

        The returned sequence must be identical across repeated calls on the
        same instance.  gwmock relies on positional consistency when building
        per-event dicts from the mapping returned by :meth:`simulate`.

        Returns:
            An ordered sequence of parameter name strings.  The sequence is the
            authoritative list of keys present in every mapping returned by
            :meth:`simulate`.
        """
        ...

    @property
    def source_type(self) -> str:
        """Non-empty routing key used by ``gwmock`` to select a signal simulator.

        The value must be a non-empty string.  ``gwmock`` uses it as a
        dictionary key to look up the matching ``gwmock-signal`` simulator
        (e.g. ``"bbh"`` → ``CBCSimulator``).  Returning an empty string or
        ``None`` at call-time is an error.

        Returns:
            Source type string (e.g. ``"bbh"``, ``"bns"``, ``"nsbh"``,
            ``"stochastic"``, ``"supernova"``, ``"cosmic_string"``).
        """
        ...

    def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
        """Draw ``n_samples`` parameter sets from the population model.

        The returned mapping must satisfy two invariants:

        1. **Key completeness and order**:
            ``list(result.keys()) == list(self.parameter_names)``.
        2. **Shape contract**: every value is a ``jax.Array`` with leading
           dimension ``n_samples``, i.e. ``result[k].shape[0] == n_samples``
           for all ``k``.

        Args:
            n_samples: Number of independent source realizations to draw.
                Must be a positive integer.
            **kwargs: Optional backend-specific keyword arguments (e.g. a JAX
                PRNG key or a configuration override).  gwmock does not pass
                any keyword arguments; they exist for direct library use.

        Returns:
            Mapping from parameter name to 1-D ``jax.Array`` of length
            ``n_samples``.
        """
        ...
