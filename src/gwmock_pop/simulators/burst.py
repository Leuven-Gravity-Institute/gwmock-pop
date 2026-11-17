"""A burst population of standard candles: Poisson in time, uniform in volume.

Every source radiates the same energy ``E_GW`` in gravitational waves. A population is a set of
components, one per burst morphology, each with its own rate per unit volume and time and its own
farthest distance. Within a component the number of sources over a span is Poisson, their times
are uniform, and their distances are uniform in Euclidean volume. Each source's ``hrss`` follows
from the energy, its distance and its central frequency through the narrow-band relation of the
LVK all-sky short-burst searches (O3: Phys. Rev. D 104, 122004, Eq. (2)):

    E_GW^iso = (pi^2 c^3 / G) r^2 f0^2 hrss^2.

A white-noise burst radiates isotropically, as in O3; its inclination only sets the ellipse its
polarizations trace. An elliptical sine-Gaussian is a rotating source,
``E_GW^rot = (2/5) E_GW^iso`` at its face-on hrss (ibid., Sec. IV), and its observed hrss^2 follows
the rotating pattern ``((1 + cos^2 i) / 2)^2 + cos^2 i``. Gaussians are broadband, the relation does
not hold for them, and they are refused.

This is the population the O3 search's rate-density upper limits describe: a component at, or
below, the limit for its morphology and a stated energy is consistent with O3. The events carry the
keys of ``gwmock-signal``'s ``BurstSimulator``, which injects them.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gwmock_pop.catalogue.mdc_list import _LAL_C_SI, _LAL_MTSUN_SI

#: ``burst_model`` names, as ``gwmock-signal``'s ``BurstSimulator`` registers them.
SINE_GAUSSIAN = "sine_gaussian"
WHITE_NOISE_BURST = "white_noise_burst"

#: Value of every drawn event's ``injection_set`` key.
STANDARD_CANDLE_LABEL = "standard_candle"

#: The keys a family may fix: its model and its shape. Everything else is the population's: the
#: time, distance and orientation are drawn so that the rate density, the volume and the energy
#: averaged over orientations mean what they say, and the hrss follows from them.
FAMILY_KEYS = frozenset({"burst_model", "quality_factor", "frequency", "low_frequency", "bandwidth", "duration"})

#: The shape each model needs from its family: a sine-Gaussian's quality factor, and a white-noise
#: burst's band and duration. A sine-Gaussian that does not fix its frequency draws one.
_REQUIRED_SHAPE = {SINE_GAUSSIAN: ("quality_factor",), WHITE_NOISE_BURST: ("low_frequency", "bandwidth", "duration")}

#: Largest ``max_distance``, Mpc. The volume is Euclidean and redshift is ignored; at 100 Mpc
#: (z of about 0.02) the luminosity distance and the volume already differ from Euclidean by a few
#: per cent, so a farther component needs a cosmological draw this population does not make.
MAX_DISTANCE_LIMIT = 100.0

#: The event keys, in a fixed order; a key a family does not use is ``None`` in that event.
PARAMETER_NAMES = (
    "coa_time",
    "peak_time",
    "right_ascension",
    "declination",
    "polarization_angle",
    "inclination",
    "distance",
    "energy",
    "hrss",
    "burst_model",
    "quality_factor",
    "frequency",
    "low_frequency",
    "bandwidth",
    "duration",
    "seed",
    "injection_set",
)

#: LAL's parsec, m; LAL defines no megaparsec.
_LAL_PC_SI = 3.085677581491367e16
_MPC_SI = _LAL_PC_SI * 1e6
#: Seconds in a Julian year, the year of a rate density in Gpc^-3 yr^-1.
_YEAR_S = 365.25 * 86_400.0
_GPC3_PER_MPC3 = 1.0e-9
#: Largest seed handed to the white-noise generator (an unsigned 32-bit integer).
_MAX_SEED = 2**32 - 1


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return value


def standard_candle_hrss(
    *, energy: float, distance: float, frequency: float, inclination: float, elliptical: bool
) -> float:
    """Return the hrss at the Earth of a standard candle.

    Args:
        energy: Energy radiated in gravitational waves, in solar masses times c^2.
        distance: Distance to the source, Mpc.
        frequency: Central frequency of the burst, Hz.
        inclination: Angle between the source's rotation axis and the line of sight, rad. Used
            only when *elliptical*.
        elliptical: True for a rotating source (an elliptical sine-Gaussian), False for an
            isotropic one (a white-noise burst).

    Returns:
        ``sqrt(G E / (pi^2 c^3)) / (r f0)`` for an isotropic source. For a rotating source, that
        times ``sqrt(5/2)`` face on, and times ``sqrt(5/4) * sqrt(((1 + cos^2 i) / 2)^2 + cos^2 i)``
        at inclination ``i``, so that the average of hrss^2 over orientations is the isotropic
        value and the energy is the one stated.

    Raises:
        ValueError: If *energy*, *distance* or *frequency* is not finite and positive, or
            *inclination* is not finite.
    """
    energy = _positive_finite(energy, "energy")
    distance = _positive_finite(distance, "distance")
    frequency = _positive_finite(frequency, "frequency")
    if not math.isfinite(inclination):
        raise ValueError(f"inclination must be finite, got {inclination!r}.")
    # G E / c^3 = energy * MTSUN_SI * c^2, with E = energy * MSUN * c^2 and MTSUN = G MSUN / c^3.
    isotropic = math.sqrt(energy * _LAL_MTSUN_SI) * _LAL_C_SI / (math.pi * distance * _MPC_SI * frequency)
    if not elliptical:
        return isotropic
    cos_iota = math.cos(inclination)
    pattern = ((1.0 + cos_iota**2) / 2.0) ** 2 + cos_iota**2
    return isotropic * math.sqrt(1.25 * pattern)


def _central_frequency(event: Mapping[str, Any]) -> float:
    if event["burst_model"] == SINE_GAUSSIAN:
        return float(event["frequency"])
    return float(event["low_frequency"]) + 0.5 * float(event["bandwidth"])


def _check_component(component: Mapping[str, Any], index: int) -> tuple[dict[str, Any], float, float]:
    """Return a component's family, rate density and farthest distance, after checking them."""
    unknown = set(component) - {"family", "rate_density", "max_distance"}
    if unknown or not isinstance(component.get("family"), Mapping):
        raise ValueError(
            f"component {index} must have a mapping 'family', 'rate_density' and 'max_distance', got {sorted(component)}."
        )
    family = dict(component["family"])
    model = family.get("burst_model")
    if model not in (SINE_GAUSSIAN, WHITE_NOISE_BURST):
        raise ValueError(
            f"component {index}: burst_model {model!r} is not supported; the energy-to-hrss relation holds for "
            f"narrow-band {SINE_GAUSSIAN!r} and {WHITE_NOISE_BURST!r} families only."
        )
    fixed = sorted(set(family) - FAMILY_KEYS)
    if fixed:
        raise ValueError(
            f"component {index}: a family fixes only its model and shape ({sorted(FAMILY_KEYS)}); "
            f"it must not fix {fixed}, which the population draws."
        )
    required = _REQUIRED_SHAPE[model]
    missing = sorted(set(required) - set(family))
    if missing:
        raise ValueError(f"component {index}: a {model!r} family must fix {missing}.")
    for key in sorted(set(family) - {"burst_model"}):
        family[key] = _positive_finite(family[key], f"component {index}: {key}")
    rate_density = float(component.get("rate_density", math.nan))
    if not (math.isfinite(rate_density) and rate_density >= 0):
        raise ValueError(f"component {index}: rate_density must be finite and non-negative, got {rate_density!r}.")
    max_distance = _positive_finite(component.get("max_distance", math.nan), f"component {index}: max_distance")
    if max_distance > MAX_DISTANCE_LIMIT:
        raise ValueError(
            f"component {index}: max_distance={max_distance!r} Mpc is beyond {MAX_DISTANCE_LIMIT:g} Mpc, where a "
            "Euclidean volume without redshift no longer describes the population."
        )
    return family, rate_density, max_distance


def _check_settings(
    start_time: float,
    end_time: float,
    energy: float,
    components: Sequence[Mapping[str, Any]],
    sine_gaussian_frequency_range: tuple[float, float],
) -> tuple[float, list[tuple[dict[str, Any], float, float]], tuple[float, float]]:
    """Return the energy, the checked components and the log frequency bounds, after checking them."""
    if not (math.isfinite(start_time) and math.isfinite(end_time) and end_time > start_time):
        raise ValueError(f"end_time must be after start_time, got [{start_time}, {end_time}).")
    energy = _positive_finite(energy, "energy")
    if not components:
        raise ValueError("components must not be empty.")
    checked = [_check_component(component, index) for index, component in enumerate(components)]
    low, high = (float(b) for b in sine_gaussian_frequency_range)
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError(
            f"sine_gaussian_frequency_range must be finite with 0 < low <= high, got {sine_gaussian_frequency_range!r}."
        )
    return energy, checked, (math.log(low), math.log(high))


def draw_standard_candle_population(  # noqa: PLR0913  # the settings of the draw
    *,
    start_time: float,
    end_time: float,
    energy: float,
    components: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
    sine_gaussian_frequency_range: tuple[float, float] = (30.0, 1800.0),
) -> list[dict[str, Any]]:
    """Return a population of standard-candle bursts in ``[start_time, end_time)``.

    Each component is ``{"family": ..., "rate_density": ..., "max_distance": ...}``: a family
    template (``burst_model`` and any fixed shape parameters), a rate per unit volume and time in
    Gpc^-3 yr^-1, and the farthest source in Mpc (at most :data:`MAX_DISTANCE_LIMIT`). Its number of
    sources is Poisson with mean ``rate_density * (4/3) pi max_distance^3 * T``, with ``T`` the span
    in Julian years. Each source draws:

    * ``peak_time`` uniform in the span, ``distance`` uniform in volume out to ``max_distance``;
    * sky position isotropic, polarization angle uniform in ``[0, pi)``, ``inclination`` uniform
      in ``cos(iota)``;
    * ``frequency`` log-uniform in *sine_gaussian_frequency_range* for a sine-Gaussian whose
      family does not fix it, and ``seed`` for a white-noise burst;

    and its ``hrss`` is :func:`standard_candle_hrss` of *energy*, its distance, its central
    frequency (a white-noise burst's is the middle of its band) and its inclination. A burst
    whose peak lies near either end of the span may run past it.

    Args:
        start_time: GPS start of the span, s.
        end_time: GPS end of the span, s.
        energy: Energy each source radiates, in solar masses times c^2.
        components: One mapping per morphology, as above.
        rng: Source of randomness; pass a seeded generator for a reproducible population.
        sine_gaussian_frequency_range: ``(low, high)`` Hz for sine-Gaussian central frequencies.

    Returns:
        One event mapping per source, in time order, each carrying its ``distance`` (Mpc), its
        ``energy`` and ``injection_set="standard_candle"``.

    Raises:
        ValueError: If the span is empty, *energy* or the frequency range is not positive,
            *components* is empty, or a component is malformed, has a negative rate density, a
            distance outside ``(0, MAX_DISTANCE_LIMIT]``, a model other than a sine-Gaussian or a
            white-noise burst, or a family fixing a key other than its model and shape.
    """
    energy, checked, log_frequency = _check_settings(
        start_time, end_time, energy, components, sine_gaussian_frequency_range
    )
    years = (end_time - start_time) / _YEAR_S

    events = []
    for family, rate_density, max_distance in checked:
        volume_gpc3 = (4.0 / 3.0) * math.pi * max_distance**3 * _GPC3_PER_MPC3
        count = int(rng.poisson(rate_density * volume_gpc3 * years))
        for peak_time in rng.uniform(start_time, end_time, size=count):
            model = family["burst_model"]
            event: dict[str, Any] = {
                "peak_time": float(peak_time),
                "right_ascension": rng.uniform(0.0, 2 * math.pi),
                "declination": math.asin(rng.uniform(-1.0, 1.0)),
                "polarization_angle": rng.uniform(0.0, math.pi),
                "inclination": math.acos(rng.uniform(-1.0, 1.0)),
                # 1 - U is in (0, 1], so no source sits at zero distance.
                "distance": max_distance * (1.0 - rng.uniform(0.0, 1.0)) ** (1.0 / 3.0),
                "energy": energy,
            }
            if model == SINE_GAUSSIAN:
                event["frequency"] = math.exp(rng.uniform(*log_frequency))
            else:
                event["seed"] = int(rng.integers(_MAX_SEED + 1))
            event.update(family)
            event["hrss"] = standard_candle_hrss(
                energy=energy,
                distance=event["distance"],
                frequency=_central_frequency(event),
                inclination=event["inclination"],
                elliptical=model == SINE_GAUSSIAN,
            )
            event["injection_set"] = STANDARD_CANDLE_LABEL
            events.append(event)
    events.sort(key=lambda event: event["peak_time"])
    return events


class StandardCandleBurstPopulation:
    """A standard-candle burst population exposed as a ``GWPopSimulator`` (``source_type="burst"``).

    The population is fixed by its span, energy, components and seed: :meth:`simulate` draws it
    whole and returns one object column per name in :data:`PARAMETER_NAMES`, with
    ``coa_time = peak_time`` so a segmenting caller places each burst by its peak.
    """

    def __init__(  # noqa: PLR0913  # the population's settings, named in a configuration
        self,
        *,
        start_time: float,
        end_time: float,
        energy: float,
        components: Sequence[Mapping[str, Any]],
        seed: int,
        sine_gaussian_frequency_range: tuple[float, float] = (30.0, 1800.0),
    ) -> None:
        """Store the settings, after checking them.

        Args:
            start_time: GPS start of the span, s.
            end_time: GPS end of the span, s.
            energy: Energy each source radiates, in solar masses times c^2.
            components: One mapping per morphology; see :func:`draw_standard_candle_population`.
            seed: Seed of the draw, a non-negative integer.
            sine_gaussian_frequency_range: ``(low, high)`` Hz for sine-Gaussian central frequencies.

        Raises:
            ValueError: If *seed* is not a non-negative integer, or a setting is invalid.
        """
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
        # Checked now, so a bad configuration fails when it is read and not when it is simulated.
        _check_settings(start_time, end_time, energy, components, sine_gaussian_frequency_range)
        self._settings = {
            "start_time": float(start_time),
            "end_time": float(end_time),
            "energy": float(energy),
            "components": [{**component, "family": dict(component["family"])} for component in components],
            "sine_gaussian_frequency_range": tuple(float(b) for b in sine_gaussian_frequency_range),
        }
        self._seed = seed

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The event keys, in a fixed order."""
        return PARAMETER_NAMES

    @property
    def source_type(self) -> str:
        """``"burst"``, the source type of ``gwmock-signal``'s ``BurstSimulator``."""
        return "burst"

    @property
    def metadata(self) -> dict[str, Any]:
        """The settings and seed that fix the population, as a copy a caller may change freely."""
        settings = copy.deepcopy(self._settings)
        return {**settings, "seed": self._seed, "generator": f"{__name__}.draw_standard_candle_population"}

    def simulate(self, n_samples: int | None = None, seed: int | None = None, **_: Any) -> dict[str, np.ndarray]:
        """Return the whole population, one object column per name in :data:`PARAMETER_NAMES`.

        Args:
            n_samples: Must be ``None``: the number of sources is drawn from the rate densities.
            seed: Ignored; the population's own seed fixes it, so it does not change with the
                seed a caller derives for other streams.

        Returns:
            Mapping from each name to a 1-D object array, one entry per source, in time order.

        Raises:
            ValueError: If *n_samples* is given.
        """
        del seed
        if n_samples is not None:
            raise ValueError("A standard-candle population draws its own size; omit n_samples.")
        events = draw_standard_candle_population(**self._settings, rng=np.random.default_rng(self._seed))
        columns = {name: np.empty(len(events), dtype=object) for name in PARAMETER_NAMES}
        for index, event in enumerate(events):
            event["coa_time"] = event["peak_time"]
            for name in PARAMETER_NAMES:
                columns[name][index] = event.get(name)
        return columns
