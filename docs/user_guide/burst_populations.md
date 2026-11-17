# Burst Populations

`StandardCandleBurstPopulation` draws unmodelled bursts from sources that each
radiate the same energy $E_\mathrm{GW}$ in gravitational waves. It is the
population the O3 all-sky short-burst search's rate-density upper limits
describe [O3, Sect. IV]: a rate density at, or below, a limit for a stated
energy gives a population consistent with O3. Its events carry the keys of
`gwmock-signal`'s `BurstSimulator`, which injects them, and its source type is
`"burst"`.

```python
from gwmock_pop.simulators import StandardCandleBurstPopulation

population = StandardCandleBurstPopulation(
    start_time=1419724818.0,
    end_time=1419724818.0 + 30 * 86400.0,
    energy=1.0e-6,  # solar masses c^2
    components=[
        {
            "family": {"burst_model": "sine_gaussian", "quality_factor": 100.0, "frequency": 235.0},
            "rate_density": 2.1e8,  # Gpc^-3 yr^-1
            "max_distance": 15.0,  # Mpc
        },
        {
            "family": {"burst_model": "white_noise_burst", "low_frequency": 100.0, "bandwidth": 100.0, "duration": 0.1},
            "rate_density": 4.7e7,
            "max_distance": 20.0,
        },
    ],
    seed=1102,
)
events = population.simulate()
```

## The draw

A population is a list of components, one per morphology, each with its own rate
density $\mathcal{R}$ and farthest distance $d_\mathrm{max}$, since the O3
limits differ by orders of magnitude between morphologies.

- **Number and times.** A component's number of sources is Poisson with mean
  $\mathcal{R} \cdot \tfrac{4}{3}\pi d_\mathrm{max}^3 \cdot T$, with $T$ the
  span in Julian years, and their peak times are uniform over the span. A burst
  near either end may run past it. Each event's `coa_time` is its `peak_time`.
- **Distances.** Uniform in Euclidean volume out to `max_distance`, at most 100
  Mpc. Redshift is ignored; at 100 Mpc ($z \approx 0.02$) that is already an
  error of a few per cent in distance and volume. Choose `max_distance` where
  the faintest source is well below the network's threshold.
- **Families.** Elliptical sine-Gaussians and white-noise bursts. Gaussians are
  refused: they are broadband, and the energy relation below does not hold for
  them [O3, Sect. IV]. A family fixes only its model and shape
  (`quality_factor`, `frequency`, `low_frequency`, `bandwidth`, `duration`); a
  sine-Gaussian that does not fix `frequency` draws it log-uniformly over
  30–1800 Hz. The population draws the times, distances, orientations and
  white-noise seeds.
- **Orientation.** Isotropic sky, uniform polarisation angle, inclination
  uniform in $\cos\iota$.
- **Seed.** The population's own `seed` fixes it; a seed passed to `simulate` is
  ignored, so the population does not change with the seeds of other streams.

## Strength

Each event's $h_\mathrm{rss}$ follows from the energy, its distance $r$ and its
central frequency $f_0$ (a white-noise burst's is the middle of its band)
through the narrow-band relation of the O3 search [O3, Eq. (2)]:

$$E_\mathrm{GW}^\mathrm{iso} = \frac{\pi^2 c^3}{G}\, r^2 f_0^2 h_\mathrm{rss}^2 .$$

A white-noise burst radiates isotropically, as in O3; its inclination only sets
the ellipse its polarisations trace. A sine-Gaussian is a rotating source,
$E_\mathrm{GW}^\mathrm{rot} = \tfrac{2}{5} E_\mathrm{GW}^\mathrm{iso}$ at its
face-on $h_\mathrm{rss}$ [O3, Sect. IV], and its observed $h_\mathrm{rss}^2$
follows the rotating pattern $\left(\tfrac{1+\cos^2\iota}{2}\right)^2 +
\cos^2\iota$. Averaged over orientations, $h_\mathrm{rss}^2$ is the isotropic
value, so every source radiates the stated energy. `standard_candle_hrss`
evaluates the relation.

## Choosing the rate density and the energy

The O3 limits scale as $E_\mathrm{GW}^{-3/2}$ [O3, Sect. IV], and a detector's
reach for a standard candle grows as $E_\mathrm{GW}^{1/2}$, so its volume grows
as $E_\mathrm{GW}^{3/2}$. The expected number of detectable sources at a fixed
fraction of a limit therefore does not depend on the energy; the energy only
sets the distances at which the sources sit. Choose the fraction for the number
of detectable bursts wanted, and the energy so that the farthest relevant source
stays within 100 Mpc.

[O3]: R. Abbott et al., "All-sky search for short gravitational-wave bursts in
the third Advanced LIGO and Advanced Virgo run", Phys. Rev. D 104, 122004
(2021), arXiv:2107.03701.
