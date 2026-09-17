"""Driving the transmitter: intensities and clock, chosen while the link moves.

The control problem
-------------------
Decoy-state BB84 has four knobs the transmitter can turn without touching the
hardware — the signal intensity :math:`\\mu`, the weak decoy :math:`\\nu`, how
the pulses are split between them, and the clock — and the best setting is a
function of conditions that change during a working day: the loss on the span,
the optical misalignment left by the compensator, and how close the detectors
are to saturating.

A deployed system sets these once, at commissioning, from a link budget.  That
is the baseline here (:class:`StaticPolicy`), and it is a good one: the
asymptotic optimum is flat enough near its peak that a well-chosen static point
is hard to beat by a little.  It is beatable by a lot only when the conditions
move far enough that the peak moves, which is exactly when an operator would
rather not be sending an engineer.

Why reinforcement learning and not a solver
--------------------------------------------
The optimum is computable in closed-ish form *if you know the channel*.  The
receiver does not: it knows the gains and error rates it measured last period,
which are noisy estimates that get noisier as the settings themselves change
(sending fewer decoy pulses to buy raw rate makes the bound that pays for the
key less certain).  The agent is therefore choosing under an observation it
partly controls, with a delay of one settling period, which is a control problem
and not an optimisation.

The reward is the key that can be *proved*, not the key that exists
------------------------------------------------------------------
Every step pays the secret bits extracted from that period's **sampled**
observables — the same bound a real protocol would run, on the same finite
counts — rather than the asymptotic rate the same settings would reach with
infinite statistics.  The two differ most on the weak decoy: asymptotically the
bound is tightest as :math:`\\nu \\to 0`, while a real run with a very dim decoy
has few decoy clicks and a bound made mostly of noise.

Where that trade actually falls is a measurement, and the measurement surprised
the author.  At the phase-5 operating point — a one-second block at a gigahertz,
a fifth of the pulses on the decoy — even :math:`\\nu = 0.01` produces tens of
thousands of decoy clicks on a metropolitan span, which is plenty, and the
trained agent drives the decoy to the floor of its range exactly as the
asymptotic formula says it should.  The finite-statistics penalty only bites at
80 km, where the agent raises the decoy back to 0.08 and, not coincidentally, is
also the one span where it fails to beat the static baseline.  Rewarding the
provable key is still the right choice — it is what a deployed system extracts —
but at this block size it does not change the answer, and saying otherwise would
be a story rather than a result.

The policy
----------
A softmax over a **fixed** set of adjustments, not the pointer-network shape of
:class:`~qnet_autopilot.agents.actor_critic.ActorCriticPolicy`.  Phase 2 scores a
variable action set — the legal moves change every step — and needs its
architecture for that.  Here the nine adjustments are always available and
always mean the same thing, so an ordinary fixed head is the right model and a
smaller one.  Everything else is deliberately the same: the numpy MLP, Adam,
advantage actor-critic with GAE, the entropy schedule, and keeping the best
checkpoint on held-out seeds rather than the last.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..agents.actor_critic import _generalised_advantages
from ..agents.normalise import FeatureNormaliser
from ..agents.numpy_nn import MLP, Adam, categorical_entropy, softmax
from ..physics import constants as C
from ..physics.decoy import (DecoySettings, apply_detector_saturation,
                             decoy_key_rate_hz, observe, optimal_signal_intensity,
                             sample_observables, single_photon_bounds)
from .session import LinkConditions
from .timing import TimingBudget, gate_efficiency, intersymbol_error, timing_spread

__all__ = [
    "ACTIONS", "KEYRATE_STATE_FEATURES", "ActionSpec",
    "KeyRateEnvConfig", "KeyRateEnv", "Transition", "EpisodeResult",
    "KeyRatePolicy", "StaticPolicy", "OraclePolicy", "commissioned_settings",
    "receiver_response", "run_episode", "TrainingConfig", "train_key_rate_policy",
]


def receiver_response(link: LinkConditions, settings: DecoySettings,
                      budget: TimingBudget | None = None,
                      dead_time_s: float = 50.0e-9,
                      afterpulse_probability: float = 0.01):
    """The channel including everything the receiver itself costs.

    Three corrections on top of the fibre, in the order a photon meets them:

    1. **the gate** accepts only the share of the arrival distribution inside
       its window, taken here at perfect alignment — keeping it aligned is
       :mod:`~qnet_autopilot.qkd.timing`'s problem, not this module's;
    2. **the slot period** the clock implies adds an inter-symbol error, which
       is what stops "turn the clock up" from being free;
    3. **dead time and afterpulsing** scale the yields down and the background
       up, which is what stops it a second time.

    Args:
        link: the span's current conditions.
        settings: the transmitter configuration — the clock enters twice, via
            the inter-symbol error and via saturation.
        budget: the timing budget; defaults to :class:`~qnet_autopilot.qkd.timing.TimingBudget`.
        dead_time_s: detector dead time.
        afterpulse_probability: afterpulse probability per click.

    Returns:
        ChannelResponse: what Bob's detectors actually see.
    """
    from dataclasses import replace

    budget = budget or TimingBudget()
    spread_s = timing_spread(budget, link.length_km)
    efficiency = float(gate_efficiency(0.0, link.gate_width_s, spread_s))
    crosstalk = float(intersymbol_error(settings.clock_rate_hz, spread_s))

    gated = replace(link, coupling_efficiency=link.coupling_efficiency * efficiency,
                    misalignment_error=float(np.clip(
                        link.misalignment_error + crosstalk, 0.0, 0.5)))
    return apply_detector_saturation(gated.response(), settings, dead_time_s,
                                     afterpulse_probability)


def commissioned_settings(link: LinkConditions,
                          settings: DecoySettings | None = None,
                          budget: TimingBudget | None = None,
                          dead_time_s: float = 50.0e-9,
                          afterpulse_probability: float = 0.01,
                          grid: Sequence[float] | None = None,
                          clock_grid: Sequence[float] | None = None
                          ) -> DecoySettings:
    """The operating point an engineer would leave the link at.

    The signal intensity and the clock are optimised against the link budget as
    commissioned — the asymptotic rate on the honest channel, with the
    receiver's own limits folded in — and everything else is left at the
    rule-of-thumb values a deployed system ships with.  In particular the weak
    decoy is *not* optimised: asymptotically its optimum is "as small as you can
    measure", which is not a number a budget calculation produces, and the rule
    of thumb is what is actually used.  That gap is deliberate and is one of the
    things phase 5 measures.

    Args:
        link: the span as commissioned.
        settings: the transmitter's other parameters; defaults to
            :class:`~qnet_autopilot.physics.decoy.DecoySettings`.
        budget: the timing budget.
        dead_time_s: detector dead time, folded into the budget.
        afterpulse_probability: afterpulse probability, likewise.
        grid: candidate signal intensities.  Defaults to 0.05 to 1.0 in 40 steps.
        clock_grid: candidate clock rates.  Defaults to eight rates spanning
            0.1 to 5 GHz, the range the agent's action space can reach.

    Returns:
        DecoySettings: with the commissioned signal intensity and clock.

    Note:
        The search is joint over the two, because they are not independent: both
        set the click rate, so the best intensity at a gigahertz is not the best
        one at five.  Optimising them one at a time lands on a worse point and
        would make the static baseline look weaker than it is.
    """
    settings = settings or DecoySettings()
    intensities = np.asarray(grid if grid is not None else np.linspace(0.05, 1.0, 40),
                             dtype=float)
    intensities = intensities[intensities > settings.decoy_intensity * 1.05]
    clocks = np.asarray(clock_grid if clock_grid is not None
                        else np.geomspace(1.0e8, 5.0e9, 8), dtype=float)

    best, best_rate = settings, -np.inf
    for clock in clocks:
        for intensity in intensities:
            candidate = settings.replace(signal_intensity=float(intensity),
                                         clock_rate_hz=float(clock))
            response = receiver_response(link, candidate, budget, dead_time_s,
                                         afterpulse_probability)
            rate = decoy_key_rate_hz(observe(response, candidate))
            if rate > best_rate:
                best, best_rate = candidate, rate
    return best


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSpec:
    """One adjustment the agent can make to the transmitter.

    Attributes:
        name: display name, and the column name in the decision log.
        signal_factor: multiplier on the signal intensity.
        decoy_factor: multiplier on the weak decoy intensity.
        clock_factor: multiplier on the clock rate.
        signal_probability_step: additive change to the share of signal pulses.

    Note:
        Multiplicative on the intensities and the clock, additive on the
        probability, because that is the geometry of each quantity: intensities
        and rates are naturally logarithmic and a share is not.  A single
        additive step on the intensity would be a 40% move at one end of the
        range and a 4% move at the other.
    """

    name: str
    signal_factor: float = 1.0
    decoy_factor: float = 1.0
    clock_factor: float = 1.0
    signal_probability_step: float = 0.0


#: The nine moves.  Small enough that every one is tried often during training,
#: coarse enough that a few steps cross the useful range of each knob.
ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec("manter"),
    ActionSpec("mu +25%", signal_factor=1.25),
    ActionSpec("mu -20%", signal_factor=0.80),
    ActionSpec("nu +25%", decoy_factor=1.25),
    ActionSpec("nu -20%", decoy_factor=0.80),
    ActionSpec("clock +50%", clock_factor=1.50),
    ActionSpec("clock -33%", clock_factor=0.67),
    ActionSpec("p_sinal +0.05", signal_probability_step=0.05),
    ActionSpec("p_sinal -0.05", signal_probability_step=-0.05),
)

#: Width of the observation vector; see :meth:`KeyRateEnv.observe`.
KEYRATE_STATE_FEATURES = 16


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

@dataclass
class KeyRateEnvConfig:
    """The link the agent drives, and how much it is allowed to move.

    Attributes:
        link: the span, as commissioned.
        initial: the transmitter configuration the episode starts from.  When
            ``lengths_km`` is set this is only a template: each episode starts
            from the commissioning point of the span it drew.
        lengths_km: spans to draw from, one per episode.  ``None`` keeps every
            episode on ``link``'s own length, which trains a specialist.  A
            policy trained at one span learns that span's operating point and
            does not transfer to a very different one — measured, and reported,
            in the phase-5 results — so a deployable policy is trained with this
            set.
        budget: the timing budget.  It is here because the clock rate is one of
            the agent's knobs and the ceiling on it belongs to the span: the
            pulse arrives spread by dispersion, and once that spread approaches
            the slot period the neighbouring slots bleed into each other.  See
            :func:`~qnet_autopilot.qkd.timing.intersymbol_error`.
        period_s: settling period; one action per period.
        steps: periods per episode.
        dead_time_s: detector dead time — what makes the clock a trade.
        afterpulse_probability: afterpulse probability per click.
        loss_drift_db: standard deviation of the slow excess-loss random walk
            over the whole episode.
        loss_step_db: size of the occasional step in excess loss — a bend, a
            connector, a patch panel touched by someone else's work.
        loss_step_probability: chance an episode contains one.
        misalignment_drift: standard deviation of the optical misalignment
            random walk over the episode, as a fraction of the starting value.
        signal_intensity_range, decoy_intensity_range, clock_range,
        signal_probability_range: the box the agent may move in.  These are
            hardware limits, not hyperparameters: an attenuator has a range, a
            driver has a maximum rate, and a protocol that never sends a vacuum
            pulse cannot measure its own dark counts.
        decoy_ratio_max: the decoy may not exceed this share of the signal.
            Above it the vacuum+weak bound stops constraining anything and the
            key rate falls off a cliff for a reason that is mathematical rather
            than physical; clamping keeps the agent's action space inside the
            regime the bound is valid in.
    """

    link: LinkConditions = field(default_factory=LinkConditions)
    initial: DecoySettings = field(default_factory=DecoySettings)
    lengths_km: tuple[float, ...] | None = None
    budget: TimingBudget = field(default_factory=TimingBudget)
    period_s: float = 1.0
    steps: int = 240
    dead_time_s: float = 50.0e-9
    afterpulse_probability: float = 0.01
    loss_drift_db: float = 1.5
    loss_step_db: float = 2.5
    loss_step_probability: float = 0.5
    misalignment_drift: float = 0.5
    signal_intensity_range: tuple[float, float] = (0.05, 1.0)
    decoy_intensity_range: tuple[float, float] = (0.01, 0.4)
    clock_range: tuple[float, float] = (1.0e8, 5.0e9)
    signal_probability_range: tuple[float, float] = (0.40, 0.85)
    minimum_decoy_probability: float = 0.05
    decoy_ratio_max: float = 0.6


@dataclass
class Transition:
    """One period: what the agent saw, what it did, what it earned."""

    state_features: NDArray[np.float64]
    action_index: int
    reward: float
    key_bits: float
    settings: DecoySettings
    excess_loss_db: float
    misalignment: float


@dataclass
class EpisodeResult:
    """One episode's outcome.

    Attributes:
        transitions: the per-period record, empty unless ``collect=True``.
        key_bits: total secret bits extracted over the episode.
        mean_key_rate_hz: those bits per second of wall clock.
        zero_key_periods: periods that produced no provable key at all — the
            failure mode that matters, and one a mean hides.
        final_settings: where the transmitter ended up.
    """

    transitions: list[Transition]
    key_bits: float
    mean_key_rate_hz: float
    zero_key_periods: int
    final_settings: DecoySettings


class KeyRateEnv:
    """A decoy-state link whose conditions move, driven one period at a time.

    The episode is a random walk in excess loss and optical misalignment, with a
    chance of one discrete loss step, over a span drawn per episode.  Nothing
    about that walk is visible to the agent except through the observables it
    measures — which is the point, and the reason a policy that reads the
    conditions directly would be answering an easier question.
    """

    def __init__(self, config: KeyRateEnvConfig | None = None,
                 rng: np.random.Generator | int | None = None):
        self.config = config or KeyRateEnvConfig()
        self.rng = np.random.default_rng(rng)
        self.settings = self.config.initial
        self._commissioned: dict[float, DecoySettings] = {}
        self.reset()

    # -- episode ------------------------------------------------------------

    def reset(self, rng: np.random.Generator | int | None = None) -> NDArray[np.float64]:
        """Start a new episode and return the first observation."""
        if rng is not None:
            self.rng = np.random.default_rng(rng)

        from dataclasses import replace

        config = self.config
        self.step_index = 0

        # The span, and with it the operating point the link was commissioned
        # at: a policy that starts every episode from the same settings on a
        # different span is being asked to recover from a mis-commissioning,
        # which is a different problem from the one under study.
        if config.lengths_km:
            length_km = float(self.rng.choice(np.asarray(config.lengths_km, dtype=float)))
            self.link = replace(config.link, length_km=length_km)
            self.settings = self._commissioned_for(length_km)
        else:
            self.link = config.link
            self.settings = config.initial

        self._excess_loss_db = self.link.excess_loss_db
        self._misalignment = self.link.misalignment_error
        self._loss_step_at = (
            int(self.rng.integers(config.steps // 4, config.steps))
            if self.rng.random() < config.loss_step_probability else -1)
        self._loss_step_size = float(self.rng.normal(config.loss_step_db,
                                                     config.loss_step_db / 3.0))

        calibrated = self.response(self.settings)
        self._calibration_gain = float(calibrated.gain(self.settings.signal_intensity))
        self._calibration_rate = decoy_key_rate_hz(observe(calibrated, self.settings))
        self._last_observables = None
        self._previous_rate_hz = self._calibration_rate

        return self.observe()

    def _commissioned_for(self, length_km: float) -> DecoySettings:
        """The commissioning point for a span, computed once and cached.

        The search is a grid over intensity and clock and costs a few hundred
        key-rate evaluations; doing it per episode would dominate training.
        """
        from dataclasses import replace

        if length_km not in self._commissioned:
            self._commissioned[length_km] = commissioned_settings(
                replace(self.config.link, length_km=length_km), self.config.initial,
                self.config.budget, self.config.dead_time_s,
                self.config.afterpulse_probability)
        return self._commissioned[length_km]

    def _conditions(self) -> LinkConditions:
        """The link as it is right now."""
        from dataclasses import replace

        return replace(self.link,
                       excess_loss_db=float(self._excess_loss_db),
                       misalignment_error=float(np.clip(self._misalignment, 0.0, 0.5)))

    def response(self, settings: DecoySettings):
        """The channel these settings see right now — see :func:`receiver_response`.

        Args:
            settings: the transmitter configuration to evaluate.

        Returns:
            ChannelResponse: ready for :func:`~qnet_autopilot.physics.decoy.observe`
            or :func:`~qnet_autopilot.physics.decoy.sample_observables`.
        """
        return receiver_response(self._conditions(), settings, self.config.budget,
                                 self.config.dead_time_s,
                                 self.config.afterpulse_probability)

    def _advance_conditions(self) -> None:
        """One period of drift in loss and misalignment."""
        config = self.config
        scale = 1.0 / np.sqrt(max(config.steps, 1))
        self._excess_loss_db = max(
            self._excess_loss_db
            + float(self.rng.normal(0.0, config.loss_drift_db * scale)), 0.0)
        self._misalignment = float(np.clip(
            self._misalignment * (1.0 + self.rng.normal(
                0.0, config.misalignment_drift * scale)),
            1e-4, 0.2))
        if self.step_index == self._loss_step_at:
            self._excess_loss_db += self._loss_step_size

    # -- observation --------------------------------------------------------

    def observe(self) -> NDArray[np.float64]:
        """What the agent sees at the start of a period.

        Fourteen numbers, all either dimensionless or logarithmic, all derived
        from the previous period's published observables and the transmitter's
        own state:

        0-2   log gain, decoy gain and vacuum gain against commissioning;
        3-4   signal and decoy error rate, in percent;
        5-6   log single-photon yield bound and its error bound, in percent;
        7     log provable key rate against commissioning;
        8-11  the current settings: mu, nu/mu, log clock, signal probability;
        12    log estimated click rate against the dead-time limit — the number
              that says how close the detectors are to saturating, which the
              agent cannot otherwise see;
        13    change in provable key rate since the previous period, the only
              memory the observation carries;
        14-15 the **absolute** operating point: log of the measured signal gain,
              and log of what the commissioning calibration says this link is
              worth in key.

    The last two are the fix for a measured failure.  With relative features
    alone the observation cannot tell a metropolitan span from a long one — a
    20% gain drop looks identical at 25 km and at 80 km — and the two regimes
    want opposite moves, because at short reach the detectors are saturating and
    at long reach the single-photon bound is starving.  A policy trained on both
    without being told which one it is in learns the average of two different
    right answers.

        Returns:
            Shape ``(KEYRATE_STATE_FEATURES,)``.
        """
        settings = self.settings
        floor = 1e-12
        if self._last_observables is None:
            gains = (1.0, 1.0, 1.0)
            errors = (self.link.misalignment_error, self.link.misalignment_error)
            single = (1.0, self.link.misalignment_error)
            rate_ratio = 1.0
            click_rate = settings.clock_rate_hz * self._calibration_gain
            absolute_gain = self._calibration_gain
        else:
            previous = self._last_observables
            calibrated = self._calibration_gain
            gains = (previous.signal_gain / max(calibrated, floor),
                     previous.decoy_gain / max(calibrated, floor),
                     previous.vacuum_gain / max(self.link.dark_count_probability, floor))
            errors = (previous.signal_error, previous.decoy_error)
            yield_bound, error_bound = single_photon_bounds(previous)
            single = (yield_bound / max(self._calibration_gain, floor), error_bound)
            rate_ratio = (decoy_key_rate_hz(previous)
                          / max(self._calibration_rate, floor))
            click_rate = settings.clock_rate_hz * previous.signal_gain
            absolute_gain = previous.signal_gain

        saturation = click_rate * self.config.dead_time_s

        return np.array([
            np.log(max(gains[0], floor)),
            np.log(max(gains[1], floor)),
            np.log(max(gains[2], floor)),
            100.0 * errors[0],
            100.0 * errors[1],
            np.log(max(single[0], floor)),
            100.0 * single[1],
            np.log(max(rate_ratio, floor)),
            settings.signal_intensity,
            settings.decoy_intensity / max(settings.signal_intensity, floor),
            np.log(settings.clock_rate_hz / C.DEFAULT_CLOCK_RATE_HZ),
            settings.signal_probability,
            np.log(max(saturation, floor)),
            np.log(max(self._previous_rate_hz, floor)
                   / max(self._calibration_rate, floor)),
            np.log(max(absolute_gain, floor)),
            np.log(max(self._calibration_rate, floor) / 1.0e6),
        ])

    # -- stepping -----------------------------------------------------------

    def apply(self, action_index: int) -> DecoySettings:
        """The settings that result from taking an action, clipped to the box."""
        config = self.config
        action = ACTIONS[action_index]
        settings = self.settings

        signal = float(np.clip(settings.signal_intensity * action.signal_factor,
                               *config.signal_intensity_range))
        decoy = float(np.clip(settings.decoy_intensity * action.decoy_factor,
                              *config.decoy_intensity_range))
        decoy = float(min(decoy, signal * config.decoy_ratio_max))
        clock = float(np.clip(settings.clock_rate_hz * action.clock_factor,
                              *config.clock_range))
        # The share of signal pulses is capped twice: by the hardware range, and
        # by leaving the weak decoy enough pulses to be measured at all. Without
        # the second cap the agent can buy raw rate until the decoy carries no
        # counts, and the bound that pays for the key stops existing.
        highest = min(config.signal_probability_range[1],
                      1.0 - settings.vacuum_probability - config.minimum_decoy_probability)
        probability = float(np.clip(
            settings.signal_probability + action.signal_probability_step,
            config.signal_probability_range[0], highest))

        return settings.replace(signal_intensity=signal, decoy_intensity=decoy,
                                clock_rate_hz=clock, signal_probability=probability)

    def step(self, action_index: int) -> tuple[NDArray[np.float64], float, bool, dict]:
        """Apply an action, run one settling period, and pay for what it produced.

        Args:
            action_index: index into :data:`ACTIONS`.

        Returns:
            ``(observation, reward, done, info)``.  The reward is the period's
            provable secret bits divided by 10\\ :sup:`6`, which keeps it near
            unity for a healthy metropolitan link — the advantage normalisation
            makes the scale irrelevant to the update, but not to the value
            function's initialisation, and starting a critic at zero against a
            target of ten million is a slow way to begin.

        Raises:
            IndexError: on an action outside :data:`ACTIONS`.
        """
        if not 0 <= action_index < len(ACTIONS):
            raise IndexError(f"action {action_index} outside 0..{len(ACTIONS) - 1}")

        self.settings = self.apply(action_index)
        response = self.response(self.settings)
        pulses = self.settings.clock_rate_hz * self.config.period_s
        observables = sample_observables(response, self.settings, pulses, rng=self.rng)

        rate_hz = decoy_key_rate_hz(observables)
        key_bits = rate_hz * self.config.period_s

        self._last_observables = observables
        self._previous_rate_hz = rate_hz
        self._advance_conditions()
        self.step_index += 1

        done = self.step_index >= self.config.steps
        info = {"key_rate_hz": rate_hz, "key_bits": key_bits,
                "settings": self.settings,
                "excess_loss_db": float(self._excess_loss_db),
                "misalignment": float(self._misalignment)}
        return self.observe(), key_bits / 1.0e6, done, info


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

class KeyRatePolicy:
    """Softmax over :data:`ACTIONS`, with a state-value critic.

    Attributes:
        actor: state features to one score per action.
        critic: state features to a value estimate.
        normaliser: running feature statistics.  On by default here, unlike
            phase 2 — the observation mixes logarithms near zero with error
            rates in percent and an intensity of order one, and the refinement
            study in phase 2 showed normalisation is what makes a mixed-scale
            observation trainable.
        temperature: softmax temperature; 1.0 is as trained.
        greedy: take the argmax instead of sampling.
    """

    def __init__(self, hidden: Sequence[int] = (64, 64),
                 rng: np.random.Generator | int | None = None,
                 name: str = "agente de taxa", normalise: bool = True):
        generator = np.random.default_rng(rng)
        self.name = name
        self.rng = generator
        self.actor = MLP([KEYRATE_STATE_FEATURES, *hidden, len(ACTIONS)],
                         rng=generator, zero_output=True)
        self.critic = MLP([KEYRATE_STATE_FEATURES, *hidden, 1], rng=generator)
        self.normaliser = FeatureNormaliser(KEYRATE_STATE_FEATURES) if normalise else None
        self.temperature = 1.0
        self.greedy = False

    # -- inference ----------------------------------------------------------

    def _encode(self, state_features: NDArray[np.float64]) -> NDArray[np.float64]:
        rows = np.atleast_2d(np.asarray(state_features, dtype=float))
        return self.normaliser.transform(rows) if self.normaliser is not None else rows

    def logits(self, state_features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.actor.forward(self._encode(state_features))[0]

    def action_probabilities(self, state_features: NDArray[np.float64],
                             temperature: float | None = None) -> NDArray[np.float64]:
        logits = self.logits(state_features)
        scale = self.temperature if temperature is None else temperature
        if scale <= 0:
            probabilities = np.zeros_like(logits)
            probabilities[int(np.argmax(logits))] = 1.0
            return probabilities
        return softmax(logits / scale)

    def act(self, state_features: NDArray[np.float64]) -> int:
        if self.greedy or self.temperature <= 0:
            return int(np.argmax(self.logits(state_features)))
        probabilities = self.action_probabilities(state_features)
        return int(self.rng.choice(len(probabilities), p=probabilities))

    def value(self, state_features: NDArray[np.float64]) -> float:
        return float(self.critic.forward(self._encode(state_features))[0, 0])

    def observe(self, state_features: NDArray[np.float64]) -> None:
        """Feed the running normaliser, during training only."""
        if self.normaliser is not None:
            self.normaliser.update(np.atleast_2d(np.asarray(state_features, dtype=float)))

    def freeze_normalisers(self) -> None:
        """Stop updating the statistics, so evaluation is deterministic."""
        if self.normaliser is not None:
            self.normaliser.freeze()

    def reset(self) -> None:
        """Nothing to clear: the policy is memoryless, the observation is not."""

    # -- training ------------------------------------------------------------

    def parameter_groups(self) -> dict[str, list[NDArray[np.float64]]]:
        return {"actor": self.actor.parameters, "critic": self.critic.parameters}

    def actor_backward(self, logit_gradient: NDArray[np.float64]
                       ) -> dict[str, list[NDArray[np.float64]]]:
        return {"actor": self.actor.backward(np.atleast_2d(logit_gradient))}

    def critic_forward(self, state_batch: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.critic.forward(self._encode(state_batch))

    def critic_backward(self, value_gradient: NDArray[np.float64]
                        ) -> dict[str, list[NDArray[np.float64]]]:
        return {"critic": self.critic.backward(value_gradient)}

    # -- serialisation -------------------------------------------------------

    def save(self, path) -> None:
        """Save the weights and the normaliser state to an ``.npz``."""
        arrays: dict[str, NDArray[np.float64]] = {}
        for index, array in enumerate(self.actor.parameters):
            arrays[f"actor_{index}"] = array
        for index, array in enumerate(self.critic.parameters):
            arrays[f"critic_{index}"] = array
        if self.normaliser is not None:
            for key, value in self.normaliser.state().items():
                arrays[f"normaliser_{key}"] = np.asarray(value)
        np.savez(path, **arrays)

    def load(self, path) -> "KeyRatePolicy":
        """Load weights saved by :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        for index, array in enumerate(self.actor.parameters):
            array[...] = data[f"actor_{index}"]
        for index, array in enumerate(self.critic.parameters):
            array[...] = data[f"critic_{index}"]
        if self.normaliser is not None:
            state = {key[len("normaliser_"):]: data[key] for key in data.files
                     if key.startswith("normaliser_")}
            if state:
                self.normaliser.load_state(state)
        return self


@dataclass
class StaticPolicy:
    """Commissioning, and then nobody touches it again.

    The settings are chosen once from the link budget — the asymptotic optimum
    at the span's commissioned loss — and held for the whole episode.  This is
    what deployed systems do, it is not a strawman, and on a link whose
    conditions stay near where they were commissioned it is very close to the
    best anything can do.

    Attributes:
        name: display name.
    """

    name: str = "estático (comissionamento)"

    def act(self, state_features: NDArray[np.float64]) -> int:
        return 0  # "manter"

    def reset(self) -> None:
        pass


@dataclass
class OraclePolicy:
    """Re-optimises every period against the channel as it truly is.

    Not deployable, and not meant to be: it reads the environment's internal
    conditions rather than the observables, and it optimises the asymptotic rate
    rather than the provable one.  The gap between it and the agent is mostly the
    part of the problem that is *information* rather than control.

    It is a reference, **not an upper bound**, and the phase-5 results contain a
    span where the agent beats it.  Two reasons, both structural: it looks one
    action ahead and takes the locally best move, so it cannot trade a worse
    period for a better next one; and it maximises the asymptotic rate while the
    score is the *provable* rate on finite counts, which prefers a slightly
    different operating point.  A true ceiling would need a dynamic program over
    the whole episode against the sampled reward, which is not something a
    controller could ever run.

    Attributes:
        env: the environment it is allowed to look inside.
        grid: candidate signal intensities.
        name: display name.
    """

    env: KeyRateEnv
    grid: Sequence[float] | None = None
    name: str = "oráculo (canal conhecido)"

    def reset(self) -> None:
        pass

    def act(self, state_features: NDArray[np.float64]) -> int:
        """Pick the action that most improves the asymptotic rate, one step ahead.

        A one-step lookahead rather than a jump to the optimum, so that it moves
        in the same action space as the agent and the comparison stays fair.
        """
        best_index, best_rate = 0, -np.inf
        for index in range(len(ACTIONS)):
            settings = self.env.apply(index)
            rate = decoy_key_rate_hz(observe(self.env.response(settings), settings))
            if rate > best_rate:
                best_index, best_rate = index, rate
        return best_index


def run_episode(env: KeyRateEnv, policy, seed: int | None = None,
                collect: bool = False) -> EpisodeResult:
    """Run one episode of a policy against the environment.

    Args:
        env: the environment; reset here, so the caller need not.
        policy: anything with ``act(state_features) -> int``.
        seed: episode seed.  Drives the conditions *and* the photon counting, so
            two policies given the same seed face the same link.
        collect: keep the per-period transitions.

    Returns:
        EpisodeResult.
    """
    state = env.reset(rng=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    transitions: list[Transition] = []
    total_bits = 0.0
    zero_periods = 0

    for _ in range(env.config.steps):
        action_index = int(policy.act(state))
        next_state, reward, done, info = env.step(action_index)
        total_bits += float(info["key_bits"])
        zero_periods += int(info["key_bits"] <= 0.0)
        if collect:
            transitions.append(Transition(
                state_features=state, action_index=action_index, reward=float(reward),
                key_bits=float(info["key_bits"]), settings=info["settings"],
                excess_loss_db=float(info["excess_loss_db"]),
                misalignment=float(info["misalignment"])))
        state = next_state
        if done:
            break

    duration_s = env.config.steps * env.config.period_s
    return EpisodeResult(transitions=transitions, key_bits=total_bits,
                         mean_key_rate_hz=total_bits / duration_s,
                         zero_key_periods=zero_periods,
                         final_settings=env.settings)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters, named as in :mod:`qnet_autopilot.agents.actor_critic`.

    Attributes:
        hidden: hidden widths, shared by actor and critic.
        learning_rate, learning_rate_end: Adam step, decayed linearly.
        gamma: discount.  Lower than phase 2's 0.995 because the reward here is
            dense and immediate — every period pays — and a horizon longer than
            the conditions' correlation time only adds variance.
        gae_lambda: GAE trace decay.
        entropy_start, entropy_end: entropy bonus, decayed linearly.
        value_coefficient: weight on the critic loss.
        batch_episodes: episodes per update.
        iterations: number of updates.
        clip_norm: gradient-norm clip.
        eval_every, eval_episodes: held-out evaluation cadence and size.
        eval_temperature: temperature to validate at.
        keep_best: return the best checkpoint on the validation seeds.
        seed: initialisation and episode seeding.
    """

    hidden: Sequence[int] = (64, 64)
    learning_rate: float = 3e-3
    learning_rate_end: float | None = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    entropy_start: float = 0.05
    entropy_end: float = 0.005
    value_coefficient: float = 0.5
    batch_episodes: int = 8
    iterations: int = 120
    clip_norm: float = 1.0
    eval_every: int = 10
    eval_episodes: int = 16
    eval_temperature: float = 1.0
    keep_best: bool = True
    seed: int = 0


@dataclass
class TrainingHistory:
    """What happened during training, for the learning curve."""

    iteration: list[int] = field(default_factory=list)
    train_key_rate_hz: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    eval_iteration: list[int] = field(default_factory=list)
    eval_key_rate_hz: list[float] = field(default_factory=list)
    best_validation: float = float("nan")

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame({
            "iteration": self.iteration,
            "train_key_rate_hz": self.train_key_rate_hz,
            "entropy": self.entropy,
            "grad_norm": self.grad_norm,
        })


def train_key_rate_policy(env_factory: Callable[[], KeyRateEnv],
                          config: TrainingConfig | None = None,
                          policy: KeyRatePolicy | None = None,
                          eval_seed: int = 700_000,
                          progress: Callable[[str], None] | None = None
                          ) -> tuple[KeyRatePolicy, TrainingHistory]:
    """Advantage actor-critic on the key-rate environment.

    Args:
        env_factory: builds a fresh environment.
        config: hyperparameters.
        policy: continue training an existing policy.
        eval_seed: base seed for validation episodes, disjoint from training.
        progress: called with a one-line summary at each evaluation.

    Returns:
        ``(policy, history)`` — the best validated checkpoint when
        ``keep_best``, otherwise the final weights.
    """
    config = config or TrainingConfig()
    policy = policy or KeyRatePolicy(hidden=config.hidden, rng=config.seed)

    groups = policy.parameter_groups()
    optimisers = {name: Adam(learning_rate=config.learning_rate,
                             clip_norm=config.clip_norm) for name in groups}

    env = env_factory()
    history = TrainingHistory()
    episode_seed = 0
    best_score = -np.inf
    best_weights: list[NDArray[np.float64]] | None = None

    for iteration in range(config.iterations):
        fraction = iteration / max(config.iterations - 1, 1)
        entropy_coefficient = (config.entropy_start
                               + fraction * (config.entropy_end - config.entropy_start))
        if config.learning_rate_end is not None:
            rate = (config.learning_rate
                    + fraction * (config.learning_rate_end - config.learning_rate))
            for optimiser in optimisers.values():
                optimiser.learning_rate = rate

        batch_states: list[NDArray[np.float64]] = []
        batch_choice: list[int] = []
        batch_advantage: list[float] = []
        batch_returns: list[float] = []
        batch_rate: list[float] = []

        policy.greedy = False
        for _ in range(config.batch_episodes):
            episode_seed += 1
            episode = run_episode(env, policy, seed=episode_seed, collect=True)

            rewards = [step.reward for step in episode.transitions]
            states = [step.state_features for step in episode.transitions]
            values = [policy.value(state) for state in states]
            advantages, returns = _generalised_advantages(
                rewards, values, config.gamma, config.gae_lambda)

            for step, advantage, target in zip(episode.transitions, advantages, returns):
                batch_states.append(step.state_features)
                batch_choice.append(step.action_index)
                batch_advantage.append(float(advantage))
                policy.observe(step.state_features)
            batch_returns.extend(returns.tolist())
            batch_rate.append(episode.mean_key_rate_hz)

        advantage_array = np.asarray(batch_advantage)
        advantage_array = ((advantage_array - advantage_array.mean())
                           / (advantage_array.std() + 1e-8))

        accumulated = {name: [np.zeros_like(array) for array in arrays]
                       for name, arrays in groups.items()}
        entropies = []

        for state_features, choice, advantage in zip(batch_states, batch_choice,
                                                     advantage_array):
            logits = policy.logits(state_features)
            probabilities = softmax(logits)
            entropy = categorical_entropy(probabilities)
            entropies.append(entropy)

            gradient = probabilities.copy()
            gradient[choice] -= 1.0
            gradient *= advantage
            log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
            gradient += entropy_coefficient * probabilities * (log_probabilities + entropy)

            for name, arrays in policy.actor_backward(gradient).items():
                for target, source in zip(accumulated[name], arrays):
                    target += source

        scale = 1.0 / max(len(batch_states), 1)
        accumulated["actor"] = [array * scale for array in accumulated["actor"]]

        state_batch = np.stack(batch_states)
        target_batch = np.asarray(batch_returns)[:, None]
        predicted = policy.critic_forward(state_batch)
        value_gradient = (2.0 * config.value_coefficient
                          * (predicted - target_batch) / len(target_batch))
        for name, arrays in policy.critic_backward(value_gradient).items():
            for target, source in zip(accumulated[name], arrays):
                target += source

        norms = {name: optimisers[name].step(groups[name], accumulated[name])
                 for name in groups}

        history.iteration.append(iteration)
        history.train_key_rate_hz.append(float(np.mean(batch_rate)))
        history.entropy.append(float(np.mean(entropies)))
        history.grad_norm.append(float(norms["actor"]))

        if config.eval_every and (iteration + 1) % config.eval_every == 0:
            score = _validate(env_factory, policy, config, eval_seed)
            history.eval_iteration.append(iteration)
            history.eval_key_rate_hz.append(score)
            if score > best_score:
                best_score = score
                best_weights = [array.copy() for array in policy.actor.parameters]
            if progress is not None:
                progress(f"[{iteration + 1:4d}/{config.iterations}] "
                         f"treino {history.train_key_rate_hz[-1] / 1e6:6.3f} Mbit/s  "
                         f"validação {score / 1e6:6.3f} Mbit/s  "
                         f"entropia {history.entropy[-1]:.3f}")

    if config.keep_best and best_weights is not None:
        for array, saved in zip(policy.actor.parameters, best_weights):
            array[...] = saved
    history.best_validation = float(best_score)
    policy.freeze_normalisers()
    return policy, history


def _validate(env_factory: Callable[[], KeyRateEnv], policy: KeyRatePolicy,
              config: TrainingConfig, eval_seed: int) -> float:
    """Mean key rate over held-out episodes, at the training temperature."""
    env = env_factory()
    temperature = policy.temperature
    policy.temperature = config.eval_temperature
    rates = [run_episode(env, policy, seed=eval_seed + index).mean_key_rate_hz
             for index in range(config.eval_episodes)]
    policy.temperature = temperature
    return float(np.mean(rates))
