"""Phase 5: decoy-state physics, attacks, detection, key-rate control, timing.

The tests are grouped by what they are protecting:

* the **closed forms** — the gain and error formulas have textbook values the
  code must reproduce exactly, and the decoy bounds must bracket the truth they
  are bounding;
* the **attack invariants** — each attack is defined by what it does *not*
  disturb, and those are the properties that make the detection problem hard.  A
  photon-number-splitting model that moved the signal gain would make the whole
  phase-5 result meaningless, and would do it quietly;
* the **leakage guards** — the detector's feature builder must not be able to
  see the ground-truth columns, and the episode split must not put an episode on
  both sides;
* the **gradients** — the temporal convolution is hand-differentiated, so it is
  checked against finite differences, exactly as the phase-2 network is;
* the **causality** — a causal model that can see one step into the future is
  still a good regressor and a useless controller, so it is tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.physics import constants as C
from qnet_autopilot.physics.decoy import (DecoySettings, apply_detector_saturation,
                                          channel_response, decoy_key_rate,
                                          decoy_key_rate_hz, estimate_from_counts,
                                          observe, optimal_signal_intensity,
                                          poisson_weights, sample_observables,
                                          single_photon_bounds)
from qnet_autopilot.qkd.attacks import (BeamSplitting, DetectorBlinding, InterceptResend,
                                        NoAttack, PhotonNumberSplitting, attack_labels)
from qnet_autopilot.qkd.session import (EVENTS, Event, LinkConditions, SessionConfig,
                                        SESSION_COLUMNS, simulate_session)


# ---------------------------------------------------------------------------
# decoy-state physics
# ---------------------------------------------------------------------------

def test_poisson_weights_sum_to_one():
    """Truncation folds the tail into the last term rather than dropping it."""
    for intensity in (0.0, 0.1, 0.5, 1.0):
        weights = np.asarray(poisson_weights(intensity)).reshape(-1)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(weights >= 0.0)


def test_gain_matches_the_closed_form():
    """``Q_mu = 1 - (1 - Y_0) exp(-eta mu)`` for the honest channel."""
    transmittance, dark = 0.03, 1e-6
    response = channel_response(transmittance, dark, misalignment_error=0.015)
    for intensity in (0.05, 0.2, 0.5, 0.9):
        expected = 1.0 - (1.0 - dark) * np.exp(-transmittance * intensity)
        assert float(response.gain(intensity)) == pytest.approx(expected, rel=1e-9)


def test_error_rate_matches_the_closed_form():
    """``E_mu Q_mu = e_0 Y_0 + e_d (1 - exp(-eta mu))``."""
    transmittance, dark, misalignment = 0.03, 1e-5, 0.02
    response = channel_response(transmittance, dark, misalignment)
    for intensity in (0.1, 0.5):
        gain = float(response.gain(intensity))
        expected = (C.DARK_COUNT_ERROR_RATE * dark
                    + misalignment * (1.0 - np.exp(-transmittance * intensity)))
        assert float(response.error_rate(intensity)) * gain == pytest.approx(
            expected, rel=1e-9)


def test_single_photon_bounds_bracket_the_truth():
    """The bound is a bound: below the real yield, above the real error."""
    response = channel_response(0.03, 1e-6, 0.015)
    settings = DecoySettings()
    observables = observe(response, settings)
    yield_bound, error_bound = single_photon_bounds(observables)

    assert 0.0 < yield_bound <= response.yields[1] + 1e-12
    assert error_bound >= response.errors[1] - 1e-12
    # and not vacuously loose at this operating point
    assert yield_bound > 0.9 * response.yields[1]


def test_key_rate_is_zero_above_the_threshold():
    """A channel at 20% error produces no provable key, at any intensity."""
    response = channel_response(0.03, 1e-6, misalignment_error=0.20)
    settings = DecoySettings()
    assert decoy_key_rate(observe(response, settings)) == 0.0


def test_key_rate_falls_with_distance():
    """More loss, less key — monotonically, over the useful range."""
    settings = DecoySettings()
    rates = [decoy_key_rate(observe(channel_response(transmittance, 1e-6, 0.015),
                                    settings))
             for transmittance in (0.3, 0.1, 0.03, 0.01, 0.003)]
    assert all(earlier > later for earlier, later in zip(rates, rates[1:]))


def test_optimal_intensity_is_interior():
    """The optimum is a peak, not an edge of the search grid."""
    response = channel_response(0.03, 1e-6, 0.015)
    settings = DecoySettings()
    best = optimal_signal_intensity(response, settings)
    assert 0.1 < best < 1.0


def test_sampling_converges_to_the_asymptotic_observables():
    """Finite counts approach the closed form as the block grows."""
    response = channel_response(0.03, 1e-6, 0.015)
    settings = DecoySettings()
    exact = observe(response, settings)
    sampled = sample_observables(response, settings, 1e10, rng=7)
    assert sampled.signal_gain == pytest.approx(exact.signal_gain, rel=2e-3)
    assert sampled.signal_error == pytest.approx(exact.signal_error, rel=5e-2)


def test_shorter_blocks_are_noisier():
    """The single-photon bound is where the finite statistics actually bite."""
    response = channel_response(0.03, 1e-6, 0.015)
    settings = DecoySettings()
    truth, _ = single_photon_bounds(observe(response, settings))

    def spread(pulses: float) -> float:
        bounds = [single_photon_bounds(
            sample_observables(response, settings, pulses, rng=seed))[0]
            for seed in range(12)]
        return float(np.std(bounds))

    assert spread(1e7) > spread(1e10)
    assert truth > 0


def test_estimate_from_counts_rejects_impossible_counts():
    settings = DecoySettings()
    with pytest.raises(ValueError, match="exceeds"):
        estimate_from_counts(11, 10, 1, 5, 10, 1, 0, 10, settings)


def test_decoy_settings_reject_a_decoy_above_the_signal():
    with pytest.raises(ValueError, match="decoy_intensity"):
        DecoySettings(signal_intensity=0.2, decoy_intensity=0.3)


def test_saturation_reduces_yields_and_raises_error():
    """Dead time costs counts; afterpulsing costs accuracy."""
    response = channel_response(0.2, 1e-6, 0.015)
    settings = DecoySettings(clock_rate_hz=5e9)
    saturated = apply_detector_saturation(response, settings, dead_time_s=50e-9,
                                          afterpulse_probability=0.02)
    assert float(saturated.gain(0.5)) < float(response.gain(0.5))
    assert float(saturated.error_rate(0.5)) > float(response.error_rate(0.5))


def test_saturation_is_a_no_op_without_dead_time():
    response = channel_response(0.03, 1e-6, 0.015)
    settings = DecoySettings()
    same = apply_detector_saturation(response, settings, dead_time_s=0.0)
    assert np.allclose(same.yields, response.yields)


# ---------------------------------------------------------------------------
# attack invariants
# ---------------------------------------------------------------------------

def _honest():
    return channel_response(0.03, 1e-6, 0.015), DecoySettings()


def test_intercept_resend_leaves_the_gain_alone():
    """It is an error attack. A gain signature would be a modelling artefact."""
    response, settings = _honest()
    attacked = InterceptResend(fraction=0.3).apply(response, settings)
    assert np.allclose(attacked.yields, response.yields)
    assert float(attacked.error_rate(settings.signal_intensity)) > float(
        response.error_rate(settings.signal_intensity))


def test_intercept_resend_adds_a_quarter_at_full_strength():
    """Full interception costs 25% on top of half the honest error."""
    response, settings = _honest()
    attacked = InterceptResend(fraction=1.0).apply(response, settings)
    expected = 0.5 * response.errors[1] + 0.25
    assert attacked.errors[1] == pytest.approx(expected, rel=1e-12)


def test_pns_preserves_the_signal_gain_and_the_error():
    """The whole point: invisible in Q_mu and E_mu, visible in the decoy."""
    response, settings = _honest()
    attacked = PhotonNumberSplitting(blocked_fraction=0.5).apply(response, settings)

    assert float(attacked.gain(settings.signal_intensity)) == pytest.approx(
        float(response.gain(settings.signal_intensity)), rel=1e-9)
    # The error moves, but by parts per million: shifting weight from single- to
    # multi-photon detections changes how much of the error is dark counts, and
    # nothing else. Three parts in 10^6 of a 1.5% error rate is four orders below
    # the sampling noise of any real block, so the attack is invisible in E_mu
    # for every purpose except this assertion.
    assert float(attacked.error_rate(settings.signal_intensity)) == pytest.approx(
        float(response.error_rate(settings.signal_intensity)), abs=1e-5)
    assert float(attacked.gain(settings.decoy_intensity)) < float(
        response.gain(settings.decoy_intensity))


def test_pns_collapses_the_single_photon_bound():
    """Decoy states catch what the gain and the error rate cannot."""
    response, settings = _honest()
    attacked = PhotonNumberSplitting(blocked_fraction=0.6).apply(response, settings)

    honest_bound, _ = single_photon_bounds(observe(response, settings))
    attacked_bound, _ = single_photon_bounds(observe(attacked, settings))
    assert attacked_bound < 0.6 * honest_bound
    assert decoy_key_rate(observe(attacked, settings)) < decoy_key_rate(
        observe(response, settings))


def test_uncompensated_pns_shows_up_in_the_gain():
    """Without the compensation the attack is visible in Q_mu — a worse attack."""
    response, settings = _honest()
    attacked = PhotonNumberSplitting(blocked_fraction=0.5, compensate=False).apply(
        response, settings)
    assert float(attacked.gain(settings.signal_intensity)) < float(
        response.gain(settings.signal_intensity))


def test_blinding_breaks_the_transmittance_and_the_vacuum():
    """Two independent tells, and no honest fault produces either."""
    response, settings = _honest()
    attacked = DetectorBlinding(click_probability=0.05, vacuum_leak=2e-4).apply(
        response, settings)
    assert float(attacked.gain(settings.signal_intensity)) > float(
        response.gain(settings.signal_intensity))
    assert attacked.yields[0] > response.yields[0]
    assert attacked.errors[0] < C.DARK_COUNT_ERROR_RATE


def test_beam_splitting_is_indistinguishable_by_construction():
    """The honest negative control: nothing Bob can measure changes."""
    response, settings = _honest()
    attacked = BeamSplitting(tapped_fraction=0.7).apply(response, settings)
    assert np.allclose(attacked.yields, response.yields)
    assert np.allclose(attacked.errors, response.errors)
    assert attacked.label == "beam_splitting"


def test_every_attack_has_a_distinct_label():
    labels = attack_labels()
    assert len(labels) == len(set(labels))
    assert "normal" in labels


def test_no_attack_returns_the_channel_unchanged():
    response, settings = _honest()
    assert NoAttack().apply(response, settings) is response


def test_strength_zero_is_indistinguishable_from_no_attack():
    """The contract that makes ramps and bursts honest: dialled to zero, nothing."""
    response, settings = _honest()
    for attack in (InterceptResend(0.3), PhotonNumberSplitting(0.6),
                   DetectorBlinding(click_probability=0.05), BeamSplitting(0.5)):
        quiet = attack.with_strength(0.0).apply(response, settings)
        assert np.allclose(quiet.yields, response.yields), attack.label
        assert np.allclose(quiet.errors, response.errors), attack.label


def test_strength_is_monotone():
    """Half an attack disturbs less than a whole one, and more than none."""
    response, settings = _honest()
    attack = InterceptResend(0.4)
    errors = [float(attack.with_strength(weight).apply(response, settings)
                    .error_rate(settings.signal_intensity))
              for weight in (0.0, 0.5, 1.0)]
    assert errors[0] < errors[1] < errors[2]

    splitting = PhotonNumberSplitting(0.8)
    yields = [float(splitting.with_strength(weight).apply(response, settings)
                    .gain(settings.decoy_intensity))
              for weight in (0.0, 0.5, 1.0)]
    assert yields[0] > yields[1] > yields[2]


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def _small_session(**overrides) -> SessionConfig:
    parameters = dict(link=LinkConditions(length_km=40.0), period_s=1.0,
                      duration_s=60.0, baseline_periods=20, min_active_periods=20)
    parameters.update(overrides)
    return SessionConfig(**parameters)


def test_session_log_has_exactly_the_documented_columns():
    frame = simulate_session(_small_session(), EVENTS[0], rng=0)
    assert list(frame.columns) == list(SESSION_COLUMNS)
    assert len(frame) == 60


def test_normal_episodes_have_no_onset():
    frame = simulate_session(_small_session(), Event(label="normal"), rng=1)
    assert not np.isfinite(frame["onset_s"].iloc[0])
    assert frame["active"].sum() == 0


def test_the_onset_leaves_room_for_the_event_to_run():
    config = _small_session()
    for seed in range(8):
        frame = simulate_session(config, EVENTS[3], rng=seed)
        active = int(frame["active"].sum())
        assert active >= config.min_active_periods - 1


def test_a_loss_event_drops_both_gains_without_moving_the_error():
    frame = simulate_session(_small_session(),
                             Event(label="loss_event", excess_loss_step_db=3.0), rng=3)
    before = frame[frame["active"] == 0]
    after = frame[frame["active"] == 1]
    assert after["signal_gain"].mean() < 0.7 * before["signal_gain"].mean()
    assert after["signal_error"].mean() == pytest.approx(
        before["signal_error"].mean(), abs=0.01)


def test_a_ramped_event_arrives_gradually():
    """A ramp must not look like a step in the first periods after the onset."""
    config = _small_session(duration_s=200.0, baseline_periods=40,
                            min_active_periods=80)
    event = Event(label="intercept_resend",
                  attack=InterceptResend(fraction=0.5), profile="ramp", ramp_s=60.0)
    frame = simulate_session(config, event, rng=5)
    active = frame[frame["active"] == 1]
    early = active.head(5)["strength"].mean()
    late = active.tail(5)["strength"].mean()
    assert early < 0.3 < late
    assert active["strength"].is_monotonic_increasing


def test_an_intermittent_event_switches_off_between_bursts():
    config = _small_session(duration_s=200.0, baseline_periods=40,
                            min_active_periods=80)
    event = Event(label="intercept_resend", attack=InterceptResend(fraction=0.5),
                  profile="intermittent", cycle_s=20.0, duty_cycle=0.5)
    frame = simulate_session(config, event, rng=6)
    after = frame[frame["time_s"] >= frame["onset_s"].iloc[0]]
    share = float(after["strength"].mean())
    assert 0.3 < share < 0.7
    assert set(np.unique(after["strength"])) == {0.0, 1.0}


def test_an_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="unknown profile"):
        Event(label="normal", profile="whenever")


def test_an_attack_can_run_during_a_thermal_excursion():
    """The adverse case: the innocent explanation is already true."""
    config = _small_session(duration_s=200.0, baseline_periods=40,
                            min_active_periods=80)
    event = Event(label="intercept_resend", attack=InterceptResend(fraction=0.2),
                  temperature_ramp_k=8.0)
    frame = simulate_session(config, event, rng=7)
    before = frame[frame["active"] == 0]
    after = frame[frame["active"] == 1]
    assert after["temperature_c"].mean() > before["temperature_c"].mean() + 1.0
    assert after["signal_error"].mean() > before["signal_error"].mean()


def test_a_session_is_reproducible():
    first = simulate_session(_small_session(), EVENTS[1], rng=11)
    second = simulate_session(_small_session(), EVENTS[1], rng=11)
    assert np.allclose(first["signal_gain"], second["signal_gain"])


def test_a_session_too_short_for_its_baseline_is_refused():
    with pytest.raises(ValueError, match="leaves no room"):
        SessionConfig(duration_s=30.0, baseline_periods=60, min_active_periods=60)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def _detection_frame(episodes: int = 2):
    import pandas as pd

    frames = []
    config = _small_session(duration_s=120.0, baseline_periods=40,
                            min_active_periods=40)
    labels = ("normal", "intercept_resend", "pns")
    episode = 0
    for label in labels:
        event = next(item for item in EVENTS if item.label == label)
        for _ in range(episodes):
            frames.append(simulate_session(config, event, rng=episode, episode=episode))
            episode += 1
    return pd.concat(frames, ignore_index=True)


def test_features_never_see_the_ground_truth():
    from qnet_autopilot.qkd.detection import (FEATURE_NAMES, FORBIDDEN_COLUMNS,
                                              FeatureWindow, build_detection_features)

    frame = _detection_frame()
    window = FeatureWindow(window=10, baseline_periods=40)
    features, labels, index = build_detection_features(frame, window)

    assert features.shape[1] == len(FEATURE_NAMES)
    for column in FORBIDDEN_COLUMNS:
        assert column not in FEATURE_NAMES
        assert column not in index.columns
    assert np.all(np.isfinite(features))


def test_rows_before_the_onset_are_labelled_normal():
    from qnet_autopilot.qkd.detection import (FeatureWindow, build_detection_features)

    frame = _detection_frame()
    _, labels, index = build_detection_features(frame, FeatureWindow(window=10,
                                                                    baseline_periods=40))
    early = index["time_s"].to_numpy() < index["onset_s"].to_numpy()
    assert np.all(labels[early] == "normal")


def test_the_baseline_periods_are_dropped():
    from qnet_autopilot.qkd.detection import (FeatureWindow, build_detection_features)

    frame = _detection_frame()
    window = FeatureWindow(window=10, baseline_periods=40)
    _, _, index = build_detection_features(frame, window)
    assert index["period"].min() >= window.baseline_periods


def test_episode_split_keeps_episodes_whole():
    from qnet_autopilot.qkd.detection import (FeatureWindow, build_detection_features,
                                              episode_split)

    frame = _detection_frame(episodes=3)
    _, _, index = build_detection_features(frame, FeatureWindow(window=10,
                                                               baseline_periods=40))
    train, test = episode_split(index, 0.7, seed=0)
    assert not np.any(train & test)
    assert set(index["episode"][train]).isdisjoint(set(index["episode"][test]))


def test_hysteresis_suppresses_isolated_calls():
    """One noisy period is not a ticket; three in a row is."""
    import pandas as pd

    from qnet_autopilot.qkd.detection import SustainedDetector

    class _Fixed:
        name = "fixo"

        def __init__(self, calls):
            self.calls = np.asarray(calls)

        def predict(self, features):
            return self.calls

    calls = ["normal", "pns", "normal", "pns", "pns", "pns", "normal"]
    index = pd.DataFrame({"episode": [0] * len(calls),
                          "period": list(range(len(calls)))})
    detector = SustainedDetector(base=_Fixed(calls), periods=3)
    out = detector.predict_sequence(np.zeros((len(calls), 1)), index)

    assert list(out) == ["normal", "normal", "normal", "normal", "normal",
                         "pns", "normal"]


def test_a_zero_threshold_reproduces_the_argmax():
    """The decision threshold may only suppress an alarm, never invent one."""
    from qnet_autopilot.qkd.detection import (FeatureWindow, build_detection_features,
                                              episode_split, fit_boosted)

    frame = _detection_frame(episodes=3)
    features, labels, index = build_detection_features(
        frame, FeatureWindow(window=10, baseline_periods=40))
    train, _ = episode_split(index, 0.7, seed=0)
    detector = fit_boosted(features, labels, train, iterations=30, seed=0)

    plain = detector.predict(features)
    detector.attack_threshold = 0.0
    assert list(detector.predict(features)) == list(plain)

    detector.attack_threshold = 1.1  # above any probability
    gated = detector.predict(features)
    assert set(np.unique(gated)) <= {"normal"}


def test_feature_window_refuses_a_baseline_shorter_than_the_window():
    from qnet_autopilot.qkd.detection import FeatureWindow

    with pytest.raises(ValueError, match="baseline_periods"):
        FeatureWindow(window=30, baseline_periods=10)


# ---------------------------------------------------------------------------
# key-rate control
# ---------------------------------------------------------------------------

def test_actions_stay_inside_the_hardware_box():
    from qnet_autopilot.qkd.keyrate import ACTIONS, KeyRateEnv, KeyRateEnvConfig

    config = KeyRateEnvConfig(steps=20)
    env = KeyRateEnv(config, rng=0)
    for _ in range(40):
        for index in range(len(ACTIONS)):
            settings = env.apply(index)
            assert (config.signal_intensity_range[0] <= settings.signal_intensity
                    <= config.signal_intensity_range[1])
            assert settings.decoy_intensity < settings.signal_intensity
            assert (config.clock_range[0] <= settings.clock_rate_hz
                    <= config.clock_range[1])
            assert settings.decoy_probability >= config.minimum_decoy_probability - 1e-9
        env.step(int(np.random.default_rng(_).integers(len(ACTIONS))))


def test_the_environment_ends_after_its_steps():
    from qnet_autopilot.qkd.keyrate import KeyRateEnv, KeyRateEnvConfig

    env = KeyRateEnv(KeyRateEnvConfig(steps=5), rng=0)
    done = False
    for _ in range(5):
        _, reward, done, info = env.step(0)
        assert reward >= 0.0
        assert info["key_rate_hz"] >= 0.0
    assert done


def test_the_observation_has_the_declared_width():
    from qnet_autopilot.qkd.keyrate import KEYRATE_STATE_FEATURES, KeyRateEnv

    env = KeyRateEnv(rng=0)
    state = env.reset(rng=0)
    assert state.shape == (KEYRATE_STATE_FEATURES,)
    assert np.all(np.isfinite(state))
    state, _, _, _ = env.step(1)
    assert np.all(np.isfinite(state))


def test_a_rejected_action_index_is_loud():
    from qnet_autopilot.qkd.keyrate import ACTIONS, KeyRateEnv

    env = KeyRateEnv(rng=0)
    with pytest.raises(IndexError):
        env.step(len(ACTIONS))


def test_the_policy_round_trips_through_a_file(tmp_path):
    from qnet_autopilot.qkd.keyrate import KeyRatePolicy

    from qnet_autopilot.qkd.keyrate import KEYRATE_STATE_FEATURES

    policy = KeyRatePolicy(rng=0)
    state = np.linspace(-1.0, 1.0, KEYRATE_STATE_FEATURES)
    before = policy.action_probabilities(state).copy()
    path = tmp_path / "policy.npz"
    policy.save(path)

    restored = KeyRatePolicy(rng=1).load(path)
    assert np.allclose(restored.action_probabilities(state), before)


def test_commissioning_beats_an_arbitrary_setting():
    from qnet_autopilot.qkd.keyrate import (commissioned_settings, receiver_response)
    from qnet_autopilot.physics.decoy import observe as observe_settings

    link = LinkConditions(length_km=50.0)
    commissioned = commissioned_settings(link)
    naive = DecoySettings(signal_intensity=0.05, decoy_intensity=0.02,
                          clock_rate_hz=1e8)

    def rate(settings):
        return decoy_key_rate_hz(observe_settings(
            receiver_response(link, settings), settings))

    assert rate(commissioned) > rate(naive)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def test_timing_spread_grows_with_distance():
    from qnet_autopilot.qkd.timing import TimingBudget, timing_spread

    budget = TimingBudget()
    spreads = [timing_spread(budget, length) for length in (10, 50, 100, 200)]
    assert all(earlier < later for earlier, later in zip(spreads, spreads[1:]))


def test_gate_efficiency_peaks_at_zero_offset():
    from qnet_autopilot.qkd.timing import gate_efficiency

    perfect = float(gate_efficiency(0.0, 500e-12, 100e-12))
    off = float(gate_efficiency(400e-12, 500e-12, 100e-12))
    assert 0.9 < perfect <= 1.0
    assert off < 0.25


def test_intersymbol_error_grows_with_the_clock():
    from qnet_autopilot.qkd.timing import intersymbol_error

    errors = [float(intersymbol_error(rate, 100e-12))
              for rate in (1e8, 1e9, 2e9, 5e9)]
    assert all(earlier < later for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 0.5


def test_thermal_delay_follows_the_coefficient():
    from qnet_autopilot.qkd.timing import ClockModel

    model = ClockModel(length_km=50.0, fractional_offset=0.0, frequency_diffusion=0.0,
                       diurnal_amplitude_k=0.0, temperature_noise_k=0.0)
    temperature = np.linspace(20.0, 25.0, 100)
    trajectory = model.simulate(100.0, 1.0, temperature_c=temperature, rng=0)
    expected = model.thermal_coefficient_s_per_km_k * 50.0 * 5.0
    assert trajectory.offset_s[-1] == pytest.approx(expected, rel=1e-9)


def test_tracking_beats_doing_nothing():
    from qnet_autopilot.qkd.timing import (AlphaBetaTracker, ClockModel, GateConfig,
                                           NoCorrection, run_sync_loop)

    trajectory = ClockModel(length_km=50.0).simulate(1800.0, 1.0, rng=3)
    gate = GateConfig(width_s=500e-12, update_period_s=1.0, latency_s=2.0)
    tracked = run_sync_loop(trajectory, AlphaBetaTracker(), gate, rng=0).summary()
    ignored = run_sync_loop(trajectory, NoCorrection(), gate, rng=0).summary()
    assert tracked["mean_efficiency"] > 5.0 * ignored["mean_efficiency"]


def test_the_gate_loop_refuses_a_fractional_latency():
    from qnet_autopilot.qkd.timing import GateConfig

    with pytest.raises(ValueError, match="integer multiple"):
        GateConfig(update_period_s=1.0, latency_s=2.5)


# ---------------------------------------------------------------------------
# the temporal compensator
# ---------------------------------------------------------------------------

def _temporal_net(window: int = 16):
    from qnet_autopilot.compensation.predictors import FeatureSpec
    from qnet_autopilot.compensation.sequence import SequenceSpec, TemporalConvNet

    feature_spec = FeatureSpec(window=window, use_covariates=True, horizon_scale=2e-3)
    spec = SequenceSpec(feature_spec=feature_spec, channels=8, kernel=3,
                        dilations=(1, 2), hidden=12)
    return TemporalConvNet(spec, rng=0), feature_spec


def test_temporal_gradients_match_finite_differences():
    """The backward pass is hand-written, so it is checked, like phase 2's."""
    network, feature_spec = _temporal_net()
    rng = np.random.default_rng(0)
    network.output_weight[...] = rng.normal(0.0, 0.3, network.output_weight.shape)

    features = rng.normal(size=(4, feature_spec.size))
    targets = rng.normal(size=(4, 3))

    def loss(flat) -> float:
        network.set_flat(flat)
        return float(np.mean((network.forward(features) - targets) ** 2))

    flat = network.get_flat().copy()
    network.set_flat(flat)
    residual = network.forward(features) - targets
    gradients = np.concatenate([array.ravel() for array in
                                network.backward(2.0 * residual / residual.size)])

    step = 1e-6
    for index in rng.choice(len(flat), 25, replace=False):
        forward = flat.copy()
        forward[index] += step
        backward = flat.copy()
        backward[index] -= step
        numeric = (loss(forward) - loss(backward)) / (2.0 * step)
        assert numeric == pytest.approx(gradients[index], abs=1e-6)


def test_the_convolution_is_causal():
    """Changing the oldest lag must not change a prediction that predates it.

    The feature vector runs newest-first, so the *last* entries of the sequence
    block are the oldest lags. A model that let them influence the output would
    still regress well and would be reading the future in the loop.
    """
    from qnet_autopilot.compensation.sequence import CausalConv1d

    layer = CausalConv1d(2, 3, kernel=3, dilation=2, rng=0)
    sequence = np.random.default_rng(1).normal(size=(1, 8, 2))
    original = layer.forward(sequence)

    disturbed = sequence.copy()
    disturbed[0, -1, :] += 10.0  # the newest step, which only the last output sees
    changed = layer.forward(disturbed)

    assert np.allclose(original[:, :-1, :], changed[:, :-1, :])
    assert not np.allclose(original[:, -1, :], changed[:, -1, :])


def test_an_untrained_temporal_model_predicts_no_increment():
    """Zero-initialised output means the model starts as persistence."""
    network, feature_spec = _temporal_net()
    features = np.random.default_rng(2).normal(size=(3, feature_spec.size))
    assert np.allclose(network.forward(features), 0.0)


def test_a_receptive_field_wider_than_the_window_is_refused():
    from qnet_autopilot.compensation.predictors import FeatureSpec
    from qnet_autopilot.compensation.sequence import fit_sequence
    from qnet_autopilot.compensation.train import Dataset

    spec = FeatureSpec(window=4, use_covariates=True, horizon_scale=1.0)
    dataset = Dataset(features=np.zeros((4, spec.size)), targets=np.zeros((4, 3)),
                      episode=np.zeros(4, dtype=np.int64), time_s=np.arange(4.0),
                      spec=spec)
    with pytest.raises(ValueError, match="receptive field"):
        fit_sequence(dataset, np.ones(4, dtype=bool), dilations=(1, 2, 4, 8))


def test_the_temporal_model_drops_into_the_phase_three_loop():
    """It must be scored by the same closed loop, or the comparison is not one."""
    from qnet_autopilot.compensation import run_closed_loop
    from qnet_autopilot.compensation.closed_loop import ControlConfig
    from qnet_autopilot.compensation.sequence import fit_sequence
    from qnet_autopilot.compensation.train import build_dataset, chronological_split
    from qnet_autopilot.physics.fiber import PolarizationDriftModel

    model = PolarizationDriftModel(vibration_amplitude_rad=0.05)
    control = ControlConfig(control_period_s=2e-3, latency_s=4e-3)
    trajectories = [model.simulate(2.0, 1e-3, rng=seed, track_rotation=True)
                    for seed in range(2)]
    dataset = build_dataset(trajectories, control,
                            __import__("qnet_autopilot.compensation.predictors",
                                       fromlist=["FeatureSpec"]).FeatureSpec(
                                window=8, use_covariates=True, horizon_scale=2e-3),
                            rng=0)
    train, validation = chronological_split(dataset, 0.7, 0.05)
    compensator, history = fit_sequence(dataset, train, validation, channels=8,
                                        dilations=(1, 2), hidden=16, epochs=3, seed=0)

    result = run_closed_loop(trajectories[0], compensator, control, rng=0)
    assert len(history) == 3
    assert np.all(np.isfinite(result.qber))
    assert 0.0 <= float(np.mean(result.qber)) <= 1.0
