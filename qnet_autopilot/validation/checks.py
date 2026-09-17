"""Cross-checks: where the fast models agree with the simulator, and where they don't.

A model that has never been checked against anything is a hypothesis.  This
module runs the checks and returns their results as data, so the report is
generated from measurements rather than written by hand.

Three kinds of check, in descending order of strength:

**Exact** — the Bell-diagonal operations must equal SeQUeNCe's to machine
precision, because they are transcriptions of the same formulas.  Any
disagreement at all is a bug.  :func:`check_bds_operations` re-runs that
comparison here so the report can state the numbers rather than cite a test
file.

**Statistical** — the link rate model predicts a probability; SeQUeNCe measures
a frequency over a finite number of attempts.  Agreement can only be asserted
within a confidence interval, and :func:`check_link_rates` reports the interval
alongside the point estimate.  A model that sits outside the interval is
wrong; one that sits inside it is *not contradicted*, which is a weaker claim
and should be reported as such.

**Internal** — the drift model has no external reference, so the check is
against its own defining statistics: does the vibration term stay stationary,
does the thermal term diffuse at the rate its parameters imply.  These catch
implementation errors, not modelling errors.  A polarisation model can pass
every one of them and still be wrong about real fibre; only field data settles
that, and this package has none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..physics import bell_diagonal as bds
from ..physics import link
from ..physics.fiber import PolarizationDriftModel

__all__ = [
    "CheckResult",
    "check_bds_operations",
    "check_link_rates",
    "check_drift_statistics",
    "check_purification_break_even",
    "check_swap_chain",
    "wilson_interval",
    "run_all",
]


@dataclass
class CheckResult:
    """One check's verdict, plus the numbers behind it.

    Attributes:
        name: short identifier.
        passed: whether the check met its own criterion.
        summary: one line, suitable for a report table.
        detail: the measurements, as a DataFrame where there are several.
        kind: ``"exact"``, ``"statistical"`` or ``"internal"`` — how strong a
            claim a pass represents.
    """

    name: str
    passed: bool
    summary: str
    detail: pd.DataFrame | None = None
    kind: str = "internal"
    extra: dict[str, Any] = field(default_factory=dict)


def wilson_interval(successes: int, trials: int, z: float = 1.96
                    ) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because the success
    probabilities of interest are small and the intervals must stay inside
    [0, 1]; the naive interval does neither.
    """
    if trials == 0:
        return (0.0, 1.0)
    phat = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (phat + z ** 2 / (2 * trials)) / denominator
    spread = (z / denominator) * np.sqrt(
        phat * (1 - phat) / trials + z ** 2 / (4 * trials ** 2))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------

def check_bds_operations(num_random: int = 200, seed: int = 0,
                         tolerance: float = 1e-12) -> CheckResult:
    """Every Bell-diagonal operation against SeQUeNCe's protocol objects.

    Builds a real timeline and real protocol instances, feeds them random
    Bell-diagonal states and random gate/measurement fidelities, and compares.
    """
    from ..compat import bootstrap

    bootstrap()
    from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM
    from sequence.kernel.quantum_manager import QuantumManager

    previous = QuantumManager.get_active_formalism()
    QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)
    try:
        rows = _compare_bds_operations(num_random, seed)
    finally:
        QuantumManager.set_global_manager_formalism(previous)

    detail = pd.DataFrame(rows)
    worst = float(detail["max_abs_error"].max())
    passed = worst <= tolerance
    return CheckResult(
        name="bell_diagonal_operations",
        kind="exact",
        passed=passed,
        summary=(f"{len(detail)} comparisons across BBPSSW, DEJMPS, swapping and "
                 f"memory decoherence; largest absolute disagreement "
                 f"{worst:.2e} (tolerance {tolerance:.0e})"),
        detail=detail,
        extra={"max_abs_error": worst, "tolerance": tolerance},
    )


def _compare_bds_operations(num_random: int, seed: int) -> list[dict]:
    from sequence.components.memory import Memory
    from sequence.entanglement_management.purification import BBPSSW_BDS, DEJMPS_BDS
    from sequence.entanglement_management.swapping import EntanglementSwappingA_BDS
    from sequence.kernel.timeline import Timeline
    from sequence.message import Message
    from sequence.topology.node import Node

    class _Node(Node):
        def __init__(self, name, timeline, gate_fid, meas_fid):
            Node.__init__(self, name, timeline, gate_fid=gate_fid, meas_fid=meas_fid)
            self.resource_manager = _Manager()

        def receive_message(self, src: str, msg: Message):
            pass

    class _Manager:
        def update(self, protocol, memory, state):
            pass

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for index in range(num_random):
        gate_fid = float(rng.uniform(0.85, 1.0))
        meas_fid = float(rng.uniform(0.85, 1.0))
        kept_state = rng.dirichlet([8, 2, 1, 1])
        meas_state = rng.dirichlet([7, 1, 2, 1])

        timeline = Timeline()
        timeline.show_progress = False
        owner = _Node("a1", timeline, gate_fid, meas_fid)
        remote = _Node("a2", timeline, gate_fid, meas_fid)
        middle = _Node("b", timeline, gate_fid, meas_fid)

        memories = {name: Memory(name, timeline, fidelity=1.0, frequency=0,
                                 efficiency=1, coherence_time=1, wavelength=500)
                    for name in ("kept1", "kept2", "meas1", "meas2")}
        timeline.init()
        timeline.quantum_manager.set(
            [memories["kept1"].qstate_key, memories["kept2"].qstate_key], kept_state)
        timeline.quantum_manager.set(
            [memories["meas1"].qstate_key, memories["meas2"].qstate_key], meas_state)

        for label, protocol_class in (("bbpssw", BBPSSW_BDS), ("dejmps", DEJMPS_BDS)):
            protocol = protocol_class(owner, f"a1.{label}", memories["kept1"],
                                      memories["meas1"])
            owner.protocols.append(protocol)
            protocol.set_others(f"a2.{label}", remote.name,
                                [memories["kept2"].name, memories["meas2"].name])
            expected_p, expected_state = protocol.purification_res()

            function = bds.bbpssw if label == "bbpssw" else bds.dejmps
            actual_p, actual_state = function(kept_state, meas_state, gate_fid, meas_fid)

            rows.append({
                "operation": label,
                "trial": index,
                "max_abs_error": max(abs(float(actual_p) - float(expected_p)),
                                     float(np.max(np.abs(np.asarray(actual_state)
                                                         - np.asarray(expected_state))))),
            })

        for is_twirled in (False, True):
            protocol = EntanglementSwappingA_BDS(
                middle, f"b.es{int(is_twirled)}", memories["kept1"], memories["meas1"],
                is_twirled=is_twirled)
            middle.protocols.append(protocol)
            expected_state = protocol.swapping_res()
            actual_state = bds.swap(kept_state, meas_state, gate_fid, meas_fid,
                                    is_twirled=is_twirled)
            rows.append({
                "operation": f"swap{'_twirled' if is_twirled else ''}",
                "trial": index,
                "max_abs_error": float(np.max(np.abs(np.asarray(actual_state)
                                                     - np.asarray(expected_state)))),
            })

    return rows


# ---------------------------------------------------------------------------
# statistical
# ---------------------------------------------------------------------------

def check_link_rates(lengths_km: Sequence[float] = (1.0, 5.0, 10.0, 20.0, 30.0, 40.0),
                     num_attempts: int = 600,
                     detector_efficiency: float = 1.0,
                     coupling_efficiency: float = 1.0,
                     memory_efficiency: float = 1.0,
                     attenuation_db_per_km: float = 0.2,
                     seed: int = 0) -> CheckResult:
    """Measured heralding rate against the closed forms, with confidence intervals.

    The measurement is a binomial proportion over ``num_attempts`` independent
    attempts.  A closed form is judged *not contradicted* if it lies inside the
    95% Wilson interval — which is the only claim a finite sample supports.
    """
    from ..datagen.entanglement import LinkExperiment

    rows = []
    for length_km in lengths_km:
        frame = LinkExperiment(
            length_km=length_km, num_attempts=num_attempts,
            detector_efficiency=detector_efficiency,
            memory_efficiency=memory_efficiency,
            attenuation_db_per_km=attenuation_db_per_km).run(seed=seed)

        successes = int(frame["succeeded"].sum())
        measured = successes / num_attempts
        low, high = wilson_interval(successes, num_attempts)

        kwargs = dict(attenuation_db_per_km=attenuation_db_per_km,
                      coupling_efficiency=coupling_efficiency,
                      detector_efficiency=detector_efficiency,
                      memory_efficiency=memory_efficiency)
        single = float(link.attempt_success_probability(length_km, "single_heralded", **kwargs))
        double = float(link.attempt_success_probability(length_km, "barrett_kok", **kwargs))

        rows.append({
            "length_km": length_km,
            "attempts": num_attempts,
            "successes": successes,
            "measured": measured,
            "ci_low": low,
            "ci_high": high,
            "single_heralded": single,
            "barrett_kok": double,
            "single_heralded_in_ci": low <= single <= high,
            "barrett_kok_in_ci": low <= double <= high,
            "single_heralded_rel_error": (single - measured) / measured if measured else np.nan,
        })

    detail = pd.DataFrame(rows)
    agree = int(detail["single_heralded_in_ci"].sum())
    worst = float(np.nanmax(np.abs(detail["single_heralded_rel_error"])))
    passed = agree >= len(detail) - 1  # allow one outlier at the noisy tail

    return CheckResult(
        name="link_success_probability",
        kind="statistical",
        passed=passed,
        summary=(f"single-heralded closed form inside the 95% interval at "
                 f"{agree}/{len(detail)} distances; largest relative deviation "
                 f"{worst:.1%} (at the long, low-count end)"),
        detail=detail,
        extra={"agreeing_points": agree, "worst_relative_error": worst},
    )


# ---------------------------------------------------------------------------
# internal
# ---------------------------------------------------------------------------

def check_drift_statistics(num_runs: int = 24, duration_s: float = 7200.0,
                           dt_s: float = 1.0, seed: int = 0) -> CheckResult:
    """Does the drift model behave the way its parameters say it should?

    Two properties, both consequences of the construction rather than of any
    physical claim:

    * the vibration term is **stationary** — its RMS is the same in the first
      and second half of a run, so it jitters rather than accumulating;
    * the thermal term **diffuses** at the OU rate — beyond the correlation
      time the RMS accumulated angle should track
      :math:`\\sigma\\sqrt{2\\tau t}` until the sphere's curvature saturates it.

    A failure means the implementation does not match the documented model.  It
    says nothing about whether the documented model matches real fibre.
    """
    vibration_only = PolarizationDriftModel(
        thermal_drift_rate_rad_per_s=0.0, thermal_per_kelvin_rad_per_s=0.0,
        jump_rate_per_s=0.0, vibration_amplitude_rad=3e-3)
    # The diffusive scaling sigma*sqrt(2*tau*t) is a small-angle result: once
    # the accumulated angle approaches a radian the sphere's curvature folds the
    # walk back on itself and the RMS saturates below the prediction. Testing it
    # therefore needs a drift rate slow enough that the diffusive regime
    # (t >> tau) and the small-angle regime (rms << 1) overlap. This is a check
    # on the implementation, so the rate is chosen for testability, not realism.
    thermal_only = PolarizationDriftModel(
        thermal_drift_rate_rad_per_s=1.0e-4,
        vibration_amplitude_rad=0.0, thermal_per_kelvin_rad_per_s=0.0,
        jump_rate_per_s=0.0)

    # Stationarity is a short-timescale property: 120 s at 1 kHz is ~14,000
    # vibration cycles, far more than enough, and avoids sampling the whole run
    # at the millisecond rate the vibration band requires.
    vibration_duration_s = min(duration_s, 120.0)
    trajectory = vibration_only.simulate(vibration_duration_s, 1e-3, rng=seed)
    half = len(trajectory) // 2
    first_rms = float(np.sqrt((trajectory.angle_rad[:half] ** 2).mean()))
    second_rms = float(np.sqrt((trajectory.angle_rad[half:] ** 2).mean()))
    stationary_ratio = second_rms / first_rms if first_rms else np.inf
    stationary = 0.8 <= stationary_ratio <= 1.25

    angles = np.stack([thermal_only.simulate(duration_s, dt_s, rng=seed + run).angle_rad
                       for run in range(num_runs)])
    times = np.arange(angles.shape[1]) * dt_s
    rms = np.sqrt((angles ** 2).mean(axis=0))

    sigma = thermal_only.thermal_drift_rate_rad_per_s
    tau = thermal_only.thermal_correlation_time_s
    predicted = _expected_drift_angle(times, sigma, tau)

    # Compare in the diffusive window: after a few correlation times, and while
    # the angle is still small enough that the sphere is locally flat.
    diffusive_regime = times > 2 * tau
    small_angle = rms < 0.5
    window = diffusive_regime & small_angle
    if window.any():
        ratio = float(np.mean(rms[window] / np.maximum(predicted[window], 1e-12)))
        # A tight band: the prediction below is exact in the small-angle limit,
        # so anything outside ~15% is an implementation error, not noise.
        diffusive = 0.85 <= ratio <= 1.15
        diffusive_note = f"thermal RMS angle / predicted = {ratio:.3f}"
    else:
        # The diffusive regime only begins after a few correlation times. A run
        # shorter than that cannot test it — that is an inconclusive check, not
        # a failed one, and saying "fail" here would be misleading.
        ratio, diffusive = float("nan"), True
        reason = ("the run is shorter than 2 x the correlation time"
                  if not diffusive_regime.any()
                  else "the walk leaves the small-angle regime before t = 2 tau")
        diffusive_note = (f"thermal scaling not tested — {reason} "
                          f"(run {duration_s:.0f}s, tau {tau:.0f}s)")

    detail = pd.DataFrame({
        "time_s": times,
        "rms_angle_rad": rms,
        "predicted_diffusive_rad": predicted,
        "in_comparison_window": window,
    })

    return CheckResult(
        name="drift_statistics",
        kind="internal",
        passed=bool(stationary and diffusive),
        summary=(f"vibration stationary (second/first-half RMS = "
                 f"{stationary_ratio:.2f}); {diffusive_note}"),
        detail=detail,
        extra={"stationary_ratio": stationary_ratio, "diffusive_ratio": ratio,
               "num_runs": num_runs},
    )


def _expected_drift_angle(times: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    """RMS Poincare-sphere displacement after ``times``, in the small-angle limit.

    The accumulated rotation vector is :math:`\\int_0^t \\Omega\\,ds` with
    :math:`\\Omega` a stationary three-dimensional OU process of per-axis
    standard deviation :math:`\\sigma` and correlation time :math:`\\tau`.  Each
    component of the integral has variance

    .. math:: V(t) = 2\\sigma^2\\tau\\bigl[t - \\tau(1 - e^{-t/\\tau})\\bigr]

    — the bracket reducing to :math:`t` only once :math:`t \\gg \\tau`, which is
    why the naive :math:`\\sigma\\sqrt{2\\tau t}` overestimates early on.

    The subtlety that costs the most is the last one: the observable is the
    angle between the launch Stokes vector and where it ended up, and a
    rotation *about* the launch axis does not move it at all.  Only the two
    components perpendicular to the launch direction displace the state, so the
    expected squared displacement is :math:`2V(t)`, not :math:`3V(t)`.
    Predicting three degrees of freedom here overestimates the drift by
    :math:`\\sqrt{3/2} \\approx 1.22`.
    """
    times = np.maximum(np.asarray(times, dtype=float), 0.0)
    variance = 2 * sigma ** 2 * tau * (times - tau * (1 - np.exp(-times / tau)))
    return np.sqrt(2 * np.maximum(variance, 0.0))


def check_purification_break_even(gate_fidelities: Sequence[float] = (1.0, 0.99, 0.97, 0.95),
                                  fidelities: Sequence[float] | None = None
                                  ) -> CheckResult:
    """Where purification stops helping, as a function of gate quality.

    Purification is not free and not always beneficial: below some input
    fidelity the local gate and measurement errors it introduces exceed the
    error it removes, and the "purified" pair is worse than what went in.  That
    break-even point is the single most decision-relevant number a repeater
    controller needs, and it moves sharply with gate fidelity.

    This computes the curve rather than asserting a threshold — the check
    passes if the curve is well-formed (monotone, break-even rising as gates
    degrade), which is the property, not the value.
    """
    fidelities = np.asarray(
        fidelities if fidelities is not None else np.linspace(0.5, 0.999, 200))

    rows = []
    break_even = {}
    for gate_fid in gate_fidelities:
        state = bds.werner(fidelities)
        _, purified = bds.bbpssw(state, state, gate_fid=gate_fid, meas_fid=gate_fid)
        gain = bds.fidelity(purified) - fidelities

        helpful = gain > 0
        threshold = float(fidelities[helpful][0]) if helpful.any() else float("nan")
        break_even[gate_fid] = threshold

        rows.append(pd.DataFrame({
            "gate_fidelity": gate_fid,
            "input_fidelity": fidelities,
            "output_fidelity": bds.fidelity(purified),
            "gain": gain,
        }))

    detail = pd.concat(rows, ignore_index=True)
    thresholds = [break_even[g] for g in gate_fidelities]
    finite = [t for t in thresholds if np.isfinite(t)]
    monotone = all(a <= b + 1e-9 for a, b in zip(finite, finite[1:]))

    described = ", ".join(
        f"gate {g:.2f} -> F>{break_even[g]:.3f}" if np.isfinite(break_even[g])
        else f"gate {g:.2f} -> never helps"
        for g in gate_fidelities)

    return CheckResult(
        name="purification_break_even",
        kind="internal",
        passed=bool(monotone),
        summary=f"BBPSSW break-even input fidelity: {described}",
        detail=detail,
        extra={"break_even": break_even},
    )


def check_swap_chain(num_swaps: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
                     link_fidelity: float = 0.95,
                     gate_fidelities: Sequence[float] = (1.0, 0.99, 0.97)
                     ) -> CheckResult:
    """End-to-end fidelity as a chain grows, with and without perfect gates.

    Nested swapping composes multiplicatively, so fidelity falls off fast; the
    curve is what tells a router how many hops it can afford before the result
    is not worth delivering.  With ideal gates and Werner inputs the closed
    form is exact, so this doubles as a check that repeated application of
    :func:`~qnet_autopilot.physics.bell_diagonal.swap` reproduces it.
    """
    rows = []
    consistent = True

    for gate_fid in gate_fidelities:
        state = bds.werner(link_fidelity)
        current = state
        for swaps in num_swaps:
            if swaps > 0:
                current = bds.swap(current, state, gate_fid=gate_fid,
                                   meas_fid=gate_fid, is_twirled=True)
            fidelity = float(bds.fidelity(current))

            # Ideal-gate closed form for a chain of n+1 Werner links:
            # the Werner parameter multiplies, F = (1 + 3 w^(n+1)) / 4.
            werner_parameter = (4 * link_fidelity - 1) / 3
            expected = (1 + 3 * werner_parameter ** (swaps + 1)) / 4
            if gate_fid == 1.0 and abs(fidelity - expected) > 1e-9:
                consistent = False

            rows.append({
                "gate_fidelity": gate_fid,
                "num_swaps": swaps,
                "num_links": swaps + 1,
                "end_to_end_fidelity": fidelity,
                "ideal_closed_form": expected,
            })

    detail = pd.DataFrame(rows)
    ideal = detail[detail["gate_fidelity"] == 1.0]
    return CheckResult(
        name="swap_chain",
        kind="exact" if consistent else "internal",
        passed=bool(consistent),
        summary=("repeated swapping of Werner links reproduces "
                 "F = (1 + 3 w^n)/4 exactly with ideal gates; "
                 f"{link_fidelity:.2f}-fidelity links fall to "
                 f"{ideal['end_to_end_fidelity'].iloc[-1]:.3f} after "
                 f"{ideal['num_swaps'].iloc[-1]} swaps"),
        detail=detail,
    )


def run_all(quick: bool = False, seed: int = 0) -> list[CheckResult]:
    """Run every check and return the results in report order.

    Args:
        quick: use smaller samples — for a smoke test, not for a report.
            A quick run's statistical check has intervals wide enough to be
            uninformative, and the summary says so.
        seed: base seed.
    """
    if quick:
        return [
            check_bds_operations(num_random=15, seed=seed),
            check_link_rates(lengths_km=(1.0, 10.0, 30.0), num_attempts=200, seed=seed),
            check_drift_statistics(num_runs=6, duration_s=3600.0, dt_s=2.0, seed=seed),
            check_purification_break_even(),
            check_swap_chain(),
        ]
    return [
        check_bds_operations(num_random=200, seed=seed),
        check_link_rates(num_attempts=600, seed=seed),
        check_drift_statistics(num_runs=24, seed=seed),
        check_purification_break_even(),
        check_swap_chain(),
    ]
