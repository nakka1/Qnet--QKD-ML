"""Sessions of a decoy-state link, labelled by what was happening to them.

One row per settling period, one episode per event.  The generator's whole job
is to make the labels *hard in the right way*, and that comes down to three
decisions:

**Severities are drawn, not fixed.**  An attack with a constant strength puts a
constant offset in the features and the classifier learns the offset.  Every
episode draws its own: how much of the traffic Eve intercepts, how many single
photons she blocks, how many decibels the bend costs, how large and how fast the
temperature excursion is.  The ranges overlap deliberately at the quiet end,
where a weak attack and a warm afternoon produce the same rise in QBER.

**Every episode has a clean baseline.**  The first minute is guaranteed
event-free, so features can be expressed relative to this link's own
calibration rather than in absolute terms.  Without it a classifier trained at
50 km would be measuring span length, and would fail on a link of a different
reach for reasons that have nothing to do with security.

**Span length varies between episodes.**  Loss changes every observable in the
set — the gain, the dark count share, the tightness of the decoy bound — so a
dataset at one length teaches a model to expect one operating point.  The
lengths here run from a metro hop to the edge of what the operating point
supports.

**Attacks do not all arrive as a step, and they do not arrive alone.**  The
first version of this generator switched every attack on at full strength at a
random moment, with nothing else happening.  Both halves of that are the
favourable case.  An eavesdropper who knows a link is monitored ramps in slowly
or works in bursts, and the hardest case for a monitoring system is an attack
that starts *during* a thermal excursion, where the thing that would have
explained the error rise innocently is already happening.  So every attack
episode draws a temporal profile — step, slow ramp, or intermittent — and a
share of them also carry a simultaneous temperature excursion.  Those episodes
are labelled by the attack, because that is the answer an operator needs; the
generator records the co-occurrence so the evaluation can report it separately.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Sequence

import numpy as np
import pandas as pd

from ..physics.decoy import DecoySettings
from ..qkd.attacks import (BeamSplitting, DetectorBlinding, InterceptResend,
                           PhotonNumberSplitting)
from ..qkd.session import (EVENTS, Event, LinkConditions, SessionConfig,
                           SESSION_COLUMNS, simulate_session)
from .schema import Dataset, DatasetMetadata, describe_columns

__all__ = ["generate_qkd_session_dataset", "draw_event", "COLUMNS"]

#: What every column means.  Enforced by :class:`~.schema.Dataset`.
COLUMNS: dict[str, str] = {
    **SESSION_COLUMNS,
    "length_km": "span length of this episode",
    "severity": "the drawn magnitude of the event, in its own units",
    "profile": "how the event arrives: step, ramp or intermittent",
    "thermal_cooccurrence": "kelvin of simultaneous temperature excursion; 0 if none",
}

#: How often each temporal profile is drawn for an attack.  Steps are still the
#: plurality — a clumsy eavesdropper exists too — but they are no longer the
#: whole dataset.
PROFILE_WEIGHTS: dict[str, float] = {"step": 0.40, "ramp": 0.35, "intermittent": 0.25}

#: Share of attack episodes that also carry a temperature excursion.  The point
#: of these is not to be a class of their own; it is that the innocent
#: explanation for a rising error rate is *already true* while the attack runs.
COOCCURRENCE_PROBABILITY = 0.40


def _draw_profile(rng: np.random.Generator) -> dict:
    """Draw a temporal profile and the parameters that go with it."""
    names = list(PROFILE_WEIGHTS)
    weights = np.asarray([PROFILE_WEIGHTS[name] for name in names], dtype=float)
    profile = str(rng.choice(names, p=weights / weights.sum()))
    return {"profile": profile,
            "ramp_s": float(rng.uniform(60.0, 300.0)),
            "cycle_s": float(rng.uniform(30.0, 120.0)),
            "duty_cycle": float(rng.uniform(0.25, 0.75))}


def draw_event(label: str, rng: np.random.Generator, honest_gain: float,
               adversarial: bool = True) -> tuple[Event, float]:
    """Draw one episode's event at a random severity, profile and co-occurrence.

    Args:
        label: which class to draw.
        rng: generator.
        honest_gain: the signal gain this link produces when nothing is wrong.
            The blinding attack's click probability is drawn *relative* to it,
            because "Bob clicks more often than the channel allows" is the
            signature, and what counts as more depends on the span.
        adversarial: draw a temporal profile and a possible simultaneous thermal
            excursion for attack classes.  ``False`` reproduces the original
            step-only generator, which is what the ablation in the report
            compares against.

    Returns:
        ``(event, severity)`` — the event, and the drawn magnitude recorded in
        the dataset so a result can be read against how hard the case was.

    Raises:
        ValueError: on an unknown label.
    """
    if label == "normal":
        return Event(label="normal"), 0.0

    if label == "thermal_drift":
        ramp_k = float(rng.uniform(4.0, 14.0))
        return Event(label=label, temperature_ramp_k=ramp_k,
                     temperature_ramp_s=float(rng.uniform(60.0, 300.0))), ramp_k

    if label == "loss_event":
        step_db = float(rng.uniform(1.0, 5.0))
        return Event(label=label, excess_loss_step_db=step_db), step_db

    # --- attacks: severity, then how it arrives and what else is happening ---
    shape: dict = {}
    if adversarial:
        shape = _draw_profile(rng)
        if rng.random() < COOCCURRENCE_PROBABILITY:
            shape["temperature_ramp_k"] = float(rng.uniform(4.0, 12.0))
            shape["temperature_ramp_s"] = float(rng.uniform(60.0, 300.0))

    if label == "intercept_resend":
        fraction = float(rng.uniform(0.02, 0.30))
        return Event(label=label, attack=InterceptResend(fraction=fraction),
                     **shape), fraction

    if label == "pns":
        blocked = float(rng.uniform(0.2, 0.8))
        return Event(label=label,
                     attack=PhotonNumberSplitting(blocked_fraction=blocked),
                     **shape), blocked

    if label == "detector_blinding":
        excess = float(rng.uniform(1.05, 1.60))
        return Event(label=label,
                     attack=DetectorBlinding(click_probability=honest_gain * excess,
                                             vacuum_leak=float(rng.uniform(5e-5, 5e-4)),
                                             residual_error=float(rng.uniform(0.004, 0.02))),
                     **shape), excess

    if label == "beam_splitting":
        tapped = float(rng.uniform(0.2, 0.8))
        return Event(label=label, attack=BeamSplitting(tapped_fraction=tapped),
                     attack_is_detectable=False, **shape), tapped

    raise ValueError(f"unknown event label {label!r}")


def generate_qkd_session_dataset(
        episodes_per_class: int = 12,
        duration_s: float = 600.0,
        period_s: float = 1.0,
        baseline_periods: int = 60,
        lengths_km: Sequence[float] = (20.0, 35.0, 50.0, 70.0),
        labels: Sequence[str] | None = None,
        settings: DecoySettings | None = None,
        link: LinkConditions | None = None,
        residual_fraction: float = 0.2,
        adversarial: bool = True,
        seed: int = 0,
) -> Dataset:
    """Simulate labelled sessions of a decoy-state link.

    Args:
        episodes_per_class: independent episodes per event class.  Classes are
            balanced by construction; the deployed prior is nothing like
            balanced, which is why the evaluation reports per-class recall and a
            false-alarm rate rather than accuracy.
        duration_s: length of each episode.
        period_s: settling period — how long counts accumulate before the ends
            publish observables.
        baseline_periods: event-free periods at the start of every episode.
        lengths_km: span lengths cycled through the episodes.
        labels: classes to generate.  Defaults to every class in
            :data:`~qnet_autopilot.qkd.session.EVENTS`.
        settings: transmitter configuration; defaults to
            :class:`~qnet_autopilot.physics.decoy.DecoySettings`.
        link: link conditions; its ``length_km`` is overridden per episode.
        residual_fraction: share of the polarisation drift the compensator
            underneath fails to remove.
        adversarial: give attacks a temporal profile and a chance of a
            simultaneous thermal excursion.  ``False`` reproduces the original
            step-only generator.
        seed: base seed; episode *i* uses ``seed + i``.

    Returns:
        Dataset: one row per settling period per episode.

    Raises:
        ValueError: for a non-positive episode count, or an unknown label.
    """
    if episodes_per_class < 1:
        raise ValueError("episodes_per_class must be at least 1")

    names = tuple(labels) if labels is not None else tuple(event.label for event in EVENTS)
    unknown = set(names) - {event.label for event in EVENTS}
    if unknown:
        raise ValueError(f"unknown event labels: {sorted(unknown)}")

    settings = settings or DecoySettings()
    link = link or LinkConditions()

    frames: list[pd.DataFrame] = []
    episode = 0
    for label in names:
        for repeat in range(episodes_per_class):
            rng = np.random.default_rng(seed + episode)
            length_km = float(lengths_km[episode % len(lengths_km)])
            episode_link = replace(link, length_km=length_km)
            config = SessionConfig(link=episode_link, settings=settings,
                                   period_s=period_s, duration_s=duration_s,
                                   baseline_periods=baseline_periods,
                                   residual_fraction=residual_fraction)

            honest_gain = float(episode_link.response().gain(settings.signal_intensity))
            event, severity = draw_event(label, rng, honest_gain, adversarial)

            frame = simulate_session(config, event, rng=rng, episode=episode)
            frame["length_km"] = length_km
            frame["severity"] = severity
            frame["profile"] = event.profile
            frame["thermal_cooccurrence"] = (event.temperature_ramp_k
                                             if event.attack is not None else 0.0)
            frames.append(frame)
            episode += 1

    frame = pd.concat(frames, ignore_index=True)

    metadata = DatasetMetadata(
        name="qkd_sessions",
        description=("One row per settling period of a decoy-state BB84 link: the "
                     "observables the two ends publish, the operator's thermometer, "
                     "and the event class that was running. Columns prefixed true_ "
                     "are for evaluation only."),
        parameters={
            "episodes_per_class": episodes_per_class,
            "duration_s": duration_s,
            "period_s": period_s,
            "baseline_periods": baseline_periods,
            "lengths_km": list(lengths_km),
            "labels": list(names),
            "residual_fraction": residual_fraction,
            "adversarial": adversarial,
            "settings": asdict(settings),
            "link": asdict(link),
        },
        seed=seed,
        columns=describe_columns(COLUMNS, frame),
    )
    return Dataset(frame=frame, metadata=metadata)
