"""Telling an attack from the weather, from the public record alone.

The problem, stated honestly
----------------------------
A deployed QKD link already has an eavesdropping detector: it stops producing
key when the QBER crosses the security threshold.  That detector is sound and
almost useless operationally.  It fires long after a weak attack has been
running, it fires for a warm afternoon, and when it fires it says nothing about
what to do — a bend needs a technician, an intercept-and-resend needs the link
taken down, and a warm afternoon needs neither.

So the question is not "is the QBER high".  It is "the observables moved; which
of the seven things that move them is this", and the answer has to come from
what the two ends publish anyway, plus a thermometer.

What the features are, and why relative
---------------------------------------
Every feature is expressed against **this episode's own commissioning
baseline** — the median of the event-free periods at the start — and not in
absolute units.  Three reasons, and they are the difference between a model that
transfers and one that does not:

* An absolute gain is mostly a statement about span length.  A model trained on
  absolute gains at 50 km learns 50 km.
* The interesting quantities are *ratios that should not have changed*.  Photon
  number splitting keeps :math:`Q_\\mu` exactly where it was and cannot keep
  :math:`Q_\\nu / Q_\\mu` there; detector blinding pushes the gain above what the
  calibrated loss allows.  Both are invisible in levels and obvious in ratios.
* Every link is commissioned.  Using the calibration is not a modelling
  convenience, it is what an operator has.

Beside the ratios sit two windowed features that carry the thermal
discrimination: the correlation between the error rate and the *rate of change*
of temperature over the trailing window, and the slope of the error rate itself.
Polarisation drift is driven by dT/dt — that is the coupling in
:class:`~qnet_autopilot.physics.fiber.PolarizationDriftModel` and it is measured
in deployed fibre — so a thermal excursion produces an error rise that is
correlated with the thermometer and decays when the temperature settles.  Eve
has no such correlation.  This is the one feature in the set that a QBER
threshold cannot have, and the report measures how much of the separation it is
carrying.

What is not a feature
---------------------
The true misalignment, the true transmittance, the true single-photon yield and
the ``active`` flag.  They are in the session log for scoring and are stripped
here; :func:`build_detection_features` raises if one reaches the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..compensation.predictors import Standardiser

__all__ = [
    "FEATURE_NAMES", "FORBIDDEN_COLUMNS", "FeatureWindow",
    "build_detection_features", "episode_split",
    "QberThreshold", "AttackDetector", "fit_forest", "fit_svm", "fit_boosted",
    "SustainedDetector", "DetectionReport", "evaluate_detector",
]

#: The feature vector, in order.  Named here rather than inferred from the frame
#: so that a model saved today cannot be fed a differently-ordered matrix
#: tomorrow.
FEATURE_NAMES: tuple[str, ...] = (
    "log_signal_gain_ratio",
    "log_decoy_gain_ratio",
    "decoy_to_signal_shift",
    "signal_error_shift",
    "decoy_error_shift",
    "log_vacuum_gain_ratio",
    "vacuum_error_shift",
    "log_single_photon_yield_ratio",
    "single_photon_error_shift",
    "log_key_rate_ratio",
    "log_gain_versus_calibration",
    "temperature_shift",
    "temperature_rate",
    "error_temperature_correlation",
    "error_trend",
    "gain_trend",
    "error_volatility",
)

#: Columns a feature builder must never see.  The first four are ground truth
#: the receiver does not have; ``active`` is the answer.
FORBIDDEN_COLUMNS: tuple[str, ...] = (
    "true_misalignment", "true_transmittance", "true_single_photon_yield", "active",
)

#: Added to gains and rates before taking a logarithm.  A settling period can
#: legitimately observe zero vacuum clicks, and ``log 0`` is not a feature.
_FLOOR = 1e-12


@dataclass(frozen=True)
class FeatureWindow:
    """How much history each feature vector is built from.

    Attributes:
        window: trailing periods used for the windowed features — the trend,
            the volatility and the error-temperature correlation.  Long enough
            to see a thermal excursion develop, short enough that the detection
            delay stays operationally useful; 30 periods at one second is half
            a minute.
        baseline_periods: periods at the start of the episode used as the
            commissioning calibration.  Must match what the session was
            generated with, or the baseline will contain the event.
    """

    window: int = 30
    baseline_periods: int = 60

    def __post_init__(self) -> None:
        if self.window < 3:
            raise ValueError("window must be at least 3 for a trend to mean anything")
        if self.baseline_periods < self.window:
            raise ValueError(
                "baseline_periods must cover at least one window, or the first "
                "feature vectors would mix calibration and event")


def build_detection_features(frame: pd.DataFrame,
                             window: FeatureWindow | None = None
                             ) -> tuple[NDArray[np.float64], NDArray[np.str_], pd.DataFrame]:
    """Turn a session log into a feature matrix, a label per row, and its index.

    Args:
        frame: the output of
            :func:`~qnet_autopilot.qkd.session.simulate_session`, or several of
            them concatenated.
        window: history lengths; defaults to :class:`FeatureWindow`.

    Returns:
        ``(features, labels, index)``.  ``features`` has one column per entry of
        :data:`FEATURE_NAMES`; ``labels`` is the class in force at that period —
        a row before the onset is labelled ``"normal"``, because it is;
        ``index`` carries ``episode``, ``time_s``, ``period``, ``onset_s``,
        ``episode_label`` and ``severity`` for scoring and splitting.

    Raises:
        ValueError: if the frame is missing a column the features need.

    Note:
        Rows before ``baseline_periods`` are dropped: they *are* the
        calibration, and scoring a model on its own baseline would report the
        false-alarm rate of a tautology.
    """
    window = window or FeatureWindow()
    required = {"episode", "time_s", "period", "label", "onset_s", "signal_gain",
                "decoy_gain", "vacuum_gain", "signal_error", "decoy_error",
                "vacuum_error", "single_photon_yield_lower",
                "single_photon_error_upper", "key_rate_hz", "temperature_c",
                "expected_gain"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"session log is missing columns: {sorted(missing)}")

    blocks: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.str_]] = []
    indices: list[pd.DataFrame] = []

    for episode, episode_frame in frame.groupby("episode", sort=True):
        episode_frame = episode_frame.sort_values("period")
        features, index = _episode_features(episode_frame, window)
        if len(index) == 0:
            continue
        blocks.append(features)
        labels.append(index["label"].to_numpy())
        indices.append(index)

    if not blocks:
        raise ValueError("no episode was long enough to produce a feature vector")

    return (np.concatenate(blocks, axis=0),
            np.concatenate(labels, axis=0),
            pd.concat(indices, ignore_index=True))


def _episode_features(frame: pd.DataFrame, window: FeatureWindow
                      ) -> tuple[NDArray[np.float64], pd.DataFrame]:
    """Features for one episode, against its own baseline."""
    baseline = frame.iloc[:window.baseline_periods]
    if len(baseline) < window.baseline_periods:
        return np.empty((0, len(FEATURE_NAMES))), frame.iloc[:0].copy()

    def level(column: str) -> float:
        return float(np.median(baseline[column].to_numpy()))

    signal_gain = frame["signal_gain"].to_numpy()
    decoy_gain = frame["decoy_gain"].to_numpy()
    vacuum_gain = frame["vacuum_gain"].to_numpy()
    signal_error = frame["signal_error"].to_numpy()
    decoy_error = frame["decoy_error"].to_numpy()
    vacuum_error = frame["vacuum_error"].to_numpy()
    single_yield = frame["single_photon_yield_lower"].to_numpy()
    single_error = frame["single_photon_error_upper"].to_numpy()
    key_rate = frame["key_rate_hz"].to_numpy()
    temperature = frame["temperature_c"].to_numpy()
    expected_gain = frame["expected_gain"].to_numpy()
    period_s = float(np.median(np.diff(frame["time_s"].to_numpy()))) if len(frame) > 1 else 1.0

    baseline_ratio = level("decoy_gain") / max(level("signal_gain"), _FLOOR)
    ratio = decoy_gain / np.maximum(signal_gain, _FLOOR)

    temperature_rate = np.gradient(temperature, period_s)

    columns = {
        "log_signal_gain_ratio": _log_ratio(signal_gain, level("signal_gain")),
        "log_decoy_gain_ratio": _log_ratio(decoy_gain, level("decoy_gain")),
        "decoy_to_signal_shift": ratio / max(baseline_ratio, _FLOOR) - 1.0,
        "signal_error_shift": signal_error - level("signal_error"),
        "decoy_error_shift": decoy_error - level("decoy_error"),
        "log_vacuum_gain_ratio": _log_ratio(vacuum_gain, level("vacuum_gain")),
        "vacuum_error_shift": vacuum_error - level("vacuum_error"),
        "log_single_photon_yield_ratio": _log_ratio(single_yield,
                                                    level("single_photon_yield_lower")),
        "single_photon_error_shift": single_error - level("single_photon_error_upper"),
        "log_key_rate_ratio": _log_ratio(key_rate, level("key_rate_hz")),
        "log_gain_versus_calibration": _log_ratio(signal_gain, 1.0) - _log_ratio(expected_gain, 1.0),
        "temperature_shift": temperature - level("temperature_c"),
        "temperature_rate": temperature_rate,
        "error_temperature_correlation": _rolling_correlation(
            signal_error, np.abs(temperature_rate), window.window),
        "error_trend": _rolling_slope(signal_error, window.window) * period_s,
        "gain_trend": _rolling_slope(_log_ratio(signal_gain, level("signal_gain")),
                                     window.window) * period_s,
        "error_volatility": _rolling_std(signal_error, window.window),
    }

    features = np.stack([columns[name] for name in FEATURE_NAMES], axis=-1)

    keep = frame["period"].to_numpy() >= window.baseline_periods
    index = pd.DataFrame({
        "episode": frame["episode"].to_numpy()[keep],
        "period": frame["period"].to_numpy()[keep],
        "time_s": frame["time_s"].to_numpy()[keep],
        "onset_s": frame["onset_s"].to_numpy()[keep],
        "episode_label": frame["label"].to_numpy()[keep],
        "severity": (frame["severity"].to_numpy()[keep] if "severity" in frame
                     else np.zeros(int(keep.sum()))),
        "label": np.where(_is_active(frame)[keep],
                          frame["label"].to_numpy()[keep], "normal"),
    })
    return features[keep], index


def _is_active(frame: pd.DataFrame) -> NDArray[np.bool_]:
    """Whether the event is running, from the onset time — never from ``active``.

    The ``active`` column is the answer and is on the forbidden list, but the
    onset time is what defines the label, so it is read here and nowhere else.
    """
    onset = frame["onset_s"].to_numpy()
    time_s = frame["time_s"].to_numpy()
    return np.where(np.isfinite(onset), time_s >= onset, False)


def _log_ratio(values: NDArray[np.float64], reference: float) -> NDArray[np.float64]:
    return np.log((np.asarray(values, dtype=float) + _FLOOR) / (reference + _FLOOR))


def _rolling_slope(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Least-squares slope over a trailing window, per sample, zero-padded at the start."""
    values = np.asarray(values, dtype=float)
    lags = np.arange(window, dtype=float)
    centred = lags - lags.mean()
    denominator = float(np.sum(centred ** 2))
    out = np.zeros_like(values)
    for index in range(window - 1, len(values)):
        segment = values[index - window + 1: index + 1]
        out[index] = float(np.sum(centred * (segment - segment.mean())) / denominator)
    return out


def _rolling_std(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    for index in range(window - 1, len(values)):
        out[index] = float(np.std(values[index - window + 1: index + 1]))
    return out


def _rolling_correlation(first: NDArray[np.float64], second: NDArray[np.float64],
                         window: int) -> NDArray[np.float64]:
    """Pearson correlation over a trailing window, zero where either side is flat."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    out = np.zeros_like(first)
    for index in range(window - 1, len(first)):
        left = first[index - window + 1: index + 1]
        right = second[index - window + 1: index + 1]
        left_spread = float(np.std(left))
        right_spread = float(np.std(right))
        if left_spread < 1e-15 or right_spread < 1e-15:
            continue
        out[index] = float(np.mean((left - left.mean()) * (right - right.mean()))
                           / (left_spread * right_spread))
    return out


def episode_split(index: pd.DataFrame, train_fraction: float = 0.7,
                  seed: int = 0) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split rows by **episode**, stratified by class.

    Rows within an episode share a baseline, a span length and a drift
    realisation, so splitting by row puts near-duplicates on both sides and
    reports a number that means nothing — the same failure the phase-3
    chronological split exists to avoid.

    Args:
        index: the index frame from :func:`build_detection_features`.
        train_fraction: share of episodes per class used for training.
        seed: shuffling seed.

    Returns:
        ``(train_mask, test_mask)`` over the rows of ``index``.
    """
    rng = np.random.default_rng(seed)
    train_episodes: set[int] = set()

    pairs = index[["episode", "episode_label"]].drop_duplicates()
    for _, group in pairs.groupby("episode_label", sort=True):
        episodes = np.array(group["episode"].to_numpy(), copy=True)
        rng.shuffle(episodes)
        cut = max(int(round(train_fraction * len(episodes))), 1)
        train_episodes.update(int(value) for value in episodes[:cut])

    episodes = index["episode"].to_numpy()
    train = np.isin(episodes, list(train_episodes))
    return train, ~train


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

@dataclass
class QberThreshold:
    """What a deployed link does today: watch the QBER, alarm when it rises.

    The comparison is against the episode's own baseline, in units of the
    baseline's own sampling noise, which is the strongest form of this baseline
    — a fixed 11% threshold would never fire on anything in these datasets
    except the heaviest attacks, and comparing against it would be a straw man.

    It answers a binary question and cannot do otherwise: a QBER rise has no
    type.  Everything it flags is reported as ``"attack"``, and
    :func:`evaluate_detector` scores it on the binary task only.

    Attributes:
        sigma: how many standard deviations above the baseline error rate counts
            as an alarm.  Set it with :meth:`calibrate` rather than by hand: an
            arbitrary value makes this baseline look bad for a reason that has
            nothing to do with what it can and cannot see.
        name: display name.
    """

    sigma: float = 5.0
    name: str = "limiar de QBER"

    @classmethod
    def calibrate(cls, features: NDArray[np.float64], labels: NDArray[np.str_],
                  train_mask: NDArray[np.bool_],
                  false_alarm_target: float = 0.01,
                  name: str = "limiar de QBER") -> "QberThreshold":
        """Choose ``sigma`` on the training rows to hit a false-alarm budget.

        This is what an operator does with an alarm threshold: pick the budget
        for nuisance tickets, then set the knob to it.  Calibrating on the same
        rows the learned models are fitted on, and scoring on the same held-out
        episodes, is what makes the comparison a comparison.

        Args:
            features, labels: from :func:`build_detection_features`.
            train_mask: the training rows.
            false_alarm_target: allowed share of event-free training periods
                that may fire.
            name: display name.

        Returns:
            QberThreshold: with ``sigma`` at the chosen quantile.

        Raises:
            ValueError: if the training rows contain no event-free periods.
        """
        event_free = train_mask & (labels == "normal")
        if not event_free.any():
            raise ValueError("no event-free training periods to calibrate against")
        scores = cls._scores(features[event_free])
        sigma = float(np.quantile(scores, 1.0 - false_alarm_target))
        return cls(sigma=sigma, name=name)

    @staticmethod
    def _scores(features: NDArray[np.float64]) -> NDArray[np.float64]:
        shift = features[:, FEATURE_NAMES.index("signal_error_shift")]
        spread = features[:, FEATURE_NAMES.index("error_volatility")]
        return shift / np.maximum(spread, 1e-6)

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.str_]:
        """Alarm on the standardised error shift.

        The feature matrix carries ``signal_error_shift`` and
        ``error_volatility``; their ratio is the z-score this baseline needs,
        and using the same matrix as the learned models keeps the comparison
        honest.
        """
        return np.where(self._scores(features) > self.sigma, "attack", "normal")


@dataclass
class AttackDetector:
    """A fitted classifier plus the scaling and the feature order it was fitted with.

    Attributes:
        model: anything with ``predict(X)``; a scikit-learn classifier here.
        standardiser: the scaling fitted on the training rows.
        classes: the labels the model can emit.
        name: display name.
        feature_names: the column order the model expects.
        attack_threshold: minimum probability mass on the non-normal classes
            before an attack is called at all.  Zero takes the model's own
            argmax.  Anything above it trades recall for precision, and that
            trade is the whole operational question — see
            :meth:`calibrate_threshold`.
    """

    model: object
    standardiser: Standardiser
    classes: tuple[str, ...]
    name: str = "aprendido"
    feature_names: tuple[str, ...] = FEATURE_NAMES
    attack_threshold: float = 0.0

    def attack_probability(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Probability mass the model puts on *something being wrong*.

        Raises:
            AttributeError: if the underlying model has no ``predict_proba``.
        """
        proba = self.model.predict_proba(self.standardiser.transform(features))
        classes = list(self.classes)
        if "normal" not in classes:
            return np.ones(len(features))
        return 1.0 - np.asarray(proba)[:, classes.index("normal")]

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.str_]:
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{self.name} was fitted on {len(self.feature_names)} features, "
                f"got {features.shape[1]}")
        scaled = self.standardiser.transform(features)
        if self.attack_threshold <= 0.0 or not hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict(scaled))

        proba = np.asarray(self.model.predict_proba(scaled))
        classes = list(self.classes)
        best = np.asarray(self.model.predict(scaled))
        if "normal" not in classes:
            return best

        # The threshold only ever *suppresses* an alarm the model already wanted
        # to raise; it never creates one. A threshold of zero therefore
        # reproduces the plain argmax exactly, which is the property that makes
        # the calibrated and uncalibrated detectors comparable.
        attack_mass = 1.0 - proba[:, classes.index("normal")]
        quiet = (best != "normal") & (attack_mass < self.attack_threshold)
        return np.where(quiet, "normal", best)

    def calibrate_threshold(self, features: NDArray[np.float64],
                            labels: NDArray[np.str_],
                            mask: NDArray[np.bool_],
                            false_alarm_target: float = 0.005) -> "AttackDetector":
        """Set :attr:`attack_threshold` to a false-alarm budget, in place.

        The balanced class weights every model here is fitted with are right for
        recall and wrong for precision: they are calibrated to a generator whose
        classes are balanced, and a deployed link's are not.  Rather than
        refitting, the decision threshold is moved until the share of event-free
        periods that fire matches the budget — which is exactly what is already
        done to the QBER baseline, so the two remain comparable.

        Args:
            features, labels: from :func:`build_detection_features`.
            mask: rows to calibrate on — the *training* episodes.
            false_alarm_target: allowed share of event-free periods that fire.

        Returns:
            ``self``, so it can be chained.

        Raises:
            ValueError: if there are no event-free rows to calibrate on.
        """
        event_free = mask & (labels == "normal")
        if not event_free.any():
            raise ValueError("no event-free rows to calibrate the threshold on")
        if not hasattr(self.model, "predict_proba"):
            return self

        # Only the rows the model already flags can be suppressed, so the budget
        # is spent on those: keep the ``target * n_clean`` most confident of
        # them and put the threshold just above the rest.
        previous = self.attack_threshold
        self.attack_threshold = 0.0
        flagged = self.predict(features[event_free]) != "normal"
        self.attack_threshold = previous

        clean_count = int(event_free.sum())
        allowed = int(round(false_alarm_target * clean_count))
        if not flagged.any() or allowed >= int(flagged.sum()):
            self.attack_threshold = 0.0
            return self

        scores = np.sort(self.attack_probability(features[event_free])[flagged])[::-1]
        self.attack_threshold = float(scores[allowed])
        return self

    def importances(self) -> dict[str, float] | None:
        """Per-feature importance, where the model exposes one."""
        values = getattr(self.model, "feature_importances_", None)
        if values is None:
            return None
        return dict(zip(self.feature_names, (float(value) for value in values)))


def fit_forest(features: NDArray[np.float64], labels: NDArray[np.str_],
               train_mask: NDArray[np.bool_],
               trees: int = 300, max_depth: int | None = None,
               seed: int = 0, name: str = "random forest") -> AttackDetector:
    """Fit a random forest on the training rows.

    The forest is the right first model here and not a placeholder: the
    signatures are threshold-shaped and interact (a gain drop *with* an error
    rise is a bend, a gain drop *without* one is not), which is what a tree
    ensemble represents natively and what a linear model cannot represent at
    all.  It also reports feature importances, and those are the part of this
    result worth reading — they say which physics the separation actually rests
    on.

    Args:
        features, labels: from :func:`build_detection_features`.
        train_mask: rows to fit on, from :func:`episode_split`.
        trees: number of trees.
        max_depth: depth cap; ``None`` grows them out.
        seed: forest seed.
        name: display name.

    Returns:
        AttackDetector: fitted.
    """
    from sklearn.ensemble import RandomForestClassifier

    standardiser = Standardiser().fit(features[train_mask])
    model = RandomForestClassifier(n_estimators=trees, max_depth=max_depth,
                                   random_state=seed, n_jobs=-1,
                                   class_weight="balanced")
    model.fit(standardiser.transform(features[train_mask]), labels[train_mask])
    return AttackDetector(model=model, standardiser=standardiser,
                          classes=tuple(model.classes_), name=name)


def fit_svm(features: NDArray[np.float64], labels: NDArray[np.str_],
            train_mask: NDArray[np.bool_],
            regularisation: float = 10.0, gamma: str | float = "scale",
            seed: int = 0, name: str = "SVM (RBF)") -> AttackDetector:
    """Fit an RBF support-vector classifier on the training rows.

    Included because it is the other model the QKD literature reaches for, and
    because it fails differently: an RBF kernel on standardised features draws
    smooth boundaries, which suits the classes that differ by a direction in
    feature space (intercept-and-resend versus thermal drift) and suits the ones
    that differ by a *conjunction* of thresholds much less well.

    Args:
        features, labels: from :func:`build_detection_features`.
        train_mask: rows to fit on.
        regularisation: the ``C`` of the soft margin.
        gamma: RBF width.
        seed: only affects tie-breaking.
        name: display name.

    Returns:
        AttackDetector: fitted.
    """
    from sklearn.svm import SVC

    standardiser = Standardiser().fit(features[train_mask])
    model = SVC(C=regularisation, gamma=gamma, class_weight="balanced",
                random_state=seed, cache_size=500)
    model.fit(standardiser.transform(features[train_mask]), labels[train_mask])
    return AttackDetector(model=model, standardiser=standardiser,
                          classes=tuple(model.classes_), name=name)


def fit_boosted(features: NDArray[np.float64], labels: NDArray[np.str_],
                train_mask: NDArray[np.bool_],
                iterations: int = 400, learning_rate: float = 0.06,
                max_leaf_nodes: int = 31, seed: int = 0,
                name: str = "gradient boosting") -> AttackDetector:
    """Fit a histogram gradient-boosting classifier on the training rows.

    Where the forest averages independent trees, boosting fits each tree to what
    the previous ones got wrong — which is the right shape for the class this
    dataset is hardest on.  A weak intercept-and-resend differs from thermal
    drift by a *small, consistent* displacement in two features at once, and the
    forest splits noise instead of finding it; boosting keeps working on
    precisely the rows it is still missing.

    Args:
        features, labels: from :func:`build_detection_features`.
        train_mask: rows to fit on, from :func:`episode_split`.
        iterations: boosting rounds.
        learning_rate: shrinkage per round.
        max_leaf_nodes: tree size.
        seed: random state.
        name: display name.

    Returns:
        AttackDetector: fitted.

    Note:
        Class weights are balanced here as they are in the forest, because the
        generator's balance is an artefact and the deployed prior is nothing
        like it.  What that costs is precision under the real prior, which
        :meth:`DetectionReport.precision_at_prior` reports rather than hides.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    standardiser = Standardiser().fit(features[train_mask])
    model = HistGradientBoostingClassifier(
        max_iter=iterations, learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes, random_state=seed,
        class_weight="balanced", early_stopping=False)
    model.fit(standardiser.transform(features[train_mask]), labels[train_mask])
    return AttackDetector(model=model, standardiser=standardiser,
                          classes=tuple(model.classes_), name=name)


@dataclass
class SustainedDetector:
    """Requires an alarm to persist before it counts as one.

    A per-period classifier answers every second independently, and a deployed
    monitoring system does not raise a ticket on one noisy second.  This wraps
    any detector and suppresses a non-normal call unless the *same* call repeats
    for ``periods`` consecutive settling periods within the same episode.

    It is not a way of making the numbers look better: it trades detection delay
    for false alarms, at a rate the report quotes both halves of.  What it
    genuinely fixes is the shape of the error — an isolated misfire on a healthy
    link is the failure an operator notices, and a three-second rule removes
    almost all of them while costing two seconds on a real attack.

    Attributes:
        base: the underlying detector.
        periods: consecutive agreeing periods required.
        name: display name; derived from the base detector when left empty.
    """

    base: object
    periods: int = 3
    name: str = ""

    def __post_init__(self) -> None:
        if self.periods < 1:
            raise ValueError("periods must be at least 1")
        if not self.name:
            self.name = f"{getattr(self.base, 'name', 'detector')} + histerese"

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.str_]:
        """The base detector's answer, unchanged.

        Without the episode index there is no sequence to be sustained over, so
        this deliberately does nothing rather than guessing an ordering.
        """
        return np.asarray(self.base.predict(features))

    def predict_sequence(self, features: NDArray[np.float64],
                         index: pd.DataFrame) -> NDArray[np.str_]:
        """Apply the hysteresis, episode by episode, in period order."""
        raw = np.asarray(self.base.predict(features))
        out = np.full(len(raw), "normal", dtype=object)

        for _, group in index.groupby("episode", sort=False):
            rows = group.sort_values("period").index.to_numpy()
            calls = raw[rows]
            run_label, run_length = None, 0
            for position, call in enumerate(calls):
                if call == run_label:
                    run_length += 1
                else:
                    run_label, run_length = call, 1
                if call != "normal" and run_length >= self.periods:
                    out[rows[position]] = call
        return out.astype(str)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class DetectionReport:
    """How a detector did, in the terms an operator would ask about.

    Attributes:
        name: the detector.
        recall: per-class share of event-active periods correctly classified.
        detection_recall: per-class share correctly flagged as *something*,
            ignoring which class — the difference between this and ``recall`` is
            how often the detector knew something was wrong but not what.
        false_alarm_rate: share of genuinely event-free periods flagged.
        sustained_false_alarm_rate: share of them flagged as part of a run of
            ``sustain`` consecutive flags — the number that decides how many
            tickets a deployed system raises, since nobody acts on one noisy
            period.
        macro_f1: unweighted mean F1 over the classes present.
        detection_delay_s: median seconds from onset to the first sustained
            correct detection, per class.  ``inf`` where an episode was never
            detected.
        false_alarms_per_hour: how often a *new* alarm starts on a healthy link.
            The rate an operator actually feels, and the one that decides
            whether a detector is deployable: a 1% per-period false alarm rate
            sounds small and is 36 tickets an hour at a one-second period.
        precision_at_prior: share of alarms that are real, if attacks occupy
            ``prior`` of the link's time.  Computed at 1% by default, because
            the balanced classes here are a property of the generator and the
            deployed prior is nowhere near them.
        prior: the prior ``precision_at_prior`` was evaluated at.
        confusion: true label to predicted label to count.
        rows: how many periods were scored.
    """

    name: str
    recall: dict[str, float]
    detection_recall: dict[str, float]
    false_alarm_rate: float
    sustained_false_alarm_rate: float
    macro_f1: float
    detection_delay_s: dict[str, float]
    false_alarms_per_hour: float
    precision_at_prior: float
    prior: float
    confusion: dict[str, dict[str, int]]
    rows: int

    def summary(self) -> str:
        """One line per class, in the order they appear."""
        lines = [f"{self.name}: {self.rows} periods, "
                 f"false alarm {self.false_alarm_rate:.4f} "
                 f"({self.false_alarms_per_hour:.1f}/h), "
                 f"precisão a {self.prior:.0%} {self.precision_at_prior:.3f}, "
                 f"macro F1 {self.macro_f1:.3f}"]
        for label in sorted(self.recall):
            delay = self.detection_delay_s.get(label, float("nan"))
            lines.append(f"  {label:<20} recall {self.recall[label]:.3f}  "
                         f"detected {self.detection_recall[label]:.3f}  "
                         f"delay {delay:.0f} s")
        return "\n".join(lines)


def evaluate_detector(detector, features: NDArray[np.float64],
                      labels: NDArray[np.str_], index: pd.DataFrame,
                      mask: NDArray[np.bool_] | None = None,
                      sustain: int = 3, period_s: float = 1.0,
                      prior: float = 0.01) -> DetectionReport:
    """Score a detector on held-out episodes.

    Args:
        detector: anything with ``predict(X) -> labels``.
        features, labels, index: from :func:`build_detection_features`.
        mask: rows to score; defaults to all of them.
        sustain: consecutive flagged periods required before a detection counts
            for the delay.  One period is a coin flip on a noisy estimator; a
            deployed system would not raise a ticket on it.
        period_s: settling period, for turning per-period rates into per-hour
            ones.
        prior: share of time attacks are assumed to occupy, for the precision
            figure.

    Returns:
        DetectionReport.

    Note:
        Accuracy is deliberately absent.  The classes are balanced here and are
        not balanced anywhere else, so an accuracy figure would be a statement
        about the generator.  Per-class recall and the false-alarm rate on
        event-free periods survive the change of prior; a macro F1 is reported
        because it is what the literature quotes.
    """
    mask = np.ones(len(labels), dtype=bool) if mask is None else mask
    features = features[mask]
    labels = labels[mask]
    index = index.loc[mask].reset_index(drop=True)
    if hasattr(detector, "predict_sequence"):
        predicted = np.asarray(detector.predict_sequence(features, index))
    else:
        predicted = np.asarray(detector.predict(features))

    present = sorted(set(labels) | set(predicted))
    confusion = {truth: {guess: 0 for guess in present} for truth in present}
    for truth, guess in zip(labels, predicted):
        confusion[truth][guess] += 1

    recall: dict[str, float] = {}
    detection_recall: dict[str, float] = {}
    for label in sorted(set(labels)):
        selector = labels == label
        total = int(selector.sum())
        recall[label] = float(np.mean(predicted[selector] == label)) if total else float("nan")
        if label == "normal":
            detection_recall[label] = recall[label]
        else:
            detection_recall[label] = float(np.mean(predicted[selector] != "normal"))

    event_free = labels == "normal"
    false_alarm = (float(np.mean(predicted[event_free] != "normal"))
                   if event_free.any() else float("nan"))

    sustained = np.zeros(len(labels), dtype=bool)
    for _, group in index.groupby("episode"):
        rows = group.index.to_numpy()
        sustained[rows] = _sustained_mask(predicted[rows] != "normal", sustain)
    sustained_false_alarm = (float(np.mean(sustained[event_free]))
                             if event_free.any() else float("nan"))

    # Alarm *onsets* per hour of healthy operation: an alarm that stays up for a
    # minute is one ticket, not sixty.
    onsets = 0
    healthy_periods = 0
    for _, group in index.groupby("episode", sort=False):
        rows = group.sort_values("period").index.to_numpy()
        calls = predicted[rows] != "normal"
        clean = labels[rows] == "normal"
        healthy_periods += int(clean.sum())
        previous = False
        for flagged, is_clean in zip(calls, clean):
            if is_clean and flagged and not previous:
                onsets += 1
            previous = flagged
    hours = healthy_periods * period_s / 3600.0
    false_alarms_per_hour = float(onsets / hours) if hours > 0 else float("nan")

    # Precision under a realistic prior: the balanced classes here are the
    # generator's, not a link's.
    attack_rows = ~event_free
    detection = (float(np.mean(predicted[attack_rows] != "normal"))
                 if attack_rows.any() else 0.0)
    true_positive = prior * detection
    false_positive = (1.0 - prior) * (false_alarm if np.isfinite(false_alarm) else 0.0)
    precision_at_prior = (float(true_positive / (true_positive + false_positive))
                          if (true_positive + false_positive) > 0 else float("nan"))

    macro_f1 = float(np.mean([_f1(labels, predicted, label)
                              for label in sorted(set(labels))]))

    delays: dict[str, float] = {}
    for label in sorted(set(index["episode_label"])):
        per_episode: list[float] = []
        for _, group in index[index["episode_label"] == label].groupby("episode"):
            onset = float(group["onset_s"].iloc[0])
            if not np.isfinite(onset):
                continue
            rows = group.index.to_numpy()
            flagged = predicted[rows] != "normal"
            times = group["time_s"].to_numpy()
            per_episode.append(_first_sustained(flagged, times, onset, sustain))
        if per_episode:
            delays[label] = float(np.median(per_episode))

    return DetectionReport(
        name=getattr(detector, "name", type(detector).__name__),
        recall=recall, detection_recall=detection_recall,
        false_alarm_rate=false_alarm,
        sustained_false_alarm_rate=sustained_false_alarm, macro_f1=macro_f1,
        detection_delay_s=delays, false_alarms_per_hour=false_alarms_per_hour,
        precision_at_prior=precision_at_prior, prior=prior,
        confusion=confusion, rows=int(len(labels)))


def _f1(labels: NDArray[np.str_], predicted: NDArray[np.str_], label: str) -> float:
    true_positive = float(np.sum((labels == label) & (predicted == label)))
    false_positive = float(np.sum((labels != label) & (predicted == label)))
    false_negative = float(np.sum((labels == label) & (predicted != label)))
    if true_positive == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    sensitivity = true_positive / (true_positive + false_negative)
    return 2.0 * precision * sensitivity / (precision + sensitivity)


def _sustained_mask(flagged: NDArray[np.bool_], sustain: int) -> NDArray[np.bool_]:
    """Mark the periods that belong to a run of ``sustain`` consecutive flags."""
    flagged = np.asarray(flagged, dtype=bool)
    if sustain <= 1 or len(flagged) < sustain:
        return flagged.copy() if sustain <= 1 else np.zeros_like(flagged)

    windows = np.lib.stride_tricks.sliding_window_view(flagged, sustain).all(axis=-1)
    out = np.zeros_like(flagged)
    for offset in range(sustain):
        out[offset:offset + len(windows)] |= windows
    return out


def _first_sustained(flagged: NDArray[np.bool_], times: NDArray[np.float64],
                     onset: float, sustain: int) -> float:
    """Seconds from onset to the first run of ``sustain`` consecutive flags."""
    after = np.nonzero(times >= onset)[0]
    for start in after:
        window = flagged[start:start + sustain]
        if len(window) == sustain and bool(np.all(window)):
            return float(times[start] - onset)
    return float("inf")
