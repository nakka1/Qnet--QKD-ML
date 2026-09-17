#!/usr/bin/env python3
"""Fit and score the phase-5 attack detectors.

    python scripts/train_detector.py --out runs/detector

Writes the dataset, the per-class scores for every detector, the feature
importances, and two ablations: one without the thermometer, and one without the
decoy-state observables.  Both exist to answer the same question — which part of
the physics is carrying the separation — and the answer is the most useful thing
in the phase-5 report.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task5 import (BASELINE_PERIODS, DATASET_SEED, DETECTOR_SEED,  # noqa: E402
                   FEATURE_WINDOW, LENGTHS_KM, LINK, MIN_ACTIVE_PERIODS, PERIOD_S,
                   RESIDUAL_FRACTION, SESSION_DURATION_S, SETTINGS,
                   UNTRAINABLE_LABELS)
from qnet_autopilot.datagen.qkd import generate_qkd_session_dataset  # noqa: E402
from qnet_autopilot.qkd.detection import (FEATURE_NAMES, QberThreshold,  # noqa: E402
                                          SustainedDetector,
                                          build_detection_features, episode_split,
                                          evaluate_detector, fit_boosted, fit_forest,
                                          fit_svm)


def _report_frame(report, subset: str) -> pd.DataFrame:
    """One row per class, so every detector's scores stack into one table."""
    rows = []
    for label in sorted(report.recall):
        rows.append({
            "detector": report.name,
            "subset": subset,
            "label": label,
            "recall": report.recall[label],
            "detection_recall": report.detection_recall[label],
            "detection_delay_s": report.detection_delay_s.get(label, float("nan")),
            "false_alarm_rate": report.false_alarm_rate,
            "sustained_false_alarm_rate": report.sustained_false_alarm_rate,
            "false_alarms_per_hour": report.false_alarms_per_hour,
            "precision_at_prior": report.precision_at_prior,
            "prior": report.prior,
            "macro_f1": report.macro_f1,
            "rows": report.rows,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/detector"))
    parser.add_argument("--episodes", type=int, default=24,
                        help="episodes per class")
    parser.add_argument("--duration", type=float, default=SESSION_DURATION_S)
    parser.add_argument("--period", type=float, default=PERIOD_S,
                        help="settling period in seconds; a longer one publishes "
                             "fewer, quieter decisions per hour")
    parser.add_argument("--residual-fraction", type=float, default=RESIDUAL_FRACTION)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DATASET_SEED)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="reuse a dataset written by a previous run")
    parser.add_argument("--step-only", action="store_true",
                        help="generate the original step-only attacks, with no "
                             "ramp, burst or simultaneous thermal excursion")
    parser.add_argument("--sustain", type=int, default=3,
                        help="periods an alarm must persist before it counts")
    parser.add_argument("--false-alarm-target", type=float, default=0.005,
                        help="share of event-free periods the decision threshold "
                             "is allowed to fire on")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dataset is not None and args.dataset.exists():
        frame = pd.read_csv(args.dataset)
        print(f"[detector] dataset lido de {args.dataset}: {len(frame)} linhas")
    else:
        dataset = generate_qkd_session_dataset(
            episodes_per_class=args.episodes, duration_s=args.duration,
            period_s=args.period, baseline_periods=BASELINE_PERIODS,
            lengths_km=LENGTHS_KM, settings=SETTINGS, link=LINK,
            residual_fraction=args.residual_fraction,
            adversarial=not args.step_only, seed=args.seed)
        path = dataset.save(args.out / "sessions.csv")
        frame = dataset.frame
        print(f"[detector] {len(frame)} linhas em {path}")

    features, labels, index = build_detection_features(frame, FEATURE_WINDOW)
    train, test = episode_split(index, 0.7, seed=DETECTOR_SEED)
    print(f"[detector] {features.shape[0]} períodos, {features.shape[1]} atributos; "
          f"{int(train.sum())} treino / {int(test.sum())} teste")

    # Beam splitting leaves no signature: training on it teaches the model to
    # split noise, and costs the normal class precision. It is scored below on
    # its own terms instead.
    trainable = train & ~np.isin(index["episode_label"].to_numpy(), UNTRAINABLE_LABELS)
    scored = test & ~np.isin(index["episode_label"].to_numpy(), UNTRAINABLE_LABELS)

    # A decision threshold calibrated on the rows the model was fitted on is
    # calibrated on rows it has memorised: the first version of this put the
    # threshold at zero because the boosting flagged almost no training period,
    # while flagging 6% of held-out ones. So the training episodes are split
    # again — models are fitted on 80% of them and the threshold is set on the
    # other 20%, which no model has seen.
    episodes = index["episode"].to_numpy()
    train_episodes = np.unique(episodes[trainable])
    shuffled = np.array(train_episodes, copy=True)
    np.random.default_rng(DETECTOR_SEED).shuffle(shuffled)
    cut = max(int(round(0.8 * len(shuffled))), 1)
    fit_mask = trainable & np.isin(episodes, shuffled[:cut])
    calibration_mask = trainable & np.isin(episodes, shuffled[cut:])
    print(f"[detector] {int(fit_mask.sum())} linhas de ajuste / "
          f"{int(calibration_mask.sum())} de calibração")

    forest = fit_forest(features, labels, fit_mask, trees=args.trees, seed=DETECTOR_SEED)
    svm = fit_svm(features, labels, fit_mask, seed=DETECTOR_SEED)
    boosted = fit_boosted(features, labels, fit_mask, seed=DETECTOR_SEED)
    threshold = QberThreshold.calibrate(features, labels, trainable, 0.01)
    print(f"[detector] limiar calibrado em sigma = {threshold.sigma:.2f}")

    # The deployable configuration: boosting, with its decision threshold moved
    # to a false-alarm budget on the training episodes, then required to persist.
    from copy import deepcopy

    gated = deepcopy(boosted)
    gated.name = "gradient boosting + limiar"
    gated.calibrate_threshold(features, labels, calibration_mask,
                              args.false_alarm_target)
    print(f"[detector] limiar de decisão em P(ataque) >= {gated.attack_threshold:.3f} "
          f"para {args.false_alarm_target:.1%} de alarme falso")

    sustained = SustainedDetector(base=gated, periods=args.sustain,
                                  name="boosting + limiar + histerese")
    detectors = (threshold, forest, svm, boosted, gated, sustained)

    tables = []
    for detector in detectors:
        report = evaluate_detector(detector, features, labels, index, scored,
                                   sustain=args.sustain, period_s=args.period)
        print()
        print(report.summary())
        tables.append(_report_frame(report, "teste"))

    # How the hard cases fare: an attack that ramps in or works in bursts, and
    # one that runs while the temperature is moving.
    hardness_rows = []
    if "profile" in frame.columns:
        profile = (frame.groupby("episode")["profile"].first()
                   .reindex(index["episode"].to_numpy()).to_numpy())
        cooccurrence = (frame.groupby("episode")["thermal_cooccurrence"].first()
                        .reindex(index["episode"].to_numpy()).to_numpy())
        attacked = scored & (labels != "normal")
        for detector in (boosted, gated, sustained):
            predicted = (detector.predict_sequence(features, index)
                         if hasattr(detector, "predict_sequence")
                         else detector.predict(features))
            for name, selector in (("degrau", profile == "step"),
                                   ("rampa", profile == "ramp"),
                                   ("intermitente", profile == "intermittent"),
                                   ("com excursão térmica", cooccurrence > 0),
                                   ("sem excursão térmica", cooccurrence == 0)):
                rows_here = attacked & selector
                if not rows_here.any():
                    continue
                hardness_rows.append({
                    "detector": detector.name, "case": name,
                    "periods": int(rows_here.sum()),
                    "recall": float(np.mean(predicted[rows_here] == labels[rows_here])),
                    "detected": float(np.mean(predicted[rows_here] != "normal"))})
        print()
        print("casos difíceis (só períodos com ataque em curso):")
        for row in hardness_rows:
            print(f"  {row['detector']:<28} {row['case']:<22} "
                  f"recall {row['recall']:.3f}  detectado {row['detected']:.3f}")
        pd.DataFrame(hardness_rows).to_csv(args.out / "hardness.csv", index=False)

    # The unlearnable class, scored separately: the only correct answer is
    # "normal", so what is measured is how often each detector invents an attack.
    beam = test & np.isin(index["episode_label"].to_numpy(), UNTRAINABLE_LABELS)
    beam_rows = []
    if beam.any():
        active = (index["time_s"].to_numpy() >= index["onset_s"].to_numpy()) & beam
        for detector in detectors:
            predicted = detector.predict(features[active])
            beam_rows.append({"detector": detector.name,
                              "periods": int(active.sum()),
                              "called_normal": float(np.mean(predicted == "normal")),
                              "called_attack": float(np.mean(predicted != "normal"))})
        print()
        print("divisor de feixe (sem assinatura física — 'normal' é a resposta certa):")
        for row in beam_rows:
            print(f"  {row['detector']:<20} normal {row['called_normal']:.3f}  "
                  f"ataque {row['called_attack']:.3f}")

    # Ablations: which physics is carrying the separation.
    ablations = {
        "sem termômetro": [index for index, name in enumerate(FEATURE_NAMES)
                           if "temperature" not in name],
        "sem decoy": [index for index, name in enumerate(FEATURE_NAMES)
                      if "decoy" not in name and "single_photon" not in name],
        "só QBER e ganho": [FEATURE_NAMES.index(name) for name in
                            ("signal_error_shift", "log_signal_gain_ratio",
                             "error_trend", "gain_trend", "error_volatility")],
    }
    for name, columns in ablations.items():
        reduced = fit_forest(features[:, columns], labels, trainable,
                             trees=args.trees, seed=DETECTOR_SEED,
                             name=f"random forest ({name})")
        reduced.feature_names = tuple(FEATURE_NAMES[column] for column in columns)
        report = evaluate_detector(reduced, features[:, columns], labels, index, scored)
        print()
        print(report.summary())
        tables.append(_report_frame(report, "teste"))

    scores = pd.concat(tables, ignore_index=True)
    scores.to_csv(args.out / "scores.csv", index=False)

    importances = forest.importances() or {}  # the forest is what exposes them
    pd.DataFrame({"feature": list(importances), "importance": list(importances.values())}
                 ).sort_values("importance", ascending=False).to_csv(
        args.out / "importances.csv", index=False)
    pd.DataFrame(beam_rows).to_csv(args.out / "beam_splitting.csv", index=False)

    confusion = evaluate_detector(boosted, features, labels, index, scored).confusion
    pd.DataFrame(confusion).T.to_csv(args.out / "confusion.csv")

    with open(args.out / "detectors.pkl", "wb") as handle:
        pickle.dump({"forest": forest, "svm": svm, "boosted": boosted,
                     "gated": gated, "threshold": threshold}, handle)
    (args.out / "settings.json").write_text(json.dumps({
        "episodes_per_class": args.episodes,
        "duration_s": args.duration,
        "residual_fraction": args.residual_fraction,
        "threshold_sigma": float(threshold.sigma),
        "adversarial": not args.step_only,
        "sustain": args.sustain,
        "false_alarm_target": args.false_alarm_target,
        "period_s": args.period,
        "attack_threshold": float(gated.attack_threshold),
        "feature_names": list(FEATURE_NAMES),
        "window": FEATURE_WINDOW.window,
        "baseline_periods": FEATURE_WINDOW.baseline_periods,
    }, indent=2))
    print(f"\n[detector] salvo em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
