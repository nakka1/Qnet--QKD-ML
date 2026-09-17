#!/usr/bin/env python3
"""Measure the three phase-5 results that are not the detector.

    python scripts/evaluate_phase5.py --part all --out runs/phase5

``keyrate``
    The trained agent against the commissioning baseline and against an oracle
    that sees the channel, over several spans; plus what the agent actually does
    with the knobs, which is the part worth reading.

``timing``
    Synchronisers against the thermal drift of the propagation delay, over a
    sweep of gate widths — and the key rate that survives each, because the
    efficiency is only interesting for what it buys.

``sequence``
    The temporal convolutional compensator against phase 3's own MLP, its
    classical baselines and the same closed loop, on the phase-3 operating
    point.  Nothing about that comparison changes except the architecture.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import task3  # noqa: E402
from task5 import (BUDGET, DEAD_TIME_S, LINK, SETTINGS, SYNC_TEST_SEEDS,  # noqa: E402
                   SYNC_TRAIN_SEEDS, TEST_SEEDS, make_commissioned, make_gate,
                   make_keyrate_env, make_timing_trajectories)
from qnet_autopilot.physics.decoy import decoy_key_rate_hz, observe  # noqa: E402
from qnet_autopilot.qkd.keyrate import (KeyRatePolicy, OraclePolicy,  # noqa: E402
                                        StaticPolicy, receiver_response, run_episode)
from qnet_autopilot.qkd.timing import (AlphaBetaTracker, ClockModel,  # noqa: E402
                                       NoCorrection, PeriodicCalibration,
                                       build_sync_dataset, fit_sync_predictor,
                                       gate_efficiency, run_sync_loop, timing_spread)


# ---------------------------------------------------------------------------
# key rate
# ---------------------------------------------------------------------------

def _load_policy(policy_path: Path) -> KeyRatePolicy:
    policy = KeyRatePolicy(rng=0)
    policy.load(policy_path)
    policy.freeze_normalisers()
    return policy


def select_temperature(policy_path: Path, spans_km, temperatures,
                       episodes: int = 12,
                       validation_seed: int = 700_000) -> tuple[float, pd.DataFrame]:
    """Pick the sampling temperature on validation seeds, as phase 2 does.

    A2C optimises the return of the *stochastic* policy, so greedifying is not
    free and is not automatically better — phase 2 measured a third of the
    performance lost by taking the argmax.  Here the optimum is shallow at long
    reach, where exploration costs more than it finds, so the right temperature
    is a question rather than an assumption.  It is answered on seeds disjoint
    from both training and test, and one temperature is chosen for the whole
    policy rather than one per span, because a deployed system has one.

    Args:
        policy_path: the saved policy.
        spans_km: spans to average over.
        temperatures: candidates; 0 is greedy.
        episodes: validation episodes per span and temperature.
        validation_seed: base seed for them.

    Returns:
        ``(best_temperature, table)``.
    """
    rows = []
    for temperature in temperatures:
        for length_km in spans_km:
            env = make_keyrate_env(length_km, rng=0,
                                   initial=make_commissioned(length_km))
            policy = _load_policy(policy_path)
            policy.temperature = float(temperature)
            policy.greedy = temperature <= 0
            rates = [run_episode(env, policy, seed=validation_seed + index).mean_key_rate_hz
                     for index in range(episodes)]
            rows.append({"temperature": float(temperature), "length_km": length_km,
                         "key_rate_hz": float(np.mean(rates))})

    table = pd.DataFrame(rows)
    pooled = table.groupby("temperature")["key_rate_hz"].mean()
    best = float(pooled.idxmax())
    print("\ntemperatura de amostragem (validação, Mbit/s)")
    print((table.pivot(index="temperature", columns="length_km",
                       values="key_rate_hz") / 1e6).to_string(
        float_format=lambda value: f"{value:8.3f}"))
    print(f"escolhida: {best:g}")
    return best, table


def evaluate_keyrate(out: Path, policy_path: Path, spans_km, episodes: int,
                     temperatures=(0.0, 0.25, 0.5, 1.0)) -> None:
    """Agent, commissioning baseline and oracle, on identical episodes."""
    temperature, sweep = select_temperature(policy_path, spans_km, temperatures)
    sweep.to_csv(out / "keyrate_temperature.csv", index=False)

    rows = []
    decisions = []

    for length_km in spans_km:
        commissioned = make_commissioned(length_km)
        env = make_keyrate_env(length_km, rng=0, initial=commissioned)
        policy = _load_policy(policy_path)
        policy.temperature = temperature
        policy.greedy = temperature <= 0

        policies = {"agente": policy, "estático": StaticPolicy(),
                    "oráculo": OraclePolicy(env=env)}
        for name, candidate in policies.items():
            for seed in TEST_SEEDS[:episodes]:
                result = run_episode(env, candidate, seed=seed,
                                     collect=(name == "agente" and seed == TEST_SEEDS[0]))
                rows.append({"length_km": length_km, "policy": name, "seed": seed,
                             "key_rate_hz": result.mean_key_rate_hz,
                             "key_bits": result.key_bits,
                             "zero_key_periods": result.zero_key_periods,
                             "final_signal_intensity": result.final_settings.signal_intensity,
                             "final_decoy_intensity": result.final_settings.decoy_intensity,
                             "final_clock_hz": result.final_settings.clock_rate_hz,
                             "final_signal_probability": result.final_settings.signal_probability})
                for step, transition in enumerate(result.transitions):
                    decisions.append({
                        "length_km": length_km, "seed": seed, "step": step,
                        "action": transition.action_index,
                        "key_bits": transition.key_bits,
                        "signal_intensity": transition.settings.signal_intensity,
                        "decoy_intensity": transition.settings.decoy_intensity,
                        "clock_hz": transition.settings.clock_rate_hz,
                        "signal_probability": transition.settings.signal_probability,
                        "excess_loss_db": transition.excess_loss_db,
                        "misalignment": transition.misalignment})

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "keyrate_episodes.csv", index=False)
    pd.DataFrame(decisions).to_csv(out / "keyrate_decisions.csv", index=False)

    summary = (frame.groupby(["length_km", "policy"])
               .agg(key_rate_hz=("key_rate_hz", "mean"),
                    key_rate_std=("key_rate_hz", "std"),
                    zero_key_periods=("zero_key_periods", "mean"),
                    signal_intensity=("final_signal_intensity", "mean"),
                    decoy_intensity=("final_decoy_intensity", "mean"),
                    clock_hz=("final_clock_hz", "mean"))
               .reset_index())
    summary.to_csv(out / "keyrate_summary.csv", index=False)

    print(f"\ntaxa de chave (média de episódios de teste, temperatura {temperature:g}, "
          f"Mbit/s)")
    pivot = summary.pivot(index="length_km", columns="policy", values="key_rate_hz") / 1e6
    pivot["agente vs estático"] = pivot["agente"] / pivot["estático"] - 1.0
    pivot["fração do oráculo"] = ((pivot["agente"] - pivot["estático"])
                                  / (pivot["oráculo"] - pivot["estático"]))
    print(pivot.to_string(float_format=lambda value: f"{value:8.3f}"))


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def evaluate_timing(out: Path, spans_km, widths_s,
                    dark_rates_hz=(10.0, 1.0e5, 1.0e6)) -> None:
    """Synchronisers against thermal delay drift, and what each buys in key.

    The gate width is swept against the dark count rate because the two only
    interact: a superconducting detector at ten counts a second does not care
    how wide the window is, and an InGaAs receiver at a kilohertz cares a great
    deal.  Quoting the synchronisation result at one of them would make the gate
    width look either free or decisive, and it is neither.
    """
    gate = make_gate()
    rows = []

    for length_km in spans_km:
        trajectories = make_timing_trajectories(SYNC_TRAIN_SEEDS, length_km)
        features, targets, episode, _ = build_sync_dataset(trajectories, gate)
        learned = fit_sync_predictor(features, targets, name="ridge + termômetro")
        spread_s = timing_spread(BUDGET, length_km)

        for width_s in widths_s:
            width_gate = make_gate(width_s=width_s)
            for seed in SYNC_TEST_SEEDS:
                trajectory = ClockModel(length_km=length_km).simulate(
                    3600.0, 1.0, rng=int(seed) + 10_000)
                for predictor in (NoCorrection(), PeriodicCalibration(interval_s=60.0),
                                  AlphaBetaTracker(), learned):
                    result = run_sync_loop(trajectory, predictor, width_gate,
                                           rng=int(seed))
                    summary = result.summary()
                    efficiency = summary["mean_efficiency"]
                    for dark_rate_hz in dark_rates_hz:
                        rows.append({
                            "length_km": length_km, "gate_width_ps": width_s * 1e12,
                            "dark_count_rate_hz": dark_rate_hz,
                            "seed": seed, "synchroniser": summary["sincronizador"],
                            "mean_efficiency": efficiency,
                            "p05_efficiency": summary["p05_efficiency"],
                            "mean_residual_ps": summary["mean_residual_ps"],
                            "uptime": summary["uptime"],
                            "key_rate_hz": _key_rate_with(length_km, width_s,
                                                          efficiency, dark_rate_hz),
                        })

        print(f"  {length_km:.0f} km: espalhamento {spread_s * 1e12:.0f} ps")

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "timing_episodes.csv", index=False)
    summary = (frame.groupby(["length_km", "gate_width_ps", "dark_count_rate_hz",
                              "synchroniser"])
               .agg(mean_efficiency=("mean_efficiency", "mean"),
                    p05_efficiency=("p05_efficiency", "mean"),
                    mean_residual_ps=("mean_residual_ps", "mean"),
                    uptime=("uptime", "mean"),
                    key_rate_hz=("key_rate_hz", "mean"))
               .reset_index())
    summary.to_csv(out / "timing_summary.csv", index=False)

    print("\ntaxa de chave por sincronizador (Mbit/s)")
    pivot = summary.pivot_table(index=["length_km", "dark_count_rate_hz", "gate_width_ps"],
                                columns="synchroniser", values="key_rate_hz") / 1e6
    print(pivot.to_string(float_format=lambda value: f"{value:8.3f}"))


def _key_rate_with(length_km: float, gate_width_s: float, efficiency: float,
                   dark_count_rate_hz: float = 10.0) -> float:
    """Key rate at a span, with the gate's width and the synchroniser's efficiency.

    The gate width enters twice and in opposite directions: it widens the
    acceptance (already in ``efficiency``) and it admits proportionally more dark
    counts.  That is the whole trade a synchroniser is buying its way out of, so
    both halves are evaluated here rather than quoting the efficiency alone.
    """
    from dataclasses import replace

    link = replace(LINK, length_km=length_km, gate_width_s=gate_width_s,
                   dark_count_rate_hz=dark_count_rate_hz)
    settings = make_commissioned(length_km)
    response = receiver_response(link, settings, BUDGET, DEAD_TIME_S)
    # ``receiver_response`` already applies the gate at perfect alignment; the
    # synchroniser's residual costs the rest.
    perfect = float(gate_efficiency(0.0, gate_width_s, timing_spread(BUDGET, length_km)))
    achieved = efficiency / max(perfect, 1e-12)
    degraded = replace(link, coupling_efficiency=link.coupling_efficiency * achieved)
    response = receiver_response(degraded, settings, BUDGET, DEAD_TIME_S)
    return decoy_key_rate_hz(observe(response, settings))


# ---------------------------------------------------------------------------
# sequence compensator
# ---------------------------------------------------------------------------

def _phase3_dataset(out: Path, control, cache: bool = True):
    """Build (or reuse) the phase-3 supervised windows.

    Building them walks twelve twenty-second trajectories at a kilohertz and
    costs minutes, and every architecture under comparison has to see exactly
    the same rows anyway — so the matrix is cached and reused rather than
    rebuilt per configuration.
    """
    from qnet_autopilot.compensation.train import Dataset, build_dataset

    path = out / "sequence_dataset.npz"
    if cache and path.exists():
        stored = np.load(path)
        return Dataset(features=stored["features"], targets=stored["targets"],
                       episode=stored["episode"], time_s=stored["time_s"],
                       spec=task3.FEATURE_SPEC)

    dataset = build_dataset(task3.make_trajectories(task3.TRAIN_SEEDS), control,
                            task3.FEATURE_SPEC, target="measured", rng=0)
    if cache:
        np.savez(path, features=dataset.features, targets=dataset.targets,
                 episode=dataset.episode, time_s=dataset.time_s)
    return dataset


def sweep_sequence(out: Path, epochs: int) -> None:
    """Validation error for a few architectures, before spending a closed loop.

    The receptive field is the parameter that matters and it is arithmetic, not
    a hyperparameter search: at a 2 ms control period one period of the 40 Hz
    vibration band is 12.5 steps, so a stack that reaches seven lags cannot see
    a full oscillation and a stack that reaches fifteen can.  This measures that.
    """
    from qnet_autopilot.compensation.sequence import fit_sequence
    from qnet_autopilot.compensation.train import chronological_split, fit_mlp

    control = task3.make_control()
    dataset = _phase3_dataset(out, control)
    train, validation = chronological_split(dataset, 0.75, 0.05)

    def score(compensator) -> float:
        features = compensator.standardiser.transform(dataset.features[validation])
        predicted = np.asarray(compensator.model.predict(features)).reshape(-1, 3)
        return float(np.mean((predicted - dataset.targets[validation]) ** 2))

    rows = []
    mlp, _ = fit_mlp(dataset, train, validation, hidden=(192, 192), epochs=epochs,
                     seed=0)
    rows.append({"model": "MLP (192, 192)", "receptive_field": task3.FEATURE_SPEC.window - 1,
                 "parameters": sum(array.size for array in mlp.model.network.parameters),
                 "validation_mse": score(mlp)})
    print(f"  MLP                          MSE {rows[-1]['validation_mse']:8.1f}")

    for channels, dilations in ((32, (1, 2)), (32, (1, 2, 4)), (48, (1, 2, 4)),
                                (64, (1, 2, 4))):
        compensator, _ = fit_sequence(dataset, train, validation, channels=channels,
                                      dilations=dilations, hidden=64, epochs=epochs,
                                      seed=0, name=f"TCN c{channels} d{dilations}")
        network = compensator.model.network
        rows.append({"model": compensator.name,
                     "receptive_field": network.spec.receptive_field,
                     "parameters": network.parameter_count(),
                     "validation_mse": score(compensator)})
        print(f"  TCN c{channels} d{dilations} campo {network.spec.receptive_field:2d}  "
              f"MSE {rows[-1]['validation_mse']:8.1f}")

    pd.DataFrame(rows).to_csv(out / "sequence_sweep.csv", index=False)


def evaluate_sequence(out: Path, epochs: int, channels: int = 48,
                      dilations=(1, 2, 4)) -> None:
    """The temporal model against phase 3's MLP, in phase 3's own closed loop."""
    from qnet_autopilot.compensation import (AlphaBetaFilter, LinearExtrapolation,
                                             NoCompensation, Persistence,
                                             run_closed_loop)
    from qnet_autopilot.compensation.sequence import fit_sequence
    from qnet_autopilot.compensation.train import (chronological_split, fit_mlp,
                                                   fit_ridge)

    control = task3.make_control()
    dataset = _phase3_dataset(out, control)
    train, validation = chronological_split(dataset, 0.75, 0.05)
    print(f"  {len(dataset)} janelas, {int(train.sum())} treino / "
          f"{int(validation.sum())} validação")

    ridge = fit_ridge(dataset, train)
    mlp, mlp_history = fit_mlp(dataset, train, validation, hidden=(192, 192),
                               epochs=epochs, seed=0)
    tcn, tcn_history = fit_sequence(dataset, train, validation, channels=channels,
                                    dilations=tuple(dilations), hidden=64,
                                    epochs=epochs, seed=0)

    def validation_mse(compensator) -> float:
        features = compensator.standardiser.transform(dataset.features[validation])
        predicted = np.asarray(compensator.model.predict(features)).reshape(-1, 3)
        return float(np.mean((predicted - dataset.targets[validation]) ** 2))

    print(f"  MSE de validação  ridge {validation_mse(ridge):.2f}  "
          f"MLP {validation_mse(mlp):.2f}  TCN {validation_mse(tcn):.2f}")

    compensators = [NoCompensation(), Persistence(), LinearExtrapolation(),
                    AlphaBetaFilter(**task3.ALPHA_BETA), ridge, mlp, tcn]
    rows = []
    for seed in task3.TEST_SEEDS:
        trajectory = task3.make_trajectories([seed])[0]
        for compensator in compensators:
            result = run_closed_loop(trajectory, compensator, control, rng=int(seed))
            summary = result.summary()
            summary["seed"] = seed
            summary["finite_key_rate_hz"] = result.finite_key_rate_hz(
                control.sifted_rate_hz, block_duration_s=1.0)
            recoveries = [value for value in result.recovery_times_s() if np.isfinite(value)]
            summary["recovery_s"] = float(np.mean(recoveries)) if recoveries else float("nan")
            rows.append(summary)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "sequence_episodes.csv", index=False)
    summary = (frame.groupby("compensator")
               .agg(mean_qber=("mean_qber", "mean"), p95_qber=("p95_qber", "mean"),
                    uptime=("uptime", "mean"),
                    key_rate_hz=("mean_key_rate_hz", "mean"),
                    finite_key_rate_hz=("finite_key_rate_hz", "mean"),
                    residual_mrad=("mean_residual_mrad", "mean"),
                    recovery_s=("recovery_s", "mean"))
               .reset_index()
               .sort_values("finite_key_rate_hz", ascending=False))
    summary.to_csv(out / "sequence_summary.csv", index=False)
    (out / "sequence_history.json").write_text(json.dumps(
        {"mlp": mlp_history, "tcn": tcn_history,
         "validation_mse": {"ridge": validation_mse(ridge), "mlp": validation_mse(mlp),
                            "tcn": validation_mse(tcn)}}, indent=2))

    print("\ncompensadores no laço fechado da fase 3")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:10.4f}"))


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="all",
                        choices=["all", "keyrate", "timing", "sequence",
                                 "sequence-sweep"])
    parser.add_argument("--out", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--policy", type=Path, default=Path("runs/keyrate/main_policy.npz"))
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--spans", type=float, nargs="+",
                        default=[25.0, 50.0, 80.0],
                        help="span lengths for the key-rate comparison")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.part in ("all", "keyrate"):
        print("== taxa de chave ==")
        evaluate_keyrate(args.out, args.policy, tuple(args.spans), args.episodes)
    if args.part in ("all", "timing"):
        print("\n== sincronização ==")
        evaluate_timing(args.out, (50.0, 80.0),
                        (200e-12, 350e-12, 500e-12, 1000e-12))
    if args.part == "sequence-sweep":
        print("== varredura de arquitetura ==")
        sweep_sequence(args.out, args.epochs)
    if args.part in ("all", "sequence"):
        print("\n== compensador sequencial ==")
        evaluate_sequence(args.out, args.epochs, args.channels,
                          tuple(args.dilations))

    print(f"\nsalvo em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
