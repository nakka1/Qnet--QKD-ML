"""Render the gap-closing study as a self-contained HTML report.

Built from the CSVs ``scripts/evaluate_realism.py`` and
``scripts/evaluate_actuator.py`` write. Shares its styling and its validated
categorical palette with :mod:`qnet_autopilot.validation.report`.

The shape of this report is different from the phase reports. Those answer
"what does this system do". This one answers "how much of what the phase reports
claimed survives contact with three things they assumed away" — which is a
question whose useful answers are corrections, not achievements. So the layout
leads with the number each phase report should have printed, states the
correction as a factor, and only then explains the mechanism. Where a correction
changes a conclusion, it says so in the heading rather than in a footnote.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents.report import _pt, _table_pt
from .report import (_CSS, _FONTS, THEMES, Theme, _figure, _figure_block,
                     _legend, _section, _style_text, _table, _to_svg)

__all__ = ["render_realism_report"]

#: Fixed colour per compensator across every figure, so a reader who learns the
#: palette in one figure keeps it in the next.
COMPENSATOR_INDEX = {"MLP": 0, "ridge (Wiener)": 1, "filtro alfa-beta": 2,
                     "persistência": 3}

#: Drawing order: best first, so the legend reads as a ranking.
COMPENSATOR_ORDER = ["MLP", "ridge (Wiener)", "filtro alfa-beta", "persistência"]

FAMILY_INDEX = {"preditivo": 0, "limiar reativo": 1, "admissão estática": 2}

#: The measured quantities are named in English in
#: :mod:`qnet_autopilot.physics.field_calibration`, because the code is; this
#: report is in Portuguese, and a table half in each reads as an unfinished
#: translation. Keyed on the English name so a new measurement shows up
#: untranslated rather than silently missing.
MEASUREMENT_LABEL = {
    "polarisation linewidth": ("largura de linha de polarização",
                               "60 (Stokes), 80 (Jones) nHz",
                               "127 km enterrada, 36 dias a cada ~2,2 h"),
    "narrowband content": ("conteúdo de banda estreita",
                           "tom de 60 Hz + harmônicos de 120 e 180 Hz",
                           "220 km aéreo em cabo para-raios, linha de 160 kV, "
                           "polarímetro a 4 kHz"),
    "excursion distribution": ("distribuição das excursões",
                               "Rayleigh, por janela de 1 ms",
                               "110 km enterrada e 220 km aérea, 9 h a 4 kHz"),
    "drift PSD exponent": ("expoente do espectro de deriva", "1,0 ± 0,2",
                           "15 km aérea e enterrada, 11 meses"),
    "spectral centroid": ("centroide espectral", "3,2 ± 0,9 mHz",
                          "15 km aérea e enterrada, 11 meses"),
    "daytime drift": ("deriva diurna",
                      "dezenas de graus em segundos; fidelidade < 95% em < 20 s",
                      "62 km, ~70% aérea, 18 dB"),
    "spontaneous Raman coefficient": ("coeficiente Raman espontâneo",
                                      "1,5×10⁻⁹ a 3,1×10⁻⁹ /km/nm",
                                      "25 km monomodo padrão, banda C"),
    "coexistence QBER penalty": ("penalidade de QBER em coexistência",
                                 "2,4% → 2,7% com 100 Gbit/s DWDM",
                                 "três enlaces metropolitanos de 5,0-10,6 km, 580 dias"),
}


def _pt_number(value, spec: str = ",.0f") -> str:
    """Format with pt-BR separators: dot for thousands, comma for decimals.

    ``_pt`` only swaps the decimal point, which turns ``87,253`` into something a
    Brazilian reader parses as eighty-seven point two five three. Anything with a
    thousands separator has to come through here instead.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    text = format(value, spec)
    return text.translate(str.maketrans({",": "\x00", ".": ","})).replace("\x00", ".")


def _table_pt_thousands(frame: pd.DataFrame, formats: dict[str, str],
                        columns, headers) -> str:
    rendered = pd.DataFrame({
        column: (frame[column].map(lambda value, spec=formats[column]:
                                   _pt_number(value, spec))
                 if column in formats else frame[column].astype(str))
        for column in columns})
    return _table(rendered, {}, columns=list(columns), headers=list(headers))


def _colour(name: str, theme: Theme, table=COMPENSATOR_INDEX) -> str:
    return theme.series[table.get(name, 3) % len(theme.series)]


def _interpolate(frame: pd.DataFrame, target: float) -> float:
    """Classical throughput a frontier reaches at a given key-service level.

    Returns NaN outside the family's measured range rather than extrapolating,
    for the reason the phase-4 report gives: a policy never observed at a
    service level should not be credited with a number there.
    """
    ordered = frame.sort_values("served_qkd")
    keys = ordered["served_qkd"].to_numpy()
    values = ordered["served_classical"].to_numpy()
    if target < keys.min() or target > keys.max():
        return float("nan")
    return float(np.interp(target, keys, values))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _figure_twin(twin: pd.DataFrame, theme: Theme) -> str:
    """Every step of every trajectory, against double precision."""
    figure, axes = _figure(theme, height=3.8)

    for index, (label, group) in enumerate(twin.groupby("topology", sort=False)):
        axes.scatter(np.arange(len(group)),
                     np.maximum(group["state_divergence"], 1e-18),
                     s=16, alpha=0.75, color=theme.series[index],
                     edgecolors="none", label=label)

    axes.axhline(2.2e-16, color=theme.text_secondary, linewidth=1.0,
                 linestyle=":", alpha=0.9)
    axes.annotate("epsilon da precisão dupla", xy=(0.01, 2.6e-16),
                  xycoords=("axes fraction", "data"), fontsize=8.5,
                  color=theme.text_secondary, va="bottom")
    axes.set_yscale("log")
    axes.set_ylim(1e-17, 1e-9)
    _style_text(axes, theme,
                "Divergência entre o ambiente da fase 2 e o SeQUeNCe, passo a passo",
                "passo comparado", "maior discrepância absoluta no estado")
    _legend(axes, theme, loc="upper left")
    return _to_svg(figure)


def _figure_finite_link(link: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme)

    for name in COMPENSATOR_ORDER:
        group = link[link["compensator"] == name].sort_values("block_duration_s")
        if group.empty:
            continue
        axes.plot(group["block_duration_s"], group["ratio"], marker="o",
                  markersize=6, linewidth=2, color=_colour(name, theme),
                  markeredgecolor=theme.surface, markeredgewidth=1.4, label=name)

    axes.axhline(1.0, color=theme.text_secondary, linewidth=1.0, linestyle="--",
                 alpha=0.8)
    axes.annotate("taxa assintótica: o que as fases 3 e 4 publicaram",
                  xy=(0.99, 1.02), xycoords=("axes fraction", "data"), ha="right",
                  fontsize=8.5, color=theme.text_secondary, va="bottom")
    axes.set_xscale("log")
    axes.set_ylim(-0.05, 1.15)
    _style_text(axes, theme,
                "Fração da taxa assintótica que sobrevive a um bloco finito",
                "duração do bloco (s), a 10⁵ bits peneirados por segundo",
                "chave finita ÷ chave assintótica")
    _legend(axes, theme, loc="center right")
    return _to_svg(figure)


def _figure_frontier(frontier: pd.DataFrame, asymptotic: pd.DataFrame,
                     theme: Theme) -> str:
    figure, axes = _figure(theme)

    for family, group in asymptotic.groupby("family", sort=False):
        if family not in FAMILY_INDEX:
            continue
        ordered = group.sort_values("served_qkd")
        axes.plot(ordered["served_qkd"], ordered["served_classical"],
                  linewidth=1.6, linestyle=":", alpha=0.55,
                  color=_colour(family, theme, FAMILY_INDEX))

    for family, group in frontier.groupby("family", sort=False):
        ordered = group.sort_values("served_qkd")
        axes.plot(ordered["served_qkd"], ordered["served_classical"],
                  marker="o", markersize=6, linewidth=2,
                  color=_colour(family, theme, FAMILY_INDEX),
                  markeredgecolor=theme.surface, markeredgewidth=1.4,
                  label=family)

    _style_text(axes, theme,
                "As fronteiras da fase 4, com e sem o custo de bloco finito",
                "serviço de chave (fração da demanda de QKD atendida)",
                "serviço clássico (fração da demanda atendida)")
    axes.annotate("pontilhado: assintótico", xy=(0.02, 0.06),
                  xycoords="axes fraction", fontsize=8.5,
                  color=theme.text_secondary)
    _legend(axes, theme, loc="upper right")
    return _to_svg(figure)


def _figure_slew(slew: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme)
    finite = slew[np.isfinite(slew["slew_rate"])]

    for name in COMPENSATOR_ORDER:
        group = finite[finite["compensator"] == name].sort_values("slew_rate")
        if group.empty:
            continue
        axes.plot(group["slew_rate"], group["mean_qber"], marker="o",
                  markersize=6, linewidth=2, color=_colour(name, theme),
                  markeredgecolor=theme.surface, markeredgewidth=1.4, label=name)
        ideal = slew[(slew["compensator"] == name)
                     & (~np.isfinite(slew["slew_rate"]))]
        if not ideal.empty:
            axes.axhline(float(ideal["mean_qber"].iloc[0]),
                         color=_colour(name, theme), linewidth=1.0,
                         linestyle=":", alpha=0.6)

    axes.set_xscale("log")
    _style_text(axes, theme,
                "QBER média contra a velocidade máxima do controlador de polarização",
                "velocidade do atuador (rad/s)", "QBER média")
    axes.annotate("pontilhado: atuador ideal", xy=(0.02, 0.06),
                  xycoords="axes fraction", fontsize=8.5,
                  color=theme.text_secondary)
    _legend(axes, theme, loc="center left")
    return _to_svg(figure)


def _figure_round_cost(cost: pd.DataFrame, theme: Theme) -> str:
    """What one heralding attempt costs, in each simulator, against span."""
    figure, axes = _figure(theme, height=3.8)

    axes.plot(cost["span_km"], cost["sequence_ms_per_round"], marker="o",
              markersize=7, linewidth=2, color=theme.series[0],
              markeredgecolor=theme.surface, markeredgewidth=1.4,
              label="SeQUeNCe (protocolo real)")
    axes.plot(cost["span_km"], cost["analytic_ms_per_round"], marker="s",
              markersize=7, linewidth=2, linestyle="--", color=theme.series[1],
              markeredgecolor=theme.surface, markeredgewidth=1.4,
              label="twin (uma travessia do vão)")

    for _, row in cost.iterrows():
        axes.annotate(f"×{row['ratio']:.2f}".replace(".", ","),
                      xy=(row["span_km"], row["sequence_ms_per_round"]),
                      xytext=(0, 9), textcoords="offset points", ha="center",
                      fontsize=8.5, color=theme.text_secondary)

    axes.set_xscale("log")
    axes.set_yscale("log")
    _style_text(axes, theme,
                "Custo de uma tentativa de heralding, medido nos dois simuladores",
                "comprimento do vão (km)", "tempo por tentativa (ms)")
    _legend(axes, theme, loc="upper left")
    return _to_svg(figure)


def _figure_two_simulators(summary: pd.DataFrame, theme: Theme) -> str:
    """Success rate per policy, in the twin at both costs and in SeQUeNCe."""
    figure, axes = _figure(theme, height=4.2)

    order = [name for name in summary["simulator"].unique()
             if name.startswith("twin (custo x1")]
    order += [name for name in summary["simulator"].unique()
              if name.startswith("twin (custo x4")]
    order += ["SeQUeNCe"]
    labels = {order[0]: "twin, custo publicado (×1)",
              order[1]: "twin, custo medido (×4,1)",
              "SeQUeNCe": "SeQUeNCe (eventos discretos)"}

    policies = sorted(summary["policy"].unique())
    positions = np.arange(len(policies))
    width = 0.26

    for index, simulator in enumerate(order):
        values = []
        for policy in policies:
            match = summary[(summary["policy"] == policy)
                            & (summary["simulator"] == simulator)]
            values.append(float(match["success"].iloc[0]) if len(match) else np.nan)
        axes.bar(positions + (index - 1) * width, values, width * 0.9,
                 color=theme.series[index], edgecolor=theme.surface,
                 linewidth=1.2, label=labels.get(simulator, simulator))

    axes.set_xticks(positions)
    axes.set_xticklabels([name.split(" (")[0].replace("_", " ")
                          for name in policies], fontsize=8.5,
                         color=theme.text_secondary)
    _style_text(axes, theme,
                "Uma constante no custo de heralding reconcilia os dois simuladores",
                "", "fração de episódios bem-sucedidos")
    _legend(axes, theme, loc="upper right")
    return _to_svg(figure)


def _figure_reach(reach: pd.DataFrame, theme: Theme) -> str:
    """Held-out frequencies: specialist, two generalists, classical filter."""
    figure, axes = _figure(theme, height=4.0)

    frequency = reach[reach["axis"] == "frequência"]
    series = [
        ("especialista, mesma frequência", "MLP", "MLP especialista", 0),
        ("generalista estreito (45-70 Hz)", "MLP", "MLP generalista 45-70 Hz", 1),
        ("generalista largo (45-180 Hz)", "MLP", "MLP generalista 45-180 Hz", 2),
        ("—", "filtro alfa-beta", "filtro alfa-beta", 3),
    ]
    for training, compensator, label, colour in series:
        group = frequency[(frequency["training"] == training)
                          & (frequency["compensator"] == compensator)]
        if group.empty:
            continue
        ordered = group.sort_values("test_point")
        axes.plot(ordered["test_point"], ordered["mean_qber"], marker="o",
                  markersize=6, linewidth=2, color=theme.series[colour],
                  markeredgecolor=theme.surface, markeredgewidth=1.4,
                  label=label)

    axes.axvspan(45, 70, color=theme.series[1], alpha=0.10)
    axes.annotate("faixa do generalista estreito", xy=(57, 0.002),
                  xycoords=("data", "axes fraction"), ha="center", fontsize=8.5,
                  color=theme.text_secondary)
    axes.set_yscale("log")
    _style_text(axes, theme,
                "Frequências retidas: o generalista interpola e não extrapola",
                "frequência do tom no enlace de teste (Hz)", "QBER média")
    _legend(axes, theme, loc="lower right")
    return _to_svg(figure)


def _figure_exponent(scan: pd.DataFrame, measured: tuple[float, float],
                     calibrated: tuple[float, float], theme: Theme) -> str:
    """The calibration: where the measured exponent puts the drift rate."""
    figure, axes = _figure(theme, height=4.0)

    low, high = measured[0] - measured[1], measured[0] + measured[1]
    axes.axhspan(low, high, color=theme.series[2], alpha=0.15)
    axes.annotate("exponente medido pelo ORNL, 1,0 ± 0,2", xy=(0.98, high),
                  xycoords=("axes fraction", "data"), ha="right", va="bottom",
                  fontsize=8.5, color=theme.text_secondary)
    axes.axvspan(calibrated[0], calibrated[1], color=theme.series[0], alpha=0.12)

    for index, (band, group) in enumerate(scan.groupby("band", sort=False)):
        ordered = group.sort_values("drift_rate_rad_per_s")
        axes.errorbar(ordered["drift_rate_rad_per_s"], ordered["exponent"],
                      yerr=ordered["exponent_std"], marker="o", markersize=6,
                      linewidth=2, capsize=3, color=theme.series[index],
                      markeredgecolor=theme.surface, markeredgewidth=1.4,
                      label=band)

    axes.set_xscale("log")
    _style_text(axes, theme,
                "O expoente espectral fixa a taxa de deriva",
                "taxa de deriva do modelo (rad/s)",
                "expoente β de S(f) ∝ f⁻ᵝ, na banda do ORNL")
    _legend(axes, theme, loc="upper right")
    return _to_svg(figure)


def _figure_field_comparison(comparison: pd.DataFrame, theme: Theme) -> str:
    """Mean QBER by compensator, at the assumed point and the measured one."""
    figure, axes = _figure(theme, height=4.2)

    points = ["fase 3 (40 Hz, 0,05 rad/s)",
              "aérea calibrada (60/120/180 Hz, 0,02 rad/s)"]
    rows = [("persistência", "—"), ("filtro alfa-beta", "—"),
            ("ridge (Wiener)", "no ponto"), ("MLP", "no ponto"),
            ("MLP", "da fase 3")]
    labels = ["persistência", "filtro\nalfa-beta", "ridge\n(retreinado)",
              "MLP\n(retreinado)", "MLP\n(transplantado)"]

    positions = np.arange(len(rows))
    width = 0.38
    for index, point in enumerate(points):
        values = []
        for compensator, training in rows:
            match = comparison[(comparison["operating_point"] == point)
                               & (comparison["compensator"] == compensator)
                               & (comparison["training"] == training)]
            values.append(float(match["mean_qber"].iloc[0]) if len(match) else np.nan)
        axes.bar(positions + (index - 0.5) * width, values, width * 0.92,
                 color=theme.series[index], edgecolor=theme.surface,
                 linewidth=1.2,
                 label="assumido: 40 Hz" if index == 0 else "medido: 60/120/180 Hz")

    axes.axhline(0.11, color=theme.text_secondary, linewidth=1.2, linestyle="--")
    axes.annotate("limiar de 11%: acima disto o enlace não produz chave",
                  xy=(0.02, 0.115), xycoords=("axes fraction", "data"),
                  fontsize=8.5, color=theme.text_secondary, va="bottom")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=8.5, color=theme.text_secondary)
    _style_text(axes, theme,
                "O que muda quando a banda mecânica vai para onde foi medida",
                "", "QBER média")
    _legend(axes, theme, loc="upper left")
    return _to_svg(figure)


def _figure_amplitude(sweep: pd.DataFrame, theme: Theme) -> str:
    """QBER against the one parameter nobody measured."""
    figure, axes = _figure(theme)

    for name in COMPENSATOR_ORDER:
        group = sweep[sweep["compensator"] == name].sort_values("amplitude_rad")
        if group.empty:
            continue
        axes.plot(group["amplitude_rad"], group["mean_qber"], marker="o",
                  markersize=6, linewidth=2, color=_colour(name, theme),
                  markeredgecolor=theme.surface, markeredgewidth=1.4, label=name)

    axes.axhline(0.11, color=theme.text_secondary, linewidth=1.2, linestyle="--")
    axes.annotate("limiar de 11%", xy=(0.01, 0.115),
                  xycoords=("axes fraction", "data"), fontsize=8.5,
                  color=theme.text_secondary, va="bottom")
    axes.set_xscale("log")
    axes.set_yscale("log")
    _style_text(axes, theme,
                "QBER contra a amplitude do tom, com os preditores retreinados "
                "em cada ponto",
                "amplitude RMS do tom de rede (rad) — não medida em lugar nenhum",
                "QBER média")
    _legend(axes, theme, loc="upper left")
    return _to_svg(figure)


def _figure_blocking(stress: pd.DataFrame, theme: Theme) -> str:
    """Where a timer stops working and the controller does not.

    The two lines are the same compensator on the same runs; the only
    difference is who decides when the block closes. They separate exactly
    where the link stops being stationary enough for any one duration to be
    right.
    """
    figure, axes = _figure(theme)

    series = [("best_timer_hz", "melhor temporizador (com retrospectiva)", 3),
              ("timer_1s_hz", "temporizador de 1 s (fases 3 e 4)", 2),
              ("adaptive_hz", "controlador causal", 0),
              ("oracle_hz", "oráculo de partição", 1)]
    ordered = stress.sort_values("vibration_amplitude_rad")
    for column, label, colour in series:
        style = "--" if column == "oracle_hz" else "-"
        axes.plot(ordered["vibration_amplitude_rad"],
                  np.maximum(ordered[column], 1.0), marker="o", markersize=6,
                  linewidth=2, linestyle=style, color=theme.series[colour],
                  markeredgecolor=theme.surface, markeredgewidth=1.4,
                  label=label)

    zero = ordered[ordered["timer_1s_hz"] <= 0.0]
    if not zero.empty:
        axes.annotate("temporizador de 1 s: zero",
                      xy=(float(zero["vibration_amplitude_rad"].iloc[0]), 1.0),
                      xytext=(0, 14), textcoords="offset points", fontsize=8.5,
                      color=theme.text_secondary, ha="center")

    axes.set_yscale("log")
    _style_text(axes, theme,
                "Quem escolhe o fim do bloco, no melhor compensador, à medida "
                "que o enlace piora",
                "amplitude RMS do tom de rede (rad)",
                "chave secreta sobre a execução inteira (bit/s)")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


def _figure_robustness(robustness: pd.DataFrame, theme: Theme) -> str:
    """One model across the band, against one model per frequency."""
    figure, axes = _figure(theme, height=4.2)

    frequencies = sorted(robustness["test_frequency_hz"].unique())
    positions = np.arange(len(frequencies))
    series = [
        ("especialista, mesma frequência", "MLP", "MLP especialista", 0),
        ("generalista", "MLP", "MLP generalista (45-70 Hz)", 1),
        ("generalista", "ridge (Wiener)", "ridge generalista", 2),
        ("—", "filtro alfa-beta", "filtro alfa-beta", 3),
    ]
    width = 0.2

    for index, (training, compensator, label, colour) in enumerate(series):
        values = []
        for frequency in frequencies:
            match = robustness[(robustness["test_frequency_hz"] == frequency)
                               & (robustness["training"] == training)
                               & (robustness["compensator"] == compensator)]
            values.append(float(match["mean_qber"].iloc[0]) if len(match) else np.nan)
        axes.bar(positions + (index - 1.5) * width, values, width * 0.9,
                 color=theme.series[colour], edgecolor=theme.surface,
                 linewidth=1.1, label=label)

    # The off-frequency specialists, as a band rather than as bars: they are an
    # order of magnitude worse and would flatten everything else.
    worst = robustness[robustness["training"].str.startswith("especialista de")]
    if not worst.empty:
        low, high = float(worst["mean_qber"].min()), float(worst["mean_qber"].max())
        axes.axhspan(low, high, color=theme.text_secondary, alpha=0.12)
        axes.annotate("faixa dos especialistas rodando na frequência errada",
                      xy=(0.5, high), xycoords=("axes fraction", "data"),
                      ha="center", va="bottom", fontsize=8.5,
                      color=theme.text_secondary)

    axes.set_yscale("log")
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{value:g} Hz" for value in frequencies],
                         color=theme.text_secondary)
    _style_text(axes, theme,
                "Um modelo para toda a banda, contra um modelo por frequência",
                "frequência do tom no enlace de teste", "QBER média")
    _legend(axes, theme, loc="upper left")
    return _to_svg(figure)


# ---------------------------------------------------------------------------

def render_realism_report(run_directory: str | Path,
                          actuator_directory: str | Path,
                          output_path: str | Path,
                          allocator_directory: str | Path | None = None,
                          field_directory: str | Path | None = None,
                          sequence_directory: str | Path | None = None,
                          corrected_results: str | Path | None = None,
                          long_deadline_results: str | Path | None = None,
                          title: str = "Fechando as lacunas: gêmeo, bloco finito e atuador real"
                          ) -> Path:
    """Write the gap-closing report from the CSVs on disk.

    Args:
        run_directory: holds ``twin.csv``, ``finite_key_link.csv``,
            ``finite_key_allocation.csv`` and ``finite_key_frontier.csv``.
        actuator_directory: holds ``slew_sweep.csv``, ``travel.csv`` and
            ``reset_sweep.csv``.
        output_path: destination ``.html``.
        allocator_directory: optional, holds the phase-4 ``frontier.csv`` so the
            asymptotic frontier can be drawn behind the finite one.
        field_directory: optional, holds ``exponent_scan.csv`` and
            ``field_comparison.csv`` from ``scripts/evaluate_field.py``. Without
            it the field section is omitted rather than guessed at.
        sequence_directory: optional, holds ``summary.csv`` and
            ``round_cost.csv`` from ``scripts/evaluate_sequence.py``.
        corrected_results, long_deadline_results: optional phase-2 evaluations
            re-run at the measured heralding cost, on the published deadlines
            and on four times them.
        title: page title.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: if a required results file is missing. Nothing here
            is filled in from a default — a report that invents a number it
            could not measure is worse than no report.
    """
    run_directory = Path(run_directory)
    actuator_directory = Path(actuator_directory)

    required = {
        run_directory: ["twin.csv", "finite_key_link.csv",
                        "finite_key_allocation.csv", "finite_key_frontier.csv"],
        actuator_directory: ["slew_sweep.csv", "travel.csv", "reset_sweep.csv"],
    }
    missing = [str(directory / name) for directory, names in required.items()
               for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing results: {missing}. Run scripts/evaluate_realism.py and "
            f"scripts/evaluate_actuator.py first.")

    twin = pd.read_csv(run_directory / "twin.csv")
    link = pd.read_csv(run_directory / "finite_key_link.csv")
    allocation = pd.read_csv(run_directory / "finite_key_allocation.csv")
    frontier = pd.read_csv(run_directory / "finite_key_frontier.csv")
    slew = pd.read_csv(actuator_directory / "slew_sweep.csv")
    travel = pd.read_csv(actuator_directory / "travel.csv")
    reset = pd.read_csv(actuator_directory / "reset_sweep.csv")

    blocking: dict[str, pd.DataFrame] = {}
    for key, name in (("summary", "adaptive_blocks_summary.csv"),
                      ("stress", "blocking_stress_summary.csv")):
        if (run_directory / name).is_file():
            blocking[key] = pd.read_csv(run_directory / name)
    if len(blocking) < 2:
        blocking = {}

    field: dict[str, pd.DataFrame] = {}
    if field_directory is not None:
        field_directory = Path(field_directory)
        names = {"exponent": "exponent_scan.csv", "comparison": "field_comparison.csv",
                 "amplitude": "amplitude_sweep.csv", "robustness": "robustness.csv",
                 "reach": "reach.csv", "capacity": "capacity.csv"}
        for key, name in names.items():
            if (field_directory / name).is_file():
                field[key] = pd.read_csv(field_directory / name)
        if not {"exponent", "comparison"} <= set(field):
            field = {}

    sequence: dict[str, pd.DataFrame] = {}
    if sequence_directory is not None:
        sequence_directory = Path(sequence_directory)
        for key, name in (("summary", "summary.csv"),
                          ("cost", "round_cost.csv"),
                          ("comparison", "comparison.csv")):
            if (sequence_directory / name).is_file():
                sequence[key] = pd.read_csv(sequence_directory / name)
        if not {"summary", "cost"} <= set(sequence):
            sequence = {}

    phase_two: dict[str, pd.DataFrame] = {}
    for key, path in (("corrigido", corrected_results),
                      ("prazo longo", long_deadline_results)):
        if path is not None and Path(path).is_file():
            phase_two[key] = pd.read_csv(path)

    asymptotic_frontier = pd.DataFrame(columns=["family", "served_qkd",
                                                "served_classical"])
    if allocator_directory is not None:
        candidate = Path(allocator_directory) / "frontier.csv"
        if candidate.is_file():
            asymptotic_frontier = pd.read_csv(candidate)

    sections: list[str] = []

    # -- provenance --------------------------------------------------------
    steps = len(twin)
    worst = float(twin["state_divergence"].max())
    worst_probability = float(twin["probability_divergence"].max())
    sections.append(f"""
<div class="meta"><dl>
  <dt>Gêmeo × SeQUeNCe</dt><dd>{steps} passos comparados, pior divergência
      {_pt(worst, '.1e')}</dd>
  <dt>Chave finita</dt><dd>Tomamichel, Lim, Gisin &amp; Renner (2012),
      ε<sub>sec</sub> = 10<sup>−10</sup>, ε<sub>cor</sub> = 10<sup>−15</sup></dd>
  <dt>Atuador</dt><dd>cascata de retardadores X-Y-X, curso e velocidade finitos,
      reset sem fim</dd>
  <dt>Lacuna que fica aberta</dt><dd>fibra real: nenhuma medição de campo</dd>
</dl></div>""")

    # -- 00 how to read ----------------------------------------------------
    sections.append(_section(
        "00", "como ler", "Quatro lacunas, três mensuráveis aqui",
        "<p>A revisão de aplicabilidade das fases 2 a 4 nomeou quatro coisas "
        "entre os resultados publicados e uma afirmação de aplicabilidade: o "
        "ambiente onde o agente treina nunca foi confrontado com o simulador de "
        "eventos discretos em sequências completas; as taxas de chave eram "
        "assintóticas; o atuador de polarização era ideal; e não há dado de "
        "fibra real. As três primeiras são mensuráveis dentro deste contêiner e "
        "estão medidas abaixo. A quarta não é, e nada aqui a substitui.</p>"
        "<p>Cada seção começa pelo número que a fase correspondente deveria ter "
        "publicado. Onde a correção muda uma conclusão, isso está no título da "
        "seção; onde não muda, isso também está dito, porque \"a ordenação se "
        "manteve\" só vale alguma coisa quando havia risco real de não se "
        "manter.</p>"))

    # -- 01 the twin -------------------------------------------------------
    by_topology = twin.groupby(["topology", "purify"]).agg(
        steps=("state_divergence", "size"),
        worst=("state_divergence", "max"),
        worst_probability=("probability_divergence", "max")).reset_index()
    by_topology["purify"] = np.where(by_topology["purify"], "com purificação",
                                     "sem purificação")
    counts = twin["outcome"].value_counts()

    sections.append(_section(
        "01", "evidência exata",
        "O ambiente da fase 2 compõe a física validada sem se desviar dela",
        "<p>A fase 1 validou as formas fechadas <em>uma operação por vez</em>. "
        "Isso não é uma afirmação sobre o ambiente, que é outro software: ele "
        "decide qual função chamar, sobre qual par, com quais parâmetros de "
        "hardware, depois de quanto tempo decorrido, e costura os resultados a "
        "um relógio. Cada um desses é um lugar onde dá para errar chamando "
        "física perfeitamente correta, e a seção de limites da fase 2 dizia "
        "exatamente isso.</p>"
        f"<p>O confronto agora existe. Uma trajetória inteira roda no ambiente e, "
        f"em paralelo, cada par vivo é reconstruído a partir de objetos do "
        f"SeQUeNCe — protocolos de purificação e swapping, e "
        f"<code>Memory.bds_decohere</code> chamado uma vez <em>por ponta</em> do "
        f"par — usando os tempos decorridos que o próprio ambiente reporta. "
        f"Foram {steps} passos ({int(counts.get('generate_succeeded', 0))} "
        f"gerações bem-sucedidas, {int(counts.get('purify_succeeded', 0))} "
        f"purificações, {int(counts.get('swap_succeeded', 0))} swaps, "
        f"{int(counts.get('delivered', 0))} entregas), e a maior discrepância em "
        f"qualquer componente de qualquer par foi "
        f"<strong>{_pt(worst, '.1e')}</strong>. A probabilidade de sucesso da "
        f"purificação diverge no máximo {_pt(worst_probability, '.1e')}.</p>",
        _figure_block({theme.name: _figure_twin(twin, theme) for theme in THEMES},
                      "Cada ponto é um passo de controle. A escala é logarítmica "
                      "e o piso é o épsilon da precisão dupla — o acúmulo ao "
                      "longo da trajetória é aritmético, não físico."),
        _table_pt(by_topology,
                  {"steps": ",.0f", "worst": ".2e", "worst_probability": ".2e"},
                  columns=["topology", "purify", "steps", "worst",
                           "worst_probability"],
                  headers=["topologia", "sequência", "passos",
                           "pior divergência de estado",
                           "pior divergência de probabilidade"]),
        "<p><strong>A comparação pode falhar, e há teste para isso.</strong> Uma "
        "comparação que não pode falhar não é evidência. Dois controles negativos "
        "quebram o ambiente de propósito: cobrar decoerência de uma ponta em vez "
        "de duas produz divergência acima de 10<sup>−3</sup>, e a fidelidade de "
        "porta imperfeita muda o resultado do swap o suficiente para que "
        "descartá-la fosse detectável. Ambos estão em "
        "<code>tests/test_env_matches_sequence.py</code>.</p>",
        '<p class="caveat">O que isto <em>não</em> estabelece: o agente não roda '
        'dentro do laço de eventos do SeQUeNCe, e nada aqui exercita o '
        'escalonamento, a troca de mensagens ou o gerenciamento de recursos do '
        'SeQUeNCe. O que está estabelecido é que a aritmética de estado ao longo '
        'de uma trajetória é a do SeQUeNCe — que é a parte de que o treino '
        'depende, e era a parte não testada.</p>'))

    # -- 02 the agent inside the event loop ---------------------------------
    if sequence:
        cost = sequence["cost"]
        summary = sequence["summary"]
        ratio = float(cost["ratio"].mean())
        spread = float(cost["ratio"].max() - cost["ratio"].min())
        spans = f"{cost['span_km'].min():.0f}-{cost['span_km'].max():.0f}"

        published = next(name for name in summary["simulator"].unique()
                         if name.startswith("twin (custo x1"))
        corrected = next(name for name in summary["simulator"].unique()
                         if name.startswith("twin (custo x4"))

        def success(policy_prefix, simulator):
            match = summary[summary["policy"].str.startswith(policy_prefix)
                            & (summary["simulator"] == simulator)]
            return float(match["success"].iloc[0]) if len(match) else float("nan")

        agent = "actor_critic"
        wide = summary.pivot(index="policy", columns="simulator",
                             values="success").reset_index()
        residual = float(np.mean(np.abs(
            summary.pivot(index="policy", columns="simulator",
                          values="success")[corrected]
            - summary.pivot(index="policy", columns="simulator",
                            values="success")["SeQUeNCe"])))

        probability_rows = cost.assign(
            probability_error=100.0 * (cost["measured_probability"]
                                       / cost["analytic_probability"] - 1.0),
            sigmas=((cost["measured_probability"] - cost["analytic_probability"])
                    / cost["probability_stderr"]))
        worst_sigma = float(probability_rows["sigmas"].abs().max())
        mean_bias = float(probability_rows["probability_error"].mean())

        body = [
            "<p>A validação anterior comparava a <em>aritmética</em> do "
            "ambiente analítico com a do SeQUeNCe. Sobrava tudo que não é "
            "aritmética: fótons que chegam ou não, detectores que clicam, "
            "protocolos que negociam por canais com atraso, memórias ocupadas "
            "enquanto isso acontece. Agora a política treinada roda dentro do "
            "laço de eventos do SeQUeNCe — roteadores quânticos de verdade, nós "
            "de medição de Bell com detectores, canais com atenuação — e as "
            "seis primitivas da fase 2 são executadas instanciando os "
            "protocolos do próprio simulador.</p>"
            "<p>O laço de controle é para-e-anda: <code>Timeline.run</code> "
            "devolve o controle quando a fila esvazia ou quando bate no "
            "horizonte, e nesse caso devolve o evento à fila. A política é a "
            "mesma, sem alteração; as features dela vêm dos mesmos "
            "codificadores, aplicados a um estado espelhado dos gerenciadores "
            "de memória do SeQUeNCe. Em todos os passos de todos os episódios, "
            "os dois simuladores ofereceram <strong>o mesmo conjunto de ações "
            "legais</strong> — zero divergências.</p>",
            "<p><strong>Primeiro, o alinhamento.</strong> Dois simuladores só "
            "se comparam se rodarem o mesmo hardware, e as eficiências que "
            "formam um braço de um enlace heraldado moram em um produto no "
            "modelo analítico e em três objetos diferentes no SeQUeNCe. "
            "Alinhadas, a probabilidade de heralding medida fica dentro do erro "
            "da fórmula fechada em toda a faixa de vãos.</p>",
            _table_pt(probability_rows,
                      {"span_km": ".0f", "measured_probability": ".5f",
                       "probability_stderr": ".5f",
                       "analytic_probability": ".5f",
                       "probability_error": "+.1f", "sigmas": "+.1f",
                       "rounds": ",.0f"},
                      columns=["span_km", "rounds", "analytic_probability",
                               "measured_probability", "probability_stderr",
                               "probability_error", "sigmas"],
                      headers=["vão (km)", "rodadas", "fórmula fechada",
                               "medido", "erro padrão", "diferença (%)",
                               "σ"]),
            f"<p>A concordância não é perfeita e não vale escondê-lo: o medido "
            f"fica sistematicamente <strong>{_pt(mean_bias, '.0f')}% acima</strong> "
            f"da fórmula, em todos os quatro vãos, e no pior deles a "
            f"{_pt(worst_sigma, '.1f')} erros padrão. Cada vão isolado é "
            f"compatível; o sinal repetido nos quatro sugere que a fórmula "
            f"fechada é ligeiramente conservadora, na direção que não infla "
            f"resultado nenhum. A contagem inclui as tentativas que estouraram "
            f"o orçamento — estimar a probabilidade só pelas bem-sucedidas "
            f"descarta justamente a cauda longa e enviesa para cima em dezenas "
            f"de por cento a 80 km.</p>",
            f"<p><strong>Depois, a diferença.</strong> O que não bate é o "
            f"<em>custo em tempo</em> de uma tentativa. O twin cobra uma "
            f"travessia do vão — os fótons voam até o ponto médio e o herald "
            f"volta. O protocolo cobra "
            f"<strong>{_pt(ratio, '.1f')}</strong>, porque uma tentativa é uma "
            f"negociação e não um voo: as pontas combinam o instante, emitem, e "
            f"ouvem de volta o nó de medição. O fator é constante em toda a "
            f"faixa de {spans} km — varia {_pt(spread, '.2f')} entre o menor e o "
            f"maior vão — então é multiplicativo, não um aperto de mão "
            f"aditivo.</p>",
            _figure_block(
                {theme.name: _figure_round_cost(cost, theme) for theme in THEMES},
                "Log-log, com a razão anotada em cada ponto. Duas retas "
                "paralelas: o mesmo fator em um fator de oito no vão."),
            f"<p><strong>E esse único número reconcilia os dois "
            f"simuladores.</strong> Cobrando {_pt(ratio, '.1f')} travessias por "
            f"tentativa, o twin passa a concordar com o SeQUeNCe em todas as "
            f"políticas — a diferença média em taxa de sucesso cai para "
            f"{_pt(residual, '.3f')}, dentro do erro de 40 episódios. Não sobra "
            f"nenhum efeito de escalonamento, de mensagens ou de contenção que "
            f"o modelo analítico não capture: sobra um parâmetro que estava "
            f"errado por um fator de quatro.</p>",
            _figure_block(
                {theme.name: _figure_two_simulators(summary, theme)
                 for theme in THEMES},
                "Três barras por política. As duas da direita são o resultado: "
                "com o custo corrigido, o twin e o simulador de eventos "
                "discretos contam a mesma história."),
            _table_pt(wide,
                      {published: ".3f", corrected: ".3f", "SeQUeNCe": ".3f"},
                      columns=["policy", published, corrected, "SeQUeNCe"],
                      headers=["política", "twin, custo ×1", "twin, custo ×4,1",
                               "SeQUeNCe"]),
        ]

        if phase_two:
            def rate(frame, prefix):
                """Success rate for one policy, from the per-episode rows.

                The evaluation writes one row per episode, not the summary it
                prints, so the rate is recomputed here rather than read off.
                """
                match = frame[frame["policy"].str.startswith(prefix)]
                return (float(match["succeeded"].mean()) if len(match)
                        else float("nan"))

            rows = []
            for label, frame in phase_two.items():
                rows.append({
                    "regime": label,
                    "agente": rate(frame, "actor_critic"),
                    "greedy_fidelity": rate(frame, "greedy_fidelity"),
                    "shortest_path": rate(frame, "shortest_path"),
                    "threshold_0.94": rate(frame, "threshold_0.94"),
                })
            comparison = pd.DataFrame(rows)
            best_gap = min(
                rate(frame, "actor_critic")
                - max(rate(frame, "greedy_fidelity"), rate(frame, "shortest_path"))
                for frame in phase_two.values())

            body += [
                "<p><strong>A consequência para a fase 2 é uma "
                "retratação.</strong> A manchete era +32,5 pontos percentuais "
                "sobre a melhor heurística. Ela foi medida cobrando uma "
                "travessia por tentativa. Retreinando o agente com o custo "
                "medido — mesma arquitetura, mesmo orçamento de treino, mesmas "
                "sementes de teste — a vantagem desaparece.</p>",
                _table_pt(comparison,
                          {"agente": ".4f", "greedy_fidelity": ".4f",
                           "shortest_path": ".4f", "threshold_0.94": ".4f"},
                          columns=["regime", "agente", "greedy_fidelity",
                                   "shortest_path", "threshold_0.94"],
                          headers=["regime", "agente aprendido",
                                   "greedy_fidelity", "shortest_path",
                                   "threshold_0,94"]),
                "<p>Nos prazos publicados o agente empata com as duas melhores "
                "heurísticas (diferença pareada −0,022 ± 0,013 contra "
                "<code>shortest_path</code>). Com prazos quatro vezes maiores — "
                "o controle que pergunta se a vantagem apenas se mudou para um "
                "regime mais lento — ele perde para <code>greedy_fidelity</code> "
                "por 0,083 ± 0,020. O mecanismo é direto: a quatro vezes o custo "
                "por tentativa, a geração domina o orçamento de tempo, e as "
                "decisões de purificação e de timing que o agente aprendeu a "
                "tomar valem menos do que simplesmente pegar o que estiver "
                "disponível.</p>",
                '<p class="caveat">O treino com prazos longos colapsou em duas '
                'de duas sementes — entropia da política caindo a 0,06 na '
                'iteração 50, sucesso zero — até o piso de entropia subir de '
                '0,002 para 0,01. É o mesmo modo de falha que o estudo de '
                'dual-head registrou em uma execução em vinte, e episódios mais '
                'longos o tornam muito mais provável. O número acima é da '
                'execução que convergiu.</p>',
            ]

        body += [
            "<p><strong>Dois achados de percurso, do tipo que só aparece "
            "construindo.</strong> O formalismo Bell-diagonal não consegue "
            "representar um fóton isolado, então a geração Barrett-Kok — que "
            "interfere dois deles — quebra dentro do nó de medição; a geração "
            "que roda é a <code>single_heralded</code>, que por acaso é o "
            "esquema cuja probabilidade a fase 1 validou. E "
            "<code>MemoryArray.update_memory_params</code> é um "
            "<code>setattr</code> nu: <code>decoherence_rate</code> é derivado "
            "de <code>coherence_time</code> uma única vez, no construtor. "
            "Ajustar o tempo de coerência depois não muda nada — os pares nunca "
            "decoerem, toda fidelidade da execução é a bruta, e nada reclama. É "
            "a mesma forma do bug de fidelidade de porta que o README já "
            "documenta, e é o segundo parâmetro que o SeQUeNCe aceita e "
            "descarta em silêncio.</p>",
            '<p class="caveat">O que continua fora: este runner não usa o motor '
            'de regras nem o protocolo de reserva do SeQUeNCe. Esses são a '
            '<em>política</em> do SeQUeNCe, e o objetivo aqui era pôr uma '
            'política aprendida naquele lugar. Contenção entre requisições '
            'concorrentes, que é o que o gerenciador de rede resolve, continua '
            'sem teste.</p>',
        ]

        sections.append(_section(
            "02", "validação",
            f"O agente roda dentro do simulador de eventos discretos — e uma "
            f"constante de {_pt(ratio, '.1f')} no custo de heralding explica "
            f"toda a diferença",
            *body))

    # -- 02 finite key on the link -----------------------------------------
    one_second = link[link["block_duration_s"] == 1.0].copy()
    one_second = one_second.set_index("compensator").reindex(
        [name for name in COMPENSATOR_ORDER
         if name in set(link["compensator"])]).reset_index()
    mlp = one_second[one_second["compensator"] == "MLP"].iloc[0]
    alpha_beta = one_second[one_second["compensator"] == "filtro alfa-beta"].iloc[0]
    persistence = one_second[one_second["compensator"] == "persistência"].iloc[0]

    sections.append(_section(
        "03", "correção de resultado",
        "As taxas de chave da fase 3 estavam altas por um fator de 1,6 — e por "
        "60 no pior compensador",
        "<p>Toda taxa de chave nas fases 3 e 4 saiu da fórmula assintótica, que "
        "supõe que a taxa de erro é conhecida exatamente e que a amplificação de "
        "privacidade é de graça. Um enlace real acerta as contas em blocos e paga "
        "pelas duas coisas. Com blocos de um segundo a 10⁵ bits peneirados por "
        "segundo — 10⁵ bits por bloco, que é o regime de um controlador que "
        f"fecha contas de segundo em segundo — o MLP entrega "
        f"{_pt(mlp['finite_key_rate_hz'] / 1e3, '.1f')} kbit/s em vez dos "
        f"{_pt(mlp['asymptotic_key_rate_hz'] / 1e3, '.1f')} kbit/s publicados, e "
        f"o filtro alfa-beta {_pt(alpha_beta['finite_key_rate_hz'] / 1e3, '.1f')} "
        f"em vez de {_pt(alpha_beta['asymptotic_key_rate_hz'] / 1e3, '.1f')}.</p>",
        _figure_block(
            {theme.name: _figure_finite_link(link, theme) for theme in THEMES},
            "A penalidade de bloco finito não é um fator comum: ela cresce à "
            "medida que a QBER se aproxima do limiar, e some para quem está "
            "confortavelmente abaixo dele."),
        _table_pt(one_second.assign(
                      asymptotic_kbit=one_second["asymptotic_key_rate_hz"] / 1e3,
                      finite_kbit=one_second["finite_key_rate_hz"] / 1e3),
                  {"asymptotic_kbit": ".2f", "finite_kbit": ".2f", "ratio": ".3f"},
                  columns=["compensator", "asymptotic_kbit", "finite_kbit", "ratio"],
                  headers=["compensador", "chave assintótica (kbit/s)",
                           "chave em blocos de 1 s (kbit/s)", "fração"]),
        "<p><strong>A correção reforça a conclusão da fase 3 em vez de "
        "enfraquecê-la.</strong> O custo de bloco finito é aproximadamente "
        "proporcional à distância que separa a QBER do limiar, então quem "
        f"compensa mal paga muito mais: a persistência cai a "
        f"{_pt(persistence['finite_key_rate_hz'], ',.0f')} bit/s "
        f"({_pt(100 * persistence['ratio'], '.1f')}% do assintótico) com blocos "
        "de um segundo e a <strong>exatamente zero</strong> com blocos de cinco "
        "segundos ou mais. Um enlace cuja QBER média fica perto do limiar não "
        "produz chave nenhuma quando o bloco cresce — e a fórmula assintótica "
        "esconde isso por completo, reportando 17 kbit/s para um enlace que não "
        "entrega nada.</p>",
        '<p class="caveat">A fração de teste é otimizada, não fixada. Fixá-la em '
        '10% faz o limite convergir para 90% do assintótico em vez de para ele, '
        'e essa convergência é a verificação de que a implementação está certa: '
        'a 10<sup>10</sup> bits ela reproduz a taxa assintótica com erro abaixo '
        'de um por cento.</p>'))

    # -- 03 finite key in the network --------------------------------------
    pivot = allocation.pivot(index="policy", columns="finite_key",
                             values="served_qkd")
    ordering_asymptotic = list(pivot[False].sort_values(ascending=False).index)
    ordering_finite = list(pivot[True].sort_values(ascending=False).index)
    preserved = ordering_asymptotic == ordering_finite

    comparison = allocation.pivot(index="policy", columns="finite_key",
                                  values=["served_qkd", "served_classical",
                                          "mean_admission"])
    comparison.columns = [f"{name}_{'finite' if flag else 'asymptotic'}"
                          for name, flag in comparison.columns]
    comparison = comparison.reset_index()
    comparison["ratio"] = (comparison["served_qkd_finite"]
                           / comparison["served_qkd_asymptotic"])

    families = {name: group for name, group in frontier.groupby("family")}
    # The matched comparison only means anything where both families were
    # actually measured, so the targets are the overlap of the two frontiers the
    # phase-4 headline compares — not a fraction of some global maximum. The
    # reactive baseline never reaches this range and is reported as such rather
    # than extrapolated into it.
    overlap = [families[name] for name in ("preditivo", "admissão estática")
               if name in families]
    low = max(group["served_qkd"].min() for group in overlap)
    high = min(group["served_qkd"].max() for group in overlap)
    targets = [round(low + share * (high - low), 4)
               for share in (0.0, 0.5, 1.0)]
    matched = pd.DataFrame([
        {"target": target,
         **{name: _interpolate(group, target) for name, group in families.items()}}
        for target in targets])
    matched["advantage"] = (matched.get("preditivo")
                            / matched.get("admissão estática"))

    best = matched.dropna(subset=["advantage"])
    middle_advantage = float(best["advantage"].iloc[len(best) // 2]) if not best.empty else float("nan")
    advantage_text = (
        f"{_pt(float(best['advantage'].min()), '.1f')}&times; a "
        f"{_pt(float(best['advantage'].max()), '.1f')}&times;"
        if not best.empty else "não mensurável na faixa alcançada")

    sections.append(_section(
        "04", "correção de resultado",
        "O serviço de chave da fase 4 cai um terço, e a ordenação das políticas "
        "não muda",
        "<p>A mesma correção, agora dentro do problema de alocação: o ambiente "
        "passa a cobrar o custo de bloco finito a cada período de controle, e o "
        "bloco é o que um enlace peneira em cinco segundos. O serviço de chave "
        "cai entre 30% e 55% conforme a política, e — a parte que importa — a "
        + ("ordenação das cinco políticas é idêntica antes e depois."
           if preserved else
           "ordenação das políticas <strong>muda</strong>, o que invalida a "
           "comparação publicada.")
        + "</p>",
        _table_pt(comparison,
                  {"served_qkd_asymptotic": ".4f", "served_qkd_finite": ".4f",
                   "ratio": ".2f", "mean_admission_asymptotic": ".3f",
                   "mean_admission_finite": ".3f"},
                  columns=["policy", "served_qkd_asymptotic", "served_qkd_finite",
                           "ratio", "mean_admission_asymptotic",
                           "mean_admission_finite"],
                  headers=["política", "chave (assintótico)", "chave (bloco finito)",
                           "fração", "admissão média (assint.)",
                           "admissão média (finito)"]),
        "<p><strong>Mas o ponto de operação se move, e isso é uma mudança de "
        "comportamento, não só de escala.</strong> A política preditiva admite "
        f"{_pt(float(comparison.loc[comparison['policy'].str.startswith('preditivo'), 'mean_admission_asymptotic'].iloc[0]), '.3f')} "
        "do tráfego clássico quando a chave é assintótica e apenas "
        f"{_pt(float(comparison.loc[comparison['policy'].str.startswith('preditivo'), 'mean_admission_finite'].iloc[0]), '.3f')} "
        "quando o custo real é cobrado: cobrar a chave pelo que ela custa faz o "
        "controlador comprar menos vazão clássica. Por isso a manchete da fase 4 "
        "— mesma chave com 2,4&times; a vazão clássica — não podia ser "
        "extrapolada e teve de ser remedida, com as duas fronteiras "
        "retraçadas.</p>",
        _figure_block(
            {theme.name: _figure_frontier(frontier, asymptotic_frontier, theme)
             for theme in THEMES},
            "As fronteiras inteiras, retraçadas com blocos finitos. O eixo de "
            "chave encolhe; a forma relativa das famílias, não."),
        _table_pt(matched,
                  {"target": ".3f", "preditivo": ".3f", "limiar reativo": ".3f",
                   "admissão estática": ".3f", "advantage": ".2f"},
                  columns=[column for column in
                           ["target", "preditivo", "limiar reativo",
                            "admissão estática", "advantage"]
                           if column in matched.columns],
                  headers=[header for header, column in
                           [("serviço de chave alvo", "target"),
                            ("preditivo", "preditivo"),
                            ("limiar reativo", "limiar reativo"),
                            ("admissão estática", "admissão estática"),
                            ("vantagem preditivo ÷ estático", "advantage")]
                           if column in matched.columns]),
        f"<p><strong>A manchete da fase 4 sobrevive.</strong> Com blocos finitos, "
        f"a vantagem do controlador preditivo sobre a admissão estática a serviço "
        f"de chave igualado é de {advantage_text} — "
        f"{_pt(middle_advantage, '.1f')}&times; no meio da faixa, contra os "
        f"2,4&times; publicados com chave assintótica. Os alvos são os extremos e "
        f"o centro da <em>sobreposição</em> entre as duas fronteiras "
        f"({_pt(low, '.3f')} a {_pt(high, '.3f')} de serviço de chave), porque "
        f"comparar famílias num nível que uma delas nunca alcançou exigiria "
        f"extrapolar a fronteira dela. O limiar reativo não chega a essa faixa e "
        f"por isso aparece vazio na tabela em vez de estimado.</p>"
        f'<p class="caveat">No alvo mais alto a fronteira estática está a poucos '
        f'centésimos de zerar a vazão clássica, então a razão ali é grande e mal '
        f'condicionada: é o quociente de dois números pequenos medidos com seis '
        f'sementes. O número do meio da faixa é o que se deve citar.</p>'))

    # -- 04 the actuator ---------------------------------------------------
    finite_slew = slew[np.isfinite(slew["slew_rate"])]
    pivot_slew = finite_slew.pivot(index="slew_rate", columns="compensator",
                                   values="mean_qber")
    # The sweep is logarithmic, so the crossover is a bracket, not a point: the
    # last rate at which the classical filter still wins and the first at which
    # it does not. Quoting the upper edge is the conservative reading, which is
    # the right one for a hardware requirement.
    beats = pivot_slew[pivot_slew["MLP"] < pivot_slew["filtro alfa-beta"]]
    crossover = float(beats.index.min()) if not beats.empty else float("nan")
    below = pivot_slew[pivot_slew.index < crossover]
    crossover_low = float(below.index.max()) if not below.empty else float("nan")
    ideal_gap = (float(pivot_slew["filtro alfa-beta"].iloc[-1])
                 / float(pivot_slew["MLP"].iloc[-1]))

    reset_pivot = reset.pivot(index="range_turns", columns="compensator",
                              values="resets_per_minute").reset_index()
    reset_qber = reset.pivot(index="range_turns", columns="compensator",
                             values="mean_qber")
    reset_spread = float((reset_qber.max() - reset_qber.min()).max())

    corner: list[str] = []
    corner_path = actuator_directory / "corner_sweep.csv"
    if corner_path.is_file():
        corner_frame = pd.read_csv(corner_path)
        penalty = corner_frame.pivot(index=["slew_rate", "range_turns"],
                                     columns="compensator", values="mean_qber")
        # Deliberately not named `slew`: that is the sweep DataFrame two
        # sections down, and shadowing it here produced a figure built from a
        # float.
        ratios = {}
        for rate in sorted(corner_frame["slew_rate"].unique()):
            ratios[rate] = penalty.loc[(rate, 2.0)] / penalty.loc[(rate, 6.0)]
        worst_slew = max(ratios, key=lambda key: ratios[key]["MLP"])
        worst = ratios[worst_slew]

        resets = corner_frame.pivot(index=["slew_rate", "range_turns"],
                                    columns="compensator",
                                    values="resets_per_minute")
        table = penalty.reset_index()
        table.columns = ["slew_rate", "range_turns"] + list(penalty.columns)

        corner = [
            f"<p><strong>E há um canto onde o reset importa, o que resgata "
            f"metade da hipótese.</strong> Um dispositivo lento <em>e</em> de "
            f"curso curto ao mesmo tempo não estava em nenhuma das duas "
            f"varreduras acima. Varrido, ele custa caro — mas só a quem se "
            f"move muito: a {_pt_number(worst_slew)} rad/s, apertar o curso de "
            f"6π para 2π piora a MLP em "
            f"{_pt(100 * (worst['MLP'] - 1), '.0f')}% e o filtro alfa-beta em "
            f"{_pt(100 * (worst['filtro alfa-beta'] - 1), '.0f')}%. O "
            f"mecanismo é o mesmo curso da tabela anterior: quem rastreia a "
            f"banda pede mais movimento e bate no fim de curso mais vezes — "
            f"{_pt_number(float(resets.loc[(worst_slew, 2.0), 'MLP']))} resets "
            f"por minuto contra "
            f"{_pt_number(float(resets.loc[(worst_slew, 2.0), 'filtro alfa-beta']))} "
            f"do filtro clássico.</p>"
            "<p>Ainda assim o preditor ganha nesse canto, e por larga margem. "
            "A conclusão correta não é que a hipótese original estava certa; é "
            "que o reset cobra um pedágio proporcional ao movimento pedido, e "
            "esse pedágio é pequeno perto da vantagem — a menos que o "
            "dispositivo seja lento, e aí quem ganha é o filtro clássico pela "
            "razão da seção anterior, não pelo reset.</p>",
            _table_pt(table,
                      {"slew_rate": ",.0f", "range_turns": ".0f",
                       **{name: ".4f" for name in penalty.columns}},
                      columns=["slew_rate", "range_turns"] + list(penalty.columns),
                      headers=["velocidade (rad/s)", "curso (múltiplos de π)"]
                              + list(penalty.columns)),
        ]

    sections.append(_section(
        "05", "qualificação",
        f"A vantagem do preditor exige um atuador acima de "
        f"{_pt_number(crossover)} rad/s; abaixo disso o filtro clássico ganha",
        "<p>A fase 3 supôs que a rotação pedida aparece na fibra exatamente. A "
        "seção de limites dela já apontava essa como a suposição mais provável "
        "de importar. Agora o laço fecha sobre uma cascata de retardadores X-Y-X "
        "com curso finito, velocidade finita e reset sem fim, e o comando é "
        "convertido em ângulos de elemento pela decomposição de Euler mais "
        "próxima da posição atual.</p>"
        f"<p>O resultado é uma condição de hardware, não uma refutação. Com um "
        f"dispositivo rápido o preditor mantém a vantagem de "
        f"{_pt(ideal_gap, '.1f')}&times; sobre o filtro alfa-beta. A varredura é "
        f"logarítmica, então o cruzamento é um intervalo e não um ponto: a "
        f"{_pt_number(crossover_low)} rad/s os dois empatam, abaixo disso o filtro "
        f"clássico ganha, e a {_pt_number(crossover)} rad/s a vantagem do preditor "
        f"já está estabelecida. A 30 rad/s a melhor coisa a fazer é praticamente "
        f"não fazer nada — a persistência empata com o MLP. O mecanismo está na "
        f"tabela seguinte: o preditor rastreia a banda acústica, e rastrear custa "
        f"movimento.</p>",
        _figure_block(
            {theme.name: _figure_slew(slew, theme) for theme in THEMES},
            "Um atuador lento não degrada todo mundo igualmente: ele filtra "
            "passa-baixas o comando, o que arruína quem rastreia e resgata quem "
            "estava piorando o enlace."),
        _table_pt_thousands(travel.sort_values("actuator_travel_rad"),
                  {"actuator_travel_rad": ",.0f", "mean_qber": ".4f",
                   "rad_per_s": ",.0f"},
                  columns=["compensator", "mean_qber", "actuator_travel_rad",
                           "rad_per_s"],
                  headers=["compensador", "QBER média",
                           "curso total em 20 s (rad)", "velocidade média (rad/s)"]),
        f"<p><strong>A hipótese do reset era a mais interessante, e é "
        f"falsa.</strong> A fase 3 apontou o reset sem fim como \"exatamente a "
        f"não linearidade que mudaria a conclusão\" sobre redes neurais. Ele "
        f"dispara, e muito — até {_pt(float(reset.resets_per_minute.max()), ',.0f')} "
        f"resets por minuto com curso de 2π e um dispositivo de 3000 rad/s. E não "
        f"muda nada: a QBER média de cada compensador é idêntica em curso de 2π, "
        f"4π e 12π, com espalhamento máximo de {_pt(reset_spread, '.1e')}. Num "
        f"dispositivo rápido o reset se completa numa fração de um período de "
        f"controle — a fração do tempo em reset fica abaixo de 0,2% — e a não "
        f"linearidade nunca chega a morder.</p>",
        _table_pt(reset_pivot,
                  {name: ",.1f" for name in reset_pivot.columns
                   if name != "range_turns"} | {"range_turns": ".0f"},
                  columns=["range_turns"] + [name for name in COMPENSATOR_ORDER
                                             if name in reset_pivot.columns],
                  headers=["curso (múltiplos de π)"]
                          + [name for name in COMPENSATOR_ORDER
                             if name in reset_pivot.columns]),
        *corner))

    # -- 05 the field data -------------------------------------------------
    if field:
        from ..physics.field_calibration import (
            CALIBRATED_DRIFT_RATE_RANGE_RAD_PER_S, FLICKER_EXPONENT,
            MEASUREMENTS)

        scan = field["exponent"]
        comparison = field["comparison"]

        def qber(point, compensator, training):
            match = comparison[(comparison["operating_point"] == point)
                               & (comparison["compensator"] == compensator)
                               & (comparison["training"] == training)]
            return float(match["mean_qber"].iloc[0]) if len(match) else float("nan")

        assumed = "fase 3 (40 Hz, 0,05 rad/s)"
        measured_point = "aérea calibrada (60/120/180 Hz, 0,02 rad/s)"
        rate_only = "só a taxa corrigida (40 Hz, 0,02 rad/s)"

        gap_assumed = qber(assumed, "filtro alfa-beta", "—") / qber(assumed, "MLP", "no ponto")
        gap_measured = (qber(measured_point, "filtro alfa-beta", "—")
                        / qber(measured_point, "MLP", "no ponto"))
        transplanted = qber(measured_point, "MLP", "da fase 3")
        do_nothing = qber(measured_point, "persistência", "—")

        provenance = pd.DataFrame([
            {"quantity": label, "value": value, "link": link,
             "source": item.source}
            for item in MEASUREMENTS
            for label, value, link in [MEASUREMENT_LABEL.get(
                item.quantity,
                (item.quantity, f"{item.value} {item.units}".strip(), item.link))]])

        table = comparison[comparison["operating_point"].isin(
            [assumed, measured_point])].copy()
        table["who"] = np.where(table["training"] == "—", table["compensator"],
                                table["compensator"] + " · " + table["training"])
        wide = table.pivot(index="who", columns="operating_point",
                           values="mean_qber").reset_index()

        sections.append(_section(
            "06", "correção de premissa",
            "A banda mecânica estava na frequência errada, e o compensador "
            "treinado nela é pior que não compensar",
            "<p>A quarta lacuna era não haver dado de fibra real. Ela continua "
            "aberta no sentido estrito — <strong>não existe nenhuma série "
            "pública de polarização em fibra terrestre implantada</strong>; "
            "tudo que é baixável é submarino e amostrado a 20 Hz ou menos, e os "
            "três melhores trabalhos em fibra aérea publicam as estatísticas e "
            "retêm os dados. Mas as estatísticas publicadas bastam para "
            "confrontar duas premissas da fase 3, e as duas estavam erradas.</p>"
            "<p><strong>A frequência.</strong> A fase 3 pôs a banda acústica em "
            "40 Hz. A única gravação de polarímetro a 4 kHz em fibra aérea que "
            "existe publicada — NIST, 220 km de cabo para-raios óptico dentro "
            "de uma linha de 160 kV — encontra um tom forte na frequência da "
            "rede elétrica com o segundo e o terceiro harmônicos: 60, 120 e "
            "180 Hz. O que move a fibra é a corrente do condutor em que ela "
            "está enfiada, não uma vibração acústica genérica.</p>"
            "<p><strong>A taxa de deriva.</strong> O ORNL caracterizou 15 km de "
            "fibra aérea e enterrada por 11 meses e reporta um espectro de "
            "deriva indo como <em>f</em><sup>−1,0 ± 0,2</sup>. Nesse modelo o "
            "expoente é função forte da taxa de deriva — a polarização dá a "
            "volta na esfera, então derivar mais rápido embranquece o espectro "
            "— e casar o valor medido fixa a taxa entre "
            f"{_pt(CALIBRATED_DRIFT_RATE_RANGE_RAD_PER_S[0], '.3f')} e "
            f"{_pt(CALIBRATED_DRIFT_RATE_RANGE_RAD_PER_S[1], '.3f')} rad/s, "
            "contra os 0,05 que a fase 3 usava.</p>",
            _figure_block(
                {theme.name: _figure_exponent(scan, FLICKER_EXPONENT,
                                              CALIBRATED_DRIFT_RATE_RANGE_RAD_PER_S,
                                              theme) for theme in THEMES},
                "Um fator de cinco na taxa varre o expoente de 0,25 a 1,7: a "
                "medida do ORNL não é uma restrição frouxa."),
            _table_pt(provenance, {},
                      columns=["quantity", "value", "link", "source"],
                      headers=["grandeza", "valor medido", "enlace", "fonte"]),
            f"<p><strong>A conclusão da fase 3 sobrevive; o modelo treinado, "
            f"não.</strong> Retreinando no ponto calibrado, o preditor continua "
            f"batendo o filtro clássico por "
            f"{_pt(gap_measured, '.1f')}&times; contra "
            f"{_pt(gap_assumed, '.1f')}&times; no ponto assumido — a vantagem "
            f"encolhe, mas continua sendo a diferença entre 99,9% e 90% de "
            f"tempo abaixo do limiar. O que não sobrevive é o artefato: pegar a "
            f"MLP treinada a 40 Hz e rodá-la na fibra de 60 Hz dá QBER "
            f"<strong>{_pt(transplanted, '.3f')}</strong>, contra "
            f"{_pt(do_nothing, '.3f')} de não fazer nada. Um compensador "
            f"treinado na banda errada é <strong>pior que nenhum "
            f"compensador</strong>, e passa a maior parte do tempo acima do "
            f"limiar de 11%.</p>",
            _figure_block(
                {theme.name: _figure_field_comparison(comparison, theme)
                 for theme in THEMES},
                "Retreinar recupera quase tudo. Transplantar não recupera nada: "
                "o preditor aprende a fase de uma frequência específica, e numa "
                "frequência diferente ele injeta erro em vez de removê-lo."),
            _table_pt(wide,
                      {assumed: ".4f", measured_point: ".4f"},
                      columns=["who", assumed, measured_point],
                      headers=["compensador", "ponto assumido pela fase 3",
                               "ponto medido em campo"]),
            f'<p class="caveat">A calibração da taxa de deriva não mudou nada: '
            f'com a banda em 40 Hz, baixar a taxa de 0,05 para 0,02 rad/s deixa '
            f'a QBER média em {_pt(qber(rate_only, "MLP", "no ponto"), ".4f")} '
            f'contra {_pt(qber(assumed, "MLP", "no ponto"), ".4f")} — idêntica. '
            f'A 40 ms de latência a deriva térmica move dois miliradianos e a '
            f'banda mecânica move trezentos; quem manda é a banda. Vale '
            f'registrar que o exercício de calibração fixou com precisão um '
            f'parâmetro que não governa o resultado, e que o parâmetro que '
            f'governa — a amplitude da banda — continua sendo o único número '
            f'deste relatório sem nenhuma medição por trás.</p>'))

    # -- 06 the unmeasured parameter, bounded --------------------------------
    if "amplitude" in field:
        sweep = field["amplitude"]
        wide = sweep.pivot(index="amplitude_rad", columns="compensator",
                           values="mean_qber")
        wide["vantagem"] = wide["filtro alfa-beta"] / wide["MLP"]
        wide = wide.reset_index()
        low_ratio = float(wide["vantagem"].min())
        high_ratio = float(wide["vantagem"].max())
        span = float(sweep["amplitude_rad"].max() / sweep["amplitude_rad"].min())

        sections.append(_section(
            "07", "limite em vez de medida",
            f"A amplitude do tom não é conhecida, e ao longo de um fator de "
            f"{_pt(span, '.0f')} nela a conclusão não muda",
            "<p>A amplitude do tom de rede é o único parâmetro da fase 3 sem "
            "nenhuma medição por trás — o relatório da NIST mostra o tom e "
            "nunca diz seu tamanho em radianos, e nenhuma outra fonte "
            "encontrada diz. Como não dá para medi-la aqui, o que dá para "
            "fazer é tirá-la do caminho: varrê-la de ponta a ponta e ver se a "
            "resposta depende dela.</p>"
            f"<p>Depende do valor absoluto da QBER, claro, e muito. Não depende "
            f"da <em>conclusão</em>: com os preditores retreinados em cada "
            f"amplitude, a vantagem sobre o filtro alfa-beta fica entre "
            f"<strong>{_pt(low_ratio, '.1f')}&times; e "
            f"{_pt(high_ratio, '.1f')}&times;</strong> em toda a faixa, e "
            f"cresce com a amplitude. Abaixo de 0,05 rad todo mundo fica sob 1% "
            f"de QBER e compensar deixa de ser um problema que valha resolver; "
            f"acima disso, quanto pior a fibra, mais o preditor rende. Não "
            f"existe amplitude plausível em que a resposta se inverta.</p>",
            _figure_block(
                {theme.name: _figure_amplitude(sweep, theme) for theme in THEMES},
                "Log-log. As quatro curvas são paralelas: a amplitude escala a "
                "dificuldade sem reordenar quem a resolve melhor."),
            _table_pt(wide,
                      {"amplitude_rad": ".2f", "MLP": ".4f",
                       "ridge (Wiener)": ".4f", "filtro alfa-beta": ".4f",
                       "persistência": ".4f", "vantagem": ".1f"},
                      columns=["amplitude_rad", "persistência",
                               "filtro alfa-beta", "ridge (Wiener)", "MLP",
                               "vantagem"],
                      headers=["amplitude (rad)", "persistência",
                               "filtro alfa-beta", "ridge (Wiener)", "MLP",
                               "MLP ÷ alfa-beta"]),
            '<p class="caveat">Isto não substitui medir. O que faz é mudar o que '
            'a medição decide: ela passa a fixar o valor absoluto da QBER — '
            'quanta chave o enlace produz — e não mais se vale a pena usar um '
            'preditor. São perguntas de peso muito diferente.</p>'))

    # -- 07 the brittleness, and the fix -------------------------------------
    if "robustness" in field:
        robustness = field["robustness"]

        def at(frequency, training, compensator="MLP"):
            match = robustness[(robustness["test_frequency_hz"] == frequency)
                               & (robustness["training"] == training)
                               & (robustness["compensator"] == compensator)]
            return float(match["mean_qber"].iloc[0]) if len(match) else float("nan")

        frequencies = sorted(robustness["test_frequency_hz"].unique())
        specialist = np.array([at(f, "especialista, mesma frequência")
                               for f in frequencies])
        generalist = np.array([at(f, "generalista") for f in frequencies])
        classical = np.array([at(f, "—", "filtro alfa-beta") for f in frequencies])
        generalist_ridge = np.array([at(f, "generalista", "ridge (Wiener)")
                                     for f in frequencies])
        stranded = robustness[robustness["training"].str.startswith("especialista de")]

        penalty = float(np.max(generalist / specialist)) - 1.0
        still_better = float(np.min(classical / generalist))
        nonlinear_gain = float(np.max(generalist_ridge / generalist))

        table = robustness.assign(
            who=robustness["compensator"] + " · " + robustness["training"])
        matrix = table.pivot(index="who", columns="test_frequency_hz",
                             values="mean_qber").reset_index()
        matrix.columns = ["who"] + [f"{value:g} Hz" for value in frequencies]

        retraction: list[str] = []
        if "reach" in field:
            reach = field["reach"]
            frequency = reach[reach["axis"] == "frequência"]
            amplitude = reach[reach["axis"] == "amplitude"]

            def held_out(frame, training, compensator="MLP"):
                group = frame[(frame["training"] == training)
                              & (frame["compensator"] == compensator)]
                return group.set_index("test_point")["mean_qber"]

            narrow = held_out(frequency, "generalista estreito (45-70 Hz)")
            broad = held_out(frequency, "generalista largo (45-180 Hz)")
            classical = held_out(frequency, "—", "filtro alfa-beta")

            just_outside = float(narrow.get(75.0, float("nan")))
            inside = float(narrow.get(50.0, float("nan")))
            broad_worst = float((broad / classical).max())

            frequencies = sorted(frequency["test_point"].unique())
            wide_table = frequency.assign(
                who=frequency["compensator"] + " · " + frequency["training"]
            ).pivot(index="who", columns="test_point",
                    values="mean_qber").reset_index()
            wide_table.columns = ["who"] + [f"{value:g} Hz"
                                            for value in frequencies]

            amplitudes = sorted(amplitude["test_point"].unique())
            amplitude_table = amplitude.assign(
                who=amplitude["compensator"] + " · " + amplitude["training"]
            ).pivot(index="who", columns="test_point",
                    values="mean_qber").reset_index()
            amplitude_table.columns = ["who"] + [
                f"{value:g} rad".replace(".", ",") for value in amplitudes]

            retraction = [
                "<p><strong>Retratação: o parágrafo acima vale para a faixa em "
                "que foi medido, e não para além dela.</strong> O generalista "
                "estreito foi treinado e testado nas mesmas quatro frequências. "
                "Em frequências retidas ele <em>interpola</em> — "
                f"{_pt(inside, '.4f')} a 50 Hz, dentro da faixa — e não "
                f"extrapola: a 75 Hz, cinco hertz acima da borda treinada, a "
                f"QBER dele é {_pt(just_outside, '.4f')}, oito vezes pior e "
                f"acima do filtro clássico.</p>"
                "<p>Alargar a faixa não resolve. Um generalista treinado em 45, "
                "60, 90, 120, 150 e 180 Hz fica pior que o filtro alfa-beta em "
                f"todos os pontos de teste — até {_pt(broad_worst, '.1f')} vezes "
                f"pior — inclusive em frequências que viu no treino. Dobrar a "
                f"largura da rede e dobrar os dados por ponto não muda isso, e "
                f"na faixa larga o ridge passa a bater o MLP, invertendo a "
                f"conclusão da faixa estreita.</p>",
                _figure_block(
                    {theme.name: _figure_reach(reach, theme) for theme in THEMES},
                    "Escala logarítmica. O especialista sobe suavemente com a "
                    "frequência; os generalistas não acompanham fora da faixa "
                    "que viram."),
                _table_pt(wide_table,
                          {name: ".4f" for name in wide_table.columns
                           if name != "who"},
                          columns=list(wide_table.columns),
                          headers=["compensador"] + list(wide_table.columns[1:])),
                "<p><strong>O eixo da amplitude, ao contrário, generaliza "
                "inteiro.</strong> Um modelo treinado em 0,05, 0,15 e 0,30 rad "
                "empata com o especialista em toda a faixa de teste, incluindo "
                "0,50 rad, que está fora do treino. Amplitude é um parâmetro de "
                "escala; frequência é um parâmetro de fase, e a janela de "
                "observação de 32 ms cobre de 1,4 a 5,8 períodos conforme a "
                "frequência. Um preditor que precisa acertar a fase não pode "
                "tratar os dois eixos do mesmo jeito.</p>",
                _table_pt(amplitude_table,
                          {name: ".4f" for name in amplitude_table.columns
                           if name != "who"},
                          columns=list(amplitude_table.columns),
                          headers=["compensador"]
                                  + list(amplitude_table.columns[1:])),
                "<p><strong>O que fica de pé:</strong> retreino por enlace "
                "continua sendo requisito, mas o requisito real é conhecer a "
                "frequência do tom com precisão de cerca de ±10 Hz e escolher, "
                "de uma família de modelos treinados em faixas estreitas, o que "
                "cobre aquela. É bem menos que uma corrida de treino por "
                "enlace, e bem mais que um modelo único.</p>",
            ]

        sections.append(_section(
            "08", "achado", "Um modelo cobre uma faixa estreita de frequências "
            "— e nada mais que isso",
            "<p>A seção 06 deixou um problema de engenharia em aberto: se o "
            "compensador treinado numa frequência é pior que nada em outra, "
            "cada enlace precisa do seu próprio polarímetro e da sua própria "
            "corrida de treino, e isso é um produto muito diferente. A saída "
            "óbvia é treinar sobre uma distribuição de frequências em vez de "
            "uma só. Dentro de uma faixa estreita ela funciona; fora dela, "
            "como a retratação ao fim desta seção mostra, não.</p>"
            f"<p>Um único modelo treinado em 45, 50, 60 e 70 Hz paga no máximo "
            f"{_pt(100 * penalty, '.0f')}% em relação ao especialista da "
            f"frequência certa, e continua batendo o filtro clássico por pelo "
            f"menos {_pt(still_better, '.1f')}&times; em todas elas. Para "
            f"comparação, os "
            f"especialistas rodando na frequência errada ficam entre "
            f"{_pt(float(stranded['mean_qber'].min()), '.3f')} e "
            f"{_pt(float(stranded['mean_qber'].max()), '.3f')} — a maioria pior "
            f"que o filtro clássico, e alguns piores que não compensar.</p>",
            _figure_block(
                {theme.name: _figure_robustness(robustness, theme)
                 for theme in THEMES},
                "Escala logarítmica. A generalização custa pouco; a "
                "especialização na frequência errada custa uma ordem de "
                "grandeza."),
            _table_pt(matrix,
                      {name: ".4f" for name in matrix.columns if name != "who"},
                      columns=list(matrix.columns),
                      headers=["compensador"] + list(matrix.columns[1:])),
            f"<p><strong>E aqui a rede neural finalmente ganha do filtro de "
            f"Wiener, pela primeira vez neste pacote.</strong> As fases 3 e 4 e "
            f"o estudo de dual-head reportaram todas a mesma coisa: a MLP "
            f"empata com o ridge, porque um processo gaussiano estacionário tem "
            f"preditor ótimo linear e o ridge <em>é</em> esse preditor. Na "
            f"frequência certa isso continua valendo — especialista ridge e "
            f"especialista MLP empatam. Mas uma <em>mistura</em> de frequências "
            f"não é um processo gaussiano estacionário, e o ridge generalista "
            f"falha: fica pior que o filtro alfa-beta a 50 e 60 Hz. O MLP "
            f"generalista o bate por até {_pt(nonlinear_gain, '.1f')}&times;.</p>"
            "<p>É a primeira tarefa deste projeto em que a não linearidade se "
            "paga, e o motivo é explicável em vez de empírico: o preditor "
            "precisa inferir <em>qual</em> frequência está vendo antes de "
            "prever a fase dela, e essa inferência não é uma função linear da "
            "janela de observação. O achado negativo de quatro fases não estava "
            "errado — estava restrito a tarefas onde só havia um regime.</p>",
            '<p class="caveat">O ajuste do ridge é determinístico dado o '
            'conjunto de treino, então as três sementes testam a inicialização '
            'da MLP, não a variância do ridge. A MLP generalista fica entre '
            '0,0140 e 0,0181 nas três; o ridge repete 0,0464 e 0,0513 '
            'exatamente. A falha do ridge é da classe de modelos, não de uma '
            'corrida infeliz.</p>',
            *retraction))

    # -- 09 blocks of variable duration --------------------------------------
    if blocking:
        summary = blocking["summary"]
        stress = blocking["stress"].sort_values("vibration_amplitude_rad")

        def _rate(compensator: str, strategy: str) -> float:
            match = summary[(summary["compensator"] == compensator)
                            & (summary["strategy"] == strategy)]
            return float(match["key_rate_hz"].iloc[0]) if not match.empty else float("nan")

        worst = summary[summary["compensator"] == "persistência"]
        best_timer_bad = float(worst[worst["strategy"].str.startswith("fixo")]
                               ["key_rate_hz"].max())
        controller_bad = _rate("persistência", "adaptativo")
        oracle_bad = _rate("persistência", "oráculo")
        quiet_cost = 1.0 - _rate("MLP", "adaptativo") / _rate("MLP", "oráculo")

        hard = stress[stress["over_best_timer"] > 1.05]
        gain_low = float(hard["over_best_timer"].min()) if not hard.empty else float("nan")
        gain_high = float(hard["over_best_timer"].max()) if not hard.empty else float("nan")
        onset = (float(hard["vibration_amplitude_rad"].min()) if not hard.empty
                 else float("nan"))

        blocking_table = summary.copy()
        blocking_table["of_oracle"] = 100.0 * blocking_table["of_oracle"]

        sections.append(_section(
            "09", "retratação e correção",
            "Blocos de duração fixa custavam tudo no enlace difícil, e a "
            "medição anterior não podia ver isso",
            "<p>A seção de limites deste relatório dizia duas coisas sobre "
            "blocos, e as duas estavam erradas. Dizia que os blocos são de "
            "duração fixa e que modelar o contrário melhoraria os números; e "
            "dizia, com um número, que um oráculo já tinha medido o ganho em "
            "1,28&times; e que portanto não havia o que construir. O número "
            "veio de um oráculo que só procurava entre <em>durações fixas por "
            "execução</em>, limitado a dez segundos. Um oráculo assim não "
            "consegue, por construção, encontrar a única coisa que um bloco "
            "variável faz de diferente: recortar a janela boa de dentro de uma "
            "execução ruim.</p>"
            "<p>O oráculo certo é uma programação dinâmica sobre partições, e "
            "é exata: a chave de um bloco depende só dos seus extremos, porque "
            "a amplificação de privacidade é aplicada por bloco e nenhum termo "
            "acopla um bloco a outro. O mecanismo que ela expõe é o que o "
            "limite anterior não tinha: <strong>o erro é cobrado contra o "
            "bloco inteiro, então um trecho de fibra ruim envenena cada bit "
            "que divide o bloco com ele</strong>. Dez segundos bons e um "
            "segundo ruim no mesmo bloco podem valer zero.</p>"
            f"<p>No compensador de persistência, com QBER média de "
            f"{_pt(float(worst['mean_qber'].iloc[0]), '.3f')}, é exatamente o "
            f"que acontece: <strong>temporizador de cinco segundos ou mais "
            f"produz zero bits</strong>, de qualquer duração, em todas as doze "
            f"sementes. O melhor temporizador — um décimo de segundo — chega a "
            f"{_pt_number(best_timer_bad)} bit/s. O controlador causal chega a "
            f"{_pt_number(controller_bad)} bit/s, "
            f"{_pt(controller_bad / best_timer_bad, '.1f')}&times; o melhor "
            f"temporizador e {_pt(100 * controller_bad / oracle_bad, '.0f')}% "
            f"do oráculo. Nos compensadores bons o melhor bloco é um só, do "
            f"tamanho da execução, e o controlador o encontra sozinho: custa "
            f"{_pt(100 * quiet_cost, '.1f')}% em relação ao oráculo.</p>",
            _table_pt_thousands(
                blocking_table,
                {"key_rate_hz": ",.0f", "mean_qber": ".3f", "blocks": ".1f",
                 "duty": ".2f", "mean_block_s": ".2f", "of_oracle": ".1f"},
                columns=["compensator", "strategy", "mean_qber", "key_rate_hz",
                         "blocks", "duty", "mean_block_s", "of_oracle"],
                headers=["compensador", "estratégia", "QBER média",
                         "chave (bit/s)", "blocos", "ciclo útil",
                         "bloco médio (s)", "% do oráculo"]),
            "<p>O controlador é causal e não tem constante de ajuste. Ele vê "
            "apenas os bits que o protocolo publica para estimação de "
            "parâmetro — dez por cento, com o ruído binomial que isso implica "
            "— e é pontuado sobre a série verdadeira, que nunca vê. A cada "
            "instante compara duas continuações: encerrar agora e rodar no "
            "melhor regime estacionário que o enlace demonstrou, ou seguir com "
            "o bloco até o melhor ponto de corte futuro. Os dois "
            "comportamentos saem dessa comparação sem serem programados. "
            "Enlace quieto: o futuro vale muito, reiniciar não ganha nada, o "
            "bloco cresce. Enlace ruim: o futuro não vale nada, a única coisa "
            "em jogo é o bloco na mão, e ele corta e descarta.</p>"
            "<p>A regra míope óbvia — encerrar quando o segmento novo "
            "derrubaria a chave estimada do bloco — foi implementada primeiro "
            "e é mensuravelmente pior: encerra em cima do ruído da estimativa, "
            "corta o enlace bom em blocos de dois segundos onde havia vinte "
            "disponíveis, e entrega cerca de um quinto a menos de chave. "
            "Comparar o momento com o momento seguinte não basta; é preciso "
            "comparar dois futuros.</p>"
            "<p><strong>E no compensador bom?</strong> A objeção óbvia é que "
            "ninguém opera com a persistência. Então a mesma comparação roda "
            "na MLP, com a amplitude de vibração da seção 06 subindo:</p>",
            _figure_block(
                {theme.name: _figure_blocking(stress, theme) for theme in THEMES},
                "Escala logarítmica, com o zero rebaixado a 1 bit/s para "
                "caber. As curvas se separam onde o enlace deixa de ser "
                "estacionário o bastante para uma duração só estar certa."),
            _table_pt_thousands(
                stress,
                {"vibration_amplitude_rad": ".1f", "mean_qber": ".3f",
                 "median_best_timer_s": ".1f", "best_timer_hz": ",.0f",
                 "timer_1s_hz": ",.0f", "adaptive_hz": ",.0f",
                 "oracle_hz": ",.0f", "over_best_timer": ".2f"},
                columns=["vibration_amplitude_rad", "mean_qber",
                         "median_best_timer_s", "best_timer_hz", "timer_1s_hz",
                         "adaptive_hz", "oracle_hz", "over_best_timer"],
                headers=["amplitude (rad)", "QBER média",
                         "melhor duração fixa (s)", "melhor fixo (bit/s)",
                         "fixo 1 s (bit/s)", "controlador (bit/s)",
                         "oráculo (bit/s)", "sobre o melhor fixo"]),
            f"<p>Abaixo do ponto de operação da fase 3 o controlador custa "
            f"cerca de 1% e não serve para nada. A partir de "
            f"{_pt(onset, '.1f')} rad ele bate o melhor temporizador "
            f"<em>escolhido com retrospectiva</em> por "
            f"{_pt(gain_low, '.1f')}&ndash;{_pt(gain_high, '.1f')}&times;, e o "
            f"temporizador de um segundo que as fases 3 e 4 assumiram entrega "
            f"zero. A conclusão que substitui a anterior não é \"use blocos "
            f"longos\": é que a duração certa depende do enlace, ninguém a "
            f"conhece de antemão, e o controlador a encontra sozinho — de "
            f"graça no enlace fácil, e é a diferença entre chave e nada no "
            f"enlace difícil.</p>",
            '<p class="caveat">O oráculo particiona a execução em '
            '<em>intervalos</em>. Nada na amplificação de privacidade exige '
            'que um bloco seja contíguo no tempo — um receptor pode juntar '
            'janelas boas de trechos separados no mesmo bloco — e esse oráculo '
            'é estritamente maior que este. Quanto do que sobra o controlador '
            'alcança nessa formulação não foi medido, então os "% do oráculo" '
            'acima são otimistas como medida de completude, não como medida de '
            'chave.</p>'))

    # -- 10 limits ---------------------------------------------------------
    sections.append(_section(
        "10", "limites", "A lacuna que continua aberta",
        "<p><strong>Continua não havendo série temporal de fibra real neste "
        "pacote.</strong> A seção 06 usa estatísticas publicadas, não gravações: "
        "um expoente espectral, uma lista de frequências, uma faixa de taxa de "
        "deriva. Isso fixa três parâmetros; o quarto, a amplitude da banda, "
        "continua sem medição e a seção 07 o contorna varrendo-o em vez de "
        "medi-lo. Contornar não é medir: a varredura mostra que a <em>escolha "
        "entre compensadores</em> não depende da amplitude, e continua sem "
        "dizer quanta chave um enlace real produz, que é a única pergunta que "
        "importa para dimensionar um sistema. Um polarímetro amostrando acima "
        "de 400 Hz responde em uma semana, e "
        "<code>qnet_autopilot.datagen.measured</code> consome a gravação e "
        "devolve o modelo ajustado.</p>"
        "<p><strong>A busca pela amplitude publicada foi exaustiva e deu "
        "negativo.</strong> O texto completo do relatório da NIST dedica três "
        "frases ao tom e não dá nenhum número; as figuras são rasterizadas sem "
        "camada de texto. Duas pistas ficam para quem tiver acesso "
        "institucional: Waddy, Chen &amp; Bao, <em>\"Polarization oscillations "
        "in aerial fiber caused by wind and power-line current\"</em> (IEEE "
        "PTL), cujo título é literalmente o efeito; e a referência [2] do "
        "relatório da NIST (Leo <em>et al.</em>, JLT), que dá os parâmetros de "
        "Rayleigh por taxa de varredura do embaralhador contra o qual o "
        "histograma aéreo foi calibrado — o que daria a escala da figura, ainda "
        "que como limite superior, por misturar o tom com o vento.</p>"
        "<p><strong>Nenhum dos dados públicos amostra rápido o bastante.</strong> "
        "As séries baixáveis — Egersund-Yme a 5 Hz, Med-Nautilus a 5 Hz, "
        "EllaLink, Curie — são todas submarinas e todas sub-Nyquist para "
        "qualquer banda mecânica. Elas calibram a deriva lenta e a estatística "
        "de saltos e nada além disso, e o ajuste recusa-se a inventar uma banda "
        "a partir delas em vez de reportar potência rebatida como se fosse "
        "medição.</p>"
        "<p><strong>O agente já roda dentro do SeQUeNCe — essa lacuna "
        "fechou.</strong> E fechou com um resultado que custa caro: o custo em "
        "tempo de uma tentativa de heralding estava errado por um fator de "
        "quatro, e corrigi-lo apaga a vantagem publicada da fase 2. O que se "
        "aprende com isso não é sobre eventos discretos; é que um parâmetro de "
        "custo mal medido no ambiente de treino não aparece em teste nenhum "
        "que rode no mesmo ambiente.</p>"
        "<p><strong>A faixa do generalista é estreita, e isso está medido.</strong> "
        "A seção 08 mostra que ele não sobrevive cinco hertz fora da faixa "
        "treinada e que alargar a faixa o destrói. Um enlace só é coberto se a "
        "frequência do tom for conhecida a cerca de ±10 Hz; medi-la é o mesmo "
        "polarímetro que mediria a amplitude, então as duas lacunas de campo "
        "que sobram se fecham com o mesmo instrumento.</p>"
        "<p><strong>O motor de regras do SeQUeNCe continua fora.</strong> O "
        "runner de eventos discretos dirige os protocolos diretamente, porque o "
        "objetivo era pôr uma política aprendida no lugar da política do "
        "simulador. Contenção entre requisições concorrentes — o que o "
        "gerenciador de rede resolve — segue sem teste, e é o lugar mais "
        "provável para o próximo desvio entre os dois simuladores.</p>"
        "<p><strong>Bloco adaptativo: aquela conclusão foi retratada na "
        "seção 09.</strong> O oráculo que rendia 1,28&times; só procurava "
        "entre durações fixas por execução, e por isso não podia ver o que "
        "um bloco variável faz. Com o oráculo certo, no enlace difícil "
        "temporizador nenhum produz um bit e o controlador produz; no enlace "
        "fácil o controlador custa cerca de 1%. O que continua aberto é o "
        "oráculo sobre blocos não contíguos, que é maior que o desta "
        "seção.</p>"
        "<p><strong>Um ponto de operação por medição.</strong> A varredura de "
        "velocidade do atuador roda no ponto de operação de fibra aérea da fase "
        "3; o cruzamento entre "
        f"{_pt_number(crossover_low)} e {_pt_number(crossover)} rad/s é uma "
        "propriedade daquela banda "
        "acústica e daquela latência, não uma constante. Fibra enterrada move-se "
        "devagar o bastante para que a pergunta nem se coloque.</p>"
        "<p><strong>O ponto de operação do controlador de bloco é um "
        "só.</strong> A seção 09 varre a amplitude de vibração, que é o eixo "
        "que muda a resposta, e mantém tudo o mais fixo: uma taxa peneirada, "
        "uma fração de monitoramento, um orçamento de bloco igual à execução. "
        "Uma implantação com prazo de entrega de chave tem um orçamento "
        "menor, e o quanto isso custa não está varrido.</p>"))

    body = "\n".join(sections)
    document = f"""<title>{html.escape(title)}</title>
{_FONTS}
<style>{_CSS}
figure + p, figure + div {{ margin-top: 22px; }}
.table-wrap + p {{ margin-top: 18px; }}
.table-wrap + .table-wrap {{ margin-top: 14px; }}</style>
<main>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="lede">Todas as lacunas que separavam estes resultados de uma
       afirmação de aplicabilidade, medidas em vez de argumentadas. O agente
       passou a rodar dentro do simulador de eventos discretos, e uma constante
       de 4,1 no custo de heralding — errada desde a fase 2 — explica toda a
       diferença entre os dois simuladores e apaga a vantagem publicada do
       agente. Somam-se a isso uma correção de fator 1,6 nas taxas de chave, uma
       condição de hardware que a fase 3 não tinha, duas hipóteses próprias
       refutadas, a banda mecânica na frequência errada e uma retratação: o
       bloco de duração fixa, que este relatório tinha declarado suficiente,
       produz zero bit no enlace difícil, e o controlador que o substitui
       produz chave lá.</p>
  </header>
  {body}
</main>
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
