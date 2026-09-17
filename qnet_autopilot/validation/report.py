"""Render the validation checks as a self-contained HTML report.

The report is generated from :mod:`qnet_autopilot.validation.checks` output, so
every number and every figure in it is a measurement made by the run that
produced it.  Nothing is transcribed by hand, which means the report cannot
drift out of date relative to the code without the check itself failing.

Figures are inlined as SVG, twice — once stepped for a light surface and once
for a dark one — and swapped by CSS, so the report reads correctly in either
theme without a rendering pass at view time.  The palette is the validated
categorical default: blue, orange, aqua, yellow, in that fixed order.
"""

from __future__ import annotations

import html
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .checks import CheckResult

__all__ = ["render_report", "THEMES", "Theme"]


@dataclass(frozen=True)
class Theme:
    """Colours for one rendering surface."""

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    series: tuple[str, ...]


#: Light and dark instances of the validated categorical palette.
THEMES = (
    Theme(name="light", surface="#fcfcfb", text_primary="#0b0b0b",
          text_secondary="#52514e", grid="#dedcd5",
          series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100")),
    Theme(name="dark", surface="#1a1a19", text_primary="#ffffff",
          text_secondary="#c3c2b7", grid="#3a3a37",
          series=("#3987e5", "#d95926", "#199e70", "#c98500")),
)


def _figure(theme: Theme, width: float = 7.4, height: float = 4.0):
    """A matplotlib figure styled for ``theme``, with recessive axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(width, height), dpi=110)
    figure.patch.set_alpha(0.0)
    axes.set_facecolor("none")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme.grid)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=theme.text_secondary, labelsize=9, length=3)
    axes.grid(True, color=theme.grid, linewidth=0.7, alpha=0.7)
    axes.set_axisbelow(True)
    return figure, axes


def _to_svg(figure) -> str:
    """Serialise a figure to an inline SVG fragment."""
    import matplotlib.pyplot as plt

    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", transparent=True)
    plt.close(figure)
    svg = buffer.getvalue()
    # Drop the XML prologue and DOCTYPE so the fragment can be inlined.
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else svg


def _style_text(axes, theme: Theme, title: str, xlabel: str, ylabel: str) -> None:
    axes.set_title(title, color=theme.text_primary, fontsize=11, loc="left", pad=12)
    axes.set_xlabel(xlabel, color=theme.text_secondary, fontsize=9)
    axes.set_ylabel(ylabel, color=theme.text_secondary, fontsize=9)


def _legend(axes, theme: Theme, loc: str = "best") -> None:
    legend = axes.legend(frameon=False, fontsize=9, loc=loc)
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _figure_link_rates(detail: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme)
    lengths = detail["length_km"].to_numpy()
    measured = detail["measured"].to_numpy()
    lower = measured - detail["ci_low"].to_numpy()
    upper = detail["ci_high"].to_numpy() - measured

    axes.plot(lengths, detail["single_heralded"], color=theme.series[0], linewidth=2,
              label="modelo: single-heralded  $\\eta^2/2$")
    axes.plot(lengths, detail["barrett_kok"], color=theme.series[1], linewidth=2,
              linestyle="--", label="modelo: Barrett-Kok  $(\\eta^2/2)^2$")
    axes.errorbar(lengths, measured, yerr=[lower, upper], fmt="o", markersize=7,
                  color=theme.series[2], ecolor=theme.series[2], elinewidth=1.6,
                  capsize=4, markeredgecolor=theme.surface, markeredgewidth=1.5,
                  label="SeQUeNCe, intervalo de Wilson 95%", zorder=5)

    axes.set_yscale("log")
    _style_text(axes, theme,
                "Probabilidade de heralding por tentativa",
                "comprimento do vão (km)", "probabilidade de sucesso")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


def _figure_purification(detail: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme)
    axes.axhline(0.0, color=theme.text_secondary, linewidth=1.2, alpha=0.6)

    for index, (gate_fidelity, group) in enumerate(detail.groupby("gate_fidelity",
                                                                  sort=False)):
        axes.plot(group["input_fidelity"], group["gain"],
                  color=theme.series[index % len(theme.series)], linewidth=2,
                  label=f"portas e medições a {gate_fidelity:.2f}")

    _style_text(axes, theme,
                "BBPSSW: fidelidade ganha (ou perdida) ao purificar dois pares iguais",
                "fidelidade de entrada", "fidelidade de saída − de entrada")
    axes.set_xlim(0.5, 1.0)
    _legend(axes, theme, loc="lower right")
    return _to_svg(figure)


def _figure_swap_chain(detail: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme)
    for index, (gate_fidelity, group) in enumerate(detail.groupby("gate_fidelity",
                                                                  sort=False)):
        axes.plot(group["num_swaps"], group["end_to_end_fidelity"], marker="o",
                  markersize=6, linewidth=2,
                  color=theme.series[index % len(theme.series)],
                  markeredgecolor=theme.surface, markeredgewidth=1.5,
                  label=f"portas e medições a {gate_fidelity:.2f}")

    axes.axhline(0.5, color=theme.text_secondary, linewidth=1.0, linestyle=":", alpha=0.8)
    axes.annotate("F = 0,5: não há mais emaranhamento a destilar",
                  xy=(0.02, 0.505), xycoords=("axes fraction", "data"),
                  color=theme.text_secondary, fontsize=8.5, va="bottom")

    _style_text(axes, theme,
                "Fidelidade ponta a ponta com swapping aninhado (enlaces a 0,95)",
                "número de swaps", "fidelidade ponta a ponta")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


def _figure_drift(detail: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme)
    axes.plot(detail["time_s"] / 60.0, detail["predicted_diffusive_rad"],
              color=theme.series[1], linewidth=2, linestyle="--",
              label="previsão  $\\sqrt{2V(t)}$")
    axes.plot(detail["time_s"] / 60.0, detail["rms_angle_rad"],
              color=theme.series[0], linewidth=2, label="deslocamento RMS simulado")

    window = detail["in_comparison_window"].to_numpy(dtype=bool)
    if window.any():
        times = detail["time_s"].to_numpy() / 60.0
        axes.axvspan(times[window].min(), times[window].max(),
                     color=theme.series[2], alpha=0.10)
        axes.annotate("janela de comparação", xy=(times[window].mean(), 0.02),
                      xycoords=("data", "axes fraction"), ha="center",
                      color=theme.text_secondary, fontsize=8.5)

    _style_text(axes, theme,
                "A deriva térmica de polarização acumula como previsto",
                "tempo (minutos)", "deslocamento RMS na esfera de Poincaré (rad)")
    _legend(axes, theme, loc="upper left")
    return _to_svg(figure)


def _figure_trajectory(theme: Theme, seed: int = 11) -> str:
    """One realisation, so the reader can see what the drift actually looks like."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..physics.constants import BB84_QBER_THRESHOLD
    from ..physics.fiber import PolarizationDriftModel

    model = PolarizationDriftModel(jump_rate_per_s=1 / 900.0)
    trajectory = model.simulate(3600.0, 0.5, rng=seed)

    figure, axes = plt.subplots(figsize=(7.4, 3.4), dpi=110)
    figure.patch.set_alpha(0.0)
    axes.set_facecolor("none")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme.grid)
    axes.tick_params(colors=theme.text_secondary, labelsize=9, length=3)
    axes.grid(True, color=theme.grid, linewidth=0.7, alpha=0.7)
    axes.set_axisbelow(True)

    minutes = trajectory.time_s / 60.0
    axes.plot(minutes, trajectory.qber, color=theme.series[0], linewidth=1.4,
              label="QBER por desalinhamento")
    axes.axhline(BB84_QBER_THRESHOLD, color=theme.series[3], linewidth=1.6,
                 linestyle="--", label=f"limiar do BB84 ({BB84_QBER_THRESHOLD:.0%})")

    for index, jump in enumerate(trajectory.jump_times_s):
        axes.axvline(jump / 60.0, color=theme.series[1], linewidth=1.2, alpha=0.7,
                     label="salto de polarização" if index == 0 else None)

    axes.set_title("Uma hora sem compensação: deriva, vibração e saltos de cabo",
                   color=theme.text_primary, fontsize=11, loc="left", pad=12)
    axes.set_xlabel("tempo (minutos)", color=theme.text_secondary, fontsize=9)
    axes.set_ylabel("QBER", color=theme.text_secondary, fontsize=9)
    legend = axes.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)

    return _to_svg(figure)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
          '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
          '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
          'family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&'
          'family=IBM+Plex+Sans:wght@400;450;600&'
          'family=IBM+Plex+Mono:wght@400;500&display=swap">')

# Neutrals carry a faint cool cast — the page is about light in glass, and a
# pure grey next to the blue series colour reads as unconsidered beside it.
_CSS = """
:root {
  --surface: #f7f8f7; --panel: #ffffff; --border: #dfe3e2; --rule: #c9d0cf;
  --text: #101413; --muted: #565f5e; --accent: #2a78d6;
  --pass-bg: #e4f2ea; --pass-fg: #0c6442;
  --fail-bg: #fbe9e9; --fail-fg: #9d1f1f;
  --code-bg: #eceff0;
  --display: "Newsreader", ui-serif, Georgia, "Times New Roman", serif;
  --body: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #101413; --panel: #181d1c; --border: #2c3433; --rule: #3a4342;
    --text: #eef2f1; --muted: #9aa4a3; --accent: #7fb1f0;
    --pass-bg: #102c1f; --pass-fg: #63c79a;
    --fail-bg: #331616; --fail-fg: #ef8c8c;
    --code-bg: #1f2524;
  }
}
:root[data-theme="dark"] {
  --surface: #101413; --panel: #181d1c; --border: #2c3433; --rule: #3a4342;
  --text: #eef2f1; --muted: #9aa4a3; --accent: #7fb1f0;
  --pass-bg: #102c1f; --pass-fg: #63c79a;
  --fail-bg: #331616; --fail-fg: #ef8c8c;
  --code-bg: #1f2524;
}

body {
  background: var(--surface); color: var(--text);
  font-family: var(--body); font-size: 15.5px; line-height: 1.65;
  padding-block: 48px 88px; padding-left: 20px; padding-right: 20px;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 830px; margin: 0 auto; display: flex; flex-direction: column; }

h1 {
  font-family: var(--display); font-weight: 500; font-size: clamp(2rem, 5vw, 2.75rem);
  line-height: 1.12; letter-spacing: -0.015em; text-wrap: balance; margin: 0;
}
.lede {
  font-family: var(--display); font-style: italic; font-size: 1.14rem;
  color: var(--muted); max-width: 60ch; margin: 12px 0 0; text-wrap: balance;
}
h2 {
  font-family: var(--display); font-weight: 500; font-size: 1.5rem;
  letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 4px;
}
.section-number {
  font-family: var(--mono); font-size: 0.72rem; font-weight: 500;
  letter-spacing: 0.14em; color: var(--accent); text-transform: uppercase;
  display: block; margin-bottom: 10px;
}
section { border-top: 1px solid var(--rule); padding-top: 26px; margin-top: 46px; }
p { margin: 0 0 15px; max-width: 68ch; }
p:last-child { margin-bottom: 0; }

.meta { margin-top: 30px; }
.meta dl {
  display: grid; grid-template-columns: max-content 1fr; gap: 7px 26px; margin: 0;
  font-size: 0.88rem;
}
.meta dt {
  font-family: var(--mono); font-size: 0.73rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); padding-top: 3px;
}
.meta dd { margin: 0; font-variant-numeric: tabular-nums; }

.checks { display: flex; flex-direction: column; gap: 0; margin-top: 22px; }
.check {
  display: grid; grid-template-columns: 92px 1fr; gap: 4px 20px;
  padding: 15px 0; border-bottom: 1px solid var(--border); align-items: baseline;
}
.check:first-child { border-top: 1px solid var(--border); }
.badge {
  font-family: var(--mono); font-size: 0.7rem; font-weight: 500;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 3px 9px; border-radius: 3px; justify-self: start;
}
.badge.pass { background: var(--pass-bg); color: var(--pass-fg); }
.badge.fail { background: var(--fail-bg); color: var(--fail-fg); }
.check-name { font-family: var(--mono); font-size: 0.92rem; }
.kind {
  font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); grid-column: 1;
}
.check-summary { color: var(--muted); font-size: 0.92rem; margin: 0; grid-column: 2; }

figure { margin: 26px 0 0; }
figure svg { max-width: 100%; height: auto; display: block; }
figcaption {
  color: var(--muted); font-size: 0.86rem; margin-top: 10px; max-width: 66ch;
  padding-left: 14px; border-left: 2px solid var(--border);
}
.table-wrap { overflow-x: auto; margin: 22px 0 0; }
table {
  border-collapse: collapse; font-size: 0.86rem; min-width: 100%;
  font-variant-numeric: tabular-nums;
}
th, td { text-align: right; padding: 8px 14px; border-bottom: 1px solid var(--border);
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
th:last-child, td:last-child { padding-right: 0; }
th {
  font-family: var(--mono); color: var(--muted); font-weight: 500; font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.08em;
  border-bottom: 1px solid var(--rule);
}
code {
  background: var(--code-bg); padding: 1px 5px; border-radius: 3px;
  font-size: 0.85em; font-family: var(--mono);
}
.caveat {
  border-left: 2px solid var(--accent); padding: 4px 0 4px 18px;
  color: var(--muted); margin: 24px 0 0; font-size: 0.93rem;
}
.caveat code { background: none; padding: 0; color: var(--text); }

.fig-dark { display: none; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .fig-light { display: none; }
  :root:not([data-theme="light"]) .fig-dark { display: block; }
}
:root[data-theme="dark"] .fig-light { display: none; }
:root[data-theme="dark"] .fig-dark { display: block; }

@media (max-width: 560px) {
  .check { grid-template-columns: 1fr; gap: 3px; }
  .kind, .check-summary { grid-column: 1; }
  .meta dl { grid-template-columns: 1fr; gap: 2px; }
  .meta dt { padding-top: 10px; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
"""


def _figure_block(svgs: dict[str, str], caption: str) -> str:
    light = svgs["light"]
    dark = svgs["dark"]
    return (f'<figure><div class="fig-light">{light}</div>'
            f'<div class="fig-dark">{dark}</div>'
            f'<figcaption>{caption}</figcaption></figure>')


def _table(frame: pd.DataFrame, formatters: dict[str, str] | None = None,
           columns: Sequence[str] | None = None,
           headers: Sequence[str] | None = None) -> str:
    formatters = formatters or {}
    columns = list(columns) if columns is not None else list(frame.columns)
    headers = list(headers) if headers is not None else columns

    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            spec = formatters.get(column)
            if spec is None:
                text = str(value)
            elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
                text = "—"
            else:
                text = format(value, spec)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return ('<div class="table-wrap"><table><thead><tr>' + head
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _summary_pt(result: CheckResult) -> str:
    """The check's summary in Portuguese, for the reader of the report.

    ``CheckResult.summary`` is written in English alongside the rest of the
    code; the report is read by a Portuguese-speaking audience, so the numbers
    are re-rendered here from ``extra`` rather than translated by string
    substitution. Anything without a translation falls back to the original.
    """
    extra = result.extra
    if result.name == "bell_diagonal_operations":
        return (f"{len(result.detail)} comparações em BBPSSW, DEJMPS, swapping e "
                f"decoerência de memória; maior discrepância absoluta "
                f"{extra.get('max_abs_error', 0):.2e} (tolerância "
                f"{extra.get('tolerance', 0):.0e})")
    if result.name == "link_success_probability":
        total = len(result.detail) if result.detail is not None else 0
        return (f"a fórmula single-heralded cai dentro do intervalo de 95% em "
                f"{extra.get('agreeing_points', 0)}/{total} distâncias; maior desvio "
                f"relativo {extra.get('worst_relative_error', 0):.1%}, na ponta longa "
                f"onde a contagem de sucessos é baixa")
    if result.name == "drift_statistics":
        ratio = extra.get("diffusive_ratio", float("nan"))
        tail = (f"deriva térmica RMS / prevista = {ratio:.3f}"
                if ratio == ratio else "escala térmica não testada nesta janela")
        return (f"vibração estacionária (RMS 2ª/1ª metade = "
                f"{extra.get('stationary_ratio', float('nan')):.2f}); {tail}")
    if result.name == "purification_break_even":
        parts = []
        for gate, threshold in extra.get("break_even", {}).items():
            parts.append(f"portas a {gate:.2f} → F&gt;{threshold:.3f}"
                         if threshold == threshold else
                         f"portas a {gate:.2f} → nunca ajuda")
        return "fidelidade mínima de entrada para o BBPSSW compensar: " + ", ".join(parts)
    if result.name == "swap_chain":
        return ("o swapping repetido de enlaces Werner reproduz exatamente "
                "F = (1 + 3·wⁿ)/4 com portas ideais; enlaces a 0,95 caem para "
                "0,713 após 6 swaps")
    return result.summary


def _section(number: str, eyebrow: str, heading: str, *body: str) -> str:
    """One report section, with an eyebrow that says how strong its evidence is."""
    return (f'<section><span class="section-number">{html.escape(number)} &middot; '
            f'{html.escape(eyebrow)}</span><h2>{heading}</h2>'
            + "".join(body) + "</section>")


def render_report(results: Iterable[CheckResult], output_path: str | Path,
                  title: str = "qnet_autopilot — validação da camada física") -> Path:
    """Write the HTML report for a set of check results.

    Args:
        results: what :func:`~qnet_autopilot.validation.checks.run_all` returned.
        output_path: destination ``.html`` file.
        title: page title.

    Returns:
        The path written.
    """
    from ..compat import bootstrap

    status = bootstrap()
    results = list(results)
    by_name = {result.name: result for result in results}

    sections: list[str] = []

    # -- provenance --------------------------------------------------------
    shimmed = ", ".join(sorted(status.shimmed)) or "nenhuma"
    sections.append(f"""
<div class="meta"><dl>
  <dt>SeQUeNCe</dt><dd>{html.escape(status.sequence_version or '?')}</dd>
  <dt>Dependências com shim</dt><dd>{html.escape(shimmed)}</dd>
  <dt>Formalismo</dt><dd>Bell-diagonal (BDS)</dd>
  <dt>Verificações</dt><dd>{sum(r.passed for r in results)} de {len(results)} passaram</dd>
</dl></div>""")

    checks_html = []
    for result in results:
        badge = "pass" if result.passed else "fail"
        label = "passou" if result.passed else "falhou"
        kind = {"exact": "exato", "statistical": "estatístico",
                "internal": "interno"}.get(result.kind, result.kind)
        checks_html.append(f"""
<div class="check">
  <span class="badge {badge}">{label}</span>
  <span class="check-name">{html.escape(result.name)}</span>
  <span class="kind">{html.escape(kind)}</span>
  <p class="check-summary">{_summary_pt(result)}</p>
</div>""")

    sections.append(_section(
        "00", "como ler", "Três níveis de evidência",
        "<p><strong>Exato</strong> significa concordância com o SeQUeNCe até a "
        "precisão de máquina — qualquer discrepância é bug. <strong>Estatístico</strong> "
        "compara uma probabilidade prevista com uma frequência medida: o máximo que se "
        "pode afirmar é que o modelo não é contradito pelo intervalo de confiança. "
        "<strong>Interno</strong> confere a implementação contra a própria definição do "
        "modelo — não contra fibra real, para a qual este pacote não tem dado de campo. "
        "As seções abaixo estão em ordem decrescente de força de evidência.</p>",
        '<div class="checks">' + "".join(checks_html) + "</div>"))

    # -- exact -------------------------------------------------------------
    bds_result = by_name.get("bell_diagonal_operations")
    if bds_result is not None and bds_result.detail is not None:
        grouped = (bds_result.detail.groupby("operation")["max_abs_error"]
                   .agg(["count", "max"]).reset_index())
        sections.append(_section(
            "01", "evidência exata",
            "As operações Bell-diagonais são idênticas às do SeQUeNCe",
            "<p>As funções de purificação, swapping e decoerência em "
            "<code>qnet_autopilot.physics.bell_diagonal</code> são transcrições "
            "vetorizadas dos protocolos do SeQUeNCe, feitas para rodar milhões de "
            "vezes dentro do laço de um agente sem instanciar timeline, nós e "
            "memórias. Elas não são uma aproximação: com estados Bell-diagonais e "
            "fidelidades de porta e medição aleatórias, a maior discrepância "
            f"observada foi <strong>{bds_result.extra.get('max_abs_error', 0):.1e}</strong>.</p>",
            _table(grouped, {"count": ",d", "max": ".2e"},
                   columns=["operation", "count", "max"],
                   headers=["operação", "comparações", "maior erro absoluto"])))

    # -- statistical -------------------------------------------------------
    link_result = by_name.get("link_success_probability")
    if link_result is not None and link_result.detail is not None:
        sections.append(_section(
            "02", "evidência estatística",
            "Taxa de heralding: fórmula fechada contra simulação",
            "<p>Este é o único ponto onde a camada analítica <em>não</em> é uma "
            "transcrição: o SeQUeNCe não tem probabilidade de sucesso configurável — "
            "ela emerge da emissão de fótons, da perda no canal e dos cliques dos "
            "detectores. A fórmula fechada é, portanto, uma previsão que precisa ser "
            "conferida.</p>",
            _figure_block(
                {theme.name: _figure_link_rates(link_result.detail, theme)
                 for theme in THEMES},
                "Detectores e acoplamento ideais, 0,2 dB/km, heralding no ponto médio. "
                "Barras: intervalo de Wilson 95% sobre as tentativas simuladas."),
            _table(
            link_result.detail,
            {"length_km": ".0f", "successes": ",d", "measured": ".4f",
             "ci_low": ".4f", "ci_high": ".4f", "single_heralded": ".4f",
             "barrett_kok": ".4f", "single_heralded_rel_error": "+.1%"},
            columns=["length_km", "successes", "measured", "ci_low", "ci_high",
                     "single_heralded", "barrett_kok", "single_heralded_rel_error"],
            headers=["km", "sucessos", "medido", "IC inf.", "IC sup.",
                     "single-heralded", "Barrett-Kok", "erro rel."]),
            '<p class="caveat">O protocolo <code>EntanglementGenerationA</code> do '
            'SeQUeNCe segue a curva <code>single_heralded</code>, não a de '
            'Barrett-Kok — que fica uma ordem de grandeza abaixo. Se você estava '
            'assumindo Barrett-Kok ao dimensionar taxas, esse é o fator que faltava. '
            'O desvio cresce nas distâncias longas, onde a contagem de sucessos cai '
            'e a estimativa binomial fica ruidosa: aumente o número de tentativas '
            'antes de concluir qualquer coisa acima de ~40 km.</p>'))

    # -- purification ------------------------------------------------------
    purification = by_name.get("purification_break_even")
    if purification is not None and purification.detail is not None:
        break_even = purification.extra.get("break_even", {})
        frame = pd.DataFrame({
            "gate": [f"{gate:.2f}" for gate in break_even],
            "threshold": [f"{value:.3f}" if value == value else "nunca ajuda"
                          for value in break_even.values()],
        })
        sections.append(_section(
            "03", "consequência de projeto", "Onde a purificação deixa de compensar",
            "<p>Purificar consome dois pares para produzir um, e as portas locais "
            "que a executam têm erro próprio. Abaixo de certa fidelidade de entrada "
            "esse erro introduzido supera o erro removido, e o par “purificado” sai "
            "pior do que entrou. O limiar se move rápido com a qualidade das portas — "
            "é o número mais decisivo que um controlador de repetidor precisa "
            "conhecer, e é por isso que o bug de fidelidade de porta descrito no "
            "README importa tanto.</p>",
            _figure_block(
                {theme.name: _figure_purification(purification.detail, theme)
                 for theme in THEMES},
                "Acima de zero, purificar ajuda. Com portas a 0,95 a curva nunca cruza "
                "zero: nessa qualidade de hardware, purificar sempre piora o par."),
            _table(frame, {},
                   columns=["gate", "threshold"],
                   headers=["fidelidade de porta e medição",
                            "fidelidade mínima de entrada"])))

    # -- swapping ----------------------------------------------------------
    chain = by_name.get("swap_chain")
    if chain is not None and chain.detail is not None:
        sections.append(_section(
            "04", "consequência de projeto",
            "Fidelidade ponta a ponta ao longo da cadeia",
            "<p>Swapping compõe multiplicativamente: o parâmetro de Werner dos "
            "enlaces se multiplica, então a fidelidade cai geometricamente com o "
            "número de saltos. Com portas ideais o resultado fechado é "
            "<code>F = (1 + 3·w<sup>n</sup>)/4</code>, e a aplicação repetida da "
            "função reproduz isso exatamente — o que também serve de verificação da "
            "composição.</p>",
            _figure_block(
                {theme.name: _figure_swap_chain(chain.detail, theme) for theme in THEMES},
                "Enlaces de fidelidade 0,95. O alcance útil de uma cadeia sem "
                "purificação é de poucos saltos, e portas imperfeitas o encurtam mais.")))

    # -- drift -------------------------------------------------------------
    drift = by_name.get("drift_statistics")
    if drift is not None and drift.detail is not None:
        sections.append(_section(
            "05", "evidência interna", "O modelo de deriva de polarização",
            "<p>Aqui não há referência externa: o modelo de fibra não tem "
            "contraparte no SeQUeNCe e este pacote não tem dado de campo. O que se "
            "pode conferir é se a implementação corresponde ao modelo documentado — "
            "se a vibração é estacionária (jitter, não deriva) e se a componente "
            "térmica difunde na taxa que seus parâmetros implicam.</p>",
            _figure_block(
                {theme.name: _figure_drift(drift.detail, theme) for theme in THEMES},
                "A previsão usa apenas dois graus de liberdade: uma rotação em torno "
                "do próprio eixo de lançamento não desloca o estado. Contar três "
                "superestima a deriva em &radic;(3/2) &asymp; 1,22."),
            _figure_block(
                {theme.name: _figure_trajectory(theme) for theme in THEMES},
                "Uma hora de fibra metropolitana enterrada, sem compensação: deriva "
                "térmica lenta sob jitter mecânico, interrompida por saltos discretos. "
                "Os saltos são o que um compensador precisa <em>recuperar</em>, não "
                "prever."),
            '<p class="caveat">Passar nessas verificações diz que o código '
            'implementa o modelo, não que o modelo descreve a sua fibra. As '
            'constantes em <code>physics/constants.py</code> são um ponto de '
            'operação plausível para fibra enterrada, documentado como centro de '
            'varredura — fibra aérea deriva uma ordem de grandeza mais rápido. '
            'Trocá-las por medições do seu enlace é uma linha.</p>'))

    body = "\n".join(sections)
    document = f"""<title>{html.escape(title)}</title>
{_FONTS}
<style>{_CSS}</style>
<main>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="lede">Uma camada analítica rápida sobre o SeQUeNCe só vale se estiver
       certa. Isto é o que foi conferido, contra o quê, e com que força.</p>
  </header>
  {body}
</main>
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
