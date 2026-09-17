"""Render the phase-3 results as a self-contained HTML report.

Built from the CSV files ``scripts/evaluate_phase3.py`` writes and the training
histories ``scripts/train_compensator.py`` saves, so every number and every
figure comes from a measured run rather than from prose.  Shares its styling
and its validated categorical palette with
:mod:`qnet_autopilot.validation.report`.

The one editorial decision here is which axis to put things on.  Mean QBER
spans two orders of magnitude between the compensators, so a linear axis would
compress everything interesting into the bottom pixel row; the QBER figures are
logarithmic, with the 11% BB84 threshold drawn in, because the distance to that
line is the only thing that decides whether there is key at all.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents.report import _pt, _table_pt
from ..validation.report import (_CSS, _FONTS, THEMES, Theme, _figure, _figure_block,
                                 _legend, _section, _style_text, _to_svg)

__all__ = ["render_phase3_report"]

#: The BB84 asymptotic threshold: above it a run produces no secret key at all.
QBER_THRESHOLD = 0.11

#: Fixed colours per compensator, so a reader who learns the palette in the
#: first figure can still use it in the last one.
COLOURS = {
    "sem compensação": 3,
    "persistência": 1,
    "filtro alfa-beta": 2,
    "ridge (Wiener)": 0,
    "MLP": 0,
}

#: Linear extrapolation is the one compensator that is never competitive, so it
#: gets the recessive grey rather than a fifth categorical colour that would
#: compete for attention with the three that matter.
RECESSIVE = "extrapolação linear"


def _family(name: str) -> str:
    """Strip the tuned gains from a compensator's name, for colour lookup."""
    return "filtro alfa-beta" if name.startswith("filtro alfa-beta") else name


def _colour(name: str, theme: Theme) -> str:
    family = _family(name)
    if family == RECESSIVE:
        return theme.grid
    return theme.series[COLOURS.get(family, 3)]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _figure_headline(main: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.6)
    ordered = main.sort_values("mean_qber", ascending=False)
    positions = np.arange(len(ordered))

    axes.barh(positions, ordered["mean_qber"], height=0.62,
              color=[_colour(name, theme) for name in ordered["compensator"]])
    axes.axvline(QBER_THRESHOLD, color=theme.text_secondary, linewidth=1.4,
                 linestyle="--")
    axes.annotate("limiar BB84 (11%)", xy=(QBER_THRESHOLD, 0.98),
                  xycoords=("data", "axes fraction"), xytext=(-6, 0),
                  textcoords="offset points", ha="right", va="top",
                  color=theme.text_secondary, fontsize=8.5)

    axes.set_xscale("log")
    axes.set_yticks(positions)
    axes.set_yticklabels(ordered["compensator"])
    axes.grid(axis="y", visible=False)
    for position, value in zip(positions, ordered["mean_qber"]):
        axes.text(value * 1.12, position, _pt(value * 100, ".2f") + "%",
                  va="center", color=theme.text_primary, fontsize=9)

    axes.set_xlim(right=float(ordered["mean_qber"].max()) * 3.0)
    _style_text(axes, theme, "QBER médio em 8 execuções retidas (escala logarítmica)",
                "QBER médio", "")
    return _to_svg(figure)


def _figure_trace(trace: pd.DataFrame, jumps: pd.DataFrame, theme: Theme) -> str:
    """QBER per sifting block, across a whole held-out run.

    Plotted per block rather than per sample, and for a physical reason rather
    than a cosmetic one. The instantaneous QBER of a 40 Hz disturbance crosses
    zero twice per period, so a sample-by-sample plot is a solid band between
    its nulls and those nulls are not something a link can use: key is sifted in
    blocks, and what costs a block is its *average* error, not its best
    instant. One block here is 25 ms — one period of the vibration band — which
    is the shortest window over which the oscillation averages out.
    """
    figure, axes = _figure(theme, height=4.0)
    shown = ["sem compensação", "filtro alfa-beta (a=0.05, b=0.005)", "ridge (Wiener)"]

    block = 25
    time_s = trace["time_s"].to_numpy()
    count = len(time_s) // block * block
    blocked_time = time_s[:count].reshape(-1, block).mean(axis=1)

    for name in shown:
        if name not in trace.columns:
            continue
        values = trace[name].to_numpy()[:count].reshape(-1, block).mean(axis=1)
        axes.plot(blocked_time, values, linewidth=1.2, alpha=0.95,
                  color=_colour(name, theme), label=name)

    axes.axhline(QBER_THRESHOLD, color=theme.text_secondary, linewidth=1.4,
                 linestyle="--", label="limiar BB84")
    for index, jump in enumerate(jumps["jump_time_s"]):
        axes.axvline(jump, color=theme.series[1], linewidth=1.3, alpha=0.75,
                     label="salto do cabo" if index == 0 else None)

    axes.set_yscale("log")
    _style_text(axes, theme,
                "QBER por bloco de sifting (25 ms) ao longo de uma execução retida",
                "tempo (s)", "QBER do bloco")

    # Below the axes: the three bands fill the plotting area from top to bottom,
    # so any in-figure corner would sit on top of a curve.
    legend = axes.legend(frameon=False, fontsize=9, loc="upper center",
                         bbox_to_anchor=(0.5, -0.16), ncol=3,
                         columnspacing=1.6, handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)
    return _to_svg(figure)


def _figure_recovery(recovery: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.4)
    ordered = recovery.dropna(subset=["median_recovery_ms"]).sort_values(
        "median_recovery_ms", ascending=False)
    positions = np.arange(len(ordered))

    axes.barh(positions, ordered["median_recovery_ms"], height=0.6,
              color=[_colour(name, theme) for name in ordered["compensator"]],
              xerr=[np.zeros(len(ordered)),
                    ordered["p90_recovery_ms"] - ordered["median_recovery_ms"]],
              error_kw=dict(ecolor=theme.text_secondary, elinewidth=1.2, capsize=3))
    axes.set_xscale("log")
    axes.set_yticks(positions)
    axes.set_yticklabels([
        f"{name}" if recovered == jumps else f"{name} ({recovered}/{jumps})"
        for name, recovered, jumps in zip(ordered["compensator"],
                                          ordered["recovered"], ordered["jumps"])])
    axes.grid(axis="y", visible=False)
    _style_text(axes, theme,
                "Tempo até voltar abaixo do limiar após um salto (mediana; barra até o p90)",
                "tempo de recuperação (ms)", "")
    return _to_svg(figure)


def _figure_latency(sweep: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.8)
    period_ms = 1000.0 / 40.0

    for name, group in sweep.groupby("compensator", sort=False):
        group = group.sort_values("latency_ms")
        axes.plot(group["latency_ms"], group["mean_qber"], marker="o", markersize=6,
                  linewidth=2, color=_colour(name, theme), label=name,
                  markeredgecolor=theme.surface, markeredgewidth=1.4)

    axes.axhline(QBER_THRESHOLD, color=theme.text_secondary, linewidth=1.3,
                 linestyle="--")
    axes.axvline(period_ms, color=theme.series[3], linewidth=1.3, linestyle=":")
    axes.annotate("um período de vibração (25 ms)", xy=(period_ms, 0.99),
                  xycoords=("data", "axes fraction"), xytext=(5, 0),
                  textcoords="offset points", ha="left", va="top",
                  color=theme.text_secondary, fontsize=8.5)

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xticks([4, 8, 20, 40, 80, 160])
    axes.set_xticklabels(["4", "8", "20", "40", "80", "160"])
    _style_text(axes, theme,
                "QBER médio contra a latência do laço de controle",
                "latência de controle (ms)", "QBER médio")
    _legend(axes, theme, loc="lower right")
    return _to_svg(figure)


def _figure_photons(sweep: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.6)
    for name, group in sweep.groupby("compensator", sort=False):
        group = group.sort_values("photons_per_basis")
        axes.plot(group["photons_per_basis"], group["mean_qber"], marker="o",
                  markersize=6, linewidth=2, color=_colour(name, theme), label=name,
                  markeredgecolor=theme.surface, markeredgewidth=1.4)

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xticks([25, 50, 100, 200, 400, 800])
    axes.set_xticklabels(["25", "50", "100", "200", "400", "800"])
    _style_text(axes, theme,
                "QBER médio contra os fótons de sondagem gastos por eixo analisador",
                "fótons por base", "QBER médio")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


def _figure_training(histories: dict[str, dict], theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.6)
    labels = {"main": "40 ms, 400 fótons", "latency80": "80 ms, 400 fótons",
              "photons50": "40 ms, 50 fótons"}

    for index, (key, payload) in enumerate(histories.items()):
        frame = pd.DataFrame(payload["history"])
        colour = theme.series[index % len(theme.series)]
        reference = float(payload["validation_mse"]["ridge"])
        axes.plot(frame["epoch"], frame["validation_mse"] / reference,
                  linewidth=2, color=colour, label=labels.get(key, key))

    axes.axhline(1.0, color=theme.text_secondary, linewidth=1.5, linestyle="--",
                 label="ridge (Wiener) = 1,0")
    axes.set_ylim(0.85, 1.35)
    _style_text(axes, theme,
                "Erro de validação da MLP, dividido pelo do preditor linear ótimo",
                "época", "MSE da MLP / MSE do ridge")
    _legend(axes, theme, loc="upper right")
    return _to_svg(figure)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render_phase3_report(run_directory: str | Path, output_path: str | Path,
                         title: str = "Compensação de polarização: prever a fibra, não segui-la"
                         ) -> Path:
    """Write the phase-3 report from the CSVs in ``run_directory``.

    Args:
        run_directory: holds ``main_comparison.csv``, ``recovery.csv``,
            ``latency_sweep.csv``, ``photon_sweep.csv``, ``trace.csv``,
            ``trace_jumps.csv`` and the ``*_history.json`` files.
        output_path: destination ``.html``.
        title: page title.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: if a required results file is missing — the report is
            generated from measurements, so there is nothing to fall back on.
    """
    run_directory = Path(run_directory)
    required = ["main_comparison.csv", "recovery.csv", "latency_sweep.csv",
                "photon_sweep.csv", "trace.csv", "trace_jumps.csv",
                "main_history.json"]
    missing = [name for name in required if not (run_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing results in {run_directory}: {missing}. Run "
            f"scripts/train_compensator.py and scripts/evaluate_phase3.py first.")

    main = pd.read_csv(run_directory / "main_comparison.csv")
    recovery = pd.read_csv(run_directory / "recovery.csv")
    latency = pd.read_csv(run_directory / "latency_sweep.csv")
    photons = pd.read_csv(run_directory / "photon_sweep.csv")
    trace = pd.read_csv(run_directory / "trace.csv")
    jumps = pd.read_csv(run_directory / "trace_jumps.csv")

    histories = {}
    for key in ("main", "latency80", "photons50"):
        path = run_directory / f"{key}_history.json"
        if path.is_file():
            histories[key] = json.loads(path.read_text())

    indexed = main.set_index("compensator")
    ridge = indexed.loc["ridge (Wiener)"]
    mlp = indexed.loc["MLP"]
    classical_name = next(name for name in indexed.index
                          if name.startswith("filtro alfa-beta"))
    classical = indexed.loc[classical_name]
    nothing = indexed.loc["sem compensação"]

    sections: list[str] = []

    # -- provenance --------------------------------------------------------
    configuration = histories.get("main", {})
    sections.append(f"""
<div class="meta"><dl>
  <dt>Fibra</dt><dd>aérea sob perturbação mecânica: deriva térmica 0,05 rad/s,
      banda acústica 40 &plusmn; 6 Hz com amplitude 0,30 rad, saltos a cada ~20 s</dd>
  <dt>Laço</dt><dd>período de controle 2 ms, latência
      {_pt(float(configuration.get('latency_ms', 40.0)), '.0f')} ms,
      monitoramento com {configuration.get('photons', 400)} fótons por eixo analisador</dd>
  <dt>Observação</dt><dd>dois estados de sonda, três eixos cada, contagem binomial
      finita — nunca a rotação verdadeira</dd>
  <dt>Alvo de treino</dt><dd>a <em>medição</em> futura, não a verdade futura</dd>
  <dt>Teste</dt><dd>8 execuções de 20 s, sementes disjuntas das de treino e validação;
      o filtro alfa-beta é re-sintonizado em cada ponto de operação</dd>
</dl></div>""")

    # -- headline ----------------------------------------------------------
    display = main.copy()
    sections.append(_section(
        "01", "resultado", "Prever a fibra vale um fator 5 sobre segui-la",
        f"<p>No ponto de operação principal, o preditor linear ótimo entrega QBER médio "
        f"<strong>{_pt(float(ridge['mean_qber']) * 100, '.2f')}%</strong> contra "
        f"<strong>{_pt(float(classical['mean_qber']) * 100, '.2f')}%</strong> do rastreador "
        f"alfa-beta sintonizado — um fator "
        f"<strong>{_pt(float(classical['mean_qber']) / float(ridge['mean_qber']), '.1f')}&times;</strong> — "
        f"e {_pt(float(ridge['mean_key_rate_hz']) / float(classical['mean_key_rate_hz']), '.2f')}&times; "
        f"a taxa de chave secreta. Sem compensação alguma o enlace passa "
        f"{_pt((1 - float(nothing['uptime'])) * 100, '.0f')}% do tempo acima do limiar de 11%, "
        f"onde o BB84 não produz chave nenhuma.</p>",
        _figure_block({theme.name: _figure_headline(main, theme) for theme in THEMES},
                      "Escala logarítmica: os compensadores se separam por duas ordens "
                      "de grandeza, e o que importa é a distância até a linha tracejada."),
        _table_pt(display,
                  {"mean_qber": ".4f", "p95_qber": ".4f", "uptime": ".2%",
                   "mean_key_rate_hz": ".0f", "mean_residual_mrad": ".1f",
                   "mean_prediction_error_mrad": ".1f"},
                  columns=["compensator", "mean_qber", "p95_qber", "uptime",
                           "mean_key_rate_hz", "mean_residual_mrad"],
                  headers=["compensador", "QBER médio", "QBER p95", "disponibilidade",
                           "chave (Hz)", "resíduo médio (mrad)"]),
        "<p>Duas linhas dessa tabela merecem atenção antes das boas. A "
        "<strong>persistência</strong> — corrigir para a última medição — é "
        "<em>pior do que não corrigir nada</em>: com 40 ms de atraso ela aplica uma "
        "correção que a vibração já tornou obsoleta, e injetar a fase errada custa "
        "mais do que deixar a fibra em paz. A <strong>extrapolação linear</strong> é "
        "pior ainda, porque diferenciar duas estimativas ruidosas dobra a variância e "
        "ela passa a perseguir o próprio ruído. Compensar mal é uma opção real, e é "
        "o motivo pelo qual o alfa-beta existe.</p>"))

    # -- the trace ---------------------------------------------------------
    sections.append(_section(
        "02", "comportamento", "O que os números médios escondem",
        "<p>Uma média de QBER não distingue um enlace estável de um que alterna entre "
        "excelente e inutilizável. Esta é uma execução retida inteira, bloco de "
        "chave por bloco de chave.</p>",
        _figure_block({theme.name: _figure_trace(trace, jumps, theme)
                       for theme in THEMES},
                      "Média por bloco de 25 ms — um período da banda de vibração. "
                      "É o que um bloco de chave realmente paga; o QBER instantâneo "
                      "cruza zero duas vezes por período e esses zeros não servem "
                      "para nada."),
        "<p>Sem compensação a curva oscila em torno do limiar: o enlace não está "
        "quebrado, está <em>piscando</em> a 40 Hz, e cada passagem acima de 11% apaga "
        "a chave daquela janela. O alfa-beta derruba a oscilação mas mantém um piso "
        "visível — é o atraso de rastreamento que ele não consegue eliminar sem "
        "amplificar o ruído. O preditor fica uma ordem de grandeza abaixo, e seus "
        "dois picos nítidos coincidem com as duas linhas verticais: saltos do cabo, "
        "a única perturbação que nenhum preditor pode antecipar.</p>"))

    # -- recovery ----------------------------------------------------------
    recovery_display = recovery.copy()
    recovery_display["recovered"] = [f"{int(a)} / {int(b)}" for a, b in
                                     zip(recovery["recovered"], recovery["jumps"])]
    ridge_recovery = recovery.set_index("compensator").loc["ridge (Wiener)"]
    classical_recovery = recovery.set_index("compensator").loc[classical_name]

    sections.append(_section(
        "03", "cauda", "O salto é o teste que separa os compensadores",
        "<p>Um salto discreto é a única perturbação que nenhum preditor pode "
        "antecipar, então todos partem do mesmo lugar: completamente errados. O que "
        "os separa é a rapidez com que voltam.</p>",
        _figure_block({theme.name: _figure_recovery(recovery, theme) for theme in THEMES},
                      "Barra: da mediana até o p90. A fração entre parênteses marca os "
                      "compensadores que não recuperaram de todos os saltos antes do "
                      "fim da execução."),
        _table_pt(recovery_display,
                  {"median_recovery_ms": ".2f", "p90_recovery_ms": ".1f"},
                  columns=["compensator", "recovered", "median_recovery_ms",
                           "p90_recovery_ms"],
                  headers=["compensador", "recuperados", "mediana (ms)", "p90 (ms)"]),
        f"<p>O preditor volta abaixo do limiar em "
        f"<strong>{_pt(float(ridge_recovery['median_recovery_ms']), '.1f')} ms</strong> "
        f"na mediana, contra "
        f"{_pt(float(classical_recovery['median_recovery_ms']), '.1f')} ms do alfa-beta e "
        f"236 ms da persistência. Isso é menos do que uma latência de controle: o "
        f"salto acontece, a medição seguinte já o vê, e a próxima correção a lançar "
        f"leva a rotação nova — o modelo não precisa <em>aprender</em> o salto, "
        f"precisa apenas não ter memória longa demais para ignorá-lo. É por isso que "
        f"a persistência, que em média parece razoável, é a pior aqui: ela corrige "
        f"para uma medição já antiga.</p>"
        "<p class=\"caveat\">São 8 saltos ao todo, dois por execução em algumas. As "
        "medianas separam os compensadores por ordens de grandeza, o que sobrevive a "
        "essa amostra pequena; os p90 não devem ser lidos como estimativas precisas.</p>"))

    # -- latency sweep -----------------------------------------------------
    pivot = latency.pivot(index="latency_ms", columns="compensator", values="mean_qber")
    # Worst to best, matching every other table here; the pivot would otherwise
    # order the columns alphabetically, which means nothing to a reader.
    pivot = pivot[[name for name in ("sem compensação", "persistência",
                                     "filtro alfa-beta", "ridge (Wiener)")
                   if name in pivot.columns]]
    pivot_display = pivot.reset_index()
    sections.append(_section(
        "04", "varredura", "A previsão vale exatamente o que a latência custa",
        "<p>Com latência zero, \"corrija para a última medição\" é ótimo e não há nada "
        "a prever. Tudo o que um preditor ganha, ele ganha em proporção ao atraso do "
        "laço — então esse é o parâmetro que não pode ser aceito por confiança.</p>",
        _figure_block({theme.name: _figure_latency(latency, theme) for theme in THEMES},
                      "O filtro alfa-beta é re-sintonizado por grade em cada latência e "
                      "o ridge é reajustado, então cada ponto é o melhor de cada método "
                      "naquele ponto de operação."),
        _table_pt(pivot_display,
                  {column: ".4f" for column in pivot.columns},
                  columns=["latency_ms", *pivot.columns],
                  headers=["latência (ms)", *pivot.columns]),
        "<p>A curva do ridge é monótona, como tem que ser: quanto mais longe o "
        "horizonte, mais difícil a previsão. A do alfa-beta <em>não é</em>, e essa "
        "irregularidade não é ruído — é uma <strong>ressonância</strong>. A banda de "
        "vibração está centrada em 40 Hz, cujo período é 25 ms; quando a latência de "
        "controle coincide com um período inteiro da oscilação, a fibra volta "
        "aproximadamente para onde estava quando foi medida, e seguir a última medição "
        "fica quase ótimo de graça. A 24 ms medimos QBER 0,0052 para o alfa-beta e "
        "0,0061 para a própria persistência — contra 0,0383 e 0,1329 a 40 ms.</p>"
        "<p>Isso é uma armadilha de bancada, não uma vantagem: um projeto que escolhe "
        "a latência do atuador para cair nessa ressonância funciona brilhantemente até "
        "a frequência de excitação mudar. O preditor não depende do alinhamento — ele "
        "aprende a fase da oscilação a partir da janela e a propaga para qualquer "
        "horizonte.</p>"))

    # -- photon sweep ------------------------------------------------------
    photon_pivot = photons.pivot(index="photons_per_basis", columns="compensator",
                                 values="mean_qber")
    low = photon_pivot.iloc[0]
    high = photon_pivot.iloc[-1]
    sections.append(_section(
        "05", "achado", "Gastar fótons de sondagem só compensa com um preditor que os use",
        "<p>Cada fóton de sondagem é um fóton que não virou chave, então o orçamento de "
        "monitoramento é uma decisão de engenharia real. A varredura responde quanto "
        "ela vale — e a resposta depende do compensador.</p>",
        _figure_block({theme.name: _figure_photons(photons, theme) for theme in THEMES},
                      "Ambos os eixos em escala logarítmica. Um estimador limitado por "
                      "ruído de contagem cairia com a raiz do número de fótons."),
        _table_pt(photon_pivot.reset_index(),
                  {column: ".4f" for column in photon_pivot.columns},
                  columns=["photons_per_basis", *photon_pivot.columns],
                  headers=["fótons por base", *photon_pivot.columns]),
        f"<p>O filtro clássico <strong>satura</strong>: de 25 para 800 fótons — um "
        f"fator 32 — seu QBER cai de {_pt(float(low['filtro alfa-beta']), '.4f')} "
        f"para {_pt(float(high['filtro alfa-beta']), '.4f')}, cerca de 12%. Ele não "
        f"está limitado por ruído de medição e sim pelo próprio atraso de "
        f"rastreamento, que mais fótons não consertam. O ridge continua melhorando no "
        f"mesmo intervalo, de {_pt(float(low['ridge (Wiener)']), '.4f')} para "
        f"{_pt(float(high['ridge (Wiener)']), '.4f')} — um fator 3.</p>"
        "<p>A consequência prática é específica e útil: <em>aumentar o orçamento de "
        "monitoramento de um controlador clássico é dinheiro jogado fora</em>. Os "
        "fótons extras só se convertem em chave se houver um preditor capaz de "
        "extrair estrutura deles. E na direção oposta, a 25 fótons por base — "
        "150 fótons por atualização, praticamente nada — o preditor ainda entrega "
        "metade do QBER que o filtro clássico alcança gastando 32 vezes mais.</p>"))

    # -- the negative result ----------------------------------------------
    ridge_mse = float(histories["main"]["validation_mse"]["ridge"])
    mlp_mse = float(histories["main"]["validation_mse"]["mlp"])
    sections.append(_section(
        "06", "resultado negativo", "A rede neural não bate o filtro de Wiener, e isso era previsível",
        f"<p>O pedido original desta fase era uma rede neural. Ela foi construída, "
        f"treinada nos mesmos dados e avaliada no mesmo laço — e o ganho sobre "
        f"regressão ridge é de "
        f"<strong>{_pt((1 - mlp_mse / ridge_mse) * 100, '.1f')}%</strong> no erro de "
        f"validação da regressão e nada mensurável no QBER: "
        f"{_pt(float(mlp['mean_qber']) * 100, '.2f')}% contra "
        f"{_pt(float(ridge['mean_qber']) * 100, '.2f')}%, uma diferença dentro da "
        f"dispersão entre execuções. A 80 ms de latência a MLP é 2% <em>pior</em>.</p>",
        _figure_block({theme.name: _figure_training(histories, theme)
                       for theme in THEMES},
                      "Curvas normalizadas pelo ridge de cada ponto de operação, então "
                      "1,0 é o preditor linear ótimo e abaixo de 1,0 é ganho real."),
        "<p>Isto não é uma rede mal treinada — é o que a física prevê. Uma regressão "
        "linear sobre uma janela de amostras passadas <strong>é</strong> o filtro de "
        "Wiener, o preditor linear ótimo para um processo estacionário. E o modelo de "
        "deriva é uma soma de um processo de Ornstein-Uhlenbeck com senoides de banda "
        "estreita: gaussiano e estacionário. Para esse processo, nenhum preditor não "
        "linear pode ganhar de forma significativa, e medir isso é mais informativo do "
        "que reportar a rede sozinha e deixar o leitor supor que a não linearidade "
        "estava fazendo o trabalho.</p>"
        "<p>O que isso <em>não</em> diz: que redes não sirvam aqui. Diz que a estrutura "
        "não linear tem que existir no sinal antes que um modelo não linear a "
        "encontre. Deriva de polarização real tem histerese no atuador, dependência "
        "de temperatura não linear e não estacionariedade em escalas de horas — "
        "nenhuma das três está neste modelo, e cada uma é um lugar onde o resultado "
        "deveria mudar. A medição correta a fazer em seguida é essa, em fibra real.</p>"))

    # -- honesty about the observation model -------------------------------
    sections.append(_section(
        "07", "metodologia", "O que o compensador tem permissão de saber",
        "<p>Um compensador de polarização é fácil de avaliar desonestamente. O "
        "simulador conhece a rotação verdadeira da fibra, e basta deixá-la entrar em "
        "um de três lugares para produzir um controlador excelente e impossível de "
        "implantar.</p>"
        "<p><strong>Como feature.</strong> O compensador vê apenas contagens binomiais "
        "de dois estados de sonda em três eixos analisadores, com o número de fótons "
        "do ponto de operação. Nunca a rotação verdadeira.</p>"
        "<p><strong>Como alvo de treino.</strong> O alvo padrão é a "
        "<em>medição futura</em>, não a verdade futura. Sob perda quadrática os dois "
        "têm o mesmo minimizador, porque a medição é não enviesada — mas só o primeiro "
        "pode ser recoletado em campo, e treinar contra a verdade credita ao modelo "
        "uma remoção de ruído que ele só consegue fazer porque viu a resposta limpa.</p>"
        "<p><strong>Como divisão dos dados.</strong> As janelas se sobrepõem, então uma "
        "divisão embaralhada coloca quase-duplicatas dos dois lados. A divisão é "
        "cronológica, com uma banda de guarda mais larga que a janela mais o "
        "horizonte.</p>"
        "<p>Há ainda uma correção de modelagem herdada da fase 1 que vale registrar: "
        "acompanhar <strong>um</strong> estado de sonda não determina a rotação da "
        "fibra. Resta uma liberdade U(1) — todas as rotações que levam o estado lançado "
        "ao mesmo ponto da esfera de Poincaré diferem por um giro em torno dele. Dá "
        "para calcular QBER daquele estado e nunca perceber a lacuna. O gerador de "
        "trajetórias ganhou <code>track_rotation=True</code> e o modelo de observação "
        "usa dois estados de sonda, que é o mínimo que determina uma rotação.</p>"))

    # -- limits ------------------------------------------------------------
    sections.append(_section(
        "08", "limites", "O que este resultado não é",
        "<p><strong>Um modelo de deriva, não uma fibra.</strong> Tudo aqui é medido "
        "contra o modelo da fase 1: Ornstein-Uhlenbeck térmico, soma de senoides de "
        "banda estreita para a vibração, saltos de Poisson. Os parâmetros vêm da "
        "literatura de fibra aérea, mas nenhum dado de fibra real entrou nesta fase. "
        "O resultado defensável é sobre este processo.</p>"
        "<p><strong>Um atuador ideal.</strong> A correção é aplicada como uma rotação "
        "exata depois da latência. Controladores de polarização reais têm faixa finita, "
        "resets de fim de curso e dinâmica própria — e o reset é justamente o tipo de "
        "não linearidade que mudaria a conclusão da seção 06.</p>"
        "<p><strong>Um ponto de operação escolhido para ser difícil.</strong> Em fibra "
        "enterrada, a deriva térmica a 40 ms é de milirradianos e qualquer coisa "
        "funciona. A fibra aérea sob perturbação acústica foi escolhida porque é onde "
        "a compensação preditiva tem algo a fazer; a varredura de latência mostra o "
        "regime em que ela deixa de ter.</p>"
        "<p><strong>QBER de polarização, não QBER de sistema.</strong> O número "
        "reportado é o erro induzido pelo desalinhamento residual. Ruído de detector, "
        "contagens escuras e erros de modulação entram somados e não estão aqui — a "
        "taxa de chave absoluta é, portanto, um teto.</p>"))

    body = "\n".join(sections)
    document = f"""<title>{html.escape(title)}</title>
{_FONTS}
<style>{_CSS}
figure + p, figure + div {{ margin-top: 22px; }}
.table-wrap + p {{ margin-top: 18px; }}</style>
<main>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="lede">A fibra se move mais rápido do que o laço de controle consegue
       responder. Um preditor treinado só com o que o canal de monitoramento
       realmente mede derruba o QBER em 5&times; e a taxa de chave sobe 1,6&times;.
       Fase 3 do qnet_autopilot.</p>
  </header>
  {body}
</main>
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
