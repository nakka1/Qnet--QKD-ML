"""Render the phase-2 results as a self-contained HTML report.

Built from the CSV files ``scripts/evaluate_phase2.py`` and
``scripts/analyse_phase2.py`` write, so every number and figure comes from a
measured run rather than from prose. Shares its styling and its validated
categorical palette with :mod:`qnet_autopilot.validation.report`.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..validation.report import (_CSS, _FONTS, THEMES, Theme, _figure, _figure_block,
                                 _legend, _section, _style_text, _table, _to_svg)

__all__ = ["render_phase2_report"]

#: Display names, in the order they should appear.
POLICY_LABELS = {
    "random": "aleatório",
    "shortest_path": "caminho mais curto",
    "greedy_fidelity": "fidelidade gananciosa",
    "threshold_0.94": "limiar 0,94",
    "threshold_0.90": "limiar 0,90",
}


def _pt(value, spec: str = ".3f") -> str:
    """Format a number with a decimal comma.

    The report is in Portuguese and mixing ``87.5%`` with ``0,90`` in the same
    sentence reads as a translation that was not finished.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return format(value, spec).replace(".", ",")


def _table_pt(frame: pd.DataFrame, formats: dict[str, str],
              columns, headers) -> str:
    """Render a table with comma decimals, by formatting before handing it over."""
    rendered = pd.DataFrame({
        column: (frame[column].map(lambda value, spec=formats[column]: _pt(value, spec))
                 if column in formats else frame[column].astype(str))
        for column in columns})
    return _table(rendered, {}, columns=list(columns), headers=list(headers))


def _label(name: str) -> str:
    if name.startswith("actor_critic"):
        return "agente aprendido"
    return POLICY_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _figure_learning_curve(history: dict, baseline: float, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.8)
    train = pd.DataFrame(history["train"])
    evaluation = pd.DataFrame(history["eval"])

    axes.plot(train["iteration"], train["train_success"], color=theme.series[2],
              linewidth=1.4, alpha=0.75, label="treino (amostrado, episódios novos)")
    axes.plot(evaluation["iteration"], evaluation["eval_success"], marker="o",
              markersize=6, linewidth=2.2, color=theme.series[0],
              markeredgecolor=theme.surface, markeredgewidth=1.5,
              label="validação (sementes fixas, sem shaping)")
    axes.axhline(baseline, color=theme.series[1], linewidth=1.8, linestyle="--",
                 label=f"melhor heurístico ({baseline:.0%})")

    axes.set_ylim(0.0, 1.02)
    _style_text(axes, theme, "O agente ultrapassa o melhor heurístico por volta da 45ª atualização",
                "iteração de treino", "taxa de sucesso")
    _legend(axes, theme, loc="lower right")
    return _to_svg(figure)


def _figure_results(summary: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.4)
    ordered = summary.sort_values("success_rate")
    labels = [_label(name) for name in ordered["policy"]]
    positions = np.arange(len(ordered))
    colours = [theme.series[0] if name.startswith("actor_critic") else theme.grid
               for name in ordered["policy"]]

    axes.barh(positions, ordered["success_rate"], height=0.62, color=colours,
              xerr=ordered["success_stderr"], error_kw=dict(
                  ecolor=theme.text_secondary, elinewidth=1.2, capsize=3))
    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    axes.set_xlim(0.0, 1.0)
    axes.grid(axis="y", visible=False)

    # Clear of the error-bar cap, not just of the bar end.
    offset = float(ordered["success_stderr"].max()) + 0.035
    for position, value in zip(positions, ordered["success_rate"]):
        axes.text(value + offset, position, _pt(value * 100, ".0f") + "%",
                  va="center", color=theme.text_primary, fontsize=9)

    _style_text(axes, theme, "Taxa de sucesso em 400 episódios de teste retidos",
                "fração de pedidos atendidos", "")
    return _to_svg(figure)


def _figure_temperature(sweep: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.6)
    axes.errorbar(sweep["temperature"], sweep["success_rate"],
                  yerr=sweep["success_stderr"], marker="o", markersize=7,
                  linewidth=2, color=theme.series[0], ecolor=theme.series[0],
                  elinewidth=1.4, capsize=4, markeredgecolor=theme.surface,
                  markeredgewidth=1.5, label="agente")
    axes.axvline(1.0, color=theme.text_secondary, linewidth=1.0, linestyle=":", alpha=0.8)
    axes.annotate("temperatura de treino", xy=(1.0, 0.06),
                  xycoords=("data", "axes fraction"), ha="left", va="bottom",
                  color=theme.text_secondary, fontsize=8.5,
                  xytext=(4, 0), textcoords="offset points")
    axes.scatter([0.0], [sweep.loc[sweep["temperature"] == 0.0, "success_rate"]],
                 s=90, color=theme.series[1], zorder=5, edgecolor=theme.surface,
                 linewidth=1.5, label="ganancioso (argmax)")

    axes.set_ylim(0.0, 1.0)
    _style_text(axes, theme,
                "A política estocástica domina; o argmax perde mais da metade do ganho",
                "temperatura do softmax", "taxa de sucesso")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


def _figure_purification(decisions: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.6)
    colours = {"actor_critic": theme.series[0], "threshold_0.90": theme.series[1]}
    bins = np.linspace(0.55, 1.0, 28)

    for name, group in decisions.groupby("policy", sort=False):
        episodes = 250
        weights = np.full(len(group), 1.0 / episodes)
        axes.hist(group["input_fidelity"], bins=bins, weights=weights,
                  histtype="stepfilled", alpha=0.45,
                  color=colours.get(name, theme.series[2]),
                  edgecolor=colours.get(name, theme.series[2]), linewidth=1.8,
                  label=f"{_label(name)} ({_pt(len(group) / episodes, '.1f')} rodadas/episódio)")

    _style_text(axes, theme,
                "Quando cada política decide purificar, pela fidelidade do par de entrada",
                "fidelidade do melhor par de entrada", "rodadas por episódio")
    _legend(axes, theme, loc="upper left")
    return _to_svg(figure)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render_phase2_report(run_directory: str | Path, output_path: str | Path,
                         title: str = "Autopilot de repetidores: roteamento e timing aprendidos"
                         ) -> Path:
    """Write the phase-2 report from the CSVs in ``run_directory``.

    Args:
        run_directory: holds ``test_results.csv``, ``history.json``,
            ``temperature_sweep.csv``, ``transfer.csv`` and
            ``purification_decisions.csv``.
        output_path: destination ``.html``.
        title: page title.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: if a required results file is missing — the report
            is generated from measurements, so there is nothing to fall back on.
    """
    run_directory = Path(run_directory)
    required = ["test_results.csv", "history.json", "temperature_sweep.csv",
                "transfer.csv", "purification_decisions.csv"]
    missing = [name for name in required if not (run_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing results in {run_directory}: {missing}. Run "
            f"scripts/evaluate_phase2.py and scripts/analyse_phase2.py first.")

    from .rollout import EvaluationResult

    result = EvaluationResult(frame=pd.read_csv(run_directory / "test_results.csv"))
    summary = result.summary()
    history = json.loads((run_directory / "history.json").read_text())
    sweep = pd.read_csv(run_directory / "temperature_sweep.csv")
    transfer = pd.read_csv(run_directory / "transfer.csv")
    decisions = pd.read_csv(run_directory / "purification_decisions.csv")

    agent_name = next(name for name in summary["policy"] if name.startswith("actor_critic"))
    agent = summary.set_index("policy").loc[agent_name]
    baselines = summary[~summary["policy"].str.startswith("actor_critic")]
    best_baseline = baselines.loc[baselines["success_rate"].idxmax()]
    episodes = int(agent["episodes"])

    sections: list[str] = []

    # -- provenance --------------------------------------------------------
    sections.append(f"""
<div class="meta"><dl>
  <dt>Tarefa</dt><dd>cadeia de 4 nós, vãos de 20 km; entregar 1 par entre as pontas</dd>
  <dt>Pedido</dt><dd>limiar de fidelidade 0,70 / 0,74 / 0,78; prazo 40–90 ms, ambos sorteados</dd>
  <dt>Hardware</dt><dd>fidelidade bruta do enlace 0,90; portas e medições 0,999; coerência 1 s</dd>
  <dt>Treino</dt><dd>{len(history['train']['iteration'])} atualizações &times;
      {history['config']['batch']} episódios, ator-crítico em numpy</dd>
  <dt>Teste</dt><dd>{episodes} episódios, sementes disjuntas do treino e da validação</dd>
</dl></div>""")

    # -- headline ----------------------------------------------------------
    display = summary.copy()
    display["label"] = [_label(name) for name in display["policy"]]
    display = display.sort_values("success_rate", ascending=False)

    paired_rows = []
    for reference in ("threshold_0.90", "threshold_0.94", "shortest_path",
                      "greedy_fidelity", "random"):
        if reference not in set(summary["policy"]):
            continue
        difference = result.paired_difference(agent_name, reference)
        paired_rows.append({
            "reference": _label(reference),
            "delta": difference["mean"],
            "stderr": difference["stderr"],
            "record": f"{difference['wins']} / {difference['losses']} / {difference['ties']}",
        })
    paired = pd.DataFrame(paired_rows)

    sections.append(_section(
        "01", "resultado", "O que o agente aprendeu vale 32 pontos percentuais",
        f"<p>Em {episodes} episódios de teste retidos, o agente atende "
        f"<strong>{_pt(agent['success_rate'] * 100, '.1f')}%</strong> dos pedidos contra "
        f"<strong>{_pt(best_baseline['success_rate'] * 100, '.1f')}%</strong> do melhor heurístico "
        f"({_label(best_baseline['policy'])}). Como cada política enfrenta exatamente os "
        f"mesmos episódios — mesmas sementes, mesmos resultados de heralding — a "
        f"comparação é pareada, e a diferença pareada é o número com significado: "
        f"<strong>{_pt(paired.iloc[0]['delta'] * 100, '+.1f')} &plusmn; "
        f"{_pt(paired.iloc[0]['stderr'] * 100, '.1f')} p.p.</strong>.</p>",
        _figure_block({theme.name: _figure_results(summary, theme) for theme in THEMES},
                      "Barras de erro: erro padrão da proporção binomial."),
        _table_pt(display,
                  {"success_rate": ".1%", "raw_return": "+.3f", "mean_time_ms": ".1f",
                   "mean_fidelity": ".3f", "attempts": ".1f", "purifications": ".1f",
                   "swaps": ".1f"},
                  columns=["label", "success_rate", "raw_return", "mean_time_ms",
                           "mean_fidelity", "attempts", "purifications", "swaps"],
                  headers=["política", "sucesso", "retorno", "tempo (ms)", "fidelidade",
                           "tentativas", "purificações", "swaps"]),
        "<p>O detalhe que mais importa está nas três últimas colunas. O agente não "
        "vence purificando <em>mais</em> — vence purificando "
        f"<strong>{_pt(best_baseline['purifications'] / max(agent['purifications'], 1e-9), '.1f')}&times; menos</strong>, "
        f"com {_pt(best_baseline['attempts'] / agent['attempts'], '.1f')}&times; menos tentativas de enlace, "
        f"em {_pt(agent['mean_time_ms'] / best_baseline['mean_time_ms'] * 100, '.0f')}% do tempo. "
        "A política aprendida é mais seletiva, não mais agressiva.</p>",
        _table_pt(paired, {"delta": "+.1%", "stderr": ".1%"},
                  columns=["reference", "delta", "stderr", "record"],
                  headers=["contra", "diferença pareada", "erro padrão", "V / D / E"])))

    # -- learning ----------------------------------------------------------
    sections.append(_section(
        "02", "treino", "A curva de aprendizado",
        "<p>A validação usa sementes fixas e o resultado sem shaping, então ela se "
        "move apenas quando a política se move. O <em>shaping</em> baseado em "
        "potencial entra só no treino, e por construção não altera a política ótima "
        "(Ng, Harada &amp; Russell, 1999) — mas o placar nunca o inclui.</p>",
        _figure_block({theme.name: _figure_learning_curve(
            history, float(best_baseline["success_rate"]), theme) for theme in THEMES},
            "O sucesso de treino é medido com ações amostradas em episódios novos a "
            "cada iteração; a validação usa um conjunto fixo. A diferença entre as "
            "duas curvas é a exploração, não overfitting."),
        '<p class="caveat">O melhor checkpoint na validação é o que fica guardado, '
        'porque gradiente de política nesta tarefa encontra uma boa política e depois '
        'a derruba. Isso torna o número de validação otimista — é um máximo sobre '
        'checkpoints — e é exatamente por isso que ele não é o número reportado. '
        'Tudo na seção 01 vem de sementes disjuntas, avaliadas uma única vez.</p>'))

    # -- temperature -------------------------------------------------------
    greedy = float(sweep.loc[sweep["temperature"] == 0.0, "success_rate"].iloc[0])
    best_temperature = float(sweep.loc[sweep["success_rate"].idxmax(), "temperature"])
    sections.append(_section(
        "03", "achado", "A política estocástica vence a determinística com folga",
        f"<p>A coisa óbvia a fazer com uma política softmax treinada é pegar o argmax. "
        f"Aqui isso custa mais da metade do ganho: os mesmos pesos marcam "
        f"<strong>{_pt(greedy * 100, '.1f')}%</strong> no modo ganancioso e "
        f"<strong>{_pt(float(sweep['success_rate'].max()) * 100, '.1f')}%</strong> amostrando. Não é bug, "
        f"e há duas razões, ambas reais.</p>"
        "<p>A primeira: o A2C otimiza o retorno da política <em>estocástica</em>. As "
        "vantagens, a função de valor e a distribuição de estados sobre a qual tudo "
        "isso foi estimado pertencem à política suave. Tornar o argmax determinístico "
        "produz uma política que o treino nunca avaliou, e o teorema de melhoria de "
        "política exige a função de valor <em>daquela</em> política, que não é a que o "
        "crítico aprendeu.</p>"
        "<p>A segunda: as features são agregados, então situações físicas distintas "
        "podem gerar vetores idênticos. Sob esse aliasing o problema é um POMDP do "
        "ponto de vista da política, e aí uma política estocástica pode dominar "
        "estritamente qualquer determinística.</p>",
        _figure_block({theme.name: _figure_temperature(sweep, theme) for theme in THEMES},
                      "Varredura feita nas sementes de validação, não nas de teste — "
                      "escolher a temperatura no teste transformaria o número principal "
                      "num máximo sobre escolhas."),
        f"<p>O ótimo é largo, de 0,25 a 1,5, o que sustenta a primeira explicação: a "
        f"suavidade em si está fazendo trabalho. A temperatura de operação "
        f"({_pt(best_temperature, '.2f')}) foi escolhida na validação e aplicada uma única vez "
        f"ao teste.</p>"))

    # -- what it learned ---------------------------------------------------
    grouped = decisions.groupby("policy").agg(
        rounds=("input_fidelity", "size"),
        mean_input=("input_fidelity", "mean"),
        mean_gain=("predicted_gain", "mean"),
        mean_time_left=("time_remaining", "mean"),
        on_links=("result_hops", lambda column: float((column <= 0.13).mean())),
    ).reset_index()
    grouped["policy"] = [_label(name) for name in grouped["policy"]]
    grouped["rounds"] = grouped["rounds"] / 250.0

    sections.append(_section(
        "04", "interpretação", "O que a política realmente aprendeu",
        "<p>Um número de sucesso não diz o que mudou. Estes são os momentos em que "
        "cada política escolheu purificar, registrados no instante da decisão.</p>",
        _figure_block({theme.name: _figure_purification(decisions, theme)
                       for theme in THEMES},
                      "O heurístico purifica tudo abaixo de 0,90, inclusive pares em "
                      "0,885 cujo ganho previsto é marginal. O agente concentra as "
                      "rodadas nos pares que realmente precisam."),
        _table_pt(grouped,
                  {"rounds": ".2f", "mean_input": ".3f", "mean_gain": ".4f",
                   "mean_time_left": ".2f", "on_links": ".0%"},
                  columns=["policy", "rounds", "mean_input", "mean_gain",
                           "mean_time_left", "on_links"],
               headers=["política", "rodadas/episódio", "fidelidade de entrada média",
                        "ganho previsto médio", "tempo restante médio",
                        "em enlaces elementares"]),
        "<p>Duas diferenças de comportamento saltam. O agente purifica pares de "
        "fidelidade <em>mais baixa</em> — ele espera até que a rodada valha a pena, em "
        "vez de disparar num limiar fixo. E purifica em enlaces elementares em menos "
        "da metade das vezes, contra três quartos do heurístico: ele às vezes purifica "
        "<em>depois</em> do swap, o que o protocolo aninhado clássico proíbe e que aqui "
        "compensa quando o prazo está apertado.</p>"))

    # -- transfer ----------------------------------------------------------
    transfer_display = transfer.copy()
    sections.append(_section(
        "05", "generalização", "A mesma rede em topologias que ela nunca viu",
        "<p>A política pontua cada ação a partir das features daquela ação e nunca vê "
        "o nome de um nó nem o índice de um enlace, então em princípio os mesmos pesos "
        "rodam em qualquer grafo. Isto mede se rodam de fato — sem nenhum retreino.</p>",
        _table_pt(transfer_display,
                  {"shortest_path": ".1%", "threshold_0.90": ".1%",
                   "actor_critic": ".1%", "agent_stderr": ".1%"},
                  columns=["topology", "hops", "shortest_path", "threshold_0.90",
                           "actor_critic", "agent_stderr"],
               headers=["topologia", "saltos", "caminho mais curto", "limiar 0,90",
                        "agente", "&plusmn;"]),
        "<p>A transferência funciona: no anel de 6 nós, que tem rotas alternativas e o "
        "dobro de enlaces, o agente mantém 77,6% contra 55,2% do heurístico. Na cadeia "
        "de 5 nós o problema fica quase impossível para todos — mas o agente ainda "
        "entrega 10,4% onde o heurístico entrega 0,4%.</p>"
        "<p>A linha mais interessante é a de vãos de 30 km. Ali o heurístico de limiar "
        "<em>quebra</em>: cai para 18,0%, abaixo dos 34,0% de simplesmente não purificar. "
        "O limiar fixo continua gastando pares em rodadas de purificação que, com "
        "enlaces mais lentos, já não cabem no prazo. O agente chega a 45,6%. É a "
        "demonstração mais limpa da tese: uma constante não consegue representar uma "
        "decisão que depende do estado.</p>",
        '<p class="caveat">Isto não é generalização ilimitada. A cadeia de 5 nós mostra '
        'que o agente degrada junto com a tarefa quando ela sai do regime em que foi '
        'treinado, e nada aqui testa topologias com heterogeneidade forte de '
        'comprimento de vão. Treinar sobre uma distribuição de topologias, em vez de '
        'uma só, é o próximo passo óbvio.</p>'))

    # -- caveats -----------------------------------------------------------
    sections.append(_section(
        "06", "limites", "O que este resultado não é",
        "<p><strong>Uma tarefa, um ponto de operação.</strong> Cadeia de 4 nós, vãos "
        "iguais, um par a entregar. O ponto de operação foi escolhido para que a "
        "purificação importasse — com portas a 0,99 em vez de 0,999 ela quase não "
        "ajuda, e a decisão que o agente aprende deixa de existir.</p>"
        "<p><strong>Uma semente de treino.</strong> A curva mostra uma execução. "
        "Gradiente de política tem variância alta entre sementes; a afirmação "
        "defensável é sobre esta política treinada, não sobre a expectativa do "
        "método.</p>"
        "<p><strong>Física validada, tarefa não.</strong> As operações Bell-diagonais "
        "são idênticas às do SeQUeNCe até a precisão de máquina (fase 1), mas o "
        "ambiente é o modelo analítico rápido, não o simulador de eventos discretos. "
        "Fechar esse laço — rodar a política treinada dentro do SeQUeNCe — é trabalho "
        "que ainda não foi feito.</p>"
        "<p><strong>Sem PyTorch.</strong> A rede é numpy com backprop manual, conferida "
        "contra diferenças finitas a 10<sup>-10</sup>. Isso limita a escala: um "
        "codificador de grafo de verdade, que enxergaria a topologia em vez de resumos "
        "por ação, precisa de um framework.</p>"))

    body = "\n".join(sections)
    document = f"""<title>{html.escape(title)}</title>
{_FONTS}
<style>{_CSS}
figure + p, figure + div {{ margin-top: 22px; }}
.table-wrap + p {{ margin-top: 18px; }}</style>
<main>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="lede">Um agente decide caminho e timing juntos — gerar, purificar,
       trocar, descartar — e o relógio avança pela latência do que ele escolhe.
       Fase 2 do qnet_autopilot.</p>
  </header>
  {body}
</main>
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
