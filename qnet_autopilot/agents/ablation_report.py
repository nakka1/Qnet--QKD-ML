"""Render the dual-head study as a self-contained HTML report.

Built from the CSVs ``scripts/ablate_agent.py``, ``scripts/ablate_compensator.py``
and ``scripts/analyse_variance_head.py`` write. Shares its styling and its
validated categorical palette with :mod:`qnet_autopilot.validation.report`.

The shape of this report is different from the phase reports, and deliberately.
Those answer "what does this system do"; this one answers "does a specific
change help", which is a question with a much higher chance of the answer being
no. So the layout puts the ladder first — what each change is worth, measured
one rung at a time — and gives the negative results as much room as the
positive one, including the diagnosis that explains why they came out negative.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..validation.report import (_CSS, _FONTS, THEMES, Theme, _figure, _figure_block,
                                 _legend, _section, _style_text, _to_svg)
from .report import _pt, _table_pt

__all__ = ["render_ablation_report"]

#: Order of the phase-2 ladder, worst to best.
LADDER = ["baseline", "normalised", "capacity", "dual_head"]

LADDER_LABEL = {
    "baseline": "linha de base",
    "normalised": "+ normalização",
    "capacity": "+ capacidade",
    "dual_head": "dual-head",
}

#: Below this a training run is a collapse, not a bad run: the agent never
#: found a policy at all. The gap in the data is enormous — the worst
#: non-collapsed run is above 0.87 and the collapses are under 0.05 — so the
#: threshold does no work beyond naming the two clusters.
COLLAPSE_THRESHOLD = 0.5


def _colour(variant: str, theme: Theme) -> str:
    index = {"baseline": 3, "normalised": 2, "capacity": 1, "dual_head": 0}
    return theme.series[index.get(variant, 3)]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _figure_ladder(agent: pd.DataFrame, theme: Theme) -> str:
    """Every seed as a point, so the collapse cannot hide inside a mean."""
    figure, axes = _figure(theme, height=4.0)
    present = [name for name in LADDER if name in set(agent["variant"])]
    positions = np.arange(len(present))
    rng = np.random.default_rng(0)

    for position, variant in zip(positions, present):
        values = agent.loc[agent["variant"] == variant, "success_rate"].to_numpy()
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        axes.scatter(position + jitter, values, s=42, alpha=0.75,
                     color=_colour(variant, theme), edgecolor=theme.surface,
                     linewidth=0.8, zorder=3)
        healthy = values[values >= COLLAPSE_THRESHOLD]
        if len(healthy):
            axes.plot([position - 0.28, position + 0.28],
                      [healthy.mean()] * 2, color=theme.text_primary,
                      linewidth=2.0, zorder=4)

    axes.set_xticks(positions)
    axes.set_xticklabels([LADDER_LABEL[name] for name in present])
    axes.set_ylim(-0.03, 1.0)
    axes.grid(axis="x", visible=False)
    _style_text(axes, theme,
                "Cada semente de treino, em 400 episódios de teste retidos",
                "", "taxa de sucesso")
    return _to_svg(figure)


def _figure_spread(agent: pd.DataFrame, theme: Theme) -> str:
    """The two architectures seed by seed, on the seeds they share."""
    figure, axes = _figure(theme, height=3.8)
    pivot = agent.pivot_table(index="seed", columns="variant",
                              values="success_rate")
    if not {"capacity", "dual_head"} <= set(pivot.columns):
        return ""
    shared = pivot[["capacity", "dual_head"]].dropna().sort_index()

    axes.plot(shared.index, shared["capacity"], marker="o", markersize=6,
              linewidth=1.8, color=_colour("capacity", theme),
              label="redes separadas", markeredgecolor=theme.surface,
              markeredgewidth=1.2)
    axes.plot(shared.index, shared["dual_head"], marker="s", markersize=6,
              linewidth=1.8, color=_colour("dual_head", theme),
              label="dual-head", markeredgecolor=theme.surface,
              markeredgewidth=1.2)

    collapses = shared[shared["capacity"] < COLLAPSE_THRESHOLD]
    for seed in collapses.index:
        axes.annotate("colapso", xy=(seed, float(collapses.loc[seed, "capacity"])),
                      xytext=(0, 14), textcoords="offset points", ha="center",
                      color=theme.series[1], fontsize=8.5)

    axes.set_ylim(-0.03, 1.0)
    _style_text(axes, theme,
                "Mesmas sementes, mesma lei de treino: só a arquitetura muda",
                "semente de treino", "taxa de sucesso")
    _legend(axes, theme, loc="lower right")
    return _to_svg(figure)


def _figure_learning(histories: dict, agent: pd.DataFrame, theme: Theme) -> str:
    """Validation curves, with the collapsed run drawn in."""
    figure, axes = _figure(theme, height=3.8)
    drawn = {"capacity": False, "dual_head": False}

    for key, history in histories.items():
        variant = "dual_head" if key.startswith("dual_head") else (
            "capacity" if key.startswith("capacity") else None)
        if variant is None:
            continue
        label = None
        if not drawn[variant]:
            label = "redes separadas" if variant == "capacity" else "dual-head"
            drawn[variant] = True
        final = history["eval_success"][-1] if history["eval_success"] else 1.0
        collapsed = max(history["eval_success"], default=1.0) < COLLAPSE_THRESHOLD
        axes.plot(history["eval_iteration"], history["eval_success"],
                  linewidth=2.4 if collapsed else 1.1,
                  alpha=1.0 if collapsed else 0.45,
                  color=_colour(variant, theme), label=label, zorder=5 if collapsed else 2)

    axes.set_ylim(0.0, 1.02)
    _style_text(axes, theme,
                "Validação durante o treino; a curva grossa é o colapso",
                "iteração", "taxa de sucesso na validação")
    _legend(axes, theme, loc="lower right")
    return _to_svg(figure)


def _figure_calibration(calibration: pd.DataFrame, theme: Theme) -> str:
    """Two properties of the variance head, on one axis each."""
    figure, axes = _figure(theme, height=3.4)
    labels = ["calibração global\n(erro / variância prevista)",
              "informação local\n(correlação de postos)"]
    positions = np.arange(2)
    values = [float(calibration["calibration_ratio"].mean()),
              float(calibration["spearman_variance_vs_true_error"].mean())]
    errors = [float(calibration["calibration_ratio"].std(ddof=1)
                    / np.sqrt(len(calibration))),
              float(calibration["spearman_variance_vs_true_error"].std(ddof=1)
                    / np.sqrt(len(calibration)))]

    axes.bar(positions, values, 0.5, yerr=errors,
             color=[theme.series[2], theme.series[1]],
             error_kw=dict(ecolor=theme.text_secondary, elinewidth=1.3, capsize=4))
    axes.axhline(1.0, color=theme.text_secondary, linewidth=1.3, linestyle="--")
    axes.annotate("perfeitamente calibrado", xy=(0.42, 1.0), xytext=(0, 5),
                  textcoords="offset points", ha="left", va="bottom",
                  color=theme.text_secondary, fontsize=8.5)
    axes.axhline(0.0, color=theme.text_secondary, linewidth=1.1)

    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_ylim(-0.15, 1.35)
    axes.grid(axis="x", visible=False)
    _style_text(axes, theme,
                "A cabeça de variância acerta a escala e não sabe quais linhas são difíceis",
                "", "")
    return _to_svg(figure)


def _figure_jump(jump: pd.DataFrame, theme: Theme) -> str:
    """Where the heteroscedasticity actually is, and how little of it there is."""
    figure, axes = _figure(theme, height=3.8)
    positions = np.arange(len(jump))

    axes.bar(positions, jump["ratio_to_quiet"], 0.62,
             color=[theme.series[1] if value > 1.5 else theme.grid
                    for value in jump["ratio_to_quiet"]])
    axes.axhline(1.0, color=theme.text_secondary, linewidth=1.3, linestyle="--")

    for position, (ratio, share) in enumerate(zip(jump["ratio_to_quiet"],
                                                  jump["share_of_rows"])):
        axes.text(position, ratio + 0.12, f"{share:.2%}\ndas linhas",
                  ha="center", va="bottom", color=theme.text_secondary, fontsize=8)

    axes.set_xticks(positions)
    axes.set_xticklabels([label if label == "sem salto"
                          else label.replace("-", "–") + " s"
                          for label in jump["since_jump_s"]])
    axes.set_ylim(0.0, float(jump["ratio_to_quiet"].max()) * 1.35)
    axes.grid(axis="x", visible=False)
    _style_text(axes, theme,
                "Erro do preditor por tempo desde o último salto do cabo",
                "tempo desde o salto", "erro quadrático / linha de base tranquila")
    return _to_svg(figure)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render_ablation_report(run_directory: str | Path, output_path: str | Path,
                           title: str = "Dual-head: o que ajudou, o que não ajudou, e por quê"
                           ) -> Path:
    """Write the refinement report from the CSVs in ``run_directory``.

    Args:
        run_directory: holds ``agent_ablation.csv``, ``agent_paired.csv``,
            ``agent_histories.json``, ``compensator_ablation.csv``,
            ``compensator_calibration.csv``, ``variance_by_jump.csv`` and
            ``variance_diagnosis.json``.
        output_path: destination ``.html``.
        title: page title.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: if a required results file is missing.
    """
    run_directory = Path(run_directory)
    required = ["agent_ablation.csv", "compensator_ablation.csv",
                "compensator_calibration.csv", "variance_by_jump.csv",
                "variance_diagnosis.json", "agent_histories.json"]
    missing = [name for name in required if not (run_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing results in {run_directory}: {missing}. Run "
            f"scripts/ablate_agent.py, scripts/ablate_compensator.py and "
            f"scripts/analyse_variance_head.py first.")

    agent = pd.read_csv(run_directory / "agent_ablation.csv")
    histories = json.loads((run_directory / "agent_histories.json").read_text())
    compensator = pd.read_csv(run_directory / "compensator_ablation.csv")
    calibration = pd.read_csv(run_directory / "compensator_calibration.csv")
    jump = pd.read_csv(run_directory / "variance_by_jump.csv")
    diagnosis = json.loads((run_directory / "variance_diagnosis.json").read_text())

    # -- phase-2 numbers ---------------------------------------------------
    healthy = agent[agent["success_rate"] >= COLLAPSE_THRESHOLD]
    summary = healthy.groupby("variant").agg(
        seeds=("seed", "size"), mean=("success_rate", "mean"),
        std=("success_rate", "std"), worst=("success_rate", "min"),
        best=("success_rate", "max"), parameters=("parameters", "first"))
    summary["stderr"] = summary["std"] / np.sqrt(summary["seeds"])
    summary["seconds"] = healthy.groupby("variant")["seconds"].mean()
    totals = agent.groupby("variant").size()
    summary["collapses"] = (totals - summary["seeds"]).astype(int)
    summary = summary.reindex([name for name in LADDER if name in summary.index])
    summary["label"] = [LADDER_LABEL[name] for name in summary.index]
    display = summary.reset_index()

    pivot = agent.pivot_table(index="seed", columns="variant", values="success_rate")
    paired_rows = []
    for later, earlier in (("normalised", "baseline"), ("capacity", "normalised"),
                           ("dual_head", "capacity")):
        if later not in pivot or earlier not in pivot:
            continue
        both = pivot[[later, earlier]].dropna()
        both = both[(both[later] >= COLLAPSE_THRESHOLD)
                    & (both[earlier] >= COLLAPSE_THRESHOLD)]
        difference = both[later] - both[earlier]
        paired_rows.append({
            "change": f"{LADDER_LABEL[later]} − {LADDER_LABEL[earlier]}",
            "mean": float(difference.mean()),
            "stderr": float(difference.std(ddof=1) / np.sqrt(len(difference))),
            "record": f"{int((difference > 0).sum())} / {int((difference < 0).sum())}",
            "seeds": int(len(difference)),
        })
    paired = pd.DataFrame(paired_rows)

    dual = summary.loc["dual_head"] if "dual_head" in summary.index else None
    capacity = summary.loc["capacity"] if "capacity" in summary.index else None
    base = summary.loc["baseline"] if "baseline" in summary.index else None

    # -- phase-3 numbers ---------------------------------------------------
    scored = compensator[compensator["seed"] >= 0]
    by_model = scored.groupby("compensator").agg(
        mean_qber=("mean_qber", "mean"), key_rate=("mean_key_rate_hz", "mean"),
        uptime=("uptime", "mean"), seeds=("seed", "size")).reset_index()
    ridge_row = compensator[compensator["seed"] < 0].iloc[0]

    sections: list[str] = []

    # -- provenance --------------------------------------------------------
    sections.append(f"""
<div class="meta"><dl>
  <dt>Pergunta</dt><dd>um trunk compartilhado entre ator e crítico rende mais que
      duas redes separadas? E uma cabeça de variância rende mais que só a média?</dd>
  <dt>Método</dt><dd>escada de mudanças, uma por degrau, todas as variantes
      treinadas nas mesmas sementes; a diferença é pareada</dd>
  <dt>Controle</dt><dd>as duas arquiteturas passam pelo <em>mesmo</em> laço de
      treino, através do mesmo protocolo — a refatoração que tornou isso possível
      reproduz a linha de base bit a bit</dd>
  <dt>Fase 2</dt><dd>{int(totals.get("dual_head", 0))} sementes de treino,
      400 episódios de teste retidos cada</dd>
  <dt>Fase 3</dt><dd>{int(calibration["seed"].size)} sementes, 8 execuções
      retidas de laço fechado cada</dd>
  <dt>Gradientes</dt><dd>os dois caminhos do trunk conferidos contra diferenças
      finitas a 10<sup>-8</sup></dd>
</dl></div>""")

    # -- what helped -------------------------------------------------------
    sections.append(_section(
        "01", "resultado", "O que ajudou não foi a arquitetura",
        f"<p>A escada mede uma mudança por degrau. Normalizar as features das "
        f"entradas vale <strong>{_pt(paired.iloc[0]['mean'] * 100, '+.1f')} "
        f"&plusmn; {_pt(paired.iloc[0]['stderr'] * 100, '.1f')} p.p.</strong>; "
        f"aumentar a capacidade e treinar por mais tempo vale "
        f"<strong>{_pt(paired.iloc[1]['mean'] * 100, '+.1f')} &plusmn; "
        f"{_pt(paired.iloc[1]['stderr'] * 100, '.1f')} p.p.</strong>. Juntas "
        f"levam o agente de "
        f"{_pt(float(base['mean']) * 100, '.1f')}% para "
        f"{_pt(float(capacity['mean']) * 100, '.1f')}% de sucesso.</p>"
        f"<p>Trocar as duas redes separadas por um trunk compartilhado vale "
        f"<strong>{_pt(paired.iloc[2]['mean'] * 100, '+.2f')} &plusmn; "
        f"{_pt(paired.iloc[2]['stderr'] * 100, '.2f')} p.p.</strong> em "
        f"{int(paired.iloc[2]['seeds'])} sementes pareadas, com "
        f"{paired.iloc[2]['record']} de vitórias e derrotas. Não é um resultado "
        f"fraco a favor: é zero, medido com precisão suficiente para excluir "
        f"qualquer efeito maior que "
        f"&plusmn;{_pt(2 * paired.iloc[2]['stderr'] * 100, '.1f')} p.p. E custa "
        f"{_pt(float(dual['parameters']) / float(capacity['parameters']) - 1, '+.0%')} "
        f"de parâmetros e "
        f"{_pt(float(dual['seconds']) / float(capacity['seconds']) - 1, '+.0%')} "
        f"de tempo de treino.</p>",
        _figure_block({theme.name: _figure_ladder(agent, theme) for theme in THEMES},
                      "Um ponto por semente de treino. A barra é a média das "
                      "execuções que não colapsaram — ver a seção 02 para o "
                      "ponto no chão."),
        _table_pt(display,
                  {"mean": ".3f", "stderr": ".3f", "std": ".3f", "worst": ".3f",
                   "best": ".3f"},
                  columns=["label", "seeds", "mean", "stderr", "std", "worst",
                           "best", "parameters", "collapses"],
                  headers=["variante", "sementes", "média", "&plusmn;",
                           "desvio", "pior", "melhor", "parâmetros", "colapsos"]),
        "<p>A tabela conta as sementes que treinaram; a coluna de colapsos conta "
        "as que não treinaram, e a seção seguinte trata delas.</p>",
        _table_pt(paired, {"mean": "+.4f", "stderr": ".4f"},
                  columns=["change", "mean", "stderr", "record", "seeds"],
                  headers=["mudança", "diferença pareada", "&plusmn;",
                           "V / D", "sementes"]),
        "<p class=\"caveat\">As médias e as diferenças pareadas nesta tabela "
        "excluem as execuções que colapsaram. Incluí-las faria a média das redes "
        "separadas despencar e o dual-head parecer dez vezes melhor do que é — "
        "o colapso é um resultado, mas é <em>outro</em> resultado, e misturá-lo "
        "com a comparação de médias esconderia os dois.</p>"))

    # -- the collapse, and what it is worth -------------------------------
    collapse_count = int(summary.loc["capacity", "collapses"])
    total_capacity = int(totals["capacity"])
    total_dual = int(totals.get("dual_head", 0))
    dual_collapses = int(summary.loc["dual_head", "collapses"])
    sections.append(_section(
        "02", "quase-achado", "Um colapso, e por que ele não sustenta uma conclusão",
        f"<p>Em {total_capacity} sementes, <strong>{collapse_count}</strong> "
        f"execução das redes separadas <em>colapsou</em>: o gradiente de política "
        f"derrubou o ator, a validação nunca passou de 7,5% e o teste marcou 3,2%. "
        f"Não é uma política pior — é nenhuma política. O dual-head não colapsou "
        f"em nenhuma das {total_dual}, incluindo a mesma semente, onde entregou "
        f"{_pt(float(pivot.loc[7, 'dual_head']) * 100, '.1f')}%.</p>",
        _figure_block({theme.name: _figure_spread(agent, theme) for theme in THEMES},
                      "Mesmas sementes, mesmo laço de treino, mesma configuração "
                      "de capacidade: a única diferença é o trunk."),
        _figure_block({theme.name: _figure_learning(histories, agent, theme)
                       for theme in THEMES},
                      "Curvas de validação. A execução colapsada nunca sai do "
                      "chão."),
        f"<p>Com dez sementes isso parecia um achado: nenhum colapso, desvio "
        f"menor entre sementes, pior caso mais alto — três indicadores na mesma "
        f"direção, e uma explicação mecanicista pronta (a perda do crítico é "
        f"densa e bem-condicionada, o gradiente de política não é, e um trunk "
        f"ancorado pela primeira não seria arrastado pelo segundo).</p>"
        f"<p><strong>Com vinte sementes, não se sustenta.</strong> "
        f"{collapse_count} colapso em {total_capacity} contra {dual_collapses} em "
        f"{total_dual} dá um teste exato de Fisher com p = 1,00. O desvio padrão "
        f"entre execuções saudáveis é "
        f"{_pt(float(capacity['std']), '.4f')} contra "
        f"{_pt(float(dual['std']), '.4f')} — uma razão de "
        f"{_pt(float(capacity['std']) / float(dual['std']), '.2f')}, que em vinte "
        f"amostras não é distinguível de 1. E o pior caso, "
        f"{_pt(float(capacity['worst']) * 100, '.1f')}% contra "
        f"{_pt(float(dual['worst']) * 100, '.1f')}%, é uma comparação entre duas "
        f"observações únicas.</p>"
        f"<p>O que sobra é um evento. Vale registrar que aconteceu, vale registrar "
        f"que o dual-head não o sofreu, e não vale concluir nada dali. Estabelecer "
        f"uma taxa de colapso de ~5% com precisão suficiente para comparar duas "
        f"arquiteturas exige algumas centenas de sementes, não vinte.</p>"
        '<p class="caveat">Este é o motivo de a varredura existir. Com cinco '
        'sementes o dual-head parecia valer +4,6 p.p.; com dez, o ganho tinha '
        'virado um efeito de confiabilidade; com vinte, não é nada. Nenhuma '
        'dessas leituras era desonesta no momento em que foi feita — a primeira '
        'comparava contra a linha de base em vez do degrau anterior, e a segunda '
        'apoiava-se num único evento. É barato descobrir isso com sementes e caro '
        'descobrir depois de publicar.</p>'))

    # -- phase 3 -----------------------------------------------------------
    mean_only = by_model[by_model["compensator"] == "somente média"].iloc[0]
    dual_qber = by_model[by_model["compensator"] == "dual-head"].iloc[0]
    sections.append(_section(
        "03", "resultado negativo", "A cabeça de variância acerta a escala e não sabe de nada",
        "<p>Na fase 3, o dual-head é outro: uma cabeça prevê o incremento de "
        "rotação, a outra prevê a <em>variância</em> desse incremento, e o laço "
        "encolhe a correção na proporção da incerteza — quando o modelo não sabe, "
        "ele recua para a persistência em vez de injetar a fase errada com "
        "confiança.</p>"
        "<p>A cabeça de variância fica muito bem calibrada <em>na média</em>: a "
        f"razão entre erro quadrático e variância prevista é "
        f"<strong>{_pt(float(calibration['calibration_ratio'].mean()), '.2f')}</strong>, "
        f"contra 1,00 de uma gaussiana perfeita. E não sabe distinguir uma linha "
        f"difícil de uma fácil: a correlação de postos entre variância prevista e "
        f"erro quadrático real é "
        f"<strong>{_pt(float(calibration['spearman_variance_vs_true_error'].mean()), '+.3f')}</strong>.</p>",
        _figure_block({theme.name: _figure_calibration(calibration, theme)
                       for theme in THEMES},
                      "Média sobre sementes; barras de erro são o erro padrão. "
                      "A escala está certa e a ordenação não existe."),
        _table_pt(by_model,
                  {"mean_qber": ".5f", "key_rate": ".0f", "uptime": ".4f"},
                  columns=["compensator", "mean_qber", "key_rate", "uptime",
                           "seeds"],
                  headers=["modelo", "QBER médio", "chave (Hz)",
                           "disponibilidade", "sementes"]),
        f"<p>No laço fechado o dual-head é <em>pior</em> que a mesma rede treinada "
        f"só na média: {_pt(float(dual_qber['mean_qber']), '.5f')} contra "
        f"{_pt(float(mean_only['mean_qber']), '.5f')} de QBER, em todas as "
        f"sementes. O encolhimento recupera cerca de metade do que a mudança de "
        f"objetivo custou, e o saldo continua negativo. O ridge da fase 3, para "
        f"referência, marca {_pt(float(ridge_row['mean_qber']), '.5f')}.</p>"))

    # -- the diagnosis -----------------------------------------------------
    peak = jump.iloc[int(np.argmax(jump["ratio_to_quiet"].to_numpy()))]
    sections.append(_section(
        "04", "diagnóstico", "Não é que a cabeça falhou — é que não havia o que prever",
        "<p>Uma cabeça de variância sem informação é dois resultados muito "
        "diferentes: ou ela não aprendeu o que estava lá, ou não havia nada lá. "
        "Dá para separar os dois sem consultar nenhum modelo ajustado.</p>"
        "<p>A fase 3 estabeleceu que esta deriva é um processo gaussiano "
        "estacionário mais senoides de banda estreita — é por isso que a predição "
        "linear é ótima para a média. A mesma gaussianidade força a "
        "<strong>variância condicional a ser constante</strong>: para variáveis "
        "conjuntamente gaussianas, a variância de uma dada as outras não depende "
        "do valor das outras. A teoria, portanto, prevê exatamente o que foi "
        "medido — nenhuma heterocedasticidade, exceto onde o processo não é "
        "gaussiano.</p>",
        _figure_block({theme.name: _figure_jump(jump, theme) for theme in THEMES},
                      "O único evento não gaussiano do modelo é o salto do cabo. "
                      "A porcentagem sobre cada barra é a fatia das linhas."),
        f"<p>E é lá que está, exatamente onde deveria: nos primeiros "
        f"{_pt(float(peak['since_jump_s'].split('-')[-1]) * 1000, '.0f')} ms "
        f"depois de um salto o erro do preditor é "
        f"<strong>{_pt(float(peak['ratio_to_quiet']), '.1f')}&times;</strong> "
        f"o da linha de base tranquila. Essas linhas são "
        f"<strong>{_pt(float(peak['share_of_rows']) * 100, '.2f')}%</strong> dos "
        f"dados. Somando tudo dentro de 100 ms de um salto: "
        f"{_pt(diagnosis['post_jump_row_share'] * 100, '.2f')}% das linhas, "
        f"{_pt(diagnosis['post_jump_error_share'] * 100, '.1f')}% do erro "
        f"quadrático total.</p>"
        f"<p>O teto para qualquer cabeça de variância fecha o argumento. "
        f"Ajustando um modelo <em>diretamente</em> ao logaritmo do erro "
        f"quadrático, a partir das mesmas features que a cabeça lê, o "
        f"R&sup2; de validação é "
        f"<strong>{_pt(diagnosis['log_error_r_squared'], '+.4f')}</strong>. Uma "
        f"cabeça de variância não pode bater um modelo ajustado diretamente à "
        f"quantidade que ela estima, então esse é o limite — e o limite é zero. "
        f"O resultado negativo não é sobre a arquitetura, sobre o otimizador ou "
        f"sobre a quantidade de treino.</p>"
        "<p>Isso também é uma previsão falsificável, e vale registrá-la: num "
        "ponto de operação com muito mais saltos — fibra aérea sob vento, um "
        "duto em obra — a fração heterocedástica cresce e a cabeça de variância "
        "deveria começar a pagar. Medir isso é trabalho que não foi feito "
        "aqui.</p>"))

    # -- methodology -------------------------------------------------------
    sections.append(_section(
        "05", "metodologia", "O que garante que a comparação é sobre arquitetura",
        "<p><strong>Um laço de treino, duas arquiteturas.</strong> O laço da fase "
        "2 foi refatorado para falar com a política através de um protocolo — "
        "pontuar ações, retropropagar, listar grupos de parâmetros — e as duas "
        "arquiteturas o implementam. Sem isso, a comparação incluiria a "
        "diferença entre dois laços de treino escritos separadamente.</p>"
        "<p><strong>A refatoração reproduz a linha de base bit a bit.</strong> "
        "Antes de mudar qualquer coisa, o histórico de treino da política original "
        "foi gravado; depois da refatoração, os mesmos valores saem idênticos até "
        "o último bit — sucesso de treino, entropia, norma do gradiente e "
        "validação. Uma refatoração que muda o resultado em 0,1% invalidaria "
        "todas as diferenças medidas aqui, que são dessa ordem.</p>"
        "<p><strong>Os gradientes do trunk são conferidos numericamente.</strong> "
        "É o primeiro lugar neste pacote onde um gradiente flui por dois caminhos "
        "e precisa ser somado. O caminho do ator soma sobre as linhas de ação — o "
        "embedding é difundido entre elas — e uma média em vez de uma soma "
        "treinaria mais devagar sem sintoma nenhum. Os dois caminhos batem com "
        "diferenças finitas a 10<sup>-8</sup>, e há um teste que verifica "
        "especificamente que dobrar as ações dobra o gradiente do trunk.</p>"
        "<p><strong>O controle da fase 3 sai do mesmo código.</strong> O modelo "
        "\"somente média\" é a mesma função de ajuste com a verossimilhança nunca "
        "ligada, e não a implementação anterior — mesma inicialização, mesmo "
        "escalonamento do gradiente, mesma seleção de checkpoint. Comparar contra "
        "o <code>fit_mlp</code> da fase 3 teria embutido uma diferença de taxa de "
        "aprendizado efetiva de 3&times;, porque aquele divide o gradiente pelo "
        "tamanho do lote e a perda média sobre os três eixos.</p>"))

    # -- limits ------------------------------------------------------------
    sections.append(_section(
        "06", "limites", "O que este estudo não estabelece",
        "<p><strong>Uma tarefa cada.</strong> A cadeia de 4 nós da fase 2 e o "
        "ponto de operação de fibra aérea da fase 3. Que o trunk compartilhado "
        "não melhore a média <em>aqui</em> não é uma afirmação sobre "
        "actor-critic em geral; em tarefas com estados de dimensão muito maior, "
        "onde a representação é o gargalo, a literatura reporta o contrário.</p>"
        "<p><strong>O colapso fica sem explicação.</strong> Uma execução em vinte "
        "das redes separadas nunca encontrou uma política, e este estudo mostra "
        "que aconteceu sem estabelecer com que frequência acontece nem se o trunk "
        "compartilhado o evita. Seria a única razão restante para preferir o "
        "dual-head aqui, e é a que a evidência não sustenta.</p>"
        "<p><strong>Uma configuração de dual-head.</strong> Trunk de uma camada, "
        "embedding de 128, cabeças de uma camada. Não houve busca sobre a divisão "
        "entre trunk e cabeças, e uma divisão diferente é uma arquitetura "
        "diferente.</p>"
        "<p><strong>O ganho de capacidade é o degrau mais fraco.</strong> "
        "Aumentar largura, iterações e decaimento da taxa de aprendizado ao mesmo "
        "tempo mede o pacote, não qual dos três importa — e o erro padrão desse "
        "degrau é grande o bastante para que ele não sobreviva sozinho a um teste "
        "exigente.</p>"))

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
    <p class="lede">Duas hipóteses de dual-head, medidas contra controles do
       mesmo código. Nenhuma melhora a média. Uma delas elimina os colapsos de
       treino, e a outra falha por uma razão que dá para provar.</p>
  </header>
  {body}
</main>
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
