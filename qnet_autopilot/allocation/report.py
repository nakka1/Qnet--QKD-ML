"""Render the phase-4 results as a self-contained HTML report.

Built from the CSVs ``scripts/evaluate_phase4.py`` writes and the forecast
scores ``scripts/train_allocator.py`` saves, so every number and figure comes
from a measured run. Shares its styling and its validated categorical palette
with :mod:`qnet_autopilot.validation.report`.

The presentation problem this phase has, and how it is handled
--------------------------------------------------------------
There are two objectives that trade against each other, so there is no single
best policy and a bar chart of "score" would be a chart of whatever weight was
chosen. Every comparison here is therefore drawn as a **frontier**: each policy
family's tuning knob traces a curve through (key served, classical served), and
one family is better than another only where its curve lies above the other's.
Where the curves do not overlap, that is said rather than papered over.
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

__all__ = ["render_phase4_report"]

#: Fixed colour per policy family, kept across every figure.
FAMILY_COLOUR = {
    "preditivo": 0,
    "limiar reativo": 1,
    "admissão estática": 2,
    "clássico primeiro": 3,
    "quântico primeiro": 3,
}

#: Fixed colour per forecaster, for the ablation figures.
FORECASTER_COLOUR = {"persistência": 2, "ridge": 0, "oráculo": 1}


def _colour(name: str, theme: Theme, table=FAMILY_COLOUR) -> str:
    return theme.series[table.get(name, 3)]


def _interpolate(frame: pd.DataFrame, target: float) -> float:
    """Classical throughput a frontier reaches at a given key-service level.

    Linear interpolation between the two bracketing measured points. Returns NaN
    outside the family's measured range rather than extrapolating: a policy that
    was never observed at a service level should not be credited with a number
    there, and refusing to guess is what keeps the frontier comparison honest.
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

def _figure_frontier(frontier: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=4.4)

    for family in ("admissão estática", "limiar reativo", "preditivo"):
        group = frontier[frontier["family"] == family].sort_values("served_qkd")
        if group.empty:
            continue
        axes.plot(group["served_classical"], group["served_qkd"], marker="o",
                  markersize=6, linewidth=2.2, color=_colour(family, theme),
                  label=family, markeredgecolor=theme.surface, markeredgewidth=1.4)

    for family, marker in (("clássico primeiro", "s"), ("quântico primeiro", "D")):
        point = frontier[frontier["family"] == family]
        if point.empty:
            continue
        axes.scatter(point["served_classical"], point["served_qkd"], s=110,
                     marker=marker, color=_colour(family, theme), zorder=5,
                     edgecolor=theme.surface, linewidth=1.5, label=family)

    axes.set_xlim(-0.03, 1.05)
    axes.set_ylim(0.0, 0.78)
    _style_text(axes, theme,
                "A fronteira: chave entregue no prazo contra vazão clássica",
                "fração do tráfego clássico entregue",
                "fração da chave QKD entregue no prazo")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


def _figure_admission_by_length(behaviour: pd.DataFrame, theme: Theme) -> str:
    """Admission per link, counting only the steps where the link owes key.

    Unconditional admission mixes the coexistence decision with "this link has
    nothing queued, so open it up", and the second is not a decision about
    coexistence at all — it is what every policy here does, including the
    trivial ones.
    """
    figure, axes = _figure(theme, height=3.8)
    column = ("mean_admission_when_owing" if "mean_admission_when_owing"
              in behaviour.columns else "mean_admission")
    for policy, group in behaviour.groupby("policy", sort=False):
        group = group.sort_values("length_km")
        axes.plot(group["length_km"], group[column], marker="o",
                  markersize=6, linewidth=2, label=policy,
                  color=_colour(policy, theme),
                  markeredgecolor=theme.surface, markeredgewidth=1.4)
    axes.set_ylim(-0.05, 1.08)
    _style_text(axes, theme,
                "Admissão aplicada quando o enlace deve chave, por comprimento do vão",
                "comprimento do vão (km)", "fração de admissão clássica")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


def _figure_ablation(ablation: pd.DataFrame, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.6)
    metrics = {"served_qkd": "chave QKD", "served_classical": "tráfego clássico"}
    positions = np.arange(len(metrics))
    width = 0.36

    for index, forecaster in enumerate(("ridge", "oráculo")):
        subset = ablation[ablation["policy"].str.startswith(forecaster)]
        means, errors = [], []
        for metric in metrics:
            rows = subset[subset["metric"] == metric]
            means.append(float(rows["mean"].mean()))
            errors.append(float(np.sqrt(np.mean(rows["stderr"] ** 2))))
        axes.bar(positions + (index - 0.5) * width, means, width * 0.9,
                 yerr=errors, label=forecaster,
                 color=_colour(forecaster, theme, FORECASTER_COLOUR),
                 error_kw=dict(ecolor=theme.text_secondary, elinewidth=1.3,
                               capsize=4))

    axes.axhline(0.0, color=theme.text_secondary, linewidth=1.3)
    axes.set_xticks(positions)
    axes.set_xticklabels(list(metrics.values()))
    axes.grid(axis="x", visible=False)
    _style_text(axes, theme,
                "Ganho sobre persistência, mesmo controlador, episódios pareados",
                "", "diferença pareada na fração entregue")
    _legend(axes, theme, loc="upper right")
    return _to_svg(figure)


def _figure_forecast_skill(scores: dict, theme: Theme) -> str:
    figure, axes = _figure(theme, height=3.6)
    ridge = np.asarray(scores["ridge_mse"])
    persistence = np.asarray(scores["persistence_mse"])
    steps = np.arange(1, len(ridge) + 1)

    axes.plot(steps, 100.0 * (1.0 - ridge / persistence), marker="o", markersize=6,
              linewidth=2.2, color=theme.series[0],
              markeredgecolor=theme.surface, markeredgewidth=1.4)
    axes.axhline(0.0, color=theme.text_secondary, linewidth=1.3, linestyle="--")
    _style_text(axes, theme,
                "O previsor bate a persistência, e cada vez mais longe no horizonte",
                "passos de controle à frente (5 s cada)",
                "redução do erro quadrático (%)")
    return _to_svg(figure)


def _figure_decisions(decisions: pd.DataFrame, theme: Theme) -> str:
    """Decisions the forecast changes, against the outcome it changes.

    Two scales on one figure because the whole point is the contrast between
    them: a solid line at eight percent and a dashed line pinned to zero. Drawn
    apart, each looks unremarkable; together they are the finding.
    """
    figure, axes = _figure(theme, height=4.2)
    grouped = (decisions.groupby(["latency_s", "forecaster"])
               .mean(numeric_only=True).reset_index())

    twin = axes.twinx()
    handles = []

    for forecaster, group in grouped.groupby("forecaster", sort=False):
        group = group.sort_values("latency_s")
        colour = _colour(forecaster, theme, FORECASTER_COLOUR)
        line, = axes.plot(group["latency_s"], 100.0 * group["decisions_changed"],
                          marker="o", markersize=6, linewidth=2.2, color=colour,
                          label=f"decisões alteradas — {forecaster}",
                          markeredgecolor=theme.surface, markeredgewidth=1.4)
        handles.append(line)

    # A band showing what "no difference" looks like at this sample size, so the
    # flat dashed lines read as a measured null rather than as an unplotted series.
    spread = float(decisions.groupby("forecaster")["delta_qkd"].std().max()) * 100.0
    twin.axhspan(-spread, spread, color=theme.grid, alpha=0.45, linewidth=0)
    twin.axhline(0.0, color=theme.text_secondary, linewidth=1.1)

    for forecaster, group in grouped.groupby("forecaster", sort=False):
        group = group.sort_values("latency_s")
        line, = twin.plot(group["latency_s"], 100.0 * group["delta_qkd"],
                          marker="s", markersize=5, linewidth=1.8, linestyle="--",
                          color=_colour(forecaster, theme, FORECASTER_COLOUR),
                          alpha=0.9, label=f"ganho de chave — {forecaster}")
        handles.append(line)

    twin.set_ylim(-1.2, 1.2)
    twin.set_ylabel("diferença na chave entregue (p.p.)",
                    color=theme.text_secondary, fontsize=9)
    twin.tick_params(colors=theme.text_secondary, labelsize=9)
    for side in ("top", "left", "bottom"):
        twin.spines[side].set_visible(False)
    twin.spines["right"].set_color(theme.grid)
    twin.grid(False)

    _style_text(axes, theme,
                "A previsão muda decisões (linha cheia) e não muda resultados (tracejada)",
                "latência de reconfiguração (s)", "decisões alteradas (%)")

    legend = axes.legend(handles=handles, frameon=False, fontsize=9,
                         loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
                         columnspacing=1.8, handlelength=1.8)
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)
    return _to_svg(figure)


def _figure_power(power: pd.DataFrame, theme: Theme) -> str:
    """Key service against the strength of the coupling.

    The two trivial policies are drawn with distinct strokes rather than
    distinct colours: they are the floor and the ceiling of the problem, not two
    more entries in the same family, and the palette's fourth colour is worth
    less here than telling them apart at a glance.
    """
    figure, axes = _figure(theme, height=3.8)

    styles = {
        "clássico primeiro": dict(color=theme.series[3], linestyle="-"),
        "quântico primeiro": dict(color=theme.text_secondary, linestyle=":"),
    }

    for policy, group in power.groupby("policy", sort=False):
        group = group.sort_values("power_mw")
        family = ("preditivo" if policy.startswith("preditivo")
                  else "admissão estática" if policy.startswith("admissão")
                  else policy)
        style = styles.get(family, dict(color=_colour(family, theme),
                                        linestyle="-"))
        axes.plot(group["power_mw"], group["served_qkd"], marker="o", markersize=6,
                  linewidth=2, label=policy, markeredgecolor=theme.surface,
                  markeredgewidth=1.4, **style)

    axes.set_xscale("log")
    axes.set_xticks([0.03, 0.1, 0.3, 1.0])
    axes.set_xticklabels(["0,03", "0,1", "0,3", "1,0"])
    axes.set_ylim(0.0, 0.82)
    _style_text(axes, theme,
                "Chave entregue contra a potência clássica lançada por enlace",
                "potência clássica em admissão total (mW)",
                "fração da chave QKD entregue")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render_phase4_report(run_directory: str | Path, output_path: str | Path,
                         title: str = "Alocação híbrida: o tráfego clássico é pago em chave"
                         ) -> Path:
    """Write the phase-4 report from the CSVs in ``run_directory``.

    Args:
        run_directory: holds ``frontier.csv``, ``ablation.csv``,
            ``latency_sweep.csv``, ``power_sweep.csv``, ``decisions.csv``,
            ``behaviour.csv`` and ``forecast_scores.json``.
        output_path: destination ``.html``.
        title: page title.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: if a required results file is missing — the report is
            generated from measurements, so there is nothing to fall back on.
    """
    run_directory = Path(run_directory)
    required = ["frontier.csv", "ablation.csv", "latency_sweep.csv",
                "power_sweep.csv", "decisions.csv", "behaviour.csv",
                "forecast_scores.json"]
    missing = [name for name in required if not (run_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing results in {run_directory}: {missing}. Run "
            f"scripts/train_allocator.py and scripts/evaluate_phase4.py first.")

    frontier = pd.read_csv(run_directory / "frontier.csv")
    ablation = pd.read_csv(run_directory / "ablation.csv")
    latency = pd.read_csv(run_directory / "latency_sweep.csv")
    power = pd.read_csv(run_directory / "power_sweep.csv")
    decisions = pd.read_csv(run_directory / "decisions.csv")
    behaviour = pd.read_csv(run_directory / "behaviour.csv")
    scores = json.loads((run_directory / "forecast_scores.json").read_text())

    quantum_first = frontier[frontier["family"] == "quântico primeiro"].iloc[0]
    classical_first = frontier[frontier["family"] == "clássico primeiro"].iloc[0]
    predictive = frontier[frontier["family"] == "preditivo"]
    reactive = frontier[frontier["family"] == "limiar reativo"]
    static = frontier[frontier["family"] == "admissão estática"]

    # Matched-service comparison. Two levels, chosen so the families being
    # compared actually reach them: a high one where the predictive policy and
    # the static baseline both have measured points, and a middle one inside the
    # reactive baseline's range as well. Comparing families at a service level
    # one of them never achieved would mean extrapolating its frontier, which is
    # exactly the kind of invented number this report is trying not to produce.
    families = {"preditivo": predictive, "limiar reativo": reactive,
                "admissão estática": static}
    targets = (0.55, 0.68)
    matched = pd.DataFrame([
        {"target": target,
         **{name: _interpolate(frame, target) for name, frame in families.items()}}
        for target in targets])

    best_predictive = predictive.loc[predictive["served_qkd"].idxmax()]
    middle = matched[matched["target"] == 0.55].iloc[0]
    high = matched[matched["target"] == 0.68].iloc[0]

    sections: list[str] = []

    # -- provenance --------------------------------------------------------
    sections.append(f"""
<div class="meta"><dl>
  <dt>Planta</dt><dd>corredor metropolitano SP–RJ: 9 nós, 11 vãos de 30 a 145 km</dd>
  <dt>Acoplamento</dt><dd>espalhamento Raman espontâneo do canal clássico
      co-propagante; −5 dBm por enlace em admissão total</dd>
  <dt>Deriva</dt><dd>resíduo do compensador da fase 3, calibrado no que ele
      mediu; um terço dos enlaces sorteado como degradado a cada episódio</dd>
  <dt>Decisão</dt><dd>fração de admissão clássica por enlace, a cada 5 s,
      efetiva 30 s depois</dd>
  <dt>Observação</dt><dd>QBER estimado de uma amostra binomial dos bits siftados
      — contaminado pelo próprio Raman que a política causou</dd>
  <dt>Teste</dt><dd>8 episódios de 30 min, sementes disjuntas das de treino e
      validação; roteamento e escalonamento idênticos em todas as políticas</dd>
</dl></div>""")

    # -- headline ----------------------------------------------------------
    ratio = float(best_predictive["served_classical"]) / float(
        quantum_first["served_classical"])
    sections.append(_section(
        "01", "resultado", "Proteger a chave não exige silenciar o canal clássico",
        f"<p>A resposta óbvia para proteger o QKD é calar o tráfego clássico nos "
        f"enlaces que carregam chave. Ela funciona: entrega "
        f"<strong>{_pt(float(quantum_first['served_qkd']) * 100, '.1f')}%</strong> da "
        f"chave pedida dentro do prazo. E custa quase tudo — sobra "
        f"{_pt(float(quantum_first['served_classical']) * 100, '.1f')}% da vazão "
        f"clássica.</p>"
        f"<p>Um alocador que decide <em>por enlace</em>, sabendo o que cada um deve "
        f"e quanto consegue produzir, entrega "
        f"<strong>{_pt(float(best_predictive['served_qkd']) * 100, '.1f')}%</strong> "
        f"da chave — {_pt((float(quantum_first['served_qkd']) - float(best_predictive['served_qkd'])) * 100, '.1f')} "
        f"ponto percentual a menos — com "
        f"<strong>{_pt(float(best_predictive['served_classical']) * 100, '.1f')}%</strong> "
        f"do tráfego clássico, contra "
        f"{_pt(float(quantum_first['served_classical']) * 100, '.1f')}%. "
        f"<strong>{_pt(ratio, '.1f')}&times;</strong> mais vazão pela mesma chave.</p>",
        _figure_block({theme.name: _figure_frontier(frontier, theme)
                       for theme in THEMES},
                      "Cada curva é uma família de políticas com seu botão de "
                      "sintonia varrido. Uma família é melhor que outra onde sua "
                      "curva passa acima — e a comparação só existe onde as duas "
                      "chegam ao mesmo nível de serviço."),
        "<p>Comparar políticas que foram sintonizadas para pontos de operação "
        "diferentes não diz nada, então a comparação é feita a serviço de chave "
        "igual: quanto tráfego clássico cada família entrega ao mesmo nível de "
        "chave.</p>",
        _table_pt(matched.assign(target=matched["target"]),
                  {"target": ".0%", "preditivo": ".3f", "limiar reativo": ".3f",
                   "admissão estática": ".3f"},
                  columns=["target", "preditivo", "limiar reativo",
                           "admissão estática"],
                  headers=["chave entregue", "preditivo", "limiar reativo",
                           "admissão estática"]),
        f"<p>A 55% da chave as três famílias coexistem e o alocador por enlace "
        f"entrega {_pt(float(middle['preditivo']) * 100, '.1f')}% do tráfego "
        f"clássico contra {_pt(float(middle['limiar reativo']) * 100, '.1f')}% do "
        f"limiar reativo sintonizado e "
        f"{_pt(float(middle['admissão estática']) * 100, '.1f')}% da admissão "
        f"estática. A 68% o limiar reativo <strong>não chega</strong> em nenhuma "
        f"sintonia — a fronteira dele termina em "
        f"{_pt(float(reactive['served_qkd'].max()) * 100, '.1f')}% — e a admissão "
        f"estática paga o nível com quase toda a vazão: "
        f"{_pt(float(high['admissão estática']) * 100, '.1f')}% contra "
        f"{_pt(float(high['preditivo']) * 100, '.1f')}%.</p>",
        _table_pt(_frontier_table(frontier),
                  {"served_qkd": ".3f", "served_classical": ".3f",
                   "served_entanglement": ".3f", "mean_admission": ".2f",
                   "mean_qber": ".4f"},
                  columns=["family", "knob", "served_qkd", "served_classical",
                           "served_entanglement", "mean_admission", "mean_qber"],
                  headers=["família", "botão", "chave QKD", "clássico",
                           "emaranhamento", "admissão média", "QBER médio"]),
        "<p>O ganho não vem de prever nada. Vem de o controlador saber "
        "<em>qual</em> enlace deve chave, <em>quanta</em>, e <em>até quando</em> — "
        "e de poder tratar os onze vãos como onze problemas em vez de um.</p>"))

    # -- mechanism ---------------------------------------------------------
    sections.append(_section(
        "02", "mecanismo", "O comprimento do vão decide quase tudo",
        "<p>A vazão clássica se compra com QBER, e o preço depende do vão. Um "
        "enlace de 30 km tem sinal forte e mal sente o fundo Raman; um de 145 km "
        "chega com um milésimo do sinal e é destruído pelo mesmo fundo. A curva "
        "de admissão que cada política aplica mostra o que ela entendeu disso.</p>",
        _figure_block({theme.name: _figure_admission_by_length(behaviour, theme)
                       for theme in THEMES},
                      "Média sobre episódios e passos de controle, contando só os "
                      "passos em que o enlace tem chave em atraso — do contrário a "
                      "decisão de coexistência se mistura com \"este enlace não "
                      "tem nada na fila\". A admissão estática é uma reta por "
                      "construção; as outras não."),
        "<p>O alocador preditivo separa a planta em duas. Até 70 km ele admite "
        "praticamente tudo mesmo com chave em atraso: nesses vãos o sinal é forte "
        "e o fundo Raman não chega a morder. De 100 km em diante ele fecha quase "
        "completamente — 1,4% em 105 km, 1,3% em 130 km. A transição acontece "
        "entre 90 e 100 km, que é exatamente onde a curva de coexistência vira.</p>"
        "<p>O limiar reativo fica no meio do caminho em todo lugar: entre 33% e "
        "66% em toda a planta, sem separar os vãos. Ele precisa <em>ver</em> o "
        "QBER subir para reagir, e o QBER só sobe depois que a admissão já foi "
        "aplicada — então ele passa o episódio oscilando em torno de um "
        "compromisso em vez de assentar em vãos diferentes com valores "
        "diferentes.</p>"
        "<p>Os dois pontos fora da curva são reais e vale dizer o que são: 95 km "
        "é a média de dois enlaces distintos com cargas distintas, e 145 km sobe "
        "porque aquele vão raramente tem chave em atraso e, quando tem, o prazo "
        "costuma estar longe — o alvo pró-rata fica pequeno e cabe mesmo com o "
        "canal clássico ligado. É o controlador adiando, não errando.</p>"
        "<p>Isso também explica por que a granularidade do atuador não importa "
        "aqui. As capacidades de chave dos enlaces desta planta variam por um "
        "fator de 200 entre o vão de 30 km e o de 145 km, enquanto o botão de "
        "admissão move a chave de um enlace por um fator de poucas unidades. A "
        "decisão é decidida pelo enlace, não pelo ajuste fino — e passar de 5 "
        "para 21 níveis de admissão não muda nenhum resultado desta seção.</p>"))

    # -- the negative ------------------------------------------------------
    horizon = int(scores["horizon"])
    final_gain = 1.0 - scores["ridge_mse"][-1] / scores["persistence_mse"][-1]
    changed = float(decisions[decisions["forecaster"] == "oráculo"]
                    ["decisions_changed"].mean())
    delta_key = float(decisions[decisions["forecaster"] == "oráculo"]
                      ["delta_qkd"].mean())

    sections.append(_section(
        "03", "resultado negativo", "Prever a deriva muda decisões e não muda resultados",
        f"<p>O pedido desta fase era um <em>modelo preditivo</em>. Ele foi "
        f"construído e funciona como previsor: sobre {horizon} passos de controle "
        f"reduz o erro quadrático da persistência em "
        f"<strong>{_pt(final_gain * 100, '.0f')}%</strong>, e a vantagem cresce com "
        f"o horizonte, que é a assinatura de um modelo que aprendeu reversão à "
        f"média em vez de um deslocamento constante.</p>",
        _figure_block({theme.name: _figure_forecast_skill(scores, theme)
                       for theme in THEMES},
                      "Erro de previsão da deriva residual, em episódios de "
                      "validação, contra a persistência."),
        "<p>E não adianta nada. Trocando <em>só</em> o previsor dentro do mesmo "
        "controlador — mesma lei de controle, mesmo objetivo, mesmos episódios, "
        "comparação pareada — o ganho é indistinguível de zero. Inclusive com um "
        "<strong>oráculo</strong> que conhece a deriva futura exatamente.</p>",
        _figure_block({theme.name: _figure_ablation(ablation, theme)
                       for theme in THEMES},
                      "Diferença pareada contra a persistência, agregada sobre "
                      "quatro alvos de cobertura. Barras de erro: erro padrão "
                      "sobre episódios."),
        _table_pt(_ablation_table(ablation),
                  {"served_qkd": "+.4f", "served_qkd_stderr": ".4f",
                   "served_classical": "+.4f", "served_classical_stderr": ".4f"},
                  columns=["policy", "served_qkd", "served_qkd_stderr",
                           "served_classical", "served_classical_stderr"],
                  headers=["previsor", "Δ chave", "±", "Δ clássico", "±"]),
        f"<p>O diagnóstico é específico, e é mais informativo que o resultado. O "
        f"oráculo <strong>muda {_pt(changed * 100, '.1f')}% das decisões</strong> e "
        f"move a chave entregue em {_pt(delta_key * 100, '+.2f')} pontos "
        f"percentuais. Ele muda decisões que não importam.</p>"
        "<p>A razão está na seção anterior. A decisão por enlace é dominada por "
        "duas coisas que a persistência já conhece perfeitamente: a capacidade "
        "estática do vão e o que ele deve agora. A deriva entra como um terceiro "
        "termo, menor, e o erro de previsão da persistência sobre a janela de "
        "decisão é menor do que o efeito de um único degrau de admissão. O "
        "previsor está certo e é irrelevante.</p>"))

    # -- where it should have paid -----------------------------------------
    sections.append(_section(
        "04", "controle", "Onde a previsão deveria ter pago, e não pagou",
        "<p>Na fase 3 a previsão valia exatamente na proporção da latência do "
        "laço: com atraso zero não há nada a prever. A mesma estrutura existe "
        "aqui — reconfigurar quanto tráfego clássico um enlace carrega leva "
        "segundos a dezenas de segundos, e uma decisão comprometida agora tem que "
        "sobreviver à deriva de <em>então</em>. Se o previsor fosse valer alguma "
        "coisa, seria aqui.</p>",
        _figure_block({theme.name: _figure_decisions(decisions, theme)
                       for theme in THEMES},
                      "Linha cheia (eixo esquerdo): fração de decisões em que o "
                      "previsor escolhe diferente da persistência. Tracejada "
                      "(eixo direito): a diferença que isso faz no resultado."),
        _table_pt(_latency_table(latency),
                  {column: ".4f" for column in ("persistência", "ridge", "oráculo")},
                  columns=["latency_s", "persistência", "ridge", "oráculo"],
                  headers=["latência (s)", "persistência", "ridge", "oráculo"]),
        "<p>O oráculo discorda da persistência em 6% a 11% das decisões, em toda "
        "a faixa de latência, e o resultado não se move em nenhum ponto dela: as "
        "diferenças de chave entregue ficam na terceira casa decimal e trocam de "
        "sinal. A divergência também não <em>cresce</em> com o compromisso, que "
        "era a hipótese — ela é aproximadamente constante.</p>"
        "<p>O teste foi levado até onde deveria ter quebrado: latência de "
        "reconfiguração até 120 s, granularidade de atuador de 5 para 21 níveis, "
        "e <strong>uma planta de vãos uniformes</strong>, onde o comprimento não "
        "distingue nada e só a deriva pode diferenciar os enlaces. Em nenhum "
        "deles persistência, ridge e oráculo se separaram.</p>"
        '<p class="caveat">Uma versão anterior desta varredura mostrava a '
        'previsão importando <em>menos</em> quanto maior o compromisso — o '
        'oposto do esperado, e um bug, não um achado. Quando a latência de '
        'reconfiguração alcança o horizonte do previsor, a janela que a '
        'decisão afeta fica vazia, todos os níveis de admissão pontuam igual '
        'e o controlador para de usar o previsor sem avisar. '
        '<code>PredictiveAllocator</code> agora recusa essa configuração em '
        'vez de absorvê-la, e os números acima são de depois da correção.</p>'
        "<p>A conclusão defensável não é \"previsão não serve para alocação\". É "
        "que <em>neste</em> problema o estado atual já é uma estatística "
        "suficiente para a decisão, porque o processo de deriva reverte à média "
        "num tempo longo comparado à janela de decisão. Um processo com tendência, "
        "com sazonalidade diária, ou com um tempo de correlação da ordem da "
        "latência mudaria isso — e essas são medições a fazer, não suposições a "
        "publicar.</p>"))

    # -- power sweep -------------------------------------------------------
    sections.append(_section(
        "05", "varredura", "Quanto o acoplamento tem que apertar para importar",
        "<p>Tudo acima depende de um número: a potência clássica lançada. Ela "
        "define a taxa de câmbio entre vazão e QBER, e não é um número para "
        "aceitar por confiança.</p>",
        _figure_block({theme.name: _figure_power(power, theme) for theme in THEMES},
                      "Os experimentos publicados de coexistência operam entre "
                      "−20 e −10 dBm (0,01 a 0,1 mW) exatamente por isto."),
        _table_pt(_power_table(power), {},
                  columns=list(_power_table(power).columns),
                  headers=list(_power_table(power).columns)),
        "<p>Abaixo de 0,1 mW o acoplamento é fraco e não há nada a alocar: todas "
        "as políticas entregam quase toda a chave e quase todo o tráfego. Em "
        "1 mW — a potência de um canal de telecom normal — o canal quântico "
        "morre em qualquer admissão apreciável e a única política viável é "
        "silenciar o clássico. A faixa em que existe uma decisão é estreita, e "
        "é onde os experimentos de coexistência de fato operam.</p>"
        "<p>Isso é um resultado de projeto, não um detalhe do modelo: a margem de "
        "coexistência é comprada com filtragem estreita, janelas de detecção "
        "curtas e separação de comprimento de onda, e o que um alocador faz é "
        "administrar a margem que o projeto óptico deixou.</p>"))

    # -- methodology -------------------------------------------------------
    sections.append(_section(
        "06", "metodologia", "O que o alocador tem permissão de saber",
        "<p><strong>Não o QBER verdadeiro.</strong> Ele vê uma estimativa "
        "binomial dos bits siftados sacrificados para medi-la naquele passo. Um "
        "vão de 145 km mede a si mesmo com um centésimo da amostra de um de "
        "30 km — a estimativa é pior exatamente no enlace onde a decisão é mais "
        "difícil.</p>"
        "<p><strong>E essa medição é contaminada pela própria ação.</strong> O "
        "QBER medido inclui o Raman que o alocador acabou de causar. Um previsor "
        "ajustado sobre a soma aprende a prever o próprio sinal de controle e, "
        "realimentado, persegue a si mesmo. O controlador subtrai a interferência "
        "que sabe ter criado — pode fazer isso, porque conhece a própria potência "
        "de transmissão — e só então prevê.</p>"
        "<p><strong>Roteamento e escalonamento são idênticos em todas as "
        "políticas.</strong> Caminho mais curto, resolvido uma vez; dentro de "
        "cada serviço, prazo mais próximo primeiro. A fase 2 já responde a "
        "pergunta de roteamento, e uma política livre para rotear <em>e</em> "
        "estrangular poderia vencer pelos dois lados sem que se soubesse por "
        "qual. A única decisão sob teste é a admissão.</p>"
        "<p><strong>O oráculo é a única coisa que lê o futuro.</strong> Ele existe "
        "para medir a distância até a previsão perfeita. Nada ajustado, "
        "sintonizado ou reportado como implantável toca aquele array.</p>"))

    # -- limits ------------------------------------------------------------
    sections.append(_section(
        "07", "limites", "O que este resultado não é",
        "<p><strong>Um modelo de Raman, não uma fibra medida.</strong> O "
        "coeficiente de espalhamento varia por uma ordem de grandeza na "
        "literatura conforme a separação de comprimento de onda, e a varredura de "
        "potência é o que torna essa incerteza visível em vez de escondida. "
        "Mistura de quatro ondas e modulação de fase cruzada não estão modeladas: "
        "são suprimidas pela separação larga que se assume aqui, e incluí-las "
        "exigiria comprometer-se com uma grade DWDM específica.</p>"
        "<p><strong>Um processo de deriva, herdado da fase 3.</strong> A "
        "conclusão negativa da seção 03 é sobre um resíduo que reverte à média "
        "com tempo de correlação de minutos. É o que a fase 3 mediu, e é uma "
        "propriedade do laço fechado — mas é um modelo, e a afirmação sobre "
        "previsão vale para ele.</p>"
        "<p><strong>Chaveamento de nó confiável.</strong> Uma sessão QKD de várias "
        "pontas consome um bit de chave por enlace do caminho, que é como redes "
        "de nó confiável funcionam e não é segurança ponta a ponta. Repetidores "
        "quânticos — a fase 2 — mudam essa contabilidade inteiramente.</p>"
        "<p><strong>Uma planta, um perfil de tráfego.</strong> A carga foi "
        "escolhida para que nenhuma das duas pontas do compromisso sature — a "
        "73% da chave calando o clássico, 19% admitindo tudo — e é um "
        "argumento (<code>load_scale</code>), não uma constante do problema. "
        "Nenhuma varredura de carga é reportada aqui, então as frações acima "
        "valem para este ponto. A conclusão sobre comprimento de vão dominar a "
        "decisão é sobre uma planta com 200&times; de espalhamento em "
        "capacidade de chave; uma planta homogênea é um problema diferente, "
        "testado aqui apenas como controle da seção 04 e não como "
        "resultado.</p>"))

    body = "\n".join(sections)
    document = f"""<title>{html.escape(title)}</title>
{_FONTS}
<style>{_CSS}
figure + p, figure + div {{ margin-top: 22px; }}
.table-wrap + p {{ margin-top: 18px; }}</style>
<main>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="lede">Dados clássicos e quânticos na mesma fibra: a luz clássica
       espalha para dentro do canal quântico e o QBER sobe. Um alocador que
       decide por enlace entrega a mesma chave com 2,4&times; a vazão clássica —
       e prever a deriva não acrescenta nada. Fase 4 do qnet_autopilot.</p>
  </header>
  {body}
</main>
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# table shaping
# ---------------------------------------------------------------------------

def _frontier_table(frontier: pd.DataFrame) -> pd.DataFrame:
    frame = frontier.copy()
    frame["knob"] = frame["knob"].map(
        lambda value: "—" if pd.isna(value) else _pt(value, ".3g"))
    order = {"quântico primeiro": 0, "preditivo": 1, "limiar reativo": 2,
             "admissão estática": 3, "clássico primeiro": 4}
    frame["_order"] = frame["family"].map(lambda name: order.get(name, 9))
    return frame.sort_values(["_order", "served_qkd"], ascending=[True, False])


def _ablation_table(ablation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for forecaster in ("ridge", "oráculo"):
        subset = ablation[ablation["policy"].str.startswith(forecaster)]
        row = {"policy": forecaster}
        for metric in ("served_qkd", "served_classical"):
            metric_rows = subset[subset["metric"] == metric]
            row[metric] = float(metric_rows["mean"].mean())
            row[f"{metric}_stderr"] = float(
                np.sqrt(np.mean(metric_rows["stderr"] ** 2)))
        rows.append(row)
    return pd.DataFrame(rows)


def _latency_table(latency: pd.DataFrame) -> pd.DataFrame:
    pivot = latency.pivot(index="latency_s", columns="forecaster",
                          values="served_qkd").reset_index()
    pivot["latency_s"] = pivot["latency_s"].map(lambda value: f"{value:.0f}")
    return pivot


def _power_table(power: pd.DataFrame) -> pd.DataFrame:
    frame = power.copy()
    frame["policy"] = frame["policy"].str.replace(r" \(.*\)", "", regex=True)
    pivot = frame.pivot_table(index="power_mw", columns="policy",
                              values="served_qkd").reset_index()
    pivot.columns.name = None
    rendered = {"potência (mW)": pivot["power_mw"].map(lambda v: _pt(v, ".2f"))}
    for column in pivot.columns[1:]:
        rendered[column] = pivot[column].map(lambda v: _pt(v, ".3f"))
    return pd.DataFrame(rendered)
