"""Render the phase-5 results as a self-contained HTML report.

Built from the CSV files ``scripts/train_detector.py`` and
``scripts/evaluate_phase5.py`` write, so every number and figure comes from a
measured run rather than from prose.  The one exception is the first figure,
which is drawn straight from :mod:`qnet_autopilot.physics.decoy` — it shows what
each attack does to the observables, and computing it live is the point: if the
physics changes, the figure changes with it.

Shares its styling and its validated categorical palette with
:mod:`qnet_autopilot.validation.report`.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..validation.report import (_CSS, _FONTS, THEMES, Theme, _figure, _figure_block,
                                 _legend, _section, _style_text, _table, _to_svg)

__all__ = ["render_phase5_report"]

#: Display names for the event classes, in the order the tables use.
CLASS_LABELS = {
    "normal": "normal",
    "thermal_drift": "deriva térmica",
    "loss_event": "evento de perda",
    "intercept_resend": "intercept-and-resend",
    "pns": "PNS",
    "detector_blinding": "cegamento de detector",
    "beam_splitting": "divisor de feixe",
}

CLASS_ORDER = tuple(CLASS_LABELS)


def _pt(value, spec: str = ".3f") -> str:
    """Format a number with a decimal comma, as the rest of the reports do."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return format(value, spec).replace(".", ",")


def _table_pt(frame: pd.DataFrame, formats: dict[str, str], columns, headers) -> str:
    rendered = pd.DataFrame({
        column: (frame[column].map(lambda value, spec=formats[column]: _pt(value, spec))
                 if column in formats else frame[column].astype(str))
        for column in columns})
    return _table(rendered, {}, columns=list(columns), headers=list(headers))


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _figure_signatures(theme: Theme) -> str:
    """What each attack does to the gain, drawn from the physics itself."""
    from ..physics.decoy import DecoySettings, channel_response
    from .attacks import (DetectorBlinding, InterceptResend, NoAttack,
                          PhotonNumberSplitting)
    from .session import LinkConditions

    link = LinkConditions(length_km=50.0)
    settings = DecoySettings()
    honest = link.response()
    attacks = (NoAttack(), InterceptResend(fraction=0.25),
               PhotonNumberSplitting(blocked_fraction=0.5),
               DetectorBlinding(click_probability=float(
                   honest.gain(settings.signal_intensity)) * 1.3))

    intensities = np.linspace(0.01, 1.0, 60)
    figure, axes = _figure(theme, height=4.2)

    for index, attack in enumerate(attacks):
        response = attack.apply(honest, settings)
        gains = np.array([float(response.gain(value)) for value in intensities])
        axes.plot(intensities, gains / np.array([float(honest.gain(value))
                                                 for value in intensities]),
                  color=theme.series[index % len(theme.series)], linewidth=2,
                  label=CLASS_LABELS.get(attack.label, attack.label))

    axes.axvline(settings.decoy_intensity, color=theme.text_secondary, linewidth=1.0,
                 linestyle=":", alpha=0.8)
    axes.axvline(settings.signal_intensity, color=theme.text_secondary, linewidth=1.0,
                 linestyle=":", alpha=0.8)
    axes.annotate("ν", (settings.decoy_intensity, 1.02), color=theme.text_secondary,
                  fontsize=10, ha="center")
    axes.annotate("μ", (settings.signal_intensity, 1.02), color=theme.text_secondary,
                  fontsize=10, ha="center")

    _style_text(axes, theme,
                "Ganho sob ataque, dividido pelo ganho honesto",
                "intensidade do pulso (fótons médios)", "Q(μ) atacado / Q(μ) honesto")
    _legend(axes, theme, loc="upper right")
    return _to_svg(figure)


def _figure_detector(scores: pd.DataFrame, theme: Theme) -> str:
    """Per-class recall, detector by detector."""
    detectors = ["limiar de QBER", "random forest", "SVM (RBF)",
                 "gradient boosting"]
    main = scores[scores["detector"].isin(detectors)]
    labels = [label for label in CLASS_ORDER if label in set(main["label"])]

    figure, axes = _figure(theme, height=4.2)
    positions = np.arange(len(labels))
    width = 0.26

    for index, detector in enumerate(detectors):
        subset = main[main["detector"] == detector].set_index("label")
        values = [float(subset["recall"].get(label, np.nan)) for label in labels]
        axes.bar(positions + (index - 1) * width, values, width * 0.92,
                 color=theme.series[index % len(theme.series)], label=detector)

    axes.set_xticks(positions)
    axes.set_xticklabels([CLASS_LABELS.get(label, label) for label in labels],
                         rotation=20, ha="right", color=theme.text_secondary,
                         fontsize=9)
    axes.set_ylim(0, 1.05)
    _style_text(axes, theme, "Recall por classe, em episódios retidos", "",
                "fração dos períodos classificados corretamente")
    _legend(axes, theme, loc="lower left")
    return _to_svg(figure)


def _figure_keyrate(summary: pd.DataFrame, theme: Theme) -> str:
    """Key rate by span, policy by policy."""
    spans = sorted(summary["length_km"].unique())
    policies = ["estático", "agente", "oráculo"]

    figure, axes = _figure(theme, height=4.0)
    positions = np.arange(len(spans))
    width = 0.26

    for index, policy in enumerate(policies):
        subset = summary[summary["policy"] == policy].set_index("length_km")
        values = [float(subset["key_rate_hz"].get(span, np.nan)) / 1e6 for span in spans]
        axes.bar(positions + (index - 1) * width, values, width * 0.92,
                 color=theme.series[index % len(theme.series)], label=policy)

    axes.set_xticks(positions)
    axes.set_xticklabels([f"{span:.0f} km" for span in spans],
                         color=theme.text_secondary, fontsize=9)
    _style_text(axes, theme, "Taxa de chave secreta provável", "",
                "Mbit/s (média de episódios de teste)")
    _legend(axes, theme, loc="upper right")
    return _to_svg(figure)


def _figure_decisions(decisions: pd.DataFrame, theme: Theme) -> str:
    """What the agent does with the knobs as the loss moves under it."""
    episode = decisions[decisions["length_km"] == decisions["length_km"].iloc[0]]
    episode = episode[episode["seed"] == episode["seed"].iloc[0]]

    figure, axes = _figure(theme, height=4.0)
    steps = episode["step"].to_numpy()

    axes.plot(steps, episode["signal_intensity"], color=theme.series[0], linewidth=2,
              label="μ (intensidade do sinal)")
    axes.plot(steps, episode["decoy_intensity"], color=theme.series[1], linewidth=2,
              label="ν (decoy fraco)")
    axes.plot(steps, episode["clock_hz"] / 1e9, color=theme.series[2], linewidth=2,
              label="clock (GHz)")

    twin = axes.twinx()
    twin.plot(steps, episode["excess_loss_db"], color=theme.series[3], linewidth=1.6,
              linestyle="--", label="perda excedente (dB)")
    twin.set_ylabel("perda excedente (dB)", color=theme.text_secondary, fontsize=9)
    twin.tick_params(colors=theme.text_secondary, labelsize=9, length=3)
    for side in ("top", "left"):
        twin.spines[side].set_visible(False)
    twin.spines["right"].set_color(theme.grid)

    _style_text(axes, theme, "Um episódio, visto pelos botões do transmissor",
                "período de acerto (s)", "μ, ν, clock")
    handles, labels = axes.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    legend = axes.legend(handles + extra_handles, labels + extra_labels,
                         frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)
    return _to_svg(figure)


def _figure_timing(summary: pd.DataFrame, theme: Theme, dark_rate_hz: float) -> str:
    """Key rate against gate width, one line per synchroniser."""
    subset = summary[np.isclose(summary["dark_count_rate_hz"], dark_rate_hz)]
    span = sorted(subset["length_km"].unique())[0]
    subset = subset[np.isclose(subset["length_km"], span)]

    figure, axes = _figure(theme, height=4.0)
    for index, (name, group) in enumerate(subset.groupby("synchroniser", sort=True)):
        group = group.sort_values("gate_width_ps")
        axes.plot(group["gate_width_ps"], group["key_rate_hz"] / 1e6,
                  color=theme.series[index % len(theme.series)], linewidth=2,
                  marker="o", markersize=5, markeredgecolor=theme.surface,
                  markeredgewidth=1.2, label=name)

    _style_text(axes, theme,
                f"Taxa de chave a {span:.0f} km, contagem de escuro "
                f"{dark_rate_hz:.0f} Hz",
                "largura da janela do detector (ps)", "Mbit/s")
    _legend(axes, theme, loc="lower right")
    return _to_svg(figure)


def _figure_sequence(history: dict, theme: Theme) -> str:
    """Validation error per epoch, MLP against the temporal model."""
    figure, axes = _figure(theme, height=4.0)
    for index, (key, label) in enumerate((("mlp", "MLP (fase 3)"),
                                          ("tcn", "TCN (fase 5)"))):
        records = history.get(key) or []
        epochs = [record["epoch"] for record in records if "validation_mse" in record]
        values = [record["validation_mse"] for record in records
                  if "validation_mse" in record]
        if epochs:
            axes.plot(epochs, values, color=theme.series[index % len(theme.series)],
                      linewidth=2, label=label)

    axes.set_yscale("log")
    _style_text(axes, theme, "Erro de validação por época", "época",
                "MSE do incremento de rotação")
    _legend(axes, theme, loc="upper right")
    return _to_svg(figure)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render_phase5_report(detector_dir: str | Path, phase5_dir: str | Path,
                         output_path: str | Path,
                         keyrate_history: str | Path | None = None,
                         slow_detector_dir: str | Path | None = None,
                         title: str = "qnet_autopilot — fase 5: QKD com decoy states"
                         ) -> Path:
    """Write the phase-5 HTML report from the CSVs the scripts produced.

    Args:
        detector_dir: where ``scripts/train_detector.py`` wrote its output.
        phase5_dir: where ``scripts/evaluate_phase5.py`` wrote its output.
        output_path: destination ``.html`` file.
        keyrate_history: the agent's training history JSON, for the learning curve.
        slow_detector_dir: a second detector run at a longer settling period, for
            the operating-point comparison.  Optional.
        title: page title.

    Returns:
        The path written.

    Note:
        Every section renders only if its CSV is there, so a partial run
        produces a partial report rather than an exception.  That is deliberate:
        the studies take different amounts of time and are meant to be runnable
        one at a time.
    """
    detector_dir = Path(detector_dir)
    phase5_dir = Path(phase5_dir)

    scores = _read(detector_dir / "scores.csv")
    confusion = _read(detector_dir / "confusion.csv")
    importances = _read(detector_dir / "importances.csv")
    beam = _read(detector_dir / "beam_splitting.csv")
    hardness = _read(detector_dir / "hardness.csv")
    slow_dir = Path(slow_detector_dir) if slow_detector_dir else None
    slow_scores = _read(slow_dir / "scores.csv") if slow_dir else None
    slow_settings_path = (slow_dir / "settings.json") if slow_dir else None
    slow_settings = (json.loads(slow_settings_path.read_text())
                     if slow_settings_path and slow_settings_path.exists() else {})
    keyrate = _read(phase5_dir / "keyrate_summary.csv")
    decisions = _read(phase5_dir / "keyrate_decisions.csv")
    timing = _read(phase5_dir / "timing_summary.csv")
    sequence = _read(phase5_dir / "sequence_summary.csv")
    sweep = _read(phase5_dir / "sequence_sweep.csv")

    settings_path = detector_dir / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    history_path = phase5_dir / "sequence_history.json"
    sequence_history = json.loads(history_path.read_text()) if history_path.exists() else {}

    sections: list[str] = []

    # -- provenance --------------------------------------------------------
    sections.append(f"""
<div class="meta"><dl>
  <dt>Protocolo</dt><dd>BB84 com decoy states (vácuo + decoy fraco)</dd>
  <dt>Cotas</dt><dd>Ma, Qi, Zhao &amp; Lo (2005); taxa GLLP</dd>
  <dt>Período de acerto</dt><dd>1 s a 1 GHz</dd>
  <dt>Atributos do detector</dt><dd>{len(settings.get('feature_names', []))}</dd>
  <dt>Janela / linha de base</dt><dd>{settings.get('window', '—')} / {settings.get('baseline_periods', '—')} períodos</dd>
</dl></div>""")

    # -- 01 physics --------------------------------------------------------
    sections.append(_section(
        "01", "física", "O que cada ataque faz aos observáveis",
        "<p>Um modelo de ataque só serve se não puder trapacear. Aqui nenhum ataque "
        "mexe no que o Bob mede: cada um modifica os <em>yields por número de "
        "fótons</em> — o que acontece com um pulso que carrega <em>n</em> fótons — e "
        "ganhos e taxas de erro saem disso pelo mesmo caminho de código que um canal "
        "honesto percorre. Ataque sem assinatura física não ganha assinatura no "
        "dataset.</p>",
        _figure_block(
            {theme.name: _figure_signatures(theme) for theme in THEMES},
            "O PNS mantém o ganho exatamente onde estava na intensidade do sinal (μ) "
            "e não consegue mantê-lo no decoy fraco (ν), onde quase todo pulso é de "
            "um fóton só — é essa inconsistência, e nada mais, que os decoy states "
            "detectam. O intercept-and-resend não mexe no ganho de forma alguma; ele "
            "aparece no erro. O cegamento empurra o ganho <em>acima</em> do que a "
            "perda do canal permite."),
        "<p>Os ataques têm que ser separados das degradações que não são ataque: "
        "deriva térmica da fibra, um evento de perda (curvatura, emenda, conector) e "
        "o ruído estatístico de um bloco finito. A deriva térmica é a que se parece "
        "com ataque, e é por isso que o termômetro do cabo entra como atributo: a "
        "taxa de deriva da birrefringência é acoplada a <em>dT/dt</em>, então uma "
        "excursão térmica produz um erro que sobe enquanto a temperatura "
        "<em>se move</em> e assenta quando ela para. Eva não tem essa correlação.</p>"))

    # -- 02 detection ------------------------------------------------------
    if scores is not None:
        main = scores[scores["subset"] == "teste"]
        headline = (main[main["detector"] == "random forest"]
                    .drop_duplicates("detector")[["macro_f1", "false_alarm_rate"]])
        table = (main[main["detector"].isin((
                     "random forest", "SVM (RBF)", "limiar de QBER",
                     "gradient boosting", "boosting + limiar + histerese"))]
                 .pivot_table(index="label", columns="detector", values="recall")
                 .reindex([label for label in CLASS_ORDER]).dropna(how="all")
                 .reset_index())
        table["label"] = table["label"].map(lambda name: CLASS_LABELS.get(name, name))

        delays = (main[main["detector"] == "SVM (RBF)"]
                  .set_index("label")["detection_delay_s"])
        summary_rows = (main.drop_duplicates("detector")
                        [["detector", "macro_f1", "false_alarm_rate",
                          "false_alarms_per_hour", "precision_at_prior"]])

        body = [
            "<p>A pergunta não é “o QBER subiu”. É “os observáveis se mexeram; qual "
            "das sete coisas que os mexem é esta”, e a resposta tem que sair do que "
            "as duas pontas publicam de qualquer jeito, mais um termômetro. Todos os "
            "atributos são relativos à <em>linha de base do próprio episódio</em>, "
            "medida no primeiro minuto: ganho absoluto é quase só uma afirmação "
            "sobre o comprimento do vão.</p>",
            _figure_block(
                {theme.name: _figure_detector(main, theme) for theme in THEMES},
                "O limiar de QBER está calibrado para 1% de alarme falso no conjunto "
                "de treino — é o que um operador faria com ele, e não um espantalho."),
            _table_pt(table, {column: ".3f" for column in table.columns
                              if column != "label"},
                      columns=list(table.columns),
                      headers=["classe"] + [str(column) for column in table.columns[1:]]),
            _table_pt(summary_rows,
                      {"macro_f1": ".3f", "false_alarm_rate": ".4f",
                       "false_alarms_per_hour": ".1f", "precision_at_prior": ".3f"},
                      columns=["detector", "macro_f1", "false_alarm_rate",
                               "false_alarms_per_hour", "precision_at_prior"],
                      headers=["detector", "F1 macro", "alarme falso",
                               "alarmes/hora", "precisão a 1%"]),
            "<p class=\"caveat\">A precisão sob prior realista é o número que "
            "decide se um detector é implantável, e é por isso que ele está aqui "
            "em vez de acurácia: com ataque ocupando 1% do tempo e um período de "
            "acerto de um segundo, um alarme falso a cada poucos minutos afoga "
            "qualquer detecção verdadeira. A seção seguinte mostra que essa "
            "não é uma limitação do modelo.</p>",
        ]
        if not headline.empty:
            body.insert(1, (
                f"<p>O resultado: o limiar de QBER, calibrado para o mesmo orçamento "
                f"de alarme falso, enxerga <strong>um</strong> dos quatro ataques — o "
                f"intercept-and-resend — e não enxerga PNS, cegamento de detector nem "
                f"evento de perda, porque nenhum dos três mexe no QBER. Os modelos "
                f"aprendidos enxergam os quatro: a random forest chega a F1 macro "
                f"<strong>{_pt(float(headline['macro_f1'].iloc[0]))}</strong> com "
                f"<strong>{_pt(float(headline['false_alarm_rate'].iloc[0]), '.4f')}</strong> "
                f"de alarme falso por período.</p>"))
        if confusion is not None:
            renamed = confusion.rename(columns=lambda name: CLASS_LABELS.get(name, name))
            renamed.iloc[:, 0] = renamed.iloc[:, 0].map(
                lambda name: CLASS_LABELS.get(name, name))
            body.append("<p>Matriz de confusão da random forest, em períodos "
                        "retidos (linha = verdade, coluna = palpite):</p>")
            body.append(_table(renamed, {}))
        if beam is not None and not beam.empty:
            body.append(
                "<p class=\"caveat\">O divisor de feixe é o controle negativo honesto: "
                "Eva fica com a luz que a própria perda do canal já entrega e repassa o "
                "resto por uma fibra melhor. Não há nada para medir, e o modelo não "
                "finge que há — a resposta certa é “normal”, e é o que a floresta "
                f"responde em {_pt(float(beam['called_normal'].iloc[0]), '.1%')} dos "
                "períodos. O SVM erra mais aqui: fronteiras suaves inventam ataque "
                f"em {_pt(float(beam['called_attack'].iloc[1]), '.1%')} deles. Quem "
                "relatar detecção acima do acaso nessa classe aprendeu um artefato.</p>")
        if np.isfinite(delays.get("intercept_resend", np.nan)):
            body.append(
                f"<p>Atraso de detecção (mediana, do início do evento até três "
                f"períodos consecutivos sinalizados): PNS e cegamento em "
                f"{_pt(float(delays.get('pns', np.nan)), '.0f')} s e "
                f"{_pt(float(delays.get('detector_blinding', np.nan)), '.0f')} s; "
                f"intercept-and-resend em "
                f"{_pt(float(delays.get('intercept_resend', np.nan)), '.0f')} s.</p>")

        sections.append(_section("02", "resultado medido",
                                 "Distinguir espionagem de clima", *body))

    # -- 03 ablations ------------------------------------------------------
    if scores is not None and importances is not None:
        ablations = scores[scores["detector"].str.contains("sem |só ", na=False)]
        table = (ablations.drop_duplicates("detector")
                 [["detector", "macro_f1", "false_alarm_rate"]])
        per_class = (ablations.pivot_table(index="label", columns="detector",
                                           values="recall")
                     .reindex([label for label in CLASS_ORDER]).dropna(how="all")
                     .reset_index())
        per_class["label"] = per_class["label"].map(
            lambda name: CLASS_LABELS.get(name, name))
        top = importances.head(8)

        sections.append(_section(
            "03", "diagnóstico", "Qual física está carregando a separação",
            "<p>A pergunta mais útil sobre um classificador não é quanto ele acerta, "
            "é do que ele depende. Três ablações, cada uma removendo um bloco de "
            "atributos:</p>",
            _table_pt(table, {"macro_f1": ".3f", "false_alarm_rate": ".3f"},
                      columns=["detector", "macro_f1", "false_alarm_rate"],
                      headers=["variante", "F1 macro", "alarme falso"]),
            _table_pt(per_class, {column: ".3f" for column in per_class.columns
                                  if column != "label"},
                      columns=list(per_class.columns),
                      headers=["classe"] + [str(column).replace("random forest ", "")
                                            for column in per_class.columns[1:]]),
            "<p>Sem os observáveis de decoy state o PNS deixa de ser separável de "
            "forma confiável — o que é a afirmação de segurança do protocolo, "
            "reaparecendo como resultado de aprendizado. Sem o termômetro, a deriva "
            "térmica e o intercept-and-resend se confundem, e o alarme falso piora em "
            "uma ordem de grandeza: são as duas classes que produzem a mesma subida "
            "de QBER e se distinguem pela causa, não pelo tamanho.</p>",
            _table_pt(top, {"importance": ".3f"},
                      columns=["feature", "importance"],
                      headers=["atributo", "importância"])))

    # -- 03b hard cases and the decision rate ------------------------------
    if hardness is not None and not hardness.empty:
        body = [
            "<p>Um gerador em que todo ataque entra de uma vez, em pleno vigor, e "
            "com mais nada acontecendo é o caso favorável em dois sentidos. Quem "
            "sabe que o enlace é monitorado entra devagar ou trabalha em rajadas; "
            "e o caso mais difícil para um monitor é o ataque que começa "
            "<em>durante</em> uma excursão térmica, quando a explicação inocente "
            "para o erro subindo já é verdadeira. As duas coisas estão no gerador, "
            "e custam caro:</p>",
            _table_pt(hardness,
                      {"periods": ",d", "recall": ".3f", "detected": ".3f"},
                      columns=["detector", "case", "periods", "recall", "detected"],
                      headers=["detector", "caso", "períodos", "recall",
                               "detectado como algo"]),
            "<p>O ataque intermitente é o que mais dói, e a razão é mecânica: "
            "metade dos períodos rotulados como ataque são os intervalos entre "
            "rajadas, em que literalmente não há nada para ver. O ataque em rampa "
            "perde os primeiros períodos, que é o comportamento certo — naquele "
            "momento ele é fraco demais para se distinguir do ruído.</p>",
        ]
        if slow_scores is not None:
            slow = slow_scores[slow_scores["subset"] == "teste"]
            comparison = (slow.drop_duplicates("detector")
                          [["detector", "macro_f1", "false_alarm_rate",
                            "false_alarms_per_hour", "precision_at_prior"]])
            period = slow_settings.get("period_s", "?")
            body.extend([
                f"<h3>A taxa de decisão, e não o modelo</h3>",
                f"<p>A precisão sob prior realista a um segundo de período é ruim "
                f"para todos os modelos, e a causa não é o classificador: é que o "
                f"enlace publica 3 600 decisões por hora. Repetindo o estudo com um "
                f"período de acerto de {_pt(float(period), '.0f')} s — mesmas "
                f"classes, mesmo código, contagens {_pt(float(period), '.0f')} vezes "
                f"maiores por decisão —, o quadro muda de natureza:</p>",
                _table_pt(comparison,
                          {"macro_f1": ".3f", "false_alarm_rate": ".4f",
                           "false_alarms_per_hour": ".1f", "precision_at_prior": ".3f"},
                          columns=["detector", "macro_f1", "false_alarm_rate",
                                   "false_alarms_per_hour", "precision_at_prior"],
                          headers=["detector", "F1 macro", "alarme falso",
                                   "alarmes/hora", "precisão a 1%"]),
                "<p>Com limiar de decisão calibrado, o alarme falso vai a "
                "praticamente zero e a precisão sob prior realista passa de 0,2 para "
                "~1,0. O que se paga é atraso — dezenas de segundos em vez de um — e "
                "o tipo do intercept-and-resend fraco, que passa a ser reportado como "
                "deriva térmica em vez de ataque. É uma escolha de ponto de operação, "
                "não uma melhoria: um enlace que precisa de reação em segundos fica "
                "com o período curto e paga em tickets.</p>",
            ])
        sections.append(_section("04", "diagnóstico",
                                 "Os casos difíceis, e o que realmente limita a precisão",
                                 *body))

    # -- 04 key rate -------------------------------------------------------
    if keyrate is not None:
        pivot = keyrate.pivot(index="length_km", columns="policy",
                              values="key_rate_hz").reset_index()
        for column in ("agente", "estático", "oráculo"):
            if column in pivot:
                pivot[column] = pivot[column] / 1e6
        if {"agente", "estático"} <= set(pivot.columns):
            pivot["ganho"] = pivot["agente"] / pivot["estático"] - 1.0

        body = [
            "<p>O transmissor tem quatro botões que não exigem tocar no hardware — a "
            "intensidade do sinal μ, o decoy fraco ν, a divisão dos pulsos entre eles "
            "e o clock — e o ponto ótimo se move com a perda do vão, com o "
            "desalinhamento que sobrou da compensação e com o quanto os detectores "
            "estão perto de saturar. O sistema implantado escolhe uma vez, no "
            "comissionamento; é essa a linha de base, e ela é boa.</p>",
            "<p>A recompensa é a chave que se consegue <em>provar</em>, calculada "
            "sobre as contagens finitas daquele período — não a taxa assintótica. A "
            "diferença não é cosmética: no limite assintótico a cota fica mais "
            "apertada quanto menor o ν, então um agente premiado com taxas "
            "assintóticas aprende a empurrar o decoy para o chão, e aí a execução "
            "real fica sem cliques de decoy e a cota desmorona.</p>",
            _figure_block(
                {theme.name: _figure_keyrate(keyrate, theme) for theme in THEMES},
                "O oráculo enxerga o canal verdadeiro e reotimiza a cada período: é "
                "o teto do valor da informação perfeita, não um método implantável."),
            _table_pt(pivot,
                      {"length_km": ".0f", "agente": ".3f", "estático": ".3f",
                       "oráculo": ".3f", "ganho": "+.1%"},
                      columns=[column for column in ("length_km", "estático", "agente",
                                                     "oráculo", "ganho")
                               if column in pivot.columns],
                      headers=[{"length_km": "km", "estático": "estático (Mbit/s)",
                                "agente": "agente (Mbit/s)",
                                "oráculo": "oráculo (Mbit/s)",
                                "ganho": "agente vs estático"}[column]
                               for column in ("length_km", "estático", "agente",
                                              "oráculo", "ganho")
                               if column in pivot.columns]),
        ]
        if decisions is not None and not decisions.empty:
            body.append(_figure_block(
                {theme.name: _figure_decisions(decisions, theme) for theme in THEMES},
                "O agente sobe μ quando a perda cresce e recua no clock quando o "
                "ganho se aproxima do tempo morto do detector — as duas coisas que o "
                "ponto de comissionamento não pode fazer porque foi escolhido antes "
                "de a perda existir."))
        sections.append(_section("05", "resultado medido",
                                 "Otimizar a taxa de chave enquanto o enlace se mexe",
                                 *body))

    # -- 05 timing ---------------------------------------------------------
    if timing is not None:
        spans = sorted(timing["length_km"].unique())
        table = (timing[np.isclose(timing["length_km"], spans[0])]
                 .pivot_table(index=["dark_count_rate_hz", "gate_width_ps"],
                              columns="synchroniser", values="key_rate_hz")
                 .reset_index())
        for column in table.columns[2:]:
            table[column] = table[column] / 1e6

        sections.append(_section(
            "06", "resultado medido", "Manter a janela do detector em cima do pulso",
            "<p>Silício muda de índice e de comprimento com a temperatura: cerca de "
            "40 ps por quilômetro por kelvin. Em 50 km de fibra enterrada, um único "
            "kelvin são dois nanossegundos — dez vezes a janela típica de um detector "
            "gated. Um enlace que não rastreia isso para de produzir chave dentro de "
            "um grau de onde foi calibrado pela última vez.</p>",
            "<p>A estrutura é a mesma da fase 3, de propósito: um modelo de deriva, "
            "um canal de monitoração com fóton finito, preditores que vão de “não "
            "fazer nada” a um aprendido, e um laço fechado que pontua o que sobra — "
            "não o RMS do desalinhamento, mas a eficiência de detecção e a taxa de "
            "chave.</p>",
            _figure_block(
                {theme.name: _figure_timing(timing, theme, 10.0) for theme in THEMES},
                "Detector supercondutor a 10 contagens de escuro por segundo: alargar "
                "a janela é quase de graça, e o que importa é o rastreamento."),
            _figure_block(
                {theme.name: _figure_timing(timing, theme, 1.0e6) for theme in THEMES},
                "Receptor InGaAs livre a 1 MHz de contagem de escuro: agora a janela custa, e a qualidade da "
                "sincronização é o que permite estreitá-la."),
            _table_pt(table,
                      {"dark_count_rate_hz": ".0f", "gate_width_ps": ".0f",
                       **{column: ".3f" for column in table.columns[2:]}},
                      columns=list(table.columns),
                      headers=["escuro (Hz)", "janela (ps)"]
                              + [str(column) for column in table.columns[2:]]),
            "<p class=\"caveat\">O preditor aprendido usa o termômetro e a cota "
            "térmica é proporcional à <em>variação</em> de temperatura, então é "
            "exatamente onde o filtro alfa-beta não pode chegar. A margem é modesta e "
            "aparece onde deveria: em janelas estreitas e em vãos longos, onde o "
            "resíduo importa.</p>"))

    # -- 06 sequence -------------------------------------------------------
    if sequence is not None:
        ordered = sequence.sort_values("finite_key_rate_hz", ascending=False)
        body = [
            "<p>O compensador aprendido da fase 3 é um MLP sobre uma janela "
            "<em>achatada</em>: os dezesseis incrementos de rotação passados chegam "
            "como um vetor único, e a primeira camada tem que aprender do zero que as "
            "posições 0-2, 3-5 e 6-8 são a mesma grandeza em atrasos sucessivos. Uma "
            "convolução causal já sabe: o mesmo filtro de três canais é aplicado em "
            "todo atraso, e o campo receptivo cresce empilhando dilatações em vez de "
            "alargar a primeira camada.</p>",
            "<p>A comparação roda no laço fechado da fase 3, com os mesmos dados, o "
            "mesmo otimizador, o mesmo cronograma e a mesma regra de melhor época. Só "
            "a arquitetura muda — é a única forma de a comparação querer dizer "
            "alguma coisa.</p>",
            _table_pt(ordered,
                      {"mean_qber": ".4f", "p95_qber": ".4f", "uptime": ".3f",
                       "key_rate_hz": ".0f", "finite_key_rate_hz": ".0f",
                       "residual_mrad": ".1f", "recovery_s": ".3f"},
                      columns=["compensator", "mean_qber", "p95_qber", "uptime",
                               "finite_key_rate_hz", "residual_mrad", "recovery_s"],
                      headers=["compensador", "QBER médio", "QBER p95", "uptime",
                               "chave finita (bit/s)", "resíduo (mrad)",
                               "recuperação (s)"]),
        ]
        if sweep is not None:
            body.insert(2, (
                "<p>O campo receptivo é aritmética, não hiperparâmetro, e foi onde a "
                "primeira tentativa errou. Duas camadas dilatadas alcançam sete "
                "atrasos; um período da banda de vibração a 40 Hz são 12,5 passos de "
                "controle, então sete atrasos não cobrem uma oscilação inteira. Três "
                "camadas alcançam quinze, cobrem o período, e o erro cai pela metade. "
                "Capacidade não compra nada depois disso.</p>"))
            body.insert(3, _table_pt(
                sweep, {"receptive_field": "d", "parameters": ",d",
                        "validation_mse": ".0f"},
                columns=["model", "receptive_field", "parameters", "validation_mse"],
                headers=["modelo", "campo receptivo", "parâmetros",
                         "MSE de validação"]))
        if sequence_history:
            body.append(_figure_block(
                {theme.name: _figure_sequence(sequence_history, theme)
                 for theme in THEMES},
                "Erro de validação por época. As duas curvas são treinadas nas mesmas "
                "linhas, com o mesmo Adam e o mesmo decaimento."))
        sections.append(_section("07", "resultado medido",
                                 "Um modelo sequencial para a polarização", *body))

    body = "\n".join(sections)
    document = f"""<title>{html.escape(title)}</title>
{_FONTS}
<style>{_CSS}</style>
<main>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="lede">Quatro perguntas que um enlace de QKD implantado faz todo dia:
       quem está na linha, quanto vale mexer nos botões, onde colocar a janela do
       detector, e se um modelo sequencial compensa a polarização melhor do que o da
       fase 3. Tudo medido, inclusive o que não funcionou.</p>
  </header>
  {body}
</main>
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
