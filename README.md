# qnet_autopilot

Camada de física e dados para controle aprendido de redes quânticas, construída
sobre o [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe).

Cinco fases prontas e validadas: **fase 1** (física e dados), **fase 2**
(roteamento + controle de repetidor aprendidos), **fase 3** (compensação de
polarização em tempo real), **fase 4** (alocação híbrida clássico/quântico) e
**fase 5** (QKD com decoy states: detecção de espionagem, otimização da taxa de
chave e sincronismo temporal). Pule para
[Fase 2](#fase-2--roteamento-e-timing-aprendidos) se quiser o agente de
roteamento, [Fase 3](#fase-3--compensação-de-polarização-em-tempo-real) para o
compensador, [Fase 4](#fase-4--alocação-híbrida-clássicoquântico) para o
alocador, ou [Fase 5](#fase-5--qkd-com-decoy-states-espionagem-taxa-de-chave-e-sincronismo)
para a camada de QKD.

## Fase 1 — física e dados

Ainda não há agente nesta fase — há o simulador, os
modelos analíticos rápidos que um agente vai consultar milhões de vezes, os
geradores de dados, e as verificações que dizem em que faixa cada um desses
modelos pode ser usado. Um agente treinado sobre física não conferida aprende a
explorar o erro do simulador, não a operar a rede.

As quatro capacidades-alvo e onde cada uma se apoia neste pacote:

| capacidade | o que esta fase entrega |
|---|---|
| Roteamento de emaranhamento (RL) | topologias, orçamento de enlace por aresta (`rate_hz`, `expected_time_s`), composição exata de fidelidade por swapping |
| Controle de repetidores (timing de purificação/swapping) | BBPSSW e DEJMPS em forma fechada e vetorizada, decoerência de memória, limiar de break-even da purificação |
| Compensação de flutuações na fibra | modelo de deriva de polarização (térmica + vibração + saltos) e dataset supervisionado com modelo de observação honesto — estendido na fase 3 para rastrear a rotação inteira |
| Alocação em redes híbridas | traço de capacidade por enlace variando no tempo + fluxo de demandas clássicas, QKD e emaranhamento — fechado na fase 4 com a física de coexistência Raman |

---

## Instalação

```bash
pip install -r requirements.txt
pip install -e .
python -m qnet_autopilot info
```

`python -m qnet_autopilot info` diz de onde o SeQUeNCe está vindo e se alguma
dependência dele está sendo substituída por um shim. Rode isso primeiro em
qualquer ambiente novo.

### Se o PyPI não estiver acessível

Este pacote foi desenvolvido num ambiente com o PyPI bloqueado por política de
egresso — nem `sequence`, nem `qutip`, nem `gmpy2`, nem `stim`. Em vez de
entregar código não executado, o pacote traz uma camada de compatibilidade:

```bash
./scripts/fetch_sequence.sh          # clona o SeQUeNCe do GitHub em vendor/
python -m qnet_autopilot info        # confirma o que está com shim
python scripts/verify_sequence.py    # roda a suíte do próprio SeQUeNCe sob os shims
```

`qnet_autopilot.compat.bootstrap()` substitui apenas o que falta:

- **`gmpy2`** → `decimal` com a precisão pedida. Usado só na aritmética de
  tempo em picossegundos de `QuantumChannel` e `Detector`.
- **`qutip` / `qutip_qip`** → numpy. Usados só para compor a matriz unitária de
  um circuito, em dois lugares. As convenções de ordem de qubits e de ordem de
  multiplicação são as do qutip e estão travadas por teste.
- **`stim`** → **não** tem shim. É um simulador Clifford de verdade; fingir
  seria produzir resultado plausível e errado. O que existe é um placeholder que
  levanta erro no primeiro uso, para que os backends ket e densidade continuem
  funcionando (o SeQUeNCe importa `stim` avidamente pelo `kernel`). Nada no
  caminho de repetidores toca o formalismo estabilizador.
- **`typing.override`** é retroportado no Python 3.11. O SeQUeNCe declara
  `requires-python >= 3.12`, mas esse decorador é o único 3.12-ismo na árvore —
  o pacote inteiro compila em 3.11.

Sob os shims, **461 dos 473 testes do próprio SeQUeNCe passam**. As 12 falhas
são todas esperadas: 10 exigem o `stim` real, 1 exige `typer`, e 1 depende de
ordem entre testes.

---

## Uso

```python
from qnet_autopilot.physics import bell_diagonal as bds, MemoryParameters, LinkBudget

# Uma decisão de purificação, em forma fechada e vetorizada.
par = bds.werner(0.88)
p_sucesso, purificado = bds.dejmps(par, par, gate_fid=0.99, meas_fid=0.99)

# Quanto tempo esse par continua útil.
memoria = MemoryParameters(coherence_time_s=1.0)
from qnet_autopilot.physics import time_to_fidelity
segundos = time_to_fidelity(0.8, purificado, memoria)

# O enlace consegue produzir pares nesse ritmo?
LinkBudget.compute(length_km=50.0)
```

Tudo em `physics` aceita arrays de forma `(..., 4)` e faz broadcast pelos eixos
à esquerda, então um lote inteiro de enlaces — ou de ambientes de RL — é uma
única chamada.

```bash
# Datasets
python -m qnet_autopilot generate polarization --out data/pol.csv --episodes 8 --duration 1800
python -m qnet_autopilot generate link         --out data/link.csv --attempts 400
python -m qnet_autopilot generate capacity     --out data/cap.csv --scenario metro
python -m qnet_autopilot generate demands      --out data/dem.csv --scenario metro

# Verificações e relatório
python -m qnet_autopilot validate --report reports/validation.html

# Orçamento de enlace, antes de simular qualquer coisa
python -m qnet_autopilot budget --scenario metro
```

Todo dataset é salvo com um `.meta.json` ao lado contendo os parâmetros, a
semente, a versão do SeQUeNCe e quais dependências estavam com shim.
`datagen.schema.load()` recusa um dataset sem esse arquivo.

---

## O que foi conferido, e com que força

`python -m qnet_autopilot validate --report reports/validation.html` gera o
relatório completo. O resumo:

| verificação | tipo | resultado |
|---|---|---|
| BBPSSW, DEJMPS, swapping, decoerência vs SeQUeNCe | exato | 800 comparações, discrepância máxima **0,00e+00** |
| Cadeia de swapping vs `F = (1 + 3·wⁿ)/4` | exato | reproduz exatamente com portas ideais |
| Probabilidade de heralding vs simulação evento a evento | estatístico | dentro do IC 95% em 6/6 distâncias |
| Estatística da deriva de polarização | interno | vibração estacionária; deriva térmica a 0,95× do previsto |
| Break-even da purificação | interno | curva monotônica, limiar sobe com erro de porta |

**"Exato" não é figura de linguagem.** As funções Bell-diagonais em
`physics/bell_diagonal.py` são transcrições vetorizadas dos protocolos do
SeQUeNCe, não aproximações. Elas existem porque avaliar um passo de purificação
no SeQUeNCe exige montar um `Timeline`, dois `Node` e quatro `Memory`, o que é
certo para simular uma rede e errado para o laço interno de um agente.
`tests/test_bds_matches_sequence.py` compara as duas implementações; se o
upstream mudar uma fórmula, o teste quebra.

**O modelo de fibra não tem referência externa.** Não há contraparte no SeQUeNCe
e este pacote não tem dado de campo. As verificações confirmam que o código
implementa o modelo documentado — não que o modelo descreva a sua fibra. As
constantes em `physics/constants.py` são um ponto de operação plausível para
fibra enterrada, documentadas como centro de varredura; fibra aérea deriva uma
ordem de grandeza mais rápido.

### Dois achados que mudam números

**A taxa de heralding segue `η²/2`, não Barrett-Kok.** O
`EntanglementGenerationA` do SeQUeNCe se comporta como o esquema de clique único;
a curva de Barrett-Kok fica uma ordem de grandeza abaixo. Se você dimensionou
taxas assumindo Barrett-Kok, esse é o fator.

**Com portas a 0,95, purificar sempre piora o par.** O limiar de fidelidade de
entrada para o BBPSSW compensar sobe rápido: 0,500 com portas ideais, 0,548 a
0,99, 0,701 a 0,97, e a 0,95 a curva de ganho nunca cruza zero. Um controlador
que purifica por política fixa está destruindo emaranhamento em metade do
espaço de parâmetros plausível.

---

## Bugs no SeQUeNCe 1.0.0

Encontrados ao rodar a suíte deles e montar as topologias. Vale reportar
upstream.

**1. `gate_fidelity` e `measurement_fidelity` são silenciosamente descartadas.**
`nx_converter.generate_config` grava as duas chaves na config de cada nó;
`RouterNetTopo._add_nodes` nunca as lê — constrói
`QuantumRouter(name, tl, memo_size, component_templates=template)` e nada mais.
Todo roteador fica com o default `gate_fid = meas_fid = 1`.

Isso não é cosmético: são exatamente os parâmetros que decidem se a purificação
ajuda ou atrapalha e com que rapidez a fidelidade degrada ao longo de uma
cadeia. Um estudo montado por esse caminho roda com operações locais perfeitas
independentemente do que configurar.

`build_topology()` contorna atribuindo os atributos após a construção — os
protocolos BDS leem `self.owner.gate_fid` em tempo de execução, então é
equivalente a tê-los passado. `tests/test_network_and_datagen.py` afirma tanto
que o contorno funciona quanto que o caminho padrão ainda descarta as chaves,
para que o dia em que o upstream corrigir isso o teste quebre e o contorno possa
ser removido em vez de virar código morto.

**2. `sequence/utils/metrics/__init__.py:139` levanta `NameError`.** A linha 55
apenas *anota* `time_provider: Timeline` sem atribuir; o global só passa a
existir depois de `register_time_provider()`. Qualquer `metrics.record()` antes
disso estoura — é por isso que 10 testes de `tests/utils/test_metrics.py` falham
no repositório deles.

**3. O formalismo quântico é estado global do processo.**
`QuantumManager.set_global_manager_formalism` não tem escopo, então montar uma
topologia Bell-diagonal muda o formalismo para tudo o que rodar depois no mesmo
processo. `LinkExperiment.run()` define e restaura o formalismo que precisa em
volta de cada execução; sem isso, um experimento de enlace falharia ou, pior,
se comportaria de forma diferente conforme o que rodou antes.

---

## Estrutura

```
qnet_autopilot/
  compat/        bootstrap + shims de gmpy2, qutip, stim; retroporte de typing.override
  physics/       constants, bell_diagonal, fiber, memory, link, coexistence,
                 field_calibration (as constantes medidas em campo, com citação),
                 blocks (onde cortar os blocos de chave: temporizador,
                 controlador causal, oráculo por programação dinâmica),
                 decoy (BB84 com decoy states: yields por número de fótons,
                 cotas de Ma 2005, taxa GLLP, saturação do detector)
  network/       scenarios (chain, ring, grid, metro), topology (→ RouterNetTopo)
  datagen/       schema (proveniência), polarization, entanglement, traffic,
                 measured (ajusta o modelo a uma gravação de polarímetro real),
                 qkd (sessões rotuladas de um enlace com decoy states)
  validation/    checks (as verificações), twin (o gêmeo contra o SeQUeNCe),
                 report, realism_report
  env/           state, actions, network_env, features — o MDP;
                 sequence_runner (as mesmas primitivas dentro do SeQUeNCe)
  agents/        base (a costura), baselines, numpy_nn, actor_critic, rollout,
                 normalise, dual_head, report, ablation_report
  compensation/  monitoring, closed_loop, predictors, train, residual,
                 heteroscedastic, actuator, sequence (convolução causal
                 dilatada, backward escrito à mão), report
  allocation/    state, env, scenario, forecast, policies, rollout, report
  qkd/           attacks (ataques como modificação dos yields), session (o
                 enlace rodando, com evento e estatística finita), detection
                 (floresta, SVM e o limiar de QBER), keyrate (ambiente e
                 agente ator-crítico), timing (deriva térmica do atraso e o
                 laço de sincronização), report
  cli.py         info | validate | generate | budget | decoy
tests/           893 testes
scripts/
  fetch_sequence.sh     clona o SeQUeNCe quando o PyPI não está acessível
  verify_sequence.py    roda a suíte do SeQUeNCe sob os shims
  minipytest.py         runner pytest-compatível para ambientes sem pytest
  task.py               a tarefa de referência da fase 2
  train_agent.py        treina o agente, com retomada
  evaluate_phase2.py    agente contra todos os baselines, em teste retido
  analyse_phase2.py     varredura de temperatura, transferência, comportamento
  task3.py              o ponto de operação da fase 3
  train_compensator.py  ajusta ridge e MLP nos mesmos dados
  evaluate_phase3.py    laço fechado, recuperação, varreduras de latência e fótons
  render_reports.py     gera os HTMLs a partir dos CSVs em runs/
  task4.py              o ponto de operação da fase 4
  train_allocator.py    ajusta o previsor de deriva
  evaluate_phase4.py    fronteiras, ablação do previsor, varreduras
  ablate_agent.py       escada de refinamentos do agente da fase 2
  ablate_compensator.py cabeça de variância contra controle de mesma origem
  analyse_variance_head.py  há heterocedasticidade para prever?
  evaluate_actuator.py  o laço da fase 3 com um controlador de curso finito
  evaluate_realism.py   gêmeo × SeQUeNCe, e chave finita nas fases 3 e 4
  evaluate_field.py     a fase 3 no ponto de operação calibrado em campo
  evaluate_sequence.py  o agente da fase 2 dentro do laço de eventos do SeQUeNCe
  task5.py              o ponto de operação da fase 5
  train_detector.py     gera as sessões, ajusta floresta e SVM, roda as ablações
  train_keyrate.py      treina o agente de taxa de chave (especialista ou generalista)
  evaluate_phase5.py    taxa de chave, sincronização e compensador sequencial
```

### Duas decisões de projeto que valem explicação

**Nomes idênticos aos do upstream.** `gate_fid`, `meas_fid`, `p_succ`,
`c_I`/`c_X`/`c_Y`/`c_Z`, `same_report`, `bds_elems` — as transcrições mantêm os
nomes do SeQUeNCe para que as duas implementações possam ser lidas lado a lado
e a divergência apareça de imediato.

**Verdade de referência fica em colunas `true_*`, e o framing supervisionado se
recusa a usá-las.** Um compensador real não conhece o ângulo verdadeiro na
esfera de Poincaré; ele tem uma estimativa de QBER a partir de uma amostra
binomial finita de fótons de monitoramento, mais sensores ambientais ruidosos.
`make_supervised_windows()` levanta erro se você passar uma coluna `true_*` como
feature, e `chronological_split()` corta no tempo com intervalo de guarda —
janelas sobrepostas embaralhadas tornam qualquer métrica de teste sem sentido.

---

## Testes

```bash
pytest                                  # 893 testes
python scripts/minipytest.py tests      # sem pytest instalado
python scripts/verify_sequence.py       # a suíte do próprio SeQUeNCe, sob os shims
```

---

## Fase 2 — roteamento e timing aprendidos

Um agente decide **caminho e timing juntos**. Separá-los é artificial: escolher
rota sem controlar purificação é meia decisão, e o acoplamento entre as duas é
justamente o que uma política aprende e um heurístico não.

### O ambiente

MDP de **uma primitiva por passo**, escolhida do conjunto legal no momento:

| primitiva | efeito | custo em tempo |
|---|---|---|
| `GENERATE(enlace)` | tenta heralding até sucesso ou orçamento | tempo amostrado até o primeiro sucesso |
| `PURIFY(a, b)` | DEJMPS/BBPSSW; consome b, e a também se falhar | um atraso de sinalização no vão do par |
| `SWAP(nó, e, d)` | funde dois pares adjacentes; perde os dois se a BSM falhar | atraso até a ponta mais distante |
| `DELIVER(par)` | entrega, se estiver acima do limiar | zero |
| `DISCARD(par)` | libera a memória | zero |
| `WAIT` | passa o tempo | um slot |

**O relógio avança pela latência da ação escolhida.** É a decisão de projeto em
que tudo se apoia: se as operações fossem instantâneas, "purificar sempre" seria
grátis e ótimo, e não haveria problema de timing a aprender. Como não são,
purificar um par de 200 km custa um milissegundo de sinalização clássica durante
o qual *todo* par armazenado na rede decoere.

A física vem inteira da camada validada da fase 1 — as mesmas funções
Bell-diagonais que batem com o SeQUeNCe até a precisão de máquina.

### A política

A rede **pontua cada ação legal a partir das features daquela ação** e tira um
softmax sobre as pontuações. Não há camada de saída de tamanho fixo, então o
número de ações legais pode mudar a cada passo — e muda — e os mesmos pesos
rodam numa topologia que o agente nunca viu. Ator-crítico com GAE(λ), bônus de
entropia decrescente e clipping de gradiente, em **numpy com backprop manual**,
conferido contra diferenças finitas a 10⁻¹⁰.

### Resultados (400 episódios de teste, sementes disjuntas do treino)

> **Retratado.** Os números abaixo cobram **uma** travessia do vão por
> tentativa de heralding. Rodando o agente dentro do laço de eventos do
> SeQUeNCe, o protocolo cobra **4,1** — e com o custo corrigido a vantagem
> desaparece: o agente retreinado empata com as melhores heurísticas nos prazos
> publicados e perde para `greedy_fidelity` com prazos quatro vezes maiores.
> A tabela fica aqui porque é o que o pacote produziu e porque a correção só faz
> sentido contra ela; veja
> [Fechando as lacunas](#fechando-as-lacunas--gêmeo-bloco-finito-atuador-real-e-dado-de-campo).

| política | sucesso | tempo | purificações | tentativas |
|---|---|---|---|---|
| aleatório | 3,2% | 64,1 ms | 1,5 | 22,4 |
| caminho mais curto | 36,0% | 44,6 ms | 0 | 13,2 |
| fidelidade gananciosa | 36,0% | 44,8 ms | 4,3 | 20,2 |
| limiar 0,94 | 50,0% | 54,6 ms | 9,7 | 24,9 |
| limiar 0,90 | 55,0% | 53,4 ms | 7,7 | 24,2 |
| **agente aprendido** | **87,5%** | **27,3 ms** | **1,8** | **12,1** |

Diferença pareada contra o melhor heurístico: **+32,5 ± 2,6 p.p.**, com 139
vitórias, 9 derrotas e 252 empates nos mesmos episódios.

O que importa está nas três últimas colunas: **o agente não vence purificando
mais.** Vence purificando 4,3× menos, com metade das tentativas de enlace, em
metade do tempo. Ele purifica pares de fidelidade *mais baixa* (0,829 contra
0,885 do heurístico) — espera até que a rodada valha a pena, em vez de disparar
num limiar fixo.

### Dois achados que vale conhecer antes de replicar

**A política estocástica vence a determinística com folga.** Pegar o argmax dos
mesmos pesos derruba o resultado de 91,2% para 36,0%. Não é bug: o A2C otimiza o
retorno da política *estocástica*, e o teorema de melhoria de política exige a
função de valor daquela política, que não é a que o crítico aprendeu. Além
disso, as features são agregados, então situações distintas podem gerar vetores
idênticos — sob esse aliasing o problema é um POMDP e uma política estocástica
pode dominar estritamente qualquer determinística. A varredura de temperatura
tem ótimo largo (0,25 a 1,5), o que sustenta a primeira explicação.

**Os heurísticos são difíceis de escrever bem, e um heurístico fraco invalida o
resultado.** A primeira versão do `ShortestPathPolicy` gerava no enlace com maior
probabilidade de heralding — que numa cadeia uniforme é sempre o mesmo — e
entupia a memória do primeiro nó até travar. Marcava **abaixo do aleatório**.
Cobrir os trechos *descobertos* primeiro é a correção, e é por isso que
`ActionFeature.SEGMENT_UNCOVERED` existe.

### Generalização para topologias não vistas

Sem nenhum retreino:

| topologia | saltos | caminho curto | limiar 0,90 | agente |
|---|---|---|---|---|
| cadeia-4 (treinada) | 3 | 36,0% | 55,2% | **86,4%** |
| cadeia-3 | 2 | 100% | 100% | 100% |
| cadeia-5 | 4 | 0,0% | 0,4% | **10,4%** |
| cadeia-4, vãos de 30 km | 3 | 34,0% | 18,0% | **45,6%** |
| anel-6 | 3 | 36,0% | 55,2% | **77,6%** |

A linha dos vãos de 30 km é a demonstração mais limpa da tese: ali o heurístico
de limiar **quebra** — cai para 18,0%, abaixo dos 34,0% de simplesmente não
purificar — porque continua gastando pares em rodadas que, com enlaces mais
lentos, já não cabem no prazo. Uma constante não representa uma decisão que
depende do estado.

### Reproduzir

```bash
python scripts/train_agent.py --iterations 150 --out runs/agent
python scripts/analyse_phase2.py --policy runs/agent/policy.npz   # escolhe a temperatura na validação
python scripts/evaluate_phase2.py --episodes 400                  # teste retido
python -c "from qnet_autopilot.agents import render_phase2_report; \
           render_phase2_report('runs/agent', 'reports/phase2.html')"
```

Treino leva ~5 minutos em CPU. A tarefa de referência está em `scripts/task.py`.

### Limites deste resultado

- **Uma tarefa, um ponto de operação.** Cadeia de 4 nós, vãos iguais, um par a
  entregar. O ponto foi escolhido para que a purificação importasse — com portas
  a 0,99 em vez de 0,999 ela quase não ajuda e a decisão deixa de existir.
- **Uma semente de treino.** Gradiente de política tem variância alta entre
  sementes; a afirmação é sobre esta política, não sobre a expectativa do método.
- **O custo de heralding estava errado por 4×, e isso apaga o resultado
  acima.** A política agora roda dentro do laço de eventos do SeQUeNCe, e a
  comparação mostra que o ambiente analítico cobrava uma travessia do vão por
  tentativa quando o protocolo cobra 4,1. Corrigido o custo, os dois simuladores
  concordam em todas as políticas e a vantagem do agente some. O que continua
  fora do laço é o motor de regras do SeQUeNCe — contenção entre requisições
  concorrentes.
- **Sem PyTorch.** A rede é numpy com backprop manual. Isso limita a escala: um
  codificador de grafo de verdade, que enxergaria a topologia em vez de resumos
  por ação, precisa de um framework.

---

## Fase 3 — compensação de polarização em tempo real

A fibra se move mais rápido do que o laço de controle consegue responder. Um
preditor que só enxerga o que o canal de monitoramento realmente mede derruba o
QBER em **5×** sobre um rastreador alfa-beta sintonizado, e multiplica a taxa de
chave secreta por **1,6×**.

Relatório completo: `reports/phase3.html`.

### O ponto de operação

Fibra **aérea** sob perturbação mecânica, e a escolha não é arbitrária: em fibra
enterrada a deriva térmica em 40 ms é de milirradianos, muito abaixo do ruído de
medição, e qualquer coisa funciona. O que desafia um compensador é a **banda
acústica** — vibração de 40 ± 6 Hz com amplitude 0,30 rad move o estado de
polarização por uma fração grande de radiano dentro de um único atraso de laço.

| | |
|---|---|
| deriva térmica | 0,05 rad/s, Ornstein-Uhlenbeck, τ = 60 s |
| vibração | 40 ± 6 Hz, amplitude 0,30 rad, 8 tons |
| saltos | Poisson, ~1 a cada 20 s, escala 0,35 rad |
| período de controle | 2 ms |
| latência do laço | 40 ms (varrida de 4 a 160 ms) |
| monitoramento | 2 estados de sonda × 3 eixos × 400 fótons |

Tudo em `scripts/task3.py`, num só lugar, para que treino e avaliação não possam
divergir.

### O modelo de observação, e uma correção da fase 1

**Um estado de sonda não determina a rotação da fibra.** Ele deixa uma liberdade
U(1): todas as rotações que levam o estado lançado ao mesmo ponto da esfera de
Poincaré diferem por um giro em torno dele. Dá para calcular o QBER daquele
estado e nunca perceber a lacuna — foi exatamente o que a fase 1 fez, e é como um
dataset de polarização acaba incapaz de treinar um controlador.

A correção: `PolarizationDriftModel.simulate(..., track_rotation=True)` guarda a
matriz de rotação inteira, e `compensation.monitoring.measure_rotation` modela o
que um receptor real vê — dois estados de sonda, cada um medido em três eixos
analisadores com um número **finito** de fótons (estimativa binomial, variância
`(1 - v²)/N`), reortonormalizados de volta para SO(3) por Gram-Schmidt.
`MonitoringConfig` **recusa** uma configuração com um único estado de sonda.

### O que o compensador tem permissão de saber

O simulador conhece a rotação verdadeira, e basta deixá-la entrar em um de três
lugares para produzir um controlador excelente e impossível de implantar:

- **Como feature.** O compensador vê apenas as contagens binomiais. Nunca a
  verdade.
- **Como alvo de treino.** O alvo padrão de `build_dataset` é a **medição
  futura**, não a verdade futura. Sob perda quadrática ambas têm o mesmo
  minimizador (a medição é não enviesada), mas só a primeira pode ser recoletada
  em campo; treinar contra a verdade credita ao modelo uma remoção de ruído que
  ele só consegue fazer porque viu a resposta limpa. `target="true"` existe, e é
  só para diagnóstico.
- **Como divisão dos dados.** Janelas se sobrepõem, então um split embaralhado
  põe quase-duplicatas dos dois lados. `chronological_split` corta no tempo com
  banda de guarda maior que a janela mais o horizonte.

### A representação

Rotações são previstas como **incrementos no espaço tangente**: tudo é expresso
como `log(R_futuro · R_agoraᵀ)`, um vetor de três componentes, e a predição é
`exp(ŷ) · R_agora`. Duas razões, nenhuma cosmética:

- **Matrizes de rotação não têm média.** A média de duas rotações não é uma
  rotação, então um modelo que regride nas entradas da matriz gasta capacidade
  reaprendendo a restrição de SO(3).
- **Torna o problema invariante a translação.** O que a fibra faz em seguida
  depende de como ela está se movendo, não de onde ela está na esfera.

### Resultados (8 execuções de 20 s, sementes disjuntas do treino)

| compensador | QBER médio | p95 | disponibilidade | chave (Hz) | resíduo (mrad) |
|---|---|---|---|---|---|
| sem compensação | 0,1316 | 0,3341 | 56,0% | 22 686 | 857,5 |
| persistência | 0,1329 | 0,3253 | 50,1% | 17 180 | 876,9 |
| extrapolação linear | 0,3692 | 0,7953 | 17,2% | 6 215 | 1 724,6 |
| filtro alfa-beta (α=0,05, β=0,005) | 0,0383 | 0,0985 | 96,6% | 53 927 | 450,8 |
| **ridge (Wiener)** | **0,0077** | **0,0211** | **99,9%** | **87 022** | **193,9** |
| MLP | 0,0074 | 0,0203 | 99,9% | 87 554 | 188,6 |

Duas linhas merecem atenção antes das boas. A **persistência** — corrigir para a
última medição — é *pior do que não corrigir nada*: com 40 ms de atraso ela
aplica uma correção que a vibração já tornou obsoleta. A **extrapolação linear** é
pior ainda, porque diferenciar duas estimativas ruidosas dobra a variância.
Compensar mal é uma opção real, e é por isso que o alfa-beta existe — e é
re-sintonizado por busca em grade em **cada** ponto de operação, para que a
comparação não seja contra um espantalho.

**Recuperação depois de um salto** (mediana / p90, em ms): ridge e MLP 0,9 / 52;
alfa-beta 11,5 / 73; persistência 236 / 473; extrapolação linear 3 010 / 6 170.
Sem compensação, só 3 dos 8 saltos se recuperam antes do fim da execução.

### Três achados que vale conhecer antes de replicar

**A rede neural não bate o filtro de Wiener.** O ganho da MLP sobre regressão
ridge é de 5,6% no erro de validação e nada mensurável no QBER; a 80 ms de
latência ela é 2% *pior*. Isso não é uma rede mal treinada — é o que a física
prevê. Uma regressão linear sobre uma janela de amostras passadas **é** o filtro
de Wiener, o preditor linear ótimo para um processo estacionário, e o modelo de
deriva é uma soma de Ornstein-Uhlenbeck com senoides de banda estreita:
gaussiano e estacionário. O que isso *não* diz é que redes não sirvam aqui —
diz que a estrutura não linear precisa existir no sinal primeiro. Histerese de
atuador, dependência não linear de temperatura e não estacionariedade em escalas
de horas são reais e não estão neste modelo.

**O alfa-beta não é monótono na latência, e é uma ressonância.** QBER médio de
0,0514 / 0,0536 / 0,0093 / 0,0373 / 0,0504 / 0,0445 em 4 / 8 / 20 / 40 / 80 /
160 ms. O período da banda de vibração é 25 ms; quando a latência coincide com um
período inteiro, a fibra volta para perto de onde estava quando foi medida e
seguir a última medição fica quase ótimo de graça. Verificado: a 24 ms o
alfa-beta marca 0,0052 e a **própria persistência** marca 0,0061, contra 0,0383 e
0,1329 a 40 ms. É uma armadilha de bancada, não uma vantagem — um projeto
sintonizado nessa ressonância funciona brilhantemente até a frequência de
excitação mudar. A curva do ridge é monótona, como tem que ser.

**Gastar fótons de sondagem só compensa com um preditor que os use.** De 25 para
800 fótons por base — um fator 32 — o QBER do filtro clássico cai 12% (0,0422 →
0,0371) e satura: ele não está limitado por ruído de medição e sim pelo próprio
atraso de rastreamento. O ridge cai por um fator 3 no mesmo intervalo (0,0190 →
0,0062). Aumentar o orçamento de monitoramento de um controlador clássico é
dinheiro jogado fora.

### Reproduzir

```bash
python scripts/train_compensator.py --out runs/compensator
python scripts/evaluate_phase3.py --out runs/compensator
python scripts/render_reports.py --phase 3
```

A avaliação completa leva ~25 minutos em CPU; `--only main recovery` roda só a
comparação principal. O laço fechado é vetorizado (`searchsorted` para saber qual
correção está em vigor em cada amostra), o que o deixou ~10× mais rápido do que a
versão amostra a amostra.

### Limites deste resultado

- **Um modelo de deriva, não uma fibra.** Os parâmetros vêm da literatura de
  fibra aérea, mas nenhum dado real entrou nesta fase.
- **A banda mecânica está na frequência errada.** Estes números usam 40 Hz;
  em fibra aérea o que se mede é o tom da rede elétrica com harmônicos, e no
  ponto corrigido a QBER da MLP retreinada sobe de 0,0078 para 0,0136 — a
  vantagem sobre o filtro clássico cai de 4,8× para 3,8× e sobrevive. O modelo
  treinado a 40 Hz, porém, **não** transfere: rodando a 60 Hz dá QBER 0,26,
  pior que não compensar. Treinar sobre a faixa 45–70 Hz cobre aquela faixa e
  nada além dela — cinco hertz fora, o modelo colapsa. Veja
  [Fechando as lacunas](#fechando-as-lacunas--gêmeo-bloco-finito-atuador-real-e-dado-de-campo).
- **Um atuador ideal — medido depois, e com consequência.** Estes números supõem
  que a rotação pedida aparece na fibra exatamente. Com uma cascata de
  retardadores de curso e velocidade finitos, a vantagem do preditor exige um
  dispositivo acima de ~250 rad/s; abaixo disso o filtro alfa-beta ganha. O
  reset de fim de curso, apontado aqui como a não linearidade decisiva, dispara
  muito e não muda nada. Ambos em
  [Fechando as lacunas](#fechando-as-lacunas--gêmeo-bloco-finito-atuador-real-e-dado-de-campo).
- **Um ponto de operação escolhido para ser difícil.** A varredura de latência
  mostra o regime em que a compensação preditiva deixa de importar.
- **QBER de polarização, não QBER de sistema.** Ruído de detector, contagens
  escuras e erros de modulação entram somados e não estão aqui. A taxa de chave
  absoluta é um teto.
- **Taxas de chave assintóticas.** As taxas acima supõem blocos infinitos. Com
  blocos de um segundo elas caem a 63% para a MLP e a 1,6% para a persistência —
  a correção está em `bb84_finite_key_rate` e medida na mesma seção. A ordenação
  não muda; ela se acentua.

---

## Fase 4 — alocação híbrida clássico/quântico

Dados clássicos e quânticos dividem a mesma fibra, e a luz clássica espalha para
dentro do canal quântico. Um alocador que decide **por enlace** entrega a mesma
chave que silenciar o canal clássico, com **2,4×** a vazão clássica. Prever a
deriva, medido contra um oráculo, não acrescenta nada — e a razão é específica.

Relatório completo: `reports/phase4.html`.

### O acoplamento, que é a peça que faltava

Sem física de coexistência, "alocação híbrida" são dois problemas de
escalonamento independentes desenhados no mesmo grafo, e admitir tudo nos dois
serviços é ótimo. Não há nada a aprender.

O acoplamento real é o **espalhamento Raman espontâneo**: um canal clássico
carrega miliwatts, e uma fração minúscula dessa potência é espalhada pela própria
sílica para dentro da banda do canal quântico. Esses fótons são
indistinguíveis do sinal no detector — chegam na banda certa, na fibra certa,
dentro do gate — e, sendo despolarizados, metade deles erra. `physics/coexistence.py`
implementa potência direta e reversa, contagens de ruído por gate, a combinação
com o erro de alinhamento, e a **inversão**: quanta potência clássica cabe sob um
teto de QBER dado o quanto a deriva já gastou.

| | |
|---|---|
| espalhamento | ρ ≈ 4×10⁻⁹ /km/nm, filtro de 0,1 nm, gate de 1 ns |
| direta | `P₀ e^{-αL} ρ Δλ L` — cresce, atinge pico em 1/α ≈ 21,7 km, depois cai |
| reversa | `P₀ ρ Δλ (1 - e^{-2αL}) / 2α` — satura; pior em vãos longos |
| QBER | `(e_alinh·N_sinal + ½N_fundo) / (N_sinal + N_fundo)`, teto em 0,5 |

O ponto de operação está ancorado nos experimentos publicados de coexistência
(Chapuran 2009, Peters 2009, Eraerds 2010): a −10 dBm um enlace de 50 km ainda
produz chave; a 0 dBm não produz nenhuma. Há um teste que trava isso.

### O problema

Planta metropolitana SP–RJ da fase 1: 9 nós, 11 vãos de 30 a 145 km. A cada 5 s o
alocador escolhe, por enlace, a fração de capacidade clássica a admitir. A
decisão **só entra em vigor 30 s depois** — reconfigurar tráfego não é
instantâneo — e essa latência é varrida.

A deriva é o **resíduo do compensador da fase 3**, não uma fibra nua: o modelo da
fase 1 passeia aleatoriamente e leva o QBER a 0,5, o que é correto para uma fibra
sem controle e errado para uma implantada. `compensation/residual.py` modela o que
um laço fechado deixa — um processo que reverte à média, com cauda direita longa
e transientes de salto — calibrado no que a fase 3 mediu (QBER médio 0,77%, p95
2,1%). Um terço dos enlaces é sorteado como degradado a cada episódio, e quais
são é resorteado para que comprimento de vão e qualidade de compensação não
fiquem colineares.

### Resultados (8 episódios de 30 min, sementes retidas)

A fronteira, não um placar: chave entregue no prazo contra vazão clássica, com o
botão de sintonia de cada família varrido.

| política | chave QKD | clássico |
|---|---|---|
| quântico primeiro (cala o clássico) | 0,694 | 0,114 |
| **preditivo (cobertura 5)** | **0,690** | **0,273** |
| preditivo (cobertura 0,1) | 0,665 | 0,348 |
| preditivo (cobertura 0,02) | 0,604 | 0,562 |
| limiar reativo sintonizado, melhor em chave | 0,603 | 0,436 |
| admissão estática 0,25 | 0,565 | 0,512 |
| clássico primeiro | 0,218 | 1,000 |

A comparação só existe onde as famílias chegam ao mesmo nível de serviço:

| chave entregue | preditivo | limiar reativo | admissão estática |
|---|---|---|---|
| 55% | **0,660** | 0,613 | 0,535 |
| 68% | **0,301** | não alcança | 0,062 |

A 68% o limiar reativo **não chega em nenhuma sintonia** — sua fronteira termina
em 60,3% da chave — e a admissão estática paga esse nível com quase toda a
vazão.

**O mecanismo** é o comprimento do vão. Contando só os passos em que o enlace tem
chave em atraso, o alocador preditivo admite ~99% até 70 km, 68% em 90 km, 28% em
100 km e ~1% de 105 a 130 km. O limiar reativo fica entre 33% e 66% em toda a
planta, sem separar nada: ele precisa *ver* o QBER subir para reagir, e o QBER só
sobe depois que a admissão já foi aplicada.

### O resultado negativo, e o diagnóstico

O pedido da fase era um *modelo preditivo*. Ele foi construído e funciona como
previsor: reduz o erro quadrático da persistência em até **43%**, e o ganho cresce
com o horizonte — a assinatura de um modelo que aprendeu reversão à média.

E não muda nada. Trocando **só** o previsor dentro do mesmo controlador — mesma
lei de controle, mesmos episódios, comparação pareada — o ganho é indistinguível
de zero, inclusive com um **oráculo que conhece a deriva futura exatamente**:
diferenças da ordem de 0,001 na fração entregue, metade delas negativas.

O diagnóstico é mais informativo que o resultado. O oráculo **muda 8,6% das
decisões e move a chave entregue em −0,02 ponto percentual**. Ele
muda decisões que não importam. A decisão por enlace é dominada por duas coisas
que a persistência já conhece: a capacidade estática do vão — as capacidades de
chave desta planta variam por um fator de 200 — e o que o enlace deve agora. A
deriva entra como um terceiro termo menor, e o erro de previsão da persistência
sobre a janela de decisão é menor que o efeito de um único degrau de admissão.

O teste foi levado até onde deveria quebrar: latência de reconfiguração até 120 s,
granularidade de atuador de 5 para 21 níveis, e uma **planta de vãos uniformes**
onde o comprimento não distingue nada e só a deriva pode. Em nenhum deles o
oráculo se separou da persistência.

A conclusão defensável não é "previsão não serve para alocação". É que *neste*
problema o estado atual já é estatística suficiente, porque a deriva reverte à
média num tempo longo comparado à janela de decisão. Um processo com tendência,
sazonalidade diária, ou tempo de correlação da ordem da latência mudaria isso —
e isso é medição a fazer, não suposição a publicar.

### Um bug que vale registrar

A varredura de latência inicialmente mostrou a previsão importando *menos* quanto
maior o compromisso — o oposto do esperado. A causa: quando a latência de
reconfiguração alcança o horizonte do previsor, a janela que a decisão afeta fica
vazia, todos os níveis de admissão pontuam igual e o controlador para de usar o
previsor silenciosamente. `PredictiveAllocator` agora **recusa** essa
configuração em vez de absorvê-la.

### Reproduzir

```bash
python scripts/train_allocator.py --out runs/allocator
python scripts/evaluate_phase4.py --out runs/allocator
python scripts/render_reports.py --phase 4
```

A avaliação completa leva ~40 minutos em CPU; `--only frontier` roda só a
fronteira. O ponto de operação está em `scripts/task4.py`.

### Limites deste resultado

- **Um modelo de Raman, não uma fibra medida.** O coeficiente varia por uma ordem
  de grandeza na literatura conforme a separação de comprimento de onda, e a
  varredura de potência torna essa incerteza visível. Mistura de quatro ondas e
  modulação de fase cruzada não estão modeladas.
- **Chaveamento de nó confiável.** Uma sessão QKD de várias pontas consome um bit
  por enlace do caminho. Não é segurança ponta a ponta; repetidores quânticos — a
  fase 2 — mudam essa contabilidade inteiramente.
- **Roteamento e escalonamento fixos.** Caminho mais curto e prazo mais próximo
  primeiro, idênticos em todas as políticas, para que a única decisão sob teste
  seja a admissão.
- **Uma planta, um perfil de tráfego.** A conclusão sobre comprimento de vão
  dominar vale para uma planta com 200× de espalhamento em capacidade de chave.
- **Chave assintótica por padrão.** `AllocationConfig(finite_key=True)` cobra o
  custo de bloco finito e derruba o serviço de chave de todas as políticas em
  30–55%, sem alterar a ordenação. O ponto de operação, porém, se move: cobrar a
  chave pelo que ela custa faz o controlador preditivo admitir menos tráfego
  clássico, e por isso a manchete de 2,4× foi remedida com as fronteiras
  retraçadas em vez de extrapolada. Ela sobrevive: 2,7× a serviço igualado.

---

## Estudo de refinamento — dual-head, e o que realmente ajudou

Duas hipóteses de *dual-head*, medidas contra controles do mesmo código.
Relatório completo: `reports/ablation.html`.

### Fase 2: trunk compartilhado entre ator e crítico

O agente da fase 2 tem **duas redes inteiramente separadas** — ator sobre
`concat(estado, ação)`, crítico sobre `estado` — que não compartilham nada. A
alternativa padrão é um trunk que codifica o estado, lido por duas cabeças.
Aqui isso não é uma refatoração: as cabeças leem coisas diferentes, então o
trunk codifica o estado uma vez e o embedding é difundido entre as linhas de
ação — o gradiente que volta é uma **soma** sobre essas linhas, mais o do
crítico.

A escada mede uma mudança por degrau, todas as variantes nas mesmas sementes:

| variante | sementes | sucesso | ± | parâmetros | colapsos |
|---|---|---|---|---|---|
| linha de base (fase 2) | 5 | 0,875 | 0,006 | 11 458 | 0 |
| + normalização de features | 5 | 0,901 | 0,009 | 11 458 | 0 |
| + capacidade e treino longo | 19 | 0,918 | 0,004 | 39 298 | 1 |
| dual-head (trunk compartilhado) | 20 | 0,916 | 0,004 | 54 274 | 0 |

| mudança | diferença pareada | ± | V / D | sementes |
|---|---|---|---|---|
| + normalização − linha de base | **+0,0255** | 0,0087 | 4 / 1 | 5 |
| + capacidade − + normalização | **+0,0245** | 0,0168 | 3 / 2 | 5 |
| dual-head − + capacidade | −0,0007 | 0,0057 | 10 / 8 | 19 |

**O que ajudou não foi a arquitetura.** Normalizar as features vale +2,6 p.p.;
capacidade e treino mais longo valem +2,4 p.p.; juntos levam o agente de 87,5%
para **91,8%** de sucesso. O trunk compartilhado vale −0,07 ± 0,57 p.p. — não um
resultado fraco a favor, mas zero, medido com precisão suficiente para excluir
qualquer efeito maior que ±1,1 p.p. — e custa +38% de parâmetros e +19% de tempo
de treino.

**O quase-achado.** Uma execução das redes separadas em vinte *colapsou*: o
gradiente de política derrubou o ator e a validação nunca passou de 7,5%. O
dual-head não colapsou em nenhuma das vinte, incluindo a mesma semente. Com dez
sementes isso parecia um efeito de confiabilidade; com vinte, 1 contra 0 dá um
teste exato de Fisher com p = 1,00 e a razão de desvios padrão é 1,15. Não
sustenta conclusão. É exatamente por isso que a varredura de sementes existe.

### Fase 3: cabeça de média + cabeça de variância

A outra hipótese: uma cabeça prevê o incremento de rotação, a outra prevê a
**variância** dele, e o laço encolhe a correção na proporção da incerteza
(média posterior sob um prior centrado em zero — o encolhimento não é um
parâmetro livre disfarçado de Bayes).

| modelo | QBER médio | chave (Hz) |
|---|---|---|
| somente média (mesmo código, verossimilhança desligada) | **0,00720** | 87 776 |
| dual-head sem encolhimento | 0,00752 | 87 335 |
| dual-head com encolhimento | 0,00737 | 87 492 |

Pior que o controle, em todas as sementes. A cabeça de variância fica **muito
bem calibrada na média** (razão erro/variância prevista 1,04, contra 1,00 de uma
gaussiana perfeita) e **não sabe distinguir uma linha difícil de uma fácil**
(correlação de postos com o erro real: +0,035).

### O diagnóstico, que é o resultado mais útil

Uma cabeça de variância sem informação é dois resultados diferentes: ou ela não
aprendeu o que estava lá, ou não havia nada lá. Dá para separar sem consultar
modelo nenhum.

A fase 3 estabeleceu que esta deriva é gaussiana estacionária mais senoides de
banda estreita — é por isso que a predição linear é ótima para a **média**. A
mesma gaussianidade força a **variância condicional a ser constante**: para
variáveis conjuntamente gaussianas, a variância de uma dada as outras não
depende do valor delas. A teoria prevê exatamente o que foi medido.

E prevê onde haveria exceção: o salto do cabo, o único evento não gaussiano do
modelo. É onde está — nos primeiros 20 ms depois de um salto o erro é **3,9×** o
da linha de base tranquila. Mas essas linhas são **0,15% dos dados**, e tudo
dentro de 100 ms de um salto soma 0,72% das linhas e 2,3% do erro total.

O teto fecha o argumento: ajustando um modelo *diretamente* ao logaritmo do erro
quadrático, das mesmas features que a cabeça lê, o **R² de validação é
−0,0001**. Uma cabeça não pode bater um modelo ajustado diretamente à quantidade
que ela estima. O limite é zero, e o resultado negativo não é sobre arquitetura,
otimizador nem quantidade de treino.

Previsão falsificável que isso deixa: num ponto de operação com muito mais
saltos, a fração heterocedástica cresce e a cabeça deveria começar a pagar.

### Três armadilhas encontradas, todas registradas no código

- **Penalidade de ridge não adimensional.** Contra features de ordem 0,005, o
  default encolhia o ajuste até o intercepto — silenciosamente, produzindo um
  previsor pior que a persistência que ele deveria bater.
- **Instabilidade de NLL gaussiana.** O gradiente da média é `(μ−y)/σ²`, então
  assim que a variância infla numa linha, aquela linha para de treinar a média.
  A primeira versão levou a variância a um bilhão de vezes a escala do alvo e o
  compensador aprendeu a se desligar. Corrigido com β-NLL (Seitzer et al.,
  ICLR 2022).
- **Grampo de variância como armadilha de mão única.** Zerar o gradiente no
  limite impede empurrar para fora *e* impede voltar: a primeira batelada que
  cruzasse o limite fixava a saída para sempre. Agora o gradiente passa quando
  aponta de volta para dentro.

### Reproduzir

```bash
python scripts/ablate_agent.py --out runs/ablation --seeds 20
python scripts/ablate_compensator.py --out runs/ablation --seeds 3
python scripts/analyse_variance_head.py --out runs/ablation
python -c "from qnet_autopilot.agents.ablation_report import render_ablation_report; \
           render_ablation_report('runs/ablation','reports/ablation.html')"
```

A varredura da fase 2 leva ~3 h em CPU (20 sementes × 4 variantes); a da fase 3,
~1 h.

---

## Fechando as lacunas — gêmeo, bloco finito, atuador real e dado de campo

A revisão de aplicabilidade nomeou quatro coisas entre estes resultados e uma
afirmação de que o sistema serve para alguma coisa. Três são mensuráveis aqui e
estão medidas. A quarta — dado de fibra real — não tem série pública nenhuma em
fibra terrestre, mas a literatura de campo publica estatísticas suficientes para
derrubar duas premissas da fase 3.

### 1. O gêmeo analítico, confrontado em laço fechado

A fase 1 validou as formas fechadas *uma operação por vez*. O ambiente onde o
agente treina é outro software: escolhe qual função chamar, sobre qual par, com
quais parâmetros, depois de quanto tempo. Agora uma trajetória inteira roda no
ambiente enquanto cada par vivo é reconstruído a partir de objetos do SeQUeNCe —
protocolos de purificação e swapping, e `Memory.bds_decohere` chamado uma vez
**por ponta** do par — usando os tempos que o próprio ambiente reporta.

| | |
|---|---|
| passos comparados | 212 (126 tentativas de geração, 91 bem-sucedidas; 24 purificações, 46 swaps, 7 entregas) |
| pior divergência de estado | 2,5 × 10⁻¹² |
| pior divergência de probabilidade | 1,1 × 10⁻¹² |

O maquinário está em `qnet_autopilot/validation/twin.py` e os testes em
`tests/test_env_matches_sequence.py`, com **dois controles negativos**: cobrar
decoerência de uma ponta em vez de duas diverge acima de 10⁻³, e a fidelidade de
porta imperfeita muda o swap o suficiente para que descartá-la fosse detectável.
Uma comparação que não pode falhar não é evidência.

O que continua fora: o agente não roda dentro do laço de eventos do SeQUeNCe.

### 2. Chave finita — as taxas publicadas estavam altas

Toda taxa de chave das fases 3 e 4 saiu da fórmula assintótica. `bb84_finite_key_length`
implementa o limite composável de Tomamichel, Lim, Gisin & Renner (*Nat. Commun.*
3 634, 2012), com a fração de teste otimizada — fixá-la faz o limite convergir
para 90% do assintótico em vez de para ele.

Blocos de um segundo a 10⁵ bits peneirados por segundo:

| compensador | assintótico | bloco de 1 s | fração |
|---|---:|---:|---:|
| MLP | 87,3 kbit/s | 54,6 kbit/s | 0,63 |
| ridge (Wiener) | 86,7 | 54,2 | 0,63 |
| filtro alfa-beta | 54,5 | 28,1 | 0,52 |
| persistência | 17,4 | 0,28 | 0,016 |

A penalidade não é um fator comum: ela cresce conforme a QBER se aproxima do
limiar. Com blocos de cinco segundos a persistência entrega **zero** — um enlace
que a fórmula assintótica descrevia como produzindo 17 kbit/s. A correção
reforça a conclusão da fase 3 em vez de enfraquecê-la.

Esse zero não é uma propriedade do enlace, e sim da suposição de que o bloco tem
duração fixa: a seção 8 troca o temporizador por um controlador e o mesmo enlace
passa a entregar 983 bit/s.

Na fase 4, `AllocationConfig(finite_key=True)` derruba o serviço de chave de
todas as políticas em 30–55% sem alterar a ordenação — mas move o ponto de
operação, porque um controlador que paga o preço real da chave compra menos
vazão clássica. Por isso as duas fronteiras foram retraçadas em vez de a
manchete ser extrapolada — e ela sobrevive: a serviço de chave igualado, dentro
da sobreposição das duas fronteiras, o controlador preditivo entrega **2,7× a
vazão clássica** da admissão estática, contra os 2,4× publicados com chave
assintótica.

### 3. O atuador real — uma condição de hardware, e uma hipótese refutada

`compensation/actuator.py` converte o comando em ângulos de uma cascata de
retardadores X-Y-X pela decomposição de Euler mais próxima da posição atual, com
curso e velocidade finitos e reset de fim de curso. Duas consequências:

- **A vantagem do preditor exige um dispositivo acima de ~250 rad/s.** Abaixo
  disso o filtro alfa-beta ganha, e a 30 rad/s a persistência empata com a MLP —
  um atuador lento filtra passa-baixas o comando, o que arruína quem rastreia e
  resgata quem estava piorando o enlace. A varredura é logarítmica, então o
  cruzamento é o intervalo 120–250 rad/s: a 120 os dois empatam, a 250 a vantagem
  do preditor já está estabelecida. O mecanismo é o curso: ridge e MLP pedem
  9.312 radianos em 20 s contra 3.933 do filtro clássico.
- **O reset dispara muito e quase não muda nada.** Até 107 resets por minuto
  com curso de 2π, e num dispositivo rápido a QBER média de cada compensador é
  idêntica em curso de 2π, 4π e 12π. A fase 3 chamou o reset de "exatamente a
  não linearidade que mudaria a conclusão" sobre redes neurais; ele se completa
  em menos de 0,2% do tempo e não morde.
- **Exceto num canto: lento *e* de curso curto ao mesmo tempo.** Aí ele cobra
  pedágio, e cobra de quem se move mais: a 250 rad/s, apertar o curso de 6π para
  2π piora a MLP em 33% e o filtro alfa-beta em 1% — 176 resets por minuto
  contra 35. O preditor ainda ganha nesse canto por larga margem, então a
  hipótese original continua não valendo; o que vale é que o reset cobra
  proporcionalmente ao movimento pedido.

### 4. Dado de campo — a premissa que caiu

**Não existe nenhuma série pública de polarização em fibra terrestre
implantada.** Tudo o que é baixável é submarino e amostrado a 20 Hz ou menos
(Egersund–Yme, Med-Nautilus, EllaLink, Curie — todos CC-BY); os três melhores
trabalhos em fibra aérea (ORNL 15 km, NIST–UMD 62 km, LA 50 km) publicam as
estatísticas e retêm os dados. Mas as estatísticas publicadas bastam para
confrontar duas premissas da fase 3, e as duas estavam erradas.

`physics/field_calibration.py` registra cada número medido com a fonte:

| grandeza | valor | enlace | fonte |
|---|---|---|---|
| largura de linha de polarização | 60–80 nHz | 127 km enterrada, 36 dias | Czegledi *et al.*, Sci. Rep. **6**:21217 (2016) |
| conteúdo de banda estreita | **60 Hz + harmônicos de 120 e 180 Hz** | 220 km aéreo em cabo para-raios, polarímetro a 4 kHz | Peterson *et al.* (NIST) |
| expoente do espectro de deriva | **1,0 ± 0,2** | 15 km aérea+enterrada, 11 meses | Ramaswamy *et al.* (ORNL) |
| deriva diurna | dezenas de graus em segundos | 62 km, ~70% aérea | Shi *et al.* (NIST/UMD) |
| coeficiente Raman espontâneo | 1,5–3,1 × 10⁻⁹ /km/nm | 25 km monomodo, banda C | Kumar, Qin & Alléaume, NJP **17**:043027 (2015) |

**A frequência estava errada.** A fase 3 pôs a banda em 40 Hz. Em fibra aérea o
que existe é o tom da rede elétrica com harmônicos — a fibra vai dentro do cabo
para-raios e é a corrente da linha que a move.

**A taxa de deriva também.** O expoente espectral é função forte da taxa neste
modelo, e casar o 1,0 ± 0,2 do ORNL fixa a taxa em 0,015–0,020 rad/s contra os
0,05 usados.

Rerodando a fase 3 no ponto calibrado, com os preditores **retreinados lá**:

| compensador | ponto assumido (40 Hz) | ponto medido (60/120/180 Hz) |
|---|---:|---:|
| persistência | 0,1311 | 0,1226 |
| filtro alfa-beta | 0,0375 | 0,0524 |
| ridge (Wiener), retreinado | 0,0080 | 0,0133 |
| MLP, retreinado | 0,0078 | 0,0136 |
| MLP, **treinada a 40 Hz** | 0,0076 | **0,2594** |

- **A conclusão sobrevive.** O preditor continua batendo o filtro clássico por
  3,8× (contra 4,8× no ponto assumido), e mantém 99,9% do tempo abaixo do
  limiar contra 90%.
- **O artefato treinado não sobrevive.** Uma MLP treinada a 40 Hz rodando numa
  fibra de 60 Hz dá QBER 0,26 — pior que não compensar (0,12) e acima do limiar
  de 11% quatro quintos do tempo. O preditor aprende a fase de uma frequência
  específica; noutra frequência ele injeta erro. Retreinar não é opcional.
- **A calibração da taxa não mudou nada.** A 40 ms de latência a deriva térmica
  move dois miliradianos e a banda mecânica move trezentos. Vale registrar: o
  exercício fixou com precisão um parâmetro que não governa o resultado.

### 5. A amplitude não é conhecida — e não precisa ser, para esta conclusão

O tamanho do tom em radianos é o único parâmetro sem nenhuma medição. A busca
deu negativo limpo: o relatório da NIST dedica três frases ao tom e não dá
número, e as figuras são rasterizadas. Em vez de chutar, varri de 0,01 a 0,5 rad
— um fator de 50 — retreinando os preditores em cada ponto:

| amplitude (rad) | persistência | filtro alfa-beta | MLP | MLP ÷ alfa-beta |
|---:|---:|---:|---:|---:|
| 0,01 | 0,0016 | 0,0006 | 0,0002 | 2,6× |
| 0,05 | 0,0052 | 0,0020 | 0,0009 | 2,2× |
| 0,10 | 0,0163 | 0,0065 | 0,0026 | 2,5× |
| 0,30 | 0,1226 | 0,0524 | 0,0136 | 3,8× |
| 0,50 | 0,2783 | 0,1339 | 0,0311 | 4,3× |

A vantagem fica entre 2,2× e 4,3× em toda a faixa e **cresce** com a amplitude.
Abaixo de 0,05 rad todo mundo fica sob 1% de QBER e compensar deixa de ser
problema. Não existe amplitude plausível em que a resposta se inverta — o que a
medição decidiria é quanta chave o enlace produz, não se vale a pena usar um
preditor.

### 6. Um modelo cobre uma faixa estreita de frequências — e nada mais

Se o compensador treinado numa frequência é pior que nada em outra, cada enlace
precisa do seu polarímetro e da sua corrida de treino. Treinando **um** modelo
sobre 45/50/60/70 Hz:

| | 45 Hz | 50 Hz | 60 Hz | 70 Hz |
|---|---:|---:|---:|---:|
| MLP especialista (mesma freq.) | 0,0093 | 0,0095 | 0,0136 | 0,0136 |
| **MLP generalista** | 0,0105 | 0,0139 | 0,0150 | 0,0163 |
| ridge generalista | 0,0208 | 0,0513 | 0,0463 | 0,0250 |
| filtro alfa-beta | 0,0305 | 0,0430 | 0,0524 | 0,0349 |
| MLP especialista de outra freq. | 0,065 – 0,305 | | | |

O generalista paga no máximo 47% em relação ao especialista e continua batendo
o filtro clássico por pelo menos 2,1× — **dentro da faixa em que foi treinado**.

**Fora dela, não.** Testado em frequências retidas, o generalista estreito
interpola (0,0139 a 50 Hz) e não extrapola: a 75 Hz, cinco hertz acima da borda,
a QBER dele é **0,1087** — oito vezes pior e acima do filtro clássico. Alargar o
treino para 45–180 Hz destrói o modelo: ele fica pior que o filtro alfa-beta em
todos os cinco pontos de teste, inclusive em frequências que viu no treino, e
dobrar a largura da rede e os dados por ponto não muda nada.

| | 50 Hz | 75 Hz | 105 Hz | 165 Hz | 240 Hz |
|---|---:|---:|---:|---:|---:|
| MLP especialista | 0,0095 | 0,0134 | 0,0230 | 0,0305 | 0,0498 |
| MLP generalista 45–70 Hz | 0,0139 | 0,1087 | 0,0933 | 0,0589 | 0,0672 |
| MLP generalista 45–180 Hz | 0,0709 | 0,1014 | 0,0712 | 0,1135 | 0,0485 |
| ridge generalista 45–180 Hz | 0,0874 | 0,0299 | 0,0380 | 0,0429 | 0,0400 |
| filtro alfa-beta | 0,0430 | 0,0420 | 0,0506 | 0,0566 | 0,0477 |

**A amplitude, ao contrário, generaliza inteira.** Um modelo treinado em
0,05–0,30 rad empata com o especialista em 0,02, 0,10, 0,20 e até 0,50 rad, que
está fora do treino. Amplitude é escala; frequência é fase, e a janela de
observação de 32 ms cobre de 1,4 a 5,8 períodos conforme a frequência.

**Retreino por enlace continua sendo requisito** — mas o requisito real é
conhecer a frequência do tom a cerca de ±10 Hz e escolher, de uma família de
modelos de faixa estreita, o que a cobre.

Na mistura **estreita** a rede neural ganha do filtro de Wiener pela primeira
vez neste pacote: o ridge generalista fica pior que o filtro alfa-beta a 50 e
60 Hz e o MLP o bate por 3,7×, porque uma mistura de frequências não é um
processo gaussiano estacionário e o preditor precisa inferir *qual* frequência
está vendo. Na mistura **larga** isso se inverte — o ridge fica melhor que o MLP
em quatro dos cinco pontos. Nenhum dos dois é confiável fora do regime em que
foi ajustado, e o achado de que a não linearidade se paga vale só para a faixa
estreita.

O que continua sem medição é a amplitude. `datagen/measured.py` consome uma
gravação de polarímetro e devolve o modelo ajustado, declarando o que a taxa de
amostragem permitiu resolver e o que não.

```python
from qnet_autopilot.datagen.measured import load_stokes, fit_drift_model

registro = load_stokes("sop.csv")            # time_s, s1, s2, s3
ajuste = fit_drift_model(registro)
print(ajuste.resolved, ajuste.unresolved)    # o que dá e o que não dá para afirmar
modelo = ajuste.to_model()                   # sem banda que a gravação não viu
```

### 7. O agente dentro do laço de eventos do SeQUeNCe

A última lacuna de validação. `env/sequence_runner.py` monta uma rede SeQUeNCe
de verdade — roteadores quânticos com arranjos de memória, nós de medição de
Bell com detectores, canais quânticos com atenuação, canais clássicos com
atraso — e executa as seis primitivas da fase 2 instanciando os protocolos do
próprio simulador. O laço é para-e-anda: `Timeline.run` devolve o controle
quando a fila esvazia ou bate no horizonte, e devolve o evento à fila, então é
retomável. A política é a treinada, sem alteração; as features vêm dos mesmos
codificadores aplicados a um estado espelhado dos gerenciadores de memória do
SeQUeNCe. Em todos os passos de todos os episódios os dois simuladores
ofereceram **o mesmo conjunto de ações legais**.

**A probabilidade de heralding bate.** Contando todas as rodadas, inclusive as
de orçamentos que falharam:

| vão | rodadas | fórmula fechada | medido | σ |
|---:|---:|---:|---:|---:|
| 10 km | 1.277 | 0,0547 | 0,0627 ± 0,0070 | +1,1 |
| 20 km | 2.084 | 0,0345 | 0,0384 ± 0,0043 | +0,9 |
| 40 km | 4.867 | 0,0138 | 0,0164 ± 0,0018 | +1,5 |
| 80 km | 35.028 | 0,0022 | 0,0023 ± 0,0003 | +0,4 |

Sistematicamente ~10% acima, dentro de 1,5σ em cada vão — a fórmula é
ligeiramente conservadora.

**O que não bate é o tempo.** O twin cobra uma travessia do vão por tentativa;
o protocolo cobra **4,1**, porque uma tentativa é uma negociação e não um voo.
O fator é constante — 4,15 / 4,12 / 4,09 / 4,02 para 10, 20, 40 e 80 km — logo
multiplicativo, não um aperto de mão aditivo.

**E esse único número reconcilia os dois simuladores:**

| política | twin, custo ×1 | twin, custo ×4,1 | SeQUeNCe |
|---|---:|---:|---:|
| agente | 0,900 | 0,225 | 0,250 |
| greedy_fidelity | 0,350 | 0,300 | 0,250 |
| shortest_path | 0,350 | 0,300 | 0,250 |
| threshold_0,94 | 0,500 | 0,050 | 0,075 |

Não sobra efeito de escalonamento, mensagens ou contenção: sobra um parâmetro
errado por um fator de quatro.

**A consequência é a retratação da fase 2.** Retreinando com o custo medido:

| regime | agente | greedy_fidelity | shortest_path | threshold_0,94 |
|---|---:|---:|---:|---:|
| prazos publicados | 0,218 | 0,240 | 0,240 | 0,075 |
| prazos ×4 | 0,375 | **0,458** | 0,328 | 0,243 |

Nos prazos publicados o agente empata (−0,022 ± 0,013 contra `shortest_path`);
com prazos quatro vezes maiores ele perde para `greedy_fidelity` por
0,083 ± 0,020. A quatro vezes o custo por tentativa a geração domina o
orçamento, e as decisões de timing que o agente aprendeu valem menos que pegar
o que estiver disponível. O treino com prazos longos colapsou em duas de duas
sementes até o piso de entropia subir de 0,002 para 0,01.

**Dois achados de percurso.** O formalismo Bell-diagonal não representa um fóton
isolado, então Barrett-Kok quebra dentro do nó de medição e a geração que roda é
a `single_heralded` — a mesma cuja probabilidade a fase 1 validou. E
`MemoryArray.update_memory_params` é um `setattr` nu: `decoherence_rate` é
derivado de `coherence_time` só no construtor, então ajustar o tempo de coerência
depois não decoere nada e toda fidelidade da execução é a bruta. É o segundo
parâmetro que o SeQUeNCe aceita e descarta em silêncio.

### 8. Bloco de duração variável: a medição anterior estava errada

**Retratação.** Uma versão anterior deste README dizia que o bloco adaptativo
tinha sido medido e não valia a pena — 1,28× no melhor caso, "use blocos longos
e pronto". Aquele número saiu de um oráculo que só procurava entre *durações
fixas por execução*, e a mais longa que ele podia escolher era dez segundos. Um
oráculo assim não pode encontrar a única coisa que um bloco variável faz de
diferente: recortar a janela boa de dentro de uma execução ruim. Trocado por
programação dinâmica sobre partições (`physics/blocks.py`, exata porque a chave
de um bloco só depende dos seus extremos), o teto é outro.

O que muda: erros são cobrados contra o bloco inteiro, então um trecho de fibra
ruim envenena cada bit que divide o bloco com ele. No enlace bem compensado isso
não acontece e o melhor bloco é um só, do tamanho da execução. No enlace ruim
acontece o tempo todo, e **nenhum temporizador de nenhuma duração produz um
único bit**.

Três estratégias, 12 sementes de teste, chave dividida pela execução inteira
(descartar não maquia nada):

| compensador | QBER | melhor fixo | fixo 1 s | controlador | oráculo | do oráculo |
|---|---|---|---|---|---|---|
| persistência | 0,134 | 518 Hz (0,1 s) | 212 Hz | **983 Hz** | 1305 Hz | 75% |
| filtro alfa-beta | 0,039 | 39 987 Hz (20 s) | 27 217 Hz | 38 783 Hz | 39 997 Hz | 97% |
| ridge (Wiener) | 0,007 | 73 029 Hz (20 s) | 54 776 Hz | 72 625 Hz | 73 084 Hz | 99,4% |
| MLP | 0,007 | 73 520 Hz (20 s) | 55 188 Hz | 73 110 Hz | 73 577 Hz | 99,4% |

O controlador é causal: decide sobre estimativas binomiais dos bits que o
protocolo publica para estimação de parâmetro, e é pontuado sobre a verdade. Não
tem constante de ajuste. A cada instante ele compara duas continuações —
encerrar agora e rodar no melhor regime estacionário que o enlace demonstrou, ou
seguir com o bloco até o melhor ponto de corte futuro — e faz o que vale mais. É
essa comparação por horizonte que produz os dois comportamentos: no enlace
quieto o futuro é bom, reiniciar não ganha nada e ele converge para um bloco
longo; no enlace ruim o futuro não vale nada, a única coisa em jogo é o bloco na
mão, e ele corta e descarta.

A regra míope óbvia — encerrar quando o segmento novo derrubaria a chave do
bloco — foi implementada primeiro e é mensuravelmente pior: encerra em cima do
ruído da estimativa e entrega um quinto da chave no enlace bom.

**E isso importa no compensador bom?** Importa, quando o enlace fica difícil.
Mesmo MLP, mesma varredura de amplitude de vibração da seção 6:

| amplitude | QBER | melhor fixo | fixo 1 s | controlador | sobre o melhor fixo |
|---|---|---|---|---|---|
| 0,3 rad | 0,008 | 72 781 Hz (20 s) | 54 576 Hz | 72 037 Hz | 0,99× |
| 0,6 rad | 0,051 | 29 914 Hz (20 s) | 21 792 Hz | 29 763 Hz | 1,00× |
| 0,9 rad | 0,202 | 2110 Hz (0,1 s) | 401 Hz | 3325 Hz | **1,58×** |
| 1,2 rad | 0,322 | 314 Hz (0,1 s) | **0 Hz** | 437 Hz | **1,39×** |

Abaixo do ponto de operação da fase 3 o controlador custa 1% e não serve para
nada. Acima de três vezes a amplitude ele bate o melhor temporizador escolhido
com retrospectiva por 1,4–1,6×, e o temporizador de um segundo que as fases 3 e
4 assumiram entrega zero. A conclusão honesta não é "use blocos longos": é que a
duração certa do bloco depende do enlace, ninguém a conhece de antemão, e o
controlador a encontra sozinho — de graça no enlace fácil, e é a diferença entre
chave e nada no enlace difícil.

### Reproduzir

```bash
python scripts/evaluate_realism.py --out runs/realism
python scripts/evaluate_actuator.py --out runs/actuator
python scripts/evaluate_field.py    --out runs/field   # inclui amplitude e robustez
python scripts/evaluate_sequence.py --out runs/sequence
python -c "from qnet_autopilot.validation.realism_report import render_realism_report; \
           render_realism_report('runs/realism','runs/actuator','reports/realism.html', \
                                 allocator_directory='runs/allocator', \
                                 field_directory='runs/field', \
                                 sequence_directory='runs/sequence')"
```

---

## Fase 5 — QKD com decoy states: espionagem, taxa de chave e sincronismo

Quatro perguntas que um enlace de QKD implantado faz todo dia, e que o resto
deste pacote não respondia porque assumia fonte de pares emaranhados, não laser
atenuado. O resultado que resume a fase: **o limiar de QBER, que é o que sistemas
implantados usam hoje, enxerga um dos quatro ataques modelados**. Um classificador
sobre os mesmos observáveis públicos enxerga os quatro, separa ataque de deriva
térmica, e diz qual é qual.

Os números abaixo são da segunda rodada, em que os ataques deixaram de entrar
sempre como degrau isolado: agora entram em rampa ou em rajada, e 40% deles
rodam **durante** uma excursão térmica, que é o caso adverso de verdade. O
problema ficou mais difícil e os números caíram — o F1 macro de 0,93 para 0,83 —
e é a versão honesta.

Relatório completo: `reports/phase5.html`.

### A física primeiro, como nas outras fases

Uma fonte de laser atenuado emite um número de fótons de Poisson por pulso, e é
disso que nasce tanto o ataque de separação de número de fótons (PNS) quanto a
defesa contra ele. `physics/decoy.py` é escrito em torno de **yields por número de
fótons** — `Y_n`, a probabilidade de o Bob registrar um clique dado que a Alice
mandou *n* fótons — e não em torno de uma taxa de erro global:

| | |
|---|---|
| ganho | `Q_μ = Σ_n Poisson(n\|μ) Y_n`; no canal honesto, `1 − (1−Y₀)e^{−ημ}` |
| erro | `E_μ Q_μ = e₀Y₀ + e_d(1 − e^{−ημ})`, com `e₀ = ½` por física, não por escolha |
| cotas | `Y₁^L` e `e₁^U` pelo protocolo vácuo + decoy fraco (Ma, Qi, Zhao & Lo, PRA 72 012326, 2005) |
| chave | GLLP: `R ≥ q[Q₁^L(1 − h(e₁^U)) − f·Q_μ·h(E_μ)]` |
| estatística | `sample_observables` sorteia as contagens binomiais de um bloco finito; as cotas rodam em cima delas, como o protocolo real roda |

A consequência dessa escolha é o que dá valor ao resto: **um ataque é escrito como
uma modificação dos `Y_n`**, e os observáveis saem disso pelo mesmo caminho de
código que um canal honesto percorre. Ataque sem assinatura física não ganha
assinatura no dataset — e um dos quatro não tem nenhuma, de propósito.

| ataque | o que a Eva faz | o que se mexe |
|---|---|---|
| intercept-and-resend | mede numa base aleatória e reenvia | só o erro: `e' = ½e + ¼` nos pulsos atacados; ganho intacto |
| PNS | bloqueia pulsos de um fóton, repassa os multi-fóton por canal melhor | `Q_μ` **exatamente** intacto, `E_μ` intacto a 3 partes em 10⁶; `Q_ν` cai, e `Y₁^L` desaba |
| cegamento de detector | luz forte põe os detectores em modo linear e força os cliques | ganho **acima** do que a perda permite, e pulso vazio começa a clicar |
| divisor de feixe | fica com a luz que a perda do canal já entrega | **nada** — e o modelo não finge que sim |

As degradações que não são ataque vêm da física que as outras fases já tinham: a
deriva térmica usa o `PolarizationDriftModel` da fase 1 com o resíduo que um
compensador da fase 3 deixa, e o evento de perda é decibel a mais no vão.

### O problema, e por que o termômetro entra

Cada episódio é um enlace rodando 10 minutos, acertando contas a cada segundo
(10⁹ pulsos por período a 1 GHz), com a fibra derivando embaixo e um evento
começando num instante que ninguém anunciou. O primeiro minuto é garantidamente
limpo — é a calibração de comissionamento, que todo enlace implantado tem —, e
**todo atributo é relativo a ela**: ganho absoluto é quase só uma afirmação sobre
o comprimento do vão.

A classe difícil é a deriva térmica, e o discriminador é físico: a taxa de deriva
da birrefringência é acoplada a **dT/dt**, então uma excursão térmica produz um
erro que sobe enquanto a temperatura *se move* e assenta quando ela para. A Eva
não tem essa correlação. É por isso que a correlação entre o erro e |dT/dt| na
janela móvel é um atributo, e é o único que um limiar de QBER não pode ter.

### Resultados — detecção (168 episódios, separação por episódio)

24 episódios por classe, vãos de 20 a 70 km, 90 720 períodos de acerto, 70% dos
episódios para treino — dos quais 20% ficam de fora do ajuste, reservados para
calibrar o limiar de decisão. O limiar de QBER está calibrado para 1% de alarme
falso no mesmo conjunto — é o que um operador faria com ele.

| classe | limiar de QBER | random forest | SVM (RBF) | boosting | boosting + limiar + histerese |
|---|---|---|---|---|---|
| normal | 0,993 | 0,948 | 0,924 | 0,938 | **0,968** |
| deriva térmica | 0,000 | 0,864 | 0,651 | **0,875** | 0,863 |
| evento de perda | 0,000 | **1,000** | 0,950 | **1,000** | 0,992 |
| intercept-and-resend | 0,000 | **0,636** | 0,602 | 0,597 | 0,513 |
| PNS | 0,000 | **0,749** | 0,416 | 0,745 | 0,709 |
| cegamento de detector | 0,000 | 0,840 | 0,831 | **0,908** | 0,792 |

| detector | F1 macro | alarme falso | alarmes/hora | precisão a 1% de prior |
|---|---|---|---|---|
| limiar de QBER | 0,128 | 0,0073 | 1,6 | 0,103 |
| random forest | **0,832** | 0,0525 | 8,9 | 0,155 |
| SVM (RBF) | 0,702 | 0,0761 | 36,4 | 0,113 |
| gradient boosting | 0,830 | 0,0618 | 14,9 | 0,135 |
| boosting + limiar de decisão | 0,825 | 0,0373 | 11,0 | 0,200 |
| boosting + limiar + histerese | 0,818 | **0,0321** | **6,8** | **0,220** |

O limiar zera a coluna de recall porque não tem tipo para dar: ele responde
"ataque" ou "normal". Mesmo na versão binária, **ele detecta um dos quatro
ataques** — o intercept-and-resend, e agora só 27% dos períodos, porque metade
deles são de ataques que entram em rampa ou em rajada. PNS (4%), cegamento (11%)
e evento de perda (3%) seguem invisíveis, porque nenhum mexe no QBER. Essa é a
afirmação central da fase, e é consequência da física, não do classificador.

**Duas métricas que não são acurácia, e por quê.** As classes aqui estão
balanceadas porque o gerador as balanceou; num enlace real, ataque é raro.
"Alarmes por hora" conta *inícios* de alarme em períodos comprovadamente limpos —
um alarme que fica de pé um minuto é um chamado, não sessenta. "Precisão a 1% de
prior" responde à pergunta do operador: se ataque ocupa 1% do tempo, que fração
dos alarmes é real? A 6,8 alarmes falsos por hora, a resposta é 22%. A seção
sobre taxa de decisão mostra que isso não é uma limitação do modelo.

**O controle negativo.** O divisor de feixe não deixa assinatura nenhuma, e a
resposta certa nos seus períodos é "normal". A floresta e o boosting respondem
isso em 91% deles; o SVM, em 83% — as fronteiras suaves inventam ataque onde não
há nada para ver. Quem relatar detecção acima do acaso nessa classe aprendeu um
artefato do gerador.

### Os casos difíceis: rampa, rajada e co-ocorrência

Um gerador em que todo ataque entra de uma vez, em pleno vigor, e com mais nada
acontecendo é o caso favorável em dois sentidos. Quem sabe que o enlace é
monitorado entra devagar ou trabalha em rajadas; e o pior caso para um monitor é
o ataque que começa **durante** uma excursão térmica, quando a explicação
inocente para o erro subindo já é verdadeira. As duas coisas estão no gerador:
todo ataque sorteia um perfil temporal (40% degrau, 35% rampa, 25% intermitente)
e 40% deles carregam uma excursão térmica simultânea.

| caso (só períodos com ataque em curso) | períodos | recall | detectado como algo |
|---|---|---|---|
| degrau | 4 449 | 0,944 | 0,994 |
| rampa | 2 678 | 0,778 | 0,970 |
| intermitente | 1 811 | 0,562 | 0,834 |
| com excursão térmica simultânea | 2 685 | 0,631 | **1,000** |
| sem excursão térmica | 6 253 | 0,897 | 0,935 |

O intermitente é o que mais dói, e a razão é mecânica: metade dos períodos
rotulados como ataque são os intervalos entre rajadas, em que literalmente não há
nada para ver. A rampa perde os primeiros períodos, que é o comportamento certo —
naquele momento o ataque é fraco demais para se distinguir do ruído.

A co-ocorrência térmica é a mais interessante: o ataque **sempre** é detectado
como *alguma coisa* (1,000), mas o tipo certo sai em 63% dos períodos contra 90%
sem excursão. Ou seja, a excursão térmica não esconde o ataque; ela rouba o
rótulo. Para um operador isso é uma degradação real mas recuperável — o alarme
sobe, o diagnóstico é que fica ambíguo.

Comparado ao gerador só-degrau da primeira versão, o F1 macro cai de 0,93 para
0,83 e o alarme falso sobe de 1,4% para 5,3%. **O número antigo media um problema
mais fácil.**

### A taxa de decisão, e não o modelo

A precisão sob prior realista a um segundo de período é ruim para todos os
modelos, e a causa não é o classificador: é que o enlace publica 3 600 decisões
por hora. Repetindo o estudo com **período de acerto de 5 s** — mesmas classes,
mesmo código, contagens cinco vezes maiores por decisão, metade dos episódios —
o quadro muda de natureza:

| detector | F1 macro | alarme falso | alarmes/hora | precisão a 1% |
|---|---|---|---|---|
| limiar de QBER | 0,114 | 0,0181 | 0,5 | 0,012 |
| random forest | 0,760 | 0,0798 | 5,0 | 0,103 |
| gradient boosting | 0,747 | 0,0843 | 3,9 | 0,101 |
| boosting + limiar de decisão | 0,718 | **0,0001** | **0,1** | **0,982** |
| boosting + limiar + histerese | 0,694 | **0,0000** | **0,0** | **1,000** |

Com o limiar de decisão calibrado em episódios que nenhum modelo viu, o alarme
falso vai praticamente a zero e a precisão sob prior realista passa de 0,22 para
~1,0. O que se paga é **atraso** — dezenas de segundos em vez de um — e o tipo do
intercept-and-resend fraco, que passa a ser reportado como deriva térmica (recall
0,000, mas ainda detectado como algo em 46% dos períodos). É uma escolha de ponto
de operação, não uma melhoria: um enlace que precisa reagir em segundos fica com
o período curto e paga em chamados.

Uma armadilha que essa parte revelou e vale registrar: a primeira versão
calibrava o limiar de decisão nas mesmas linhas em que o modelo foi ajustado. O
boosting sinaliza quase nenhum período limpo do *treino* — ele os memorizou — e
sinaliza 6% dos retidos, então o limiar saía em zero e o alarme falso **piorava**.
Calibrar em 20% dos episódios de treino, fora do ajuste, é o que faz funcionar.

### Qual física carrega a separação

Três ablações, cada uma removendo um bloco de atributos da floresta:

| variante | F1 macro | alarme falso | PNS | deriva térmica | intercept-and-resend |
|---|---|---|---|---|---|
| completa | 0,832 | 0,053 | 0,749 | 0,864 | 0,636 |
| sem termômetro | 0,768 | 0,129 | 0,753 | **0,425** | 0,536 |
| sem observáveis de decoy | 0,770 | 0,050 | **0,400** | 0,881 | 0,620 |
| só QBER e ganho | 0,622 | 0,161 | **0,098** | 0,390 | 0,542 |

Sem os observáveis de decoy state o PNS cai de 0,75 para 0,40, e com só QBER e
ganho vai a 0,10. Isso é a afirmação de segurança do protocolo reaparecendo como
resultado de aprendizado: sem decoy, PNS é quase invisível. Sem o termômetro, a
deriva térmica cai pela metade e o alarme falso dobra e meio — deriva térmica e
intercept-and-resend produzem a mesma subida de QBER e se distinguem pela causa.

A importância dos atributos diz a mesma coisa em outra ordem: `temperature_shift`
(0,134), `log_gain_versus_calibration` (0,100) e `log_signal_gain_ratio` (0,096)
lideram, seguidos pelos dois de decoy — termômetro, calibração e decoy, que são
exatamente as três ablações que doem.

### Resultados — taxa de chave (agente × comissionamento × oráculo)

O transmissor tem quatro botões que não exigem tocar no hardware: μ, ν, a divisão
dos pulsos e o clock. O ponto ótimo se move com a perda do vão, com o
desalinhamento que sobrou da compensação, e com o quanto os detectores estão
perto de saturar — o tempo morto de 50 ns e o erro de intersímbolo que a dispersão
impõe ao clock são o que impede "subir o clock" de ser de graça.

A recompensa é a chave que se consegue **provar**, calculada sobre as contagens
finitas do período — não a taxa assintótica. Onde essa troca cai é medição, e a
medição contrariou a expectativa: num bloco de um segundo a 1 GHz com um quinto
dos pulsos no decoy, mesmo ν = 0,01 produz dezenas de milhares de cliques, e **o
agente empurra o decoy para o piso da faixa** (0,010 contra os 0,1 do
comissionamento) exatamente como a fórmula assintótica manda.

Agente generalista (treinado sorteando o vão entre 25 e 80 km), 24 episódios de
teste por vão, sementes disjuntas das de treino e das de validação:

| vão | estático (comissionamento) | agente | oráculo | ganho | fração da lacuna |
|---|---|---|---|---|---|
| 25 km | 3,180 Mbit/s | **3,782** | 3,817 | **+18,9%** | 94% |
| 35 km | 2,732 | **3,092** | 3,059 | +13,2% | 110% |
| 50 km | 1,899 | **2,128** | 2,152 | +12,1% | 91% |
| 65 km | 1,038 | **1,138** | 1,220 | +9,6% | 55% |
| 80 km | 0,489 | **0,532** | 0,713 | +8,8% | 19% |

O agente ganha em todos os vãos, e até 50 km recupera mais de 90% da lacuna até o
oráculo. Em 35 km ele **passa** do oráculo, e isso não é erro de medição: o
oráculo olha uma ação à frente e escolhe a localmente melhor, então não consegue
trocar um período pior por um próximo melhor, e otimiza a taxa assintótica
enquanto a pontuação é a taxa provável sobre contagem finita. Ele é uma
referência, não um teto.

**Duas correções levaram a esse resultado, e a primeira versão perdia 14,7% em
80 km.** Vale registrar as duas:

1. **O agente não sabia em que regime estava.** Todos os atributos eram relativos
   à calibração, e uma queda de 20% no ganho é idêntica em 25 e em 80 km — mas os
   dois regimes querem movimentos opostos, porque em vão curto os detectores
   saturam e em vão longo a cota de um fóton passa fome. Duas entradas absolutas
   (log do ganho medido e log da taxa de comissionamento) resolveram: 80 km passou
   de −14,7% para +1,6% só com isso.
2. **A temperatura de amostragem não tinha sido varrida.** A fase 2 mediu que
   greedificar uma política treinada por A2C custa um terço do desempenho; aqui,
   com ótimo raso em vão longo, a resposta é o contrário — 0,25 bate 1,0 em quatro
   dos cinco vãos. A varredura roda em sementes de validação disjuntas do teste e
   escolhe **uma** temperatura para toda a política, porque um sistema implantado
   tem uma.

Antes das duas, o agente treinado só em 50 km recuperava toda a lacuna ali e
**colapsava 86% em 80 km**; sortear o vão por episódio no treino corrigiu o
colapso, e as duas correções acima fecharam o resto.

### Resultados — sincronização temporal

A sílica muda de índice e de comprimento com a temperatura: cerca de 40 ps por
quilômetro por kelvin. Em 50 km, **um kelvin são dois nanossegundos** — dez vezes
a janela de um detector gated. A estrutura do estudo é deliberadamente a mesma da
fase 3: modelo de deriva, canal de monitoração com fóton finito (a média do
histograma de chegada tem erro `σ/√N`), preditores que vão de "não fazer nada" a
um aprendido, laço fechado com latência, e a pontuação é o que sobra — eficiência
de detecção e taxa de chave, não o RMS do desalinhamento.

Taxa de chave a 50 km, por largura da janela e por detector (Mbit/s, média de 12
execuções de uma hora):

| escuro | janela | sem correção | recalibração 60 s | filtro alfa-beta | ridge + termômetro |
|---|---|---|---|---|---|
| 10 Hz (SNSPD) | 200 ps | 0,136 | 1,001 | 1,464 | **1,508** |
| | 500 ps | 0,309 | 1,631 | 1,958 | **1,971** |
| | 1000 ps | 0,530 | 1,924 | **2,007** | **2,007** |
| 10⁵ Hz | 200 ps | 0,071 | 0,936 | 1,408 | **1,452** |
| | 500 ps | 0,151 | 1,500 | 1,844 | **1,857** |
| | 1000 ps | 0,236 | 1,696 | 1,787 | 1,787 |
| 10⁶ Hz (InGaAs livre) | 200 ps | 0,000 | 0,429 | 0,948 | **0,998** |
| | 350 ps | 0,000 | 0,536 | 1,064 | **1,099** |
| | 500 ps | 0,000 | 0,528 | 0,960 | 0,977 |
| | 1000 ps | 0,000 | 0,128 | 0,253 | 0,253 |

Quatro coisas que essa tabela diz, e que valem mais do que o placar:

1. **Não rastrear não é uma opção.** Um enlace calibrado e deixado quieto perde
   85% da chave em uma hora com detector silencioso, e **toda** ela com detector
   ruidoso. Isso não é um resultado de aprendizado de máquina, é o coeficiente
   termo-óptico da sílica.
2. **Recalibrar periodicamente custa em dois lugares** — a janela fica velha entre
   varreduras, e a varredura em si para o enlace. A 60 s de intervalo ela entrega
   83% do que o rastreamento contínuo entrega, e cai para 54% com detector
   ruidoso, onde a janela estreita que ela não consegue sustentar é justamente o
   que faria diferença.
3. **A largura ótima da janela depende do detector, e só existe quando o escuro
   importa.** Com SNSPD, alargar é quase de graça e o melhor é a janela mais
   larga. A 10⁶ contagens por segundo o ótimo é 350 ps, e uma janela de 1 ns
   custa **77%** da chave. Em 80 km com esse detector não há chave em largura
   nenhuma.
4. **A margem do modelo aprendido é modesta e aparece onde deveria**: em janela
   estreita e detector ruidoso (+5,3% sobre o filtro alfa-beta a 200 ps e 10⁶ Hz),
   e some quando a janela é larga o bastante para o resíduo não importar. Ele usa
   o termômetro, e a componente térmica é proporcional à *variação* de
   temperatura — exatamente o que um filtro alfa-beta, que só vê a série do
   atraso, não pode inferir.

### Resultados — compensador sequencial de polarização

O compensador aprendido da fase 3 é um MLP sobre uma janela **achatada**: os
quinze incrementos de rotação passados chegam como um vetor único de 45 entradas,
e a primeira camada precisa aprender do zero que as posições 0-2, 3-5 e 6-8 são a
mesma grandeza em atrasos sucessivos. Uma convolução causal já sabe disso — o
mesmo filtro de três canais é aplicado em todo atraso — e é o prior certo para a
banda de vibração, que é a mesma forma onde quer que ela caia na janela.

`compensation/sequence.py` implementa uma pilha de convoluções causais dilatadas
em numpy, com backward escrito à mão e conferido contra diferenças finitas. A
interface não muda: ela expõe `predict(X) -> (n, 3)` sobre a **mesma matriz de
atributos** que `compensation/train.build_dataset` já produz, entra no mesmo
`LearnedCompensator` e é pontuada pelo mesmo laço fechado. Só a arquitetura muda.

**O campo receptivo é aritmética, não hiperparâmetro** — e foi onde a primeira
tentativa errou:

| modelo | campo receptivo | parâmetros | MSE de validação |
|---|---|---|---|
| MLP (192, 192) — fase 3 | 15 | 47 043 | **3 438** |
| TCN c32, dilatações (1, 2) | 7 | 5 923 | 7 035 |
| TCN c32, dilatações (1, 2, 4) | 15 | 9 027 | 3 583 |
| TCN c48, dilatações (1, 2, 4) | 15 | 17 923 | 3 586 |
| TCN c64, dilatações (1, 2, 4) | 15 | 29 891 | 3 592 |

Com duas camadas dilatadas a pilha alcança sete atrasos. Um período da banda de
vibração a 40 Hz são 25 ms, e o período de controle é 2 ms: **12,5 passos**. Sete
atrasos não cobrem uma oscilação inteira, e o modelo fica com o dobro do erro.
Com três camadas o campo chega a quinze, cobre o período, e o erro cai pela
metade. Capacidade não compra nada depois disso — triplicar os canais move o MSE
em 0,2%, na direção errada.

No laço fechado da fase 3 (12 execuções de 20 s, sementes disjuntas do treino,
fibra aérea com banda de vibração forte e atuador ideal):

| compensador | QBER médio | QBER p95 | uptime | chave finita | resíduo |
|---|---|---|---|---|---|
| MLP (fase 3) | **0,0071** | 0,0191 | 0,9988 | **55 145 bit/s** | 185 mrad |
| ridge (Wiener) | 0,0074 | 0,0200 | 0,9987 | 54 722 | 191 mrad |
| **TCN (fase 5)** | 0,0075 | 0,0203 | 0,9987 | 54 636 | 192 mrad |
| filtro alfa-beta | 0,0417 | 0,1083 | 0,9512 | 25 183 | 469 mrad |
| sem compensação | 0,1425 | 0,3582 | 0,5417 | 6 323 | 895 mrad |
| persistência | 0,1344 | 0,3330 | 0,5012 | 217 | 880 mrad |
| extrapolação linear | 0,3707 | 0,7955 | 0,1715 | 0 | 1 730 mrad |

A leitura honesta: **a convolução causal iguala o MLP com um quinto dos
parâmetros, e não o supera.** O prior de equivariância temporal economiza
capacidade, não erro. Para um laço que roda a um quilohertz num controlador
embarcado isso é uma vantagem real — cinco vezes menos multiplicações por passo —
mas quem esperava que "usar CNN/LSTM" melhorasse a previsão em si não é o que
está medido aqui. O que domina o erro é o campo receptivo, e o MLP da fase 3 já
tinha o dele certo por construção, porque via a janela inteira.

### Reproduzir

```bash
python scripts/train_detector.py   --out runs/detector --episodes 24
python scripts/train_detector.py   --out runs/detector_slow --episodes 12 \
                                   --period 5 --duration 3000   # taxa de decisão
python scripts/train_keyrate.py    --out runs/keyrate  --km 25 35 50 65 80 \
                                   --iterations 200 --tag generalist
python scripts/evaluate_phase5.py  --part all --out runs/phase5 \
                                   --policy runs/keyrate/generalist_policy.npz \
                                   --spans 25 35 50 65 80
python -c "from qnet_autopilot.qkd.report import render_phase5_report; \
           render_phase5_report('runs/detector','runs/phase5','reports/phase5.html', \
                                slow_detector_dir='runs/detector_slow')"
```

`--step-only` no `train_detector.py` reproduz o gerador antigo, em que todo
ataque entra de uma vez — é a comparação que mostra o quanto o problema fácil
inflava os números. O ponto de operação inteiro está em `scripts/task5.py`, como
nas fases anteriores, e `python -m qnet_autopilot decoy --optimise` imprime o
ponto de comissionamento de um vão sem treinar nada.

### Limites deste resultado

- **O gerador é o modelo.** Todos os números acima são sobre sessões simuladas
  pelo `physics/decoy.py` deste pacote. As cotas são as publicadas e os ataques
  são escritos como modificação dos yields, o que impede as trapaças mais óbvias,
  mas nenhuma fibra real foi medida. Um classificador treinado aqui é ponto de
  partida para um enlace real, não um detector pronto.
- **Um ataque por episódio.** Rampa, rajada e co-ocorrência com excursão térmica
  estão cobertas; dois ataques simultâneos, ou um ataque durante uma curvatura,
  não estão. O segundo é o caso que eu esperaria quebrar a separação entre
  "evento de perda" e "PNS", já que os dois derrubam ganho.
- **A precisão sob prior realista depende do período de acerto, e o estudo tem só
  dois pontos** (1 s e 5 s, e o de 5 s com metade dos episódios). A curva entre
  eles, e o ponto em que o atraso deixa de ser aceitável, não foram mapeados.
- **O agente de taxa de chave ainda perde a maior parte da lacuna em vão longo**
  (recupera 19% dela em 80 km). O diagnóstico é que o ótimo é raso e a cota de um
  fóton é ruidosa ali; um agente com memória mais longa que um período, ou um
  período de acerto maior no próprio laço de controle, são os caminhos plausíveis
  e não foram testados.
- **O oráculo é referência, não teto** — olha uma ação à frente e otimiza a taxa
  assintótica, e o agente o supera em 35 km. Um teto de verdade exigiria
  programação dinâmica sobre o episódio inteiro contra a recompensa amostrada.
- **A sincronização assume clock recuperado.** O offset fracionário de 10⁻¹²
  corresponde a um clock recuperado do canal clássico que divide a fibra, que é o
  que um sistema implantado faz. Com oscilador livre (10⁻¹⁰) o termo linear domina
  e o problema vira trivial para qualquer rastreador — o que torna o resultado
  *menos* interessante, não mais.
- **Nada disso passou pelo laço de eventos do SeQUeNCe.** O runner de eventos
  discretos da fase 2 cobre geração de emaranhamento, não BB84 prepare-and-measure:
  o SeQUeNCe não modela decoy states nem ataques nativamente. A camada de física
  aqui é fechada e validada contra as fórmulas publicadas, não contra simulação de
  eventos.

---

## O que viria depois

As quatro capacidades pedidas para a rede de emaranhamento estão implementadas,
medidas e documentadas; três das quatro lacunas de aplicabilidade estão fechadas
acima; e a camada de QKD da fase 5 acrescenta outras quatro. O que falta:

- **A amplitude do tom de rede, medida.** Frequência, taxa de deriva e
  coeficiente Raman vêm de medições publicadas; a amplitude não. A varredura
  mostra que a escolha entre compensadores não depende dela, mas quanta chave o
  enlace produz depende, e é isso que se dimensiona. Um polarímetro acima de
  400 Hz e uma semana de gravação fecham — o ajuste que consome o arquivo já
  existe. Duas pistas bibliográficas para quem tiver acesso pago: Waddy, Chen &
  Bao (IEEE PTL, *"Polarization oscillations in aerial fiber caused by wind and
  power-line current"*) e a referência [2] do relatório da NIST.
- **Um compensador que funcione numa faixa larga de frequências.** Medido: não
  existe com esta janela de observação e este conjunto de features, e capacidade
  não resolve. O caminho provável é dar ao modelo a frequência estimada como
  entrada explícita, ou alongar a janela até cobrir vários períodos da menor
  frequência de interesse.
- **Contenção entre requisições concorrentes.** O runner de eventos discretos
  dirige os protocolos diretamente e não usa o motor de regras do SeQUeNCe, que
  é onde mora o efeito de várias requisições disputando as mesmas memórias.
- **Um atuador lento *e* de curso curto ao mesmo tempo.** É o único canto onde a
  hipótese do reset ainda poderia se sustentar, e não foi varrido.
- **O controlador de bloco contra um oráculo mais forte.** O oráculo atual
  particiona a execução em *intervalos*. Nada na amplificação de privacidade
  exige que um bloco seja contíguo no tempo: um receptor pode juntar as janelas
  boas de trechos separados no mesmo bloco, e esse oráculo (um prefixo dos
  segmentos ordenados por erro) é estritamente maior. Quanto do que sobra o
  controlador alcança nessa formulação não foi medido.

Além disso, para a fase 2: treinar sobre uma *distribuição* de topologias em vez
de uma só.

Nota de ambiente: o container onde isto foi desenvolvido não tem PyTorch e não
consegue instalá-lo. Por isso o agente é numpy nativo, com a interface
`agents.base.Policy` como costura — plugar uma implementação em Torch na sua
máquina é escrever uma classe com dois métodos, sem tocar no ambiente, nos
baselines nem na avaliação.
