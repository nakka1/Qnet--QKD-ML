"""The phase-4 benchmark: a hybrid metro plant where classical traffic costs key.

One place for the operating point, so tuning, evaluation and the report cannot
drift apart.

Why these numbers
-----------------
**The plant** is the phase-1 ``metro`` scenario — the Sao Paulo-Rio corridor,
nine nodes and eleven spans from 30 to 145 km. The span spread is the point: at
any given classical launch power, a 30 km link barely notices the Raman
background and a 145 km link is destroyed by it, so a single global admission
setting is wrong everywhere at once and the decision has to be per link.

**The launch power** is -5 dBm per link at full admission. Published coexistence
experiments run classical channels at roughly -10 to -20 dBm precisely because
0 dBm swamps the quantum channel, and the measurements bear that out here: at
0 dBm even a quarter admission kills every span longer than about 90 km, so the
dial degenerates to on/off and there is nothing to schedule. At -10 dBm the
opposite happens — everything survives full admission and there is nothing to
schedule either. -5 dBm is where the dial has graded consequences and where the
drift level decides which links can afford classical traffic. It is swept.

**The drift** is what phase 3's compensator leaves, not a bare fibre: this
allocator sits on top of that loop. ``COMPENSATED`` is calibrated to the
residual the phase-3 ridge compensator actually achieved; ``DEGRADED`` is a link
whose loop is running but losing. A third of links are drawn degraded each
episode, and which ones is redrawn per episode so that span length and drift
quality are not collinear.

**The load** is set so neither end of the trade-off saturates: silencing the
classical channel serves about 73% of the offered key, admitting everything
serves about 19%.
"""
from __future__ import annotations

import numpy as np

from qnet_autopilot.allocation import AllocationConfig, HybridAllocationEnv, build_demands
from qnet_autopilot.network import scenario

SCENARIO = "metro"
CONTROL_PERIOD_S = 5.0
DURATION_S = 1800.0

#: -5 dBm at full admission. Swept in the report.
CLASSICAL_POWER_W = 3.0e-4

#: Control periods the forecaster predicts ahead. Thirty is 150 seconds. The
#: residual's autocorrelation is long gone by then, so the far end of the
#: forecast carries little — but the horizon has to **exceed the longest
#: reconfiguration latency swept**, because a decision that lands after the
#: forecast ends cannot be evaluated against it, and the policy refuses that
#: configuration rather than quietly ignoring its own forecast.
FORECAST_HORIZON = 30
FORECAST_WINDOW = 6

TRAIN_SEEDS = tuple(range(0, 12))
VALIDATION_SEEDS = tuple(range(100, 106))
TEST_SEEDS = tuple(range(500, 512))


def make_config(**overrides) -> AllocationConfig:
    parameters = dict(control_period_s=CONTROL_PERIOD_S, duration_s=DURATION_S,
                      classical_power_full_w=CLASSICAL_POWER_W)
    parameters.update(overrides)
    return AllocationConfig(**parameters)


def make_env(seed: int = 0, load_scale: float = 1.0, config=None,
             **config_overrides) -> HybridAllocationEnv:
    """Build the benchmark environment with its demand stream.

    The demand seed is derived from, but not equal to, the episode seed: the
    fibre and the traffic should vary together across episodes without one
    being a function of the other.
    """
    graph = scenario(SCENARIO)
    demands = build_demands(graph, duration_s=DURATION_S, seed=seed,
                            load_scale=load_scale)
    return HybridAllocationEnv(graph, demands,
                               config or make_config(**config_overrides))
