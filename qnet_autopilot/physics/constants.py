"""Physical constants and default hardware parameters, with their provenance.

Two kinds of number live here and they are **not** interchangeable:

* **Constants** — physics.  The speed of light, the refractive index of silica.
  These are not tunable.
* **Defaults** — plausible values for a metropolitan quantum network built from
  today's components.  Every one of them is a modelling choice.  They are
  collected here, with a source or a justification each, so that a result can
  be traced back to the assumption that produced it, and so that swapping in
  measured values from real hardware is a one-line change rather than a hunt
  through the codebase.

Nothing in this package hard-codes a default: every model takes its parameters
as arguments and merely *falls back* to these.  If you are modelling a specific
testbed, build a :class:`~qnet_autopilot.physics.fiber.FiberParameters` from its
measurements instead of editing this file.

Units are SI unless the name says otherwise.  Where SeQUeNCe uses a different
unit for the same quantity (it works in picoseconds and dB/m, not seconds and
dB/km) the conversion lives in the adapter, not here — see
:mod:`qnet_autopilot.network.topology`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Speed of light in vacuum (m/s).
SPEED_OF_LIGHT_VACUUM = 2.99792458e8

#: Group index of standard single-mode fibre (SMF-28) at 1550 nm, dimensionless.
#: Vendor datasheet value; gives ~4.9 us of latency per km.
FIBER_GROUP_INDEX = 1.4682

#: Speed of light in that fibre (m/s).
SPEED_OF_LIGHT_FIBER = SPEED_OF_LIGHT_VACUUM / FIBER_GROUP_INDEX

#: Planck constant (J s).
PLANCK = 6.62607015e-34

# ---------------------------------------------------------------------------
# fibre defaults
# ---------------------------------------------------------------------------

#: Telecom C-band wavelength quantum signals are usually placed at (nm).
#: The loss minimum of silica.
DEFAULT_WAVELENGTH_NM = 1550.0

#: Attenuation of standard SMF at 1550 nm (dB/km).  Datasheet figures run
#: 0.17-0.22; 0.2 is the number the repeater literature uses, which makes
#: results comparable with published rate-vs-distance curves.
DEFAULT_ATTENUATION_DB_PER_KM = 0.2

#: Attenuation length: the distance over which transmittance falls by 1/e (km).
#: ``L_att = 10 / (alpha * ln 10)``.  ~21.7 km at 0.2 dB/km.
DEFAULT_ATTENUATION_LENGTH_KM = 10.0 / (DEFAULT_ATTENUATION_DB_PER_KM * 2.302585092994046)

# ---------------------------------------------------------------------------
# polarisation drift defaults
# ---------------------------------------------------------------------------
#
# Polarisation drift in installed fibre is dominated by two very different
# processes, and any model that uses a single timescale will misrepresent one
# of them:
#
#   * Thermal drift — slow, large.  Temperature changes the birefringence of
#     the whole span, so the state of polarisation wanders over the entire
#     Poincare sphere on a timescale of minutes to hours.  Reported drift for
#     buried metropolitan fibre is a fraction of a radian per minute; aerial
#     and duct fibre is faster and diurnal.
#   * Mechanical vibration — fast, small.  Traffic, wind, machinery and
#     handling modulate stress-induced birefringence at acoustic frequencies
#     (roughly 10 Hz to a few kHz), producing small-amplitude jitter on top of
#     the thermal wander.
#
# Field measurements vary by orders of magnitude between deployments (buried
# vs aerial, urban vs rural, quiet vs trafficked), so these defaults are a
# *plausible buried-metro* operating point, not a universal figure. Treat them
# as the centre of a sweep, not as ground truth.

#: Thermal drift: correlation time of the birefringence random walk (s).
DEFAULT_THERMAL_CORRELATION_TIME_S = 600.0

#: Thermal drift: steady-state RMS rotation rate on the Poincare sphere (rad/s).
#:
#: The number that matters is not this rate but the timescale it implies. The
#: angular velocity is an OU process, so beyond its correlation time the
#: accumulated angle diffuses as ``sigma * sqrt(2 * tau * t)``. At 1e-3 rad/s
#: with a 600 s correlation time, the state of polarisation wanders roughly a
#: radian in half an hour and covers the sphere in a few hours — the buried
#: metropolitan regime.
#:
#: Scale this, not the correlation time, to change deployment:
#:   * ~1e-4  quiet buried fibre, stable for a working day
#:   * ~1e-3  buried metropolitan (this default)
#:   * ~1e-2  aerial or duct fibre near traffic; sphere-scale drift in minutes
#:   * ~1e-1  aerial fibre in wind — tracking, not prediction, is the only option
DEFAULT_THERMAL_DRIFT_RATE_RAD_PER_S = 1.0e-3

#: Vibration: centre frequency of the mechanical noise band (Hz).
DEFAULT_VIBRATION_FREQUENCY_HZ = 120.0

#: Vibration: RMS angular amplitude on the Poincare sphere (rad).
DEFAULT_VIBRATION_AMPLITUDE_RAD = 2.0e-3

#: Rate of discrete polarisation jumps — a cable moved, a splice disturbed
#: (events per second).  Rare, but they dominate the tail of the QBER
#: distribution and are what a compensator must recover from rather than track.
DEFAULT_JUMP_RATE_PER_S = 1.0 / 3600.0

#: Typical angular size of such a jump on the Poincare sphere (rad).
DEFAULT_JUMP_SCALE_RAD = 0.35

# ---------------------------------------------------------------------------
# component defaults
# ---------------------------------------------------------------------------

#: Single-photon detector efficiency.  Superconducting nanowire detectors reach
#: 0.9+ in the lab; 0.7 is a conservative deployed figure.
DEFAULT_DETECTOR_EFFICIENCY = 0.7

#: Detector dark count rate (Hz).
DEFAULT_DARK_COUNT_RATE_HZ = 10.0

#: Detector timing resolution (ps).
DEFAULT_TIME_RESOLUTION_PS = 50.0

#: Fibre-to-chip and fibre-to-memory coupling efficiency (each interface).
DEFAULT_COUPLING_EFFICIENCY = 0.5

#: Ceiling on the success probability of a linear-optics Bell-state
#: measurement with photon-number-resolving-free detectors: only two of the
#: four Bell states are distinguishable.
LINEAR_OPTICS_BSM_CEILING = 0.5

#: Quantum memory coherence time (s).
DEFAULT_MEMORY_COHERENCE_TIME_S = 1.0

#: Raw fidelity of a freshly heralded memory-memory pair.
DEFAULT_RAW_FIDELITY = 0.93

#: Probability a local two-qubit gate is ideal (SeQUeNCe's ``gate_fid``).
DEFAULT_GATE_FIDELITY = 0.99

#: Probability a local measurement is reported correctly (SeQUeNCe's ``meas_fid``).
DEFAULT_MEAS_FIDELITY = 0.99

#: Attempt rate of the entanglement generation protocol (Hz).
DEFAULT_ATTEMPT_RATE_HZ = 2.0e6

# ---------------------------------------------------------------------------
# QKD defaults
# ---------------------------------------------------------------------------

#: Error-correction inefficiency factor ``f`` in the secret key rate formula.
#: 1.16 is the usual figure for practical Cascade / LDPC reconciliation.
DEFAULT_ERROR_CORRECTION_INEFFICIENCY = 1.16

#: QBER above which BB84 yields no secret key (asymptotic, one-way
#: reconciliation).  The exact threshold depends on the protocol variant;
#: 0.11 is the standard asymptotic BB84 number.
BB84_QBER_THRESHOLD = 0.11

# ---------------------------------------------------------------------------
# decoy-state BB84 defaults
# ---------------------------------------------------------------------------
#
# The entanglement side of this package assumes a heralded pair source.  A
# deployed prepare-and-measure QKD link does not have one: it has an attenuated
# laser, whose pulses carry a Poisson number of photons.  That is the whole
# reason decoy states exist, and the reason the numbers below are separate from
# the component defaults above — they describe a *different transmitter*.

#: Mean photon number of the signal state.  The asymptotic optimum for
#: decoy-state BB84 sits near 0.5 for metropolitan losses (Ma, Qi, Zhao & Lo,
#: Phys. Rev. A 72 012326, 2005); it is a default, and phase 5's agent moves it.
DEFAULT_SIGNAL_INTENSITY = 0.5

#: Mean photon number of the weak decoy state.  Has to be well below the signal
#: to constrain the single-photon yield, and well above the dark count floor to
#: be measurable at all.
DEFAULT_DECOY_INTENSITY = 0.1

#: Share of pulses sent as signal, weak decoy and vacuum.  The vacuum state
#: measures the dark count rate directly, which is what makes the vacuum+weak
#: decoy protocol tight.
DEFAULT_DECOY_PROBABILITIES = (0.7, 0.2, 0.1)

#: Intrinsic error of a detected photon from optical misalignment — the
#: polarisation-independent floor left after compensation.  1-1.5% is the usual
#: figure for a well-aligned deployed system.
DEFAULT_MISALIGNMENT_ERROR = 0.015

#: Error rate of a dark count: it is uncorrelated with the bit Alice sent, so it
#: is wrong half the time.  Physics, not a default.
DARK_COUNT_ERROR_RATE = 0.5

#: Pulse repetition rate of the transmitter (Hz).  1 GHz is a routine figure for
#: a deployed decoy-state system.
DEFAULT_CLOCK_RATE_HZ = 1.0e9

#: Basis-sifting factor. Standard BB84 keeps half the detections; efficient
#: BB84 (biased basis choice) approaches one at the cost of a longer run.
DEFAULT_SIFTING_FACTOR = 0.5

#: Detector gate width (s).  A gated system only accepts clicks inside this
#: window around the expected arrival, which is what keeps the dark count
#: contribution down — and what makes timing drift matter.
#:
#: The width is a trade against the arrival spread, not a free choice: a pulse
#: that has travelled 50 km of standard fibre arrives with an RMS spread near
#: 100 ps (see :func:`qnet_autopilot.qkd.timing.timing_spread`), so a window
#: much narrower than half a nanosecond starts throwing away photons even when
#: it is perfectly centred.  500 ps keeps ~98% of them at that reach while
#: admitting a few times fewer dark counts than a nanosecond window would.
DEFAULT_GATE_WIDTH_S = 500.0e-12

# ---------------------------------------------------------------------------
# timing defaults
# ---------------------------------------------------------------------------

#: Thermal coefficient of propagation delay in standard SMF (s per km per K).
#: Dominated by the thermo-optic coefficient of silica, with a smaller term from
#: thermal expansion; measured values cluster around 37-40 ps/km/K near room
#: temperature.  Over 50 km a single kelvin is therefore ~2 ns of delay — an
#: order of magnitude wider than the detector gate, which is why a gated
#: receiver has to track it.
DEFAULT_DELAY_THERMAL_COEFFICIENT_S_PER_KM_K = 40.0e-12

#: Fractional frequency offset between Alice's and Bob's clocks (dimensionless),
#: *after* the clock has been recovered from the classical synchronisation
#: channel that shares the fibre — which is what a deployed system does, and the
#: assumption everything in :mod:`qnet_autopilot.qkd.timing` is written under.
#: 1e-12 is a recovered clock; a free-running oven-controlled oscillator is
#: ~1e-10 and a plain crystal ~1e-9, and setting this to either turns the timing
#: problem into a trivially linear one where the thermal term no longer matters.
DEFAULT_CLOCK_FRACTIONAL_OFFSET = 1.0e-12

#: Random walk of that offset (per sqrt(second)), the part a constant-rate
#: correction cannot absorb.
DEFAULT_CLOCK_FREQUENCY_DIFFUSION = 3.0e-14


#: One-way traversals of a span per heralding attempt, measured against
#: SeQUeNCe's ``SingleHeraldedA`` protocol running on real channels.
#:
#: The analytic environment originally charged one — the photons fly to the
#: midpoint and the herald comes back. The protocol costs four, because an
#: attempt is a negotiation and not a flight: the two ends agree on a slot,
#: then emit, then the Bell-state-measurement node reports. The factor was
#: measured at 4.20, 4.09, 4.07 and 4.03 for spans of 10, 20, 40 and 80 km, so
#: it is multiplicative and not an additive handshake — see
#: ``scripts/evaluate_sequence.py`` and the report's discrete-event section.
MEASURED_HERALDING_ROUND_TRIPS = 4.1
