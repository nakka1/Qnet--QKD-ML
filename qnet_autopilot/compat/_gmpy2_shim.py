"""Pure-Python stand-in for the subset of ``gmpy2`` that SeQUeNCe uses.

SeQUeNCe reaches for ``gmpy2`` in exactly two places
(``sequence.components.optical_channel`` and ``sequence.components.detector``)
and only for high-precision *timing* arithmetic: converting between picosecond
timestamps and time-bin indices without accumulating float error over long
simulated horizons.

The functions it touches are ``mpfr``, ``mpz``, ``floor``, ``ceil``, ``rint``
and ``get_context().precision``.  All of them are expressible with the standard
library's :mod:`decimal`, which gives arbitrary decimal precision.  The
mapping is:

===================  ==========================================
gmpy2                this module
===================  ==========================================
``mpfr(x)``          :class:`decimal.Decimal` at the active precision
``mpz(x)``           Python ``int`` (already unbounded)
``floor``/``ceil``   ``Decimal`` rounding, returning ``Decimal``
``rint``             round-half-to-even, returning ``Decimal``
``get_context()``    a context whose ``precision`` is in *bits*
===================  ==========================================

Precision is quoted in bits by gmpy2 and in decimal digits by ``decimal``;
``bits * log10(2)`` converts between them, so SeQUeNCe's ``precision = 80``
becomes 25 decimal digits — slightly more headroom than the 24 digits its
comment asks for.

This is a compatibility shim, not a reimplementation of gmpy2: it is slower
than the real library and covers only the names above.  Install the real
``gmpy2`` whenever the environment allows it; :mod:`qnet_autopilot.compat`
prefers it automatically.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal

__all__ = ["mpfr", "mpz", "floor", "ceil", "rint", "get_context", "context"]

_BITS_PER_DIGIT = math.log10(2)


class _Context:
    """Mimics ``gmpy2.get_context()``: precision expressed in bits."""

    def __init__(self) -> None:
        self._precision = 53  # IEEE double, gmpy2's own default

    @property
    def precision(self) -> int:
        return self._precision

    @precision.setter
    def precision(self, bits: int) -> None:
        self._precision = int(bits)
        # +1 digit of guard so round-trips at the requested bit precision are exact
        decimal.getcontext().prec = max(1, int(math.ceil(int(bits) * _BITS_PER_DIGIT)) + 1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<gmpy2-shim context precision={self._precision} bits>"


_CONTEXT = _Context()


def get_context() -> _Context:
    """Return the process-wide precision context (gmpy2 API)."""
    return _CONTEXT


context = get_context  # gmpy2 exposes both spellings


def mpfr(value) -> Decimal:
    """Arbitrary-precision float.

    Accepts ``int``, ``float``, ``str`` and ``Decimal``.  ``float`` inputs are
    converted through their exact binary value (``Decimal(float)``), matching
    what ``gmpy2.mpfr`` does for a binary source.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, str)):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(value)
    # numpy scalars and anything else that knows how to become a float
    return Decimal(float(value))


def mpz(value) -> int:
    """Arbitrary-precision integer.  Python's ``int`` already is one."""
    if isinstance(value, Decimal):
        return int(value.to_integral_value(rounding=decimal.ROUND_FLOOR))
    return int(value)


def floor(value) -> Decimal:
    return mpfr(value).to_integral_value(rounding=decimal.ROUND_FLOOR)


def ceil(value) -> Decimal:
    return mpfr(value).to_integral_value(rounding=decimal.ROUND_CEILING)


def rint(value) -> Decimal:
    """Round to nearest integer, ties to even — gmpy2's default rounding."""
    return mpfr(value).to_integral_value(rounding=decimal.ROUND_HALF_EVEN)
